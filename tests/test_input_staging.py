"""Exercise private staging with real CPU tensors and the native fallback.

CUDA device guards and transfer destinations map to a CPU test boundary;
these checks do not establish CUDA lifetime or performance behavior.
"""

import ast
from contextlib import contextmanager
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch


SOURCE = Path(os.environ["VLLM_PATCHED_SOURCE"])
HELPER = Path("vllm/v1/worker/gpu/cc_input_staging.py")
CUDA_DEVICE = torch.device("cuda:0")
CPU_DEVICE = torch.device("cpu")
MAX_CONTEXT = 1_048_576
MAX_REQUESTS = 32
SPECULATIVE_DEPTH = 5
TENSOR_PARALLEL_SIZES = (4, 8)
UNSUPPORTED_TENSOR_PARALLEL_SIZES = (2, 16)


def load(relative, namespace, names=None):
    path = SOURCE / relative
    selected = [node for node in ast.parse(path.read_text()).body
                if isinstance(node, (ast.FunctionDef, ast.Assign))
                and (names is None or isinstance(node, ast.FunctionDef)
                     and node.name in names)]
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


class StagingTests(unittest.TestCase):
    def setUp(self):
        self.fallback_calls, self.transfers = [], []
        native = load(Path("vllm/utils/torch_utils.py"),
                      {"np": np, "torch": torch, "PIN_MEMORY": False},
                      {"async_tensor_h2d"})["async_tensor_h2d"]

        def fallback(data, device=None, dtype=None, out=None):
            self.fallback_calls.append((data, device, dtype, out))
            return native(data, device=device, dtype=dtype, out=out)

        self.namespace = load(HELPER, {"np": np, "torch": torch, "PIN_MEMORY": True,
                                       "async_tensor_h2d": fallback})
        self.copy = self.namespace["private_tensor_h2d"]
        self.supports = self.namespace["supports_text_staging"]

    @contextmanager
    def cpu_boundary(self, output=None, capturing=False):
        original_to = torch.Tensor.to
        original_device = torch.Tensor.device

        def cpu_to(tensor, *args, **kwargs):
            if kwargs.get("device") == CUDA_DEVICE:
                self.transfers.append(tensor)
                kwargs["device"] = CPU_DEVICE
            return original_to(tensor, *args, **kwargs)

        def device(tensor):
            if tensor is output:
                return CUDA_DEVICE
            return original_device.__get__(tensor, torch.Tensor)

        with (
            patch.object(torch.Tensor, "to", cpu_to),
            patch.object(torch.Tensor, "device", property(device)),
            patch.object(torch.cuda, "is_current_stream_capturing", return_value=capturing),
        ):
            yield

    def test_index_mapping_owns_storage_and_converts_pointer_sized_integers(self):
        indices = np.array([3, 0, 2, 1], dtype=np.intp)
        expected = torch.tensor(indices, dtype=torch.int32)
        with self.cpu_boundary():
            output = self.copy(indices, device=CUDA_DEVICE, dtype=torch.int32)
        self.assertEqual(output.dtype, torch.int32)
        torch.testing.assert_close(output, expected)
        self.assertNotEqual(output.data_ptr(), indices.ctypes.data)
        indices[:] = 0
        torch.testing.assert_close(output, expected)
        self.assertEqual(len(self.transfers), 1)
        self.assertFalse(self.transfers[0].is_pinned())
        self.assertFalse(self.fallback_calls)

    def test_cumulative_logits_keep_native_dtype_without_aliasing_scheduler(self):
        cumulative = np.array([0, 6, 12, 18], dtype=np.int32)
        expected = torch.tensor(cumulative)
        with self.cpu_boundary():
            output = self.copy(cumulative, device=CUDA_DEVICE)
        torch.testing.assert_close(output, expected)
        self.assertEqual(output.dtype, torch.int32)
        cumulative[:] = 0
        torch.testing.assert_close(output, expected)
        self.assertFalse(self.fallback_calls)

    def test_persistent_output_is_fully_overwritten_and_preserves_identity(self):
        cumulative = np.array([0, 3, 9, 9, 9, 9], dtype=np.int32)
        expected = torch.tensor(cumulative)
        destination = torch.full(expected.shape, -1, dtype=torch.int32)
        pointer = destination.data_ptr()
        with self.cpu_boundary(output=destination):
            output = self.copy(cumulative, out=destination)
        self.assertIs(output, destination)
        self.assertEqual(output.data_ptr(), pointer)
        torch.testing.assert_close(output, expected)
        cumulative[:] = 0
        torch.testing.assert_close(output, expected)
        self.assertFalse(self.fallback_calls)

    def test_unsupported_input_layouts_execute_native_fallback(self):
        cases = (
            np.arange(8, dtype=np.int32)[::2],
            np.arange(6, dtype=np.int32).reshape(2, 3),
            np.arange(4, dtype=np.float32),
            np.array([0], dtype=np.int32),
            torch.arange(4, dtype=torch.int32),
            [0, 2, 4],
        )
        for data in cases:
            with self.subTest(data=repr(data)):
                before = len(self.fallback_calls)
                with self.cpu_boundary():
                    output = self.copy(data, device=CUDA_DEVICE)
                self.assertEqual(len(self.fallback_calls), before + 1)
                self.assertIs(self.fallback_calls[-1][0], data)
                torch.testing.assert_close(output, torch.as_tensor(data))

    def test_offset_output_capture_and_unavailable_pinning_preserve_native_behavior(self):
        data = np.array([0, 4, 8, 8], dtype=np.int32)
        backing = torch.full((data.size + 2,), -1, dtype=torch.int32)
        output = backing[1:-1]
        with self.cpu_boundary(output=output):
            self.assertIs(self.copy(data, out=output), output)
        self.assertEqual(len(self.fallback_calls), 1)
        torch.testing.assert_close(output, torch.tensor(data))
        self.assertEqual((backing[0].item(), backing[-1].item()), (-1, -1))
        for capturing, pin_memory in ((True, True), (False, False)):
            self.namespace["PIN_MEMORY"] = pin_memory
            before = len(self.fallback_calls)
            with self.cpu_boundary(capturing=capturing):
                result = self.copy(data, device=CUDA_DEVICE)
            self.assertEqual(len(self.fallback_calls), before + 1)
            torch.testing.assert_close(result, torch.tensor(data))

    def test_text_guard_accepts_supported_topologies_without_encoder_attribute(self):
        modules, classes = {}, {}
        definitions = (
            ("ec_connector", ("ECConnector",)),
            ("model_states.mamba_hybrid", ("MambaHybridModelState",)),
            ("sample.sampler", ("Sampler", "LogitBiasState", "PenaltiesState", "BadWordsState")),
            ("spec_decode.mtp.speculator", ("MTPSpeculator",)),
        )
        for suffix, names in definitions:
            module = ModuleType("vllm.v1.worker.gpu." + suffix)
            for name in names:
                classes[name] = type(name, (), {})
                setattr(module, name, classes[name])
            modules[module.__name__] = module
        connector = ModuleType("vllm.v1.worker.gpu.kv_connector")
        connector.NO_OP_KV_CONNECTOR = object()
        modules[connector.__name__] = connector
        state = classes["MambaHybridModelState"]()
        state.prompt_embeds_state, state.supports_mm_inputs = None, False
        sampler = classes["Sampler"]()
        sampler.logits_processors = [classes[name]() for name in (
            "LogitBiasState", "PenaltiesState", "BadWordsState")]
        input_buffers = object()
        speculator = classes["MTPSpeculator"]()
        speculator.target_input_buffers = input_buffers
        speculator.input_buffers = object()
        speculator.supports_mm_inputs = False
        parallel = SimpleNamespace(tensor_parallel_size=TENSOR_PARALLEL_SIZES[0],
            data_parallel_size=1, pipeline_parallel_size=1,
            prefill_context_parallel_size=1, decode_context_parallel_size=1,
            enable_dbo=False, use_ubatching=False, enable_batch_sharded_sampling=False,
            enable_expert_parallel=False)
        spec = SimpleNamespace(method="mtp", num_speculative_tokens=SPECULATIVE_DEPTH,
            num_speculative_tokens_per_batch_size=None, enable_adaptive_verification=False,
            rejection_sample_method="standard")
        runner = SimpleNamespace(parallel_config=parallel, speculative_config=spec,
            model_state=state, lora_config=None, is_pooling_model=False,
            is_encoder_decoder=False, is_first_pp_rank=True, is_last_pp_rank=True,
            use_pp=False, use_dcp=False, dp_size=1, dcp_size=1,
            scheduler_config=SimpleNamespace(async_scheduling=True),
            max_model_len=MAX_CONTEXT, max_num_reqs=MAX_REQUESTS,
            model_config=SimpleNamespace(served_model_name="glm-5-3-flash",
                logits_processors=None, enable_prompt_embeds=False),
            num_speculative_steps=SPECULATIVE_DEPTH, sampler=sampler, speculator=speculator,
            kv_connector=connector.NO_OP_KV_CONNECTOR,
            ec_connector=classes["ECConnector"](), input_buffers=input_buffers,
            uses_inputs_embeds=False, supports_mm_inputs=False, encoder_cache=None,
            device=CUDA_DEVICE, main_stream=object(), adaptive_verification=None,
            pcp_manager=None, ubatch_runner=None, batch_sharder=None, fast_prefill=None,
            aux_output_connector=None, pooling_runner=None, pp_handler=None)
        batch = SimpleNamespace(num_tokens=8, req_ids=["request"])
        request = SimpleNamespace(mm_features=[], prompt_embeds=None, prompt_is_token_ids=None)
        scheduler = SimpleNamespace(num_scheduled_tokens={"request": 8},
            scheduled_encoder_inputs={}, scheduled_new_reqs=[request])
        with (
            patch.dict(sys.modules, modules), self.cpu_boundary(),
            patch.object(torch.cuda, "current_device", return_value=CUDA_DEVICE.index),
            patch.object(torch.cuda, "current_stream", return_value=runner.main_stream),
        ):
            for tensor_parallel_size in TENSOR_PARALLEL_SIZES:
                with self.subTest(tensor_parallel_size=tensor_parallel_size):
                    parallel.tensor_parallel_size = tensor_parallel_size
                    self.assertTrue(self.supports(runner, scheduler, batch, 0))
                    request.mm_features = [object()]
                    self.assertFalse(self.supports(runner, scheduler, batch, 0))
                    request.mm_features = []
                    batch.req_ids.append("request")
                    self.assertFalse(self.supports(runner, scheduler, batch, 0))
                    batch.req_ids.pop()
                    spec.num_speculative_tokens_per_batch_size = [(1, 8, 3)]
                    self.assertFalse(self.supports(runner, scheduler, batch, 0))
                    spec.num_speculative_tokens_per_batch_size = None
            for tensor_parallel_size in UNSUPPORTED_TENSOR_PARALLEL_SIZES:
                parallel.tensor_parallel_size = tensor_parallel_size
                self.assertFalse(self.supports(runner, scheduler, batch, 0))


if __name__ == "__main__":
    unittest.main()
