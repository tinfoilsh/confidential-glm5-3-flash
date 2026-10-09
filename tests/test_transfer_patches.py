"""CPU checks of patched vLLM methods; these do not qualify CUDA execution."""

import ast
import contextlib
from contextvars import ContextVar
from enum import Enum, auto
import gc
import os
from pathlib import Path
import queue
import sys
import threading
from types import MethodType, ModuleType, SimpleNamespace
import traceback
import unittest
from unittest.mock import patch
import weakref

import numpy as np


SOURCE = Path(os.environ["VLLM_PATCHED_SOURCE"]) / "vllm"
SAMPLING_FIELDS = ("temperature", "top_p", "top_k", "min_p", "seeds")
POOL_DEPTH = 2
REQUEST_SLOTS = 4
VOCABULARY_SIZE = 100
WAIT_SECONDS = 5
STOP = object()


def execute_nodes(relative, names, namespace, class_name=None):
    path = SOURCE / relative
    tree = ast.parse(path.read_text())
    body = tree.body
    if class_name:
        body = next(node.body for node in body if isinstance(node, ast.ClassDef)
                    and node.name == class_name)
    selected = [node for node in body
                if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names
                or isinstance(node, ast.Assign) and any(
                    isinstance(target, ast.Name) and target.id in names
                    for target in node.targets)]
    unit = ast.Module(body=[ast.ImportFrom(module="__future__", names=[
        ast.alias(name="annotations")], level=0), *selected], type_ignores=[])
    exec(compile(ast.fix_missing_locations(unit), str(path), "exec"), namespace)
    return namespace


class ArrayMirror:
    def __init__(self, size, dtype):
        self.np = np.zeros(size, dtype=dtype)
        self.slots = [np.zeros_like(self.np) for _ in range(POOL_DEPTH)]
        self.slot, self.uploads, self.fail = 0, 0, False
        self.gpu = self.copy_to_uva()

    def copy_to_uva(self):
        if self.fail:
            raise RuntimeError("upload failure")
        self.slot = (self.slot + 1) % POOL_DEPTH
        self.slots[self.slot][:] = self.np
        self.gpu = self.slots[self.slot]
        self.uploads += 1
        return self.gpu


class Array:
    def __init__(self, value):
        self.value, self.non_blocking = value, False

    def __getitem__(self, index):
        return Array(self.value[index])

    def copy_(self, other, non_blocking=False):
        np.copyto(self.value, other.value)
        self.non_blocking = non_blocking
        return self


class TransferTests(unittest.TestCase):
    def states(self, enabled=True):
        namespace = dict(np=np, UvaBackedTensor=ArrayMirror,
                         torch=SimpleNamespace(float32=np.float32, int32=np.int32,
                                               int64=np.int64),
                         envs=SimpleNamespace(VLLM_TINFOIL_CC_OPTIMIZATIONS=enabled))
        execute_nodes("v1/worker/gpu/sample/states.py",
                      {"SamplingStates", "NO_LOGPROBS", "_NP_INT64_MIN", "_NP_INT64_MAX"},
                      namespace)
        return namespace["SamplingStates"](REQUEST_SLOTS, VOCABULARY_SIZE)

    def test_sampling_refresh_preserves_values_and_live_views_on_clean_steps(self):
        state = self.states()
        settings = SimpleNamespace(temperature=0.7, top_k=-1, top_p=0.8,
                                   min_p=0.1, seed=37, logprobs=-1)
        self.assertTrue(state.add_request(0, settings))
        state.apply_staged_writes()
        views = {field: getattr(state, field).gpu for field in SAMPLING_FIELDS}
        uploads = {field: getattr(state, field).uploads for field in SAMPLING_FIELDS}
        state.apply_staged_writes()
        for field in SAMPLING_FIELDS:
            mirror = getattr(state, field)
            np.testing.assert_array_equal(mirror.gpu, mirror.np)
            self.assertIs(mirror.gpu, views[field])
            self.assertEqual(mirror.uploads, uploads[field])
        self.assertEqual(state.top_k.gpu[0], VOCABULARY_SIZE)
        self.assertEqual(state.num_logprobs[0], VOCABULARY_SIZE)
        settings.seed = 91
        state.add_request(0, settings)
        state.apply_staged_writes()
        self.assertEqual(state.seeds.gpu[0], 91)
        self.assertEqual(views["seeds"][0], 37)

    def test_failed_refresh_remains_dirty_and_disabled_policy_keeps_refreshing(self):
        state = self.states()
        state.top_k.fail = True
        with self.assertRaisesRegex(RuntimeError, "upload failure"):
            state.apply_staged_writes()
        self.assertTrue(state._parameters_dirty)
        state.top_k.fail = False
        state.apply_staged_writes()
        self.assertFalse(state._parameters_dirty)
        state = self.states(enabled=False)
        state.apply_staged_writes()
        uploads = state.seeds.uploads
        state.apply_staged_writes()
        self.assertEqual(state.seeds.uploads, uploads + 1)

    def test_nonblocking_copy_preserves_full_prefix_and_empty_values(self):
        for enabled in (False, True):
            namespace = {"envs": SimpleNamespace(VLLM_TINFOIL_CC_OPTIMIZATIONS=enabled)}
            execute_nodes("v1/worker/gpu/buffer_utils.py", {"uva"}, namespace,
                          class_name="NonUvaBuffer")
            source = np.arange(12, dtype=np.int64).reshape(4, 3)
            for prefix in (None, 0, 2, 4):
                before = SimpleNamespace(cpu=Array(source),
                                         _uva=Array(np.zeros_like(source)))
                result = namespace["uva"](before, prefix)
                expected = np.zeros_like(source)
                expected[:prefix] = source[:prefix]
                np.testing.assert_array_equal(before._uva.value, expected)
                self.assertEqual(result.non_blocking, enabled)


class Tensor:
    def __init__(self, values, dtype, fixture):
        self.values = np.array(values)
        self.dtype, self.fixture, self.device = dtype, fixture, fixture.device
        self.ndim, self.shape = self.values.ndim, self.values.shape

    def to(self, target, non_blocking):
        if target != "cpu" or non_blocking is not True:
            raise AssertionError("Unexpected copy policy")
        f = self.fixture
        f.operations.append(("copy", threading.current_thread().name,
                             getattr(f.local, "inference", False)))
        if f.copy_callback:
            f.copy_callback(self)
        return SimpleNamespace(numpy=lambda: self.values.copy())


class Output:
    def __init__(self, count):
        self.req_ids = list(range(count))
        self.sampled_token_ids = None
        self.prompt_logprobs_dict, self.prompt_token_id_logprobs_dict = {}, {}


class SamplerOutput:
    def __init__(self, sampled, counts):
        self.sampled_token_ids, self.num_sampled = sampled, counts
        self.logprobs_tensors = self.num_nans = self.sampling_mask_tensors = None


class ReadbackFixture:
    def __init__(self):
        self.operations, self.created, self.copy_callback = [], [], None
        self.local = threading.local()
        self.device = SimpleNamespace(type="cuda", index=1)
        self.gate = threading.Event()
        self.gate.set()
        f = self

        class Stream:
            def __init__(self, name):
                self.name, self.generation = name, 0

            def wait_event(self, event):
                f.operations.append(("wait_event", event.generation))

            def wait_stream(self, stream):
                f.operations.append(("wait_stream", stream.generation))

        class Event:
            def __init__(self, blocking=False):
                self.generation = None

            def record(self, stream):
                self.generation = stream.generation

            def synchronize(self):
                f.operations.append(("synchronize", self.generation))

        @contextlib.contextmanager
        def inference_mode():
            previous = getattr(f.local, "inference", False)
            f.local.inference = True
            try:
                yield
            finally:
                f.local.inference = previous

        self.main, self.copy = Stream("main"), Stream("copy")
        torch = SimpleNamespace(Tensor=Tensor, int64="int64", int32="int32",
            inference_mode=inference_mode, cuda=SimpleNamespace(Event=Event,
                stream=lambda value: contextlib.nullcontext(), set_stream=lambda value: None))
        namespace = dict(torch=torch, contextlib=contextlib,
            AsyncModelRunnerOutput=type("AsyncModelRunnerOutput", (), {}),
            envs=SimpleNamespace(VLLM_RAISE_ON_LOGIT_NANS=False))
        execute_nodes("v1/worker/gpu/async_utils.py",
                      {"AsyncOutput", "async_copy_to_np", "stream"}, namespace)
        self.original = namespace["AsyncOutput"]
        namespace.update(ContextVar=ContextVar, RLock=threading.RLock,
                         current_thread=threading.current_thread,
                         ModelRunnerOutput=Output, SamplerOutput=SamplerOutput)
        helper_tree = ast.parse((SOURCE / "v1/worker/gpu/cc_readback.py").read_text())
        names = {node.name for node in helper_tree.body
                 if isinstance(node, (ast.ClassDef, ast.FunctionDef))}
        names.update(target.id for node in helper_tree.body if isinstance(node, ast.Assign)
                     for target in node.targets if isinstance(target, ast.Name))
        execute_nodes("v1/worker/gpu/cc_readback.py", names, namespace)
        self.guard = namespace["_supported_config"]
        self.supported = True
        namespace["_supported_config"] = lambda worker: self.supported
        self.namespace = namespace
        self.runner = SimpleNamespace(main_stream=self.main, output_copy_stream=self.copy,
                                      device=self.device)
        self.policy = namespace["CCReadbackPolicy"](self.runner)
        self.runner.cc_readback = self.policy
        worker = SimpleNamespace(rank=1, device=self.device, model_runner=self.runner)

        class Status(Enum):
            SUCCESS = auto()
            FAILURE = auto()

        self.status = Status
        native = dict(traceback=traceback,
            logger=SimpleNamespace(exception=lambda message: None),
            AsyncModelRunnerOutput=namespace["AsyncModelRunnerOutput"],
            WorkerProc=SimpleNamespace(ResponseStatus=Status))
        execute_nodes("v1/executor/multiproc_executor.py",
                      {"_execute_worker_rpc", "enqueue_output", "handle_output"},
                      native, class_name="WorkerProc")
        self.responses = queue.Queue()
        self.process = SimpleNamespace(rank=1, use_async_scheduling=True,
            async_output_queue=queue.Queue(), worker=SimpleNamespace(worker=worker),
            worker_response_mq=SimpleNamespace(enqueue=self.responses.put))
        for name in ("_execute_worker_rpc", "enqueue_output", "handle_output"):
            setattr(self.process, name, MethodType(native[name], self.process))

        def consumer():
            while True:
                value = f.process.async_output_queue.get()
                if value is STOP:
                    return
                if not f.gate.wait(WAIT_SECONDS):
                    raise RuntimeError("Consumer barrier timed out")
                f.process.enqueue_output(value)

        self.process.async_output_copy_thread = threading.Thread(
            target=consumer, name="WorkerAsyncOutputCopy", daemon=True)
        self.process.async_output_copy_thread.start()

    def arguments(self, values=None, counts=None):
        values = [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]] if values is None else values
        counts = [2, 0] if counts is None else counts
        sampled = Tensor(values, "int64", self)
        count_tensor = Tensor(counts, "int32", self)
        return dict(model_runner_output=Output(len(counts)),
            sampler_output=SamplerOutput(sampled, count_tensor),
            num_sampled_tokens=count_tensor, main_stream=self.main, copy_stream=self.copy,
            check_ep_fault=False, pending_aux_output=None)

    def rpc(self, arguments, output_rank=1, method="sample_tokens"):
        def sample():
            output = self.policy.make_output(**arguments)
            self.created.append(output)
            self.main.generation += 1
            return output
        setattr(self.process.worker, method, sample)
        self.process._execute_worker_rpc((method, (), {}, output_rank))

    def result(self):
        return self.responses.get(timeout=WAIT_SECONDS)

    def close(self):
        self.gate.set()
        self.process.async_output_queue.put(STOP)
        self.process.async_output_copy_thread.join(WAIT_SECONDS)
        if self.process.async_output_copy_thread.is_alive():
            raise RuntimeError("Consumer did not exit")


class ReadbackTests(unittest.TestCase):
    def fixture(self):
        fixture = ReadbackFixture()
        self.addCleanup(fixture.close)
        return fixture

    def test_deferred_generations_keep_producer_boundary_and_native_conversion(self):
        f = self.fixture()
        f.gate.clear()
        f.rpc(f.arguments())
        f.rpc(f.arguments([[21, 22, 23, 24, 25, 26]], [4]))
        self.assertFalse(any(row[0] == "copy" for row in f.operations))
        self.assertEqual(len(f.policy.pending), 2)
        with self.assertRaisesRegex(ValueError, "different consumer thread"):
            f.created[0].get_output()
        f.gate.set()
        self.assertEqual(f.result()[1].sampled_token_ids, [[1, 2], []])
        self.assertEqual(f.result()[1].sampled_token_ids, [[21, 22, 23, 24]])
        self.assertEqual([row for row in f.operations if row[0] == "wait_event"],
                         [("wait_event", 0), ("wait_event", 1)])
        self.assertTrue(all(row[1:] == ("WorkerAsyncOutputCopy", True)
                            for row in f.operations if row[0] == "copy"))
        self.assertFalse(f.policy.pending)
        self.assertIsNone(f.policy.context.get())

    def test_nonreply_skips_copies_and_unsupported_output_uses_native_fallback(self):
        f = self.fixture()
        f.rpc(f.arguments(), output_rank=0)
        self.assertFalse(any(row[0] == "copy" for row in f.operations))
        self.assertTrue(f.responses.empty())
        for mode in ("all_ranks", "different_method", "unsupported", "one_token", "nan_counts"):
            with self.subTest(mode=mode):
                f.supported = mode != "unsupported"
                arguments = f.arguments([[44]], [1]) if mode == "one_token" else f.arguments()
                if mode == "nan_counts":
                    arguments["sampler_output"].num_nans = Tensor([0, 0], "int64", f)
                f.rpc(arguments, output_rank=None if mode == "all_ranks" else 1,
                      method="take_draft_token_ids" if mode == "different_method" else "sample_tokens")
                self.assertIs(type(f.created[-1]), f.original)
                self.assertEqual(f.result()[0], f.status.SUCCESS)
        self.assertFalse(f.policy.pending)


    def test_configuration_guard_rejects_unqualified_topology_and_sampling_modes(self):
        f = self.fixture()
        modules, classes = {}, {}
        for suffix, name in (
            ("model_runner", "GPUModelRunner"),
            ("model_states.mamba_hybrid", "MambaHybridModelState"),
            ("sample.sampler", "Sampler"),
            ("spec_decode.mtp.speculator", "MTPSpeculator"),
            ("spec_decode.multi_module_mtp.speculator", "MultiModuleMTPSpeculator"),
            ("spec_decode.rejection_sampler", "RejectionSampler"),
        ):
            module = ModuleType("vllm.v1.worker.gpu." + suffix)
            classes[name] = type(name, (), {})
            setattr(module, name, classes[name])
            modules[module.__name__] = module
        runner = classes["GPUModelRunner"]()
        runner.device = f.device
        runner.max_model_len, runner.max_num_reqs = 1_048_576, 32
        runner.scheduler_config = SimpleNamespace(async_scheduling=True)
        runner.model_config = SimpleNamespace(served_model_name="glm-5-3-flash")
        runner.speculative_config = SimpleNamespace(method="mtp", num_speculative_tokens=5,
            num_speculative_tokens_per_batch_size=None, enable_adaptive_verification=False,
            rejection_sample_method="standard")
        runner.num_speculative_steps = 5
        runner.sampler = classes["Sampler"]()
        runner.rejection_sampler = classes["RejectionSampler"]()
        runner.rejection_sampler.sampler = runner.sampler
        runner.speculator = classes["MTPSpeculator"]()
        runner.speculator.acceptance_estimator = None
        runner.speculator.supports_mm_inputs = False
        runner.model_state = classes["MambaHybridModelState"]()
        runner.model_state.recoverssm = None
        parallel = SimpleNamespace(tensor_parallel_size=8, data_parallel_size=1,
            pipeline_parallel_size=1, prefill_context_parallel_size=1,
            decode_context_parallel_size=1, enable_expert_parallel=False)
        worker = SimpleNamespace(model_runner=runner, parallel_config=parallel,
                                 rank=1, device=f.device)
        with patch.dict(sys.modules, modules):
            self.assertTrue(f.guard(worker))
            for target, field, unsupported in (
                (parallel, "tensor_parallel_size", 4),
                (parallel, "pipeline_parallel_size", 2),
                (parallel, "enable_expert_parallel", True),
                (runner, "max_model_len", 32768),
                (runner, "sampler", object()),
                (runner.speculator, "acceptance_estimator", object()),
                (runner.speculator, "supports_mm_inputs", True),
                (runner.model_state, "recoverssm", object()),
                (runner.scheduler_config, "async_scheduling", False),
                (runner.speculative_config, "num_speculative_tokens_per_batch_size",
                 [(1, 8, 3)]),
                (runner.speculative_config, "rejection_sample_method", "synthetic"),
            ):
                with self.subTest(field=field):
                    saved = getattr(target, field)
                    setattr(target, field, unsupported)
                    self.assertFalse(f.guard(worker))
                    setattr(target, field, saved)

    def test_copy_failure_retains_sources_and_forces_subsequent_native_fallback(self):
        f = self.fixture()
        copies = []
        def fail_second(tensor):
            copies.append(tensor)
            if len(copies) == 2:
                raise RuntimeError("second copy failure")
        f.copy_callback = fail_second
        f.rpc(f.arguments())
        self.assertEqual(f.result(), (f.status.FAILURE, "second copy failure"))
        output = weakref.ref(f.created[-1])
        sources = weakref.ref(f.created[-1].sampler_output)
        f.created.clear()
        copies.clear()
        gc.collect()
        self.assertIsNotNone(output())
        self.assertIsNotNone(sources())
        f.copy_callback = None
        f.rpc(f.arguments())
        self.assertIs(type(f.created[-1]), f.original)
        self.assertEqual(f.result()[0], f.status.SUCCESS)
        self.assertEqual(len(f.policy.pending), 1)

    def test_dispatch_failure_restores_context_and_preserves_unconsumed_sources(self):
        f = self.fixture()
        def failing_sample():
            f.policy.make_output(**f.arguments())
            raise RuntimeError("postprocess failure")
        f.process.worker.sample_tokens = failing_sample
        f.process._execute_worker_rpc(("sample_tokens", (), {}, 1))
        self.assertEqual(f.result(), (f.status.FAILURE, "postprocess failure"))
        self.assertIsNone(f.policy.context.get())
        self.assertEqual(len(f.policy.pending), 1)
        outside = f.policy.make_output(**f.arguments())
        self.assertIs(type(outside), f.original)


if __name__ == "__main__":
    unittest.main()
