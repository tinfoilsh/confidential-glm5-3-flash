"""Bounded CUDA checks for the installed confidential-computing transfer patches.

Run with VLLM_DISABLE_UVA=1 and VLLM_TINFOIL_CC_OPTIMIZATIONS=1 before
loading model weights. CUDA_VISIBLE_DEVICES can select one physical GPU.
These checks cover transfer behavior, not full-model or throughput qualification.
"""

from contextlib import contextmanager
import os
import queue
import threading
from types import SimpleNamespace
import unittest
import weakref

import numpy as np
import torch


POOL_DEPTH = 2
POOL_GENERATIONS = 6
BUFFER_ELEMENTS = 8
PREFIX_ELEMENTS = 3
PARTIAL_ROWS = 1
REQUEST_COUNT = 3
SPECULATIVE_DEPTH = 5
TOKENS_PER_REQUEST = SPECULATIVE_DEPTH + 1
READBACK_GENERATIONS = 2
GENERATION_STRIDE = REQUEST_COUNT * TOKENS_PER_REQUEST
TOKEN_VALUE_OFFSET = 1
COUNT_CLIPPING = (0, 2, TOKENS_PER_REQUEST)
THREAD_TIMEOUT_SECONDS = 30
STOP = object()


def setUpModule():
    if not torch.cuda.is_available():
        raise unittest.SkipTest("CUDA is unavailable")
    for name in ("VLLM_DISABLE_UVA", "VLLM_TINFOIL_CC_OPTIMIZATIONS"):
        if os.environ.get(name) != "1":
            raise RuntimeError(
                f"Set {name}=1 before importing the patched vLLM modules"
            )


class CudaCase(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cuda", torch.cuda.current_device())
        self.main_stream = torch.cuda.Stream(device=self.device)
        self.copy_stream = torch.cuda.Stream(device=self.device)
        self.inference = torch.inference_mode()
        self.inference.__enter__()
        self.addCleanup(self.inference.__exit__, None, None, None)
        self.addCleanup(torch.cuda.synchronize, self.device)

    def assert_gpu_values(self, actual, expected):
        self.assertEqual(actual.device, self.device)
        observed = actual.cpu()
        wanted = torch.as_tensor(np.array(expected, copy=True), dtype=actual.dtype)
        torch.testing.assert_close(observed, wanted, rtol=0, atol=0)


class CudaBufferTests(CudaCase):
    def test_non_uva_full_prefix_and_empty_copies(self):
        from vllm.utils.platform_utils import is_uva_available
        from vllm.v1.worker.gpu.buffer_utils import NonUvaBuffer

        self.assertFalse(is_uva_available())
        for dtype in (torch.int32, torch.int64, torch.float32):
            for prefix in (None, 0, PREFIX_ELEMENTS, BUFFER_ELEMENTS):
                with self.subTest(dtype=dtype, prefix=prefix):
                    with torch.cuda.stream(self.main_stream):
                        buffer = NonUvaBuffer(BUFFER_ELEMENTS, dtype=dtype)
                        values = torch.arange(BUFFER_ELEMENTS, dtype=dtype) + 1
                        buffer.cpu.copy_(values)
                        output = buffer.uva(prefix)
                        copied = output.clone()
                        done = torch.cuda.Event()
                        done.record(self.main_stream)
                    done.synchronize()
                    expected = values[:prefix]
                    self.assertEqual(tuple(output.shape), tuple(expected.shape))
                    self.assert_gpu_values(copied, expected.numpy())
                    if prefix is None:
                        self.assertIs(output, buffer._uva)

    def test_round_robin_pool_reuses_slots_only_after_gpu_readers_retire(self):
        from vllm.v1.worker.gpu.buffer_utils import NonUvaBuffer, UvaBufferPool

        reader_stream = torch.cuda.Stream(device=self.device)
        with torch.cuda.stream(self.main_stream):
            pool = UvaBufferPool(
                BUFFER_ELEMENTS, dtype=torch.int32, max_concurrency=POOL_DEPTH
            )
        self.assertIs(pool._buffer_cls, NonUvaBuffer)
        retired, observations, pointers = [], [], []
        for generation in range(POOL_GENERATIONS):
            if generation >= POOL_DEPTH:
                retired[generation - POOL_DEPTH].synchronize()
            values = np.arange(BUFFER_ELEMENTS, dtype=np.int32) + generation
            with torch.cuda.stream(self.main_stream):
                view = pool.copy_to_uva(values)
                ready = torch.cuda.Event()
                ready.record(self.main_stream)
            pointers.append(view.data_ptr())
            with torch.cuda.stream(reader_stream):
                reader_stream.wait_event(ready)
                observation = view.clone()
                done = torch.cuda.Event()
                done.record(reader_stream)
            retired.append(done)
            observations.append((observation, values.copy()))
        retired[-1].synchronize()
        self.assertEqual(len(set(pointers)), POOL_DEPTH)
        self.assertEqual(pointers[POOL_DEPTH:], pointers[:-POOL_DEPTH])
        for observation, expected in observations:
            self.assert_gpu_values(observation, expected)

    def test_metadata_full_cache_partial_publication_and_full_restoration(self):
        from vllm.v1.worker.gpu.buffer_utils import CpuMetadataTensor

        for shape in ((BUFFER_ELEMENTS,), (POOL_DEPTH, BUFFER_ELEMENTS)):
            with self.subTest(shape=shape), torch.cuda.stream(self.main_stream):
                mirror = CpuMetadataTensor(
                    shape, dtype=torch.int32, max_concurrency=POOL_DEPTH
                )
                self.assertTrue(mirror._cache_unchanged)
                mirror.np[:] = np.arange(np.prod(shape), dtype=np.int32).reshape(shape)
                expected = mirror.np.copy()
                full = mirror.copy_to_uva()
                self.main_stream.synchronize()
                self.assert_gpu_values(full, expected)
                current_slot = mirror.pool._curr
                self.assertIs(mirror.copy_to_uva(), full)
                self.assertEqual(mirror.pool._curr, current_slot)

                mirror.np[:PARTIAL_ROWS] += GENERATION_STRIDE
                expected = mirror.np.copy()
                partial = mirror.copy_to_uva(PARTIAL_ROWS)
                self.main_stream.synchronize()
                self.assert_gpu_values(partial, expected[:PARTIAL_ROWS])
                self.assertIsNone(mirror._last_published)
                restored = mirror.copy_to_uva()
                self.main_stream.synchronize()
                self.assertEqual(tuple(restored.shape), shape)
                self.assert_gpu_values(restored, expected)
                np.testing.assert_array_equal(mirror._last_published, expected)
                self.assertIs(mirror.copy_to_uva(), restored)

    def test_failed_metadata_upload_invalidates_cache_and_retries_real_copy(self):
        from vllm.v1.worker.gpu.buffer_utils import CpuMetadataTensor

        with torch.cuda.stream(self.main_stream):
            mirror = CpuMetadataTensor(
                BUFFER_ELEMENTS, dtype=torch.int32, max_concurrency=POOL_DEPTH
            )
            mirror.np[:] = np.arange(BUFFER_ELEMENTS, dtype=np.int32)
            published = mirror.copy_to_uva()
            self.main_stream.synchronize()
            expected_before = mirror.np.copy()
            mirror.np[:] += GENERATION_STRIDE
            expected_after = mirror.np.copy()
            next_slot = (mirror.pool._curr + 1) % POOL_DEPTH
            slot = mirror.pool._uva_bufs[next_slot]
            original_device_buffer = slot._uva
            # A real shape mismatch fails before CUDA dispatch without poisoning it.
            slot._uva = original_device_buffer[:PREFIX_ELEMENTS]
            try:
                with self.assertRaises(RuntimeError):
                    mirror.copy_to_uva()
            finally:
                slot._uva = original_device_buffer
            self.assertIsNone(mirror._last_published)
            self.assertIs(mirror.gpu, published)
            self.assert_gpu_values(published, expected_before)
            retried = mirror.copy_to_uva()
            self.main_stream.synchronize()
            self.assert_gpu_values(retried, expected_after)
            np.testing.assert_array_equal(mirror._last_published, expected_after)
            self.assertIs(mirror.copy_to_uva(), retried)


class CudaInputStagingTests(CudaCase):
    def test_private_staging_survives_scheduler_reuse_and_preserves_output_storage(
        self,
    ):
        from vllm.v1.worker.gpu.cc_input_staging import PIN_MEMORY, private_tensor_h2d

        self.assertTrue(
            PIN_MEMORY, "The measured staging path requires pinned-memory support"
        )
        indices = np.arange(BUFFER_ELEMENTS, dtype=np.intp)[::-1].copy()
        cumulative = np.arange(BUFFER_ELEMENTS, dtype=np.int32) * SPECULATIVE_DEPTH
        query_start = np.array([0, 2, 5, 5, 5, 5, 5, 5], dtype=np.int32)
        expected_indices = indices.copy()
        expected_cumulative = cumulative.copy()
        expected_query = query_start.copy()
        with torch.cuda.stream(self.main_stream):
            destination = torch.full(
                query_start.shape, -1, dtype=torch.int32, device=self.device
            )
            pointer = destination.data_ptr()
            mapped = private_tensor_h2d(indices, device=self.device, dtype=torch.int32)
            logits = private_tensor_h2d(cumulative, device=self.device)
            query = private_tensor_h2d(query_start, out=destination)
            indices[:] = -1
            cumulative[:] = -1
            query_start[:] = -1
            done = torch.cuda.Event()
            done.record(self.main_stream)
        done.synchronize()
        self.assertEqual(mapped.dtype, torch.int32)
        self.assertIs(query, destination)
        self.assertEqual(query.data_ptr(), pointer)
        self.assert_gpu_values(mapped, expected_indices)
        self.assert_gpu_values(logits, expected_cumulative)
        self.assert_gpu_values(query, expected_query)

    def test_noncontiguous_and_offset_inputs_preserve_native_fallback_results(self):
        from vllm.v1.worker.gpu.cc_input_staging import private_tensor_h2d

        strided = np.arange(BUFFER_ELEMENTS * POOL_DEPTH, dtype=np.int32)[::POOL_DEPTH]
        matrix = np.arange(BUFFER_ELEMENTS, dtype=np.int32).reshape(POOL_DEPTH, -1)
        with torch.cuda.stream(self.main_stream):
            copied_strided = private_tensor_h2d(strided, device=self.device)
            copied_matrix = private_tensor_h2d(matrix, device=self.device)
            backing = torch.full(
                (BUFFER_ELEMENTS + POOL_DEPTH,),
                -1,
                dtype=torch.int32,
                device=self.device,
            )
            output = backing[1:-1]
            returned = private_tensor_h2d(strided, out=output)
            done = torch.cuda.Event()
            done.record(self.main_stream)
        done.synchronize()
        self.assertIs(returned, output)
        self.assert_gpu_values(copied_strided, strided)
        self.assert_gpu_values(copied_matrix, matrix)
        self.assert_gpu_values(output, strided)
        expected_backing = np.full(BUFFER_ELEMENTS + POOL_DEPTH, -1, dtype=np.int32)
        expected_backing[1:-1] = strided
        self.assert_gpu_values(backing, expected_backing)


class OutputConsumer:
    def __init__(self, device):
        self.device = device
        self.inputs = queue.Queue()
        self.results = queue.Queue()
        self.ready = threading.Event()
        self.release = threading.Event()
        self.release.set()
        self.startup_error = None
        self.thread = threading.Thread(
            target=self.run, name="CudaReadbackConsumer", daemon=True
        )

    def run(self):
        try:
            torch.cuda.set_device(self.device)
        except BaseException as error:
            self.startup_error = error
            self.ready.set()
            return
        self.ready.set()
        while True:
            output = self.inputs.get()
            if output is STOP:
                return
            if not self.release.wait(THREAD_TIMEOUT_SECONDS):
                self.results.put((False, TimeoutError("Consumer barrier timed out")))
                return
            try:
                result = output.get_output()
            except BaseException as error:
                self.results.put((False, error))
            else:
                self.results.put((True, result))

    def start(self):
        self.thread.start()
        if not self.ready.wait(THREAD_TIMEOUT_SECONDS):
            raise TimeoutError("Consumer initialization timed out")
        if self.startup_error is not None:
            raise self.startup_error
        return self

    def result(self):
        succeeded, result = self.results.get(timeout=THREAD_TIMEOUT_SECONDS)
        if not succeeded:
            raise result
        return result

    def close(self):
        self.release.set()
        self.inputs.put(STOP)
        self.thread.join(THREAD_TIMEOUT_SECONDS)
        if self.thread.is_alive():
            raise TimeoutError("Consumer did not stop")


class CudaReadbackTests(CudaCase):
    def setUp(self):
        super().setUp()
        from vllm.v1.worker.gpu.cc_readback import CCReadbackPolicy

        runner = SimpleNamespace(
            device=self.device,
            main_stream=self.main_stream,
            output_copy_stream=self.copy_stream,
        )
        self.policy = CCReadbackPolicy(runner)
        self.consumer = OutputConsumer(self.device)
        self.addCleanup(self.consumer.close)
        self.consumer.start()

    def arguments(self, generation=0):
        from vllm.v1.outputs import ModelRunnerOutput
        from vllm.v1.worker.gpu.sample.output import SamplerOutput

        request_ids = [f"request-{index}" for index in range(REQUEST_COUNT)]
        counts = torch.tensor(COUNT_CLIPPING, dtype=torch.int32, device=self.device)
        sampled = torch.arange(
            REQUEST_COUNT * TOKENS_PER_REQUEST, dtype=torch.int64, device=self.device
        ).reshape(REQUEST_COUNT, TOKENS_PER_REQUEST)
        sampled.add_(TOKEN_VALUE_OFFSET + generation * GENERATION_STRIDE)
        sampler = SamplerOutput(
            sampled_token_ids=sampled,
            logprobs_tensors=None,
            num_nans=None,
            num_sampled=counts,
            num_rejected=torch.zeros_like(counts),
        )
        return dict(
            model_runner_output=ModelRunnerOutput(
                req_ids=request_ids,
                req_id_to_index={
                    request: index for index, request in enumerate(request_ids)
                },
            ),
            sampler_output=sampler,
            num_sampled_tokens=counts,
            main_stream=self.main_stream,
            copy_stream=self.copy_stream,
            check_ep_fault=False,
            pending_aux_output=None,
        )

    @contextmanager
    def selection(self, is_reply=True):
        token = self.policy.context.set((is_reply, self.consumer.thread))
        try:
            yield
        finally:
            self.policy.reset(token)

    def expected_ids(self, generation):
        start = TOKEN_VALUE_OFFSET + generation * GENERATION_STRIDE
        rows = np.arange(start, start + GENERATION_STRIDE).reshape(
            REQUEST_COUNT, TOKENS_PER_REQUEST
        )
        return [row[:count].tolist() for row, count in zip(rows, COUNT_CLIPPING)]

    def test_deferred_outputs_retain_sources_and_clip_counts_on_consumer_thread(self):
        from vllm.v1.worker.gpu.cc_readback import _DeferredOutput

        self.consumer.release.clear()
        outputs, source_references = [], []
        for generation in range(READBACK_GENERATIONS):
            with torch.cuda.stream(self.main_stream), self.selection():
                arguments = self.arguments(generation)
                output = self.policy.make_output(**arguments)
                self.assertIs(type(output), _DeferredOutput)
                self.assertEqual(output.producer_event.device, self.device)
                self.assertFalse(hasattr(output, "sampled_token_ids"))
                source_references.append(weakref.ref(arguments["sampler_output"]))
                del arguments
            self.consumer.inputs.put(output)
            outputs.append(output)
        self.assertEqual(len(self.policy.pending), READBACK_GENERATIONS)
        self.assertTrue(all(reference() is not None for reference in source_references))
        with self.assertRaisesRegex(ValueError, "different consumer thread"):
            outputs[0].get_output()
        self.consumer.release.set()
        results = [self.consumer.result() for _ in range(READBACK_GENERATIONS)]
        for generation, (result, output) in enumerate(zip(results, outputs)):
            self.assertEqual(result.sampled_token_ids, self.expected_ids(generation))
            self.assertTrue(output.producer_event.query())
            self.assertTrue(output.copy_event.query())
            self.assertTrue(output.attempted)
            self.assertIsNone(output.failed)
        self.assertFalse(self.policy.pending)
        self.assertIsNone(self.policy.context.get())
        self.consumer.inputs.put(outputs[0])
        self.assertIs(self.consumer.result(), results[0])

    def test_optional_outputs_use_native_readback_and_nonreply_skips_ordinary_copy(
        self,
    ):
        from vllm.v1.worker.gpu.async_utils import AsyncOutput

        with torch.cuda.stream(self.main_stream), self.selection(is_reply=False):
            ordinary = self.arguments()
            skipped = self.policy.make_output(**ordinary)
            self.assertIs(skipped, ordinary["model_runner_output"])
            self.assertFalse(self.policy.pending)

        for is_reply in (True, False):
            with self.subTest(is_reply=is_reply):
                with torch.cuda.stream(self.main_stream), self.selection(is_reply):
                    arguments = self.arguments()
                    arguments["sampler_output"].num_nans = torch.zeros(
                        REQUEST_COUNT, dtype=torch.int32, device=self.device
                    )
                    arguments["model_runner_output"].prompt_token_id_logprobs_dict = {
                        "request-0": torch.arange(
                            PREFIX_ELEMENTS, dtype=torch.float32, device=self.device
                        )
                    }
                    output = self.policy.make_output(**arguments)
                    self.assertIs(type(output), AsyncOutput)
                self.consumer.inputs.put(output)
                result = self.consumer.result()
                self.assertEqual(result.sampled_token_ids, self.expected_ids(0))
                self.assertEqual(
                    result.num_nans_in_logits,
                    {f"request-{index}": 0 for index in range(REQUEST_COUNT)},
                )
                torch.testing.assert_close(
                    result.prompt_token_id_logprobs_dict["request-0"],
                    torch.arange(PREFIX_ELEMENTS, dtype=torch.float32),
                    rtol=0,
                    atol=0,
                )
                self.assertFalse(self.policy.pending)


if __name__ == "__main__":
    unittest.main(verbosity=2)
