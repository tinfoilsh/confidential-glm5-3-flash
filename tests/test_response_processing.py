"""CPU response regressions; no CUDA, CC, throughput, or server-latency qualification."""

import asyncio
from copy import deepcopy
import json
import unittest

from starlette.responses import JSONResponse

import response_fixture as f


HEADERS = {"x-response-fixture": "native"}
IDLE_BACKLOG = (0, 0, 0, 0)
LARGE_ROWS = f.serializer.LARGE_PROMPT_LOGPROBS_ROWS + 1
CLAMP_FLOOR = f.HELPERS["PROMPT_LOGPROB_FLOOR"]
SURROGATE = "\ud800"


def rows():
    return [None, {}, {1: f.Logprob(-float("inf"), 1, "é"),
                      2: f.Logprob(-0.0, None, "雪"), 3: f.Logprob(-1.25, 2, "😀")}]


def large_rows():
    return [None] + [{1: f.Logprob(-float("inf"), 1, "é雪😀")} for _ in range(LARGE_ROWS)]


async def completion_outcome(outputs, request, *, cooperative):
    try:
        response = (await f.Completion(outputs)._create_completion(request) if cooperative
                    else f.synchronous_completion(outputs, request))
        encoded = (await f.serializer.completion_json_response(response, headers=HEADERS)
                   if cooperative else JSONResponse(response.model_dump(), headers=HEADERS))
        return "body", encoded.body, dict(encoded.headers)
    except Exception as error:
        detail = error.errors(include_url=False) if hasattr(error, "errors") else str(error)
        return type(error).__name__, detail


class HandlerArrivalTests(unittest.IsolatedAsyncioTestCase):
    async def test_ordinary_output_arriving_through_handler_precedes_score_output(self):
        handler = f.Handler()
        self.addAsyncCleanup(handler.stop)
        large, small = handler.add("large"), handler.add("small")
        async with asyncio.timeout(f.WAIT_SECONDS):
            await handler.submit([f.engine_output(handler.tokenizer, "large"),
                                  f.engine_output(handler.tokenizer, "small", heavy=False)])
            ordinary = await small.get()
            self.assertTrue(ordinary.finished)
            self.assertEqual(ordinary.outputs[0].token_ids,
                             [handler.tokenizer.convert_tokens_to_ids("hello")])
            self.assertTrue(large.output is None,
                            "The native handler finished score-heavy output before ordinary output")
            self.assertTrue((await large.get()).finished)
            await handler.owner.output_dispatcher.wait_idle()
        self.assertEqual(handler.backlog[-1], IDLE_BACKLOG)
        self.assertFalse(handler.processor.request_states)


class DispatchTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.handler = f.Handler()
        self.addAsyncCleanup(self.handler.stop)
        await asyncio.sleep(0)
        self.dispatcher = self.handler.owner.output_dispatcher

    def assert_retired(self):
        self.assertFalse(self.dispatcher.work)
        self.assertEqual(self.handler.backlog[-1], IDLE_BACKLOG)
        self.assertTrue(all(all(value >= 0 for value in reading) for reading in self.handler.backlog))

    async def idle(self):
        await asyncio.wait_for(self.dispatcher.wait_idle(), f.WAIT_SECONDS)
        self.assert_retired()

    async def test_ordinary_request_progresses_while_scores_preserve_request_order(self):
        h = self.handler
        large, small = h.add("large"), h.add("small")
        output = f.engine_output(h.tokenizer, "large", finish=None)
        tensors = output.new_prompt_logprobs_tensors
        await self.dispatcher.submit(output, 0, f.ENGINE_TIMESTAMP, f.ENGINE_TIMESTAMP)
        await asyncio.sleep(0)
        self.assertIsNone(large.output)
        await self.dispatcher.submit(f.engine_output(h.tokenizer, "small", heavy=False),
                                     1, f.SECOND_TIMESTAMP, f.SECOND_TIMESTAMP)
        self.assertTrue(small.get_nowait().finished)
        self.assertIsNone(large.output)
        await self.dispatcher.submit(f.engine_output(h.tokenizer, "large", heavy=False),
                                     0, f.SECOND_TIMESTAMP, f.SECOND_TIMESTAMP)
        await self.idle()
        result = large.get_nowait()
        self.assertEqual(result.outputs[0].token_ids,
                         [h.tokenizer.convert_tokens_to_ids("hello")] * 2)
        self.assertEqual(result.outputs[0].cumulative_logprob, -0.5)
        self.assertEqual(result.outputs[0].finish_reason, "length")
        self.assertEqual(len(result.prompt_logprobs), len(tensors.logprobs) + 1)
        decoded = []
        for token_ids, row in zip(tensors.logprob_token_ids.tolist(), result.prompt_logprobs[1:]):
            selected, hello, special, repeated = token_ids
            self.assertEqual(list(row), [selected, hello, special])
            self.assertEqual((row[selected].logprob, row[selected].rank), (-0.5, 3))
            self.assertEqual(row[hello].decoded_token, "hello")
            self.assertEqual(row[special].logprob, -float("inf"))
            decoded.append(row[repeated].decoded_token)
        repetitions = len(tensors.logprobs) // len(f.UTF8_TEXT.encode("utf-8"))
        self.assertEqual("".join(decoded), f.UTF8_TEXT * repetitions)
        records = [entry for entry in h.records if entry["iteration_stats"] is not None]
        self.assertEqual([entry["engine_idx"] for entry in records], [1, 0, 0])
        self.assertFalse(h.processor.request_states)

    async def test_abort_before_task_start_and_request_id_reuse_retire_only_old_work(self):
        h = self.handler
        old = h.add("reused")
        await self.dispatcher.submit(f.engine_output(h.tokenizer, "reused"), 0, f.ENGINE_TIMESTAMP)
        await h.methods["abort"](h.owner, "reused", internal=True)
        new = h.add("reused")
        await self.dispatcher.submit(f.engine_output(h.tokenizer, "reused"), 0, f.SECOND_TIMESTAMP)
        await self.idle()
        aborted, completed = old.get_nowait(), new.get_nowait()
        self.assertEqual(aborted.outputs[0].finish_reason, "abort")
        self.assertEqual(aborted.prompt_logprobs, [None])
        self.assertEqual(completed.outputs[0].finish_reason, "length")
        self.assertEqual(h.aborts, ["reused"])

    async def test_abort_active_request_preserves_waiting_request_and_shutdown_retires_work(self):
        h = self.handler
        active, waiting = h.add("active"), h.add("waiting")
        await self.dispatcher.submit(f.engine_output(h.tokenizer, "active"), 0, f.ENGINE_TIMESTAMP)
        await self.dispatcher.submit(f.engine_output(h.tokenizer, "waiting"), 0, f.ENGINE_TIMESTAMP)
        await asyncio.sleep(0)
        self.assertEqual(self.dispatcher.active_heavy, 1)
        await h.methods["abort"](h.owner, "active", internal=True)
        await self.idle()
        self.assertEqual(active.get_nowait().prompt_logprobs, [None])
        self.assertEqual(waiting.get_nowait().outputs[0].finish_reason, "length")
        shutdown = h.add("shutdown")
        await self.dispatcher.submit(f.engine_output(h.tokenizer, "shutdown"), 0, f.ENGINE_TIMESTAMP)
        await asyncio.sleep(0)
        await h.stop()
        self.assert_retired()
        self.assertTrue(self.dispatcher.closed)
        self.assertIsNone(shutdown.get_nowait())
        self.assertEqual(h.processor.request_states["shutdown"].logprobs_processor.prompt_logprobs, [None])

    async def test_native_child_conversion_error_reaches_all_queues_and_retires_work(self):
        h = self.handler
        failed, waiting = h.add("failed"), h.add("waiting")
        output = f.engine_output(h.tokenizer, "failed")
        tensors = output.new_prompt_logprobs_tensors
        output.new_prompt_logprobs_tensors = tensors._replace(selected_token_ranks=tensors.selected_token_ranks[:0])
        with self.assertLogs(f.LOGGER_NAME, level="ERROR"):
            await h.submit([output, f.engine_output(h.tokenizer, "waiting")])
            await asyncio.wait_for(h.owner.output_handler, f.WAIT_SECONDS)
        for queue in (failed, waiting):
            with self.assertRaises(IndexError):
                queue.get_nowait()
        self.assert_retired()

    async def test_engine_error_keeps_original_exception_identity(self):
        h = self.handler
        first, second = h.add("first"), h.add("second")
        await self.dispatcher.submit(f.engine_output(h.tokenizer, "first"), 0, f.ENGINE_TIMESTAMP)
        error = RuntimeError("fixture engine output failure")
        with self.assertLogs(f.LOGGER_NAME, level="ERROR"):
            await h.incoming.put(error)
            await asyncio.wait_for(h.owner.output_handler, f.WAIT_SECONDS)
        for queue in (first, second):
            with self.assertRaises(RuntimeError) as observed:
                queue.get_nowait()
            self.assertIs(observed.exception, error)
        self.assert_retired()


class CompletionTests(unittest.IsolatedAsyncioTestCase):
    async def compare(self, outputs, request, success=True):
        expected = await completion_outcome(deepcopy(outputs), request, cooperative=False)
        actual = await completion_outcome(deepcopy(outputs), request, cooperative=True)
        self.assertEqual(actual, expected)
        self.assertEqual(actual[0] == "body", success)
        return actual

    async def test_nullable_unicode_echo_token_ids_and_multiple_choices_keep_native_bytes(self):
        for values in (None, [], rows()):
            for n in (1, 2):
                for flags in ({}, {"echo": True}, {"return_token_ids": True},
                              {"echo": True, "return_token_ids": True}):
                    with self.subTest(values=values, n=n, flags=flags):
                        await self.compare([f.completion_output(values, n=n)], f.completion_request(n=n, **flags))

    async def test_large_conversion_and_encoding_keep_values_bytes_headers_and_identity(self):
        values = large_rows()
        request = f.completion_request(n=2, echo=True, return_token_ids=True)
        output = f.completion_output(values, n=2)
        identities = [id(row) for row in values]
        response = await f.Completion([output])._create_completion(request)
        self.assertTrue(f.serializer._use_cooperative_encoding(response))
        self.assertEqual(values[-1][1].logprob, CLAMP_FLOOR)
        self.assertEqual([id(row) for row in values], identities)
        choice_ids = [id(choice) for choice in response.choices]
        row_ids = [id(choice.prompt_logprobs) for choice in response.choices]
        actual = await f.serializer.completion_json_response(response, headers=HEADERS)
        expected = JSONResponse(response.model_dump(), headers=HEADERS)
        self.assertEqual((actual.body, dict(actual.headers)), (expected.body, dict(expected.headers)))
        self.assertEqual([id(choice) for choice in response.choices], choice_ids)
        self.assertEqual([id(choice.prompt_logprobs) for choice in response.choices], row_ids)
        decoded = json.loads(actual.body)
        self.assertEqual(decoded["choices"][0]["prompt_logprobs"][-1]["1"]["decoded_token"], "é雪😀")
        self.assertEqual(decoded["usage"]["prompt_tokens"], len(values))
        await self.compare([f.completion_output(large_rows(), n=2)], request)

    async def test_validation_nonfinite_and_multiple_prompt_error_order_match_sync_api(self):
        cases = (
            ([None, {"invalid-id": f.Logprob(-1, 1, "bad")}], "stop"),
            ([None, {1: f.Logprob(float("nan"), 1, "bad")}], "stop"),
            ([None, {1: f.Logprob(float("inf"), 1, "bad")}], "stop"),
            (rows(), "error"), ([None, {1: object()}], "stop"))
        for values, finish in cases:
            with self.subTest(finish=finish, values=values):
                await self.compare([f.completion_output(values, finish=finish)], f.completion_request(), success=False)
        output = f.completion_output([None, {1: object()}])
        output.prompt_token_ids = None
        result = await self.compare([output], f.completion_request(), success=False)
        self.assertEqual(result[0], "AssertionError")
        result = await self.compare([f.completion_output(rows(), finish="error"),
                                     f.completion_output([None, {1: object()}])],
                                    f.completion_request(prompts=2), success=False)
        self.assertEqual(result[0], "GenerationError")
        await self.compare([f.completion_output(rows()) for _ in range(2)], f.completion_request(prompts=2))

    async def test_json_errors_precede_surrogate_encoding_errors_across_native_batches(self):
        for invalid in (float("nan"), float("inf")):
            values = [None] * LARGE_ROWS
            values[1] = {1: f.Logprob(-1, 1, SURROGATE)}
            values[-1] = {1: f.Logprob(invalid, 1, "late")}
            result = await self.compare([f.completion_output(values)], f.completion_request(), success=False)
            self.assertEqual(result[0], "ValueError")
        values[-1] = {1: f.Logprob(-1, 1, "late")}
        result = await self.compare([f.completion_output(values)], f.completion_request(), success=False)
        self.assertEqual(result[0], "UnicodeEncodeError")

    async def test_cancel_actual_conversion_returns_disconnect_without_later_mutation(self):
        values = large_rows()
        native = f.HELPERS["clamp_prompt_logprobs_async"]

        async def observe_clamp(prompt_rows):
            asyncio.get_running_loop().call_soon(task.cancel)
            return await native(prompt_rows)

        f.HELPERS["clamp_prompt_logprobs_async"] = observe_clamp
        try:
            task = asyncio.create_task(f.Completion([f.completion_output(values)])._create_completion(f.completion_request()))
            response = await asyncio.wait_for(task, f.WAIT_SECONDS)
            self.assertEqual(values[1][1].logprob, CLAMP_FLOOR)
            self.assertEqual(values[-1][1].logprob, -float("inf"))
        finally:
            f.HELPERS["clamp_prompt_logprobs_async"] = native
        self.assertIsInstance(response, f.ErrorResponse)
        self.assertEqual(response.error.message, "Client disconnected")
        retired = [row[1].logprob for row in values[1:]]
        await asyncio.sleep(0)
        self.assertEqual([row[1].logprob for row in values[1:]], retired)

    async def test_cancel_encoding_preserves_response_and_allows_native_equivalent_retry(self):
        response = await f.Completion([f.completion_output(large_rows())])._create_completion(f.completion_request())
        expected = JSONResponse(response.model_dump(), headers=HEADERS)
        task = asyncio.create_task(f.serializer.completion_json_response(response, headers=HEADERS))
        await asyncio.sleep(0)
        self.assertFalse(task.done())
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        actual = await f.serializer.completion_json_response(response, headers=HEADERS)
        self.assertEqual((actual.body, dict(actual.headers)), (expected.body, dict(expected.headers)))


if __name__ == "__main__":
    unittest.main()
