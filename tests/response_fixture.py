"""Native CPU response components with only engine and renderer boundaries replaced."""

import ast
import asyncio
from collections.abc import Sequence as GenericSequence
from functools import partial
import logging
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import cast

import numpy as np
from tokenizers import AddedToken, Tokenizer, decoders, models
import torch
from transformers import TokenizersBackend


SOURCE = Path(os.environ["VLLM_PATCHED_SOURCE"])
sys.path.insert(0, str(SOURCE))

from vllm import envs
from vllm.entrypoints.generate.base.protocol import RequestResponseMetadata, SpeculativeDecodingMetrics
from vllm.entrypoints.openai.completion import cooperative_response as serializer
from vllm.entrypoints.openai.completion.protocol import CompletionRequest, CompletionResponse, CompletionResponseChoice
from vllm.entrypoints.serve.engine.protocol import ErrorInfo, ErrorResponse, PromptTokenUsageInfo, UsageInfo
from vllm.exceptions import GenerationError
from vllm.logprobs import Logprob
from vllm.outputs import CompletionOutput, RequestOutput
from vllm.platforms import current_platform
from vllm.sampling_params import BeamSearchParams, RequestOutputKind, SamplingParams
from vllm.tokenizers.hf import maybe_make_thread_pool
from vllm.utils.async_utils import merge_async_iterators
from vllm.utils.collection_utils import as_list
from vllm.v1.engine import EngineCoreOutput, EngineCoreOutputs, EngineCoreRequest, FinishReason
from vllm.v1.engine import output_processor
from vllm.v1.engine.output_dispatcher import RequestOutputDispatcher
from vllm.v1.metrics.stats import IterationStats, SchedulerStats
from vllm.v1.outputs import LogprobsLists, LogprobsTensors


for native_module in (output_processor, serializer, sys.modules[RequestOutputDispatcher.__module__]):
    if not Path(native_module.__file__).absolute().is_relative_to(SOURCE.absolute()):
        raise RuntimeError(f"Native component was imported outside VLLM_PATCHED_SOURCE: {native_module.__file__}")


MODEL_NAME = "gpt-oss-120b"
REQUEST_ID = "fixture"
COMPLETION_ID = "cmpl-" + REQUEST_ID
FIXED_TIME = 1000
CONTEXT_LIMIT = 1_048_576
MAX_OUTPUT_TOKENS = 16
PROMPT_TOP_LOGPROBS = 3
BYTE_VALUES = 256
BYTE_OFFSET = 1
UNKNOWN, SPECIAL = "<unk>", "<|special|>"
UTF8_TEXT = "€😀"
PROMPT_VALUES = (-0.5, -1.0, -float("inf"), -0.5)
SCORE_CHUNKS = 4
ENGINE_TIMESTAMP = 42.0
SECOND_TIMESTAMP = 43.0
WAIT_SECONDS = 10
LOGGER_NAME = "response_component_fixture"
logger = logging.getLogger(LOGGER_NAME)


def execute(relative, names, namespace, owner=None):
    path = SOURCE / "vllm" / relative
    tree = ast.parse(path.read_text(), filename=str(path))
    body = tree.body if owner is None else next(
        node.body for node in tree.body if isinstance(node, ast.ClassDef) and node.name == owner)
    selected, found = [], set()
    for node in body:
        defined = ({node.name} if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
                   else {target.id for target in node.targets if isinstance(target, ast.Name)}
                   if isinstance(node, ast.Assign) else set())
        if defined & names:
            selected.append(node)
            found.update(defined & names)
    if found != names:
        raise RuntimeError(f"Missing native definitions in {path}: {names - found}")
    unit = ast.Module(body=ast.parse("from __future__ import annotations").body + selected,
                      type_ignores=[])
    exec(compile(ast.fix_missing_locations(unit), str(path), "exec"), namespace)
    return namespace


def byte_tokenizer():
    vocab = {UNKNOWN: 0, **{f"<0x{value:02X}>": value + BYTE_OFFSET for value in range(BYTE_VALUES)}}
    vocab.update({"hello": len(vocab), SPECIAL: len(vocab) + 1})
    backend = Tokenizer(models.BPE(vocab, [], unk_token=UNKNOWN, byte_fallback=True))
    backend.decoder = decoders.Sequence([decoders.ByteFallback(), decoders.Fuse()])
    backend.add_special_tokens([AddedToken(SPECIAL, special=True)])
    return maybe_make_thread_pool(TokenizersBackend(
        tokenizer_object=backend, unk_token=UNKNOWN, additional_special_tokens=[SPECIAL],
        clean_up_tokenization_spaces=False))


def engine_request(tokenizer, request_id):
    return EngineCoreRequest(
        request_id=request_id, external_req_id=request_id,
        prompt_token_ids=tokenizer.encode("hello", add_special_tokens=False),
        mm_features=None, sampling_params=SamplingParams(
            max_tokens=MAX_OUTPUT_TOKENS, prompt_logprobs=PROMPT_TOP_LOGPROBS, logprobs=1,
            output_kind=RequestOutputKind.CUMULATIVE, skip_special_tokens=False,
            spaces_between_special_tokens=False),
        pooling_params=None, arrival_time=0, lora_request=None, cache_salt=None,
        data_parallel_rank=None)


def engine_output(tokenizer, request_id, *, heavy=True, finish=FinishReason.LENGTH):
    hello, special = tokenizer.convert_tokens_to_ids(["hello", SPECIAL])
    tensors = None
    if heavy:
        repetitions = (SCORE_CHUNKS * output_processor.PROMPT_LOGPROBS_CHUNK_ENTRIES
                       // (len(PROMPT_VALUES) * len(UTF8_TEXT.encode("utf-8")))) + 1
        ids = list(UTF8_TEXT.encode("utf-8")) * repetitions
        tensors = LogprobsTensors(
            torch.tensor([[value + BYTE_OFFSET, hello, special, value + BYTE_OFFSET]
                          for value in ids], dtype=torch.int64),
            torch.tensor([PROMPT_VALUES] * len(ids), dtype=torch.float32),
            torch.ones(len(ids), dtype=torch.int64))
    return EngineCoreOutput(
        request_id=request_id, new_token_ids=[hello],
        new_logprobs=LogprobsLists(np.array([[hello, special]], dtype=np.int64),
                                  np.array([[-0.25, -0.5]], dtype=np.float32),
                                  np.array([1], dtype=np.int64)),
        new_prompt_logprobs_tensors=tensors,
        prompt_token_id_logprobs=torch.tensor([[-0.5, -1.0]], dtype=torch.float32),
        finish_reason=finish)


class Handler:
    def __init__(self):
        self.tokenizer = byte_tokenizer()
        self.processor = output_processor.OutputProcessor(self.tokenizer, log_stats=True)
        self.incoming, self.aborts, self.records, self.backlog = asyncio.Queue(), [], [], []

        async def receive():
            value = await self.incoming.get()
            if isinstance(value, Exception):
                raise value
            return value

        async def abort(request_ids):
            self.aborts.extend(request_ids)

        self.owner = SimpleNamespace(
            output_handler=None, output_processor=self.processor, log_stats=True, log_requests=False,
            logger_manager=SimpleNamespace(record=lambda **values: self.records.append(values),
                                           record_output_processing=lambda *values: self.backlog.append(values)),
            engine_core=SimpleNamespace(get_output_async=receive, abort_requests_async=abort),
            renderer=SimpleNamespace(mm_processor_cache=None, stat_mm_cache=lambda: None))
        namespace = dict(globals(), PROMPT_LOGPROBS_CHUNK_ENTRIES=output_processor.PROMPT_LOGPROBS_CHUNK_ENTRIES)
        execute("v1/engine/async_llm.py", {"OUTPUT_PROCESSING_HEAVY_SLOTS", "OUTPUT_PROCESSING_TENSOR_BUDGET_BYTES"}, namespace)
        self.methods = execute("v1/engine/async_llm.py", {"_run_output_handler", "abort"}, namespace, "AsyncLLM")
        self.methods["_run_output_handler"](self.owner)

    def add(self, request_id):
        request = engine_request(self.tokenizer, request_id)
        queue = output_processor.RequestOutputCollector(request.sampling_params.output_kind, request_id)
        self.processor.add_request(request, "hello", queue=queue)
        return queue

    async def submit(self, outputs, timestamp=ENGINE_TIMESTAMP):
        await self.incoming.put(EngineCoreOutputs(outputs=outputs, timestamp=timestamp,
                                                 scheduler_stats=SchedulerStats()))

    async def stop(self):
        self.owner.output_handler.cancel()
        await asyncio.gather(self.owner.output_handler, return_exceptions=True)


class CompletionBoundary:
    def __init__(self, outputs):
        self.outputs = outputs
        self.default_sampling_params, self.override_max_tokens = {}, None
        self.model_config = SimpleNamespace(max_model_len=CONTEXT_LIMIT)
        self.models = SimpleNamespace(model_name=lambda lora: MODEL_NAME)
        self.renderer = SimpleNamespace(tokenizer=None)
        self.engine_client = SimpleNamespace(generate=self.generate)
        self.enable_prompt_tokens_details, self.enable_per_request_metrics = True, False
        self.system_fingerprint = "cpu-component-fixture"

    async def generate(self, engine_input, sampling_params, request_id, **kwargs):
        yield self.outputs[int(request_id.rsplit("-", 1)[1])]

    async def render_completion_request(self, request):
        return [{"prompt_token_ids": output.prompt_token_ids or [1], "prompt": output.prompt}
                for output in self.outputs]

    _base_request_id = staticmethod(lambda raw, request_id: request_id or REQUEST_ID)
    _maybe_get_adapters = staticmethod(lambda request: None)
    _get_data_parallel_rank = staticmethod(lambda raw: None)
    _extract_prompt_len = staticmethod(lambda engine_input: len(engine_input["prompt_token_ids"]))
    _extract_prompt_text = staticmethod(lambda engine_input: engine_input["prompt"])
    _log_inputs = staticmethod(lambda *args, **kwargs: None)
    _get_session_id = staticmethod(lambda request, raw: None)
    _get_priority = staticmethod(lambda request, raw: request.priority)

    def create_error_response(self, message):
        return ErrorResponse(error=ErrorInfo(message=message, type="BadRequestError", param=None, code=400))


def load_completion():
    namespace = dict(globals(), time=SimpleNamespace(time=lambda: FIXED_TIME))
    execute("entrypoints/generate/base/serving.py", {
        "clamp_prompt_logprobs", "clamp_prompt_logprobs_async", "build_spec_decoding_metrics",
        "PROMPT_LOGPROB_NEGATIVE_INFINITY", "PROMPT_LOGPROB_FLOOR", "PROMPT_LOGPROB_CLAMP_ENTRIES"}, namespace)
    execute("entrypoints/generate/base/serving.py", {"_raise_if_error"}, namespace, "GenerateBaseServing")
    execute("entrypoints/serve/utils/api_utils.py", {"get_max_tokens"}, namespace)
    names = {"_create_completion", "request_output_to_completion_response", "_request_output_to_completion_response"}
    execute("entrypoints/openai/completion/serving.py", names, namespace, "OpenAIServingCompletion")
    cls = type("OpenAIServingCompletion", (CompletionBoundary,),
               {name: namespace[name] for name in names | {"_raise_if_error"}})
    namespace["OpenAIServingCompletion"] = cls
    return cls, namespace


Completion, HELPERS = load_completion()


def completion_output(rows, n=1, finish="stop"):
    size = len(rows) if rows is not None else len(UTF8_TEXT)
    return RequestOutput(
        request_id=REQUEST_ID, prompt="Input", prompt_token_ids=list(range(size)), prompt_logprobs=rows,
        outputs=[CompletionOutput(index=index, text="answer", token_ids=[BYTE_OFFSET],
                                  cumulative_logprob=None, logprobs=None, finish_reason=finish)
                 for index in range(n)], finished=True, num_cached_tokens=0)


def completion_request(prompts=1, **kwargs):
    return CompletionRequest(model=MODEL_NAME, request_id=REQUEST_ID, max_tokens=1,
                             prompt=["Input"] * prompts, prompt_logprobs=PROMPT_TOP_LOGPROBS, **kwargs)


def synchronous_completion(outputs, request):
    return Completion(outputs).request_output_to_completion_response(
        outputs, request, COMPLETION_ID, FIXED_TIME, MODEL_NAME, None,
        RequestResponseMetadata(request_id=COMPLETION_ID))
