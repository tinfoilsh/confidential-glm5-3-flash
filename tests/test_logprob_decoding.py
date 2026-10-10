"""Exercise real tokenizers and prompt-logprob assembly without GPU imports.

Set VLLM_PATCHED_SOURCE to an applied vLLM source tree for local execution.
These component checks do not establish server availability under load.
"""

import ast
from dataclasses import dataclass
import itertools
import os
from pathlib import Path
import sys
from types import ModuleType
import unittest
from unittest.mock import patch

import numpy as np
from tokenizers import AddedToken, Tokenizer, decoders, models, pre_tokenizers
import torch
from transformers import TokenizersBackend


SOURCE = Path(os.environ["VLLM_PATCHED_SOURCE"])
UNKNOWN = "<unk>"
SPECIAL = "<|special|>"
BYTE_VALUES = 256
BYTE_OFFSET = 1
REPETITIONS = 4096
UTF8_TEXT = "€😀€😀"
PROMPT_TOP_LOGPROBS = 3
PROMPT_VALUES = (-0.5, -1.0, -float("inf"), -0.5)
UTF8_CHUNK_BOUNDARY = 2


def load(relative, namespace=None, names=None):
    path = SOURCE / relative
    tree = ast.parse(path.read_text(), filename=str(path))
    if names is None:
        tree.body = [node for node in tree.body if not (
            isinstance(node, ast.ImportFrom) and node.module == "vllm.tokenizers"
        )]
    else:
        selected = [node for node in tree.body if isinstance(
            node, (ast.ClassDef, ast.FunctionDef)) and node.name in names]
        if {node.name for node in selected} != set(names):
            raise RuntimeError(f"Required source definitions are missing in {path}")
        tree.body = ast.parse("from __future__ import annotations").body + selected
    module = ModuleType("decoding_test_" + relative.replace("/", "_")[:-3])
    module.__dict__.update(namespace or {})
    sys.modules[module.__name__] = module
    exec(compile(tree, str(path), "exec"), module.__dict__)
    return module


def tokenizer(backend):
    backend.add_special_tokens([AddedToken(SPECIAL, special=True)])
    return TokenizersBackend(
        tokenizer_object=backend, unk_token=UNKNOWN,
        additional_special_tokens=[SPECIAL], clean_up_tokenization_spaces=False,
    )


def word_tokenizer(word="alpha"):
    backend = Tokenizer(models.WordLevel(
        {UNKNOWN: 0, word: 1, " beta": 2, " .": 3, SPECIAL: 4}, unk_token=UNKNOWN,
    ))
    backend.decoder = decoders.Fuse()
    return tokenizer(backend)


def metaspace_tokenizer():
    pieces = [UNKNOWN, "▁hello", "▁▁indent", "world", SPECIAL]
    backend = Tokenizer(models.Unigram(
        [(piece, -float(index)) for index, piece in enumerate(pieces)], unk_id=0,
    ))
    backend.pre_tokenizer = pre_tokenizers.Metaspace(
        replacement="▁", prepend_scheme="always",
    )
    backend.decoder = decoders.Metaspace(replacement="▁", prepend_scheme="always")
    return tokenizer(backend)


def byte_tokenizer():
    vocab = {UNKNOWN: 0, **{
        f"<0x{value:02X}>": value + BYTE_OFFSET for value in range(BYTE_VALUES)
    }}
    vocab.update({"hello": len(vocab), SPECIAL: len(vocab) + 1})
    backend = Tokenizer(models.BPE(vocab, [], unk_token=UNKNOWN, byte_fallback=True))
    backend.decoder = decoders.Sequence([decoders.ByteFallback(), decoders.Fuse()])
    return tokenizer(backend)


class LogprobDecodingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.helper = load("vllm/tokenizers/detokenizer_utils.py")
        data = load("vllm/logprobs.py")
        cls.processor = load("vllm/v1/engine/logprobs.py", {
            "np": np, "dataclass": dataclass, "NONES": itertools.repeat(None),
            "convert_ids_list_to_tokens": cls.helper.convert_ids_list_to_tokens,
            **{name: getattr(data, name) for name in (
                "FlatLogprobs", "append_logprobs_for_next_position",
                "create_prompt_logprobs", "create_sample_logprobs",
            )},
        }, {"LogprobsProcessor"}).LogprobsProcessor

    def test_repeated_tokens_decode_once_and_preserve_order(self):
        cases = (
            (word_tokenizer(), [" .", "alpha", SPECIAL, " beta"]),
            (metaspace_tokenizer(), ["▁▁indent", "world", SPECIAL, "▁hello"]),
        )
        expected_text = (
            [" .", "alpha", SPECIAL, " beta"],
            ["  indent", "world", SPECIAL, " hello"],
        )
        for (backend, pieces), expected in zip(cases, expected_text):
            with self.subTest(pieces=pieces):
                distinct = backend.convert_tokens_to_ids(pieces)
                ids = distinct * REPETITIONS
                original = list(ids)
                with patch.object(backend, "decode", wraps=backend.decode) as decode:
                    actual = self.helper.convert_ids_list_to_tokens(backend, ids)
                    self.assertEqual(actual, expected * REPETITIONS)
                    self.assertEqual(
                        [call.args[0] for call in decode.call_args_list],
                        [[token_id] for token_id in distinct],
                    )
                    decode.reset_mock()
                    self.assertEqual(self.helper.convert_ids_list_to_tokens(backend, []), [])
                    decode.assert_not_called()
                self.assertEqual(ids, original)

    def test_every_byte_token_preserves_individual_decoding(self):
        backend = byte_tokenizer()
        ids = list(range(len(backend))) * 2
        expected = [backend.decode([token_id]) or "" for token_id in ids]
        self.assertEqual(self.helper.convert_ids_list_to_tokens(backend, ids), expected)

    def test_cache_does_not_cross_tokenizers_configuration_or_vocabulary_changes(self):
        first, second = word_tokenizer("alpha"), word_tokenizer("omega")
        decode = self.helper.convert_ids_list_to_tokens
        self.assertEqual(decode(first, [1, 1]), ["alpha", "alpha"])
        self.assertEqual(decode(second, [1, 1]), ["omega", "omega"])
        self.assertEqual(decode(first, [3, 3]), [" .", " ."])
        first.clean_up_tokenization_spaces = True
        self.assertEqual(decode(first, [3, 3]), [".", "."])
        first.add_tokens(["new-vocabulary-token"])
        added_id = first.convert_tokens_to_ids("new-vocabulary-token")
        self.assertEqual(decode(first, [added_id, added_id]), ["new-vocabulary-token"] * 2)

    def test_prompt_logprobs_preserve_utf8_across_prefill_chunks(self):
        backend = byte_tokenizer()
        hello = backend.convert_tokens_to_ids("hello")
        special = backend.convert_tokens_to_ids(SPECIAL)
        sampled_ids = [value + BYTE_OFFSET for value in UTF8_TEXT.encode("utf-8")]
        rows = [[token_id, hello, special, token_id] for token_id in sampled_ids]
        tensors = (
            torch.tensor(rows, dtype=torch.int64),
            torch.tensor([PROMPT_VALUES] * len(rows), dtype=torch.float32),
            torch.ones(len(rows), dtype=torch.int64),
        )
        processor = self.processor(
            tokenizer=backend, logprobs=None, prompt_logprobs=[None],
            cumulative_logprob=None, num_logprobs=None,
            num_prompt_logprobs=PROMPT_TOP_LOGPROBS,
        )
        for start, end in ((0, UTF8_CHUNK_BOUNDARY), (UTF8_CHUNK_BOUNDARY, len(rows))):
            processor._update_prompt_logprobs(tuple(tensor[start:end] for tensor in tensors))
        self.assertIsNone(processor.prompt_logprobs[0])
        self.assertEqual(len(processor.prompt_logprobs), len(rows) + 1)
        decoded = []
        for token_id, output in zip(sampled_ids, processor.prompt_logprobs[1:]):
            self.assertEqual(list(output), [token_id, hello, special])
            self.assertEqual(output[token_id].logprob, PROMPT_VALUES[0])
            self.assertEqual(output[token_id].rank, 3)
            self.assertEqual(output[hello].logprob, PROMPT_VALUES[1])
            self.assertEqual(output[hello].rank, 1)
            self.assertEqual(output[hello].decoded_token, "hello")
            self.assertEqual(output[special].logprob, PROMPT_VALUES[2])
            self.assertEqual(output[special].rank, 2)
            self.assertEqual(output[special].decoded_token, SPECIAL)
            decoded.append(output[token_id].decoded_token)
        self.assertEqual("".join(decoded), UTF8_TEXT)
        self.assertEqual(tensors[0].tolist(), rows)


if __name__ == "__main__":
    unittest.main()
