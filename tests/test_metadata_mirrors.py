"""Exercise patched metadata publication with real CPU tensors and native callers.

The staged-write kernel runs through a CPU boundary; CUDA ordering is not tested.
Set VLLM_PATCHED_SOURCE to an applied vLLM source tree for local execution.
"""

import ast
from collections.abc import Iterable, Sequence
from functools import partial
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch


REQUEST_SLOTS = 4
POOL_DEPTH = 2
GPU_ROOT = Path("vllm/v1/worker/gpu")
MIRROR_FIELDS = (
    "num_allowed_token_ids",
    "num_logit_bias",
    "min_lens",
    "num_stop_token_ids",
    "restore_when_all_masked",
)


def source_root():
    configured = os.environ.get("VLLM_PATCHED_SOURCE")
    if not configured:
        raise RuntimeError("Set VLLM_PATCHED_SOURCE to the patched vLLM source tree")
    return Path(configured)


def load_classes(relative, names, namespace):
    path = source_root() / relative
    tree = ast.parse(path.read_text())
    selected = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name in names
    ]
    if {node.name for node in selected} != set(names):
        raise RuntimeError(f"Patched source classes are missing in {path}")
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias("annotations")], level=0
            ),
            *selected,
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    exec(compile(module, str(path), "exec"), namespace)
    return namespace


class CpuStagedWriteKernel:
    def __getitem__(self, grid):
        def write(output, stride, indices, starts, contents, ends, *args, **kwargs):
            offset = 0
            for row, start, end in zip(
                indices.tolist(), starts.tolist(), ends.tolist()
            ):
                output[row, start : start + end - offset].copy_(contents[offset:end])
                offset = end

        return write


class MetadataMirrorTests(unittest.TestCase):
    def setUp(self):
        platforms = ModuleType("vllm.platforms")
        platforms.current_platform = SimpleNamespace(device_type="cpu")
        self.enterContext(patch.dict(sys.modules, {"vllm.platforms": platforms}))
        self.envs = SimpleNamespace(VLLM_TINFOIL_CC_OPTIMIZATIONS=True)
        self.buffers = load_classes(
            GPU_ROOT / "buffer_utils.py",
            (
                "UvaBuffer",
                "NonUvaBuffer",
                "UvaBufferPool",
                "UvaBackedTensor",
                "CpuMetadataTensor",
                "StagedWriteTensor",
            ),
            {
                "np": np,
                "torch": torch,
                "envs": self.envs,
                "Sequence": Sequence,
                "Iterable": Iterable,
                "partial": partial,
                "is_uva_available": lambda: False,
                "logger": SimpleNamespace(warning_once=lambda *args: None),
                "_DEFAULT_MAX_CONCURRENCY": POOL_DEPTH,
                "async_tensor_h2d": lambda values, device, dtype: torch.tensor(
                    values, device=device, dtype=dtype
                ),
                "_apply_write_kernel": CpuStagedWriteKernel(),
            },
        )

    def mirror(self):
        return self.buffers["CpuMetadataTensor"](REQUEST_SLOTS, dtype=torch.int32)

    def test_full_publication_reuses_only_unchanged_values(self):
        mirror = self.mirror()
        mirror.np[:] = [1, 2, 3, 4]
        first = mirror.copy_to_uva()
        first_slot = mirror.pool._curr
        self.assertIs(mirror.copy_to_uva(), first)
        self.assertEqual(mirror.pool._curr, first_slot)
        mirror.np[0] = 9
        self.assertEqual(first.tolist(), [1, 2, 3, 4])
        second = mirror.copy_to_uva()
        self.assertEqual(second.tolist(), [9, 2, 3, 4])
        self.assertNotEqual(second.data_ptr(), first.data_ptr())
        self.assertIs(mirror.copy_to_uva(), second)

    def test_failure_does_not_mark_metadata_as_published(self):
        mirror = self.mirror()
        mirror.np[:] = [1, 2, 3, 4]
        mirror.copy_to_uva()
        mirror.np[0] = 9
        with patch.object(
            mirror.pool, "copy_to_uva", side_effect=RuntimeError("upload")
        ):
            with self.assertRaisesRegex(RuntimeError, "upload"):
                mirror.copy_to_uva()
        self.assertIsNone(mirror._last_published)
        self.assertEqual(mirror.copy_to_uva().tolist(), [9, 2, 3, 4])

    def test_partial_publication_invalidates_full_snapshot(self):
        mirror = self.mirror()
        mirror.np[:] = [1, 2, 3, 4]
        mirror.copy_to_uva()
        first = mirror.copy_to_uva(2)
        self.assertEqual(first.tolist(), [1, 2])
        self.assertIsNone(mirror._last_published)
        second = mirror.copy_to_uva(2)
        self.assertNotEqual(first.data_ptr(), second.data_ptr())
        self.assertEqual(mirror.copy_to_uva().tolist(), [1, 2, 3, 4])

    def test_disabled_gate_preserves_native_publication(self):
        self.envs.VLLM_TINFOIL_CC_OPTIMIZATIONS = False
        mirror = self.mirror()
        mirror.np[:] = [1, 2, 3, 4]
        first = mirror.copy_to_uva()
        second = mirror.copy_to_uva()
        self.assertNotEqual(first.data_ptr(), second.data_ptr())
        self.assertEqual(second.tolist(), [1, 2, 3, 4])
        self.assertIsNone(mirror._last_published)

    def test_reused_request_preserves_same_count_payload_writes_and_resets(self):
        bias = load_classes(
            GPU_ROOT / "sample/logit_bias.py",
            ("LogitBiasState",),
            {
                **self.buffers,
                "LogitsProcessor": object,
                "MAX_NUM_ALLOWED_TOKEN_IDS": 1024,
                "MAX_NUM_LOGIT_BIAS_TOKENS": 1024,
                "MAX_NUM_STOP_TOKEN_IDS": 128,
            },
        )
        requests = SimpleNamespace(
            prompt_len=SimpleNamespace(np=np.array([7, 11, 13, 17], dtype=np.int32)),
            max_num_reqs=REQUEST_SLOTS,
            device=torch.device("cpu"),
        )
        state = bias["LogitBiasState"](None, requests)
        settings = SimpleNamespace(
            allowed_token_ids=[3, 4],
            logit_bias={5: 0.5},
            min_tokens=2,
            all_stop_token_ids=[9, 10],
            structured_outputs=object(),
        )
        state.add_request(0, settings)
        state.apply_staged_writes()
        previous = {field: getattr(state, field).gpu for field in MIRROR_FIELDS}
        settings.allowed_token_ids = [7, 8]
        settings.logit_bias = {11: 0.75}
        settings.all_stop_token_ids = [12, 13]
        state.add_request(0, settings)
        state.apply_staged_writes()
        for field in MIRROR_FIELDS:
            self.assertIs(getattr(state, field).gpu, previous[field])
        self.assertEqual(state.allowed_token_ids.gpu[0, :2].tolist(), [7, 8])
        self.assertEqual(state.logit_bias_token_ids.gpu[0, :1].tolist(), [11])
        self.assertEqual(state.logit_bias.gpu[0, :1].tolist(), [0.75])
        self.assertEqual(state.stop_token_ids.gpu[0, :2].tolist(), [12, 13])
        settings.allowed_token_ids = None
        settings.logit_bias = None
        settings.min_tokens = 0
        settings.all_stop_token_ids = []
        settings.structured_outputs = None
        self.assertFalse(state.add_request(0, settings))
        state.apply_staged_writes()
        self.assertEqual(
            [getattr(state, field).gpu[0].item() for field in MIRROR_FIELDS],
            [0, 0, 7, 0, 0],
        )

    def test_same_count_block_replacement_still_publishes_block_ids(self):
        blocks = load_classes(
            GPU_ROOT / "block_table.py", ("BlockTables",), dict(self.buffers)
        )
        tables = blocks["BlockTables"](
            block_sizes=[4],
            max_num_reqs=REQUEST_SLOTS,
            max_num_batched_tokens=8,
            max_num_blocks_per_group=[4],
            device=torch.device("cpu"),
            kernel_block_sizes=[4],
        )
        tables.append_block_ids(0, ([2, 3],), overwrite=True)
        tables.apply_staged_writes()
        counts = tables.num_blocks.gpu
        tables.append_block_ids(0, ([7, 8],), overwrite=True)
        tables.apply_staged_writes()
        self.assertIs(tables.num_blocks.gpu, counts)
        self.assertEqual(tables.num_blocks.gpu[0].tolist(), [2, 0, 0, 0])
        self.assertEqual(tables.block_tables[0].gpu[0, :2].tolist(), [7, 8])


if __name__ == "__main__":
    unittest.main()
