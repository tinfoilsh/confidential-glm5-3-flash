"""Bake FlashInfer artifacts from the installed version's pinned manifests."""

from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.util
import json
import logging
import os
from pathlib import Path, PurePosixPath

CONTEXT = Path(__file__).resolve().parent
REPORT_PATH = Path("/opt/tinfoil/flashinfer-artifacts-report.json")
DOWNLOAD_WORKERS = 4
FLASHINFER_VERSION = "0.7.0.post1"
TARGET_CPU_ARCH = "x86_64"
TARGET_DSL_ARCH = "sm_103a"
EXPECTED_GENERIC_ENTRIES = 42738
EXPECTED_DSL_ENTRIES = 368
EXPECTED_DSL_BYTES = 149782912
CUDA_HEADER_NAMES = ("cusparse.h", "cublas_v2.h", "cusolverDn.h")
GEMM_MODULES = (
    "trtllm_gemm",
    "trtllm_gemm_sm107",
    "trtllm_low_latency_gemm",
    "trtllm_low_latency_gemm_sm107",
)
MOE_MODULES = ("fused_moe_trtllm_sm100", "fused_moe_trtllm_sm107")


def checked_artifact(name, expected, allow_download=False):
    from flashinfer.jit.cubin_loader import (
        FLASHINFER_CUBIN_DIR,
        get_artifact,
        load_cubin,
    )

    data = (
        get_artifact(name, expected)
        if allow_download
        else load_cubin(str(FLASHINFER_CUBIN_DIR / name), expected)
    )
    actual = hashlib.sha256(data).hexdigest()
    if not data or actual != expected:
        raise RuntimeError(f"Artifact verification failed: {name}")
    return data


def main():
    logging.basicConfig(level=logging.INFO)
    if os.environ.get("FLASHINFER_CUBIN_CHECKSUM_DISABLED"):
        raise RuntimeError("Artifact checksum validation must remain enabled")
    manifest = json.loads((CONTEXT / "flashinfer-artifacts.json").read_text())
    spec = importlib.util.find_spec("flashinfer")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("FlashInfer is missing")
    package_root = Path(next(iter(spec.submodule_search_locations)))
    for name, expected in manifest["flashinfer_source_files"].items():
        actual = hashlib.sha256((package_root / name).read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(f"FlashInfer source mismatch: {name}")

    import flashinfer
    from flashinfer.artifacts import ArtifactPath, CheckSumHash
    from flashinfer.jit import env
    from flashinfer.jit.cubin_loader import ensure_symlink

    if flashinfer.__version__ != FLASHINFER_VERSION:
        raise RuntimeError(f"Unexpected FlashInfer version: {flashinfer.__version__}")
    artifacts = {}
    checksum_manifests = {}
    dsl_prefix = f"{ArtifactPath.DSL_FMHA}{TARGET_CPU_ARCH}/{TARGET_DSL_ARCH}/"
    selected_manifests = [
        f"{path}checksums.txt"
        for path in (
            ArtifactPath.TRTLLM_GEN_FMHA,
            ArtifactPath.TRTLLM_GEN_BMM,
            ArtifactPath.TRTLLM_GEN_GEMM,
            ArtifactPath.DEEPGEMM,
            dsl_prefix,
        )
    ]
    for name in selected_manifests:
        expected = CheckSumHash.map_checksums[name]
        data = checked_artifact(name, expected)
        checksum_manifests[name] = expected
        for line in data.decode().splitlines():
            digest, relative = line.split()
            relative_path = PurePosixPath(relative)
            if relative_path.is_absolute() or ".." in relative_path.parts:
                raise RuntimeError(f"Unexpected artifact path: {relative}")
            artifact = str(PurePosixPath(name).parent / relative_path)
            if artifact in artifacts and artifacts[artifact] != digest:
                raise RuntimeError(f"Conflicting checksums: {artifact}")
            artifacts[artifact] = digest

    dsl_entries = sum(name.startswith(dsl_prefix) for name in artifacts)
    generic_entries = len(artifacts) - dsl_entries
    if (generic_entries, dsl_entries) != (
        EXPECTED_GENERIC_ENTRIES,
        EXPECTED_DSL_ENTRIES,
    ):
        raise RuntimeError("The pinned artifact counts differ from the audited scope")

    def download(item):
        name, expected = item
        is_dsl = name.startswith(dsl_prefix)
        data = checked_artifact(name, expected, allow_download=is_dsl)
        return len(data) if is_dsl else 0

    with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as pool:
        dsl_bytes = sum(pool.map(download, sorted(artifacts.items())))
    if dsl_bytes != EXPECTED_DSL_BYTES:
        raise RuntimeError("The pinned DSL payload size differs from the audited scope")

    gemm_relative = Path("flashinfer/trtllm/gemm/trtllmGen_gemm_export")
    bmm_relative = Path("flashinfer/trtllm/batched_gemm/trtllmGen_bmm_export")
    gemm_target = (
        env.FLASHINFER_CUBIN_DIR
        / ArtifactPath.TRTLLM_GEN_GEMM
        / "include/trtllmGen_gemm_export"
    )
    bmm_target = (
        env.FLASHINFER_CUBIN_DIR
        / ArtifactPath.TRTLLM_GEN_BMM
        / "include/trtllmGen_bmm_export"
    )
    links = {
        env.FLASHINFER_CUBIN_DIR / bmm_relative: bmm_target,
        env.FLASHINFER_GEN_SRC_DIR / bmm_relative: bmm_target,
        **{
            env.FLASHINFER_GEN_SRC_DIR
            / "trtllm_export"
            / module
            / gemm_relative: gemm_target
            for module in GEMM_MODULES
        },
        **{
            env.FLASHINFER_GEN_SRC_DIR
            / "trtllm_export"
            / module
            / bmm_relative: bmm_target
            for module in MOE_MODULES
        },
    }
    for link, target in links.items():
        if not target.is_dir():
            raise RuntimeError(f"Export headers are missing: {target}")
        ensure_symlink(link, target)

    site_packages = package_root.parent
    header_roots = [Path("/usr/local/cuda/include")]
    header_roots.extend(sorted((site_packages / "nvidia").glob("*/include")))
    headers = {
        name: [str(root / name) for root in header_roots if (root / name).is_file()]
        for name in CUDA_HEADER_NAMES
    }
    report = {
        "flashinfer_version": flashinfer.__version__,
        "scope": {
            "hardware": "B300",
            "compute_capability": "10.3",
            "cpu_architecture": TARGET_CPU_ARCH,
            "dsl_architecture": TARGET_DSL_ARCH,
            "retained_and_verified_generic_entries": generic_entries,
            "download_scope": "only missing or invalid SM103a DSL objects",
            "dsl_entries": dsl_entries,
            "dsl_bytes": dsl_bytes,
            "other_dsl_architectures_baked": False,
            "other_hardware_validated": False,
        },
        "cubin_directory": str(env.FLASHINFER_CUBIN_DIR),
        "generated_directory": str(env.FLASHINFER_GEN_SRC_DIR),
        "checksum_manifests": checksum_manifests,
        "artifacts": artifacts,
        "symlinks": {str(link): str(target) for link, target in links.items()},
        "shipped_cuda_headers": headers,
        "globaltimer_compilation": "pending_gpu_probe_no_include_path_change",
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n")
    logging.info("Verified %d artifacts; report: %s", len(artifacts), REPORT_PATH)


if __name__ == "__main__":
    main()
