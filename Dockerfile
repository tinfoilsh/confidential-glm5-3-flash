# syntax=docker/dockerfile:1.6
#
# Patched vLLM image for GLM-5.3-Flash NVFP4.
#
# Base is the upstream GLM-5.3-Flash tag, digest-pinned for attestation and
# because the tag moves: on 2026-09-09 it was re-pointed off ZJY0516's
# glm-release fork build (0.1.dev20051+g487ecf187) onto upstream main
# @385dce36, once vllm#53906 merged. That commit is post-v0.29.0 -- the
# v0.29.0 release branch was cut 2026-08-31 and does NOT carry GLM-5.3-Flash
# (no Glm5Next* in its registry), which is why the recipe pairs
# min_vllm_version 0.29.0 with nightly_required: true. This digest is the
# same image as vllm/vllm-openai:nightly-385dce36bcee42309924a5ece951a96db3dce7f2.
# It brings FlashInfer 0.6.18 (sparse MLA needs >=0.6.18) and makes the V2
# model runner the default (vllm#53183).
#
# NOTE: this image self-reports "vllm 0.28.1rc1.dev580+g385dce36b" -- lower
# than 0.29.0, but NOT older. v0.29.0 was tagged on a release branch that was
# never merged back into main, so setuptools-scm on main still counts from the
# last tag reachable there (0.28.1rc1). The code is 408 commits past the
# v0.29.0 branch point. Do not "upgrade" this to a 0.29.x tag expecting a
# newer vLLM -- released 0.29.x cannot load GLM-5.3-Flash at all.
ARG VLLM_BASE_IMAGE=vllm/vllm-openai:glm53-flash@sha256:819ec9c063412e5730d1b0e82046ba540d1bf991f3c4f661a849aae8a0c52374
FROM ${VLLM_BASE_IMAGE}

# Patches are -p1 unified diffs rooted at /; they target
# usr/local/lib/python3.12/dist-packages/... to match the base image.
COPY patches/ /tmp/tinfoil-patches/
RUN set -eux; \
    test -x /usr/bin/patch; \
    cd /; \
    for p in /tmp/tinfoil-patches/*.patch; do \
        echo "Applying $(basename "$p")"; \
        /usr/bin/patch -p1 --no-backup-if-mismatch --fuzz=0 < "$p"; \
    done; \
    find /usr/local/lib/python3.12/dist-packages/vllm -name '__pycache__' -type d -exec rm -rf {} + || true; \
    rm -rf /tmp/tinfoil-patches; \
    python3 -c "import vllm; print('vllm', vllm.__version__, 'with tinfoil patches')"

# Bake FlashInfer cubins at build time: the enclave has no egress for JIT
# downloads and the container rootfs is read-only, so the symlinks
# ensure_symlink() would create at runtime are pre-created here.
RUN set -eux; \
    if ! command -v flashinfer >/dev/null 2>&1; then \
        echo "flashinfer CLI not present; skipping cubin bake"; exit 0; \
    fi; \
    flashinfer download-cubin; \
    cubin_dir=$(python3 -c "import flashinfer_cubin, os; print(os.path.join(os.path.dirname(flashinfer_cubin.__file__), 'cubins'))"); \
    du -sh "$cubin_dir"; \
    mkdir -p "$cubin_dir/flashinfer/trtllm/batched_gemm" "$cubin_dir/flashinfer/trtllm/gemm"; \
    for d in "$cubin_dir"/*/; do \
        gemm_dir=$(find "$d" -maxdepth 3 -type d -name "trtllmGen_gemm_export" 2>/dev/null | head -1); \
        bmm_dir=$(find "$d" -maxdepth 3 -type d -name "trtllmGen_bmm_export" 2>/dev/null | head -1); \
        if [ -n "$gemm_dir" ]; then \
            ln -sf "$gemm_dir" "$cubin_dir/flashinfer/trtllm/gemm/trtllmGen_gemm_export"; \
        fi; \
        if [ -n "$bmm_dir" ]; then \
            ln -sf "$bmm_dir" "$cubin_dir/flashinfer/trtllm/batched_gemm/trtllmGen_bmm_export"; \
        fi; \
    done; \
    python3 -c "import flashinfer; print('flashinfer', flashinfer.__version__, 'cubins baked')"
