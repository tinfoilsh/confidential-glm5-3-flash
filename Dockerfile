# syntax=docker/dockerfile:1.6
#
# Patched vLLM image for GLM-5.3-Flash NVFP4. Base is the dedicated
# upstream GLM-5.3-Flash tag, digest-pinned for attestation.
ARG VLLM_BASE_IMAGE=vllm/vllm-openai:glm53-flash@sha256:2c6da6c6f16ed15c91e412d896dba13701f25fe1861eaec9ddaa4db34d1d21c4
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
