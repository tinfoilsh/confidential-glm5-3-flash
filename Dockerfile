# syntax=docker/dockerfile:1.6
#
# Patched vLLM image for GLM-5.3-Flash NVFP4.
#
# vLLM 0.31.0 at db9527a46873454610df6dbedf79a36d6bf1a7f6,
# with FlashInfer 0.7.0.post1 and CUDA 13.0.2, pinned for attestation.
ARG VLLM_BASE_IMAGE=vllm/vllm-openai@sha256:a4a4c0437bf7240089da5f08aa370c4aee17ae5290f7a3b468825ee26c4c3a6b
ARG SIDECAR_IMAGE=ghcr.io/tinfoilsh/inference-sidecar@sha256:65ce23d6560c46a1e8614ede187fcbf9798b267aa33878905b4872404787f47d
FROM ${SIDECAR_IMAGE} AS sidecar

FROM ${VLLM_BASE_IMAGE}

ENV VLLM_DISABLE_UVA=1 \
    TINFOIL_REPETITION_DETECTION=64,4,16

# Patches are -p1 unified diffs rooted at /; they target
# usr/local/lib/python3.12/dist-packages/... to match the base image.
COPY patches/ /tmp/tinfoil-patches/
RUN set -eux; \
    test "$VLLM_BUILD_COMMIT" = "db9527a46873454610df6dbedf79a36d6bf1a7f6"; \
    python3 -c "from importlib.metadata import version; assert version('vllm') == '0.31.0'"; \
    test -x /usr/bin/patch; \
    cd /; \
    for p in /tmp/tinfoil-patches/*.patch; do \
        echo "Applying $(basename "$p")"; \
        /usr/bin/patch -p1 --batch --forward --no-backup-if-mismatch --fuzz=0 < "$p"; \
    done; \
    find /usr/local/lib/python3.12/dist-packages/vllm -name '__pycache__' -type d -exec rm -rf {} + || true; \
    rm -rf /tmp/tinfoil-patches; \
    python3 -c "import vllm; print('vllm', vllm.__version__, 'with tinfoil patches')"

# Verify the base's generic artifacts and bake the missing B300 SM103a
# artifacts. Runtime JIT output and locks use the writable cache mounts.
COPY scripts/bake-artifacts.py scripts/flashinfer-artifacts.json /tmp/tinfoil-artifacts/
RUN FLASHINFER_CUDA_ARCH_LIST=10.3a python3 /tmp/tinfoil-artifacts/bake-artifacts.py && \
    rm -rf /tmp/tinfoil-artifacts

COPY --from=sidecar /inference-sidecar /opt/tinfoil/inference-sidecar
ENTRYPOINT ["/opt/tinfoil/inference-sidecar", "vllm", "serve"]
