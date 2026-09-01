# confidential-glm5-3-flash

vLLM image for GLM-5.3-Flash (NVFP4) on 8x NVIDIA Blackwell.

Follows the [upstream vLLM recipe](https://recipes.vllm.ai/zai-org/GLM-5.3-Flash)
(nvfp4 variant). Deviations:

- Base image digest-pinned for reproducible, attestable builds.
- Weights pinned to `RedHatAI/GLM-5.3-Flash-NVFP4@36c184c6` and served
  from a verified model pack.
- FlashInfer cubins baked at build time (the container runs offline).
- Patches in `patches/`, one line each in the header of the patch file.
