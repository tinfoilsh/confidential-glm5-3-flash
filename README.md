# confidential-glm5-3-flash

vLLM image for GLM-5.3-Flash (NVFP4) on 8x NVIDIA Blackwell.

Follows the [upstream vLLM recipe](https://recipes.vllm.ai/zai-org/GLM-5.3-Flash)
(nvfp4 variant). The recipe asks for vLLM 0.29.0 *and* a nightly: the v0.29.0
release branch predates GLM-5.3-Flash support (vllm#53906), so the base is
upstream main just after v0.29.0.

Deviations:

- Base image digest-pinned for reproducible, attestable builds. The
  `glm53-flash` tag is mutable and has already changed lineage once.
- Weights pinned to `RedHatAI/GLM-5.3-Flash-NVFP4@240131d6` and served
  from a verified model pack.
- FlashInfer cubins baked at build time (the container runs offline).
- Patches in `patches/`, one line each in the header of the patch file.
- Runaway-generation guards (GLM-5.3-Flash can loop in long tool-calling
  sessions, vllm#54337): `--chat-template-content-format=string`, a
  `max_new_tokens` fallback of 131072 for requests that omit `max_tokens`,
  and `patches/0003`, which arms vLLM's built-in repetition detector for
  requests that do not set `repetition_detection`. Tune the detector with
  the container env `TINFOIL_REPETITION_DETECTION="max,min,count"` (`"0"`
  disables); tripped requests finish with `finish_reason=repetition`.
