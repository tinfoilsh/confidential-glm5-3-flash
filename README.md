# confidential-glm5-3-flash

vLLM image for GLM-5.3-Flash (NVFP4) on 4x NVIDIA B300 (TP4).

Uses pinned vLLM 0.31.0 with the V2 engine.

Deviations:

- Base image digest-pinned for reproducible, attestable builds.
- Weights pinned to `RedHatAI/GLM-5.3-Flash-NVFP4@240131d6` and served
  from a verified model pack.
- FlashInfer cubins baked at build time (the container runs offline).
- Patches in `patches/`.
- Runaway-generation guards (GLM-5.3-Flash can loop in long tool-calling
  sessions, vllm#54337): `--chat-template-content-format=string`, a
  `max_new_tokens` fallback of 131072 for requests that omit `max_tokens`,
  and `patches/0002`, which arms vLLM's built-in repetition detector for
  requests that do not set `repetition_detection`. Tune the detector with
  the container env `TINFOIL_REPETITION_DETECTION="max,min,count"` (`"0"`
  disables); tripped requests finish with `finish_reason=repetition`.
