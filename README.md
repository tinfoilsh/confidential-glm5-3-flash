# confidential-glm5-3-flash

GLM-5.3-Flash NVFP4 on four NVIDIA B300 GPUs with confidential computing,
1,048,576-token context, and static five-token MTP. Weights remain pinned to
`RedHatAI/GLM-5.3-Flash-NVFP4@240131d6` in the verified model pack.

This candidate uses digest-pinned vLLM 0.31.0 with the V2 model runner and
CVM 0.14.13. The [patch inventory](patches/README.md) describes explicit UVA
fallback, reduced metadata transfers, and deferred reply readback. Set
`VLLM_TINFOIL_CC_OPTIMIZATIONS=0` to disable the transfer optimizations while
retaining the required `VLLM_DISABLE_UVA=1` fallback.

One deployment uses TP4 with 56 CPUs and 768 GiB RAM. Reply readback and
text-input staging admit TP4 and TP8 within the guarded static-MTP configuration.
The TP4 extension still requires native GPU and serving qualification.

FlashInfer cubins are checksum-verified and baked for offline B300 startup.
Generated code and locks use writable, executable caches. The existing
inference sidecar, authentication, model pack, parsers, and generation defaults
remain in the serving configuration.

Runaway-generation guards use `--chat-template-content-format=string`, a
131072-token fallback for requests that omit `max_tokens`, and the built-in
repetition detector. Set `TINFOIL_REPETITION_DETECTION="max,min,count"` to tune
its default `64,4,16`, or `"0"` to disable it. Requests can override the detector;
tripped requests finish with `finish_reason=repetition`.

The source-patched image is a candidate for qualification. Its policies were
measured using live worker patches; those measurements do not validate this new
image. Read-only GPU startup, correctness and 1M-context checks, and repeated
matched performance runs through the customer-facing stream remain required
before promotion. See [validation and release criteria](docs/qualification.md).
