These patches target vLLM 0.31.0 at
`db9527a46873454610df6dbedf79a36d6bf1a7f6` and FlashInfer 0.7.0.post1 in the
pinned base image. Docker applies them in numeric order at `/` with `patch -p1`
and zero fuzz. They contain ordinary source changes; no live hook loader or
worker debug extension is needed.

| Patch | Purpose |
| --- | --- |
| 0001 | Disable UVA independently of pinned-memory capability; register the opt-in transfer policy. |
| 0002 | Preserve the default repetition detector and explicit request overrides. |
| 0003 | Upload sampling parameters after changes; retain the dirty state when an upload fails. |
| 0004 | Use nonblocking explicit H2D copies while preserving native pool ownership. |
| 0005 | Skip unused ordinary output copies on nonreply ranks; submit reply copies on the existing output thread after the producer event. |
| 0006 | Reuse unchanged block-count and five sampling-metadata device views, with private CPU snapshots and invalidation on partial uploads or failures. |
| 0007 | Use private pageable staging for three text-input arrays in the supported static-MTP configuration; preserve native handling elsewhere. |
| 0008 | Place FlashInfer routing-header links and locks in the writable generated-code directory. |

Patches 0003–0007 are enabled by `VLLM_TINFOIL_CC_OPTIMIZATIONS=1`. Reply
readback retains native handling for optional outputs and unsupported serving
configurations. Patches 0005 and 0007 admit only TP4 and TP8; their TP4 paths
require separate qualification. Metadata caching never skips payload updates
just because counts match. Input staging copies into owned CPU storage before H2D, including
the complete persistent query-start array.

Patch 0008 is a packaging adaptation for the read-only root filesystem and
requires GPU startup qualification. It keeps cubin storage immutable; the bake
manifest verifies the patched FlashInfer source and its artifact checksums.
