The candidate combines vLLM 0.31.0 with sampling refresh, nonblocking explicit
H2D, nonreply-copy elimination, deferred reply readback, unchanged block-count
and sampling-metadata mirrors, and private text-input staging. Static MTP uses
five speculative tokens; the long-prefill threshold stays zero. Dynamic MTP
and unmeasured metadata-omission changes are outside this candidate.

The source port removes the experiment's live-hook machinery. It and the new
FlashInfer read-only routing repair require qualification as a new image.
Historical backend-chunk timings do not establish customer-facing streaming
latency or final-image correctness. Raw trials and analysis remain in the
private lab repository.

Build without publishing a release:

```sh
docker build --tag glm-flash-cc-candidate .
```

The build pins the upstream image, checks vLLM's version and build commit,
applies patches with zero fuzz, and verifies FlashInfer source and artifact
hashes. Its artifact report is `/opt/tinfoil/flashinfer-artifacts-report.json`.
Release workflows are manually dispatched and are not needed to review this PR.

The packaged image passes read-only startup on B300 with CC enabled and no debug
extensions. All 64 native CUDA test executions across eight GPUs and 17 serving
checks pass. Uncached recall passes at 32,763 and 1,047,273 prompt tokens.
Cancellation, matched performance, customer-stream latency, and production
attestation remain separate qualification gates.

The CPU policy tests execute the actual patched function and class bodies with
CPU boundaries for CUDA operations. They cover ownership, reused request rows,
partial updates, failed-copy retries, fallback behavior, and deferred output
ordering. They do not establish GPU lifetime or CUDA stream correctness.
Use a virtual environment with NumPy and CPU PyTorch, and point it at a source
tree containing the fully applied `vllm/` directory:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install numpy torch
VLLM_PATCHED_SOURCE=/path/to/patched/site-packages \
  .venv/bin/python -m unittest discover -s tests -v
```

Before promotion, qualify the built image on B300 with the production read-only
root filesystem, offline cubins, CC enabled, and debug extensions absent.
Verify serving APIs, tools and reasoning, repetition behavior, cancellation,
request-slot reuse, token correctness, and uncached near-1M-context retrieval.
Check worker engagement and restore the matched reference between comparisons.

Acceptance requires repeated matched runs and reverse-order confirmation of a
combined throughput gain of at least 5%, beyond observed variation. Preserve
correctness, CC, and 1M context. Report every trial against both criteria sets:

| Metric relative to the matched reference | Original | Revised |
| --- | --- | --- |
| p95 time to first token | ≤ +5% | ≤ +5% |
| p95 time per output token | ≤ +5% | ≤ +10% |
| p95 and p99 customer-visible streaming gaps | Unspecified | ≤ +5% |

Time per output token is a secondary guardrail. Review each request's worst
pauses and stall frequency in addition to pooled percentiles. Label backend
chunk measurements explicitly; they cannot satisfy the customer-stream gate.
Measure the final combination against the original serving reference as well
as the matched v0.31 reference; do not add percentages from separate experiments.
