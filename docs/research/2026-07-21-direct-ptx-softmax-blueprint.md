# Direct-PTX softmax, masking, and log-sum-exp blueprint

Date: 2026-07-21

Tracking issue: #840

Parent blueprint: `2026-07-20-fused-attention-championship-blueprint.md`

## Verdict

This increment establishes the issue-#840 assembly line and implements the
first exact contiguous Ampere FP32 row-softmax family. It specializes
`(rows,columns) = (256,128), (2048,64), (2048,128), and (8192,128)`.
Production remains fail-closed until a cell clears every clean-run and
executed-spill release gate; explicit experiment override is required for an
unpromoted cell. The per-family rollout gate is
`AIDOTNET_DIRECT_PTX_SOFTMAX=1`; the master direct-PTX gate also admits the
family, but the per-cell performance gate still applies. This draft does not
close #840.

The executable 62-cell manifest assigns all public/resident CUDA softmax forward and
backward routes, FP16 routes, log-softmax, log-sum-exp, masking, scatter and
hierarchical variants, the fused linear-softmax boundary, and the shared
attention online-softmax ABI. A row marked planned is not a generic dynamic
kernel: it must be split into exact architecture/dtype/layout/shape cells.

## Assembly-line contract

1. Assign each scoped public and backend entry point to exactly one manifest
   cell before implementation.
2. Freeze phase, formula, numerical mode, dtype, architecture, exact logical
   and physical extent, alignment, alias policy, and fallback.
3. Validate the tensor contract once at admission. The PTX launch receives
   pointers only; dimensions, strides, masks, and modes are module identity.
4. Use a warp or tile owner that loads each element once, retains partials in
   registers/shared memory through reduction and consumers, and commits only
   final outputs. Never materialize global max, sum, or probability buffers.
5. Use `cp.async` or shared-memory tiling only when cross-thread reuse repays
   staging. A single-warp C64/C128 row has no such reuse, so register retention
   and warp shuffles are the lower-traffic design.
6. Prewarm before CUDA graph capture. Reject nonzero local bytes, excessive
   resources, insufficient occupancy, unmeasured architectures, aliases, and
   physical-extent mismatches.
7. Compare only resident NVIDIA paths with identical semantics. Promote an
   exact cell only after three uncontended runs beat the strongest eligible
   current-AiDotNet/PyTorch/NVIDIA competitor by the production gate and
   Nsight proves zero executed spills/local traffic.

## Register-resident dataflow

A warp-row block contains four, eight, or sixteen warps; one warp owns one
row. For C128, each lane owns four contiguous values and issues one aligned
FP32x4 load and one aligned FP32x4 final store; C64 uses FP32x2. The values
remain registers from the sole vector input transaction through maximum
reduction, exponentiation, sum reduction, reciprocal, normalization, and the
sole final vector output transaction. The bounded geometry set is
`{128,256,512}` threads; an exact shape maps to one evidence-selected entry,
while the benchmark-only override can reproduce every candidate.

The vector input uses the normal cache-all path. These tensors are resident
outputs of preceding GPU stages. Cache policy remains part of the exact
measured candidate identity; no policy is promoted from intuition alone.

The small `(256,128)` experiment also has a four-warp CTA-row candidate. Each
thread retains one exponential in a register; a single 16-byte shared region
is reused first for four warp maxima and then four warp sums. No score,
exponential, or probability is materialized globally. This candidate is not
selected merely because it exists: its four barriers and scalar transactions
must beat the warp-row candidate and cuDNN in clean measurements.

```text
coalesced global input loads (one/element)
  -> lane value registers
  -> 5-shuffle warp maximum
  -> subtract max + exp2, retained in the same registers
  -> 5-shuffle warp sum + reciprocal
  -> normalize retained exponentials
  -> coalesced final global output stores (one/element)
```

The established AiDotNet CUDA route writes exponentials to the global output,
rereads them after the row sum, and rewrites normalized outputs. The direct
kernel removes that intermediate global round trip. Warp-row candidates use
no shared memory or barriers. The small CTA-row candidate reuses four
contiguous FP32 shared words: lane 0 of each warp writes one distinct word,
then lanes 0-3 of a warp read distinct 4-byte banks, so neither phase has an
intra-warp bank conflict. Every candidate has no global scratch, temporary
VRAM, or dynamic stride/layout branch. `cp.async` is inapplicable because no
input element has cross-thread reuse; wider rows get separately measured
shared-memory/async families rather than weakening this ABI.

## Formal ABI and admission

The launch contains two 64-bit pointers and no scalar parameters:

| Tensor | Layout | Exact extent | Access | Alignment |
|---|---|---|---|---:|
| input | row-major | `[rows,columns]` FP32 | read | 16 B |
| output | row-major | `[rows,columns]` FP32 | write | 16 B |

Input/output overlap is rejected. Admission checks the rollout gate, CUDA
availability, exact Ampere architecture, supported and promoted shape bucket,
exact allocation extent, physical layout, dtype, alignment, and aliasing once.
Stable rejection reasons distinguish feature-disabled, CUDA-unavailable,
architecture-not-validated, shape-not-implemented,
performance-gate-not-met, and physical-extent-mismatch. Unsupported calls use
the established CUDA route. Capture requires a prewarmed bounded cache entry.

## Correctness and runtime proof

Focused tests enforce:

- two pointer parameters, exact C128 load/store counts, ten warp shuffles,
  four exponentials, and one reciprocal;
- no `.local`, stride token, scalar shape parameter, or generic dynamic-shape
  branch; warp-row PTX additionally has no `.shared` or barrier, while the
  CTA-row emitter is pinned to one 16-byte shared declaration and four
  barriers;
- direct-driver and backend parity with a double-precision oracle;
- exact physical extents, non-aliasing, disabled-by-default admission, stable
  fallback reasons, prewarm, graph capture, bounded module disposal, and zero
  managed allocation across 40 hot calls;
- driver-JIT budgets of at most 32 registers/thread, zero local bytes, at most
  16 static shared bytes, and a geometry-specific minimum occupancy.

The maximum observed direct-vs-double-oracle error is `3.0e-8`.

## GPU championship protocol

The executable TUI reports current AiDotNet CUDA eager/graph, direct PTX
eager/graph, and resident PyTorch CUDA eager/graph/compiled/compiled-graph for
every exact cell. CPU MKL and OpenBLAS are intentionally ineligible: they do
not execute the same resident NVIDIA workload.

Each run uses 30 warmups and 101 samples. Device samples average a captured
batch of 50 resident launches; E2E samples measure one launch plus
synchronization. Upload, download, allocation, compilation, and graph
construction are outside timing. Every backend receives the same
formula-generated FP32 input for a run and shape. Useful work is five FLOPs
and eight bytes per element; reported GFLOPS, TFLOPS, and GB/s use the device
median. Every row also prints
mean/median/P95/P99 device and E2E latency, managed bytes/call, temporary
device bytes, maximum error, registers, static shared bytes, local bytes, and
active blocks/SM. .NET rows use the thread allocation counter; Python rows use
a separately warmed `tracemalloc` median peak per call, outside latency timing.

Clean three-run promotion evidence is recorded only when no unrelated process
is using the GPU. A measurement taken while another CUDA workload was active
is deliberately excluded rather than selected as favorable evidence.

## Resource and spill status

Driver-JIT inspection on the RTX 3080 SM86 reports:

| Last validated candidate | Registers/thread | Static shared B | Local B/thread | Blocks/SM |
|---|---:|---:|---:|---:|
| C64 warp-row, 128 threads | 16 | 0 | 0 | 12 |
| C128 warp-row, 128 threads | 20 | 0 | 0 | 12 |

Those rows describe the last driver-validated vector candidates, not the
pending final cache/CTA selection and not promotion evidence. The final table
is replaced only after the exact selected PTX has been JIT-inspected.

The loader rejects a module outside the declared budget before it enters the
cache. This is proof of zero allocated local bytes, not proof of zero executed
SASS spill/local instructions. Promotion additionally requires the repository
Nsight target and CSV verifier to observe every executed spill, local-load,
local-store, and local-spilling-request counter as zero. No such claim is made
when NVIDIA performance-counter permission is unavailable.

## Reproduction

```powershell
dotnet run --project tests/AiDotNet.Tensors.Benchmarks/AiDotNet.Tensors.Benchmarks.csproj `
  -c Release -- --direct-ptx-softmax 3

powershell -ExecutionPolicy Bypass -File `
  tests/AiDotNet.Tensors.Benchmarks/Profiling/run-direct-ptx-ncu.ps1 `
  -Target softmax -NcuPath <path-to-ncu.exe>
```

## Next #840 increments

1. Fuse scale/bias/causal-or-explicit mask with softmax and eligible consumers
   so scores/probabilities never become global intermediates.
2. Add log-softmax/log-sum-exp forward and backward with a shared saved-state
   contract, then softmax backward and FP16/BF16 shape families.
3. Add wider-row shared-memory and async families only after banking,
   alignment, coalescing, and performance evidence justify them.
4. Complete scatter, hierarchical, sparsemax, Taylor, spherical, masking, and
   fused linear-softmax manifest cells with edge-case/determinism matrices.
5. Add independently emitted and measured Ada, Hopper, and Blackwell modules;
   Ampere evidence never promotes another architecture.
