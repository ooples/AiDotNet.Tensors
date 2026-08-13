# Direct PTX dense solver blueprint (issue #853)

Status: experimental, disabled by default, and not promoted. The implementation and non-GPU validation infrastructure are present. No CUDA kernel, performance benchmark, or Nsight target was executed while preparing this change, so all hardware evidence below is explicitly pending.

## Scope implemented

The first full solver-family specialization is FP32, contiguous row-major, batch-of-4x4. Fourteen separately emitted entry points cover the issue's decomposition, solver, and required backward surface:

| Operation | Direct PTX dataflow | Established fallback |
|---|---|---|
| Cholesky | One thread loads 16 values, computes lower factor in registers, writes factor and info | `parity211_cholesky` NVRTC, then managed fallback |
| LU factor | One thread performs four partial-pivot stages in registers and writes packed LU plus zero-based pivots | `parity211_lu_factor` NVRTC, then managed fallback |
| QR reduced/complete/R-only | One thread performs fixed modified Gram-Schmidt in registers and writes Q and R; square complete/reduced extents coincide and R-only projects R | `parity211_qr_reduced` NVRTC, then managed fallback |
| Symmetric Eigh upper/lower | Separate entry points symmetrize from the authoritative triangle, perform fixed Jacobi sweeps, sort eigenpairs, and write W and V | Upper uses `parity211_eigh` NVRTC; lower uses managed fallback |
| Reduced SVD | One thread forms AtA, performs Jacobi eigensolve, reconstructs U, and writes U/S/Vh | Managed SVD fallback (no previous executable CUDA route) |
| LU solve | One thread applies four pivots and performs forward/back substitution for one vector RHS | Managed LU-solve fallback (no previous executable CUDA route) |
| LDL factor/solve | Separate kernels perform symmetric diagonal pivoting and the three-stage vector solve in registers | Managed LDL fallback |
| General solve/SolveEx | One fused kernel performs partial-pivot LU and vector solve without a global factor, writing solution and first-zero-pivot info | Managed structure-aware solver fallback |
| Triangular solve | Separate upper/lower non-unit register substitution kernels | Managed BLAS TRSM fallback |
| Cholesky backward | One thread evaluates the lower-factor Murray/Stan formula with all matrix intermediates in registers | Existing tensor-primitive backward fallback |
| Solve backward | One thread solves the transposed system and writes both the negative outer-product matrix gradient and RHS gradient | Existing tensor-primitive backward fallback |

`IExtendedLinalgBackend` and `DirectGpuTensorEngine` expose these candidates without widening the established backend contract. Public routes cover LU/P-L-U projection, every square QR mode, both Eigh triangles, SVD/SvdVals/current-q-reserved SvdLowRank, LU/LDL/general/triangular solves, square vector Lstsq, and the wired Cholesky/Solve backward paths. Every exact-contract rejection preserves the established implementation.

## Physical ABI

All admitted PTX entry points contain only 64-bit device pointers. There are no matrix order, batch, stride, leading dimension, layout, triangle, RHS-count, or mode parameters in the kernel ABI.

The host validates the complete contract once:

- exact SM86 architecture identity;
- FP32 matrices with exact logical and physical extent `[batch,4,4]`;
- exact batch bucket in `1024, 4096, 16384, 65536`;
- canonical contiguous row-major storage with zero byte offset;
- exact allocation byte extent;
- 16-byte matrix/vector/pivot alignment (4-byte info alignment for Cholesky);
- disjoint inputs and outputs;
- lower-only Cholesky/LDL, separately baked upper/lower Eigh and triangular modes, square QR/SVD, and one vector RHS for solve kernels;
- per-operation opt-in gate.

Any mismatch returns a stable reason and immediately executes the established path. Other Ampere variants, Ada, Hopper, and Blackwell do not inherit SM86 admission.

## Memory traffic and register ownership

Each thread owns one 64-byte matrix. A 4x4 matrix is small enough that shared-memory staging and `cp.async` would add synchronization and instruction overhead without increasing reuse. The intended fast path therefore uses vectorized global loads into registers, keeps all factorization state and intermediates in registers, and performs one final output commit.

There are no workspace allocations and no global intermediate tensors in any direct kernel. The exact expected bytes per matrix are recorded by the benchmark. This choice must still be validated by the championship benchmark; it is an architecture hypothesis, not a performance claim.

## Gates and lifecycle

The master gate is `AIDOTNET_DIRECT_PTX=1`. Independent rollout gates are:

- `AIDOTNET_DIRECT_PTX_CHOLESKY_4X4=1`
- `AIDOTNET_DIRECT_PTX_LU_FACTOR_4X4=1`
- `AIDOTNET_DIRECT_PTX_QR_4X4=1`
- `AIDOTNET_DIRECT_PTX_EIGH_4X4=1`
- `AIDOTNET_DIRECT_PTX_SVD_4X4=1`
- `AIDOTNET_DIRECT_PTX_LU_SOLVE_4X4=1`
- `AIDOTNET_DIRECT_PTX_LDL_FACTOR_4X4=1`
- `AIDOTNET_DIRECT_PTX_LDL_SOLVE_4X4=1`
- `AIDOTNET_DIRECT_PTX_SOLVE_4X4=1`
- `AIDOTNET_DIRECT_PTX_TRIANGULAR_SOLVE_4X4=1`
- `AIDOTNET_DIRECT_PTX_SOLVER_BACKWARD_4X4=1`

The process-start gate is allocation-free. Benchmark/test overrides are thread-local.

The module key is operation, exact batch, and selected block geometry. Device UUID, exact SM, and driver are implicit in the owning backend/runtime instance and recorded in every audit. The bounded cold-path tuner compares exactly 16, 32, 64, 128, and 256 threads. The 16-thread candidate deliberately trades half-warp lane utilization for 64 independent B=1024 CTAs, which can fill a 64-SM-class GPU where a full-warp CTA exposes only 32 blocks. Its timing events are externally observable nodes inside the captured launch group, excluding host submission gaps. B=1024 uses 1,000 launches per tuning sample because the competing geometries differ by fractions of a microsecond; larger buckets retain 10 launches so tuning cost stays bounded. A later geometry must beat the current smaller winner by at least 1% so a near-tie cannot become a different cold-start plan. Plans are bounded to 4 Cholesky cells and 64 solver-family cells; module caches are bounded to 12 and 192 entries respectively. Prewarm installs the deterministic 128-thread plan. Cold graph capture is rejected. A prewarmed selected module is pinned for the lifetime of the backend once captured, and every plan/module is deterministically disposed with the backend.

## Correctness matrix

The opt-in driver theory covers every operation. Required evidence before promotion includes:

The register Cholesky specialization derives each diagonal and its reusable inverse from a
one-Newton-refined `rsqrt.approx.f32` seed; this avoids a serialized square-root-plus-divide
sequence. The refined-seed arithmetic is part of the audited blueprint semantics and remains
subject to the unchanged `2e-5` release residual gate.

- SPD and non-SPD Cholesky with exact `info` parity;
- LU ties, singular pivots, zero-based pivot order, and reconstruction;
- QR orthogonality/reconstruction, rank-deficient columns, and sign-insensitive comparison;
- repeated/clustered eigenvalues, ordering, eigenvector residual, and upper-triangle semantics;
- zero/repeated singular values, descending order, reconstruction, and sign ambiguity;
- LU solve pivot ordering, singular NaN behavior, and vector RHS;
- LDL pivot/reconstruction and three-stage solve behavior;
- general solve info, upper/lower triangular substitution, and square least-squares diagnostics;
- Cholesky and vector-solve backward gradients against the fp64 formula;
- every eligibility rejection, fallback reason, prewarm, capture, pin, bounded eviction, and disposal path;
- public-route and gradient-tape parity against the existing high-precision implementation.

None of these GPU assertions is claimed as passed in this change. Static emitter, manifest, architecture, and gate tests can run without CUDA; driver tests are named `DriverOnly` and remain opt-in.

## Championship evidence protocol

Every result requires three independent clean processes, resident preallocated buffers, CUDA-event device timing, synchronized end-to-end timing, and the complete GPU/SM/UUID/driver/runtime fingerprint. The AiDotNet lane uses at least 30 warmups and 101 samples. Device timing first measures a 1,000-launch captured calibration batch, then independently expands each resident method's graph to target a minimum 32 ms device interval, capped at 32,000 launches. Every final distribution therefore retains at least 1,000 launches per sample and reports its exact count. Externally observable start/stop event nodes remain inside that same graph, so host submission gaps cannot be mislabeled as device work; equal-duration calibration amortizes fixed WDDM desktop-preemption quanta without reducing the sample count or loosening the P95 gate. Candidate and incumbent graphs rotate within every sample round so neither receives a different scheduler or power-state window. A captured method reuses the isolated candidate-kernel distribution because capture changes submission rather than kernel execution. End-to-end timing remains separately measured for resident and captured submission, with resident, graph, and incumbent actions interleaved across 101 rotating rounds so their production costs remain visible.

The production CUDA backend, legacy cuBLAS/cuDNN context, and standalone direct-PTX
runtime use the driver's `CU_CTX_SCHED_SPIN` policy by default. On the validation
WDDM host, the driver's automatic policy parked the synchronization thread and added
roughly 20–25 us of wake latency around 1–5 us kernels, masking real device wins in
end-to-end use. Spin scheduling is a system-wide latency policy rather than a
solver-specific shortcut: every CUDA kernel benefits, at the explicit cost of an
actively waiting CPU thread while synchronization is pending. Deployments that
prioritize idle CPU usage can set `AIDOTNET_CUDA_CONTEXT_SCHEDULING` to `auto`,
`yield`, or `blocking`; invalid values fail during context creation instead of
silently selecting a policy.

The release runner also admits a process only after both GPU and host are quiet. On
Windows it samples `GetSystemTimes` before admission and across the complete child
process, records average CPU occupancy, subtracts 1.5 logical processors for the
benchmark's own submission/spin-wait work, and rejects an attempt when the remaining
foreign CPU load exceeds 20%. This is an environment rejection, not a performance
retry: saturated-host medians remain preserved in rejected evidence and cannot be
promoted as a stable win.

The external PyTorch lane calibrates one CUDA-event-timed call before choosing its measurement plan. Calls below 1 ms retain 101 samples and one to ten launches per device sample; calls at or above 1 ms use at least 21 samples and one launch. Warmup work targets 10 ms but is bounded to 3–30 calls. This keeps close competitors on the full 101-sample protocol while preventing pathologically slow vendor calls from being repeated without bound. For example, PyTorch QR at batch 4096 measured about 0.60 seconds per eager call on the validation host; fixed 10x batching executed 1,212 calls per row, whereas 21 one-launch samples still produce a meaningful empirical P95 and are repeated in all three independent processes. Required eager measurements run first in one uninterrupted resident process so context and library warm state are comparable across the matrix; the parent releases each shape's tensors before advancing. Only after that phase does each optional CUDA-graph capture execute in a disposable child. An unsupported capture may invalidate that child, but it cannot poison or context-switch the eager lane. The release gate validates the sampling metadata and requires every eager cell before accepting external evidence; the correctness, 1.10x device/E2E median, and device-P95 requirements are unchanged.

The orchestrator checks each complete AiDotNet solver attempt against the same
internal correctness, resource, 1.10x device/E2E median, and device-P95 gates
before starting its external competitor process. A failed attempt is preserved
as rejected evidence and retried only within the configured contamination-retry
budget. Every accepted process must still pass every unchanged threshold; a
repeatable kernel loss exhausts that bounded budget and fails the run. This
prevents a transient WDDM-wide latency excursion from launching hours of external
work while retaining the excursion for diagnosis instead of silently averaging
it away.

Run the complete release orchestrator. It starts each AiDotNet and PyTorch run in
a separate clean process and joins the rows through the release gate:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File tests/AiDotNet.Tensors.Benchmarks/Profiling/run-direct-ptx-release-evidence.ps1 -Runs 3 -Issue853Only
```

For a diagnostic rerun of one oracle cell, append
`--operation <name> --batch <1024|4096|16384|65536>`. Filtered output is
diagnostic only; release evidence still requires the complete matrix in three
independent processes.

Run only the PyTorch/cuSOLVER eager and CUDA-graph competitors for diagnosis:

```powershell
python tests/AiDotNet.Tensors.Benchmarks/BaselineRunners/py/run_direct_ptx_solver4x4_competitors.py --runs 3
```

The output is JSON-lines. Group rows by `(run, operation, batch)` before selecting the winner. FLOP rates for iterative Eigh/SVD use the fixed algorithm counts documented in the harness, not vendor peak FLOPS.

### Pending grouped result table

Every cell is pending because GPU execution was intentionally skipped.

| Operation | Batch | Competitor | Device mean/median/P95/P99 us | E2E mean/median/P95/P99 us | GFLOPS | GB/s | Managed B/call | Temp/peak VRAM B | Max error | Regs/shared/local/occupancy | Winner |
|---|---:|---|---|---|---:|---:|---:|---:|---:|---|---|
| Cholesky | 1024/4096/16384/65536 | Direct PTX graph; Direct PTX resident; AiDotNet NVRTC; PyTorch eager; PyTorch graph/cuSOLVER | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| LU factor | 1024/4096/16384/65536 | Direct PTX graph; Direct PTX resident; AiDotNet NVRTC; PyTorch eager; PyTorch graph/cuSOLVER | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| Reduced QR | 1024/4096/16384/65536 | Direct PTX graph; Direct PTX resident; AiDotNet NVRTC; PyTorch eager; PyTorch graph/cuSOLVER | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| Symmetric Eigh (upper) | 1024/4096/16384/65536 | Direct PTX graph; Direct PTX resident; AiDotNet NVRTC; PyTorch eager; PyTorch graph/cuSOLVER | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| Symmetric Eigh (lower) | 1024/4096/16384/65536 | Direct PTX graph; Direct PTX resident; PyTorch eager; PyTorch graph/cuSOLVER | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| Reduced SVD | 1024/4096/16384/65536 | Direct PTX graph; Direct PTX resident; PyTorch eager; PyTorch graph/cuSOLVER | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| LU solve | 1024/4096/16384/65536 | Direct PTX graph; Direct PTX resident; PyTorch eager; PyTorch graph/cuSOLVER | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| LDL factor | 1024/4096/16384/65536 | Direct PTX graph; Direct PTX resident; PyTorch eager; PyTorch graph | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| LDL solve | 1024/4096/16384/65536 | Direct PTX graph; Direct PTX resident; PyTorch eager; PyTorch graph | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| General solve/SolveEx | 1024/4096/16384/65536 | Direct PTX graph; Direct PTX resident; PyTorch eager; PyTorch graph/cuSOLVER | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| Triangular lower | 1024/4096/16384/65536 | Direct PTX graph; Direct PTX resident; PyTorch eager; PyTorch graph | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| Triangular upper | 1024/4096/16384/65536 | Direct PTX graph; Direct PTX resident; PyTorch eager; PyTorch graph | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| Cholesky backward | 1024/4096/16384/65536 | Direct PTX graph; Direct PTX resident; PyTorch autograd eager; PyTorch graph when capture-supported | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| Solve backward | 1024/4096/16384/65536 | Direct PTX graph; Direct PTX resident; PyTorch autograd eager; PyTorch graph when capture-supported | pending | pending | pending | pending | pending | pending | pending | pending | pending |

No CPU number may be used to declare a GPU winner.

## Zero-spill proof

Build the benchmark and run the deterministic target:

```powershell
dotnet build tests/AiDotNet.Tensors.Benchmarks/AiDotNet.Tensors.Benchmarks.csproj -c Release -f net10.0
pwsh tests/AiDotNet.Tensors.Benchmarks/Profiling/run-direct-ptx-ncu.ps1 -Target solvers-4x4
```

The target launches 56 entries: fourteen entry points times four batch buckets. The verifier requires all expected rows and all requested metrics. Promotion requires driver-reported local bytes equal to zero and Nsight evidence of zero executed register spills, local loads, and local stores for every selected tuned variant.

## Promotion gate

No cell is promoted until all of the following are attached for every batch bucket:

- full correctness and fallback matrix passes;
- at least 1.10x device median speedup over the strongest eligible NVIDIA competitor;
- candidate P95 is no worse than competitor P95 plus 10 percent;
- zero hot managed allocation and zero avoidable temporary VRAM;
- operation-specific numerical tolerance passes;
- JIT resource budget passes;
- complete Nsight zero-local/zero-spill evidence passes;
- three clean runs and environment fingerprints are present.

Until then, all per-kernel gates remain off by default and all shapes remain unpromoted.
