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

The module key is operation, exact batch, and selected block geometry. Device UUID, exact SM, and driver are implicit in the owning backend/runtime instance and recorded in every audit. The bounded cold-path tuner compares exactly 64, 128, and 256 threads. Plans are bounded to 4 Cholesky cells and 64 solver-family cells; modules are bounded to 12 and 192 variants respectively. Prewarm installs the deterministic 128-thread plan. Cold graph capture is rejected. A prewarmed selected module is pinned for the lifetime of the backend once captured, and every plan/module is deterministically disposed with the backend.

## Correctness matrix

The opt-in driver theory covers every operation. Required evidence before promotion includes:

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

Every result requires three independent clean runs, at least 30 warmups, 101 samples, resident preallocated buffers, CUDA-event device timing, synchronized end-to-end timing, and the complete GPU/SM/UUID/driver/runtime fingerprint.

Run AiDotNet candidates and established CUDA kernels:

```powershell
dotnet run --project tests/AiDotNet.Tensors.Benchmarks -c Release -f net10.0 -- --direct-ptx-solvers-4x4 3
```

Run the PyTorch/cuSOLVER eager and CUDA-graph competitors:

```powershell
python tests/AiDotNet.Tensors.Benchmarks/BaselineRunners/py/run_direct_ptx_solver4x4_competitors.py --runs 3
```

The output is JSON-lines plus a TUI table. Group rows by `(run, operation, batch)` before selecting the winner. FLOP rates for iterative Eigh/SVD use the fixed algorithm counts documented in the harness, not vendor peak FLOPS.

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
