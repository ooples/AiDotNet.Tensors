# Custom JIT GEMM microkernel for AiDotNet.Tensors (managed Xbyak): match MKL/oneDNN GFLOP/s on our own thread pool

## Summary / motivation

On an AVX2-only rig (no AVX-512), PyTorch-CPU beats AiDotNet on the small-K/N
transformer/FFN GEMM shapes. Root-caused via per-op profiling:

- The managed `SimdGemm` kernel hits **70–113 GFLOP/s** on the small-K/N shapes
  (QKV `[4096,64]×[64,192]`, FFN `[..,64]×[64,128]`, `[..,128]×[128,64]`) vs
  **~170** on large square — RyuJIT can't schedule the 12-accumulator 6×16 tile
  without spilling, and can't software-pipeline.
- oneDNN's `dnnl_sgemm` (already shipped as `dnnl.dll`) hits **543 GFLOP/s** on
  those exact shapes (JIT'd, shape-specialized brgemm) — **but regresses
  end-to-end** (6/16 → 3/16 shapes vs PyTorch) because it brings its **own
  OpenMP thread pool that oversubscribes** AiDotNet's `PersistentParallelExecutor`
  (CNN bs128 +104%; confirmed: 16-thread 10.5 ms → 1-thread 5.9 ms).

**Conclusion:** the GFLOPS are achievable on this rig, but a bolted-on native
library can't deliver them end-to-end. We need **our own JIT'd GEMM microkernel**
driven by AiDotNet's existing single thread pool.

## Proof-of-concept results (validated, this rig, AVX2)

| Milestone | Result |
|---|---|
| Emit raw AVX2 machine code in C#, run via `delegate* unmanaged`, correct | ✅ (maxErr ~1e-6) |
| Hand-emitted 6×16 compute-bound microkernel vs RyuJIT, single-core | **114 vs 88 GFLOP/s (1.3×)** |
| Same kernel fanned out across our own pool (1→16 threads) | **517 GFLOP/s @16t** (matches oneDNN's 543, **no oversubscription**) |
| Naive production driver on real small-K shapes | 28–42 GFLOP/s — **pack/scatter-bound**, not kernel-bound |

PoC lives at `C:\Users\yolan\asmjit` (encoder + 6×16 kernel + scaling + driver).

## Mechanism (confirmed working)

- `VirtualAlloc(PAGE_READWRITE)` → write code bytes → `VirtualProtect(PAGE_EXECUTE_READ)` (W^X-clean).
- Call via `delegate* unmanaged[Cdecl]<float*,float*,float*,long,void>`.
- Minimal AVX2 VEX encoder (2-/3-byte VEX): `vxorps`, `vmovups`, `vbroadcastss`,
  `vfmadd231ps`, `vmovups` store, `add`/`dec`/`jnz`, `vzeroupper`, `ret`.
- **Gotcha proven in testing:** disp8/imm8 sign-extension — any offset >127 must
  use disp32/imm32 (a `sub rsp,160` silently became `add rsp,96` → AV). The
  verification harness must catch this class of bug.

## Work items (Phase 2 → production)

### 2a. ldc-aware direct-write kernel
Pass row-stride (`ldc`) so the kernel stores straight into `C[M,N]` — eliminate
the scalar scatter that dominates small-K in the naive driver. (5th arg on the
Win64 stack; variable-stride store addressing.)

### 2b. Packing strategy — MEASURED: small-K needs an unpacked kernel
Driver K-sweep on `[4092,*]×[*,192]` (16t, pack+compute): K=64 → 47, K=256 → 139,
K=512 → 214 GFLOP/s. Even K=512 (214) trails the pre-packed scaling test (527),
so **per-call A-packing is itself a dominant cost** at these shapes. The managed
`DirectKernel6x16` beats the packed JIT driver on small-K (70–113 vs 47)
*because it's unpacked*. ⇒ Phase 2 needs **two JIT kernels dispatched by K**:
- **unpacked stride-reading** (reads A via `lda`, B via `ldb`, writes C via `ldc`;
  no pack, no scatter) for small-K (transformer/FFN/QKV) — the hot case.
- **packed** (current 6×16) for large-K, where packing amortizes (527 GFLOP/s).
Unpacked kernel challenge: 6 strided A-row broadcasts/k under tight GP-register
budget (hoist 6 row pointers or use index-register addressing).

### 2b-2. Per-call prologue amortization (macro-kernel)
The 10× XMM6–15 save/restore runs **per tile**. Emit a **macro-kernel** that
loops all N/16 col-blocks (and ideally the M row-blocks) in one call so the
prologue is paid once per row-panel, not once per 6×16 tile (~12× fewer
prologues on the QKV shape). Confirmed relevant: small-K amortizes the prologue
over only K iterations.

### 2c. Masked edge kernels
M%6 and N%16 tails — emit masked-load/store variants (`vmaskmovps`) or a
generic M×N masked kernel. Required for arbitrary shapes (4096 not divisible by 6).

### 2d. Software pipelining + prefetch
Issue next-iteration B-loads ahead of the current FMAs; `prefetcht0` the next
A/B panel. This is where per-core goes from 114 toward oneDNN's per-core rate.

### 2e. Blocking (Mc/Nc/Kc) + driver
Reuse the existing `SgemmTiled` blocking framework; call the JIT kernel as the
microkernel. Keep B-panel in L1 (Kc tuned to L1d).

### 2f. Thread-pool integration (THE thing that sank oneDNN)
Drive parallel tiles via AiDotNet's `PersistentParallelExecutor` — one pool, no
oversubscription. This is the architectural reason to own the kernel.

### 2g. Shape→kernel JIT cache
JIT once per (M-tail, N-tail, K, ISA) signature; cache the fn-ptr. Weights have
fixed shapes, so inference JITs once and reuses forever. Bound the cache.

### 3. Breadth (later)
- **AVX-512** encoder (EVEX) — 16-wide, where present (this rig lacks it; most servers have it).
- **ARM64 NEON/SVE** encoder — Apple/Graviton.
- **Epilogue fusion** — bias + activation + residual folded into the store phase.
  This is the thing `dnnl_sgemm` *can't* do (pure GEMM), attacking the GEMM AND
  the allocation problem in one pass.

## Cross-cutting requirements

- **Verification harness:** disassemble emitted bytes (capstone/ref) + diff vs a
  scalar reference for every JIT'd shape; fuzz offsets to catch disp/imm overflow.
- **W^X / cross-platform exec memory:** Windows `VirtualAlloc`/`VirtualProtect`;
  Linux `mmap`/`mprotect`; macOS `MAP_JIT` + `pthread_jit_write_protect_np`.
- **Fallback:** keep the managed `SimdGemm` kernel as the portable path
  (non-x64, or if exec-memory allocation is denied by policy).
- **Supply-chain independence:** this removes the `dnnl.dll`/MKL dependency for
  the hot GEMM path entirely.

## Open questions (flagged: plan may not cover everything)

- Is the goal to fully replace `SimdGemm`, or only intercept the small-K/N shapes
  where it loses (keep managed for large square where it's already ~competitive)?
- Threadpool ownership: does the JIT kernel call `PersistentParallelExecutor`, or
  does the existing `SgemmTiled` driver call the JIT microkernel per tile?
- AOT/trimming/`delegate* unmanaged` interplay; antivirus/EDR reaction to RWX→RX
  pages in a managed process (real-world deployment concern).
- Numerical reproducibility vs the managed/native paths (reduction order).
