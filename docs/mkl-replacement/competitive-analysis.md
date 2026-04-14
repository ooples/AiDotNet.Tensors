# Competitive GEMM Analysis: Closing the MKL Square-Shape Gap

Target: AiDotNet.Tensors `SgemmTiled` + `MicroKernel6x16` (12 YMM accumulators, Mc=128, Nc=4096, Kc=512, Mr=6, Nr=16). Trails MKL 1.35-1.64x at 512²/1024², beats MKL by 17% at LM-head shapes. Goal: identify specific techniques MKL uses internally (extracted from open-source kernels known to encode the same techniques).

---

## 1. BLIS (`bli_sgemm_haswell_asm_6x16`)

- **Shape**: **6×16** — identical to ours. Chosen because 6×16/8 = 12 YMM accumulators fits the 16-reg budget with 4 scratch, and Mr=6 lets one A broadcast feed two B-vector FMAs, so FMA throughput (2/cycle on Haswell) matches issue width.
- **Packing**: A is packed as 6-row panels, contiguous along K (`add imm(4*6*4), rax` — 24 bytes/step). B is packed as 16-col panels of two adjacent YMMs (`add imm(4*16*4), rbx`). Both packed **K-major** so the microkernel walks contiguously.
- **Registers**: ymm4–ymm15 = 12 C accumulators. ymm0–ymm1 = preloaded B rows. ymm2–ymm3 = A broadcasts (ping-ponged so the next broadcast issues while current FMAs retire).
- **Prefetch**: Single `prefetch(0, mem(rax, 64*4))` per 4-way unrolled iter — **256 B ahead on A only**. No B prefetch (B already hot from packing). No `prefetcht1` for next panel. C is prefetched **once before the loop**, not inside.
- **JIT**: None (static asm). Kc is a runtime loop, not an immediate.
- **NT stores**: **No.** `vmovups` only. C is assumed reused by the caller (beta=1 case).
- **Unroll**: **4-way K unroll**, 12 FMAs per K-iter (6 A-broadcasts × 2 B-rows).

## 2. libxsmm (Intel engineer's JIT)

- **Shape**: Fully variable — emits per (M, N) combo. Common: M-blocking 1/4/vector-multiples, N-blocking 1-3. Accumulator reg index computed as `16 - (n_blocking * m_blocking)`.
- **Packing**: A column-major contiguous loads (vector_length strided); B single-element + `vbroadcastss`.
- **Prefetch**: `PREFETCHT0` emitted **every 8 K-iters** at distance `l_pf_dist = 16` (~1 KB). Distance is tunable at JIT time per cache geometry.
- **JIT specialization** (the MKL "secret sauce"): **K-loop is fully unrolled as straight-line code when K is small**. No branch, no loop counter. Separate JIT paths per `m_blocking` (1 vs 4 vs generic); register allocation differs per path. B offsets become **immediates** (`l_b_offset` literal).
- **NT stores**: No — stores are outside the microkernel.
- **This is what we lack most**: size-specialized, K-fully-unrolled code per hot shape.

## 3. OpenBLAS (`sgemm_kernel_16x4_haswell.S`)

- **Shape**: **16×4** — inverted aspect vs ours. Rationale: 16 rows of A fit in 2 YMM loads (`vmovups`), and 4 B columns need only 4 broadcasts. 8 YMM accumulators vs our 12 — frees registers for **deeper prefetch scheduling and software pipelining**.
- **Packing**: A contiguous 16-float rows. B is repacked into BUFFER1 with 4-element grouping for `vbroadcastss` efficiency.
- **Prefetch**: `prefetcht0 A_PR1(AO,...)` and `prefetcht0 B_PR1(BO,...)` with **A_PR1 = B_PR1 = 512 B**. Prefetch is issued **both** on A and B every 4 iterations in the unrolled body — denser than BLIS.
- **Registers**: ymm4–ymm11 (8 accs), ymm0–ymm1 A loads, ymm2–ymm3 B broadcasts.
- **Unroll**: **8-way K unroll** (vs our 4). `andq $-8, %rax` gates the fast path.
- **NT stores**: No.

## 4. oneDNN (`jit_avx2_kernel_sgemm_kern.cpp`, MKL-DNN descendant)

- **Shape**: JIT-variable; recursive halving of `unroll_x`/`unroll_y` (8→4→2→1).
- **Prefetch**: Uses **three tiers**: `prefetcht0` (B, L1), `prefetcht2` (A next panel, L3), and **`prefetchw`** (write intent on C). `prefetcht2` on `AA_` (the *next* A panel) is the critical missing piece — it keeps the next Mc×Kc block hot in L2/L3 while the current one streams.
- **Registers**: Acc start at `zmm_acc_idx_`, zero-init with `vpxorq`.
- **JIT specialization**: Nested switches on unroll factors generate distinct code paths per size.
- **NT stores**: No.

---

## Recommendations (Ranked by Expected Impact)

1. **Emit K-fully-unrolled microkernel via IL/System.Reflection.Emit for hot Kc values** (e.g., Kc∈{256, 512}). libxsmm's killer feature — removes the K-loop branch entirely, lets the CPU front-end schedule 2000+ FMAs as one basic block. Projected: 10-20% at square shapes where our Kc=512 loop overhead dominates.
2. **Add `prefetcht2` on the next A panel (`AA_`-style) inside the Mc loop.** oneDNN does this; BLIS does not; MKL does. At 1024² we spill to L3 on every Mc-step — a single `Prefetch0/Prefetch1` on the next Ac block ~256 B ahead will hide the reload. Use `Sse.Prefetch1` (t1) or `Sse.Prefetch2` (t2) in C# — supported since .NET Core 3.0.
3. **Increase K unroll from 4 to 8** (OpenBLAS ratio). Our 4-way unroll generates a loop-closing branch every 48 FMAs; 8-way doubles that to 96. Costs 0 registers (FMA targets are the same C accumulators).
4. **Add `prefetcht0` on B inside the inner loop**, not just A. BLIS skips B prefetch because B is hot from packing — but our packed B may not be (heap-allocated, maybe not cache-aligned). A single `Sse.Prefetch0(bPtr + 512)` per 4-iter costs 1 µop.
5. **Prefetch C at `prefetchw` intent** before the store-back phase. `Sse.PrefetchNonTemporal` won't help (we want C in cache for β=1); instead use the RFO hint via a dummy write or the 4.5+ `Prefetch1` on the C rows once per Mr×Nr tile.
6. **Ping-pong A broadcast registers (ymm2/ymm3) explicitly**. BLIS alternates them so the next `vbroadcastss` is issued before the previous's FMA retires. Confirm your C# JIT (RyuJIT) actually schedules the `Avx.BroadcastScalarToVector256` calls this way — if it serializes them, hand-unroll in source.
7. **Align packed A/B buffers to 64 B (cache-line) and prefer aligned `vmovaps` over `vmovups`** for the B load. BLIS uses `vmovaps` (aligned); we likely use `vmovups`. On Haswell the penalty is small, but on Zen2 (Ryzen 9 3950X, your benchmark rig) misaligned loads cost 1 extra cycle per cache-line crossing.

**Non-recommendations** (explicitly ruled out by all four libraries):
- Non-temporal stores for C — no one uses them for GEMM; C is always reused.
- Switching to 16×4 shape — 6×16 is optimal for Haswell/Zen2 FMA:broadcast ratio; OpenBLAS's 16×4 is a different register-budget tradeoff, not faster.
- `prefetcht1` on current stream — wastes the L1 hint.
