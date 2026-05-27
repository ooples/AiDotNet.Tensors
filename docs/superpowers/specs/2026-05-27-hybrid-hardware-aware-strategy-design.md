# Hybrid Hardware-Aware GEMM Strategy Selection — Design

**Date:** 2026-05-27
**Branch / PR:** `perf/blas-managed-fp64-small-square-microkernel` (PR #462)
**Issue:** Part of #375 (Sub-G of mega-issue #368 — replace OpenBLAS with managed BLAS)

## 1. Problem

`Dispatcher.SelectStrategy` chooses the GEMM packing strategy (Streaming / PackAOnly /
PackBoth) with a **static, hardware-agnostic heuristic**. Two sessions independently
tuned it and collided: a `k≤128 → Streaming` rule (measured optimal on a 16-core AVX2
box) versus a `work<1M → Streaming` rule (#464, measured optimal on a Ryzen 9 3950X).
Both are correct *for their hardware* — e.g. on 512×512×64 the AVX2 box wants Streaming
(80 GFLOPS) while the Ryzen wants blocking. **No single static threshold is right for all
hardware**, so baking in either one regresses the other and overwrites a colleague's
merged tests.

The codebase already isolates *per-shape blocking params* (Mc/Nc/Kc) per hardware via
`HardwareFingerprint` + the `BlasManagedAutotune` disk cache. But the **strategy choice
itself never became hardware-aware**. That is the gap this design closes.

## 2. Goal

Make strategy selection hardware-aware in a way that **surpasses** MKL and PyTorch rather
than merely matching them:

- **MKL** ships frozen static per-arch tables — never learns the deployment's workload.
- **torch.compile** learns, but *blocks* on autotune (measured 5× worse cold-start in the
  Layer-D benchmark) and forgets on restart (in-memory only).

We beat both by combining instant static defaults with **non-blocking** background
learning, **cross-restart** persistence, and **shippable** pre-measured caches.

## 3. Architecture — four layers

`SelectStrategy` consults, in precedence order (after the existing explicit-mode and
prepack-handle guards, and *below* the Sub-S machine-code fast path which still intercepts
aligned plain GEMM in `Gemm` before strategy selection):

```
SelectStrategy(m, n, k, options):
  if options.PackingMode != Auto        → return it
  if PackedA/PackedB present            → PackBoth (consume the handle)
  fp = HardwareFingerprint.Current
  1. PersistentStrategyCache.TryGet(shape, fp)  → return   // learned OR shipped pre-warm (highest)
  2. sightings = SightingTracker.Increment(shape)
     if sightings >= 2: BackgroundAutotuner.Enqueue(shape, fp)   // async, never blocks
  3. return StrategyDefaultTable.Route(fp.SimdClass, m, n, k)     // instant cold-start default
```

Trajectory per shape: **instant-good** (table) → **silently-optimal** (async refine, no
block) → **permanently-optimal** (persisted) → **optimal-on-arrival** (shipped pre-warm).

The Sub-S machine-code path (origin #409) is unchanged and still optimal for aligned plain
GEMM; the hybrid governs the remaining shapes (transposed, epilogue-fused, unaligned, or
where the machine kernel is unavailable).

## 4. Components

| Component | File | Responsibility |
|---|---|---|
| `StrategyDefaultTable` (new) | `Engines/BlasManaged/Dispatcher/StrategyDefaultTable.cs` | Hardcoded `(simd, vendor, cpuBucket) → routing params` map (streaming work cutoff, streaming-k cutoff, special-case flags). **Keyed on `{simd, vendor, cpuBucket}`, NOT SIMD class alone — see G1.** Seeded from measured AVX2-Intel data; AVX2-AMD (Ryzen), AVX512, NEON, scalar entries added as measured. Unknown keys fall back to the nearest coarser entry (same simd, any vendor) then a conservative scalar default. Pure function, no I/O. |
| `HardwareFingerprint` (extend) | `Helpers/Autotune/HardwareFingerprint.cs` | Add `SimdClass`, `Vendor`, and `CpuBucket` accessors (cpuBucket = small/medium/large core-count bands, e.g. `≤4 / 5–16 / >16`) — together the table key. The full per-machine fingerprint string still keys the on-disk cache. |
| `PersistentStrategyCache` (extend `BlasManagedAutotune`) | `Engines/BlasManaged/Autotune/BlasManagedAutotune.cs` | Add `PackingMode` to the persisted per-(shape,fingerprint) record; `TryGetStrategy` / `StoreStrategy`. **Cache key includes a `KernelVersion` token (G2)** so a kernel/build change invalidates stale tunings instead of serving a now-suboptimal strategy forever. Reuses `~/.aidotnet/autotune/{fp}/` and adds the version to the record (mismatched-version entries are ignored on read and overwritten on next measure). |
| `BackgroundAutotuner` (new) | `Engines/BlasManaged/Autotune/BackgroundAutotuner.cs` | Single bounded background worker at **below-normal thread priority (G3)**. Dequeues (shape, fp), sweeps Streaming/PackAOnly/PackBoth on freshly-allocated scratch buffers **in the serving path's current `BlasMode` (G5)**, times each (best of N), writes the winner + KernelVersion to `PersistentStrategyCache`. **Skips shapes above a work ceiling (G3)** — large shapes already route well via the table and their sweep is too expensive to run under load. Never touches caller data. De-dups via an in-flight set (G4). |
| `SightingTracker` (new, same file) | `Engines/BlasManaged/Autotune/BackgroundAutotuner.cs` | **Bounded LRU** (cap ~4096 shapes) shape→count (G4) — no unbounded growth on dynamic-shape workloads. Gates measurement to the 2nd+ sighting so one-shot shapes never sweep, and holds the in-flight de-dup set so concurrent first-callers enqueue a shape once. |
| Pre-warm pipeline (new) | benchmark `--prewarm-autotune` mode + shipped-resource loader | CI sweeps the catalog per fingerprint, writes cache files tagged with the current `KernelVersion`; package ships them; on startup, if no *version-matching* user cache entry exists for the current fingerprint, seed `PersistentStrategyCache` from the shipped resource. Shipped entries seed only when absent locally (local learned entries win). |

**Storage split:** the **default table is hardcoded C#** (small curated `{simd,vendor,cpuBucket}`
set, type-safe, zero I/O); the **learned + pre-warm cache is on-disk** (reuses the existing
autotune persistence format and fingerprint directory layout, plus the new KernelVersion tag).

**KernelVersion (G2):** a build-stamped constant (e.g. assembly-info hash or a manually-bumped
`BlasKernelVersion` int incremented whenever a microkernel/strategy/blocking change lands).
Both the on-disk learned cache and the shipped pre-warm carry it; reads ignore mismatches.

## 5. Data flow, threading, error handling

- **Async measurement safety.** The background worker sweeps on **freshly-allocated
  same-shape buffers** (fixed seed) — never the caller's A/B/C — so a concurrent
  measurement cannot corrupt a live GEMM. One bounded worker; queue cap ~64; enqueue is a
  non-blocking try-add that drops on full (measurement is best-effort). Trigger only on the
  2nd+ sighting of a shape, de-duplicated via an in-flight set.
- **Contention control (G3).** The worker runs at `ThreadPriority.BelowNormal`. It
  **skips shapes whose work (M·N·K) exceeds a ceiling** (default ~8M) — large shapes
  already route well from the table and a multi-strategy sweep of them under load is exactly
  the latency-stealing case we must avoid; their tuning, if wanted, comes from the offline
  pre-warm pipeline, not live background measurement. The worker also yields between
  candidates so a busy box deprioritises tuning behind real work.
- **Measurement fidelity (G5).** The sweep runs in the **same `BlasMode`** (deterministic
  vs fast) the serving path is using, so the measured winner reflects the real parallel
  path. Isolated cache state is an accepted approximation — strategy choice is coarse and
  cache-residency effects are second-order for it; this is documented, not hidden.
- **Correctness invariant.** Every strategy the hybrid can pick produces **bit-identical**
  results (they are the same kernels) — table vs learned vs measured routing changes only
  speed, never output. This is the central safety property and a required test.
- **Error handling.** Measurement exception → caught, shape skipped, table default stands.
  Cache I/O failure → non-fatal (existing behavior). Fingerprint-detection failure →
  conservative scalar table entry. Missing pre-warm resource → silently fall through to
  live behavior.
- **Sub-S interaction.** Unchanged. Sub-S precedes strategy selection for aligned plain
  GEMM; the hybrid optimizes the rest.
- **Opt-out.** A process-global switch (default ON for the table + persistence; background
  measurement default ON but bounded) lets a deployment force pure-static behavior if
  desired. Background measurement honors the existing deterministic-mode constraints (it
  never alters the serving path's reduction order).

## 6. Testing

- `StrategyDefaultTable`: each `{simd,vendor,cpuBucket}` key returns the expected route for
  representative shapes; **distinct entries for AVX2-Intel vs AVX2-AMD (G1 regression test)**;
  unknown-key fallback chain (exact → same-simd → scalar).
- `PersistentStrategyCache`: store strategy → reload → hit; fingerprint isolation; **a
  KernelVersion mismatch is ignored on read (G2 regression test)**.
- `BackgroundAutotuner` + `SightingTracker`: no enqueue on 1st sighting; enqueue on 2nd;
  **concurrent first-callers enqueue once (dedup, G4)**; **large shape above the work ceiling
  is never enqueued (G3)**; **LRU evicts beyond cap (G4)**; cache populated after the worker
  runs; **caller C buffer is never mutated by a concurrent measurement**; serving thread
  latency unaffected (assertion).
- `SelectStrategy` precedence: learned > table; prepack → PackBoth; explicit mode honored.
- Pre-warm: a shipped cache file seeds the persistent cache on a fresh fingerprint; a
  version-mismatched shipped file is not seeded.
- **Bit-exactness invariant**: identical output regardless of which layer chose the
  strategy (table / learned / measured), across FP32 and FP64.

## 7. Phasing (all on PR #462)

1. **Lever check + `StrategyDefaultTable` + `{simd,vendor,cpuBucket}` fingerprint keys +
   wire `SelectStrategy`.** Begin with a quick instrumented measurement of how much
   real-workload GEMM bypasses Sub-S and reaches strategy selection (G6) — this both
   validates the investment and identifies which shapes the table must cover. Then replace
   the static k≤128/work<1M heuristic with the per-`{simd,vendor,cpuBucket}` table (matches
   MKL static, resolves the collision deterministically — distinct Intel/AMD entries).
2. Extend `BlasManagedAutotune` to persist + consult `PackingMode`, with the `KernelVersion`
   tag → cross-restart persistence that self-invalidates on kernel changes (G2).
3. `BackgroundAutotuner` + bounded-LRU `SightingTracker` → non-blocking, contention-controlled,
   mode-matched background learning (the differentiator torch.compile can't match), with the
   G3/G4/G5 hardening above.
4. Pre-warm pipeline → optimal-on-arrival for catalog shapes per fingerprint (version-tagged).

## 8. Non-goals

- Beating MKL on *tiny-shape raw GEMM throughput* — that is kernel-bound (hand-written asm),
  not a routing problem; out of scope.
- Changing the Sub-S machine-code path or the kernels themselves.
- A networked/shared tuning service — persistence is local-disk + shipped-resource only.

## 9. Risks (post-hardening)

- **Background-thread contention** (G3) — mitigated by below-normal priority, a work-ceiling
  skip for large shapes, single bounded worker, 2nd-sighting + dedup gating, and
  drop-on-full. Residual risk on a 100%-pinned box is bounded to small-shape sweeps at low
  priority; offline pre-warm covers large shapes instead.
- **Learned-cache staleness** across kernel changes (G2) — the `KernelVersion` tag
  invalidates mismatched entries on read, so a kernel change can't serve a stale strategy.
- **Default-table staleness** — the hardcoded table is a *default*, superseded by measured
  cache entries; it self-heals on hardware that runs measurement, and a wrong table entry
  only costs a suboptimal-but-correct strategy until measurement overrides it.
- **Lever size** (G6) — strategy selection is a smaller lever than kernel quality, and Sub-S
  already handles aligned plain GEMM. Phase 1's lever check quantifies the real win before
  Phases 3–4 invest; if the bypass-coverage is small, Phases 3–4 can be deferred.
- **Cross-session collision on the dispatcher** (parallel agents on #462) — mitigated by
  landing the table behind a clean new file and keeping `SelectStrategy` edits minimal.
- **Dynamic-shape memory** (G4) — bounded-LRU `SightingTracker` caps growth.
