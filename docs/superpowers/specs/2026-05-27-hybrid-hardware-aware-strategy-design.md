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
| `StrategyDefaultTable` (new) | `Engines/BlasManaged/Dispatcher/StrategyDefaultTable.cs` | Hardcoded `SimdClass → routing params` map (streaming work cutoff, streaming-k cutoff, special-case flags). Seeded from measured AVX2 data; AVX512 / Ryzen-AVX2 / NEON / scalar entries added as measured. Pure function, no I/O. |
| `HardwareFingerprint.SimdClass` (extend) | `Helpers/Autotune/HardwareFingerprint.cs` | Coarse SIMD-class accessor (`avx512`/`avx2`/`sse2`/`neon`/`scalar`) — the table key, independent of CPU count. |
| `PersistentStrategyCache` (extend `BlasManagedAutotune`) | `Engines/BlasManaged/Autotune/BlasManagedAutotune.cs` | Add `PackingMode` to the persisted per-(shape,fingerprint) record; `TryGetStrategy` / `StoreStrategy`. Reuses the existing `~/.aidotnet/autotune/{fp}/` JSON format and fingerprint isolation. |
| `BackgroundAutotuner` (new) | `Engines/BlasManaged/Autotune/BackgroundAutotuner.cs` | Single bounded background worker. Dequeues (shape, fp), sweeps Streaming/PackAOnly/PackBoth on freshly-allocated scratch buffers, times each (best of N), writes the winner to `PersistentStrategyCache`. Never touches caller data. |
| `SightingTracker` (new, same file) | `Engines/BlasManaged/Autotune/BackgroundAutotuner.cs` | Concurrent shape→count; gates measurement to the 2nd+ sighting so one-shot shapes never sweep. |
| Pre-warm pipeline (new) | benchmark `--prewarm-autotune` mode + shipped-resource loader | CI sweeps the catalog per fingerprint, writes cache files; package ships them; on startup, if no user cache exists for the current fingerprint, seed `PersistentStrategyCache` from the shipped resource. |

**Storage split:** the **default table is hardcoded C#** (small curated arch set, type-safe,
zero I/O); the **learned + pre-warm cache is on-disk** (reuses the existing autotune
persistence format and fingerprint directory layout).

## 5. Data flow, threading, error handling

- **Async measurement safety.** The background worker sweeps on **freshly-allocated
  same-shape buffers** (fixed seed) — never the caller's A/B/C — so a concurrent
  measurement cannot corrupt a live GEMM. One bounded worker; queue cap ~64; enqueue is a
  non-blocking try-add that drops on full (measurement is best-effort). Trigger only on the
  2nd+ sighting of a shape.
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

- `StrategyDefaultTable`: each SIMD class returns the expected route for representative
  shapes (the catalog worst-loss set).
- `PersistentStrategyCache`: store strategy → reload → hit; fingerprint isolation (entry
  for one fp not served to another).
- `BackgroundAutotuner` + `SightingTracker`: no enqueue on 1st sighting; enqueue on 2nd;
  cache populated after the worker runs; **caller C buffer is never mutated by a concurrent
  measurement**; serving thread latency unaffected (assertion).
- `SelectStrategy` precedence: learned > table; prepack → PackBoth; explicit mode honored.
- Pre-warm: a shipped cache file seeds the persistent cache on a fresh fingerprint.
- **Bit-exactness invariant**: identical output regardless of which layer chose the
  strategy (table / learned / measured), across FP32 and FP64.

## 7. Phasing (all on PR #462)

1. `StrategyDefaultTable` + `HardwareFingerprint.SimdClass` + wire `SelectStrategy` →
   replaces the static k≤128/work<1M heuristic with a per-SIMD-class table (matches MKL
   static, resolves the collision deterministically).
2. Extend `BlasManagedAutotune` to persist + consult `PackingMode` → cross-restart
   persistence.
3. `BackgroundAutotuner` + `SightingTracker` → non-blocking background learning (the
   differentiator torch.compile can't match).
4. Pre-warm pipeline → optimal-on-arrival for catalog shapes per fingerprint.

## 8. Non-goals

- Beating MKL on *tiny-shape raw GEMM throughput* — that is kernel-bound (hand-written asm),
  not a routing problem; out of scope.
- Changing the Sub-S machine-code path or the kernels themselves.
- A networked/shared tuning service — persistence is local-disk + shipped-resource only.

## 9. Risks

- **Background-thread contention** with the serving workload on a fully-loaded box. Mitigated
  by a single bounded worker, 2nd-sighting gating, and best-effort drop-on-full.
- **Table staleness** across kernel changes — the table is a *default*, superseded by
  measured cache entries, so staleness self-heals on hardware that runs measurement.
- **Cross-session collision on the dispatcher** (parallel agents on #462) — mitigated by
  landing the table behind a clean new file and keeping `SelectStrategy` edits minimal.
