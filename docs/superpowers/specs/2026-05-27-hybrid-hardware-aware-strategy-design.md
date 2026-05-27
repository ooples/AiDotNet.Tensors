# Hybrid Hardware-Aware GEMM Strategy Selection — Design

**Date:** 2026-05-27
**Branch / PR:** `perf/blas-managed-fp64-small-square-microkernel` (PR #462)
**Issue:** Part of #375 (Sub-G of mega-issue #368 — replace OpenBLAS with managed BLAS)

## 1. Problem

`Dispatcher.SelectStrategy` chooses the GEMM packing strategy (Streaming / PackAOnly /
PackBoth) with a **static, hardware-agnostic heuristic**. Two sessions independently
tuned it on **different hardware** and collided: a `k≤128 → Streaming` rule (measured on
`x64-amd-avx2-cpu16`) versus a `work<1M → Streaming` rule (#464, measured on a Ryzen 9
3950X = `x64-amd-avx2-cpu32`). Both are correct *for their machine* — e.g. on 512×512×64
the 16-thread box wants Streaming (80 GFLOPS) while the 32-thread Ryzen wants blocking.
**No single static threshold is right for all hardware**, so baking in either one regresses
the other and overwrites a colleague's merged tests.

The deeper point — and the actual goal — is that this is not a two-machine problem to
hand-resolve; it is the general need to **adapt the strategy to whatever hardware the
library runs on**. Crucially, the measurement/learning layers below make the *source* of
any divergence (core count, microarchitecture, even measurement methodology) irrelevant:
the system measures ground truth on each real machine with the current kernels. The shipped
default table is only a cold-start *seed*; the learned per-fingerprint cache is the truth
for that box. So we never need to pre-decide whether a given divergence is "really
hardware" — the empirical layer settles it per machine.

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
| `StrategyDefaultTable` (new) | `Engines/BlasManaged/Dispatcher/StrategyDefaultTable.cs` | Hardcoded `(simd, vendor, cpuBucket, shapeBucket) → strategy` map — the *seed tier* of the unified map (G9), returning a strategy per shape-bucket, NOT a separate threshold model. **Keyed on `{simd, vendor, cpuBucket}`, NOT SIMD class alone — see G1.** Seeded from measured `amd-avx2-cpu16` data; `amd-avx2-cpu32` (Ryzen), AVX512, NEON, scalar entries added as measured. Unknown keys fall back to the nearest coarser entry (same simd, any vendor/cpu) then a conservative scalar default. Pure function, no I/O. |
| `HardwareFingerprint` (extend) | `Helpers/Autotune/HardwareFingerprint.cs` | Add `SimdClass`, `Vendor`, and `CpuBucket` accessors (cpuBucket = small/medium/large core-count bands, e.g. `≤4 / 5–16 / >16`) — together the table key. The full per-machine fingerprint string still keys the on-disk cache. |
| `PersistentStrategyCache` (extend `BlasManagedAutotune`) | `Engines/BlasManaged/Autotune/BlasManagedAutotune.cs` | Add `PackingMode` to the persisted per-(shape,fingerprint) record; `TryGetStrategy` / `StoreStrategy`. **Cache key includes a `KernelVersion` token (G2)** so a kernel/build change invalidates stale tunings instead of serving a now-suboptimal strategy forever. Reuses `~/.aidotnet/autotune/{fp}/` and adds the version to the record (mismatched-version entries are ignored on read and overwritten on next measure). |
| `BackgroundAutotuner` (new) | `Engines/BlasManaged/Autotune/BackgroundAutotuner.cs` | Single bounded background worker at **below-normal thread priority (G3)**. Dequeues (shape, fp), sweeps Streaming/PackAOnly/PackBoth on freshly-allocated scratch buffers **in the serving path's current `BlasMode` (G5)**, times each (best of N), writes the winner + KernelVersion to `PersistentStrategyCache`. **Skips shapes above a work ceiling (G3)** — large shapes already route well via the table and their sweep is too expensive to run under load. Never touches caller data. De-dups via an in-flight set (G4). |
| `SightingTracker` (new, same file) | `Engines/BlasManaged/Autotune/BackgroundAutotuner.cs` | **Bounded LRU** (cap ~4096 shapes) shape→count (G4) — no unbounded growth on dynamic-shape workloads. Gates measurement to the 2nd+ sighting so one-shot shapes never sweep, and holds the in-flight de-dup set so concurrent first-callers enqueue a shape once. |
| Pre-warm pipeline (new) | benchmark `--prewarm-autotune` mode + shipped-resource loader | CI sweeps the catalog per fingerprint, writes cache files tagged with the current `KernelVersion`; package ships them; on startup, if no *version-matching* user cache entry exists for the current fingerprint, seed `PersistentStrategyCache` from the shipped resource. Shipped entries seed only when absent locally (local learned entries win). |

**Storage split:** the **default table is hardcoded C#** (small curated `{simd,vendor,cpuBucket}`
set, type-safe, zero I/O); the **learned + pre-warm cache is on-disk** (reuses the existing
autotune persistence format and fingerprint directory layout, plus the new KernelVersion tag).

**KernelVersion (G2, G8):** an **auto-derived** token — a content hash over the kernel-relevant
sources (the `Strategies/`, `Microkernels/`, `Jit/`, and dispatcher blocking-param files),
combined into a short stamp at build time (e.g. a source-generator or a checked-in hash
updated by a build step). **Not a manually-bumped int** — that would be forgotten on a kernel
change and silently serve a stale tuning (the G8 footgun). Both the on-disk learned cache and
the shipped pre-warm carry it; reads ignore mismatches and the next measurement overwrites.

**Unified representation (G9):** there is ONE concept — `(hardwareKey, shapeBucket) → (strategy,
blocking)`. The default table is the *shipped/seed tier* of that map (coarse, hardcoded); the
disk cache is the *learned tier* (exact shape, per fingerprint). The table does NOT expose
thresholds as a separate model — it returns a strategy for a shape bucket, exactly like the
cache, so both tiers are queried and reasoned about identically.

**Strategy + blocking are one unit (G11):** the existing `BlasManagedAutotune` record already
holds Mc/Nc/Kc/Axis/ThreadCount; `PackingMode` joins it as part of the *same* tuned tuple. The
background sweep measures the strategy **and** re-tunes its blocking together, then stores them
atomically — never a new strategy paired with blocking tuned for the old one.

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
- **Hot-path lookup cost (G13).** The lookup chain (fingerprint → cache → table) must not
  re-introduce the per-call overhead the Streaming short-circuit just removed. The
  fingerprint key is computed once and cached; the cache + table are single dictionary
  hits on a struct key. The chain only runs on the strategy-selection path — the tiny-shape
  and Sub-S fast paths return *before* it, so the smallest/hottest shapes pay nothing. A
  micro-benchmark guards that `SelectStrategy` stays sub-microsecond.
- **Multi-process write safety (G10).** The background writer increases write frequency to
  the shared `~/.aidotnet/autotune/{fp}/` dir. Writes are atomic (write-temp + rename);
  reads tolerate a partially-written/corrupt file (treat as miss). Concurrent processes may
  redundantly measure the same shape — wasteful but correct (last atomic write wins).
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
- **Anti-regression guard (G12)**: a perf test asserts the table's routing is **no worse
  than the current (fixed) static heuristic** across the catalog on the local fingerprint —
  so a hand-seeded table entry can't ship a regression. "Expected route" assertions are
  paired with this empirical check rather than standing alone (which would be circular).
- **Hot-path micro-bench (G13)**: `SelectStrategy` median latency stays sub-µs with the
  lookup chain active; tiny/Sub-S shapes confirmed to bypass it.

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

## 10. As-built (2026-05-27, PR #462)

Implemented in 4 phases on `perf/blas-managed-fp64-small-square-microkernel`. Deltas from
the design above:

- **Trans-aware throughout (refinement).** The lever-check (Task 1.0) confirmed only ~1/6
  catalog shapes reach strategy selection — Sub-S handles non-transposed aligned GEMM, so
  the shapes the hybrid governs are *transposed*. Accordingly `SightingTracker.ShapeId`,
  `BackgroundAutotuner.Observe`, the measurement, and the learned-cache key all carry
  `transA/transB`, and `SelectStrategy` gained a trans-aware overload. Without this the
  learned key would never match real (transposed) calls.
- **`#464` dispatch tests converted to hardware-aware.** `SmallShapeStreamingDispatchTests`
  asserted universal routes (incompatible with per-hardware routing); the hardware-dependent
  cases now assert the table per explicit `HwKey` (cpu16→Streaming, cpu32→blocking). The
  `MediumSquare` bucket requires a true cube (m==n==k) so thin-K 512×512×64 isn't conflated
  with 128³.
- **`BackgroundAutotuner.Enabled` switch.** Default ON (production). The test assembly
  disables it at load (`[ModuleInitializer]`) so its concurrent measurement load can't
  perturb timing-sensitive tests; the worker-exercising test re-enables locally inside the
  serialized collection.
- **KernelVersion (G8) interim.** `assembly-version + manual KernelEpoch` rather than a full
  source content-hash (deferred to a source-generator). Combining with the assembly version
  bounds the staleness window to a single dev build.
- **net471.** `BlockingCollection`/`ThreadPriority`/`Thread` are available there; the worker
  compiles and runs on both TFMs (no no-op needed).
- **Pre-warm dir ships empty.** The `*.prewarm.json` glob matches nothing until CI runs
  `--prewarm-autotune` per arch and commits trustworthy sweeps; dev-box single-run output is
  intentionally not committed.
- **De-flake (pre-existing).** Serialized the reduction-order mutators (`DeterministicModeTests`,
  `ParallelForOrSerialDispatchTests`) and bit-exact victims (`PartialMCorrectnessTest`,
  `Avx2PackTransBFp32Test`, `SmallShapeStreamingDispatchTests`) into `BlasManaged-Stats-Serial`;
  tagged the unguarded `Fp64x12` GFLOPS gate `[Trait Performance]`.

**Verification:** 493 BlasManaged/ScalarKernel tests pass ×3 (no flakes); both TFMs build;
head-to-head shows no regression on the Sub-S-handled plain shapes (the hybrid governs the
transposed shapes outside that catalog).

## 11. Post-verification corrections (the as-built §10 claimed clean completion prematurely)

A follow-up audit found two real defects that the originally-skipped G12/G13 guards would
have caught at the time:

- **G13 — hot-path disk-read regression (severe).** `SelectStrategy`'s learned-cache consult
  called `AutotuneCache.Lookup`, which does `File.Exists + ReadAllText + JSON parse` *per
  call* (it is not memory-cached). Measured **77 µs/call** — roughly doubling every
  transposed GEMM's dispatch cost. Fixed with an in-memory memo in `BlasManagedAutotune`
  (`_strategyMemo`): disk touched at most once per distinct shape; `StoreStrategy` refreshes
  it (version-gated). Re-measured **757 ns/call** (100× faster, sub-µs). Committed G13 latency
  guard (`HybridStrategyPerfTests`, default-on, 10 µs gate).
- **G12 — table mis-calibration.** The seed table was calibrated from *non-transposed* A/B
  data, but it is consulted *only* for transposed shapes (Sub-S handles non-transposed). So
  transposed 512×512×64 routed to Streaming = **2.5× slower** than PackBoth. Recalibrated the
  cpu16 `ThinK` entry to PackBoth (the transposed optimum from `--prewarm-autotune`); updated
  the dispatch tests. The learned/prewarm layer already measured this correctly, so it
  self-heals — only the static cold-start seed was wrong. The G12 wall-clock guard is env-gated
  (`AIDOTNET_RUN_HYBRID_PERF=1`) as it is contention-sensitive; routing correctness is
  guarded deterministically by `SmallShapeStreamingDispatchTests`.

**Remaining known limitations (disclosed, not yet addressed):**
- `_strategyMemo` is unbounded (bounded in practice by a workload's distinct-shape count;
  the on-disk cache is likewise per-shape unbounded).
- The earlier-session `EnableAutotuneV2` (Gemm-level, opt-in, default off) coexists with this
  hybrid (SelectStrategy-level) — redundant; a future consolidation should pick one.
- The hybrid's end-to-end *speedup* on real transposed workloads is **not yet demonstrated** —
  only routing correctness, no-regression on the Sub-S path, and sub-µs dispatch are verified.
  The lever is bounded (~1/6 catalog shapes reach strategy selection).

## 12. Limitations addressed + win demonstrated (follow-up)

- **Memo unbounded → fixed.** `_strategyMemo` now caps at 8192 entries (clear-on-overflow);
  only trips on an adversarial dynamic-shape stream.
- **EnableAutotuneV2 redundancy → removed.** The earlier opt-in Gemm-level autotune
  (`EnableAutotuneV2` + `AutotuneCacheV2` + synchronous warmup sweep) was deleted; the hybrid
  (per-fingerprint, disk-persistent, non-blocking, default-on) supersedes it.
- **Hybrid win → demonstrated** (`--hybrid-win`, 6 transposed shapes, 16-core AVX2):
  always-Streaming 365 ms, always-PackBoth 122 ms, **hybrid (per-shape) 110 ms**. The hybrid is
  **3.3× vs always-Streaming** (it never makes the catastrophic 512×512×64-transB Streaming
  choice) and **1.11× vs the best single fixed strategy** (it picks Streaming for small shapes
  and PackBoth for ThinK). Honest: the edge over the best fixed policy is modest; the dominant
  value is avoiding the wrong static choice a hardware-agnostic heuristic would make on some
  shape/hardware.
