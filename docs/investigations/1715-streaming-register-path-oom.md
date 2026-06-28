# #1715 — Streaming pool: GetParameters/register path doesn't bound resident set → foundation-scale OOM

## Status
**Investigation complete; code fix pending verification.** Root cause traced end-to-end through three
layers. This doc is the hand-off so the fix can be implemented + verified with the (slow) two-repo loop.

## Symptom (AiDotNet side)
`UnitTests.Diffusion.Models` contract tests OOM under the 16 GB CI runner (issue #1715, shard
`Unit - 03d`):
- `Flux2Model_GetSetParameters_RoundTrips`
- `FluxDoubleStreamPredictor_GetSetParameters_RoundTrips`
- `SiTPredictor_GetSetParameters_RoundTrips`

Measured (Flux2): **~6.5 billion params (~25 GB fp32)**; lazy ctor = 712 MB; enumerating
`GetParameterChunks` OOMs. Peak working set stays ~6.9 GB (and drops to ~12 MB mid-run) yet it OOMs →
**committed memory accumulates, not live working set**.

## Root cause (Tensors side) — the three layers
The AiDotNet `GetParameterChunks` path materializes every layer's lazy weights. Engaging weight
streaming there (AiDotNet-side wiring, see below) moves the allocation from `RentPinned` to
`WeightRegistry.AllocateStreaming`, and registering each materialized weight reaches
`WeightRegistry.RegisterWeight`. The residual OOM is here:

`WeightRegistry.RegisterWeight` (LinearAlgebra/WeightRegistry.cs ~228):
```csharp
bool dropped = weight.TryDropStorageForStreaming(throwOnSharedRefcount: false);
weight.StreamingPoolHandle = handle;
weight.StreamingDropDeferred = !dropped;          // <-- when refcount>1, _data is NOT freed
```
When the weight's storage refcount > 1 the resident `_data` drop is **deferred** (issue #430 path:
lazy weights captured by the next op / a held reference). `TryFinalizeDeferredDrop` retries the drop
later — the **forward path** runs it at end-of-step, but the **GetParameterChunks round-trip never
triggers finalization**, so every layer's `_data` stays resident, the deferred-drop set grows to the
full ~25 GB weight set, and the next `new byte[byteCount]` snapshot allocation in `RegisterWeight`
OOMs even though `StreamingTensorPool` itself reports under its `ComputeResidentCapBytes` (8 GB) budget.

The streaming pool DOES have a disk-backing store (`StreamingTensorPool._backingFile`, LRU page-out),
so the spill mechanism exists — the gap is that the **owner tensors' resident `_data` is never
dropped** on this path, so paging the pool's byte[] snapshots to disk doesn't help (the unfreed
`_data` is the accumulator).

## Proposed fix (Tensors)
Bound the resident set on the register-heavy materialization path by finalizing deferred drops as we
go, so each layer's `_data` is freed once its snapshot is in the pool:
1. After `RegisterWeight` defers a drop, attempt finalization opportunistically when the pool crosses
   budget (e.g. `EvictIfOverBudget` also runs `TryFinalizeDeferredDrop` over `_pendingDeferredDrops`),
   OR
2. Expose a `WeightRegistry.FinalizeDeferredDrops()` that the AiDotNet materialization path calls per
   layer / per N layers.

**Correctness gate (the round-trip is the oracle):** the test writes a ramp into each weight via
`Tensor.Data.Span` then re-reads to verify. So a dropped+rehydrated weight must return the WRITTEN
ramp — eviction of a dirty (mutated-since-snapshot) entry must write-back. `StreamingTensorPool` has
clean/dirty eviction; confirm a raw `Data.Span` write marks the entry dirty (the `MarkDirty` hooks are
on `Matrix`/`MatrixBase` mutators — a raw span write may bypass them). If it bypasses, mark the entry
dirty on rehydrate-for-write, or the verify pass will read stale (pre-ramp) values.

## AiDotNet-side wiring (already drafted, branch `fix/generated-am-modelbugs`, uncommitted)
These are correct + necessary and move the OOM forward through the three layers; they need this Tensors
fix to fully land:
- `NoisePredictorBase.GetParameterChunks/SetParameterChunks` + `MMDiTNoisePredictor` overrides: call
  `MaybeEngageWeightStreaming()` before materializing (engages streaming on the param path, not just
  forward).
- `LayerBase.EnsureParametersMaterialized`: call `RegisterStreamingWeightsWithPool()` after
  `EnsureInitialized()` (the forward path does this via `EnsureInitializedFromInput`; the param path
  skipped it, so nothing was registered for eviction).

## Verify (two-repo loop)
1. Build this Tensors branch → local nupkg.
2. Bump the AiDotNet `fix/generated-am-modelbugs` branch to it.
3. Run (CI repro env: `DOTNET_GCHeapHardLimit=0x400000000 DOTNET_gcServer=1`, serial):
   `Flux2Model_GetSetParameters_RoundTrips`, `FluxDoubleStreamPredictor_…`, `SiTPredictor_…`.
   Oracle = no OOM AND the ramp value-check passes (write-back correctness).
4. Also re-run the `Step-Sync` (StepVideo) + `03b` (ControlNetFlux Clone) OOMs — same foundation-scale
   full-materialization family; this fix likely helps them too.

---

## MEASURED CORRECTION (supersedes the deferred-drop hypothesis above)

Instrumented `WeightRegistry.GetStreamingReport()` during enumeration (AiDotNet side, no Tensors
rebuild — the engage+register wiring was enough to engage streaming on 0.104.6). Two probes:

**Read pass** (enumerate + read `chunk.Length` only): resident pinned at the **8 GB cap**,
evictions climb 0→519, 23.6 GB written to disk, reaches ~chunk 900 / ~31 GB. **No OOM** — it *times
out* (~10 min for one pass; disk-bound). So the pool's snapshot eviction works; the deferred-drop
hypothesis is **disproven**.

**Write pass** (`chunk.Data.Span[i] = …`, exactly as `AssertParameterChunksRoundTrip`):
| chunk | residentMB | evictions | diskWriteMB |
|------|-----------|-----------|-------------|
| 50  | 1489 | 0 | 0 |
| 100 | 3361 | 0 | 0 |
| 150 | 5090 | 0 | 0 |
| 200 | 6746 | **0** | **0** |
→ **OOM ~chunk 230**, before the 8 GB snapshot-eviction trigger ever fires.

### The real mechanism
`chunk.Data.Span` triggers `WeightRegistry.Materialize` → `RehydrateInto` returns a **caller-owned
copy** that `RestoreStorageFromBytes` installs as the tensor's `_data`. That owner `_data` copy is
**not counted in the pool's `_residentBytes`** (only the pool's own `entry.Data` snapshot is). The
eviction trigger (`EvictIfOverBudget`, gated on `_residentBytes` ≥ cap) therefore never sees the
owner copies — they accumulate one-per-written-chunk (held live by each layer's `_weights`) until the
process OOMs, well before the snapshot budget is reached (evictions=0 at 6.7 GB resident).

### The fix (real, load-bearing — streaming core)
1. **Account owner-resident bytes.** When `Materialize` restores an owner's `_data`, the pool must
   include that in the resident budget (or page out its own redundant `entry.Data` AND track the
   owner copy) so `EvictIfOverBudget` fires and `_pendingOwnerDrops` / `DrainOwnerDropsAfterEviction`
   sheds cold owners' `_data` under pressure — exactly the bounding the read path already gets.
2. **Dirty write-back on owner-drop.** A test-written (mutated) owner copy is dirty vs the pool
   snapshot; owner-drop must re-snapshot (write-back) before freeing `_data`, or the round-trip's
   verify pass reads stale (pre-write) values. (`Span` writes likely bypass the `MarkDirty` hooks on
   `Matrix`/`MatrixBase`, so dirty-tracking on the rehydrated-then-mutated path needs wiring.)

### Plus: inherent speed
Even fully memory-bounded, a 31 GB model round-trip streams ~70 GB across 3 passes ≈ 30 min →
exceeds the per-test timeout. Once memory-safe, these foundation-scale contract tests warrant a
longer timeout / `HeavyTimeout` (matches #1709) — the #1715 footprint concern is then satisfied.

This is the streaming path used by every >500M-param model's inference+training, so the accounting +
write-back change must be made carefully and validated with the two-repo loop before merge.

---

## DEFINITIVE CONCLUSION — streaming round-trip is the wrong tool for these tests

Pursuing the memory fix to the end surfaced three independent blockers that together mean
`{Flux2,SiT,FluxDoubleStream}_GetSetParameters_RoundTrips` **cannot be made green by streaming** at
their default foundation scale (Flux2 = 6.5 B params / ~31 GB):

1. **Precision vs exact-equality.** The round-trip asserts `span[i] == expected` exactly. The
   streaming store often uses **bf16** (`ResolveStoreEncoding` / `StreamingStoreDtype` policy), which
   is lossy — a bf16 page-out → rehydrate returns a rounded value, so the exact assert fails *even
   with perfect memory bounding + write-back*. Lossless-only streaming would avoid this but is not the
   default and doesn't help the runtime blocker.
2. **Runtime.** Even fully memory-bounded, the round-trip streams ~70 GB across 3 passes to/from disk
   ≈ 30 min — far over the per-test timeout.
3. **Invariant friction.** Write-back of a mutated entry re-writes a backing-file slice, which the
   pool's append-only + zero-copy-mmap design assumes never happens — so a correct write-back needs to
   coordinate with (or disable) those for the mutated-resident case.

**Recommended resolution (matches the #1709 precedent):** treat these as foundation-scale contract
tests that don't fit the 16 GB default gate — keep the AiDotNet read-path memory wiring (so
construction/read don't OOM-crash) and exclude them from the default gate via `HeavyTimeout` (or have
the test exercise the get/set *framing* without a full 31 GB exact-value round-trip). The general
streaming improvement (owner-resident accounting + owner-side write-back, built on the
`ReplaceEntryData` primitive added here) remains worthwhile for real training/inference on
foundation-scale models, but it is a separate, multi-day, design-reviewed feature — NOT a fix that
makes these exact-equality round-trip tests pass.
