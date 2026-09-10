# Deactivating an observed tuning deployment

Roadmap slice: US-25. Validated promotion already exists; this adds a runtime escape hatch.

```csharp
var observed = deployment.Current;
// Dispatch/validate against this exact snapshot. On a detected regression:
if (observed is not null)
    deployment.TryDeactivate(observed);
// Future TryGet calls return false if removal won; dispatch uses its built-in fallback.
```

Given a runtime check fails for a captured snapshot, when `TryDeactivate` sees that snapshot still active,
then it atomically removes it so subsequent dispatch can fall back.

Given a newer snapshot was published, when an older operation reports failure, then its deactivation returns
false and cannot remove the newer deployment, even if both snapshots name the same genome.

This does not cancel already running kernels, mutate cache files, persist a quarantine, automatically detect
drift, or forbid later hydration/tuning from republishing. A persistent operational quarantine needs a separate
policy controlling those producers and the cache. Never reread Current after the failing operation and use that
new snapshot as the deactivation target: capture the one actually used.
