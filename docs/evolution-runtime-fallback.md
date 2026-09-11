# Quarantine, rollback and in-memory deactivation

Roadmap slice: US-25. Validated promotion already exists. These APIs add an explicit runtime response to
caller-observed regression; they do not implement an automatic monitor.

## Persistent quarantine and optional rollback

Pass `QuarantinedKernelTuningStore<TConfiguration>` through the tuner's existing `store` parameter, wrapping
your existing store if needed. Use the same canonical, absolute, private journal directory for every cooperating
producer. Capture both the exact active snapshot used by the operation and any prior validated snapshot you
intend to retain for rollback. The default remains opt-out; existing constructors and package pins are unchanged.

```csharp
var store = new QuarantinedKernelTuningStore<MyConfiguration>(absolutePrivateJournalDirectory, existingStore);
// Supply store: store when constructing EvolutionKernelAutotuner<MyConfiguration>.
// A monitor observes a regression against the exact snapshot used by the operation:
var evidence = new KernelTuningRegressionEvidence(
    KernelTuningRegressionReason.Latency,
    policyVersion: "paired-p95-ms-v1",
    rawEvidenceSha256: retainedRawMeasurementsSha256,
    observedValue: observedP95Milliseconds,
    maximumAllowedValue: allowedP95Milliseconds,
    observedAt: observationTime);
var result = await tuner.QuarantineAsync(observedSnapshot, evidence, priorValidatedSnapshot, cancellationToken);
// WasApplied: the exact snapshot was removed and blocked in this process.
// WasPersisted / ReceiptPath: this call retained a complete write-once regression receipt.
// RollbackDeployment / WasRollbackPersisted: distinguish activation from winner-store persistence.
```

This is a usage fragment: the application supplies immutable `MyConfiguration`, its codec, the configured tuner,
actual monitoring measurements, their raw artifact, and the optional retained prior snapshot. No runtime measurement
or raw artifact is manufactured by the quarantine API. The policy version defines the statistic and its units;
both observed value and limit use those units, and the observed larger-is-worse statistic must exceed its limit.

Given a regression for the exact active snapshot, when `QuarantineAsync` runs, then it deactivates that snapshot,
blocks its canonical configuration, and attempts a durable receipt before publishing any rollback.

Given a later snapshot is active, when stale evidence arrives, then `WasApplied` is false and the later snapshot
is preserved, including a newer validation of the same genome.

Given a retained prior snapshot matches the tuning identity, canonical codec hash and current deployment
validator, when it is not quarantined, then it may be restored and separately persisted. When quarantine was
applied but no eligible prior was published, the handle remains empty for built-in fallback. Stale unapplied
requests leave the current deployment untouched. This is explicit rollback to caller-retained evidence, not an automatic
search through deployment history or a new measurement of the prior configuration.

Given a winner was loaded before quarantine, when final publication is attempted afterward, then publication
rechecks admission under the same process-local journal gate and rejects it. Load-only filtering is insufficient.

Given a complete receipt is retained, when another tuner loads that journal, then the same configuration is
inadmissible even if selected by a new run. The key includes the exact tuning envelope, codec id/version and
canonical configuration hash; the source run-state hash is retained as evidence, not used to bypass quarantine.
Changed device, protocol or codec identity is a separate applicability envelope requiring caller policy.

Given a corrupt, empty or unknown-version record exists, when admission checks its key, then it is denied without
parsing the record. Missing or unreadable journal state also denies admission. Valid receipts contain bounded
strict-UTF8 configuration payloads, identifiers and structured regression evidence; they do not embed raw evidence
or credentials. The caller must protect journal and evidence-artifact permissions.

Given a receipt write fails, when quarantine finishes, then the configuration remains blocked across wrappers
sharing that journal in this process, but `WasPersisted` is false and restart safety is not established. Receipt
writes use create-new temporary files, a disk flush and a non-overwriting move. Existing receipts are retained;
incomplete or conflicting `.pending` files are left recoverable. A preexisting receipt is not falsely reported
as this invocation's successful write. No automatic deletion, unblock or release API exists in this slice.

Given cancellation arrives before mutation, when quarantine checks it, then it changes nothing. After mutation
starts, persistence and rollback finish without cancellation so cancellation cannot interrupt the safety response.

Publication and quarantine are serialized within this process, not across processes. Already active handles
elsewhere are not automatically revoked; an explicit `TryHydrate` rechecks and deactivates an inadmissible current
snapshot. Concurrent cross-process publication, filesystem aliases, journal tampering, in-flight operations,
automatic drift detection, bounded revalidation, and program/AutoML deployment registries remain outside this
slice. Serving `TryGet` still performs only a volatile snapshot read: no file I/O, hashes, locks or monitor calls.

## In-memory deactivation only

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
policy controlling those producers and the cache, such as the opt-in decorator above. Never reread Current after the failing operation and use that
new snapshot as the deactivation target: capture the one actually used.

## Verification scope

The tests use deterministic fake kernel configurations and measurements: they verify deployment and persistence
contracts, not GPU performance or automatic detection quality. They exercise the real tuner, cache and filesystem,
including publication barriers, stale observations, corrupt tombstones, write failures, independent journal-state
reload, immutable evidence, invalid rollback and unchanged legacy behavior.

Cross-runtime verification also exposed an existing winner-cache write failure on .NET Framework: appending a
temporary suffix turned a usable 224-character destination into a failing 263-character path. Both cache and
quarantine now use short unique sibling temporary filenames, retaining same-directory rename semantics. The
regression test uses the original test paths, not shortened fixture paths or a suppressed failure.

Run the committed tests through the main test project:

```text
dotnet test tests/AiDotNet.Tensors.Tests/AiDotNet.Tensors.Tests.csproj -c Release -f net10.0 -p:GeneratePackageOnBuild=false --filter FullyQualifiedName~Helpers.Autotune
dotnet test tests/AiDotNet.Tensors.Tests/AiDotNet.Tensors.Tests.csproj -c Release -f net471 -p:GeneratePackageOnBuild=false --filter FullyQualifiedName~Helpers.Autotune
```

The net471 test command requires a compatible .NET Framework runtime (local Windows verification). The library
also builds for net8.0; the main test project does not target net8.0. No package publication, native kernel change,
paid model call or automatic production rollout is part of this change.
