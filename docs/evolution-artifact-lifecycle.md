# Explicit kernel artifact lifecycle

US-25 / [Evolution issue #43](https://github.com/ooples/AiDotNet.Evolution/issues/43).
This is the Tensors consumer implementation, not a program/AutoML deployment registry or a claim of competitive speedup.

## Application integration

1. Construct `KernelTuningApplicabilityEnvelope` from the typed kernel identity and SHA-256 fingerprints of the runtime, compiler/toolchain, dataset and workload/dispatch contract. Include every material input in these application-owned manifests. Hash an explicit versioned “not applicable” manifest for an unused dimension; do not silently omit it.
2. Register a validated `KernelTuningDeploymentSnapshot<TConfiguration>` using `KernelTuningArtifactRegistry<TConfiguration>.Register`. Retain its `ArtifactId`. Registration stores the canonical configuration, exact paired raw timing/correctness/resource evidence, codec identity and applicability. Loading verifies them but never activates a candidate.
3. Explicitly authorize `PromoteArtifactAsync` with `KernelTuningArtifactPromotionPolicy`. It replays the candidate against the exact current deployment, retains the fresh evidence even when rejected, checks median/noise/lower-bound/P95 gates, and uses compare-and-swap publication. A stale replay cannot replace a changed incumbent. `WasPromoted`, artifact `IsDurable`, and winner-cache `WasPersisted` are different outcomes.
4. Wrap an initially valid private tuner in `KernelTuningLifecycle<TConfiguration>`. Dispatch through `Select(observedEnvelope)` at controlled batch boundaries. Supply a known-valid fallback for **each** observed environment. A mismatch or failed quarantine admission returns that fallback and coalesces one pending request; it never launches work inside a kernel or from a hidden timer.
5. Drain pending work with `RetunePendingAsync(factory, seeds, idleGate, token)`. The factory must return a private tuner, deployment registry and envelope-isolated persistence store; the controller rejects reuse of its active handle and inflated evaluation/proposal budgets. The factory's fallback/finalist protocol must be valid for the requested envelope. Choose finite lifetime admission, search evaluation/proposal, timeout, grace and cooldown limits. Cancellation/failure consumes its admission. Timed-out uncooperative work retains the single capacity slot until it settles and cannot be adopted late. Search limits do not count the separate finalist replay's internal measurements; the replay provider must bound those itself.
6. Feed exact-deployment, predeclared raw timing windows into `ObserveLatencyAsync`, with strictly increasing observation timestamps and an optional prior `ArtifactId`. After the configured consecutive P95 breaches, the controller retains **all** contributing windows and invokes persistent quarantine. A healthy window resets the breach sequence. A missing, corrupt or incompatible prior does not prevent quarantine; dispatch falls back when rollback is unavailable. The raw evidence digest is linked from the quarantine receipt and can be exported with `Read` for diagnosis and future-search exclusion.

```csharp
// tuner, envelope, codec, seeds and idleGate are the application's typed experiment inputs.
var artifacts = new KernelTuningArtifactRegistry<MyKernelConfiguration>(privateAbsolutePath);
var approval = new KernelTuningArtifactPromotionPolicy(
    new KernelTuningOptions { MinimumPromotionRatio = 1.05, MaximumP95LatencyRatio = 1.0 });
var limits = new KernelTuningLifecyclePolicy(
    maximumRetunes: 3, evaluationsPerRetune: 50, proposalsPerRetune: 100,
    timeout: TimeSpan.FromMinutes(2), gracePeriod: TimeSpan.FromSeconds(5),
    cooldown: TimeSpan.FromMinutes(10), monitoringSamples: 9, consecutiveBreaches: 2);
var lifecycle = new KernelTuningLifecycle<MyKernelConfiguration>(
    tuner, envelope, artifacts, codec, limits, approval, KnownValidFallback);
var configuration = lifecycle.Select(observedEnvelope); // dispatch using this selection
// A separately authorized idle worker calls RetunePendingAsync; selection itself does no search.
```

## Safety and durability boundaries

- Use a quarantined store for monitoring/rollback. Admission is refreshed at selection boundaries, including receipts written by another process. This is not instantaneous distributed revocation: in-flight kernels and callers bypassing the controller are outside its control.
- Artifact hashes establish integrity, not publisher authenticity. Protect the absolute registry path and its parents from untrusted writers; direct link checks do not provide an adversarial filesystem sandbox. Configurations/codecs must obey canonical serialization and must not be mutated by other callers.
- Objects are bounded to 4 MiB, parsed with depth/duplicate-property checks and content-addressed without replacement. Combined monitoring windows are bounded to 65,536 timings. The application owns disk quotas, evidence export and retention; failed pending files are deliberately retained.
- Native power-loss durability is currently verified only by the existing Linux file/directory/ancestor barriers on supported architectures. Existing identical objects are re-flushed before reporting fresh durability. Windows can retain visible artifacts but reports `IsDurable == false`; default promotion therefore refuses activation there. Setting `requireDurableArtifact: false` is an explicit application acceptance of best-effort storage, not a new durability guarantee.
- Quarantine retains its stricter existing rule: rollback requires its durable receipt. On an unsupported or failed barrier it deactivates the regressing deployment and leaves dispatch on the application's valid fallback. Evidence-write failure also deactivates and latches fallback in this controller; operator intervention is required. No successful evidence receipt is claimed in that case.
- A new controller resets lifetime admission accounting. Use external application accounting for restart-stable or distributed budgets. These CPU contract tests authorize no API spending and establish no real-GPU or model-driven competitive performance result.

## Maintained verification gate

`tests/AiDotNet.Tensors.Autotune.Tests` references the real production project and links the complete existing autotune test directory. It builds independently of unrelated CLI/generator/GPU-heavy test suites; the original full-suite project and CI are unchanged. Its coverage settings measure the autotune namespace only, avoiding unrelated whole-library instrumentation overhead.

```powershell
dotnet test tests/AiDotNet.Tensors.Autotune.Tests/AiDotNet.Tensors.Autotune.Tests.csproj -c Release -f net10.0 -p:GeneratePackageOnBuild=false --logger "trx;LogFileName=lifecycle.trx"
```

Repeat the final gate for supported runtime targets `net8.0` and `net471` (Framework execution requires Windows). The additional `Evolution lifecycle source integration` CI checks pinned Evolution source `255feb24369702a32ea9db7a3f8a0b7a847d2762`; it does **not** substitute for the repository's normal package-based build or prove unpublished dependencies are available from NuGet. Retain current-head TRX/coverage receipts before marking the PR ready.
