using System.Text;
using System.Text.Json;
using System.Runtime.InteropServices;
using AiDotNet.Evolution;
using AiDotNet.Tensors.Helpers.Autotune;
using Xunit;

namespace AiDotNet.Tensors.Tests.Helpers.Autotune;

public sealed partial class EvolutionKernelAutotunerTests
{
    private static KernelTuningApplicabilityEnvelope Envelope(char runtime = 'a', char compiler = 'b',
        char dataset = 'c', char workload = 'd', KernelTuningIdentity? identity = null) =>
        new(identity ?? Identity(), new string(runtime, 64), new string(compiler, 64), new string(dataset, 64), new string(workload, 64));

    private KernelTuningArtifactRegistry<FakeKernelConfiguration> ArtifactRegistry(string suffix = "artifacts") => new(Journal(suffix));
    private static KernelTuningLifecyclePolicy LifecyclePolicy(int maximumRetunes = 1, int breaches = 2,
        TimeSpan? timeout = null) => new(maximumRetunes, 3, 3, timeout ?? TimeSpan.FromSeconds(10),
            TimeSpan.FromMilliseconds(100), TimeSpan.Zero, monitoringSamples: 3, consecutiveBreaches: breaches);
    private static KernelTuningArtifactPromotionPolicy PromotionPolicy(double minimum = 1.03) =>
        new(new KernelTuningOptions { MinimumPromotionRatio = minimum }, requireDurableArtifact: false);

    [Fact]
    public void ArtifactRegistry_RoundTripsExactEvidenceAndRegistrationIsIdempotent()
    {
        var registry = ArtifactRegistry();
        var original = Snapshot(Identity(), Seeds()[2], 0);
        var first = registry.Register(original, Envelope(), new FakeKernelCodec());
        var second = registry.Register(original, Envelope(), new FakeKernelCodec());
        Assert.Equal(first.ArtifactId, second.ArtifactId);
        Assert.Single(Directory.GetFiles(Journal("artifacts"), "*.artifact"));
        var decoded = new KernelTuningArtifactRegistry<FakeKernelConfiguration>(Journal("artifacts"))
            .Load(first.ArtifactId, Envelope(), new FakeKernelCodec());
        Assert.Equal(original.Configuration, decoded.Configuration);
        Assert.Equal(original.GenomeId, decoded.GenomeId);
        Assert.Equal(original.PromotionEvidence.Samples, decoded.PromotionEvidence.Samples);
        Assert.Equal(original.EvidenceRole, decoded.EvidenceRole);
        bool durable = RuntimeInformation.IsOSPlatform(OSPlatform.Linux) &&
            RuntimeInformation.ProcessArchitecture is Architecture.X86 or Architecture.X64 or Architecture.Arm or Architecture.Arm64;
        Assert.Equal(durable, first.IsDurable);
        Assert.Equal(durable, second.IsDurable);
    }

    [Theory]
    [InlineData("runtime")]
    [InlineData("compiler")]
    [InlineData("dataset")]
    [InlineData("workload")]
    [InlineData("device")]
    public void ArtifactRegistry_RejectsEveryChangedApplicabilityDimension(string dimension)
    {
        var registry = ArtifactRegistry();
        var receipt = registry.Register(Snapshot(Identity(), Seeds()[2], 0), Envelope(), new FakeKernelCodec());
        var identity = Identity();
        var changedIdentity = new KernelTuningIdentity(identity.Kernel, identity.Shape,
            new KernelTuningDeviceFingerprint(KernelTuningDeviceKind.NvidiaGpu, "different-driver", "same-model"),
            identity.Backend, identity.SearchSpaceVersion, identity.BenchmarkProtocolVersion);
        var changed = Envelope(dimension == "runtime" ? 'e' : 'a', dimension == "compiler" ? 'e' : 'b',
            dimension == "dataset" ? 'e' : 'c', dimension == "workload" ? 'e' : 'd', dimension == "device" ? changedIdentity : identity);
        Assert.Throws<InvalidDataException>(() => registry.Load(receipt.ArtifactId, changed, new FakeKernelCodec()));
    }

    [Fact]
    public void ArtifactRegistry_RejectsTamperingTraversalAndRehashedDuplicateProperties()
    {
        var registry = ArtifactRegistry();
        var receipt = registry.Register(Snapshot(Identity(), Seeds()[2], 0), Envelope(), new FakeKernelCodec());
        byte[] original = registry.Read(receipt.ArtifactId);
        Assert.Throws<ArgumentException>(() => registry.Read("../outside"));
        byte[] duplicate = Encoding.UTF8.GetBytes(Encoding.UTF8.GetString(original).Replace("\"SchemaVersion\":1", "\"SchemaVersion\":1,\"SchemaVersion\":1"));
        var duplicateReceipt = registry.RetainEvidence(duplicate);
        Assert.Throws<InvalidDataException>(() => registry.Load(duplicateReceipt.ArtifactId, Envelope(), new FakeKernelCodec()));
        File.WriteAllBytes(Path.Combine(Journal("artifacts"), receipt.ArtifactId + ".artifact"), new byte[] { 1, 2 });
        Assert.Throws<InvalidDataException>(() => registry.Read(receipt.ArtifactId));
        Assert.Throws<InvalidDataException>(() => registry.Register(Snapshot(Identity(), Seeds()[2], 0), Envelope(), new FakeKernelCodec()));
    }

    [Fact]
    public async Task ArtifactPromotion_FreshReplayUsesExplicitFrozenPolicyAndRetainsRejectedEvidence()
    {
        var registry = ArtifactRegistry();
        var artifact = registry.Register(Snapshot(Identity(), Seeds()[2], 0), Envelope(), new FakeKernelCodec());
        var tuner = CreateTuner(new(), new MemoryStore(), MeasurePassed);
        var options = new KernelTuningOptions { MinimumPromotionRatio = 100 };
        var policy = new KernelTuningArtifactPromotionPolicy(options, requireDurableArtifact: false);
        options.MinimumPromotionRatio = 1; // Cannot change the already authorized thresholds.
        var rejected = await tuner.PromoteArtifactAsync(registry, artifact.ArtifactId, Envelope(), policy);
        Assert.False(rejected.WasPromoted);
        Assert.Null(tuner.Deployment.Current);
        Assert.NotEmpty(registry.Read(rejected.Evidence.ArtifactId));
        var promoted = await tuner.PromoteArtifactAsync(registry, artifact.ArtifactId, Envelope(), PromotionPolicy());
        Assert.True(promoted.WasPromoted);
        Assert.Same(promoted.ActiveDeployment, tuner.Deployment.Current);
        Assert.Equal(Seeds()[2], promoted.ActiveDeployment!.Configuration);
    }

    [Fact]
    public async Task ArtifactPromotion_StaleReplayCannotReplaceANewerDeployment()
    {
        var registry = ArtifactRegistry();
        var artifact = registry.Register(Snapshot(Identity(), Seeds()[2], 0), Envelope(), new FakeKernelCodec());
        var deployments = new KernelTuningDeploymentRegistry<FakeKernelConfiguration>();
        var tuner = new EvolutionKernelAutotuner<FakeKernelConfiguration>(Identity(), new FakeKernelCodec(),
            new FakeKernelVariation(), MeasurePassed, new CallbackFinalist(() =>
                deployments.GetOrCreate(Identity()).Publish(Snapshot(Identity(), Seeds()[1], 0))),
            EngineOptions(), deploymentRegistry: deployments, store: new MemoryStore());
        var result = await tuner.PromoteArtifactAsync(registry, artifact.ArtifactId, Envelope(), PromotionPolicy());
        Assert.False(result.WasPromoted);
        Assert.Equal(Seeds()[1], tuner.Deployment.Current!.Configuration);
    }

    [Fact]
    public async Task Lifecycle_DriftFallsBackAndBoundedRetuneAdoptsOnlyCurrentEnvelope()
    {
        var original = CreateTuner(new(), new MemoryStore(), MeasurePassed);
        await original.TuneAsync(Seeds());
        var lifecycle = new KernelTuningLifecycle<FakeKernelConfiguration>(original, Envelope(), ArtifactRegistry(),
            new FakeKernelCodec(), LifecyclePolicy(), PromotionPolicy(), _ => Seeds()[0]);
        Assert.Equal(Seeds()[2], lifecycle.Select(Envelope()));
        Assert.Equal(Seeds()[0], lifecycle.Select(Envelope(runtime: 'e')));
        Assert.True(lifecycle.RetuneRequested);
        int factories = 0;
        EvolutionKernelAutotuner<FakeKernelConfiguration> Factory(KernelTuningApplicabilityEnvelope requested, EvolutionEngineOptions options)
        {
            factories++;
            return new(requested.Identity, new FakeKernelCodec(), new FakeKernelVariation(), MeasurePassed,
                Finalist(), options, store: new MemoryStore());
        }
        Assert.Equal(KernelTuningRetuneStatus.Completed, await lifecycle.RetunePendingAsync(Factory, Seeds(), new ImmediateIdleGate()));
        Assert.Equal(Seeds()[2], lifecycle.Select(Envelope(runtime: 'e')));
        Assert.False(lifecycle.RetuneRequested);
        lifecycle.Select(Envelope(runtime: 'f'));
        Assert.Equal(KernelTuningRetuneStatus.BudgetDenied, await lifecycle.RetunePendingAsync(Factory, Seeds(), new ImmediateIdleGate()));
        Assert.Equal(1, factories);
        Assert.Equal(1, lifecycle.AdmittedRetunes);
    }

    [Fact]
    public async Task Lifecycle_RejectsFactoryBudgetInflationBeforeAnyEvaluation()
    {
        int calls = 0;
        var lifecycle = new KernelTuningLifecycle<FakeKernelConfiguration>(CreateTuner(new(), new MemoryStore(), MeasurePassed),
            Envelope(), ArtifactRegistry(), new FakeKernelCodec(), LifecyclePolicy(), PromotionPolicy(), _ => Seeds()[0]);
        lifecycle.Select(Envelope(runtime: 'e'));
        await Assert.ThrowsAsync<InvalidOperationException>(() => lifecycle.RetunePendingAsync((requested, options) =>
        {
            options.MaxEvaluationAttempts = 4;
            return new(requested.Identity, new FakeKernelCodec(), new FakeKernelVariation(),
                (configuration, context, token) => { calls++; return MeasurePassed(configuration, context, token); },
                Finalist(), options, store: new MemoryStore());
        }, Seeds(), new ImmediateIdleGate()));
        Assert.Equal(0, calls);
        Assert.Equal(1, lifecycle.AdmittedRetunes);
    }

    [Fact]
    public async Task Lifecycle_MonitorRetainsAllRawWindowsAndRestoresExactPriorArtifactAfterCommit()
    {
        // Inject only the OS commit boundary; the native boundary has separate platform tests.
        var store = new QuarantinedKernelTuningStore<FakeKernelConfiguration>(Journal(), new MemoryStore(),
            new RecordedCommitOperations(QuarantineCommitMode.LinuxDirectorySync, ReceiptFailure.None));
        var tuner = CreateTuner(new(), store, MeasurePassed);
        var run = await tuner.TuneAsync(Seeds());
        var registry = ArtifactRegistry();
        var prior = registry.Register(run.IncumbentDeployment, Envelope(), new FakeKernelCodec());
        var lifecycle = new KernelTuningLifecycle<FakeKernelConfiguration>(tuner, Envelope(), registry,
            new FakeKernelCodec(), LifecyclePolicy(), PromotionPolicy(), _ => Seeds()[0]);
        var samples = Enumerable.Repeat(TimeSpan.FromSeconds(1), 3).ToArray();
        var now = DateTimeOffset.UtcNow;
        Assert.Null(await lifecycle.ObserveLatencyAsync(run.ActiveDeployment, samples, now, prior.ArtifactId));
        var result = await lifecycle.ObserveLatencyAsync(run.ActiveDeployment, samples, now.AddSeconds(1), prior.ArtifactId);
        Assert.NotNull(result);
        Assert.True(result.WasApplied);
        Assert.NotNull(result.RollbackDeployment);
        Assert.Equal(run.IncumbentDeployment.GenomeId, result.RollbackDeployment.GenomeId);
        Assert.Equal(run.IncumbentDeployment.Configuration, lifecycle.Select(Envelope()));
        string quarantinePath = Assert.Single(Directory.GetFiles(Journal(), "*.quarantine.json"));
        using var document = JsonDocument.Parse(File.ReadAllBytes(quarantinePath));
        string evidenceId = document.RootElement.GetProperty("Evidence").GetProperty("RawEvidenceSha256").GetString()!;
        using var raw = JsonDocument.Parse(registry.Read(evidenceId));
        var windows = raw.RootElement.GetProperty("Windows");
        Assert.Equal(2, windows.GetArrayLength());
        Assert.All(windows.EnumerateArray(), window => Assert.Equal(3, window.GetProperty("SampleTicks").GetArrayLength()));
        Assert.Null(await lifecycle.ObserveLatencyAsync(run.ActiveDeployment, samples, now.AddSeconds(2)));
    }

    [Theory]
    [InlineData("missing")]
    [InlineData("corrupt")]
    [InlineData("incompatible")]
    [InlineData("invalid")]
    public async Task Lifecycle_UnavailablePriorCannotPreventRegressionQuarantine(string failure)
    {
        var tuner = CreateTuner(new(), new QuarantinedKernelTuningStore<FakeKernelConfiguration>(Journal(), new MemoryStore()), MeasurePassed);
        var run = await tuner.TuneAsync(Seeds());
        var registry = ArtifactRegistry();
        string prior = failure == "invalid" ? "../outside" : new string('a', 64);
        if (failure is "corrupt" or "incompatible")
        {
            prior = registry.Register(run.IncumbentDeployment,
                failure == "incompatible" ? Envelope(runtime: 'e') : Envelope(), new FakeKernelCodec()).ArtifactId;
            if (failure == "corrupt") File.WriteAllText(Path.Combine(Journal("artifacts"), prior + ".artifact"), "corrupt");
        }
        var lifecycle = new KernelTuningLifecycle<FakeKernelConfiguration>(tuner, Envelope(), registry,
            new FakeKernelCodec(), LifecyclePolicy(breaches: 1), PromotionPolicy(), _ => Seeds()[0]);
        var result = await lifecycle.ObserveLatencyAsync(run.ActiveDeployment,
            Enumerable.Repeat(TimeSpan.FromSeconds(1), 3).ToArray(), DateTimeOffset.UtcNow, prior);
        Assert.NotNull(result);
        Assert.True(result.WasApplied);
        Assert.Null(result.RollbackDeployment);
        Assert.Null(tuner.Deployment.Current);
        Assert.Equal(Seeds()[0], lifecycle.Select(Envelope()));
        Assert.True(lifecycle.RetuneRequested);
    }

    [Fact]
    public async Task Lifecycle_EvidenceLossDeactivatesAndLatchesFallbackWithoutReload()
    {
        var tuner = CreateTuner(new(), new QuarantinedKernelTuningStore<FakeKernelConfiguration>(Journal(), new MemoryStore()), MeasurePassed);
        var run = await tuner.TuneAsync(Seeds());
        var registry = ArtifactRegistry();
        var lifecycle = new KernelTuningLifecycle<FakeKernelConfiguration>(tuner, Envelope(), registry,
            new FakeKernelCodec(), LifecyclePolicy(breaches: 1), PromotionPolicy(), _ => Seeds()[0]);
        Directory.Move(Journal("artifacts"), Journal("displaced-artifacts"));
        await Assert.ThrowsAnyAsync<IOException>(() => lifecycle.ObserveLatencyAsync(run.ActiveDeployment,
            Enumerable.Repeat(TimeSpan.FromSeconds(1), 3).ToArray(), DateTimeOffset.UtcNow));
        Assert.Null(tuner.Deployment.Current);
        Assert.Equal(Seeds()[0], lifecycle.Select(Envelope()));
        Assert.False(lifecycle.RetuneRequested);
    }

    [Fact]
    public async Task Lifecycle_HealthyWindowResetsBreachesAndRepeatedTimestampIsRejected()
    {
        var tuner = CreateTuner(new(), new QuarantinedKernelTuningStore<FakeKernelConfiguration>(Journal(), new MemoryStore()), MeasurePassed);
        var run = await tuner.TuneAsync(Seeds());
        var lifecycle = new KernelTuningLifecycle<FakeKernelConfiguration>(tuner, Envelope(), ArtifactRegistry(),
            new FakeKernelCodec(), LifecyclePolicy(), PromotionPolicy(), _ => Seeds()[0]);
        var bad = Enumerable.Repeat(TimeSpan.FromSeconds(1), 3).ToArray();
        var good = Enumerable.Repeat(run.ActiveDeployment.Measurement.Timing.P95, 3).ToArray();
        var now = DateTimeOffset.UtcNow;
        Assert.Null(await lifecycle.ObserveLatencyAsync(run.ActiveDeployment, bad, now));
        await Assert.ThrowsAsync<ArgumentException>(() => lifecycle.ObserveLatencyAsync(run.ActiveDeployment, bad, now));
        Assert.Null(await lifecycle.ObserveLatencyAsync(run.ActiveDeployment, good, now.AddSeconds(1)));
        Assert.Null(await lifecycle.ObserveLatencyAsync(run.ActiveDeployment, bad, now.AddSeconds(2)));
        Assert.Same(run.ActiveDeployment, tuner.Deployment.Current);
        Assert.True((await lifecycle.ObserveLatencyAsync(run.ActiveDeployment, bad, now.AddSeconds(3)))!.WasApplied);
    }

    [Fact]
    public async Task Lifecycle_StaleWorkCannotAdoptAndRequestsCoalesceToNewestEnvelope()
    {
        var lifecycle = new KernelTuningLifecycle<FakeKernelConfiguration>(CreateTuner(new(), new MemoryStore(), MeasurePassed),
            Envelope(), ArtifactRegistry(), new FakeKernelCodec(), LifecyclePolicy(maximumRetunes: 2), PromotionPolicy(), _ => Seeds()[0]);
        var idle = new HeldIdleGate();
        lifecycle.Select(Envelope(runtime: 'e'));
        var running = lifecycle.RetunePendingAsync(FreshTuner, Seeds(), idle);
        await Entered(idle);
        lifecycle.Select(Envelope(runtime: 'f'));
        lifecycle.Select(Envelope(compiler: 'e'));
        Assert.Equal(KernelTuningRetuneStatus.Busy, await lifecycle.RetunePendingAsync(FreshTuner, Seeds(), new ImmediateIdleGate()));
        idle.Release.TrySetResult(true);
        Assert.Equal(KernelTuningRetuneStatus.Stale, await running);
        string? requestedKey = null;
        Assert.Equal(KernelTuningRetuneStatus.Completed, await lifecycle.RetunePendingAsync((envelope, options) =>
        { requestedKey = envelope.StableKey; return FreshTuner(envelope, options); }, Seeds(), new ImmediateIdleGate()));
        Assert.Equal(Envelope(compiler: 'e').StableKey, requestedKey);
        Assert.Equal(Seeds()[2], lifecycle.Select(Envelope(compiler: 'e')));
    }

    [Fact]
    public async Task Lifecycle_AbandonedIdleWaitKeepsCapacityAndCannotAdoptLateResult()
    {
        var lifecycle = new KernelTuningLifecycle<FakeKernelConfiguration>(CreateTuner(new(), new MemoryStore(), MeasurePassed),
            Envelope(), ArtifactRegistry(), new FakeKernelCodec(), LifecyclePolicy(maximumRetunes: 2, timeout: TimeSpan.FromMilliseconds(500)),
            PromotionPolicy(), _ => Seeds()[0]);
        var idle = new HeldIdleGate();
        lifecycle.Select(Envelope(runtime: 'e'));
        var running = lifecycle.RetunePendingAsync(FreshTuner, Seeds(), idle);
        try
        {
            await Entered(idle);
            Assert.Equal(KernelTuningRetuneStatus.Abandoned, await running);
            Assert.Equal(Seeds()[0], lifecycle.Select(Envelope(runtime: 'e')));
            Assert.Equal(KernelTuningRetuneStatus.Busy, await lifecycle.RetunePendingAsync(FreshTuner, Seeds(), new ImmediateIdleGate()));
            Assert.Equal(1, lifecycle.AdmittedRetunes);
        }
        finally { idle.Release.TrySetResult(true); }
    }

    [Fact]
    public void LifecyclePolicy_BoundsTotalRetainedMonitoringEvidence()
        => Assert.Throws<ArgumentException>(() => new KernelTuningLifecyclePolicy(1, 3, 3, TimeSpan.FromSeconds(1),
            TimeSpan.Zero, TimeSpan.Zero, monitoringSamples: 4096, consecutiveBreaches: 100));

    private static EvolutionKernelAutotuner<FakeKernelConfiguration> FreshTuner(
        KernelTuningApplicabilityEnvelope envelope, EvolutionEngineOptions options) =>
        new(envelope.Identity, new FakeKernelCodec(), new FakeKernelVariation(), MeasurePassed, Finalist(), options, store: new MemoryStore());

    private sealed class CallbackFinalist : IKernelTuningFinalistEvaluator<FakeKernelConfiguration>
    {
        private readonly Action _callback;
        internal CallbackFinalist(Action callback) => _callback = callback;
        public ValueTask<KernelTuningFinalistReplay<FakeKernelConfiguration>> ReplayAsync(KernelTuningIdentity identity,
            FakeKernelConfiguration candidate, KernelTuningDeploymentSnapshot<FakeKernelConfiguration>? activeDeployment,
            CancellationToken cancellationToken = default)
        {
            var replay = Finalist().ReplayAsync(identity, candidate, activeDeployment, cancellationToken);
            _callback();
            return replay;
        }
    }

    private static async Task Entered(HeldIdleGate idle)
    {
        Assert.Same(idle.Entered.Task, await Task.WhenAny(idle.Entered.Task, Task.Delay(TimeSpan.FromSeconds(5))));
    }

    private sealed class HeldIdleGate : IKernelTuningIdleGate
    {
        internal TaskCompletionSource<bool> Entered { get; } = new(TaskCreationOptions.RunContinuationsAsynchronously);
        internal TaskCompletionSource<bool> Release { get; } = new(TaskCreationOptions.RunContinuationsAsynchronously);
        public async ValueTask WaitUntilIdleAsync(KernelTuningIdentity identity, CancellationToken cancellationToken = default)
        { Entered.TrySetResult(true); await Release.Task.ConfigureAwait(false); }
    }

    private sealed class ImmediateIdleGate : IKernelTuningIdleGate
    {
        public ValueTask WaitUntilIdleAsync(KernelTuningIdentity identity, CancellationToken cancellationToken = default)
        { cancellationToken.ThrowIfCancellationRequested(); return default; }
    }
}
