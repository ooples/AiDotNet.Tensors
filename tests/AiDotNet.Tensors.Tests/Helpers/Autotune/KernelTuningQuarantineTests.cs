using System.Text;
using System.Text.Json;
using System.Runtime.InteropServices;
using AiDotNet.Evolution;
using AiDotNet.Tensors.Helpers.Autotune;
using Xunit;

namespace AiDotNet.Tensors.Tests.Helpers.Autotune;

public sealed partial class EvolutionKernelAutotunerTests
{
    private string Journal(string suffix = "journal") => Path.Combine(_temporaryCachePath, suffix);

    private static void AssertNativePersistence(KernelTuningQuarantineResult<FakeKernelConfiguration> result)
    {
        // Only Linux currently has a verified no-overwrite rename + directory barrier.
        // Do not infer the expected guarantee from the implementation's selected backend.
        bool supportsDurability = RuntimeInformation.IsOSPlatform(OSPlatform.Linux) &&
            RuntimeInformation.ProcessArchitecture is Architecture.X86 or Architecture.X64 or Architecture.Arm or Architecture.Arm64;
        Assert.Equal(supportsDurability, result.WasPersisted);
        if (supportsDurability) Assert.IsType<string>(result.ReceiptPath);
        else Assert.Null(result.ReceiptPath);
    }

    [Fact]
    public void TypedCachePath_StoreRoundTripsWithoutSuppressingWriteErrors()
    {
        var kernel = new KernelId(Identity().Kernel.Category, "typed-evolution-" + Identity().StableKey);
        AutotuneCache.Store(kernel, Identity().Shape, new KernelChoice { Variant = "typed-path-probe" });
        KernelChoice loaded = Assert.IsType<KernelChoice>(AutotuneCache.Lookup(kernel, Identity().Shape));
        Assert.Equal("typed-path-probe", loaded.Variant);
    }

    private static KernelTuningRegressionEvidence Regression() => new(
        KernelTuningRegressionReason.Latency, "p95-ms-v1", new string('a', 64), 12, 10,
        new DateTimeOffset(2026, 9, 10, 12, 0, 0, TimeSpan.FromHours(-4)));

    [Fact]
    public async Task Quarantine_PersistsExactEvidence_AndBlocksFreshJournalHydration()
    {
        var inner = new MemoryStore();
        var store = new QuarantinedKernelTuningStore<FakeKernelConfiguration>(Journal(), inner);
        var tuner = CreateTuner(new(), store, MeasurePassed);
        var run = await tuner.TuneAsync(Seeds());
        var result = await tuner.QuarantineAsync(run.ActiveDeployment, Regression());

        Assert.True(result.WasApplied);
        AssertNativePersistence(result);
        Assert.False(result.WasRollbackPersisted);
        Assert.Null(result.RollbackDeployment);
        Assert.Null(tuner.Deployment.Current);
        Assert.False(tuner.TryHydrate());
        Assert.False(store.TryStore(run.ActiveDeployment, new FakeKernelCodec()));
        var anotherRun = new KernelTuningDeploymentSnapshot<FakeKernelConfiguration>(Identity(),
            run.ActiveDeployment.Configuration, run.ActiveDeployment.GenomeId, run.ActiveDeployment.Measurement,
            "another-run-state", run.ActiveDeployment.PromotionEvidence, run.ActiveDeployment.EvidenceRole);
        Assert.False(store.TryPublish(tuner.Deployment, anotherRun, new FakeKernelCodec(), false));
        string receiptPath = Assert.Single(Directory.GetFiles(Journal(), "*.quarantine.json"));
        using var receipt = JsonDocument.Parse(File.ReadAllBytes(receiptPath));
        var root = receipt.RootElement;
        Assert.Equal("tensor-kernel-quarantine-v1", root.GetProperty("Schema").GetString());
        Assert.Equal(run.ActiveDeployment.Identity.StableKey, root.GetProperty("Identity").GetString());
        Assert.Equal(run.ActiveDeployment.RunStateHash, root.GetProperty("RunStateHash").GetString());
        Assert.Equal(new FakeKernelCodec().Serialize(run.ActiveDeployment.Configuration),
            Encoding.UTF8.GetString(Convert.FromBase64String(Assert.IsType<string>(root.GetProperty("PayloadBase64").GetString()))));
        Assert.Equal(Regression().RawEvidenceSha256, root.GetProperty("Evidence").GetProperty("RawEvidenceSha256").GetString());
        Assert.Equal(TimeSpan.Zero, root.GetProperty("Evidence").GetProperty("ObservedAt").GetDateTimeOffset().Offset);

        // A different journal root has no process-local block: only the retained file can deny this load.
        Directory.CreateDirectory(Journal("fresh-process-state"));
        File.Copy(receiptPath, Path.Combine(Journal("fresh-process-state"), Path.GetFileName(receiptPath)));
        var fresh = CreateTuner(new(), new QuarantinedKernelTuningStore<FakeKernelConfiguration>(
            Journal("fresh-process-state"), inner), MeasurePassed);
        Assert.False(fresh.TryHydrate());
        Assert.Null(fresh.Deployment.Current);
        var retuned = await tuner.TuneAsync(Seeds());
        Assert.NotEqual(run.ActiveDeployment.GenomeId, retuned.ActiveDeployment.GenomeId);
    }

    [Fact]
    public async Task Quarantine_RestoresEligiblePrior_AndStaleObservationCannotRemoveIt()
    {
        var store = new QuarantinedKernelTuningStore<FakeKernelConfiguration>(Journal());
        var tuner = CreateTuner(new(), store, MeasurePassed);
        var run = await tuner.TuneAsync(Seeds());
        var prior = Snapshot(Identity(), Seeds()[1], 0);
        var result = await tuner.QuarantineAsync(run.ActiveDeployment, Regression(), prior);
        AssertNativePersistence(result);
        Assert.True(result.WasRollbackPersisted);
        Assert.Same(prior, result.RollbackDeployment);
        Assert.Same(prior, tuner.Deployment.Current);
        Assert.True(CreateTuner(new(), store, MeasurePassed).TryHydrate());

        string receiptPath = Assert.Single(Directory.GetFiles(Journal(), "*.quarantine.json"));
        byte[] original = File.ReadAllBytes(receiptPath);
        var stale = await tuner.QuarantineAsync(run.ActiveDeployment, Regression());
        Assert.False(stale.WasApplied);
        Assert.False(stale.WasPersisted);
        Assert.Null(stale.ReceiptPath);
        Assert.Same(prior, tuner.Deployment.Current);
        Assert.Equal(original, File.ReadAllBytes(receiptPath));
    }

    [Fact]
    public async Task Quarantine_WriteFailureBlocksAcrossWrappers_ButDoesNotClaimDurability()
    {
        var inner = new MemoryStore();
        var store = new QuarantinedKernelTuningStore<FakeKernelConfiguration>(Journal(), inner);
        var tuner = CreateTuner(new(), store, MeasurePassed);
        var run = await tuner.TuneAsync(Seeds());
        Directory.Move(Journal(), Journal("moved"));
        var result = await tuner.QuarantineAsync(run.ActiveDeployment, Regression());
        Assert.True(result.WasApplied);
        Assert.False(result.WasPersisted);
        Assert.Null(result.ReceiptPath);
        Assert.Null(tuner.Deployment.Current);
        var recreated = new QuarantinedKernelTuningStore<FakeKernelConfiguration>(Journal(), inner);
        Assert.False(CreateTuner(new(), recreated, MeasurePassed).TryHydrate());
    }

    [Theory]
    [InlineData("")]
    [InlineData("{invalid-json")]
    [InlineData("{\"Schema\":\"future-format\"}")]
    public void Quarantine_AnyExistingRecordDeniesWithoutParsing(string contents)
    {
        var snapshot = Snapshot(Identity(), Seeds()[2], 0);
        var codec = new FakeKernelCodec();
        var store = new QuarantinedKernelTuningStore<FakeKernelConfiguration>(Journal(), new FixedLoadStore(snapshot));
        string key = EvolutionHash.Combine(new[] { "tensor-kernel-quarantine-v1", Identity().StableKey,
            codec.Id, codec.VersionHash, snapshot.GenomeId });
        File.WriteAllText(Path.Combine(Journal(), key + ".quarantine.json"), contents);
        Assert.False(store.TryLoad(Identity(), codec, out _));
        Assert.False(store.TryPublish(new KernelTuningDeploymentRegistry<FakeKernelConfiguration>().GetOrCreate(Identity()),
            snapshot, codec, false));
    }

    [Fact]
    public async Task Quarantine_AdmissionAtPublicationRejectsPreviouslyLoadedWinner()
    {
        var inner = new MemoryStore();
        var store = new QuarantinedKernelTuningStore<FakeKernelConfiguration>(Journal(), inner);
        var tuner = CreateTuner(new(), store, MeasurePassed);
        var run = await tuner.TuneAsync(Seeds());
        Assert.True(store.TryLoad(Identity(), new FakeKernelCodec(), out var loaded));
        using var entered = new ManualResetEventSlim();
        using var release = new ManualResetEventSlim();
        var otherHandle = new KernelTuningDeploymentRegistry<FakeKernelConfiguration>().GetOrCreate(Identity());
        var codec = new CallbackQuarantineCodec(() =>
        {
            entered.Set();
            if (!release.Wait(TimeSpan.FromSeconds(10))) throw new TimeoutException();
        });
        Task<bool> publication = Task.Run(() => store.TryPublish(otherHandle, loaded!, codec, true));
        try
        {
            Assert.True(entered.Wait(TimeSpan.FromSeconds(10)));
            AssertNativePersistence(await tuner.QuarantineAsync(run.ActiveDeployment, Regression()));
        }
        finally { release.Set(); }
        Assert.False(await publication);
        Assert.Null(otherHandle.Current);
    }

    [Theory]
    [InlineData("same")]
    [InlineData("identity")]
    [InlineData("hash")]
    [InlineData("validator")]
    public async Task Quarantine_InvalidRollbackLeavesBuiltInFallback(string invalidity)
    {
        var registry = new KernelTuningDeploymentRegistry<FakeKernelConfiguration>();
        var store = new QuarantinedKernelTuningStore<FakeKernelConfiguration>(Journal());
        bool rejectPrior = false;
        var tuner = new EvolutionKernelAutotuner<FakeKernelConfiguration>(Identity(), new FakeKernelCodec(),
            new FakeKernelVariation(), MeasurePassed, Finalist(), EngineOptions(), deploymentRegistry: registry,
            store: store, deploymentValidator: c => !rejectPrior || c.Variant != FakeKernelVariant.Fast);
        var run = await tuner.TuneAsync(Seeds());
        var prior = Snapshot(Identity(), Seeds()[1], 0);
        if (invalidity == "same") prior = Snapshot(Identity(), run.ActiveDeployment.Configuration, 10);
        if (invalidity == "identity") prior = Snapshot(DifferentIdentity(), Seeds()[1], 0);
        if (invalidity == "hash") prior = new(prior.Identity, prior.Configuration, "wrong-hash", prior.Measurement,
            prior.RunStateHash, prior.PromotionEvidence, prior.EvidenceRole);
        if (invalidity == "validator") rejectPrior = true;
        var result = await tuner.QuarantineAsync(run.ActiveDeployment, Regression(), prior);
        AssertNativePersistence(result);
        Assert.Null(result.RollbackDeployment);
        Assert.False(result.WasRollbackPersisted);
        Assert.False(tuner.Deployment.TryGet(out _));
    }

    [Fact]
    public async Task Quarantine_CancellationAndInvalidExpectedDoNotMutate()
    {
        var store = new QuarantinedKernelTuningStore<FakeKernelConfiguration>(Journal());
        var tuner = CreateTuner(new(), store, MeasurePassed);
        var run = await tuner.TuneAsync(Seeds());
        using var canceled = new CancellationTokenSource();
        canceled.Cancel();
        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => tuner.QuarantineAsync(
            run.ActiveDeployment, Regression(), cancellationToken: canceled.Token));
        await Assert.ThrowsAsync<ArgumentNullException>(() => tuner.QuarantineAsync(null!, Regression()));
        await Assert.ThrowsAsync<ArgumentNullException>(() => tuner.QuarantineAsync(run.ActiveDeployment, null!));
        await Assert.ThrowsAsync<ArgumentException>(() => tuner.QuarantineAsync(Snapshot(DifferentIdentity(), Seeds()[2], 0), Regression()));
        var invalid = new KernelTuningDeploymentSnapshot<FakeKernelConfiguration>(Identity(), Seeds()[2], "bad-hash",
            run.ActiveDeployment.Measurement, "run", run.ActiveDeployment.PromotionEvidence, run.ActiveDeployment.EvidenceRole);
        await Assert.ThrowsAsync<ArgumentException>(() => tuner.QuarantineAsync(invalid, Regression()));
        Assert.Same(run.ActiveDeployment, tuner.Deployment.Current);
        Assert.Empty(Directory.GetFiles(Journal()));
    }

    [Fact]
    public async Task Quarantine_IsExplicit_AndHotReadsNeverConsultJournal()
    {
        var legacy = CreateTuner(new(), new MemoryStore(), MeasurePassed);
        var run = await legacy.TuneAsync(Seeds());
        await Assert.ThrowsAsync<InvalidOperationException>(() => legacy.QuarantineAsync(run.ActiveDeployment, Regression()));
        Assert.True(legacy.Deployment.TryDeactivate(run.ActiveDeployment));
        Assert.True(legacy.TryHydrate()); // Existing in-memory deactivation semantics remain unchanged.

        var store = new QuarantinedKernelTuningStore<FakeKernelConfiguration>(Journal());
        var tuner = CreateTuner(new(), store, MeasurePassed);
        var deployed = (await tuner.TuneAsync(Seeds())).ActiveDeployment;
        Directory.Move(Journal(), Journal("offline"));
        Assert.True(tuner.Deployment.TryGet(out var configuration));
        Assert.Equal(deployed.Configuration, configuration);
        Assert.False(store.TryLoad(Identity(), new FakeKernelCodec(), out _));
    }

    [Theory]
    [InlineData("relative/path")]
    [InlineData("")]
    public void Quarantine_JournalRequiresAbsolutePrivateDirectory(string path) =>
        Assert.Throws<ArgumentException>(() => new QuarantinedKernelTuningStore<FakeKernelConfiguration>(path));

    [Theory]
    [InlineData("policy")]
    [InlineData("digest")]
    [InlineData("reason")]
    [InlineData("observed")]
    [InlineData("limit")]
    [InlineData("time")]
    public void Quarantine_RegressionEvidenceRejectsInvalidFields(string field)
    {
        Assert.ThrowsAny<ArgumentException>(() => new KernelTuningRegressionEvidence(
            field == "reason" ? (KernelTuningRegressionReason)99 : KernelTuningRegressionReason.Latency,
            field == "policy" ? "bad\uD800" : "policy-v1", field == "digest" ? "bad" : new string('a', 64),
            field == "observed" ? double.NaN : 2, field == "limit" ? -1 : 1,
            field == "time" ? default : DateTimeOffset.UtcNow));
    }

    [Theory]
    [InlineData("oversize")]
    [InlineData("unicode")]
    [InlineData("empty-id")]
    [InlineData("version-change")]
    [InlineData("exception")]
    public void Quarantine_RejectsUnboundedOrUnstableCodecs(string failure)
    {
        var snapshot = Snapshot(Identity(), Seeds()[2], 0);
        var store = new QuarantinedKernelTuningStore<FakeKernelConfiguration>(Journal(), new FixedLoadStore(snapshot));
        var codec = new CallbackQuarantineCodec(() => { if (failure == "exception") throw new IOException(); })
        {
            Payload = failure == "oversize" ? new string('x', 65537) : failure == "unicode" ? "\uD800" : null,
            EmptyId = failure == "empty-id",
            ChangeVersion = failure == "version-change"
        };
        Assert.False(store.TryLoad(Identity(), codec, out _));
        Assert.False(store.CanDeploy(Identity(), snapshot.Configuration, codec));
    }

    private static KernelTuningIdentity DifferentIdentity() => new(Identity().Kernel, Identity().Shape,
        Identity().Device, Identity().Backend, new KernelSearchSpaceVersion(999), Identity().BenchmarkProtocolVersion);

    [Fact]
    public async Task Quarantine_HydrationPublicationCannotRacePastReceipt()
    {
        var inner = new MemoryStore();
        var store = new QuarantinedKernelTuningStore<FakeKernelConfiguration>(Journal(), inner);
        var tuner = CreateTuner(new(), store, MeasurePassed);
        var run = await tuner.TuneAsync(Seeds());
        using var entered = new ManualResetEventSlim();
        using var release = new ManualResetEventSlim();
        int serializations = 0;
        var codec = new CallbackQuarantineCodec(() =>
        {
            if (++serializations != 4) return; // TryLoad, CanDeploy, canonical hash, final publication.
            entered.Set();
            if (!release.Wait(TimeSpan.FromSeconds(10))) throw new TimeoutException();
        });
        var loading = new EvolutionKernelAutotuner<FakeKernelConfiguration>(Identity(), codec,
            new FakeKernelVariation(), MeasurePassed, Finalist(), EngineOptions(), store: store);
        Task<bool> hydration = Task.Run(loading.TryHydrate);
        try
        {
            Assert.True(entered.Wait(TimeSpan.FromSeconds(10)));
            AssertNativePersistence(await tuner.QuarantineAsync(run.ActiveDeployment, Regression()));
        }
        finally { release.Set(); }
        Assert.False(await hydration);
        Assert.Null(loading.Deployment.Current);
    }

    [Fact]
    public async Task Quarantine_ExplicitHydrationRechecksOtherActiveHandle()
    {
        var store = new QuarantinedKernelTuningStore<FakeKernelConfiguration>(Journal());
        var tuner = CreateTuner(new(), store, MeasurePassed);
        var run = await tuner.TuneAsync(Seeds());
        var other = CreateTuner(new(), store, MeasurePassed);
        Assert.True(other.TryHydrate());
        await tuner.QuarantineAsync(run.ActiveDeployment, Regression());
        Assert.True(other.Deployment.TryGet(out _)); // No implicit dispatch-time filesystem check.
        Assert.False(other.TryHydrate());
        Assert.Null(other.Deployment.Current);
    }

    [Fact]
    public async Task Quarantine_RollbackStoreFailureDoesNotUndoSafeInMemoryRollback()
    {
        var active = Snapshot(Identity(), Seeds()[2], 0);
        var store = new QuarantinedKernelTuningStore<FakeKernelConfiguration>(Journal(), new FailWriteQuarantineStore(active));
        var tuner = CreateTuner(new(), store, MeasurePassed);
        Assert.True(tuner.TryHydrate());
        var prior = Snapshot(Identity(), Seeds()[1], 0);
        var result = await tuner.QuarantineAsync(active, Regression(), prior);
        Assert.True(result.WasApplied);
        AssertNativePersistence(result);
        Assert.False(result.WasRollbackPersisted);
        Assert.Same(prior, tuner.Deployment.Current);
    }

    [Fact]
    public async Task Quarantine_ExistingReceiptIsNotOverwrittenOrClaimedAsNewPersistence()
    {
        var active = Snapshot(Identity(), Seeds()[2], 0);
        var store = new QuarantinedKernelTuningStore<FakeKernelConfiguration>(Journal(), new FixedLoadStore(active));
        var tuner = CreateTuner(new(), store, MeasurePassed);
        Assert.True(tuner.TryHydrate());
        var codec = new FakeKernelCodec();
        string key = EvolutionHash.Combine(new[] { "tensor-kernel-quarantine-v1", Identity().StableKey,
            codec.Id, codec.VersionHash, active.GenomeId });
        string path = Path.Combine(Journal(), key + ".quarantine.json");
        File.WriteAllText(path, "first receipt");
        var result = await tuner.QuarantineAsync(active, Regression());
        Assert.True(result.WasApplied);
        Assert.False(result.WasPersisted);
        Assert.Equal("first receipt", File.ReadAllText(path));
        Assert.Single(Directory.GetFiles(Journal(), "*.pending"));
        Assert.Null(tuner.Deployment.Current);
    }

    [Theory]
    [InlineData("throw")]
    [InlineData("identity")]
    public void Quarantine_StoreFailureOrWrongIdentityCannotHydrate(string failure)
    {
        IKernelTuningStore<FakeKernelConfiguration> inner = failure == "throw" ? new ThrowingStore() :
            new FixedLoadStore(Snapshot(DifferentIdentity(), Seeds()[2], 0));
        var store = new QuarantinedKernelTuningStore<FakeKernelConfiguration>(Journal(), inner);
        Assert.False(store.TryLoad(Identity(), new FakeKernelCodec(), out _));
    }

    [Theory]
    [InlineData("spaces")]
    [InlineData("long")]
    [InlineData("control")]
    [InlineData("equal")]
    [InlineData("infinite")]
    [InlineData("limit-nan")]
    [InlineData("uppercase-digest")]
    public void Quarantine_EvidenceRejectsAdditionalBoundaryValues(string field)
    {
        string policy = field == "spaces" ? " " : field == "long" ? new string('x', 257) :
            field == "control" ? "bad\nlabel" : "v1";
        Assert.ThrowsAny<ArgumentException>(() => new KernelTuningRegressionEvidence(
            KernelTuningRegressionReason.Latency, policy, new string(field == "uppercase-digest" ? 'A' : 'a', 64),
            field == "equal" ? 1 : field == "infinite" ? double.PositiveInfinity : 2,
            field == "limit-nan" ? double.NaN : 1, DateTimeOffset.UtcNow));
    }

    [Fact]
    public void Quarantine_RejectsRootAndDriveRelativePaths()
    {
        Assert.Throws<ArgumentException>(() => new QuarantinedKernelTuningStore<FakeKernelConfiguration>(Path.GetPathRoot(Journal())!));
        if (Path.DirectorySeparatorChar != '\\') return;
        Assert.Throws<ArgumentException>(() => new QuarantinedKernelTuningStore<FakeKernelConfiguration>("C:relative"));
        Assert.Throws<ArgumentException>(() => new QuarantinedKernelTuningStore<FakeKernelConfiguration>("\\relative"));
    }

    [Fact]
    public void Quarantine_PayloadByteLimitAndNullArgumentsFailClosed()
    {
        var snapshot = Snapshot(Identity(), Seeds()[2], 0);
        var store = new QuarantinedKernelTuningStore<FakeKernelConfiguration>(Journal(), new FixedLoadStore(snapshot));
        var codec = new CallbackQuarantineCodec(() => { }) { Payload = new string('\u20ac', 30000) };
        Assert.False(store.TryLoad(Identity(), codec, out _));
        Assert.False(store.TryLoad(null!, new FakeKernelCodec(), out _));
        Assert.False(store.TryStore(null!, new FakeKernelCodec()));
        Assert.False(store.TryStore(snapshot, null!));
        Assert.False(store.CanDeploy(null!, snapshot.Configuration, new FakeKernelCodec()));
        Assert.False(store.TryPublish(new KernelTuningDeploymentRegistry<FakeKernelConfiguration>().GetOrCreate(Identity()),
            snapshot, codec, true));
    }

    private sealed class FailWriteQuarantineStore : IKernelTuningStore<FakeKernelConfiguration>
    {
        private readonly KernelTuningDeploymentSnapshot<FakeKernelConfiguration> _snapshot;
        internal FailWriteQuarantineStore(KernelTuningDeploymentSnapshot<FakeKernelConfiguration> snapshot) => _snapshot = snapshot;
        public bool TryLoad(KernelTuningIdentity identity, IEvolutionGenomeCodec<FakeKernelConfiguration> codec,
            out KernelTuningDeploymentSnapshot<FakeKernelConfiguration>? snapshot)
        { snapshot = _snapshot; return true; }
        public bool TryStore(KernelTuningDeploymentSnapshot<FakeKernelConfiguration> snapshot,
            IEvolutionGenomeCodec<FakeKernelConfiguration> codec) => false;
    }

    private sealed class CallbackQuarantineCodec : IEvolutionGenomeCodec<FakeKernelConfiguration>
    {
        private readonly Action _callback;
        private int _versionReads;
        internal CallbackQuarantineCodec(Action callback) => _callback = callback;
        internal string? Payload { get; set; }
        internal bool EmptyId { get; set; }
        internal bool ChangeVersion { get; set; }
        public string Id => EmptyId ? "" : new FakeKernelCodec().Id;
        public string VersionHash => ChangeVersion ? (++_versionReads).ToString() : new FakeKernelCodec().VersionHash;
        public string Serialize(FakeKernelConfiguration configuration) { _callback(); return Payload ?? new FakeKernelCodec().Serialize(configuration); }
        public FakeKernelConfiguration Deserialize(string payload) => new FakeKernelCodec().Deserialize(payload);
    }
}
