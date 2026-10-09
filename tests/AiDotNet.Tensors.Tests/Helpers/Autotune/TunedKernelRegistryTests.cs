using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using AiDotNet.Tensors.Helpers.Autotune.TunedKernels;
using Xunit;

namespace AiDotNet.Tensors.Tests.Helpers.Autotune;

[CollectionDefinition(Name, DisableParallelization = true)]
public sealed class TunedKernelRegistryCollection
{
    public const string Name = "TunedKernelRegistry";
}

/// <summary>
/// The tuned-kernel registry's selection logic, exercised with synthetic candidates whose output and device time
/// are scripted, so every gate rule is checked deterministically without a GPU.
/// </summary>
[Collection(TunedKernelRegistryCollection.Name)]
public sealed class TunedKernelRegistryTests : IDisposable
{
    private readonly TunedKernelMode? _mode = TunedKernelPolicy.ModeOverride;
    private readonly bool _persist = TunedKernelPolicy.PersistDecisions;
    private readonly double _budget = TunedKernelPolicy.BudgetMilliseconds;

    public TunedKernelRegistryTests()
    {
        TunedKernelPolicy.ModeOverride = TunedKernelMode.Tune;
        TunedKernelPolicy.PersistDecisions = false;
        TunedKernelPolicy.BudgetMilliseconds = 60_000;
        TunedKernelPolicy.ResetBudget();
        TunedKernelPolicy.SetOverride(TunedKernelOp.Elementwise, null);
        TunedKernelProfiles.ClearRuntimeProfiles();
    }

    public void Dispose()
    {
        TunedKernelPolicy.ModeOverride = _mode;
        TunedKernelPolicy.PersistDecisions = _persist;
        TunedKernelPolicy.BudgetMilliseconds = _budget;
        TunedKernelPolicy.ResetBudget();
        TunedKernelPolicy.SetOverride(TunedKernelOp.Elementwise, null);
        TunedKernelProfiles.ClearRuntimeProfiles();
    }

    private struct Args
    {
        public float[] Output;
    }

    private sealed class Fake : ITunedKernelCandidate<Args>
    {
        public Fake(string id, float value, double ms, bool deterministic = true, int maxColumns = int.MaxValue)
        {
            Id = id; Value = value; Milliseconds = ms; IsDeterministic = deterministic; MaxColumns = maxColumns;
        }
        public float Value { get; set; }
        public bool ThrowOnExecute { get; set; }
        public string Id { get; }
        public TunedKernelOrigin Origin => TunedKernelOrigin.Generated;
        public bool IsDeterministic { get; }
        public double Milliseconds { get; }
        public int MaxColumns { get; }
        public int Executions { get; private set; }
        public bool IsApplicable(in TunedShape shape) => shape[1] <= MaxColumns;
        public void Execute(in Args args)
        {
            Executions++;
            if (ThrowOnExecute) throw new InvalidOperationException(Id + " failed at dispatch");
            for (int i = 0; i < args.Output.Length; i++) args.Output[i] = Value + i;
        }
    }

    private sealed class Harness : ITunedKernelHarness<Args>
    {
        public bool Measurable { get; set; } = true;
        public int Measurements { get; private set; }
        public bool CanMeasure(in Args args) => Measurable;
        public double RelativeTolerance => 1e-4;
        public float[] SnapshotOutput(in Args args) => (float[])args.Output.Clone();
        public void PoisonOutput(in Args args)
        {
            for (int i = 0; i < args.Output.Length; i++) args.Output[i] = float.NaN;
        }
        public double MeasureMilliseconds(ITunedKernelCandidate<Args> candidate, in Args args, int repetitions)
        {
            Measurements++;
            candidate.Execute(args);
            return ((Fake)candidate).Milliseconds * repetitions;
        }
    }

    private static TunedKernelSlot<Args> Slot(Harness harness, bool deterministic, params Fake[] candidates) =>
        new(TunedKernelOp.Elementwise, "test:device", harness, () => deterministic, candidates);

    private static TunedShape Shape(int rows = 4, int cols = 8) => TunedShape.Of2(TunedKernelDType.Float32, rows, cols);

    private static Args NewArgs() => new() { Output = new float[16] };

    [Fact]
    public void FasterCorrectCandidate_IsPromoted_WithEvidence()
    {
        var reference = new Fake("ref", 1f, 1.0);
        var fast = new Fake("fast", 1f, 0.5);
        var slot = Slot(new Harness(), false, reference, fast);

        var chosen = slot.Resolve(Shape(), NewArgs());

        Assert.Same(fast, chosen);
        var decision = Assert.Single(slot.CachedDecisions);
        Assert.Equal(TunedKernelDecisionReason.Tuned, decision.Reason);
        Assert.Equal("ref", decision.ReferenceId);
        Assert.True(decision.MedianSpeedup > 1.9 && decision.MedianSpeedup < 2.1, decision.ToString());
        Assert.True(decision.LowerSpeedupBound > 1.0);
        Assert.Equal(0.0, decision.MaxRelativeError);
    }

    [Fact]
    public void ACachedTunedCandidateThatFailsAtDispatch_IsRetired_AndTheCallerFallsBack()
    {
        var reference = new Fake("ref", 1f, 1.0);
        var fast = new Fake("fast", 1f, 0.5);
        var slot = Slot(new Harness(), false, reference, fast);
        Assert.Same(fast, slot.Resolve(Shape(), NewArgs()));

        fast.ThrowOnExecute = true;   // e.g. its module was evicted and the call is under capture
        Assert.False(slot.TryExecute(Shape(), NewArgs()));   // the engine op then runs its own reference path
        Assert.Empty(slot.CachedDecisions);

        Assert.Same(reference, slot.Resolve(Shape(), NewArgs()));
        var decision = Assert.Single(slot.CachedDecisions);
        Assert.Equal("ref", decision.CandidateId);
        Assert.Contains(decision.Rejected, r => r.StartsWith("fast:", StringComparison.Ordinal));
        Assert.True(slot.TryExecute(Shape(), NewArgs()));
    }

    [Fact]
    public void AReferenceThatFailsAtDispatch_StillThrows()
    {
        var reference = new Fake("ref", 1f, 1.0);
        var slot = Slot(new Harness(), false, reference);
        Assert.Same(reference, slot.Resolve(Shape(), NewArgs()));
        reference.ThrowOnExecute = true;
        Assert.Throws<InvalidOperationException>(() => slot.TryExecute(Shape(), NewArgs()));
    }

    [Fact]
    public void ANonFiniteReferenceOutput_IsNotCached_SoALaterCallTunesTheShape()
    {
        var reference = new Fake("ref", float.NaN, 1.0);
        var fast = new Fake("fast", float.NaN, 0.5);
        var slot = Slot(new Harness(), false, reference, fast);

        Assert.Same(reference, slot.Resolve(Shape(), NewArgs()));
        Assert.Empty(slot.CachedDecisions);   // nothing cached, so nothing persisted either

        reference.Value = 1f;
        fast.Value = 1f;
        Assert.Same(fast, slot.Resolve(Shape(), NewArgs()));
        Assert.Equal(TunedKernelDecisionReason.Tuned, Assert.Single(slot.CachedDecisions).Reason);
    }

    [Fact]
    public void FasterWrongCandidate_IsRejected_BeforeTiming()
    {
        var reference = new Fake("ref", 1f, 1.0);
        var wrong = new Fake("wrong", 2f, 0.1);
        var slot = Slot(new Harness(), false, reference, wrong);

        Assert.Same(reference, slot.Resolve(Shape(), NewArgs()));
        var decision = Assert.Single(slot.CachedDecisions);
        Assert.Equal(TunedKernelDecisionReason.ReferenceWon, decision.Reason);
        Assert.Contains(decision.Rejected, r => r.StartsWith("wrong: error", StringComparison.Ordinal));
    }

    private sealed class NoOp : ITunedKernelCandidate<Args>
    {
        public string Id => "noop";
        public TunedKernelOrigin Origin => TunedKernelOrigin.External;
        public bool IsDeterministic => true;
        public bool IsApplicable(in TunedShape shape) => true;
        public void Execute(in Args args) { }
    }

    private sealed class NoOpHarness : ITunedKernelHarness<Args>
    {
        private readonly Harness _inner = new();
        public bool CanMeasure(in Args args) => true;
        public double RelativeTolerance => 1e-4;
        public void PoisonOutput(in Args args) => _inner.PoisonOutput(args);
        public float[] SnapshotOutput(in Args args) => _inner.SnapshotOutput(args);
        public double MeasureMilliseconds(ITunedKernelCandidate<Args> candidate, in Args args, int repetitions) =>
            candidate is NoOp ? 0.001 * repetitions : _inner.MeasureMilliseconds(candidate, args, repetitions);
    }

    [Fact]
    public void CandidateThatWritesNothing_IsRejected_NotPromotedOnTheReferencesOutput()
    {
        var reference = new Fake("ref", 1f, 1.0);
        var slot = new TunedKernelSlot<Args>(TunedKernelOp.Elementwise, "test:device", new NoOpHarness(),
            () => false, reference);
        slot.AddCandidate(new NoOp());

        Assert.Same(reference, slot.Resolve(Shape(), NewArgs()));
        var decision = Assert.Single(slot.CachedDecisions);
        Assert.Equal(TunedKernelDecisionReason.ReferenceWon, decision.Reason);
        Assert.Contains(decision.Rejected, r => r.StartsWith("noop: error", StringComparison.Ordinal));
    }

    [Fact]
    public void MarginalCandidate_BelowTheSpeedupFloor_IsNotPromoted()
    {
        var reference = new Fake("ref", 1f, 1.0);
        var marginal = new Fake("marginal", 1f, 0.99);
        var slot = Slot(new Harness(), false, reference, marginal);

        Assert.Same(reference, slot.Resolve(Shape(), NewArgs()));
        Assert.Equal(TunedKernelDecisionReason.ReferenceWon, Assert.Single(slot.CachedDecisions).Reason);
    }

    [Fact]
    public void FastestOfSeveralPassingCandidates_Wins()
    {
        var reference = new Fake("ref", 1f, 1.0);
        var a = new Fake("a", 1f, 0.6);
        var b = new Fake("b", 1f, 0.3);
        var c = new Fake("c", 1f, 0.5);
        var slot = Slot(new Harness(), false, reference, a, b, c);

        Assert.Same(b, slot.Resolve(Shape(), NewArgs()));
    }

    [Fact]
    public void Decision_IsCachedPerShape_AndNotRemeasured()
    {
        var harness = new Harness();
        var reference = new Fake("ref", 1f, 1.0);
        var fast = new Fake("fast", 1f, 0.5);
        var slot = Slot(harness, false, reference, fast);

        slot.Resolve(Shape(), NewArgs());
        int measured = harness.Measurements;
        for (int i = 0; i < 5; i++) Assert.Same(fast, slot.Resolve(Shape(), NewArgs()));
        Assert.Equal(measured, harness.Measurements);

        slot.Resolve(Shape(cols: 9), NewArgs());
        Assert.True(harness.Measurements > measured, "a new shape class is measured");
        Assert.Equal(2, slot.CachedDecisions.Count);
    }

    [Fact]
    public void Unmeasurable_ServesReference_WithoutCaching()
    {
        var harness = new Harness { Measurable = false };
        var reference = new Fake("ref", 1f, 1.0);
        var fast = new Fake("fast", 1f, 0.5);
        var slot = Slot(harness, false, reference, fast);

        Assert.Same(reference, slot.Resolve(Shape(), NewArgs()));
        Assert.Empty(slot.CachedDecisions);

        harness.Measurable = true;
        Assert.Same(fast, slot.Resolve(Shape(), NewArgs()));
    }

    [Fact]
    public void DeterministicMode_ExcludesNonDeterministicCandidates_AndCachesSeparately()
    {
        bool deterministic = true;
        var reference = new Fake("ref", 1f, 1.0);
        var racy = new Fake("racy", 1f, 0.2, deterministic: false);
        var slot = new TunedKernelSlot<Args>(TunedKernelOp.Elementwise, "test:device", new Harness(),
            () => deterministic, reference, racy);

        Assert.Same(reference, slot.Resolve(Shape(), NewArgs()));
        deterministic = false;
        Assert.Same(racy, slot.Resolve(Shape(), NewArgs()));
        deterministic = true;
        Assert.Same(reference, slot.Resolve(Shape(), NewArgs()));
    }

    [Fact]
    public void Override_PicksTheNamedCandidate_WithoutMeasuring()
    {
        var harness = new Harness();
        var reference = new Fake("ref", 1f, 1.0);
        var slow = new Fake("slow", 1f, 5.0);
        var slot = Slot(harness, false, reference, slow);
        TunedKernelPolicy.SetOverride(TunedKernelOp.Elementwise, "slow");

        Assert.Same(slow, slot.Resolve(Shape(), NewArgs()));
        Assert.Equal(0, harness.Measurements);
        Assert.Equal(TunedKernelDecisionReason.Override, Assert.Single(slot.CachedDecisions).Reason);
    }

    [Fact]
    public void PinnedMode_UsesProfileThenReference_NeverMeasures()
    {
        TunedKernelPolicy.ModeOverride = TunedKernelMode.Pinned;
        var harness = new Harness();
        var reference = new Fake("ref", 1f, 1.0);
        var fast = new Fake("fast", 1f, 0.5);
        var slot = Slot(harness, false, reference, fast);

        Assert.Same(reference, slot.Resolve(Shape(), NewArgs()));
        Assert.Equal(TunedKernelDecisionReason.Pinned, Assert.Single(slot.CachedDecisions).Reason);

        TunedKernelProfiles.AddProfile(
            "{\"schemaVersion\":1,\"device\":\"test:*\",\"entries\":[{\"op\":\"Elementwise\",\"shape\":\"f32:4x9\"," +
            "\"deterministic\":true,\"candidate\":\"fast\",\"speedup\":2.0}]}");
        Assert.Same(fast, slot.Resolve(Shape(cols: 9), NewArgs()));
        Assert.Equal(0, harness.Measurements);
    }

    [Fact]
    public void ProfileEntry_NamingAnInapplicableCandidate_FallsThrough()
    {
        TunedKernelProfiles.AddProfile(
            "{\"schemaVersion\":1,\"device\":\"test:device\",\"entries\":[{\"op\":\"Elementwise\",\"shape\":\"f32:4x64\"," +
            "\"deterministic\":true,\"candidate\":\"narrow\",\"speedup\":2.0}]}");
        var reference = new Fake("ref", 1f, 1.0);
        var narrow = new Fake("narrow", 1f, 0.5, maxColumns: 32);
        var slot = Slot(new Harness(), false, reference, narrow);

        Assert.Same(reference, slot.Resolve(Shape(cols: 64), NewArgs()));
        Assert.Equal(TunedKernelDecisionReason.OnlyCandidate, Assert.Single(slot.CachedDecisions).Reason);
    }

    [Fact]
    public void ExhaustedBudget_KeepsTheReference()
    {
        TunedKernelPolicy.BudgetMilliseconds = 0;
        var harness = new Harness();
        var slot = Slot(harness, false, new Fake("ref", 1f, 1.0), new Fake("fast", 1f, 0.5));

        Assert.Equal("ref", slot.Resolve(Shape(), NewArgs())?.Id);
        Assert.Equal(0, harness.Measurements);
        Assert.Equal(TunedKernelDecisionReason.BudgetExhausted, Assert.Single(slot.CachedDecisions).Reason);
    }

    [Fact]
    public void AddCandidate_InvalidatesDecisions_AndCannotReplaceTheReference()
    {
        var reference = new Fake("ref", 1f, 1.0);
        var slot = Slot(new Harness(), false, reference, new Fake("a", 1f, 0.9));
        slot.Resolve(Shape(), NewArgs());
        Assert.Single(slot.CachedDecisions);

        var external = new Fake("external.x@1", 1f, 0.25);
        slot.AddCandidate(external);
        Assert.Empty(slot.CachedDecisions);
        Assert.Same(external, slot.Resolve(Shape(), NewArgs()));
        Assert.Throws<InvalidOperationException>(() => slot.AddCandidate(new Fake("ref", 1f, 0.1)));
    }

    [Fact]
    public void NoApplicableCandidate_ReturnsNull()
    {
        var slot = Slot(new Harness(), false, new Fake("ref", 1f, 1.0, maxColumns: 4));
        Assert.Null(slot.Resolve(Shape(cols: 8), NewArgs()));
        Assert.False(slot.TryExecute(Shape(cols: 8), NewArgs()));
    }

    [Fact]
    public void TunedShape_EqualityHashAndTextRoundTrip()
    {
        var a = TunedShape.Create(TunedKernelDType.Float32, new[] { 8192, 32, 0, -1 });
        var b = TunedShape.Create(TunedKernelDType.Float32, new[] { 8192, 32, 0, -1 });
        var c = TunedShape.Create(TunedKernelDType.Float64, new[] { 8192, 32, 0, -1 });
        var d = TunedShape.Create(TunedKernelDType.Float32, new[] { 8192, 32, 0 });
        Assert.Equal(a, b);
        Assert.Equal(a.GetHashCode(), b.GetHashCode());
        Assert.NotEqual(a, c);
        Assert.NotEqual(a, d);
        Assert.Equal("f32:8192x32x0x-1", a.ToString());
        Assert.True(TunedShape.TryParse(a.ToString(), out var parsed));
        Assert.Equal(a, parsed);
        Assert.False(TunedShape.TryParse("q9:1x2", out _));
        Assert.Throws<ArgumentOutOfRangeException>(() => TunedShape.Create(1, new int[TunedShape.Capacity + 1]));
    }

    [Fact]
    public void ExportedProfile_RoundTripsThroughLookup()
    {
        var reference = new Fake("ref", 1f, 1.0);
        var fast = new Fake("fast", 1f, 0.5);
        var slot = Slot(new Harness(), true, reference, fast);
        slot.Resolve(Shape(), NewArgs());

        string json = TunedKernelProfiles.ExportProfile("test:device", slot.CachedDecisions, deterministic: true);
        TunedKernelProfiles.AddProfile(json);
        Assert.True(TunedKernelProfiles.TryLookup(TunedKernelOp.Elementwise, "test:device", Shape(), true, out var id));
        Assert.Equal("fast", id);
        Assert.False(TunedKernelProfiles.TryLookup(TunedKernelOp.Elementwise, "other:device", Shape(), true, out _));
    }

    [Fact]
    public void ShippedProfile_IsEmbedded_AndServesTheDevGpu()
    {
        Assert.True(TunedKernelProfiles.TryLookup(TunedKernelOp.Softmax, "cuda:sm75:NVIDIA GeForce GTX 1660 Ti",
            TunedShape.Of2(TunedKernelDType.Float32, 8192, 32), deterministic: true, out var id));
        Assert.Equal("generated.softmax.lanes16", id);
        Assert.False(TunedKernelProfiles.TryLookup(TunedKernelOp.Softmax, "cuda:sm86:NVIDIA GeForce RTX 3090",
            TunedShape.Of2(TunedKernelDType.Float32, 8192, 32), deterministic: true, out _));
    }

    private static ExternalKernelArtifact ValidArtifact()
    {
        const string ptx = ".version 6.4\n.target sm_75\n.address_size 64\n.visible .entry evolved_softmax(\n)\n{\nret;\n}\n";
        return new ExternalKernelArtifact
        {
            SchemaVersion = 1,
            Id = "evolved.softmax",
            Version = "3",
            Abi = "row-softmax-v1",
            Target = "sm_75",
            EntryPoint = "evolved_softmax",
            Ptx = ptx,
            PtxSha256 = ExternalKernelArtifact.ComputePtxSha256(ptx),
            Deterministic = true,
            Launch = new ExternalKernelLaunch { BlockX = 256, GridRule = ExternalKernelGridRule.RowsPerBlock, UnitsPerBlock = 8 },
            Constraints = new ExternalKernelConstraints { MinColumns = 1, MaxColumns = 1024 },
        };
    }

    [Fact]
    public void ExternalArtifact_RoundTrips_AndMapsToItsOpFamily()
    {
        var artifact = ValidArtifact();
        var parsed = ExternalKernelArtifact.Parse(artifact.ToJson());
        Assert.Equal(TunedKernelOp.Softmax, parsed.Op);
        Assert.Equal("external.evolved.softmax@3", parsed.CandidateId);
        Assert.Equal(75, parsed.TargetSm);
        Assert.True(parsed.Supports(Shape(cols: 1024)));
        Assert.False(parsed.Supports(Shape(cols: 1025)));
    }

    [Fact]
    public void ExternalArtifact_RejectsTamperedOrMalformedDocuments()
    {
        var tampered = ValidArtifact();
        tampered.Ptx += "// edited\n";
        Assert.Throws<InvalidDataException>(() => ExternalKernelArtifact.Parse(tampered.ToJson()));

        var unknownAbi = ValidArtifact();
        unknownAbi.Abi = "conv-magic-v9";
        Assert.Throws<InvalidDataException>(() => ExternalKernelArtifact.Parse(unknownAbi.ToJson()));

        var missingEntry = ValidArtifact();
        missingEntry.EntryPoint = "other_name";
        Assert.Throws<InvalidDataException>(() => ExternalKernelArtifact.Parse(missingEntry.ToJson()));

        var hugeBlock = ValidArtifact();
        hugeBlock.Launch.BlockX = 2048;
        Assert.Throws<InvalidDataException>(() => ExternalKernelArtifact.Parse(hugeBlock.ToJson()));

        var badSchema = ValidArtifact();
        badSchema.SchemaVersion = 2;
        Assert.Throws<InvalidDataException>(() => ExternalKernelArtifact.Parse(badSchema.ToJson()));

        Assert.Throws<InvalidDataException>(() => ExternalKernelArtifact.Parse("{ not json"));
    }

    [Fact]
    public void ExternalArtifact_DirectoryLoad_ReportsInvalidFilesAndLoadsValidOnes()
    {
        string dir = Path.Combine(Path.GetTempPath(), "tuned-kernel-artifacts-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        try
        {
            File.WriteAllText(Path.Combine(dir, "a.kernel.json"), ValidArtifact().ToJson());
            var bad = ValidArtifact();
            bad.PtxSha256 = new string('0', 64);
            File.WriteAllText(Path.Combine(dir, "b.kernel.json"), bad.ToJson());
            File.WriteAllText(Path.Combine(dir, "ignored.txt"), "not an artifact");

            var errors = new List<string>();
            var loaded = ExternalKernelArtifact.LoadDirectory(dir, errors);
            Assert.Single(loaded);
            Assert.Single(errors);
            Assert.StartsWith("b.kernel.json", errors[0]);
        }
        finally
        {
            Directory.Delete(dir, recursive: true);
        }
    }
}
