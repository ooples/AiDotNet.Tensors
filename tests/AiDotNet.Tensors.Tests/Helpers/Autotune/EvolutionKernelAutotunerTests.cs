using System.Globalization;
using AiDotNet.Evolution;
using AiDotNet.Tensors.Helpers.Autotune;
using Xunit;

namespace AiDotNet.Tensors.Tests.Helpers.Autotune;

/// <summary>
/// GPU-free contract tests for the typed evolution adapter. The benchmark delegate is the backend seam: these tests
/// use deterministic CPU values, while CUDA/PTX consumers pass the same adapter a delegate that launches and times
/// the real kernel on its device.
/// </summary>
public sealed class EvolutionKernelAutotunerTests
{
    [Fact]
    public async Task TuneAsync_SelectsAndPublishesTheFastestTypedConfiguration()
    {
        var deployment = new KernelTuningDeployment<FakeKernelConfiguration>();
        var tuner = CreateTuner(
            deployment,
            (configuration, _, _) => new ValueTask<KernelTuningMeasurement>(Measure(configuration)));

        Assert.False(deployment.TryGet(out _));

        EvolutionKernelTuningResult<FakeKernelConfiguration> result = await tuner.TuneAsync(Seeds());

        Assert.True(deployment.TryGet(out FakeKernelConfiguration active));
        Assert.Equal(FakeKernelVariant.Wide, active.Variant);
        Assert.Equal(32, active.TileEdge);
        Assert.Equal(active, result.Deployment.Configuration);
        Assert.Equal(240.0, result.Deployment.MeasuredGflops);
        Assert.Equal(3, result.Run.Counters.EvaluationAttempts);
        Assert.Equal(Identity().StableKey, result.Deployment.Identity.StableKey);
    }

    [Fact]
    public async Task BackgroundTune_KeepsServingThePreviousWinnerUntilTheReplacementCompletes()
    {
        var deployment = new KernelTuningDeployment<FakeKernelConfiguration>();
        EvolutionKernelAutotuner<FakeKernelConfiguration> initial = CreateTuner(
            deployment,
            (configuration, _, _) => new ValueTask<KernelTuningMeasurement>(Measure(configuration)));
        await initial.TuneAsync(Seeds());
        KernelTuningDeploymentSnapshot<FakeKernelConfiguration> previous = deployment.Current!;

        var measurementStarted = new TaskCompletionSource<bool>(TaskCreationOptions.RunContinuationsAsynchronously);
        var releaseMeasurement = new TaskCompletionSource<bool>(TaskCreationOptions.RunContinuationsAsynchronously);
        EvolutionKernelAutotuner<FakeKernelConfiguration> replacement = CreateTuner(
            deployment,
            async (configuration, _, cancellationToken) =>
            {
                measurementStarted.TrySetResult(true);
                using (cancellationToken.Register(() => releaseMeasurement.TrySetCanceled()))
                    await releaseMeasurement.Task;
                return Measure(configuration, bonus: 50);
            });

        Task<EvolutionKernelTuningResult<FakeKernelConfiguration>> pending =
            replacement.TuneInBackgroundAsync(Seeds());
        await measurementStarted.Task;

        Assert.Same(previous, deployment.Current);
        Assert.True(deployment.TryGet(out FakeKernelConfiguration whileTuning));
        Assert.Equal(previous.Configuration, whileTuning);

        releaseMeasurement.TrySetResult(true);
        EvolutionKernelTuningResult<FakeKernelConfiguration> completed = await pending;

        Assert.NotSame(previous, deployment.Current);
        Assert.Same(completed.Deployment, deployment.Current);
        Assert.Equal(290.0, completed.Deployment.MeasuredGflops);
    }

    [Fact]
    public void Identity_CoversDeviceShapeAndBothProtocolVersions()
    {
        KernelTuningIdentity baseline = Identity();
        var otherShape = new KernelTuningIdentity(
            baseline.Kernel,
            new ShapeProfile(1, 64, 64, 1024),
            baseline.Device,
            baseline.SearchSpaceVersion,
            baseline.BenchmarkVersion);
        var otherDevice = new KernelTuningIdentity(
            baseline.Kernel,
            baseline.Shape,
            new GpuDeviceFingerprint(GpuVendorKind.Nvidia, "Fake GPU", 8, 6, 550, "fake-1"),
            baseline.SearchSpaceVersion,
            baseline.BenchmarkVersion);
        var otherSearchSpace = new KernelTuningIdentity(
            baseline.Kernel, baseline.Shape, baseline.Device, "space-v2", baseline.BenchmarkVersion);
        var otherBenchmark = new KernelTuningIdentity(
            baseline.Kernel, baseline.Shape, baseline.Device, baseline.SearchSpaceVersion, "protocol-v2");

        Assert.NotEqual(baseline.StableKey, otherShape.StableKey);
        Assert.NotEqual(baseline.StableKey, otherDevice.StableKey);
        Assert.NotEqual(baseline.StableKey, otherSearchSpace.StableKey);
        Assert.NotEqual(baseline.StableKey, otherBenchmark.StableKey);
    }

    [Fact]
    public void Measurement_DefensivelyCopiesExtensibleMetricsAndRejectsReservedNames()
    {
        var source = new Dictionary<string, double>(StringComparer.Ordinal) { ["occupancy"] = 0.75 };
        var measurement = new KernelTuningMeasurement(100, TimeSpan.FromMilliseconds(2), 4096, source);
        source["occupancy"] = 0.1;

        Assert.Equal(0.75, measurement.AdditionalMetrics["occupancy"]);
        Assert.Throws<ArgumentException>(() => new KernelTuningMeasurement(
            100,
            TimeSpan.Zero,
            additionalMetrics: new Dictionary<string, double>
            {
                [EvolutionKernelAutotunerMetrics.ThroughputGflops] = 99
            }));
    }

    private static EvolutionKernelAutotuner<FakeKernelConfiguration> CreateTuner(
        KernelTuningDeployment<FakeKernelConfiguration> deployment,
        Func<FakeKernelConfiguration, EvolutionEvaluationContext, CancellationToken,
            ValueTask<KernelTuningMeasurement>> benchmark) =>
        new(
            Identity(),
            new FakeKernelCodec(),
            new FakeKernelVariation(),
            benchmark,
            new EvolutionEngineOptions
            {
                RunId = "typed-kernel-test",
                Seed = 42,
                MaxEvaluationAttempts = 3,
                MaxProposals = 3,
                MaxGenerations = 0,
                ProposalBatchSize = 3,
                MaxDegreeOfParallelism = 1,
                IslandCount = 1,
                MigrationInterval = 0,
                MigrantsPerIsland = 1
            },
            deployment: deployment);

    private static KernelTuningIdentity Identity() => new(
        new KernelId("test", "fake-gpu-kernel"),
        new ShapeProfile(1, 64, 64, 4096),
        new GpuDeviceFingerprint(GpuVendorKind.Nvidia, "Fake GPU", 8, 6, 550, "fake-0"),
        "space-v1",
        "protocol-v1");

    private static IReadOnlyList<FakeKernelConfiguration> Seeds() => new[]
    {
        new FakeKernelConfiguration(FakeKernelVariant.Safe, 8),
        new FakeKernelConfiguration(FakeKernelVariant.Fast, 16),
        new FakeKernelConfiguration(FakeKernelVariant.Wide, 32)
    };

    private static KernelTuningMeasurement Measure(FakeKernelConfiguration configuration, double bonus = 0)
    {
        double throughput = configuration.Variant switch
        {
            FakeKernelVariant.Safe => 80,
            FakeKernelVariant.Fast => 160,
            FakeKernelVariant.Wide => 240,
            _ => throw new ArgumentOutOfRangeException(nameof(configuration))
        };
        return new KernelTuningMeasurement(throughput + bonus, TimeSpan.FromMilliseconds(1), 1024);
    }

    private enum FakeKernelVariant
    {
        Safe = 0,
        Fast = 1,
        Wide = 2
    }

    private readonly struct FakeKernelConfiguration : IEquatable<FakeKernelConfiguration>
    {
        public FakeKernelConfiguration(FakeKernelVariant variant, int tileEdge)
        {
            if (!Enum.IsDefined(typeof(FakeKernelVariant), variant))
                throw new ArgumentOutOfRangeException(nameof(variant));
            if (tileEdge <= 0) throw new ArgumentOutOfRangeException(nameof(tileEdge));
            Variant = variant;
            TileEdge = tileEdge;
        }

        public FakeKernelVariant Variant { get; }

        public int TileEdge { get; }

        public bool Equals(FakeKernelConfiguration other) =>
            Variant == other.Variant && TileEdge == other.TileEdge;

        public override bool Equals(object? obj) => obj is FakeKernelConfiguration other && Equals(other);

        public override int GetHashCode() => ((int)Variant * 397) ^ TileEdge;
    }

    private sealed class FakeKernelCodec : IEvolutionGenomeCodec<FakeKernelConfiguration>
    {
        public string Id => "fake-kernel-codec";

        public string VersionHash => "v1";

        public string Serialize(FakeKernelConfiguration genome) => string.Concat(
            ((int)genome.Variant).ToString(CultureInfo.InvariantCulture),
            ":",
            genome.TileEdge.ToString(CultureInfo.InvariantCulture));

        public FakeKernelConfiguration Deserialize(string payload)
        {
            string[] parts = payload.Split(':');
            if (parts.Length != 2 ||
                !int.TryParse(parts[0], NumberStyles.None, CultureInfo.InvariantCulture, out int variant) ||
                !int.TryParse(parts[1], NumberStyles.None, CultureInfo.InvariantCulture, out int tile))
            {
                throw new InvalidDataException("Invalid fake kernel configuration.");
            }
            return new FakeKernelConfiguration((FakeKernelVariant)variant, tile);
        }
    }

    private sealed class FakeKernelVariation : IVariationOperator<FakeKernelConfiguration>
    {
        public string Id => "fake-kernel-variation";

        public string VersionHash => "v1";

        public ValueTask<FakeKernelConfiguration> ProposeAsync(
            EvolutionVariationContext<FakeKernelConfiguration> context,
            CancellationToken cancellationToken = default)
        {
            cancellationToken.ThrowIfCancellationRequested();
            IReadOnlyList<FakeKernelConfiguration> seeds = Seeds();
            return new ValueTask<FakeKernelConfiguration>(seeds[context.Random.NextInt(seeds.Count)]);
        }
    }
}
