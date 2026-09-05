using System.Globalization;
using AiDotNet.Evolution;
using AiDotNet.Tensors.Helpers.Autotune;

namespace AiDotNet.Tensors.Engines.BlasManaged;

/// <summary>Immutable typed genome for managed GEMM strategy, blocking, and CPU parallelism.</summary>
public readonly record struct BlasManagedGemmConfiguration(
    PackingMode PackingMode,
    ParallelismAxis ParallelismAxis,
    int Mc,
    int Nc,
    int Kc,
    int ThreadCount);

/// <summary>Offline/startup evolutionary search for the combinatorial managed GEMM strategy space.</summary>
public static class BlasManagedEvolutionAutotuner
{
    private static readonly KernelId Fp32Kernel = new("blas-managed", "gemm-typed-evolution-fp32");
    private static readonly KernelId Fp64Kernel = new("blas-managed", "gemm-typed-evolution-fp64");

    /// <summary>Creates validated heuristic seeds plus any locally valid external/pre-warm seeds.</summary>
    public static IReadOnlyList<BlasManagedGemmConfiguration> GetSeeds<T>(
        int m,
        int n,
        int k,
        bool transA,
        bool transB,
        bool deterministic,
        IEnumerable<BlasManagedGemmConfiguration>? additionalSeeds = null)
        where T : unmanaged
    {
        ValidateArguments<T>(m, n, k);
        int processorCount = Math.Max(1, Environment.ProcessorCount);
        ParallelismAxis heuristicAxis = AxisSelector.Select(
            m, n, k, mr: 4, nr: 8, processorCount, deterministic);
        PackingMode heuristicMode = StrategyDefaultTable.Route(HardwareFingerprint.Key, m, n, k);
        var seeds = new List<BlasManagedGemmConfiguration>();
        var seen = new HashSet<BlasManagedGemmConfiguration>();
        ShapeProfile shape = BlasManagedAutotune.EncodeShape<T>(
            m, n, k, transA, transB, mr: 0, nr: 0, hasEpilogue: false, deterministic);
        var active = BlasManagedAutotune.TryLookupStrategy(shape);
        if (active is { } activeStrategy)
        {
            AddIfValid(new BlasManagedGemmConfiguration(
                activeStrategy.Mode,
                activeStrategy.Axis,
                activeStrategy.Mc,
                activeStrategy.Nc,
                activeStrategy.Kc,
                activeStrategy.ThreadCount), transB, deterministic, seen, seeds);
        }
        AddIfValid(new BlasManagedGemmConfiguration(
            heuristicMode,
            heuristicAxis,
            64, 64, 64,
            heuristicAxis == ParallelismAxis.None ? 1 : processorCount), transB, deterministic, seen, seeds);

        PackingMode[] modes = transB
            ? new[] { PackingMode.ForceStreaming, PackingMode.ForcePackBoth }
            : new[] { PackingMode.ForceStreaming, PackingMode.ForcePackAOnly, PackingMode.ForcePackBoth };
        for (int i = 0; i < modes.Length; i++)
        {
            AddIfValid(new BlasManagedGemmConfiguration(
                modes[i], heuristicAxis, 64, 64, 64,
                heuristicAxis == ParallelismAxis.None ? 1 : processorCount), transB, deterministic, seen, seeds);
        }

        if (additionalSeeds is not null)
        {
            foreach (BlasManagedGemmConfiguration seed in additionalSeeds)
                AddIfValid(seed, transB, deterministic, seen, seeds);
        }
        return seeds.ToArray();
    }

    /// <summary>Creates a correctness-first CPU tuner without changing dispatch until a winner is promoted.</summary>
    public static EvolutionKernelAutotuner<BlasManagedGemmConfiguration> Create<T>(
        int m,
        int n,
        int k,
        bool transA,
        bool transB,
        bool deterministic,
        Func<BlasManagedGemmConfiguration, EvolutionEvaluationContext, CancellationToken,
            ValueTask<KernelTuningTrialResult>> evaluator,
        KernelSearchSpaceVersion searchSpaceVersion,
        KernelBenchmarkProtocolVersion benchmarkProtocolVersion,
        EvolutionEngineOptions? engineOptions = null,
        KernelTuningOptions? tuningOptions = null,
        IEvolutionCheckpointStore? checkpointStore = null,
        KernelTuningDeploymentRegistry<BlasManagedGemmConfiguration>? deploymentRegistry = null,
        IKernelTuningStore<BlasManagedGemmConfiguration>? store = null)
        where T : unmanaged
    {
        ValidateArguments<T>(m, n, k);
        if (evaluator is null) throw new ArgumentNullException(nameof(evaluator));
        ShapeProfile shape = BlasManagedAutotune.EncodeShape<T>(
            m, n, k, transA, transB, mr: 0, nr: 0, hasEpilogue: false, deterministic);
        var identity = new KernelTuningIdentity(
            typeof(T) == typeof(float) ? Fp32Kernel : Fp64Kernel,
            shape,
            KernelTuningDeviceFingerprint.CurrentCpu(),
            searchSpaceVersion,
            benchmarkProtocolVersion);
        return new EvolutionKernelAutotuner<BlasManagedGemmConfiguration>(
            identity,
            new BlasManagedGemmCodec(),
            new BlasManagedGemmVariation(transB, deterministic),
            (configuration, context, cancellationToken) =>
            {
                KernelTuningTrialResult? invalid = ValidateConfiguration(
                    configuration, transB, deterministic);
                return invalid is null
                    ? evaluator(configuration, context, cancellationToken)
                    : new ValueTask<KernelTuningTrialResult>(invalid);
            },
            engineOptions,
            tuningOptions,
            checkpointStore: checkpointStore,
            deploymentRegistry: deploymentRegistry,
            store: store,
            deploymentValidator: configuration =>
                ValidateConfiguration(configuration, transB, deterministic) is null);
    }

    /// <summary>
    /// Runs a fixed-budget search and publishes a promoted result into the existing lock-free BlasManaged memo.
    /// </summary>
    public static async Task<EvolutionKernelTuningResult<BlasManagedGemmConfiguration>> TuneAsync<T>(
        int m,
        int n,
        int k,
        bool transA,
        bool transB,
        bool deterministic,
        Func<BlasManagedGemmConfiguration, EvolutionEvaluationContext, CancellationToken,
            ValueTask<KernelTuningTrialResult>> evaluator,
        KernelSearchSpaceVersion searchSpaceVersion,
        KernelBenchmarkProtocolVersion benchmarkProtocolVersion,
        IEnumerable<BlasManagedGemmConfiguration>? additionalSeeds = null,
        EvolutionEngineOptions? engineOptions = null,
        KernelTuningOptions? tuningOptions = null,
        IEvolutionCheckpointStore? checkpointStore = null,
        KernelTuningDeploymentRegistry<BlasManagedGemmConfiguration>? deploymentRegistry = null,
        IKernelTuningStore<BlasManagedGemmConfiguration>? store = null,
        CancellationToken cancellationToken = default)
        where T : unmanaged
    {
        IReadOnlyList<BlasManagedGemmConfiguration> seeds = GetSeeds<T>(
            m, n, k, transA, transB, deterministic, additionalSeeds);
        EvolutionKernelAutotuner<BlasManagedGemmConfiguration> tuner = Create<T>(
            m, n, k, transA, transB, deterministic, evaluator,
            searchSpaceVersion, benchmarkProtocolVersion,
            engineOptions, tuningOptions, checkpointStore, deploymentRegistry, store);
        if (seeds.Count > tuner.MaximumProposals)
            seeds = seeds.Take(tuner.MaximumProposals).ToArray();
        EvolutionKernelTuningResult<BlasManagedGemmConfiguration> result =
            await tuner.TuneAsync(seeds, cancellationToken).ConfigureAwait(false);
        BlasManagedGemmConfiguration winner = result.ActiveDeployment.Configuration;
        ShapeProfile shape = BlasManagedAutotune.EncodeShape<T>(
            m, n, k, transA, transB, mr: 0, nr: 0, hasEpilogue: false, deterministic);
        BlasManagedAutotune.StoreStrategy(
            shape,
            winner.PackingMode,
            winner.ParallelismAxis,
            winner.Mc,
            winner.Nc,
            winner.Kc,
            winner.ThreadCount,
            BlasKernelVersion.Current);
        return result;
    }

    private static void AddIfValid(
        BlasManagedGemmConfiguration configuration,
        bool transB,
        bool deterministic,
        ISet<BlasManagedGemmConfiguration> seen,
        ICollection<BlasManagedGemmConfiguration> target)
    {
        if (ValidateConfiguration(configuration, transB, deterministic) is null && seen.Add(configuration))
            target.Add(configuration);
    }

    private static KernelTuningTrialResult? ValidateConfiguration(
        BlasManagedGemmConfiguration configuration,
        bool transB,
        bool deterministic)
    {
        if (!Enum.IsDefined(typeof(PackingMode), configuration.PackingMode) ||
            configuration.PackingMode is PackingMode.Auto or PackingMode.DisableAutotune)
        {
            return KernelTuningTrialResult.Rejected(
                KernelTuningTrialStatus.InvalidConfiguration,
                "Evolution requires a concrete managed GEMM packing strategy.");
        }
        if (!Enum.IsDefined(typeof(ParallelismAxis), configuration.ParallelismAxis))
            return KernelTuningTrialResult.Rejected(KernelTuningTrialStatus.InvalidConfiguration);
        if (transB && configuration.PackingMode == PackingMode.ForcePackAOnly)
        {
            return KernelTuningTrialResult.Rejected(
                KernelTuningTrialStatus.InvalidConfiguration,
                "Pack-A-only aliases PackBoth for transposed B and is not a distinct candidate.");
        }
        if (deterministic && configuration.ParallelismAxis == ParallelismAxis.K)
        {
            return KernelTuningTrialResult.Rejected(
                KernelTuningTrialStatus.InvalidConfiguration,
                "K-axis reduction is not admitted by deterministic mode.");
        }
        if (configuration.Mc <= 0 || configuration.Nc <= 0 || configuration.Kc <= 0 ||
            configuration.Mc % 16 != 0 || configuration.Nc % 16 != 0 || configuration.Kc % 16 != 0 ||
            configuration.Mc > 1024 || configuration.Nc > 1024 || configuration.Kc > 1024)
        {
            return KernelTuningTrialResult.Rejected(
                KernelTuningTrialStatus.InvalidConfiguration,
                "Managed GEMM block sizes must be positive multiples of 16 no larger than 1024.");
        }
        int processorCount = Math.Max(1, Environment.ProcessorCount);
        if (configuration.ThreadCount <= 0 || configuration.ThreadCount > processorCount)
        {
            return KernelTuningTrialResult.Rejected(
                KernelTuningTrialStatus.ResourceLimitExceeded,
                "The configuration oversubscribes the current CPU.");
        }
        if (configuration.ParallelismAxis == ParallelismAxis.None && configuration.ThreadCount != 1)
        {
            return KernelTuningTrialResult.Rejected(
                KernelTuningTrialStatus.InvalidConfiguration,
                "Sequential execution must use one thread.");
        }
        return null;
    }

    private static void ValidateArguments<T>(int m, int n, int k)
        where T : unmanaged
    {
        if (typeof(T) != typeof(float) && typeof(T) != typeof(double))
            throw new NotSupportedException("Managed evolutionary GEMM tuning supports float and double.");
        if (m <= 0) throw new ArgumentOutOfRangeException(nameof(m));
        if (n <= 0) throw new ArgumentOutOfRangeException(nameof(n));
        if (k <= 0) throw new ArgumentOutOfRangeException(nameof(k));
    }

    private sealed class BlasManagedGemmCodec : IEvolutionGenomeCodec<BlasManagedGemmConfiguration>
    {
        public string Id => "blas-managed-gemm-typed";
        public string VersionHash => "1";

        public string Serialize(BlasManagedGemmConfiguration genome)
        {
            ValidatePayload(genome);
            return string.Join("|", new[]
            {
                Format((int)genome.PackingMode),
                Format((int)genome.ParallelismAxis),
                Format(genome.Mc), Format(genome.Nc), Format(genome.Kc), Format(genome.ThreadCount)
            });
        }

        public BlasManagedGemmConfiguration Deserialize(string payload)
        {
            if (payload is null) throw new ArgumentNullException(nameof(payload));
            string[] values = payload.Split('|');
            if (values.Length != 6) throw new InvalidDataException("Invalid managed GEMM genome field count.");
            var result = new BlasManagedGemmConfiguration(
                (PackingMode)Parse(values[0]),
                (ParallelismAxis)Parse(values[1]),
                Parse(values[2]), Parse(values[3]), Parse(values[4]), Parse(values[5]));
            ValidatePayload(result);
            return result;
        }

        private static void ValidatePayload(BlasManagedGemmConfiguration value)
        {
            if (!Enum.IsDefined(typeof(PackingMode), value.PackingMode) ||
                !Enum.IsDefined(typeof(ParallelismAxis), value.ParallelismAxis) ||
                value.Mc <= 0 || value.Nc <= 0 || value.Kc <= 0 || value.ThreadCount <= 0)
            {
                throw new InvalidDataException("The managed GEMM genome contains an invalid typed field.");
            }
        }

        private static int Parse(string value) =>
            int.TryParse(value, NumberStyles.Integer, CultureInfo.InvariantCulture, out int parsed)
                ? parsed
                : throw new InvalidDataException("The managed GEMM genome contains an invalid integer.");

        private static string Format(int value) => value.ToString(CultureInfo.InvariantCulture);
    }

    private sealed class BlasManagedGemmVariation : IVariationOperator<BlasManagedGemmConfiguration>
    {
        private static readonly PackingMode[] Modes =
            { PackingMode.ForceStreaming, PackingMode.ForcePackAOnly, PackingMode.ForcePackBoth };
        private static readonly ParallelismAxis[] Axes =
            { ParallelismAxis.None, ParallelismAxis.M, ParallelismAxis.N, ParallelismAxis.K, ParallelismAxis.MN_2D };
        private static readonly int[] Blocks = { 32, 64, 128, 256, 512 };
        private readonly bool _transB;
        private readonly bool _deterministic;

        internal BlasManagedGemmVariation(bool transB, bool deterministic)
        {
            _transB = transB;
            _deterministic = deterministic;
        }

        public string Id => "blas-managed-gemm-constrained-variation";
        public string VersionHash => "1";

        public ValueTask<BlasManagedGemmConfiguration> ProposeAsync(
            EvolutionVariationContext<BlasManagedGemmConfiguration> context,
            CancellationToken cancellationToken = default)
        {
            cancellationToken.ThrowIfCancellationRequested();
            BlasManagedGemmConfiguration value = context.Parent.Candidate.CanonicalGenome.Genome;
            int mutationCount = 1 + context.Random.NextInt(2);
            for (int i = 0; i < mutationCount; i++) value = Mutate(value, context.Random);
            if (_transB && value.PackingMode == PackingMode.ForcePackAOnly)
                value = value with { PackingMode = PackingMode.ForcePackBoth };
            if (_deterministic && value.ParallelismAxis == ParallelismAxis.K)
                value = value with { ParallelismAxis = ParallelismAxis.M };
            if (value.ParallelismAxis == ParallelismAxis.None)
                value = value with { ThreadCount = 1 };
            return new ValueTask<BlasManagedGemmConfiguration>(value);
        }

        private static BlasManagedGemmConfiguration Mutate(
            BlasManagedGemmConfiguration value,
            StableRandom random) => random.NextInt(6) switch
        {
            0 => value with { PackingMode = Modes[random.NextInt(Modes.Length)] },
            1 => value with { ParallelismAxis = Axes[random.NextInt(Axes.Length)] },
            2 => value with { Mc = Blocks[random.NextInt(Blocks.Length)] },
            3 => value with { Nc = Blocks[random.NextInt(Blocks.Length)] },
            4 => value with { Kc = Blocks[random.NextInt(Blocks.Length)] },
            5 => value with { ThreadCount = 1 + random.NextInt(Math.Max(1, Environment.ProcessorCount)) },
            _ => throw new InvalidOperationException()
        };
    }
}
