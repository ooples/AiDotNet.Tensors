using System.Globalization;
using AiDotNet.Evolution;
using AiDotNet.Tensors.Helpers;
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

/// <summary>Typed search policy for managed GEMM tuning.</summary>
public enum BlasManagedGemmSearchMode
{
    /// <summary>Uses heuristic seeds followed by bounded evolutionary variation.</summary>
    Evolutionary = 0,
    /// <summary>Requires the entire canonical space to fit the configured budgets and evaluates all of it.</summary>
    Exhaustive = 1,
    /// <summary>Uses exhaustive search when the entire space fits; otherwise uses evolutionary search.</summary>
    ExhaustiveWhenFeasible = 2,
}

/// <summary>Offline/startup evolutionary search for the combinatorial managed GEMM strategy space.</summary>
public static class BlasManagedEvolutionAutotuner
{
    private static readonly KernelId Fp32Kernel = new("blas-managed", "gemm-typed-evolution-fp32");
    private static readonly KernelId Fp64Kernel = new("blas-managed", "gemm-typed-evolution-fp64");

    /// <summary>
    /// Loads a locally persisted, identity-bound deployment and activates it for exact production replay.
    /// Arbitrary configurations cannot enter this restart path without matching canonical genome evidence.
    /// </summary>
    public static bool TryActivatePersisted<T>(
        int m,
        int n,
        int k,
        bool transA,
        bool transB,
        bool deterministic,
        KernelSearchSpaceVersion searchSpaceVersion,
        KernelBenchmarkProtocolVersion benchmarkProtocolVersion,
        IKernelTuningStore<BlasManagedGemmConfiguration>? store = null)
        where T : unmanaged
    {
        ValidateArguments<T>(m, n, k);
        var identity = CreateIdentity<T>(
            m, n, k, transA, transB, deterministic,
            searchSpaceVersion, benchmarkProtocolVersion);
        var codec = new BlasManagedGemmCodec();
        IKernelTuningStore<BlasManagedGemmConfiguration> resolvedStore =
            store ?? new AutotuneCacheKernelTuningStore<BlasManagedGemmConfiguration>();
        KernelTuningDeploymentSnapshot<BlasManagedGemmConfiguration>? snapshot;
        try
        {
            if (!resolvedStore.TryLoad(identity, codec, out snapshot) ||
                snapshot is null ||
                !string.Equals(snapshot.Identity.StableKey, identity.StableKey, StringComparison.Ordinal) ||
                ValidateConfiguration<T>(snapshot.Configuration, m, n, k, transB, deterministic) is not null)
            {
                return false;
            }
            string payload = codec.Serialize(snapshot.Configuration);
            if (!string.Equals(EvolutionHash.Compute(payload), snapshot.GenomeId, StringComparison.Ordinal))
                return false;
        }
        catch
        {
            return false;
        }

        ShapeProfile shape = BlasManagedAutotune.EncodeShape<T>(
            m, n, k, transA, transB, mr: 0, nr: 0, hasEpilogue: false, deterministic);
        ActivateValidated(shape, snapshot.Configuration);
        return true;
    }

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
        int processorCount = Math.Max(1, Math.Min(
            Environment.ProcessorCount,
            CpuParallelSettings.MaxDegreeOfParallelism));
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
            AddIfValid<T>(Canonicalize<T>(new BlasManagedGemmConfiguration(
                activeStrategy.Mode,
                activeStrategy.Axis,
                activeStrategy.Mc,
                activeStrategy.Nc,
                activeStrategy.Kc,
                activeStrategy.ThreadCount), m, n, k, deterministic),
                m, n, k, transB, deterministic, seen, seeds);
        }
        AddIfValid<T>(Canonicalize<T>(new BlasManagedGemmConfiguration(
            heuristicMode,
            heuristicAxis,
            64, 64, 64,
            heuristicAxis == ParallelismAxis.None ? 1 : processorCount), m, n, k, deterministic),
            m, n, k, transB, deterministic, seen, seeds);

        PackingMode[] modes = transB
            ? new[] { PackingMode.ForceStreaming, PackingMode.ForcePackBoth }
            : new[] { PackingMode.ForceStreaming, PackingMode.ForcePackAOnly, PackingMode.ForcePackBoth };
        for (int i = 0; i < modes.Length; i++)
        {
            AddIfValid<T>(Canonicalize<T>(new BlasManagedGemmConfiguration(
                modes[i], heuristicAxis, 64, 64, 64,
                heuristicAxis == ParallelismAxis.None ? 1 : processorCount), m, n, k, deterministic),
                m, n, k, transB, deterministic, seen, seeds);
        }

        if (additionalSeeds is not null)
        {
            foreach (BlasManagedGemmConfiguration seed in additionalSeeds)
                AddIfValid<T>(seed, m, n, k, transB, deterministic, seen, seeds);
        }
        return seeds.ToArray();
    }

    /// <summary>
    /// Attempts to materialize the complete canonical configuration space without exceeding a caller-owned bound.
    /// A false result returns no partial space, so it cannot accidentally be presented as exhaustive evidence.
    /// </summary>
    public static bool TryGetExhaustiveConfigurations<T>(
        int m,
        int n,
        int k,
        bool transA,
        bool transB,
        bool deterministic,
        int maximumConfigurationCount,
        out IReadOnlyList<BlasManagedGemmConfiguration> configurations)
        where T : unmanaged
    {
        ValidateArguments<T>(m, n, k);
        if (maximumConfigurationCount <= 0)
            throw new ArgumentOutOfRangeException(nameof(maximumConfigurationCount));
        int processorCount = Math.Max(1, Math.Min(
            Environment.ProcessorCount,
            CpuParallelSettings.MaxDegreeOfParallelism));
        var values = new List<BlasManagedGemmConfiguration>();
        var seen = new HashSet<BlasManagedGemmConfiguration>();

        bool TryAdd(BlasManagedGemmConfiguration value)
        {
            if (ValidateConfiguration<T>(value, m, n, k, transB, deterministic) is not null ||
                !seen.Add(value))
            {
                return true;
            }
            if (values.Count == maximumConfigurationCount)
                return false;
            values.Add(value);
            return true;
        }

        if (!TryAdd(new BlasManagedGemmConfiguration(
                PackingMode.Auto, ParallelismAxis.None, 0, 0, 0, 0)))
        {
            configurations = Array.Empty<BlasManagedGemmConfiguration>();
            return false;
        }

        foreach (ParallelismAxis axis in Enum.GetValues(typeof(ParallelismAxis)))
        {
            int firstThreadCount = axis == ParallelismAxis.None ? 1 : 2;
            int lastThreadCount = axis == ParallelismAxis.None ? 1 : processorCount;
            for (int threadCount = firstThreadCount; threadCount <= lastThreadCount; threadCount++)
            {
                if (!TryAdd(new BlasManagedGemmConfiguration(
                        PackingMode.ForceStreaming, axis, 0, 0, 0, threadCount)))
                {
                    configurations = Array.Empty<BlasManagedGemmConfiguration>();
                    return false;
                }
            }
        }

        PackingMode[] packedModes = transB
            ? new[] { PackingMode.ForcePackBoth }
            : new[] { PackingMode.ForcePackAOnly, PackingMode.ForcePackBoth };
        for (int modeIndex = 0; modeIndex < packedModes.Length; modeIndex++)
        {
            PackingMode mode = packedModes[modeIndex];
            var (mr, nr) = BlasManaged.GetEvolutionTuningTile<T>(mode, deterministic);
            if (m % mr != 0 || n % nr != 0)
                continue;
            for (int mc = mr; mc <= m; mc += mr)
            {
                int firstNc = mode == PackingMode.ForcePackAOnly ? 0 : nr;
                int lastNc = mode == PackingMode.ForcePackAOnly ? 0 : n;
                int ncStep = mode == PackingMode.ForcePackAOnly ? 1 : nr;
                for (int nc = firstNc; nc <= lastNc; nc += ncStep)
                {
                    for (int kc = 1; kc <= k; kc++)
                    {
                        foreach (ParallelismAxis axis in Enum.GetValues(typeof(ParallelismAxis)))
                        {
                            int firstThreadCount = axis == ParallelismAxis.None ? 1 : 2;
                            int lastThreadCount = axis == ParallelismAxis.None ? 1 : processorCount;
                            for (int threadCount = firstThreadCount;
                                 threadCount <= lastThreadCount;
                                 threadCount++)
                            {
                                if (!TryAdd(new BlasManagedGemmConfiguration(
                                        mode, axis, mc, nc, kc, threadCount)))
                                {
                                    configurations = Array.Empty<BlasManagedGemmConfiguration>();
                                    return false;
                                }
                            }
                        }
                    }
                }
            }
        }

        configurations = values.AsReadOnly();
        return true;
    }

    /// <summary>
    /// Creates the first-party CPU experiment used for both search measurements and sealed
    /// candidate-versus-production replay. The production <see cref="PackingMode.Auto"/> policy is
    /// the first-run incumbent, so specialized fast paths remain part of the comparison.
    /// </summary>
    public static KernelTuningExperiment<BlasManagedGemmConfiguration> CreateExperiment<T>(
        int m,
        int n,
        int k,
        bool transA,
        bool transB,
        bool deterministic,
        int inputSeed = 1729,
        int warmupCount = 2,
        int searchSampleCount = 5,
        int holdoutSampleCount = 9)
        where T : unmanaged
    {
        ValidateArguments<T>(m, n, k);
        var backend = new BlasManagedGemmExperimentBackend<T>(
            m, n, k, transA, transB, deterministic, inputSeed);
        var incumbent = new BlasManagedGemmConfiguration(
            PackingMode.Auto, ParallelismAxis.None, 0, 0, 0, 0);
        return new KernelTuningExperiment<BlasManagedGemmConfiguration>(
            backend,
            new StopwatchKernelTuningTimer(),
            incumbent,
            new KernelTuningWorkload(2d * m * n * k, KernelTuningWorkUnit.FloatingPointOperations),
            KernelTuningTimingScope.SteadyStateExecution,
            warmupCount,
            searchSampleCount,
            holdoutSampleCount);
    }

    /// <summary>
    /// Runs managed GEMM tuning through the first-party deterministic-input, scalar-oracle,
    /// paired-holdout experiment. Callers provide budgets and versions, never fabricated scores.
    /// </summary>
    public static Task<EvolutionKernelTuningResult<BlasManagedGemmConfiguration>> TuneAsync<T>(
        int m,
        int n,
        int k,
        bool transA,
        bool transB,
        bool deterministic,
        KernelSearchSpaceVersion searchSpaceVersion,
        KernelBenchmarkProtocolVersion benchmarkProtocolVersion,
        IEnumerable<BlasManagedGemmConfiguration>? additionalSeeds = null,
        EvolutionEngineOptions? engineOptions = null,
        KernelTuningOptions? tuningOptions = null,
        IEvolutionCheckpointStore? checkpointStore = null,
        KernelTuningDeploymentRegistry<BlasManagedGemmConfiguration>? deploymentRegistry = null,
        IKernelTuningStore<BlasManagedGemmConfiguration>? store = null,
        int inputSeed = 1729,
        int warmupCount = 2,
        int searchSampleCount = 5,
        int holdoutSampleCount = 9,
        BlasManagedGemmSearchMode searchMode = BlasManagedGemmSearchMode.ExhaustiveWhenFeasible,
        CancellationToken cancellationToken = default)
        where T : unmanaged
    {
        KernelTuningExperiment<BlasManagedGemmConfiguration> experiment = CreateExperiment<T>(
            m, n, k, transA, transB, deterministic,
            inputSeed, warmupCount, searchSampleCount, holdoutSampleCount);
        return TuneAsync<T>(
            m, n, k, transA, transB, deterministic,
            experiment.EvaluateAsync,
            experiment,
            searchSpaceVersion,
            benchmarkProtocolVersion,
            additionalSeeds,
            engineOptions,
            tuningOptions,
            checkpointStore,
            deploymentRegistry,
            store,
            searchMode,
            cancellationToken);
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
        IKernelTuningFinalistEvaluator<BlasManagedGemmConfiguration> finalistEvaluator,
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
        if (finalistEvaluator is null) throw new ArgumentNullException(nameof(finalistEvaluator));
        KernelTuningIdentity identity = CreateIdentity<T>(
            m, n, k, transA, transB, deterministic,
            searchSpaceVersion, benchmarkProtocolVersion);
        return new EvolutionKernelAutotuner<BlasManagedGemmConfiguration>(
            identity,
            new BlasManagedGemmCodec(),
            new BlasManagedGemmVariation<T>(m, n, k, transB, deterministic),
            (configuration, context, cancellationToken) =>
            {
                KernelTuningTrialResult? invalid = ValidateConfiguration<T>(
                    configuration, m, n, k, transB, deterministic);
                return invalid is null
                    ? evaluator(configuration, context, cancellationToken)
                    : new ValueTask<KernelTuningTrialResult>(invalid);
            },
            finalistEvaluator,
            engineOptions,
            tuningOptions,
            checkpointStore: checkpointStore,
            deploymentRegistry: deploymentRegistry,
            store: store,
            deploymentValidator: configuration =>
                ValidateConfiguration<T>(configuration, m, n, k, transB, deterministic) is null);
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
        IKernelTuningFinalistEvaluator<BlasManagedGemmConfiguration> finalistEvaluator,
        KernelSearchSpaceVersion searchSpaceVersion,
        KernelBenchmarkProtocolVersion benchmarkProtocolVersion,
        IEnumerable<BlasManagedGemmConfiguration>? additionalSeeds = null,
        EvolutionEngineOptions? engineOptions = null,
        KernelTuningOptions? tuningOptions = null,
        IEvolutionCheckpointStore? checkpointStore = null,
        KernelTuningDeploymentRegistry<BlasManagedGemmConfiguration>? deploymentRegistry = null,
        IKernelTuningStore<BlasManagedGemmConfiguration>? store = null,
        BlasManagedGemmSearchMode searchMode = BlasManagedGemmSearchMode.Evolutionary,
        CancellationToken cancellationToken = default)
        where T : unmanaged
    {
        IReadOnlyList<BlasManagedGemmConfiguration> seeds = GetSeeds<T>(
            m, n, k, transA, transB, deterministic, additionalSeeds);
        EvolutionKernelAutotuner<BlasManagedGemmConfiguration> tuner = Create<T>(
            m, n, k, transA, transB, deterministic, evaluator,
            finalistEvaluator,
            searchSpaceVersion, benchmarkProtocolVersion,
            engineOptions, tuningOptions, checkpointStore, deploymentRegistry, store);
        if (!Enum.IsDefined(typeof(BlasManagedGemmSearchMode), searchMode))
            throw new ArgumentOutOfRangeException(nameof(searchMode));
        int exhaustiveBound = Math.Min(
            tuner.MaximumProposals,
            tuner.MaximumEvaluationAttempts);
        IReadOnlyList<BlasManagedGemmConfiguration> completeSpace =
            Array.Empty<BlasManagedGemmConfiguration>();
        bool hasCompleteSpace = searchMode != BlasManagedGemmSearchMode.Evolutionary &&
            TryGetExhaustiveConfigurations<T>(
                m, n, k, transA, transB, deterministic,
                exhaustiveBound,
                out completeSpace);
        if (searchMode == BlasManagedGemmSearchMode.Exhaustive && !hasCompleteSpace)
        {
            throw new InvalidOperationException(
                "The complete managed GEMM search space exceeds the configured tuning budgets.");
        }
        if (seeds.Count > tuner.MaximumProposals)
            seeds = seeds.Take(tuner.MaximumProposals).ToArray();
        EvolutionKernelTuningResult<BlasManagedGemmConfiguration> result = hasCompleteSpace
            ? await tuner.TuneExhaustiveAsync(completeSpace, cancellationToken).ConfigureAwait(false)
            : await tuner.TuneAsync(seeds, cancellationToken).ConfigureAwait(false);
        BlasManagedGemmConfiguration winner = result.ActiveDeployment.Configuration;
        ShapeProfile shape = BlasManagedAutotune.EncodeShape<T>(
            m, n, k, transA, transB, mr: 0, nr: 0, hasEpilogue: false, deterministic);
        if (winner.PackingMode != PackingMode.Auto)
        {
            BlasManagedAutotune.StoreStrategy(
                shape,
                winner.PackingMode,
                winner.ParallelismAxis,
                winner.Mc,
                winner.Nc,
                winner.Kc,
                winner.ThreadCount,
                BlasKernelVersion.Current);
        }
        ActivateValidated(shape, winner);
        return result;
    }

    private static KernelTuningIdentity CreateIdentity<T>(
        int m,
        int n,
        int k,
        bool transA,
        bool transB,
        bool deterministic,
        KernelSearchSpaceVersion searchSpaceVersion,
        KernelBenchmarkProtocolVersion benchmarkProtocolVersion)
        where T : unmanaged
    {
        ShapeProfile shape = BlasManagedAutotune.EncodeShape<T>(
            m, n, k, transA, transB, mr: 0, nr: 0, hasEpilogue: false, deterministic);
        return new KernelTuningIdentity(
            typeof(T) == typeof(float) ? Fp32Kernel : Fp64Kernel,
            shape,
            KernelTuningDeviceFingerprint.CurrentCpu(),
            KernelTuningBackend.ManagedCpu,
            searchSpaceVersion,
            benchmarkProtocolVersion);
    }

    private static void ActivateValidated(
        ShapeProfile shape,
        BlasManagedGemmConfiguration configuration)
    {
        if (configuration.PackingMode == PackingMode.Auto)
            BlasManagedAutotune.RemoveEvolutionStrategy(shape);
        else
            BlasManagedAutotune.PublishEvolutionStrategy(shape, configuration);
    }

    private static void AddIfValid<T>(
        BlasManagedGemmConfiguration configuration,
        int m,
        int n,
        int k,
        bool transB,
        bool deterministic,
        ISet<BlasManagedGemmConfiguration> seen,
        ICollection<BlasManagedGemmConfiguration> target)
        where T : unmanaged
    {
        if (ValidateConfiguration<T>(configuration, m, n, k, transB, deterministic) is null && seen.Add(configuration))
            target.Add(configuration);
    }

    private static KernelTuningTrialResult? ValidateConfiguration<T>(
        BlasManagedGemmConfiguration configuration,
        int m,
        int n,
        int k,
        bool transB,
        bool deterministic)
        where T : unmanaged
    {
        if (!Enum.IsDefined(typeof(PackingMode), configuration.PackingMode) ||
            configuration.PackingMode == PackingMode.DisableAutotune)
        {
            return KernelTuningTrialResult.Rejected(
                KernelTuningTrialStatus.InvalidConfiguration,
                "Evolution does not benchmark the DisableAutotune compatibility mode.");
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
        if (configuration.PackingMode == PackingMode.Auto)
        {
            bool isCanonicalBaseline = configuration.ParallelismAxis == ParallelismAxis.None &&
                configuration.Mc == 0 && configuration.Nc == 0 && configuration.Kc == 0 &&
                configuration.ThreadCount == 0;
            return isCanonicalBaseline
                ? null
                : KernelTuningTrialResult.Rejected(
                    KernelTuningTrialStatus.InvalidConfiguration,
                    "The Auto incumbent is a fixed production policy and cannot carry candidate overrides.");
        }
        var (mr, nr) = BlasManaged.GetEvolutionTuningTile<T>(configuration.PackingMode, deterministic);
        bool validBlocking = configuration.PackingMode switch
        {
            PackingMode.ForceStreaming => configuration.Mc == 0 && configuration.Nc == 0 && configuration.Kc == 0,
            PackingMode.ForcePackAOnly =>
                configuration.Mc > 0 && configuration.Mc <= m && configuration.Mc % mr == 0 &&
                configuration.Nc == 0 && configuration.Kc > 0 && configuration.Kc <= k,
            PackingMode.ForcePackBoth =>
                configuration.Mc > 0 && configuration.Mc <= m && configuration.Mc % mr == 0 &&
                configuration.Nc > 0 && configuration.Nc <= n && configuration.Nc % nr == 0 &&
                configuration.Kc > 0 && configuration.Kc <= k,
            _ => false
        };
        if (!validBlocking)
        {
            return KernelTuningTrialResult.Rejected(
                KernelTuningTrialStatus.InvalidConfiguration,
                "Managed GEMM blocking must be canonical for the active packing strategy, shape, and microkernel tile.");
        }
        if (configuration.PackingMode != PackingMode.ForceStreaming &&
            (m % mr != 0 || n % nr != 0))
        {
            return KernelTuningTrialResult.Rejected(
                KernelTuningTrialStatus.InvalidConfiguration,
                "Packed evolutionary candidates require an aligned shape so replay cannot alias through a streaming tail path.");
        }
        int processorCount = Math.Max(1, Math.Min(
            Environment.ProcessorCount,
            CpuParallelSettings.MaxDegreeOfParallelism));
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
        if (configuration.ParallelismAxis != ParallelismAxis.None && configuration.ThreadCount < 2)
        {
            return KernelTuningTrialResult.Rejected(
                KernelTuningTrialStatus.InvalidConfiguration,
                "A parallel axis requires at least two threads.");
        }
        bool axisCanExecute = configuration.PackingMode switch
        {
            PackingMode.ForceStreaming => configuration.ParallelismAxis switch
            {
                ParallelismAxis.None => true,
                ParallelismAxis.M => (long)m * n * k >= AxisSelector.ParallelWorkThreshold && m >= 16,
                ParallelismAxis.N => (long)m * n * k >= AxisSelector.ParallelWorkThreshold &&
                    n >= configuration.ThreadCount * 16,
                ParallelismAxis.K => !deterministic &&
                    (long)m * n * k >= AxisSelector.ParallelWorkThreshold,
                _ => false
            },
            PackingMode.ForcePackAOnly => configuration.ParallelismAxis switch
            {
                ParallelismAxis.None => true,
                ParallelismAxis.N => n >= configuration.ThreadCount * nr * 2,
                _ => false
            },
            PackingMode.ForcePackBoth => configuration.ParallelismAxis switch
            {
                ParallelismAxis.None => true,
                ParallelismAxis.M => (m + configuration.Mc - 1) / configuration.Mc >= 2,
                ParallelismAxis.MN_2D =>
                    ((m + configuration.Mc - 1) / configuration.Mc) *
                    ((n + configuration.Nc - 1) / configuration.Nc) >= 2,
                _ => false
            },
            _ => false
        };
        if (!axisCanExecute)
        {
            return KernelTuningTrialResult.Rejected(
                KernelTuningTrialStatus.InvalidConfiguration,
                "The requested parallelism axis is not a distinct executable path for this strategy and shape.");
        }
        return null;
    }

    private static BlasManagedGemmConfiguration Canonicalize<T>(
        BlasManagedGemmConfiguration configuration,
        int m,
        int n,
        int k,
        bool deterministic)
        where T : unmanaged
    {
        if (!Enum.IsDefined(typeof(PackingMode), configuration.PackingMode))
            return configuration;
        var (mr, nr) = BlasManaged.GetEvolutionTuningTile<T>(configuration.PackingMode, deterministic);
        ParallelismAxis axis = configuration.PackingMode switch
        {
            PackingMode.ForceStreaming when configuration.ParallelismAxis == ParallelismAxis.MN_2D => ParallelismAxis.M,
            PackingMode.ForceStreaming => configuration.ParallelismAxis,
            PackingMode.ForcePackAOnly when configuration.ParallelismAxis == ParallelismAxis.N => ParallelismAxis.N,
            PackingMode.ForcePackAOnly => ParallelismAxis.None,
            PackingMode.ForcePackBoth when configuration.ParallelismAxis == ParallelismAxis.MN_2D => ParallelismAxis.MN_2D,
            PackingMode.ForcePackBoth when configuration.ParallelismAxis == ParallelismAxis.None => ParallelismAxis.None,
            PackingMode.ForcePackBoth => ParallelismAxis.M,
            _ => configuration.ParallelismAxis
        };
        if (deterministic && axis == ParallelismAxis.K) axis = ParallelismAxis.M;
        int maximumThreads = Math.Max(1, Math.Min(
            Environment.ProcessorCount,
            CpuParallelSettings.MaxDegreeOfParallelism));
        if (maximumThreads < 2) axis = ParallelismAxis.None;
        int threadCount = axis == ParallelismAxis.None
            ? 1
            : Math.Max(2, Math.Min(Math.Max(2, configuration.ThreadCount), maximumThreads));
        BlasManagedGemmConfiguration canonical = configuration.PackingMode switch
        {
            PackingMode.Auto => new BlasManagedGemmConfiguration(
                PackingMode.Auto, ParallelismAxis.None, 0, 0, 0, 0),
            PackingMode.ForceStreaming => configuration with
            {
                ParallelismAxis = axis,
                Mc = 0,
                Nc = 0,
                Kc = 0,
                ThreadCount = threadCount
            },
            PackingMode.ForcePackAOnly => configuration with
            {
                ParallelismAxis = axis,
                Mc = CanonicalBlock(configuration.Mc, m, mr),
                Nc = 0,
                Kc = Math.Min(Math.Max(1, configuration.Kc), k),
                ThreadCount = threadCount
            },
            PackingMode.ForcePackBoth => configuration with
            {
                ParallelismAxis = axis,
                Mc = CanonicalBlock(configuration.Mc, m, mr),
                Nc = CanonicalBlock(configuration.Nc, n, nr),
                Kc = Math.Min(Math.Max(1, configuration.Kc), k),
                ThreadCount = threadCount
            },
            _ => configuration
        };
        bool executableAxis = canonical.PackingMode switch
        {
            PackingMode.Auto => true,
            PackingMode.ForceStreaming => canonical.ParallelismAxis switch
            {
                ParallelismAxis.None => true,
                ParallelismAxis.M => (long)m * n * k >= AxisSelector.ParallelWorkThreshold && m >= 16,
                ParallelismAxis.N => (long)m * n * k >= AxisSelector.ParallelWorkThreshold &&
                    n >= canonical.ThreadCount * 16,
                ParallelismAxis.K => !deterministic &&
                    (long)m * n * k >= AxisSelector.ParallelWorkThreshold,
                _ => false
            },
            PackingMode.ForcePackAOnly => canonical.ParallelismAxis == ParallelismAxis.None ||
                (canonical.ParallelismAxis == ParallelismAxis.N && n >= canonical.ThreadCount * nr * 2),
            PackingMode.ForcePackBoth => canonical.ParallelismAxis switch
            {
                ParallelismAxis.None => true,
                ParallelismAxis.M => (m + canonical.Mc - 1) / canonical.Mc >= 2,
                ParallelismAxis.MN_2D =>
                    ((m + canonical.Mc - 1) / canonical.Mc) *
                    ((n + canonical.Nc - 1) / canonical.Nc) >= 2,
                _ => false
            },
            _ => false
        };
        return executableAxis
            ? canonical
            : canonical with { ParallelismAxis = ParallelismAxis.None, ThreadCount = 1 };
    }

    private static int CanonicalBlock(int requested, int dimension, int alignment)
    {
        int clamped = Math.Min(requested > 0 ? requested : 64, dimension);
        int aligned = clamped - clamped % alignment;
        return aligned > 0 ? aligned : dimension;
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
        public string VersionHash => "2";

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
            bool validBlocks = value.PackingMode switch
            {
                PackingMode.Auto => value.Mc == 0 && value.Nc == 0 && value.Kc == 0 &&
                    value.ParallelismAxis == ParallelismAxis.None && value.ThreadCount == 0,
                PackingMode.ForceStreaming => value.Mc == 0 && value.Nc == 0 && value.Kc == 0,
                PackingMode.ForcePackAOnly => value.Mc > 0 && value.Nc == 0 && value.Kc > 0,
                PackingMode.ForcePackBoth => value.Mc > 0 && value.Nc > 0 && value.Kc > 0,
                _ => false
            };
            if (!Enum.IsDefined(typeof(PackingMode), value.PackingMode) ||
                !Enum.IsDefined(typeof(ParallelismAxis), value.ParallelismAxis) ||
                !validBlocks || (value.PackingMode != PackingMode.Auto && value.ThreadCount <= 0))
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

    private sealed class BlasManagedGemmVariation<T> : IVariationOperator<BlasManagedGemmConfiguration>
        where T : unmanaged
    {
        private static readonly PackingMode[] Modes =
            { PackingMode.ForceStreaming, PackingMode.ForcePackAOnly, PackingMode.ForcePackBoth };
        private static readonly ParallelismAxis[] Axes =
            { ParallelismAxis.None, ParallelismAxis.M, ParallelismAxis.N, ParallelismAxis.K, ParallelismAxis.MN_2D };
        private static readonly int[] Blocks = { 32, 64, 128, 256, 512 };
        private readonly bool _transB;
        private readonly bool _deterministic;
        private readonly int _m;
        private readonly int _n;
        private readonly int _k;

        internal BlasManagedGemmVariation(int m, int n, int k, bool transB, bool deterministic)
        {
            _m = m;
            _n = n;
            _k = k;
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
            return new ValueTask<BlasManagedGemmConfiguration>(
                Canonicalize<T>(value, _m, _n, _k, _deterministic));
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
