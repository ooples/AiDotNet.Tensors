using System.Globalization;
using AiDotNet.Evolution;
using AiDotNet.Tensors.Helpers.Autotune;

namespace AiDotNet.Tensors.Engines.DirectGpu.OpenCL;

/// <summary>
/// Immutable, label-free OpenCL GEMM genome. Every field changes generated code or launch geometry.
/// </summary>
public readonly record struct OpenClGemmConfiguration(
    GemmKernelTemplate KernelTemplate,
    int TileM,
    int TileN,
    int TileK,
    int ThreadTileM,
    int ThreadTileN,
    int VectorWidthM,
    int VectorWidthN,
    bool UseDoubleBuffering,
    bool UseVectorizedLoads,
    int KReg,
    int KUnroll,
    bool UseSubgroupOps,
    bool StrideM,
    bool StrideN,
    bool CacheA,
    bool CacheB,
    int MdimaSize,
    int NdimbSize,
    bool UseTrueVectorLds,
    bool UseColumnMajorA)
{
    internal static OpenClGemmConfiguration FromGemmConfig(GemmConfig configuration) => new(
        configuration.KernelTemplate,
        configuration.TileM,
        configuration.TileN,
        configuration.TileK,
        configuration.ThreadTileM,
        configuration.ThreadTileN,
        configuration.VectorWidthM,
        configuration.VectorWidthN,
        configuration.UseDoubleBuffering,
        configuration.UseVectorizedLoads,
        configuration.KReg,
        configuration.KUnroll,
        configuration.UseSubgroupOps,
        configuration.StrideM,
        configuration.StrideN,
        configuration.CacheA,
        configuration.CacheB,
        configuration.MdimaSize,
        configuration.NdimbSize,
        configuration.UseTrueVectorLDS,
        configuration.UseColumnMajorA);

    internal GemmConfig ToGemmConfig() => new()
    {
        KernelTemplate = KernelTemplate,
        KernelName = KernelTemplate switch
        {
            GemmKernelTemplate.Tuned => "evolved-tuned",
            GemmKernelTemplate.ClBlastBaselineK0 => "evolved-clblast-k0",
            GemmKernelTemplate.ClBlastBaselineK1 => "evolved-clblast-k1",
            _ => throw new ArgumentOutOfRangeException(nameof(KernelTemplate))
        },
        TileM = TileM,
        TileN = TileN,
        TileK = TileK,
        ThreadTileM = ThreadTileM,
        ThreadTileN = ThreadTileN,
        VectorWidthM = VectorWidthM,
        VectorWidthN = VectorWidthN,
        UseDoubleBuffering = UseDoubleBuffering,
        UseVectorizedLoads = UseVectorizedLoads,
        KReg = KReg,
        KUnroll = KUnroll,
        UseSubgroupOps = UseSubgroupOps,
        StrideM = StrideM,
        StrideN = StrideN,
        CacheA = CacheA,
        CacheB = CacheB,
        MdimaSize = MdimaSize,
        NdimbSize = NdimbSize,
        UseTrueVectorLDS = UseTrueVectorLds,
        UseColumnMajorA = UseColumnMajorA
    };
}

/// <summary>Evolution support for GEMM's large constrained schedule space.</summary>
public sealed partial class GemmAutoTuner
{
    private static readonly KernelId EvolutionKernelId = new("gemm", "opencl-typed-evolution");

    /// <summary>
    /// Returns locally valid heuristic configurations plus optional Bayesian, persisted, or community seeds.
    /// Downloaded/community values are never trusted merely because they deserialize.
    /// </summary>
    public IReadOnlyList<OpenClGemmConfiguration> GetEvolutionSeeds(
        int m,
        int n,
        int k,
        GpuCapabilities capabilities,
        IEnumerable<GemmConfig>? additionalSeeds = null)
    {
        ValidateEvolutionArguments(m, n, k, capabilities);
        var seeds = new List<OpenClGemmConfiguration>();
        var seen = new HashSet<OpenClGemmConfiguration>();
        lock (_cacheLock)
        {
            if (_cache.TryGetValue((m, n, k), out GemmConfig active))
                TryAddValidSeed(active, capabilities, seen, seeds);
        }
        foreach (GemmConfig candidate in GetCandidateConfigs(m, n, k, capabilities))
            TryAddValidSeed(candidate, capabilities, seen, seeds);
        if (additionalSeeds is not null)
        {
            foreach (GemmConfig candidate in additionalSeeds)
                TryAddValidSeed(candidate, capabilities, seen, seeds);
        }
        if (seeds.Count == 0)
            throw new InvalidOperationException("The OpenCL device admits no valid GEMM seed configuration.");
        return seeds.ToArray();
    }

    /// <summary>Creates a persistent correctness-first tuner for one shape and physical device.</summary>
    public EvolutionKernelAutotuner<OpenClGemmConfiguration> CreateEvolutionTuner(
        int m,
        int n,
        int k,
        GpuCapabilities capabilities,
        GpuDeviceFingerprint fingerprint,
        Func<GemmConfig, EvolutionEvaluationContext, CancellationToken,
            ValueTask<KernelTuningTrialResult>> evaluator,
        KernelSearchSpaceVersion searchSpaceVersion,
        KernelBenchmarkProtocolVersion benchmarkProtocolVersion,
        EvolutionEngineOptions? engineOptions = null,
        KernelTuningOptions? tuningOptions = null,
        IEvolutionCheckpointStore? checkpointStore = null,
        KernelTuningDeploymentRegistry<OpenClGemmConfiguration>? deploymentRegistry = null,
        IKernelTuningStore<OpenClGemmConfiguration>? store = null)
    {
        ValidateEvolutionArguments(m, n, k, capabilities);
        if (evaluator is null) throw new ArgumentNullException(nameof(evaluator));
        var identity = new KernelTuningIdentity(
            EvolutionKernelId,
            new ShapeProfile(m, n, k),
            fingerprint,
            searchSpaceVersion,
            benchmarkProtocolVersion);
        return new EvolutionKernelAutotuner<OpenClGemmConfiguration>(
            identity,
            new OpenClGemmCodec(),
            new OpenClGemmVariation(capabilities),
            async (configuration, context, cancellationToken) =>
            {
                GemmConfig candidate = configuration.ToGemmConfig();
                KernelTuningTrialResult? invalid = ValidateEvolutionCandidate(candidate, capabilities);
                if (invalid is not null) return invalid;
                return await evaluator(candidate, context, cancellationToken).ConfigureAwait(false);
            },
            engineOptions,
            tuningOptions,
            checkpointStore: checkpointStore,
            deploymentRegistry: deploymentRegistry,
            store: store,
            deploymentValidator: configuration =>
                ValidateEvolutionCandidate(configuration.ToGemmConfig(), capabilities) is null);
    }

    /// <summary>
    /// Executes a fixed-budget evolutionary run seeded by the existing heuristic and optional Bayesian winners.
    /// </summary>
    public async Task<EvolutionKernelTuningResult<OpenClGemmConfiguration>> TuneWithEvolutionAsync(
        int m,
        int n,
        int k,
        GpuCapabilities capabilities,
        GpuDeviceFingerprint fingerprint,
        Func<GemmConfig, EvolutionEvaluationContext, CancellationToken,
            ValueTask<KernelTuningTrialResult>> evaluator,
        KernelSearchSpaceVersion searchSpaceVersion,
        KernelBenchmarkProtocolVersion benchmarkProtocolVersion,
        IEnumerable<GemmConfig>? additionalSeeds = null,
        EvolutionEngineOptions? engineOptions = null,
        KernelTuningOptions? tuningOptions = null,
        IEvolutionCheckpointStore? checkpointStore = null,
        KernelTuningDeploymentRegistry<OpenClGemmConfiguration>? deploymentRegistry = null,
        IKernelTuningStore<OpenClGemmConfiguration>? store = null,
        CancellationToken cancellationToken = default)
    {
        IReadOnlyList<OpenClGemmConfiguration> seeds = GetEvolutionSeeds(
            m, n, k, capabilities, additionalSeeds);
        EvolutionKernelAutotuner<OpenClGemmConfiguration> tuner = CreateEvolutionTuner(
            m, n, k, capabilities, fingerprint, evaluator,
            searchSpaceVersion, benchmarkProtocolVersion,
            engineOptions, tuningOptions, checkpointStore, deploymentRegistry, store);
        if (seeds.Count > tuner.MaximumProposals)
            seeds = seeds.Take(tuner.MaximumProposals).ToArray();
        EvolutionKernelTuningResult<OpenClGemmConfiguration> result =
            await tuner.TuneAsync(seeds, cancellationToken).ConfigureAwait(false);
        lock (_cacheLock)
            _cache[(m, n, k)] = result.ActiveDeployment.Configuration.ToGemmConfig();
        return result;
    }

    private static void TryAddValidSeed(
        GemmConfig candidate,
        GpuCapabilities capabilities,
        ISet<OpenClGemmConfiguration> seen,
        ICollection<OpenClGemmConfiguration> seeds)
    {
        if (ValidateEvolutionCandidate(candidate, capabilities) is not null) return;

        OpenClGemmConfiguration typed = OpenClGemmConfiguration.FromGemmConfig(candidate);
        if (seen.Add(typed)) seeds.Add(typed);
    }

    private static KernelTuningTrialResult? ValidateEvolutionCandidate(
        GemmConfig candidate,
        GpuCapabilities capabilities)
    {
        if (!Enum.IsDefined(typeof(GemmKernelTemplate), candidate.KernelTemplate))
            return KernelTuningTrialResult.Rejected(KernelTuningTrialStatus.InvalidConfiguration);
        if (candidate.TileM <= 0 || candidate.TileN <= 0 || candidate.TileK <= 0 ||
            candidate.ThreadTileM <= 0 || candidate.ThreadTileN <= 0 ||
            candidate.VectorWidthM <= 0 || candidate.VectorWidthN <= 0 ||
            candidate.KReg <= 0 || candidate.KUnroll <= 0 ||
            candidate.MdimaSize <= 0 || candidate.NdimbSize <= 0)
        {
            return KernelTuningTrialResult.Rejected(
                KernelTuningTrialStatus.InvalidConfiguration,
                "OpenCL GEMM geometry fields must be positive.");
        }
        if (candidate.UseSubgroupOps && !capabilities.SupportsSubgroups)
        {
            return KernelTuningTrialResult.Rejected(
                KernelTuningTrialStatus.ResourceLimitExceeded,
                "The candidate requires subgroup operations not supported by this device.");
        }
        long workGroupSize = (long)candidate.ThreadTileM * candidate.ThreadTileN;
        if (workGroupSize <= 0 || workGroupSize > capabilities.MaxWorkGroupSize)
        {
            return KernelTuningTrialResult.Rejected(
                KernelTuningTrialStatus.ResourceLimitExceeded,
                "The candidate exceeds the device work-group limit.");
        }
        long localBytes = EstimateLocalMemoryBytes(candidate);
        if (localBytes > capabilities.LocalMemoryBytes)
        {
            return KernelTuningTrialResult.Rejected(
                KernelTuningTrialStatus.ResourceLimitExceeded,
                "The candidate exceeds the device local-memory limit.");
        }
        string? validationError = DynamicGemmKernel.ValidateConfig(candidate);
        return validationError is null
            ? null
            : KernelTuningTrialResult.Rejected(
                KernelTuningTrialStatus.InvalidConfiguration, validationError);
    }

    private static long EstimateLocalMemoryBytes(GemmConfig candidate)
    {
        try
        {
            checked
            {
                int vectorM = Math.Max(1, candidate.VectorWidthM);
                int vectorN = Math.Max(1, candidate.VectorWidthN);
                if (candidate.KernelTemplate != GemmKernelTemplate.Tuned)
                {
                    long floats =
                        (candidate.CacheA ? (long)candidate.TileK * candidate.TileM / vectorM : 0) +
                        (candidate.CacheB ? (long)candidate.TileK * candidate.TileN / vectorN : 0);
                    return floats * sizeof(float);
                }

                long outputM = candidate.TileM / candidate.ThreadTileM;
                long outputN = candidate.TileN / candidate.ThreadTileN;
                bool doubleBuffered = candidate.UseDoubleBuffering && outputM * outputN <= 16;
                long multiplier = doubleBuffered ? 2L : 1L;
                return multiplier * candidate.TileK *
                       ((long)candidate.TileM + candidate.TileN + 2L) * sizeof(float);
            }
        }
        catch (OverflowException)
        {
            return long.MaxValue;
        }
    }

    private static void ValidateEvolutionArguments(int m, int n, int k, GpuCapabilities capabilities)
    {
        if (m <= 0) throw new ArgumentOutOfRangeException(nameof(m));
        if (n <= 0) throw new ArgumentOutOfRangeException(nameof(n));
        if (k <= 0) throw new ArgumentOutOfRangeException(nameof(k));
        if (capabilities is null) throw new ArgumentNullException(nameof(capabilities));
        if (capabilities.MaxWorkGroupSize <= 0)
            throw new ArgumentOutOfRangeException(nameof(capabilities.MaxWorkGroupSize));
        if (capabilities.LocalMemoryBytes <= 0)
            throw new ArgumentOutOfRangeException(nameof(capabilities.LocalMemoryBytes));
    }

    private sealed class OpenClGemmCodec : IEvolutionGenomeCodec<OpenClGemmConfiguration>
    {
        public string Id => "opencl-gemm-typed";
        public string VersionHash => "2";

        public string Serialize(OpenClGemmConfiguration genome)
        {
            ValidatePayloadGenome(genome);
            return string.Join("|", new[]
            {
                Format((int)genome.KernelTemplate),
                Format(genome.TileM), Format(genome.TileN), Format(genome.TileK),
                Format(genome.ThreadTileM), Format(genome.ThreadTileN),
                Format(genome.VectorWidthM), Format(genome.VectorWidthN),
                Format(genome.UseDoubleBuffering), Format(genome.UseVectorizedLoads),
                Format(genome.KReg), Format(genome.KUnroll), Format(genome.UseSubgroupOps),
                Format(genome.StrideM), Format(genome.StrideN), Format(genome.CacheA), Format(genome.CacheB),
                Format(genome.MdimaSize), Format(genome.NdimbSize),
                Format(genome.UseTrueVectorLds), Format(genome.UseColumnMajorA)
            });
        }

        public OpenClGemmConfiguration Deserialize(string payload)
        {
            if (payload is null) throw new ArgumentNullException(nameof(payload));
            string[] values = payload.Split('|');
            if (values.Length != 21) throw new InvalidDataException("Invalid OpenCL GEMM genome field count.");
            int index = 0;
            var genome = new OpenClGemmConfiguration(
                (GemmKernelTemplate)ParseInt(values[index++]),
                ParseInt(values[index++]), ParseInt(values[index++]), ParseInt(values[index++]),
                ParseInt(values[index++]), ParseInt(values[index++]),
                ParseInt(values[index++]), ParseInt(values[index++]),
                ParseBoolean(values[index++]), ParseBoolean(values[index++]),
                ParseInt(values[index++]), ParseInt(values[index++]), ParseBoolean(values[index++]),
                ParseBoolean(values[index++]), ParseBoolean(values[index++]),
                ParseBoolean(values[index++]), ParseBoolean(values[index++]),
                ParseInt(values[index++]), ParseInt(values[index++]),
                ParseBoolean(values[index++]), ParseBoolean(values[index]));
            ValidatePayloadGenome(genome);
            return genome;
        }

        private static void ValidatePayloadGenome(OpenClGemmConfiguration genome)
        {
            if (!Enum.IsDefined(typeof(GemmKernelTemplate), genome.KernelTemplate) ||
                genome.TileM <= 0 || genome.TileN <= 0 || genome.TileK <= 0 ||
                genome.ThreadTileM <= 0 || genome.ThreadTileN <= 0 ||
                genome.VectorWidthM <= 0 || genome.VectorWidthN <= 0 ||
                genome.KReg <= 0 || genome.KUnroll <= 0 ||
                genome.MdimaSize <= 0 || genome.NdimbSize <= 0)
            {
                throw new InvalidDataException("The OpenCL GEMM genome contains an invalid typed field.");
            }
        }

        private static int ParseInt(string value) =>
            int.TryParse(value, NumberStyles.Integer, CultureInfo.InvariantCulture, out int parsed)
                ? parsed
                : throw new InvalidDataException("The OpenCL GEMM genome contains an invalid integer.");

        private static bool ParseBoolean(string value) => value switch
        {
            "0" => false,
            "1" => true,
            _ => throw new InvalidDataException("The OpenCL GEMM genome contains an invalid Boolean.")
        };

        private static string Format(int value) => value.ToString(CultureInfo.InvariantCulture);
        private static string Format(bool value) => value ? "1" : "0";
    }

    private sealed class OpenClGemmVariation : IVariationOperator<OpenClGemmConfiguration>
    {
        private static readonly int[] TileSizes = { 16, 32, 64, 128, 256 };
        private static readonly int[] KTileSizes = { 8, 16, 32 };
        private static readonly int[] ThreadDimensions = { 4, 8, 16, 32 };
        private static readonly int[] VectorWidths = { 1, 2, 4, 8 };
        private static readonly int[] RegisterTiles = { 1, 2, 4 };
        private static readonly int[] UnrollFactors = { 1, 2, 4, 8 };
        private static readonly int[] CooperativeDimensions = { 8, 16, 32, 64 };
        private readonly GpuCapabilities _capabilities;

        internal OpenClGemmVariation(GpuCapabilities capabilities) => _capabilities = capabilities;

        public string Id => "opencl-gemm-constrained-variation";
        public string VersionHash => "2";

        public ValueTask<OpenClGemmConfiguration> ProposeAsync(
            EvolutionVariationContext<OpenClGemmConfiguration> context,
            CancellationToken cancellationToken = default)
        {
            cancellationToken.ThrowIfCancellationRequested();
            OpenClGemmConfiguration candidate = SelectParent(context);
            int mutationCount = 1 + context.Random.NextInt(3);
            for (int i = 0; i < mutationCount; i++) candidate = Mutate(candidate, context.Random);
            if (!_capabilities.SupportsSubgroups && candidate.UseSubgroupOps)
                candidate = candidate with { UseSubgroupOps = false };
            return new ValueTask<OpenClGemmConfiguration>(candidate);
        }

        private static OpenClGemmConfiguration SelectParent(
            EvolutionVariationContext<OpenClGemmConfiguration> context)
        {
            if (context.Inspirations.Count > 0 && context.Random.NextDouble() < 0.3d)
            {
                return context.Inspirations[context.Random.NextInt(context.Inspirations.Count)]
                    .Candidate.CanonicalGenome.Genome;
            }
            return context.Parent.Candidate.CanonicalGenome.Genome;
        }

        private static OpenClGemmConfiguration Mutate(
            OpenClGemmConfiguration value,
            StableRandom random) => random.NextInt(21) switch
        {
            0 => value with { KernelTemplate = (GemmKernelTemplate)random.NextInt(3) },
            1 => value with { TileM = Pick(TileSizes, random) },
            2 => value with { TileN = Pick(TileSizes, random) },
            3 => value with { TileK = Pick(KTileSizes, random) },
            4 => value with { ThreadTileM = Pick(ThreadDimensions, random) },
            5 => value with { ThreadTileN = Pick(ThreadDimensions, random) },
            6 => value with { VectorWidthM = Pick(VectorWidths, random) },
            7 => value with { VectorWidthN = Pick(VectorWidths, random) },
            8 => value with { UseDoubleBuffering = !value.UseDoubleBuffering },
            9 => value with { UseVectorizedLoads = !value.UseVectorizedLoads },
            10 => value with { KReg = Pick(RegisterTiles, random) },
            11 => value with { KUnroll = Pick(UnrollFactors, random) },
            12 => value with { UseSubgroupOps = !value.UseSubgroupOps },
            13 => value with { StrideM = !value.StrideM },
            14 => value with { StrideN = !value.StrideN },
            15 => value with { CacheA = !value.CacheA },
            16 => value with { CacheB = !value.CacheB },
            17 => value with { MdimaSize = Pick(CooperativeDimensions, random) },
            18 => value with { NdimbSize = Pick(CooperativeDimensions, random) },
            19 => value with { UseTrueVectorLds = !value.UseTrueVectorLds },
            20 => value with { UseColumnMajorA = !value.UseColumnMajorA },
            _ => throw new InvalidOperationException()
        };

        private static int Pick(IReadOnlyList<int> values, StableRandom random) =>
            values[random.NextInt(values.Count)];
    }
}
