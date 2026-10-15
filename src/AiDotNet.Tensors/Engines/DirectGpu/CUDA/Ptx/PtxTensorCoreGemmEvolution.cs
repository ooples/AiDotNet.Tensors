using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using AiDotNet.Evolution;
using AiDotNet.Tensors.Helpers.Autotune;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// Evolves the generated Tensor-Core GEMM (<see cref="PtxTensorCoreGemmGenerator"/>, TF32 or FP16 operands with FP32
/// output) for one shape on one CUDA device, with AiDotNet.Evolution through the same scaffold as
/// <see cref="PtxGemmEvolution"/>: compile, fp64-oracle validation, CUDA-event timing, paired finalist replay.
/// </summary>
internal static class PtxTensorCoreGemmEvolution
{
    internal static readonly KernelId EvolutionKernelId = new("gemm", "cuda-ptx-tensor-core-evolution");
    internal static readonly KernelSearchSpaceVersion SearchSpaceVersion = new(1);
    internal static readonly KernelBenchmarkProtocolVersion BenchmarkProtocolVersion = new(1);

    internal static bool Admits(PtxTensorCoreGemmConfiguration configuration, PtxGemmLayout layout, int m, int n, int k,
        PtxTensorCorePrecision precision) =>
        PtxTensorCoreGemmGenerator.Admits(configuration, layout, m, n, k,
            layout.TransposeA ? m : k, layout.TransposeB ? k : n, n, 0, 0, 0, precision);

    /// <summary>
    /// Starting points kept where they admit the problem; the first is the default incumbent: the largest block tile that
    /// still launches a block per multiprocessor, else the one launching the most blocks.
    /// </summary>
    internal static IReadOnlyList<PtxTensorCoreGemmConfiguration> Seeds(PtxGemmLayout layout, int m, int n, int k,
        PtxTensorCorePrecision precision, int multiprocessors = 68)
    {
        var candidates = new[]
        {
            new PtxTensorCoreGemmConfiguration(128, 128, 16, 64, 64, 2, 8),
            new PtxTensorCoreGemmConfiguration(128, 128, 32, 64, 64, 2, 0),
            new PtxTensorCoreGemmConfiguration(128, 128, 16, 64, 32, 3, 4),
            new PtxTensorCoreGemmConfiguration(128, 64, 16, 64, 32, 3, 4),
            new PtxTensorCoreGemmConfiguration(64, 128, 16, 32, 64, 3, 4),
            new PtxTensorCoreGemmConfiguration(64, 64, 16, 32, 32, 3, 4),
            new PtxTensorCoreGemmConfiguration(64, 64, 32, 32, 32, 2, 8),
        };
        var seeds = new List<PtxTensorCoreGemmConfiguration>();
        foreach (var candidate in candidates)
            if (Admits(candidate, layout, m, n, k, precision)) seeds.Add(candidate);
        if (seeds.Count == 0)
            throw new InvalidOperationException($"No Tensor-Core GEMM seed admits M={m} N={n} K={k} ({layout.Name}, {precision}).");
        static long Blocks(PtxTensorCoreGemmConfiguration c, int m, int n) =>
            (long)((m + c.BlockM - 1) / c.BlockM) * ((n + c.BlockN - 1) / c.BlockN);
        PtxTensorCoreGemmConfiguration chosen = seeds[0];
        bool filled = false;
        foreach (var c in seeds)
        {
            bool fills = Blocks(c, m, n) >= multiprocessors;
            if (fills && (!filled || c.BlockM * c.BlockN > chosen.BlockM * chosen.BlockN)) { chosen = c; filled = true; }
            else if (!filled && Blocks(c, m, n) > Blocks(chosen, m, n)) chosen = c;
        }
        seeds.Remove(chosen);
        seeds.Insert(0, chosen);
        return seeds;
    }

    /// <summary>Runs a fixed-budget evolutionary search for one shape, layout and precision.</summary>
    internal static async Task<EvolutionKernelTuningResult<PtxTensorCoreGemmConfiguration>> TuneAsync(
        DirectPtxRuntime runtime, PtxGemmLayout layout, int m, int n, int k, PtxTensorCorePrecision precision,
        EvolutionEngineOptions? engineOptions = null,
        KernelTuningOptions? tuningOptions = null,
        IEvolutionCheckpointStore? checkpointStore = null,
        KernelTuningDeploymentRegistry<PtxTensorCoreGemmConfiguration>? deploymentRegistry = null,
        IKernelTuningStore<PtxTensorCoreGemmConfiguration>? store = null,
        Action<PtxTensorCoreGemmConfiguration, KernelTuningTrialResult>? onTrial = null,
        CancellationToken cancellationToken = default)
    {
        if (runtime is null) throw new ArgumentNullException(nameof(runtime));
        if (m <= 0) throw new ArgumentOutOfRangeException(nameof(m));
        if (n <= 0) throw new ArgumentOutOfRangeException(nameof(n));
        if (k <= 0) throw new ArgumentOutOfRangeException(nameof(k));
        if (runtime.ComputeCapabilityMajor < 8)
            throw new PlatformNotSupportedException("Tensor-Core GEMM tuning needs compute capability 8.0+.");

        IReadOnlyList<PtxTensorCoreGemmConfiguration> seeds =
            Seeds(layout, m, n, k, precision, PtxGemmEvolution.MultiprocessorCount(runtime));
        int lda = layout.TransposeA ? m : k, ldb = layout.TransposeB ? k : n;
        using var backend = new PtxGemmEvolution.ExperimentBackend<PtxTensorCoreGemmConfiguration>(runtime, layout, m, n, k,
            configuration =>
            {
                var kernel = new PtxTensorCoreGemmKernel(runtime, configuration, layout, precision);
                return new PtxGemmEvolution.LoadedGemm(kernel, kernel.Audit,
                    (a, b, c) => kernel.Launch(a, b, c, m, n, k, lda, ldb, n, 1f, 0f));
            },
            configuration => configuration.Name,
            halfOperands: precision == PtxTensorCorePrecision.Fp16);
        var experiment = new KernelTuningExperiment<PtxTensorCoreGemmConfiguration>(
            backend,
            new DeviceEventKernelTuningTimer(new PtxGemmEvolution.CudaEventClock(runtime)),
            seeds[0],
            new KernelTuningWorkload(checked(2d * m * n * k), KernelTuningWorkUnit.FloatingPointOperations),
            KernelTuningTimingScope.SteadyStateExecution);
        var identity = new KernelTuningIdentity(
            EvolutionKernelId,
            new ShapeProfile(m, n, k, (layout.TransposeA ? 1 : 0) | (layout.TransposeB ? 2 : 0) | ((int)precision << 2)),
            PtxGemmEvolution.Fingerprint(runtime),
            KernelTuningBackend.Cuda,
            SearchSpaceVersion,
            BenchmarkProtocolVersion);
        var tuner = new EvolutionKernelAutotuner<PtxTensorCoreGemmConfiguration>(
            identity,
            new Codec(),
            new Variation(layout, m, n, k, precision),
            async (configuration, context, token) =>
            {
                KernelTuningTrialResult result = await EvaluateAsync(configuration, context, token).ConfigureAwait(false);
                onTrial?.Invoke(configuration, result);
                return result;
            },
            experiment,
            engineOptions,
            tuningOptions,
            checkpointStore: checkpointStore,
            deploymentRegistry: deploymentRegistry,
            store: store,
            deploymentValidator: configuration => Admits(configuration, layout, m, n, k, precision));

        async ValueTask<KernelTuningTrialResult> EvaluateAsync(
            PtxTensorCoreGemmConfiguration configuration, EvolutionEvaluationContext context, CancellationToken token)
        {
            string? invalid = configuration.Invalidity(layout, precision);
            if (invalid is not null)
                return KernelTuningTrialResult.Rejected(KernelTuningTrialStatus.InvalidConfiguration, invalid);
            if (!Admits(configuration, layout, m, n, k, precision))
                return KernelTuningTrialResult.Rejected(KernelTuningTrialStatus.InvalidConfiguration,
                    "The configuration does not admit this problem's alignment.");
            string? compileError = backend.TryCompile(configuration);
            if (compileError is not null)
                return KernelTuningTrialResult.Rejected(KernelTuningTrialStatus.CompilationFailed, compileError);
            return await experiment.EvaluateAsync(configuration, context, token).ConfigureAwait(false);
        }

        if (seeds.Count > tuner.MaximumProposals)
        {
            var trimmed = new List<PtxTensorCoreGemmConfiguration>();
            for (int i = 0; i < tuner.MaximumProposals; i++) trimmed.Add(seeds[i]);
            seeds = trimmed;
        }
        return await tuner.TuneAsync(seeds, cancellationToken).ConfigureAwait(false);
    }

    private sealed class Codec : IEvolutionGenomeCodec<PtxTensorCoreGemmConfiguration>
    {
        public string Id => "cuda-ptx-tensor-core-gemm-typed";
        public string VersionHash => "1";

        public string Serialize(PtxTensorCoreGemmConfiguration genome) => string.Join("|", new[]
        {
            F(genome.BlockM), F(genome.BlockN), F(genome.BlockK), F(genome.WarpM), F(genome.WarpN), F(genome.Stages),
            F(genome.SharedPad),
        });

        public PtxTensorCoreGemmConfiguration Deserialize(string payload)
        {
            if (payload is null) throw new ArgumentNullException(nameof(payload));
            string[] v = payload.Split('|');
            if (v.Length != 7) throw new InvalidDataException("Invalid Tensor-Core GEMM genome field count.");
            return new PtxTensorCoreGemmConfiguration(P(v[0]), P(v[1]), P(v[2]), P(v[3]), P(v[4]), P(v[5]), P(v[6]));
        }

        private static string F(int value) => value.ToString(CultureInfo.InvariantCulture);

        private static int P(string value) =>
            int.TryParse(value, NumberStyles.Integer, CultureInfo.InvariantCulture, out int parsed)
                ? parsed
                : throw new InvalidDataException("Invalid integer in the Tensor-Core GEMM genome.");
    }

    /// <summary>One to three mutations of a parent (or an inspiration), repaired until the child admits the problem.</summary>
    private sealed class Variation : IVariationOperator<PtxTensorCoreGemmConfiguration>
    {
        private static readonly int[] Tiles = { 64, 128, 256 };
        private static readonly int[] KTiles = { 16, 32, 64 };
        private static readonly int[] WarpTiles = { 32, 64 };
        private static readonly int[] StageCounts = { 2, 3, 4 };
        private static readonly int[] Pads = { 0, 4, 8 };
        private readonly PtxGemmLayout _layout;
        private readonly int _m, _n, _k;
        private readonly PtxTensorCorePrecision _precision;

        internal Variation(PtxGemmLayout layout, int m, int n, int k, PtxTensorCorePrecision precision)
        {
            _layout = layout;
            _m = m;
            _n = n;
            _k = k;
            _precision = precision;
        }

        public string Id => "cuda-ptx-tensor-core-gemm-constrained-variation";
        public string VersionHash => "1";

        public ValueTask<PtxTensorCoreGemmConfiguration> ProposeAsync(
            EvolutionVariationContext<PtxTensorCoreGemmConfiguration> context, CancellationToken cancellationToken = default)
        {
            cancellationToken.ThrowIfCancellationRequested();
            PtxTensorCoreGemmConfiguration parent = context.Inspirations.Count > 0 && context.Random.NextDouble() < 0.3d
                ? context.Inspirations[context.Random.NextInt(context.Inspirations.Count)].Candidate.CanonicalGenome.Genome
                : context.Parent.Candidate.CanonicalGenome.Genome;
            for (int attempt = 0; attempt < 64; attempt++)
            {
                PtxTensorCoreGemmConfiguration child = parent;
                int mutations = 1 + context.Random.NextInt(3);
                for (int i = 0; i < mutations; i++) child = Mutate(child, context.Random);
                if (child != parent && Admits(child, _layout, _m, _n, _k, _precision))
                    return new ValueTask<PtxTensorCoreGemmConfiguration>(child);
            }
            return new ValueTask<PtxTensorCoreGemmConfiguration>(parent);
        }

        private static PtxTensorCoreGemmConfiguration Mutate(PtxTensorCoreGemmConfiguration value, StableRandom random) =>
            random.NextInt(7) switch
            {
                0 => value with { BlockM = Pick(Tiles, random) },
                1 => value with { BlockN = Pick(Tiles, random) },
                2 => value with { BlockK = Pick(KTiles, random) },
                3 => value with { WarpM = Pick(WarpTiles, random) },
                4 => value with { WarpN = Pick(WarpTiles, random) },
                5 => value with { Stages = Pick(StageCounts, random) },
                6 => value with { SharedPad = Pick(Pads, random) },
                _ => throw new InvalidOperationException(),
            };

        private static int Pick(int[] values, StableRandom random) => values[random.NextInt(values.Length)];
    }
}
