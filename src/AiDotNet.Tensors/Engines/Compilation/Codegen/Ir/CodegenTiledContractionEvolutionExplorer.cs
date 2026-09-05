using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using AiDotNet.Evolution;
using AiDotNet.Tensors.Helpers.Autotune;

namespace AiDotNet.Tensors.Engines.Compilation.Codegen.Ir;

/// <summary>
/// Explores typed PTX contraction schedules without bypassing the complete
/// measured-search gate that owns production codegen dispatch.
/// </summary>
/// <remarks>
/// A result is a development candidate, not a production winner. Add a useful
/// discovered schedule to <see cref="CodegenTiledContractionSchedule.SearchSpace"/>
/// and run the full codegen championship so every applicable schedule receives
/// the established correctness, stability, and direct-finalist evidence.
/// </remarks>
public static class CodegenTiledContractionEvolutionExplorer
{
    /// <summary>Runs a bounded offline evolutionary search over valid schedule geometry.</summary>
    public static async Task<EvolutionKernelTuningResult<CodegenTiledContractionSchedule>> ExploreAsync(
        CodegenKernelSpec spec,
        KernelId kernel,
        GpuDeviceFingerprint device,
        int computeMajor,
        int computeMinor,
        Func<CodegenTiledContractionSchedule, CodegenTiledContractionPlan,
            EvolutionEvaluationContext, CancellationToken,
            ValueTask<KernelTuningTrialResult>> evaluator,
        KernelSearchSpaceVersion searchSpaceVersion,
        IEnumerable<CodegenTiledContractionSchedule>? additionalSeeds = null,
        EvolutionEngineOptions? engineOptions = null,
        KernelTuningOptions? tuningOptions = null,
        IEvolutionCheckpointStore? checkpointStore = null,
        IKernelTuningStore<CodegenTiledContractionSchedule>? store = null,
        CancellationToken cancellationToken = default)
    {
        if (spec is null) throw new ArgumentNullException(nameof(spec));
        if (evaluator is null) throw new ArgumentNullException(nameof(evaluator));
        if (computeMajor <= 0) throw new ArgumentOutOfRangeException(nameof(computeMajor));
        if (computeMinor < 0) throw new ArgumentOutOfRangeException(nameof(computeMinor));
        if (!CodegenTiledContractionPlan.TryCreate(
                spec, out CodegenTiledContractionPlan? modelled, out string reason) ||
            modelled is null)
        {
            throw new ArgumentException(
                "The codegen spec is not a tiled contraction: " + reason, nameof(spec));
        }

        var codec = new ScheduleCodec();
        var variation = new ScheduleVariation(modelled);
        KernelTuningIdentity identity = CreateIdentity(
            spec,
            modelled,
            kernel,
            device,
            computeMajor,
            computeMinor,
            searchSpaceVersion);
        EvolutionEngineOptions resolvedOptions = engineOptions ?? DefaultEngineOptions(identity);
        var tuner = new EvolutionKernelAutotuner<CodegenTiledContractionSchedule>(
            identity,
            codec,
            variation,
            async (schedule, context, token) =>
            {
                if (!CodegenTiledContractionPlan.TryCreate(
                        spec, schedule, out CodegenTiledContractionPlan? plan, out string refusal) ||
                    plan is null)
                {
                    return KernelTuningTrialResult.Rejected(
                        KernelTuningTrialStatus.InvalidConfiguration, refusal);
                }
                return await evaluator(schedule, plan, context, token).ConfigureAwait(false);
            },
            resolvedOptions,
            tuningOptions,
            checkpointStore: checkpointStore,
            store: store,
            deploymentValidator: schedule =>
                CodegenTiledContractionPlan.TryCreate(spec, schedule, out _, out _));

        CodegenTiledContractionSchedule[] seeds = BuildSeeds(
            spec, modelled, additionalSeeds, codec, tuner.MaximumProposals);
        return await tuner.TuneAsync(seeds, cancellationToken).ConfigureAwait(false);
    }

    internal static KernelTuningIdentity CreateIdentity(
        CodegenKernelSpec spec,
        CodegenTiledContractionPlan plan,
        KernelId kernel,
        GpuDeviceFingerprint device,
        int computeMajor,
        int computeMinor,
        KernelSearchSpaceVersion searchSpaceVersion)
    {
        if (spec is null) throw new ArgumentNullException(nameof(spec));
        if (plan is null) throw new ArgumentNullException(nameof(plan));
        CodegenAutotuneIdentity codegenIdentity = CodegenAutotuneIdentity.Create(
            spec,
            device.LocalKey,
            computeMajor,
            computeMinor);
        var fingerprintedKernel = new KernelId(
            kernel.Category,
            EvolutionHash.Combine(new[]
            {
                "codegen-contraction-evolution-v1",
                kernel.Name,
                codegenIdentity.Target,
                codegenIdentity.SpecFingerprint,
                codegenIdentity.EmitterFingerprint
            }));
        return new KernelTuningIdentity(
            fingerprintedKernel,
            new ShapeProfile(
                plan.Batch,
                plan.M,
                plan.N,
                plan.K,
                computeMajor,
                computeMinor),
            KernelTuningDeviceFingerprint.FromGpu(device),
            searchSpaceVersion,
            new KernelBenchmarkProtocolVersion(CodegenMeasurementProtocol.Version));
    }

    private static EvolutionEngineOptions DefaultEngineOptions(KernelTuningIdentity identity) => new()
    {
        RunId = "codegen-contraction-" + identity.StableKey,
        MaxEvaluationAttempts = 64,
        MaxProposals = 512,
        MaxGenerations = 512,
        ProposalBatchSize = 1,
        MaxDegreeOfParallelism = 1,
        IslandCount = 1,
        MigrationInterval = 0,
        MigrantsPerIsland = 1
    };

    private static CodegenTiledContractionSchedule[] BuildSeeds(
        CodegenKernelSpec spec,
        CodegenTiledContractionPlan modelled,
        IEnumerable<CodegenTiledContractionSchedule>? additionalSeeds,
        ScheduleCodec codec,
        int maximumProposals)
    {
        var candidates = new List<CodegenTiledContractionSchedule>
        {
            new(
                modelled.TileM,
                modelled.TileN,
                modelled.TileK,
                modelled.ThreadTileM,
                modelled.ThreadTileN,
                modelled.RegisterPrefetch)
        };
        candidates.AddRange(CodegenTiledContractionSchedule.SearchSpace);
        if (additionalSeeds is not null) candidates.AddRange(additionalSeeds);

        var result = new List<CodegenTiledContractionSchedule>(
            Math.Min(candidates.Count, maximumProposals));
        var seen = new HashSet<string>(StringComparer.Ordinal);
        foreach (CodegenTiledContractionSchedule candidate in candidates)
        {
            if (!CodegenTiledContractionPlan.TryCreate(
                    spec, candidate, out CodegenTiledContractionPlan? plan, out _) ||
                plan is null)
                continue;
            string canonical = codec.Serialize(candidate);
            if (seen.Add(canonical)) result.Add(candidate);
            if (result.Count == maximumProposals) break;
        }
        if (result.Count == 0)
            throw new InvalidOperationException("No valid tiled contraction seed was available.");
        return result.ToArray();
    }

    private sealed class ScheduleCodec : IEvolutionGenomeCodec<CodegenTiledContractionSchedule>
    {
        public string Id => "codegen-tiled-contraction-schedule";
        public string VersionHash => "1";

        public string Serialize(CodegenTiledContractionSchedule genome)
        {
            Validate(genome);
            return string.Join("|", new[]
            {
                Format(genome.TileM),
                Format(genome.TileN),
                Format(genome.TileK),
                Format(genome.ThreadTileM),
                Format(genome.ThreadTileN),
                genome.RegisterPrefetch ? "1" : "0"
            });
        }

        public CodegenTiledContractionSchedule Deserialize(string payload)
        {
            if (payload is null) throw new ArgumentNullException(nameof(payload));
            string[] values = payload.Split('|');
            if (values.Length != 6)
                throw new InvalidDataException("The codegen contraction schedule is malformed.");
            int registerPrefetch = Parse(values[5]);
            if (registerPrefetch is not (0 or 1))
                throw new InvalidDataException("The register-prefetch field must be boolean.");
            var schedule = new CodegenTiledContractionSchedule(
                Parse(values[0]),
                Parse(values[1]),
                Parse(values[2]),
                Parse(values[3]),
                Parse(values[4]),
                registerPrefetch == 1);
            Validate(schedule);
            return schedule;
        }

        private static void Validate(CodegenTiledContractionSchedule schedule)
        {
            if (schedule is null) throw new ArgumentNullException(nameof(schedule));
            if (schedule.TileM <= 0 || schedule.TileN <= 0 || schedule.TileK <= 0 ||
                schedule.ThreadTileM <= 0 || schedule.ThreadTileN <= 0)
                throw new InvalidDataException("Codegen contraction schedule fields must be positive.");
        }

        private static int Parse(string value) =>
            int.TryParse(value, NumberStyles.None, CultureInfo.InvariantCulture, out int parsed)
                ? parsed
                : throw new InvalidDataException("The codegen contraction schedule contains an invalid integer.");

        private static string Format(int value) => value.ToString(CultureInfo.InvariantCulture);
    }

    private sealed class ScheduleVariation : IVariationOperator<CodegenTiledContractionSchedule>
    {
        private readonly int[] _tileM;
        private readonly int[] _tileN;
        private readonly int[] _tileK;
        private readonly int[] _threadTileM;
        private readonly int[] _threadTileN;

        internal ScheduleVariation(CodegenTiledContractionPlan plan)
        {
            _tileM = Divisors(plan.M, 128, 4);
            _tileN = Divisors(plan.N, 128, 4);
            _tileK = Divisors(plan.K, 64, plan.MatrixReductionMajor ? 1 : 4);
            _threadTileM = Divisors(plan.M, 16, 1);
            _threadTileN = Divisors(plan.N, 16, 1);
        }

        public string Id => "codegen-tiled-contraction-geometry-variation";
        public string VersionHash => "1";

        public ValueTask<CodegenTiledContractionSchedule> ProposeAsync(
            EvolutionVariationContext<CodegenTiledContractionSchedule> context,
            CancellationToken cancellationToken = default)
        {
            cancellationToken.ThrowIfCancellationRequested();
            CodegenTiledContractionSchedule parent =
                context.Parent.Candidate.CanonicalGenome.Genome;
            int tileM = parent.TileM;
            int tileN = parent.TileN;
            int tileK = parent.TileK;
            int threadTileM = parent.ThreadTileM;
            int threadTileN = parent.ThreadTileN;
            bool registerPrefetch = parent.RegisterPrefetch;
            int mutationCount = 1 + context.Random.NextInt(3);
            for (int mutation = 0; mutation < mutationCount; mutation++)
            {
                switch (context.Random.NextInt(6))
                {
                    case 0: tileM = Pick(_tileM, context.Random); break;
                    case 1: tileN = Pick(_tileN, context.Random); break;
                    case 2: tileK = Pick(_tileK, context.Random); break;
                    case 3: threadTileM = Pick(_threadTileM, context.Random); break;
                    case 4: threadTileN = Pick(_threadTileN, context.Random); break;
                    case 5: registerPrefetch = !registerPrefetch; break;
                }
            }
            return new ValueTask<CodegenTiledContractionSchedule>(
                new CodegenTiledContractionSchedule(
                    tileM, tileN, tileK, threadTileM, threadTileN, registerPrefetch));
        }

        private static int Pick(int[] values, StableRandom random) =>
            values[random.NextInt(values.Length)];

        private static int[] Divisors(int extent, int maximum, int quantum)
        {
            var values = new List<int>();
            for (int value = quantum; value <= Math.Min(extent, maximum); value += quantum)
                if (extent % value == 0) values.Add(value);
            if (values.Count == 0)
                throw new InvalidOperationException("The contraction extent has no valid schedule divisors.");
            return values.ToArray();
        }
    }
}
