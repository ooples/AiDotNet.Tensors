using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using AiDotNet.Evolution;
using AiDotNet.Tensors.Helpers.Autotune;

namespace AiDotNet.Tensors.Engines.Einsum;

/// <summary>Result of an explicit offline evolutionary einsum path search.</summary>
public sealed class EinsumEvolutionTuningResult
{
    internal EinsumEvolutionTuningResult(
        EvolutionKernelTuningResult<EinsumContractionOrder> evolution,
        EinsumPath proposedPath,
        EinsumPath activePath)
    {
        Evolution = evolution ?? throw new ArgumentNullException(nameof(evolution));
        ProposedPath = proposedPath ?? throw new ArgumentNullException(nameof(proposedPath));
        ActivePath = activePath ?? throw new ArgumentNullException(nameof(activePath));
    }

    /// <summary>Detailed deterministic evolution run result.</summary>
    public EvolutionKernelTuningResult<EinsumContractionOrder> Evolution { get; }

    /// <summary>Best locally valid path proposed by this run.</summary>
    public EinsumPath ProposedPath { get; }

    /// <summary>Path deployed after applying the measured promotion threshold.</summary>
    public EinsumPath ActivePath { get; }

    /// <summary>Whether this run replaced the previously deployed path.</summary>
    public bool WasPromoted => Evolution.WasPromoted;
}

/// <summary>
/// Evolves typed einsum contraction orders during an explicit offline or idle
/// workflow and publishes only correctness-validated, stably measured winners.
/// </summary>
/// <remarks>
/// This API is intentionally absent from the serving path. Normal einsum calls
/// perform a bounded in-memory lookup through <see cref="EinsumPathOptimizer.Optimize"/>;
/// an application or benchmark invokes this tuner separately and supplies the
/// real executor measurement delegate.
/// </remarks>
public static class EinsumEvolutionAutotuner
{
    /// <summary>Tunes a contraction order and records the active typed path for future dispatch.</summary>
    public static async Task<EinsumEvolutionTuningResult> TuneAsync(
        EinsumShapeBinding binding,
        KernelTuningDeviceFingerprint device,
        Func<EinsumPath, EvolutionEvaluationContext, CancellationToken,
            ValueTask<KernelTuningTrialResult>> evaluator,
        KernelSearchSpaceVersion searchSpaceVersion,
        KernelBenchmarkProtocolVersion benchmarkProtocolVersion,
        IEnumerable<EinsumContractionOrder>? additionalSeeds = null,
        EvolutionEngineOptions? engineOptions = null,
        KernelTuningOptions? tuningOptions = null,
        IEvolutionCheckpointStore? checkpointStore = null,
        KernelTuningDeploymentRegistry<EinsumContractionOrder>? deploymentRegistry = null,
        IKernelTuningStore<EinsumContractionOrder>? store = null,
        CancellationToken cancellationToken = default)
    {
        if (binding is null) throw new ArgumentNullException(nameof(binding));
        if (evaluator is null) throw new ArgumentNullException(nameof(evaluator));
        EnsureSearchable(binding);

        var codec = new EinsumContractionOrderCodec(binding.Equation.Operands.Count);
        var variation = new EinsumContractionOrderVariation(binding.Equation.Operands.Count);
        KernelTuningIdentity identity = CreateIdentity(
            binding, device, searchSpaceVersion, benchmarkProtocolVersion);
        EvolutionEngineOptions resolvedEngineOptions = engineOptions ?? DefaultEngineOptions(identity);
        var tuner = new EvolutionKernelAutotuner<EinsumContractionOrder>(
            identity,
            codec,
            variation,
            async (order, context, token) =>
            {
                EinsumPath path;
                try
                {
                    path = EinsumPathOptimizer.BuildPath(
                        binding, order, EinsumPathStrategy.Evolutionary);
                }
                catch (ArgumentException exception)
                {
                    return KernelTuningTrialResult.Rejected(
                        KernelTuningTrialStatus.InvalidConfiguration,
                        exception.Message);
                }
                catch (OverflowException exception)
                {
                    return KernelTuningTrialResult.Rejected(
                        KernelTuningTrialStatus.ResourceLimitExceeded,
                        exception.Message);
                }
                return await evaluator(path, context, token).ConfigureAwait(false);
            },
            resolvedEngineOptions,
            tuningOptions,
            checkpointStore: checkpointStore,
            deploymentRegistry: deploymentRegistry,
            store: store,
            deploymentValidator: order => IsValidForBinding(binding, order));

        EinsumContractionOrder[] seeds = BuildSeeds(
            binding, additionalSeeds, codec, tuner.MaximumProposals);
        EvolutionKernelTuningResult<EinsumContractionOrder> evolution =
            await tuner.TuneAsync(seeds, cancellationToken).ConfigureAwait(false);
        EinsumPath proposed = EinsumPathOptimizer.BuildPath(
            binding,
            evolution.ProposedWinner.Configuration,
            EinsumPathStrategy.Evolutionary);
        EinsumPath active = EinsumPathOptimizer.BuildPath(
            binding,
            evolution.ActiveDeployment.Configuration,
            EinsumPathStrategy.Evolutionary);
        EinsumPathCache.PublishEvolution(binding, identity, evolution.ActiveDeployment);
        return new EinsumEvolutionTuningResult(evolution, proposed, active);
    }

    /// <summary>
    /// Tunes after a caller-owned idle gate admits the work, so benchmarking
    /// never competes with foreground inference by accident.
    /// </summary>
    public static async Task<EinsumEvolutionTuningResult> TuneInBackgroundAsync(
        EinsumShapeBinding binding,
        KernelTuningDeviceFingerprint device,
        Func<EinsumPath, EvolutionEvaluationContext, CancellationToken,
            ValueTask<KernelTuningTrialResult>> evaluator,
        KernelSearchSpaceVersion searchSpaceVersion,
        KernelBenchmarkProtocolVersion benchmarkProtocolVersion,
        IKernelTuningIdleGate idleGate,
        IEnumerable<EinsumContractionOrder>? additionalSeeds = null,
        EvolutionEngineOptions? engineOptions = null,
        KernelTuningOptions? tuningOptions = null,
        IEvolutionCheckpointStore? checkpointStore = null,
        KernelTuningDeploymentRegistry<EinsumContractionOrder>? deploymentRegistry = null,
        IKernelTuningStore<EinsumContractionOrder>? store = null,
        CancellationToken cancellationToken = default)
    {
        if (idleGate is null) throw new ArgumentNullException(nameof(idleGate));
        if (binding is null) throw new ArgumentNullException(nameof(binding));
        KernelTuningIdentity identity = CreateIdentity(
            binding, device, searchSpaceVersion, benchmarkProtocolVersion);
        await idleGate.WaitUntilIdleAsync(identity, cancellationToken).ConfigureAwait(false);
        return await Task.Run(
            () => TuneAsync(
                binding,
                device,
                evaluator,
                searchSpaceVersion,
                benchmarkProtocolVersion,
                additionalSeeds,
                engineOptions,
                tuningOptions,
                checkpointStore,
                deploymentRegistry,
                store,
                cancellationToken),
            cancellationToken).ConfigureAwait(false);
    }

    private static KernelTuningIdentity CreateIdentity(
        EinsumShapeBinding binding,
        KernelTuningDeviceFingerprint device,
        KernelSearchSpaceVersion searchSpaceVersion,
        KernelBenchmarkProtocolVersion benchmarkProtocolVersion) =>
        EinsumPathCache.CreateIdentity(
            binding, device, searchSpaceVersion, benchmarkProtocolVersion);

    private static EvolutionEngineOptions DefaultEngineOptions(KernelTuningIdentity identity) => new()
    {
        RunId = "einsum-path-" + identity.StableKey,
        MaxEvaluationAttempts = 64,
        MaxProposals = 512,
        MaxGenerations = 512,
        ProposalBatchSize = 1,
        MaxDegreeOfParallelism = 1,
        IslandCount = 1,
        MigrationInterval = 0,
        MigrantsPerIsland = 1
    };

    private static EinsumContractionOrder[] BuildSeeds(
        EinsumShapeBinding binding,
        IEnumerable<EinsumContractionOrder>? additionalSeeds,
        EinsumContractionOrderCodec codec,
        int maximumProposals)
    {
        var seeds = new List<EinsumContractionOrder>
        {
            EinsumPathOptimizer.Greedy(binding).ContractionOrder
        };
        if (additionalSeeds is not null) seeds.AddRange(additionalSeeds);

        var unique = new List<EinsumContractionOrder>(Math.Min(seeds.Count, maximumProposals));
        var seen = new HashSet<string>(StringComparer.Ordinal);
        foreach (EinsumContractionOrder seed in seeds)
        {
            string canonical = codec.Serialize(seed);
            if (seen.Add(canonical)) unique.Add(seed);
            if (unique.Count == maximumProposals) break;
        }
        return unique.ToArray();
    }

    private static void EnsureSearchable(EinsumShapeBinding binding)
    {
        if (binding.Equation.Operands.Count < 3)
            throw new ArgumentException(
                "Evolutionary path search requires at least three operands.", nameof(binding));
    }

    private static bool IsValidForBinding(
        EinsumShapeBinding binding,
        EinsumContractionOrder order)
    {
        try
        {
            _ = EinsumPathOptimizer.BuildPath(binding, order, EinsumPathStrategy.Evolutionary);
            return true;
        }
        catch (ArgumentException)
        {
            return false;
        }
        catch (OverflowException)
        {
            return false;
        }
        catch (KeyNotFoundException)
        {
            return false;
        }
    }

    internal sealed class EinsumContractionOrderCodec : IEvolutionGenomeCodec<EinsumContractionOrder>
    {
        private readonly int _operandCount;

        internal EinsumContractionOrderCodec(int operandCount)
        {
            _operandCount = operandCount;
        }

        public string Id => "einsum-contraction-order";
        public string VersionHash => "1";

        public string Serialize(EinsumContractionOrder genome)
        {
            Validate(genome);
            return string.Join(";", genome.Pairs.Select(pair =>
                pair.LeftIndex.ToString(CultureInfo.InvariantCulture) + ":" +
                pair.RightIndex.ToString(CultureInfo.InvariantCulture)));
        }

        public EinsumContractionOrder Deserialize(string payload)
        {
            if (payload is null) throw new ArgumentNullException(nameof(payload));
            if (payload.Length == 0)
            {
                var empty = new EinsumContractionOrder(Array.Empty<EinsumContractionPair>());
                Validate(empty);
                return empty;
            }

            string[] values = payload.Split(';');
            var pairs = new EinsumContractionPair[values.Length];
            for (int i = 0; i < values.Length; i++)
            {
                string[] indices = values[i].Split(':');
                if (indices.Length != 2 ||
                    !int.TryParse(indices[0], NumberStyles.None, CultureInfo.InvariantCulture, out int left) ||
                    !int.TryParse(indices[1], NumberStyles.None, CultureInfo.InvariantCulture, out int right))
                    throw new InvalidDataException("The einsum contraction genome is malformed.");
                pairs[i] = new EinsumContractionPair(left, right);
            }
            var order = new EinsumContractionOrder(pairs);
            Validate(order);
            return order;
        }

        private void Validate(EinsumContractionOrder order)
        {
            if (order is null) throw new ArgumentNullException(nameof(order));
            int expected = _operandCount - 1;
            if (order.Pairs.Count != expected)
                throw new InvalidDataException(
                    "The einsum contraction genome has the wrong number of steps.");
            for (int i = 0; i < order.Pairs.Count; i++)
            {
                int liveCount = _operandCount - i;
                if (order.Pairs[i].LeftIndex < 0 ||
                    order.Pairs[i].RightIndex <= order.Pairs[i].LeftIndex ||
                    order.Pairs[i].RightIndex >= liveCount)
                    throw new InvalidDataException(
                        "The einsum contraction genome indexes outside the live operand list.");
            }
        }
    }

    private sealed class EinsumContractionOrderVariation : IVariationOperator<EinsumContractionOrder>
    {
        private readonly int _operandCount;

        internal EinsumContractionOrderVariation(int operandCount)
        {
            _operandCount = operandCount;
        }

        public string Id => "einsum-contraction-order-variation";
        public string VersionHash => "1";

        public ValueTask<EinsumContractionOrder> ProposeAsync(
            EvolutionVariationContext<EinsumContractionOrder> context,
            CancellationToken cancellationToken = default)
        {
            cancellationToken.ThrowIfCancellationRequested();
            EinsumContractionPair[] pairs =
                context.Parent.Candidate.CanonicalGenome.Genome.Pairs.ToArray();

            if (context.Inspirations.Count != 0 && context.Random.NextInt(2) == 0)
            {
                int sourceIndex = context.Random.NextInt(context.Inspirations.Count);
                IReadOnlyList<EinsumContractionPair> source =
                    context.Inspirations[sourceIndex].Candidate.CanonicalGenome.Genome.Pairs;
                int crossoverIndex = context.Random.NextInt(pairs.Length);
                pairs[crossoverIndex] = source[crossoverIndex];
            }

            int mutationCount = 1 + context.Random.NextInt(Math.Min(3, pairs.Length));
            for (int mutation = 0; mutation < mutationCount; mutation++)
            {
                int step = context.Random.NextInt(pairs.Length);
                int liveCount = _operandCount - step;
                int left = context.Random.NextInt(liveCount - 1);
                int right = left + 1 + context.Random.NextInt(liveCount - left - 1);
                pairs[step] = new EinsumContractionPair(left, right);
            }
            return new ValueTask<EinsumContractionOrder>(new EinsumContractionOrder(pairs));
        }
    }
}
