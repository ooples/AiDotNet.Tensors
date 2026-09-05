using System;
using System.Collections.Generic;
using System.Globalization;
using System.Text;
using AiDotNet.Evolution;
using AiDotNet.Tensors.Helpers.Autotune;

namespace AiDotNet.Tensors.Engines.Einsum;

/// <summary>
/// Serialization boundary between typed einsum paths and the legacy generic
/// autotune cache payload.
/// </summary>
internal static class EinsumPathCache
{
    private const int PayloadVersion = 1;
    private const string VersionParameter = "path-version";
    private const string PairParameter = "contraction-pairs";
    private const string FlopsParameter = "estimated-flops";
    private const string GreedyVariant = "greedy-path-v1";
    private const int MaximumMemoryEntries = 512;
    private static readonly object Sync = new();
    private static readonly Dictionary<MemoryKey, EinsumPath> Paths = new();
    private static readonly Queue<MemoryKey> InsertionOrder = new();

    internal static ShapeProfile CreateShape(EinsumShapeBinding binding)
    {
        if (binding is null) throw new ArgumentNullException(nameof(binding));
        var dimensions = new List<int>();
        dimensions.Add(binding.OperandShapes.Count);
        foreach (int[] operandShape in binding.OperandShapes)
        {
            dimensions.Add(operandShape.Length);
            dimensions.AddRange(operandShape);
        }
        return new ShapeProfile(dimensions.ToArray());
    }

    internal static KernelTuningIdentity CreateIdentity(
        EinsumShapeBinding binding,
        KernelTuningDeviceFingerprint device,
        KernelSearchSpaceVersion searchSpaceVersion,
        KernelBenchmarkProtocolVersion benchmarkProtocolVersion)
    {
        if (binding is null) throw new ArgumentNullException(nameof(binding));
        return new KernelTuningIdentity(
            new KernelId("einsum-evolution", EvolutionHash.Compute(binding.Equation.Source)),
            CreateShape(binding),
            device,
            searchSpaceVersion,
            benchmarkProtocolVersion);
    }

    internal static bool TryLoad(
        EinsumShapeBinding binding,
        KernelTuningIdentity identity,
        out EinsumPath? path)
    {
        if (binding is null) throw new ArgumentNullException(nameof(binding));
        if (identity is null) throw new ArgumentNullException(nameof(identity));
        ShapeProfile shape = CreateShape(binding);
        ValidateIdentity(binding, identity, shape);
        var memoryKey = new MemoryKey(identity.StableKey);
        lock (Sync)
        {
            if (Paths.TryGetValue(memoryKey, out EinsumPath? remembered))
            {
                path = remembered;
                return true;
            }
        }

        var codec = new EinsumEvolutionAutotuner.EinsumContractionOrderCodec(
            binding.Equation.Operands.Count);
        var deploymentStore = new AutotuneCacheKernelTuningStore<EinsumContractionOrder>();
        if (deploymentStore.TryLoad(
                identity,
                codec,
                out KernelTuningDeploymentSnapshot<EinsumContractionOrder>? snapshot) &&
            snapshot is not null)
        {
            try
            {
                EinsumPath evolved = EinsumPathOptimizer.BuildPath(
                    binding,
                    snapshot.Configuration,
                    EinsumPathStrategy.Evolutionary);
                Remember(memoryKey, evolved);
                path = evolved;
                return true;
            }
            catch (ArgumentException)
            {
            }
            catch (OverflowException)
            {
            }
            catch (KeyNotFoundException)
            {
            }
        }

        KernelChoice? choice = AutotuneCache.Lookup(CreateKernel(identity), shape);
        if (choice is null || choice.Parameters is null ||
            !TryReadStrategy(choice.Variant, out EinsumPathStrategy strategy) ||
            !choice.Parameters.TryGetValue(VersionParameter, out string? versionText) ||
            !int.TryParse(versionText, NumberStyles.None, CultureInfo.InvariantCulture, out int version) ||
            version != PayloadVersion ||
            !choice.Parameters.TryGetValue(PairParameter, out string? pairText) ||
            !TryParseOrder(pairText, out EinsumContractionOrder? order) || order is null)
        {
            path = null;
            return false;
        }

        try
        {
            EinsumPath candidate = EinsumPathOptimizer.BuildPath(binding, order, strategy);
            if (!choice.Parameters.TryGetValue(FlopsParameter, out string? flopsText) ||
                !long.TryParse(flopsText, NumberStyles.None, CultureInfo.InvariantCulture, out long storedFlops) ||
                storedFlops != candidate.TotalFlops)
            {
                path = null;
                return false;
            }
            Remember(memoryKey, candidate);
            path = candidate;
            return true;
        }
        catch (ArgumentException)
        {
            path = null;
            return false;
        }
        catch (OverflowException)
        {
            path = null;
            return false;
        }
        catch (KeyNotFoundException)
        {
            path = null;
            return false;
        }
    }

    internal static bool TryStore(
        EinsumShapeBinding binding,
        KernelTuningIdentity identity,
        EinsumPath path)
    {
        if (binding is null) throw new ArgumentNullException(nameof(binding));
        if (identity is null) throw new ArgumentNullException(nameof(identity));
        if (path is null) throw new ArgumentNullException(nameof(path));
        if (path.Strategy != EinsumPathStrategy.Greedy)
            throw new ArgumentException(
                "Only deterministic greedy paths use the planning cache.", nameof(path));

        ShapeProfile shape = CreateShape(binding);
        ValidateIdentity(binding, identity, shape);
        EinsumPath validatedPath = EinsumPathOptimizer.BuildPath(
            binding, path.ContractionOrder, path.Strategy);
        if (validatedPath.TotalFlops != path.TotalFlops)
            throw new ArgumentException(
                "The supplied einsum path does not match its typed contraction order.", nameof(path));
        bool stored = AutotuneCache.TryStore(
            CreateKernel(identity),
            shape,
            new KernelChoice
            {
                Variant = GreedyVariant,
                Parameters = new Dictionary<string, string>(StringComparer.Ordinal)
                {
                    [VersionParameter] = PayloadVersion.ToString(CultureInfo.InvariantCulture),
                    [PairParameter] = Serialize(validatedPath.ContractionOrder),
                    [FlopsParameter] = validatedPath.TotalFlops.ToString(CultureInfo.InvariantCulture)
                },
                MeasuredGflops = 0,
                MeasuredTimeMs = 0
            });
        Remember(new MemoryKey(identity.StableKey), validatedPath);
        return stored;
    }

    internal static void PublishEvolution(
        EinsumShapeBinding binding,
        KernelTuningIdentity identity,
        KernelTuningDeploymentSnapshot<EinsumContractionOrder> snapshot)
    {
        if (binding is null) throw new ArgumentNullException(nameof(binding));
        if (identity is null) throw new ArgumentNullException(nameof(identity));
        if (snapshot is null) throw new ArgumentNullException(nameof(snapshot));
        ShapeProfile shape = CreateShape(binding);
        ValidateIdentity(binding, identity, shape);
        if (!string.Equals(
                snapshot.Identity.StableKey,
                identity.StableKey,
                StringComparison.Ordinal))
        {
            throw new ArgumentException(
                "The deployment snapshot does not match the einsum tuning identity.",
                nameof(snapshot));
        }

        EinsumPath path = EinsumPathOptimizer.BuildPath(
            binding,
            snapshot.Configuration,
            EinsumPathStrategy.Evolutionary);
        Remember(new MemoryKey(identity.StableKey), path);
    }

    internal static void ClearMemoryForTests()
    {
        lock (Sync)
        {
            Paths.Clear();
            InsertionOrder.Clear();
        }
    }

    private static KernelId CreateKernel(KernelTuningIdentity identity) =>
        new("einsum-path", identity.StableKey);

    private static void ValidateIdentity(
        EinsumShapeBinding binding,
        KernelTuningIdentity identity,
        ShapeProfile shape)
    {
        var expectedKernel = new KernelId(
            "einsum-evolution", EvolutionHash.Compute(binding.Equation.Source));
        if (identity.Kernel != expectedKernel || !identity.Shape.Equals(shape))
            throw new ArgumentException(
                "The tuning identity does not describe this einsum binding.", nameof(identity));
    }

    private static string Serialize(EinsumContractionOrder order)
    {
        var text = new StringBuilder();
        for (int i = 0; i < order.Pairs.Count; i++)
        {
            if (i != 0) text.Append(';');
            EinsumContractionPair pair = order.Pairs[i];
            text.Append(pair.LeftIndex.ToString(CultureInfo.InvariantCulture));
            text.Append(':');
            text.Append(pair.RightIndex.ToString(CultureInfo.InvariantCulture));
        }
        return text.ToString();
    }

    private static bool TryParseOrder(string payload, out EinsumContractionOrder? order)
    {
        if (payload.Length == 0)
        {
            order = new EinsumContractionOrder(Array.Empty<EinsumContractionPair>());
            return true;
        }

        string[] pairTexts = payload.Split(';');
        var pairs = new EinsumContractionPair[pairTexts.Length];
        for (int i = 0; i < pairTexts.Length; i++)
        {
            string[] indices = pairTexts[i].Split(':');
            if (indices.Length != 2 ||
                !int.TryParse(indices[0], NumberStyles.None, CultureInfo.InvariantCulture, out int left) ||
                !int.TryParse(indices[1], NumberStyles.None, CultureInfo.InvariantCulture, out int right) ||
                left < 0 || right <= left)
            {
                order = null;
                return false;
            }
            pairs[i] = new EinsumContractionPair(left, right);
        }
        order = new EinsumContractionOrder(pairs);
        return true;
    }

    private static bool TryReadStrategy(string value, out EinsumPathStrategy strategy)
    {
        if (string.Equals(value, GreedyVariant, StringComparison.Ordinal))
        {
            strategy = EinsumPathStrategy.Greedy;
            return true;
        }
        strategy = default;
        return false;
    }

    private static void Remember(MemoryKey key, EinsumPath path)
    {
        lock (Sync)
        {
            if (Paths.ContainsKey(key))
            {
                Paths[key] = path;
                return;
            }
            Paths.Add(key, path);
            InsertionOrder.Enqueue(key);
            while (Paths.Count > MaximumMemoryEntries && InsertionOrder.Count != 0)
                Paths.Remove(InsertionOrder.Dequeue());
        }
    }

    private readonly record struct MemoryKey(string IdentityKey);
}
