// Copyright (c) AiDotNet. All rights reserved.

using System;
using System.Collections.Generic;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Compilation;

/// <summary>
/// Shared helpers used by <see cref="IFusionPattern"/> implementations.
/// </summary>
internal static class PatternHelpers
{
    /// <summary>
    /// Rewires the output tensor's <c>LazySource</c> to the fused node so
    /// downstream consumers find it via <see cref="ILazyNode.GetInputNodes"/>.
    /// Works for both float and double; no-op for other element types
    /// (which do not currently participate in lazy-graph compilation).
    /// </summary>
    public static void SetLazySource(ILazyNode node)
    {
        switch (node)
        {
            case LazyNode<float> f: f.Output.LazySource = f; break;
            case LazyNode<double> d: d.Output.LazySource = d; break;
        }
    }

    /// <summary>True if <paramref name="consumer"/> reads from <paramref name="producer"/>.</summary>
    public static bool IsConsumerOf(ILazyNode consumer, ILazyNode producer)
    {
        foreach (var input in consumer.GetInputNodes())
            if (ReferenceEquals(input, producer))
                return true;
        return false;
    }

    public static bool FanOutAtMostOne(
        ILazyNode node,
        IReadOnlyDictionary<ILazyNode, int> consumerCounts)
        => !consumerCounts.TryGetValue(node, out var fanOut) || fanOut <= 1;

    public static bool IsAdd(LazyNodeType type)
        => type == LazyNodeType.Add || type == LazyNodeType.BroadcastAdd;

    public static bool IsMultiply(LazyNodeType type)
        => type == LazyNodeType.Multiply || type == LazyNodeType.BroadcastMultiply;

    public static bool TryGetFloatScalar(LazyNode<float> node, out float value)
    {
        value = default;
        if (node.SavedState is null || node.SavedState.Length == 0 || node.SavedState[0] is null)
            return false;
        try
        {
            value = Convert.ToSingle(node.SavedState[0]);
            return !(float.IsNaN(value) || float.IsInfinity(value));
        }
        catch
        {
            value = default;
            return false;
        }
    }

    public static LazyNode<float>? FindNextFloat(
        IReadOnlyList<ILazyNode> nodes,
        int start,
        int maxInclusive,
        HashSet<ILazyNode> alreadyRemoved,
        Func<LazyNode<float>, bool> predicate)
    {
        int end = Math.Min(nodes.Count - 1, maxInclusive);
        for (int i = Math.Max(0, start); i <= end; i++)
        {
            if (alreadyRemoved.Contains(nodes[i])) continue;
            if (nodes[i] is LazyNode<float> candidate && predicate(candidate))
                return candidate;
        }
        return null;
    }

    public static bool TryGetOtherInput(
        LazyNode<float> node,
        Tensor<float> known,
        out Tensor<float> other)
    {
        if (ReferenceEquals(node.Input0, known) && node.Input1 is not null)
        {
            other = node.Input1;
            return true;
        }
        if (node.Input1 is not null && ReferenceEquals(node.Input1, known))
        {
            other = node.Input0;
            return true;
        }
        other = node.Input0;
        return false;
    }

    public static bool ConsumesOutput(LazyNode<float> consumer, LazyNode<float> producer)
        => ReferenceEquals(consumer.Input0, producer.Output)
            || (consumer.Input1 is not null && ReferenceEquals(consumer.Input1, producer.Output))
            || (consumer.Input2 is not null && ReferenceEquals(consumer.Input2, producer.Output));

    public static bool SameShape(Tensor<float> a, Tensor<float> b)
    {
        if (a._shape.Length != b._shape.Length) return false;
        for (int i = 0; i < a._shape.Length; i++)
            if (a._shape[i] != b._shape[i]) return false;
        return true;
    }
}
