// Copyright (c) AiDotNet. All rights reserved.

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
}
