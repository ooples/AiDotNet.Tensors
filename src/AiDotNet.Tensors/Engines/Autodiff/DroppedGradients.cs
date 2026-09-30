using System;
using System.Collections.Generic;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Autodiff;

/// <summary>
/// The gradients a cached backward replay computed but did not return (every tensor's gradient except the requested
/// sources'), handed from the replay to the tape's cleanup so their device buffers can go back to the pool.
/// </summary>
/// <remarks>
/// A replay (CompiledBackwardWalk, RebindablePlanCache) returns only the requested gradients. The rest used to be
/// dropped with its dictionary and freed only when the GC ran. The replay runs synchronously inside
/// GradientTape.ComputeGradients on one thread, so a per-thread slot connects the two without widening either
/// signature. The tape always takes the slot right after the replay, so nothing lingers into the next step.
/// </remarks>
internal static class DroppedGradients<T>
{
    [ThreadStatic] private static List<Tensor<T>>? t_dropped;

    /// <summary>Records every gradient in <paramref name="all"/> that <paramref name="returned"/> does not contain.</summary>
    internal static void Stash(Dictionary<Tensor<T>, Tensor<T>> all, Dictionary<Tensor<T>, Tensor<T>> returned)
    {
        List<Tensor<T>>? dropped = null;
        foreach (var pair in all)
            if (!returned.ContainsKey(pair.Key) && pair.Value is not null)
                (dropped ??= new List<Tensor<T>>()).Add(pair.Value);
        t_dropped = dropped;
    }

    /// <summary>Takes (and clears) the gradients the last replay on this thread dropped.</summary>
    internal static List<Tensor<T>>? Take()
    {
        var dropped = t_dropped;
        t_dropped = null;
        return dropped;
    }
}
