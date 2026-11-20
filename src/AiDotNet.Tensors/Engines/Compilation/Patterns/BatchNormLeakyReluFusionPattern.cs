// Copyright (c) CrowdTrainer LLC. All rights reserved.
// Built-in fusion: training BatchNorm -> LeakyReLU as one op on the GPU engine.
using System.Collections.Generic;
using AiDotNet.Tensors.Engines.Autodiff;

namespace AiDotNet.Tensors.Engines.Compilation;

/// <summary>
/// Matches a training-mode <c>BatchNorm</c> node whose normalized output feeds exactly one <c>LeakyReLU</c> of positive
/// slope (a GAN discriminator's conv -&gt; BN -&gt; LeakyReLU) and emits one <c>BatchNormLeakyReLU</c> node: one kernel
/// writes LeakyReLU(BN(x)) and the backward scales and differentiates in one kernel, as
/// <see cref="BatchNormReluFusionPattern"/> does for ReLU. Each fused pair saves two full passes over the activation in
/// each direction, and the BatchNorm output is never stored.
/// </summary>
/// <remarks>
/// GPU-recorded graphs only (the CPU's fused kernels take ReLU alone). A slope of zero or below is left to the unfused
/// pair: the backward's mask comes from the LeakyReLU output, whose sign is its input's only for a positive slope.
/// </remarks>
internal sealed class BatchNormLeakyReluFusionPattern : IFusionPattern
{
    public string Name => "BatchNormLeakyReLU";

    /// <inheritdoc />
    public bool IsDifferentiable => true;   // records BatchNormLeakyReluBackward

    public bool TryFuse(
        IReadOnlyList<ILazyNode> nodes, int index,
        IReadOnlyDictionary<ILazyNode, int> consumerCounts,
        HashSet<ILazyNode> alreadyRemoved,
        out ILazyNode? fused)
    {
        fused = null;
        if (nodes[index] is not LazyNode<float> candidate || !candidate.RecordingEngine.Is<DirectGpuTensorEngine>())
            return false;
        if (!BatchNormFusionMatch.TryMatchBatchNorm(nodes, index, consumerCounts, alreadyRemoved,
                consumer => consumer is LazyNode<float> { OpName: "LeakyReLU", SavedState: { Length: 1 } state }
                            && state[0] is double s && s > 0.0 && s <= float.MaxValue,
                out var bn, out var leaky, out var statistics, out var mean, out var variance, out var epsilon))
            return false;
        // A successful match sets every output; checked here because net471 has no NotNullWhen to say so.
        if (bn is null || leaky is null || statistics is null || mean is null || variance is null
            || leaky.SavedState is not { Length: 1 } leakyState || leakyState[0] is not double slope)
            return false;
        var bnInputs = bn.GetInputsArray();
        var node = BatchNormFusionMatch.BuildFusedNode(bn, bnInputs[0], bnInputs[1], bnInputs[2], null, leaky.Output,
            statistics, mean, variance, epsilon, Name, BackwardFunctions<float>.BatchNormLeakyReluBackward,
            leakySlope: slope);
        fused = node;
        PatternHelpers.SetLazySource(node);
        alreadyRemoved.Add(bn);
        alreadyRemoved.Add(leaky);
        foreach (var s in statistics) alreadyRemoved.Add(s);
        return true;
    }
}
