// Copyright (c) CrowdTrainer LLC. All rights reserved.
// Built-in fusion: training BatchNorm -> ReLU as one op on the GPU engine.
using System.Collections.Generic;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Compilation;

/// <summary>
/// Matches a training-mode <c>BatchNorm</c> node whose normalized output feeds exactly one <c>ReLU</c> (the conv -&gt; BN
/// -&gt; ReLU of every residual network) and emits one <c>BatchNormReLU</c> node: one kernel writes ReLU(BN(x)) and the
/// backward masks and differentiates in one kernel. Each fused pair saves two full passes over the activation in each
/// direction (the BN output written and re-read, the ReLU's own read and write).
/// </summary>
/// <remarks>
/// <para>Float graphs recorded on <see cref="DirectGpuTensorEngine"/> or <see cref="CpuEngine"/> are fused (the CPU only
/// for 4D inputs, which its fused kernels take). The BatchNorm's batch-statistic nodes (<c>BatchNormBatchMean</c> /
/// <c>BatchNormBatchVariance</c>) read the normalized output only to order themselves after it; they are absorbed, and the
/// fused node keeps writing those statistic tensors as their producer, so a reader of them (a running-statistics EMA) is
/// ordered after it.</para>
/// <para>The fused node refreshes the BatchNorm's saved mean and variance tensors every replay, exactly as the
/// BatchNorm node does, because its backward reads them.</para>
/// </remarks>
internal sealed class BatchNormReluFusionPattern : IFusionPattern
{
    public string Name => "BatchNormReLU";

    /// <inheritdoc />
    public bool IsDifferentiable => true;   // records BatchNormReluBackward

    public bool TryFuse(
        IReadOnlyList<ILazyNode> nodes, int index,
        IReadOnlyDictionary<ILazyNode, int> consumerCounts,
        HashSet<ILazyNode> alreadyRemoved,
        out ILazyNode? fused)
    {
        fused = null;
        if (!BatchNormFusionMatch.TryMatchBatchNorm(nodes, index, consumerCounts, alreadyRemoved,
                consumer => consumer is LazyNode<float> { OpType: LazyNodeType.ReLU },
                out var bn, out var relu, out var statistics, out var mean, out var variance, out var epsilon))
            return false;
        // A successful match sets every output; checked here because net471 has no NotNullWhen to say so.
        if (bn is null || relu is null || statistics is null || mean is null || variance is null) return false;
        var bnInputs = bn.GetInputsArray();
        var node = BatchNormFusionMatch.BuildFusedNode(bn, bnInputs[0], bnInputs[1], bnInputs[2], null, relu.Output,
            statistics, mean, variance, epsilon, "BatchNormReLU", BackwardFunctions<float>.BatchNormReluBackward);
        fused = node;
        PatternHelpers.SetLazySource(node);
        alreadyRemoved.Add(bn);
        alreadyRemoved.Add(relu);
        foreach (var s in statistics) alreadyRemoved.Add(s);
        return true;
    }
}
