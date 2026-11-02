// Copyright (c) CrowdTrainer LLC. All rights reserved.
// Built-in fusion: training BatchNorm -> ReLU6 as one node.
using System.Collections.Generic;
using AiDotNet.Tensors.Engines.Autodiff;

namespace AiDotNet.Tensors.Engines.Compilation;

/// <summary>
/// Matches a training-mode <c>BatchNorm</c> node whose normalized output feeds exactly one <c>ReLU6</c> (the expansion
/// and depthwise stages of every MobileNetV2 block) and emits one <c>BatchNormReLU6</c> node. A training step keeps every
/// recorded tensor until its backward has run, and the ReLU6's backward read its input, so each pair kept the BatchNorm
/// output beside the ReLU6 output. The fused node keeps only its output; its backward takes the mask from that output
/// (<see cref="BackwardFunctions{T}.BatchNormRelu6Backward"/>).
/// </summary>
/// <remarks>
/// The match, the absorbed batch-statistic nodes and the refreshed saved mean and variance are those of
/// <see cref="BatchNormReluFusionPattern"/>. PyTorch's MobileNetV2 runs its ReLU6 in place on the BatchNorm output to the
/// same end.
/// </remarks>
internal sealed class BatchNormRelu6FusionPattern : IFusionPattern
{
    public string Name => "BatchNormReLU6";

    /// <inheritdoc />
    public bool IsDifferentiable => true;   // records BatchNormRelu6Backward

    public bool TryFuse(
        IReadOnlyList<ILazyNode> nodes, int index,
        IReadOnlyDictionary<ILazyNode, int> consumerCounts,
        HashSet<ILazyNode> alreadyRemoved,
        out ILazyNode? fused)
    {
        fused = null;
        if (!BatchNormFusionMatch.TryMatchBatchNorm(nodes, index, consumerCounts, alreadyRemoved,
                consumer => consumer is LazyNode<float> { OpName: "ReLU6" },
                out var bn, out var relu6, out var statistics, out var mean, out var variance, out var epsilon))
            return false;
        // A successful match sets every output; checked here because net471 has no NotNullWhen to say so.
        if (bn is null || relu6 is null || statistics is null || mean is null || variance is null) return false;
        var bnInputs = bn.GetInputsArray();
        var node = BatchNormFusionMatch.BuildFusedNode(bn, bnInputs[0], bnInputs[1], bnInputs[2], null, relu6.Output,
            statistics, mean, variance, epsilon, Name, BackwardFunctions<float>.BatchNormRelu6Backward, capAtSix: true);
        fused = node;
        PatternHelpers.SetLazySource(node);
        alreadyRemoved.Add(bn);
        alreadyRemoved.Add(relu6);
        foreach (var s in statistics) alreadyRemoved.Add(s);
        return true;
    }
}
