// Copyright (c) CrowdTrainer LLC. All rights reserved.
// Built-in fusion: convolution -> channel bias -> training BatchNorm -> activation, the bias folded into the BatchNorm.
using System.Collections.Generic;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Compilation;

/// <summary>
/// Matches a <c>Conv2D</c> or <c>ConvTranspose2D</c> whose only reader is a <c>TensorChannelBiasAdd</c> whose only reader
/// is a training-mode <c>BatchNorm</c> feeding one ReLU, ReLU6 or LeakyReLU of positive slope (a DCGAN or ResNet block
/// whose convolutions keep their bias), and emits one node that runs the convolution without the bias and the fused
/// BatchNorm + activation on it.
/// </summary>
/// <remarks>
/// <para>A training BatchNorm subtracts its batch mean per channel, so a per-channel constant added before it cancels:
/// BN(z + b) = BN(z) (the batch mean moves by b, the variance not at all), and the bias's gradient, the per-channel sum of
/// the BatchNorm's input gradient, is exactly zero (that gradient sums to zero over each channel). The fused node therefore
/// skips the bias pass and its gradient reduction, and gives the bias an exact zero gradient (the unfused graph computes
/// rounding noise around zero), so an optimizer still steps it as a parameter with a zero gradient. The bias is not
/// dropped: the batch-mean outputs the layer reads to update its running mean are mean(z) + b, as unfused, so an
/// inference-mode forward of the same layer is unchanged.</para>
/// <para>The normalized values are BN(z) rather than BN(fl(z + b)), within float rounding of the unfused graph.</para>
/// </remarks>
internal sealed class ConvBiasBatchNormFusionPattern : IFusionPattern
{
    public string Name => "ConvBiasBatchNorm";

    /// <inheritdoc />
    public bool IsDifferentiable => true;   // records ConvBiasBatchNormBackward

    public bool TryFuse(
        IReadOnlyList<ILazyNode> nodes, int index,
        IReadOnlyDictionary<ILazyNode, int> consumerCounts,
        HashSet<ILazyNode> alreadyRemoved,
        out ILazyNode? fused)
    {
        fused = null;
        if (nodes[index] is not LazyNode<float> conv || !conv.RecordingEngine.Is<DirectGpuTensorEngine>()
            || !(conv.OpName == "Conv2D" && conv.OpType == LazyNodeType.Conv2D || conv.OpName == "ConvTranspose2D")
            || conv.BackwardFn is null || conv.SavedState is null || conv.Output is not { Rank: 4 } z
            || conv.GetInputsArray() is not { Length: 2 } convInputs
            || !PatternHelpers.FanOutAtMostOne(conv, consumerCounts))
            return false;
        if (SoleConsumer(nodes, index, conv, alreadyRemoved) is not { OpName: "TensorChannelBiasAdd" } biasAdd
            || !ReferenceEquals(biasAdd.Input0, z) || biasAdd.Input1 is not { Rank: 1 } bias
            || bias.Shape._dims[0] != z.Shape._dims[1] || !PatternHelpers.SameShape(biasAdd.Output, z)
            || !PatternHelpers.FanOutAtMostOne(biasAdd, consumerCounts)
            || SoleConsumer(nodes, index, biasAdd, alreadyRemoved) is not { OpName: "BatchNorm" } bnCandidate)
            return false;
        int bnIndex = -1;
        for (int j = index + 1; j < nodes.Count; j++)
            if (ReferenceEquals(nodes[j], bnCandidate)) { bnIndex = j; break; }
        if (bnIndex < 0) return false;

        // The activation the BatchNorm feeds: the same three the BatchNorm fusions take without a residual.
        if (!BatchNormFusionMatch.TryMatchBatchNorm(nodes, bnIndex, consumerCounts, alreadyRemoved,
                consumer => consumer is LazyNode<float> { OpType: LazyNodeType.ReLU }
                            || consumer is LazyNode<float> { OpName: "ReLU6" }
                            || consumer is LazyNode<float> { OpName: "LeakyReLU", SavedState: { Length: 1 } state }
                               && state[0] is double s && s > 0.0 && s <= float.MaxValue,
                out var bn, out var activation, out var statistics, out var mean, out var variance, out var epsilon))
            return false;
        if (bn is null || activation is null || statistics is null || mean is null || variance is null) return false;
        var bnInputs = bn.GetInputsArray();
        if (!ReferenceEquals(bnInputs[0], biasAdd.Output)) return false;
        var gamma = bnInputs[1];
        var beta = bnInputs[2];

        string innerName;
        BackwardFunction<float> innerBackward;
        bool capAtSix = false;
        double? slope = null;
        if (activation.OpType == LazyNodeType.ReLU)
        {
            innerName = "BatchNormReLU";
            innerBackward = BackwardFunctions<float>.BatchNormReluBackward;
        }
        else if (activation.OpName == "LeakyReLU")
        {
            innerName = "BatchNormLeakyReLU";
            innerBackward = BackwardFunctions<float>.BatchNormLeakyReluBackward;
            slope = (double)activation.SavedState![0];
        }
        else
        {
            innerName = "BatchNormReLU6";
            innerBackward = BackwardFunctions<float>.BatchNormRelu6Backward;
            capAtSix = true;
        }

        // The BatchNorm + activation on the bias-free convolution output, its statistic outputs the batch statistics
        // of z: the mean ones get the bias added after (mean(z + b) = mean(z) + b; the variance is unchanged).
        var inner = BatchNormFusionMatch.BuildFusedNode(bn, z, gamma, beta, null, activation.Output, statistics, mean,
            variance, epsilon, innerName, innerBackward, capAtSix, slope);
        var meanOutputs = new List<Tensor<float>>(statistics.Count);
        foreach (var statistic in statistics)
            if (statistic.OpName == "BatchNormBatchMean") meanOutputs.Add(statistic.Output);

        var x = convInputs[0];
        var kernel = convInputs[1];
        var convExecute = conv.Execute;
        var innerExecute = inner.Execute;
        // Channels-last (ChannelsLastLayoutPass): the BatchNorm over a channels-last z takes ReLU and LeakyReLU (not
        // ReLU6) and channel counts in fours.
        int act = activation.OpType == LazyNodeType.ReLU ? 1 : activation.OpName == "LeakyReLU" ? 2 : 0;
        float actSlope = slope is { } sv ? (float)sv : 0f;
        var description = ChannelsLastConvolution.Describe(conv);
        var slot = description is null ? null
            : new ChannelsLastSlot(conv, description, x, outputEligible: act != 0 && (z.Shape._dims[1] & 3) == 0);
        var varianceOutputs = new List<Tensor<float>>(statistics.Count);
        foreach (var statistic in statistics)
            if (statistic.OpName != "BatchNormBatchMean") varianceOutputs.Add(statistic.Output);
        double bnEpsilon = epsilon;
        var node = new LazyNode<float>(LazyNodeType.Custom, Name, new[] { x, kernel, bias, gamma, beta }, activation.Output,
            (eng, result) =>
            {
                if (slot is { Any: true })
                {
                    var device = eng.TryGet<DirectGpuTensorEngine>(out var g) ? g
                        : throw new System.InvalidOperationException("A channels-last plan step ran on a non-GPU engine.");
                    var convolved = slot.Convolution.Forward(device, x, kernel, slot.InputNhwc, slot.OutputNhwc)
                        ?? throw new System.InvalidOperationException("The channels-last convolution did not run.");
                    DirectGpuTensorEngine.CopyResultInto(eng, convolved, z);
                    if (slot.OutputNhwc)
                    {
                        var normalized = device.TryBatchNormChannelsLast(z, gamma, beta, bnEpsilon, act, actSlope,
                                out var freshMean, out var freshVar)
                            ?? throw new System.InvalidOperationException("The channels-last BatchNorm did not run.");
                        DirectGpuTensorEngine.CopyResultInto(eng, normalized, result);
                        DirectGpuTensorEngine.CopyResultInto(eng, freshMean!, mean);
                        DirectGpuTensorEngine.CopyResultInto(eng, freshVar!, variance);
                        foreach (var varianceOutput in varianceOutputs)
                            DirectGpuTensorEngine.CopyResultInto(eng, variance, varianceOutput);
                    }
                    else
                    {
                        innerExecute(eng, result);
                    }
                    foreach (var meanOutput in meanOutputs)
                        DirectGpuTensorEngine.CopyResultInto(eng, eng.TensorAdd(mean, bias), meanOutput);
                    return;
                }
                convExecute(eng, z);        // the convolution alone: its bias cancels in the BatchNorm
                innerExecute(eng, result);  // BatchNorm + activation, writing the batch statistics of z
                // Out of place: the inner node may have aliased a batch-mean output to the saved mean its backward
                // normalizes z with, which must stay mean(z).
                foreach (var meanOutput in meanOutputs)
                {
                    var shifted = eng.TensorAdd(mean, bias);
                    DirectGpuTensorEngine.CopyResultInto(eng, shifted, meanOutput);
                }
            },
            BackwardFunctions<float>.ConvBiasBatchNormBackward,
            new object?[] { conv.BackwardFn, conv.SavedState, innerBackward, inner.SavedState!, z, slot, act,
                (double)actSlope }!)
        {
            RecordingEngine = conv.RecordingEngine,
        };
        // The fused node writes the absorbed statistic outputs (BuildFusedNode made the inner node their producer).
        foreach (var statistic in statistics)
            statistic.Output.LazySource = node;
        fused = node;
        PatternHelpers.SetLazySource(node);
        alreadyRemoved.Add(conv);
        alreadyRemoved.Add(biasAdd);
        alreadyRemoved.Add(bn);
        alreadyRemoved.Add(activation);
        foreach (var statistic in statistics) alreadyRemoved.Add(statistic);
        return true;
    }

    /// <summary>The one node after <paramref name="index"/> that reads <paramref name="producer"/>, or null.</summary>
    private static LazyNode<float>? SoleConsumer(IReadOnlyList<ILazyNode> nodes, int index, ILazyNode producer,
        HashSet<ILazyNode> alreadyRemoved)
    {
        LazyNode<float>? found = null;
        for (int j = index + 1; j < nodes.Count; j++)
        {
            if (alreadyRemoved.Contains(nodes[j]) || !PatternHelpers.IsConsumerOf(nodes[j], producer)) continue;
            if (found is not null || nodes[j] is not LazyNode<float> typed) return null;
            found = typed;
        }
        return found;
    }
}
