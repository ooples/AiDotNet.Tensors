// Copyright (c) CrowdTrainer LLC. All rights reserved.
// Built-in fusion: Conv2D -> channel bias add [-> ReLU] as one op on the GPU engine.
using System.Collections.Generic;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Compilation;

/// <summary>
/// Matches a float <c>Conv2D</c> recorded on <see cref="DirectGpuTensorEngine"/> whose output feeds exactly one
/// <c>TensorChannelBiasAdd</c> (a convolution layer's bias), optionally followed by exactly one ReLU, and emits one
/// <c>FusedConv2DBias</c> node: <see cref="IEngine.FusedConv2D{T}"/> writes act(conv + bias) in one pass, and the
/// backward reduces the bias gradient and runs the convolution backward
/// (<see cref="BackwardFunctions{T}.FusedConv2DBiasActivationBackward"/>). The raw convolution output (and with a ReLU
/// the biased one) is never stored: a GPU training step kept one more full activation per convolution for each.
/// </summary>
/// <remarks>
/// The CPU plan fuses the same chain its own way (its convolution epilogue). A stride-2 convolution that shares its input
/// with another stride-2 convolution is left alone: the plan computes the input gradient of ResNet's downsampling pair in
/// one call (CompiledTrainingPlan.DetectShortcutConvPairs), which needs the plain convolutions.
/// </remarks>
internal sealed class ConvBiasFusionPattern : IFusionPattern
{
    public string Name => "Conv2DBias";

    /// <inheritdoc />
    public bool IsDifferentiable => true;   // records FusedConv2DBiasActivationBackward

    public bool TryFuse(
        IReadOnlyList<ILazyNode> nodes, int index,
        IReadOnlyDictionary<ILazyNode, int> consumerCounts,
        HashSet<ILazyNode> alreadyRemoved,
        out ILazyNode? fused)
    {
        fused = null;
        if (nodes[index] is not LazyNode<float> conv || conv.OpType != LazyNodeType.Conv2D || conv.OpName != "Conv2D"
            || conv.RecordingEngine is not DirectGpuTensorEngine
            || conv.SavedState is not { Length: 3 } saved || saved[0] is not int[] stride || saved[1] is not int[] padding
            || saved[2] is not int[] dilation || stride.Length != 2 || padding.Length != 2 || dilation.Length != 2
            || conv.Input1 is not { } kernel || !PatternHelpers.FanOutAtMostOne(conv, consumerCounts))
            return false;
        var x = conv.Input0;
        if ((stride[0] == 2 || stride[1] == 2) && SharesInputWithAnotherStridedConv(nodes, conv, alreadyRemoved))
            return false;
        if (SoleConsumer(nodes, index, conv, alreadyRemoved) is not { OpName: "TensorChannelBiasAdd" } biasAdd
            || !ReferenceEquals(biasAdd.Input0, conv.Output) || biasAdd.Input1 is not { Rank: 1 } bias
            || !PatternHelpers.SameShape(biasAdd.Output, conv.Output))
            return false;
        LazyNode<float>? relu = null;
        if (PatternHelpers.FanOutAtMostOne(biasAdd, consumerCounts)
            && SoleConsumer(nodes, index, biasAdd, alreadyRemoved) is { OpType: LazyNodeType.ReLU } r
            && ReferenceEquals(r.Input0, biasAdd.Output))
            relu = r;

        var output = relu?.Output ?? biasAdd.Output;
        var activation = relu is null ? FusedActivationType.None : FusedActivationType.ReLU;
        int sH = stride[0], sW = stride[1], pH = padding[0], pW = padding[1], dH = dilation[0], dW = dilation[1];
        var node = new LazyNode<float>(LazyNodeType.Custom, "FusedConv2DBias", new[] { x, kernel, bias }, output,
            (eng, result) =>
            {
                var computed = eng.FusedConv2D(x, kernel, bias, sH, sW, pH, pW, dH, dW, activation);
                DirectGpuTensorEngine.CopyResultInto(eng, computed, result);
            },
            BackwardFunctions<float>.FusedConv2DBiasActivationBackward,
            new object[] { stride, padding, dilation, relu is not null })
        {
            RecordingEngine = conv.RecordingEngine,
        };
        fused = node;
        PatternHelpers.SetLazySource(node);
        alreadyRemoved.Add(conv);
        alreadyRemoved.Add(biasAdd);
        if (relu is not null) alreadyRemoved.Add(relu);
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

    private static bool SharesInputWithAnotherStridedConv(IReadOnlyList<ILazyNode> nodes, LazyNode<float> conv,
        HashSet<ILazyNode> alreadyRemoved)
    {
        foreach (var node in nodes)
            if (!ReferenceEquals(node, conv) && !alreadyRemoved.Contains(node)
                && node is LazyNode<float> { OpType: LazyNodeType.Conv2D } other && ReferenceEquals(other.Input0, conv.Input0)
                && other.SavedState is { Length: >= 1 } s && s[0] is int[] otherStride && otherStride.Length == 2
                && (otherStride[0] == 2 || otherStride[1] == 2))
                return true;
        return false;
    }
}
