// Copyright (c) CrowdTrainer LLC. All rights reserved.
// Built-in fusion: DepthwiseConv2D -> channel bias add as one node on the GPU engine.
using System.Collections.Generic;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Compilation;

/// <summary>
/// Matches a float <c>DepthwiseConv2D</c> recorded on <see cref="DirectGpuTensorEngine"/> whose output feeds exactly one
/// <c>TensorChannelBiasAdd</c> (a depthwise convolution layer's bias) and emits one <c>DepthwiseConv2DBias</c> node: the
/// convolution is written into the node's output and the bias added there in place
/// (<see cref="DirectGpuTensorEngine.TryChannelBiasAddInPlace{T}"/>). Neither backward reads the raw convolution output,
/// yet a compiled step kept it beside the biased one for the whole step: MobileNetV2 held 17 of them, 12 MB.
/// </summary>
/// <remarks>
/// The add is the one <c>TensorChannelBiasAdd</c> computes, and the backward
/// (<see cref="BackwardFunctions{T}.DepthwiseConv2DBiasBackward"/>) reduces the bias gradient and runs the depthwise
/// backward on the same gradient, so the fused node trains bit for bit like the two nodes it replaces.
/// </remarks>
internal sealed class DepthwiseConvBiasFusionPattern : IFusionPattern
{
    public string Name => "DepthwiseConv2DBias";

    /// <inheritdoc />
    public bool IsDifferentiable => true;   // records DepthwiseConv2DBiasBackward

    public bool TryFuse(
        IReadOnlyList<ILazyNode> nodes, int index,
        IReadOnlyDictionary<ILazyNode, int> consumerCounts,
        HashSet<ILazyNode> alreadyRemoved,
        out ILazyNode? fused)
    {
        fused = null;
        if (nodes[index] is not LazyNode<float> conv || conv.OpType != LazyNodeType.DepthwiseConv2D
            || conv.OpName != "DepthwiseConv2D" || conv.RecordingEngine is not DirectGpuTensorEngine
            || conv.SavedState is not { Length: 2 } saved || saved[0] is not int[] stride || saved[1] is not int[] padding
            || stride.Length != 2 || padding.Length != 2 || conv.Input1 is not { } kernel
            || !PatternHelpers.FanOutAtMostOne(conv, consumerCounts))
            return false;
        var x = conv.Input0;
        if (SoleConsumer(nodes, index, conv, alreadyRemoved) is not { OpName: "TensorChannelBiasAdd" } biasAdd
            || !ReferenceEquals(biasAdd.Input0, conv.Output) || biasAdd.Input1 is not { Rank: 1 } bias
            || !PatternHelpers.SameShape(biasAdd.Output, conv.Output))
            return false;

        var node = new LazyNode<float>(LazyNodeType.Custom, "DepthwiseConv2DBias", new[] { x, kernel, bias }, biasAdd.Output,
            (eng, result) =>
            {
                var computed = eng.DepthwiseConv2D(x, kernel, stride, padding);
                DirectGpuTensorEngine.CopyResultInto(eng, computed, result);
                if (eng is DirectGpuTensorEngine device && device.TryChannelBiasAddInPlace(result, bias)) return;
                var biased = eng.TensorChannelBiasAdd(result, bias);
                DirectGpuTensorEngine.CopyResultInto(eng, biased, result);
            },
            BackwardFunctions<float>.DepthwiseConv2DBiasBackward,
            new object[] { stride, padding })
        {
            RecordingEngine = conv.RecordingEngine,
        };
        fused = node;
        PatternHelpers.SetLazySource(node);
        alreadyRemoved.Add(conv);
        alreadyRemoved.Add(biasAdd);
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
