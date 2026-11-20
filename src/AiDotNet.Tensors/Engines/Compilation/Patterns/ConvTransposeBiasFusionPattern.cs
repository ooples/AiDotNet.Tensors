// Copyright (c) CrowdTrainer LLC. All rights reserved.
// Built-in fusion: ConvTranspose2D -> channel bias add as one op on the GPU engine.
using System.Collections.Generic;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Compilation;

/// <summary>
/// Matches a float <c>ConvTranspose2D</c> recorded on <see cref="DirectGpuTensorEngine"/> whose output feeds exactly one
/// <c>TensorChannelBiasAdd</c> with a rank-1 bias (a transposed-convolution layer's bias) and emits one
/// <c>FusedConvTranspose2DBias</c> node: <see cref="IEngine.FusedConvTranspose2D{T}"/> adds the bias inside the
/// transposed convolution where the backend can (CUDA's single-GEMM route adds it in its overlap-add), so the biased
/// output is written once instead of written, read and written again, and the unbiased one is never stored. The backward
/// is the bias gradient (the output gradient summed over batch and space) and the transposed convolution's own backward
/// (<see cref="BackwardFunctions{T}.FusedConvTranspose2DBiasBackward"/>).
/// </summary>
internal sealed class ConvTransposeBiasFusionPattern : IFusionPattern
{
    public string Name => "ConvTranspose2DBias";

    /// <inheritdoc />
    public bool IsDifferentiable => true;   // records FusedConvTranspose2DBiasBackward

    public bool TryFuse(
        IReadOnlyList<ILazyNode> nodes, int index,
        IReadOnlyDictionary<ILazyNode, int> consumerCounts,
        HashSet<ILazyNode> alreadyRemoved,
        out ILazyNode? fused)
    {
        fused = null;
        if (nodes[index] is not LazyNode<float> conv || conv.OpName != "ConvTranspose2D"
            || !conv.RecordingEngine.Is<DirectGpuTensorEngine>()
            || conv.SavedState is not { Length: 2 } saved || saved[0] is not int[] stride || saved[1] is not int[] padding
            || stride.Length != 2 || padding.Length != 2
            || conv.Input1 is not { Rank: 4 } kernel || conv.Input0 is not { Rank: 4 } x || conv.Output is not { Rank: 4 } y
            || !PatternHelpers.FanOutAtMostOne(conv, consumerCounts))
            return false;
        LazyNode<float>? biasAdd = null;
        for (int j = index + 1; j < nodes.Count; j++)
        {
            if (alreadyRemoved.Contains(nodes[j]) || !PatternHelpers.IsConsumerOf(nodes[j], conv)) continue;
            if (biasAdd is not null || nodes[j] is not LazyNode<float> typed) return false;
            biasAdd = typed;
        }
        if (biasAdd is not { OpName: "TensorChannelBiasAdd" } || !ReferenceEquals(biasAdd.Input0, conv.Output)
            || biasAdd.Input1 is not { Rank: 1 } bias || bias.Shape._dims[0] != y.Shape._dims[1]
            || !PatternHelpers.SameShape(biasAdd.Output, conv.Output))
            return false;

        int sH = stride[0], sW = stride[1], pH = padding[0], pW = padding[1];
        // The output padding the recorded shape implies (the recording takes it as an argument; it is not saved).
        int opH = y.Shape._dims[2] - ((x.Shape._dims[2] - 1) * sH - 2 * pH + kernel.Shape._dims[2]);
        int opW = y.Shape._dims[3] - ((x.Shape._dims[3] - 1) * sW - 2 * pW + kernel.Shape._dims[3]);
        if (opH < 0 || opW < 0) return false;
        var node = new LazyNode<float>(LazyNodeType.Custom, "FusedConvTranspose2DBias", new[] { x, kernel, bias },
            biasAdd.Output,
            (eng, result) =>
            {
                var computed = eng.FusedConvTranspose2D(x, kernel, bias, sH, sW, pH, pW, opH, opW, FusedActivationType.None);
                DirectGpuTensorEngine.CopyResultInto(eng, computed, result);
            },
            BackwardFunctions<float>.FusedConvTranspose2DBiasBackward,
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
}
