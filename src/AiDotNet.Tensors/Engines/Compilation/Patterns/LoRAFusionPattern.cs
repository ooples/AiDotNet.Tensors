// Copyright (c) AiDotNet. All rights reserved.
// Issue #301 — LoRA-forward chain detection for graph-mode auto-fusion.

using System.Collections.Generic;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Compilation;

/// <summary>
/// Matches the LoRA adapter forward chain — five lazy nodes:
/// <code>
///   intermed = MatMul(input, A)            // [batch, rank]
///   delta    = MatMul(intermed, B)         // [batch, out]
///   scaled   = MultiplyScalar(delta, α)    // [batch, out]
///   output   = Add(baseOutput, scaled)     // [batch, out]
/// </code>
/// and emits a single <see cref="LazyNodeType.FusedLoRA"/> node that
/// dispatches to <see cref="AiDotNet.Tensors.Helpers.CpuFusedOperations.FusedLoRAForward"/>.
/// </summary>
internal sealed class LoRAFusionPattern : IFusionPattern
{
    public string Name => "LoRA";

    public bool TryFuse(
        IReadOnlyList<ILazyNode> nodes, int index,
        IReadOnlyDictionary<ILazyNode, int> consumerCounts,
        HashSet<ILazyNode> alreadyRemoved,
        out ILazyNode? fused)
    {
        fused = null;
        if (nodes[index] is not LazyNode<float> matmulA || matmulA.OpType != LazyNodeType.MatMul)
            return false;
        if (!PatternHelpers.FanOutAtMostOne(matmulA, consumerCounts))
            return false;

        var matmulB = PatternHelpers.FindNextFloat(
            nodes, index + 1, index + 6, alreadyRemoved,
            n => n.OpType == LazyNodeType.MatMul
                && PatternHelpers.ConsumesOutput(n, matmulA)
                && PatternHelpers.FanOutAtMostOne(n, consumerCounts));
        if (matmulB is null)
            return false;

        var scaled = PatternHelpers.FindNextFloat(
            nodes, index + 1, index + 8, alreadyRemoved,
            n => n.OpType == LazyNodeType.MultiplyScalar
                && PatternHelpers.ConsumesOutput(n, matmulB)
                && PatternHelpers.FanOutAtMostOne(n, consumerCounts));
        if (scaled is null || !PatternHelpers.TryGetFloatScalar(scaled, out var scaling))
            return false;

        var add = PatternHelpers.FindNextFloat(
            nodes, index + 1, index + 10, alreadyRemoved,
            n => PatternHelpers.IsAdd(n.OpType)
                && PatternHelpers.ConsumesOutput(n, scaled));
        if (add is null || !PatternHelpers.TryGetOtherInput(add, scaled.Output, out var baseOutput))
            return false;

        fused = Build(matmulA, matmulB, scaled, add, baseOutput, scaling);
        if (fused is null)
            return false;

        PatternHelpers.SetLazySource(fused);
        alreadyRemoved.Add(matmulA);
        alreadyRemoved.Add(matmulB);
        alreadyRemoved.Add(scaled);
        alreadyRemoved.Add(add);
        return true;
    }

    private static LazyNode<float>? Build(
        LazyNode<float> matmulA,
        LazyNode<float> matmulB,
        LazyNode<float> scaled,
        LazyNode<float> add,
        Tensor<float> baseOutput,
        float scaling)
    {
        if (!ReferenceEquals(matmulB.Input0, matmulA.Output))
            return null;

        var input = matmulA.Input0;
        var loraA = matmulA.Input1;
        var loraB = matmulB.Input1;
        if (loraA is null || loraB is null)
            return null;

        if (input.Rank != 2 || loraA.Rank != 2 || loraB.Rank != 2 || baseOutput.Rank != 2 || add.Output.Rank != 2)
            return null;

        int batch = input._shape[0];
        int inFeat = input._shape[1];
        int rank = loraA._shape[1];
        int outFeat = loraB._shape[1];

        if (loraA._shape[0] != inFeat || loraB._shape[0] != rank)
            return null;
        if (baseOutput._shape[0] != batch || baseOutput._shape[1] != outFeat)
            return null;
        if (add.Output._shape[0] != batch || add.Output._shape[1] != outFeat)
            return null;
        if (!PatternHelpers.SameShape(scaled.Output, add.Output))
            return null;

        matmulA.Output.LazySource = null;
        matmulB.Output.LazySource = null;
        scaled.Output.LazySource = null;
        add.Output.LazySource = null;

        var capturedInput = input;
        var capturedBase = baseOutput;
        var capturedA = loraA;
        var capturedB = loraB;
        var capturedScaling = scaling;
        var finalOutput = add.Output;

        return new LazyNode<float>(
            LazyNodeType.FusedLoRA,
            "FusedLoRAForward",
            new[] { capturedInput, capturedBase, capturedA, capturedB },
            finalOutput,
            (eng, output) =>
            {
                if (eng is DirectGpuTensorEngine gpu)
                {
                    var eager = gpu.FusedLoRAForward(capturedInput, capturedBase, capturedA, capturedB, capturedScaling);
                    eager.AsSpan().CopyTo(output.AsWritableSpan());
                }
                else
                {
                    CpuFusedOperations.FusedLoRAForward(
                        capturedInput, capturedBase, capturedA, capturedB, capturedScaling, output);
                }
            },
            backwardFn: null,
            savedState: new object[] { capturedScaling });
    }
}
