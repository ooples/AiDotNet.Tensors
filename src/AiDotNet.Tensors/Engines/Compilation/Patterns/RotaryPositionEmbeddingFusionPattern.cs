// Copyright (c) CrowdTrainer LLC. All rights reserved.

using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Compilation.GeneratedKernels;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Compilation;

/// <summary>
/// Matches decomposed split-half Rotary Positional Embedding (RoPE) subgraphs:
/// <c>x0 = Slice(x, 0..D/2), x1 = Slice(x, D/2..D), Concat([x0*cos - x1*sin, x1*cos + x0*sin], axis=-1)</c>
/// and collapses all 9 primitive nodes into a single <c>FusedRotaryPositionEmbedding</c> node with exact
/// orthogonal adjoint backward (<c>sin -&gt; -sin</c>).
/// </summary>
internal sealed class RotaryPositionEmbeddingFusionPattern : IFusionPattern
{
    public string Name => "RotaryPositionEmbedding";

    public bool IsDifferentiable => true;

    public bool TryFuse(
        IReadOnlyList<ILazyNode> nodes,
        int index,
        IReadOnlyDictionary<ILazyNode, int> consumerCounts,
        HashSet<ILazyNode> alreadyRemoved,
        out ILazyNode? fused)
    {
        fused = null;
        if (nodes[index] is not LazyNode<float> slice0
            || slice0.OpType != LazyNodeType.Custom
            || slice0.OpName != "TensorSlice")
        {
            return false;
        }

        var fullInput = slice0.Input0;
        int rank = fullInput.Rank;
        if (rank < 2)
            return false;
        int headDim = fullInput._shape[rank - 1];
        if (headDim < 2 || (headDim & 1) != 0)
            return false;
        int halfDim = headDim >> 1;

        if (!IsTrailingHalfSlice(slice0, fullInput._shape, 0, halfDim))
            return false;
        if (!HasExactConsumerCount(slice0, consumerCounts, 2))
            return false;

        // Locate sibling second-half slice x1 = TensorSlice(fullInput, [..., halfDim], [..., halfDim]).
        var slice1 = PatternHelpers.FindNextFloat(
            nodes,
            index + 1,
            index + 6,
            alreadyRemoved,
            n => n.OpType == LazyNodeType.Custom
                && n.OpName == "TensorSlice"
                && ReferenceEquals(n.Input0, fullInput)
                && IsTrailingHalfSlice(n, fullInput._shape, halfDim, halfDim)
                && HasExactConsumerCount(n, consumerCounts, 2));

        if (slice1 is null)
            return false;

        // Locate Subtract(Mul(x0, cos), Mul(x1, sin)) within window.
        var subNode = PatternHelpers.FindNextFloat(
            nodes,
            index + 2,
            index + 12,
            alreadyRemoved,
            n => (n.OpType == LazyNodeType.Subtract || n.OpType == LazyNodeType.BroadcastSubtract)
                && n.Input1 is not null
                && PatternHelpers.FanOutAtMostOne(n, consumerCounts));

        if (subNode is null || subNode.Input1 is null)
            return false;
        if (subNode.Input0.LazySource is not LazyNode<float> mul0Cos
            || subNode.Input1.LazySource is not LazyNode<float> mul1Sin
            || alreadyRemoved.Contains(mul0Cos)
            || alreadyRemoved.Contains(mul1Sin)
            || !PatternHelpers.IsMultiply(mul0Cos.OpType)
            || !PatternHelpers.IsMultiply(mul1Sin.OpType)
            || !PatternHelpers.FanOutAtMostOne(mul0Cos, consumerCounts)
            || !PatternHelpers.FanOutAtMostOne(mul1Sin, consumerCounts))
        {
            return false;
        }

        if (!PatternHelpers.TryGetOtherInput(mul0Cos, slice0.Output, out var cosTensor))
            return false;
        if (!PatternHelpers.TryGetOtherInput(mul1Sin, slice1.Output, out var sinTensor))
            return false;

        // Locate Add(Mul(x1, cos), Mul(x0, sin)) (in either operand order).
        var addNode = PatternHelpers.FindNextFloat(
            nodes,
            index + 2,
            index + 14,
            alreadyRemoved,
            n => PatternHelpers.IsAdd(n.OpType)
                && n.Input1 is not null
                && PatternHelpers.FanOutAtMostOne(n, consumerCounts));

        if (addNode is null || addNode.Input1 is null)
            return false;
        if (addNode.Input0.LazySource is not LazyNode<float> addLhs
            || addNode.Input1.LazySource is not LazyNode<float> addRhs
            || alreadyRemoved.Contains(addLhs)
            || alreadyRemoved.Contains(addRhs)
            || !PatternHelpers.IsMultiply(addLhs.OpType)
            || !PatternHelpers.IsMultiply(addRhs.OpType)
            || !PatternHelpers.FanOutAtMostOne(addLhs, consumerCounts)
            || !PatternHelpers.FanOutAtMostOne(addRhs, consumerCounts))
        {
            return false;
        }

        LazyNode<float>? mul1Cos = null;
        LazyNode<float>? mul0Sin = null;
        if (MatchesPair(addLhs, slice1.Output, cosTensor) && MatchesPair(addRhs, slice0.Output, sinTensor))
        {
            mul1Cos = addLhs;
            mul0Sin = addRhs;
        }
        else if (MatchesPair(addRhs, slice1.Output, cosTensor) && MatchesPair(addLhs, slice0.Output, sinTensor))
        {
            mul1Cos = addRhs;
            mul0Sin = addLhs;
        }
        if (mul1Cos is null || mul0Sin is null)
            return false;

        // Locate terminal Concat([subNode.Output, addNode.Output], axis = rank - 1).
        var concatNode = PatternHelpers.FindNextFloat(
            nodes,
            index + 4,
            index + 16,
            alreadyRemoved,
            n =>
            {
                if (n.OpType != LazyNodeType.Custom || (n.OpName != "Concat" && n.OpName != "Concatenate"))
                    return false;
                var concatInputs = n.GetInputsArray();
                return concatInputs.Length == 2
                       && ReferenceEquals(concatInputs[0], subNode.Output)
                       && ReferenceEquals(concatInputs[1], addNode.Output);
            });

        if (concatNode is null || !TryExtractConcatAxis(concatNode, rank, out int concatAxis) || concatAxis != rank - 1)
            return false;

        int seqLen = fullInput._shape[rank - 2];
        int outerCount = fullInput.Length / (seqLen * headDim);
        if (seqLen <= 0 || outerCount <= 0)
            return false;

        if (!ValidateTrigShape(cosTensor, seqLen, halfDim, headDim, out int trigStride)
            || !ValidateTrigShape(sinTensor, seqLen, halfDim, headDim, out int sinStride)
            || sinStride != trigStride)
        {
            return false;
        }

        LazyNode<float>[] intermediates = { slice0, slice1, mul0Cos, mul1Sin, subNode, mul1Cos, mul0Sin, addNode };
        foreach (var inter in intermediates)
        {
            inter.Output.LazySource = null;
        }

        var node = new LazyNode<float>(
            LazyNodeType.Custom,
            "FusedRotaryPositionEmbedding",
            new[] { fullInput, cosTensor, sinTensor },
            concatNode.Output,
            (eng, output) =>
            {
                if (eng.TryGet<DirectGpuTensorEngine>(out var gpu)
                    && gpu.TryExecuteFusedRotarySplitHalfInto(
                        fullInput, cosTensor, sinTensor, output, outerCount, seqLen, headDim, trigStride, inverseAdjoint: false))
                {
                    return;
                }

                if (!eng.Is<DirectGpuTensorEngine>()
                    && fullInput.IsContiguous
                    && cosTensor.IsContiguous
                    && sinTensor.IsContiguous
                    && output.IsContiguous)
                {
                    CompiledFusedKernels.RotaryPositionEmbeddingSplitHalf(
                        fullInput.AsSpan(),
                        cosTensor.AsSpan(),
                        sinTensor.AsSpan(),
                        output.AsWritableSpan(),
                        outerCount,
                        seqLen,
                        headDim,
                        inverseAdjoint: false);
                    output.IncrementVersion();
                    return;
                }

                ExecuteRopeFallback(eng, fullInput, cosTensor, sinTensor, output, rank, halfDim, inverseAdjoint: false);
            },
            (gradOutput, inputs, _, _, engine, grads) =>
            {
                var x = inputs[0];
                var cos = inputs[1];
                var sin = inputs[2];
                if (!DifferentiableOps.IsGradientRequired(x))
                    return;

                var dX = TensorAllocator.Rent<float>(x._shape);
                if (engine.TryGet<DirectGpuTensorEngine>(out var gpu)
                    && gpu.TryExecuteFusedRotarySplitHalfInto(
                        gradOutput, cos, sin, dX, outerCount, seqLen, headDim, trigStride, inverseAdjoint: true))
                {
                    DifferentiableOps.AccumulateGrad(grads, x, dX, engine);
                    return;
                }

                if (!engine.Is<DirectGpuTensorEngine>()
                    && gradOutput.IsContiguous
                    && cos.IsContiguous
                    && sin.IsContiguous
                    && dX.IsContiguous)
                {
                    CompiledFusedKernels.RotaryPositionEmbeddingSplitHalf(
                        gradOutput.AsSpan(),
                        cos.AsSpan(),
                        sin.AsSpan(),
                        dX.AsWritableSpan(),
                        outerCount,
                        seqLen,
                        headDim,
                        inverseAdjoint: true);
                    dX.IncrementVersion();
                    DifferentiableOps.AccumulateGrad(grads, x, dX, engine);
                    return;
                }

                ExecuteRopeFallback(engine, gradOutput, cos, sin, dX, rank, halfDim, inverseAdjoint: true);
                DifferentiableOps.AccumulateGrad(grads, x, dX, engine);
            },
            null)
        {
            RecordingEngine = concatNode.RecordingEngine
        };

        fused = node;
        PatternHelpers.SetLazySource(node);
        foreach (var inter in intermediates)
        {
            alreadyRemoved.Add(inter);
        }
        alreadyRemoved.Add(concatNode);
        return true;
    }

    private static bool IsTrailingHalfSlice(LazyNode<float> slice, int[] fullShape, int expectedStart, int halfDim)
    {
        if (slice.SavedState is not { Length: >= 1 } state || state[0] is not int[] starts)
            return false;
        if (starts.Length != fullShape.Length || slice.OutputShape.Length != fullShape.Length)
            return false;
        for (int d = 0; d < fullShape.Length - 1; d++)
        {
            if (starts[d] != 0 || slice.OutputShape[d] != fullShape[d])
                return false;
        }
        return starts[fullShape.Length - 1] == expectedStart
               && slice.OutputShape[fullShape.Length - 1] == halfDim;
    }

    private static bool HasExactConsumerCount(ILazyNode node, IReadOnlyDictionary<ILazyNode, int> counts, int expected)
        => counts.TryGetValue(node, out int c) && c == expected;

    private static bool MatchesPair(LazyNode<float> mul, Tensor<float> a, Tensor<float> b)
        => (ReferenceEquals(mul.Input0, a) && ReferenceEquals(mul.Input1, b))
           || (ReferenceEquals(mul.Input0, b) && ReferenceEquals(mul.Input1, a));

    private static bool TryExtractConcatAxis(LazyNode<float> concat, int rank, out int axis)
    {
        axis = -1;
        if (concat.SavedState is not { Length: >= 1 } state || state[0] is not int raw)
            return false;
        axis = raw < 0 ? raw + rank : raw;
        return axis >= 0 && axis < rank;
    }

    private static bool ValidateTrigShape(Tensor<float> trig, int seqLen, int halfDim, int headDim, out int trigStride)
    {
        trigStride = 0;
        if (trig.Rank < 2)
            return false;
        int last = trig._shape[trig.Rank - 1];
        int prev = trig._shape[trig.Rank - 2];
        if (prev != seqLen || (last != halfDim && last != headDim))
            return false;
        for (int d = 0; d < trig.Rank - 2; d++)
        {
            if (trig._shape[d] != 1)
                return false;
        }
        trigStride = last;
        return true;
    }

    private static void ExecuteRopeFallback(
        IEngine eng,
        Tensor<float> x,
        Tensor<float> cos,
        Tensor<float> sin,
        Tensor<float> dst,
        int rank,
        int halfDim,
        bool inverseAdjoint)
    {
        var start0 = new int[rank];
        var start1 = new int[rank];
        var len = (int[])x._shape.Clone();
        len[rank - 1] = halfDim;
        start1[rank - 1] = halfDim;

        var x0 = eng.TensorSlice(x, start0, len);
        var x1 = eng.TensorSlice(x, start1, len);
        var sEff = inverseAdjoint ? eng.TensorNegate(sin) : sin;

        var m0c = new Tensor<float>(len);
        var m1s = new Tensor<float>(len);
        var m1c = new Tensor<float>(len);
        var m0s = new Tensor<float>(len);
        eng.TensorBroadcastMultiplyInto(m0c, x0, cos);
        eng.TensorBroadcastMultiplyInto(m1s, x1, sEff);
        eng.TensorBroadcastMultiplyInto(m1c, x1, cos);
        eng.TensorBroadcastMultiplyInto(m0s, x0, sEff);

        var y0 = eng.TensorSubtract(m0c, m1s);
        var y1 = eng.TensorAdd(m1c, m0s);
        var res = eng.Concat(new Tensor<float>[] { y0, y1 }, rank - 1);
        DirectGpuTensorEngine.CopyResultInto(eng, res, dst);
    }
}
