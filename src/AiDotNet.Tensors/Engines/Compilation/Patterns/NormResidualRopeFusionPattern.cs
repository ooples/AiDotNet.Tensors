// Copyright (c) CrowdTrainer LLC. All rights reserved.

using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Compilation.GeneratedKernels;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Compilation;

/// <summary>
/// Cross-layer vertical fusion pattern matching <c>Add(x, residual) -&gt; RMSNorm(sum, gamma, eps) -&gt; SplitHalfRoPE(norm, cos, sin)</c>
/// (11 primitive lazy nodes) and collapsing them into a single <c>FusedRmsNormResidualRope</c> node without materializing
/// intermediate residual-sum or normalized hidden tensors in global memory.
/// </summary>
internal sealed class NormResidualRopeFusionPattern : IFusionPattern
{
    public string Name => "RmsNormResidualRopeCrossLayer";

    public bool IsDifferentiable => true;

    public bool TryFuse(
        IReadOnlyList<ILazyNode> nodes,
        int index,
        IReadOnlyDictionary<ILazyNode, int> consumerCounts,
        HashSet<ILazyNode> alreadyRemoved,
        out ILazyNode? fused)
    {
        fused = null;
        if (nodes[index] is not LazyNode<float> addNode
            || addNode.OpType != LazyNodeType.Add
            || addNode.Input1 is null
            || !PatternHelpers.SameShape(addNode.Input0, addNode.Input1)
            || !PatternHelpers.FanOutAtMostOne(addNode, consumerCounts))
        {
            return false;
        }

        int rmsIdx = -1;
        LazyNode<float>? rmsNode = null;
        int searchEnd = Math.Min(nodes.Count - 1, index + 5);
        for (int j = index + 1; j <= searchEnd; j++)
        {
            if (alreadyRemoved.Contains(nodes[j])) continue;
            if (nodes[j] is LazyNode<float> candidate
                && candidate.OpType == LazyNodeType.Custom
                && candidate.OpName == "RMSNorm"
                && ReferenceEquals(candidate.Input0, addNode.Output))
            {
                rmsNode = candidate;
                rmsIdx = j;
                break;
            }
        }

        if (rmsNode is null || rmsNode.Input1 is null)
            return false;
        // RMSNorm output must feed ONLY the two half-slices of split-half RoPE, plus its own RMS statistic node when
        // nothing reads or retains that statistic (the fused node drops it with the RMSNorm).
        LazyNode<float>? rmsStat = null;
        for (int j = rmsIdx + 1; j < nodes.Count && j <= rmsIdx + 2; j++)
        {
            if (nodes[j] is LazyNode<float> stat && stat.OpName == "RMSNormRms" && ReferenceEquals(stat.Input0, rmsNode.Output))
            {
                if (alreadyRemoved.Contains(stat) || (consumerCounts.TryGetValue(stat, out int sc) && sc > 0))
                    return false;
                rmsStat = stat;
                break;
            }
        }
        if (!consumerCounts.TryGetValue(rmsNode, out int rmsConsumers) || rmsConsumers != (rmsStat is null ? 2 : 3))
            return false;

        int rank = rmsNode.Output.Rank;
        if (rank < 2)
            return false;
        int headDim = rmsNode.Output._shape[rank - 1];
        if (headDim < 2 || (headDim & 1) != 0 || rmsNode.Input1.Length != headDim)
            return false;

        // Reuse RotaryPositionEmbeddingFusionPattern starting at the first TensorSlice consuming rmsNode.Output!
        int slice0Idx = -1;
        for (int j = rmsIdx + 1; j < nodes.Count && j <= rmsIdx + 4; j++)
        {
            if (alreadyRemoved.Contains(nodes[j])) continue;
            if (nodes[j] is LazyNode<float> s
                && s.OpType == LazyNodeType.Custom
                && s.OpName == "TensorSlice"
                && ReferenceEquals(s.Input0, rmsNode.Output))
            {
                slice0Idx = j;
                break;
            }
        }
        if (slice0Idx < 0)
            return false;

        var ropeRemovedProbe = new HashSet<ILazyNode>(alreadyRemoved);
        var ropePattern = new RotaryPositionEmbeddingFusionPattern();
        if (!ropePattern.TryFuse(nodes, slice0Idx, consumerCounts, ropeRemovedProbe, out var ropeFusedRaw)
            || ropeFusedRaw is not LazyNode<float> ropeFused)
        {
            return false;
        }

        if (rmsNode.SavedState is not { Length: >= 2 } rmsState || rmsState[1] is null)
            return false;
        float epsilon = Convert.ToSingle(rmsState[1], System.Globalization.CultureInfo.InvariantCulture);
        var savedRmsTensor = rmsState[0] as Tensor<float>;

        var ropeInputs = ropeFused.GetInputsArray();
        if (ropeInputs.Length != 3)
            return false;

        var xInput = addNode.Input0;
        var resInput = addNode.Input1;
        var gamma = rmsNode.Input1;
        var cosTensor = ropeInputs[1];
        var sinTensor = ropeInputs[2];

        int seqLen = rmsNode.Output._shape[rank - 2];
        int outerCount = rmsNode.Output.Length / (seqLen * headDim);
        int rows = outerCount * seqLen;
        var rmsHolder = savedRmsTensor ?? new Tensor<float>(new[] { rows });
        int trigStride = cosTensor._shape[cosTensor.Rank - 1];
        var ropeInput = ropeInputs[0];

        addNode.Output.LazySource = null;
        rmsNode.Output.LazySource = null;

        var fusedNode = new LazyNode<float>(
            LazyNodeType.Custom,
            "FusedRmsNormResidualRope",
            new[] { xInput, resInput, gamma, cosTensor, sinTensor },
            ropeFused.Output,
            (eng, output) =>
            {
                if (eng.TryGet<DirectGpuTensorEngine>(out var gpuDirect)
                    && gpuDirect.TryExecuteFusedRmsNormResidualRopeInto(
                        xInput,
                        resInput,
                        gamma,
                        cosTensor,
                        sinTensor,
                        output,
                        outerCount,
                        seqLen,
                        headDim,
                        cosTensor._shape[cosTensor.Rank - 1],
                        epsilon))
                {
                    return;
                }

                if (!eng.Is<DirectGpuTensorEngine>()
                    && xInput.IsContiguous
                    && resInput.IsContiguous
                    && gamma.IsContiguous
                    && cosTensor.IsContiguous
                    && sinTensor.IsContiguous
                    && output.IsContiguous)
                {
                    var normTemp = TensorAllocator.Rent<float>(output._shape);
                    CompiledFusedKernels.RmsNormResidualForward(
                        xInput.AsSpan(),
                        resInput.AsSpan(),
                        gamma.AsSpan(),
                        normTemp.AsWritableSpan(),
                        Span<float>.Empty,
                        rmsHolder.AsWritableSpan(),
                        rows,
                        headDim,
                        epsilon);
                    CompiledFusedKernels.RotaryPositionEmbeddingSplitHalf(
                        normTemp.AsSpan(),
                        cosTensor.AsSpan(),
                        sinTensor.AsSpan(),
                        output.AsWritableSpan(),
                        outerCount,
                        seqLen,
                        headDim,
                        inverseAdjoint: false);
                    output.IncrementVersion();
                    rmsHolder.IncrementVersion();
                    return;
                }

                var normOut = new Tensor<float>(output._shape);
                if (eng.TryGet<DirectGpuTensorEngine>(out var gpu)
                    && gpu.TryExecuteFusedRmsNormResidualInto(xInput, resInput, gamma, normOut, rmsHolder, rows, headDim, epsilon)
                    && gpu.TryExecuteFusedRotarySplitHalfInto(
                        normOut, cosTensor, sinTensor, output, outerCount, seqLen, headDim,
                        cosTensor._shape[cosTensor.Rank - 1], inverseAdjoint: false))
                {
                    return;
                }

                var sum = eng.TensorAdd(xInput, resInput);
                var normed = eng.RMSNorm(sum, gamma, epsilon, out var freshRms);
                DirectGpuTensorEngine.CopyResultInto(eng, freshRms, rmsHolder);
                // The RoPE node reads the RMSNorm output tensor, whose producer this fusion removed: give it this
                // forward's normalized values before it runs.
                DirectGpuTensorEngine.CopyResultInto(eng, normed, ropeInput);
                ropeFused.Execute(eng, output);
            },
            (gradOutput, inputs, _, _, engine, grads) =>
            {
                var x = inputs[0];
                var res = inputs[1];
                var g = inputs[2];
                var cos = inputs[3];
                var sin = inputs[4];
                bool needX = DifferentiableOps.IsGradientRequired(x);
                bool needRes = DifferentiableOps.IsGradientRequired(res);
                bool needGamma = DifferentiableOps.IsGradientRequired(g);
                bool needCos = DifferentiableOps.IsGradientRequired(cos);
                bool needSin = DifferentiableOps.IsGradientRequired(sin);
                if (!needX && !needRes && !needGamma && !needCos && !needSin)
                    return;

                // Logical-order operands: a strided view (a transpose) has no flat span.
                var xc = x.Contiguous();
                var resc = res.Contiguous();
                var gc = g.Contiguous();
                var cosc = cos.Contiguous();
                var sinc = sin.Contiguous();
                var gOut = gradOutput.Contiguous();

                // The RMS (and the RoPE input) recomputed from this step's inputs: the single-kernel CUDA forward
                // writes no statistics, so a holder could still carry an earlier step's values.
                var normed = TensorAllocator.Rent<float>(x._shape);
                var rms = new float[rows];
                CompiledFusedKernels.RmsNormResidualForward(
                    xc.AsSpan(), resc.AsSpan(), gc.AsSpan(), normed.AsWritableSpan(), Span<float>.Empty, rms,
                    rows, headDim, epsilon);

                if (needCos || needSin)
                {
                    var dCos = needCos ? TensorAllocator.Rent<float>(cos._shape) : null;
                    var dSin = needSin ? TensorAllocator.Rent<float>(sin._shape) : null;
                    RotaryPositionEmbeddingFusionPattern.TableGradients(
                        gOut.AsSpan(), normed.AsSpan(),
                        dCos is not null ? dCos.AsWritableSpan() : Span<float>.Empty,
                        dSin is not null ? dSin.AsWritableSpan() : Span<float>.Empty,
                        outerCount, seqLen, headDim, trigStride);
                    if (dCos is not null) { dCos.IncrementVersion(); DifferentiableOps.AccumulateGrad(grads, cos, dCos, engine); }
                    if (dSin is not null) { dSin.IncrementVersion(); DifferentiableOps.AccumulateGrad(grads, sin, dSin, engine); }
                }

                if (!needX && !needRes && !needGamma)
                    return;

                var dNorm = TensorAllocator.Rent<float>(x._shape);
                CompiledFusedKernels.RotaryPositionEmbeddingSplitHalf(
                    gOut.AsSpan(),
                    cosc.AsSpan(),
                    sinc.AsSpan(),
                    dNorm.AsWritableSpan(),
                    outerCount,
                    seqLen,
                    headDim,
                    inverseAdjoint: true);
                dNorm.IncrementVersion();

                var sharedGrad = TensorAllocator.Rent<float>(x._shape);
                var gammaGrad = needGamma ? TensorAllocator.Rent<float>(g._shape) : null;
                CompiledFusedKernels.RmsNormResidualBackward(
                    dNorm.AsSpan(),
                    xc.AsSpan(),
                    resc.AsSpan(),
                    gc.AsSpan(),
                    rms,
                    sharedGrad.AsWritableSpan(),
                    gammaGrad is not null ? gammaGrad.AsWritableSpan() : Span<float>.Empty,
                    rows,
                    headDim);
                sharedGrad.IncrementVersion();

                if (needX)
                    DifferentiableOps.AccumulateGrad(grads, x, sharedGrad, engine);
                if (needRes)
                    DifferentiableOps.AccumulateGrad(grads, res, needX ? sharedGrad.Clone() : sharedGrad, engine);
                if (gammaGrad is not null)
                {
                    gammaGrad.IncrementVersion();
                    DifferentiableOps.AccumulateGrad(grads, g, gammaGrad, engine);
                }
            },
            null)
        {
            RecordingEngine = ropeFused.RecordingEngine
        };

        fused = fusedNode;
        PatternHelpers.SetLazySource(fusedNode);
        foreach (var rem in ropeRemovedProbe)
        {
            alreadyRemoved.Add(rem);
        }
        alreadyRemoved.Add(addNode);
        alreadyRemoved.Add(rmsNode);
        if (rmsStat is not null)
        {
            rmsStat.Output.LazySource = null;
            alreadyRemoved.Add(rmsStat);
        }
        return true;
    }
}
