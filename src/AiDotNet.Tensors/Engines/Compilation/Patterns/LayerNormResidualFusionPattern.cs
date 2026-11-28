// Copyright (c) CrowdTrainer LLC. All rights reserved.

using System;
using System.Collections.Generic;
using System.Globalization;
using System.Reflection;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Compilation.GeneratedKernels;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Compilation;

/// <summary>
/// Matches <c>Add(input, residual) -&gt; LayerNorm(sum, gamma, beta, epsilon)</c> and emits a single
/// <c>FusedLayerNormResidual</c> node that executes without materializing the intermediate residual-sum tensor
/// in global memory (using compile-time generated AVX2/FMA SIMD on CPU and warp-shuffle NVRTC CUDA on GPU).
/// </summary>
internal sealed class LayerNormResidualFusionPattern : IFusionPattern
{
    public string Name => "LayerNormResidual";

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

        int lnIndex = -1;
        LazyNode<float>? lnNode = null;
        int searchEnd = Math.Min(nodes.Count - 1, index + 6);
        for (int j = index + 1; j <= searchEnd; j++)
        {
            if (alreadyRemoved.Contains(nodes[j])) continue;
            if (nodes[j] is LazyNode<float> candidate
                && candidate.OpType == LazyNodeType.Custom
                && candidate.OpName == "LayerNorm"
                && ReferenceEquals(candidate.Input0, addNode.Output))
            {
                lnNode = candidate;
                lnIndex = j;
                break;
            }
        }

        if (lnNode is null || lnNode.Input1 is null || lnNode.Input2 is null)
            return false;

        // Inspect optional auxiliary LayerNormMean / LayerNormVariance nodes recorded right after LayerNorm.
        // Only fuse when those auxiliary statistic nodes themselves have zero downstream consumers, so we can
        // drop them cleanly without forcing extra variance conversions.
        LazyNode<float>? meanAux = null;
        LazyNode<float>? varAux = null;
        int statEnd = Math.Min(nodes.Count - 1, lnIndex + 3);
        for (int j = lnIndex + 1; j <= statEnd; j++)
        {
            if (nodes[j] is LazyNode<float> statNode && ReferenceEquals(statNode.Input0, lnNode.Output))
            {
                // Retained (blocked, so already in the removed set) or read: the statistic needs the LayerNorm
                // writer this fusion would remove.
                if ((statNode.OpName is "LayerNormMean" or "LayerNormVariance") && alreadyRemoved.Contains(statNode))
                    return false;
                if (statNode.OpName == "LayerNormMean")
                {
                    if (consumerCounts.TryGetValue(statNode, out int mc) && mc > 0) return false;
                    meanAux = statNode;
                }
                else if (statNode.OpName == "LayerNormVariance")
                {
                    if (consumerCounts.TryGetValue(statNode, out int vc) && vc > 0) return false;
                    varAux = statNode;
                }
            }
        }

        var gamma = lnNode.Input1;
        var beta = lnNode.Input2;
        int cols = gamma.Length;
        if (cols <= 0 || beta.Length != cols || addNode.Output.Length == 0 || addNode.Output.Length % cols != 0)
            return false;

        int normDims = gamma.Rank;
        if (normDims > addNode.Output.Rank)
            return false;
        for (int d = 0; d < normDims; d++)
        {
            if (gamma._shape[d] != addNode.Output._shape[addNode.Output.Rank - normDims + d])
                return false;
        }

        if (!TryExtractLayerNormEpsilon(lnNode, out float epsilon))
            return false;

        int rows = addNode.Output.Length / cols;
        var capturedInput = addNode.Input0;
        var capturedResidual = addNode.Input1;
        var capturedGamma = gamma;
        var capturedBeta = beta;
        var meanHolder = new Tensor<float>(new[] { rows });
        var invStdHolder = new Tensor<float>(new[] { rows });

        addNode.Output.LazySource = null;
        if (meanAux is not null) meanAux.Output.LazySource = null;
        if (varAux is not null) varAux.Output.LazySource = null;

        var node = new LazyNode<float>(
            LazyNodeType.Custom,
            "FusedLayerNormResidual",
            new[] { capturedInput, capturedResidual, capturedGamma, capturedBeta },
            lnNode.Output,
            (eng, output) =>
            {
                if (eng.TryGet<DirectGpuTensorEngine>(out var gpu)
                    && gpu.TryExecuteFusedLayerNormResidualInto(
                        capturedInput, capturedResidual, capturedGamma, capturedBeta,
                        output, meanHolder, invStdHolder, rows, cols, epsilon))
                {
                    return;
                }

                if (!eng.Is<DirectGpuTensorEngine>()
                    && capturedInput.IsContiguous
                    && capturedResidual.IsContiguous
                    && capturedGamma.IsContiguous
                    && capturedBeta.IsContiguous
                    && output.IsContiguous)
                {
                    CompiledFusedKernels.LayerNormResidualForward(
                        capturedInput.AsSpan(),
                        capturedResidual.AsSpan(),
                        capturedGamma.AsSpan(),
                        capturedBeta.AsSpan(),
                        output.AsWritableSpan(),
                        meanHolder.AsWritableSpan(),
                        invStdHolder.AsWritableSpan(),
                        rows,
                        cols,
                        epsilon);
                    output.IncrementVersion();
                    meanHolder.IncrementVersion();
                    invStdHolder.IncrementVersion();
                    return;
                }

                var sum = eng.TensorAdd(capturedInput, capturedResidual);
                var r = eng.LayerNorm(sum, capturedGamma, capturedBeta, epsilon, out var freshMean, out var freshVar);
                DirectGpuTensorEngine.CopyResultInto(eng, r, output);
                var mSpan = freshMean.AsSpan();
                var vSpan = freshVar.AsSpan();
                var mDst = meanHolder.AsWritableSpan();
                var invDst = invStdHolder.AsWritableSpan();
                for (int i = 0; i < rows; i++)
                {
                    mDst[i] = mSpan[i];
                    invDst[i] = 1.0f / GeneratedKernelSimdMath.SqrtScalar(vSpan[i] + epsilon);
                }
                meanHolder.IncrementVersion();
                invStdHolder.IncrementVersion();
            },
            (gradOutput, inputs, _, _, engine, grads) =>
            {
                var x = inputs[0];
                var res = inputs[1];
                var g = inputs[2];
                var b = inputs[3];
                bool needX = DifferentiableOps.IsGradientRequired(x);
                bool needRes = DifferentiableOps.IsGradientRequired(res);
                bool needGamma = DifferentiableOps.IsGradientRequired(g);
                bool needBeta = DifferentiableOps.IsGradientRequired(b);
                if (!needX && !needRes && !needGamma && !needBeta)
                    return;

                if (!engine.Is<DirectGpuTensorEngine>()
                    && gradOutput.IsContiguous
                    && x.IsContiguous
                    && res.IsContiguous
                    && g.IsContiguous)
                {
                    var sharedGrad = TensorAllocator.Rent<float>(x._shape);
                    var gammaGrad = needGamma ? TensorAllocator.Rent<float>(g._shape) : null;
                    var betaGrad = needBeta ? TensorAllocator.Rent<float>(b._shape) : null;
                    CompiledFusedKernels.LayerNormResidualBackward(
                        gradOutput.AsSpan(),
                        x.AsSpan(),
                        res.AsSpan(),
                        g.AsSpan(),
                        meanHolder.AsSpan(),
                        invStdHolder.AsSpan(),
                        sharedGrad.AsWritableSpan(),
                        gammaGrad is not null ? gammaGrad.AsWritableSpan() : Span<float>.Empty,
                        betaGrad is not null ? betaGrad.AsWritableSpan() : Span<float>.Empty,
                        rows,
                        cols);
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
                    if (betaGrad is not null)
                    {
                        betaGrad.IncrementVersion();
                        DifferentiableOps.AccumulateGrad(grads, b, betaGrad, engine);
                    }
                    return;
                }

                // Device fallback computes true variance from invStdHolder for LayerNormBackward.
                var varTensor = new Tensor<float>(meanHolder._shape);
                var invSpan = invStdHolder.AsSpan();
                var varDst = varTensor.AsWritableSpan();
                for (int i = 0; i < rows; i++)
                {
                    float inv = invSpan[i];
                    varDst[i] = Math.Max(0.0f, (1.0f / (inv * inv)) - epsilon);
                }
                var sum = engine.TensorAdd(x, res);
                var gradSum = engine.LayerNormBackward(gradOutput, sum, g, meanHolder, varTensor, epsilon, out var dGamma, out var dBeta);
                if (needX)
                    DifferentiableOps.AccumulateGrad(grads, x, gradSum, engine);
                if (needRes)
                    DifferentiableOps.AccumulateGrad(grads, res, needX ? gradSum.Clone() : gradSum, engine);
                if (needGamma)
                    DifferentiableOps.AccumulateGrad(grads, g, dGamma, engine);
                if (needBeta)
                    DifferentiableOps.AccumulateGrad(grads, b, dBeta, engine);
            },
            new object[] { (double)epsilon })
        {
            RecordingEngine = lnNode.RecordingEngine
        };

        fused = node;
        PatternHelpers.SetLazySource(node);
        alreadyRemoved.Add(addNode);
        alreadyRemoved.Add(lnNode);
        if (meanAux is not null) alreadyRemoved.Add(meanAux);
        if (varAux is not null) alreadyRemoved.Add(varAux);
        return true;
    }

    private static bool TryExtractLayerNormEpsilon(LazyNode<float> lnNode, out float epsilon)
    {
        epsilon = 1e-5f;
        if (lnNode.SavedState is not { Length: >= 1 } state)
            return false;

        // Generic LayerNorm records { stateRef, stateRef, (double)epsilon }; inference records { (double)epsilon };
        // CpuEngine.LayerNormCompiled records { stats, LayerNormRowShape(rows, features, eps) }.
        for (int i = state.Length - 1; i >= 0; i--)
        {
            object? entry = state[i];
            if (entry is double d)
            {
                epsilon = (float)d;
                return epsilon >= 0.0f && !float.IsNaN(epsilon) && !float.IsInfinity(epsilon);
            }
            if (entry is float f)
            {
                epsilon = f;
                return epsilon >= 0.0f && !float.IsNaN(epsilon) && !float.IsInfinity(epsilon);
            }
            if (entry is not null)
            {
                var field = entry.GetType().GetField("Epsilon", BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic);
                if (field?.GetValue(entry) is float fieldFloat)
                {
                    epsilon = fieldFloat;
                    return epsilon >= 0.0f && !float.IsNaN(epsilon) && !float.IsInfinity(epsilon);
                }
                var prop = entry.GetType().GetProperty("Epsilon", BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic);
                if (prop?.GetValue(entry) is float propFloat)
                {
                    epsilon = propFloat;
                    return epsilon >= 0.0f && !float.IsNaN(epsilon) && !float.IsInfinity(epsilon);
                }
            }
        }

        return false;
    }
}
