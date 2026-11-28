// Copyright (c) CrowdTrainer LLC. All rights reserved.

using System;
using System.Collections.Generic;
using System.Globalization;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Compilation.GeneratedKernels;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Compilation;

/// <summary>
/// Matches <c>Add(input, residual) -&gt; RMSNorm(sum, gamma, epsilon, out rms)</c> and emits a single
/// <c>FusedRmsNormResidual</c> node that executes without materializing the intermediate residual-sum tensor
/// in global memory (using compile-time generated AVX2/FMA SIMD on CPU and warp-shuffle NVRTC CUDA on GPU).
/// </summary>
internal sealed class RmsNormResidualFusionPattern : IFusionPattern
{
    public string Name => "RmsNormResidual";

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

        var rmsNode = PatternHelpers.FindNextFloat(
            nodes,
            index + 1,
            index + 6,
            alreadyRemoved,
            n => n.OpType == LazyNodeType.Custom
                && n.OpName == "RMSNorm"
                && ReferenceEquals(n.Input0, addNode.Output));

        if (rmsNode is null || rmsNode.Input1 is null)
            return false;

        var gamma = rmsNode.Input1;
        int cols = gamma.Length;
        if (cols <= 0 || addNode.Output.Length == 0 || addNode.Output.Length % cols != 0)
            return false;

        int normDims = gamma.Rank;
        if (normDims > addNode.Output.Rank)
            return false;
        for (int d = 0; d < normDims; d++)
        {
            if (gamma._shape[d] != addNode.Output._shape[addNode.Output.Rank - normDims + d])
                return false;
        }

        if (!TryExtractRmsState(rmsNode, out Tensor<float>? savedRmsTensor, out float epsilon))
            return false;

        int rows = addNode.Output.Length / cols;
        var capturedInput = addNode.Input0;
        var capturedResidual = addNode.Input1;
        var capturedGamma = gamma;
        var rmsHolder = savedRmsTensor ?? new Tensor<float>(new[] { rows });

        addNode.Output.LazySource = null;

        var node = new LazyNode<float>(
            LazyNodeType.Custom,
            "FusedRmsNormResidual",
            new[] { capturedInput, capturedResidual, capturedGamma },
            rmsNode.Output,
            (eng, output) =>
            {
                if (eng.TryGet<DirectGpuTensorEngine>(out var gpu)
                    && gpu.TryExecuteFusedRmsNormResidualInto(
                        capturedInput, capturedResidual, capturedGamma, output, rmsHolder, rows, cols, epsilon))
                {
                    return;
                }

                if (!eng.Is<DirectGpuTensorEngine>()
                    && capturedInput.IsContiguous
                    && capturedResidual.IsContiguous
                    && capturedGamma.IsContiguous
                    && output.IsContiguous
                    && rmsHolder.IsContiguous)
                {
                    CompiledFusedKernels.RmsNormResidualForward(
                        capturedInput.AsSpan(),
                        capturedResidual.AsSpan(),
                        capturedGamma.AsSpan(),
                        output.AsWritableSpan(),
                        Span<float>.Empty,
                        rmsHolder.AsWritableSpan(),
                        rows,
                        cols,
                        epsilon);
                    output.IncrementVersion();
                    rmsHolder.IncrementVersion();
                    return;
                }

                var sum = eng.TensorAdd(capturedInput, capturedResidual);
                var r = eng.RMSNorm(sum, capturedGamma, epsilon, out var freshRms);
                DirectGpuTensorEngine.CopyResultInto(eng, r, output);
                DirectGpuTensorEngine.CopyResultInto(eng, freshRms, rmsHolder);
            },
            (gradOutput, inputs, _, _, engine, grads) =>
            {
                var x = inputs[0];
                var res = inputs[1];
                var g = inputs[2];
                bool needX = DifferentiableOps.IsGradientRequired(x);
                bool needRes = DifferentiableOps.IsGradientRequired(res);
                bool needGamma = DifferentiableOps.IsGradientRequired(g);
                if (!needX && !needRes && !needGamma)
                    return;

                if (!engine.Is<DirectGpuTensorEngine>()
                    && gradOutput.IsContiguous
                    && x.IsContiguous
                    && res.IsContiguous
                    && g.IsContiguous
                    && rmsHolder.IsContiguous)
                {
                    var sharedGrad = TensorAllocator.Rent<float>(x._shape);
                    var gammaGrad = needGamma ? TensorAllocator.Rent<float>(g._shape) : null;
                    CompiledFusedKernels.RmsNormResidualBackward(
                        gradOutput.AsSpan(),
                        x.AsSpan(),
                        res.AsSpan(),
                        g.AsSpan(),
                        rmsHolder.AsSpan(),
                        sharedGrad.AsWritableSpan(),
                        gammaGrad is not null ? gammaGrad.AsWritableSpan() : Span<float>.Empty,
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
                    return;
                }

                var sum = engine.TensorAdd(x, res);
                var gradSum = engine.RMSNormBackward(gradOutput, sum, g, rmsHolder, epsilon, out var dGamma);
                if (needX)
                    DifferentiableOps.AccumulateGrad(grads, x, gradSum, engine);
                if (needRes)
                    DifferentiableOps.AccumulateGrad(grads, res, needX ? gradSum.Clone() : gradSum, engine);
                if (needGamma)
                    DifferentiableOps.AccumulateGrad(grads, g, dGamma, engine);
            },
            new object[] { rmsHolder, (double)epsilon })
        {
            RecordingEngine = rmsNode.RecordingEngine
        };

        fused = node;
        PatternHelpers.SetLazySource(node);
        alreadyRemoved.Add(addNode);
        alreadyRemoved.Add(rmsNode);
        return true;
    }

    private static bool TryExtractRmsState(LazyNode<float> rmsNode, out Tensor<float>? rmsTensor, out float epsilon)
    {
        rmsTensor = null;
        epsilon = 1e-5f;
        if (rmsNode.SavedState is not { Length: >= 2 } state)
            return false;
        rmsTensor = state[0] as Tensor<float>;
        if (state[1] is null)
            return false;
        try
        {
            epsilon = Convert.ToSingle(state[1], CultureInfo.InvariantCulture);
            return epsilon >= 0.0f && !float.IsNaN(epsilon) && !float.IsInfinity(epsilon);
        }
        catch
        {
            return false;
        }
    }
}
