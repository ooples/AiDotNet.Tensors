// Copyright (c) CrowdTrainer LLC. All rights reserved.

using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Compilation.GeneratedKernels;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Compilation;

/// <summary>
/// Matches decomposed <c>LogSoftmax(logits, -1) -&gt; Multiply(target, _) -&gt; ReduceSum -&gt; [ReduceMean | DivideScalar | MultiplyScalar] -&gt; Negate</c>
/// (in either reduction/negation order) and fuses the entire chain into a single <c>FusedLogSoftmaxCrossEntropy</c> node
/// with zero intermediate <c>[rows, vocab]</c> global-memory allocations in forward or backward.
/// </summary>
internal sealed class LogSoftmaxCrossEntropyFusionPattern : IFusionPattern
{
    public string Name => "LogSoftmaxCrossEntropy";

    public bool IsDifferentiable => true;

    public bool TryFuse(
        IReadOnlyList<ILazyNode> nodes,
        int index,
        IReadOnlyDictionary<ILazyNode, int> consumerCounts,
        HashSet<ILazyNode> alreadyRemoved,
        out ILazyNode? fused)
    {
        fused = null;
        if (nodes[index] is not LazyNode<float> logSmNode
            || logSmNode.OpType != LazyNodeType.Custom
            || logSmNode.OpName != "LogSoftmax"
            || !PatternHelpers.FanOutAtMostOne(logSmNode, consumerCounts))
        {
            return false;
        }

        var logits = logSmNode.Input0;
        if (logits.Rank < 2 || logits.Length == 0)
            return false;
        int classAxis = logits.Rank - 1;
        if (!TryExtractAxis(logSmNode, logits.Rank, out int recordedAxis) || recordedAxis != classAxis)
            return false;
        int cols = logits._shape[classAxis];
        if (cols <= 0 || logits.Length % cols != 0)
            return false;
        int rows = logits.Length / cols;

        // Next: Multiply(target, logSoftmax) or Multiply(logSoftmax, target)
        var mulNode = PatternHelpers.FindNextFloat(
            nodes,
            index + 1,
            index + 4,
            alreadyRemoved,
            n => n.OpType == LazyNodeType.Multiply
                && n.Input1 is not null
                && PatternHelpers.FanOutAtMostOne(n, consumerCounts)
                && PatternHelpers.ConsumesOutput(n, logSmNode));

        if (mulNode is null || !PatternHelpers.TryGetOtherInput(mulNode, logSmNode.Output, out var target))
            return false;
        if (!PatternHelpers.SameShape(logits, target))
            return false;

        // Next: ReduceSum over classAxis (or over all axes).
        var reduceSumNode = PatternHelpers.FindNextFloat(
            nodes,
            index + 2,
            index + 6,
            alreadyRemoved,
            n => n.OpType == LazyNodeType.ReduceSum && PatternHelpers.ConsumesOutput(n, mulNode));

        if (reduceSumNode is null || !TryGetReduceAxes(reduceSumNode, logits.Rank, out var sumAxes, out bool keepDims) || keepDims)
            return false;

        bool summedAllAxes = ReducesAllAxes(sumAxes, logits.Rank);
        bool summedClassAxisOnly = sumAxes.Length == 1 && sumAxes[0] == classAxis;
        if (!summedAllAxes && !summedClassAxisOnly)
            return false;
        if (!PatternHelpers.FanOutAtMostOne(reduceSumNode, consumerCounts))
            return false;

        var matchedIntermediates = new List<LazyNode<float>>(5) { logSmNode, mulNode, reduceSumNode };
        LazyNode<float>? terminalNode = null;
        float overallScale = 0.0f;

        if (summedClassAxisOnly)
        {
            int batchRank = logits.Rank - 1;
            // Expect ReduceMean(batchAxes) -> Negate OR Negate -> ReduceMean(batchAxes)
            var stepA = PatternHelpers.FindNextFloat(
                nodes,
                index + 3,
                index + 8,
                alreadyRemoved,
                n => PatternHelpers.ConsumesOutput(n, reduceSumNode));
            if (stepA is null)
                return false;

            if (stepA.OpType == LazyNodeType.ReduceMean
                && PatternHelpers.FanOutAtMostOne(stepA, consumerCounts)
                && IsFullBatchScalarReduceMean(stepA, batchRank, rows))
            {
                var negNode = PatternHelpers.FindNextFloat(
                    nodes,
                    index + 4,
                    index + 10,
                    alreadyRemoved,
                    n => n.OpType == LazyNodeType.Negate && PatternHelpers.ConsumesOutput(n, stepA));
                if (negNode is null)
                    return false;
                matchedIntermediates.Add(stepA);
                terminalNode = negNode;
                overallScale = 1.0f / rows;
            }
            else if (stepA.OpType == LazyNodeType.Negate && PatternHelpers.FanOutAtMostOne(stepA, consumerCounts))
            {
                var meanNode = PatternHelpers.FindNextFloat(
                    nodes,
                    index + 4,
                    index + 10,
                    alreadyRemoved,
                    n => n.OpType == LazyNodeType.ReduceMean && PatternHelpers.ConsumesOutput(n, stepA));
                if (meanNode is null || !IsFullBatchScalarReduceMean(meanNode, batchRank, rows))
                {
                    return false;
                }
                matchedIntermediates.Add(stepA);
                terminalNode = meanNode;
                overallScale = 1.0f / rows;
            }
            else
            {
                return false;
            }
        }
        else
        {
            // summedAllAxes: expect Negate -> DivideScalar/MultiplyScalar OR MultiplyScalar(-s) OR DivideScalar -> Negate
            var stepA = PatternHelpers.FindNextFloat(
                nodes,
                index + 3,
                index + 8,
                alreadyRemoved,
                n => PatternHelpers.ConsumesOutput(n, reduceSumNode));
            if (stepA is null)
                return false;

            if (stepA.OpType == LazyNodeType.MultiplyScalar && PatternHelpers.TryGetFloatScalar(stepA, out float mScalar) && mScalar < 0.0f)
            {
                terminalNode = stepA;
                overallScale = -mScalar;
            }
            else if (stepA.OpType == LazyNodeType.Negate)
            {
                if (!consumerCounts.TryGetValue(stepA, out int negConsumers) || negConsumers == 0)
                {
                    // Pure -sum(target * logSoftmax(logits)) with unit scale.
                    terminalNode = stepA;
                    overallScale = 1.0f;
                }
                else if (negConsumers == 1)
                {
                    var scaleNode = PatternHelpers.FindNextFloat(
                        nodes,
                        index + 4,
                        index + 10,
                        alreadyRemoved,
                        n => PatternHelpers.ConsumesOutput(n, stepA));
                    if (scaleNode is null)
                        return false;
                    if (scaleNode.OpType == LazyNodeType.DivideScalar && PatternHelpers.TryGetFloatScalar(scaleNode, out float denom) && denom > 0.0f)
                    {
                        matchedIntermediates.Add(stepA);
                        terminalNode = scaleNode;
                        overallScale = 1.0f / denom;
                    }
                    else if (scaleNode.OpType == LazyNodeType.MultiplyScalar && PatternHelpers.TryGetFloatScalar(scaleNode, out float mult) && mult > 0.0f)
                    {
                        matchedIntermediates.Add(stepA);
                        terminalNode = scaleNode;
                        overallScale = mult;
                    }
                    else
                    {
                        return false;
                    }
                }
                else
                {
                    return false;
                }
            }
            else if (stepA.OpType == LazyNodeType.DivideScalar
                && PatternHelpers.FanOutAtMostOne(stepA, consumerCounts)
                && PatternHelpers.TryGetFloatScalar(stepA, out float divVal)
                && divVal > 0.0f)
            {
                var negAfterDiv = PatternHelpers.FindNextFloat(
                    nodes,
                    index + 4,
                    index + 10,
                    alreadyRemoved,
                    n => n.OpType == LazyNodeType.Negate && PatternHelpers.ConsumesOutput(n, stepA));
                if (negAfterDiv is null)
                    return false;
                matchedIntermediates.Add(stepA);
                terminalNode = negAfterDiv;
                overallScale = 1.0f / divVal;
            }
            else
            {
                return false;
            }
        }

        if (terminalNode is null || overallScale <= 0.0f)
            return false;

        foreach (var inter in matchedIntermediates)
        {
            inter.Output.LazySource = null;
        }

        var rowMaxState = new float[rows];
        var rowLseState = new float[rows];
        // Set by the fused forward: the backward's fused path reads its row statistics, so it runs only after it.
        bool statsFromFusedForward = false;

        var node = new LazyNode<float>(
            LazyNodeType.Custom,
            "FusedLogSoftmaxCrossEntropy",
            new[] { logits, target },
            terminalNode.Output,
            (eng, output) =>
            {
                if (!eng.Is<DirectGpuTensorEngine>()
                    && logits.IsContiguous
                    && target.IsContiguous
                    && output.IsContiguous)
                {
                    float activeMean = CompiledFusedKernels.LogSoftmaxCrossEntropyForward(
                        logits.AsSpan(),
                        target.AsSpan(),
                        rowMaxState,
                        rowLseState,
                        rows,
                        cols,
                        out int active);
                    statsFromFusedForward = true;
                    // Recover exact total NLL = activeMean * max(1, active) and multiply by overallScale.
                    double totalNll = (double)activeMean * (active > 0 ? active : 1);
                    float finalLoss = (float)(totalNll * overallScale);
                    var dst = output.AsWritableSpan();
                    if (!dst.IsEmpty) dst[0] = finalLoss;
                    output.IncrementVersion();
                    return;
                }

                // Exactly the recorded loss on every other engine: overallScale * -sum(target * logSoftmax(logits)).
                // (No TensorSoftmaxCrossEntropy shortcut: it averages over supervised rows only, and its 1/rows test
                // also matched custom scales close to 1/rows.)
                statsFromFusedForward = false;
                var lp = eng.TensorLogSoftmax(logits, classAxis);
                var prod = eng.TensorMultiply(target, lp);
                var allAxes = new int[logits.Rank];
                for (int a = 0; a < allAxes.Length; a++) allAxes[a] = a;
                var sumAll = eng.ReduceSum(prod, allAxes, keepDims: false);
                var scaledLoss = eng.TensorMultiplyScalar(sumAll, -overallScale);
                DirectGpuTensorEngine.CopyResultInto(eng, scaledLoss, output);
            },
            (gradOutput, inputs, _, _, engine, grads) =>
            {
                var z = inputs[0];
                var t = inputs[1];
                bool needZ = DifferentiableOps.IsGradientRequired(z);
                bool needT = DifferentiableOps.IsGradientRequired(t);
                if (!needZ && !needT)
                    return;

                if (statsFromFusedForward
                    && !engine.Is<DirectGpuTensorEngine>()
                    && z.IsContiguous
                    && t.IsContiguous
                    && gradOutput.IsContiguous)
                {
                    var gSpan = gradOutput.AsSpan();
                    float upstream = gSpan.IsEmpty ? 1.0f : gSpan[0];
                    if (needZ)
                    {
                        var dZ = TensorAllocator.Rent<float>(z._shape);
                        // The final factor, applied once (no multiply by the active-row count and divide back).
                        CompiledFusedKernels.LogSoftmaxCrossEntropyBackward(
                            upstream * overallScale,
                            z.AsSpan(),
                            t.AsSpan(),
                            rowMaxState,
                            rowLseState,
                            dZ.AsWritableSpan(),
                            rows,
                            cols);
                        dZ.IncrementVersion();
                        DifferentiableOps.AccumulateGrad(grads, z, dZ, engine);
                    }
                    if (needT)
                    {
                        var dT = TensorAllocator.Rent<float>(t._shape);
                        var zSpan = z.AsSpan();
                        var dTSpan = dT.AsWritableSpan();
                        float tScale = -upstream * overallScale;
                        for (int r = 0; r < rows; r++)
                        {
                            float m = rowMaxState[r];
                            float lse = rowLseState[r];
                            int baseIdx = r * cols;
                            for (int c = 0; c < cols; c++)
                            {
                                // log-softmax in shifted coordinates, as the forward saved it.
                                dTSpan[baseIdx + c] = tScale * ((zSpan[baseIdx + c] - m) - lse);
                            }
                        }
                        dT.IncrementVersion();
                        DifferentiableOps.AccumulateGrad(grads, t, dT, engine);
                    }
                    return;
                }

                var lp = engine.TensorLogSoftmax(z, classAxis);
                var gScalar = gradOutput.AsSpan()[0] * (-overallScale);
                if (needT)
                {
                    DifferentiableOps.AccumulateGrad(grads, t, engine.TensorMultiplyScalar(lp, gScalar), engine);
                }
                if (needZ)
                {
                    var dLp = engine.TensorMultiplyScalar(t, gScalar);
                    var sm = engine.TensorExp(lp);
                    var rowSumGrad = engine.ReduceSum(dLp, new[] { classAxis }, keepDims: true);
                    var scaledSm = new Tensor<float>(sm._shape);
                    engine.TensorBroadcastMultiplyInto(scaledSm, sm, rowSumGrad);
                    var dZ = engine.TensorSubtract(dLp, scaledSm);
                    DifferentiableOps.AccumulateGrad(grads, z, dZ, engine);
                }
            },
            null)
        {
            RecordingEngine = terminalNode.RecordingEngine
        };

        fused = node;
        PatternHelpers.SetLazySource(node);
        foreach (var inter in matchedIntermediates)
        {
            alreadyRemoved.Add(inter);
        }
        alreadyRemoved.Add(terminalNode);
        return true;
    }

    private static bool TryExtractAxis(LazyNode<float> node, int rank, out int axis)
    {
        axis = -1;
        if (node.SavedState is not { Length: >= 1 } state || state[0] is not int rawAxis)
            return false;
        axis = rawAxis < 0 ? rawAxis + rank : rawAxis;
        return axis >= 0 && axis < rank;
    }

    private static bool TryGetReduceAxes(LazyNode<float> node, int rank, out int[] normalizedAxes, out bool keepDims)
    {
        normalizedAxes = Array.Empty<int>();
        keepDims = false;
        if (node.SavedState is not { Length: >= 1 } state || state[0] is not int[] rawAxes)
            return false;
        if (state.Length >= 2 && state[1] is bool kd)
            keepDims = kd;

        normalizedAxes = new int[rawAxes.Length];
        for (int i = 0; i < rawAxes.Length; i++)
        {
            int a = rawAxes[i] < 0 ? rawAxes[i] + rank : rawAxes[i];
            if (a < 0 || a >= rank) return false;
            normalizedAxes[i] = a;
        }
        return true;
    }

    private static bool IsFullBatchScalarReduceMean(LazyNode<float> meanNode, int batchRank, int expectedRows)
    {
        if (meanNode.Output.Length != 1 || meanNode.Output.Rank > 1)
            return false;
        if (batchRank == 1 && expectedRows == 1 && meanNode.SavedState is { Length: >= 2 } s && s[1] is true)
            return false;

        // CpuEngine.TryRecordReduceMeanFloat stores SavedState = [ int[] { outer, axisSize, inner } ]
        if (meanNode.SavedState is { Length: 1 } compiledState
            && compiledState[0] is int[] triple
            && triple.Length == 3
            && triple[0] == 1
            && triple[1] == expectedRows
            && triple[2] == 1)
        {
            return true;
        }

        return TryGetReduceAxes(meanNode, batchRank, out var axes, out bool keepDims)
            && !keepDims
            && ReducesAllAxes(axes, batchRank);
    }

    private static bool ReducesAllAxes(int[] axes, int rank)
    {
        if (axes.Length != rank) return false;
        var seen = new bool[rank];
        for (int i = 0; i < axes.Length; i++)
        {
            int a = axes[i];
            if (a < 0 || a >= rank || seen[a]) return false;
            seen[a] = true;
        }
        return true;
    }
}
