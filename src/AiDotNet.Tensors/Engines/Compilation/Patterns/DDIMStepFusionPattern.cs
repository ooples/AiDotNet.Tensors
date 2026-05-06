// Copyright (c) AiDotNet. All rights reserved.
// Issue #301 — DDIM sampler-update chain detection for graph-mode auto-fusion.

using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Tensors.Engines.Compilation;

/// <summary>
/// Matches the DDIM sampler-update elementwise chain and emits a
/// single <see cref="LazyNodeType.FusedDDIMStep"/> node. The kernel
/// itself ships now (see
/// <see cref="AiDotNet.Tensors.Helpers.CpuFusedOperations.FusedDDIMStep"/>).
/// </summary>
internal sealed class DDIMStepFusionPattern : IFusionPattern
{
    public string Name => "DDIMStep";

    public bool TryFuse(
        IReadOnlyList<ILazyNode> nodes, int index,
        IReadOnlyDictionary<ILazyNode, int> consumerCounts,
        HashSet<ILazyNode> alreadyRemoved,
        out ILazyNode? fused)
    {
        fused = null;
        if (nodes[index] is not LazyNode<float> noiseScale || noiseScale.OpType != LazyNodeType.MultiplyScalar)
            return false;
        if (!PatternHelpers.FanOutAtMostOne(noiseScale, consumerCounts)
            || !PatternHelpers.TryGetFloatScalar(noiseScale, out var sqrtOneMinusAt))
            return false;

        var epsilon = noiseScale.Input0;

        var subtract = PatternHelpers.FindNextFloat(
            nodes, index + 1, index + 8, alreadyRemoved,
            n => n.OpType == LazyNodeType.Subtract
                && n.Input1 is not null
                && ReferenceEquals(n.Input1, noiseScale.Output)
                && PatternHelpers.FanOutAtMostOne(n, consumerCounts));
        if (subtract is null)
            return false;

        var xT = subtract.Input0;
        if (!PatternHelpers.SameShape(xT, epsilon))
            return false;

        var divide = PatternHelpers.FindNextFloat(
            nodes, index + 1, index + 10, alreadyRemoved,
            n => n.OpType == LazyNodeType.DivideScalar
                && PatternHelpers.ConsumesOutput(n, subtract)
                && PatternHelpers.FanOutAtMostOne(n, consumerCounts));
        if (divide is null || !PatternHelpers.TryGetFloatScalar(divide, out var sqrtAt))
            return false;

        var x0Scale = PatternHelpers.FindNextFloat(
            nodes, index + 1, index + 12, alreadyRemoved,
            n => n.OpType == LazyNodeType.MultiplyScalar
                && PatternHelpers.ConsumesOutput(n, divide)
                && PatternHelpers.FanOutAtMostOne(n, consumerCounts));
        if (x0Scale is null || !PatternHelpers.TryGetFloatScalar(x0Scale, out var sqrtAtMinus1))
            return false;

        var epsPrevScale = PatternHelpers.FindNextFloat(
            nodes, index + 1, index + 12, alreadyRemoved,
            n => !ReferenceEquals(n, noiseScale)
                && n.OpType == LazyNodeType.MultiplyScalar
                && ReferenceEquals(n.Input0, epsilon)
                && PatternHelpers.FanOutAtMostOne(n, consumerCounts));
        if (epsPrevScale is null || !PatternHelpers.TryGetFloatScalar(epsPrevScale, out var sqrtOneMinusAtMinus1))
            return false;

        var add = PatternHelpers.FindNextFloat(
            nodes, index + 1, index + 14, alreadyRemoved,
            n => PatternHelpers.IsAdd(n.OpType)
                && PatternHelpers.ConsumesOutput(n, x0Scale)
                && PatternHelpers.ConsumesOutput(n, epsPrevScale));
        if (add is null || !PatternHelpers.SameShape(add.Output, xT))
            return false;

        float alphaBarT = sqrtAt * sqrtAt;
        float alphaBarTMinus1 = sqrtAtMinus1 * sqrtAtMinus1;
        if (!IsValidSchedule(alphaBarT, alphaBarTMinus1, sqrtOneMinusAt, sqrtOneMinusAtMinus1))
            return false;

        subtract.Output.LazySource = null;
        divide.Output.LazySource = null;
        x0Scale.Output.LazySource = null;
        epsPrevScale.Output.LazySource = null;
        noiseScale.Output.LazySource = null;
        add.Output.LazySource = null;

        var capturedXt = xT;
        var capturedEps = epsilon;
        var capturedAlphaT = alphaBarT;
        var capturedAlphaTm1 = alphaBarTMinus1;
        var finalOutput = add.Output;

        fused = new LazyNode<float>(
            LazyNodeType.FusedDDIMStep,
            "FusedDDIMStep",
            new[] { capturedXt, capturedEps },
            finalOutput,
            (eng, output) =>
            {
                if (eng is DirectGpuTensorEngine gpu)
                {
                    var eager = gpu.FusedDDIMStep(capturedXt, capturedEps, capturedAlphaT, capturedAlphaTm1);
                    eager.AsSpan().CopyTo(output.AsWritableSpan());
                }
                else
                {
                    CpuFusedOperations.FusedDDIMStep(
                        capturedXt, capturedEps, capturedAlphaT, capturedAlphaTm1, output);
                }
            },
            backwardFn: null,
            savedState: new object[] { capturedAlphaT, capturedAlphaTm1 });

        PatternHelpers.SetLazySource(fused);
        alreadyRemoved.Add(noiseScale);
        alreadyRemoved.Add(subtract);
        alreadyRemoved.Add(divide);
        alreadyRemoved.Add(x0Scale);
        alreadyRemoved.Add(epsPrevScale);
        alreadyRemoved.Add(add);
        return true;
    }

    private static bool IsValidSchedule(
        float alphaBarT,
        float alphaBarTMinus1,
        float sqrtOneMinusAt,
        float sqrtOneMinusAtMinus1)
    {
        if (!(alphaBarT > 0f && alphaBarT <= 1f))
            return false;
        if (!(alphaBarTMinus1 >= 0f && alphaBarTMinus1 <= 1f))
            return false;

        const float Tolerance = 2e-3f;
        float atComplement = sqrtOneMinusAt * sqrtOneMinusAt;
        float atm1Complement = sqrtOneMinusAtMinus1 * sqrtOneMinusAtMinus1;
        return Math.Abs(alphaBarT + atComplement - 1f) <= Tolerance
            && Math.Abs(alphaBarTMinus1 + atm1Complement - 1f) <= Tolerance;
    }
}
