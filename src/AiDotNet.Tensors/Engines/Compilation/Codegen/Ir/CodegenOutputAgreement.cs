// Copyright (c) AiDotNet. All rights reserved.

using System;

namespace AiDotNet.Tensors.Engines.Compilation.Codegen.Ir;

/// <summary>Shared fail-closed numerical agreement gate for generated candidates.</summary>
internal static class CodegenOutputAgreement
{
    internal static bool Agrees(
        float[] candidate, float[] reference, double tolerance, out double deviation,
        out long worstIndex, out float actual, out float expected)
    {
        if (candidate is null) throw new ArgumentNullException(nameof(candidate));
        if (reference is null) throw new ArgumentNullException(nameof(reference));
        if (tolerance < 0) throw new ArgumentOutOfRangeException(nameof(tolerance));

        worstIndex = -1;
        actual = 0;
        expected = 0;
        if (candidate.Length != reference.Length)
        {
            deviation = double.PositiveInfinity;
            return false;
        }

        double worst = 0, scale = 0;
        for (long e = 0; e < candidate.LongLength; e++)
        {
            if (float.IsNaN(candidate[e]) || float.IsInfinity(candidate[e]) ||
                float.IsNaN(reference[e]) || float.IsInfinity(reference[e]))
            {
                deviation = double.PositiveInfinity;
                worstIndex = e;
                actual = candidate[e];
                expected = reference[e];
                return false;
            }
            double difference = Math.Abs(candidate[e] - reference[e]);
            if (difference > worst)
            {
                worst = difference;
                worstIndex = e;
                actual = candidate[e];
                expected = reference[e];
            }
            scale = Math.Max(scale, Math.Abs((double)reference[e]));
        }
        deviation = scale > 0 ? worst / scale : worst;
        return deviation <= tolerance;
    }
}
