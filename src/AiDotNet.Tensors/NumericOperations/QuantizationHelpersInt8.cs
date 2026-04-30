// Copyright (c) AiDotNet. All rights reserved.

using System;

namespace AiDotNet.Tensors.NumericOperations;

/// <summary>
/// int8 symmetric per-group quantization. Companion to the int4 / int2 /
/// int3 / NF4 / FP4 paths in <see cref="QuantizationHelpers"/>; matches
/// the GGUF Q8_0 layout (block of 32 with one float scale per block).
/// </summary>
public static class QuantizationHelpersInt8
{
    /// <summary>Quantize a float span to int8 with per-group symmetric
    /// scales. Default group size of 32 matches llama.cpp Q8_0.</summary>
    public static QuantizationScale QuantizeInt8(
        ReadOnlySpan<float> src, Span<sbyte> dst, int groupSize = 32)
    {
        if (dst.Length != src.Length)
            throw new ArgumentException($"dst.Length {dst.Length} must equal src.Length {src.Length} for int8.");
        if (groupSize <= 0)
            throw new ArgumentOutOfRangeException(nameof(groupSize), "groupSize must be > 0.");
        int n = src.Length;
        int groups = (n + groupSize - 1) / groupSize;
        var scales = new float[groups];

        for (int g = 0; g < groups; g++)
        {
            int start = g * groupSize;
            int end = Math.Min(start + groupSize, n);
            float maxAbs = 0f;
            for (int i = start; i < end; i++)
            {
                float a = Math.Abs(src[i]);
                if (a > maxAbs) maxAbs = a;
            }
            // sbyte range is [-128, 127]; clamp denominator to 127 so
            // the maxAbs element maps to ±127.
            float scale = maxAbs / 127f;
            if (scale == 0f) scale = 1f; // avoid div-by-zero on all-zero group
            scales[g] = scale;
            float invScale = 1f / scale;
            for (int i = start; i < end; i++)
            {
                // GGUF Q8_0 / llama.cpp use roundf (round-half-AWAY-from-zero),
                // not Math.Round's default banker's-rounding. Mismatch here
                // would produce off-by-one quant indices vs the reference
                // and break GGUF interop on edge values like 0.5.
                int q = (int)Math.Round(src[i] * invScale, MidpointRounding.AwayFromZero);
                if (q < -127) q = -127;
                if (q > 127) q = 127;
                dst[i] = (sbyte)q;
            }
        }
        return new QuantizationScale(scales, groupSize);
    }

    /// <summary>Inverse of <see cref="QuantizeInt8"/>.</summary>
    public static void DequantizeInt8(
        ReadOnlySpan<sbyte> src, QuantizationScale scale, Span<float> dst)
    {
        if (scale is null) throw new ArgumentNullException(nameof(scale));
        // Symmetric-only contract — asymmetric requires a separate kernel.
        if (scale.ZeroPoints.Length != 0)
            throw new NotSupportedException(
                "DequantizeInt8 supports symmetric quantization only (ZeroPoints must be empty).");
        if (dst.Length != src.Length)
            throw new ArgumentException($"dst.Length {dst.Length} must equal src.Length {src.Length}.");
        // Empty-input guard: with src.Length == 0 and scale.GroupSize <= 0,
        // the group derivation below would divide by zero.
        if (src.Length == 0)
        {
            if (scale.Scales.Length > 1)
                throw new ArgumentException(
                    $"scale.Scales.Length {scale.Scales.Length} must be 0 or 1 when src is empty.",
                    nameof(scale));
            return;
        }
        int groupSize = scale.GroupSize <= 0 ? src.Length : scale.GroupSize;
        int groups = (src.Length + groupSize - 1) / groupSize;
        if (scale.Scales.Length != groups)
            throw new ArgumentException(
                $"scale.Scales.Length {scale.Scales.Length} must match group count {groups} (n={src.Length}, group={groupSize}).");

        for (int g = 0; g < groups; g++)
        {
            int start = g * groupSize;
            int end = Math.Min(start + groupSize, src.Length);
            float s = scale.Scales[g];
            for (int i = start; i < end; i++) dst[i] = src[i] * s;
        }
    }
}
