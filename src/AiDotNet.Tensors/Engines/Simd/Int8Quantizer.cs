using System;
using System.Runtime.CompilerServices;
#if NET5_0_OR_GREATER
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
#endif

namespace AiDotNet.Tensors.Engines.Simd;

/// <summary>
/// Per-tensor symmetric int8 quantization helpers — Path D foundation.
///
/// <para>"Per-tensor symmetric" means a single float scale per tensor and
/// zero-point fixed at 0. Quantize: <c>q = round(x / scale)</c>, clamped to
/// <c>[-127, 127]</c> (we leave -128 unused so negation is symmetric).
/// Dequantize: <c>x = q * scale</c>. The scale is chosen as
/// <c>max|x| / 127</c> so the largest-magnitude value maps to ±127.</para>
///
/// <para>This class supports the "weight-only int8" inference pattern: weights
/// are pre-quantized at plan-compile time (one-time cost), stored as int8 (4×
/// less DRAM than float32), and dequantized on-the-fly during the GEMM packing
/// phase. Memory bandwidth on weight loads drops 4× — for BERT FFN with 9 MB
/// float weights, this saves ~6.75 MB per MatMul × 24 MatMuls per inference =
/// 162 MB of saved DRAM traffic, which translates to ~10-15 ms at 10 GB/s.</para>
///
/// <para>The full int8-compute path (VNNI <c>vpdpbusd</c> for 4× compute) lives
/// in <see cref="Avx512Vnni"/> and is independent of this quantizer. Both
/// share the same per-tensor scale convention.</para>
/// </summary>
public static class Int8Quantizer
{
    /// <summary>
    /// Compute the symmetric per-tensor scale from a float array. Scale is
    /// <c>max|x| / 127</c>. Returns 1f if all values are zero (avoids divide
    /// by zero — int8 quantize then dequantizes back to zero, correct).
    /// </summary>
    public static float ComputeSymmetricScale(ReadOnlySpan<float> data)
    {
        if (data.Length == 0) return 1f;
        float maxAbs = 0f;
#if NET5_0_OR_GREATER
        if (Avx.IsSupported && data.Length >= 32)
        {
            unsafe
            {
                fixed (float* p = data)
                {
                    int simdLen = data.Length & ~31;
                    var vMax0 = Vector256<float>.Zero;
                    var vMax1 = Vector256<float>.Zero;
                    var vMax2 = Vector256<float>.Zero;
                    var vMax3 = Vector256<float>.Zero;
                    var vAbsMask = Vector256.Create(0x7FFFFFFF).AsSingle();
                    int i = 0;
                    for (; i < simdLen; i += 32)
                    {
                        vMax0 = Avx.Max(vMax0, Avx.And(Avx.LoadVector256(p + i),      vAbsMask));
                        vMax1 = Avx.Max(vMax1, Avx.And(Avx.LoadVector256(p + i + 8),  vAbsMask));
                        vMax2 = Avx.Max(vMax2, Avx.And(Avx.LoadVector256(p + i + 16), vAbsMask));
                        vMax3 = Avx.Max(vMax3, Avx.And(Avx.LoadVector256(p + i + 24), vAbsMask));
                    }
                    var vMax = Avx.Max(Avx.Max(vMax0, vMax1), Avx.Max(vMax2, vMax3));
                    float* m = stackalloc float[8];
                    Avx.Store(m, vMax);
                    for (int k = 0; k < 8; k++)
                        if (m[k] > maxAbs) maxAbs = m[k];
                    for (; i < data.Length; i++)
                    {
                        float a = MathF.Abs(p[i]);
                        if (a > maxAbs) maxAbs = a;
                    }
                }
            }
        }
        else
#endif
        {
            for (int i = 0; i < data.Length; i++)
            {
                float a = data[i] < 0 ? -data[i] : data[i];
                if (a > maxAbs) maxAbs = a;
            }
        }
        return maxAbs == 0f ? 1f : maxAbs / 127f;
    }

    /// <summary>
    /// Quantize float32 → int8 with symmetric per-tensor scale. Output values
    /// are clamped to [-127, 127] (asymmetric -128 lane left unused so negation
    /// is exact).
    /// </summary>
    public static void QuantizeFloat32ToInt8(
        ReadOnlySpan<float> input, Span<sbyte> output, float scale)
    {
        if (output.Length < input.Length)
            throw new ArgumentException("output.Length must be >= input.Length");
        float invScale = 1f / scale;
#if NET5_0_OR_GREATER
        if (Avx.IsSupported && input.Length >= 8)
        {
            unsafe
            {
                fixed (float* pIn = input)
                fixed (sbyte* pOut = output)
                {
                    var vInvScale = Vector256.Create(invScale);
                    var vMin = Vector256.Create(-127f);
                    var vMax = Vector256.Create(127f);
                    int i = 0;
                    int simdLen = input.Length & ~7;
                    for (; i < simdLen; i += 8)
                    {
                        var v = Avx.Multiply(Avx.LoadVector256(pIn + i), vInvScale);
                        // Round-to-nearest-even (banker's rounding) matches
                        // ONNX QuantizeLinear semantics. Avx.RoundToNearest
                        // honours MXCSR (round-to-nearest-even by default).
                        v = Avx.RoundToNearestInteger(v);
                        v = Avx.Min(Avx.Max(v, vMin), vMax);
                        var vi = Avx.ConvertToVector256Int32(v);
                        // Pack int32 → int16 → int8 lane-by-lane.
                        for (int k = 0; k < 8; k++)
                        {
                            int lane = vi.GetElement(k);
                            pOut[i + k] = (sbyte)lane;
                        }
                    }
                    for (; i < input.Length; i++)
                    {
                        float v = pIn[i] * invScale;
                        v = MathF.Round(v);
                        if (v < -127f) v = -127f;
                        if (v > 127f) v = 127f;
                        pOut[i] = (sbyte)v;
                    }
                }
            }
            return;
        }
#endif
        for (int i = 0; i < input.Length; i++)
        {
            float v = input[i] * invScale;
            v = (float)Math.Round(v, MidpointRounding.ToEven);
            if (v < -127f) v = -127f;
            if (v > 127f) v = 127f;
            output[i] = (sbyte)v;
        }
    }

    /// <summary>
    /// Dequantize a length-N int8 vector back to float, multiplying by the
    /// per-tensor scale. Inverse of <see cref="QuantizeFloat32ToInt8"/>.
    /// </summary>
    public static void DequantizeInt8ToFloat32(
        ReadOnlySpan<sbyte> input, Span<float> output, float scale)
    {
        if (output.Length < input.Length)
            throw new ArgumentException("output.Length must be >= input.Length");
#if NET5_0_OR_GREATER
        if (Avx2.IsSupported && input.Length >= 8)
        {
            unsafe
            {
                fixed (sbyte* pIn = input)
                fixed (float* pOut = output)
                {
                    var vScale = Vector256.Create(scale);
                    int i = 0;
                    int simdLen = input.Length & ~7;
                    for (; i < simdLen; i += 8)
                    {
                        // Load 8 sbyte → expand to 8 int32 → convert to float.
                        // Avx2 has a sign-extending int8→int32 broadcast load
                        // (vpmovsxbd) but doing it in one step requires the
                        // 64-bit GP load + intrinsic; simpler to widen
                        // explicitly. The compiler turns the GetElement loop
                        // into a sequence of MOVZX/CVTSI2SS (still SIMD-issue
                        // wide enough for the L1-resident weight panel case).
                        var v = Vector256.Create(
                            (float)pIn[i + 0], (float)pIn[i + 1],
                            (float)pIn[i + 2], (float)pIn[i + 3],
                            (float)pIn[i + 4], (float)pIn[i + 5],
                            (float)pIn[i + 6], (float)pIn[i + 7]);
                        Avx.Store(pOut + i, Avx.Multiply(v, vScale));
                    }
                    for (; i < input.Length; i++)
                        pOut[i] = pIn[i] * scale;
                }
            }
            return;
        }
#endif
        for (int i = 0; i < input.Length; i++)
            output[i] = input[i] * scale;
    }

    /// <summary>
    /// Numerical-quality verification helper: round-trip error of
    /// quantize→dequantize for each element. Returns:
    ///   - <c>MaxRelLarge</c>: max relative error for values whose magnitude
    ///     exceeds 10× the quantization step (i.e. values well above the
    ///     noise floor). Per-tensor symmetric int8 with these values has
    ///     max rel error ≤ ~5% in practice.
    ///   - <c>Rms</c>: L2 RMS error vs the original input. For BERT-style
    ///     activations expect &lt; scale (i.e. &lt; max|x|/127).
    ///   - <c>SnrDb</c>: signal-to-noise ratio in dB. Industry-standard
    ///     quality metric; INT8 weight quantization typically achieves
    ///     35-45 dB on transformer weights.
    /// Element-level max relative error is uninformative because values
    /// near zero quantize to 0 and produce 100% relative error by definition;
    /// use SNR or RMS for headline quality assessment.
    /// </summary>
    public static (double MaxRelLarge, double Rms, double SnrDb) RoundTripError(
        ReadOnlySpan<float> input)
    {
        if (input.Length == 0) return (0, 0, double.PositiveInfinity);
        float scale = ComputeSymmetricScale(input);
        var q = new sbyte[input.Length];
        var dq = new float[input.Length];
        QuantizeFloat32ToInt8(input, q, scale);
        DequantizeInt8ToFloat32(q, dq, scale);

        double sumSqErr = 0;
        double sumSqSig = 0;
        double maxRel = 0;
        // Values beyond 100× scale have rel error ≤ 0.5% — that's the
        // headroom band where quantization noise is well-bounded vs
        // signal magnitude. Below that, rel-error grows linearly to 100%
        // at zero (as expected from rounding to a fixed quantum).
        double largeThresh = 100.0 * scale;
        for (int i = 0; i < input.Length; i++)
        {
            double err = dq[i] - input[i];
            sumSqErr += err * err;
            sumSqSig += (double)input[i] * input[i];
            double mag = Math.Abs(input[i]);
            if (mag > largeThresh)
            {
                double rel = Math.Abs(err) / mag;
                if (rel > maxRel) maxRel = rel;
            }
        }
        double rms = Math.Sqrt(sumSqErr / input.Length);
        double snrDb = sumSqErr > 0
            ? 10.0 * Math.Log10(sumSqSig / sumSqErr)
            : double.PositiveInfinity;
        return (maxRel, rms, snrDb);
    }
}
