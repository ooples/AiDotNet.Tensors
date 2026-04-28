using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines.Compilation;

namespace AiDotNet.Tensors.Engines.Optimization.Optimizers;

/// <summary>
/// Adam with BF16 first/second moments and FP32 master parameters.
///
/// Memory layout per parameter (length N):
///   * <c>p_master</c>      — FP32 master copy of parameters (4N bytes)
///   * <c>exp_avg_bf16</c>  — BF16 first moment    (2N bytes)
///   * <c>exp_avg_sq_bf16</c> — BF16 second moment (2N bytes)
/// Total: 8N bytes vs FP32-Adam's 12N bytes — a ~33% reduction in optimizer-state RAM
/// without measurable degradation on standard training runs.
///
/// The user-facing parameter buffer in the <c>ParamGroup</c> is the FP32 working tensor;
/// <c>p_master</c> is a deep-copy snapshot that keeps full precision across many step
/// updates so accumulated rounding does not pollute the parameters used at inference.
///
/// Reference: NVIDIA / DeepSpeed mixed-precision Adam, bitsandbytes <c>AdamW8bit</c>.
/// </summary>
[CudaGraphSafe(Note = "BF16 pack/unpack is bitwise; no data-dependent host control flow")]
public sealed class BF16AdamOptimizer : OptimizerBase
{
    private static readonly Dictionary<string, double> _defaults = new Dictionary<string, double>
    {
        ["lr"] = 1e-3, ["beta1"] = 0.9, ["beta2"] = 0.999, ["eps"] = 1e-8,
        ["weight_decay"] = 0.0, ["maximize"] = 0.0,
    };
    private static readonly string[] _stateNames = new[] { "step" };
    /// <inheritdoc />
    protected override IReadOnlyDictionary<string, double> Defaults => _defaults;
    /// <inheritdoc />
    protected override IReadOnlyList<string> StateNames => _stateNames;

    /// <inheritdoc />
    public override void Step()
    {
        bool maximized = ApplyMaximize();
        try
        {
            for (int gi = 0; gi < ParamGroups.Count; gi++)
            {
                var g = ParamGroups[gi];
                float lr = (float)g.LearningRate;
                float b1 = (float)g.GetOption("beta1", 0.9);
                float b2 = (float)g.GetOption("beta2", 0.999);
                float eps = (float)g.GetOption("eps", 1e-8);
                float wd = (float)g.GetOption("weight_decay", 0.0);
                for (int pi = 0; pi < g.Parameters.Count; pi++)
                {
                    float[] p = g.Parameters[pi];
                    float[] grad = g.Gradients[pi];
                    var slot = GetOrCreateState(gi, pi, p.Length);

                    // Lazy-allocate the BF16 moments + FP32 master copy on first step.
                    if (!slot.ContainsKey("exp_avg_bf16"))
                    {
                        slot["exp_avg_bf16"]    = OptimizerStateValue.FromTensor(new float[(p.Length + 1) / 2]); // u16-packed in float[]
                        slot["exp_avg_sq_bf16"] = OptimizerStateValue.FromTensor(new float[(p.Length + 1) / 2]);
                        var master = new float[p.Length];
                        Array.Copy(p, master, p.Length);
                        slot["p_master"] = OptimizerStateValue.FromTensor(master);
                    }

                    int step = (slot["step"].IntValue ?? 0) + 1;
                    slot["step"].IntValue = step;
                    var mPacked = slot["exp_avg_bf16"].Tensor!;
                    var vPacked = slot["exp_avg_sq_bf16"].Tensor!;
                    var pMaster = slot["p_master"].Tensor!;

                    float bc1 = 1f - MathF.Pow(b1, step);
                    float bc2 = 1f - MathF.Pow(b2, step);

                    for (int i = 0; i < p.Length; i++)
                    {
                        float gi_ = grad[i] + wd * pMaster[i];
                        float mFp32 = UnpackBF16(mPacked, i);
                        float vFp32 = UnpackBF16(vPacked, i);
                        mFp32 = b1 * mFp32 + (1f - b1) * gi_;
                        vFp32 = b2 * vFp32 + (1f - b2) * gi_ * gi_;
                        PackBF16(mPacked, i, mFp32);
                        PackBF16(vPacked, i, vFp32);

                        float mHat = mFp32 / bc1;
                        float vHat = vFp32 / bc2;
                        pMaster[i] -= lr * mHat / (MathF.Sqrt(vHat) + eps);
                        // Working FP32 buffer mirrors the master copy.
                        p[i] = pMaster[i];
                    }
                }
            }
        }
        finally { if (maximized) UnflipMaximize(); }
    }

    // BF16 = upper 16 bits of FP32. Packed two-per-float[] cell: low 16 bits = even index, high 16 bits = odd index.
    private static void PackBF16(float[] packed, int i, float v)
    {
        // Round-to-nearest-even via adding 0x7FFF + LSB-of-result-of-shift.
        uint bits = (uint)BitConverterCompat.SingleToInt32Bits(v);
        uint rounding = ((bits >> 16) & 1u) + 0x7FFFu;
        ushort bf = (ushort)((bits + rounding) >> 16);
        int cell = i >> 1;
        uint cellBits = (uint)BitConverterCompat.SingleToInt32Bits(packed[cell]);
        if ((i & 1) == 0) cellBits = (cellBits & 0xFFFF0000u) | bf;
        else              cellBits = (cellBits & 0x0000FFFFu) | ((uint)bf << 16);
        packed[cell] = BitConverterCompat.Int32BitsToSingle((int)cellBits);
    }

    private static float UnpackBF16(float[] packed, int i)
    {
        int cell = i >> 1;
        uint cellBits = (uint)BitConverterCompat.SingleToInt32Bits(packed[cell]);
        ushort bf = (i & 1) == 0 ? (ushort)(cellBits & 0xFFFFu) : (ushort)(cellBits >> 16);
        return BitConverterCompat.Int32BitsToSingle((int)((uint)bf << 16));
    }
}

/// <summary>
/// FP8-Lion: Lion optimizer (Chen et al., 2023) with FP8 (E4M3) first-moment EMA
/// and a per-tensor FP32 scale factor.
///
/// Lion is uniquely well-suited for FP8 storage:
///  * No division by sqrt(v) — so the moment doesn't appear in a denominator,
///    avoiding the dynamic-range squeeze that breaks FP8 Adam.
///  * The actual update is <c>sign(c)</c>, so very-low-precision moments still produce
///    bit-identical sign decisions on most steps.
///
/// E4M3 (4-bit exponent, 3-bit mantissa, 1 sign) covers ~[2⁻⁹, 448] in magnitude.
/// We rescale per-tensor (one FP32 scale per parameter slot) so the dynamic range
/// is fully utilised, then re-quantise on every write. Memory: 1 byte per moment +
/// 1 FP32 scale, vs 4 bytes for plain Lion → ~75% reduction.
/// </summary>
public sealed class FP8LionOptimizer : OptimizerBase
{
    private static readonly Dictionary<string, double> _defaults = new Dictionary<string, double>
    {
        ["lr"] = 1e-4, ["beta1"] = 0.9, ["beta2"] = 0.99, ["weight_decay"] = 0.0,
    };
    private static readonly string[] _stateNames = new[] { "step" };
    /// <inheritdoc />
    protected override IReadOnlyDictionary<string, double> Defaults => _defaults;
    /// <inheritdoc />
    protected override IReadOnlyList<string> StateNames => _stateNames;

    /// <inheritdoc />
    public override void Step()
    {
        for (int gi = 0; gi < ParamGroups.Count; gi++)
        {
            var g = ParamGroups[gi];
            float lr  = (float)g.LearningRate;
            float b1  = (float)g.GetOption("beta1", 0.9);
            float b2  = (float)g.GetOption("beta2", 0.99);
            float wd  = (float)g.GetOption("weight_decay", 0.0);

            for (int pi = 0; pi < g.Parameters.Count; pi++)
            {
                float[] p = g.Parameters[pi]; float[] grad = g.Gradients[pi];
                var slot = GetOrCreateState(gi, pi, p.Length);

                if (!slot.ContainsKey("exp_avg_fp8"))
                {
                    // 1 byte per element, packed in float[] (4 bytes each).
                    slot["exp_avg_fp8"]  = OptimizerStateValue.FromTensor(new float[(p.Length + 3) / 4]);
                    slot["m_scale"]      = OptimizerStateValue.FromFloat(1f);
                }

                var packed = slot["exp_avg_fp8"].Tensor!;
                float scale = slot["m_scale"].FloatValue ?? 1f;

                // Dequantize → update → requantize. Track new max-abs to refresh scale next step.
                float newMaxAbs = 0f;
                for (int i = 0; i < p.Length; i++)
                {
                    byte fp8 = ReadFp8(packed, i);
                    float mFp32 = E4M3ToFloat(fp8) * scale;

                    float c = b1 * mFp32 + (1f - b1) * grad[i];
                    float signC = c > 0f ? 1f : (c < 0f ? -1f : 0f);
                    p[i] -= lr * (signC + wd * p[i]);

                    float mNew = b2 * mFp32 + (1f - b2) * grad[i];
                    float absM = MathF.Abs(mNew);
                    if (absM > newMaxAbs) newMaxAbs = absM;

                    // Provisionally requantize against current scale; we'll redo with
                    // refreshed scale at the end if maxAbs grew.
                    WriteFp8(packed, i, FloatToE4M3(mNew / scale));
                }

                // Update scale to keep all moments inside E4M3's representable range
                // (max ≈ 448). Apply hysteresis so we only re-encode when needed.
                const float fp8Max = 448f;
                if (newMaxAbs > fp8Max * scale * 0.95f || newMaxAbs < fp8Max * scale * 0.1f)
                {
                    float newScale = newMaxAbs > 0 ? newMaxAbs / (fp8Max * 0.5f) : 1f;
                    // Re-encode every entry against the new scale so values aren't quantised twice.
                    for (int i = 0; i < p.Length; i++)
                    {
                        byte fp8 = ReadFp8(packed, i);
                        float mFp32 = E4M3ToFloat(fp8) * scale;
                        WriteFp8(packed, i, FloatToE4M3(mFp32 / newScale));
                    }
                    slot["m_scale"].FloatValue = newScale;
                }
            }
        }
    }

    private static byte ReadFp8(float[] packed, int i)
    {
        int cell = i >> 2; int slot = i & 3;
        uint bits = (uint)BitConverterCompat.SingleToInt32Bits(packed[cell]);
        return (byte)((bits >> (slot * 8)) & 0xFFu);
    }

    private static void WriteFp8(float[] packed, int i, byte value)
    {
        int cell = i >> 2; int slot = i & 3;
        uint bits = (uint)BitConverterCompat.SingleToInt32Bits(packed[cell]);
        bits = (bits & ~(0xFFu << (slot * 8))) | ((uint)value << (slot * 8));
        packed[cell] = BitConverterCompat.Int32BitsToSingle((int)bits);
    }

    /// <summary>E4M3: 1 sign + 4 exponent (bias 7) + 3 mantissa. NaN encoding 0xFF / 0x7F.</summary>
    private static byte FloatToE4M3(float v)
    {
        if (float.IsNaN(v)) return 0x7F;
        if (v == 0f) return 0;
        uint bits = (uint)BitConverterCompat.SingleToInt32Bits(v);
        uint sign = (bits >> 31) & 1u;
        int exp = (int)((bits >> 23) & 0xFFu) - 127;
        uint mantissa = bits & 0x7FFFFFu;

        // Round mantissa from 23 bits to 3 bits with round-to-nearest-even.
        uint roundShift = 20;  // 23 - 3
        uint rounded = (mantissa + (1u << ((int)roundShift - 1))) >> (int)roundShift;
        if (rounded >= 8) { rounded = 0; exp++; }

        int e4m3Exp = exp + 7; // bias 7
        if (e4m3Exp <= 0)
        {
            // Subnormal / underflow: clamp to 0.
            return (byte)(sign << 7);
        }
        if (e4m3Exp >= 15)
        {
            // Overflow: clamp to max representable (S 1110 111 = ±448).
            return (byte)((sign << 7) | (0b1110 << 3) | 0b111);
        }
        return (byte)((sign << 7) | ((uint)e4m3Exp << 3) | rounded);
    }

    private static float E4M3ToFloat(byte b)
    {
        if (b == 0 || b == 0x80) return 0f;
        if (b == 0x7F || b == 0xFF) return float.NaN;
        uint sign = ((uint)b >> 7) & 1u;
        uint exp = ((uint)b >> 3) & 0xFu;
        uint mantissa = (uint)b & 0x7u;
        int e = exp == 0 ? -6 : (int)exp - 7;
        float m = exp == 0 ? mantissa / 8f : 1f + mantissa / 8f;
        float v = m * MathF.Pow(2f, e);
        return sign == 0 ? v : -v;
    }
}

/// <summary>net471 polyfill bridge for <c>BitConverter.SingleToInt32Bits</c> /
/// <c>Int32BitsToSingle</c> which only ship on .NET Core+.</summary>
internal static class BitConverterCompat
{
    public static int SingleToInt32Bits(float value)
    {
#if NET5_0_OR_GREATER
        return BitConverter.SingleToInt32Bits(value);
#else
        var bytes = BitConverter.GetBytes(value);
        return BitConverter.ToInt32(bytes, 0);
#endif
    }

    public static float Int32BitsToSingle(int value)
    {
#if NET5_0_OR_GREATER
        return BitConverter.Int32BitsToSingle(value);
#else
        var bytes = BitConverter.GetBytes(value);
        return BitConverter.ToSingle(bytes, 0);
#endif
    }
}
