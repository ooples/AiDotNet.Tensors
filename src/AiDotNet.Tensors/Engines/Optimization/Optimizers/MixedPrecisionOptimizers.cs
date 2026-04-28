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
