// Copyright (c) CrowdTrainer LLC. All rights reserved.

using System;
using AiDotNet.Tensors.Engines.DirectGpu;

namespace AiDotNet.Tensors.Engines.Gpu;

/// <summary>
/// The host twin of <see cref="GpuOptimizer.TryAdamBf16Step"/> and the int8
/// <see cref="GpuOptimizer.TryAdam8BitStep(LinearAlgebra.Tensor{float}, LinearAlgebra.Tensor{float}, IGpuBuffer?, IGpuBuffer?, IGpuBuffer?, IGpuBuffer?, float, float, float, float, float, float, int, float, bool)"/>:
/// the same update math, on state laid out exactly as the device buffers are, so an optimizer keeps one compressed
/// state whether a step runs on the device or the host, and moves it between them with a plain upload/download.
/// </summary>
/// <remarks>
/// <para>Layouts (identical to the device buffers):</para>
/// <list type="bullet">
/// <item>bf16 moments: <c>2 * length</c> bytes each, little-endian bfloat16 per element, round-to-nearest-even.</item>
/// <item>int8 moments: <c>length</c> bytes each (indices into the signed / unsigned dynamic codebooks) plus one float
/// scale (the block absmax) per <c>blockSize</c> block for each of m and v.</item>
/// </list>
/// <para>Weight decay: <c>decoupledWeightDecay</c> false adds <c>wd * param</c> to the gradient (Adam's L2), true scales
/// the parameter by <c>1 - lr * wd</c> before the step (AdamW).</para>
/// </remarks>
public static class HostCompressedMomentAdam
{
    /// <summary>One Adam/AdamW step with bfloat16 moments. <paramref name="step"/> is 1-based; step 1 ignores the stored moments.</summary>
    public static void Bf16Step(float[] param, float[] grad, byte[] m, byte[] v, int length,
        float lr, float beta1, float beta2, float epsilon, float weightDecay, int step, bool decoupledWeightDecay)
    {
        Validate(param, grad, length);
        if (m is null) throw new ArgumentNullException(nameof(m));
        if (v is null) throw new ArgumentNullException(nameof(v));
        if (m.Length < 2L * length || v.Length < 2L * length)
            throw new ArgumentException("bf16 moments need 2 bytes per element.");
        if (step < 1) throw new ArgumentOutOfRangeException(nameof(step), step, "The Adam step is 1-based.");
        CompressedMomentHyperparameters.Validate(lr, beta1, beta2, epsilon, weightDecay);
        CompressedMomentHostFallback.AdamBf16(param, grad, m, v, lr, beta1, beta2, epsilon, weightDecay, step, length,
            decoupledWeightDecay);
    }

    /// <summary>One Adam/AdamW step with int8 block-quantized moments. <paramref name="step"/> is 1-based.</summary>
    public static void Int8Step(float[] param, float[] grad, byte[] mQuant, byte[] vQuant, float[] mScales, float[] vScales,
        int length, int blockSize, float lr, float beta1, float beta2, float epsilon, float weightDecay, int step,
        bool decoupledWeightDecay)
    {
        Validate(param, grad, length);
        if (mQuant is null) throw new ArgumentNullException(nameof(mQuant));
        if (vQuant is null) throw new ArgumentNullException(nameof(vQuant));
        if (mScales is null) throw new ArgumentNullException(nameof(mScales));
        if (vScales is null) throw new ArgumentNullException(nameof(vScales));
        if (blockSize <= 0) throw new ArgumentOutOfRangeException(nameof(blockSize));
        if (step < 1) throw new ArgumentOutOfRangeException(nameof(step), step, "The Adam step is 1-based.");
        CompressedMomentHyperparameters.Validate(lr, beta1, beta2, epsilon, weightDecay);
        int blocks = BlockCount(length, blockSize);
        if (mQuant.Length < length || vQuant.Length < length) throw new ArgumentException("int8 moments need 1 byte per element.");
        if (mScales.Length < blocks || vScales.Length < blocks) throw new ArgumentException("int8 moments need one scale per block.");
        float beta1Pow = MathF.Pow(beta1, step), beta2Pow = MathF.Pow(beta2, step);
        CompressedMomentHostFallback.Adam8Bit(param, grad, mQuant, vQuant, mScales, vScales, lr, beta1, beta2, epsilon,
            1f - beta1, 1f - beta2, 1f - beta1Pow, 1f - beta2Pow, blockSize, length, blocks, weightDecay, decoupledWeightDecay);
    }

    /// <summary>The number of int8 scale blocks for <paramref name="length"/> elements.</summary>
    public static int BlockCount(int length, int blockSize)
    {
        if (length < 0) throw new ArgumentOutOfRangeException(nameof(length));
        if (blockSize <= 0) throw new ArgumentOutOfRangeException(nameof(blockSize));
        return (int)(((long)length + blockSize - 1) / blockSize);
    }

    private static void Validate(float[] param, float[] grad, int length)
    {
        if (param is null) throw new ArgumentNullException(nameof(param));
        if (grad is null) throw new ArgumentNullException(nameof(grad));
        if (length < 0 || param.Length < length || grad.Length < length)
            throw new ArgumentOutOfRangeException(nameof(length));
    }
}
