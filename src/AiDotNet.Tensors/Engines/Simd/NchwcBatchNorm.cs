using System;
using System.Runtime.CompilerServices;
using static AiDotNet.Tensors.Compatibility.MethodImplHelper;
#if NET5_0_OR_GREATER
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
#endif

namespace AiDotNet.Tensors.Engines.Simd;

/// <summary>
/// Inference-mode batch normalization for NCHWc8 and NCHW float tensors.
///
/// <para>Decomposes ONNX BN into a single pre-computed FMA:
/// <c>out = scale' * x + bias'</c> where
/// <c>scale' = gamma / sqrt(var + eps)</c> and
/// <c>bias' = beta - scale' * mean</c>. That collapses six broadcast ops
/// (sub, add, sqrt, div, mul, add) into one fused FMA per element.</para>
///
/// <para>Both layouts read the input and write the output in place through pinned pointers: one load, one FMA and
/// one store per element, which makes the op bandwidth-bound. An optional clamp applies a ReLU or ReLU6 in the same
/// pass. Earlier versions copied the input into a pooled array and the result back out, three passes over each
/// activation instead of one (a [128, 96, 16, 16] batch norm ran at 4 GB/s).</para>
/// </summary>
internal static class NchwcBatchNorm
{
    public const int CBlock = 8;

    /// <summary>
    /// Inference BN on NCHWc8 layout <c>[N, C/8, H, W, 8]</c>.
    /// <paramref name="gamma"/>/<paramref name="beta"/>/<paramref name="mean"/>/<paramref name="variance"/> have
    /// shape [C] in channel-major order, which is also the lane order of the packed layout.
    /// </summary>
    public static void RunNchwc8(
        ReadOnlySpan<float> input,
        ReadOnlySpan<float> gamma,
        ReadOnlySpan<float> beta,
        ReadOnlySpan<float> mean,
        ReadOnlySpan<float> variance,
        float epsilon,
        Span<float> output,
        int N, int C, int H, int W)
        => RunNchwc8(input, gamma, beta, mean, variance, epsilon, output, N, C, H, W,
            float.NegativeInfinity, float.PositiveInfinity);

    /// <summary>
    /// <see cref="RunNchwc8(ReadOnlySpan{float}, ReadOnlySpan{float}, ReadOnlySpan{float}, ReadOnlySpan{float}, ReadOnlySpan{float}, float, Span{float}, int, int, int, int)"/>
    /// with each output clamped to [<paramref name="clampLow"/>, <paramref name="clampHigh"/>] (0 and +inf for a
    /// ReLU, 0 and 6 for a ReLU6) in the same pass.
    /// </summary>
    [MethodImpl(Hot)]
    public static unsafe void RunNchwc8(
        ReadOnlySpan<float> input,
        ReadOnlySpan<float> gamma,
        ReadOnlySpan<float> beta,
        ReadOnlySpan<float> mean,
        ReadOnlySpan<float> variance,
        float epsilon,
        Span<float> output,
        int N, int C, int H, int W,
        float clampLow, float clampHigh)
    {
        // Shape + buffer-length invariants. NCHWc8 requires C divisible by
        // CBlock; gamma/beta/mean/variance are per-channel (length C); input
        // and output are N·C·H·W floats. Catching this here gives a clear
        // error instead of a downstream AccessViolation when the kernel
        // indexes past a wrongly-sized scale or bias span.
        if (N < 0 || C < 0 || H < 0 || W < 0)
            throw new ArgumentException("N/C/H/W must all be non-negative.");
        if (C % CBlock != 0)
            throw new ArgumentException(
                $"RunNchwc8 requires C ({C}) divisible by CBlock ({CBlock}).", nameof(C));
        long expected = (long)N * C * H * W;
        if (input.Length < expected || output.Length < expected)
            throw new ArgumentException(
                $"input/output must hold at least N·C·H·W = {expected} floats.");
        if (gamma.Length < C || beta.Length < C || mean.Length < C || variance.Length < C)
            throw new ArgumentException(
                $"gamma/beta/mean/variance must each hold at least C = {C} floats.");

        int cg = C / CBlock;
        int spatial = H * W;
        var packedScale = System.Buffers.ArrayPool<float>.Shared.Rent(C);
        var packedBias = System.Buffers.ArrayPool<float>.Shared.Rent(C);
        try
        {
            PackScaleBias(gamma, beta, mean, variance, epsilon, C, packedScale, packedBias);
            fixed (float* inPtr = input)
            fixed (float* outPtr = output)
            fixed (float* scalePtr = packedScale)
            fixed (float* biasPtr = packedBias)
            {
                // ParallelForOrSerial returns after every task has run, so the pinned addresses outlive the work.
                nint inAddr = (nint)inPtr, outAddr = (nint)outPtr, scaleAddr = (nint)scalePtr, biasAddr = (nint)biasPtr;
                long totalOps = (long)N * cg * spatial * CBlock;
                AiDotNet.Tensors.Helpers.CpuParallelSettings.ParallelForOrSerial(0, N * cg, totalOps, task =>
                {
                    int ocg = task % cg;
                    long groupBase = (long)task * spatial * CBlock;
                    ScaleBiasClampPacked((float*)inAddr + groupBase, (float*)outAddr + groupBase, spatial,
                        (float*)scaleAddr + ocg * CBlock, (float*)biasAddr + ocg * CBlock, clampLow, clampHigh);
                });
            }
        }
        finally
        {
            System.Buffers.ArrayPool<float>.Shared.Return(packedBias);
            System.Buffers.ArrayPool<float>.Shared.Return(packedScale);
        }
    }

    /// <summary>
    /// Inference BN on NCHW layout <c>[N, C, H, W]</c>. Shared with the NCHWc
    /// path — we always collapse to pre-combined scale/bias and emit one FMA.
    /// </summary>
    public static void RunNchw(
        ReadOnlySpan<float> input,
        ReadOnlySpan<float> gamma,
        ReadOnlySpan<float> beta,
        ReadOnlySpan<float> mean,
        ReadOnlySpan<float> variance,
        float epsilon,
        Span<float> output,
        int N, int C, int H, int W)
        => RunNchw(input, gamma, beta, mean, variance, epsilon, output, N, C, H, W,
            float.NegativeInfinity, float.PositiveInfinity);

    /// <summary>
    /// <see cref="RunNchw(ReadOnlySpan{float}, ReadOnlySpan{float}, ReadOnlySpan{float}, ReadOnlySpan{float}, ReadOnlySpan{float}, float, Span{float}, int, int, int, int)"/>
    /// with each output clamped to [<paramref name="clampLow"/>, <paramref name="clampHigh"/>] (0 and +inf for a
    /// ReLU, 0 and 6 for a ReLU6) in the same pass.
    /// </summary>
    [MethodImpl(Hot)]
    public static unsafe void RunNchw(
        ReadOnlySpan<float> input,
        ReadOnlySpan<float> gamma,
        ReadOnlySpan<float> beta,
        ReadOnlySpan<float> mean,
        ReadOnlySpan<float> variance,
        float epsilon,
        Span<float> output,
        int N, int C, int H, int W,
        float clampLow, float clampHigh)
    {
        if (N < 0 || C < 0 || H < 0 || W < 0)
            throw new ArgumentException("N/C/H/W must all be non-negative.");
        long expected = (long)N * C * H * W;
        if (input.Length < expected || output.Length < expected)
            throw new ArgumentException(
                $"input/output must hold at least N·C·H·W = {expected} floats.");
        if (gamma.Length < C || beta.Length < C || mean.Length < C || variance.Length < C)
            throw new ArgumentException(
                $"gamma/beta/mean/variance must each hold at least C = {C} floats.");

        int spatial = H * W;
        var packedScale = System.Buffers.ArrayPool<float>.Shared.Rent(C);
        var packedBias = System.Buffers.ArrayPool<float>.Shared.Rent(C);
        try
        {
            PackScaleBias(gamma, beta, mean, variance, epsilon, C, packedScale, packedBias);
            fixed (float* inPtr = input)
            fixed (float* outPtr = output)
            {
                // ParallelForOrSerial returns after every task has run, so the pinned addresses outlive the work.
                nint inAddr = (nint)inPtr, outAddr = (nint)outPtr;
                AiDotNet.Tensors.Helpers.CpuParallelSettings.ParallelForOrSerial(0, N * C, expected, task =>
                {
                    int c = task % C;
                    long chanBase = (long)task * spatial;
                    ScaleBiasClamp((float*)inAddr + chanBase, (float*)outAddr + chanBase, spatial,
                        packedScale[c], packedBias[c], clampLow, clampHigh);
                });
            }
        }
        finally
        {
            System.Buffers.ArrayPool<float>.Shared.Return(packedBias);
            System.Buffers.ArrayPool<float>.Shared.Return(packedScale);
        }
    }

    // scale' = gamma / sqrt(var + eps) and bias' = beta - scale' * mean, computed in double as before.
    private static void PackScaleBias(ReadOnlySpan<float> gamma, ReadOnlySpan<float> beta, ReadOnlySpan<float> mean,
        ReadOnlySpan<float> variance, float epsilon, int C, float[] packedScale, float[] packedBias)
    {
        for (int c = 0; c < C; c++)
        {
            double s = gamma[c] / Math.Sqrt((double)variance[c] + epsilon);
            packedScale[c] = (float)s;
            packedBias[c] = beta[c] - (float)(s * mean[c]);
        }
    }

    // A clamp to (-inf, +inf) is the identity for every finite value and for infinities, and it propagates NaN, so the
    // unclamped call shares this kernel. The vector forms pass the bound FIRST: x86 max/min return the second operand
    // when either is NaN, so max(low, NaN) is NaN, as PyTorch's relu(NaN) is, where max(NaN, low) would be low.
    [MethodImpl(MethodImplOptions.AggressiveInlining | Hot)]
    private static float Clamp(float v, float low, float high)
    {
        if (v < low) v = low;
        if (v > high) v = high;
        return v;
    }

    // The tail rounds as the vector body does: one fused rounding where the body uses FMA, so a plane's last elements
    // (and an NCHW result against the all-vector NCHWc8 one) agree bit for bit.
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float MulAdd(float x, float s, float b)
    {
#if NET5_0_OR_GREATER
        if (Fma.IsSupported) return MathF.FusedMultiplyAdd(x, s, b);
#endif
        return x * s + b;
    }

    // One NCHW channel plane: out[i] = clamp(in[i] * s + b).
    [MethodImpl(MethodImplOptions.AggressiveInlining | Hot)]
    private static unsafe void ScaleBiasClamp(float* input, float* output, int count, float s, float b,
        float low, float high)
    {
        int i = 0;
#if NET5_0_OR_GREATER
        if (Avx.IsSupported && Fma.IsSupported)
        {
            var vS = Vector256.Create(s);
            var vB = Vector256.Create(b);
            var vLow = Vector256.Create(low);
            var vHigh = Vector256.Create(high);
            for (; i + 8 <= count; i += 8)
            {
                var v = Fma.MultiplyAdd(Avx.LoadVector256(input + i), vS, vB);
                Avx.Store(output + i, Avx.Min(vHigh, Avx.Max(vLow, v)));
            }
        }
#else
        int lanes = System.Numerics.Vector<float>.Count;
        if (System.Numerics.Vector.IsHardwareAccelerated && count >= lanes)
        {
            var vS = new System.Numerics.Vector<float>(s);
            var vB = new System.Numerics.Vector<float>(b);
            var vLow = new System.Numerics.Vector<float>(low);
            var vHigh = new System.Numerics.Vector<float>(high);
            for (; i + lanes <= count; i += lanes)
            {
                var v = Unsafe.ReadUnaligned<System.Numerics.Vector<float>>(input + i) * vS + vB;
                Unsafe.WriteUnaligned(output + i,
                    System.Numerics.Vector.Min(vHigh, System.Numerics.Vector.Max(vLow, v)));
            }
        }
#endif
        for (; i < count; i++)
            output[i] = Clamp(MulAdd(input[i], s, b), low, high);
    }

    // One NCHWc8 channel group: every spatial cell holds CBlock lanes with per-lane scale and bias.
    [MethodImpl(MethodImplOptions.AggressiveInlining | Hot)]
    private static unsafe void ScaleBiasClampPacked(float* input, float* output, int spatial, float* scale, float* bias,
        float low, float high)
    {
#if NET5_0_OR_GREATER
        if (Avx.IsSupported && Fma.IsSupported)
        {
            var vS = Avx.LoadVector256(scale);
            var vB = Avx.LoadVector256(bias);
            var vLow = Vector256.Create(low);
            var vHigh = Vector256.Create(high);
            for (int sp = 0; sp < spatial; sp++)
            {
                float* p = input + sp * CBlock;
                var v = Fma.MultiplyAdd(Avx.LoadVector256(p), vS, vB);
                Avx.Store(output + sp * CBlock, Avx.Min(vHigh, Avx.Max(vLow, v)));
            }
            return;
        }
#else
        // CBlock == 8 maps to the 8-lane AVX2 Vector<float>; 4-lane SSE2-only hosts use the scalar loop.
        if (System.Numerics.Vector.IsHardwareAccelerated && System.Numerics.Vector<float>.Count == CBlock)
        {
            var vS = Unsafe.ReadUnaligned<System.Numerics.Vector<float>>(scale);
            var vB = Unsafe.ReadUnaligned<System.Numerics.Vector<float>>(bias);
            var vLow = new System.Numerics.Vector<float>(low);
            var vHigh = new System.Numerics.Vector<float>(high);
            for (int sp = 0; sp < spatial; sp++)
            {
                var v = Unsafe.ReadUnaligned<System.Numerics.Vector<float>>(input + sp * CBlock) * vS + vB;
                Unsafe.WriteUnaligned(output + sp * CBlock,
                    System.Numerics.Vector.Min(vHigh, System.Numerics.Vector.Max(vLow, v)));
            }
            return;
        }
#endif
        for (int sp = 0; sp < spatial; sp++)
            for (int cb = 0; cb < CBlock; cb++)
            {
                int idx = sp * CBlock + cb;
                output[idx] = Clamp(input[idx] * scale[cb] + bias[cb], low, high);
            }
    }
}