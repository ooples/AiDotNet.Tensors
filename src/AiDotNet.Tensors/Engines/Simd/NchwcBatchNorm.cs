using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
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
/// <para>The NCHWc8 path processes contiguous 8-wide lanes with a single
/// <c>Vector256&lt;float&gt;</c> FMA per spatial cell — bandwidth-bound
/// on modern CPUs (one load, one store, effectively one FMA).</para>
/// </summary>
internal static class NchwcBatchNorm
{
    public const int CBlock = 8;

    /// <summary>
    /// Inference BN on NCHWc8 layout <c>[N, C/8, H, W, 8]</c>.
    /// <paramref name="scaleArr"/>/<paramref name="biasArr"/> have shape [C] in
    /// channel-major order; the method packs them on the fly into [C/8, 8] to
    /// match the lane layout.
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
    {
        int cg = C / CBlock;
        int spatial = H * W;
        int hwC = spatial * CBlock;

        // Rent everything from ArrayPool. Old code allocated packedScale/Bias
        // fresh plus input.ToArray() plus new float[output.Length] plus a
        // fresh float[CBlock=8] INSIDE the SIMD hot loop (spatial × 8 allocs
        // per channel group, dominant cost on BN-heavy models).
        var packedScale = System.Buffers.ArrayPool<float>.Shared.Rent(C);
        var packedBias  = System.Buffers.ArrayPool<float>.Shared.Rent(C);
        try
        {
            for (int c = 0; c < C; c++)
            {
                double s = gamma[c] / Math.Sqrt((double)variance[c] + epsilon);
                packedScale[c] = (float)s;
                packedBias[c] = beta[c] - (float)(s * mean[c]);
            }

            var inArr  = System.Buffers.ArrayPool<float>.Shared.Rent(input.Length);
            var outArr = System.Buffers.ArrayPool<float>.Shared.Rent(output.Length);
            try
            {
                input.CopyTo(inArr.AsSpan(0, input.Length));

#if NET5_0_OR_GREATER
                bool useSimd = Avx.IsSupported && Fma.IsSupported;
#endif
                long totalOps = (long)N * cg * spatial;
                if (totalOps >= 64 * 1024)
                {
                    Parallel.For(0, N * cg, task =>
                        ProcessChannelGroup(inArr, outArr, task, cg, spatial, hwC, packedScale, packedBias
#if NET5_0_OR_GREATER
                            , useSimd
#endif
                        ));
                }
                else
                {
                    for (int task = 0; task < N * cg; task++)
                        ProcessChannelGroup(inArr, outArr, task, cg, spatial, hwC, packedScale, packedBias
#if NET5_0_OR_GREATER
                            , useSimd
#endif
                        );
                }

                outArr.AsSpan(0, output.Length).CopyTo(output);
            }
            finally
            {
                System.Buffers.ArrayPool<float>.Shared.Return(outArr);
                System.Buffers.ArrayPool<float>.Shared.Return(inArr);
            }
        }
        finally
        {
            System.Buffers.ArrayPool<float>.Shared.Return(packedBias);
            System.Buffers.ArrayPool<float>.Shared.Return(packedScale);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void ProcessChannelGroup(
        float[] inArr, float[] outArr,
        int task, int cg, int spatial, int hwC,
        float[] packedScale, float[] packedBias
#if NET5_0_OR_GREATER
        , bool useSimd
#endif
    )
    {
        int n = task / cg;
        int ocg = task % cg;
        int groupBase = (n * cg + ocg) * hwC;
        int scaleBase = ocg * CBlock;

#if NET5_0_OR_GREATER
        if (useSimd)
        {
            // Vector load the scale/bias packed lanes — one 32-byte load each.
            var vScale = Unsafe.ReadUnaligned<Vector256<float>>(
                ref Unsafe.As<float, byte>(ref packedScale[scaleBase]));
            var vBias = Unsafe.ReadUnaligned<Vector256<float>>(
                ref Unsafe.As<float, byte>(ref packedBias[scaleBase]));

            for (int sp = 0; sp < spatial; sp++)
            {
                int idx = groupBase + sp * CBlock;
                // One 32-byte load + FMA + 32-byte store per 8 output
                // elements. No scalar unpack, no fresh float[8] alloc.
                var vIn = Unsafe.ReadUnaligned<Vector256<float>>(
                    ref Unsafe.As<float, byte>(ref inArr[idx]));
                var vOut = Fma.MultiplyAdd(vIn, vScale, vBias);
                Unsafe.WriteUnaligned(
                    ref Unsafe.As<float, byte>(ref outArr[idx]), vOut);
            }
        }
        else
#endif
        {
            for (int sp = 0; sp < spatial; sp++)
            {
                int idx = groupBase + sp * CBlock;
                for (int cb = 0; cb < CBlock; cb++)
                {
                    outArr[idx + cb] =
                        inArr[idx + cb] * packedScale[scaleBase + cb] + packedBias[scaleBase + cb];
                }
            }
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
    {
        int spatial = H * W;
        // Rent pre-combined scale/bias from ArrayPool — old code allocated
        // these fresh every call. Small (C floats), but 53 BN calls per
        // ResNet inference adds up to ~53 × 2 × C × sizeof(float) allocs.
        var packedScale = System.Buffers.ArrayPool<float>.Shared.Rent(C);
        var packedBias  = System.Buffers.ArrayPool<float>.Shared.Rent(C);
        try
        {
            for (int c = 0; c < C; c++)
            {
                double s = gamma[c] / Math.Sqrt((double)variance[c] + epsilon);
                packedScale[c] = (float)s;
                packedBias[c] = beta[c] - (float)(s * mean[c]);
            }

            // Old code did input.ToArray() + new float[output.Length] + a
            // fresh `new float[8]` allocation INSIDE the SIMD loop (6 K+
            // allocations per 50 K-element call, dominating the op cost).
            // Rent input + output arrays from ArrayPool instead. Input copy
            // is unavoidable when the caller passes a ReadOnlySpan (can't
            // project back to an array without MemoryMarshal API that's
            // Span-API-gated), but it's a single bulk CopyTo not 6 K small
            // allocations.
            var inArr  = System.Buffers.ArrayPool<float>.Shared.Rent(input.Length);
            var outArr = System.Buffers.ArrayPool<float>.Shared.Rent(output.Length);
            try
            {
                input.CopyTo(inArr.AsSpan(0, input.Length));

#if NET5_0_OR_GREATER
                bool useSimd = Avx.IsSupported && Fma.IsSupported;
#endif
                long totalOps = (long)N * C * spatial;
                if (totalOps >= 64 * 1024)
                {
                    Parallel.For(0, N * C, task =>
                        ProcessChannel(inArr, outArr, task, C, spatial, packedScale, packedBias
#if NET5_0_OR_GREATER
                            , useSimd
#endif
                        ));
                }
                else
                {
                    for (int task = 0; task < N * C; task++)
                        ProcessChannel(inArr, outArr, task, C, spatial, packedScale, packedBias
#if NET5_0_OR_GREATER
                            , useSimd
#endif
                        );
                }

                outArr.AsSpan(0, output.Length).CopyTo(output);
            }
            finally
            {
                System.Buffers.ArrayPool<float>.Shared.Return(outArr);
                System.Buffers.ArrayPool<float>.Shared.Return(inArr);
            }
        }
        finally
        {
            System.Buffers.ArrayPool<float>.Shared.Return(packedBias);
            System.Buffers.ArrayPool<float>.Shared.Return(packedScale);
        }
    }

    /// <summary>
    /// Per-channel FMA kernel extracted so the Parallel.For lambda can call
    /// a named static method (clearer than capturing refs into closures,
    /// which the compiler rejects with CS8175 anyway). Uses
    /// <c>Unsafe.ReadUnaligned</c>/<c>WriteUnaligned</c> for the SIMD
    /// load/store — one 32-byte op each, versus the old code's
    /// <c>Vector256.Create(scalar, scalar, ..., scalar)</c> from 8 array
    /// reads + <c>CopyTo(new float[8])</c> + 8 array writes per vector
    /// (which also allocated a fresh float[8] every iteration — the
    /// dominant cost of the old kernel on BN-heavy models).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void ProcessChannel(
        float[] inArr, float[] outArr,
        int task, int C, int spatial,
        float[] packedScale, float[] packedBias
#if NET5_0_OR_GREATER
        , bool useSimd
#endif
    )
    {
        int n = task / C;
        int c = task % C;
        int chanBase = (n * C + c) * spatial;
        float s = packedScale[c];
        float b = packedBias[c];

#if NET5_0_OR_GREATER
        if (useSimd && spatial >= 8)
        {
            var vS = Vector256.Create(s);
            var vB = Vector256.Create(b);
            int i = 0;
            for (; i + 8 <= spatial; i += 8)
            {
                int idx = chanBase + i;
                // One 32-byte vector load — no scalar unpack.
                var vIn = Unsafe.ReadUnaligned<Vector256<float>>(
                    ref Unsafe.As<float, byte>(ref inArr[idx]));
                var vOut = Fma.MultiplyAdd(vIn, vS, vB);
                // One 32-byte vector store — no fresh float[8] alloc.
                Unsafe.WriteUnaligned(
                    ref Unsafe.As<float, byte>(ref outArr[idx]), vOut);
            }
            for (; i < spatial; i++)
                outArr[chanBase + i] = inArr[chanBase + i] * s + b;
        }
        else
#endif
        {
            for (int i = 0; i < spatial; i++)
                outArr[chanBase + i] = inArr[chanBase + i] * s + b;
        }
    }
}
