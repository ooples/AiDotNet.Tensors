// Copyright (c) CrowdTrainer LLC. All rights reserved.
#if NET5_0_OR_GREATER
using System;
using System.Buffers;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Tensors.Engines.Simd;

/// <summary>
/// Stride-1 convolutions with one to three output channels, or with a short reduction (C kH kW <= 32): a network's
/// RGB or grey first layer, and its input gradient, which runs as a forward convolution with one to three outputs (the
/// transposed-convolution identity). The im2col route gives them an M = 1..3 or K <= 32 GEMM per image.
/// Here the output is computed on the zero-padded input's flattened grid: output position q = i * Wp + j reads tap
/// (a, b) at q + a * Wp + b, so every tap of a run of positions is one contiguous (shifted) load. Runs are 32 positions of an output row (columns past the row are computed and dropped), stored straight into
/// y. Per tap and channel, four vectors of positions (two rows of four for a single output channel) are loaded once and
/// multiplied by each output channel's broadcast weight.
/// </summary>
internal static unsafe class SmallOutConv
{
    /// <summary>Test switch: false routes every convolution away from this kernel.</summary>
    internal static bool s_disable;

    /// <summary>Calls of <see cref="Forward"/> on this thread (engagement probe for tests).</summary>
    [ThreadStatic] internal static int t_runs;

    private static readonly bool s_env = Environment.GetEnvironmentVariable("AIDOTNET_SMALL_OUT_CONV") != "0";

    private const int Run = 32;   // positions per run and row (four vectors)

    /// <summary>Whether <see cref="Forward"/> takes the convolution (stride 1, undilated): one to three outputs with at
    /// least 64 multiplies per output (fewer leave the run loop's overhead in front), or any number of outputs over at
    /// most 32 multiplies each (an RGB or grey first layer: im2col would build a K = C kH kW <= 32 GEMM per image).</summary>
    internal static bool Applies(int inChannels, int outChannels, int kernelH, int kernelW, int strideH, int strideW,
        int dilationH, int dilationW, int padH, int padW)
    {
        if (s_disable || !s_env || !Avx2.IsSupported || !Fma.IsSupported) return false;
        if (strideH != 1 || strideW != 1 || dilationH != 1 || dilationW != 1 || padH < 0 || padW < 0
            || kernelH > 7 || kernelW > 7 || outChannels < 1 || inChannels < 1) return false;
        long reduction = (long)inChannels * kernelH * kernelW;
        return outChannels <= 3 ? reduction >= 64 : reduction <= 32;
    }

    /// <summary>y [N, K, oH, oW] = the stride-1 convolution of x [N, C, H, W] with w [K, C, kH, kW], three output
    /// channels per pass over a run of positions.</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal static void Forward(float[] x, int xOff, float[] w, int wOff, float[] y, int yOff, int batch, int inChannels,
        int height, int width, int outChannels, int kernelH, int kernelW, int padH, int padW)
    {
        int hp = height + 2 * padH, wp = width + 2 * padW;
        int outH = hp - kernelH + 1, outW = wp - kernelW + 1;
        if (outH <= 0 || outW <= 0) return;
        t_runs++;
        // Runs follow the output rows: 32 positions of one row (the last one of a row partly past it), so a run's results
        // are contiguous in y. One output channel takes two rows per run (eight accumulators).
        int rowsPerRun = outChannels == 1 ? 2 : 1;
        int chunks = (outW + Run - 1) / Run;
        int rowRuns = (outH + rowsPerRun - 1) / rowsPerRun;
        int runs = rowRuns * chunks;
        long grid = (long)outH * outW;
        // A run reads up to Run - 1 past its start, rowsPerRun - 1 rows down, at tap offsets up to (kH - 1) Wp + kW - 1.
        int plane = (hp + rowsPerRun) * wp + Run + kernelW;
        int taps = kernelH * kernelW;
        var pool = ArrayPool<float>.Shared;
        float[] padded = pool.Rent(batch * inChannels * plane);
        // Weights as [group][C][tap][3] (three output channels per tap, zero past K): one broadcast per output and tap.
        int groups = (outChannels + 2) / 3;
        int groupWeights = inChannels * taps * 3;
        float[] wt = pool.Rent(groups * groupWeights);
        try
        {
            Array.Clear(wt, 0, groups * groupWeights);
            for (int o = 0; o < outChannels; o++)
                for (int c = 0; c < inChannels; c++)
                    for (int t = 0; t < taps; t++)
                        wt[(o / 3) * groupWeights + (c * taps + t) * 3 + o % 3] = w[wOff + (o * inChannels + c) * taps + t];
            CpuParallelSettings.ParallelForOrSerial(0, batch * inChannels, (long)batch * inChannels * plane, (job) =>
            {
                Array.Clear(padded, job * plane, plane);
                int src = xOff + job * height * width, dst = job * plane + padH * wp + padW;
                for (int i = 0; i < height; i++)
                    Array.Copy(x, src + i * width, padded, dst + i * wp, width);
            }, deterministicSafe: true);

            // About eight tasks per worker.
            int runsPerTask = Math.Max(1, (int)Math.Min(runs, (long)runs * batch / (8L * Math.Max(1, CpuParallelSettings.MaxDegreeOfParallelism))));
            int tasksPerImage = (runs + runsPerTask - 1) / runsPerTask;
            CpuParallelSettings.ParallelForOrSerial(0, batch * tasksPerImage,
                (long)batch * grid * inChannels * taps * outChannels, [MethodImpl(MethodImplOptions.AggressiveOptimization)] (task) =>
            {
                int n = task / tasksPerImage, r0 = (task % tasksPerImage) * runsPerTask;
                int r1 = Math.Min(runs, r0 + runsPerTask);
                float* tile = stackalloc float[3 * Run];
                fixed (float* px = padded, pw = wt, py = y)
                {
                    float* image = px + (long)n * inChannels * plane;
                    float* yImage = py + yOff + (long)n * outChannels * grid;
                    for (int r = r0; r < r1; r++)
                    {
                        int i = (r / chunks) * rowsPerRun, j0 = (r % chunks) * Run;
                        int q0 = i * wp + j0;
                        long bytes = (long)Math.Min(Run, outW - j0) * sizeof(float);
                        float* yRow = yImage + (long)i * outW + j0;
                        if (outChannels == 1)
                        {
                            RunOne(image, pw, tile, q0, inChannels, plane, wp, kernelH, kernelW);
                            Buffer.MemoryCopy(tile, yRow, bytes, bytes);
                            if (i + 1 < outH) Buffer.MemoryCopy(tile + Run, yRow + outW, bytes, bytes);
                            continue;
                        }
                        for (int g = 0; g < groups; g++)
                        {
#if NET9_0_OR_GREATER
                            if (UseAvx512)
                                RunThree512(image, pw + (long)g * groupWeights, tile, q0, inChannels, plane, wp, kernelH, kernelW);
                            else
#endif
                                RunThree(image, pw + (long)g * groupWeights, tile, q0, inChannels, plane, wp, kernelH, kernelW);
                            int o0 = g * 3, count = Math.Min(3, outChannels - o0);
                            for (int o = 0; o < count; o++)
                                Buffer.MemoryCopy(tile + o * Run, yRow + (long)(o0 + o) * grid, bytes, bytes);
                        }
                    }
                }
            }, deterministicSafe: true);
        }
        finally
        {
            pool.Return(padded);
            pool.Return(wt);
        }
    }

    // Three outputs (zero weights past K) x 32 positions: twelve accumulators, per tap and channel four loads and
    // three broadcasts.
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void RunThree(float* image, float* pw, float* tile, int q0, int inChannels, int plane, int wp,
        int kernelH, int kernelW)
    {
        Vector256<float> a00 = Vector256<float>.Zero, a01 = a00, a02 = a00, a03 = a00;
        Vector256<float> a10 = a00, a11 = a00, a12 = a00, a13 = a00;
        Vector256<float> a20 = a00, a21 = a00, a22 = a00, a23 = a00;
        float* wc = pw;
        for (int c = 0; c < inChannels; c++)
        {
            float* xc = image + (long)c * plane + q0;
            for (int a = 0; a < kernelH; a++)
            {
                float* xr = xc + a * wp;
                for (int b = 0; b < kernelW; b++, wc += 3)
                {
                    float* s = xr + b;
                    var x0 = Avx.LoadVector256(s); var x1 = Avx.LoadVector256(s + 8);
                    var x2 = Avx.LoadVector256(s + 16); var x3 = Avx.LoadVector256(s + 24);
                    var g = Vector256.Create(wc[0]);
                    a00 = Fma.MultiplyAdd(x0, g, a00); a01 = Fma.MultiplyAdd(x1, g, a01);
                    a02 = Fma.MultiplyAdd(x2, g, a02); a03 = Fma.MultiplyAdd(x3, g, a03);
                    g = Vector256.Create(wc[1]);
                    a10 = Fma.MultiplyAdd(x0, g, a10); a11 = Fma.MultiplyAdd(x1, g, a11);
                    a12 = Fma.MultiplyAdd(x2, g, a12); a13 = Fma.MultiplyAdd(x3, g, a13);
                    g = Vector256.Create(wc[2]);
                    a20 = Fma.MultiplyAdd(x0, g, a20); a21 = Fma.MultiplyAdd(x1, g, a21);
                    a22 = Fma.MultiplyAdd(x2, g, a22); a23 = Fma.MultiplyAdd(x3, g, a23);
                }
            }
        }
        Avx.Store(tile, a00); Avx.Store(tile + 8, a01); Avx.Store(tile + 16, a02); Avx.Store(tile + 24, a03);
        Avx.Store(tile + 32, a10); Avx.Store(tile + 40, a11); Avx.Store(tile + 48, a12); Avx.Store(tile + 56, a13);
        Avx.Store(tile + 64, a20); Avx.Store(tile + 72, a21); Avx.Store(tile + 80, a22); Avx.Store(tile + 88, a23);
    }

#if NET9_0_OR_GREATER
    /// <summary>Tests only: run the AVX-512 form on a host without AVX-512 (software Vector512, same arithmetic).</summary>
    internal static bool s_forceAvx512ForTest;

    /// <summary>Whether the AVX-512 run is taken where the hardware has it (A/B switch, and for tests).</summary>
    internal static bool s_useAvx512 = Environment.GetEnvironmentVariable("AIDOTNET_SMALL_OUT_CONV_AVX512") != "0";

    private static bool UseAvx512 => s_useAvx512 && (Vector512.IsHardwareAccelerated || s_forceAvx512ForTest);

    // RunThree sixteen positions per vector: six accumulators, per tap and channel two loads and three broadcasts. Each
    // output's FMA chain is RunThree's (taps in the same order, one rounding per FMA): bit-identical results.
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void RunThree512(float* image, float* pw, float* tile, int q0, int inChannels, int plane, int wp,
        int kernelH, int kernelW)
    {
        Vector512<float> a00 = Vector512<float>.Zero, a01 = a00, a10 = a00, a11 = a00, a20 = a00, a21 = a00;
        float* wc = pw;
        for (int c = 0; c < inChannels; c++)
        {
            float* xc = image + (long)c * plane + q0;
            for (int a = 0; a < kernelH; a++)
            {
                float* xr = xc + a * wp;
                for (int b = 0; b < kernelW; b++, wc += 3)
                {
                    float* s = xr + b;
                    var x0 = Vector512.Load(s); var x1 = Vector512.Load(s + 16);
                    var g = Vector512.Create(wc[0]);
                    a00 = Vector512.FusedMultiplyAdd(x0, g, a00); a01 = Vector512.FusedMultiplyAdd(x1, g, a01);
                    g = Vector512.Create(wc[1]);
                    a10 = Vector512.FusedMultiplyAdd(x0, g, a10); a11 = Vector512.FusedMultiplyAdd(x1, g, a11);
                    g = Vector512.Create(wc[2]);
                    a20 = Vector512.FusedMultiplyAdd(x0, g, a20); a21 = Vector512.FusedMultiplyAdd(x1, g, a21);
                }
            }
        }
        Vector512.Store(a00, tile); Vector512.Store(a01, tile + 16);
        Vector512.Store(a10, tile + 32); Vector512.Store(a11, tile + 48);
        Vector512.Store(a20, tile + 64); Vector512.Store(a21, tile + 80);
    }
#endif

    // One output x two rows of 32 positions: eight accumulators, per tap and channel eight loads and one broadcast.
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void RunOne(float* image, float* pw, float* tile, int q0, int inChannels, int plane, int wp,
        int kernelH, int kernelW)
    {
        Vector256<float> a0 = Vector256<float>.Zero, a1 = a0, a2 = a0, a3 = a0, a4 = a0, a5 = a0, a6 = a0, a7 = a0;
        float* wc = pw;
        for (int c = 0; c < inChannels; c++)
        {
            float* xc = image + (long)c * plane + q0;
            for (int a = 0; a < kernelH; a++)
            {
                float* xr = xc + a * wp;
                for (int b = 0; b < kernelW; b++, wc += 3)
                {
                    float* s = xr + b;
                    var g = Vector256.Create(wc[0]);
                    a0 = Fma.MultiplyAdd(Avx.LoadVector256(s), g, a0);
                    a1 = Fma.MultiplyAdd(Avx.LoadVector256(s + 8), g, a1);
                    a2 = Fma.MultiplyAdd(Avx.LoadVector256(s + 16), g, a2);
                    a3 = Fma.MultiplyAdd(Avx.LoadVector256(s + 24), g, a3);
                    float* t = s + wp;
                    a4 = Fma.MultiplyAdd(Avx.LoadVector256(t), g, a4);
                    a5 = Fma.MultiplyAdd(Avx.LoadVector256(t + 8), g, a5);
                    a6 = Fma.MultiplyAdd(Avx.LoadVector256(t + 16), g, a6);
                    a7 = Fma.MultiplyAdd(Avx.LoadVector256(t + 24), g, a7);
                }
            }
        }
        Avx.Store(tile, a0); Avx.Store(tile + 8, a1); Avx.Store(tile + 16, a2); Avx.Store(tile + 24, a3);
        Avx.Store(tile + 32, a4); Avx.Store(tile + 40, a5); Avx.Store(tile + 48, a6); Avx.Store(tile + 56, a7);
    }
}
#endif
