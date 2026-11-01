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
/// Stride-1 convolutions with one to three output channels: the input gradient of a network's first layer (RGB or
/// grey input) runs as such a forward convolution (the transposed-convolution identity), and the im2col route gives it
/// an M = 1..3 GEMM per image. Here the output is computed on the zero-padded input's flattened grid: output position
/// q = i * Wp + j reads tap (a, b) at q + a * Wp + b, so every tap of a run of positions is one contiguous (shifted)
/// load. Positions with j >= outW are computed and dropped. Per tap and channel, four (or, for a single output channel,
/// eight) vectors of positions are loaded once and multiplied by each output channel's broadcast weight.
/// </summary>
internal static unsafe class SmallOutConv
{
    /// <summary>Test switch: false routes every convolution away from this kernel.</summary>
    internal static bool s_disable;

    /// <summary>Calls of <see cref="Forward"/> on this thread (engagement probe for tests).</summary>
    [ThreadStatic] internal static int t_runs;

    private static readonly bool s_env = Environment.GetEnvironmentVariable("AIDOTNET_SMALL_OUT_CONV") != "0";

    private const int Run = 32;   // positions per run (four vectors), eight vectors for one output channel

    /// <summary>Whether <see cref="Forward"/> takes the convolution: one to three outputs, stride 1, undilated, and at
    /// least 64 multiplies per output (fewer leave the run loop's overhead in front).</summary>
    internal static bool Applies(int inChannels, int outChannels, int kernelH, int kernelW, int strideH, int strideW,
        int dilationH, int dilationW, int padH, int padW)
    {
        if (s_disable || !s_env || !Avx2.IsSupported || !Fma.IsSupported) return false;
        return outChannels >= 1 && outChannels <= 3 && strideH == 1 && strideW == 1 && dilationH == 1 && dilationW == 1
            && padH >= 0 && padW >= 0 && kernelH <= 7 && kernelW <= 7
            && (long)inChannels * kernelH * kernelW >= 64;
    }

    /// <summary>y [N, K, oH, oW] = the stride-1 convolution of x [N, C, H, W] with w [K, C, kH, kW] (K = 1..3).</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal static void Forward(float[] x, int xOff, float[] w, int wOff, float[] y, int yOff, int batch, int inChannels,
        int height, int width, int outChannels, int kernelH, int kernelW, int padH, int padW)
    {
        int hp = height + 2 * padH, wp = width + 2 * padW;
        int outH = hp - kernelH + 1, outW = wp - kernelW + 1;
        if (outH <= 0 || outW <= 0) return;
        t_runs++;
        int span = outChannels == 1 ? 2 * Run : Run;
        int grid = outH * wp;                                   // flattened output positions per image
        int runs = (grid + span - 1) / span;
        // The last run reads up to span - 1 past the grid, at offsets up to (kH - 1) Wp + kW - 1.
        int plane = hp * wp + span + kernelW;
        int taps = kernelH * kernelW;
        var pool = ArrayPool<float>.Shared;
        float[] padded = pool.Rent(batch * inChannels * plane);
        // Weights as [C][tap][3] (three output channels per tap, zero past K): one broadcast per output and tap.
        float[] wt = pool.Rent(inChannels * taps * 3);
        try
        {
            Array.Clear(wt, 0, inChannels * taps * 3);
            for (int o = 0; o < outChannels; o++)
                for (int c = 0; c < inChannels; c++)
                    for (int t = 0; t < taps; t++)
                        wt[(c * taps + t) * 3 + o] = w[wOff + (o * inChannels + c) * taps + t];
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
                float* tile = stackalloc float[3 * 2 * Run];
                fixed (float* px = padded, pw = wt, py = y)
                {
                    float* image = px + (long)n * inChannels * plane;
                    float* yImage = py + yOff + (long)n * outChannels * outH * outW;
                    for (int r = r0; r < r1; r++)
                    {
                        int q0 = r * span;
                        if (outChannels == 1) RunOne(image, pw, tile, q0, inChannels, plane, wp, kernelH, kernelW);
                        else RunThree(image, pw, tile, q0, inChannels, plane, wp, kernelH, kernelW);
                        // Scatter the run's positions on the output grid (j < outW) to y.
                        int q1 = Math.Min(grid, q0 + span);
                        int i = q0 / wp, j = q0 - i * wp;
                        for (int q = q0; q < q1; q++)
                        {
                            if (j < outW)
                            {
                                long at = (long)i * outW + j;
                                for (int o = 0; o < outChannels; o++)
                                    yImage[(long)o * outH * outW + at] = tile[o * span + (q - q0)];
                            }
                            if (++j == wp) { j = 0; i++; }
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

    // One output x 64 positions: eight accumulators, per tap and channel eight loads and one broadcast.
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
                    a4 = Fma.MultiplyAdd(Avx.LoadVector256(s + 32), g, a4);
                    a5 = Fma.MultiplyAdd(Avx.LoadVector256(s + 40), g, a5);
                    a6 = Fma.MultiplyAdd(Avx.LoadVector256(s + 48), g, a6);
                    a7 = Fma.MultiplyAdd(Avx.LoadVector256(s + 56), g, a7);
                }
            }
        }
        Avx.Store(tile, a0); Avx.Store(tile + 8, a1); Avx.Store(tile + 16, a2); Avx.Store(tile + 24, a3);
        Avx.Store(tile + 32, a4); Avx.Store(tile + 40, a5); Avx.Store(tile + 48, a6); Avx.Store(tile + 56, a7);
    }
}
#endif
