// Copyright (c) CrowdTrainer LLC. All rights reserved.
// A transposed convolution's output padding adds rows and columns at the far edge of its output, and the kernel taps
// that reach them DO write them (PyTorch's semantics, and what the forward kernel does). Its gradients must therefore
// read those rows too: the data gradient is a convolution over the WHOLE output gradient (as on the CPU engine), and
// the weight gradient sums over every output position. The hand-written kernels used to stop at outHeight -
// outputPadH, so whenever a tap reached the padding both gradients silently dropped its contribution.

using System;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.Tensors.Engines.DirectGpu.OpenCL;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

[Collection("DirectGpuSerial")]
public class ConvTranspose2DOutputPaddingGradientTests
{
    private static CudaBackend OpenBackendOrSkip()
    {
        // Both prerequisites, not the driver alone: CudaBackend sets IsAvailable false without NVRTC as well
        // (CudaBackend.cs, "if (!CudaNativeBindings.IsAvailable || !NvrtcNativeBindings.IsAvailable)"), and
        // IsCudaAvailable is that same pair. Gating on the driver alone would turn a machine that merely lacks the
        // runtime compiler into a failure where the whole suite means it to skip.
        Skip.IfNot(CudaBackend.IsCudaAvailable, "CUDA driver or NVRTC is unavailable.");
        var backend = new CudaBackend();
        if (!backend.IsAvailable)
        {
            backend.Dispose();
            throw new InvalidOperationException("CUDA and NVRTC were detected, but the CUDA backend failed to initialize.");
        }
        return backend;
    }

    private static OpenClBackend OpenOpenClOrSkip()
    {
        Skip.IfNot(DirectOpenClContext.IsAvailable && DirectOpenClContext.GetDeviceCount() > 0, "No OpenCL device.");
        var backend = new OpenClBackend(deviceIndex: 0);
        if (!backend.IsAvailable)
        {
            backend.Dispose();
            throw new InvalidOperationException("An OpenCL device was detected, but the OpenCL backend failed to initialize.");
        }
        return backend;
    }

    private static float[] Fill(int count, int salt)
    {
        var values = new float[count];
        for (int i = 0; i < count; i++) values[i] = (float)((((i * 37 + salt * 101) % 97) - 48) / 64.0);
        return values;
    }

    private static void AssertClose(double[] expected, float[] actual, string label)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            double error = Math.Abs(expected[i] - actual[i]) / Math.Max(1.0, Math.Abs(expected[i]));
            Assert.True(error <= 1e-5, $"{label}: element {i} expected {expected[i]}, got {actual[i]}");
        }
    }

    /// <summary>
    /// Forward, data gradient and weight gradient against a direct fp64 transposed convolution, at output padding 0
    /// and 1 (stride 2: a 3x3 tap reaches the padding row and column). CUDA, where the bug was found.
    /// </summary>
    [SkippableTheory]
    [InlineData(0)]
    [InlineData(1)]
    public void TheGradientsCoverTheOutputPadding(int outputPad)
    {
        using var backend = OpenBackendOrSkip();
        AssertTheGradientsCoverTheOutputPadding(backend, outputPad);
    }

    /// <summary>
    /// The same three comparisons on OpenCL, whose kernels carried the same clamp. The body is the backend contract,
    /// not one vendor's: it is the only arm that runs on a host with a non-NVIDIA GPU, and the CUDA arm above skips
    /// there, so without it this fix would ship unexecuted everywhere the suite actually runs.
    /// </summary>
    [SkippableTheory]
    [InlineData(0)]
    [InlineData(1)]
    public void TheGradientsCoverTheOutputPaddingOnOpenCl(int outputPad)
    {
        using var backend = OpenOpenClOrSkip();
        AssertTheGradientsCoverTheOutputPadding(backend, outputPad);
    }

    private static void AssertTheGradientsCoverTheOutputPadding(IDirectGpuBackend backend, int outputPad)
    {
        const int N = 2, Ci = 3, Co = 4, H = 5, W = 4, K = 3, S = 2, P = 1;
        int outH = (H - 1) * S - 2 * P + K + outputPad, outW = (W - 1) * S - 2 * P + K + outputPad;
        float[] x = Fill(N * Ci * H * W, 1), w = Fill(Ci * Co * K * K, 2), dy = Fill(N * Co * outH * outW, 3);
        var y = new double[dy.Length];
        var dx = new double[x.Length];
        var dw = new double[w.Length];
        // y[n, co, ih*s - p + kh, iw*s - p + kw] += x[n, ci, ih, iw] * w[ci, co, kh, kw]
        for (int n = 0; n < N; n++)
        for (int ci = 0; ci < Ci; ci++)
        for (int ih = 0; ih < H; ih++)
        for (int iw = 0; iw < W; iw++)
        for (int co = 0; co < Co; co++)
        for (int kh = 0; kh < K; kh++)
        for (int kw = 0; kw < K; kw++)
        {
            int oh = ih * S - P + kh, ow = iw * S - P + kw;
            if (oh < 0 || oh >= outH || ow < 0 || ow >= outW) continue;
            int xi = ((n * Ci + ci) * H + ih) * W + iw, yi = ((n * Co + co) * outH + oh) * outW + ow;
            int wi = ((ci * Co + co) * K + kh) * K + kw;
            y[yi] += (double)x[xi] * w[wi];
            dx[xi] += (double)dy[yi] * w[wi];
            dw[wi] += (double)x[xi] * dy[yi];
        }

        using var bx = backend.AllocateBuffer(x);
        using var bw = backend.AllocateBuffer(w);
        using var bdy = backend.AllocateBuffer(dy);
        using var by = backend.AllocateBuffer(y.Length);
        using var bdx = backend.AllocateBuffer(x.Length);
        using var bdw = backend.AllocateBuffer(w.Length);
        backend.ConvTranspose2D(bx, bw, by, N, Ci, H, W, Co, outH, outW, K, K, S, S, P, P, outputPad, outputPad);
        backend.ConvTranspose2DBackwardInput(bdy, bw, bdx, N, Ci, H, W, Co, outH, outW, K, K, S, S, P, P,
            outputPad, outputPad);
        backend.ConvTranspose2DBackwardKernel(bx, bdy, bdw, N, Ci, H, W, Co, outH, outW, K, K, S, S, P, P,
            outputPad, outputPad);
        backend.Synchronize();
        AssertClose(y, backend.DownloadBuffer(by), $"forward, output padding {outputPad}");
        AssertClose(dx, backend.DownloadBuffer(bdx), $"data gradient, output padding {outputPad}");
        AssertClose(dw, backend.DownloadBuffer(bdw), $"weight gradient, output padding {outputPad}");
    }
}
