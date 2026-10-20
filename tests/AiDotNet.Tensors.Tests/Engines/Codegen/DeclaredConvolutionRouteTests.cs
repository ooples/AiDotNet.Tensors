// Copyright (c) AiDotNet. All rights reserved.
// The declared convolution route (#890 phase 4b): CudaBackend runs a convolution, its data gradient or its weight
// gradient as an implicit-GEMM program (optionally split), in the tuned slots' argument contract. The bar is a direct
// fp64 convolution written out here, independent of the spec and its interpreter, so the slots' operand roles
// (First/Second differ between the two gradients) are checked as well as the arithmetic.

using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Codegen;

[Collection("DirectGpuSerial")]
public class DeclaredConvolutionRouteTests
{
    private static CudaBackend OpenBackendOrSkip()
    {
        Skip.IfNot(CudaNativeBindings.IsAvailable, "CUDA driver is unavailable.");
        var backend = new CudaBackend();
        if (!backend.IsAvailable)
        {
            backend.Dispose();
            throw new InvalidOperationException("CUDA driver was detected, but the CUDA backend failed to initialize.");
        }
        return backend;
    }

    private readonly record struct Geometry(int N, int C, int K, int H, int W, int Kernel, int Stride, int Pad, int Dilation)
    {
        public int OutH => (H + 2 * Pad - Dilation * (Kernel - 1) - 1) / Stride + 1;
        public int OutW => (W + 2 * Pad - Dilation * (Kernel - 1) - 1) / Stride + 1;
    }

    private static Geometry Shape(int which) => which switch
    {
        0 => new Geometry(2, 6, 10, 9, 8, 3, 2, 1, 1),     // stride 2, ragged tiles
        1 => new Geometry(2, 4, 5, 11, 11, 3, 1, 2, 2),    // dilation 2
        2 => new Geometry(2, 8, 12, 7, 7, 1, 1, 0, 1),     // 1x1
        _ => new Geometry(1, 4, 8, 16, 16, 3, 1, 1, 1),    // a weight-gradient reduction a split divides (oh = 16)
    };

    private static float[] Fill(int count, int salt)
    {
        var values = new float[count];
        for (int i = 0; i < count; i++) values[i] = (float)((((i * 37 + salt * 101) % 97) - 48) / 64.0);
        return values;
    }

    // y[n,k,oh,ow] = sum over c, kh, kw of x[n,c,ih,iw] w[k,c,kh,kw], with ih = oh*s - p + kh*d (zero outside).
    private static void ForEachTap(Geometry g, Action<int, int, int, int, int, int, int, int> visit)
    {
        for (int n = 0; n < g.N; n++)
        for (int k = 0; k < g.K; k++)
        for (int oh = 0; oh < g.OutH; oh++)
        for (int ow = 0; ow < g.OutW; ow++)
        for (int c = 0; c < g.C; c++)
        for (int kh = 0; kh < g.Kernel; kh++)
        for (int kw = 0; kw < g.Kernel; kw++)
        {
            int ih = oh * g.Stride - g.Pad + kh * g.Dilation, iw = ow * g.Stride - g.Pad + kw * g.Dilation;
            if (ih < 0 || ih >= g.H || iw < 0 || iw >= g.W) continue;
            int x = ((n * g.C + c) * g.H + ih) * g.W + iw;
            int y = ((n * g.K + k) * g.OutH + oh) * g.OutW + ow;
            int w = ((k * g.C + c) * g.Kernel + kh) * g.Kernel + kw;
            visit(x, y, w, n, k, c, kh, kw);
        }
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

    /// <summary>The forward pass, plain and with the bias + ReLU fused into the program or run after it.</summary>
    [SkippableTheory]
    [InlineData(0, 64, 64, 0)]
    [InlineData(0, 128, 64, 1)]
    [InlineData(1, 64, 64, 2)]
    [InlineData(2, 128, 64, 1)]
    public void TheForwardRouteMatchesADirectConvolution(int shape, int tileM, int tileN, int epilogue)
    {
        using var backend = OpenBackendOrSkip();
        Skip.IfNot(backend.DeclaredKernelsEnabled, "The declared route needs the sm_75+ codegen conveyor.");
        var g = Shape(shape);
        float[] x = Fill(g.N * g.C * g.H * g.W, 1), w = Fill(g.K * g.C * g.Kernel * g.Kernel, 2), b = Fill(g.K, 3);
        var expected = new double[g.N * g.K * g.OutH * g.OutW];
        ForEachTap(g, (xi, yi, wi, n, k, c, kh, kw) => expected[yi] += (double)x[xi] * w[wi]);
        bool bias = epilogue != 0, relu = epilogue == 1;
        if (bias)
            for (int i = 0; i < expected.Length; i++)
            {
                double v = expected[i] + b[i / (g.OutH * g.OutW) % g.K];
                expected[i] = relu ? Math.Max(0.0, v) : v;
            }

        using var bx = backend.AllocateBuffer(x);
        using var bw = backend.AllocateBuffer(w);
        using var bb = backend.AllocateBuffer(b);
        using var by = backend.AllocateBuffer(expected.Length);
        var args = new CudaConv2DArgs(bx, bw, by, g.N, g.C, g.H, g.W, g.K, g.OutH, g.OutW, g.Kernel, g.Kernel,
            g.Stride, g.Stride, g.Pad, g.Pad, g.Dilation, g.Dilation, bias ? bb : null,
            relu ? FusedActivationType.ReLU : FusedActivationType.None);
        long before = backend.DeclaredProgramLaunches;
        var schedule = new DeclaredConvSchedule(tileM, tileN, 0);
        Assert.True(backend.TryDeclaredConv2DForward(args, schedule), "declined: " + backend.DeclaredConvDeclineReason(
            new DeclaredConvKey(DeclaredConvPass.Forward, g.N, g.C, g.K, g.H, g.W, g.Kernel, g.Stride, g.Pad, g.Dilation,
                relu, schedule)));
        backend.Synchronize();
        Assert.True(backend.DeclaredProgramLaunches > before);
        AssertClose(expected, backend.DownloadBuffer(by), $"forward shape {shape} m{tileM}n{tileN} epilogue {epilogue}");
    }

    /// <summary>The data gradient (First = dOut, Second = kernel) and the weight gradient (First = input,
    /// Second = dOut), unsplit and split.</summary>
    [SkippableTheory]
    [InlineData(0, false, 64, 64, 0)]
    [InlineData(1, false, 128, 64, 0)]
    [InlineData(2, false, 64, 64, 0)]
    [InlineData(0, true, 64, 64, 0)]
    [InlineData(1, true, 128, 64, 0)]
    [InlineData(3, true, 128, 64, 2)]
    [InlineData(3, true, 128, 64, 4)]
    public void TheBackwardRoutesMatchADirectConvolution(int shape, bool weights, int tileM, int tileN, int split)
    {
        using var backend = OpenBackendOrSkip();
        Skip.IfNot(backend.DeclaredKernelsEnabled, "The declared route needs the sm_75+ codegen conveyor.");
        var g = Shape(shape);
        float[] x = Fill(g.N * g.C * g.H * g.W, 1), w = Fill(g.K * g.C * g.Kernel * g.Kernel, 2);
        float[] dy = Fill(g.N * g.K * g.OutH * g.OutW, 4);
        double[] expected = weights ? new double[w.Length] : new double[x.Length];
        if (weights) ForEachTap(g, (xi, yi, wi, n, k, c, kh, kw) => expected[wi] += (double)dy[yi] * x[xi]);
        else ForEachTap(g, (xi, yi, wi, n, k, c, kh, kw) => expected[xi] += (double)dy[yi] * w[wi]);

        using var bFirst = backend.AllocateBuffer(weights ? x : dy);
        using var bSecond = backend.AllocateBuffer(weights ? dy : w);
        // Pre-filled, so a gradient element the program fails to write cannot pass as a zero.
        using var bOut = backend.AllocateBuffer(Fill(expected.Length, 5));
        var args = new TunedConv2DBackwardArgs(bFirst, bSecond, bOut, weights, g.N, g.C, g.H, g.W, g.K, g.OutH,
            g.OutW, g.Kernel, g.Kernel, g.Stride, g.Stride, g.Pad, g.Pad, g.Dilation, g.Dilation);
        long before = backend.DeclaredProgramLaunches;
        var schedule = new DeclaredConvSchedule(tileM, tileN, split);
        Assert.True(backend.TryDeclaredConv2DBackward(args, schedule), "declined: " + backend.DeclaredConvDeclineReason(
            new DeclaredConvKey(weights ? DeclaredConvPass.BackwardWeights : DeclaredConvPass.BackwardData, g.N, g.C, g.K,
                g.H, g.W, g.Kernel, g.Stride, g.Pad, g.Dilation, false, schedule)));
        backend.Synchronize();
        Assert.True(backend.DeclaredProgramLaunches > before);
        AssertClose(expected, backend.DownloadBuffer(bOut),
            $"{(weights ? "dW" : "dX")} shape {shape} m{tileM}n{tileN} split {split}");
    }

    /// <summary>
    /// The data gradient split by output parity (CodegenParitySplit), whose passes each write a strided slice of dX:
    /// every element written once, by the pass that owns its class (dX starts filled, so a missed one shows).
    /// </summary>
    [SkippableTheory]
    [InlineData(0, 64)]
    [InlineData(0, 128)]
    [InlineData(4, 64)]
    public void TheParityDataGradientMatchesADirectConvolution(int shape, int tileM)
    {
        using var backend = OpenBackendOrSkip();
        Skip.IfNot(backend.DeclaredKernelsEnabled, "The declared route needs the sm_75+ codegen conveyor.");
        var g = shape == 4 ? new Geometry(2, 8, 16, 15, 14, 3, 2, 1, 1) : Shape(shape);
        float[] w = Fill(g.K * g.C * g.Kernel * g.Kernel, 2), dy = Fill(g.N * g.K * g.OutH * g.OutW, 4);
        var expected = new double[g.N * g.C * g.H * g.W];
        ForEachTap(g, (xi, yi, wi, n, k, c, kh, kw) => expected[xi] += (double)dy[yi] * w[wi]);
        using var bdy = backend.AllocateBuffer(dy);
        using var bw = backend.AllocateBuffer(w);
        using var bdx = backend.AllocateBuffer(Fill(expected.Length, 5));
        var args = new TunedConv2DBackwardArgs(bdy, bw, bdx, false, g.N, g.C, g.H, g.W, g.K, g.OutH, g.OutW,
            g.Kernel, g.Kernel, g.Stride, g.Stride, g.Pad, g.Pad, g.Dilation, g.Dilation);
        var schedule = new DeclaredConvSchedule(tileM, 64, 0, Parity: true);
        Assert.True(backend.TryDeclaredConv2DBackward(args, schedule), "declined: " + backend.DeclaredConvDeclineReason(
            new DeclaredConvKey(DeclaredConvPass.BackwardData, g.N, g.C, g.K, g.H, g.W, g.Kernel, g.Stride, g.Pad,
                g.Dilation, false, schedule)));
        backend.Synchronize();
        AssertClose(expected, backend.DownloadBuffer(bdx), $"parity dX shape {shape} m{tileM}");
    }

#if NET5_0_OR_GREATER
    /// <summary>Once a key's program is built, a call allocates nothing: the geometry check, the key and the
    /// bindings are values (a small convolution is launch-bound, so host time is its time).</summary>
    [SkippableFact]
    public void ARepeatedCallAllocatesNothing()
    {
        using var backend = OpenBackendOrSkip();
        Skip.IfNot(backend.DeclaredKernelsEnabled, "The declared route needs the sm_75+ codegen conveyor.");
        var g = Shape(0);
        using var bx = backend.AllocateBuffer(Fill(g.N * g.C * g.H * g.W, 1));
        using var bw = backend.AllocateBuffer(Fill(g.K * g.C * 9, 2));
        using var by = backend.AllocateBuffer(g.N * g.K * g.OutH * g.OutW);
        var args = new CudaConv2DArgs(bx, bw, by, g.N, g.C, g.H, g.W, g.K, g.OutH, g.OutW, 3, 3, g.Stride, g.Stride,
            g.Pad, g.Pad, 1, 1);
        var schedule = new DeclaredConvSchedule(64, 64, 0);
        Assert.True(backend.TryDeclaredConv2DForward(args, schedule));   // builds the program
        for (int i = 0; i < 50; i++) backend.TryDeclaredConv2DForward(args, schedule);
        backend.Synchronize();
        // One-off runtime costs (a lock inflating, a method tiering up) measure tens of bytes in all; any per-call
        // allocation is at least an object header, 24 bytes a call, so 1000 calls under 1000 bytes is none.
        long before = GC.GetAllocatedBytesForCurrentThread();
        for (int i = 0; i < 1000; i++) backend.TryDeclaredConv2DForward(args, schedule);
        long allocated = GC.GetAllocatedBytesForCurrentThread() - before;
        backend.Synchronize();
        Assert.True(allocated < 1000, allocated + " bytes over 1000 calls");
    }
#endif

    /// <summary>What the builder does not take declines, so the slot's other routes serve it: a rectangular window,
    /// a split no reduction axis divides, and the declared switch off.</summary>
    [SkippableFact]
    public void UnsupportedCallsDecline()
    {
        using var backend = OpenBackendOrSkip();
        Skip.IfNot(backend.DeclaredKernelsEnabled, "The declared route needs the sm_75+ codegen conveyor.");
        using var x = backend.AllocateBuffer(2 * 4 * 9 * 9);
        using var w = backend.AllocateBuffer(8 * 4 * 3 * 1);
        using var y = backend.AllocateBuffer(2 * 8 * 7 * 9);
        Assert.Null(CudaBackend.DeclaredConvSpec(DeclaredConvPass.Forward, 2, 4, 8, 9, 9, 3, 1, 1, 1, 0, 0, 1, 1, 7, 9,
            biasRelu: false));
        Assert.False(backend.TryDeclaredConv2DForward(new CudaConv2DArgs(x, w, y, 2, 4, 9, 9, 8, 7, 9, 3, 1, 1, 1, 0, 0,
            1, 1), new DeclaredConvSchedule(64, 64, 0)));

        var g = Shape(0);   // dW reduction over n = 2, oh = 5, ow = 4: no axis longer than 4 that 4 divides
        using var bx = backend.AllocateBuffer(g.N * g.C * g.H * g.W);
        using var bdy = backend.AllocateBuffer(g.N * g.K * g.OutH * g.OutW);
        using var bdw = backend.AllocateBuffer(g.K * g.C * 9);
        var args = new TunedConv2DBackwardArgs(bx, bdy, bdw, true, g.N, g.C, g.H, g.W, g.K, g.OutH, g.OutW, 3, 3,
            g.Stride, g.Stride, g.Pad, g.Pad, 1, 1);
        Assert.False(backend.TryDeclaredConv2DBackward(args, new DeclaredConvSchedule(128, 64, 4)));

        backend.DeclaredKernelsSwitch = false;
        Assert.False(backend.TryDeclaredConv2DBackward(args, new DeclaredConvSchedule(64, 64, 0)));
    }
}
