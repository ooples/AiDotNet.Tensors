using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Kernels;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// The 8x8-micro-tile weight-gradient kernels (conv2d_wgrad_r8_64x128 / _128x64; the dW slot's r8 candidates) on every
/// ResNet-18 (batch 8, 32x32) layer shape, with automatic and single split-K, against the implicit-GEMM route. Each
/// configuration's time is reported next to the implicit kernel's.
/// </summary>
[Collection("DirectGpuSerial")]
public sealed class WgradR8KernelTests : IClassFixture<DirectGpuTensorEngineTestFixture>
{
    private readonly DirectGpuTensorEngineTestFixture _fixture;
    private readonly ITestOutputHelper _output;

    public WgradR8KernelTests(DirectGpuTensorEngineTestFixture fixture, ITestOutputHelper output)
    {
        _fixture = fixture;
        _output = output;
    }

    private static float[] Rand(int n, int seed, float scale)
    {
        var rng = new Random(seed);
        var a = new float[n];
        for (int i = 0; i < n; i++) a[i] = (float)((rng.NextDouble() * 2 - 1) * scale);
        return a;
    }

    // (inC, size, outC, kernel, stride, pad): stem, the four stages, and the strided / 1x1 shortcut layers.
    public static IEnumerable<object[]> Shapes() => new[]
    {
        new object[] { 3, 32, 64, 3, 1, 1 }, new object[] { 64, 32, 64, 3, 1, 1 },
        new object[] { 64, 32, 128, 3, 2, 1 }, new object[] { 64, 32, 128, 1, 2, 0 },
        new object[] { 128, 16, 128, 3, 1, 1 }, new object[] { 128, 16, 256, 3, 2, 1 }, new object[] { 128, 16, 256, 1, 2, 0 },
        new object[] { 256, 8, 256, 3, 1, 1 }, new object[] { 256, 8, 512, 3, 2, 1 }, new object[] { 256, 8, 512, 1, 2, 0 },
        new object[] { 512, 4, 512, 3, 1, 1 },
    };

    [SkippableTheory]
    [MemberData(nameof(Shapes))]
    public void R8_wgrad_kernels_match_the_implicit_route(int inC, int size, int outC, int k, int s, int p)
    {
        Skip.IfNot(_fixture.IsAvailable, "No GPU device.");
        var gpu = (DirectGpuTensorEngine)_fixture.Engine!;
        var cuda = gpu.GetBackend() as CudaBackend;
        Skip.If(cuda is null, "CUDA only.");
        const int batch = 8;
        int outSize = (size + 2 * p - k) / s + 1;
        using var x = cuda!.AllocateBuffer(Rand(batch * inC * size * size, 1, 1f));
        using var dy = cuda.AllocateBuffer(Rand(batch * outC * outSize * outSize, 2, 1f));
        using var reference = cuda.AllocateBuffer(outC * inC * k * k);
        using var dw = cuda.AllocateBuffer(outC * inC * k * k);
        var refArgs = new TunedConv2DBackwardArgs(x, dy, reference, true, batch, inC, size, size, outC, outSize, outSize, k, k, s, s, p, p, 1, 1);
        var args = new TunedConv2DBackwardArgs(x, dy, dw, true, batch, inC, size, size, outC, outSize, outSize, k, k, s, s, p, p, 1, 1);
        Assert.True(cuda.Conv2DBackwardKernelViaImplicit(refArgs), "implicit route refused");
        var expected = cuda.DownloadBuffer(reference);
        double scale = Math.Max(1.0, expected.Max(v => Math.Abs((double)v)));

        double Time(Action launch)
        {
            for (int i = 0; i < 10; i++) launch();
            gpu.SynchronizeStream();
            double best = double.MaxValue;
            for (int r = 0; r < 7; r++)
            {
                var sw = Stopwatch.StartNew();
                for (int i = 0; i < 50; i++) launch();
                gpu.SynchronizeStream();
                best = Math.Min(best, sw.Elapsed.TotalMilliseconds * 1000 / 50);
            }
            return best;
        }

        double implicitUs = Time(() => cuda.Conv2DBackwardKernelViaImplicit(refArgs));
        var line = new System.Text.StringBuilder($"[dW {inC}->{outC} {size}x{size} k{k} s{s}] implicit {implicitUs:F1}");
        // Every ResNet-18 layer's outH * outW is a multiple of the K tile, so vectorDy runs the r8v kernels here.
        foreach (var (tm, tn) in CudaConvWgradKernels.TileShapes)
            foreach (bool vectorDy in new[] { false, true })
            foreach (int splits in new[] { 0, 1 })
            {
                string name = $"{(vectorDy ? "r8v" : "r8")} {tm}x{tn}";
                cuda.Fill(dw, float.NaN, dw.Size);
                Assert.True(cuda.TryConv2DBackwardKernelR8(args, tm, tn, splits, vectorDy), $"{name} refused");
                var actual = cuda.DownloadBuffer(dw);
                double maxErr = 0;
                for (int i = 0; i < expected.Length; i++) maxErr = Math.Max(maxErr, Math.Abs(expected[i] - actual[i]) / scale);
                Assert.True(maxErr < 1e-5, $"{name} splits {splits}: max err {maxErr:E2}");
                double us = Time(() => cuda.TryConv2DBackwardKernelR8(args, tm, tn, splits, vectorDy));
                line.Append($" | {name}/{(splits == 0 ? "auto" : splits.ToString())} {us:F1}");
            }
        _output.WriteLine(line.ToString());
    }

    // (inC, h, w, outC, kernel, stride, pad, dilation): non-square inputs and outputs, dilation 2, and a 1x1 shortcut whose
    // P = inC is odd -- the kernel takes H/W, outH/outW and dH/dW separately, and the ResNet shapes above are all square
    // with dilation 1.
    public static IEnumerable<object[]> IrregularShapes() => new[]
    {
        new object[] { 16, 20, 12, 48, 3, 1, 1, 1 },
        new object[] { 24, 17, 23, 40, 3, 2, 1, 1 },
        new object[] { 32, 18, 14, 64, 3, 1, 2, 2 },   // dilation 2, padding 2 keeps the size
        new object[] { 33, 15, 9, 70, 1, 1, 0, 1 },    // 1x1 shortcut, odd P
        new object[] { 24, 16, 8, 40, 3, 1, 1, 1 },    // outH * outW = 128: the r8v (vector dY) kernels, ragged channels
    };

    /// <summary>
    /// Correctness only (no timing) on irregular shapes, at the automatic split count and at fixed counts of 3 and 7: a
    /// large GPU can pick one split for every shape, which would leave the split-K reduction and its per-slice offsets
    /// untested.
    /// </summary>
    [SkippableTheory]
    [MemberData(nameof(IrregularShapes))]
    public void R8_wgrad_kernels_match_the_implicit_route_on_irregular_shapes_and_forced_splits(
        int inC, int h, int w, int outC, int k, int s, int p, int d)
    {
        Skip.IfNot(_fixture.IsAvailable, "No GPU device.");
        var gpu = (DirectGpuTensorEngine)_fixture.Engine!;
        var cuda = gpu.GetBackend() as CudaBackend;
        Skip.If(cuda is null, "CUDA only.");
        const int batch = 4;
        int outH = (h + 2 * p - d * (k - 1) - 1) / s + 1, outW = (w + 2 * p - d * (k - 1) - 1) / s + 1;
        using var x = cuda!.AllocateBuffer(Rand(batch * inC * h * w, 3, 1f));
        using var dy = cuda.AllocateBuffer(Rand(batch * outC * outH * outW, 4, 1f));
        using var reference = cuda.AllocateBuffer(outC * inC * k * k);
        using var dw = cuda.AllocateBuffer(outC * inC * k * k);
        var refArgs = new TunedConv2DBackwardArgs(x, dy, reference, true, batch, inC, h, w, outC, outH, outW, k, k, s, s, p, p, d, d);
        var args = new TunedConv2DBackwardArgs(x, dy, dw, true, batch, inC, h, w, outC, outH, outW, k, k, s, s, p, p, d, d);
        Assert.True(cuda.Conv2DBackwardKernelViaImplicit(refArgs), "implicit route refused");
        var expected = cuda.DownloadBuffer(reference);
        double scale = Math.Max(1.0, expected.Max(v => Math.Abs((double)v)));

        foreach (var (tm, tn) in CudaConvWgradKernels.TileShapes)
            foreach (bool vectorDy in new[] { false, true })
            foreach (int splits in new[] { 0, 3, 7 })
            {
                cuda.Fill(dw, float.NaN, dw.Size);
                Assert.True(cuda.TryConv2DBackwardKernelR8(args, tm, tn, splits, vectorDy), $"r8 {tm}x{tn} refused");
                var actual = cuda.DownloadBuffer(dw);
                double maxErr = 0;
                for (int i = 0; i < expected.Length; i++) maxErr = Math.Max(maxErr, Math.Abs(expected[i] - actual[i]) / scale);
                Assert.True(maxErr < 1e-5, $"r8 {tm}x{tn} splits {splits} [{inC}->{outC} {h}x{w} k{k} s{s} p{p} d{d}]: max err {maxErr:E2}");
            }
    }
}
