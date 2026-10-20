using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// The four-phase stride-2 input gradient (conv2d_dgrad_s2k3_g*) against a float64 reference on ResNet-18's downsampling
/// layers and odd shapes (partial channel tiles, output-channel counts off the eight-channel step, every admitted output
/// width), for every in-block group count with and without split slices: within 1e-5 of sum|terms|, every element
/// written, bit-reproducible. Shapes it does not compute are refused.
/// </summary>
[Collection("DirectGpuSerial")]
public sealed class ConvDgradS2Tests
{
    private readonly ITestOutputHelper _out;
    public ConvDgradS2Tests(ITestOutputHelper output) => _out = output;

    private static float[] Rand(int n, Random rng)
    {
        var a = new float[n];
        for (int i = 0; i < n; i++) a[i] = (float)(rng.NextDouble() * 2 - 1);
        return a;
    }

    // batch, input-gradient channels C, forward output channels K, input size (twice the output size)
    public static IEnumerable<object[]> Shapes() => new[]
    {
        new object[] { 8, 64, 128, 32 },    // ResNet-18 stage 2 entry: 32 tiles of whole rows
        new object[] { 8, 128, 256, 16 },
        new object[] { 8, 256, 512, 8 },    // deep: the grid choice splits the channels
        new object[] { 3, 40, 24, 16 },     // a partial 32-channel tile; three eight-channel steps
        new object[] { 2, 72, 16, 64 },     // output width 32: one row per block
        new object[] { 2, 12, 20, 8 },      // output width 4: eight rows (two images) per block; K off the eight-channel step
    };

    public static IEnumerable<object[]> ShapesGrids() =>
        Shapes().SelectMany(s => new[] { (0, 0), (1, 1), (2, 1), (4, 1), (1, 3), (4, 2) }
            .Select(g => s.Concat(new object[] { g.Item1, g.Item2 }).ToArray()));

    [SkippableTheory]
    [MemberData(nameof(ShapesGrids))]
    public void InputGradient_MatchesFloat64ReferenceAndIsBitReproducible(int batch, int inC, int outC, int size, int groups, int splits)
    {
        Skip.IfNot(CudaBackend.IsCudaAvailable, "CUDA is not available.");
        using var backend = CudaBackend.CreateOrThrow();
        Skip.IfNot(backend.EnsureDgradS2Module(), "Stride-2 input-gradient kernels did not compile.");
        int o = size / 2;
        var rng = new Random(5150 + batch * 7 + inC * 3 + outC);
        var gy = Rand(batch * outC * o * o, rng);
        var wt = Rand(outC * inC * 9, rng);
        using var dy = backend.AllocateBuffer(gy);
        using var w = backend.AllocateBuffer(wt);
        var poison = Enumerable.Repeat(float.NaN, batch * inC * size * size).ToArray();
        using var x1 = backend.AllocateBuffer(poison);
        using var x2 = backend.AllocateBuffer(poison);
        foreach (var x in new[] { x1, x2 })
            Assert.True(backend.TryConv2DBackwardInputS2K3(new TunedConv2DBackwardArgs(dy, w, x, false, batch, inC, size, size, outC, o, o,
                3, 3, 2, 2, 1, 1, 1, 1), groups, splits), "the stride-2 input gradient refused an admitted shape");
        backend.Synchronize();
        float[] got = backend.DownloadBuffer(x1), again = backend.DownloadBuffer(x2);
        double worst = 0;
        for (int b = 0; b < batch; b++)
            for (int c = 0; c < inC; c++)
                for (int ih = 0; ih < size; ih++)
                    for (int iw = 0; iw < size; iw++)
                    {
                        double s = 0, m = 0;
                        for (int kh = 0; kh < 3; kh++)
                        {
                            int th = ih + 1 - kh;
                            if (th < 0 || (th & 1) != 0 || th / 2 >= o) continue;
                            for (int kw = 0; kw < 3; kw++)
                            {
                                int tw = iw + 1 - kw;
                                if (tw < 0 || (tw & 1) != 0 || tw / 2 >= o) continue;
                                for (int k = 0; k < outC; k++)
                                {
                                    double t = (double)gy[((b * outC + k) * o + th / 2) * o + tw / 2] * wt[((k * inC + c) * 3 + kh) * 3 + kw];
                                    s += t; m += Math.Abs(t);
                                }
                            }
                        }
                        int idx = ((b * inC + c) * size + ih) * size + iw;
                        Assert.False(float.IsNaN(got[idx]), $"dX[{idx}] was never written");
                        Assert.True(TestHelpers.MathCompat.SingleToInt32Bits(got[idx]) == TestHelpers.MathCompat.SingleToInt32Bits(again[idx]),
                            $"dX[{idx}] differs between two identical runs");
                        double rel = Math.Abs(got[idx] - s) / Math.Max(1.0, m);
                        worst = Math.Max(worst, rel);
                        Assert.True(rel <= 1e-5, $"dX[{b},{c},{ih},{iw}] = {got[idx]} vs {s} (|terms| {m})");
                    }
        _out.WriteLine($"g{groups} s{splits}: max err / sum|terms| = {worst:E2}");
    }

    // The measured best (groups, splits) of every ResNet-18 downsampling layer on an RTX 3080 (68 SMs), batch 8.
    [Theory]
    [InlineData(128, 128, 68, 2, 1)]   // 64 -> 128 at 32x32
    [InlineData(64, 256, 68, 4, 1)]    // 128 -> 256 at 16x16
    [InlineData(32, 512, 68, 4, 2)]    // 256 -> 512 at 8x8
    [InlineData(12, 24, 68, 4, 1)]     // three eight-channel steps: never split
    [InlineData(192, 16, 68, 1, 1)]    // enough tiles alone
    public void The_grid_choice_fills_the_device(long blocks, int outC, int sms, int groups, int splits)
    {
        CudaBackend.ChooseDgradS2Grid(blocks, outC, sms, out int g, out int s);
        Assert.Equal((groups, splits), (g, s));
    }

    [SkippableFact]
    public void Shapes_it_does_not_compute_are_refused()
    {
        Skip.IfNot(CudaBackend.IsCudaAvailable, "CUDA is not available.");
        using var backend = CudaBackend.CreateOrThrow();
        using var dy = backend.AllocateBuffer(8 * 64 * 32 * 32);
        using var w = backend.AllocateBuffer(64 * 64 * 9);
        using var dx = backend.AllocateBuffer(8 * 64 * 64 * 64);
        TunedConv2DBackwardArgs A(int inC, int h, int outH, int k, int s, int p) =>
            new(dy, w, dx, false, 2, inC, h, h, 8, outH, outH, k, k, s, s, p, p, 1, 1);
        Assert.False(backend.TryConv2DBackwardInputS2K3(A(16, 32, 32, 3, 1, 1)), "stride 1");
        Assert.False(backend.TryConv2DBackwardInputS2K3(A(16, 32, 16, 1, 2, 0)), "1x1");
        Assert.False(backend.TryConv2DBackwardInputS2K3(A(18, 32, 16, 3, 2, 1)), "channels off a multiple of four");
        Assert.False(backend.TryConv2DBackwardInputS2K3(A(16, 31, 16, 3, 2, 1)), "odd input");
        Assert.False(backend.TryConv2DBackwardInputS2K3(A(16, 24, 12, 3, 2, 1)), "output width not dividing the block");
        Assert.False(backend.TryConv2DBackwardInputS2K3(A(16, 128, 64, 3, 2, 1)), "output width beyond the block");
    }

    // The wiring: the input-gradient slot offers the kernel as a candidate, never as its reference.
    [SkippableFact]
    public void The_input_gradient_slot_registers_the_candidate()
    {
        Skip.IfNot(CudaBackend.IsCudaAvailable, "CUDA is not available.");
        Skip.If(Environment.GetEnvironmentVariable("AIDOTNET_CONV_S2K3_DGRAD") == "0", "The candidate is switched off.");
        using var backend = CudaBackend.CreateOrThrow();
        var ids = backend.Conv2DBackwardDataSlot.CandidateIds;
        Assert.Contains("nvrtc.conv2d_dx.s2k3", ids);
        Assert.NotEqual("nvrtc.conv2d_dx.s2k3", ids[0]);
    }

    /// <summary>
    /// ResNet-18's downsampling layers at batch 8, interleaved round by round (min / median us per call): this kernel with
    /// its grid choice against the engine's phase kernel (with its weight relayout and reduce). Gated on
    /// AIDOTNET_PERF_RUNNER=1.
    /// </summary>
    [SkippableFact]
    public void Bench_S2K3VsEngine()
    {
        Skip.IfNot(Environment.GetEnvironmentVariable("AIDOTNET_PERF_RUNNER") == "1", "Set AIDOTNET_PERF_RUNNER=1 on a quiet GPU machine.");
        Skip.IfNot(CudaBackend.IsCudaAvailable, "CUDA is not available.");
        using var backend = CudaBackend.CreateOrThrow();
        Skip.IfNot(backend.EnsureDgradS2Module(), "Stride-2 input-gradient kernels did not compile.");
        const int rounds = 41, burst = 10;
        var rng = new Random(29);
        foreach (var (inC, size, outC) in new[] { (64, 32, 128), (128, 16, 256), (256, 8, 512) })
        {
            int o = size / 2;
            using var dy = backend.AllocateBuffer(Rand(8 * outC * o * o, rng));
            using var w = backend.AllocateBuffer(Rand(outC * inC * 9, rng));
            using var dx = backend.AllocateBuffer(8 * inC * size * size);
            var a = new TunedConv2DBackwardArgs(dy, w, dx, false, 8, inC, size, size, outC, o, o, 3, 3, 2, 2, 1, 1, 1, 1);
            Func<bool>[] routes = { () => backend.TryConv2DBackwardInputS2K3(a), () => backend.Conv2DBackwardInputViaEngine(a) };
            var times = routes.Select(_ => new List<double>()).ToArray();
            for (int r = 0; r < rounds; r++)
                for (int i = 0; i < routes.Length; i++)
                {
                    for (int j = 0; j < 3; j++) Assert.True(routes[i]());
                    backend.Synchronize();
                    var sw = Stopwatch.StartNew();
                    for (int j = 0; j < burst; j++) routes[i]();
                    backend.Synchronize();
                    times[i].Add(sw.Elapsed.TotalMilliseconds * 1000 / burst);
                }
            string Stat(List<double> t) { var s = t.OrderBy(v => v).ToArray(); return $"{s[0]:F1} / {s[s.Length / 2]:F1}"; }
            _out.WriteLine($"[{inC}->{outC} s2 at {size}x{size}] s2k3 {Stat(times[0])} us, engine {Stat(times[1])} us");
        }
    }
}
