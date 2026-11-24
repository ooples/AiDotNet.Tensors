using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// A GPU convolution with a bias and an activation (FusedConv2D) must equal the CPU's for every route the backend can
/// pick, and the same convolution without them (Conv2D) must too: since the shape key includes the epilogue, the two
/// calls are tuned apart and may run different routes. MobileNet-like shapes (small batch, 1x1 and 3x3, strides 1 and
/// 2, few channels), each run twice so the second call takes the tuned decision.
/// </summary>
[Collection("DirectGpuSerial")]
public sealed class FusedConvEpilogueShapeSweepTests : IDisposable
{
    private readonly IEngine _prior = AiDotNetEngine.Current;
    private readonly ITestOutputHelper _output;

    public FusedConvEpilogueShapeSweepTests(ITestOutputHelper output) => _output = output;

    public void Dispose() => AiDotNetEngine.Current = _prior;

    private static Tensor<float> Rand(int[] shape, int seed, float scale)
    {
        var rng = RandomHelper.CreateSeededRandom(seed);
        var t = new Tensor<float>(shape);
        for (int i = 0; i < t.Length; i++) t[i] = (float)((rng.NextDouble() * 2 - 1) * scale);
        return t;
    }

    private static double MaxRelError(Tensor<float> expected, Tensor<float> actual)
    {
        var e = expected.ToArray();
        var a = actual.ToArray();
        double scale = 1e-6, err = 0;
        foreach (var v in e) scale = Math.Max(scale, Math.Abs(v));
        for (int i = 0; i < e.Length; i++) err = Math.Max(err, Math.Abs(e[i] - a[i]));
        return err / scale;
    }

    [SkippableFact]
    public void Gpu_convolutions_with_and_without_an_epilogue_match_the_cpu_on_small_shapes()
    {
        DirectGpuTensorEngine? gpu = null;
        try
        {
            var candidate = new DirectGpuTensorEngine();
            if (candidate.IsGpuAvailable) gpu = candidate;
            else candidate.Dispose();
        }
        catch (Exception ex) when (ex is PlatformNotSupportedException or DllNotFoundException) { }
        Skip.If(gpu is null, "GPU backend did not resolve.");
        var cpu = new CpuEngine();
        var failures = new List<string>();
        using (gpu)
        {
            int seed = 1;
            foreach (int batch in new[] { 4 })
            foreach (int spatial in new[] { 32, 16, 8, 4, 2 })
            foreach (var (inC, outC) in new[] { (3, 16), (16, 8), (8, 48), (48, 8), (16, 96), (96, 24), (24, 144), (144, 24) })
            foreach (var (k, stride) in new[] { (1, 1), (3, 1), (3, 2) })
            foreach (var act in new[] { FusedActivationType.None, FusedActivationType.ReLU, FusedActivationType.ReLU6 })
            foreach (bool withBias in new[] { false, true })
            {
                int pad = k / 2;
                var x = Rand([batch, inC, spatial, spatial], seed++, 1f);
                var w = Rand([outC, inC, k, k], seed++, 0.3f);
                var b = withBias ? Rand([outC], seed++, 0.5f) : null;
                AiDotNetEngine.Current = cpu;
                var expected = cpu.FusedConv2D(x, w, b, stride, stride, pad, pad, 1, 1, act);
                AiDotNetEngine.Current = gpu!;
                for (int run = 0; run < 2; run++)
                {
                    var fused = gpu!.FusedConv2D(x, w, b, stride, stride, pad, pad, 1, 1, act);
                    double err = MaxRelError(expected, fused);
                    if (err > 5e-3)
                        failures.Add($"FusedConv2D b{batch} {inC}->{outC} {spatial}x{spatial} k{k} s{stride} bias={withBias} {act} run{run}: {err:E2}");
                }
                if (b is null && act == FusedActivationType.None)
                {
                    var plain = gpu!.Conv2D(x, w, stride, pad, 1);
                    double err = MaxRelError(expected, plain);
                    if (err > 5e-3)
                        failures.Add($"Conv2D b{batch} {inC}->{outC} {spatial}x{spatial} k{k} s{stride}: {err:E2}");
                }
            }
        }
        foreach (var f in failures) _output.WriteLine(f);
        Assert.True(failures.Count == 0, failures.Count + " mismatches; first: " + (failures.Count > 0 ? failures[0] : ""));
    }
}
