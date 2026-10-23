using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.Metal;
using AiDotNet.Tensors.Engines.DirectGpu.Vulkan;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// #1432: the fused Winograd F(2x2, 3x3) forward (<see cref="WinogradFusedForwardTemplate"/>; WGSL in
/// WebGpuWinogradFusedKernels) on Vulkan, WebGPU and Metal, against a CPU convolution accumulated in double, with and
/// without the bias + ReLU epilogue its output transform applies.
/// </summary>
[Collection("DirectGpuSerial")]
public class WinogradFusedForwardPortTests
{
    private readonly ITestOutputHelper _output;

    public WinogradFusedForwardPortTests(ITestOutputHelper output) => _output = output;

    // batch, inC, outC, size, pad, epilogue: partial channel steps (5, 3 input channels), partial 32-channel and 32-tile
    // blocks, several output-channel blocks, every admitted padding, a 1x1 image, and ResNet-18 at batch 8.
    public static readonly TheoryData<int, int, int, int, int, bool> Shapes = new()
    {
        { 2, 5, 7, 9, 1, false },
        { 2, 40, 70, 10, 1, true },
        { 3, 8, 6, 6, 2, false },
        { 1, 3, 4, 1, 1, true },
        { 2, 16, 24, 11, 0, true },
        { 8, 64, 64, 56, 1, false },
        { 8, 512, 512, 7, 1, true },
    };

    [SkippableTheory]
    [MemberData(nameof(Shapes))]
    public void Vulkan_matches_the_cpu_convolution(int batch, int inC, int outC, int size, int pad, bool epilogue)
    {
        var backend = VulkanBackend.Instance;
        Skip.IfNot(backend.Initialize() && backend.IsGlslCompilerAvailable, "Vulkan with a GLSL compiler is not available.");
        Check(backend, batch, inC, outC, size, pad, epilogue, a => backend.TryConv2DForwardWinogradFused(a));
    }

    [SkippableTheory]
    [MemberData(nameof(Shapes))]
    public void Metal_matches_the_cpu_convolution(int batch, int inC, int outC, int size, int pad, bool epilogue)
    {
        Skip.IfNot(System.Runtime.InteropServices.RuntimeInformation.IsOSPlatform(System.Runtime.InteropServices.OSPlatform.OSX),
            "Metal is only available on macOS.");
        using var backend = new MetalBackend();
        Skip.IfNot(backend.IsAvailable, "Metal is not available.");
        Check(backend, batch, inC, outC, size, pad, epilogue, a => backend.TryConv2DForwardWinogradFused(a));
    }

#if NET7_0_OR_GREATER
    [SkippableTheory]
    [MemberData(nameof(Shapes))]
    public void WebGpu_matches_the_cpu_convolution(int batch, int inC, int outC, int size, int pad, bool epilogue)
    {
        var backend = DawnHostedWebGpu.BackendOrSkip();
        Check(backend, batch, inC, outC, size, pad, epilogue, a => backend.TryConv2DForwardWinogradFused(a));
    }

    // The WGSL instance declares exactly the template's 4-channel shared memory: 16 KB, WebGPU's guaranteed minimum.
    [Fact]
    public void The_wgsl_instance_fits_webgpus_guaranteed_workgroup_storage()
    {
        int floats = WinogradFusedForwardTemplate.SharedFloats(
            AiDotNet.Tensors.Engines.DirectGpu.WebGpu.WebGpuWinogradFusedKernels.ChannelStep,
            AiDotNet.Tensors.Engines.DirectGpu.WebGpu.WebGpuWinogradFusedKernels.PartChannels);
        Assert.Equal(16384, floats * sizeof(float));
        Assert.Contains($"var<workgroup> sm: array<f32, {floats}>;",
            AiDotNet.Tensors.Engines.DirectGpu.WebGpu.WebGpuWinogradFusedKernels.Source);
    }
#endif

    // The 8-channel instance Metal and Vulkan run declares 32 KB, Apple GPUs' threadgroup-memory limit.
    [Fact]
    public void The_eight_channel_instance_fits_metals_threadgroup_memory()
    {
        var spec = WinogradFusedForwardTemplate.Spec(8, 16);
        Assert.Equal(32768, spec.SharedFloats * sizeof(float));
        Assert.Contains("threadgroup float sm[8192]",
            WinogradF23KernelTemplate.EmitModule(new[] { spec }, WinogradKernelDialect.Metal));
    }

    [SkippableFact]
    public void A_strided_convolution_is_refused()
    {
        var backend = VulkanBackend.Instance;
        Skip.IfNot(backend.Initialize() && backend.IsGlslCompilerAvailable, "Vulkan with a GLSL compiler is not available.");
        using var x = backend.AllocateBuffer(1 * 4 * 8 * 8);
        using var w = backend.AllocateBuffer(4 * 4 * 9);
        using var y = backend.AllocateBuffer(1 * 4 * 4 * 4);
        Assert.False(backend.TryConv2DForwardWinogradFused(new TunedConv2DArgs(x, w, y, 1, 4, 8, 8, 4, 4, 4, 3, 3, 2, 2,
            1, 1, 1, 1)));
    }

    // Per-layer time of the fused route against the two transform + GEMM routes. Set AIDOTNET_PERF_RUNNER=1 on a quiet
    // GPU machine; reports data, asserts nothing about speed.
    [SkippableTheory]
    [MemberData(nameof(WinogradF43PortTests.BenchShapes), MemberType = typeof(WinogradF43PortTests))]
    [Trait("Category", "Performance")]
    public void Bench_fused_vs_pipelines(string name, int batch, int inC, int outC, int size)
    {
        Skip.IfNot(Environment.GetEnvironmentVariable("AIDOTNET_PERF_RUNNER") == "1", "Set AIDOTNET_PERF_RUNNER=1 on a quiet GPU machine.");
        bool any = false;
        var vk = VulkanBackend.Instance;
        if (vk.Initialize() && vk.IsGlslCompilerAvailable)
        {
            Bench(vk, "vulkan", name, batch, inC, outC, size, a => vk.TryConv2DForwardWinograd(a),
                a => vk.TryConv2DForwardWinogradF43(a), a => vk.TryConv2DForwardWinogradFused(a));
            any = true;
        }
#if NET7_0_OR_GREATER
        var wg = DawnHostedWebGpu.BackendOrNull();
        if (wg is not null)
        {
            Bench(wg, "webgpu", name, batch, inC, outC, size, a => wg.TryConv2DForwardWinograd(a),
                a => wg.TryConv2DForwardWinogradF43(a), a => wg.TryConv2DForwardWinogradFused(a));
            any = true;
        }
#endif
        Skip.IfNot(any, "No Vulkan or WebGPU device.");
    }

    private void Bench(IDirectGpuBackend backend, string label, string name, int batch, int inC, int outC, int size,
        Func<TunedConv2DArgs, bool> f23, Func<TunedConv2DArgs, bool> f43, Func<TunedConv2DArgs, bool> fused)
    {
        using var bx = backend.AllocateBuffer(Random(batch * inC * size * size, 1));
        using var bw = backend.AllocateBuffer(Random(outC * inC * 9, 2));
        using var by = backend.AllocateBuffer(batch * outC * size * size);
        var args = new TunedConv2DArgs(bx, bw, by, batch, inC, size, size, outC, size, size, 3, 3, 1, 1, 1, 1, 1, 1);
        double Median(Func<TunedConv2DArgs, bool> route)
        {
            for (int i = 0; i < 3; i++) route(args);
            backend.Synchronize();
            var times = new double[9];
            for (int r = 0; r < times.Length; r++)
            {
                var sw = System.Diagnostics.Stopwatch.StartNew();
                route(args);
                backend.Synchronize();
                times[r] = sw.Elapsed.TotalMilliseconds;
            }
            Array.Sort(times);
            return times[times.Length / 2];
        }
        // Interleaved rounds: F(2x2) + GEMM, F(4x4) + GEMM, fused.
        double a1 = Median(f23), b1 = Median(f43), c1 = Median(fused), a2 = Median(f23), b2 = Median(f43), c2 = Median(fused);
        _output.WriteLine($"{label} {name}: F23+GEMM {a1:F3}/{a2:F3}, F43+GEMM {b1:F3}/{b2:F3}, fused {c1:F3}/{c2:F3} ms");
    }

    private void Check(IDirectGpuBackend backend, int batch, int inC, int outC, int size, int pad, bool epilogue,
        Func<TunedConv2DArgs, bool> forward)
    {
        int outSize = size + 2 * pad - 2;
        var x = Random(batch * inC * size * size, 1);
        var w = Random(outC * inC * 9, 2);
        var bias = Random(outC, 3);
        using var bx = backend.AllocateBuffer(x);
        using var bw = backend.AllocateBuffer(w);
        using var bb = backend.AllocateBuffer(bias);
        using var by = backend.AllocateBuffer(batch * outC * outSize * outSize);
        backend.Fill(by, float.NaN, batch * outC * outSize * outSize);
        var args = epilogue
            ? new TunedConv2DArgs(bx, bw, by, batch, inC, size, size, outC, outSize, outSize, 3, 3, 1, 1, pad, pad, 1, 1,
                bb, FusedActivationType.ReLU)
            : new TunedConv2DArgs(bx, bw, by, batch, inC, size, size, outC, outSize, outSize, 3, 3, 1, 1, pad, pad, 1, 1);
        Assert.True(forward(args), "the fused Winograd forward refused an admitted shape");
        backend.Synchronize();
        Compare($"N{batch} C{inC} K{outC} {size}x{size} pad{pad} epilogue={epilogue}",
            Forward(x, w, epilogue ? bias : null, batch, inC, outC, size, pad), backend.DownloadBuffer(by));
    }

    // fp32 F(2x2) against a double-accumulated sum, relative to the largest magnitude in the result (as
    // WinogradF23PortTests).
    private const double Tolerance = 1e-4;

    private void Compare(string what, double[] expected, float[] actual)
    {
        Assert.Equal(expected.Length, actual.Length);
        double scale = 1e-12, maxErr = 0;
        foreach (var e in expected) scale = Math.Max(scale, Math.Abs(e));
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.False(float.IsNaN(actual[i]) || float.IsInfinity(actual[i]), $"{what}: element {i} is {actual[i]}");
            maxErr = Math.Max(maxErr, Math.Abs(expected[i] - actual[i]) / scale);
        }
        _output.WriteLine($"[{what}] max err / max|ref| {maxErr:E2}");
        Assert.True(maxErr < Tolerance, $"{what}: max err / max|ref| {maxErr:E2}");
    }

    private static float[] Random(int n, int seed)
    {
        var rng = new Random(seed);
        var a = new float[n];
        for (int i = 0; i < n; i++) a[i] = (float)(rng.NextDouble() * 2 - 1);
        return a;
    }

    private static double[] Forward(float[] x, float[] w, float[]? bias, int n, int c, int k, int s, int pad)
    {
        int o = s + 2 * pad - 2;
        var y = new double[n * k * o * o];
        for (int b = 0; b < n; b++)
            for (int kk = 0; kk < k; kk++)
                for (int oh = 0; oh < o; oh++)
                    for (int ow = 0; ow < o; ow++)
                    {
                        double sum = 0;
                        for (int cc = 0; cc < c; cc++)
                            for (int i = 0; i < 3; i++)
                                for (int j = 0; j < 3; j++)
                                {
                                    int ih = oh + i - pad, iw = ow + j - pad;
                                    if (ih < 0 || ih >= s || iw < 0 || iw >= s) continue;
                                    sum += (double)x[((b * c + cc) * s + ih) * s + iw] * w[((kk * c + cc) * 3 + i) * 3 + j];
                                }
                        if (bias is not null) sum = Math.Max(0.0, sum + bias[kk]);
                        y[((b * k + kk) * o + oh) * o + ow] = sum;
                    }
        return y;
    }
}
