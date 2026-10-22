using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.Tensors.Engines.DirectGpu.Metal;
using AiDotNet.Tensors.Engines.DirectGpu.OpenCL;
using AiDotNet.Tensors.Engines.DirectGpu.Vulkan;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// #1432: the shared Winograd F(4x4, 3x3) pipeline (<see cref="WinogradF43Pipeline"/>) on every backend's
/// <see cref="IWinogradF43Transforms"/>, against a CPU convolution accumulated in double, with and without the bias + ReLU
/// epilogue the output transform applies. HIP's module is compiled under NVRTC (its dialect is CUDA C++).
/// </summary>
[Collection("DirectGpuSerial")]
public class WinogradF43PortTests
{
    private readonly ITestOutputHelper _output;

    public WinogradF43PortTests(ITestOutputHelper output) => _output = output;

    // batch, inC, outC, size, pad, epilogue: ragged last tiles (sizes not a multiple of 4), every admitted padding, a 1x1
    // image, and the epilogue's bias + ReLU.
    public static readonly TheoryData<int, int, int, int, int, bool> Shapes = new()
    {
        { 2, 5, 7, 9, 1, false },
        { 2, 16, 24, 11, 0, true },
        { 3, 8, 6, 6, 2, false },
        { 1, 3, 4, 1, 1, true },
        { 4, 32, 32, 16, 1, true },
        { 1, 64, 48, 14, 1, false },
    };

    [SkippableTheory]
    [MemberData(nameof(Shapes))]
    public void OpenCl_matches_the_cpu_convolution(int batch, int inC, int outC, int size, int pad, bool epilogue)
    {
        Skip.IfNot(DirectOpenClContext.IsAvailable && DirectOpenClContext.GetDeviceCount() > 0, "No OpenCL device.");
        using var backend = new OpenClBackend(deviceIndex: 0);
        Skip.IfNot(backend.IsAvailable, "OpenCL backend did not initialize.");
        Check(backend, batch, inC, outC, size, pad, epilogue, a => WinogradF43Pipeline.TryForward(backend, backend, a));
    }

    [SkippableTheory]
    [MemberData(nameof(Shapes))]
    public void Cuda_matches_the_cpu_convolution(int batch, int inC, int outC, int size, int pad, bool epilogue)
    {
        Skip.IfNot(CudaBackend.IsCudaAvailable, "CUDA is not available.");
        using var backend = new CudaBackend();
        Skip.IfNot(backend.IsAvailable, "CUDA backend did not initialize.");
        Check(backend, batch, inC, outC, size, pad, epilogue, a => WinogradF43Pipeline.TryForward(backend, backend, a));
    }

    [SkippableTheory]
    [MemberData(nameof(Shapes))]
    public void Vulkan_matches_the_cpu_convolution(int batch, int inC, int outC, int size, int pad, bool epilogue)
    {
        var backend = VulkanBackend.Instance;
        Skip.IfNot(backend.Initialize() && backend.IsGlslCompilerAvailable, "Vulkan with a GLSL compiler is not available.");
        Check(backend, batch, inC, outC, size, pad, epilogue, a => backend.TryConv2DForwardWinogradF43(a));
    }

    [SkippableTheory]
    [MemberData(nameof(Shapes))]
    public void Metal_matches_the_cpu_convolution(int batch, int inC, int outC, int size, int pad, bool epilogue)
    {
        Skip.IfNot(System.Runtime.InteropServices.RuntimeInformation.IsOSPlatform(System.Runtime.InteropServices.OSPlatform.OSX),
            "Metal is only available on macOS.");
        using var backend = new MetalBackend();
        Skip.IfNot(backend.IsAvailable, "Metal is not available.");
        Check(backend, batch, inC, outC, size, pad, epilogue, a => backend.TryConv2DForwardWinogradF43(a));
    }

#if NET7_0_OR_GREATER
    [SkippableTheory]
    [MemberData(nameof(Shapes))]
    public void WebGpu_matches_the_cpu_convolution(int batch, int inC, int outC, int size, int pad, bool epilogue)
    {
        var backend = DawnHostedWebGpu.BackendOrSkip();
        Check(backend, batch, inC, outC, size, pad, epilogue, a => backend.TryConv2DForwardWinogradF43(a));
    }
#endif

    [SkippableFact]
    public void Hip_module_compiles_as_cuda_cxx()
    {
        Skip.IfNot(NvrtcNativeBindings.IsAvailable, "NVRTC is not available.");
        IntPtr program = IntPtr.Zero;
        Assert.Equal(NvrtcResult.Success, NvrtcNativeBindings.nvrtcCreateProgram(ref program,
            WinogradF43KernelTemplate.GetModuleSource(WinogradKernelDialect.Hip), "winograd_f43_hip", 0, IntPtr.Zero,
            IntPtr.Zero));
        try
        {
            var result = NvrtcNativeBindings.nvrtcCompileProgram(program, 1, new[] { "--gpu-architecture=compute_75" });
            Assert.True(result == NvrtcResult.Success, $"the HIP dialect did not compile ({result})");
        }
        finally
        {
            NvrtcNativeBindings.nvrtcDestroyProgram(ref program);
        }
    }

    // A shape the pipeline does not compute (stride 2) is refused with nothing launched.
    [SkippableFact]
    public void A_strided_convolution_is_refused()
    {
        var backend = VulkanBackend.Instance;
        Skip.IfNot(backend.Initialize() && backend.IsGlslCompilerAvailable, "Vulkan with a GLSL compiler is not available.");
        using var x = backend.AllocateBuffer(1 * 4 * 8 * 8);
        using var w = backend.AllocateBuffer(4 * 4 * 9);
        using var y = backend.AllocateBuffer(1 * 4 * 4 * 4);
        Assert.False(backend.TryConv2DForwardWinogradF43(new TunedConv2DArgs(x, w, y, 1, 4, 8, 8, 4, 4, 4, 3, 3, 2, 2, 1, 1,
            1, 1)));
    }

    // Per-layer time of F(4x4) against F(2x2) and the backend's slot choice before this route existed. Set
    // AIDOTNET_PERF_RUNNER=1 on a quiet GPU machine; reports data, asserts nothing about speed.
    public static readonly TheoryData<string, int, int, int, int> BenchShapes = new()
    {
        { "VGG-16 conv1_2", 1, 64, 64, 224 }, { "VGG-16 conv2_2", 1, 128, 128, 112 }, { "VGG-16 conv3_3", 1, 256, 256, 56 },
        { "VGG-16 conv4_3", 1, 512, 512, 28 }, { "VGG-16 conv5_3", 1, 512, 512, 14 },
        { "ResNet-18 64@56", 8, 64, 64, 56 }, { "ResNet-18 128@28", 8, 128, 128, 28 },
        { "ResNet-18 256@14", 8, 256, 256, 14 }, { "ResNet-18 512@7", 8, 512, 512, 7 },
    };

    [SkippableTheory]
    [MemberData(nameof(BenchShapes))]
    [Trait("Category", "Performance")]
    public void Bench_F43_vs_F23(string name, int batch, int inC, int outC, int size)
    {
        Skip.IfNot(Environment.GetEnvironmentVariable("AIDOTNET_PERF_RUNNER") == "1", "Set AIDOTNET_PERF_RUNNER=1 on a quiet GPU machine.");
        var vk = VulkanBackend.Instance;
        bool any = false;
        if (vk.Initialize() && vk.IsGlslCompilerAvailable)
        {
            Bench(vk, "vulkan", name, batch, inC, outC, size, a => vk.TryConv2DForwardWinograd(a), a => vk.TryConv2DForwardWinogradF43(a));
            any = true;
        }
#if NET7_0_OR_GREATER
        var wg = DawnHostedWebGpu.BackendOrNull();
        if (wg is not null)
        {
            Bench(wg, "webgpu", name, batch, inC, outC, size, a => wg.TryConv2DForwardWinograd(a), a => wg.TryConv2DForwardWinogradF43(a));
            any = true;
        }
#endif
        Skip.IfNot(any, "No Vulkan or WebGPU device.");
    }

    private void Bench(IDirectGpuBackend backend, string label, string name, int batch, int inC, int outC, int size,
        Func<TunedConv2DArgs, bool> f23, Func<TunedConv2DArgs, bool> f43)
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
        // Interleaved: F(2x2), F(4x4), F(2x2), F(4x4).
        double a1 = Median(f23), b1 = Median(f43), a2 = Median(f23), b2 = Median(f43);
        _output.WriteLine($"{label} {name}: F(2x2) {a1:F3}/{a2:F3} ms, F(4x4) {b1:F3}/{b2:F3} ms, ratio {Math.Min(a1, a2) / Math.Min(b1, b2):F2}x");
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
        Assert.True(forward(args), "the Winograd F(4x4) forward refused an admitted shape");
        backend.Synchronize();
        var expected = Forward(x, w, epilogue ? bias : null, batch, inC, outC, size, pad);
        Compare($"N{batch} C{inC} K{outC} {size}x{size} pad{pad} epilogue={epilogue}", expected, backend.DownloadBuffer(by));
    }

    // fp32 F(4x4) against a double-accumulated sum, relative to the largest magnitude in the result. Its coefficients
    // reach 8 and 1/24, but the measured error stays under 1e-5 (RTX 3080, every backend here), so it holds F(2x2)'s bound.
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