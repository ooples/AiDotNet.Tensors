using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.HIP;
using AiDotNet.Tensors.Engines.DirectGpu.Metal;
using AiDotNet.Tensors.Engines.DirectGpu.OpenCL;
using AiDotNet.Tensors.Engines.DirectGpu.Vulkan;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// The per-channel convolution epilogue (<see cref="IChannelEpilogueBackend"/>) on every backend against the separate
/// steps computed on the host in float: y = act(((z + bias) * scale) + shift) with each operand present or absent and
/// each activation, and dz = act'(y) * dy * scale. Each step rounds on its own, so the kernels match the host bit for bit
/// on CUDA, OpenCL, Vulkan and HIP; WebGPU and Metal may contract a multiply and an add, so they are held to 1 ulp
/// relative. A tensor past one launch's grid runs in chunks. HIP's module is compiled as CUDA C++ under NVRTC.
/// </summary>
[Collection("DirectGpuSerial")]
public sealed class ChannelEpiloguePortTests
{
    // batch, channels, spatial.
    public static IEnumerable<object[]> Shapes() => new[]
    {
        new object[] { 2, 3, 5 },
        new object[] { 4, 64, 256 },
        new object[] { 1, 7, 1 },
    };

    [SkippableTheory]
    [MemberData(nameof(Shapes))]
    public void Cuda_matches_the_separate_steps(int batch, int channels, int spatial)
    {
        Skip.IfNot(AiDotNet.Tensors.Engines.DirectGpu.CUDA.CudaBackend.IsCudaAvailable, "CUDA is not available.");
        using var backend = new AiDotNet.Tensors.Engines.DirectGpu.CUDA.CudaBackend();
        Skip.IfNot(backend.IsAvailable, "CUDA backend did not initialize.");
        CheckAll(backend, batch, channels, spatial, exact: true);
    }

    [SkippableTheory]
    [MemberData(nameof(Shapes))]
    public void OpenCl_matches_the_separate_steps(int batch, int channels, int spatial)
    {
        Skip.IfNot(DirectOpenClContext.IsAvailable && DirectOpenClContext.GetDeviceCount() > 0, "No OpenCL device.");
        using var backend = new OpenClBackend(deviceIndex: 0);
        Skip.IfNot(backend.IsAvailable, "OpenCL backend did not initialize.");
        CheckAll(backend, batch, channels, spatial, exact: true);
    }

    [SkippableTheory]
    [MemberData(nameof(Shapes))]
    public void Vulkan_matches_the_separate_steps(int batch, int channels, int spatial)
    {
        var backend = VulkanBackend.Instance;
        Skip.IfNot(backend.Initialize() && backend.IsGlslCompilerAvailable, "Vulkan with a GLSL compiler is not available.");
        CheckAll(backend, batch, channels, spatial, exact: true);
    }

    [SkippableTheory]
    [MemberData(nameof(Shapes))]
    public void Metal_matches_the_separate_steps(int batch, int channels, int spatial)
    {
        Skip.IfNot(System.Runtime.InteropServices.RuntimeInformation.IsOSPlatform(System.Runtime.InteropServices.OSPlatform.OSX),
            "Metal is only available on macOS.");
        using var backend = new MetalBackend();
        Skip.IfNot(backend.IsAvailable, "Metal is not available.");
        CheckAll(backend, batch, channels, spatial, exact: false);
    }

    [SkippableTheory]
    [MemberData(nameof(Shapes))]
    public void Hip_matches_the_separate_steps(int batch, int channels, int spatial)
    {
        using var backend = new HipBackend();
        Skip.IfNot(backend.IsAvailable, "HIP is not available.");
        CheckAll(backend, batch, channels, spatial, exact: true);
    }

#if NET7_0_OR_GREATER
    [SkippableTheory]
    [MemberData(nameof(Shapes))]
    public void WebGpu_matches_the_separate_steps(int batch, int channels, int spatial)
    {
        var backend = DawnHostedWebGpu.BackendOrSkip();
        CheckAll(backend, batch, channels, spatial, exact: false);
    }
#endif

    [SkippableFact]
    public void Hip_module_compiles_as_cuda_cxx()
    {
        Skip.IfNot(AiDotNet.Tensors.Engines.DirectGpu.CUDA.NvrtcNativeBindings.IsAvailable, "NVRTC is not available.");
        string source = ChannelEpilogueKernelTemplate.GetModuleSource(WinogradKernelDialect.Hip);
        IntPtr program = IntPtr.Zero;
        Assert.Equal(AiDotNet.Tensors.Engines.DirectGpu.CUDA.NvrtcResult.Success,
            AiDotNet.Tensors.Engines.DirectGpu.CUDA.NvrtcNativeBindings.nvrtcCreateProgram(ref program, source,
                "epilogue_hip", 0, IntPtr.Zero, IntPtr.Zero));
        try
        {
            var result = AiDotNet.Tensors.Engines.DirectGpu.CUDA.NvrtcNativeBindings.nvrtcCompileProgram(program, 1,
                new[] { "--gpu-architecture=compute_75" });
            Assert.True(result == AiDotNet.Tensors.Engines.DirectGpu.CUDA.NvrtcResult.Success,
                $"the HIP dialect did not compile ({result})");
        }
        finally
        {
            AiDotNet.Tensors.Engines.DirectGpu.CUDA.NvrtcNativeBindings.nvrtcDestroyProgram(ref program);
        }
    }

    [Fact]
    public void Slopes_that_are_not_positive_and_finite_are_refused()
    {
        Assert.True(ChannelEpilogueActivation.Accepts(ChannelEpilogueActivation.LeakyRelu, 0.2f));
        Assert.False(ChannelEpilogueActivation.Accepts(ChannelEpilogueActivation.LeakyRelu, 0f));
        Assert.False(ChannelEpilogueActivation.Accepts(ChannelEpilogueActivation.LeakyRelu, -0.1f));
        Assert.False(ChannelEpilogueActivation.Accepts(ChannelEpilogueActivation.LeakyRelu, float.NaN));
        Assert.False(ChannelEpilogueActivation.Accepts(ChannelEpilogueActivation.LeakyRelu, float.PositiveInfinity));
        Assert.False(ChannelEpilogueActivation.Accepts(3, 0f));
    }

    private static float[] Rand(int n, int seed, float scale = 1f, float offset = 0f)
    {
        var rng = new Random(seed);
        var a = new float[n];
        for (int i = 0; i < n; i++) a[i] = (float)((rng.NextDouble() * 2 - 1) * scale) + offset;
        return a;
    }

    private static void CheckAll(IDirectGpuBackend backend, int batch, int channels, int spatial, bool exact)
    {
        var epilogue = Assert.IsAssignableFrom<IChannelEpilogueBackend>(backend);
        int n = batch * channels * spatial;
        var z = Rand(n, 1);
        // A few exact zeros and a NaN: the ReLU keeps NaN, and the mask treats zero as off.
        z[0] = 0f;
        if (n > 3) z[3] = float.NaN;
        var dy = Rand(n, 2);
        var b = Rand(channels, 3, 0.5f);
        var s = Rand(channels, 4, 0.5f, 1f);
        var t = Rand(channels, 5, 0.5f);
        using var zBuf = backend.AllocateBuffer(z);
        using var dyBuf = backend.AllocateBuffer(dy);
        using var bBuf = backend.AllocateBuffer(b);
        using var sBuf = backend.AllocateBuffer(s);
        using var tBuf = backend.AllocateBuffer(t);
        using var yBuf = backend.AllocateBuffer(n);
        using var dzBuf = backend.AllocateBuffer(n);
        const float slope = 0.2f;
        for (int mask = 0; mask < 8; mask++)
            foreach (int act in new[] { ChannelEpilogueActivation.Identity, ChannelEpilogueActivation.Relu,
                         ChannelEpilogueActivation.LeakyRelu })
            {
                bool hasB = (mask & 1) != 0, hasS = (mask & 2) != 0, hasT = (mask & 4) != 0;
                Assert.True(epilogue.TryChannelEpilogue(zBuf, hasB ? bBuf : null, hasS ? sBuf : null, hasT ? tBuf : null, yBuf,
                    batch, channels, spatial, act, slope), $"forward refused (mask {mask}, act {act})");
                Assert.True(epilogue.TryChannelEpilogueBackward(dyBuf, yBuf, hasS ? sBuf : null, dzBuf, batch, channels,
                    spatial, act, slope), $"backward refused (mask {mask}, act {act})");
                backend.Synchronize();
                float[] y = backend.DownloadBuffer(yBuf), dz = backend.DownloadBuffer(dzBuf);
                for (int i = 0; i < n; i++)
                {
                    int c = i / spatial % channels;
                    float v = z[i];
                    if (hasB) v = v + b[c];
                    if (hasS) v = v * s[c];
                    if (hasT) v = v + t[c];
                    if (act == ChannelEpilogueActivation.Relu) v = float.IsNaN(v) ? v : v > 0f ? v : 0f;
                    else if (act == ChannelEpilogueActivation.LeakyRelu) v = v > 0f ? v : slope * v;
                    float g = dy[i];
                    if (act == ChannelEpilogueActivation.Relu) g = v > 0f ? g : 0f;
                    else if (act == ChannelEpilogueActivation.LeakyRelu) g = v > 0f ? g : slope * g;
                    if (hasS) g = g * s[c];
                    Same(v, y[i], exact, $"y[{i}] (mask {mask}, act {act})");
                    Same(g, dz[i], exact, $"dz[{i}] (mask {mask}, act {act})");
                }
            }
    }

    private static void Same(float expected, float actual, bool exact, string label)
    {
        if (float.IsNaN(expected)) { Assert.True(float.IsNaN(actual), $"{label}: {actual}, expected NaN"); return; }
        if (exact)
            Assert.True(TestHelpers.MathCompat.SingleToInt32Bits(expected) == TestHelpers.MathCompat.SingleToInt32Bits(actual),
                $"{label}: {actual} vs the separate steps' {expected}");
        else
            Assert.True(Math.Abs(expected - actual) <= 1.2e-7 * Math.Max(1.0, Math.Abs(expected)),
                $"{label}: {actual} vs the separate steps' {expected}");
    }
}
