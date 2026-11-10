#if NET6_0_OR_GREATER
// Copyright (c) CrowdTrainer LLC. All rights reserved.

using System;
using System.Linq;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Diagnostics;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.Tensors.Engines.DirectGpu.HIP;
using AiDotNet.Tensors.Engines.DirectGpu.Metal;
using AiDotNet.Tensors.Engines.DirectGpu.OpenCL;
using AiDotNet.Tensors.Engines.DirectGpu.Vulkan;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// The causal depthwise convolution on the GPU engine under a tape: the forward and both gradients must match the CPU
/// engine and stay on the device.
/// </summary>
[Collection("VulkanGlobalState")]
public sealed class CausalDepthwiseConv1DTapeGpuTests : IClassFixture<DirectGpuTensorEngineTestFixture>
{
    private readonly DirectGpuTensorEngineTestFixture _fixture;

    public CausalDepthwiseConv1DTapeGpuTests(DirectGpuTensorEngineTestFixture fixture) => _fixture = fixture;

    internal static Tensor<float> Rand(int[] shape, int seed)
    {
        var rng = new Random(seed);
        var t = new Tensor<float>(shape);
        for (int i = 0; i < t.Length; i++) t[i] = (float)(rng.NextDouble() - 0.5);
        return t;
    }

    internal static void AssertClose(float[] e, float[] a, string name)
    {
        Assert.Equal(e.Length, a.Length);
        double scale = Math.Max(e.Max(v => Math.Abs(v)), 1e-6);
        for (int i = 0; i < e.Length; i++)
            Assert.True(Math.Abs(e[i] - a[i]) <= 1e-4 * scale, $"{name}[{i}]: cpu {e[i]}, gpu {a[i]} (scale {scale})");
    }

    [SkippableTheory]
    [InlineData(2, 9, 40, 4)]
    [InlineData(1, 3, 33, 4)]       // more taps than steps
    [InlineData(4, 64, 96, 2)]
    public void TapeForwardAndGradients_MatchCpu_OnDevice(int rows, int seqLen, int channels, int taps)
    {
        Skip.IfNot(_fixture.IsAvailable, "No GPU device.");
        var gpu = _fixture.Engine!;
        var cpu = new CpuEngine();
        int[] shape = [rows, seqLen, channels];
        var x = Rand(shape, 1);
        var w = Rand([taps, channels], 2);
        var r = Rand(shape, 3);

        (float[] Y, float[] Dx, float[] Dw) Run(CpuEngine e, bool onDevice)
        {
            using var tape = new GradientTape<float>();
            var y = e.CausalDepthwiseConv1D(x, w);
            if (onDevice)
                Assert.True(y.IsGpuResident || y.HasPendingGpuData, "CausalDepthwiseConv1D ran on the host under a tape");
            var loss = e.ReduceSum(e.TensorMultiply(y, r), [0, 1, 2], keepDims: false);
            var g = tape.ComputeGradients(loss, [x, w]);
            if (onDevice)
            {
                Assert.True(g[x].IsGpuResident || g[x].HasPendingGpuData, "dx was computed on the host");
                Assert.True(g[w].IsGpuResident || g[w].HasPendingGpuData, "dw was computed on the host");
            }
            return (y.ToArray(), g[x].ToArray(), g[w].ToArray());
        }

        var c = Run(cpu, false);
        var d = Run(gpu, true);
        AssertClose(c.Y, d.Y, "y");
        AssertClose(c.Dx, d.Dx, "dx");
        AssertClose(c.Dw, d.Dw, "dw");
    }
}

/// <summary>
/// Every backend's <see cref="ICausalDepthwiseConvBackend"/> kernels driven directly, against the CPU engine, with no
/// host-device traffic during the launches.
/// </summary>
[Collection("DirectGpuSerial")]
public sealed class CausalDepthwiseConv1DBackendParityTests
{
    public static TheoryData<int, int, int, int> Shapes => new()
    {
        { 2, 9, 40, 4 },
        { 1, 3, 33, 4 },     // taps > seqLen
        { 3, 17, 70, 1 },    // one tap; channels not a multiple of 32
    };

    private static void Check(IDirectGpuBackend backend, int rows, int seqLen, int channels, int taps)
    {
        var conv = Assert.IsAssignableFrom<ICausalDepthwiseConvBackend>(backend);
        var cpu = new CpuEngine();
        int[] shape = [rows, seqLen, channels];
        var x = CausalDepthwiseConv1DTapeGpuTests.Rand(shape, 11);
        var w = CausalDepthwiseConv1DTapeGpuTests.Rand([taps, channels], 12);
        var dy = CausalDepthwiseConv1DTapeGpuTests.Rand(shape, 13);
        var y = cpu.CausalDepthwiseConv1D(x, w).ToArray();
        var dx = cpu.CausalDepthwiseConv1DBackwardInput(dy, w).ToArray();
        var dw = cpu.CausalDepthwiseConv1DBackwardKernel(dy, x, taps).ToArray();

        using var xB = backend.AllocateBuffer(x.ToArray());
        using var wB = backend.AllocateBuffer(w.ToArray());
        using var dyB = backend.AllocateBuffer(dy.ToArray());
        using var yB = backend.AllocateBuffer(x.Length);
        using var dxB = backend.AllocateBuffer(x.Length);
        using var dwB = backend.AllocateBuffer(taps * channels);
        GpuResidencyScope scope;
        using (scope = GpuResidencyScope.Begin(captureOperations: true))
        {
            conv.CausalDepthwiseConv1DForward(xB, wB, yB, rows, seqLen, channels, taps);
            conv.CausalDepthwiseConv1DBackwardInput(dyB, wB, dxB, rows, seqLen, channels, taps);
            conv.CausalDepthwiseConv1DBackwardKernel(dyB, xB, dwB, rows, seqLen, channels, taps);
        }
        Assert.True(scope.Uploads + scope.Downloads == 0,
            "the causal convolution crossed the host-device boundary: "
            + string.Join(", ", scope.Events.Select(e => $"{e.Kind} {e.Bytes} B {e.Operation}")));
        CausalDepthwiseConv1DTapeGpuTests.AssertClose(y, backend.DownloadBuffer(yB), "y");
        CausalDepthwiseConv1DTapeGpuTests.AssertClose(dx, backend.DownloadBuffer(dxB), "dx");
        CausalDepthwiseConv1DTapeGpuTests.AssertClose(dw, backend.DownloadBuffer(dwB), "dw");
    }

    [SkippableTheory]
    [MemberData(nameof(Shapes))]
    public void OpenCl_MatchesCpu_OnDevice(int rows, int seqLen, int channels, int taps)
    {
        using var backend = new OpenClBackend();
        Skip.IfNot(backend.IsAvailable, "OpenCL is not available.");
        Check(backend, rows, seqLen, channels, taps);
    }

    [SkippableTheory]
    [MemberData(nameof(Shapes))]
    public void Vulkan_MatchesCpu_OnDevice(int rows, int seqLen, int channels, int taps)
    {
        var backend = VulkanBackend.Instance;
        Skip.IfNot(backend.Initialize() && backend.IsGlslCompilerAvailable, "Vulkan with a GLSL compiler is not available.");
        Check(backend, rows, seqLen, channels, taps);
    }

    [SkippableTheory]
    [MemberData(nameof(Shapes))]
    public void Cuda_MatchesCpu_OnDevice(int rows, int seqLen, int channels, int taps)
    {
        using var backend = new CudaBackend();
        Skip.IfNot(backend.IsAvailable, "CUDA is not available.");
        Check(backend, rows, seqLen, channels, taps);
    }

    [SkippableTheory]
    [MemberData(nameof(Shapes))]
    public void Hip_MatchesCpu_OnDevice(int rows, int seqLen, int channels, int taps)
    {
        using var backend = new HipBackend();
        Skip.IfNot(backend.IsAvailable, "HIP is not available.");
        Check(backend, rows, seqLen, channels, taps);
    }

    [SkippableTheory]
    [MemberData(nameof(Shapes))]
    public void Metal_MatchesCpu_OnDevice(int rows, int seqLen, int channels, int taps)
    {
        Skip.IfNot(OperatingSystem.IsMacOS() || OperatingSystem.IsIOS(), "Metal is only available on macOS/iOS.");
        using var backend = new MetalBackend();
        Skip.IfNot(backend.IsAvailable, "Metal is not available.");
        Check(backend, rows, seqLen, channels, taps);
    }

#if NET7_0_OR_GREATER
    [SkippableTheory]
    [MemberData(nameof(Shapes))]
    public void WebGpu_MatchesCpu_OnDevice(int rows, int seqLen, int channels, int taps)
    {
        var backend = DawnHostedWebGpu.BackendOrSkip();   // the real backend, executing on Dawn; shared, so not disposed
        Check(backend, rows, seqLen, channels, taps);
    }
#endif
}
#endif
