using System;
using System.Linq;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// A training-mode BatchNorm on the GPU: the forward used to fill two running-statistic buffers it never reads and turn
/// the kernel's saved 1/sqrt(var+eps) into variance with three launches; the backward turned it back with three more.
/// ResNet-18 runs 20 of these a step. Each conversion is now one launch where the backend has LayerNorm's fused kernels,
/// with the same arithmetic, so the results must not change.
/// </summary>
[Collection("DirectGpuSerial")]
public sealed class BatchNormLaunchCountTests : IClassFixture<DirectGpuTensorEngineTestFixture>
{
    private readonly DirectGpuTensorEngineTestFixture _fixture;
    private readonly ITestOutputHelper _output;

    public BatchNormLaunchCountTests(DirectGpuTensorEngineTestFixture fixture, ITestOutputHelper output)
    {
        _fixture = fixture;
        _output = output;
    }

    private static Tensor<float> Rand(int[] shape, int seed, float scale, float offset = 0f)
    {
        var rng = new Random(seed);
        var t = new Tensor<float>(shape);
        for (int i = 0; i < t.Length; i++) t[i] = offset + (float)((rng.NextDouble() * 2 - 1) * scale);
        return t;
    }

    [SkippableFact]
    public void Training_batch_norm_forward_and_backward_take_two_launches_each_and_match_the_cpu()
    {
        Skip.IfNot(_fixture.IsAvailable, "No GPU device.");
        var gpu = _fixture.Engine as DirectGpuTensorEngine;
        Skip.If(gpu is null, "Not a DirectGpu engine.");
        Skip.IfNot(gpu!.GetBackend() is ILayerNormBackwardPartsBackend, "The backend has no fused statistic conversions.");
        IEngine engine = gpu;
        var cpu = new CpuEngine();

        var x = Rand([4, 8, 5, 5], 1, 1f, 0.3f);
        var gamma = Rand([8], 2, 0.5f, 1f);
        var beta = Rand([8], 3, 0.2f);
        var gradOut = Rand([4, 8, 5, 5], 4, 1f);
        const double eps = 1e-5;

        engine.BatchNorm(x, gamma, beta, eps, out _, out _);   // warm: kernels and buffer pools
        long before = GpuLaunchProbe.TotalEver;
        var y = engine.BatchNorm(x, gamma, beta, eps, out var mean, out var variance);
        long forwardLaunches = GpuLaunchProbe.TotalEver - before;
        before = GpuLaunchProbe.TotalEver;
        var gradIn = engine.BatchNormBackward(gradOut, x, gamma, mean, variance, eps, out var gradGamma, out var gradBeta);
        long backwardLaunches = GpuLaunchProbe.TotalEver - before;
        _output.WriteLine($"forward {forwardLaunches} launches, backward {backwardLaunches}");

        // forward: the batchnorm kernel + one variance conversion; backward: one inverse-std conversion + its kernel.
        Assert.True(forwardLaunches <= 2, $"BatchNorm forward took {forwardLaunches} launches (was 6)");
        Assert.True(backwardLaunches <= 2, $"BatchNorm backward took {backwardLaunches} launches (was 4)");

        var yRef = cpu.BatchNorm(x, gamma, beta, eps, out var meanRef, out var varRef);
        var gradInRef = cpu.BatchNormBackward(gradOut, x, gamma, meanRef, varRef, eps, out var gradGammaRef, out var gradBetaRef);
        void Close(Tensor<float> expected, Tensor<float> actual, string what)
        {
            var e = expected.ToArray();
            var a = actual.ToArray();
            for (int i = 0; i < e.Length; i++)
                Assert.True(Math.Abs(e[i] - a[i]) <= 1e-4f * Math.Max(1f, Math.Abs(e[i])), $"{what}[{i}]: gpu {a[i]} vs cpu {e[i]}");
        }
        Close(yRef, y, "output");
        Close(meanRef, mean, "mean");
        Close(varRef, variance, "variance");
        Close(gradInRef, gradIn, "gradInput");
        Close(gradGammaRef, gradGamma, "gradGamma");
        Close(gradBetaRef, gradBeta, "gradBeta");
    }

    // Register-resident training batch norm on mean + variance (IBatchNormVarianceBackend): one launch each way for a
    // channel slice of at most 8192 elements with spatial % 4 == 0 -- every register width ResNet-18 uses (8 float4s per
    // thread at 32x32, 2 at 16x16, 1 below) plus an odd channel count -- and the two-launch kernels otherwise. That is
    // CUDA's float4 kernel; the portable kernel (OpenCL, HIP, Metal, Vulkan, WebGPU) reads scalars, so on those backends
    // any slice a variant covers is one launch, the float4 condition included.
    [SkippableTheory]
    [InlineData(8, 64, 32, true)]
    [InlineData(8, 128, 16, true)]
    [InlineData(8, 256, 8, true)]
    [InlineData(8, 512, 4, true)]
    [InlineData(3, 5, 6, true)]
    [InlineData(3, 5, 7, false)]    // spatial 49: not float4 rows
    [InlineData(64, 4, 32, false)]  // 65536-element channel slice: over the register budget
    public void Register_resident_batch_norm_takes_one_launch_each_way_and_matches_the_cpu(int batch, int channels, int size, bool resident)
    {
        Skip.IfNot(_fixture.IsAvailable, "No GPU device.");
        var gpu = _fixture.Engine as DirectGpuTensorEngine;
        Skip.If(gpu is null, "Not a DirectGpu engine.");
        Skip.IfNot(gpu!.GetBackend() is IBatchNormVarianceBackend, "The backend has no register-resident batch norm.");
        IEngine engine = gpu;
        var cpu = new CpuEngine();
        var x = Rand([batch, channels, size, size], 11, 1f, 0.3f);
        var gamma = Rand([channels], 12, 0.5f, 1f);
        var beta = Rand([channels], 13, 0.2f);
        var gradOut = Rand([batch, channels, size, size], 14, 1f);
        const double eps = 1e-5;

        engine.BatchNorm(x, gamma, beta, eps, out var warmMean, out var warmVar);   // warm: kernels and buffer pools
        engine.BatchNormBackward(gradOut, x, gamma, warmMean, warmVar, eps, out _, out _);
        long before = GpuLaunchProbe.TotalEver;
        var y = engine.BatchNorm(x, gamma, beta, eps, out var mean, out var variance);
        long forwardLaunches = GpuLaunchProbe.TotalEver - before;
        before = GpuLaunchProbe.TotalEver;
        var gradIn = engine.BatchNormBackward(gradOut, x, gamma, mean, variance, eps, out var gradGamma, out var gradBeta);
        long backwardLaunches = GpuLaunchProbe.TotalEver - before;
        _output.WriteLine($"[{batch},{channels},{size}x{size}] forward {forwardLaunches} launches, backward {backwardLaunches}");
        bool residentHere = gpu.GetBackend() is AiDotNet.Tensors.Engines.DirectGpu.CUDA.CudaBackend
            ? resident
            : BatchNormRegisterKernelTemplate.VariantFor((long)batch * size * size) != 0;
        Assert.Equal(residentHere ? 1 : 2, forwardLaunches);
        Assert.Equal(residentHere ? 1 : 2, backwardLaunches);

        var yRef = cpu.BatchNorm(x, gamma, beta, eps, out var meanRef, out var varRef);
        var gradInRef = cpu.BatchNormBackward(gradOut, x, gamma, meanRef, varRef, eps, out var gradGammaRef, out var gradBetaRef);
        void Close(Tensor<float> expected, Tensor<float> actual, string what, float tol)
        {
            var e = expected.ToArray();
            var a = actual.ToArray();
            for (int i = 0; i < e.Length; i++)
                Assert.True(Math.Abs(e[i] - a[i]) <= tol * Math.Max(1f, Math.Abs(e[i])), $"{what}[{i}]: gpu {a[i]} vs cpu {e[i]}");
        }
        Close(yRef, y, "output", 1e-4f);
        Close(meanRef, mean, "mean", 1e-5f);
        Close(varRef, variance, "variance", 1e-5f);
        Close(gradInRef, gradIn, "gradInput", 1e-4f);
        // The affine gradients sum over the whole slice (up to 65536 terms here): scale the bound by its length.
        float sliceTol = 1e-6f * batch * size * size;
        Close(gradGammaRef, gradGamma, "gradGamma", Math.Max(1e-4f, sliceTol));
        Close(gradBetaRef, gradBeta, "gradBeta", Math.Max(1e-4f, sliceTol));
    }
}
