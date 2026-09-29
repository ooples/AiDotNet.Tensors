using System;
using System.Linq;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Diagnostics;
using AiDotNet.Tensors.Engines.Gpu;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// Under a gradient tape the GPU FusedConv2D ran the convolution on the host: the CPU base it deferred to calls the
/// non-virtual CpuEngine.Conv2D(int[]...). It now runs on the device and records the same Conv2D node.
/// </summary>
[Collection("DirectGpuSerial")]
public sealed class TapedFusedConv2DGpuTests
{
    private const int Batch = 2, InChannels = 3, Size = 8, OutChannels = 4, KernelSize = 3, Stride = 1, Padding = 1;
    // The device and host convolutions sum in different orders; float agreement to this relative level.
    private const float RelativeTolerance = 1e-4f;

    private static float[] Rand(int n, int seed)
    {
        var rng = new Random(seed);
        var d = new float[n];
        for (int i = 0; i < n; i++) d[i] = (float)(rng.NextDouble() - 0.5);
        return d;
    }

    private static (float Loss, float[] DInput, float[] DKernel, float[] DBias) Gradients(IEngine engine, Func<Tensor<float>, Tensor<float>> place)
    {
        var x = place(new Tensor<float>(Rand(Batch * InChannels * Size * Size, 1), new[] { Batch, InChannels, Size, Size }));
        var k = place(new Tensor<float>(Rand(OutChannels * InChannels * KernelSize * KernelSize, 2), new[] { OutChannels, InChannels, KernelSize, KernelSize }));
        var b = place(new Tensor<float>(Rand(OutChannels, 3), new[] { OutChannels }));
        using var tape = new GradientTape<float>();
        var y = engine.FusedConv2D(x, k, b, Stride, Stride, Padding, Padding, 1, 1, FusedActivationType.ReLU);
        var loss = engine.ReduceSum(engine.TensorSquare(y), null);
        var grads = tape.ComputeGradients(loss, new[] { x, k, b });
        return (loss.GetFlat(0), grads[x].ToArray(), grads[k].ToArray(), grads[b].ToArray());
    }

    private static void AssertClose(string name, float[] expected, float[] actual)
    {
        Assert.Equal(expected.Length, actual.Length);
        float scale = expected.Max(v => Math.Abs(v));
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - actual[i]) <= RelativeTolerance * Math.Max(1f, scale),
                $"{name}[{i}]: GPU {actual[i]} vs CPU {expected[i]}");
    }

    [SkippableFact]
    public void TapedFusedConv2D_OnTheGpu_MatchesTheCpuGradients_WithoutLeavingTheDevice()
    {
        using var gpu = new DirectGpuTensorEngine();
        Skip.If(!gpu.IsGpuAvailable, "needs a DirectGpu backend (CUDA/OpenCL/...).");

        var cpu = Gradients(new CpuEngine(), t => t);

        (float Loss, float[] DInput, float[] DKernel, float[] DBias) onGpu;
        GpuResidencyScope scope;
        using (scope = GpuResidencyScope.Begin(captureOperations: true))
            onGpu = Gradients(gpu, t => gpu.UploadToGpu(t, GpuTensorRole.General));

        var convCrossings = scope.Events.Where(e => e.Operation is { } op && op.Contains("FusedConv2D")).ToList();
        Assert.True(convCrossings.Count == 0,
            "the taped convolution crossed the host/device boundary: " +
            string.Join(", ", convCrossings.Select(e => $"{e.Kind} {e.Bytes} B")));

        Assert.True(Math.Abs(onGpu.Loss - cpu.Loss) <= RelativeTolerance * Math.Max(1f, Math.Abs(cpu.Loss)),
            $"loss: GPU {onGpu.Loss} vs CPU {cpu.Loss}");
        AssertClose("dInput", cpu.DInput, onGpu.DInput);
        AssertClose("dKernel", cpu.DKernel, onGpu.DKernel);
        AssertClose("dBias", cpu.DBias, onGpu.DBias);
    }
}
