// Copyright (c) AiDotNet. All rights reserved.

using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// On CUDA the GPU MSE loss was reachable only through IGpuBatchExecution (Vulkan), and under a tape it always ran on
/// the CPU base -- downloading predictions and targets every training step. It is now composed from recorded device
/// ops, so value and gradient stay on the device.
/// </summary>
[Collection("VulkanGlobalState")]
public sealed class GpuMseLossResidencyTests : IClassFixture<DirectGpuTensorEngineTestFixture>
{
    private readonly DirectGpuTensorEngineTestFixture _fixture;

    public GpuMseLossResidencyTests(DirectGpuTensorEngineTestFixture fixture) => _fixture = fixture;

    [SkippableFact]
    public void MseLossUnderTape_MatchesCpu_WithoutHostTraffic()
    {
        Skip.IfNot(_fixture.IsAvailable, "No GPU device.");
        var gpu = _fixture.Engine!;
        var previous = AiDotNetEngine.Current;
        try
        {
            var rng = new Random(21);
            var x = new Tensor<float>(new[] { 16, 8 });
            var t = new Tensor<float>(new[] { 16, 8 });
            for (int i = 0; i < x.Length; i++) { x[i] = (float)(rng.NextDouble() * 2 - 1); t[i] = (float)(rng.NextDouble() * 2 - 1); }

            (float loss, float[] grad) Run(IEngine e, out long readback, out string sites)
            {
                AiDotNetEngine.Current = e;
                using var tape = new GradientTape<float>();
                var prediction = e.TensorTanh(x);
                GpuLaunchProbe.Reset();
                var loss = e.TensorMSELoss(prediction, t);
                var g = tape.ComputeGradients(loss, new[] { x })[x];
                readback = GpuLaunchProbe.ReadbackBytes;
                sites = string.Join("; ", GpuLaunchProbe.ReadbackSites);
                Assert.Equal(new[] { 1 }, loss.Shape.ToArray());
                return (loss[0], g.ToArray());
            }

            var (cpuLoss, cpuGrad) = Run(new CpuEngine(), out _, out _);
            bool savedCapture = GpuLaunchProbe.CaptureReadbackSites;
            float gpuLoss;
            float[] gpuGrad;
            long readback;
            string where;
            try
            {
                GpuLaunchProbe.CaptureReadbackSites = true;
                (gpuLoss, gpuGrad) = Run(gpu, out readback, out where);
            }
            finally
            {
                GpuLaunchProbe.CaptureReadbackSites = savedCapture;
            }
            Assert.True(readback <= 64, $"MSE forward+backward read back {readback} bytes: {where}");
            Assert.True(Math.Abs(cpuLoss - gpuLoss) < 1e-5f, $"loss cpu {cpuLoss} gpu {gpuLoss}");
            for (int i = 0; i < cpuGrad.Length; i++)
                Assert.True(Math.Abs(cpuGrad[i] - gpuGrad[i]) < 1e-5f, $"d/dx[{i}] cpu {cpuGrad[i]} gpu {gpuGrad[i]}");
        }
        finally
        {
            AiDotNetEngine.Current = previous;
        }
    }
}
