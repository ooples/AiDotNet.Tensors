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
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// The fused Gated DeltaNet BPTT backward of every backend that implements
/// <see cref="IGatedDeltaNetBackwardBackend"/>, driven directly rather than through whichever backend the engine
/// picked: all five gradients must match the managed backward, and the launch must move nothing across the
/// host-device boundary.
/// </summary>
[Collection("DirectGpuSerial")]
public sealed class GatedDeltaNetScanBackwardBackendParityTests
{
    private const int Batch = 2, SeqLen = 12, ModelDim = 32, NumHeads = 4, HeadDim = ModelDim / NumHeads;

    private static float[] Values(int count, int seed, bool gate)
    {
        var rng = RandomHelper.CreateSeededRandom(seed);
        var v = new float[count];
        for (int i = 0; i < count; i++)
            v[i] = gate ? (float)(0.2 + 0.75 * rng.NextDouble()) : (float)(rng.NextDouble() - 0.5);
        return v;
    }

    private static void Check(IDirectGpuBackend backend)
    {
        var gdn = Assert.IsAssignableFrom<IGatedDeltaNetBackwardBackend>(backend);
        int n = Batch * SeqLen * ModelDim, ng = Batch * SeqLen * NumHeads;
        int[] shape = { Batch, SeqLen, ModelDim }, gateShape = { Batch, SeqLen, NumHeads };
        var host = new[]
        {
            Values(n, 31, false), Values(n, 32, false), Values(n, 33, false),
            Values(ng, 34, true), Values(ng, 35, true),
        };
        var dOut = Values(n, 36, false);

        // Managed reference: loss = sum(out * dOut), so the output gradient is dOut.
        var cpu = new CpuEngine();
        var inputs = host.Select((h, i) => new Tensor<float>(h, i < 3 ? shape : gateShape)).ToArray();
        var expected = new float[5][];
        using (var tape = new GradientTape<float>())
        {
            var output = cpu.GatedDeltaNetScanForward(inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], NumHeads);
            var loss = cpu.ReduceSum(cpu.TensorMultiply(output, new Tensor<float>(dOut, shape)), null);
            var grads = tape.ComputeGradients(loss, inputs);
            for (int i = 0; i < 5; i++) expected[i] = grads[inputs[i]].ToArray();
        }

        var bufs = host.Select(backend.AllocateBuffer).ToArray();
        using var dOutB = backend.AllocateBuffer(dOut);
        var gradBufs = new[] { n, n, n, ng, ng }.Select(backend.AllocateBuffer).ToArray();
        try
        {
            GpuResidencyScope scope;
            using (scope = GpuResidencyScope.Begin(captureOperations: true))
            {
                gdn.GatedDeltaNetScanBackward(dOutB, bufs[0], bufs[1], bufs[2], bufs[3], bufs[4],
                    gradBufs[0], gradBufs[1], gradBufs[2], gradBufs[3], gradBufs[4],
                    Batch, SeqLen, ModelDim, NumHeads, HeadDim);
            }
            Assert.True(scope.Uploads + scope.Downloads == 0,
                "the scan backward crossed the host-device boundary: "
                + string.Join(", ", scope.Events.Select(e => $"{e.Kind} {e.Bytes} B {e.Operation}")));

            string[] names = { "dQ", "dK", "dV", "dAlpha", "dBeta" };
            for (int g = 0; g < 5; g++)
            {
                var actual = backend.DownloadBuffer(gradBufs[g]);
                double scale = Math.Max(expected[g].Max(Math.Abs), 1e-6);
                for (int i = 0; i < expected[g].Length; i++)
                    Assert.True(Math.Abs(expected[g][i] - actual[i]) <= 2e-4 * scale,
                        $"{names[g]}[{i}]: cpu {expected[g][i]}, gpu {actual[i]} (scale {scale})");
            }
        }
        finally
        {
            foreach (var b in bufs) b.Dispose();
            foreach (var b in gradBufs) b.Dispose();
        }
    }

    [SkippableFact]
    public void OpenCl_ScanBackward_MatchesCpu_OnDevice()
    {
        using var backend = new OpenClBackend();
        Skip.IfNot(backend.IsAvailable, "OpenCL is not available.");
        Check(backend);
    }

    [SkippableFact]
    public void Vulkan_ScanBackward_MatchesCpu_OnDevice()
    {
        var backend = VulkanBackend.Instance;
        Skip.IfNot(backend.Initialize() && backend.IsGlslCompilerAvailable, "Vulkan with a GLSL compiler is not available.");
        Check(backend);
    }

    [SkippableFact]
    public void Cuda_ScanBackward_MatchesCpu_OnDevice()
    {
        using var backend = new CudaBackend();
        Skip.IfNot(backend.IsAvailable, "CUDA is not available.");
        Check(backend);
    }

    [SkippableFact]
    public void Hip_ScanBackward_MatchesCpu_OnDevice()
    {
        using var backend = new HipBackend();
        Skip.IfNot(backend.IsAvailable, "HIP is not available.");
        Check(backend);
    }

    [SkippableFact]
    public void Metal_ScanBackward_MatchesCpu_OnDevice()
    {
        Skip.IfNot(OperatingSystem.IsMacOS() || OperatingSystem.IsIOS(), "Metal is only available on macOS/iOS.");
        using var backend = new MetalBackend();
        Skip.IfNot(backend.IsAvailable, "Metal is not available.");
        Check(backend);
    }

#if NET7_0_OR_GREATER
    [SkippableFact]
    public void WebGpu_ScanBackward_MatchesCpu_OnDevice()
    {
        var backend = DawnHostedWebGpu.BackendOrSkip();   // the real backend, executing on Dawn; shared, so not disposed
        Check(backend);
    }
#endif
}
#endif
