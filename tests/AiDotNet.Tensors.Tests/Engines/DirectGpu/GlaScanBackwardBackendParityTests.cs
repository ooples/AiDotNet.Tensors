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
/// The checkpointed GLA BPTT backward of every backend, driven directly rather than through whichever backend the
/// engine picked: all four gradients must match the managed backward, and the launch must move nothing across the
/// host-device boundary. The sequence lengths cover one step, a whole number of <c>ceil(sqrt(S))</c> chunks, and a
/// partial last chunk.
/// </summary>
[Collection("DirectGpuSerial")]
public sealed class GlaScanBackwardBackendParityTests
{
    public static TheoryData<int, int, int, int> Shapes => new()
    {
        { 1, 1, 8, 2 },    // chunk 1, one chunk
        { 2, 16, 32, 4 },  // chunk 4, four whole chunks
        { 1, 37, 16, 2 },  // chunk 7, five whole chunks and a 2-step tail
    };

    private static float[] Values(int count, int seed, bool gate)
    {
        var rng = RandomHelper.CreateSeededRandom(seed);
        var v = new float[count];
        for (int i = 0; i < count; i++)
            v[i] = gate ? (float)(0.2 + 0.75 * rng.NextDouble()) : (float)(rng.NextDouble() - 0.5);
        return v;
    }

    private static void Check(IDirectGpuBackend backend, int batch, int seqLen, int modelDim, int numHeads)
    {
        int headDim = modelDim / numHeads;
        int n = batch * seqLen * modelDim, ng = batch * seqLen * numHeads;
        int[] shape = { batch, seqLen, modelDim }, gateShape = { batch, seqLen, numHeads };
        var host = new[] { Values(n, 41, false), Values(n, 42, false), Values(n, 43, false), Values(ng, 44, true) };
        var dOut = Values(n, 45, false);

        // Managed reference: loss = sum(out * dOut), so the output gradient is dOut.
        var cpu = new CpuEngine();
        var inputs = host.Select((h, i) => new Tensor<float>(h, i < 3 ? shape : gateShape)).ToArray();
        var expected = new float[4][];
        using (var tape = new GradientTape<float>())
        {
            var output = cpu.GlaScanForward(inputs[0], inputs[1], inputs[2], inputs[3], numHeads);
            var loss = cpu.ReduceSum(cpu.TensorMultiply(output, new Tensor<float>(dOut, shape)), null);
            var grads = tape.ComputeGradients(loss, inputs);
            for (int i = 0; i < 4; i++) expected[i] = grads[inputs[i]].ToArray();
        }

        var bufs = host.Select(backend.AllocateBuffer).ToArray();
        using var dOutB = backend.AllocateBuffer(dOut);
        var gradBufs = new[] { n, n, n, ng }.Select(backend.AllocateBuffer).ToArray();
        try
        {
            GpuResidencyScope scope;
            using (scope = GpuResidencyScope.Begin(captureOperations: true))
            {
                backend.GlaScanBackward(dOutB, bufs[0], bufs[1], bufs[2], bufs[3],
                    gradBufs[0], gradBufs[1], gradBufs[2], gradBufs[3],
                    batch, seqLen, modelDim, numHeads, headDim);
            }
            Assert.True(scope.Uploads + scope.Downloads == 0,
                "the scan backward crossed the host-device boundary: "
                + string.Join(", ", scope.Events.Select(e => $"{e.Kind} {e.Bytes} B {e.Operation}")));

            string[] names = { "dQ", "dK", "dV", "dGate" };
            for (int g = 0; g < 4; g++)
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

    [SkippableTheory]
    [MemberData(nameof(Shapes))]
    public void OpenCl_GlaBackward_MatchesCpu_OnDevice(int batch, int seqLen, int modelDim, int numHeads)
    {
        using var backend = new OpenClBackend();
        Skip.IfNot(backend.IsAvailable, "OpenCL is not available.");
        Check(backend, batch, seqLen, modelDim, numHeads);
    }

    [SkippableTheory]
    [MemberData(nameof(Shapes))]
    public void Vulkan_GlaBackward_MatchesCpu_OnDevice(int batch, int seqLen, int modelDim, int numHeads)
    {
        var backend = VulkanBackend.Instance;
        Skip.IfNot(backend.Initialize() && backend.IsGlslCompilerAvailable, "Vulkan with a GLSL compiler is not available.");
        Check(backend, batch, seqLen, modelDim, numHeads);
    }

    [SkippableTheory]
    [MemberData(nameof(Shapes))]
    public void Cuda_GlaBackward_MatchesCpu_OnDevice(int batch, int seqLen, int modelDim, int numHeads)
    {
        using var backend = new CudaBackend();
        Skip.IfNot(backend.IsAvailable, "CUDA is not available.");
        Check(backend, batch, seqLen, modelDim, numHeads);
    }

    [SkippableTheory]
    [MemberData(nameof(Shapes))]
    public void Hip_GlaBackward_MatchesCpu_OnDevice(int batch, int seqLen, int modelDim, int numHeads)
    {
        using var backend = new HipBackend();
        Skip.IfNot(backend.IsAvailable, "HIP is not available.");
        Check(backend, batch, seqLen, modelDim, numHeads);
    }

    [SkippableTheory]
    [MemberData(nameof(Shapes))]
    public void Metal_GlaBackward_MatchesCpu_OnDevice(int batch, int seqLen, int modelDim, int numHeads)
    {
        Skip.IfNot(OperatingSystem.IsMacOS() || OperatingSystem.IsIOS(), "Metal is only available on macOS/iOS.");
        using var backend = new MetalBackend();
        Skip.IfNot(backend.IsAvailable, "Metal is not available.");
        Check(backend, batch, seqLen, modelDim, numHeads);
    }

#if NET7_0_OR_GREATER
    [SkippableTheory]
    [MemberData(nameof(Shapes))]
    public void WebGpu_GlaBackward_MatchesCpu_OnDevice(int batch, int seqLen, int modelDim, int numHeads)
    {
        var backend = DawnHostedWebGpu.BackendOrSkip();   // the real backend, executing on Dawn; shared, so not disposed
        Check(backend, batch, seqLen, modelDim, numHeads);
    }
#endif
}
#endif
