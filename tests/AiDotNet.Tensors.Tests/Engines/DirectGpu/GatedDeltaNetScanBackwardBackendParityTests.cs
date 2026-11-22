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
/// The fused Gated DeltaNet scan of every backend that implements <see cref="IGatedDeltaNetBackwardBackend"/> and
/// <see cref="IGatedDeltaRuleForwardBackend"/>, driven directly rather than through whichever backend the engine picked,
/// under both <see cref="GatedDeltaRule"/>s: the forward output and all five gradients must match the managed scan, and
/// the backward launch must move nothing across the host-device boundary.
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

    /// <summary>
    /// Both reduction modes: the atomic accumulation, and deterministic mode's fixed-order partial sums, which must
    /// also give bit-identical gradients on a repeat run.
    /// </summary>
    private static void Check(IDirectGpuBackend backend)
    {
        var saved = BlasProvider.GetThreadLocalDeterministicMode();
        try
        {
            foreach (var rule in new[] { GatedDeltaRule.ScaledKeys, GatedDeltaRule.Standard })
            {
                BlasProvider.SetThreadLocalDeterministicMode(false);
                CheckOnce(backend, rule);
                BlasProvider.SetThreadLocalDeterministicMode(true);
                var first = CheckOnce(backend, rule);
                var second = CheckOnce(backend, rule);
                for (int g = 0; g < first.Length; g++)
                    for (int i = 0; i < first[g].Length; i++)
                        Assert.True(BitConverter.SingleToInt32Bits(first[g][i]) == BitConverter.SingleToInt32Bits(second[g][i]),
                            $"{rule}: deterministic mode gave gradient {g} element {i} as {first[g][i]:R}, then {second[g][i]:R}");
            }
        }
        finally
        {
            BlasProvider.SetThreadLocalDeterministicMode(saved);
        }
    }

    private static float[][] CheckOnce(IDirectGpuBackend backend, GatedDeltaRule rule)
    {
        var gdn = Assert.IsAssignableFrom<IGatedDeltaNetBackwardBackend>(backend);
        var forward = Assert.IsAssignableFrom<IGatedDeltaRuleForwardBackend>(backend);
        // The kernels' terms of each rule: the key scale, and whether the prediction reads the decayed state.
        float keyScale = rule == GatedDeltaRule.Standard ? 1f : GatedDeltaNetBackwardResolver.OriginalKeyScale(HeadDim);
        bool decayFirst = rule == GatedDeltaRule.Standard;
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
        float[] expectedOut;
        using (var tape = new GradientTape<float>())
        {
            var output = cpu.GatedDeltaNetScanForward(inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], NumHeads, rule);
            expectedOut = output.ToArray();
            var loss = cpu.ReduceSum(cpu.TensorMultiply(output, new Tensor<float>(dOut, shape)), null);
            var grads = tape.ComputeGradients(loss, inputs);
            for (int i = 0; i < 5; i++) expected[i] = grads[inputs[i]].ToArray();
        }

        var bufs = host.Select(backend.AllocateBuffer).ToArray();
        using var dOutB = backend.AllocateBuffer(dOut);
        var gradBufs = new[] { n, n, n, ng, ng }.Select(backend.AllocateBuffer).ToArray();
        using var outB = backend.AllocateBuffer(n);
        try
        {
            forward.GatedDeltaNetScanForward(bufs[0], bufs[1], bufs[2], bufs[3], bufs[4], outB,
                Batch, SeqLen, ModelDim, NumHeads, HeadDim, keyScale, decayFirst);
            var gotOut = backend.DownloadBuffer(outB);
            double outScale = Math.Max(expectedOut.Max(Math.Abs), 1e-6);
            for (int i = 0; i < n; i++)
                Assert.True(Math.Abs(expectedOut[i] - gotOut[i]) <= 2e-4 * outScale,
                    $"{rule} out[{i}]: cpu {expectedOut[i]}, gpu {gotOut[i]} (scale {outScale})");

            GpuResidencyScope scope;
            using (scope = GpuResidencyScope.Begin(captureOperations: true))
            {
                gdn.GatedDeltaNetScanBackward(dOutB, bufs[0], bufs[1], bufs[2], bufs[3], bufs[4],
                    gradBufs[0], gradBufs[1], gradBufs[2], gradBufs[3], gradBufs[4],
                    Batch, SeqLen, ModelDim, NumHeads, HeadDim, keyScale, decayFirst);
            }
            Assert.True(scope.Uploads + scope.Downloads == 0,
                "the scan backward crossed the host-device boundary: "
                + string.Join(", ", scope.Events.Select(e => $"{e.Kind} {e.Bytes} B {e.Operation}")));

            string[] names = { "dQ", "dK", "dV", "dAlpha", "dBeta" };
            var results = new float[5][];
            for (int g = 0; g < 5; g++)
            {
                var actual = backend.DownloadBuffer(gradBufs[g]);
                double scale = Math.Max(expected[g].Max(Math.Abs), 1e-6);
                for (int i = 0; i < expected[g].Length; i++)
                    Assert.True(Math.Abs(expected[g][i] - actual[i]) <= 2e-4 * scale,
                        $"{rule} {names[g]}[{i}]: cpu {expected[g][i]}, gpu {actual[i]} (scale {scale})");
                results[g] = actual;
            }
            return results;
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
