// Copyright (c) CrowdTrainer LLC. All rights reserved.

using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// Verifies that <see cref="DirectGpuTensorEngine.GatedDeltaNetScanForward{T}"/> stays 100% on-device under an active
/// <see cref="GradientTape{T}"/> (no CPU fallback or host readback) and that its fused GPU BPTT backward
/// (<c>gated_delta_scan_recompute</c> + <c>gated_delta_scan_backward</c>) matches <see cref="CpuEngine"/> across all
/// five gradients (<c>dQ</c>, <c>dK</c>, <c>dV</c>, <c>dAlpha</c>, <c>dBeta</c>), as well as verifying that the new
/// <c>float[]</c> fast path in <see cref="CpuEngine"/> matches finite differences.
/// </summary>
[Collection("VulkanGlobalState")]
public sealed class GatedDeltaNetScanTapeGpuTests : IClassFixture<DirectGpuTensorEngineTestFixture>
{
    private readonly DirectGpuTensorEngineTestFixture _fixture;

    public GatedDeltaNetScanTapeGpuTests(DirectGpuTensorEngineTestFixture fixture) => _fixture = fixture;

    private static Tensor<float> Rand(int[] shape, int seed, bool gate = false)
    {
        var rng = new Random(seed);
        var t = new Tensor<float>(shape);
        for (int i = 0; i < t.Length; i++)
            t[i] = gate ? (float)(0.2 + 0.75 * rng.NextDouble()) : (float)(rng.NextDouble() - 0.5);
        return t;
    }

    private static (Tensor<float> Out, Dictionary<Tensor<float>, Tensor<float>> Grads) Run(
        IEngine engine, Tensor<float>[] inputs, Tensor<float> r, int numHeads, bool expectOnDevice = false)
    {
        using var tape = new GradientTape<float>();
        var output = engine.GatedDeltaNetScanForward(inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], numHeads);
        if (expectOnDevice)
            Assert.True(output.IsGpuResident || output.HasPendingGpuData,
                "GatedDeltaNetScanForward ran on the host (CPU fallback) under an active tape");
        var loss = engine.ReduceSum(engine.TensorMultiply(output, r), [0, 1, 2], keepDims: false);
        var grads = tape.ComputeGradients(loss, inputs);
        if (expectOnDevice)
        {
            string[] names = ["dQ", "dK", "dV", "dAlpha", "dBeta"];
            for (int i = 0; i < inputs.Length; i++)
                if (grads.TryGetValue(inputs[i], out var g))
                    Assert.True(g.IsGpuResident || g.HasPendingGpuData,
                        $"{names[i]} was computed on the host instead of IGatedDeltaNetBackwardBackend");
        }
        return (output, grads);
    }

    private static void AssertClose(Tensor<float> expected, Tensor<float> actual, string name)
    {
        var e = expected.ToArray();
        var a = actual.ToArray();
        Assert.Equal(e.Length, a.Length);
        double scale = Math.Max(e.Max(Math.Abs), 1e-6);
        for (int i = 0; i < e.Length; i++)
            Assert.True(Math.Abs(e[i] - a[i]) <= 2e-4 * scale, $"{name}[{i}]: cpu {e[i]}, gpu {a[i]} (scale {scale})");
    }

    [SkippableTheory]
    [InlineData(2, 16, 64, 4)]    // headDim 16
    [InlineData(8, 64, 256, 8)]   // Hybrid 3:1 GDN:MLA LM training shape: batch=8, seq=64, 8 heads x 32 features
    public void TapeForwardAndGradients_StayOnGpu_AndMatchCpu(int batch, int seqLen, int modelDim, int numHeads)
    {
        Skip.IfNot(_fixture.IsAvailable, "No GPU device.");
        var gpu = _fixture.Engine!;
        var cpu = new CpuEngine();
        int[] shape = [batch, seqLen, modelDim];
        int[] gateShape = [batch, seqLen, numHeads];
        var inputs = new[]
        {
            Rand(shape, 11),
            Rand(shape, 12),
            Rand(shape, 13),
            Rand(gateShape, 14, gate: true),
            Rand(gateShape, 15, gate: true)
        };
        var r = Rand(shape, 16);

        var (cpuOut, cpuGrads) = Run(cpu, inputs, r, numHeads);
        var (gpuOut, gpuGrads) = Run(gpu, inputs, r, numHeads, expectOnDevice: true);

        AssertClose(cpuOut, gpuOut, "output");
        string[] names = ["dQ", "dK", "dV", "dAlpha", "dBeta"];
        for (int i = 0; i < 5; i++)
        {
            Assert.True(gpuGrads.ContainsKey(inputs[i]), $"{names[i]} missing from the GPU tape");
            AssertClose(cpuGrads[inputs[i]], gpuGrads[inputs[i]], names[i]);
        }
    }

    [Fact]
    public void CpuFloatFastPath_MatchesDoubleReference()
    {
        var cpu = new CpuEngine();
        int batch = 2, seqLen = 10, modelDim = 24, numHeads = 3;
        int[] shape = [batch, seqLen, modelDim];
        int[] gateShape = [batch, seqLen, numHeads];
        var inputsF = new[]
        {
            Rand(shape, 21),
            Rand(shape, 22),
            Rand(shape, 23),
            Rand(gateShape, 24, gate: true),
            Rand(gateShape, 25, gate: true)
        };
        var rF = Rand(shape, 26);

        Tensor<double> ToDouble(Tensor<float> f)
        {
            var d = new Tensor<double>(f.Shape.ToArray());
            var fa = f.ToArray();
            for (int i = 0; i < fa.Length; i++) d[i] = fa[i];
            return d;
        }

        var inputsD = inputsF.Select(ToDouble).ToArray();
        var rD = ToDouble(rF);

        var (outF, gradsF) = Run(cpu, inputsF, rF, numHeads);
        using var tapeD = new GradientTape<double>();
        var outD = cpu.GatedDeltaNetScanForward(inputsD[0], inputsD[1], inputsD[2], inputsD[3], inputsD[4], numHeads);
        var lossD = cpu.ReduceSum(cpu.TensorMultiply(outD, rD), [0, 1, 2], keepDims: false);
        var gradsD = tapeD.ComputeGradients(lossD, inputsD);

        var outFa = outF.ToArray();
        var outDa = outD.ToArray();
        for (int i = 0; i < outFa.Length; i++)
            Assert.True(Math.Abs(outFa[i] - outDa[i]) <= 1e-4, $"out[{i}]: float {outFa[i]} vs double {outDa[i]}");

        string[] names = ["dQ", "dK", "dV", "dAlpha", "dBeta"];
        for (int i = 0; i < 5; i++)
        {
            var gFa = gradsF[inputsF[i]].ToArray();
            var gDa = gradsD[inputsD[i]].ToArray();
            double scale = Math.Max(gDa.Max(Math.Abs), 1e-6);
            for (int j = 0; j < gFa.Length; j++)
                Assert.True(Math.Abs(gFa[j] - gDa[j]) <= 2e-4 * scale, $"{names[i]}[{j}]: float {gFa[j]} vs double {gDa[j]}");
        }
    }
}
