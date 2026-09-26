// Copyright (c) AiDotNet. All rights reserved.

using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// Under an active GradientTape the GPU engine used to route GlaScanForward (and therefore its BPTT backward) to the
/// managed CPU recurrence, so every training step of a GLA / Born-linear language model ran its scan on the CPU with
/// device-host copies around it. The scan now stays on the device and records a tape node whose backward is
/// backend.GlaScanBackward. These tests pin the forward and all four gradients (dQ, dK, dV, dGate) to the CPU engine.
/// </summary>
[Collection("VulkanGlobalState")]
public sealed class GlaScanTapeGpuTests : IClassFixture<DirectGpuTensorEngineTestFixture>
{
    private readonly DirectGpuTensorEngineTestFixture _fixture;

    public GlaScanTapeGpuTests(DirectGpuTensorEngineTestFixture fixture) => _fixture = fixture;

    private static Tensor<float> Rand(int[] shape, int seed, bool gate = false)
    {
        var rng = new Random(seed);
        var t = new Tensor<float>(shape);
        for (int i = 0; i < t.Length; i++)
            t[i] = gate ? (float)(0.5 + 0.5 * rng.NextDouble()) : (float)(rng.NextDouble() - 0.5);
        return t;
    }

    private static (Tensor<float> Out, Dictionary<Tensor<float>, Tensor<float>> Grads) Run(
        IEngine engine, Tensor<float>[] inputs, Tensor<float> r, int numHeads)
    {
        using var tape = new GradientTape<float>();
        var output = engine.GlaScanForward(inputs[0], inputs[1], inputs[2], inputs[3], numHeads);
        var loss = engine.ReduceSum(engine.TensorMultiply(output, r), [0, 1, 2], keepDims: false);
        return (output, tape.ComputeGradients(loss, inputs));
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
    [InlineData(2, 16, 800, 8)]   // the Born-linear LM shape: 8 heads x 100 features
    public void TapeForwardAndGradients_MatchCpu(int batch, int seqLen, int modelDim, int numHeads)
    {
        Skip.IfNot(_fixture.IsAvailable, "No GPU device.");
        var gpu = _fixture.Engine!;
        var cpu = new CpuEngine();
        int[] shape = [batch, seqLen, modelDim];
        var inputs = new[] { Rand(shape, 1), Rand(shape, 2), Rand(shape, 3), Rand([batch, seqLen, numHeads], 4, gate: true) };
        var r = Rand(shape, 5);

        var (cpuOut, cpuGrads) = Run(cpu, inputs, r, numHeads);
        var (gpuOut, gpuGrads) = Run(gpu, inputs, r, numHeads);

        AssertClose(cpuOut, gpuOut, "output");
        string[] names = ["dQ", "dK", "dV", "dGate"];
        for (int i = 0; i < 4; i++)
        {
            Assert.True(gpuGrads.ContainsKey(inputs[i]), $"{names[i]} missing from the GPU tape");
            AssertClose(cpuGrads[inputs[i]], gpuGrads[inputs[i]], names[i]);
        }
    }
}
