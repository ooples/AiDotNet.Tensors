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

/// <summary>
/// MatMulTransposedBackward (and the fused linear-activation backwards) took a CPU fast path whenever no tape was
/// recording -- i.e. always during backward -- unless a resident CUDA-graph step was active. On the GPU engine in
/// eager training that path downloaded both operands and the upstream gradient and ran host SimdGemm. They are now
/// CPU-engine-only (like MatMulBackward), so on the GPU engine the gradients are computed on the device. The
/// residency assertion is what distinguishes the paths: both produce numerically equal gradients.
/// </summary>
[Collection("VulkanGlobalState")]
public sealed class MatMulTransposedTapeGpuTests : IClassFixture<DirectGpuTensorEngineTestFixture>
{
    private readonly DirectGpuTensorEngineTestFixture _fixture;

    public MatMulTransposedTapeGpuTests(DirectGpuTensorEngineTestFixture fixture) => _fixture = fixture;

    [SkippableFact]
    public void Backward_StaysOnDevice_AndMatchesCpu()
    {
        Skip.IfNot(_fixture.IsAvailable, "No GPU device.");
        var gpu = _fixture.Engine!;
        var cpu = new CpuEngine();
        var rng = new Random(9);
        Tensor<float> R(params int[] s)
        {
            var t = new Tensor<float>(s);
            for (int i = 0; i < t.Length; i++) t[i] = (float)(rng.NextDouble() - 0.5);
            return t;
        }

        // M*K*N = 512*256*1024 = 134M MACs: above the fast path's machine-dependent SimdGemm.ParallelWorkThreshold,
        // so the old CPU path would engage (a smaller shape fell under it and made this test unable to fail).
        var a = R(512, 256);
        var b = R(1024, 256);
        var r = R(512, 1024);
        Dictionary<Tensor<float>, Tensor<float>> Grads(IEngine e)
        {
            using var tape = new GradientTape<float>();
            var loss = e.ReduceSum(e.TensorMultiply(e.TensorMatMulTransposed(a, b), r), [0, 1], keepDims: false);
            return tape.ComputeGradients(loss, [a, b]);
        }

        var expected = Grads(cpu);
        var actual = Grads(gpu);
        foreach (var (t, name) in new[] { (a, "dA"), (b, "dB") })
        {
            var g = actual[t];
            Assert.True(g.IsGpuResident || g.HasPendingGpuData, $"{name} was computed on the host (CPU fast path)");
            var e = expected[t].ToArray();
            var x = g.ToArray();
            double scale = Math.Max(e.Max(Math.Abs), 1e-6);
            for (int i = 0; i < e.Length; i++)
                Assert.True(Math.Abs(e[i] - x[i]) <= 1e-4 * scale, $"{name}[{i}]: cpu {e[i]}, gpu {x[i]}");
        }
    }
}
