using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// <see cref="IEngine.MultiHeadAttentionCore{T}"/> on the GPU engine (composed from device primitives) against the
/// CPU engine's fused kernel: forward and tape gradients, with and without causal masking.
/// </summary>
[Collection("VulkanGlobalState")]
public sealed class MultiHeadAttentionCoreGpuTests : IClassFixture<DirectGpuTensorEngineTestFixture>
{
    private readonly DirectGpuTensorEngineTestFixture _fixture;

    public MultiHeadAttentionCoreGpuTests(DirectGpuTensorEngineTestFixture fixture) => _fixture = fixture;

    [SkippableTheory]
    [InlineData(false)]
    [InlineData(true)]
    public void Gpu_MatchesCpu_ForwardAndGradients(bool causal)
    {
        Skip.IfNot(_fixture.IsAvailable, "No GPU device.");
        const int batch = 2, seq = 11, heads = 3, dim = 8;
        var (cpuOut, cpuGrads) = Run(new CpuEngine(), batch, seq, heads, dim, causal);
        var (gpuOut, gpuGrads) = Run(_fixture.Engine!, batch, seq, heads, dim, causal);
        AssertClose(cpuOut, gpuOut, "output");
        for (int i = 0; i < 3; i++) AssertClose(cpuGrads[i], gpuGrads[i], $"grad[{i}]");
    }

    private static (float[] Output, float[][] Grads) Run(IEngine engine, int batch, int seq, int heads, int dim, bool causal)
    {
        var rng = new Random(9);
        Tensor<float> Make()
        {
            var a = new float[batch * seq * heads * dim];
            for (int i = 0; i < a.Length; i++) a[i] = (float)(rng.NextDouble() * 2 - 1);
            return new Tensor<float>(a, new[] { batch, seq, heads * dim });
        }
        var q = Make(); var k = Make(); var v = Make(); var g = Make();
        var previous = AiDotNetEngine.Current;
        AiDotNetEngine.Current = engine;
        try
        {
            using var tape = new GradientTape<float>();
            var output = engine.MultiHeadAttentionCore(q, k, v, heads, causal: causal);
            var loss = engine.ReduceSum(engine.TensorMultiply(output, g), null);
            var grads = tape.ComputeGradients(loss, new[] { q, k, v });
            return (output.ToArray(), new[] { grads[q].ToArray(), grads[k].ToArray(), grads[v].ToArray() });
        }
        finally
        {
            AiDotNetEngine.Current = previous;
        }
    }

    private static void AssertClose(float[] expected, float[] actual, string what)
    {
        Assert.Equal(expected.Length, actual.Length);
        double maxErr = 0;
        for (int i = 0; i < expected.Length; i++) maxErr = Math.Max(maxErr, Math.Abs(expected[i] - actual[i]));
        Assert.True(maxErr < 1e-4, $"{what}: GPU vs CPU max |error| {maxErr:G4}");
    }
}
