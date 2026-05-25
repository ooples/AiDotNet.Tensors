using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// Tests for <see cref="CpuEngine.MlpForward{T}"/> — the fused multi-layer
/// perceptron forward primitive added for the AIsEval MLP inference gap
/// (issue #436 P1). Verifies the primitive produces the same output as both a
/// hand-rolled MatMul + bias + activation reference and the chained
/// <see cref="CpuEngine.FusedLinear{T}"/> path it composes.
/// </summary>
public class MlpForwardTests
{
    private readonly CpuEngine _engine = new();

    [Fact]
    public void MlpForward_MatchesNaiveReference_Float()
    {
        // 3-layer MLP, ReLU hidden, no output activation — the AIsEval shape
        // family (Dense→Dense→Dense classification head), shrunk for the test.
        var rng = new Random(2026);
        int b = 4;
        int[] dims = { 12, 8, 5, 3 };

        var input = MakeRandom(rng, b, dims[0]);
        var weights = new List<Tensor<float>>();
        var biases = new List<Tensor<float>?>();
        for (int i = 0; i + 1 < dims.Length; i++)
        {
            weights.Add(MakeRandom(rng, dims[i], dims[i + 1]));  // [in, out]
            biases.Add(MakeRandom(rng, dims[i + 1]));
        }

        var fused = _engine.MlpForward(input, weights, biases,
            FusedActivationType.ReLU, FusedActivationType.None);

        var reference = NaiveMlp(input, weights, biases, b, dims);

        AssertClose(fused, reference, 1e-4f);
    }

    [Fact]
    public void MlpForward_MatchesChainedFusedLinear_Float()
    {
        // The primitive is documented as equivalent to chaining FusedLinear
        // per layer — assert that contract directly, including a non-None
        // output activation (Sigmoid) to exercise the last-layer branch.
        var rng = new Random(7);
        int b = 3;
        int[] dims = { 10, 6, 4 };

        var input = MakeRandom(rng, b, dims[0]);
        var weights = new List<Tensor<float>>();
        var biases = new List<Tensor<float>?>();
        for (int i = 0; i + 1 < dims.Length; i++)
        {
            weights.Add(MakeRandom(rng, dims[i], dims[i + 1]));
            biases.Add(i == 0 ? MakeRandom(rng, dims[i + 1]) : null); // mix bias / no-bias
        }

        var fused = _engine.MlpForward(input, weights, biases,
            FusedActivationType.ReLU, FusedActivationType.Sigmoid);

        var x = input;
        for (int i = 0; i + 1 < dims.Length; i++)
        {
            var act = i == dims.Length - 2 ? FusedActivationType.Sigmoid : FusedActivationType.ReLU;
            x = _engine.FusedLinear(x, weights[i], biases[i], act);
        }

        AssertClose(fused, x, 1e-6f);
    }

    [Fact]
    public void MlpForward_MatchesNaiveReference_Double()
    {
        // Generic-T path (double) must match the same naive reference.
        var rng = new Random(99);
        int b = 2;
        int[] dims = { 7, 5, 3 };

        var input = MakeRandomD(rng, b, dims[0]);
        var weights = new List<Tensor<double>>();
        var biases = new List<Tensor<double>?>();
        for (int i = 0; i + 1 < dims.Length; i++)
        {
            weights.Add(MakeRandomD(rng, dims[i], dims[i + 1]));
            biases.Add(MakeRandomD(rng, dims[i + 1]));
        }

        var fused = _engine.MlpForward(input, weights, biases,
            FusedActivationType.ReLU, FusedActivationType.None);

        var reference = NaiveMlpD(input, weights, biases, b, dims);

        var sf = fused.AsSpan();
        var sr = reference.AsSpan();
        Assert.Equal(sr.Length, sf.Length);
        for (int i = 0; i < sf.Length; i++)
            Assert.True(Math.Abs(sf[i] - sr[i]) < 1e-9,
                $"Mismatch at {i}: fused={sf[i]}, ref={sr[i]}");
    }

    [Fact]
    public void MlpForward_UnderGraphMode_Throws()
    {
        var rng = new Random(1);
        var input = MakeRandom(rng, 2, 4);
        var weights = new List<Tensor<float>> { MakeRandom(rng, 4, 3) };
        var biases = new List<Tensor<float>?> { MakeRandom(rng, 3) };

        using (GraphMode.Enable())
        {
            Assert.Throws<InvalidOperationException>(() =>
                _engine.MlpForward(input, weights, biases,
                    FusedActivationType.ReLU, FusedActivationType.None));
        }
    }

    [Fact]
    public void MlpForward_MismatchedBiasCount_Throws()
    {
        var rng = new Random(1);
        var input = MakeRandom(rng, 2, 4);
        var weights = new List<Tensor<float>> { MakeRandom(rng, 4, 3), MakeRandom(rng, 3, 2) };
        var biases = new List<Tensor<float>?> { MakeRandom(rng, 3) }; // one short

        Assert.Throws<ArgumentException>(() =>
            _engine.MlpForward(input, weights, biases,
                FusedActivationType.ReLU, FusedActivationType.None));
    }

    [Fact]
    public void MlpForward_EmptyWeights_Throws()
    {
        var rng = new Random(1);
        var input = MakeRandom(rng, 2, 4);
        Assert.Throws<ArgumentException>(() =>
            _engine.MlpForward(input, new List<Tensor<float>>(), new List<Tensor<float>?>(),
                FusedActivationType.ReLU, FusedActivationType.None));
    }

    // ----------------- Helpers -----------------

    private static Tensor<float> MakeRandom(Random rng, params int[] shape)
    {
        var t = Tensor<float>.CreateZeros(shape);
        var span = t.AsWritableSpan();
        for (int i = 0; i < span.Length; i++)
            span[i] = (float)(rng.NextDouble() * 0.6 - 0.3);
        return t;
    }

    private static Tensor<double> MakeRandomD(Random rng, params int[] shape)
    {
        var t = Tensor<double>.CreateZeros(shape);
        var span = t.AsWritableSpan();
        for (int i = 0; i < span.Length; i++)
            span[i] = rng.NextDouble() * 0.6 - 0.3;
        return t;
    }

    private static void AssertClose(Tensor<float> a, Tensor<float> b, float atol)
    {
        var sa = a.AsSpan();
        var sb = b.AsSpan();
        Assert.Equal(sb.Length, sa.Length);
        for (int i = 0; i < sa.Length; i++)
            Assert.True(MathF.Abs(sa[i] - sb[i]) < atol,
                $"Mismatch at {i}: fused={sa[i]:G6}, ref={sb[i]:G6}");
    }

    // Hand-rolled MLP: per layer out[m,o] = relu(sum_k x[m,k]*W[k,o] + b[o]),
    // with the last layer left linear (None).
    private static Tensor<float> NaiveMlp(
        Tensor<float> input, List<Tensor<float>> weights, List<Tensor<float>?> biases, int b, int[] dims)
    {
        var x = input.AsSpan().ToArray();
        int rows = b;
        for (int layer = 0; layer + 1 < dims.Length; layer++)
        {
            int din = dims[layer], dout = dims[layer + 1];
            var w = weights[layer].AsSpan();
            var bias = biases[layer];
            var biasSpan = bias is null ? default : bias.AsSpan();
            var y = new float[rows * dout];
            for (int m = 0; m < rows; m++)
                for (int o = 0; o < dout; o++)
                {
                    float acc = bias is null ? 0f : biasSpan[o];
                    for (int kk = 0; kk < din; kk++)
                        acc += x[m * din + kk] * w[kk * dout + o];
                    bool lastLayer = layer + 2 == dims.Length;
                    y[m * dout + o] = lastLayer ? acc : MathF.Max(0f, acc);
                }
            x = y;
        }
        var outT = Tensor<float>.CreateZeros(new[] { rows, dims[^1] });
        x.AsSpan().CopyTo(outT.AsWritableSpan());
        return outT;
    }

    private static Tensor<double> NaiveMlpD(
        Tensor<double> input, List<Tensor<double>> weights, List<Tensor<double>?> biases, int b, int[] dims)
    {
        var x = input.AsSpan().ToArray();
        int rows = b;
        for (int layer = 0; layer + 1 < dims.Length; layer++)
        {
            int din = dims[layer], dout = dims[layer + 1];
            var w = weights[layer].AsSpan();
            var bias = biases[layer];
            var biasSpan = bias is null ? default : bias.AsSpan();
            var y = new double[rows * dout];
            for (int m = 0; m < rows; m++)
                for (int o = 0; o < dout; o++)
                {
                    double acc = bias is null ? 0.0 : biasSpan[o];
                    for (int kk = 0; kk < din; kk++)
                        acc += x[m * din + kk] * w[kk * dout + o];
                    bool lastLayer = layer + 2 == dims.Length;
                    y[m * dout + o] = lastLayer ? acc : Math.Max(0.0, acc);
                }
            x = y;
        }
        var outT = Tensor<double>.CreateZeros(new[] { rows, dims[^1] });
        x.AsSpan().CopyTo(outT.AsWritableSpan());
        return outT;
    }
}
