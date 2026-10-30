// Copyright (c) AiDotNet. All rights reserved.

using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

/// <summary>
/// A compiled linear layer over a rank-3 input ([batch, seq, in] x transpose([out, in]) + bias) computes what eager
/// does. Found by the AiDotNet model sweep: RNNTransducer's first FullyConnectedLayer on [1, 64, 32] replayed
/// uninitialized memory.
/// </summary>
public sealed class CompiledRank3LinearTests
{
    private static Tensor<float> Random(int[] shape, int seed)
    {
        var rng = new Random(seed);
        int n = 1;
        foreach (var d in shape) n *= d;
        var data = new float[n];
        for (int i = 0; i < n; i++) data[i] = (float)(rng.NextDouble() * 2 - 1);
        return new Tensor<float>(data, shape);
    }

    public static TheoryData<int, int, int, int, int> Shapes => new()
    {
        { 1, 64, 32, 16, 0 }, { 1, 64, 32, 16, 1 }, { 2, 8, 32, 16, 0 }, { 1, 1, 32, 16, 0 }, { 3, 5, 7, 4, 1 },
    };

    [Theory]
    // The matmul packs a frozen B once; B produced by the plan (variant 1) must never be frozen. The 1x64 rows take
    // the packed-B path; 8 rows do not.
    [InlineData(0, 1, 64)]
    [InlineData(1, 1, 64)]
    [InlineData(2, 1, 64)]
    [InlineData(0, 1, 8)]
    [InlineData(0, 1, 16)]
    [InlineData(0, 1, 32)]
    [InlineData(0, 2, 64)]
    [InlineData(1, 1, 8)]
    public void MatMulOfATracedTranspose_ReadsTheTransposeEveryReplay(int variant, int batch, int seq)
    {
        var engine = new CpuEngine();
        var weights = Random(new[] { 16, 32 }, 1);
        var weightsT = engine.TensorTranspose(weights);
        var input = Random(new[] { batch, seq, 32 }, 3);
        Tensor<float> Forward(Tensor<float> x) => variant switch
        {
            0 => engine.TensorMatMul(x, weightsT),                 // matmul with a pre-transposed constant
            1 => engine.TensorMatMul(x, engine.TensorTranspose(weights)),
            _ => engine.ReLU(x),
        };
        var expected = Forward(input).AsSpan().ToArray();
        using var cache = new CompiledModelCache<float>();
        var plan = cache.GetOrCompileInference(input._shape, () => Forward(input));
        plan.SetInputs(new[] { Random(new[] { batch, seq, 32 }, 3) });
        var actual = plan.Execute().AsSpan().ToArray();
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - actual[i]) <= 1e-4f + 1e-4f * Math.Abs(expected[i]),
                $"[{i}] eager {expected[i]} compiled {actual[i]}");
    }

    /// <summary>
    /// B a constant non-contiguous view (a permuted weight, as HippoMemoryCellLayer's A^T): a frozen-weight pack of B
    /// must read it through its strides. Found by the AiDotNet model sweep (HiPPO layer diverging at layer 0).
    /// </summary>
    [Theory]
    [InlineData(64, 8)]
    [InlineData(64, 32)]
    [InlineData(8, 8)]
    [InlineData(256, 16)]
    public void MatMulByAPermutedConstant_CompiledMatchesEager(int rows, int n)
    {
        var engine = new CpuEngine();
        var a = Random(new[] { n, n }, 5);
        var aT = engine.TensorPermute(a, new[] { 1, 0 });
        var input = Random(new[] { rows, n }, 6);
        Tensor<float> Forward(Tensor<float> x) => engine.TensorAdd(x, engine.TensorMatMul(x, aT));
        var expected = Forward(input).AsSpan().ToArray();
        using var cache = new CompiledModelCache<float>();
        var plan = cache.GetOrCompileInference(input, () => Forward(input));
        plan.SetInputs(new[] { input });
        var actual = plan.Execute().AsSpan().ToArray();
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - actual[i]) <= 1e-4f + 1e-4f * Math.Abs(expected[i]),
                $"[{i}] eager {expected[i]} compiled {actual[i]}");
    }

    [Theory]
    [MemberData(nameof(Shapes))]
    public void Rank3Linear_CompiledMatchesEager(int batch, int seq, int inF, int outF, int activation)
    {
        var engine = new CpuEngine();
        var weights = Random(new[] { outF, inF }, 1);
        var biases = Random(new[] { outF }, 2);
        var input = Random(new[] { batch, seq, inF }, 3);
        Tensor<float> Forward(Tensor<float> x)
        {
            var linear = engine.TensorMatMul(x, engine.TensorTranspose(weights));
            var biased = engine.TensorAdd(linear, engine.Reshape(biases, new[] { 1, outF }));
            return activation == 1 ? engine.ReLU(biased) : biased;
        }

        var expected = Forward(input).AsSpan().ToArray();
        using var cache = new CompiledModelCache<float>();
        var plan = cache.GetOrCompileInference(input._shape, () => Forward(input));
        var request = Random(new[] { batch, seq, inF }, 3);
        plan.SetInputs(new[] { request });
        var actual = plan.Execute().AsSpan().ToArray();
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - actual[i]) <= 1e-4f + 1e-4f * Math.Abs(expected[i]),
                $"[{i}] eager {expected[i]} compiled {actual[i]}");
    }
}
