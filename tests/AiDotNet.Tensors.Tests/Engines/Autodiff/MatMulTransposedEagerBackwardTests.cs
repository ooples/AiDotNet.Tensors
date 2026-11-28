using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines.Autodiff;

// The eager (tape) backward of C = A·Bᵀ: gradients against a double-precision reference on both dB kernels (more
// rows than outputs, and not), and a steady-state allocation bound -- its two gradient arrays go back to the pool once
// they have been added into the accumulated gradients, so repeated steps allocate no operand-sized array.
public sealed class MatMulTransposedEagerBackwardTests
{
    private readonly ITestOutputHelper _output;
    public MatMulTransposedEagerBackwardTests(ITestOutputHelper output) { _output = output; }

    private static Tensor<float> Random(int[] shape, int seed)
    {
        var rng = new Random(seed);
        var t = new Tensor<float>(shape);
        for (int i = 0; i < t.Length; i++) t[i] = (float)(rng.NextDouble() * 2 - 1);
        return t;
    }

    [Theory]
    [InlineData(96, 80, 16)]   // M > N: dB through the transposed-copy product
    [InlineData(16, 80, 96)]   // M <= N: dB through the transposed-A GEMM
    [InlineData(1, 7, 5)]
    public void Gradients_MatchReference(int m, int k, int n)
    {
        var e = new CpuEngine();
        var a = Random(new[] { m, k }, 1);
        var b = Random(new[] { n, k }, 2);
        var g = Random(new[] { m, n }, 3);
        Dictionary<Tensor<float>, Tensor<float>> grads;
        using (var tape = new GradientTape<float>())
        {
            var c = e.TensorMatMulTransposed(a, b);
            var loss = e.ReduceSum(e.TensorMultiply(c, g), null);
            grads = tape.ComputeGradients(loss, new[] { a, b });
        }
        var dA = grads[a];
        var dB = grads[b];
        for (int i = 0; i < m; i++)
            for (int j = 0; j < k; j++)
            {
                double acc = 0;
                for (int o = 0; o < n; o++) acc += (double)g[i * n + o] * b[o * k + j];
                Assert.True(Math.Abs(acc - dA[i * k + j]) <= 1e-4 * (1 + Math.Abs(acc)), $"dA[{i},{j}]: {acc} vs {dA[i * k + j]}");
            }
        for (int o = 0; o < n; o++)
            for (int j = 0; j < k; j++)
            {
                double acc = 0;
                for (int i = 0; i < m; i++) acc += (double)g[i * n + o] * a[i * k + j];
                Assert.True(Math.Abs(acc - dB[o * k + j]) <= 1e-4 * (1 + Math.Abs(acc)), $"dB[{o},{j}]: {acc} vs {dB[o * k + j]}");
            }
    }

    // GC.GetAllocatedBytesForCurrentThread is .NET Core / .NET 5+ only.
#if NET5_0_OR_GREATER
    [Fact]
    public void BackwardKernel_SteadyState_AllocatesNoOperandSizedArray()
    {
        const int m = 512, k = 256, n = 128;
        var e = new CpuEngine();
        var a = Random(new[] { m, k }, 4);
        var b = Random(new[] { n, k }, 5);
        var dC = Random(new[] { m, n }, 6);
        var inputs = new[] { a, b };
        var output = e.TensorMatMulTransposed(a, b);
        var grads = new Dictionary<Tensor<float>, Tensor<float>>(AiDotNet.Tensors.Engines.Autodiff.ReferenceEqualityComparer<Tensor<float>>.Instance);
        // The first call donates its arrays to the accumulator; every later call adds into it and hands its own back.
        for (int i = 0; i < 16; i++)
            BackwardFunctions<float>.MatMulTransposedBackward(dC, inputs, output, Array.Empty<object>(), e, grads);
        const int steps = 2000;
        long before = GC.GetAllocatedBytesForCurrentThread();
        for (int i = 0; i < steps; i++)
            BackwardFunctions<float>.MatMulTransposedBackward(dC, inputs, output, Array.Empty<object>(), e, grads);
        long perCall = (GC.GetAllocatedBytesForCurrentThread() - before) / steps;
        _output.WriteLine($"MatMulTransposedBackward: {perCall} B/call");
        long smallestOperand = (long)n * k * sizeof(float);
        Assert.True(perCall < smallestOperand / 4, $"{perCall} B per backward call; dB alone is {smallestOperand} B");
    }
#endif
}
