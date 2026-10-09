using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Autodiff;

/// <summary>
/// A float32 matmul backward below SimdGemm's parallel threshold runs its two GEMMs directly - on native BLAS when one
/// is installed, on SimdGemm otherwise. Without BLAS it used to drop to the generic engine fallback (two engine matmuls
/// and a materialized transpose), ~12x slower on an LSTM's per-step [32,64]x[256,64]^T products.
/// </summary>
public class MatMulBackwardSmallShapeTests
{
    private sealed class CountingEngine : CpuEngine
    {
        public int MatMulCalls;

        public override Tensor<T> TensorMatMul<T>(Tensor<T> a, Tensor<T> b)
        {
            MatMulCalls++;
            return base.TensorMatMul(a, b);
        }
    }

    private static Tensor<float> Rand(int[] shape, int seed)
    {
        var rng = RandomHelper.CreateSeededRandom(seed);
        var t = new Tensor<float>(shape);
        for (int i = 0; i < t.Length; i++) t[i] = (float)(rng.NextDouble() - 0.5);
        return t;
    }

    private static void AssertClose(float[] expected, Tensor<float> actual, string name)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - actual[i]) < 1e-4f, $"{name}[{i}]: expected {expected[i]}, got {actual[i]}");
    }

    [Fact]
    public void TransposedBackward_SmallShape_RunsDirectGemmsWithCorrectGradients()
    {
        const int M = 32, K = 64, N = 256; // C[M,N] = A[M,K] . B[N,K]^T, well below the parallel threshold
        var a = Rand(new[] { M, K }, 1);
        var b = Rand(new[] { N, K }, 2);
        var dC = Rand(new[] { M, N }, 3);
        var engine = new CountingEngine();
        var grads = new Dictionary<Tensor<float>, Tensor<float>>();

        BackwardFunctions<float>.MatMulTransposedBackward(
            dC, new[] { a, b }, new Tensor<float>(new[] { M, N }), Array.Empty<object>(), engine, grads);

        Assert.Equal(0, engine.MatMulCalls);
        var gA = new float[M * K];
        var gB = new float[N * K];
        for (int i = 0; i < M; i++)
            for (int j = 0; j < N; j++)
            {
                float g = dC[i * N + j];
                for (int p = 0; p < K; p++)
                {
                    gA[i * K + p] += g * b[j * K + p];
                    gB[j * K + p] += g * a[i * K + p];
                }
            }
        AssertClose(gA, grads[a], "gradA");
        AssertClose(gB, grads[b], "gradB");
    }

    [Fact]
    public void Backward_SmallShape_RunsDirectGemmsWithCorrectGradients()
    {
        const int M = 32, K = 64, N = 256; // C[M,N] = A[M,K] . B[K,N]
        var a = Rand(new[] { M, K }, 4);
        var b = Rand(new[] { K, N }, 5);
        var dC = Rand(new[] { M, N }, 6);
        var engine = new CountingEngine();
        var grads = new Dictionary<Tensor<float>, Tensor<float>>();

        BackwardFunctions<float>.MatMulBackward(
            dC, new[] { a, b }, new Tensor<float>(new[] { M, N }), Array.Empty<object>(), engine, grads);

        Assert.Equal(0, engine.MatMulCalls);
        var gA = new float[M * K];
        var gB = new float[K * N];
        for (int i = 0; i < M; i++)
            for (int j = 0; j < N; j++)
            {
                float g = dC[i * N + j];
                for (int p = 0; p < K; p++)
                {
                    gA[i * K + p] += g * b[p * N + j];
                    gB[p * N + j] += g * a[i * K + p];
                }
            }
        AssertClose(gA, grads[a], "gradA");
        AssertClose(gB, grads[b], "gradB");
    }
}
