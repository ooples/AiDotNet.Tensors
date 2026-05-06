// Copyright (c) AiDotNet. All rights reserved.

using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

public class FusedIssue301GraphFusionTests
{
    private static Tensor<float> RandomTensor(int[] shape, int seed, double scale = 1.0)
    {
        var rng = new Random(seed);
        int n = 1;
        foreach (var d in shape) n *= d;
        var data = new float[n];
        for (int i = 0; i < n; i++)
            data[i] = (float)((rng.NextDouble() * 2.0 - 1.0) * scale);
        return new Tensor<float>(data, shape);
    }

    [Fact]
    public void LoRAChain_CompilesToSingleFusedStep()
    {
        var engine = new CpuEngine();
        var input = RandomTensor(new[] { 4, 8 }, seed: 1);
        var baseOut = RandomTensor(new[] { 4, 6 }, seed: 2);
        var loraA = RandomTensor(new[] { 8, 2 }, seed: 3, scale: 0.1);
        var loraB = RandomTensor(new[] { 2, 6 }, seed: 4, scale: 0.1);
        const float scaling = 0.5f;
        var expected = new Tensor<float>(new[] { 4, 6 });
        CpuFusedOperations.FusedLoRAForward(input, baseOut, loraA, loraB, scaling, expected);

        ICompiledPlan<float> plan;
        using (var scope = GraphMode.Enable())
        {
            var hidden = engine.TensorMatMul(input, loraA);
            var delta = engine.TensorMatMul(hidden, loraB);
            var scaled = engine.TensorMultiplyScalar(delta, scaling);
            var output = engine.TensorAdd(baseOut, scaled);
            plan = scope.CompileInference<float>(output);
        }

        try
        {
            Assert.Equal(1, plan.StepCount);
            AssertClose(expected, plan.Execute(), tolerance: 1e-4f);
        }
        finally
        {
            plan.Dispose();
        }
    }

    [Fact]
    public void DDIMChain_CompilesToSingleFusedStep()
    {
        var engine = new CpuEngine();
        var xT = RandomTensor(new[] { 32 }, seed: 7);
        var eps = RandomTensor(new[] { 32 }, seed: 8, scale: 0.25);
        const float alphaBarT = 0.64f;
        const float alphaBarTMinus1 = 0.81f;
        float sqrtAt = MathF.Sqrt(alphaBarT);
        float sqrtOneMinusAt = MathF.Sqrt(1f - alphaBarT);
        float sqrtAtMinus1 = MathF.Sqrt(alphaBarTMinus1);
        float sqrtOneMinusAtMinus1 = MathF.Sqrt(1f - alphaBarTMinus1);

        var expected = new Tensor<float>(new[] { 32 });
        CpuFusedOperations.FusedDDIMStep(xT, eps, alphaBarT, alphaBarTMinus1, expected);

        ICompiledPlan<float> plan;
        using (var scope = GraphMode.Enable())
        {
            var noise = engine.TensorMultiplyScalar(eps, sqrtOneMinusAt);
            var x0Numerator = engine.TensorSubtract(xT, noise);
            var x0Pred = engine.TensorDivideScalar(x0Numerator, sqrtAt);
            var prevX0 = engine.TensorMultiplyScalar(x0Pred, sqrtAtMinus1);
            var prevNoise = engine.TensorMultiplyScalar(eps, sqrtOneMinusAtMinus1);
            var output = engine.TensorAdd(prevX0, prevNoise);
            plan = scope.CompileInference<float>(output);
        }

        try
        {
            Assert.Equal(1, plan.StepCount);
            AssertClose(expected, plan.Execute(), tolerance: 1e-5f);
        }
        finally
        {
            plan.Dispose();
        }
    }

    [Fact]
    public void SparseLinearChain_CompilesToSingleFusedStep()
    {
        var engine = new CpuEngine();
        var input = RandomTensor(new[] { 3, 8 }, seed: 11);
        var bias = RandomTensor(new[] { 5 }, seed: 12, scale: 0.05);
        var weight = new Tensor<float>(new[]
        {
            0.10f, 0f, 0f, 0f, -0.07f,
            0f, 0f, 0.04f, 0f, 0f,
            0f, -0.03f, 0f, 0f, 0f,
            0.06f, 0f, 0f, 0f, 0f,
            0f, 0f, 0f, 0.08f, 0f,
            0f, 0f, 0f, 0f, 0f,
            0f, 0.02f, 0f, 0f, 0f,
            0f, 0f, -0.05f, 0f, 0f,
        }, new[] { 8, 5 });

        var eager = engine.ReLU(engine.TensorBroadcastAdd(engine.TensorMatMul(input, weight), bias));

        ICompiledPlan<float> plan;
        using (var scope = GraphMode.Enable())
        {
            var matmul = engine.TensorMatMul(input, weight);
            var biased = engine.TensorBroadcastAdd(matmul, bias);
            var output = engine.ReLU(biased);
            plan = scope.CompileInference<float>(output);
        }

        try
        {
            Assert.Equal(1, plan.StepCount);
            AssertClose(eager, plan.Execute(), tolerance: 1e-5f);
        }
        finally
        {
            plan.Dispose();
        }
    }

    private static void AssertClose(Tensor<float> expected, Tensor<float> actual, float tolerance)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            float e = expected.GetFlat(i);
            float a = actual.GetFlat(i);
            float diff = MathF.Abs(e - a);
            float allowed = tolerance * MathF.Max(1f, MathF.Abs(e));
            Assert.True(diff <= allowed, $"Mismatch at {i}: expected={e}, actual={a}, diff={diff}, allowed={allowed}.");
        }
    }
}
