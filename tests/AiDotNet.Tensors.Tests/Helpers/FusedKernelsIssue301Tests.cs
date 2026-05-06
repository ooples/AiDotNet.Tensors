// Copyright (c) AiDotNet. All rights reserved.
// Issue #301 — numerical-equivalence tests for the new fused kernels.

using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Helpers;

public class FusedKernelsIssue301Tests
{
    private static Tensor<float> RandomTensor(int[] shape, int seed, double scale = 1.0)
    {
        var rng = new Random(seed);
        int n = 1;
        foreach (var d in shape) n *= d;
        var data = new float[n];
        for (int i = 0; i < n; i++) data[i] = (float)((rng.NextDouble() * 2.0 - 1.0) * scale);
        return new Tensor<float>(data, shape);
    }

    // ─────────────────────────────────────────────────────────────────
    // FusedLoRAForward
    // ─────────────────────────────────────────────────────────────────

    [Theory]
    [InlineData(1, 64, 4, 32)]
    [InlineData(8, 256, 8, 128)]
    [InlineData(32, 512, 16, 512)]
    public void FusedLoRAForward_MatchesDecomposed(int batch, int inFeat, int rank, int outFeat)
    {
        var input = RandomTensor(new[] { batch, inFeat }, seed: 1);
        var baseOut = RandomTensor(new[] { batch, outFeat }, seed: 2);
        var loraA = RandomTensor(new[] { inFeat, rank }, seed: 3, scale: 0.1);
        var loraB = RandomTensor(new[] { rank, outFeat }, seed: 4, scale: 0.1);
        float scaling = 0.5f;
        var output = new Tensor<float>(new[] { batch, outFeat });

        CpuFusedOperations.FusedLoRAForward(input, baseOut, loraA, loraB, scaling, output);

        // Reference: the decomposed sequence the kernel replaces.
        var engine = new CpuEngine();
        var intermed = engine.TensorMatMul(input, loraA);          // [batch, rank]
        var delta = engine.TensorMatMul(intermed, loraB);           // [batch, out]
        // Expected[b, j] = baseOut[b, j] + scaling * delta[b, j].
        var expected = new float[batch * outFeat];
        var baseSpan = baseOut.AsSpan();
        var deltaSpan = delta.AsSpan();
        for (int i = 0; i < expected.Length; i++)
            expected[i] = baseSpan[i] + scaling * deltaSpan[i];

        var got = output.AsSpan();
        for (int i = 0; i < expected.Length; i++)
        {
            // Tolerance: 1e-4 relative across the dense matmul chain.
            float diff = Math.Abs(got[i] - expected[i]);
            float scale = 1e-4f * Math.Max(1f, Math.Abs(expected[i]));
            Assert.True(diff <= scale,
                $"Mismatch at idx {i}: got={got[i]}, expected={expected[i]}, diff={diff}.");
        }
    }

    [Fact]
    public void FusedLoRAForward_RejectsShapeMismatch()
    {
        var input = RandomTensor(new[] { 2, 16 }, seed: 1);
        var baseOut = RandomTensor(new[] { 2, 8 }, seed: 2);
        var loraA = RandomTensor(new[] { 16, 4 }, seed: 3);
        var loraB = RandomTensor(new[] { 99, 8 }, seed: 4); // wrong inner dim
        var output = new Tensor<float>(new[] { 2, 8 });
        Assert.Throws<ArgumentException>(() =>
            CpuFusedOperations.FusedLoRAForward(input, baseOut, loraA, loraB, 1f, output));
    }

    [Fact]
    public void FusedLoRAForward_AllowsOutputToAliasBaseOutput()
    {
        const int batch = 3, inFeat = 8, rank = 2, outFeat = 5;
        var input = RandomTensor(new[] { batch, inFeat }, seed: 21);
        var baseOut = RandomTensor(new[] { batch, outFeat }, seed: 22);
        var loraA = RandomTensor(new[] { inFeat, rank }, seed: 23, scale: 0.1);
        var loraB = RandomTensor(new[] { rank, outFeat }, seed: 24, scale: 0.1);
        var originalBase = new Tensor<float>(baseOut.AsSpan().ToArray(), new[] { batch, outFeat });
        var expected = new Tensor<float>(new[] { batch, outFeat });

        CpuFusedOperations.FusedLoRAForward(input, originalBase, loraA, loraB, 0.75f, expected);
        CpuFusedOperations.FusedLoRAForward(input, baseOut, loraA, loraB, 0.75f, baseOut);

        for (int i = 0; i < expected.Length; i++)
            Assert.Equal(expected.GetFlat(i), baseOut.GetFlat(i), precision: 5);
    }

    // ─────────────────────────────────────────────────────────────────
    // FusedDDIMStep
    // ─────────────────────────────────────────────────────────────────

    [Theory]
    [InlineData(64)]
    [InlineData(1024)]
    [InlineData(16384)]
    public void FusedDDIMStep_MatchesNaiveTwoStage(int len)
    {
        var xT = RandomTensor(new[] { len }, seed: 7);
        var eps = RandomTensor(new[] { len }, seed: 8, scale: 0.5);
        float aBarT = 0.5f;
        float aBarTm1 = 0.7f;

        var output = new Tensor<float>(new[] { len });
        CpuFusedOperations.FusedDDIMStep(xT, eps, aBarT, aBarTm1, output);

        // Reference (naive two-stage DDIM, double precision):
        //   x_0_pred = (x_t − sqrt(1 − ᾱ_t) · ε) / sqrt(ᾱ_t)
        //   x_{t-1} = sqrt(ᾱ_{t-1}) · x_0_pred + sqrt(1 − ᾱ_{t-1}) · ε
        double sqrtAt = Math.Sqrt(aBarT);
        double sqrtAtm1 = Math.Sqrt(aBarTm1);
        double sqrt1mAt = Math.Sqrt(1.0 - aBarT);
        double sqrt1mAtm1 = Math.Sqrt(1.0 - aBarTm1);
        var xtSpan = xT.AsSpan();
        var epsSpan = eps.AsSpan();
        var got = output.AsSpan();
        for (int i = 0; i < len; i++)
        {
            double x0pred = (xtSpan[i] - sqrt1mAt * epsSpan[i]) / sqrtAt;
            double expected = sqrtAtm1 * x0pred + sqrt1mAtm1 * epsSpan[i];
            float diff = (float)Math.Abs(got[i] - expected);
            float tol = 1e-5f * Math.Max(1f, (float)Math.Abs(expected));
            Assert.True(diff <= tol,
                $"DDIM mismatch at idx {i}: got={got[i]}, expected={expected}, diff={diff}.");
        }
    }

    [Fact]
    public void FusedDDIMStep_RejectsNonPositiveAlphaBarT()
    {
        var x = RandomTensor(new[] { 8 }, seed: 1);
        var eps = RandomTensor(new[] { 8 }, seed: 2);
        var output = new Tensor<float>(new[] { 8 });
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            CpuFusedOperations.FusedDDIMStep(x, eps, alphaBarT: 0f, alphaBarTMinus1: 0.5f, output));
    }

    // ─────────────────────────────────────────────────────────────────
    // FusedSparseLinear
    // ─────────────────────────────────────────────────────────────────

    [Theory]
    [InlineData(1, 64, 32, 0.5)]
    [InlineData(8, 256, 128, 0.9)]
    [InlineData(4, 1024, 256, 0.95)]
    public void FusedSparseLinear_MatchesDenseMaskedReference(int batch, int inFeat, int outFeat, double sparsity)
    {
        // Build a dense [out, in] weight matrix, then mask it to the
        // requested sparsity, then convert to CSR.
        var rng = new Random(11);
        var dense = new float[outFeat * inFeat];
        for (int j = 0; j < outFeat; j++)
            for (int k = 0; k < inFeat; k++)
            {
                if (rng.NextDouble() < sparsity) continue; // zero this entry
                dense[j * inFeat + k] = (float)((rng.NextDouble() * 2.0 - 1.0) * 0.1);
            }

        // Build CSR.
        var rowOffsets = new int[outFeat + 1];
        var colIndices = new System.Collections.Generic.List<int>();
        var values = new System.Collections.Generic.List<float>();
        rowOffsets[0] = 0;
        for (int j = 0; j < outFeat; j++)
        {
            for (int k = 0; k < inFeat; k++)
            {
                float v = dense[j * inFeat + k];
                if (v != 0f) { colIndices.Add(k); values.Add(v); }
            }
            rowOffsets[j + 1] = colIndices.Count;
        }

        var input = RandomTensor(new[] { batch, inFeat }, seed: 13);
        var bias = RandomTensor(new[] { outFeat }, seed: 17, scale: 0.1);
        var values_t = new Tensor<float>(values.ToArray(), new[] { values.Count });
        var output = new Tensor<float>(new[] { batch, outFeat });
        CpuFusedOperations.FusedSparseLinear(
            input, rowOffsets, colIndices.ToArray(), values_t, bias,
            FusedActivationType.None, output);

        // Reference: dense [batch, in] · [in, out]^T (treating dense as
        // [out, in]) plus bias, no activation.
        var inSpan = input.AsSpan();
        var biasSpan = bias.AsSpan();
        var got = output.AsSpan();
        for (int b = 0; b < batch; b++)
            for (int j = 0; j < outFeat; j++)
            {
                double sum = biasSpan[j];
                for (int k = 0; k < inFeat; k++)
                    sum += inSpan[b * inFeat + k] * dense[j * inFeat + k];
                float diff = (float)Math.Abs(got[b * outFeat + j] - sum);
                float tol = 1e-4f * Math.Max(1f, (float)Math.Abs(sum));
                Assert.True(diff <= tol,
                    $"SparseLinear mismatch at [{b}, {j}]: got={got[b * outFeat + j]}, expected={sum}.");
            }
    }

    [Fact]
    public void FusedSparseLinear_RejectsNonMonotonicCsrRowOffsets()
    {
        var input = RandomTensor(new[] { 2, 4 }, seed: 31);
        var values = new Tensor<float>(new[] { 1f, 2f, 3f }, new[] { 3 });
        var output = new Tensor<float>(new[] { 2, 2 });

        Assert.Throws<ArgumentException>(() =>
            CpuFusedOperations.FusedSparseLinear(
                input,
                new[] { 0, 3, 2 },
                new[] { 0, 1, 2 },
                values,
                bias: null,
                FusedActivationType.None,
                output));
    }
}
