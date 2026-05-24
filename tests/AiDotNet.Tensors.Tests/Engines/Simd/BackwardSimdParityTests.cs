using System;
using AiDotNet.Tensors.Engines.Simd;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Simd;

/// <summary>
/// Parity tests for the SIMD backward-pass kernels (audit-2026-05 phase 5 §E).
/// On net471 these exercise the new BCL <c>Vector&lt;T&gt;</c> paths; on net10 they
/// exercise the AVX2 / scalar paths. Each asserts the kernel matches an
/// independent scalar reference. Sizes deliberately include a non-multiple-of-
/// lane-width tail so the scalar remainder loop is covered on every host.
/// </summary>
public unsafe class BackwardSimdParityTests
{
    [Theory]
    [InlineData(3, 37)]   // features 37 → tail past 8/4-lane widths
    [InlineData(4, 64)]
    [InlineData(1, 9)]
    public void SoftmaxBackward_Float_MatchesScalarReference(int batch, int features)
    {
        var rng = new Random(20260524);
        var grad = new float[batch * features];
        var sm = new float[batch * features];
        for (int i = 0; i < grad.Length; i++) grad[i] = (float)(rng.NextDouble() * 2 - 1);
        // softmax rows (positive, sum to 1) — realistic saved forward output.
        for (int b = 0; b < batch; b++)
        {
            double sum = 0;
            for (int j = 0; j < features; j++) { sm[b * features + j] = (float)rng.NextDouble(); sum += sm[b * features + j]; }
            for (int j = 0; j < features; j++) sm[b * features + j] /= (float)sum;
        }

        var got = new float[batch * features];
        fixed (float* gp = grad, sp = sm, op = got)
            SimdKernels.SoftmaxBackwardUnsafe(gp, sp, op, batch, features);

        var expected = new float[batch * features];
        for (int b = 0; b < batch; b++)
        {
            int off = b * features;
            float dot = 0;
            for (int j = 0; j < features; j++) dot += grad[off + j] * sm[off + j];
            for (int j = 0; j < features; j++) expected[off + j] = sm[off + j] * (grad[off + j] - dot);
        }

        for (int i = 0; i < got.Length; i++)
            Assert.True(MathF.Abs(got[i] - expected[i]) < 1e-5f,
                $"SoftmaxBackward float mismatch at {i}: {got[i]:G6} vs {expected[i]:G6}");
    }

    [Theory]
    [InlineData(3, 37)]
    [InlineData(2, 64)]
    public void SoftmaxBackward_Double_MatchesScalarReference(int batch, int features)
    {
        var rng = new Random(7);
        var grad = new double[batch * features];
        var sm = new double[batch * features];
        for (int i = 0; i < grad.Length; i++) grad[i] = rng.NextDouble() * 2 - 1;
        for (int b = 0; b < batch; b++)
        {
            double sum = 0;
            for (int j = 0; j < features; j++) { sm[b * features + j] = rng.NextDouble(); sum += sm[b * features + j]; }
            for (int j = 0; j < features; j++) sm[b * features + j] /= sum;
        }

        var got = new double[batch * features];
        fixed (double* gp = grad, sp = sm, op = got)
            SimdKernels.SoftmaxBackwardDouble(gp, sp, op, batch, features);

        var expected = new double[batch * features];
        for (int b = 0; b < batch; b++)
        {
            int off = b * features;
            double dot = 0;
            for (int j = 0; j < features; j++) dot += grad[off + j] * sm[off + j];
            for (int j = 0; j < features; j++) expected[off + j] = sm[off + j] * (grad[off + j] - dot);
        }

        for (int i = 0; i < got.Length; i++)
            Assert.True(Math.Abs(got[i] - expected[i]) < 1e-12,
                $"SoftmaxBackward double mismatch at {i}: {got[i]:G17} vs {expected[i]:G17}");
    }

    [Theory]
    [InlineData(37)]
    [InlineData(64)]
    [InlineData(5)]
    public void GeluBackward_Float_MatchesScalarReference(int length)
    {
        const float sqrtTwoPi = 0.7978845608028654f;
        const float coeff = 0.044715f;
        var rng = new Random(99);
        var grad = new float[length];
        var input = new float[length];
        for (int i = 0; i < length; i++) { grad[i] = (float)(rng.NextDouble() * 2 - 1); input[i] = (float)(rng.NextDouble() * 6 - 3); }

        var got = new float[length];
        fixed (float* gp = grad, ip = input, op = got)
            SimdKernels.GeluBackwardUnsafe(gp, ip, op, length);

        // Reference uses the same tanh-approx GELU' formula (exact MathF.Tanh);
        // the SIMD path uses a polynomial tanh, so tolerance covers the
        // fast-poly accuracy class the net10 path also carries.
        for (int i = 0; i < length; i++)
        {
            float x = input[i];
            float inner = sqrtTwoPi * (x + coeff * x * x * x);
            float t = MathF.Tanh(inner);
            float sech2 = 1f - t * t;
            float kPrime = sqrtTwoPi * (1f + 3f * coeff * x * x);
            float deriv = 0.5f * (1f + t) + 0.5f * x * sech2 * kPrime;
            float expected = grad[i] * deriv;
            Assert.True(MathF.Abs(got[i] - expected) < 3e-3f,
                $"GeluBackward float mismatch at {i}: {got[i]:G6} vs {expected:G6} (x={x:G4})");
        }
    }
}
