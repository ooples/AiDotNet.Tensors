// Copyright (c) AiDotNet. All rights reserved.
// Issue #302 — numerical-equivalence tests for VSA primitives.

using System;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Helpers;

public class VsaPrimitivesIssue302Tests
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
    // HopfieldRetrieve
    // ─────────────────────────────────────────────────────────────────

    [Theory]
    [InlineData(16, 32, 1.0f)]
    [InlineData(128, 64, 2.0f)]
    [InlineData(1024, 128, 5.0f)]
    public void HopfieldRetrieve_MatchesNaivePerRowReference(int n, int d, float beta)
    {
        var store = RandomTensor(new[] { n, d }, seed: 1);
        var query = RandomTensor(new[] { d }, seed: 2);
        var got = new Tensor<float>(new[] { n });

        CpuVsaOperations.HopfieldRetrieve(store, query, beta, got);

        // Reference: per-row dot, scale, max-shift softmax (in double).
        var storeSpan = store.AsSpan();
        var qSpan = query.AsSpan();
        double scale = beta / Math.Sqrt(d);
        var scores = new double[n];
        for (int i = 0; i < n; i++)
        {
            double dot = 0;
            for (int k = 0; k < d; k++) dot += storeSpan[i * d + k] * qSpan[k];
            scores[i] = dot * scale;
        }
        double max = scores[0];
        for (int i = 1; i < n; i++) if (scores[i] > max) max = scores[i];
        double sum = 0;
        var expected = new double[n];
        for (int i = 0; i < n; i++) { expected[i] = Math.Exp(scores[i] - max); sum += expected[i]; }
        for (int i = 0; i < n; i++) expected[i] /= sum;

        var gotSpan = got.AsSpan();
        // Softmax outputs sum to 1; tolerance is per-element vs reference.
        double softmaxSum = 0;
        for (int i = 0; i < n; i++) softmaxSum += gotSpan[i];
        Assert.InRange(softmaxSum, 1.0 - 1e-4, 1.0 + 1e-4);

        for (int i = 0; i < n; i++)
        {
            double diff = Math.Abs(gotSpan[i] - expected[i]);
            // Allow 1e-4 abs + 1e-4 rel slop; softmax with a sharp max
            // tail can magnify rounding differences.
            Assert.True(diff <= 1e-4 + 1e-4 * Math.Abs(expected[i]),
                $"Hopfield mismatch at {i}: got={gotSpan[i]}, expected={expected[i]}, diff={diff}.");
        }
    }

    [Fact]
    public void HopfieldRetrieve_LargeBetaConcentratesOnArgmax()
    {
        // Make one store row exactly equal to query (perfect match) and
        // verify a large beta drives the softmax to that index.
        const int N = 32, D = 16;
        var store = RandomTensor(new[] { N, D }, seed: 5);
        var query = RandomTensor(new[] { D }, seed: 6);
        // Plant query as row 7 (verbatim).
        var storeSpan = store.AsWritableSpan();
        var qSpan = query.AsSpan();
        for (int k = 0; k < D; k++) storeSpan[7 * D + k] = qSpan[k];

        var alphas = new Tensor<float>(new[] { N });
        CpuVsaOperations.HopfieldRetrieve(store, query, beta: 50f, alphas);

        var aSpan = alphas.AsSpan();
        Assert.True(aSpan[7] > 0.5f,
            $"Expected planted row 7 to dominate softmax with beta=50; got alpha[7]={aSpan[7]}.");
    }

    [Fact]
    public void HopfieldRetrieve_RejectsNanBeta()
    {
        var store = RandomTensor(new[] { 4, 4 }, seed: 1);
        var query = RandomTensor(new[] { 4 }, seed: 2);
        var alphas = new Tensor<float>(new[] { 4 });
        Assert.Throws<ArgumentException>(() =>
            CpuVsaOperations.HopfieldRetrieve(store, query, float.NaN, alphas));
    }

    [Fact]
    public void HopfieldRetrieve_RejectsZeroWidthStore()
    {
        var store = new Tensor<float>(Array.Empty<float>(), new[] { 4, 0 });
        var query = new Tensor<float>(Array.Empty<float>(), new[] { 0 });
        var alphas = new Tensor<float>(new[] { 4 });

        Assert.Throws<ArgumentException>(() =>
            CpuVsaOperations.HopfieldRetrieve(store, query, beta: 1f, alphas));
    }

    // ─────────────────────────────────────────────────────────────────
    // HrrBindBatch / HrrUnbindBatch
    // ─────────────────────────────────────────────────────────────────

    [Theory]
    [InlineData(1, 8)]
    [InlineData(4, 16)]
    [InlineData(2, 64)]
    public void HrrBindBatch_MatchesCircularConvolutionReference(int B, int N)
    {
        var a = RandomTensor(new[] { B, N }, seed: 11);
        var b = RandomTensor(new[] { B, N }, seed: 13);
        var got = new Tensor<float>(new[] { B, N });

        CpuVsaOperations.HrrBindBatch(a, b, got);

        // Reference: direct circular convolution.
        // out[b, i] = sum_k a[b, k] · b[b, (i − k) mod N]
        var aS = a.AsSpan();
        var bS = b.AsSpan();
        var gS = got.AsSpan();
        for (int row = 0; row < B; row++)
        {
            for (int i = 0; i < N; i++)
            {
                double sum = 0;
                for (int k = 0; k < N; k++)
                {
                    int idx = ((i - k) % N + N) % N;
                    sum += aS[row * N + k] * bS[row * N + idx];
                }
                double diff = Math.Abs(gS[row * N + i] - sum);
                double tol = 1e-3 + 1e-3 * Math.Abs(sum);
                Assert.True(diff <= tol,
                    $"HrrBind mismatch at [{row}, {i}]: got={gS[row * N + i]}, expected={sum}, diff={diff}.");
            }
        }
    }

    [Theory]
    [InlineData(1, 16)]
    [InlineData(4, 32)]
    public void HrrUnbindBatch_MatchesCircularCorrelationReference(int B, int N)
    {
        var bound = RandomTensor(new[] { B, N }, seed: 21);
        var b = RandomTensor(new[] { B, N }, seed: 23);
        var got = new Tensor<float>(new[] { B, N });

        CpuVsaOperations.HrrUnbindBatch(bound, b, got);

        // Reference: circular correlation = circular convolution with the
        // INVOLUTION of b. The involution flips the index: b_inv[k] = b[(-k) mod N]
        // so out[i] = sum_k bound[k] · b_inv[(i − k) mod N]
        //          = sum_k bound[k] · b[(k − i) mod N]    (substituting (-x) mod N)
        var boundS = bound.AsSpan();
        var bS = b.AsSpan();
        var gS = got.AsSpan();
        for (int row = 0; row < B; row++)
        {
            for (int i = 0; i < N; i++)
            {
                double sum = 0;
                for (int k = 0; k < N; k++)
                {
                    int idx = ((k - i) % N + N) % N;
                    sum += boundS[row * N + k] * bS[row * N + idx];
                }
                double diff = Math.Abs(gS[row * N + i] - sum);
                double tol = 1e-3 + 1e-3 * Math.Abs(sum);
                Assert.True(diff <= tol,
                    $"HrrUnbind mismatch at [{row}, {i}]: got={gS[row * N + i]}, expected={sum}, diff={diff}.");
            }
        }
    }

    [Fact]
    public void Hrr_BindThenUnbind_RecoversA_WhenBIsUnitary()
    {
        // HRR algebra identity: unbind(bind(a, b), b) == a holds exactly
        // (modulo FFT roundoff) for UNITARY b — a vector whose FFT has
        // unit-magnitude bins. Random b is NOT unitary, and the recovered
        // signal differs from a by a per-bin |B[k]|² scaling, so a random-b
        // round-trip would mostly test FFT accuracy mixed with that
        // scaling, not the algebraic identity.
        //
        // The HRR identity vector e = [1, 0, 0, ..., 0] is exactly
        // unitary: FFT(e)[k] = 1 for all k. So unbind(bind(a, e), e)
        // recovers a within FFT roundoff.
        const int B = 2, N = 32;
        var a = RandomTensor(new[] { B, N }, seed: 31);
        var bData = new float[B * N];
        for (int row = 0; row < B; row++) bData[row * N + 0] = 1f;
        var b = new Tensor<float>(bData, new[] { B, N });

        var bound = new Tensor<float>(new[] { B, N });
        CpuVsaOperations.HrrBindBatch(a, b, bound);
        var recovered = new Tensor<float>(new[] { B, N });
        CpuVsaOperations.HrrUnbindBatch(bound, b, recovered);

        var aS = a.AsSpan();
        var rS = recovered.AsSpan();
        double sumSqA = 0, sumSqDiff = 0;
        for (int i = 0; i < B * N; i++)
        {
            sumSqA += aS[i] * aS[i];
            double d = aS[i] - rS[i];
            sumSqDiff += d * d;
        }
        double relErr = Math.Sqrt(sumSqDiff / Math.Max(1e-20, sumSqA));
        // Two FFT round-trips at N=32 with the identity vector should
        // recover bit-close to a; tolerance is generous against any
        // FFT-implementation rounding choice.
        Assert.True(relErr < 1e-4,
            $"HRR bind-then-unbind relative error {relErr:E4} exceeds 1e-4 with unitary b — algebra broken.");
    }
}
