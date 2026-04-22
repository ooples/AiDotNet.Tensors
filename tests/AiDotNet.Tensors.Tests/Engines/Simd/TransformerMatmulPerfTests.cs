// Copyright (c) AiDotNet. All rights reserved.
// Integration tests for the transformer-FFN matmul fixes (issues
// #242/#243/#244). These are correctness tests that also exercise the
// code paths where the perf fixes live — a correctness regression here
// means one of the fixes has a math bug. A separate BenchmarkDotNet
// suite covers the perf numbers themselves (issue #245).

using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Simd;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Simd;

public class TransformerMatmulPerfTests
{
    private readonly CpuEngine _cpu = new();

    // --- Helpers -------------------------------------------------------

    private static Tensor<double> RandD(int seed, params int[] shape)
    {
        var rng = new Random(seed);
        int total = 1;
        foreach (var d in shape) total *= d;
        var data = new double[total];
        for (int i = 0; i < total; i++) data[i] = rng.NextDouble() * 2.0 - 1.0;
        return new Tensor<double>(data, shape);
    }

    private static Tensor<float> RandF(int seed, params int[] shape)
    {
        var rng = new Random(seed);
        int total = 1;
        foreach (var d in shape) total *= d;
        var data = new float[total];
        for (int i = 0; i < total; i++) data[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        return new Tensor<float>(data, shape);
    }

    private static void AssertMatmulClose(Tensor<double> actual, double[,] reference, double tol = 1e-9)
    {
        int m = reference.GetLength(0), n = reference.GetLength(1);
        Assert.Equal(new[] { m, n }, actual.Shape.ToArray());
        var data = actual.AsSpan();
        for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
        {
            double diff = Math.Abs(data[i * n + j] - reference[i, j]);
            double scale = 1 + Math.Abs(reference[i, j]);
            if (diff > tol * scale)
                throw new Xunit.Sdk.XunitException(
                    $"matmul mismatch [{i},{j}]: actual={data[i * n + j]}, ref={reference[i, j]}, diff={diff}");
        }
    }

    private static double[,] NaiveMatMul(double[,] a, double[,] b)
    {
        int m = a.GetLength(0), k = a.GetLength(1), n = b.GetLength(1);
        var c = new double[m, n];
        for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
        {
            double s = 0;
            for (int kk = 0; kk < k; kk++) s += a[i, kk] * b[kk, j];
            c[i, j] = s;
        }
        return c;
    }

    private static double[,] ToMatrix(Tensor<double> t)
    {
        int m = t._shape[0], n = t._shape[1];
        var r = new double[m, n];
        var s = t.AsSpan();
        for (int i = 0; i < m; i++) for (int j = 0; j < n; j++) r[i, j] = s[i * n + j];
        return r;
    }

    // --- #244: 2D (M × N) parallelism correctness ---------------------
    //
    // These shapes are representative of transformer FFN backward passes.
    // Before the fix, MultiplyBlocked parallelised only over M, so at
    // M=32/64 only 1-2 tasks spawned. 2D parallelism must not change the
    // result — if it does, there's a race condition or mis-partition.

    [Theory]
    [InlineData(32, 2048, 512)]   // ChronosBolt T5-small FFN backward
    [InlineData(64, 2048, 512)]   // ditto, larger batch
    [InlineData(8, 1024, 4096)]   // MOMENT FFN backward
    [InlineData(16, 256, 1024)]   // TimesFM
    [InlineData(128, 512, 512)]   // square-ish, should hit M-axis path
    public void MultiplyBlocked_2D_Parallel_Correctness_Transformer_Shapes(int m, int k, int n)
    {
        var a = RandD(1, m, k);
        var b = RandD(2, k, n);
        var reference = NaiveMatMul(ToMatrix(a), ToMatrix(b));
        var result = _cpu.TensorMatMul(a, b);
        // Double-precision; tolerance loose enough to absorb the order-of-summation
        // delta between the naive reference and the blocked kernel.
        AssertMatmulClose(result, reference, tol: 1e-8);
    }

    [Fact]
    public void MultiplyBlocked_2D_Parallel_Float_Correctness()
    {
        // Small-M float path — must also be race-free.
        var a = RandF(3, 32, 1024);
        var b = RandF(4, 1024, 512);
        var result = _cpu.TensorMatMul(a, b);
        // Cross-check with naive scalar.
        int m = 32, k = 1024, n = 512;
        var ra = a.AsSpan(); var rb = b.AsSpan();
        var expected = new float[m * n];
        for (int i = 0; i < m; i++) for (int j = 0; j < n; j++)
        {
            float s = 0;
            for (int kk = 0; kk < k; kk++) s += ra[i * k + kk] * rb[kk * n + j];
            expected[i * n + j] = s;
        }
        var got = result.AsSpan();
        for (int i = 0; i < m * n; i++)
            Assert.True(Math.Abs(expected[i] - got[i]) < 1e-3f * (1 + Math.Abs(expected[i])),
                $"float matmul mismatch at {i}: got={got[i]}, exp={expected[i]}");
    }

    // --- #243: SimdGemm.Dgemm direct kernel correctness ---------------

    [Theory]
    [InlineData(32, 512, 256)]
    [InlineData(64, 2048, 512)]
    [InlineData(1, 1, 1)]
    [InlineData(1, 64, 1)]   // Degenerate M=N=1; stresses the scalar-tail path.
    [InlineData(5, 7, 3)]    // Non-block-multiple, forces edge handling.
    public void SimdGemm_Dgemm_MatchesScalarReference(int m, int k, int n)
    {
        var rng = new Random(m + k + n);
        var a = new double[m * k];
        var b = new double[k * n];
        for (int i = 0; i < a.Length; i++) a[i] = rng.NextDouble() * 2 - 1;
        for (int i = 0; i < b.Length; i++) b[i] = rng.NextDouble() * 2 - 1;

        // Scalar reference.
        var expected = new double[m * n];
        for (int i = 0; i < m; i++) for (int j = 0; j < n; j++)
        {
            double s = 0;
            for (int kk = 0; kk < k; kk++) s += a[i * k + kk] * b[kk * n + j];
            expected[i * n + j] = s;
        }

        var c = new double[m * n];
        SimdGemm.Dgemm(a, b, c, m, k, n);

        for (int i = 0; i < m * n; i++)
            Assert.True(Math.Abs(expected[i] - c[i]) < 1e-9 * (1 + Math.Abs(expected[i])),
                $"Dgemm mismatch at {i}: expected={expected[i]}, got={c[i]}");
    }

    [Fact]
    public void SimdGemm_Dgemm_ZeroK_WritesZeros()
    {
        var a = Array.Empty<double>();
        var b = Array.Empty<double>();
        var c = new double[4 * 3];
        for (int i = 0; i < c.Length; i++) c[i] = 42.0;  // Pre-fill.
        SimdGemm.Dgemm(a, b, c, 4, 0, 3);
        foreach (var v in c) Assert.Equal(0.0, v);
    }

    // --- #242: BLAS opt-in gate behaviour -----------------------------
    //
    // We can't deterministically assert that OpenBLAS is loaded (depends
    // on AIDOTNET_USE_BLAS + the native lib being on PATH). What we CAN
    // assert is that:
    //   - the default build (env var unset) reports BLAS unavailable
    //   - the backend name is the expected sentinel
    //   - matmul still works regardless of which path fires

    [Fact]
    public void BlasProvider_DefaultBuild_ReportsUnavailable()
    {
        // This test relies on the env var being unset in CI, which is
        // the normal case. If a dev has AIDOTNET_USE_BLAS=1 locally, the
        // assertion changes meaning — skip cleanly.
        var envVar = Environment.GetEnvironmentVariable("AIDOTNET_USE_BLAS");
        if (!string.IsNullOrWhiteSpace(envVar)) return;

        Assert.False(BlasProvider.IsAvailable);
        Assert.False(BlasProvider.HasNativeSgemm);
        Assert.False(BlasProvider.HasNativeDgemm);
        Assert.Contains("SimdGemm", BlasProvider.BackendName);
    }

    [Fact]
    public void BlasProvider_BackendName_AlwaysReturnsValue()
    {
        // Never null/empty — diagnostic strings must always be present
        // so downstream logging doesn't NRE.
        var name = BlasProvider.BackendName;
        Assert.False(string.IsNullOrWhiteSpace(name));
    }

    // --- End-to-end: transformer FFN backward produces correct gradient ---
    //
    // The fix path is exercised by every matmul; this test picks a realistic
    // shape pair and checks the result matches a scalar reference.

    [Fact]
    public void TransformerFFN_MatmulChain_ProducesCorrectResult()
    {
        // Simulated FFN backward: (batch*seq, hidden) × (hidden, ff_dim)
        // then × (ff_dim, hidden). ChronosBolt T5-small at batch=32.
        var x = RandD(10, 32, 512);
        var w1 = RandD(11, 512, 2048);
        var w2 = RandD(12, 2048, 512);

        var h = _cpu.TensorMatMul(x, w1);
        var y = _cpu.TensorMatMul(h, w2);
        Assert.Equal(new[] { 32, 512 }, y.Shape.ToArray());

        // Cross-check: manual computation via naive matmul.
        var xm = ToMatrix(x); var w1m = ToMatrix(w1); var w2m = ToMatrix(w2);
        var hRef = NaiveMatMul(xm, w1m);
        var yRef = NaiveMatMul(hRef, w2m);
        AssertMatmulClose(y, yRef, tol: 1e-6);
    }
}
