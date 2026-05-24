using System;
using AiDotNet.Tensors.Engines.BlasManaged;
using Xunit;
using BlasManagedLib = AiDotNet.Tensors.Engines.BlasManaged.BlasManaged;

namespace AiDotNet.Tensors.Tests.Engines.BlasManaged;

/// <summary>
/// Sub-issue D (#372) task D.1: regression test for the partial-M silent-drop bug.
///
/// <para>
/// Before D.1, <see cref="BlasManagedLib.Gemm{T}"/> with shapes where m % 4 != 0
/// would silently produce all-zero output for trailing rows. This happened because:
/// </para>
///
/// <list type="number">
///   <item>BlasManaged.Gemm fell back to scalar Mr=Nr=4 when m % 8 != 0</item>
///   <item>But never re-checked m % 4, so e.g. m=49 still got scalar 4 dispatch</item>
///   <item>PackBothStrategy's inner loop has 'if (ir + mr > effectiveMc) break;'
///   which silently exits before computing the partial-M tail</item>
///   <item>c.Clear() at entry to BlasManaged.Gemm zero-initialized C, so the
///   skipped rows ended up as zeros — no exception, no test failure</item>
/// </list>
///
/// <para>
/// D.1 routes m % mr != 0 shapes through StreamingStrategy which has no row-alignment
/// constraint. Correctness restored.
/// </para>
/// </summary>
public class PartialMCorrectnessTest
{
    /// <summary>
    /// Reference scalar implementation for FP32 GEMM: C = A · B (no transpose).
    /// Used as the ground-truth comparison for partial-M shape tests.
    /// </summary>
    private static void ReferenceGemmFp32(
        ReadOnlySpan<float> a, int lda,
        ReadOnlySpan<float> b, int ldb,
        Span<float> c, int ldc,
        int m, int n, int k)
    {
        c.Clear();
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
            {
                float sum = 0;
                for (int p = 0; p < k; p++) sum += a[i * lda + p] * b[p * ldb + j];
                c[i * ldc + j] = sum;
            }
    }

    [Theory]
    [InlineData(49, 64, 32)]    // ResNet50_layer4-ish: m % 4 = 1
    [InlineData(197, 64, 32)]   // ViT-ish: m % 4 = 1
    [InlineData(3, 16, 8)]      // tiny m, m % 4 = 3
    [InlineData(7, 32, 16)]     // m % 4 = 3, m % 8 = 7
    [InlineData(13, 24, 12)]    // m % 4 = 1, n % 8 = 0
    public void Gemm_BitExact_With_Reference_For_Partial_M_FP32(int M, int N, int K)
    {
        var rng = new Random(42);
        var a = new float[M * K];
        var b = new float[K * N];
        var cBlas = new float[M * N];
        var cRef = new float[M * N];
        for (int i = 0; i < a.Length; i++) a[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int i = 0; i < b.Length; i++) b[i] = (float)(rng.NextDouble() * 2 - 1);

        BlasManagedLib.Gemm<float>(a, K, false, b, N, false, cBlas, N, M, N, K);
        ReferenceGemmFp32(a, K, b, N, cRef, N, M, N, K);

        // Allow 1e-4 absolute tolerance (FP32 accumulation over K terms).
        for (int i = 0; i < cBlas.Length; i++)
        {
            float delta = Math.Abs(cBlas[i] - cRef[i]);
            Assert.True(delta < 1e-4f,
                $"M={M} N={N} K={K} at [{i / N}, {i % N}]: " +
                $"BlasManaged={cBlas[i]:G6} reference={cRef[i]:G6} delta={delta:G6}");
        }
    }

    [Fact]
    public void Gemm_Trailing_Row_Is_Not_Silently_Zero_For_m49()
    {
        // The original bug: row 48 of a 49-row output was always 0.
        const int M = 49, N = 64, K = 32;
        var rng = new Random(42);
        var a = new float[M * K];
        var b = new float[K * N];
        var c = new float[M * N];
        for (int i = 0; i < a.Length; i++) a[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int i = 0; i < b.Length; i++) b[i] = (float)(rng.NextDouble() * 2 - 1);

        BlasManagedLib.Gemm<float>(a, K, false, b, N, false, c, N, M, N, K);

        // Row 48 (the trailing partial-M row) must have non-zero values.
        bool row48Nonzero = false;
        for (int j = 0; j < N; j++)
            if (c[48 * N + j] != 0) { row48Nonzero = true; break; }
        Assert.True(row48Nonzero,
            "Row 48 of a 49-row output is all-zero — partial-M bug regressed");
    }
}
