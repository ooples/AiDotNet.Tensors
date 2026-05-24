using System;
using AiDotNet.Tensors.Engines.BlasManaged;
using Xunit;
using BlasManagedLib = AiDotNet.Tensors.Engines.BlasManaged.BlasManaged;

namespace AiDotNet.Tensors.Tests.Engines.BlasManaged;

/// <summary>
/// Sub-R (#408): verify the GEMV bypass produces correct output for the M=1,
/// N=1, and K=1 cases, and matches the scalar reference within FP32/FP64 bounds.
/// </summary>
public class GemvBypassTest
{
    private static void NaiveGemmFp32(
        ReadOnlySpan<float> a, int lda, bool transA,
        ReadOnlySpan<float> b, int ldb, bool transB,
        Span<float> c, int ldc, int m, int n, int k)
    {
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
            {
                double sum = 0;
                for (int kk = 0; kk < k; kk++)
                {
                    double av = transA ? a[kk * lda + i] : a[i * lda + kk];
                    double bv = transB ? b[j * ldb + kk] : b[kk * ldb + j];
                    sum += av * bv;
                }
                c[i * ldc + j] = (float)sum;
            }
    }

    private static void NaiveGemmFp64(
        ReadOnlySpan<double> a, int lda, bool transA,
        ReadOnlySpan<double> b, int ldb, bool transB,
        Span<double> c, int ldc, int m, int n, int k)
    {
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
            {
                double sum = 0;
                for (int kk = 0; kk < k; kk++)
                {
                    double av = transA ? a[kk * lda + i] : a[i * lda + kk];
                    double bv = transB ? b[j * ldb + kk] : b[kk * ldb + j];
                    sum += av * bv;
                }
                c[i * ldc + j] = sum;
            }
    }

    [Theory]
    [InlineData(1, 64, 128)]   // row × matrix, AVX2 lane-aligned
    [InlineData(1, 100, 128)]  // N tail
    [InlineData(1, 8, 256)]
    [InlineData(1, 16, 64)]
    [InlineData(1, 1024, 768)] // BERT LM-head shape
    public void M1_RowTimesMatrix_FP32_Matches_Reference(int M, int N, int K)
    {
        var rng = new Random(42);
        var a = new float[M * K];
        var b = new float[K * N];
        var cMgr = new float[M * N];
        var cRef = new float[M * N];
        for (int i = 0; i < a.Length; i++) a[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int i = 0; i < b.Length; i++) b[i] = (float)(rng.NextDouble() * 2 - 1);

        NaiveGemmFp32(a, K, false, b, N, false, cRef, N, M, N, K);
        BlasManagedLib.Gemm<float>(a, K, false, b, N, false, cMgr, N, M, N, K);

        double maxDelta = 0;
        for (int i = 0; i < cRef.Length; i++)
            maxDelta = Math.Max(maxDelta, Math.Abs(cRef[i] - cMgr[i]));
        Assert.True(maxDelta < 1e-3, $"M=1 FP32 GEMV drift {maxDelta:G6}");
    }

    [Theory]
    [InlineData(64, 1, 128)]
    [InlineData(100, 1, 128)]
    [InlineData(256, 1, 64)]
    [InlineData(768, 1, 1024)]
    public void N1_MatrixTimesCol_FP32_Matches_Reference(int M, int N, int K)
    {
        var rng = new Random(7);
        var a = new float[M * K];
        var b = new float[K * N];
        var cMgr = new float[M * N];
        var cRef = new float[M * N];
        for (int i = 0; i < a.Length; i++) a[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int i = 0; i < b.Length; i++) b[i] = (float)(rng.NextDouble() * 2 - 1);

        NaiveGemmFp32(a, K, false, b, N, false, cRef, N, M, N, K);
        BlasManagedLib.Gemm<float>(a, K, false, b, N, false, cMgr, N, M, N, K);

        double maxDelta = 0;
        for (int i = 0; i < cRef.Length; i++)
            maxDelta = Math.Max(maxDelta, Math.Abs(cRef[i] - cMgr[i]));
        Assert.True(maxDelta < 1e-3, $"N=1 FP32 GEMV drift {maxDelta:G6}");
    }

    [Theory]
    [InlineData(64, 32, 1)]
    [InlineData(128, 100, 1)]
    public void K1_OuterProduct_FP32_Matches_Reference(int M, int N, int K)
    {
        var rng = new Random(11);
        var a = new float[M * K];
        var b = new float[K * N];
        var cMgr = new float[M * N];
        var cRef = new float[M * N];
        for (int i = 0; i < a.Length; i++) a[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int i = 0; i < b.Length; i++) b[i] = (float)(rng.NextDouble() * 2 - 1);

        NaiveGemmFp32(a, K, false, b, N, false, cRef, N, M, N, K);
        BlasManagedLib.Gemm<float>(a, K, false, b, N, false, cMgr, N, M, N, K);

        double maxDelta = 0;
        for (int i = 0; i < cRef.Length; i++)
            maxDelta = Math.Max(maxDelta, Math.Abs(cRef[i] - cMgr[i]));
        Assert.True(maxDelta < 1e-5, $"K=1 outer product FP32 drift {maxDelta:G6}");
    }

    [Fact]
    public void M1_FP64_Matches_Reference()
    {
        const int M = 1, N = 256, K = 512;
        var rng = new Random(13);
        var a = new double[M * K];
        var b = new double[K * N];
        var cMgr = new double[M * N];
        var cRef = new double[M * N];
        for (int i = 0; i < a.Length; i++) a[i] = rng.NextDouble() * 2 - 1;
        for (int i = 0; i < b.Length; i++) b[i] = rng.NextDouble() * 2 - 1;

        NaiveGemmFp64(a, K, false, b, N, false, cRef, N, M, N, K);
        BlasManagedLib.Gemm<double>(a, K, false, b, N, false, cMgr, N, M, N, K);

        double maxDelta = 0;
        for (int i = 0; i < cRef.Length; i++)
            maxDelta = Math.Max(maxDelta, Math.Abs(cRef[i] - cMgr[i]));
        Assert.True(maxDelta < 1e-9, $"M=1 FP64 GEMV drift {maxDelta:G6}");
    }

    [Theory]
    [InlineData(true, false)]   // a transposed
    [InlineData(false, true)]   // b transposed
    [InlineData(true, true)]    // both transposed
    public void M1_Trans_Variants_FP32_Match_Reference(bool transA, bool transB)
    {
        const int M = 1, N = 128, K = 64;
        var rng = new Random(17);
        // For transA=true, a is stored as [K, 1] (K rows × 1 col)
        // For transB=true, b is stored as [N, K] (N rows × K cols)
        var aRows = transA ? K : M;
        var aCols = transA ? M : K;
        var bRows = transB ? N : K;
        var bCols = transB ? K : N;
        var a = new float[aRows * aCols];
        var b = new float[bRows * bCols];
        var cMgr = new float[M * N];
        var cRef = new float[M * N];
        for (int i = 0; i < a.Length; i++) a[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int i = 0; i < b.Length; i++) b[i] = (float)(rng.NextDouble() * 2 - 1);

        int lda = aCols, ldb = bCols;
        NaiveGemmFp32(a, lda, transA, b, ldb, transB, cRef, N, M, N, K);
        BlasManagedLib.Gemm<float>(a, lda, transA, b, ldb, transB, cMgr, N, M, N, K);

        double maxDelta = 0;
        for (int i = 0; i < cRef.Length; i++)
            maxDelta = Math.Max(maxDelta, Math.Abs(cRef[i] - cMgr[i]));
        Assert.True(maxDelta < 1e-3, $"M=1 transA={transA} transB={transB} FP32 drift {maxDelta:G6}");
    }
}
