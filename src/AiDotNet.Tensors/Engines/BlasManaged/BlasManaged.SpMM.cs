using System;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.Interfaces;

namespace AiDotNet.Tensors.Engines.BlasManaged;

public static partial class BlasManaged
{
    /// <summary>
    /// Sparse×dense matrix multiply: C = α·A·B + β·C, where A is a sparse rows×cols
    /// matrix (<paramref name="a"/>, CSR or CSC) and B is dense cols×n. C is rows×n.
    /// CSR is the cache-friendly primary path (each B row read contiguously; output
    /// rows independent → deterministic and row-parallel). FP32/FP64 run a typed,
    /// auto-vectorized inner loop; other T use the generic numeric-ops path. Any
    /// <see cref="BlasOptions{T}.Epilogue"/> (bias / activation / skip / dropout /
    /// output-scale) is fused after the multiply — a single pass that vendor sparse
    /// libraries cannot match. Aligns with MKL Sparse BLAS mkl_sparse_*_mm.
    /// </summary>
    public static void SpMM<T>(
        T alpha, SparseLayout<T> a,
        ReadOnlySpan<T> b, int ldb, int n, T beta,
        Span<T> c, int ldc,
        in BlasOptions<T> options = default) where T : unmanaged
    {
        int rows = a.Rows, cols = a.Cols;
        if (rows <= 0 || n <= 0) return;

        bool csr = a.Format == SparseLayoutFormat.Csr;
        int threads = ResolveSpMMThreads(options.NumThreads, options.Mode, rows);

        if (typeof(T) == typeof(float))
        {
            SpMMFloat(
                (float)(object)alpha, csr, a.Pointers, a.Indices,
                MemoryMarshal.Cast<T, float>(a.Values),
                MemoryMarshal.Cast<T, float>(b), ldb, n, (float)(object)beta,
                MemoryMarshal.Cast<T, float>(c), ldc, rows, cols, threads);
        }
        else if (typeof(T) == typeof(double))
        {
            SpMMDouble(
                (double)(object)alpha, csr, a.Pointers, a.Indices,
                MemoryMarshal.Cast<T, double>(a.Values),
                MemoryMarshal.Cast<T, double>(b), ldb, n, (double)(object)beta,
                MemoryMarshal.Cast<T, double>(c), ldc, rows, cols, threads);
        }
        else
        {
            SpMMGeneric(alpha, csr, a.Pointers, a.Indices, a.Values, b, ldb, n, beta, c, ldc, rows, cols);
        }

        // Fused epilogue (bias / activation / skip / dropout / output-scale) — same
        // machinery GEMM uses, applied to the logical rows×n output tile.
        var epilogue = options.Epilogue;
        EpilogueChain.Apply<T>(c, ldc, rows, n, in epilogue);
    }

    // -1/1 → serial; 0 → auto (Parallel default DOP); >1 → that many. Tiny problems
    // stay serial to avoid scheduling overhead dominating.
    private static int ResolveSpMMThreads(int numThreads, BlasMode mode, int rows)
    {
        if (numThreads == 1 || numThreads == -1) return 1;
        if (rows < 64) return 1;
        return numThreads; // 0 = auto, >1 = pinned
    }

    private static unsafe void SpMMDouble(
        double alpha, bool csr, ReadOnlySpan<int> ptr, ReadOnlySpan<int> ind, ReadOnlySpan<double> val,
        ReadOnlySpan<double> b, int ldb, int n, double beta, Span<double> c, int ldc,
        int rows, int cols, int threads)
    {
        fixed (int* ptrP = ptr, indP = ind)
        fixed (double* valP = val, bP = b, cP = c)
        {
            nint ptrA = (nint)ptrP, indA = (nint)indP, valA = (nint)valP, bA = (nint)bP, cA = (nint)cP;

            if (csr)
            {
                // Row-local: each output row computed by exactly one worker, in fixed
                // nonzero order → bit-exact regardless of thread count.
                void CsrRow(int i)
                {
                    int* pp = (int*)ptrA; int* ip = (int*)indA;
                    double* vp = (double*)valA; double* bb = (double*)bA; double* cc = (double*)cA;
                    double* crow = cc + (long)i * ldc;
                    for (int j = 0; j < n; j++) crow[j] *= beta;
                    int s = pp[i], e = pp[i + 1];
                    for (int q = s; q < e; q++)
                    {
                        int k = ip[q];
                        double av = alpha * vp[q];
                        double* brow = bb + (long)k * ldb;
                        for (int j = 0; j < n; j++) crow[j] += av * brow[j];
                    }
                }

                if (threads == 1) { for (int i = 0; i < rows; i++) CsrRow(i); }
                else
                {
                    var po = new ParallelOptions();
                    if (threads > 1) po.MaxDegreeOfParallelism = threads;
                    Parallel.For(0, rows, po, CsrRow);
                }
            }
            else
            {
                // CSC: scatter into rows → not row-local; serial fixed-order keeps it
                // deterministic. Pre-scale C by beta once, then accumulate columns.
                double* cc = (double*)cA; double* bb = (double*)bA;
                int* pp = (int*)ptrA; int* ip = (int*)indA; double* vp = (double*)valA;
                for (int i = 0; i < rows; i++) { double* crow = cc + (long)i * ldc; for (int j = 0; j < n; j++) crow[j] *= beta; }
                for (int k = 0; k < cols; k++)
                {
                    double* brow = bb + (long)k * ldb;
                    int s = pp[k], e = pp[k + 1];
                    for (int q = s; q < e; q++)
                    {
                        int i = ip[q];
                        double av = alpha * vp[q];
                        double* crow = cc + (long)i * ldc;
                        for (int j = 0; j < n; j++) crow[j] += av * brow[j];
                    }
                }
            }
        }
    }

    private static unsafe void SpMMFloat(
        float alpha, bool csr, ReadOnlySpan<int> ptr, ReadOnlySpan<int> ind, ReadOnlySpan<float> val,
        ReadOnlySpan<float> b, int ldb, int n, float beta, Span<float> c, int ldc,
        int rows, int cols, int threads)
    {
        fixed (int* ptrP = ptr, indP = ind)
        fixed (float* valP = val, bP = b, cP = c)
        {
            nint ptrA = (nint)ptrP, indA = (nint)indP, valA = (nint)valP, bA = (nint)bP, cA = (nint)cP;

            if (csr)
            {
                void CsrRow(int i)
                {
                    int* pp = (int*)ptrA; int* ip = (int*)indA;
                    float* vp = (float*)valA; float* bb = (float*)bA; float* cc = (float*)cA;
                    float* crow = cc + (long)i * ldc;
                    for (int j = 0; j < n; j++) crow[j] *= beta;
                    int s = pp[i], e = pp[i + 1];
                    for (int q = s; q < e; q++)
                    {
                        int k = ip[q];
                        float av = alpha * vp[q];
                        float* brow = bb + (long)k * ldb;
                        for (int j = 0; j < n; j++) crow[j] += av * brow[j];
                    }
                }

                if (threads == 1) { for (int i = 0; i < rows; i++) CsrRow(i); }
                else
                {
                    var po = new ParallelOptions();
                    if (threads > 1) po.MaxDegreeOfParallelism = threads;
                    Parallel.For(0, rows, po, CsrRow);
                }
            }
            else
            {
                float* cc = (float*)cA; float* bb = (float*)bA;
                int* pp = (int*)ptrA; int* ip = (int*)indA; float* vp = (float*)valA;
                for (int i = 0; i < rows; i++) { float* crow = cc + (long)i * ldc; for (int j = 0; j < n; j++) crow[j] *= beta; }
                for (int k = 0; k < cols; k++)
                {
                    float* brow = bb + (long)k * ldb;
                    int s = pp[k], e = pp[k + 1];
                    for (int q = s; q < e; q++)
                    {
                        int i = ip[q];
                        float av = alpha * vp[q];
                        float* crow = cc + (long)i * ldc;
                        for (int j = 0; j < n; j++) crow[j] += av * brow[j];
                    }
                }
            }
        }
    }

    private static void SpMMGeneric<T>(
        T alpha, bool csr, ReadOnlySpan<int> ptr, ReadOnlySpan<int> ind, ReadOnlySpan<T> val,
        ReadOnlySpan<T> b, int ldb, int n, T beta, Span<T> c, int ldc, int rows, int cols) where T : unmanaged
    {
        var ops = MathHelper.GetNumericOperations<T>();
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < n; j++)
            {
                int ci = i * ldc + j;
                c[ci] = ops.Multiply(beta, c[ci]);
            }

        if (csr)
        {
            for (int i = 0; i < rows; i++)
            {
                int s = ptr[i], e = ptr[i + 1];
                int cBase = i * ldc;
                for (int q = s; q < e; q++)
                {
                    int k = ind[q];
                    T av = ops.Multiply(alpha, val[q]);
                    int bBase = k * ldb;
                    for (int j = 0; j < n; j++)
                        c[cBase + j] = ops.Add(c[cBase + j], ops.Multiply(av, b[bBase + j]));
                }
            }
        }
        else
        {
            for (int k = 0; k < cols; k++)
            {
                int bBase = k * ldb;
                int s = ptr[k], e = ptr[k + 1];
                for (int q = s; q < e; q++)
                {
                    int i = ind[q];
                    T av = ops.Multiply(alpha, val[q]);
                    int cBase = i * ldc;
                    for (int j = 0; j < n; j++)
                        c[cBase + j] = ops.Add(c[cBase + j], ops.Multiply(av, b[bBase + j]));
                }
            }
        }
    }
}
