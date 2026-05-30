using System;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.Interfaces;

namespace AiDotNet.Tensors.Engines.BlasManaged;

public static partial class BlasManaged
{
    /// <summary>
    /// Sparse×dense matrix multiply: C = α·A·B + β·C, where A is a sparse rows×cols
    /// matrix (<paramref name="a"/>, CSR or CSC) and B is dense cols×n. C is rows×n.
    /// CSR is the cache-friendly primary path (each B row read contiguously; output
    /// rows independent → deterministic). Aligns with MKL Sparse BLAS mkl_sparse_*_mm.
    /// </summary>
    public static void SpMM<T>(
        T alpha, SparseLayout<T> a,
        ReadOnlySpan<T> b, int ldb, int n, T beta,
        Span<T> c, int ldc,
        in BlasOptions<T> options = default) where T : unmanaged
    {
        int rows = a.Rows, cols = a.Cols;
        if (rows <= 0 || n <= 0) return;
        var ops = MathHelper.GetNumericOperations<T>();

        // Pre-scale C by beta (whole logical rows×n output tile).
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < n; j++)
            {
                int ci = i * ldc + j;
                c[ci] = ops.Multiply(beta, c[ci]);
            }

        var ptr = a.Pointers;
        var idx = a.Indices;
        var val = a.Values;

        if (a.Format == SparseLayoutFormat.Csr)
        {
            // Row-local: C[i,:] += (α·v)·B[k,:]. Rows independent → bit-exact.
            for (int i = 0; i < rows; i++)
            {
                int start = ptr[i], end = ptr[i + 1];
                int cBase = i * ldc;
                for (int p = start; p < end; p++)
                {
                    int k = idx[p];
                    T av = ops.Multiply(alpha, val[p]);
                    int bBase = k * ldb;
                    for (int j = 0; j < n; j++)
                        c[cBase + j] = ops.Add(c[cBase + j], ops.Multiply(av, b[bBase + j]));
                }
            }
        }
        else
        {
            // CSC: iterate columns of A, scatter into the matching C rows. Fixed
            // column-then-nonzero order keeps the accumulation deterministic.
            for (int k = 0; k < cols; k++)
            {
                int start = ptr[k], end = ptr[k + 1];
                int bBase = k * ldb;
                for (int p = start; p < end; p++)
                {
                    int i = idx[p];
                    T av = ops.Multiply(alpha, val[p]);
                    int cBase = i * ldc;
                    for (int j = 0; j < n; j++)
                        c[cBase + j] = ops.Add(c[cBase + j], ops.Multiply(av, b[bBase + j]));
                }
            }
        }
    }
}
