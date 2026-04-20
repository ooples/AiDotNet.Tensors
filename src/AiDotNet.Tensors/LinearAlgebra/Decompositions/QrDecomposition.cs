using System;

namespace AiDotNet.Tensors.LinearAlgebra.Decompositions;

/// <summary>
/// Householder-based QR factorization. Supports three output modes:
/// <list type="bullet">
///   <item><c>"reduced"</c> (default): <c>Q</c> is <c>M×K</c>, <c>R</c> is <c>K×N</c>, <c>K = min(M,N)</c>.</item>
///   <item><c>"complete"</c>: <c>Q</c> is <c>M×M</c>, <c>R</c> is <c>M×N</c>.</item>
///   <item><c>"r"</c>: only <c>R</c> is returned (<c>Q</c> is empty).</item>
/// </list>
/// </summary>
internal static class QrDecomposition
{
    internal static (Tensor<T> Q, Tensor<T> R) Compute<T>(Tensor<T> input, string mode)
        where T : unmanaged, IEquatable<T>, IComparable<T>
    {
        if (input is null) throw new ArgumentNullException(nameof(input));
        if (input.Rank < 2) throw new ArgumentException("QR requires at least a 2D tensor.", nameof(input));
        if (mode != "reduced" && mode != "complete" && mode != "r")
            throw new ArgumentException($"Unknown QR mode '{mode}'. Expected 'reduced', 'complete', or 'r'.", nameof(mode));

        int rank = input.Rank;
        int m = input.Shape[rank - 2];
        int n = input.Shape[rank - 1];
        int k = Math.Min(m, n);
        int qCols = mode == "complete" ? m : k;

        var qShape = (int[])input._shape.Clone();
        qShape[rank - 2] = m;
        qShape[rank - 1] = qCols;
        var rShape = (int[])input._shape.Clone();
        rShape[rank - 2] = qCols;
        rShape[rank - 1] = n;

        var Q = mode == "r" ? new Tensor<T>(new int[rank]) : new Tensor<T>(qShape);
        var R = new Tensor<T>(rShape);
        int batch = BatchSize(input._shape, rank);

        var inData = input.Contiguous().GetDataArray();
        var qData = Q.GetDataArray();
        var rData = R.GetDataArray();
        int inStride = m * n;
        int qStride = m * qCols;
        int rStride = qCols * n;

        for (int b = 0; b < batch; b++)
        {
            ComputeSingle(inData, b * inStride, qData, qData.Length == 0 ? 0 : b * qStride,
                rData, b * rStride, m, n, qCols, mode != "r");
        }

        return (Q, R);
    }

    private static void ComputeSingle<T>(
        T[] a, int offA, T[] q, int offQ, T[] r, int offR,
        int m, int n, int qCols, bool computeQ)
        where T : unmanaged, IEquatable<T>, IComparable<T>
    {
        // Householder QR: for each column j, compute a reflector that zeros
        // the sub-diagonal of A[:, j] from row j+1 down, apply to trailing
        // submatrix, and (optionally) accumulate into Q.
        int k = Math.Min(m, n);
        var work = new double[m];
        var trail = new double[n];

        // Copy A into R (upper part accumulates R; scratch region below diagonal
        // temporarily holds the reflector vectors).
        var aD = new double[m * n];
        for (int i = 0; i < m * n; i++) aD[i] = ToDouble(a[offA + i]);

        var vs = new double[k][];
        var betas = new double[k];

        for (int j = 0; j < k; j++)
        {
            int colLen = m - j;
            var v = new double[colLen];
            double norm2 = 0;
            for (int i = 0; i < colLen; i++)
            {
                v[i] = aD[(j + i) * n + j];
                norm2 += v[i] * v[i];
            }
            double norm = Math.Sqrt(norm2);
            double alpha = -Math.Sign(v[0]) * norm;
            if (v[0] == 0.0 && alpha == 0.0) alpha = norm; // edge case
            v[0] -= alpha;
            double vNorm2 = 0;
            for (int i = 0; i < colLen; i++) vNorm2 += v[i] * v[i];
            double beta = vNorm2 == 0.0 ? 0.0 : 2.0 / vNorm2;
            vs[j] = v;
            betas[j] = beta;

            // Apply to trailing submatrix.
            for (int c = j; c < n; c++)
            {
                double dot = 0;
                for (int i = 0; i < colLen; i++) dot += v[i] * aD[(j + i) * n + c];
                double scale = beta * dot;
                for (int i = 0; i < colLen; i++)
                    aD[(j + i) * n + c] -= scale * v[i];
            }
        }

        // Copy R (upper triangle of aD) into r.
        for (int i = 0; i < qCols; i++)
        {
            for (int c = 0; c < n; c++)
            {
                r[offR + i * n + c] = FromDouble<T>(i <= c && i < m ? aD[i * n + c] : 0.0);
            }
        }

        if (!computeQ) return;

        // Accumulate Q by applying reflectors in reverse to the identity (size m × qCols).
        var qD = new double[m * qCols];
        for (int i = 0; i < Math.Min(m, qCols); i++) qD[i * qCols + i] = 1.0;

        for (int j = k - 1; j >= 0; j--)
        {
            var v = vs[j];
            double beta = betas[j];
            if (beta == 0.0) continue;
            int colLen = m - j;

            for (int c = 0; c < qCols; c++)
            {
                double dot = 0;
                for (int i = 0; i < colLen; i++) dot += v[i] * qD[(j + i) * qCols + c];
                double scale = beta * dot;
                for (int i = 0; i < colLen; i++)
                    qD[(j + i) * qCols + c] -= scale * v[i];
            }
        }

        for (int i = 0; i < m * qCols; i++) q[offQ + i] = FromDouble<T>(qD[i]);
    }

    private static int BatchSize(int[] shape, int rank)
    {
        int n = 1;
        for (int i = 0; i < rank - 2; i++) n *= shape[i];
        return n;
    }

    private static double ToDouble<T>(T v)
    {
        if (typeof(T) == typeof(float)) return (float)(object)v!;
        if (typeof(T) == typeof(double)) return (double)(object)v!;
        throw new NotSupportedException($"QR requires float or double, got {typeof(T).Name}.");
    }

    private static T FromDouble<T>(double v)
    {
        if (typeof(T) == typeof(float)) return (T)(object)(float)v;
        if (typeof(T) == typeof(double)) return (T)(object)v;
        throw new NotSupportedException($"QR requires float or double, got {typeof(T).Name}.");
    }
}
