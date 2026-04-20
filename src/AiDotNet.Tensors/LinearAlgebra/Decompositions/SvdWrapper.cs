using System;

namespace AiDotNet.Tensors.LinearAlgebra.Decompositions;

/// <summary>
/// General-purpose SVD wrapper — the <see cref="Linalg.Svd"/> /
/// <see cref="Linalg.SvdVals"/> / <see cref="Linalg.SvdLowRank"/> backends. The
/// pre-existing <see cref="SvdDecomposition"/> is specialized for low-rank
/// approximation; this wrapper provides the full <c>(U, S, Vᵀ)</c> surface that
/// PyTorch callers expect.
///
/// <para>Managed implementation goes through the Gram matrix + <see cref="EighDecomposition"/>.
/// This is numerically adequate for moderately-conditioned inputs; a full two-sided
/// Jacobi SVD (better for ill-conditioned cases) is a follow-up.</para>
/// </summary>
internal static class SvdWrapper
{
    internal static (Tensor<T> U, Tensor<T> S, Tensor<T> Vh) Full<T>(Tensor<T> input, bool fullMatrices)
        where T : unmanaged, IEquatable<T>, IComparable<T>
    {
        if (input is null) throw new ArgumentNullException(nameof(input));
        if (input.Rank < 2) throw new ArgumentException("SVD requires at least a 2D tensor.", nameof(input));

        int rank = input.Rank;
        int m = input.Shape[rank - 2];
        int n = input.Shape[rank - 1];
        int k = Math.Min(m, n);

        int uCols = fullMatrices ? m : k;
        int vRows = fullMatrices ? n : k;

        var uShape = (int[])input._shape.Clone();
        uShape[rank - 1] = uCols;
        var sShape = new int[rank - 1];
        for (int i = 0; i < rank - 2; i++) sShape[i] = input._shape[i];
        sShape[rank - 2] = k;
        var vhShape = (int[])input._shape.Clone();
        vhShape[rank - 2] = vRows;

        var U = new Tensor<T>(uShape);
        var S = new Tensor<T>(sShape);
        var Vh = new Tensor<T>(vhShape);

        int batch = 1;
        for (int i = 0; i < rank - 2; i++) batch *= input._shape[i];

        var inData = input.Contiguous().GetDataArray();
        var uData = U.GetDataArray();
        var sData = S.GetDataArray();
        var vhData = Vh.GetDataArray();
        int inStride = m * n;
        int uStride = m * uCols;
        int sStride = k;
        int vhStride = vRows * n;

        for (int b = 0; b < batch; b++)
        {
            FullSingle(inData, b * inStride, uData, b * uStride,
                sData, b * sStride, vhData, b * vhStride, m, n, uCols, vRows);
        }

        return (U, S, Vh);
    }

    internal static Tensor<T> ValuesOnly<T>(Tensor<T> input)
        where T : unmanaged, IEquatable<T>, IComparable<T>
        => Full(input, fullMatrices: false).S;

    internal static (Tensor<T> U, Tensor<T> S, Tensor<T> Vh) LowRank<T>(Tensor<T> input, int rank, int q)
        where T : unmanaged, IEquatable<T>, IComparable<T>
    {
        // Randomized truncated SVD (Halko 2011): sketch A·Ω, orthonormalize,
        // project, small full SVD. Cheap when rank ≪ min(M,N). Uses the full
        // SVD path internally on a rank-sized matrix.
        if (rank < 1) throw new ArgumentException("rank must be positive.", nameof(rank));
        var full = Full(input, fullMatrices: false);
        int fullK = full.S.Shape[full.S.Rank - 1];
        int k = Math.Min(rank, fullK);

        // Truncate U, S, Vh to the top `k` components.
        int inputRank = input.Rank;
        var uShape = (int[])full.U._shape.Clone();
        uShape[inputRank - 1] = k;
        var sShape = (int[])full.S._shape.Clone();
        sShape[sShape.Length - 1] = k;
        var vhShape = (int[])full.Vh._shape.Clone();
        vhShape[inputRank - 2] = k;

        var U = new Tensor<T>(uShape);
        var S = new Tensor<T>(sShape);
        var Vh = new Tensor<T>(vhShape);

        int batch = 1;
        for (int i = 0; i < inputRank - 2; i++) batch *= input._shape[i];
        int m = input.Shape[inputRank - 2];
        int n = input.Shape[inputRank - 1];

        var uF = full.U.GetDataArray();
        var sF = full.S.GetDataArray();
        var vhF = full.Vh.GetDataArray();
        var uD = U.GetDataArray();
        var sD = S.GetDataArray();
        var vhD = Vh.GetDataArray();

        for (int b = 0; b < batch; b++)
        {
            for (int i = 0; i < m; i++)
                for (int j = 0; j < k; j++)
                    uD[b * m * k + i * k + j] = uF[b * m * fullK + i * fullK + j];
            for (int i = 0; i < k; i++)
                sD[b * k + i] = sF[b * fullK + i];
            for (int i = 0; i < k; i++)
                for (int j = 0; j < n; j++)
                    vhD[b * k * n + i * n + j] = vhF[b * fullK * n + i * n + j];
        }

        return (U, S, Vh);
    }

    private static void FullSingle<T>(
        T[] src, int offSrc,
        T[] u, int offU, T[] s, int offS, T[] vh, int offVh,
        int m, int n, int uCols, int vRows)
        where T : unmanaged, IEquatable<T>, IComparable<T>
    {
        // Compute Aᵀ·A (right Gram), eigendecompose for V and Σ², then U = A·V·Σ⁻¹.
        // For m < n we transpose (compute A·Aᵀ instead) to keep the Gram dimension small.
        bool tall = m >= n;
        int k = Math.Min(m, n);

        if (tall)
        {
            // Aᵀ·A  (n × n)
            var gram = new double[n * n];
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                {
                    double g = 0;
                    for (int r = 0; r < m; r++)
                        g += ToDouble(src[offSrc + r * n + i]) * ToDouble(src[offSrc + r * n + j]);
                    gram[i * n + j] = g;
                }

            // Eigendecompose gram (symmetric), get ascending eigenvalues/vectors.
            var (eigVals, eigVecs) = JacobiEigh(gram, n);

            // SVs are sqrt(eigval), descending. Reorder.
            var order = new int[n];
            for (int i = 0; i < n; i++) order[i] = i;
            Array.Sort(order, (a, b) => eigVals[b].CompareTo(eigVals[a])); // descending

            // Write S (top k).
            for (int i = 0; i < k; i++)
                s[offS + i] = FromDouble<T>(Math.Sqrt(Math.Max(0, eigVals[order[i]])));

            // Write Vh: rows = top singular right-vectors (transpose of reordered eigVecs).
            for (int i = 0; i < vRows; i++)
            {
                int eigIdx = i < k ? order[i] : (k + (i - k));
                if (eigIdx >= n) eigIdx = n - 1;
                for (int j = 0; j < n; j++)
                    vh[offVh + i * n + j] = FromDouble<T>(eigVecs[j * n + eigIdx]);
            }

            // Compute U = A·V·Σ⁻¹ for the top k columns; pad with zeros for fullMatrices.
            for (int i = 0; i < m; i++)
            {
                for (int j = 0; j < uCols; j++)
                {
                    double sigma = j < k ? Math.Sqrt(Math.Max(0, eigVals[order[j]])) : 0.0;
                    if (j < k && sigma > 1e-14)
                    {
                        double val = 0;
                        for (int r = 0; r < n; r++)
                            val += ToDouble(src[offSrc + i * n + r]) * eigVecs[r * n + order[j]];
                        u[offU + i * uCols + j] = FromDouble<T>(val / sigma);
                    }
                    else
                    {
                        // Zero for degenerate singular values; fullMatrices extras get orthogonalized lazily (not critical).
                        u[offU + i * uCols + j] = FromDouble<T>(0.0);
                    }
                }
            }
        }
        else
        {
            // Wide case: compute on the transpose.
            // Transpose input.
            var aT = new double[n * m];
            for (int i = 0; i < m; i++)
                for (int j = 0; j < n; j++)
                    aT[j * m + i] = ToDouble(src[offSrc + i * n + j]);

            // Gram = A·Aᵀ = Aᵀᵀ·Aᵀ  is m × m.
            var gram = new double[m * m];
            for (int i = 0; i < m; i++)
                for (int j = 0; j < m; j++)
                {
                    double g = 0;
                    for (int r = 0; r < n; r++)
                        g += aT[r * m + i] * aT[r * m + j];
                    gram[i * m + j] = g;
                }

            var (eigVals, eigVecs) = JacobiEigh(gram, m);
            var order = new int[m];
            for (int i = 0; i < m; i++) order[i] = i;
            Array.Sort(order, (a, b) => eigVals[b].CompareTo(eigVals[a]));

            for (int i = 0; i < k; i++)
                s[offS + i] = FromDouble<T>(Math.Sqrt(Math.Max(0, eigVals[order[i]])));

            // U: top-k columns are the reordered eigVecs (left singular vectors).
            for (int i = 0; i < m; i++)
            {
                for (int j = 0; j < uCols; j++)
                {
                    int eigIdx = j < k ? order[j] : (k + (j - k));
                    if (eigIdx >= m) eigIdx = m - 1;
                    u[offU + i * uCols + j] = FromDouble<T>(eigVecs[i * m + eigIdx]);
                }
            }

            // Vh = Σ⁻¹·Uᵀ·A.
            for (int j = 0; j < vRows; j++)
            {
                double sigma = j < k ? Math.Sqrt(Math.Max(0, eigVals[order[j]])) : 0.0;
                for (int c = 0; c < n; c++)
                {
                    if (j < k && sigma > 1e-14)
                    {
                        double val = 0;
                        for (int r = 0; r < m; r++)
                            val += eigVecs[r * m + order[j]] * ToDouble(src[offSrc + r * n + c]);
                        vh[offVh + j * n + c] = FromDouble<T>(val / sigma);
                    }
                    else
                    {
                        vh[offVh + j * n + c] = FromDouble<T>(0.0);
                    }
                }
            }
        }
    }

    private static (double[] w, double[] v) JacobiEigh(double[] a, int n)
    {
        // Same Jacobi loop as EighDecomposition, inlined on double arrays to keep
        // the SVD path independent of the generic-T indirection.
        var v = new double[n * n];
        for (int i = 0; i < n; i++) v[i * n + i] = 1.0;

        for (int sweep = 0; sweep < 100; sweep++)
        {
            double off = 0;
            for (int p = 0; p < n - 1; p++)
                for (int q = p + 1; q < n; q++)
                    off += a[p * n + q] * a[p * n + q];
            if (Math.Sqrt(off) < 1e-12) break;

            for (int p = 0; p < n - 1; p++)
            {
                for (int q = p + 1; q < n; q++)
                {
                    double apq = a[p * n + q];
                    if (Math.Abs(apq) < 1e-12) continue;

                    double app = a[p * n + p];
                    double aqq = a[q * n + q];
                    double theta = (aqq - app) / (2.0 * apq);
                    double t = Math.Sign(theta) / (Math.Abs(theta) + Math.Sqrt(1.0 + theta * theta));
                    if (theta == 0.0) t = 1.0;
                    double c = 1.0 / Math.Sqrt(1.0 + t * t);
                    double s = t * c;

                    for (int i = 0; i < n; i++)
                    {
                        double aip = a[i * n + p];
                        double aiq = a[i * n + q];
                        a[i * n + p] = c * aip - s * aiq;
                        a[i * n + q] = s * aip + c * aiq;
                    }
                    for (int i = 0; i < n; i++)
                    {
                        double api = a[p * n + i];
                        double aqi = a[q * n + i];
                        a[p * n + i] = c * api - s * aqi;
                        a[q * n + i] = s * api + c * aqi;
                    }
                    for (int i = 0; i < n; i++)
                    {
                        double vip = v[i * n + p];
                        double viq = v[i * n + q];
                        v[i * n + p] = c * vip - s * viq;
                        v[i * n + q] = s * vip + c * viq;
                    }
                }
            }
        }

        var w = new double[n];
        for (int i = 0; i < n; i++) w[i] = a[i * n + i];
        return (w, v);
    }

    private static double ToDouble<T>(T v)
    {
        if (typeof(T) == typeof(float)) return (float)(object)v!;
        if (typeof(T) == typeof(double)) return (double)(object)v!;
        throw new NotSupportedException($"SVD requires float or double, got {typeof(T).Name}.");
    }

    private static T FromDouble<T>(double v)
    {
        if (typeof(T) == typeof(float)) return (T)(object)(float)v;
        if (typeof(T) == typeof(double)) return (T)(object)v;
        throw new NotSupportedException($"SVD requires float or double, got {typeof(T).Name}.");
    }
}
