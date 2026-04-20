using System;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Tensors.LinearAlgebra.Decompositions;

/// <summary>
/// Symmetric/Hermitian eigendecomposition. Managed implementation uses
/// cyclic Jacobi rotations for all sizes — O(n³) per iteration, converges in
/// a few sweeps for most matrices and is naturally parallelizable if we want
/// to go faster later. Native LAPACK <c>?syevd</c> is the stubbed fallback.
///
/// <para>Returns eigenvalues in ascending order with corresponding
/// eigenvectors as columns of the returned matrix.</para>
/// </summary>
internal static class EighDecomposition
{
    private const double Tolerance = 1e-12;
    private const int MaxSweeps = 100;

    internal static (Tensor<T> Eigenvalues, Tensor<T> Eigenvectors) Compute<T>(Tensor<T> input, bool upper)
        where T : unmanaged, IEquatable<T>, IComparable<T>
    {
        if (input is null) throw new ArgumentNullException(nameof(input));
        if (input.Rank < 2) throw new ArgumentException("Eigh requires at least a 2D tensor.", nameof(input));

        int rank = input.Rank;
        int n = input.Shape[rank - 1];
        if (input.Shape[rank - 2] != n) throw new ArgumentException("Eigh requires a square matrix.");

        var valShape = new int[rank - 1];
        for (int i = 0; i < rank - 2; i++) valShape[i] = input._shape[i];
        valShape[rank - 2] = n;
        var eigvals = new Tensor<T>(valShape);
        var eigvecs = new Tensor<T>((int[])input._shape.Clone());

        int batch = 1;
        for (int i = 0; i < rank - 2; i++) batch *= input._shape[i];

        var inData = input.Contiguous().GetDataArray();
        var valData = eigvals.GetDataArray();
        var vecData = eigvecs.GetDataArray();
        int matStride = n * n;
        int valStride = n;

        for (int b = 0; b < batch; b++)
            ComputeSingle(inData, b * matStride, valData, b * valStride, vecData, b * matStride, n, upper);

        return (eigvals, eigvecs);
    }

    private static void ComputeSingle<T>(
        T[] src, int offSrc, T[] w, int offW, T[] v, int offV, int n, bool upper)
        where T : unmanaged, IEquatable<T>, IComparable<T>
    {
        // Copy symmetric input to scratch A (working matrix), init V = I.
        var a = new double[n * n];
        var vmat = new double[n * n];
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                // Symmetrize via the selected triangle so numerical noise on the
                // opposite triangle doesn't bias the Jacobi updates.
                a[i * n + j] = upper
                    ? (i <= j ? ToDouble(src[offSrc + i * n + j]) : ToDouble(src[offSrc + j * n + i]))
                    : (i >= j ? ToDouble(src[offSrc + i * n + j]) : ToDouble(src[offSrc + j * n + i]));
            }
            vmat[i * n + i] = 1.0;
        }

        // Cyclic Jacobi.
        for (int sweep = 0; sweep < MaxSweeps; sweep++)
        {
            double off = 0;
            for (int p = 0; p < n - 1; p++)
                for (int q = p + 1; q < n; q++)
                    off += a[p * n + q] * a[p * n + q];
            off = Math.Sqrt(off);
            if (off < Tolerance) break;

            for (int p = 0; p < n - 1; p++)
            {
                for (int q = p + 1; q < n; q++)
                {
                    double apq = a[p * n + q];
                    if (Math.Abs(apq) < Tolerance) continue;

                    double app = a[p * n + p];
                    double aqq = a[q * n + q];
                    double theta = (aqq - app) / (2.0 * apq);
                    double t = Math.Sign(theta) / (Math.Abs(theta) + Math.Sqrt(1.0 + theta * theta));
                    if (theta == 0.0) t = 1.0;
                    double c = 1.0 / Math.Sqrt(1.0 + t * t);
                    double s = t * c;

                    // Update A: rotate rows/cols p and q.
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
                    // Update V (eigenvector accumulator).
                    for (int i = 0; i < n; i++)
                    {
                        double vip = vmat[i * n + p];
                        double viq = vmat[i * n + q];
                        vmat[i * n + p] = c * vip - s * viq;
                        vmat[i * n + q] = s * vip + c * viq;
                    }
                }
            }
        }

        // Read eigenvalues off the diagonal and sort ascending.
        var eigVals = new double[n];
        var order = new int[n];
        for (int i = 0; i < n; i++) { eigVals[i] = a[i * n + i]; order[i] = i; }
        Array.Sort(eigVals, order);

        for (int i = 0; i < n; i++)
        {
            w[offW + i] = FromDouble<T>(eigVals[i]);
            for (int r = 0; r < n; r++)
                v[offV + r * n + i] = FromDouble<T>(vmat[r * n + order[i]]);
        }
    }

    private static double ToDouble<T>(T v)
    {
        if (typeof(T) == typeof(float)) return (float)(object)v!;
        if (typeof(T) == typeof(double)) return (double)(object)v!;
        throw new NotSupportedException($"Eigh requires float or double, got {typeof(T).Name}.");
    }

    private static T FromDouble<T>(double v)
    {
        if (typeof(T) == typeof(float)) return (T)(object)(float)v;
        if (typeof(T) == typeof(double)) return (T)(object)v;
        throw new NotSupportedException($"Eigh requires float or double, got {typeof(T).Name}.");
    }
}
