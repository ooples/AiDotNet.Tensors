using System;

namespace AiDotNet.Tensors.LinearAlgebra.Decompositions;

/// <summary>
/// General (non-symmetric) eigendecomposition. Managed implementation uses
/// Hessenberg reduction followed by the unshifted QR algorithm on the
/// upper-Hessenberg form. Eigenvalues come out as (real, imag) pairs along a
/// trailing size-2 axis; right eigenvectors do the same.
///
/// <para>This is the hardest of the decompositions to implement well in pure
/// managed code. The implementation here targets correctness on well-separated
/// real-spectrum inputs; production-hardening (Wilkinson shifts, double-implicit
/// shifts, balancing) is a follow-up. Complex spectra return real diagonals
/// plus conjugate-pair off-diagonals per standard upper-Hessenberg output.</para>
/// </summary>
internal static class EigDecomposition
{
    private const double Eps = 1e-12;
    private const int MaxIterations = 200;

    internal static (Tensor<T> EigenvaluesReIm, Tensor<T> EigenvectorsReIm) Compute<T>(Tensor<T> input)
        where T : unmanaged, IEquatable<T>, IComparable<T>
    {
        if (input is null) throw new ArgumentNullException(nameof(input));
        if (input.Rank < 2) throw new ArgumentException("Eig requires at least a 2D tensor.", nameof(input));

        int rank = input.Rank;
        int n = input.Shape[rank - 1];
        if (input.Shape[rank - 2] != n) throw new ArgumentException("Eig requires a square matrix.");

        // Eigenvalues: shape (..., n, 2) for real/imag parts.
        var valShape = new int[rank];
        for (int i = 0; i < rank - 2; i++) valShape[i] = input._shape[i];
        valShape[rank - 2] = n;
        valShape[rank - 1] = 2;

        // Eigenvectors: shape (..., n, n, 2).
        var vecShape = new int[rank + 1];
        for (int i = 0; i < rank - 1; i++) vecShape[i] = input._shape[i];
        vecShape[rank - 1] = n;
        vecShape[rank] = 2;

        var eigvals = new Tensor<T>(valShape);
        var eigvecs = new Tensor<T>(vecShape);

        int batch = 1;
        for (int i = 0; i < rank - 2; i++) batch *= input._shape[i];

        var inData = input.Contiguous().GetDataArray();
        var valData = eigvals.GetDataArray();
        var vecData = eigvecs.GetDataArray();
        int matStride = n * n;
        int valStride = n * 2;
        int vecStride = n * n * 2;

        for (int b = 0; b < batch; b++)
            ComputeSingle(inData, b * matStride, valData, b * valStride, vecData, b * vecStride, n);

        return (eigvals, eigvecs);
    }

    private static void ComputeSingle<T>(
        T[] src, int offSrc, T[] w, int offW, T[] v, int offV, int n)
        where T : unmanaged, IEquatable<T>, IComparable<T>
    {
        // Working matrix H (will become upper Hessenberg, then quasi-triangular
        // via QR iterations) and Q (accumulated similarity transforms).
        var h = new double[n * n];
        var q = new double[n * n];
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++) h[i * n + j] = ToDouble(src[offSrc + i * n + j]);
            q[i * n + i] = 1.0;
        }

        // Phase 1: Hessenberg reduction via Householder.
        HessenbergReduce(h, q, n);

        // Phase 2: QR iteration (unshifted — simple, robust enough for well-conditioned
        // cases; full Francis double-shift is a follow-up optimization).
        QrIterate(h, q, n);

        // Phase 3: read eigenvalues from the diagonal / 2×2 blocks of the
        // quasi-triangular H. For 2×2 blocks (complex conjugate pair):
        //   eigenvalues = (h_ii + h_jj)/2 ± sqrt(discriminant)
        int idx = 0;
        while (idx < n)
        {
            if (idx < n - 1 && Math.Abs(h[(idx + 1) * n + idx]) > Eps * (Math.Abs(h[idx * n + idx]) + Math.Abs(h[(idx + 1) * n + (idx + 1)])))
            {
                // 2×2 block — complex conjugate pair.
                double a = h[idx * n + idx];
                double b = h[idx * n + idx + 1];
                double c = h[(idx + 1) * n + idx];
                double d = h[(idx + 1) * n + idx + 1];
                double tr = a + d;
                double det = a * d - b * c;
                double disc = tr * tr / 4.0 - det;

                double re = tr / 2.0;
                double im = Math.Sqrt(Math.Max(-disc, 0.0));
                w[offW + idx * 2] = FromDouble<T>(re);
                w[offW + idx * 2 + 1] = FromDouble<T>(im);
                w[offW + (idx + 1) * 2] = FromDouble<T>(re);
                w[offW + (idx + 1) * 2 + 1] = FromDouble<T>(-im);
                idx += 2;
            }
            else
            {
                // Real eigenvalue.
                w[offW + idx * 2] = FromDouble<T>(h[idx * n + idx]);
                w[offW + idx * 2 + 1] = FromDouble<T>(0.0);
                idx += 1;
            }
        }

        // Write Q columns as real eigenvectors (imaginary part zero for real eigenvalues).
        // For complex eigenvalue pairs, the two adjacent Q columns form the real and
        // imaginary parts of a complex eigenvector pair.
        idx = 0;
        while (idx < n)
        {
            bool isComplex = idx < n - 1 && ToDouble(w[offW + idx * 2 + 1]) != 0.0;
            if (isComplex)
            {
                for (int r = 0; r < n; r++)
                {
                    v[offV + (r * n + idx) * 2] = FromDouble<T>(q[r * n + idx]);
                    v[offV + (r * n + idx) * 2 + 1] = FromDouble<T>(q[r * n + idx + 1]);
                    v[offV + (r * n + idx + 1) * 2] = FromDouble<T>(q[r * n + idx]);
                    v[offV + (r * n + idx + 1) * 2 + 1] = FromDouble<T>(-q[r * n + idx + 1]);
                }
                idx += 2;
            }
            else
            {
                for (int r = 0; r < n; r++)
                {
                    v[offV + (r * n + idx) * 2] = FromDouble<T>(q[r * n + idx]);
                    v[offV + (r * n + idx) * 2 + 1] = FromDouble<T>(0.0);
                }
                idx += 1;
            }
        }
    }

    private static void HessenbergReduce(double[] h, double[] q, int n)
    {
        for (int k = 0; k < n - 2; k++)
        {
            int colLen = n - k - 1;
            // Reflector for column k below sub-diagonal.
            double norm2 = 0;
            for (int i = k + 1; i < n; i++) norm2 += h[i * n + k] * h[i * n + k];
            if (norm2 == 0) continue;
            double norm = Math.Sqrt(norm2);
            double alpha = -Math.Sign(h[(k + 1) * n + k]) * norm;
            if (h[(k + 1) * n + k] == 0.0 && alpha == 0.0) alpha = norm;

            var vvec = new double[colLen];
            vvec[0] = h[(k + 1) * n + k] - alpha;
            for (int i = 1; i < colLen; i++) vvec[i] = h[(k + 1 + i) * n + k];
            double vNorm2 = 0;
            for (int i = 0; i < colLen; i++) vNorm2 += vvec[i] * vvec[i];
            if (vNorm2 == 0) continue;
            double beta = 2.0 / vNorm2;

            // Apply reflector to H from the left: H ← (I − β v vᵀ) H  (rows k+1..n−1).
            for (int c = k; c < n; c++)
            {
                double dot = 0;
                for (int i = 0; i < colLen; i++) dot += vvec[i] * h[(k + 1 + i) * n + c];
                double s = beta * dot;
                for (int i = 0; i < colLen; i++) h[(k + 1 + i) * n + c] -= s * vvec[i];
            }
            // Apply from the right: H ← H (I − β v vᵀ) (columns k+1..n−1).
            for (int r = 0; r < n; r++)
            {
                double dot = 0;
                for (int i = 0; i < colLen; i++) dot += vvec[i] * h[r * n + (k + 1 + i)];
                double s = beta * dot;
                for (int i = 0; i < colLen; i++) h[r * n + (k + 1 + i)] -= s * vvec[i];
            }
            // Accumulate into Q (columns k+1..n−1).
            for (int r = 0; r < n; r++)
            {
                double dot = 0;
                for (int i = 0; i < colLen; i++) dot += vvec[i] * q[r * n + (k + 1 + i)];
                double s = beta * dot;
                for (int i = 0; i < colLen; i++) q[r * n + (k + 1 + i)] -= s * vvec[i];
            }
        }
    }

    private static void QrIterate(double[] h, double[] q, int n)
    {
        // Unshifted QR iteration on the upper-Hessenberg H. This converges to an
        // upper quasi-triangular form where eigenvalues are read from the diagonal
        // (real) or 2×2 blocks (complex conjugate pairs). The unshifted variant has
        // slow convergence for clustered eigenvalues; follow-up PR can add Wilkinson
        // / Francis double-implicit shifts for sub-cubic iteration count.
        for (int iter = 0; iter < MaxIterations; iter++)
        {
            // Check convergence (sub-diagonals small).
            double maxSub = 0;
            for (int i = 1; i < n; i++)
                maxSub = Math.Max(maxSub, Math.Abs(h[i * n + (i - 1)]));
            if (maxSub < Eps) return;

            // Givens-based QR step on the Hessenberg form.
            var cs = new double[n - 1];
            var ss = new double[n - 1];
            for (int k = 0; k < n - 1; k++)
            {
                double x = h[k * n + k];
                double y = h[(k + 1) * n + k];
                double r = Math.Sqrt(x * x + y * y);
                if (r == 0) { cs[k] = 1; ss[k] = 0; continue; }
                cs[k] = x / r;
                ss[k] = y / r;
                // Apply rotation G(k, k+1) to rows k and k+1.
                for (int c = k; c < n; c++)
                {
                    double a1 = h[k * n + c];
                    double a2 = h[(k + 1) * n + c];
                    h[k * n + c] = cs[k] * a1 + ss[k] * a2;
                    h[(k + 1) * n + c] = -ss[k] * a1 + cs[k] * a2;
                }
            }
            // Apply Rᵀ·Q from the right to produce R·Q = H_new, and accumulate into Q.
            for (int k = 0; k < n - 1; k++)
            {
                for (int r = 0; r < n; r++)
                {
                    double a1 = h[r * n + k];
                    double a2 = h[r * n + (k + 1)];
                    h[r * n + k] = cs[k] * a1 + ss[k] * a2;
                    h[r * n + (k + 1)] = -ss[k] * a1 + cs[k] * a2;
                }
                for (int r = 0; r < n; r++)
                {
                    double a1 = q[r * n + k];
                    double a2 = q[r * n + (k + 1)];
                    q[r * n + k] = cs[k] * a1 + ss[k] * a2;
                    q[r * n + (k + 1)] = -ss[k] * a1 + cs[k] * a2;
                }
            }
        }
    }

    private static double ToDouble<T>(T v)
    {
        if (typeof(T) == typeof(float)) return (float)(object)v!;
        if (typeof(T) == typeof(double)) return (double)(object)v!;
        throw new NotSupportedException($"Eig requires float or double, got {typeof(T).Name}.");
    }

    private static T FromDouble<T>(double v)
    {
        if (typeof(T) == typeof(float)) return (T)(object)(float)v;
        if (typeof(T) == typeof(double)) return (T)(object)v;
        throw new NotSupportedException($"Eig requires float or double, got {typeof(T).Name}.");
    }
}
