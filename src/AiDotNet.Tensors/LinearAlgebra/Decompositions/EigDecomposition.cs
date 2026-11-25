using System;

namespace AiDotNet.Tensors.LinearAlgebra.Decompositions;

/// <summary>
/// General (non-symmetric) eigendecomposition, the managed equivalent of LAPACK <c>dgeev</c>.
/// Eigenvalues come out as (real, imag) pairs along a trailing size-2 axis; right eigenvectors do
/// the same, one per column.
///
/// <para>float and double inputs run in double precision: diagonal balancing (EISPACK
/// <c>balanc</c>, scaling only), Householder reduction to upper-Hessenberg form (<c>orthes</c>/<c>ortran</c>),
/// then the Francis double-shift QR algorithm with deflation and exceptional shifts, followed by
/// back-substitution on the real Schur form for the eigenvectors (<c>hqr2</c>). Everything works in
/// place on two n×n buffers that are allocated once per call and reused across the batch, so a call
/// allocates O(n²) regardless of iteration count. A QR iteration that fails to converge throws
/// <see cref="ArithmeticException"/> (LAPACK <c>info &gt; 0</c>) instead of returning an unconverged
/// diagonal.</para>
///
/// <para>Each eigenvector has unit Euclidean norm and, for a complex eigenvector, a real component of
/// largest modulus, as <c>dgeev</c> returns. The positive-imaginary member of a conjugate pair comes
/// first.</para>
/// </summary>
internal static class EigDecomposition
{
    /// <summary>QR sweeps allowed per eigenvalue before the iteration is declared unconverged (LAPACK uses 30).</summary>
    private const int MaxIterationsPerEigenvalue = 30;

    private static readonly double Eps = Math.Pow(2.0, -52);

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
        using var valDataLease = eigvals.LeaseArray();
        var valData = valDataLease.Array;
        using var vecDataLease = eigvecs.LeaseArray();
        var vecData = vecDataLease.Array;
        int matStride = n * n;
        int valStride = n * 2;
        int vecStride = n * n * 2;

        if (typeof(T) != typeof(float) && typeof(T) != typeof(double))
        {
            // decimal / Half / BFloat16: Hessenberg + Francis double-shift QR carried out in T.
            for (int b = 0; b < batch; b++)
                GenericLinalg<T>.Eig(inData, b * matStride, valData, b * valStride, vecData, b * vecStride, n);
            return (eigvals, eigvecs);
        }

        if (n == 0 || batch == 0) return (eigvals, eigvecs);
        var work = new Workspace(n);
        for (int b = 0; b < batch; b++)
        {
            for (int i = 0; i < matStride; i++) work.H[i] = ToDouble(inData[b * matStride + i]);
            work.Solve();
            int offW = b * valStride, offV = b * vecStride;
            for (int k = 0; k < n; k++)
            {
                valData[offW + k * 2] = FromDouble<T>(work.D[k]);
                valData[offW + k * 2 + 1] = FromDouble<T>(work.E[k]);
            }
            work.WriteEigenvectors(vecData, offV);
        }

        return (eigvals, eigvecs);
    }

    /// <summary>
    /// The O(n²) state of one dgeev-style solve: H (input, then Hessenberg, then real Schur form and
    /// the triangular eigenvectors), V (Householder/QR transforms, then the eigenvectors), and O(n) vectors.
    /// </summary>
    private sealed class Workspace
    {
        private readonly int _n;
        internal readonly double[] H;
        internal readonly double[] V;
        internal readonly double[] D;
        internal readonly double[] E;
        private readonly double[] _ort;
        private readonly double[] _scale;

        internal Workspace(int n)
        {
            _n = n;
            H = new double[n * n];
            V = new double[n * n];
            D = new double[n];
            E = new double[n];
            _ort = new double[n];
            _scale = new double[n];
        }

        internal void Solve()
        {
            // dgeev rejects non-finite input; report NaN eigenpairs rather than iterate on NaN shifts.
            if (HasNonFinite()) { FillNaN(); return; }
            Balance();
            Orthes();
            Hqr2();
            Unbalance();
        }

        private bool HasNonFinite()
        {
            for (int i = 0; i < H.Length; i++)
                if (double.IsNaN(H[i]) || double.IsInfinity(H[i])) return true;
            return false;
        }

        private void FillNaN()
        {
            for (int i = 0; i < _n; i++) { D[i] = double.NaN; E[i] = double.NaN; }
            for (int i = 0; i < V.Length; i++) V[i] = double.NaN;
        }

        /// <summary>
        /// Diagonal similarity H ← S⁻¹·H·S with power-of-two S chosen so each row and column have
        /// comparable norms (EISPACK <c>balanc</c> without permutation; exact in floating point).
        /// </summary>
        private void Balance()
        {
            const double radix = 2.0, sqrdx = radix * radix;
            int n = _n;
            for (int i = 0; i < n; i++) _scale[i] = 1.0;
            bool done = false;
            while (!done)
            {
                done = true;
                for (int i = 0; i < n; i++)
                {
                    double c = 0.0, r = 0.0;
                    for (int j = 0; j < n; j++)
                    {
                        if (j == i) continue;
                        c += Math.Abs(H[j * n + i]);
                        r += Math.Abs(H[i * n + j]);
                    }
                    if (c == 0.0 || r == 0.0) continue;
                    double g = r / radix, f = 1.0, s = c + r;
                    while (c < g) { f *= radix; c *= sqrdx; }
                    g = r * radix;
                    while (c > g) { f /= radix; c /= sqrdx; }
                    if ((c + r) / f < 0.95 * s)
                    {
                        done = false;
                        g = 1.0 / f;
                        _scale[i] *= f;
                        for (int j = 0; j < n; j++) H[i * n + j] *= g;
                        for (int j = 0; j < n; j++) H[j * n + i] *= f;
                    }
                }
            }
        }

        /// <summary>Eigenvectors of S⁻¹·A·S map back to eigenvectors of A through x = S·x'.</summary>
        private void Unbalance()
        {
            int n = _n;
            for (int i = 0; i < n; i++)
            {
                double f = _scale[i];
                if (f == 1.0) continue;
                for (int j = 0; j < n; j++) V[i * n + j] *= f;
            }
        }

        /// <summary>
        /// Householder reduction of H to upper-Hessenberg form, accumulating the orthogonal similarity
        /// in V (EISPACK <c>orthes</c> + <c>ortran</c>).
        /// </summary>
        private void Orthes()
        {
            int n = _n, low = 0, high = n - 1;
            var h = H; var v = V; var ort = _ort;
            for (int m = low + 1; m <= high - 1; m++)
            {
                double scale = 0.0;
                for (int i = m; i <= high; i++) scale += Math.Abs(h[i * n + (m - 1)]);
                if (scale == 0.0) continue;

                double hh = 0.0;
                for (int i = high; i >= m; i--)
                {
                    ort[i] = h[i * n + (m - 1)] / scale;
                    hh += ort[i] * ort[i];
                }
                double g = Math.Sqrt(hh);
                if (ort[m] > 0) g = -g;
                hh -= ort[m] * g;
                ort[m] -= g;

                // H = (I - u·uᵀ/h)·H·(I - u·uᵀ/h)
                for (int j = m; j < n; j++)
                {
                    double f = 0.0;
                    for (int i = high; i >= m; i--) f += ort[i] * h[i * n + j];
                    f /= hh;
                    for (int i = m; i <= high; i++) h[i * n + j] -= f * ort[i];
                }
                for (int i = 0; i <= high; i++)
                {
                    double f = 0.0;
                    for (int j = high; j >= m; j--) f += ort[j] * h[i * n + j];
                    f /= hh;
                    for (int j = m; j <= high; j++) h[i * n + j] -= f * ort[j];
                }
                ort[m] = scale * ort[m];
                h[m * n + (m - 1)] = scale * g;
            }

            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                    v[i * n + j] = i == j ? 1.0 : 0.0;

            for (int m = high - 1; m >= low + 1; m--)
            {
                if (h[m * n + (m - 1)] == 0.0) continue;
                for (int i = m + 1; i <= high; i++) ort[i] = h[i * n + (m - 1)];
                for (int j = m; j <= high; j++)
                {
                    double g = 0.0;
                    for (int i = m; i <= high; i++) g += ort[i] * v[i * n + j];
                    // Double division avoids possible underflow.
                    g = g / ort[m] / h[m * n + (m - 1)];
                    for (int i = m; i <= high; i++) v[i * n + j] += g * ort[i];
                }
            }
        }

        private static void Cdiv(double xr, double xi, double yr, double yi, out double cr, out double ci)
        {
            double r, d;
            if (Math.Abs(yr) > Math.Abs(yi))
            {
                r = yi / yr;
                d = yr + r * yi;
                cr = (xr + r * xi) / d;
                ci = (xi - r * xr) / d;
            }
            else
            {
                r = yr / yi;
                d = yi + r * yr;
                cr = (r * xr + xi) / d;
                ci = (r * xi - xr) / d;
            }
        }

        /// <summary>
        /// Francis double-shift QR on the Hessenberg H down to real Schur form, accumulating the
        /// transforms in V, then back-substitution for the eigenvectors (EISPACK <c>hqr2</c>).
        /// </summary>
        private void Hqr2()
        {
            int nn = _n;
            int n = nn - 1;
            int low = 0, high = nn - 1;
            var h = H; var v = V; var d = D; var e = E;
            double exshift = 0.0;
            double p = 0, q = 0, r = 0, s = 0, z = 0, t, w, x, y;

            double norm = 0.0;
            for (int i = 0; i < nn; i++)
                for (int j = Math.Max(i - 1, 0); j < nn; j++)
                    norm += Math.Abs(h[i * nn + j]);

            int maxIter = MaxIterationsPerEigenvalue * Math.Max(nn, 10);
            int iter = 0, totalIter = 0;
            while (n >= low)
            {
                // Look for a single small sub-diagonal element.
                int l = n;
                while (l > low)
                {
                    s = Math.Abs(h[(l - 1) * nn + (l - 1)]) + Math.Abs(h[l * nn + l]);
                    if (s == 0.0) s = norm;
                    if (Math.Abs(h[l * nn + (l - 1)]) < Eps * s) break;
                    l--;
                }

                if (l == n)
                {
                    // One root found.
                    h[n * nn + n] += exshift;
                    d[n] = h[n * nn + n];
                    e[n] = 0.0;
                    n--;
                    iter = 0;
                }
                else if (l == n - 1)
                {
                    // Two roots found.
                    w = h[n * nn + (n - 1)] * h[(n - 1) * nn + n];
                    p = (h[(n - 1) * nn + (n - 1)] - h[n * nn + n]) / 2.0;
                    q = p * p + w;
                    z = Math.Sqrt(Math.Abs(q));
                    h[n * nn + n] += exshift;
                    h[(n - 1) * nn + (n - 1)] += exshift;
                    x = h[n * nn + n];

                    if (q >= 0)
                    {
                        // Real pair: rotate the 2×2 block to upper triangular.
                        z = p >= 0 ? p + z : p - z;
                        d[n - 1] = x + z;
                        d[n] = d[n - 1];
                        if (z != 0.0) d[n] = x - w / z;
                        e[n - 1] = 0.0;
                        e[n] = 0.0;
                        x = h[n * nn + (n - 1)];
                        s = Math.Abs(x) + Math.Abs(z);
                        p = x / s;
                        q = z / s;
                        r = Math.Sqrt(p * p + q * q);
                        p /= r;
                        q /= r;
                        for (int j = n - 1; j < nn; j++)
                        {
                            z = h[(n - 1) * nn + j];
                            h[(n - 1) * nn + j] = q * z + p * h[n * nn + j];
                            h[n * nn + j] = q * h[n * nn + j] - p * z;
                        }
                        for (int i = 0; i <= n; i++)
                        {
                            z = h[i * nn + (n - 1)];
                            h[i * nn + (n - 1)] = q * z + p * h[i * nn + n];
                            h[i * nn + n] = q * h[i * nn + n] - p * z;
                        }
                        for (int i = low; i <= high; i++)
                        {
                            z = v[i * nn + (n - 1)];
                            v[i * nn + (n - 1)] = q * z + p * v[i * nn + n];
                            v[i * nn + n] = q * v[i * nn + n] - p * z;
                        }
                    }
                    else
                    {
                        // Complex conjugate pair, positive imaginary part first.
                        d[n - 1] = x + p;
                        d[n] = x + p;
                        e[n - 1] = z;
                        e[n] = -z;
                    }
                    n -= 2;
                    iter = 0;
                }
                else
                {
                    // No convergence yet: form the double shift.
                    x = h[n * nn + n];
                    y = 0.0;
                    w = 0.0;
                    if (l < n)
                    {
                        y = h[(n - 1) * nn + (n - 1)];
                        w = h[n * nn + (n - 1)] * h[(n - 1) * nn + n];
                    }

                    // Wilkinson's original ad hoc shift.
                    if (iter == 10)
                    {
                        exshift += x;
                        for (int i = low; i <= n; i++) h[i * nn + i] -= x;
                        s = Math.Abs(h[n * nn + (n - 1)]) + Math.Abs(h[(n - 1) * nn + (n - 2)]);
                        x = y = 0.75 * s;
                        w = -0.4375 * s * s;
                    }

                    // MATLAB's ad hoc shift.
                    if (iter == 30)
                    {
                        s = (y - x) / 2.0;
                        s = s * s + w;
                        if (s > 0)
                        {
                            s = Math.Sqrt(s);
                            if (y < x) s = -s;
                            s = x - w / ((y - x) / 2.0 + s);
                            for (int i = low; i <= n; i++) h[i * nn + i] -= s;
                            exshift += s;
                            x = y = w = 0.964;
                        }
                    }

                    iter++;
                    if (++totalIter > maxIter)
                        throw new ArithmeticException("Eig: the QR algorithm failed to converge (LAPACK info > 0).");

                    // Look for two consecutive small sub-diagonal elements.
                    int m = n - 2;
                    while (m >= l)
                    {
                        z = h[m * nn + m];
                        r = x - z;
                        s = y - z;
                        p = (r * s - w) / h[(m + 1) * nn + m] + h[m * nn + (m + 1)];
                        q = h[(m + 1) * nn + (m + 1)] - z - r - s;
                        r = h[(m + 2) * nn + (m + 1)];
                        s = Math.Abs(p) + Math.Abs(q) + Math.Abs(r);
                        p /= s;
                        q /= s;
                        r /= s;
                        if (m == l) break;
                        if (Math.Abs(h[m * nn + (m - 1)]) * (Math.Abs(q) + Math.Abs(r)) <
                            Eps * (Math.Abs(p) * (Math.Abs(h[(m - 1) * nn + (m - 1)]) + Math.Abs(z) +
                                                  Math.Abs(h[(m + 1) * nn + (m + 1)]))))
                            break;
                        m--;
                    }

                    for (int i = m + 2; i <= n; i++)
                    {
                        h[i * nn + (i - 2)] = 0.0;
                        if (i > m + 2) h[i * nn + (i - 3)] = 0.0;
                    }

                    // Double QR step involving rows l:n and columns m:n.
                    for (int k = m; k <= n - 1; k++)
                    {
                        bool notlast = k != n - 1;
                        if (k != m)
                        {
                            p = h[k * nn + (k - 1)];
                            q = h[(k + 1) * nn + (k - 1)];
                            r = notlast ? h[(k + 2) * nn + (k - 1)] : 0.0;
                            x = Math.Abs(p) + Math.Abs(q) + Math.Abs(r);
                            if (x == 0.0) continue;
                            p /= x;
                            q /= x;
                            r /= x;
                        }

                        s = Math.Sqrt(p * p + q * q + r * r);
                        if (p < 0) s = -s;
                        if (s == 0) continue;

                        if (k != m) h[k * nn + (k - 1)] = -s * x;
                        else if (l != m) h[k * nn + (k - 1)] = -h[k * nn + (k - 1)];
                        p += s;
                        x = p / s;
                        y = q / s;
                        z = r / s;
                        q /= p;
                        r /= p;

                        // Row modification.
                        for (int j = k; j < nn; j++)
                        {
                            p = h[k * nn + j] + q * h[(k + 1) * nn + j];
                            if (notlast)
                            {
                                p += r * h[(k + 2) * nn + j];
                                h[(k + 2) * nn + j] -= p * z;
                            }
                            h[k * nn + j] -= p * x;
                            h[(k + 1) * nn + j] -= p * y;
                        }

                        // Column modification.
                        int iMax = Math.Min(n, k + 3);
                        for (int i = 0; i <= iMax; i++)
                        {
                            p = x * h[i * nn + k] + y * h[i * nn + (k + 1)];
                            if (notlast)
                            {
                                p += z * h[i * nn + (k + 2)];
                                h[i * nn + (k + 2)] -= p * r;
                            }
                            h[i * nn + k] -= p;
                            h[i * nn + (k + 1)] -= p * q;
                        }

                        // Accumulate transformations.
                        for (int i = low; i <= high; i++)
                        {
                            p = x * v[i * nn + k] + y * v[i * nn + (k + 1)];
                            if (notlast)
                            {
                                p += z * v[i * nn + (k + 2)];
                                v[i * nn + (k + 2)] -= p * r;
                            }
                            v[i * nn + k] -= p;
                            v[i * nn + (k + 1)] -= p * q;
                        }
                    }
                }
            }

            // Zero matrix: every eigenvalue is 0 and V (the identity) already holds a basis.
            if (norm == 0.0) return;

            // Back-substitute to find the eigenvectors of the quasi-triangular Schur form.
            for (n = nn - 1; n >= 0; n--)
            {
                p = d[n];
                q = e[n];

                if (q == 0)
                {
                    // Real vector.
                    int l = n;
                    h[n * nn + n] = 1.0;
                    for (int i = n - 1; i >= 0; i--)
                    {
                        w = h[i * nn + i] - p;
                        r = 0.0;
                        for (int j = l; j <= n; j++) r += h[i * nn + j] * h[j * nn + n];
                        if (e[i] < 0.0)
                        {
                            z = w;
                            s = r;
                        }
                        else
                        {
                            l = i;
                            if (e[i] == 0.0)
                            {
                                h[i * nn + n] = w != 0.0 ? -r / w : -r / (Eps * norm);
                            }
                            else
                            {
                                x = h[i * nn + (i + 1)];
                                y = h[(i + 1) * nn + i];
                                q = (d[i] - p) * (d[i] - p) + e[i] * e[i];
                                t = (x * s - z * r) / q;
                                h[i * nn + n] = t;
                                h[(i + 1) * nn + n] = Math.Abs(x) > Math.Abs(z) ? (-r - w * t) / x : (-s - y * t) / z;
                            }

                            // Overflow control.
                            t = Math.Abs(h[i * nn + n]);
                            if (Eps * t * t > 1)
                                for (int j = i; j <= n; j++) h[j * nn + n] /= t;
                        }
                    }
                }
                else if (q < 0)
                {
                    // Complex vector, kept as columns n-1 (real part) and n (imaginary part).
                    int l = n - 1;
                    if (Math.Abs(h[n * nn + (n - 1)]) > Math.Abs(h[(n - 1) * nn + n]))
                    {
                        h[(n - 1) * nn + (n - 1)] = q / h[n * nn + (n - 1)];
                        h[(n - 1) * nn + n] = -(h[n * nn + n] - p) / h[n * nn + (n - 1)];
                    }
                    else
                    {
                        Cdiv(0.0, -h[(n - 1) * nn + n], h[(n - 1) * nn + (n - 1)] - p, q, out double cr, out double ci);
                        h[(n - 1) * nn + (n - 1)] = cr;
                        h[(n - 1) * nn + n] = ci;
                    }
                    h[n * nn + (n - 1)] = 0.0;
                    h[n * nn + n] = 1.0;
                    for (int i = n - 2; i >= 0; i--)
                    {
                        double ra = 0.0, sa = 0.0;
                        for (int j = l; j <= n; j++)
                        {
                            ra += h[i * nn + j] * h[j * nn + (n - 1)];
                            sa += h[i * nn + j] * h[j * nn + n];
                        }
                        w = h[i * nn + i] - p;

                        if (e[i] < 0.0)
                        {
                            z = w;
                            r = ra;
                            s = sa;
                        }
                        else
                        {
                            l = i;
                            if (e[i] == 0)
                            {
                                Cdiv(-ra, -sa, w, q, out double cr, out double ci);
                                h[i * nn + (n - 1)] = cr;
                                h[i * nn + n] = ci;
                            }
                            else
                            {
                                // Solve the complex 2×2 system.
                                x = h[i * nn + (i + 1)];
                                y = h[(i + 1) * nn + i];
                                double vr = (d[i] - p) * (d[i] - p) + e[i] * e[i] - q * q;
                                double vi = (d[i] - p) * 2.0 * q;
                                if (vr == 0.0 && vi == 0.0)
                                    vr = Eps * norm * (Math.Abs(w) + Math.Abs(q) + Math.Abs(x) + Math.Abs(y) + Math.Abs(z));
                                Cdiv(x * r - z * ra + q * sa, x * s - z * sa - q * ra, vr, vi, out double cr, out double ci);
                                h[i * nn + (n - 1)] = cr;
                                h[i * nn + n] = ci;
                                if (Math.Abs(x) > Math.Abs(z) + Math.Abs(q))
                                {
                                    h[(i + 1) * nn + (n - 1)] = (-ra - w * h[i * nn + (n - 1)] + q * h[i * nn + n]) / x;
                                    h[(i + 1) * nn + n] = (-sa - w * h[i * nn + n] - q * h[i * nn + (n - 1)]) / x;
                                }
                                else
                                {
                                    Cdiv(-r - y * h[i * nn + (n - 1)], -s - y * h[i * nn + n], z, q, out cr, out ci);
                                    h[(i + 1) * nn + (n - 1)] = cr;
                                    h[(i + 1) * nn + n] = ci;
                                }
                            }

                            // Overflow control.
                            t = Math.Max(Math.Abs(h[i * nn + (n - 1)]), Math.Abs(h[i * nn + n]));
                            if (Eps * t * t > 1)
                            {
                                for (int j = i; j <= n; j++)
                                {
                                    h[j * nn + (n - 1)] /= t;
                                    h[j * nn + n] /= t;
                                }
                            }
                        }
                    }
                }
            }

            // Back-transform: V ← V·T maps Schur-form eigenvectors to those of the balanced matrix.
            for (int j = nn - 1; j >= low; j--)
            {
                for (int i = low; i <= high; i++)
                {
                    z = 0.0;
                    int kMax = Math.Min(j, high);
                    for (int k = low; k <= kMax; k++) z += v[i * nn + k] * h[k * nn + j];
                    v[i * nn + j] = z;
                }
            }
        }

        /// <summary>
        /// Writes the eigenvectors in V as (re, im) columns, each scaled to unit Euclidean norm with
        /// a real, positive component of largest modulus when complex (the <c>dgeev</c> normalization).
        /// </summary>
        internal void WriteEigenvectors<T>(T[] dst, int off)
        {
            int n = _n;
            var v = V;
            int k = 0;
            while (k < n)
            {
                if (E[k] > 0.0 && k + 1 < n)
                {
                    // Columns k and k+1 hold Re and Im of the eigenvector for D[k] + i·E[k].
                    double nrm2 = 0.0, best = -1.0, ur = 1.0, ui = 0.0;
                    for (int i = 0; i < n; i++)
                    {
                        double a = v[i * n + k], b = v[i * n + k + 1];
                        double m2 = a * a + b * b;
                        nrm2 += m2;
                        if (m2 > best) { best = m2; ur = a; ui = b; }
                    }
                    // Multiply by conj(x_max) / (|x_max|·‖x‖).
                    double denom = Math.Sqrt(best) * Math.Sqrt(nrm2);
                    double cr = denom > 0.0 ? ur / denom : 1.0, ci = denom > 0.0 ? -ui / denom : 0.0;
                    for (int i = 0; i < n; i++)
                    {
                        double a = v[i * n + k], b = v[i * n + k + 1];
                        double re = a * cr - b * ci, im = a * ci + b * cr;
                        dst[off + (i * n + k) * 2] = FromDouble<T>(re);
                        dst[off + (i * n + k) * 2 + 1] = FromDouble<T>(im);
                        dst[off + (i * n + k + 1) * 2] = FromDouble<T>(re);
                        dst[off + (i * n + k + 1) * 2 + 1] = FromDouble<T>(-im);
                    }
                    k += 2;
                }
                else
                {
                    double nrm2 = 0.0;
                    for (int i = 0; i < n; i++) nrm2 += v[i * n + k] * v[i * n + k];
                    double inv = nrm2 > 0.0 ? 1.0 / Math.Sqrt(nrm2) : 1.0;
                    for (int i = 0; i < n; i++)
                    {
                        dst[off + (i * n + k) * 2] = FromDouble<T>(v[i * n + k] * inv);
                        dst[off + (i * n + k) * 2 + 1] = FromDouble<T>(0.0);
                    }
                    k += 1;
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
