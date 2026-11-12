using System;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.Interfaces;

namespace AiDotNet.Tensors.LinearAlgebra.Decompositions;

/// <summary>
/// A real scalar of element type <typeparamref name="T"/> with arithmetic operators routed through
/// <see cref="INumericOperations{T}"/>. Lets the generic linear-algebra kernels be written in the
/// same notation as their float/double counterparts while every operation stays in <typeparamref name="T"/>
/// (no round trip through <see cref="double"/>).
/// </summary>
internal readonly struct Real<T> : IEquatable<Real<T>>
{
    internal static readonly INumericOperations<T> Ops = MathHelper.GetNumericOperations<T>();

    internal readonly T V;

    internal Real(T v) { V = v; }

    internal static Real<T> Zero => new(Ops.Zero);
    internal static Real<T> One => new(Ops.One);

    /// <summary>A small integer constant (exactly representable in every supported element type).</summary>
    internal static Real<T> Int(int value) => new(Ops.FromDouble(value));

    public static implicit operator Real<T>(T v) => new(v);

    public static Real<T> operator +(Real<T> a, Real<T> b) => new(Ops.Add(a.V, b.V));
    public static Real<T> operator -(Real<T> a, Real<T> b) => new(Ops.Subtract(a.V, b.V));
    public static Real<T> operator *(Real<T> a, Real<T> b) => new(Ops.Multiply(a.V, b.V));
    public static Real<T> operator /(Real<T> a, Real<T> b) => new(Ops.Divide(a.V, b.V));
    public static Real<T> operator -(Real<T> a) => new(Ops.Negate(a.V));
    public static bool operator <(Real<T> a, Real<T> b) => Ops.LessThan(a.V, b.V);
    public static bool operator >(Real<T> a, Real<T> b) => Ops.GreaterThan(a.V, b.V);
    public static bool operator <=(Real<T> a, Real<T> b) => Ops.LessThanOrEquals(a.V, b.V);
    public static bool operator >=(Real<T> a, Real<T> b) => Ops.GreaterThanOrEquals(a.V, b.V);
    public static bool operator ==(Real<T> a, Real<T> b) => Ops.Equals(a.V, b.V);
    public static bool operator !=(Real<T> a, Real<T> b) => !Ops.Equals(a.V, b.V);

    public bool Equals(Real<T> other) => Ops.Equals(V, other.V);
    public override bool Equals(object? obj) => obj is Real<T> other && Equals(other);
    public override int GetHashCode() => V is null ? 0 : V.GetHashCode();
    public override string ToString() => V is null ? string.Empty : V.ToString() ?? string.Empty;

    internal static Real<T> Abs(Real<T> a) => new(Ops.Abs(a.V));
    internal static Real<T> Max(Real<T> a, Real<T> b) => a >= b ? a : b;
    internal bool IsZero => Ops.Equals(V, Ops.Zero);
    internal bool IsNaN => Ops.IsNaN(V);

    /// <summary>|a| carrying the sign of b (Fortran SIGN).</summary>
    internal static Real<T> CopySign(Real<T> a, Real<T> b) => b >= Zero ? Abs(a) : -Abs(a);

    /// <summary>
    /// Square root to the full precision of <typeparamref name="T"/>. The numeric-operations square
    /// root of some types (decimal) is only double-accurate, so the seed is polished with two Newton
    /// steps <c>r ← (r + x/r)/2</c> carried out in <typeparamref name="T"/>.
    /// </summary>
    internal static Real<T> Sqrt(Real<T> x)
    {
        if (!(x > Zero)) return new(Ops.Sqrt(x.V));
        Real<T> r = new(Ops.Sqrt(x.V));
        if (!(r > Zero)) return r;
        Real<T> two = Int(2);
        for (int i = 0; i < 2; i++) r = (r + x / r) / two;
        return r;
    }

    /// <summary>sqrt(a² + b²) without intermediate overflow.</summary>
    internal static Real<T> Hypot(Real<T> a, Real<T> b)
    {
        Real<T> aa = Abs(a), bb = Abs(b);
        Real<T> m = Max(aa, bb);
        if (m.IsZero) return Zero;
        Real<T> x = aa / m, y = bb / m;
        return m * Sqrt(x * x + y * y);
    }
}

/// <summary>
/// Element-type-generic dense linear-algebra kernels — the managed reference for every real
/// field type that has <see cref="INumericOperations{T}"/> but no dedicated float/double fast
/// path (decimal, Half, BFloat16). All arithmetic stays in <typeparamref name="T"/>; convergence
/// thresholds scale with the type's own machine epsilon, so decimal results are accurate to
/// roughly 1e-26 rather than to double precision.
///
/// <para>Algorithms and conventions follow LAPACK (the backend PyTorch's <c>torch.linalg</c> uses):
/// <list type="bullet">
///   <item>Cholesky: right-looking <c>?potrf</c>; info = order of the first non-positive minor.</item>
///   <item>LU: partial-pivot Doolittle <c>?getrf</c>; 0-based pivots; a zero pivot is skipped.</item>
///   <item>QR: Householder <c>?geqrf</c>/<c>?orgqr</c> with the same reflector sign as the float path.</item>
///   <item>Eigh: cyclic Jacobi; eigenvalues ascending, eigenvectors as columns.</item>
///   <item>Eig: Householder Hessenberg reduction + Francis double-shift QR (<c>?hseqr</c>), eigenvalues in
///   Schur order with the positive-imaginary member of each conjugate pair first, eigenvectors by
///   inverse iteration normalized to unit 2-norm with the largest component real (<c>?geev</c>).</item>
///   <item>SVD: one-sided (Hestenes) Jacobi; singular values descending, orthonormal completion of U/Vᵀ
///   for full matrices (<c>?gesvd</c>).</item>
/// </list></para>
/// </summary>
internal static class GenericLinalg<T>
    where T : unmanaged, IEquatable<T>, IComparable<T>
{
    /// <summary>Machine epsilon, published atomically by generic static initialization.</summary>
    private static readonly Real<T> Eps = MachineEpsilon();
    /// <summary>True when <typeparamref name="T"/> is a real field type the generic kernels support.</summary>
    internal static bool IsSupported =>
        MathHelper.IsFloatingPoint<T>() || typeof(T) == typeof(decimal);

    /// <summary>Whether <typeparamref name="T"/> can represent NaN (decimal cannot).</summary>
    internal static bool HasNaN => MathHelper.IsFloatingPoint<T>();

    /// <summary>
    /// Rejects element types that are not a field (integers, bool): a factorization needs exact
    /// division, which those types lack. <c>torch.linalg</c> rejects integer dtypes the same way.
    /// </summary>
    internal static void EnsureSupported(string op)
    {
        if (!IsSupported)
            throw new NotSupportedException(
                $"{op} requires a real floating-point element type (float, double, decimal, Half or BFloat16); " +
                $"got {typeof(T).Name}. Integer element types are not supported, matching torch.linalg.");
    }

    private static Real<T> MachineEpsilon()
    {
        if (!IsSupported) return Real<T>.Zero;
        Real<T> one = Real<T>.One, two = Real<T>.Int(2), e = one;
        for (int i = 0; i < 400; i++)
        {
            Real<T> half = e / two;
            if (one + half == one) break;
            e = half;
        }
        return e;
    }

    private static Real<T>[] Load(T[] src, int off, int count)
    {
        var r = new Real<T>[count];
        for (int i = 0; i < count; i++) r[i] = src[off + i];
        return r;
    }

    // ── Cholesky ───────────────────────────────────────────────────────────

    /// <summary>In-place Cholesky of the n×n block at <paramref name="off"/>. Returns 0 or the
    /// 1-based order of the first leading minor that is not positive definite.</summary>
    internal static int CholeskyFactor(T[] a, int off, int n, bool upper)
    {
        EnsureSupported("Cholesky");
        for (int j = 0; j < n; j++)
        {
            Real<T> diag = a[off + j * n + j];
            for (int k = 0; k < j; k++)
            {
                Real<T> e = upper ? a[off + k * n + j] : a[off + j * n + k];
                diag -= e * e;
            }
            if (!(diag > Real<T>.Zero)) return j + 1;
            Real<T> sqrtDiag = Real<T>.Sqrt(diag);
            a[off + j * n + j] = sqrtDiag.V;
            for (int i = j + 1; i < n; i++)
            {
                Real<T> sum = upper ? a[off + j * n + i] : a[off + i * n + j];
                for (int k = 0; k < j; k++)
                {
                    sum -= upper
                        ? (Real<T>)a[off + k * n + i] * a[off + k * n + j]
                        : (Real<T>)a[off + i * n + k] * a[off + j * n + k];
                }
                Real<T> val = sum / sqrtDiag;
                if (upper) a[off + j * n + i] = val.V;
                else a[off + i * n + j] = val.V;
            }
        }
        return 0;
    }

    // ── LU ─────────────────────────────────────────────────────────────────

    /// <summary>In-place partial-pivot LU (<c>getrf</c>) of the m×n block; 0-based pivots.</summary>
    internal static void LuFactor(T[] a, int offA, int[] piv, int offPiv, int m, int n)
    {
        EnsureSupported("LU");
        int k = Math.Min(m, n);
        for (int j = 0; j < k; j++)
        {
            int pivRow = j;
            Real<T> maxAbs = Real<T>.Abs(a[offA + j * n + j]);
            for (int i = j + 1; i < m; i++)
            {
                Real<T> v = Real<T>.Abs(a[offA + i * n + j]);
                if (v > maxAbs) { maxAbs = v; pivRow = i; }
            }
            piv[offPiv + j] = pivRow;
            if (pivRow != j)
            {
                for (int c = 0; c < n; c++)
                    (a[offA + pivRow * n + c], a[offA + j * n + c]) = (a[offA + j * n + c], a[offA + pivRow * n + c]);
            }

            Real<T> pivVal = a[offA + j * n + j];
            if (pivVal.IsZero) continue;
            for (int i = j + 1; i < m; i++)
            {
                Real<T> factor = (Real<T>)a[offA + i * n + j] / pivVal;
                a[offA + i * n + j] = factor.V;
                for (int c = j + 1; c < n; c++)
                    a[offA + i * n + c] = ((Real<T>)a[offA + i * n + c] - factor * a[offA + j * n + c]).V;
            }
        }
    }

    /// <summary>Solves A·X = B in place from a packed LU factor (<c>getrs</c>). A zero pivot yields NaN
    /// when <typeparamref name="T"/> has one, otherwise zero (callers report it through info).</summary>
    internal static void LuSolve(T[] lu, int offLU, int[] piv, int offPiv, T[] x, int offX, int n, int colsX)
    {
        EnsureSupported("LU solve");
        for (int i = 0; i < n; i++)
        {
            int p = piv[offPiv + i];
            if (p == i) continue;
            for (int c = 0; c < colsX; c++)
                (x[offX + i * colsX + c], x[offX + p * colsX + c]) = (x[offX + p * colsX + c], x[offX + i * colsX + c]);
        }
        for (int i = 1; i < n; i++)
            for (int c = 0; c < colsX; c++)
            {
                Real<T> sum = x[offX + i * colsX + c];
                for (int j = 0; j < i; j++) sum -= (Real<T>)lu[offLU + i * n + j] * x[offX + j * colsX + c];
                x[offX + i * colsX + c] = sum.V;
            }
        T singular = HasNaN ? Real<T>.Ops.FromDouble(double.NaN) : Real<T>.Ops.Zero;
        for (int i = n - 1; i >= 0; i--)
        {
            Real<T> pivVal = lu[offLU + i * n + i];
            for (int c = 0; c < colsX; c++)
            {
                Real<T> sum = x[offX + i * colsX + c];
                for (int j = i + 1; j < n; j++) sum -= (Real<T>)lu[offLU + i * n + j] * x[offX + j * colsX + c];
                x[offX + i * colsX + c] = pivVal.IsZero ? singular : (sum / pivVal).V;
            }
        }
    }

    /// <summary>1-based index of the first zero on the diagonal of a packed n×n factor, or 0.</summary>
    internal static int FirstZeroDiagonal(T[] lu, int off, int n)
    {
        for (int k = 0; k < n; k++)
            if (((Real<T>)lu[off + k * n + k]).IsZero) return k + 1;
        return 0;
    }

    /// <summary>det = (−1)^swaps · Π diag(LU) for one packed n×n factor.</summary>
    internal static T DetFromLu(T[] lu, int offLU, int[] piv, int offPiv, int n)
    {
        EnsureSupported("Det");
        Real<T> val = Real<T>.One;
        bool negate = false;
        for (int i = 0; i < n; i++)
        {
            val *= lu[offLU + i * n + i];
            if (piv[offPiv + i] != i) negate = !negate;
        }
        return (negate ? -val : val).V;
    }

    // ── Triangular solve ───────────────────────────────────────────────────

    /// <summary>In-place triangular solve (<c>trsm</c>, left side, alpha = 1) of one n×nrhs block.</summary>
    internal static void TriangularSolve(
        T[] a, int offA, T[] x, int offX, int n, int nrhs, bool upper, bool transpose, bool unitDiagonal)
    {
        EnsureSupported("Triangular solve");
        bool backward = upper ^ transpose;
        for (int c = 0; c < nrhs; c++)
        {
            for (int ii = 0; ii < n; ii++)
            {
                int i = backward ? n - 1 - ii : ii;
                int jStart = backward ? i + 1 : 0;
                int jEnd = backward ? n : i;
                Real<T> sum = x[offX + i * nrhs + c];
                for (int j = jStart; j < jEnd; j++)
                {
                    int aRow = transpose ? j : i;
                    int aCol = transpose ? i : j;
                    sum -= (Real<T>)a[offA + aRow * n + aCol] * x[offX + j * nrhs + c];
                }
                x[offX + i * nrhs + c] = unitDiagonal ? sum.V : (sum / a[offA + i * n + i]).V;
            }
        }
    }

    // ── QR ─────────────────────────────────────────────────────────────────

    /// <summary>Householder QR of one m×n block into Q (m×qCols) and R (qCols×n).</summary>
    internal static void Qr(T[] a, int offA, T[] q, int offQ, T[] r, int offR, int m, int n, int qCols, bool computeQ)
    {
        EnsureSupported("QR");
        int k = Math.Min(m, n);
        var aD = Load(a, offA, m * n);
        var vs = new Real<T>[k][];
        var betas = new Real<T>[k];
        Real<T> two = Real<T>.Int(2);

        for (int j = 0; j < k; j++)
        {
            int colLen = m - j;
            var v = new Real<T>[colLen];
            for (int i = 0; i < colLen; i++) v[i] = aD[(j + i) * n + j];
            Real<T> norm = Norm2(v);
            // A dimensionless reflector avoids squaring the input scale (overflow in Half/decimal).
            if (!norm.IsZero)
                for (int i = 0; i < colLen; i++) v[i] /= norm;
            Real<T> alpha = norm.IsZero ? Real<T>.Zero : v[0] > Real<T>.Zero ? -Real<T>.One : Real<T>.One;
            v[0] -= alpha;
            Real<T> vNorm2 = Real<T>.Zero;
            for (int i = 0; i < colLen; i++) vNorm2 += v[i] * v[i];
            Real<T> beta = vNorm2.IsZero ? Real<T>.Zero : two / vNorm2;
            vs[j] = v;
            betas[j] = beta;

            for (int c = j; c < n; c++)
            {
                Real<T> dot = Real<T>.Zero;
                for (int i = 0; i < colLen; i++) dot += v[i] * aD[(j + i) * n + c];
                Real<T> scale = beta * dot;
                for (int i = 0; i < colLen; i++) aD[(j + i) * n + c] -= scale * v[i];
            }
        }

        for (int i = 0; i < qCols; i++)
            for (int c = 0; c < n; c++)
                r[offR + i * n + c] = i <= c && i < m ? aD[i * n + c].V : Real<T>.Ops.Zero;

        if (!computeQ) return;

        var qD = new Real<T>[m * qCols];
        for (int i = 0; i < qD.Length; i++) qD[i] = Real<T>.Zero;
        for (int i = 0; i < Math.Min(m, qCols); i++) qD[i * qCols + i] = Real<T>.One;
        for (int j = k - 1; j >= 0; j--)
        {
            var v = vs[j];
            Real<T> beta = betas[j];
            if (beta.IsZero) continue;
            int colLen = m - j;
            for (int c = 0; c < qCols; c++)
            {
                Real<T> dot = Real<T>.Zero;
                for (int i = 0; i < colLen; i++) dot += v[i] * qD[(j + i) * qCols + c];
                Real<T> scale = beta * dot;
                for (int i = 0; i < colLen; i++) qD[(j + i) * qCols + c] -= scale * v[i];
            }
        }
        for (int i = 0; i < qD.Length; i++) q[offQ + i] = qD[i].V;
    }

    private static Real<T> Norm2(Real<T>[] v)
    {
        // Scaled 2-norm (LAPACK ?nrm2 style) so squares cannot overflow narrow types or decimal.
        Real<T> scale = Real<T>.Zero;
        for (int i = 0; i < v.Length; i++) scale = Real<T>.Max(scale, Real<T>.Abs(v[i]));
        if (scale.IsZero) return Real<T>.Zero;
        Real<T> ss = Real<T>.Zero;
        for (int i = 0; i < v.Length; i++) { Real<T> t = v[i] / scale; ss += t * t; }
        return scale * Real<T>.Sqrt(ss);
    }

    // ── Symmetric eigendecomposition ───────────────────────────────────────

    /// <summary>Cyclic Jacobi rotation parameters (c, s) that annihilate the (p, q) entry.</summary>
    private static (Real<T> C, Real<T> S) JacobiRotation(Real<T> app, Real<T> aqq, Real<T> apq)
    {
        Real<T> one = Real<T>.One, two = Real<T>.Int(2);
        Real<T> theta = (aqq - app) / (two * apq);
        Real<T> absTheta = Real<T>.Abs(theta);
        Real<T> t;
        if (theta.IsZero) t = one;
        // For huge θ, θ² would overflow decimal/Half; t → 1/(2θ) is exact to working precision there.
        else if (absTheta > one / Real<T>.Sqrt(Eps)) t = one / (two * theta);
        else t = Real<T>.CopySign(one, theta) / (absTheta + Real<T>.Sqrt(one + theta * theta));
        Real<T> c = one / Real<T>.Sqrt(one + t * t);
        return (c, t * c);
    }

    /// <summary>Eigenvalues (ascending) and eigenvectors (columns) of one symmetric n×n block.</summary>
    internal static void Eigh(T[] src, int offSrc, T[] w, int offW, T[] v, int offV, int n, bool upper)
    {
        EnsureSupported("Eigh");
        var a = new Real<T>[n * n];
        var vm = new Real<T>[n * n];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
            {
                bool own = upper ? i <= j : i >= j;
                a[i * n + j] = own ? src[offSrc + i * n + j] : src[offSrc + j * n + i];
                vm[i * n + j] = i == j ? Real<T>.One : Real<T>.Zero;
            }

        Real<T> scale = Real<T>.Zero;
        for (int i = 0; i < n * n; i++) scale = Real<T>.Max(scale, Real<T>.Abs(a[i]));
        Real<T> skip = Eps * scale;

        for (int sweep = 0; sweep < 100; sweep++)
        {
            bool rotated = false;
            for (int p = 0; p < n - 1; p++)
            {
                for (int q = p + 1; q < n; q++)
                {
                    Real<T> apq = a[p * n + q];
                    if (Real<T>.Abs(apq) <= skip) continue;
                    rotated = true;
                    var (c, s) = JacobiRotation(a[p * n + p], a[q * n + q], apq);
                    for (int i = 0; i < n; i++)
                    {
                        Real<T> aip = a[i * n + p], aiq = a[i * n + q];
                        a[i * n + p] = c * aip - s * aiq;
                        a[i * n + q] = s * aip + c * aiq;
                    }
                    for (int i = 0; i < n; i++)
                    {
                        Real<T> api = a[p * n + i], aqi = a[q * n + i];
                        a[p * n + i] = c * api - s * aqi;
                        a[q * n + i] = s * api + c * aqi;
                    }
                    a[p * n + q] = Real<T>.Zero;
                    a[q * n + p] = Real<T>.Zero;
                    for (int i = 0; i < n; i++)
                    {
                        Real<T> vip = vm[i * n + p], viq = vm[i * n + q];
                        vm[i * n + p] = c * vip - s * viq;
                        vm[i * n + q] = s * vip + c * viq;
                    }
                }
            }
            if (!rotated) break;
        }

        var vals = new Real<T>[n];
        for (int i = 0; i < n; i++) vals[i] = a[i * n + i];
        var order = StableOrder(vals, descending: false);
        for (int i = 0; i < n; i++)
        {
            w[offW + i] = vals[order[i]].V;
            for (int r = 0; r < n; r++) v[offV + r * n + i] = vm[r * n + order[i]].V;
        }
    }

    private static int[] StableOrder(Real<T>[] keys, bool descending)
    {
        int n = keys.Length;
        var order = new int[n];
        for (int i = 0; i < n; i++) order[i] = i;
        for (int i = 1; i < n; i++)
        {
            int cur = order[i];
            int j = i - 1;
            while (j >= 0 && (descending ? keys[order[j]] < keys[cur] : keys[order[j]] > keys[cur]))
            {
                order[j + 1] = order[j];
                j--;
            }
            order[j + 1] = cur;
        }
        return order;
    }

    // ── General eigendecomposition ─────────────────────────────────────────

    /// <summary>
    /// Eigenpairs of one general real n×n block. <paramref name="w"/> receives (re, im) pairs,
    /// <paramref name="v"/> the right eigenvectors as (re, im) columns — the layout of
    /// <see cref="EigDecomposition"/>.
    /// </summary>
    internal static void Eig(T[] src, int offSrc, T[] w, int offW, T[] v, int offV, int n)
    {
        EnsureSupported("Eig");
        if (n == 0) return;
        var a0 = Load(src, offSrc, n * n);
        Real<T> scale = Real<T>.Zero;
        for (int i = 0; i < n * n; i++) scale = Real<T>.Max(scale, Real<T>.Abs(a0[i]));
        if (scale.IsZero) scale = Real<T>.One;
        // QR's products and inverse iteration operate on an O(1) matrix, not on squared input magnitudes.
        for (int i = 0; i < a0.Length; i++) a0[i] /= scale;
        var h = (Real<T>[])a0.Clone();
        HessenbergReduce(h, n);
        var (wr, wi) = FrancisQr(h, n);

        for (int k = 0; k < n; k++)
        {
            w[offW + k * 2] = (wr[k] * scale).V;
            w[offW + k * 2 + 1] = (wi[k] * scale).V;
            var (xr, xi) = InverseIteration(a0, n, wr[k], wi[k], Real<T>.One, wr, wi, k, v, offV);
            for (int r = 0; r < n; r++)
            {
                v[offV + (r * n + k) * 2] = xr[r].V;
                v[offV + (r * n + k) * 2 + 1] = xi[r].V;
            }
        }
    }

    private static void HessenbergReduce(Real<T>[] h, int n)
    {
        Real<T> two = Real<T>.Int(2);
        for (int k = 0; k < n - 2; k++)
        {
            int colLen = n - k - 1;
            var vv = new Real<T>[colLen];
            for (int i = 0; i < colLen; i++) vv[i] = h[(k + 1 + i) * n + k];
            Real<T> norm = Norm2(vv);
            if (norm.IsZero) continue;
            for (int i = 0; i < colLen; i++) vv[i] /= norm;
            Real<T> alpha = vv[0] > Real<T>.Zero ? -Real<T>.One : Real<T>.One;
            vv[0] -= alpha;
            Real<T> vNorm2 = Real<T>.Zero;
            for (int i = 0; i < colLen; i++) vNorm2 += vv[i] * vv[i];
            if (vNorm2.IsZero) continue;
            Real<T> beta = two / vNorm2;

            for (int c = k; c < n; c++)
            {
                Real<T> dot = Real<T>.Zero;
                for (int i = 0; i < colLen; i++) dot += vv[i] * h[(k + 1 + i) * n + c];
                Real<T> s = beta * dot;
                for (int i = 0; i < colLen; i++) h[(k + 1 + i) * n + c] -= s * vv[i];
            }
            for (int r = 0; r < n; r++)
            {
                Real<T> dot = Real<T>.Zero;
                for (int i = 0; i < colLen; i++) dot += vv[i] * h[r * n + (k + 1 + i)];
                Real<T> s = beta * dot;
                for (int i = 0; i < colLen; i++) h[r * n + (k + 1 + i)] -= s * vv[i];
            }
            for (int i = k + 2; i < n; i++) h[i * n + k] = Real<T>.Zero;
        }
    }

    /// <summary>
    /// Eigenvalues of an upper-Hessenberg matrix by the Francis double-shift QR algorithm with
    /// small-subdiagonal deflation and exceptional shifts (EISPACK <c>hqr</c>). Eigenvalue k sits at
    /// diagonal position k of the converged Schur form; the positive-imaginary member of a complex
    /// conjugate pair comes first, as in LAPACK <c>?geev</c>.
    /// </summary>
    private static (Real<T>[] Wr, Real<T>[] Wi) FrancisQr(Real<T>[] hIn, int nIn)
    {
        // 1-based working copy so the classic formulation can be followed index for index.
        int n = nIn;
        var a = new Real<T>[(n + 1) * (n + 1)];
        for (int i = 0; i < a.Length; i++) a[i] = Real<T>.Zero;
        int ld = n + 1;
        for (int i = 1; i <= n; i++)
            for (int j = 1; j <= n; j++)
                a[i * ld + j] = hIn[(i - 1) * n + (j - 1)];
        var wr = new Real<T>[n + 1];
        var wi = new Real<T>[n + 1];
        for (int i = 0; i <= n; i++) { wr[i] = Real<T>.Zero; wi[i] = Real<T>.Zero; }

        Real<T> zero = Real<T>.Zero;
        Real<T> half = Real<T>.One / Real<T>.Int(2);
        Real<T> threeQuarters = Real<T>.Int(3) / Real<T>.Int(4);
        Real<T> exShift = -(Real<T>.Int(7) / Real<T>.Int(16));

        Real<T> anorm = zero;
        for (int i = 1; i <= n; i++)
            for (int j = Math.Max(i - 1, 1); j <= n; j++)
                anorm += Real<T>.Abs(a[i * ld + j]);

        int nn = n;
        Real<T> t = zero;
        int maxIts = 30 * Math.Max(n, 10);
        // s, x, y, z and w are assigned before each read, so they start unset (an initial zero was never read).
        Real<T> p = zero, q = zero, r = zero, s, x, y, z, w;
        while (nn >= 1)
        {
            int its = 0;
            int l;
            do
            {
                for (l = nn; l >= 2; l--)
                {
                    s = Real<T>.Abs(a[(l - 1) * ld + (l - 1)]) + Real<T>.Abs(a[l * ld + l]);
                    if (s.IsZero) s = anorm;
                    if (Real<T>.Abs(a[l * ld + (l - 1)]) + s == s)
                    {
                        a[l * ld + (l - 1)] = zero;
                        break;
                    }
                }
                x = a[nn * ld + nn];
                if (l == nn)
                {
                    wr[nn] = x + t;
                    wi[nn] = zero;
                    nn--;
                }
                else
                {
                    y = a[(nn - 1) * ld + (nn - 1)];
                    w = a[nn * ld + (nn - 1)] * a[(nn - 1) * ld + nn];
                    if (l == nn - 1)
                    {
                        p = half * (y - x);
                        q = p * p + w;
                        z = Real<T>.Sqrt(Real<T>.Abs(q));
                        x += t;
                        if (q >= zero)
                        {
                            z = p + Real<T>.CopySign(z, p);
                            wr[nn - 1] = wr[nn] = x + z;
                            if (!z.IsZero) wr[nn] = x - w / z;
                            wi[nn - 1] = wi[nn] = zero;
                        }
                        else
                        {
                            wr[nn - 1] = wr[nn] = x + p;
                            wi[nn - 1] = z;
                            wi[nn] = -z;
                        }
                        nn -= 2;
                    }
                    else
                    {
                        if (its >= maxIts)
                            throw new ArithmeticException(
                                $"Eig: the QR algorithm failed to converge for {typeof(T).Name} input (LAPACK info > 0).");
                        if (its == 10 || its == 20)
                        {
                            t += x;
                            for (int i = 1; i <= nn; i++) a[i * ld + i] -= x;
                            s = Real<T>.Abs(a[nn * ld + (nn - 1)]) + Real<T>.Abs(a[(nn - 1) * ld + (nn - 2)]);
                            y = x = threeQuarters * s;
                            w = exShift * s * s;
                        }
                        ++its;
                        int m;
                        for (m = nn - 2; m >= l; m--)
                        {
                            z = a[m * ld + m];
                            r = x - z;
                            s = y - z;
                            p = (r * s - w) / a[(m + 1) * ld + m] + a[m * ld + (m + 1)];
                            q = a[(m + 1) * ld + (m + 1)] - z - r - s;
                            r = a[(m + 2) * ld + (m + 1)];
                            s = Real<T>.Abs(p) + Real<T>.Abs(q) + Real<T>.Abs(r);
                            p /= s;
                            q /= s;
                            r /= s;
                            if (m == l) break;
                            Real<T> u = Real<T>.Abs(a[m * ld + (m - 1)]) * (Real<T>.Abs(q) + Real<T>.Abs(r));
                            Real<T> vv = Real<T>.Abs(p) * (Real<T>.Abs(a[(m - 1) * ld + (m - 1)]) + Real<T>.Abs(z) +
                                Real<T>.Abs(a[(m + 1) * ld + (m + 1)]));
                            if (u + vv == vv) break;
                        }
                        for (int i = m + 2; i <= nn; i++)
                        {
                            a[i * ld + (i - 2)] = zero;
                            if (i != m + 2) a[i * ld + (i - 3)] = zero;
                        }
                        for (int k = m; k <= nn - 1; k++)
                        {
                            if (k != m)
                            {
                                p = a[k * ld + (k - 1)];
                                q = a[(k + 1) * ld + (k - 1)];
                                r = zero;
                                if (k != nn - 1) r = a[(k + 2) * ld + (k - 1)];
                                x = Real<T>.Abs(p) + Real<T>.Abs(q) + Real<T>.Abs(r);
                                if (!x.IsZero)
                                {
                                    p /= x;
                                    q /= x;
                                    r /= x;
                                }
                            }
                            s = Real<T>.CopySign(Real<T>.Sqrt(p * p + q * q + r * r), p);
                            if (!s.IsZero)
                            {
                                if (k == m)
                                {
                                    if (l != m) a[k * ld + (k - 1)] = -a[k * ld + (k - 1)];
                                }
                                else
                                {
                                    a[k * ld + (k - 1)] = -s * x;
                                }
                                p += s;
                                x = p / s;
                                y = q / s;
                                z = r / s;
                                q /= p;
                                r /= p;
                                for (int j = k; j <= nn; j++)
                                {
                                    p = a[k * ld + j] + q * a[(k + 1) * ld + j];
                                    if (k != nn - 1)
                                    {
                                        p += r * a[(k + 2) * ld + j];
                                        a[(k + 2) * ld + j] -= p * z;
                                    }
                                    a[(k + 1) * ld + j] -= p * y;
                                    a[k * ld + j] -= p * x;
                                }
                                int mmin = nn < k + 3 ? nn : k + 3;
                                for (int i = l; i <= mmin; i++)
                                {
                                    p = x * a[i * ld + k] + y * a[i * ld + (k + 1)];
                                    if (k != nn - 1)
                                    {
                                        p += z * a[i * ld + (k + 2)];
                                        a[i * ld + (k + 2)] -= p * r;
                                    }
                                    a[i * ld + (k + 1)] -= p * q;
                                    a[i * ld + k] -= p;
                                }
                            }
                        }
                    }
                }
            } while (l < nn - 1);
        }

        var outR = new Real<T>[n];
        var outI = new Real<T>[n];
        for (int i = 0; i < n; i++) { outR[i] = wr[i + 1]; outI[i] = wi[i + 1]; }
        return (outR, outI);
    }
    // Remove only vectors for the same eigenvalue; vectors for distinct eigenvalues of a non-normal
    // matrix need not be orthogonal. Two modified Gram-Schmidt passes preserve repeated eigenspaces.
    private static void ProjectRepeatedEigenvectors(Real<T>[] x, int n, bool complex, Real<T> re, Real<T> im,
        Real<T>[] wr, Real<T>[] wi, int count, T[] vectors, int offset)
    {
        Real<T> tol = Real<T>.Int(8) * Eps * (Real<T>.One + Real<T>.Abs(re) + Real<T>.Abs(im));
        for (int pass = 0; pass < 2; pass++)
            for (int k = 0; k < count; k++)
            {
                if (Real<T>.Abs(wr[k] - re) > tol || Real<T>.Abs(wi[k] - im) > tol) continue;
                Real<T> dotR = Real<T>.Zero, dotI = Real<T>.Zero, normSq = Real<T>.Zero;
                for (int i = 0; i < n; i++)
                {
                    Real<T> vr = vectors[offset + (i * n + k) * 2];
                    Real<T> vi = vectors[offset + (i * n + k) * 2 + 1];
                    Real<T> xi = complex ? x[i + n] : Real<T>.Zero;
                    dotR += vr * x[i] + vi * xi;
                    dotI += vr * xi - vi * x[i];
                    normSq += vr * vr + vi * vi;
                }
                if (normSq.IsZero) continue;
                dotR /= normSq; dotI /= normSq;
                for (int i = 0; i < n; i++)
                {
                    Real<T> vr = vectors[offset + (i * n + k) * 2];
                    Real<T> vi = vectors[offset + (i * n + k) * 2 + 1];
                    x[i] -= vr * dotR - vi * dotI;
                    if (complex) x[i + n] -= vr * dotI + vi * dotR;
                }
            }
    }

    private static Real<T> EigenResidual(Real<T>[] a, Real<T>[] x, int n, bool complex, Real<T> re, Real<T> im)
    {
        Real<T> residual = Real<T>.Zero;
        for (int i = 0; i < n; i++)
        {
            Real<T> xi = complex ? x[i + n] : Real<T>.Zero;
            Real<T> rr = -re * x[i] + im * xi;
            Real<T> ri = -im * x[i] - re * xi;
            for (int j = 0; j < n; j++)
            {
                rr += a[i * n + j] * x[j];
                if (complex) ri += a[i * n + j] * x[j + n];
            }
            residual = Real<T>.Max(residual, Real<T>.Abs(rr) + Real<T>.Abs(ri));
        }
        return residual;
    }



    /// <summary>
    /// Right eigenvector of <paramref name="a"/> for λ = re + i·im by inverse iteration with a shift
    /// √ε·scale off λ. The complex system (A − sI)(xr + i·xi) = b is solved as the equivalent 2n×2n real
    /// system. The result has unit 2-norm and, for complex λ, its largest component is made real.
    /// </summary>
    private static (Real<T>[] Xr, Real<T>[] Xi) InverseIteration(
        Real<T>[] a, int n, Real<T> re, Real<T> im, Real<T> scale,
        Real<T>[] wr, Real<T>[] wi, int eigenIndex, T[] previousVectors, int vectorOffset)
    {
        bool complex = !im.IsZero;
        int m = complex ? 2 * n : n;
        Real<T> delta = Real<T>.Sqrt(Eps) * scale;
        Real<T> shift = re >= Real<T>.Zero ? re + delta : re - delta;
        var mat = new Real<T>[m * m];
        for (int i = 0; i < mat.Length; i++) mat[i] = Real<T>.Zero;
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
            {
                Real<T> val = i == j ? a[i * n + j] - shift : a[i * n + j];
                mat[i * m + j] = val;
                if (complex) mat[(i + n) * m + (j + n)] = val;
            }
        if (complex)
            for (int i = 0; i < n; i++)
            {
                mat[i * m + (i + n)] = im;
                mat[(i + n) * m + i] = -im;
            }

        var piv = new int[m];
        LuInPlace(mat, piv, m, Eps * scale);

        // Try a basis of seeds: an all-ones seed can be exactly orthogonal to the desired eigenspace.
        // For a repeated eigenvalue, remove already found directions after every solve. Linear combinations
        // inside that eigenspace remain eigenvectors, so independent eigenvectors are not collapsed together.
        var x = new Real<T>[m];
        Real<T> bestResidual = Real<T>.Zero;
        bool haveBest = false;
        for (int seed = 0; seed < n; seed++)
        {
            var candidate = new Real<T>[m];
            for (int i = 0; i < m; i++) candidate[i] = i == seed ? Real<T>.One : Real<T>.Zero;
            bool usable = true;
            for (int iter = 0; iter < 3; iter++)
            {
                SolveInPlace(mat, piv, candidate, m);
                ProjectRepeatedEigenvectors(candidate, n, complex, re, im, wr, wi, eigenIndex, previousVectors, vectorOffset);
                Real<T> nrm = Norm2(candidate);
                if (nrm <= Eps) { usable = false; break; }
                for (int i = 0; i < m; i++) candidate[i] /= nrm;
            }
            if (!usable) continue;
            Real<T> residual = EigenResidual(a, candidate, n, complex, re, im);
            if (!haveBest || residual < bestResidual)
            {
                haveBest = true;
                bestResidual = residual;
                Array.Copy(candidate, x, m);
            }
        }
        // A defective matrix need not have another independent vector for a repeated eigenvalue.
        // Retain an ordinary inverse iterate in that case, as geev does, rather than returning a zero vector.
        if (!haveBest || bestResidual > Real<T>.Int(64) * Eps)
        {
            for (int seed = 0; seed < n; seed++)
            {
                var candidate = new Real<T>[m];
                for (int i = 0; i < m; i++) candidate[i] = i == seed ? Real<T>.One : Real<T>.Zero;
                for (int iter = 0; iter < 3; iter++)
                {
                    SolveInPlace(mat, piv, candidate, m);
                    Real<T> nrm = Norm2(candidate);
                    if (nrm.IsZero) break;
                    for (int i = 0; i < m; i++) candidate[i] /= nrm;
                }
                Real<T> residual = EigenResidual(a, candidate, n, complex, re, im);
                if (!haveBest || residual < bestResidual)
                {
                    haveBest = true;
                    bestResidual = residual;
                    Array.Copy(candidate, x, m);
                }
            }
        }

        var xr = new Real<T>[n];
        var xi = new Real<T>[n];
        for (int i = 0; i < n; i++) { xr[i] = x[i]; xi[i] = complex ? x[i + n] : Real<T>.Zero; }

        if (complex)
        {
            // Rotate by conj(z_max)/|z_max| so the largest component is real and positive.
            int best = 0;
            Real<T> bestMag = Real<T>.Zero;
            for (int i = 0; i < n; i++)
            {
                Real<T> mag = Real<T>.Hypot(xr[i], xi[i]);
                if (mag > bestMag) { bestMag = mag; best = i; }
            }
            if (!bestMag.IsZero)
            {
                Real<T> cr = xr[best] / bestMag, ci = -(xi[best] / bestMag);
                for (int i = 0; i < n; i++)
                {
                    Real<T> nr = xr[i] * cr - xi[i] * ci;
                    Real<T> ni = xr[i] * ci + xi[i] * cr;
                    xr[i] = nr;
                    xi[i] = ni;
                }
                xi[best] = Real<T>.Zero;
            }
        }

        var all = new Real<T>[2 * n];
        for (int i = 0; i < n; i++) { all[i] = xr[i]; all[i + n] = xi[i]; }
        Real<T> cnorm = Norm2(all);
        if (!cnorm.IsZero)
            for (int i = 0; i < n; i++) { xr[i] /= cnorm; xi[i] /= cnorm; }
        return (xr, xi);
    }

    /// <summary>Partial-pivot LU of a dense m×m scratch matrix; exact-zero pivots are replaced by
    /// <paramref name="tiny"/> (the standard inverse-iteration perturbation).</summary>
    private static void LuInPlace(Real<T>[] a, int[] piv, int m, Real<T> tiny)
    {
        if (tiny.IsZero) tiny = Eps;
        for (int col = 0; col < m; col++)
        {
            int p = col;
            Real<T> best = Real<T>.Abs(a[col * m + col]);
            for (int r = col + 1; r < m; r++)
            {
                Real<T> val = Real<T>.Abs(a[r * m + col]);
                if (val > best) { best = val; p = r; }
            }
            piv[col] = p;
            if (p != col)
                for (int j = 0; j < m; j++) (a[col * m + j], a[p * m + j]) = (a[p * m + j], a[col * m + j]);
            if (a[col * m + col].IsZero) a[col * m + col] = tiny;
            Real<T> d = a[col * m + col];
            for (int r = col + 1; r < m; r++)
            {
                Real<T> f = a[r * m + col] / d;
                a[r * m + col] = f;
                if (f.IsZero) continue;
                for (int j = col + 1; j < m; j++) a[r * m + j] -= f * a[col * m + j];
            }
        }
    }

    private static void SolveInPlace(Real<T>[] lu, int[] piv, Real<T>[] x, int m)
    {
        for (int i = 0; i < m; i++)
            if (piv[i] != i) (x[i], x[piv[i]]) = (x[piv[i]], x[i]);
        for (int i = 1; i < m; i++)
            for (int j = 0; j < i; j++) x[i] -= lu[i * m + j] * x[j];
        for (int i = m - 1; i >= 0; i--)
        {
            Real<T> s = x[i];
            for (int j = i + 1; j < m; j++) s -= lu[i * m + j] * x[j];
            x[i] = s / lu[i * m + i];
        }
    }

    // ── SVD ────────────────────────────────────────────────────────────────

    /// <summary>
    /// SVD of one m×n block: U (m×uCols), S (min(m,n), descending), Vᵀ (vRows×n).
    /// </summary>
    internal static void Svd(T[] src, int offSrc, T[] u, int offU, T[] s, int offS, T[] vh, int offVh,
        int m, int n, int uCols, int vRows)
    {
        EnsureSupported("SVD");
        int k = Math.Min(m, n);
        var a = Load(src, offSrc, m * n);
        if (m >= n)
        {
            var (uw, sw, vw) = SvdTall(a, m, n, uCols);
            for (int i = 0; i < m * uCols; i++) u[offU + i] = uw[i].V;
            for (int i = 0; i < k; i++) s[offS + i] = sw[i].V;
            // Vᵀ rows are the columns of V (n×n).
            for (int i = 0; i < vRows; i++)
                for (int j = 0; j < n; j++) vh[offVh + i * n + j] = vw[j * n + i].V;
        }
        else
        {
            // Aᵀ = U'·S·V'ᵀ  ⇒  A = V'·S·U'ᵀ: U = V' (m×m), Vᵀ = U'ᵀ.
            var at = new Real<T>[n * m];
            for (int i = 0; i < m; i++)
                for (int j = 0; j < n; j++) at[j * m + i] = a[i * n + j];
            var (uw, sw, vw) = SvdTall(at, n, m, vRows);
            for (int i = 0; i < m; i++)
                for (int j = 0; j < uCols; j++) u[offU + i * uCols + j] = vw[i * m + j].V;
            for (int i = 0; i < k; i++) s[offS + i] = sw[i].V;
            for (int i = 0; i < vRows; i++)
                for (int j = 0; j < n; j++) vh[offVh + i * n + j] = uw[j * vRows + i].V;
        }
    }

    /// <summary>
    /// One-sided Jacobi SVD for m ≥ n. Returns U (m×uCols, orthonormal columns completed for rank
    /// deficiency and full matrices), S (n, descending) and V (n×n, columns in the same order).
    /// </summary>
    private static (Real<T>[] U, Real<T>[] S, Real<T>[] V) SvdTall(Real<T>[] a, int m, int n, int uCols)
    {
        var w = (Real<T>[])a.Clone();
        var v = new Real<T>[n * n];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++) v[i * n + j] = i == j ? Real<T>.One : Real<T>.Zero;

        Real<T> tol = Eps * Real<T>.Int(Math.Max(m, 1));
        Real<T> one = Real<T>.One, two = Real<T>.Int(2);
        var colP = new Real<T>[m];
        var colQ = new Real<T>[m];
        for (int sweep = 0; sweep < 100; sweep++)
        {
            bool rotated = false;
            for (int p = 0; p < n - 1; p++)
            {
                for (int q = p + 1; q < n; q++)
                {
                    for (int i = 0; i < m; i++) { colP[i] = w[i * n + p]; colQ[i] = w[i * n + q]; }
                    Real<T> np = Norm2(colP), nq = Norm2(colQ);
                    if (np.IsZero || nq.IsZero) continue;
                    Real<T> gamma = Real<T>.Zero;
                    for (int i = 0; i < m; i++) gamma += (colP[i] / np) * (colQ[i] / nq);
                    // gamma is now the cosine between the columns; rotate unless already orthogonal.
                    if (Real<T>.Abs(gamma) <= tol) continue;
                    rotated = true;
                    Real<T> zeta = (nq / np - np / nq) / (two * gamma);
                    Real<T> absZeta = Real<T>.Abs(zeta);
                    Real<T> t;
                    if (zeta.IsZero) t = one;
                    else if (absZeta > one / Real<T>.Sqrt(Eps)) t = one / (two * zeta);
                    else t = Real<T>.CopySign(one, zeta) / (absZeta + Real<T>.Sqrt(one + zeta * zeta));
                    Real<T> c = one / Real<T>.Sqrt(one + t * t);
                    Real<T> sn = c * t;
                    for (int i = 0; i < m; i++)
                    {
                        w[i * n + p] = c * colP[i] - sn * colQ[i];
                        w[i * n + q] = sn * colP[i] + c * colQ[i];
                    }
                    for (int i = 0; i < n; i++)
                    {
                        Real<T> vp = v[i * n + p], vq = v[i * n + q];
                        v[i * n + p] = c * vp - sn * vq;
                        v[i * n + q] = sn * vp + c * vq;
                    }
                }
            }
            if (!rotated) break;
        }

        var sigma = new Real<T>[n];
        for (int j = 0; j < n; j++)
        {
            for (int i = 0; i < m; i++) colP[i] = w[i * n + j];
            sigma[j] = Norm2(colP);
        }
        var order = StableOrder(sigma, descending: true);
        Real<T> sMax = n > 0 ? sigma[order[0]] : Real<T>.Zero;
        Real<T> zeroCut = sMax * Eps * Real<T>.Int(Math.Max(m, n));

        var sOut = new Real<T>[n];
        var vOut = new Real<T>[n * n];
        var uOut = new Real<T>[m * uCols];
        for (int i = 0; i < uOut.Length; i++) uOut[i] = Real<T>.Zero;
        var filled = new bool[uCols];
        for (int j = 0; j < n; j++)
        {
            int src = order[j];
            sOut[j] = sigma[src];
            for (int i = 0; i < n; i++) vOut[i * n + j] = v[i * n + src];
            if (j < uCols && sigma[src] > zeroCut && !sigma[src].IsZero)
            {
                for (int i = 0; i < m; i++) uOut[i * uCols + j] = w[i * n + src] / sigma[src];
                filled[j] = true;
            }
        }
        CompleteOrthonormalColumns(uOut, m, uCols, filled);
        return (uOut, sOut, vOut);
    }

    /// <summary>
    /// Fills every column not marked in <paramref name="filled"/> with a unit vector orthogonal to all
    /// columns filled so far (modified Gram–Schmidt with re-orthogonalization over the standard basis).
    /// </summary>
    private static void CompleteOrthonormalColumns(Real<T>[] u, int m, int cols, bool[] filled)
    {
        var cand = new Real<T>[m];
        var best = new Real<T>[m];
        for (int j = 0; j < cols; j++)
        {
            if (filled[j]) continue;
            Real<T> bestNorm = Real<T>.Zero;
            for (int e = 0; e < m; e++)
            {
                for (int i = 0; i < m; i++) cand[i] = i == e ? Real<T>.One : Real<T>.Zero;
                for (int pass = 0; pass < 2; pass++)
                    for (int c = 0; c < cols; c++)
                    {
                        if (!filled[c]) continue;
                        Real<T> dot = Real<T>.Zero;
                        for (int i = 0; i < m; i++) dot += u[i * cols + c] * cand[i];
                        for (int i = 0; i < m; i++) cand[i] -= dot * u[i * cols + c];
                    }
                Real<T> nrm = Norm2(cand);
                if (nrm > bestNorm)
                {
                    bestNorm = nrm;
                    Array.Copy(cand, best, m);
                }
            }
            if (bestNorm.IsZero) continue;
            for (int i = 0; i < m; i++) u[i * cols + j] = best[i] / bestNorm;
            filled[j] = true;
        }
    }
}
