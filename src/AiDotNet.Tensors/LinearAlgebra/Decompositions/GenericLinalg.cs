using System;
using System.Collections.Generic;
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
///   back-substitution on the real Schur form (<c>hqr2</c>/<c>?trevc</c>) in O(n²) workspace, normalized to unit
///   2-norm with the largest component real (<c>?geev</c>). Used for decimal; Half and BFloat16 run the double
///   kernel in <see cref="EigDecomposition"/> and round the result.</item>
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

    // Halvings tried while searching for the machine epsilon: decimal's is 1e-28 (about 2^-93), the smallest of the
    // supported types, so the search always ends well before this.
    private const int EpsilonSearchLimit = 400;

    // Cyclic Jacobi sweeps (Eigh and the one-sided SVD) before the loop stops: Jacobi converges quadratically, so a
    // well-posed problem needs a handful.
    private const int MaxJacobiSweeps = 100;

    // A Jacobi loop that reaches MaxJacobiSweeps still rotating has converged to rounding level when its largest
    // remaining off-diagonal value (Eigh) or column cosine (SVD) is within this many times n times the rotation
    // threshold: in Half and BFloat16 each rotation reintroduces off-diagonal rounding of the threshold's order, so a
    // sweep can keep rotating there. Beyond it the decomposition has not converged and the loop throws.
    private const int JacobiStallFactor = 4;


    // Francis QR iterations allowed per eigenvalue (EISPACK hqr's 30 per eigenvalue, at least 10 eigenvalues' worth).
    private const int QrIterationsPerEigenvalue = 30;
    private const int MinQrEigenvalueBudget = 10;


    private static Real<T> MachineEpsilon()
    {
        if (!IsSupported) return Real<T>.Zero;
        Real<T> one = Real<T>.One, two = Real<T>.Int(2), e = one;
        for (int i = 0; i < EpsilonSearchLimit; i++)
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

        bool converged = false;
        Real<T> largestRotated = Real<T>.Zero;
        for (int sweep = 0; sweep < MaxJacobiSweeps; sweep++)
        {
            bool rotated = false;
            largestRotated = Real<T>.Zero;
            for (int p = 0; p < n - 1; p++)
            {
                for (int q = p + 1; q < n; q++)
                {
                    Real<T> apq = a[p * n + q];
                    if (Real<T>.Abs(apq) <= skip) continue;
                    rotated = true;
                    largestRotated = Real<T>.Max(largestRotated, Real<T>.Abs(apq));
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
            if (!rotated) { converged = true; break; }
        }
        if (!converged && largestRotated > Real<T>.Int(JacobiStallFactor * n) * skip)
            throw new ArithmeticException(
                $"Eigh: the Jacobi iteration failed to converge for {typeof(T).Name} input in {MaxJacobiSweeps} sweeps (LAPACK info > 0).");

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
    /// O(n²) working state of a general eigendecomposition in <typeparamref name="T"/>: H (the scaled input,
    /// then its Hessenberg and real Schur forms, then the triangular eigenvectors), V (the accumulated
    /// orthogonal transforms, then the eigenvectors) and O(n) vectors. One instance serves every matrix of
    /// a batch; nothing is allocated per QR iteration or per eigenvector.
    /// </summary>
    internal sealed class EigWorkspace
    {
        private readonly int _n;
        private readonly Real<T>[] _h, _v, _d, _e, _ort;

        internal EigWorkspace(int n)
        {
            _n = n;
            _h = new Real<T>[n * n];
            _v = new Real<T>[n * n];
            _d = new Real<T>[n];
            _e = new Real<T>[n];
            _ort = new Real<T>[n];
        }

        /// <summary>
        /// Eigenpairs of one general real n×n block. <paramref name="w"/> receives (re, im) pairs,
        /// <paramref name="v"/> the right eigenvectors as (re, im) columns — the layout of
        /// <see cref="EigDecomposition"/>.
        /// </summary>
        internal void Solve(T[] src, int offSrc, T[] w, int offW, T[] v, int offV)
        {
            int n = _n;
            if (n == 0) return;
            Real<T> scale = Real<T>.Zero;
            for (int i = 0; i < n * n; i++) scale = Real<T>.Max(scale, Real<T>.Abs(src[offSrc + i]));
            if (scale.IsZero) scale = Real<T>.One;
            // The QR products and the back-substitution operate on an O(1) matrix, not on squared input magnitudes.
            for (int i = 0; i < n * n; i++) _h[i] = (Real<T>)src[offSrc + i] / scale;

            Orthes();
            Hqr2();

            for (int k = 0; k < n; k++)
            {
                w[offW + k * 2] = (_d[k] * scale).V;
                w[offW + k * 2 + 1] = (_e[k] * scale).V;
            }
            WriteEigenvectors(v, offV);
        }

        /// <summary>Householder reduction to upper-Hessenberg form, accumulating the similarity in V (EISPACK orthes/ortran).</summary>
        private void Orthes()
        {
            int n = _n, high = n - 1;
            var h = _h; var v = _v; var ort = _ort;
            Real<T> zero = Real<T>.Zero;
            for (int m = 1; m <= high - 1; m++)
            {
                Real<T> scale = zero;
                for (int i = m; i <= high; i++) scale += Real<T>.Abs(h[i * n + (m - 1)]);
                if (scale.IsZero) continue;

                Real<T> hh = zero;
                for (int i = high; i >= m; i--)
                {
                    ort[i] = h[i * n + (m - 1)] / scale;
                    hh += ort[i] * ort[i];
                }
                Real<T> g = Real<T>.Sqrt(hh);
                if (ort[m] > zero) g = -g;
                hh -= ort[m] * g;
                ort[m] -= g;

                for (int j = m; j < n; j++)
                {
                    Real<T> f = zero;
                    for (int i = high; i >= m; i--) f += ort[i] * h[i * n + j];
                    f /= hh;
                    for (int i = m; i <= high; i++) h[i * n + j] -= f * ort[i];
                }
                for (int i = 0; i <= high; i++)
                {
                    Real<T> f = zero;
                    for (int j = high; j >= m; j--) f += ort[j] * h[i * n + j];
                    f /= hh;
                    for (int j = m; j <= high; j++) h[i * n + j] -= f * ort[j];
                }
                ort[m] = scale * ort[m];
                h[m * n + (m - 1)] = scale * g;
            }

            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                    v[i * n + j] = i == j ? Real<T>.One : zero;

            for (int m = high - 1; m >= 1; m--)
            {
                if (h[m * n + (m - 1)].IsZero) continue;
                for (int i = m + 1; i <= high; i++) ort[i] = h[i * n + (m - 1)];
                for (int j = m; j <= high; j++)
                {
                    Real<T> g = zero;
                    for (int i = m; i <= high; i++) g += ort[i] * v[i * n + j];
                    g = g / ort[m] / h[m * n + (m - 1)];
                    for (int i = m; i <= high; i++) v[i * n + j] += g * ort[i];
                }
            }
        }

        private static void Cdiv(Real<T> xr, Real<T> xi, Real<T> yr, Real<T> yi, out Real<T> cr, out Real<T> ci)
        {
            Real<T> r, d;
            if (Real<T>.Abs(yr) > Real<T>.Abs(yi))
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
        /// Rescales the partial eigenvector (rows <paramref name="from"/>..<paramref name="to"/> of columns
        /// <paramref name="c0"/>..<paramref name="c1"/>) when |num| / |den| would exceed 1/√ε, and returns the
        /// factor so the caller scales the right-hand sides it holds. Keeps a near-repeated eigenvalue's tiny
        /// divisor from overflowing types without infinity (decimal throws OverflowException).
        /// </summary>
        private Real<T> GuardQuotient(Real<T> num, Real<T> den, int from, int to, int c0, int c1)
        {
            Real<T> big = Real<T>.One / Real<T>.Sqrt(Eps);
            Real<T> limit = big * den;
            if (!(num > limit) || num.IsZero) return Real<T>.One;
            Real<T> f = limit / num;
            int n = _n;
            for (int j = from; j <= to; j++)
                for (int c = c0; c <= c1; c++)
                    _h[j * n + c] *= f;
            return f;
        }

        /// <summary>
        /// Francis double-shift QR down to real Schur form with the transforms accumulated in V, then
        /// back-substitution for the eigenvectors (EISPACK hqr2).
        /// </summary>
        private void Hqr2()
        {
            int nn = _n;
            int n = nn - 1;
            int high = nn - 1;
            var h = _h; var v = _v; var d = _d; var e = _e;
            Real<T> zero = Real<T>.Zero, one = Real<T>.One, two = Real<T>.Int(2);
            Real<T> threeQuarters = Real<T>.Int(3) / Real<T>.Int(4);
            Real<T> exShiftFactor = -(Real<T>.Int(7) / Real<T>.Int(16));
            Real<T> matlabShift = Real<T>.Int(964) / Real<T>.Int(1000);
            Real<T> exshift = zero;
            Real<T> p = zero, q = zero, r = zero, s = zero, z = zero, t, w, x, y;

            Real<T> norm = zero;
            for (int i = 0; i < nn; i++)
                for (int j = Math.Max(i - 1, 0); j < nn; j++)
                    norm += Real<T>.Abs(h[i * nn + j]);

            Real<T> deflationFloor = Real<T>.Int(nn) * Eps * norm;
            int maxIter = QrIterationsPerEigenvalue * Math.Max(nn, MinQrEigenvalueBudget);
            int iter = 0, totalIter = 0;
            while (n >= 0)
            {
                int l = n;
                while (l > 0)
                {
                    s = Real<T>.Abs(h[(l - 1) * nn + (l - 1)]) + Real<T>.Abs(h[l * nn + l]);
                    if (s.IsZero) s = norm;
                    // decimal is fixed-point below 1: nothing smaller than 1e-28 exists, and a sub-diagonal
                    // stagnates at a few units of that. Once s < 1, ε·s drops under that granularity (or to zero),
                    // so the relative test alone can never deflate; the absolute floor n·ε·‖H‖ (LAPACK's smlnum
                    // role) keeps the deflation normwise backward stable and reachable.
                    Real<T> sub = Real<T>.Abs(h[l * nn + (l - 1)]);
                    if (sub + s == s || sub < Eps * s || sub <= deflationFloor) break;
                    l--;
                }

                if (l == n)
                {
                    h[n * nn + n] += exshift;
                    d[n] = h[n * nn + n];
                    e[n] = zero;
                    n--;
                    iter = 0;
                }
                else if (l == n - 1)
                {
                    w = h[n * nn + (n - 1)] * h[(n - 1) * nn + n];
                    p = (h[(n - 1) * nn + (n - 1)] - h[n * nn + n]) / two;
                    q = p * p + w;
                    z = Real<T>.Sqrt(Real<T>.Abs(q));
                    h[n * nn + n] += exshift;
                    h[(n - 1) * nn + (n - 1)] += exshift;
                    x = h[n * nn + n];

                    if (q >= zero)
                    {
                        z = p >= zero ? p + z : p - z;
                        d[n - 1] = x + z;
                        d[n] = d[n - 1];
                        if (!z.IsZero) d[n] = x - w / z;
                        e[n - 1] = zero;
                        e[n] = zero;
                        x = h[n * nn + (n - 1)];
                        s = Real<T>.Abs(x) + Real<T>.Abs(z);
                        p = x / s;
                        q = z / s;
                        r = Real<T>.Sqrt(p * p + q * q);
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
                        for (int i = 0; i <= high; i++)
                        {
                            z = v[i * nn + (n - 1)];
                            v[i * nn + (n - 1)] = q * z + p * v[i * nn + n];
                            v[i * nn + n] = q * v[i * nn + n] - p * z;
                        }
                    }
                    else
                    {
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
                    x = h[n * nn + n];
                    y = zero;
                    w = zero;
                    if (l < n)
                    {
                        y = h[(n - 1) * nn + (n - 1)];
                        w = h[n * nn + (n - 1)] * h[(n - 1) * nn + n];
                    }

                    if (iter == 10)
                    {
                        exshift += x;
                        for (int i = 0; i <= n; i++) h[i * nn + i] -= x;
                        s = Real<T>.Abs(h[n * nn + (n - 1)]) + Real<T>.Abs(h[(n - 1) * nn + (n - 2)]);
                        x = y = threeQuarters * s;
                        w = exShiftFactor * s * s;
                    }

                    if (iter == 30)
                    {
                        s = (y - x) / two;
                        s = s * s + w;
                        if (s > zero)
                        {
                            s = Real<T>.Sqrt(s);
                            if (y < x) s = -s;
                            s = x - w / ((y - x) / two + s);
                            for (int i = 0; i <= n; i++) h[i * nn + i] -= s;
                            exshift += s;
                            x = y = w = matlabShift;
                        }
                    }

                    iter++;
                    if (++totalIter > maxIter)
                        throw new ArithmeticException(
                            $"Eig: the QR algorithm failed to converge for {typeof(T).Name} input (LAPACK info > 0).");

                    int m = n - 2;
                    while (m >= l)
                    {
                        z = h[m * nn + m];
                        r = x - z;
                        s = y - z;
                        p = (r * s - w) / h[(m + 1) * nn + m] + h[m * nn + (m + 1)];
                        q = h[(m + 1) * nn + (m + 1)] - z - r - s;
                        r = h[(m + 2) * nn + (m + 1)];
                        s = Real<T>.Abs(p) + Real<T>.Abs(q) + Real<T>.Abs(r);
                        p /= s;
                        q /= s;
                        r /= s;
                        if (m == l) break;
                        if (Real<T>.Abs(h[m * nn + (m - 1)]) * (Real<T>.Abs(q) + Real<T>.Abs(r)) <
                            Eps * (Real<T>.Abs(p) * (Real<T>.Abs(h[(m - 1) * nn + (m - 1)]) + Real<T>.Abs(z) +
                                                     Real<T>.Abs(h[(m + 1) * nn + (m + 1)]))))
                            break;
                        m--;
                    }

                    for (int i = m + 2; i <= n; i++)
                    {
                        h[i * nn + (i - 2)] = zero;
                        if (i > m + 2) h[i * nn + (i - 3)] = zero;
                    }

                    for (int k = m; k <= n - 1; k++)
                    {
                        bool notlast = k != n - 1;
                        if (k != m)
                        {
                            p = h[k * nn + (k - 1)];
                            q = h[(k + 1) * nn + (k - 1)];
                            r = notlast ? h[(k + 2) * nn + (k - 1)] : zero;
                            x = Real<T>.Abs(p) + Real<T>.Abs(q) + Real<T>.Abs(r);
                            if (x.IsZero) continue;
                            p /= x;
                            q /= x;
                            r /= x;
                        }

                        s = Real<T>.Sqrt(p * p + q * q + r * r);
                        if (p < zero) s = -s;
                        if (s.IsZero) continue;

                        if (k != m) h[k * nn + (k - 1)] = -s * x;
                        else if (l != m) h[k * nn + (k - 1)] = -h[k * nn + (k - 1)];
                        p += s;
                        x = p / s;
                        y = q / s;
                        z = r / s;
                        q /= p;
                        r /= p;

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

                        for (int i = 0; i <= high; i++)
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
            if (norm.IsZero) return;

            Real<T> tiny = Eps * norm;
            for (n = nn - 1; n >= 0; n--)
            {
                p = d[n];
                q = e[n];

                if (q.IsZero)
                {
                    int l = n;
                    h[n * nn + n] = one;
                    for (int i = n - 1; i >= 0; i--)
                    {
                        w = h[i * nn + i] - p;
                        r = zero;
                        for (int j = l; j <= n; j++) r += h[i * nn + j] * h[j * nn + n];
                        if (e[i] < zero)
                        {
                            z = w;
                            s = r;
                        }
                        else
                        {
                            l = i;
                            if (e[i].IsZero)
                            {
                                Real<T> den = w.IsZero ? tiny : w;
                                Real<T> f = GuardQuotient(Real<T>.Abs(r), Real<T>.Abs(den), i + 1, n, n, n);
                                r *= f;
                                h[i * nn + n] = -r / den;
                            }
                            else
                            {
                                x = h[i * nn + (i + 1)];
                                y = h[(i + 1) * nn + i];
                                q = (d[i] - p) * (d[i] - p) + e[i] * e[i];
                                t = (x * s - z * r) / q;
                                h[i * nn + n] = t;
                                h[(i + 1) * nn + n] = Real<T>.Abs(x) > Real<T>.Abs(z) ? (-r - w * t) / x : (-s - y * t) / z;
                            }

                            // Overflow control; (ε·t)·t keeps the product itself in range.
                            t = Real<T>.Abs(h[i * nn + n]);
                            if (Eps * t * t > one)
                                for (int j = i; j <= n; j++) h[j * nn + n] /= t;
                        }
                    }
                }
                else if (q < zero)
                {
                    int l = n - 1;
                    if (Real<T>.Abs(h[n * nn + (n - 1)]) > Real<T>.Abs(h[(n - 1) * nn + n]))
                    {
                        h[(n - 1) * nn + (n - 1)] = q / h[n * nn + (n - 1)];
                        h[(n - 1) * nn + n] = -(h[n * nn + n] - p) / h[n * nn + (n - 1)];
                    }
                    else
                    {
                        Cdiv(zero, -h[(n - 1) * nn + n], h[(n - 1) * nn + (n - 1)] - p, q, out var cr, out var ci);
                        h[(n - 1) * nn + (n - 1)] = cr;
                        h[(n - 1) * nn + n] = ci;
                    }
                    h[n * nn + (n - 1)] = zero;
                    h[n * nn + n] = one;
                    for (int i = n - 2; i >= 0; i--)
                    {
                        Real<T> ra = zero, sa = zero;
                        for (int j = l; j <= n; j++)
                        {
                            ra += h[i * nn + j] * h[j * nn + (n - 1)];
                            sa += h[i * nn + j] * h[j * nn + n];
                        }
                        w = h[i * nn + i] - p;

                        if (e[i] < zero)
                        {
                            z = w;
                            r = ra;
                            s = sa;
                        }
                        else
                        {
                            l = i;
                            if (e[i].IsZero)
                            {
                                Real<T> f = GuardQuotient(Real<T>.Max(Real<T>.Abs(ra), Real<T>.Abs(sa)),
                                    Real<T>.Max(Real<T>.Abs(w), Real<T>.Abs(q)), i + 1, n, n - 1, n);
                                ra *= f;
                                sa *= f;
                                Cdiv(-ra, -sa, w, q, out var cr, out var ci);
                                h[i * nn + (n - 1)] = cr;
                                h[i * nn + n] = ci;
                            }
                            else
                            {
                                x = h[i * nn + (i + 1)];
                                y = h[(i + 1) * nn + i];
                                Real<T> vr = (d[i] - p) * (d[i] - p) + e[i] * e[i] - q * q;
                                Real<T> vi = (d[i] - p) * two * q;
                                if (vr.IsZero && vi.IsZero)
                                    vr = tiny * (Real<T>.Abs(w) + Real<T>.Abs(q) + Real<T>.Abs(x) + Real<T>.Abs(y) + Real<T>.Abs(z));
                                Real<T> nr = x * r - z * ra + q * sa, ni = x * s - z * sa - q * ra;
                                // The partial vector below row i+1 is shared by this row pair, so the saved r and s scale too.
                                Real<T> f = GuardQuotient(Real<T>.Max(Real<T>.Abs(nr), Real<T>.Abs(ni)),
                                    Real<T>.Max(Real<T>.Abs(vr), Real<T>.Abs(vi)), i + 2, n, n - 1, n);
                                if (f != one)
                                {
                                    ra *= f; sa *= f; r *= f; s *= f;
                                    nr = x * r - z * ra + q * sa;
                                    ni = x * s - z * sa - q * ra;
                                }
                                Cdiv(nr, ni, vr, vi, out var cr, out var ci);
                                h[i * nn + (n - 1)] = cr;
                                h[i * nn + n] = ci;
                                if (Real<T>.Abs(x) > Real<T>.Abs(z) + Real<T>.Abs(q))
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

                            t = Real<T>.Max(Real<T>.Abs(h[i * nn + (n - 1)]), Real<T>.Abs(h[i * nn + n]));
                            if (Eps * t * t > one)
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

            // V ← V·T: Schur-form eigenvectors back to eigenvectors of the input.
            for (int j = nn - 1; j >= 0; j--)
            {
                for (int i = 0; i <= high; i++)
                {
                    z = zero;
                    int kMax = Math.Min(j, high);
                    for (int k = 0; k <= kMax; k++) z += v[i * nn + k] * h[k * nn + j];
                    v[i * nn + j] = z;
                }
            }
        }

        /// <summary>
        /// Writes V as (re, im) columns with unit 2-norm and, for a complex vector, a real positive component
        /// of largest modulus (?geev). Norms are scaled by the largest modulus so squares cannot overflow.
        /// </summary>
        private void WriteEigenvectors(T[] dst, int off)
        {
            int n = _n;
            var v = _v;
            Real<T> zero = Real<T>.Zero;
            int k = 0;
            while (k < n)
            {
                bool complex = _e[k] > zero && k + 1 < n;
                int best = 0;
                Real<T> bestMag = zero;
                for (int i = 0; i < n; i++)
                {
                    Real<T> mag = complex ? Real<T>.Hypot(v[i * n + k], v[i * n + k + 1]) : Real<T>.Abs(v[i * n + k]);
                    if (mag > bestMag) { bestMag = mag; best = i; }
                }
                Real<T> ss = zero;
                if (!bestMag.IsZero)
                    for (int i = 0; i < n; i++)
                    {
                        Real<T> a = v[i * n + k] / bestMag;
                        ss += a * a;
                        if (complex)
                        {
                            Real<T> b = v[i * n + k + 1] / bestMag;
                            ss += b * b;
                        }
                    }
                // x / ‖x‖ = (x / |x_max|) / sqrt(ss); for complex x also multiply by conj(x_max) / |x_max|.
                Real<T> norm = bestMag.IsZero ? Real<T>.One : Real<T>.Sqrt(ss);
                Real<T> cr = Real<T>.One, ci = zero;
                if (complex && !bestMag.IsZero)
                {
                    cr = v[best * n + k] / bestMag;
                    ci = -(v[best * n + k + 1] / bestMag);
                }
                Real<T> scale = bestMag.IsZero ? Real<T>.One : bestMag;
                for (int i = 0; i < n; i++)
                {
                    Real<T> a = v[i * n + k] / scale;
                    Real<T> b = complex ? v[i * n + k + 1] / scale : zero;
                    Real<T> re = (a * cr - b * ci) / norm, im = (a * ci + b * cr) / norm;
                    if (complex && i == best) im = zero;
                    dst[off + (i * n + k) * 2] = re.V;
                    dst[off + (i * n + k) * 2 + 1] = im.V;
                    if (complex)
                    {
                        dst[off + (i * n + k + 1) * 2] = re.V;
                        dst[off + (i * n + k + 1) * 2 + 1] = (-im).V;
                    }
                }
                k += complex ? 2 : 1;
            }
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
        Real<T> cancelTol = Real<T>.Int(2) * Real<T>.Sqrt(Real<T>.Int(Math.Max(4, n))) * Eps;
        Real<T> one = Real<T>.One, two = Real<T>.Int(2);
        var absA = new Real<T>[m * n];
        for (int i = 0; i < absA.Length; i++) absA[i] = Real<T>.Abs(a[i]);
        var colP = new Real<T>[m];
        var colQ = new Real<T>[m];
        bool converged = false;
        Real<T> largestCosine = Real<T>.Zero;
        for (int sweep = 0; sweep < MaxJacobiSweeps; sweep++)
        {
            bool rotated = false;
            largestCosine = Real<T>.Zero;
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
                    largestCosine = Real<T>.Max(largestCosine, Real<T>.Abs(gamma));
                    // The smaller-to-larger norm ratio r lies in [0, 1]. Below sqrt(eps), |zeta| is huge and t = 1/(2 zeta) =
                    // ±gamma r / (1 - r²), formed from r: the quotient of a normal by a subnormal norm overflowed to
                    // infinity, t became -0, the rotation did nothing and the sweep limit threw.
                    Real<T> ratio = np <= nq ? np / nq : nq / np;
                    Real<T> t;
                    if (ratio < Real<T>.Sqrt(Eps))
                    {
                        t = gamma * ratio / (one - ratio * ratio);
                        if (np > nq) t = -t;
                    }
                    else
                    {
                        Real<T> zeta = (nq / np - np / nq) / (two * gamma);
                        Real<T> absZeta = Real<T>.Abs(zeta);
                        if (zeta.IsZero) t = one;
                        else if (absZeta > one / Real<T>.Sqrt(Eps)) t = one / (two * zeta);
                        else t = Real<T>.CopySign(one, zeta) / (absZeta + Real<T>.Sqrt(one + zeta * zeta));
                    }
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
            if (!rotated) { converged = true; break; }
            DeflateCancelledColumns(a, absA, w, v, m, n, cancelTol);
        }
        if (!converged && largestCosine > Real<T>.Int(JacobiStallFactor * n) * tol)
            throw new ArithmeticException(
                $"Svd: the Jacobi iteration failed to converge for {typeof(T).Name} input in {MaxJacobiSweeps} sweeps (LAPACK info > 0).");

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
    /// Zeroes any working column <c>w[:, j]</c> whose fresh residual <c>A · V[:, j]</c> satisfies the
    /// componentwise Oettli-Prager bound <c>|(A · V)[i, j]| &lt;= cancelTol · (|A| · |V|)[i, j]</c> in every row.
    /// In rank-deficient inputs with nullity &gt;= 2, two cancelled nullspace columns otherwise retain pure
    /// rounding residuals whose normalized inner product stays O(1), cycling forever across sweeps. Testing
    /// the fresh entrywise relative bound against <c>|A| · |V|</c> preserves both row-graded and column-graded
    /// tiny singular values down to subnormal scales.
    /// </summary>
    private static void DeflateCancelledColumns(
        Real<T>[] a, Real<T>[] absA, Real<T>[] w, Real<T>[] v, int m, int n, Real<T> cancelTol)
    {
        for (int j = 0; j < n; j++)
        {
            bool anyNonZero = false;
            for (int i = 0; i < m; i++)
            {
                if (!w[i * n + j].IsZero)
                {
                    anyNonZero = true;
                    break;
                }
            }
            if (!anyNonZero) continue;

            bool cancelled = true;
            for (int i = 0; i < m; i++)
            {
                Real<T> sum = Real<T>.Zero;
                Real<T> scale = Real<T>.Zero;
                int rowOff = i * n;
                for (int k = 0; k < n; k++)
                {
                    Real<T> vkj = v[k * n + j];
                    sum += a[rowOff + k] * vkj;
                    scale += absA[rowOff + k] * Real<T>.Abs(vkj);
                }
                if (Real<T>.Abs(sum) > cancelTol * scale || Real<T>.Abs(w[rowOff + j]) > cancelTol * scale)
                {
                    cancelled = false;
                    break;
                }
            }

            if (cancelled)
            {
                for (int i = 0; i < m; i++) w[i * n + j] = Real<T>.Zero;
            }
        }
    }

    /// <summary>
    /// Fills every column not marked in <paramref name="filled"/> with a unit vector orthogonal to the filled columns:
    /// the trailing columns of Q from the Householder QR of the filled columns (LAPACK's orgqr completion).
    /// </summary>
    /// <remarks>
    /// O(m²·k) for k filled columns. It replaced trying every standard basis vector for every missing column, which
    /// was about 2m⁴ operations (2·10¹² for a full-matrices SVD of a 1000 x 2 input).
    /// </remarks>
    private static void CompleteOrthonormalColumns(Real<T>[] u, int m, int cols, bool[] filled)
    {
        var known = new List<int>();
        var missing = new List<int>();
        for (int j = 0; j < cols; j++) (filled[j] ? known : missing).Add(j);
        if (missing.Count == 0) return;
        int k = known.Count;

        // Householder QR of the m x k matrix of filled columns; reflector c acts on rows c..m-1.
        var f = new Real<T>[m * k];
        for (int i = 0; i < m; i++)
            for (int c = 0; c < k; c++) f[i * k + c] = u[i * cols + known[c]];
        var reflectors = new Real<T>[]?[Math.Min(k, m)];
        for (int c = 0; c < reflectors.Length; c++)
        {
            var x = new Real<T>[m - c];
            for (int i = c; i < m; i++) x[i - c] = f[i * k + c];
            Real<T> norm = Norm2(x);
            if (norm.IsZero) continue;
            x[0] += Real<T>.CopySign(norm, x[0]);
            Real<T> vNorm = Norm2(x);
            if (vNorm.IsZero) continue;
            for (int i = 0; i < x.Length; i++) x[i] /= vNorm;
            reflectors[c] = x;
            for (int col = c + 1; col < k; col++)
            {
                Real<T> dot = Real<T>.Zero;
                for (int i = c; i < m; i++) dot += x[i - c] * f[i * k + col];
                Real<T> twice = dot + dot;
                for (int i = c; i < m; i++) f[i * k + col] -= twice * x[i - c];
            }
        }

        // Q e_(k + t) for the t-th missing column: apply H_(k-1) first, H_0 last.
        var y = new Real<T>[m];
        for (int t = 0; t < missing.Count && k + t < m; t++)
        {
            for (int i = 0; i < m; i++) y[i] = i == k + t ? Real<T>.One : Real<T>.Zero;
            for (int c = reflectors.Length - 1; c >= 0; c--)
            {
                var v = reflectors[c];
                if (v is null) continue;
                Real<T> dot = Real<T>.Zero;
                for (int i = c; i < m; i++) dot += v[i - c] * y[i];
                Real<T> twice = dot + dot;
                for (int i = c; i < m; i++) y[i] -= twice * v[i - c];
            }
            int j = missing[t];
            for (int i = 0; i < m; i++) u[i * cols + j] = y[i];
            filled[j] = true;
        }
    }
}
