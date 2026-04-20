using System;

namespace AiDotNet.Tensors.LinearAlgebra.Solvers;

/// <summary>
/// Iterative Krylov-subspace solvers — <see cref="Linalg.CG"/>, <see cref="Linalg.GMRES"/>,
/// <see cref="Linalg.BiCGSTAB"/>. PyTorch ships none of these; users reach for
/// scipy. This ships them as first-class tensor ops.
///
/// <para>All three accept an optional <paramref name="preconditioner"/> delegate
/// that must return <c>M⁻¹·r</c>. Left preconditioning only for this first pass.
/// Non-batched (2D / 1D inputs) for v1; batched Krylov solvers are a follow-up
/// because the restart/convergence policy needs per-batch state.</para>
/// </summary>
internal static class IterativeSolvers
{
    internal static Tensor<T> CG<T>(Tensor<T> a, Tensor<T> b,
        int maxIter, double tol, Func<Tensor<T>, Tensor<T>>? precond)
        where T : unmanaged, IEquatable<T>, IComparable<T>
    {
        if (a.Rank != 2 || a.Shape[0] != a.Shape[1]) throw new ArgumentException("CG needs a 2D square matrix.");
        if (b.Rank != 1) throw new ArgumentException("CG needs a 1D RHS.");
        int n = a.Shape[0];

        var aD = a.GetDataArray();
        var bD = b.GetDataArray();

        // Start at x = 0.
        var x = new double[n];
        var r = new double[n];
        for (int i = 0; i < n; i++) r[i] = ToDouble(bD[i]);
        var z = precond is null ? (double[])r.Clone() : ApplyPrecond(precond, r);
        var p = (double[])z.Clone();

        double rzOld = Dot(r, z);
        double rrNorm0 = Math.Sqrt(Dot(r, r));

        for (int iter = 0; iter < maxIter; iter++)
        {
            var Ap = MatVec(aD, p, n);
            double pAp = Dot(p, Ap);
            if (pAp == 0) break;
            double alpha = rzOld / pAp;

            for (int i = 0; i < n; i++) x[i] += alpha * p[i];
            for (int i = 0; i < n; i++) r[i] -= alpha * Ap[i];
            if (Math.Sqrt(Dot(r, r)) / Math.Max(rrNorm0, 1e-30) < tol) break;

            z = precond is null ? (double[])r.Clone() : ApplyPrecond(precond, r);
            double rzNew = Dot(r, z);
            double beta = rzNew / rzOld;
            for (int i = 0; i < n; i++) p[i] = z[i] + beta * p[i];
            rzOld = rzNew;
        }

        var result = new Tensor<T>(new[] { n });
        var rdst = result.GetDataArray();
        for (int i = 0; i < n; i++) rdst[i] = FromDouble<T>(x[i]);
        return result;
    }

    internal static Tensor<T> GMRES<T>(Tensor<T> a, Tensor<T> b,
        int maxIter, int restart, double tol, Func<Tensor<T>, Tensor<T>>? precond)
        where T : unmanaged, IEquatable<T>, IComparable<T>
    {
        if (a.Rank != 2 || a.Shape[0] != a.Shape[1]) throw new ArgumentException("GMRES needs a 2D square matrix.");
        if (b.Rank != 1) throw new ArgumentException("GMRES needs a 1D RHS.");
        int n = a.Shape[0];

        var aD = a.GetDataArray();
        var bD = b.GetDataArray();

        var x = new double[n];
        double rrNorm0 = 0;
        for (int i = 0; i < n; i++) rrNorm0 += ToDouble(bD[i]) * ToDouble(bD[i]);
        rrNorm0 = Math.Sqrt(rrNorm0);
        if (rrNorm0 == 0) return new Tensor<T>(new[] { n });

        int total = 0;
        while (total < maxIter)
        {
            // Residual r = b − A·x.
            var Ax = MatVec(aD, x, n);
            var r = new double[n];
            for (int i = 0; i < n; i++) r[i] = ToDouble(bD[i]) - Ax[i];
            if (precond != null) r = ApplyPrecond(precond, r);

            double beta = Math.Sqrt(Dot(r, r));
            if (beta / rrNorm0 < tol) break;

            // Arnoldi iteration up to `restart`.
            int m = Math.Min(restart, maxIter - total);
            var V = new double[m + 1][];
            V[0] = new double[n];
            for (int i = 0; i < n; i++) V[0][i] = r[i] / beta;
            var H = new double[m + 1, m];

            int actual = m;
            for (int j = 0; j < m; j++)
            {
                var w = MatVec(aD, V[j], n);
                if (precond != null) w = ApplyPrecond(precond, w);
                for (int i = 0; i <= j; i++)
                {
                    H[i, j] = Dot(V[i], w);
                    for (int k = 0; k < n; k++) w[k] -= H[i, j] * V[i][k];
                }
                H[j + 1, j] = Math.Sqrt(Dot(w, w));
                if (H[j + 1, j] < 1e-14) { actual = j + 1; break; }
                V[j + 1] = new double[n];
                for (int k = 0; k < n; k++) V[j + 1][k] = w[k] / H[j + 1, j];
            }

            // Solve least-squares min || β·e1 − H·y || via Givens-based QR on the small Hessenberg.
            var y = SolveLeastSquaresHessenberg(H, beta, actual);
            for (int j = 0; j < actual; j++)
                for (int i = 0; i < n; i++) x[i] += y[j] * V[j][i];

            total += actual;
            double newResid = Residual(aD, x, bD, n);
            if (newResid / rrNorm0 < tol) break;
        }

        var result = new Tensor<T>(new[] { n });
        var rdst = result.GetDataArray();
        for (int i = 0; i < n; i++) rdst[i] = FromDouble<T>(x[i]);
        return result;
    }

    internal static Tensor<T> BiCGSTAB<T>(Tensor<T> a, Tensor<T> b,
        int maxIter, double tol, Func<Tensor<T>, Tensor<T>>? precond)
        where T : unmanaged, IEquatable<T>, IComparable<T>
    {
        if (a.Rank != 2 || a.Shape[0] != a.Shape[1]) throw new ArgumentException("BiCGSTAB needs a 2D square matrix.");
        if (b.Rank != 1) throw new ArgumentException("BiCGSTAB needs a 1D RHS.");
        int n = a.Shape[0];
        var aD = a.GetDataArray();
        var bD = b.GetDataArray();

        var x = new double[n];
        var r = new double[n];
        for (int i = 0; i < n; i++) r[i] = ToDouble(bD[i]);
        var rHat = (double[])r.Clone();
        var p = new double[n];
        var v = new double[n];
        double rho = 1, alpha = 1, omega = 1;
        double rrNorm0 = Math.Sqrt(Dot(r, r));
        if (rrNorm0 == 0) return new Tensor<T>(new[] { n });

        for (int iter = 0; iter < maxIter; iter++)
        {
            double rhoNew = Dot(rHat, r);
            if (rhoNew == 0) break;
            double beta = (rhoNew / rho) * (alpha / omega);
            for (int i = 0; i < n; i++) p[i] = r[i] + beta * (p[i] - omega * v[i]);

            var pHat = precond is null ? p : ApplyPrecond(precond, p);
            v = MatVec(aD, pHat, n);
            alpha = rhoNew / Dot(rHat, v);

            var s = new double[n];
            for (int i = 0; i < n; i++) s[i] = r[i] - alpha * v[i];
            if (Math.Sqrt(Dot(s, s)) / rrNorm0 < tol)
            {
                for (int i = 0; i < n; i++) x[i] += alpha * pHat[i];
                break;
            }

            var sHat = precond is null ? s : ApplyPrecond(precond, s);
            var t = MatVec(aD, sHat, n);
            omega = Dot(t, s) / Dot(t, t);
            for (int i = 0; i < n; i++) x[i] += alpha * pHat[i] + omega * sHat[i];
            for (int i = 0; i < n; i++) r[i] = s[i] - omega * t[i];
            if (Math.Sqrt(Dot(r, r)) / rrNorm0 < tol) break;
            rho = rhoNew;
        }

        var result = new Tensor<T>(new[] { n });
        var rdst = result.GetDataArray();
        for (int i = 0; i < n; i++) rdst[i] = FromDouble<T>(x[i]);
        return result;
    }

    // ── Kernel helpers ──────────────────────────────────────────────────────

    private static double[] MatVec<T>(T[] a, double[] x, int n)
    {
        var r = new double[n];
        for (int i = 0; i < n; i++)
        {
            double s = 0;
            for (int j = 0; j < n; j++) s += ToDouble(a[i * n + j]) * x[j];
            r[i] = s;
        }
        return r;
    }

    private static double Dot(double[] a, double[] b)
    {
        double s = 0;
        for (int i = 0; i < a.Length; i++) s += a[i] * b[i];
        return s;
    }

    private static double Residual<T>(T[] a, double[] x, T[] b, int n)
    {
        double s = 0;
        for (int i = 0; i < n; i++)
        {
            double ax = 0;
            for (int j = 0; j < n; j++) ax += ToDouble(a[i * n + j]) * x[j];
            double d = ToDouble(b[i]) - ax;
            s += d * d;
        }
        return Math.Sqrt(s);
    }

    private static double[] SolveLeastSquaresHessenberg(double[,] H, double beta, int m)
    {
        // Givens QR on the (m+1) × m Hessenberg, then back-solve for y.
        // Copy to working arrays.
        var h = new double[m + 1, m];
        for (int i = 0; i <= m; i++) for (int j = 0; j < m; j++) h[i, j] = H[i, j];
        var g = new double[m + 1];
        g[0] = beta;
        var cs = new double[m];
        var ss = new double[m];
        for (int k = 0; k < m; k++)
        {
            // Compute Givens rotation to zero h[k+1, k].
            double a = h[k, k], b = h[k + 1, k];
            double r = Math.Sqrt(a * a + b * b);
            if (r == 0) { cs[k] = 1; ss[k] = 0; continue; }
            cs[k] = a / r;
            ss[k] = b / r;
            // Apply to row k and k+1 of h.
            for (int j = k; j < m; j++)
            {
                double t1 = cs[k] * h[k, j] + ss[k] * h[k + 1, j];
                double t2 = -ss[k] * h[k, j] + cs[k] * h[k + 1, j];
                h[k, j] = t1;
                h[k + 1, j] = t2;
            }
            // Apply to g.
            double g1 = cs[k] * g[k] + ss[k] * g[k + 1];
            double g2 = -ss[k] * g[k] + cs[k] * g[k + 1];
            g[k] = g1;
            g[k + 1] = g2;
        }
        // Back-solve h[:m, :m] · y = g[:m].
        var y = new double[m];
        for (int i = m - 1; i >= 0; i--)
        {
            double s = g[i];
            for (int j = i + 1; j < m; j++) s -= h[i, j] * y[j];
            y[i] = h[i, i] == 0 ? 0 : s / h[i, i];
        }
        return y;
    }

    private static double[] ApplyPrecond<T>(Func<Tensor<T>, Tensor<T>> precond, double[] v)
        where T : unmanaged, IEquatable<T>, IComparable<T>
    {
        var input = new Tensor<T>(new[] { v.Length });
        var id = input.GetDataArray();
        for (int i = 0; i < v.Length; i++) id[i] = FromDouble<T>(v[i]);
        var result = precond(input);
        var rd = result.GetDataArray();
        var o = new double[v.Length];
        for (int i = 0; i < v.Length; i++) o[i] = ToDouble(rd[i]);
        return o;
    }

    private static double ToDouble<T>(T v)
    {
        if (typeof(T) == typeof(float)) return (float)(object)v!;
        if (typeof(T) == typeof(double)) return (double)(object)v!;
        throw new NotSupportedException();
    }

    private static T FromDouble<T>(double v)
    {
        if (typeof(T) == typeof(float)) return (T)(object)(float)v;
        if (typeof(T) == typeof(double)) return (T)(object)v;
        throw new NotSupportedException();
    }
}
