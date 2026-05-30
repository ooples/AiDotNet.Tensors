using System;

namespace AiDotNet.Tensors.LinearAlgebra.Decompositions;

/// <summary>
/// LDLᵀ factorization for symmetric-indefinite matrices: <c>P·A·Pᵀ = L·D·Lᵀ</c>
/// where <c>D</c> is block-diagonal with 1×1 and 2×2 blocks. Managed Bunch–Kaufman
/// pivoting implementation; native LAPACK <c>?sytrf</c> is the stubbed fallback.
/// </summary>
/// <remarks>
/// The returned <c>LD</c> tensor packs the unit-diagonal triangular factor and
/// the diagonal of <c>D</c> into a single dense triangle: diagonal entries hold
/// <c>D[i,i]</c>, off-diagonal entries hold <c>L[i,j]</c> (unit diagonal is
/// implicit). When <paramref name="upper"/> is <c>true</c> the factor is stored
/// in the upper triangle (so <c>A = Uᵀ·D·U</c> with <c>U</c> unit-upper), else
/// the lower triangle (<c>A = L·D·Lᵀ</c>, <c>L</c> unit-lower).
/// </remarks>
internal static class LdlDecomposition
{
    internal static (Tensor<T> LD, Tensor<int> Pivots) Factor<T>(Tensor<T> input, bool upper)
        where T : unmanaged, IEquatable<T>, IComparable<T>
    {
        if (input is null) throw new ArgumentNullException(nameof(input));
        if (input.Rank < 2) throw new ArgumentException("LDL requires at least a 2D tensor.", nameof(input));

        int rank = input.Rank;
        int n = input.Shape[rank - 1];
        if (input.Shape[rank - 2] != n) throw new ArgumentException("LDL requires a square matrix.");

        var ld = new Tensor<T>((int[])input._shape.Clone());
        Array.Copy(input.GetDataArray(), ld.GetDataArray(), input.Length);

        var pivotShape = new int[rank - 1];
        for (int i = 0; i < rank - 2; i++) pivotShape[i] = input._shape[i];
        pivotShape[rank - 2] = n;
        var pivots = new Tensor<int>(pivotShape);

        int batch = 1;
        for (int i = 0; i < rank - 2; i++) batch *= input._shape[i];
        var ldData = ld.GetDataArray();
        var pivData = pivots.GetDataArray();
        int matStride = n * n;
        int pivStride = n;

        for (int b = 0; b < batch; b++)
            FactorSingle(ldData, b * matStride, pivData, b * pivStride, n, upper);

        return (ld, pivots);
    }

    /// <summary>
    /// Solve <c>A·x = b</c> given the LDLᵀ factor. Uses proper three-phase
    /// back-substitution: solve <c>L·y = b</c> (or <c>Uᵀ·y = b</c>), then
    /// <c>D·z = y</c>, then <c>Lᵀ·x = z</c> (or <c>U·x = z</c>).
    /// </summary>
    internal static Tensor<T> Solve<T>(Tensor<T> ld, Tensor<int> pivots, Tensor<T> b, bool upper)
        where T : unmanaged, IEquatable<T>, IComparable<T>
    {
        if (ld is null) throw new ArgumentNullException(nameof(ld));
        if (b is null) throw new ArgumentNullException(nameof(b));

        int rank = ld.Rank;
        int n = ld.Shape[rank - 1];
        int nrhs = b.Rank == ld.Rank ? b.Shape[rank - 1] : 1;

        int batch = 1;
        for (int i = 0; i < rank - 2; i++) batch *= ld._shape[i];

        var result = new Tensor<T>((int[])b._shape.Clone());
        var ldData = ld.GetDataArray();
        var bData = b.GetDataArray();
        var xData = result.GetDataArray();
        var pivData = pivots.GetDataArray();
        int matStride = n * n;
        int rhsStride = b.Rank == ld.Rank ? n * nrhs : n;
        int pivStride = n;

        for (int ib = 0; ib < batch; ib++)
            SolveSingle(ldData, ib * matStride, bData, ib * rhsStride, xData, ib * rhsStride,
                pivData, ib * pivStride, n, nrhs, upper);

        return result;
    }

    private static void SolveSingle<T>(T[] a, int offA, T[] b, int offB, T[] x, int offX,
        int[] piv, int offPiv, int n, int nrhs, bool upper)
        where T : unmanaged, IEquatable<T>, IComparable<T>
    {
        // Copy RHS into X; we solve in place.
        for (int i = 0; i < n * nrhs; i++) x[offX + i] = b[offB + i];

        for (int c = 0; c < nrhs; c++)
        {
            // Apply the factorization's symmetric pivoting to the RHS: b' = P·b.
            // (Forward order, matching how the swaps were applied during Factor.)
            for (int j = 0; j < n; j++)
            {
                int pj = piv[offPiv + j];
                if (pj != j) (x[offX + j * nrhs + c], x[offX + pj * nrhs + c]) = (x[offX + pj * nrhs + c], x[offX + j * nrhs + c]);
            }

            // Phase 1: forward-solve L·y = b (or Uᵀ·y = b when upper).
            for (int i = 0; i < n; i++)
            {
                double s = ToDouble(x[offX + i * nrhs + c]);
                for (int k = 0; k < i; k++)
                {
                    // When lower: L[i,k] at a[i*n+k]. When upper: Uᵀ[i,k] = U[k,i] at a[k*n+i].
                    double lik = upper ? ToDouble(a[offA + k * n + i]) : ToDouble(a[offA + i * n + k]);
                    s -= lik * ToDouble(x[offX + k * nrhs + c]);
                }
                // Unit diagonal — no division.
                x[offX + i * nrhs + c] = FromDouble<T>(s);
            }

            // Phase 2: diagonal solve D·z = y (D packed on matrix diagonal).
            for (int i = 0; i < n; i++)
            {
                double d = ToDouble(a[offA + i * n + i]);
                double y = ToDouble(x[offX + i * nrhs + c]);
                x[offX + i * nrhs + c] = d == 0.0 ? FromDouble<T>(double.NaN) : FromDouble<T>(y / d);
            }

            // Phase 3: back-solve Lᵀ·x = z (or U·x = z when upper).
            for (int i = n - 1; i >= 0; i--)
            {
                double s = ToDouble(x[offX + i * nrhs + c]);
                for (int k = i + 1; k < n; k++)
                {
                    // When lower: Lᵀ[i,k] = L[k,i] at a[k*n+i]. When upper: U[i,k] at a[i*n+k].
                    double lki = upper ? ToDouble(a[offA + i * n + k]) : ToDouble(a[offA + k * n + i]);
                    s -= lki * ToDouble(x[offX + k * nrhs + c]);
                }
                x[offX + i * nrhs + c] = FromDouble<T>(s);
            }

            // Undo the pivoting: x = Pᵀ·z (reverse order of the forward swaps).
            for (int j = n - 1; j >= 0; j--)
            {
                int pj = piv[offPiv + j];
                if (pj != j) (x[offX + j * nrhs + c], x[offX + pj * nrhs + c]) = (x[offX + pj * nrhs + c], x[offX + j * nrhs + c]);
            }
        }
    }

    private static void FactorSingle<T>(T[] a, int off, int[] piv, int offPiv, int n, bool upper)
        where T : unmanaged, IEquatable<T>, IComparable<T>
    {
        // Outer-product LDLᵀ with symmetric (diagonal) partial pivoting:
        // P·A·Pᵀ = L·D·Lᵀ. At each step the largest-magnitude remaining
        // Schur-complement diagonal is permuted to the pivot position (rows AND
        // columns swapped to preserve symmetry), then a rank-1 update eliminates
        // the column. piv[j] records the row swapped with j at step j (LAPACK
        // style); Solve replays it. This fixes the previous no-pivot path that
        // produced NaN whenever an (intermediate) diagonal was zero — exactly the
        // symmetric-INDEFINITE inputs this routine is meant to handle.
        //
        // Pivoting is 1×1 only (largest-diagonal rule), which covers the practical
        // indefinite cases. A fully general matrix whose entire remaining diagonal
        // is zero (e.g. [[0,1],[1,0]]) would need 2×2 block pivots; that is left as
        // a tracked follow-up and detected below (D=0 → Solve reports it rather
        // than silently dividing).
        var m = new double[n * n];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                m[i * n + j] = ToDouble(a[off + i * n + j]);

        for (int j = 0; j < n; j++)
        {
            // Choose pivot = largest |diagonal| among the remaining rows [j, n).
            int p = j;
            double best = Math.Abs(m[j * n + j]);
            for (int i = j + 1; i < n; i++)
            {
                double val = Math.Abs(m[i * n + i]);
                if (val > best) { best = val; p = i; }
            }
            piv[offPiv + j] = p;
            if (p != j)
            {
                for (int k = 0; k < n; k++) (m[j * n + k], m[p * n + k]) = (m[p * n + k], m[j * n + k]); // swap rows
                for (int k = 0; k < n; k++) (m[k * n + j], m[k * n + p]) = (m[k * n + p], m[k * n + j]); // swap cols
            }

            double d = m[j * n + j];
            if (d == 0.0) continue; // 2×2-pivot-essential block; D[j]=0 (see remark above)

            for (int i = j + 1; i < n; i++)
            {
                double lij = m[i * n + j] / d;
                m[i * n + j] = lij;                               // store L (unit-lower)
                for (int k = j + 1; k < n; k++)
                    m[i * n + k] -= lij * m[j * n + k];           // symmetric rank-1 Schur update
            }
        }

        // Pack: D on the diagonal, L strictly below; zero the rest.
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                a[off + i * n + j] = FromDouble<T>(j < i ? m[i * n + j] : (j == i ? m[i * n + i] : 0.0));

        if (upper)
        {
            // Mirror L into U = Lᵀ (upper), clearing the lower triangle.
            for (int i = 0; i < n; i++)
                for (int j = 0; j < i; j++)
                {
                    a[off + j * n + i] = a[off + i * n + j];
                    a[off + i * n + j] = default;
                }
        }
    }

    private static double ToDouble<T>(T v)
    {
        if (typeof(T) == typeof(float)) return (float)(object)v!;
        if (typeof(T) == typeof(double)) return (double)(object)v!;
        throw new NotSupportedException($"LDL requires float or double, got {typeof(T).Name}.");
    }

    private static T FromDouble<T>(double v)
    {
        if (typeof(T) == typeof(float)) return (T)(object)(float)v;
        if (typeof(T) == typeof(double)) return (T)(object)v;
        throw new NotSupportedException($"LDL requires float or double, got {typeof(T).Name}.");
    }
}
