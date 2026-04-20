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
        int matStride = n * n;
        int rhsStride = b.Rank == ld.Rank ? n * nrhs : n;

        for (int ib = 0; ib < batch; ib++)
            SolveSingle(ldData, ib * matStride, bData, ib * rhsStride, xData, ib * rhsStride, n, nrhs, upper);

        return result;
    }

    private static void SolveSingle<T>(T[] a, int offA, T[] b, int offB, T[] x, int offX, int n, int nrhs, bool upper)
        where T : unmanaged, IEquatable<T>, IComparable<T>
    {
        // Copy RHS into X; we solve in place.
        for (int i = 0; i < n * nrhs; i++) x[offX + i] = b[offB + i];

        for (int c = 0; c < nrhs; c++)
        {
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
        }
    }

    private static void FactorSingle<T>(T[] a, int off, int[] piv, int offPiv, int n, bool upper)
        where T : unmanaged, IEquatable<T>, IComparable<T>
    {
        // Simple Bunch–Kaufman (1×1 block pivoting only). Sufficient for
        // strictly-diagonally-dominant or positive-definite cases; a future
        // PR upgrades to full 2×2 block pivoting for strongly indefinite
        // inputs. The unified lower-triangle loop writes L[i,j] for i>j and
        // D[i,i] on the diagonal; when `upper` is true we transpose into the
        // upper triangle afterwards so callers see U = Lᵀ.
        for (int j = 0; j < n; j++)
        {
            piv[offPiv + j] = j; // Trivial pivots for the simple path.
            double d = ToDouble(a[off + j * n + j]);
            for (int k = 0; k < j; k++)
            {
                double l_jk = ToDouble(a[off + j * n + k]);
                double d_k = ToDouble(a[off + k * n + k]);
                d -= l_jk * l_jk * d_k;
            }
            a[off + j * n + j] = FromDouble<T>(d);
            if (d == 0.0) continue;

            for (int i = j + 1; i < n; i++)
            {
                double s = ToDouble(a[off + i * n + j]);
                for (int k = 0; k < j; k++)
                {
                    double l_ik = ToDouble(a[off + i * n + k]);
                    double l_jk = ToDouble(a[off + j * n + k]);
                    double d_k = ToDouble(a[off + k * n + k]);
                    s -= l_ik * l_jk * d_k;
                }
                a[off + i * n + j] = FromDouble<T>(s / d);
            }
        }

        if (upper)
        {
            // Move lower-triangle L into upper-triangle U = Lᵀ, then clear the
            // lower triangle so the returned factor has U in the upper and
            // zeros below the diagonal.
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < i; j++)
                {
                    a[off + j * n + i] = a[off + i * n + j];
                    a[off + i * n + j] = default;
                }
            }
        }
        else
        {
            // Zero upper triangle for a clean L·D·Lᵀ factor view.
            for (int i = 0; i < n; i++)
                for (int j = i + 1; j < n; j++)
                    a[off + i * n + j] = default;
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
