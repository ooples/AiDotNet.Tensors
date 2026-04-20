using System;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Tensors.LinearAlgebra.Decompositions;

/// <summary>
/// LU factorization with partial (row) pivoting — <c>P·A = L·U</c>. Managed
/// Doolittle implementation; native LAPACK <c>?getrf</c> is the fallback tier
/// wired through <see cref="LapackProvider.TryGetrf(int, int, Span{float}, int, Span{int}, out int)"/>
/// but disabled in the current build.
///
/// <para>
/// Supports batched input <c>(..., M, N)</c>. The factor matrix is stored in
/// packed form: lower triangle of the output holds <c>L</c> (unit diagonal
/// implicit), upper triangle holds <c>U</c>. The pivot tensor holds row
/// permutations in 0-indexed LAPACK form (<c>pivots[i]</c> = row swapped with <c>i</c>).
/// </para>
/// </summary>
internal static class LuDecomposition
{
    /// <summary>
    /// Computes the full <c>(P, L, U)</c> triple such that <c>P·A = L·U</c>.
    /// Suitable for users who want explicit <c>L</c> and <c>U</c> matrices;
    /// prefer <see cref="Factor"/> + <see cref="Solve"/> for solving systems.
    /// </summary>
    internal static (Tensor<T> P, Tensor<T> L, Tensor<T> U) Compute<T>(Tensor<T> input)
        where T : unmanaged, IEquatable<T>, IComparable<T>
    {
        if (input is null) throw new ArgumentNullException(nameof(input));
        if (input.Rank < 2) throw new ArgumentException("LU requires at least a 2D tensor.", nameof(input));

        int rank = input.Rank;
        int m = input.Shape[rank - 2];
        int n = input.Shape[rank - 1];
        int k = Math.Min(m, n);

        var (luPacked, pivots) = Factor(input);
        int batch = BatchSize(input._shape, rank);

        var lShape = (int[])input._shape.Clone();
        lShape[rank - 1] = k;
        var uShape = (int[])input._shape.Clone();
        uShape[rank - 2] = k;
        var pShape = new int[rank];
        for (int i = 0; i < rank - 2; i++) pShape[i] = input._shape[i];
        pShape[rank - 2] = m;
        pShape[rank - 1] = m;

        var L = new Tensor<T>(lShape);
        var U = new Tensor<T>(uShape);
        var P = new Tensor<T>(pShape);

        for (int b = 0; b < batch; b++)
        {
            UnpackLU(luPacked, pivots, L, U, P, b, m, n, k);
        }

        return (P, L, U);
    }

    /// <summary>
    /// LAPACK <c>getrf</c>-style factorization. Returns the packed L\U factor
    /// and the row-pivot vector. Non-throwing for singular inputs — callers
    /// check the pivot info via <see cref="Solve"/>'s failure mode.
    /// </summary>
    internal static (Tensor<T> LU, Tensor<int> Pivots) Factor<T>(Tensor<T> input)
        where T : unmanaged, IEquatable<T>, IComparable<T>
    {
        if (input is null) throw new ArgumentNullException(nameof(input));
        if (input.Rank < 2) throw new ArgumentException("LU requires at least a 2D tensor.", nameof(input));

        int rank = input.Rank;
        int m = input.Shape[rank - 2];
        int n = input.Shape[rank - 1];
        int k = Math.Min(m, n);

        var lu = input.Contiguous();
        // Clone so we don't clobber caller's data when the source was a view.
        var luCopy = new Tensor<T>((int[])lu._shape.Clone());
        Array.Copy(lu.GetDataArray(), luCopy.GetDataArray(), lu.Length);

        var pivotShape = new int[rank - 1];
        for (int i = 0; i < rank - 2; i++) pivotShape[i] = input._shape[i];
        pivotShape[rank - 2] = k;
        var pivots = new Tensor<int>(pivotShape);

        int batch = BatchSize(input._shape, rank);
        var luData = luCopy.GetDataArray();
        var pivData = pivots.GetDataArray();
        int matStride = m * n;
        int pivStride = k;

        for (int b = 0; b < batch; b++)
        {
            FactorSingle(luData, b * matStride, pivData, b * pivStride, m, n);
        }

        return (luCopy, pivots);
    }

    /// <summary>Solve <c>A·X = B</c> from precomputed factors.</summary>
    internal static Tensor<T> Solve<T>(Tensor<T> lu, Tensor<int> pivots, Tensor<T> b)
        where T : unmanaged, IEquatable<T>, IComparable<T>
    {
        if (lu is null) throw new ArgumentNullException(nameof(lu));
        if (pivots is null) throw new ArgumentNullException(nameof(pivots));
        if (b is null) throw new ArgumentNullException(nameof(b));
        if (lu.Rank < 2) throw new ArgumentException("LU tensor must be at least 2D.");
        if (b.Rank < 1) throw new ArgumentException("RHS must be at least 1D.");

        int rank = lu.Rank;
        int n = lu.Shape[rank - 1];
        if (lu.Shape[rank - 2] != n) throw new ArgumentException("LU factor must be square.");

        bool bIsVector = b.Rank == lu.Rank - 1;
        int nrhs = bIsVector ? 1 : b.Shape[b.Rank - 1];

        var xShape = (int[])b._shape.Clone();
        var x = new Tensor<T>(xShape);
        Array.Copy(b.GetDataArray(), x.GetDataArray(), b.Length);

        int batch = BatchSize(lu._shape, rank);
        var luData = lu.GetDataArray();
        var pivData = pivots.GetDataArray();
        var xData = x.GetDataArray();
        int luStride = n * n;
        int pivStride = n;
        int xStride = bIsVector ? n : n * nrhs;

        for (int batchIdx = 0; batchIdx < batch; batchIdx++)
        {
            SolveSingle(
                luData, batchIdx * luStride,
                pivData, batchIdx * pivStride,
                xData, batchIdx * xStride,
                n, nrhs, bIsVector);
        }

        return x;
    }

    // ── Scalar kernels ──────────────────────────────────────────────────────

    private static unsafe void FactorSingle<T>(T[] a, int offA, int[] piv, int offPiv, int m, int n)
        where T : unmanaged, IEquatable<T>, IComparable<T>
    {
        // Managed Doolittle with partial pivoting over T ∈ {float, double}.
        // Ops operate on generic T via a tiny numeric helper; the hot path runs
        // through GetRef/SetRef which the JIT inlines for primitive types.
        int k = Math.Min(m, n);

        // Try LAPACK native tier first (always false today, but the dispatch is
        // wired so a future PR can activate MKL without touching this file).
        if (typeof(T) == typeof(float) && LapackProvider.HasLapack)
        {
            var aSpan = new Span<T>(a, offA, m * n);
            var pivSpan = new Span<int>(piv, offPiv, k);
            if (LapackProvider.TryGetrf(m, n, System.Runtime.InteropServices.MemoryMarshal.Cast<T, float>(aSpan),
                    m, pivSpan, out _))
                return;
        }

        // Managed reference.
        for (int j = 0; j < k; j++)
        {
            // Find pivot row in column j, starting at row j.
            int pivRow = j;
            double maxAbs = Math.Abs(ToDouble(a[offA + j * n + j]));
            for (int i = j + 1; i < m; i++)
            {
                double v = Math.Abs(ToDouble(a[offA + i * n + j]));
                if (v > maxAbs) { maxAbs = v; pivRow = i; }
            }
            piv[offPiv + j] = pivRow;

            // Row swap if needed.
            if (pivRow != j)
            {
                for (int c = 0; c < n; c++)
                {
                    (a[offA + pivRow * n + c], a[offA + j * n + c]) =
                        (a[offA + j * n + c], a[offA + pivRow * n + c]);
                }
            }

            // Skip update if pivot is zero (singular — caller detects via Solve).
            double pivVal = ToDouble(a[offA + j * n + j]);
            if (pivVal == 0.0) continue;

            // Scale column below pivot.
            for (int i = j + 1; i < m; i++)
            {
                double factor = ToDouble(a[offA + i * n + j]) / pivVal;
                a[offA + i * n + j] = FromDouble<T>(factor);
                // Trailing submatrix update.
                for (int c = j + 1; c < n; c++)
                {
                    double updated = ToDouble(a[offA + i * n + c]) - factor * ToDouble(a[offA + j * n + c]);
                    a[offA + i * n + c] = FromDouble<T>(updated);
                }
            }
        }
    }

    private static void SolveSingle<T>(
        T[] lu, int offLU, int[] piv, int offPiv,
        T[] x, int offX, int n, int nrhs, bool bIsVector)
        where T : unmanaged, IEquatable<T>, IComparable<T>
    {
        int colsX = bIsVector ? 1 : nrhs;

        // Apply pivots to RHS.
        for (int i = 0; i < n; i++)
        {
            int p = piv[offPiv + i];
            if (p != i)
            {
                for (int c = 0; c < colsX; c++)
                {
                    int idxI = offX + i * colsX + c;
                    int idxP = offX + p * colsX + c;
                    (x[idxI], x[idxP]) = (x[idxP], x[idxI]);
                }
            }
        }

        // Forward substitution: solve L·y = Pb. L is unit-diagonal lower triangular.
        for (int i = 1; i < n; i++)
        {
            for (int c = 0; c < colsX; c++)
            {
                double sum = ToDouble(x[offX + i * colsX + c]);
                for (int j = 0; j < i; j++)
                {
                    sum -= ToDouble(lu[offLU + i * n + j]) * ToDouble(x[offX + j * colsX + c]);
                }
                x[offX + i * colsX + c] = FromDouble<T>(sum);
            }
        }

        // Backward substitution: solve U·x = y.
        for (int i = n - 1; i >= 0; i--)
        {
            double pivVal = ToDouble(lu[offLU + i * n + i]);
            for (int c = 0; c < colsX; c++)
            {
                double sum = ToDouble(x[offX + i * colsX + c]);
                for (int j = i + 1; j < n; j++)
                {
                    sum -= ToDouble(lu[offLU + i * n + j]) * ToDouble(x[offX + j * colsX + c]);
                }
                if (pivVal == 0.0)
                    x[offX + i * colsX + c] = FromDouble<T>(double.NaN);
                else
                    x[offX + i * colsX + c] = FromDouble<T>(sum / pivVal);
            }
        }
    }

    private static void UnpackLU<T>(
        Tensor<T> luPacked, Tensor<int> pivots,
        Tensor<T> L, Tensor<T> U, Tensor<T> P,
        int batchIdx, int m, int n, int k)
        where T : unmanaged, IEquatable<T>, IComparable<T>
    {
        var luData = luPacked.GetDataArray();
        var pivData = pivots.GetDataArray();
        var lData = L.GetDataArray();
        var uData = U.GetDataArray();
        var pData = P.GetDataArray();
        int luStride = m * n;
        int pivStride = k;
        int lStride = m * k;
        int uStride = k * n;
        int pStride = m * m;

        int luBase = batchIdx * luStride;
        int pivBase = batchIdx * pivStride;
        int lBase = batchIdx * lStride;
        int uBase = batchIdx * uStride;
        int pBase = batchIdx * pStride;

        // L: lower of LU with unit diagonal, width k.
        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < k; j++)
            {
                if (i > j) lData[lBase + i * k + j] = luData[luBase + i * n + j];
                else if (i == j) lData[lBase + i * k + j] = FromDouble<T>(1.0);
                else lData[lBase + i * k + j] = default;
            }
        }

        // U: upper of LU, height k.
        for (int i = 0; i < k; i++)
        {
            for (int j = 0; j < n; j++)
            {
                if (i <= j) uData[uBase + i * n + j] = luData[luBase + i * n + j];
                else uData[uBase + i * n + j] = default;
            }
        }

        // P: identity with pivot swaps applied.
        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < m; j++)
                pData[pBase + i * m + j] = i == j ? FromDouble<T>(1.0) : default;
        }
        for (int i = 0; i < k; i++)
        {
            int p = pivData[pivBase + i];
            if (p != i)
            {
                for (int c = 0; c < m; c++)
                {
                    (pData[pBase + i * m + c], pData[pBase + p * m + c]) =
                        (pData[pBase + p * m + c], pData[pBase + i * m + c]);
                }
            }
        }
    }

    // ── Numeric helpers ─────────────────────────────────────────────────────

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
        throw new NotSupportedException($"Linalg ops require float or double, got {typeof(T).Name}.");
    }

    private static T FromDouble<T>(double v)
    {
        if (typeof(T) == typeof(float)) return (T)(object)(float)v;
        if (typeof(T) == typeof(double)) return (T)(object)v;
        throw new NotSupportedException($"Linalg ops require float or double, got {typeof(T).Name}.");
    }
}
