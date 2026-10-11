using System;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Tensors.LinearAlgebra.Decompositions;

/// <summary>
/// Cholesky factorization of a symmetric positive-definite matrix —
/// <c>A = L·Lᵀ</c> (or <c>A = Uᵀ·U</c>). Managed right-looking implementation;
/// native LAPACK <c>?potrf</c> is the stubbed fallback tier.
/// </summary>
internal static class CholeskyDecomposition
{
    /// <summary>
    /// Computes the Cholesky factor. Returns <c>Info[b] == 0</c> on success;
    /// a positive value <c>k</c> indicates the leading minor of order k was not
    /// positive definite (the factorization was aborted at that minor).
    /// </summary>
    internal static (Tensor<T> Factor, Tensor<int> Info) Compute<T>(Tensor<T> input, bool upper)
        where T : unmanaged, IEquatable<T>, IComparable<T>
    {
        if (input is null) throw new ArgumentNullException(nameof(input));
        if (input.Rank < 2) throw new ArgumentException("Cholesky requires at least a 2D tensor.", nameof(input));

        int rank = input.Rank;
        int n = input.Shape[rank - 1];
        if (input.Shape[rank - 2] != n)
            throw new ArgumentException("Cholesky requires a square matrix.", nameof(input));

        var factor = new Tensor<T>((int[])input._shape.Clone());
        Array.Copy(input.GetDataArray(), factor.GetDataArray(), input.Length);

        var infoShape = new int[Math.Max(rank - 2, 1)];
        if (rank > 2)
        {
            for (int i = 0; i < rank - 2; i++) infoShape[i] = input._shape[i];
        }
        else
        {
            infoShape[0] = 1;
        }
        var info = new Tensor<int>(infoShape);

        int batch = BatchSize(input._shape, rank);
        var fData = factor.GetDataArray();
        var iData = info.GetDataArray();
        int matStride = n * n;

        for (int b = 0; b < batch; b++)
        {
            int status = FactorSingle(fData, b * matStride, n, upper);
            iData[b] = status;
            // Zero out the opposite triangle for a clean factor view.
            ZeroOpposite(fData, b * matStride, n, upper);
        }

        return (factor, info);
    }

    /// <summary>Solves <c>A·X = B</c> via Cholesky (assumes <paramref name="factor"/> is L or U as stored).</summary>
    internal static Tensor<T> Solve<T>(Tensor<T> factor, Tensor<T> b, bool upper)
        where T : unmanaged, IEquatable<T>, IComparable<T>
    {
        // A = L·Lᵀ  ⇒  solve L·y = b, then Lᵀ·x = y.
        var y = Solvers.LinearSolvers.SolveTriangularInternal(factor, b, upper: upper, transpose: upper, unitDiagonal: false);
        return Solvers.LinearSolvers.SolveTriangularInternal(factor, y, upper: upper, transpose: !upper, unitDiagonal: false);
    }

    // ── Kernels ─────────────────────────────────────────────────────────────

    private static int FactorSingle<T>(T[] a, int off, int n, bool upper)
        where T : unmanaged, IEquatable<T>, IComparable<T>
    {
        // Try native tier — stubbed false today but dispatches both fp32 and fp64.
        if (LapackProvider.HasLapack)
        {
            if (typeof(T) == typeof(float))
            {
                var span = new Span<T>(a, off, n * n);
                if (LapackProvider.TryPotrf(upper, n,
                    System.Runtime.InteropServices.MemoryMarshal.Cast<T, float>(span), n, out int info))
                    return info;
            }
            else if (typeof(T) == typeof(double))
            {
                var span = new Span<T>(a, off, n * n);
                if (LapackProvider.TryPotrf(upper, n,
                    System.Runtime.InteropServices.MemoryMarshal.Cast<T, double>(span), n, out int info))
                    return info;
            }
        }

        // Managed right-looking Cholesky. Fails gracefully: returns k+1 when the
        // leading minor of order k+1 is not positive definite.
        if (upper)
        {
            for (int j = 0; j < n; j++)
            {
                double diag = ToDouble(a[off + j * n + j]);
                for (int k = 0; k < j; k++)
                    diag -= ToDouble(a[off + k * n + j]) * ToDouble(a[off + k * n + j]);
                if (diag <= 0.0) return j + 1;
                double sqrtDiag = Math.Sqrt(diag);
                a[off + j * n + j] = FromDouble<T>(sqrtDiag);
                for (int i = j + 1; i < n; i++)
                {
                    double sum = ToDouble(a[off + j * n + i]);
                    for (int k = 0; k < j; k++)
                        sum -= ToDouble(a[off + k * n + i]) * ToDouble(a[off + k * n + j]);
                    a[off + j * n + i] = FromDouble<T>(sum / sqrtDiag);
                }
            }
        }
        else
        {
            for (int j = 0; j < n; j++)
            {
                double diag = ToDouble(a[off + j * n + j]);
                for (int k = 0; k < j; k++)
                    diag -= ToDouble(a[off + j * n + k]) * ToDouble(a[off + j * n + k]);
                if (diag <= 0.0) return j + 1;
                double sqrtDiag = Math.Sqrt(diag);
                a[off + j * n + j] = FromDouble<T>(sqrtDiag);
                for (int i = j + 1; i < n; i++)
                {
                    double sum = ToDouble(a[off + i * n + j]);
                    for (int k = 0; k < j; k++)
                        sum -= ToDouble(a[off + i * n + k]) * ToDouble(a[off + j * n + k]);
                    a[off + i * n + j] = FromDouble<T>(sum / sqrtDiag);
                }
            }
        }
        return 0;
    }

    private static void ZeroOpposite<T>(T[] a, int off, int n, bool upper)
        where T : unmanaged, IEquatable<T>, IComparable<T>
    {
        T zero = default;
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                if (upper ? i > j : i < j)
                    a[off + i * n + j] = zero;
            }
        }
    }

    // ── Helpers ─────────────────────────────────────────────────────────────

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
        throw new NotSupportedException($"Cholesky requires float or double, got {typeof(T).Name}.");
    }

    private static T FromDouble<T>(double v)
    {
        if (typeof(T) == typeof(float)) return (T)(object)(float)v;
        if (typeof(T) == typeof(double)) return (T)(object)v;
        throw new NotSupportedException($"Cholesky requires float or double, got {typeof(T).Name}.");
    }
}
