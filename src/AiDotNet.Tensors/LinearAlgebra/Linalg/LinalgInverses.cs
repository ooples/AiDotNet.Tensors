using System;
using AiDotNet.Tensors.LinearAlgebra.Decompositions;
using AiDotNet.Tensors.LinearAlgebra.Solvers;

namespace AiDotNet.Tensors.LinearAlgebra;

/// <summary>Inv / InvEx / Pinv — inverses and Moore–Penrose pseudoinverse.</summary>
internal static class LinalgInverses
{
    internal static Tensor<T> Inv<T>(Tensor<T> input)
        where T : unmanaged, IEquatable<T>, IComparable<T>
    {
        // Solve A·X = I via LU factorization.
        if (input.Rank < 2) throw new ArgumentException("Inv needs a square matrix.", nameof(input));
        int rank = input.Rank;
        int n = input.Shape[rank - 1];
        if (input.Shape[rank - 2] != n) throw new ArgumentException("Inv needs a square matrix.");
        var eye = Eye<T>(input._shape, n);
        return LinearSolvers.Solve(input, eye);
    }

    internal static (Tensor<T> Inverse, Tensor<int> Info) InvEx<T>(Tensor<T> input)
        where T : unmanaged, IEquatable<T>, IComparable<T>
    {
        if (input.Rank < 2) throw new ArgumentException("InvEx needs a square matrix.", nameof(input));
        int rank = input.Rank;
        int n = input.Shape[rank - 1];
        var eye = Eye<T>(input._shape, n);
        return LinearSolvers.SolveEx(input, eye);
    }

    internal static Tensor<T> Pinv<T>(Tensor<T> input, double? rcond)
        where T : unmanaged, IEquatable<T>, IComparable<T>
    {
        // Pinv = V·Σ⁺·Uᵀ, where Σ⁺ is the reciprocal of singular values above threshold.
        var (U, S, Vh) = SvdWrapper.Full(input, fullMatrices: false);
        int rank = input.Rank;
        int m = input.Shape[rank - 2];
        int n = input.Shape[rank - 1];
        int k = Math.Min(m, n);
        double eps = typeof(T) == typeof(float) ? 1.19e-7 : 2.22e-16;
        double cutoff = rcond ?? (Math.Max(m, n) * eps);

        var sData = S.GetDataArray();
        var uData = U.GetDataArray();
        var vhData = Vh.GetDataArray();

        // Result shape: (..., N, M).
        var pShape = (int[])input._shape.Clone();
        pShape[rank - 2] = n;
        pShape[rank - 1] = m;
        var pinv = new Tensor<T>(pShape);
        var pData = pinv.GetDataArray();

        int batch = 1;
        for (int i = 0; i < rank - 2; i++) batch *= input._shape[i];
        int uStride = m * k;
        int sStride = k;
        int vhStride = k * n;
        int pStride = n * m;

        for (int b = 0; b < batch; b++)
        {
            // Find max singular value for relative threshold.
            double maxS = 0;
            for (int i = 0; i < k; i++) maxS = Math.Max(maxS, ToDouble(sData[b * sStride + i]));
            double threshold = cutoff * maxS;

            // Compute V·Σ⁺·Uᵀ.
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < m; j++)
                {
                    double val = 0;
                    for (int r = 0; r < k; r++)
                    {
                        double sigma = ToDouble(sData[b * sStride + r]);
                        if (sigma <= threshold) continue;
                        double vri = ToDouble(vhData[b * vhStride + r * n + i]);  // Vh[r,i] = V[i,r]
                        double urj = ToDouble(uData[b * uStride + j * k + r]);    // U[j,r]
                        val += vri * urj / sigma;
                    }
                    pData[b * pStride + i * m + j] = FromDouble<T>(val);
                }
            }
        }

        return pinv;
    }

    // ── Helpers ─────────────────────────────────────────────────────────────

    private static Tensor<T> Eye<T>(int[] inputShape, int n)
        where T : unmanaged, IEquatable<T>, IComparable<T>
    {
        var shape = (int[])inputShape.Clone();
        var eye = new Tensor<T>(shape);
        var data = eye.GetDataArray();
        int batch = 1;
        for (int i = 0; i < inputShape.Length - 2; i++) batch *= inputShape[i];
        int stride = n * n;
        T one = FromDouble<T>(1.0);
        for (int b = 0; b < batch; b++)
            for (int i = 0; i < n; i++)
                data[b * stride + i * n + i] = one;
        return eye;
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
