using System;
using AiDotNet.Tensors.LinearAlgebra.Decompositions;

namespace AiDotNet.Tensors.LinearAlgebra;

/// <summary>
/// Scalar summaries — <see cref="Linalg.Det"/>, <see cref="Linalg.SlogDet"/>,
/// <see cref="Linalg.MatrixRank"/>, <see cref="Linalg.Cond"/>.
/// </summary>
internal static class LinalgScalars
{
    internal static Tensor<T> Det<T>(Tensor<T> input)
        where T : unmanaged, IEquatable<T>, IComparable<T>
    {
        if (input is null) throw new ArgumentNullException(nameof(input));
        if (input.Rank < 2) throw new ArgumentException("Det needs a 2D+ tensor.", nameof(input));
        int rank = input.Rank;
        int n = input.Shape[rank - 1];
        if (input.Shape[rank - 2] != n)
            throw new ArgumentException("Det needs a square matrix.", nameof(input));

        // det(A) = (sign of perm) · product of LU diagonal.
        var (lu, pivots) = LuDecomposition.Factor(input);

        var outShape = rank > 2
            ? TakePrefix(input._shape, rank - 2)
            : new[] { 1 };
        var det = new Tensor<T>(outShape);
        var detData = det.GetDataArray();
        var luData = lu.GetDataArray();
        var pivData = pivots.GetDataArray();

        int batch = 1;
        for (int i = 0; i < rank - 2; i++) batch *= input._shape[i];

        for (int b = 0; b < batch; b++)
        {
            double val = 1.0;
            int sign = 1;
            for (int i = 0; i < n; i++)
            {
                val *= ToDouble(luData[b * n * n + i * n + i]);
                if (pivData[b * n + i] != i) sign = -sign;
            }
            detData[b] = FromDouble<T>(sign * val);
        }

        return det;
    }

    internal static (Tensor<T> Sign, Tensor<T> LogAbsDet) SlogDet<T>(Tensor<T> input)
        where T : unmanaged, IEquatable<T>, IComparable<T>
    {
        if (input is null) throw new ArgumentNullException(nameof(input));
        if (input.Rank < 2) throw new ArgumentException("SlogDet needs a 2D+ tensor.", nameof(input));
        int rank = input.Rank;
        int n = input.Shape[rank - 1];
        if (input.Shape[rank - 2] != n)
            throw new ArgumentException("SlogDet needs a square matrix.", nameof(input));

        var (lu, pivots) = LuDecomposition.Factor(input);

        var outShape = rank > 2 ? TakePrefix(input._shape, rank - 2) : new[] { 1 };
        var sign = new Tensor<T>(outShape);
        var logAbs = new Tensor<T>(outShape);
        var signData = sign.GetDataArray();
        var laData = logAbs.GetDataArray();
        var luData = lu.GetDataArray();
        var pivData = pivots.GetDataArray();

        int batch = 1;
        for (int i = 0; i < rank - 2; i++) batch *= input._shape[i];

        for (int b = 0; b < batch; b++)
        {
            int s = 1;
            double logAbsDet = 0;
            for (int i = 0; i < n; i++)
            {
                double v = ToDouble(luData[b * n * n + i * n + i]);
                if (v < 0) s = -s;
                logAbsDet += Math.Log(Math.Abs(v));
                if (pivData[b * n + i] != i) s = -s;
            }
            signData[b] = FromDouble<T>(s);
            laData[b] = FromDouble<T>(logAbsDet);
        }

        return (sign, logAbs);
    }

    internal static Tensor<int> MatrixRank<T>(Tensor<T> input, double? atol, double? rtol, bool hermitian)
        where T : unmanaged, IEquatable<T>, IComparable<T>
    {
        var svd = SvdWrapper.Full(input, fullMatrices: false);
        int rank = input.Rank;
        int m = input.Shape[rank - 2];
        int n = input.Shape[rank - 1];
        int k = Math.Min(m, n);

        double eps = typeof(T) == typeof(float) ? 1.19e-7 : 2.22e-16;
        double atolEff = atol ?? 0.0;
        double rtolEff = rtol ?? (Math.Max(m, n) * eps);

        var outShape = rank > 2 ? TakePrefix(input._shape, rank - 2) : new[] { 1 };
        var result = new Tensor<int>(outShape);
        var rData = result.GetDataArray();
        var sData = svd.S.GetDataArray();

        int batch = 1;
        for (int i = 0; i < rank - 2; i++) batch *= input._shape[i];

        for (int b = 0; b < batch; b++)
        {
            double maxS = 0;
            for (int i = 0; i < k; i++) maxS = Math.Max(maxS, ToDouble(sData[b * k + i]));
            double threshold = Math.Max(atolEff, rtolEff * maxS);
            int count = 0;
            for (int i = 0; i < k; i++)
                if (ToDouble(sData[b * k + i]) > threshold) count++;
            rData[b] = count;
        }

        return result;
    }

    internal static Tensor<T> Cond<T>(Tensor<T> input, object? p)
        where T : unmanaged, IEquatable<T>, IComparable<T>
    {
        // Default p = 2 (ratio of largest to smallest singular value).
        var pOrd = p ?? 2.0;
        bool useSvd = pOrd is double d && (d == 2.0 || d == -2.0)
                   || pOrd is int ip && (ip == 2 || ip == -2)
                   || pOrd is string s && (s == "fro" || s == "nuc");

        int rank = input.Rank;
        var outShape = rank > 2 ? TakePrefix(input._shape, rank - 2) : new[] { 1 };
        var result = new Tensor<T>(outShape);
        var rData = result.GetDataArray();

        if (useSvd)
        {
            var svd = SvdWrapper.Full(input, fullMatrices: false);
            int m = input.Shape[rank - 2];
            int n = input.Shape[rank - 1];
            int k = Math.Min(m, n);
            var sData = svd.S.GetDataArray();
            int batch = 1;
            for (int i = 0; i < rank - 2; i++) batch *= input._shape[i];

            for (int b = 0; b < batch; b++)
            {
                double maxS = 0, minS = double.PositiveInfinity;
                for (int i = 0; i < k; i++)
                {
                    double v = ToDouble(sData[b * k + i]);
                    if (v > maxS) maxS = v;
                    if (v < minS) minS = v;
                }
                double condVal;
                if (pOrd is string str)
                {
                    if (str == "fro")
                    {
                        double fro2 = 0;
                        for (int i = 0; i < k; i++) fro2 += ToDouble(sData[b * k + i]) * ToDouble(sData[b * k + i]);
                        // For square matrices, cond_fro(A) = ||A||_F * ||A^-1||_F.
                        // ||A^-1||_F² = Σ 1/σᵢ²
                        double inv2 = 0;
                        for (int i = 0; i < k; i++)
                        {
                            double v = ToDouble(sData[b * k + i]);
                            if (v > 0) inv2 += 1.0 / (v * v);
                        }
                        condVal = Math.Sqrt(fro2 * inv2);
                    }
                    else // "nuc" — ||A||_nuc · ||A⁻¹||_nuc = (Σσᵢ) · (Σ 1/σᵢ).
                    {
                        double nucA = 0;
                        double nucInv = 0;
                        for (int i = 0; i < k; i++)
                        {
                            double v = ToDouble(sData[b * k + i]);
                            nucA += v;
                            if (v > 0) nucInv += 1.0 / v;
                            else { nucInv = double.PositiveInfinity; break; }
                        }
                        condVal = double.IsInfinity(nucInv) ? double.PositiveInfinity : nucA * nucInv;
                    }
                }
                else if (pOrd is double dp && dp < 0 || pOrd is int ipn && ipn < 0)
                    condVal = minS > 0 ? minS / maxS : double.PositiveInfinity;
                else
                    condVal = minS > 0 ? maxS / minS : double.PositiveInfinity;
                rData[b] = FromDouble<T>(condVal);
            }
        }
        else
        {
            // Norm-based cond (||A||_p · ||A⁻¹||_p). Expensive but correct for p ∈ {1, inf, -1, -inf}.
            var inv = LinalgInverses.Inv(input);
            var aNorm = LinalgNorms.MatrixNorm(input, pOrd, null!, false);
            var iNorm = LinalgNorms.MatrixNorm(inv, pOrd, null!, false);
            var anData = aNorm.GetDataArray();
            var inData = iNorm.GetDataArray();
            for (int b = 0; b < rData.Length; b++)
                rData[b] = FromDouble<T>(ToDouble(anData[b]) * ToDouble(inData[b]));
        }

        return result;
    }

    private static int[] TakePrefix(int[] shape, int len)
    {
        var result = new int[len];
        for (int i = 0; i < len; i++) result[i] = shape[i];
        return result;
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
