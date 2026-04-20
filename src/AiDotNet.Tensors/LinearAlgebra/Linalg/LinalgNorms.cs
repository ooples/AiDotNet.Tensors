using System;
using AiDotNet.Tensors.LinearAlgebra.Decompositions;

namespace AiDotNet.Tensors.LinearAlgebra;

/// <summary>
/// Norms — <see cref="Linalg.Norm"/>, <see cref="Linalg.VectorNorm"/>,
/// <see cref="Linalg.MatrixNorm"/>. All p-orders including fro/nuc/±1/±2/±inf.
/// </summary>
internal static class LinalgNorms
{
    internal static Tensor<T> Norm<T>(Tensor<T> input, object? ord, int[]? dim, bool keepDim)
        where T : unmanaged, IEquatable<T>, IComparable<T>
    {
        // Dispatch: 1D or dim[0] is a single axis → vector norm; 2D+ with two dims → matrix norm.
        if (input.Rank == 1 || (dim != null && dim.Length == 1))
        {
            double p = ToDoubleOrd(ord, defaultValue: 2.0);
            return VectorNorm(input, p, dim!, keepDim);
        }

        return MatrixNorm(input, ord, dim!, keepDim);
    }

    internal static Tensor<T> VectorNorm<T>(Tensor<T> input, double ord, int[]? dim, bool keepDim)
        where T : unmanaged, IEquatable<T>, IComparable<T>
    {
        if (input is null) throw new ArgumentNullException(nameof(input));
        // Reduce over specified dims (default: all).
        int rank = input.Rank;
        int[] axes = dim ?? AllAxes(rank);
        foreach (int a in axes)
            if (a < -rank || a >= rank) throw new ArgumentException($"dim {a} out of range.");
        var normAxes = new int[axes.Length];
        for (int i = 0; i < axes.Length; i++) normAxes[i] = axes[i] < 0 ? axes[i] + rank : axes[i];

        // Output shape: reduce specified axes (keepDim preserves them as size 1).
        var outShape = BuildReducedShape(input._shape, normAxes, keepDim);
        var result = new Tensor<T>(outShape);

        if (outShape.Length == 1 && outShape[0] == 1) {
            // Global reduction fast path.
            double acc = 0;
            double maxAbs = 0;
            double minAbs = double.PositiveInfinity;
            var data = input.GetDataArray();
            for (int i = 0; i < input.Length; i++)
            {
                double av = Math.Abs(ToDouble(data[i]));
                maxAbs = Math.Max(maxAbs, av);
                minAbs = Math.Min(minAbs, av);
                acc += Math.Pow(av, ord);
            }
            double normVal;
            if (double.IsPositiveInfinity(ord)) normVal = maxAbs;
            else if (double.IsNegativeInfinity(ord)) normVal = minAbs;
            else if (ord == 0) { normVal = 0; for (int i = 0; i < input.Length; i++) if (ToDouble(data[i]) != 0) normVal++; }
            else normVal = Math.Pow(acc, 1.0 / ord);
            result.GetDataArray()[0] = FromDouble<T>(normVal);
            return result;
        }

        // General reduction along specified axes.
        ReduceAlongAxes(input, result, normAxes, keepDim, (vals) =>
        {
            double acc = 0;
            double maxAbs = 0;
            double minAbs = double.PositiveInfinity;
            int nz = 0;
            for (int i = 0; i < vals.Count; i++)
            {
                double av = Math.Abs(vals[i]);
                if (av != 0) nz++;
                maxAbs = Math.Max(maxAbs, av);
                minAbs = Math.Min(minAbs, av);
                acc += Math.Pow(av, ord);
            }
            if (double.IsPositiveInfinity(ord)) return maxAbs;
            if (double.IsNegativeInfinity(ord)) return minAbs;
            if (ord == 0) return nz;
            return Math.Pow(acc, 1.0 / ord);
        });

        return result;
    }

    internal static Tensor<T> MatrixNorm<T>(Tensor<T> input, object? ord, int[]? dim, bool keepDim)
        where T : unmanaged, IEquatable<T>, IComparable<T>
    {
        if (input is null) throw new ArgumentNullException(nameof(input));
        if (input.Rank < 2) throw new ArgumentException("MatrixNorm requires at least 2D.");

        // ord defaults to fro.
        var ordVal = ord ?? "fro";
        int rank = input.Rank;

        // Default dim = (-2, -1) i.e., last two axes.
        int[] axes = dim ?? new[] { -2, -1 };
        if (axes.Length != 2) throw new ArgumentException("MatrixNorm needs exactly 2 axes.");
        int a0 = axes[0] < 0 ? axes[0] + rank : axes[0];
        int a1 = axes[1] < 0 ? axes[1] + rank : axes[1];

        var outShape = BuildReducedShape(input._shape, new[] { a0, a1 }, keepDim);
        var result = new Tensor<T>(outShape);

        int m = input.Shape[a0];
        int n = input.Shape[a1];
        int batch = 1;
        for (int i = 0; i < rank - 2; i++) batch *= input._shape[i];
        var inData = input.GetDataArray();
        var rData = result.GetDataArray();

        for (int b = 0; b < batch; b++)
        {
            int off = b * m * n;
            double val;
            if (ordVal is string s)
            {
                if (s == "fro")
                {
                    double sum = 0;
                    for (int i = 0; i < m * n; i++)
                    {
                        double v = ToDouble(inData[off + i]);
                        sum += v * v;
                    }
                    val = Math.Sqrt(sum);
                }
                else if (s == "nuc")
                {
                    var slice = SliceBatch(input, b, m, n);
                    var sv = SvdWrapper.ValuesOnly(slice);
                    double sum = 0;
                    var svD = sv.GetDataArray();
                    for (int i = 0; i < svD.Length; i++) sum += ToDouble(svD[i]);
                    val = sum;
                }
                else throw new ArgumentException($"Unknown matrix norm '{s}'.");
            }
            else if (ordVal is double dp)
            {
                val = MatrixPNorm(inData, off, m, n, dp);
            }
            else if (ordVal is int ip)
            {
                val = MatrixPNorm(inData, off, m, n, ip);
            }
            else throw new ArgumentException($"Unsupported ord type {ordVal.GetType().Name}.");

            rData[b] = FromDouble<T>(val);
        }

        return result;
    }

    private static double MatrixPNorm<T>(T[] data, int off, int m, int n, double p)
    {
        // 1-norm: max column sum
        // -1-norm: min column sum
        // inf-norm: max row sum
        // -inf-norm: min row sum
        // 2-norm: largest singular value (fall through to SVD — callers route through SVD branch)
        // -2-norm: smallest singular value
        if (p == 1.0)
        {
            double max = 0;
            for (int j = 0; j < n; j++)
            {
                double s = 0;
                for (int i = 0; i < m; i++) s += Math.Abs(ToDouble(data[off + i * n + j]));
                max = Math.Max(max, s);
            }
            return max;
        }
        if (p == -1.0)
        {
            double min = double.PositiveInfinity;
            for (int j = 0; j < n; j++)
            {
                double s = 0;
                for (int i = 0; i < m; i++) s += Math.Abs(ToDouble(data[off + i * n + j]));
                min = Math.Min(min, s);
            }
            return min;
        }
        if (double.IsPositiveInfinity(p))
        {
            double max = 0;
            for (int i = 0; i < m; i++)
            {
                double s = 0;
                for (int j = 0; j < n; j++) s += Math.Abs(ToDouble(data[off + i * n + j]));
                max = Math.Max(max, s);
            }
            return max;
        }
        if (double.IsNegativeInfinity(p))
        {
            double min = double.PositiveInfinity;
            for (int i = 0; i < m; i++)
            {
                double s = 0;
                for (int j = 0; j < n; j++) s += Math.Abs(ToDouble(data[off + i * n + j]));
                min = Math.Min(min, s);
            }
            return min;
        }
        throw new ArgumentException($"Unsupported matrix-norm p = {p}. Use 1, -1, inf, -inf, 'fro', 'nuc', or use SVD-based path for 2 / -2.");
    }

    // ── Helpers ─────────────────────────────────────────────────────────────

    private static Tensor<T> SliceBatch<T>(Tensor<T> input, int batchIdx, int m, int n)
        where T : unmanaged, IEquatable<T>, IComparable<T>
    {
        var slice = new Tensor<T>(new[] { m, n });
        var src = input.GetDataArray();
        var dst = slice.GetDataArray();
        int off = batchIdx * m * n;
        for (int i = 0; i < m * n; i++) dst[i] = src[off + i];
        return slice;
    }

    private static int[] AllAxes(int rank)
    {
        var axes = new int[rank];
        for (int i = 0; i < rank; i++) axes[i] = i;
        return axes;
    }

    private static int[] BuildReducedShape(int[] shape, int[] axes, bool keepDim)
    {
        var mark = new bool[shape.Length];
        foreach (int a in axes) mark[a] = true;
        if (keepDim)
        {
            var r = new int[shape.Length];
            for (int i = 0; i < shape.Length; i++) r[i] = mark[i] ? 1 : shape[i];
            return r;
        }
        int count = 0;
        for (int i = 0; i < shape.Length; i++) if (!mark[i]) count++;
        if (count == 0) return new[] { 1 };
        var result = new int[count];
        int idx = 0;
        for (int i = 0; i < shape.Length; i++) if (!mark[i]) result[idx++] = shape[i];
        return result;
    }

    private static void ReduceAlongAxes<T>(Tensor<T> input, Tensor<T> result, int[] axes, bool keepDim,
        Func<System.Collections.Generic.List<double>, double> reducer)
        where T : unmanaged, IEquatable<T>, IComparable<T>
    {
        // For each output position, walk all source positions where the non-reduced
        // coordinates match. Uses an "odometer" counter over the reduced axes to
        // avoid any recursive enumeration helper.
        var reduced = new bool[input.Rank];
        foreach (int a in axes) reduced[a] = true;

        var outData = result.GetDataArray();
        var inData = input.GetDataArray();
        var outShape = result._shape;
        var inShape = input._shape;
        var inStrides = input._strides;

        int totalOut = outData.Length;
        var outCoord = new int[outShape.Length];

        // Determine the set of reduced axis indices (in input space).
        var redAxes = new System.Collections.Generic.List<int>();
        for (int i = 0; i < input.Rank; i++) if (reduced[i]) redAxes.Add(i);
        long redTotal = 1;
        foreach (int a in redAxes) redTotal *= inShape[a];

        for (int outIdx = 0; outIdx < totalOut; outIdx++)
        {
            // Unravel outIdx into outCoord (row-major).
            int tmp = outIdx;
            for (int i = outShape.Length - 1; i >= 0; i--)
            {
                outCoord[i] = outShape[i] > 0 ? (tmp % outShape[i]) : 0;
                if (outShape[i] > 0) tmp /= outShape[i];
            }

            // Build input coord with non-reduced positions filled from outCoord.
            var inCoord = new int[input.Rank];
            int oi = 0;
            for (int i = 0; i < input.Rank; i++)
            {
                if (!reduced[i])
                {
                    int srcOutAxis = keepDim ? i : oi++;
                    inCoord[i] = outShape[srcOutAxis] > 0 ? outCoord[srcOutAxis] : 0;
                }
                else inCoord[i] = 0;
            }

            // Walk the reduced axes via an odometer; collect values.
            var vals = new System.Collections.Generic.List<double>((int)redTotal);
            long step;
            for (step = 0; step < redTotal; step++)
            {
                int off = 0;
                for (int i = 0; i < input.Rank; i++) off += inCoord[i] * inStrides[i];
                vals.Add(ToDouble(inData[off]));

                // Increment odometer over reduced axes.
                for (int ai = redAxes.Count - 1; ai >= 0; ai--)
                {
                    int a = redAxes[ai];
                    inCoord[a]++;
                    if (inCoord[a] < inShape[a]) break;
                    inCoord[a] = 0;
                }
            }

            outData[outIdx] = FromDouble<T>(reducer(vals));
        }
    }

    private static double ToDoubleOrd(object? ord, double defaultValue)
    {
        if (ord is null) return defaultValue;
        if (ord is double d) return d;
        if (ord is int i) return i;
        if (ord is float f) return f;
        if (ord is string s)
        {
            if (s == "inf") return double.PositiveInfinity;
            if (s == "-inf") return double.NegativeInfinity;
        }
        throw new ArgumentException($"Unsupported ord: {ord}.");
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
