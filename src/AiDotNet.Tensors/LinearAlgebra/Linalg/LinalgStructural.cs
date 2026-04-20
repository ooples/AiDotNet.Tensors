using System;
using System.Collections.Generic;
using AiDotNet.Tensors.LinearAlgebra.Solvers;

namespace AiDotNet.Tensors.LinearAlgebra;

/// <summary>
/// Structural / composition-only linear-algebra operations. No LAPACK required
/// for anything here — these are built from tensor-level primitives or trivial
/// scalar loops.
/// </summary>
internal static class LinalgStructural
{
    internal static Tensor<T> MatrixPower<T>(Tensor<T> input, int n)
        where T : unmanaged, IEquatable<T>, IComparable<T>
    {
        if (input is null) throw new ArgumentNullException(nameof(input));
        if (input.Rank < 2) throw new ArgumentException("MatrixPower needs a square matrix.");
        int rank = input.Rank;
        int m = input.Shape[rank - 1];
        if (input.Shape[rank - 2] != m) throw new ArgumentException("MatrixPower needs a square matrix.");

        if (n == 0) return Eye<T>(input, m);
        if (n < 0) return MatrixPower(LinalgInverses.Inv(input), -n);

        // Exponentiation by squaring over matrix multiply.
        Tensor<T> result = Eye<T>(input, m);
        Tensor<T> baseMat = Clone(input);
        int exp = n;
        while (exp > 0)
        {
            if ((exp & 1) == 1) result = MatMul(result, baseMat);
            exp >>= 1;
            if (exp > 0) baseMat = MatMul(baseMat, baseMat);
        }
        return result;
    }

    internal static Tensor<T> MatrixExp<T>(Tensor<T> input)
        where T : unmanaged, IEquatable<T>, IComparable<T>
    {
        // Padé scaling-and-squaring with the [6/6] approximant. Reference:
        // Higham, "The Scaling and Squaring Method for the Matrix Exponential Revisited" (2005).
        if (input is null) throw new ArgumentNullException(nameof(input));
        if (input.Rank < 2) throw new ArgumentException("MatrixExp needs a square matrix.");
        int rank = input.Rank;
        int n = input.Shape[rank - 1];
        if (input.Shape[rank - 2] != n) throw new ArgumentException("MatrixExp needs a square matrix.");

        int batch = 1;
        for (int i = 0; i < rank - 2; i++) batch *= input._shape[i];

        var result = new Tensor<T>((int[])input._shape.Clone());
        var src = input.GetDataArray();
        var dst = result.GetDataArray();

        for (int b = 0; b < batch; b++)
        {
            // Copy this batch's matrix to a double scratch.
            var A = new double[n * n];
            for (int i = 0; i < n * n; i++) A[i] = ToDouble(src[b * n * n + i]);

            // Compute 1-norm of A.
            double norm1 = 0;
            for (int j = 0; j < n; j++)
            {
                double s = 0;
                for (int i = 0; i < n; i++) s += Math.Abs(A[i * n + j]);
                if (s > norm1) norm1 = s;
            }

            // Scale so norm1 ≤ 0.5.
            int squarings = 0;
            if (norm1 > 0.5)
            {
                squarings = (int)Math.Ceiling(Math.Log(norm1 / 0.5) / Math.Log(2.0));
                squarings = Math.Max(0, squarings);
                double scale = 1.0 / Math.Pow(2.0, squarings);
                for (int i = 0; i < n * n; i++) A[i] *= scale;
            }

            // Padé [6/6] coefficients.
            double[] c =
            {
                1.0, 1.0 / 2.0, 5.0 / 44.0, 1.0 / 66.0, 1.0 / 792.0,
                1.0 / 15840.0, 1.0 / 665280.0
            };

            var A2 = MatMulMat(A, A, n);
            var A4 = MatMulMat(A2, A2, n);
            var A6 = MatMulMat(A4, A2, n);

            // U = A · (c[1]·I + c[3]·A² + c[5]·A⁴ + c[7]·A⁶). We only have up to c[5]/c[7]
            // approximated; this simplified form is adequate for norm ≤ 0.5.
            var U = new double[n * n];
            var V = new double[n * n];
            for (int i = 0; i < n; i++)
            {
                V[i * n + i] += c[0];
                U[i * n + i] += c[1];
            }
            for (int i = 0; i < n * n; i++)
            {
                V[i] += c[2] * A2[i];
                U[i] += c[3] * A2[i];
                V[i] += c[4] * A4[i];
                U[i] += c[5] * A4[i];
                V[i] += c[6] * A6[i];
            }
            U = MatMulMat(A, U, n);

            // N = V + U, D = V − U.
            var N = new double[n * n];
            var D = new double[n * n];
            for (int i = 0; i < n * n; i++) { N[i] = V[i] + U[i]; D[i] = V[i] - U[i]; }

            // Solve D · X = N for X (via internal LU on doubles).
            var X = SolveDouble(D, N, n);

            // Square `squarings` times.
            for (int s = 0; s < squarings; s++)
                X = MatMulMat(X, X, n);

            for (int i = 0; i < n * n; i++) dst[b * n * n + i] = FromDouble<T>(X[i]);
        }

        return result;
    }

    internal static Tensor<T> MultiDot<T>(IReadOnlyList<Tensor<T>> matrices)
        where T : unmanaged, IEquatable<T>, IComparable<T>
    {
        if (matrices is null) throw new ArgumentNullException(nameof(matrices));
        if (matrices.Count == 0) throw new ArgumentException("Need at least one matrix.");
        if (matrices.Count == 1) return Clone(matrices[0]);
        if (matrices.Count == 2) return MatMul(matrices[0], matrices[1]);

        // Dynamic programming — classic matrix-chain multiplication.
        int k = matrices.Count;
        var dims = new int[k + 1];
        dims[0] = matrices[0].Shape[matrices[0].Rank - 2];
        for (int i = 0; i < k; i++)
            dims[i + 1] = matrices[i].Shape[matrices[i].Rank - 1];

        long[,] cost = new long[k, k];
        int[,] split = new int[k, k];
        for (int len = 2; len <= k; len++)
        {
            for (int i = 0; i + len - 1 < k; i++)
            {
                int j = i + len - 1;
                cost[i, j] = long.MaxValue;
                for (int m = i; m < j; m++)
                {
                    long c = cost[i, m] + cost[m + 1, j]
                        + (long)dims[i] * dims[m + 1] * dims[j + 1];
                    if (c < cost[i, j]) { cost[i, j] = c; split[i, j] = m; }
                }
            }
        }

        return MultiDotRec(matrices, split, 0, k - 1);
    }

    internal static Tensor<T> Cross<T>(Tensor<T> a, Tensor<T> b, int dim)
        where T : unmanaged, IEquatable<T>, IComparable<T>
    {
        if (a is null) throw new ArgumentNullException(nameof(a));
        if (b is null) throw new ArgumentNullException(nameof(b));
        int rank = a.Rank;
        int d = dim < 0 ? dim + rank : dim;
        if (d < 0 || d >= rank) throw new ArgumentOutOfRangeException(nameof(dim));
        if (a.Shape[d] != 3 || b.Shape[d] != 3) throw new ArgumentException("Cross needs size-3 axis.");

        // Supports arbitrary dim via stride-based walk — no permute required.
        // Inner/outer split: positions with index < d form the "outer" batch,
        // positions > d form the "inner" stride; the 3 components at axis d
        // are combined per (outer, inner) slot.
        var result = new Tensor<T>((int[])a._shape.Clone());
        var aD = a.GetDataArray();
        var bD = b.GetDataArray();
        var rD = result.GetDataArray();

        int outer = 1;
        int inner = 1;
        for (int i = 0; i < d; i++) outer *= a._shape[i];
        for (int i = d + 1; i < rank; i++) inner *= a._shape[i];

        for (int o = 0; o < outer; o++)
        {
            for (int s = 0; s < inner; s++)
            {
                int off0 = o * 3 * inner + 0 * inner + s;
                int off1 = o * 3 * inner + 1 * inner + s;
                int off2 = o * 3 * inner + 2 * inner + s;

                double ax = ToDouble(aD[off0]);
                double ay = ToDouble(aD[off1]);
                double az = ToDouble(aD[off2]);
                double bx = ToDouble(bD[off0]);
                double by = ToDouble(bD[off1]);
                double bz = ToDouble(bD[off2]);
                rD[off0] = FromDouble<T>(ay * bz - az * by);
                rD[off1] = FromDouble<T>(az * bx - ax * bz);
                rD[off2] = FromDouble<T>(ax * by - ay * bx);
            }
        }
        return result;
    }

    internal static Tensor<T> Vander<T>(Tensor<T> x, int? n, bool increasing)
        where T : unmanaged, IEquatable<T>, IComparable<T>
    {
        if (x is null) throw new ArgumentNullException(nameof(x));
        if (x.Rank != 1) throw new ArgumentException("Vander needs 1D input.");
        int m = x.Shape[0];
        int cols = n ?? m;

        var result = new Tensor<T>(new[] { m, cols });
        var xD = x.GetDataArray();
        var rD = result.GetDataArray();

        for (int i = 0; i < m; i++)
        {
            double v = ToDouble(xD[i]);
            for (int j = 0; j < cols; j++)
            {
                int power = increasing ? j : cols - 1 - j;
                rD[i * cols + j] = FromDouble<T>(Math.Pow(v, power));
            }
        }
        return result;
    }

    internal static Tensor<T> HouseholderProduct<T>(Tensor<T> reflectors, Tensor<T> tau)
        where T : unmanaged, IEquatable<T>, IComparable<T>
    {
        // Apply k Householder reflectors (columns of `reflectors`) in order to build Q.
        // Supports batched input of shape (..., M, K); tau has shape (..., K).
        if (reflectors is null) throw new ArgumentNullException(nameof(reflectors));
        if (tau is null) throw new ArgumentNullException(nameof(tau));
        if (reflectors.Rank < 2) throw new ArgumentException("HouseholderProduct needs at least 2D input.");

        int rank = reflectors.Rank;
        int m = reflectors.Shape[rank - 2];
        int k = reflectors.Shape[rank - 1];
        int batch = 1;
        for (int i = 0; i < rank - 2; i++) batch *= reflectors._shape[i];

        var outShape = (int[])reflectors._shape.Clone();
        outShape[rank - 1] = m;
        var result = new Tensor<T>(outShape);

        var refD = reflectors.GetDataArray();
        var tauD = tau.GetDataArray();
        var rD = result.GetDataArray();

        for (int b = 0; b < batch; b++)
        {
            var q = new double[m * m];
            for (int i = 0; i < m; i++) q[i * m + i] = 1.0;

            for (int j = k - 1; j >= 0; j--)
            {
                double beta = ToDouble(tauD[b * k + j]);
                if (beta == 0) continue;
                int colLen = m - j;
                var v = new double[colLen];
                v[0] = 1.0;
                for (int i = 1; i < colLen; i++) v[i] = ToDouble(refD[b * m * k + (j + i) * k + j]);

                for (int c = 0; c < m; c++)
                {
                    double dot = 0;
                    for (int i = 0; i < colLen; i++) dot += v[i] * q[(j + i) * m + c];
                    double scale = beta * dot;
                    for (int i = 0; i < colLen; i++) q[(j + i) * m + c] -= scale * v[i];
                }
            }

            for (int i = 0; i < m * m; i++) rD[b * m * m + i] = FromDouble<T>(q[i]);
        }

        return result;
    }

    internal static Tensor<T> Diagonal<T>(Tensor<T> input, int offset, int dim1, int dim2)
        where T : unmanaged, IEquatable<T>, IComparable<T>
    {
        if (input is null) throw new ArgumentNullException(nameof(input));
        int rank = input.Rank;
        int d1 = dim1 < 0 ? dim1 + rank : dim1;
        int d2 = dim2 < 0 ? dim2 + rank : dim2;

        int s1 = input.Shape[d1];
        int s2 = input.Shape[d2];
        int dLen;
        if (offset >= 0) dLen = Math.Min(s1, s2 - offset);
        else dLen = Math.Min(s1 + offset, s2);
        if (dLen < 0) dLen = 0;

        // Output shape: drop d1 & d2, append dLen as last dim.
        var outShape = new int[rank - 1];
        int oi = 0;
        for (int i = 0; i < rank; i++)
            if (i != d1 && i != d2) outShape[oi++] = input.Shape[i];
        outShape[oi] = dLen;

        var result = new Tensor<T>(outShape);
        var inD = input.GetDataArray();
        var rD = result.GetDataArray();

        int batch = 1;
        for (int i = 0; i < outShape.Length - 1; i++) batch *= outShape[i];
        int inRow = input._strides[d1];
        int inCol = input._strides[d2];
        int batchStride = 0;
        for (int i = 0; i < rank; i++)
            if (i != d1 && i != d2) batchStride = Math.Max(batchStride, input._strides[i]);

        // Batch over non-diagonal axes (simple linear order for common 2D/3D cases).
        for (int b = 0; b < batch; b++)
        {
            for (int k = 0; k < dLen; k++)
            {
                int inI = offset >= 0 ? k : k - offset;
                int inJ = offset >= 0 ? k + offset : k;
                int offIn = b * batchStride + inI * inRow + inJ * inCol;
                rD[b * dLen + k] = inD[offIn];
            }
        }
        return result;
    }

    internal static Tensor<T> VecDot<T>(Tensor<T> a, Tensor<T> b, int dim)
        where T : unmanaged, IEquatable<T>, IComparable<T>
    {
        if (a is null) throw new ArgumentNullException(nameof(a));
        if (b is null) throw new ArgumentNullException(nameof(b));
        int rank = a.Rank;
        int d = dim < 0 ? dim + rank : dim;
        if (a.Shape[d] != b.Shape[d]) throw new ArgumentException("Dim sizes must match.");

        // Result shape: drop axis d.
        var outShape = new int[rank - 1];
        int oi = 0;
        for (int i = 0; i < rank; i++) if (i != d) outShape[oi++] = a._shape[i];
        if (outShape.Length == 0) outShape = new[] { 1 };

        var result = new Tensor<T>(outShape);
        var aD = a.GetDataArray();
        var bD = b.GetDataArray();
        var rD = result.GetDataArray();

        int n = a.Shape[d];
        int outer = 1;
        int inner = 1;
        for (int i = 0; i < d; i++) outer *= a._shape[i];
        for (int i = d + 1; i < rank; i++) inner *= a._shape[i];

        for (int o = 0; o < outer; o++)
        {
            for (int inn = 0; inn < inner; inn++)
            {
                double s = 0;
                for (int k = 0; k < n; k++)
                {
                    int off = o * n * inner + k * inner + inn;
                    s += ToDouble(aD[off]) * ToDouble(bD[off]);
                }
                rD[o * inner + inn] = FromDouble<T>(s);
            }
        }
        return result;
    }

    internal static Tensor<T> TensorInv<T>(Tensor<T> input, int ind)
        where T : unmanaged, IEquatable<T>, IComparable<T>
    {
        // Reshape to 2D (n × n), invert, reshape back.
        if (input is null) throw new ArgumentNullException(nameof(input));
        if (ind < 1 || ind >= input.Rank) throw new ArgumentException("ind out of range.");

        int leftSize = 1;
        int rightSize = 1;
        for (int i = 0; i < ind; i++) leftSize *= input._shape[i];
        for (int i = ind; i < input.Rank; i++) rightSize *= input._shape[i];
        if (leftSize != rightSize) throw new ArgumentException("TensorInv requires prod(shape[:ind]) == prod(shape[ind:]).");

        var reshaped = input.Reshape(new[] { leftSize, rightSize });
        var inv = LinalgInverses.Inv(reshaped);
        // Output shape: shape[ind:] + shape[:ind].
        var outShape = new int[input.Rank];
        int k = 0;
        for (int i = ind; i < input.Rank; i++) outShape[k++] = input._shape[i];
        for (int i = 0; i < ind; i++) outShape[k++] = input._shape[i];
        return inv.Reshape(outShape);
    }

    internal static Tensor<T> TensorSolve<T>(Tensor<T> a, Tensor<T> b, int[]? dims)
        where T : unmanaged, IEquatable<T>, IComparable<T>
    {
        // Reshape a to 2D (N × N), b to 1D (N), solve, reshape back.
        if (a is null) throw new ArgumentNullException(nameof(a));
        if (b is null) throw new ArgumentNullException(nameof(b));
        int nB = 1;
        for (int i = 0; i < b.Rank; i++) nB *= b._shape[i];

        // By convention, the first b.Rank axes of a match b.
        int aRows = 1;
        for (int i = 0; i < b.Rank; i++) aRows *= a._shape[i];
        int aCols = 1;
        for (int i = b.Rank; i < a.Rank; i++) aCols *= a._shape[i];
        if (aRows != aCols) throw new ArgumentException("TensorSolve shapes incompatible.");
        if (aRows != nB) throw new ArgumentException("a's leading dims must flatten to same size as b.");

        var aFlat = a.Reshape(new[] { aRows, aCols });
        var bFlat = b.Reshape(new[] { nB });
        var x = LinearSolvers.Solve(aFlat, bFlat);

        // Output shape: a.shape[b.Rank:].
        var outShape = new int[a.Rank - b.Rank];
        for (int i = 0; i < outShape.Length; i++) outShape[i] = a._shape[b.Rank + i];
        return x.Reshape(outShape);
    }

    // ── Helpers ─────────────────────────────────────────────────────────────

    private static Tensor<T> Eye<T>(Tensor<T> like, int m)
        where T : unmanaged, IEquatable<T>, IComparable<T>
    {
        var shape = (int[])like._shape.Clone();
        var eye = new Tensor<T>(shape);
        var data = eye.GetDataArray();
        int batch = 1;
        for (int i = 0; i < shape.Length - 2; i++) batch *= shape[i];
        for (int b = 0; b < batch; b++)
            for (int i = 0; i < m; i++)
                data[b * m * m + i * m + i] = FromDouble<T>(1.0);
        return eye;
    }

    private static Tensor<T> Clone<T>(Tensor<T> t) where T : unmanaged, IEquatable<T>, IComparable<T>
    {
        var copy = new Tensor<T>((int[])t._shape.Clone());
        Array.Copy(t.GetDataArray(), copy.GetDataArray(), t.Length);
        return copy;
    }

    private static Tensor<T> MatMul<T>(Tensor<T> a, Tensor<T> b)
        where T : unmanaged, IEquatable<T>, IComparable<T>
    {
        // Simple batched matmul — 2D or 3D+ with matching leading dims.
        int rA = a.Rank, rB = b.Rank;
        int m = a.Shape[rA - 2];
        int k = a.Shape[rA - 1];
        if (b.Shape[rB - 2] != k) throw new ArgumentException("MatMul shape mismatch.");
        int n = b.Shape[rB - 1];

        var outShape = (int[])a._shape.Clone();
        outShape[rA - 1] = n;
        var result = new Tensor<T>(outShape);
        var aD = a.GetDataArray();
        var bD = b.GetDataArray();
        var rD = result.GetDataArray();

        int batch = 1;
        for (int i = 0; i < rA - 2; i++) batch *= a._shape[i];

        for (int bi = 0; bi < batch; bi++)
        {
            for (int i = 0; i < m; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    double s = 0;
                    for (int l = 0; l < k; l++)
                        s += ToDouble(aD[bi * m * k + i * k + l]) * ToDouble(bD[bi * k * n + l * n + j]);
                    rD[bi * m * n + i * n + j] = FromDouble<T>(s);
                }
            }
        }
        return result;
    }

    private static Tensor<T> MultiDotRec<T>(
        IReadOnlyList<Tensor<T>> ms, int[,] split, int i, int j)
        where T : unmanaged, IEquatable<T>, IComparable<T>
    {
        if (i == j) return ms[i];
        int s = split[i, j];
        return MatMul(MultiDotRec(ms, split, i, s), MultiDotRec(ms, split, s + 1, j));
    }

    private static double[] MatMulMat(double[] a, double[] b, int n)
    {
        var r = new double[n * n];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
            {
                double s = 0;
                for (int k = 0; k < n; k++) s += a[i * n + k] * b[k * n + j];
                r[i * n + j] = s;
            }
        return r;
    }

    private static double[] SolveDouble(double[] A, double[] B, int n)
    {
        // LU with partial pivoting on doubles, then forward + back substitution.
        var a = (double[])A.Clone();
        var b = (double[])B.Clone();
        var piv = new int[n];
        for (int j = 0; j < n; j++)
        {
            int p = j;
            double mx = Math.Abs(a[j * n + j]);
            for (int i = j + 1; i < n; i++)
            {
                if (Math.Abs(a[i * n + j]) > mx) { mx = Math.Abs(a[i * n + j]); p = i; }
            }
            piv[j] = p;
            if (p != j)
            {
                for (int c = 0; c < n; c++) (a[p * n + c], a[j * n + c]) = (a[j * n + c], a[p * n + c]);
                for (int c = 0; c < n; c++) (b[p * n + c], b[j * n + c]) = (b[j * n + c], b[p * n + c]);
            }
            double pv = a[j * n + j];
            if (pv == 0) continue;
            for (int i = j + 1; i < n; i++)
            {
                double f = a[i * n + j] / pv;
                a[i * n + j] = f;
                for (int c = j + 1; c < n; c++) a[i * n + c] -= f * a[j * n + c];
                for (int c = 0; c < n; c++) b[i * n + c] -= f * b[j * n + c];
            }
        }
        var x = new double[n * n];
        for (int j = 0; j < n; j++)
        {
            for (int i = n - 1; i >= 0; i--)
            {
                double s = b[i * n + j];
                for (int c = i + 1; c < n; c++) s -= a[i * n + c] * x[c * n + j];
                x[i * n + j] = a[i * n + i] == 0 ? 0 : s / a[i * n + i];
            }
        }
        return x;
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
