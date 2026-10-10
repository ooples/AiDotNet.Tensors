using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Tensors.LinearAlgebra;

/// <summary>
/// torch.linalg / torch functions over the existing decompositions: logdet, the Cholesky solve and inverse, the LU
/// unpacking, and the LAPACK-style Householder pair geqrf / ormqr. Matrices are the last two axes; leading axes batch.
/// </summary>
public static partial class Linalg
{
    /// <summary>log(det(A)): NaN where the determinant is negative, -∞ where it is zero (<c>torch.logdet</c>).</summary>
    public static Tensor<T> Logdet<T>(Tensor<T> input)
        where T : unmanaged, IEquatable<T>, IComparable<T>
        => AiDotNetEngine.Current.TensorLog(Det(input));

    /// <summary>
    /// Solves A·X = B given A's Cholesky factor (<c>torch.cholesky_solve</c>): A = L·Lᵀ for the lower factor, or Uᵀ·U
    /// when <paramref name="upper"/>.
    /// </summary>
    public static Tensor<T> CholeskySolve<T>(Tensor<T> b, Tensor<T> factor, bool upper = false)
        where T : unmanaged, IEquatable<T>, IComparable<T>
    {
        if (b == null) throw new ArgumentNullException(nameof(b));
        if (factor == null) throw new ArgumentNullException(nameof(factor));
        var transposed = TransposeMatrices(factor);
        // Lower: L·y = b, then Lᵀ·x = y. Upper: Uᵀ·y = b, then U·x = y.
        var y = SolveTriangular(upper ? transposed : factor, b, upper: false);
        return SolveTriangular(upper ? factor : transposed, y, upper: true);
    }

    /// <summary>A⁻¹ given A's Cholesky factor (<c>torch.cholesky_inverse</c>).</summary>
    public static Tensor<T> CholeskyInverse<T>(Tensor<T> factor, bool upper = false)
        where T : unmanaged, IEquatable<T>, IComparable<T>
    {
        if (factor == null) throw new ArgumentNullException(nameof(factor));
        return CholeskySolve(Identity<T>(factor), factor, upper);
    }

    // The [..., n, n] identity matching a [..., n, n] tensor.
    private static Tensor<T> Identity<T>(Tensor<T> like)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        int n = like.Shape[like.Rank - 1];
        var identity = new Tensor<T>(like.Shape.ToArray());
        using var spanLease = identity.LeaseWritable();
        var span = spanLease.Span;
        for (int start = 0; start < span.Length; start += n * n)
            for (int i = 0; i < n; i++) span[start + i * n + i] = ops.One;
        return identity;
    }

    private static Tensor<T> TransposeMatrices<T>(Tensor<T> t)
    {
        var perm = new int[t.Rank];
        for (int i = 0; i < perm.Length; i++) perm[i] = i;
        perm[t.Rank - 1] = t.Rank - 2;
        perm[t.Rank - 2] = t.Rank - 1;
        return AiDotNetEngine.Current.TensorPermute(t, perm);
    }

    /// <summary>
    /// (P, L, U) from a factorization in <see cref="LuFactor{T}"/>'s packed form (<c>torch.lu_unpack</c>), so that
    /// A = P·L·U. The pivots are LuFactor's 0-based row swaps (PyTorch's are 1-based).
    /// </summary>
    public static (Tensor<T> P, Tensor<T> L, Tensor<T> U) LuUnpack<T>(Tensor<T> lu, Tensor<int> pivots)
        where T : unmanaged, IEquatable<T>, IComparable<T>
    {
        if (lu == null) throw new ArgumentNullException(nameof(lu));
        if (pivots == null) throw new ArgumentNullException(nameof(pivots));
        var ops = MathHelper.GetNumericOperations<T>();
        int m = lu.Shape[lu.Rank - 2], n = lu.Shape[lu.Rank - 1], k = Math.Min(m, n);
        int batch = lu.Length / (m * n);
        var lead = System.Linq.Enumerable.ToArray(System.Linq.Enumerable.Take(lu.Shape.ToArray(), lu.Rank - 2));
        var source = (lu.IsContiguous ? lu : lu.Contiguous()).AsSpan().ToArray();
        var piv = (pivots.IsContiguous ? pivots : pivots.Contiguous()).AsSpan().ToArray();
        var p = new Tensor<T>([.. lead, m, m]);
        var l = new Tensor<T>([.. lead, m, k]);
        var u = new Tensor<T>([.. lead, k, n]);
        using var psLease = p.LeaseWritable();
        var ps = psLease.Span;
        using var lsLease = l.LeaseWritable();
        var ls = lsLease.Span;
        using var usLease = u.LeaseWritable();
        var us = usLease.Span;
        for (int b = 0; b < batch; b++)
        {
            for (int i = 0; i < m; i++)
                for (int j = 0; j < n; j++)
                {
                    var v = source[b * m * n + i * n + j];
                    if (j < k && i > j) ls[b * m * k + i * k + j] = v;
                    if (i < k && j >= i) us[b * k * n + i * n + j] = v;
                }
            for (int i = 0; i < k; i++) ls[b * m * k + i * k + i] = ops.One;
            // Apply the row swaps to the identity, then P is its transpose (A = P·L·U).
            var order = new int[m];
            for (int i = 0; i < m; i++) order[i] = i;
            for (int i = 0; i < k; i++)
            {
                int swap = piv[b * k + i];
                (order[i], order[swap]) = (order[swap], order[i]);
            }
            for (int i = 0; i < m; i++) ps[b * m * m + order[i] * m + i] = ops.One;
        }
        return (p, l, u);
    }

    /// <summary>
    /// Householder QR in LAPACK's packed form (<c>torch.geqrf</c>): R on and above the diagonal, the Householder
    /// vectors (implicit unit head) below it, and their scaling factors τ, so that Q = H₀·H₁·…·Hₖ₋₁ with
    /// Hᵢ = I - τᵢ·vᵢ·vᵢᵀ.
    /// </summary>
    public static (Tensor<T> A, Tensor<T> Tau) Geqrf<T>(Tensor<T> input)
        where T : unmanaged, IEquatable<T>, IComparable<T>
    {
        if (input == null) throw new ArgumentNullException(nameof(input));
        var ops = MathHelper.GetNumericOperations<T>();
        int m = input.Shape[input.Rank - 2], n = input.Shape[input.Rank - 1], k = Math.Min(m, n);
        int batch = input.Length / (m * n);
        var a = ToDoubles(input);
        var tau = new double[batch * k];
        for (int b = 0; b < batch; b++)
        {
            int o = b * m * n;
            for (int j = 0; j < k; j++)
            {
                double norm = 0;
                for (int i = j; i < m; i++) norm += a[o + i * n + j] * a[o + i * n + j];
                norm = Math.Sqrt(norm);
                double alpha = a[o + j * n + j];
                if (norm == 0) continue;
                double beta = alpha >= 0 ? -norm : norm;
                double scale = 1 / (alpha - beta);
                for (int i = j + 1; i < m; i++) a[o + i * n + j] *= scale;
                double t = (beta - alpha) / beta;
                tau[b * k + j] = t;
                a[o + j * n + j] = beta;
                // A[j:, j+1:] -= τ·v·(vᵀ·A[j:, j+1:]), v = (1, a[j+1:, j]).
                for (int c = j + 1; c < n; c++)
                {
                    double dot = a[o + j * n + c];
                    for (int i = j + 1; i < m; i++) dot += a[o + i * n + j] * a[o + i * n + c];
                    dot *= t;
                    a[o + j * n + c] -= dot;
                    for (int i = j + 1; i < m; i++) a[o + i * n + c] -= dot * a[o + i * n + j];
                }
            }
        }
        var shape = input.Shape.ToArray();
        var tauShape = System.Linq.Enumerable.ToArray(System.Linq.Enumerable.Take(shape, input.Rank - 2));
        return (FromDoubles<T>(a, shape), FromDoubles<T>(tau, [.. tauShape, k]));
    }

    /// <summary>
    /// Multiplies <paramref name="other"/> by Q (or Qᵀ) from <see cref="Geqrf{T}"/>'s output, on the left or right
    /// (<c>torch.ormqr</c>).
    /// </summary>
    public static Tensor<T> Ormqr<T>(Tensor<T> a, Tensor<T> tau, Tensor<T> other, bool left = true, bool transpose = false)
        where T : unmanaged, IEquatable<T>, IComparable<T>
    {
        if (a == null) throw new ArgumentNullException(nameof(a));
        if (tau == null) throw new ArgumentNullException(nameof(tau));
        if (other == null) throw new ArgumentNullException(nameof(other));
        // torch.ormqr's shape contract: matching batch dimensions, Q's order m matching other's multiplied side,
        // and k ≤ min(m, n) reflectors.
        if (a.Rank < 2 || other.Rank != a.Rank || tau.Rank != a.Rank - 1)
            throw new ArgumentException($"ormqr needs a and other of the same rank ≥ 2 and tau one rank lower; got {a.Rank}, {other.Rank} and {tau.Rank}.");
        for (int axis = 0; axis < a.Rank - 2; axis++)
            if (other.Shape[axis] != a.Shape[axis] || tau.Shape[axis] != a.Shape[axis])
                throw new ArgumentException("a, tau and other must have the same batch dimensions.");
        int m = a.Shape[a.Rank - 2], n = a.Shape[a.Rank - 1], k = tau.Shape[tau.Rank - 1];
        int rows = other.Shape[other.Rank - 2], cols = other.Shape[other.Rank - 1];
        if ((left ? rows : cols) != m)
            throw new ArgumentException($"Q is {m}x{m}, but other's {(left ? "rows" : "columns")} number {(left ? rows : cols)}.", nameof(other));
        if (k > Math.Min(m, n)) throw new ArgumentException($"tau holds {k} reflectors, more than min(m, n) = {Math.Min(m, n)}.", nameof(tau));
        int batch = rows * cols == 0 ? 0 : other.Length / (rows * cols);
        var av = ToDoubles(a);
        var tv = ToDoubles(tau);
        var c = ToDoubles(other);
        for (int b = 0; b < batch; b++)
        {
            int ao = b * m * n, co = b * rows * cols;
            // Q·C applies H_{k-1} first; Qᵀ·C applies H₀ first; on the right the orders flip.
            bool forward = left == transpose;
            for (int step = 0; step < k; step++)
            {
                int j = forward ? step : k - 1 - step;
                double t = tv[b * k + j];
                if (t == 0) continue;
                double V(int i) => i == j ? 1 : i > j ? av[ao + i * n + j] : 0;
                if (left)
                {
                    for (int col = 0; col < cols; col++)
                    {
                        double dot = 0;
                        for (int i = j; i < rows; i++) dot += V(i) * c[co + i * cols + col];
                        dot *= t;
                        for (int i = j; i < rows; i++) c[co + i * cols + col] -= dot * V(i);
                    }
                }
                else
                {
                    for (int row = 0; row < rows; row++)
                    {
                        double dot = 0;
                        for (int i = j; i < cols; i++) dot += c[co + row * cols + i] * V(i);
                        dot *= t;
                        for (int i = j; i < cols; i++) c[co + row * cols + i] -= dot * V(i);
                    }
                }
            }
        }
        return FromDoubles<T>(c, other.Shape.ToArray());
    }

    private static double[] ToDoubles<T>(Tensor<T> t)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        var span = (t.IsContiguous ? t : t.Contiguous()).AsSpan();
        var result = new double[span.Length];
        for (int i = 0; i < span.Length; i++) result[i] = ops.ToDouble(span[i]);
        return result;
    }

    private static Tensor<T> FromDoubles<T>(double[] values, int[] shape)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        var result = new Tensor<T>(shape);
        using var spanLease = result.LeaseWritable();
        var span = spanLease.Span;
        for (int i = 0; i < values.Length; i++) span[i] = ops.FromDouble(values[i]);
        return result;
    }
}
