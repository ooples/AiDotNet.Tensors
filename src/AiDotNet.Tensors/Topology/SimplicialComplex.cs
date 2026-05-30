using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Tensors.Engines.BlasManaged;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Topology;

/// <summary>
/// Represents a simplicial complex with boundary and Laplacian operators.
/// </summary>
public sealed class SimplicialComplex
{
    private readonly Dictionary<int, HashSet<Simplex>> _simplicesByDimension = new();

    public int MaxDimension => _simplicesByDimension.Count == 0 ? -1 : _simplicesByDimension.Keys.Max();

    public void AddSimplex(Simplex simplex, bool includeFaces = true)
    {
        if (simplex is null)
            throw new ArgumentNullException(nameof(simplex));

        AddSimplexInternal(simplex);

        if (!includeFaces || simplex.Dimension <= 0)
            return;

        foreach (var face in simplex.Boundary())
        {
            AddSimplex(face.Face, true);
        }
    }

    public IReadOnlyList<Simplex> GetSimplices(int dimension)
    {
        if (!_simplicesByDimension.TryGetValue(dimension, out var set))
            return Array.Empty<Simplex>();

        return set.OrderBy(s => string.Join(",", s.Vertices)).ToList();
    }

    public Matrix<T> BoundaryOperator<T>(int k)
    {
        var kSimplices = GetSimplices(k);
        var kMinusOneSimplices = GetSimplices(k - 1);
        var ops = MathHelper.GetNumericOperations<T>();

        var boundary = new Matrix<T>(kMinusOneSimplices.Count, kSimplices.Count);
        if (kSimplices.Count == 0 || kMinusOneSimplices.Count == 0)
            return boundary;

        var rowMap = new Dictionary<Simplex, int>();
        for (int i = 0; i < kMinusOneSimplices.Count; i++)
            rowMap[kMinusOneSimplices[i]] = i;

        for (int col = 0; col < kSimplices.Count; col++)
        {
            var simplex = kSimplices[col];
            foreach (var (sign, face) in simplex.Boundary())
            {
                if (!rowMap.TryGetValue(face, out int row))
                    continue;

                boundary[row, col] = ops.FromDouble(sign);
            }
        }

        return boundary;
    }

    public Matrix<T> IncidenceMatrix<T>(int k)
    {
        var boundary = BoundaryOperator<T>(k);
        var ops = MathHelper.GetNumericOperations<T>();

        for (int i = 0; i < boundary.Rows; i++)
        {
            for (int j = 0; j < boundary.Columns; j++)
            {
                if (!ops.Equals(boundary[i, j], ops.Zero))
                    boundary[i, j] = ops.One;
            }
        }

        return boundary;
    }

    public Matrix<T> HodgeLaplacian<T>(int k) where T : unmanaged
    {
        var kSimplices = GetSimplices(k);
        if (kSimplices.Count == 0)
            return new Matrix<T>(0, 0);

        // #379: the two Hodge terms are symmetric rank-k updates (Bᵀ·B and B·Bᵀ),
        // computed via the SYRK kernel (≈half the FLOPs of a dense GEMM) then
        // symmetrized to a full matrix before the element-wise add.
        Matrix<T> result = new Matrix<T>(kSimplices.Count, kSimplices.Count);
        if (k > 0)
        {
            var bK = BoundaryOperator<T>(k);                 // Bₖᵀ·Bₖ
            result = (Matrix<T>)result.Add(SyrkFull(bK, trans: true));
        }

        if (k < MaxDimension)
        {
            var bKPlus = BoundaryOperator<T>(k + 1);
            if (bKPlus.Rows > 0 && bKPlus.Columns > 0)
                result = (Matrix<T>)result.Add(SyrkFull(bKPlus, trans: false)); // Bₖ₊₁·Bₖ₊₁ᵀ
        }

        return result;
    }

    /// <summary>
    /// Computes the full symmetric matrix α=1·op(A)·op(A)ᵀ via the SYRK kernel
    /// (lower triangle) then mirrors it to the upper triangle. trans=true gives
    /// Aᵀ·A (n = A.Columns); trans=false gives A·Aᵀ (n = A.Rows).
    /// </summary>
    private static Matrix<T> SyrkFull<T>(Matrix<T> a, bool trans) where T : unmanaged
    {
        var ops = MathHelper.GetNumericOperations<T>();
        int n = trans ? a.Columns : a.Rows;
        int kk = trans ? a.Rows : a.Columns;
        var term = new Matrix<T>(n, n);
        if (kk == 0 || n == 0) return term;

        BlasManaged.Syrk<T>(
            Uplo.Lower, trans, n, kk, ops.One,
            a.AsSpan(), a.Columns, ops.Zero, term.AsWritableSpan(), n);

        // Mirror lower → upper to produce the full symmetric matrix.
        var span = term.AsWritableSpan();
        for (int i = 0; i < n; i++)
            for (int j = 0; j < i; j++)
                span[j * n + i] = span[i * n + j];
        return term;
    }

    private void AddSimplexInternal(Simplex simplex)
    {
        if (!_simplicesByDimension.TryGetValue(simplex.Dimension, out var set))
        {
            set = new HashSet<Simplex>();
            _simplicesByDimension[simplex.Dimension] = set;
        }

        set.Add(simplex);
    }
}
