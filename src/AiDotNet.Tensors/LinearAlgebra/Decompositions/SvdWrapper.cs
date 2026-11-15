using System;

namespace AiDotNet.Tensors.LinearAlgebra.Decompositions;

/// <summary>
/// General-purpose SVD wrapper — the <see cref="Linalg.Svd"/> /
/// <see cref="Linalg.SvdVals"/> / <see cref="Linalg.SvdLowRank"/> backends. The
/// pre-existing <see cref="SvdDecomposition"/> is specialized for low-rank
/// approximation; this wrapper provides the full <c>(U, S, Vᵀ)</c> surface that
/// PyTorch callers expect.
///
/// <para>The managed implementation is the one-sided Jacobi SVD for every element type (float iterates in
/// double). It never forms AᵀA, so small singular values keep their relative accuracy.</para>
/// </summary>
internal static class SvdWrapper
{
    internal static (Tensor<T> U, Tensor<T> S, Tensor<T> Vh) Full<T>(Tensor<T> input, bool fullMatrices)
        where T : unmanaged, IEquatable<T>, IComparable<T>
    {
        if (input is null) throw new ArgumentNullException(nameof(input));
        if (input.Rank < 2) throw new ArgumentException("SVD requires at least a 2D tensor.", nameof(input));

        int rank = input.Rank;
        int m = input.Shape[rank - 2];
        int n = input.Shape[rank - 1];
        int k = Math.Min(m, n);

        int uCols = fullMatrices ? m : k;
        int vRows = fullMatrices ? n : k;

        var uShape = (int[])input._shape.Clone();
        uShape[rank - 1] = uCols;
        var sShape = new int[rank - 1];
        for (int i = 0; i < rank - 2; i++) sShape[i] = input._shape[i];
        sShape[rank - 2] = k;
        var vhShape = (int[])input._shape.Clone();
        vhShape[rank - 2] = vRows;

        var U = new Tensor<T>(uShape);
        var S = new Tensor<T>(sShape);
        var Vh = new Tensor<T>(vhShape);

        int batch = 1;
        for (int i = 0; i < rank - 2; i++) batch *= input._shape[i];

        var inData = input.Contiguous().GetDataArray();
        using var uDataLease = U.LeaseArray();
        var uData = uDataLease.Array;
        using var sDataLease = S.LeaseArray();
        var sData = sDataLease.Array;
        using var vhDataLease = Vh.LeaseArray();
        var vhData = vhDataLease.Array;
        int inStride = m * n;
        int uStride = m * uCols;
        int sStride = k;
        int vhStride = vRows * n;

        for (int b = 0; b < batch; b++)
        {
            FullSingle(inData, b * inStride, uData, b * uStride,
                sData, b * sStride, vhData, b * vhStride, m, n, uCols, vRows);
        }

        return (U, S, Vh);
    }

    internal static Tensor<T> ValuesOnly<T>(Tensor<T> input)
        where T : unmanaged, IEquatable<T>, IComparable<T>
        => Full(input, fullMatrices: false).S;

    internal static (Tensor<T> U, Tensor<T> S, Tensor<T> Vh) LowRank<T>(Tensor<T> input, int rank, int q)
        where T : unmanaged, IEquatable<T>, IComparable<T>
    {
        // Truncated SVD: computes the full SVD and returns the top-`rank`
        // components. A full Halko 2011 randomized sketch (A·Ω →
        // orthonormalize → project → small SVD) would be cheaper at
        // rank ≪ min(M,N); that path lives behind this API for when it
        // lands, but today `q` (subspace iterations) and the randomization
        // are not yet applied.
        if (rank < 1) throw new ArgumentException("rank must be positive.", nameof(rank));
        _ = q; // reserved for future randomized sketch, see comment above.
        var full = Full(input, fullMatrices: false);
        int fullK = full.S.Shape[full.S.Rank - 1];
        int k = Math.Min(rank, fullK);

        // Truncate U, S, Vh to the top `k` components.
        int inputRank = input.Rank;
        var uShape = (int[])full.U._shape.Clone();
        uShape[inputRank - 1] = k;
        var sShape = (int[])full.S._shape.Clone();
        sShape[sShape.Length - 1] = k;
        var vhShape = (int[])full.Vh._shape.Clone();
        vhShape[inputRank - 2] = k;

        var U = new Tensor<T>(uShape);
        var S = new Tensor<T>(sShape);
        var Vh = new Tensor<T>(vhShape);

        int batch = 1;
        for (int i = 0; i < inputRank - 2; i++) batch *= input._shape[i];
        int m = input.Shape[inputRank - 2];
        int n = input.Shape[inputRank - 1];

        using var uFLease = full.U.LeaseArray();
        var uF = uFLease.Array;
        using var sFLease = full.S.LeaseArray();
        var sF = sFLease.Array;
        using var vhFLease = full.Vh.LeaseArray();
        var vhF = vhFLease.Array;
        using var uDLease = U.LeaseArray();
        var uD = uDLease.Array;
        using var sDLease = S.LeaseArray();
        var sD = sDLease.Array;
        using var vhDLease = Vh.LeaseArray();
        var vhD = vhDLease.Array;

        for (int b = 0; b < batch; b++)
        {
            for (int i = 0; i < m; i++)
                for (int j = 0; j < k; j++)
                    uD[b * m * k + i * k + j] = uF[b * m * fullK + i * fullK + j];
            for (int i = 0; i < k; i++)
                sD[b * k + i] = sF[b * fullK + i];
            for (int i = 0; i < k; i++)
                for (int j = 0; j < n; j++)
                    vhD[b * k * n + i * n + j] = vhF[b * fullK * n + i * n + j];
        }

        return (U, S, Vh);
    }

    private static void FullSingle<T>(
        T[] src, int offSrc,
        T[] u, int offU, T[] s, int offS, T[] vh, int offVh,
        int m, int n, int uCols, int vRows)
        where T : unmanaged, IEquatable<T>, IComparable<T>
    {
        // Every element type runs the one-sided Jacobi SVD (LAPACK gesvj's method), which works on A itself: the
        // Gram route this replaced eigendecomposed AᵀA, squaring the condition number, so a singular value below
        // about sqrt(eps)·σ_max came back as noise (9.2e-9 for the rank-2 [[.1,.2,.3],[.4,.5,.6],[.7,.8,.9]],
        // whose third value is about 1e-17), which made rank and nullity tolerances wrong. float is widened to
        // double for the iteration and rounded once.
        if (src is float[] srcF && u is float[] uF && s is float[] sF && vh is float[] vhF)
        {
            int k = Math.Min(m, n);
            var a = new double[m * n];
            for (int i = 0; i < a.Length; i++) a[i] = srcF[offSrc + i];
            var uD = new double[m * uCols];
            var sD = new double[k];
            var vhD = new double[vRows * n];
            GenericLinalg<double>.Svd(a, 0, uD, 0, sD, 0, vhD, 0, m, n, uCols, vRows);
            for (int i = 0; i < uD.Length; i++) uF[offU + i] = (float)uD[i];
            for (int i = 0; i < k; i++) sF[offS + i] = (float)sD[i];
            for (int i = 0; i < vhD.Length; i++) vhF[offVh + i] = (float)vhD[i];
            return;
        }

        GenericLinalg<T>.Svd(src, offSrc, u, offU, s, offS, vh, offVh, m, n, uCols, vRows);
    }
}
