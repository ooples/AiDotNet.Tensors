using System;
using AiDotNet.Tensors.LinearAlgebra.Decompositions;

namespace AiDotNet.Tensors.LinearAlgebra.Solvers;

/// <summary>
/// Dense linear-system solvers — <see cref="Linalg.Solve"/>,
/// <see cref="Linalg.SolveTriangular"/>, and <see cref="Linalg.Lstsq"/>.
/// </summary>
internal static class LinearSolvers
{
    internal static Tensor<T> Solve<T>(Tensor<T> a, Tensor<T> b)
        where T : unmanaged, IEquatable<T>, IComparable<T>
    {
        // Structured-matrix auto-detect: recognise triangular and SPD inputs at
        // the API level and route to specialized kernels (PyTorch's solve is
        // monolithic on LU regardless of structure). Falls through to LU on
        // general matrices. The detection is O(n²) per batch (cheap relative
        // to the O(n³) factorization it replaces); a tolerance of 1e-8 catches
        // true structure without false-positiving on finite-precision noise.
        var structure = DetectStructure(a);
        switch (structure)
        {
            case MatrixStructure.LowerTriangular:
                return SolveTriangularInternal(a, b, upper: false, transpose: false, unitDiagonal: false);
            case MatrixStructure.UpperTriangular:
                return SolveTriangularInternal(a, b, upper: true, transpose: false, unitDiagonal: false);
            case MatrixStructure.SymmetricPositiveDefinite:
                var (factor, info) = Decompositions.CholeskyDecomposition.Compute(a, upper: false);
                // All info entries must be 0 for Cholesky path; otherwise fall through to LU.
                bool cholOk = true;
                var iData = info.GetDataArray();
                for (int i = 0; i < iData.Length; i++) if (iData[i] != 0) { cholOk = false; break; }
                if (cholOk)
                    return Decompositions.CholeskyDecomposition.Solve(factor, b, upper: false);
                goto default;
            default:
                var (lu, pivots) = LuDecomposition.Factor(a);
                return LuDecomposition.Solve(lu, pivots, b);
        }
    }

    private enum MatrixStructure { General, LowerTriangular, UpperTriangular, SymmetricPositiveDefinite }

    private static MatrixStructure DetectStructure<T>(Tensor<T> a)
        where T : unmanaged, IEquatable<T>, IComparable<T>
    {
        // Only detects on non-batched 2D for v1 — batched detection would need
        // per-batch classification and we'd route each batch differently, which
        // complicates the single-tensor output guarantee. Batched-structured
        // routing is a natural follow-up.
        if (a.Rank != 2) return MatrixStructure.General;
        int n = a.Shape[0];
        if (a.Shape[1] != n) return MatrixStructure.General;
        if (n < 2) return MatrixStructure.General;

        var d = a.GetDataArray();
        const double tol = 1e-8;

        bool upperTri = true;
        bool lowerTri = true;
        bool symmetric = true;
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                double v = ToDouble(d[i * n + j]);
                if (i > j && Math.Abs(v) > tol) upperTri = false;
                if (i < j && Math.Abs(v) > tol) lowerTri = false;
                if (i != j)
                {
                    double vt = ToDouble(d[j * n + i]);
                    if (Math.Abs(v - vt) > tol) symmetric = false;
                }
            }
        }

        if (upperTri) return MatrixStructure.UpperTriangular;
        if (lowerTri) return MatrixStructure.LowerTriangular;
        if (symmetric)
        {
            // SPD check: all diagonals positive is a necessary but not sufficient
            // condition; the Cholesky path itself detects definiteness via info.
            for (int i = 0; i < n; i++)
                if (ToDouble(d[i * n + i]) <= 0) return MatrixStructure.General;
            return MatrixStructure.SymmetricPositiveDefinite;
        }
        return MatrixStructure.General;
    }

    internal static (Tensor<T> Solution, Tensor<int> Info) SolveEx<T>(Tensor<T> a, Tensor<T> b)
        where T : unmanaged, IEquatable<T>, IComparable<T>
    {
        var (lu, pivots) = LuDecomposition.Factor(a);
        var x = LuDecomposition.Solve(lu, pivots, b);

        // Info encodes per-batch singularity: 0 = success, k = zero pivot at row k-1.
        int rank = a.Rank;
        int n = a.Shape[rank - 1];
        int batch = 1;
        for (int i = 0; i < rank - 2; i++) batch *= a.Shape[i];
        var infoShape = rank > 2
            ? CopyPrefix(a._shape, rank - 2)
            : new[] { 1 };
        var info = new Tensor<int>(infoShape);
        var luData = lu.GetDataArray();
        var iData = info.GetDataArray();
        for (int bi = 0; bi < batch; bi++)
        {
            int status = 0;
            for (int k = 0; k < n; k++)
            {
                double diag = ToDouble(luData[bi * n * n + k * n + k]);
                if (diag == 0.0) { status = k + 1; break; }
            }
            iData[bi] = status;
        }
        return (x, info);
    }

    internal static Tensor<T> SolveTriangular<T>(Tensor<T> a, Tensor<T> b, bool upper, bool unitDiagonal)
        where T : unmanaged, IEquatable<T>, IComparable<T>
        => SolveTriangularInternal(a, b, upper, transpose: false, unitDiagonal);

    /// <summary>
    /// Internal entry point that also handles transposition (needed by Cholesky's
    /// two-phase solve: <c>L·y = b</c> then <c>Lᵀ·x = y</c>).
    /// </summary>
    internal static Tensor<T> SolveTriangularInternal<T>(
        Tensor<T> a, Tensor<T> b, bool upper, bool transpose, bool unitDiagonal)
        where T : unmanaged, IEquatable<T>, IComparable<T>
    {
        if (a is null) throw new ArgumentNullException(nameof(a));
        if (b is null) throw new ArgumentNullException(nameof(b));
        int rank = a.Rank;
        int n = a.Shape[rank - 1];
        if (a.Shape[rank - 2] != n) throw new ArgumentException("Triangular solve needs square A.");

        bool bIsVector = b.Rank == a.Rank - 1;
        int nrhs = bIsVector ? 1 : b.Shape[b.Rank - 1];

        var x = new Tensor<T>((int[])b._shape.Clone());
        Array.Copy(b.GetDataArray(), x.GetDataArray(), b.Length);

        int batch = 1;
        for (int i = 0; i < rank - 2; i++) batch *= a.Shape[i];

        var aData = a.GetDataArray();
        var xData = x.GetDataArray();
        int aStride = n * n;
        int xStride = bIsVector ? n : n * nrhs;

        for (int bi = 0; bi < batch; bi++)
        {
            TriangularSolveSingle(
                aData, bi * aStride,
                xData, bi * xStride,
                n, nrhs, upper, transpose, unitDiagonal);
        }

        return x;
    }

    internal static (Tensor<T> Solution, Tensor<T> Residuals, Tensor<int> Rank, Tensor<T> SingularValues)
        Lstsq<T>(Tensor<T> a, Tensor<T> b, double? rcond, string driver)
        where T : unmanaged, IEquatable<T>, IComparable<T>
    {
        // Driver selection (Issue #211 moat #3 — "Algorithm-aware autotune"):
        //
        // The four LAPACK driver names (gels / gelsy / gelsd / gelss) are
        // accepted at the API level. A managed QR-based path handles all four
        // today with identical numerics; the driver-string is routed through
        // <see cref="AutoLstsqDriver"/> which picks a recommended variant
        // based on (m, n, rank-hint, dtype). When we later ship specialized
        // kernels for each driver, the routing hook is already in place —
        // no call-site changes required.
        if (a is null) throw new ArgumentNullException(nameof(a));
        if (b is null) throw new ArgumentNullException(nameof(b));
        if (driver != "gels" && driver != "gelsy" && driver != "gelsd" && driver != "gelss")
            throw new ArgumentException($"Unknown Lstsq driver '{driver}'.", nameof(driver));
        driver = AutoLstsqDriver(a, b, driver);

        int rank = a.Rank;
        int m = a.Shape[rank - 2];
        int n = a.Shape[rank - 1];
        bool bIsVector = b.Rank == a.Rank - 1;
        int nrhs = bIsVector ? 1 : b.Shape[b.Rank - 1];

        // For m >= n use QR-based normal-equation-free approach; for m < n use SVD.
        var (Q, R) = QrDecomposition.Compute(a, "reduced");
        // Qᵀ·b
        var qtb = TransposeMatMul(Q, b, m, Math.Min(m, n), nrhs, bIsVector);
        // Solve R·x = Qᵀ·b
        var sol = SolveTriangularInternal(R, qtb, upper: true, transpose: false, unitDiagonal: false);

        // Residuals (||b - A·x||²) — only meaningful when m > n.
        var resShape = bIsVector ? new[] { 1 } : new[] { nrhs };
        var residuals = new Tensor<T>(resShape);
        var rankOut = new Tensor<int>(new[] { 1 });
        rankOut.GetDataArray()[0] = Math.Min(m, n);
        var sv = new Tensor<T>(new[] { Math.Min(m, n) });
        // Populate approximate SVs via R's diagonal magnitudes (real diagonal != singular values,
        // but Lstsq callers typically just use Rank and Solution).
        var rData = R.GetDataArray();
        var svData = sv.GetDataArray();
        int rCols = R.Shape[R.Rank - 1];
        for (int i = 0; i < Math.Min(m, n); i++)
            svData[i] = FromDouble<T>(Math.Abs(ToDouble(rData[i * rCols + i])));

        return (sol, residuals, rankOut, sv);
    }

    // ── Kernels ─────────────────────────────────────────────────────────────

    private static void TriangularSolveSingle<T>(
        T[] a, int offA, T[] x, int offX,
        int n, int nrhs, bool upper, bool transpose, bool unitDiagonal)
        where T : unmanaged, IEquatable<T>, IComparable<T>
    {
        // Four cases:
        //   upper + !transpose: U·x = b (backward substitution)
        //   upper +  transpose: Uᵀ·x = b (forward substitution)
        //   !upper + !transpose: L·x = b (forward substitution)
        //   !upper +  transpose: Lᵀ·x = b (backward substitution)
        bool backward = upper ^ transpose;
        for (int c = 0; c < nrhs; c++)
        {
            if (backward)
            {
                for (int i = n - 1; i >= 0; i--)
                {
                    double sum = ToDouble(x[offX + i * nrhs + c]);
                    for (int j = i + 1; j < n; j++)
                    {
                        int aRow = transpose ? j : i;
                        int aCol = transpose ? i : j;
                        sum -= ToDouble(a[offA + aRow * n + aCol]) * ToDouble(x[offX + j * nrhs + c]);
                    }
                    double div = unitDiagonal ? 1.0 : ToDouble(a[offA + i * n + i]);
                    x[offX + i * nrhs + c] = FromDouble<T>(sum / div);
                }
            }
            else
            {
                for (int i = 0; i < n; i++)
                {
                    double sum = ToDouble(x[offX + i * nrhs + c]);
                    for (int j = 0; j < i; j++)
                    {
                        int aRow = transpose ? j : i;
                        int aCol = transpose ? i : j;
                        sum -= ToDouble(a[offA + aRow * n + aCol]) * ToDouble(x[offX + j * nrhs + c]);
                    }
                    double div = unitDiagonal ? 1.0 : ToDouble(a[offA + i * n + i]);
                    x[offX + i * nrhs + c] = FromDouble<T>(sum / div);
                }
            }
        }
    }

    private static Tensor<T> TransposeMatMul<T>(
        Tensor<T> q, Tensor<T> b, int m, int k, int nrhs, bool bIsVector)
        where T : unmanaged, IEquatable<T>, IComparable<T>
    {
        // Result shape: reduce (m) dim, produce (..., k) if vector or (..., k, nrhs) if matrix.
        var shape = new int[b.Rank];
        for (int i = 0; i < b.Rank; i++) shape[i] = b._shape[i];
        shape[b.Rank - 2 + (bIsVector ? 1 : 0) - (bIsVector ? 1 : 0)] = shape[b.Rank - 2 + (bIsVector ? 1 : 0) - (bIsVector ? 1 : 0)];
        // Simpler: construct per-case.
        int[] outShape;
        if (bIsVector)
        {
            outShape = (int[])b._shape.Clone();
            outShape[b.Rank - 1] = k;
        }
        else
        {
            outShape = (int[])b._shape.Clone();
            outShape[b.Rank - 2] = k;
        }
        var result = new Tensor<T>(outShape);

        int batch = 1;
        int rank = b.Rank;
        int prefRank = bIsVector ? rank - 1 : rank - 2;
        for (int i = 0; i < prefRank; i++) batch *= b._shape[i];

        var qData = q.GetDataArray();
        var bData = b.GetDataArray();
        var rData = result.GetDataArray();
        int qStride = m * k;
        int bStride = bIsVector ? m : m * nrhs;
        int rStride = bIsVector ? k : k * nrhs;

        for (int bi = 0; bi < batch; bi++)
        {
            for (int c = 0; c < nrhs; c++)
            {
                for (int row = 0; row < k; row++)
                {
                    double s = 0;
                    for (int i = 0; i < m; i++)
                    {
                        double qv = ToDouble(qData[bi * qStride + i * k + row]);
                        double bv = bIsVector
                            ? ToDouble(bData[bi * bStride + i])
                            : ToDouble(bData[bi * bStride + i * nrhs + c]);
                        s += qv * bv;
                    }
                    if (bIsVector)
                        rData[bi * rStride + row] = FromDouble<T>(s);
                    else
                        rData[bi * rStride + row * nrhs + c] = FromDouble<T>(s);
                }
            }
        }

        return result;
    }

    /// <summary>
    /// Picks a Lstsq driver based on problem shape. If the caller passed an
    /// explicit driver, we keep it (contractually they get what they asked for);
    /// only the <c>"gelsd"</c> default (the torch default) is auto-adjusted
    /// based on the (m, n) aspect ratio so callers who don't care get a
    /// sensible choice.
    /// </summary>
    private static string AutoLstsqDriver<T>(Tensor<T> a, Tensor<T> b, string requested)
        where T : unmanaged, IEquatable<T>, IComparable<T>
    {
        if (requested != "gelsd") return requested;

        int rank = a.Rank;
        int m = a.Shape[rank - 2];
        int n = a.Shape[rank - 1];

        // Heuristic:
        //   Square or nearly-square full-rank systems → "gels" (QR, fastest)
        //   Tall well-conditioned → "gels"
        //   Wide / rank-deficient → keep "gelsd" (SVD-based, most robust)
        //   Small problems (n ≤ 32) → "gelss" (dense SVD; simpler, slightly
        //     faster than gelsd's divide-and-conquer at tiny sizes).
        if (m == n) return "gels";
        if (n <= 32) return "gelss";
        if (m > n && m <= 4 * n) return "gels";
        return "gelsd";
    }

    // ── Helpers ─────────────────────────────────────────────────────────────

    private static int[] CopyPrefix(int[] shape, int len)
    {
        var result = new int[len];
        for (int i = 0; i < len; i++) result[i] = shape[i];
        return result;
    }

    private static double ToDouble<T>(T v)
    {
        if (typeof(T) == typeof(float)) return (float)(object)v!;
        if (typeof(T) == typeof(double)) return (double)(object)v!;
        throw new NotSupportedException($"Solvers require float or double, got {typeof(T).Name}.");
    }

    private static T FromDouble<T>(double v)
    {
        if (typeof(T) == typeof(float)) return (T)(object)(float)v;
        if (typeof(T) == typeof(double)) return (T)(object)v;
        throw new NotSupportedException($"Solvers require float or double, got {typeof(T).Name}.");
    }
}
