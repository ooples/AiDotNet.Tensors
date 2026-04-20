using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra.Decompositions;
using AiDotNet.Tensors.LinearAlgebra.Solvers;

namespace AiDotNet.Tensors.LinearAlgebra;

/// <summary>
/// Top-level public API surface for linear algebra operations — the AiDotNet.Tensors
/// analogue of PyTorch's <c>torch.linalg</c> namespace. Every op accepts batched
/// input (leading batch dims) and returns results in the same batched shape.
///
/// <para>
/// See issue #211 for the parity checklist. The managed tier is the primary
/// implementation today; native LAPACK / cuSOLVER bindings are wired through
/// <see cref="Helpers.LapackProvider"/> for future activation without call-site
/// churn.
/// </para>
///
/// <para>
/// Design notes:
/// <list type="bullet">
///   <item>All ops operate on <see cref="Tensor{T}"/> with <c>T = float</c> or
///   <c>T = double</c>. Integer types are not supported (most decompositions are
///   inherently real-valued).</item>
///   <item>Batched ops accept any leading batch-dim count. Shape conventions follow
///   PyTorch: <c>(..., M, N)</c> for matrices, <c>(..., N)</c> for vectors.</item>
///   <item>Differentiable ops are wired through <see cref="Engines.Autodiff.DifferentiableOps"/>
///   when a <see cref="Engines.Autodiff.GradientTape{T}"/> is active.</item>
/// </list>
/// </para>
/// </summary>
public static class Linalg
{
    // ═══════════════════════════════════════════════════════════════════════
    // DECOMPOSITIONS
    // ═══════════════════════════════════════════════════════════════════════

    /// <summary>
    /// Symmetric/Hermitian eigendecomposition. Returns (eigenvalues ascending,
    /// eigenvectors as columns). Input is assumed symmetric — the upper or lower
    /// triangle is read based on <paramref name="upper"/>.
    /// </summary>
    /// <param name="input">Batched symmetric matrix of shape <c>(..., N, N)</c>.</param>
    /// <param name="upper">When true, read the upper triangle; otherwise the lower.</param>
    /// <returns>Tuple (eigenvalues <c>(..., N)</c>, eigenvectors <c>(..., N, N)</c>).</returns>
    public static (Tensor<T> Eigenvalues, Tensor<T> Eigenvectors) Eigh<T>(Tensor<T> input, bool upper = false)
        where T : unmanaged, IEquatable<T>, IComparable<T>
        => EighDecomposition.Compute(input, upper);

    /// <summary>Eigenvalues only of a symmetric/Hermitian matrix.</summary>
    public static Tensor<T> Eigvalsh<T>(Tensor<T> input, bool upper = false)
        where T : unmanaged, IEquatable<T>, IComparable<T>
        => EighDecomposition.Compute(input, upper).Eigenvalues;

    /// <summary>
    /// General (non-symmetric) eigendecomposition. Returns complex eigenvalues
    /// and right eigenvectors packed as interleaved real/imag pairs along a
    /// trailing size-2 axis.
    /// </summary>
    public static (Tensor<T> EigenvaluesReIm, Tensor<T> EigenvectorsReIm) Eig<T>(Tensor<T> input)
        where T : unmanaged, IEquatable<T>, IComparable<T>
        => EigDecomposition.Compute(input);

    /// <summary>Eigenvalues only of a general matrix (returned as real/imag pairs).</summary>
    public static Tensor<T> Eigvals<T>(Tensor<T> input)
        where T : unmanaged, IEquatable<T>, IComparable<T>
        => EigDecomposition.Compute(input).EigenvaluesReIm;

    /// <summary>QR factorization. Supports <paramref name="mode"/> = "reduced" | "complete" | "r".</summary>
    public static (Tensor<T> Q, Tensor<T> R) QR<T>(Tensor<T> input, string mode = "reduced")
        where T : unmanaged, IEquatable<T>, IComparable<T>
        => QrDecomposition.Compute(input, mode);

    /// <summary>Cholesky factorization of a symmetric positive-definite matrix.</summary>
    /// <param name="input">Batched SPD matrix of shape <c>(..., N, N)</c>.</param>
    /// <param name="upper">When true, return the upper triangular factor; otherwise lower.</param>
    public static Tensor<T> Cholesky<T>(Tensor<T> input, bool upper = false)
        where T : unmanaged, IEquatable<T>, IComparable<T>
        => CholeskyDecomposition.Compute(input, upper).Factor;

    /// <summary>Cholesky factorization returning both the factor and an <c>info</c> flag.</summary>
    public static (Tensor<T> Factor, Tensor<int> Info) CholeskyEx<T>(Tensor<T> input, bool upper = false)
        where T : unmanaged, IEquatable<T>, IComparable<T>
        => CholeskyDecomposition.Compute(input, upper);

    /// <summary>LU decomposition. Returns P, L, U such that <c>PA = LU</c> (partial pivoting).</summary>
    public static (Tensor<T> P, Tensor<T> L, Tensor<T> U) LU<T>(Tensor<T> input)
        where T : unmanaged, IEquatable<T>, IComparable<T>
        => LuDecomposition.Compute(input);

    /// <summary>LU factorization returning the packed factor and pivots (<c>getrf</c> form).</summary>
    public static (Tensor<T> LU, Tensor<int> Pivots) LuFactor<T>(Tensor<T> input)
        where T : unmanaged, IEquatable<T>, IComparable<T>
        => LuDecomposition.Factor(input);

    /// <summary>Solve <c>AX = B</c> given a precomputed LU factorization.</summary>
    public static Tensor<T> LuSolve<T>(Tensor<T> lu, Tensor<int> pivots, Tensor<T> b)
        where T : unmanaged, IEquatable<T>, IComparable<T>
        => LuDecomposition.Solve(lu, pivots, b);

    /// <summary>LDL factorization for symmetric-indefinite systems.</summary>
    public static (Tensor<T> LD, Tensor<int> Pivots) LdlFactor<T>(Tensor<T> input, bool upper = false)
        where T : unmanaged, IEquatable<T>, IComparable<T>
        => LdlDecomposition.Factor(input, upper);

    /// <summary>LDL solve.</summary>
    public static Tensor<T> LdlSolve<T>(Tensor<T> ld, Tensor<int> pivots, Tensor<T> b, bool upper = false)
        where T : unmanaged, IEquatable<T>, IComparable<T>
        => LdlDecomposition.Solve(ld, pivots, b, upper);

    /// <summary>Singular Value Decomposition — upgraded wrapper around <see cref="SvdDecomposition"/>.</summary>
    public static (Tensor<T> U, Tensor<T> S, Tensor<T> Vh) Svd<T>(Tensor<T> input, bool fullMatrices = true)
        where T : unmanaged, IEquatable<T>, IComparable<T>
        => SvdWrapper.Full(input, fullMatrices);

    /// <summary>Singular values only.</summary>
    public static Tensor<T> SvdVals<T>(Tensor<T> input)
        where T : unmanaged, IEquatable<T>, IComparable<T>
        => SvdWrapper.ValuesOnly(input);

    /// <summary>
    /// Randomized low-rank SVD via <paramref name="q"/> subspace iterations. Cheap
    /// approximation for <paramref name="rank"/> ≪ min(M, N).
    /// </summary>
    public static (Tensor<T> U, Tensor<T> S, Tensor<T> Vh) SvdLowRank<T>(
        Tensor<T> input, int rank, int q = 2)
        where T : unmanaged, IEquatable<T>, IComparable<T>
        => SvdWrapper.LowRank(input, rank, q);

    // ═══════════════════════════════════════════════════════════════════════
    // SOLVERS
    // ═══════════════════════════════════════════════════════════════════════

    /// <summary>Solve the general linear system <c>A·X = B</c>.</summary>
    public static Tensor<T> Solve<T>(Tensor<T> a, Tensor<T> b)
        where T : unmanaged, IEquatable<T>, IComparable<T>
    {
        var result = LinearSolvers.Solve(a, b);
        DifferentiableOps.RecordBinary("Linalg.Solve", result, a, b, LinalgBackward.SolveBackward<T>());
        return result;
    }

    /// <summary>Solve, returning the solution and an <c>info</c> flag per batch element.</summary>
    public static (Tensor<T> Solution, Tensor<int> Info) SolveEx<T>(Tensor<T> a, Tensor<T> b)
        where T : unmanaged, IEquatable<T>, IComparable<T>
        => LinearSolvers.SolveEx(a, b);

    /// <summary>Solve a triangular system. <paramref name="upper"/> selects U or L.</summary>
    public static Tensor<T> SolveTriangular<T>(Tensor<T> a, Tensor<T> b, bool upper, bool unitDiagonal = false)
        where T : unmanaged, IEquatable<T>, IComparable<T>
        => LinearSolvers.SolveTriangular(a, b, upper, unitDiagonal);

    /// <summary>Least-squares solve. <paramref name="driver"/> ∈ {"gels","gelsy","gelsd","gelss"}.</summary>
    public static (Tensor<T> Solution, Tensor<T> Residuals, Tensor<int> Rank, Tensor<T> SingularValues)
        Lstsq<T>(Tensor<T> a, Tensor<T> b, double? rcond = null, string driver = "gelsd")
        where T : unmanaged, IEquatable<T>, IComparable<T>
        => LinearSolvers.Lstsq(a, b, rcond, driver);

    // ═══════════════════════════════════════════════════════════════════════
    // INVERSES
    // ═══════════════════════════════════════════════════════════════════════

    /// <summary>Matrix inverse.</summary>
    public static Tensor<T> Inv<T>(Tensor<T> input)
        where T : unmanaged, IEquatable<T>, IComparable<T>
    {
        var result = LinalgInverses.Inv(input);
        DifferentiableOps.RecordUnary("Linalg.Inv", result, input, LinalgBackward.InvBackward<T>());
        return result;
    }

    /// <summary>Inverse with <c>info</c> per batch element (no throw on singular).</summary>
    public static (Tensor<T> Inverse, Tensor<int> Info) InvEx<T>(Tensor<T> input)
        where T : unmanaged, IEquatable<T>, IComparable<T>
        => LinalgInverses.InvEx(input);

    /// <summary>Moore–Penrose pseudoinverse via SVD.</summary>
    public static Tensor<T> Pinv<T>(Tensor<T> input, double? rcond = null)
        where T : unmanaged, IEquatable<T>, IComparable<T>
        => LinalgInverses.Pinv(input, rcond);

    // ═══════════════════════════════════════════════════════════════════════
    // SCALAR SUMMARIES
    // ═══════════════════════════════════════════════════════════════════════

    /// <summary>Determinant.</summary>
    public static Tensor<T> Det<T>(Tensor<T> input)
        where T : unmanaged, IEquatable<T>, IComparable<T>
    {
        var result = LinalgScalars.Det(input);
        DifferentiableOps.RecordUnary("Linalg.Det", result, input, LinalgBackward.DetBackward<T>());
        return result;
    }

    /// <summary>(sign, log|det|) decomposition — avoids overflow for ill-conditioned matrices.</summary>
    public static (Tensor<T> Sign, Tensor<T> LogAbsDet) SlogDet<T>(Tensor<T> input)
        where T : unmanaged, IEquatable<T>, IComparable<T>
    {
        var (sign, logAbs) = LinalgScalars.SlogDet(input);
        // Record backward on the log-abs branch (sign has zero gradient).
        DifferentiableOps.RecordUnary("Linalg.SlogDet", logAbs, input, LinalgBackward.SlogDetBackward<T>());
        return (sign, logAbs);
    }

    /// <summary>Matrix rank via SVD.</summary>
    public static Tensor<int> MatrixRank<T>(Tensor<T> input, double? atol = null, double? rtol = null, bool hermitian = false)
        where T : unmanaged, IEquatable<T>, IComparable<T>
        => LinalgScalars.MatrixRank(input, atol, rtol, hermitian);

    /// <summary>Condition number. <paramref name="p"/> ∈ {1, 2, ∞, -1, -2, -∞, "fro", "nuc"}.</summary>
    public static Tensor<T> Cond<T>(Tensor<T> input, object p = null!)
        where T : unmanaged, IEquatable<T>, IComparable<T>
        => LinalgScalars.Cond(input, p);

    // ═══════════════════════════════════════════════════════════════════════
    // NORMS
    // ═══════════════════════════════════════════════════════════════════════

    /// <summary>Generic norm — dispatches on rank: 1D → vector norm; 2D → matrix norm.</summary>
    public static Tensor<T> Norm<T>(Tensor<T> input, object ord = null!, int[] dim = null!, bool keepDim = false)
        where T : unmanaged, IEquatable<T>, IComparable<T>
        => LinalgNorms.Norm(input, ord, dim, keepDim);

    /// <summary>Vector norm along specified dimension(s).</summary>
    public static Tensor<T> VectorNorm<T>(Tensor<T> input, double ord = 2.0, int[] dim = null!, bool keepDim = false)
        where T : unmanaged, IEquatable<T>, IComparable<T>
        => LinalgNorms.VectorNorm(input, ord, dim, keepDim);

    /// <summary>Matrix norm. <paramref name="ord"/> ∈ {1, 2, ∞, -1, -2, -∞, "fro", "nuc"}.</summary>
    public static Tensor<T> MatrixNorm<T>(Tensor<T> input, object ord = null!, int[] dim = null!, bool keepDim = false)
        where T : unmanaged, IEquatable<T>, IComparable<T>
        => LinalgNorms.MatrixNorm(input, ord, dim, keepDim);

    // ═══════════════════════════════════════════════════════════════════════
    // STRUCTURAL
    // ═══════════════════════════════════════════════════════════════════════

    /// <summary>A^n for integer n (including negative via Inv).</summary>
    public static Tensor<T> MatrixPower<T>(Tensor<T> input, int n)
        where T : unmanaged, IEquatable<T>, IComparable<T>
    {
        var result = LinalgStructural.MatrixPower(input, n);
        DifferentiableOps.RecordUnary("Linalg.MatrixPower", result, input,
            LinalgBackward.MatrixPowerBackward<T>(), savedState: new object[] { n });
        return result;
    }

    /// <summary>Matrix exponential via Padé scaling-and-squaring.</summary>
    public static Tensor<T> MatrixExp<T>(Tensor<T> input)
        where T : unmanaged, IEquatable<T>, IComparable<T>
        => LinalgStructural.MatrixExp(input);

    /// <summary>Multiple matrix multiplication with optimal parenthesization (dynamic programming).</summary>
    public static Tensor<T> MultiDot<T>(IReadOnlyList<Tensor<T>> matrices)
        where T : unmanaged, IEquatable<T>, IComparable<T>
        => LinalgStructural.MultiDot(matrices);

    /// <summary>3D cross product along <paramref name="dim"/>.</summary>
    public static Tensor<T> Cross<T>(Tensor<T> a, Tensor<T> b, int dim = -1)
        where T : unmanaged, IEquatable<T>, IComparable<T>
        => LinalgStructural.Cross(a, b, dim);

    /// <summary>Vandermonde matrix.</summary>
    public static Tensor<T> Vander<T>(Tensor<T> x, int? n = null, bool increasing = false)
        where T : unmanaged, IEquatable<T>, IComparable<T>
        => LinalgStructural.Vander(x, n, increasing);

    /// <summary>Build an orthogonal matrix from Householder reflectors.</summary>
    public static Tensor<T> HouseholderProduct<T>(Tensor<T> reflectors, Tensor<T> tau)
        where T : unmanaged, IEquatable<T>, IComparable<T>
        => LinalgStructural.HouseholderProduct(reflectors, tau);

    /// <summary>Extract diagonal along specified axes.</summary>
    public static Tensor<T> Diagonal<T>(Tensor<T> input, int offset = 0, int dim1 = -2, int dim2 = -1)
        where T : unmanaged, IEquatable<T>, IComparable<T>
        => LinalgStructural.Diagonal(input, offset, dim1, dim2);

    /// <summary>Dot product along a specified dimension.</summary>
    public static Tensor<T> VecDot<T>(Tensor<T> a, Tensor<T> b, int dim = -1)
        where T : unmanaged, IEquatable<T>, IComparable<T>
        => LinalgStructural.VecDot(a, b, dim);

    /// <summary>Multilinear-tensor inverse under a specified number of trailing axes.</summary>
    public static Tensor<T> TensorInv<T>(Tensor<T> input, int ind = 2)
        where T : unmanaged, IEquatable<T>, IComparable<T>
        => LinalgStructural.TensorInv(input, ind);

    /// <summary>Solve a multilinear-tensor equation.</summary>
    public static Tensor<T> TensorSolve<T>(Tensor<T> a, Tensor<T> b, int[] dims = null!)
        where T : unmanaged, IEquatable<T>, IComparable<T>
        => LinalgStructural.TensorSolve(a, b, dims);

    // ═══════════════════════════════════════════════════════════════════════
    // ITERATIVE SOLVERS (bonus — beyond PyTorch)
    // ═══════════════════════════════════════════════════════════════════════

    /// <summary>
    /// Preconditioned conjugate gradient. Solves <c>A·x = b</c> for SPD <c>A</c>.
    /// This ships beyond PyTorch's surface (scipy territory there).
    /// </summary>
    public static Tensor<T> CG<T>(Tensor<T> a, Tensor<T> b, int maxIter = 1000, double tol = 1e-6,
        Func<Tensor<T>, Tensor<T>>? preconditioner = null)
        where T : unmanaged, IEquatable<T>, IComparable<T>
        => IterativeSolvers.CG(a, b, maxIter, tol, preconditioner);

    /// <summary>Generalized Minimum Residual method for general linear systems.</summary>
    public static Tensor<T> GMRES<T>(Tensor<T> a, Tensor<T> b, int maxIter = 1000, int restart = 30, double tol = 1e-6,
        Func<Tensor<T>, Tensor<T>>? preconditioner = null)
        where T : unmanaged, IEquatable<T>, IComparable<T>
        => IterativeSolvers.GMRES(a, b, maxIter, restart, tol, preconditioner);

    /// <summary>BiConjugate Gradient Stabilized for general linear systems.</summary>
    public static Tensor<T> BiCGSTAB<T>(Tensor<T> a, Tensor<T> b, int maxIter = 1000, double tol = 1e-6,
        Func<Tensor<T>, Tensor<T>>? preconditioner = null)
        where T : unmanaged, IEquatable<T>, IComparable<T>
        => IterativeSolvers.BiCGSTAB(a, b, maxIter, tol, preconditioner);
}
