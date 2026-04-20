using System;
using System.Runtime.CompilerServices;

namespace AiDotNet.Tensors.Helpers;

/// <summary>
/// External LAPACK provider — mirrors <see cref="BlasProvider"/>'s tiered-dispatch
/// pattern for decompositions and solvers. The current build uses managed C#
/// implementations for every op; the native-binding tier (Intel MKL / OpenBLAS /
/// Accelerate / cuSOLVER) is wired through <c>Try*</c> gates that always return
/// <c>false</c> today. Each gate is the extension point for a follow-up that
/// re-enables MKL LAPACK or binds cuSOLVER without touching any caller.
///
/// <para>
/// Tier order at every entry point:
/// <list type="number">
///   <item><b>Native LAPACK</b> (Intel MKL LAPACKE, OpenBLAS LAPACK, Accelerate vDSP) —
///   disabled in the supply-chain-independence build. Reserved for the follow-up PR
///   that re-adds the P/Invoke loader behind a runtime flag.</item>
///   <item><b>Managed reference</b> — pure C# Doolittle/Householder/Jacobi kernels.
///   Deterministic across thread counts, target correctness first and performance
///   second. This is the default tier for every op today.</item>
///   <item><b>SIMD-accelerated</b> — AVX/NEON kernels for specific hot paths
///   (triangular solve, banded matvec). Reserved for follow-ups.</item>
/// </list>
/// </para>
/// </summary>
internal static class LapackProvider
{
    // ── Gate flags ──────────────────────────────────────────────────────────
    // All false today; consumer code structured so flipping them on does not
    // require any call-site changes beyond implementing the native tier.

    internal static bool HasLapack => false;
    internal static bool HasCuSolver => false;
    internal static bool HasRocSolver => false;
    internal static bool HasAccelerate => false;

    // ── Tiered dispatch: LU factor with partial pivoting ────────────────────

    /// <summary>
    /// Attempts LAPACK <c>?getrf</c>-style LU factorization with partial pivoting.
    /// Returns <c>false</c> (no fallback) when no native LAPACK is available.
    /// Callers fall through to the managed implementation in the decomposition class.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static bool TryGetrf(
        int m, int n, Span<float> a, int lda, Span<int> ipiv, out int info)
    {
        info = 0;
        return false;
    }

    /// <inheritdoc cref="TryGetrf(int, int, Span{float}, int, Span{int}, out int)"/>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static bool TryGetrf(
        int m, int n, Span<double> a, int lda, Span<int> ipiv, out int info)
    {
        info = 0;
        return false;
    }

    // ── Tiered dispatch: Cholesky ───────────────────────────────────────────

    /// <summary>Attempts LAPACK <c>?potrf</c>. Always false in the stub build.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static bool TryPotrf(bool upper, int n, Span<float> a, int lda, out int info)
    {
        info = 0;
        return false;
    }

    /// <inheritdoc cref="TryPotrf(bool, int, Span{float}, int, out int)"/>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static bool TryPotrf(bool upper, int n, Span<double> a, int lda, out int info)
    {
        info = 0;
        return false;
    }

    // ── Tiered dispatch: QR ─────────────────────────────────────────────────

    /// <summary>Attempts LAPACK <c>?geqrf</c>. Always false in the stub build.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static bool TryGeqrf(
        int m, int n, Span<float> a, int lda, Span<float> tau, out int info)
    {
        info = 0;
        return false;
    }

    // ── Tiered dispatch: symmetric eigendecomposition ───────────────────────

    /// <summary>Attempts LAPACK <c>?syevd</c>. Always false in the stub build.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static bool TrySyevd(
        bool computeVectors, bool upper, int n,
        Span<float> a, int lda, Span<float> w, out int info)
    {
        info = 0;
        return false;
    }
}
