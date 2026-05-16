using System;

namespace AiDotNet.Tensors.Engines.BlasManaged;

/// <summary>
/// CPU architecture identifier used in <see cref="KernelKey"/> to distinguish
/// hand-written and JIT-emitted microkernels. The dispatcher selects an arch
/// based on runtime <c>CpuFeatures</c> probing; the kernel key caches that
/// choice so future calls with the same shape hit the same microkernel variant.
/// </summary>
public enum CpuArch : byte
{
    /// <summary>Pure-scalar fallback. Used on net471, non-SIMD hosts, and as the ground-truth reference.</summary>
    Scalar = 0,
    /// <summary>AVX2 + FMA path (Haswell, ~2013 and later x64).</summary>
    Avx2 = 1,
    /// <summary>AVX-512F path (Skylake-X, Sapphire Rapids, Zen 4+).</summary>
    Avx512 = 2,
    /// <summary>ARM64 Neon path (Apple Silicon, Graviton, Cobalt).</summary>
    NeonAArch64 = 3,
}

/// <summary>
/// Composite key identifying a microkernel variant for a specific shape, layout,
/// epilogue, precision, and architecture. Used by the JIT cache
/// (<c>JittedKernelCache</c> in Phase J) and the autotune cache (Phase H) to look
/// up shape-specialized kernels without re-deriving the dispatch decision.
/// </summary>
/// <remarks>
/// Value-equality semantics. All fields participate in equality and hashing.
/// Different shapes, layouts, transposition flags, packing modes, epilogue chains,
/// element types, or architectures produce different keys.
/// </remarks>
public readonly struct KernelKey : IEquatable<KernelKey>
{
    /// <summary>Rows of C (and rows of op(A)).</summary>
    public int M { get; init; }
    /// <summary>Cols of C (and cols of op(B)).</summary>
    public int N { get; init; }
    /// <summary>Inner reduction dimension (cols of op(A) = rows of op(B)).</summary>
    public int K { get; init; }
    /// <summary>Leading dimension of A in the caller's storage layout.</summary>
    public int Lda { get; init; }
    /// <summary>Leading dimension of B.</summary>
    public int Ldb { get; init; }
    /// <summary>Leading dimension of C.</summary>
    public int Ldc { get; init; }
    /// <summary>True if op(A) = A^T.</summary>
    public bool TransA { get; init; }
    /// <summary>True if op(B) = B^T.</summary>
    public bool TransB { get; init; }
    /// <summary>Packing strategy chosen for this shape (post-autotune resolution).</summary>
    public PackingMode Packing { get; init; }
    /// <summary>Bit-packed presence flags for the fused epilogue chain (bias / activation / skip / dropout / scale).</summary>
    public byte EpilogueFlags { get; init; }
    /// <summary>Element type — typically <see cref="float"/> or <see cref="double"/>.</summary>
    public Type ElemType { get; init; }
    /// <summary>CPU architecture path resolved by the dispatcher.</summary>
    public CpuArch Arch { get; init; }

    /// <inheritdoc/>
    public bool Equals(KernelKey other) =>
        M == other.M && N == other.N && K == other.K
        && Lda == other.Lda && Ldb == other.Ldb && Ldc == other.Ldc
        && TransA == other.TransA && TransB == other.TransB
        && Packing == other.Packing && EpilogueFlags == other.EpilogueFlags
        && ElemType == other.ElemType && Arch == other.Arch;

    /// <inheritdoc/>
    public override bool Equals(object? obj) => obj is KernelKey k && Equals(k);

    /// <inheritdoc/>
    public override int GetHashCode()
    {
        // Manual hash combine — consistent with the rest of the codebase and
        // compatible with net471 which does not have System.HashCode.Combine.
        unchecked
        {
            int hash = 17;
            hash = hash * 23 + M;
            hash = hash * 23 + N;
            hash = hash * 23 + K;
            hash = hash * 23 + Lda;
            hash = hash * 23 + Ldb;
            hash = hash * 23 + Ldc;
            hash = hash * 23 + (TransA ? 1 : 0);
            hash = hash * 23 + (TransB ? 1 : 0);
            hash = hash * 23 + (int)Packing;
            hash = hash * 23 + EpilogueFlags;
            hash = hash * 23 + (ElemType?.GetHashCode() ?? 0);
            hash = hash * 23 + (int)Arch;
            return hash;
        }
    }

    /// <summary>Equality operator.</summary>
    public static bool operator ==(KernelKey left, KernelKey right) => left.Equals(right);
    /// <summary>Inequality operator.</summary>
    public static bool operator !=(KernelKey left, KernelKey right) => !left.Equals(right);
}
