using System;

namespace AiDotNet.Tensors.Engines.BlasManaged;

/// <summary>
/// BLIS-style managed GEMM kernel. Replaces Avx512Sgemm + SimdGemm as the
/// codebase's primary GEMM path. See docs/superpowers/specs/2026-05-16-blas-managed-design.md.
/// </summary>
public static class BlasManaged
{
    /// <summary>
    /// Computes C = op(A) · op(B), where op(X) is X or X^T.
    ///
    /// <para>
    /// The strategy (packing mode) is selected by <see cref="Dispatcher.SelectStrategy{T}"/>
    /// based on the user's <see cref="BlasOptions{T}.PackingMode"/> override or, when it is
    /// <see cref="PackingMode.Auto"/>, a static heuristic on K and M·N work size:
    /// </para>
    ///
    /// <list type="bullet">
    ///   <item>K &lt; 32 or M·N &lt; 1024 → <see cref="StreamingStrategy"/> (no pack overhead).</item>
    ///   <item>K &lt; 128 → <see cref="PackAOnlyStrategy"/> (pack A; B reuse too low to justify packing).</item>
    ///   <item>Otherwise → <see cref="PackBothStrategy"/> (pack A and B for maximum cache locality).</item>
    /// </list>
    ///
    /// <para>
    /// <c>C</c> is always zeroed on entry so the result is C := op(A)·op(B),
    /// not C += op(A)·op(B). A future beta=1 option may skip this zero.
    /// </para>
    ///
    /// <para>
    /// <see cref="PackingMode.DisableAutotune"/> falls through to the default Auto heuristic
    /// in this phase; the autotune cache is added in Phase H.
    /// </para>
    ///
    /// <para>
    /// An early return is taken when any of M, N, or K is zero; caller-supplied C is preserved.
    /// </para>
    ///
    /// <para>
    /// Optional epilogue (bias, activation, skip, dropout, output scale) and other advanced
    /// settings are specified via <paramref name="options"/>.
    /// Defaults when omitted: no epilogue, single-threaded, no autotune key.
    /// </para>
    /// </summary>
    public static void Gemm<T>(
        ReadOnlySpan<T> a, int lda, bool transA,
        ReadOnlySpan<T> b, int ldb, bool transB,
        Span<T> c, int ldc,
        int m, int n, int k,
        in BlasOptions<T> options = default) where T : unmanaged
    {
        if (m <= 0 || n <= 0 || k <= 0) return;

        // Strategies do read-modify-write on C; zero it here so the first call
        // produces C = A · B (not C += A · B). Future versions may expose a
        // beta=1 option that skips this zero, but Phase B's contract is C := A · B.
        c.Clear();

        PackingMode strategy = Dispatcher.SelectStrategy(m, n, k, options);

        // PackAOnly does not support transB=true in Phase B — fall back to
        // PackBoth (which absorbs the transpose into its pack) or Streaming
        // (which handles transB via in-place index branching).
        if (strategy == PackingMode.ForcePackAOnly && transB)
            strategy = PackingMode.ForcePackBoth;

        // Pick SIMD-aware (mr, nr) for PackBoth using the AVX-512 → AVX2 → Neon → scalar
        // hierarchy. Fall back to (4, 4) scalar when the shape is not an exact
        // multiple of the chosen tile — tail handling added in Phase G.
        // PackAOnly always uses scalar (4, 4) because its strided-B path has no
        // AVX2/AVX-512 RunStridedB variant yet (deferred to Phase Cx).
        var (mr, nr) = PickMicrokernelTile<T>();
        if (m % mr != 0 || n % nr != 0)
        {
            mr = 4; nr = 4;
        }

        switch (strategy)
        {
            case PackingMode.ForcePackBoth:
                PackBothStrategy.Run<T>(
                    a, lda, transA,
                    b, ldb, transB,
                    c, ldc,
                    m, n, k,
                    mc: 64, nc: 64, kc: 64,
                    mr: mr, nr: nr);
                break;
            case PackingMode.ForcePackAOnly:
                // PackAOnly currently has no AVX2 strided-B kernel; always scalar Mr=Nr=4.
                // TODO(Phase Cx): add Avx2 RunStridedB variants and wire them here.
                PackAOnlyStrategy.Run<T>(
                    a, lda, transA,
                    b, ldb,
                    c, ldc,
                    m, n, k,
                    mc: 64, kc: 64,
                    mr: 4, nr: 4);
                break;
            case PackingMode.ForceStreaming:
                // Streaming dispatches AVX2 internally — no Mr/Nr tiling parameter.
                StreamingStrategy.Run<T>(
                    a, lda, transA,
                    b, ldb, transB,
                    c, ldc,
                    m, n, k);
                break;
            case PackingMode.DisableAutotune:
                // DisableAutotune is a power-user override that bypasses the
                // cache (relevant in Phase H). In Phase B with no autotune
                // cache, it falls through to the default heuristic for the
                // current Auto-mode choice.
                {
                    var defaultedOptions = new BlasOptions<T> { PackingMode = PackingMode.Auto };
                    PackingMode fallback = Dispatcher.SelectStrategy(m, n, k, defaultedOptions);
                    if (fallback == PackingMode.ForcePackAOnly && transB)
                        fallback = PackingMode.ForcePackBoth;
                    switch (fallback)
                    {
                        case PackingMode.ForcePackBoth:
                            PackBothStrategy.Run<T>(a, lda, transA, b, ldb, transB, c, ldc, m, n, k, 64, 64, 64, mr, nr);
                            break;
                        case PackingMode.ForcePackAOnly:
                            PackAOnlyStrategy.Run<T>(a, lda, transA, b, ldb, c, ldc, m, n, k, 64, 64, 4, 4);
                            break;
                        case PackingMode.ForceStreaming:
                            StreamingStrategy.Run<T>(a, lda, transA, b, ldb, transB, c, ldc, m, n, k);
                            break;
                    }
                }
                break;
            case PackingMode.Auto:
                throw new InvalidOperationException("Dispatcher.SelectStrategy should never return Auto.");
            default:
                throw new NotSupportedException($"Unknown PackingMode: {strategy}");
        }
    }

    /// <summary>
    /// Pick the microkernel (mr, nr) tile widths based on element type and
    /// runtime SIMD availability. The selection follows a four-tier hierarchy
    /// — AVX-512 → AVX2 → Neon → scalar — so the widest available vector ISA
    /// is used. On any host only one of {AVX-512, AVX2, Neon} can be active:
    /// AVX paths are x64-only; Neon is ARM64-only.
    /// <list type="bullet">
    ///   <item>FP64: (8, 16) AVX-512 → (4, 8) AVX2 → (4, 4) Neon or scalar.</item>
    ///   <item>FP32: (16, 16) AVX-512 → (8, 8) AVX2 → (8, 4) Neon → (4, 4) scalar.</item>
    /// </list>
    /// For FP64 the Neon tile (Mr=4, Nr=4) is identical to the scalar tile, so
    /// no new branch is needed — <see cref="PackBothStrategy"/> picks between
    /// <see cref="NeonFp64_4x4"/> and <see cref="ScalarFp64_4x4"/> at dispatch time.
    /// For FP32 the Neon tile (Mr=8, Nr=4) differs from scalar (4, 4) so it
    /// requires its own branch here.
    /// This selection drives the layout of packed-A and packed-B, so it must
    /// match the microkernel the strategy ultimately dispatches to.
    /// </summary>
    private static (int Mr, int Nr) PickMicrokernelTile<T>() where T : unmanaged
    {
        if (typeof(T) == typeof(double))
        {
            if (Avx512Fp64_8x16.IsSupported) return (8, 16);
            if (Avx2Fp64_4x8.IsSupported) return (4, 8);
            // Neon FP64 uses (4, 4) — same as scalar; DispatchMicrokernel picks the right kernel.
            return (4, 4);
        }
        if (typeof(T) == typeof(float))
        {
            if (Avx512Fp32_16x16.IsSupported) return (16, 16);
            if (Avx2Fp32_8x8.IsSupported) return (8, 8);
            if (NeonFp32_8x4.IsSupported) return (8, 4);
            return (4, 4);
        }
        return (4, 4);
    }
}
