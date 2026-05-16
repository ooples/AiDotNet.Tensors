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

        switch (strategy)
        {
            case PackingMode.ForcePackBoth:
                PackBothStrategy.Run<T>(
                    a, lda, transA,
                    b, ldb, transB,
                    c, ldc,
                    m, n, k,
                    mc: 64, nc: 64, kc: 64,
                    mr: 4, nr: 4);
                break;
            case PackingMode.ForcePackAOnly:
                PackAOnlyStrategy.Run<T>(
                    a, lda, transA,
                    b, ldb,
                    c, ldc,
                    m, n, k,
                    mc: 64, kc: 64,
                    mr: 4, nr: 4);
                break;
            case PackingMode.ForceStreaming:
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
                            PackBothStrategy.Run<T>(a, lda, transA, b, ldb, transB, c, ldc, m, n, k, 64, 64, 64, 4, 4);
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
}
