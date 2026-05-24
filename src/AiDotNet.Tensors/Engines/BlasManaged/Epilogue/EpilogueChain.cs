using System;

namespace AiDotNet.Tensors.Engines.BlasManaged;

/// <summary>
/// Orchestrates the BlasManaged fused-epilogue chain. After the GEMM
/// strategy writes C, this runs the active epilogue stages in fixed order:
/// bias → activation → skip → dropout → output-scale.
///
/// <para>
/// "Fused" in Phase I means "applied as a post-pass after the strategy
/// completes" — a single chain call per Gemm invocation. Future work can
/// integrate stages INTO each microkernel (saving the C re-read), but the
/// post-pass design keeps the strategy and microkernel code simple.
/// </para>
///
/// <para>
/// The fast path (Flags == None) is a single null-check + early return —
/// no overhead for callers that don't use the epilogue.
/// </para>
/// </summary>
internal static class EpilogueChain
{
    /// <summary>
    /// Apply the active epilogue stages from <paramref name="epilogue"/> to
    /// the C output matrix.
    /// </summary>
    public static void Apply<T>(
        Span<T> c, int ldc, int m, int n,
        in Epilogue<T> epilogue) where T : unmanaged
    {
        var flags = EpilogueFlagsCompute.Compute(in epilogue);
        if (flags == EpilogueFlags.None) return;

        if (flags.HasFlag(EpilogueFlags.HasBias))
            BiasEpilogue.Apply<T>(c, ldc, m, n, epilogue.BiasN);

        if (flags.HasFlag(EpilogueFlags.HasActivation))
            ActivationEpilogue.Apply<T>(c, ldc, m, n, epilogue.Activation);

        if (flags.HasFlag(EpilogueFlags.HasSkip))
            SkipEpilogue.Apply<T>(c, ldc, m, n, epilogue.SkipMxN);

        if (flags.HasFlag(EpilogueFlags.HasDropout))
            DropoutEpilogue.Apply<T>(c, ldc, m, n, epilogue.DropoutMask);

        if (flags.HasFlag(EpilogueFlags.HasOutputScale))
            OutputScaleEpilogue.Apply<T>(c, ldc, m, n, epilogue.OutputScale);
    }
}
