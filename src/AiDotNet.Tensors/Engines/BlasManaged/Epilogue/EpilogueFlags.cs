using System;

namespace AiDotNet.Tensors.Engines.BlasManaged;

/// <summary>
/// Bit-packed presence flags for the fused epilogue chain. Each flag indicates
/// whether a particular stage is active for the current GEMM call. The
/// dispatcher uses these to skip inactive stages in the hot path.
///
/// <para>
/// The non-fused case (Flags = 0) is the most common path — Gemm with no
/// epilogue. The flags fast-path eliminates branches in the inner loop.
/// </para>
/// </summary>
[Flags]
internal enum EpilogueFlags : byte
{
    None = 0,
    HasBias = 1 << 0,
    HasActivation = 1 << 1,
    HasSkip = 1 << 2,
    HasDropout = 1 << 3,
    HasOutputScale = 1 << 4,
}

/// <summary>
/// Helpers for computing <see cref="EpilogueFlags"/> from a <see cref="Epilogue{T}"/>
/// instance. Centralized so all strategies / microkernels see the same
/// "is this stage active?" semantics.
/// </summary>
internal static class EpilogueFlagsCompute
{
    /// <summary>
    /// Derive the active-stage flags from the supplied epilogue. A stage is
    /// considered active when its presence field is non-default:
    /// <list type="bullet">
    ///   <item>HasBias: BiasN.Length &gt; 0</item>
    ///   <item>HasActivation: Activation != FusedActivationType.None</item>
    ///   <item>HasSkip: SkipMxN.Length &gt; 0</item>
    ///   <item>HasDropout: DropoutMask != 0</item>
    ///   <item>HasOutputScale: <see cref="Epilogue{T}.HasOutputScale"/> is true (explicit opt-in,
    ///         lets callers request scale=0) OR <see cref="Epilogue{T}.OutputScale"/> is non-default
    ///         (backward-compat with the old "default=disabled" convention).</item>
    /// </list>
    /// </summary>
    public static EpilogueFlags Compute<T>(in Epilogue<T> epilogue) where T : unmanaged
    {
        EpilogueFlags flags = EpilogueFlags.None;

        if (!epilogue.BiasN.IsEmpty)
            flags |= EpilogueFlags.HasBias;

        if (epilogue.Activation != AiDotNet.Tensors.Engines.FusedActivationType.None)
            flags |= EpilogueFlags.HasActivation;

        if (!epilogue.SkipMxN.IsEmpty)
            flags |= EpilogueFlags.HasSkip;

        if (epilogue.DropoutMask != 0)
            flags |= EpilogueFlags.HasDropout;

        // OutputScale resolution (per PR #366 copilot review):
        //  - Explicit opt-in via HasOutputScale=true → always apply, even scale=0
        //    (used for intentionally zeroing the output for masking/debug).
        //  - Backward-compat: HasOutputScale=false AND OutputScale != default(T)
        //    still activates the stage so existing callers that just set
        //    `OutputScale = 2.0` continue to work.
        //  - HasOutputScale=false AND OutputScale == default(T) → stage inactive.
        if (epilogue.HasOutputScale
            || !System.Collections.Generic.EqualityComparer<T>.Default.Equals(epilogue.OutputScale, default))
        {
            flags |= EpilogueFlags.HasOutputScale;
        }

        return flags;
    }
}
