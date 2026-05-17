using System.Runtime.CompilerServices;

namespace AiDotNet.Tensors.Engines.BlasManaged;

/// <summary>
/// Detects whether dynamic code generation (System.Reflection.Emit) is
/// supported on the current runtime. Used by <see cref="JittedKernelCache"/>
/// to skip JIT emission under NativeAOT publishing (where DynamicMethod is
/// not supported).
///
/// <para>
/// On NativeAOT, <see cref="RuntimeFeature.IsDynamicCodeSupported"/> returns
/// <c>false</c>, and the JIT cache short-circuits to "no cached kernel" on
/// every lookup. Callers fall back to the hand-written microkernels — same
/// correctness, ~5-15% slower on hot shapes.
/// </para>
/// </summary>
internal static class NativeAotDetector
{
    /// <summary>
    /// True when the runtime supports dynamic code generation
    /// (<see cref="System.Reflection.Emit.DynamicMethod"/> is functional).
    /// False under NativeAOT and similar AOT-only environments.
    /// </summary>
    public static bool IsDynamicCodeSupported
    {
        get
        {
#if NET5_0_OR_GREATER
            return RuntimeFeature.IsDynamicCodeSupported;
#else
            // Pre-net5 (net471): no NativeAOT, DynamicMethod always supported.
            return true;
#endif
        }
    }
}
