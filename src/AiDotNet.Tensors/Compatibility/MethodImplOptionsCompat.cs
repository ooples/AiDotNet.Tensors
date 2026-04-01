using System.Runtime.CompilerServices;

namespace AiDotNet.Tensors.Compatibility;

/// <summary>
/// Provides <see cref="MethodImplOptions.AggressiveOptimization"/> as a constant
/// that compiles on both .NET 5+ and .NET Framework 4.7.1.
/// </summary>
/// <remarks>
/// <see cref="MethodImplOptions.AggressiveOptimization"/> (value 512) was introduced
/// in .NET Core 3.0. On .NET Framework, the JIT silently ignores unknown flags,
/// so passing the raw value is harmless.
///
/// Usage: add <c>using static AiDotNet.Tensors.Compatibility.MethodImplHelper;</c>
/// then use <c>[MethodImpl(Hot)]</c> or <c>[MethodImpl(HotInline)]</c>.
/// </remarks>
internal static class MethodImplHelper
{
    /// <summary>
    /// AggressiveOptimization — forces Tier1 JIT from the first call.
    /// Prevents tiered compilation from producing different FP results.
    /// </summary>
    internal const MethodImplOptions Hot =
#if NETFRAMEWORK
        (MethodImplOptions)512;
#else
        MethodImplOptions.AggressiveOptimization;
#endif

    /// <summary>
    /// AggressiveInlining | AggressiveOptimization — for small hot-path methods.
    /// </summary>
    internal const MethodImplOptions HotInline =
        MethodImplOptions.AggressiveInlining | Hot;
}
