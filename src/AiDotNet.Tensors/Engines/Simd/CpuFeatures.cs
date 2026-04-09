using System.Runtime.CompilerServices;
#if NET5_0_OR_GREATER
using System.Runtime.Intrinsics.X86;
#endif

namespace AiDotNet.Tensors.Engines.Simd;

/// <summary>
/// CPU feature detection for optimal kernel dispatch.
///
/// Key differences that affect kernel selection:
/// - Intel: vpgatherdd is fast (4-8 cycles) → table-driven kernels win
/// - AMD Zen: vpgatherdd is slow (12-20 cycles) → polynomial kernels win
/// - Intel: vdivps throughput 5 cycles → divide is acceptable
/// - AMD Zen: vdivps throughput 10 cycles → avoid divides when possible
/// </summary>
internal static class CpuFeatures
{
#pragma warning disable CS0649 // Fields assigned only under #if NET5_0_OR_GREATER
    private static bool _detected;
    private static bool _isIntel;
    private static bool _isAMD;
    private static bool _hasAVX2;
    private static bool _hasFMA;
    private static bool _hasFastGather;
#pragma warning restore CS0649 // Intel: yes, AMD Zen 1-4: no

    /// <summary>True if CPU is Intel (fast gather, moderate divide).</summary>
    internal static bool IsIntel { get { EnsureDetected(); return _isIntel; } }

    /// <summary>True if CPU is AMD (slow gather, prefer polynomial paths).</summary>
    internal static bool IsAMD { get { EnsureDetected(); return _isAMD; } }

    /// <summary>True if vpgatherdd is fast (&lt;8 cycles). Intel: yes, AMD Zen: no.</summary>
    internal static bool HasFastGather { get { EnsureDetected(); return _hasFastGather; } }

    /// <summary>True if AVX2 is supported.</summary>
    internal static bool HasAVX2 { get { EnsureDetected(); return _hasAVX2; } }

    /// <summary>True if FMA is supported.</summary>
    internal static bool HasFMA { get { EnsureDetected(); return _hasFMA; } }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void EnsureDetected()
    {
        if (_detected) return;
        Detect();
    }

    private static void Detect()
    {
#if NET5_0_OR_GREATER
        _hasAVX2 = Avx2.IsSupported;
        _hasFMA = Fma.IsSupported;

        // Detect vendor from CPU brand string or ISA hints
        try
        {
            // On .NET, we can check the processor name
            var processorName = Environment.GetEnvironmentVariable("PROCESSOR_IDENTIFIER") ?? "";
            if (processorName.Contains("Intel", StringComparison.OrdinalIgnoreCase))
            {
                _isIntel = true;
                _hasFastGather = true;
            }
            else if (processorName.Contains("AMD", StringComparison.OrdinalIgnoreCase)
                  || processorName.Contains("Ryzen", StringComparison.OrdinalIgnoreCase)
                  || processorName.Contains("EPYC", StringComparison.OrdinalIgnoreCase))
            {
                _isAMD = true;
                _hasFastGather = false; // Zen 1-4 all have slow gather
            }
            else
            {
                // Unknown — assume conservative (no fast gather)
                _hasFastGather = false;
            }
        }
        catch
        {
            _hasFastGather = false;
        }
#endif
        _detected = true;
    }
}
