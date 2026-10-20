using System.Runtime.CompilerServices;
using static AiDotNet.Tensors.Compatibility.MethodImplHelper;
#if NET5_0_OR_GREATER
using System.Runtime.Intrinsics.X86;
#endif

namespace AiDotNet.Tensors.Engines.Simd;

/// <summary>
/// CPU feature detection for optimal kernel dispatch. Kernels are chosen by instruction set alone; no kernel
/// depends on the CPU vendor.
/// </summary>
internal static class CpuFeatures
{
#pragma warning disable CS0649 // Fields assigned only under #if NET5_0_OR_GREATER
    private static bool _detected;
    private static bool _hasAVX2;
    private static bool _hasFMA;
    private static bool _hasAVX512F;
    private static bool _hasAVX512BW;
    private static bool _hasAVX512DQ;
    private static bool _hasAVX512VNNI;
    private static bool _hasAMXTile;
#pragma warning restore CS0649

    /// <summary>True if AVX2 is supported.</summary>
    internal static bool HasAVX2 { get { EnsureDetected(); return _hasAVX2; } }

    /// <summary>True if FMA is supported.</summary>
    internal static bool HasFMA { get { EnsureDetected(); return _hasFMA; } }

    /// <summary>True if AVX-512F (foundation) is supported — enables 16-wide
    /// Vector512&lt;float&gt; FMA. Requires .NET 8+ at compile time.</summary>
    internal static bool HasAVX512F { get { EnsureDetected(); return _hasAVX512F; } }

    /// <summary>True if AVX-512BW (byte / word ops) is supported.</summary>
    internal static bool HasAVX512BW { get { EnsureDetected(); return _hasAVX512BW; } }

    /// <summary>True if AVX-512DQ (double / qword ops) is supported.</summary>
    internal static bool HasAVX512DQ { get { EnsureDetected(); return _hasAVX512DQ; } }

    /// <summary>True if AVX-512 VNNI is supported — int8 dot-product
    /// instructions for quantised matmul. Cascade Lake and later.</summary>
    internal static bool HasAVX512VNNI { get { EnsureDetected(); return _hasAVX512VNNI; } }

    /// <summary>True if AMX tile instructions are supported — Sapphire Rapids
    /// and later Xeons, enables the tile-based int8 matmul kernel.</summary>
    internal static bool HasAMX { get { EnsureDetected(); return _hasAMXTile; } }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void EnsureDetected()
    {
        if (_detected) return;
        Detect();
    }

    [MethodImpl(Hot)]
    private static void Detect()
    {
#if NET5_0_OR_GREATER
        _hasAVX2 = Avx2.IsSupported;
        _hasFMA = Fma.IsSupported;
#if NET8_0_OR_GREATER
        // AVX-512 and AMX intrinsics live under System.Runtime.Intrinsics.X86
        // on .NET 8+. Earlier TFMs keep the flags false and fall back to AVX2.
        _hasAVX512F    = Avx512F.IsSupported;
        _hasAVX512BW   = Avx512BW.IsSupported;
        _hasAVX512DQ   = Avx512DQ.IsSupported;
        // AVX-512 VNNI and AVX-512 VBMI use separate CPUID bits
        // (CPUID.(EAX=7,ECX=0):ECX[bit 11] for VNNI vs ECX[bit 1] for VBMI),
        // so the earlier Avx512Vbmi.IsSupported proxy was unsafe: Ice Lake
        // ships VNNI without full VBMI support, so we would have reported
        // "no VNNI" on machines that actually have it.
        //
        // .NET 8/9's System.Runtime.Intrinsics.X86 does not yet expose an
        // Avx512Vnni type with an IsSupported probe (the 256-bit AvxVnni
        // is exposed, but 512-bit VNNI is not). Until .NET exposes the type
        // directly (or we ship a P/Invoke into CPUID), report false and let
        // callers route to AVX-512F / AVX2 paths. The dispatch tables below
        // all have non-VNNI fallbacks.
        _hasAVX512VNNI = false;
#endif
#endif
        _detected = true;
    }
}
