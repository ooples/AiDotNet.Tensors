using System.Runtime.InteropServices;
#if NETCOREAPP3_0_OR_GREATER || NET5_0_OR_GREATER
using System.Runtime.Intrinsics.X86;
using System.Runtime.Intrinsics.Arm;
#endif

namespace AiDotNet.Tensors.Helpers.Autotune;

/// <summary>
/// Computes a compact, stable fingerprint of the current host's CPU — used by
/// <see cref="AutotuneCache"/> to prevent cross-machine cache poisoning.
///
/// The fingerprint distinguishes:
/// <list type="bullet">
///   <item>Architecture (x64, arm64, x86, arm32)</item>
///   <item>Vendor (intel, amd, arm, apple, other)</item>
///   <item>Highest SIMD level usable at runtime (avx512, avx2, sse4.2, sse2, neon, none)</item>
///   <item>Logical processor count (affects parallel kernels)</item>
/// </list>
///
/// A migrated machine (different CPU, same user profile) produces a different
/// fingerprint and keeps older entries isolated — stale tunings are never
/// served on wrong hardware.
/// </summary>
public static class HardwareFingerprint
{
    // Lazy-computed, never changes during process lifetime.
    private static string? _cachedFingerprint;
    private static readonly object _lock = new();

    /// <summary>
    /// Returns the current host's fingerprint string, e.g.
    /// <c>"x64-intel-avx2-cpu16"</c> or <c>"arm64-apple-neon-cpu10"</c>.
    /// Suitable for use as a filesystem-safe cache-directory name.
    /// </summary>
    public static string Current
    {
        get
        {
            if (_cachedFingerprint is not null) return _cachedFingerprint;
            lock (_lock)
            {
                _cachedFingerprint ??= Compute();
                return _cachedFingerprint;
            }
        }
    }

    /// <summary>
    /// For tests only: forces recomputation on the next <see cref="Current"/>
    /// access. Does not clear the cache's on-disk contents — the caller is
    /// responsible for deleting the previous fingerprint's directory if that's
    /// the intent.
    /// </summary>
    internal static void InvalidateForTests()
    {
        lock (_lock) _cachedFingerprint = null;
    }

    private static string Compute()
    {
        string arch = RuntimeInformation.ProcessArchitecture switch
        {
            Architecture.X64   => "x64",
            Architecture.X86   => "x86",
            Architecture.Arm64 => "arm64",
            Architecture.Arm   => "arm32",
            _                  => RuntimeInformation.ProcessArchitecture.ToString().ToLowerInvariant(),
        };

        string vendor = DetectVendor();
        string simd   = DetectSimdLevel();
        int cpus      = Environment.ProcessorCount;

        return $"{arch}-{vendor}-{simd}-cpu{cpus}";
    }

    private static string DetectVendor()
    {
#if NETCOREAPP3_0_OR_GREATER || NET5_0_OR_GREATER
        // CPUID leaf 0 returns the vendor string in EBX|EDX|ECX for x86/x64.
        if (X86Base.IsSupported &&
            (RuntimeInformation.ProcessArchitecture == Architecture.X64 ||
             RuntimeInformation.ProcessArchitecture == Architecture.X86))
        {
            try
            {
                var (_, ebx, ecx, edx) = X86Base.CpuId(0, 0);
                // Decode "GenuineIntel", "AuthenticAMD", etc.
                Span<byte> vendorBytes = stackalloc byte[12];
                WriteInt(vendorBytes, 0, ebx);
                WriteInt(vendorBytes, 4, edx);
                WriteInt(vendorBytes, 8, ecx);
                string raw = System.Text.Encoding.ASCII.GetString(vendorBytes);
                if (raw.Contains("Intel", StringComparison.OrdinalIgnoreCase))        return "intel";
                if (raw.Contains("AMD", StringComparison.OrdinalIgnoreCase))          return "amd";
                if (raw.Contains("Hygon", StringComparison.OrdinalIgnoreCase))        return "hygon";
                return "other-x86";
            }
            catch
            {
                // CPUID may throw under sandbox / simulation; fall through.
            }
        }
#endif
        // ARM has no universal vendor query equivalent; distinguish by OS where useful.
        if (RuntimeInformation.ProcessArchitecture == Architecture.Arm64 ||
            RuntimeInformation.ProcessArchitecture == Architecture.Arm)
        {
            if (RuntimeInformation.IsOSPlatform(OSPlatform.OSX)) return "apple";
            return "arm";
        }

        // Fallback for runtimes without System.Runtime.Intrinsics (e.g. net471):
        // try Windows' PROCESSOR_IDENTIFIER env var, e.g.
        //   "Intel64 Family 6 Model 158 Stepping 9, GenuineIntel"
        //   "AMD64 Family 25 Model 33 Stepping 0, AuthenticAMD"
        // Without this fallback every x64 net471 host collapses to the same
        // "unknown" bucket and Intel/AMD machines share cache entries — exactly
        // the cross-machine poisoning the fingerprint exists to prevent.
        return VendorFromProcessorIdentifier();
    }

    private static string VendorFromProcessorIdentifier()
    {
        try
        {
            string? id = Environment.GetEnvironmentVariable("PROCESSOR_IDENTIFIER");
            if (!string.IsNullOrWhiteSpace(id))
            {
                if (id!.IndexOf("Intel",  StringComparison.OrdinalIgnoreCase) >= 0) return "intel";
                if (id.IndexOf("AMD",    StringComparison.OrdinalIgnoreCase) >= 0) return "amd";
                if (id.IndexOf("Hygon",  StringComparison.OrdinalIgnoreCase) >= 0) return "hygon";
                if (id.IndexOf("ARM",    StringComparison.OrdinalIgnoreCase) >= 0) return "arm";
            }
        }
        catch { /* env access can throw under sandbox; fall through. */ }
        return "unknown";
    }

    private static string DetectSimdLevel()
    {
#if NETCOREAPP3_0_OR_GREATER || NET5_0_OR_GREATER
        // Query highest usable x86 SIMD level.
        if (Avx512F.IsSupported) return "avx512";
        if (Avx2.IsSupported)    return "avx2";
        if (Avx.IsSupported)     return "avx";
        if (Sse42.IsSupported)   return "sse4.2";
        if (Sse2.IsSupported)    return "sse2";

        // ARM NEON / AdvSimd.
        if (AdvSimd.IsSupported) return "neon";
#endif
        // Fallback for net471: we cannot probe AVX/AVX2 support without the
        // intrinsics API, but we can give a coarse architectural floor based
        // on the process bitness. x64 implies SSE2 by definition (it's part of
        // the AMD64 ABI); narrowing further requires CPUID, which is out of
        // reach on this target. "sse2-fallback" is distinct from the full
        // "sse2" tag so operators can tell tuned-on-net471 from
        // tuned-on-modern-runtime in their cache directories.
        if (RuntimeInformation.ProcessArchitecture == Architecture.X64) return "sse2-fallback";
        if (RuntimeInformation.ProcessArchitecture == Architecture.Arm64) return "neon-fallback";
        return "none";
    }

    private static void WriteInt(Span<byte> dest, int offset, int value)
    {
        dest[offset + 0] = (byte)(value       & 0xFF);
        dest[offset + 1] = (byte)((value >> 8) & 0xFF);
        dest[offset + 2] = (byte)((value >> 16) & 0xFF);
        dest[offset + 3] = (byte)((value >> 24) & 0xFF);
    }
}
