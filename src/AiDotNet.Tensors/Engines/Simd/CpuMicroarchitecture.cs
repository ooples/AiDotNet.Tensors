// Copyright (c) CrowdTrainer LLC. All rights reserved.
#if NET5_0_OR_GREATER
using System;
using System.Runtime.Intrinsics.X86;

namespace AiDotNet.Tensors.Engines.Simd;

/// <summary>
/// CPU traits that change which kernel is fastest but that the instruction-set flags do not show.
/// </summary>
internal static class CpuMicroarchitecture
{
    /// <summary>
    /// Whether the host runs 512-bit vector instructions at half rate: AMD Zen 4 (family 19h with AVX-512) executes each
    /// as two 256-bit halves. Its 512-bit direct conv kernels then lose to Winograd where full-rate hosts (Intel Xeon,
    /// AMD Zen 5) win: ResNet-18's 256 -> 256 input gradient on 8x8 planes ran 2.89 ms direct against 2.08 Winograd on
    /// an EPYC 9V74, and 1.76 against 2.12 on a Xeon 8573C.
    /// </summary>
    internal static readonly bool HalfRateVector512 = DetectHalfRateVector512();

    /// <summary>Whether a full-rate 512-bit unit runs the direct kernels: AVX-512 present and not <see cref="HalfRateVector512"/>.</summary>
    internal static bool FullRateVector512 =>
#if NET8_0_OR_GREATER
        System.Runtime.Intrinsics.Vector512.IsHardwareAccelerated && !HalfRateVector512;
#else
        false;
#endif

    /// <summary>The decision of <see cref="HalfRateVector512"/> from CPUID's vendor, display family and AVX-512 support.</summary>
    internal static bool IsHalfRateVector512(string vendor, int family, bool avx512)
        => avx512 && vendor == "AuthenticAMD" && family == 0x19;

    private static bool DetectHalfRateVector512()
    {
#if NET8_0_OR_GREATER
        if (!X86Base.IsSupported || !Avx512F.IsSupported) return false;
        try
        {
            var (_, ebx, ecx, edx) = X86Base.CpuId(0, 0);
            Span<byte> vendor = stackalloc byte[12];
            BitConverter.TryWriteBytes(vendor, ebx);
            BitConverter.TryWriteBytes(vendor.Slice(4), edx);
            BitConverter.TryWriteBytes(vendor.Slice(8), ecx);
            int eax = X86Base.CpuId(1, 0).Eax;
            int baseFamily = (eax >> 8) & 0xF;
            int family = baseFamily == 0xF ? baseFamily + ((eax >> 20) & 0xFF) : baseFamily;
            return IsHalfRateVector512(System.Text.Encoding.ASCII.GetString(vendor), family, avx512: true);
        }
        catch (PlatformNotSupportedException)
        {
            return false;
        }
#else
        return false;
#endif
    }
}
#endif
