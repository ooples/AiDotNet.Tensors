// Copyright (c) AiDotNet. All rights reserved.

using System;
using System.Runtime.InteropServices;

namespace AiDotNet.Tensors.Engines.DirectGpu.HIP;

/// <summary>
/// Raw P/Invoke bindings into <c>librocrand</c> (rocrand.dll on Windows,
/// librocrand.so.1 on Linux). Mirrors the cuRAND surface so the
/// <see cref="RocRand"/> wrapper is a drop-in alternative on AMD GPUs.
/// rocRAND keeps the same generator-type constants as cuRAND for the
/// algorithms shared between them (Philox, MRG, Sobol).
///
/// <para>Like the CUDA bindings, every entry has an <see cref="IsAvailable"/>
/// probe so builds work on hosts without ROCm installed; missing-lib
/// failures are caught and the wrapper falls back to the managed Philox
/// path (which is bit-equivalent to rocRAND's Philox).</para>
/// </summary>
internal static class RocRandNative
{
    // Windows ships the lib as rocrand.dll; Linux ships librocrand.so.1.
    // .NET's loader probes both with a DllImportResolver below.
    private const string Lib = "rocrand";

#if NET5_0_OR_GREATER
    static RocRandNative()
    {
        AiDotNet.Tensors.Engines.NativeLibraryResolverRegistry.Register(ResolveLib);
    }

    private static IntPtr ResolveLib(string name, System.Reflection.Assembly asm, DllImportSearchPath? path)
    {
        if (name != Lib) return IntPtr.Zero;
        if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux))
        {
            if (NativeLibrary.TryLoad("librocrand.so.1", asm, path, out var h)) return h;
            if (NativeLibrary.TryLoad("librocrand.so", asm, path, out h)) return h;
        }
        return IntPtr.Zero;
    }
#endif

    public enum Status
    {
        Success = 0,
        VersionMismatch = 100,
        NotInitialized = 101,
        AllocationFailed = 102,
        TypeError = 103,
        OutOfRange = 104,
        LengthNotMultiple = 105,
        DoublePrecisionRequired = 106,
        LaunchFailure = 201,
        InternalError = 999,
    }

    /// <summary>rocRAND generator types — value-compatible with the
    /// matching cuRAND constants for the shared algorithms.</summary>
    public enum GeneratorType
    {
        PseudoXorwow = 101,
        PseudoMrg32K3A = 121,
        PseudoMtgp32 = 141,
        PseudoMt19937 = 142,
        PseudoPhilox4_32_10 = 161,
        QuasiSobol32 = 201,
        QuasiScrambledSobol32 = 202,
        QuasiSobol64 = 203,
        QuasiScrambledSobol64 = 204,
    }

    [DllImport(Lib)] public static extern Status rocrand_create_generator(out IntPtr generator, GeneratorType type);
    [DllImport(Lib)] public static extern Status rocrand_destroy_generator(IntPtr generator);
    [DllImport(Lib)] public static extern Status rocrand_set_seed(IntPtr generator, ulong seed);
    [DllImport(Lib)] public static extern Status rocrand_set_offset(IntPtr generator, ulong offset);
    [DllImport(Lib)] public static extern Status rocrand_generate_uniform(IntPtr generator, IntPtr outputPtr, ulong num);
    [DllImport(Lib)] public static extern Status rocrand_generate_uniform_double(IntPtr generator, IntPtr outputPtr, ulong num);
    [DllImport(Lib)] public static extern Status rocrand_generate_normal(IntPtr generator, IntPtr outputPtr, ulong num, float mean, float stddev);
    [DllImport(Lib)] public static extern Status rocrand_generate_normal_double(IntPtr generator, IntPtr outputPtr, ulong num, double mean, double stddev);

    public static readonly bool IsAvailable = Probe();

    private static bool Probe()
    {
        try
        {
            var status = rocrand_create_generator(out var gen, GeneratorType.PseudoPhilox4_32_10);
            if (status == Status.Success) { rocrand_destroy_generator(gen); return true; }
            return false;
        }
        catch (DllNotFoundException) { return false; }
        catch (EntryPointNotFoundException) { return false; }
        catch (BadImageFormatException) { return false; }
        catch { return false; }
    }
}
