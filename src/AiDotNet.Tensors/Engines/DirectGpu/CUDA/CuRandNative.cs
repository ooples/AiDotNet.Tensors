// Copyright (c) AiDotNet. All rights reserved.

using System;
using System.Runtime.InteropServices;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA;

/// <summary>
/// Raw P/Invoke bindings into <c>libcurand</c> (curand64_10.dll on
/// Windows, libcurand.so.10 on Linux). Mirrors the signatures we use;
/// not the whole cuRAND API surface — just the entry points the
/// <see cref="CuRand"/> wrapper needs for Philox / Sobol generation.
///
/// <para>Every call is wrapped in a try/catch in the consumer because
/// the native lib may be absent — the host probe inside
/// <see cref="CuRand"/> falls back to the managed CPU Philox path so
/// builds work without CUDA installed.</para>
/// </summary>
internal static class CuRandNative
{
    private const string Lib = "curand64_10";

    /// <summary>cuRAND status codes — only the success and "unavailable"
    /// codes are checked at this level; the wrapper turns the rest into
    /// exceptions.</summary>
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
        PreexistingFailure = 202,
        InitializationFailed = 203,
        ArchMismatch = 204,
        InternalError = 999,
    }

    /// <summary>cuRAND generator types.</summary>
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

    [DllImport(Lib)]
    public static extern Status curandCreateGenerator(out IntPtr generator, GeneratorType type);

    [DllImport(Lib)]
    public static extern Status curandDestroyGenerator(IntPtr generator);

    [DllImport(Lib)]
    public static extern Status curandSetPseudoRandomGeneratorSeed(IntPtr generator, ulong seed);

    [DllImport(Lib)]
    public static extern Status curandSetGeneratorOffset(IntPtr generator, ulong offset);

    [DllImport(Lib)]
    public static extern Status curandGenerateUniform(IntPtr generator, IntPtr outputPtr, ulong num);

    [DllImport(Lib)]
    public static extern Status curandGenerateUniformDouble(IntPtr generator, IntPtr outputPtr, ulong num);

    [DllImport(Lib)]
    public static extern Status curandGenerateNormal(IntPtr generator, IntPtr outputPtr, ulong num, float mean, float stddev);

    [DllImport(Lib)]
    public static extern Status curandGenerateNormalDouble(IntPtr generator, IntPtr outputPtr, ulong num, double mean, double stddev);

    /// <summary>Probes whether the cuRAND lib is loadable + the create
    /// entry point works. Cached at module init so the hot path is a
    /// single static-bool read.</summary>
    public static readonly bool IsAvailable = Probe();

    private static bool Probe()
    {
        try
        {
            var status = curandCreateGenerator(out var gen, GeneratorType.PseudoPhilox4_32_10);
            if (status == Status.Success)
            {
                curandDestroyGenerator(gen);
                return true;
            }
            return false;
        }
        catch (DllNotFoundException) { return false; }
        catch (EntryPointNotFoundException) { return false; }
        catch (BadImageFormatException) { return false; }
        catch { return false; }
    }
}
