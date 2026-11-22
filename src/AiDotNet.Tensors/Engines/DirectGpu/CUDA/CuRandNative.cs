// Copyright (c) CrowdTrainer LLC. All rights reserved.

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

    static CuRandNative()
    {
#if NETCOREAPP
        // CUDA 13 installs its DLLs in bin\x64, which its installer does not put on PATH (only bin), so the default
        // search found no curand64_10 on such a machine and every cuRAND path fell back to the CPU generator (the
        // CUDA lane's skip gate failed on it). Probe the CUDA install as cuBLAS does.
        NativeLibraryResolverRegistry.Register(ResolveCuRandLibrary);
#endif
        IsAvailable = Probe();
    }

#if NETCOREAPP

    private static readonly string[] WindowsCandidates = ["curand64_10"];
    private static readonly string[] LinuxCandidates = ["libcurand.so.10", "libcurand.so"];

    private static IntPtr ResolveCuRandLibrary(string libraryName, System.Reflection.Assembly assembly, DllImportSearchPath? searchPath)
    {
        if (!libraryName.StartsWith("curand64_", StringComparison.Ordinal) &&
            !libraryName.StartsWith("libcurand", StringComparison.Ordinal))
            return IntPtr.Zero;
        bool windows = RuntimeInformation.IsOSPlatform(OSPlatform.Windows);
        var candidates = windows ? WindowsCandidates : LinuxCandidates;
        foreach (var candidate in candidates)
            if (NativeLibrary.TryLoad(candidate, assembly, searchPath, out var handle))
                return handle;
        var dirs = CuBlasNative.GetCudaBinDirectories();
        foreach (var candidate in candidates)
            foreach (var dir in dirs)
                if (NativeLibrary.TryLoad(System.IO.Path.Combine(dir, windows ? candidate + ".dll" : candidate), out var handle))
                    return handle;
        return IntPtr.Zero;
    }
#endif

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

    /// <summary>A generator that runs on the host and writes host memory. Same output as the device generator of
    /// the same type; needs no CUDA context.</summary>
    [DllImport(Lib)]
    public static extern Status curandCreateGeneratorHost(out IntPtr generator, GeneratorType type);

    /// <summary>Raw 32-bit outputs.</summary>
    [DllImport(Lib)]
    public static extern Status curandGenerate(IntPtr generator, IntPtr outputPtr, ulong num);

    /// <summary>The number of dimensions of a quasi-random generator (default 1).</summary>
    [DllImport(Lib)]
    public static extern Status curandSetQuasiRandomGeneratorDimensions(IntPtr generator, uint dimensions);

    /// <summary>A pointer to cuRAND's table of 32-bit direction vectors: 20,000 dimensions of 32 words each.
    /// <paramref name="set"/> 101 is the Joe-Kuo set (CURAND_DIRECTION_VECTORS_32_JOEKUO6).</summary>
    [DllImport(Lib)]
    public static extern Status curandGetDirectionVectors32(out IntPtr vectors, int set);

    [DllImport(Lib)]
    public static extern Status curandSetPseudoRandomGeneratorSeed(IntPtr generator, ulong seed);

    [DllImport(Lib)]
    public static extern Status curandSetGeneratorOffset(IntPtr generator, ulong offset);

    /// <summary>The stream the generator's kernels run on (zero is the legacy default stream).</summary>
    [DllImport(Lib)]
    public static extern Status curandSetStream(IntPtr generator, IntPtr stream);

    [DllImport(Lib)]
    public static extern Status curandGenerateUniform(IntPtr generator, IntPtr outputPtr, ulong num);

    [DllImport(Lib)]
    public static extern Status curandGenerateUniformDouble(IntPtr generator, IntPtr outputPtr, ulong num);

    [DllImport(Lib)]
    public static extern Status curandGenerateNormal(IntPtr generator, IntPtr outputPtr, ulong num, float mean, float stddev);

    [DllImport(Lib)]
    public static extern Status curandGenerateNormalDouble(IntPtr generator, IntPtr outputPtr, ulong num, double mean, double stddev);

    [DllImport(Lib)]
    public static extern Status curandGetVersion(out int version);

    /// <summary>Whether the cuRAND library loads and the driver reports a CUDA device. Cached at module init so the
    /// hot path is a single static-bool read.</summary>
    /// <remarks>Both checks are permanent facts about the host. The probe used to create a generator instead, which
    /// starts the CUDA runtime on the device: a transient failure there (another thread's stream capture open on the
    /// primary context, a device busy for a moment) was cached for the life of the process, and every cuRAND path
    /// then ran on the CPU - the CUDA lane saw all its cuRAND tests skip in one run and pass in the next. A device
    /// call that fails now falls back to the CPU generator for that call only.</remarks>
    public static readonly bool IsAvailable;

    private static bool Probe()
    {
        try
        {
            if (curandGetVersion(out _) != Status.Success) return false;
            return CuBlasNative.cuInit(0) == CudaResult.Success
                && CuBlasNative.cuDeviceGetCount(out int devices) == CudaResult.Success && devices > 0;
        }
        catch (DllNotFoundException) { return false; }
        catch (EntryPointNotFoundException) { return false; }
        catch (BadImageFormatException) { return false; }
        catch { return false; }
    }
}
