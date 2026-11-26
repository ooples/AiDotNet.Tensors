using System;
using System.Runtime.CompilerServices;

namespace AiDotNet.Tensors.Engines;

/// <summary>
/// Module-load-time request for GPU auto-detection. On .NET 6+ this fires the moment
/// the AiDotNet.Tensors assembly is loaded and records that detection is owed; the
/// detection runs on the first read of <see cref="AiDotNetEngine.Current"/> — before any
/// model construction or tensor op uses an engine — so consumers get GPU acceleration
/// without an explicit <see cref="AiDotNetEngine.AutoDetectAndConfigureGpu"/> call, and a
/// process that never asks for an engine never probes for a GPU.
/// </summary>
/// <remarks>
/// <para>
/// Defaults are "always on, opt-out via environment variable" so paper-canonical
/// performance is the out-of-the-box behaviour. Previously
/// <see cref="AiDotNetEngine.Current"/> defaulted to <see cref="CpuEngine"/>
/// and nothing in the consumer code, test base, or model classes called
/// <see cref="AiDotNetEngine.AutoDetectAndConfigureGpu"/>, so every
/// <c>TryForwardGpuOptimized(...)</c> in the model layer was effectively
/// dead code on the default path.
/// </para>
/// <para>
/// <b>Opt-out:</b> set <c>AIDOTNET_DISABLE_GPU</c> to any non-empty value
/// before the process starts (it's checked at module-init time, BEFORE
/// the AiDotNet.Tensors assembly's static state is touched).
/// </para>
/// <para>
/// <b>Failure modes:</b> if <see cref="DirectGpuTensorEngine"/>'s ctor
/// throws (no GPU, missing driver, missing CUDA runtime, etc.), the
/// initializer swallows the exception and leaves <see cref="AiDotNetEngine.Current"/>
/// on its CPU default. The initializer itself MUST NOT throw — a thrown
/// ModuleInitializer aborts the entire AppDomain.
/// </para>
/// </remarks>
internal static class GpuAutoDetectModuleInit
{
#if NET6_0_OR_GREATER
#pragma warning disable CA2255 // ModuleInitializer used intentionally
    [ModuleInitializer]
#pragma warning restore CA2255
    internal static void AutoDetectGpuIfNotOptedOut()
    {
        try
        {
            // BEFORE the opt-out gate, deliberately. This installs the crash-time diagnostics dump
            // (AIDOTNET_GPU_DIAGNOSTICS_DUMP) and costs nothing when that variable is unset.
            //
            // It cannot live in GpuKernelDiagnostics's static constructor alone: a static
            // constructor runs on FIRST USE of the type, so a process that never touches the class
            // -- a CPU-only CI shard, or a run that dies before reaching any GPU code -- never
            // installs the handler and writes no journal. That is backwards for crash forensics,
            // where the runs you most need evidence from are exactly the ones that got nowhere.
            // Observed: the variable was set on every shard of a full CI matrix and not one file
            // was produced.
            //
            // Registered even when AIDOTNET_DISABLE_GPU is set, because the buffer-residency
            // counters and the launch journal are diagnostics about the process, not GPU work being
            // requested, and a host dying with GPU disabled is still worth attributing.
            DirectGpu.GpuKernelDiagnostics.EnsureRegistered();

            // Opt-out gate: AIDOTNET_DISABLE_GPU set to any non-empty
            // value skips auto-detection. Mirrors BlasEnvDefault's
            // env-var pattern on the consumer side. Users who explicitly
            // want CPU (numerical reproducibility, fp64 on consumer GPUs
            // that throttle double precision, benchmarking baselines)
            // set this before process startup.
            // Set, the detection is still owed rather than skipped: the first read of Current runs it, and it returns
            // at once on the variable, unless GpuBackendOptions.RequiredBackend was set by then, which it refuses
            // instead of handing out the CPU engine the requirement rules out.

            // AutoDetectAndConfigureGpu already wraps DirectGpuTensorEngine
            // construction in try/catch and falls back to CPU on failure,
            // so this call is safe at module init even without a GPU /
            // driver / CUDA runtime present.
            //
            // Module-init verbosity is OFF by default — emitting [AiDotNet]
            // status to stdout at assembly-load time can break test runners
            // that parse stdout (xUnit reporters, MSBuild diagnostics),
            // hosted services that route stdout to structured loggers, and
            // any consumer that expects clean program output. Users who
            // WANT init-time logging have ONE opt-in: set the
            // AIDOTNET_VERBOSE_INIT env var BEFORE process startup. The
            // AiDotNetEngine.Logger property is too late at this point —
            // module initializers run at assembly load, before any consumer
            // code can set in-assembly statics — so Logger only affects
            // log lines emitted by AutoDetectAndConfigureGpu / ResetToCpu
            // calls made AFTER the assembly is loaded. AIDOTNET_QUIET
            // overrides AIDOTNET_VERBOSE_INIT for callers who want zero noise.
            var verboseInit = !string.IsNullOrEmpty(
                Environment.GetEnvironmentVariable("AIDOTNET_VERBOSE_INIT"));
            // The detection itself is owed, not run here: it happens on the first read of
            // AiDotNetEngine.Current. Probing at assembly load made every consumer pay for it -- about two
            // seconds and a page of driver log -- including the ones that call only the SIMD kernels or the
            // helpers and never ask for an engine. Anything that does ask for one still gets the GPU without
            // an explicit AutoDetectAndConfigureGpu call, which is what this initializer exists to guarantee.
            AiDotNetEngine.DeferGpuAutoDetection(verbose: verboseInit);
        }
        catch
        {
            // Defensive belt-and-suspenders: AutoDetectAndConfigureGpu
            // already catches its own exceptions, but a thrown
            // ModuleInitializer takes down the entire AppDomain. Keep
            // a final guard so module load never fails on a GPU probe.
        }
    }
#endif
}
