using System;
using System.Runtime.CompilerServices;

namespace AiDotNet.Tensors.Engines;

/// <summary>
/// Module-load-time GPU auto-detection. On .NET 6+ this fires the moment
/// the AiDotNet.Tensors assembly is loaded — before any model construction
/// or tensor op — so consumers get GPU acceleration without an explicit
/// <see cref="AiDotNetEngine.AutoDetectAndConfigureGpu"/> call.
/// </summary>
/// <remarks>
/// <para>
/// Mirrors the pattern used by the consumer-side <c>BlasEnvDefault</c>
/// initializer (which auto-sets <c>AIDOTNET_USE_BLAS=1</c>): defaults are
/// "always on, opt-out via environment variable" so paper-canonical
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
/// the AiDotNet.Tensors assembly's static state is touched). Same pattern
/// as the existing <c>AIDOTNET_USE_BLAS</c> env var.
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
            // Opt-out gate: AIDOTNET_DISABLE_GPU set to any non-empty
            // value skips auto-detection. Mirrors BlasEnvDefault's
            // env-var pattern on the consumer side. Users who explicitly
            // want CPU (numerical reproducibility, fp64 on consumer GPUs
            // that throttle double precision, benchmarking baselines)
            // set this before process startup.
            var disable = Environment.GetEnvironmentVariable("AIDOTNET_DISABLE_GPU");
            if (!string.IsNullOrEmpty(disable))
                return;

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
            AiDotNetEngine.AutoDetectAndConfigureGpu(verbose: verboseInit);
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
