using System;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA;

/// <summary>
/// Resolves the CUDA context host-wait policy shared by the production backend, legacy
/// cuBLAS/cuDNN context, and standalone direct-PTX runtime. Latency-sensitive tensor dispatch
/// defaults to <c>CU_CTX_SCHED_SPIN</c>; callers that prefer lower idle CPU use can select another
/// driver policy explicitly.
/// </summary>
internal static class CudaContextScheduling
{
    internal const string EnvironmentVariable = "AIDOTNET_CUDA_CONTEXT_SCHEDULING";
    internal const uint Auto = 0u;
    internal const uint Spin = 1u;
    internal const uint Yield = 2u;
    internal const uint Blocking = 4u;

    internal static uint ResolveFromEnvironment() =>
        Resolve(Environment.GetEnvironmentVariable(EnvironmentVariable));

    internal static uint Resolve(string? value) => value?.Trim().ToLowerInvariant() switch
    {
        null or "" or "spin" => Spin,
        "auto" => Auto,
        "yield" => Yield,
        "blocking" => Blocking,
        _ => throw new InvalidOperationException(
            $"{EnvironmentVariable} must be one of: spin, auto, yield, blocking.")
    };

    internal static string Describe(uint flags) => flags switch
    {
        Auto => "auto",
        Spin => "spin",
        Yield => "yield",
        Blocking => "blocking",
        _ => $"unknown({flags})"
    };
}
