using System;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA;

/// <summary>
/// Resolves the CUDA host-wait policy (<see cref="GpuHostWaitPolicy"/>) shared by the production backend, legacy
/// cuBLAS/cuDNN context, and standalone direct-PTX runtime: the context scheduling flag a context is created with and
/// the way stream waits block. The typed <see cref="GpuHostWaitOptions.Policy"/> decides it unless the
/// <c>AIDOTNET_CUDA_CONTEXT_SCHEDULING</c> environment variable names a policy. The default,
/// <see cref="GpuHostWaitPolicy.PollThenBlock"/>, keeps <c>CU_CTX_SCHED_SPIN</c> for latency-sensitive dispatch and
/// has stream waits block once they outlast a short poll.
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

    /// <summary>The context scheduling flag of <paramref name="value"/> (an environment value; empty: the typed option).</summary>
    internal static uint Resolve(string? value) => Flags(ResolvePolicy(value));

    internal static GpuHostWaitPolicy ResolvePolicy(string? value) => value?.Trim().ToLowerInvariant() switch
    {
        null or "" => GpuHostWaitOptions.Policy,
        "spin" => GpuHostWaitPolicy.Spin,
        "auto" => GpuHostWaitPolicy.Auto,
        "yield" => GpuHostWaitPolicy.Yield,
        "blocking" => GpuHostWaitPolicy.BlockingSync,
        "poll-then-block" or "pollthenblock" => GpuHostWaitPolicy.PollThenBlock,
        _ => throw new InvalidOperationException(
            $"{EnvironmentVariable} must be one of: spin, auto, yield, blocking, poll-then-block.")
    };

    /// <summary>The <c>CU_CTX_SCHED_*</c> flag a policy creates its context with.</summary>
    internal static uint Flags(GpuHostWaitPolicy policy) => policy switch
    {
        GpuHostWaitPolicy.Yield => Yield,
        GpuHostWaitPolicy.BlockingSync => Blocking,
        GpuHostWaitPolicy.Auto => Auto,
        // PollThenBlock spins in the driver's own short waits; its stream waits block once the poll window has passed.
        _ => Spin,
    };

    // The environment's policy, read once: 0 unread, 1 none (unset or not a policy name), else 2 + the policy.
    private static int s_environmentPolicy;

    /// <summary>The policy stream waits follow: the environment variable's when it names one, else the typed option.</summary>
    internal static GpuHostWaitPolicy EffectivePolicy
    {
        get
        {
            int state = System.Threading.Volatile.Read(ref s_environmentPolicy);
            if (state == 0)
            {
                string? value = Environment.GetEnvironmentVariable(EnvironmentVariable);
                state = 1;
                // A misspelled value already failed the backend's construction (ResolveFromEnvironment); here it reads as unset.
                if (!string.IsNullOrWhiteSpace(value))
                {
                    try { state = 2 + (int)ResolvePolicy(value); }
                    catch (InvalidOperationException) { state = 1; }
                }
                System.Threading.Volatile.Write(ref s_environmentPolicy, state);
            }
            return state >= 2 ? (GpuHostWaitPolicy)(state - 2) : GpuHostWaitOptions.Policy;
        }
    }

    internal static string Describe(uint flags) => flags switch
    {
        Auto => "auto",
        Spin => "spin",
        Yield => "yield",
        Blocking => "blocking",
        _ => $"unknown({flags})"
    };
}
