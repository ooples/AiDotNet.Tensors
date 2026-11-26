using System;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA;

/// <summary>
/// Resolves the CUDA host-wait policy (<see cref="GpuHostWaitPolicy"/>) shared by the production backend, legacy
/// cuBLAS/cuDNN context, and standalone direct-PTX runtime: the context scheduling flag a context is created with and
/// the way stream waits block. The typed <see cref="GpuHostWaitOptions.Policy"/> decides it. The deprecated
/// <c>AIDOTNET_CUDA_CONTEXT_SCHEDULING</c> environment variable is honored, with a warning, only while the typed policy
/// has never been set. The default, <see cref="GpuHostWaitPolicy.PollThenBlock"/>, keeps <c>CU_CTX_SCHED_SPIN</c> for
/// latency-sensitive dispatch and has stream waits block once they outlast a short poll.
/// </summary>
internal static class CudaContextScheduling
{
    internal const string EnvironmentVariable = "AIDOTNET_CUDA_CONTEXT_SCHEDULING";
    internal const uint Auto = 0u;
    internal const uint Spin = 1u;
    internal const uint Yield = 2u;
    internal const uint Blocking = 4u;

    /// <summary>
    /// The scheduling flag a new context is created with (<see cref="EffectivePolicy"/>). A misspelled deprecated
    /// environment value, when it is the one that decides, throws: a caller error, not a missing device.
    /// </summary>
    // The policy whose flag the first CUDA context was created with (-1: no context yet).
    private static int s_contextPolicy = -1;
    private static int s_policyChangeWarned;

    internal static uint ResolveContextFlags()
    {
        var resolved = ResolveContextPolicy();
        System.Threading.Interlocked.CompareExchange(ref s_contextPolicy, (int)resolved, -1);
        return Flags(resolved);
    }

    /// <summary>
    /// Called when <see cref="GpuHostWaitOptions.Policy"/> is set: a context that already exists keeps the scheduling
    /// flag it was created with, which is reported once rather than ignored silently.
    /// </summary>
    internal static void NotePolicySet(GpuHostWaitPolicy policy)
    {
        int created = System.Threading.Volatile.Read(ref s_contextPolicy);
        if (created < 0 || Flags((GpuHostWaitPolicy)created) == Flags(policy)) return;
        if (System.Threading.Interlocked.Exchange(ref s_policyChangeWarned, 1) == 0)
            System.Diagnostics.Trace.TraceWarning(
                $"{nameof(GpuHostWaitOptions)}.{nameof(GpuHostWaitOptions.Policy)} was set to {policy} after a CUDA context was "
                + $"created with {(GpuHostWaitPolicy)created}'s scheduling flag; existing contexts keep it. Stream waits use "
                + "the new policy; set it before the first GPU engine to change the flag too.");
    }

    private static GpuHostWaitPolicy ResolveContextPolicy()
    {
        if (!GpuHostWaitOptions.IsPolicySet)
        {
            string? value = Environment.GetEnvironmentVariable(EnvironmentVariable);
            if (!string.IsNullOrWhiteSpace(value)) _ = ParseLegacy(value!);   // throws on a misspelling
        }
        return EffectivePolicy;
    }

    /// <summary>The context scheduling flag a deprecated environment value names (empty: the default policy's).</summary>
    internal static uint Resolve(string? value) => Flags(ResolvePolicy(value));

    /// <summary>The policy a deprecated environment value names (empty: the default policy).</summary>
    internal static GpuHostWaitPolicy ResolvePolicy(string? value)
        => string.IsNullOrWhiteSpace(value) ? GpuHostWaitPolicy.PollThenBlock : ParseLegacy(value!);

    private static GpuHostWaitPolicy ParseLegacy(string value) => value.Trim().ToLowerInvariant() switch
    {
        "spin" => GpuHostWaitPolicy.Spin,
        "auto" => GpuHostWaitPolicy.Auto,
        "yield" => GpuHostWaitPolicy.Yield,
        "blocking" => GpuHostWaitPolicy.BlockingSync,
        "poll-then-block" or "pollthenblock" => GpuHostWaitPolicy.PollThenBlock,
        _ => throw new InvalidOperationException(
            $"{EnvironmentVariable} must be one of: spin, auto, yield, blocking, poll-then-block. It is deprecated: "
            + $"set {nameof(GpuHostWaitOptions)}.{nameof(GpuHostWaitOptions.Policy)} instead.")
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

    // The deprecated environment variable's policy, read once: 0 unread, 1 none (unset or not a policy name), else
    // 2 + the policy.
    private static int s_environmentPolicy;
    private static int s_warned;

    /// <summary>
    /// The policy in effect: <see cref="GpuHostWaitOptions.Policy"/> once it has been set; before that the deprecated
    /// environment variable's when it names one (warned once), else the typed default.
    /// </summary>
    internal static GpuHostWaitPolicy EffectivePolicy
    {
        get
        {
            if (GpuHostWaitOptions.IsPolicySet) return GpuHostWaitOptions.Policy;
            int state = System.Threading.Volatile.Read(ref s_environmentPolicy);
            if (state == 0)
            {
                string? value = Environment.GetEnvironmentVariable(EnvironmentVariable);
                state = 1;
                // A misspelled value fails the backend's construction (ResolveContextFlags); here it reads as unset.
                if (!string.IsNullOrWhiteSpace(value))
                {
                    try { state = 2 + (int)ParseLegacy(value!); }
                    catch (InvalidOperationException) { state = 1; }
                }
                System.Threading.Volatile.Write(ref s_environmentPolicy, state);
            }
            if (state < 2) return GpuHostWaitOptions.Policy;
            if (System.Threading.Interlocked.Exchange(ref s_warned, 1) == 0)
                System.Diagnostics.Trace.TraceWarning(
                    $"{EnvironmentVariable} is deprecated and will be removed: set {nameof(GpuHostWaitOptions)}."
                    + $"{nameof(GpuHostWaitOptions.Policy)} instead, which wins over it once set.");
            return (GpuHostWaitPolicy)(state - 2);
        }
    }

    /// <summary>Tests: forgets the environment variable's cached reading and the warning.</summary>
    internal static void ResetEnvironmentCacheForTests()
    {
        System.Threading.Volatile.Write(ref s_environmentPolicy, 0);
        System.Threading.Volatile.Write(ref s_warned, 0);
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
