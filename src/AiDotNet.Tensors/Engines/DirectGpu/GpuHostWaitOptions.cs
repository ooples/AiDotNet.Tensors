// Copyright (c) CrowdTrainer LLC. All rights reserved.
using System;

namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>
/// How a host thread waits for GPU work it has to see finished (a stream synchronization, such as the one before a
/// device-to-host read). The policy also sets the CUDA context's scheduling flag (<c>CU_CTX_SCHED_*</c>), which governs
/// the driver's own blocking calls.
/// </summary>
public enum GpuHostWaitPolicy
{
    /// <summary>
    /// Default. Polls the stream for <see cref="GpuHostWaitOptions.PollWindow"/> (a wait that short ends exactly as a
    /// spin wait does), then blocks on a blocking-sync event, so a wait that lasts (a device starved of memory by other
    /// processes, a long kernel) does not keep a CPU core busy for its whole length. The context runs with
    /// <c>CU_CTX_SCHED_SPIN</c>.
    /// </summary>
    PollThenBlock = 0,

    /// <summary><c>CU_CTX_SCHED_SPIN</c>: every wait spins on a core until the device finishes, however long that takes.</summary>
    Spin = 1,

    /// <summary><c>CU_CTX_SCHED_YIELD</c>: every wait yields its thread between polls (the core stays busy when nothing else runs).</summary>
    Yield = 2,

    /// <summary><c>CU_CTX_SCHED_BLOCKING_SYNC</c>: every wait blocks on an OS primitive; the host wakes after the device signals.</summary>
    BlockingSync = 3,

    /// <summary>
    /// <c>CU_CTX_SCHED_AUTO</c> (the CUDA runtime's <c>cudaDeviceScheduleAuto</c> default): the driver spins while the
    /// process has fewer active CUDA contexts than logical processors and yields otherwise.
    /// </summary>
    Auto = 4,
}

/// <summary>
/// Process-wide host-wait settings of the CUDA backend (<see cref="GpuHostWaitPolicy"/>).
/// </summary>
/// <remarks>
/// The scheduling flag is applied when a CUDA context is created, so set <see cref="Policy"/> before the first GPU
/// engine is constructed; a context that is already active keeps its flag. <see cref="GpuHostWaitPolicy.PollThenBlock"/>
/// stream waits read the settings on every wait. The older <c>AIDOTNET_CUDA_CONTEXT_SCHEDULING</c> environment variable
/// is deprecated: it is read only while <see cref="Policy"/> has never been set, with a warning, and a set
/// <see cref="Policy"/> always wins over it.
/// </remarks>
public static class GpuHostWaitOptions
{
    private static int _policy = (int)GpuHostWaitPolicy.PollThenBlock;
    private static int _policySet;
    private static long _pollWindowTicks = TimeSpan.FromMilliseconds(2).Ticks;

    /// <summary>The host-wait policy. Default: <see cref="GpuHostWaitPolicy.PollThenBlock"/>.</summary>
    public static GpuHostWaitPolicy Policy
    {
        get => (GpuHostWaitPolicy)System.Threading.Volatile.Read(ref _policy);
        set
        {
            if (value < GpuHostWaitPolicy.PollThenBlock || value > GpuHostWaitPolicy.Auto)
                throw new ArgumentOutOfRangeException(nameof(value), value, "Unknown host-wait policy.");
            System.Threading.Volatile.Write(ref _policy, (int)value);
            System.Threading.Volatile.Write(ref _policySet, 1);
            CUDA.CudaContextScheduling.NotePolicySet(value);
        }
    }

    /// <summary>Whether <see cref="Policy"/> has been set (then it wins over the deprecated environment variable).</summary>
    internal static bool IsPolicySet => System.Threading.Volatile.Read(ref _policySet) != 0;

    /// <summary>Tests: puts the policy and whether it was set back as they were.</summary>
    internal static void RestoreForTests(GpuHostWaitPolicy policy, bool set)
    {
        System.Threading.Volatile.Write(ref _policy, (int)policy);
        System.Threading.Volatile.Write(ref _policySet, set ? 1 : 0);
    }

    /// <summary>
    /// How long a <see cref="GpuHostWaitPolicy.PollThenBlock"/> wait polls before it blocks. Default: 2 ms. Zero blocks
    /// at once when the work is not already finished.
    /// </summary>
    public static TimeSpan PollWindow
    {
        get => TimeSpan.FromTicks(System.Threading.Interlocked.Read(ref _pollWindowTicks));
        set
        {
            if (value < TimeSpan.Zero)
                throw new ArgumentOutOfRangeException(nameof(value), value, "The poll window cannot be negative.");
            System.Threading.Interlocked.Exchange(ref _pollWindowTicks, value.Ticks);
        }
    }
}
