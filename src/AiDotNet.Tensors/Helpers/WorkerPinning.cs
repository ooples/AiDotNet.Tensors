namespace AiDotNet.Tensors.Helpers;

/// <summary>
/// Whether the CPU worker pool pins its threads to distinct physical cores.
/// </summary>
/// <remarks>
/// <para>
/// Without pinning the OS may place two busy workers on the two SMT siblings of one core, and a
/// parallel op then waits on its slowest, core-sharing chunk. On a 16-core/32-thread Ryzen 9 3950X,
/// pinning at 16 threads made a transformer attention block 1.13x faster and narrowed the spread of
/// identical GEMM tiles from 1.33-2.23 ms to 1.32-1.79 ms. At 32 threads, where the extra workers sit
/// on SMT siblings regardless, it measured no clear gain (#653).
/// </para>
/// <para>
/// Pinning is Windows-only, never overrides a restricted process affinity, and is skipped on machines
/// with more than one processor group. The calling thread, which runs one share of every parallel op,
/// is never pinned.
/// </para>
/// </remarks>
public enum WorkerPinning
{
    /// <summary>
    /// Pin while <see cref="CpuParallelSettings.MaxDegreeOfParallelism"/> is at most the physical core
    /// count, where it measured a win. Above it, and where that pinning does not apply (more than 64 logical
    /// processors, or a process limited to part of the machine), each worker is still bound to a physical core of
    /// its own so the chunk it works on stays in that core's caches across GC suspensions.
    /// </summary>
    Auto,

    /// <summary>Always pin workers to physical cores, round-robin (bound to a core of its own where that pinning does not apply).</summary>
    Always,

    /// <summary>Never pin or bind; the OS places every worker.</summary>
    Never,
}
