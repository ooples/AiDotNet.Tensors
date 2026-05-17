using System;

namespace AiDotNet.Tensors.Engines.BlasManaged;

/// <summary>
/// Utility for K-axis parallelism: partitions a K-range [0, K) into
/// <paramref name="numThreads"/> contiguous sub-ranges of (roughly) equal size.
///
/// <para>
/// Used by the K-axis parallel strategy (not yet integrated into the
/// dispatcher — see Phase G5 / AxisSelector). Each thread is assigned a
/// sub-range; it accumulates into its private partial-C buffer. After all
/// threads finish, ReductionTree merges the partials in fixed pairwise order
/// for bit-deterministic output.
/// </para>
///
/// <para>
/// Partition strategy: balanced split. K may not divide numThreads evenly —
/// the first (K % numThreads) threads get one extra K element.
/// </para>
/// </summary>
internal static class KAxisDriver
{
    /// <summary>
    /// Compute the K-range assigned to <paramref name="threadIndex"/> for a
    /// problem with total K = <paramref name="k"/> split across
    /// <paramref name="numThreads"/> threads.
    /// </summary>
    /// <param name="k">Total K dimension.</param>
    /// <param name="numThreads">Number of partition threads.</param>
    /// <param name="threadIndex">Thread index in [0, numThreads).</param>
    /// <returns>(start, length) — the thread covers K indices [start, start + length).</returns>
    public static (int Start, int Length) GetThreadRange(int k, int numThreads, int threadIndex)
    {
        if (k < 0) throw new ArgumentOutOfRangeException(nameof(k));
        if (numThreads <= 0) throw new ArgumentOutOfRangeException(nameof(numThreads));
        if (threadIndex < 0 || threadIndex >= numThreads)
            throw new ArgumentOutOfRangeException(nameof(threadIndex));

        int baseSize = k / numThreads;
        int extras = k % numThreads;

        // First `extras` threads get one extra element.
        int start, length;
        if (threadIndex < extras)
        {
            length = baseSize + 1;
            start = threadIndex * length;
        }
        else
        {
            length = baseSize;
            start = extras * (baseSize + 1) + (threadIndex - extras) * baseSize;
        }

        return (start, length);
    }

    /// <summary>
    /// Returns the smallest power-of-two upper bound for <paramref name="numThreads"/>.
    /// Useful when allocating partials arrays sized for ReductionTree.
    /// </summary>
    public static int RoundUpToPowerOfTwo(int numThreads)
    {
        if (numThreads <= 1) return 1;
        int p = 1;
        while (p < numThreads) p <<= 1;
        return p;
    }
}
