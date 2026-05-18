using System;

namespace AiDotNet.Tensors.Engines.BlasManaged;

/// <summary>
/// Deterministic pairwise-sum reduction over N partial accumulators.
/// Used by the K-axis split parallelism: each thread accumulates a partial
/// C[Mc × Nc] for its K-range; ReductionTree sums all partials into a single
/// C in a FIXED pairwise order regardless of which thread finished first.
///
/// <para>
/// Determinism rationale: floating-point addition is non-associative
/// (a + b + c rounds differently than a + (b + c) for large magnitudes).
/// If the reduction order depends on thread completion order (e.g., naive
/// <c>for thread: C += partials[thread]</c>), the bit pattern of the result
/// changes from run to run. The tree reduction in fixed thread-id pairwise
/// order eliminates this variance.
/// </para>
///
/// <para>
/// Algorithm: pair adjacent slots (0+1, 2+3, ...), then pairs of pairs
/// ((0+1)+(2+3)), and so on. The final result lands in slot 0. Total
/// passes = ceil(log2(N)). Each pass does O(elementsPerSlot * numActiveSlots) work.
/// </para>
/// </summary>
internal static class ReductionTree
{
    /// <summary>
    /// Sum N partial buffers in fixed pairwise order into <paramref name="partials"/>[0].
    /// On return, <c>partials[0]</c> holds the reduced sum; the other slots are
    /// undefined (they've been used as scratch).
    /// </summary>
    /// <param name="partials">Array of partial accumulators. Each element is a <see cref="Memory{T}"/> of <paramref name="elementCount"/> doubles.</param>
    /// <param name="elementCount">Number of doubles in each partial slot.</param>
    public static void ReducePairwiseFp64(Memory<double>[] partials, int elementCount)
    {
        if (partials is null) throw new ArgumentNullException(nameof(partials));
        if (partials.Length == 0) return;
        if (partials.Length == 1) return;  // Already reduced.

        // Validate input lengths.
        for (int i = 0; i < partials.Length; i++)
        {
            if (partials[i].Length < elementCount)
                throw new ArgumentException($"partials[{i}] is shorter than elementCount.", nameof(partials));
        }

        // Pairwise reduction in fixed order: (0+1), (2+3), ... then ((0+1)+(2+3)) ...
        int active = partials.Length;
        while (active > 1)
        {
            int half = active / 2;
            for (int i = 0; i < half; i++)
            {
                // partials[2*i] += partials[2*i + 1]
                var dst = partials[2 * i].Span;
                var src = partials[2 * i + 1].Span;
                for (int e = 0; e < elementCount; e++)
                    dst[e] += src[e];
            }
            // CodeRabbit #366: compact merged pairs FIRST. The previous
            // version copied the odd remainder into partials[half] before
            // compaction; for active=5 (half=2) that overwrites slot 2,
            // which still held the merged (2+3) pair, silently dropping it
            // from the reduction.
            // Compact merged values from even indices 0,2,...,2*(half-1)
            // into contiguous slots 0,1,...,half-1.
            for (int i = 1; i < half; i++)
            {
                partials[i] = partials[2 * i];
            }
            if (active % 2 == 1)
            {
                // Rebind slot `half` to the lone odd's Memory<T> handle — a
                // reference copy, not a span copy. Safe now because
                // compaction has already moved any pair away from slot half.
                partials[half] = partials[active - 1];
                active = half + 1;
            }
            else
            {
                active = half;
            }
        }
    }

    /// <summary>
    /// FP32 mirror of <see cref="ReducePairwiseFp64"/>.
    /// </summary>
    public static void ReducePairwiseFp32(Memory<float>[] partials, int elementCount)
    {
        if (partials is null) throw new ArgumentNullException(nameof(partials));
        if (partials.Length == 0) return;
        if (partials.Length == 1) return;

        for (int i = 0; i < partials.Length; i++)
        {
            if (partials[i].Length < elementCount)
                throw new ArgumentException($"partials[{i}] is shorter than elementCount.", nameof(partials));
        }

        int active = partials.Length;
        while (active > 1)
        {
            int half = active / 2;
            for (int i = 0; i < half; i++)
            {
                var dst = partials[2 * i].Span;
                var src = partials[2 * i + 1].Span;
                for (int e = 0; e < elementCount; e++)
                    dst[e] += src[e];
            }
            // CodeRabbit #366: same compact-before-rebind ordering fix as
            // the FP64 path — see ReducePairwiseFp64 for rationale.
            for (int i = 1; i < half; i++)
            {
                partials[i] = partials[2 * i];
            }
            if (active % 2 == 1)
            {
                partials[half] = partials[active - 1];
                active = half + 1;
            }
            else
            {
                active = half;
            }
        }
    }
}
