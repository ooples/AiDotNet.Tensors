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
            // Odd remainder: the last slot moves to slot `half` if active was odd.
            if (active % 2 == 1)
            {
                // Move partials[active-1] into the new slot [half]. The new slot
                // [half] was previously written by the even-pair loop (since half
                // pairs touch even indices 0,2,...,2*(half-1)), so partials[half]
                // is a free slot at this point. Copy data into it.
                var src = partials[active - 1].Span;
                var dst = partials[half].Span;
                for (int e = 0; e < elementCount; e++)
                    dst[e] = src[e];
                // After this copy the active windows are: partials[0], partials[2],
                // partials[4], ..., partials[2*(half-1)], partials[half].
                // We compact them to partials[0..active] below.
                active = half + 1;
            }
            else
            {
                active = half;
            }
            // Compact: merged values sit at even indices 0,2,...,2*(half-1).
            // Move them to contiguous slots 0,1,...,half-1 for the next pass.
            for (int i = 1; i < half; i++)
            {
                partials[i] = partials[2 * i];
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
            if (active % 2 == 1)
            {
                var src = partials[active - 1].Span;
                var dst = partials[half].Span;
                for (int e = 0; e < elementCount; e++)
                    dst[e] = src[e];
                active = half + 1;
            }
            else
            {
                active = half;
            }
            for (int i = 1; i < half; i++)
            {
                partials[i] = partials[2 * i];
            }
        }
    }
}
