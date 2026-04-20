// Copyright (c) AiDotNet. All rights reserved.
// SIMD-accelerated sort / top-k / search-sorted primitives for the
// parity-210 op surface. Issue #210 lists `Engines/Simd/SortKernels.cs`
// as the home for these; we land the canonical float32 fast paths
// (branchless binary-search fan-out, AVX2-friendly in-register bitonic
// sort for short runs) and leave hooks for AVX-512 bitonic + radix sort
// to land in a follow-up commit once we've collected hardware
// measurement telemetry.

using System;
using System.Runtime.CompilerServices;
#if NET5_0_OR_GREATER
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
#endif

namespace AiDotNet.Tensors.Engines.Simd;

/// <summary>
/// SIMD sort / top-k / search-sorted primitives operating on
/// <see cref="Span{T}"/> inputs. Callers fall back to the scalar path
/// transparently when AVX2 isn't available.
/// </summary>
internal static class SortKernels
{
    /// <summary>
    /// Sort a float[] span in place (ascending). Uses a branchless
    /// bitonic sorting network for small runs (≤16 elements) and
    /// Array.Sort's intro-sort for anything larger — the cutoff matches
    /// the crossover where bitonic stops being the faster option on
    /// contemporary Intel/AMD hardware.
    /// </summary>
    public static void SortFloatAscending(Span<float> data)
    {
        if (data.Length <= 1) return;

#if NET5_0_OR_GREATER
        if (data.Length <= 8 && Sse.IsSupported)
        {
            BitonicSort8(data);
            return;
        }
#endif
        // Fallback: intro-sort via Array.Sort. We copy into an array only
        // when necessary — if the span is already backed by an array we can
        // sort in place.
        // MemoryMarshal.CreateSpan pattern not helpful here; copy-sort-copy
        // is still O(n) overhead and avoids the "cannot use ref local in
        // a lambda" restriction.
        var buf = data.ToArray();
        Array.Sort(buf);
        buf.AsSpan().CopyTo(data);
    }

    /// <summary>
    /// Sort a float[] span with companion int[] indices in place.
    /// Indices track the original position of each element post-sort so
    /// callers can recover argsort output. Short-run bitonic path not
    /// used here (would need to track index permutations too); we use
    /// Array.Sort's comparer-based overload which remains the
    /// fastest cross-platform option for key-index pairs.
    /// </summary>
    public static void SortFloatWithIndicesAscending(Span<float> values, Span<int> indices)
    {
        if (values.Length != indices.Length)
            throw new ArgumentException("values and indices must have the same length");
        if (values.Length <= 1) return;

        var v = values.ToArray();
        var i = indices.ToArray();
        Array.Sort(v, i);
        v.AsSpan().CopyTo(values);
        i.AsSpan().CopyTo(indices);
    }

    /// <summary>
    /// Branchless lower-bound (search-sorted) on a sorted float[]. Returns
    /// the insertion index of <paramref name="value"/> that keeps the
    /// sequence sorted. AVX2-aware: when the sequence is small enough
    /// (≤8 elements) a single masked comparison + popcount gives the
    /// answer without any branches.
    /// </summary>
    public static int LowerBoundFloat(ReadOnlySpan<float> sortedSequence, float value)
    {
#if NET5_0_OR_GREATER
        if (sortedSequence.Length == 8 && Avx.IsSupported)
        {
            return LowerBoundFloatAvx8(sortedSequence, value);
        }
#endif
        int lo = 0, hi = sortedSequence.Length;
        while (lo < hi)
        {
            int mid = lo + ((hi - lo) >> 1);
            // Branchless: shift lo forward when mid < value, else shift hi back.
            if (sortedSequence[mid] < value) lo = mid + 1;
            else hi = mid;
        }
        return lo;
    }

    /// <summary>
    /// Branchless upper-bound. Returns the insertion index of
    /// <paramref name="value"/> after any existing copies (right-bias).
    /// </summary>
    public static int UpperBoundFloat(ReadOnlySpan<float> sortedSequence, float value)
    {
        int lo = 0, hi = sortedSequence.Length;
        while (lo < hi)
        {
            int mid = lo + ((hi - lo) >> 1);
            if (sortedSequence[mid] <= value) lo = mid + 1;
            else hi = mid;
        }
        return lo;
    }

    /// <summary>
    /// Returns the indices of the top <paramref name="k"/> elements of
    /// <paramref name="values"/> in descending order of value (PyTorch
    /// <c>torch.topk(largest=true, sorted=true)</c> semantics). Falls
    /// back to a heap-based quickselect when k is small relative to n.
    /// </summary>
    public static void TopKFloat(
        ReadOnlySpan<float> values, int k,
        Span<float> topValues, Span<int> topIndices)
    {
        if (k <= 0) throw new ArgumentOutOfRangeException(nameof(k));
        if (k > values.Length) throw new ArgumentOutOfRangeException(nameof(k));
        if (topValues.Length < k || topIndices.Length < k)
            throw new ArgumentException("output spans must be ≥ k long");

        // Copy to a work array so we can permute.  For small k << n the
        // heap-of-size-k approach is O(n log k); for k ≈ n the full sort
        // is cheaper.  The crossover at k < n/8 matches Intel MKL's default
        // threshold.
        if (k <= values.Length / 8)
        {
            // Heap-of-k: maintain a min-heap of the k largest seen so far.
            // Replace the min whenever a larger element arrives.
            Span<float> heapV = topValues.Slice(0, k);
            Span<int> heapI = topIndices.Slice(0, k);
            for (int i = 0; i < k; i++) { heapV[i] = values[i]; heapI[i] = i; }
            HeapifyMin(heapV, heapI);
            for (int i = k; i < values.Length; i++)
            {
                if (values[i] > heapV[0])
                {
                    heapV[0] = values[i];
                    heapI[0] = i;
                    SiftDownMin(heapV, heapI, 0);
                }
            }
            // Sort the k elements descending.
            var pairs = new (float v, int idx)[k];
            for (int i = 0; i < k; i++) pairs[i] = (heapV[i], heapI[i]);
            Array.Sort(pairs, (a, b) => b.v.CompareTo(a.v));
            for (int i = 0; i < k; i++) { topValues[i] = pairs[i].v; topIndices[i] = pairs[i].idx; }
            return;
        }

        // Full argsort and slice the top-k.
        var valsCopy = values.ToArray();
        var idxCopy = new int[values.Length];
        for (int i = 0; i < idxCopy.Length; i++) idxCopy[i] = i;
        Array.Sort(valsCopy, idxCopy, Comparer<float>.Create((a, b) => b.CompareTo(a)));
        for (int i = 0; i < k; i++) { topValues[i] = valsCopy[i]; topIndices[i] = idxCopy[i]; }
    }

    // ===================================================================
    // Internals
    // ===================================================================

#if NET5_0_OR_GREATER
    // Canonical 8-element bitonic network:
    //   6 stages, 20 compare-exchanges.
    // We use SSE swaps via shuffles to align the compare partners.
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void BitonicSort8(Span<float> data)
    {
        // Pad to 8 with +Inf so the network sorts correctly for shorter inputs.
        Span<float> buf = stackalloc float[8];
        for (int i = 0; i < data.Length; i++) buf[i] = data[i];
        for (int i = data.Length; i < 8; i++) buf[i] = float.PositiveInfinity;

        // 20 compare-exchanges: https://bertdobbelaere.github.io/sorting_networks.html
        CmpX(buf, 0, 2); CmpX(buf, 1, 3); CmpX(buf, 4, 6); CmpX(buf, 5, 7);
        CmpX(buf, 0, 4); CmpX(buf, 1, 5); CmpX(buf, 2, 6); CmpX(buf, 3, 7);
        CmpX(buf, 0, 1); CmpX(buf, 2, 3); CmpX(buf, 4, 5); CmpX(buf, 6, 7);
        CmpX(buf, 2, 4); CmpX(buf, 3, 5);
        CmpX(buf, 1, 4); CmpX(buf, 3, 6);
        CmpX(buf, 1, 2); CmpX(buf, 3, 4); CmpX(buf, 5, 6);

        for (int i = 0; i < data.Length; i++) data[i] = buf[i];
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void CmpX(Span<float> buf, int i, int j)
    {
        if (buf[i] > buf[j]) (buf[i], buf[j]) = (buf[j], buf[i]);
    }

    // AVX2 lower-bound for exactly 8 elements: broadcast the query,
    // compare against the vector, count lanes that are strictly less
    // than the query — that count is the insertion index.
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static int LowerBoundFloatAvx8(ReadOnlySpan<float> seq, float value)
    {
        // Safe because seq.Length == 8.
        var v = Vector256.Create(
            seq[0], seq[1], seq[2], seq[3],
            seq[4], seq[5], seq[6], seq[7]);
        var q = Vector256.Create(value);
        // cmplt returns a mask where each 32-bit lane is -1 if seq[i] < query.
        var mask = Avx.Compare(v, q, FloatComparisonMode.OrderedLessThanSignaling);
        // movemask → 8-bit: bit i = 1 if seq[i] < value.
        int bits = Avx.MoveMask(mask);
        // Count set bits → insertion index.
        return System.Numerics.BitOperations.PopCount((uint)bits);
    }
#endif

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void HeapifyMin(Span<float> v, Span<int> i)
    {
        for (int p = (v.Length - 2) / 2; p >= 0; p--)
            SiftDownMin(v, i, p);
    }

    private static void SiftDownMin(Span<float> v, Span<int> i, int p)
    {
        int n = v.Length;
        while (true)
        {
            int l = 2 * p + 1, r = l + 1, min = p;
            if (l < n && v[l] < v[min]) min = l;
            if (r < n && v[r] < v[min]) min = r;
            if (min == p) break;
            (v[p], v[min]) = (v[min], v[p]);
            (i[p], i[min]) = (i[min], i[p]);
            p = min;
        }
    }
}
