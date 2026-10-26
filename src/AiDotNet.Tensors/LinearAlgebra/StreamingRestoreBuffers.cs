// Copyright (c) AiDotNet. All rights reserved.

using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using System.Threading;
using AiDotNet.Tensors.Engines.Autodiff;

namespace AiDotNet.Tensors.LinearAlgebra;

/// <summary>
/// Recycles the arrays a streaming weight's restore allocates (#2322). Weight streaming rebuilds each weight
/// from its stored bytes for every forward and backward and drops it again afterwards, so a large model allocated a
/// fresh array per weight per pass: ~6 GB of the ~18 GB a 661M-parameter training step allocated.
/// </summary>
/// <remarks>
/// <para>
/// Only arrays this class handed out are recycled, never a caller's memory. A dropped array is not reusable at once:
/// a closure recorded on a gradient tape during the step may still read it in the backward pass. So it is
/// parked, and parked arrays join the pool only when no tape is active anywhere in the process: at once during
/// inference, and when the outermost tape of a training step is disposed. A restore takes an array of exactly its
/// length from the pool and refills it completely.
/// </para>
/// <para>
/// Pooled and parked arrays together hold at most <see cref="GpuOffloadOptions.StreamingRestoreBufferPoolBytes"/> per
/// element type. Arrays beyond it are left to the garbage collector as before, so the pool trades a bounded amount of
/// resident memory for allocation and never grows without limit. A cap of zero turns recycling off.
/// </para>
/// <para>
/// A reused array keeps its identity, so every cache keyed by array identity must treat the refill as a mutation.
/// The caller invalidates the derived-weight caches (<see cref="Engines.InferenceWeightCache.Invalidate"/>) and
/// the GPU upload epoch of the storage it wraps the array in.
/// </para>
/// </remarks>
internal static class StreamingRestoreBuffers<T>
{
    private static readonly object s_lock = new object();
    private static readonly Dictionary<int, Stack<T[]>> s_pool = new Dictionary<int, Stack<T[]>>();
    private static readonly List<T[]> s_parked = new List<T[]>();
    private static readonly ConditionalWeakTable<T[], object> s_issued = new ConditionalWeakTable<T[], object>();
    private static readonly object s_issuedMarker = new object();
    private static long s_pooledBytes;
    private static long s_parkedBytes;

    /// <summary>Only float and double weights are restored into recycled arrays.</summary>
    internal static bool Supported => typeof(T) == typeof(float) || typeof(T) == typeof(double);

    private static int ElementSize => typeof(T) == typeof(double) ? sizeof(double) : sizeof(float);

    /// <summary>Bytes currently pooled for reuse (diagnostics and tests).</summary>
    internal static long PooledBytes { get { lock (s_lock) return s_pooledBytes; } }

    /// <summary>Bytes parked until the next step boundary (diagnostics and tests).</summary>
    internal static long ParkedBytes { get { lock (s_lock) return s_parkedBytes; } }

    private static long s_recycledRents;

    /// <summary>Restores served from the pool rather than a new array (diagnostics and tests).</summary>
    internal static long RecycledRentCount => Interlocked.Read(ref s_recycledRents);

    /// <summary>
    /// An array of exactly <paramref name="length"/> elements for a restore to fill completely: a pooled one when
    /// available (<paramref name="recycled"/> is then true and its old contents are garbage), else a new one.
    /// </summary>
    internal static T[] Rent(int length, out bool recycled)
    {
        recycled = false;
        T[]? array = null;
        if (Supported && length > 0)
        {
            lock (s_lock)
            {
                if (s_pool.TryGetValue(length, out var stack) && stack.Count > 0)
                {
                    array = stack.Pop();
                    s_pooledBytes -= (long)length * ElementSize;
                }
            }
        }

        if (array is not null)
        {
            recycled = true;
            Interlocked.Increment(ref s_recycledRents);
            return array;
        }

        array = new T[length];
        if (Supported && length > 0) s_issued.Add(array, s_issuedMarker);
        return array;
    }

    /// <summary>
    /// Hands back an array a dropped weight no longer uses. Ignored unless <see cref="Rent"/> issued it. It becomes
    /// reusable at the next moment no gradient tape is active.
    /// </summary>
    internal static void Park(T[] array)
    {
        if (!Supported || array is null || array.Length == 0) return;
        if (!s_issued.TryGetValue(array, out _)) return;
        long cap = WeightRegistry.RestoreBufferPoolCapBytes();
        if (cap <= 0) return;
        long bytes = (long)array.Length * ElementSize;
        lock (s_lock)
        {
            // Parked arrays stay reachable until the step boundary, so they count against the same cap as the
            // pooled ones: together they are the memory recycling holds back from the garbage collector.
            if (s_parkedBytes + s_pooledBytes + bytes > cap) return;
            s_parked.Add(array);
            s_parkedBytes += bytes;
        }

        if (!DifferentiableOps.AnyTapeActive()) ReleaseParked();
    }

    /// <summary>
    /// Moves the parked arrays into the pool, up to the cap. Called once no gradient tape is active, when no closure
    /// recorded during the step can still read them.
    /// </summary>
    internal static void ReleaseParked()
    {
        if (!Supported) return;
        long cap = WeightRegistry.RestoreBufferPoolCapBytes();
        lock (s_lock)
        {
            if (s_parked.Count == 0) return;
            foreach (var array in s_parked)
            {
                long bytes = (long)array.Length * ElementSize;
                if (s_pooledBytes + bytes > cap) continue;
                if (!s_pool.TryGetValue(array.Length, out var stack))
                {
                    stack = new Stack<T[]>();
                    s_pool[array.Length] = stack;
                }
                stack.Push(array);
                s_pooledBytes += bytes;
            }
            s_parked.Clear();
            s_parkedBytes = 0;
        }
    }

    /// <summary>Drops every pooled and parked array (cap changes, tests).</summary>
    internal static void Clear()
    {
        lock (s_lock)
        {
            s_pool.Clear();
            s_parked.Clear();
            s_pooledBytes = 0;
            s_parkedBytes = 0;
        }
    }
}

/// <summary>Releases the parked restore arrays of every element type the restore path recycles.</summary>
internal static class StreamingRestoreBuffers
{
    internal static void ReleaseParked()
    {
        StreamingRestoreBuffers<float>.ReleaseParked();
        StreamingRestoreBuffers<double>.ReleaseParked();
    }

    internal static void Clear()
    {
        StreamingRestoreBuffers<float>.Clear();
        StreamingRestoreBuffers<double>.Clear();
    }
}
