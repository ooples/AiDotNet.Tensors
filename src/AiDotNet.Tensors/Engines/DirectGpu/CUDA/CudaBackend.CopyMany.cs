// Copyright (c) CrowdTrainer LLC. All rights reserved.
// CudaBackend.IBatchedCopyBackend: several device copies in one launch.
using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.IO;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation.Codegen;
using AiDotNet.Tensors.Engines.DevicePrimitives;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ir;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Kernels;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;
using AiDotNet.Tensors.Engines.Gpu;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA;

/// <summary>CudaBackend: several device copies in one launch (copy_many8).</summary>
public sealed partial class CudaBackend : IBatchedCopyBackend
{
    /// <inheritdoc />
    unsafe void IBatchedCopyBackend.CopyMany(IReadOnlyList<IGpuBuffer> sources, IReadOnlyList<IGpuBuffer> destinations, IReadOnlyList<int> lengths)
    {
        if (sources is null) throw new ArgumentNullException(nameof(sources));
        if (destinations is null) throw new ArgumentNullException(nameof(destinations));
        if (lengths is null) throw new ArgumentNullException(nameof(lengths));
        if (destinations.Count != sources.Count || lengths.Count != sources.Count)
            throw new ArgumentException("CopyMany needs one destination and one length per source.");
        for (int i = 0; i < lengths.Count; i++)
        {
            if (lengths[i] < 0) throw new ArgumentOutOfRangeException(nameof(lengths), "A copy length is negative.");
            // The kernel copies lengths[i] elements: both buffers must hold them.
            GpuBufferCapacity.Require(sources[i], lengths[i], $"{nameof(sources)}[{i}]");
            GpuBufferCapacity.Require(destinations[i], lengths[i], $"{nameof(destinations)}[{i}]");
        }
        RequireDisjointCopies(sources, destinations, lengths);
        if (!IsAvailable)
            throw new GpuCapabilityUnavailableException("CUDA backend is not available.");
        if (!_kernelCache.TryGetValue("copy_many8", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: copy_many8");
        const int Slots = 8, Threads = 256;
        var handles = stackalloc IntPtr[2 * Slots];
        var n = stackalloc int[Slots];
        var args = stackalloc void*[3 * Slots];
        for (int k = 0; k < 2 * Slots; k++) args[k] = &handles[k];
        for (int k = 0; k < Slots; k++) args[2 * Slots + k] = &n[k];
        using var _ = PushContext();
        for (int start = 0; start < sources.Count; start += Slots)
        {
            int count = Math.Min(Slots, sources.Count - start);
            int longest = FillCopyGroup(sources, destinations, lengths, start, count, Slots, handles, n);
            if (longest == 0) continue;
            uint blocks = (uint)Math.Min(1024, (longest + Threads - 1) / Threads);
            LaunchKernel2D(kernel, blocks, (uint)count, Threads, 1, args);
        }
    }

    // copy_many8 runs its copies concurrently: a destination that overlaps another copy's source or destination would
    // race with it, as would a copy onto a shifted view of its own source. Reads may overlap; a copy onto exactly its
    // own source is element-for-element and allowed.
    private static void RequireDisjointCopies(IReadOnlyList<IGpuBuffer> sources, IReadOnlyList<IGpuBuffer> destinations,
        IReadOnlyList<int> lengths)
    {
        for (int i = 0; i < lengths.Count; i++)
        {
            if (lengths[i] == 0) continue;
            if (destinations[i].Handle != sources[i].Handle
                && OverlapRanges(destinations[i], lengths[i], sources[i], lengths[i]))
                throw new ArgumentException($"CopyMany's copy {i} writes a shifted view of its own source.", nameof(destinations));
            for (int j = 0; j < lengths.Count; j++)
            {
                if (i == j || lengths[j] == 0) continue;
                if (OverlapRanges(destinations[i], lengths[i], destinations[j], lengths[j])
                    || OverlapRanges(destinations[i], lengths[i], sources[j], lengths[j]))
                    throw new ArgumentException($"CopyMany's copy {i} writes memory copy {j} reads or writes; the copies must not overlap.", nameof(destinations));
            }
        }
    }

    private static bool OverlapRanges(IGpuBuffer a, int aCount, IGpuBuffer b, int bCount)
    {
        ulong a0 = (ulong)a.Handle.ToInt64(), b0 = (ulong)b.Handle.ToInt64();
        return a0 < b0 + (ulong)bCount * sizeof(float) && b0 < a0 + (ulong)aCount * sizeof(float);
    }

    // Loads one launch's slots: its copies' handles and lengths, unused slots repeating the last copy with length 0.
    // Returns the longest copy, which sizes the launch. CopyMany has checked every buffer against its length.
    private static unsafe int FillCopyGroup(IReadOnlyList<IGpuBuffer> sources, IReadOnlyList<IGpuBuffer> destinations,
        IReadOnlyList<int> lengths, int start, int count, int slots, IntPtr* handles, int* n)
    {
        int longest = 0;
        for (int k = 0; k < slots; k++)
        {
            int i = start + Math.Min(k, count - 1);
            handles[k] = sources[i].Handle;
            handles[slots + k] = destinations[i].Handle;
            n[k] = k < count ? lengths[i] : 0;   // unused slots copy nothing
            longest = Math.Max(longest, n[k]);
        }
        return longest;
    }
}
