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
            if (lengths[i] < 0) throw new ArgumentOutOfRangeException(nameof(lengths), "A copy length is negative.");
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
            int count = Math.Min(Slots, sources.Count - start), longest = 0;
            for (int k = 0; k < Slots; k++)
            {
                int i = start + Math.Min(k, count - 1);
                handles[k] = sources[i].Handle;
                handles[Slots + k] = destinations[i].Handle;
                n[k] = k < count ? lengths[i] : 0;   // unused slots copy nothing
                longest = Math.Max(longest, n[k]);
            }
            if (longest == 0) continue;
            uint blocks = (uint)Math.Min(1024, (longest + Threads - 1) / Threads);
            LaunchKernel2D(kernel, blocks, (uint)count, Threads, 1, args);
        }
    }
}
