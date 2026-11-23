using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// A captured graph's private memory: what the capture allocates and releases while it is still capturing is reused by
/// its later allocations (PyTorch's graph-private pool), instead of every transient staying allocated for the graph's
/// lifetime. Replays must still compute exactly what the uncaptured body does.
/// </summary>
public class CaptureMemoryReuseTests
{
    private const int N = 1 << 20;   // 4 MB per scratch buffer
    private const int Iterations = 8;

    private static CudaBackend OpenBackendOrSkip()
    {
        Skip.IfNot(CudaBackend.IsCudaAvailable, "CUDA driver or NVRTC is unavailable.");
        return new CudaBackend();
    }

    // acc += 2 per iteration through a fresh scratch buffer: s = acc + 1 (s allocated zeroed, then written), acc = s + 1.
    private static void Body(CudaBackend backend, IGpuBuffer acc, IGpuBuffer ones, List<IntPtr> scratchAddresses,
        List<IGpuBuffer>? keep = null)
    {
        for (int i = 0; i < Iterations; i++)
        {
            var s = backend.AllocateBuffer(N);
            scratchAddresses.Add(s.Handle);
            backend.Add(acc, ones, s, N);
            backend.Add(s, ones, acc, N);
            if (keep is null) s.Dispose();
            else keep.Add(s);
        }
    }

    [SkippableFact]
    public void TransientsReleasedInsideACapture_AreReusedByItsLaterAllocations_AndReplaysStayExact()
    {
        using var backend = OpenBackendOrSkip();
        using var acc = backend.AllocateBuffer(new float[N]);
        using var ones = backend.AllocateBuffer(Enumerable.Repeat(1f, N).ToArray());

        var addresses = new List<IntPtr>();
        IntPtr graph = backend.CaptureGraph(() => Body(backend, acc, ones, addresses));
        Assert.NotEqual(IntPtr.Zero, graph);
        try
        {
            // Every scratch buffer after the first reuses the memory the previous one released: one block, not eight.
            Assert.Equal(Iterations, addresses.Count);
            Assert.Equal(1, addresses.Distinct().Count());

            // The capture executed nothing; each replay adds 2 per iteration, in order.
            backend.LaunchCapturedGraph(graph);
            backend.LaunchCapturedGraph(graph);
            var result = backend.DownloadBuffer(acc);
            Assert.All(new[] { result[0], result[N / 2], result[N - 1] }, v => Assert.Equal(2f * 2 * Iterations, v));
        }
        finally
        {
            backend.DestroyCapturedGraph(graph);
        }
    }

    [SkippableFact]
    public void TransientsStillAliveAtTheirNextAllocation_AreNotReused()
    {
        using var backend = OpenBackendOrSkip();
        using var acc = backend.AllocateBuffer(new float[N]);
        using var ones = backend.AllocateBuffer(Enumerable.Repeat(1f, N).ToArray());

        var addresses = new List<IntPtr>();
        var keep = new List<IGpuBuffer>();
        IntPtr graph = backend.CaptureGraph(() => Body(backend, acc, ones, addresses, keep));
        Assert.NotEqual(IntPtr.Zero, graph);
        try
        {
            // Nothing was released while capturing, so nothing may be handed out twice.
            Assert.Equal(Iterations, addresses.Distinct().Count());
            backend.LaunchCapturedGraph(graph);
            Assert.Equal(2f * Iterations, backend.DownloadBuffer(acc)[N - 1]);
        }
        finally
        {
            backend.DestroyCapturedGraph(graph);
            foreach (var b in keep) b.Dispose();
        }
    }
}
