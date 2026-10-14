using System.Linq;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// CUDA's overlapped upload copies host data on a separate copy stream into a staging buffer and then device-to-device
/// into the destination on the compute stream. Everything queued on the compute stream afterwards must see exactly the
/// data of the most recent upload -- including back-to-back uploads that reuse the two staging slots.
/// </summary>
public class CudaOverlappedUploadTests
{
    [SkippableFact]
    public void EveryUploadIsWhatTheComputeStreamSeesNext()
    {
        using var backend = new CudaBackend();
        Skip.IfNot(backend.IsAvailable, "CUDA is not available.");
        var overlapped = (IOverlappedUploadBackend)backend;
        const int n = 65536;
        using var destination = backend.AllocateBuffer(new float[n]);
        using var copy = backend.AllocateBuffer(new float[n]);
        for (int round = 0; round < 7; round++)
        {
            var data = Enumerable.Range(0, n).Select(i => (float)(round * 1000 + i % 997)).ToArray();
            Assert.True(overlapped.TryUploadOverlapped(data, destination));
            // A compute-stream op queued right after the upload must read the new contents.
            backend.Copy(destination, copy, n);
            var seen = backend.DownloadBuffer(copy);
            for (int i = 0; i < n; i++)
                if (seen[i] != data[i])
                    Assert.Fail($"round {round}, element {i}: {seen[i]} != {data[i]}");
        }
    }

    /// <summary>
    /// A slot's staging buffer is allocated (and zero-filled) in order on the compute stream. With the compute stream
    /// busy, the copy stream's first upload into it ran before that fill, which then overwrote the data: the parity
    /// Transformer's first graph replay trained on partly zeroed labels.
    /// </summary>
    [SkippableFact]
    public void FirstUploadIsNotLostBehindABusyComputeStream()
    {
        using var backend = new CudaBackend();
        Skip.IfNot(backend.IsAvailable, "CUDA is not available.");
        var overlapped = (IOverlappedUploadBackend)backend;
        const int m = 1024, n = 4096;
        using var a = backend.AllocateBuffer(new float[m * m]);
        using var b = backend.AllocateBuffer(new float[m * m]);
        using var c = backend.AllocateBuffer(new float[m * m]);
        using var destination = backend.AllocateBuffer(new float[n]);
        using var copy = backend.AllocateBuffer(new float[n]);
        backend.Synchronize();
        // Keep the compute stream busy for many milliseconds, so the staging allocation queued next waits behind it.
        for (int i = 0; i < 40; i++) backend.Gemm(a, b, c, m, m, m);
        var data = Enumerable.Range(0, n).Select(i => 1f + i).ToArray();
        Assert.True(overlapped.TryUploadOverlapped(data, destination));
        backend.Copy(destination, copy, n);
        var seen = backend.DownloadBuffer(copy);
        for (int i = 0; i < n; i++)
            if (seen[i] != data[i])
                Assert.Fail($"element {i}: {seen[i]} != {data[i]}");
    }

    [SkippableFact]
    public void UploadsQueuedWithoutWaitingKeepTheirOrder()
    {
        using var backend = new CudaBackend();
        Skip.IfNot(backend.IsAvailable, "CUDA is not available.");
        var overlapped = (IOverlappedUploadBackend)backend;
        const int n = 4096;
        using var destination = backend.AllocateBuffer(new float[n]);
        var snapshots = Enumerable.Range(0, 5).Select(_ => backend.AllocateBuffer(new float[n])).ToArray();
        try
        {
            // Five uploads with a compute-stream copy after each, nothing read back until the end: each copy must hold
            // its own round's data although the host refilled the two slots meanwhile.
            for (int round = 0; round < 5; round++)
            {
                var data = Enumerable.Range(0, n).Select(i => (float)(round + 1) * 10000 + i).ToArray();
                Assert.True(overlapped.TryUploadOverlapped(data, destination));
                backend.Copy(destination, snapshots[round], n);
            }
            for (int round = 0; round < 5; round++)
            {
                var seen = backend.DownloadBuffer(snapshots[round]);
                for (int i = 0; i < n; i++)
                    if (seen[i] != (round + 1) * 10000 + i)
                        Assert.Fail($"round {round}, element {i}: {seen[i]}");
            }
        }
        finally
        {
            foreach (var s in snapshots) s.Dispose();
        }
    }
}
