using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// The stream-ordered CUDA allocator reuses disposed buffers from the host pool instead of paying
/// cuMemAllocAsync + cuMemFreeAsync per temporary (three driver calls with the zero-fill, ~13 us — as much as the
/// kernel launch it serves). Reuse must keep the fresh-allocation contract (zero-initialised) and stay ordered on
/// the one stream, so a queued kernel that still reads the old contents finishes before the new owner writes.
/// </summary>
[Collection("DirectGpuSerial")]
public class CudaStreamOrderedReuseTests
{
    private readonly ITestOutputHelper _out;
    public CudaStreamOrderedReuseTests(ITestOutputHelper output) => _out = output;

    /// <summary>
    /// Skips only when CUDA is confirmed absent (no driver or NVRTC). On a CUDA machine the engine is
    /// constructed outside any catch, so an initialization failure fails the test instead of reading
    /// as a skip. An engine that selects a different backend is a selection policy, not a failure.
    /// </summary>
    private static bool TryCuda(out DirectGpuTensorEngine? engine, out CudaBackend? cuda)
    {
        engine = null; cuda = null;
        if (!CudaBackend.IsCudaAvailable) return false;

        var candidate = new DirectGpuTensorEngine();
        if (candidate.GetBackend() is CudaBackend backend && backend.IsAvailable)
        {
            engine = candidate; cuda = backend;
            return true;
        }
        candidate.Dispose();
        return false;
    }

    [SkippableFact]
    public void A_disposed_buffer_is_reused_and_comes_back_zeroed()
    {
        Skip.IfNot(TryCuda(out var engine, out var cuda), "CUDA backend not available.");
        using (engine!)
        {
            var first = cuda!.AllocateBuffer(37);
            cuda.Fill(first, 7f, 37);
            var address = first.Handle;
            first.Dispose();

            using var second = cuda.AllocateBuffer(37);
            _out.WriteLine($"first=0x{(long)address:X} second=0x{(long)second.Handle:X}");
            Assert.Equal(address, second.Handle);            // the pooled allocation was reused
            Assert.Equal(37, second.Size);
            Assert.All(cuda.DownloadBuffer(second), v => Assert.Equal(0f, v));
        }
    }

    [SkippableFact]
    public void An_uninitialized_allocation_skips_the_fill()
    {
        // The whole point: an output the next kernel overwrites must not pay a memset (about a launch on WDDM).
        // Reusing the previous owner's buffer makes that observable — its contents survive.
        Skip.IfNot(TryCuda(out var engine, out var cuda), "CUDA backend not available.");
        using (engine!)
        {
            var first = cuda!.AllocateBuffer(53);
            cuda.Fill(first, 7f, 53);
            var address = first.Handle;
            first.Dispose();

            using var second = cuda.AllocateBufferUninitialized(53);
            Assert.Equal(address, second.Handle);
            Assert.All(cuda.DownloadBuffer(second), v => Assert.Equal(7f, v));
        }
    }

    /// <summary>
    /// Uploads are staged through page-locked memory instead of blocking the host on a stream sync. The managed
    /// source must be free the moment AllocateBuffer returns, and every consumer must still see the uploaded values.
    /// </summary>
    [SkippableFact]
    public void A_staged_upload_owns_its_data_and_is_seen_by_every_consumer()
    {
        Skip.IfNot(TryCuda(out var engine, out var cuda), "CUDA backend not available.");
        using (engine!)
        {
            const int n = 4096;
            var source = new float[n];
            using var doubled = cuda!.AllocateBuffer(n);
            for (int round = 0; round < 40; round++)
            {
                for (int i = 0; i < n; i++) source[i] = round * 1000 + i;
                using var uploaded = cuda.AllocateBuffer(source);
                for (int i = 0; i < source.Length; i++) source[i] = -9f;                          // the host copy is free immediately
                cuda.Add(uploaded, uploaded, doubled, n);         // same-stream consumer
                var d = cuda.DownloadBuffer(doubled);
                var u = cuda.DownloadBuffer(uploaded);
                for (int i = 0; i < n; i += 97)
                {
                    Assert.Equal(round * 1000 + i, u[i]);
                    Assert.Equal(2f * (round * 1000 + i), d[i]);
                }
            }
        }
    }

    [SkippableFact]
    public void A_staged_upload_is_ordered_before_work_on_another_stream()
    {
        Skip.IfNot(TryCuda(out var engine, out var cuda), "CUDA backend not available.");
        using (engine!)
        {
            const int n = 1 << 18;
            var source = new float[n];
            for (int i = 0; i < n; i++) source[i] = i;
            var other = cuda!.CreateStream(AiDotNet.Tensors.Engines.Gpu.GpuStreamType.Compute);
            try
            {
                using var target = cuda.AllocateBuffer(n);
                for (int round = 0; round < 10; round++)
                {
                    using var uploaded = cuda.AllocateBuffer(source);
                    cuda.CopyBufferAsync(uploaded, target, n, other);   // consumer on a DIFFERENT stream
                    cuda.SynchronizeStream(other);
                    var copied = cuda.DownloadBuffer(target);
                    for (int i = 0; i < n; i += 1009) Assert.Equal(i, copied[i]);
                }
            }
            finally { other.Dispose(); }
        }
    }

    [SkippableFact]
    public void Staged_uploads_beyond_the_in_flight_cap_still_land_correctly()
    {
        // 96 uploads of 1 MB exceed the 64 MB in-flight cap: the oldest slots are waited on and recycled.
        Skip.IfNot(TryCuda(out var engine, out var cuda), "CUDA backend not available.");
        using (engine!)
        {
            const int n = 1 << 18;
            var buffers = new List<AiDotNet.Tensors.Engines.DirectGpu.IGpuBuffer>();
            try
            {
                for (int k = 0; k < 96; k++)
                {
                    var data = new float[n];
                    for (int i = 0; i < data.Length; i++) data[i] = k;
                    buffers.Add(cuda!.AllocateBuffer(data));
                }
                for (int k = 0; k < buffers.Count; k += 7)
                    Assert.All(cuda!.DownloadBuffer(buffers[k]), v => Assert.Equal((float)k, v));
            }
            finally { foreach (var b in buffers) b.Dispose(); }
        }
    }

    /// <summary>
    /// The in-place uploads that refresh a captured training step's input, labels and token indices are staged through
    /// page-locked memory too (a copy from the pageable array made the host wait for the previous step on the device).
    /// Rewriting ONE buffer every round, with a consumer queued between rewrites, checks both halves of the contract:
    /// each consumer sees the upload queued before it (stream order), and the host array is reusable on return.
    /// </summary>
    [SkippableFact]
    public void In_place_uploads_into_one_buffer_are_stream_ordered_and_own_their_data()
    {
        Skip.IfNot(TryCuda(out var engine, out var cuda), "CUDA backend not available.");
        using (engine!)
        {
            const int n = 1 << 16;
            const int rounds = 24;
            var source = new float[n];
            var indices = new int[n];
            using var target = cuda!.AllocateBuffer(n);
            using var intTarget = cuda.AllocateIntBuffer(n);
            var sinks = new List<AiDotNet.Tensors.Engines.DirectGpu.IGpuBuffer>();
            try
            {
                for (int round = 0; round < rounds; round++)
                {
                    for (int i = 0; i < n; i++) { source[i] = round * 10 + (i & 7); indices[i] = round * 3 + i; }
                    cuda.UploadBufferInPlace(source, target);
                    cuda.UploadIntBufferInPlace(indices, intTarget);
                    for (int i = 0; i < n; i++) { source[i] = -1f; indices[i] = -1; }   // the host copies are free on return
                    var sink = cuda.AllocateBuffer(n);
                    cuda.Add(target, target, sink, n);   // queued after this round's upload, before the next one
                    sinks.Add(sink);
                }
                for (int round = 0; round < rounds; round++)
                {
                    var doubled = cuda.DownloadBuffer(sinks[round]);
                    for (int i = 0; i < n; i += 61) Assert.Equal(2f * (round * 10 + (i & 7)), doubled[i]);
                }
                var lastInts = cuda.DownloadBuffer(intTarget);
                for (int i = 0; i < n; i += 61)
                    Assert.Equal((rounds - 1) * 3 + i, System.BitConverter.ToInt32(System.BitConverter.GetBytes(lastInts[i]), 0));
            }
            finally { foreach (var sink in sinks) sink.Dispose(); }
        }
    }

    [SkippableFact]
    public void Reuse_is_ordered_after_queued_work_that_still_reads_the_old_contents()
    {
        // A kernel queued BEFORE the dispose reads the old buffer; the new owner then overwrites it. Stream order must
        // make the queued read see the old values, and the new owner see its own.
        Skip.IfNot(TryCuda(out var engine, out var cuda), "CUDA backend not available.");
        using (engine!)
        {
            const int n = 1 << 16;
            using var sink = cuda!.AllocateBuffer(n);
            for (int round = 0; round < 50; round++)
            {
                var old = cuda.AllocateBuffer(n);
                cuda.Fill(old, round + 1f, n);
                cuda.Add(old, old, sink, n);                    // queued read of the old contents
                old.Dispose();                                  // returns to the pool while the Add may still be queued
                using var reused = cuda.AllocateBuffer(n);
                cuda.Fill(reused, -1f, n);                      // the new owner's write
                var result = cuda.DownloadBuffer(sink);
                Assert.All(result, v => Assert.Equal(2f * (round + 1), v));
                Assert.All(cuda.DownloadBuffer(reused), v => Assert.Equal(-1f, v));
            }
        }
    }
}
