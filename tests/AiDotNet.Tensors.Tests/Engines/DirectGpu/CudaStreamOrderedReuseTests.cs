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

    private static bool TryCuda(out DirectGpuTensorEngine? engine, out CudaBackend? cuda)
    {
        engine = null; cuda = null;
        try
        {
            var candidate = new DirectGpuTensorEngine();
            if (candidate.GetBackend() is CudaBackend backend && backend.IsAvailable)
            {
                engine = candidate; cuda = backend;
                return true;
            }
            candidate.Dispose();
        }
        catch (Exception) { }
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
