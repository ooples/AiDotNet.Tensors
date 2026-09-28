using System;
using System.Reflection;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// The driver reuses a destroyed context's handle VALUE for a later context, and the later context can be handed the
/// device addresses the destroyed one used. A GPU buffer finalized after its context died queues its device free
/// with that context's handle; the drain only checked the handle was "live", so once a new context held the same
/// handle it freed a live allocation of an unrelated engine. Measured only in the full test suite (hundreds of
/// contexts): captured training steps replayed garbage losses against a true 0.39.
/// </summary>
[Collection("DirectGpuSerial")]
public class ContextHandleReuseTests
{
    [SkippableFact]
    public void A_queued_free_from_a_dead_context_never_frees_memory_of_a_new_context_with_the_same_handle()
    {
        DirectGpuTensorEngine? gpu = null;
        try { gpu = new DirectGpuTensorEngine(); } catch (Exception) { }
        Skip.IfNot(gpu is not null && gpu.IsGpuAvailable && gpu.GetBackend() is CudaBackend, "CUDA backend did not resolve.");
        using (gpu)
        {
            var cuda = (CudaBackend)gpu!.GetBackend()!;
            var pattern = new float[4096];
            for (int i = 0; i < pattern.Length; i++) pattern[i] = i + 0.5f;
            var live = cuda.AllocateBuffer(pattern);
            var ctx = (IntPtr)live.GetType().GetField("_context", BindingFlags.NonPublic | BindingFlags.Instance)!.GetValue(live)!;
            var stream = (IntPtr)live.GetType().GetField("_asyncFreeStream", BindingFlags.NonPublic | BindingFlags.Instance)!.GetValue(live)!;
            long generation = CudaBackend.ContextGenerationOf(ctx);
            Assert.True(generation > 0, "the live context is not registered");

            // A free queued by a buffer of an EARLIER context that had this same handle value.
            CudaBackend.PendingFinalizerFrees.Enqueue((live.Handle, ctx, generation - 1, stream));
            CudaBackend.DrainPendingFinalizerFrees();
            cuda.Synchronize();

            // Were the memory freed, the pool would hand the same address to the next allocation of this size.
            var reuse = cuda.AllocateBuffer(new float[4096]);
            cuda.Synchronize();
            var read = cuda.DownloadBuffer(live);
            Assert.Equal(pattern, read);
            reuse.Dispose();
            live.Dispose();
        }
    }
}
