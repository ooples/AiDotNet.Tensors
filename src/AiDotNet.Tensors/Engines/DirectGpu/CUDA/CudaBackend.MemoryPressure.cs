// Copyright (c) CrowdTrainer LLC. All rights reserved.
namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA;

// cuMemGetInfo, the backed-off sweep and the deferred-free drain live in CudaBackend.cs.
public sealed partial class CudaBackend : IDeviceMemoryPressure
{
    bool IDeviceMemoryPressure.TryGetDeviceMemory(out ulong free, out ulong total) => TryGetDeviceMemory(out free, out total);

    void IDeviceMemoryPressure.ReclaimUnderPressure() => ReclaimUnderPressure();

    void IDeviceMemoryPressure.ReclaimNow() => ReclaimNow();

    int IDeviceMemoryPressure.ReleaseCachedBuffers()
    {
        if (!IsAvailable || IsCapturingOnThisThread) return 0;
        // Finalizer-queued and event-deferred frees whose work has completed go back first, so the drains below free
        // them too.
        if (CanRunDeferredFrees()) DrainPendingFinalizerFrees();
        DrainDeferredFrees(blockOldest: false);
        int drained = _bufferPool.DrainAll();
        if (_asyncMemPool != IntPtr.Zero)
        {
            // Stream-ordered frees stay reserved in the device's default pool (its release threshold is maxed so
            // churn reuses them): torch.cuda.empty_cache's other half. A cuMemFreeAsync reaches the pool once this
            // backend's streams pass it, so wait on those streams (never the shared primary context, which another
            // thread may be capturing on), then trim. The trim releases only memory no allocation is using.
            SynchronizeOwnStreams("stream sync (release cached device memory)");
            using (PushContext())
            {
                var status = CuBlasNative.cuMemPoolTrimTo(_asyncMemPool, 0); // lgtm[cs/call-to-unmanaged-code] CUDA memory pools have no managed API.
                if (status != CudaResult.Success)
                    System.Diagnostics.Trace.TraceWarning(
                        $"[CudaBackend] cuMemPoolTrimTo failed while releasing cached device memory: {status}.");
            }
        }
        return drained;
    }

    /// <remarks>Allocation drains the buffer pool and retries once on out-of-memory.</remarks>
    bool IDeviceMemoryPressure.ReclaimsOnAllocationFailure => true;
}