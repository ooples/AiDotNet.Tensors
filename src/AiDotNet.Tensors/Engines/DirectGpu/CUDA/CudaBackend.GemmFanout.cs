using System;
using System.Threading;
using AiDotNet.Tensors.Engines.Gpu;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA;

/// <summary>
/// #1315: the <c>*Fanout</c> GEMMs run each slice on the stream <see cref="GpuStreamScheduler"/> leased it. Every slice
/// used to run on the backend's own stream: nothing bound cuBLAS (or the generated kernels) to the lease, so the slices
/// ran one after another and the completion events the scheduler recorded on the leases covered no work.
/// </summary>
public sealed partial class CudaBackend
{
    // The leased stream the calling thread's GEMMs run on while it issues a fan-out slice; zero otherwise. Per thread,
    // so another thread's GEMMs on this backend keep the backend's stream.
    private readonly ThreadLocal<IntPtr> _gemmFanoutStream = new();

    /// <summary>The stream a GEMM issued now runs on: a fan-out slice's lease, or zero for the backend's stream.</summary>
    private IntPtr GemmLaunchStream => _gemmFanoutStream.Value;

    /// <summary>Tests: called with the stream each fan-out slice's GEMM was bound to.</summary>
    internal Action<IntPtr>? FanoutSliceStreamObserver { get; set; }

    /// <summary>
    /// Starts a fan-out: the slices issued through the returned scope's <see cref="GemmFanout.Bind"/> run on their
    /// leased streams, after the work already queued on the backend's stream. Disposing it puts this thread's GEMMs
    /// back on the backend's stream.
    /// </summary>
    private GemmFanout BeginGemmFanout() => new(this);

    /// <summary>One fan-out's ordering event and the binding of this thread's GEMMs to each slice's stream.</summary>
    private sealed class GemmFanout : IDisposable
    {
        private readonly CudaBackend _backend;
        private IntPtr _ready;

        internal GemmFanout(CudaBackend backend)
        {
            _backend = backend;
            // While this thread captures a CUDA graph its GEMMs go to the capture stream. Slices on other streams would
            // join the capture, and the fan-out's host-side wait on their events is not allowed inside one, so a
            // captured fan-out keeps running its slices in order on the capture stream.
            if (backend._stream != backend._mainStream) return;
            backend.EnsureContextCurrent();
            CuBlasNative.CheckCudaResult(CudaNativeBindings.cuEventCreate(out _ready, CuEventDisableTiming), "cuEventCreate(fanout)");
            try
            {
                // The slices read what earlier work on the backend's stream wrote.
                CuBlasNative.CheckCudaResult(CudaNativeBindings.cuEventRecord(_ready, backend._stream), "cuEventRecord(fanout)");
            }
            catch
            {
                CudaNativeBindings.cuEventDestroy(_ready);
                _ready = IntPtr.Zero;
                throw;
            }
        }

        /// <summary>Binds this thread's next GEMM to <paramref name="stream"/>, ordered after the backend's prior work.</summary>
        internal void Bind(IGpuStream stream)
        {
            IntPtr handle = stream.Handle;
            if (_ready == IntPtr.Zero || handle == IntPtr.Zero || handle == _backend._stream)
            {
                _backend._gemmFanoutStream.Value = IntPtr.Zero;
                _backend.FanoutSliceStreamObserver?.Invoke(_backend._stream);
                return;
            }
            CuBlasNative.CheckCudaResult(CudaNativeBindings.cuStreamWaitEvent(handle, _ready, 0), "cuStreamWaitEvent(fanout)");
            _backend._gemmFanoutStream.Value = handle;
            _backend.FanoutSliceStreamObserver?.Invoke(handle);
        }

        public void Dispose()
        {
            _backend._gemmFanoutStream.Value = IntPtr.Zero;
            // Put this thread's cuBLAS handle back on the backend's stream now, with its workspace re-attached, rather
            // than leaving it on the last lease until the next GEMM.
            if (_backend._threadCublas is { IsValueCreated: true } && _backend.IsCublasAvailable) _ = _backend._cublasHandle;
            if (_ready != IntPtr.Zero)
            {
                // Destroying an event that streams still wait on is allowed: the driver releases it once they are done.
                CudaNativeBindings.cuEventDestroy(_ready);
                _ready = IntPtr.Zero;
            }
        }
    }

    // CU_EVENT_DISABLE_TIMING: an ordering-only event is cheaper to record and wait on.
    private const uint CuEventDisableTiming = 0x2;
}
