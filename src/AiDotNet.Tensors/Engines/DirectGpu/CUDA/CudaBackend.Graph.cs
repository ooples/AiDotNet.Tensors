using System;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA;

/// <summary>
/// CUDA-graph capture/replay (training-step Phase 3). Collapses a repeated,
/// pure-GPU kernel sequence into a single <c>cuGraphLaunch</c>, removing the
/// per-kernel CPU launch overhead that dominates small-model GPU training
/// (the Phase-0 probe measured ~5.5× launch-overhead removal on this stream).
/// </summary>
/// <remarks>
/// Contract for a capturable region (CUDA forbids host interaction during capture):
/// <list type="bullet">
/// <item>PURE GPU — no host reads / <c>cuMemcpyDtoH</c> / synchronizes inside the captured action.</item>
/// <item>STABLE device buffers — the captured kernels' pointer args are frozen at
/// capture time; replay re-runs the same kernels on the same buffers. To feed new
/// data each step, overwrite the SAME input buffer's CONTENTS (<c>cuMemcpyHtoD</c>
/// into the stable pointer) before <see cref="LaunchCapturedGraph"/>; the pointer is
/// unchanged so the graph stays valid.</item>
/// <item>Scalar-by-value kernel args are also frozen at capture (e.g. Adam's bias-
/// correction <c>step</c>). Sequences whose scalar args change every step
/// (Adam/AdamW/AdaMax/Nadam/LAMB/AMSGrad) need re-capture or <c>cuGraphExecUpdate</c>;
/// the step-free optimizers (SGD/Momentum/NAG/LARS/RMSprop/Adagrad/AdaDelta/Lion/FTRL)
/// replay bit-exact.</item>
/// </list>
/// The GPU-resident optimizer (in-place weight + moment update, no gradient download)
/// is what makes the optimizer portion of a step host-read-free and thus capturable.
/// </remarks>
public sealed partial class CudaBackend
{
    /// <summary>
    /// This backend's CUDA context handle. Exposed so the GPU offload allocator
    /// can SHARE it (allocate pinned moment/weight buffers in the same context
    /// as the compute kernels), avoiding cross-context access in the
    /// GPU-resident optimizer step.
    /// </summary>
    internal IntPtr CudaContextHandle => _cudaContext;

    /// <summary>
    /// Records the GPU kernel launches issued by <paramref name="launch"/> on this
    /// backend's stream into a CUDA graph and returns an instantiated, replayable
    /// graph-exec handle (or <see cref="IntPtr.Zero"/> on failure / unavailable GPU).
    /// The caller owns the handle and must free it with
    /// <see cref="DestroyCapturedGraph"/>.
    /// </summary>
    private static void GcDiag(string s)
    {
        if (System.Environment.GetEnvironmentVariable("AIDOTNET_GRAPH_CAPTURE_DEBUG") != "1") return;
        try { System.IO.File.AppendAllText(System.IO.Path.Combine(System.IO.Path.GetTempPath(),
            "aidotnet_graphcapture_diag.txt"), "[GRAPH-CAPTURE] " + s + System.Environment.NewLine); } catch { }
    }

    public IntPtr CaptureGraph(Action launch)
    {
        if (!IsAvailable || launch is null) return IntPtr.Zero;
        using var _ = PushContext();

        var rc = CudaNativeBindings.cuStreamBeginCapture(_stream, CudaNativeBindings.CU_STREAM_CAPTURE_MODE_THREAD_LOCAL);
        if (rc != CudaResult.Success) { GcDiag($"beginCapture FAILED rc={rc}"); return IntPtr.Zero; }

        // If the action throws, abort the capture so the stream is not left in
        // capturing state (which would break every subsequent op on it).
        try
        {
            launch();
        }
        catch (Exception ex)
        {
            { var st = ex.StackTrace?.Replace("\r", "").Replace("\n", " >> ") ?? ""; GcDiag($"launch() THREW {ex.GetType().Name}: {ex.Message} | {st.Substring(0, System.Math.Min(2500, st.Length))}"); }
            CudaNativeBindings.cuStreamEndCapture(_stream, out var abortedGraph);
            if (abortedGraph != IntPtr.Zero) CudaNativeBindings.cuGraphDestroy(abortedGraph);
            return IntPtr.Zero;
        }

        rc = CudaNativeBindings.cuStreamEndCapture(_stream, out var graph);
        if (rc != CudaResult.Success || graph == IntPtr.Zero) { GcDiag($"endCapture FAILED rc={rc} graph={(graph != IntPtr.Zero)} (a non-capturable op — sync HtoD/DtoH or cuMemAlloc — was issued during launch)"); return IntPtr.Zero; }

        rc = CudaNativeBindings.cuGraphInstantiate(out var graphExec, graph, 0UL);
        CudaNativeBindings.cuGraphDestroy(graph); // exec holds its own copy; the template graph is no longer needed
        if (rc != CudaResult.Success) { GcDiag($"instantiate FAILED rc={rc}"); return IntPtr.Zero; }

        GcDiag("capture+instantiate SUCCEEDED");
        return graphExec;
    }

    /// <summary>
    /// Replays a previously <see cref="CaptureGraph"/>'d sequence with a single
    /// <c>cuGraphLaunch</c> on this backend's stream. Refresh any input-buffer
    /// CONTENTS (same pointers) before calling to feed new per-step data.
    /// </summary>
    public void LaunchCapturedGraph(IntPtr graphExec)
    {
        if (graphExec == IntPtr.Zero) return;
        using var _ = PushContext();
        CuBlasNative.CheckCudaResult(
            CudaNativeBindings.cuGraphLaunch(graphExec, _stream), "cuGraphLaunch");
    }

    /// <summary>
    /// Re-records <paramref name="relaunch"/> into a fresh graph and updates the
    /// EXISTING <paramref name="graphExec"/> in place (cuGraphExecUpdate) instead of
    /// re-instantiating. This extends graph replay to optimizers whose kernel SCALAR
    /// args change every step — notably the Adam family's bias-correction <c>step</c>
    /// (and AdaMax/Nadam/AMSGrad): topology is identical step-to-step, only the
    /// scalar differs, so the update is cheap and the handle stays valid for
    /// <see cref="LaunchCapturedGraph"/>. Returns true when the in-place update
    /// succeeded; false means the topology diverged and the caller must re-capture
    /// via <see cref="CaptureGraph"/> + <see cref="DestroyCapturedGraph"/>.
    /// </summary>
    public bool TryUpdateCapturedGraph(IntPtr graphExec, Action relaunch)
    {
        if (graphExec == IntPtr.Zero || relaunch is null || !IsAvailable) return false;
        using var _ = PushContext();

        if (CudaNativeBindings.cuStreamBeginCapture(_stream, CudaNativeBindings.CU_STREAM_CAPTURE_MODE_THREAD_LOCAL) != CudaResult.Success)
            return false;
        try { relaunch(); }
        catch { CudaNativeBindings.cuStreamEndCapture(_stream, out var g); if (g != IntPtr.Zero) CudaNativeBindings.cuGraphDestroy(g); return false; }

        if (CudaNativeBindings.cuStreamEndCapture(_stream, out var newGraph) != CudaResult.Success || newGraph == IntPtr.Zero)
            return false;

        var info = default(CudaNativeBindings.CUgraphExecUpdateResultInfo);
        var rc = CudaNativeBindings.cuGraphExecUpdate(graphExec, newGraph, ref info);
        CudaNativeBindings.cuGraphDestroy(newGraph);
        return rc == CudaResult.Success && info.result == 0; // 0 == CU_GRAPH_EXEC_UPDATE_SUCCESS
    }

    /// <summary>Frees a captured graph-exec handle. Safe to call with <see cref="IntPtr.Zero"/>.</summary>
    public void DestroyCapturedGraph(IntPtr graphExec)
    {
        if (graphExec == IntPtr.Zero) return;
        using var _ = PushContext();
        CudaNativeBindings.cuGraphExecDestroy(graphExec);
    }

    /// <summary>
    /// Host→device copy of raw bytes into a byte/blob GPU buffer (e.g. the int8
    /// quantized moments and double per-block scales of the 8-bit optimizer).
    /// </summary>
    public unsafe void UploadBytes(IGpuBuffer buffer, byte[] data)
    {
        if (buffer is null) throw new ArgumentNullException(nameof(buffer));
        if (data is null) throw new ArgumentNullException(nameof(data));
        using var _ = PushContext();
        fixed (byte* src = data)
        {
            CuBlasNative.CheckCudaResult(
                CuBlasNative.cuMemcpyHtoD(buffer.Handle, (IntPtr)src, (ulong)data.Length), "cuMemcpyHtoD(bytes)");
        }
    }

    /// <summary>Device→host copy of raw bytes (synchronizes the compute stream first).</summary>
    public unsafe byte[] DownloadBytes(IGpuBuffer buffer, int byteCount)
    {
        if (buffer is null) throw new ArgumentNullException(nameof(buffer));
        using var _ = PushContext();
        CuBlasNative.CheckCudaResult(CudaNativeBindings.cuStreamSynchronize(_stream), "cuStreamSynchronize(downloadBytes)");
        var dst = new byte[byteCount];
        fixed (byte* d = dst)
        {
            CuBlasNative.CheckCudaResult(
                CuBlasNative.cuMemcpyDtoH((IntPtr)d, buffer.Handle, (ulong)byteCount), "cuMemcpyDtoH(bytes)");
        }
        return dst;
    }

    /// <summary>
    /// Sets <paramref name="byteCount"/> bytes of <paramref name="buffer"/> to
    /// <paramref name="value"/> on the compute stream (cuMemsetD8Async). Used by the
    /// compiled-training-step graph capture to zero accumulating gradient buffers AS
    /// A GPU OP inside the captured sequence — a host-side Array.Clear would not be
    /// replayed by cuGraphLaunch, so the grads would accumulate across replays.
    /// </summary>
    public void MemsetBuffer(IGpuBuffer buffer, byte value, long byteCount)
    {
        if (buffer is null || byteCount <= 0) return;
        using var _ = PushContext();
        CuBlasNative.CheckCudaResult(
            CudaNativeBindings.cuMemsetD8Async(buffer.Handle, value, (ulong)byteCount, _stream), "cuMemsetD8Async");
    }

    /// <summary>
    /// Async device→device copy on the compute stream (cuMemcpyDtoDAsync). The
    /// synchronous CopyBuffer/cuMemcpyDtoD would abort stream capture, so the
    /// graph-captured step re-seeds the loss gradient through this instead.
    /// </summary>
    public void CopyBufferDtoD(IGpuBuffer source, IGpuBuffer destination, long byteCount)
    {
        if (source is null || destination is null || byteCount <= 0) return;
        using var _ = PushContext();
        CuBlasNative.CheckCudaResult(
            CudaNativeBindings.cuMemcpyDtoDAsync(destination.Handle, source.Handle, (ulong)byteCount, _stream), "cuMemcpyDtoDAsync(graph)");
    }

    /// <summary>True while this backend's stream is mid-capture (diagnostics).</summary>
    public bool IsStreamCapturing()
    {
        if (!IsAvailable) return false;
        using var _ = PushContext();
        if (CudaNativeBindings.cuStreamIsCapturing(_stream, out int status) != CudaResult.Success)
            return false;
        return status != 0;
    }

    /// <summary>Raw capture status (0=NONE, 1=ACTIVE, 2=INVALIDATED, -1=unknown). Used to pinpoint the FIRST
    /// op that issues a non-capturable operation (which silently invalidates the whole capture).</summary>
    public int StreamCaptureStatusRaw()
    {
        if (!IsAvailable) return -1;
        using var _ = PushContext();
        if (CudaNativeBindings.cuStreamIsCapturing(_stream, out int status) != CudaResult.Success)
            return -1;
        return status;
    }
}
