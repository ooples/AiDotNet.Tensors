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
    public IntPtr CaptureGraph(Action launch)
    {
        if (!IsAvailable || launch is null) return IntPtr.Zero;
        using var _ = PushContext();

        var rc = CudaNativeBindings.cuStreamBeginCapture(_stream, CudaNativeBindings.CU_STREAM_CAPTURE_MODE_THREAD_LOCAL);
        if (rc != CudaResult.Success) return IntPtr.Zero;

        // If the action throws, abort the capture so the stream is not left in
        // capturing state (which would break every subsequent op on it).
        try
        {
            launch();
        }
        catch
        {
            CudaNativeBindings.cuStreamEndCapture(_stream, out var abortedGraph);
            if (abortedGraph != IntPtr.Zero) CudaNativeBindings.cuGraphDestroy(abortedGraph);
            return IntPtr.Zero;
        }

        rc = CudaNativeBindings.cuStreamEndCapture(_stream, out var graph);
        if (rc != CudaResult.Success || graph == IntPtr.Zero) return IntPtr.Zero;

        rc = CudaNativeBindings.cuGraphInstantiate(out var graphExec, graph, 0UL);
        CudaNativeBindings.cuGraphDestroy(graph); // exec holds its own copy; the template graph is no longer needed
        if (rc != CudaResult.Success) return IntPtr.Zero;

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

    /// <summary>True while this backend's stream is mid-capture (diagnostics).</summary>
    public bool IsStreamCapturing()
    {
        if (!IsAvailable) return false;
        using var _ = PushContext();
        if (CudaNativeBindings.cuStreamIsCapturing(_stream, out int status) != CudaResult.Success)
            return false;
        return status != 0;
    }
}
