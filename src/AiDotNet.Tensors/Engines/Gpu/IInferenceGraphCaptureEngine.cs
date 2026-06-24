// Copyright (c) AiDotNet. All rights reserved.

using System;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Gpu;

/// <summary>
/// #1650: optional capability interface for engines that can capture a model's eager GPU forward into
/// a replayable CUDA graph (the TensorRT-LLM / vLLM decode-graph lever). Consumers typed against the
/// core <c>IEngine</c> downcast to this interface (<c>engine is IInferenceGraphCaptureEngine cap</c>)
/// to reach the capture surface without a concrete-type cast.
/// <para>
/// Kept off the base engine interface — like <see cref="IGpuHalfPrecisionBackend"/> — because only the
/// CUDA-backed <c>DirectGpuTensorEngine</c> ships graph capture; making it a base member would force
/// every CPU/other-backend engine to implement no-ops or throw <c>NotSupportedException</c>.
/// </para>
/// </summary>
public interface IInferenceGraphCaptureEngine
{
    /// <summary>True when this engine is backed by a backend that can capture/replay CUDA graphs.</summary>
    bool SupportsInferenceGraphCapture { get; }

    /// <summary>Records the GPU kernel launches issued by <paramref name="forward"/> into a CUDA graph and
    /// returns a replayable handle (<see cref="IntPtr.Zero"/> on failure / a non-capturable op). Must be
    /// called inside an <see cref="EnterResidentCaptureScope"/> so the forward runs resident (capturable).</summary>
    IntPtr CaptureGpuGraph(Action forward);

    /// <summary>Replays a graph captured by <see cref="CaptureGpuGraph"/> as a single cuGraphLaunch.</summary>
    void LaunchGpuGraph(IntPtr graphExec);

    /// <summary>Frees a captured graph handle.</summary>
    void DestroyGpuGraph(IntPtr graphExec);

    /// <summary>Enters the resident capture path (eviction suspended, resident in-place ops engaged). The
    /// returned scope reverses it on Dispose — keep it alive for the captured graph's lifetime. Null on a
    /// non-supporting backend.</summary>
    IDisposable? EnterResidentCaptureScope();

    /// <summary>Uploads <paramref name="t"/>'s current host data into its EXISTING resident GPU buffer in
    /// place (stable pointer) so a replayed graph reads fresh per-call data. No-op without a resident buffer.</summary>
    void RefreshResidentInputInPlace<T>(Tensor<T> t);

    /// <summary>Downloads <paramref name="t"/>'s resident GPU buffer (which a replayed graph just wrote) into
    /// a fresh host array, forcing a fresh DtoH each replay. Null if the tensor has no resident buffer.</summary>
    float[]? DownloadResidentBuffer<T>(Tensor<T> t);
}
