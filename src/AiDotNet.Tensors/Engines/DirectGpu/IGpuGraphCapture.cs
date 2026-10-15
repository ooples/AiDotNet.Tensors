// Copyright (c) AiDotNet. All rights reserved.
namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>
/// Recording the launches of a delegate into a replayable device graph (#1154). A backend without device graphs
/// reports <see cref="SupportsGraphCapture"/> false and returns <see cref="System.IntPtr.Zero"/> from
/// <see cref="CaptureGraph"/>, which every caller already treats as "run eagerly".
/// </summary>
internal interface IGpuGraphCapture : IDirectGpuBackend
{
    /// <summary>True when <see cref="CaptureGraph"/> can produce a replayable graph on this backend.</summary>
    bool SupportsGraphCapture { get; }

    /// <summary>False once the device context is gone (backend disposed), so nothing can be read back.</summary>
    bool ContextIsLive { get; }

    /// <summary>True while a capture is recording on the calling thread's stream.</summary>
    bool IsStreamCapturing();

    /// <summary>The driver's raw capture status for diagnostics: 0 none, 1 active, 2 invalidated.</summary>
    int StreamCaptureStatusRaw();

    /// <summary>Records <paramref name="launch"/>'s device work; a handle to replay, or zero when it cannot.</summary>
    System.IntPtr CaptureGraph(System.Action launch);

    /// <summary>Replays a graph <see cref="CaptureGraph"/> returned.</summary>
    void LaunchCapturedGraph(System.IntPtr graphExec);

    /// <summary>
    /// Queues a graph <see cref="CaptureGraph"/> returned without waiting for it: work queued on the backend's
    /// stream afterwards runs after it, but the host must wait before reading anything it writes.
    /// </summary>
    void EnqueueCapturedGraph(System.IntPtr graphExec);

    /// <summary>Frees a graph <see cref="CaptureGraph"/> returned.</summary>
    void DestroyCapturedGraph(System.IntPtr graphExec);
}