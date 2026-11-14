// Copyright (c) CrowdTrainer LLC. All rights reserved.
namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>A captured graph's kernel, memcpy and memset nodes (see <see cref="IGpuGraphCapture.LastCaptureNodeCounts"/>).</summary>
internal readonly struct GraphNodeCounts
{
    internal GraphNodeCounts(int kernel, int memcpy, int memset, string? memsetSources = null)
    {
        Kernel = kernel;
        Memcpy = memcpy;
        Memset = memset;
        MemsetSources = memsetSources;
    }

    /// <summary>Kernel launch nodes.</summary>
    internal int Kernel { get; }

    /// <summary>Memory copy nodes.</summary>
    internal int Memcpy { get; }

    /// <summary>Memory set nodes.</summary>
    internal int Memset { get; }

    /// <summary>
    /// Each memset node's byte count and the kernel it feeds (or follows, for a memset nothing depends on), separated by
    /// "; ", so a step-graph ratchet that sees a new memset can name its producer. Null when there are no memset nodes or
    /// the backend cannot introspect its graph.
    /// </summary>
    internal string? MemsetSources { get; }
}

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

    /// <summary>
    /// Why the calling thread's most recent <see cref="CaptureGraph"/> returned zero (#1096 F4): the exception the captured
    /// body threw, or the driver call that failed. A backend ends a capture cleanly when its body throws and returns a
    /// zero handle, so without this the caller knows only that capture failed, not why. Null after a capture that
    /// succeeded, before any capture on this thread, or on a backend without graph capture.
    /// </summary>
    string? LastCaptureFailure { get; }

    /// <summary>
    /// The node counts of the graph the calling thread's most recent <see cref="CaptureGraph"/> returned: what a
    /// captured step costs in launches and copies (#1096 P0.7). Null after a capture that returned zero, before any
    /// capture on this thread, or on a backend without graph capture.
    /// </summary>
    GraphNodeCounts? LastCaptureNodeCounts { get; }

    /// <summary>Replays a graph <see cref="CaptureGraph"/> returned.</summary>
    void LaunchCapturedGraph(System.IntPtr graphExec);

    /// <summary>
    /// Queues a graph <see cref="CaptureGraph"/> returned without waiting for it: work queued on the backend's
    /// stream afterwards runs after it, but the host must wait before reading anything it writes.
    /// </summary>
    void EnqueueCapturedGraph(System.IntPtr graphExec);

    /// <summary>Frees a graph <see cref="CaptureGraph"/> returned.</summary>
    void DestroyCapturedGraph(System.IntPtr graphExec);

    /// <summary>
    /// Page-locked step I/O for a graph that reads <paramref name="inputs"/> (the first <paramref name="lengths"/>
    /// floats of each): see <see cref="IGraphStepIo"/>. Null where the backend has none (every backend without device
    /// graphs); the caller then refreshes inputs and reads results through ordinary copies.
    /// </summary>
    IGraphStepIo? CreateStepIo(System.Collections.Generic.IReadOnlyList<IGpuBuffer> inputs,
        System.Collections.Generic.IReadOnlyList<int> lengths);
}