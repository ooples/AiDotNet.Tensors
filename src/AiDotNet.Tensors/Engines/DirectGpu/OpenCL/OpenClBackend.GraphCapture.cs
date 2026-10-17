// Copyright (c) AiDotNet. All rights reserved.
namespace AiDotNet.Tensors.Engines.DirectGpu.OpenCL;

// No device graphs on this backend: work runs eagerly, which is what a zero capture handle tells every caller.
public sealed partial class OpenClBackend : IGpuGraphCapture
{
    bool IGpuGraphCapture.SupportsGraphCapture => false;

    bool IGpuGraphCapture.ContextIsLive => IsAvailable;

    bool IGpuGraphCapture.IsStreamCapturing() => false;

    int IGpuGraphCapture.StreamCaptureStatusRaw() => 0;

    System.IntPtr IGpuGraphCapture.CaptureGraph(System.Action launch)
    {
        if (launch is null) throw new System.ArgumentNullException(nameof(launch));
        return System.IntPtr.Zero;
    }

    void IGpuGraphCapture.LaunchCapturedGraph(System.IntPtr graphExec)
        => throw new System.ArgumentException("This backend never returns a graph handle.", nameof(graphExec));

    void IGpuGraphCapture.EnqueueCapturedGraph(System.IntPtr graphExec)
        => throw new System.ArgumentException("This backend never returns a graph handle.", nameof(graphExec));

    void IGpuGraphCapture.DestroyCapturedGraph(System.IntPtr graphExec)
        => throw new System.ArgumentException("This backend never returns a graph handle.", nameof(graphExec));

    IGraphStepIo? IGpuGraphCapture.CreateStepIo(System.Collections.Generic.IReadOnlyList<IGpuBuffer> inputs,
        System.Collections.Generic.IReadOnlyList<int> lengths) => null;
}
