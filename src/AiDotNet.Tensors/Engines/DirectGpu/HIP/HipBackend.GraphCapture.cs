// Copyright (c) AiDotNet. All rights reserved.
namespace AiDotNet.Tensors.Engines.DirectGpu.HIP;

// No device graphs on this backend: work runs eagerly, which is what a zero capture handle tells every caller.
public sealed partial class HipBackend : IGpuGraphCapture
{
    bool IGpuGraphCapture.SupportsGraphCapture => false;

    bool IGpuGraphCapture.IsStreamCapturing() => false;

    System.IntPtr IGpuGraphCapture.CaptureGraph(System.Action launch)
    {
        if (launch is null) throw new System.ArgumentNullException(nameof(launch));
        return System.IntPtr.Zero;
    }

    void IGpuGraphCapture.LaunchCapturedGraph(System.IntPtr graphExec)
        => throw new System.ArgumentException("This backend never returns a graph handle.", nameof(graphExec));

    void IGpuGraphCapture.DestroyCapturedGraph(System.IntPtr graphExec)
        => throw new System.ArgumentException("This backend never returns a graph handle.", nameof(graphExec));
}
