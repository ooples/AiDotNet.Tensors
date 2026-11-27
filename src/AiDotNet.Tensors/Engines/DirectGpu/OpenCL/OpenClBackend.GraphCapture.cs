// Copyright (c) CrowdTrainer LLC. All rights reserved.
namespace AiDotNet.Tensors.Engines.DirectGpu.OpenCL;

// No device graphs on this backend: work runs eagerly, which is what a zero capture handle tells every caller.
public sealed partial class OpenClBackend : IGpuGraphCapture
{
    bool IGpuGraphCapture.SupportsGraphCapture => false;

    bool IGpuGraphCapture.ContextIsLive => IsAvailable;

    string? IGpuGraphCapture.LastCaptureFailure => null;

    GraphNodeCounts? IGpuGraphCapture.LastCaptureNodeCounts => null;

    bool IGpuGraphCapture.IsStreamCapturing() => false;
    bool IGpuGraphCapture.CaptureActiveOnThisThread => false;

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

// No device graphs on this backend, so no plan holds step memory across its steps: nothing idle is lent, and an owner's
// reclaim has nothing to take back. Allocations between steps come from the backend's own allocator.
public sealed partial class OpenClBackend : IIdleDeviceMemoryLender, IIdleLoanCaptureGuard
{
    // This backend lends nothing (LendIdle is a no-op), so there is no loan to move before a capture.
    System.IDisposable IIdleLoanCaptureGuard.SuspendLoansForCapture() => NoLoans.Instance;

    private sealed class NoLoans : System.IDisposable
    {
        public static readonly NoLoans Instance = new();
        public void Dispose() { }
    }

    void IIdleDeviceMemoryLender.LendIdle(object owner, System.Collections.Generic.IReadOnlyCollection<IGpuBuffer> buffers)
    {
        if (owner is null) throw new System.ArgumentNullException(nameof(owner));
        if (buffers is null) throw new System.ArgumentNullException(nameof(buffers));
    }

    void IIdleDeviceMemoryLender.LendIdleOnDemand(object owner,
        System.Func<System.Collections.Generic.IReadOnlyCollection<IGpuBuffer>?> collect)
    {
        if (owner is null) throw new System.ArgumentNullException(nameof(owner));
        if (collect is null) throw new System.ArgumentNullException(nameof(collect));
    }

    void IIdleDeviceMemoryLender.ReclaimIdle(object owner)
    {
        if (owner is null) throw new System.ArgumentNullException(nameof(owner));
    }

    // Nothing is lent here, so nothing is ever moved.
    void IIdleDeviceMemoryLender.MoveLiveLoans()
    {
    }
}
