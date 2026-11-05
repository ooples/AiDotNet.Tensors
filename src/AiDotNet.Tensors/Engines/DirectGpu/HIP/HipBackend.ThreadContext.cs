// Copyright (c) CrowdTrainer LLC. All rights reserved.
namespace AiDotNet.Tensors.Engines.DirectGpu.HIP;

public sealed partial class HipBackend : IThreadContextBackend
{
    // The device each thread last bound through this method (-1: none), so the common case skips the driver call.
    [System.ThreadStatic] private static int t_boundDevice;
    [System.ThreadStatic] private static bool t_bound;
    [System.ThreadStatic] private static bool t_drainingExternalReleases;

    /// <summary>Binds <paramref name="device"/> to this thread again, keeping the bound-device cache in step.</summary>
    private static void RebindThreadDevice(int device)
    {
        var result = HipNativeBindings.hipSetDevice(device);
        if (result != HipError.Success) throw new HipException("hipSetDevice failed restoring this thread's HIP device.", result);
        t_boundDevice = device;
        t_bound = true;
    }

    /// <inheritdoc/>
    /// <remarks>hipSetDevice is per thread: without it, a thread other than the creating one runs on device 0.</remarks>
    void IThreadContextBackend.EnsureContextCurrent()
    {
        // An operation is about to run: hand back imported DLPack memory whose tensor was finalized, as the DLPack
        // contract promises ("at the backend's next operation"), not only at the next allocation. The drain itself
        // binds devices through this method, so it does not re-enter.
        if (!PendingExternalReleases.IsEmpty && !t_drainingExternalReleases)
        {
            t_drainingExternalReleases = true;
            try { DrainPendingExternalReleases(); }
            finally { t_drainingExternalReleases = false; }
        }
        if (!IsAvailable || (t_bound && t_boundDevice == _hipDeviceIndex)) return;
        var result = HipNativeBindings.hipSetDevice(_hipDeviceIndex);
        if (result != HipError.Success) throw new HipException("hipSetDevice failed binding the HIP device to this thread.", result);
        t_boundDevice = _hipDeviceIndex;
        t_bound = true;
    }

    /// <remarks>HIP runs on the device's primary context; there is no context object to share.</remarks>
    IntPtr IThreadContextBackend.NativeContextHandle => IntPtr.Zero;
}
