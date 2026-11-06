// Copyright (c) CrowdTrainer LLC. All rights reserved.
namespace AiDotNet.Tensors.Engines.DirectGpu.HIP;

public sealed partial class HipBackend : IThreadContextBackend
{
    [System.ThreadStatic] private static bool t_drainingExternalReleases;

    /// <summary>Binds <paramref name="device"/> to this thread again.</summary>
    private static void RebindThreadDevice(int device) =>
        HipNativeBindings.SetCurrentDevice(device, "hipSetDevice(restore this thread's device)");

    /// <summary>
    /// Hands back imported DLPack memory whose tensor was finalized. The drain binds devices through
    /// EnsureContextCurrent, so a thread-static guard keeps it from re-entering itself.
    /// </summary>
    private static void DrainPendingExternalReleasesOnThisThread()
    {
        if (PendingExternalReleases.IsEmpty || t_drainingExternalReleases) return;
        t_drainingExternalReleases = true;
        try { DrainPendingExternalReleases(); }
        finally { t_drainingExternalReleases = false; }
    }

    /// <inheritdoc/>
    /// <remarks>hipSetDevice is per thread: without it, a thread other than the creating one runs on device 0.</remarks>
    void IThreadContextBackend.EnsureContextCurrent()
    {
        // An operation is about to run: hand back imported DLPack memory whose tensor was finalized, as the DLPack
        // contract promises ("at the backend's next operation"), not only at the next allocation.
        DrainPendingExternalReleasesOnThisThread();
        if (!IsAvailable) return;
        // Constructors and foreign libraries can call hipSetDevice outside this binder. A managed cache cannot
        // establish which device is current; query HIP before deciding whether a rebind is necessary.
        if (HipNativeBindings.GetCurrentDevice("hipGetDevice(context binding)") != _hipDeviceIndex)
            HipNativeBindings.SetCurrentDevice(_hipDeviceIndex, "hipSetDevice(bind the HIP device to this thread)");
    }

    /// <remarks>HIP runs on the device's primary context; there is no context object to share.</remarks>
    IntPtr IThreadContextBackend.NativeContextHandle => IntPtr.Zero;
}
