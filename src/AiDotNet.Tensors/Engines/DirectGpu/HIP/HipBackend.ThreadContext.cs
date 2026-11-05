// Copyright (c) CrowdTrainer LLC. All rights reserved.
namespace AiDotNet.Tensors.Engines.DirectGpu.HIP;

public sealed partial class HipBackend : IThreadContextBackend
{
    [System.ThreadStatic] private static bool t_drainingExternalReleases;

    /// <summary>Binds <paramref name="device"/> to this thread again.</summary>
    private static void RebindThreadDevice(int device)
    {
        var result = HipNativeBindings.hipSetDevice(device);
        if (result != HipError.Success) throw new HipException("hipSetDevice failed restoring this thread's HIP device.", result);
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
        if (!IsAvailable) return;
        // Constructors and foreign libraries can call hipSetDevice outside this binder. A managed cache cannot
        // establish which device is current; query HIP before deciding whether a rebind is necessary.
        int current = -1;
        HipNativeBindings.CheckError(HipNativeBindings.hipGetDevice(ref current), "hipGetDevice(context binding)");
        if (current != _hipDeviceIndex) RebindThreadDevice(_hipDeviceIndex);
    }

    /// <remarks>HIP runs on the device's primary context; there is no context object to share.</remarks>
    IntPtr IThreadContextBackend.NativeContextHandle => IntPtr.Zero;
}
