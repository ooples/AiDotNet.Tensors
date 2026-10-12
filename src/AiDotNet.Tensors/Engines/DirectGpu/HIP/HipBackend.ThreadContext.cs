// Copyright (c) AiDotNet. All rights reserved.
namespace AiDotNet.Tensors.Engines.DirectGpu.HIP;

public sealed partial class HipBackend : IThreadContextBackend
{
    // The device each thread last bound through this method (-1: none), so the common case skips the driver call.
    [System.ThreadStatic] private static int t_boundDevice;
    [System.ThreadStatic] private static bool t_bound;

    /// <inheritdoc/>
    /// <remarks>hipSetDevice is per thread: without it, a thread other than the creating one runs on device 0.</remarks>
    void IThreadContextBackend.EnsureContextCurrent()
    {
        if (!IsAvailable || (t_bound && t_boundDevice == _hipDeviceIndex)) return;
        var result = HipNativeBindings.hipSetDevice(_hipDeviceIndex);
        if (result != HipError.Success) throw new HipException("hipSetDevice failed binding the HIP device to this thread.", result);
        t_boundDevice = _hipDeviceIndex;
        t_bound = true;
    }

    /// <remarks>HIP runs on the device's primary context; there is no context object to share.</remarks>
    IntPtr IThreadContextBackend.NativeContextHandle => IntPtr.Zero;
}
