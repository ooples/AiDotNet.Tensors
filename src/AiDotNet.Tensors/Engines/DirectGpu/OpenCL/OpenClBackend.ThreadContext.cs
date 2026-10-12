// Copyright (c) AiDotNet. All rights reserved.
namespace AiDotNet.Tensors.Engines.DirectGpu.OpenCL;

public sealed partial class OpenClBackend : IThreadContextBackend
{
    /// <inheritdoc/>
    /// <remarks>OpenCL command queues are not bound to a thread.</remarks>
    void IThreadContextBackend.EnsureContextCurrent()
    {
    }

    /// <remarks>No driver context object is shared with other components here.</remarks>
    IntPtr IThreadContextBackend.NativeContextHandle => IntPtr.Zero;
}
