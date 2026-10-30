#if NET7_0_OR_GREATER
// Copyright (c) CrowdTrainer LLC. All rights reserved.
namespace AiDotNet.Tensors.Engines.DirectGpu.WebGpu;

public sealed partial class WebGpuBackend : IThreadContextBackend
{
    /// <inheritdoc/>
    /// <remarks>The WebGPU device is not bound to a thread.</remarks>
    void IThreadContextBackend.EnsureContextCurrent()
    {
    }

    /// <remarks>No driver context object is shared with other components here.</remarks>
    IntPtr IThreadContextBackend.NativeContextHandle => IntPtr.Zero;
}
#endif
