// Copyright (c) AiDotNet. All rights reserved.
namespace AiDotNet.Tensors.Engines.DirectGpu.Metal;

public sealed partial class MetalBackend : IThreadContextBackend
{
    /// <inheritdoc/>
    /// <remarks>Metal devices and command queues are not bound to a thread.</remarks>
    void IThreadContextBackend.EnsureContextCurrent()
    {
    }
}
