// Copyright (c) AiDotNet. All rights reserved.
namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA;

public sealed partial class CudaBackend : IThreadContextBackend
{
    /// <inheritdoc/>
    /// <remarks>Re-asserts the per-thread CUDA context (no-op when already current).</remarks>
    void IThreadContextBackend.EnsureContextCurrent() => EnsureContextCurrent();
}
