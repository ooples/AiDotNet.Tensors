// Copyright (c) AiDotNet. All rights reserved.
namespace AiDotNet.Tensors.Engines.DirectGpu.Vulkan;

public sealed unsafe partial class VulkanBackend : IThreadContextBackend
{
    /// <inheritdoc/>
    /// <remarks>A Vulkan device is not bound to a thread; queue access is locked internally.</remarks>
    void IThreadContextBackend.EnsureContextCurrent()
    {
    }

    /// <remarks>No driver context object is shared with other components here.</remarks>
    IntPtr IThreadContextBackend.NativeContextHandle => IntPtr.Zero;
}
