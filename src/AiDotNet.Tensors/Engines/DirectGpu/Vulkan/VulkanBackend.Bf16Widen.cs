// Copyright (c) CrowdTrainer LLC. All rights reserved.

namespace AiDotNet.Tensors.Engines.DirectGpu.Vulkan;

public sealed partial class VulkanBackend : IBf16WidenBackend
{
    bool IBf16WidenBackend.CanWidenBf16 => IsAvailable;

    void IBf16WidenBackend.WidenBf16ToFp32(IGpuBuffer bf16, IGpuBuffer fp32, int count)
    {
        if (GpuBufferCapacity.IsEmptyLaunch(count)) return;
        GpuBufferCapacity.Require(bf16, (count + 1) / 2, nameof(bf16)); GpuBufferCapacity.Require(fp32, count, nameof(fp32));
        GlslDispatchN(VulkanAuditKernels.Bf16ToFloat, count,
            new IGpuBuffer[] { bf16, fp32 },
            new uint[] { (uint)count });
    }
}
