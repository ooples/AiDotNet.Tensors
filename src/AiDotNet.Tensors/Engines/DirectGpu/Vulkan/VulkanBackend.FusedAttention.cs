// Copyright (c) AiDotNet. All rights reserved.
// Vulkan IFusedAttentionBackend (#1179): the portable fused attention over the shared fa_p_ kernels.
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using P = AiDotNet.Tensors.Engines.DirectGpu.PortableFusedAttention;

namespace AiDotNet.Tensors.Engines.DirectGpu.Vulkan;

public sealed partial class VulkanBackend : IFusedAttentionBackend
{
    bool IFusedAttentionBackend.SupportsFusedAttention(int headDim) => P.Supports(headDim);

    void IFusedAttentionBackend.FusedAttentionForward(IGpuBuffer q, IGpuBuffer k, IGpuBuffer v, IGpuBuffer o, IGpuBuffer lse,
        IGpuBuffer? mask, in FusedAttentionParams p, int headDim)
        => P.Forward(this, LaunchSolver, q, k, v, o, lse, mask, p, headDim);

    void IFusedAttentionBackend.FusedAttentionBackward(IGpuBuffer q, IGpuBuffer k, IGpuBuffer v, IGpuBuffer o, IGpuBuffer dO,
        IGpuBuffer lse, IGpuBuffer? mask, IGpuBuffer dQ, IGpuBuffer dK, IGpuBuffer dV, in FusedAttentionParams p, int headDim)
        => P.Backward(this, LaunchSolver, q, k, v, o, dO, lse, mask, dQ, dK, dV, p, headDim);
}
