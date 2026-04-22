// Copyright (c) AiDotNet. All rights reserved.
namespace AiDotNet.Tensors.Engines.DirectGpu.Vulkan;

public sealed unsafe partial class VulkanBackend : IRoiBackend
{
    // RoIAlign push: 7 ints + 1 float + 2 ints = 40 bytes
    // RoIPool push: 7 ints + 1 float = 32 bytes
    private const uint RoiAlignPushSize = 40u;
    private const uint RoiPoolPushSize = 32u;

    public void RoIAlign(IGpuBuffer input, IGpuBuffer boxes, IGpuBuffer output,
        int N, int C, int H, int W, int K, int outH, int outW,
        float spatialScale, int samplingRatio, bool aligned)
    {
        int total = K * C * outH * outW;
        if (total <= 0) return;
        uint ss = BitConverter.ToUInt32(BitConverter.GetBytes(spatialScale), 0);
        var pc = new uint[] {
            (uint)N, (uint)C, (uint)H, (uint)W, (uint)K, (uint)outH, (uint)outW,
            ss, (uint)samplingRatio, (uint)(aligned ? 1 : 0)
        };
        GlslBinaryOp(VulkanRoiKernels.RoIAlign, input, boxes, output, total, pc, RoiAlignPushSize);
    }

    public void RoIPool(IGpuBuffer input, IGpuBuffer boxes, IGpuBuffer output,
        int N, int C, int H, int W, int K, int outH, int outW, float spatialScale)
    {
        int total = K * C * outH * outW;
        if (total <= 0) return;
        uint ss = BitConverter.ToUInt32(BitConverter.GetBytes(spatialScale), 0);
        var pc = new uint[] {
            (uint)N, (uint)C, (uint)H, (uint)W, (uint)K, (uint)outH, (uint)outW, ss
        };
        GlslBinaryOp(VulkanRoiKernels.RoIPool, input, boxes, output, total, pc, RoiPoolPushSize);
    }

    // Push constants for PsRoI: RoIAlign = 8 ints + 1 float + 1 int = 40;
    // RoIPool = 8 ints + 1 float = 36.
    private const uint PsRoiAlignPushSize = 40u;
    private const uint PsRoiPoolPushSize = 36u;

    public void PsRoIAlign(IGpuBuffer input, IGpuBuffer boxes, IGpuBuffer output,
        int N, int C, int H, int W, int K, int outH, int outW, int outputChannels,
        float spatialScale, int samplingRatio)
    {
        int total = K * outputChannels * outH * outW;
        if (total <= 0) return;
        uint ss = BitConverter.ToUInt32(BitConverter.GetBytes(spatialScale), 0);
        var pc = new uint[] {
            (uint)N, (uint)C, (uint)H, (uint)W, (uint)K, (uint)outH, (uint)outW,
            (uint)outputChannels, ss, (uint)samplingRatio
        };
        GlslBinaryOp(VulkanRoiKernels.PsRoIAlign, input, boxes, output, total, pc, PsRoiAlignPushSize);
    }

    public void PsRoIPool(IGpuBuffer input, IGpuBuffer boxes, IGpuBuffer output,
        int N, int C, int H, int W, int K, int outH, int outW, int outputChannels,
        float spatialScale)
    {
        int total = K * outputChannels * outH * outW;
        if (total <= 0) return;
        uint ss = BitConverter.ToUInt32(BitConverter.GetBytes(spatialScale), 0);
        var pc = new uint[] {
            (uint)N, (uint)C, (uint)H, (uint)W, (uint)K, (uint)outH, (uint)outW,
            (uint)outputChannels, ss
        };
        GlslBinaryOp(VulkanRoiKernels.PsRoIPool, input, boxes, output, total, pc, PsRoiPoolPushSize);
    }
}
