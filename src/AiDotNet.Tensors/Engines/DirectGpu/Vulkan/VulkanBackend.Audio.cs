// Copyright (c) AiDotNet. All rights reserved.
namespace AiDotNet.Tensors.Engines.DirectGpu.Vulkan;

public sealed unsafe partial class VulkanBackend : IAudioBackend
{
    public void AmplitudeToDB(IGpuBuffer input, IGpuBuffer output, int length,
        float minAmplitude, float topDb, bool clipTopDb)
    {
        if (length <= 0) return;
        uint mA = BitConverter.ToUInt32(BitConverter.GetBytes(minAmplitude), 0);
        uint tD = BitConverter.ToUInt32(BitConverter.GetBytes(topDb), 0);
        var pc = new uint[] { (uint)length, mA, tD, (uint)(clipTopDb ? 1 : 0) };
        GlslUnaryOp(VulkanAudioKernels.AmplitudeToDB, input, output, length, pc, 16u);
    }

    public void MuLawEncoding(IGpuBuffer input, IGpuBuffer output, int length, int qc)
    {
        if (length <= 0) return;
        var pc = new uint[] { (uint)length, (uint)qc };
        GlslUnaryOp(VulkanAudioKernels.MuLawEncoding, input, output, length, pc, 8u);
    }

    public void MuLawDecoding(IGpuBuffer input, IGpuBuffer output, int length, int qc)
    {
        if (length <= 0) return;
        var pc = new uint[] { (uint)length, (uint)qc };
        GlslUnaryOp(VulkanAudioKernels.MuLawDecoding, input, output, length, pc, 8u);
    }

    public void ComputeDeltas(IGpuBuffer input, IGpuBuffer output,
        int leading, int timeAxis, int winLength)
    {
        int total = leading * timeAxis;
        if (total <= 0) return;
        var pc = new uint[] { (uint)leading, (uint)timeAxis, (uint)winLength };
        GlslUnaryOp(VulkanAudioKernels.ComputeDeltas, input, output, total, pc, 12u);
    }

    public void Resample(IGpuBuffer input, IGpuBuffer output,
        int leading, int inLen, int outLen, int up, int down, int halfWidth)
    {
        int total = leading * outLen;
        if (total <= 0) return;
        var pc = new uint[] {
            (uint)leading, (uint)inLen, (uint)outLen, (uint)up, (uint)down, (uint)halfWidth
        };
        GlslUnaryOp(VulkanAudioKernels.Resample, input, output, total, pc, 24u);
    }
}
