// Copyright (c) CrowdTrainer LLC. All rights reserved.

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA;

public sealed partial class CudaBackend : IBf16WidenBackend
{
    bool IBf16WidenBackend.CanWidenBf16 => IsAvailable && _kernelCache.ContainsKey("convert_bf16_to_fp32_native");

    void IBf16WidenBackend.WidenBf16ToFp32(IGpuBuffer bf16, IGpuBuffer fp32, int count)
    {
        if (count <= 0) return;
        LaunchUnaryFp16("convert_bf16_to_fp32_native", bf16, fp32, count);
    }
}
