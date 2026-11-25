// Copyright (c) CrowdTrainer LLC. All rights reserved.
#if NET8_0_OR_GREATER
using System;

namespace AiDotNet.Tensors.Engines.DirectGpu.WebGpu;

public sealed partial class WebGpuBackend : IBf16WidenBackend
{
    bool IBf16WidenBackend.CanWidenBf16 => IsAvailable;

    void IBf16WidenBackend.WidenBf16ToFp32(IGpuBuffer bf16, IGpuBuffer fp32, int count)
    {
        if (GpuBufferCapacity.IsEmptyLaunch(count)) return;
        GpuBufferCapacity.Require(bf16, (count + 1) / 2, nameof(bf16)); GpuBufferCapacity.Require(fp32, count, nameof(fp32));
        DispatchNBufferAsync("Audit:Bf16ToFloat", WebGpuAuditKernels.Bf16ToFloat, "main",
            new IGpuBuffer[] { bf16, fp32 }, new float[] { BitConverter.Int32BitsToSingle(count), 0f, 0f, 0f }, count).GetAwaiter().GetResult();
    }
}
#endif
