// Copyright (c) CrowdTrainer LLC. All rights reserved.

namespace AiDotNet.Tensors.Engines.DirectGpu.Metal;

public sealed partial class MetalBackend : IBf16WidenBackend
{
    bool IBf16WidenBackend.CanWidenBf16 => IsAvailable;

    void IBf16WidenBackend.WidenBf16ToFp32(IGpuBuffer bf16, IGpuBuffer fp32, int count)
    {
        ThrowIfDisposed();
        var pipeline = GetParity210Pipeline("parity210_bf16_to_float");
        if (GpuBufferCapacity.IsEmptyLaunch(count)) return;
        GpuBufferCapacity.Require(bf16, (count + 1) / 2, nameof(bf16)); GpuBufferCapacity.Require(fp32, count, nameof(fp32));
        var (tgr, tpg) = pipeline.Calculate1DDispatch(count);
        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        encoder.SetBuffer((MetalGpuBuffer)bf16, 0);
        encoder.SetBuffer((MetalGpuBuffer)fp32, 1);
        encoder.SetBytes(count, 2);
        encoder.DispatchThreadgroups(tgr, tpg);
    }
}
