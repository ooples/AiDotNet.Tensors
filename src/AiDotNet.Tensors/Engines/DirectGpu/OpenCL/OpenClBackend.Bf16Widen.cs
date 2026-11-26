// Copyright (c) CrowdTrainer LLC. All rights reserved.

namespace AiDotNet.Tensors.Engines.DirectGpu.OpenCL
{
    public sealed partial class OpenClBackend : IBf16WidenBackend
    {
        bool IBf16WidenBackend.CanWidenBf16 => _context != null && _kernelCache.ContainsKey("bf16_to_float");

        void IBf16WidenBackend.WidenBf16ToFp32(IGpuBuffer bf16, IGpuBuffer fp32, int count)
        {
            if (_context == null) throw new GpuCapabilityUnavailableException("The OpenCL context is unavailable.");
            if (GpuBufferCapacity.IsEmptyLaunch(count)) return;
            GpuBufferCapacity.Require(bf16, (count + 1) / 2, nameof(bf16)); GpuBufferCapacity.Require(fp32, count, nameof(fp32));
            var k = _kernelCache["bf16_to_float"];
            k.SetArg(0, ((DirectOpenClGpuBuffer)bf16).Buffer.Handle);
            k.SetArg(1, ((DirectOpenClGpuBuffer)fp32).Buffer.Handle);
            k.SetArg(2, count);
            k.Execute1D(count, CalculateOptimalWorkGroupSize1D(count));
        }
    }
}
