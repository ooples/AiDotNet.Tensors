// Copyright (c) CrowdTrainer LLC. All rights reserved.

namespace AiDotNet.Tensors.Engines.DirectGpu.HIP;

public sealed partial class HipBackend : IBf16WidenBackend
{
    bool IBf16WidenBackend.CanWidenBf16 => IsAvailable && _parity210Module != System.IntPtr.Zero
        && _kernelCache.ContainsKey("parity210_bf16_to_float");

    unsafe void IBf16WidenBackend.WidenBf16ToFp32(IGpuBuffer bf16, IGpuBuffer fp32, int count)
    {
        var kernel = ResolveParity210Kernel("parity210_bf16_to_float");
        if (GpuBufferCapacity.IsEmptyLaunch(count)) return;
        GpuBufferCapacity.Require(bf16, (count + 1) / 2, nameof(bf16)); GpuBufferCapacity.Require(fp32, count, nameof(fp32));
        uint gsz = (uint)((count + DefaultBlockSize - 1) / DefaultBlockSize);
        System.IntPtr a0 = bf16.Handle; System.IntPtr a1 = fp32.Handle; int a2 = count;
        void** args = stackalloc void*[3];
        args[0] = &a0; args[1] = &a1; args[2] = &a2;
        LaunchKernel(kernel, gsz, DefaultBlockSize, args);
    }
}
