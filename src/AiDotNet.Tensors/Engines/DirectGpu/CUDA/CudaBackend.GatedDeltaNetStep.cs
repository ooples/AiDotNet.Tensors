// Copyright (c) CrowdTrainer LLC. All rights reserved.
// CudaBackend.IGatedDeltaNetStepBackend: the gated delta rule's decode step in one launch.
using System;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA;

public sealed partial class CudaBackend : IGatedDeltaNetStepBackend
{
    /// <inheritdoc />
    unsafe void IGatedDeltaNetStepBackend.GatedDeltaNetStep(IGpuBuffer state, IGpuBuffer query, IGpuBuffer key,
        IGpuBuffer value, IGpuBuffer alpha, IGpuBuffer beta, IGpuBuffer output, int rows, int dv, int dk,
        float keyScale, bool decayFirst)
    {
        if (rows <= 0 || dv <= 0 || dk <= 0) throw new ArgumentOutOfRangeException(nameof(rows), "rows, dv and dk must be positive.");
        if (!IsAvailable) throw new GpuCapabilityUnavailableException("CUDA backend is not available.");
        if (!_kernelCache.TryGetValue("gdn_step", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: gdn_step");
        using var _ = PushContext();
        IntPtr s = state.Handle, q = query.Handle, k = key.Handle, v = value.Handle, a = alpha.Handle, b = beta.Handle, o = output.Handle;
        int r = rows, vd = dv, kd = dk, decay = decayFirst ? 1 : 0;
        void** args = stackalloc void*[12];
        args[0] = &s; args[1] = &q; args[2] = &k; args[3] = &v; args[4] = &a; args[5] = &b; args[6] = &o;
        args[7] = &r; args[8] = &vd; args[9] = &kd; args[10] = &keyScale; args[11] = &decay;
        const int Threads = 256;
        long warps = (long)rows * dv;
        LaunchKernel(kernel, (uint)((warps * 32 + Threads - 1) / Threads), Threads, args);
    }
}
