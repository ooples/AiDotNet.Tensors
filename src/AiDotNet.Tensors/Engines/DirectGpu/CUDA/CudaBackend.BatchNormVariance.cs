// Copyright (c) AiDotNet. All rights reserved.
using System;
namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA;

public sealed partial class CudaBackend : IBatchNormVarianceBackend
{
    private const int BatchNormRegisterMaxFloat4PerThread = 8;
    private const int BatchNormRegisterThreads = 256;

    /// <summary>The kernel suffix (float4s per thread) and block size for a channel slice of <paramref name="n4"/>
    /// float4s, or 0 when the slice does not fit the register-resident variant.</summary>
    private static int BatchNormRegisterVariant(int batch, int spatial, out int threads)
    {
        threads = 0;
        if (batch <= 0 || spatial <= 0 || (spatial & 3) != 0) return 0;
        long n4 = (long)batch * (spatial >> 2);
        if (n4 > (long)BatchNormRegisterThreads * BatchNormRegisterMaxFloat4PerThread) return 0;
        int k4 = 1;
        while ((long)k4 * BatchNormRegisterThreads < n4) k4 <<= 1;
        threads = (int)System.Math.Min(BatchNormRegisterThreads, ((n4 + k4 - 1) / k4 + 31) / 32 * 32);
        return k4;
    }

    unsafe bool IBatchNormVarianceBackend.TryBatchNormTrainingWithVariance(IGpuBuffer input, IGpuBuffer output,
        IGpuBuffer gamma, IGpuBuffer beta, IGpuBuffer saveMean, IGpuBuffer saveVariance,
        int batch, int channels, int spatial, float epsilon, bool relu)
    {
        int k4 = BatchNormRegisterVariant(batch, spatial, out int threads);
        long n = (long)batch * channels * spatial;
        if (k4 == 0 || channels <= 0 || !IsAvailable || input.Size < n || output.Size < n || gamma.Size < channels
            || beta.Size < channels || saveMean.Size < channels || saveVariance.Size < channels
            || !_kernelCache.TryGetValue((relu ? "batchnorm_fwd_relu_reg" : "batchnorm_fwd_reg") + k4, out var kernel))
            return false;
        using var _ = PushContext();
        IntPtr pi = input.Handle, po = output.Handle, pg = gamma.Handle, pb = beta.Handle, pm = saveMean.Handle,
            pv = saveVariance.Handle;
        int b = batch, c = channels, s = spatial;
        float eps = epsilon;
        void** args = stackalloc void*[10];
        args[0] = &pi; args[1] = &po; args[2] = &pg; args[3] = &pb; args[4] = &pm; args[5] = &pv;
        args[6] = &b; args[7] = &c; args[8] = &s; args[9] = &eps;
        LaunchKernel(kernel, (uint)channels, (uint)threads, args);
        return true;
    }

    unsafe bool IBatchNormVarianceBackend.TryBatchNormBackwardFromVariance(IGpuBuffer gradOutput, IGpuBuffer input,
        IGpuBuffer gamma, IGpuBuffer saveMean, IGpuBuffer saveVariance, IGpuBuffer gradInput, IGpuBuffer gradGamma,
        IGpuBuffer gradBeta, int batch, int channels, int spatial, float epsilon, IGpuBuffer? reluOutput)
    {
        int k4 = BatchNormRegisterVariant(batch, spatial, out int threads);
        long n = (long)batch * channels * spatial;
        if (k4 == 0 || channels <= 0 || !IsAvailable || gradOutput.Size < n || input.Size < n || gradInput.Size < n
            || gamma.Size < channels || saveMean.Size < channels || saveVariance.Size < channels
            || gradGamma.Size < channels || gradBeta.Size < channels || (reluOutput is not null && reluOutput.Size < n)
            || !_kernelCache.TryGetValue((reluOutput is null ? "batchnorm_bwd_reg" : "batchnorm_bwd_relu_reg") + k4, out var kernel))
            return false;
        using var _ = PushContext();
        IntPtr pro = reluOutput?.Handle ?? IntPtr.Zero;
        IntPtr pdy = gradOutput.Handle, px = input.Handle, pg = gamma.Handle, pm = saveMean.Handle, pv = saveVariance.Handle,
            pdx = gradInput.Handle, pdg = gradGamma.Handle, pdb = gradBeta.Handle;
        int b = batch, c = channels, s = spatial;
        float eps = epsilon;
        void** args = stackalloc void*[13];
        args[0] = &pdy; args[1] = &pro; args[2] = &px; args[3] = &pg; args[4] = &pm; args[5] = &pv; args[6] = &pdx;
        args[7] = &pdg; args[8] = &pdb; args[9] = &b; args[10] = &c; args[11] = &s; args[12] = &eps;
        LaunchKernel(kernel, (uint)channels, (uint)threads, args);
        return true;
    }
}
