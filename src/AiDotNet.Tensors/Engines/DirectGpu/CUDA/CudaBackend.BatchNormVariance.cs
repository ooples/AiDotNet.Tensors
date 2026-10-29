// Copyright (c) CrowdTrainer LLC. All rights reserved.
using System;
namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA;

public sealed partial class CudaBackend : IBatchNormVarianceBackend
{
    private const int BatchNormRegisterMaxFloat4PerThread = 8;
    private const int BatchNormRegisterThreads = 256;

    // AIDOTNET_BN_MASK_FROM_INPUT=0 keeps reading the ReLU output in the fused BatchNorm+ReLU backward (A/B, tests).
    internal static bool s_batchNormMaskFromInput = Environment.GetEnvironmentVariable("AIDOTNET_BN_MASK_FROM_INPUT") != "0";

    /// <summary>Fused BatchNorm+ReLU backward launches that rebuilt the mask from the input (tests).</summary>
    internal static long BatchNormMaskFromInputLaunches;

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
        int batch, int channels, int spatial, float epsilon, bool relu, IGpuBuffer? residual)
    {
        int k4 = BatchNormRegisterVariant(batch, spatial, out int threads);
        long n = (long)batch * channels * spatial;
        string name = residual is not null ? "batchnorm_fwd_addrelu_reg" : relu ? "batchnorm_fwd_relu_reg" : "batchnorm_fwd_reg";
        if (k4 == 0 || channels <= 0 || !IsAvailable || input.Size < n || output.Size < n || gamma.Size < channels
            || beta.Size < channels || saveMean.Size < channels || saveVariance.Size < channels
            || (residual is not null && (!relu || residual.Size < n))
            || !_kernelCache.TryGetValue(name + k4, out var kernel))
            return false;
        using var _ = PushContext();
        IntPtr pr = residual?.Handle ?? IntPtr.Zero;
        IntPtr pi = input.Handle, po = output.Handle, pg = gamma.Handle, pb = beta.Handle, pm = saveMean.Handle,
            pv = saveVariance.Handle;
        int b = batch, c = channels, s = spatial;
        float eps = epsilon;
        void** args = stackalloc void*[11];
        args[0] = &pi; args[1] = &pr; args[2] = &po; args[3] = &pg; args[4] = &pb; args[5] = &pm; args[6] = &pv;
        args[7] = &b; args[8] = &c; args[9] = &s; args[10] = &eps;
        LaunchKernel(kernel, (uint)channels, (uint)threads, args);
        return true;
    }

    unsafe bool IBatchNormVarianceBackend.TryBatchNormBackwardFromVariance(IGpuBuffer gradOutput, IGpuBuffer input,
        IGpuBuffer gamma, IGpuBuffer saveMean, IGpuBuffer saveVariance, IGpuBuffer gradInput, IGpuBuffer gradGamma,
        IGpuBuffer gradBeta, int batch, int channels, int spatial, float epsilon, IGpuBuffer? reluOutput, IGpuBuffer? gradResidual,
        IGpuBuffer? beta, IGpuBuffer? gradOutput2)
    {
        // BN -> ReLU without a residual: rebuild the mask from x (one activation read fewer) when beta is supplied.
        bool rebuildMask = s_batchNormMaskFromInput && reluOutput is not null && gradResidual is null && beta is not null
                           && beta.Size >= channels;
        int k4 = BatchNormRegisterVariant(batch, spatial, out int threads);
        long n = (long)batch * channels * spatial;
        if (k4 == 0 || channels <= 0 || !IsAvailable || gradOutput.Size < n || input.Size < n || gradInput.Size < n
            || gamma.Size < channels || saveMean.Size < channels || saveVariance.Size < channels
            || gradGamma.Size < channels || gradBeta.Size < channels || (reluOutput is not null && reluOutput.Size < n)
            || (gradOutput2 is not null && gradOutput2.Size < n)
            || (gradResidual is not null && (reluOutput is null || gradResidual.Size < n))
            || !_kernelCache.TryGetValue((gradResidual is not null ? "batchnorm_bwd_addrelu_reg"
                : reluOutput is null ? "batchnorm_bwd_reg" : rebuildMask ? "batchnorm_bwd_relumask_reg" : "batchnorm_bwd_relu_reg") + k4,
                out var kernel))
            return false;
        using var _ = PushContext();
        IntPtr pro = rebuildMask ? beta!.Handle : reluOutput?.Handle ?? IntPtr.Zero, pgr = gradResidual?.Handle ?? IntPtr.Zero;
        IntPtr pdy = gradOutput.Handle, px = input.Handle, pg = gamma.Handle, pm = saveMean.Handle, pv = saveVariance.Handle,
            pdx = gradInput.Handle, pdg = gradGamma.Handle, pdb = gradBeta.Handle;
        int b = batch, c = channels, s = spatial;
        float eps = epsilon;
        IntPtr pdy2 = gradOutput2?.Handle ?? IntPtr.Zero;
        void** args = stackalloc void*[15];
        args[0] = &pdy; args[1] = &pro; args[2] = &pgr; args[3] = &px; args[4] = &pg; args[5] = &pm; args[6] = &pv;
        args[7] = &pdx; args[8] = &pdg; args[9] = &pdb; args[10] = &b; args[11] = &c; args[12] = &s; args[13] = &eps;
        args[14] = &pdy2;
        LaunchKernel(kernel, (uint)channels, (uint)threads, args);
        if (rebuildMask) System.Threading.Interlocked.Increment(ref BatchNormMaskFromInputLaunches);
        return true;
    }
}
