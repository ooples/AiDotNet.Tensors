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

    bool IBatchNormVarianceBackend.AcceptsTrainingShape(int batch, int channels, int spatial)
        => IsAvailable && channels > 0
           && (BatchNormRegisterVariant(batch, spatial, out _) != 0 || BatchNormSplitCount(batch, channels, spatial) != 0);

    // ---- Split variant: channel slices over the register variant's 8192 elements ----
    // AIDOTNET_BN_SPLIT=0 sends those slices back to the generic BatchNorm / BatchNormBackward kernels (A/B, tests).
    internal static bool s_batchNormSplit = Environment.GetEnvironmentVariable("AIDOTNET_BN_SPLIT") != "0";
    /// <summary>Split-variant launches, forward and backward (tests).</summary>
    internal static long BatchNormSplitLaunches;
    private const int BatchNormSplitFloat4PerBlock = 1024;   // 4 float4s per thread at 256 threads
    private const int BatchNormSplitMax = 128;
    private IGpuBuffer? _batchNormSplitPartials;
    private readonly object _batchNormSplitLock = new();

    /// <summary>Chunks per channel for the split variant, or 0 when the register variant takes the slice (or the shape
    /// is outside both: spatial not a multiple of four).</summary>
    private static int BatchNormSplitCount(int batch, int channels, int spatial)
    {
        if (!s_batchNormSplit || batch <= 0 || channels <= 0 || spatial <= 0 || (spatial & 3) != 0) return 0;
        long units = (long)batch * (spatial >> 2);
        if (units <= (long)BatchNormRegisterThreads * BatchNormRegisterMaxFloat4PerThread || units > int.MaxValue) return 0;
        long splits = Math.Min(BatchNormSplitMax, (units + BatchNormSplitFloat4PerBlock - 1) / BatchNormSplitFloat4PerBlock);
        if ((long)channels * splits * 3 > int.MaxValue) return 0;
        return (int)splits;
    }

    // The partial-sum scratch, shared by every split launch (they run in stream order). It grows only outside a stream
    // capture, which cannot allocate; a capture that needs more declines and its caller takes the generic kernels.
    private IGpuBuffer? EnsureBatchNormSplitPartials(int floats)
    {
        if (_batchNormSplitPartials is { } held && held.Size >= floats) return held;
        if (IsStreamCapturing()) return null;
        int size = Math.Max(floats, _batchNormSplitPartials is null ? floats : (int)Math.Min(int.MaxValue, 2L * _batchNormSplitPartials.Size));
        _batchNormSplitPartials?.Dispose();
        _batchNormSplitPartials = AllocateBuffer(size);
        return _batchNormSplitPartials;
    }

    private unsafe bool TryBatchNormSplitForward(IGpuBuffer input, IGpuBuffer output, IGpuBuffer gamma, IGpuBuffer beta,
        IGpuBuffer saveMean, IGpuBuffer saveVariance, int batch, int channels, int spatial, float epsilon, bool relu,
        IGpuBuffer? residual)
    {
        int splits = BatchNormSplitCount(batch, channels, spatial);
        string name = residual is not null ? "batchnorm_fwd_addrelu_split" : relu ? "batchnorm_fwd_relu_split" : "batchnorm_fwd_split";
        if (splits == 0 || !_kernelCache.TryGetValue("batchnorm_split_stats", out var stats)
            || !_kernelCache.TryGetValue(name, out var apply))
            return false;
        lock (_batchNormSplitLock)
        {
            if (EnsureBatchNormSplitPartials(channels * splits * 3) is not { } partials) return false;
            using var _ = PushContext();
            IntPtr pi = input.Handle, pr = residual?.Handle ?? IntPtr.Zero, po = output.Handle, pg = gamma.Handle,
                pb = beta.Handle, pm = saveMean.Handle, pv = saveVariance.Handle, pp = partials.Handle;
            int b = batch, c = channels, s = spatial, n = splits;
            float eps = epsilon;
            void** statArgs = stackalloc void*[6];
            statArgs[0] = &pi; statArgs[1] = &pp; statArgs[2] = &b; statArgs[3] = &c; statArgs[4] = &s; statArgs[5] = &n;
            LaunchKernel2D(stats, (uint)channels, (uint)splits, 1, 256, 1, statArgs);
            void** args = stackalloc void*[13];
            args[0] = &pi; args[1] = &pr; args[2] = &po; args[3] = &pg; args[4] = &pb; args[5] = &pm; args[6] = &pv;
            args[7] = &pp; args[8] = &b; args[9] = &c; args[10] = &s; args[11] = &eps; args[12] = &n;
            LaunchKernel2D(apply, (uint)channels, (uint)splits, 1, 256, 1, args);
        }
        System.Threading.Interlocked.Increment(ref BatchNormSplitLaunches);
        return true;
    }

    private unsafe bool TryBatchNormSplitBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gamma,
        IGpuBuffer saveMean, IGpuBuffer saveVariance, IGpuBuffer gradInput, IGpuBuffer gradGamma, IGpuBuffer gradBeta,
        int batch, int channels, int spatial, float epsilon, IGpuBuffer? reluOutput, IGpuBuffer? gradResidual,
        bool rebuildMask, IGpuBuffer? beta, IGpuBuffer? gradOutput2)
    {
        int splits = BatchNormSplitCount(batch, channels, spatial);
        string prefix = gradResidual is not null ? "batchnorm_bwd_addrelu_split" : reluOutput is null ? "batchnorm_bwd_split"
            : rebuildMask ? "batchnorm_bwd_relumask_split" : "batchnorm_bwd_relu_split";
        if (splits == 0 || !_kernelCache.TryGetValue(prefix + "_reduce", out var reduce)
            || !_kernelCache.TryGetValue(prefix + "_apply", out var apply))
            return false;
        lock (_batchNormSplitLock)
        {
            if (EnsureBatchNormSplitPartials(channels * splits * 2) is not { } partials) return false;
            using var _ = PushContext();
            IntPtr pdy = gradOutput.Handle, pms = rebuildMask ? beta!.Handle : reluOutput?.Handle ?? IntPtr.Zero,
                pdy2 = gradOutput2?.Handle ?? IntPtr.Zero, pgr = gradResidual?.Handle ?? IntPtr.Zero, px = input.Handle,
                pg = gamma.Handle, pm = saveMean.Handle, pv = saveVariance.Handle, pdx = gradInput.Handle,
                pdg = gradGamma.Handle, pdb = gradBeta.Handle, pp = partials.Handle;
            int b = batch, c = channels, s = spatial, n = splits;
            float eps = epsilon;
            void** reduceArgs = stackalloc void*[13];
            reduceArgs[0] = &pdy; reduceArgs[1] = &pms; reduceArgs[2] = &pdy2; reduceArgs[3] = &px; reduceArgs[4] = &pg;
            reduceArgs[5] = &pm; reduceArgs[6] = &pv; reduceArgs[7] = &pp; reduceArgs[8] = &b; reduceArgs[9] = &c;
            reduceArgs[10] = &s; reduceArgs[11] = &eps; reduceArgs[12] = &n;
            LaunchKernel2D(reduce, (uint)channels, (uint)splits, 1, 256, 1, reduceArgs);
            void** args = stackalloc void*[17];
            args[0] = &pdy; args[1] = &pms; args[2] = &pdy2; args[3] = &pgr; args[4] = &px; args[5] = &pg; args[6] = &pm;
            args[7] = &pv; args[8] = &pdx; args[9] = &pdg; args[10] = &pdb; args[11] = &pp; args[12] = &b; args[13] = &c;
            args[14] = &s; args[15] = &eps; args[16] = &n;
            LaunchKernel2D(apply, (uint)channels, (uint)splits, 1, 256, 1, args);
        }
        System.Threading.Interlocked.Increment(ref BatchNormSplitLaunches);
        if (rebuildMask) System.Threading.Interlocked.Increment(ref BatchNormMaskFromInputLaunches);
        return true;
    }

    unsafe bool IBatchNormVarianceBackend.TryBatchNormTrainingWithVariance(IGpuBuffer input, IGpuBuffer output,
        IGpuBuffer gamma, IGpuBuffer beta, IGpuBuffer saveMean, IGpuBuffer saveVariance,
        int batch, int channels, int spatial, float epsilon, bool relu, IGpuBuffer? residual)
    {
        int k4 = BatchNormRegisterVariant(batch, spatial, out int threads);
        long n = (long)batch * channels * spatial;
        if (k4 == 0 && channels > 0 && IsAvailable && input.Size >= n && output.Size >= n && gamma.Size >= channels
            && beta.Size >= channels && saveMean.Size >= channels && saveVariance.Size >= channels
            && (residual is null || (relu && residual.Size >= n)))
            return TryBatchNormSplitForward(input, output, gamma, beta, saveMean, saveVariance, batch, channels, spatial,
                epsilon, relu, residual);
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
        if (k4 == 0 && channels > 0 && IsAvailable && gradOutput.Size >= n && input.Size >= n && gradInput.Size >= n
            && gamma.Size >= channels && saveMean.Size >= channels && saveVariance.Size >= channels
            && gradGamma.Size >= channels && gradBeta.Size >= channels && (reluOutput is null || reluOutput.Size >= n)
            && (gradOutput2 is null || gradOutput2.Size >= n)
            && (gradResidual is null || (reluOutput is not null && gradResidual.Size >= n)))
            return TryBatchNormSplitBackward(gradOutput, input, gamma, saveMean, saveVariance, gradInput, gradGamma,
                gradBeta, batch, channels, spatial, epsilon, reluOutput, gradResidual, rebuildMask, beta, gradOutput2);
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
