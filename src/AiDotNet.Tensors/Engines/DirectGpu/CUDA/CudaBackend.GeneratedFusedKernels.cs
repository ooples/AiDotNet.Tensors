// Copyright (c) CrowdTrainer LLC. All rights reserved.

using System;
using AiDotNet.Tensors.Engines.Compilation.FusionTileIr;
using AiDotNet.Tensors.Engines.Compilation.GeneratedKernels;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA;

public sealed partial class CudaBackend
{
    private int _generatedRowFusedModuleCompiled;
    private int _crossLayerNormRopeModuleCompiled;
    private readonly object _generatedRowFusedLock = new();

    private bool EnsureGeneratedRowFusedKernelsLoaded()
    {
        if (System.Threading.Volatile.Read(ref _generatedRowFusedModuleCompiled) != 0)
            return true;
        if (!IsAvailable || IsStreamCapturing())
            return false;

        lock (_generatedRowFusedLock)
        {
            if (_generatedRowFusedModuleCompiled != 0)
                return true;
            using var _ = PushContext();
            CuBlasNative.CheckCudaResult(CuBlasNative.cuDeviceGet(out int cudaDevice, DeviceOrdinal), "cuDeviceGet(generated row fusions)");
            CompileKernelModule(
                cudaDevice,
                CompiledFusedKernels.UniversalRowFusedCudaSource,
                CompiledFusedKernels.UniversalRowFusedCudaModuleName,
                CompiledFusedKernels.UniversalRowFusedCudaKernelNames,
                useFastMath: false);
            System.Threading.Volatile.Write(ref _generatedRowFusedModuleCompiled, 1);
            return true;
        }
    }

    /// <summary>
    /// Launches the compile-time generated fused residual-add + RMSNorm warp-shuffle CUDA kernel in a single launch.
    /// </summary>
    internal unsafe bool TryLaunchFusedRmsNormResidual(
        IGpuBuffer input,
        IGpuBuffer residual,
        IGpuBuffer gamma,
        IGpuBuffer output,
        IGpuBuffer rmsOut,
        int rows,
        int cols,
        float epsilon)
    {
        if (rows <= 0 || cols <= 0 || !EnsureGeneratedRowFusedKernelsLoaded())
            return false;
        if (!_kernelCache.TryGetValue("fused_rmsnorm_residual_f32", out IntPtr kernel))
            return false;

        int total = checked(rows * cols);
        GpuBufferCapacity.Require(input, total, nameof(input));
        GpuBufferCapacity.Require(residual, total, nameof(residual));
        GpuBufferCapacity.Require(gamma, cols, nameof(gamma));
        GpuBufferCapacity.Require(output, total, nameof(output));
        GpuBufferCapacity.Require(rmsOut, rows, nameof(rmsOut));

        IntPtr pX = input.Handle;
        IntPtr pRes = residual.Handle;
        IntPtr pGamma = gamma.Handle;
        IntPtr pOut = output.Handle;
        IntPtr pRms = rmsOut.Handle;
        int r = rows;
        int c = cols;
        float eps = epsilon;

        void** args = stackalloc void*[8];
        args[0] = &pX;
        args[1] = &pRes;
        args[2] = &pGamma;
        args[3] = &pOut;
        args[4] = &pRms;
        args[5] = &r;
        args[6] = &c;
        args[7] = &eps;

        uint blockX = SelectRowReductionBlockSize(cols);
        using (PushContext())
        {
            lock (GpuDispatchLock)
            {
                LaunchKernel3D(kernel, (uint)rows, 1, 1, blockX, 1, 1, args, 0);
            }
        }
        return true;
    }

    /// <summary>
    /// Launches the compile-time generated fused residual-add + LayerNorm warp-shuffle CUDA kernel in a single launch.
    /// </summary>
    internal unsafe bool TryLaunchFusedLayerNormResidual(
        IGpuBuffer input,
        IGpuBuffer residual,
        IGpuBuffer gamma,
        IGpuBuffer beta,
        IGpuBuffer output,
        IGpuBuffer meanOut,
        IGpuBuffer invStdOut,
        int rows,
        int cols,
        float epsilon)
    {
        if (rows <= 0 || cols <= 0 || !EnsureGeneratedRowFusedKernelsLoaded())
            return false;
        if (!_kernelCache.TryGetValue("fused_layernorm_residual_f32", out IntPtr kernel))
            return false;

        int total = checked(rows * cols);
        GpuBufferCapacity.Require(input, total, nameof(input));
        GpuBufferCapacity.Require(residual, total, nameof(residual));
        GpuBufferCapacity.Require(gamma, cols, nameof(gamma));
        GpuBufferCapacity.Require(beta, cols, nameof(beta));
        GpuBufferCapacity.Require(output, total, nameof(output));
        GpuBufferCapacity.Require(meanOut, rows, nameof(meanOut));
        GpuBufferCapacity.Require(invStdOut, rows, nameof(invStdOut));

        IntPtr pX = input.Handle;
        IntPtr pRes = residual.Handle;
        IntPtr pGamma = gamma.Handle;
        IntPtr pBeta = beta.Handle;
        IntPtr pOut = output.Handle;
        IntPtr pMean = meanOut.Handle;
        IntPtr pInvStd = invStdOut.Handle;
        int r = rows;
        int c = cols;
        float eps = epsilon;

        void** args = stackalloc void*[10];
        args[0] = &pX;
        args[1] = &pRes;
        args[2] = &pGamma;
        args[3] = &pBeta;
        args[4] = &pOut;
        args[5] = &pMean;
        args[6] = &pInvStd;
        args[7] = &r;
        args[8] = &c;
        args[9] = &eps;

        uint blockX = SelectRowReductionBlockSize(cols);
        using (PushContext())
        {
            lock (GpuDispatchLock)
            {
                LaunchKernel3D(kernel, (uint)rows, 1, 1, blockX, 1, 1, args, 0);
            }
        }
        return true;
    }

    private static uint SelectRowReductionBlockSize(int cols)
    {
        if (cols <= 64) return 64;
        if (cols <= 128) return 128;
        if (cols <= 256) return 256;
        if (cols <= 512) return 512;
        return 1024;
    }

    internal static bool TryDispatchRmsNormResidual(
        IDirectGpuBackend backend,
        IGpuBuffer input,
        IGpuBuffer residual,
        IGpuBuffer gamma,
        IGpuBuffer output,
        IGpuBuffer rmsOut,
        int rows,
        int cols,
        float epsilon)
        => backend is CudaBackend cuda
           && cuda.TryLaunchFusedRmsNormResidual(input, residual, gamma, output, rmsOut, rows, cols, epsilon);

    internal static bool TryDispatchLayerNormResidual(
        IDirectGpuBackend backend,
        IGpuBuffer input,
        IGpuBuffer residual,
        IGpuBuffer gamma,
        IGpuBuffer beta,
        IGpuBuffer output,
        IGpuBuffer meanOut,
        IGpuBuffer invStdOut,
        int rows,
        int cols,
        float epsilon)
        => backend is CudaBackend cuda
           && cuda.TryLaunchFusedLayerNormResidual(input, residual, gamma, beta, output, meanOut, invStdOut, rows, cols, epsilon);

    internal static unsafe bool TryDispatchRotarySplitHalf(
        IDirectGpuBackend backend,
        IGpuBuffer input,
        IGpuBuffer cos,
        IGpuBuffer sin,
        IGpuBuffer output,
        int outerCount,
        int seqLen,
        int headDim,
        int trigStride,
        bool inverseAdjoint)
    {
        if (backend is not CudaBackend cuda || outerCount <= 0 || seqLen <= 0 || headDim <= 0 || (headDim & 1) != 0)
            return false;
        if (!cuda.EnsureGeneratedRowFusedKernelsLoaded() || !cuda._kernelCache.TryGetValue("fused_rope_splithalf_f32", out IntPtr kernel))
            return false;

        int total = checked(outerCount * seqLen * headDim);
        int halfDim = headDim >> 1;
        GpuBufferCapacity.Require(input, total, nameof(input));
        GpuBufferCapacity.Require(output, total, nameof(output));
        GpuBufferCapacity.Require(cos, checked(seqLen * halfDim), nameof(cos));
        GpuBufferCapacity.Require(sin, checked(seqLen * halfDim), nameof(sin));

        IntPtr pX = input.Handle;
        IntPtr pCos = cos.Handle;
        IntPtr pSin = sin.Handle;
        IntPtr pOut = output.Handle;
        int o = outerCount;
        int s = seqLen;
        int d = headDim;
        int ts = trigStride;
        float sign = inverseAdjoint ? -1.0f : 1.0f;

        void** args = stackalloc void*[9];
        args[0] = &pX;
        args[1] = &pCos;
        args[2] = &pSin;
        args[3] = &pOut;
        args[4] = &o;
        args[5] = &s;
        args[6] = &d;
        args[7] = &ts;
        args[8] = &sign;

        int totalPairs = total >> 1;
        uint blockX = 256;
        uint gridX = (uint)Math.Min(4096, (totalPairs + (int)blockX - 1) / (int)blockX);
        if (gridX == 0) gridX = 1;

        using (cuda.PushContext())
        {
            lock (GpuDispatchLock)
            {
                cuda.LaunchKernel3D(kernel, gridX, 1, 1, blockX, 1, 1, args, 0);
            }
        }
        return true;
    }

    private bool EnsureCrossLayerNormRopeKernelLoaded()
    {
        if (System.Threading.Volatile.Read(ref _crossLayerNormRopeModuleCompiled) != 0)
            return true;
        if (!IsAvailable || IsStreamCapturing())
            return false;

        lock (_generatedRowFusedLock)
        {
            if (_crossLayerNormRopeModuleCompiled != 0)
                return true;
            using var _ = PushContext();
            CuBlasNative.CheckCudaResult(CuBlasNative.cuDeviceGet(out int cudaDevice, DeviceOrdinal), "cuDeviceGet(cross-layer norm+rope)");
            var tile = FusionTileLinker.CreateCrossLayerNormResidualRopeTile(useLayerNorm: false);
            CompileKernelModule(
                cudaDevice,
                tile.CudaDeviceFragment,
                "aidn_crosslayer_norm_rope",
                new[] { "fused_rmsnorm_residual_rope_f32" },
                useFastMath: false);
            System.Threading.Volatile.Write(ref _crossLayerNormRopeModuleCompiled, 1);
            return true;
        }
    }

    internal static unsafe bool TryDispatchRmsNormResidualRope(
        IDirectGpuBackend backend,
        IGpuBuffer input,
        IGpuBuffer residual,
        IGpuBuffer gamma,
        IGpuBuffer cos,
        IGpuBuffer sin,
        IGpuBuffer ropeOut,
        int outerCount,
        int seqLen,
        int headDim,
        int trigStride,
        float epsilon)
    {
        if (backend is not CudaBackend cuda || outerCount <= 0 || seqLen <= 0 || headDim <= 0 || headDim > 2048 || (headDim & 1) != 0)
            return false;
        if (!cuda.EnsureCrossLayerNormRopeKernelLoaded()
            || !cuda._kernelCache.TryGetValue("fused_rmsnorm_residual_rope_f32", out IntPtr kernel))
        {
            return false;
        }

        int total = checked(outerCount * seqLen * headDim);
        int halfDim = headDim >> 1;
        GpuBufferCapacity.Require(input, total, nameof(input));
        GpuBufferCapacity.Require(residual, total, nameof(residual));
        GpuBufferCapacity.Require(gamma, headDim, nameof(gamma));
        GpuBufferCapacity.Require(cos, checked(seqLen * trigStride), nameof(cos));
        GpuBufferCapacity.Require(sin, checked(seqLen * trigStride), nameof(sin));
        GpuBufferCapacity.Require(ropeOut, total, nameof(ropeOut));

        IntPtr pX = input.Handle;
        IntPtr pRes = residual.Handle;
        IntPtr pGamma = gamma.Handle;
        IntPtr pCos = cos.Handle;
        IntPtr pSin = sin.Handle;
        IntPtr pOut = ropeOut.Handle;
        IntPtr pResSumNull = IntPtr.Zero;
        int o = outerCount;
        int s = seqLen;
        int d = headDim;
        int ts = trigStride;
        float eps = epsilon;

        void** args = stackalloc void*[12];
        args[0] = &pX;
        args[1] = &pRes;
        args[2] = &pGamma;
        args[3] = &pCos;
        args[4] = &pSin;
        args[5] = &pOut;
        args[6] = &pResSumNull;
        args[7] = &o;
        args[8] = &s;
        args[9] = &d;
        args[10] = &ts;
        args[11] = &eps;

        uint blockX = SelectRowReductionBlockSize(headDim);
        uint rows = (uint)checked(outerCount * seqLen);
        uint sharedMemBytes = (uint)(headDim * sizeof(float));

        using (cuda.PushContext())
        {
            lock (GpuDispatchLock)
            {
                cuda.LaunchKernel3D(kernel, rows, 1, 1, blockX, 1, 1, args, sharedMemBytes);
            }
        }
        return true;
    }
}
