// Copyright (c) CrowdTrainer LLC. All rights reserved.
using System;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Kernels;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA;

public sealed partial class CudaBackend : IFusedHouseholderChain
{
    internal static long FusedHouseholderForwardRuns;
    internal static long FusedHouseholderBackwardRuns;
    internal static bool UseFusedHouseholder = Environment.GetEnvironmentVariable("AIDOTNET_CUDA_FUSED_HOUSEHOLDER") != "0";

    void IFusedHouseholderChain.HouseholderChainForward(
        IGpuBuffer input,
        IGpuBuffer reflectors,
        IGpuBuffer output,
        int rows,
        int numReflectors,
        int dim,
        bool transpose,
        double normFloor)
    {
        if (!TryHouseholderChainForward(input, reflectors, output, rows, numReflectors, dim, transpose, normFloor))
            PortableHouseholderChain.Forward(this, input, reflectors, output, rows, numReflectors, dim, transpose, normFloor);
    }

    void IFusedHouseholderChain.HouseholderChainBackward(
        IGpuBuffer gradOutput,
        IGpuBuffer input,
        IGpuBuffer reflectors,
        IGpuBuffer gradInput,
        IGpuBuffer gradReflectors,
        int rows,
        int numReflectors,
        int dim,
        bool transpose,
        double normFloor)
    {
        if (!TryHouseholderChainBackward(gradOutput, input, reflectors, gradInput, gradReflectors, rows, numReflectors, dim, transpose, normFloor))
            PortableHouseholderChain.Backward(this, gradOutput, input, reflectors, gradInput, gradReflectors, rows, numReflectors, dim, transpose, normFloor);
    }

    private unsafe bool TryHouseholderChainForward(
        IGpuBuffer input,
        IGpuBuffer reflectors,
        IGpuBuffer output,
        int rows,
        int numReflectors,
        int dim,
        bool transpose,
        double normFloor)
    {
        if (rows == 0) return true;
        if (!UseFusedHouseholder
            || rows < 0
            || numReflectors < 1 || numReflectors > CudaHouseholderKernels.MaxReflectors
            || dim < 1 || dim > CudaHouseholderKernels.MaxDim
            || numReflectors * dim > CudaHouseholderKernels.MaxForwardReflectorElements
            || !(normFloor > 0.0))
        {
            return false;
        }
        if (!_kernelCache.TryGetValue("householder_chain_fwd", out var kernel))
            return false;

        using var _ = PushContext();
        uint numBlocks = (uint)Math.Min(Math.Max(1, (rows + CudaHouseholderKernels.WarpsPerBlock - 1) / CudaHouseholderKernels.WarpsPerBlock), 512);
        uint smemBytes = (uint)((numReflectors * dim + 4) * sizeof(float));
        IntPtr xPtr = input.Handle;
        IntPtr uPtr = reflectors.Handle;
        IntPtr yPtr = output.Handle;
        int trans = transpose ? 1 : 0;
        float sqrtFloor = (float)Math.Sqrt(normFloor);
        void** args = stackalloc void*[8];
        args[0] = &xPtr;
        args[1] = &uPtr;
        args[2] = &yPtr;
        args[3] = &rows;
        args[4] = &numReflectors;
        args[5] = &dim;
        args[6] = &trans;
        args[7] = &sqrtFloor;
        LaunchKernelWithSharedMem(kernel, numBlocks, CudaHouseholderKernels.BlockThreads, smemBytes, args);
        System.Threading.Interlocked.Increment(ref FusedHouseholderForwardRuns);
        return true;
    }

    private unsafe bool TryHouseholderChainBackward(
        IGpuBuffer gradOutput,
        IGpuBuffer input,
        IGpuBuffer reflectors,
        IGpuBuffer gradInput,
        IGpuBuffer gradReflectors,
        int rows,
        int numReflectors,
        int dim,
        bool transpose,
        double normFloor)
    {
        if (rows == 0)
        {
            Fill(gradReflectors, 0f, checked(numReflectors * dim));
            return true;
        }
        if (!UseFusedHouseholder
            || rows < 0
            || numReflectors < 1 || numReflectors > CudaHouseholderKernels.MaxReflectors
            || dim < 1 || dim > CudaHouseholderKernels.MaxDim
            || numReflectors * dim > CudaHouseholderKernels.MaxBackwardReflectorElements
            || !(normFloor > 0.0))
        {
            return false;
        }
        if (!_kernelCache.TryGetValue("householder_chain_bwd_rows", out var rowsKernel)
            || !_kernelCache.TryGetValue("householder_chain_bwd_reduce", out var reduceKernel))
        {
            return false;
        }

        using var _ = PushContext();
        int numBlocks = Math.Min(Math.Max(1, (rows + CudaHouseholderKernels.WarpsPerBlock - 1) / CudaHouseholderKernels.WarpsPerBlock), 256);
        int kd = checked(numReflectors * dim);
        uint smemBytes = (uint)((5 * kd + 36) * sizeof(float));

        using var partials = AllocateBuffer(checked(numBlocks * kd));
        using var uHatBuf = AllocateBuffer(kd);
        using var metaBuf = AllocateBuffer(checked(2 * numReflectors));

        IntPtr dyPtr = gradOutput.Handle;
        IntPtr xPtr = input.Handle;
        IntPtr uPtr = reflectors.Handle;
        IntPtr dxPtr = gradInput.Handle;
        IntPtr partPtr = partials.Handle;
        IntPtr uHatPtr = uHatBuf.Handle;
        IntPtr metaPtr = metaBuf.Handle;
        int trans = transpose ? 1 : 0;
        float sqrtFloor = (float)Math.Sqrt(normFloor);

        void** rowArgs = stackalloc void*[12];
        rowArgs[0] = &dyPtr;
        rowArgs[1] = &xPtr;
        rowArgs[2] = &uPtr;
        rowArgs[3] = &dxPtr;
        rowArgs[4] = &partPtr;
        rowArgs[5] = &uHatPtr;
        rowArgs[6] = &metaPtr;
        rowArgs[7] = &rows;
        rowArgs[8] = &numReflectors;
        rowArgs[9] = &dim;
        rowArgs[10] = &trans;
        rowArgs[11] = &sqrtFloor;
        LaunchKernelWithSharedMem(rowsKernel, (uint)numBlocks, CudaHouseholderKernels.BlockThreads, smemBytes, rowArgs);

        IntPtr dUPtr = gradReflectors.Handle;
        void** redArgs = stackalloc void*[7];
        redArgs[0] = &partPtr;
        redArgs[1] = &uHatPtr;
        redArgs[2] = &metaPtr;
        redArgs[3] = &dUPtr;
        redArgs[4] = &numBlocks;
        redArgs[5] = &numReflectors;
        redArgs[6] = &dim;
        LaunchKernel(reduceKernel, (uint)numReflectors, CudaHouseholderKernels.ReduceBlockThreads, redArgs);

        System.Threading.Interlocked.Increment(ref FusedHouseholderBackwardRuns);
        return true;
    }
}
