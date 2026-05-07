// Copyright (c) AiDotNet. All rights reserved.

using System;

namespace AiDotNet.Tensors.Engines.DirectGpu.HIP;

public sealed partial class HipBackend
{
    public unsafe void FusedLoRAForward(
        IGpuBuffer input,
        IGpuBuffer baseOutput,
        IGpuBuffer loraA,
        IGpuBuffer loraB,
        IGpuBuffer output,
        int batchSize,
        int inputFeatures,
        int rank,
        int outputFeatures,
        float scaling)
    {
        if (!_kernelCache.TryGetValue("fused_lora_forward", out var kernel))
            throw new InvalidOperationException("HIP kernel not found: fused_lora_forward");

        // Two-stage kernel: one block per batch row, dynamic shared memory
        // holds proj[rank]. See CudaFusedAdvancedKernels.cs for the design.
        uint blockX = (uint)System.Math.Min(System.Math.Max(outputFeatures, 1), (int)DefaultBlockSize);
        uint grid = (uint)System.Math.Max(batchSize, 1);
        uint sharedMemBytes = checked((uint)rank * sizeof(float));

        IntPtr pInput = input.Handle, pBase = baseOutput.Handle, pA = loraA.Handle, pB = loraB.Handle, pOut = output.Handle;
        void** args = stackalloc void*[10];
        args[0] = &pInput; args[1] = &pBase; args[2] = &pA; args[3] = &pB; args[4] = &pOut;
        args[5] = &batchSize; args[6] = &inputFeatures; args[7] = &rank; args[8] = &outputFeatures; args[9] = &scaling;
        LaunchKernelWithSharedMem(kernel, grid, blockX, sharedMemBytes, args);
        Synchronize();
    }

    public unsafe void FusedDDIMStep(
        IGpuBuffer xT,
        IGpuBuffer epsilonTheta,
        IGpuBuffer output,
        int size,
        float alphaBarT,
        float alphaBarTMinus1)
    {
        if (!_kernelCache.TryGetValue("fused_ddim_step", out var kernel))
            throw new InvalidOperationException("HIP kernel not found: fused_ddim_step");
        // size==0 → 0-block grid, which HIP rejects with hipErrorInvalidConfiguration.
        // alphaBarT must be > 0 because the kernel divides by sqrt(alphaBarT);
        // ᾱ=0 produces ±Inf coefficients that NaN-poison every element.
        if (size <= 0) return;
        if (!(alphaBarT > 0f && alphaBarT <= 1f))
            throw new ArgumentOutOfRangeException(nameof(alphaBarT),
                $"alphaBarT must be in (0, 1]; got {alphaBarT}.");
        if (!(alphaBarTMinus1 >= 0f && alphaBarTMinus1 <= 1f))
            throw new ArgumentOutOfRangeException(nameof(alphaBarTMinus1),
                $"alphaBarTMinus1 must be in [0, 1]; got {alphaBarTMinus1}.");
        IntPtr pX = xT.Handle, pE = epsilonTheta.Handle, pOut = output.Handle;
        void** args = stackalloc void*[6];
        args[0] = &pX; args[1] = &pE; args[2] = &pOut;
        args[3] = &size; args[4] = &alphaBarT; args[5] = &alphaBarTMinus1;
        LaunchKernel(kernel, (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize), DefaultBlockSize, args);
        Synchronize();
    }

    public unsafe void FusedSparseLinear(
        IGpuBuffer input,
        IGpuBuffer packedCsr,
        IGpuBuffer sparseValues,
        IGpuBuffer bias,
        IGpuBuffer output,
        int batchSize,
        int inputFeatures,
        int outputFeatures,
        int nnz,
        int hasBias,
        int activation)
    {
        if (!_kernelCache.TryGetValue("fused_sparse_linear", out var kernel))
            throw new InvalidOperationException("HIP kernel not found: fused_sparse_linear");

        // Reject negative dims and 64-bit overflow before computing the grid.
        // Plain `int` multiplication for batchSize * outputFeatures wraps
        // silently to a corrupt grid count when the product exceeds
        // int.MaxValue (~2.1B), which HIP either rejects or maps to OOB
        // memory access. Promote to long, validate, then cast.
        if (batchSize <= 0 || outputFeatures <= 0) return;
        long totalLong = (long)batchSize * outputFeatures;
        if (totalLong > int.MaxValue)
            throw new ArgumentOutOfRangeException(nameof(batchSize),
                $"batchSize ({batchSize}) * outputFeatures ({outputFeatures}) = {totalLong} " +
                "exceeds int.MaxValue; split the dispatch into chunks.");
        int total = (int)totalLong;

        IntPtr pInput = input.Handle, pCsr = packedCsr.Handle, pValues = sparseValues.Handle, pBias = bias.Handle, pOut = output.Handle;
        void** args = stackalloc void*[11];
        args[0] = &pInput; args[1] = &pCsr; args[2] = &pValues; args[3] = &pBias; args[4] = &pOut;
        args[5] = &batchSize; args[6] = &inputFeatures; args[7] = &outputFeatures; args[8] = &nnz;
        args[9] = &hasBias; args[10] = &activation;
        LaunchKernel(kernel, (uint)((total + DefaultBlockSize - 1) / DefaultBlockSize), DefaultBlockSize, args);
        Synchronize();
    }
}
