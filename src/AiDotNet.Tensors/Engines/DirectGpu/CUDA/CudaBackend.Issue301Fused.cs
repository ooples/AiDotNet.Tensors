// Copyright (c) AiDotNet. All rights reserved.

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA;

public sealed partial class CudaBackend
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
        if (!_kernelCache.TryGetValue("issue301_fused_lora_forward", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: issue301_fused_lora_forward.");
        using var _ = PushContext();
        uint grid = (uint)((batchSize * outputFeatures + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr pInput = input.Handle, pBase = baseOutput.Handle, pA = loraA.Handle, pB = loraB.Handle, pOut = output.Handle;
        void** args = stackalloc void*[10];
        args[0] = &pInput; args[1] = &pBase; args[2] = &pA; args[3] = &pB; args[4] = &pOut;
        args[5] = &batchSize; args[6] = &inputFeatures; args[7] = &rank; args[8] = &outputFeatures; args[9] = &scaling;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
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
        if (!_kernelCache.TryGetValue("issue301_fused_ddim_step", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: issue301_fused_ddim_step.");
        using var _ = PushContext();
        uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr pX = xT.Handle, pE = epsilonTheta.Handle, pOut = output.Handle;
        void** args = stackalloc void*[6];
        args[0] = &pX; args[1] = &pE; args[2] = &pOut;
        args[3] = &size; args[4] = &alphaBarT; args[5] = &alphaBarTMinus1;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
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
        if (!_kernelCache.TryGetValue("issue301_fused_sparse_linear", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: issue301_fused_sparse_linear.");
        using var _ = PushContext();
        uint grid = (uint)((batchSize * outputFeatures + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr pInput = input.Handle, pCsr = packedCsr.Handle, pValues = sparseValues.Handle, pBias = bias.Handle, pOut = output.Handle;
        void** args = stackalloc void*[11];
        args[0] = &pInput; args[1] = &pCsr; args[2] = &pValues; args[3] = &pBias; args[4] = &pOut;
        args[5] = &batchSize; args[6] = &inputFeatures; args[7] = &outputFeatures; args[8] = &nnz;
        args[9] = &hasBias; args[10] = &activation;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
        Synchronize();
    }
}
