// Copyright (c) AiDotNet. All rights reserved.

namespace AiDotNet.Tensors.Engines.DirectGpu.Metal;

public sealed partial class MetalBackend
{
    public void FusedLoRAForward(
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
        DispatchIssue301LoRA(input, baseOutput, loraA, loraB, output,
            batchSize, inputFeatures, rank, outputFeatures, scaling);
    }

    public void FusedDDIMStep(
        IGpuBuffer xT,
        IGpuBuffer epsilonTheta,
        IGpuBuffer output,
        int size,
        float alphaBarT,
        float alphaBarTMinus1)
    {
        ThrowIfDisposed();
        if (size <= 0) return;
        if (xT is not MetalGpuBuffer xBuffer ||
            epsilonTheta is not MetalGpuBuffer eBuffer ||
            output is not MetalGpuBuffer oBuffer)
        {
            throw new ArgumentException("Buffers must be MetalGpuBuffer");
        }

        var pipeline = GetPipeline("Issue301Fused", _issue301FusedLibrary, "issue301_fused_ddim_step");
        var (threadgroups, threadsPerGroup) = pipeline.Calculate1DDispatch(size);
        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        encoder.SetBuffer(xBuffer, 0);
        encoder.SetBuffer(eBuffer, 1);
        encoder.SetBuffer(oBuffer, 2);
        encoder.SetBytes((uint)size, 3);
        encoder.SetBytes(alphaBarT, 4);
        encoder.SetBytes(alphaBarTMinus1, 5);
        encoder.DispatchThreadgroups(threadgroups, threadsPerGroup);
    }

    public void FusedSparseLinear(
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
        ThrowIfDisposed();
        int total = batchSize * outputFeatures;
        if (total <= 0) return;
        if (input is not MetalGpuBuffer iBuffer ||
            packedCsr is not MetalGpuBuffer csrBuffer ||
            sparseValues is not MetalGpuBuffer vBuffer ||
            bias is not MetalGpuBuffer bBuffer ||
            output is not MetalGpuBuffer oBuffer)
        {
            throw new ArgumentException("Buffers must be MetalGpuBuffer");
        }

        var pipeline = GetPipeline("Issue301Fused", _issue301FusedLibrary, "issue301_fused_sparse_linear");
        var (threadgroups, threadsPerGroup) = pipeline.Calculate1DDispatch(total);
        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        encoder.SetBuffer(iBuffer, 0);
        encoder.SetBuffer(csrBuffer, 1);
        encoder.SetBuffer(vBuffer, 2);
        encoder.SetBuffer(bBuffer, 3);
        encoder.SetBuffer(oBuffer, 4);
        encoder.SetBytes((uint)batchSize, 5);
        encoder.SetBytes((uint)inputFeatures, 6);
        encoder.SetBytes((uint)outputFeatures, 7);
        encoder.SetBytes((uint)nnz, 8);
        encoder.SetBytes((uint)hasBias, 9);
        encoder.SetBytes((uint)activation, 10);
        encoder.DispatchThreadgroups(threadgroups, threadsPerGroup);
    }

    private void DispatchIssue301LoRA(
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
        ThrowIfDisposed();
        int total = batchSize * outputFeatures;
        if (total <= 0) return;
        if (input is not MetalGpuBuffer iBuffer ||
            baseOutput is not MetalGpuBuffer baseBuffer ||
            loraA is not MetalGpuBuffer aBuffer ||
            loraB is not MetalGpuBuffer bBuffer ||
            output is not MetalGpuBuffer oBuffer)
        {
            throw new ArgumentException("Buffers must be MetalGpuBuffer");
        }

        var pipeline = GetPipeline("Issue301Fused", _issue301FusedLibrary, "issue301_fused_lora_forward");
        var (threadgroups, threadsPerGroup) = pipeline.Calculate1DDispatch(total);
        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        encoder.SetBuffer(iBuffer, 0);
        encoder.SetBuffer(baseBuffer, 1);
        encoder.SetBuffer(aBuffer, 2);
        encoder.SetBuffer(bBuffer, 3);
        encoder.SetBuffer(oBuffer, 4);
        encoder.SetBytes((uint)batchSize, 5);
        encoder.SetBytes((uint)inputFeatures, 6);
        encoder.SetBytes((uint)rank, 7);
        encoder.SetBytes((uint)outputFeatures, 8);
        encoder.SetBytes(scaling, 9);
        encoder.DispatchThreadgroups(threadgroups, threadsPerGroup);
    }
}
