// Copyright (c) AiDotNet. All rights reserved.

using System;
using static AiDotNet.Tensors.Engines.DirectGpu.Metal.MetalNativeBindings;

namespace AiDotNet.Tensors.Engines.DirectGpu.Metal;

public sealed partial class MetalBackend
{
    private void EnsureFusedAdvancedLibraryLoaded(string opName)
    {
        // Pattern matches MetalBackend's other optional libraries (Roi,
        // Linalg, Geometry): the constructor catches compilation failures
        // and leaves the library handle as IntPtr.Zero. Calling GetPipeline
        // with a null library handle would deref into native code; refuse
        // dispatch with a clear NotSupportedException at the call site.
        if (_fusedAdvancedLibrary == IntPtr.Zero)
            throw new NotSupportedException(
                $"Metal fused-advanced kernels are not available — the kernel " +
                $"library failed to compile during backend initialization. " +
                $"{opName} cannot be dispatched. Fall back to the eager decomposed path.");
    }

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
        DispatchFusedLoRA(input, baseOutput, loraA, loraB, output,
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
        // Match the alpha-schedule validation in CUDA / HIP / Vulkan dispatch.
        // Without this guard, the kernel computes 1/sqrt(alphaBarT) on the
        // raw input, producing NaN / Inf output for ᾱ <= 0.
        if (!(alphaBarT > 0f && alphaBarT <= 1f))
            throw new ArgumentOutOfRangeException(nameof(alphaBarT),
                $"alphaBarT must be in (0, 1]; got {alphaBarT}.");
        if (!(alphaBarTMinus1 >= 0f && alphaBarTMinus1 <= 1f))
            throw new ArgumentOutOfRangeException(nameof(alphaBarTMinus1),
                $"alphaBarTMinus1 must be in [0, 1]; got {alphaBarTMinus1}.");
        EnsureFusedAdvancedLibraryLoaded(nameof(FusedDDIMStep));
        if (xT is not MetalGpuBuffer xBuffer ||
            epsilonTheta is not MetalGpuBuffer eBuffer ||
            output is not MetalGpuBuffer oBuffer)
        {
            throw new ArgumentException("Buffers must be MetalGpuBuffer");
        }

        var pipeline = GetPipeline("FusedAdvancedKernels", _fusedAdvancedLibrary, "fused_ddim_step");
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
        // batchSize * outputFeatures wraps silently to a corrupt grid count
        // when the int product exceeds int.MaxValue (~2.1B). At 50000 × 50000
        // = 2.5B the previous code computed `total = -1.79e9`, the `<= 0`
        // guard returned early, and the dispatch was silently dropped.
        // Promote to long, validate, then cast — matches CUDA / HIP.
        if (batchSize <= 0 || outputFeatures <= 0) return;
        long totalLong = (long)batchSize * outputFeatures;
        if (totalLong > int.MaxValue)
            throw new ArgumentOutOfRangeException(nameof(batchSize),
                $"batchSize ({batchSize}) * outputFeatures ({outputFeatures}) = {totalLong} " +
                "exceeds int.MaxValue; split the dispatch into chunks.");
        int total = (int)totalLong;
        EnsureFusedAdvancedLibraryLoaded(nameof(FusedSparseLinear));
        if (input is not MetalGpuBuffer iBuffer ||
            packedCsr is not MetalGpuBuffer csrBuffer ||
            sparseValues is not MetalGpuBuffer vBuffer ||
            bias is not MetalGpuBuffer bBuffer ||
            output is not MetalGpuBuffer oBuffer)
        {
            throw new ArgumentException("Buffers must be MetalGpuBuffer");
        }

        var pipeline = GetPipeline("FusedAdvancedKernels", _fusedAdvancedLibrary, "fused_sparse_linear");
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

    private void DispatchFusedLoRA(
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
        if (batchSize <= 0 || rank <= 0 || outputFeatures <= 0) return;
        EnsureFusedAdvancedLibraryLoaded(nameof(FusedLoRAForward));
        if (input is not MetalGpuBuffer iBuffer ||
            baseOutput is not MetalGpuBuffer baseBuffer ||
            loraA is not MetalGpuBuffer aBuffer ||
            loraB is not MetalGpuBuffer bBuffer ||
            output is not MetalGpuBuffer oBuffer)
        {
            throw new ArgumentException("Buffers must be MetalGpuBuffer");
        }

        var pipeline = GetPipeline("FusedAdvancedKernels", _fusedAdvancedLibrary, "fused_lora_forward");

        // Two-stage layout: one threadgroup per batch row. Each group runs
        // up to 256 threads, capped by outputFeatures, and uses
        // rank * sizeof(float) bytes of dynamically-sized threadgroup memory
        // for the proj[] cache.
        const uint TargetThreadsPerGroup = 256u;
        uint threadsPerGroupX = (uint)Math.Min(outputFeatures, (int)TargetThreadsPerGroup);
        if (threadsPerGroupX < 1u) threadsPerGroupX = 1u;
        uint sharedBytes = checked((uint)rank * sizeof(float));

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
        encoder.SetThreadgroupMemoryLength(sharedBytes, 0);
        encoder.DispatchThreadgroups(
            new MTLSize((ulong)batchSize, 1, 1),
            new MTLSize(threadsPerGroupX, 1, 1));
    }
}
