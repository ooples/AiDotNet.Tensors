// Copyright (c) CrowdTrainer LLC. All rights reserved.

using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using AiDotNet.Tensors.Engines.Compilation.Codegen;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ir;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines;

public partial class DirectGpuTensorEngine
{
    // Keyed weakly by the graph: each fused node owns its graph, so an entry (and its compiled kernel) lives exactly as
    // long as the node that replays it, and a retired plan's graphs no longer pin kernels for the engine's lifetime.
    // The inner map is per backend executor, since the same graph can be compiled for more than one.
    private readonly ConditionalWeakTable<CodegenGraph, ConcurrentDictionary<INativeGpuCodegenExecutor, CodegenKernel?>> _compiledCodegenCache = new();

    /// <summary>
    /// Executes a fused <see cref="CodegenGraph"/> on the active GPU backend in a single native kernel launch
    /// without materializing intermediate tensors in global device memory.
    /// </summary>
    internal bool TryExecuteCodegenGraphInto(
        CodegenGraph graph,
        IReadOnlyList<Tensor<float>> inputs,
        Tensor<float> destination)
    {
        using var __gpuOp = EnterGpuOp();
        if (!TryGetBackend(out var backend) || backend is not INativeGpuCodegenExecutor executor)
            return false;

        var perExecutor = _compiledCodegenCache.GetValue(graph,
            static _ => new ConcurrentDictionary<INativeGpuCodegenExecutor, CodegenKernel?>());
        var kernel = perExecutor.GetOrAdd(executor, e =>
        {
            var emitted = e.EmitCodegenKernel(graph, CodegenElementType.Float32);
            return !emitted.Declined && emitted.Kernel is not null && e.CanExecuteCodegenKernel(emitted.Kernel)
                ? emitted.Kernel
                : null;
        });

        if (kernel is null)
            return false;

        var ownedInputs = new OwnedBuffer[inputs.Count];
        var rawInputs = new IGpuBuffer[inputs.Count];
        var outBuf = default(OwnedBuffer);
        bool ownershipTransferred = false;
        try
        {
            for (int i = 0; i < inputs.Count; i++)
            {
                ownedInputs[i] = GetOrAllocateBuffer(backend, inputs[i]);
                rawInputs[i] = ownedInputs[i].Buffer;
            }

            if (TryGetNonAliasingResidentDestBuffer(backend, destination, rawInputs, out var directDest) && directDest is not null)
            {
                executor.ExecuteCodegenKernelAsync(kernel, rawInputs, new[] { directDest }, destination.Length)
                    .GetAwaiter()
                    .GetResult();
                RearmStateAfterReplay(destination, directDest);
                return true;
            }

            outBuf = AllocateFullyWrittenOutputBuffer(backend, destination.Length);
            executor.ExecuteCodegenKernelAsync(kernel, rawInputs, new[] { outBuf.Buffer }, destination.Length)
                .GetAwaiter()
                .GetResult();

            var wrapped = DeferTensorResult<float>(backend, outBuf.Buffer, destination.Length, (int[])destination._shape.Clone());
            ownershipTransferred = true;
            CopyResultInto(this, wrapped, destination);
            return true;
        }
        catch
        {
            if (!ownershipTransferred)
            {
                outBuf.Dispose();
            }
            if (ThrowOnGpuKernelFallback) throw;
            return false;
        }
        finally
        {
            for (int i = 0; i < ownedInputs.Length; i++)
            {
                ownedInputs[i].Dispose();
            }
        }
    }

    internal bool TryExecuteFusedRmsNormResidualInto(
        Tensor<float> input,
        Tensor<float> residual,
        Tensor<float> gamma,
        Tensor<float> output,
        Tensor<float> rmsOut,
        int rows,
        int cols,
        float epsilon)
    {
        using var __gpuOp = EnterGpuOp();
        if (!TryGetBackend(out var backend))
            return false;

        var bufOut = default(OwnedBuffer);
        var bufRms = default(OwnedBuffer);
        bool transferredOut = false;
        bool transferredRms = false;
        try
        {
            using var bufX = GetOrAllocateBuffer(backend, input);
            using var bufRes = GetOrAllocateBuffer(backend, residual);
            using var bufGamma = GetOrAllocateBuffer(backend, gamma);

            if (TryGetNonAliasingResidentDestBuffer(
                    backend, output, new[] { bufX.Buffer, bufRes.Buffer, bufGamma.Buffer }, out var directOut)
                && directOut is not null)
            {
                // The saved-RMS target must not alias any input or the primary output. Only a resident rmsOut buffer
                // that is distinct from every binding is written in place; otherwise a fresh device buffer receives the
                // RMS values and is copied into rmsOut afterwards.
                bool rmsDirect = TryGetNonAliasingResidentDestBuffer(
                        backend, rmsOut, new[] { bufX.Buffer, bufRes.Buffer, bufGamma.Buffer, directOut }, out var directRms)
                    && directRms is not null;
                IGpuBuffer rmsTarget;
                if (rmsDirect)
                {
                    rmsTarget = directRms!;
                }
                else
                {
                    bufRms = AllocateFullyWrittenOutputBuffer(backend, rows);
                    rmsTarget = bufRms.Buffer;
                }

                if (!CudaBackend.TryDispatchRmsNormResidual(
                        backend, bufX.Buffer, bufRes.Buffer, bufGamma.Buffer, directOut, rmsTarget, rows, cols, epsilon))
                {
                    bufRms.Dispose();
                    return false;
                }

                RearmStateAfterReplay(output, directOut);
                if (rmsDirect)
                {
                    RearmStateAfterReplay(rmsOut, rmsTarget);
                }
                else
                {
                    var directRmsTensor = DeferTensorResult<float>(backend, bufRms.Buffer, rows, (int[])rmsOut._shape.Clone());
                    transferredRms = true;
                    CopyResultInto(this, directRmsTensor, rmsOut);
                }
                return true;
            }

            bufOut = AllocateFullyWrittenOutputBuffer(backend, output.Length);
            bufRms = AllocateFullyWrittenOutputBuffer(backend, rows);

            if (!CudaBackend.TryDispatchRmsNormResidual(
                    backend, bufX.Buffer, bufRes.Buffer, bufGamma.Buffer, bufOut.Buffer, bufRms.Buffer, rows, cols, epsilon))
            {
                bufOut.Dispose();
                bufRms.Dispose();
                return false;
            }

            var outTensor = DeferTensorResult<float>(backend, bufOut.Buffer, output.Length, (int[])output._shape.Clone());
            transferredOut = true;
            var rmsTensor = DeferTensorResult<float>(backend, bufRms.Buffer, rows, (int[])rmsOut._shape.Clone());
            transferredRms = true;
            CopyResultInto(this, outTensor, output);
            CopyResultInto(this, rmsTensor, rmsOut);
            return true;
        }
        catch
        {
            if (!transferredOut) bufOut.Dispose();
            if (!transferredRms) bufRms.Dispose();
            if (ThrowOnGpuKernelFallback) throw;
            return false;
        }
    }

    internal bool TryExecuteFusedLayerNormResidualInto(
        Tensor<float> input,
        Tensor<float> residual,
        Tensor<float> gamma,
        Tensor<float> beta,
        Tensor<float> output,
        Tensor<float> meanOut,
        Tensor<float> invStdOut,
        int rows,
        int cols,
        float epsilon)
    {
        using var __gpuOp = EnterGpuOp();
        if (!TryGetBackend(out var backend))
            return false;

        var bufOut = default(OwnedBuffer);
        var bufMean = default(OwnedBuffer);
        var bufInvStd = default(OwnedBuffer);
        bool transferredOut = false;
        bool transferredMean = false;
        bool transferredInvStd = false;
        try
        {
            using var bufX = GetOrAllocateBuffer(backend, input);
            using var bufRes = GetOrAllocateBuffer(backend, residual);
            using var bufGamma = GetOrAllocateBuffer(backend, gamma);
            using var bufBeta = GetOrAllocateBuffer(backend, beta);
            bufOut = AllocateFullyWrittenOutputBuffer(backend, output.Length);
            bufMean = AllocateFullyWrittenOutputBuffer(backend, rows);
            bufInvStd = AllocateFullyWrittenOutputBuffer(backend, rows);

            if (!CudaBackend.TryDispatchLayerNormResidual(
                    backend, bufX.Buffer, bufRes.Buffer, bufGamma.Buffer, bufBeta.Buffer,
                    bufOut.Buffer, bufMean.Buffer, bufInvStd.Buffer, rows, cols, epsilon))
            {
                bufOut.Dispose();
                bufMean.Dispose();
                bufInvStd.Dispose();
                return false;
            }

            var outTensor = DeferTensorResult<float>(backend, bufOut.Buffer, output.Length, (int[])output._shape.Clone());
            transferredOut = true;
            var meanTensor = DeferTensorResult<float>(backend, bufMean.Buffer, rows, (int[])meanOut._shape.Clone());
            transferredMean = true;
            var invStdTensor = DeferTensorResult<float>(backend, bufInvStd.Buffer, rows, (int[])invStdOut._shape.Clone());
            transferredInvStd = true;
            CopyResultInto(this, outTensor, output);
            CopyResultInto(this, meanTensor, meanOut);
            CopyResultInto(this, invStdTensor, invStdOut);
            return true;
        }
        catch
        {
            if (!transferredOut) bufOut.Dispose();
            if (!transferredMean) bufMean.Dispose();
            if (!transferredInvStd) bufInvStd.Dispose();
            if (ThrowOnGpuKernelFallback) throw;
            return false;
        }
    }

    internal bool TryExecuteFusedRotarySplitHalfInto(
        Tensor<float> input,
        Tensor<float> cos,
        Tensor<float> sin,
        Tensor<float> output,
        int outerCount,
        int seqLen,
        int headDim,
        int trigStride,
        bool inverseAdjoint)
    {
        using var __gpuOp = EnterGpuOp();
        if (!TryGetBackend(out var backend))
            return false;

        var bufOut = default(OwnedBuffer);
        bool transferred = false;
        try
        {
            using var bufX = GetOrAllocateBuffer(backend, input);
            using var bufCos = GetOrAllocateBuffer(backend, cos);
            using var bufSin = GetOrAllocateBuffer(backend, sin);
            bufOut = AllocateFullyWrittenOutputBuffer(backend, output.Length);

            if (!CudaBackend.TryDispatchRotarySplitHalf(
                    backend, bufX.Buffer, bufCos.Buffer, bufSin.Buffer, bufOut.Buffer,
                    outerCount, seqLen, headDim, trigStride, inverseAdjoint))
            {
                bufOut.Dispose();
                return false;
            }

            var outTensor = DeferTensorResult<float>(backend, bufOut.Buffer, output.Length, (int[])output._shape.Clone());
            transferred = true;
            CopyResultInto(this, outTensor, output);
            return true;
        }
        catch
        {
            if (!transferred) bufOut.Dispose();
            if (ThrowOnGpuKernelFallback) throw;
            return false;
        }
    }

    internal bool TryExecuteFusedRmsNormResidualRopeInto(
        Tensor<float> input,
        Tensor<float> residual,
        Tensor<float> gamma,
        Tensor<float> cos,
        Tensor<float> sin,
        Tensor<float> output,
        int outerCount,
        int seqLen,
        int headDim,
        int trigStride,
        float epsilon)
    {
        using var __gpuOp = EnterGpuOp();
        if (!TryGetBackend(out var backend))
            return false;

        var bufOut = default(OwnedBuffer);
        bool transferred = false;
        try
        {
            using var bufX = GetOrAllocateBuffer(backend, input);
            using var bufRes = GetOrAllocateBuffer(backend, residual);
            using var bufGamma = GetOrAllocateBuffer(backend, gamma);
            using var bufCos = GetOrAllocateBuffer(backend, cos);
            using var bufSin = GetOrAllocateBuffer(backend, sin);

            if (TryGetNonAliasingResidentDestBuffer(
                    backend,
                    output,
                    new[] { bufX.Buffer, bufRes.Buffer, bufGamma.Buffer, bufCos.Buffer, bufSin.Buffer },
                    out var directOut)
                && directOut is not null)
            {
                if (!CudaBackend.TryDispatchRmsNormResidualRope(
                        backend, bufX.Buffer, bufRes.Buffer, bufGamma.Buffer, bufCos.Buffer, bufSin.Buffer,
                        directOut, outerCount, seqLen, headDim, trigStride, epsilon))
                {
                    return false;
                }

                RearmStateAfterReplay(output, directOut);
                return true;
            }

            bufOut = AllocateFullyWrittenOutputBuffer(backend, output.Length);
            if (!CudaBackend.TryDispatchRmsNormResidualRope(
                    backend, bufX.Buffer, bufRes.Buffer, bufGamma.Buffer, bufCos.Buffer, bufSin.Buffer,
                    bufOut.Buffer, outerCount, seqLen, headDim, trigStride, epsilon))
            {
                bufOut.Dispose();
                return false;
            }

            var outTensor = DeferTensorResult<float>(backend, bufOut.Buffer, output.Length, (int[])output._shape.Clone());
            transferred = true;
            CopyResultInto(this, outTensor, output);
            return true;
        }
        catch
        {
            if (!transferred) bufOut.Dispose();
            if (ThrowOnGpuKernelFallback) throw;
            return false;
        }
    }

    private static bool TryGetNonAliasingResidentDestBuffer(
        IDirectGpuBackend backend,
        Tensor<float> destination,
        IReadOnlyList<IGpuBuffer> inputs,
        out IGpuBuffer? destBuffer)
    {
        if (destination._gpuBuffer is { } candidate
            && candidate.Handle != IntPtr.Zero
            && candidate.Size >= destination.Length
            && ReferenceEquals(destination._gpuBackend, backend))
        {
            for (int i = 0; i < inputs.Count; i++)
            {
                if (inputs[i].Handle == candidate.Handle)
                {
                    destBuffer = null;
                    return false;
                }
            }

            destBuffer = candidate;
            return true;
        }

        destBuffer = null;
        return false;
    }
}
