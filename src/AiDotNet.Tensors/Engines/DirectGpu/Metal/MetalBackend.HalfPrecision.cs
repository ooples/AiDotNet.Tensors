// Copyright (c) AiDotNet. All rights reserved.
// IGpuHalfPrecisionBackend implementation for the Metal backend (issue #560):
// FP16 GEMM on the GPU via an MSL compute kernel, so the backend-agnostic
// DirectGpuTensorEngine FP16 matmul path runs on Apple-silicon / Metal devices.

using System;
using AiDotNet.Tensors.Engines.Gpu;

namespace AiDotNet.Tensors.Engines.DirectGpu.Metal;

/// <summary>
/// Half-precision GEMM dispatch for the <see cref="IGpuHalfPrecisionBackend"/>
/// capability interface. Mirrors the CUDA backend's row-major contract:
/// C = A·B with A = M×K, B = K×N, C = M×N.
/// <para>
/// FP16 operands are packed two halves per 32-bit word (the layout
/// <see cref="ConvertToFp16"/> produces); the MSL kernel reinterprets the buffer
/// as <c>device const half*</c> and accumulates in FP32 — the AMP-standard
/// mixed-precision matmul (matches <c>cublasGemmEx(CUDA_R_16F, COMPUTE_32F)</c>).
/// </para>
/// </summary>
public sealed partial class MetalBackend : IGpuHalfPrecisionBackend
{
    /// <inheritdoc/>
    public bool SupportsHgemm => IsAvailable;

    /// <inheritdoc/>
    public void GemmFp16In32fOut(IGpuBuffer aFp16, IGpuBuffer bFp16, IGpuBuffer cFp32,
        int m, int n, int k)
    {
        ThrowIfDisposed();
        ValidateHalfGemmArgs(aFp16, bFp16, cFp32, m, n, k);

        if (aFp16 is not MetalGpuBuffer aBuffer ||
            bFp16 is not MetalGpuBuffer bBuffer ||
            cFp32 is not MetalGpuBuffer cBuffer)
        {
            throw new ArgumentException("Buffers must be MetalGpuBuffer");
        }

        var pipeline = GetPipeline("Matrix", _matrixLibrary, "matmul_fp16_fp32out");
        var (threadgroups, threadsPerGroup) = pipeline.Calculate2DDispatch(n, m);

        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        encoder.SetBuffer(aBuffer, 0);
        encoder.SetBuffer(bBuffer, 1);
        encoder.SetBuffer(cBuffer, 2);
        encoder.SetBytes((uint)m, 3);
        encoder.SetBytes((uint)n, 4);
        encoder.SetBytes((uint)k, 5);
        encoder.DispatchThreadgroups(threadgroups, threadsPerGroup);
    }

    /// <inheritdoc/>
    public void Hgemm(IGpuBuffer aFp16, IGpuBuffer bFp16, IGpuBuffer cFp16,
        int m, int n, int k)
    {
        ThrowIfDisposed();
        ValidateHalfGemmArgs(aFp16, bFp16, cFp16, m, n, k);

        // Accumulate in FP32 into a scratch buffer, then round the result down to
        // FP16 (matches cublasHgemm's FP16 output). Keeping the accumulation in
        // FP32 avoids the worst half-precision drift; ConvertToFp16 then packs
        // the result to halves. A separate scratch avoids reading and writing the
        // same buffer in one pass.
        using var scratch = AllocateBuffer(m * n);
        GemmFp16In32fOut(aFp16, bFp16, scratch, m, n, k);
        ConvertToFp16(scratch, cFp16, m * n);
    }

    private static void ValidateHalfGemmArgs(IGpuBuffer a, IGpuBuffer b, IGpuBuffer c,
        int m, int n, int k)
    {
        if (a is null) throw new ArgumentNullException(nameof(a));
        if (b is null) throw new ArgumentNullException(nameof(b));
        if (c is null) throw new ArgumentNullException(nameof(c));
        // Report the offending dimension by name (mirrors the CUDA backend).
        if (m <= 0) throw new ArgumentOutOfRangeException(nameof(m), "Dimensions must be positive.");
        if (n <= 0) throw new ArgumentOutOfRangeException(nameof(n), "Dimensions must be positive.");
        if (k <= 0) throw new ArgumentOutOfRangeException(nameof(k), "Dimensions must be positive.");
    }
}
