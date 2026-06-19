// Copyright (c) AiDotNet. All rights reserved.
// IGpuHalfPrecisionBackend implementation for the Metal backend (issue #560):
// FP16 GEMM on the GPU via an MSL compute kernel, so the backend-agnostic
// DirectGpuTensorEngine FP16 matmul path runs on Apple-silicon / Metal devices.

using System;
using AiDotNet.Tensors.Engines.Gpu;
using static AiDotNet.Tensors.Engines.DirectGpu.Metal.MetalNativeBindings;

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
    /// <remarks>The fused backward is two dispatches of the <c>matmul_fp16_backward</c> MSL kernel in the same
    /// matrix library as the forward GEMM, so it tracks device availability like <see cref="SupportsHgemm"/>.</remarks>
    public bool SupportsFp16FusedBackward => IsAvailable;

    /// <inheritdoc/>
    /// <remarks>
    /// Two dispatches of the transposed <c>matmul_fp16_backward</c> MSL kernel (half in, FP32 accumulate, no
    /// materialized transpose): gradA[M,K] = gradC·Bᵀ (transA=0, transB=1) and gradB[K,N] = Aᵀ·gradC
    /// (transA=1, transB=0). The kernel writes FP32; FP16-output grads run into an FP32 scratch then
    /// <c>ConvertToFp16</c> (mirrors <see cref="Hgemm"/>).
    /// </remarks>
    public void MatMulBackwardFp16Fused(
        IGpuBuffer gradCFp16, IGpuBuffer aFp16, IGpuBuffer bFp16,
        IGpuBuffer gradAOut, IGpuBuffer gradBOut,
        int m, int n, int k, bool gradOutHalf)
    {
        ThrowIfDisposed();
        if (gradCFp16 is null) throw new ArgumentNullException(nameof(gradCFp16));
        if (aFp16 is null) throw new ArgumentNullException(nameof(aFp16));
        if (bFp16 is null) throw new ArgumentNullException(nameof(bFp16));
        if (gradAOut is null) throw new ArgumentNullException(nameof(gradAOut));
        if (gradBOut is null) throw new ArgumentNullException(nameof(gradBOut));
        if (m <= 0) throw new ArgumentOutOfRangeException(nameof(m), "Dimensions must be positive.");
        if (n <= 0) throw new ArgumentOutOfRangeException(nameof(n), "Dimensions must be positive.");
        if (k <= 0) throw new ArgumentOutOfRangeException(nameof(k), "Dimensions must be positive.");

        // gradA[M,K] = gradC[M,N] · Bᵀ : Mo=M, No=K, Kc=N; A=gradC (transA=0), B=Bfwd (transB=1).
        DispatchBackwardGemm(gradCFp16, bFp16, gradAOut, m, k, n, transA: 0, transB: 1, gradOutHalf);
        // gradB[K,N] = Aᵀ · gradC[M,N] : Mo=K, No=N, Kc=M; A=Afwd (transA=1), B=gradC (transB=0).
        DispatchBackwardGemm(aFp16, gradCFp16, gradBOut, k, n, m, transA: 1, transB: 0, gradOutHalf);
    }

    /// <summary>
    /// One dispatch of the transposed <c>matmul_fp16_backward</c> kernel into <paramref name="gradOut"/>. The
    /// kernel writes FP32; when <paramref name="gradOutHalf"/> it writes an FP32 scratch then packs to FP16.
    /// </summary>
    private void DispatchBackwardGemm(IGpuBuffer a, IGpuBuffer b, IGpuBuffer gradOut,
        int mo, int no, int kc, int transA, int transB, bool gradOutHalf)
    {
        if (a is not MetalGpuBuffer aBuffer || b is not MetalGpuBuffer bBuffer)
            throw new ArgumentException("Buffers must be MetalGpuBuffer");

        IGpuBuffer? scratch = gradOutHalf ? AllocateBuffer(mo * no) : null;
        try
        {
            var dest = scratch ?? gradOut;
            if (dest is not MetalGpuBuffer cBuffer)
                throw new ArgumentException("Buffers must be MetalGpuBuffer");

            var pipeline = GetPipeline("Matrix", _matrixLibrary, "matmul_fp16_backward");
            var (threadgroups, threadsPerGroup) = pipeline.Calculate2DDispatch(no, mo);
            using (var encoder = _commandQueue.CreateScopedComputeEncoder())
            {
                encoder.SetPipelineState(pipeline.Handle);
                encoder.SetBuffer(aBuffer, 0);
                encoder.SetBuffer(bBuffer, 1);
                encoder.SetBuffer(cBuffer, 2);
                encoder.SetBytes((uint)mo, 3);
                encoder.SetBytes((uint)no, 4);
                encoder.SetBytes((uint)kc, 5);
                encoder.SetBytes((uint)transA, 6);
                encoder.SetBytes((uint)transB, 7);
                encoder.DispatchThreadgroups(threadgroups, threadsPerGroup);
            }
            if (scratch is not null)
                ConvertToFp16(scratch, gradOut, mo * no);
        }
        finally
        {
            scratch?.Dispose();
        }
    }

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

    /// <summary>
    /// True when the FP16-native op kernels are available — they live in the same matrix MSL library as
    /// the FP16 GEMM, so they compile whenever the backend does.
    /// </summary>
    public bool SupportsFp16NativeOps => IsAvailable;

    /// <summary>GELU over a half buffer: out[i] = gelu(in[i]); half in/out, FP32 math.</summary>
    public void Fp16Gelu(IGpuBuffer input, IGpuBuffer output, int n)
        => DispatchFp16Unary("fp16_gelu_native", input, output, n);

    /// <summary>ReLU over a half buffer: out[i] = max(in[i], 0); half in/out, FP32 math.</summary>
    public void Fp16Relu(IGpuBuffer input, IGpuBuffer output, int n)
        => DispatchFp16Unary("fp16_relu_native", input, output, n);

    /// <summary>Residual add over half buffers: out[i] = a[i] + b[i]; half in/out, FP32 accumulate.</summary>
    public void Fp16Add(IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, int n)
    {
        ThrowIfDisposed();
        if (a is null) throw new ArgumentNullException(nameof(a));
        if (b is null) throw new ArgumentNullException(nameof(b));
        if (output is null) throw new ArgumentNullException(nameof(output));
        if (n <= 0) throw new ArgumentOutOfRangeException(nameof(n), "Element count must be positive.");
        if (a is not MetalGpuBuffer aBuf || b is not MetalGpuBuffer bBuf || output is not MetalGpuBuffer oBuf)
            throw new ArgumentException("Buffers must be MetalGpuBuffer");

        var pipeline = GetPipeline("Matrix", _matrixLibrary, "fp16_add_native");
        var (threadgroups, threadsPerGroup) = pipeline.Calculate1DDispatch(n);
        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        encoder.SetBuffer(aBuf, 0);
        encoder.SetBuffer(bBuf, 1);
        encoder.SetBuffer(oBuf, 2);
        encoder.SetBytes((uint)n, 3);
        encoder.DispatchThreadgroups(threadgroups, threadsPerGroup);
    }

    /// <summary>Row softmax over the last axis of a half buffer: one threadgroup per row, FP32 max/sum,
    /// half in/out. Metal counterpart of the CUDA fp16_softmax_native.</summary>
    public void Fp16Softmax(IGpuBuffer input, IGpuBuffer output, int rows, int cols)
    {
        ThrowIfDisposed();
        if (input is null) throw new ArgumentNullException(nameof(input));
        if (output is null) throw new ArgumentNullException(nameof(output));
        if (rows <= 0 || cols <= 0) throw new ArgumentException($"rows/cols must be positive (rows={rows}, cols={cols}).");
        if (input is not MetalGpuBuffer inBuf || output is not MetalGpuBuffer outBuf)
            throw new ArgumentException("Buffers must be MetalGpuBuffer");

        var pipeline = GetPipeline("Matrix", _matrixLibrary, "fp16_softmax_native");
        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        encoder.SetBuffer(inBuf, 0);
        encoder.SetBuffer(outBuf, 1);
        encoder.SetBytes((uint)rows, 2);
        encoder.SetBytes((uint)cols, 3);
        encoder.DispatchThreadgroups(new MTLSize((ulong)rows, 1, 1), new MTLSize(256, 1, 1)); // one threadgroup per row
    }

    /// <summary>Row layernorm over the last axis of a half buffer with half gamma/beta: one threadgroup per
    /// row, FP32 mean/var, half in/out; optionally writes per-row FP32 mean/variance. Metal counterpart of
    /// the CUDA fp16_layernorm_native.</summary>
    public void Fp16LayerNorm(IGpuBuffer input, IGpuBuffer gamma, IGpuBuffer beta, IGpuBuffer output,
        IGpuBuffer meanFp32, IGpuBuffer varFp32, int rows, int cols, float eps)
    {
        ThrowIfDisposed();
        if (input is null) throw new ArgumentNullException(nameof(input));
        if (gamma is null) throw new ArgumentNullException(nameof(gamma));
        if (beta is null) throw new ArgumentNullException(nameof(beta));
        if (output is null) throw new ArgumentNullException(nameof(output));
        if (rows <= 0 || cols <= 0) throw new ArgumentException($"rows/cols must be positive (rows={rows}, cols={cols}).");
        if (eps <= 0f || float.IsNaN(eps) || float.IsInfinity(eps))
            throw new ArgumentOutOfRangeException(nameof(eps), eps, "eps must be finite and positive.");
        if (input is not MetalGpuBuffer inBuf || gamma is not MetalGpuBuffer gBuf || beta is not MetalGpuBuffer bBuf || output is not MetalGpuBuffer oBuf)
            throw new ArgumentException("Buffers must be MetalGpuBuffer");

        // The kernel always writes mean/var; supply temporaries when the caller passes none.
        IGpuBuffer? tmpMean = null, tmpVar = null;
        var meanBuf = meanFp32 ?? (tmpMean = AllocateBuffer(rows));
        var varBuf = varFp32 ?? (tmpVar = AllocateBuffer(rows));
        try
        {
            if (meanBuf is not MetalGpuBuffer mBuf || varBuf is not MetalGpuBuffer vBuf)
                throw new ArgumentException("Buffers must be MetalGpuBuffer");
            var pipeline = GetPipeline("Matrix", _matrixLibrary, "fp16_layernorm_native");
            using var encoder = _commandQueue.CreateScopedComputeEncoder();
            encoder.SetPipelineState(pipeline.Handle);
            encoder.SetBuffer(inBuf, 0);
            encoder.SetBuffer(gBuf, 1);
            encoder.SetBuffer(bBuf, 2);
            encoder.SetBuffer(oBuf, 3);
            encoder.SetBuffer(mBuf, 4);
            encoder.SetBuffer(vBuf, 5);
            encoder.SetBytes((uint)rows, 6);
            encoder.SetBytes((uint)cols, 7);
            encoder.SetBytes(eps, 8);
            encoder.DispatchThreadgroups(new MTLSize((ulong)rows, 1, 1), new MTLSize(256, 1, 1)); // one threadgroup per row
        }
        finally
        {
            tmpMean?.Dispose();
            tmpVar?.Dispose();
        }
    }

    private void DispatchFp16Unary(string kernelName, IGpuBuffer input, IGpuBuffer output, int n)
    {
        ThrowIfDisposed();
        if (input is null) throw new ArgumentNullException(nameof(input));
        if (output is null) throw new ArgumentNullException(nameof(output));
        if (n <= 0) throw new ArgumentOutOfRangeException(nameof(n), "Element count must be positive.");
        if (input is not MetalGpuBuffer inBuf || output is not MetalGpuBuffer outBuf)
            throw new ArgumentException("Buffers must be MetalGpuBuffer");

        var pipeline = GetPipeline("Matrix", _matrixLibrary, kernelName);
        var (threadgroups, threadsPerGroup) = pipeline.Calculate1DDispatch(n);
        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        encoder.SetBuffer(inBuf, 0);
        encoder.SetBuffer(outBuf, 1);
        encoder.SetBytes((uint)n, 2);
        encoder.DispatchThreadgroups(threadgroups, threadsPerGroup);
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
