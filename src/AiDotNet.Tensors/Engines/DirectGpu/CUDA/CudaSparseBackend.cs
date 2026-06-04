// Copyright (c) AiDotNet. All rights reserved.

using System;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA;

/// <summary>
/// CSR · dense → dense SpMM using AiDotNet's own managed CUDA kernel
/// (<c>csr_spmm</c> in <see cref="Kernels.CudaSparseKernels"/>), exposed with the
/// same host-array entry shape as <see cref="CuSparseBackend"/>: allocate device
/// buffers, upload the CSR triple + dense B, launch, download the dense result.
///
/// <para>This is the supply-chain-removal replacement for native cuSPARSE
/// (issue #515, P6): <see cref="LinearAlgebra.Sparse.SparseOps"/> makes this the
/// default GPU SpMM path, keeping <see cref="CuSparseBackend"/> only as an
/// availability fallback. Unlike cuSPARSE it needs no external library — only the
/// CUDA driver + NVRTC the rest of <see cref="CudaBackend"/> already requires.</para>
///
/// <para>The launched <c>csr_spmm</c> kernel accumulates each output element in a
/// single thread over the row's stored order, so the result is deterministic and
/// matches the managed CPU CSR reference to FMA tolerance. The warp-per-row kernel
/// variant, FP64 coverage, and the HIP/ROCm mirror are tracked as #515 follow-ups.</para>
/// </summary>
internal static class CudaSparseBackend
{
    /// <summary>Whether the managed custom-kernel SpMM path can run here. It needs
    /// only a constructible <see cref="CudaBackend"/> (CUDA driver + NVRTC); no
    /// cuSPARSE library, which is the whole point of the supply-chain removal.</summary>
    public static bool IsAvailable => CudaBackend.IsCudaAvailable;

    /// <summary>
    /// CSR · dense → dense, where the inputs are host-side <c>float[]</c> arrays in
    /// CSR layout. Returns the dense output flat array of length <c>rows · n</c>.
    /// Mirrors <see cref="CuSparseBackend.SpMM"/> but dispatches the managed
    /// <c>csr_spmm</c> kernel instead of <c>cusparseSpMM</c>.
    /// </summary>
    public static float[] SpMM(
        int[] rowPtr, int[] colIdx, float[] values,
        float[] b, int rows, int cols, int n)
    {
        if (!IsAvailable)
            throw new InvalidOperationException("CUDA custom-kernel SpMM backend is not available.");

        var backend = CudaBackend.CreateOrThrow();
        var output = new float[rows * n];

        // Every allocation sits inside the try so a throw anywhere along the
        // upload chain still hits the finally and frees the device buffers
        // already allocated (mirrors CuSparseBackend's leak-safe cleanup). The
        // null checks keep each cleanup branch independent on partial init.
        IGpuBuffer? valuesBuf = null, bBuf = null, outBuf = null;
        IGpuBuffer? rowPtrBuf = null, colIdxBuf = null;
        try
        {
            valuesBuf = backend.AllocateBuffer(values);
            bBuf = backend.AllocateBuffer(b);
            outBuf = backend.AllocateBuffer(output);

            // int[] payloads ride a byte buffer (matches CuSparseBackend).
            rowPtrBuf = backend.AllocateByteBuffer(rowPtr.Length * sizeof(int));
            colIdxBuf = backend.AllocateByteBuffer(colIdx.Length * sizeof(int));
            UploadInts(rowPtrBuf.Handle, rowPtr);
            UploadInts(colIdxBuf.Handle, colIdx);

            // Argument order matches IDirectGpuBackend.CsrSpMM:
            // (values, colIndices, rowPointers, denseB, output, M, K, N, nnz).
            backend.CsrSpMM(valuesBuf, colIdxBuf, rowPtrBuf, bBuf, outBuf,
                rows, cols, n, values.Length);

            backend.DownloadBuffer(outBuf, output);
            return output;
        }
        finally
        {
            outBuf?.Dispose();
            bBuf?.Dispose();
            valuesBuf?.Dispose();
            rowPtrBuf?.Dispose();
            colIdxBuf?.Dispose();
        }
    }

    /// <summary>Synchronous host→device copy for an <c>int[]</c> payload via the
    /// driver's <c>cuMemcpyHtoD</c> (same mechanism as <see cref="CuSparseBackend"/>).</summary>
    private static unsafe void UploadInts(IntPtr device, int[] host)
    {
        ulong byteCount = (ulong)host.Length * sizeof(int);
        fixed (int* src = host)
        {
            var status = CuBlasNative.cuMemcpyHtoD(device, (IntPtr)src, byteCount);
            CuBlasNative.CheckCudaResult(status, "cuMemcpyHtoD(int[])");
        }
    }
}
