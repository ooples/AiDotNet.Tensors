// Copyright (c) AiDotNet. All rights reserved.

using System;
using System.Runtime.InteropServices;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA;

/// <summary>
/// High-level cuSPARSE adapter that ships CSR data to the GPU, runs
/// <c>cusparseSpMM</c>, and copies the result back. Used by
/// <c>SparseOps.SparseMatMul</c> when both
/// <see cref="CuSparseNative.IsAvailable"/> and
/// <see cref="CudaBackend.IsCudaAvailable"/> report ready.
///
/// <para>Each call:
/// <list type="number">
///   <item>creates short-lived device buffers for <c>rowPtr</c>,
///         <c>colIdx</c>, <c>values</c>, <c>B</c>, <c>output</c>;</item>
///   <item>creates <c>cusparseSpMatDescr</c> + two <c>cusparseDnMatDescr</c>;</item>
///   <item>queries the workspace size, allocates it, runs SpMM;</item>
///   <item>downloads the result and frees every device buffer.</item>
/// </list>
/// The driver-API context is pushed/popped through the existing
/// <see cref="CudaBackend"/> so callers don't need a CUDA-aware
/// surrounding lifecycle.</para>
///
/// <para><b>Hardware caveat:</b> not testable on hosts without
/// <c>libcusparse</c>. The probe in <see cref="CuSparseNative.IsAvailable"/>
/// gates the dispatch; on machines where it returns false the SIMD CPU
/// path remains canonical and this class is never reached. Will be
/// validated end-to-end on a CUDA runner once one is available — the
/// glue is unit-tested by mocking the entry points.</para>
/// </summary>
internal static class CuSparseBackend
{
    /// <summary>Whether the cuSPARSE GPU dispatch path can run on this
    /// host. Both libraries (cuSPARSE and CUDA itself) must be loadable
    /// AND the calling assembly must be able to construct a
    /// <see cref="CudaBackend"/>.</summary>
    public static bool IsAvailable => CuSparseNative.IsAvailable && CudaBackend.IsCudaAvailable;

    /// <summary>
    /// CSR · dense → dense, where the inputs are host-side <c>float[]</c>
    /// arrays in CSR layout. Returns the dense output flat array of
    /// length <c>rows · n</c>.
    /// </summary>
    public static float[] SpMM(
        int[] rowPtr, int[] colIdx, float[] values,
        float[] b, int rows, int cols, int n)
    {
        if (!IsAvailable)
            throw new InvalidOperationException("cuSPARSE backend is not available.");

        var backend = CudaBackend.CreateOrThrow();
        var output = new float[rows * n];

        // All allocations sit inside the try/finally so an exception
        // anywhere along the upload/descriptor-create chain still hits
        // the cleanup branch — without this, a throw between the first
        // AllocateBuffer and the try would leak every device buffer
        // already allocated above (including any HGlobal pins). The
        // null-checks in the finally make every cleanup branch
        // independent so we don't skip later cleanups on partial init.
        IGpuBuffer? aValuesBuf = null, bBuf = null, outBuf = null;
        IGpuBuffer? rowPtrBuf = null, colIdxBuf = null, workspaceBuf = null;
        IntPtr handle = IntPtr.Zero;
        IntPtr matA = IntPtr.Zero, matB = IntPtr.Zero, matC = IntPtr.Zero;
        IntPtr alphaPin = IntPtr.Zero, betaPin = IntPtr.Zero, workspace = IntPtr.Zero;
        try
        {
            aValuesBuf = backend.AllocateBuffer(values);
            bBuf = backend.AllocateBuffer(b);
            outBuf = backend.AllocateBuffer(output);

            // int[] payloads need byte-buffer transport.
            rowPtrBuf = backend.AllocateByteBuffer(rowPtr.Length * sizeof(int));
            colIdxBuf = backend.AllocateByteBuffer(colIdx.Length * sizeof(int));
            UploadInts(rowPtrBuf.Handle, rowPtr);
            UploadInts(colIdxBuf.Handle, colIdx);

            alphaPin = Marshal.AllocHGlobal(sizeof(float));
            betaPin = Marshal.AllocHGlobal(sizeof(float));
            Marshal.StructureToPtr(1.0f, alphaPin, false);
            Marshal.StructureToPtr(0.0f, betaPin, false);

            CheckSp(CuSparseNative.cusparseCreate(out handle), "cusparseCreate");
            CheckSp(CuSparseNative.cusparseCreateCsr(out matA,
                rows, cols, values.Length,
                rowPtrBuf.Handle, colIdxBuf.Handle, aValuesBuf.Handle,
                CuSparseNative.IndexType.I32, CuSparseNative.IndexType.I32,
                idxBase: 0, CuSparseNative.DataType.R32F),
                "cusparseCreateCsr");
            CheckSp(CuSparseNative.cusparseCreateDnMat(out matB,
                cols, n, n, bBuf.Handle, CuSparseNative.DataType.R32F, order: 0 /* row-major */),
                "cusparseCreateDnMat(B)");
            CheckSp(CuSparseNative.cusparseCreateDnMat(out matC,
                rows, n, n, outBuf.Handle, CuSparseNative.DataType.R32F, order: 0),
                "cusparseCreateDnMat(C)");

            CheckSp(CuSparseNative.cusparseSpMM_bufferSize(handle,
                CuSparseNative.Operation.NonTranspose, CuSparseNative.Operation.NonTranspose,
                alphaPin, matA, matB, betaPin, matC,
                CuSparseNative.DataType.R32F, CuSparseNative.SpMMAlg.Default,
                out ulong bufferSize),
                "cusparseSpMM_bufferSize");
            if (bufferSize > 0)
            {
                workspaceBuf = backend.AllocateByteBuffer((int)bufferSize);
                workspace = workspaceBuf.Handle;
            }

            CheckSp(CuSparseNative.cusparseSpMM(handle,
                CuSparseNative.Operation.NonTranspose, CuSparseNative.Operation.NonTranspose,
                alphaPin, matA, matB, betaPin, matC,
                CuSparseNative.DataType.R32F, CuSparseNative.SpMMAlg.Default,
                workspace),
                "cusparseSpMM");

            backend.DownloadBuffer(outBuf, output);
            return output;
        }
        finally
        {
            if (matC != IntPtr.Zero) CuSparseNative.cusparseDestroyDnMat(matC);
            if (matB != IntPtr.Zero) CuSparseNative.cusparseDestroyDnMat(matB);
            if (matA != IntPtr.Zero) CuSparseNative.cusparseDestroySpMat(matA);
            if (handle != IntPtr.Zero) CuSparseNative.cusparseDestroy(handle);
            workspaceBuf?.Dispose();
            outBuf?.Dispose();
            bBuf?.Dispose();
            aValuesBuf?.Dispose();
            rowPtrBuf?.Dispose();
            colIdxBuf?.Dispose();
            if (alphaPin != IntPtr.Zero) Marshal.FreeHGlobal(alphaPin);
            if (betaPin != IntPtr.Zero) Marshal.FreeHGlobal(betaPin);
        }
    }

    private static unsafe void UploadInts(IntPtr device, int[] host)
    {
        // Reuse the driver's HtoD copy. CuBlasNative.cuMemcpyHtoD is
        // synchronous and lives on the same library handle the
        // CudaBackend already loaded.
        ulong byteCount = (ulong)host.Length * sizeof(int);
        fixed (int* src = host)
        {
            var status = CuBlasNative.cuMemcpyHtoD(device, (IntPtr)src, byteCount);
            CuBlasNative.CheckCudaResult(status, "cuMemcpyHtoD(int[])");
        }
    }

    private static void CheckSp(CuSparseNative.Status s, string what)
    {
        if (s != CuSparseNative.Status.Success)
            throw new InvalidOperationException($"{what} failed with cuSPARSE status {s}.");
    }
}
