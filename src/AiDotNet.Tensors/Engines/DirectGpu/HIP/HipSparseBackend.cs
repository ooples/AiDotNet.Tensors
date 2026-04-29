// Copyright (c) AiDotNet. All rights reserved.

using System;
using System.Runtime.InteropServices;

namespace AiDotNet.Tensors.Engines.DirectGpu.HIP;

/// <summary>
/// rocSPARSE adapter — HIP/ROCm equivalent of
/// <see cref="CUDA.CuSparseBackend"/>. Same shape (allocate buffers,
/// build descriptors, call SpMM, copy back), different runtime. Used
/// by <c>SparseOps.SparseMatMul</c> when running on AMD GPUs.
///
/// <para><b>Hardware caveat:</b> not testable on hosts without ROCm.
/// The dispatch is gated behind both <see cref="RocSparseNative.IsAvailable"/>
/// and the HIP runtime probe; on machines where either is missing the
/// SIMD CPU tier serves the call and this class is never reached.
/// End-to-end validation runs on a ROCm runner once one is in CI.</para>
/// </summary>
internal static class HipSparseBackend
{
    /// <summary>Whether the rocSPARSE GPU dispatch path can run on this
    /// host. Requires both <c>librocsparse</c> AND a HIP runtime
    /// constructible via <see cref="HipBackend.IsHipAvailable"/>.</summary>
    public static bool IsAvailable => RocSparseNative.IsAvailable && HipBackend.IsHipAvailable;

    /// <summary>CSR · dense → dense — float32 only at the moment, the
    /// type combination dominant in production sparse training.</summary>
    public static float[] SpMM(
        int[] rowPtr, int[] colIdx, float[] values,
        float[] b, int rows, int cols, int n)
    {
        if (!IsAvailable)
            throw new InvalidOperationException("HIP rocSPARSE backend is not available.");

        var backend = new HipBackend();
        if (!backend.IsAvailable)
            throw new InvalidOperationException("HIP backend failed to initialise.");

        var aValuesBuf = backend.AllocateBuffer(values);
        var bBuf = backend.AllocateBuffer(b);
        var output = new float[rows * n];
        var outBuf = backend.AllocateBuffer(output);
        var rowPtrBuf = backend.AllocateByteBuffer(rowPtr.Length * sizeof(int));
        var colIdxBuf = backend.AllocateByteBuffer(colIdx.Length * sizeof(int));
        UploadInts(rowPtrBuf.Handle, rowPtr);
        UploadInts(colIdxBuf.Handle, colIdx);

        IntPtr handle = IntPtr.Zero;
        IntPtr matA = IntPtr.Zero, matB = IntPtr.Zero, matC = IntPtr.Zero;
        IntPtr alphaPin = Marshal.AllocHGlobal(sizeof(float));
        IntPtr betaPin = Marshal.AllocHGlobal(sizeof(float));
        IGpuBuffer? workspaceBuf = null;
        try
        {
            Marshal.StructureToPtr(1.0f, alphaPin, false);
            Marshal.StructureToPtr(0.0f, betaPin, false);

            CheckSp(RocSparseNative.rocsparse_create_handle(out handle), "rocsparse_create_handle");
            CheckSp(RocSparseNative.rocsparse_create_csr_descr(out matA,
                rows, cols, values.Length,
                rowPtrBuf.Handle, colIdxBuf.Handle, aValuesBuf.Handle,
                RocSparseNative.IndexType.I32, RocSparseNative.IndexType.I32,
                RocSparseNative.IndexBase.Zero, RocSparseNative.DataType.R32F),
                "rocsparse_create_csr_descr");
            CheckSp(RocSparseNative.rocsparse_create_dnmat_descr(out matB,
                cols, n, n, bBuf.Handle, RocSparseNative.DataType.R32F, order: 0),
                "rocsparse_create_dnmat_descr(B)");
            CheckSp(RocSparseNative.rocsparse_create_dnmat_descr(out matC,
                rows, n, n, outBuf.Handle, RocSparseNative.DataType.R32F, order: 0),
                "rocsparse_create_dnmat_descr(C)");

            // rocSPARSE SpMM is two-stage: query-buffer-size, then compute.
            ulong bufferSize = 0;
            CheckSp(RocSparseNative.rocsparse_spmm(handle,
                RocSparseNative.Operation.NonTranspose, RocSparseNative.Operation.NonTranspose,
                alphaPin, matA, matB, betaPin, matC,
                RocSparseNative.DataType.R32F, RocSparseNative.SpMMAlg.Default,
                stage: 0, ref bufferSize, IntPtr.Zero),
                "rocsparse_spmm(buffer-size)");
            IntPtr workspace = IntPtr.Zero;
            if (bufferSize > 0)
            {
                workspaceBuf = backend.AllocateByteBuffer((int)bufferSize);
                workspace = workspaceBuf.Handle;
            }
            CheckSp(RocSparseNative.rocsparse_spmm(handle,
                RocSparseNative.Operation.NonTranspose, RocSparseNative.Operation.NonTranspose,
                alphaPin, matA, matB, betaPin, matC,
                RocSparseNative.DataType.R32F, RocSparseNative.SpMMAlg.Default,
                stage: 2, ref bufferSize, workspace),
                "rocsparse_spmm(compute)");

            backend.DownloadBuffer(outBuf, output);
            return output;
        }
        finally
        {
            if (matC != IntPtr.Zero) RocSparseNative.rocsparse_destroy_dnmat_descr(matC);
            if (matB != IntPtr.Zero) RocSparseNative.rocsparse_destroy_dnmat_descr(matB);
            if (matA != IntPtr.Zero) RocSparseNative.rocsparse_destroy_spmat_descr(matA);
            if (handle != IntPtr.Zero) RocSparseNative.rocsparse_destroy_handle(handle);
            workspaceBuf?.Dispose();
            outBuf.Dispose();
            bBuf.Dispose();
            aValuesBuf.Dispose();
            rowPtrBuf.Dispose();
            colIdxBuf.Dispose();
            Marshal.FreeHGlobal(alphaPin);
            Marshal.FreeHGlobal(betaPin);
        }
    }

    private static unsafe void UploadInts(IntPtr device, int[] host)
    {
        var sizeBytes = (UIntPtr)((ulong)host.Length * sizeof(int));
        fixed (int* src = host)
        {
            var status = HipNativeBindings.hipMemcpy(device, (IntPtr)src, sizeBytes, HipMemcpyKind.HostToDevice);
            HipNativeBindings.CheckError(status, "hipMemcpy(int[])");
        }
    }

    private static void CheckSp(RocSparseNative.Status s, string what)
    {
        if (s != RocSparseNative.Status.Success)
            throw new InvalidOperationException($"{what} failed with rocSPARSE status {s}.");
    }
}
