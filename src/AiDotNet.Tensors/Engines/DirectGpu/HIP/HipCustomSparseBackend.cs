// Copyright (c) AiDotNet. All rights reserved.

using System;

namespace AiDotNet.Tensors.Engines.DirectGpu.HIP;

/// <summary>
/// HIP/ROCm equivalent of <see cref="CUDA.CudaSparseBackend"/>: CSR · dense → dense
/// SpMM using AiDotNet's own managed HIP kernels (<c>csr_spmm</c> /
/// <c>csr_spmm_warp</c> / <c>csr_spmm_double</c> in
/// <see cref="Kernels.HipSparseKernels"/>), with the same host-array entry shape as
/// <see cref="HipSparseBackend"/> but dispatching the managed kernels instead of
/// rocSPARSE — the supply-chain-removal default for AMD GPUs (issue #515, P6).
///
/// <para><b>Status (#515): implemented but NOT yet wired into
/// <see cref="LinearAlgebra.Sparse.SparseOps"/> dispatch.</b> The only cheap
/// availability signal, <see cref="HipBackend.IsHipAvailable"/>, reports true whenever
/// the HIP runtime sees a device (it can route over a CUDA device) even when the HIP
/// kernel toolchain can't build a usable binary for the active arch (e.g. a
/// gfx-mismatched / header-missing install). Routing on that signal would replace the
/// safe CPU/vendor fallback with a failing backend. Wiring this in needs a
/// kernel-load-strict availability gate, validated end-to-end on a healthy ROCm
/// runner. The kernels + launches are a faithful mirror of the CUDA path so the wiring
/// is a small follow-up once such a runner exists.</para>
/// </summary>
internal static class HipCustomSparseBackend
{
    /// <summary>N (dense column count) at or below which the warp-per-row kernel is
    /// preferred over the 2-D scalar kernel (mirrors
    /// <see cref="CUDA.CudaSparseBackend"/>; first-cut, to be tuned by the #515 bar).</summary>
    private const int WarpColumnThreshold = 64;

    /// <summary>Needs only a constructible HIP runtime — no rocSPARSE library, which
    /// is the point of the supply-chain removal.</summary>
    public static bool IsAvailable => HipBackend.IsHipAvailable;

    /// <summary>CSR · dense → dense, host-side <c>float[]</c> CSR inputs. Warp-per-row
    /// kernel for small/moderate N, 2-D scalar for wide N.</summary>
    public static float[] SpMM(
        int[] rowPtr, int[] colIdx, float[] values,
        float[] b, int rows, int cols, int n)
    {
        if (!IsAvailable)
            throw new InvalidOperationException("HIP custom-kernel SpMM backend is not available.");

        var backend = new HipBackend();
        if (!backend.IsAvailable)
            throw new InvalidOperationException("HIP backend failed to initialise.");

        var output = new float[rows * n];

        IGpuBuffer? valuesBuf = null, bBuf = null, outBuf = null;
        IGpuBuffer? rowPtrBuf = null, colIdxBuf = null;
        try
        {
            valuesBuf = backend.AllocateBuffer(values);
            bBuf = backend.AllocateBuffer(b);
            outBuf = backend.AllocateBuffer(output);
            rowPtrBuf = backend.AllocateByteBuffer(rowPtr.Length * sizeof(int));
            colIdxBuf = backend.AllocateByteBuffer(colIdx.Length * sizeof(int));
            UploadInts(rowPtrBuf.Handle, rowPtr);
            UploadInts(colIdxBuf.Handle, colIdx);

            if (n <= WarpColumnThreshold)
                backend.CsrSpMMWarp(valuesBuf, colIdxBuf, rowPtrBuf, bBuf, outBuf,
                    rows, cols, n, values.Length);
            else
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

    /// <summary>FP64 CSR · dense → dense (issue #515). Double payloads ride byte
    /// buffers, dispatched through <c>csr_spmm_double</c>.</summary>
    public static double[] SpMM(
        int[] rowPtr, int[] colIdx, double[] values,
        double[] b, int rows, int cols, int n)
    {
        if (!IsAvailable)
            throw new InvalidOperationException("HIP custom-kernel SpMM backend is not available.");

        var backend = new HipBackend();
        if (!backend.IsAvailable)
            throw new InvalidOperationException("HIP backend failed to initialise.");

        var output = new double[rows * n];

        IGpuBuffer? valuesBuf = null, bBuf = null, outBuf = null;
        IGpuBuffer? rowPtrBuf = null, colIdxBuf = null;
        try
        {
            valuesBuf = backend.AllocateByteBuffer(values.Length * sizeof(double));
            bBuf = backend.AllocateByteBuffer(b.Length * sizeof(double));
            outBuf = backend.AllocateByteBuffer(output.Length * sizeof(double));
            UploadDoubles(valuesBuf.Handle, values);
            UploadDoubles(bBuf.Handle, b);

            rowPtrBuf = backend.AllocateByteBuffer(rowPtr.Length * sizeof(int));
            colIdxBuf = backend.AllocateByteBuffer(colIdx.Length * sizeof(int));
            UploadInts(rowPtrBuf.Handle, rowPtr);
            UploadInts(colIdxBuf.Handle, colIdx);

            backend.CsrSpMMDouble(valuesBuf, colIdxBuf, rowPtrBuf, bBuf, outBuf,
                rows, cols, n, values.Length);

            DownloadDoubles(outBuf.Handle, output);
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

    private static unsafe void UploadInts(IntPtr device, int[] host)
    {
        var bytes = (UIntPtr)((ulong)host.Length * sizeof(int));
        fixed (int* src = host)
            HipNativeBindings.CheckError(
                HipNativeBindings.hipMemcpy(device, (IntPtr)src, bytes, HipMemcpyKind.HostToDevice),
                "hipMemcpy(int[])");
    }

    private static unsafe void UploadDoubles(IntPtr device, double[] host)
    {
        var bytes = (UIntPtr)((ulong)host.Length * sizeof(double));
        fixed (double* src = host)
            HipNativeBindings.CheckError(
                HipNativeBindings.hipMemcpy(device, (IntPtr)src, bytes, HipMemcpyKind.HostToDevice),
                "hipMemcpy(double[])");
    }

    private static unsafe void DownloadDoubles(IntPtr device, double[] host)
    {
        var bytes = (UIntPtr)((ulong)host.Length * sizeof(double));
        fixed (double* dst = host)
            HipNativeBindings.CheckError(
                HipNativeBindings.hipMemcpy((IntPtr)dst, device, bytes, HipMemcpyKind.DeviceToHost),
                "hipMemcpy(double[] download)");
    }
}
