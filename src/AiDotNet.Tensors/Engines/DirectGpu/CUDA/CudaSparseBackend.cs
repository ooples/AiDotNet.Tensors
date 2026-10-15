// Copyright (c) AiDotNet. All rights reserved.

using System;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA;

/// <summary>
/// CSR · dense → dense SpMM using AiDotNet's own managed CUDA kernels
/// (<c>csr_spmm</c> / <c>csr_spmm_warp</c> / <c>csr_spmm_double</c> in
/// <see cref="Kernels.CudaSparseKernels"/>), behind host-array entry points: allocate
/// device buffers, upload the CSR triple + dense B, launch, download the dense result.
///
/// <para>This replaced native cuSPARSE (issue #515): it is the GPU SpMM path
/// <see cref="LinearAlgebra.Sparse.SparseOps"/> uses on NVIDIA, and it needs no external
/// library, only the CUDA driver + NVRTC the rest of <see cref="CudaBackend"/> already
/// requires.</para>
///
/// <para>The launched kernels accumulate each output element in a single thread over
/// the row's stored order, so results are deterministic and match the managed CPU
/// CSR reference to FMA tolerance. The float path picks the warp-per-row kernel for
/// small/moderate N (coalesced B reads) and the 2-D scalar kernel for wide N; the
/// FP64 path uses the scalar double kernel.</para>
/// </summary>
internal static class CudaSparseBackend
{
    /// <summary>N (dense column count) at or below which the warp-per-row kernel is
    /// preferred over the 2-D scalar kernel. First-cut threshold — to be tuned by the
    /// #515 perf bar on real hardware (the warp kernel covers columns 32-at-a-time per
    /// row, so it wins while N is small; wide N favours the scalar kernel's extra
    /// block-level parallelism).</summary>
    private const int WarpColumnThreshold = 64;

    /// <summary>Whether the managed custom-kernel SpMM path can run here. It needs
    /// only a constructible <see cref="CudaBackend"/> (CUDA driver + NVRTC); no
    /// cuSPARSE library, which is the whole point of the supply-chain removal.</summary>
    public static bool IsAvailable => CudaBackend.IsCudaAvailable;

    // One backend for every call. These entry points used to build a new CudaBackend per call and never dispose
    // it: each call paid backend construction plus NVRTC for kernels the previous call had already compiled, and
    // left the backend behind, so a host-array SpMM took 65-140 ms and got slower with every call. Calls are
    // serialized on the shared backend's stream, and the lease makes its context current on the calling thread
    // (the raw cuMemcpy copies below need it).
    private static readonly object BackendGate = new object();
    private static CudaBackend? _sharedBackend;

    private readonly struct BackendLease : IDisposable
    {
        public CudaBackend Backend { get; }

        private BackendLease(CudaBackend backend) => Backend = backend;

        public static BackendLease Acquire()
        {
            System.Threading.Monitor.Enter(BackendGate);
            try
            {
                var backend = _sharedBackend ??= CudaBackend.CreateOrThrow();
                backend.EnsureContextCurrent();
                return new BackendLease(backend);
            }
            catch
            {
                System.Threading.Monitor.Exit(BackendGate);
                throw;
            }
        }

        public void Dispose() => System.Threading.Monitor.Exit(BackendGate);
    }

    /// <summary>
    /// CSR · dense → dense, host-side <c>float[]</c> CSR inputs. Returns the dense
    /// output flat array of length <c>rows · n</c>, from the managed
    /// <c>csr_spmm</c> / <c>csr_spmm_warp</c> / <c>csr_spmm_vec4</c> kernels.
    /// </summary>
    public static float[] SpMM(
        int[] rowPtr, int[] colIdx, float[] values,
        float[] b, int rows, int cols, int n)
    {
        if (!IsAvailable)
            throw new GpuCapabilityUnavailableException("CUDA custom-kernel SpMM backend is not available.");

        using var lease = BackendLease.Acquire();
        var backend = lease.Backend;
        var output = new float[rows * n];

        // Every allocation sits inside the try so a throw anywhere along the
        // upload chain still hits the finally and frees the device buffers
        // already allocated.
        IGpuBuffer? valuesBuf = null, bBuf = null, outBuf = null;
        IGpuBuffer? rowPtrBuf = null, colIdxBuf = null;
        try
        {
            valuesBuf = backend.AllocateBuffer(values);
            bBuf = backend.AllocateBuffer(b);
            outBuf = backend.AllocateBuffer(output);

            // int[] payloads ride a byte buffer.
            rowPtrBuf = backend.AllocateByteBuffer(rowPtr.Length * sizeof(int));
            colIdxBuf = backend.AllocateByteBuffer(colIdx.Length * sizeof(int));
            UploadInts(rowPtrBuf.Handle, rowPtr);
            UploadInts(colIdxBuf.Handle, colIdx);

            // Kernel pick (first-cut; to be tuned by the #515 perf bar):
            //   N % 4 == 0      -> vec4 (128-bit aligned float4 dense loads)
            //   N <= 64         -> warp-per-row (coalesced lane-per-column)
            //   otherwise       -> 2-D scalar
            // Argument order matches IDirectGpuBackend.CsrSpMM:
            // (values, colIndices, rowPointers, denseB, output, M, K, N, nnz).
            if (n % 4 == 0)
                backend.CsrSpMMVec4(valuesBuf, colIdxBuf, rowPtrBuf, bBuf, outBuf,
                    rows, cols, n, values.Length);
            else if (n <= WarpColumnThreshold)
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

    /// <summary>
    /// SDDMM with host-side inputs: <c>out[p] = Σ_k x[rowIndices[p], k] · y[colIndices[p], k]</c>.
    /// <paramref name="x"/> is [M, innerK] row-major, <paramref name="y"/> is [Ny, innerK] row-major.
    /// Returns the <c>nnz</c>-length values array. Dispatches the managed <c>sddmm</c> kernel.
    /// </summary>
    public static float[] SDDMM(
        int[] rowIndices, int[] colIndices,
        float[] x, float[] y, int innerK)
    {
        if (!IsAvailable)
            throw new GpuCapabilityUnavailableException("CUDA custom-kernel SDDMM backend is not available.");

        int nnz = rowIndices.Length;
        var output = new float[nnz];
        if (nnz == 0) return output;

        using var lease = BackendLease.Acquire();
        var backend = lease.Backend;

        IGpuBuffer? rowBuf = null, colBuf = null, xBuf = null, yBuf = null, outBuf = null;
        try
        {
            xBuf = backend.AllocateBuffer(x);
            yBuf = backend.AllocateBuffer(y);
            outBuf = backend.AllocateBuffer(output);
            rowBuf = backend.AllocateByteBuffer(rowIndices.Length * sizeof(int));
            colBuf = backend.AllocateByteBuffer(colIndices.Length * sizeof(int));
            UploadInts(rowBuf.Handle, rowIndices);
            UploadInts(colBuf.Handle, colIndices);

            backend.CsrSddmm(rowBuf, colBuf, xBuf, yBuf, outBuf, nnz, innerK);

            backend.DownloadBuffer(outBuf, output);
            return output;
        }
        finally
        {
            outBuf?.Dispose();
            yBuf?.Dispose();
            xBuf?.Dispose();
            rowBuf?.Dispose();
            colBuf?.Dispose();
        }
    }

    /// <summary>
    /// FP64 CSR · dense → dense (issue #515). Same shape as the float
    /// <see cref="SpMM(int[], int[], float[], float[], int, int, int)"/> but the
    /// value / dense / output payloads are <c>double</c>, carried on byte buffers
    /// and dispatched through <c>csr_spmm_double</c>. This fills the gap where the
    /// double tier previously had no GPU path at all.
    /// </summary>
    public static double[] SpMM(
        int[] rowPtr, int[] colIdx, double[] values,
        double[] b, int rows, int cols, int n)
    {
        if (!IsAvailable)
            throw new GpuCapabilityUnavailableException("CUDA custom-kernel SpMM backend is not available.");

        using var lease = BackendLease.Acquire();
        var backend = lease.Backend;
        var output = new double[rows * n];

        IGpuBuffer? valuesBuf = null, bBuf = null, outBuf = null;
        IGpuBuffer? rowPtrBuf = null, colIdxBuf = null;
        try
        {
            // double payloads ride byte buffers (CudaBackend's typed buffer API is
            // float-only); raw HtoD/DtoH copies move them.
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

    /// <summary>Synchronous host→device copy for an <c>int[]</c> payload via the
    /// driver's <c>cuMemcpyHtoD</c>.</summary>
    private static unsafe void UploadInts(IntPtr device, int[] host)
    {
        ulong byteCount = (ulong)host.Length * sizeof(int);
        fixed (int* src = host)
        {
            var status = CuBlasNative.cuMemcpyHtoD(device, (IntPtr)src, byteCount);
            CuBlasNative.CheckCudaResult(status, "cuMemcpyHtoD(int[])");
        }
    }

    /// <summary>Synchronous host→device copy for a <c>double[]</c> payload.</summary>
    private static unsafe void UploadDoubles(IntPtr device, double[] host)
    {
        ulong byteCount = (ulong)host.Length * sizeof(double);
        fixed (double* src = host)
        {
            var status = CuBlasNative.cuMemcpyHtoD(device, (IntPtr)src, byteCount);
            CuBlasNative.CheckCudaResult(status, "cuMemcpyHtoD(double[])");
        }
    }

    /// <summary>Synchronous device→host copy into a <c>double[]</c> via <c>cuMemcpyDtoH</c>.</summary>
    private static unsafe void DownloadDoubles(IntPtr device, double[] host)
    {
        ulong byteCount = (ulong)host.Length * sizeof(double);
        fixed (double* dst = host)
        {
            var status = CuBlasNative.cuMemcpyDtoH((IntPtr)dst, device, byteCount);
            CuBlasNative.CheckCudaResult(status, "cuMemcpyDtoH(double[])");
        }
    }
}
