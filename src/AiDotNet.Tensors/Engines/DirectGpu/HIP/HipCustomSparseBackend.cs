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
/// <para>Routed by <see cref="LinearAlgebra.Sparse.SparseOps"/> ahead of rocSPARSE for float and double SpMM and for
/// SDDMM. Availability is kernel-load strict (see <see cref="Shared"/>).</para>
/// </summary>
internal static class HipCustomSparseBackend
{
    /// <summary>N (dense column count) at or below which the warp-per-row kernel is
    /// preferred over the 2-D scalar kernel (mirrors
    /// <see cref="CUDA.CudaSparseBackend"/>; first-cut, to be tuned by the #515 bar).</summary>
    private const int WarpColumnThreshold = 64;

    private static readonly object SharedLock = new object();
    private static readonly object CallLock = new object();
    private static HipBackend? _shared;
    private static bool _sharedTried;

    /// <summary>
    /// The process-lifetime backend these host-array entry points run on, built on first use. A HipBackend compiles
    /// every kernel module when constructed (about 50 s on an RX 5500 XT), so one per call, as these wrappers used to
    /// do, made the path unusable. Null unless construction succeeded AND the SpMM kernels actually compiled: the HIP
    /// runtime seeing a device (HipBackend.IsHipAvailable) does not prove the toolchain can build for its arch.
    /// </summary>
    internal static HipBackend? Shared
    {
        get
        {
            lock (SharedLock)
            {
                if (_sharedTried) return _shared;
                _sharedTried = true;
                if (!HipBackend.IsHipAvailable) return null;
                HipBackend? candidate = null;
                try
                {
                    candidate = new HipBackend();
                    if (candidate.IsAvailable && Array.TrueForAll(RequiredKernels, candidate.HasCompiledKernel))
                    {
                        _shared = candidate;
                        candidate = null;
                    }
                    else
                    {
                        System.Diagnostics.Trace.TraceWarning(
                            "HIP custom sparse backend unavailable: the HIP backend did not initialise or did not compile the SpMM kernels.");
                    }
                }
                catch (Exception ex) when (ex is DllNotFoundException or EntryPointNotFoundException or InvalidOperationException or HipException)
                {
                    System.Diagnostics.Trace.TraceWarning($"HIP custom sparse backend unavailable ({ex.GetType().Name}): {ex.Message}");
                }
                finally
                {
                    candidate?.Dispose();
                }
                return _shared;
            }
        }
    }

    private static readonly string[] RequiredKernels = { "csr_spmm", "csr_spmm_warp", "csr_spmm_vec4", "csr_spmm_double", "sddmm" };

    /// <summary>
    /// Whether the managed kernels can run here: a HIP backend that initialised and compiled them. No rocSPARSE library,
    /// which is the point of the supply-chain removal. The first query builds the shared backend.
    /// </summary>
    public static bool IsAvailable => Shared is not null;

    /// <summary>CSR · dense → dense, host-side <c>float[]</c> CSR inputs. Warp-per-row
    /// kernel for small/moderate N, 2-D scalar for wide N.</summary>
    public static float[] SpMM(
        int[] rowPtr, int[] colIdx, float[] values,
        float[] b, int rows, int cols, int n)
    {
        HipBackend backend = Shared ?? throw new GpuCapabilityUnavailableException("HIP custom-kernel sparse backend is not available.");
        // One backend, one stream: calls from several threads must not interleave their uploads, launches and downloads.
        lock (CallLock)
        {
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

                // Same pick as CudaSparseBackend: vec4 when N % 4 == 0, warp-per-row for narrow N, else 2-D scalar.
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
    }

    /// <summary>SDDMM host wrapper: <c>out[p] = Σ_k x[rowIndices[p], k] · y[colIndices[p], k]</c>.
    /// CUDA mirror: <see cref="CUDA.CudaSparseBackend.SDDMM"/>.</summary>
    public static float[] SDDMM(
        int[] rowIndices, int[] colIndices,
        float[] x, float[] y, int innerK)
    {
        HipBackend backend = Shared ?? throw new GpuCapabilityUnavailableException("HIP custom-kernel sparse backend is not available.");
        // One backend, one stream: calls from several threads must not interleave their uploads, launches and downloads.
        lock (CallLock)
        {
            int nnz = rowIndices.Length;
            var output = new float[nnz];
            if (nnz == 0) return output;

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
    }

    /// <summary>FP64 CSR · dense → dense (issue #515). Double payloads ride byte
    /// buffers, dispatched through <c>csr_spmm_double</c>.</summary>
    public static double[] SpMM(
        int[] rowPtr, int[] colIdx, double[] values,
        double[] b, int rows, int cols, int n)
    {
        HipBackend backend = Shared ?? throw new GpuCapabilityUnavailableException("HIP custom-kernel sparse backend is not available.");
        // One backend, one stream: calls from several threads must not interleave their uploads, launches and downloads.
        lock (CallLock)
        {
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
