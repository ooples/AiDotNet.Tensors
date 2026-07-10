// Copyright (c) AiDotNet. All rights reserved.
// Metal host-array CSR·dense SpMM + SDDMM dispatch — Metal-native compute (MSL)
// alternative to the MPS vendor path (MpsSparseBackend). Where MpsSparseBackend
// wraps MPSSparseMatrixVectorMultiplication (requires an Apple-Silicon runner
// to validate the descriptor lifecycle, hence IsAvailable = false at rest),
// this dispatcher runs the AiDotNet-owned MSL kernels in MetalSparseKernels.cs
// through the same pipeline machinery as every other Metal op — no vendor lib
// dependency.

using System;

namespace AiDotNet.Tensors.Engines.DirectGpu.Metal
{
    /// <summary>
    /// Host-array CSR · dense → dense SpMM and SDDMM for the Metal backend. Wires
    /// <c>SparseOps.SpMM</c> and <c>SparseOps.SDDMM</c>'s GPU tier to the Metal
    /// compute pipelines defined in <see cref="MetalSparseKernels"/> via
    /// <see cref="MetalBackend.CsrSpMM"/> / <see cref="MetalBackend.CsrSddmm"/>.
    ///
    /// <para>On the SparseOps tier list, this sits after CUDA / cuSPARSE / HIP /
    /// MPS but before OpenCL — MPS wins when present (vendor library), Metal-native
    /// compute is the fallback for Apple hosts where MPS sparse isn't wired.</para>
    /// </summary>
    internal static class MetalSparseBackend
    {
        // MetalBackend construction touches the Metal device + command queue + shader
        // library compilation — building one per SpMM call would be ruinous. Cache a
        // single shared instance (lazy, double-checked) for the process lifetime.
        private static readonly object _gate = new object();
        private static MetalBackend? _cached;

        /// <summary>True when Metal is available on this host (macOS + Metal-compatible GPU).</summary>
        public static bool IsAvailable => MetalNativeBindings.IsPlatformSupported;

        // The buffer pool may hand back a buffer physically LARGER than requested;
        // DownloadBuffer copies the buffer's physical element count, which would
        // overflow a snug destination. Download into a buffer-sized scratch and
        // copy back exactly the logical elements. Mirrors OpenClSparseBackend.DownloadExact.
        private static void DownloadExact(MetalBackend backend, IGpuBuffer buffer, float[] destination)
        {
            if (buffer.Size == destination.Length)
            {
                backend.DownloadBuffer(buffer, destination);
                return;
            }
            var scratch = new float[buffer.Size];
            backend.DownloadBuffer(buffer, scratch);
            Array.Copy(scratch, destination, destination.Length);
        }

        private static MetalBackend GetOrCreate()
        {
            var existing = _cached;
            if (existing is not null && existing.IsAvailable)
                return existing;

            lock (_gate)
            {
                if (_cached is not null && _cached.IsAvailable)
                    return _cached;

                var backend = new MetalBackend();
                if (!backend.IsAvailable)
                    throw new InvalidOperationException("Metal SpMM backend failed to initialise.");
                _cached = backend;
                return backend;
            }
        }

        /// <summary>
        /// CSR · dense → dense with host-side <c>float[]</c> CSR inputs (values/rowPtr/colIdx) and
        /// a dense <c>float[]</c> B. Returns the dense output flat array of length <c>rows · n</c>.
        /// </summary>
        public static float[] SpMM(
            int[] rowPtr, int[] colIdx, float[] values,
            float[] b, int rows, int cols, int n)
        {
            if (!IsAvailable)
                throw new InvalidOperationException("Metal SpMM backend is not available.");

            var backend = GetOrCreate();
            var output = new float[rows * n];
            if (rows == 0 || n == 0) return output;

            IGpuBuffer? valuesBuf = null, bBuf = null, outBuf = null;
            IGpuBuffer? rowPtrBuf = null, colIdxBuf = null;
            try
            {
                // Metal AllocateIntBuffer(int[]) rejects empty arrays; nnz=0 needs a
                // 1-element scratch (the kernel never reads it — rowPtr's diffs are all 0).
                valuesBuf = values.Length > 0
                    ? backend.AllocateBuffer(values)
                    : backend.AllocateBuffer(1);
                bBuf = backend.AllocateBuffer(b);
                outBuf = backend.AllocateBuffer(output);
                rowPtrBuf = backend.AllocateIntBuffer(rowPtr);
                colIdxBuf = colIdx.Length > 0
                    ? backend.AllocateIntBuffer(colIdx)
                    : backend.AllocateIntBuffer(new[] { 0 });

                backend.CsrSpMM(valuesBuf, colIdxBuf, rowPtrBuf, bBuf, outBuf,
                    rows, cols, n, values.Length);

                DownloadExact(backend, outBuf, output);
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
        /// SDDMM with host-side inputs: for each pattern entry p computes
        /// <c>out[p] = Σ_k x[rowIndices[p], k] · y[colIndices[p], k]</c>. <paramref name="x"/> is
        /// row-major [M, innerK], <paramref name="y"/> is row-major [Ny, innerK]. Returns the
        /// <c>nnz</c>-length values array. Runs the Metal <c>sparse_sddmm</c> compute shader.
        /// </summary>
        public static float[] SDDMM(
            int[] rowIndices, int[] colIndices,
            float[] x, float[] y, int innerK)
        {
            if (!IsAvailable)
                throw new InvalidOperationException("Metal SDDMM backend is not available.");

            int nnz = rowIndices.Length;
            var output = new float[nnz];
            if (nnz == 0) return output;

            var backend = GetOrCreate();
            IGpuBuffer? rowBuf = null, colBuf = null, xBuf = null, yBuf = null, outBuf = null;
            try
            {
                rowBuf = backend.AllocateIntBuffer(rowIndices);
                colBuf = backend.AllocateIntBuffer(colIndices);
                xBuf = backend.AllocateBuffer(x);
                yBuf = backend.AllocateBuffer(y);
                outBuf = backend.AllocateBuffer(output);

                backend.CsrSddmm(rowBuf, colBuf, xBuf, yBuf, outBuf, nnz, innerK);

                DownloadExact(backend, outBuf, output);
                return output;
            }
            finally
            {
                outBuf?.Dispose();
                yBuf?.Dispose();
                xBuf?.Dispose();
                colBuf?.Dispose();
                rowBuf?.Dispose();
            }
        }
    }
}
