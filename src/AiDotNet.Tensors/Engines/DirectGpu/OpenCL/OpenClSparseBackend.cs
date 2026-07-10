// Copyright (c) AiDotNet. All rights reserved.
// OpenCL host-array CSR·dense SpMM dispatch (AMD / Intel GPUs).

using System;

namespace AiDotNet.Tensors.Engines.DirectGpu.OpenCL
{
    /// <summary>
    /// Host-array CSR · dense → dense SpMM for the OpenCL backend, mirroring
    /// <see cref="CUDA.CudaSparseBackend"/> and <see cref="HIP.HipCustomSparseBackend"/>. Wires
    /// <c>SparseOps.SpMM</c>'s GPU tier to the existing <c>csr_spmm</c> OpenCL kernel via
    /// <see cref="OpenClBackend.CsrSpMM"/> so AMD/Intel GPUs accelerate sparse matmul (and,
    /// through it, the tape-aware sparse autograd backward's dB = Aᵀ·grad).
    /// </summary>
    internal static class OpenClSparseBackend
    {
        // Unlike CUDA/HIP (cheap to construct), an OpenClBackend eagerly compiles its full
        // kernel program on construction, so building one per SpMM call would be ruinous. Cache a
        // single shared instance (lazy, double-checked) and reuse it for every dispatch. The
        // instance lives for the process lifetime — the same lifetime the GPU context would have.
        private static readonly object _gate = new object();
        private static OpenClBackend? _cached;

        /// <summary>True when an OpenCL device/context is available on this host.</summary>
        public static bool IsAvailable => OpenClBackend.IsOpenClAvailable;

        // The buffer pool may hand back a buffer physically LARGER than requested (e.g. a small
        // odd-sized output renting an 8-element buffer left over from a prior op). DownloadBuffer
        // copies the buffer's physical element count, which would overflow a snug destination, so
        // download into a buffer-sized scratch and copy back exactly the logical elements.
        private static void DownloadExact(OpenClBackend backend, IGpuBuffer buffer, float[] destination)
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

        private static OpenClBackend GetOrCreate()
        {
            var existing = _cached;
            if (existing is not null && existing.IsAvailable)
                return existing;

            lock (_gate)
            {
                if (_cached is not null && _cached.IsAvailable)
                    return _cached;

                var backend = new OpenClBackend(0);
                if (!backend.IsAvailable)
                    throw new InvalidOperationException("OpenCL SpMM backend failed to initialise.");
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
                throw new InvalidOperationException("OpenCL SpMM backend is not available.");

            var backend = GetOrCreate();
            var output = new float[rows * n];

            // Allocate inside the try so a throw along the upload chain still frees whatever was
            // already allocated (mirrors the CUDA/HIP leak-safe cleanup).
            IGpuBuffer? valuesBuf = null, bBuf = null, outBuf = null;
            IGpuBuffer? rowPtrBuf = null, colIdxBuf = null;
            try
            {
                valuesBuf = backend.AllocateBuffer(values);
                bBuf = backend.AllocateBuffer(b);
                outBuf = backend.AllocateBuffer(output);
                // AllocateIntBuffer stores the int bit-patterns; the csr_spmm kernel binds these
                // buffers as int* and reads them back losslessly.
                rowPtrBuf = backend.AllocateIntBuffer(rowPtr);
                colIdxBuf = backend.AllocateIntBuffer(colIdx);

                // CsrSpMM arg order: (values, colIndices, rowPointers, denseB, output, M, K, N, nnz).
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
        /// <c>nnz</c>-length values array. Runs the OpenCL <c>sddmm</c> kernel on the GPU.
        /// </summary>
        public static float[] SDDMM(
            int[] rowIndices, int[] colIndices,
            float[] x, float[] y, int innerK)
        {
            if (!IsAvailable)
                throw new InvalidOperationException("OpenCL SDDMM backend is not available.");

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

                // Adaptive dispatch (beyond-industry-standard): for large innerK, use the
                // workgroup-per-nnz collaborative kernel where 256 work-items share the reduction
                // via local memory. Reduces per-nnz latency from O(innerK) to O(log₂ 256) +
                // O(innerK/256). Threshold matches the Vulkan/Metal tiers.
                const int CollabInnerKThreshold = 64;
                if (innerK >= CollabInnerKThreshold)
                    backend.CsrSddmmCollab(rowBuf, colBuf, xBuf, yBuf, outBuf, nnz, innerK);
                else
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
