// Copyright (c) AiDotNet. All rights reserved.
// Vulkan host-array CSR·dense SpMM + SDDMM dispatch — mirrors OpenClSparseBackend
// so any Vulkan-capable device (desktop GPU, mobile GPU via cross-vendor Vulkan,
// Apple via MoltenVK) accelerates SparseOps.SpMM / SparseOps.SDDMM and, through
// them, the tape-aware sparse autograd backward.

using System;

namespace AiDotNet.Tensors.Engines.DirectGpu.Vulkan
{
    /// <summary>
    /// Host-array CSR · dense → dense SpMM and SDDMM for the Vulkan backend. Wires
    /// <c>SparseOps.SpMM</c> and <c>SparseOps.SDDMM</c>'s GPU tier to the Vulkan
    /// compute pipelines defined in <see cref="VulkanSparseKernels"/> via
    /// <see cref="VulkanBackend.CsrSpMM"/> / <see cref="VulkanBackend.CsrSddmm"/>.
    ///
    /// <para>Vulkan sits after CUDA, HIP, MPS, and OpenCL in the SpMM/SDDMM tier list
    /// so vendor libraries win when present; Vulkan is the portable fallback for
    /// devices without a vendor-native sparse path (Adreno, Mali, ARM Immortalis,
    /// MoltenVK-on-Apple when MPS is unavailable, cross-vendor desktop drivers).</para>
    /// </summary>
    internal static class VulkanSparseBackend
    {
        // VulkanBackend construction allocates device + command pool + descriptor
        // pool and compiles the shader cache — expensive. Reuse the process-lifetime
        // shared instance (VulkanBackend.Instance) instead of constructing our own.
        /// <summary>True when Vulkan is available on this host (loader + physical device).</summary>
        public static bool IsAvailable => VulkanDevicePrimitives.IsAvailable;

        // The buffer pool may hand back a buffer physically LARGER than requested;
        // DownloadBuffer copies the buffer's physical element count, which would
        // overflow a snug destination. Download into a buffer-sized scratch and
        // copy back exactly the logical elements. Mirrors OpenClSparseBackend.DownloadExact.
        private static void DownloadExact(VulkanBackend backend, IGpuBuffer buffer, float[] destination)
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

        private static VulkanBackend GetOrCreate()
        {
            var backend = VulkanBackend.Instance;
            if (!backend.IsAvailable)
                throw new InvalidOperationException("Vulkan SpMM backend failed to initialise.");
            // GLSL runtime compilation is required — sparse kernels are compiled from
            // GLSL source strings, not pre-baked SPIR-V. If shaderc isn't available,
            // throw so the SparseOps tier list falls through to a different backend
            // rather than dispatching into a kernel that would fail to compile.
            if (!backend.IsGlslCompilerAvailable)
                throw new InvalidOperationException(
                    "Vulkan sparse compute requires the runtime GLSL compiler (libshaderc). " +
                    "Install shaderc or route to a different GPU backend (CUDA/OpenCL).");
            return backend;
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
                throw new InvalidOperationException("Vulkan SpMM backend is not available.");

            var backend = GetOrCreate();
            var output = new float[rows * n];
            if (rows == 0 || n == 0) return output;

            IGpuBuffer? valuesBuf = null, bBuf = null, outBuf = null;
            IGpuBuffer? rowPtrBuf = null, colIdxBuf = null;
            try
            {
                valuesBuf = values.Length > 0
                    ? backend.AllocateBuffer(values)
                    : backend.AllocateBuffer(1); // Vulkan buffers must be > 0 bytes; nnz=0 output is zero anyway.
                bBuf = backend.AllocateBuffer(b);
                outBuf = backend.AllocateBuffer(output);
                rowPtrBuf = backend.AllocateIntBuffer(rowPtr);
                colIdxBuf = colIdx.Length > 0
                    ? backend.AllocateIntBuffer(colIdx)
                    : backend.AllocateIntBuffer(1);

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
        /// <c>nnz</c>-length values array. Runs the Vulkan <c>sddmm</c> compute shader on the GPU.
        /// </summary>
        public static float[] SDDMM(
            int[] rowIndices, int[] colIndices,
            float[] x, float[] y, int innerK)
        {
            if (!IsAvailable)
                throw new InvalidOperationException("Vulkan SDDMM backend is not available.");

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
