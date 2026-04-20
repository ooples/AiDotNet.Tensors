// Copyright (c) AiDotNet. All rights reserved.
// Secondary capability interface for backends that ship native GPU kernels
// for the torch.linalg op surface. DirectGpuTensorEngine type-tests against
// this interface when dispatching Cholesky / LU / QR / Eigh; backends that
// don't implement it transparently fall through to the CpuEngine path.

namespace AiDotNet.Tensors.Engines.DirectGpu
{
    /// <summary>
    /// Optional capability interface for GPU backends that provide native
    /// kernels for the core <c>torch.linalg</c> decompositions — issue #211
    /// moat #2 ("Batched Cholesky / LU / QR on GPU"). All methods are
    /// synchronous per project convention; async variants live on the
    /// concrete backend types where needed (e.g. WebGPU's dispatch).
    ///
    /// <para>Inputs and outputs are all fp32 <see cref="IGpuBuffer"/> with
    /// the matrix stored row-major in the last two dims. Batching is flat:
    /// the caller multiplies any leading batch dimensions into a single
    /// <c>batchCount</c> scalar, and the kernel processes each batch slice
    /// independently. This avoids per-backend batch-dim semantics while
    /// keeping kernels straightforward.</para>
    /// </summary>
    public interface ILinalgBackend
    {
        /// <summary>
        /// Cholesky factorization <c>A = L · Lᵀ</c> (lower) or <c>A = Uᵀ · U</c>
        /// (upper) for symmetric positive-definite A. Per-batch <paramref name="info"/>
        /// is written 0 on success, k+1 if the leading k+1 × k+1 minor was not SPD.
        /// </summary>
        /// <param name="input">Packed (batchCount * n * n) fp32 SPD matrices.</param>
        /// <param name="output">Packed factor (lower or upper triangle populated; opposite zeroed).</param>
        /// <param name="info">Per-batch int32 status (length batchCount).</param>
        /// <param name="upper">When true, produce upper-triangular factor; else lower.</param>
        void LinalgCholesky(IGpuBuffer input, IGpuBuffer output, IGpuBuffer info,
            int batchCount, int n, bool upper);

        /// <summary>
        /// LU factorization with partial row pivoting — <c>P·A = L·U</c>. The
        /// packed output stores <c>L</c> in the strict lower triangle (unit
        /// diagonal implicit) and <c>U</c> in the upper + diagonal.
        /// </summary>
        /// <param name="input">Packed (batchCount * m * n) fp32 matrices.</param>
        /// <param name="output">Packed L\U factor of the same shape.</param>
        /// <param name="pivots">Per-batch row-pivot vector (batchCount * min(m, n) int32).</param>
        void LinalgLuFactor(IGpuBuffer input, IGpuBuffer output, IGpuBuffer pivots,
            int batchCount, int m, int n);

        /// <summary>
        /// QR factorization via Householder reflectors. Reduced mode only for v1
        /// (Q is m × k, R is k × n, k = min(m, n)). Complete / R-only modes use
        /// the CPU fallback until the GPU Householder aggregation lands.
        /// </summary>
        /// <param name="input">Packed input (batchCount * m * n).</param>
        /// <param name="q">Packed Q output (batchCount * m * k).</param>
        /// <param name="r">Packed R output (batchCount * k * n).</param>
        void LinalgQrReduced(IGpuBuffer input, IGpuBuffer q, IGpuBuffer r,
            int batchCount, int m, int n);

        /// <summary>
        /// Symmetric / Hermitian eigendecomposition via cyclic Jacobi rotations.
        /// Writes eigenvalues ascending into <paramref name="eigenvalues"/> and
        /// corresponding eigenvectors as columns of <paramref name="eigenvectors"/>.
        /// </summary>
        /// <param name="input">Packed symmetric (batchCount * n * n).</param>
        /// <param name="eigenvalues">Packed (batchCount * n).</param>
        /// <param name="eigenvectors">Packed (batchCount * n * n).</param>
        void LinalgEigh(IGpuBuffer input, IGpuBuffer eigenvalues, IGpuBuffer eigenvectors,
            int batchCount, int n);
    }
}
