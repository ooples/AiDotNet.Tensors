// Copyright (c) AiDotNet. All rights reserved.
// Metal GPU backend - Sparse, Comparison, and Statistics operations.

using static AiDotNet.Tensors.Engines.DirectGpu.Metal.MetalNativeBindings;

namespace AiDotNet.Tensors.Engines.DirectGpu.Metal;

public sealed partial class MetalBackend
{
    #region Sparse Operations (2:4 Structured Sparsity)

    /// <summary>
    /// Enforce 2:4 structured sparsity pattern.
    /// </summary>
    public void Enforce2x4Sparsity(IGpuBuffer denseInput, IGpuBuffer sparseValues, IGpuBuffer sparseIndices, int M, int K)
    {
        ThrowIfDisposed();
        if (M <= 0 || K <= 0) return;
        if (K % 4 != 0)
            throw new ArgumentException("The K dimension must be divisible by four.", nameof(K));
        int groups = checked(M * (K / 4));
        DispatchResidentMetal("enforce_2x4", groups,
            new[] { denseInput, sparseValues, sparseIndices }, (uint)M, (uint)K);
    }

    /// <summary>
    /// Decompress 2:4 sparse format to dense.
    /// </summary>
    public void Decompress2x4Sparse(IGpuBuffer sparseValues, IGpuBuffer sparseIndices, IGpuBuffer denseOutput, int M, int K)
    {
        ThrowIfDisposed();
        if (M <= 0 || K <= 0) return;
        if (K % 4 != 0)
            throw new ArgumentException("The K dimension must be divisible by four.", nameof(K));
        int count = checked(M * K);
        bool aliasesInput = ReferenceEquals(denseOutput, sparseValues) || ReferenceEquals(denseOutput, sparseIndices);
        using var temporary = aliasesInput ? AllocateBuffer(count) : null;
        IGpuBuffer target = temporary ?? denseOutput;
        DispatchResidentMetal("decompress_2x4", count,
            new[] { sparseValues, sparseIndices, target }, (uint)M, (uint)K);
        if (temporary is not null) Copy(temporary, denseOutput, count);
    }

    /// <summary>
    /// Sparse GEMM with 2:4 structured sparsity.
    /// </summary>
    public void SparseGemm(IGpuBuffer sparseAValues, IGpuBuffer sparseAIndices,
        IGpuBuffer B, IGpuBuffer C, int M, int N, int K, float alpha = 1.0f, float beta = 0.0f)
    {
        ThrowIfDisposed();

        // Decompress A and use regular GEMM
        var denseA = AllocateBuffer(M * K);
        Decompress2x4Sparse(sparseAValues, sparseAIndices, denseA, M, K);
        Gemm(denseA, B, C, M, N, K, alpha, beta);
        ((MetalGpuBuffer)denseA).Dispose();
    }

    /// <summary>
    /// Fused sparse GEMM with bias and ReLU.
    /// </summary>
    public IGpuBuffer SparseGemmBiasRelu(IGpuBuffer sparseAValues, IGpuBuffer sparseAIndices,
        IGpuBuffer B, IGpuBuffer bias, int M, int N, int K)
    {
        ThrowIfDisposed();

        // Decompress A and use regular fused operation
        var denseA = AllocateBuffer(M * K);
        Decompress2x4Sparse(sparseAValues, sparseAIndices, denseA, M, K);
        var result = GemmBiasRelu(denseA, B, bias, M, N, K);
        ((MetalGpuBuffer)denseA).Dispose();
        return result;
    }

    #endregion

    #region CSR Sparse Operations

    private const string SparseLibName = "Sparse";

    private MetalPipelineState GetSparsePipeline(string kernelName)
    {
        if (_sparseLibrary == IntPtr.Zero)
            throw new InvalidOperationException(
                "Metal Sparse compute library was not compiled — install a Metal-capable runtime " +
                "or route to a different GPU backend.");
        return GetPipeline(SparseLibName, _sparseLibrary, kernelName);
    }

    private void DispatchCsrSegmentedMetal(IGpuBuffer columns, IGpuBuffer rows, IGpuBuffer input,
        IGpuBuffer output, int rowCount, int inputRows, int features, uint operation, float epsilon)
    {
        int count = checked(rowCount * features);
        if (count <= 0) return;
        bool aliasesInput = ReferenceEquals(input, output);
        using var temporary = aliasesInput ? AllocateBuffer(count) : null;
        IGpuBuffer target = temporary ?? output;
        DispatchResidentMetal("csr_segmented", count, new[] { columns, rows, input, target },
            (uint)rowCount, (uint)inputRows, (uint)features, operation,
            unchecked((uint)SingleToInt32BitsCompat(epsilon)));
        if (temporary is not null) Copy(temporary, output, count);
    }

    /// <summary>
    /// CSR · dense → dense: <c>C[M,N] = A[M,K] · B[K,N]</c> with A supplied as CSR
    /// (values / colIndices / rowPointers). All five buffers are GPU-resident and
    /// the compute stays on-device — replaces the previous CPU download-compute-upload
    /// stub with a real MSL dispatch (<see cref="MetalSparseKernels.Source"/>).
    /// One thread per output element (row, col); row = gid / N, col = gid % N.
    /// </summary>
    public void CsrSpMM(IGpuBuffer csrValues, IGpuBuffer csrColIndices, IGpuBuffer csrRowPointers,
        IGpuBuffer denseB, IGpuBuffer output, int M, int K, int N, int nnz)
    {
        ThrowIfDisposed();
        if (M <= 0 || N <= 0) return;

        long total = (long)M * N;
        if (total > int.MaxValue)
            throw new ArgumentOutOfRangeException(nameof(M),
                $"CSR SpMM output too large for a single Metal dispatch (M*N = {total}).");

        var pipeline = GetSparsePipeline("sparse_csr_spmm");
        var (threadgroups, threadsPerGroup) = pipeline.Calculate1DDispatch((int)total);
        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        encoder.SetBuffer((MetalGpuBuffer)csrValues, 0);
        encoder.SetBuffer((MetalGpuBuffer)csrColIndices, 1);
        encoder.SetBuffer((MetalGpuBuffer)csrRowPointers, 2);
        encoder.SetBuffer((MetalGpuBuffer)denseB, 3);
        encoder.SetBuffer((MetalGpuBuffer)output, 4);
        encoder.SetBytes(M, 5);
        encoder.SetBytes(K, 6);
        encoder.SetBytes(N, 7);
        encoder.SetBytes(nnz, 8);
        encoder.DispatchThreadgroups(threadgroups, threadsPerGroup);
    }

    /// <summary>
    /// Empirically-picked crossover: below this innerK, the thread-per-nnz base kernel
    /// wins because each thread's serial dot-product fits in a few cycles and the
    /// collaborative kernel's 256-thread tree reduction is pure overhead. Above this,
    /// the reduction latency dominates and the collaborative kernel wins. 64 matches
    /// the typical attention head_dim boundary. Mirror of the Vulkan tier's threshold.
    /// </summary>
    private const int SddmmCollabInnerKThreshold = 64;

    /// <summary>
    /// SDDMM: <c>output[p] = Σ_k x[rowIndices[p], k] · y[colIndices[p], k]</c> for each
    /// pattern non-zero p ∈ [0, nnz). All five buffers are GPU-resident.
    ///
    /// <para><b>Adaptive dispatch (beyond-industry-standard):</b> for large <c>innerK</c>
    /// (typical attention head_dim ≥ 64), routes to the collaborative kernel
    /// <c>sparse_sddmm_collab</c> — one threadgroup per non-zero, 256 threads collaborate
    /// on the innerK reduction via threadgroup memory + tree reduction. Reduces per-nnz
    /// latency from O(innerK) to O(log₂ 256) + O(innerK/256). MPS's sparse SDDMM (when
    /// available) uses fixed thread-per-nnz for all shapes.</para>
    /// </summary>
    public void CsrSddmm(IGpuBuffer rowIndices, IGpuBuffer colIndices,
        IGpuBuffer x, IGpuBuffer y, IGpuBuffer output,
        int nnz, int innerK)
    {
        ThrowIfDisposed();
        if (nnz <= 0) return;

        if (innerK >= SddmmCollabInnerKThreshold)
        {
            var pipeline = GetSparsePipeline("sparse_sddmm_collab");
            var threadgroups = new MTLSize((ulong)nnz, 1, 1);
            var threadsPerGroup = new MTLSize(
                (ulong)MetalSparseKernels.SddmmCollabThreadgroupSize, 1, 1);
            using var encoder = _commandQueue.CreateScopedComputeEncoder();
            encoder.SetPipelineState(pipeline.Handle);
            encoder.SetBuffer((MetalGpuBuffer)rowIndices, 0);
            encoder.SetBuffer((MetalGpuBuffer)colIndices, 1);
            encoder.SetBuffer((MetalGpuBuffer)x, 2);
            encoder.SetBuffer((MetalGpuBuffer)y, 3);
            encoder.SetBuffer((MetalGpuBuffer)output, 4);
            encoder.SetBytes(nnz, 5);
            encoder.SetBytes(innerK, 6);
            // Threadgroup memory: 256 floats for the shared reduction buffer at binding 0.
            encoder.SetThreadgroupMemoryLength(
                (uint)(MetalSparseKernels.SddmmCollabThreadgroupSize * sizeof(float)), 0);
            encoder.DispatchThreadgroups(threadgroups, threadsPerGroup);
        }
        else
        {
            var pipeline = GetSparsePipeline("sparse_sddmm");
            var (threadgroups, threadsPerGroup) = pipeline.Calculate1DDispatch(nnz);
            using var encoder = _commandQueue.CreateScopedComputeEncoder();
            encoder.SetPipelineState(pipeline.Handle);
            encoder.SetBuffer((MetalGpuBuffer)rowIndices, 0);
            encoder.SetBuffer((MetalGpuBuffer)colIndices, 1);
            encoder.SetBuffer((MetalGpuBuffer)x, 2);
            encoder.SetBuffer((MetalGpuBuffer)y, 3);
            encoder.SetBuffer((MetalGpuBuffer)output, 4);
            encoder.SetBytes(nnz, 5);
            encoder.SetBytes(innerK, 6);
            encoder.DispatchThreadgroups(threadgroups, threadsPerGroup);
        }
    }

    /// <summary>
    /// Fused CSR SpMM with bias.
    /// </summary>
    public void CsrSpMMBias(IGpuBuffer csrValues, IGpuBuffer csrColIndices, IGpuBuffer csrRowPointers,
        IGpuBuffer denseB, IGpuBuffer bias, IGpuBuffer output, int M, int K, int N, int nnz)
    {
        ThrowIfDisposed();
        CsrSpMM(csrValues, csrColIndices, csrRowPointers, denseB, output, M, K, N, nnz);
        BiasAdd(output, bias, output, M, N);
    }

    /// <summary>
    /// Scatter-add for graph edges.
    /// </summary>
    public void ScatterAddEdges(IGpuBuffer input, IGpuBuffer sourceIndices, IGpuBuffer targetIndices,
        IGpuBuffer? edgeValues, IGpuBuffer output, int numNodes, int numEdges, int features)
    {
        ThrowIfDisposed();
        int count = checked(numNodes * features);
        if (count <= 0) return;
        bool aliasesInput = ReferenceEquals(input, output);
        using var inputCopy = aliasesInput ? AllocateBuffer(count) : null;
        IGpuBuffer effectiveInput = input;
        if (inputCopy is not null)
        {
            Copy(input, inputCopy, count);
            effectiveInput = inputCopy;
        }
        using var unusedEdgeValues = edgeValues is null ? AllocateBuffer(1) : null;
        if (unusedEdgeValues is not null) Fill(unusedEdgeValues, 1f, 1);
        IGpuBuffer effectiveEdgeValues = edgeValues ?? unusedEdgeValues!;
        DispatchResidentMetal("scatter_add_edges_deterministic", count,
            new[] { effectiveInput, sourceIndices, targetIndices, effectiveEdgeValues, output },
            (uint)numNodes, (uint)numEdges, (uint)features, edgeValues is null ? 0u : 1u);
    }

    /// <summary>
    /// CSR segmented max aggregation.
    /// </summary>
    public void CsrSegmentedMax(IGpuBuffer csrColIndices, IGpuBuffer csrRowPointers,
        IGpuBuffer input, IGpuBuffer output, int M, int K, int N)
    {
        ThrowIfDisposed();
        DispatchCsrSegmentedMetal(csrColIndices, csrRowPointers, input, output, M, K, N, 0u, 0f);
    }

    /// <summary>
    /// CSR segmented min aggregation.
    /// </summary>
    public void CsrSegmentedMin(IGpuBuffer csrColIndices, IGpuBuffer csrRowPointers,
        IGpuBuffer input, IGpuBuffer output, int M, int K, int N)
    {
        ThrowIfDisposed();
        DispatchCsrSegmentedMetal(csrColIndices, csrRowPointers, input, output, M, K, N, 1u, 0f);
    }

    /// <summary>
    /// CSR segmented standard deviation.
    /// </summary>
    public void CsrSegmentedStdDev(IGpuBuffer csrColIndices, IGpuBuffer csrRowPointers,
        IGpuBuffer input, IGpuBuffer output, int M, int K, int N, float epsilon = 1e-8f)
    {
        ThrowIfDisposed();
        DispatchCsrSegmentedMetal(csrColIndices, csrRowPointers, input, output, M, K, N, 2u, epsilon);
    }

    #endregion

    #region Comparison Operations

    /// <summary>
    /// Element-wise greater than comparison.
    /// </summary>
    public void GreaterThan(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int size)
    {
        ThrowIfDisposed();
        DispatchResidentMetal("comparison_binary", size, new[] { A, B, C }, (uint)size, 0u);
    }

    /// <summary>
    /// Element-wise less than comparison.
    /// </summary>
    public void LessThan(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int size)
    {
        ThrowIfDisposed();
        DispatchResidentMetal("comparison_binary", size, new[] { A, B, C }, (uint)size, 1u);
    }

    /// <summary>
    /// Element-wise equality comparison.
    /// </summary>
    public void Equal(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int size)
    {
        ThrowIfDisposed();
        DispatchResidentMetal("comparison_binary", size, new[] { A, B, C }, (uint)size, 2u);
    }

    /// <summary>
    /// Conditional selection: C = condition ? A : B
    /// </summary>
    public void Where(IGpuBuffer condition, IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int size)
    {
        ThrowIfDisposed();
        DispatchResidentMetal("comparison_where", size, new[] { condition, A, B, C }, (uint)size);
    }

    /// <summary>
    /// Not equal to scalar comparison.
    /// </summary>
    public void NotEqualScalar(IGpuBuffer A, IGpuBuffer C, float scalar, int size)
    {
        ThrowIfDisposed();
        DispatchResidentMetal("comparison_not_equal_scalar", size, new[] { A, C },
            (uint)size, unchecked((uint)SingleToInt32BitsCompat(scalar)));
    }

    #endregion

    #region Statistics Operations

    /// <summary>
    /// Variance along axis.
    /// </summary>
    public void VarAxis(IGpuBuffer A, IGpuBuffer mean, IGpuBuffer variance, int outerSize, int reduceSize)
    {
        ThrowIfDisposed();
        if (reduceSize <= 0) throw new ArgumentOutOfRangeException(nameof(reduceSize));
        DispatchResidentMetal("variance_axis", outerSize, new[] { A, mean, variance },
            (uint)outerSize, (uint)reduceSize);
    }

    /// <summary>
    /// ArgMax along axis.
    /// </summary>
    public void ArgMax(IGpuBuffer A, IGpuBuffer indices, int outerSize, int reduceSize)
    {
        ThrowIfDisposed();
        if (reduceSize <= 0) throw new ArgumentOutOfRangeException(nameof(reduceSize));
        DispatchResidentMetal("arg_extrema_axis", outerSize, new[] { A, indices },
            (uint)outerSize, (uint)reduceSize, 1u);
    }

    /// <summary>
    /// ArgMin along axis.
    /// </summary>
    public void ArgMin(IGpuBuffer A, IGpuBuffer indices, int outerSize, int reduceSize)
    {
        ThrowIfDisposed();
        if (reduceSize <= 0) throw new ArgumentOutOfRangeException(nameof(reduceSize));
        DispatchResidentMetal("arg_extrema_axis", outerSize, new[] { A, indices },
            (uint)outerSize, (uint)reduceSize, 0u);
    }

    /// <summary>
    /// Top-K selection.
    /// </summary>
    public void TopK(IGpuBuffer A, IGpuBuffer values, IGpuBuffer indices, int outerSize, int reduceSize, int k, bool sorted = true)
    {
        ThrowIfDisposed();
        if (outerSize <= 0 || reduceSize <= 0 || k <= 0) return;
        if (k > reduceSize) throw new ArgumentOutOfRangeException(nameof(k), "k cannot exceed reduceSize.");
        DispatchResidentMetal("topk_axis_serial", outerSize, new[] { A, values, indices },
            (uint)outerSize, (uint)reduceSize, (uint)k);
    }

    #endregion
}
