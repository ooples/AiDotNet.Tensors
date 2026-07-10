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

        var denseData = DownloadBuffer(denseInput);
        var sparseValuesData = new float[M * K / 2];
        var sparseIndicesData = new byte[M * K / 4];

        for (int row = 0; row < M; row++)
        {
            for (int group = 0; group < K / 4; group++)
            {
                int baseIdx = row * K + group * 4;

                // Find 2 largest magnitude elements in group of 4
                var indices = new int[4];
                var values = new float[4];
                for (int i = 0; i < 4; i++)
                {
                    indices[i] = i;
                    values[i] = MathF.Abs(denseData[baseIdx + i]);
                }

                // Sort by magnitude descending
                for (int i = 0; i < 3; i++)
                {
                    for (int j = i + 1; j < 4; j++)
                    {
                        if (values[j] > values[i])
                        {
                            (values[i], values[j]) = (values[j], values[i]);
                            (indices[i], indices[j]) = (indices[j], indices[i]);
                        }
                    }
                }

                // Store the top 2 values
                int sparseValueOffset = row * K / 2 + group * 2;
                sparseValuesData[sparseValueOffset] = denseData[baseIdx + indices[0]];
                sparseValuesData[sparseValueOffset + 1] = denseData[baseIdx + indices[1]];

                // Pack indices into byte (2 bits per index)
                byte packedIndices = (byte)((indices[0] & 0x3) | ((indices[1] & 0x3) << 2));
                sparseIndicesData[row * K / 4 + group] = packedIndices;
            }
        }

        UploadToBuffer(sparseValues, sparseValuesData);

        // Convert byte indices to float for upload
        var floatIndices = new float[(M * K / 4 + 3) / 4];
        Buffer.BlockCopy(sparseIndicesData, 0, floatIndices, 0, sparseIndicesData.Length);
        UploadToBuffer(sparseIndices, floatIndices);
    }

    /// <summary>
    /// Decompress 2:4 sparse format to dense.
    /// </summary>
    public void Decompress2x4Sparse(IGpuBuffer sparseValues, IGpuBuffer sparseIndices, IGpuBuffer denseOutput, int M, int K)
    {
        ThrowIfDisposed();

        var sparseValuesData = DownloadBuffer(sparseValues);
        var floatIndices = DownloadBuffer(sparseIndices);
        var sparseIndicesData = new byte[M * K / 4];
        Buffer.BlockCopy(floatIndices, 0, sparseIndicesData, 0, sparseIndicesData.Length);

        var denseData = new float[M * K];

        for (int row = 0; row < M; row++)
        {
            for (int group = 0; group < K / 4; group++)
            {
                int sparseValueOffset = row * K / 2 + group * 2;
                byte packedIndices = sparseIndicesData[row * K / 4 + group];

                int idx0 = packedIndices & 0x3;
                int idx1 = (packedIndices >> 2) & 0x3;

                int baseIdx = row * K + group * 4;
                denseData[baseIdx + idx0] = sparseValuesData[sparseValueOffset];
                denseData[baseIdx + idx1] = sparseValuesData[sparseValueOffset + 1];
            }
        }

        UploadToBuffer(denseOutput, denseData);
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

        // Add bias
        var outputData = DownloadBuffer(output);
        var biasData = DownloadBuffer(bias);

        for (int row = 0; row < M; row++)
        {
            for (int col = 0; col < N; col++)
            {
                outputData[row * N + col] += biasData[col];
            }
        }

        UploadToBuffer(output, outputData);
    }

    /// <summary>
    /// Scatter-add for graph edges.
    /// </summary>
    public void ScatterAddEdges(IGpuBuffer input, IGpuBuffer sourceIndices, IGpuBuffer targetIndices,
        IGpuBuffer? edgeValues, IGpuBuffer output, int numNodes, int numEdges, int features)
    {
        ThrowIfDisposed();

        var inputData = DownloadBuffer(input);
        var srcIdx = DownloadIntBuffer(sourceIndices, numEdges);
        var tgtIdx = DownloadIntBuffer(targetIndices, numEdges);
        float[]? edgeData = edgeValues is not null ? DownloadBuffer(edgeValues) : null;

        var outputData = DownloadBuffer(output);

        for (int e = 0; e < numEdges; e++)
        {
            int src = srcIdx[e];
            int tgt = tgtIdx[e];
            float weight = edgeData?[e] ?? 1.0f;

            for (int f = 0; f < features; f++)
            {
                outputData[tgt * features + f] += weight * inputData[src * features + f];
            }
        }

        UploadToBuffer(output, outputData);
    }

    /// <summary>
    /// CSR segmented max aggregation.
    /// </summary>
    public void CsrSegmentedMax(IGpuBuffer csrColIndices, IGpuBuffer csrRowPointers,
        IGpuBuffer input, IGpuBuffer output, int M, int K, int N)
    {
        ThrowIfDisposed();

        var colIndices = DownloadIntBuffer(csrColIndices, -1);
        var rowPointers = DownloadIntBuffer(csrRowPointers, M + 1);
        var inputData = DownloadBuffer(input);

        var outputData = new float[M * N];
        for (int i = 0; i < outputData.Length; i++) outputData[i] = float.NegativeInfinity;

        for (int row = 0; row < M; row++)
        {
            int rowStart = rowPointers[row];
            int rowEnd = rowPointers[row + 1];

            for (int i = rowStart; i < rowEnd; i++)
            {
                int k = colIndices[i];
                for (int f = 0; f < N; f++)
                {
                    outputData[row * N + f] = MathF.Max(outputData[row * N + f], inputData[k * N + f]);
                }
            }

            // Handle empty rows
            if (rowStart == rowEnd)
            {
                for (int f = 0; f < N; f++)
                {
                    outputData[row * N + f] = 0;
                }
            }
        }

        UploadToBuffer(output, outputData);
    }

    /// <summary>
    /// CSR segmented min aggregation.
    /// </summary>
    public void CsrSegmentedMin(IGpuBuffer csrColIndices, IGpuBuffer csrRowPointers,
        IGpuBuffer input, IGpuBuffer output, int M, int K, int N)
    {
        ThrowIfDisposed();

        var colIndices = DownloadIntBuffer(csrColIndices, -1);
        var rowPointers = DownloadIntBuffer(csrRowPointers, M + 1);
        var inputData = DownloadBuffer(input);

        var outputData = new float[M * N];
        for (int i = 0; i < outputData.Length; i++) outputData[i] = float.PositiveInfinity;

        for (int row = 0; row < M; row++)
        {
            int rowStart = rowPointers[row];
            int rowEnd = rowPointers[row + 1];

            for (int i = rowStart; i < rowEnd; i++)
            {
                int k = colIndices[i];
                for (int f = 0; f < N; f++)
                {
                    outputData[row * N + f] = MathF.Min(outputData[row * N + f], inputData[k * N + f]);
                }
            }

            // Handle empty rows
            if (rowStart == rowEnd)
            {
                for (int f = 0; f < N; f++)
                {
                    outputData[row * N + f] = 0;
                }
            }
        }

        UploadToBuffer(output, outputData);
    }

    /// <summary>
    /// CSR segmented standard deviation.
    /// </summary>
    public void CsrSegmentedStdDev(IGpuBuffer csrColIndices, IGpuBuffer csrRowPointers,
        IGpuBuffer input, IGpuBuffer output, int M, int K, int N, float epsilon = 1e-8f)
    {
        ThrowIfDisposed();

        var colIndices = DownloadIntBuffer(csrColIndices, -1);
        var rowPointers = DownloadIntBuffer(csrRowPointers, M + 1);
        var inputData = DownloadBuffer(input);

        var outputData = new float[M * N];

        for (int row = 0; row < M; row++)
        {
            int rowStart = rowPointers[row];
            int rowEnd = rowPointers[row + 1];
            int count = rowEnd - rowStart;

            if (count == 0)
            {
                continue;
            }

            // Compute mean
            var mean = new float[N];
            for (int i = rowStart; i < rowEnd; i++)
            {
                int k = colIndices[i];
                for (int f = 0; f < N; f++)
                {
                    mean[f] += inputData[k * N + f];
                }
            }
            for (int f = 0; f < N; f++)
            {
                mean[f] /= count;
            }

            // Compute variance
            var variance = new float[N];
            for (int i = rowStart; i < rowEnd; i++)
            {
                int k = colIndices[i];
                for (int f = 0; f < N; f++)
                {
                    float diff = inputData[k * N + f] - mean[f];
                    variance[f] += diff * diff;
                }
            }
            for (int f = 0; f < N; f++)
            {
                variance[f] /= count;
                outputData[row * N + f] = MathF.Sqrt(variance[f] + epsilon);
            }
        }

        UploadToBuffer(output, outputData);
    }

    #endregion

    #region Comparison Operations

    /// <summary>
    /// Element-wise greater than comparison.
    /// </summary>
    public void GreaterThan(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int size)
    {
        ThrowIfDisposed();

        var aData = DownloadBuffer(A);
        var bData = DownloadBuffer(B);
        var cData = new float[size];

        for (int i = 0; i < size; i++)
        {
            cData[i] = aData[i] > bData[i] ? 1.0f : 0.0f;
        }

        UploadToBuffer(C, cData);
    }

    /// <summary>
    /// Element-wise less than comparison.
    /// </summary>
    public void LessThan(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int size)
    {
        ThrowIfDisposed();

        var aData = DownloadBuffer(A);
        var bData = DownloadBuffer(B);
        var cData = new float[size];

        for (int i = 0; i < size; i++)
        {
            cData[i] = aData[i] < bData[i] ? 1.0f : 0.0f;
        }

        UploadToBuffer(C, cData);
    }

    /// <summary>
    /// Element-wise equality comparison.
    /// </summary>
    public void Equal(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int size)
    {
        ThrowIfDisposed();

        var aData = DownloadBuffer(A);
        var bData = DownloadBuffer(B);
        var cData = new float[size];

        for (int i = 0; i < size; i++)
        {
            cData[i] = MathF.Abs(aData[i] - bData[i]) < 1e-7f ? 1.0f : 0.0f;
        }

        UploadToBuffer(C, cData);
    }

    /// <summary>
    /// Conditional selection: C = condition ? A : B
    /// </summary>
    public void Where(IGpuBuffer condition, IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int size)
    {
        ThrowIfDisposed();

        var condData = DownloadBuffer(condition);
        var aData = DownloadBuffer(A);
        var bData = DownloadBuffer(B);
        var cData = new float[size];

        for (int i = 0; i < size; i++)
        {
            cData[i] = condData[i] != 0 ? aData[i] : bData[i];
        }

        UploadToBuffer(C, cData);
    }

    /// <summary>
    /// Not equal to scalar comparison.
    /// </summary>
    public void NotEqualScalar(IGpuBuffer A, IGpuBuffer C, float scalar, int size)
    {
        ThrowIfDisposed();

        var aData = DownloadBuffer(A);
        var cData = new float[size];

        for (int i = 0; i < size; i++)
        {
            cData[i] = MathF.Abs(aData[i] - scalar) >= 1e-7f ? 1.0f : 0.0f;
        }

        UploadToBuffer(C, cData);
    }

    #endregion

    #region Statistics Operations

    /// <summary>
    /// Variance along axis.
    /// </summary>
    public void VarAxis(IGpuBuffer A, IGpuBuffer mean, IGpuBuffer variance, int outerSize, int reduceSize)
    {
        ThrowIfDisposed();

        var aData = DownloadBuffer(A);
        var meanData = new float[outerSize];
        var varData = new float[outerSize];

        for (int i = 0; i < outerSize; i++)
        {
            // Compute mean
            float sum = 0;
            for (int j = 0; j < reduceSize; j++)
            {
                sum += aData[i * reduceSize + j];
            }
            meanData[i] = sum / reduceSize;

            // Compute variance
            float varSum = 0;
            for (int j = 0; j < reduceSize; j++)
            {
                float diff = aData[i * reduceSize + j] - meanData[i];
                varSum += diff * diff;
            }
            varData[i] = varSum / reduceSize;
        }

        UploadToBuffer(mean, meanData);
        UploadToBuffer(variance, varData);
    }

    /// <summary>
    /// ArgMax along axis.
    /// </summary>
    public void ArgMax(IGpuBuffer A, IGpuBuffer indices, int outerSize, int reduceSize)
    {
        ThrowIfDisposed();

        var aData = DownloadBuffer(A);
        var indicesData = new float[outerSize];

        for (int i = 0; i < outerSize; i++)
        {
            float maxVal = float.NegativeInfinity;
            int maxIdx = 0;
            for (int j = 0; j < reduceSize; j++)
            {
                if (aData[i * reduceSize + j] > maxVal)
                {
                    maxVal = aData[i * reduceSize + j];
                    maxIdx = j;
                }
            }
            indicesData[i] = maxIdx;
        }

        UploadToBuffer(indices, indicesData);
    }

    /// <summary>
    /// ArgMin along axis.
    /// </summary>
    public void ArgMin(IGpuBuffer A, IGpuBuffer indices, int outerSize, int reduceSize)
    {
        ThrowIfDisposed();

        var aData = DownloadBuffer(A);
        var indicesData = new float[outerSize];

        for (int i = 0; i < outerSize; i++)
        {
            float minVal = float.PositiveInfinity;
            int minIdx = 0;
            for (int j = 0; j < reduceSize; j++)
            {
                if (aData[i * reduceSize + j] < minVal)
                {
                    minVal = aData[i * reduceSize + j];
                    minIdx = j;
                }
            }
            indicesData[i] = minIdx;
        }

        UploadToBuffer(indices, indicesData);
    }

    /// <summary>
    /// Top-K selection.
    /// </summary>
    public void TopK(IGpuBuffer A, IGpuBuffer values, IGpuBuffer indices, int outerSize, int reduceSize, int k, bool sorted = true)
    {
        ThrowIfDisposed();

        var aData = DownloadBuffer(A);
        var valuesData = new float[outerSize * k];
        var indicesData = new float[outerSize * k];

        for (int i = 0; i < outerSize; i++)
        {
            // Extract row
            var rowValues = new (float value, int index)[reduceSize];
            for (int j = 0; j < reduceSize; j++)
            {
                rowValues[j] = (aData[i * reduceSize + j], j);
            }

            // Partial sort to get top K
            Array.Sort(rowValues, (a, b) => b.value.CompareTo(a.value));

            // Copy top K
            for (int j = 0; j < k; j++)
            {
                valuesData[i * k + j] = rowValues[j].value;
                indicesData[i * k + j] = rowValues[j].index;
            }
        }

        UploadToBuffer(values, valuesData);
        UploadToBuffer(indices, indicesData);
    }

    #endregion
}
