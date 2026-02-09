// Copyright (c) AiDotNet. All rights reserved.
// Metal GPU backend - Sparse, Comparison, and Statistics operations.

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

    /// <summary>
    /// CSR sparse matrix-dense matrix multiplication.
    /// </summary>
    public void CsrSpMM(IGpuBuffer csrValues, IGpuBuffer csrColIndices, IGpuBuffer csrRowPointers,
        IGpuBuffer denseB, IGpuBuffer output, int M, int K, int N, int nnz)
    {
        ThrowIfDisposed();

        var values = DownloadBuffer(csrValues);
        var colIndices = DownloadIntBuffer(csrColIndices, nnz);
        var rowPointers = DownloadIntBuffer(csrRowPointers, M + 1);
        var denseData = DownloadBuffer(denseB);

        var outputData = new float[M * N];

        for (int row = 0; row < M; row++)
        {
            int rowStart = rowPointers[row];
            int rowEnd = rowPointers[row + 1];

            for (int col = 0; col < N; col++)
            {
                float sum = 0;
                for (int i = rowStart; i < rowEnd; i++)
                {
                    int k = colIndices[i];
                    sum += values[i] * denseData[k * N + col];
                }
                outputData[row * N + col] = sum;
            }
        }

        UploadToBuffer(output, outputData);
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
