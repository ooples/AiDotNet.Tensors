// Copyright (c) CrowdTrainer LLC. All rights reserved.
namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>
/// <see cref="IFusedIndexKernels"/> for backends without the fused CUDA kernels, staged through the host with the
/// kernels' exact semantics: an index is round-half-to-even of the float, and NaN or an index outside the range reads
/// (or writes) nothing. The capability flags stay false on these backends, so the engine composes these results from
/// device ops instead; these are the correct reference forms behind the contract.
/// </summary>
internal static class PortableFusedIndexKernels
{
    private static int Index(float value, int bound)
    {
        if (float.IsNaN(value)) return -1;
        double rounded = Math.Round(value, MidpointRounding.ToEven);
        return rounded >= 0 && rounded < bound ? (int)rounded : -1;
    }

    public static void GatherClassValues(IDirectGpuBackend backend, IGpuBuffer values, IGpuBuffer classIndices, IGpuBuffer output,
        int rows, int numClasses)
    {
        if (rows <= 0) return;
        float[] v = backend.DownloadBuffer(values), cls = backend.DownloadBuffer(classIndices);
        var result = new float[rows];
        for (int r = 0; r < rows; r++)
        {
            int c = Index(cls[r], numClasses);
            result[r] = c < 0 ? 0f : v[(long)r * numClasses + c];
        }
        backend.UploadBufferInPlace(result, output);
    }

    public static void ScatterClassGrad(IDirectGpuBackend backend, IGpuBuffer gradOutput, IGpuBuffer classIndices, IGpuBuffer grad,
        int rows, int numClasses)
    {
        if (rows <= 0 || numClasses <= 0) return;
        float[] g = backend.DownloadBuffer(gradOutput), cls = backend.DownloadBuffer(classIndices);
        var result = new float[checked(rows * numClasses)];
        for (int r = 0; r < rows; r++)
        {
            int c = Index(cls[r], numClasses);
            if (c >= 0) result[(long)r * numClasses + c] = g[r];
        }
        backend.UploadBufferInPlace(result, grad);
    }

    public static void EmbeddingFromFloatIds(IDirectGpuBackend backend, IGpuBuffer ids, IGpuBuffer table, IGpuBuffer output,
        int numIndices, int embeddingDim, int vocabSize)
    {
        if (numIndices <= 0 || embeddingDim <= 0) return;
        float[] id = backend.DownloadBuffer(ids), t = backend.DownloadBuffer(table);
        var result = new float[checked(numIndices * embeddingDim)];
        for (int i = 0; i < numIndices; i++)
        {
            int row = Index(id[i], vocabSize);
            if (row >= 0) Array.Copy(t, (long)row * embeddingDim, result, (long)i * embeddingDim, embeddingDim);
        }
        backend.UploadBufferInPlace(result, output);
    }

    // gradTable[id] += gradOutput[i] in ascending i. That is also the deterministic kernel's fixed order, so one routine
    // serves both (the caller zeroes gradTable first; rows of ids that never occur keep their contents).
    public static void EmbeddingBackwardFromFloatIds(IDirectGpuBackend backend, IGpuBuffer gradOutput, IGpuBuffer ids,
        IGpuBuffer gradTable, int numIndices, int embeddingDim, int vocabSize)
    {
        if (numIndices <= 0 || embeddingDim <= 0 || vocabSize <= 0) return;
        float[] g = backend.DownloadBuffer(gradOutput), id = backend.DownloadBuffer(ids);
        float[] table = backend.DownloadBuffer(gradTable);
        int tableCount = checked(vocabSize * embeddingDim);
        if (table.Length > tableCount) Array.Resize(ref table, tableCount);
        for (int i = 0; i < numIndices; i++)
        {
            int row = Index(id[i], vocabSize);
            if (row < 0) continue;
            long dst = (long)row * embeddingDim, src = (long)i * embeddingDim;
            for (int e = 0; e < embeddingDim; e++) table[dst + e] += g[src + e];
        }
        backend.UploadBufferInPlace(table, gradTable);
    }
}