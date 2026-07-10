// Copyright (c) AiDotNet. All rights reserved.

using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu.OpenCL;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Gpu;

/// <summary>
/// Verifies the OpenCL (AMD/Intel) CSR·dense SpMM path wired into SparseOps. Runs the managed
/// csr_spmm kernel on a real OpenCL device and checks it against a CPU dense reference. Skips
/// (no-op) when no OpenCL device is present, so it is a live GPU check on OpenCL hosts and inert
/// elsewhere.
/// </summary>
public class OpenClSparseSpMMTests
{
    private static bool GpuPresent => OpenClSparseBackend.IsAvailable;

    // Dense reference C = A·B computed on the CPU from the same CSR + dense inputs.
    private static float[] DenseReference(int[] rowPtr, int[] colIdx, float[] values, float[] b, int rows, int cols, int n)
    {
        var c = new float[rows * n];
        for (int i = 0; i < rows; i++)
            for (int p = rowPtr[i]; p < rowPtr[i + 1]; p++)
            {
                int j = colIdx[p];
                float v = values[p];
                for (int col = 0; col < n; col++)
                    c[i * n + col] += v * b[j * n + col];
            }
        return c;
    }

    [Fact]
    public void OpenClSpMM_SmallMatrix_MatchesDenseReference()
    {
        if (!GpuPresent) return; // no OpenCL device on this host — inert.

        // A = [[1,0,2,0],[0,3,0,4],[5,0,6,0],[0,7,0,8]]  (CSR)
        int[] rowPtr = { 0, 2, 4, 6, 8 };
        int[] colIdx = { 0, 2, 1, 3, 0, 2, 1, 3 };
        float[] values = { 1, 2, 3, 4, 5, 6, 7, 8 };
        int rows = 4, cols = 4, n = 2;
        float[] b = { 1, 2, 3, 4, 5, 6, 7, 8 }; // 4x2 row-major

        var gpu = OpenClSparseBackend.SpMM(rowPtr, colIdx, values, b, rows, cols, n);
        var expected = DenseReference(rowPtr, colIdx, values, b, rows, cols, n);

        Assert.Equal(expected.Length, gpu.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.Equal(expected[i], gpu[i], 3);
    }

    [Fact]
    public void OpenClSpMM_LargeMatrix_MatchesDenseReference_AndExercisesGpuDispatch()
    {
        if (!GpuPresent) return;

        // 512x512 sparse A (~4 nnz/row, deterministic pattern) · 512x512 dense B.
        // rows*n = 262144 == GpuDispatchThreshold, so SparseOps.SparseMatMul routes to the GPU tier.
        const int rows = 512, cols = 512, n = 512;
        const int perRow = 4;
        int nnz = rows * perRow;
        var rowPtr = new int[rows + 1];
        var colIdx = new int[nnz];
        var values = new float[nnz];
        for (int i = 0; i < rows; i++)
        {
            rowPtr[i] = i * perRow;
            for (int t = 0; t < perRow; t++)
            {
                int p = i * perRow + t;
                colIdx[p] = (i * 7 + t * 131) % cols; // deterministic spread
                values[p] = 1f + ((p * 13) % 5);
            }
        }
        rowPtr[rows] = nnz;

        var b = new float[cols * n];
        for (int i = 0; i < b.Length; i++)
            b[i] = ((i * 3) % 7) - 3; // small deterministic values, some negative

        var gpu = OpenClSparseBackend.SpMM(rowPtr, colIdx, values, b, rows, cols, n);
        var expected = DenseReference(rowPtr, colIdx, values, b, rows, cols, n);

        Assert.Equal(expected.Length, gpu.Length);
        // Column indices can repeat within a row (allowed in CSR); the GPU kernel and the CPU
        // reference both accumulate, so results match up to float rounding.
        for (int i = 0; i < expected.Length; i++)
            Assert.Equal(expected[i], gpu[i], 2);
    }
}
