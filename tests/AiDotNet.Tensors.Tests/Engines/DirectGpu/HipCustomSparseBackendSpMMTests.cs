using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines.DirectGpu.HIP;
using AiDotNet.Tensors.Helpers;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// The HIP mirror of <see cref="CudaSparseBackendSpMMTests"/> (#515): the managed CSR SpMM / SDDMM kernels that
/// SparseOps routes ahead of rocSPARSE on AMD, against a CPU reference with the kernels' per-output reduction order, so
/// an indexing or launch bug shows as O(1) error. Two such bugs shipped unseen because nothing ran these on hardware:
/// every SpMM launcher cast the byte-buffer CSR indices to HipGpuBuffer (InvalidCastException), and csr_spmm_warp
/// grouped 64 lanes per row while its launcher sized the grid for 32, leaving the second half of the rows unwritten.
/// </summary>
[Collection("DirectGpuSerial")]
public class HipCustomSparseBackendSpMMTests
{
    // Kernel pick: N%4==0 -> vec4, else N<=64 -> warp, else 2-D scalar. Enough rows that a launch covering only some of
    // them cannot pass.
    public static IEnumerable<object[]> Shapes() => new[]
    {
        new object[] { 64, 48, 16, 0.20 },    // vec4
        new object[] { 256, 256, 64, 0.05 },  // vec4
        new object[] { 100, 70, 1, 0.30 },    // warp (SpMV-like)
        new object[] { 300, 90, 30, 0.10 },   // warp
        new object[] { 200, 100, 130, 0.08 }, // 2-D scalar
    };

    [SkippableTheory]
    [MemberData(nameof(Shapes))]
    public void HipFloat_MatchesCpuReference(int rows, int cols, int n, double density)
    {
        Skip.IfNot(HipCustomSparseBackend.IsAvailable, "No HIP backend with the managed sparse kernels on this host.");
        var rng = RandomHelper.CreateSeededRandom(1234);
        var (rowPtr, colIdx) = BuildCsrPattern(rows, cols, density, rng, out int nnz);
        var vals = RandF(nnz, rng);
        var b = RandF(cols * n, rng);

        var want = new float[rows * n];
        for (int r = 0; r < rows; r++)
            for (int j = 0; j < n; j++)
            {
                float acc = 0f;
                for (int p = rowPtr[r]; p < rowPtr[r + 1]; p++) acc += vals[p] * b[colIdx[p] * n + j];
                want[r * n + j] = acc;
            }
        var got = HipCustomSparseBackend.SpMM(rowPtr, colIdx, vals, b, rows, cols, n);

        Assert.Equal(want.Length, got.Length);
        double rel = RelErr(got, want);
        Assert.True(rel < 1e-3, $"HIP float SpMM rel error {rel:E2} > 1e-3 at {rows}x{cols}x{n} (density {density}).");
    }

    [SkippableFact]
    public void HipFloat_IsDeterministic_AcrossRepeatedCalls()
    {
        Skip.IfNot(HipCustomSparseBackend.IsAvailable, "No HIP backend with the managed sparse kernels on this host.");
        var rng = RandomHelper.CreateSeededRandom(99);
        const int rows = 200, cols = 160, n = 48;
        var (rowPtr, colIdx) = BuildCsrPattern(rows, cols, 0.1, rng, out int nnz);
        var vals = RandF(nnz, rng);
        var b = RandF(cols * n, rng);

        Assert.Equal(
            HipCustomSparseBackend.SpMM(rowPtr, colIdx, vals, b, rows, cols, n),
            HipCustomSparseBackend.SpMM(rowPtr, colIdx, vals, b, rows, cols, n));
    }

    [SkippableTheory]
    [MemberData(nameof(Shapes))]
    public void HipDouble_MatchesCpuReference(int rows, int cols, int n, double density)
    {
        Skip.IfNot(HipCustomSparseBackend.IsAvailable, "No HIP backend with the managed sparse kernels on this host.");
        var rng = RandomHelper.CreateSeededRandom(4321);
        var (rowPtr, colIdx) = BuildCsrPattern(rows, cols, density, rng, out int nnz);
        var vals = new double[nnz];
        for (int i = 0; i < nnz; i++) vals[i] = rng.NextDouble() * 2 - 1;
        var b = new double[cols * n];
        for (int i = 0; i < b.Length; i++) b[i] = rng.NextDouble() * 2 - 1;

        var want = new double[rows * n];
        for (int r = 0; r < rows; r++)
            for (int j = 0; j < n; j++)
            {
                double acc = 0.0;
                for (int p = rowPtr[r]; p < rowPtr[r + 1]; p++) acc += vals[p] * b[colIdx[p] * n + j];
                want[r * n + j] = acc;
            }
        var got = HipCustomSparseBackend.SpMM(rowPtr, colIdx, vals, b, rows, cols, n);

        double maxAbs = 0, maxMag = 0;
        for (int i = 0; i < want.Length; i++) { maxAbs = Math.Max(maxAbs, Math.Abs(got[i] - want[i])); maxMag = Math.Max(maxMag, Math.Abs(want[i])); }
        double rel = maxAbs / Math.Max(1e-9, maxMag);
        Assert.True(rel < 1e-9, $"HIP double SpMM rel error {rel:E2} > 1e-9 at {rows}x{cols}x{n} (density {density}).");
    }

    [SkippableFact]
    public void HipSddmm_MatchesCpuReference()
    {
        Skip.IfNot(HipCustomSparseBackend.IsAvailable, "No HIP backend with the managed sparse kernels on this host.");
        var rng = RandomHelper.CreateSeededRandom(7);
        const int rows = 150, cols = 120, innerK = 24;
        var (rowPtr, colIdx) = BuildCsrPattern(rows, cols, 0.1, rng, out int nnz);
        var rowIndices = new int[nnz];
        for (int r = 0; r < rows; r++)
            for (int p = rowPtr[r]; p < rowPtr[r + 1]; p++) rowIndices[p] = r;
        var x = RandF(rows * innerK, rng);
        var y = RandF(cols * innerK, rng);

        var got = HipCustomSparseBackend.SDDMM(rowIndices, colIdx, x, y, innerK);
        var want = new float[nnz];
        for (int p = 0; p < nnz; p++)
            for (int k = 0; k < innerK; k++) want[p] += x[rowIndices[p] * innerK + k] * y[colIdx[p] * innerK + k];
        Assert.True(RelErr(got, want) < 1e-4, $"HIP SDDMM rel error {RelErr(got, want):E2}.");
    }

    private static (int[] rowPtr, int[] colIdx) BuildCsrPattern(int rows, int cols, double density, Random rng, out int nnz)
    {
        var rowPtr = new int[rows + 1];
        var colList = new List<int>();
        for (int r = 0; r < rows; r++)
        {
            rowPtr[r] = colList.Count;
            for (int c = 0; c < cols; c++)
                if (rng.NextDouble() < density) colList.Add(c);
        }
        rowPtr[rows] = colList.Count;
        nnz = colList.Count;
        return (rowPtr, colList.ToArray());
    }

    private static float[] RandF(int len, Random rng)
    {
        var a = new float[len];
        for (int i = 0; i < len; i++) a[i] = (float)(rng.NextDouble() * 2 - 1);
        return a;
    }

    private static double RelErr(float[] got, float[] want)
    {
        double maxAbs = 0, maxMag = 0;
        for (int i = 0; i < want.Length; i++)
        {
            maxAbs = Math.Max(maxAbs, Math.Abs(got[i] - want[i]));
            maxMag = Math.Max(maxMag, Math.Abs(want[i]));
        }
        return maxAbs / Math.Max(1e-6, maxMag);
    }
}