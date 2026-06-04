using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.Tensors.Helpers;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// Parity + determinism guard for the managed custom-kernel CSR SpMM path that
/// #515 (P6) makes the default GPU SpMM (replacing native cuSPARSE). The
/// <c>csr_spmm</c> CUDA kernel accumulates each output element in a single thread
/// over the row's stored order, so it must (a) match a hand-written CPU CSR
/// reference to FMA tolerance and (b) be bit-identical across repeated calls.
///
/// <para>These run only where the CUDA driver is present (the self-hosted GPU
/// runner) — on a CPU-only host they skip. The CPU reference is the oracle; it
/// uses the same per-output sequential reduction order as the kernel, so any
/// indexing/launch bug shows up as O(1) error, not rounding.</para>
/// </summary>
public class CudaSparseBackendSpMMTests
{
    // CPU CSR · dense reference. Same reduction order as csr_spmm (ascending nnz
    // per row, accumulate into one scalar), so a correct kernel matches to ~1e-3
    // relative (the GPU may FMA-contract val*B+sum; the CPU does separate mul+add).
    private static float[] CpuReference(
        int[] rowPtr, int[] colIdx, float[] vals, float[] b, int rows, int n)
    {
        var outp = new float[rows * n];
        for (int r = 0; r < rows; r++)
            for (int j = 0; j < n; j++)
            {
                float acc = 0f;
                for (int p = rowPtr[r]; p < rowPtr[r + 1]; p++)
                    acc += vals[p] * b[colIdx[p] * n + j];
                outp[r * n + j] = acc;
            }
        return outp;
    }

    // Random CSR with ascending column indices per row (the layout csr_spmm expects).
    private static (int[] rowPtr, int[] colIdx, float[] vals) BuildCsr(
        int rows, int cols, double density, Random rng)
    {
        var rowPtr = new int[rows + 1];
        var colList = new List<int>();
        var valList = new List<float>();
        for (int r = 0; r < rows; r++)
        {
            rowPtr[r] = colList.Count;
            for (int c = 0; c < cols; c++)
                if (rng.NextDouble() < density)
                {
                    colList.Add(c);
                    valList.Add((float)(rng.NextDouble() * 2 - 1));
                }
        }
        rowPtr[rows] = colList.Count;
        return (rowPtr, colList.ToArray(), valList.ToArray());
    }

    private static float[] RandDense(int len, Random rng)
    {
        var b = new float[len];
        for (int i = 0; i < len; i++) b[i] = (float)(rng.NextDouble() * 2 - 1);
        return b;
    }

    [SkippableTheory]
    [InlineData(64, 48, 16, 0.20)]    // small N — exercises the warp-relevant regime
    [InlineData(128, 96, 128, 0.10)]  // moderate, square-ish dense
    [InlineData(256, 256, 64, 0.05)]  // larger + sparser
    [InlineData(100, 70, 1, 0.30)]    // N=1 (SpMV-like) edge
    public void CustomCsrSpMM_MatchesCpuReference(int rows, int cols, int n, double density)
    {
        Skip.IfNot(CudaSparseBackend.IsAvailable,
            "CUDA driver not available; managed custom-kernel SpMM path inactive on this host.");

        var rng = RandomHelper.CreateSeededRandom(1234);
        var (rowPtr, colIdx, vals) = BuildCsr(rows, cols, density, rng);
        var b = RandDense(cols * n, rng);

        var want = CpuReference(rowPtr, colIdx, vals, b, rows, n);
        var got = CudaSparseBackend.SpMM(rowPtr, colIdx, vals, b, rows, cols, n);

        Assert.Equal(want.Length, got.Length);
        double maxAbs = 0, maxMag = 0;
        for (int i = 0; i < want.Length; i++)
        {
            maxAbs = Math.Max(maxAbs, Math.Abs(got[i] - want[i]));
            maxMag = Math.Max(maxMag, Math.Abs(want[i]));
        }
        double rel = maxAbs / Math.Max(1e-6, maxMag);
        Assert.True(rel < 1e-3,
            $"custom csr_spmm rel error {rel:E2} exceeds 1e-3 at shape {rows}x{cols}x{n} (density {density}) — likely a kernel/launch bug.");
    }

    [SkippableFact]
    public void CustomCsrSpMM_IsDeterministic_AcrossRepeatedCalls()
    {
        Skip.IfNot(CudaSparseBackend.IsAvailable,
            "CUDA driver not available; managed custom-kernel SpMM path inactive on this host.");

        var rng = RandomHelper.CreateSeededRandom(99);
        const int rows = 200, cols = 160, n = 48;
        var (rowPtr, colIdx, vals) = BuildCsr(rows, cols, 0.1, rng);
        var b = RandDense(cols * n, rng);

        var first = CudaSparseBackend.SpMM(rowPtr, colIdx, vals, b, rows, cols, n);
        var second = CudaSparseBackend.SpMM(rowPtr, colIdx, vals, b, rows, cols, n);

        // Each output element is reduced by exactly one thread in stored order, so
        // repeated launches must be bit-identical (no atomics, no cross-thread race).
        Assert.Equal(first, second);
    }
}
