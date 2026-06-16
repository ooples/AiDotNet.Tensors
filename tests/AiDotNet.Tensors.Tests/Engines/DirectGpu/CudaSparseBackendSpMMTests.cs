using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.Tensors.Helpers;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// Parity + determinism guards for the managed custom-kernel CSR SpMM that #515 (P6)
/// makes the default GPU SpMM (replacing native cuSPARSE). Covers the CUDA float warp
/// + scalar kernels and the CUDA FP64 kernel. Every kernel accumulates each output
/// element in one thread over the row's stored order, so each must (a) match a
/// hand-written CPU CSR reference and (b) be bit-identical across repeated launches.
///
/// <para>These run only where the CUDA driver + NVRTC are present (the self-hosted
/// runner) — on a host without them they skip. The CPU reference is the oracle and
/// uses the same per-output reduction order as the kernels, so an indexing/launch bug
/// shows up as O(1) error, not rounding. (The HIP mirror is implemented but not yet
/// wired/validated — it needs a healthy ROCm runner; see HipCustomSparseBackend.)</para>
/// </summary>
[Collection("DirectGpuSerial")]
public class CudaSparseBackendSpMMTests
{
    private static float[] CpuReference(int[] rowPtr, int[] colIdx, float[] vals, float[] b, int rows, int n)
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

    private static double[] CpuReferenceDouble(int[] rowPtr, int[] colIdx, double[] vals, double[] b, int rows, int n)
    {
        var outp = new double[rows * n];
        for (int r = 0; r < rows; r++)
            for (int j = 0; j < n; j++)
            {
                double acc = 0.0;
                for (int p = rowPtr[r]; p < rowPtr[r + 1]; p++)
                    acc += vals[p] * b[colIdx[p] * n + j];
                outp[r * n + j] = acc;
            }
        return outp;
    }

    // Random CSR with ascending column indices per row (the layout the kernels expect).
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

    private static double[] RandD(int len, Random rng)
    {
        var a = new double[len];
        for (int i = 0; i < len; i++) a[i] = rng.NextDouble() * 2 - 1;
        return a;
    }

    private static double RelErr(ReadOnlySpan<float> got, ReadOnlySpan<float> want)
    {
        double maxAbs = 0, maxMag = 0;
        for (int i = 0; i < want.Length; i++)
        {
            maxAbs = Math.Max(maxAbs, Math.Abs(got[i] - want[i]));
            maxMag = Math.Max(maxMag, Math.Abs(want[i]));
        }
        return maxAbs / Math.Max(1e-6, maxMag);
    }

    private static double RelErr(ReadOnlySpan<double> got, ReadOnlySpan<double> want)
    {
        double maxAbs = 0, maxMag = 0;
        for (int i = 0; i < want.Length; i++)
        {
            maxAbs = Math.Max(maxAbs, Math.Abs(got[i] - want[i]));
            maxMag = Math.Max(maxMag, Math.Abs(want[i]));
        }
        return maxAbs / Math.Max(1e-9, maxMag);
    }

    // Kernel pick: N%4==0 -> vec4, else N<=64 -> warp, else 2-D scalar. The cases
    // below exercise ALL THREE float kernels against the same CPU reference.
    public static IEnumerable<object[]> Shapes() => new[]
    {
        new object[] { 64, 48, 16, 0.20 },    // N%4==0 -> vec4 (128-bit loads)
        new object[] { 128, 96, 128, 0.10 },  // N%4==0 -> vec4
        new object[] { 256, 256, 64, 0.05 },  // N%4==0 -> vec4
        new object[] { 100, 70, 1, 0.30 },    // N=1 (SpMV-like), N<=64 -> warp
        new object[] { 200, 100, 130, 0.08 }, // N>64, N%4!=0 -> 2-D scalar
    };

    [SkippableTheory]
    [MemberData(nameof(Shapes))]
    public void CudaFloat_MatchesCpuReference(int rows, int cols, int n, double density)
    {
        Skip.IfNot(CudaSparseBackend.IsAvailable, "CUDA driver/NVRTC not available on this host.");

        var rng = RandomHelper.CreateSeededRandom(1234);
        var (rowPtr, colIdx) = BuildCsrPattern(rows, cols, density, rng, out int nnz);
        var vals = RandF(nnz, rng);
        var b = RandF(cols * n, rng);

        var want = CpuReference(rowPtr, colIdx, vals, b, rows, n);
        var got = CudaSparseBackend.SpMM(rowPtr, colIdx, vals, b, rows, cols, n);

        Assert.Equal(want.Length, got.Length);
        double rel = RelErr(got, want);
        Assert.True(rel < 1e-3, $"CUDA float csr_spmm rel error {rel:E2} > 1e-3 at {rows}x{cols}x{n} (density {density}).");
    }

    [SkippableFact]
    public void CudaFloat_IsDeterministic_AcrossRepeatedCalls()
    {
        Skip.IfNot(CudaSparseBackend.IsAvailable, "CUDA driver/NVRTC not available on this host.");

        var rng = RandomHelper.CreateSeededRandom(99);
        const int rows = 200, cols = 160, n = 48;
        var (rowPtr, colIdx) = BuildCsrPattern(rows, cols, 0.1, rng, out int nnz);
        var vals = RandF(nnz, rng);
        var b = RandF(cols * n, rng);

        var first = CudaSparseBackend.SpMM(rowPtr, colIdx, vals, b, rows, cols, n);
        var second = CudaSparseBackend.SpMM(rowPtr, colIdx, vals, b, rows, cols, n);
        Assert.Equal(first, second);
    }

    [SkippableTheory]
    [MemberData(nameof(Shapes))]
    public void CudaDouble_MatchesCpuReference(int rows, int cols, int n, double density)
    {
        Skip.IfNot(CudaSparseBackend.IsAvailable, "CUDA driver/NVRTC not available on this host.");

        var rng = RandomHelper.CreateSeededRandom(4321);
        var (rowPtr, colIdx) = BuildCsrPattern(rows, cols, density, rng, out int nnz);
        var vals = RandD(nnz, rng);
        var b = RandD(cols * n, rng);

        var want = CpuReferenceDouble(rowPtr, colIdx, vals, b, rows, n);
        var got = CudaSparseBackend.SpMM(rowPtr, colIdx, vals, b, rows, cols, n);

        Assert.Equal(want.Length, got.Length);
        double rel = RelErr(got, want);
        Assert.True(rel < 1e-9, $"CUDA double csr_spmm_double rel error {rel:E2} > 1e-9 at {rows}x{cols}x{n} (density {density}).");
    }
}
