using System;
using System.Collections.Generic;
using System.Diagnostics;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.Tests.Engines.BlasManaged;
using AiDotNet.Tensors.Tests.Engines.BlasManaged.Catalog;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// Authoritative GPU SpMM bench (#515, P6 final step): times AiDotNet's managed CUDA
/// CSR kernels (csr_spmm / _warp / _vec4, picked by <see cref="CudaSparseBackend"/>'s
/// heuristic) against native cuSPARSE <c>cusparseSpMM</c> over the
/// <see cref="SpecializedShapeCatalog.SpMM"/> workload shapes, and prints the win rate
/// + max-loss multiple the owner freezes into <see cref="SpecializedPerfBar"/>'s
/// <c>GpuSpMM*</c> constants. Once frozen, this also gates: custom must clear the bar.
///
/// <para>Runner-only: gated on <c>AIDOTNET_PERF_RUNNER=1</c> (the authoritative GPU
/// runner) AND a usable CUDA driver — it no-ops/skips everywhere else, so it can't
/// fabricate numbers on hardware that can't run the kernels. Shapes are benched in
/// FP32 (the precision the cuSPARSE host wrapper compares at); the FP64 kernel's
/// correctness is covered by the parity tests, not here.</para>
/// </summary>
[Trait("Category", "Perf")]
public class GpuSpMMBenchHarness
{
    private readonly ITestOutputHelper _out;
    public GpuSpMMBenchHarness(ITestOutputHelper o) => _out = o;

    [SkippableFact]
    public void Bench_CustomVsCuSparse_FreezeGpuSpMMBar()
    {
        Skip.IfNot(Environment.GetEnvironmentVariable("AIDOTNET_PERF_RUNNER") == "1",
            "Set AIDOTNET_PERF_RUNNER=1 on the authoritative GPU runner to bench GPU SpMM.");
        Skip.IfNot(CudaSparseBackend.IsAvailable, "CUDA driver/NVRTC not available on this host.");

        bool haveCuSparse = CuSparseBackend.IsAvailable;
        _out.WriteLine($"custom = managed CUDA csr_spmm/_warp/_vec4 ; cuSPARSE available = {haveCuSparse}");
        _out.WriteLine($"{"shape",-32}{"nnz",11}{"custom GF/s",14}{"cuSPARSE GF/s",16}{"ratio",9}");

        var rng = RandomHelper.CreateSeededRandom(515);
        int wins = 0, compared = 0;
        double maxLoss = 1.0;

        foreach (var s in SpecializedShapeCatalog.SpMM)
        {
            double density = s.DensityPercent / 100.0;
            var (rowPtr, colIdx) = BuildCsr(s.Rows, s.Cols, density, rng, out int nnz);
            var vals = RandF(nnz, rng);
            var b = RandF(s.Cols * s.N, rng);
            double flop = 2.0 * nnz * s.N;

            double customMs = MinMs(20, () => CudaSparseBackend.SpMM(rowPtr, colIdx, vals, b, s.Rows, s.Cols, s.N));
            double customGf = flop / (customMs * 1e6);

            double cuGf = double.NaN;
            string ratio = "    --";
            if (haveCuSparse)
            {
                double cuMs = MinMs(20, () => CuSparseBackend.SpMM(rowPtr, colIdx, vals, b, s.Rows, s.Cols, s.N));
                cuGf = flop / (cuMs * 1e6);
                double r = cuMs / customMs;   // > 1 => custom faster (a win)
                ratio = $"{r,7:F2}x";
                compared++;
                if (r >= 1.0) wins++; else maxLoss = Math.Max(maxLoss, 1.0 / r);
            }

            string cuCol = double.IsNaN(cuGf) ? "--" : cuGf.ToString("F1");
            _out.WriteLine($"{s.Name,-32}{nnz,11}{customGf,14:F1}{cuCol,16}{ratio,9}");
        }

        if (compared == 0)
        {
            _out.WriteLine("cuSPARSE unavailable — custom throughput reported only; the bar needs a cuSPARSE baseline to freeze.");
            return;
        }

        int winRate = (int)Math.Round(100.0 * wins / compared);
        _out.WriteLine("");
        _out.WriteLine($"=> custom wins {wins}/{compared} = {winRate}% ; max loss {maxLoss:F2}x");
        _out.WriteLine($"=> FREEZE into SpecializedPerfBar: GpuSpMMMinWinRatePercent = {winRate}; GpuSpMMMaxLossMultiple = {maxLoss:F2}");

        if (SpecializedPerfBar.GpuSpMMBarFrozen)
        {
            Assert.True(winRate >= SpecializedPerfBar.GpuSpMMMinWinRatePercent,
                $"GPU SpMM win rate {winRate}% < frozen bar {SpecializedPerfBar.GpuSpMMMinWinRatePercent}%.");
            Assert.True(maxLoss <= SpecializedPerfBar.GpuSpMMMaxLossMultiple,
                $"GPU SpMM max loss {maxLoss:F2}x > frozen bar {SpecializedPerfBar.GpuSpMMMaxLossMultiple}x.");
        }
        else
        {
            _out.WriteLine("(GpuSpMM bar not yet frozen — informational run; set the constants above to start gating.)");
        }
    }

    private static double MinMs(int iters, Action f)
    {
        for (int i = 0; i < 3; i++) f();   // warmup (JIT/NVRTC + caches)
        double m = double.MaxValue;
        for (int i = 0; i < iters; i++)
        {
            var sw = Stopwatch.StartNew();
            f();
            sw.Stop();
            m = Math.Min(m, sw.Elapsed.TotalMilliseconds);
        }
        return m;
    }

    private static (int[] rowPtr, int[] colIdx) BuildCsr(int rows, int cols, double density, Random rng, out int nnz)
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
}
