using System;
using System.Diagnostics;
using AiDotNet.Tensors.Engines.Compilation;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

// Not a correctness test — a manual timing harness comparing the three double CPU fused-Adam
// paths (jagged multi, flat-moment multi, contiguous single-pass) on two workload shapes:
// "few large" (VALLEX-like transformer weight tensors) vs "many small" (per-layer biases /
// LayerNorm). Run explicitly:
//   dotnet test --filter FullyQualifiedName~FusedOptimizerPathBenchmark
// Skipped in normal runs so it never gates CI on wall-clock.
public class FusedOptimizerPathBenchmark
{
    private readonly ITestOutputHelper _out;
    public FusedOptimizerPathBenchmark(ITestOutputHelper o) => _out = o;

    private const string SkipReason = "manual perf harness; run explicitly with --filter";

    [Theory(Skip = SkipReason)]
    [InlineData("few-large (VALLEX-like)", 40, 262144)]   // 40 tensors x 256K = ~10.5M doubles
    [InlineData("many-small (biases/LN)", 4000, 256)]     // 4000 tensors x 256 = ~1M doubles
    [InlineData("many-tiny non-aligned", 40000, 7)]       // worst case for per-tensor tails
    public void ComparePaths(string label, int count, int lenEach)
    {
        const double lr = 5e-4, b1 = 0.9, b2 = 0.999, eps = 1e-8, wd = 1e-2;
        var lens = new int[count];
        var offsets = new int[count];
        int total = 0;
        for (int t = 0; t < count; t++) { lens[t] = lenEach; offsets[t] = total; total += lenEach; }

        var rng = new Random(7);
        double[][] pJag = new double[count][], pFlatM = new double[count][], pCon = new double[count][], g = new double[count][];
        double[][] mJag = new double[count][], vJag = new double[count][];
        var mFlat = new double[total]; var vFlat = new double[total];
        var mFlat2 = new double[total]; var vFlat2 = new double[total];
        var pStage = new double[total]; var gStage = new double[total];
        for (int t = 0; t < count; t++)
        {
            pJag[t] = new double[lenEach]; pFlatM[t] = new double[lenEach]; pCon[t] = new double[lenEach];
            g[t] = new double[lenEach]; mJag[t] = new double[lenEach]; vJag[t] = new double[lenEach];
            for (int i = 0; i < lenEach; i++)
            {
                double pv = rng.NextDouble(), gv = rng.NextDouble() * 0.1;
                pJag[t][i] = pFlatM[t][i] = pCon[t][i] = pv;
                g[t][i] = gv;
            }
        }

        const int warm = 5, iters = 30;
        // Warmup + time each path (AdamW).
        double tJag = Time(() => FusedOptimizer.AdamWUpdateSimdMulti(pJag, g, mJag, vJag, lens, count, lr, b1, b2, eps, wd, 0.1, 0.001), warm, iters);
        double tFlatM = Time(() => FusedOptimizer.AdamWUpdateSimdMultiFlatMoments(pFlatM, g, mFlat, vFlat, lens, offsets, count, lr, b1, b2, eps, wd, 0.1, 0.001), warm, iters);
        double tCon = Time(() => FusedOptimizer.AdamWUpdateSimdContiguous(pCon, g, lens, offsets, count, pStage, gStage, mFlat2, vFlat2, total, lr, b1, b2, eps, wd, 0.1, 0.001), warm, iters);

        _out.WriteLine($"=== {label}: {count} tensors x {lenEach} = {total:N0} doubles ===");
        _out.WriteLine($"  jagged multi      : {tJag,8:F3} ms/step");
        _out.WriteLine($"  flat-moment multi : {tFlatM,8:F3} ms/step  ({tJag / tFlatM:F2}x vs jagged)");
        _out.WriteLine($"  contiguous 1-pass : {tCon,8:F3} ms/step  ({tJag / tCon:F2}x vs jagged)");
    }

    private static double Time(Action a, int warm, int iters)
    {
        for (int i = 0; i < warm; i++) a();
        var sw = Stopwatch.StartNew();
        for (int i = 0; i < iters; i++) a();
        sw.Stop();
        return sw.Elapsed.TotalMilliseconds / iters;
    }
}
