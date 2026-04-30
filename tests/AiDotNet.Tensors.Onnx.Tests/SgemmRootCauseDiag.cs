// SimdGemm.SgemmWithCachedB / SgemmWithInt8CachedB are gated to .NET 5+
// in src/AiDotNet.Tensors/Engines/Simd/SimdGemm.cs (the AVX-512 / Vector256
// paths they wrap aren't available on net471). Mirror that gate here so
// the net471 leg of this multi-target test project compiles.
#if NET5_0_OR_GREATER
using System.Diagnostics;
using AiDotNet.Tensors.Engines.Simd;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Onnx.Tests;

/// <summary>
/// Root-cause harness for the BERT FFN MatMul bottleneck. FFN up
/// [256, 768] × [768, 3072] runs at ~115 GFLOP/s in <c>SimdGemm.Sgemm</c>
/// vs ORT's ~400 GFLOP/s — a ~3.5× kernel gap on a 16-core AMD box
/// (AVX2, no AVX-512, no VML).
///
/// <para>This tests the hypotheses:</para>
/// <list type="number">
///   <item><b>Parallel dispatch overhead</b> — measure single-thread
///   <see cref="SimdGemm.SgemmSequential"/> to see what one core can
///   actually deliver. If parallel = 4×sequential, we have linear
///   scaling. If parallel &lt; 4×sequential, we're losing to barriers.</item>
///   <item><b>Packing overhead</b> — Sgemm packs A and B per panel.
///   A direct non-packed scalar-edge kernel might win on small-m
///   shapes where packing doesn't amortise. We contrast
///   SgemmSequential (packed) vs SgemmScalar (no packing) on sequential.</item>
///   <item><b>Kernel throughput</b> — 1.2 GFLOP single-thread time gives
///   us the peak per-core achievable. Compare to AVX2 peak of
///   64 GFLOP/s (2 FMAs/cycle × 8 lanes × 4 GHz).</item>
/// </list>
///
/// <para>Gated behind <c>AIDOTNET_RUN_PERF_HARNESS=1</c>.</para>
/// </summary>
public class SgemmRootCauseDiag
{
    private readonly ITestOutputHelper _output;
    public SgemmRootCauseDiag(ITestOutputHelper output) { _output = output; }

    private const int Warmup = 3;
    private const int Iters = 10;

    [SkippableFact]
    public void LocaliseBertFfnSgemmBottleneck()
    {
        Skip.IfNot(
            System.Environment.GetEnvironmentVariable("AIDOTNET_RUN_PERF_HARNESS") == "1",
            "Set AIDOTNET_RUN_PERF_HARNESS=1 to run this evidence harness.");

        (int m, int k, int n, string label)[] cases =
        {
            (256, 768,  3072, "BERT FFN up    [256, 768]×[768,3072]"),
            (256, 3072, 768,  "BERT FFN down  [256,3072]×[3072,768]"),
            (256, 768,  768,  "BERT QKV proj  [256, 768]×[768, 768]"),
        };

        foreach (var cs in cases)
        {
            double flops = 2.0 * cs.m * cs.k * cs.n;
            _output.WriteLine($"=== {cs.label}  ({flops / 1e6:F1} MFLOP) ===");

            double tParallel = TimeSgemmParallel(cs.m, cs.k, cs.n);
            _output.WriteLine($"  [1] SimdGemm.Sgemm (parallel, packed):        {tParallel * 1000:F1} µs  ({flops / tParallel / 1e9:F1} GFLOP/s)");

            double tSequential = TimeSgemmSequential(cs.m, cs.k, cs.n);
            _output.WriteLine($"  [2] SimdGemm.SgemmSequential (1 core, packed):{tSequential * 1000:F1} µs  ({flops / tSequential / 1e6:F0} GFLOP/s)");

            double scalingEfficiency = tSequential / tParallel / System.Environment.ProcessorCount;
            _output.WriteLine($"     Parallel scaling: {tSequential / tParallel:F2}× on {System.Environment.ProcessorCount} cores  ({scalingEfficiency * 100:F0}% efficiency)");

            double tEmbarrassingly = TimeEmbarrassinglyParallel(cs.m, cs.k, cs.n);
            _output.WriteLine($"  [3] 16× independent SgemmSequential, Parallel.For: {tEmbarrassingly * 1000:F1} µs  ({16.0 * flops / tEmbarrassingly / 1e6:F0} GFLOP/s aggregate)");
            double idealEmbarrassingly = tSequential;
            _output.WriteLine($"     Parallel-For scaling vs serial 1×: {16.0 * tSequential / 16.0 / tEmbarrassingly:F2}× ({16.0 * tSequential / tEmbarrassingly / 16.0 * 100:F0}% of ideal 1 call/core)");

            // Path A measurement: how much of the parallel Sgemm time is
            // spent in packing B? A memory-bound strided→tile repack of
            // a ~9 MB B matrix at DDR4 bandwidth would be 100-500 µs parallel.
            // If we pre-pack once at plan compile time, this vanishes.
            double tPackBParallel = TimePackBOnlyParallel(cs.k, cs.n);
            _output.WriteLine($"  [4] PackB only, 16× parallel (memory-bound):    {tPackBParallel * 1000:F1} µs  ({(double)cs.k * cs.n * 4 / tPackBParallel / 1e6:F0} MB/s)");
            _output.WriteLine($"     PackB fraction of parallel Sgemm: {tPackBParallel / tParallel * 100:F1}%");

            // Path A vs Path D: cached float B vs cached int8 B
            double tCachedFloat = TimeSgemmCachedFloat(cs.m, cs.k, cs.n);
            _output.WriteLine($"  [5] SgemmWithCachedB (Path A, float):           {tCachedFloat * 1000:F1} µs  ({flops / tCachedFloat / 1e6:F0} GFLOP/s)");

            double tCachedInt8 = TimeSgemmCachedInt8(cs.m, cs.k, cs.n);
            _output.WriteLine($"  [6] SgemmWithInt8CachedB (Path D, int8 weight): {tCachedInt8 * 1000:F1} µs  ({flops / tCachedInt8 / 1e6:F0} GFLOP/s)");
            _output.WriteLine($"     Path D vs Path A: {tCachedFloat / tCachedInt8:F2}× ({(tCachedFloat - tCachedInt8) * 1000:+F1;-F1} µs)");

            _output.WriteLine("");
        }
    }

    private static double TimeSgemmParallel(int m, int k, int n)
    {
        var a = Rand(0xE01, m * k);
        var b = Rand(0xE02, k * n);
        var c = new float[m * n];

        for (int i = 0; i < Warmup; i++)
            SimdGemm.Sgemm(a, b, c, m, k, n);

        var sw = Stopwatch.StartNew();
        for (int i = 0; i < Iters; i++)
            SimdGemm.Sgemm(a, b, c, m, k, n);
        sw.Stop();
        return sw.Elapsed.TotalMilliseconds / Iters;
    }

    private static double TimeSgemmSequential(int m, int k, int n)
    {
        var a = Rand(0xE03, m * k);
        var b = Rand(0xE04, k * n);
        var c = new float[m * n];

        for (int i = 0; i < Warmup; i++)
            SimdGemm.SgemmSequential(a, b, c, m, k, n);

        var sw = Stopwatch.StartNew();
        for (int i = 0; i < Iters; i++)
            SimdGemm.SgemmSequential(a, b, c, m, k, n);
        sw.Stop();
        return sw.Elapsed.TotalMilliseconds / Iters;
    }

    // Test: 16 INDEPENDENT problems run across 16 cores via Parallel.For.
    // Each worker owns its own (a, b, c) — no sharing, no contention on
    // packed buffers or output writes. This is the theoretical best case
    // for parallel scaling. If this gets 12× scaling (3/4 efficiency), the
    // problem is SgemmTiled's dispatch. If this ALSO gets only 3×, the
    // problem is system-level (thermal throttling, memory bandwidth,
    // thread pool contention).
    private static double TimeEmbarrassinglyParallel(int m, int k, int n)
    {
        int cores = System.Environment.ProcessorCount;
        var bufs = new (float[] a, float[] b, float[] c)[cores];
        for (int i = 0; i < cores; i++)
            bufs[i] = (Rand(0xE10 + i, m * k), Rand(0xE20 + i, k * n), new float[m * n]);

        for (int warm = 0; warm < Warmup; warm++)
            System.Threading.Tasks.Parallel.For(0, cores, i =>
                SimdGemm.SgemmSequential(bufs[i].a, bufs[i].b, bufs[i].c, m, k, n));

        var sw = Stopwatch.StartNew();
        for (int it = 0; it < Iters; it++)
            System.Threading.Tasks.Parallel.For(0, cores, i =>
                SimdGemm.SgemmSequential(bufs[i].a, bufs[i].b, bufs[i].c, m, k, n));
        sw.Stop();
        return sw.Elapsed.TotalMilliseconds / Iters;
    }

    private static double TimeSgemmCachedFloat(int m, int k, int n)
    {
        var a = Rand(0xE05, m * k);
        var b = Rand(0xE06, k * n);
        var c = new float[m * n];
        for (int i = 0; i < Warmup; i++) SimdGemm.SgemmWithCachedB(a, b, c, m, k, n);
        var sw = Stopwatch.StartNew();
        for (int i = 0; i < Iters; i++) SimdGemm.SgemmWithCachedB(a, b, c, m, k, n);
        sw.Stop();
        return sw.Elapsed.TotalMilliseconds / Iters;
    }

    private static double TimeSgemmCachedInt8(int m, int k, int n)
    {
        var a = Rand(0xE07, m * k);
        var b = Rand(0xE08, k * n);
        var c = new float[m * n];
        for (int i = 0; i < Warmup; i++) SimdGemm.SgemmWithInt8CachedB(a, b, c, m, k, n);
        var sw = Stopwatch.StartNew();
        for (int i = 0; i < Iters; i++) SimdGemm.SgemmWithInt8CachedB(a, b, c, m, k, n);
        sw.Stop();
        return sw.Elapsed.TotalMilliseconds / Iters;
    }

    // Proxy for PackB's cost: PackB rearranges B into a tile-striped
    // layout. It's memory-bound (reads k*n floats, writes k*n floats),
    // so its cost tracks copy-bandwidth closely. We split the copy across
    // 16 chunks to mirror the parallel pack dispatch.
    private static double TimePackBOnlyParallel(int k, int n)
    {
        var b = Rand(0xE05, k * n);
        var packed = new float[k * n];
        int cores = System.Environment.ProcessorCount;

        for (int warm = 0; warm < Warmup; warm++)
            System.Threading.Tasks.Parallel.For(0, cores, i =>
            {
                int chunk = (k * n + cores - 1) / cores;
                int start = i * chunk;
                int len = System.Math.Min(chunk, k * n - start);
                if (len > 0)
                    System.Array.Copy(b, start, packed, start, len);
            });

        var sw = Stopwatch.StartNew();
        for (int it = 0; it < Iters; it++)
            System.Threading.Tasks.Parallel.For(0, cores, i =>
            {
                int chunk = (k * n + cores - 1) / cores;
                int start = i * chunk;
                int len = System.Math.Min(chunk, k * n - start);
                if (len > 0)
                    System.Array.Copy(b, start, packed, start, len);
            });
        sw.Stop();
        return sw.Elapsed.TotalMilliseconds / Iters;
    }

    private static float[] Rand(int seed, int n)
    {
        var rng = new System.Random(seed);
        var a = new float[n];
        for (int i = 0; i < n; i++) a[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        return a;
    }
}
#endif
