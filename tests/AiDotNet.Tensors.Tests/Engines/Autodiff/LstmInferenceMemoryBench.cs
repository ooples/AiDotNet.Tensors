using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines.Autodiff;

/// <summary>
/// Probes whether LSTM inference (LstmSequenceForward) grows memory per call or per batch on the
/// Tensors side — the third-party benchmark reported LSTM inference RSS rising with batch.
/// </summary>
public class LstmInferenceMemoryBench
{
    private readonly ITestOutputHelper _o;
    public LstmInferenceMemoryBench(ITestOutputHelper o) => _o = o;

    // Process-wide / per-thread allocation counter. net471 ships neither
    // GC.GetTotalAllocatedBytes nor GetAllocatedBytesForCurrentThread, so
    // we fall back to GetTotalMemory (live managed heap) on that TFM —
    // close enough for the short bench windows below.
    private static long AllocatedBytes()
    {
#if NET6_0_OR_GREATER
        return GC.GetTotalAllocatedBytes(precise: true);
#else
        return GC.GetTotalMemory(forceFullCollection: false);
#endif
    }

    private static Tensor<float> Rand(int[] shape, int seed)
    {
        var rng = new Random(seed);
        var t = new Tensor<float>(shape);
        var s = t.AsWritableSpan();
        for (int i = 0; i < s.Length; i++) s[i] = (float)(rng.NextDouble() * 0.1 - 0.05);
        return t;
    }

    [Fact(Skip = "Benchmark — run manually")]
    public void Probe_LstmInferenceMemory()
    {
        var eng = new CpuEngine();
        int inDim = 32, hidden = 64, seq = 20;
        var wIh = Rand(new[] { 4 * hidden, inDim }, 1);
        var wHh = Rand(new[] { 4 * hidden, hidden }, 2);
        var bIh = Rand(new[] { 4 * hidden }, 3);
        var bHh = Rand(new[] { 4 * hidden }, 4);

        _o.WriteLine($"ENGINE={eng.GetType().Name}  LSTM in={inDim} hidden={hidden} seq={seq}");
        foreach (int B in new[] { 1, 8, 32, 128 })
        {
            var x = Rand(new[] { B, seq, inDim }, 10 + B);
            // warmup
            for (int i = 0; i < 20; i++) eng.LstmSequenceForward(x, null, null, wIh, wHh, bIh, bHh, returnSequences: true);

            GC.Collect(); GC.WaitForPendingFinalizers(); GC.Collect();
            long heapBefore = MemoryMetrics.ManagedHeapBytes;
            long allocBefore = AllocatedBytes();
            const int iters = 200;
            double peakRss;
            using (var sampler = new MemoryMetrics.PeakRssSampler(2))
            {
                for (int i = 0; i < iters; i++) eng.LstmSequenceForward(x, null, null, wIh, wHh, bIh, bHh, returnSequences: true);
                peakRss = sampler.PeakMb;
            }
            long allocAfter = AllocatedBytes();
            GC.Collect(); GC.WaitForPendingFinalizers(); GC.Collect();
            long heapAfter = MemoryMetrics.ManagedHeapBytes;

            double allocKbPerCall = (allocAfter - allocBefore) / 1024.0 / iters;
            double heapGrowthMb = (heapAfter - heapBefore) / (1024.0 * 1024.0);
            _o.WriteLine($"  B={B,3}: alloc/call {allocKbPerCall,8:F1} KB | heap-growth-after-{iters}-calls {heapGrowthMb,7:F2} MB | peak-RSS {peakRss,7:F1} MB");
        }
        _o.WriteLine("  (alloc/call ~0 + heap-growth ~0 => pooled, no leak; the report's 'rise' is then the");
        _o.WriteLine("   batch-scaling output buffer + retained cache footprint, not a Tensors per-call leak.)");

        // Multi-shape sweep: mimic the benchmark exercising MANY distinct shapes (4 models x batches
        // 1/8/32/128 x intermediate ops). Measure how much the per-shape (unbounded-global) caches
        // retain. This is the 1.6 GB-vs-330 MB suspect.
        GC.Collect(); GC.WaitForPendingFinalizers(); GC.Collect();
        double rssStart = MemoryMetrics.CurrentProcessRssMb;
        double heapStart = MemoryMetrics.ManagedHeapMb;
        int shapes = 0;
        foreach (int B in new[] { 1, 8, 32, 128 })
            foreach (int d in new[] { 128, 256, 512, 768 })
            {
                var aa = Rand(new[] { B * 16, d }, B + d);
                var ww = Rand(new[] { d, d }, d);
                for (int i = 0; i < 5; i++) eng.TensorMatMul(aa, ww); // distinct [B*16,d]x[d,d] shapes
                shapes++;
            }
        GC.Collect(); GC.WaitForPendingFinalizers(); GC.Collect();
        _o.WriteLine($"  MULTI-SHAPE ({shapes} distinct matmul shapes): RSS {rssStart:F0} -> {MemoryMetrics.CurrentProcessRssMb:F0} MB | managed-heap {heapStart:F0} -> {MemoryMetrics.ManagedHeapMb:F0} MB");
    }
}
