using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Benchmarks.PyTorchComparison;

/// <summary>
/// Issue #436 dev probe: the AIsEval MHA ([128, 32, 64], 4 heads) through the batch-fused path against the three-phase
/// path, at several parallel degrees, interleaved round by round; and the two paths' outputs compared, with and without
/// a mask. Run: <c>--aiseval-mha-probe</c>.
/// </summary>
internal static class MhaFusionProbe
{
    public static void Run()
    {
        const int batch = 128, seq = 32, dModel = 64, heads = 4;
        var rng = new Random(436);
        Tensor<float> Rand(float s, params int[] shape) =>
            new(Enumerable.Range(0, shape.Aggregate(1, (a, b) => a * b)).Select(_ => (float)(rng.NextDouble() * 2 - 1) * s).ToArray(), shape);
        var x = Rand(1f, batch, seq, dModel);
        var q = Rand(0.125f, dModel, dModel);
        var k = Rand(0.125f, dModel, dModel);
        var v = Rand(0.125f, dModel, dModel);
        var o = Rand(0.125f, dModel, dModel);
        var engine = new CpuEngine();
        bool saved = CpuEngine.MhaBatchFusedEnabled;
        int savedDop = CpuParallelSettings.MaxDegreeOfParallelism;

        float[] Run(bool fused, Tensor<bool>? mask)
        {
            CpuEngine.MhaBatchFusedEnabled = fused;
            try { return engine.MultiHeadAttentionForward(x, q, k, v, o, heads, mask).ToArray(); }
            finally { CpuEngine.MhaBatchFusedEnabled = saved; }
        }

        double Diff(float[] a, float[] b)
        {
            double worst = 0, big = 1e-30;
            for (int i = 0; i < a.Length; i++) { worst = Math.Max(worst, Math.Abs(a[i] - b[i])); big = Math.Max(big, Math.Abs(a[i])); }
            return worst / big;
        }

        Console.WriteLine($"fused vs three-phase, no mask: {Diff(Run(false, null), Run(true, null)):G3}");
        // A per-batch padding mask [batch, 1, seq, seq]: batch b sees its first seq - (b % 7) keys.
        var maskData = new bool[batch * seq * seq];
        for (int b = 0; b < batch; b++)
            for (int i = 0; i < seq; i++)
                for (int j = 0; j < seq; j++)
                    maskData[(b * seq + i) * seq + j] = j < seq - (b % 7);
        var mask = new Tensor<bool>(maskData, new[] { batch, 1, seq, seq });
        Console.WriteLine($"fused vs three-phase, padding mask: {Diff(Run(false, mask), Run(true, mask)):G3}");

        {
            for (int i = 0; i < 50; i++) engine.MultiHeadAttentionForward(x, q, k, v, o, heads);
            long before = GC.GetAllocatedBytesForCurrentThread();
            int g0 = GC.CollectionCount(0), g1 = GC.CollectionCount(1), g2 = GC.CollectionCount(2);
            const int n = 500;
            for (int i = 0; i < n; i++) engine.MultiHeadAttentionForward(x, q, k, v, o, heads);
            Console.WriteLine($"MultiHeadAttentionForward: {(GC.GetAllocatedBytesForCurrentThread() - before) / n:N0} B allocated per call " +
                $"(calling thread); GCs over {n} calls: gen0 {GC.CollectionCount(0) - g0}, gen1 {GC.CollectionCount(1) - g1}, gen2 {GC.CollectionCount(2) - g2}");
        }
        {
            for (int i = 0; i < 50; i++) engine.MultiHeadAttentionForward(x, q, k, v, o, heads).Dispose();
            long before = GC.GetAllocatedBytesForCurrentThread();
            int g0 = GC.CollectionCount(0), g2 = GC.CollectionCount(2);
            const int n = 500;
            for (int i = 0; i < n; i++) engine.MultiHeadAttentionForward(x, q, k, v, o, heads).Dispose();
            Console.WriteLine($"with Dispose: {(GC.GetAllocatedBytesForCurrentThread() - before) / n:N0} B allocated per call; " +
                $"GCs over {n} calls: gen0 {GC.CollectionCount(0) - g0}, gen2 {GC.CollectionCount(2) - g2}");
        }
        var configs = new List<(string Name, Action Body)>();
        {
            var xa2 = x.ToArray();
            var fw2 = new float[64 * 192];
            float[] qa = q.ToArray(), ka = k.ToArray(), va = v.ToArray();
            for (int r = 0; r < 64; r++)
                for (int c = 0; c < 64; c++)
                {
                    fw2[r * 192 + c] = qa[r * 64 + c];
                    fw2[r * 192 + 64 + c] = ka[r * 64 + c];
                    fw2[r * 192 + 128 + c] = va[r * 64 + c];
                }
            configs.Add(("kernel dop=32, real weights, rented output", () =>
            {
                CpuParallelSettings.MaxDegreeOfParallelism = 32;
                try
                {
                    var outT = AutoTensorCache.RentOrAllocate<float>(new[] { batch, seq, dModel });
                    using var lease = outT.LeaseArray();
                    CpuEngine.MultiHeadAttentionBatchFused(xa2, fw2, o, null, lease.Array, batch, seq, dModel, heads, dModel / heads);
                }
                finally { CpuParallelSettings.MaxDegreeOfParallelism = savedDop; }
            }));
        }
        configs.Add(("4x GetMaterializedStreamingInt8/Int4", () =>
        {
            q.GetMaterializedStreamingInt8(); k.GetMaterializedStreamingInt8(); v.GetMaterializedStreamingInt8(); o.GetMaterializedStreamingInt8();
            q.GetMaterializedStreamingInt4(); k.GetMaterializedStreamingInt4(); v.GetMaterializedStreamingInt4(); o.GetMaterializedStreamingInt4();
        }));
        configs.Add(("3x weight Lease", () =>
        {
            using (var l1 = q.Lease()) { } using (var l2 = k.Lease()) { } using (var l3 = v.Lease()) { }
        }));
        configs.Add(("tape/graph checks", () =>
        {
            _ = AiDotNet.Tensors.Engines.Compilation.GraphMode.IsActive;
            _ = AiDotNet.Tensors.Engines.Autodiff.DifferentiableOps.ThreadTapeActive();
        }));
        configs.Add(("RentOrAllocate output [128,32,64]", () => { AutoTensorCache.RentOrAllocate<float>(new[] { batch, seq, dModel }); }));
        configs.Add(("ArrayPool rent+return 786K + 262K", () =>
        {
            var p1 = System.Buffers.ArrayPool<float>.Shared.Rent(batch * seq * 3 * dModel);
            var p2 = System.Buffers.ArrayPool<float>.Shared.Rent(batch * seq * dModel);
            System.Buffers.ArrayPool<float>.Shared.Return(p1);
            System.Buffers.ArrayPool<float>.Shared.Return(p2);
        }));
        configs.Add(("input.GetReadOnlyDataArray", () => { x.GetReadOnlyDataArray(); }));
        configs.Add(("batch-fused dop=32 + Dispose", () =>
        {
            CpuParallelSettings.MaxDegreeOfParallelism = 32;
            try { engine.MultiHeadAttentionForward(x, q, k, v, o, heads).Dispose(); }
            finally { CpuParallelSettings.MaxDegreeOfParallelism = savedDop; }
        }));
        // The per-batch-element GEMMs of the fused path, single-threaded.
        foreach (var (mm, kk, nn) in new[] { (32, 64, 192), (32, 16, 32), (32, 32, 16), (32, 64, 64) })
        {
            var a = new float[mm * kk];
            var bm = new float[kk * nn];
            var cm = new float[mm * nn];
            int m1 = mm, k1 = kk, n1 = nn;
            configs.Add(($"SgemmSequential {m1}x{k1}x{n1}", () =>
                AiDotNet.Tensors.Engines.Simd.SimdGemm.SgemmSequential(a, bm, cm, m1, k1, n1)));
        }
        {
            var xa = x.ToArray();
            var fw = new float[64 * 192];
            var outA = new float[batch * seq * dModel];
            foreach (int dop in new[] { 1, 32 })
            {
                int d = dop;
                configs.Add(($"kernel only dop={d}", () =>
                {
                    CpuParallelSettings.MaxDegreeOfParallelism = d;
                    try { CpuEngine.MultiHeadAttentionBatchFused(xa, fw, o, null, outA, batch, seq, dModel, heads, dModel / heads); }
                    finally { CpuParallelSettings.MaxDegreeOfParallelism = savedDop; }
                }));
            }
        }
        configs.Add(("batch-fused dop=1", () =>
        {
            CpuEngine.MhaBatchFusedEnabled = true;
            CpuParallelSettings.MaxDegreeOfParallelism = 1;
            try { engine.MultiHeadAttentionForward(x, q, k, v, o, heads); }
            finally
            {
                CpuEngine.MhaBatchFusedEnabled = saved;
                CpuParallelSettings.MaxDegreeOfParallelism = savedDop;
            }
        }));
        foreach (int dop in new[] { 16, 32, 64, Environment.ProcessorCount })
            foreach (bool fused in new[] { false, true })
            {
                int d = dop;
                bool f = fused;
                configs.Add(($"{(f ? "batch-fused" : "three-phase")} dop={d}", () =>
                {
                    CpuEngine.MhaBatchFusedEnabled = f;
                    CpuParallelSettings.MaxDegreeOfParallelism = d;
                    try { engine.MultiHeadAttentionForward(x, q, k, v, o, heads); }
                    finally
                    {
                        CpuEngine.MhaBatchFusedEnabled = saved;
                        CpuParallelSettings.MaxDegreeOfParallelism = savedDop;
                    }
                }));
            }

        // Dispatch anatomy: per-chunk start and end (us after the dispatch began) for a fixed ~46 us of serial work per
        // chunk (four 32x64x192 GEMMs), over many dispatches; reports medians across dispatches per chunk rank.
        foreach (int chunks in new[] { 16, 32, 64 })
        {
            var a = new float[32 * 64]; var bm = new float[64 * 192];
            var cs = Enumerable.Range(0, chunks).Select(_ => new float[32 * 192]).ToArray();
            var starts = new double[chunks]; var ends = new double[chunks];
            var startMed = new List<double>[chunks]; var endMed = new List<double>[chunks];
            for (int i = 0; i < chunks; i++) { startMed[i] = new(); endMed[i] = new(); }
            var clock = new Stopwatch();
            for (int rep = 0; rep < 300; rep++)
            {
                clock.Restart();
                PersistentParallelExecutor.Instance.Execute(chunks, chunks, c =>
                {
                    starts[c] = clock.Elapsed.TotalMilliseconds * 1000;
                    for (int g = 0; g < 4; g++)
                        AiDotNet.Tensors.Engines.Simd.SimdGemm.SgemmSequential(a, bm, cs[c], 32, 64, 192);
                    ends[c] = clock.Elapsed.TotalMilliseconds * 1000;
                });
                double total = clock.Elapsed.TotalMilliseconds * 1000;
                if (rep < 50) continue;
                var order = Enumerable.Range(0, chunks).OrderBy(i => starts[i]).ToArray();
                for (int i = 0; i < chunks; i++) { startMed[i].Add(starts[order[i]]); endMed[i].Add(ends[order[i]] - starts[order[i]]); }
                startMed[0].Add(0); // keep lists aligned in count
                endMed[0].Add(total);
            }
            double Med(List<double> l) { var z = l.OrderBy(v => v).ToArray(); return z[z.Length / 2]; }
            Console.WriteLine($"dispatch x{chunks}: start of chunk rank 0/25%/50%/75%/last = " +
                string.Join(" / ", new[] { 0, chunks / 4, chunks / 2, 3 * chunks / 4, chunks - 1 }.Select(i => Med(startMed[i]).ToString("F1"))) +
                $" us; work per chunk median {Med(endMed[chunks / 2]):F1} us; last chunk work {Med(endMed[chunks - 1]):F1} us");
        }

        const int rounds = 5, iters = 100;
        var medians = configs.Select(_ => new double[rounds]).ToArray();
        var sw = new Stopwatch();
        var t = new double[iters];
        for (int r = 0; r < rounds; r++)
            for (int c = 0; c < configs.Count; c++)
            {
                var warm = Stopwatch.StartNew();
                while (warm.Elapsed.TotalSeconds < 0.1) configs[c].Body();
                for (int i = 0; i < iters; i++) { sw.Restart(); configs[c].Body(); sw.Stop(); t[i] = sw.Elapsed.TotalMilliseconds; }
                Array.Sort(t);
                medians[c][r] = t[iters / 2];
            }
        Console.WriteLine($"{"configuration",-28} {"median ms",10} {"min round",10} {"max round",10}");
        for (int c = 0; c < configs.Count; c++)
        {
            var m = medians[c].OrderBy(z => z).ToArray();
            Console.WriteLine($"{configs[c].Name,-28} {m[rounds / 2],10:F4} {m[0],10:F4} {m[rounds - 1],10:F4}");
        }
    }
}
