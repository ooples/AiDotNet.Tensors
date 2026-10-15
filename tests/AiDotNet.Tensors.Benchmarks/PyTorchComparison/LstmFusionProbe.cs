using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Simd;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Benchmarks.PyTorchComparison;

/// <summary>
/// Issue #436 dev probe: the AIsEval LSTM ([128, 32, 32] to hidden 64, last step) and its input-projection GEMM,
/// interleaved round by round. AIDOTNET_LSTM_PARALLEL_MINROWS (read at startup) sets the rows per recurrence chunk.
/// Run: <c>--aiseval-lstm-probe</c>.
/// </summary>
internal static class LstmFusionProbe
{
    public static void Run()
    {
        const int batch = 128, seq = 32, inF = 32, hidden = 64;
        var rng = new Random(436);
        Tensor<float> Rand(float s, params int[] shape) =>
            new(Enumerable.Range(0, shape.Aggregate(1, (a, b) => a * b)).Select(_ => (float)(rng.NextDouble() * 2 - 1) * s).ToArray(), shape);
        var x = Rand(1f, batch, seq, inF);
        var wIh = Rand(0.125f, 4 * hidden, inF);
        var wHh = Rand(0.125f, 4 * hidden, hidden);
        var engine = new CpuEngine();
        int savedDop = CpuParallelSettings.MaxDegreeOfParallelism;
        Console.WriteLine($"AIDOTNET_LSTM_PARALLEL_MINROWS={Environment.GetEnvironmentVariable("AIDOTNET_LSTM_PARALLEL_MINROWS") ?? "(default 8)"}");

        var configs = new List<(string Name, Action Body)>();
        foreach (int dop in new[] { 16, 32, 64, Environment.ProcessorCount })
        {
            int d = dop;
            configs.Add(($"LstmSequenceForward dop={d}", () =>
            {
                CpuParallelSettings.MaxDegreeOfParallelism = d;
                try { engine.LstmSequenceForward(x, null, null, wIh, wHh, null, null).Dispose(); }
                finally { CpuParallelSettings.MaxDegreeOfParallelism = savedDop; }
            }));
        }
        var xa = x.ToArray();
        var wa = wIh.ToArray();
        var wx = new float[batch * seq * 4 * hidden];
        configs.Add(("input projection Sgemm 4096x32x256 (parallel)", () =>
            SimdGemm.Sgemm(xa, inF, false, wa, inF, true, wx, batch * seq, inF, 4 * hidden)));
        var wT = new float[inF * 4 * hidden];
        var chunkWx = new float[8 * seq * 4 * hidden];
        configs.Add(("input projection of one 8-row chunk (serial)", () =>
            SimdGemm.SgemmSequential(xa.AsSpan(0, 8 * seq * inF), wT, chunkWx, 8 * seq, inF, 4 * hidden)));
        var h = new float[8 * hidden];
        var whT = new float[hidden * 4 * hidden];
        var g = new float[8 * 4 * hidden];
        configs.Add(("32 steps of 8x64x256 recurrent GEMM (serial)", () =>
        {
            for (int t = 0; t < seq; t++) SimdGemm.SgemmSequential(h, whT, g, 8, hidden, 4 * hidden);
        }));

        const int rounds = 5, iters = 200;
        var medians = configs.Select(_ => new double[rounds]).ToArray();
        var sw = new Stopwatch();
        var times = new double[iters];
        for (int r = 0; r < rounds; r++)
            for (int c = 0; c < configs.Count; c++)
            {
                var warm = Stopwatch.StartNew();
                while (warm.Elapsed.TotalSeconds < 0.1) configs[c].Body();
                for (int i = 0; i < iters; i++) { sw.Restart(); configs[c].Body(); sw.Stop(); times[i] = sw.Elapsed.TotalMilliseconds; }
                Array.Sort(times);
                medians[c][r] = times[iters / 2];
            }
        Console.WriteLine($"{"configuration",-48} {"median ms",10} {"min round",10} {"max round",10}");
        for (int c = 0; c < configs.Count; c++)
        {
            var m = medians[c].OrderBy(z => z).ToArray();
            Console.WriteLine($"{configs[c].Name,-48} {m[rounds / 2],10:F4} {m[0],10:F4} {m[rounds - 1],10:F4}");
        }
    }
}
