// Copyright (c) AiDotNet. All rights reserved.

using System;
using System.Diagnostics;
using System.Globalization;
using System.Linq;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Benchmarks;

/// <summary>
/// #1353: times the persistent fused basic block (one cooperative launch) against the unfused route through the
/// engine (Conv2D, BatchNorm with batch statistics, ReLU, Conv2D, BatchNorm, add, ReLU), per block, at the issue's
/// ResNet-18 stage shapes, batch 8. Both sides run resident and are timed as a batch of back-to-back blocks between two
/// device synchronizations; each side keeps its best of several rounds. Compare with torch's per-block time from
/// tools/parity (fused_basic_block_torch.py) on the same GPU.
/// </summary>
internal static class FusedBasicBlockExperiment
{
    public static int Run()
    {
        CudaBackend backend;
        try { backend = new CudaBackend(); }
        catch (Exception ex) when (ex is DllNotFoundException or TypeInitializationException or InvalidOperationException)
        {
            Console.WriteLine("Fused basic block: CUDA unavailable (" + ex.GetType().Name + "); nothing measured.");
            return 2;
        }
        using (backend)
        {
            if (!backend.IsAvailable || !backend.SupportsFusedBasicBlock(8, 64, 8, 8))
            {
                Console.WriteLine("Fused basic block: CUDA with cooperative launch is not available; nothing measured.");
                return 2;
            }
            var engine = new DirectGpuTensorEngine(new DirectGpuEngine(backend));
            Console.WriteLine("Fused basic block (training forward, batch 8) on " + backend.DeviceName);
            Console.WriteLine("shape      fused us   grid  unfused us  speedup");
            foreach (var (c, hw) in new[] { (64, 32), (128, 16), (256, 8), (512, 4) })
            {
                const int n = 8;
                var rng = new Random(c + hw);
                float[] R(int len, double scale, double shift = 0) =>
                    Enumerable.Range(0, len).Select(_ => (float)(shift + (rng.NextDouble() * 2 - 1) * scale)).ToArray();
                double he = Math.Sqrt(2.0 / (c * 9));
                float[] x = R(n * c * hw * hw, 1), w1 = R(c * c * 9, he), w2 = R(c * c * 9, he);
                float[] g1 = R(c, 0.2, 1), b1 = R(c, 0.1), g2 = R(c, 0.2, 1), b2 = R(c, 0.1);

                using var dx = backend.AllocateBuffer(x);
                using var dw1 = backend.AllocateBuffer(w1);
                using var dw2 = backend.AllocateBuffer(w2);
                using var dg1 = backend.AllocateBuffer(g1);
                using var db1 = backend.AllocateBuffer(b1);
                using var dg2 = backend.AllocateBuffer(g2);
                using var db2 = backend.AllocateBuffer(b2);
                using var dc1 = backend.AllocateBuffer(x.Length);
                using var dc2 = backend.AllocateBuffer(x.Length);
                using var dy = backend.AllocateBuffer(x.Length);
                using var dstats = backend.AllocateBuffer(4 * c);
                using var dbar = backend.AllocateBuffer(2);
                int grid = 0;
                double fused = Best(backend, () => grid = backend.FusedBasicBlockForward(dx, dw1, dg1, db1, dw2, dg2, db2,
                    dc1, dc2, dstats, dy, dbar, n, c, hw, hw, 1e-5f));

                var tx = new Tensor<float>(x, new[] { n, c, hw, hw });
                var tw1 = new Tensor<float>(w1, new[] { c, c, 3, 3 });
                var tw2 = new Tensor<float>(w2, new[] { c, c, 3, 3 });
                var tg1 = new Tensor<float>(g1, new[] { c }); var tb1 = new Tensor<float>(b1, new[] { c });
                var tg2 = new Tensor<float>(g2, new[] { c }); var tb2 = new Tensor<float>(b2, new[] { c });
                double unfused = Best(backend, () =>
                {
                    var h1 = engine.ReLU(engine.BatchNorm(engine.Conv2D(tx, tw1, 1, 1, 1), tg1, tb1, 1e-5, out _, out _));
                    var h2 = engine.BatchNorm(engine.Conv2D(h1, tw2, 1, 1, 1), tg2, tb2, 1e-5, out _, out _);
                    engine.ReLU(engine.TensorAdd(h2, tx));
                });
                Console.WriteLine(string.Format(CultureInfo.InvariantCulture, "{0,-10} {1,8:F1} {2,6} {3,11:F1} {4,7:F2}x",
                    c + "@" + hw, fused, grid, unfused, unfused / fused));
            }
            return 0;
        }
    }

    /// <summary>Microseconds per block: the best of 7 rounds of 50 back-to-back blocks between two syncs.</summary>
    private static double Best(CudaBackend backend, Action block)
    {
        for (int i = 0; i < 10; i++) block();
        backend.Synchronize();
        double best = double.MaxValue;
        for (int round = 0; round < 7; round++)
        {
            var sw = Stopwatch.StartNew();
            for (int i = 0; i < 50; i++) block();
            backend.Synchronize();
            best = Math.Min(best, sw.Elapsed.TotalMilliseconds * 1000.0 / 50);
        }
        return best;
    }
}
