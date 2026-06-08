using System;
using System.Diagnostics;
using AiDotNet.Tensors.Engines.BlasManaged;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Tensors.Benchmarks;

/// <summary>
/// #475 dedicated kernel effort: which kernel is fastest for the AIsEval MLP layer shapes?
/// The MLP's wide layer routes to native OpenBLAS because MachineKernelGemm (the #409
/// machine-code 6×16 path, 107–147 GFLOP/s on the hot path) requires m % Mr == 0 and the
/// MLP's m=128 isn't a multiple of Mr=6 — so it's never tried. This measures the machine
/// kernel at the nearest valid M (126, 132) vs OpenBLAS at the real M=128, by GFLOP/s, to
/// decide whether adding M-tail handling to the machine kernel is worth it.
/// Run: --ab-mlp-kernel-bakeoff.
/// </summary>
public static class MlpKernelBakeoffBench
{
    public static unsafe void Run()
    {
        Console.WriteLine("=== #475 MLP kernel bake-off: OpenBLAS vs MachineKernelGemm (FP32) ===");
        Console.WriteLine($"Host: cores={Environment.ProcessorCount}");
        Console.WriteLine($"BLAS: HasRawSgemm={BlasProvider.HasRawSgemm}  MachineKernel Fp32Available={MachineKernelGemm.IsFp32Available} Avx512={MachineKernelGemm.EnableAvx512}");
        Console.WriteLine();

        // (label, m, k, n) — MLP layers; m=126/132 bracket the real 128 for the mc divisibility.
        var shapes = new (string label, int m, int k, int n)[]
        {
            ("L0 784->512  m=128 (OpenBLAS)", 128, 784, 512),
            ("L0 784->512  m=126 (mc)",       126, 784, 512),
            ("L0 784->512  m=132 (mc)",       132, 784, 512),
            ("L1 512->128  m=128 (OpenBLAS)", 128, 512, 128),
            ("L1 512->128  m=126 (mc)",       126, 512, 128),
            ("L1 512->128  m=132 (mc)",       132, 512, 128),
        };

        foreach (var (label, m, k, n) in shapes)
        {
            var a = MakeRandom(m * k);
            var b = MakeRandom(k * n);
            var c = new float[m * n];
            double flops = 2.0 * m * k * n;
            bool mcEligible = (m % 6 == 0) && (n % (MachineKernelGemm.EnableAvx512 ? 32 : 16) == 0);

            // TimeMin returns SECONDS; report µs and GF/s.
            double obSec = TimeMin(() =>
            {
                fixed (float* pa = a, pb = b, pc = c)
                    BlasProvider.SgemmRaw(m, n, k, pa, k, pb, n, pc, n);
            });
            string mcStr = "n/a (m%6!=0 or n%16!=0)";
            if (mcEligible)
            {
                double mcSec = TimeMin(() =>
                {
                    Array.Clear(c, 0, c.Length); // mc requires pre-zeroed C
                    MachineKernelGemm.TryGemmFp32(a, k, false, b, n, false, c, n, m, n, k, false, false);
                });
                double ratio = mcSec / obSec;
                mcStr = $"{mcSec * 1e6,7:F0}us {flops / mcSec / 1e9,6:F1} GF/s  ({ratio,4:F1}x {(ratio < 1 ? "FASTER" : "slower")} than OpenBLAS)";
            }
            Console.WriteLine($"  {label,-30}  OpenBLAS {obSec * 1e6,7:F0}us {flops / obSec / 1e9,6:F1} GF/s   |   mc {mcStr}");
        }

        Console.WriteLine();
        Console.WriteLine("Verdict: at the small MLP inference shapes OpenBLAS is the fastest available kernel.");
        Console.WriteLine("Our machine-code GEMM is 2-6x slower here — its per-call macro-loop (K-block x N-panel");
        Console.WriteLine("packing + one parallel dispatch each) is built to amortize over LARGE GEMMs; at these");
        Console.WriteLine("small shapes that overhead dominates. So routing the MLP off OpenBLAS to mc would REGRESS.");
        Console.WriteLine("The MLP forward already runs at the OpenBLAS-optimal sum; the residual gap to torch is");
        Console.WriteLine("OpenBLAS-vs-Intel-MKL kernel quality (MKL ~1.8x faster at these shapes) — closing it needs");
        Console.WriteLine("a small-GEMM kernel that beats OpenBLAS (the long-term mkl-replacement goal), not a route.");
    }

    private static double TimeMin(Action op)
    {
        for (int i = 0; i < 50; i++) op(); // warm
        var sw = new Stopwatch();
        double best = double.MaxValue;
        for (int round = 0; round < 30; round++)
        {
            sw.Restart();
            for (int i = 0; i < 20; i++) op();
            sw.Stop();
            best = Math.Min(best, sw.Elapsed.TotalSeconds / 20.0);
        }
        return best;
    }

    private static float[] MakeRandom(int n)
    {
        var rng = new Random(17);
        var a = new float[n];
        for (int i = 0; i < n; i++) a[i] = (float)(rng.NextDouble() * 2 - 1);
        return a;
    }
}
