using System;
using System.Diagnostics;
using AiDotNet.Tensors.Engines.Simd;
using AiDotNet.Tensors.Helpers;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// #475 probe: at the AIsEval MLP per-layer GEMM shapes (Dense 784→512→128→10,
/// bs=128), measure the managed variants MlpForward could route to — the current
/// BlasManaged-dispatch <c>Sgemm</c> vs the direct-serial <c>SgemmSequential</c> —
/// plus native OpenBLAS for reference. The LSTM work (#503) found SgemmSequential
/// beats the dispatch/pack path for small-K shapes; this checks whether the MLP
/// small layers (512→128, 128→10) are similarly misrouted. Env-gated, run on
/// dedicated hardware: AIDOTNET_RUN_PERF_GATES=1.
/// </summary>
[Trait("Category", "Perf")]
public class MlpGemmRoutingProbe
{
    private readonly ITestOutputHelper _out;
    public MlpGemmRoutingProbe(ITestOutputHelper o) => _out = o;

    [Fact]
    public unsafe void Probe_PerLayerGemmVariants()
    {
        if (Environment.GetEnvironmentVariable("AIDOTNET_RUN_PERF_GATES") != "1")
        {
            _out.WriteLine("Skip: set AIDOTNET_RUN_PERF_GATES=1 to run the MLP GEMM routing probe.");
            return;
        }

        bool hasBlas = BlasProvider.HasRawSgemm;
        _out.WriteLine($"HasRawSgemm(OpenBLAS)={hasBlas}");
        _out.WriteLine($"{"layer (MxKxN)",-20}{"Sgemm GF/s",14}{"Seq GF/s",12}{"OpenBLAS GF/s",16}{"winner",12}");

        // AIsEval MLP layers at bs=128: (M, K, N)
        var layers = new (string name, int M, int K, int N)[]
        {
            ("L0 128x784x512", 128, 784, 512),
            ("L1 128x512x128", 128, 512, 128),
            ("L2 128x128x10",  128, 128, 10),
        };

        var rng = RandomHelper.CreateSeededRandom(475);
        foreach (var (name, M, K, N) in layers)
        {
            var a = RandF(M * K, rng);
            var b = RandF(K * N, rng);
            var c = new float[M * N];
            double flop = 2.0 * M * N * K;

            double sgemmMs = MinMs(2000, () => SimdGemm.Sgemm(a, b, c, M, K, N));
            double seqMs   = MinMs(2000, () => SimdGemm.SgemmSequential(a, b, c, M, K, N));

            double blasMs = double.NaN;
            if (hasBlas)
            {
                var cb = new float[M * N];
                fixed (float* pa = a, pb = b, pc = cb)
                {
                    float* la = pa, lb = pb, lc = pc;
                    blasMs = MinMs(2000, () => BlasProvider.SgemmRaw(M, N, K, la, K, lb, N, lc, N));
                }
            }

            double sgemmGf = flop / (sgemmMs * 1e6);
            double seqGf = flop / (seqMs * 1e6);
            double blasGf = flop / (blasMs * 1e6);

            string winner = sgemmMs <= seqMs ? "Sgemm" : "Sequential";
            if (hasBlas && blasMs < Math.Min(sgemmMs, seqMs)) winner = "OpenBLAS";

            string blasCol = double.IsNaN(blasGf) ? "--" : blasGf.ToString("F1");
            _out.WriteLine($"{name,-20}{sgemmGf,14:F1}{seqGf,12:F1}{blasCol,16}{winner,12}");
        }
    }

    private static double MinMs(int iters, Action f)
    {
        for (int i = 0; i < 50; i++) f();
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

    private static float[] RandF(int len, Random rng)
    {
        var a = new float[len];
        for (int i = 0; i < len; i++) a[i] = (float)(rng.NextDouble() * 2 - 1);
        return a;
    }
}
