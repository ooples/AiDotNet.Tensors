// SCRATCH (probe branch only): the h2h MLP's GEMMs (batch 128) against torch, same host, in-process interleaved.
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Parity;

public class ZzMlpGemmBench
{
    private readonly ITestOutputHelper _o;
    public ZzMlpGemmBench(ITestOutputHelper o) => _o = o;

    private static float[] Rand(Random r, int n)
    {
        var d = new float[n];
        for (int i = 0; i < n; i++) d[i] = (float)(r.NextDouble() * 2 - 1);
        return d;
    }

    private void Line(string s) { _o.WriteLine(s); Console.WriteLine(s); }

    private static (double Min, double Med) Measure(Action run)
    {
        var sw0 = Stopwatch.StartNew(); while (sw0.ElapsedMilliseconds < 250) run();
        var t = new List<double>();
        for (int round = 0; round < 25; round++)
        {
            double best = double.MaxValue;
            for (int i = 0; i < 5; i++) { var sw = Stopwatch.StartNew(); run(); best = Math.Min(best, sw.Elapsed.TotalMilliseconds); }
            t.Add(best);
        }
        t.Sort();
        return (t[0], t[t.Count / 2]);
    }

    [Fact]
    public void Time()
    {
        if (Environment.GetEnvironmentVariable("GITHUB_ACTIONS") != "true" && Environment.GetEnvironmentVariable("ZZ_CONV") != "1") return;
        int threads = int.TryParse(Environment.GetEnvironmentVariable("ZZ_THREADS"), out int t) ? t : Environment.ProcessorCount;
        CpuParallelSettings.MaxDegreeOfParallelism = threads;
        Line($"ZM threads {threads}");
        ZzDeepConvBenchTorch.Run(_o, "tools/parity/zz_mlp_torch.py", threads);
        var engine = new CpuEngine();
        var rng = new Random(7);
        foreach (var (m, k, n) in new[] { (128, 784, 512), (128, 512, 256), (128, 256, 10), (64, 3136, 128), (64, 128, 10) })
        {
            float[] x = Rand(rng, m * k), w = Rand(rng, k * n), dy = Rand(rng, m * n), b = Rand(rng, n);
            float[] dw = new float[k * n], dx = new float[m * k];
            var xt = new Tensor<float>(x, new[] { m, k }); var wt = new Tensor<float>(w, new[] { k, n }); var bt = new Tensor<float>(b, new[] { n });
            var f = Measure(() => engine.FusedLinear(xt, wt, bt, FusedActivationType.ReLU));
            var gw = Measure(() => BackwardFunctions<float>.BackwardGemmWeightGradient(x, dy, dw, m, k, n));
            var gx = Measure(() => BackwardFunctions<float>.BackwardGemmInputGradient(dy, w, dx, m, k, n));
            Line($"ZM m{m} {k}->{n} fwd {f.Min:F4}/{f.Med:F4}  dW {gw.Min:F4}/{gw.Med:F4}  dX {gx.Min:F4}/{gx.Med:F4}");
        }
        foreach (var (mm, k, n) in new[] { (128, 784, 512), (128, 512, 256), (128, 256, 10), (64, 3136, 128), (64, 128, 10) })
        {
            float[] x = Rand(rng, mm * k), w = Rand(rng, k * n), o = new float[mm * n];
            var cached = Measure(() => AiDotNet.Tensors.Engines.Simd.SimdGemm.SgemmWithCachedB(x, w, o, mm, k, n));
            var blas = Measure(() => AiDotNet.Tensors.Helpers.BlasProvider.Gemm(mm, n, k, x, 0, k, w, 0, n, o, 0, n));
            var simd = Measure(() => { o.AsSpan().Clear(); AiDotNet.Tensors.Engines.Simd.SimdGemm.Sgemm(x.AsSpan(0, mm * k), k, false, w.AsSpan(0, k * n), n, false, o.AsSpan(0, mm * n), mm, k, n); });
            float[] dyR = Rand(rng, mm * n), dwR = new float[k * n];
            var dwCur = Measure(() => BackwardFunctions<float>.BackwardGemmWeightGradient(x, dyR, dwR, mm, k, n));
            var dwSimd = Measure(() => AiDotNet.Tensors.Engines.Simd.SimdGemm.Sgemm(x.AsSpan(0, mm * k), k, true, dyR.AsSpan(0, mm * n), n, false, dwR.AsSpan(0, k * n), k, mm, n));
            var dwAuto = Measure(() => AiDotNet.Tensors.Engines.BlasManaged.BlasManaged.Gemm<float>(x, k, true, dyR, n, false, dwR.AsSpan(0, k * n), n, k, n, mm));
            {
                var o2 = new AiDotNet.Tensors.Engines.BlasManaged.BlasOptions<float>();
                string strat = "";
                foreach (var pm in new[] { AiDotNet.Tensors.Engines.BlasManaged.PackingMode.ForcePackBoth, AiDotNet.Tensors.Engines.BlasManaged.PackingMode.ForcePackAOnly, AiDotNet.Tensors.Engines.BlasManaged.PackingMode.ForceStreaming })
                {
                    var pmc = pm;
                    var r = Measure(() => AiDotNet.Tensors.Engines.BlasManaged.BlasManaged.Gemm<float>(x, k, true, dyR, n, false, dwR.AsSpan(0, k * n), n, k, n, mm,
                        new AiDotNet.Tensors.Engines.BlasManaged.BlasOptions<float> { PackingMode = pmc }));
                    strat += $"  {pm} {r.Min:F4}/{r.Med:F4}";
                }
                Line($"ZM dWstrat m{mm} {k}->{n} sel {AiDotNet.Tensors.Engines.BlasManaged.Dispatcher.SelectStrategy<float>(k, n, mm, true, false, in o2)}{strat}");
            }
            Line($"ZM dWroute m{mm} {k}->{n} current {dwCur.Min:F4}/{dwCur.Med:F4}  sgemmT {dwSimd.Min:F4}/{dwSimd.Med:F4}  blasAuto {dwAuto.Min:F4}/{dwAuto.Med:F4}");
            float[] dxR = new float[mm * k];
            var dxCur = Measure(() => BackwardFunctions<float>.BackwardGemmInputGradient(dyR, w, dxR, mm, k, n));
            var dxDis = Measure(() => AiDotNet.Tensors.Engines.BlasManaged.BlasManaged.Gemm<float>(dyR, n, false, w, n, true, dxR.AsSpan(0, mm * k), k, mm, k, n,
                new AiDotNet.Tensors.Engines.BlasManaged.BlasOptions<float> { PackingMode = AiDotNet.Tensors.Engines.BlasManaged.PackingMode.DisableAutotune }));
            var dxAuto = Measure(() => AiDotNet.Tensors.Engines.BlasManaged.BlasManaged.Gemm<float>(dyR, n, false, w, n, true, dxR.AsSpan(0, mm * k), k, mm, k, n));
            var wT = new float[n * k];
            for (int r = 0; r < k; r++) for (int cc = 0; cc < n; cc++) wT[cc * k + r] = w[r * n + cc];
            var dxPreT = Measure(() => AiDotNet.Tensors.Engines.BlasManaged.BlasManaged.Gemm<float>(dyR, n, false, wT, k, false, dxR.AsSpan(0, mm * k), k, mm, k, n,
                new AiDotNet.Tensors.Engines.BlasManaged.BlasOptions<float> { PackingMode = AiDotNet.Tensors.Engines.BlasManaged.PackingMode.DisableAutotune }));
            var wT2 = new float[n * k];
            var dxTn = Measure(() =>
            {
                AiDotNet.Tensors.Engines.Simd.SimdTranspose.Transpose(w.AsSpan(0, k * n), wT2.AsSpan(0, n * k), k, n);
                AiDotNet.Tensors.Engines.Simd.SimdGemm.Sgemm(dyR.AsSpan(0, mm * n), n, false, wT2.AsSpan(0, n * k), k, false, dxR.AsSpan(0, mm * k), mm, n, k);
            });
            var dxNn = Measure(() => AiDotNet.Tensors.Engines.Simd.SimdGemm.Sgemm(dyR.AsSpan(0, mm * n), n, false, wT2.AsSpan(0, n * k), k, false, dxR.AsSpan(0, mm * k), mm, n, k));
            Line($"ZM dXnn m{mm} {k}->{n} transpose+nn {dxTn.Min:F4}/{dxTn.Med:F4}  nn-only {dxNn.Min:F4}/{dxNn.Med:F4}  vec512 {AiDotNet.Tensors.Engines.Simd.SimdGemm.UseVector512Direct}");
            Line($"ZM dXroute m{mm} {k}->{n} current {dxCur.Min:F4}/{dxCur.Med:F4}  blasDis {dxDis.Min:F4}/{dxDis.Med:F4}  blasAuto {dxAuto.Min:F4}/{dxAuto.Med:F4}  blasWT {dxPreT.Min:F4}/{dxPreT.Med:F4}");
            Line($"ZM route m{mm} {k}->{n} cachedB {cached.Min:F4}/{cached.Med:F4}  blas {blas.Min:F4}/{blas.Med:F4}  sgemm {simd.Min:F4}/{simd.Med:F4}");
        }
        ZzDeepConvBenchTorch.Run(_o, "tools/parity/zz_mlp_torch.py", threads);
    }
}

internal static class ZzDeepConvBenchTorch
{
    public static void Run(ITestOutputHelper o, string script, int threads)
    {
        var py = Environment.GetEnvironmentVariable("PARITY_PYTHON");
        if (string.IsNullOrEmpty(py)) return;
        var dir = AppContext.BaseDirectory;
        while (dir != null && !System.IO.File.Exists(System.IO.Path.Combine(dir, script))) dir = System.IO.Path.GetDirectoryName(dir);
        if (dir == null) return;
        var psi = new ProcessStartInfo(py, System.IO.Path.Combine(dir, script) + " " + threads) { RedirectStandardOutput = true, UseShellExecute = false };
        using var proc = Process.Start(psi)!;
        string? l;
        while ((l = proc.StandardOutput.ReadLine()) != null) { o.WriteLine(l); Console.WriteLine(l); }
        proc.WaitForExit();
    }
}
