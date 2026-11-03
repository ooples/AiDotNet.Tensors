// SCRATCH (probe branch only): 512-bit direct tile efficiency, single thread on L1-resident data, and NN GEMM thread scaling.
#if NET9_0_OR_GREATER
using System;
using System.Collections.Generic;
using System.Diagnostics;
using AiDotNet.Tensors.Engines.Simd;
using AiDotNet.Tensors.Helpers;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Parity;

[Collection("ZzSerial")]
public class ZzKernelBench
{
    private readonly ITestOutputHelper _o;
    public ZzKernelBench(ITestOutputHelper o) => _o = o;
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
    public unsafe void Time()
    {
        if (Environment.GetEnvironmentVariable("GITHUB_ACTIONS") != "true") return;
        var r = new Random(3);
        const int k = 256, reps = 2000;
        var a = new float[6 * k]; var strip = new float[k * 32]; var c = new float[6 * 32];
        for (int i = 0; i < a.Length; i++) a[i] = (float)r.NextDouble();
        for (int i = 0; i < strip.Length; i++) strip[i] = (float)r.NextDouble();
        double flops = 2.0 * 6 * 32 * k * reps;
        var t512 = Measure(() =>
        {
            fixed (float* pa = a, pb = strip, pc = c)
                for (int i = 0; i < reps; i++) SimdGemm.DirectKernel6x32Store(pa, k, pb, 32, pc, 32, k);
        });
        Line($"ZZ kern 6x32 zmm k{k} L1: {flops / (t512.Min * 1e6):F1} GFLOP/s (min {t512.Min:F3} ms, med {t512.Med:F3})  vec512 {SimdGemm.UseVector512Direct}");

        int threads0 = CpuParallelSettings.MaxDegreeOfParallelism;
        int kc0 = SimdGemm.s_stripKc; bool pack0 = SimdGemm.s_packStrip512;
        foreach (var (m, kk, n) in new[] { (128, 784, 512), (128, 512, 256) })
        {
            var A = new float[m * kk]; var B = new float[kk * n]; var C = new float[m * n];
            for (int i = 0; i < A.Length; i++) A[i] = (float)r.NextDouble();
            for (int i = 0; i < B.Length; i++) B[i] = (float)r.NextDouble();
            double f = 2.0 * m * kk * n;
            foreach (int th in new[] { 1, 2 })
            {
                CpuParallelSettings.MaxDegreeOfParallelism = th;
                var parts = new List<string>();
                var refC = new float[m * n];
                SimdGemm.s_zzPackA2D = false; SimdGemm.s_stripKc = 256;
                SimdGemm.Sgemm(A, kk, false, B, n, false, refC, m, kk, n);
                foreach (var (kcv, pk, pa, p16) in new[] { (256, true, false, false), (256, true, false, true), (256, true, true, false), (0, true, false, false) })
                {
                    SimdGemm.s_stripKc = kcv; SimdGemm.s_packStrip512 = pk; SimdGemm.s_zzPackA2D = pa; SimdGemm.s_zzPack16 = p16;
                    var tm = Measure(() => SimdGemm.Sgemm(A, kk, false, B, n, false, C, m, kk, n));
                    int diff = 0; for (int q = 0; q < C.Length; q++) if (BitConverter.SingleToInt32Bits(C[q]) != BitConverter.SingleToInt32Bits(refC[q])) diff++;
                    parts.Add($"kc{kcv}{(pk ? "" : "-nopack")}{(pa ? "-packA" : "")}{(p16 ? "-pack16" : "")} {tm.Min:F4} ({f / (tm.Min * 1e6):F0}) d{diff}");
                }
                SimdGemm.s_zzPackA2D = false; SimdGemm.s_zzPack16 = false;
                Line($"ZZ kern kcsweep {m}x{kk}x{n} t{th}: {string.Join("  ", parts)}");
            }
        }
        SimdGemm.s_stripKc = kc0; SimdGemm.s_packStrip512 = pack0;
        // dW = Xᵀ·dY through the transposed-A path, with and without AVX2 strip packing.
        foreach (var (m, kk, n) in new[] { (784, 128, 512), (512, 128, 256) })
        {
            var X = new float[kk * m]; var D = new float[kk * n]; var W = new float[m * n]; var Wref = new float[m * n];
            for (int i = 0; i < X.Length; i++) X[i] = (float)r.NextDouble();
            for (int i = 0; i < D.Length; i++) D[i] = (float)r.NextDouble();
            double f = 2.0 * m * kk * n;
            SimdGemm.s_zzPack16 = false;
            SimdGemm.Sgemm(X, m, true, D, n, false, Wref, m, kk, n);
            foreach (int th in new[] { 1, 2 })
            {
                CpuParallelSettings.MaxDegreeOfParallelism = th;
                var parts = new List<string>();
                foreach (bool p16 in new[] { false, true })
                {
                    SimdGemm.s_zzPack16 = p16;
                    var tm = Measure(() => SimdGemm.Sgemm(X, m, true, D, n, false, W, m, kk, n));
                    int diff = 0; for (int q = 0; q < W.Length; q++) if (BitConverter.SingleToInt32Bits(W[q]) != BitConverter.SingleToInt32Bits(Wref[q])) diff++;
                    parts.Add($"{(p16 ? "pack16" : "base")} {tm.Min:F4} ({f / (tm.Min * 1e6):F0}) d{diff}");
                }
                SimdGemm.s_zzPack16 = false;
                Line($"ZZ kern dW {m}x{kk}x{n} t{th}: {string.Join("  ", parts)}");
            }
        }
        CpuParallelSettings.MaxDegreeOfParallelism = threads0;
        foreach (var (m, kk, n) in new[] { (128, 784, 512), (128, 512, 256), (64, 128, 3136) })
        {
            var A = new float[m * kk]; var B = new float[kk * n]; var C = new float[m * n];
            for (int i = 0; i < A.Length; i++) A[i] = (float)r.NextDouble();
            for (int i = 0; i < B.Length; i++) B[i] = (float)r.NextDouble();
            double f = 2.0 * m * kk * n;
            var parts = new List<string>();
            foreach (int th in new[] { 1, 2, 4 })
            {
                CpuParallelSettings.MaxDegreeOfParallelism = th;
                var tm = Measure(() => SimdGemm.Sgemm(A, kk, false, B, n, false, C, m, kk, n));
                parts.Add($"t{th} {tm.Min:F4} ms {f / (tm.Min * 1e6):F0} GF/s");
            }
            CpuParallelSettings.MaxDegreeOfParallelism = threads0;
            Line($"ZZ kern sgemm {m}x{kk}x{n}: {string.Join("  ", parts)}");
        }
    }
}
#endif
