using System;
using System.Diagnostics;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.BlasManaged;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Tensors.Benchmarks;

/// <summary>
/// Sub-S (#409) Phase S.1 — microkernel-only GFLOPS harness.
///
/// <para>
/// Calls each BlasManaged microkernel (<see cref="Avx2Fp32_8x8"/>,
/// <see cref="Avx2Fp64_4x8"/>, <see cref="Avx512Fp32_16x16"/>,
/// <see cref="Avx512Fp64_8x16"/>) directly in a hot loop — <b>no strategy, no
/// dispatcher, no autotune, no packing cost</b> — and reports achieved GFLOPS.
/// </para>
///
/// <para>
/// The "% of peak" denominator is a <b>measured, register-only FMA ceiling</b>
/// for the same vector width on the same machine (a tight loop of independent
/// dependent-FMA chains that saturates the FMA issue ports), rather than a
/// hard-coded clock × flops/cycle figure that would be wrong under turbo / on a
/// different microarchitecture. That measured ceiling is the practical
/// "theoretical maximum" the issue's acceptance ratio (≥0.6, target ≥0.7) is
/// taken against — a microkernel cannot exceed the rate at which the cores can
/// retire FMAs.
/// </para>
///
/// <para>Run via <c>dotnet run -c Release -- --microkernel-gflops</c>.</para>
/// </summary>
public static class MicrokernelGflopsBench
{
    // Kc per kernel invocation — matches the issue's "Kc=256" baseline point and
    // is representative of the L1-resident K-block the BlasManaged loop feeds.
    private const int Kc = 256;

    // Sink prevents the JIT from eliminating the calibration FMA chains as dead
    // code (their results are folded into this and printed at the end of Run).
    private static double s_sink;

    public static void Run()
    {
        Console.WriteLine("=== Microkernel GFLOPS (Sub-S #409, Phase S.1) ===");
        Console.WriteLine($"ProcessorCount: {Environment.ProcessorCount}");
        Console.WriteLine(TensorPrimitivesCore.GetHardwareAccelerationInfo());
        Console.WriteLine($"AVX2={Avx2.IsSupported} FMA={Fma.IsSupported} AVX-512F={Avx512F.IsSupported}");
        Console.WriteLine("Single-thread; Kc=" + Kc + ". '% peak' is vs a measured register-only FMA ceiling.");
        Console.WriteLine();

        // ── AVX2 FP32 8×8 (the primary kernel — biggest contributor to the gap)
        if (Avx2Fp32_8x8.IsSupported)
        {
            double peak = MeasureAvx2Fp32PeakGflops();
            BenchAvx2Fp32_8x8(peak);
        }
        else Console.WriteLine("Avx2Fp32_8x8: not supported on this CPU.\n");

        // ── AVX2 FP32 6×16 (#409 S.3 — higher arithmetic intensity than 8×8)
        if (Avx2Fp32_6x16.IsSupported)
        {
            double peak = MeasureAvx2Fp32PeakGflops();
            BenchAvx2Fp32_6x16(peak);
        }
        else Console.WriteLine("Avx2Fp32_6x16: not supported on this CPU.\n");

        // ── AVX2 FP64 4×8
        if (Avx2Fp64_4x8.IsSupported)
        {
            double peak = MeasureAvx2Fp64PeakGflops();
            BenchAvx2Fp64_4x8(peak);
        }
        else Console.WriteLine("Avx2Fp64_4x8: not supported on this CPU.\n");

        // ── AVX-512 FP32 16×16
        if (Avx512Fp32_16x16.IsSupported)
        {
            double peak = MeasureAvx512Fp32PeakGflops();
            BenchAvx512Fp32_16x16(peak);
        }
        else Console.WriteLine("Avx512Fp32_16x16: not supported on this CPU.\n");

        // ── AVX-512 FP64 8×16
        if (Avx512Fp64_8x16.IsSupported)
        {
            double peak = MeasureAvx512Fp64PeakGflops();
            BenchAvx512Fp64_8x16(peak);
        }
        else Console.WriteLine("Avx512Fp64_8x16: not supported on this CPU.\n");

        Console.WriteLine($"(sink={s_sink:G4})");
    }

    // ─────────────────────────────────────────────────────────────────────
    // Kernel benchmarks. flops/call = 2 · Mr · Nr · Kc (one mul + one add per
    // MAC). Buffers are L1-resident so we measure compute, not bandwidth.
    // ─────────────────────────────────────────────────────────────────────

    private static void BenchAvx2Fp32_8x8(double peakGflops)
    {
        const int Mr = Avx2Fp32_8x8.Mr, Nr = Avx2Fp32_8x8.Nr;
        var packedA = MakeRandomF(Kc * Mr);
        var packedB = MakeRandomF(Kc * Nr);
        var c = new float[Mr * Nr];

        Action call = () => Avx2Fp32_8x8.Run(packedA, packedB, c, Nr, Kc);
        double gflops = TimeKernel(call, 2.0 * Mr * Nr * Kc, warmup: 2_000, iters: 2_000_000);
        Report("Avx2Fp32_8x8   ", Mr, Nr, gflops, peakGflops);
    }

    private static void BenchAvx2Fp32_6x16(double peakGflops)
    {
        const int Mr = Avx2Fp32_6x16.Mr, Nr = Avx2Fp32_6x16.Nr;
        var packedA = MakeRandomF(Kc * Mr);
        var packedB = MakeRandomF(Kc * Nr);
        var c = new float[Mr * Nr];

        Action call = () => Avx2Fp32_6x16.Run(packedA, packedB, c, Nr, Kc);
        double gflops = TimeKernel(call, 2.0 * Mr * Nr * Kc, warmup: 2_000, iters: 2_000_000);
        Report("Avx2Fp32_6x16  ", Mr, Nr, gflops, peakGflops);
    }

    private static void BenchAvx2Fp64_4x8(double peakGflops)
    {
        const int Mr = Avx2Fp64_4x8.Mr, Nr = Avx2Fp64_4x8.Nr;
        var packedA = MakeRandomD(Kc * Mr);
        var packedB = MakeRandomD(Kc * Nr);
        var c = new double[Mr * Nr];

        Action call = () => Avx2Fp64_4x8.Run(packedA, packedB, c, Nr, Kc);
        double gflops = TimeKernel(call, 2.0 * Mr * Nr * Kc, warmup: 2_000, iters: 2_000_000);
        Report("Avx2Fp64_4x8   ", Mr, Nr, gflops, peakGflops);
    }

    private static void BenchAvx512Fp32_16x16(double peakGflops)
    {
        const int Mr = Avx512Fp32_16x16.Mr, Nr = Avx512Fp32_16x16.Nr;
        var packedA = MakeRandomF(Kc * Mr);
        var packedB = MakeRandomF(Kc * Nr);
        var c = new float[Mr * Nr];

        Action call = () => Avx512Fp32_16x16.Run(packedA, packedB, c, Nr, Kc);
        double gflops = TimeKernel(call, 2.0 * Mr * Nr * Kc, warmup: 2_000, iters: 1_000_000);
        Report("Avx512Fp32_16x16", Mr, Nr, gflops, peakGflops);
    }

    private static void BenchAvx512Fp64_8x16(double peakGflops)
    {
        const int Mr = Avx512Fp64_8x16.Mr, Nr = Avx512Fp64_8x16.Nr;
        var packedA = MakeRandomD(Kc * Mr);
        var packedB = MakeRandomD(Kc * Nr);
        var c = new double[Mr * Nr];

        Action call = () => Avx512Fp64_8x16.Run(packedA, packedB, c, Nr, Kc);
        double gflops = TimeKernel(call, 2.0 * Mr * Nr * Kc, warmup: 2_000, iters: 1_000_000);
        Report("Avx512Fp64_8x16", Mr, Nr, gflops, peakGflops);
    }

    private static double TimeKernel(Action call, double flopsPerCall, int warmup, int iters)
    {
        for (int i = 0; i < warmup; i++) call();
        var sw = Stopwatch.StartNew();
        for (int i = 0; i < iters; i++) call();
        sw.Stop();
        double seconds = sw.Elapsed.TotalSeconds;
        return flopsPerCall * iters / seconds / 1e9;
    }

    private static void Report(string name, int mr, int nr, double gflops, double peakGflops)
    {
        double ratio = peakGflops > 0 ? gflops / peakGflops : 0;
        Console.WriteLine($"{name} {mr}x{nr}: {gflops,8:F1} GFLOPS  | peak {peakGflops,8:F1} GFLOPS  | {ratio,5:P0} of peak");
    }

    // ─────────────────────────────────────────────────────────────────────
    // Register-only FMA ceilings. Independent dependent-FMA chains hide the
    // ~4-5 cycle FMA latency across the 2 FMA issue ports, so the loop retires
    // FMAs at the port-bound maximum — the practical peak for this width/clock
    // on this machine. flops = iters · chains · lanes · 2.
    //
    // CHAIN COUNT MATTERS: the RyuJIT enregisters ~8 YMM accumulators before it
    // starts spilling. FmaCeilingProbe measured FP64 at 4ch=22, 8ch=46, 10ch=19,
    // 12ch=22 GFLOPS — i.e. 8 chains saturates the ports, but 10+ SPILL and crater
    // throughput. An earlier 12-chain version of this calibration therefore
    // UNDER-measured the ceiling ~2× (reported ~22 instead of ~46) and made the
    // microkernels look "at ceiling" when they actually have ~2× headroom. Use 8.
    // ─────────────────────────────────────────────────────────────────────

    private const int PeakChains = 8;
    private const long PeakIters = 80_000_000;

    private static double MeasureAvx2Fp32PeakGflops()
    {
        var a = Vector256.Create(1.0000001f);
        var b = Vector256.Create(0.9999999f);
        Vector256<float> c0 = Vector256<float>.Zero, c1 = c0, c2 = c0, c3 = c0,
            c4 = c0, c5 = c0, c6 = c0, c7 = c0;
        // warmup
        for (long i = 0; i < 1_000_000; i++) c0 = Fma.MultiplyAdd(a, b, c0);
        var sw = Stopwatch.StartNew();
        for (long i = 0; i < PeakIters; i++)
        {
            c0 = Fma.MultiplyAdd(a, b, c0); c1 = Fma.MultiplyAdd(a, b, c1);
            c2 = Fma.MultiplyAdd(a, b, c2); c3 = Fma.MultiplyAdd(a, b, c3);
            c4 = Fma.MultiplyAdd(a, b, c4); c5 = Fma.MultiplyAdd(a, b, c5);
            c6 = Fma.MultiplyAdd(a, b, c6); c7 = Fma.MultiplyAdd(a, b, c7);
        }
        sw.Stop();
        var sum = Avx.Add(Avx.Add(Avx.Add(c0, c1), Avx.Add(c2, c3)), Avx.Add(Avx.Add(c4, c5), Avx.Add(c6, c7)));
        s_sink += sum.GetElement(0);
        return 2.0 * PeakChains * 8 * PeakIters / sw.Elapsed.TotalSeconds / 1e9;
    }

    private static double MeasureAvx2Fp64PeakGflops()
    {
        var a = Vector256.Create(1.0000001);
        var b = Vector256.Create(0.9999999);
        Vector256<double> c0 = Vector256<double>.Zero, c1 = c0, c2 = c0, c3 = c0,
            c4 = c0, c5 = c0, c6 = c0, c7 = c0;
        for (long i = 0; i < 1_000_000; i++) c0 = Fma.MultiplyAdd(a, b, c0);
        var sw = Stopwatch.StartNew();
        for (long i = 0; i < PeakIters; i++)
        {
            c0 = Fma.MultiplyAdd(a, b, c0); c1 = Fma.MultiplyAdd(a, b, c1);
            c2 = Fma.MultiplyAdd(a, b, c2); c3 = Fma.MultiplyAdd(a, b, c3);
            c4 = Fma.MultiplyAdd(a, b, c4); c5 = Fma.MultiplyAdd(a, b, c5);
            c6 = Fma.MultiplyAdd(a, b, c6); c7 = Fma.MultiplyAdd(a, b, c7);
        }
        sw.Stop();
        var sum = Avx.Add(Avx.Add(Avx.Add(c0, c1), Avx.Add(c2, c3)), Avx.Add(Avx.Add(c4, c5), Avx.Add(c6, c7)));
        s_sink += sum.GetElement(0);
        return 2.0 * PeakChains * 4 * PeakIters / sw.Elapsed.TotalSeconds / 1e9;
    }

    private static double MeasureAvx512Fp32PeakGflops()
    {
        var a = Vector512.Create(1.0000001f);
        var b = Vector512.Create(0.9999999f);
        Vector512<float> c0 = Vector512<float>.Zero, c1 = c0, c2 = c0, c3 = c0,
            c4 = c0, c5 = c0, c6 = c0, c7 = c0;
        for (long i = 0; i < 1_000_000; i++) c0 = Avx512F.FusedMultiplyAdd(a, b, c0);
        var sw = Stopwatch.StartNew();
        for (long i = 0; i < PeakIters; i++)
        {
            c0 = Avx512F.FusedMultiplyAdd(a, b, c0); c1 = Avx512F.FusedMultiplyAdd(a, b, c1);
            c2 = Avx512F.FusedMultiplyAdd(a, b, c2); c3 = Avx512F.FusedMultiplyAdd(a, b, c3);
            c4 = Avx512F.FusedMultiplyAdd(a, b, c4); c5 = Avx512F.FusedMultiplyAdd(a, b, c5);
            c6 = Avx512F.FusedMultiplyAdd(a, b, c6); c7 = Avx512F.FusedMultiplyAdd(a, b, c7);
        }
        sw.Stop();
        var sum = Avx512F.Add(Avx512F.Add(Avx512F.Add(c0, c1), Avx512F.Add(c2, c3)), Avx512F.Add(Avx512F.Add(c4, c5), Avx512F.Add(c6, c7)));
        s_sink += sum.GetElement(0);
        return 2.0 * PeakChains * 16 * PeakIters / sw.Elapsed.TotalSeconds / 1e9;
    }

    private static double MeasureAvx512Fp64PeakGflops()
    {
        var a = Vector512.Create(1.0000001);
        var b = Vector512.Create(0.9999999);
        Vector512<double> c0 = Vector512<double>.Zero, c1 = c0, c2 = c0, c3 = c0,
            c4 = c0, c5 = c0, c6 = c0, c7 = c0;
        for (long i = 0; i < 1_000_000; i++) c0 = Avx512F.FusedMultiplyAdd(a, b, c0);
        var sw = Stopwatch.StartNew();
        for (long i = 0; i < PeakIters; i++)
        {
            c0 = Avx512F.FusedMultiplyAdd(a, b, c0); c1 = Avx512F.FusedMultiplyAdd(a, b, c1);
            c2 = Avx512F.FusedMultiplyAdd(a, b, c2); c3 = Avx512F.FusedMultiplyAdd(a, b, c3);
            c4 = Avx512F.FusedMultiplyAdd(a, b, c4); c5 = Avx512F.FusedMultiplyAdd(a, b, c5);
            c6 = Avx512F.FusedMultiplyAdd(a, b, c6); c7 = Avx512F.FusedMultiplyAdd(a, b, c7);
        }
        sw.Stop();
        var sum = Avx512F.Add(Avx512F.Add(Avx512F.Add(c0, c1), Avx512F.Add(c2, c3)), Avx512F.Add(Avx512F.Add(c4, c5), Avx512F.Add(c6, c7)));
        s_sink += sum.GetElement(0);
        return 2.0 * PeakChains * 8 * PeakIters / sw.Elapsed.TotalSeconds / 1e9;
    }

    private static float[] MakeRandomF(int n)
    {
        var rng = RandomHelper.CreateSeededRandom(42);
        var a = new float[n];
        for (int i = 0; i < n; i++) a[i] = (float)(rng.NextDouble() * 2 - 1);
        return a;
    }

    private static double[] MakeRandomD(int n)
    {
        var rng = RandomHelper.CreateSeededRandom(42);
        var a = new double[n];
        for (int i = 0; i < n; i++) a[i] = rng.NextDouble() * 2 - 1;
        return a;
    }
}
