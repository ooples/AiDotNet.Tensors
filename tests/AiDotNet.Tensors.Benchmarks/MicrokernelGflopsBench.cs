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

    // #378 AVX-512-BF16 Phase 1: verify the emitted VDPBF16PS machine code. There is no
    // Avx512Bf16 .NET intrinsic, so the instruction is reached only via raw EVEX bytes; run
    // this UNDER INTEL SDE on a host without AVX-512-BF16 (SDE emulates VDPBF16PS — natively
    // it would fault #UD). Returns false on mismatch / no executable memory.
    public static bool VerifyVdpbf16()
    {
        Console.WriteLine("=== VDPBF16PS machine-code probe (EVEX encoding + BF16 dot-product) ===");
        var rng = new Random(123);
        var a = new ushort[16];
        var b = new ushort[16];
        for (int i = 0; i < 16; i++)
        {
            a[i] = FloatToBf16((float)(rng.NextDouble() * 2 - 1));
            b[i] = FloatToBf16((float)(rng.NextDouble() * 2 - 1));
        }
        var c = new float[8];

        bool ran;
        try
        {
            ran = AiDotNet.Tensors.Engines.BlasManaged.Jit.MachineCodeBf16Kernel.TryRunProbe(a, b, c);
        }
        catch (Exception e)
        {
            Console.WriteLine("  probe threw (likely #UD — not under SDE, or bad encoding): " + e.Message);
            return false;
        }
        if (!ran)
        {
            Console.WriteLine("  executable memory unavailable (NativeAOT / hardened) — cannot verify");
            return false;
        }

        bool pass = true;
        for (int i = 0; i < 8; i++)
        {
            float exp = Bf16ToFloat(a[2 * i]) * Bf16ToFloat(b[2 * i])
                      + Bf16ToFloat(a[2 * i + 1]) * Bf16ToFloat(b[2 * i + 1]);
            float got = c[i];
            bool ok = Math.Abs(got - exp) <= 1e-4f * Math.Max(1f, Math.Abs(exp));
            pass &= ok;
            Console.WriteLine($"  lane {i}: got {got,11:G6}  exp {exp,11:G6}  {(ok ? "OK" : "FAIL")}");
        }
        Console.WriteLine(pass ? "VDPBF16PS VERIFIED" : "VDPBF16PS FAILED");
        return pass;
    }

    // #378 AVX-512-BF16 Phase 2: verify the full BF16 GEMM machine-code microkernel end to end.
    // Shape M=5,K=7,N=11 exercises every edge: ragged M (4+1), ragged N (8+3), odd K (pair pad).
    // Run UNDER INTEL SDE on a non-AVX-512-BF16 host (VDPBF16PS is emulated). False on mismatch.
    public static bool VerifyBf16Gemm()
    {
        Console.WriteLine("=== BF16 GEMM machine-code microkernel (MR4xNR8, VDPBF16PS K-loop) ===");
        const int m = 5, k = 7, n = 11;
        var rng = new Random(7);
        var a = new ushort[m * k];
        var b = new ushort[k * n];
        for (int i = 0; i < a.Length; i++) a[i] = FloatToBf16((float)(rng.NextDouble() * 2 - 1));
        for (int i = 0; i < b.Length; i++) b[i] = FloatToBf16((float)(rng.NextDouble() * 2 - 1));
        var c = new float[m * n];

        bool ran;
        try
        {
            ran = AiDotNet.Tensors.Engines.BlasManaged.Jit.MachineCodeBf16Kernel.TryGemm(a, b, c, m, k, n);
        }
        catch (Exception e)
        {
            Console.WriteLine("  GEMM threw (likely #UD — not under SDE, or bad encoding): " + e.Message);
            return false;
        }
        if (!ran)
        {
            Console.WriteLine("  executable memory unavailable — cannot verify");
            return false;
        }

        bool pass = true;
        int bad = 0;
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
            {
                double truth = 0;
                for (int kk = 0; kk < k; kk++)
                    truth += (double)Bf16ToFloat(a[i * k + kk]) * Bf16ToFloat(b[kk * n + j]);
                double got = c[i * n + j];
                bool ok = Math.Abs(got - truth) <= 1e-3 * Math.Max(1.0, Math.Abs(truth));
                if (!ok) { pass = false; if (bad++ < 6) Console.WriteLine($"  C[{i},{j}] got {got:G6} exp {truth:G6} FAIL"); }
            }
        Console.WriteLine(pass ? $"BF16 GEMM VERIFIED ({m}x{k}x{n}, all {m * n} elements)" : "BF16 GEMM FAILED");
        return pass;
    }

    // #380 AMX Phase 1: verify a single tdpbf16ps tile op — C[16,16] += A[16,32]·B[16,16] in FP32,
    // B in the AMX row-pair VNNI layout. Run UNDER INTEL SDE (-spr) on a non-AMX host (the tile
    // instructions are emulated). Returns false on mismatch / no executable memory.
    public static bool VerifyAmxTile()
    {
        Console.WriteLine("=== AMX tdpbf16ps tile op (C16x16 += A16x32 . B16x16, VNNI) ===");
        const int M = 16, K = 32, N = 16;

        var rng = new Random(31);
        // Logical A[M,K] and B[K,N] as BF16.
        var a = new ushort[M * K];
        var bLogical = new ushort[K * N];
        for (int i = 0; i < a.Length; i++) a[i] = FloatToBf16((float)(rng.NextDouble() * 2 - 1));
        for (int i = 0; i < bLogical.Length; i++) bLogical[i] = FloatToBf16((float)(rng.NextDouble() * 2 - 1));

        // Pack B into the AMX VNNI layout: row r (0..K/2-1) = {B[2r,n], B[2r+1,n]} over n.
        var bVnni = new ushort[(K / 2) * (N * 2)];
        for (int r = 0; r < K / 2; r++)
            for (int n = 0; n < N; n++)
            {
                bVnni[r * (N * 2) + 2 * n] = bLogical[(2 * r) * N + n];
                bVnni[r * (N * 2) + 2 * n + 1] = bLogical[(2 * r + 1) * N + n];
            }

        var c = new float[M * N];
        bool ran;
        try
        {
            ran = AiDotNet.Tensors.Engines.BlasManaged.Jit.MachineCodeAmxKernel.TryRunTileProbe(a, bVnni, c);
        }
        catch (Exception e)
        {
            Console.WriteLine("  tile op threw (likely #UD — not under SDE, or bad encoding): " + e.Message);
            return false;
        }
        if (!ran)
        {
            Console.WriteLine("  executable memory unavailable — cannot verify");
            return false;
        }

        bool pass = true;
        int bad = 0;
        for (int i = 0; i < M; i++)
            for (int j = 0; j < N; j++)
            {
                double truth = 0;
                for (int kk = 0; kk < K; kk++)
                    truth += (double)Bf16ToFloat(a[i * K + kk]) * Bf16ToFloat(bLogical[kk * N + j]);
                double got = c[i * N + j];
                bool ok = Math.Abs(got - truth) <= 1e-3 * Math.Max(1.0, Math.Abs(truth));
                if (!ok) { pass = false; if (bad++ < 6) Console.WriteLine($"  C[{i},{j}] got {got:G6} exp {truth:G6} FAIL"); }
            }
        Console.WriteLine(pass ? $"AMX TILE VERIFIED ({M}x{K}x{N}, all {M * N} elements)" : "AMX TILE FAILED");
        return pass;
    }

    private static ushort FloatToBf16(float f) => (ushort)(BitConverter.SingleToUInt32Bits(f) >> 16);
    private static float Bf16ToFloat(ushort h) => BitConverter.UInt32BitsToSingle((uint)h << 16);
}
