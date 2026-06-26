using System;
using System.Diagnostics;
using AiDotNet.Tensors.Engines.BlasManaged;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Tensors.Benchmarks;

/// <summary>
/// #475 medium-axis routing A/B. The deterministic-mode default heuristic
/// (<see cref="AxisSelector"/>) skips the M-axis when <c>k &gt; 256</c> (intending finer
/// K/2D splits for large K), but in deterministic mode the K-axis is forbidden, so a
/// "medium" shape (m gives enough M-blocks, n only modestly large) falls through to the
/// N-axis — which under-subscribes the cores vs the M-axis. This measures each forced
/// axis vs the live default heuristic and vs native OpenBLAS, at full DOP, deterministic
/// mode, warmed min-of-N, through the real <see cref="BlasManaged.Gemm{T}"/> strategy path.
///
/// <para>
/// <c>PackingMode.DisableAutotune</c> is used so the heuristic (and thus the
/// <see cref="AxisSelector.ForceAxisForTest"/> override) runs on every call instead of the
/// cache returning a stored decision — by default autotune is heuristic-only, so this is
/// representative of the production Auto path.
/// </para>
///
/// <para>Run: <c>--ab-axis-routing</c>.</para>
/// </summary>
public static class AxisRoutingAbBench
{
    public static void Run()
    {
        Console.WriteLine("=== #475 medium-axis routing A/B (FP32, full DOP, deterministic) ===");
        Console.WriteLine($"Host: cores={Environment.ProcessorCount}  HasRawSgemm={BlasProvider.HasRawSgemm}  Deterministic={BlasProvider.IsDeterministicMode}");
        Console.WriteLine("Per shape: native OpenBLAS bar, the live heuristic's pick, and each forced axis (GF/s, min-of-N).");
        Console.WriteLine();

        int procs = Environment.ProcessorCount;
        const int mr = 6, nr = 16; // FP32 AVX2 6x16 panel tile

        // (label, m, n, k) — diffusion-relevant FP32 GEMM shapes.
        var shapes = new (string label, int m, int n, int k)[]
        {
            ("medium    384x1024x1024", 384, 1024, 1024),
            ("square   1024x1024x1024", 1024, 1024, 1024),
            ("attn-proj 256x1536x1536", 256, 1536, 1536),
            ("ffn-up    384x6144x1536", 384, 6144, 1536),
            ("ffn-big   384x4096x3456", 384, 4096, 3456),
        };

        foreach (var (label, m, n, k) in shapes)
        {
            var a = MakeRandom(m * k);
            var b = MakeRandom(k * n);
            var c = new float[m * n];
            double flops = 2.0 * m * n * k;
            var heuristic = AxisSelector.Select(m, n, k, mr, nr, procs, BlasProvider.IsDeterministicMode);

            double obGf  = BlasProvider.HasRawSgemm ? Gf(flops, TimeMinNative(a, b, c, m, n, k)) : double.NaN;
            double defGf = Gf(flops, TimeMinManaged(a, b, c, m, n, k, null));
            double mGf   = Gf(flops, TimeMinManaged(a, b, c, m, n, k, ParallelismAxis.M));
            double nGf   = Gf(flops, TimeMinManaged(a, b, c, m, n, k, ParallelismAxis.N));
            double gGf   = Gf(flops, TimeMinManaged(a, b, c, m, n, k, ParallelismAxis.MN_2D));

            double best = Math.Max(mGf, Math.Max(nGf, gGf));
            string bestAxis = best == mGf ? "M" : best == nGf ? "N" : "2D";

            Console.WriteLine($"  {label}   (heuristic picks {heuristic})");
            Console.WriteLine($"    OpenBLAS {obGf,6:F0} | default {defGf,6:F0} | M {mGf,6:F0}  N {nGf,6:F0}  2D {gGf,6:F0}  GF/s");
            Console.WriteLine($"    best-forced {bestAxis} {best,6:F0} -> default is {defGf / best * 100,4:F0}% of best (gap {best - defGf,5:F0} GF/s)");
            Console.WriteLine();
        }
    }

    /// <summary>
    /// Single-thread GEMM profiling repro — pins MaxDOP=1 and hammers one shape through the
    /// managed PackBoth path for a fixed wall-time, so a PMU profiler (PerfView CPU-counter
    /// sampling) can attribute the single-core stall (the ~72 vs OpenBLAS ~103 GF/s gap).
    /// Run: <c>--profile-gemm-st [seconds=25] [shape=ffn-up|ffn-big|square|attn|medium]</c>.
    /// </summary>
    public static void ProfileSingleThread(int seconds, string shape)
    {
        CpuParallelSettings.MaxDegreeOfParallelism = 1; // profile the single-core path
        (int m, int n, int k) = shape switch
        {
            "ffn-big" => (384, 4096, 3456),
            "square"  => (1024, 1024, 1024),
            "attn"    => (256, 1536, 1536),
            "medium"  => (384, 1024, 1024),
            _         => (384, 6144, 1536), // ffn-up
        };
        var a = MakeRandom(m * k);
        var b = MakeRandom(k * n);
        var c = new float[m * n];
        double flops = 2.0 * m * n * k;
        Console.WriteLine($"=== single-thread GEMM profile repro: {shape} {m}x{n}x{k}  MaxDOP={CpuParallelSettings.MaxDegreeOfParallelism}  {seconds}s ===");
        for (int i = 0; i < 5; i++) GemmOnce(a, b, c, m, n, k); // warm
        var total = Stopwatch.StartNew();
        var one = new Stopwatch();
        long iters = 0;
        double bestSec = double.MaxValue;
        while (total.Elapsed.TotalSeconds < seconds)
        {
            one.Restart();
            GemmOnce(a, b, c, m, n, k);
            one.Stop();
            bestSec = Math.Min(bestSec, one.Elapsed.TotalSeconds);
            iters++;
        }
        Console.WriteLine($"iters={iters}  best {flops / bestSec / 1e9:F1} GF/s  avg {flops / (total.Elapsed.TotalSeconds / iters) / 1e9:F1} GF/s  (sink {c[0]:E2})");
    }

    private static void GemmOnce(float[] a, float[] b, float[] c, int m, int n, int k)
    {
        var opts = new BlasOptions<float> { PackingMode = PackingMode.DisableAutotune };
        BlasManaged.Gemm<float>(a, k, false, b, n, false, c, n, m, n, k, in opts);
    }

    private static double TimeMinGemm(float[] a, float[] b, float[] c, int m, int n, int k)
    {
        double work = (double)m * n * k;
        int iters = work > 2e9 ? 2 : work > 5e8 ? 4 : 10;
        for (int i = 0; i < 3 * iters; i++) GemmOnce(a, b, c, m, n, k);
        var sw = new Stopwatch();
        double best = double.MaxValue;
        for (int r = 0; r < 12; r++) { sw.Restart(); for (int i = 0; i < iters; i++) GemmOnce(a, b, c, m, n, k); sw.Stop(); best = Math.Min(best, sw.Elapsed.TotalSeconds / iters); }
        return best;
    }

    /// <summary>
    /// #475 machine-code MACRO kernel A/B: the whole Mr-sweep in asm (RyuJIT off the hot path) vs
    /// the current managed-loop path, single- and multi-thread, with a bit-exactness check.
    /// Run: <c>--ab-macro</c>.
    /// </summary>
    public static void MacroAb()
    {
        Console.WriteLine("=== #475 MACRO kernel A/B (FP32) — whole Mr-sweep in machine code ===");
        Console.WriteLine($"MacroAvailable={MachineKernelGemm.IsFp32MacroAvailable}  cores={Environment.ProcessorCount}");
        var shapes = new (string label, int m, int n, int k)[]
        {
            ("medium  384x1024x1024", 384, 1024, 1024),
            ("attn    256x1536x1536", 256, 1536, 1536),
            ("ffn-up  384x6144x1536", 384, 6144, 1536),
            ("ffn-big 384x4096x3456", 384, 4096, 3456),
        };
        foreach (var (label, m, n, k) in shapes)
        {
            var a = MakeRandom(m * k);
            var b = MakeRandom(k * n);
            double flops = 2.0 * m * n * k;
            var cref = new float[m * n];
            var c = new float[m * n];
            foreach (int dop in new[] { 1, 32 })
            {
                CpuParallelSettings.MaxDegreeOfParallelism = dop;
                PackBothStrategy.s_macroKernel = false;
                Array.Clear(cref, 0, cref.Length); GemmOnce(a, b, cref, m, n, k);
                double baseSec = TimeMinGemm(a, b, c, m, n, k);
                PackBothStrategy.s_macroKernel = true;
                Array.Clear(c, 0, c.Length); GemmOnce(a, b, c, m, n, k);
                double maxErr = 0;
                for (int i = 0; i < c.Length; i++) { double e = Math.Abs(c[i] - cref[i]); if (e > maxErr) maxErr = e; }
                double macroSec = TimeMinGemm(a, b, c, m, n, k);
                PackBothStrategy.s_macroKernel = false;
                Console.WriteLine($"  {label} DOP={dop,2}: base {flops / baseSec / 1e9,6:F0}  macro {flops / macroSec / 1e9,6:F0} GF/s  ({baseSec / macroSec:F2}x)  maxErr={maxErr:E1}");
            }
        }
        CpuParallelSettings.MaxDegreeOfParallelism = Environment.ProcessorCount;
    }

    /// <summary>
    /// #475 Phase 0a: sweep the FP32 panel K-unroll (4/8/2/6) through the macro kernel, single- and
    /// multi-thread, vs the U=4 reference, with a bit-exactness check. OpenBLAS's Zen sgemm uses 8.
    /// Run: <c>--ab-kunroll</c>.
    /// </summary>
    public static void KUnrollSweep()
    {
        Console.WriteLine("=== #475 Phase 0a: FP32 panel K-unroll sweep (macro kernel) ===");
        var shapes = new (string label, int m, int n, int k)[]
        {
            ("ffn-up  384x6144x1536", 384, 6144, 1536),
            ("ffn-big 384x4096x3456", 384, 4096, 3456),
            ("medium  384x1024x1024", 384, 1024, 1024),
        };
        int[] unrolls = { 4, 8, 2, 6 };
        foreach (var (label, m, n, k) in shapes)
        {
            var a = MakeRandom(m * k);
            var b = MakeRandom(k * n);
            double flops = 2.0 * m * n * k;
            // Reference: default kernel (U=4), managed base path, single thread.
            MachineCodeFmaKernel.PanelKUnroll = 4; MachineKernelGemm.ResetFp32Kernels();
            PackBothStrategy.s_macroKernel = false;
            CpuParallelSettings.MaxDegreeOfParallelism = 1;
            var cref = new float[m * n];
            GemmOnce(a, b, cref, m, n, k);

            var c = new float[m * n];
            PackBothStrategy.s_macroKernel = true;
            foreach (int u in unrolls)
            {
                MachineCodeFmaKernel.PanelKUnroll = u; MachineKernelGemm.ResetFp32Kernels();
                _ = MachineKernelGemm.IsFp32MacroAvailable; // force re-emit
                foreach (int dop in new[] { 1, 32 })
                {
                    CpuParallelSettings.MaxDegreeOfParallelism = dop;
                    Array.Clear(c, 0, c.Length); GemmOnce(a, b, c, m, n, k);
                    double maxErr = 0;
                    for (int i = 0; i < c.Length; i++) { double e = Math.Abs(c[i] - cref[i]); if (e > maxErr) maxErr = e; }
                    double sec = TimeMinGemm(a, b, c, m, n, k);
                    Console.WriteLine($"  {label} U={u} DOP={dop,2}: {flops / sec / 1e9,6:F0} GF/s  maxErr={maxErr:E1}");
                }
            }
        }
        MachineCodeFmaKernel.PanelKUnroll = 4; MachineKernelGemm.ResetFp32Kernels();
        PackBothStrategy.s_macroKernel = false;
        CpuParallelSettings.MaxDegreeOfParallelism = Environment.ProcessorCount;
    }

    /// <summary>
    /// #475 Phase 1: JIT specialized FP32 GEMM A/B vs native OpenBLAS on small/medium shapes (the
    /// libxsmm regime). Correctness vs a double-precision scalar reference (small float-rounding
    /// maxErr expected, not 0). Run: <c>--ab-jit</c>.
    /// </summary>
    public static unsafe void JitAb()
    {
        Console.WriteLine("=== #475 Phase 1: JIT specialized FP32 GEMM A/B (single-thread) ===");
        Console.WriteLine($"JIT supported={JitGemmGenerator.IsSupported}  HasRawSgemm={BlasProvider.HasRawSgemm}");
        var shapes = new (int m, int n, int k)[]
        {
            (6,16,64),(12,32,128),(24,48,128),(48,64,256),(64,64,128),
            (96,96,256),(128,128,256),(64,512,512),(128,256,512),(256,256,256),
        };
        foreach (var (m, n, k) in shapes)
        {
            var a = MakeRandom(m * k);
            var b = MakeRandom(k * n);
            var cref = new float[m * n];
            ScalarGemm(a, b, cref, m, n, k);
            var c = new float[m * n];
            bool ok;
            fixed (float* pa = a, pb = b, pc = c) ok = JitGemmGenerator.TryRunFp32(pa, k, pb, n, pc, n, m, n, k);
            double maxErr = 0;
            for (int i = 0; i < c.Length; i++) { double e = Math.Abs((double)c[i] - cref[i]); if (e > maxErr) maxErr = e; }
            double flops = 2.0 * m * n * k;
            double jitSec = ok ? TimeMinPtr(a, b, c, m, n, k, jit: true) : double.NaN;
            double obSec = BlasProvider.HasRawSgemm ? TimeMinPtr(a, b, c, m, n, k, jit: false) : double.NaN;
            string cmp = (BlasProvider.HasRawSgemm && ok) ? $"  {obSec / jitSec:F2}x vs OpenBLAS" : "";
            Console.WriteLine($"  {m,4}x{n,4}x{k,4}  ok={ok} maxErr={maxErr:E1}  JIT {flops / jitSec / 1e9,5:F0}  OB {flops / obSec / 1e9,5:F0} GF/s{cmp}");
        }
    }

    private static void ScalarGemm(float[] a, float[] b, float[] c, int m, int n, int k)
    {
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
            {
                double s = 0;
                for (int p = 0; p < k; p++) s += (double)a[i * k + p] * b[p * n + j];
                c[i * n + j] = (float)s;
            }
    }

    private static unsafe double TimeMinPtr(float[] a, float[] b, float[] c, int m, int n, int k, bool jit)
    {
        Action op = jit
            ? () => { fixed (float* pa = a, pb = b, pc = c) JitGemmGenerator.TryRunFp32(pa, k, pb, n, pc, n, m, n, k); }
            : () => { fixed (float* pa = a, pb = b, pc = c) BlasProvider.SgemmRaw(m, n, k, pa, k, pb, n, pc, n); };
        return TimeMin(op, m, n, k);
    }

    /// <summary>
    /// #475 medium/large gap: sweep the PackBoth blocking (Mc/Nc/Kc) against OpenBLAS's Zen sgemm
    /// P/Q/R (320/320/large) on the diffusion shapes, single- and multi-thread, vs the OpenBLAS bar.
    /// Tests whether precise blocking closes the 1.3-1.7x single-core gap. Run: <c>--ab-blocking</c>.
    /// </summary>
    public static unsafe void BlockingSweep()
    {
        Console.WriteLine("=== #475 medium/large blocking A/B (vs OpenBLAS P/Q/R = 320/320) ===");
        Console.WriteLine($"HasRawSgemm={BlasProvider.HasRawSgemm}  cores={Environment.ProcessorCount}");
        var shapes = new (string label, int m, int n, int k)[]
        {
            ("ffn-up 384x6144x1536", 384, 6144, 1536),
            ("medium 384x1024x1024", 384, 1024, 1024),
            ("square 1024x1024x1024", 1024, 1024, 1024),
        };
        var configs = new (string name, int? mc, int? nc, int? kc)[]
        {
            ("default", null, null, null), ("Kc512", null, null, 512),
            ("Nc256", null, 256, null), ("Nc128", null, 128, null),
            ("Nc256Kc512", null, 256, 512), ("Nc128Kc512", null, 128, 512),
        };
        PackBothStrategy.s_macroKernel = true;
        foreach (var (label, m, n, k) in shapes)
        {
            var a = MakeRandom(m * k); var b = MakeRandom(k * n); var c = new float[m * n];
            double flops = 2.0 * m * n * k;
            Console.WriteLine($"  {label}:");
            foreach (var (name, mc, nc, kc) in configs)
            {
                AutotuneDispatcher.s_overrideMc = mc; AutotuneDispatcher.s_overrideNc = nc; AutotuneDispatcher.s_overrideKc = kc;
                foreach (int dop in new[] { 1, 32 })
                {
                    CpuParallelSettings.MaxDegreeOfParallelism = dop;
                    double sec = TimeMinGemm(a, b, c, m, n, k);
                    Console.WriteLine($"    {name,-11} DOP={dop,2}: {flops / sec / 1e9,6:F0} GF/s");
                }
            }
            AutotuneDispatcher.s_overrideMc = AutotuneDispatcher.s_overrideNc = AutotuneDispatcher.s_overrideKc = null;
            if (BlasProvider.HasRawSgemm)
                foreach (int dop in new[] { 1, 32 })
                {
                    CpuParallelSettings.MaxDegreeOfParallelism = dop;
                    double sec = TimeMinPtr(a, b, c, m, n, k, jit: false);
                    Console.WriteLine($"    OpenBLAS    DOP={dop,2}: {flops / sec / 1e9,6:F0} GF/s");
                }
        }
        AutotuneDispatcher.s_overrideMc = AutotuneDispatcher.s_overrideNc = AutotuneDispatcher.s_overrideKc = null;
        PackBothStrategy.s_macroKernel = false;
        CpuParallelSettings.MaxDegreeOfParallelism = Environment.ProcessorCount;
    }

    /// <summary>
    /// #475 thread-scaling curve: GF/s + scaling + efficiency at DOP 1..32 on the diffusion shapes,
    /// vs OpenBLAS. Locates WHERE multi-thread scaling breaks (SMT/CCX/contention) before changing
    /// anything. Run: <c>--ab-scaling</c> (with AIDOTNET_USE_BLAS=1 for the OB bar).
    /// </summary>
    public static unsafe void ScalingSweep()
    {
        Console.WriteLine("=== #475 thread-scaling curve (16C/32T Zen2) ===");
        Console.WriteLine($"PhysicalCoreCount={CpuParallelSettings.PhysicalCoreCount}  Logical={Environment.ProcessorCount}");
        // Same-process A/B: the SMT cap ON vs OFF at the default DOP (noise-robust, min-of-N).
        Console.WriteLine("--- SMT-cap A/B at default DOP (cap=physical vs cap=off) ---");
        CpuParallelSettings.MaxDegreeOfParallelism = Environment.ProcessorCount;
        PackBothStrategy.s_macroKernel = true;
        foreach (var (m, n, k) in new[] { (384, 6144, 1536), (384, 1024, 1024), (1024, 1024, 1024) })
        {
            var a = MakeRandom(m * k); var b = MakeRandom(k * n); var c = new float[m * n];
            double flops = 2.0 * m * n * k;
            CpuParallelSettings.CapGemmAtPhysicalCores = true;
            double on = flops / TimeMinGemm(a, b, c, m, n, k) / 1e9;
            CpuParallelSettings.CapGemmAtPhysicalCores = false;
            double off = flops / TimeMinGemm(a, b, c, m, n, k) / 1e9;
            CpuParallelSettings.CapGemmAtPhysicalCores = true;
            Console.WriteLine($"    {m}x{n}x{k}: cap-on {on,5:F0}  cap-off {off,5:F0} GF/s  ({on / off:F2}x)");
        }
        PackBothStrategy.s_macroKernel = false;
        var shapes = new (string label, int m, int n, int k)[]
        {
            ("ffn    384x6144x1536", 384, 6144, 1536),
            ("medium 384x1024x1024", 384, 1024, 1024),
            ("square 1024x1024x1024", 1024, 1024, 1024),
        };
        int[] dops = { 1, 2, 4, 8, 16, 24, 32 };
        PackBothStrategy.s_macroKernel = true;
        foreach (var (label, m, n, k) in shapes)
        {
            var a = MakeRandom(m * k); var b = MakeRandom(k * n); var c = new float[m * n];
            double flops = 2.0 * m * n * k;
            Console.WriteLine($"  {label}:");
            double s1 = 0;
            foreach (int dop in dops)
            {
                CpuParallelSettings.MaxDegreeOfParallelism = dop;
                double gf = flops / TimeMinGemm(a, b, c, m, n, k) / 1e9;
                if (dop == 1) s1 = gf;
                Console.WriteLine($"    DOP={dop,2}: {gf,6:F0} GF/s   {gf / s1,4:F1}x   eff={gf / s1 / dop * 100,3:F0}%");
            }
            if (BlasProvider.HasRawSgemm)
            {
                CpuParallelSettings.MaxDegreeOfParallelism = 32;
                Console.WriteLine($"    OpenBLAS (auto-threaded): {flops / TimeMinPtr(a, b, c, m, n, k, jit: false) / 1e9,6:F0} GF/s");
            }
        }
        PackBothStrategy.s_macroKernel = false;
        CpuParallelSettings.MaxDegreeOfParallelism = Environment.ProcessorCount;
    }

    /// <summary>
    /// #85 GotoBLAS A/B: the low-barrier jc-blocked path (B packed per Nc-block into shared L3,
    /// A packed once with disjoint-row reads, one parallel region per Nc-block) vs the current
    /// N-axis path (whole shared A re-read by every thread per N-block) and OpenBLAS, on the wide-N
    /// diffusion shapes at the default DOP. Also asserts bit-exactness vs the default path. Run
    /// <c>--ab-gotoblas</c> (add AIDOTNET_USE_BLAS=1 for the OpenBLAS bar).
    /// </summary>
    public static unsafe void GotoBlasSweep()
    {
        Console.WriteLine("=== #85 GotoBLAS jc-blocked path A/B (vs N-axis + OpenBLAS) ===");
        PackBothStrategy.s_gotoBlasAuto = false; // bench drives the path explicitly via s_forceGotoBlas
        Console.WriteLine($"HasRawSgemm={BlasProvider.HasRawSgemm}  PhysicalCores={CpuParallelSettings.PhysicalCoreCount}  Logical={Environment.ProcessorCount}");
        var shapes = new (string label, int m, int n, int k)[]
        {
            ("ffn-up   384x6144x1536", 384, 6144, 1536),
            ("medium   384x1024x1024", 384, 1024, 1024),
            ("tall-ffn 768x4096x1024", 768, 4096, 1024),
        };
        PackBothStrategy.s_macroKernel = true;
        foreach (var (label, m, n, k) in shapes)
        {
            var a = MakeRandom(m * k); var b = MakeRandom(k * n);
            var cRef = new float[m * n]; var cGoto = new float[m * n];
            double flops = 2.0 * m * n * k;
            // Correctness: GotoBLAS path must be bit-identical to the default (same K-panel order).
            PackBothStrategy.s_forceGotoBlas = false; GemmOnce(a, b, cRef, m, n, k);
            PackBothStrategy.s_forceGotoBlas = true; GemmOnce(a, b, cGoto, m, n, k);
            PackBothStrategy.s_forceGotoBlas = false;
            double maxErr = 0;
            for (int i = 0; i < cRef.Length; i++) maxErr = Math.Max(maxErr, Math.Abs(cRef[i] - cGoto[i]));
            Console.WriteLine($"  {label}: maxErr(goto vs default)={maxErr:E2}");
            var c = new float[m * n];
            foreach (int dop in new[] { 16, 32 })
            {
                CpuParallelSettings.MaxDegreeOfParallelism = dop;
                PackBothStrategy.s_forceGotoBlas = false;
                double dN = flops / TimeMinGemm(a, b, c, m, n, k) / 1e9;
                PackBothStrategy.s_forceGotoBlas = true;
                double dG = flops / TimeMinGemm(a, b, c, m, n, k) / 1e9;
                PackBothStrategy.s_forceGotoBlas = false;
                Console.WriteLine($"    DOP={dop,2}: N-axis {dN,5:F0}   GotoBLAS {dG,5:F0} GF/s   ({dG / dN:F2}x)");
            }
            if (BlasProvider.HasRawSgemm)
            {
                CpuParallelSettings.MaxDegreeOfParallelism = 32;
                Console.WriteLine($"    OpenBLAS (auto-threaded): {flops / TimeMinPtr(a, b, c, m, n, k, jit: false) / 1e9,5:F0} GF/s");
            }
        }
        PackBothStrategy.s_macroKernel = false;
        PackBothStrategy.s_forceGotoBlas = false;
        CpuParallelSettings.MaxDegreeOfParallelism = Environment.ProcessorCount;
    }

    /// <summary>
    /// #85 GotoBLAS nc tuning: the wide-N losses come from many small Nc-blocks (12 for ffn at
    /// nc=512) = many barriers + a small shared B re-read by all ic-blocks. Sweep nc upward to cut
    /// the barrier count (bigger shared B per block, still L3-resident up to ~9 MB) vs the N-axis
    /// baseline, at the default DOP. Run <c>--ab-gotoblas-nc</c>.
    /// </summary>
    public static unsafe void GotoBlasNcSweep()
    {
        Console.WriteLine("=== #85 GotoBLAS nc tuning (cut wide-N barrier count) ===");
        Console.WriteLine($"PhysicalCores={CpuParallelSettings.PhysicalCoreCount}  Logical={Environment.ProcessorCount}");
        PackBothStrategy.s_gotoBlasAuto = false; // bench drives the path explicitly via s_forceGotoBlas
        var shapes = new (string label, int m, int n, int k)[]
        {
            ("ffn-up   384x6144x1536", 384, 6144, 1536),
            ("tall-ffn 768x4096x1024", 768, 4096, 1024),
            ("medium   384x1024x1024", 384, 1024, 1024),
        };
        int[] ncs = { 512, 1024, 1536, 2048, 3072, 6144 };
        PackBothStrategy.s_macroKernel = true;
        CpuParallelSettings.MaxDegreeOfParallelism = 32;
        foreach (var (label, m, n, k) in shapes)
        {
            var a = MakeRandom(m * k); var b = MakeRandom(k * n); var c = new float[m * n];
            double flops = 2.0 * m * n * k;
            PackBothStrategy.s_forceGotoBlas = false;
            AutotuneDispatcher.s_overrideNc = null;
            double dN = flops / TimeMinGemm(a, b, c, m, n, k) / 1e9;
            Console.WriteLine($"  {label}: N-axis baseline {dN,5:F0} GF/s");
            PackBothStrategy.s_forceGotoBlas = true;
            foreach (int nc in ncs)
            {
                if (nc > n) continue;
                AutotuneDispatcher.s_overrideNc = nc;
                double dG = flops / TimeMinGemm(a, b, c, m, n, k) / 1e9;
                Console.WriteLine($"    GotoBLAS nc={nc,4}: {dG,5:F0} GF/s   ({dG / dN:F2}x)");
            }
            AutotuneDispatcher.s_overrideNc = null;
            PackBothStrategy.s_forceGotoBlas = false;
        }
        PackBothStrategy.s_macroKernel = false;
        CpuParallelSettings.MaxDegreeOfParallelism = Environment.ProcessorCount;
    }

    /// <summary>
    /// #85 GotoBLAS crossover: at fixed m, k the GotoBLAS path wins for moderate N and loses for very
    /// wide N. Ladder n/k from 1x to 4x to locate the routing threshold. Interleaved A/B (N-axis vs
    /// GotoBLAS) per n, min-of-N, default DOP. Run <c>--ab-gotoblas-x</c>.
    /// </summary>
    public static unsafe void GotoBlasCrossover()
    {
        Console.WriteLine("=== #85 GotoBLAS crossover (n/k ladder, m=384 k=1024, nc=512) ===");
        PackBothStrategy.s_gotoBlasAuto = false; // bench drives the path explicitly via s_forceGotoBlas
        CpuParallelSettings.MaxDegreeOfParallelism = 32;
        PackBothStrategy.s_macroKernel = true;
        int m = 384, k = 1024;
        foreach (int n in new[] { 1024, 1280, 1536, 2048, 2560, 3072, 4096 })
        {
            var a = MakeRandom(m * k); var b = MakeRandom(k * n); var c = new float[m * n];
            double flops = 2.0 * m * n * k;
            // 3 interleaved A/B reps to beat the box noise.
            double bestN = 0, bestG = 0;
            for (int rep = 0; rep < 3; rep++)
            {
                PackBothStrategy.s_forceGotoBlas = false; AutotuneDispatcher.s_overrideNc = null;
                bestN = Math.Max(bestN, flops / TimeMinGemm(a, b, c, m, n, k) / 1e9);
                PackBothStrategy.s_forceGotoBlas = true; AutotuneDispatcher.s_overrideNc = 512;
                bestG = Math.Max(bestG, flops / TimeMinGemm(a, b, c, m, n, k) / 1e9);
            }
            PackBothStrategy.s_forceGotoBlas = false; AutotuneDispatcher.s_overrideNc = null;
            Console.WriteLine($"  n={n,4} (n/k={n / 1024.0:F1}x): N-axis {bestN,5:F0}  GotoBLAS {bestG,5:F0}  ({bestG / bestN:F2}x)");
        }
        PackBothStrategy.s_macroKernel = false;
        CpuParallelSettings.MaxDegreeOfParallelism = Environment.ProcessorCount;
    }

    /// <summary>
    /// #85 multi-thread diagnosis: GF/s + BUSY-CORE utilization (process CPU-time / wall / core) for the
    /// production large-float path at full DOP. Busy-cores is the scientific discriminator vs OpenBLAS's
    /// scaling: LOW busy-cores ⇒ we stall on barriers/sync (OB uses lock-free per-slice flags, not
    /// barriers); HIGH busy-cores but low GF/s ⇒ memory-bandwidth bound (private per-thread buffers vs
    /// OB's one shared sa/sb). Run <c>--profile-gemm-mt [seconds] [shape]</c>.
    /// </summary>
    public static void ProfileMultiThread(int seconds, string shape)
    {
        CpuParallelSettings.MaxDegreeOfParallelism = Environment.ProcessorCount;
        (int m, int n, int k) = shape switch
        {
            "ffn-big" => (384, 4096, 3456),
            "square"  => (1024, 1024, 1024),
            "attn"    => (256, 1536, 1536),
            "medium"  => (384, 1024, 1024),
            _         => (384, 6144, 1536), // ffn-up
        };
        var a = MakeRandom(m * k); var b = MakeRandom(k * n); var c = new float[m * n];
        double flops = 2.0 * m * n * k;
        int cores = Environment.ProcessorCount;
        Console.WriteLine($"=== multi-thread GEMM profile: {shape} {m}x{n}x{k}  DOP={cores}  {seconds}s ===");
        for (int i = 0; i < 10; i++) GemmOnce(a, b, c, m, n, k); // warm
        var proc = System.Diagnostics.Process.GetCurrentProcess();
        var cpu0 = proc.TotalProcessorTime;
        var total = Stopwatch.StartNew();
        var one = new Stopwatch();
        long iters = 0; double bestSec = double.MaxValue;
        while (total.Elapsed.TotalSeconds < seconds)
        {
            one.Restart(); GemmOnce(a, b, c, m, n, k); one.Stop();
            bestSec = Math.Min(bestSec, one.Elapsed.TotalSeconds);
            iters++;
        }
        double wall = total.Elapsed.TotalSeconds;
        double cpuSec = (proc.TotalProcessorTime - cpu0).TotalSeconds;
        double busyCores = cpuSec / wall;
        Console.WriteLine($"  best {flops / bestSec / 1e9:F0} GF/s   avg {flops / (wall / iters) / 1e9:F0} GF/s");
        Console.WriteLine($"  busy-cores {busyCores:F1} / {cores}  ({busyCores / cores * 100:F0}% util)   iters={iters}  (sink {c[0]:E2})");
        Console.WriteLine($"  per-busy-core {flops / (wall / iters) / 1e9 / busyCores:F1} GF/s/core   (OB on this box ~59/core)");
    }

    /// <summary>
    /// #85 routing fix A/B: GotoGemmFp32.RunParallel (per-tile private pack) vs the N-axis path
    /// (shared-A) for thin-M m%6==0 shapes. PerfView pinned RunParallel's redundant per-tile A-repack
    /// (~50x for wide-N) as the per-core killer; N-axis shares A. Toggled in-process via
    /// BlasManaged.s_disableGotoGemm. Confirms the win + the M boundary before retuning BeatsPackBoth.
    /// Run <c>--ab-goto-vs-naxis</c>.
    /// </summary>
    public static void GotoVsNaxisSweep()
    {
        Console.WriteLine("=== #85 GotoGemm RunParallel vs N-axis (thin-M, shared-A) ===");
        CpuParallelSettings.MaxDegreeOfParallelism = Environment.ProcessorCount;
        var shapes = new (string label, int m, int n, int k)[]
        {
            ("384x1024x1024 n=k  ", 384, 1024, 1024),
            ("384x2048x1024 n=2k ", 384, 2048, 1024),
            ("384x4096x1536 ffn  ", 384, 4096, 1536),
            ("384x6144x1536 ffn-up", 384, 6144, 1536),
            ("768x2048x2048 m=768", 768, 2048, 2048),
            ("768x4096x1024 tall ", 768, 4096, 1024),
            ("1536x1536x1536 sq  ", 1536, 1536, 1536),
        };
        foreach (var (label, m, n, k) in shapes)
        {
            var a = MakeRandom(m * k); var b = MakeRandom(k * n); var c = new float[m * n];
            double flops = 2.0 * m * n * k;
            double bestG = 0, bestN = 0;
            for (int rep = 0; rep < 3; rep++)
            {
                BlasManaged.s_disableGotoGemm = false; // GotoGemmFp32 path
                bestG = Math.Max(bestG, flops / TimeMinGemm(a, b, c, m, n, k) / 1e9);
                BlasManaged.s_disableGotoGemm = true;  // PackBoth (N-axis for thin-M wide-N)
                bestN = Math.Max(bestN, flops / TimeMinGemm(a, b, c, m, n, k) / 1e9);
            }
            BlasManaged.s_disableGotoGemm = false;
            string winner = bestN > bestG ? "N-axis" : "GotoGemm";
            Console.WriteLine($"  {label}: GotoGemm {bestG,5:F0}  N-axis {bestN,5:F0} GF/s  ({bestN / bestG:F2}x)  -> {winner}");
        }
    }

    /// <summary>
    /// #85 CCX pool vs RunParallel A/B: PerfView/busy-core showed the CCX pinned-pool barriers strand
    /// cores (square 1024: 6/32 busy, 237 GF/s) while barrier-free RunParallel gets 13.3 cores / 598.
    /// Toggle CcxGemmPool.s_disable in-process across balanced shapes to see if the barrier loss holds
    /// at scale (it should — per-core is equal, so the lost cores are pure barrier stall). Run
    /// <c>--ab-ccx-vs-rp</c>.
    /// </summary>
    public static void CcxVsRunParallelSweep()
    {
        Console.WriteLine("=== #85 CCX pool (barriers) vs RunParallel (barrier-free) ===");
        CpuParallelSettings.MaxDegreeOfParallelism = Environment.ProcessorCount;
        var shapes = new (int m, int n, int k)[]
        {
            (1024, 1024, 1024), (1280, 1280, 1280), (1536, 1536, 1536),
            (1792, 1792, 1792), (2048, 2048, 2048),
        };
        foreach (var (m, n, k) in shapes)
        {
            var a = MakeRandom(m * k); var b = MakeRandom(k * n); var c = new float[m * n];
            double flops = 2.0 * m * n * k;
            double bestCcx = 0, bestRp = 0;
            for (int rep = 0; rep < 3; rep++)
            {
                AiDotNet.Tensors.Engines.BlasManaged.CcxGemmPool.s_disable = false;
                bestCcx = Math.Max(bestCcx, flops / TimeMinGemm(a, b, c, m, n, k) / 1e9);
                AiDotNet.Tensors.Engines.BlasManaged.CcxGemmPool.s_disable = true;
                bestRp = Math.Max(bestRp, flops / TimeMinGemm(a, b, c, m, n, k) / 1e9);
            }
            AiDotNet.Tensors.Engines.BlasManaged.CcxGemmPool.s_disable = false;
            string w = bestRp > bestCcx ? "RunParallel" : "CCX";
            Console.WriteLine($"  {m}x{n}x{k}: CCX {bestCcx,5:F0}  RunParallel {bestRp,5:F0} GF/s  ({bestRp / bestCcx:F2}x)  -> {w}");
        }
    }

    /// <summary>
    /// #85 spin-barrier A/B: OpenBLAS spins (YIELDING) at its sync points; our CcxGemmPool used a .NET
    /// Barrier that PARKS lanes (the 6/32-busy stall). Compare CCX with the lock-free spin barrier vs the
    /// .NET Barrier — forced at all sizes (s_ignoreWorkGate) — for GF/s AND bit-exact output. If spin lifts
    /// the small/large balanced shapes to the 1536³ per-core level, widen the CCX window. Run <c>--ab-spinbar</c>.
    /// </summary>
    public static void SpinBarrierSweep()
    {
        Console.WriteLine("=== #85 CCX spin barrier vs .NET Barrier (forced CCX, all sizes) ===");
        CpuParallelSettings.MaxDegreeOfParallelism = Environment.ProcessorCount;
        AiDotNet.Tensors.Engines.BlasManaged.CcxGemmPool.s_ignoreWorkGate = true;
        AiDotNet.Tensors.Engines.BlasManaged.CcxGemmPool.s_disable = false;
        var shapes = new (int m, int n, int k)[]
        {
            (1024, 1024, 1024), (1536, 1536, 1536), (2048, 2048, 2048), (1280, 1280, 1280),
        };
        foreach (var (m, n, k) in shapes)
        {
            var a = MakeRandom(m * k); var b = MakeRandom(k * n);
            var cBar = new float[m * n]; var cSpin = new float[m * n];
            double flops = 2.0 * m * n * k;
            // Correctness: spin must be bit-identical to the barrier (same compute order).
            AiDotNet.Tensors.Engines.BlasManaged.CcxGemmPool.s_spinBarrier = false; GemmOnce(a, b, cBar, m, n, k);
            AiDotNet.Tensors.Engines.BlasManaged.CcxGemmPool.s_spinBarrier = true; GemmOnce(a, b, cSpin, m, n, k);
            double maxErr = 0; for (int i = 0; i < cBar.Length; i++) maxErr = Math.Max(maxErr, Math.Abs(cBar[i] - cSpin[i]));
            var c = new float[m * n];
            double bestBar = 0, bestSpin = 0;
            for (int rep = 0; rep < 3; rep++)
            {
                AiDotNet.Tensors.Engines.BlasManaged.CcxGemmPool.s_spinBarrier = false;
                bestBar = Math.Max(bestBar, flops / TimeMinGemm(a, b, c, m, n, k) / 1e9);
                AiDotNet.Tensors.Engines.BlasManaged.CcxGemmPool.s_spinBarrier = true;
                bestSpin = Math.Max(bestSpin, flops / TimeMinGemm(a, b, c, m, n, k) / 1e9);
            }
            Console.WriteLine($"  {m}x{n}x{k}: barrier {bestBar,5:F0}  spin {bestSpin,5:F0} GF/s  ({bestSpin / bestBar:F2}x)  maxErr={maxErr:E1}");
        }
        AiDotNet.Tensors.Engines.BlasManaged.CcxGemmPool.s_ignoreWorkGate = false;
        AiDotNet.Tensors.Engines.BlasManaged.CcxGemmPool.s_spinBarrier = true;
    }

    /// <summary>
    /// #85 CCX blocking sweep: CCX hits OB-level (66 GF/s/core) only at 1536³; sweep the 2D kc
    /// (AIDOTNET_CCX_KC / s_kc2dOverride) on the forced-CCX path to see if a different K-block extends
    /// that to 1024³/2048³ (fewer K-panels = fewer barriers + better L1 fit). Run <c>--ab-ccx-block</c>.
    /// </summary>
    public static void CcxBlockingSweep()
    {
        Console.WriteLine("=== #85 CCX 2D kc blocking sweep (forced CCX) ===");
        CpuParallelSettings.MaxDegreeOfParallelism = Environment.ProcessorCount;
        AiDotNet.Tensors.Engines.BlasManaged.CcxGemmPool.s_ignoreWorkGate = true;
        AiDotNet.Tensors.Engines.BlasManaged.CcxGemmPool.s_disable = false;
        AiDotNet.Tensors.Engines.BlasManaged.CcxGemmPool.s_spinBarrier = false;
        int[] mcs = { 0, 48, 96, 144, 192, 240 }; // 0 = default heuristic
        var shapes = new (int m, int n, int k)[] { (1024, 1024, 1024), (1536, 1536, 1536), (2048, 2048, 2048) };
        foreach (var (m, n, k) in shapes)
        {
            var a = MakeRandom(m * k); var b = MakeRandom(k * n); var c = new float[m * n];
            double flops = 2.0 * m * n * k;
            Console.WriteLine($"  {m}x{n}x{k}:");
            foreach (int mc in mcs)
            {
                AiDotNet.Tensors.Engines.BlasManaged.CcxGemmPool.s_mc2dOverride = mc;
                double best = 0;
                for (int rep = 0; rep < 3; rep++) best = Math.Max(best, flops / TimeMinGemm(a, b, c, m, n, k) / 1e9);
                Console.WriteLine($"    mc={(mc == 0 ? "dflt" : mc.ToString()),4}: {best,5:F0} GF/s   ({best / 16:F0}/core)");
            }
        }
        AiDotNet.Tensors.Engines.BlasManaged.CcxGemmPool.s_mc2dOverride = 0;
        AiDotNet.Tensors.Engines.BlasManaged.CcxGemmPool.s_ignoreWorkGate = false;
    }

    private static double Gf(double flops, double sec) => flops / sec / 1e9;

    private static unsafe double TimeMinNative(float[] a, float[] b, float[] c, int m, int n, int k)
        => TimeMin(() => { fixed (float* pa = a, pb = b, pc = c) BlasProvider.SgemmRaw(m, n, k, pa, k, pb, n, pc, n); }, m, n, k);

    private static double TimeMinManaged(float[] a, float[] b, float[] c, int m, int n, int k, ParallelismAxis? force)
    {
        return TimeMin(() =>
        {
            var opts = new BlasOptions<float> { PackingMode = PackingMode.DisableAutotune };
            AxisSelector.ForceAxisForTest = force;
            try { BlasManaged.Gemm<float>(a, k, false, b, n, false, c, n, m, n, k, in opts); }
            finally { AxisSelector.ForceAxisForTest = null; }
        }, m, n, k);
    }

    private static double TimeMin(Action op, int m, int n, int k)
    {
        double work = (double)m * n * k;
        int iters = work > 2e9 ? 2 : work > 5e8 ? 4 : 10;
        const int rounds = 12;
        for (int i = 0; i < 3 * iters; i++) op(); // warm
        var sw = new Stopwatch();
        double best = double.MaxValue;
        for (int r = 0; r < rounds; r++)
        {
            sw.Restart();
            for (int i = 0; i < iters; i++) op();
            sw.Stop();
            best = Math.Min(best, sw.Elapsed.TotalSeconds / iters);
        }
        return best;
    }

    private static float[] MakeRandom(int len)
    {
        var rng = new Random(17);
        var x = new float[len];
        for (int i = 0; i < len; i++) x[i] = (float)(rng.NextDouble() * 2 - 1);
        return x;
    }
}
