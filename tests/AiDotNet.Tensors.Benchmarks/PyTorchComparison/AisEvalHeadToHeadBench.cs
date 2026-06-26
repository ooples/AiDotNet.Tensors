using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Runtime.InteropServices;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using TorchSharp;

namespace AiDotNet.Tensors.Benchmarks.PyTorchComparison;

/// <summary>
/// Issue #436 — fresh same-machine head-to-head of the three fused inference
/// primitives (<c>MlpForward</c>, <c>MultiHeadAttentionForward</c>,
/// <c>LstmSequenceForward</c>) vs the equivalent TorchSharp (libtorch/MKL)
/// modules, at the exact AIsEval shapes the issue flagged as losing badly.
///
/// <para>
/// The bar is deliberately strict: a family is declared a WIN only when
/// <b>AiDotNet's p95 latency is below PyTorch's median</b> — i.e. even our
/// slow-tail beats their typical case, which is "outside noise" in the truest
/// sense. We report median, p95, p99, and the p95-vs-median verdict per family.
/// </para>
///
/// <para>
/// Both sides run forward-only (grad disabled), all-cores, on freshly random
/// inputs, after a warmup + forced GC settle. Shapes mirror the AIsEval CPU
/// scaffold: MLP <c>Dense(784→512→128→10)</c> @ bs=128, MHA <c>[128,32,64]</c>
/// h=4, LSTM <c>[128,32,32]→64</c>.
/// </para>
///
/// Run with: <c>dotnet run -c Release -- --ab-aiseval-h2h</c>
/// </summary>
internal static class AisEvalHeadToHeadBench
{
    private const int Warmup = 50;
    private const int Iters = 400;
    // Min-of-N robustness: each measurement runs Rounds independent windows and
    // keeps the window with the lowest median — the least load-perturbed sample.
    // Both sides are measured identically so the comparison stays fair, and a
    // transient OS/GC/neighbor spike in one window can't decide the verdict.
    private const int Rounds = 7;

    public static void Run()
    {
        torch.set_grad_enabled(false);
        // Match thread budgets: AiDotNet's CPU primitives use all logical cores;
        // pin torch to the same so neither side wins on thread count alone.
        torch.set_num_threads(Environment.ProcessorCount);

        Console.WriteLine("=== Issue #436 — AIsEval head-to-head: AiDotNet fused primitives vs TorchSharp (CPU) ===");
        Console.WriteLine($"Host: cores={Environment.ProcessorCount}, torch threads={torch.get_num_threads()}");
        Console.WriteLine("Win criterion: AiDotNet p95 < PyTorch median (beat their typical with our slow-tail).");
        Console.WriteLine();

        var engine = new CpuEngine();
        var results = new List<Row>
        {
            MeasureMlp(engine),
            MeasureMha(engine),
            MeasureLstm(engine),
        };

        Console.WriteLine();
        Console.WriteLine($"{"Model",-14}{"AiDN med",11}{"AiDN p95",11}{"Torch med",11}{"Torch p95",11}{"p95/med",9}  Verdict");
        Console.WriteLine(new string('-', 88));
        int wins = 0;
        foreach (var r in results)
        {
            bool win = r.AiP95 < r.TorchMedian;
            if (win) wins++;
            double ratio = r.TorchMedian > 0 ? r.AiP95 / r.TorchMedian : double.PositiveInfinity;
            Console.WriteLine($"{r.Name,-14}{r.AiMedian,10:F3}ms{r.AiP95,10:F3}ms{r.TorchMedian,10:F3}ms{r.TorchP95,10:F3}ms{ratio,8:F2}  {(win ? "WIN" : "LOSS")}");
        }
        Console.WriteLine(new string('-', 88));
        Console.WriteLine($"{wins}/{results.Count} families beat PyTorch on the p95<median bar.");
    }

    // --- MLP: Dense(784→512)→ReLU→Dense(512→128)→ReLU→Dense(128→10) @ bs=128 ---
    private static Row MeasureMlp(CpuEngine engine)
    {
        const int bs = 128;
        var input = Tensor<float>.CreateRandom(bs, 784);
        var weights = new List<Tensor<float>>
        {
            Tensor<float>.CreateRandom(784, 512),
            Tensor<float>.CreateRandom(512, 128),
            Tensor<float>.CreateRandom(128, 10),
        };
        var biases = new List<Tensor<float>?>
        {
            Tensor<float>.CreateRandom(512),
            Tensor<float>.CreateRandom(128),
            Tensor<float>.CreateRandom(10),
        };
        var (aiMed, aiP95) = TimeAi(() =>
            engine.MlpForward(input, weights, biases, FusedActivationType.ReLU, FusedActivationType.None));

        var mlp = torch.nn.Sequential(
            ("fc1", torch.nn.Linear(784, 512)),
            ("relu1", torch.nn.ReLU()),
            ("fc2", torch.nn.Linear(512, 128)),
            ("relu2", torch.nn.ReLU()),
            ("fc3", torch.nn.Linear(128, 10)));
        mlp.eval();
        using var tInput = torch.randn(bs, 784);
        var (tMed, tP95) = TimeTorch(() => mlp.forward(tInput));

        return Print("MLP", aiMed, aiP95, tMed, tP95);
    }

    // --- MHA: self-attention [128,32,64], heads=4 ---
    private static Row MeasureMha(CpuEngine engine)
    {
        const int batch = 128, seq = 32, dModel = 64, heads = 4;
        var input = Tensor<float>.CreateRandom(batch, seq, dModel);
        var qW = Tensor<float>.CreateRandom(dModel, dModel);
        var kW = Tensor<float>.CreateRandom(dModel, dModel);
        var vW = Tensor<float>.CreateRandom(dModel, dModel);
        var oW = Tensor<float>.CreateRandom(dModel, dModel);
        var (aiMed, aiP95) = TimeAi(() =>
            engine.MultiHeadAttentionForward(input, qW, kW, vW, oW, heads));

        // TorchSharp's MultiheadAttention has no batch_first; it expects
        // [seq, batch, embed]. Same FLOPs as AiDotNet's [batch, seq, embed] —
        // fair for a latency comparison. need_weights:false skips the attention-
        // weight materialization AiDotNet also doesn't return.
        var mha = torch.nn.MultiheadAttention(dModel, heads);
        mha.eval();
        using var x = torch.randn(seq, batch, dModel);
        var (tMed, tP95) = TimeTorch(() =>
        {
            var (o, _) = mha.forward(x, x, x, null, false, null);
            return o;
        });

        return Print("Transformer", aiMed, aiP95, tMed, tP95);
    }

    // --- LSTM: [128,32,32] → hidden 64, last-step output ---
    private static Row MeasureLstm(CpuEngine engine)
    {
        const int batch = 128, seq = 32, inF = 32, hidden = 64;
        var input = Tensor<float>.CreateRandom(batch, seq, inF);
        var wIh = Tensor<float>.CreateRandom(4 * hidden, inF);
        var wHh = Tensor<float>.CreateRandom(4 * hidden, hidden);
        var (aiMed, aiP95) = TimeAi(() =>
            engine.LstmSequenceForward(input, null, null, wIh, wHh, null, null));

        var lstm = torch.nn.LSTM(inF, hidden, batchFirst: true);
        lstm.eval();
        using var x = torch.randn(batch, seq, inF);
        var (tMed, tP95) = TimeTorch(() =>
        {
            var (output, _, _) = lstm.forward(x);
            return output;
        });

        return Print("LSTM", aiMed, aiP95, tMed, tP95);
    }

    /// <summary>
    /// Diagnostic: re-measure each primitive with a <see cref="NativeInferencePool"/>
    /// active (pre-pinned weights + native activation buffers, zero-GC), and report
    /// per-call allocation. Quantifies the achievable floor vs the baseline path so
    /// we know whether the gap is allocation/jitter (closable here) or raw BLAS
    /// kernel quality (OpenBLAS-vs-MKL, needs deeper work). Run: --ab-aiseval-diag.
    /// </summary>
    public static void Diag()
    {
        torch.set_grad_enabled(false);
        torch.set_num_threads(Environment.ProcessorCount);
        Console.WriteLine("=== Issue #436 — diagnostic: zero-alloc pool floor + per-call alloc ===");
        Console.WriteLine($"Host: cores={Environment.ProcessorCount}");
        Console.WriteLine();

        var engine = new CpuEngine();

        // MLP
        {
            const int bs = 128;
            var input = Tensor<float>.CreateRandom(bs, 784);
            var weights = new List<Tensor<float>> {
                Tensor<float>.CreateRandom(784, 512), Tensor<float>.CreateRandom(512, 128), Tensor<float>.CreateRandom(128, 10) };
            var biases = new List<Tensor<float>?> {
                Tensor<float>.CreateRandom(512), Tensor<float>.CreateRandom(128), Tensor<float>.CreateRandom(10) };
            Func<Tensor<float>> fwd = () => engine.MlpForward(input, weights, biases, FusedActivationType.ReLU, FusedActivationType.None);
            ReportDiag("MLP baseline", fwd);
            using (NativeInferencePool.Create()) ReportDiag("MLP pooled  ", fwd);
        }
        // LSTM
        {
            const int batch = 128, seq = 32, inF = 32, hidden = 64;
            var input = Tensor<float>.CreateRandom(batch, seq, inF);
            var wIh = Tensor<float>.CreateRandom(4 * hidden, inF);
            var wHh = Tensor<float>.CreateRandom(4 * hidden, hidden);
            Func<Tensor<float>> fwd = () => engine.LstmSequenceForward(input, null, null, wIh, wHh, null, null);
            ReportDiag("LSTM baseline", fwd);
            using (NativeInferencePool.Create()) ReportDiag("LSTM pooled  ", fwd);
        }
        // MHA
        {
            const int batch = 128, seq = 32, dModel = 64, heads = 4;
            var input = Tensor<float>.CreateRandom(batch, seq, dModel);
            var qW = Tensor<float>.CreateRandom(dModel, dModel); var kW = Tensor<float>.CreateRandom(dModel, dModel);
            var vW = Tensor<float>.CreateRandom(dModel, dModel); var oW = Tensor<float>.CreateRandom(dModel, dModel);
            Func<Tensor<float>> fwd = () => engine.MultiHeadAttentionForward(input, qW, kW, vW, oW, heads);
            ReportDiag("MHA baseline", fwd);
            using (NativeInferencePool.Create()) ReportDiag("MHA pooled  ", fwd);
        }
    }

    /// <summary>
    /// Raw-GEMM compute floor: the 3 MLP GEMMs (bias+ReLU folded in) run directly
    /// through the loaded native BLAS with ZERO managed allocation and zero Tensor
    /// ceremony — the absolute best our current backend can do for this workload.
    /// If this floor already exceeds PyTorch's median, the p95&lt;median bar is
    /// unreachable without a faster GEMM backend (OpenBLAS-vs-MKL), not an
    /// allocation/dispatch fix. Run: --ab-aiseval-floor.
    /// </summary>
    public static unsafe void RawGemmFloor()
    {
        torch.set_grad_enabled(false);
        torch.set_num_threads(Environment.ProcessorCount);
        Console.WriteLine("=== Issue #436 — raw-GEMM compute floor (MLP shapes, zero-alloc native BLAS) ===");
        Console.WriteLine($"Host: cores={Environment.ProcessorCount}");
        Console.WriteLine($"BLAS backend: HasRawSgemm={BlasProvider.HasRawSgemm}, HasNativeSgemm={BlasProvider.HasNativeSgemm}, IsMklVerified={BlasProvider.IsMklVerified}");
        Console.WriteLine();

        const int bs = 128;
        // 3 layers: (128x784)*(784x512), (128x512)*(512x128), (128x128)*(128x10)
        var x0 = NewAligned(bs * 784);
        var w0 = NewAligned(784 * 512); var h0 = NewAligned(bs * 512);
        var w1 = NewAligned(512 * 128); var h1 = NewAligned(bs * 128);
        var w2 = NewAligned(128 * 10);  var y = NewAligned(bs * 10);
        var rng = new Random(7);
        foreach (var buf in new[] { x0, w0, w1, w2 })
            for (int i = 0; i < buf.Length; i++) buf[i] = (float)(rng.NextDouble() * 2 - 1);

        Action gemm3 = () =>
        {
            fixed (float* px = x0, pw0 = w0, ph0 = h0, pw1 = w1, ph1 = h1, pw2 = w2, py = y)
            {
                BlasProvider.SgemmRaw(bs, 512, 784, px, 784, pw0, 512, ph0, 512);
                for (int i = 0; i < bs * 512; i++) if (ph0[i] < 0) ph0[i] = 0; // ReLU
                BlasProvider.SgemmRaw(bs, 128, 512, ph0, 512, pw1, 128, ph1, 128);
                for (int i = 0; i < bs * 128; i++) if (ph1[i] < 0) ph1[i] = 0;
                BlasProvider.SgemmRaw(bs, 10, 128, ph1, 128, pw2, 10, py, 10);
            }
        };

        // MKL variant: same 3 GEMMs through the verified-MKL entry point (the
        // backend torch uses), to see whether routing inference GEMMs to MKL
        // instead of the OpenBLAS raw fptr closes the kernel-quality gap.
        Action gemm3Mkl = () =>
        {
            BlasProvider.MklSgemmZeroOffset(bs, 512, 784, x0, 784, w0, 512, h0, 512);
            for (int i = 0; i < bs * 512; i++) if (h0[i] < 0) h0[i] = 0;
            BlasProvider.MklSgemmZeroOffset(bs, 128, 512, h0, 512, w1, 128, h1, 128);
            for (int i = 0; i < bs * 128; i++) if (h1[i] < 0) h1[i] = 0;
            BlasProvider.MklSgemmZeroOffset(bs, 10, 128, h1, 128, w2, 10, y, 10);
        };

        // Managed variant: all 3 layers through SgemmWithCachedB (the BlasManaged
        // cached-B machine-code path, #409). Weights are stable arrays so the
        // pre-pack cache hits. Tests whether the managed microkernel beats native
        // BLAS at these small inference shapes (the #475 routing question).
        Action gemm3Managed = () =>
        {
            AiDotNet.Tensors.Engines.Simd.SimdGemm.SgemmWithCachedB(x0.AsSpan(0, bs * 784), w0, h0.AsSpan(0, bs * 512), bs, 784, 512);
            for (int i = 0; i < bs * 512; i++) if (h0[i] < 0) h0[i] = 0;
            AiDotNet.Tensors.Engines.Simd.SimdGemm.SgemmWithCachedB(h0.AsSpan(0, bs * 512), w1, h1.AsSpan(0, bs * 128), bs, 512, 128);
            for (int i = 0; i < bs * 128; i++) if (h1[i] < 0) h1[i] = 0;
            AiDotNet.Tensors.Engines.Simd.SimdGemm.SgemmWithCachedB(h1.AsSpan(0, bs * 128), w2, y.AsSpan(0, bs * 10), bs, 128, 10);
        };

        // OpenBLAS thread-count sweep: tiny GEMMs (esp. 128x10x128) may be hurt
        // by spinning all cores — torch's MKL uses small-GEMM thread heuristics.
        var sw = new Stopwatch();
        Console.WriteLine("  OpenBLAS thread sweep (raw 3xGEMM):");
        foreach (int t in new[] { 1, 2, 4, 8, 16 })
        {
            if (t > Environment.ProcessorCount) break;
            // Scope the override so the prior OpenBLAS thread state is restored after each
            // iteration (and after the loop), instead of hardcoding ProcessorCount — which
            // would contaminate the later measurements in this same process.
            using (BlasProvider.ScopeOpenBlasThreads(t))
            {
                for (int i = 0; i < Warmup; i++) gemm3();
                SettleGc();
                var tt = new double[Iters];
                for (int i = 0; i < Iters; i++) { sw.Restart(); gemm3(); sw.Stop(); tt[i] = sw.Elapsed.TotalMilliseconds; }
                var (tm, tp) = Percentiles(tt);
                Console.WriteLine($"    threads={t,2}: med {tm,7:F3}ms  p95 {tp,7:F3}ms");
            }
        }

        for (int i = 0; i < Warmup; i++) gemm3();
        SettleGc();
        var times = new double[Iters];
        for (int i = 0; i < Iters; i++) { sw.Restart(); gemm3(); sw.Stop(); times[i] = sw.Elapsed.TotalMilliseconds; }
        var (med, p95) = Percentiles(times);

        double mklMed = double.NaN, mklP95 = double.NaN;
        if (BlasProvider.IsMklVerified)
        {
            for (int i = 0; i < Warmup; i++) gemm3Mkl();
            SettleGc();
            var mt = new double[Iters];
            for (int i = 0; i < Iters; i++) { sw.Restart(); gemm3Mkl(); sw.Stop(); mt[i] = sw.Elapsed.TotalMilliseconds; }
            (mklMed, mklP95) = Percentiles(mt);
        }

        // Managed machine-code path timing
        for (int i = 0; i < Warmup; i++) gemm3Managed();
        SettleGc();
        var gt = new double[Iters];
        for (int i = 0; i < Iters; i++) { sw.Restart(); gemm3Managed(); sw.Stop(); gt[i] = sw.Elapsed.TotalMilliseconds; }
        var (mgMed, mgP95) = Percentiles(gt);

        // torch reference at the same shapes
        var mlp = torch.nn.Sequential(
            ("fc1", torch.nn.Linear(784, 512)), ("relu1", torch.nn.ReLU()),
            ("fc2", torch.nn.Linear(512, 128)), ("relu2", torch.nn.ReLU()),
            ("fc3", torch.nn.Linear(128, 10)));
        mlp.eval();
        using var tInput = torch.randn(bs, 784);
        var (tMed, tP95) = TimeTorch(() => mlp.forward(tInput));

        Console.WriteLine($"  raw 3xGEMM (OpenBLAS fptr) : med {med,7:F3}ms  p95 {p95,7:F3}ms");
        if (!double.IsNaN(mklMed))
            Console.WriteLine($"  raw 3xGEMM (MKL verified) : med {mklMed,7:F3}ms  p95 {mklP95,7:F3}ms");
        Console.WriteLine($"  raw 3xGEMM (managed mc)   : med {mgMed,7:F3}ms  p95 {mgP95,7:F3}ms");
        Console.WriteLine($"  torch MLP                 : med {tMed,7:F3}ms  p95 {tP95,7:F3}ms");
        Console.WriteLine();
        double bestFloor = double.IsNaN(mklMed) ? med : Math.Min(med, mklMed);
        Console.WriteLine(bestFloor < tMed
            ? $"  → best floor ({bestFloor:F3}) < torch median ({tMed:F3}): p95<median bar REACHABLE if we route to the faster kernel + kill alloc."
            : $"  → best floor ({bestFloor:F3}) >= torch median ({tMed:F3}): bar UNREACHABLE on available backends (kernel-quality gap).");
    }

    private static float[] NewAligned(int n) => new float[n];

    /// <summary>
    /// MaxDOP sweep for the managed-SimdGemm primitives (MHA, LSTM): do they
    /// oversubscribe threads on these small-batch shapes the way OpenBLAS did
    /// for MLP? If a low cap wins, it's a cheap production lever. Run:
    /// --ab-aiseval-dopsweep.
    /// </summary>
    /// <summary>
    /// Raw GEMM (TensorMatMul) multi-thread SCALING sweep at DiT/transformer sizes, with a native
    /// TorchSharp (libtorch/MKL) reference. Roots the diffusion model-family CPU test timeouts: the
    /// managed BLAS scales poorly + trails native. Run: --ab-gemm-dop.
    /// </summary>
    // Time-budgeted MIN-of-N timer: warms up, then takes the BEST (min) of 8 rounds — the round
    // that caught the highest boost / least OS-jitter. On a 64-core Threadripper the all-core clock
    // throttles vs the 1-thread boost, and the box has run-to-run variance; min-of-N makes the GEMM
    // A/B reproducible (a single 10-rep timing swung ~2x run-to-run). Adaptive reps per round so a
    // 100ms square-2048 op and a 1ms attn-proj op both get fair coverage.
    private static double MinMs(Action op)
    {
        var w = Stopwatch.StartNew();
        while (w.ElapsedMilliseconds < 100) op();           // warm caches + boost
        double best = double.PositiveInfinity;
        for (int r = 0; r < 8; r++)
        {
            var sw = Stopwatch.StartNew();
            int reps = 0;
            do { op(); reps++; } while (sw.Elapsed.TotalMilliseconds < 40);
            sw.Stop();
            best = Math.Min(best, sw.Elapsed.TotalMilliseconds / reps);
        }
        return best;
    }

    public static void GemmDopSweep()
    {
        torch.set_grad_enabled(false);
        int P = Environment.ProcessorCount;
        Console.WriteLine($"=== GEMM gap decomposition: managed BLAS vs native MKL (cores={P}) ===");
        Console.WriteLine("GFLOP/s, min-of-8 time-budgeted. scal = Nthread/1thread. gap = MKL/managed.");
        var engine = new CpuEngine();
        var sizes = new (int M, int K, int N, string tag)[]
        {
            (256, 1152, 1152, "attn-proj"),
            (256, 1152, 4608, "mlp-fc"),
            (1024, 1024, 1024, "square1024"),
            (2048, 2048, 2048, "square2048"),
        };
        int saved = CpuParallelSettings.MaxDegreeOfParallelism;
        try
        {
            Console.WriteLine($"{"shape",-11}{"mgd1t",8}{"mgdNt",8}{"mgdScal",9}{"mkl1t",8}{"mklNt",8}{"mklScal",9}{"gap1t",7}{"gapNt",7}");
            foreach (var (M, K, N, tag) in sizes)
            {
                var a = Tensor<float>.CreateRandom(M, K);
                var b = Tensor<float>.CreateRandom(K, N);
                double gf = 2.0 * M * K * N / 1e9;
                double Mgd(int dop) { CpuParallelSettings.MaxDegreeOfParallelism = dop; return gf / (MinMs(() => { var _ = engine.TensorMatMul(a, b); }) / 1000); }
                double m1 = Mgd(1), mN = Mgd(P);
                double k1 = 0, kN = 0;
                try
                {
                    var ta = torch.rand(M, K); var tb = torch.rand(K, N);
                    double Mkl(int t) { torch.set_num_threads(t); return gf / (MinMs(() => { using var _ = torch.matmul(ta, tb); }) / 1000); }
                    k1 = Mkl(1); kN = Mkl(P);
                }
                catch (Exception e) { Console.WriteLine($"  torch ref failed: {e.Message}"); }
                Console.WriteLine($"{tag,-11}{m1,8:F1}{mN,8:F1}{mN / m1,8:F1}x{k1,8:F1}{kN,8:F1}{kN / k1,8:F1}x{(m1 > 0 ? k1 / m1 : 0),6:F1}x{(mN > 0 ? kN / mN : 0),6:F1}x");
            }
        }
        finally { CpuParallelSettings.MaxDegreeOfParallelism = saved; }
    }

    /// <summary>
    /// Blocking sweep: for each shape, override (Mc,Nc,Kc) via AutotuneDispatcher.BlockOverride
    /// (set on the main thread where Decide runs) and measure all-core GFLOP/s, to find the
    /// blocking that maximizes scaling — the dominant gap vs MKL. Run: --ab-gemm-block.
    /// </summary>
    /// <summary>
    /// Thread-count sweep (in-process, min-of-8): find the DOP that maximizes all-core GEMM on this
    /// 64-core/128-thread chip. If best DOP &lt; 128, the parallel framework is over-subscribing
    /// (SMT/memory contention) — capping threads would be a free scaling win. Run: --ab-gemm-dopfine.
    /// </summary>
    /// <summary>
    /// Pack-vs-kernel CPU-time split (PackBothProfiler) at DOP=all, to quantify the framework
    /// overhead. KernelTicks/PackTicks are summed over worker threads. High pack% = the lever.
    /// Run: --ab-gemm-pack.
    /// </summary>
    public static void GemmPackProfile()
    {
        int P = Environment.ProcessorCount;
        CpuParallelSettings.MaxDegreeOfParallelism = P;
        Console.WriteLine($"=== GEMM pack-vs-kernel CPU-time split @ {P} cores (sum over threads) ===");
        var engine = new CpuEngine();
        var sizes = new (int M, int K, int N, string tag)[]
        {
            (256, 1152, 1152, "attn-proj"),
            (256, 1152, 4608, "mlp-fc"),
            (2048, 2048, 2048, "square2048"),
        };
        foreach (var (M, K, N, tag) in sizes)
        {
            var a = Tensor<float>.CreateRandom(M, K);
            var b = Tensor<float>.CreateRandom(K, N);
            for (int i = 0; i < 5; i++) { var _ = engine.TensorMatMul(a, b); }
            AiDotNet.Tensors.Engines.BlasManaged.PackBothProfiler.Reset();
            AiDotNet.Tensors.Engines.BlasManaged.PackBothProfiler.Enabled = true;
            int reps = 50;
            for (int i = 0; i < reps; i++) { var _ = engine.TensorMatMul(a, b); }
            AiDotNet.Tensors.Engines.BlasManaged.PackBothProfiler.Enabled = false;
            double pa = AiDotNet.Tensors.Engines.BlasManaged.PackBothProfiler.PackAMs;
            double pb = AiDotNet.Tensors.Engines.BlasManaged.PackBothProfiler.PackBMs;
            double kr = AiDotNet.Tensors.Engines.BlasManaged.PackBothProfiler.KernelMs;
            double tot = pa + pb + kr; if (tot <= 0) tot = 1;
            Console.WriteLine($"{tag,-11} packA={pa / tot * 100,5:F1}%  packB={pb / tot * 100,5:F1}%  kernel={kr / tot * 100,5:F1}%   (kernel={kr:F0}ms packB={pb:F0}ms packA={pa:F0}ms, {reps} reps)");
        }
    }

    /// <summary>
    /// Single-tile FP32 microkernel bakeoff: 6x16 vs 4x24 register blocking, emitted machine code,
    /// verified correct vs a naive reference, then timed in isolation (hot-L1, no pack/parallel).
    /// Tests the Zen2 load-port hypothesis (4x24 = 4 broadcasts vs 6x16 = 6). Run: --ab-gemm-tile.
    /// </summary>
    public static unsafe void GemmTileBakeoff()
    {
        Console.WriteLine("=== FP32 single-tile microkernel bakeoff (6x16 vs 4x24) ===");
        var blockings = new (int mr, int nrYmm, string tag)[] { (6, 2, "6x16"), (4, 3, "4x24") };
        const int K = 512;
        var rng = new Random(7);
        foreach (var (mr, nrYmm, tag) in blockings)
        {
            int nr = nrYmm * 8;
            var A = new float[mr * K]; var B = new float[K * nr];
            for (int i = 0; i < A.Length; i++) A[i] = (float)(rng.NextDouble() - 0.5);
            for (int i = 0; i < B.Length; i++) B[i] = (float)(rng.NextDouble() - 0.5);
            var Apk = new float[K * mr]; var Bpk = new float[K * nr];
            for (int row = 0; row < mr; row++) for (int k = 0; k < K; k++) Apk[k * mr + row] = A[row * K + k];
            for (int k = 0; k < K; k++) for (int col = 0; col < nr; col++) Bpk[k * nr + col] = B[k * nr + col];
            var Cref = new float[mr * nr];
            for (int row = 0; row < mr; row++)
                for (int col = 0; col < nr; col++)
                {
                    double s = 0; for (int k = 0; k < K; k++) s += (double)A[row * K + k] * B[k * nr + col];
                    Cref[row * nr + col] = (float)s;
                }
            var C = new float[mr * nr];
            var code = AiDotNet.Tensors.Engines.BlasManaged.MachineCodeFmaKernel.EmitFp32TileWindows(mr, nrYmm);
            using var mem = AiDotNet.Tensors.Engines.BlasManaged.ExecutableMemory.TryAllocate(code);
            if (mem is null || mem.Pointer == IntPtr.Zero) { Console.WriteLine($"{tag}: exec alloc failed"); continue; }
            var kern = (delegate* unmanaged<float*, float*, float*, long, long, void>)(void*)mem.Pointer;
            Array.Clear(C, 0, C.Length);
            fixed (float* pa = Apk, pb = Bpk, pc = C) kern(pa, pb, pc, nr * 4L, K);
            double maxErr = 0, maxRef = 0;
            for (int i = 0; i < C.Length; i++) { maxErr = Math.Max(maxErr, Math.Abs(C[i] - Cref[i])); maxRef = Math.Max(maxRef, Math.Abs(Cref[i])); }
            double relErr = maxRef > 0 ? maxErr / maxRef : maxErr;
            bool ok = relErr < 1e-4;
            double gf = 2.0 * mr * nr * K / 1e9;
            double best = double.PositiveInfinity;
            fixed (float* pa = Apk, pb = Bpk, pc = C)
            {
                for (int w = 0; w < 3000; w++) kern(pa, pb, pc, nr * 4L, K);
                for (int r = 0; r < 8; r++)
                {
                    var swr = Stopwatch.StartNew(); int reps = 0;
                    do { kern(pa, pb, pc, nr * 4L, K); reps++; } while (swr.Elapsed.TotalMilliseconds < 30);
                    swr.Stop(); best = Math.Min(best, swr.Elapsed.TotalMilliseconds / reps);
                }
            }
            Console.WriteLine($"{tag,-6} correct={ok} (relErr {relErr:E2})  single-tile {gf / (best / 1000),6:F1} GFLOP/s  ({mr} bcast+{nrYmm} load /12 FMA)");
        }
    }

    /// <summary>
    /// GotoGemmFp32 rewrite: single-thread correctness (vs naive ref, incl. an unaligned shape that
    /// exercises the M/N tails) + single-thread GFLOP/s. Run: --ab-goto-gemm.
    /// </summary>
    public static unsafe void GotoGemmBench()
    {
        Console.WriteLine("=== GotoGemmFp32 (rewrite) single-thread correctness + perf ===");
        if (!AiDotNet.Tensors.Engines.BlasManaged.GotoGemmFp32.IsAvailable) { Console.WriteLine("GotoGemmFp32 not available on this CPU/OS"); return; }
        int mc = AiDotNet.Tensors.Engines.BlasManaged.GotoGemmFp32.DefaultMc;
        int nc = AiDotNet.Tensors.Engines.BlasManaged.GotoGemmFp32.DefaultNc;
        int kc = AiDotNet.Tensors.Engines.BlasManaged.GotoGemmFp32.DefaultKc;
        var rng = new Random(11);
        var shapes = new (int M, int N, int K, string tag)[]
        {
            (252, 1152, 1152, "attn-proj"),
            (250, 1150, 1150, "unaligned"),
            (1020, 1024, 1024, "square1024"),
            (2046, 2048, 2048, "square2048"),
        };
        foreach (var (M, N, K, tag) in shapes)
        {
            var A = new float[(long)M * K]; var B = new float[(long)K * N]; var C = new float[(long)M * N];
            for (int i = 0; i < A.Length; i++) A[i] = (float)(rng.NextDouble() - 0.5);
            for (int i = 0; i < B.Length; i++) B[i] = (float)(rng.NextDouble() - 0.5);
            fixed (float* pa = A, pb = B, pc = C)
                AiDotNet.Tensors.Engines.BlasManaged.GotoGemmFp32.RunSingle(pa, K, pb, N, pc, N, M, N, K, mc, nc, kc);
            var rv = new Random(5); double maxRel = 0;
            for (int t = 0; t < 200; t++)
            {
                int r = rv.Next(M), col = rv.Next(N);
                double s = 0; for (int k = 0; k < K; k++) s += (double)A[(long)r * K + k] * B[(long)k * N + col];
                double err = Math.Abs(C[(long)r * N + col] - s);
                maxRel = Math.Max(maxRel, Math.Abs(s) > 1e-3 ? err / Math.Abs(s) : err);
            }
            bool ok = maxRel < 1e-3;
            double gf = 2.0 * M * N * K / 1e9, best = double.PositiveInfinity;
            fixed (float* pa = A, pb = B, pc = C)
            {
                var w = Stopwatch.StartNew(); while (w.ElapsedMilliseconds < 100) AiDotNet.Tensors.Engines.BlasManaged.GotoGemmFp32.RunSingle(pa, K, pb, N, pc, N, M, N, K, mc, nc, kc);
                for (int rr = 0; rr < 6; rr++)
                {
                    var sw = Stopwatch.StartNew(); int reps = 0;
                    do { AiDotNet.Tensors.Engines.BlasManaged.GotoGemmFp32.RunSingle(pa, K, pb, N, pc, N, M, N, K, mc, nc, kc); reps++; } while (sw.Elapsed.TotalMilliseconds < 40);
                    sw.Stop(); best = Math.Min(best, sw.Elapsed.TotalMilliseconds / reps);
                }
            }
            Console.WriteLine($"{tag,-11} correct={ok} (maxRel {maxRel:E2})  single-thread {gf / (best / 1000),6:F1} GFLOP/s");
        }
    }

    /// <summary>
    /// GotoGemmFp32 PARALLEL (the rewrite) vs native MKL, all cores. Correctness-checked + min-of-8.
    /// The decisive measurement: does the L2-resident tile-grid decomposition beat the existing
    /// 10-27%-of-MKL ceiling? Run: --ab-goto-par.
    /// </summary>
    public static unsafe void GotoGemmParBench()
    {
        torch.set_grad_enabled(false);
        int P = Environment.ProcessorCount;
        Console.WriteLine($"=== GotoGemmFp32 PARALLEL block sweep vs MKL (cores={P}) ===");
        if (!AiDotNet.Tensors.Engines.BlasManaged.GotoGemmFp32.IsAvailable) { Console.WriteLine("not available"); return; }
        int kc = AiDotNet.Tensors.Engines.BlasManaged.GotoGemmFp32.DefaultKc;
        int saved = CpuParallelSettings.MaxDegreeOfParallelism;
        CpuParallelSettings.MaxDegreeOfParallelism = P;
        var engine = new CpuEngine();
        var rng = new Random(13);
        var shapes = new (int M, int N, int K, string tag)[]
        {
            (256, 1152, 1152, "attn-proj"), (256, 1152, 4608, "mlp-fc"),
            (1024, 1024, 1024, "square1024"), (2048, 2048, 2048, "square2048"),
        };
        // Fixed candidate defaults (not a full sweep): find the best all-rounder to wire as the default.
        var cands = new (int mc, int nc)[] { (120, 128), (96, 128), (120, 256), (192, 256), (96, 96), (240, 256) };
        try
        {
            // Warm the chip to steady all-core thermal state so MKL and ours are measured under the
            // SAME boost/throttle conditions (this box swings ~4x cold-vs-warm — measure fairly).
            torch.set_num_threads(P);
            { var wa = torch.rand(2048, 2048); var wb = torch.rand(2048, 2048);
              var ww = Stopwatch.StartNew(); while (ww.ElapsedMilliseconds < 1500) { using var _ = torch.matmul(wa, wb); } }

            foreach (var (M, N, K, tag) in shapes)
            {
                var A = new float[(long)M * K]; var B = new float[(long)K * N]; var C = new float[(long)M * N];
                for (int i = 0; i < A.Length; i++) A[i] = (float)(rng.NextDouble() - 0.5);
                for (int i = 0; i < B.Length; i++) B[i] = (float)(rng.NextDouble() - 0.5);
                double gf = 2.0 * M * N * K / 1e9;
                double bestGf = 0; int bMc = 0, bNc = 0; string bCorrect = "?";
                fixed (float* pa = A, pb = B, pc = C)
                {
                    foreach (var (mc, nc) in cands)
                    {
                        AiDotNet.Tensors.Engines.BlasManaged.GotoGemmFp32.RunParallel(pa, K, pb, N, pc, N, M, N, K, mc, nc, kc);
                        double maxRel = 0; var rv = new Random(5);
                        for (int t = 0; t < 60; t++)
                        {
                            int r = rv.Next(M), col = rv.Next(N);
                            double s = 0; for (int kk = 0; kk < K; kk++) s += (double)A[(long)r * K + kk] * B[(long)kk * N + col];
                            double err = Math.Abs(C[(long)r * N + col] - s);
                            maxRel = Math.Max(maxRel, Math.Abs(s) > 1e-3 ? err / Math.Abs(s) : err);
                        }
                        double gbest = double.PositiveInfinity;
                        var w = Stopwatch.StartNew(); while (w.ElapsedMilliseconds < 30) AiDotNet.Tensors.Engines.BlasManaged.GotoGemmFp32.RunParallel(pa, K, pb, N, pc, N, M, N, K, mc, nc, kc);
                        for (int r = 0; r < 4; r++)
                        {
                            var sw = Stopwatch.StartNew(); int reps = 0;
                            do { AiDotNet.Tensors.Engines.BlasManaged.GotoGemmFp32.RunParallel(pa, K, pb, N, pc, N, M, N, K, mc, nc, kc); reps++; } while (sw.Elapsed.TotalMilliseconds < 30);
                            sw.Stop(); gbest = Math.Min(gbest, sw.Elapsed.TotalMilliseconds / reps);
                        }
                        double g = gf / (gbest / 1000);
                        Console.WriteLine($"    cand mc={mc,4} nc={nc,4}: {g,7:F0} GF  {(maxRel < 1e-3 ? "OK" : "WRONG " + maxRel.ToString("E1"))}");
                        if (g > bestGf) { bestGf = g; bMc = mc; bNc = nc; bCorrect = maxRel < 1e-3 ? "OK" : "WRONG " + maxRel.ToString("E1"); }
                    }
                }
                // Thermally-fair three-way: alternate existing(engine) / ours(best) / MKL back-to-back, min-of-10 each.
                var ta = torch.rand(M, K); var tb = torch.rand(K, N);
                var ea = Tensor<float>.CreateRandom(M, K); var eb = Tensor<float>.CreateRandom(K, N);
                CpuParallelSettings.MaxDegreeOfParallelism = P;
                double oursMs = double.PositiveInfinity, mklMs = double.PositiveInfinity, exMs = double.PositiveInfinity;
                fixed (float* pa = A, pb = B, pc = C)
                {
                    for (int r = 0; r < 10; r++)
                    {
                        { var sw = Stopwatch.StartNew(); int reps = 0; do { var _ = engine.TensorMatMul(ea, eb); reps++; } while (sw.Elapsed.TotalMilliseconds < 30); sw.Stop(); exMs = Math.Min(exMs, sw.Elapsed.TotalMilliseconds / reps); }
                        { var sw = Stopwatch.StartNew(); int reps = 0; do { AiDotNet.Tensors.Engines.BlasManaged.GotoGemmFp32.RunParallel(pa, K, pb, N, pc, N, M, N, K, bMc, bNc, kc); reps++; } while (sw.Elapsed.TotalMilliseconds < 30); sw.Stop(); oursMs = Math.Min(oursMs, sw.Elapsed.TotalMilliseconds / reps); }
                        { var sw = Stopwatch.StartNew(); int reps = 0; do { using var _ = torch.matmul(ta, tb); reps++; } while (sw.Elapsed.TotalMilliseconds < 30); sw.Stop(); mklMs = Math.Min(mklMs, sw.Elapsed.TotalMilliseconds / reps); }
                    }
                }
                double oursGf = gf / (oursMs / 1000), mklGf = gf / (mklMs / 1000), exGf = gf / (exMs / 1000);
                int tiles = bMc > 0 ? ((M + bMc - 1) / bMc) * ((N + bNc - 1) / bNc) : 0;
                Console.WriteLine($"{tag,-11} exist={exGf,7:F0}  ours={oursGf,7:F0} (mc={bMc,4} nc={bNc,4} {(P>0?(double)tiles/P:0),3:F1}xP)  MKL={mklGf,7:F0}   ours/exist={(exGf>0?oursGf/exGf:0),4:F2}x  ours/MKL={(mklGf > 0 ? oursGf / mklGf * 100 : 0),4:F0}%  {bCorrect}");
            }
        }
        finally { CpuParallelSettings.MaxDegreeOfParallelism = saved; }
    }

    /// <summary>
    /// #19 bf16-B vs fp32 GEMM (the bandwidth doubler): B stored bf16 (half the bytes), A fp32, fp32
    /// accumulate. Measures whether halving the re-read operand's bytes beats fp32 — and MKL's fp32 — on
    /// the memory-bound squares. Reports bf16/fp32, bf16/MKL, and the bf16 rounding error. Run: --ab-bf16.
    /// </summary>
    public static unsafe void GotoBf16Bench()
    {
        torch.set_grad_enabled(false);
        int P = Environment.ProcessorCount;
        if (!AiDotNet.Tensors.Engines.BlasManaged.GotoGemmFp32.IsAvailable) { Console.WriteLine("not available"); return; }
        int saved = CpuParallelSettings.MaxDegreeOfParallelism;
        CpuParallelSettings.MaxDegreeOfParallelism = P;
        var rng = new Random(13);
        var shapes = new (int M, int N, int K, string tag)[] { (1024, 1024, 1024, "sq1024"), (2048, 2048, 2048, "sq2048"), (4096, 4096, 4096, "sq4096") };
        Console.WriteLine($"=== bf16-B vs fp32 GEMM (bandwidth doubler), cores={P} ===");
        try
        {
            torch.set_num_threads(P);
            { var wa = torch.rand(2048, 2048); var wb = torch.rand(2048, 2048); var ww = Stopwatch.StartNew(); while (ww.ElapsedMilliseconds < 1500) { using var _ = torch.matmul(wa, wb); } }
            foreach (var (M, N, K, tag) in shapes)
            {
                var A = new float[(long)M * K]; var B = new float[(long)K * N]; var Cf = new float[(long)M * N]; var Cb = new float[(long)M * N];
                for (int i = 0; i < A.Length; i++) A[i] = (float)(rng.NextDouble() - 0.5);
                for (int i = 0; i < B.Length; i++) B[i] = (float)(rng.NextDouble() - 0.5);
                var (mc, nc, kc) = AiDotNet.Tensors.Engines.BlasManaged.GotoGemmFp32.ChooseParallelBlocks(M, N);
                double gf = 2.0 * M * N * K / 1e9, maxRel = 0, f32 = double.PositiveInfinity, bf16 = double.PositiveInfinity, mkl = double.PositiveInfinity;
                var ta = torch.rand(M, K); var tb = torch.rand(K, N);
                fixed (float* pa = A, pb = B, pcf = Cf, pcb = Cb)
                {
                    AiDotNet.Tensors.Engines.BlasManaged.GotoGemmFp32.RunParallel(pa, K, pb, N, pcf, N, M, N, K, mc, nc, kc);
                    AiDotNet.Tensors.Engines.BlasManaged.GotoGemmFp32.RunParallelBf16(pa, K, pb, N, pcb, N, M, N, K, mc, nc, kc);
                    // Frobenius-relative error ||Cb-Cf||/||Cf|| — per-element relative blows up on near-zero dot
                    // products (random A,B cancel to ~0), so it's a bogus accuracy metric for GEMM.
                    double sumE2 = 0, sumC2 = 0;
                    for (long i = 0; i < (long)M * N; i += 7) { double e = (double)Cb[i] - Cf[i]; sumE2 += e * e; sumC2 += (double)Cf[i] * Cf[i]; }
                    maxRel = sumC2 > 0 ? Math.Sqrt(sumE2 / sumC2) : 0;
                    for (int r = 0; r < 8; r++)
                    {
                        { var sw = Stopwatch.StartNew(); int n = 0; do { AiDotNet.Tensors.Engines.BlasManaged.GotoGemmFp32.RunParallel(pa, K, pb, N, pcf, N, M, N, K, mc, nc, kc); n++; } while (sw.Elapsed.TotalMilliseconds < 30); sw.Stop(); f32 = Math.Min(f32, sw.Elapsed.TotalMilliseconds / n); }
                        { var sw = Stopwatch.StartNew(); int n = 0; do { AiDotNet.Tensors.Engines.BlasManaged.GotoGemmFp32.RunParallelBf16(pa, K, pb, N, pcb, N, M, N, K, mc, nc, kc); n++; } while (sw.Elapsed.TotalMilliseconds < 30); sw.Stop(); bf16 = Math.Min(bf16, sw.Elapsed.TotalMilliseconds / n); }
                        { var sw = Stopwatch.StartNew(); int n = 0; do { using var _ = torch.matmul(ta, tb); n++; } while (sw.Elapsed.TotalMilliseconds < 30); sw.Stop(); mkl = Math.Min(mkl, sw.Elapsed.TotalMilliseconds / n); }
                    }
                }
                double f32Gf = gf / (f32 / 1000), bf16Gf = gf / (bf16 / 1000), mklGf = gf / (mkl / 1000);
                Console.WriteLine($"{tag,-8} fp32={f32Gf,7:F0}  bf16={bf16Gf,7:F0}  MKL={mklGf,7:F0}   bf16/fp32={(f32Gf > 0 ? bf16Gf / f32Gf : 0),4:F2}x  bf16/MKL={(mklGf > 0 ? bf16Gf / mklGf * 100 : 0),4:F0}%   bf16-relErr={maxRel:E2}");
            }
        }
        finally { CpuParallelSettings.MaxDegreeOfParallelism = saved; }
    }

    /// <summary>
    /// #23 one-level Strassen vs standard fp32 GEMM vs MKL on large squares. 7 sub-multiplies instead of 8
    /// (~12.5% fewer multiply-FLOPs) to attack the all-core AVX2 FMA-throttle ceiling (the binding
    /// constraint per #19). Reports str/fp32, str/MKL, and the Frobenius accuracy hit. Run: --ab-strassen.
    /// </summary>
    public static unsafe void GotoStrassenBench()
    {
        torch.set_grad_enabled(false);
        int P = Environment.ProcessorCount;
        if (!AiDotNet.Tensors.Engines.BlasManaged.GotoGemmFp32.IsAvailable) { Console.WriteLine("not available"); return; }
        int saved = CpuParallelSettings.MaxDegreeOfParallelism;
        CpuParallelSettings.MaxDegreeOfParallelism = P;
        var rng = new Random(13);
        var shapes = new (int n, string tag)[] { (2048, "sq2048"), (4096, "sq4096") };
        Console.WriteLine($"=== one-level Strassen vs fp32 vs MKL, cores={P} ===");
        try
        {
            torch.set_num_threads(P);
            { var wa = torch.rand(2048, 2048); var wb = torch.rand(2048, 2048); var ww = Stopwatch.StartNew(); while (ww.ElapsedMilliseconds < 1500) { using var _ = torch.matmul(wa, wb); } }
            foreach (var (n, tag) in shapes)
            {
                var A = new float[(long)n * n]; var B = new float[(long)n * n]; var Cf = new float[(long)n * n]; var Cs = new float[(long)n * n];
                for (int i = 0; i < A.Length; i++) A[i] = (float)(rng.NextDouble() - 0.5);
                for (int i = 0; i < B.Length; i++) B[i] = (float)(rng.NextDouble() - 0.5);
                var (mc, nc, kc) = AiDotNet.Tensors.Engines.BlasManaged.GotoGemmFp32.ChooseParallelBlocks(n, n);
                double gf = 2.0 * n * n * n / 1e9, relErr, f32 = double.PositiveInfinity, str = double.PositiveInfinity, mkl = double.PositiveInfinity;
                var ta = torch.rand(n, n); var tb = torch.rand(n, n);
                fixed (float* pa = A, pb = B, pcf = Cf, pcs = Cs)
                {
                    AiDotNet.Tensors.Engines.BlasManaged.GotoGemmFp32.RunParallel(pa, n, pb, n, pcf, n, n, n, n, mc, nc, kc);
                    AiDotNet.Tensors.Engines.BlasManaged.StrassenGemm.RunSquare(pa, n, pb, n, pcs, n, n);
                    double sumE2 = 0, sumC2 = 0;
                    for (long i = 0; i < (long)n * n; i += 7) { double e = (double)Cs[i] - Cf[i]; sumE2 += e * e; sumC2 += (double)Cf[i] * Cf[i]; }
                    relErr = sumC2 > 0 ? Math.Sqrt(sumE2 / sumC2) : 0;
                    for (int r = 0; r < 8; r++)
                    {
                        { var sw = Stopwatch.StartNew(); int q = 0; do { AiDotNet.Tensors.Engines.BlasManaged.GotoGemmFp32.RunParallel(pa, n, pb, n, pcf, n, n, n, n, mc, nc, kc); q++; } while (sw.Elapsed.TotalMilliseconds < 30); sw.Stop(); f32 = Math.Min(f32, sw.Elapsed.TotalMilliseconds / q); }
                        { var sw = Stopwatch.StartNew(); int q = 0; do { AiDotNet.Tensors.Engines.BlasManaged.StrassenGemm.RunSquare(pa, n, pb, n, pcs, n, n); q++; } while (sw.Elapsed.TotalMilliseconds < 30); sw.Stop(); str = Math.Min(str, sw.Elapsed.TotalMilliseconds / q); }
                        { var sw = Stopwatch.StartNew(); int q = 0; do { using var _ = torch.matmul(ta, tb); q++; } while (sw.Elapsed.TotalMilliseconds < 30); sw.Stop(); mkl = Math.Min(mkl, sw.Elapsed.TotalMilliseconds / q); }
                    }
                }
                double f32Gf = gf / (f32 / 1000), strGf = gf / (str / 1000), mklGf = gf / (mkl / 1000);
                Console.WriteLine($"{tag,-8} fp32={f32Gf,7:F0}  strassen={strGf,7:F0}  MKL={mklGf,7:F0}   str/fp32={(f32Gf > 0 ? strGf / f32Gf : 0),4:F2}x  str/MKL={(mklGf > 0 ? strGf / mklGf * 100 : 0),4:F0}%   str-relErr={relErr:E2}");
            }
        }
        finally { CpuParallelSettings.MaxDegreeOfParallelism = saved; }
    }

    /// <summary>
    /// Tight engine.TensorMatMul loop for an external profiler (PerfView / dotnet-trace) to find where the
    /// per-call time ACTUALLY goes (GEMM kernel vs output alloc vs dispatch vs GC) — no guessing. Shape via
    /// env PROFILE_SHAPE (sq1024 default; dit-attnout / dit-mlp2 / sq2048). Runs PROFILE_SECONDS (default 20).
    /// Run: --profile-gemm.
    /// </summary>
    public static void ProfileGemm()
    {
        var engine = new CpuEngine();
        CpuParallelSettings.MaxDegreeOfParallelism = int.TryParse(Environment.GetEnvironmentVariable("DOP"), out var dop) ? dop : Environment.ProcessorCount;
        int M = 1024, N = 1024, K = 1024;
        switch (Environment.GetEnvironmentVariable("PROFILE_SHAPE"))
        {
            case "dit-attnout": M = 256; N = 1152; K = 1152; break;
            case "dit-mlp2": M = 256; N = 1152; K = 4608; break;
            case "dit-qkv": M = 256; N = 3456; K = 1152; break;
            case "sq2048": M = 2048; N = 2048; K = 2048; break;
            case "sq768": M = 768; N = 768; K = 768; break;
        }
        double secs = double.TryParse(Environment.GetEnvironmentVariable("PROFILE_SECONDS"), out var s) ? s : 20;
        if (int.TryParse(Environment.GetEnvironmentVariable("KC"), out var kcv)) AiDotNet.Tensors.Engines.BlasManaged.GotoGemmFp32.s_kcOverride = kcv;
        var ea = Tensor<float>.CreateRandom(M, K); var eb = Tensor<float>.CreateRandom(K, N);
        for (int i = 0; i < 20; i++) { var _ = engine.TensorMatMul(ea, eb); } // warm
        Console.WriteLine($"PROFILING engine.TensorMatMul [{M}x{N}x{K}] for {secs}s — attach profiler now");
        AiDotNet.Tensors.Engines.BlasManaged.GotoGemmFp32.ResetTiming();
        AiDotNet.Tensors.Engines.BlasManaged.GotoGemmFp32.s_timing = true;
        var sw = Stopwatch.StartNew(); long n = 0;
        while (sw.Elapsed.TotalSeconds < secs) { var _ = engine.TensorMatMul(ea, eb); n++; }
        AiDotNet.Tensors.Engines.BlasManaged.GotoGemmFp32.s_timing = false;
        Console.WriteLine($"done: {n} matmuls in {sw.Elapsed.TotalSeconds:F1}s = {n / sw.Elapsed.TotalSeconds:F0}/s, {2.0 * M * N * K / 1e9 * n / sw.Elapsed.TotalSeconds:F0} GFLOP/s");
        Console.WriteLine("RunTile pack-vs-kernel (summed across worker threads):");
        AiDotNet.Tensors.Engines.BlasManaged.GotoGemmFp32.ReportTiming();
    }

    /// <summary>
    /// Validate + benchmark the PRODUCTION engine GEMM path (engine.TensorMatMul → BlasManaged.Gemm →
    /// CcxGemmPool for large balanced shapes). Proves the CCX wiring is BIT-IDENTICAL to the per-tile
    /// fallback (toggling CcxGemmPool.s_disable) — so it's correct + Deterministic-mode safe — then
    /// times CCX vs per-tile vs MKL (warm, interleaved, min-of-10). Run: --ab-prod.
    /// </summary>
    public static void GotoProductionBench()
    {
        torch.set_grad_enabled(false);
        int P = Environment.ProcessorCount;
        var engine = new CpuEngine();
        int saved = CpuParallelSettings.MaxDegreeOfParallelism;
        CpuParallelSettings.MaxDegreeOfParallelism = P;
        Console.WriteLine($"=== PRODUCTION engine GEMM: CCX vs per-tile vs MKL (cores={P}) ===");
        // squares + the DiT-XL/2 forward GEMMs (hidden=1152, the diffusion timeout source): M=tokens(256),
        // QKV proj, attn-out, MLP fc1/fc2. These are what the ARDiffusion model-family tests actually run.
        var shapes = new (int M, int N, int K, string tag)[]
        {
            (768, 768, 768, "sq768"), (1024, 1024, 1024, "sq1024"), (2048, 2048, 2048, "sq2048"),
            (4096, 4096, 4096, "sq4096"),
            (256, 3456, 1152, "dit-qkv"), (256, 1152, 1152, "dit-attnout"),
            (256, 4608, 1152, "dit-mlp1"), (256, 1152, 4608, "dit-mlp2"),
        };
        try
        {
            torch.set_num_threads(P);
            { var wa = torch.rand(2048, 2048); var wb = torch.rand(2048, 2048); var ww = Stopwatch.StartNew(); while (ww.ElapsedMilliseconds < 1500) { using var _ = torch.matmul(wa, wb); } }
            foreach (var (M, N, K, tag) in shapes)
            {
                var ea = Tensor<float>.CreateRandom(M, K); var eb = Tensor<float>.CreateRandom(K, N);
                double gf = 2.0 * M * N * K / 1e9;
                // Correctness: GotoGemm output must be BIT-IDENTICAL to the old (PackBoth) path? No — different
                // kc/order ⇒ different rounding. Instead verify against a sampled FP64 reference dot.
                var ec = engine.TensorMatMul(ea, eb);
                double maxRel = 0; var rv = new Random(5);
                for (int t = 0; t < 60; t++)
                {
                    int rr = rv.Next(M), col = rv.Next(N);
                    double s = 0; for (int kk = 0; kk < K; kk++) s += (double)ea[rr * K + kk] * eb[kk * N + col];
                    double err = Math.Abs(ec[rr * N + col] - s);
                    maxRel = Math.Max(maxRel, Math.Abs(s) > 1e-3 ? err / Math.Abs(s) : err);
                }

                // All through engine.TensorMatMul (same wrapper), interleaved min-of-10: per-tile / CCX-1D /
                // CCX-2D / MKL — the clean A/B that decides the production heuristic (1D vs 2D per shape).
                var prevForce1D = AiDotNet.Tensors.Engines.BlasManaged.CcxGemmPool.s_force1D;
                var ta = torch.rand(M, K); var tb = torch.rand(K, N);
                double ptMs = double.PositiveInfinity, d1Ms = double.PositiveInfinity, d2Ms = double.PositiveInfinity, mklMs = double.PositiveInfinity;
                for (int r = 0; r < 10; r++)
                {
                    AiDotNet.Tensors.Engines.BlasManaged.CcxGemmPool.s_disable = true;
                    { var sw = Stopwatch.StartNew(); int reps = 0; do { var _ = engine.TensorMatMul(ea, eb); reps++; } while (sw.Elapsed.TotalMilliseconds < 30); sw.Stop(); ptMs = Math.Min(ptMs, sw.Elapsed.TotalMilliseconds / reps); }
                    AiDotNet.Tensors.Engines.BlasManaged.CcxGemmPool.s_disable = false;
                    AiDotNet.Tensors.Engines.BlasManaged.CcxGemmPool.s_force1D = true;
                    { var sw = Stopwatch.StartNew(); int reps = 0; do { var _ = engine.TensorMatMul(ea, eb); reps++; } while (sw.Elapsed.TotalMilliseconds < 30); sw.Stop(); d1Ms = Math.Min(d1Ms, sw.Elapsed.TotalMilliseconds / reps); }
                    AiDotNet.Tensors.Engines.BlasManaged.CcxGemmPool.s_force1D = false;
                    { var sw = Stopwatch.StartNew(); int reps = 0; do { var _ = engine.TensorMatMul(ea, eb); reps++; } while (sw.Elapsed.TotalMilliseconds < 30); sw.Stop(); d2Ms = Math.Min(d2Ms, sw.Elapsed.TotalMilliseconds / reps); }
                    { var sw = Stopwatch.StartNew(); int reps = 0; do { using var _ = torch.matmul(ta, tb); reps++; } while (sw.Elapsed.TotalMilliseconds < 30); sw.Stop(); mklMs = Math.Min(mklMs, sw.Elapsed.TotalMilliseconds / reps); }
                }
                AiDotNet.Tensors.Engines.BlasManaged.CcxGemmPool.s_force1D = prevForce1D;
                AiDotNet.Tensors.Engines.BlasManaged.CcxGemmPool.s_disable = false;
                double ptGf = gf / (ptMs / 1000), d1Gf = gf / (d1Ms / 1000), d2Gf = gf / (d2Ms / 1000), mklGf = gf / (mklMs / 1000);
                double best = Math.Max(ptGf, Math.Max(d1Gf, d2Gf));
                Console.WriteLine($"{tag,-9} pertile={ptGf,6:F0}  CCX1D={d1Gf,6:F0}  CCX2D={d2Gf,6:F0}  MKL={mklGf,6:F0}   1D/pt={(ptGf>0?d1Gf/ptGf:0),4:F2}x  2D/pt={(ptGf>0?d2Gf/ptGf:0),4:F2}x  best/MKL={(mklGf>0?best/mklGf*100:0),3:F0}%  {(maxRel < 1e-3 ? "OK" : "WRONG")}");
            }
        }
        finally { CpuParallelSettings.MaxDegreeOfParallelism = saved; }
    }

    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern bool GetLogicalProcessorInformationEx(
        int relationshipType, IntPtr buffer, ref uint returnedLength);

    /// <summary>
    /// CPU topology probe (foundation for the CCX-aware GEMM scheme): enumerate the L3 cache domains
    /// (= CCXs on Zen). Each L3 domain is a set of logical cores that share one last-level cache; the
    /// hierarchical GEMM will pin a thread-group per domain and replicate the packed-B panel into each
    /// domain's L3 so it is reused WITHOUT cross-CCX Infinity-Fabric traffic (the measured 2x-to-MKL gap).
    /// Run: --cpu-topology. On the 3990X expect 16 L3 domains × 8 logical cores (4 phys + SMT), 16 MB each.
    /// </summary>
    public static void CpuTopologyProbe()
    {
        const int RelationCache = 2;
        Console.WriteLine($"=== CPU topology (logical procs={Environment.ProcessorCount}) ===");
        uint len = 0;
        GetLogicalProcessorInformationEx(RelationCache, IntPtr.Zero, ref len); // size query (returns false)
        if (len == 0) { Console.WriteLine("GetLogicalProcessorInformationEx size query returned 0 — unsupported."); return; }
        IntPtr buf = Marshal.AllocHGlobal((int)len);
        try
        {
            if (!GetLogicalProcessorInformationEx(RelationCache, buf, ref len))
            { Console.WriteLine($"GetLogicalProcessorInformationEx failed: {Marshal.GetLastWin32Error()}"); return; }

            // Walk variable-length SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX records.
            // Record: +0 DWORD Relationship, +4 DWORD Size, then CACHE_RELATIONSHIP at +8:
            //   cache+0 Level, cache+4 CacheSize(DWORD), cache+32 GROUP_AFFINITY{ +0 KAFFINITY Mask(8), +8 WORD Group }
            //   → record offsets: Level@8, CacheSize@12, Mask@40, Group@48.
            long ptr = (long)buf, end = ptr + len;
            int l3 = 0, totalCores = 0;
            while (ptr < end)
            {
                int rel = Marshal.ReadInt32((IntPtr)ptr);
                int size = Marshal.ReadInt32((IntPtr)(ptr + 4));
                if (size <= 0) break;
                if (rel == RelationCache)
                {
                    byte level = Marshal.ReadByte((IntPtr)(ptr + 8));
                    if (level == 3)
                    {
                        int cacheBytes = Marshal.ReadInt32((IntPtr)(ptr + 12));
                        long mask = Marshal.ReadInt64((IntPtr)(ptr + 40));
                        short group = Marshal.ReadInt16((IntPtr)(ptr + 48));
                        int cores = System.Numerics.BitOperations.PopCount((ulong)mask);
                        totalCores += cores;
                        Console.WriteLine($"  L3 #{++l3,2}: group={group} cores={cores,2} L3={cacheBytes / (1024 * 1024)}MB mask=0x{(ulong)mask:X16}");
                    }
                }
                ptr += size;
            }
            Console.WriteLine($"Total L3 (CCX) domains: {l3}, summed logical cores: {totalCores}");
        }
        finally { Marshal.FreeHGlobal(buf); }

        // NUMA nodes + processor groups (RelationAll=0xffff): NumaNode=1, Group=4. Decides #18 — if 1 NUMA
        // node (NPS1), NUMA-local allocation is a no-op (uniform DRAM; CCX-L3 pinning is the real locality).
        uint len2 = 0; GetLogicalProcessorInformationEx(0xffff, IntPtr.Zero, ref len2);
        if (len2 == 0) { Console.WriteLine("NUMA/group query unsupported"); return; }
        IntPtr buf2 = Marshal.AllocHGlobal((int)len2);
        try
        {
            if (!GetLogicalProcessorInformationEx(0xffff, buf2, ref len2)) { Console.WriteLine("NUMA/group query failed"); return; }
            long ptr = (long)buf2, end = ptr + len2; int numa = 0, groups = 0, maxGroupCount = 0;
            while (ptr < end)
            {
                int rel = Marshal.ReadInt32((IntPtr)ptr);
                int size = Marshal.ReadInt32((IntPtr)(ptr + 4));
                if (size <= 0) break;
                if (rel == 1) numa++;                 // RelationNumaNode
                else if (rel == 4) { groups++; maxGroupCount = Marshal.ReadInt16((IntPtr)(ptr + 8)); } // RelationGroup → ActiveGroupCount@+8
                ptr += size;
            }
            Console.WriteLine($"NUMA nodes: {numa}   processor-group records: {groups} (ActiveGroupCount={maxGroupCount})");
            Console.WriteLine(numa <= 1
                ? "→ NPS1 / single NUMA node: NUMA-local alloc is a NO-OP; CCX-L3 pinning is the locality lever."
                : $"→ {numa} NUMA nodes: NUMA-local allocation (#18) is worth implementing.");
        }
        finally { Marshal.FreeHGlobal(buf2); }
    }

    public static void GemmDopFine()
    {
        int P = Environment.ProcessorCount;
        Console.WriteLine($"=== GEMM thread-count sweep (optimal DOP) cores={P} ===");
        var engine = new CpuEngine();
        var sizes = new (int M, int K, int N, string tag)[]
        {
            (256, 1152, 1152, "attn-proj"),
            (256, 1152, 4608, "mlp-fc"),
            (1024, 1024, 1024, "square1024"),
            (2048, 2048, 2048, "square2048"),
        };
        int saved = CpuParallelSettings.MaxDegreeOfParallelism;
        int[] dops = { 8, 16, 32, 48, 64, 96, 128 };
        try
        {
            foreach (var (M, K, N, tag) in sizes)
            {
                var a = Tensor<float>.CreateRandom(M, K);
                var b = Tensor<float>.CreateRandom(K, N);
                double gf = 2.0 * M * K * N / 1e9;
                var sb = new System.Text.StringBuilder($"{tag,-11}");
                double best = 0; int bestDop = 0;
                foreach (int d in dops)
                {
                    CpuParallelSettings.MaxDegreeOfParallelism = d;
                    double g = gf / (MinMs(() => { var _ = engine.TensorMatMul(a, b); }) / 1000);
                    sb.Append($" {d}t={g,5:F0}");
                    if (g > best) { best = g; bestDop = d; }
                }
                Console.WriteLine(sb + $"  BEST={best:F0}@{bestDop}t");
            }
        }
        finally { CpuParallelSettings.MaxDegreeOfParallelism = saved; }
    }

    public static void GemmBlockingSweep()
    {
        torch.set_grad_enabled(false);
        int P = Environment.ProcessorCount;
        Console.WriteLine($"=== GEMM blocking sweep (Mc/Nc/Kc via BlockOverride) @ DOP={P} cores ===");
        var engine = new CpuEngine();
        int saved = CpuParallelSettings.MaxDegreeOfParallelism;
        CpuParallelSettings.MaxDegreeOfParallelism = P;
        var sizes = new (int M, int K, int N, string tag, double mkl)[]
        {
            (256, 1152, 1152, "attn-proj", 1305),
            (256, 1152, 4608, "mlp-fc", 1245),
            (1024, 1024, 1024, "square1024", 1107),
            (2048, 2048, 2048, "square2048", 1602),
        };
        int[] mcs = { 18, 60, 120, 192, 256, 384, 512 };
        int[] ncs = { 256, 512, 1024, 2048 };
        int[] kcs = { 128, 256, 512 };
        try
        {
            foreach (var (M, K, N, tag, mkl) in sizes)
            {
                var a = Tensor<float>.CreateRandom(M, K);
                var b = Tensor<float>.CreateRandom(K, N);
                double gf = 2.0 * M * K * N / 1e9;
                double best = 0; (int mc, int nc, int kc) bestcfg = (0, 0, 0);
                foreach (int mc in mcs)
                    foreach (int nc in ncs)
                        foreach (int kc in kcs)
                        {
                            if (mc > M + 64 || nc > N + 64 || kc > K + 64) continue;
                            AiDotNet.Tensors.Engines.BlasManaged.AutotuneDispatcher.BlockOverride = (mc, nc, kc);
                            double g;
                            try { g = gf / (MinMs(() => { var _ = engine.TensorMatMul(a, b); }) / 1000); }
                            finally { AiDotNet.Tensors.Engines.BlasManaged.AutotuneDispatcher.BlockOverride = null; }
                            if (g > best) { best = g; bestcfg = (mc, nc, kc); }
                        }
                Console.WriteLine($"{tag,-11} BEST {best,7:F0} GFLOP/s @ mc={bestcfg.mc} nc={bestcfg.nc} kc={bestcfg.kc}   (MKL {mkl,5:F0}, {best / mkl * 100,3:F0}% of MKL)");
            }
        }
        finally { CpuParallelSettings.MaxDegreeOfParallelism = saved; }
    }

    public static void DopSweep()
    {
        Console.WriteLine("=== Issue #436 — CpuParallelSettings MaxDOP sweep (MHA, LSTM) ===");
        Console.WriteLine($"Host: cores={Environment.ProcessorCount}");
        var engine = new CpuEngine();

        const int batch = 128, seq = 32, dModel = 64, heads = 4;
        var mIn = Tensor<float>.CreateRandom(batch, seq, dModel);
        var qW = Tensor<float>.CreateRandom(dModel, dModel); var kW = Tensor<float>.CreateRandom(dModel, dModel);
        var vW = Tensor<float>.CreateRandom(dModel, dModel); var oW = Tensor<float>.CreateRandom(dModel, dModel);
        Func<Tensor<float>> mha = () => engine.MultiHeadAttentionForward(mIn, qW, kW, vW, oW, heads);

        const int inF = 32, hidden = 64;
        var lIn = Tensor<float>.CreateRandom(batch, seq, inF);
        var wIh = Tensor<float>.CreateRandom(4 * hidden, inF);
        var wHh = Tensor<float>.CreateRandom(4 * hidden, hidden);
        Func<Tensor<float>> lstm = () => engine.LstmSequenceForward(lIn, null, null, wIh, wHh, null, null);

        int saved = AiDotNet.Tensors.Helpers.CpuParallelSettings.MaxDegreeOfParallelism;
        try
        {
            foreach (int dop in new[] { 1, 2, 4, 8, 16 })
            {
                if (dop > Environment.ProcessorCount) break;
                AiDotNet.Tensors.Helpers.CpuParallelSettings.MaxDegreeOfParallelism = dop;
                var (mm, mp) = MeasureRobust(() => mha());
                var (lm, lp) = MeasureRobust(() => lstm());
                Console.WriteLine($"  MaxDOP={dop,2}:  MHA med {mm,7:F3} p95 {mp,7:F3}  |  LSTM med {lm,7:F3} p95 {lp,7:F3}");
            }
        }
        finally { AiDotNet.Tensors.Helpers.CpuParallelSettings.MaxDegreeOfParallelism = saved; }
    }

    private static void ReportDiag(string label, Func<Tensor<float>> forward)
    {
        for (int i = 0; i < Warmup; i++) forward();
        SettleGc();
        long allocStart = GC.GetAllocatedBytesForCurrentThread();
        var times = new double[Iters];
        var sw = new Stopwatch();
        for (int i = 0; i < Iters; i++)
        {
            sw.Restart();
            forward();
            sw.Stop();
            times[i] = sw.Elapsed.TotalMilliseconds;
        }
        double allocPerCall = (GC.GetAllocatedBytesForCurrentThread() - allocStart) / (double)Iters;
        var (med, p95) = Percentiles(times);
        Console.WriteLine($"  {label}  med {med,7:F3}ms  p95 {p95,7:F3}ms  alloc {allocPerCall,10:F0} B/call");
    }

    /// <summary>
    /// PR #531 validation: count StreamingWorkerPool dispatch/park events per primitive
    /// to see whether the low-latency-pool keep-alive even targets these workloads'
    /// critical path. Run: --ab-aiseval-poolstats.
    /// </summary>
    public static void PoolStats()
    {
        Console.WriteLine("=== PR #531 — StreamingWorkerPool usage per AIsEval primitive ===");
        Console.WriteLine($"Host: cores={Environment.ProcessorCount}. Counters: parallel dispatch / serial<grain / serial-contended / park events.");
        Console.WriteLine();
        var engine = new CpuEngine();
        const int reps = 200;

        void Probe(string name, Action run)
        {
            for (int i = 0; i < 20; i++) run(); // warm
            AiDotNet.Tensors.Engines.BlasManaged.Pool.StreamingWorkerPool.ResetPoolStats();
            AiDotNet.Tensors.Helpers.CpuParallelSettings.ResetParallelForStats();
            for (int i = 0; i < reps; i++) run();
            var (par, serG, serC, park) = AiDotNet.Tensors.Engines.BlasManaged.Pool.StreamingWorkerPool.PoolStatsSnapshot();
            long pfor = AiDotNet.Tensors.Helpers.CpuParallelSettings.ParallelForStatsSnapshot();
            Console.WriteLine($"  {name,-12} over {reps} calls: StreamingPool[parallel={par} parks={park}]  vs  Parallel.For dispatches={pfor}");
            Console.WriteLine($"  {name,-12} per call         : StreamingPool parallel={par / (double)reps:F1}  Parallel.For={pfor / (double)reps:F1}");
        }

        bool managedPass = Environment.GetEnvironmentVariable("POOLSTATS_MANAGED") == "1";
        if (managedPass)
        {
            AiDotNet.Tensors.Engines.BlasManaged.BlasManaged.PreferManaged = true;
            Console.WriteLine("  (BlasManaged.PreferManaged = true — forcing GEMMs onto the managed StreamingWorkerPool path)");
        }

        {
            const int bs = 128;
            var input = Tensor<float>.CreateRandom(bs, 784);
            var weights = new List<Tensor<float>> { Tensor<float>.CreateRandom(784, 512), Tensor<float>.CreateRandom(512, 128), Tensor<float>.CreateRandom(128, 10) };
            var biases = new List<Tensor<float>?> { Tensor<float>.CreateRandom(512), Tensor<float>.CreateRandom(128), Tensor<float>.CreateRandom(10) };
            Probe("MLP", () => engine.MlpForward(input, weights, biases, FusedActivationType.ReLU, FusedActivationType.None));
        }
        {
            const int batch = 128, seq = 32, dModel = 64, heads = 4;
            var input = Tensor<float>.CreateRandom(batch, seq, dModel);
            var qW = Tensor<float>.CreateRandom(dModel, dModel); var kW = Tensor<float>.CreateRandom(dModel, dModel);
            var vW = Tensor<float>.CreateRandom(dModel, dModel); var oW = Tensor<float>.CreateRandom(dModel, dModel);
            Probe("Transformer", () => engine.MultiHeadAttentionForward(input, qW, kW, vW, oW, heads));
        }
        {
            const int batch = 128, seq = 32, inF = 32, hidden = 64;
            var input = Tensor<float>.CreateRandom(batch, seq, inF);
            var wIh = Tensor<float>.CreateRandom(4 * hidden, inF); var wHh = Tensor<float>.CreateRandom(4 * hidden, hidden);
            Probe("LSTM", () => engine.LstmSequenceForward(input, null, null, wIh, wHh, null, null));
        }
    }

    /// <summary>
    /// PR #531 validation: per-dispatch latency + allocation of the general parallel-op
    /// path (CpuParallelSettings.ParallelForOrSerial) through raw Parallel.For (the .NET
    /// ThreadPool) vs the low-latency cooperative pool. This is the regime the prototype
    /// targeted — many small back-to-back parallel dispatches. Run: --ab-pool-wiring.
    /// </summary>
    public static void PoolWiringBench()
    {
        Console.WriteLine("=== PR #531 — ParallelForOrSerial: Parallel.For vs cooperative pool ===");
        Console.WriteLine($"Host: cores={Environment.ProcessorCount}");

        // Correctness: the cooperative path must match a serial reference — including
        // when many threads dispatch concurrently (the scheduler's reason to exist).
        {
            var src = new float[20000];
            var r2 = new Random(5);
            for (int i = 0; i < src.Length; i++) src[i] = (float)(r2.NextDouble() * 2 - 1);
            var refOut = new float[src.Length];
            for (int i = 0; i < src.Length; i++) refOut[i] = MathF.Max(0f, src[i] * 2f);

            AiDotNet.Tensors.Helpers.CpuParallelSettings.UseCooperativePool = true;
            var o1 = new float[src.Length];
            AiDotNet.Tensors.Helpers.CpuParallelSettings.ParallelForOrSerial(0, src.Length, src.Length, i => o1[i] = MathF.Max(0f, src[i] * 2f));
            bool single = true;
            for (int i = 0; i < src.Length; i++) if (o1[i] != refOut[i]) single = false;

            // Concurrent: raw threads (not Parallel.For, which would set _inParallelRegion
            // and serialize the nested dispatch) each run their own dispatch simultaneously.
            int badThreads = 0;
            var threads = new System.Threading.Thread[8];
            for (int t = 0; t < threads.Length; t++)
            {
                threads[t] = new System.Threading.Thread(() =>
                {
                    for (int rep = 0; rep < 50; rep++)
                    {
                        var ot = new float[src.Length];
                        AiDotNet.Tensors.Helpers.CpuParallelSettings.ParallelForOrSerial(0, src.Length, src.Length, i => ot[i] = MathF.Max(0f, src[i] * 2f));
                        for (int i = 0; i < src.Length; i++) if (ot[i] != refOut[i]) { System.Threading.Interlocked.Increment(ref badThreads); break; }
                    }
                });
                threads[t].Start();
            }
            foreach (var th in threads) th.Join();
            AiDotNet.Tensors.Helpers.CpuParallelSettings.UseCooperativePool = false;
            Console.WriteLine($"  correctness: single={(single ? "PASS" : "FAIL")}  concurrent(8x50)={(badThreads == 0 ? "PASS" : $"FAIL({badThreads})")}");
        }

        const int rows = 256;
        const long totalWork = 64 * 1024; // above the 32K serial grain → parallel path
        var buf = new float[rows * 256];
        var rng = new Random(1);
        for (int i = 0; i < buf.Length; i++) buf[i] = (float)(rng.NextDouble() * 2 - 1);
        int cols = buf.Length / rows;

        Action op = () => AiDotNet.Tensors.Helpers.CpuParallelSettings.ParallelForOrSerial(0, rows, totalWork, r =>
        {
            int b = r * cols;
            for (int j = 0; j < cols; j++) { var v = buf[b + j]; buf[b + j] = v < 0 ? 0 : v; }
        });

        void Measure(string label, bool coop)
        {
            AiDotNet.Tensors.Helpers.CpuParallelSettings.UseCooperativePool = coop;
            for (int i = 0; i < 500; i++) op(); // warm
            SettleGc();
            long a0 = GC.GetAllocatedBytesForCurrentThread();
            const int n = 4000;
            var times = new double[n];
            var sw = new Stopwatch();
            for (int i = 0; i < n; i++) { sw.Restart(); op(); sw.Stop(); times[i] = sw.Elapsed.TotalMilliseconds * 1000.0; }
            double alloc = (GC.GetAllocatedBytesForCurrentThread() - a0) / (double)n;
            var (med, p95) = Percentiles(times);
            Console.WriteLine($"  {label,-14}: med {med,7:F2}us  p95 {p95,7:F2}us  alloc {alloc,7:F0} B/dispatch");
        }

        Measure("Parallel.For", false);
        Measure("cooperative", true);
        Measure("Parallel.For", false); // re-measure to confirm the delta isn't drift
        Measure("cooperative", true);
        AiDotNet.Tensors.Helpers.CpuParallelSettings.UseCooperativePool = false;
    }

    /// <summary>
    /// The DiT/transformer MLP BLOCK (fc1 K→H, GELU, fc2 H→N) — ours-fused (cached pre-packed weights +
    /// fused activation, the framework advantage MKL can't match: frozen weights pack ONCE) vs the torch
    /// (MKL) sequence (matmul + gelu + matmul, re-packs weights every call). The GEMMs are at MKL's HW
    /// ceiling; the win must come from the weight-pack amortization + the fused activation. Run: --ab-mlp-block.
    /// </summary>
    public static void MlpBlockBench()
    {
        torch.set_grad_enabled(false);
        int P = Environment.ProcessorCount; torch.set_num_threads(P);
        var engine = new CpuEngine();
        var shapes = new (int M, int K, int H, int N, string tag)[]
        {
            (256, 1152, 4608, 1152, "DiT-XL/2 MLP m=256"),
            (1024, 1152, 4608, 1152, "DiT MLP m=1024"),
            (256, 1024, 4096, 1024, "MLP m=256 d=1024"),
        };
        Console.WriteLine($"=== MLP block fc1+GELU+fc2 (SAME data, tanh-gelu, correctness-checked): ours-fused(cached W) vs torch(MKL), cores={P} ===");
        static float[] R(long n, Random r) { var a = new float[n]; for (long i = 0; i < n; i++) a[i] = (float)(r.NextDouble() - 0.5); return a; }
        foreach (var (M, K, H, N, tag) in shapes)
        {
            var rng = new Random(7);
            float[] X = R((long)M * K, rng), W1 = R((long)K * H, rng), b1 = R(H, rng), W2 = R((long)H * N, rng), b2 = R(N, rng);
            float[] Hb = new float[(long)M * H], Y = new float[(long)M * N];
            // torch tensors from the SAME data (so outputs are comparable).
            using var tX = torch.tensor(X).reshape(M, K); using var tW1 = torch.tensor(W1).reshape(K, H); using var tb1 = torch.tensor(b1);
            using var tW2 = torch.tensor(W2).reshape(H, N); using var tb2 = torch.tensor(b2);

            // ours-fused (cached-B + fused tanh-GELU)
            CpuFusedOperations.FusedGemmBiasActivation(X, W1, b1, Hb, M, H, K, FusedActivationType.GELU);
            CpuFusedOperations.FusedGemmBiasActivation(Hb, W2, b2, Y, M, N, H, FusedActivationType.None);
            // reference output for correctness (torch, tanh-gelu to match our approx)
            float relErr;
            using (var h = torch.matmul(tX, tW1).add_(tb1))
            using (var g = torch.nn.functional.gelu(h))
            using (var y = torch.matmul(g, tW2).add_(tb2))
            {
                var yref = y.data<float>().ToArray();
                double se = 0, sr = 0; for (long i = 0; i < (long)M * N; i++) { double e = (double)Y[i] - yref[i]; se += e * e; sr += (double)yref[i] * yref[i]; }
                relErr = (float)(sr > 0 ? Math.Sqrt(se / sr) : 0);
            }

            var (ofMed, _) = MeasureRobust(() =>
            {
                CpuFusedOperations.FusedGemmBiasActivation(X, W1, b1, Hb, M, H, K, FusedActivationType.GELU);
                CpuFusedOperations.FusedGemmBiasActivation(Hb, W2, b2, Y, M, N, H, FusedActivationType.None);
            });
            // torch FULL block + torch MATMUL-ONLY (decompose: is torch slow on matmul or on the glue?),
            // each at its BEST thread count (give MKL its best shot — rule out oversubscription artifacts).
            double tBlock = double.MaxValue, tMM = double.MaxValue;
            foreach (int th in new[] { 32, 64, 128 })
            {
                torch.set_num_threads(th);
                var (b, _) = MeasureRobust(() => { using var h = torch.matmul(tX, tW1).add_(tb1); using var g = torch.nn.functional.gelu(h); using var y = torch.matmul(g, tW2).add_(tb2); });
                var (mm, _) = MeasureRobust(() => { using var h = torch.matmul(tX, tW1); using var y = torch.matmul(h, tW2); });
                tBlock = Math.Min(tBlock, b); tMM = Math.Min(tMM, mm);
            }
            torch.set_num_threads(P);
            Console.WriteLine($"{tag,-20} torch-block={tBlock,7:F3}ms (mm-only={tMM,6:F3})  ours-fused={ofMed,7:F3}ms  ours/torch={ofMed / tBlock,4:F2}x  {(ofMed < tBlock ? "WIN" : "LOSS")}  relErr={relErr:E2}");
        }
    }

    /// <summary>
    /// #24 short-M diagnosis: sweep M at fixed wide-N (the DiT MLP shape N=4608, K=1152) measuring our raw
    /// per-tile GEMM (GotoGemmFp32.RunParallel) vs MKL — to see WHERE throughput falls off vs M (a cliff at
    /// short M = the short-M problem) and quantify the gap. Plus the pack-vs-kernel split at M=256 (is short-M
    /// pack-bound or kernel-bound?). Run: --ab-shortm.
    /// </summary>
    public static unsafe void ShortMBench()
    {
        torch.set_grad_enabled(false);
        int P = Environment.ProcessorCount; torch.set_num_threads(P);
        int N = 4608, K = 1152;
        var Ms = new[] { 64, 128, 256, 512, 1024, 2048 };
        var rng = new Random(7);
        static float[] R(long n, Random r) { var a = new float[n]; for (long i = 0; i < n; i++) a[i] = (float)(r.NextDouble() - 0.5); return a; }
        float[] Bw = R((long)K * N, rng);
        using var tB = torch.tensor(Bw).reshape(K, N);
        { using var wa = torch.randn(1024, K); var ww = Stopwatch.StartNew(); while (ww.ElapsedMilliseconds < 1200) { using var _ = torch.matmul(wa, tB); } }
        Console.WriteLine($"=== short-M sweep N={N} K={K}: ours per-tile (RunParallel) vs MKL, cores={P} ===");
        double Min(Action a, int reps) { double best = double.MaxValue; for (int r = 0; r < reps; r++) { var sw = Stopwatch.StartNew(); int q = 0; do { a(); q++; } while (sw.Elapsed.TotalMilliseconds < 40); sw.Stop(); best = Math.Min(best, sw.Elapsed.TotalMilliseconds / q); } return best; }
        foreach (int M in Ms)
        {
            float[] A = R((long)M * K, rng); float[] C = new float[(long)M * N];
            using var tA = torch.tensor(A).reshape(M, K);
            var (mc, nc, kc) = AiDotNet.Tensors.Engines.BlasManaged.GotoGemmFp32.ChooseParallelBlocks(M, N);
            double gf = 2.0 * M * N * K / 1e9;
            double pt, mkl;
            fixed (float* pa = A, pb = Bw, pc = C)
            {
                var paL = (nint)pa; var pbL = (nint)pb; var pcL = (nint)pc;
                pt = Min(() => AiDotNet.Tensors.Engines.BlasManaged.GotoGemmFp32.RunParallel((float*)paL, K, (float*)pbL, N, (float*)pcL, N, M, N, K, mc, nc, kc), 6);
            }
            mkl = Min(() => { using var _ = torch.matmul(tA, tB); }, 6);
            double ptGf = gf / (pt / 1000), mklGf = gf / (mkl / 1000);
            Console.WriteLine($"M={M,4}: pertile={ptGf,6:F0}GF  MKL={mklGf,6:F0}GF  pt/MKL={ptGf / mklGf * 100,4:F0}%   (mc={mc} nc={nc} kc={kc}, B-panel kc*nc={(long)kc * nc * 4 / 1024}KB)");
        }
        {
            int M = 256; float[] A = R((long)M * K, rng); float[] C = new float[(long)M * N];
            double gf = 2.0 * M * N * K / 1e9;
            // WRAPPER ISOLATION (same run): dit-mlp2 (M=256,K=4608,N=1152) routes to GotoGemm in BOTH the
            // engine and direct paths (BeatsPackBoth: k>=2n) → direct-vs-engine is the pure engine wrapper
            // (output alloc + dispatch + GC), no routing confound. torch is wrapped too (fair).
            {
                int m2 = 256, k2 = 4608, n2 = 1152; double gf2 = 2.0 * m2 * n2 * k2 / 1e9;
                float[] A2 = R((long)m2 * k2, rng), B2 = R((long)k2 * n2, rng), C2 = new float[(long)m2 * n2];
                var (mc2, nc2, kc2) = AiDotNet.Tensors.Engines.BlasManaged.GotoGemmFp32.ChooseParallelBlocks(m2, n2);
                var engine = new CpuEngine();
                var ea = Tensor<float>.CreateRandom(m2, k2); var eb = Tensor<float>.CreateRandom(k2, n2);
                using var t2a = torch.tensor(A2).reshape(m2, k2); using var t2b = torch.tensor(B2).reshape(k2, n2);
                for (int w = 0; w < 5; w++) { var _ = engine.TensorMatMul(ea, eb); using var __ = torch.matmul(t2a, t2b); }
                double dir2; fixed (float* pa = A2, pb = B2, pc = C2) { var paL = (nint)pa; var pbL = (nint)pb; var pcL = (nint)pc; dir2 = Min(() => AiDotNet.Tensors.Engines.BlasManaged.GotoGemmFp32.RunParallel((float*)paL, k2, (float*)pbL, n2, (float*)pcL, n2, m2, n2, k2, mc2, nc2, kc2), 6); }
                double eng2 = Min(() => { var _ = engine.TensorMatMul(ea, eb); }, 6);
                double mkl2 = Min(() => { using var _ = torch.matmul(t2a, t2b); }, 6);
                Console.WriteLine($"--- WRAPPER (dit-mlp2 256x1152x4608, both GotoGemm): direct={gf2 / (dir2 / 1000):F0}GF  engine={gf2 / (eng2 / 1000):F0}GF  MKL={gf2 / (mkl2 / 1000):F0}GF  direct/engine={(dir2 > 0 ? eng2 / dir2 : 0):F2}x ---");
            }
            Console.WriteLine($"--- M=256 blocking sweep (mc x nc, kc=256): tiles=ceil(256/mc)*ceil(4608/nc) vs 128 cores ---");
            foreach (int mcv in new[] { 32, 48, 64, 96, 128 })
                foreach (int ncv in new[] { 64, 128, 256, 512 })
                {
                    long panel = (long)256 * ncv * 4;
                    if (panel > 700 * 1024) continue;
                    int tiles = ((256 + mcv - 1) / mcv) * ((N + ncv - 1) / ncv);
                    double t; fixed (float* pa = A, pb = Bw, pc = C) { var paL = (nint)pa; var pbL = (nint)pb; var pcL = (nint)pc; t = Min(() => AiDotNet.Tensors.Engines.BlasManaged.GotoGemmFp32.RunParallel((float*)paL, K, (float*)pbL, N, (float*)pcL, N, M, N, K, mcv, ncv, 256), 5); }
                    Console.WriteLine($"   mc={mcv,3} nc={ncv,3}: {gf / (t / 1000),6:F0}GF  tiles={tiles,4}  Bpanel={(long)256 * ncv * 4 / 1024}KB");
                }
            var (mc, nc, kc) = AiDotNet.Tensors.Engines.BlasManaged.GotoGemmFp32.ChooseParallelBlocks(M, N);
            AiDotNet.Tensors.Engines.BlasManaged.GotoGemmFp32.s_timing = true;
            AiDotNet.Tensors.Engines.BlasManaged.GotoGemmFp32.ResetTiming();
            fixed (float* pa = A, pb = Bw, pc = C)
                for (int it = 0; it < 50; it++) AiDotNet.Tensors.Engines.BlasManaged.GotoGemmFp32.RunParallel(pa, K, pb, N, pc, N, M, N, K, mc, nc, kc);
            AiDotNet.Tensors.Engines.BlasManaged.GotoGemmFp32.s_timing = false;
            Console.WriteLine($"--- M=256 pack-vs-kernel at default mc={mc} nc={nc} kc={kc} (summed across workers) ---");
            AiDotNet.Tensors.Engines.BlasManaged.GotoGemmFp32.ReportTiming();
        }
        // dit-mlp2 (deep-K narrow-N: 256x1152x4608) — does smaller nc (more tiles) recover the
        // under-parallelization (72 tiles@nc=64), or is split-K needed? Sweep mc/nc; compare to MKL.
        {
            int m2 = 256, n2 = 1152, k2 = 4608; double gf2 = 2.0 * m2 * n2 * k2 / 1e9;
            float[] A2 = R((long)m2 * k2, rng), B2 = R((long)k2 * n2, rng), C2 = new float[(long)m2 * n2];
            using var t2a = torch.tensor(A2).reshape(m2, k2); using var t2b = torch.tensor(B2).reshape(k2, n2);
            double mkl2 = Min(() => { using var _ = torch.matmul(t2a, t2b); }, 6);
            Console.WriteLine($"--- dit-mlp2 (256x1152x4608) blocking sweep, MKL={gf2 / (mkl2 / 1000):F0}GF ---");
            foreach (int mcv in new[] { 32, 48, 64, 128 })
                foreach (int ncv in new[] { 16, 32, 64, 128 })
                {
                    int tiles = ((m2 + mcv - 1) / mcv) * ((n2 + ncv - 1) / ncv);
                    double t; fixed (float* pa = A2, pb = B2, pc = C2) { var paL = (nint)pa; var pbL = (nint)pb; var pcL = (nint)pc; t = Min(() => AiDotNet.Tensors.Engines.BlasManaged.GotoGemmFp32.RunParallel((float*)paL, k2, (float*)pbL, n2, (float*)pcL, n2, m2, n2, k2, mcv, ncv, 256), 5); }
                    Console.WriteLine($"   mc={mcv,3} nc={ncv,3}: {gf2 / (t / 1000),6:F0}GF  tiles={tiles,4}  ({gf2 / (t / 1000) / (gf2 / (mkl2 / 1000)) * 100:F0}% MKL)");
                }
        }
        // ROUTING A/B (same run): does improved GotoGemm now BEAT PackBoth at the DiT shapes? The
        // BeatsPackBoth gate routed small-M-wide-N to PackBoth BEFORE the short-M blocking fix; re-decide.
        Console.WriteLine("--- ROUTING: GotoGemm (RunParallel) vs PackBoth (BlasManaged, goto disabled) vs MKL ---");
        var ditShapes = new (int m, int n, int k, string tag)[] { (256, 3456, 1152, "qkv"), (256, 1152, 1152, "attnout"), (256, 4608, 1152, "mlp1"), (256, 1152, 4608, "mlp2") };
        foreach (var (dm, dn, dk, dtag) in ditShapes)
        {
            float[] A3 = R((long)dm * dk, rng), B3 = R((long)dk * dn, rng), C3 = new float[(long)dm * dn];
            using var t3a = torch.tensor(A3).reshape(dm, dk); using var t3b = torch.tensor(B3).reshape(dk, dn);
            var (mc3, nc3, kc3) = AiDotNet.Tensors.Engines.BlasManaged.GotoGemmFp32.ChooseParallelBlocks(dm, dn);
            double gf3 = 2.0 * dm * dn * dk / 1e9;
            double goto3; fixed (float* pa = A3, pb = B3, pc = C3) { var paL = (nint)pa; var pbL = (nint)pb; var pcL = (nint)pc; goto3 = Min(() => AiDotNet.Tensors.Engines.BlasManaged.GotoGemmFp32.RunParallel((float*)paL, dk, (float*)pbL, dn, (float*)pcL, dn, dm, dn, dk, mc3, nc3, kc3), 6); }
            bool savedDis = AiDotNet.Tensors.Engines.BlasManaged.BlasManaged.s_disableGotoGemm;
            AiDotNet.Tensors.Engines.BlasManaged.BlasManaged.s_disableGotoGemm = true;
            double pb3 = Min(() => AiDotNet.Tensors.Engines.BlasManaged.BlasManaged.Gemm<float>(A3.AsSpan(0, dm * dk), dk, false, B3.AsSpan(0, dk * dn), dn, false, C3.AsSpan(0, dm * dn), dn, dm, dn, dk, new AiDotNet.Tensors.Engines.BlasManaged.BlasOptions<float> { PackingMode = AiDotNet.Tensors.Engines.BlasManaged.PackingMode.DisableAutotune }), 6);
            AiDotNet.Tensors.Engines.BlasManaged.BlasManaged.s_disableGotoGemm = savedDis;
            double mkl3 = Min(() => { using var _ = torch.matmul(t3a, t3b); }, 6);
            double g = gf3 / (goto3 / 1000), p = gf3 / (pb3 / 1000), mk = gf3 / (mkl3 / 1000);
            Console.WriteLine($"   {dtag,-8} GotoGemm={g,6:F0}GF  PackBoth={p,6:F0}GF  MKL={mk,6:F0}GF  goto/pb={g / p,4:F2}x  → route to {(g > p * 1.05 ? "GotoGemm" : "PackBoth")}");
        }
        // SPLIT-K A/B (same run): deep-K short-M is under-parallelized; does split-K (shape-based G) beat
        // per-tile RunParallel + stay correct? Shapes: mlp2 + a deeper synthetic.
        Console.WriteLine("--- SPLIT-K: RunParallel (per-tile) vs RunParallelSplitK vs MKL (deep-K short-M) ---");
        var skShapes = new (int m, int n, int k, string tag)[] { (256, 1152, 4608, "mlp2 k=4608"), (256, 1152, 9216, "deep k=9216"), (512, 1152, 8192, "m512 k=8192") };
        foreach (var (dm, dn, dk, dtag) in skShapes)
        {
            if (!AiDotNet.Tensors.Engines.BlasManaged.GotoGemmFp32.ShouldSplitK(dm, dn, dk)) { Console.WriteLine($"   {dtag,-12} (ShouldSplitK=false, skipped)"); continue; }
            float[] A4 = R((long)dm * dk, rng), B4 = R((long)dk * dn, rng), Cp = new float[(long)dm * dn], Cs = new float[(long)dm * dn];
            using var t4a = torch.tensor(A4).reshape(dm, dk); using var t4b = torch.tensor(B4).reshape(dk, dn);
            var (mc4, nc4, kc4) = AiDotNet.Tensors.Engines.BlasManaged.GotoGemmFp32.ChooseParallelBlocks(dm, dn);
            int G = AiDotNet.Tensors.Engines.BlasManaged.GotoGemmFp32.SplitKGroups(dm, dn, dk);
            double gf4 = 2.0 * dm * dn * dk / 1e9, relErr;
            double pt, sk; fixed (float* pa = A4, pb = B4, pcp = Cp, pcs = Cs)
            {
                var paL = (nint)pa; var pbL = (nint)pb; var pcpL = (nint)pcp; var pcsL = (nint)pcs;
                AiDotNet.Tensors.Engines.BlasManaged.GotoGemmFp32.RunParallel((float*)paL, dk, (float*)pbL, dn, (float*)pcpL, dn, dm, dn, dk, mc4, nc4, kc4);
                AiDotNet.Tensors.Engines.BlasManaged.GotoGemmFp32.RunParallelSplitK((float*)paL, dk, (float*)pbL, dn, (float*)pcsL, dn, dm, dn, dk, mc4, nc4, kc4);
                double se = 0, sr = 0; for (long i = 0; i < (long)dm * dn; i += 7) { double e = (double)Cs[i] - Cp[i]; se += e * e; sr += (double)Cp[i] * Cp[i]; }
                relErr = sr > 0 ? Math.Sqrt(se / sr) : 0;
                pt = Min(() => AiDotNet.Tensors.Engines.BlasManaged.GotoGemmFp32.RunParallel((float*)paL, dk, (float*)pbL, dn, (float*)pcpL, dn, dm, dn, dk, mc4, nc4, kc4), 6);
                sk = Min(() => AiDotNet.Tensors.Engines.BlasManaged.GotoGemmFp32.RunParallelSplitK((float*)paL, dk, (float*)pbL, dn, (float*)pcsL, dn, dm, dn, dk, mc4, nc4, kc4), 6);
            }
            double mkl4 = Min(() => { using var _ = torch.matmul(t4a, t4b); }, 6);
            double ptG = gf4 / (pt / 1000), skG = gf4 / (sk / 1000), mkG = gf4 / (mkl4 / 1000);
            Console.WriteLine($"   {dtag,-12} G={G} per-tile={ptG,6:F0}GF  split-K={skG,6:F0}GF  MKL={mkG,6:F0}GF  sk/pt={skG / ptG,4:F2}x  sk/MKL={skG / mkG * 100,3:F0}%  relErr={relErr:E1}");
        }
    }

    private static (double median, double p95) TimeAi(Func<Tensor<float>> forward)
        => MeasureRobust(() => forward());

    private static (double median, double p95) TimeTorch(Func<torch.Tensor> forward)
        => MeasureRobust(() => forward().Dispose());

    /// <summary>
    /// Runs <see cref="Rounds"/> independent measurement windows of
    /// <see cref="Iters"/> iterations each and returns the (median, p95) of the
    /// window with the LOWEST median — the least load-perturbed sample. This is
    /// the min-of-N noise-robustness contract used for both sides.
    /// </summary>
    private static (double median, double p95) MeasureRobust(Action run)
    {
        for (int i = 0; i < Warmup; i++) run();
        var sw = new Stopwatch();
        double bestMedian = double.MaxValue, bestP95 = double.MaxValue;
        for (int r = 0; r < Rounds; r++)
        {
            SettleGc();
            var times = new double[Iters];
            for (int i = 0; i < Iters; i++)
            {
                sw.Restart();
                run();
                sw.Stop();
                times[i] = sw.Elapsed.TotalMilliseconds;
            }
            var (m, p) = Percentiles(times);
            if (m < bestMedian) { bestMedian = m; bestP95 = p; }
        }
        return (bestMedian, bestP95);
    }

    private static (double median, double p95) Percentiles(double[] times)
    {
        Array.Sort(times);
        int n = times.Length;
        return (times[n / 2], times[Math.Min(n - 1, (int)(n * 0.95))]);
    }

    private static void SettleGc()
    {
        GC.Collect(GC.MaxGeneration, GCCollectionMode.Forced, blocking: true);
        GC.WaitForPendingFinalizers();
        GC.Collect(GC.MaxGeneration, GCCollectionMode.Forced, blocking: true);
    }

    private static Row Print(string name, double aiMed, double aiP95, double tMed, double tP95)
    {
        bool win = aiP95 < tMed;
        Console.WriteLine($"  {name,-12} AiDN med {aiMed,7:F3} p95 {aiP95,7:F3}  |  torch med {tMed,7:F3} p95 {tP95,7:F3}  → {(win ? "WIN" : "LOSS")}");
        return new Row { Name = name, AiMedian = aiMed, AiP95 = aiP95, TorchMedian = tMed, TorchP95 = tP95 };
    }

    private sealed class Row
    {
        public string Name = "";
        public double AiMedian, AiP95, TorchMedian, TorchP95;
    }
}
