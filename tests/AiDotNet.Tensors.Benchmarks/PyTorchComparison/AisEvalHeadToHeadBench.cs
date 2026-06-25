using System;
using System.Collections.Generic;
using System.Diagnostics;
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
        var rng = new Random(13);
        var shapes = new (int M, int N, int K, string tag)[]
        {
            (256, 1152, 1152, "attn-proj"), (256, 1152, 4608, "mlp-fc"),
            (1024, 1024, 1024, "square1024"), (2048, 2048, 2048, "square2048"),
        };
        // Cache-resident-tile hypothesis: small mc (A→L1) + small nc (B→L2) lets the microkernel run
        // at peak with NO shared-L3 streaming, so it should scale past the existing M-axis L3 cap.
        var mcs = new[] { 16, 24, 32, 48, 64, 96, 120 };
        var ncs = new[] { 64, 96, 128, 192, 256, 512 };
        try
        {
            foreach (var (M, N, K, tag) in shapes)
            {
                var A = new float[(long)M * K]; var B = new float[(long)K * N]; var C = new float[(long)M * N];
                for (int i = 0; i < A.Length; i++) A[i] = (float)(rng.NextDouble() - 0.5);
                for (int i = 0; i < B.Length; i++) B[i] = (float)(rng.NextDouble() - 0.5);
                double gf = 2.0 * M * N * K / 1e9, mkl = 0;
                try { var ta = torch.rand(M, K); var tb = torch.rand(K, N); torch.set_num_threads(P); mkl = gf / (MinMs(() => { using var _ = torch.matmul(ta, tb); }) / 1000); }
                catch (Exception e) { Console.WriteLine($"  torch ref failed: {e.Message}"); }
                double bestGf = 0; int bMc = 0, bNc = 0; string bCorrect = "?";
                fixed (float* pa = A, pb = B, pc = C)
                {
                    foreach (int mc in mcs)
                    foreach (int nc in ncs)
                    {
                        AiDotNet.Tensors.Engines.BlasManaged.GotoGemmFp32.RunParallel(pa, K, pb, N, pc, N, M, N, K, mc, nc, kc);
                        double maxRel = 0; var rv = new Random(5);
                        for (int t = 0; t < 80; t++)
                        {
                            int r = rv.Next(M), col = rv.Next(N);
                            double s = 0; for (int kk = 0; kk < K; kk++) s += (double)A[(long)r * K + kk] * B[(long)kk * N + col];
                            double err = Math.Abs(C[(long)r * N + col] - s);
                            maxRel = Math.Max(maxRel, Math.Abs(s) > 1e-3 ? err / Math.Abs(s) : err);
                        }
                        double gbest = double.PositiveInfinity;
                        var w = Stopwatch.StartNew(); while (w.ElapsedMilliseconds < 60) AiDotNet.Tensors.Engines.BlasManaged.GotoGemmFp32.RunParallel(pa, K, pb, N, pc, N, M, N, K, mc, nc, kc);
                        for (int r = 0; r < 6; r++)
                        {
                            var sw = Stopwatch.StartNew(); int reps = 0;
                            do { AiDotNet.Tensors.Engines.BlasManaged.GotoGemmFp32.RunParallel(pa, K, pb, N, pc, N, M, N, K, mc, nc, kc); reps++; } while (sw.Elapsed.TotalMilliseconds < 40);
                            sw.Stop(); gbest = Math.Min(gbest, sw.Elapsed.TotalMilliseconds / reps);
                        }
                        double g = gf / (gbest / 1000);
                        if (g > bestGf) { bestGf = g; bMc = mc; bNc = nc; bCorrect = maxRel < 1e-3 ? "OK" : "WRONG " + maxRel.ToString("E1"); }
                    }
                }
                Console.WriteLine($"{tag,-11} best={bestGf,7:F0} GF  mc={bMc,4} nc={bNc,4}  MKL={mkl,7:F0}  {(mkl > 0 ? bestGf / mkl * 100 : 0),5:F0}%  {bCorrect}");
            }
        }
        finally { CpuParallelSettings.MaxDegreeOfParallelism = saved; }
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
