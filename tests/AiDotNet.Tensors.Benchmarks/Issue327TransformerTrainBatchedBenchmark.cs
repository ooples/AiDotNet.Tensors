#if NET8_0_OR_GREATER
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Runtime.Intrinsics.X86;
using System.Threading;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Benchmarks;

/// <summary>
/// Issue #327 comprehensive baseline harness — captures every measurement
/// needed to diagnose why TrainBatched at d=128/L=4/B=32 runs 6-10× slower
/// than PyTorch CPU and uses only 4 of 32 cores.
///
/// <para><b>Phases</b>:</para>
/// <list type="bullet">
///   <item><b>Phase 0 — Environment</b>: BLAS backend, AVX2/AVX-512 status,
///   serial-grain threshold, MaxParallelism, processor count.</item>
///   <item><b>Phase A — Per-shape matmul</b>: 5 dominant shapes from the
///   issue, with wall ms, active cores, GFLOPS achieved, and PyTorch ref gap.</item>
///   <item><b>Phase B — Forward scaling</b>: single-layer + L=4 forward only,
///   no autodiff tape, to isolate kernel cost from autodiff overhead.</item>
///   <item><b>Phase C — Train step (L=4 full)</b>: forward + backward via
///   GradientTape.ComputeGradients + Adam-style optimizer in place. Matches
///   the issue's exact consumer config. Reports per-phase split: forward /
///   backward / optimizer.</item>
///   <item><b>Phase D — Sustained CPU probe</b>: 100ms sampling of CPU/wall
///   ratio during a 5-second train window. Reports min/median/max active
///   cores + Gen0/1/2 GC counts.</item>
/// </list>
///
/// <para><b>Invocation</b>:
/// <c>dotnet run -c Release --project tests/AiDotNet.Tensors.Benchmarks -- --327-transformer</c></para>
/// </summary>
public static class Issue327TransformerTrainBatchedBenchmark
{
    // Consumer-reported config from issue #327
    private const int B = 32;
    private const int Ctx = 64;
    private const int D = 128;
    private const int Heads = 4;
    private const int FfDim = 512;
    private const int Vocab = 8192;
    private const int Layers = 4;
    private const int HeadDim = D / Heads;

    private static volatile float _sink;

    public static void Run()
    {
        var engine = new CpuEngine();
        AiDotNetEngine.Current = engine;

        Console.WriteLine();
        Console.WriteLine("============================================================");
        Console.WriteLine(" Issue #327 — Transformer TrainBatched comprehensive baseline");
        Console.WriteLine("============================================================");

        PrintEnvironment();

        Console.WriteLine();
        Console.WriteLine("─── Phase A — Per-shape matmul wall ────────────────────────");
        RunPerShapeMatmuls(engine);

        Console.WriteLine();
        Console.WriteLine("─── Phase B — Forward scaling (L=1 vs L=4, no tape) ────────");
        RunForwardScaling(engine);

        Console.WriteLine();
        Console.WriteLine("─── Phase C — Full Train step (L=4, fwd+bwd+opt) ───────────");
        RunFullTrainStep(engine, persistent: false);

        Console.WriteLine();
        Console.WriteLine("─── Phase C2 — Same step, Persistent tape (compiled chain) ─");
        RunFullTrainStep(engine, persistent: true);

        Console.WriteLine();
        Console.WriteLine("─── Phase D — Sustained CPU utilization probe (5s window) ──");
        RunSustainedCpuProbe(engine);

        Console.WriteLine();
        Console.WriteLine($"Sink: {_sink:F6}");
    }

    private static void PrintEnvironment()
    {
        Console.WriteLine();
        Console.WriteLine("Environment:");
        Console.WriteLine($"  CPU                    : {Environment.ProcessorCount} logical cores");
        Console.WriteLine($"  AVX2                   : {Avx2.IsSupported}");
        Console.WriteLine($"  FMA                    : {Fma.IsSupported}");
        Console.WriteLine($"  AVX-512F               : {Avx512F.IsSupported}");
        Console.WriteLine($"  BLAS backend           : {BlasProvider.BackendName}");
        Console.WriteLine($"  BLAS available         : {BlasProvider.IsAvailable}");
        Console.WriteLine($"  Deterministic mode     : {BlasProvider.IsDeterministicMode}");
        Console.WriteLine($"  SerialGrainSize        : {PersistentParallelExecutor.DefaultSerialGrainSize:N0}");
        Console.WriteLine($"  MaxDegreeOfParallelism : {CpuParallelSettings.MaxDegreeOfParallelism}");
        Console.WriteLine($"  Config                 : d={D}, L={Layers}, heads={Heads}, ffn={FfDim},");
        Console.WriteLine($"                           V={Vocab}, B={B}, ctx={Ctx}");
    }

    // ────────────────────────────────────────────────────────────────────
    // Phase A — per-shape matmul wall + active cores + GFLOPS
    // ────────────────────────────────────────────────────────────────────

    private static void RunPerShapeMatmuls(CpuEngine engine)
    {
        // The 5 dominant shapes from the issue's "Per-shape PyTorch reference" table.
        // M·K·N gives the FMA count; cycles assume 16-core Zen 2 @ ~3.5 GHz × 16 cores
        // × 8 SP FMA lanes × 2 ops/FMA = ~896 GFLOPS theoretical peak.
        // We report GFLOPS achieved = 2·M·K·N / wallSec / 1e9.
        var shapes = new (string Label, int[] AShape, int[] BShape, double PyTorchMs)[]
        {
            ("Encoder attn QKV proj    ", new[] { B, Ctx, D },              new[] { D, 3 * D },          0.3),
            ("Encoder attn scores      ", new[] { B, Heads, Ctx, HeadDim }, new[] { B, Heads, HeadDim, Ctx }, 0.1),
            ("Encoder FFN up           ", new[] { B, Ctx, D },              new[] { D, FfDim },          0.4),
            ("Encoder FFN down         ", new[] { B, Ctx, FfDim },          new[] { FfDim, D },          0.4),
            ("Output proj (V=8192)     ", new[] { B, Ctx, D },              new[] { D, Vocab },          3.0),
        };

        Console.WriteLine();
        Console.WriteLine($"  {"Op",-28}{"A.shape",-22}{"B.shape",-22}{"ms",10}{"PT ms",10}{"Gap",8}{"Cores",8}{"GFLOPS",10}");

        var rng = new Random(42);
        foreach (var (label, aShape, bShape, pyMs) in shapes)
        {
            var a = MakeFloatTensor(aShape, rng);
            var b = MakeFloatTensor(bShape, rng);
            long flops = ComputeFlops(aShape, bShape);
            var (ms, activeCores) = TimeMatmul(engine, a, b, warmup: 5, iters: 100);
            string gap = pyMs > 0 ? $"{ms / pyMs:F2}x" : "—";
            double gflops = flops / 1e9 / (ms / 1000.0);
            Console.WriteLine(
                $"  {label,-28}{ShapeStr(aShape),-22}{ShapeStr(bShape),-22}{ms,10:F3}{pyMs,10:F3}{gap,8}{activeCores,8:F1}{gflops,10:F1}");
        }
    }

    private static long ComputeFlops(int[] aShape, int[] bShape)
    {
        // 2·M·K·N FMA pairs (mul+add counted as 2 flops).
        // For ND × 2D: batch is the leading product of aShape, M·N from the last 2 dims.
        // For ND × ND: per-batch M·K·N × batch.
        long m, k, n;
        if (bShape.Length == 2)
        {
            int batch = 1;
            for (int i = 0; i < aShape.Length - 2; i++) batch *= aShape[i];
            m = (long)batch * aShape[aShape.Length - 2];
            k = aShape[aShape.Length - 1];
            n = bShape[1];
            return 2L * m * k * n;
        }
        // ND × ND
        int batch2 = 1;
        for (int i = 0; i < aShape.Length - 2; i++) batch2 *= aShape[i];
        m = aShape[aShape.Length - 2];
        k = aShape[aShape.Length - 1];
        n = bShape[bShape.Length - 1];
        return 2L * batch2 * m * k * n;
    }

    private static (double msPerIter, double activeCores) TimeMatmul(
        CpuEngine engine, Tensor<float> a, Tensor<float> b, int warmup, int iters)
    {
        for (int i = 0; i < warmup; i++)
        {
            var r = engine.TensorMatMul(a, b);
            _sink += r.GetFlatIndexValue(0);
        }

        var proc = Process.GetCurrentProcess();
        proc.Refresh();
        TimeSpan cpuBefore = proc.TotalProcessorTime;
        var sw = Stopwatch.StartNew();
        for (int i = 0; i < iters; i++)
        {
            var r = engine.TensorMatMul(a, b);
            _sink += r.GetFlatIndexValue(0);
        }
        sw.Stop();
        proc.Refresh();
        TimeSpan cpuAfter = proc.TotalProcessorTime;
        double wallSec = sw.Elapsed.TotalSeconds;
        double cpuSec = (cpuAfter - cpuBefore).TotalSeconds;
        return (sw.Elapsed.TotalMilliseconds / iters, cpuSec / wallSec);
    }

    // ────────────────────────────────────────────────────────────────────
    // Phase B — forward scaling: L=1 vs L=4 to isolate per-layer cost
    // ────────────────────────────────────────────────────────────────────

    private static void RunForwardScaling(CpuEngine engine)
    {
        Console.WriteLine();
        for (int layers = 1; layers <= Layers; layers *= 2)
        {
            if (layers > Layers) break;
            var weights = MakeWeights(layers);
            var input = MakeFloatTensor(new[] { B, Ctx, D }, new Random(42));

            const int warmup = 10;
            const int iters = 30;
            for (int i = 0; i < warmup; i++)
            {
                var y = ForwardL(engine, input, weights, layers);
                _sink += y.GetFlatIndexValue(0);
            }

            var proc = Process.GetCurrentProcess();
            proc.Refresh();
            TimeSpan cpuBefore = proc.TotalProcessorTime;
            var sw = Stopwatch.StartNew();
            for (int i = 0; i < iters; i++)
            {
                var y = ForwardL(engine, input, weights, layers);
                _sink += y.GetFlatIndexValue(0);
            }
            sw.Stop();
            proc.Refresh();
            TimeSpan cpuAfter = proc.TotalProcessorTime;

            double ms = sw.Elapsed.TotalMilliseconds / iters;
            double cores = (cpuAfter - cpuBefore).TotalSeconds / sw.Elapsed.TotalSeconds;
            Console.WriteLine($"  L={layers,-3} forward only: {ms,8:F3} ms/step  active cores: {cores,5:F1}");
        }
        var allFour = MakeWeights(Layers);
        Console.WriteLine($"  (L=4 expected: ~4× L=1; if not, dispatch overhead is per-call, not per-layer)");
    }

    // ────────────────────────────────────────────────────────────────────
    // Phase C — full L=4 Train step with autograd backward + Adam-ish opt
    // ────────────────────────────────────────────────────────────────────

    private static void RunFullTrainStep(CpuEngine engine, bool persistent)
    {
        var weights = MakeWeights(Layers);
        var input = MakeFloatTensor(new[] { B, Ctx, D }, new Random(42));

        // Optimizer state: m, v moments for Adam (same shape as each weight)
        var optM = weights.Select(w => MakeFloatTensor(w._shape, new Random(0))).ToArray();
        var optV = weights.Select(w => MakeFloatTensor(w._shape, new Random(0))).ToArray();

        const int warmup = 3;
        const int iters = 20;

        // For persistent mode, hold one tape across all iters so the
        // compiled-chain cache engages on iter 2+. For non-persistent,
        // each call creates a fresh tape inside DoTrainStep.
        GradientTape<float>? sharedTape = persistent
            ? new GradientTape<float>(new GradientTapeOptions { Persistent = true })
            : null;

        // Warmup
        for (int i = 0; i < warmup; i++)
        {
            DoTrainStep(engine, input, weights, optM, optV, sharedTape, out _, out _, out _, out _, out _, out _, out _);
        }

        // Per-phase alloc tracking: measure forward / backward / opt separately
        long fwdAllocTotal = 0, bwdAllocTotal = 0, optAllocTotal = 0;

        long gen0Before = GC.CollectionCount(0);
        long gen1Before = GC.CollectionCount(1);
        long gen2Before = GC.CollectionCount(2);
        long allocBefore = GC.GetTotalAllocatedBytes(precise: true);

        var proc = Process.GetCurrentProcess();
        proc.Refresh();
        TimeSpan cpuBefore = proc.TotalProcessorTime;

        var phaseTotals = new (double fwdMs, double bwdMs, double optMs)[iters];
        var sw = Stopwatch.StartNew();
        // Sanity: verify gradients are actually computed (non-empty for sources).
        // Persistent-tape pattern can silently break across iters when the cached
        // chain references stale entries from iter 1, so we capture grad-count on
        // every iter and assert at the end.
        int sourcesPresentLastIter = 0;
        for (int i = 0; i < iters; i++)
        {
            DoTrainStep(engine, input, weights, optM, optV, sharedTape,
                out var fwdMs, out var bwdMs, out var optMs,
                out var fwdAlloc, out var bwdAlloc, out var optAlloc,
                out var sourcesPresent);
            phaseTotals[i] = (fwdMs, bwdMs, optMs);
            fwdAllocTotal += fwdAlloc;
            bwdAllocTotal += bwdAlloc;
            optAllocTotal += optAlloc;
            sourcesPresentLastIter = sourcesPresent;
        }
        sw.Stop();

        sharedTape?.Dispose();

        proc.Refresh();
        TimeSpan cpuAfter = proc.TotalProcessorTime;
        long allocAfter = GC.GetTotalAllocatedBytes(precise: true);
        long gen0After = GC.CollectionCount(0);
        long gen1After = GC.CollectionCount(1);
        long gen2After = GC.CollectionCount(2);

        double msPer = sw.Elapsed.TotalMilliseconds / iters;
        double medianFwd = Median(phaseTotals.Select(p => p.fwdMs).ToArray());
        double medianBwd = Median(phaseTotals.Select(p => p.bwdMs).ToArray());
        double medianOpt = Median(phaseTotals.Select(p => p.optMs).ToArray());
        long allocPer = (allocAfter - allocBefore) / iters;
        double cpuSec = (cpuAfter - cpuBefore).TotalSeconds;
        double wallSec = sw.Elapsed.TotalSeconds;
        double activeCores = cpuSec / wallSec;

        Console.WriteLine();
        Console.WriteLine($"  Train step (L={Layers}, persistent={persistent}):");
        Console.WriteLine($"    Per-iter wall    : {msPer,8:F3} ms");
        Console.WriteLine($"    Per-phase median : fwd {medianFwd:F3} ms / bwd {medianBwd:F3} ms / opt {medianOpt:F3} ms");
        Console.WriteLine($"    CPU/wall         : {cpuSec:F2}s / {wallSec:F2}s = {activeCores:F2} active cores ({100 * activeCores / Environment.ProcessorCount:F0}%)");
        Console.WriteLine($"    Allocation       : {allocPer / 1024.0:F1} KB/iter (= {allocPer / 1024.0 / 1024.0:F1} MB/iter)");
        Console.WriteLine($"      fwd alloc     : {(fwdAllocTotal / iters) / 1024.0 / 1024.0,7:F1} MB/iter");
        Console.WriteLine($"      bwd alloc     : {(bwdAllocTotal / iters) / 1024.0 / 1024.0,7:F1} MB/iter");
        Console.WriteLine($"      opt alloc     : {(optAllocTotal / iters) / 1024.0 / 1024.0,7:F1} MB/iter");
        Console.WriteLine($"    Sources w/ grad   : {sourcesPresentLastIter}/{weights.Length} on last iter " +
                          $"{(sourcesPresentLastIter == weights.Length ? "(all weights got gradients — correct)" : "(MISSING GRADIENTS — measurement invalid)")}");
        Console.WriteLine($"    GC during run    : Gen0 {gen0After - gen0Before,3}  Gen1 {gen1After - gen1Before,3}  Gen2 {gen2After - gen2Before,3} (over {iters} iters)");
        Console.WriteLine();
        Console.WriteLine($"  Issue close target : ≤ 100.000 ms/step, ≥ 20 active cores");
        Console.WriteLine($"  Stretch (parity)   : ≤  50.000 ms/step, ≥ 28 active cores");
        Console.WriteLine($"  Status             : ms-step  : {(msPer <= 50.0 ? "PASS stretch" : msPer <= 100.0 ? "PASS issue-close" : "FAIL")}");
        Console.WriteLine($"                       cores    : {(activeCores >= 28 ? "PASS stretch" : activeCores >= 20 ? "PASS issue-close" : "FAIL")}");
    }

    private static void DoTrainStep(
        CpuEngine engine,
        Tensor<float> input,
        Tensor<float>[] weights,
        Tensor<float>[] optM,
        Tensor<float>[] optV,
        GradientTape<float>? sharedTape,
        out double fwdMs,
        out double bwdMs,
        out double optMs,
        out long fwdAlloc,
        out long bwdAlloc,
        out long optAlloc,
        out int sourcesPresent)
    {
        sourcesPresent = 0;
        var swFwd = Stopwatch.StartNew();
        Dictionary<Tensor<float>, Tensor<float>> grads;
        Tensor<float> loss;
        GradientTape<float>? ownTape = sharedTape is null ? new GradientTape<float>() : null;
        var tape = sharedTape ?? ownTape!;
        GradientsScope<float>? gradScope = null;
        long allocFwdStart = GC.GetTotalAllocatedBytes(precise: true);
        try
        {
            var y = ForwardL(engine, input, weights, Layers);
            // ReduceSum returns a tape-connected Tensor<float> (scalar
            // shape) — TensorSum returns a raw T which is OFF the graph
            // and breaks ComputeGradients. Issue #327 baseline needs
            // a real backward walk to measure backward cost.
            loss = engine.ReduceSum(y, axes: null, keepDims: false);

            swFwd.Stop();
            fwdMs = swFwd.Elapsed.TotalMilliseconds;
            long allocFwdEnd = GC.GetTotalAllocatedBytes(precise: true);
            fwdAlloc = allocFwdEnd - allocFwdStart;

            var swBwd = Stopwatch.StartNew();
            gradScope = tape.ComputeGradientsScope(loss, weights);
            grads = gradScope.Grads;
            swBwd.Stop();
            bwdMs = swBwd.Elapsed.TotalMilliseconds;
            bwdAlloc = GC.GetTotalAllocatedBytes(precise: true) - allocFwdEnd;

            foreach (var w in weights) if (grads.ContainsKey(w)) sourcesPresent++;
        }
        finally
        {
            ownTape?.Dispose();
        }
        long allocOptStart = GC.GetTotalAllocatedBytes(precise: true);

        // In-place Adam-style optimizer update.
        var swOpt = Stopwatch.StartNew();
        const float lr = 1e-3f;
        const float beta1 = 0.9f, beta2 = 0.999f, eps = 1e-8f;
        for (int wi = 0; wi < weights.Length; wi++)
        {
            if (!grads.TryGetValue(weights[wi], out var g)) continue;
            var w = weights[wi];
            var m = optM[wi];
            var v = optV[wi];
            // Element-wise: m = β1·m + (1-β1)·g; v = β2·v + (1-β2)·g²; w -= lr · m / (sqrt(v) + ε)
            var wData = w.Data.Span;
            var gData = g.Data.Span;
            var mData = m.Data.Span;
            var vData = v.Data.Span;
            int len = w.Length;
            for (int idx = 0; idx < len; idx++)
            {
                float gi = gData[idx];
                mData[idx] = beta1 * mData[idx] + (1 - beta1) * gi;
                vData[idx] = beta2 * vData[idx] + (1 - beta2) * gi * gi;
                wData[idx] -= lr * mData[idx] / (MathF.Sqrt(vData[idx]) + eps);
            }
        }
        swOpt.Stop();
        optMs = swOpt.Elapsed.TotalMilliseconds;
        optAlloc = GC.GetTotalAllocatedBytes(precise: true) - allocOptStart;

        // Optimizer has consumed gradients; dispose the scope so the
        // per-iter gradient tensors get pooled back to AutoTensorCache
        // for the next iter's backward to reuse.
        gradScope?.Dispose();

        _sink += loss.GetFlatIndexValue(0);
    }

    // ────────────────────────────────────────────────────────────────────
    // Phase D — sustained CPU sampling during a 5-second train window
    // ────────────────────────────────────────────────────────────────────

    private static void RunSustainedCpuProbe(CpuEngine engine)
    {
        var weights = MakeWeights(Layers);
        var input = MakeFloatTensor(new[] { B, Ctx, D }, new Random(42));
        var optM = weights.Select(w => MakeFloatTensor(w._shape, new Random(0))).ToArray();
        var optV = weights.Select(w => MakeFloatTensor(w._shape, new Random(0))).ToArray();

        // 5-second sustained train run with 100ms CPU sampling.
        const double targetSec = 5.0;
        const int sampleIntervalMs = 100;
        var samples = new List<double>();

        // Background sampler thread: every 100ms, reads CPU time delta & wall delta.
        var stopSampler = new ManualResetEventSlim(false);
        var sampleProc = Process.GetCurrentProcess();
        sampleProc.Refresh();
        TimeSpan lastCpu = sampleProc.TotalProcessorTime;
        var lastWall = Stopwatch.GetTimestamp();
        var sampler = new Thread(() =>
        {
            while (!stopSampler.Wait(sampleIntervalMs))
            {
                sampleProc.Refresh();
                TimeSpan nowCpu = sampleProc.TotalProcessorTime;
                long nowWall = Stopwatch.GetTimestamp();
                double cpuDelta = (nowCpu - lastCpu).TotalSeconds;
                double wallDelta = (nowWall - lastWall) / (double)Stopwatch.Frequency;
                if (wallDelta > 0)
                {
                    lock (samples) samples.Add(cpuDelta / wallDelta);
                }
                lastCpu = nowCpu;
                lastWall = nowWall;
            }
        }) { IsBackground = true };

        // Warmup so JIT is stable
        DoTrainStep(engine, input, weights, optM, optV, sharedTape: null, out _, out _, out _, out _, out _, out _, out _);

        sampler.Start();
        var sw = Stopwatch.StartNew();
        int steps = 0;
        while (sw.Elapsed.TotalSeconds < targetSec)
        {
            DoTrainStep(engine, input, weights, optM, optV, sharedTape: null, out _, out _, out _, out _, out _, out _, out _);
            steps++;
        }
        sw.Stop();
        stopSampler.Set();
        sampler.Join(TimeSpan.FromSeconds(1));

        double[] sortedSamples;
        lock (samples) sortedSamples = samples.OrderBy(s => s).ToArray();

        double minCores = sortedSamples.Length > 0 ? sortedSamples[0] : 0;
        double maxCores = sortedSamples.Length > 0 ? sortedSamples[^1] : 0;
        double medianCores = sortedSamples.Length > 0 ? sortedSamples[sortedSamples.Length / 2] : 0;
        double avgCores = sortedSamples.Length > 0 ? sortedSamples.Average() : 0;
        double p10 = sortedSamples.Length > 0 ? sortedSamples[Math.Max(0, sortedSamples.Length / 10)] : 0;
        double p90 = sortedSamples.Length > 0 ? sortedSamples[Math.Min(sortedSamples.Length - 1, 9 * sortedSamples.Length / 10)] : 0;

        Console.WriteLine();
        Console.WriteLine($"  Sustained probe over {sw.Elapsed.TotalSeconds:F2}s ({steps} steps, {sortedSamples.Length} samples @ {sampleIntervalMs}ms):");
        Console.WriteLine($"    Min active cores       : {minCores,6:F2}");
        Console.WriteLine($"    p10 active cores       : {p10,6:F2}");
        Console.WriteLine($"    Median active cores    : {medianCores,6:F2}");
        Console.WriteLine($"    Average active cores   : {avgCores,6:F2}");
        Console.WriteLine($"    p90 active cores       : {p90,6:F2}");
        Console.WriteLine($"    Max active cores       : {maxCores,6:F2}");
        Console.WriteLine($"    HW logical cores       : {Environment.ProcessorCount}");
        Console.WriteLine($"    Median utilization     : {100 * medianCores / Environment.ProcessorCount,5:F1}%");
        Console.WriteLine($"    Steady-state ms/step   : {sw.Elapsed.TotalMilliseconds / steps,8:F3} ms");
    }

    // ────────────────────────────────────────────────────────────────────
    // Helpers
    // ────────────────────────────────────────────────────────────────────

    /// <summary>
    /// Forward pass over L stacked transformer-encoder-style layers. Each
    /// layer runs: QKV proj → Wo proj → FFN-up → GELU → FFN-down. The
    /// output projection to vocab runs once at the end. Faithfully hits
    /// every matmul shape from the issue.
    /// </summary>
    private static Tensor<float> ForwardL(
        CpuEngine engine,
        Tensor<float> input,
        Tensor<float>[] weights,
        int layers)
    {
        // weights layout per layer: [wQkv, wO, wF1, wF2], with wOut at the end
        var x = input;
        for (int l = 0; l < layers; l++)
        {
            int offset = l * 4;
            var wQkv = weights[offset + 0];
            var wO   = weights[offset + 1];
            var wF1  = weights[offset + 2];
            var wF2  = weights[offset + 3];

            // QKV projection: produces [B,Ctx,3D]. We collapse 3D back to D
            // by slicing the first D channels (proxy for taking the Q
            // portion). This keeps wQkv on the autodiff graph so its
            // gradient flows correctly.
            var qkv = engine.TensorMatMul(x, wQkv);
            var qProxy = engine.TensorSlice(qkv, new[] { 0, 0, 0 }, new[] { B, Ctx, D });

            // Wo applied to qProxy so Wo also remains live on the graph.
            var attnOut = engine.TensorMatMul(qProxy, wO);

            // FFN
            var f1 = engine.TensorMatMul(attnOut, wF1);
            var f1g = engine.GELU(f1);
            var f2 = engine.TensorMatMul(f1g, wF2);
            x = f2;
        }
        // Final output projection
        var wOut = weights[layers * 4];
        return engine.TensorMatMul(x, wOut);
    }

    private static Tensor<float>[] MakeWeights(int layers)
    {
        // 4 weights per layer + 1 output projection at the end
        var weights = new Tensor<float>[layers * 4 + 1];
        var rng = new Random(123);
        for (int l = 0; l < layers; l++)
        {
            int offset = l * 4;
            weights[offset + 0] = MakeFloatTensor(new[] { D, 3 * D }, rng);
            weights[offset + 1] = MakeFloatTensor(new[] { D, D }, rng);
            weights[offset + 2] = MakeFloatTensor(new[] { D, FfDim }, rng);
            weights[offset + 3] = MakeFloatTensor(new[] { FfDim, D }, rng);
        }
        weights[layers * 4] = MakeFloatTensor(new[] { D, Vocab }, rng);
        return weights;
    }

    private static Tensor<float> MakeFloatTensor(int[] shape, Random rng)
    {
        int total = 1;
        foreach (var s in shape) total *= s;
        var data = new float[total];
        // Xavier-ish: small magnitude so loss stays bounded
        float scale = MathF.Sqrt(2.0f / shape[0]);
        for (int i = 0; i < total; i++) data[i] = (float)(rng.NextDouble() * 2.0 - 1.0) * scale;
        return new Tensor<float>(data, shape);
    }

    private static string ShapeStr(int[] shape) => "[" + string.Join(",", shape) + "]";

    private static double Median(double[] values)
    {
        var sorted = values.OrderBy(v => v).ToArray();
        return sorted.Length == 0 ? 0 : sorted[sorted.Length / 2];
    }
}
#endif
