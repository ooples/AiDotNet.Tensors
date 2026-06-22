using System;
using System.Collections.Generic;
using System.Diagnostics;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

// #642 P0 verification: time a single 3x3 stride-1 pad-1 Conv2D (the SD-UNet resblock
// conv, routed through FusedConvHelper.Conv2DFused) at a chosen MaxDegreeOfParallelism.
// Sweep maxdop=1..N: speedup/N = parallel efficiency = core utilization for that shape.
internal static class Program
{
    private static int Main(string[] args)
    {
        AiDotNetEngine.Current = new CpuEngine();
        var eng = (CpuEngine)AiDotNetEngine.Current;

        // #653: optionally disable the hand-emitted machine-code GEMM kernel (reflection) so the
        // SAME shape can be measured machine-code vs RyuJIT to isolate the per-core codegen gap.
        if (Environment.GetEnvironmentVariable("AIDOTNET_MK_OFF") == "1")
        {
            var mk = typeof(CpuEngine).Assembly.GetType("AiDotNet.Tensors.Engines.BlasManaged.MachineKernelGemm");
            mk?.GetProperty("Enabled", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Static)
              ?.SetValue(null, false);
            Console.Error.WriteLine("[probe] MachineKernelGemm.Enabled=false");
        }

        if (args.Length > 0 && args[0] == "--resblock") return RunResblock(eng, args);
        if (args.Length > 0 && args[0] == "--attnblock") return RunAttnBlock(eng, args);
        if (args.Length > 0 && args[0] == "--act") return RunAct(eng, args);
        if (args.Length > 0 && args[0] == "--gemm") return RunGemm(eng, args);
        if (args.Length > 0 && args[0] == "--gemmprofile") return RunGemmProfile(eng, args);
        if (args.Length > 0 && args[0] == "--gpu") return RunGpu(args);
        if (args.Length > 0 && args[0] == "--trainbench") return RunTrainbench(eng, args);
        if (args.Length > 0 && args[0] == "--flashbwd") return RunFlashBwd(eng, args);

        int maxdop = ArgI(args, "--maxdop", Environment.ProcessorCount);
        int inC = ArgI(args, "--inc", 256);
        int outC = ArgI(args, "--outc", 256);
        int sp = ArgI(args, "--sp", 16);
        int reps = ArgI(args, "--reps", 12);
        bool util = HasFlag(args, "--util");

        if (maxdop < 1 || inC < 1 || outC < 1 || sp < 1 || reps < 1)
        {
            Console.WriteLine("Error: All numeric arguments must be at least 1");
            return 1;
        }

        CpuParallelSettings.MaxDegreeOfParallelism = maxdop;

        var rng = new Random(0);
        var input = Rand(new[] { 1, inC, sp, sp }, rng);
        var kernel = Rand(new[] { outC, inC, 3, 3 }, rng);

        var sw = Stopwatch.StartNew();
        var o = eng.Conv2D(input, kernel, 1, 1, 1);   // 3x3 stride1 pad1 -> same spatial
        sw.Stop();
        double warm = sw.Elapsed.TotalMilliseconds;

        var times = new double[reps];
        using var meter = util ? ParallelUtilizationMeter.Start() : null;   // `using` disposes on exception paths too (#654 review)
        for (int i = 0; i < reps; i++)
        {
            var s = Stopwatch.StartNew();
            o = eng.Conv2D(input, kernel, 1, 1, 1);
            s.Stop();
            times[i] = s.Elapsed.TotalMilliseconds;
        }
        string u = StopMeter(meter, maxdop);
        Array.Sort(times);
        Console.WriteLine(
            $"CONV inC={inC} outC={outC} sp={sp}x{sp} out.len={o.Length} maxdop={maxdop} " +
            $"procs={Environment.ProcessorCount} warmup_ms={warm:F1} median_ms={times[reps / 2]:F2} min_ms={times[0]:F2} max_ms={times[times.Length - 1]:F2}{u}");
        return 0;
    }

    // P0 (#653): a Transformer/DiT block — the attention op-mix that a diffusion/LLM forward
    // runs (QKV projections, per-head QK^T/softmax/AV batched GEMMs, output projection,
    // residual+LayerNorm, FFN). Sweep maxdop to measure attention parallel efficiency the same
    // way --resblock measures the conv mix. This is a perf-SHAPE probe: weights/activations are
    // random and the per-head tensors are allocated at the exact attention GEMM shapes, so it
    // measures the dispatch/parallel behavior of those op shapes (not numerical attention).
    private static int RunAttnBlock(CpuEngine eng, string[] a)
    {
        int maxdop = ArgI(a, "--maxdop", Environment.ProcessorCount);
        int S = ArgI(a, "--s", 256);    // sequence length (tokens)
        int D = ArgI(a, "--d", 768);    // model dim
        int H = ArgI(a, "--h", 12);     // heads
        int blocks = ArgI(a, "--blocks", 6);
        int reps = ArgI(a, "--reps", 10);
        bool util = HasFlag(a, "--util");

        if (maxdop < 1 || S < 1 || D < 1 || H < 1 || blocks < 1 || reps < 1)
        {
            Console.Error.WriteLine("attnblock: --maxdop, --s, --d, --h, --blocks, --reps must all be >= 1.");
            return 2;
        }
        if (D % H != 0)
        {
            Console.Error.WriteLine($"attnblock: --d ({D}) must be a multiple of --h ({H}).");
            return 2;
        }
        int Dh = D / H;

        CpuParallelSettings.MaxDegreeOfParallelism = maxdop;

        var rng = new Random(0);
        var x0 = Rand(new[] { S, D }, rng);
        var wq = Rand(new[] { D, D }, rng);
        var wk = Rand(new[] { D, D }, rng);
        var wv = Rand(new[] { D, D }, rng);
        var wo = Rand(new[] { D, D }, rng);
        var w1 = Rand(new[] { D, 4 * D }, rng);
        var w2 = Rand(new[] { 4 * D, D }, rng);
        var gamma = Rand(new[] { D }, rng);
        var beta = Rand(new[] { D }, rng);
        // per-head GEMM operands at attention shapes (Q [H,S,Dh], K^T [H,Dh,S], V [H,S,Dh])
        var qh = Rand(new[] { H, S, Dh }, rng);
        var kht = Rand(new[] { H, Dh, S }, rng);
        var vh = Rand(new[] { H, S, Dh }, rng);

        Func<Tensor<float>, Tensor<float>> blk = inp =>
        {
            // QKV projections (the large [S,D]x[D,D] GEMMs)
            var q = eng.BatchMatMul(inp, wq);
            var k = eng.BatchMatMul(inp, wk);
            var v = eng.BatchMatMul(inp, wv);
            float guard = q[0] + k[0] + v[0];
            // per-head scores -> softmax -> context (the many small batched GEMMs)
            var scores = eng.BatchMatMul(qh, kht);     // [H,S,S]
            scores = eng.Softmax(scores, -1);
            var ctx = eng.BatchMatMul(scores, vh);     // [H,S,Dh]
            guard += ctx[0];
            // output projection + residual + norm
            var ao = eng.BatchMatMul(inp, wo);         // [S,D]
            ao = eng.TensorAdd(inp, ao);
            ao = eng.LayerNorm(ao, gamma, beta, 1e-5, out _, out _);
            // FFN (D -> 4D -> D)
            var h1 = eng.BatchMatMul(ao, w1);          // [S,4D]
            eng.SwishInPlace(h1);
            var h2 = eng.BatchMatMul(h1, w2);          // [S,D]
            var y = eng.TensorAdd(ao, h2);
            y = eng.LayerNorm(y, gamma, beta, 1e-5, out _, out _);
            if (guard == float.PositiveInfinity) Console.Write(""); // keep guard live
            return y;
        };

        var sw = Stopwatch.StartNew();
        var yy = x0;
        for (int b = 0; b < blocks; b++) yy = blk(yy);
        sw.Stop();
        double warm = sw.Elapsed.TotalMilliseconds;

        var times = new double[reps];
        using var meter = util ? ParallelUtilizationMeter.Start() : null;   // `using` disposes on exception paths too (#654 review)
        for (int i = 0; i < reps; i++)
        {
            var s = Stopwatch.StartNew();
            yy = x0;
            for (int b = 0; b < blocks; b++) yy = blk(yy);
            s.Stop();
            times[i] = s.Elapsed.TotalMilliseconds;
        }
        string u = StopMeter(meter, maxdop);
        Array.Sort(times);
        Console.WriteLine(
            $"ATTNBLOCK S={S} D={D} H={H} Dh={Dh} blocks={blocks} maxdop={maxdop} procs={Environment.ProcessorCount} " +
            $"warmup_ms={warm:F1} median_ms={times[reps / 2]:F2} min_ms={times[0]:F2} max_ms={times[times.Length - 1]:F2}{u}");
        return 0;
    }

    // #1662 lever #4 / #1 proof harness: per-step TRAINING cost (fwd + bwd + optimizer step)
    // for a residual-FFN stack on the autodiff tape. Backward runs through
    // tape.ComputeGradientsStreaming (the optimizer-in-backward path), so this isolates the
    // per-step backward ALLOCATION (the audit target) and per-step wall time. Emits one
    // machine-readable line so trainbench_torch.py (same shape/optimizer) can be diffed
    // against it to prove AiDotNet beats PyTorch CPU on time / alloc / peak-RSS.
    //
    // Shape mirrors the #1624 acceptance model's FFN block (SimCSE dim=384, 10 layers). A
    // residual MLP stack (TensorMatMul + GELU) deliberately hammers matmul-backward — the
    // hottest backward op and the primary arena-bypass suspect — and records robustly on the
    // tape (attention's manual Backward is not a tape op, so it is exercised by the #3 parity
    // test instead, not here).
    private static int RunTrainbench(CpuEngine eng, string[] a)
    {
        int maxdop = ArgI(a, "--maxdop", Environment.ProcessorCount);
        int S = ArgI(a, "--s", 128);        // sequence length (rows) / conv spatial
        int D = ArgI(a, "--d", 384);        // model dim / conv channels
        int layers = ArgI(a, "--layers", 10);
        int reps = ArgI(a, "--reps", 20);
        int warmup = ArgI(a, "--warmup", 5);
        string block = ArgS(a, "--block", "mlp"); // mlp | attn | conv
        if (maxdop < 1 || S < 1 || D < 1 || layers < 1 || reps < 1 || warmup < 0)
        {
            Console.Error.WriteLine("trainbench: --maxdop,--s,--d,--layers,--reps must be >= 1 (--warmup >= 0).");
            return 2;
        }
        CpuParallelSettings.MaxDegreeOfParallelism = maxdop;
        // Real training (NeuralNetworkBase) wraps each step in a per-step TensorArena that
        // recycles fwd+bwd scratch. Mirror that by default so the metric reflects the REAL
        // per-step churn (what bypasses the arena) — the lever #4 audit target. --no-arena
        // measures the un-pooled worst case.
        bool arenaOn = !HasFlag(a, "--no-arena");

        var rng = new Random(0);
        ITrainStep step = block switch
        {
            "attn" => new TrainbenchAttnStep(eng, S, D, layers, rng),
            "conv" => new TrainbenchConvStep(eng, S, D, layers, rng),
            "mlp"  => new TrainbenchStep(eng, S, D, layers, rng),
            _      => null!,
        };
        if (step is null) { Console.Error.WriteLine($"trainbench: --block must be mlp|attn|conv (got '{block}')."); return 2; }

        // Weights are constructed above (before the arena) so they stay persistent across
        // Reset(); only per-step scratch recycles.
        using var arena = arenaOn ? TensorArena.Create() : null;

        for (int i = 0; i < warmup; i++) { arena?.Reset(); step.RunStep(); }

        var times = new double[reps];
        long peakWs = 0;
        long allocStart = GC.GetAllocatedBytesForCurrentThread();
        for (int i = 0; i < reps; i++)
        {
            arena?.Reset();
            var sw = Stopwatch.StartNew();
            step.RunStep();
            sw.Stop();
            times[i] = sw.Elapsed.TotalMilliseconds;
            peakWs = Math.Max(peakWs, Process.GetCurrentProcess().WorkingSet64);
        }
        long allocTotal = GC.GetAllocatedBytesForCurrentThread() - allocStart;
        Array.Sort(times);
        double perStepAllocMB = allocTotal / (double)reps / (1024.0 * 1024.0);
        Console.WriteLine(
            $"TRAINBENCH engine=aidotnet block={block} arena={(arenaOn ? "on" : "off")} S={S} D={D} layers={layers} maxdop={maxdop} " +
            $"procs={Environment.ProcessorCount} median_ms={times[reps / 2]:F3} min_ms={times[0]:F3} " +
            $"alloc_mb_per_step={perStepAllocMB:F3} peak_ws_mb={peakWs / (1024.0 * 1024.0):F1} " +
            $"last_loss={step.LastLoss:E3}");
        return 0;
    }

    // #1662 lever #3 memory proof: peak LIVE managed set during one FusedAttention.Backward at a
    // long sequence, tiled (default) vs --full (forces the full-matrix path at the same shape).
    // The full path keeps the [B,H,S,S] score/prob matrices simultaneously reachable (O(S^2));
    // the tiled path holds only one O(S*tile) tile + the outputs at a time. The sampler FORCES a
    // collection per sample (GC.GetTotalMemory(true)) so only REACHABLE bytes count — otherwise
    // the tiled path's transient per-tile garbage (it recomputes scores 3x) would dominate and
    // hide the peak-live win. Timing is perturbed by the forced GCs, so this mode reports memory,
    // not speed (use --trainbench / plain backward_ms elsewhere for timing).
    private static int RunFlashBwd(CpuEngine eng, string[] a)
    {
        int B = ArgI(a, "--b", 1);
        int H = ArgI(a, "--h", 8);
        int S = ArgI(a, "--s", 2048);
        int Dh = ArgI(a, "--dh", 64);
        bool full = HasFlag(a, "--full");
        if (B < 1 || H < 1 || S < 1 || Dh < 1) { Console.Error.WriteLine("flashbwd: --b,--h,--s,--dh must be >= 1."); return 2; }

        var rng = new Random(0);
        var q = Rand(new[] { B, H, S, Dh }, rng);
        var k = Rand(new[] { B, H, S, Dh }, rng);
        var v = Rand(new[] { B, H, S, Dh }, rng);
        var dO = Rand(new[] { B, H, S, Dh }, rng);

        var toggle = typeof(FusedAttention<float>).GetField(
            "ForceFullBackwardForBench",
            System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Static);
        toggle?.SetValue(null, full);

        // Warmup (JIT + first-touch); result discarded.
        var (wq, _, _) = FusedAttention<float>.Backward(dO, q, k, v, engine: eng);
        if (wq[0] == float.PositiveInfinity) Console.Write("");

        GC.Collect(); GC.WaitForPendingFinalizers(); GC.Collect();
        long baseline = GC.GetTotalMemory(true);
        long peak = baseline;
        bool stop = false;
        var sampler = new System.Threading.Thread(() =>
        {
            while (!System.Threading.Volatile.Read(ref stop))
            {
                long m = GC.GetTotalMemory(true); // forced collection -> reachable bytes only
                if (m > peak) peak = m;
                System.Threading.Thread.Sleep(10);
            }
        }) { IsBackground = true };
        sampler.Start();

        var sw = Stopwatch.StartNew();
        var (gq, gk, gv) = FusedAttention<float>.Backward(dO, q, k, v, engine: eng);
        sw.Stop();

        System.Threading.Volatile.Write(ref stop, true);
        sampler.Join();
        toggle?.SetValue(null, false); // restore

        float sink = gq[0] + gk[0] + gv[0];
        Console.WriteLine(
            $"FLASHBWD path={(full ? "full" : "tiled")} B={B} H={H} S={S} Dh={Dh} " +
            $"backward_ms={sw.Elapsed.TotalMilliseconds:F1} peak_managed_mb={(peak - baseline) / (1024.0 * 1024.0):F1} " +
            $"(sink={sink:E1})");
        return 0;
    }

    // A single training block whose per-step fwd+bwd allocation the trainbench measures.
    private interface ITrainStep { void RunStep(); float LastLoss { get; } }

    private const float TbLr = 1e-3f;

    // In-place SGD update applied inside the streaming-backward callback.
    private static void SgdApply(Tensor<float> src, Tensor<float> g)
    {
        if (g is null || g.Length == 0) return;
        for (int i = 0; i < src.Length; i++) src[i] -= TbLr * g[i];
    }

    private static Tensor<float> Scaled(Tensor<float> t, float s)
    {
        for (int i = 0; i < t.Length; i++) t[i] *= s;
        return t;
    }

    // Owns the residual-FFN weights and runs one fwd+bwd+SGD step on the tape. SGD (not Adam)
    // so the optimizer step itself contributes ~no allocation — the metric isolates fwd+bwd.
    private sealed class TrainbenchStep : ITrainStep
    {
        private readonly CpuEngine _eng;
        private readonly int _layers;
        private readonly Tensor<float>[] _w1; // [D, 4D]
        private readonly Tensor<float>[] _w2; // [4D, D]
        private readonly Tensor<float> _x;    // [S, D] fixed input

        public float LastLoss { get; private set; }

        public TrainbenchStep(CpuEngine eng, int s, int d, int layers, Random rng)
        {
            _eng = eng; _layers = layers;
            _x = Rand(new[] { s, d }, rng);
            _w1 = new Tensor<float>[layers];
            _w2 = new Tensor<float>[layers];
            // Scale weights small so the residual stack stays numerically tame over many steps.
            for (int l = 0; l < layers; l++)
            {
                _w1[l] = Scaled(Rand(new[] { d, 4 * d }, rng), 0.02f);
                _w2[l] = Scaled(Rand(new[] { 4 * d, d }, rng), 0.02f);
            }
        }

        public void RunStep()
        {
            using var tape = new GradientTape<float>();
            var h = _x;
            for (int l = 0; l < _layers; l++)
            {
                var f = _eng.TensorMatMul(h, _w1[l]);   // [S, 4D]
                f = _eng.GELU(f);
                f = _eng.TensorMatMul(f, _w2[l]);        // [S, D]
                h = _eng.TensorAdd(h, f);                // residual
            }
            // Scalar MSE-to-zero loss so the whole stack is on one connected graph.
            var loss = _eng.ReduceSum(_eng.TensorMultiply(h, h));
            LastLoss = loss.Length > 0 ? loss[0] : 0f;

            var sources = new List<Tensor<float>>(_layers * 2);
            for (int l = 0; l < _layers; l++) { sources.Add(_w1[l]); sources.Add(_w2[l]); }
            tape.ComputeGradientsStreaming(loss, sources, SgdApply);
        }
    }

    // Single-head attention block (q@k^T -> softmax -> @v -> proj -> residual -> LayerNorm),
    // built from tape primitives so Softmax/LayerNorm/MatMulTransposed BACKWARD are exercised
    // — the attention backward path #1662 lever #4 names. Verifies it stays arena-bounded.
    private sealed class TrainbenchAttnStep : ITrainStep
    {
        private readonly CpuEngine _eng;
        private readonly int _layers, _d;
        private readonly float _scale;
        private readonly Tensor<float>[] _wq, _wk, _wv, _wo, _gamma, _beta;
        private readonly Tensor<float> _x; // [S, D]

        public float LastLoss { get; private set; }

        public TrainbenchAttnStep(CpuEngine eng, int s, int d, int layers, Random rng)
        {
            _eng = eng; _layers = layers; _d = d;
            _scale = (float)(1.0 / Math.Sqrt(d));
            _x = Rand(new[] { s, d }, rng);
            _wq = new Tensor<float>[layers]; _wk = new Tensor<float>[layers];
            _wv = new Tensor<float>[layers]; _wo = new Tensor<float>[layers];
            _gamma = new Tensor<float>[layers]; _beta = new Tensor<float>[layers];
            for (int l = 0; l < layers; l++)
            {
                _wq[l] = Scaled(Rand(new[] { d, d }, rng), 0.02f);
                _wk[l] = Scaled(Rand(new[] { d, d }, rng), 0.02f);
                _wv[l] = Scaled(Rand(new[] { d, d }, rng), 0.02f);
                _wo[l] = Scaled(Rand(new[] { d, d }, rng), 0.02f);
                _gamma[l] = Scaled(Rand(new[] { d }, rng), 0f); // gamma=0+1 below
                for (int i = 0; i < d; i++) _gamma[l][i] = 1f;
                _beta[l] = Scaled(Rand(new[] { d }, rng), 0f);
            }
        }

        public void RunStep()
        {
            using var tape = new GradientTape<float>();
            var h = _x;
            for (int l = 0; l < _layers; l++)
            {
                var q = _eng.TensorMatMul(h, _wq[l]);                 // [S, D]
                var k = _eng.TensorMatMul(h, _wk[l]);
                var v = _eng.TensorMatMul(h, _wv[l]);
                var scores = _eng.TensorMatMulTransposed(q, k);       // q @ k^T -> [S, S]
                scores = _eng.TensorMultiplyScalar(scores, _scale);
                var p = _eng.Softmax(scores, -1);                     // [S, S]
                var ctx = _eng.TensorMatMul(p, v);                    // [S, D]
                var o = _eng.TensorMatMul(ctx, _wo[l]);               // [S, D]
                var res = _eng.TensorAdd(h, o);                       // residual
                h = _eng.TensorLayerNorm(res, _gamma[l], _beta[l]);   // LayerNorm
            }
            var loss = _eng.ReduceSum(_eng.TensorMultiply(h, h));
            LastLoss = loss.Length > 0 ? loss[0] : 0f;

            var sources = new List<Tensor<float>>(_layers * 6);
            for (int l = 0; l < _layers; l++)
            { sources.Add(_wq[l]); sources.Add(_wk[l]); sources.Add(_wv[l]); sources.Add(_wo[l]); sources.Add(_gamma[l]); sources.Add(_beta[l]); }
            tape.ComputeGradientsStreaming(loss, sources, SgdApply);
        }
    }

    // Residual conv stack (Conv2D 3x3 pad-1 -> GELU -> residual) so Conv2D BACKWARD (im2col /
    // col2im scratch) is exercised — the conv backward path #1662 lever #4 names.
    private sealed class TrainbenchConvStep : ITrainStep
    {
        private readonly CpuEngine _eng;
        private readonly int _layers;
        private readonly Tensor<float>[] _kernels; // [C, C, 3, 3]
        private readonly Tensor<float> _x;         // [1, C, sp, sp]

        public float LastLoss { get; private set; }

        public TrainbenchConvStep(CpuEngine eng, int sp, int channels, int layers, Random rng)
        {
            _eng = eng; _layers = layers;
            _x = Rand(new[] { 1, channels, sp, sp }, rng);
            _kernels = new Tensor<float>[layers];
            for (int l = 0; l < layers; l++)
                _kernels[l] = Scaled(Rand(new[] { channels, channels, 3, 3 }, rng), 0.02f);
        }

        public void RunStep()
        {
            using var tape = new GradientTape<float>();
            var h = _x;
            for (int l = 0; l < _layers; l++)
            {
                var y = _eng.Conv2D(h, _kernels[l], 1, 1, 1); // 3x3 stride1 pad1 -> same spatial
                y = _eng.GELU(y);
                h = _eng.TensorAdd(h, y);                     // residual
            }
            var loss = _eng.ReduceSum(_eng.TensorMultiply(h, h));
            LastLoss = loss.Length > 0 ? loss[0] : 0f;

            var sources = new List<Tensor<float>>(_layers);
            for (int l = 0; l < _layers; l++) sources.Add(_kernels[l]);
            tape.ComputeGradientsStreaming(loss, sources, SgdApply);
        }
    }

    // P4 (#653): the elementwise-activation stragglers (GELU/Swish/Tanh/Sigmoid) on a single
    // large contiguous tensor — an FFN intermediate ([S,4D]) or UNet feature map. These ran
    // SIMD-but-single-threaded between the parallelized GEMMs; this probe times them out-of-place
    // (the *Into path the forward DAG uses) at a chosen maxdop so we can see them fan across cores.
    private static int RunAct(CpuEngine eng, string[] a)
    {
        int maxdop = ArgI(a, "--maxdop", Environment.ProcessorCount);
        int n = ArgI(a, "--n", 8 * 1024 * 1024);  // 8M floats ~= S=2048 x 4D=4096 FFN intermediate
        int reps = ArgI(a, "--reps", 40);
        bool util = HasFlag(a, "--util");
        string op = a.Length > 1 && !a[1].StartsWith("--") ? a[1] : "gelu";

        if (maxdop < 1 || n < 1 || reps < 1)
        {
            Console.Error.WriteLine("act: --maxdop, --n, --reps must all be >= 1.");
            return 2;
        }

        CpuParallelSettings.MaxDegreeOfParallelism = maxdop;

        var rng = new Random(0);
        var src = Rand(new[] { n }, rng);
        var dst = new Tensor<float>(new[] { n });

        Action run = op switch
        {
            "swish"   => () => eng.SwishInto(dst, src),
            "tanh"    => () => eng.TanhInto(dst, src),
            "sigmoid" => () => eng.SigmoidInto(dst, src),
            "relu"    => () => eng.ReLUInto(dst, src),
            "mish"    => () => eng.MishInto(dst, src),
            _         => () => eng.GELUInto(dst, src),
        };

        run(); // warmup (untimed)
        double sink = dst[0];

        var times = new double[reps];
        using var meter = util ? ParallelUtilizationMeter.Start() : null;   // `using` disposes on exception paths too (#654 review)
        for (int i = 0; i < reps; i++)
        {
            var s = Stopwatch.StartNew();
            run();
            s.Stop();
            times[i] = s.Elapsed.TotalMilliseconds;
        }
        string u = StopMeter(meter, maxdop);
        Array.Sort(times);
        Console.WriteLine(
            $"ACT op={op} n={n} maxdop={maxdop} procs={Environment.ProcessorCount} " +
            $"median_ms={times[reps / 2]:F3} min_ms={times[0]:F3} max_ms={times[times.Length - 1]:F3} (sink={sink:E1}){u}");
        return 0;
    }

    // P4 (#653): a single dense GEMM [M,K]x[K,N] — the transformer QKV/O projection and FFN
    // matmul shapes — at a chosen maxdop. Isolates whether intra-op GEMM parallelism saturates
    // cores (the cooperative-pool fan-out), separate from the block-level barriers/small ops.
    private static int RunGemm(CpuEngine eng, string[] a)
    {
        int maxdop = ArgI(a, "--maxdop", Environment.ProcessorCount);
        int M = ArgI(a, "--m", 512);
        int K = ArgI(a, "--k", 1024);
        int N = ArgI(a, "--n", 4096);
        int reps = ArgI(a, "--reps", 30);
        bool util = HasFlag(a, "--util");

        if (maxdop < 1 || M < 1 || K < 1 || N < 1 || reps < 1)
        {
            Console.Error.WriteLine("gemm: --maxdop, --m, --k, --n, --reps must all be >= 1.");
            return 2;
        }

        CpuParallelSettings.MaxDegreeOfParallelism = maxdop;

        var rng = new Random(0);
        var lhs = Rand(new[] { M, K }, rng);
        var rhs = Rand(new[] { K, N }, rng);

        var o = eng.BatchMatMul(lhs, rhs); // warm
        float sink = o[0];

        var times = new double[reps];
        using var meter = util ? ParallelUtilizationMeter.Start() : null;   // `using` disposes on exception paths too (#654 review)
        for (int i = 0; i < reps; i++)
        {
            var s = Stopwatch.StartNew();
            o = eng.BatchMatMul(lhs, rhs);
            s.Stop();
            times[i] = s.Elapsed.TotalMilliseconds;
        }
        string u = StopMeter(meter, maxdop);
        Array.Sort(times);
        Console.WriteLine(
            $"GEMM M={M} K={K} N={N} fma={(double)M * K * N:E1} maxdop={maxdop} procs={Environment.ProcessorCount} " +
            $"median_ms={times[reps / 2]:F3} min_ms={times[0]:F3} max_ms={times[times.Length - 1]:F3} (sink={sink:E1}){u}");
        return 0;
    }

    // P4 (#653): single-thread pack-vs-kernel attribution via PackBothProfiler (reflection,
    // since it's internal). Answers whether the per-core gap is the RyuJIT microkernel or the
    // pack/blocking overhead — which decides whether a machine-code microkernel is worth it.
    private static int RunGemmProfile(CpuEngine eng, string[] a)
    {
        int M = ArgI(a, "--m", 512);
        int K = ArgI(a, "--k", 1024);
        int N = ArgI(a, "--n", 4096);
        int reps = ArgI(a, "--reps", 20);
        CpuParallelSettings.MaxDegreeOfParallelism = 1; // single-thread attribution

        var asm = typeof(CpuEngine).Assembly;
        var prof = asm.GetType("AiDotNet.Tensors.Engines.BlasManaged.PackBothProfiler");
        if (prof == null) { Console.Error.WriteLine("PackBothProfiler not found"); return 2; }
        const System.Reflection.BindingFlags SF = System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Static;
        var enabledF = prof.GetField("Enabled", SF);
        var resetM = prof.GetMethod("Reset", SF);
        var packB = prof.GetProperty("PackBMs", SF);
        var packA = prof.GetProperty("PackAMs", SF);
        var kernel = prof.GetProperty("KernelMs", SF);

        var rng = new Random(0);
        var lhs = Rand(new[] { M, K }, rng);
        var rhs = Rand(new[] { K, N }, rng);
        var o = eng.BatchMatMul(lhs, rhs); // warm
        float sink = o[0];

        enabledF.SetValue(null, true);
        resetM.Invoke(null, null);
        var sw = Stopwatch.StartNew();
        for (int i = 0; i < reps; i++) o = eng.BatchMatMul(lhs, rhs);
        sw.Stop();
        enabledF.SetValue(null, false);

        double total = sw.Elapsed.TotalMilliseconds;
        double pb = (double)packB.GetValue(null), pa = (double)packA.GetValue(null), kr = (double)kernel.GetValue(null);
        double accounted = pb + pa + kr;
        double gflops = reps * 2.0 * M * K * N / (total / 1000.0) / 1e9;
        Console.WriteLine(
            $"GEMMPROFILE M={M} K={K} N={N} reps={reps} total_ms={total:F1} GFLOPs={gflops:F1} | " +
            $"kernel_ms={kr:F1} ({100*kr/accounted:F0}%) packA_ms={pa:F1} ({100*pa/accounted:F0}%) " +
            $"packB_ms={pb:F1} ({100*pb/accounted:F0}%) accounted_ms={accounted:F1} (sink={sink:E1})");
        return 0;
    }

    // P2 (#642): an SD-style ResBlock stack (GroupNorm -> Swish -> Conv -> GroupNorm ->
    // Swish -> Conv -> residual add) — the real per-op mix of a diffusion forward. Sweep
    // maxdop to see whether the WHOLE stack (not just conv) saturates cores after the conv
    // fixes, and profile to find the next bottleneck (norms/activations/elementwise/barriers).
    private static int RunResblock(CpuEngine eng, string[] a)
    {
        int maxdop = ArgI(a, "--maxdop", Environment.ProcessorCount);
        int C = ArgI(a, "--c", 256);
        int sp = ArgI(a, "--sp", 16);
        int blocks = ArgI(a, "--blocks", 8);
        int reps = ArgI(a, "--reps", 10);
        const int groups = 32;

        CpuParallelSettings.MaxDegreeOfParallelism = maxdop;

        var rng = new Random(0);
        var x0 = Rand(new[] { 1, C, sp, sp }, rng);
        var gamma = Rand(new[] { C }, rng);
        var beta = Rand(new[] { C }, rng);
        var k1 = Rand(new[] { C, C, 3, 3 }, rng);
        var k2 = Rand(new[] { C, C, 3, 3 }, rng);

        Func<Tensor<float>, Tensor<float>> block = x =>
        {
            var h = eng.GroupNorm(x, groups, gamma, beta, 1e-5, out _, out _);
            eng.SwishInPlace(h);
            h = eng.Conv2D(h, k1, 1, 1, 1);
            h = eng.GroupNorm(h, groups, gamma, beta, 1e-5, out _, out _);
            eng.SwishInPlace(h);
            h = eng.Conv2D(h, k2, 1, 1, 1);
            return eng.TensorAdd(x, h);
        };

        var sw = Stopwatch.StartNew();
        var y = x0;
        for (int b = 0; b < blocks; b++) y = block(y);
        sw.Stop();
        double warm = sw.Elapsed.TotalMilliseconds;

        var times = new double[reps];
        var meter = HasFlag(a, "--util") ? ParallelUtilizationMeter.Start() : null;
        for (int i = 0; i < reps; i++)
        {
            var s = Stopwatch.StartNew();
            y = x0;
            for (int b = 0; b < blocks; b++) y = block(y);
            s.Stop();
            times[i] = s.Elapsed.TotalMilliseconds;
        }
        string u = StopMeter(meter, maxdop);
        Array.Sort(times);
        Console.WriteLine(
            $"RESBLOCK C={C} sp={sp}x{sp} blocks={blocks} maxdop={maxdop} procs={Environment.ProcessorCount} " +
            $"warmup_ms={warm:F1} median_ms={times[reps / 2]:F2} min_ms={times[0]:F2} max_ms={times[times.Length - 1]:F2}{u}");
        return 0;
    }

    // P3 (#642): GPU utilization baseline. Drives the GPU engine through a sustained
    // Conv2D loop (the diffusion-relevant op) so nvidia-smi can sample GPU util, and
    // compares ms/op to the CPU path to detect a silent CPU fallback (DirectGpuTensorEngine
    // does NOT override Conv2D). Reads the output each iter to force GPU->host sync so we
    // measure real completed work, not just queued launches.
    private static int RunGpu(string[] args)
    {
        int secs = ArgI(args, "--secs", 12);
        int C = ArgI(args, "--c", 256);
        int sp = ArgI(args, "--sp", 16);
        bool fused = ArgI(args, "--fused", 0) != 0;

        var gpu = new AiDotNet.Tensors.Engines.DirectGpuTensorEngine();
        AiDotNetEngine.Current = gpu;
        Console.WriteLine($"engine={gpu.GetType().Name} SupportsGpu={gpu.SupportsGpu}");

        bool rb = ArgI(args, "--rb", 0) != 0;
        var rng = new Random(0);
        var input = Rand(new[] { 1, C, sp, sp }, rng);
        var kernel = Rand(new[] { C, C, 3, 3 }, rng);
        var k2 = Rand(new[] { C, C, 3, 3 }, rng);
        var gamma = Rand(new[] { C }, rng);
        var beta = Rand(new[] { C }, rng);

        int groups = ArgI(args, "--groups", 32);
        bool noAdd = ArgI(args, "--noadd", 0) != 0;        // skip the residual add (localize corruption)
        bool gnOnly = ArgI(args, "--gnonly", 0) != 0;      // just GroupNorm (isolate the norm op)
        Func<Tensor<float>> oneIter = rb
            ? () =>
            {
                if (gnOnly) return gpu.GroupNorm(input, groups, gamma, beta, 1e-5, out _, out _);
                // ResBlock: GN -> Swish -> Conv -> GN -> Swish -> Conv -> residual add
                var h = gpu.GroupNorm(input, groups, gamma, beta, 1e-5, out _, out _);
                gpu.SwishInPlace(h);
                h = gpu.Conv2D(h, kernel, 1, 1, 1);
                h = gpu.GroupNorm(h, groups, gamma, beta, 1e-5, out _, out _);
                gpu.SwishInPlace(h);
                h = gpu.Conv2D(h, k2, 1, 1, 1);
                return noAdd ? h : gpu.TensorAdd(input, h);
            }
        : () => gpu.Conv2D(input, kernel, 1, 1, 1);

        // --capture: #642 P3 option B endgame. Record the resident op graph, then CUDA-graph
        // CAPTURE its compute-kernel sequence (H2D uploads stay outside capture — they cuMemAlloc,
        // which is illegal mid-capture) and REPLAY it with zero per-launch overhead. Validates
        // (a) the captured graph replays correctly (output matches eager) and (b) the launch-cost
        // reduction vs issuing the same kernels directly.
        if (ArgI(args, "--capture", 0) != 0)
            return RunCapture(gpu, oneIter, rb ? "ResBlock" : "Conv2D");

        // --deferred: run each iter inside a DeferredScope so ops record into the fused
        // execution graph (device-resident, multi-stream) instead of executing eagerly per op
        // — the substrate a CUDA graph would capture (#642 P3 option B).
        bool deferred = ArgI(args, "--deferred", 0) != 0;
        Func<Tensor<float>> baseIter = oneIter;
        if (deferred)
        {
            bool bufdump = ArgI(args, "--bufdump", 0) != 0;
            oneIter = () =>
            {
                using var scope = gpu.BeginDeferredScope();
                var r = baseIter();
                var ds = scope as AiDotNet.Tensors.Engines.Gpu.DeferredScope;
                if (ds != null && bufdump)
                {
                    // #642 P3: dump COMPILED-graph node buffer handles to find the residual-add
                    // aliasing collision. H(handle) collisions across nodes with overlapping
                    // lifetimes = two logical buffers sharing one device pointer.
                    var g = ds.Compile();
                    int idx = 0;
                    string Hx(AiDotNet.Tensors.Engines.DirectGpu.IGpuBuffer? b)
                        => b == null ? "null" : $"0x{b.Handle.ToInt64():X}/{b.Size}";
                    foreach (var node in g.TopologicalOrder)
                    {
                        var line = new System.Text.StringBuilder($"  [{idx++}] {node.NodeType}");
                        if (node is AiDotNet.Tensors.Engines.Gpu.Graph.KernelNode kn)
                        {
                            line.Append(':').Append(kn.KernelType).Append(" in=[");
                            foreach (var t in kn.InputTensors) line.Append(Hx(t.TryGetGpuBuffer())).Append(' ');
                            line.Append("] out=[");
                            foreach (var t in kn.OutputTensors) line.Append(Hx(t.TryGetGpuBuffer())).Append(' ');
                            line.Append(']');
                        }
                        else if (node is AiDotNet.Tensors.Engines.Gpu.Graph.TransferNode tn)
                            line.Append(' ').Append(tn.TransferType).Append(" src=").Append(Hx(tn.SourceBuffer)).Append(" dst=").Append(Hx(tn.DestinationBuffer));
                        line.Append($" deps={node.Dependencies.Count}");
                        Console.WriteLine(line.ToString());
                    }
                }
                else if (ds != null)
                {
                    var sb = new System.Text.StringBuilder("  NODES: ");
                    foreach (var node in ds.GraphBuilder.Nodes)
                    {
                        sb.Append(node.NodeType);
                        if (node is AiDotNet.Tensors.Engines.Gpu.Graph.KernelNode kn) sb.Append(':').Append(kn.KernelType);
                        sb.Append('[').Append(node.Dependencies.Count).Append("dep] ");
                    }
                    Console.WriteLine(sb.ToString());
                }
                scope?.Execute();
                var st = (scope as AiDotNet.Tensors.Engines.Gpu.DeferredScope)?.GetStatistics();
                if (st != null)
                    Console.WriteLine($"  GRAPH recorded={st.OperationsRecorded} afterCompile={st.NodesAfterCompilation} eliminated={st.EliminatedOperations} fused={st.FusedOperations}");
                return r;
            };

            // Correctness gate: deferred output must match eager output for the same input.
            // --deffirst: run the deferred path BEFORE the eager reference, to test whether a
            // prior eager run pollutes `input`'s cached GPU buffer (eager→deferred staleness).
            bool defFirst = ArgI(args, "--deffirst", 0) != 0;
            try
            {
                Tensor<float> eagerO, defO;
                if (defFirst) { defO = oneIter(); eagerO = baseIter(); }
                else { eagerO = baseIter(); defO = oneIter(); }
                double maxAbsDiff = 0, defAbsMax = 0, defNonZero = 0;
                int len = Math.Min(eagerO.Length, defO.Length);
                for (int i = 0; i < len; i++)
                {
                    maxAbsDiff = Math.Max(maxAbsDiff, Math.Abs((double)eagerO[i] - defO[i]));
                    defAbsMax = Math.Max(defAbsMax, Math.Abs((double)defO[i]));
                    if (defO[i] != 0) defNonZero++;
                }
                Console.WriteLine($"CORRECTNESS deferred-vs-eager maxAbsDiff={maxAbsDiff:E3} (len={len}) defAbsMax={defAbsMax:E3} defNonZero={defNonZero / len:P0}");
                Console.WriteLine($"  eager[0..4]={eagerO[0]:F4} {eagerO[1]:F4} {eagerO[2]:F4} {eagerO[3]:F4}");
                Console.WriteLine($"  defrd[0..4]={defO[0]:F4} {defO[1]:F4} {defO[2]:F4} {defO[3]:F4}");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"DEFERRED-PATH-FAILED: {ex.GetType().Name}: {ex.Message}");
                return 3;
            }
        }

        var w = Stopwatch.StartNew();
        var o = oneIter();
        float sink = o[0];   // force materialize / GPU->host sync (kernel compile + first run)
        w.Stop();
        Console.WriteLine($"warmup {(rb ? "resblock" : "conv")}{(deferred ? "+deferred" : "")} {w.Elapsed.TotalMilliseconds:F0}ms (sink={sink:E2})");

        int syncEvery = ArgI(args, "--syncevery", 1);
        long n = 0;
        var sw = Stopwatch.StartNew();
        while (sw.Elapsed.TotalMilliseconds < secs * 1000.0)
        {
            o = oneIter();
            if (n % syncEvery == 0) sink += o[0];   // host->GPU sync every N iters
            n++;
        }
        sink += o[0]; // final sync
        sw.Stop();
        Console.WriteLine(
            $"GPU {(rb ? "ResBlock" : "Conv2D")} [1,{C},{sp}x{sp}] {n} iters / {sw.Elapsed.TotalSeconds:F1}s = " +
            $"{sw.Elapsed.TotalMilliseconds / n:F3} ms/iter  ({n / sw.Elapsed.TotalSeconds:F0} iters/s)  sink={sink:E2}");
        return 0;
    }

    // #642 P3 option B: CUDA-graph CAPTURE + REPLAY of the resident deferred graph's compute
    // kernels. The recorded graph allocates all buffers at RECORD time, so the kernel sequence
    // replayed by Execute is alloc-free — the capturable subset. H2D upload nodes (TransferNode)
    // cuMemAlloc a temp buffer per run, which is illegal during capture, so they're run only in
    // warmup (to populate the resident buffers) and SKIPPED inside the captured region.
    private static int RunCapture(AiDotNet.Tensors.Engines.DirectGpuTensorEngine gpu, Func<Tensor<float>> work, string label)
    {
        var backend = gpu.GetBackend();
        if (backend == null) { Console.WriteLine("CAPTURE: no GPU backend"); return 2; }

        // Eager reference (no scope) for the correctness check.
        var eager = work();
        var eagerCopy = new float[eager.Length];
        for (int i = 0; i < eager.Length; i++) eagerCopy[i] = eager[i];

        // Record the op graph into a deferred scope (NOT `using` — keep the recorded buffers alive
        // through capture/replay; the scope's release-deferral gate frees them on Dispose at the end).
        if (gpu.BeginDeferredScope() is not AiDotNet.Tensors.Engines.Gpu.DeferredScope scope)
        {
            Console.WriteLine("CAPTURE: BeginDeferredScope did not return a DeferredScope (GPU unavailable?)");
            return 2;
        }
        Tensor<float> r;
        AiDotNet.Tensors.Engines.Gpu.Graph.ExecutionGraph graph;
        try
        {
            r = work();                 // records H2D + resident compute kernels (buffers allocated now)
            graph = scope.Compile();    // ends recording, runs optimizer passes
        }
        catch (Exception ex) { Console.WriteLine($"CAPTURE record/compile FAILED: {ex.GetType().Name}: {ex.Message}"); scope.Dispose(); return 3; }

        int kernelCount = 0, transferCount = 0;
        foreach (var node in graph.TopologicalOrder)
        {
            if (node is AiDotNet.Tensors.Engines.Gpu.Graph.KernelNode) kernelCount++;
            else if (node is AiDotNet.Tensors.Engines.Gpu.Graph.TransferNode) transferCount++;
        }
        Console.WriteLine($"CAPTURE {label}: {graph.TopologicalOrder.Count} nodes ({kernelCount} kernels, {transferCount} transfers)");

        // Warmup: run the full graph once (H2D populates resident buffers + JIT compiles kernels).
        foreach (var node in graph.TopologicalOrder) node.Execute(backend);
        backend.Synchronize();

        if (backend is not AiDotNet.Tensors.Engines.DirectGpu.CUDA.CudaBackend cudaBackend)
        { Console.WriteLine($"CAPTURE: backend {backend.GetType().Name} is not CUDA — capture path is CUDA-only"); scope.Dispose(); return 0; }

        try
        {
            // Drive capture/replay through the PRODUCTION API: GraphedInferenceStep wraps
            // Prepare(warmup) -> Capture(BeginCapture; forward; EndCapture) -> Replay. The forward
            // closure is the capture-safe compute-kernel subset (no H2D alloc, no final sync).
            using var step = new AiDotNet.Tensors.Engines.Compilation.Codegen.CudaGraph.GraphedInferenceStep(
                backend, cudaBackend.DefaultStream.Handle,
                () => graph.ExecuteComputeKernelsNoSync(backend),
                new AiDotNet.Tensors.Engines.Compilation.Codegen.CudaGraph.GraphedInferenceStepOptions { ThrowOnUnsupported = false });
            step.Prepare();
            step.Capture();
            Console.WriteLine($"CAPTURE: GraphedInferenceStep HasGraph={step.HasGraph}");
            if (!step.HasGraph) { Console.WriteLine("CAPTURE: graph capture unsupported"); scope.Dispose(); return 0; }

            // Replay once, then validate: r downloads the output buffer that replay last wrote.
            step.Replay();
            double maxAbsDiff = 0;
            int len = Math.Min(eagerCopy.Length, r.Length);
            for (int i = 0; i < len; i++) maxAbsDiff = Math.Max(maxAbsDiff, Math.Abs((double)eagerCopy[i] - r[i]));
            Console.WriteLine($"CAPTURE replay-vs-eager maxAbsDiff={maxAbsDiff:E3} (len={len})  {(maxAbsDiff < 1e-3 ? "CORRECT" : "WRONG")}");

            // Timing: graph replay vs issuing the same kernels directly.
            const int REP = 1000;
            var swG = Stopwatch.StartNew();
            for (int i = 0; i < REP; i++) step.Replay();
            swG.Stop();
            var swD = Stopwatch.StartNew();
            for (int i = 0; i < REP; i++) graph.ExecuteComputeKernelsNoSync(backend);
            backend.Synchronize();
            swD.Stop();
            double g = swG.Elapsed.TotalMilliseconds, d = swD.Elapsed.TotalMilliseconds;
            Console.WriteLine($"CAPTURE {REP} replays={g:F1}ms ({g / REP:F3} ms/replay)  |  {REP} direct={d:F1}ms ({d / REP:F3} ms)  speedup={d / Math.Max(0.01, g):F2}x");
        }
        catch (Exception ex) { Console.WriteLine($"CAPTURE FAILED: {ex.GetType().Name}: {ex.Message}"); }
        finally { scope.Dispose(); }
        return 0;
    }

    private static int ArgI(string[] a, string f, int d)
    {
        int i = Array.IndexOf(a, f);
        return i >= 0 && i + 1 < a.Length && int.TryParse(a[i + 1], out var v) ? v : d;
    }

    private static bool HasFlag(string[] a, string f) => Array.IndexOf(a, f) >= 0;

    private static string ArgS(string[] a, string f, string d)
    {
        int i = Array.IndexOf(a, f);
        return i >= 0 && i + 1 < a.Length ? a[i + 1] : d;
    }

    // Stop a utilization meter (if any) and format its reading.
    //  busyCores = mean busy CPU cores over the window (process CPU-time / wall) — pool-agnostic.
    //  cpuUtil%  = busyCores / maxdop (how much of the requested parallelism was actually used).
    //  coopActive/coopPeak = cooperative-pool-specific fan-out (0 reveals work parallelizing
    //                        through a DIFFERENT pool than CooperativeGemmScheduler).
    private static string StopMeter(ParallelUtilizationMeter meter, int maxdop)
    {
        if (meter is null) return string.Empty;
        meter.Dispose();
        double busy = meter.MeanBusyCores;
        double cpuUtil = maxdop > 0 ? busy / maxdop * 100.0 : 0.0;
        return $" busyCores={busy:F2} cpuUtil={cpuUtil:F0}% coopActive={meter.MeanActiveWorkers:F2} coopPeak={meter.PeakActiveWorkers}";
    }

    private static Tensor<float> Rand(int[] s, Random r)
    {
        var t = new Tensor<float>(s);
        for (int i = 0; i < t.Length; i++) t[i] = (float)(r.NextDouble() - 0.5);
        return t;
    }
}
