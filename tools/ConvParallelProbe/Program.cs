using System;
using System.Diagnostics;
using AiDotNet.Tensors.Engines;
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

        // #653: force AutoTracer.ShouldRecord=false (the training / inference-without-compile hot
        // path) so the end-to-end probes measure the recording-off fast path.
        if (Environment.GetEnvironmentVariable("AIDOTNET_NO_AUTOTRACE") == "1")
        {
            // AutoTracer.ShouldRecord = Enabled && !GraphMode && EnableCompilation && !ThreadTapeActive.
            // EnableCompilation is the PUBLIC, documented disable knob (AutoTracer.cs). Current returns
            // a FRESH Default copy when the thread has no installed options, so mutating it in place is
            // lost — take the copy, disable compilation, and install it via SetCurrent so it persists.
            var codecOpts = AiDotNet.Tensors.Engines.Optimization.TensorCodecOptions.Current;
            codecOpts.EnableCompilation = false;
            AiDotNet.Tensors.Engines.Optimization.TensorCodecOptions.SetCurrent(codecOpts);
            Console.Error.WriteLine("[probe] TensorCodecOptions.EnableCompilation=false (AutoTracer recording off)");
        }

        if (args.Length > 0 && args[0] == "--allocbench") return RunAllocBench(eng, args);
        if (args.Length > 0 && args[0] == "--resblock") return RunResblock(eng, args);
        if (args.Length > 0 && args[0] == "--attnblock") return RunAttnBlock(eng, args);
        if (args.Length > 0 && args[0] == "--act") return RunAct(eng, args);
        if (args.Length > 0 && args[0] == "--gemm") return RunGemm(eng, args);
        if (args.Length > 0 && args[0] == "--gemmprofile") return RunGemmProfile(eng, args);
        if (args.Length > 0 && args[0] == "--gpu") return RunGpu(args);

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

        // #653/#657 follow-up: forward caching-allocator. With --arena, ONE TensorArena lives
        // across the whole rep loop and Reset()s before each full forward — so every per-op
        // TensorAllocator.Rent inside blk() hits the arena (Tier 0) instead of allocating a
        // fresh GC array, and the prior forward's intermediates are recycled. Reset is PER
        // FORWARD (not per block): a block's output is the next block's input, so resetting
        // mid-forward would recycle a live tensor. x0 is allocated before the arena, so it
        // survives Reset and is a safe re-entry point each iteration.
        // #653/#657: --arena runs the forward inside a TensorArena (PyTorch-caching-allocator
        // equivalent); Reset() per rep recycles the prior iteration's intermediate arrays.
        // (Resolved against main's AIDOTNET_ARENA env-var variant — kept the more careful flag
        // version: it warms the arena before measuring and uses the per-thread alloc counter,
        // which the single-threaded alloc window needs; folded in main's arena={...} reporting.)
        bool useArena = HasFlag(a, "--arena");
        TensorArena? arena = useArena ? TensorArena.Create() : null;
        var times = new double[reps];
        using var meter = util ? ParallelUtilizationMeter.Start() : null;   // `using` disposes on exception paths too (#654 review)
        // Warm the arena (first rep allocates; subsequent reps reuse) before the alloc window.
        if (arena is not null) { arena.Reset(); yy = x0; for (int b = 0; b < blocks; b++) yy = blk(yy); }
        GC.Collect(); GC.WaitForPendingFinalizers(); GC.Collect();
        long allocStart = GC.GetAllocatedBytesForCurrentThread();
        for (int i = 0; i < reps; i++)
        {
            if (arena is not null) arena.Reset();
            var s = Stopwatch.StartNew();
            yy = x0;
            for (int b = 0; b < blocks; b++) yy = blk(yy);
            s.Stop();
            times[i] = s.Elapsed.TotalMilliseconds;
        }
        arena?.Dispose();
        long allocBytes = GC.GetAllocatedBytesForCurrentThread() - allocStart;
        string u = StopMeter(meter, maxdop);
        Array.Sort(times);
        Console.WriteLine(
            $"ATTNBLOCK S={S} D={D} H={H} Dh={Dh} blocks={blocks} maxdop={maxdop} procs={Environment.ProcessorCount} " +
            $"arena={useArena} warmup_ms={warm:F1} median_ms={times[reps / 2]:F2} min_ms={times[0]:F2} max_ms={times[times.Length - 1]:F2} " +
            $"alloc_MB_per_fwd={allocBytes / 1048576.0 / reps:F3}{u}");
        return 0;
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

    // #653: measure per-op allocated bytes + time when AutoTracer.ShouldRecord is FALSE (the
    // training / no-compilation hot path) — isolates the autotrace closure allocate-then-discard
    // overhead. ShouldRecord is forced false via the public TensorCodecOptions knob. Tiny shapes
    // so the per-op closure alloc is visible against the (unavoidable) result-tensor alloc.
    private static int RunAllocBench(CpuEngine eng, string[] a)
    {
        int M = ArgI(a, "--m", 8), K = ArgI(a, "--k", 8), N = ArgI(a, "--n", 8);
        int reps = ArgI(a, "--reps", 200000);
        if (M < 1 || K < 1 || N < 1 || reps < 1)
        {
            Console.Error.WriteLine("allocbench: --m, --k, --n, --reps must all be >= 1.");
            return 2;
        }
        // ShouldRecord = Enabled && !GraphMode && EnableCompilation && !ThreadTapeActive.
        // Disabling compilation via the PUBLIC, documented knob (AutoTracer.cs) forces
        // ShouldRecord=false (the training / no-compilation hot path this benchmarks) — no
        // reflection. Current returns a FRESH Default copy when the thread has no installed
        // options, so mutate the copy and install it via SetCurrent (a bare property set on
        // Current would be discarded). One-time set before the timed loop, so it never touches
        // the per-op measurement.
        var codecOpts = AiDotNet.Tensors.Engines.Optimization.TensorCodecOptions.Current;
        codecOpts.EnableCompilation = false;
        AiDotNet.Tensors.Engines.Optimization.TensorCodecOptions.SetCurrent(codecOpts);
        Console.Error.WriteLine("[probe] TensorCodecOptions.EnableCompilation=false (AutoTracer.ShouldRecord=false)");

        var rng = new Random(0);
        var lhs = Rand(new[] { M, K }, rng);
        var rhs = Rand(new[] { K, N }, rng);
        var warm = eng.BatchMatMul(lhs, rhs); float sink = warm[0];

        GC.Collect(); GC.WaitForPendingFinalizers(); GC.Collect();
        long startBytes = GC.GetAllocatedBytesForCurrentThread();
        var sw = Stopwatch.StartNew();
        for (int i = 0; i < reps; i++) { var o = eng.BatchMatMul(lhs, rhs); sink += o[0]; }
        sw.Stop();
        long bytes = GC.GetAllocatedBytesForCurrentThread() - startBytes;
        Console.WriteLine(
            $"ALLOCBENCH op=BatchMatMul {M}x{K}x{N} reps={reps} " +
            $"bytes_per_op={(double)bytes / reps:F1} ns_per_op={sw.Elapsed.TotalMilliseconds * 1e6 / reps:F1} " +
            $"total_MB={bytes / 1048576.0:F1} (sink={sink:E1})");
        return 0;
    }

    private static int ArgI(string[] a, string f, int d)
    {
        int i = Array.IndexOf(a, f);
        return i >= 0 && i + 1 < a.Length && int.TryParse(a[i + 1], out var v) ? v : d;
    }

    private static bool HasFlag(string[] a, string f) => Array.IndexOf(a, f) >= 0;

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
