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

        if (args.Length > 0 && args[0] == "--resblock") return RunResblock(eng, args);
        if (args.Length > 0 && args[0] == "--gpu") return RunGpu(args);

        int maxdop = ArgI(args, "--maxdop", Environment.ProcessorCount);
        int inC = ArgI(args, "--inc", 256);
        int outC = ArgI(args, "--outc", 256);
        int sp = ArgI(args, "--sp", 16);
        int reps = ArgI(args, "--reps", 12);

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
        for (int i = 0; i < reps; i++)
        {
            var s = Stopwatch.StartNew();
            o = eng.Conv2D(input, kernel, 1, 1, 1);
            s.Stop();
            times[i] = s.Elapsed.TotalMilliseconds;
        }
        Array.Sort(times);
        Console.WriteLine(
            $"CONV inC={inC} outC={outC} sp={sp}x{sp} out.len={o.Length} maxdop={maxdop} " +
            $"procs={Environment.ProcessorCount} warmup_ms={warm:F1} median_ms={times[reps / 2]:F2} min_ms={times[0]:F2} max_ms={times[times.Length - 1]:F2}");
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
        for (int i = 0; i < reps; i++)
        {
            var s = Stopwatch.StartNew();
            y = x0;
            for (int b = 0; b < blocks; b++) y = block(y);
            s.Stop();
            times[i] = s.Elapsed.TotalMilliseconds;
        }
        Array.Sort(times);
        Console.WriteLine(
            $"RESBLOCK C={C} sp={sp}x{sp} blocks={blocks} maxdop={maxdop} procs={Environment.ProcessorCount} " +
            $"warmup_ms={warm:F1} median_ms={times[reps / 2]:F2} min_ms={times[0]:F2} max_ms={times[times.Length - 1]:F2}");
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
            oneIter = () =>
            {
                using var scope = gpu.BeginDeferredScope();
                var r = baseIter();
                var ds = scope as AiDotNet.Tensors.Engines.Gpu.DeferredScope;
                if (ds != null)
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
        var scope = (AiDotNet.Tensors.Engines.Gpu.DeferredScope)gpu.BeginDeferredScope()!;
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
        var cap = new AiDotNet.Tensors.Engines.Gpu.CudaGraphScope(backend, cudaBackend.DefaultStream.Handle);
        if (!cap.IsSupported) { Console.WriteLine("CAPTURE: CudaGraphScope not supported on this device"); cap.Dispose(); scope.Dispose(); return 0; }

        try
        {
            // Capture ONLY the compute kernels (skip H2D/alloc/barrier — they alloc or sync).
            cap.BeginCapture();
            foreach (var node in graph.TopologicalOrder)
                if (node is AiDotNet.Tensors.Engines.Gpu.Graph.KernelNode) node.Execute(backend);
            cap.EndCapture();
            Console.WriteLine($"CAPTURE: cuStreamBeginCapture/EndCapture/Instantiate OK (HasGraph={cap.HasGraph})");

            // Replay once, then validate: r downloads the output buffer that replay last wrote.
            cap.Replay();
            double maxAbsDiff = 0;
            int len = Math.Min(eagerCopy.Length, r.Length);
            for (int i = 0; i < len; i++) maxAbsDiff = Math.Max(maxAbsDiff, Math.Abs((double)eagerCopy[i] - r[i]));
            Console.WriteLine($"CAPTURE replay-vs-eager maxAbsDiff={maxAbsDiff:E3} (len={len})  {(maxAbsDiff < 1e-3 ? "CORRECT" : "WRONG")}");

            // Timing: graph replay vs issuing the same kernels directly.
            const int REP = 1000;
            var swG = Stopwatch.StartNew();
            for (int i = 0; i < REP; i++) cap.Replay();
            swG.Stop();
            var swD = Stopwatch.StartNew();
            for (int i = 0; i < REP; i++)
            {
                foreach (var node in graph.TopologicalOrder)
                    if (node is AiDotNet.Tensors.Engines.Gpu.Graph.KernelNode) node.Execute(backend);
            }
            backend.Synchronize();
            swD.Stop();
            double g = swG.Elapsed.TotalMilliseconds, d = swD.Elapsed.TotalMilliseconds;
            Console.WriteLine($"CAPTURE {REP} replays={g:F1}ms ({g / REP:F3} ms/replay)  |  {REP} direct={d:F1}ms ({d / REP:F3} ms)  speedup={d / Math.Max(0.01, g):F2}x");
        }
        catch (Exception ex) { Console.WriteLine($"CAPTURE FAILED: {ex.GetType().Name}: {ex.Message}"); }
        finally { cap.Dispose(); scope.Dispose(); }
        return 0;
    }

    private static int ArgI(string[] a, string f, int d)
    {
        int i = Array.IndexOf(a, f);
        return i >= 0 && i + 1 < a.Length && int.TryParse(a[i + 1], out var v) ? v : d;
    }

    private static Tensor<float> Rand(int[] s, Random r)
    {
        var t = new Tensor<float>(s);
        for (int i = 0; i < t.Length; i++) t[i] = (float)(r.NextDouble() - 0.5);
        return t;
    }
}
