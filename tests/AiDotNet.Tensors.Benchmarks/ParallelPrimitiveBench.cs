using System.Diagnostics;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Tensors.Benchmarks;

/// <summary>
/// Head-to-head of the CPU parallel-dispatch primitives so we can pick the
/// superior one to replace bare <c>System.Threading.Tasks.Parallel.For</c> at
/// layer call sites. Measures wall-clock per dispatch across the full regime
/// span — from dispatch-overhead-dominated (tiny per-chunk work, where the
/// .NET ThreadPool's LowLevelLifoSemaphore wait dominates) to compute-bound
/// (large per-chunk work, where raw scheduling barely matters).
///
/// Contenders:
///   seq        — sequential for-loop (baseline / floor for tiny work)
///   Parallel   — System.Threading.Tasks.Parallel.For (the thing we'd replace)
///   Lightweight— CpuParallelSettings.LightweightParallel (persistent workers)
///   LwGrain    — LightweightParallel with a totalWork grain gate (serial-fallback)
///   PfOrSerial — CpuParallelSettings.ParallelForOrSerial (grain gate -> Parallel.For)
///
/// Every contender performs the SAME total floating-point work per trial; only
/// the partitioning/dispatch differs. Reports median + mean + p95 (ns) and the
/// speed-up vs Parallel.For so the win/loss is unambiguous per regime.
/// </summary>
public static class ParallelPrimitiveBench
{
    // Per-chunk compute cost (inner FLOP iterations). Spans dispatch-overhead-
    // dominated (16) to firmly compute-bound (1_048_576).
    private static readonly int[] WorkPerChunk = { 16, 256, 4_096, 65_536, 1_048_576 };
    // Chunk counts: 4 (under-subscribed), and around the core count.
    private static readonly int[] ChunkCounts = { 4, 16, 64 };

    private const int WarmupRuns = 50;
    private const int TimedRuns = 400;

    private static double[] _output = System.Array.Empty<double>();
    // Non-volatile (C# forbids volatile double); a static-field write the JIT
    // cannot prove dead is enough to keep the kernel from being elided.
    private static double _sink;

    public static void Run()
    {
        int cores = System.Environment.ProcessorCount;
        Console.WriteLine("=== Parallel primitive head-to-head ===");
        Console.WriteLine($"Cores={cores}  MaxDoP={CpuParallelSettings.MaxDegreeOfParallelism}  " +
                          $"warmup={WarmupRuns} timed={TimedRuns}");
        Console.WriteLine("Same total FLOPs per trial; only dispatch differs. Lower ns = better.");
        Console.WriteLine();

        // Warm the persistent worker pool + the ThreadPool so first-call spawn
        // cost is not charged to any single contender.
        _output = new double[8192];
        for (int i = 0; i < 200; i++)
        {
            CpuParallelSettings.LightweightParallel(64, c => _output[c] += 1);
            System.Threading.Tasks.Parallel.For(0, 64, c => _output[c] += 1);
        }

        foreach (int chunks in ChunkCounts)
        {
            foreach (int work in WorkPerChunk)
            {
                _output = new double[chunks];
                long totalWork = (long)chunks * work;

                Sample seq = Time(() => RunSeq(chunks, work));
                Sample par = Time(() => RunParallelFor(chunks, work));
                Sample lw = Time(() => RunLightweight(chunks, work));
                Sample lwg = Time(() => RunLightweightGrain(chunks, work, totalWork));
                Sample pfs = Time(() => RunParallelForOrSerial(chunks, work, totalWork));

                Console.WriteLine($"chunks={chunks,-3} work/chunk={work,-9} (totalWork={totalWork,-12})");
                Print("  seq       ", seq, par.Median);
                Print("  Parallel  ", par, par.Median);
                Print("  Lightweight", lw, par.Median);
                Print("  LwGrain   ", lwg, par.Median);
                Print("  PfOrSerial", pfs, par.Median);
                Console.WriteLine($"  WINNER: {Winner(("seq", seq.Median), ("Parallel", par.Median), ("Lightweight", lw.Median), ("LwGrain", lwg.Median), ("PfOrSerial", pfs.Median))}");
                Console.WriteLine();
            }
        }
    }

    // --- contenders (identical work, different dispatch) ---

    private static void RunSeq(int chunks, int work)
    {
        for (int c = 0; c < chunks; c++) _output[c] = ChunkKernel(c, work);
    }

    private static void RunParallelFor(int chunks, int work)
    {
        System.Threading.Tasks.Parallel.For(0, chunks, c => _output[c] = ChunkKernel(c, work));
    }

    private static void RunLightweight(int chunks, int work)
    {
        CpuParallelSettings.LightweightParallel(chunks, c => _output[c] = ChunkKernel(c, work));
    }

    private static void RunLightweightGrain(int chunks, int work, long totalWork)
    {
        CpuParallelSettings.LightweightParallel(chunks, totalWork, c => _output[c] = ChunkKernel(c, work));
    }

    private static void RunParallelForOrSerial(int chunks, int work, long totalWork)
    {
        CpuParallelSettings.ParallelForOrSerial(0, chunks, totalWork, c => _output[c] = ChunkKernel(c, work), deterministicSafe: true);
    }

    // Deterministic per-chunk FLOP kernel that cannot be optimized away.
    private static double ChunkKernel(int chunk, int work)
    {
        double acc = chunk * 0.5 + 1.0;
        for (int i = 0; i < work; i++)
            acc = acc * 1.0000001 + 0.5 - (acc * 0.25);
        return acc;
    }

    // --- timing: warmup, then TimedRuns Stopwatch samples, report median/mean/p95 ns ---

    private readonly struct Sample
    {
        public readonly double Median, Mean, P95;
        public Sample(double median, double mean, double p95) { Median = median; Mean = mean; P95 = p95; }
    }

    private static Sample Time(System.Action trial)
    {
        for (int i = 0; i < WarmupRuns; i++) trial();

        var samples = new double[TimedRuns];
        var sw = new Stopwatch();
        for (int i = 0; i < TimedRuns; i++)
        {
            sw.Restart();
            trial();
            sw.Stop();
            samples[i] = sw.Elapsed.TotalMilliseconds * 1_000_000.0; // ns
            _sink = _output[0];
        }
        System.Array.Sort(samples);
        double sum = 0; foreach (var s in samples) sum += s;
        return new Sample(samples[TimedRuns / 2], sum / TimedRuns, samples[(int)(TimedRuns * 0.95)]);
    }

    private static void Print(string name, Sample s, double baselineMedianNs)
    {
        double ratio = baselineMedianNs / s.Median; // >1 => this is faster than Parallel.For
        string speed = ratio >= 1 ? $"{ratio:F2}x faster" : $"{1 / ratio:F2}x slower";
        Console.WriteLine($"{name}: median={s.Median,12:N0}ns mean={s.Mean,12:N0}ns p95={s.P95,12:N0}ns  ({speed} vs Parallel)");
    }

    private static string Winner(params (string name, double ns)[] rows)
    {
        var best = rows[0];
        foreach (var r in rows) if (r.ns < best.ns) best = r;
        return $"{best.name} ({best.ns:N0}ns)";
    }

    // #475 idle-CPU check: simulate a training loop of medium parallel ops separated
    // by serial inter-op work (LayerNorm/residual/tape between GEMMs), measuring both
    // wall-clock AND process CPU-seconds so we can see whether the adaptive warm-spin
    // re-introduces the "12 idle cores burning" #475 reverted. Run with the default
    // warm window and again with AIDOTNET_PPE_WARMWINDOW_US=0 (park immediately).
    public static void CpuBurn()
    {
        int cores = System.Environment.ProcessorCount;
        var proc = System.Diagnostics.Process.GetCurrentProcess();
        _output = new double[64];
        int chunks = 16, work = 4096;          // totalWork 65536 — the medium/floor regime
        long totalWork = (long)chunks * work;
        int gapWork = 6000;                     // serial inter-op work between dispatches

        string win = System.Environment.GetEnvironmentVariable("AIDOTNET_PPE_WARMWINDOW_US") ?? "(default 200)";
        for (int i = 0; i < 800; i++)           // warm up JIT + pool
            CpuParallelSettings.LightweightParallel(chunks, totalWork, c => _output[c & 63] = ChunkKernel(c, work));

        int iters = 12000;
        var cpu0 = proc.TotalProcessorTime;
        var sw = Stopwatch.StartNew();
        for (int i = 0; i < iters; i++)
        {
            CpuParallelSettings.LightweightParallel(chunks, totalWork, c => _output[c & 63] = ChunkKernel(c, work));
            _sink = ChunkKernel(i, gapWork);    // serial inter-op gap (workers idle here)
        }
        sw.Stop();
        double cpuSec = (proc.TotalProcessorTime - cpu0).TotalSeconds;
        double wallSec = sw.Elapsed.TotalSeconds;
        Console.WriteLine($"=== CPU-burn sim (WARMWINDOW_US={win}) cores={cores} ===");
        Console.WriteLine($"iters={iters} chunks={chunks} work/chunk={work} gapWork={gapWork}");
        Console.WriteLine($"  wall={wallSec * 1000:F1}ms  perDispatch={wallSec / iters * 1e6:F2}us  " +
                          $"CPU={cpuSec:F2}s  avgCoresBusy={cpuSec / wallSec:F1} / {cores}");
    }
}
