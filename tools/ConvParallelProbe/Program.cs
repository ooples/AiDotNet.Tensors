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

        int maxdop = ArgI(args, "--maxdop", Environment.ProcessorCount);
        int inC = ArgI(args, "--inc", 256);
        int outC = ArgI(args, "--outc", 256);
        int sp = ArgI(args, "--sp", 16);
        int reps = ArgI(args, "--reps", 12);

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
            $"procs={Environment.ProcessorCount} warmup_ms={warm:F1} median_ms={times[reps / 2]:F2} min_ms={times[0]:F2}");
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
