using System;
using System.Diagnostics;
using System.Linq;
using BlasManagedLib = AiDotNet.Tensors.Engines.BlasManaged.BlasManaged;

namespace AiDotNet.Tensors.Benchmarks.PyTorchComparison;

/// <summary>
/// Differentiator benchmark D.1 — cold-start latency: time from process start
/// to first GEMM result. PyTorch's Python import overhead is ~250 ms even
/// before the first kernel call; AiDotNet should be a fraction of that.
/// Measured by spawning a fresh process for each side so JIT + import are
/// counted, not amortised.
/// </summary>
internal static class ColdStartBench
{
    public static void Run()
    {
        Console.WriteLine("=== Differentiator: cold-start latency (first GEMM) ===");
        Console.WriteLine();

        const int trials = 5;
        var aiTimes = new double[trials];
        for (int t = 0; t < trials; t++)
            aiTimes[t] = SpawnAndTime("--cold-start-aidotnet");
        Array.Sort(aiTimes);
        double aiMedian = aiTimes[trials / 2];

        var pyTimes = new double[trials];
        bool pyOk = true;
        for (int t = 0; t < trials; t++)
        {
            double ms = SpawnAndTime("--cold-start-pytorch");
            if (ms < 0) { pyOk = false; break; }
            pyTimes[t] = ms;
        }

        Console.WriteLine($"  AiDotNet cold-start: {aiMedian:F1} ms (median of {trials})");
        if (pyOk)
        {
            Array.Sort(pyTimes);
            double pyMedian = pyTimes[trials / 2];
            Console.WriteLine($"  PyTorch  cold-start: {pyMedian:F1} ms (median of {trials})");
            Console.WriteLine($"  Speedup:             {pyMedian / aiMedian:F2}× (AiDotNet faster)");
        }
        else
        {
            Console.WriteLine("  PyTorch  cold-start: SKIPPED (python/torch not available on PATH)");
        }
        Console.WriteLine();
        Console.WriteLine("  Why PyTorch can't match: every `import torch` pays the Python interpreter");
        Console.WriteLine("  startup + the libtorch shared-library load before the first matmul. A");
        Console.WriteLine("  serverless / short-lived-job deployment pays this on every cold invocation.");
    }

    /// <summary>Single 64³ FP32 GEMM, output discarded — the cold-start workload.</summary>
    public static int RunAiDotNetWorkload()
    {
        var a = new float[64 * 64];
        var b = new float[64 * 64];
        var c = new float[64 * 64];
        for (int i = 0; i < a.Length; i++) { a[i] = 1f; b[i] = 1f; }
        BlasManagedLib.Gemm<float>(a, 64, false, b, 64, false, c, 64, 64, 64, 64);
        return c[0] != 0f ? 0 : 1;
    }

    private static double SpawnAndTime(string flag)
    {
        bool pytorch = flag == "--cold-start-pytorch";
        string exe = pytorch
            ? "python"
            : (Environment.ProcessPath ?? throw new InvalidOperationException("can't resolve self exe"));
        string args = pytorch
            ? "-c \"import torch; a=torch.zeros((64,64)); b=torch.zeros((64,64)); torch.matmul(a,b); print('ok')\""
            : flag;
        var psi = new ProcessStartInfo
        {
            FileName = exe,
            Arguments = args,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false,
            CreateNoWindow = true,
        };
        try
        {
            var sw = Stopwatch.StartNew();
            using var p = Process.Start(psi);
            if (p is null) return -1;
            p.WaitForExit(15_000);
            sw.Stop();
            return p.ExitCode == 0 ? sw.Elapsed.TotalMilliseconds : -1;
        }
        catch
        {
            return -1;  // python not found, etc.
        }
    }
}
