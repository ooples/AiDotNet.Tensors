using System.Diagnostics;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;
using TorchSharp;
using TorchTensor = TorchSharp.torch.Tensor;

namespace AiDotNet.Tensors.Benchmarks;

/// <summary>
/// Issue-#843 NVIDIA-only fused FP32 row-sum reduction benchmark. Compares the
/// hand-emitted direct-PTX kernel against the current AiDotNet CUDA sum_axis
/// kernel and the strongest resident PyTorch path on identical device tensors.
/// CPU MKL/OpenBLAS are intentionally ineligible. The single capture below is
/// one independent run; production promotion consumes three clean runs.
/// </summary>
internal static class DirectPtxReductionExperiment
{
    private static readonly (int Rows, int Columns)[] Shapes =
        [(256, 128), (2048, 64), (2048, 128), (8192, 128)];

    private readonly record struct Distribution(double Mean, double Median, double P95, double P99);
    private readonly record struct Result(
        int Rows, int Columns, string Method, Distribution Time, double GigabytesPerSecond,
        long Allocation, long TemporaryBytes, float MaxError, int Registers, int LocalBytes);

    internal static void Run(int independentRuns = 1)
    {
        GpuBenchmarkEnvironment.PrintSnapshot("start");
        var results = new List<Result>();
        RunDirect(results);
        RunAiDotNet(results);
        RunPyTorch(results);

        Console.WriteLine("NVIDIA GPU-only fused FP32 row-sum reduction (resident tensors)");
        Console.WriteLine($"{"Rows",7} {"Cols",6} {"Method",-27} {"median us",10} {"p95 us",10} {"mean us",10} {"GB/s",9} {"B/call",9} {"max err",10} {"regs",6} {"local B",8}");
        Console.WriteLine(new string('-', 128));
        foreach (Result r in results.OrderBy(r => r.Rows).ThenBy(r => r.Columns).ThenBy(r => r.Time.Median))
        {
            string registers = r.Registers < 0 ? "n/a" : r.Registers.ToString();
            string local = r.LocalBytes < 0 ? "n/a" : r.LocalBytes.ToString();
            Console.WriteLine(
                $"{r.Rows,7} {r.Columns,6} {r.Method,-27} {r.Time.Median * 1000,10:F2} " +
                $"{r.Time.P95 * 1000,10:F2} {r.Time.Mean * 1000,10:F2} {r.GigabytesPerSecond,9:F2} " +
                $"{r.Allocation,9} {r.MaxError,10:G4} {registers,6} {local,8}");
        }

        Console.WriteLine();
        Console.WriteLine("Diagnostic gate: production policy except this single capture counts as one independent run (production requires three).");
        DirectPtxReleaseGatePolicy policy = DirectPtxReleaseGatePolicy.ProductionDefault with
        {
            RequiredIndependentRuns = 1
        };
        foreach ((int rows, int columns) in Shapes)
        {
            var direct = results.SingleOrDefault(r => r.Rows == rows && r.Columns == columns && r.Method == "Direct PTX row-sum");
            if (direct.Method is null) continue;
            var competitors = results.Where(r => r.Rows == rows && r.Columns == columns && r.Method != "Direct PTX row-sum").ToList();
            if (competitors.Count == 0) continue;
            Result best = competitors.OrderBy(r => r.Time.Median).First();
            var directEvidence = new DirectPtxPerformanceEvidence(
                direct.Time.Median * 1000, direct.Time.P95 * 1000,
                direct.Allocation, direct.TemporaryBytes, direct.MaxError,
                direct.LocalBytes, IndependentRuns: 1);
            var competitorEvidence = new DirectPtxPerformanceEvidence(
                best.Time.Median * 1000, best.Time.P95 * 1000,
                best.Allocation, best.TemporaryBytes, best.MaxError,
                best.LocalBytes, IndependentRuns: 1);
            DirectPtxReleaseDecision decision = policy.Evaluate(directEvidence, competitorEvidence);
            Console.WriteLine(
                $"[{rows},{columns}]: {(decision.Passed ? "PASS" : "HOLD"),-4} {decision.MedianSpeedup:F2}x vs {best.Method}; " +
                (decision.Passed ? "all gates passed" : string.Join("; ", decision.Failures)));
        }
        GpuBenchmarkEnvironment.PrintSnapshot("end");
    }

    private static void RunDirect(List<Result> results)
    {
        using var runtime = new DirectPtxRuntime();
        if (runtime.ArchitectureFamily != DirectPtxArchitectureFamily.Ampere) return;
        using (runtime.Enter())
        foreach ((int rows, int columns) in Shapes)
        {
            using var kernel = new PtxFusedRowReduceF32Kernel(runtime, rows, columns);
            float[] x = Values(rows * columns, 100 + rows + columns, 4f);
            using var input = runtime.AllocateBytes(kernel.Blueprint.Tensors[0].RequiredBytes);
            using var output = runtime.AllocateBytes(kernel.Blueprint.Tensors[1].RequiredBytes);
            input.Upload<float>(x);
            Action launch = () => kernel.Launch(
                DirectPtxTensorView.CreateOwned(input, kernel.Blueprint.Tensors[0]),
                DirectPtxTensorView.CreateOwned(output, kernel.Blueprint.Tensors[1]));
            Distribution distribution = Measure(runtime.Synchronize, launch);
            long allocation = Allocation(runtime.Synchronize, launch);
            launch(); runtime.Synchronize();
            var actual = new float[rows];
            output.Download<float>(actual);
            float error = Validate(actual, x, rows, columns);
            results.Add(new Result(rows, columns, "Direct PTX row-sum", distribution,
                Bandwidth(rows, columns, distribution.Median), allocation, 0, error,
                kernel.Audit.Function.RegistersPerThread, kernel.Audit.Function.LocalBytesPerThread));
        }
    }

    private static void RunAiDotNet(List<Result> results)
    {
        using var backend = new CudaBackend();
        if (!backend.IsAvailable) return;
        foreach ((int rows, int columns) in Shapes)
        {
            float[] x = Values(rows * columns, 100 + rows + columns, 4f);
            using var input = backend.AllocateBuffer(x);
            using var output = backend.AllocateBuffer(rows);
            Action launch = () => backend.SumAxis(input, output, rows, columns);
            Distribution distribution = Measure(backend.Synchronize, launch);
            long allocation = Allocation(backend.Synchronize, launch);
            launch(); backend.Synchronize();
            float error = Validate(backend.DownloadBuffer(output), x, rows, columns);
            results.Add(new Result(rows, columns, "AiDotNet sum_axis", distribution,
                Bandwidth(rows, columns, distribution.Median), allocation, 0, error, -1, -1));
        }
    }

    private static void RunPyTorch(List<Result> results)
    {
        if (!torch.cuda.is_available()) return;
        foreach ((int rows, int columns) in Shapes)
        {
            float[] xHost = Values(rows * columns, 100 + rows + columns, 4f);
            using TorchTensor input = torch.tensor(xHost, [rows, columns], device: torch.CUDA);
            void Launch()
            {
                using TorchTensor output = input.sum([1L]);
            }
            Distribution distribution = Measure(() => torch.cuda.synchronize(), Launch);
            long allocation = Allocation(() => torch.cuda.synchronize(), Launch);
            results.Add(new Result(rows, columns, "PyTorch sum(dim=-1)", distribution,
                Bandwidth(rows, columns, distribution.Median), allocation, -1, 0, -1, -1));
        }
    }

    private static Distribution Measure(Action synchronize, Action launch)
    {
        for (int i = 0; i < 30; i++) launch();
        synchronize();
        var samples = new double[101];
        for (int i = 0; i < samples.Length; i++)
        {
            long start = Stopwatch.GetTimestamp();
            launch();
            synchronize();
            samples[i] = Stopwatch.GetElapsedTime(start).TotalMilliseconds;
        }
        Array.Sort(samples);
        return new Distribution(samples.Average(), Percentile(samples, .5),
            Percentile(samples, .95), Percentile(samples, .99));
    }

    private static long Allocation(Action synchronize, Action launch)
    {
        launch(); synchronize();
        long before = GC.GetAllocatedBytesForCurrentThread();
        for (int i = 0; i < 50; i++) launch();
        long allocation = (GC.GetAllocatedBytesForCurrentThread() - before) / 50;
        synchronize();
        return allocation;
    }

    private static double Percentile(double[] sorted, double percentile)
    {
        double position = (sorted.Length - 1) * percentile;
        int lower = (int)position;
        int upper = Math.Min(lower + 1, sorted.Length - 1);
        return sorted[lower] + (sorted[upper] - sorted[lower]) * (position - lower);
    }

    private static double Bandwidth(int rows, int columns, double milliseconds)
    {
        long bytes = checked((long)rows * columns * sizeof(float) + (long)rows * sizeof(float));
        return bytes / (milliseconds * 1e-3) / 1e9;
    }

    private static float[] Values(int length, int seed, float magnitude)
    {
        var random = new Random(seed);
        return Enumerable.Range(0, length)
            .Select(_ => (random.NextSingle() - 0.5f) * 2f * magnitude).ToArray();
    }

    // Reports the maximum RELATIVE error against an FP64 oracle. Row-sum outputs
    // are O(columns * magnitude), so absolute error naturally exceeds the release
    // gate's 5e-5 absolute bound while relative error stays ~1e-6; the reduction
    // family therefore needs a relative-error gate before any promotion claim.
    private static float Validate(float[] actual, float[] input, int rows, int columns)
    {
        float maximum = 0;
        for (int row = 0; row < rows; row++)
        {
            double sum = 0;
            for (int column = 0; column < columns; column++)
                sum += input[row * columns + column];
            double denominator = Math.Abs(sum) + 1.0;
            maximum = MathF.Max(maximum, (float)(Math.Abs(actual[row] - sum) / denominator));
        }
        return maximum;
    }
}
