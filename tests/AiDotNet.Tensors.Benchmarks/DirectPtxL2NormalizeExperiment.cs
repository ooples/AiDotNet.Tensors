using System.Diagnostics;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;
using TorchSharp;
using TorchTensor = TorchSharp.torch.Tensor;

namespace AiDotNet.Tensors.Benchmarks;

/// <summary>
/// Issue-#843 fusion enhancement: NVIDIA-only FP32 row L2-normalize benchmark.
/// Demonstrates the fusion lesson — the direct-PTX kernel folds the
/// sum-of-squares reduction and the broadcast rescale into a single pass, so the
/// per-row norm is never materialized and the input is read exactly once, while
/// the naive path reduces then divides (extra global round-trip). Compares
/// against the current AiDotNet normalize_l2 kernel and the strongest resident
/// PyTorch F.normalize on identical device tensors. CPU MKL/OpenBLAS ineligible.
/// In-process repetitions are diagnostic only; promotion consumes three clean,
/// separately launched captures.
/// </summary>
internal static class DirectPtxL2NormalizeExperiment
{
    private static readonly (int Rows, int Columns)[] Shapes =
        [(256, 128), (2048, 64), (2048, 128), (8192, 128)];
    private const float Epsilon = 1e-12f;

    private readonly record struct Distribution(double Mean, double Median, double P95, double P99);
    private readonly record struct Result(
        int Rows, int Columns, string Method, Distribution Time, double GigabytesPerSecond,
        long Allocation, long TemporaryBytes, float MaxError, int Registers, int LocalBytes);

    internal static void Run(int repetitions = 1)
    {
        if (repetitions <= 0) throw new ArgumentOutOfRangeException(nameof(repetitions));
        for (int repetition = 1; repetition <= repetitions; repetition++)
            RunRepetition(repetition, repetitions);
    }

    private static void RunRepetition(int repetition, int repetitions)
    {
        GpuBenchmarkEnvironment.PrintSnapshot($"start-{repetition}");
        var results = new List<Result>();
        RunDirect(results);
        RunAiDotNet(results);
        RunPyTorch(results);

        Console.WriteLine($"NVIDIA GPU-only fused FP32 row L2-normalize (resident tensors), diagnostic repetition {repetition}/{repetitions}");
        Console.WriteLine($"{"Rows",7} {"Cols",6} {"Method",-27} {"median us",10} {"p95 us",10} {"mean us",10} {"GB/s",9} {"B/call",9} {"max rel",10} {"regs",6} {"local B",8}");
        Console.WriteLine(new string('-', 128));
        foreach (Result r in results.OrderBy(r => r.Rows).ThenBy(r => r.Columns).ThenBy(r => r.Time.Median))
        {
            string registers = r.Registers < 0 ? "n/a" : r.Registers.ToString();
            string local = r.LocalBytes < 0 ? "n/a" : r.LocalBytes.ToString();
            Console.WriteLine(
                $"{r.Rows,7} {r.Columns,6} {r.Method,-27} {r.Time.Median * 1000,10:F2} {r.Time.P95 * 1000,10:F2} " +
                $"{r.Time.Mean * 1000,10:F2} {r.GigabytesPerSecond,9:F2} {r.Allocation,9} {r.MaxError,10:G4} {registers,6} {local,8}");
        }

        Console.WriteLine();
        Console.WriteLine("Diagnostic gate: each in-process repetition counts as one diagnostic only; production still requires three clean, separately launched captures.");
        DirectPtxReleaseGatePolicy policy = DirectPtxReleaseGatePolicy.ProductionDefault with { RequiredIndependentRuns = 1 };
        foreach ((int rows, int columns) in Shapes)
        {
            var direct = results.SingleOrDefault(r => r.Rows == rows && r.Columns == columns && r.Method == "Direct PTX L2-normalize");
            if (direct.Method is null) continue;
            var competitors = results.Where(r => r.Rows == rows && r.Columns == columns && r.Method != "Direct PTX L2-normalize").ToList();
            if (competitors.Count == 0) continue;
            Result best = competitors.OrderBy(r => r.Time.Median).First();
            var directEvidence = new DirectPtxPerformanceEvidence(
                direct.Time.Median * 1000, direct.Time.P95 * 1000,
                direct.Allocation, direct.TemporaryBytes, direct.MaxError, direct.LocalBytes, IndependentRuns: 1);
            var competitorEvidence = new DirectPtxPerformanceEvidence(
                best.Time.Median * 1000, best.Time.P95 * 1000,
                best.Allocation, best.TemporaryBytes, best.MaxError, best.LocalBytes, IndependentRuns: 1);
            DirectPtxReleaseDecision decision = policy.Evaluate(directEvidence, competitorEvidence);
            Console.WriteLine(
                $"[{rows},{columns}]: {(decision.Passed ? "PASS" : "HOLD"),-4} {decision.MedianSpeedup:F2}x vs {best.Method}; " +
                (decision.Passed ? "all gates passed" : string.Join("; ", decision.Failures)));
        }
        GpuBenchmarkEnvironment.PrintSnapshot($"end-{repetition}");
    }

    private static void RunDirect(List<Result> results)
    {
        using var runtime = new DirectPtxRuntime();
        if (!DirectPtxArchitecture.HasValidatedRowReduction(
                runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor)) return;
        using (runtime.Enter())
        foreach ((int rows, int columns) in Shapes)
        {
            using var kernel = new PtxFusedRowL2NormalizeF32Kernel(runtime, rows, columns);
            float[] input = Values(rows * columns, 100 + rows + columns);
            using var inputBuffer = runtime.AllocateBytes(kernel.Blueprint.Tensors[0].RequiredBytes);
            using var outputBuffer = runtime.AllocateBytes(kernel.Blueprint.Tensors[1].RequiredBytes);
            inputBuffer.Upload<float>(input);
            Action launch = () => kernel.Launch(
                DirectPtxTensorView.CreateOwned(inputBuffer, kernel.Blueprint.Tensors[0]),
                DirectPtxTensorView.CreateOwned(outputBuffer, kernel.Blueprint.Tensors[1]));
            Distribution distribution = Measure(runtime.Synchronize, launch);
            long allocation = Allocation(runtime.Synchronize, launch);
            launch(); runtime.Synchronize();
            var actual = new float[rows * columns];
            outputBuffer.Download<float>(actual);
            float error = Validate(actual, input, rows, columns);
            results.Add(new Result(rows, columns, "Direct PTX L2-normalize", distribution,
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
            float[] input = Values(rows * columns, 100 + rows + columns);
            using var inputBuffer = backend.AllocateBuffer(input);
            using var outputBuffer = backend.AllocateBuffer(rows * columns);
            Action launch = () => backend.NormalizeL2(inputBuffer, outputBuffer, rows, columns);
            Distribution distribution = Measure(backend.Synchronize, launch);
            long allocation = Allocation(backend.Synchronize, launch);
            results.Add(new Result(rows, columns, "AiDotNet normalize_l2", distribution,
                Bandwidth(rows, columns, distribution.Median), allocation, 0, 0f, -1, -1));
        }
    }

    private static void RunPyTorch(List<Result> results)
    {
        if (!torch.cuda.is_available()) return;
        foreach ((int rows, int columns) in Shapes)
        {
            float[] input = Values(rows * columns, 100 + rows + columns);
            using TorchTensor x = torch.tensor(input, [rows, columns], device: torch.CUDA);
            void Launch()
            {
                using TorchTensor output = torch.nn.functional.normalize(x, p: 2, dim: 1, eps: Epsilon);
            }
            Distribution distribution = Measure(() => torch.cuda.synchronize(), Launch);
            long allocation = Allocation(() => torch.cuda.synchronize(), Launch);
            results.Add(new Result(rows, columns, "PyTorch F.normalize", distribution,
                Bandwidth(rows, columns, distribution.Median), allocation, -1, 0f, -1, -1));
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
        long bytes = checked((long)rows * columns * sizeof(float) * 2L);
        return bytes / (milliseconds * 1e-3) / 1e9;
    }

    private static float[] Values(int length, int seed)
    {
        var random = new Random(seed);
        return Enumerable.Range(0, length).Select(_ => (random.NextSingle() * 2f - 1f) * 4f).ToArray();
    }

    private static float Validate(float[] actual, float[] input, int rows, int columns)
    {
        float maximum = 0;
        for (int row = 0; row < rows; row++)
        {
            double sumSquares = 0;
            for (int col = 0; col < columns; col++)
            {
                double v = input[row * columns + col];
                sumSquares += v * v;
            }
            double invNorm = 1.0 / Math.Sqrt(sumSquares + Epsilon);
            for (int col = 0; col < columns; col++)
            {
                float expected = (float)(input[row * columns + col] * invNorm);
                float denominator = MathF.Abs(expected) + 1e-3f;
                maximum = MathF.Max(maximum, MathF.Abs(actual[row * columns + col] - expected) / denominator);
            }
        }
        return maximum;
    }
}
