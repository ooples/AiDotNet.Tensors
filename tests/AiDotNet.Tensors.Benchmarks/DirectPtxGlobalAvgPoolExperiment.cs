using System.Diagnostics;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;
using TorchSharp;
using TorchTensor = TorchSharp.torch.Tensor;

namespace AiDotNet.Tensors.Benchmarks;

/// <summary>
/// Issue-#842 NVIDIA-only FP32 global-average-pool benchmark. Compares the
/// hand-emitted direct-PTX kernel against the current AiDotNet CUDA
/// global_avgpool2d kernel and the strongest resident PyTorch mean(dim) path on
/// identical device tensors. CPU MKL/OpenBLAS are intentionally ineligible.
/// In-process repetitions are diagnostic only; promotion consumes three
/// separately launched, clean captures.
/// </summary>
internal static class DirectPtxGlobalAvgPoolExperiment
{
    // (batch, channels, spatial=H*W). rows = batch*channels must hit an admitted bucket.
    private static readonly (int Batch, int Channels, int Spatial)[] Shapes =
        [(2, 128, 128), (4, 512, 64), (4, 512, 128), (16, 512, 128)];

    private readonly record struct Distribution(double Mean, double Median, double P95, double P99);
    private readonly record struct Result(
        int Rows, int Spatial, string Method, Distribution Time, double GigabytesPerSecond,
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

        Console.WriteLine($"NVIDIA GPU-only FP32 global average pool (resident tensors), diagnostic repetition {repetition}/{repetitions}");
        Console.WriteLine($"{"Rows",7} {"HW",6} {"Method",-27} {"median us",10} {"p95 us",10} {"mean us",10} {"GB/s",9} {"B/call",9} {"max rel",10} {"regs",6} {"local B",8}");
        Console.WriteLine(new string('-', 128));
        foreach (Result r in results.OrderBy(r => r.Rows).ThenBy(r => r.Spatial).ThenBy(r => r.Time.Median))
        {
            string registers = r.Registers < 0 ? "n/a" : r.Registers.ToString();
            string local = r.LocalBytes < 0 ? "n/a" : r.LocalBytes.ToString();
            Console.WriteLine(
                $"{r.Rows,7} {r.Spatial,6} {r.Method,-27} {r.Time.Median * 1000,10:F2} {r.Time.P95 * 1000,10:F2} " +
                $"{r.Time.Mean * 1000,10:F2} {r.GigabytesPerSecond,9:F2} {r.Allocation,9} {r.MaxError,10:G4} {registers,6} {local,8}");
        }

        Console.WriteLine();
        Console.WriteLine("Diagnostic gate: each in-process repetition counts as one diagnostic only; production still requires three clean, separately launched captures.");
        DirectPtxReleaseGatePolicy policy = DirectPtxReleaseGatePolicy.ProductionDefault with { RequiredIndependentRuns = 1 };
        foreach ((int batch, int channels, int spatial) in Shapes)
        {
            int rows = batch * channels;
            var direct = results.SingleOrDefault(r => r.Rows == rows && r.Spatial == spatial && r.Method == "Direct PTX global avgpool");
            if (direct.Method is null) continue;
            var competitors = results.Where(r => r.Rows == rows && r.Spatial == spatial && r.Method != "Direct PTX global avgpool").ToList();
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
                $"[{rows},{spatial}]: {(decision.Passed ? "PASS" : "HOLD"),-4} {decision.MedianSpeedup:F2}x vs {best.Method}; " +
                (decision.Passed ? "all gates passed" : string.Join("; ", decision.Failures)));
        }
        GpuBenchmarkEnvironment.PrintSnapshot($"end-{repetition}");
    }

    private static void RunDirect(List<Result> results)
    {
        using var runtime = new DirectPtxRuntime();
        if (runtime.ArchitectureFamily != DirectPtxArchitectureFamily.Ampere) return;
        using (runtime.Enter())
        foreach ((int batch, int channels, int spatial) in Shapes)
        {
            int rows = batch * channels;
            using var kernel = new PtxFusedGlobalAvgPoolF32Kernel(runtime, rows, spatial);
            float[] input = Values(rows * spatial, 100 + rows + spatial);
            using var inputBuffer = runtime.AllocateBytes(kernel.Blueprint.Tensors[0].RequiredBytes);
            using var outputBuffer = runtime.AllocateBytes(kernel.Blueprint.Tensors[1].RequiredBytes);
            inputBuffer.Upload<float>(input);
            Action launch = () => kernel.Launch(
                DirectPtxTensorView.CreateOwned(inputBuffer, kernel.Blueprint.Tensors[0]),
                DirectPtxTensorView.CreateOwned(outputBuffer, kernel.Blueprint.Tensors[1]));
            Distribution distribution = Measure(runtime.Synchronize, launch);
            long allocation = Allocation(runtime.Synchronize, launch);
            launch(); runtime.Synchronize();
            var actual = new float[rows];
            outputBuffer.Download<float>(actual);
            float error = Validate(actual, input, rows, spatial);
            results.Add(new Result(rows, spatial, "Direct PTX global avgpool", distribution,
                Bandwidth(rows, spatial, distribution.Median), allocation, 0, error,
                kernel.Audit.Function.RegistersPerThread, kernel.Audit.Function.LocalBytesPerThread));
        }
    }

    private static void RunAiDotNet(List<Result> results)
    {
        using var backend = new CudaBackend();
        if (!backend.IsAvailable) return;
        foreach ((int batch, int channels, int spatial) in Shapes)
        {
            int rows = batch * channels;
            float[] input = Values(rows * spatial, 100 + rows + spatial);
            using var inputBuffer = backend.AllocateBuffer(input);
            using var outputBuffer = backend.AllocateBuffer(rows);
            Action launch = () => backend.GlobalAvgPool2D(inputBuffer, outputBuffer, batch, channels, spatial, 1);
            Distribution distribution = Measure(backend.Synchronize, launch);
            long allocation = Allocation(backend.Synchronize, launch);
            results.Add(new Result(rows, spatial, "AiDotNet global_avgpool2d", distribution,
                Bandwidth(rows, spatial, distribution.Median), allocation, 0, 0f, -1, -1));
        }
    }

    private static void RunPyTorch(List<Result> results)
    {
        if (!torch.cuda.is_available()) return;
        foreach ((int batch, int channels, int spatial) in Shapes)
        {
            int rows = batch * channels;
            float[] input = Values(rows * spatial, 100 + rows + spatial);
            using TorchTensor x = torch.tensor(input, [rows, spatial], device: torch.CUDA);
            void Launch()
            {
                using TorchTensor output = x.mean([1L]);
            }
            Distribution distribution = Measure(() => torch.cuda.synchronize(), Launch);
            long allocation = Allocation(() => torch.cuda.synchronize(), Launch);
            results.Add(new Result(rows, spatial, "PyTorch mean(dim=-1)", distribution,
                Bandwidth(rows, spatial, distribution.Median), allocation, -1, 0f, -1, -1));
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

    private static double Bandwidth(int rows, int spatial, double milliseconds)
    {
        long bytes = checked((long)rows * spatial * sizeof(float) + (long)rows * sizeof(float));
        return bytes / (milliseconds * 1e-3) / 1e9;
    }

    private static float[] Values(int length, int seed)
    {
        var random = new Random(seed);
        return Enumerable.Range(0, length).Select(_ => (random.NextSingle() * 2f - 1f) * 4f).ToArray();
    }

    private static float Validate(float[] actual, float[] input, int rows, int spatial)
    {
        float maximum = 0;
        for (int row = 0; row < rows; row++)
        {
            double sum = 0;
            for (int s = 0; s < spatial; s++)
                sum += input[row * spatial + s];
            float expected = (float)(sum / spatial);
            maximum = MathF.Max(maximum, MathF.Abs(actual[row] - expected) / (MathF.Abs(expected) + 1f));
        }
        return maximum;
    }
}
