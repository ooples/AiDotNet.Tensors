using System.Diagnostics;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;
using TorchSharp;
using TorchTensor = TorchSharp.torch.Tensor;

namespace AiDotNet.Tensors.Benchmarks;

/// <summary>
/// Issue-#844 NVIDIA-only FP32 embedding gather benchmark. Compares the
/// hand-emitted direct-PTX kernel against the current AiDotNet CUDA
/// embedding_forward kernel and the strongest resident PyTorch index_select on
/// identical device tensors. CPU MKL/OpenBLAS are intentionally ineligible.
/// In-process repetitions are diagnostic only; production promotion consumes
/// three separately launched, clean captures.
/// </summary>
internal static class DirectPtxGatherExperiment
{
    private const int TableRows = 4096;
    private static readonly (int NumIndices, int FeatureSize)[] Shapes =
        [(256, 128), (2048, 64), (2048, 128), (8192, 128)];

    private readonly record struct Distribution(double Mean, double Median, double P95, double P99);
    private readonly record struct Result(
        int NumIndices, int FeatureSize, string Method, Distribution Time, double GigabytesPerSecond,
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

        Console.WriteLine(
            $"NVIDIA GPU-only FP32 embedding gather (resident tensors), diagnostic repetition {repetition}/{repetitions}");
        Console.WriteLine($"{"NIdx",7} {"Feat",6} {"Method",-27} {"median us",10} {"p95 us",10} {"mean us",10} {"GB/s",9} {"B/call",9} {"max err",10} {"regs",6} {"local B",8}");
        Console.WriteLine(new string('-', 128));
        foreach (Result r in results.OrderBy(r => r.NumIndices).ThenBy(r => r.FeatureSize).ThenBy(r => r.Time.Median))
        {
            string registers = r.Registers < 0 ? "n/a" : r.Registers.ToString();
            string local = r.LocalBytes < 0 ? "n/a" : r.LocalBytes.ToString();
            Console.WriteLine(
                $"{r.NumIndices,7} {r.FeatureSize,6} {r.Method,-27} {r.Time.Median * 1000,10:F2} " +
                $"{r.Time.P95 * 1000,10:F2} {r.Time.Mean * 1000,10:F2} {r.GigabytesPerSecond,9:F2} " +
                $"{r.Allocation,9} {r.MaxError,10:G4} {registers,6} {local,8}");
        }

        Console.WriteLine();
        Console.WriteLine("Diagnostic gate: each in-process repetition counts as one diagnostic only; production still requires three clean, separately launched captures.");
        DirectPtxReleaseGatePolicy policy = DirectPtxReleaseGatePolicy.ProductionDefault with { RequiredIndependentRuns = 1 };
        foreach ((int numIndices, int featureSize) in Shapes)
        {
            var direct = results.SingleOrDefault(r => r.NumIndices == numIndices && r.FeatureSize == featureSize && r.Method == "Direct PTX gather");
            if (direct.Method is null) continue;
            var competitors = results.Where(r => r.NumIndices == numIndices && r.FeatureSize == featureSize && r.Method != "Direct PTX gather").ToList();
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
                $"[{numIndices},{featureSize}]: {(decision.Passed ? "PASS" : "HOLD"),-4} {decision.MedianSpeedup:F2}x vs {best.Method}; " +
                (decision.Passed ? "all gates passed" : string.Join("; ", decision.Failures)));
        }
        GpuBenchmarkEnvironment.PrintSnapshot($"end-{repetition}");
    }

    private static void RunDirect(List<Result> results)
    {
        using var runtime = new DirectPtxRuntime();
        if (runtime.ArchitectureFamily != DirectPtxArchitectureFamily.Ampere) return;
        using (runtime.Enter())
        foreach ((int numIndices, int featureSize) in Shapes)
        {
            using var kernel = new PtxFusedGatherF32Kernel(runtime, numIndices, featureSize);
            float[] table = Values(TableRows * featureSize, 100 + numIndices + featureSize);
            int[] idx = Indices(numIndices, TableRows, 900 + numIndices + featureSize);
            using var source = runtime.AllocateBytes((nuint)table.Length * sizeof(float));
            using var indices = runtime.AllocateBytes((nuint)numIndices * sizeof(int));
            using var output = runtime.AllocateBytes(kernel.Blueprint.Tensors[2].RequiredBytes);
            source.Upload<float>(table);
            indices.Upload<int>(idx);
            Action launch = () => kernel.Launch(
                DirectPtxTensorView.CreateOwned(indices, kernel.Blueprint.Tensors[0]),
                DirectPtxTensorView.CreateOwned(source, kernel.Blueprint.Tensors[1]),
                DirectPtxTensorView.CreateOwned(output, kernel.Blueprint.Tensors[2]));
            Distribution distribution = Measure(runtime.Synchronize, launch);
            long allocation = Allocation(runtime.Synchronize, launch);
            launch(); runtime.Synchronize();
            var actual = new float[numIndices * featureSize];
            output.Download<float>(actual);
            float error = Validate(actual, table, idx, featureSize);
            results.Add(new Result(numIndices, featureSize, "Direct PTX gather", distribution,
                Bandwidth(numIndices, featureSize, distribution.Median), allocation, 0, error,
                kernel.Audit.Function.RegistersPerThread, kernel.Audit.Function.LocalBytesPerThread));
        }
    }

    private static void RunAiDotNet(List<Result> results)
    {
        using var backend = new CudaBackend();
        if (!backend.IsAvailable) return;
        foreach ((int numIndices, int featureSize) in Shapes)
        {
            float[] table = Values(TableRows * featureSize, 100 + numIndices + featureSize);
            int[] idx = Indices(numIndices, TableRows, 900 + numIndices + featureSize);
            using var source = backend.AllocateBuffer(table);
            using var indices = backend.AllocateIntBuffer(idx);
            using var output = backend.AllocateBuffer(numIndices * featureSize);
            Action launch = () => backend.Gather(source, indices, output, numIndices, featureSize);
            Distribution distribution = Measure(backend.Synchronize, launch);
            long allocation = Allocation(backend.Synchronize, launch);
            launch(); backend.Synchronize();
            float error = Validate(backend.DownloadBuffer(output), table, idx, featureSize);
            results.Add(new Result(numIndices, featureSize, "AiDotNet embedding_forward", distribution,
                Bandwidth(numIndices, featureSize, distribution.Median), allocation, 0, error, -1, -1));
        }
    }

    private static void RunPyTorch(List<Result> results)
    {
        if (!torch.cuda.is_available()) return;
        foreach ((int numIndices, int featureSize) in Shapes)
        {
            float[] table = Values(TableRows * featureSize, 100 + numIndices + featureSize);
            int[] idx = Indices(numIndices, TableRows, 900 + numIndices + featureSize);
            using TorchTensor source = torch.tensor(table, [TableRows, featureSize], device: torch.CUDA);
            using TorchTensor indices = torch.tensor(
                idx.Select(i => (long)i).ToArray(), [numIndices], device: torch.CUDA);
            void Launch()
            {
                using TorchTensor output = source.index_select(0, indices);
            }
            Distribution distribution = Measure(() => torch.cuda.synchronize(), Launch);
            long allocation = Allocation(() => torch.cuda.synchronize(), Launch);
            results.Add(new Result(numIndices, featureSize, "PyTorch index_select", distribution,
                Bandwidth(numIndices, featureSize, distribution.Median), allocation, -1, 0, -1, -1));
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

    private static double Bandwidth(int numIndices, int featureSize, double milliseconds)
    {
        long bytes = checked((long)numIndices * featureSize * sizeof(float) * 2L + (long)numIndices * sizeof(int));
        return bytes / (milliseconds * 1e-3) / 1e9;
    }

    private static float[] Values(int length, int seed)
    {
        var random = new Random(seed);
        return Enumerable.Range(0, length).Select(_ => (random.NextSingle() - 0.5f) * 16f).ToArray();
    }

    private static int[] Indices(int numIndices, int tableRows, int seed)
    {
        var random = new Random(seed);
        return Enumerable.Range(0, numIndices).Select(_ => random.Next(tableRows)).ToArray();
    }

    private static float Validate(float[] actual, float[] table, int[] indices, int featureSize)
    {
        float maximum = 0;
        for (int i = 0; i < indices.Length; i++)
            for (int f = 0; f < featureSize; f++)
                maximum = MathF.Max(maximum,
                    MathF.Abs(actual[i * featureSize + f] - table[indices[i] * featureSize + f]));
        return maximum;
    }
}
