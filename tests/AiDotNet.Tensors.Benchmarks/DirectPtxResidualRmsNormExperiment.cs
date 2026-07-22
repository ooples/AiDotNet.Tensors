using System.Diagnostics;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;
using TorchSharp;
using TorchTensor = TorchSharp.torch.Tensor;

namespace AiDotNet.Tensors.Benchmarks;

/// <summary>Second-blueprint NVIDIA-only fused residual + RMSNorm benchmark.</summary>
internal static class DirectPtxResidualRmsNormExperiment
{
    private const int Dimension = 64;
    private const float Epsilon = 1e-5f;
    private readonly record struct Distribution(double Mean, double Median, double P95, double P99);
    private readonly record struct Result(
        int Rows, string Method, Distribution Time, double GigabytesPerSecond,
        long Allocation, long TemporaryBytes, float MaxError, int Registers, int LocalBytes);

    internal static void Run()
    {
        GpuBenchmarkEnvironment.PrintSnapshot("start");
        int[] rowCounts = [32, 256, 2048, 8192];
        var results = new List<Result>();
        RunDirect(rowCounts, results);
        RunAiDotNet(rowCounts, results);
        RunPyTorch(rowCounts, results);

        Console.WriteLine("NVIDIA GPU-only fused residual + RMSNorm D=64 (resident FP32 tensors)");
        Console.WriteLine($"{"Rows",7} {"Method",-29} {"median us",10} {"p95 us",10} {"p99 us",10} {"mean us",10} {"GB/s",9} {"B/call",9} {"tmp MiB",9} {"max err",10} {"regs",6} {"local B",8}");
        Console.WriteLine(new string('-', 135));
        foreach (Result result in results.OrderBy(r => r.Rows).ThenBy(r => r.Time.Median))
        {
            string temporary = result.TemporaryBytes < 0
                ? "n/a"
                : (result.TemporaryBytes / 1048576.0).ToString("F3");
            string registers = result.Registers < 0 ? "n/a" : result.Registers.ToString();
            string local = result.LocalBytes < 0 ? "n/a" : result.LocalBytes.ToString();
            Console.WriteLine(
                $"{result.Rows,7} {result.Method,-29} {result.Time.Median * 1000,10:F2} " +
                $"{result.Time.P95 * 1000,10:F2} {result.Time.P99 * 1000,10:F2} " +
                $"{result.Time.Mean * 1000,10:F2} {result.GigabytesPerSecond,9:F2} " +
                $"{result.Allocation,9} {temporary,9} {result.MaxError,10:G4} {registers,6} {local,8}");
        }

        Console.WriteLine();
        Console.WriteLine("Diagnostic gate: production policy except this single capture counts as one independent run (production requires three).");
        DirectPtxReleaseGatePolicy policy = DirectPtxReleaseGatePolicy.ProductionDefault with
        {
            RequiredIndependentRuns = 1
        };
        foreach (int rows in rowCounts)
        {
            Result direct = results.Single(r => r.Rows == rows && r.Method == "Direct PTX fused");
            Result best = results.Where(r => r.Rows == rows && r.Method != "Direct PTX fused")
                .OrderBy(r => r.Time.Median).First();
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
                $"rows={rows,-5}: {(decision.Passed ? "PASS" : "HOLD"),-4} {decision.MedianSpeedup:F2}x vs {best.Method}; " +
                (decision.Passed ? "all gates passed" : string.Join("; ", decision.Failures)));
        }
        GpuBenchmarkEnvironment.PrintSnapshot("end");
    }

    private static void RunDirect(int[] rowCounts, List<Result> results)
    {
        using var runtime = new DirectPtxRuntime();
        if (runtime.ArchitectureFamily != DirectPtxArchitectureFamily.Ampere) return;
        using (runtime.Enter())
        foreach (int rows in rowCounts)
        {
            using var kernel = new PtxFusedResidualRmsNormD64Kernel(runtime, rows, Epsilon);
            float[] x = Values(rows * Dimension, 100 + rows, 1f);
            float[] residualHost = Values(rows * Dimension, 200 + rows, 0.25f);
            float[] gammaHost = Enumerable.Range(0, Dimension).Select(i => 0.75f + i / 256f).ToArray();
            using var input = runtime.AllocateBytes(kernel.InputBytes);
            using var residual = runtime.AllocateBytes(kernel.InputBytes);
            using var gamma = runtime.AllocateBytes(PtxFusedResidualRmsNormD64Kernel.GammaBytes);
            using var output = runtime.AllocateBytes(kernel.OutputBytes);
            using var rms = runtime.AllocateBytes(kernel.RmsBytes);
            input.Upload<float>(x);
            residual.Upload<float>(residualHost);
            gamma.Upload<float>(gammaHost);
            Action launch = () => kernel.Launch(
                DirectPtxTensorView.CreateOwned(input, kernel.Blueprint.Tensors[0]),
                DirectPtxTensorView.CreateOwned(residual, kernel.Blueprint.Tensors[1]),
                DirectPtxTensorView.CreateOwned(gamma, kernel.Blueprint.Tensors[2]),
                DirectPtxTensorView.CreateOwned(output, kernel.Blueprint.Tensors[3]),
                DirectPtxTensorView.CreateOwned(rms, kernel.Blueprint.Tensors[4]));
            Distribution distribution = Measure(runtime.Synchronize, launch);
            long allocation = Allocation(runtime.Synchronize, launch);
            launch(); runtime.Synchronize();
            var actual = new float[x.Length];
            output.Download<float>(actual);
            float error = Validate(actual, x, residualHost, gammaHost, rows);
            results.Add(new Result(rows, "Direct PTX fused", distribution,
                Bandwidth(rows, distribution.Median), allocation, 0, error,
                kernel.FunctionInfo.RegistersPerThread, kernel.FunctionInfo.LocalBytesPerThread));
        }
    }

    private static void RunAiDotNet(int[] rowCounts, List<Result> results)
    {
        using var backend = new CudaBackend();
        foreach (int rows in rowCounts)
        {
            float[] x = Values(rows * Dimension, 100 + rows, 1f);
            float[] residualHost = Values(rows * Dimension, 200 + rows, 0.25f);
            float[] gammaHost = Enumerable.Range(0, Dimension).Select(i => 0.75f + i / 256f).ToArray();
            using var input = backend.AllocateBuffer(x);
            using var residual = backend.AllocateBuffer(residualHost);
            using var gamma = backend.AllocateBuffer(gammaHost);
            using var sum = backend.AllocateBuffer(x.Length);
            using var output = backend.AllocateBuffer(x.Length);
            using var rms = backend.AllocateBuffer(rows);
            Action launch = () =>
            {
                backend.Add(input, residual, sum, x.Length);
                backend.RmsNorm(sum, output, gamma, rms, rows, Dimension, Epsilon);
            };
            Distribution distribution = Measure(backend.Synchronize, launch);
            long allocation = Allocation(backend.Synchronize, launch);
            launch(); backend.Synchronize();
            float error = Validate(backend.DownloadBuffer(output), x, residualHost, gammaHost, rows);
            results.Add(new Result(rows, "AiDotNet add + RMSNorm", distribution,
                Bandwidth(rows, distribution.Median), allocation, (long)x.Length * sizeof(float),
                error, -1, -1));
        }
    }

    private static void RunPyTorch(int[] rowCounts, List<Result> results)
    {
        if (!torch.cuda.is_available()) return;
        foreach (int rows in rowCounts)
        {
            float[] xHost = Values(rows * Dimension, 100 + rows, 1f);
            float[] residualHost = Values(rows * Dimension, 200 + rows, 0.25f);
            float[] gammaHost = Enumerable.Range(0, Dimension).Select(i => 0.75f + i / 256f).ToArray();
            using TorchTensor input = torch.tensor(xHost, [rows, Dimension], device: torch.CUDA);
            using TorchTensor residual = torch.tensor(residualHost, [rows, Dimension], device: torch.CUDA);
            using TorchTensor gamma = torch.tensor(gammaHost, [Dimension], device: torch.CUDA);
            void Launch()
            {
                using TorchTensor sum = input.add(residual);
                using TorchTensor square = sum.mul(sum);
                using TorchTensor mean = square.mean([1L], keepdim: true);
                using TorchTensor rms = mean.add(Epsilon).sqrt();
                using TorchTensor output = sum.div(rms).mul_(gamma);
            }
            Distribution distribution = Measure(() => torch.cuda.synchronize(), Launch);
            long allocation = Allocation(() => torch.cuda.synchronize(), Launch);
            results.Add(new Result(rows, "PyTorch eager composition", distribution,
                Bandwidth(rows, distribution.Median), allocation, -1, 0, -1, -1));
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

    private static double Bandwidth(int rows, double milliseconds)
    {
        long bytes = checked((long)rows * Dimension * sizeof(float) * 3L +
            Dimension * sizeof(float) + rows * sizeof(float));
        return bytes / (milliseconds * 1e-3) / 1e9;
    }

    private static float[] Values(int length, int seed, float magnitude)
    {
        var random = new Random(seed);
        return Enumerable.Range(0, length)
            .Select(_ => (random.NextSingle() - 0.5f) * 2f * magnitude).ToArray();
    }

    private static float Validate(
        float[] actual, float[] input, float[] residual, float[] gamma, int rows)
    {
        float maximum = 0;
        for (int row = 0; row < rows; row++)
        {
            float sumSquares = 0;
            for (int d = 0; d < Dimension; d++)
            {
                float value = input[row * Dimension + d] + residual[row * Dimension + d];
                sumSquares += value * value;
            }
            float rms = MathF.Sqrt(sumSquares / Dimension + Epsilon);
            for (int d = 0; d < Dimension; d++)
            {
                float expected = (input[row * Dimension + d] + residual[row * Dimension + d]) /
                    rms * gamma[d];
                maximum = MathF.Max(maximum,
                    MathF.Abs(actual[row * Dimension + d] - expected));
            }
        }
        return maximum;
    }
}
