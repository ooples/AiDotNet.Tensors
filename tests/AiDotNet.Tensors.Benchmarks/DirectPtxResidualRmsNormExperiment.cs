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
    private readonly struct Error
    {
        internal Error(float maxAbsolute, float maxRelative) =>
            (MaxAbsolute, MaxRelative) = (maxAbsolute, maxRelative);
        internal float MaxAbsolute { get; }
        internal float MaxRelative { get; }
    }

    private readonly struct Distribution
    {
        internal Distribution(double mean, double median, double p95, double p99) =>
            (Mean, Median, P95, P99) = (mean, median, p95, p99);

        internal double Mean { get; }
        internal double Median { get; }
        internal double P95 { get; }
        internal double P99 { get; }
    }

    private readonly struct Result
    {
        internal Result(int rows, string method, Distribution time, double gigabytesPerSecond,
            long allocation, long temporaryBytes, Error error, int registers,
            int sharedBytes, int localBytes, double occupancy) =>
            (Rows, Method, Time, GigabytesPerSecond, Allocation, TemporaryBytes, Error,
                Registers, StaticSharedBytes, DynamicSharedBytes, LocalBytes, Occupancy) =
            (rows, method, time, gigabytesPerSecond, allocation, temporaryBytes, error,
                registers, sharedBytes, sharedBytes < 0 ? -1 : 0, localBytes, occupancy);

        internal int Rows { get; }
        internal string Method { get; }
        internal Distribution Time { get; }
        internal double GigabytesPerSecond { get; }
        internal long Allocation { get; }
        internal long TemporaryBytes { get; }
        internal Error Error { get; }
        internal int Registers { get; }
        internal int StaticSharedBytes { get; }
        internal int DynamicSharedBytes { get; }
        internal int LocalBytes { get; }
        internal double Occupancy { get; }
    }

    internal static void Run()
    {
        GpuBenchmarkEnvironment.RequireIdleGpu("residual-rmsnorm-start");
        GpuBenchmarkEnvironment.PrintSnapshot("start");
        int[] rowCounts = [32, 256, 2048, 8192];
        var results = new List<Result>();
        RunDirect(rowCounts, results);
        GpuBenchmarkEnvironment.RequireIdleGpu("residual-rmsnorm-framework-baselines");
        RunAiDotNet(rowCounts, results);
        RunPyTorch(rowCounts, results);
        GpuBenchmarkEnvironment.RequireNoForeignCompute("residual-rmsnorm-end");

        Console.WriteLine("NVIDIA GPU-only fused residual + RMSNorm D=64 (resident FP32 tensors)");
        Console.WriteLine("Correctness max-absolute tolerance: 5e-5 for both normalized output and saved RMS.");
        Console.WriteLine("Effective FLOPs = 5 * elements + 3 * rows (add, square/reduction, normalize/affine, sqrt/reciprocal).");
        Console.WriteLine($"{"Rows",7} {"Method",-29} {"median us",10} {"p95 us",10} {"p99 us",10} {"mean us",10} {"GB/s",9} {"GFLOPS",10} {"TFLOPS",9} {"managed B",9} {"tmp MiB",9} {"max abs",10} {"max rel",10} {"regs",6} {"static B",8} {"dynamic B",9} {"local B",8} {"occ",7}");
        Console.WriteLine(new string('-', 202));
        foreach (Result result in results.OrderBy(r => r.Rows).ThenBy(r => r.Time.Median))
        {
            string temporary = result.TemporaryBytes < 0
                ? "n/a"
                : (result.TemporaryBytes / 1048576.0).ToString("F3");
            string registers = result.Registers < 0 ? "n/a" : result.Registers.ToString();
            string staticShared = result.StaticSharedBytes < 0 ? "n/a" : result.StaticSharedBytes.ToString();
            string dynamicShared = result.DynamicSharedBytes < 0 ? "n/a" : result.DynamicSharedBytes.ToString();
            string local = result.LocalBytes < 0 ? "n/a" : result.LocalBytes.ToString();
            string occupancy = double.IsNaN(result.Occupancy) ? "n/a" : result.Occupancy.ToString("P0");
            double gflops = EffectiveGflops(result.Rows, result.Time.Median);
            Console.WriteLine(
                $"{result.Rows,7} {result.Method,-29} {result.Time.Median * 1000,10:F2} " +
                $"{result.Time.P95 * 1000,10:F2} {result.Time.P99 * 1000,10:F2} " +
                $"{result.Time.Mean * 1000,10:F2} {result.GigabytesPerSecond,9:F2} " +
                $"{gflops,10:F2} {gflops / 1000,9:F4} " +
                $"{result.Allocation,9} {temporary,9} {result.Error.MaxAbsolute,10:G4} " +
                $"{result.Error.MaxRelative,10:G4} {registers,6} " +
                $"{staticShared,8} {dynamicShared,9} {local,8} {occupancy,7}");
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
                direct.Allocation, direct.TemporaryBytes, direct.Error.MaxAbsolute,
                direct.LocalBytes, IndependentRuns: 1);
            var competitorEvidence = new DirectPtxPerformanceEvidence(
                best.Time.Median * 1000, best.Time.P95 * 1000,
                best.Allocation, best.TemporaryBytes, best.Error.MaxAbsolute,
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
            GpuBenchmarkEnvironment.RequireNoForeignCompute($"residual-rmsnorm-direct-{rows}");
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
            var actualRms = new float[rows];
            rms.Download<float>(actualRms);
            Error error = Validate(actual, actualRms, x, residualHost, gammaHost, rows);
            double occupancy = kernel.Audit.ActiveBlocksPerMultiprocessor *
                kernel.Audit.BlockThreads / (double)runtime.MaxThreadsPerMultiprocessor;
            results.Add(new Result(rows, "Direct PTX fused", distribution,
                Bandwidth(rows, distribution.Median), allocation, 0, error,
                kernel.FunctionInfo.RegistersPerThread, kernel.FunctionInfo.StaticSharedBytes,
                kernel.FunctionInfo.LocalBytesPerThread, occupancy));
        }
    }

    private static void RunAiDotNet(int[] rowCounts, List<Result> results)
    {
        using var backend = new CudaBackend();
        foreach (int rows in rowCounts)
        {
            GpuBenchmarkEnvironment.RequireNoForeignCompute($"residual-rmsnorm-aidotnet-{rows}");
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
            Error error = Validate(
                backend.DownloadBuffer(output), backend.DownloadBuffer(rms),
                x, residualHost, gammaHost, rows);
            results.Add(new Result(rows, "AiDotNet add + RMSNorm", distribution,
                Bandwidth(rows, distribution.Median), allocation, (long)x.Length * sizeof(float),
                error, -1, -1, -1, double.NaN));
        }
    }

    private static void RunPyTorch(int[] rowCounts, List<Result> results)
    {
        if (!torch.cuda.is_available()) return;
        foreach (int rows in rowCounts)
        {
            GpuBenchmarkEnvironment.RequireNoForeignCompute($"residual-rmsnorm-pytorch-{rows}");
            float[] xHost = Values(rows * Dimension, 100 + rows, 1f);
            float[] residualHost = Values(rows * Dimension, 200 + rows, 0.25f);
            float[] gammaHost = Enumerable.Range(0, Dimension).Select(i => 0.75f + i / 256f).ToArray();
            using TorchTensor input = torch.tensor(xHost, [rows, Dimension], device: torch.CUDA);
            using TorchTensor residual = torch.tensor(residualHost, [rows, Dimension], device: torch.CUDA);
            using TorchTensor gamma = torch.tensor(gammaHost, [Dimension], device: torch.CUDA);
            (TorchTensor Output, TorchTensor Rms) Compute()
            {
                using TorchTensor sum = input.add(residual);
                using TorchTensor square = sum.mul(sum);
                using TorchTensor mean = square.mean([1L], keepdim: true);
                using TorchTensor shifted = mean.add(Epsilon);
                TorchTensor rms = shifted.sqrt();
                TorchTensor output = sum.div(rms).mul_(gamma);
                return (output, rms);
            }
            void Launch()
            {
                (TorchTensor output, TorchTensor rms) = Compute();
                using (output) using (rms) { }
            }
            Error error;
            (TorchTensor checkOutput, TorchTensor checkRms) = Compute();
            using (checkOutput)
            using (checkRms)
            using (TorchTensor outputCpu = checkOutput.cpu())
            using (TorchTensor rmsCpu = checkRms.cpu())
                error = Validate(
                    outputCpu.data<float>().ToArray(), rmsCpu.data<float>().ToArray(),
                    xHost, residualHost, gammaHost, rows);
            Distribution distribution = Measure(() => torch.cuda.synchronize(), Launch);
            long allocation = Allocation(() => torch.cuda.synchronize(), Launch);
            long temporary = (long)xHost.Length * sizeof(float) * 2 +
                (long)rows * sizeof(float) * 2;
            results.Add(new Result(rows, "PyTorch eager composition", distribution,
                Bandwidth(rows, distribution.Median), allocation, temporary, error,
                -1, -1, -1, double.NaN));
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
#if NET5_0_OR_GREATER
        long before = GC.GetAllocatedBytesForCurrentThread();
        for (int i = 0; i < 50; i++) { launch(); synchronize(); }
        long allocation = (GC.GetAllocatedBytesForCurrentThread() - before) / 50;
#else
        long before = GC.GetTotalMemory(forceFullCollection: false);
        for (int i = 0; i < 50; i++) { launch(); synchronize(); }
        long allocation = Math.Max(0, GC.GetTotalMemory(forceFullCollection: false) - before) / 50;
#endif
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

    private static double EffectiveGflops(int rows, double milliseconds)
    {
        long elements = checked((long)rows * Dimension);
        long operations = checked(5L * elements + 3L * rows);
        return operations / milliseconds / 1e6;
    }

    private static float[] Values(int length, int seed, float magnitude)
    {
        var random = new Random(seed);
        return Enumerable.Range(0, length)
            .Select(_ => (random.NextSingle() - 0.5f) * 2f * magnitude).ToArray();
    }

    private static Error Validate(
        float[] actual, float[] actualRms,
        float[] input, float[] residual, float[] gamma, int rows)
    {
        float maxAbsolute = 0;
        float maxRelative = 0;
        for (int row = 0; row < rows; row++)
        {
            float sumSquares = 0;
            for (int d = 0; d < Dimension; d++)
            {
                float value = input[row * Dimension + d] + residual[row * Dimension + d];
                sumSquares += value * value;
            }
            float rms = MathF.Sqrt(sumSquares / Dimension + Epsilon);
            AccumulateError(actualRms[row], rms, ref maxAbsolute, ref maxRelative);
            for (int d = 0; d < Dimension; d++)
            {
                float expected = (input[row * Dimension + d] + residual[row * Dimension + d]) /
                    rms * gamma[d];
                float actualValue = actual[row * Dimension + d];
                AccumulateError(actualValue, expected, ref maxAbsolute, ref maxRelative);
            }
        }
        if (!float.IsFinite(maxAbsolute) || !float.IsFinite(maxRelative) || maxAbsolute >= 5e-5f)
            throw new InvalidOperationException(
                $"Residual RMSNorm validation failed: max abs {maxAbsolute:G9}, " +
                $"max symmetric relative {maxRelative:G9}");
        return new Error(maxAbsolute, maxRelative);
    }

    private static void AccumulateError(
        float actual, float expected, ref float maxAbsolute, ref float maxRelative)
    {
        float absolute = MathF.Abs(actual - expected);
        float relative = 2f * absolute / (MathF.Abs(actual) + MathF.Abs(expected) + 1e-3f);
        maxAbsolute = MathF.Max(maxAbsolute, absolute);
        maxRelative = MathF.Max(maxRelative, relative);
    }
}
