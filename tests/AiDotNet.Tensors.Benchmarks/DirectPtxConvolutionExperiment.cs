using System.Diagnostics;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

namespace AiDotNet.Tensors.Benchmarks;

/// <summary>
/// Resident NVIDIA-only issue-#841 evidence harness. It intentionally refuses
/// to turn one process into promotion evidence; release evidence requires this
/// command in three clean processes plus the external PyTorch runner.
/// </summary>
internal static class DirectPtxConvolutionExperiment
{
    private readonly record struct Distribution(double Mean, double Median, double P95, double P99);
    private readonly record struct Result(
        string Method, Distribution Time, double Gflops, long ManagedBytes,
        long TemporaryDeviceBytes, float MaximumAbsoluteError,
        int Registers, int StaticSharedBytes, int LocalBytes, int ActiveBlocksPerSm);

    internal static void Run(bool includeExternal)
    {
        GpuBenchmarkEnvironment.RequireIdleGpu("direct-ptx-convolution-start");
        GpuBenchmarkEnvironment.PrintSnapshot("direct-ptx-convolution-start");
        float[] input = Values(PtxFusedConv2DNchwK1Kernel.InputBytes / sizeof(float), 841);
        float[] weights = Values(PtxFusedConv2DNchwK1Kernel.WeightBytes / sizeof(float), 842);
        float[] bias = Values(PtxFusedConv2DNchwK1Kernel.BiasBytes / sizeof(float), 843);
        float[] expected = Oracle(input, weights, bias);
        var results = new List<Result>
        {
            RunDirectPtx(input, weights, bias, expected),
            RunEstablishedAiDotNet(input, weights, bias, expected)
        };

        Console.WriteLine("Issue #841 resident FP32 NCHW Conv2D 1x1 + bias + ReLU");
        Console.WriteLine("Shape: N1/C64/H16/W16/K64; convolution FLOPs exclude bias/ReLU.");
        Console.WriteLine("Screening only: promotion requires 3 clean processes, PyTorch competitors, and Nsight evidence.");
        Console.WriteLine($"{"Method",-38} {"median us",11} {"P95 us",11} {"P99 us",11} {"mean us",11} {"GFLOPS",11} {"TFLOPS",10} {"managed B",11} {"temp B",11} {"max abs",11} {"regs",7} {"static B",9} {"local B",8} {"blocks/SM",10}");
        Console.WriteLine(new string('-', 184));
        foreach (Result result in results.OrderBy(result => result.Time.Median))
        {
            Console.WriteLine(
                $"{result.Method,-38} {result.Time.Median * 1000,11:F2} " +
                $"{result.Time.P95 * 1000,11:F2} {result.Time.P99 * 1000,11:F2} " +
                $"{result.Time.Mean * 1000,11:F2} {result.Gflops,11:F2} " +
                $"{result.Gflops / 1000,10:F4} {result.ManagedBytes,11} " +
                $"{Format(result.TemporaryDeviceBytes),11} {result.MaximumAbsoluteError,11:G4} " +
                $"{Format(result.Registers),7} {Format(result.StaticSharedBytes),9} " +
                $"{Format(result.LocalBytes),8} {Format(result.ActiveBlocksPerSm),10}");
        }

        if (includeExternal) RunPyTorchCompetitors();
        GpuBenchmarkEnvironment.RequireNoForeignCompute("direct-ptx-convolution-end");
        GpuBenchmarkEnvironment.PrintSnapshot("direct-ptx-convolution-end");
    }

    private static Result RunDirectPtx(
        float[] inputHost, float[] weightHost, float[] biasHost, float[] expected)
    {
        using var runtime = new DirectPtxRuntime();
        using var kernel = new PtxFusedConv2DNchwK1Kernel(runtime);
        using var input = runtime.AllocateBytes((nuint)PtxFusedConv2DNchwK1Kernel.InputBytes);
        using var weights = runtime.AllocateBytes((nuint)PtxFusedConv2DNchwK1Kernel.WeightBytes);
        using var bias = runtime.AllocateBytes((nuint)PtxFusedConv2DNchwK1Kernel.BiasBytes);
        using var output = runtime.AllocateBytes((nuint)PtxFusedConv2DNchwK1Kernel.OutputBytes);
        input.Upload<float>(inputHost);
        weights.Upload<float>(weightHost);
        bias.Upload<float>(biasHost);
        DirectPtxTensorView inputView = DirectPtxTensorView.CreateOwned(input, kernel.Blueprint.Tensors[0]);
        DirectPtxTensorView weightView = DirectPtxTensorView.CreateOwned(weights, kernel.Blueprint.Tensors[1]);
        DirectPtxTensorView biasView = DirectPtxTensorView.CreateOwned(bias, kernel.Blueprint.Tensors[2]);
        DirectPtxTensorView outputView = DirectPtxTensorView.CreateOwned(output, kernel.Blueprint.Tensors[3]);
        void Launch() => kernel.Launch(inputView, weightView, biasView, outputView);
        Distribution distribution = Measure(runtime.Synchronize, Launch);
        long allocation = Allocation(runtime.Synchronize, Launch);
        Launch();
        runtime.Synchronize();
        var actual = new float[PtxFusedConv2DNchwK1Kernel.OutputElements];
        output.Download<float>(actual);
        return new Result(
            "Direct PTX fused (experimental)", distribution,
            Gflops(distribution.Median), allocation, 0, MaximumError(actual, expected),
            kernel.FunctionInfo.RegistersPerThread, kernel.FunctionInfo.StaticSharedBytes,
            kernel.FunctionInfo.LocalBytesPerThread, kernel.Audit.ActiveBlocksPerMultiprocessor);
    }

    private static Result RunEstablishedAiDotNet(
        float[] inputHost, float[] weightHost, float[] biasHost, float[] expected)
    {
        using var backend = new CudaBackend();
        using var input = backend.AllocateBuffer(inputHost);
        using var weights = backend.AllocateBuffer(weightHost);
        using var bias = backend.AllocateBuffer(biasHost);
        using var output = backend.AllocateBuffer(PtxFusedConv2DNchwK1Kernel.OutputElements);
        void Launch()
        {
            backend.Conv2D(input, weights, output, 1, 64, 16, 16, 64, 16, 16,
                1, 1, 1, 1, 0, 0, 1, 1);
            backend.Conv2DBiasAdd(output, bias, 1, 64, 16 * 16);
            backend.Relu(output, output, PtxFusedConv2DNchwK1Kernel.OutputElements);
        }
        Distribution distribution = Measure(backend.Synchronize, Launch);
        long allocation = Allocation(backend.Synchronize, Launch);
        Launch();
        backend.Synchronize();
        return new Result(
            "AiDotNet established CUDA/cuDNN", distribution,
            Gflops(distribution.Median), allocation, -1,
            MaximumError(backend.DownloadBuffer(output), expected),
            -1, -1, -1, -1);
    }

    private static Distribution Measure(Action synchronize, Action launch)
    {
        for (int warmup = 0; warmup < 30; warmup++) launch();
        synchronize();
        var samples = new double[101];
        for (int sample = 0; sample < samples.Length; sample++)
        {
            long start = Stopwatch.GetTimestamp();
            launch();
            synchronize();
            samples[sample] = Stopwatch.GetElapsedTime(start).TotalMilliseconds;
        }
        Array.Sort(samples);
        return new Distribution(samples.Average(), Percentile(samples, 0.5),
            Percentile(samples, 0.95), Percentile(samples, 0.99));
    }

    private static long Allocation(Action synchronize, Action launch)
    {
        launch();
        synchronize();
        long before = GC.GetAllocatedBytesForCurrentThread();
        for (int iteration = 0; iteration < 50; iteration++)
        {
            launch();
            synchronize();
        }
        return (GC.GetAllocatedBytesForCurrentThread() - before) / 50;
    }

    private static double Percentile(double[] sorted, double percentile)
    {
        double position = (sorted.Length - 1) * percentile;
        int lower = (int)position;
        int upper = Math.Min(lower + 1, sorted.Length - 1);
        return sorted[lower] + (sorted[upper] - sorted[lower]) * (position - lower);
    }

    private static double Gflops(double milliseconds)
    {
        const long operations = 2L * PtxFusedConv2DNchwK1Kernel.OutputElements *
            PtxFusedConv2DNchwK1Kernel.InputChannels;
        return operations / milliseconds / 1e6;
    }

    private static float[] Values(long length, int seed)
    {
        var random = new Random(seed);
        return Enumerable.Range(0, checked((int)length))
            .Select(_ => (random.NextSingle() - 0.5f) * 0.25f).ToArray();
    }

    private static float[] Oracle(float[] input, float[] weights, float[] bias)
    {
        var output = new float[PtxFusedConv2DNchwK1Kernel.OutputElements];
        for (int outputChannel = 0; outputChannel < 64; outputChannel++)
        for (int spatial = 0; spatial < 16 * 16; spatial++)
        {
            double sum = bias[outputChannel];
            for (int inputChannel = 0; inputChannel < 64; inputChannel++)
                sum += input[inputChannel * 16 * 16 + spatial] *
                    weights[outputChannel * 64 + inputChannel];
            output[outputChannel * 16 * 16 + spatial] = MathF.Max(0, (float)sum);
        }
        return output;
    }

    private static float MaximumError(float[] actual, float[] expected)
    {
        if (actual.Length != expected.Length)
            throw new InvalidOperationException("Convolution output extent mismatch.");
        float maximum = 0;
        for (int index = 0; index < actual.Length; index++)
            maximum = MathF.Max(maximum, MathF.Abs(actual[index] - expected[index]));
        return maximum;
    }

    private static string Format(int value) => value < 0 ? "n/a" : value.ToString();
    private static string Format(long value) => value < 0 ? "n/a" : value.ToString();

    private static void RunPyTorchCompetitors()
    {
        string script = Path.Combine(AppContext.BaseDirectory, "BaselineRunners", "py",
            "run_direct_ptx_convolution_competitors.py");
        if (!File.Exists(script))
            throw new FileNotFoundException("PyTorch convolution competitor runner is missing.", script);
        var start = new ProcessStartInfo
        {
            FileName = Environment.GetEnvironmentVariable("PYTHON") ?? "python",
            Arguments = "\"" + script.Replace("\"", "\\\"") + "\"",
            UseShellExecute = false
        };
        using Process process = Process.Start(start) ??
            throw new InvalidOperationException("Could not start PyTorch convolution competitors.");
        if (!process.WaitForExit((int)TimeSpan.FromMinutes(30).TotalMilliseconds))
        {
            process.Kill(entireProcessTree: true);
            throw new TimeoutException("PyTorch convolution competitors exceeded 30 minutes.");
        }
        if (process.ExitCode != 0)
            throw new InvalidOperationException(
                $"PyTorch convolution competitors exited with code {process.ExitCode}.");
    }
}
