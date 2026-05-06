#if NET8_0_OR_GREATER
using System.Diagnostics;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Benchmarks;

public static class Issue304GemvBenchmark
{
    private static readonly GemvShape[] Shapes =
    [
        new(16, 16),
        new(32, 16),
        new(64, 16),
        new(128, 32),
        new(256, 32),
        new(512, 64),
        new(1_024, 64),
        new(4_096, 128),
        new(20_000, 128),
    ];

    private const int QueryVariants = 8;
    private const int WarmupRuns = 12;
    private const int TimedRuns = 80;
    private const int ChunkSize = 256;
    private const int GemvParallelThreshold = 128 * 1024;

    private static volatile float _sink;

    public static void Run()
    {
        var cpuEngine = new CpuEngine();
        AiDotNetEngine.Current = cpuEngine;

        Console.WriteLine("Issue #304 multi-shape GEMV benchmark");
        Console.WriteLine("Shapes: [rows,cols] x [cols,1]");
        Console.WriteLine($"CPU threads: {Environment.ProcessorCount}");
        Console.WriteLine(TensorPrimitivesCore.GetHardwareAccelerationInfo());
        Console.WriteLine();

        RunCpuBenchmarks(cpuEngine);
        RunGpuBenchmarks();
        RunRawPyTorchBenchmarks();

        Console.WriteLine();
        Console.WriteLine($"Sink: {_sink:F6}");
    }

    private static void RunCpuBenchmarks(CpuEngine cpuEngine)
    {
        Console.WriteLine("AiDotNet CPU");
        PrintHeader();

        foreach (var shape in Shapes)
        {
            var data = CreateDataSet(shape);
            VerifyCorrectness(cpuEngine, shape, data);

            var directStats = Measure(shape, run =>
            {
                var query = data.QueriesData[run % QueryVariants];
                DirectGemv(data.WeightsData, query, data.DirectOutput, shape.Rows, shape.Cols);
                return data.DirectOutput[run % shape.Rows];
            });
            Print(shape, "Direct fused row Dot", directStats);

            var eagerStats = Measure(shape, run =>
            {
                var result = cpuEngine.TensorMatMul(data.Weights, data.Queries[run % QueryVariants]);
                return result.AsSpan()[run % shape.Rows];
            });
            Print(shape, "AiDotNet CPU eager", eagerStats);

            using var shapeCache = new CompiledModelCache<float>();
            var shapeTraceQuery = new Tensor<float>((float[])data.QueriesData[0].Clone(), new[] { shape.Cols, 1 });
            var shapePlan = shapeCache.GetOrCompileInference(
                shapeTraceQuery._shape,
                () => data.Weights.MatrixMultiply(shapeTraceQuery));
            var shapeInputs = new Tensor<float>[1];
            var shapePlanStats = Measure(shape, run =>
            {
                shapeInputs[0] = data.Queries[run % QueryVariants];
                shapePlan.SetInputs(shapeInputs);
                var output = shapePlan.Execute();
                return output.AsSpan()[run % shape.Rows];
            });
            Print(shape, "AiDotNet CPU compiled SetInputs", shapePlanStats);

            using var tensorCache = new CompiledModelCache<float>();
            var tensorTraceQuery = new Tensor<float>((float[])data.QueriesData[0].Clone(), new[] { shape.Cols, 1 });
            _ = tensorCache.GetOrCompileInference(
                tensorTraceQuery,
                () => data.Weights.MatrixMultiply(tensorTraceQuery)).Execute();
            var cacheHitStats = Measure(shape, run =>
            {
                var input = data.Queries[run % QueryVariants];
                var plan = tensorCache.GetOrCompileInference(input, () => data.Weights.MatrixMultiply(input));
                var output = plan.Execute();
                return output.AsSpan()[run % shape.Rows];
            });
            Print(shape, "AiDotNet CPU cache hit", cacheHitStats);

            Console.WriteLine();
        }
    }

    private static void RunGpuBenchmarks()
    {
        using var gpuEngine = TryCreateGpuEngine();
        if (gpuEngine is null)
        {
            Console.WriteLine("AiDotNet GPU skipped: DirectGpuTensorEngine is not available.");
            Console.WriteLine();
            return;
        }

        Console.WriteLine("AiDotNet GPU");
        Console.WriteLine($"Engine: {gpuEngine.Name}");
        PrintHeader();

        foreach (var shape in Shapes)
        {
            var data = CreateDataSet(shape);
            DirectGemv(data.WeightsData, data.QueriesData[0], data.Expected, shape.Rows, shape.Cols);

            var gpuCheck = gpuEngine.TensorMatMul(data.Weights, data.Queries[0]);
            AssertClose("GPU TensorMatMul", gpuCheck.AsSpan(), data.Expected);

            var publicStats = Measure(shape, run =>
            {
                var result = gpuEngine.TensorMatMul(data.Weights, data.Queries[run % QueryVariants]);
                return result.AsSpan()[run % shape.Rows];
            }, GetGpuPublicInnerIterations(shape));
            Print(shape, "AiDotNet GPU TensorMatMul+read", publicStats);

            var backend = gpuEngine.GetBackend();
            if (backend is not null)
            {
                using var weightsBuffer = backend.AllocateBuffer(data.WeightsData);
                using var outputBuffer = backend.AllocateBuffer(shape.Rows);
                var queryBuffers = new IGpuBuffer[QueryVariants];
                try
                {
                    for (int i = 0; i < QueryVariants; i++)
                        queryBuffers[i] = backend.AllocateBuffer(data.QueriesData[i]);

                    backend.Gemm(weightsBuffer, queryBuffers[0], outputBuffer, shape.Rows, 1, shape.Cols);
                    backend.Synchronize();
                    var backendCheck = backend.DownloadBuffer(outputBuffer);
                    AssertClose("GPU backend Gemm", backendCheck, data.Expected);

                    var deviceStats = MeasureNoResult(shape, run =>
                    {
                        var queryBuffer = queryBuffers[run % QueryVariants];
                        backend.Gemm(weightsBuffer, queryBuffer, outputBuffer, shape.Rows, 1, shape.Cols);
                        backend.Synchronize();
                    }, GetGpuDeviceInnerIterations(shape));
                    Print(shape, "AiDotNet GPU backend Gemm sync", deviceStats, printChecksum: false);
                }
                finally
                {
                    for (int i = 0; i < queryBuffers.Length; i++)
                        queryBuffers[i]?.Dispose();
                }
            }

            Console.WriteLine();
        }
    }

    private static DirectGpuTensorEngine? TryCreateGpuEngine()
    {
        try
        {
            var engine = new DirectGpuTensorEngine();
            if (engine.IsGpuAvailable)
                return engine;

            engine.Dispose();
            return null;
        }
        catch (Exception ex)
        {
            Console.WriteLine($"AiDotNet GPU initialization failed: {ex.GetType().Name}: {ex.Message}");
            return null;
        }
    }

    private static void RunRawPyTorchBenchmarks()
    {
        Console.WriteLine("Raw PyTorch baseline");

        string scriptPath = Path.Combine(AppContext.BaseDirectory, "BaselineRunners", "py", "issue304_pytorch_gemv.py");
        if (!File.Exists(scriptPath))
        {
            Console.WriteLine($"PyTorch baseline skipped: script not found at {scriptPath}");
            Console.WriteLine();
            return;
        }

        string sizes = string.Join(";", Shapes.Select(shape => $"{shape.Rows}x{shape.Cols}"));
        var pythonExecutable = Environment.GetEnvironmentVariable("AIDOTNET_BENCHMARK_PYTHON");
        if (string.IsNullOrWhiteSpace(pythonExecutable))
            pythonExecutable = "python";

        Console.WriteLine($"Python: {pythonExecutable}");

        var startInfo = new ProcessStartInfo(pythonExecutable)
        {
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false,
            CreateNoWindow = true
        };
        startInfo.ArgumentList.Add(scriptPath);
        startInfo.ArgumentList.Add("--sizes");
        startInfo.ArgumentList.Add(sizes);
        startInfo.ArgumentList.Add("--runs");
        startInfo.ArgumentList.Add(TimedRuns.ToString());
        startInfo.ArgumentList.Add("--warmup");
        startInfo.ArgumentList.Add(WarmupRuns.ToString());
        startInfo.ArgumentList.Add("--device");
        startInfo.ArgumentList.Add("all");

        using var process = Process.Start(startInfo);
        if (process is null)
        {
            Console.WriteLine("PyTorch baseline skipped: failed to start python.");
            Console.WriteLine();
            return;
        }

        string stdout = process.StandardOutput.ReadToEnd();
        string stderr = process.StandardError.ReadToEnd();
        if (!process.WaitForExit(180_000))
        {
            process.Kill();
            Console.WriteLine("PyTorch baseline timed out.");
            Console.WriteLine();
            return;
        }

        if (!string.IsNullOrWhiteSpace(stdout))
            Console.Write(stdout);
        if (process.ExitCode != 0 || !string.IsNullOrWhiteSpace(stderr))
            Console.WriteLine(stderr.Trim());

        Console.WriteLine();
    }

    private static void VerifyCorrectness(CpuEngine engine, GemvShape shape, DataSet data)
    {
        DirectGemv(data.WeightsData, data.QueriesData[0], data.Expected, shape.Rows, shape.Cols);

        var eager = engine.TensorMatMul(data.Weights, data.Queries[0]);
        AssertClose("eager TensorMatMul", eager.AsSpan(), data.Expected);

        using var cache = new CompiledModelCache<float>();
        var tensorTraceQuery = new Tensor<float>((float[])data.Queries[0].GetDataArray().Clone(), new[] { shape.Cols, 1 });
        var plan = cache.GetOrCompileInference(tensorTraceQuery, () => data.Weights.MatrixMultiply(tensorTraceQuery));
        AssertClose("compiled first execute", plan.Execute().AsSpan(), data.Expected);

        var expectedSecond = new float[shape.Rows];
        DirectGemv(data.Weights.GetDataArray(), data.Queries[1].GetDataArray(), expectedSecond, shape.Rows, shape.Cols);
        var plan2 = cache.GetOrCompileInference(data.Queries[1], () => data.Weights.MatrixMultiply(data.Queries[1]));
        AssertClose("compiled cache hit execute", plan2.Execute().AsSpan(), expectedSecond);

        if (!ReferenceEquals(plan, plan2))
            throw new InvalidOperationException("CompiledModelCache recompiled for same-shape query instead of returning the cached plan.");
    }

    private static Stats Measure(GemvShape shape, Func<int, float> action, int? innerIterationsOverride = null)
    {
        int innerIterations = innerIterationsOverride ?? GetCpuInnerIterations(shape);
        for (int i = 0; i < WarmupRuns; i++)
        {
            for (int j = 0; j < innerIterations; j++)
                _sink = action(i + j);
        }

        GC.Collect();
        GC.WaitForPendingFinalizers();
        GC.Collect();

        var timings = new double[TimedRuns];
        double checksum = 0;
        for (int i = 0; i < TimedRuns; i++)
        {
            float value = 0;
            long start = Stopwatch.GetTimestamp();
            for (int j = 0; j < innerIterations; j++)
                value = action(i + j);
            long stop = Stopwatch.GetTimestamp();

            timings[i] = (stop - start) * 1000.0 / Stopwatch.Frequency / innerIterations;
            checksum += value;
            _sink = value;
        }

        return Summarize(timings, checksum);
    }

    private static Stats MeasureNoResult(GemvShape shape, Action<int> action, int? innerIterationsOverride = null)
    {
        int innerIterations = innerIterationsOverride ?? GetCpuInnerIterations(shape);
        for (int i = 0; i < WarmupRuns; i++)
        {
            for (int j = 0; j < innerIterations; j++)
                action(i + j);
        }

        var timings = new double[TimedRuns];
        for (int i = 0; i < TimedRuns; i++)
        {
            long start = Stopwatch.GetTimestamp();
            for (int j = 0; j < innerIterations; j++)
                action(i + j);
            long stop = Stopwatch.GetTimestamp();

            timings[i] = (stop - start) * 1000.0 / Stopwatch.Frequency / innerIterations;
        }

        return Summarize(timings, checksum: 0);
    }

    private static Stats Summarize(double[] timings, double checksum)
    {
        Array.Sort(timings);
        double sum = 0;
        foreach (double timing in timings)
            sum += timing;

        return new Stats(
            MeanMs: sum / timings.Length,
            MedianMs: timings[timings.Length / 2],
            MinMs: timings[0],
            MaxMs: timings[^1],
            Checksum: checksum);
    }

    private static int GetCpuInnerIterations(GemvShape shape)
    {
        long ops = (long)shape.Rows * shape.Cols;
        if (ops <= 1_024) return 1_000;
        if (ops <= 8_192) return 300;
        if (ops <= 65_536) return 100;
        if (ops <= 524_288) return 20;
        return 1;
    }

    private static int GetGpuPublicInnerIterations(GemvShape shape)
    {
        long ops = (long)shape.Rows * shape.Cols;
        if (ops <= 1_024) return 20;
        if (ops <= 8_192) return 10;
        if (ops <= 65_536) return 5;
        return 1;
    }

    private static int GetGpuDeviceInnerIterations(GemvShape shape)
    {
        long ops = (long)shape.Rows * shape.Cols;
        if (ops <= 1_024) return 200;
        if (ops <= 8_192) return 100;
        if (ops <= 65_536) return 50;
        if (ops <= 524_288) return 10;
        return 1;
    }

    private static void PrintHeader()
    {
        Console.WriteLine($"{"Shape",-18} {"Method",-36} {"Mean ms",10} {"Median ms",10} {"Min ms",10} {"Max ms",10} {"GFLOP/s",10} {"Checksum",12}");
        Console.WriteLine(new string('-', 124));
    }

    private static void Print(GemvShape shape, string method, Stats stats, bool printChecksum = true)
    {
        double gflops = (2.0 * shape.Rows * shape.Cols) / (stats.MeanMs / 1000.0) / 1_000_000_000.0;
        string checksum = printChecksum ? stats.Checksum.ToString("F4") : "n/a";
        Console.WriteLine($"{shape,-18} {method,-36} {stats.MeanMs,10:F5} {stats.MedianMs,10:F5} {stats.MinMs,10:F5} {stats.MaxMs,10:F5} {gflops,10:F2} {checksum,12}");
    }

    private static void DirectGemv(float[] matrix, float[] vector, float[] output, int rows, int cols)
    {
        if ((long)rows * cols < GemvParallelThreshold)
        {
            for (int row = 0; row < rows; row++)
            {
                output[row] = TensorPrimitivesCore.Dot(
                    matrix.AsSpan(row * cols, cols),
                    vector.AsSpan(0, cols));
            }

            return;
        }

        int chunks = (rows + ChunkSize - 1) / ChunkSize;
        Parallel.For(0, chunks, chunk =>
        {
            int start = chunk * ChunkSize;
            int end = Math.Min(start + ChunkSize, rows);
            for (int row = start; row < end; row++)
            {
                output[row] = TensorPrimitivesCore.Dot(
                    matrix.AsSpan(row * cols, cols),
                    vector.AsSpan(0, cols));
            }
        });
    }

    private static DataSet CreateDataSet(GemvShape shape)
    {
        var weightsData = CreateData(shape.Rows * shape.Cols, seedOffset: 304 + shape.Rows + shape.Cols);
        var queriesData = new float[QueryVariants][];
        var queries = new Tensor<float>[QueryVariants];
        for (int i = 0; i < QueryVariants; i++)
        {
            queriesData[i] = CreateData(shape.Cols, seedOffset: 10_000 + shape.Rows + i * 97);
            queries[i] = new Tensor<float>(queriesData[i], new[] { shape.Cols, 1 });
        }

        return new DataSet(
            WeightsData: weightsData,
            QueriesData: queriesData,
            Weights: new Tensor<float>(weightsData, new[] { shape.Rows, shape.Cols }),
            Queries: queries,
            Expected: new float[shape.Rows],
            DirectOutput: new float[shape.Rows]);
    }

    private static float[] CreateData(int length, int seedOffset)
    {
        var data = new float[length];
        for (int i = 0; i < length; i++)
            data[i] = DeterministicValue(i + seedOffset);
        return data;
    }

    private static float DeterministicValue(int value)
    {
        unchecked
        {
            uint x = (uint)value * 747796405u + 2891336453u;
            x = ((x >> ((int)(x >> 28) + 4)) ^ x) * 277803737u;
            x = (x >> 22) ^ x;
            return ((x & 0xFFFF) / 65535f) - 0.5f;
        }
    }

    private static void AssertClose(string name, ReadOnlySpan<float> actual, ReadOnlySpan<float> expected)
    {
        if (actual.Length != expected.Length)
            throw new InvalidOperationException($"{name} length mismatch: {actual.Length} != {expected.Length}.");

        for (int i = 0; i < actual.Length; i++)
        {
            float tolerance = 1e-4f + Math.Abs(expected[i]) * 1e-4f;
            if (Math.Abs(actual[i] - expected[i]) > tolerance)
                throw new InvalidOperationException($"{name} mismatch at {i}: actual={actual[i]}, expected={expected[i]}.");
        }
    }

    private readonly record struct GemvShape(int Rows, int Cols)
    {
        public override string ToString() => $"[{Rows},{Cols}]x[{Cols},1]";
    }

    private sealed record DataSet(
        float[] WeightsData,
        float[][] QueriesData,
        Tensor<float> Weights,
        Tensor<float>[] Queries,
        float[] Expected,
        float[] DirectOutput);

    private readonly record struct Stats(double MeanMs, double MedianMs, double MinMs, double MaxMs, double Checksum);
}
#endif
