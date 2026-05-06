#if NET8_0_OR_GREATER
using System.Diagnostics;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using TorchSharp;
using TorchTensor = TorchSharp.torch.Tensor;

namespace AiDotNet.Tensors.Benchmarks;

public static class Issue304GemvBenchmark
{
    private const int Rows = 20_000;
    private const int Cols = 128;
    private const int QueryVariants = 8;
    private const int WarmupRuns = 12;
    private const int TimedRuns = 80;
    private const int ChunkSize = 256;

    private static volatile float _sink;

    public static void Run()
    {
        var engine = new CpuEngine();
        AiDotNetEngine.Current = engine;

        Console.WriteLine("Issue #304 GEMV and compiled-cache benchmark");
        Console.WriteLine($"Shape: [{Rows:N0},{Cols}] x [{Cols},1]");
        Console.WriteLine($"CPU threads: {Environment.ProcessorCount}");
        Console.WriteLine(TensorPrimitivesCore.GetHardwareAccelerationInfo());
        Console.WriteLine();

        var weightsData = CreateData(Rows * Cols, seedOffset: 304);
        var queriesData = CreateQueries();
        var expected = new float[Rows];
        var directOutput = new float[Rows];
        var weights = new Tensor<float>(weightsData, new[] { Rows, Cols });
        var queries = new Tensor<float>[QueryVariants];
        for (int i = 0; i < QueryVariants; i++)
            queries[i] = new Tensor<float>(queriesData[i], new[] { Cols, 1 });

        DirectGemv(weightsData, queriesData[0], expected);
        VerifyCorrectness(engine, weights, queries, expected);

        using var shapeCache = new CompiledModelCache<float>();
        var shapeTraceQuery = new Tensor<float>((float[])queriesData[0].Clone(), new[] { Cols, 1 });
        var shapePlan = shapeCache.GetOrCompileInference(shapeTraceQuery._shape, () => weights.MatrixMultiply(shapeTraceQuery));
        var shapeInputs = new Tensor<float>[1];
        shapePlan.SetInputs(new[] { queries[0] });
        _sink = shapePlan.Execute().AsSpan()[0];

        using var tensorCache = new CompiledModelCache<float>();
        var tensorTraceQuery = new Tensor<float>((float[])queriesData[0].Clone(), new[] { Cols, 1 });
        var tensorPlan = tensorCache.GetOrCompileInference(tensorTraceQuery, () => weights.MatrixMultiply(tensorTraceQuery));
        _sink = tensorPlan.Execute().AsSpan()[0];

        Console.WriteLine($"{"Method",-42} {"Mean ms",10} {"Median ms",10} {"Min ms",10} {"Max ms",10} {"GFLOP/s",10} {"Checksum",12}");
        Console.WriteLine(new string('-', 112));

        var directStats = Measure(run =>
        {
            var query = queriesData[run % QueryVariants];
            DirectGemv(weightsData, query, directOutput);
            return directOutput[run % Rows];
        });
        Print("Direct fused row Dot", directStats);

        var eagerStats = Measure(run =>
        {
            var result = engine.TensorMatMul(weights, queries[run % QueryVariants]);
            return result.AsSpan()[run % Rows];
        });
        Print("AiDotNet eager TensorMatMul", eagerStats);

        var shapePlanStats = Measure(run =>
        {
            shapeInputs[0] = queries[run % QueryVariants];
            shapePlan.SetInputs(shapeInputs);
            var output = shapePlan.Execute();
            return output.AsSpan()[run % Rows];
        });
        Print("AiDotNet compiled SetInputs+Execute", shapePlanStats);

        var cacheHitStats = Measure(run =>
        {
            var input = queries[run % QueryVariants];
            var plan = tensorCache.GetOrCompileInference(input, () => weights.MatrixMultiply(input));
            var output = plan.Execute();
            return output.AsSpan()[run % Rows];
        });
        Print("AiDotNet cache hit+Execute", cacheHitStats);

        TryRunTorchSharp(weightsData, queriesData);

        Console.WriteLine();
        Console.WriteLine($"Sink: {_sink:F6}");
    }

    private static void VerifyCorrectness(CpuEngine engine, Tensor<float> weights, Tensor<float>[] queries, float[] expected)
    {
        var eager = engine.TensorMatMul(weights, queries[0]);
        AssertClose("eager TensorMatMul", eager.AsSpan(), expected);

        using var cache = new CompiledModelCache<float>();
        var tensorTraceQuery = new Tensor<float>((float[])queries[0].GetDataArray().Clone(), new[] { Cols, 1 });
        var plan = cache.GetOrCompileInference(tensorTraceQuery, () => weights.MatrixMultiply(tensorTraceQuery));
        AssertClose("compiled first execute", plan.Execute().AsSpan(), expected);

        var expectedSecond = new float[Rows];
        DirectGemv(weights.GetDataArray(), queries[1].GetDataArray(), expectedSecond);
        var plan2 = cache.GetOrCompileInference(queries[1], () => weights.MatrixMultiply(queries[1]));
        AssertClose("compiled cache hit execute", plan2.Execute().AsSpan(), expectedSecond);

        if (!ReferenceEquals(plan, plan2))
            throw new InvalidOperationException("CompiledModelCache recompiled for same-shape query instead of returning the cached plan.");

        using var shapeCache = new CompiledModelCache<float>();
        var shapeTraceQuery = new Tensor<float>((float[])queries[0].GetDataArray().Clone(), new[] { Cols, 1 });
        var shapePlan = shapeCache.GetOrCompileInference(shapeTraceQuery._shape, () => weights.MatrixMultiply(shapeTraceQuery));
        var shapeInputs = new Tensor<float>[1];
        for (int i = 0; i < queries.Length; i++)
        {
            DirectGemv(weights.GetDataArray(), queries[i].GetDataArray(), expectedSecond);
            shapeInputs[0] = queries[i];
            shapePlan.SetInputs(shapeInputs);
            AssertClose($"shape-overload compiled execute variant {i}", shapePlan.Execute().AsSpan(), expectedSecond);
        }

        Console.WriteLine("Correctness: eager, tensor-overload cache hit, and shape-overload SetInputs match direct fused row Dot.");
        Console.WriteLine();
    }

    private static void TryRunTorchSharp(float[] weightsData, float[][] queriesData)
    {
        try
        {
            torch.set_grad_enabled(false);
            var device = torch.CPU;
            using var torchWeights = torch.tensor(weightsData, new long[] { Rows, Cols }, device: device);
            var torchQueries = new TorchTensor[QueryVariants];
            try
            {
                for (int i = 0; i < QueryVariants; i++)
                    torchQueries[i] = torch.tensor(queriesData[i], new long[] { Cols, 1 }, device: device);

                using (var warmup = torch.matmul(torchWeights, torchQueries[0]))
                    _sink = 0.5f + _sink;

                var torchStats = Measure(run =>
                {
                    using var result = torch.matmul(torchWeights, torchQueries[run % QueryVariants]);
                    GC.KeepAlive(result);
                    return run;
                });
                Print("TorchSharp CPU torch.matmul", torchStats, printChecksum: false);
            }
            finally
            {
                for (int i = 0; i < torchQueries.Length; i++)
                    torchQueries[i]?.Dispose();
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"TorchSharp CPU torch.matmul skipped: {ex.GetType().Name}: {ex.Message}");
        }
    }

    private static Stats Measure(Func<int, float> action)
    {
        for (int i = 0; i < WarmupRuns; i++)
            _sink = action(i);

        GC.Collect();
        GC.WaitForPendingFinalizers();
        GC.Collect();

        var timings = new double[TimedRuns];
        double checksum = 0;
        for (int i = 0; i < TimedRuns; i++)
        {
            long start = Stopwatch.GetTimestamp();
            float value = action(i);
            long stop = Stopwatch.GetTimestamp();
            timings[i] = (stop - start) * 1000.0 / Stopwatch.Frequency;
            checksum += value;
            _sink = value;
        }

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

    private static void Print(string method, Stats stats, bool printChecksum = true)
    {
        double gflops = (2.0 * Rows * Cols) / (stats.MeanMs / 1000.0) / 1_000_000_000.0;
        string checksum = printChecksum ? stats.Checksum.ToString("F4") : "n/a";
        Console.WriteLine($"{method,-42} {stats.MeanMs,10:F4} {stats.MedianMs,10:F4} {stats.MinMs,10:F4} {stats.MaxMs,10:F4} {gflops,10:F2} {checksum,12}");
    }

    private static void DirectGemv(float[] matrix, float[] vector, float[] output)
    {
        int chunks = (Rows + ChunkSize - 1) / ChunkSize;
        Parallel.For(0, chunks, chunk =>
        {
            int start = chunk * ChunkSize;
            int end = Math.Min(start + ChunkSize, Rows);
            for (int row = start; row < end; row++)
            {
                output[row] = TensorPrimitivesCore.Dot(
                    matrix.AsSpan(row * Cols, Cols),
                    vector.AsSpan(0, Cols));
            }
        });
    }

    private static float[] CreateData(int length, int seedOffset)
    {
        var data = new float[length];
        for (int i = 0; i < length; i++)
            data[i] = DeterministicValue(i + seedOffset);
        return data;
    }

    private static float[][] CreateQueries()
    {
        var queries = new float[QueryVariants][];
        for (int i = 0; i < QueryVariants; i++)
            queries[i] = CreateData(Cols, 10_000 + i * 97);
        return queries;
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

    private readonly record struct Stats(double MeanMs, double MedianMs, double MinMs, double MaxMs, double Checksum);
}
#endif
