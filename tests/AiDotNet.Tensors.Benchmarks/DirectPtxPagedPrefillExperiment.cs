using System.Diagnostics;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

namespace AiDotNet.Tensors.Benchmarks;

/// <summary>Issue #834 resident paged-prefill championship matrix.</summary>
internal static class DirectPtxPagedPrefillExperiment
{
    private const int Dimension = 64;
    private const float Scale = 0.125f;
    private const int Warmups = 30;
    private const int Samples = 101;

    private readonly record struct Shape(
        string Name, int QueryHeads, int KeyValueHeads,
        int Queries, int Start, int BlockSize, int PoolBlocks);
    private readonly record struct Distribution(double Mean, double Median, double P95, double P99);
    private readonly record struct CellEvidence(
        int Run, Shape Shape, Distribution Direct, Distribution Current,
        long DirectBytes, long CurrentBytes, float DirectError, float CurrentError,
        DirectPtxKernelAudit Audit);

    private static readonly Shape[] Shapes =
    [
        new("prefill-mha", 8, 8, 4, 12, 16, 4),
        new("prefill-gqa", 8, 2, 8, 24, 16, 4),
        new("prefill-mqa", 8, 1, 16, 48, 16, 6),
        new("prefill-long", 8, 2, 16, 112, 16, 10)
    ];

    internal static void Run(int independentRuns = 3)
    {
        if (independentRuns <= 0) throw new ArgumentOutOfRangeException(nameof(independentRuns));
        Console.WriteLine(
            $"Direct-PTX paged prefill: {independentRuns} independent run(s), " +
            $"{Warmups} warmups + {Samples} synchronized E2E samples/cell");
        Console.WriteLine(
            $"{"Run",3} {"Shape",-12} {"Method",-27} {"median us",10} {"p95 us",10} " +
            $"{"p99 us",10} {"mean us",10} {"GFLOPS",9} {"B/call",9} {"tmp MiB",9} " +
            $"{"max err",10} {"regs",5} {"shared",7} {"local",5} {"occ",4}");
        Console.WriteLine(new string('-', 156));
        var evidence = new List<CellEvidence>(independentRuns * Shapes.Length);
        for (int run = 1; run <= independentRuns; run++)
        {
            using var backend = new CudaBackend();
            if (!backend.IsDirectPtxPagedPrefillEnabled)
                throw new InvalidOperationException(
                    "Set AIDOTNET_DIRECT_PTX_PAGED_PREFILL=1 on an Ampere GPU.");
            if (run == 1) Console.WriteLine($"GPU: {backend.DeviceName}");
            foreach (Shape shape in Shapes) evidence.Add(RunCell(backend, run, shape));
        }
        PrintReleaseGate(evidence, independentRuns);
    }

    private static CellEvidence RunCell(CudaBackend backend, int run, Shape shape)
    {
        int queryElements = shape.Queries * shape.QueryHeads * Dimension;
        int keyValueElements = shape.PoolBlocks * shape.BlockSize * shape.KeyValueHeads * Dimension;
        int maximumKeyLength = shape.Start + shape.Queries;
        int logicalBlocks = (maximumKeyLength + shape.BlockSize - 1) / shape.BlockSize;
        var random = new Random(20261000 + run * 10_000 + keyValueElements + queryElements);
        float[] qHost = Values(random, queryElements);
        float[] kHost = Values(random, keyValueElements);
        float[] vHost = Values(random, keyValueElements);
        int[] tableHost = Enumerable.Range(0, logicalBlocks)
            .Select(i => shape.PoolBlocks - 1 - i).ToArray();
        float[] expected = Oracle(qHost, kHost, vHost, tableHost, shape);

        using var q = backend.AllocateBuffer(qHost);
        using var k = backend.AllocateBuffer(kHost);
        using var v = backend.AllocateBuffer(vHost);
        using var table = backend.AllocateIntBuffer(tableHost);
        using var directOutput = backend.AllocateBuffer(queryElements);
        using var currentOutput = backend.AllocateBuffer(queryElements);

        void DirectLaunch()
        {
            if (!backend.TryDirectPtxPagedPrefillD64(
                q, k, v, table, directOutput,
                shape.QueryHeads, shape.KeyValueHeads,
                shape.Queries, shape.Start, shape.BlockSize, Scale))
                throw new InvalidOperationException(backend.DirectPtxLastError);
        }

        void CurrentLaunch() => backend.PagedAttentionPrefillCurrentInto(
            q, k, v, table, currentOutput,
            shape.QueryHeads, shape.KeyValueHeads, Dimension,
            shape.BlockSize, shape.Queries, shape.Start, Scale);

        DirectLaunch(); CurrentLaunch(); backend.Synchronize();
        float directError = MaximumError(backend.DownloadBuffer(directOutput), expected);
        float currentError = MaximumError(backend.DownloadBuffer(currentOutput), expected);
        Distribution direct = Measure(backend, DirectLaunch);
        long directBytes = MeasureAllocation(backend, DirectLaunch);
        if (!backend.TryGetDirectPtxPagedPrefillAudit(
            shape.QueryHeads, shape.KeyValueHeads, shape.Queries, shape.Start,
            shape.BlockSize, shape.PoolBlocks, Scale, out DirectPtxKernelAudit audit))
            throw new InvalidOperationException("No audit for measured paged-prefill module.");
        Print(run, shape, "Direct PTX register-online", direct,
            Gflops(shape, direct.Median), directBytes, directError,
            audit.Function.RegistersPerThread, audit.Function.StaticSharedBytes,
            audit.Function.LocalBytesPerThread, audit.ActiveBlocksPerMultiprocessor);

        Distribution current = Measure(backend, CurrentLaunch);
        long currentBytes = MeasureAllocation(backend, CurrentLaunch);
        Print(run, shape, "AiDotNet paged NVRTC", current,
            Gflops(shape, current.Median), currentBytes, currentError,
            -1, -1, -1, -1);
        return new CellEvidence(
            run, shape, direct, current, directBytes, currentBytes,
            directError, currentError, audit);
    }

    private static Distribution Measure(CudaBackend backend, Action action)
    {
        for (int i = 0; i < Warmups; i++) action();
        backend.Synchronize();
        var samples = new double[Samples];
        double tickToMicroseconds = 1_000_000.0 / Stopwatch.Frequency;
        for (int i = 0; i < samples.Length; i++)
        {
            long start = Stopwatch.GetTimestamp();
            action(); backend.Synchronize();
            samples[i] = (Stopwatch.GetTimestamp() - start) * tickToMicroseconds;
        }
        Array.Sort(samples);
        return new(samples.Average(), Percentile(samples, 0.50),
            Percentile(samples, 0.95), Percentile(samples, 0.99));
    }

    private static long MeasureAllocation(CudaBackend backend, Action action)
    {
        for (int i = 0; i < 8; i++) action();
        backend.Synchronize();
        long before = GC.GetAllocatedBytesForCurrentThread();
        for (int i = 0; i < Samples; i++) action();
        long result = (GC.GetAllocatedBytesForCurrentThread() - before) / Samples;
        backend.Synchronize();
        return result;
    }

    private static float[] Oracle(
        float[] query, float[] key, float[] value, int[] table, Shape shape)
    {
        var output = new float[query.Length];
        var logits = new float[shape.Start + shape.Queries];
        int ratio = shape.QueryHeads / shape.KeyValueHeads;
        for (int queryIndex = 0; queryIndex < shape.Queries; queryIndex++)
        for (int head = 0; head < shape.QueryHeads; head++)
        {
            int kvHead = head / ratio;
            int keyLength = shape.Start + queryIndex + 1;
            int queryBase = (queryIndex * shape.QueryHeads + head) * Dimension;
            float maximum = float.NegativeInfinity;
            for (int token = 0; token < keyLength; token++)
            {
                int physicalToken = table[token / shape.BlockSize] * shape.BlockSize + token % shape.BlockSize;
                int keyBase = (physicalToken * shape.KeyValueHeads + kvHead) * Dimension;
                float dot = 0;
                for (int d = 0; d < Dimension; d++) dot += query[queryBase + d] * key[keyBase + d];
                logits[token] = dot * Scale;
                maximum = MathF.Max(maximum, logits[token]);
            }
            float denominator = 0;
            for (int token = 0; token < keyLength; token++)
            {
                logits[token] = MathF.Exp(logits[token] - maximum);
                denominator += logits[token];
            }
            for (int token = 0; token < keyLength; token++)
            {
                int physicalToken = table[token / shape.BlockSize] * shape.BlockSize + token % shape.BlockSize;
                int valueBase = (physicalToken * shape.KeyValueHeads + kvHead) * Dimension;
                float probability = logits[token] / denominator;
                for (int d = 0; d < Dimension; d++)
                    output[queryBase + d] += probability * value[valueBase + d];
            }
        }
        return output;
    }

    private static void PrintReleaseGate(List<CellEvidence> evidence, int independentRuns)
    {
        Console.WriteLine();
        Console.WriteLine("Paired-run production gate against the established AiDotNet paged CUDA path:");
        DirectPtxReleaseGatePolicy policy = DirectPtxReleaseGatePolicy.ProductionDefault;
        foreach (Shape shape in Shapes)
        {
            CellEvidence[] cells = evidence.Where(cell => cell.Shape == shape).ToArray();
            var failures = new List<string>();
            double minimumSpeedup = double.PositiveInfinity;
            foreach (CellEvidence cell in cells)
            {
                var candidate = new DirectPtxPerformanceEvidence(
                    cell.Direct.Median, cell.Direct.P95, cell.DirectBytes, 0,
                    cell.DirectError, cell.Audit.Function.LocalBytesPerThread, independentRuns);
                var competitor = new DirectPtxPerformanceEvidence(
                    cell.Current.Median, cell.Current.P95, cell.CurrentBytes, 0,
                    cell.CurrentError, -1, independentRuns);
                DirectPtxReleaseDecision decision = policy.Evaluate(candidate, competitor);
                minimumSpeedup = Math.Min(minimumSpeedup, decision.MedianSpeedup);
                if (!decision.Passed)
                    failures.Add($"run {cell.Run}: {string.Join(", ", decision.Failures)}");
            }
            bool passed = cells.Length == independentRuns && failures.Count == 0;
            Console.WriteLine(
                $"{shape.Name,-12}: {(passed ? "PASS" : "HOLD"),-4} min median speedup {minimumSpeedup:F2}x" +
                (passed ? "; every paired gate passed" : $"; {string.Join("; ", failures)}"));
        }
    }

    private static void Print(
        int run, Shape shape, string method, Distribution time, double gflops,
        long bytes, float error, int registers, int shared, int local, int occupancy)
    {
        Console.WriteLine(
            $"{run,3} {shape.Name,-12} {method,-27} {time.Median,10:F2} {time.P95,10:F2} " +
            $"{time.P99,10:F2} {time.Mean,10:F2} {gflops,9:F2} {bytes,9} {0,9:F3} " +
            $"{error,10:G4} {Metric(registers),5} {Metric(shared),7} {Metric(local),5} {Metric(occupancy),4}");
    }

    private static double Gflops(Shape shape, double microseconds)
    {
        long keyVisits = (long)shape.Queries * (shape.Start + 1) +
            (long)shape.Queries * (shape.Queries - 1) / 2;
        double flops = 4.0 * shape.QueryHeads * keyVisits * Dimension;
        return flops / (microseconds * 1e-6) / 1e9;
    }

    private static float[] Values(Random random, int count) =>
        Enumerable.Range(0, count).Select(_ => (random.NextSingle() - 0.5f) * 0.5f).ToArray();

    private static float MaximumError(float[] actual, float[] expected)
    {
        float maximum = 0;
        for (int i = 0; i < actual.Length; i++) maximum = MathF.Max(maximum, MathF.Abs(actual[i] - expected[i]));
        return maximum;
    }

    private static double Percentile(double[] sorted, double percentile)
    {
        double index = (sorted.Length - 1) * percentile;
        int lower = (int)index, upper = Math.Min(lower + 1, sorted.Length - 1);
        return sorted[lower] + (sorted[upper] - sorted[lower]) * (index - lower);
    }

    private static string Metric(int value) => value < 0 ? "n/a" : value.ToString();
}
