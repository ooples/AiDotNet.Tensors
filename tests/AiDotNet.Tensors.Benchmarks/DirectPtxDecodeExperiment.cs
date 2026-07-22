using System.Diagnostics;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

namespace AiDotNet.Tensors.Benchmarks;

/// <summary>Issue #834 resident decode/paged-decode championship matrix.</summary>
internal static class DirectPtxDecodeExperiment
{
    private const int Dimension = 64;
    private const float Scale = 0.125f;
    private const int Warmups = 30;
    private const int Samples = 101;

    private readonly record struct Shape(
        string Name, bool Paged, int QueryHeads, int KeyValueHeads,
        int Sequence, int BlockSize, int PoolBlocks);

    private readonly record struct Distribution(double Mean, double Median, double P95, double P99);

    private readonly record struct CellEvidence(
        int Run, Shape Shape, Distribution Direct, Distribution Current,
        long DirectBytes, long CurrentBytes, long CurrentTemporaryBytes,
        float DirectError, float CurrentError, DirectPtxKernelAudit Audit);

    private static readonly Shape[] Shapes =
    [
        new("dense-mha", false, 8, 8, 32, 0, 0),
        new("dense-gqa", false, 8, 2, 64, 0, 0),
        new("dense-mqa", false, 8, 1, 128, 0, 0),
        new("paged-gqa", true, 8, 2, 64, 16, 6),
        new("paged-mqa", true, 8, 1, 128, 16, 10)
    ];

    internal static void Run(int independentRuns = 3)
    {
        Console.WriteLine(
            $"Direct-PTX decode family: {independentRuns} independent run(s), " +
            $"{Warmups} warmups + {Samples} synchronized E2E samples/cell");
        Console.WriteLine(
            $"{"Run",3} {"Shape",-11} {"Method",-26} {"median us",10} {"p95 us",10} " +
            $"{"p99 us",10} {"mean us",10} {"GFLOPS",9} {"B/call",9} {"tmp MiB",9} " +
            $"{"max err",10} {"regs",5} {"shared",7} {"local",5} {"occ",4}");
        Console.WriteLine(new string('-', 154));
        var evidence = new List<CellEvidence>(independentRuns * Shapes.Length);
        for (int run = 1; run <= independentRuns; run++)
        {
            using var backend = new CudaBackend();
            if (!backend.IsDirectPtxFlashDecodeEnabled || !backend.IsDirectPtxPagedDecodeEnabled)
                throw new InvalidOperationException(
                    "Set AIDOTNET_DIRECT_PTX_FLASH_DECODE=1 and AIDOTNET_DIRECT_PTX_PAGED_DECODE=1 on Ampere.");
            if (run == 1) Console.WriteLine($"GPU: {backend.DeviceName}");
            foreach (Shape shape in Shapes) evidence.Add(RunCell(backend, run, shape));
        }
        PrintReleaseGate(evidence, independentRuns);
    }

    private static CellEvidence RunCell(CudaBackend backend, int run, Shape shape)
    {
        int queryElements = shape.QueryHeads * Dimension;
        int keyValueElements = shape.Paged
            ? shape.PoolBlocks * shape.BlockSize * shape.KeyValueHeads * Dimension
            : shape.Sequence * shape.KeyValueHeads * Dimension;
        var random = new Random(20260900 + run * 10_000 + keyValueElements);
        float[] qHost = RandomValues(random, queryElements);
        float[] kHost = RandomValues(random, keyValueElements);
        float[] vHost = RandomValues(random, keyValueElements);
        int logicalBlocks = shape.Paged
            ? (shape.Sequence + shape.BlockSize - 1) / shape.BlockSize
            : 0;
        int[] tableHost = shape.Paged
            ? Enumerable.Range(0, logicalBlocks).Select(i => shape.PoolBlocks - 1 - i).ToArray()
            : [];

        using var q = backend.AllocateBuffer(qHost);
        using var k = backend.AllocateBuffer(kHost);
        using var v = backend.AllocateBuffer(vHost);
        using var table = shape.Paged ? backend.AllocateIntBuffer(tableHost) : null;
        using var directOutput = backend.AllocateBuffer(queryElements);
        using var currentOutput = backend.AllocateBuffer(queryElements);
        int splits = Math.Min(shape.Sequence, 8);
        using var partialM = shape.Paged ? null : backend.AllocateBuffer(shape.QueryHeads * splits);
        using var partialL = shape.Paged ? null : backend.AllocateBuffer(shape.QueryHeads * splits);
        using var partialAcc = shape.Paged
            ? null
            : backend.AllocateBuffer(shape.QueryHeads * splits * Dimension);

        void DirectLaunch()
        {
            bool launched = shape.Paged
                ? backend.TryDirectPtxPagedDecodeD64(
                    q, k, v, table!, directOutput,
                    shape.QueryHeads, shape.KeyValueHeads,
                    shape.BlockSize, shape.Sequence, Scale)
                : backend.TryDirectPtxFlashDecodeD64(
                    q, k, v, directOutput,
                    shape.QueryHeads, shape.KeyValueHeads, shape.Sequence, Scale);
            if (!launched) throw new InvalidOperationException(backend.DirectPtxLastError);
        }

        void CurrentLaunch()
        {
            if (shape.Paged)
                backend.PagedAttentionDecodeCurrentInto(
                    q, k, v, table!, currentOutput,
                    shape.QueryHeads, shape.KeyValueHeads, Dimension,
                    shape.BlockSize, shape.Sequence, Scale);
            else
                backend.FlashDecodeCurrentInto(
                    q, k, v, partialM!, partialL!, partialAcc!, currentOutput,
                    shape.QueryHeads, shape.KeyValueHeads, Dimension,
                    shape.Sequence, Scale, splits);
        }

        DirectLaunch(); CurrentLaunch(); backend.Synchronize();
        float[] expected = Oracle(qHost, kHost, vHost, tableHost, shape);
        float directError = MaximumError(backend.DownloadBuffer(directOutput), expected);
        float currentError = MaximumError(backend.DownloadBuffer(currentOutput), expected);
        Distribution direct = Measure(backend, DirectLaunch);
        long directBytes = MeasureAllocation(backend, DirectLaunch);
        AssertAudit(backend, shape, out DirectPtxKernelAudit audit);
        Print(run, shape, "Direct PTX register-online", direct,
            Gflops(shape, direct.Median), directBytes, 0, directError,
            audit.Function.RegistersPerThread, audit.Function.StaticSharedBytes,
            audit.Function.LocalBytesPerThread, audit.ActiveBlocksPerMultiprocessor);

        Distribution current = Measure(backend, CurrentLaunch);
        long currentBytes = MeasureAllocation(backend, CurrentLaunch);
        long temporaryBytes = shape.Paged
            ? 0
            : ((long)shape.QueryHeads * splits * 2 +
                (long)shape.QueryHeads * splits * Dimension) * sizeof(float);
        Print(run, shape, shape.Paged ? "AiDotNet paged NVRTC" : "AiDotNet FlashDecode NVRTC", current,
            Gflops(shape, current.Median), currentBytes, temporaryBytes, currentError,
            -1, -1, -1, -1);
        return new CellEvidence(
            run, shape, direct, current, directBytes, currentBytes,
            temporaryBytes, directError, currentError, audit);
    }

    private static void PrintReleaseGate(List<CellEvidence> evidence, int independentRuns)
    {
        Console.WriteLine();
        Console.WriteLine(
            "Paired-run production gate against the established AiDotNet CUDA path " +
            "(all independent runs must pass):");
        DirectPtxReleaseGatePolicy policy = DirectPtxReleaseGatePolicy.ProductionDefault;
        foreach (Shape shape in Shapes)
        {
            CellEvidence[] cells = evidence.Where(cell => cell.Shape == shape).ToArray();
            var failures = new List<string>();
            double minimumSpeedup = double.PositiveInfinity;
            foreach (CellEvidence cell in cells)
            {
                var candidate = new DirectPtxPerformanceEvidence(
                    cell.Direct.Median, cell.Direct.P95, cell.DirectBytes,
                    TemporaryDeviceBytes: 0, cell.DirectError,
                    cell.Audit.Function.LocalBytesPerThread, independentRuns);
                var competitor = new DirectPtxPerformanceEvidence(
                    cell.Current.Median, cell.Current.P95, cell.CurrentBytes,
                    cell.CurrentTemporaryBytes, cell.CurrentError,
                    LocalBytesPerThread: -1, independentRuns);
                DirectPtxReleaseDecision decision = policy.Evaluate(candidate, competitor);
                minimumSpeedup = Math.Min(minimumSpeedup, decision.MedianSpeedup);
                if (!decision.Passed)
                    failures.Add($"run {cell.Run}: {string.Join(", ", decision.Failures)}");
            }
            bool passed = cells.Length == independentRuns && failures.Count == 0;
            if (cells.Length != independentRuns)
                failures.Add($"captured-runs={cells.Length}<{independentRuns}");
            Console.WriteLine(
                $"{shape.Name,-11}: {(passed ? "PASS" : "HOLD"),-4} min median speedup {minimumSpeedup:F2}x" +
                (passed ? "; every paired gate passed" : $"; {string.Join("; ", failures)}"));
        }
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

    private static float[] Oracle(float[] q, float[] k, float[] v, int[] table, Shape shape)
    {
        var output = new float[shape.QueryHeads * Dimension];
        var logits = new float[shape.Sequence];
        int ratio = shape.QueryHeads / shape.KeyValueHeads;
        for (int head = 0; head < shape.QueryHeads; head++)
        {
            int kvHead = head / ratio;
            float maximum = float.NegativeInfinity;
            for (int token = 0; token < shape.Sequence; token++)
            {
                int physicalToken = shape.Paged
                    ? table[token / shape.BlockSize] * shape.BlockSize + token % shape.BlockSize
                    : token;
                int keyBase = (physicalToken * shape.KeyValueHeads + kvHead) * Dimension;
                float dot = 0;
                for (int d = 0; d < Dimension; d++) dot += q[head * Dimension + d] * k[keyBase + d];
                logits[token] = dot * Scale;
                maximum = MathF.Max(maximum, logits[token]);
            }
            float denominator = 0;
            for (int token = 0; token < shape.Sequence; token++)
            {
                logits[token] = MathF.Exp(logits[token] - maximum);
                denominator += logits[token];
            }
            for (int token = 0; token < shape.Sequence; token++)
            {
                int physicalToken = shape.Paged
                    ? table[token / shape.BlockSize] * shape.BlockSize + token % shape.BlockSize
                    : token;
                int valueBase = (physicalToken * shape.KeyValueHeads + kvHead) * Dimension;
                float probability = logits[token] / denominator;
                for (int d = 0; d < Dimension; d++)
                    output[head * Dimension + d] += probability * v[valueBase + d];
            }
        }
        return output;
    }

    private static void AssertAudit(CudaBackend backend, Shape shape, out DirectPtxKernelAudit audit)
    {
        if (!backend.TryGetDirectPtxDecodeAudit(
            shape.Paged, shape.QueryHeads, shape.KeyValueHeads, shape.Sequence,
            shape.BlockSize, shape.PoolBlocks, Scale, out audit))
            throw new InvalidOperationException("No audit for measured decode module.");
    }

    private static void Print(
        int run, Shape shape, string method, Distribution time, double gflops,
        long bytes, long temporaryBytes, float error,
        int registers, int shared, int local, int occupancy)
    {
        Console.WriteLine(
            $"{run,3} {shape.Name,-11} {method,-26} {time.Median,10:F2} {time.P95,10:F2} " +
            $"{time.P99,10:F2} {time.Mean,10:F2} {gflops,9:F2} {bytes,9} " +
            $"{temporaryBytes / 1048576.0,9:F3} {error,10:G4} " +
            $"{Metric(registers),5} {Metric(shared),7} {Metric(local),5} {Metric(occupancy),4}");
    }

    private static float[] RandomValues(Random random, int count) =>
        Enumerable.Range(0, count).Select(_ => (random.NextSingle() - 0.5f) * 0.5f).ToArray();

    private static double Gflops(Shape shape, double microseconds) =>
        4.0 * shape.QueryHeads * shape.Sequence * Dimension / (microseconds * 1e-6) / 1e9;

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
