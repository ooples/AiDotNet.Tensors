using System.Diagnostics;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

namespace AiDotNet.Tensors.Benchmarks;

/// <summary>Issue #834 deterministic probability-backward championship matrix.</summary>
internal static class DirectPtxAttentionBackwardExperiment
{
    private const int Dimension = 64;
    private const float Scale = 0.125f;
    private const int Warmups = 30;
    private const int Samples = 101;

    private readonly record struct Shape(
        string Name, int QueryHeads, int KeyValueHeads,
        int QuerySequence, int KeyValueSequence);
    private readonly record struct Distribution(double Mean, double Median, double P95, double P99);
    private readonly record struct CellEvidence(
        int Run, Shape Shape, Distribution Direct, Distribution Current,
        long DirectBytes, long CurrentBytes, float DirectError, float CurrentError,
        DirectPtxKernelAudit RowDeltaAudit,
        DirectPtxKernelAudit GradQueryAudit, DirectPtxKernelAudit GradKeyValueAudit);

    private static readonly Shape[] Shapes =
    [
        new("backward-mha", 8, 8, 16, 16),
        new("backward-gqa", 8, 2, 16, 32),
        new("backward-mqa", 8, 1, 32, 16),
        new("backward-long", 8, 2, 64, 64)
    ];

    internal static void Run(int independentRuns = 3)
    {
        if (independentRuns <= 0) throw new ArgumentOutOfRangeException(nameof(independentRuns));
        bool previousDeterminism = GpuDeterminism.IsActive;
        AiDotNetEngine.SetDeterministicMode(true);
        try
        {
            Console.WriteLine(
                $"Direct-PTX deterministic attention backward: {independentRuns} independent run(s), " +
                $"{Warmups} warmups + {Samples} synchronized E2E samples/cell");
            Console.WriteLine(
                $"{"Run",3} {"Shape",-14} {"Method",-29} {"median us",10} {"p95 us",10} " +
                $"{"p99 us",10} {"mean us",10} {"GFLOPS",9} {"B/call",9} {"tmp MiB",9} " +
                $"{"max err",10} {"regs d/dq/dkv",13} {"shared",7} {"local",5} {"occ",10}");
            Console.WriteLine(new string('-', 170));
            var evidence = new List<CellEvidence>(independentRuns * Shapes.Length);
            for (int run = 1; run <= independentRuns; run++)
            {
                using var backend = new CudaBackend();
                if (!backend.IsDirectPtxAttentionBackwardEnabled)
                    throw new InvalidOperationException(
                        "Set AIDOTNET_DIRECT_PTX_ATTENTION_BACKWARD=1 on an Ampere GPU.");
                if (run == 1) Console.WriteLine($"GPU: {backend.DeviceName}; deterministic current lane forced");
                foreach (Shape shape in Shapes) evidence.Add(RunCell(backend, run, shape));
            }
            PrintReleaseGate(evidence, independentRuns);
        }
        finally
        {
            AiDotNetEngine.SetDeterministicMode(previousDeterminism);
        }
    }

    private static CellEvidence RunCell(CudaBackend backend, int run, Shape shape)
    {
        const int batch = 1;
        int queryElements = batch * shape.QueryHeads * shape.QuerySequence * Dimension;
        int keyValueElements = batch * shape.KeyValueHeads * shape.KeyValueSequence * Dimension;
        var random = new Random(20261100 + run * 10_000 + queryElements + keyValueElements);
        float[] gradOutputHost = Values(random, queryElements);
        float[] queryHost = Values(random, queryElements);
        float[] keyHost = Values(random, keyValueElements);
        float[] valueHost = Values(random, keyValueElements);
        float[] probabilitiesHost = Probabilities(queryHost, keyHost, shape);
        (float[] expectedQ, float[] expectedK, float[] expectedV) = Oracle(
            gradOutputHost, queryHost, keyHost, valueHost, probabilitiesHost, shape);

        using var gradOutput = backend.AllocateBuffer(gradOutputHost);
        using var query = backend.AllocateBuffer(queryHost);
        using var key = backend.AllocateBuffer(keyHost);
        using var value = backend.AllocateBuffer(valueHost);
        using var probabilities = backend.AllocateBuffer(probabilitiesHost);
        using var directGradQuery = backend.AllocateBuffer(queryElements);
        using var directGradKey = backend.AllocateBuffer(keyValueElements);
        using var directGradValue = backend.AllocateBuffer(keyValueElements);
        using var currentGradQuery = backend.AllocateBuffer(queryElements);
        using var currentGradKey = backend.AllocateBuffer(keyValueElements);
        using var currentGradValue = backend.AllocateBuffer(keyValueElements);

        void DirectLaunch()
        {
            if (!backend.TryDirectPtxAttentionBackwardD64(
                gradOutput, query, key, value, probabilities,
                directGradQuery, directGradKey, directGradValue,
                batch, shape.QueryHeads, shape.KeyValueHeads,
                shape.QuerySequence, shape.KeyValueSequence, Dimension, Scale))
                throw new InvalidOperationException(backend.DirectPtxLastError);
        }

        void CurrentLaunch() => backend.GroupedQueryAttentionBackwardCurrentInto(
            gradOutput, query, key, value, probabilities,
            currentGradQuery, currentGradKey, currentGradValue,
            batch, shape.QueryHeads, shape.KeyValueHeads,
            shape.QuerySequence, shape.KeyValueSequence, Dimension, Scale,
            shape.QueryHeads / shape.KeyValueHeads);

        DirectLaunch(); CurrentLaunch(); backend.Synchronize();
        float directError = MaximumGradientError(
            backend, directGradQuery, directGradKey, directGradValue,
            expectedQ, expectedK, expectedV);
        float currentError = MaximumGradientError(
            backend, currentGradQuery, currentGradKey, currentGradValue,
            expectedQ, expectedK, expectedV);
        Distribution direct = Measure(backend, DirectLaunch);
        long directBytes = MeasureAllocation(backend, DirectLaunch);
        if (!backend.TryGetDirectPtxAttentionBackwardAudits(
            batch, shape.QueryHeads, shape.KeyValueHeads,
            shape.QuerySequence, shape.KeyValueSequence, Scale,
            out DirectPtxKernelAudit rowDeltaAudit,
            out DirectPtxKernelAudit gradQueryAudit,
            out DirectPtxKernelAudit gradKeyValueAudit))
            throw new InvalidOperationException("No audit for measured attention-backward module.");
        Print(run, shape, "Direct PTX deterministic", direct,
            Gflops(shape, direct.Median), directBytes, directError,
            rowDeltaAudit.Function.RegistersPerThread,
            gradQueryAudit.Function.RegistersPerThread,
            gradKeyValueAudit.Function.RegistersPerThread,
            gradKeyValueAudit.Function.StaticSharedBytes,
            Math.Max(rowDeltaAudit.Function.LocalBytesPerThread,
                Math.Max(gradQueryAudit.Function.LocalBytesPerThread,
                    gradKeyValueAudit.Function.LocalBytesPerThread)),
            rowDeltaAudit.ActiveBlocksPerMultiprocessor,
            gradQueryAudit.ActiveBlocksPerMultiprocessor,
            gradKeyValueAudit.ActiveBlocksPerMultiprocessor);

        Distribution current = Measure(backend, CurrentLaunch);
        long currentBytes = MeasureAllocation(backend, CurrentLaunch);
        Print(run, shape, "AiDotNet deterministic NVRTC", current,
            Gflops(shape, current.Median), currentBytes, currentError,
            -1, -1, -1, -1, -1, -1, -1, -1);
        return new CellEvidence(
            run, shape, direct, current, directBytes, currentBytes,
            directError, currentError, rowDeltaAudit, gradQueryAudit, gradKeyValueAudit);
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
            action();
            backend.Synchronize();
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
        long bytes = (GC.GetAllocatedBytesForCurrentThread() - before) / Samples;
        backend.Synchronize();
        return bytes;
    }

    private static float[] Probabilities(float[] query, float[] key, Shape shape)
    {
        var probabilities = new float[
            shape.QueryHeads * shape.QuerySequence * shape.KeyValueSequence];
        int ratio = shape.QueryHeads / shape.KeyValueHeads;
        for (int qh = 0; qh < shape.QueryHeads; qh++)
        for (int qi = 0; qi < shape.QuerySequence; qi++)
        {
            int kvh = qh / ratio;
            int queryBase = (qh * shape.QuerySequence + qi) * Dimension;
            int probabilityBase = (qh * shape.QuerySequence + qi) * shape.KeyValueSequence;
            float maximum = float.NegativeInfinity;
            for (int ki = 0; ki < shape.KeyValueSequence; ki++)
            {
                int keyBase = (kvh * shape.KeyValueSequence + ki) * Dimension;
                float score = 0;
                for (int d = 0; d < Dimension; d++)
                    score += query[queryBase + d] * key[keyBase + d];
                score *= Scale;
                probabilities[probabilityBase + ki] = score;
                maximum = MathF.Max(maximum, score);
            }
            float sum = 0;
            for (int ki = 0; ki < shape.KeyValueSequence; ki++)
            {
                float probability = MathF.Exp(probabilities[probabilityBase + ki] - maximum);
                probabilities[probabilityBase + ki] = probability;
                sum += probability;
            }
            for (int ki = 0; ki < shape.KeyValueSequence; ki++)
                probabilities[probabilityBase + ki] /= sum;
        }
        return probabilities;
    }

    private static (float[] GradQuery, float[] GradKey, float[] GradValue) Oracle(
        float[] gradOutput, float[] query, float[] key, float[] value,
        float[] probabilities, Shape shape)
    {
        var gradQuery = new float[query.Length];
        var gradKey = new float[key.Length];
        var gradValue = new float[value.Length];
        var gradProbability = new float[shape.KeyValueSequence];
        int ratio = shape.QueryHeads / shape.KeyValueHeads;
        for (int qh = 0; qh < shape.QueryHeads; qh++)
        for (int qi = 0; qi < shape.QuerySequence; qi++)
        {
            int kvh = qh / ratio;
            int queryBase = (qh * shape.QuerySequence + qi) * Dimension;
            int probabilityBase = (qh * shape.QuerySequence + qi) * shape.KeyValueSequence;
            float delta = 0;
            for (int ki = 0; ki < shape.KeyValueSequence; ki++)
            {
                int valueBase = (kvh * shape.KeyValueSequence + ki) * Dimension;
                float dot = 0;
                for (int d = 0; d < Dimension; d++)
                    dot += gradOutput[queryBase + d] * value[valueBase + d];
                gradProbability[ki] = dot;
                delta += probabilities[probabilityBase + ki] * dot;
            }
            for (int ki = 0; ki < shape.KeyValueSequence; ki++)
            {
                int keyValueBase = (kvh * shape.KeyValueSequence + ki) * Dimension;
                float probability = probabilities[probabilityBase + ki];
                float gradScore = Scale * probability * (gradProbability[ki] - delta);
                for (int d = 0; d < Dimension; d++)
                {
                    gradQuery[queryBase + d] += gradScore * key[keyValueBase + d];
                    gradKey[keyValueBase + d] += gradScore * query[queryBase + d];
                    gradValue[keyValueBase + d] += probability * gradOutput[queryBase + d];
                }
            }
        }
        return (gradQuery, gradKey, gradValue);
    }

    private static float MaximumGradientError(
        CudaBackend backend,
        IGpuBuffer gradQuery, IGpuBuffer gradKey, IGpuBuffer gradValue,
        float[] expectedQ, float[] expectedK, float[] expectedV) =>
        MathF.Max(
            MaximumError(backend.DownloadBuffer(gradQuery), expectedQ),
            MathF.Max(
                MaximumError(backend.DownloadBuffer(gradKey), expectedK),
                MaximumError(backend.DownloadBuffer(gradValue), expectedV)));

    private static void PrintReleaseGate(List<CellEvidence> evidence, int independentRuns)
    {
        Console.WriteLine();
        Console.WriteLine(
            "Paired-run production gate against established deterministic AiDotNet CUDA:");
        DirectPtxReleaseGatePolicy policy = DirectPtxReleaseGatePolicy.ProductionDefault;
        foreach (Shape shape in Shapes)
        {
            CellEvidence[] cells = evidence.Where(cell => cell.Shape == shape).ToArray();
            var failures = new List<string>();
            double minimumSpeedup = double.PositiveInfinity;
            foreach (CellEvidence cell in cells)
            {
                int localBytes = Math.Max(
                    cell.RowDeltaAudit.Function.LocalBytesPerThread,
                    Math.Max(cell.GradQueryAudit.Function.LocalBytesPerThread,
                        cell.GradKeyValueAudit.Function.LocalBytesPerThread));
                var candidate = new DirectPtxPerformanceEvidence(
                    cell.Direct.Median, cell.Direct.P95, cell.DirectBytes, 0,
                    cell.DirectError, localBytes, independentRuns);
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
                $"{shape.Name,-14}: {(passed ? "PASS" : "HOLD"),-4} " +
                $"min median speedup {minimumSpeedup:F2}x" +
                (passed ? "; every paired gate passed" : $"; {string.Join("; ", failures)}"));
        }
    }

    private static void Print(
        int run, Shape shape, string method, Distribution time, double gflops,
        long bytes, float error, int rowDeltaRegisters,
        int gradQueryRegisters, int gradKeyValueRegisters,
        int shared, int local, int rowDeltaOccupancy,
        int gradQueryOccupancy, int gradKeyValueOccupancy)
    {
        string registers = rowDeltaRegisters < 0
            ? "n/a"
            : $"{rowDeltaRegisters}/{gradQueryRegisters}/{gradKeyValueRegisters}";
        string occupancy = rowDeltaOccupancy < 0
            ? "n/a"
            : $"{rowDeltaOccupancy}/{gradQueryOccupancy}/{gradKeyValueOccupancy}";
        Console.WriteLine(
            $"{run,3} {shape.Name,-14} {method,-29} {time.Median,10:F2} {time.P95,10:F2} " +
            $"{time.P99,10:F2} {time.Mean,10:F2} {gflops,9:F2} {bytes,9} {0,9:F3} " +
            $"{error,10:G4} {registers,13} {Metric(shared),7} {Metric(local),5} {occupancy,10}");
    }

    private static double Gflops(Shape shape, double microseconds)
    {
        double usefulFlops = 8.0 * shape.QueryHeads * shape.QuerySequence *
            shape.KeyValueSequence * Dimension;
        return usefulFlops / (microseconds * 1e-6) / 1e9;
    }

    private static float[] Values(Random random, int count) =>
        Enumerable.Range(0, count).Select(_ => (random.NextSingle() - 0.5f) * 0.5f).ToArray();

    private static float MaximumError(float[] actual, float[] expected)
    {
        float maximum = 0;
        for (int i = 0; i < actual.Length; i++)
            maximum = MathF.Max(maximum, MathF.Abs(actual[i] - expected[i]));
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
