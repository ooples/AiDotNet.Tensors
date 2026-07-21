using System.Diagnostics;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;
using AiDotNet.Tensors.Engines.Gpu;

namespace AiDotNet.Tensors.Benchmarks;

/// <summary>Issue #834 Flash/recomputation-backward championship matrix.</summary>
internal static class DirectPtxFlashAttentionBackwardExperiment
{
    private const int Dimension = 64;
    private const float Scale = 0.125f;
    private const int Warmups = 30;
    private const int Samples = 101;
    private const int LaunchesPerDeviceSample = 10;

    private readonly record struct Shape(
        string Name, int Heads, int QuerySequence, int KeyValueSequence,
        bool IsCausal, bool HasBias);
    private readonly record struct Distribution(double Mean, double Median, double P95, double P99);
    private readonly record struct CellEvidence(
        int Run, Shape Shape,
        Distribution DirectDevice, Distribution CurrentDevice,
        Distribution DirectEndToEnd, Distribution CurrentEndToEnd,
        long DirectBytes, long CurrentBytes, float DirectError, float CurrentError,
        DirectPtxKernelAudit GradQueryAudit, DirectPtxKernelAudit GradKeyValueAudit);

    private static readonly Shape[] Shapes =
    [
        new("flash-bwd-mha", 8, 16, 16, false, false),
        new("flash-bwd-rect", 8, 16, 32, false, false),
        new("flash-bwd-bias", 8, 16, 32, false, true),
        new("flash-bwd-causal", 8, 32, 32, true, false),
        new("flash-bwd-long", 8, 64, 64, true, false)
    ];

    internal static void Run(int independentRuns = 3)
    {
        if (independentRuns <= 0) throw new ArgumentOutOfRangeException(nameof(independentRuns));
        bool previousDeterminism = GpuDeterminism.IsActive;
        AiDotNetEngine.SetDeterministicMode(true);
        try
        {
            Console.WriteLine(
                $"Direct-PTX FlashAttention recomputation backward: {independentRuns} independent run(s), " +
                $"{Warmups} warmups + {Samples} CUDA-event and synchronized E2E samples/cell");
            Console.WriteLine(
                $"Device samples average {LaunchesPerDeviceSample} consecutive launches; " +
                "the production gate uses device time and E2E remains diagnostic.");
            Console.WriteLine(
                $"{"Run",3} {"Shape",-16} {"Method",-29} {"dev med",9} {"dev p95",9} " +
                $"{"dev p99",9} {"E2E med",9} {"E2E p95",9} {"E2E p99",9} " +
                $"{"GFLOPS",9} {"B/call",9} {"tmp MiB",9} " +
                $"{"max err",10} {"regs dq/dkv",11} {"shared",7} {"local",5} {"occ",7}");
            Console.WriteLine(new string('-', 184));
            var evidence = new List<CellEvidence>(independentRuns * Shapes.Length);
            for (int run = 1; run <= independentRuns; run++)
            {
                using var backend = new CudaBackend();
                if (!backend.IsDirectPtxFlashAttentionBackwardEnabled)
                    throw new InvalidOperationException(
                        "Set AIDOTNET_DIRECT_PTX_FLASH_ATTENTION_BACKWARD=1 on an Ampere GPU.");
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
        int queryElements = batch * shape.Heads * shape.QuerySequence * Dimension;
        int keyValueElements = batch * shape.Heads * shape.KeyValueSequence * Dimension;
        var random = new Random(20261200 + run * 10_000 + queryElements + keyValueElements);
        float[] gradOutputHost = Values(random, queryElements);
        float[] queryHost = Values(random, queryElements);
        float[] keyHost = Values(random, keyValueElements);
        float[] valueHost = Values(random, keyValueElements);
        float[]? biasHost = shape.HasBias
            ? Values(random, shape.Heads * shape.QuerySequence * shape.KeyValueSequence)
            : null;
        (float[] probabilities, float[] outputHost, float[] statsHost) =
            ForwardReference(queryHost, keyHost, valueHost, shape, biasHost);
        (float[] expectedQ, float[] expectedK, float[] expectedV) = Oracle(
            gradOutputHost, queryHost, keyHost, valueHost, probabilities, shape);

        using var gradOutput = backend.AllocateBuffer(gradOutputHost);
        using var query = backend.AllocateBuffer(queryHost);
        using var key = backend.AllocateBuffer(keyHost);
        using var value = backend.AllocateBuffer(valueHost);
        using var output = backend.AllocateBuffer(outputHost);
        using var stats = backend.AllocateBuffer(statsHost);
        using var directGradQuery = backend.AllocateBuffer(queryElements);
        using var directGradKey = backend.AllocateBuffer(keyValueElements);
        using var directGradValue = backend.AllocateBuffer(keyValueElements);
        using var currentGradQuery = backend.AllocateBuffer(queryElements);
        using var currentGradKey = backend.AllocateBuffer(keyValueElements);
        using var currentGradValue = backend.AllocateBuffer(keyValueElements);
        using IGpuBuffer? bias = biasHost is null ? null : backend.AllocateBuffer(biasHost);

        void DirectLaunch()
        {
            if (!backend.TryDirectPtxFlashAttentionBackwardD64(
                gradOutput, query, key, value, output, stats,
                directGradQuery, directGradKey, directGradValue,
                batch, shape.Heads, shape.QuerySequence, shape.KeyValueSequence,
                Dimension, Scale, shape.IsCausal, bias, biasBatchStride: 0))
                throw new InvalidOperationException(backend.DirectPtxLastError);
        }

        void CurrentLaunch() => backend.FlashAttentionBackwardCurrentInto(
            gradOutput, query, key, value, output, stats,
            currentGradQuery, currentGradKey, currentGradValue,
            batch, shape.Heads, shape.QuerySequence, shape.KeyValueSequence,
            Dimension, Scale, shape.IsCausal, bias, biasBatchStride: 0);

        DirectLaunch(); CurrentLaunch(); backend.Synchronize();
        float directError = MaximumGradientError(
            backend, directGradQuery, directGradKey, directGradValue,
            expectedQ, expectedK, expectedV);
        float currentError = MaximumGradientError(
            backend, currentGradQuery, currentGradKey, currentGradValue,
            expectedQ, expectedK, expectedV);
        Distribution directDevice = MeasureDevice(backend, DirectLaunch);
        Distribution directEndToEnd = MeasureEndToEnd(backend, DirectLaunch);
        long directBytes = MeasureAllocation(backend, DirectLaunch);
        if (!backend.TryGetDirectPtxFlashAttentionBackwardAudits(
            batch, shape.Heads, shape.QuerySequence, shape.KeyValueSequence,
            Scale, shape.IsCausal,
            out DirectPtxKernelAudit gradQueryAudit,
            out DirectPtxKernelAudit gradKeyValueAudit,
            shape.HasBias ? 0 : -1))
            throw new InvalidOperationException("No audit for measured FlashAttention-backward module.");
        Print(run, shape, "Direct PTX deterministic", directDevice, directEndToEnd,
            Gflops(shape, directDevice.Median), directBytes, directError,
            gradQueryAudit.Function.RegistersPerThread,
            gradKeyValueAudit.Function.RegistersPerThread,
            Math.Max(gradQueryAudit.Function.StaticSharedBytes,
                gradKeyValueAudit.Function.StaticSharedBytes),
            Math.Max(gradQueryAudit.Function.LocalBytesPerThread,
                gradKeyValueAudit.Function.LocalBytesPerThread),
            gradQueryAudit.ActiveBlocksPerMultiprocessor,
            gradKeyValueAudit.ActiveBlocksPerMultiprocessor);

        Distribution currentDevice = MeasureDevice(backend, CurrentLaunch);
        Distribution currentEndToEnd = MeasureEndToEnd(backend, CurrentLaunch);
        long currentBytes = MeasureAllocation(backend, CurrentLaunch);
        Print(run, shape, "AiDotNet deterministic NVRTC", currentDevice, currentEndToEnd,
            Gflops(shape, currentDevice.Median), currentBytes, currentError,
            -1, -1, -1, -1, -1, -1);
        return new CellEvidence(
            run, shape, directDevice, currentDevice, directEndToEnd, currentEndToEnd,
            directBytes, currentBytes,
            directError, currentError, gradQueryAudit, gradKeyValueAudit);
    }

    private static Distribution MeasureDevice(CudaBackend backend, Action action)
    {
        for (int i = 0; i < Warmups; i++) action();
        backend.Synchronize();
        var samples = new double[Samples];
        using IGpuEvent start = backend.CreateEvent(enableTiming: true);
        using IGpuEvent end = backend.CreateEvent(enableTiming: true);
        for (int i = 0; i < samples.Length; i++)
        {
            backend.RecordEvent(start, backend.DefaultStream);
            for (int launch = 0; launch < LaunchesPerDeviceSample; launch++) action();
            backend.RecordEvent(end, backend.DefaultStream);
            end.Synchronize();
            samples[i] = backend.GetEventElapsedTime(start, end) * 1_000.0 /
                LaunchesPerDeviceSample;
        }
        return Summarize(samples);
    }

    private static Distribution MeasureEndToEnd(CudaBackend backend, Action action)
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
        return Summarize(samples);
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

    private static (float[] Probabilities, float[] Output, float[] Stats) ForwardReference(
        float[] query, float[] key, float[] value, Shape shape, float[]? attentionBias)
    {
        var probabilities = new float[shape.Heads * shape.QuerySequence * shape.KeyValueSequence];
        var output = new float[query.Length];
        var stats = new float[shape.Heads * shape.QuerySequence];
        for (int h = 0; h < shape.Heads; h++)
        for (int qi = 0; qi < shape.QuerySequence; qi++)
        {
            int queryBase = (h * shape.QuerySequence + qi) * Dimension;
            int probabilityBase = (h * shape.QuerySequence + qi) * shape.KeyValueSequence;
            float maximum = float.NegativeInfinity;
            for (int ki = 0; ki < shape.KeyValueSequence; ki++)
            {
                if (shape.IsCausal && ki > qi) continue;
                int keyBase = (h * shape.KeyValueSequence + ki) * Dimension;
                float score = 0;
                for (int d = 0; d < Dimension; d++)
                    score += query[queryBase + d] * key[keyBase + d];
                score *= Scale;
                if (attentionBias is not null)
                    score += attentionBias[probabilityBase + ki];
                probabilities[probabilityBase + ki] = score;
                maximum = MathF.Max(maximum, score);
            }
            float sum = 0;
            for (int ki = 0; ki < shape.KeyValueSequence; ki++)
            {
                if (shape.IsCausal && ki > qi)
                {
                    probabilities[probabilityBase + ki] = 0;
                    continue;
                }
                float probability = MathF.Exp(probabilities[probabilityBase + ki] - maximum);
                probabilities[probabilityBase + ki] = probability;
                sum += probability;
            }
            stats[h * shape.QuerySequence + qi] = maximum + MathF.Log(sum);
            for (int ki = 0; ki < shape.KeyValueSequence; ki++)
            {
                float probability = probabilities[probabilityBase + ki] / sum;
                probabilities[probabilityBase + ki] = probability;
                int valueBase = (h * shape.KeyValueSequence + ki) * Dimension;
                for (int d = 0; d < Dimension; d++)
                    output[queryBase + d] += probability * value[valueBase + d];
            }
        }
        return (probabilities, output, stats);
    }

    private static (float[] GradQuery, float[] GradKey, float[] GradValue) Oracle(
        float[] gradOutput, float[] query, float[] key, float[] value,
        float[] probabilities, Shape shape)
    {
        var gradQuery = new float[query.Length];
        var gradKey = new float[key.Length];
        var gradValue = new float[value.Length];
        var gradProbability = new float[shape.KeyValueSequence];
        for (int h = 0; h < shape.Heads; h++)
        for (int qi = 0; qi < shape.QuerySequence; qi++)
        {
            int queryBase = (h * shape.QuerySequence + qi) * Dimension;
            int probabilityBase = (h * shape.QuerySequence + qi) * shape.KeyValueSequence;
            float delta = 0;
            for (int ki = 0; ki < shape.KeyValueSequence; ki++)
            {
                int valueBase = (h * shape.KeyValueSequence + ki) * Dimension;
                float dot = 0;
                for (int d = 0; d < Dimension; d++)
                    dot += gradOutput[queryBase + d] * value[valueBase + d];
                gradProbability[ki] = dot;
                delta += probabilities[probabilityBase + ki] * dot;
            }
            for (int ki = 0; ki < shape.KeyValueSequence; ki++)
            {
                int keyValueBase = (h * shape.KeyValueSequence + ki) * Dimension;
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
                    cell.GradQueryAudit.Function.LocalBytesPerThread,
                    cell.GradKeyValueAudit.Function.LocalBytesPerThread);
                var candidate = new DirectPtxPerformanceEvidence(
                    cell.DirectDevice.Median, cell.DirectDevice.P95, cell.DirectBytes, 0,
                    cell.DirectError, localBytes, independentRuns);
                var competitor = new DirectPtxPerformanceEvidence(
                    cell.CurrentDevice.Median, cell.CurrentDevice.P95, cell.CurrentBytes, 0,
                    cell.CurrentError, -1, independentRuns);
                DirectPtxReleaseDecision decision = policy.Evaluate(candidate, competitor);
                minimumSpeedup = Math.Min(minimumSpeedup, decision.MedianSpeedup);
                if (!decision.Passed)
                    failures.Add($"run {cell.Run}: {string.Join(", ", decision.Failures)}");
            }
            bool passed = cells.Length == independentRuns && failures.Count == 0;
            Console.WriteLine(
                $"{shape.Name,-16}: {(passed ? "PASS" : "HOLD"),-4} " +
                $"min median speedup {minimumSpeedup:F2}x" +
                (passed ? "; every paired gate passed" : $"; {string.Join("; ", failures)}"));
        }
    }

    private static void Print(
        int run, Shape shape, string method,
        Distribution device, Distribution endToEnd, double gflops,
        long bytes, float error, int gradQueryRegisters, int gradKeyValueRegisters,
        int shared, int local, int gradQueryOccupancy, int gradKeyValueOccupancy)
    {
        string registers = gradQueryRegisters < 0
            ? "n/a"
            : $"{gradQueryRegisters}/{gradKeyValueRegisters}";
        string occupancy = gradQueryOccupancy < 0
            ? "n/a"
            : $"{gradQueryOccupancy}/{gradKeyValueOccupancy}";
        Console.WriteLine(
            $"{run,3} {shape.Name,-16} {method,-29} " +
            $"{device.Median,9:F2} {device.P95,9:F2} {device.P99,9:F2} " +
            $"{endToEnd.Median,9:F2} {endToEnd.P95,9:F2} {endToEnd.P99,9:F2} " +
            $"{gflops,9:F2} {bytes,9} {0,9:F3} " +
            $"{error,10:G4} {registers,11} {Metric(shared),7} {Metric(local),5} {occupancy,7}");
    }

    private static double Gflops(Shape shape, double microseconds)
    {
        double usefulFlops = 8.0 * shape.Heads * shape.QuerySequence *
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

    private static Distribution Summarize(double[] samples)
    {
        Array.Sort(samples);
        return new(samples.Average(), Percentile(samples, 0.50),
            Percentile(samples, 0.95), Percentile(samples, 0.99));
    }

    private static string Metric(int value) => value < 0 ? "n/a" : value.ToString();
}
