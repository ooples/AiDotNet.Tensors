using System.Diagnostics;
using System.Text.Json;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;
using AiDotNet.Tensors.Engines.Gpu;

namespace AiDotNet.Tensors.Benchmarks;

/// <summary>
/// Issue-#850 resident NVIDIA diagnostic for interleaved FP32 complex multiply.
/// Production promotion also consumes the companion PyTorch CUDA records.
/// </summary>
internal static class DirectPtxComplexMultiplyExperiment
{
    private const int Warmups = 30;
    private const int Samples = 101;
    private const int LaunchesPerDeviceSample = 50;
    private const int ThroughputOperationsPerGraph = 64;
    private const double RequiredGain = 1.10;
    private const double SemanticTolerance = 1.0e-5;
    private static readonly int[] Shapes = [65536, 262144, 1048576, 4194304];

    private readonly record struct Distribution(
        double Mean, double Median, double P95, double P99);

    private readonly record struct Result(
        int Run,
        int NumPairs,
        string Method,
        Distribution Device,
        Distribution EndToEnd,
        double Gflops,
        double GigabytesPerSecond,
        long ManagedBytes,
        long TemporaryDeviceBytes,
        float MaxError,
        int Registers,
        int SharedBytes,
        int LocalBytes,
        int ActiveBlocks);

    private readonly record struct OracleResult(
        int Run,
        int NumPairs,
        string Lane,
        StableTimer.PairResult Timing,
        double DirectError,
        double IncumbentError,
        DirectPtxKernelAudit Audit);

    internal static void Run(int independentRuns = 3, bool oracleOnly = false)
    {
        if (independentRuns <= 0)
            throw new ArgumentOutOfRangeException(nameof(independentRuns));
        var results = new List<Result>();
        var oracle = new List<OracleResult>();
        GpuBenchmarkEnvironment.RequireIdleGpu("complex-multiply-suite-start");
        for (int run = 1; run <= independentRuns; run++)
        {
            GpuBenchmarkEnvironment.PrintSnapshot($"complex-multiply-start-{run}");
            if (!oracleOnly)
            {
                RunDirect(run, results);
                RunEstablished(run, results);
            }
            RunOracle(run, oracle);
            GpuBenchmarkEnvironment.PrintSnapshot($"complex-multiply-end-{run}");
        }
        GpuBenchmarkEnvironment.RequireNoForeignCompute(
            "complex-multiply-suite-end", afterSuite: true);
        if (!oracleOnly) Print(results);
        PrintOracle(oracle);
        Console.WriteLine("Production verdict remains HOLD until the companion PyTorch CUDA records and Nsight evidence are joined with these results.");
    }

    private static void RunDirect(int run, List<Result> results)
    {
        using var backend = new CudaBackend();
        if (!backend.IsAvailable)
        {
            Console.Error.WriteLine(
                "[complex-multiply] CUDA backend unavailable; the candidate lane produced no rows.");
            return;
        }
        bool originalExperiment = DirectPtxFeatureGate.ComplexMultiplyExperimentOverride;
        bool? originalGate = DirectPtxFeatureGate.ComplexMultiplyGateOverride;
        try
        {
            DirectPtxFeatureGate.ComplexMultiplyExperimentOverride = true;
            DirectPtxFeatureGate.ComplexMultiplyGateOverride = true;
            foreach (int numPairs in Shapes)
            {
                if (!backend.PrewarmDirectPtxComplexMultiply(numPairs))
                    throw new InvalidOperationException(
                        $"Direct PTX prewarm failed for {numPairs} pairs: " +
                        (backend.DirectPtxLastError ?? "no diagnostic"));
                float[] left = Values(numPairs * 2, 1000 + numPairs);
                float[] right = Values(numPairs * 2, 2000 + numPairs);
                using var leftBuffer = backend.AllocateBuffer(left);
                using var rightBuffer = backend.AllocateBuffer(right);
                using var outputBuffer = backend.AllocateBuffer(numPairs * 2);
                Action launch = () => backend.ComplexMultiply(
                    leftBuffer, rightBuffer, outputBuffer, numPairs);
                long dispatchesBefore = backend.DirectPtxComplexMultiplyDispatchCount;
                Distribution device = MeasureDevice(backend, launch);
                Distribution endToEnd = MeasureEndToEnd(backend.Synchronize, launch);
                long managedBytes = MeasureAllocation(backend.Synchronize, launch);
                launch();
                backend.Synchronize();
                long expectedDispatches =
                    Warmups + Samples * LaunchesPerDeviceSample +
                    Warmups + Samples + 8 + Samples + 1;
                if (backend.DirectPtxComplexMultiplyDispatchCount - dispatchesBefore !=
                    expectedDispatches)
                    throw new InvalidOperationException(
                        "A measured candidate launch fell back from the direct-PTX route.");
                var actual = new float[left.Length];
                backend.DownloadBuffer(outputBuffer, actual);
                float error = Validate(actual, left, right, numPairs);
                if (!backend.TryGetDirectPtxComplexMultiplyAudit(numPairs, out var audit))
                    throw new InvalidOperationException("The prewarmed direct-PTX module has no audit record.");
                results.Add(CreateResult(
                    run, numPairs, "Direct PTX", device, endToEnd, managedBytes, 0,
                    error, audit.Function.RegistersPerThread,
                    audit.Function.StaticSharedBytes,
                    audit.Function.LocalBytesPerThread,
                    audit.ActiveBlocksPerMultiprocessor));
            }
        }
        finally
        {
            DirectPtxFeatureGate.ComplexMultiplyGateOverride = originalGate;
            DirectPtxFeatureGate.ComplexMultiplyExperimentOverride = originalExperiment;
        }
    }

    private static void RunEstablished(int run, List<Result> results)
    {
        using var backend = new CudaBackend();
        if (!backend.IsAvailable)
        {
            Console.Error.WriteLine(
                "[complex-multiply] CUDA backend unavailable; the established lane produced no rows.");
            return;
        }
        bool? originalGate = DirectPtxFeatureGate.ComplexMultiplyGateOverride;
        try
        {
            DirectPtxFeatureGate.ComplexMultiplyGateOverride = false;
            foreach (int numPairs in Shapes)
            {
                float[] left = Values(numPairs * 2, 1000 + numPairs);
                float[] right = Values(numPairs * 2, 2000 + numPairs);
                using var leftBuffer = backend.AllocateBuffer(left);
                using var rightBuffer = backend.AllocateBuffer(right);
                using var outputBuffer = backend.AllocateBuffer(numPairs * 2);
                Action launch = () => backend.ComplexMultiply(
                    leftBuffer, rightBuffer, outputBuffer, numPairs);
                long directDispatchesBefore = backend.DirectPtxComplexMultiplyDispatchCount;
                Distribution device = MeasureDevice(backend, launch);
                Distribution endToEnd = MeasureEndToEnd(backend.Synchronize, launch);
                long managedBytes = MeasureAllocation(backend.Synchronize, launch);
                launch();
                backend.Synchronize();
                if (backend.DirectPtxComplexMultiplyDispatchCount != directDispatchesBefore)
                    throw new InvalidOperationException(
                        "The established lane unexpectedly entered the direct-PTX route.");
                var actual = new float[numPairs * 2];
                backend.DownloadBuffer(outputBuffer, actual);
                float error = Validate(actual, left, right, numPairs);
                results.Add(CreateResult(
                    run, numPairs, "AiDotNet NVRTC", device, endToEnd,
                    managedBytes, 0, error, -1, -1, -1, -1));
            }
        }
        finally
        {
            DirectPtxFeatureGate.ComplexMultiplyGateOverride = originalGate;
        }
    }

    private static void RunOracle(int run, List<OracleResult> results)
    {
        using var direct = new CudaBackend();
        using var incumbent = new CudaBackend();
        if (!direct.IsAvailable || !incumbent.IsAvailable)
        {
            Console.Error.WriteLine(
                "[complex-multiply] CUDA backend unavailable; the oracle lane produced no rows.");
            return;
        }

        bool originalExperiment = DirectPtxFeatureGate.ComplexMultiplyExperimentOverride;
        bool? originalGate = DirectPtxFeatureGate.ComplexMultiplyGateOverride;
        try
        {
            DirectPtxFeatureGate.ComplexMultiplyExperimentOverride = true;
            foreach (int numPairs in Shapes)
            {
                DirectPtxFeatureGate.ComplexMultiplyGateOverride = true;
                if (!direct.PrewarmDirectPtxComplexMultiply(numPairs))
                    throw new InvalidOperationException(
                        $"Direct PTX oracle prewarm failed for {numPairs} pairs: " +
                        (direct.DirectPtxLastError ?? "no diagnostic"));

                float[] left = Values(numPairs * 2, 1000 + numPairs);
                float[] right = Values(numPairs * 2, 2000 + numPairs);
                using var directLeft = direct.AllocateBuffer(left);
                using var directRight = direct.AllocateBuffer(right);
                using var directOutput = direct.AllocateBuffer(numPairs * 2);
                using var incumbentLeft = incumbent.AllocateBuffer(left);
                using var incumbentRight = incumbent.AllocateBuffer(right);
                using var incumbentOutput = incumbent.AllocateBuffer(numPairs * 2);

                void DirectLaunch()
                {
                    DirectPtxFeatureGate.ComplexMultiplyGateOverride = true;
                    direct.ComplexMultiply(
                        directLeft, directRight, directOutput, numPairs);
                }

                void IncumbentLaunch()
                {
                    DirectPtxFeatureGate.ComplexMultiplyGateOverride = false;
                    incumbent.ComplexMultiply(
                        incumbentLeft, incumbentRight, incumbentOutput, numPairs);
                }

                long directDispatchesBefore =
                    direct.DirectPtxComplexMultiplyDispatchCount;
                long incumbentDispatchesBefore =
                    incumbent.DirectPtxComplexMultiplyDispatchCount;
                DirectLaunch();
                direct.Synchronize();
                IncumbentLaunch();
                incumbent.Synchronize();
                var directActual = new float[left.Length];
                var incumbentActual = new float[left.Length];
                direct.DownloadBuffer(directOutput, directActual);
                incumbent.DownloadBuffer(incumbentOutput, incumbentActual);
                double directError = Validate(directActual, left, right, numPairs);
                double incumbentError = Validate(incumbentActual, left, right, numPairs);
                if (!direct.TryGetDirectPtxComplexMultiplyAudit(numPairs, out var audit))
                    throw new InvalidOperationException(
                        "The prewarmed direct-PTX module has no audit record.");

                bool correctnessPassed =
                    IsFinite(directError) && IsFinite(incumbentError) &&
                    directError <= SemanticTolerance &&
                    incumbentError <= SemanticTolerance;
                StableTimer.PairResult timing = default;
                string lane = "correctness-rejected";

                if (correctnessPassed)
                {
                    IntPtr directGraph = IntPtr.Zero;
                    IntPtr incumbentGraph = IntPtr.Zero;
                    try
                    {
                        directGraph = CaptureRepeatedGraph(
                            direct, DirectLaunch, ThroughputOperationsPerGraph);
                        incumbentGraph = CaptureRepeatedGraph(
                            incumbent, IncumbentLaunch, ThroughputOperationsPerGraph);
                        if (directGraph != IntPtr.Zero && incumbentGraph != IntPtr.Zero)
                        {
                            lane = "repeated-cuda-graph-host-paired";
                            timing = StableTimer.MeasureCalibratedHostPair(
                                () => direct.EnqueueCapturedGraph(directGraph),
                                direct.Synchronize,
                                () => incumbent.EnqueueCapturedGraph(incumbentGraph),
                                incumbent.Synchronize,
                                operationsPerLaunchA: ThroughputOperationsPerGraph,
                                operationsPerLaunchB: ThroughputOperationsPerGraph,
                                targetBatchMilliseconds: 20.0);
                        }
                        else
                        {
                            lane = "public-launch-host-paired-capture-fallback";
                            timing = StableTimer.MeasureCalibratedHostPair(
                                DirectLaunch, direct.Synchronize,
                                IncumbentLaunch, incumbent.Synchronize,
                                targetBatchMilliseconds: 20.0);
                        }
                    }
                    finally
                    {
                        // Cleanup runs on the exception path too. Letting a
                        // DestroyCapturedGraph failure escape here would replace the
                        // original capture or benchmark exception with a less useful
                        // one, so report it and preserve the primary failure.
                        try { direct.DestroyCapturedGraph(directGraph); }
                        catch (Exception ex)
                        {
                            Console.Error.WriteLine(
                                "[complex-multiply] candidate graph cleanup failed: " + ex.Message);
                        }
                        try { incumbent.DestroyCapturedGraph(incumbentGraph); }
                        catch (Exception ex)
                        {
                            Console.Error.WriteLine(
                                "[complex-multiply] established graph cleanup failed: " + ex.Message);
                        }
                    }
                }

                if (direct.DirectPtxComplexMultiplyDispatchCount <=
                    directDispatchesBefore)
                    throw new InvalidOperationException(
                        "The candidate oracle lane did not enter direct PTX.");
                if (incumbent.DirectPtxComplexMultiplyDispatchCount !=
                    incumbentDispatchesBefore)
                    throw new InvalidOperationException(
                        "The incumbent oracle lane unexpectedly entered direct PTX.");

                results.Add(new OracleResult(
                    run, numPairs, lane, timing, directError, incumbentError, audit));
            }
        }
        finally
        {
            DirectPtxFeatureGate.ComplexMultiplyGateOverride = originalGate;
            DirectPtxFeatureGate.ComplexMultiplyExperimentOverride = originalExperiment;
        }
    }

    private static IntPtr CaptureRepeatedGraph(
        CudaBackend backend, Action launch, int operations)
    {
        return backend.CaptureGraph(() =>
        {
            for (int i = 0; i < operations; i++) launch();
        });
    }

    private static Result CreateResult(
        int run,
        int numPairs,
        string method,
        Distribution device,
        Distribution endToEnd,
        long managedBytes,
        long temporaryDeviceBytes,
        float error,
        int registers,
        int sharedBytes,
        int localBytes,
        int activeBlocks)
    {
        const double flopsPerPair = 6.0;
        double seconds = device.Median * 1e-6;
        double gflops = flopsPerPair * numPairs / seconds / 1e9;
        double bytes = 3.0 * numPairs * 2 * sizeof(float);
        double gbps = bytes / seconds / 1e9;
        return new Result(
            run, numPairs, method, device, endToEnd, gflops, gbps,
            managedBytes, temporaryDeviceBytes, error, registers, sharedBytes,
            localBytes, activeBlocks);
    }

    private static Distribution MeasureDevice(CudaBackend backend, Action launch)
    {
        for (int i = 0; i < Warmups; i++) launch();
        backend.Synchronize();
        var samples = new double[Samples];
        using IGpuEvent start = backend.CreateEvent(enableTiming: true);
        using IGpuEvent end = backend.CreateEvent(enableTiming: true);
        for (int sample = 0; sample < samples.Length; sample++)
        {
            backend.RecordEvent(start, backend.DefaultStream);
            for (int i = 0; i < LaunchesPerDeviceSample; i++) launch();
            backend.RecordEvent(end, backend.DefaultStream);
            end.Synchronize();
            samples[sample] = backend.GetEventElapsedTime(start, end) * 1000.0 /
                LaunchesPerDeviceSample;
        }
        return Summarize(samples);
    }

    private static Distribution MeasureEndToEnd(Action synchronize, Action launch)
    {
        for (int i = 0; i < Warmups; i++) launch();
        synchronize();
        var samples = new double[Samples];
        double tickToMicroseconds = 1_000_000.0 / Stopwatch.Frequency;
        for (int i = 0; i < samples.Length; i++)
        {
            long start = Stopwatch.GetTimestamp();
            launch();
            synchronize();
            samples[i] = (Stopwatch.GetTimestamp() - start) * tickToMicroseconds;
        }
        return Summarize(samples);
    }

    private static long MeasureAllocation(Action synchronize, Action launch)
    {
        for (int i = 0; i < 8; i++) launch();
        synchronize();
        long before = GC.GetAllocatedBytesForCurrentThread();
        for (int i = 0; i < Samples; i++) launch();
        long bytes = (GC.GetAllocatedBytesForCurrentThread() - before) / Samples;
        synchronize();
        return bytes;
    }

    private static Distribution Summarize(double[] samples)
    {
        Array.Sort(samples);
        return new Distribution(
            samples.Average(), Percentile(samples, .50),
            Percentile(samples, .95), Percentile(samples, .99));
    }

    private static double Percentile(double[] sorted, double percentile)
    {
        double position = (sorted.Length - 1) * percentile;
        int lower = (int)position;
        int upper = Math.Min(lower + 1, sorted.Length - 1);
        return sorted[lower] + (sorted[upper] - sorted[lower]) * (position - lower);
    }

    private static float[] Values(int length, int seed)
    {
        var random = new Random(seed);
        return Enumerable.Range(0, length)
            .Select(_ => (random.NextSingle() * 2f - 1f) * 2f).ToArray();
    }

    private static float Validate(
        float[] actual,
        float[] left,
        float[] right,
        int numPairs)
    {
        float maximum = 0;
        for (int pair = 0; pair < numPairs; pair++)
        {
            int offset = pair * 2;
            double ar = left[offset], ai = left[offset + 1];
            double br = right[offset], bi = right[offset + 1];
            float expectedReal = (float)(ar * br - ai * bi);
            float expectedImaginary = (float)(ar * bi + ai * br);
            maximum = MathF.Max(maximum, MathF.Abs(actual[offset] - expectedReal));
            maximum = MathF.Max(maximum, MathF.Abs(actual[offset + 1] - expectedImaginary));
        }
        return maximum;
    }

    private static void Print(IReadOnlyList<Result> results)
    {
        Console.WriteLine(
            $"{"Run",3} {"Pairs",9} {"Method",-18} {"dev med",9} {"dev p95",9} {"dev p99",9} " +
            $"{"E2E med",9} {"E2E p95",9} {"E2E p99",9} {"GFLOPS",9} {"GB/s",9} " +
            $"{"managed",9} {"temp B",9} {"max err",10} {"regs",5} {"shared",7} {"local",5} {"occ",4}");
        Console.WriteLine(new string('-', 177));
        foreach (Result result in results.OrderBy(r => r.Run)
                     .ThenBy(r => r.NumPairs).ThenBy(r => r.Device.Median))
        {
            Console.WriteLine(
                $"{result.Run,3} {result.NumPairs,9} {result.Method,-18} " +
                $"{result.Device.Median,9:F2} {result.Device.P95,9:F2} {result.Device.P99,9:F2} " +
                $"{result.EndToEnd.Median,9:F2} {result.EndToEnd.P95,9:F2} {result.EndToEnd.P99,9:F2} " +
                $"{result.Gflops,9:F2} {result.GigabytesPerSecond,9:F2} {result.ManagedBytes,9} " +
                $"{result.TemporaryDeviceBytes,9} {result.MaxError,10:G4} {Dash(result.Registers),5} " +
                $"{Dash(result.SharedBytes),7} {Dash(result.LocalBytes),5} {Dash(result.ActiveBlocks),4}");
            Console.WriteLine("complex_multiply_evidence_json=" + JsonSerializer.Serialize(new
            {
                status = "ok",
                run = result.Run,
                pairs = result.NumPairs,
                method = result.Method,
                device_mean_us = result.Device.Mean,
                device_median_us = result.Device.Median,
                device_p95_us = result.Device.P95,
                device_p99_us = result.Device.P99,
                e2e_mean_us = result.EndToEnd.Mean,
                e2e_median_us = result.EndToEnd.Median,
                e2e_p95_us = result.EndToEnd.P95,
                e2e_p99_us = result.EndToEnd.P99,
                gflops = result.Gflops,
                effective_gbps = result.GigabytesPerSecond,
                managed_bytes = result.ManagedBytes,
                temporary_device_bytes = result.TemporaryDeviceBytes,
                max_error = result.MaxError,
                registers_per_thread = result.Registers,
                static_shared_bytes = result.SharedBytes,
                local_bytes_per_thread = result.LocalBytes,
                active_blocks_per_sm = result.ActiveBlocks
            }));
        }
    }

    private static void PrintOracle(IReadOnlyList<OracleResult> results)
    {
        foreach (OracleResult result in results.OrderBy(r => r.Run)
                     .ThenBy(r => r.NumPairs))
        {
            bool correctnessPassed =
                IsFinite(result.DirectError) && IsFinite(result.IncumbentError) &&
                result.DirectError <= SemanticTolerance &&
                result.IncumbentError <= SemanticTolerance;
            bool measurable = correctnessPassed && result.Timing.Stable;
            double incumbentOverDirect = measurable
                ? 1.0 / result.Timing.Ratio
                : double.NaN;
            string measurementStatus = !correctnessPassed
                ? "correctness-failed"
                : !result.Timing.Stable
                    ? "unstable"
                    : "stable";
            string verdict = !measurable
                ? "not-measurable"
                : incumbentOverDirect >= RequiredGain
                    ? "win"
                    : incumbentOverDirect <= 1.0 / RequiredGain
                        ? "loss"
                        : "tie";

            Console.WriteLine(
                $"oracle | {result.Run} | complex-multiply | pairs={result.NumPairs} | " +
                $"lane={result.Lane} | direct={result.Timing.A.Describe()} | " +
                $"incumbent={result.Timing.B.Describe()} | " +
                $"incumbent/direct={(measurable ? incumbentOverDirect.ToString("0.00") + "x" : "-")} | " +
                $"correctness={measurementStatus} " +
                $"(direct={result.DirectError:E3}, incumbent={result.IncumbentError:E3}, " +
                $"tolerance={SemanticTolerance:E1}) | {verdict}");
            Console.WriteLine("complex_multiply_oracle_json=" +
                JsonSerializer.Serialize(new
                {
                    kind = "direct-ptx-complex-multiply-oracle",
                    run = result.Run,
                    operation = "complex-multiply",
                    pairs = result.NumPairs,
                    lane = result.Lane,
                    logical_operations_per_graph = result.Lane.StartsWith(
                        "repeated-cuda-graph", StringComparison.Ordinal)
                            ? ThroughputOperationsPerGraph
                            : 1,
                    direct_median_us = result.Timing.A.Stable
                        ? result.Timing.A.Microseconds
                        : (double?)null,
                    incumbent_median_us = result.Timing.B.Stable
                        ? result.Timing.B.Microseconds
                        : (double?)null,
                    incumbent_over_direct = measurable
                        ? incumbentOverDirect
                        : (double?)null,
                    direct_relative_spread = FiniteOrNull(
                        result.Timing.A.RelativeSpread),
                    incumbent_relative_spread = FiniteOrNull(
                        result.Timing.B.RelativeSpread),
                    paired_ratio_relative_spread = FiniteOrNull(
                        result.Timing.RelativeSpread),
                    attempts = result.Timing.Samples,
                    required_gain = RequiredGain,
                    direct_semantic_error = FiniteOrNull(result.DirectError),
                    incumbent_semantic_error = FiniteOrNull(result.IncumbentError),
                    semantic_tolerance = SemanticTolerance,
                    correctness_passed = correctnessPassed,
                    measurement_status = measurementStatus,
                    verdict,
                    registers_per_thread = result.Audit.Function.RegistersPerThread,
                    static_shared_bytes = result.Audit.Function.StaticSharedBytes,
                    local_bytes_per_thread = result.Audit.Function.LocalBytesPerThread,
                    active_blocks_per_sm = result.Audit.ActiveBlocksPerMultiprocessor,
                    diagnostic_only = true,
                    promotion = false
                }));
        }
    }

    private static bool IsFinite(double value) =>
        !double.IsNaN(value) && !double.IsInfinity(value);

    private static double? FiniteOrNull(double value) =>
        IsFinite(value) ? value : null;

    private static string Dash(int value) => value < 0 ? "-" : value.ToString();
}
