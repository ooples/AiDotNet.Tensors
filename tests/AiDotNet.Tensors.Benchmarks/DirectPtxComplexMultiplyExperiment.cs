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

    internal static void Run(int independentRuns = 3)
    {
        if (independentRuns <= 0)
            throw new ArgumentOutOfRangeException(nameof(independentRuns));
        var results = new List<Result>();
        for (int run = 1; run <= independentRuns; run++)
        {
            GpuBenchmarkEnvironment.PrintSnapshot($"complex-multiply-start-{run}");
            RunDirect(run, results);
            RunEstablished(run, results);
            GpuBenchmarkEnvironment.PrintSnapshot($"complex-multiply-end-{run}");
        }
        Print(results);
        Console.WriteLine("Production verdict remains HOLD until the companion PyTorch CUDA records and Nsight evidence are joined with these results.");
    }

    private static void RunDirect(int run, List<Result> results)
    {
        using var backend = new CudaBackend();
        if (!backend.IsAvailable) return;
        bool originalExperiment = DirectPtxFeatureGate.ComplexMultiplyExperimentOverride;
        bool? originalGate = DirectPtxFeatureGate.ComplexMultiplyGateOverride;
        try
        {
            DirectPtxFeatureGate.ComplexMultiplyExperimentOverride = true;
            DirectPtxFeatureGate.ComplexMultiplyGateOverride = true;
            foreach (int numPairs in Shapes)
            {
                if (!backend.PrewarmDirectPtxComplexMultiply(numPairs))
                    continue;
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
        if (!backend.IsAvailable) return;
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

    private static string Dash(int value) => value < 0 ? "-" : value.ToString();
}
