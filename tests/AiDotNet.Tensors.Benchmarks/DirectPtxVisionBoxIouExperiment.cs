using System.Diagnostics;
using System.Text.Json;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;
using AiDotNet.Tensors.Engines.Gpu;

namespace AiDotNet.Tensors.Benchmarks;

/// <summary>Issue #851 resident pairwise BoxIoU evidence matrix.</summary>
internal static class DirectPtxVisionBoxIouExperiment
{
    private const int Warmups = 30;
    private const int Samples = 101;
    private const int LaunchesPerDeviceSample = 25;
    private const double OperationsPerCell = 20.0;
    private const double AlgorithmicBytesPerCell = 36.0;

    private readonly record struct Shape(string Name, int N, int M);
    private readonly record struct Distribution(double Mean, double Median, double P95, double P99);
    private readonly record struct Evidence(
        int Run, Shape Shape, string Method,
        Distribution Device, Distribution EndToEnd,
        long ManagedBytes, long TemporaryDeviceBytes,
        double MaximumError, DirectPtxKernelAudit? Audit);

    private static readonly Shape[] Shapes =
    [
        new("n256-m256", 256, 256),
        new("n1024-m256", 1024, 256),
        new("n1024-m1024", 1024, 1024),
        new("n4096-m256", 4096, 256)
    ];

    internal static void Run(int independentRuns = 3)
    {
        if (independentRuns <= 0) throw new ArgumentOutOfRangeException(nameof(independentRuns));
        bool? oldExperiment = DirectPtxFeatureGate.VisionBoxIouExperimentOverride;
        bool? oldRoute = DirectPtxFeatureGate.VisionBoxIouGateOverride;
        DirectPtxFeatureGate.VisionBoxIouExperimentOverride = true;
        try
        {
            Console.WriteLine(
                $"Direct-PTX pairwise BoxIoU: {independentRuns} runs, {Warmups} warmups, " +
                $"{Samples} samples; device samples average {LaunchesPerDeviceSample} launches.");
            Console.WriteLine("All inputs, outputs, modules, and CUDA graphs are resident before timing.");
            PrintHeader();
            for (int run = 1; run <= independentRuns; run++)
            {
                GpuBenchmarkEnvironment.RequireNoForeignCompute(
                    $"vision-box-iou-run-{run}-start");
                using var backend = new CudaBackend();
                Console.WriteLine($"environment_json={JsonSerializer.Serialize(new
                {
                    run,
                    gpu = backend.DeviceName,
                    framework = Environment.Version.ToString(),
                    os = Environment.OSVersion.ToString(),
                    warmups = Warmups,
                    samples = Samples,
                    launches_per_device_sample = LaunchesPerDeviceSample
                })}");
                foreach (Shape shape in Shapes)
                    foreach (Evidence cell in RunCell(backend, run, shape)) Print(cell);
                backend.Synchronize();
                GpuBenchmarkEnvironment.RequireNoForeignCompute(
                    $"vision-box-iou-run-{run}-end");
            }
            Console.WriteLine(
                "HOLD: run the companion PyTorch harness and Nsight verifier; no shape is promoted by this command alone.");
        }
        finally
        {
            DirectPtxFeatureGate.VisionBoxIouGateOverride = oldRoute;
            DirectPtxFeatureGate.VisionBoxIouExperimentOverride = oldExperiment;
        }
    }

    private static IReadOnlyList<Evidence> RunCell(CudaBackend backend, int run, Shape shape)
    {
        var random = new Random(20260722 + run * 1000 + shape.N + shape.M);
        float[] boxesA = Boxes(random, shape.N);
        float[] boxesB = Boxes(random, shape.M);
        using var a = backend.AllocateBuffer(boxesA);
        using var b = backend.AllocateBuffer(boxesB);
        using var directOutput = backend.AllocateBuffer(checked(shape.N * shape.M));
        using var currentOutput = backend.AllocateBuffer(checked(shape.N * shape.M));

        DirectPtxFeatureGate.VisionBoxIouGateOverride = true;
        if (!backend.PrewarmDirectPtxVisionBoxIou(shape.N, shape.M))
            throw new InvalidOperationException(
                $"Direct PTX prewarm failed: {backend.DirectPtxLastError ?? "unknown"}");
        void DirectLaunch() => backend.BoxIou(a, b, directOutput, shape.N, shape.M);
        DirectLaunch();
        backend.Synchronize();
        long directDispatches = backend.DirectPtxVisionBoxIouDispatchCount;
        if (!backend.TryGetDirectPtxVisionBoxIouAudit(shape.N, shape.M, out var audit))
            throw new InvalidOperationException("No audit was retained for the measured direct-PTX module.");

        DirectPtxFeatureGate.VisionBoxIouGateOverride = false;
        void CurrentLaunch() => backend.BoxIou(a, b, currentOutput, shape.N, shape.M);
        CurrentLaunch();
        backend.Synchronize();
        if (backend.DirectPtxVisionBoxIouDispatchCount != directDispatches)
            throw new InvalidOperationException("The established AiDotNet cell routed through direct PTX.");

        float[] directHost = backend.DownloadBuffer(directOutput);
        float[] currentHost = backend.DownloadBuffer(currentOutput);
        double directError = MaximumError(boxesA, boxesB, directHost, shape.N, shape.M);
        double currentError = MaximumError(boxesA, boxesB, currentHost, shape.N, shape.M);
        if (!double.IsFinite(directError) || directError > 2e-6)
            throw new InvalidOperationException(
                $"Direct PTX BoxIoU error {directError:G9} exceeds the 2e-6 oracle gate.");
        if (!double.IsFinite(currentError) || currentError > 2e-6)
            throw new InvalidOperationException(
                $"Established BoxIoU error {currentError:G9} exceeds the 2e-6 oracle gate.");

        DirectPtxFeatureGate.VisionBoxIouGateOverride = true;
        IntPtr directGraph = backend.CaptureGraph(DirectLaunch);
        DirectPtxFeatureGate.VisionBoxIouGateOverride = false;
        IntPtr currentGraph = backend.CaptureGraph(CurrentLaunch);
        if (directGraph == IntPtr.Zero || currentGraph == IntPtr.Zero)
            throw new InvalidOperationException("Both public routes must be CUDA-graph capturable after prewarm.");
        try
        {
            void DirectGraphLaunch() => backend.EnqueueCapturedGraph(directGraph);
            void CurrentGraphLaunch() => backend.EnqueueCapturedGraph(currentGraph);
            var cells = new List<Evidence>(4);
            DirectPtxFeatureGate.VisionBoxIouGateOverride = true;
            cells.Add(Measure(backend, run, shape, "Direct PTX", DirectLaunch, 0, directError, audit));
            cells.Add(Measure(backend, run, shape, "Direct PTX CUDA graph", DirectGraphLaunch, 0, directError, audit));
            DirectPtxFeatureGate.VisionBoxIouGateOverride = false;
            cells.Add(Measure(backend, run, shape, "AiDotNet NVRTC", CurrentLaunch, 0, currentError, null));
            cells.Add(Measure(backend, run, shape, "AiDotNet NVRTC graph", CurrentGraphLaunch, 0, currentError, null));
            return cells;
        }
        finally
        {
            backend.DestroyCapturedGraph(directGraph);
            backend.DestroyCapturedGraph(currentGraph);
        }
    }

    private static Evidence Measure(
        CudaBackend backend, int run, Shape shape, string method, Action action,
        long temporaryDeviceBytes, double error, DirectPtxKernelAudit? audit) =>
        new(run, shape, method,
            MeasureDevice(backend, action), MeasureEndToEnd(backend, action),
            MeasureAllocation(backend, action), temporaryDeviceBytes, error, audit);

    private static Distribution MeasureDevice(CudaBackend backend, Action action)
    {
        for (int i = 0; i < Warmups; i++) action();
        backend.Synchronize();
        var values = new double[Samples];
        using IGpuEvent start = backend.CreateEvent(enableTiming: true);
        using IGpuEvent end = backend.CreateEvent(enableTiming: true);
        for (int i = 0; i < values.Length; i++)
        {
            backend.RecordEvent(start, backend.DefaultStream);
            for (int launch = 0; launch < LaunchesPerDeviceSample; launch++) action();
            backend.RecordEvent(end, backend.DefaultStream);
            end.Synchronize();
            values[i] = backend.GetEventElapsedTime(start, end) * 1000.0 / LaunchesPerDeviceSample;
        }
        return Summarize(values);
    }

    private static Distribution MeasureEndToEnd(CudaBackend backend, Action action)
    {
        for (int i = 0; i < Warmups; i++) action();
        backend.Synchronize();
        var values = new double[Samples];
        double toMicroseconds = 1_000_000.0 / Stopwatch.Frequency;
        for (int i = 0; i < values.Length; i++)
        {
            long start = Stopwatch.GetTimestamp();
            action();
            backend.Synchronize();
            values[i] = (Stopwatch.GetTimestamp() - start) * toMicroseconds;
        }
        return Summarize(values);
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

    private static Distribution Summarize(double[] values)
    {
        Array.Sort(values);
        return new(values.Average(), Percentile(values, 0.50),
            Percentile(values, 0.95), Percentile(values, 0.99));
    }

    private static double Percentile(double[] sorted, double percentile)
    {
        double position = percentile * (sorted.Length - 1);
        int lower = (int)Math.Floor(position);
        int upper = (int)Math.Ceiling(position);
        return sorted[lower] + (sorted[upper] - sorted[lower]) * (position - lower);
    }

    private static void PrintHeader()
    {
        Console.WriteLine(
            $"{"Run",3} {"Shape",-13} {"Method",-24} {"dev mean",9} {"dev med",9} " +
            $"{"dev p95",9} {"dev p99",9} {"E2E med",9} {"GFLOPS",9} {"GB/s",9} " +
            $"{"managed B",10} {"temp B",8} {"max err",10} {"regs",5} {"shared",7} {"local",5} {"blocks/SM",9}");
    }

    private static void Print(Evidence cell)
    {
        double cells = (double)cell.Shape.N * cell.Shape.M;
        double gflops = cells * OperationsPerCell / (cell.Device.Median * 1000.0);
        double bandwidth = cells * AlgorithmicBytesPerCell / (cell.Device.Median * 1000.0);
        int registers = cell.Audit?.Function.RegistersPerThread ?? -1;
        int shared = cell.Audit?.Function.StaticSharedBytes ?? -1;
        int local = cell.Audit?.Function.LocalBytesPerThread ?? -1;
        int blocks = cell.Audit?.ActiveBlocksPerMultiprocessor ?? -1;
        Console.WriteLine(
            $"{cell.Run,3} {cell.Shape.Name,-13} {cell.Method,-24} " +
            $"{cell.Device.Mean,9:F2} {cell.Device.Median,9:F2} {cell.Device.P95,9:F2} {cell.Device.P99,9:F2} " +
            $"{cell.EndToEnd.Median,9:F2} {gflops,9:F2} {bandwidth,9:F2} " +
            $"{cell.ManagedBytes,10} {cell.TemporaryDeviceBytes,8} {cell.MaximumError,10:G4} " +
            $"{Dash(registers),5} {Dash(shared),7} {Dash(local),5} {Dash(blocks),9}");
        Console.WriteLine("vision_iou_evidence_json=" + JsonSerializer.Serialize(new
        {
            status = "ok",
            run = cell.Run,
            shape = cell.Shape.Name,
            method = cell.Method,
            device_mean_us = cell.Device.Mean,
            device_median_us = cell.Device.Median,
            device_p95_us = cell.Device.P95,
            device_p99_us = cell.Device.P99,
            e2e_mean_us = cell.EndToEnd.Mean,
            e2e_median_us = cell.EndToEnd.Median,
            e2e_p95_us = cell.EndToEnd.P95,
            e2e_p99_us = cell.EndToEnd.P99,
            gflops,
            algorithmic_gbps = bandwidth,
            managed_bytes = cell.ManagedBytes,
            temporary_device_bytes = cell.TemporaryDeviceBytes,
            max_error = cell.MaximumError,
            registers_per_thread = registers,
            static_shared_bytes = shared,
            dynamic_shared_bytes = cell.Audit is null ? (int?)null : 0,
            local_bytes_per_thread = local,
            active_blocks_per_sm = blocks,
            max_threads_per_block = cell.Audit?.Function.MaxThreadsPerBlock,
            ptx_version = cell.Audit?.Function.PtxVersion,
            binary_version = cell.Audit?.Function.BinaryVersion,
            blueprint = cell.Audit?.BlueprintId,
            ptx_sha256 = cell.Audit?.PtxSha256,
            device_fingerprint = cell.Audit?.DeviceFingerprint,
            jit_info_log = cell.Audit?.JitInfoLog
        }));
    }

    private static string Dash(int value) => value < 0 ? "-" : value.ToString();

    private static float[] Boxes(Random random, int count)
    {
        var boxes = new float[count * 4];
        for (int i = 0; i < count; i++)
        {
            float x = random.NextSingle() * 512f;
            float y = random.NextSingle() * 512f;
            boxes[i * 4] = x;
            boxes[i * 4 + 1] = y;
            boxes[i * 4 + 2] = x + random.NextSingle() * 128f;
            boxes[i * 4 + 3] = y + random.NextSingle() * 128f;
        }
        return boxes;
    }

    private static double MaximumError(float[] a, float[] b, float[] output, int n, int m)
    {
        double maximum = 0;
        for (int i = 0; i < n; i++)
        for (int j = 0; j < m; j++)
        {
            double areaA = Math.Max((double)a[i * 4 + 2] - a[i * 4], 0) *
                Math.Max((double)a[i * 4 + 3] - a[i * 4 + 1], 0);
            double areaB = Math.Max((double)b[j * 4 + 2] - b[j * 4], 0) *
                Math.Max((double)b[j * 4 + 3] - b[j * 4 + 1], 0);
            double intersection =
                Math.Max(Math.Min((double)a[i * 4 + 2], b[j * 4 + 2]) -
                    Math.Max((double)a[i * 4], b[j * 4]), 0) *
                Math.Max(Math.Min((double)a[i * 4 + 3], b[j * 4 + 3]) -
                    Math.Max((double)a[i * 4 + 1], b[j * 4 + 1]), 0);
            double union = areaA + areaB - intersection;
            double expected = union > 0 ? intersection / union : 0;
            maximum = Math.Max(maximum, Math.Abs(expected - output[i * m + j]));
        }
        return maximum;
    }
}
