using System.Diagnostics;
using System.Globalization;
using System.Text.Json;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;
using AiDotNet.Tensors.Engines.Gpu;

namespace AiDotNet.Tensors.Benchmarks;

/// <summary>Issue #849 resident NVIDIA dropout championship scaffold.</summary>
internal static class DirectPtxRngDropoutExperiment
{
    private const int Warmups = 30;
    private const int Samples = 101;
    private const int DeviceLaunches = 10;
    private const int ThroughputOperationsPerGraph = 64;
    private const double RequiredGain = 1.10;
    private const float DropoutRate = 0.1f;
    private const ulong Seed = 0x8490_1234_5678_9ABCul;

    private readonly record struct Shape(string Name, int Elements);
    private readonly record struct Distribution(double Mean, double Median, double P95, double P99);
    private readonly record struct OracleEvidence(
        StableTimer.PairResult Timing,
        double DirectError,
        double IncumbentError);
    private readonly record struct Cell(
        int Run,
        Shape Shape,
        string Method,
        Distribution Device,
        Distribution EndToEnd,
        double GbPerSecond,
        long ManagedBytes,
        long TemporaryDeviceBytes,
        double MaximumError,
        DirectPtxKernelAudit? Audit,
        OracleEvidence? PairedEvidence);
    private sealed record PythonCell(
        string Status,
        int Run,
        string Shape,
        string Method,
        double DeviceMeanUs,
        double DeviceMedianUs,
        double DeviceP95Us,
        double DeviceP99Us,
        double EndToEndMeanUs,
        double EndToEndMedianUs,
        double EndToEndP95Us,
        double EndToEndP99Us,
        double GbPerSecond,
        long PeakDeviceBytes,
        double MaxError);

    private static readonly Shape[] Shapes =
    [
        new("dropout-n4096", 4_096),
        new("dropout-n65536", 65_536),
        new("dropout-n1048576", 1_048_576)
    ];

    internal static void Run(int independentRuns = 3, bool includeExternal = true)
    {
        if (independentRuns <= 0) throw new ArgumentOutOfRangeException(nameof(independentRuns));
        GpuBenchmarkEnvironment.RequireIdleGpu("direct-ptx-rng-dropout-start");
        Console.WriteLine(
            $"Direct PTX Philox dropout: {independentRuns} clean run(s), {Warmups} warmups + " +
            $"{Samples} samples/cell; {DeviceLaunches} resident launches/device sample.");
        var cells = new List<Cell>();
        for (int run = 1; run <= independentRuns; run++)
        {
            CudaBackend directBackend;
            CudaBackend establishedBackend;
            try
            {
                DirectPtxFeatureGate.TestOverride = true;
                directBackend = new CudaBackend();
                DirectPtxFeatureGate.TestOverride = false;
                establishedBackend = new CudaBackend();
            }
            finally
            {
                DirectPtxFeatureGate.TestOverride = null;
            }
            using (directBackend)
            using (establishedBackend)
            {
                if (!directBackend.IsDirectPtxRngDropoutEnabled)
                    throw new InvalidOperationException(
                        "The exact admitted SM is required for the direct-PTX contender.");
                if (establishedBackend.IsDirectPtxRngDropoutEnabled)
                    throw new InvalidOperationException(
                        "The AiDotNet baseline backend unexpectedly admitted direct PTX.");
                if (run == 1) Console.WriteLine($"GPU: {directBackend.DeviceName}");
                foreach (Shape shape in Shapes)
                    cells.AddRange(RunShape(directBackend, establishedBackend, run, shape));
            }
        }

        IReadOnlyList<PythonCell> python = includeExternal
            ? RunPython(independentRuns)
            : Array.Empty<PythonCell>();
        Print(cells, python);
        GpuBenchmarkEnvironment.RequireNoForeignCompute("direct-ptx-rng-dropout-end");
    }

    private static IEnumerable<Cell> RunShape(
        CudaBackend backend,
        CudaBackend establishedBackend,
        int run,
        Shape shape)
    {
        var random = new Random(849_000 + run * 10_000 + shape.Elements);
        float[] inputHost = Enumerable.Range(0, shape.Elements)
            .Select(_ => (float)(random.NextDouble() * 2.0 - 1.0)).ToArray();
        using var input = backend.AllocateBuffer(inputHost);
        using var establishedInput = establishedBackend.AllocateBuffer(inputHost);
        using var directOutput = backend.AllocateBuffer(shape.Elements);
        using var directMask = backend.AllocateBuffer(shape.Elements);
        using var currentOutput = establishedBackend.AllocateBuffer(shape.Elements);
        using var currentMask = establishedBackend.AllocateBuffer(shape.Elements);
        if (!backend.PrewarmDirectPtxRngDropoutF32(shape.Elements))
            throw new InvalidOperationException(backend.DirectPtxLastError ?? "PTX prewarm failed.");

        void DirectLaunch() => backend.Dropout(
            input, directOutput, directMask, shape.Elements, DropoutRate, Seed, training: true);
        void CurrentLaunch()
        {
            establishedBackend.DropoutMask(
                currentMask, shape.Elements, 1.0f - DropoutRate, Seed);
            establishedBackend.Multiply(
                establishedInput, currentMask, currentOutput, shape.Elements);
        }

        long publicDispatchBefore = backend.DirectPtxRngDropoutDispatchCount;
        DirectLaunch();
        CurrentLaunch();
        backend.Synchronize();
        establishedBackend.Synchronize();
        if (backend.DirectPtxRngDropoutDispatchCount != publicDispatchBefore + 1 ||
            backend.DirectPtxLastError is not null)
            throw new InvalidOperationException(
                $"The public dropout route did not dispatch direct PTX exclusively: " +
                $"count={backend.DirectPtxRngDropoutDispatchCount - publicDispatchBefore}; " +
                $"error={backend.DirectPtxLastError ?? "none"}.");
        double directError = DirectOracleError(
            backend, inputHost, directOutput, directMask, shape.Elements);
        double currentError = MaskApplicationError(
            establishedBackend, inputHost, currentOutput, currentMask);
        if (!backend.TryGetDirectPtxRngDropoutAudit(shape.Elements, out DirectPtxKernelAudit audit))
            throw new InvalidOperationException("The prewarmed PTX module has no audit record.");
        Console.WriteLine($"audit run={run} shape={shape.Name}: {audit.ToJson()}");

        IntPtr graph = IntPtr.Zero;
        IntPtr directThroughputGraph = IntPtr.Zero;
        IntPtr currentThroughputGraph = IntPtr.Zero;
        try
        {
            graph = backend.CaptureGraph(DirectLaunch);
            directThroughputGraph = CaptureRepeatedGraph(
                backend, DirectLaunch, ThroughputOperationsPerGraph);
            currentThroughputGraph = CaptureRepeatedGraph(
                establishedBackend, CurrentLaunch, ThroughputOperationsPerGraph);
            if (graph == IntPtr.Zero || directThroughputGraph == IntPtr.Zero ||
                currentThroughputGraph == IntPtr.Zero)
                throw new InvalidOperationException(
                    "Could not capture the prewarmed dropout comparison graphs.");

            StableTimer.PairResult paired = StableTimer.MeasureCalibratedHostPair(
                () => backend.EnqueueCapturedGraph(directThroughputGraph),
                backend.Synchronize,
                () => establishedBackend.EnqueueCapturedGraph(currentThroughputGraph),
                establishedBackend.Synchronize,
                operationsPerLaunchA: ThroughputOperationsPerGraph,
                operationsPerLaunchB: ThroughputOperationsPerGraph);
            void GraphLaunch() => backend.EnqueueCapturedGraph(graph);
            Cell graphCell = Measure(
                backend, run, shape, "Direct PTX CUDA graph", GraphLaunch,
                directError, audit, temporaryDeviceBytes: 0);
            yield return graphCell with
            {
                PairedEvidence = new OracleEvidence(paired, directError, currentError)
            };
            long directDispatchBefore = backend.DirectPtxRngDropoutDispatchCount;
            Cell direct = Measure(backend, run, shape, "Direct PTX fused", DirectLaunch,
                directError, audit, temporaryDeviceBytes: 0);
            const long expectedMeasuredDispatches =
                Warmups + Samples * DeviceLaunches + Warmups + Samples + 8L + Samples;
            long actualMeasuredDispatches =
                backend.DirectPtxRngDropoutDispatchCount - directDispatchBefore;
            if (actualMeasuredDispatches != expectedMeasuredDispatches ||
                backend.DirectPtxLastError is not null)
                throw new InvalidOperationException(
                    $"Direct PTX routing changed during measurement: expected " +
                    $"{expectedMeasuredDispatches}, observed {actualMeasuredDispatches}; " +
                    $"error={backend.DirectPtxLastError ?? "none"}.");
            yield return direct;
            yield return Measure(establishedBackend, run, shape, "AiDotNet current NVRTC x2", CurrentLaunch,
                currentError, null, temporaryDeviceBytes: 0);
        }
        finally
        {
            backend.DestroyCapturedGraph(graph);
            backend.DestroyCapturedGraph(directThroughputGraph);
            establishedBackend.DestroyCapturedGraph(currentThroughputGraph);
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

    private static Cell Measure(
        CudaBackend backend,
        int run,
        Shape shape,
        string method,
        Action action,
        double maximumError,
        DirectPtxKernelAudit? audit,
        long temporaryDeviceBytes)
    {
        Distribution device = MeasureDevice(backend, action);
        Distribution endToEnd = MeasureEndToEnd(backend, action);
        long managedBytes = MeasureAllocation(backend, action);
        // Required mask traffic is part of the public operation: one input
        // read plus output and saved-mask writes.
        double gbps = 3.0 * shape.Elements * sizeof(float) /
            (device.Median * 1e-6) / 1e9;
        return new Cell(
            run, shape, method, device, endToEnd, gbps,
            managedBytes, temporaryDeviceBytes, maximumError, audit, null);
    }

    private static Distribution MeasureDevice(CudaBackend backend, Action action)
    {
        for (int i = 0; i < Warmups; i++) action();
        backend.Synchronize();
        var values = new double[Samples];
        using IGpuEvent start = backend.CreateEvent(enableTiming: true);
        using IGpuEvent end = backend.CreateEvent(enableTiming: true);
        for (int sample = 0; sample < Samples; sample++)
        {
            backend.RecordEvent(start, backend.DefaultStream);
            for (int launch = 0; launch < DeviceLaunches; launch++) action();
            backend.RecordEvent(end, backend.DefaultStream);
            end.Synchronize();
            values[sample] = backend.GetEventElapsedTime(start, end) * 1_000.0 / DeviceLaunches;
        }
        return Summarize(values);
    }

    private static Distribution MeasureEndToEnd(CudaBackend backend, Action action)
    {
        for (int i = 0; i < Warmups; i++) action();
        backend.Synchronize();
        var values = new double[Samples];
        double tickUs = 1_000_000.0 / Stopwatch.Frequency;
        for (int sample = 0; sample < Samples; sample++)
        {
            long start = Stopwatch.GetTimestamp();
            action();
            backend.Synchronize();
            values[sample] = (Stopwatch.GetTimestamp() - start) * tickUs;
        }
        return Summarize(values);
    }

    private static long MeasureAllocation(CudaBackend backend, Action action)
    {
        for (int i = 0; i < 8; i++) action();
        backend.Synchronize();
#if NET6_0_OR_GREATER
        long before = GC.GetAllocatedBytesForCurrentThread();
        for (int i = 0; i < Samples; i++) action();
        long bytes = (GC.GetAllocatedBytesForCurrentThread() - before) / Samples;
#else
        long bytes = -1;
        for (int i = 0; i < Samples; i++) action();
#endif
        backend.Synchronize();
        return bytes;
    }

    private static double DirectOracleError(
        CudaBackend backend,
        float[] input,
        IGpuBuffer output,
        IGpuBuffer mask,
        int elements)
    {
        float[] actualOutput = backend.DownloadBuffer(output);
        float[] actualMask = backend.DownloadBuffer(mask);
        float keep = 1.0f - DropoutRate;
        uint threshold = (uint)Math.Floor((double)keep * 4_294_967_296.0);
        float inverse = 1.0f / keep;
        double error = 0;
        for (ulong counter = 0; counter < (ulong)elements / 4; counter++)
        {
            var words = PtxFusedPhiloxDropoutF32Kernel.GenerateUInt4(Seed, 0, counter);
            uint[] word = [words.X0, words.X1, words.X2, words.X3];
            for (int lane = 0; lane < 4; lane++)
            {
                int index = checked((int)(counter * 4 + (ulong)lane));
                float expectedMask = word[lane] < threshold ? inverse : 0.0f;
                error = Math.Max(error, Math.Abs(actualMask[index] - expectedMask));
                error = Math.Max(error, Math.Abs(actualOutput[index] - input[index] * expectedMask));
            }
        }
        return error;
    }

    private static double MaskApplicationError(
        CudaBackend backend,
        float[] input,
        IGpuBuffer output,
        IGpuBuffer mask)
    {
        float[] actualOutput = backend.DownloadBuffer(output);
        float[] actualMask = backend.DownloadBuffer(mask);
        double error = 0;
        for (int i = 0; i < input.Length; i++)
            error = Math.Max(error, Math.Abs(actualOutput[i] - input[i] * actualMask[i]));
        return error;
    }

    private static IReadOnlyList<PythonCell> RunPython(int runs)
    {
        string script = Path.Combine(
            AppContext.BaseDirectory, "BaselineRunners", "py",
            "run_direct_ptx_rng_dropout_competitors.py");
        if (!File.Exists(script))
            script = Path.Combine(AppContext.BaseDirectory, "run_direct_ptx_rng_dropout_competitors.py");
        if (!File.Exists(script)) throw new FileNotFoundException("PyTorch dropout peer is missing.", script);
        var start = new ProcessStartInfo
        {
            FileName = Environment.GetEnvironmentVariable("PYTHON") ?? "python",
            UseShellExecute = false,
            RedirectStandardOutput = true,
            RedirectStandardError = true
        };
        start.ArgumentList.Add(script);
        start.ArgumentList.Add("--runs");
        start.ArgumentList.Add(runs.ToString(CultureInfo.InvariantCulture));
        using Process process = Process.Start(start) ??
            throw new InvalidOperationException("Could not start the PyTorch CUDA peer.");
        var cells = new List<PythonCell>();
        while (process.StandardOutput.ReadLine() is { } line)
        {
            using JsonDocument json = JsonDocument.Parse(line);
            JsonElement root = json.RootElement;
            cells.Add(new PythonCell(
                ReadString(root, "status"), ReadInt(root, "run"),
                ReadString(root, "shape"), ReadString(root, "method"),
                ReadDouble(root, "device_mean_us"), ReadDouble(root, "device_median_us"),
                ReadDouble(root, "device_p95_us"), ReadDouble(root, "device_p99_us"),
                ReadDouble(root, "e2e_mean_us"), ReadDouble(root, "e2e_median_us"),
                ReadDouble(root, "e2e_p95_us"), ReadDouble(root, "e2e_p99_us"),
                ReadDouble(root, "gb_per_second"), ReadLong(root, "peak_device_bytes"),
                ReadDouble(root, "max_error")));
        }
        string error = process.StandardError.ReadToEnd();
        process.WaitForExit();
        if (process.ExitCode != 0)
            throw new InvalidOperationException($"PyTorch peer failed ({process.ExitCode}): {error}");
        return cells;
    }

    private static void Print(IReadOnlyList<Cell> cells, IReadOnlyList<PythonCell> python)
    {
        Console.WriteLine(
            "run | shape | method | device mean/median/p95/p99 us | e2e mean/median/p95/p99 us | GB/s | managed B/call | temp device B | max error | regs/shared/local/blocks-SM");
        foreach (Cell cell in cells)
        {
            string resources = cell.Audit is null ? "pending" :
                $"{cell.Audit.Function.RegistersPerThread}/{cell.Audit.Function.StaticSharedBytes}/" +
                $"{cell.Audit.Function.LocalBytesPerThread}/{cell.Audit.ActiveBlocksPerMultiprocessor}";
            Console.WriteLine(
                $"{cell.Run} | {cell.Shape.Name} | {cell.Method} | " +
                $"{Format(cell.Device)} | {Format(cell.EndToEnd)} | {cell.GbPerSecond:F2} | " +
                $"{cell.ManagedBytes} | {cell.TemporaryDeviceBytes} | {cell.MaximumError:E3} | {resources}");
        }
        foreach (PythonCell cell in python.Where(cell => cell.Status == "ok"))
            Console.WriteLine(
                $"{cell.Run} | {cell.Shape} | {cell.Method} | " +
                $"{cell.DeviceMeanUs:F3}/{cell.DeviceMedianUs:F3}/{cell.DeviceP95Us:F3}/{cell.DeviceP99Us:F3} | " +
                $"{cell.EndToEndMeanUs:F3}/{cell.EndToEndMedianUs:F3}/{cell.EndToEndP95Us:F3}/{cell.EndToEndP99Us:F3} | " +
                $"{cell.GbPerSecond:F2} | n/a | {cell.PeakDeviceBytes} | {cell.MaxError:E3} | pending");
        foreach (Cell cell in cells.Where(cell => cell.PairedEvidence is not null))
            PrintPairedEvidence(cell);
    }

    private static void PrintPairedEvidence(Cell cell)
    {
        OracleEvidence evidence = cell.PairedEvidence!.Value;
        StableTimer.PairResult timing = evidence.Timing;
        double incumbentOverDirect = 1.0 / timing.Ratio;
        const double semanticTolerance = 1.0e-4;
        bool correctnessPassed =
            double.IsFinite(evidence.DirectError) &&
            double.IsFinite(evidence.IncumbentError) &&
            evidence.DirectError <= semanticTolerance &&
            evidence.IncumbentError <= semanticTolerance;
        string measurementStatus = !correctnessPassed
            ? "correctness-failed"
            : !timing.Stable
                ? "unstable"
                : "stable";
        string verdict = !correctnessPassed || !timing.Stable
            ? "not-measurable"
            : incumbentOverDirect >= RequiredGain
                ? "win"
                : incumbentOverDirect <= 1.0 / RequiredGain
                    ? "loss"
                    : "tie";
        Console.WriteLine(
            $"oracle | {cell.Run} | {cell.Shape.Name} | " +
            $"direct={timing.A.Describe()} | incumbent={timing.B.Describe()} | " +
            $"direct/incumbent={(correctnessPassed ? timing.DescribeRatio() : "-")} | " +
            $"correctness={measurementStatus} " +
            $"(direct={evidence.DirectError:E3}, incumbent={evidence.IncumbentError:E3}, " +
            $"tolerance={semanticTolerance:E1}) | {verdict}");
        Console.WriteLine(JsonSerializer.Serialize(new
        {
            kind = "rng-dropout-kernel-throughput-oracle",
            run = cell.Run,
            shape = cell.Shape.Name,
            lane = "repeated-cuda-graph-host-paired",
            logical_operations_per_graph = ThroughputOperationsPerGraph,
            direct_median_us = timing.A.Stable ? timing.A.Microseconds : (double?)null,
            incumbent_median_us = timing.B.Stable ? timing.B.Microseconds : (double?)null,
            incumbent_over_direct = correctnessPassed && timing.Stable
                ? incumbentOverDirect
                : (double?)null,
            direct_relative_spread = FiniteOrNull(timing.A.RelativeSpread),
            incumbent_relative_spread = FiniteOrNull(timing.B.RelativeSpread),
            paired_ratio_relative_spread = FiniteOrNull(timing.RelativeSpread),
            attempts = timing.Samples,
            required_gain = RequiredGain,
            direct_semantic_error = FiniteOrNull(evidence.DirectError),
            incumbent_semantic_error = FiniteOrNull(evidence.IncumbentError),
            semantic_tolerance = semanticTolerance,
            correctness_passed = correctnessPassed,
            measurement_status = measurementStatus,
            verdict,
            diagnostic_only = true,
            promotion = false
        }));
    }

    private static double? FiniteOrNull(double value) =>
        double.IsNaN(value) || double.IsInfinity(value) ? null : value;

    private static Distribution Summarize(double[] values)
    {
        Array.Sort(values);
        return new Distribution(values.Average(), Percentile(values, 0.50),
            Percentile(values, 0.95), Percentile(values, 0.99));
    }

    private static double Percentile(double[] values, double percentile)
    {
        double position = (values.Length - 1) * percentile;
        int lower = (int)position;
        int upper = Math.Min(lower + 1, values.Length - 1);
        return values[lower] + (values[upper] - values[lower]) * (position - lower);
    }

    private static string Format(Distribution value) =>
        $"{value.Mean:F3}/{value.Median:F3}/{value.P95:F3}/{value.P99:F3}";
    private static string ReadString(JsonElement root, string name) => root.GetProperty(name).GetString() ?? string.Empty;
    private static int ReadInt(JsonElement root, string name) => root.GetProperty(name).GetInt32();
    private static long ReadLong(JsonElement root, string name) => root.GetProperty(name).GetInt64();
    private static double ReadDouble(JsonElement root, string name) => root.GetProperty(name).GetDouble();
}
