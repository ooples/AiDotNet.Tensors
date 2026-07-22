using System.Diagnostics;
using System.Text.Json;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;
using AiDotNet.Tensors.Engines.Gpu;

namespace AiDotNet.Tensors.Benchmarks;

/// <summary>Issue #846 resident NVIDIA RG-LRU championship harness.</summary>
internal static class DirectPtxRecurrentExperiment
{
    private const int Warmups = 30;
    private const int Samples = 101;
    private const int LaunchesPerDeviceSample = 10;
    private const int Batch = PtxFusedRgLruScan128x256Kernel.Batch;
    private const int Sequence = PtxFusedRgLruScan128x256Kernel.SequenceLength;
    private const int Dimension = PtxFusedRgLruScan128x256Kernel.RecurrentDimension;
    private const double EstimatedFlops = Batch * Sequence * Dimension * 12.0 + Dimension * 4.0;

    private readonly record struct Distribution(double Mean, double Median, double P95, double P99);
    private sealed record ExternalRecord(
        string Status, int Run, string Method,
        double MeanUs, double MedianUs, double P95Us, double P99Us,
        long PeakDeviceBytes, long TemporaryDeviceBytes, double MaxError);

    internal static void Run(int independentRuns = 3, bool includeExternal = true)
    {
        if (independentRuns <= 0) throw new ArgumentOutOfRangeException(nameof(independentRuns));
        GpuBenchmarkEnvironment.RequireIdleGpu("direct-ptx-rglru-start");
        bool? previous = DirectPtxFeatureGate.TestOverride;
        DirectPtxFeatureGate.TestOverride = true;
        try
        {
            Console.WriteLine(
                $"RG-LRU [B={Batch},S={Sequence},D={Dimension}], FP32 resident tensors; " +
                $"{Warmups} warmups + {Samples} samples x {LaunchesPerDeviceSample} device launches; " +
                $"{independentRuns} independent runs");
            PrintHeader();
            for (int run = 1; run <= independentRuns; run++) RunAiDotNet(run);
            if (includeExternal)
                foreach (ExternalRecord record in RunPython(independentRuns)) Print(record);
            Console.WriteLine(
                "release_gate=NOT_PROMOTED_UNTIL_3_RUN_CORRECTNESS_PERFORMANCE_AND_NSIGHT_EVIDENCE_PASS");
        }
        finally
        {
            DirectPtxFeatureGate.TestOverride = previous;
            GpuBenchmarkEnvironment.RequireNoForeignCompute("direct-ptx-rglru-end");
        }
    }

    private static void RunAiDotNet(int run)
    {
        using var backend = new CudaBackend();
        if (!backend.IsDirectPtxRgLruEnabled)
            throw new InvalidOperationException("The RG-LRU prototype requires exact SM86 admission.");
        var random = new Random(846_000 + run);
        int elements = Batch * Sequence * Dimension;
        float[] valueHost = Values(random, elements, 0.25f);
        float[] recurrenceHost = Values(random, elements, 0.25f, 0.5f);
        float[] inputGateHost = Values(random, elements, 0.25f, 0.5f);
        float[] decayHost = Values(random, Dimension, 0.5f);
        double[] oracle = Oracle(valueHost, recurrenceHost, inputGateHost, decayHost);
        using var value = backend.AllocateBuffer(valueHost);
        using var recurrence = backend.AllocateBuffer(recurrenceHost);
        using var inputGate = backend.AllocateBuffer(inputGateHost);
        using var decay = backend.AllocateBuffer(decayHost);
        using var directOutput = backend.AllocateBuffer(elements);
        using var currentOutput = backend.AllocateBuffer(elements);

        if (!backend.PrewarmDirectPtxRgLruScan())
            throw new InvalidOperationException(backend.DirectPtxLastError);
        void DirectLaunch()
        {
            if (!backend.TryDirectPtxRgLruScanForward(
                value, recurrence, inputGate, decay, directOutput, Batch, Sequence, Dimension))
                throw new InvalidOperationException(backend.DirectPtxLastError);
        }
        void CurrentLaunch() => backend.LaunchLegacyRgLruScanForward(
            value, recurrence, inputGate, decay, currentOutput, Batch, Sequence, Dimension);

        DirectLaunch();
        CurrentLaunch();
        backend.Synchronize();
        double directError = MaximumError(backend.DownloadBuffer(directOutput), oracle);
        double currentError = MaximumError(backend.DownloadBuffer(currentOutput), oracle);
        IntPtr graph = backend.CaptureGraph(DirectLaunch);
        if (graph == IntPtr.Zero)
            throw new InvalidOperationException("Could not capture prewarmed RG-LRU direct PTX.");
        try
        {
            void GraphLaunch() => backend.EnqueueCapturedGraph(graph);
            if (!backend.TryGetDirectPtxRgLruAudit(out DirectPtxKernelAudit audit))
                throw new InvalidOperationException("No RG-LRU audit exists after prewarm.");
            Console.WriteLine("rglru_environment_json=" + JsonSerializer.Serialize(new
            {
                run,
                gpu = backend.DeviceName,
                device_fingerprint = audit.DeviceFingerprint,
                dotnet = Environment.Version.ToString(),
                os = System.Runtime.InteropServices.RuntimeInformation.OSDescription,
                architecture = System.Runtime.InteropServices.RuntimeInformation.ProcessArchitecture.ToString(),
                warmups = Warmups,
                samples = Samples,
                launches_per_device_sample = LaunchesPerDeviceSample
            }));
            Console.WriteLine("rglru_audit_json=" + audit.ToJson());
            Print(run, "Direct PTX CUDA graph", MeasureDevice(backend, GraphLaunch),
                MeasureAllocation(backend, GraphLaunch), 0, directError, audit);
            Print(run, "Direct PTX fused", MeasureDevice(backend, DirectLaunch),
                MeasureAllocation(backend, DirectLaunch), 0, directError, audit);
            Print(run, "AiDotNet current NVRTC", MeasureDevice(backend, CurrentLaunch),
                MeasureAllocation(backend, CurrentLaunch), 0, currentError, null);
        }
        finally
        {
            backend.DestroyCapturedGraph(graph);
        }
    }

    private static Distribution MeasureDevice(CudaBackend backend, Action action)
    {
        for (int index = 0; index < Warmups; index++) action();
        backend.Synchronize();
        var timings = new double[Samples];
        using IGpuEvent start = backend.CreateEvent(enableTiming: true);
        using IGpuEvent stop = backend.CreateEvent(enableTiming: true);
        for (int sample = 0; sample < Samples; sample++)
        {
            backend.RecordEvent(start, backend.DefaultStream);
            for (int launch = 0; launch < LaunchesPerDeviceSample; launch++) action();
            backend.RecordEvent(stop, backend.DefaultStream);
            stop.Synchronize();
            timings[sample] = backend.GetEventElapsedTime(start, stop) * 1_000.0 /
                LaunchesPerDeviceSample;
        }
        return Summarize(timings);
    }

    private static long MeasureAllocation(CudaBackend backend, Action action)
    {
        for (int index = 0; index < 8; index++) action();
        backend.Synchronize();
        long before = GC.GetAllocatedBytesForCurrentThread();
        for (int index = 0; index < Samples; index++) action();
        long bytes = (GC.GetAllocatedBytesForCurrentThread() - before) / Samples;
        backend.Synchronize();
        return bytes;
    }

    private static IReadOnlyList<ExternalRecord> RunPython(int runs)
    {
        string script = Path.Combine(AppContext.BaseDirectory, "BaselineRunners", "py",
            "run_direct_ptx_rglru_competitors.py");
        if (!File.Exists(script))
            throw new FileNotFoundException("The issue #846 PyTorch CUDA harness was not copied.", script);
        var start = new ProcessStartInfo
        {
            FileName = Environment.GetEnvironmentVariable("PYTHON") ?? "python",
            UseShellExecute = false,
            RedirectStandardOutput = true,
            RedirectStandardError = true
        };
        start.ArgumentList.Add(script);
        start.ArgumentList.Add("--runs");
        start.ArgumentList.Add(runs.ToString(System.Globalization.CultureInfo.InvariantCulture));
        start.ArgumentList.Add("--json-lines");
        using Process process = Process.Start(start) ??
            throw new InvalidOperationException("Could not start the PyTorch RG-LRU baseline.");
        var records = new List<ExternalRecord>();
        while (process.StandardOutput.ReadLine() is { } line)
        {
            using JsonDocument document = JsonDocument.Parse(line);
            JsonElement root = document.RootElement;
            records.Add(new ExternalRecord(
                root.GetProperty("status").GetString() ?? "",
                root.GetProperty("run").GetInt32(),
                root.GetProperty("method").GetString() ?? "",
                root.GetProperty("mean_us").GetDouble(),
                root.GetProperty("median_us").GetDouble(),
                root.GetProperty("p95_us").GetDouble(),
                root.GetProperty("p99_us").GetDouble(),
                root.GetProperty("peak_device_bytes").GetInt64(),
                root.GetProperty("temporary_device_bytes").GetInt64(),
                root.GetProperty("max_error").GetDouble()));
        }
        string error = process.StandardError.ReadToEnd();
        process.WaitForExit();
        if (process.ExitCode != 0)
            throw new InvalidOperationException(
                $"PyTorch CUDA RG-LRU baseline failed with exit {process.ExitCode}: {error}");
        return records;
    }

    private static void PrintHeader()
    {
        Console.WriteLine(
            $"{"Run",3} {"Operation",-17} {"Method",-25} {"mean us",9} {"median us",10} " +
            $"{"p95 us",9} {"p99 us",9} {"Gupdates/s",11} {"GFLOPS est",11} " +
            $"{"managed B",10} {"temp/peak B",11} {"max error",10} {"regs",5} {"shared",7} {"local",5} {"blocks/SM",9}");
        Console.WriteLine(new string('-', 174));
    }

    private static void Print(
        int run, string method, Distribution timing, long managedBytes,
        long temporaryBytes, double error, DirectPtxKernelAudit? audit)
    {
        double seconds = timing.Median * 1e-6;
        Console.WriteLine(
            $"{run,3} {"RG-LRU S128 D256",-17} {method,-25} " +
            $"{timing.Mean,9:F2} {timing.Median,10:F2} {timing.P95,9:F2} {timing.P99,9:F2} " +
            $"{Batch * Sequence * Dimension / seconds / 1e9,11:F3} " +
            $"{EstimatedFlops / seconds / 1e9,11:F3} {managedBytes,10} {temporaryBytes,11} " +
            $"{error,10:G4} {Value(audit?.Function.RegistersPerThread),5} " +
            $"{Value(audit?.Function.StaticSharedBytes),7} {Value(audit?.Function.LocalBytesPerThread),5} " +
            $"{Value(audit?.ActiveBlocksPerMultiprocessor),9}");
    }

    private static void Print(ExternalRecord record)
    {
        if (!string.Equals(record.Status, "ok", StringComparison.Ordinal)) return;
        Print(record.Run, record.Method,
            new Distribution(record.MeanUs, record.MedianUs, record.P95Us, record.P99Us),
            -1, Math.Max(record.PeakDeviceBytes, record.TemporaryDeviceBytes),
            record.MaxError, null);
    }

    private static string Value(int? value) => value?.ToString() ?? "-";

    private static Distribution Summarize(double[] values)
    {
        Array.Sort(values);
        return new Distribution(values.Average(), Percentile(values, 0.50),
            Percentile(values, 0.95), Percentile(values, 0.99));
    }

    private static double Percentile(double[] sorted, double q)
    {
        double position = (sorted.Length - 1) * q;
        int lower = (int)position;
        int upper = Math.Min(lower + 1, sorted.Length - 1);
        return sorted[lower] + (sorted[upper] - sorted[lower]) * (position - lower);
    }

    private static float[] Values(Random random, int count, float scale, float bias = 0f)
    {
        var result = new float[count];
        for (int index = 0; index < result.Length; index++)
            result[index] = bias + ((float)random.NextDouble() - 0.5f) * scale;
        return result;
    }

    private static double[] Oracle(float[] value, float[] recurrence, float[] inputGate, float[] decay)
    {
        var output = new double[value.Length];
        for (int channel = 0; channel < Dimension; channel++)
        {
            double state = 0;
            double channelDecay = 1.0 / (1.0 + Math.Exp(decay[channel]));
            for (int timestep = 0; timestep < Sequence; timestep++)
            {
                int offset = timestep * Dimension + channel;
                double a = recurrence[offset] * channelDecay;
                double scale = Math.Sqrt(Math.Max(0, 1 - a * a));
                state = a * state + scale * inputGate[offset] * value[offset];
                output[offset] = state;
            }
        }
        return output;
    }

    private static double MaximumError(float[] actual, double[] expected)
    {
        double maximum = 0;
        for (int index = 0; index < actual.Length; index++)
            maximum = Math.Max(maximum, Math.Abs(actual[index] - expected[index]));
        return maximum;
    }
}
