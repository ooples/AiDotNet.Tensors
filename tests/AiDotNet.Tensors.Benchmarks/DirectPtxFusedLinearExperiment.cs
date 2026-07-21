using System.Diagnostics;
using System.Text.Json;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;
using AiDotNet.Tensors.Engines.Gpu;

namespace AiDotNet.Tensors.Benchmarks;

/// <summary>Issue #836 M=1 FP32 fused linear + bias + tanh-GELU bakeoff.</summary>
internal static class DirectPtxFusedLinearExperiment
{
    private const int Warmups = 30;
    private const int Samples = 101;
    private const int DeviceLaunches = 10;

    private readonly record struct Shape(string Name, int K, int N);
    private readonly record struct Distribution(double Mean, double Median, double P95, double P99);
    private sealed record PythonRecord(
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
        long PeakDeviceBytes,
        double MaxError);

    private static readonly Shape[] Shapes =
    [
        new("decode-256x256", 256, 256),
        new("decode-up-512x2048", 512, 2048),
        new("decode-up-1024x4096", 1024, 4096)
    ];

    internal static void Run(int independentRuns = 1)
    {
        if (independentRuns <= 0) throw new ArgumentOutOfRangeException(nameof(independentRuns));
        bool? previous = DirectPtxFeatureGate.TestOverride;
        bool previousExperiment = DirectPtxFeatureGate.FusedLinearExperimentOverride;
        DirectPtxFeatureGate.FusedLinearExperimentOverride = true;
        try
        {
            Console.WriteLine(
                $"Direct-PTX fused linear+bias+tanh-GELU: {independentRuns} run(s), " +
                $"{Warmups} warmups + {Samples} samples/cell");
            Console.WriteLine(
                "All operands/output are resident and preallocated; device samples average " +
                $"{DeviceLaunches} launches.");
            PrintHeader();
            for (int run = 1; run <= independentRuns; run++)
            {
                using var backend = new CudaBackend();
                using var cublasLt = new CuBlasLtMatmul();
                if (run == 1) Console.WriteLine($"GPU: {backend.DeviceName}");
                foreach (Shape shape in Shapes) RunCell(backend, cublasLt, run, shape);
            }
            foreach (PythonRecord record in RunPython(independentRuns))
                if (record.Status == "ok") PrintPython(record);
        }
        finally
        {
            DirectPtxFeatureGate.TestOverride = previous;
            DirectPtxFeatureGate.FusedLinearExperimentOverride = previousExperiment;
        }
    }

    private static void RunCell(CudaBackend backend, CuBlasLtMatmul cublasLt, int run, Shape shape)
    {
        var random = new Random(20261400 + run * 10_000 + shape.K + shape.N);
        float[] inputHost = Values(random, shape.K, 0.125f);
        float[] inputMajorWeightHost = Values(random, shape.K * shape.N, 0.0625f);
        float[] outputMajorWeightHost = TransposeWeights(
            inputMajorWeightHost, shape.K, shape.N);
        float[] biasHost = Values(random, shape.N, 0.0625f);
        float[] expected = Oracle(inputHost, inputMajorWeightHost, biasHost, shape.K, shape.N);
        using var input = backend.AllocateBuffer(inputHost);
        using var inputMajorWeights = backend.AllocateBuffer(inputMajorWeightHost);
        using var outputMajorWeights = backend.AllocateBuffer(outputMajorWeightHost);
        using var bias = backend.AllocateBuffer(biasHost);
        using var directOutput = backend.AllocateBuffer(shape.N);
        using var currentOutput = backend.AllocateBuffer(shape.N);
        using var cublasLtOutput = backend.AllocateBuffer(shape.N);

        void CurrentLaunch() => backend.FusedLinearGELU(
            input, inputMajorWeights, bias, currentOutput, 1, shape.K, shape.N);
        void DirectLaunch()
        {
            if (!backend.TryDirectPtxFusedLinearGeluM1(
                input, outputMajorWeights, bias, directOutput, shape.K, shape.N))
                throw new InvalidOperationException(backend.DirectPtxLastError);
        }
        void CuBlasLtLaunch() => cublasLt.MatmulFused(
            aDev: inputMajorWeights.Handle, m: shape.N, k: shape.K, transA: false,
            bDev: input.Handle, n: 1, transB: false,
            cDev: IntPtr.Zero, dDev: cublasLtOutput.Handle,
            biasDev: bias.Handle, epilogue: CublasLtEpilogue.GELUBias,
            stream: backend.DefaultStream.Handle,
            dtype: CublasDataType.Float32,
            computeType: CublasComputeType.Float32);

        DirectPtxFeatureGate.TestOverride = false;
        CurrentLaunch();
        backend.Synchronize();
        float currentError = MaximumError(backend.DownloadBuffer(currentOutput), expected);
        Distribution currentDevice = MeasureDevice(backend, CurrentLaunch);
        Distribution currentEndToEnd = MeasureEndToEnd(backend, CurrentLaunch);
        long currentAllocation = MeasureAllocation(backend, CurrentLaunch);
        Distribution currentGraphDevice;
        Distribution currentGraphEndToEnd;
        long currentGraphAllocation;
        IntPtr currentGraph = backend.CaptureGraph(CurrentLaunch);
        try
        {
            void CurrentGraphLaunch() => backend.LaunchCapturedGraph(currentGraph);
            currentGraphDevice = MeasureDevice(backend, CurrentGraphLaunch);
            currentGraphEndToEnd = MeasureEndToEnd(backend, CurrentGraphLaunch);
            currentGraphAllocation = MeasureAllocation(backend, CurrentGraphLaunch);
        }
        finally { backend.DestroyCapturedGraph(currentGraph); }

        CuBlasLtLaunch();
        backend.Synchronize();
        float cublasLtError = MaximumError(backend.DownloadBuffer(cublasLtOutput), expected);
        Distribution cublasLtDevice = MeasureDevice(backend, CuBlasLtLaunch);
        Distribution cublasLtEndToEnd = MeasureEndToEnd(backend, CuBlasLtLaunch);
        long cublasLtAllocation = MeasureAllocation(backend, CuBlasLtLaunch);
        Distribution cublasLtGraphDevice;
        Distribution cublasLtGraphEndToEnd;
        long cublasLtGraphAllocation;
        IntPtr cublasLtGraph = backend.CaptureGraph(CuBlasLtLaunch);
        try
        {
            void CuBlasLtGraphLaunch() => backend.LaunchCapturedGraph(cublasLtGraph);
            cublasLtGraphDevice = MeasureDevice(backend, CuBlasLtGraphLaunch);
            cublasLtGraphEndToEnd = MeasureEndToEnd(backend, CuBlasLtGraphLaunch);
            cublasLtGraphAllocation = MeasureAllocation(backend, CuBlasLtGraphLaunch);
        }
        finally { backend.DestroyCapturedGraph(cublasLtGraph); }

        DirectPtxFeatureGate.TestOverride = true;
        if (!backend.PrewarmDirectPtxFusedLinearGeluM1(shape.K, shape.N))
            throw new InvalidOperationException(backend.DirectPtxLastError);
        DirectLaunch();
        backend.Synchronize();
        float directError = MaximumError(backend.DownloadBuffer(directOutput), expected);
        Distribution directDevice = MeasureDevice(backend, DirectLaunch);
        Distribution directEndToEnd = MeasureEndToEnd(backend, DirectLaunch);
        long directAllocation = MeasureAllocation(backend, DirectLaunch);
        Distribution directGraphDevice;
        Distribution directGraphEndToEnd;
        long directGraphAllocation;
        IntPtr directGraph = backend.CaptureGraph(DirectLaunch);
        try
        {
            void DirectGraphLaunch() => backend.LaunchCapturedGraph(directGraph);
            directGraphDevice = MeasureDevice(backend, DirectGraphLaunch);
            directGraphEndToEnd = MeasureEndToEnd(backend, DirectGraphLaunch);
            directGraphAllocation = MeasureAllocation(backend, DirectGraphLaunch);
        }
        finally { backend.DestroyCapturedGraph(directGraph); }
        if (!backend.TryGetDirectPtxFusedLinearAudit(
            shape.K, shape.N, out DirectPtxKernelAudit audit))
            throw new InvalidOperationException("No audit for the measured fused-linear module.");

        Print(run, shape, "Direct PTX", directDevice, directEndToEnd,
            directAllocation, 0, directError,
            audit.Function.RegistersPerThread, audit.Function.StaticSharedBytes,
            audit.Function.LocalBytesPerThread, audit.ActiveBlocksPerMultiprocessor);
        Print(run, shape, "Direct PTX graph", directGraphDevice, directGraphEndToEnd,
            directGraphAllocation, 0, directError,
            audit.Function.RegistersPerThread, audit.Function.StaticSharedBytes,
            audit.Function.LocalBytesPerThread, audit.ActiveBlocksPerMultiprocessor);
        Print(run, shape, "AiDotNet NVRTC", currentDevice, currentEndToEnd,
            currentAllocation, 0, currentError, -1, -1, -1, -1);
        Print(run, shape, "AiDotNet graph", currentGraphDevice, currentGraphEndToEnd,
            currentGraphAllocation, 0, currentError, -1, -1, -1, -1);
        Print(run, shape, "NVIDIA cuBLASLt", cublasLtDevice, cublasLtEndToEnd,
            cublasLtAllocation, 0, cublasLtError, -1, -1, -1, -1);
        Print(run, shape, "cuBLASLt graph", cublasLtGraphDevice, cublasLtGraphEndToEnd,
            cublasLtGraphAllocation, 0, cublasLtError, -1, -1, -1, -1);
    }

    private static Distribution MeasureDevice(CudaBackend backend, Action action)
    {
        for (int i = 0; i < Warmups; i++) action();
        backend.Synchronize();
        var values = new double[Samples];
        using IGpuEvent start = backend.CreateEvent(enableTiming: true);
        using IGpuEvent end = backend.CreateEvent(enableTiming: true);
        for (int sample = 0; sample < values.Length; sample++)
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
        double tickToMicroseconds = 1_000_000.0 / Stopwatch.Frequency;
        for (int sample = 0; sample < values.Length; sample++)
        {
            long start = Stopwatch.GetTimestamp();
            action();
            backend.Synchronize();
            values[sample] = (Stopwatch.GetTimestamp() - start) * tickToMicroseconds;
        }
        return Summarize(values);
    }

    private static long MeasureAllocation(CudaBackend backend, Action action)
    {
        for (int i = 0; i < 8; i++) action();
        backend.Synchronize();
        long before = GC.GetAllocatedBytesForCurrentThread();
        for (int i = 0; i < Samples; i++) action();
        long bytesPerCall = (GC.GetAllocatedBytesForCurrentThread() - before) / Samples;
        backend.Synchronize();
        return bytesPerCall;
    }

    private static Distribution Summarize(double[] values)
    {
        Array.Sort(values);
        return new Distribution(values.Average(), Percentile(values, 0.50),
            Percentile(values, 0.95), Percentile(values, 0.99));
    }

    private static IReadOnlyList<PythonRecord> RunPython(int runs)
    {
        string script = Path.Combine(
            AppContext.BaseDirectory, "BaselineRunners", "py",
            "run_direct_ptx_fused_linear_competitors.py");
        if (!File.Exists(script))
            script = Path.Combine(AppContext.BaseDirectory,
                "run_direct_ptx_fused_linear_competitors.py");
        if (!File.Exists(script))
            throw new FileNotFoundException("The issue #836 PyTorch harness was not copied to output.", script);
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
            throw new InvalidOperationException("Could not start the PyTorch CUDA baseline.");
        var records = new List<PythonRecord>();
        while (process.StandardOutput.ReadLine() is { } line)
        {
            using JsonDocument document = JsonDocument.Parse(line);
            JsonElement root = document.RootElement;
            records.Add(new PythonRecord(
                root.GetProperty("status").GetString()!,
                root.GetProperty("run").GetInt32(),
                root.GetProperty("shape").GetString()!,
                root.GetProperty("method").GetString()!,
                root.GetProperty("device_mean_us").GetDouble(),
                root.GetProperty("device_median_us").GetDouble(),
                root.GetProperty("device_p95_us").GetDouble(),
                root.GetProperty("device_p99_us").GetDouble(),
                root.GetProperty("e2e_mean_us").GetDouble(),
                root.GetProperty("e2e_median_us").GetDouble(),
                root.GetProperty("e2e_p95_us").GetDouble(),
                root.GetProperty("e2e_p99_us").GetDouble(),
                root.GetProperty("peak_device_bytes").GetInt64(),
                root.GetProperty("max_error").GetDouble()));
        }
        string stderr = process.StandardError.ReadToEnd();
        process.WaitForExit();
        if (process.ExitCode != 0)
            throw new InvalidOperationException(
                $"PyTorch CUDA baseline exited {process.ExitCode}: {stderr}");
        return records;
    }

    private static void PrintPython(PythonRecord record)
    {
        Shape shape = Shapes.Single(candidate => candidate.Name == record.Shape);
        Print(record.Run, shape, record.Method,
            new Distribution(record.DeviceMeanUs, record.DeviceMedianUs,
                record.DeviceP95Us, record.DeviceP99Us),
            new Distribution(record.EndToEndMeanUs, record.EndToEndMedianUs,
                record.EndToEndP95Us, record.EndToEndP99Us),
            allocation: -1, temporaryBytes: record.PeakDeviceBytes,
            error: (float)record.MaxError,
            registers: -1, shared: -1, local: -1, blocks: -1);
    }

    private static double Percentile(double[] sorted, double percentile)
    {
        double position = (sorted.Length - 1) * percentile;
        int lower = (int)position;
        int upper = Math.Min(lower + 1, sorted.Length - 1);
        return sorted[lower] + (sorted[upper] - sorted[lower]) * (position - lower);
    }

    private static void PrintHeader()
    {
        Console.WriteLine(
            "run shape                 method            dev mean/med/p95/p99 us       " +
            "e2e mean/med/p95/p99 us       TFLOPS  GFLOPS  allocB tmpB err      R/S/L/B");
        Console.WriteLine(new string('-', 169));
    }

    private static void Print(
        int run, Shape shape, string method,
        Distribution device, Distribution endToEnd,
        long allocation, long temporaryBytes, float error,
        int registers, int shared, int local, int blocks)
    {
        double flops = 2.0 * shape.K * shape.N;
        double tflops = flops / (device.Median * 1e-6) / 1e12;
        double gflops = tflops * 1_000.0;
        Console.WriteLine(
            $"{run,3} {shape.Name,-21} {method,-17} " +
            $"{device.Mean,6:F2}/{device.Median,6:F2}/{device.P95,6:F2}/{device.P99,6:F2}   " +
            $"{endToEnd.Mean,6:F2}/{endToEnd.Median,6:F2}/{endToEnd.P95,6:F2}/{endToEnd.P99,6:F2}   " +
            $"{tflops,6:F3} {gflops,7:F1} {allocation,7} {temporaryBytes,4} {error,8:E1} " +
            $"{registers,2}/{shared,1}/{local,1}/{blocks,2}");
    }

    private static float[] Values(Random random, int count, float scale) =>
        Enumerable.Range(0, count)
            .Select(_ => (random.NextSingle() * 2f - 1f) * scale).ToArray();

    private static float[] Oracle(
        float[] input, float[] weights, float[] bias, int inputFeatures, int outputFeatures)
    {
        var output = new float[outputFeatures];
        for (int column = 0; column < outputFeatures; column++)
        {
            float value = bias[column];
            for (int inner = 0; inner < inputFeatures; inner++)
                value += input[inner] * weights[inner * outputFeatures + column];
            output[column] = 0.5f * value *
                (1f + MathF.Tanh(0.7978845608f *
                    (value + 0.044715f * value * value * value)));
        }
        return output;
    }

    private static float[] TransposeWeights(float[] inputMajor, int inputFeatures, int outputFeatures)
    {
        var outputMajor = new float[inputMajor.Length];
        for (int inner = 0; inner < inputFeatures; inner++)
        for (int output = 0; output < outputFeatures; output++)
            outputMajor[output * inputFeatures + inner] =
                inputMajor[inner * outputFeatures + output];
        return outputMajor;
    }

    private static float MaximumError(float[] actual, float[] expected)
    {
        if (actual.Length != expected.Length) return float.PositiveInfinity;
        float maximum = 0;
        for (int i = 0; i < actual.Length; i++)
            maximum = MathF.Max(maximum, MathF.Abs(actual[i] - expected[i]));
        return maximum;
    }
}
