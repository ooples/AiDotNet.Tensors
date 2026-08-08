using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Text.Json;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;
using AiDotNet.Tensors.Engines.Gpu;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Tensors.Benchmarks;

/// <summary>Issue #837 symmetric W8A8 decode projection championship.</summary>
internal static class DirectPtxW8A8LinearExperiment
{
    private const int Warmups = 30;
    private const int Samples = 101;
    private const int DeviceLaunches = 50;
    private const int K = 1024;
    private const int N = 4096;
    private static IntPtr _nvrtcBuiltins;

    private readonly record struct Distribution(double Mean, double Median, double P95, double P99);

    internal static void Run(int independentRuns = 3)
    {
        if (independentRuns <= 0) throw new ArgumentOutOfRangeException(nameof(independentRuns));
        bool? previous = DirectPtxFeatureGate.TestOverride;
        bool previousExperiment = DirectPtxFeatureGate.QuantizedLinearExperimentOverride;
        DirectPtxFeatureGate.QuantizedLinearExperimentOverride = true;
        try
        {
            LoadBundledNvrtcBuiltins();
            Console.WriteLine(
                $"Direct-PTX symmetric W8A8 fused linear+bias+tanh-GELU M=1 K={K} N={N}: " +
                $"{independentRuns} run(s), {Warmups} warmups + {Samples} samples/cell");
            Console.WriteLine(
                $"Resident exact ABI; device samples average {DeviceLaunches} launches. " +
                "GOPS/TOPS count 2*K*N integer MAC operations; they are not FP FLOPS.");
            PrintHeader();
            for (int run = 1; run <= independentRuns; run++) RunCell(run);
            PrintPyTorchEligibility();
        }
        finally
        {
            DirectPtxFeatureGate.TestOverride = previous;
            DirectPtxFeatureGate.QuantizedLinearExperimentOverride = previousExperiment;
        }
    }

    private static void LoadBundledNvrtcBuiltins()
    {
        if (!OperatingSystem.IsWindows() || _nvrtcBuiltins != IntPtr.Zero) return;
        string nativeDirectory = Path.Combine(
            AppContext.BaseDirectory, "runtimes", "win-x64", "native");
        if (!Directory.Exists(nativeDirectory)) return;
        string? builtins = Directory.EnumerateFiles(
            nativeDirectory, "nvrtc-builtins64_*.dll").OrderBy(path => path).LastOrDefault();
        if (builtins is not null) _nvrtcBuiltins = NativeLibrary.Load(builtins);
    }

    private static void RunCell(int run)
    {
        using var backend = new CudaBackend();
        using var cublasLt = new CuBlasLtMatmul();
        if (run == 1) Console.WriteLine($"GPU: {backend.DeviceName}");
        var random = RandomHelper.CreateSeededRandom(20261700 + run);
        sbyte[] inputHost = QuantizedValues(random, K);
        sbyte[] weightsHost = QuantizedValues(random, K * N);
        sbyte[] inputMajorWeightsHost = TransposeOutputMajor(weightsHost, K, N);
        const float activationScaleHost = 0.01f;
        float[] weightScalesHost = Enumerable.Range(0, N)
            .Select(_ => 0.004f + random.NextSingle() * 0.001f).ToArray();
        float[] biasHost = Enumerable.Range(0, N)
            .Select(_ => (random.NextSingle() * 2f - 1f) * 0.02f).ToArray();
        float[] expected = Oracle(
            inputHost, weightsHost, activationScaleHost,
            weightScalesHost, biasHost);

        using var input = backend.AllocateByteBuffer(K);
        using var weights = backend.AllocateByteBuffer(K * N);
        using var inputMajorWeights = backend.AllocateByteBuffer(K * N);
        using var activationScale = backend.AllocateBuffer([activationScaleHost]);
        using var weightScales = backend.AllocateBuffer(weightScalesHost);
        using var bias = backend.AllocateBuffer(biasHost);
        using var directOutput = backend.AllocateBuffer(N);
        using var baselineOutput = backend.AllocateBuffer(N);
        using var cublasLtAccumulator = backend.AllocateByteBuffer(N * sizeof(int));
        using var cublasLtOutput = backend.AllocateBuffer(N);
        backend.UploadByteBuffer(input, ToBytes(inputHost));
        backend.UploadByteBuffer(weights, ToBytes(weightsHost));
        backend.UploadByteBuffer(inputMajorWeights, ToBytes(inputMajorWeightsHost));
        backend.Synchronize();

        void BaselineLaunch() => backend.FusedLinearGELUW8A8TransposedM1(
            input, weights, activationScale, weightScales, bias, baselineOutput, K, N);
        void DirectLaunch()
        {
            if (!backend.TryDirectPtxFusedLinearGeluW8A8M1(
                input, weights, activationScale, weightScales, bias, directOutput, K, N))
                throw new InvalidOperationException(backend.DirectPtxLastError);
        }
        void CuBlasLtLaunch()
        {
            cublasLt.MatmulInt8ToInt32(
                inputMajorWeights.Handle, N, K, false,
                input.Handle, 1, false,
                cublasLtAccumulator.Handle, backend.DefaultStream.Handle);
            backend.W8A8DequantBiasGelu(
                cublasLtAccumulator, activationScale, weightScales, bias,
                cublasLtOutput, N);
        }

        DirectPtxFeatureGate.TestOverride = false;
        BaselineLaunch();
        backend.Synchronize();
        float baselineError = MaximumError(backend.DownloadBuffer(baselineOutput), expected);
        MeasureAndPrint(run, "AiDotNet NVRTC", backend, BaselineLaunch,
            baselineError, temporaryBytes: 0);
        MeasureGraphAndPrint(run, "AiDotNet NVRTC graph", backend, BaselineLaunch,
            baselineError, temporaryBytes: 0);

        try
        {
            CuBlasLtLaunch();
            backend.Synchronize();
            float cublasLtError = MaximumError(backend.DownloadBuffer(cublasLtOutput), expected);
            MeasureAndPrint(run, "NVIDIA cuBLASLt", backend, CuBlasLtLaunch,
                cublasLtError, temporaryBytes: N * sizeof(int));
            MeasureGraphAndPrint(run, "cuBLASLt graph", backend, CuBlasLtLaunch,
                cublasLtError, temporaryBytes: N * sizeof(int));
        }
        catch (Exception exception)
        {
            Console.WriteLine(
                $"{run,3} m1-k1024-n4096      {"NVIDIA cuBLASLt",-23} INELIGIBLE: " +
                exception.Message.Replace(Environment.NewLine, " ", StringComparison.Ordinal));
        }

        DirectPtxFeatureGate.TestOverride = true;
        if (!backend.PrewarmDirectPtxFusedLinearGeluW8A8M1(K, N))
            throw new InvalidOperationException(backend.DirectPtxLastError);
        DirectLaunch();
        backend.Synchronize();
        float directError = MaximumError(backend.DownloadBuffer(directOutput), expected);
        if (!backend.TryGetDirectPtxQuantizedLinearAudit(K, N, out DirectPtxKernelAudit audit))
            throw new InvalidOperationException("No audit for the measured W8A8 PTX module.");
        MeasureAndPrint(run, "Direct PTX DP4A", backend, DirectLaunch,
            directError, temporaryBytes: 0, audit);
        MeasureGraphAndPrint(run, "Direct PTX graph", backend, DirectLaunch,
            directError, temporaryBytes: 0, audit);
        Console.WriteLine(
            $"    audit={audit.BlueprintId} sha={audit.PtxSha256} " +
            $"device={audit.DeviceFingerprint}");
    }

    private static void MeasureAndPrint(
        int run, string method, CudaBackend backend, Action launch,
        float error, long temporaryBytes, DirectPtxKernelAudit? audit = null)
    {
        Distribution device = MeasureDevice(backend, launch);
        Distribution e2e = MeasureEndToEnd(backend, launch);
        long allocation = MeasureAllocation(backend, launch);
        Print(run, method, device, e2e, allocation, temporaryBytes, error, audit);
    }

    private static void MeasureGraphAndPrint(
        int run, string method, CudaBackend backend, Action launch,
        float error, long temporaryBytes, DirectPtxKernelAudit? audit = null)
    {
        IntPtr graph = backend.CaptureGraph(launch);
        try
        {
            void GraphLaunch() => backend.LaunchCapturedGraph(graph);
            MeasureAndPrint(run, method, backend, GraphLaunch, error, temporaryBytes, audit);
        }
        finally { backend.DestroyCapturedGraph(graph); }
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
        double scale = 1_000_000.0 / Stopwatch.Frequency;
        for (int sample = 0; sample < values.Length; sample++)
        {
            long start = Stopwatch.GetTimestamp();
            action();
            backend.Synchronize();
            values[sample] = (Stopwatch.GetTimestamp() - start) * scale;
        }
        return Summarize(values);
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

    private static Distribution Summarize(double[] values)
    {
        Array.Sort(values);
        return new Distribution(values.Average(), Percentile(values, 0.50),
            Percentile(values, 0.95), Percentile(values, 0.99));
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
            "run shape                 method                  dev mean/med/p95/p99 us       " +
            "e2e mean/med/p95/p99 us       TOPS    GOPS  allocB  tmpB err      R/S/L/B");
        Console.WriteLine(new string('-', 178));
    }

    private static void Print(
        int run, string method, Distribution device, Distribution e2e,
        long allocation, long temporaryBytes, float error, DirectPtxKernelAudit? audit)
    {
        double operations = 2.0 * K * N;
        double tops = operations / (device.Median * 1e-6) / 1e12;
        string resources = audit is null ? "-/-/-/-" :
            $"{audit.Function.RegistersPerThread}/{audit.Function.StaticSharedBytes}/" +
            $"{audit.Function.LocalBytesPerThread}/{audit.ActiveBlocksPerMultiprocessor}";
        Console.WriteLine(
            $"{run,3} m1-k1024-n4096      {method,-23} " +
            $"{device.Mean,6:F2}/{device.Median,6:F2}/{device.P95,6:F2}/{device.P99,6:F2}   " +
            $"{e2e.Mean,6:F2}/{e2e.Median,6:F2}/{e2e.P95,6:F2}/{e2e.P99,6:F2}   " +
            $"{tops,6:F3} {tops * 1_000.0,7:F1} {allocation,7} {temporaryBytes,5} " +
            $"{error,8:E1} {resources}");
        var c = System.Globalization.CultureInfo.InvariantCulture;
        int localBytes = audit?.Function.LocalBytesPerThread ?? -1;
        Console.WriteLine(
            "w8a8_evidence_json={" +
            $"\"rows\":{K},\"columns\":{N}," +
            $"\"method\":\"{method}\"," +
            $"\"median_us\":{device.Median.ToString("R", c)}," +
            $"\"p95_us\":{device.P95.ToString("R", c)}," +
            $"\"managed_bytes\":{allocation},\"temp_bytes\":{temporaryBytes}," +
            $"\"max_error\":{error.ToString("R", c)},\"tolerance\":5e-4,\"local_bytes\":{localBytes}" +
            "}");
    }

    private static void PrintPyTorchEligibility()
    {
        string script = Path.Combine(AppContext.BaseDirectory, "BaselineRunners", "py",
            "run_direct_ptx_w8a8_linear_competitors.py");
        if (!File.Exists(script))
            script = Path.Combine(AppContext.BaseDirectory,
                "run_direct_ptx_w8a8_linear_competitors.py");
        if (!File.Exists(script))
        {
            Console.WriteLine("PyTorch _int_mm: INELIGIBLE (eligibility probe was not copied).");
            return;
        }
        var start = new ProcessStartInfo
        {
            FileName = Environment.GetEnvironmentVariable("PYTHON") ?? "python",
            UseShellExecute = false,
            RedirectStandardOutput = true,
            RedirectStandardError = true
        };
        start.ArgumentList.Add(script);
        using Process process = Process.Start(start) ??
            throw new InvalidOperationException("Could not start the PyTorch W8A8 eligibility probe.");
        string line = process.StandardOutput.ReadLine() ?? string.Empty;
        string stderr = process.StandardError.ReadToEnd();
        process.WaitForExit();
        if (process.ExitCode != 0)
            throw new InvalidOperationException(
                $"PyTorch W8A8 eligibility probe exited {process.ExitCode}: {stderr}");
        JsonElement root = JsonDocument.Parse(line).RootElement;
        Console.WriteLine(
            $"PyTorch {root.GetProperty("method").GetString()}: " +
            $"{root.GetProperty("status").GetString()!.ToUpperInvariant()} " +
            $"({root.GetProperty("reason").GetString()})");
    }

    private static sbyte[] QuantizedValues(Random random, int count)
    {
        var result = new sbyte[count];
        for (int i = 0; i < result.Length; i++) result[i] = (sbyte)random.Next(-16, 17);
        return result;
    }

    private static byte[] ToBytes(sbyte[] values)
    {
        var result = new byte[values.Length];
        Buffer.BlockCopy(values, 0, result, 0, result.Length);
        return result;
    }

    private static sbyte[] TransposeOutputMajor(sbyte[] outputMajor, int inputFeatures, int outputFeatures)
    {
        var inputMajor = new sbyte[outputMajor.Length];
        for (int output = 0; output < outputFeatures; output++)
        for (int input = 0; input < inputFeatures; input++)
            inputMajor[input * outputFeatures + output] =
                outputMajor[output * inputFeatures + input];
        return inputMajor;
    }

    private static float[] Oracle(
        sbyte[] input, sbyte[] weights, float activationScale,
        float[] weightScales, float[] bias)
    {
        var output = new float[N];
        for (int column = 0; column < N; column++)
        {
            int accumulator = 0;
            int weightBase = column * K;
            for (int inner = 0; inner < K; inner++)
                accumulator += input[inner] * weights[weightBase + inner];
            float value = accumulator * activationScale * weightScales[column] + bias[column];
            output[column] = 0.5f * value *
                (1f + MathF.Tanh(0.7978845608f *
                    (value + 0.044715f * value * value * value)));
        }
        return output;
    }

    private static float MaximumError(float[] actual, float[] expected)
    {
        float maximum = 0;
        for (int i = 0; i < actual.Length; i++)
            maximum = MathF.Max(maximum, MathF.Abs(actual[i] - expected[i]));
        return maximum;
    }
}
