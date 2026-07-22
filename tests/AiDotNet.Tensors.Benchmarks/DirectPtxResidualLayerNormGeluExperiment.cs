using System.Diagnostics;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;
using AiDotNet.Tensors.Engines.Gpu;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Tensors.Benchmarks;

/// <summary>Issue #838 D=64 residual+bias+LayerNorm+tanh-GELU championship.</summary>
internal static class DirectPtxResidualLayerNormGeluExperiment
{
    private const int Dimension = 64;
    private const int Warmups = 30;
    private const int Samples = 101;
    private const int DeviceLaunches = 50;
    private static readonly int[] RowBuckets = [256, 2048, 8192];

    private readonly record struct Distribution(double Mean, double Median, double P95, double P99);

    internal static void Run(int independentRuns = 3)
    {
        if (independentRuns <= 0) throw new ArgumentOutOfRangeException(nameof(independentRuns));
        bool? previousGate = DirectPtxFeatureGate.TestOverride;
        bool previousExperiment = DirectPtxFeatureGate.NormalizationExperimentOverride;
        DirectPtxFeatureGate.NormalizationExperimentOverride = true;
        try
        {
            Console.WriteLine(
                $"Direct-PTX residual+bias+LayerNorm+tanh-GELU D={Dimension}: " +
                $"{independentRuns} run(s), {Warmups} warmups + {Samples} samples/cell");
            Console.WriteLine(
                $"Device samples average {DeviceLaunches} launches; useful FLOP model is " +
                "19*rows*D+3*rows (sqrt/rcp/tanh count as one each).");
            PrintHeader();
            for (int run = 1; run <= independentRuns; run++)
            foreach (int rows in RowBuckets)
                RunCell(run, rows);
            RunPyTorchCompetitors(independentRuns);
        }
        finally
        {
            DirectPtxFeatureGate.TestOverride = previousGate;
            DirectPtxFeatureGate.NormalizationExperimentOverride = previousExperiment;
        }
    }

    private static void RunCell(int run, int rows)
    {
        using var backend = new CudaBackend();
        if (run == 1 && rows == RowBuckets[0]) Console.WriteLine($"GPU: {backend.DeviceName}");
        int elements = checked(rows * Dimension);
        var random = RandomHelper.CreateSeededRandom(20261800 + run * 100 + rows);
        float[] inputHost = Values(random, elements, 1f);
        float[] residualHost = Values(random, elements, 0.25f);
        float[] biasHost = Values(random, Dimension, 0.05f);
        float[] gammaHost = Enumerable.Range(0, Dimension)
            .Select(i => 0.75f + i / 256f).ToArray();
        float[] betaHost = Values(random, Dimension, 0.025f);
        float[] expected = Oracle(
            inputHost, residualHost, biasHost, gammaHost, betaHost, rows, 1e-5f);

        using var input = backend.AllocateBuffer(inputHost);
        using var residual = backend.AllocateBuffer(residualHost);
        using var bias = backend.AllocateBuffer(biasHost);
        using var gamma = backend.AllocateBuffer(gammaHost);
        using var beta = backend.AllocateBuffer(betaHost);
        using var baselineOutput = backend.AllocateBuffer(elements);
        using var directOutput = backend.AllocateBuffer(elements);

        void BaselineLaunch() => backend.FusedResidualBiasLayerNormGeluD64(
            input, residual, bias, gamma, beta, baselineOutput, rows);
        void DirectLaunch()
        {
            if (!backend.TryDirectPtxFusedResidualBiasLayerNormGeluD64(
                input, residual, bias, gamma, beta, directOutput, rows))
                throw new InvalidOperationException(backend.DirectPtxLastError);
        }

        DirectPtxFeatureGate.TestOverride = false;
        BaselineLaunch();
        backend.Synchronize();
        float baselineError = MaximumError(backend.DownloadBuffer(baselineOutput), expected);
        MeasureAndPrint(run, rows, "AiDotNet CUDA eager", backend, BaselineLaunch,
            baselineError, temporaryBytes: 0);
        MeasureGraphAndPrint(run, rows, "AiDotNet CUDA graph", backend, BaselineLaunch,
            baselineError, temporaryBytes: 0);

        DirectPtxFeatureGate.TestOverride = true;
        if (!backend.PrewarmDirectPtxFusedResidualBiasLayerNormGeluD64(rows))
            throw new InvalidOperationException(backend.DirectPtxLastError);
        DirectLaunch();
        backend.Synchronize();
        float directError = MaximumError(backend.DownloadBuffer(directOutput), expected);
        if (!backend.TryGetDirectPtxResidualLayerNormGeluAudit(
            rows, 1e-5f, out DirectPtxKernelAudit audit))
            throw new InvalidOperationException("No audit for measured normalization PTX module.");
        MeasureAndPrint(run, rows, "Direct PTX eager", backend, DirectLaunch,
            directError, temporaryBytes: 0, audit);
        MeasureGraphAndPrint(run, rows, "Direct PTX graph", backend, DirectLaunch,
            directError, temporaryBytes: 0, audit);
        Console.WriteLine(
            $"    audit={audit.BlueprintId} sha={audit.PtxSha256} device={audit.DeviceFingerprint}");
    }

    private static void MeasureAndPrint(
        int run,
        int rows,
        string method,
        CudaBackend backend,
        Action launch,
        float error,
        long temporaryBytes,
        DirectPtxKernelAudit? audit = null)
    {
        Distribution device = MeasureDevice(backend, launch);
        Distribution e2e = MeasureEndToEnd(backend, launch);
        long allocation = MeasureAllocation(backend, launch);
        Print(run, rows, method, device, e2e, allocation, temporaryBytes, error, audit);
    }

    private static void MeasureGraphAndPrint(
        int run,
        int rows,
        string method,
        CudaBackend backend,
        Action launch,
        float error,
        long temporaryBytes,
        DirectPtxKernelAudit? audit = null)
    {
        Distribution device = MeasureDevice(backend, launch);
        IntPtr graph = backend.CaptureGraph(launch);
        try
        {
            void GraphLaunch() => backend.LaunchCapturedGraph(graph);
            Distribution e2e = MeasureEndToEnd(backend, GraphLaunch);
            long allocation = MeasureAllocation(backend, GraphLaunch);
            Print(run, rows, method, device, e2e, allocation,
                temporaryBytes, error, audit);
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
        IntPtr batchGraph = backend.CaptureGraph(() =>
        {
            for (int launch = 0; launch < DeviceLaunches; launch++) action();
        });
        try
        {
            for (int i = 0; i < Warmups; i++) backend.LaunchCapturedGraph(batchGraph);
            backend.Synchronize();
            for (int sample = 0; sample < values.Length; sample++)
            {
                backend.RecordEvent(start, backend.DefaultStream);
                backend.LaunchCapturedGraph(batchGraph);
                backend.RecordEvent(end, backend.DefaultStream);
                end.Synchronize();
                values[sample] = backend.GetEventElapsedTime(start, end) * 1_000.0 / DeviceLaunches;
            }
        }
        finally { backend.DestroyCapturedGraph(batchGraph); }
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
        long before = PtxCompat.GetAllocatedBytesForCurrentThread();
        for (int i = 0; i < Samples; i++) action();
        long bytes = (PtxCompat.GetAllocatedBytesForCurrentThread() - before) / Samples;
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
            "run rows  method                    dev mean/med/p95/p99 us       " +
            "e2e mean/med/p95/p99 us       GFLOPS  GB/s allocB tmpB err      R/S/L/B");
        Console.WriteLine(new string('-', 164));
    }

    private static void Print(
        int run,
        int rows,
        string method,
        Distribution device,
        Distribution e2e,
        long allocation,
        long temporaryBytes,
        float error,
        DirectPtxKernelAudit? audit)
    {
        double operations = (19.0 * Dimension + 3.0) * rows;
        double gflops = operations / (device.Median * 1e-6) / 1e9;
        double usefulBytes = (3.0 * rows * Dimension + 3.0 * Dimension) * sizeof(float);
        double bandwidth = usefulBytes / (device.Median * 1e-6) / 1e9;
        string resources = audit is null ? "-/-/-/-" :
            $"{audit.Function.RegistersPerThread}/{audit.Function.StaticSharedBytes}/" +
            $"{audit.Function.LocalBytesPerThread}/{audit.ActiveBlocksPerMultiprocessor}";
        Console.WriteLine(
            $"{run,3} {rows,5} {method,-25} " +
            $"{device.Mean,6:F2}/{device.Median,6:F2}/{device.P95,6:F2}/{device.P99,6:F2}   " +
            $"{e2e.Mean,6:F2}/{e2e.Median,6:F2}/{e2e.P95,6:F2}/{e2e.P99,6:F2}   " +
            $"{gflops,7:F1} {bandwidth,5:F1} {allocation,6} {temporaryBytes,4} " +
            $"{error,8:E1} {resources}");
        var c = System.Globalization.CultureInfo.InvariantCulture;
        int localBytes = audit?.Function.LocalBytesPerThread ?? -1;
        Console.WriteLine(
            "layernorm_evidence_json={" +
            $"\"rows\":{rows},\"columns\":{Dimension}," +
            $"\"method\":\"{method}\"," +
            $"\"median_us\":{device.Median.ToString("R", c)}," +
            $"\"p95_us\":{device.P95.ToString("R", c)}," +
            $"\"managed_bytes\":{allocation},\"temp_bytes\":{temporaryBytes}," +
            $"\"max_error\":{error.ToString("R", c)},\"tolerance\":2e-4,\"local_bytes\":{localBytes}" +
            "}");
    }

    private static float[] Values(Random random, int count, float scale)
    {
        var values = new float[count];
        for (int i = 0; i < count; i++)
            values[i] = (random.NextSingle() * 2f - 1f) * scale;
        return values;
    }

    private static float[] Oracle(
        float[] input,
        float[] residual,
        float[] bias,
        float[] gamma,
        float[] beta,
        int rows,
        float epsilon)
    {
        var output = new float[input.Length];
        var rowValues = new double[Dimension];
        for (int row = 0; row < rows; row++)
        {
            int rowBase = row * Dimension;
            double sum = 0;
            for (int feature = 0; feature < Dimension; feature++)
            {
                double value = input[rowBase + feature] + residual[rowBase + feature] + bias[feature];
                rowValues[feature] = value;
                sum += value;
            }
            double mean = sum / Dimension;
            double variance = 0;
            for (int feature = 0; feature < Dimension; feature++)
            {
                double difference = rowValues[feature] - mean;
                variance += difference * difference;
            }
            double inverseStandardDeviation = 1.0 / Math.Sqrt(variance / Dimension + epsilon);
            for (int feature = 0; feature < Dimension; feature++)
            {
                double value = (rowValues[feature] - mean) * inverseStandardDeviation *
                    gamma[feature] + beta[feature];
                output[rowBase + feature] = (float)(0.5 * value *
                    (1.0 + Math.Tanh(0.7978845608 *
                        (value + 0.044715 * value * value * value))));
            }
        }
        return output;
    }

    private static float MaximumError(float[] actual, float[] expected)
    {
        float maximum = 0;
        for (int i = 0; i < actual.Length; i++)
            maximum = PtxCompat.Max(maximum, PtxCompat.Abs(actual[i] - expected[i]));
        return maximum;
    }

    private static void RunPyTorchCompetitors(int independentRuns)
    {
        string script = Path.Combine(AppContext.BaseDirectory, "BaselineRunners", "py",
            "run_direct_ptx_residual_layernorm_gelu_competitors.py");
        if (!File.Exists(script))
            script = Path.Combine(AppContext.BaseDirectory,
                "run_direct_ptx_residual_layernorm_gelu_competitors.py");
        if (!File.Exists(script))
        {
            Console.WriteLine("PyTorch GPU competitors: INELIGIBLE (runner was not copied).");
            return;
        }
        var start = new ProcessStartInfo
        {
            FileName = Environment.GetEnvironmentVariable("PYTHON") ?? "python",
            UseShellExecute = false
        };
        start.ArgumentList.Add(script);
        start.ArgumentList.Add("--runs");
        start.ArgumentList.Add(independentRuns.ToString(System.Globalization.CultureInfo.InvariantCulture));
        using Process process = Process.Start(start) ??
            throw new InvalidOperationException("Could not start the PyTorch normalization runner.");
        process.WaitForExit();
        if (process.ExitCode != 0)
            throw new InvalidOperationException(
                $"PyTorch normalization runner exited with code {process.ExitCode}.");
    }
}
