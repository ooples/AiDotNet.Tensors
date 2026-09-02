using System.Diagnostics;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;
using AiDotNet.Tensors.Engines.Gpu;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Tensors.Benchmarks;

/// <summary>Issue #839 exact contiguous FP32 tanh-GeGLU championship.</summary>
internal static class DirectPtxGeGluExperiment
{
    private const int Warmups = 30;
    private const int Samples = 101;
    private const int DeviceLaunches = 50;
    private static double _flopsPerOutput = 10.0;
    private static double _bytesPerOutput = 12.0;
    private static readonly (int Outer, int HalfDimension)[] Shapes =
        [(1, 4096), (32, 4096), (256, 4096), (256, 11008)];

    private readonly record struct Distribution(double Mean, double Median, double P95, double P99);

    internal static void Run(int independentRuns = 3)
    {
        if (independentRuns <= 0) throw new ArgumentOutOfRangeException(nameof(independentRuns));
        bool? previousGate = DirectPtxFeatureGate.TestOverride;
        bool previousExperiment = DirectPtxFeatureGate.GeGluExperimentOverride;
        DirectPtxFeatureGate.GeGluExperimentOverride = true;
        _flopsPerOutput = 10.0;
        _bytesPerOutput = 12.0;
        try
        {
            Console.WriteLine(
                $"Direct-PTX FP32 tanh-GeGLU: {independentRuns} run(s), " +
                $"{Warmups} warmups + {Samples} samples/cell");
            Console.WriteLine(
                $"Device samples average {DeviceLaunches} launches; useful model is " +
                "10 FLOPs/output and 12 useful bytes/output.");
            PrintHeader();
            for (int run = 1; run <= independentRuns; run++)
            foreach ((int outer, int halfDimension) in Shapes)
                RunCell(run, outer, halfDimension);
            RunPyTorchCompetitors(independentRuns);
        }
        finally
        {
            DirectPtxFeatureGate.TestOverride = previousGate;
            DirectPtxFeatureGate.GeGluExperimentOverride = previousExperiment;
        }
    }

    internal static void RunBackward(int independentRuns = 3)
    {
        if (independentRuns <= 0) throw new ArgumentOutOfRangeException(nameof(independentRuns));
        bool? previousGate = DirectPtxFeatureGate.TestOverride;
        bool previousExperiment = DirectPtxFeatureGate.GeGluExperimentOverride;
        DirectPtxFeatureGate.GeGluExperimentOverride = true;
        _flopsPerOutput = 26.0;
        _bytesPerOutput = 20.0;
        try
        {
            Console.WriteLine(
                $"Direct-PTX FP32 tanh-GeGLU backward: {independentRuns} run(s), " +
                $"{Warmups} warmups + {Samples} samples/cell");
            Console.WriteLine(
                $"Device samples average {DeviceLaunches} launches; useful model is " +
                "26 FLOPs/output and 20 useful bytes/output.");
            PrintHeader();
            for (int run = 1; run <= independentRuns; run++)
            foreach ((int outer, int halfDimension) in Shapes)
                RunBackwardCell(run, outer, halfDimension);
            RunPyTorchBackwardCompetitors(independentRuns);
        }
        finally
        {
            DirectPtxFeatureGate.TestOverride = previousGate;
            DirectPtxFeatureGate.GeGluExperimentOverride = previousExperiment;
        }
    }

    internal static void RunSwiGluForward(int independentRuns = 3)
    {
        if (independentRuns <= 0) throw new ArgumentOutOfRangeException(nameof(independentRuns));
        bool? previousGate = DirectPtxFeatureGate.TestOverride;
        bool previousExperiment = DirectPtxFeatureGate.SwiGluExperimentOverride;
        DirectPtxFeatureGate.SwiGluExperimentOverride = true;
        _flopsPerOutput = 10.0;
        _bytesPerOutput = 12.0;
        try
        {
            Console.WriteLine(
                $"Direct-PTX FP32 SwiGLU: {independentRuns} run(s), " +
                $"{Warmups} warmups + {Samples} samples/cell");
            PrintHeader();
            for (int run = 1; run <= independentRuns; run++)
            foreach ((int outer, int halfDimension) in Shapes)
                RunSwiGluCell(run, outer, halfDimension);
        }
        finally
        {
            DirectPtxFeatureGate.TestOverride = previousGate;
            DirectPtxFeatureGate.SwiGluExperimentOverride = previousExperiment;
        }
    }

    private static void RunSwiGluCell(int run, int outerSize, int halfDimension)
    {
        using var backend = new CudaBackend();
        if (run == 1 && outerSize == Shapes[0].Outer && halfDimension == Shapes[0].HalfDimension)
            Console.WriteLine($"GPU: {backend.DeviceName}");
        int outputElements = checked(outerSize * halfDimension);
        int inputElements = checked(2 * outputElements);
        var random = RandomHelper.CreateSeededRandom(20262100 + run * 100 + outerSize + halfDimension);
        float[] inputHost = Enumerable.Range(0, inputElements)
            .Select(_ => (random.NextSingle() * 2f - 1f) * 2f).ToArray();
        float[] expected = SwiGluOracle(inputHost, outerSize, halfDimension);
        using var input = backend.AllocateBuffer(inputHost);
        using var baselineOutput = backend.AllocateBuffer(outputElements);
        using var directOutput = backend.AllocateBuffer(outputElements);

        void BaselineLaunch() => backend.SwiGluForward(
            input, baselineOutput, outerSize, halfDimension);
        void DirectLaunch()
        {
            if (!backend.TryDirectPtxSwiGluForward(
                input, directOutput, outerSize, halfDimension))
                throw new InvalidOperationException(backend.DirectPtxLastError);
        }

        DirectPtxFeatureGate.TestOverride = false;
        BaselineLaunch();
        backend.Synchronize();
        float baselineError = MaximumError(backend.DownloadBuffer(baselineOutput), expected);
        MeasureAndPrint(run, outerSize, halfDimension, "AiDotNet CUDA eager", backend,
            BaselineLaunch, baselineError, temporaryBytes: 0);
        MeasureGraphAndPrint(run, outerSize, halfDimension, "AiDotNet CUDA graph", backend,
            BaselineLaunch, baselineError, temporaryBytes: 0);

        DirectPtxFeatureGate.TestOverride = true;
        if (!backend.PrewarmDirectPtxSwiGluForward(outerSize, halfDimension))
            throw new InvalidOperationException(backend.DirectPtxLastError);
        DirectLaunch();
        backend.Synchronize();
        float directError = MaximumError(backend.DownloadBuffer(directOutput), expected);
        if (!backend.TryGetDirectPtxSwiGluAudit(
            outerSize, halfDimension, out DirectPtxKernelAudit audit))
            throw new InvalidOperationException("No audit for measured SwiGLU PTX module.");
        MeasureAndPrint(run, outerSize, halfDimension, "Direct PTX eager", backend,
            DirectLaunch, directError, temporaryBytes: 0, audit);
        MeasureGraphAndPrint(run, outerSize, halfDimension, "Direct PTX graph", backend,
            DirectLaunch, directError, temporaryBytes: 0, audit);
    }

    // Reference SwiGLU: output = value * SiLU(gate), gate = input second half.
    private static float[] SwiGluOracle(float[] input, int outerSize, int halfDimension)
    {
        var output = new float[outerSize * halfDimension];
        for (int row = 0; row < outerSize; row++)
        {
            int inputBase = row * halfDimension * 2;
            int outputBase = row * halfDimension;
            for (int feature = 0; feature < halfDimension; feature++)
            {
                double gate = input[inputBase + halfDimension + feature];
                double silu = gate / (1.0 + Math.Exp(-gate));
                output[outputBase + feature] = (float)(input[inputBase + feature] * silu);
            }
        }
        return output;
    }

    private static void RunCell(int run, int outerSize, int halfDimension)
    {
        using var backend = new CudaBackend();
        if (run == 1 && outerSize == Shapes[0].Outer && halfDimension == Shapes[0].HalfDimension)
            Console.WriteLine($"GPU: {backend.DeviceName}");
        int outputElements = checked(outerSize * halfDimension);
        int inputElements = checked(2 * outputElements);
        var random = RandomHelper.CreateSeededRandom(20261900 + run * 100 + outerSize + halfDimension);
        float[] inputHost = Enumerable.Range(0, inputElements)
            .Select(_ => (random.NextSingle() * 2f - 1f) * 2f).ToArray();
        float[] expected = Oracle(inputHost, outerSize, halfDimension);
        using var input = backend.AllocateBuffer(inputHost);
        using var baselineOutput = backend.AllocateBuffer(outputElements);
        using var directOutput = backend.AllocateBuffer(outputElements);

        void BaselineLaunch() => backend.GeGluForward(
            input, baselineOutput, outerSize, halfDimension);
        void DirectLaunch()
        {
            if (!backend.TryDirectPtxGeGluForward(
                input, directOutput, outerSize, halfDimension))
                throw new InvalidOperationException(backend.DirectPtxLastError);
        }

        DirectPtxFeatureGate.TestOverride = false;
        BaselineLaunch();
        backend.Synchronize();
        float baselineError = MaximumError(backend.DownloadBuffer(baselineOutput), expected);
        MeasureAndPrint(run, outerSize, halfDimension, "AiDotNet CUDA eager", backend,
            BaselineLaunch, baselineError, temporaryBytes: 0);
        MeasureGraphAndPrint(run, outerSize, halfDimension, "AiDotNet CUDA graph", backend,
            BaselineLaunch, baselineError, temporaryBytes: 0);

        DirectPtxFeatureGate.TestOverride = true;
        if (!backend.PrewarmDirectPtxGeGluForward(outerSize, halfDimension))
            throw new InvalidOperationException(backend.DirectPtxLastError);
        DirectLaunch();
        backend.Synchronize();
        float directError = MaximumError(backend.DownloadBuffer(directOutput), expected);
        if (!backend.TryGetDirectPtxGeGluAudit(
            outerSize, halfDimension, out DirectPtxKernelAudit audit))
            throw new InvalidOperationException("No audit for measured GeGLU PTX module.");
        MeasureAndPrint(run, outerSize, halfDimension, "Direct PTX eager", backend,
            DirectLaunch, directError, temporaryBytes: 0, audit);
        MeasureGraphAndPrint(run, outerSize, halfDimension, "Direct PTX graph", backend,
            DirectLaunch, directError, temporaryBytes: 0, audit);
        Console.WriteLine(
            $"    audit={audit.BlueprintId} sha={audit.PtxSha256} device={audit.DeviceFingerprint}");
    }

    private static void RunBackwardCell(int run, int outerSize, int halfDimension)
    {
        using var backend = new CudaBackend();
        if (run == 1 && outerSize == Shapes[0].Outer && halfDimension == Shapes[0].HalfDimension)
            Console.WriteLine($"GPU: {backend.DeviceName}");
        int outputElements = checked(outerSize * halfDimension);
        int inputElements = checked(2 * outputElements);
        var random = RandomHelper.CreateSeededRandom(20262000 + run * 100 + outerSize + halfDimension);
        float[] inputHost = Enumerable.Range(0, inputElements)
            .Select(_ => (random.NextSingle() * 2f - 1f) * 2f).ToArray();
        float[] gradOutputHost = Enumerable.Range(0, outputElements)
            .Select(_ => random.NextSingle() * 2f - 1f).ToArray();
        float[] expected = BackwardOracle(
            gradOutputHost, inputHost, outerSize, halfDimension);
        using var gradOutput = backend.AllocateBuffer(gradOutputHost);
        using var input = backend.AllocateBuffer(inputHost);
        using var baselineGradInput = backend.AllocateBuffer(inputElements);
        using var directGradInput = backend.AllocateBuffer(inputElements);

        void BaselineLaunch() => backend.GeGluBackward(
            gradOutput, input, baselineGradInput, outerSize, halfDimension);
        void DirectLaunch()
        {
            if (!backend.TryDirectPtxGeGluBackward(
                gradOutput, input, directGradInput, outerSize, halfDimension))
                throw new InvalidOperationException(backend.DirectPtxLastError);
        }

        DirectPtxFeatureGate.TestOverride = false;
        BaselineLaunch();
        backend.Synchronize();
        float baselineError = MaximumError(backend.DownloadBuffer(baselineGradInput), expected);
        MeasureAndPrint(run, outerSize, halfDimension, "AiDotNet CUDA eager", backend,
            BaselineLaunch, baselineError, temporaryBytes: 0);
        MeasureGraphAndPrint(run, outerSize, halfDimension, "AiDotNet CUDA graph", backend,
            BaselineLaunch, baselineError, temporaryBytes: 0);

        DirectPtxFeatureGate.TestOverride = true;
        if (!backend.PrewarmDirectPtxGeGluBackward(outerSize, halfDimension))
            throw new InvalidOperationException(backend.DirectPtxLastError);
        DirectLaunch();
        backend.Synchronize();
        float directError = MaximumError(backend.DownloadBuffer(directGradInput), expected);
        if (!backend.TryGetDirectPtxGeGluBackwardAudit(
            outerSize, halfDimension, out DirectPtxKernelAudit audit))
            throw new InvalidOperationException("No audit for measured GeGLU-backward PTX module.");
        MeasureAndPrint(run, outerSize, halfDimension, "Direct PTX eager", backend,
            DirectLaunch, directError, temporaryBytes: 0, audit);
        MeasureGraphAndPrint(run, outerSize, halfDimension, "Direct PTX graph", backend,
            DirectLaunch, directError, temporaryBytes: 0, audit);
        Console.WriteLine(
            $"    audit={audit.BlueprintId} sha={audit.PtxSha256} device={audit.DeviceFingerprint}");
    }

    private static void MeasureAndPrint(
        int run, int outerSize, int halfDimension, string method,
        CudaBackend backend, Action launch, float error, long temporaryBytes,
        DirectPtxKernelAudit? audit = null)
    {
        Distribution device = MeasureDevice(backend, launch);
        Distribution e2e = MeasureEndToEnd(backend, launch);
        long allocation = MeasureAllocation(backend, launch);
        Print(run, outerSize, halfDimension, method, device, e2e,
            allocation, temporaryBytes, error, audit);
    }

    private static void MeasureGraphAndPrint(
        int run, int outerSize, int halfDimension, string method,
        CudaBackend backend, Action launch, float error, long temporaryBytes,
        DirectPtxKernelAudit? audit = null)
    {
        Distribution device = MeasureDevice(backend, launch);
        IntPtr graph = backend.CaptureGraph(launch);
        try
        {
            void GraphLaunch() => backend.LaunchCapturedGraph(graph);
            Distribution e2e = MeasureEndToEnd(backend, GraphLaunch);
            long allocation = MeasureAllocation(backend, GraphLaunch);
            Print(run, outerSize, halfDimension, method, device, e2e,
                allocation, temporaryBytes, error, audit);
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
            "run outer D     method                    dev mean/med/p95/p99 us       " +
            "e2e mean/med/p95/p99 us       GFLOPS  GB/s allocB tmpB err      R/S/L/B");
        Console.WriteLine(new string('-', 176));
    }

    private static void Print(
        int run, int outerSize, int halfDimension, string method,
        Distribution device, Distribution e2e, long allocation,
        long temporaryBytes, float error, DirectPtxKernelAudit? audit)
    {
        double outputs = (double)outerSize * halfDimension;
        double gflops = outputs * _flopsPerOutput / (device.Median * 1e-6) / 1e9;
        double bandwidth = outputs * _bytesPerOutput / (device.Median * 1e-6) / 1e9;
        string resources = audit is null ? "-/-/-/-" :
            $"{audit.Function.RegistersPerThread}/{audit.Function.StaticSharedBytes}/" +
            $"{audit.Function.LocalBytesPerThread}/{audit.ActiveBlocksPerMultiprocessor}";
        Console.WriteLine(
            $"{run,3} {outerSize,5} {halfDimension,5} {method,-25} " +
            $"{device.Mean,6:F2}/{device.Median,6:F2}/{device.P95,6:F2}/{device.P99,6:F2}   " +
            $"{e2e.Mean,6:F2}/{e2e.Median,6:F2}/{e2e.P95,6:F2}/{e2e.P99,6:F2}   " +
            $"{gflops,7:F1} {bandwidth,5:F1} {allocation,6} {temporaryBytes,4} " +
            $"{error,8:E1} {resources}");
        var c = System.Globalization.CultureInfo.InvariantCulture;
        int localBytes = audit?.Function.LocalBytesPerThread ?? -1;
        Console.WriteLine(
            "glu_evidence_json={" +
            $"\"rows\":{outerSize},\"columns\":{halfDimension}," +
            $"\"method\":\"{method}\"," +
            $"\"median_us\":{device.Median.ToString("R", c)}," +
            $"\"p95_us\":{device.P95.ToString("R", c)}," +
            $"\"managed_bytes\":{allocation},\"temp_bytes\":{temporaryBytes}," +
            $"\"max_error\":{error.ToString("R", c)},\"tolerance\":2e-4,\"local_bytes\":{localBytes}" +
            "}");
    }

    private static float[] Oracle(float[] input, int outerSize, int halfDimension)
    {
        var output = new float[outerSize * halfDimension];
        for (int row = 0; row < outerSize; row++)
        {
            int inputBase = row * halfDimension * 2;
            int outputBase = row * halfDimension;
            for (int feature = 0; feature < halfDimension; feature++)
            {
                double gate = input[inputBase + halfDimension + feature];
                double inner = 0.7978845608 * (gate + 0.044715 * gate * gate * gate);
                double gelu = 0.5 * gate * (1.0 + Math.Tanh(inner));
                output[outputBase + feature] = (float)(input[inputBase + feature] * gelu);
            }
        }
        return output;
    }

    private static float[] BackwardOracle(
        float[] gradOutput, float[] input, int outerSize, int halfDimension)
    {
        var gradInput = new float[outerSize * halfDimension * 2];
        for (int row = 0; row < outerSize; row++)
        {
            int inputBase = row * halfDimension * 2;
            int outputBase = row * halfDimension;
            for (int feature = 0; feature < halfDimension; feature++)
            {
                double value = input[inputBase + feature];
                double gate = input[inputBase + halfDimension + feature];
                double grad = gradOutput[outputBase + feature];
                double inner = 0.7978845608 * (gate + 0.044715 * gate * gate * gate);
                double tanh = Math.Tanh(inner);
                double gelu = 0.5 * gate * (1.0 + tanh);
                double derivative = 0.5 * (1.0 + tanh) +
                    0.5 * gate * (1.0 - tanh * tanh) * 0.7978845608 *
                    (1.0 + 0.134145 * gate * gate);
                gradInput[inputBase + feature] = (float)(grad * gelu);
                gradInput[inputBase + halfDimension + feature] =
                    (float)(grad * value * derivative);
            }
        }
        return gradInput;
    }

    private static float MaximumError(float[] actual, float[] expected)
    {
        float maximum = 0;
        for (int i = 0; i < actual.Length; i++)
            maximum = MathF.Max(maximum, MathF.Abs(actual[i] - expected[i]));
        return maximum;
    }

    private static void RunPyTorchCompetitors(int independentRuns)
    {
        string script = Path.Combine(AppContext.BaseDirectory, "BaselineRunners", "py",
            "run_direct_ptx_geglu_competitors.py");
        if (!File.Exists(script))
            script = Path.Combine(AppContext.BaseDirectory, "run_direct_ptx_geglu_competitors.py");
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
        start.ArgumentList.Add(independentRuns.ToString(
            System.Globalization.CultureInfo.InvariantCulture));
        using Process process = Process.Start(start) ??
            throw new InvalidOperationException("Could not start the PyTorch GeGLU runner.");
        process.WaitForExit();
        if (process.ExitCode != 0)
            throw new InvalidOperationException(
                $"PyTorch GeGLU runner exited with code {process.ExitCode}.");
    }

    private static void RunPyTorchBackwardCompetitors(int independentRuns)
    {
        string script = Path.Combine(AppContext.BaseDirectory, "BaselineRunners", "py",
            "run_direct_ptx_geglu_competitors.py");
        if (!File.Exists(script))
            script = Path.Combine(AppContext.BaseDirectory, "run_direct_ptx_geglu_competitors.py");
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
        start.ArgumentList.Add(independentRuns.ToString(
            System.Globalization.CultureInfo.InvariantCulture));
        start.ArgumentList.Add("--phase");
        start.ArgumentList.Add("backward");
        using Process process = Process.Start(start) ??
            throw new InvalidOperationException("Could not start the PyTorch GeGLU-backward runner.");
        process.WaitForExit();
        if (process.ExitCode != 0)
            throw new InvalidOperationException(
                $"PyTorch GeGLU-backward runner exited with code {process.ExitCode}.");
    }
}
