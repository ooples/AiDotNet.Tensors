using System.Diagnostics;
using System.Runtime.InteropServices;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;
using AiDotNet.Tensors.Engines.Gpu;

namespace AiDotNet.Tensors.Benchmarks;

/// <summary>Issue #840 exact contiguous FP32 row-softmax championship.</summary>
internal static class DirectPtxSoftmaxExperiment
{
    private const int Warmups = 30;
    private const int Samples = 101;
    private const int DeviceLaunches = 50;
    private static readonly (int Rows, int Columns)[] Shapes =
        [(256, 128), (2048, 64), (2048, 128), (8192, 128)];

    private readonly record struct Distribution(
        double Mean, double Median, double P95, double P99);

    internal static void Run(int independentRuns = 1)
    {
        if (independentRuns <= 0) throw new ArgumentOutOfRangeException(nameof(independentRuns));
        bool? previousGate = DirectPtxFeatureGate.TestOverride;
        bool previousExperiment = DirectPtxFeatureGate.SoftmaxExperimentOverride;
        DirectPtxFeatureGate.SoftmaxExperimentOverride = true;
        try
        {
            PrintHostFingerprint();
            Console.WriteLine(
                $"Direct-PTX FP32 row softmax: {independentRuns} run(s), " +
                $"{Warmups} warmups + {Samples} samples/cell");
            Console.WriteLine(
                $"Device samples average {DeviceLaunches} launches; useful model is " +
                "5 FLOPs/element and 8 useful bytes/element.");
            PrintHeader();
            for (int run = 1; run <= independentRuns; run++)
            foreach ((int rows, int columns) in Shapes)
                RunCell(run, rows, columns);
            RunCudnnCompetitors(independentRuns);
            RunPyTorchCompetitors(independentRuns);
        }
        finally
        {
            DirectPtxFeatureGate.TestOverride = previousGate;
            DirectPtxFeatureGate.SoftmaxExperimentOverride = previousExperiment;
        }
    }

    internal static void RunCudnnOnly(int independentRuns = 1)
    {
        if (independentRuns <= 0) throw new ArgumentOutOfRangeException(nameof(independentRuns));
        PrintHostFingerprint();
        PrintHeader();
        RunCudnnCompetitors(independentRuns);
    }

    internal static void RunAiDotNetOnly(
        int independentRuns = 1,
        int blockThreads = 0)
    {
        if (independentRuns <= 0) throw new ArgumentOutOfRangeException(nameof(independentRuns));
        if (blockThreads is not (0 or 128 or 256 or 512))
            throw new ArgumentOutOfRangeException(nameof(blockThreads));
        bool? previousGate = DirectPtxFeatureGate.TestOverride;
        bool previousExperiment = DirectPtxFeatureGate.SoftmaxExperimentOverride;
        int previousBlockThreads = DirectPtxFeatureGate.SoftmaxBlockThreadsExperimentOverride;
        DirectPtxFeatureGate.SoftmaxExperimentOverride = true;
        DirectPtxFeatureGate.SoftmaxBlockThreadsExperimentOverride = blockThreads;
        try
        {
            PrintHostFingerprint();
            Console.WriteLine(blockThreads == 0
                ? "Direct PTX launch geometry: selected exact-shape mapping"
                : $"Direct PTX launch geometry: {blockThreads} threads/block");
            PrintHeader();
            for (int run = 1; run <= independentRuns; run++)
            foreach ((int rows, int columns) in Shapes)
                RunCell(run, rows, columns);
        }
        finally
        {
            DirectPtxFeatureGate.TestOverride = previousGate;
            DirectPtxFeatureGate.SoftmaxExperimentOverride = previousExperiment;
            DirectPtxFeatureGate.SoftmaxBlockThreadsExperimentOverride = previousBlockThreads;
        }
    }

    private static void RunCell(int run, int rows, int columns)
    {
        using var backend = new CudaBackend();
        if (run == 1 && rows == Shapes[0].Rows && columns == Shapes[0].Columns)
            Console.WriteLine($"GPU: {backend.DeviceName}");
        int elements = checked(rows * columns);
        float[] inputHost = BuildInput(elements, 20262100 + run * 100 + rows + columns);
        float[] expected = Oracle(inputHost, rows, columns);
        using var input = backend.AllocateBuffer(inputHost);
        using var baselineOutput = backend.AllocateBuffer(elements);
        using var directOutput = backend.AllocateBuffer(elements);

        void BaselineLaunch() => backend.Softmax(input, baselineOutput, rows, columns);
        void DirectLaunch()
        {
            if (!backend.TryDirectPtxSoftmax(input, directOutput, rows, columns))
                throw new InvalidOperationException(backend.DirectPtxLastError);
        }

        // Force the established path even when a developer has enabled the
        // process-wide softmax gate. The unpromoted experiment override must
        // never leak into the baseline cell.
        DirectPtxFeatureGate.SoftmaxExperimentOverride = false;
        DirectPtxFeatureGate.TestOverride = false;
        BaselineLaunch();
        backend.Synchronize();
        float baselineError = MaximumError(backend.DownloadBuffer(baselineOutput), expected);
        MeasureAndPrint(run, rows, columns, "AiDotNet CUDA eager", backend,
            BaselineLaunch, baselineError, temporaryBytes: 0);
        MeasureGraphAndPrint(run, rows, columns, "AiDotNet CUDA graph", backend,
            BaselineLaunch, baselineError, temporaryBytes: 0);

        DirectPtxFeatureGate.SoftmaxExperimentOverride = true;
        DirectPtxFeatureGate.TestOverride = true;
        if (!backend.PrewarmDirectPtxSoftmax(rows, columns))
            throw new InvalidOperationException(backend.DirectPtxLastError);
        DirectLaunch();
        backend.Synchronize();
        float directError = MaximumError(backend.DownloadBuffer(directOutput), expected);
        if (!backend.TryGetDirectPtxSoftmaxAudit(
            rows, columns, out DirectPtxKernelAudit audit))
            throw new InvalidOperationException("No audit for measured softmax PTX module.");
        MeasureAndPrint(run, rows, columns, "Direct PTX eager", backend,
            DirectLaunch, directError, temporaryBytes: 0, audit);
        MeasureGraphAndPrint(run, rows, columns, "Direct PTX graph", backend,
            DirectLaunch, directError, temporaryBytes: 0, audit);
        Console.WriteLine(
            $"    audit={audit.BlueprintId} sha={audit.PtxSha256} device={audit.DeviceFingerprint}");
    }

    private static void MeasureAndPrint(
        int run, int rows, int columns, string method,
        CudaBackend backend, Action launch, float error, long temporaryBytes,
        DirectPtxKernelAudit? audit = null)
    {
        Distribution device = MeasureDevice(backend, launch);
        Distribution e2e = MeasureEndToEnd(backend, launch);
        long allocation = MeasureAllocation(backend, launch);
        Print(run, rows, columns, method, device, e2e,
            allocation, temporaryBytes, error, audit);
    }

    private static void MeasureGraphAndPrint(
        int run, int rows, int columns, string method,
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
            Print(run, rows, columns, method, device, e2e,
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
                values[sample] =
                    backend.GetEventElapsedTime(start, end) * 1_000.0 / DeviceLaunches;
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
            "run rows  cols  method                    dev mean/med/p95/p99 us       " +
            "e2e mean/med/p95/p99 us       GFLOPS TFLOPS  GB/s allocB tmpB err      R/S/L/B");
        Console.WriteLine(new string('-', 184));
    }

    private static void PrintHostFingerprint()
    {
        Console.WriteLine(
            $"Host: OS={RuntimeInformation.OSDescription}; " +
            $"runtime={RuntimeInformation.FrameworkDescription}; " +
            $"arch={RuntimeInformation.ProcessArchitecture}; " +
            $"CUDA-driver={CudaNativeBindings.DriverVersion}; " +
            $"stopwatch-hz={Stopwatch.Frequency}");
    }

    private static void Print(
        int run, int rows, int columns, string method,
        Distribution device, Distribution e2e, long allocation,
        long temporaryBytes, float error, DirectPtxKernelAudit? audit)
    {
        double elements = (double)rows * columns;
        double gflops = elements * 5.0 / (device.Median * 1e-6) / 1e9;
        double tflops = gflops / 1_000.0;
        double bandwidth = elements * 8.0 / (device.Median * 1e-6) / 1e9;
        string resources = audit is null ? "-/-/-/-" :
            $"{audit.Function.RegistersPerThread}/{audit.Function.StaticSharedBytes}/" +
            $"{audit.Function.LocalBytesPerThread}/{audit.ActiveBlocksPerMultiprocessor}";
        Console.WriteLine(
            $"{run,3} {rows,5} {columns,5} {method,-25} " +
            $"{device.Mean,6:F2}/{device.Median,6:F2}/{device.P95,6:F2}/{device.P99,6:F2}   " +
            $"{e2e.Mean,6:F2}/{e2e.Median,6:F2}/{e2e.P95,6:F2}/{e2e.P99,6:F2}   " +
            $"{gflops,7:F1} {tflops,6:F3} {bandwidth,5:F1} {allocation,6} {temporaryBytes,4} " +
            $"{error,8:E1} {resources}");
    }

    private static float[] Oracle(float[] input, int rows, int columns)
    {
        var output = new float[input.Length];
        for (int row = 0; row < rows; row++)
        {
            int rowBase = row * columns;
            double maximum = double.NegativeInfinity;
            for (int column = 0; column < columns; column++)
                maximum = Math.Max(maximum, input[rowBase + column]);
            double sum = 0;
            for (int column = 0; column < columns; column++)
                sum += Math.Exp(input[rowBase + column] - maximum);
            for (int column = 0; column < columns; column++)
                output[rowBase + column] =
                    (float)(Math.Exp(input[rowBase + column] - maximum) / sum);
        }
        return output;
    }

    private static float[] BuildInput(int elements, int seed)
    {
        var input = new float[elements];
        for (int i = 0; i < input.Length; i++)
            input[i] = ((i * 17 + seed) % 257 - 128) / 32f;
        return input;
    }

    private static float MaximumError(float[] actual, float[] expected)
    {
        float maximum = 0;
        for (int i = 0; i < actual.Length; i++)
            maximum = MathF.Max(maximum, MathF.Abs(actual[i] - expected[i]));
        return maximum;
    }

    private static void RunCudnnCompetitors(int independentRuns)
    {
        ConfigureCudnnLibraryDirectory();
        if (!CuDnnContext.IsAvailable)
        {
            Console.WriteLine("cuDNN GPU competitor: INELIGIBLE (cuDNN is unavailable).");
            return;
        }
        try
        {
            Console.WriteLine($"cuDNN GPU competitor: version {CuDnnContext.Version}");
            for (int run = 1; run <= independentRuns; run++)
            foreach ((int rows, int columns) in Shapes)
                RunCudnnCell(run, rows, columns);
        }
        catch (Exception exception)
        {
            Console.WriteLine(
                $"cuDNN GPU competitor: INELIGIBLE ({exception.GetType().Name}: " +
                $"{exception.Message}).");
        }
    }

    private static void ConfigureCudnnLibraryDirectory()
    {
        if (!OperatingSystem.IsWindows()) return;
        try
        {
            var start = new ProcessStartInfo
            {
                FileName = Environment.GetEnvironmentVariable("PYTHON") ?? "python",
                UseShellExecute = false,
                RedirectStandardOutput = true,
                RedirectStandardError = true
            };
            start.ArgumentList.Add("-c");
            start.ArgumentList.Add(
                "import os,torch;print(os.path.join(os.path.dirname(torch.__file__),'lib'))");
            using Process process = Process.Start(start) ??
                throw new InvalidOperationException("Could not query the PyTorch library path.");
            string directory = process.StandardOutput.ReadToEnd().Trim();
            string error = process.StandardError.ReadToEnd().Trim();
            process.WaitForExit();
            if (process.ExitCode != 0 || !Directory.Exists(directory))
                throw new InvalidOperationException(
                    $"PyTorch library discovery failed ({process.ExitCode}): {error}");
            if (!SetDllDirectory(directory))
                throw new InvalidOperationException(
                    $"SetDllDirectory failed for '{directory}' (Win32 {Marshal.GetLastWin32Error()}).");
        }
        catch (Exception exception)
        {
            Console.WriteLine(
                $"cuDNN dependency discovery warning: {exception.GetType().Name}: " +
                exception.Message);
        }
    }

    [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    private static extern bool SetDllDirectory(string pathName);

    private static void RunCudnnCell(int run, int rows, int columns)
    {
        int elements = checked(rows * columns);
        float[] inputHost = BuildInput(elements, 20262100 + run * 100 + rows + columns);
        float[] expected = Oracle(inputHost, rows, columns);
        using var backend = new CudaBackend();
        backend.EnsureContextCurrent();
        using var context = CuDnnContext.ForCurrentContext();
        using var input = backend.AllocateBuffer(inputHost);
        using var output = backend.AllocateBuffer(elements);
        using var inputDescriptor = new CuDnnTensorDescriptor();
        using var outputDescriptor = new CuDnnTensorDescriptor();
        inputDescriptor.Set4D(CuDnnNative.CudnnDataType.Float, rows, columns, 1, 1);
        outputDescriptor.Set4D(CuDnnNative.CudnnDataType.Float, rows, columns, 1, 1);
        IntPtr stream = backend.DefaultStream.Handle;
        CuDnnContext.CheckStatus(
            CuDnnNative.cudnnSetStream(context.Handle, stream),
            "cudnnSetStream(softmax benchmark)");
        void Launch()
        {
            float alpha = 1f, beta = 0f;
            CuDnnContext.CheckStatus(
                CuDnnNative.cudnnSoftmaxForward(
                    context.Handle,
                    CuDnnNative.CudnnSoftmaxAlgorithm.Accurate,
                    CuDnnNative.CudnnSoftmaxMode.Channel,
                    ref alpha,
                    inputDescriptor.Handle, input.Handle,
                    ref beta,
                    outputDescriptor.Handle, output.Handle),
                "cudnnSoftmaxForward");
        }
        void Synchronize() => backend.Synchronize();

        for (int i = 0; i < Warmups; i++) Launch();
        Synchronize();
        float error = MaximumError(backend.DownloadBuffer(output), expected);
        Distribution device = MeasureCudnnDevice(stream, Launch);
        Distribution eagerE2e = MeasureCudnnEndToEnd(Launch, Synchronize);
        long allocation = MeasureCudnnAllocation(Launch, Synchronize);
        Print(run, rows, columns, "cuDNN accurate eager", device, eagerE2e,
            allocation, 0, error, audit: null);

        (IntPtr graph, IntPtr graphExec) = CaptureCudnnGraph(stream, Launch, 1);
        try
        {
            void GraphLaunch() => CheckCuda(
                CudaNativeBindings.cuGraphLaunch(graphExec, stream),
                "cuGraphLaunch(cuDNN softmax)");
            Distribution graphE2e = MeasureCudnnEndToEnd(GraphLaunch, Synchronize);
            long graphAllocation = MeasureCudnnAllocation(GraphLaunch, Synchronize);
            Print(run, rows, columns, "cuDNN accurate graph", device, graphE2e,
                graphAllocation, 0, error, audit: null);
        }
        finally
        {
            _ = CudaNativeBindings.cuGraphExecDestroy(graphExec);
            _ = CudaNativeBindings.cuGraphDestroy(graph);
        }
    }

    private static Distribution MeasureCudnnDevice(IntPtr stream, Action launch)
    {
        (IntPtr graph, IntPtr graphExec) = CaptureCudnnGraph(
            stream, launch, DeviceLaunches);
        CheckCuda(CudaNativeBindings.cuEventCreate(out IntPtr start, 0),
            "cuEventCreate(start)");
        CheckCuda(CudaNativeBindings.cuEventCreate(out IntPtr end, 0),
            "cuEventCreate(end)");
        try
        {
            for (int i = 0; i < Warmups; i++)
                CheckCuda(CudaNativeBindings.cuGraphLaunch(graphExec, stream),
                    "cuGraphLaunch(cuDNN warmup)");
            CheckCuda(CudaNativeBindings.cuStreamSynchronize(stream),
                "cuStreamSynchronize(cuDNN warmup)");
            var values = new double[Samples];
            for (int sample = 0; sample < values.Length; sample++)
            {
                CheckCuda(CudaNativeBindings.cuEventRecord(start, stream),
                    "cuEventRecord(start)");
                CheckCuda(CudaNativeBindings.cuGraphLaunch(graphExec, stream),
                    "cuGraphLaunch(cuDNN device sample)");
                CheckCuda(CudaNativeBindings.cuEventRecord(end, stream),
                    "cuEventRecord(end)");
                CheckCuda(CudaNativeBindings.cuEventSynchronize(end),
                    "cuEventSynchronize(end)");
                CheckCuda(CudaNativeBindings.cuEventElapsedTime(
                    out float milliseconds, start, end), "cuEventElapsedTime");
                values[sample] = milliseconds * 1_000.0 / DeviceLaunches;
            }
            return Summarize(values);
        }
        finally
        {
            _ = CudaNativeBindings.cuEventDestroy(end);
            _ = CudaNativeBindings.cuEventDestroy(start);
            _ = CudaNativeBindings.cuGraphExecDestroy(graphExec);
            _ = CudaNativeBindings.cuGraphDestroy(graph);
        }
    }

    private static Distribution MeasureCudnnEndToEnd(Action launch, Action synchronize)
    {
        for (int i = 0; i < Warmups; i++) launch();
        synchronize();
        var values = new double[Samples];
        double scale = 1_000_000.0 / Stopwatch.Frequency;
        for (int sample = 0; sample < values.Length; sample++)
        {
            long start = Stopwatch.GetTimestamp();
            launch();
            synchronize();
            values[sample] = (Stopwatch.GetTimestamp() - start) * scale;
        }
        return Summarize(values);
    }

    private static long MeasureCudnnAllocation(Action launch, Action synchronize)
    {
        for (int i = 0; i < 8; i++) launch();
        synchronize();
        long before = GC.GetAllocatedBytesForCurrentThread();
        for (int i = 0; i < Samples; i++) launch();
        long bytes = (GC.GetAllocatedBytesForCurrentThread() - before) / Samples;
        synchronize();
        return bytes;
    }

    private static (IntPtr Graph, IntPtr GraphExec) CaptureCudnnGraph(
        IntPtr stream, Action launch, int launches)
    {
        CheckCuda(CudaNativeBindings.cuStreamBeginCapture(
            stream, CudaNativeBindings.CU_STREAM_CAPTURE_MODE_THREAD_LOCAL),
            "cuStreamBeginCapture(cuDNN softmax)");
        for (int i = 0; i < launches; i++) launch();
        CheckCuda(CudaNativeBindings.cuStreamEndCapture(stream, out IntPtr graph),
            "cuStreamEndCapture(cuDNN softmax)");
        try
        {
            CheckCuda(CudaNativeBindings.cuGraphInstantiate(
                out IntPtr graphExec, graph, 0),
                "cuGraphInstantiate(cuDNN softmax)");
            return (graph, graphExec);
        }
        catch
        {
            _ = CudaNativeBindings.cuGraphDestroy(graph);
            throw;
        }
    }

    private static void CheckCuda(CudaResult result, string operation)
    {
        if (result != CudaResult.Success)
            throw new InvalidOperationException($"{operation} failed: {result}.");
    }

    private static void RunPyTorchCompetitors(int independentRuns)
    {
        string script = Path.Combine(AppContext.BaseDirectory, "BaselineRunners", "py",
            "run_direct_ptx_softmax_competitors.py");
        if (!File.Exists(script))
            script = Path.Combine(AppContext.BaseDirectory, "run_direct_ptx_softmax_competitors.py");
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
            throw new InvalidOperationException("Could not start the PyTorch softmax runner.");
        process.WaitForExit();
        if (process.ExitCode != 0)
            throw new InvalidOperationException(
                $"PyTorch softmax runner exited with code {process.ExitCode}.");
    }
}
