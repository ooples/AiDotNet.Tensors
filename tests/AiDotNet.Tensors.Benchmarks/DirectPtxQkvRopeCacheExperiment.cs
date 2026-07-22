using System.Diagnostics;
using System.Text.Json;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;
using AiDotNet.Tensors.Engines.Gpu;

namespace AiDotNet.Tensors.Benchmarks;

/// <summary>Issue #835 decode QKV projection/RoPE/KV-cache championship matrix.</summary>
internal static class DirectPtxQkvRopeCacheExperiment
{
    private const int Dimension = PtxFusedQkvRopeCacheD64Kernel.HeadDimension;
    private const int Warmups = 30;
    private const int Samples = 101;
    private const int LaunchesPerDeviceSample = 10;

    private readonly record struct Shape(string Name, int Heads, int Capacity, int Position);
    private readonly record struct Distribution(double Mean, double Median, double P95, double P99);
    private readonly record struct CellEvidence(
        int Run,
        Shape Shape,
        Distribution GraphDevice,
        Distribution GraphEndToEnd,
        Distribution DirectDevice,
        Distribution CurrentDevice,
        Distribution DirectEndToEnd,
        Distribution CurrentEndToEnd,
        long GraphBytes,
        long DirectBytes,
        long CurrentBytes,
        double DirectError,
        double CurrentError,
        DirectPtxKernelAudit Audit);
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
        double Tflops,
        long PeakDeviceBytes,
        double MaxError);

    private static readonly Shape[] Shapes =
    [
        new("decode-h4", 4, 16, 0),
        new("decode-h8", 8, 64, 17),
        new("decode-h16", 16, 128, 127)
    ];

    internal static void Run(int independentRuns = 3, bool includeExternal = true)
    {
        if (independentRuns <= 0) throw new ArgumentOutOfRangeException(nameof(independentRuns));
        Console.WriteLine(
            $"Direct-PTX fused QKV + bias + RoPE + KV-cache: {independentRuns} independent run(s), " +
            $"{Warmups} warmups + {Samples} CUDA-event and synchronized E2E samples/cell");
        Console.WriteLine(
            $"Device samples average {LaunchesPerDeviceSample} resident launches. " +
            "All tensors and competitor workspaces are allocated before timing.");
        var evidence = new List<CellEvidence>(independentRuns * Shapes.Length);
        for (int run = 1; run <= independentRuns; run++)
        {
            using var backend = new CudaBackend();
            if (!backend.IsDirectPtxQkvRopeCacheEnabled)
                throw new InvalidOperationException(
                    "Set AIDOTNET_DIRECT_PTX_QKV_ROPE_CACHE=1 on an Ampere GPU.");
            if (run == 1)
                Console.WriteLine($"GPU: {backend.DeviceName}; FP32 resident decode-token comparison");
            foreach (Shape shape in Shapes) evidence.Add(RunCell(backend, run, shape));
        }

        IReadOnlyList<PythonRecord> python = includeExternal
            ? RunPython(independentRuns)
            : Array.Empty<PythonRecord>();
        PrintGrouped(evidence, python, independentRuns);
        PrintReleaseGate(evidence, python, independentRuns, includeExternal);
    }

    private static CellEvidence RunCell(CudaBackend backend, int run, Shape shape)
    {
        int model = shape.Heads * Dimension;
        int projection = 3 * model;
        int cacheElements = shape.Capacity * model;
        var random = new Random(20261300 + run * 10_000 + model + shape.Position);
        float[] inputHost = Values(random, model, 0.125f);
        float[] weightsHost = Values(random, projection * model, 0.0625f);
        float[] biasHost = Values(random, projection, 0.0625f);
        (float[] cosineHost, float[] sineHost) = RopeTables(shape.Capacity);
        float[] initialKeyCache = Values(random, cacheElements, 0.125f);
        float[] initialValueCache = Values(random, cacheElements, 0.125f);
        (double[] expectedQ, double[] expectedK, double[] expectedV) = Oracle(
            inputHost, weightsHost, biasHost, cosineHost, sineHost,
            initialKeyCache, initialValueCache, shape);

        using var input = backend.AllocateBuffer(inputHost);
        using var weights = backend.AllocateBuffer(weightsHost);
        using var bias = backend.AllocateBuffer(biasHost);
        using var cosine = backend.AllocateBuffer(cosineHost);
        using var sine = backend.AllocateBuffer(sineHost);
        using var directQuery = backend.AllocateBuffer(model);
        using var directKeyCache = backend.AllocateBuffer(initialKeyCache);
        using var directValueCache = backend.AllocateBuffer(initialValueCache);
        using var currentQuery = backend.AllocateBuffer(model);
        using var currentKeyCache = backend.AllocateBuffer(initialKeyCache);
        using var currentValueCache = backend.AllocateBuffer(initialValueCache);
        using var projected = backend.AllocateBuffer(projection);
        using var biased = backend.AllocateBuffer(projection);
        using var rotatedQueryKey = backend.AllocateBuffer(2 * model);
        long directDispatchBefore = backend.DirectPtxQkvRopeCacheDispatchCount;

        void DirectLaunch()
        {
            backend.QkvProjectionRoPECacheD64(
                input, weights, bias, cosine, sine,
                directQuery, directKeyCache, directValueCache,
                shape.Heads, shape.Capacity, shape.Position);
        }

        void CurrentLaunch()
        {
            backend.MatMulTransposed(input, weights, projected, 1, projection, model);
            backend.BiasAdd(projected, bias, biased, 1, projection);
            backend.RopeInterleaved(
                biased, cosine, sine, rotatedQueryKey,
                rows: 2 * shape.Heads, headDim: Dimension,
                seqLen: 1, startPosition: shape.Position);
            backend.Copy(rotatedQueryKey, 0, currentQuery, 0, model);
            backend.Copy(rotatedQueryKey, model, currentKeyCache, shape.Position * model, model);
            backend.Copy(biased, 2 * model, currentValueCache, shape.Position * model, model);
        }

        DirectLaunch();
        CurrentLaunch();
        backend.Synchronize();
        double directError = MaximumError(
            backend, directQuery, directKeyCache, directValueCache,
            expectedQ, expectedK, expectedV);
        double currentError = MaximumError(
            backend, currentQuery, currentKeyCache, currentValueCache,
            expectedQ, expectedK, expectedV);
        IntPtr graph = backend.CaptureGraph(DirectLaunch);
        if (graph == IntPtr.Zero)
            throw new InvalidOperationException(
                $"Could not capture the public direct-PTX route for {shape.Name}.");
        try
        {
            void GraphLaunch() => backend.EnqueueCapturedGraph(graph);

            Distribution graphDevice = MeasureDevice(backend, GraphLaunch);
            Distribution directDevice = MeasureDevice(backend, DirectLaunch);
            Distribution currentDevice = MeasureDevice(backend, CurrentLaunch);
            Distribution graphEndToEnd = MeasureEndToEnd(backend, GraphLaunch);
            Distribution directEndToEnd = MeasureEndToEnd(backend, DirectLaunch);
            Distribution currentEndToEnd = MeasureEndToEnd(backend, CurrentLaunch);
            long graphBytes = MeasureAllocation(backend, GraphLaunch);
            long directBytes = MeasureAllocation(backend, DirectLaunch);
            long currentBytes = MeasureAllocation(backend, CurrentLaunch);
            const long expectedDirectDispatches =
                2L + Warmups + Samples * LaunchesPerDeviceSample +
                Warmups + Samples + 8L + Samples;
            long actualDirectDispatches =
                backend.DirectPtxQkvRopeCacheDispatchCount - directDispatchBefore;
            if (actualDirectDispatches != expectedDirectDispatches ||
                backend.DirectPtxLastError is not null)
                throw new InvalidOperationException(
                    $"Direct PTX routing was not exclusive for {shape.Name}: " +
                    $"expected {expectedDirectDispatches} dispatches, observed " +
                    $"{actualDirectDispatches}; last error={backend.DirectPtxLastError ?? "none"}.");
            if (!backend.TryGetDirectPtxQkvRopeCacheAudit(
                shape.Heads, shape.Capacity, shape.Position, out DirectPtxKernelAudit audit))
                throw new InvalidOperationException("No audit for measured QKV/RoPE/cache module.");

            return new CellEvidence(
                run, shape, graphDevice, graphEndToEnd,
                directDevice, currentDevice, directEndToEnd, currentEndToEnd,
                graphBytes, directBytes, currentBytes, directError, currentError, audit);
        }
        finally
        {
            backend.DestroyCapturedGraph(graph);
        }
    }

    private static void PrintGrouped(
        IReadOnlyList<CellEvidence> evidence,
        IReadOnlyList<PythonRecord> python,
        int independentRuns)
    {
        PrintHeader();
        for (int run = 1; run <= independentRuns; run++)
        foreach (Shape shape in Shapes)
        {
            CellEvidence cell = evidence.Single(candidate =>
                candidate.Run == run && candidate.Shape == shape);
            Print(run, shape, "Direct PTX CUDA graph", cell.GraphDevice,
                cell.GraphEndToEnd, Tflops(shape, cell.GraphDevice.Median),
                cell.GraphBytes, 0, cell.DirectError,
                cell.Audit.Function.RegistersPerThread,
                cell.Audit.Function.StaticSharedBytes,
                cell.Audit.Function.LocalBytesPerThread,
                cell.Audit.ActiveBlocksPerMultiprocessor);
            Print(run, shape, "Direct PTX fused", cell.DirectDevice, cell.DirectEndToEnd,
                Tflops(shape, cell.DirectDevice.Median), cell.DirectBytes, 0,
                cell.DirectError, cell.Audit.Function.RegistersPerThread,
                cell.Audit.Function.StaticSharedBytes,
                cell.Audit.Function.LocalBytesPerThread,
                cell.Audit.ActiveBlocksPerMultiprocessor);
            long currentTemporaryBytes =
                8L * shape.Heads * Dimension * sizeof(float);
            Print(run, shape, "AiDotNet cuBLAS+NVRTC", cell.CurrentDevice,
                cell.CurrentEndToEnd, Tflops(shape, cell.CurrentDevice.Median),
                cell.CurrentBytes, currentTemporaryBytes, cell.CurrentError,
                -1, -1, -1, -1);
            foreach (PythonRecord record in python.Where(record =>
                record.Status == "ok" && record.Run == run &&
                record.Shape == shape.Name).OrderBy(record => record.DeviceMedianUs))
                PrintPython(record);
        }
    }

    private static Distribution MeasureDevice(CudaBackend backend, Action action)
    {
        for (int i = 0; i < Warmups; i++) action();
        backend.Synchronize();
        var samples = new double[Samples];
        using IGpuEvent start = backend.CreateEvent(enableTiming: true);
        using IGpuEvent end = backend.CreateEvent(enableTiming: true);
        for (int i = 0; i < samples.Length; i++)
        {
            backend.RecordEvent(start, backend.DefaultStream);
            for (int launch = 0; launch < LaunchesPerDeviceSample; launch++) action();
            backend.RecordEvent(end, backend.DefaultStream);
            end.Synchronize();
            samples[i] = backend.GetEventElapsedTime(start, end) * 1_000.0 /
                LaunchesPerDeviceSample;
        }
        return Summarize(samples);
    }

    private static Distribution MeasureEndToEnd(CudaBackend backend, Action action)
    {
        for (int i = 0; i < Warmups; i++) action();
        backend.Synchronize();
        var samples = new double[Samples];
        double tickToMicroseconds = 1_000_000.0 / Stopwatch.Frequency;
        for (int i = 0; i < samples.Length; i++)
        {
            long start = Stopwatch.GetTimestamp();
            action();
            backend.Synchronize();
            samples[i] = (Stopwatch.GetTimestamp() - start) * tickToMicroseconds;
        }
        return Summarize(samples);
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

    private static IReadOnlyList<PythonRecord> RunPython(int runs)
    {
        string script = Path.Combine(
            AppContext.BaseDirectory, "BaselineRunners", "py",
            "run_direct_ptx_qkv_rope_cache_competitors.py");
        if (!File.Exists(script))
            script = Path.Combine(AppContext.BaseDirectory,
                "run_direct_ptx_qkv_rope_cache_competitors.py");
        if (!File.Exists(script))
            throw new FileNotFoundException("The issue #835 PyTorch harness was not copied to output.", script);
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
                ReadString(root, "status"),
                ReadInt(root, "run"),
                ReadString(root, "shape"),
                ReadString(root, "method"),
                ReadDouble(root, "device_mean_us"),
                ReadDouble(root, "device_median_us"),
                ReadDouble(root, "device_p95_us"),
                ReadDouble(root, "device_p99_us"),
                ReadDouble(root, "e2e_mean_us"),
                ReadDouble(root, "e2e_median_us"),
                ReadDouble(root, "e2e_p95_us"),
                ReadDouble(root, "e2e_p99_us"),
                ReadDouble(root, "tflops"),
                ReadLong(root, "peak_device_bytes"),
                ReadDouble(root, "max_error")));
        }
        string error = process.StandardError.ReadToEnd();
        process.WaitForExit();
        if (process.ExitCode != 0)
            throw new InvalidOperationException(
                $"PyTorch CUDA baseline failed with exit {process.ExitCode}: {error}");
        return records;

        static string ReadString(JsonElement element, string name) =>
            element.TryGetProperty(name, out JsonElement value) ? value.GetString() ?? "" : "";
        static int ReadInt(JsonElement element, string name) =>
            element.TryGetProperty(name, out JsonElement value) ? value.GetInt32() : 0;
        static long ReadLong(JsonElement element, string name) =>
            element.TryGetProperty(name, out JsonElement value) ? value.GetInt64() : 0;
        static double ReadDouble(JsonElement element, string name) =>
            element.TryGetProperty(name, out JsonElement value) ? value.GetDouble() : 0;
    }

    private static void PrintHeader()
    {
        Console.WriteLine(
            $"{"Run",3} {"Shape",-11} {"Method",-23} {"dev mean",9} {"dev med",9} " +
            $"{"dev p95",9} {"dev p99",9} {"E2E mean",9} {"E2E med",9} " +
            $"{"E2E p95",9} {"E2E p99",9} {"TFLOPS",8} {"managed B",9} " +
            $"{"temp/peak B",11} {"max err",10} " +
            $"{"regs",5} {"shared",7} {"local",5} {"occ",4}");
        Console.WriteLine(new string('-', 203));
    }

    private static void Print(
        int run,
        Shape shape,
        string method,
        Distribution device,
        Distribution endToEnd,
        double tflops,
        long bytes,
        long temporaryBytes,
        double error,
        int registers,
        int shared,
        int local,
        int occupancy)
    {
        Console.WriteLine(
            $"{run,3} {shape.Name,-11} {method,-23} " +
            $"{device.Mean,9:F2} {device.Median,9:F2} {device.P95,9:F2} {device.P99,9:F2} " +
            $"{endToEnd.Mean,9:F2} {endToEnd.Median,9:F2} {endToEnd.P95,9:F2} {endToEnd.P99,9:F2} " +
            $"{tflops,8:F3} {ManagedBytes(bytes),9} {temporaryBytes,11} " +
            $"{error,10:G4} {Dash(registers),5} {Dash(shared),7} {Dash(local),5} {Dash(occupancy),4}");
        Console.WriteLine("qkv_evidence_json=" + JsonSerializer.Serialize(new
        {
            status = "ok",
            run,
            shape = shape.Name,
            method,
            device_mean_us = device.Mean,
            device_median_us = device.Median,
            device_p95_us = device.P95,
            device_p99_us = device.P99,
            e2e_mean_us = endToEnd.Mean,
            e2e_median_us = endToEnd.Median,
            e2e_p95_us = endToEnd.P95,
            e2e_p99_us = endToEnd.P99,
            tflops,
            managed_bytes = bytes,
            temporary_device_bytes = temporaryBytes,
            max_error = error,
            registers_per_thread = registers,
            static_shared_bytes = shared,
            local_bytes_per_thread = local,
            active_blocks_per_sm = occupancy
        }));
    }

    private static string ManagedBytes(long value) => value < 0 ? "n/a" : value.ToString();

    private static void PrintPython(PythonRecord record)
    {
        var device = new Distribution(
            record.DeviceMeanUs, record.DeviceMedianUs, record.DeviceP95Us, record.DeviceP99Us);
        var e2e = new Distribution(
            record.EndToEndMeanUs, record.EndToEndMedianUs,
            record.EndToEndP95Us, record.EndToEndP99Us);
        Shape shape = Shapes.Single(candidate => candidate.Name == record.Shape);
        Print(record.Run, shape, record.Method, device, e2e, record.Tflops,
            -1, record.PeakDeviceBytes, record.MaxError, -1, -1, -1, -1);
    }

    private static string Dash(int value) => value < 0 ? "-" : value.ToString();

    private static void PrintReleaseGate(
        IReadOnlyList<CellEvidence> evidence,
        IReadOnlyList<PythonRecord> python,
        int expectedRuns,
        bool includeExternal)
    {
        bool complete = evidence.Count == expectedRuns * Shapes.Length &&
            (!includeExternal ||
                python.Count(record => record.Status == "ok") == expectedRuns * Shapes.Length * 2);
        bool correct = evidence.All(cell =>
            cell.DirectError <= 2e-5f && cell.CurrentError <= 2e-5f) &&
            python.Where(record => record.Status == "ok").All(record => record.MaxError <= 2e-5);
        bool resources = evidence.All(cell =>
            cell.Audit.Function.RegistersPerThread <= 48 &&
            cell.Audit.Function.StaticSharedBytes == 0 &&
            cell.Audit.Function.LocalBytesPerThread == 0 &&
            cell.Audit.ActiveBlocksPerMultiprocessor >= 8);
        bool allocation = evidence.All(cell => cell.GraphBytes == 0 && cell.DirectBytes == 0);
        bool currentChampion = evidence.All(cell =>
            cell.CurrentDevice.Median / cell.GraphDevice.Median >= 1.10 &&
            cell.GraphDevice.P95 <= cell.CurrentDevice.P95 * 1.10 &&
            cell.CurrentEndToEnd.Median / cell.GraphEndToEnd.Median >= 1.10);
        bool pythonChampion = !includeExternal || python.Where(record => record.Status == "ok").All(peer =>
        {
            CellEvidence cell = evidence.Single(candidate =>
                candidate.Run == peer.Run && candidate.Shape.Name == peer.Shape);
            return peer.DeviceMedianUs / cell.GraphDevice.Median >= 1.10 &&
                cell.GraphDevice.P95 <= peer.DeviceP95Us * 1.10 &&
                peer.EndToEndMedianUs / cell.GraphEndToEnd.Median >= 1.10;
        });
        Console.WriteLine(new string('-', 203));
        foreach (CellEvidence cell in evidence)
        {
            PythonRecord[] peers = python.Where(record =>
                record.Status == "ok" && record.Run == cell.Run &&
                record.Shape == cell.Shape.Name).ToArray();
            double pythonDevice = peers.Length == 0 ? 0 :
                peers.Min(peer => peer.DeviceMedianUs / cell.GraphDevice.Median);
            double pythonEndToEnd = peers.Length == 0 ? 0 :
                peers.Min(peer => peer.EndToEndMedianUs / cell.GraphEndToEnd.Median);
            Console.WriteLine(
                $"gate run {cell.Run} {cell.Shape.Name}: " +
                $"vs AiDotNet {cell.CurrentDevice.Median / cell.GraphDevice.Median:F2}x device, " +
                $"{cell.CurrentEndToEnd.Median / cell.GraphEndToEnd.Median:F2}x E2E; " +
                (peers.Length == 0
                    ? "PyTorch missing"
                    : $"vs strongest PyTorch {pythonDevice:F2}x device, " +
                      $"{pythonEndToEnd:F2}x E2E"));
        }
        bool pass = complete && correct && resources && allocation &&
            currentChampion && pythonChampion;
        Console.WriteLine(pass
            ? includeExternal
                ? "PASS: complete requested-run matrix, <=2e-5 error, bounded device P95, zero JIT local/shared/temp/managed allocation, and >=1.10x over every NVIDIA competitor."
                : "PASS: complete AiDotNet matrix, <=2e-5 error, bounded device P95, zero JIT local/shared/temp/managed allocation, and >=1.10x over the current AiDotNet CUDA path."
            : $"FAIL: complete={complete}, correct={correct}, resources={resources}, " +
              $"allocation={allocation}, AiDotNetChampion={currentChampion}, PyTorchChampion={pythonChampion}.");
        if (!pass) Environment.ExitCode = 2;
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

    private static double Tflops(Shape shape, double microseconds)
    {
        double model = shape.Heads * Dimension;
        double flops = 6.0 * model * model;
        return flops / (microseconds * 1e-6) / 1e12;
    }

    private static double MaximumError(
        CudaBackend backend,
        IGpuBuffer query,
        IGpuBuffer keyCache,
        IGpuBuffer valueCache,
        double[] expectedQuery,
        double[] expectedKey,
        double[] expectedValue)
    {
        double error = 0;
        Update(backend.DownloadBuffer(query), expectedQuery);
        Update(backend.DownloadBuffer(keyCache), expectedKey);
        Update(backend.DownloadBuffer(valueCache), expectedValue);
        return error;

        void Update(float[] actual, double[] expected)
        {
            for (int i = 0; i < expected.Length; i++)
                error = Math.Max(error, Math.Abs(actual[i] - expected[i]));
        }
    }

    private static float[] Values(Random random, int count, float magnitude) =>
        Enumerable.Range(0, count)
            .Select(_ => (random.NextSingle() * 2f - 1f) * magnitude)
            .ToArray();

    private static (float[] Cosine, float[] Sine) RopeTables(int capacity)
    {
        var cosine = new float[capacity * (Dimension / 2)];
        var sine = new float[cosine.Length];
        for (int position = 0; position < capacity; position++)
        for (int pair = 0; pair < Dimension / 2; pair++)
        {
            float angle = position * MathF.Pow(10_000f, -2f * pair / Dimension);
            cosine[position * (Dimension / 2) + pair] = MathF.Cos(angle);
            sine[position * (Dimension / 2) + pair] = MathF.Sin(angle);
        }
        return (cosine, sine);
    }

    private static (double[] Query, double[] KeyCache, double[] ValueCache) Oracle(
        float[] input,
        float[] weights,
        float[] bias,
        float[] cosine,
        float[] sine,
        float[] initialKeyCache,
        float[] initialValueCache,
        Shape shape)
    {
        int model = shape.Heads * Dimension;
        var projection = new double[3 * model];
        for (int output = 0; output < projection.Length; output++)
        {
            double sum = bias[output];
            int weightBase = output * model;
            for (int inner = 0; inner < model; inner++)
                sum += (double)input[inner] * weights[weightBase + inner];
            projection[output] = sum;
        }
        var query = new double[model];
        var keyCache = Array.ConvertAll(initialKeyCache, value => (double)value);
        var valueCache = Array.ConvertAll(initialValueCache, value => (double)value);
        int cacheBase = shape.Position * model;
        int ropeBase = shape.Position * (Dimension / 2);
        for (int head = 0; head < shape.Heads; head++)
        for (int pair = 0; pair < Dimension / 2; pair++)
        {
            int feature = pair * 2;
            int index = head * Dimension + feature;
            double c = cosine[ropeBase + pair];
            double s = sine[ropeBase + pair];
            double qe = projection[index], qo = projection[index + 1];
            double ke = projection[model + index], ko = projection[model + index + 1];
            query[index] = qe * c - qo * s;
            query[index + 1] = qe * s + qo * c;
            keyCache[cacheBase + index] = ke * c - ko * s;
            keyCache[cacheBase + index + 1] = ke * s + ko * c;
            valueCache[cacheBase + index] = projection[2 * model + index];
            valueCache[cacheBase + index + 1] = projection[2 * model + index + 1];
        }
        return (query, keyCache, valueCache);
    }
}
