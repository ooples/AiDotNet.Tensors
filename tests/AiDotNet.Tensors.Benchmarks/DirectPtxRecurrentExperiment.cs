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
    private const int OracleOperationsPerGraph = 64;
    private const double SemanticTolerance = 2e-5;
    private const double RequiredGain = 1.10;
    private const int Batch = PtxFusedRgLruScan128x256Kernel.Batch;
    private const int Sequence = PtxFusedRgLruScan128x256Kernel.SequenceLength;
    private const int Dimension = PtxFusedRgLruScan128x256Kernel.RecurrentDimension;
    private const double EstimatedFlops = Batch * Sequence * Dimension * 12.0 + Dimension * 4.0;

    private readonly record struct Distribution(double Mean, double Median, double P95, double P99);
    private sealed record ExternalRecord(
        string Status, int Run, string Method,
        double MeanUs, double MedianUs, double P95Us, double P99Us,
        long PeakDeviceBytes, long TemporaryDeviceBytes, double MaxError);
    private sealed record PairedTailEvidence(
        Distribution Direct,
        Distribution Incumbent,
        Distribution DirectOverIncumbent);
    private sealed record OracleEvidence(
        StableTimer.PairResult Timing,
        PairedTailEvidence Tail);
    private sealed record NativeEvidence(
        int Run,
        Distribution DirectGraph,
        Distribution Incumbent,
        double DirectError,
        double IncumbentError,
        OracleEvidence Oracle);

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
            var nativeEvidence = new List<NativeEvidence>(independentRuns);
            for (int run = 1; run <= independentRuns; run++)
                nativeEvidence.Add(RunAiDotNet(run));
            IReadOnlyList<ExternalRecord> externalEvidence = Array.Empty<ExternalRecord>();
            if (includeExternal)
            {
                externalEvidence = RunPython(independentRuns);
                foreach (ExternalRecord record in externalEvidence) Print(record);
            }
            PrintChampionshipOracle(
                nativeEvidence, externalEvidence, includeExternal, independentRuns);
        }
        finally
        {
            DirectPtxFeatureGate.TestOverride = previous;
            GpuBenchmarkEnvironment.RequireNoForeignCompute(
                "direct-ptx-rglru-end", afterSuite: true);
        }
    }

    private static NativeEvidence RunAiDotNet(int run)
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
        if (!IsFinite(directError) || !IsFinite(currentError) ||
            directError > SemanticTolerance || currentError > SemanticTolerance)
            throw new InvalidOperationException(
                $"RG-LRU correctness gate failed: direct={directError:E3}, " +
                $"incumbent={currentError:E3}, tolerance={SemanticTolerance:E1}.");
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
            Distribution directGraphTiming = MeasureDevice(backend, GraphLaunch);
            Print(run, "Direct PTX CUDA graph", directGraphTiming,
                MeasureAllocation(backend, GraphLaunch), 0, directError, audit);
            Print(run, "Direct PTX fused", MeasureDevice(backend, DirectLaunch),
                MeasureAllocation(backend, DirectLaunch), 0, directError, audit);
            Distribution incumbentTiming = MeasureDevice(backend, CurrentLaunch);
            Print(run, "AiDotNet current NVRTC", incumbentTiming,
                MeasureAllocation(backend, CurrentLaunch), 0, currentError, null);
            OracleEvidence oracleEvidence = RunPairedOracle(
                run, backend, GraphLaunch, DirectLaunch, CurrentLaunch,
                directError, currentError, audit);
            return new NativeEvidence(
                run, directGraphTiming, incumbentTiming,
                directError, currentError, oracleEvidence);
        }
        finally
        {
            backend.DestroyCapturedGraph(graph);
        }
    }

    private static OracleEvidence RunPairedOracle(
        int run,
        CudaBackend backend,
        Action directTailLaunch,
        Action directLaunch,
        Action incumbentLaunch,
        double directError,
        double incumbentError,
        DirectPtxKernelAudit audit)
    {
        PairedTailEvidence tail = MeasurePairedDeviceDistributions(
            backend, directTailLaunch, incumbentLaunch);
        IntPtr directGraph = IntPtr.Zero;
        IntPtr incumbentGraph = IntPtr.Zero;
        string lane;
        StableTimer.PairResult timing;
        long directDispatchesBefore = backend.DirectPtxRgLruDispatchCount;
        try
        {
            directGraph = CaptureRepeatedGraph(
                backend, directLaunch, OracleOperationsPerGraph);
            incumbentGraph = CaptureRepeatedGraph(
                backend, incumbentLaunch, OracleOperationsPerGraph);
            if (directGraph != IntPtr.Zero && incumbentGraph != IntPtr.Zero)
            {
                lane = "repeated-cuda-graph-host-paired";
                timing = StableTimer.MeasureCalibratedHostPair(
                    () => backend.EnqueueCapturedGraph(directGraph), backend.Synchronize,
                    () => backend.EnqueueCapturedGraph(incumbentGraph), backend.Synchronize,
                    operationsPerLaunchA: OracleOperationsPerGraph,
                    operationsPerLaunchB: OracleOperationsPerGraph,
                    targetBatchMilliseconds: 50.0);
            }
            else
            {
                lane = "public-launch-host-paired-capture-fallback";
                timing = StableTimer.MeasureCalibratedHostPair(
                    directLaunch, backend.Synchronize,
                    incumbentLaunch, backend.Synchronize,
                    targetBatchMilliseconds: 50.0);
            }
        }
        finally
        {
            if (directGraph != IntPtr.Zero) backend.DestroyCapturedGraph(directGraph);
            if (incumbentGraph != IntPtr.Zero) backend.DestroyCapturedGraph(incumbentGraph);
        }

        if (backend.DirectPtxRgLruDispatchCount - directDispatchesBefore <
            OracleOperationsPerGraph)
            throw new InvalidOperationException(
                "The paired RG-LRU candidate capture did not enter direct PTX.");

        bool measurable = timing.Stable;
        bool tailPassed = tail.Direct.P95 <= tail.Incumbent.P95 * RequiredGain;
        double incumbentOverDirect = measurable ? 1.0 / timing.Ratio : double.NaN;
        string measurementStatus = measurable ? "stable" : "unstable";
        string verdict = !measurable
            ? "not-measurable"
            : incumbentOverDirect >= RequiredGain
                ? "win"
                : incumbentOverDirect <= 1.0 / RequiredGain
                    ? "loss"
                    : "tie";

        Console.WriteLine(
            $"oracle | {run} | rglru-b1-s128-d256 | lane={lane} | " +
            $"direct={timing.A.Describe()} | incumbent={timing.B.Describe()} | " +
            $"incumbent/direct={(measurable ? incumbentOverDirect.ToString("0.00") + "x" : "-")} | " +
            $"paired-tail-p95={tail.DirectOverIncumbent.P95:0.00}x | " +
            $"correctness=passed (direct={directError:E3}, incumbent={incumbentError:E3}, " +
            $"tolerance={SemanticTolerance:E1}) | {verdict}");
        Console.WriteLine("rglru_oracle_json=" + JsonSerializer.Serialize(new
        {
            kind = "direct-ptx-rglru-oracle",
            run,
            operation = "rglru-b1-s128-d256",
            lane,
            logical_operations_per_graph = lane.StartsWith(
                "repeated-cuda-graph", StringComparison.Ordinal)
                    ? OracleOperationsPerGraph
                    : 1,
            direct_median_us = timing.A.Stable ? timing.A.Microseconds : (double?)null,
            incumbent_median_us = timing.B.Stable ? timing.B.Microseconds : (double?)null,
            incumbent_over_direct = measurable ? incumbentOverDirect : (double?)null,
            direct_relative_spread = FiniteOrNull(timing.A.RelativeSpread),
            incumbent_relative_spread = FiniteOrNull(timing.B.RelativeSpread),
            paired_ratio_relative_spread = FiniteOrNull(timing.RelativeSpread),
            direct_tail_p95_us = tail.Direct.P95,
            incumbent_tail_p95_us = tail.Incumbent.P95,
            direct_over_incumbent_tail_p95 = tail.DirectOverIncumbent.P95,
            tail_passed = tailPassed,
            attempts = timing.Samples,
            required_gain = RequiredGain,
            direct_semantic_error = directError,
            incumbent_semantic_error = incumbentError,
            semantic_tolerance = SemanticTolerance,
            correctness_passed = true,
            measurement_status = measurementStatus,
            verdict,
            registers_per_thread = audit.Function.RegistersPerThread,
            static_shared_bytes = audit.Function.StaticSharedBytes,
            local_bytes_per_thread = audit.Function.LocalBytesPerThread,
            active_blocks_per_sm = audit.ActiveBlocksPerMultiprocessor,
            diagnostic_only = false,
            promotion = measurable && tailPassed
        }));
        return new OracleEvidence(timing, tail);
    }

    private static void PrintChampionshipOracle(
        IReadOnlyList<NativeEvidence> native,
        IReadOnlyList<ExternalRecord> external,
        bool includeExternal,
        int requestedRuns)
    {
        bool nativeComplete = requestedRuns >= 3 && native.Count == requestedRuns;
        bool nativeStable = nativeComplete && native.All(record =>
            record.Oracle.Timing.Stable &&
            record.DirectError <= SemanticTolerance &&
            record.IncumbentError <= SemanticTolerance);
        bool nativeWins = nativeStable && native.All(record =>
            1.0 / record.Oracle.Timing.Ratio >= RequiredGain);
        bool tailsPass = nativeStable && native.All(record =>
            record.Oracle.Tail.Direct.P95 <=
                record.Oracle.Tail.Incumbent.P95 * RequiredGain);

        ExternalRecord[] eligibleExternal = external.Where(record =>
            string.Equals(record.Status, "ok", StringComparison.Ordinal) &&
            IsFinite(record.MedianUs) && record.MedianUs > 0 &&
            IsFinite(record.MaxError) && record.MaxError <= SemanticTolerance).ToArray();
        bool externalComplete = includeExternal &&
            eligibleExternal.Length == requestedRuns * 2 &&
            Enumerable.Range(1, requestedRuns).All(run =>
                eligibleExternal.Count(record => record.Run == run) == 2);

        double directMedian = MedianAcross(native
            .Where(record => record.Oracle.Timing.Stable)
            .Select(record => record.Oracle.Timing.A.Microseconds));
        double incumbentMedian = MedianAcross(native
            .Where(record => record.Oracle.Timing.Stable)
            .Select(record => record.Oracle.Timing.B.Microseconds));
        double fastestExternalMedian = eligibleExternal.Length == 0
            ? double.NaN
            : eligibleExternal.Min(record => record.MedianUs);
        double fastestCompetitorMedian = IsFinite(fastestExternalMedian)
            ? Math.Min(incumbentMedian, fastestExternalMedian)
            : incumbentMedian;
        double fastestCompetitorOverDirect =
            directMedian > 0 && IsFinite(fastestCompetitorMedian)
                ? fastestCompetitorMedian / directMedian
                : double.NaN;
        bool championshipPassed = nativeWins && tailsPass && externalComplete &&
            fastestCompetitorOverDirect >= RequiredGain;

        Console.WriteLine("rglru_championship_json=" + JsonSerializer.Serialize(new
        {
            kind = "direct-ptx-rglru-championship",
            operation = "rglru-b1-s128-d256",
            requested_runs = requestedRuns,
            native_runs = native.Count,
            eligible_external_rows = eligibleExternal.Length,
            direct_median_us = FiniteOrNull(directMedian),
            incumbent_median_us = FiniteOrNull(incumbentMedian),
            fastest_external_median_us = FiniteOrNull(fastestExternalMedian),
            fastest_competitor_median_us = FiniteOrNull(fastestCompetitorMedian),
            fastest_competitor_over_direct = FiniteOrNull(fastestCompetitorOverDirect),
            required_gain = RequiredGain,
            native_stable = nativeStable,
            native_wins = nativeWins,
            tails_passed = tailsPass,
            external_complete = externalComplete,
            championship_passed = championshipPassed,
            promotion = championshipPassed
        }));
        Console.WriteLine(championshipPassed
            ? "release_gate=PROMOTED_EXACT_SM86_FROM_COMPLETE_CHAMPIONSHIP_EVIDENCE"
            : "release_gate=NOT_PROMOTED_INCOMPLETE_OR_FAILED_CHAMPIONSHIP_EVIDENCE");

        if (requestedRuns >= 3 && includeExternal && !championshipPassed)
            throw new InvalidOperationException(
                "RG-LRU championship evidence did not satisfy every promotion gate.");
    }

    private static IntPtr CaptureRepeatedGraph(
        CudaBackend backend, Action launch, int operations) =>
        backend.CaptureGraph(() =>
        {
            for (int i = 0; i < operations; i++) launch();
        });

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

    private static PairedTailEvidence MeasurePairedDeviceDistributions(
        CudaBackend backend,
        Action direct,
        Action incumbent)
    {
        for (int index = 0; index < Warmups; index++)
        {
            direct();
            incumbent();
        }
        backend.Synchronize();

        var directTimings = new double[Samples];
        var incumbentTimings = new double[Samples];
        var ratios = new double[Samples];
        using IGpuEvent start = backend.CreateEvent(enableTiming: true);
        using IGpuEvent stop = backend.CreateEvent(enableTiming: true);
        for (int sample = 0; sample < Samples; sample++)
        {
            if ((sample & 1) == 0)
            {
                directTimings[sample] = MeasureDeviceBatch(
                    backend, direct, start, stop);
                incumbentTimings[sample] = MeasureDeviceBatch(
                    backend, incumbent, start, stop);
            }
            else
            {
                incumbentTimings[sample] = MeasureDeviceBatch(
                    backend, incumbent, start, stop);
                directTimings[sample] = MeasureDeviceBatch(
                    backend, direct, start, stop);
            }
            ratios[sample] = directTimings[sample] / incumbentTimings[sample];
        }
        return new PairedTailEvidence(
            Summarize(directTimings),
            Summarize(incumbentTimings),
            Summarize(ratios));
    }

    private static double MeasureDeviceBatch(
        CudaBackend backend,
        Action action,
        IGpuEvent start,
        IGpuEvent stop)
    {
        backend.RecordEvent(start, backend.DefaultStream);
        for (int launch = 0; launch < LaunchesPerDeviceSample; launch++) action();
        backend.RecordEvent(stop, backend.DefaultStream);
        stop.Synchronize();
        return backend.GetEventElapsedTime(start, stop) * 1_000.0 /
            LaunchesPerDeviceSample;
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
        start.Environment["AIDOTNET_BENCHMARK_ORCHESTRATOR_PID"] =
            Environment.ProcessId.ToString(System.Globalization.CultureInfo.InvariantCulture);
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

    private static bool IsFinite(double value) =>
        !double.IsNaN(value) && !double.IsInfinity(value);

    private static double? FiniteOrNull(double value) =>
        IsFinite(value) ? value : null;

    private static double MedianAcross(IEnumerable<double> values)
    {
        double[] ordered = values.Where(IsFinite).OrderBy(value => value).ToArray();
        return ordered.Length == 0 ? double.NaN : Percentile(ordered, 0.50);
    }

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
