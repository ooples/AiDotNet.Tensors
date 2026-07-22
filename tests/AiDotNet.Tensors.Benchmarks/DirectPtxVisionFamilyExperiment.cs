using System.Diagnostics;
using System.Text.Json;
using AiDotNet.Tensors;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;
using AiDotNet.Tensors.Engines.Gpu;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Benchmarks;

/// <summary>
/// Resident, preallocated AiDotNet comparison matrix for the non-BoxIoU
/// issue-#851 kernels. PyTorch/torchvision competitors are measured by the
/// companion Python process so neither framework's context pollutes the other.
/// </summary>
internal static class DirectPtxVisionFamilyExperiment
{
    private const int Warmups = 30;
    private const int Samples = 101;
    private const int Launches = 25;

    private readonly record struct Distribution(
        double Mean, double Median, double P95, double P99);

    private readonly record struct WorkModel(
        double Units, double Flops, double Bytes, long TemporaryDeviceBytes);

    internal static void Run(int independentRuns = 3)
    {
        if (independentRuns <= 0) throw new ArgumentOutOfRangeException(nameof(independentRuns));
        bool? old = DirectPtxFeatureGate.VisionBoxIouExperimentOverride;
        try
        {
            DirectPtxFeatureGate.VisionBoxIouExperimentOverride = true;
            for (int run = 1; run <= independentRuns; run++)
            {
                GpuBenchmarkEnvironment.RequireNoForeignCompute(
                    $"vision-family-run-{run}-start");
                using var backend = new CudaBackend();
                Console.WriteLine("vision_family_environment_json=" + JsonSerializer.Serialize(new
                {
                    run,
                    gpu = backend.DeviceName,
                    framework = Environment.Version.ToString(),
                    os = Environment.OSVersion.ToString(),
                    warmups = Warmups,
                    samples = Samples,
                    launches_per_device_sample = Launches
                }));
                foreach (DirectPtxVisionSpec spec in Specs())
                    RunCell(backend, run, spec);
                backend.Synchronize();
                GpuBenchmarkEnvironment.RequireNoForeignCompute(
                    $"vision-family-run-{run}-end");
            }
        }
        finally
        {
            DirectPtxFeatureGate.VisionBoxIouExperimentOverride = old;
        }
    }

    private static void RunCell(CudaBackend backend, int run, DirectPtxVisionSpec spec)
    {
        DirectPtxVisionDefinition definition = PtxVisionEmitter.Emit(
            spec, DirectPtxArchitectureFamily.Ampere, 8, 6);
        DirectPtxVisionSpec? pairedSpec = PairedSpec(spec);
        DirectPtxVisionDefinition? pairedDefinition = pairedSpec is { } partner
            ? PtxVisionEmitter.Emit(partner, DirectPtxArchitectureFamily.Ampere, 8, 6)
            : null;
        using var cell = new CellBuffers(backend, definition, pairedDefinition, spec);
        if (!backend.PrewarmDirectPtxVisionKernel(spec))
            throw new InvalidOperationException(
                $"PTX prewarm failed for {spec.Operation}: {backend.DirectPtxLastError}");
        if (pairedSpec is { } paired && !backend.PrewarmDirectPtxVisionKernel(paired))
            throw new InvalidOperationException(
                $"PTX prewarm failed for {paired.Operation}: {backend.DirectPtxLastError}");

        cell.LaunchDirect();
        cell.LaunchCurrent();
        backend.Synchronize();
        double establishedRouteError = cell.MaximumWriteError();
        (double? directOracleError, double? currentOracleError) =
            cell.MaximumCpuOracleErrors();
        double error = Math.Max(establishedRouteError, directOracleError ?? 0);
        double tolerance = CorrectnessTolerance(spec);
        if (!double.IsFinite(error) || error > tolerance)
            throw new InvalidOperationException(
                $"Direct PTX {OperationLabel(spec, pairedSpec is not null)} error " +
                $"{error:G9} exceeds the {tolerance:G9} established-route gate.");
        if (!backend.TryGetDirectPtxVisionAudit(spec, out DirectPtxKernelAudit audit))
            throw new InvalidOperationException($"Missing PTX audit for {spec.Operation}.");
        DirectPtxKernelAudit? pairedAudit = null;
        if (pairedSpec is { } auditSpec)
        {
            if (!backend.TryGetDirectPtxVisionAudit(auditSpec, out DirectPtxKernelAudit value))
                throw new InvalidOperationException($"Missing PTX audit for {auditSpec.Operation}.");
            pairedAudit = value;
        }

        WorkModel model = Model(spec, definition, pairedDefinition);
        IntPtr directGraph = backend.CaptureGraph(cell.LaunchDirect);
        IntPtr currentGraph = backend.CaptureGraph(cell.LaunchCurrent);
        if (directGraph == IntPtr.Zero || currentGraph == IntPtr.Zero)
            throw new InvalidOperationException(
                $"Graph capture failed after prewarm for {spec.Operation}.");
        try
        {
            MeasureAndPrint(backend, run, spec, "Direct PTX", cell.LaunchDirect,
                error, establishedRouteError, directOracleError,
                model, audit, pairedAudit);
            MeasureAndPrint(backend, run, spec, "Direct PTX CUDA graph",
                () => backend.EnqueueCapturedGraph(directGraph), error,
                establishedRouteError, directOracleError, model, audit, pairedAudit);
            MeasureAndPrint(backend, run, spec, "AiDotNet NVRTC", cell.LaunchCurrent,
                currentOracleError ?? 0, 0, currentOracleError, model, null);
            MeasureAndPrint(backend, run, spec, "AiDotNet NVRTC CUDA graph",
                () => backend.EnqueueCapturedGraph(currentGraph), currentOracleError ?? 0,
                0, currentOracleError, model, null);
        }
        finally
        {
            backend.DestroyCapturedGraph(directGraph);
            backend.DestroyCapturedGraph(currentGraph);
        }
    }

    private static void MeasureAndPrint(
        CudaBackend backend,
        int run,
        DirectPtxVisionSpec spec,
        string method,
        Action launch,
        double error,
        double establishedRouteError,
        double? highPrecisionOracleError,
        WorkModel model,
        DirectPtxKernelAudit? audit,
        DirectPtxKernelAudit? pairedAudit = null)
    {
        Distribution device = Device(backend, launch);
        Distribution e2e = EndToEnd(backend, launch);
        long managed = Allocated(backend, launch);
        double gflops = model.Flops == 0 ? 0 :
            model.Units * model.Flops / (device.Median * 1000.0);
        double gbps = model.Units * model.Bytes / (device.Median * 1000.0);
        Console.WriteLine("vision_family_evidence_json=" + JsonSerializer.Serialize(new
        {
            status = "ok",
            run,
            operation = OperationLabel(spec, pairedAudit is not null),
            specialization = audit is null ? "established" : pairedAudit is null
                ? audit.BlueprintId
                : $"{audit.BlueprintId}+{pairedAudit.BlueprintId}",
            method,
            device_mean_us = device.Mean,
            device_median_us = device.Median,
            device_p95_us = device.P95,
            device_p99_us = device.P99,
            e2e_mean_us = e2e.Mean,
            e2e_median_us = e2e.Median,
            e2e_p95_us = e2e.P95,
            e2e_p99_us = e2e.P99,
            gflops,
            algorithmic_gbps = gbps,
            managed_bytes_per_call = managed,
            temporary_device_bytes = model.TemporaryDeviceBytes,
            maximum_error = error,
            established_route_error = establishedRouteError,
            high_precision_oracle_error = highPrecisionOracleError,
            registers_per_thread = audit is null ? (int?)null : Math.Max(
                audit.Function.RegistersPerThread,
                pairedAudit?.Function.RegistersPerThread ?? 0),
            static_shared_bytes = audit is null ? (int?)null : Math.Max(
                audit.Function.StaticSharedBytes,
                pairedAudit?.Function.StaticSharedBytes ?? 0),
            dynamic_shared_bytes = audit is null ? (int?)null : 0,
            local_bytes_per_thread = audit is null ? (int?)null : Math.Max(
                audit.Function.LocalBytesPerThread,
                pairedAudit?.Function.LocalBytesPerThread ?? 0),
            active_blocks_per_sm = audit is null ? (int?)null : Math.Min(
                audit.ActiveBlocksPerMultiprocessor,
                pairedAudit?.ActiveBlocksPerMultiprocessor ?? int.MaxValue),
            max_threads_per_block = audit is null ? (int?)null : Math.Min(
                audit.Function.MaxThreadsPerBlock,
                pairedAudit?.Function.MaxThreadsPerBlock ?? int.MaxValue),
            ptx_version = audit is null ? (int?)null : Math.Max(
                audit.Function.PtxVersion,
                pairedAudit?.Function.PtxVersion ?? 0),
            binary_version = audit is null ? (int?)null : Math.Max(
                audit.Function.BinaryVersion,
                pairedAudit?.Function.BinaryVersion ?? 0),
            ptx_sha256 = audit is null ? null : pairedAudit is null
                ? audit.PtxSha256
                : $"{audit.PtxSha256}+{pairedAudit.PtxSha256}",
            device_fingerprint = audit?.DeviceFingerprint,
            jit_info_log = audit is null ? null : pairedAudit is null
                ? audit.JitInfoLog
                : $"{audit.JitInfoLog}\n--- paired module ---\n{pairedAudit.JitInfoLog}"
        }));
    }

    private static Distribution Device(CudaBackend backend, Action launch)
    {
        Warm(backend, launch);
        var values = new double[Samples];
        using IGpuEvent start = backend.CreateEvent(enableTiming: true);
        using IGpuEvent end = backend.CreateEvent(enableTiming: true);
        for (int i = 0; i < values.Length; i++)
        {
            backend.RecordEvent(start, backend.DefaultStream);
            for (int j = 0; j < Launches; j++) launch();
            backend.RecordEvent(end, backend.DefaultStream);
            end.Synchronize();
            values[i] = backend.GetEventElapsedTime(start, end) * 1000.0 / Launches;
        }
        return Summary(values);
    }

    private static Distribution EndToEnd(CudaBackend backend, Action launch)
    {
        Warm(backend, launch);
        var values = new double[Samples];
        double scale = 1_000_000.0 / Stopwatch.Frequency;
        for (int i = 0; i < values.Length; i++)
        {
            long start = Stopwatch.GetTimestamp();
            launch();
            backend.Synchronize();
            values[i] = (Stopwatch.GetTimestamp() - start) * scale;
        }
        return Summary(values);
    }

    private static long Allocated(CudaBackend backend, Action launch)
    {
        Warm(backend, launch);
        long before = GC.GetAllocatedBytesForCurrentThread();
        for (int i = 0; i < Samples; i++) launch();
        long bytes = (GC.GetAllocatedBytesForCurrentThread() - before) / Samples;
        backend.Synchronize();
        return bytes;
    }

    private static void Warm(CudaBackend backend, Action launch)
    {
        for (int i = 0; i < Warmups; i++) launch();
        backend.Synchronize();
    }

    private static Distribution Summary(double[] values)
    {
        Array.Sort(values);
        return new(values.Average(), Percentile(values, .5),
            Percentile(values, .95), Percentile(values, .99));
    }

    private static double Percentile(double[] values, double percentile)
    {
        double position = percentile * (values.Length - 1);
        int lower = (int)Math.Floor(position), upper = (int)Math.Ceiling(position);
        return values[lower] + (values[upper] - values[lower]) * (position - lower);
    }

    private static string OperationLabel(DirectPtxVisionSpec spec, bool paired) =>
        !paired ? spec.Operation.ToString() : spec.Operation switch
        {
            DirectPtxVisionOperation.IouFamilyBackwardA => "IouFamilyBackward(A+B)",
            DirectPtxVisionOperation.Meshgrid2D =>
                $"Meshgrid2D({(((spec.Flags & 2) != 0) ? "xy" : "ij")},both-outputs)",
            _ => spec.Operation.ToString()
        };

    private static double CorrectnessTolerance(DirectPtxVisionSpec spec) =>
        spec.Operation switch
        {
            DirectPtxVisionOperation.CompleteBoxIou or
            DirectPtxVisionOperation.CIoULoss => 2e-3,
            DirectPtxVisionOperation.IoULossBackward or
            DirectPtxVisionOperation.GIoULossBackward or
            DirectPtxVisionOperation.DIoULossBackward or
            DirectPtxVisionOperation.CIoULossBackward or
            DirectPtxVisionOperation.IouFamilyBackwardA => 2e-2,
            DirectPtxVisionOperation.RoiAlign or
            DirectPtxVisionOperation.PsRoiAlign => 2e-5,
            _ => 2e-6
        };

    private static WorkModel Model(
        DirectPtxVisionSpec spec,
        DirectPtxVisionDefinition definition,
        DirectPtxVisionDefinition? pairedDefinition)
    {
        double units = spec.Operation switch
        {
            DirectPtxVisionOperation.GeneralizedBoxIou or
            DirectPtxVisionOperation.DistanceBoxIou or
            DirectPtxVisionOperation.CompleteBoxIou => (double)spec.D0 * spec.D1,
            DirectPtxVisionOperation.MasksToBoxes => (double)spec.D0 * spec.D1 * spec.D2,
            DirectPtxVisionOperation.RoiAlign or DirectPtxVisionOperation.RoiPool or
            DirectPtxVisionOperation.PsRoiAlign or DirectPtxVisionOperation.PsRoiPool =>
                (double)spec.D4 * (spec.D7 == 0 ? spec.D1 : spec.D7) * spec.D5 * spec.D6,
            DirectPtxVisionOperation.Cross3 => (double)spec.D0 * spec.D1,
            DirectPtxVisionOperation.Meshgrid2D => (double)spec.D0 * spec.D1 * 2,
            DirectPtxVisionOperation.IouFamilyBackwardA => (double)spec.D0 * spec.D1,
            _ => spec.D0
        };
        double flops = spec.Operation switch
        {
            DirectPtxVisionOperation.GeneralizedBoxIou => 32,
            DirectPtxVisionOperation.DistanceBoxIou => 38,
            DirectPtxVisionOperation.CompleteBoxIou => 56,
            DirectPtxVisionOperation.Cross3 => 9,
            DirectPtxVisionOperation.Meshgrid2D => 0,
            _ => 16
        };
        double totalBytes = definition.Blueprint.Tensors.Sum(
            tensor => (double)tensor.RequiredBytes);
        if (pairedDefinition is not null)
        {
            if (spec.Operation == DirectPtxVisionOperation.IouFamilyBackwardA)
            {
                // The backward owner modules share all three read-only inputs.
                totalBytes += pairedDefinition.Blueprint.Tensors[3].RequiredBytes;
            }
            else
            {
                totalBytes += pairedDefinition.Blueprint.Tensors.Sum(
                    tensor => (double)tensor.RequiredBytes);
            }
        }
        long temporaryDeviceBytes = spec.Operation == DirectPtxVisionOperation.Nms
            ? checked((long)spec.D0 * sizeof(float))
            : 0;
        return new(units, flops, totalBytes / Math.Max(units, 1), temporaryDeviceBytes);
    }

    private static DirectPtxVisionSpec? PairedSpec(DirectPtxVisionSpec spec)
    {
        if (spec.Operation == DirectPtxVisionOperation.IouFamilyBackwardA &&
            DirectPtxVisionSpecializations.TryPairwiseBackward(
                ownerA: false, spec.D0, spec.D1, spec.D2,
                out DirectPtxVisionSpec backward))
            return backward;
        if (spec.Operation == DirectPtxVisionOperation.Meshgrid2D &&
            (spec.Flags & 1) == 0 &&
            DirectPtxVisionSpecializations.TryMeshgrid2D(
                spec.D0, spec.D1, 1, (spec.Flags & 2) != 0,
                out DirectPtxVisionSpec mesh))
            return mesh;
        return null;
    }

    private static IEnumerable<DirectPtxVisionSpec> Specs()
    {
        yield return new(DirectPtxVisionOperation.GeneralizedBoxIou, 256, 256);
        yield return new(DirectPtxVisionOperation.DistanceBoxIou, 256, 256);
        yield return new(DirectPtxVisionOperation.CompleteBoxIou, 256, 256);
        yield return new(DirectPtxVisionOperation.BoxArea, 4096);
        yield return new(DirectPtxVisionOperation.BoxConvert, 4096, 0, 2);
        yield return new(DirectPtxVisionOperation.IoULoss, 4096);
        yield return new(DirectPtxVisionOperation.GIoULoss, 4096);
        yield return new(DirectPtxVisionOperation.DIoULoss, 4096);
        yield return new(DirectPtxVisionOperation.CIoULoss, 4096);
        yield return new(DirectPtxVisionOperation.IoULossBackward, 4096);
        yield return new(DirectPtxVisionOperation.GIoULossBackward, 4096);
        yield return new(DirectPtxVisionOperation.DIoULossBackward, 4096);
        yield return new(DirectPtxVisionOperation.CIoULossBackward, 4096);
        yield return new(DirectPtxVisionOperation.IouFamilyBackwardA, 256, 256, 0);
        yield return new(DirectPtxVisionOperation.Nms, 256,
            Flags: 0, ScalarBits: BitConverter.SingleToInt32Bits(.5f));
        yield return new(DirectPtxVisionOperation.Nms, 256,
            Flags: 1, ScalarBits: BitConverter.SingleToInt32Bits(.5f));
        yield return new(DirectPtxVisionOperation.MasksToBoxes, 256, 28, 28);
        yield return new(DirectPtxVisionOperation.RoiAlign,
            1, 256, 56, 56, 256, 7, 7, 256, 2 | 0x100,
            BitConverter.SingleToInt32Bits(.25f));
        yield return new(DirectPtxVisionOperation.RoiPool,
            1, 256, 56, 56, 256, 7, 7, 256, 0,
            BitConverter.SingleToInt32Bits(.25f));
        yield return new(DirectPtxVisionOperation.PsRoiAlign,
            1, 196, 56, 56, 256, 7, 7, 4, 2,
            BitConverter.SingleToInt32Bits(.25f));
        yield return new(DirectPtxVisionOperation.PsRoiPool,
            1, 196, 56, 56, 256, 7, 7, 4, 0,
            BitConverter.SingleToInt32Bits(.25f));
        yield return new(DirectPtxVisionOperation.Cross3, 1024, 1);
        yield return new(DirectPtxVisionOperation.Meshgrid2D, 1024, 256, Flags: 0);
        yield return new(DirectPtxVisionOperation.Meshgrid2D, 1024, 256, Flags: 2);
    }

    private sealed class CellBuffers : IDisposable
    {
        private readonly CudaBackend _backend;
        private readonly DirectPtxVisionDefinition _definition;
        private readonly DirectPtxVisionSpec _spec;
        private readonly IGpuBuffer[] _direct;
        private readonly IGpuBuffer[] _current;
        private readonly IGpuBuffer? _extraInputDirect;
        private readonly IGpuBuffer? _extraInputCurrent;
        private readonly IGpuBuffer? _extraOutputDirect;
        private readonly IGpuBuffer? _extraOutputCurrent;

        internal CellBuffers(
            CudaBackend backend,
            DirectPtxVisionDefinition definition,
            DirectPtxVisionDefinition? pairedDefinition,
            DirectPtxVisionSpec spec)
        {
            _backend = backend;
            _definition = definition;
            _spec = spec;
            _direct = Allocate(definition);
            _current = Allocate(definition);
            if (spec.Operation is DirectPtxVisionOperation.IouFamilyBackwardA or
                DirectPtxVisionOperation.IouFamilyBackwardB)
            {
                int other = spec.Operation == DirectPtxVisionOperation.IouFamilyBackwardA
                    ? spec.D1 : spec.D0;
                _extraOutputDirect = backend.AllocateBuffer(new float[checked(other * 4)]);
                _extraOutputCurrent = backend.AllocateBuffer(new float[checked(other * 4)]);
            }
            else if (spec.Operation == DirectPtxVisionOperation.Meshgrid2D)
            {
                if (pairedDefinition is null)
                    throw new InvalidOperationException("Meshgrid benchmark requires both output modules.");
                DirectPtxTensorContract source = pairedDefinition.Blueprint.Tensors[0];
                DirectPtxTensorContract output = pairedDefinition.Blueprint.Tensors[1];
                _extraInputDirect = backend.AllocateBuffer(InitialData(source, 2));
                _extraInputCurrent = backend.AllocateBuffer(InitialData(source, 2));
                _extraOutputDirect = backend.AllocateBuffer(
                    new float[checked((int)(output.RequiredBytes / 4))]);
                _extraOutputCurrent = backend.AllocateBuffer(
                    new float[checked((int)(output.RequiredBytes / 4))]);
            }
        }

        private IGpuBuffer[] Allocate(DirectPtxVisionDefinition definition)
        {
            var result = new IGpuBuffer[definition.Blueprint.Tensors.Count];
            for (int i = 0; i < result.Length; i++)
                result[i] = _backend.AllocateBuffer(InitialData(
                    definition.Blueprint.Tensors[i], i));
            return result;
        }

        private float[] InitialData(DirectPtxTensorContract contract, int argument)
        {
            int length = checked((int)(contract.RequiredBytes / 4));
            var data = new float[length];
            if ((contract.Access & DirectPtxTensorAccess.Read) == 0) return data;

            if (contract.Layout == DirectPtxPhysicalLayout.BoxXyxy)
            {
                var random = new Random(851_000 + (int)_spec.Operation * 17 + argument);
                for (int i = 0; i < length / 4; i++)
                {
                    float x = (float)random.NextDouble() * 40f;
                    float y = (float)random.NextDouble() * 40f;
                    data[i * 4] = x;
                    data[i * 4 + 1] = y;
                    data[i * 4 + 2] = x + 1f + (float)random.NextDouble() * 15f;
                    data[i * 4 + 3] = y + 1f + (float)random.NextDouble() * 15f;
                }
                return data;
            }
            if (contract.Layout == DirectPtxPhysicalLayout.RoiBoxes)
            {
                for (int i = 0; i < length / 5; i++)
                {
                    data[i * 5] = 0;
                    data[i * 5 + 1] = i % 24;
                    data[i * 5 + 2] = (i * 3) % 24;
                    data[i * 5 + 3] = data[i * 5 + 1] + 16;
                    data[i * 5 + 4] = data[i * 5 + 2] + 16;
                }
                return data;
            }
            if (_spec.Operation == DirectPtxVisionOperation.Nms)
            {
                if (argument == 1)
                    for (int i = 0; i < length; i++) data[i] = 1f - i / (float)length;
                else if (argument == 2)
                    for (int i = 0; i < length; i++) data[i] = i % 8;
                return data;
            }
            if (_spec.Operation == DirectPtxVisionOperation.MasksToBoxes && argument == 0)
            {
                for (int i = 0; i < length; i++) data[i] = i % 23 == 0 ? 1f : 0f;
                return data;
            }
            if (contract.Access == DirectPtxTensorAccess.Read ||
                contract.Access == DirectPtxTensorAccess.ReadWrite)
            {
                for (int i = 0; i < length; i++)
                    data[i] = ((i * 17 + argument * 13) % 101 - 50) / 50f;
            }
            return data;
        }

        internal void LaunchDirect()
        {
            if (_spec.Operation == DirectPtxVisionOperation.Nms)
                _backend.Fill(_direct[3], 0f, _spec.D0);
            if (_spec.Operation == DirectPtxVisionOperation.IouFamilyBackwardA)
            {
                if (!DirectPtxVisionSpecializations.TryPairwiseBackward(
                        ownerA: false, _spec.D0, _spec.D1, _spec.D2,
                        out DirectPtxVisionSpec paired) ||
                    !_backend.TryDirectPtxVisionIouFamilyBackward(
                        _spec, paired, _direct[0], _direct[1], _direct[2],
                        _direct[3], _extraOutputDirect!))
                    throw new InvalidOperationException(
                        $"Direct PTX IoU-family backward launch failed: {_backend.DirectPtxLastError}");
                return;
            }
            if (_spec.Operation == DirectPtxVisionOperation.Meshgrid2D)
            {
                if (!((IDirectPtxVisionBackend)_backend).TryDirectPtxMeshgrid2DPair(
                        _direct[0], _extraInputDirect!, _direct[1], _extraOutputDirect!,
                        _spec.D0, _spec.D1, (_spec.Flags & 2) != 0))
                    throw new InvalidOperationException(
                        $"Direct PTX meshgrid pair launch failed: {_backend.DirectPtxLastError}");
                return;
            }
            if (!_backend.TryDirectPtxVisionKernel(
                    _spec, _direct[0], At(_direct, 1), At(_direct, 2),
                    At(_direct, 3), At(_direct, 4), At(_direct, 5)))
                throw new InvalidOperationException(
                    $"Direct PTX {_spec.Operation} launch failed: {_backend.DirectPtxLastError}");
        }

        internal void LaunchCurrent()
        {
            bool? old = DirectPtxFeatureGate.VisionBoxIouExperimentOverride;
            DirectPtxFeatureGate.VisionBoxIouExperimentOverride = false;
            try
            {
                switch (_spec.Operation)
                {
                    case DirectPtxVisionOperation.GeneralizedBoxIou:
                        _backend.GeneralizedBoxIou(_current[0], _current[1], _current[2], _spec.D0, _spec.D1); break;
                    case DirectPtxVisionOperation.DistanceBoxIou:
                        _backend.DistanceBoxIou(_current[0], _current[1], _current[2], _spec.D0, _spec.D1); break;
                    case DirectPtxVisionOperation.CompleteBoxIou:
                        _backend.CompleteBoxIou(_current[0], _current[1], _current[2], _spec.D0, _spec.D1); break;
                    case DirectPtxVisionOperation.BoxArea:
                        _backend.BoxArea(_current[0], _current[1], _spec.D0); break;
                    case DirectPtxVisionOperation.BoxConvert:
                        _backend.BoxConvert(_current[0], _current[1], _spec.D0, _spec.D1, _spec.D2); break;
                    case DirectPtxVisionOperation.IoULoss:
                        _backend.IoULoss(_current[0], _current[1], _current[2], _spec.D0); break;
                    case DirectPtxVisionOperation.GIoULoss:
                        _backend.GIoULoss(_current[0], _current[1], _current[2], _spec.D0); break;
                    case DirectPtxVisionOperation.DIoULoss:
                        _backend.DIoULoss(_current[0], _current[1], _current[2], _spec.D0); break;
                    case DirectPtxVisionOperation.CIoULoss:
                        _backend.CIoULoss(_current[0], _current[1], _current[2], _spec.D0); break;
                    case DirectPtxVisionOperation.IoULossBackward:
                        _backend.IoULossBackward(_current[0], _current[1], _current[2], _current[3], _spec.D0); break;
                    case DirectPtxVisionOperation.GIoULossBackward:
                        _backend.GIoULossBackward(_current[0], _current[1], _current[2], _current[3], _spec.D0); break;
                    case DirectPtxVisionOperation.DIoULossBackward:
                        _backend.DIoULossBackward(_current[0], _current[1], _current[2], _current[3], _spec.D0); break;
                    case DirectPtxVisionOperation.CIoULossBackward:
                        _backend.CIoULossBackward(_current[0], _current[1], _current[2], _current[3], _spec.D0); break;
                    case DirectPtxVisionOperation.IouFamilyBackwardA:
                    {
                        _backend.IouFamilyBackward(_current[0], _current[1], _current[2],
                            _current[3], _extraOutputCurrent!,
                            _spec.D0, _spec.D1, _spec.D2);
                        break;
                    }
                    case DirectPtxVisionOperation.Nms:
                        _backend.Fill(_current[3], 0f, _spec.D0);
                        _backend.Nms(_current[0], _current[1], _current[2], _current[3],
                            _current[4], _current[5], _spec.D0,
                            BitConverter.Int32BitsToSingle(_spec.ScalarBits), _spec.Flags & 1); break;
                    case DirectPtxVisionOperation.MasksToBoxes:
                        _backend.MasksToBoxes(_current[0], _current[1], _spec.D0, _spec.D1, _spec.D2); break;
                    case DirectPtxVisionOperation.RoiAlign:
                        _backend.RoIAlign(_current[0], _current[1], _current[2], _spec.D0, _spec.D1,
                            _spec.D2, _spec.D3, _spec.D4, _spec.D5, _spec.D6,
                            BitConverter.Int32BitsToSingle(_spec.ScalarBits), _spec.Flags & 0xff,
                            (_spec.Flags & 0x100) != 0); break;
                    case DirectPtxVisionOperation.RoiPool:
                        _backend.RoIPool(_current[0], _current[1], _current[2], _spec.D0, _spec.D1,
                            _spec.D2, _spec.D3, _spec.D4, _spec.D5, _spec.D6,
                            BitConverter.Int32BitsToSingle(_spec.ScalarBits)); break;
                    case DirectPtxVisionOperation.PsRoiAlign:
                        _backend.PsRoIAlign(_current[0], _current[1], _current[2], _spec.D0, _spec.D1,
                            _spec.D2, _spec.D3, _spec.D4, _spec.D5, _spec.D6, _spec.D7,
                            BitConverter.Int32BitsToSingle(_spec.ScalarBits), _spec.Flags & 0xff); break;
                    case DirectPtxVisionOperation.PsRoiPool:
                        _backend.PsRoIPool(_current[0], _current[1], _current[2], _spec.D0, _spec.D1,
                            _spec.D2, _spec.D3, _spec.D4, _spec.D5, _spec.D6, _spec.D7,
                            BitConverter.Int32BitsToSingle(_spec.ScalarBits)); break;
                    case DirectPtxVisionOperation.Cross3:
                        _backend.Cross3(_current[0], _current[1], _current[2], _spec.D0, _spec.D1); break;
                    case DirectPtxVisionOperation.Meshgrid2D:
                    {
                        bool xy = (_spec.Flags & 2) != 0;
                        if (!xy)
                        {
                            _backend.RepeatElements(_current[0], _current[1], _spec.D0, 1, _spec.D1);
                            _backend.TileLastAxis(_extraInputCurrent!, _extraOutputCurrent!,
                                1, _spec.D1, _spec.D0);
                        }
                        else
                        {
                            _backend.TileLastAxis(_current[0], _current[1], 1, _spec.D0, _spec.D1);
                            _backend.RepeatElements(_extraInputCurrent!, _extraOutputCurrent!,
                                _spec.D1, 1, _spec.D0);
                        }
                        break;
                    }
                    default: throw new NotSupportedException(_spec.Operation.ToString());
                }
            }
            finally
            {
                DirectPtxFeatureGate.VisionBoxIouExperimentOverride = old;
            }
        }

        internal double MaximumWriteError()
        {
            double maximum = 0;
            for (int i = 0; i < _definition.Blueprint.Tensors.Count; i++)
            {
                if ((_definition.Blueprint.Tensors[i].Access & DirectPtxTensorAccess.Write) == 0)
                    continue;
                float[] left = _backend.DownloadBuffer(_direct[i]);
                float[] right = _backend.DownloadBuffer(_current[i]);
                for (int j = 0; j < left.Length; j++)
                    maximum = Math.Max(maximum, Math.Abs(left[j] - right[j]));
            }
            if (_spec.Operation is DirectPtxVisionOperation.IouFamilyBackwardA or
                DirectPtxVisionOperation.Meshgrid2D)
            {
                float[] left = _backend.DownloadBuffer(_extraOutputDirect!);
                float[] right = _backend.DownloadBuffer(_extraOutputCurrent!);
                for (int i = 0; i < left.Length; i++)
                    maximum = Math.Max(maximum, Math.Abs(left[i] - right[i]));
            }
            return maximum;
        }

        internal (double? Direct, double? Current) MaximumCpuOracleErrors()
        {
            double[][]? oracle = BuildCpuOracle();
            if (oracle is null) return (null, null);
            float[][] direct = SemanticOutputs(_direct, _extraOutputDirect);
            float[][] current = SemanticOutputs(_current, _extraOutputCurrent);
            if (direct.Length != oracle.Length || current.Length != oracle.Length)
                throw new InvalidOperationException(
                    $"Oracle output count mismatch for {_spec.Operation}.");
            return (MaximumError(direct, oracle), MaximumError(current, oracle));
        }

        private float[][] SemanticOutputs(IGpuBuffer[] buffers, IGpuBuffer? extra)
        {
            if (_spec.Operation == DirectPtxVisionOperation.Nms)
                return [_backend.DownloadBuffer(buffers[4]), _backend.DownloadBuffer(buffers[5])];
            if (_spec.Operation is DirectPtxVisionOperation.IouFamilyBackwardA or
                DirectPtxVisionOperation.Meshgrid2D)
                return [_backend.DownloadBuffer(buffers[^1]), _backend.DownloadBuffer(extra!)];
            return _definition.Blueprint.Tensors
                .Select((contract, index) => (contract, index))
                .Where(item => (item.contract.Access & DirectPtxTensorAccess.Write) != 0)
                .Select(item => _backend.DownloadBuffer(buffers[item.index]))
                .ToArray();
        }

        private static double MaximumError(float[][] actual, double[][] expected)
        {
            double maximum = 0;
            for (int output = 0; output < expected.Length; output++)
            {
                if (actual[output].Length != expected[output].Length)
                    return double.PositiveInfinity;
                for (int i = 0; i < expected[output].Length; i++)
                    maximum = Math.Max(maximum,
                        Math.Abs(actual[output][i] - expected[output][i]));
            }
            return maximum;
        }

        private double[][]? BuildCpuOracle()
        {
            var cpu = new CpuEngine();
            switch (_spec.Operation)
            {
                case DirectPtxVisionOperation.GeneralizedBoxIou:
                case DirectPtxVisionOperation.DistanceBoxIou:
                case DirectPtxVisionOperation.CompleteBoxIou:
                {
                    using Tensor<double> a = Host(_direct[0], _spec.D0, 4);
                    using Tensor<double> b = Host(_direct[1], _spec.D1, 4);
                    using Tensor<double> output = _spec.Operation switch
                    {
                        DirectPtxVisionOperation.GeneralizedBoxIou => cpu.GeneralizedBoxIou(a, b),
                        DirectPtxVisionOperation.DistanceBoxIou => cpu.DistanceBoxIou(a, b),
                        _ => cpu.CompleteBoxIou(a, b)
                    };
                    return [Copy(output)];
                }
                case DirectPtxVisionOperation.BoxArea:
                {
                    using Tensor<double> boxes = Host(_direct[0], _spec.D0, 4);
                    using Tensor<double> output = cpu.BoxArea(boxes);
                    return [Copy(output)];
                }
                case DirectPtxVisionOperation.BoxConvert:
                {
                    using Tensor<double> boxes = Host(_direct[0], _spec.D0, 4);
                    using Tensor<double> output = cpu.BoxConvert(
                        boxes, (BoxFormat)_spec.D1, (BoxFormat)_spec.D2);
                    return [Copy(output)];
                }
                case DirectPtxVisionOperation.IoULoss:
                case DirectPtxVisionOperation.GIoULoss:
                case DirectPtxVisionOperation.DIoULoss:
                case DirectPtxVisionOperation.CIoULoss:
                {
                    using Tensor<double> predicted = Host(_direct[0], _spec.D0, 4);
                    using Tensor<double> target = Host(_direct[1], _spec.D0, 4);
                    using Tensor<double> output = _spec.Operation switch
                    {
                        DirectPtxVisionOperation.IoULoss => cpu.TensorIoULoss(predicted, target),
                        DirectPtxVisionOperation.GIoULoss => cpu.TensorGIoULoss(predicted, target),
                        DirectPtxVisionOperation.DIoULoss => cpu.TensorDIoULoss(predicted, target),
                        _ => cpu.TensorCIoULoss(predicted, target)
                    };
                    return [Copy(output)];
                }
                case DirectPtxVisionOperation.IoULossBackward:
                case DirectPtxVisionOperation.GIoULossBackward:
                case DirectPtxVisionOperation.DIoULossBackward:
                case DirectPtxVisionOperation.CIoULossBackward:
                    // The aligned backward methods are backend-only. Their
                    // established analytical CUDA implementation is the
                    // independent reference reported by established_route_error.
                    return null;
                case DirectPtxVisionOperation.IouFamilyBackwardA:
                {
                    using Tensor<double> grad = Host(_direct[0], _spec.D0, _spec.D1);
                    using Tensor<double> boxesA = Host(_direct[1], _spec.D0, 4);
                    using Tensor<double> boxesB = Host(_direct[2], _spec.D1, 4);
                    (Tensor<double> gradA, Tensor<double> gradB) pair = _spec.D2 switch
                    {
                        0 => cpu.BoxIouBackward(grad, boxesA, boxesB),
                        1 => cpu.GeneralizedBoxIouBackward(grad, boxesA, boxesB),
                        2 => cpu.DistanceBoxIouBackward(grad, boxesA, boxesB),
                        _ => cpu.CompleteBoxIouBackward(grad, boxesA, boxesB)
                    };
                    using (pair.gradA)
                    using (pair.gradB)
                        return [Copy(pair.gradA), Copy(pair.gradB)];
                }
                case DirectPtxVisionOperation.Nms:
                {
                    using Tensor<double> boxes = Host(_direct[0], _spec.D0, 4);
                    using Tensor<double> scores = Host(_direct[1], _spec.D0);
                    Tensor<int>? classes = null;
                    try
                    {
                        if ((_spec.Flags & 1) != 0)
                            classes = HostInt(_direct[2], _spec.D0);
                        using Tensor<int> indices = classes is null
                            ? cpu.Nms(boxes, scores, 0.5)
                            : cpu.BatchedNms(boxes, scores, classes, 0.5);
                        var output = new double[_spec.D0];
                        ReadOnlySpan<int> kept = indices.AsSpan();
                        for (int i = 0; i < kept.Length; i++) output[i] = kept[i];
                        return [output, [kept.Length]];
                    }
                    finally { classes?.Dispose(); }
                }
                case DirectPtxVisionOperation.MasksToBoxes:
                {
                    using Tensor<double> masks = Host(
                        _direct[0], _spec.D0, _spec.D1, _spec.D2);
                    using Tensor<int> output = cpu.MasksToBoxes(masks);
                    return [Copy(output)];
                }
                case DirectPtxVisionOperation.RoiAlign:
                case DirectPtxVisionOperation.RoiPool:
                case DirectPtxVisionOperation.PsRoiAlign:
                case DirectPtxVisionOperation.PsRoiPool:
                {
                    using Tensor<double> input = Host(
                        _direct[0], _spec.D0, _spec.D1, _spec.D2, _spec.D3);
                    using Tensor<double> boxes = Host(_direct[1], _spec.D4, 5);
                    float scale = BitConverter.Int32BitsToSingle(_spec.ScalarBits);
                    using Tensor<double> output = _spec.Operation switch
                    {
                        DirectPtxVisionOperation.RoiAlign => cpu.RoIAlign(
                            input, boxes, _spec.D5, _spec.D6, scale,
                            _spec.Flags & 0xff, (_spec.Flags & 0x100) != 0),
                        DirectPtxVisionOperation.RoiPool => cpu.RoIPool(
                            input, boxes, _spec.D5, _spec.D6, scale),
                        DirectPtxVisionOperation.PsRoiAlign => cpu.PsRoIAlign(
                            input, boxes, _spec.D5, _spec.D6, _spec.D7,
                            scale, _spec.Flags & 0xff),
                        _ => cpu.PsRoIPool(
                            input, boxes, _spec.D5, _spec.D6, _spec.D7, scale)
                    };
                    return [Copy(output)];
                }
                case DirectPtxVisionOperation.Cross3:
                {
                    using Tensor<double> a = Host(_direct[0], _spec.D0, 3, _spec.D1);
                    using Tensor<double> b = Host(_direct[1], _spec.D0, 3, _spec.D1);
                    using Tensor<double> output = cpu.TensorCross(a, b, 1);
                    return [Copy(output)];
                }
                case DirectPtxVisionOperation.Meshgrid2D:
                {
                    using Tensor<double> x = Host(_direct[0], _spec.D0);
                    using Tensor<double> y = Host(_extraInputDirect!, _spec.D1);
                    Tensor<double>[] grids = cpu.TensorMeshgrid(
                        [x, y], (_spec.Flags & 2) != 0 ? "xy" : "ij");
                    try { return [Copy(grids[0]), Copy(grids[1])]; }
                    finally
                    {
                        grids[0].Dispose();
                        grids[1].Dispose();
                    }
                }
                default:
                    throw new NotSupportedException(
                        $"No CPU oracle is assigned for {_spec.Operation}.");
            }
        }

        private Tensor<double> Host(IGpuBuffer buffer, params int[] shape)
        {
            float[] values = _backend.DownloadBuffer(buffer);
            var converted = new double[values.Length];
            for (int i = 0; i < values.Length; i++) converted[i] = values[i];
            return new Tensor<double>(converted, shape);
        }

        private Tensor<int> HostInt(IGpuBuffer buffer, params int[] shape)
        {
            float[] values = _backend.DownloadBuffer(buffer);
            var converted = new int[values.Length];
            for (int i = 0; i < values.Length; i++) converted[i] = (int)values[i];
            return new Tensor<int>(converted, shape);
        }

        private static double[] Copy(Tensor<double> tensor) => tensor.AsSpan().ToArray();

        private static double[] Copy(Tensor<int> tensor)
        {
            ReadOnlySpan<int> values = tensor.AsSpan();
            var result = new double[values.Length];
            for (int i = 0; i < result.Length; i++) result[i] = values[i];
            return result;
        }

        private static IGpuBuffer? At(IGpuBuffer[] buffers, int index) =>
            index < buffers.Length ? buffers[index] : null;

        public void Dispose()
        {
            foreach (IGpuBuffer buffer in _direct) buffer.Dispose();
            foreach (IGpuBuffer buffer in _current) buffer.Dispose();
            _extraInputDirect?.Dispose();
            _extraInputCurrent?.Dispose();
            _extraOutputDirect?.Dispose();
            _extraOutputCurrent?.Dispose();
        }
    }
}
