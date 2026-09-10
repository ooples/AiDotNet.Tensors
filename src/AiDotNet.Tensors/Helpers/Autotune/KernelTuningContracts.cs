using System.Globalization;
using AiDotNet.Evolution;

namespace AiDotNet.Tensors.Helpers.Autotune;

/// <summary>Execution device families relevant to tuning compatibility and coordination.</summary>
public enum KernelTuningDeviceKind
{
    /// <summary>Host processor.</summary>
    Cpu = 0,
    /// <summary>NVIDIA GPU.</summary>
    NvidiaGpu = 1,
    /// <summary>AMD GPU.</summary>
    AmdGpu = 2,
    /// <summary>Intel GPU.</summary>
    IntelGpu = 3,
    /// <summary>Apple GPU.</summary>
    AppleGpu = 4,
    /// <summary>Other accelerator.</summary>
    OtherAccelerator = 5
}

/// <summary>
/// Runtime/compiler backend that turns a typed kernel configuration into device code.
/// Kept separate from <see cref="KernelTuningDeviceKind"/> because the same physical
/// GPU can execute through more than one incompatible backend (for example HIP and
/// OpenCL on AMD, or CUDA and OpenCL on NVIDIA).
/// </summary>
public enum KernelTuningBackend
{
    /// <summary>Managed CPU implementation.</summary>
    ManagedCpu = 0,
    /// <summary>.NET JIT or intrinsic-emitted CPU implementation.</summary>
    DotNetJit = 1,
    /// <summary>NVIDIA CUDA driver path, including NVRTC and direct PTX.</summary>
    Cuda = 2,
    /// <summary>OpenCL C compiled by the selected device driver.</summary>
    OpenCl = 3,
    /// <summary>AMD HIP/ROCm runtime compilation path.</summary>
    Hip = 4,
    /// <summary>Apple Metal Shading Language path.</summary>
    Metal = 5,
    /// <summary>Vulkan SPIR-V path.</summary>
    Vulkan = 6,
    /// <summary>WebGPU WGSL path.</summary>
    WebGpu = 7
}

/// <summary>Typed identity of a CPU or accelerator used for cache isolation and tuning coordination.</summary>
public readonly record struct KernelTuningDeviceFingerprint
{
    /// <summary>Creates a validated device fingerprint.</summary>
    public KernelTuningDeviceFingerprint(KernelTuningDeviceKind kind, string localKey, string modelKey)
    {
        if (!Enum.IsDefined(typeof(KernelTuningDeviceKind), kind))
            throw new ArgumentOutOfRangeException(nameof(kind));
        if (string.IsNullOrWhiteSpace(localKey))
            throw new ArgumentException("A per-device key is required.", nameof(localKey));
        if (string.IsNullOrWhiteSpace(modelKey))
            throw new ArgumentException("A model-level key is required.", nameof(modelKey));
        Kind = kind;
        LocalKey = localKey;
        ModelKey = modelKey;
    }

    /// <summary>Gets the device family.</summary>
    public KernelTuningDeviceKind Kind { get; }
    /// <summary>Gets the per-physical-device and driver key.</summary>
    public string LocalKey { get; }
    /// <summary>Gets the shareable model and driver key.</summary>
    public string ModelKey { get; }

    /// <summary>Creates a fingerprint for the current host CPU.</summary>
    public static KernelTuningDeviceFingerprint CurrentCpu() => new(
        KernelTuningDeviceKind.Cpu,
        AutotuneCache.CurrentHardwareFingerprint,
        AutotuneCache.CurrentHardwareFingerprint);

    /// <summary>Converts a structured GPU fingerprint without losing its local/model distinction.</summary>
    public static KernelTuningDeviceFingerprint FromGpu(GpuDeviceFingerprint device)
    {
        if (string.IsNullOrWhiteSpace(device.UniqueId))
            throw new ArgumentException("A valid GPU fingerprint is required.", nameof(device));
        return new KernelTuningDeviceFingerprint(
            device.VendorKind switch
            {
                GpuVendorKind.Nvidia => KernelTuningDeviceKind.NvidiaGpu,
                GpuVendorKind.Amd => KernelTuningDeviceKind.AmdGpu,
                GpuVendorKind.Intel => KernelTuningDeviceKind.IntelGpu,
                GpuVendorKind.Apple => KernelTuningDeviceKind.AppleGpu,
                GpuVendorKind.Other => KernelTuningDeviceKind.OtherAccelerator,
                _ => throw new ArgumentOutOfRangeException(nameof(device))
            },
            device.LocalKey,
            device.ModelKey);
    }
}

/// <summary>Version of a typed kernel configuration space and its mutation rules.</summary>
public readonly record struct KernelSearchSpaceVersion
{
    /// <summary>Creates a positive search-space version.</summary>
    public KernelSearchSpaceVersion(int value)
    {
        if (value <= 0) throw new ArgumentOutOfRangeException(nameof(value));
        Value = value;
    }

    /// <summary>Gets the positive version number.</summary>
    public int Value { get; }

    /// <inheritdoc />
    public override string ToString() => Value.ToString(CultureInfo.InvariantCulture);
}

/// <summary>Version of the correctness and timing protocol used to evaluate a kernel.</summary>
public readonly record struct KernelBenchmarkProtocolVersion
{
    /// <summary>Creates a positive benchmark-protocol version.</summary>
    public KernelBenchmarkProtocolVersion(int value)
    {
        if (value <= 0) throw new ArgumentOutOfRangeException(nameof(value));
        Value = value;
    }

    /// <summary>Gets the positive version number.</summary>
    public int Value { get; }

    /// <inheritdoc />
    public override string ToString() => Value.ToString(CultureInfo.InvariantCulture);
}

/// <summary>Stable identity of one kernel-tuning problem on one device and input shape.</summary>
public sealed class KernelTuningIdentity
{
    /// <summary>Creates a kernel-tuning identity.</summary>
    public KernelTuningIdentity(
        KernelId kernel,
        ShapeProfile shape,
        KernelTuningDeviceFingerprint device,
        KernelTuningBackend backend,
        KernelSearchSpaceVersion searchSpaceVersion,
        KernelBenchmarkProtocolVersion benchmarkProtocolVersion)
    {
        if (string.IsNullOrWhiteSpace(kernel.Category))
            throw new ArgumentException("A kernel category is required.", nameof(kernel));
        if (string.IsNullOrWhiteSpace(kernel.Name))
            throw new ArgumentException("A kernel name is required.", nameof(kernel));
        if (string.IsNullOrWhiteSpace(device.LocalKey))
            throw new ArgumentException("A valid device fingerprint is required.", nameof(device));
        if (!Enum.IsDefined(typeof(KernelTuningBackend), backend))
            throw new ArgumentOutOfRangeException(nameof(backend));
        if (searchSpaceVersion.Value <= 0) throw new ArgumentOutOfRangeException(nameof(searchSpaceVersion));
        if (benchmarkProtocolVersion.Value <= 0)
            throw new ArgumentOutOfRangeException(nameof(benchmarkProtocolVersion));

        Kernel = kernel;
        Shape = shape is null
            ? throw new ArgumentNullException(nameof(shape))
            : new ShapeProfile(shape.Dimensions);
        Device = device;
        Backend = backend;
        SearchSpaceVersion = searchSpaceVersion;
        BenchmarkProtocolVersion = benchmarkProtocolVersion;
        StableKey = EvolutionHash.Combine(new[]
        {
            "tensor-kernel-tuning-identity-v3",
            Kernel.ToFileStem(),
            Shape.ToFileStem(),
            Device.LocalKey,
            ((int)Backend).ToString(CultureInfo.InvariantCulture),
            SearchSpaceVersion.ToString(),
            BenchmarkProtocolVersion.ToString()
        });
    }

    /// <summary>Creates an identity for a GPU without exposing stringly device keys.</summary>
    public KernelTuningIdentity(
        KernelId kernel,
        ShapeProfile shape,
        GpuDeviceFingerprint device,
        KernelTuningBackend backend,
        KernelSearchSpaceVersion searchSpaceVersion,
        KernelBenchmarkProtocolVersion benchmarkProtocolVersion)
        : this(
            kernel,
            shape,
            KernelTuningDeviceFingerprint.FromGpu(device),
            backend,
            searchSpaceVersion,
            benchmarkProtocolVersion)
    {
    }

    /// <summary>Gets the tuned kernel family.</summary>
    public KernelId Kernel { get; }
    /// <summary>Gets an immutable copy of the input shape profile.</summary>
    public ShapeProfile Shape { get; }
    /// <summary>Gets the physical device and driver identity.</summary>
    public KernelTuningDeviceFingerprint Device { get; }
    /// <summary>Gets the runtime/compiler backend whose artifacts and measurements are compatible.</summary>
    public KernelTuningBackend Backend { get; }
    /// <summary>Gets the typed search-space version.</summary>
    public KernelSearchSpaceVersion SearchSpaceVersion { get; }
    /// <summary>Gets the typed correctness and timing protocol version.</summary>
    public KernelBenchmarkProtocolVersion BenchmarkProtocolVersion { get; }
    /// <summary>Gets a stable hash covering every compatibility input.</summary>
    public string StableKey { get; }
}

/// <summary>Typed measurements understood by the kernel quality-diversity adapter.</summary>
public enum KernelTuningMetric
{
    /// <summary>Billions of floating-point operations per second.</summary>
    ThroughputGflops = 0,
    /// <summary>Median measured execution latency.</summary>
    MedianLatencyMilliseconds = 1,
    /// <summary>95th-percentile measured execution latency.</summary>
    P95LatencyMilliseconds = 2,
    /// <summary>Temporary workspace allocated by the candidate.</summary>
    WorkspaceBytes = 3,
    /// <summary>Base-two logarithm of workspace bytes plus one.</summary>
    Log2WorkspaceBytes = 4,
    /// <summary>Estimated or measured device occupancy from zero to one.</summary>
    OccupancyRatio = 5,
    /// <summary>Registers consumed per thread.</summary>
    RegistersPerThread = 6,
    /// <summary>Candidate compilation latency.</summary>
    CompileMilliseconds = 7,
    /// <summary>Largest locally validated numerical error.</summary>
    MaximumNumericalError = 8,
    /// <summary>Base-ten logarithm of the numerical error with a finite floor.</summary>
    Log10NumericalError = 9,
    /// <summary>Number of kernel launches in one evaluated operation.</summary>
    KernelLaunchCount = 10
}

/// <summary>How much correctness evidence a successful trial carries.</summary>
public enum KernelTuningValidationScope
{
    /// <summary>The candidate output was compared with a trusted reference.</summary>
    Output = 0,
    /// <summary>Both output and gradients were compared with a trusted reference.</summary>
    OutputAndGradient = 1
}

/// <summary>Terminal result of a correctness-first kernel trial.</summary>
public enum KernelTuningTrialStatus
{
    /// <summary>Compilation, correctness checks, and robust timing all succeeded.</summary>
    Passed = 0,
    /// <summary>The typed schedule violates a static constraint.</summary>
    InvalidConfiguration = 1,
    /// <summary>The candidate exceeds a device resource limit.</summary>
    ResourceLimitExceeded = 2,
    /// <summary>The candidate could not be compiled.</summary>
    CompilationFailed = 3,
    /// <summary>The output differs from the trusted reference.</summary>
    OutputMismatch = 4,
    /// <summary>The gradient differs from the trusted reference.</summary>
    GradientMismatch = 5,
    /// <summary>Benchmark execution failed after correctness validation.</summary>
    BenchmarkFailed = 6,
    /// <summary>A descriptor required by the search policy was not measured by this backend.</summary>
    RequiredMetricUnavailable = 7
}

/// <summary>
/// Converts fractional timing units without the whole-millisecond rounding performed by .NET Framework's
/// <see cref="TimeSpan.FromMilliseconds(double)"/> and <see cref="TimeSpan.FromSeconds(double)"/> implementations.
/// </summary>
internal static class KernelTuningDuration
{
    internal static TimeSpan FromMilliseconds(double milliseconds) =>
        FromUnits(milliseconds, TimeSpan.TicksPerMillisecond, nameof(milliseconds));

    internal static TimeSpan FromSeconds(double seconds) =>
        FromUnits(seconds, TimeSpan.TicksPerSecond, nameof(seconds));

    private static TimeSpan FromUnits(double value, long ticksPerUnit, string parameterName)
    {
        if (double.IsNaN(value) || double.IsInfinity(value))
            throw new ArgumentOutOfRangeException(parameterName, value, "A duration must be finite.");

        double ticks = value * ticksPerUnit;
        if (double.IsNaN(ticks) || double.IsInfinity(ticks))
            throw new ArgumentOutOfRangeException(parameterName, value, "The duration exceeds the TimeSpan range.");

        try
        {
            long roundedTicks = checked((long)Math.Round(ticks, MidpointRounding.AwayFromZero));
            return TimeSpan.FromTicks(roundedTicks);
        }
        catch (OverflowException)
        {
            throw new ArgumentOutOfRangeException(
                parameterName, value, "The duration exceeds the TimeSpan range.");
        }
    }
}

/// <summary>Robust latency statistics computed from repeated device measurements.</summary>
public sealed class KernelTimingStatistics
{
    /// <summary>Minimum number of measured samples accepted by the tuning infrastructure.</summary>
    public const int MinimumSampleCount = 3;

    private readonly IReadOnlyList<TimeSpan> _samples;

    private KernelTimingStatistics(
        int sampleCount,
        TimeSpan median,
        TimeSpan p95,
        IReadOnlyList<TimeSpan> samples)
    {
        SampleCount = sampleCount;
        Median = median;
        P95 = p95;
        _samples = samples;
    }

    /// <summary>Gets the number of post-warmup samples.</summary>
    public int SampleCount { get; }
    /// <summary>Gets the median latency.</summary>
    public TimeSpan Median { get; }
    /// <summary>Gets the nearest-rank 95th-percentile latency.</summary>
    public TimeSpan P95 { get; }
    /// <summary>Gets the immutable raw samples when this instance was computed locally.</summary>
    public IReadOnlyList<TimeSpan> Samples => _samples;
    /// <summary>Gets whether raw samples, rather than only a persisted summary, are available.</summary>
    public bool HasRawSamples => _samples.Count == SampleCount;

    /// <summary>Computes immutable statistics from post-warmup samples.</summary>
    public static KernelTimingStatistics FromSamples(IEnumerable<TimeSpan> samples)
    {
        if (samples is null) throw new ArgumentNullException(nameof(samples));
        TimeSpan[] ordered = samples.ToArray();
        if (ordered.Length < MinimumSampleCount)
            throw new ArgumentException(
                $"At least {MinimumSampleCount} post-warmup measurements are required.", nameof(samples));
        for (int i = 0; i < ordered.Length; i++)
        {
            if (ordered[i] <= TimeSpan.Zero)
                throw new ArgumentOutOfRangeException(nameof(samples), "Every timing sample must be finite and positive.");
        }

        Array.Sort(ordered);
        long medianTicks = ordered.Length % 2 == 0
            ? ordered[ordered.Length / 2 - 1].Ticks +
              (ordered[ordered.Length / 2].Ticks - ordered[ordered.Length / 2 - 1].Ticks) / 2L
            : ordered[ordered.Length / 2].Ticks;
        int p95Index = Math.Max(0, (int)Math.Ceiling(ordered.Length * 0.95d) - 1);
        return new KernelTimingStatistics(
            ordered.Length,
            TimeSpan.FromTicks(medianTicks),
            ordered[p95Index],
            Array.AsReadOnly(ordered));
    }

    internal static KernelTimingStatistics FromSummary(int sampleCount, double medianMs, double p95Ms)
    {
        if (sampleCount < MinimumSampleCount) throw new ArgumentOutOfRangeException(nameof(sampleCount));
        if (!KernelTuningMeasurement.IsFinite(medianMs) || medianMs <= 0)
            throw new ArgumentOutOfRangeException(nameof(medianMs));
        if (!KernelTuningMeasurement.IsFinite(p95Ms) || p95Ms < medianMs)
            throw new ArgumentOutOfRangeException(nameof(p95Ms));
        return new KernelTimingStatistics(
            sampleCount,
            KernelTuningDuration.FromMilliseconds(medianMs),
            KernelTuningDuration.FromMilliseconds(p95Ms),
            Array.Empty<TimeSpan>());
    }
}

/// <summary>Numerical evidence required before a candidate may be benchmarked or deployed.</summary>
public sealed class KernelTuningCorrectnessEvidence
{
    /// <summary>Creates validated output, and optionally gradient, evidence.</summary>
    public KernelTuningCorrectnessEvidence(
        KernelTuningValidationScope scope,
        double outputAbsoluteError,
        double outputRelativeError,
        double outputAbsoluteTolerance,
        double outputRelativeTolerance,
        double gradientAbsoluteError = 0,
        double gradientRelativeError = 0,
        double gradientAbsoluteTolerance = 0,
        double gradientRelativeTolerance = 0)
    {
        if (!Enum.IsDefined(typeof(KernelTuningValidationScope), scope))
            throw new ArgumentOutOfRangeException(nameof(scope));
        ValidateError(outputAbsoluteError, nameof(outputAbsoluteError));
        ValidateError(outputRelativeError, nameof(outputRelativeError));
        ValidateError(outputAbsoluteTolerance, nameof(outputAbsoluteTolerance));
        ValidateError(outputRelativeTolerance, nameof(outputRelativeTolerance));
        if (outputAbsoluteError > outputAbsoluteTolerance && outputRelativeError > outputRelativeTolerance)
            throw new ArgumentException("Output errors exceed both configured tolerances.", nameof(outputAbsoluteError));

        ValidateError(gradientAbsoluteError, nameof(gradientAbsoluteError));
        ValidateError(gradientRelativeError, nameof(gradientRelativeError));
        ValidateError(gradientAbsoluteTolerance, nameof(gradientAbsoluteTolerance));
        ValidateError(gradientRelativeTolerance, nameof(gradientRelativeTolerance));
        if (scope == KernelTuningValidationScope.OutputAndGradient &&
            gradientAbsoluteError > gradientAbsoluteTolerance &&
            gradientRelativeError > gradientRelativeTolerance)
        {
            throw new ArgumentException("Gradient errors exceed both configured tolerances.", nameof(gradientAbsoluteError));
        }

        Scope = scope;
        OutputAbsoluteError = outputAbsoluteError;
        OutputRelativeError = outputRelativeError;
        OutputAbsoluteTolerance = outputAbsoluteTolerance;
        OutputRelativeTolerance = outputRelativeTolerance;
        GradientAbsoluteError = gradientAbsoluteError;
        GradientRelativeError = gradientRelativeError;
        GradientAbsoluteTolerance = gradientAbsoluteTolerance;
        GradientRelativeTolerance = gradientRelativeTolerance;
    }

    /// <summary>Gets the validation scope.</summary>
    public KernelTuningValidationScope Scope { get; }
    /// <summary>Gets maximum absolute output error.</summary>
    public double OutputAbsoluteError { get; }
    /// <summary>Gets maximum relative output error.</summary>
    public double OutputRelativeError { get; }
    /// <summary>Gets allowed absolute output error.</summary>
    public double OutputAbsoluteTolerance { get; }
    /// <summary>Gets allowed relative output error.</summary>
    public double OutputRelativeTolerance { get; }
    /// <summary>Gets maximum absolute gradient error.</summary>
    public double GradientAbsoluteError { get; }
    /// <summary>Gets maximum relative gradient error.</summary>
    public double GradientRelativeError { get; }
    /// <summary>Gets allowed absolute gradient error.</summary>
    public double GradientAbsoluteTolerance { get; }
    /// <summary>Gets allowed relative gradient error.</summary>
    public double GradientRelativeTolerance { get; }

    internal double MaximumError => Math.Max(
        Math.Max(OutputAbsoluteError, OutputRelativeError),
        Scope == KernelTuningValidationScope.OutputAndGradient
            ? Math.Max(GradientAbsoluteError, GradientRelativeError)
            : 0d);

    private static void ValidateError(double value, string parameterName)
    {
        if (!KernelTuningMeasurement.IsFinite(value) || value < 0)
            throw new ArgumentOutOfRangeException(parameterName);
    }
}

/// <summary>Device-resource measurements used for real quality-diversity descriptors.</summary>
public sealed class KernelTuningResourceUsage
{
    /// <summary>Creates validated resource measurements.</summary>
    public KernelTuningResourceUsage(
        long workspaceBytes,
        double occupancyRatio,
        int registersPerThread,
        TimeSpan compileTime,
        int kernelLaunchCount = 1)
        : this(
            KernelTuningResourceMetric<long>.Measured(workspaceBytes),
            KernelTuningResourceMetric<double>.Measured(occupancyRatio),
            KernelTuningResourceMetric<int>.Measured(registersPerThread),
            KernelTuningResourceMetric<TimeSpan>.Measured(compileTime),
            KernelTuningResourceMetric<int>.Measured(kernelLaunchCount))
    {
    }

    /// <summary>Creates resource evidence with an explicit evidence state per metric.</summary>
    public KernelTuningResourceUsage(
        KernelTuningResourceMetric<long> workspaceBytes,
        KernelTuningResourceMetric<double> occupancyRatio,
        KernelTuningResourceMetric<int> registersPerThread,
        KernelTuningResourceMetric<TimeSpan> compileTime,
        KernelTuningResourceMetric<int> kernelLaunchCount)
    {
        ValidateMetric(workspaceBytes, value => value >= 0, nameof(workspaceBytes));
        ValidateMetric(
            occupancyRatio,
            value => KernelTuningMeasurement.IsFinite(value) && value >= 0 && value <= 1,
            nameof(occupancyRatio));
        ValidateMetric(registersPerThread, value => value >= 0, nameof(registersPerThread));
        ValidateMetric(compileTime, value => value >= TimeSpan.Zero, nameof(compileTime));
        ValidateMetric(kernelLaunchCount, value => value > 0, nameof(kernelLaunchCount));
        WorkspaceBytesMetric = workspaceBytes;
        OccupancyRatioMetric = occupancyRatio;
        RegistersPerThreadMetric = registersPerThread;
        CompileTimeMetric = compileTime;
        KernelLaunchCountMetric = kernelLaunchCount;
    }

    /// <summary>Gets typed temporary-workspace evidence.</summary>
    public KernelTuningResourceMetric<long> WorkspaceBytesMetric { get; }
    /// <summary>Gets typed occupancy evidence.</summary>
    public KernelTuningResourceMetric<double> OccupancyRatioMetric { get; }
    /// <summary>Gets typed register evidence.</summary>
    public KernelTuningResourceMetric<int> RegistersPerThreadMetric { get; }
    /// <summary>Gets typed compile-time evidence.</summary>
    public KernelTuningResourceMetric<TimeSpan> CompileTimeMetric { get; }
    /// <summary>Gets typed launch-count evidence.</summary>
    public KernelTuningResourceMetric<int> KernelLaunchCountMetric { get; }

    /// <summary>Gets available temporary-workspace evidence in bytes.</summary>
    public long WorkspaceBytes => WorkspaceBytesMetric.Value;
    /// <summary>Gets available occupancy evidence in the inclusive range zero to one.</summary>
    public double OccupancyRatio => OccupancyRatioMetric.Value;
    /// <summary>Gets available register-use evidence per thread.</summary>
    public int RegistersPerThread => RegistersPerThreadMetric.Value;
    /// <summary>Gets available candidate compilation-latency evidence.</summary>
    public TimeSpan CompileTime => CompileTimeMetric.Value;
    /// <summary>Gets available operation kernel-launch-count evidence.</summary>
    public int KernelLaunchCount => KernelLaunchCountMetric.Value;

    /// <summary>Creates CPU evidence without inventing GPU occupancy or register values.</summary>
    public static KernelTuningResourceUsage ForCpu(
        long workspaceBytes,
        TimeSpan compileTime,
        int operationCount = 1) => new(
        KernelTuningResourceMetric<long>.Measured(workspaceBytes),
        KernelTuningResourceMetric<double>.NotApplicable(),
        KernelTuningResourceMetric<int>.NotApplicable(),
        KernelTuningResourceMetric<TimeSpan>.Measured(compileTime),
        KernelTuningResourceMetric<int>.Measured(operationCount));

    internal bool TryGetMetric(KernelTuningMetric metric, out double value)
    {
        value = 0;
        switch (metric)
        {
            case KernelTuningMetric.WorkspaceBytes:
                if (!WorkspaceBytesMetric.TryGetValue(out long workspace)) return false;
                value = workspace;
                return true;
            case KernelTuningMetric.Log2WorkspaceBytes:
                if (!WorkspaceBytesMetric.TryGetValue(out workspace)) return false;
                value = Math.Log(workspace + 1d, 2d);
                return true;
            case KernelTuningMetric.OccupancyRatio:
                return OccupancyRatioMetric.TryGetValue(out value);
            case KernelTuningMetric.RegistersPerThread:
                if (!RegistersPerThreadMetric.TryGetValue(out int registers)) return false;
                value = registers;
                return true;
            case KernelTuningMetric.CompileMilliseconds:
                if (!CompileTimeMetric.TryGetValue(out TimeSpan compile)) return false;
                value = compile.TotalMilliseconds;
                return true;
            case KernelTuningMetric.KernelLaunchCount:
                if (!KernelLaunchCountMetric.TryGetValue(out int launches)) return false;
                value = launches;
                return true;
            default:
                return false;
        }
    }

    private static void ValidateMetric<T>(
        KernelTuningResourceMetric<T> metric,
        Func<T, bool> predicate,
        string parameterName)
        where T : struct
    {
        if (metric.HasValue &&
            (!metric.TryGetValue(out T value) || !predicate(value)))
        {
            throw new ArgumentOutOfRangeException(parameterName);
        }
    }
}

/// <summary>One locally validated, repeatedly measured kernel result.</summary>
public sealed class KernelTuningMeasurement
{
    /// <summary>Creates a correctness-gated measurement.</summary>
    public KernelTuningMeasurement(
        KernelTuningWorkload workload,
        KernelTuningTimingScope timingScope,
        KernelTimingStatistics timing,
        KernelTuningResourceUsage resources,
        KernelTuningCorrectnessEvidence correctness)
    {
        if (!Enum.IsDefined(typeof(KernelTuningTimingScope), timingScope))
            throw new ArgumentOutOfRangeException(nameof(timingScope));
        Workload = workload;
        TimingScope = timingScope;
        Timing = timing ?? throw new ArgumentNullException(nameof(timing));
        Resources = resources ?? throw new ArgumentNullException(nameof(resources));
        Correctness = correctness ?? throw new ArgumentNullException(nameof(correctness));
        PerformanceRatePerSecond = workload.UnitsPerOperation / Timing.Median.TotalSeconds;
        if (!IsFinite(PerformanceRatePerSecond) || PerformanceRatePerSecond <= 0)
            throw new ArgumentOutOfRangeException(nameof(timing));
    }

    /// <summary>Gets the typed work represented by each timing sample.</summary>
    public KernelTuningWorkload Workload { get; }
    /// <summary>Gets the boundary included in each timing sample.</summary>
    public KernelTuningTimingScope TimingScope { get; }
    /// <summary>Gets the derived work rate used as the maximize-direction search quality.</summary>
    public double PerformanceRatePerSecond { get; }
    /// <summary>Gets billions of work units per second for the legacy cache transport.</summary>
    internal double BillionsOfWorkUnitsPerSecond => PerformanceRatePerSecond / 1e9d;
    /// <summary>Gets measured GFLOP/s when the declared work unit is floating-point operations.</summary>
    public double ThroughputGflops => Workload.Unit == KernelTuningWorkUnit.FloatingPointOperations
        ? BillionsOfWorkUnitsPerSecond
        : throw new InvalidOperationException("GFLOP/s is defined only for a floating-point-operation workload.");
    /// <summary>Gets robust timing statistics.</summary>
    public KernelTimingStatistics Timing { get; }
    /// <summary>Gets device-resource measurements.</summary>
    public KernelTuningResourceUsage Resources { get; }
    /// <summary>Gets local correctness evidence.</summary>
    public KernelTuningCorrectnessEvidence Correctness { get; }

    internal bool TryGetMetric(KernelTuningMetric metric, out double value)
    {
        switch (metric)
        {
            case KernelTuningMetric.ThroughputGflops:
                value = Workload.Unit == KernelTuningWorkUnit.FloatingPointOperations
                    ? ThroughputGflops
                    : 0;
                return Workload.Unit == KernelTuningWorkUnit.FloatingPointOperations;
            case KernelTuningMetric.MedianLatencyMilliseconds:
                value = Timing.Median.TotalMilliseconds;
                return true;
            case KernelTuningMetric.P95LatencyMilliseconds:
                value = Timing.P95.TotalMilliseconds;
                return true;
            case KernelTuningMetric.MaximumNumericalError:
                value = Correctness.MaximumError;
                return true;
            case KernelTuningMetric.Log10NumericalError:
                value = Math.Log10(Math.Max(Correctness.MaximumError, 1e-16d));
                return true;
            default:
                return Resources.TryGetMetric(metric, out value);
        }
    }

    internal double GetRequiredMetric(KernelTuningMetric metric) =>
        TryGetMetric(metric, out double value)
            ? value
            : throw new InvalidOperationException($"Kernel metric '{metric}' has no numeric evidence for this backend.");

    internal static bool IsFinite(double value) => !double.IsNaN(value) && !double.IsInfinity(value);
}

/// <summary>Typed trial result that prevents invalid kernels from entering the archive.</summary>
public sealed class KernelTuningTrialResult
{
    private KernelTuningTrialResult(
        KernelTuningTrialStatus status,
        KernelTuningMeasurement? measurement,
        string diagnostic)
    {
        Status = status;
        Measurement = measurement;
        Diagnostic = diagnostic;
    }

    /// <summary>Gets the typed terminal status.</summary>
    public KernelTuningTrialStatus Status { get; }
    /// <summary>Gets the measurement for a passed trial.</summary>
    public KernelTuningMeasurement? Measurement { get; }
    /// <summary>Gets a bounded human-readable diagnostic; programs branch on <see cref="Status"/>.</summary>
    public string Diagnostic { get; }

    /// <summary>Creates a passed result from mandatory correctness and timing evidence.</summary>
    public static KernelTuningTrialResult Passed(KernelTuningMeasurement measurement) =>
        new(KernelTuningTrialStatus.Passed, measurement ?? throw new ArgumentNullException(nameof(measurement)), string.Empty);

    /// <summary>Creates a typed rejected or failed result.</summary>
    public static KernelTuningTrialResult Rejected(KernelTuningTrialStatus status, string? diagnostic = null)
    {
        if (status == KernelTuningTrialStatus.Passed || !Enum.IsDefined(typeof(KernelTuningTrialStatus), status))
            throw new ArgumentOutOfRangeException(nameof(status));
        string bounded = diagnostic ?? string.Empty;
        if (bounded.Length > 4096) bounded = bounded.Substring(0, 4096);
        return new KernelTuningTrialResult(status, null, bounded);
    }
}

/// <summary>One typed archive axis backed by a measured resource or correctness property.</summary>
public sealed class KernelTuningDescriptorDefinition
{
    /// <summary>Creates a bounded descriptor axis.</summary>
    public KernelTuningDescriptorDefinition(
        KernelTuningMetric metric,
        double minimum,
        double maximum,
        int binCount,
        EvolutionOutOfRangePolicy outOfRangePolicy = EvolutionOutOfRangePolicy.Clamp)
    {
        if (!Enum.IsDefined(typeof(KernelTuningMetric), metric))
            throw new ArgumentOutOfRangeException(nameof(metric));
        _ = new EvolutionDescriptorDefinition(
            KernelTuningMetricNames.Get(metric), minimum, maximum, binCount, outOfRangePolicy);
        Metric = metric;
        Minimum = minimum;
        Maximum = maximum;
        BinCount = binCount;
        OutOfRangePolicy = outOfRangePolicy;
    }

    /// <summary>Gets the typed metric placed on this axis.</summary>
    public KernelTuningMetric Metric { get; }
    /// <summary>Gets the finite lower bound.</summary>
    public double Minimum { get; }
    /// <summary>Gets the finite upper bound.</summary>
    public double Maximum { get; }
    /// <summary>Gets the number of interior bins.</summary>
    public int BinCount { get; }
    /// <summary>Gets the out-of-range policy.</summary>
    public EvolutionOutOfRangePolicy OutOfRangePolicy { get; }

    internal EvolutionDescriptorDefinition ToEvolutionDefinition() => new(
        KernelTuningMetricNames.Get(Metric), Minimum, Maximum, BinCount, OutOfRangePolicy);
}

/// <summary>Selects the resource descriptor profile used by the typed tuning archive.</summary>
public enum KernelTuningArchiveProfile
{
    /// <summary>Selects the built-in profile from the tuning identity's device kind.</summary>
    DeviceDefault = 0,
    /// <summary>Uses CPU resource axes.</summary>
    Cpu = 1,
    /// <summary>
    /// Uses occupancy and register-allocation axes. Select this only when the backend reports
    /// numeric evidence for both metrics; portable GPU backends should use <see cref="PortableGpu"/>.
    /// </summary>
    Gpu = 2,
    /// <summary>Uses the explicitly supplied <see cref="KernelTuningOptions.ArchiveDescriptors"/>.</summary>
    Custom = 3,
    /// <summary>Uses only GPU metrics every backend can provide without inventing occupancy or registers.</summary>
    PortableGpu = 4,
}

/// <summary>Promotion and archive policy for typed kernel tuning.</summary>
public sealed class KernelTuningOptions
{
    private static readonly IReadOnlyList<KernelTuningDescriptorDefinition> DefaultGpuDescriptors =
        Array.AsReadOnly(new KernelTuningDescriptorDefinition[]
    {
        new(KernelTuningMetric.Log2WorkspaceBytes, 0, 40, 16),
        new(KernelTuningMetric.OccupancyRatio, 0, 1, 10),
        new(KernelTuningMetric.RegistersPerThread, 0, 512, 16)
    });

    private static readonly IReadOnlyList<KernelTuningDescriptorDefinition> DefaultCpuDescriptors =
        Array.AsReadOnly(new KernelTuningDescriptorDefinition[]
    {
        new(KernelTuningMetric.Log2WorkspaceBytes, 0, 40, 16),
        new(KernelTuningMetric.KernelLaunchCount, 1, 256, 16)
    });

    private static readonly IReadOnlyList<KernelTuningDescriptorDefinition> PortableGpuDescriptors =
        Array.AsReadOnly(new KernelTuningDescriptorDefinition[]
    {
        new(KernelTuningMetric.Log2WorkspaceBytes, 0, 40, 16),
        new(KernelTuningMetric.Log10NumericalError, -16, 0, 16),
        new(KernelTuningMetric.KernelLaunchCount, 1, 256, 16)
    });

    /// <summary>Gets or sets the minimum throughput ratio required to replace an active winner.</summary>
    public double MinimumPromotionRatio { get; set; } = GpuFirstRunAutotuner.MinimumPromotionRatio;

    /// <summary>Gets or sets the largest candidate/incumbent P95 latency ratio accepted for promotion.</summary>
    public double MaximumP95LatencyRatio { get; set; } = 1d;

    /// <summary>Gets or sets how the archive resource axes are selected.</summary>
    public KernelTuningArchiveProfile ArchiveProfile { get; set; } = KernelTuningArchiveProfile.DeviceDefault;

    /// <summary>
    /// Gets or sets the real resource/correctness axes used when <see cref="ArchiveProfile"/> is
    /// <see cref="KernelTuningArchiveProfile.Custom"/>.
    /// </summary>
    public IReadOnlyList<KernelTuningDescriptorDefinition> ArchiveDescriptors { get; set; } =
        Array.Empty<KernelTuningDescriptorDefinition>();

    internal KernelTuningOptions SnapshotAndValidate(KernelTuningDeviceKind deviceKind)
    {
        if (!KernelTuningMeasurement.IsFinite(MinimumPromotionRatio) || MinimumPromotionRatio < 1d)
            throw new ArgumentOutOfRangeException(nameof(MinimumPromotionRatio));
        if (!KernelTuningMeasurement.IsFinite(MaximumP95LatencyRatio) || MaximumP95LatencyRatio <= 0)
            throw new ArgumentOutOfRangeException(nameof(MaximumP95LatencyRatio));
        if (!Enum.IsDefined(typeof(KernelTuningArchiveProfile), ArchiveProfile))
            throw new ArgumentOutOfRangeException(nameof(ArchiveProfile));
        if (!Enum.IsDefined(typeof(KernelTuningDeviceKind), deviceKind))
            throw new ArgumentOutOfRangeException(nameof(deviceKind));
        if (ArchiveDescriptors is null)
            throw new ArgumentNullException(nameof(ArchiveDescriptors));

        IReadOnlyList<KernelTuningDescriptorDefinition> resolved = ArchiveProfile switch
        {
            KernelTuningArchiveProfile.DeviceDefault => deviceKind == KernelTuningDeviceKind.Cpu
                ? DefaultCpuDescriptors
                : PortableGpuDescriptors,
            KernelTuningArchiveProfile.Cpu => DefaultCpuDescriptors,
            KernelTuningArchiveProfile.Gpu => DefaultGpuDescriptors,
            KernelTuningArchiveProfile.Custom => ArchiveDescriptors,
            KernelTuningArchiveProfile.PortableGpu => PortableGpuDescriptors,
            _ => throw new ArgumentOutOfRangeException(nameof(ArchiveProfile))
        };
        if (ArchiveProfile != KernelTuningArchiveProfile.Custom && ArchiveDescriptors.Count != 0)
        {
            throw new ArgumentException(
                "Explicit archive descriptors require the Custom archive profile.",
                nameof(ArchiveDescriptors));
        }
        if (resolved.Count == 0)
            throw new ArgumentException("At least one archive descriptor is required.", nameof(ArchiveDescriptors));
        var seen = new HashSet<KernelTuningMetric>();
        KernelTuningDescriptorDefinition[] copy = resolved.ToArray();
        for (int i = 0; i < copy.Length; i++)
        {
            if (copy[i] is null) throw new ArgumentException("Archive descriptors cannot contain null.", nameof(ArchiveDescriptors));
            if (!seen.Add(copy[i].Metric))
                throw new ArgumentException("Archive descriptor metrics must be unique.", nameof(ArchiveDescriptors));
        }
        return new KernelTuningOptions
        {
            MinimumPromotionRatio = MinimumPromotionRatio,
            MaximumP95LatencyRatio = MaximumP95LatencyRatio,
            ArchiveProfile = ArchiveProfile == KernelTuningArchiveProfile.DeviceDefault
                ? deviceKind == KernelTuningDeviceKind.Cpu
                    ? KernelTuningArchiveProfile.Cpu
                    : KernelTuningArchiveProfile.PortableGpu
                : ArchiveProfile,
            ArchiveDescriptors = Array.AsReadOnly(copy)
        };
    }

    internal bool QualifiesForPromotion(KernelTuningPairedEvidence evidence)
    {
        if (evidence is null) throw new ArgumentNullException(nameof(evidence));
        double requiredSpeedup = Math.Max(MinimumPromotionRatio, evidence.CalibratedNoiseRatio);
        return evidence.MedianSpeedup >= requiredSpeedup &&
               evidence.LowerSpeedupBound >= 1d &&
               evidence.P95LatencyRatio <= MaximumP95LatencyRatio;
    }
}

internal static class KernelTuningMetricNames
{
    internal static string Get(KernelTuningMetric metric) => metric switch
    {
        KernelTuningMetric.ThroughputGflops => "throughput-gflops",
        KernelTuningMetric.MedianLatencyMilliseconds => "median-latency-milliseconds",
        KernelTuningMetric.P95LatencyMilliseconds => "p95-latency-milliseconds",
        KernelTuningMetric.WorkspaceBytes => "workspace-bytes",
        KernelTuningMetric.Log2WorkspaceBytes => "log2-workspace-bytes",
        KernelTuningMetric.OccupancyRatio => "occupancy-ratio",
        KernelTuningMetric.RegistersPerThread => "registers-per-thread",
        KernelTuningMetric.CompileMilliseconds => "compile-milliseconds",
        KernelTuningMetric.MaximumNumericalError => "maximum-numerical-error",
        KernelTuningMetric.Log10NumericalError => "log10-numerical-error",
        KernelTuningMetric.KernelLaunchCount => "kernel-launch-count",
        _ => throw new ArgumentOutOfRangeException(nameof(metric))
    };
}
