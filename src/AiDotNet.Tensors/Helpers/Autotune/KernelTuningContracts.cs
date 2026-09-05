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
        KernelSearchSpaceVersion searchSpaceVersion,
        KernelBenchmarkProtocolVersion benchmarkProtocolVersion)
    {
        if (string.IsNullOrWhiteSpace(kernel.Category))
            throw new ArgumentException("A kernel category is required.", nameof(kernel));
        if (string.IsNullOrWhiteSpace(kernel.Name))
            throw new ArgumentException("A kernel name is required.", nameof(kernel));
        if (string.IsNullOrWhiteSpace(device.LocalKey))
            throw new ArgumentException("A valid device fingerprint is required.", nameof(device));
        if (searchSpaceVersion.Value <= 0) throw new ArgumentOutOfRangeException(nameof(searchSpaceVersion));
        if (benchmarkProtocolVersion.Value <= 0)
            throw new ArgumentOutOfRangeException(nameof(benchmarkProtocolVersion));

        Kernel = kernel;
        Shape = shape is null
            ? throw new ArgumentNullException(nameof(shape))
            : new ShapeProfile(shape.Dimensions);
        Device = device;
        SearchSpaceVersion = searchSpaceVersion;
        BenchmarkProtocolVersion = benchmarkProtocolVersion;
        StableKey = EvolutionHash.Combine(new[]
        {
            "tensor-kernel-tuning-identity-v2",
            Kernel.ToFileStem(),
            Shape.ToFileStem(),
            Device.LocalKey,
            SearchSpaceVersion.ToString(),
            BenchmarkProtocolVersion.ToString()
        });
    }

    /// <summary>Creates an identity for a GPU without exposing stringly device keys.</summary>
    public KernelTuningIdentity(
        KernelId kernel,
        ShapeProfile shape,
        GpuDeviceFingerprint device,
        KernelSearchSpaceVersion searchSpaceVersion,
        KernelBenchmarkProtocolVersion benchmarkProtocolVersion)
        : this(
            kernel,
            shape,
            KernelTuningDeviceFingerprint.FromGpu(device),
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
    BenchmarkFailed = 6
}

/// <summary>Robust latency statistics computed from repeated device measurements.</summary>
public sealed class KernelTimingStatistics
{
    /// <summary>Minimum number of measured samples accepted by the tuning infrastructure.</summary>
    public const int MinimumSampleCount = 3;

    private KernelTimingStatistics(int sampleCount, TimeSpan median, TimeSpan p95)
    {
        SampleCount = sampleCount;
        Median = median;
        P95 = p95;
    }

    /// <summary>Gets the number of post-warmup samples.</summary>
    public int SampleCount { get; }
    /// <summary>Gets the median latency.</summary>
    public TimeSpan Median { get; }
    /// <summary>Gets the nearest-rank 95th-percentile latency.</summary>
    public TimeSpan P95 { get; }

    /// <summary>Computes immutable statistics from post-warmup samples.</summary>
    public static KernelTimingStatistics FromSamples(IEnumerable<TimeSpan> samples)
    {
        if (samples is null) throw new ArgumentNullException(nameof(samples));
        double[] milliseconds = samples.Select(sample => sample.TotalMilliseconds).ToArray();
        if (milliseconds.Length < MinimumSampleCount)
            throw new ArgumentException(
                $"At least {MinimumSampleCount} post-warmup measurements are required.", nameof(samples));
        for (int i = 0; i < milliseconds.Length; i++)
        {
            if (!KernelTuningMeasurement.IsFinite(milliseconds[i]) || milliseconds[i] <= 0)
                throw new ArgumentOutOfRangeException(nameof(samples), "Every timing sample must be finite and positive.");
        }

        Array.Sort(milliseconds);
        double median = milliseconds.Length % 2 == 0
            ? (milliseconds[milliseconds.Length / 2 - 1] + milliseconds[milliseconds.Length / 2]) / 2d
            : milliseconds[milliseconds.Length / 2];
        int p95Index = Math.Max(0, (int)Math.Ceiling(milliseconds.Length * 0.95d) - 1);
        return new KernelTimingStatistics(
            milliseconds.Length,
            TimeSpan.FromMilliseconds(median),
            TimeSpan.FromMilliseconds(milliseconds[p95Index]));
    }

    internal static KernelTimingStatistics FromSummary(int sampleCount, double medianMs, double p95Ms)
    {
        if (sampleCount < MinimumSampleCount) throw new ArgumentOutOfRangeException(nameof(sampleCount));
        if (!KernelTuningMeasurement.IsFinite(medianMs) || medianMs <= 0)
            throw new ArgumentOutOfRangeException(nameof(medianMs));
        if (!KernelTuningMeasurement.IsFinite(p95Ms) || p95Ms < medianMs)
            throw new ArgumentOutOfRangeException(nameof(p95Ms));
        return new KernelTimingStatistics(
            sampleCount, TimeSpan.FromMilliseconds(medianMs), TimeSpan.FromMilliseconds(p95Ms));
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
    {
        if (workspaceBytes < 0) throw new ArgumentOutOfRangeException(nameof(workspaceBytes));
        if (!KernelTuningMeasurement.IsFinite(occupancyRatio) || occupancyRatio < 0 || occupancyRatio > 1)
            throw new ArgumentOutOfRangeException(nameof(occupancyRatio));
        if (registersPerThread < 0) throw new ArgumentOutOfRangeException(nameof(registersPerThread));
        if (compileTime < TimeSpan.Zero) throw new ArgumentOutOfRangeException(nameof(compileTime));
        if (kernelLaunchCount <= 0) throw new ArgumentOutOfRangeException(nameof(kernelLaunchCount));
        WorkspaceBytes = workspaceBytes;
        OccupancyRatio = occupancyRatio;
        RegistersPerThread = registersPerThread;
        CompileTime = compileTime;
        KernelLaunchCount = kernelLaunchCount;
    }

    /// <summary>Gets temporary workspace in bytes.</summary>
    public long WorkspaceBytes { get; }
    /// <summary>Gets occupancy in the inclusive range zero to one.</summary>
    public double OccupancyRatio { get; }
    /// <summary>Gets registers consumed per thread.</summary>
    public int RegistersPerThread { get; }
    /// <summary>Gets candidate compilation latency.</summary>
    public TimeSpan CompileTime { get; }
    /// <summary>Gets the operation's kernel-launch count.</summary>
    public int KernelLaunchCount { get; }
}

/// <summary>One locally validated, repeatedly measured kernel result.</summary>
public sealed class KernelTuningMeasurement
{
    /// <summary>Creates a correctness-gated measurement.</summary>
    public KernelTuningMeasurement(
        double throughputGflops,
        KernelTimingStatistics timing,
        KernelTuningResourceUsage resources,
        KernelTuningCorrectnessEvidence correctness)
    {
        if (!IsFinite(throughputGflops) || throughputGflops <= 0)
            throw new ArgumentOutOfRangeException(nameof(throughputGflops));
        ThroughputGflops = throughputGflops;
        Timing = timing ?? throw new ArgumentNullException(nameof(timing));
        Resources = resources ?? throw new ArgumentNullException(nameof(resources));
        Correctness = correctness ?? throw new ArgumentNullException(nameof(correctness));
    }

    /// <summary>Gets measured throughput.</summary>
    public double ThroughputGflops { get; }
    /// <summary>Gets robust timing statistics.</summary>
    public KernelTimingStatistics Timing { get; }
    /// <summary>Gets device-resource measurements.</summary>
    public KernelTuningResourceUsage Resources { get; }
    /// <summary>Gets local correctness evidence.</summary>
    public KernelTuningCorrectnessEvidence Correctness { get; }

    internal double GetMetric(KernelTuningMetric metric) => metric switch
    {
        KernelTuningMetric.ThroughputGflops => ThroughputGflops,
        KernelTuningMetric.MedianLatencyMilliseconds => Timing.Median.TotalMilliseconds,
        KernelTuningMetric.P95LatencyMilliseconds => Timing.P95.TotalMilliseconds,
        KernelTuningMetric.WorkspaceBytes => Resources.WorkspaceBytes,
        KernelTuningMetric.Log2WorkspaceBytes => Math.Log(Resources.WorkspaceBytes + 1d, 2d),
        KernelTuningMetric.OccupancyRatio => Resources.OccupancyRatio,
        KernelTuningMetric.RegistersPerThread => Resources.RegistersPerThread,
        KernelTuningMetric.CompileMilliseconds => Resources.CompileTime.TotalMilliseconds,
        KernelTuningMetric.MaximumNumericalError => Correctness.MaximumError,
        KernelTuningMetric.Log10NumericalError => Math.Log10(Math.Max(Correctness.MaximumError, 1e-16d)),
        KernelTuningMetric.KernelLaunchCount => Resources.KernelLaunchCount,
        _ => throw new ArgumentOutOfRangeException(nameof(metric))
    };

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

/// <summary>Promotion and archive policy for typed kernel tuning.</summary>
public sealed class KernelTuningOptions
{
    private static readonly IReadOnlyList<KernelTuningDescriptorDefinition> DefaultDescriptors =
        Array.AsReadOnly(new KernelTuningDescriptorDefinition[]
    {
        new(KernelTuningMetric.Log2WorkspaceBytes, 0, 40, 16),
        new(KernelTuningMetric.OccupancyRatio, 0, 1, 10),
        new(KernelTuningMetric.RegistersPerThread, 0, 512, 16)
    });

    /// <summary>Gets or sets the minimum throughput ratio required to replace an active winner.</summary>
    public double MinimumPromotionRatio { get; set; } = GpuFirstRunAutotuner.MinimumPromotionRatio;

    /// <summary>Gets or sets the real resource/correctness axes used by MAP-Elites.</summary>
    public IReadOnlyList<KernelTuningDescriptorDefinition> ArchiveDescriptors { get; set; } = DefaultDescriptors;

    internal KernelTuningOptions SnapshotAndValidate()
    {
        if (!KernelTuningMeasurement.IsFinite(MinimumPromotionRatio) || MinimumPromotionRatio < 1d)
            throw new ArgumentOutOfRangeException(nameof(MinimumPromotionRatio));
        if (ArchiveDescriptors is null || ArchiveDescriptors.Count == 0)
            throw new ArgumentException("At least one archive descriptor is required.", nameof(ArchiveDescriptors));
        var seen = new HashSet<KernelTuningMetric>();
        KernelTuningDescriptorDefinition[] copy = ArchiveDescriptors.ToArray();
        for (int i = 0; i < copy.Length; i++)
        {
            if (copy[i] is null) throw new ArgumentException("Archive descriptors cannot contain null.", nameof(ArchiveDescriptors));
            if (!seen.Add(copy[i].Metric))
                throw new ArgumentException("Archive descriptor metrics must be unique.", nameof(ArchiveDescriptors));
        }
        return new KernelTuningOptions
        {
            MinimumPromotionRatio = MinimumPromotionRatio,
            ArchiveDescriptors = Array.AsReadOnly(copy)
        };
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
