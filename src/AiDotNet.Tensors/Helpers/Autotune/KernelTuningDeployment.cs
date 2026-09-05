using System.Collections.Concurrent;
using System.Globalization;
using AiDotNet.Evolution;

namespace AiDotNet.Tensors.Helpers.Autotune;

/// <summary>An immutable, locally validated configuration activated for production dispatch.</summary>
public sealed class KernelTuningDeploymentSnapshot<TConfiguration>
    where TConfiguration : notnull
{
    /// <summary>Creates a validated snapshot for a custom tuning store or deployment workflow.</summary>
    public KernelTuningDeploymentSnapshot(
        KernelTuningIdentity identity,
        TConfiguration configuration,
        string genomeId,
        KernelTuningMeasurement measurement,
        string runStateHash)
    {
        Identity = identity ?? throw new ArgumentNullException(nameof(identity));
        Configuration = configuration is null
            ? throw new ArgumentNullException(nameof(configuration))
            : configuration;
        GenomeId = string.IsNullOrWhiteSpace(genomeId)
            ? throw new ArgumentException("A canonical genome id is required.", nameof(genomeId))
            : genomeId;
        Measurement = measurement ?? throw new ArgumentNullException(nameof(measurement));
        RunStateHash = string.IsNullOrWhiteSpace(runStateHash)
            ? throw new ArgumentException("A source run-state hash is required.", nameof(runStateHash))
            : runStateHash;
    }

    /// <summary>Gets the exact kernel, device, shape, and protocol identity.</summary>
    public KernelTuningIdentity Identity { get; }
    /// <summary>Gets the typed launch configuration used by dispatch.</summary>
    public TConfiguration Configuration { get; }
    /// <summary>Gets the canonical configuration identity.</summary>
    public string GenomeId { get; }
    /// <summary>Gets the correctness-gated measurement that justified deployment.</summary>
    public KernelTuningMeasurement Measurement { get; }
    /// <summary>Gets the deterministic state hash of the selecting run, or the persisted source hash.</summary>
    public string RunStateHash { get; }
}

/// <summary>Lock-free deployment handle for one pre-resolved tuning identity.</summary>
public sealed class KernelTuningDeployment<TConfiguration>
    where TConfiguration : notnull
{
    private readonly string _identityKey;
    private KernelTuningDeploymentSnapshot<TConfiguration>? _current;

    internal KernelTuningDeployment(KernelTuningIdentity identity)
    {
        _identityKey = (identity ?? throw new ArgumentNullException(nameof(identity))).StableKey;
    }

    /// <summary>Gets the current immutable snapshot through one volatile reference read.</summary>
    public KernelTuningDeploymentSnapshot<TConfiguration>? Current => Volatile.Read(ref _current);

    /// <summary>Reads the active typed configuration without file I/O, parsing, reflection, or evolution.</summary>
    public bool TryGet(out TConfiguration? configuration)
    {
        KernelTuningDeploymentSnapshot<TConfiguration>? snapshot = Volatile.Read(ref _current);
        if (snapshot is null)
        {
            configuration = default;
            return false;
        }
        configuration = snapshot.Configuration;
        return true;
    }

    internal void Publish(KernelTuningDeploymentSnapshot<TConfiguration> snapshot)
    {
        ValidateIdentity(snapshot);
        Volatile.Write(ref _current, snapshot);
    }

    internal bool TryPublishIfEmpty(KernelTuningDeploymentSnapshot<TConfiguration> snapshot)
    {
        ValidateIdentity(snapshot);
        return Interlocked.CompareExchange(ref _current, snapshot, null) is null;
    }

    private void ValidateIdentity(KernelTuningDeploymentSnapshot<TConfiguration> snapshot)
    {
        if (snapshot is null) throw new ArgumentNullException(nameof(snapshot));
        if (!string.Equals(snapshot.Identity.StableKey, _identityKey, StringComparison.Ordinal))
            throw new InvalidOperationException("A deployment handle cannot publish a different tuning identity.");
    }
}

/// <summary>
/// Resolves keyed deployment handles outside dispatch; each returned handle serves through one volatile read.
/// </summary>
public sealed class KernelTuningDeploymentRegistry<TConfiguration>
    where TConfiguration : notnull
{
    private readonly ConcurrentDictionary<string, KernelTuningDeployment<TConfiguration>> _deployments =
        new(StringComparer.Ordinal);

    /// <summary>Gets or creates the stable handle for an identity.</summary>
    public KernelTuningDeployment<TConfiguration> GetOrCreate(KernelTuningIdentity identity)
    {
        if (identity is null) throw new ArgumentNullException(nameof(identity));
        return _deployments.GetOrAdd(identity.StableKey, _ => new KernelTuningDeployment<TConfiguration>(identity));
    }
}

/// <summary>Typed persistence seam for deployed tuning winners.</summary>
public interface IKernelTuningStore<TConfiguration>
    where TConfiguration : notnull
{
    /// <summary>Loads and validates a winner for the exact identity.</summary>
    bool TryLoad(
        KernelTuningIdentity identity,
        IEvolutionGenomeCodec<TConfiguration> codec,
        out KernelTuningDeploymentSnapshot<TConfiguration>? snapshot);

    /// <summary>Persists an immutable winner. Cache failure must not invalidate the in-memory winner.</summary>
    bool TryStore(
        KernelTuningDeploymentSnapshot<TConfiguration> snapshot,
        IEvolutionGenomeCodec<TConfiguration> codec);
}

/// <summary>Adapter that confines the legacy string cache schema to a serialization boundary.</summary>
public sealed class AutotuneCacheKernelTuningStore<TConfiguration> : IKernelTuningStore<TConfiguration>
    where TConfiguration : notnull
{
    private const string Variant = "typed-evolution-v1";
    private const string IdentityKey = "identity";
    private const string CodecIdKey = "codec-id";
    private const string CodecVersionKey = "codec-version";
    private const string GenomePayloadKey = "genome-payload";
    private const string GenomeIdKey = "genome-id";
    private const string RunStateHashKey = "run-state-hash";
    private const string SampleCountKey = "sample-count";
    private const string P95MillisecondsKey = "p95-milliseconds";
    private const string WorkspaceBytesKey = "workspace-bytes";
    private const string OccupancyRatioKey = "occupancy-ratio";
    private const string RegistersPerThreadKey = "registers-per-thread";
    private const string CompileMillisecondsKey = "compile-milliseconds";
    private const string KernelLaunchCountKey = "kernel-launch-count";
    private const string ValidationScopeKey = "validation-scope";
    private const string OutputAbsoluteErrorKey = "output-absolute-error";
    private const string OutputRelativeErrorKey = "output-relative-error";
    private const string OutputAbsoluteToleranceKey = "output-absolute-tolerance";
    private const string OutputRelativeToleranceKey = "output-relative-tolerance";
    private const string GradientAbsoluteErrorKey = "gradient-absolute-error";
    private const string GradientRelativeErrorKey = "gradient-relative-error";
    private const string GradientAbsoluteToleranceKey = "gradient-absolute-tolerance";
    private const string GradientRelativeToleranceKey = "gradient-relative-tolerance";

    /// <inheritdoc />
    public bool TryLoad(
        KernelTuningIdentity identity,
        IEvolutionGenomeCodec<TConfiguration> codec,
        out KernelTuningDeploymentSnapshot<TConfiguration>? snapshot)
    {
        snapshot = null;
        if (identity is null || codec is null) return false;
        try
        {
            KernelChoice? choice = AutotuneCache.Lookup(CacheKernel(identity), identity.Shape);
            if (choice is null || !string.Equals(choice.Variant, Variant, StringComparison.Ordinal) ||
                choice.Parameters is null ||
                !TryGet(choice.Parameters, IdentityKey, out string persistedIdentity) ||
                !string.Equals(persistedIdentity, identity.StableKey, StringComparison.Ordinal) ||
                !TryGet(choice.Parameters, CodecIdKey, out string codecId) ||
                !string.Equals(codecId, codec.Id, StringComparison.Ordinal) ||
                !TryGet(choice.Parameters, CodecVersionKey, out string codecVersion) ||
                !string.Equals(codecVersion, codec.VersionHash, StringComparison.Ordinal) ||
                !TryGet(choice.Parameters, GenomePayloadKey, out string payload) ||
                !TryGet(choice.Parameters, GenomeIdKey, out string genomeId) ||
                !TryGet(choice.Parameters, RunStateHashKey, out string runStateHash) ||
                !TryInt(choice.Parameters, SampleCountKey, out int sampleCount) ||
                !TryDouble(choice.Parameters, P95MillisecondsKey, out double p95Ms) ||
                !TryLong(choice.Parameters, WorkspaceBytesKey, out long workspaceBytes) ||
                !TryDouble(choice.Parameters, OccupancyRatioKey, out double occupancy) ||
                !TryInt(choice.Parameters, RegistersPerThreadKey, out int registers) ||
                !TryDouble(choice.Parameters, CompileMillisecondsKey, out double compileMs) ||
                !TryInt(choice.Parameters, KernelLaunchCountKey, out int launches) ||
                !TryInt(choice.Parameters, ValidationScopeKey, out int scopeValue) ||
                !TryDouble(choice.Parameters, OutputAbsoluteErrorKey, out double outputAbs) ||
                !TryDouble(choice.Parameters, OutputRelativeErrorKey, out double outputRel) ||
                !TryDouble(choice.Parameters, OutputAbsoluteToleranceKey, out double outputAbsTol) ||
                !TryDouble(choice.Parameters, OutputRelativeToleranceKey, out double outputRelTol) ||
                !TryDouble(choice.Parameters, GradientAbsoluteErrorKey, out double gradientAbs) ||
                !TryDouble(choice.Parameters, GradientRelativeErrorKey, out double gradientRel) ||
                !TryDouble(choice.Parameters, GradientAbsoluteToleranceKey, out double gradientAbsTol) ||
                !TryDouble(choice.Parameters, GradientRelativeToleranceKey, out double gradientRelTol) ||
                !KernelTuningMeasurement.IsFinite(choice.MeasuredGflops) || choice.MeasuredGflops <= 0 ||
                !KernelTuningMeasurement.IsFinite(choice.MeasuredTimeMs) || choice.MeasuredTimeMs <= 0)
            {
                return false;
            }

            TConfiguration configuration = codec.Deserialize(payload);
            if (configuration is null) return false;
            string? canonicalPayload = codec.Serialize(configuration);
            if (canonicalPayload is null ||
                !string.Equals(canonicalPayload, payload, StringComparison.Ordinal) ||
                !string.Equals(EvolutionHash.Compute(payload), genomeId, StringComparison.Ordinal))
            {
                return false;
            }

            var timing = KernelTimingStatistics.FromSummary(sampleCount, choice.MeasuredTimeMs, p95Ms);
            var resources = new KernelTuningResourceUsage(
                workspaceBytes, occupancy, registers, TimeSpan.FromMilliseconds(compileMs), launches);
            var correctness = new KernelTuningCorrectnessEvidence(
                (KernelTuningValidationScope)scopeValue,
                outputAbs, outputRel, outputAbsTol, outputRelTol,
                gradientAbs, gradientRel, gradientAbsTol, gradientRelTol);
            var measurement = new KernelTuningMeasurement(choice.MeasuredGflops, timing, resources, correctness);
            snapshot = new KernelTuningDeploymentSnapshot<TConfiguration>(
                identity, configuration, genomeId, measurement, runStateHash);
            return true;
        }
        catch
        {
            snapshot = null;
            return false;
        }
    }

    /// <inheritdoc />
    public bool TryStore(
        KernelTuningDeploymentSnapshot<TConfiguration> snapshot,
        IEvolutionGenomeCodec<TConfiguration> codec)
    {
        if (snapshot is null) throw new ArgumentNullException(nameof(snapshot));
        if (codec is null) throw new ArgumentNullException(nameof(codec));
        KernelTuningMeasurement measurement = snapshot.Measurement;
        KernelTuningCorrectnessEvidence correctness = measurement.Correctness;
        KernelTuningResourceUsage resources = measurement.Resources;
        string? genomePayload = codec.Serialize(snapshot.Configuration);
        if (genomePayload is null)
            throw new InvalidOperationException("The kernel configuration codec returned a null payload.");
        var parameters = new Dictionary<string, string>(StringComparer.Ordinal)
        {
            [IdentityKey] = snapshot.Identity.StableKey,
            [CodecIdKey] = codec.Id,
            [CodecVersionKey] = codec.VersionHash,
            [GenomePayloadKey] = genomePayload,
            [GenomeIdKey] = snapshot.GenomeId,
            [RunStateHashKey] = snapshot.RunStateHash,
            [SampleCountKey] = Format(measurement.Timing.SampleCount),
            [P95MillisecondsKey] = Format(measurement.Timing.P95.TotalMilliseconds),
            [WorkspaceBytesKey] = Format(resources.WorkspaceBytes),
            [OccupancyRatioKey] = Format(resources.OccupancyRatio),
            [RegistersPerThreadKey] = Format(resources.RegistersPerThread),
            [CompileMillisecondsKey] = Format(resources.CompileTime.TotalMilliseconds),
            [KernelLaunchCountKey] = Format(resources.KernelLaunchCount),
            [ValidationScopeKey] = Format((int)correctness.Scope),
            [OutputAbsoluteErrorKey] = Format(correctness.OutputAbsoluteError),
            [OutputRelativeErrorKey] = Format(correctness.OutputRelativeError),
            [OutputAbsoluteToleranceKey] = Format(correctness.OutputAbsoluteTolerance),
            [OutputRelativeToleranceKey] = Format(correctness.OutputRelativeTolerance),
            [GradientAbsoluteErrorKey] = Format(correctness.GradientAbsoluteError),
            [GradientRelativeErrorKey] = Format(correctness.GradientRelativeError),
            [GradientAbsoluteToleranceKey] = Format(correctness.GradientAbsoluteTolerance),
            [GradientRelativeToleranceKey] = Format(correctness.GradientRelativeTolerance)
        };
        if (!string.Equals(EvolutionHash.Compute(parameters[GenomePayloadKey]), snapshot.GenomeId, StringComparison.Ordinal))
            throw new InvalidOperationException("The snapshot genome id does not match its canonical payload.");

        return AutotuneCache.TryStore(
            CacheKernel(snapshot.Identity),
            snapshot.Identity.Shape,
            new KernelChoice
            {
                Variant = Variant,
                Parameters = parameters,
                MeasuredGflops = measurement.ThroughputGflops,
                MeasuredTimeMs = measurement.Timing.Median.TotalMilliseconds
            });
    }

    private static KernelId CacheKernel(KernelTuningIdentity identity) => new(
        identity.Kernel.Category,
        string.Concat("typed-evolution-", identity.StableKey));

    private static bool TryGet(IReadOnlyDictionary<string, string> values, string key, out string value)
    {
        if (values.TryGetValue(key, out string? candidate) && !string.IsNullOrWhiteSpace(candidate))
        {
            value = candidate;
            return true;
        }

        value = string.Empty;
        return false;
    }

    private static bool TryInt(IReadOnlyDictionary<string, string> values, string key, out int value)
    {
        value = default;
        return values.TryGetValue(key, out string? raw) &&
               int.TryParse(raw, NumberStyles.Integer, CultureInfo.InvariantCulture, out value);
    }

    private static bool TryLong(IReadOnlyDictionary<string, string> values, string key, out long value)
    {
        value = default;
        return values.TryGetValue(key, out string? raw) &&
               long.TryParse(raw, NumberStyles.Integer, CultureInfo.InvariantCulture, out value);
    }

    private static bool TryDouble(IReadOnlyDictionary<string, string> values, string key, out double value)
    {
        value = default;
        return values.TryGetValue(key, out string? raw) &&
               double.TryParse(raw, NumberStyles.Float, CultureInfo.InvariantCulture, out value);
    }

    private static string Format(int value) => value.ToString(CultureInfo.InvariantCulture);
    private static string Format(long value) => value.ToString(CultureInfo.InvariantCulture);
    private static string Format(double value) => value.ToString("R", CultureInfo.InvariantCulture);
}

internal static class KernelTuningCoordinator
{
    private static readonly ConcurrentDictionary<string, SemaphoreSlim> Gates = new(StringComparer.Ordinal);

    internal static async ValueTask<IDisposable> EnterAsync(
        KernelTuningDeviceFingerprint device,
        CancellationToken cancellationToken)
    {
        SemaphoreSlim gate = Gates.GetOrAdd(device.LocalKey, _ => new SemaphoreSlim(1, 1));
        await gate.WaitAsync(cancellationToken).ConfigureAwait(false);
        return new Lease(gate);
    }

    private sealed class Lease : IDisposable
    {
        private SemaphoreSlim? _gate;
        internal Lease(SemaphoreSlim gate) => _gate = gate;
        public void Dispose() => Interlocked.Exchange(ref _gate, null)?.Release();
    }
}

/// <summary>Admission gate required before background tuning may consume device capacity.</summary>
public interface IKernelTuningIdleGate
{
    /// <summary>Waits until the target device is safe for background tuning.</summary>
    ValueTask WaitUntilIdleAsync(KernelTuningIdentity identity, CancellationToken cancellationToken = default);
}
