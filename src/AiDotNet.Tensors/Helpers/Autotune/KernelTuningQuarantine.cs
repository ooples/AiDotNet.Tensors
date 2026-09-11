using System.Collections.Concurrent;
using System.Text;
using System.Text.Json;
using AiDotNet.Evolution;

namespace AiDotNet.Tensors.Helpers.Autotune;

/// <summary>The runtime regression reported by the caller's monitoring policy.</summary>
public enum KernelTuningRegressionReason
{
    /// <summary>A measured latency statistic exceeded its limit.</summary>
    Latency = 0,
    /// <summary>A measured correctness error exceeded its tolerance.</summary>
    Correctness = 1,
    /// <summary>A measured resource statistic exceeded its limit.</summary>
    ResourceUsage = 2
}

/// <summary>Bounded caller-supplied regression evidence; this object does not run a monitor.</summary>
public sealed class KernelTuningRegressionEvidence
{
    /// <summary>Creates evidence for a finite, nonnegative, larger-is-worse statistic exceeding its limit.</summary>
    public KernelTuningRegressionEvidence(
        KernelTuningRegressionReason reason, string policyVersion, string rawEvidenceSha256,
        double observedValue, double maximumAllowedValue, DateTimeOffset observedAt)
    {
        if (!Enum.IsDefined(typeof(KernelTuningRegressionReason), reason))
            throw new ArgumentOutOfRangeException(nameof(reason));
        QuarantineEncoding.ValidateLabel(policyVersion, nameof(policyVersion));
        if (rawEvidenceSha256 is null || rawEvidenceSha256.Length != 64 ||
            rawEvidenceSha256.Any(c => !((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'))))
            throw new ArgumentException("Evidence must have a lowercase SHA256 digest.", nameof(rawEvidenceSha256));
        if (!KernelTuningMeasurement.IsFinite(maximumAllowedValue) || maximumAllowedValue < 0)
            throw new ArgumentOutOfRangeException(nameof(maximumAllowedValue));
        if (!KernelTuningMeasurement.IsFinite(observedValue) || observedValue <= maximumAllowedValue)
            throw new ArgumentOutOfRangeException(nameof(observedValue));
        if (observedAt == default) throw new ArgumentOutOfRangeException(nameof(observedAt));
        Reason = reason;
        PolicyVersion = policyVersion;
        RawEvidenceSha256 = rawEvidenceSha256;
        ObservedValue = observedValue;
        MaximumAllowedValue = maximumAllowedValue;
        ObservedAt = observedAt.ToUniversalTime();
    }

    /// <summary>Gets the category of regression.</summary>
    public KernelTuningRegressionReason Reason { get; }
    /// <summary>Gets the caller's versioned measurement and decision policy.</summary>
    public string PolicyVersion { get; }
    /// <summary>Gets the digest of the separately retained raw evidence artifact.</summary>
    public string RawEvidenceSha256 { get; }
    /// <summary>Gets the observed larger-is-worse statistic in the policy's units.</summary>
    public double ObservedValue { get; }
    /// <summary>Gets the maximum permitted statistic in the same units.</summary>
    public double MaximumAllowedValue { get; }
    /// <summary>Gets the UTC time of observation.</summary>
    public DateTimeOffset ObservedAt { get; }
}

/// <summary>Separates in-memory safety from successful persistence and optional rollback.</summary>
public sealed class KernelTuningQuarantineResult<TConfiguration> where TConfiguration : notnull
{
    internal KernelTuningQuarantineResult(bool applied, bool persisted, string? receiptPath,
        KernelTuningDeploymentSnapshot<TConfiguration>? rollback, bool rollbackPersisted)
    {
        WasApplied = applied;
        WasPersisted = persisted;
        ReceiptPath = receiptPath;
        RollbackDeployment = rollback;
        WasRollbackPersisted = rollbackPersisted;
    }

    /// <summary>Gets whether the exact observed snapshot was deactivated and blocked in this process.</summary>
    public bool WasApplied { get; }
    /// <summary>Gets whether this invocation durably retained a complete regression receipt.</summary>
    public bool WasPersisted { get; }
    /// <summary>Gets the durably retained receipt path, or null, including for best-effort-only writes.</summary>
    public string? ReceiptPath { get; }
    /// <summary>Gets the supplied prior snapshot published by this operation, or null if none was published.</summary>
    public KernelTuningDeploymentSnapshot<TConfiguration>? RollbackDeployment { get; }
    /// <summary>Gets whether the published rollback was saved by the underlying winner store.</summary>
    public bool WasRollbackPersisted { get; }
}

/// <summary>Opt-in persistent quarantine decorator for an existing typed winner store.</summary>
/// <remarks>
/// Use one canonical private journal directory and this decorator for every producer sharing a deployment.
/// A record's presence blocks its configuration, including corrupt or unknown records. Records are never
/// deleted automatically. Configuration identity includes the tuning envelope and codec id/version, but not
/// the selecting run hash: another run cannot silently retry the same quarantined configuration.
/// Publication and quarantine are serialized within this process. Other processes check durable records on
/// admission, but already active handles and simultaneous cross-process publication are not revoked atomically.
/// Filesystem aliases, hostile journal tampering and automatic drift detection are outside this contract.
/// Failed writes retain a process-local block; they do not establish restart safety. Dispatch reads remain I/O-free.
/// Durable publication currently requires Linux x86/x64/ARM32/ARM64 file/directory synchronization. Other platforms retain a
/// best-effort tombstone but report persistence failure. Storage must honor the operating system's barriers.
/// </remarks>
public sealed class QuarantinedKernelTuningStore<TConfiguration> : IKernelTuningStore<TConfiguration>
    where TConfiguration : notnull
{
    private readonly string _directory;
    private readonly QuarantineJournalState _state;
    private readonly IKernelTuningStore<TConfiguration> _inner;
    private readonly IQuarantineCommitOperations _commitOperations;

    /// <summary>Creates the explicit journal directory; initialization failures are reported to the caller.</summary>
    public QuarantinedKernelTuningStore(string journalDirectory, IKernelTuningStore<TConfiguration>? inner = null)
        : this(journalDirectory, inner, QuarantineReceiptCommit.Native)
    {
    }

    internal QuarantinedKernelTuningStore(string journalDirectory, IKernelTuningStore<TConfiguration>? inner,
        IQuarantineCommitOperations commitOperations)
    {
        _commitOperations = commitOperations ?? throw new ArgumentNullException(nameof(commitOperations));
        if (string.IsNullOrWhiteSpace(journalDirectory) || !Path.IsPathRooted(journalDirectory) ||
            (Path.DirectorySeparatorChar == '\\' && !journalDirectory.StartsWith("\\\\", StringComparison.Ordinal) &&
             (journalDirectory.Length < 3 || journalDirectory[1] != ':' ||
              (journalDirectory[2] != '\\' && journalDirectory[2] != '/'))))
            throw new ArgumentException("An absolute private journal directory is required.", nameof(journalDirectory));
        _directory = Path.GetFullPath(journalDirectory).TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar);
        if (string.IsNullOrEmpty(_directory) || string.Equals(_directory, Path.GetPathRoot(_directory)?.TrimEnd(
                Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar), StringComparison.OrdinalIgnoreCase))
            throw new ArgumentException("The filesystem root cannot be a quarantine journal.", nameof(journalDirectory));
        Directory.CreateDirectory(_directory);
        _state = QuarantineJournalState.For(_directory);
        _inner = inner ?? new AutotuneCacheKernelTuningStore<TConfiguration>();
    }

    /// <inheritdoc />
    public bool TryLoad(KernelTuningIdentity identity, IEvolutionGenomeCodec<TConfiguration> codec,
        out KernelTuningDeploymentSnapshot<TConfiguration>? snapshot)
    {
        snapshot = null;
        try
        {
            if (!_inner.TryLoad(identity, codec, out var loaded) || loaded is null ||
                !string.Equals(identity.StableKey, loaded.Identity.StableKey, StringComparison.Ordinal)) return false;
            Entry entry = Describe(loaded, codec);
            lock (_state.Gate)
            {
                if (!IsAdmitted(entry.Key)) return false;
                snapshot = loaded;
                return true;
            }
        }
        catch { return false; }
    }

    /// <inheritdoc />
    public bool TryStore(KernelTuningDeploymentSnapshot<TConfiguration> snapshot,
        IEvolutionGenomeCodec<TConfiguration> codec)
    {
        try
        {
            Entry entry = Describe(snapshot, codec);
            lock (_state.Gate) { if (!IsAdmitted(entry.Key)) return false; }
            // User stores run outside the journal lock. A racing stale write remains inadmissible on load.
            if (!_inner.TryStore(snapshot, codec)) return false;
            lock (_state.Gate) { return IsAdmitted(entry.Key); }
        }
        catch { return false; }
    }

    internal bool CanDeploy(KernelTuningIdentity identity, TConfiguration configuration,
        IEvolutionGenomeCodec<TConfiguration> codec)
    {
        try
        {
            Entry entry = Describe(identity, configuration, codec);
            lock (_state.Gate) { return IsAdmitted(entry.Key); }
        }
        catch { return false; }
    }

    internal bool TryPublish(KernelTuningDeployment<TConfiguration> deployment,
        KernelTuningDeploymentSnapshot<TConfiguration> snapshot, IEvolutionGenomeCodec<TConfiguration> codec,
        bool onlyIfEmpty)
    {
        try
        {
            // Caller codec execution must precede the lock; it may reenter the tuner.
            Entry entry = Describe(snapshot, codec);
            lock (_state.Gate)
            {
                if (!IsAdmitted(entry.Key)) return false;
                if (onlyIfEmpty) return deployment.TryPublishIfEmpty(snapshot);
                deployment.Publish(snapshot);
                return true;
            }
        }
        catch { return false; }
    }

    internal KernelTuningQuarantineResult<TConfiguration> Quarantine(
        KernelTuningDeployment<TConfiguration> deployment, KernelTuningDeploymentSnapshot<TConfiguration> expected,
        IEvolutionGenomeCodec<TConfiguration> codec, KernelTuningRegressionEvidence evidence,
        KernelTuningDeploymentSnapshot<TConfiguration>? prior, CancellationToken cancellationToken)
    {
        Entry entry = Describe(expected, codec);
        QuarantineEncoding.ValidateLabel(expected.RunStateHash, nameof(expected));
        Entry? rollbackEntry = null;
        if (prior is not null)
        {
            try { rollbackEntry = Describe(prior, codec); }
            catch { prior = null; }
        }
        byte[] receipt = JsonSerializer.SerializeToUtf8Bytes(new
        {
            Schema = "tensor-kernel-quarantine-v1",
            entry.Key,
            entry.Identity,
            entry.CodecId,
            entry.CodecVersion,
            entry.GenomeId,
            entry.PayloadBase64,
            expected.RunStateHash,
            Evidence = evidence
        });
        bool persisted;
        KernelTuningDeploymentSnapshot<TConfiguration>? rollback = null;
        string path = RecordPath(entry.Key);
        lock (_state.Gate)
        {
            cancellationToken.ThrowIfCancellationRequested();
            // CAS also protects callers that independently invoke in-memory TryDeactivate.
            if (!deployment.TryDeactivate(expected))
                return new KernelTuningQuarantineResult<TConfiguration>(false, false, null, null, false);
            _state.Blocked.Add(entry.Key);
            persisted = TryWriteReceipt(path, receipt);
            if (prior is not null && rollbackEntry is not null &&
                string.Equals(prior.Identity.StableKey, expected.Identity.StableKey, StringComparison.Ordinal) &&
                IsAdmitted(rollbackEntry.Key) && deployment.TryPublishIfEmpty(prior)) rollback = prior;
        }
        bool rollbackPersisted = rollback is not null && TryStore(rollback, codec);
        return new KernelTuningQuarantineResult<TConfiguration>(true, persisted, persisted ? path : null,
            rollback, rollbackPersisted);
    }

    private bool IsAdmitted(string key)
    {
        if (_state.Blocked.Contains(key)) return false;
        try
        {
            // Exists() hides permission errors. Only a missing record in a readable directory allows admission.
            if ((File.GetAttributes(_directory) & FileAttributes.Directory) == 0) return false;
            using var record = new FileStream(RecordPath(key), FileMode.Open, FileAccess.Read, FileShare.ReadWrite);
            return false; // Any existing record, even empty/corrupt/future-schema, is a tombstone.
        }
        catch (FileNotFoundException) { return DirectoryIsReadable(); }
        catch { return false; }
    }

    private bool DirectoryIsReadable()
    {
        try
        {
            using var entries = Directory.EnumerateFileSystemEntries(_directory).GetEnumerator();
            _ = entries.MoveNext();
            return true;
        }
        catch { return false; }
    }

    private string RecordPath(string key) => Path.Combine(_directory, key + ".quarantine.json");

    private bool TryWriteReceipt(string path, byte[] receipt)
    {
        // Conservative status: a preexisting or corrupt receipt is still blocking, but is not this write's success.
        try
        {
            string? directory = Path.GetDirectoryName(path);
            if (string.IsNullOrEmpty(directory)) return false;
            string pending = Path.Combine(directory, ".quarantine-" + Guid.NewGuid().ToString("N") + ".pending");
            using (var stream = new FileStream(pending, FileMode.CreateNew, FileAccess.Write, FileShare.None))
            {
                stream.Write(receipt, 0, receipt.Length);
                stream.Flush(true);
            }
            return QuarantineReceiptCommit.TryCommit(pending, path, _commitOperations);
        }
        catch { return false; }
    }

    private static Entry Describe(KernelTuningDeploymentSnapshot<TConfiguration> snapshot,
        IEvolutionGenomeCodec<TConfiguration> codec)
    {
        if (snapshot is null) throw new ArgumentNullException(nameof(snapshot));
        Entry entry = Describe(snapshot.Identity, snapshot.Configuration, codec);
        if (!string.Equals(entry.GenomeId, snapshot.GenomeId, StringComparison.Ordinal))
            throw new ArgumentException("The snapshot does not match its canonical configuration.", nameof(snapshot));
        return entry;
    }

    private static Entry Describe(KernelTuningIdentity identity, TConfiguration configuration,
        IEvolutionGenomeCodec<TConfiguration> codec)
    {
        if (identity is null) throw new ArgumentNullException(nameof(identity));
        if (codec is null) throw new ArgumentNullException(nameof(codec));
        string id = codec.Id;
        string version = codec.VersionHash;
        QuarantineEncoding.ValidateLabel(id, nameof(codec));
        QuarantineEncoding.ValidateLabel(version, nameof(codec));
        string payload = codec.Serialize(configuration);
        if (payload is null || payload.Length > 65536)
            throw new ArgumentException("Quarantine payloads must not exceed 64K UTF16 code units.", nameof(configuration));
        byte[] bytes = QuarantineEncoding.StrictUtf8.GetBytes(payload);
        if (bytes.Length > 65536)
            throw new ArgumentException("Quarantine payloads must not exceed 64 KiB UTF8.", nameof(configuration));
        if (!string.Equals(id, codec.Id, StringComparison.Ordinal) ||
            !string.Equals(version, codec.VersionHash, StringComparison.Ordinal))
            throw new InvalidOperationException("The codec identity changed during serialization.");
        string genomeId = EvolutionHash.Compute(payload);
        string key = EvolutionHash.Combine(new[] { "tensor-kernel-quarantine-v1", identity.StableKey, id, version, genomeId });
        return new Entry(key, identity.StableKey, id, version, genomeId, Convert.ToBase64String(bytes));
    }

    private sealed class Entry
    {
        internal Entry(string key, string identity, string codecId, string codecVersion, string genomeId, string payloadBase64)
        {
            Key = key; Identity = identity; CodecId = codecId; CodecVersion = codecVersion;
            GenomeId = genomeId; PayloadBase64 = payloadBase64;
        }
        public string Key { get; }
        public string Identity { get; }
        public string CodecId { get; }
        public string CodecVersion { get; }
        public string GenomeId { get; }
        public string PayloadBase64 { get; }
    }
}

internal sealed class QuarantineJournalState
{
    private static readonly ConcurrentDictionary<string, QuarantineJournalState> Journals = new(
        Path.DirectorySeparatorChar == '\\' ? StringComparer.OrdinalIgnoreCase : StringComparer.Ordinal);
    internal object Gate { get; } = new();
    internal HashSet<string> Blocked { get; } = new(StringComparer.Ordinal);
    internal static QuarantineJournalState For(string directory) => Journals.GetOrAdd(directory, _ => new());
}

internal static class QuarantineEncoding
{
    internal static readonly UTF8Encoding StrictUtf8 = new(false, true);
    internal static void ValidateLabel(string value, string parameterName)
    {
        if (string.IsNullOrWhiteSpace(value) || value.Length > 256 || value.Any(char.IsControl))
            throw new ArgumentException("A nonempty printable label of at most 256 characters is required.", parameterName);
        try { _ = StrictUtf8.GetByteCount(value); }
        catch (EncoderFallbackException error) { throw new ArgumentException("Labels must be valid Unicode.", parameterName, error); }
    }
}
