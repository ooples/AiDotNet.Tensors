using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using AiDotNet.Evolution;

namespace AiDotNet.Tensors.Helpers.Autotune;

/// <summary>Explicit applicability inputs beyond the device/shape/protocol tuning identity.</summary>
/// <remarks>Hashes are supplied by the application, not inferred attestations. Use an explicit versioned hash for an inapplicable dimension.</remarks>
public sealed class KernelTuningApplicabilityEnvelope
{
    /// <summary>Creates an exact-match envelope. Changing any input requires fresh validation or retuning.</summary>
    public KernelTuningApplicabilityEnvelope(KernelTuningIdentity identity, string runtimeSha256,
        string compilerSha256, string datasetSha256, string workloadSha256)
    {
        Identity = identity ?? throw new ArgumentNullException(nameof(identity));
        foreach (string value in new[] { runtimeSha256, compilerSha256, datasetSha256, workloadSha256 })
            KernelArtifactEncoding.RequireDigest(value);
        RuntimeSha256 = runtimeSha256;
        CompilerSha256 = compilerSha256;
        DatasetSha256 = datasetSha256;
        WorkloadSha256 = workloadSha256;
        StableKey = EvolutionHash.Combine(new[] { "kernel-applicability-v1", identity.StableKey,
            runtimeSha256, compilerSha256, datasetSha256, workloadSha256 });
    }

    /// <summary>Gets the kernel/device/backend/shape/protocol identity.</summary>
    public KernelTuningIdentity Identity { get; }
    /// <summary>Gets the runtime fingerprint.</summary>
    public string RuntimeSha256 { get; }
    /// <summary>Gets the compiler/toolchain fingerprint.</summary>
    public string CompilerSha256 { get; }
    /// <summary>Gets the dataset fingerprint.</summary>
    public string DatasetSha256 { get; }
    /// <summary>Gets the workload and dispatch-contract fingerprint.</summary>
    public string WorkloadSha256 { get; }
    /// <summary>Gets the exact applicability key.</summary>
    public string StableKey { get; }
}

/// <summary>A visible immutable object receipt, with power-loss durability reported separately.</summary>
public sealed class KernelTuningArtifactReceipt
{
    internal KernelTuningArtifactReceipt(string id, bool durable) { ArtifactId = id; IsDurable = durable; }
    /// <summary>Gets the SHA-256 of the exact retained bytes.</summary>
    public string ArtifactId { get; }
    /// <summary>Gets whether file and supported directory durability barriers succeeded in this invocation.</summary>
    public bool IsDurable { get; }
}

/// <summary>Bounded content-addressed configuration/evidence registry, independent of the mutable winner cache.</summary>
/// <remarks>
/// The registry never promotes a loaded artifact. Replay and application policy remain mandatory.
/// Protect the directory against untrusted writers; content hashes establish integrity, not authenticity.
/// Failed pending files are retained. Storage retention/capacity is managed by the application.
/// </remarks>
public sealed class KernelTuningArtifactRegistry<TConfiguration> where TConfiguration : notnull
{
    private readonly string _directory;
    private const int MaximumBytes = 4 * 1024 * 1024;
    private static readonly JsonSerializerOptions JsonOptions = new() { MaxDepth = 32 };

    /// <summary>Creates a registry in an explicitly selected absolute directory.</summary>
    public KernelTuningArtifactRegistry(string directory)
    {
        if (string.IsNullOrWhiteSpace(directory) || !Path.IsPathRooted(directory))
            throw new ArgumentException("An absolute private registry directory is required.", nameof(directory));
        _directory = Path.GetFullPath(directory);
        Directory.CreateDirectory(_directory);
        KernelArtifactEncoding.RefuseLink(_directory);
    }

    /// <summary>Registers the exact canonical configuration and its paired validation evidence without activating it.</summary>
    public KernelTuningArtifactReceipt Register(KernelTuningDeploymentSnapshot<TConfiguration> snapshot,
        KernelTuningApplicabilityEnvelope envelope, IEvolutionGenomeCodec<TConfiguration> codec)
    {
        if (snapshot is null) throw new ArgumentNullException(nameof(snapshot));
        if (envelope is null) throw new ArgumentNullException(nameof(envelope));
        if (codec is null) throw new ArgumentNullException(nameof(codec));
        if (snapshot.Identity.StableKey != envelope.Identity.StableKey)
            throw new ArgumentException("Snapshot and applicability identity differ.", nameof(envelope));
        string codecId = codec.Id, codecVersion = codec.VersionHash;
        KernelChoice choice = AutotuneCacheKernelTuningStore<TConfiguration>.Encode(snapshot, codec);
        var strictUtf8 = new UTF8Encoding(false, true);
        foreach (var value in choice.Parameters.Values)
            if (value.Length > MaximumBytes || strictUtf8.GetByteCount(value) > MaximumBytes)
                throw new InvalidDataException("Kernel artifact field exceeds its byte bound.");
        // Wall time is not part of artifact identity; registering the same evidence is idempotent.
        choice.RecordedAtUtc = new DateTime(1970, 1, 1, 0, 0, 0, DateTimeKind.Utc);
        byte[] bytes = JsonSerializer.SerializeToUtf8Bytes(new ArtifactDocument
        {
            SchemaVersion = 1, EnvelopeKey = envelope.StableKey, RuntimeSha256 = envelope.RuntimeSha256,
            CompilerSha256 = envelope.CompilerSha256, DatasetSha256 = envelope.DatasetSha256,
            WorkloadSha256 = envelope.WorkloadSha256, Choice = choice
        }, JsonOptions);
        if (bytes.Length > MaximumBytes) throw new InvalidDataException("Kernel artifact exceeds its byte bound.");
        // Verify the same decoding/canonicalization contract before publishing, not only when loading later.
        if (!AutotuneCacheKernelTuningStore<TConfiguration>.TryDecode(envelope.Identity, codec, choice, out _))
            throw new InvalidDataException("Kernel artifact failed its validation round trip.");
        if (codec.Id != codecId || codec.VersionHash != codecVersion ||
            choice.Parameters["codec-id"] != codecId || choice.Parameters["codec-version"] != codecVersion)
            throw new InvalidDataException("Kernel artifact codec changed during registration.");
        return Write(bytes);
    }

    /// <summary>Loads an integrity-checked artifact for the exact envelope and codec; this does not authorize promotion.</summary>
    public KernelTuningDeploymentSnapshot<TConfiguration> Load(string artifactId,
        KernelTuningApplicabilityEnvelope envelope, IEvolutionGenomeCodec<TConfiguration> codec)
    {
        if (envelope is null) throw new ArgumentNullException(nameof(envelope));
        if (codec is null) throw new ArgumentNullException(nameof(codec));
        byte[] bytes = Read(artifactId);
        string codecId = codec.Id, codecVersion = codec.VersionHash;
        using var parsed = JsonDocument.Parse(bytes, new JsonDocumentOptions { MaxDepth = 32 });
        KernelArtifactEncoding.RequireUniqueProperties(parsed.RootElement);
        var document = JsonSerializer.Deserialize<ArtifactDocument>(bytes, JsonOptions)
            ?? throw new InvalidDataException("Missing kernel artifact.");
        if (document.SchemaVersion != 1 || document.EnvelopeKey != envelope.StableKey ||
            document.RuntimeSha256 != envelope.RuntimeSha256 || document.CompilerSha256 != envelope.CompilerSha256 ||
            document.DatasetSha256 != envelope.DatasetSha256 || document.WorkloadSha256 != envelope.WorkloadSha256 ||
            !AutotuneCacheKernelTuningStore<TConfiguration>.TryDecode(envelope.Identity, codec, document.Choice, out var snapshot) ||
            snapshot is null)
            throw new InvalidDataException("Kernel artifact is invalid or outside the requested applicability envelope.");
        if (codec.Id != codecId || codec.VersionHash != codecVersion)
            throw new InvalidDataException("Kernel artifact codec changed during loading.");
        return snapshot;
    }

    /// <summary>Retains bounded raw monitoring evidence before it is referenced by a regression receipt.</summary>
    public KernelTuningArtifactReceipt RetainEvidence(byte[] evidence)
    {
        if (evidence is null) throw new ArgumentNullException(nameof(evidence));
        if (evidence.Length == 0 || evidence.Length > MaximumBytes) throw new ArgumentOutOfRangeException(nameof(evidence));
        return Write((byte[])evidence.Clone());
    }

    /// <summary>Reads exact retained bytes with size, link and SHA-256 checks.</summary>
    public byte[] Read(string artifactId)
    {
        KernelArtifactEncoding.RequireDigest(artifactId);
        KernelArtifactEncoding.RefuseLink(_directory);
        string path = Path.Combine(_directory, artifactId + ".artifact");
        KernelArtifactEncoding.RefuseLink(path);
        using var file = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read);
        if (file.Length <= 0 || file.Length > MaximumBytes) throw new InvalidDataException("Invalid artifact size.");
        var bytes = new byte[(int)file.Length];
        int offset = 0;
        while (offset < bytes.Length)
        {
            int count = file.Read(bytes, offset, bytes.Length - offset);
            if (count == 0) throw new EndOfStreamException();
            offset += count;
        }
        if (file.ReadByte() != -1 || KernelArtifactEncoding.Hash(bytes) != artifactId)
            throw new InvalidDataException("Artifact content hash mismatch.");
        return bytes;
    }

    private KernelTuningArtifactReceipt Write(byte[] bytes)
    {
        KernelArtifactEncoding.RefuseLink(_directory);
        string id = KernelArtifactEncoding.Hash(bytes);
        string destination = Path.Combine(_directory, id + ".artifact");
        if (File.Exists(destination)) return ConfirmExisting(id, destination);
        string pending = Path.Combine(_directory, Guid.NewGuid().ToString("N") + ".pending");
        using (var stream = new FileStream(pending, FileMode.CreateNew, FileAccess.Write, FileShare.None))
        { stream.Write(bytes, 0, bytes.Length); stream.Flush(flushToDisk: true); }
        try
        {
            bool durable = QuarantineReceiptCommit.TryCommit(pending, destination, QuarantineReceiptCommit.Native);
            Read(id); // A visible object and a verified directory barrier are distinct results.
            return new(id, durable);
        }
        catch (IOException) when (File.Exists(destination))
        { return ConfirmExisting(id, destination); } // A cooperating identical writer may have won the race.
    }

    private KernelTuningArtifactReceipt ConfirmExisting(string id, string destination)
    {
        Read(id);
        bool durable;
        try { durable = QuarantineReceiptCommit.TryConfirmDurable(destination, QuarantineReceiptCommit.Native); }
        catch (Exception error) when (error is IOException or UnauthorizedAccessException) { durable = false; }
        Read(id);
        return new(id, durable);
    }

    private sealed class ArtifactDocument
    {
        public ArtifactDocument() { }
        public int SchemaVersion { get; set; }
        public string EnvelopeKey { get; set; } = string.Empty;
        public string RuntimeSha256 { get; set; } = string.Empty;
        public string CompilerSha256 { get; set; } = string.Empty;
        public string DatasetSha256 { get; set; } = string.Empty;
        public string WorkloadSha256 { get; set; } = string.Empty;
        public KernelChoice? Choice { get; set; }
    }
}

internal static class KernelArtifactEncoding
{
    internal static void RequireDigest(string value)
    {
        if (value is null || value.Length != 64 || value.Any(c => !(c >= '0' && c <= '9' || c >= 'a' && c <= 'f')))
            throw new ArgumentException("A lowercase SHA-256 digest is required.", nameof(value));
    }

    internal static string Hash(byte[] bytes)
    {
        using var hash = SHA256.Create();
        return BitConverter.ToString(hash.ComputeHash(bytes)).Replace("-", string.Empty).ToLowerInvariant();
    }

    internal static void RefuseLink(string path)
    {
        if ((File.GetAttributes(path) & FileAttributes.ReparsePoint) != 0)
            throw new IOException("Linked artifact paths are not supported.");
    }

    internal static void RequireUniqueProperties(JsonElement value)
    {
        if (value.ValueKind == JsonValueKind.Object)
        {
            var names = new HashSet<string>(StringComparer.Ordinal);
            foreach (var property in value.EnumerateObject())
            {
                if (!names.Add(property.Name)) throw new InvalidDataException("Duplicate artifact property.");
                RequireUniqueProperties(property.Value);
            }
        }
        else if (value.ValueKind == JsonValueKind.Array)
            foreach (var child in value.EnumerateArray()) RequireUniqueProperties(child);
    }
}
