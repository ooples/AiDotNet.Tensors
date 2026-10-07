using System.Text.Json;

namespace AiDotNet.Tensors.Helpers.Autotune.TunedKernels;

/// <summary>
/// Shipped and persisted kernel choices. A shipped profile is an embedded <c>*.tuned-profile.json</c> resource that
/// records, per device, the candidate the evidence gate selected for each (op, shape class); it lets a known device
/// skip online measurement and is what <see cref="TunedKernelMode.Pinned"/> serves. Persisted decisions are the
/// gate's own results on this machine, written through <see cref="AutotuneCache"/> so a later process reuses them.
/// </summary>
/// <remarks>
/// Profile format (schema 1):
/// <code>
/// { "schemaVersion": 1, "device": "cuda:sm75:NVIDIA GeForce GTX 1660 Ti",
///   "entries": [ { "op": "Softmax", "shape": "f32:8192x32", "deterministic": true,
///                  "candidate": "tuned.softmax.lanes32", "speedup": 9.1 } ] }
/// </code>
/// A <c>device</c> ending in <c>*</c> matches every device key with that prefix. An entry naming a candidate that is
/// not registered or not applicable is ignored and the slot falls through to measurement or the reference.
/// </remarks>
public static class TunedKernelProfiles
{
    private const int SchemaVersion = 1;
    private const string ResourceSuffix = ".tuned-profile.json";
    private static readonly KernelId PersistCategory = new("tuned-kernel", "registry");
    private static readonly Lazy<IReadOnlyList<ProfileDocument>> s_shipped = new(LoadShipped);
    private static readonly List<ProfileDocument> s_runtime = new();
    private static readonly object s_runtimeLock = new();

    /// <summary>Adds a profile document at runtime (hosts shipping their own device profiles, and tests).</summary>
    public static void AddProfile(string json)
    {
        var doc = Parse(json) ?? throw new ArgumentException("Not a schema-1 tuned-kernel profile.", nameof(json));
        lock (s_runtimeLock) s_runtime.Add(doc);
    }

    /// <summary>Removes runtime-added profiles (tests).</summary>
    internal static void ClearRuntimeProfiles()
    {
        lock (s_runtimeLock) s_runtime.Clear();
    }

    /// <summary>Looks up a candidate id for the key: runtime profiles, then shipped, then persisted decisions.</summary>
    /// <param name="poolKey">Identifies the slot's candidate set; persisted decisions made against a different
    /// set (an older build, or before an external artifact joined) are not reused, so new candidates get measured.</param>
    public static bool TryLookup(TunedKernelOp op, string device, in TunedShape shape, bool deterministic,
        out string? candidateId, string poolKey = "")
    {
        string shapeText = shape.ToString();
        ProfileDocument[] runtime;
        lock (s_runtimeLock) runtime = s_runtime.ToArray();
        foreach (var doc in runtime.Concat(s_shipped.Value))
        {
            if (!DeviceMatches(doc.Device, device)) continue;
            foreach (var e in doc.Entries)
            {
                if (e.Op == op.ToString() && e.Shape == shapeText &&
                    (!deterministic || e.Deterministic) && !string.IsNullOrEmpty(e.Candidate))
                {
                    candidateId = e.Candidate;
                    return true;
                }
            }
        }

        candidateId = null;
        if (!TunedKernelPolicy.PersistDecisions) return false;
        try
        {
            var choice = AutotuneCache.Lookup(PersistKey(op, device, poolKey), PersistShape(shape, deterministic));
            if (choice is null || string.IsNullOrEmpty(choice.Variant)) return false;
            candidateId = choice.Variant;
            return true;
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException or InvalidDataException or JsonException)
        {
            return false;
        }
    }

    /// <summary>Persists a measured decision so later processes on this machine reuse it.</summary>
    internal static void Persist(TunedKernelDecision decision, bool deterministic, string poolKey = "")
    {
        if (!TunedKernelPolicy.PersistDecisions) return;
        if (decision.Reason != TunedKernelDecisionReason.Tuned &&
            decision.Reason != TunedKernelDecisionReason.ReferenceWon) return;
        var choice = new KernelChoice
        {
            Variant = decision.CandidateId,
            MeasuredTimeMs = double.IsNaN(decision.CandidateMilliseconds) ? 0 : decision.CandidateMilliseconds,
            Parameters = new Dictionary<string, string>
            {
                ["reason"] = decision.Reason.ToString(),
                ["reference"] = decision.ReferenceId,
                ["speedup"] = decision.MedianSpeedup.ToString("R", System.Globalization.CultureInfo.InvariantCulture),
                ["device"] = decision.Device,
            },
        };
        AutotuneCache.TryStore(PersistKey(decision.Op, decision.Device, poolKey), PersistShape(decision.Shape, deterministic), choice);
    }

    /// <summary>Serializes decisions as a shippable profile document for <paramref name="device"/>.</summary>
    public static string ExportProfile(string device, IEnumerable<TunedKernelDecision> decisions, bool deterministic)
    {
        var entries = decisions
            .Where(d => d.Device == device &&
                        (d.Reason == TunedKernelDecisionReason.Tuned || d.Reason == TunedKernelDecisionReason.ReferenceWon))
            .Select(d => new ProfileEntry
            {
                Op = d.Op.ToString(),
                Shape = d.Shape.ToString(),
                Deterministic = deterministic,
                Candidate = d.CandidateId,
                Speedup = double.IsNaN(d.MedianSpeedup) ? 1.0 : Math.Round(d.MedianSpeedup, 3),
            })
            .ToList();
        return JsonSerializer.Serialize(new ProfileDocument { SchemaVersion = SchemaVersion, Device = device, Entries = entries },
            new JsonSerializerOptions { WriteIndented = true });
    }

    private static KernelId PersistKey(TunedKernelOp op, string device, string poolKey) =>
        new(PersistCategory.Category, op + "@" + device + "#" + poolKey);

    private static ShapeProfile PersistShape(in TunedShape shape, bool deterministic)
    {
        int[] dims = shape.ToArray();
        var all = new int[dims.Length + 2];
        all[0] = shape.DType;
        all[1] = deterministic ? 1 : 0;
        Array.Copy(dims, 0, all, 2, dims.Length);
        return new ShapeProfile(all);
    }

    private static bool DeviceMatches(string pattern, string device) =>
        pattern.EndsWith("*", StringComparison.Ordinal)
            ? device.StartsWith(pattern.Substring(0, pattern.Length - 1), StringComparison.Ordinal)
            : string.Equals(pattern, device, StringComparison.Ordinal);

    private static IReadOnlyList<ProfileDocument> LoadShipped()
    {
        var docs = new List<ProfileDocument>();
        var asm = typeof(TunedKernelProfiles).Assembly;
        foreach (string name in asm.GetManifestResourceNames())
        {
            if (!name.EndsWith(ResourceSuffix, StringComparison.OrdinalIgnoreCase)) continue;
            try
            {
                using var stream = asm.GetManifestResourceStream(name);
                if (stream is null) continue;
                using var reader = new StreamReader(stream);
                var doc = Parse(reader.ReadToEnd());
                if (doc is not null) docs.Add(doc);
            }
            catch (Exception ex) when (ex is IOException or JsonException)
            {
                System.Diagnostics.Trace.TraceWarning($"Ignoring unreadable tuned-kernel profile {name}: {ex.Message}");
            }
        }
        return docs;
    }

    private static ProfileDocument? Parse(string json)
    {
        var doc = JsonSerializer.Deserialize<ProfileDocument>(json);
        if (doc is null || doc.SchemaVersion != SchemaVersion || string.IsNullOrEmpty(doc.Device)) return null;
        doc.Entries ??= new List<ProfileEntry>();
        return doc;
    }

    private sealed class ProfileDocument
    {
        [System.Text.Json.Serialization.JsonPropertyName("schemaVersion")]
        public int SchemaVersion { get; set; }
        [System.Text.Json.Serialization.JsonPropertyName("device")]
        public string Device { get; set; } = string.Empty;
        [System.Text.Json.Serialization.JsonPropertyName("entries")]
        public List<ProfileEntry> Entries { get; set; } = new();
    }

    private sealed class ProfileEntry
    {
        [System.Text.Json.Serialization.JsonPropertyName("op")]
        public string Op { get; set; } = string.Empty;
        [System.Text.Json.Serialization.JsonPropertyName("shape")]
        public string Shape { get; set; } = string.Empty;
        [System.Text.Json.Serialization.JsonPropertyName("deterministic")]
        public bool Deterministic { get; set; }
        [System.Text.Json.Serialization.JsonPropertyName("candidate")]
        public string Candidate { get; set; } = string.Empty;
        [System.Text.Json.Serialization.JsonPropertyName("speedup")]
        public double Speedup { get; set; }
    }
}
