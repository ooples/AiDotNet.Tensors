using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace AiDotNet.Tensors.Helpers.Autotune.TunedKernels;

/// <summary>How the launch grid is derived from the op's shape.</summary>
public enum ExternalKernelGridRule
{
    /// <summary>gridX = ceil(rows / RowsPerBlock).</summary>
    RowsPerBlock,
    /// <summary>gridX = ceil(columns / ColumnsPerBlock).</summary>
    ColumnsPerBlock,
}

/// <summary>Launch configuration of an external kernel.</summary>
public sealed class ExternalKernelLaunch
{
    /// <summary>Block dimension x.</summary>
    [JsonPropertyName("blockX")] public int BlockX { get; set; } = 256;
    /// <summary>Block dimension y.</summary>
    [JsonPropertyName("blockY")] public int BlockY { get; set; } = 1;
    /// <summary>Dynamic shared memory bytes.</summary>
    [JsonPropertyName("sharedMemoryBytes")] public int SharedMemoryBytes { get; set; }
    /// <summary>How the grid is derived.</summary>
    [JsonPropertyName("gridRule")] public ExternalKernelGridRule GridRule { get; set; }
    /// <summary>Rows or columns handled by one block, per <see cref="GridRule"/>.</summary>
    [JsonPropertyName("unitsPerBlock")] public int UnitsPerBlock { get; set; } = 1;
}

/// <summary>Shape limits within which the artifact claims to be valid.</summary>
public sealed class ExternalKernelConstraints
{
    /// <summary>Smallest supported row length (columns).</summary>
    [JsonPropertyName("minColumns")] public int MinColumns { get; set; } = 1;
    /// <summary>Largest supported row length (columns).</summary>
    [JsonPropertyName("maxColumns")] public int MaxColumns { get; set; } = int.MaxValue;
    /// <summary>Row length must be a multiple of this.</summary>
    [JsonPropertyName("columnsMultipleOf")] public int ColumnsMultipleOf { get; set; } = 1;
}

/// <summary>The producer's own evidence. Informational: the consuming registry re-gates every artifact locally.</summary>
public sealed class ExternalKernelEvidence
{
    /// <summary>Device key the evidence was measured on.</summary>
    [JsonPropertyName("device")] public string Device { get; set; } = string.Empty;
    /// <summary>Shapes measured, in <see cref="TunedShape"/> text form.</summary>
    [JsonPropertyName("shapes")] public List<string> Shapes { get; set; } = new();
    /// <summary>Median speedup over the incumbent.</summary>
    [JsonPropertyName("medianSpeedup")] public double MedianSpeedup { get; set; }
    /// <summary>Worst relative error against the reference.</summary>
    [JsonPropertyName("maxRelativeError")] public double MaxRelativeError { get; set; }
    /// <summary>Incumbent candidate id the evidence was measured against.</summary>
    [JsonPropertyName("incumbent")] public string Incumbent { get; set; } = string.Empty;
}

/// <summary>
/// A versioned, externally evolved kernel (for example from <c>AiDotNet.Evolution.Ptx</c>): PTX text, target SM, the
/// argument ABI of the op family it implements, its launch configuration, and the producer's evidence. The registry
/// consumes it as an <see cref="TunedKernelOrigin.External"/> candidate; it is never trusted on its own evidence —
/// it becomes a default for a shape only after passing the same local correctness and paired-timing gate as every
/// other candidate.
/// </summary>
/// <remarks>
/// <para>Integrity: <see cref="PtxSha256"/> must equal the SHA-256 of <see cref="Ptx"/> (UTF-8, lowercase hex).
/// Content hashes establish integrity, not authenticity: load artifacts only from a directory protected against
/// untrusted writers (the same rule as <see cref="KernelTuningArtifactRegistry{TConfiguration}"/>).</para>
/// <para>ABIs (all fp32, row-major, contiguous):</para>
/// <list type="bullet">
/// <item><c>row-softmax-v1</c>: (const float* in, float* out, int rows, int n)</item>
/// <item><c>row-softmax-backward-v1</c>: (const float* dY, const float* Y, float* dX, int rows, int n)</item>
/// <item><c>layernorm-forward-v1</c>: (const float* x, float* y, const float* gamma, const float* beta, float* mean,
/// float* invStd, int rows, int n, float eps)</item>
/// <item><c>layernorm-backward-input-v1</c>: (const float* dY, const float* x, const float* gamma, const float* mean,
/// const float* invStd, float* dX, int rows, int n)</item>
/// <item><c>layernorm-grad-params-v1</c>: (const float* dY, const float* x, const float* mean, const float* invStd,
/// float* dGamma, float* dBeta, int rows, int n)</item>
/// <item><c>rmsnorm-grad-gamma-v1</c>: (const float* dY, const float* x, const float* rms, float* dGamma, int rows,
/// int n)</item>
/// </list>
/// </remarks>
public sealed class ExternalKernelArtifact
{
    /// <summary>The only schema version this build reads.</summary>
    public const int CurrentSchemaVersion = 1;

    /// <summary>ABI names and the op family each implements.</summary>
    public static readonly IReadOnlyDictionary<string, TunedKernelOp> Abis = new Dictionary<string, TunedKernelOp>(StringComparer.Ordinal)
    {
        ["row-softmax-v1"] = TunedKernelOp.Softmax,
        ["row-softmax-backward-v1"] = TunedKernelOp.SoftmaxBackward,
        ["layernorm-forward-v1"] = TunedKernelOp.LayerNorm,
        ["layernorm-backward-input-v1"] = TunedKernelOp.LayerNormBackward,
        ["layernorm-grad-params-v1"] = TunedKernelOp.LayerNormGradParameters,
        ["rmsnorm-grad-gamma-v1"] = TunedKernelOp.RmsNormGradGamma,
    };

    /// <summary>Schema version.</summary>
    [JsonPropertyName("schemaVersion")] public int SchemaVersion { get; set; }
    /// <summary>Candidate id, unique within its op family; prefixed "external." when consumed.</summary>
    [JsonPropertyName("id")] public string Id { get; set; } = string.Empty;
    /// <summary>Producer version of this kernel.</summary>
    [JsonPropertyName("version")] public string Version { get; set; } = string.Empty;
    /// <summary>Argument ABI; see the remarks.</summary>
    [JsonPropertyName("abi")] public string Abi { get; set; } = string.Empty;
    /// <summary>Minimum SM, e.g. "sm_75".</summary>
    [JsonPropertyName("target")] public string Target { get; set; } = string.Empty;
    /// <summary>Kernel entry point name in the PTX.</summary>
    [JsonPropertyName("entryPoint")] public string EntryPoint { get; set; } = string.Empty;
    /// <summary>PTX source text.</summary>
    [JsonPropertyName("ptx")] public string Ptx { get; set; } = string.Empty;
    /// <summary>SHA-256 of <see cref="Ptx"/>.</summary>
    [JsonPropertyName("ptxSha256")] public string PtxSha256 { get; set; } = string.Empty;
    /// <summary>Whether two runs on the same input are bit-identical (no atomics).</summary>
    [JsonPropertyName("deterministic")] public bool Deterministic { get; set; }
    /// <summary>Launch configuration.</summary>
    [JsonPropertyName("launch")] public ExternalKernelLaunch Launch { get; set; } = new();
    /// <summary>Shape limits.</summary>
    [JsonPropertyName("constraints")] public ExternalKernelConstraints Constraints { get; set; } = new();
    /// <summary>Producer evidence (informational).</summary>
    [JsonPropertyName("evidence")] public ExternalKernelEvidence Evidence { get; set; } = new();

    /// <summary>The op family the ABI implements.</summary>
    [JsonIgnore] public TunedKernelOp Op => Abis[Abi];

    /// <summary>The candidate id the registry uses.</summary>
    [JsonIgnore] public string CandidateId => "external." + Id + "@" + Version;

    /// <summary>The minimum SM as an integer (75 for "sm_75").</summary>
    [JsonIgnore]
    public int TargetSm =>
        Target.StartsWith("sm_", StringComparison.Ordinal) &&
        int.TryParse(Target.Substring(3), System.Globalization.NumberStyles.None,
            System.Globalization.CultureInfo.InvariantCulture, out int sm) ? sm : -1;

    /// <summary>Parses and validates an artifact document; throws <see cref="InvalidDataException"/> when invalid.</summary>
    public static ExternalKernelArtifact Parse(string json)
    {
        if (json is null) throw new ArgumentNullException(nameof(json));
        if (json.Length > 4 * 1024 * 1024) throw new InvalidDataException("Kernel artifact exceeds its 4 MiB bound.");
        ExternalKernelArtifact? artifact;
        try { artifact = JsonSerializer.Deserialize<ExternalKernelArtifact>(json, JsonOptions); }
        catch (JsonException ex) { throw new InvalidDataException("Kernel artifact is not valid JSON: " + ex.Message, ex); }
        if (artifact is null) throw new InvalidDataException("Empty kernel artifact.");
        artifact.Validate();
        return artifact;
    }

    /// <summary>Serializes the artifact (the producer side of the contract).</summary>
    public string ToJson() => JsonSerializer.Serialize(this, JsonOptions);

    /// <summary>Computes the hash <see cref="PtxSha256"/> must carry.</summary>
    public static string ComputePtxSha256(string ptx)
    {
        using var sha = SHA256.Create();
        return BitConverter.ToString(sha.ComputeHash(Encoding.UTF8.GetBytes(ptx))).Replace("-", string.Empty)
            .ToLowerInvariant();
    }

    /// <summary>Validates every field; throws <see cref="InvalidDataException"/> naming the first problem.</summary>
    public void Validate()
    {
        if (SchemaVersion != CurrentSchemaVersion)
            throw new InvalidDataException($"Unsupported kernel artifact schema {SchemaVersion}.");
        if (!IsToken(Id)) throw new InvalidDataException("Artifact id must be a non-empty [A-Za-z0-9._-] token.");
        if (!IsToken(Version)) throw new InvalidDataException("Artifact version must be a non-empty [A-Za-z0-9._-] token.");
        if (!Abis.ContainsKey(Abi)) throw new InvalidDataException($"Unknown kernel ABI '{Abi}'.");
        if (TargetSm < 0) throw new InvalidDataException($"Target '{Target}' is not of the form sm_NN.");
        if (!IsToken(EntryPoint)) throw new InvalidDataException("Entry point must be a non-empty identifier.");
        if (string.IsNullOrEmpty(Ptx) || Ptx.IndexOf('\0') >= 0) throw new InvalidDataException("PTX text is empty or contains NUL.");
        if (!string.Equals(ComputePtxSha256(Ptx), PtxSha256, StringComparison.Ordinal))
            throw new InvalidDataException("PTX SHA-256 does not match ptxSha256.");
        if (Ptx.IndexOf(".entry " + EntryPoint, StringComparison.Ordinal) < 0 &&
            Ptx.IndexOf(".entry\t" + EntryPoint, StringComparison.Ordinal) < 0)
            throw new InvalidDataException($"PTX does not declare entry point '{EntryPoint}'.");
        if (Launch is null) throw new InvalidDataException("Launch configuration is required.");
        if (Launch.BlockX <= 0 || Launch.BlockY <= 0 || (long)Launch.BlockX * Launch.BlockY > 1024)
            throw new InvalidDataException("Block dimensions must be positive with at most 1024 threads.");
        if (Launch.SharedMemoryBytes < 0 || Launch.SharedMemoryBytes > 48 * 1024)
            throw new InvalidDataException("Dynamic shared memory must be within 0..48 KiB.");
        if (Launch.UnitsPerBlock <= 0) throw new InvalidDataException("unitsPerBlock must be positive.");
        if (!Enum.IsDefined(typeof(ExternalKernelGridRule), Launch.GridRule))
            throw new InvalidDataException("Unknown grid rule.");
        if (Constraints is null) throw new InvalidDataException("Constraints are required.");
        if (Constraints.MinColumns < 1 || Constraints.MaxColumns < Constraints.MinColumns || Constraints.ColumnsMultipleOf < 1)
            throw new InvalidDataException("Invalid column constraints.");
    }

    /// <summary>Whether the artifact claims to support a (rows, columns) shape.</summary>
    public bool Supports(in TunedShape shape) =>
        shape.Count == 2 && shape.DType == TunedKernelDType.Float32 &&
        shape[1] >= Constraints.MinColumns && shape[1] <= Constraints.MaxColumns &&
        shape[1] % Constraints.ColumnsMultipleOf == 0;

    /// <summary>Loads every <c>*.kernel.json</c> artifact in a directory; invalid files are reported, not loaded.</summary>
    public static IReadOnlyList<ExternalKernelArtifact> LoadDirectory(string directory, ICollection<string>? errors = null)
    {
        var result = new List<ExternalKernelArtifact>();
        if (string.IsNullOrWhiteSpace(directory) || !Directory.Exists(directory)) return result;
        foreach (string file in Directory.GetFiles(directory, "*.kernel.json").OrderBy(f => f, StringComparer.Ordinal))
        {
            try
            {
                var info = new FileInfo(file);
                if ((info.Attributes & FileAttributes.ReparsePoint) != 0)
                    throw new InvalidDataException("Linked artifact paths are not supported.");
                result.Add(Parse(File.ReadAllText(file)));
            }
            catch (Exception ex) when (ex is IOException or InvalidDataException or UnauthorizedAccessException)
            {
                errors?.Add(Path.GetFileName(file) + ": " + ex.Message);
            }
        }
        return result;
    }

    private static bool IsToken(string? s)
    {
        if (s is null || s.Length == 0 || s.Length > 128) return false;
        foreach (char c in s)
            if (!(char.IsLetterOrDigit(c) && c < 128) && c != '_' && c != '.' && c != '-') return false;
        return true;
    }

    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        WriteIndented = true,
        MaxDepth = 16,
        Converters = { new JsonStringEnumConverter() },
    };
}
