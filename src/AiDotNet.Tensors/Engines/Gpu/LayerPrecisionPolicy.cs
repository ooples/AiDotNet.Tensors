namespace AiDotNet.Tensors.Engines.Gpu;

/// <summary>
/// Per-layer precision overrides for mixed-precision training. Layers whose
/// name matches an exact entry or substring pattern run at the overridden
/// precision; the rest inherit <see cref="DefaultPrecision"/>.
///
/// <para>Exact-name matches win over substring patterns when both apply
/// (so a policy that says "keep 'bn1' in FP32 but the pattern '*norm*' in
/// FP16" does what you'd expect — 'bn1' stays FP32).</para>
///
/// <para>Default factory presets — <see cref="ForFP16"/>, <see cref="ForBF16"/>,
/// <see cref="ForFP8"/> — keep normalization / softmax / loss / embedding
/// layers in FP32, matching PyTorch's <c>torch.cuda.amp.autocast</c>
/// allowlist semantics.</para>
/// </summary>
public sealed class LayerPrecisionPolicy
{
    private readonly Dictionary<string, PrecisionMode> _exact =
        new(StringComparer.OrdinalIgnoreCase);
    private readonly List<(string pattern, PrecisionMode mode)> _patterns = new();

    /// <summary>Precision for layers not matched by any exact entry or pattern.</summary>
    public PrecisionMode DefaultPrecision { get; }

    /// <summary>Construct a policy with the supplied default precision.</summary>
    public LayerPrecisionPolicy(PrecisionMode defaultPrecision = PrecisionMode.Float16)
    {
        DefaultPrecision = defaultPrecision;
    }

    /// <summary>
    /// Sets precision for an exact layer name. Overrides any substring
    /// pattern that would otherwise match.
    /// </summary>
    public LayerPrecisionPolicy SetPrecision(string exactLayerName, PrecisionMode precision)
    {
        if (string.IsNullOrEmpty(exactLayerName))
            throw new ArgumentException("Layer name cannot be null or empty.", nameof(exactLayerName));
        _exact[exactLayerName] = precision;
        return this;
    }

    /// <summary>
    /// Adds a case-insensitive substring match. Any layer whose name contains
    /// <paramref name="substringPattern"/> gets <paramref name="precision"/>
    /// unless an exact entry overrides it.
    /// </summary>
    public LayerPrecisionPolicy AddPattern(string substringPattern, PrecisionMode precision)
    {
        if (string.IsNullOrEmpty(substringPattern))
            throw new ArgumentException("Pattern cannot be null or empty.", nameof(substringPattern));
        _patterns.Add((substringPattern, precision));
        return this;
    }

    /// <summary>Syntactic sugar — keeps layers matching <paramref name="pattern"/> in FP32.</summary>
    public LayerPrecisionPolicy KeepInFP32(string pattern)
        => AddPattern(pattern, PrecisionMode.Float32);

    /// <summary>Returns the effective precision for <paramref name="layerName"/>.</summary>
    public PrecisionMode GetLayerPrecision(string layerName)
    {
        if (string.IsNullOrEmpty(layerName)) return DefaultPrecision;
        if (_exact.TryGetValue(layerName, out var exactMode))
            return exactMode;
        for (int i = 0; i < _patterns.Count; i++)
        {
            var (pattern, mode) = _patterns[i];
            if (layerName.IndexOf(pattern, StringComparison.OrdinalIgnoreCase) >= 0)
                return mode;
        }
        return DefaultPrecision;
    }

    /// <summary>
    /// True iff this layer should SKIP mixed-precision autocast — i.e. it
    /// resolves to <see cref="PrecisionMode.Float32"/>. Used by the training
    /// loop to bypass autocast bookkeeping for FP32-pinned layers.
    /// </summary>
    public bool ShouldSkipMixedPrecision(string layerName)
        => GetLayerPrecision(layerName) == PrecisionMode.Float32;

    /// <summary>
    /// Default FP16 mixed-precision policy: compute layers in FP16, keep
    /// normalisation / softmax / loss / embedding in FP32. Matches the
    /// PyTorch <c>autocast</c> allowlist semantics.
    /// </summary>
    public static LayerPrecisionPolicy ForFP16() => BuildDefault(PrecisionMode.Float16);

    /// <summary>BF16 variant of <see cref="ForFP16"/>.</summary>
    public static LayerPrecisionPolicy ForBF16() => BuildDefault(PrecisionMode.BFloat16);

    /// <summary>FP8 variant of <see cref="ForFP16"/>. Keeps the same
    /// FP32 allowlist but defaults everything else to FP8.</summary>
    public static LayerPrecisionPolicy ForFP8() => BuildDefault(PrecisionMode.Float8E4M3);

    private static LayerPrecisionPolicy BuildDefault(PrecisionMode compute)
    {
        var policy = new LayerPrecisionPolicy(compute);
        // Stability-critical layers — reductions and large-range ops — stay FP32.
        policy.KeepInFP32("norm");      // batchnorm, layernorm, groupnorm, instancenorm
        policy.KeepInFP32("softmax");
        policy.KeepInFP32("loss");
        policy.KeepInFP32("embedding"); // large vocab matmuls, stability matters
        return policy;
    }
}
