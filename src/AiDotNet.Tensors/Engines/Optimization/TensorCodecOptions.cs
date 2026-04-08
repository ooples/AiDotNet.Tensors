namespace AiDotNet.Tensors.Engines.Optimization;

/// <summary>
/// Configuration for TensorCodec optimization passes.
/// Controls which radical computation transformations are enabled.
/// Thread-static ambient context following GradientTape/GraphMode pattern.
/// </summary>
public sealed class TensorCodecOptions
{
    [ThreadStatic]
    private static TensorCodecOptions? _current;

    /// <summary>Gets the active TensorCodec options for this thread, or the default.</summary>
    public static TensorCodecOptions Current => _current ?? Default;

    /// <summary>Default options: dataflow fusion and algebraic backward enabled, spectral opt-in.</summary>
    public static readonly TensorCodecOptions Default = new();

    /// <summary>Sets thread-local options. Pass null to revert to defaults.</summary>
    public static void SetCurrent(TensorCodecOptions? options) => _current = options;

    /// <summary>Master switch for auto-compilation. When false, all compilation is disabled
    /// and execution falls through to the eager path.</summary>
    public bool EnableCompilation { get; set; } = true;

    /// <summary>Phase B: Fuse consecutive linear layers into a single multi-layer kernel
    /// that keeps data in registers/L1 across layer boundaries.</summary>
    public bool EnableDataflowFusion { get; set; } = true;

    /// <summary>Phase C: Symbolically simplify the backward graph at compile time
    /// using CSE, double-transpose elimination, and associative regrouping.</summary>
    public bool EnableAlgebraicBackward { get; set; } = true;

    /// <summary>Phase A: SVD-factorize frozen weight matrices for faster inference.
    /// Opt-in because it introduces bounded approximation error.</summary>
    public bool EnableSpectralDecomposition { get; set; }

    /// <summary>Maximum approximation error per element for spectral decomposition.</summary>
    public float SpectralErrorTolerance { get; set; } = 1e-5f;

    /// <summary>Minimum rank reduction factor to justify spectral decomposition (2 = 50% rank reduction).</summary>
    public int SpectralMinRankReduction { get; set; } = 2;

    /// <summary>Maximum hidden dimension for dataflow fusion L1 residency.
    /// H * Mr * sizeof(float) must fit in L1 cache (32KB).</summary>
    public int DataflowFusionMaxHidden { get; set; } = 512;

    /// <summary>Phase 7.3: Enable mixed precision (fp16 forward, fp32 backward). Opt-in.</summary>
    public bool EnableMixedPrecision { get; set; }
}
