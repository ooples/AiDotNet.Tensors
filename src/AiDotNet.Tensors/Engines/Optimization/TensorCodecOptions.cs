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

    /// <summary>
    /// Default options: dataflow fusion and algebraic backward enabled, spectral opt-in.
    /// Returns a fresh copy each time to prevent global state corruption — callers can
    /// freely mutate the returned instance and pass it to SetCurrent().
    /// </summary>
    public static TensorCodecOptions Default => new();

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

    /// <summary>Maximum approximation error per element for spectral decomposition.
    /// Used as energyThreshold = 1.0 - tolerance for SVD rank selection.</summary>
    public float SpectralErrorTolerance { get; set; } = 1e-5f;

    /// <summary>Maximum hidden dimension for dataflow fusion L1 residency.
    /// H * Mr * sizeof(float) must fit in L1 cache (32KB).</summary>
    public int DataflowFusionMaxHidden { get; set; } = 512;

    /// <summary>Phase 4.1: Fold BatchNorm into Conv2D weights at compile time.</summary>
    public bool EnableConvBnFusion { get; set; } = true;

    /// <summary>Phase 4.2: Fuse attention Q@K^T->Softmax->V patterns.</summary>
    public bool EnableAttentionFusion { get; set; } = true;

    /// <summary>Phase 4.3: Merge consecutive pointwise ops into fewer dispatch steps.</summary>
    public bool EnablePointwiseFusion { get; set; } = true;

    /// <summary>Phase 4.5: Precompute static subgraphs at compile time.</summary>
    public bool EnableConstantFolding { get; set; } = true;

    /// <summary>Phase 6.2: Deduplicate identical computations across layers.</summary>
    public bool EnableForwardCSE { get; set; } = true;

    /// <summary>Phase 7.1: Group independent MatMuls into batched calls.</summary>
    public bool EnableBlasBatch { get; set; } = true;

    /// <summary>Phase 7.3: Enable mixed precision (fp16 forward, fp32 backward). Opt-in.</summary>
    public bool EnableMixedPrecision { get; set; }
}
