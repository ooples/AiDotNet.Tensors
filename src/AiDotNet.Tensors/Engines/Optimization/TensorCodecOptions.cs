using AiDotNet.Tensors.Helpers;

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

    /// <summary>
    /// Sets thread-local options. Pass null to revert to defaults.
    /// <para>
    /// Also installs a thread-local BLAS determinism override reflecting
    /// <paramref name="options"/>'s <see cref="Deterministic"/> flag — so a caller
    /// who does <c>SetCurrent(new TensorCodecOptions { Deterministic = false })</c>
    /// actually observes the override end-to-end (it threads through the cache key
    /// and any future determinism-divergent backend), without affecting any other
    /// thread's policy. Passing null clears the override and the thread inherits
    /// the process-wide default again.
    /// </para>
    /// <para>
    /// <b>Mutation caveat:</b> mutating <c>Current.Deterministic</c> after
    /// <c>SetCurrent</c> does NOT re-sync the BLAS override — the sync point is
    /// <c>SetCurrent</c> itself. Call <c>SetCurrent</c> again after any post-install
    /// mutation to take effect.
    /// </para>
    /// </summary>
    public static void SetCurrent(TensorCodecOptions? options)
    {
        _current = options;
        BlasProvider.SetThreadLocalDeterministicMode(options?.Deterministic);
    }

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

    /// <summary>
    /// When true, tensor operations route through deterministic code paths — floating-point
    /// reductions (matmul, softmax, dot products, etc.) produce bit-identical results across
    /// runs on the same hardware, regardless of thread count. Defaults to <c>true</c>.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>Design note — why default-on:</b> after the MKL.NET removal in #131/#163, every
    /// CPU matmul routes through SimdGemm's bit-exact AVX2 blocked kernel; deterministic
    /// matmul therefore costs nothing relative to the non-deterministic alternative.
    /// Defaulting to deterministic gives reproducible training runs, unit tests that don't
    /// drift, and CUDA-graph-safe kernels out of the box — with no performance tax to
    /// justify keeping the user out of that mode.
    /// </para>
    /// <para>
    /// <b>How to opt out:</b> set <c>Deterministic = false</c> on a <see cref="TensorCodecOptions"/>
    /// instance and install it via <see cref="SetCurrent"/> for thread-local effect, or
    /// call <c>AiDotNetEngine.SetDeterministicMode(false)</c> for process-wide effect.
    /// </para>
    /// <para>
    /// <b>Cache invalidation:</b> the compile cache mixes the current deterministic state
    /// (process-wide value or thread-local override, whichever wins) into the plan key,
    /// so switching this setting invalidates any plans compiled under the opposite setting
    /// automatically — no manual cache clear required.
    /// </para>
    /// </remarks>
    public bool Deterministic { get; set; } = true;

    /// <summary>
    /// When true (default) <see cref="Engines.DirectGpu.DirectGpuEngine.Conv2D"/>
    /// is eligible for cuDNN dispatch on a CUDA backend when the cuDNN
    /// GPU-pointer wiring is active and the runtime has cuDNN available; set
    /// to false to force the generic CUDA kernel. Opt-out exists mostly for
    /// debugging and for reproducing numerical behaviour that differs between
    /// cuDNN and the hand-written kernel (~ULP at the last accumulation).
    /// <para><b>Current runtime behaviour:</b> this flag expresses intent —
    /// Conv2D still executes the generic CUDA kernel on all paths until the
    /// cuDNN GPU-pointer wiring lands. When that wiring ships, flipping this
    /// to <c>false</c> will force the existing generic kernel and the default
    /// <c>true</c> will start routing through cuDNN.</para>
    /// </summary>
    public bool UseCudnn { get; set; } = true;

    /// <summary>
    /// When true (default) <see cref="Engines.DirectGpu.DirectGpuEngine.BatchNorm"/>
    /// is eligible for cuDNN BatchNorm dispatch on a CUDA backend when the
    /// cuDNN GPU-pointer wiring is active and the runtime has cuDNN available;
    /// set to false to force the generic kernel.
    /// <para><b>Current runtime behaviour:</b> like <see cref="UseCudnn"/>,
    /// this is a policy flag — BatchNorm still executes the generic CUDA
    /// kernel until the cuDNN GPU-pointer wiring lands.</para>
    /// </summary>
    public bool UseCudnnBatchNorm { get; set; } = true;

    /// <summary>
    /// When true (default) matmul / batched GEMM routes through the cuBLAS
    /// wrapper on a CUDA backend when cuBLAS is available; set to false to
    /// force the generic kernel.
    /// </summary>
    public bool UseCublas { get; set; } = true;
}
