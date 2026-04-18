using AiDotNet.Tensors.Engines.DirectGpu;

namespace AiDotNet.Tensors.Engines.Gpu;

/// <summary>
/// Precision mode for mixed precision training.
/// </summary>
public enum PrecisionMode
{
    /// <summary>Full fp32 precision (default).</summary>
    Float32,

    /// <summary>Half precision fp16 — 2x memory savings, faster on Tensor Cores.</summary>
    Float16,

    /// <summary>Brain floating point bf16 — same range as fp32, less precision. Preferred for training.</summary>
    BFloat16,

    /// <summary>FP8 E4M3 — 4x memory savings vs fp32, forward-pass default on NVIDIA H100+.</summary>
    Float8E4M3,

    /// <summary>FP8 E5M2 — wider range, less precision. Backward-pass default on NVIDIA H100+.</summary>
    Float8E5M2,
}

/// <summary>
/// Enables automatic mixed precision for GPU operations, equivalent to PyTorch's
/// <c>torch.cuda.amp.autocast()</c>. Within this scope, eligible operations automatically
/// use fp16/bf16 for computation while accumulating gradients in fp32.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Mixed precision training uses lower precision (fp16) for most
/// computations to run faster and use less memory, while keeping a full-precision (fp32)
/// copy of weights for accurate gradient updates. This can nearly double training speed
/// on GPUs with Tensor Cores (NVIDIA Volta and newer).</para>
///
/// <para><b>Usage:</b></para>
/// <code>
/// using var autocast = new AutocastScope(PrecisionMode.Float16);
/// var output = model.Forward(input);  // Runs in fp16
/// // Gradients are accumulated in fp32 automatically
/// </code>
/// </remarks>
public sealed class AutocastScope : IDisposable
{
    [ThreadStatic]
    private static AutocastScope? _current;

    private readonly AutocastScope? _previous;
    private bool _disposed;

    /// <summary>
    /// Gets the currently active autocast scope, or null if none.
    /// </summary>
    public static AutocastScope? Current => _current;

    /// <summary>
    /// Gets the precision mode for this scope.
    /// </summary>
    public PrecisionMode Precision { get; }

    /// <summary>
    /// Gets whether autocast is currently enabled (any scope is active).
    /// </summary>
    public static bool IsEnabled => _current is not null;

    /// <summary>
    /// Gets the active precision mode, defaulting to Float32 when no scope is active.
    /// </summary>
    public static PrecisionMode ActivePrecision => _current?.Precision ?? PrecisionMode.Float32;

    /// <summary>
    /// Optional per-layer precision policy. When non-null,
    /// <see cref="ShouldAutocast"/> defers to it for any call whose op name
    /// also appears as a layer name — giving the user fine-grained control
    /// over which layers keep FP32 compute even under a global FP16 scope.
    /// </summary>
    public LayerPrecisionPolicy? Policy { get; }

    // Per-name FP32 / FP16 tensor cache. Lets layers store an FP32 master
    // weight alongside an FP16 compute copy, both keyed by the same name.
    // The cache is scope-local — Dispose clears it.
    private readonly Dictionary<string, LinearAlgebra.Tensor<float>> _fp32Cache = new();
    private readonly Dictionary<string, LinearAlgebra.Tensor<Half>> _fp16Cache = new();

    /// <summary>
    /// Creates a new autocast scope with the specified precision.
    /// </summary>
    /// <param name="precision">The target precision for GPU operations.</param>
    public AutocastScope(PrecisionMode precision = PrecisionMode.Float16)
        : this(precision, policy: null)
    {
    }

    /// <summary>
    /// Creates a new autocast scope with the specified precision and
    /// optional per-layer precision policy.
    /// </summary>
    /// <param name="precision">The target precision for GPU operations.</param>
    /// <param name="policy">Per-layer overrides. Null means the scope-wide
    /// <paramref name="precision"/> applies to every layer.</param>
    public AutocastScope(PrecisionMode precision, LayerPrecisionPolicy? policy)
    {
        if (precision == PrecisionMode.BFloat16)
            throw new NotSupportedException("BFloat16 autocast is not yet implemented. Use Float16 instead.");
        Precision = precision;
        Policy = policy;
        _previous = _current;
        _current = this;
    }

    /// <summary>
    /// Registers <paramref name="fp32"/> as the FP32 master copy for
    /// <paramref name="name"/>, casts it to FP16 compute copy, and caches
    /// both. Subsequent calls with the same name return the cached FP16
    /// copy without re-casting.
    /// </summary>
    public LinearAlgebra.Tensor<Half> RegisterAndCastToFP16(string name, LinearAlgebra.Tensor<float> fp32)
    {
        if (string.IsNullOrEmpty(name)) throw new ArgumentException("Name required.", nameof(name));
        if (fp32 is null) throw new ArgumentNullException(nameof(fp32));
        if (_disposed) throw new ObjectDisposedException(nameof(AutocastScope));

        _fp32Cache[name] = fp32;
        if (_fp16Cache.TryGetValue(name, out var existing)) return existing;

        var fp32Data = fp32.GetDataArray();
        var fp16Tensor = new LinearAlgebra.Tensor<Half>(fp32._shape);
        var fp16Span = fp16Tensor.AsWritableSpan();
        for (int i = 0; i < fp32Data.Length; i++)
            fp16Span[i] = (Half)fp32Data[i];
        _fp16Cache[name] = fp16Tensor;
        return fp16Tensor;
    }

    /// <summary>Returns the cached FP32 master tensor for <paramref name="name"/>, or null.</summary>
    public LinearAlgebra.Tensor<float>? GetFP32Tensor(string name)
        => _fp32Cache.TryGetValue(name, out var t) ? t : null;

    /// <summary>Returns the cached FP16 compute tensor for <paramref name="name"/>, or null.</summary>
    public LinearAlgebra.Tensor<Half>? GetFP16Tensor(string name)
        => _fp16Cache.TryGetValue(name, out var t) ? t : null;

    /// <summary>True iff a tensor is registered under <paramref name="name"/>.</summary>
    public bool HasTensor(string name) => _fp32Cache.ContainsKey(name);

    /// <summary>Clears the per-name cache. Called automatically on <see cref="Dispose"/>.</summary>
    public void ClearTensors()
    {
        _fp32Cache.Clear();
        _fp16Cache.Clear();
    }

    /// <summary>
    /// True iff the configured policy forces <paramref name="layerName"/>
    /// into FP32. Consumed by training-loop orchestration to bypass
    /// autocast bookkeeping for FP32-pinned layers (matches PyTorch's
    /// <c>torch.cuda.amp.autocast(enabled=False)</c> nested scope).
    /// </summary>
    public bool ShouldUseFP32(string layerName)
        => Policy?.GetLayerPrecision(layerName) == PrecisionMode.Float32;

    /// <summary>
    /// True iff the layer's effective precision is MORE precise than the
    /// scope-wide <see cref="Precision"/>. Layers matching this predicate
    /// run in FP32 / FP16 / BF16 rather than the default. Typical callers:
    /// training-loop wrappers that decide whether to insert a cast around
    /// a layer's compute.
    /// </summary>
    public bool ShouldUseHigherPrecision(string layerName)
    {
        if (Policy is null) return false;
        var layerPrec = Policy.GetLayerPrecision(layerName);
        return PrecisionRank(layerPrec) > PrecisionRank(Precision);
    }

    /// <summary>Cast an FP16 tensor to FP32 (pure-CPU widen).</summary>
    public static LinearAlgebra.Tensor<float> CastToFP32(LinearAlgebra.Tensor<Half> fp16)
    {
        if (fp16 is null) throw new ArgumentNullException(nameof(fp16));
        var result = new LinearAlgebra.Tensor<float>(fp16._shape);
        var src = fp16.AsSpan();
        var dst = result.AsWritableSpan();
        for (int i = 0; i < src.Length; i++) dst[i] = (float)src[i];
        return result;
    }

    /// <summary>Cast an FP32 tensor to FP16 (pure-CPU narrow).</summary>
    public static LinearAlgebra.Tensor<Half> CastToFP16(LinearAlgebra.Tensor<float> fp32)
    {
        if (fp32 is null) throw new ArgumentNullException(nameof(fp32));
        var result = new LinearAlgebra.Tensor<Half>(fp32._shape);
        var src = fp32.AsSpan();
        var dst = result.AsWritableSpan();
        for (int i = 0; i < src.Length; i++) dst[i] = (Half)src[i];
        return result;
    }

    // Higher rank = more precise. Used by ShouldUseHigherPrecision to
    // compare layer vs scope precision.
    private static int PrecisionRank(PrecisionMode m) => m switch
    {
        PrecisionMode.Float32     => 4,
        PrecisionMode.BFloat16    => 3,
        PrecisionMode.Float16     => 2,
        PrecisionMode.Float8E5M2  => 1,
        PrecisionMode.Float8E4M3  => 0,
        _                          => 0,
    };

    /// <summary>
    /// Operations that benefit from fp16 compute (PyTorch allowlist).
    /// Norms, losses, softmax, and reductions must stay fp32 for numerical stability.
    /// </summary>
    private static readonly HashSet<string> _fp16AllowedOps = new(StringComparer.OrdinalIgnoreCase)
    {
        "MatMul", "BatchMatMul", "Gemm", "Linear", "FusedLinear",
        "Conv1D", "Conv2D", "Conv3D", "ConvTranspose2D", "DepthwiseConv2D",
        "Add", "Subtract", "Multiply", "Divide",
        "ReLU", "GELU", "SiLU", "Swish", "LeakyReLU", "Sigmoid", "Tanh",
        "Mish", "HardSwish", "ELU", "SELU", "Softplus", "HardSigmoid",
        "Dropout", "Embedding"
    };

    /// <summary>
    /// Returns true if the given operation should use fp16 when autocast is active.
    /// Norms, losses, softmax, and reductions always stay fp32.
    /// </summary>
    public static bool ShouldAutocast(string operationName)
    {
        if (!IsEnabled || ActivePrecision == PrecisionMode.Float32)
            return false;
        return _fp16AllowedOps.Contains(operationName);
    }

    /// <summary>
    /// Converts a fp32 GPU buffer to the active precision for computation.
    /// Returns null if no conversion is needed. Caller MUST dispose the returned buffer.
    /// </summary>
    public static IGpuBuffer? MaybeConvertInput(IDirectGpuBackend backend, IGpuBuffer fp32Buffer, int size)
    {
        if (!IsEnabled || ActivePrecision == PrecisionMode.Float32)
            return null;

        var fp16Buffer = backend.AllocateBuffer(size);
        backend.ConvertToFp16(fp32Buffer, fp16Buffer, size);
        return fp16Buffer;
    }

    /// <summary>
    /// Converts a result buffer back to fp32 for gradient accumulation.
    /// </summary>
    public static IGpuBuffer? MaybeConvertOutput(IDirectGpuBackend backend, IGpuBuffer resultBuffer, int size)
    {
        if (!IsEnabled || ActivePrecision == PrecisionMode.Float32)
            return null;

        var fp32Buffer = backend.AllocateBuffer(size);
        backend.ConvertToFp32(resultBuffer, fp32Buffer, size);
        return fp32Buffer;
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _current = _previous;
        ClearTensors();
    }
}
