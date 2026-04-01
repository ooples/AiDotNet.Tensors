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
    BFloat16
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
    /// Creates a new autocast scope with the specified precision.
    /// </summary>
    /// <param name="precision">The target precision for GPU operations.</param>
    public AutocastScope(PrecisionMode precision = PrecisionMode.Float16)
    {
        Precision = precision;
        _previous = _current;
        _current = this;
    }

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
    }
}
