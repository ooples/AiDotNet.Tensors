namespace AiDotNet.Tensors.Engines.Autodiff;

/// <summary>
/// Configuration options for <see cref="GradientTape{T}"/>.
/// Properties use init-only setters to prevent mutation of the shared <see cref="Default"/> instance.
/// </summary>
public sealed class GradientTapeOptions
{
    /// <summary>
    /// Default options: non-persistent, unlimited entries, records in-place ops.
    /// </summary>
    public static readonly GradientTapeOptions Default = new();

    /// <summary>
    /// Whether the tape is persistent (can compute gradients multiple times).
    /// When false, <see cref="GradientTape{T}.ComputeGradients"/> clears the tape after use.
    /// </summary>
    public bool Persistent { get; init; }

    /// <summary>
    /// Maximum number of tape entries. 0 means unlimited.
    /// When the limit is reached, the oldest entries are discarded.
    /// </summary>
    public int MaxEntries { get; init; }

    /// <summary>
    /// Whether to record in-place operations by saving a copy of the input before mutation.
    /// When false, in-place operations are not recorded (gradients will not flow through them).
    /// </summary>
    public bool RecordInPlace { get; init; } = true;

    /// <summary>
    /// Whether to enable tensor hooks (RegisterHook, RetainGrad).
    /// Disabled by default to avoid dictionary allocation overhead.
    /// </summary>
    public bool EnableHooks { get; init; }
}
