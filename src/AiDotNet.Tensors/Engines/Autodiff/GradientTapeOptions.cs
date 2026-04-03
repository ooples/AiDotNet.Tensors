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

    /// <summary>
    /// Whether to enable graph caching across training steps.
    /// When enabled, the tape builds a rolling hash of the op sequence during recording.
    /// If the same graph structure is seen again, the backward traversal plan is reused
    /// (skipping dead-node elimination and reachability analysis).
    /// </summary>
    public bool EnableGraphCaching { get; init; }

    /// <summary>
    /// Maximum number of cached graphs when <see cref="EnableGraphCaching"/> is true.
    /// Default 4 handles train + validation + different batch sizes.
    /// </summary>
    public int GraphCacheCapacity { get; init; } = 4;
}
