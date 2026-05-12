namespace AiDotNet.Tensors.Engines.Autodiff;

/// <summary>
/// Configuration options for <see cref="GradientTape{T}"/>.
/// Properties use init-only setters to prevent mutation of the shared <see cref="Default"/> instance.
/// </summary>
public sealed class GradientTapeOptions
{
    /// <summary>
    /// Default options: <see cref="Persistent"/> = <c>true</c>, unlimited entries,
    /// records in-place ops.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <c>Persistent = true</c> by default gates the <c>AutoTrainingCompiler</c>
    /// fast path (see <see cref="AiDotNet.Tensors.Engines.Compilation.AutoTrainingCompiler"/>).
    /// On profile data from a single VGG11 / ResNet50 Train step,
    /// <c>ComputeGradients</c> dominated 65–73 % of wall time when
    /// non-persistent; turning persistent on lets the compiler replay a
    /// flat-indexed compiled backward graph instead of walking the tape
    /// entry list + dictionary-keyed gradient lookups per op, cutting that
    /// fraction substantially.
    /// </para>
    /// <para>
    /// Clone-then-train safety is handled inside <c>AutoTrainingCompiler</c>
    /// via two cooperating mechanisms: (1) the cache lookup key composes
    /// <c>ComputeStructureHash</c> (op + shape + element type) with
    /// <c>IncludeTargetIdentity</c> (<c>RuntimeHelpers.GetHashCode</c> over
    /// <c>sources</c>) so a cloned model with fresh parameter tensors
    /// produces a different cache key and triggers a fresh compile rather
    /// than replaying the original's backward plan on the wrong tensors;
    /// (2) <c>TryCompileBackward</c> refuses to store a plan when
    /// <c>sources</c> is null, so the structure-only fallback can never be
    /// cached and a null-sources caller cannot accidentally hit a
    /// clone-mismatched plan.
    /// </para>
    /// </remarks>
    public static readonly GradientTapeOptions Default = new() { Persistent = true };

    /// <summary>
    /// Whether the tape is persistent (can compute gradients multiple times).
    /// When false, <see cref="GradientTape{T}.ComputeGradients"/> clears the tape after use.
    /// Defaults to <c>true</c> (see <see cref="Default"/> remarks).
    /// </summary>
    public bool Persistent { get; init; } = true;

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
