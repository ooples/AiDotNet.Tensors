namespace AiDotNet.Tensors.Engines.Autodiff;

/// <summary>
/// Controls the lifetime of the recorded graph after a streaming backward pass.
/// </summary>
public enum StreamingGraphRetentionMode
{
    /// <summary>
    /// Release activations as soon as their backward step has consumed them. This is the
    /// memory-bounded default and makes the recorded graph unavailable for another backward.
    /// </summary>
    ReleaseAfterBackward = 0,

    /// <summary>
    /// Keep the recorded graph until the tape is disposed so the same deterministic graph can
    /// be replayed. Use this for algorithms that must inspect gradients before committing updates.
    /// </summary>
    RetainUntilTapeDisposal = 1,
}

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
    /// When the limit is reached, further operations are not recorded (the entries already on the tape are kept), so
    /// gradients do not flow through ops recorded past the limit. A create-graph backward that would differentiate a
    /// composite of recorded ops falls back to the op's first-order kernel when the composite does not fit, so every
    /// input still gets its exact gradient (not differentiable again, as nothing more fits on the tape).
    /// </summary>
    public int MaxEntries { get; init; }

    /// <summary>
    /// Whether to record in-place operations by saving a copy of the input before mutation.
    /// When false, in-place operations are not recorded (gradients will not flow through them).
    /// </summary>
    public bool RecordInPlace { get; init; } = true;

    /// <summary>
    /// Controls whether <see cref="GradientTape{T}.ComputeGradientsStreaming"/> releases the
    /// recorded graph after each backward or retains it for replay until tape disposal.
    /// Defaults to <see cref="StreamingGraphRetentionMode.ReleaseAfterBackward"/>.
    /// </summary>
    public StreamingGraphRetentionMode StreamingGraphRetention { get; init; }
        = StreamingGraphRetentionMode.ReleaseAfterBackward;

    /// <summary>
    /// Precision used only for fan-out gradient additions. Backward kernels continue to use the
    /// active autocast precision. Defaults to FP32 accumulation for numerical stability.
    /// </summary>
    public GradientAccumulationPrecision GradientAccumulationPrecision { get; init; }
        = GradientAccumulationPrecision.Float32;

    /// <summary>
    /// Whether to enable tensor hooks (RegisterHook, RetainGrad).
    /// Disabled by default to avoid dictionary allocation overhead.
    /// </summary>
    public bool EnableHooks { get; init; }

    /// <summary>
    /// When true, this tape never owns/creates a <see cref="AiDotNet.Tensors.Helpers.TensorArena"/>
    /// and never <c>Reset()</c>s the active arena on Dispose. Set for transient INNER tapes created
    /// DURING an outer tape's backward pass — notably the <c>GradientCheckpointing</c> recompute tape.
    /// Such a tape borrows the active arena for scratch, but must NOT manage its step lifecycle:
    /// because the outer tape nulls the thread-current tape while walking backward, an inner tape sees
    /// <c>_parent == null</c> and would otherwise be mistaken for a step-boundary tape and reset the
    /// OUTER tape's arena mid-backward, corrupting every accumulating gradient (#734).
    /// </summary>
    public bool SuppressArenaScope { get; init; }

    /// <summary>
    /// When true, a CPU backward hands each forward activation's memory back to the step's arena after the last
    /// backward step that reads it, so later backward work reuses it instead of growing the step's memory.
    /// </summary>
    /// <remarks>
    /// <para>Without it, a training step holds every activation until the step ends, so its memory is the forward's
    /// activations plus everything backward rents. PyTorch frees each activation once backward has used it.</para>
    /// <para><b>Only set this when nothing reads the forward's tensors after backward</b>, other than the loss and the
    /// returned gradients, which stay valid. A prediction, feature map or any other forward output read after
    /// <c>ComputeGradients</c> may hold another tensor's data. A second backward on the same tape throws, because its
    /// activations are gone. Applies to a top-level tape on a CPU engine inside a <see cref="Helpers.TensorArena"/>
    /// (a tape opens one by default); elsewhere it does nothing.</para>
    /// </remarks>
    public bool ReleaseActivationsDuringBackward { get; init; }

    /// <summary>
    /// When true, a GPU backward frees each device-only forward intermediate right after the last backward step that
    /// reads it, with no download, instead of leaving it for the garbage collector.
    /// </summary>
    /// <remarks>
    /// <para>Off by default, so a tape never invalidates a tensor its caller still holds: PyTorch keeps a forward output
    /// valid for as long as Python references it. .NET cannot see those references without a collection, so freeing at
    /// last use is an opt-in for loops that own their forward's tensors (a training step), where it returns 30-40% of a
    /// step's dead device memory without the blocking collections waiting for the GC costs.</para>
    /// <para>An opted-in graph's first backward run is a probe: it frees nothing, and any of its intermediates read
    /// after that backward, until the same graph runs backward again, are kept on every later run. A read the probe
    /// could not see throws an error naming the producing op; keep such a tensor with
    /// <see cref="GradientTape{T}.Retain(AiDotNet.Tensors.LinearAlgebra.Tensor{T})"/>. The loss, the sources and the
    /// returned gradients are never freed. On a CPU engine it does nothing (see
    /// <see cref="ReleaseActivationsDuringBackward"/>).</para>
    /// </remarks>
    public bool ReleaseDeviceActivationsAtLastUse { get; init; }

    /// <summary>
    /// When true, the loss and gradients this tape's backward returned go back to the step's arena when the tape is
    /// disposed, instead of being held out of reuse for one more step.
    /// </summary>
    /// <remarks>
    /// <para>The hold exists for callers that read the gradients after disposing the tape (an optimizer step or a
    /// gradient clip outside its scope, issue #1031). It costs a second set of gradient buffers: while step N runs,
    /// step N-1's are still held, so a training loop's memory carries every parameter gradient twice.</para>
    /// <para><b>Only set this when the update finishes inside the tape's scope</b> and nothing keeps the returned
    /// tensors past it. Applies to a top-level tape.</para>
    /// </remarks>
    public bool ReleaseResultsAtDispose { get; init; }
}
