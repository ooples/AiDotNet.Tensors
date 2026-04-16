using System.IO;
using System.Threading;
using System.Threading.Tasks;
using AiDotNet.Tensors.Engines.Compilation.Serialization;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Compilation;

/// <summary>
/// A compiled inference plan that replays a pre-optimized computation graph.
/// Compile once from a traced forward pass, replay forever with zero allocation.
///
/// Plans are cached by input shape — recompilation occurs automatically
/// when input dimensions change.
/// </summary>
/// <typeparam name="T">The tensor element type.</typeparam>
public interface ICompiledPlan<T> : IDisposable
{
    /// <summary>Executes the compiled plan and returns the output tensor.</summary>
    Tensor<T> Execute();

    /// <summary>Checks whether this plan is valid for the given input shape.</summary>
    bool IsValid(int[] inputShape);

    /// <summary>Number of compiled execution steps.</summary>
    int StepCount { get; }

    /// <summary>
    /// Composes this plan with <paramref name="next"/> into a single stitched
    /// plan whose <see cref="Execute"/> runs <i>this</i>'s steps followed by
    /// <i>next</i>'s steps as one flat delegate array. The intermediate
    /// tensor (this plan's final output) shares backing storage with next's
    /// captured input after the call, so data flows through without any new
    /// <see cref="Tensor{T}"/> materialization or per-execute copy.
    /// Production pipelines like
    /// <c>tokenizer → encoder → transformer → classifier</c> become one
    /// compiled kernel with no inter-plan materialization overhead.
    /// </summary>
    /// <param name="next">The plan to run after this one. Its captured input
    /// shape must equal this plan's final output shape — validated at stitch
    /// time, not at execute time.</param>
    /// <returns>A new <see cref="ICompiledPlan{T}"/> whose
    /// <see cref="Execute"/> is the composed pipeline. The stitched result
    /// holds strong references to <c>this</c> and <paramref name="next"/>
    /// (keeping them GC-reachable for the stitched plan's lifetime), but
    /// does <b>not</b> take disposal ownership — disposing the stitched
    /// plan releases its own resources and leaves the originals operable.
    /// This lets a single plan participate in multiple stitched pipelines.
    /// Disposing the originals <i>before</i> the stitched plan, however, is
    /// a contract violation and will cause the stitched plan's
    /// <see cref="Execute"/> to throw <see cref="ObjectDisposedException"/>
    /// (rather than silently corrupt through freed GCHandles).</returns>
    /// <exception cref="ArgumentNullException"><paramref name="next"/> is null.</exception>
    /// <exception cref="ObjectDisposedException"><c>this</c> or
    /// <paramref name="next"/> has already been disposed.</exception>
    /// <exception cref="ArgumentException">The output shape of this plan
    /// does not match the input shape of <paramref name="next"/>; or either
    /// plan has no steps to compose.</exception>
    /// <exception cref="NotSupportedException"><paramref name="next"/> is not
    /// the built-in <c>CompiledInferencePlan&lt;T&gt;</c>. Stitching is
    /// implementation-specific because it needs to splice each plan's
    /// internal step array; third-party implementers can opt in by
    /// providing their own <c>ThenAsync</c> on a concrete subtype.</exception>
    /// <remarks>
    /// <para>
    /// <b>Naming — why <c>ThenAsync</c> without <see cref="System.Threading.Tasks.Task"/>?</b>
    /// The name follows the issue #170 spec verbatim. The return type is
    /// synchronous because stitch-time work (shape validation, storage
    /// rebind, step-array splice) is trivial and there's no IO or long-
    /// running compute to yield on. Runtime analyzers that flag the Async
    /// suffix without a Task return (e.g. VSTHRD200) should be suppressed
    /// at call sites if they're noisy.
    /// </para>
    /// <para>
    /// <b>Semantics:</b> stitching is associative —
    /// <c>(a.ThenAsync(b)).ThenAsync(c)</c> is structurally equivalent to
    /// <c>a.ThenAsync(b.ThenAsync(c))</c> (same flat step sequence in the
    /// same order) — and re-entrant: a stitched plan can be the operand of
    /// another <c>ThenAsync</c> call. Each call validates shapes immediately
    /// and throws at composition time rather than failing late at execute.
    /// </para>
    /// <para>
    /// <b>Side effect on <paramref name="next"/>:</b> after this call,
    /// <paramref name="next"/>'s captured input tensor shares backing
    /// storage with <c>this</c>'s final output. Running
    /// <paramref name="next"/> standalone afterwards will read data
    /// produced by <c>this</c>'s most recent execution, not whatever the
    /// caller writes into next's input. The documented pattern is: once a
    /// plan is stitched into a pipeline, drive it through the stitched
    /// plan's <see cref="Execute"/>.
    /// </para>
    /// <para>
    /// <b>BINARY/SOURCE-BREAKING CHANGE WARNING (issue #170):</b> adding
    /// this member to <see cref="ICompiledPlan{T}"/> is both a
    /// source-breaking change for any downstream consumer that implements
    /// this interface directly (not just uses it) <b>and</b> a
    /// binary-breaking change for already-compiled external implementers
    /// at runtime. The built-in <c>CompiledInferencePlan&lt;T&gt;</c> in
    /// this assembly is updated alongside this interface addition, so
    /// internal consumers are unaffected — but third-party implementers
    /// must recompile against this assembly and add a corresponding method.
    /// No default-interface-member polyfill is provided because .NET
    /// Framework 4.7.1 (one of this library's target frameworks) does not
    /// support default interface members.
    /// </para>
    /// </remarks>
    ICompiledPlan<T> ThenAsync(ICompiledPlan<T> next);

    /// <summary>
    /// Serializes this plan to <paramref name="stream"/>. The output includes
    /// the full operation graph, pre-allocated buffer shapes, model weights,
    /// and a version stamp (format version + tensor-codec version + hardware
    /// fingerprint). Load the result with
    /// <see cref="CompiledPlanLoader.LoadInferenceAsync{T}"/>.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>BINARY/SOURCE-BREAKING CHANGE WARNING (issue #166):</b> adding this
    /// member to <see cref="ICompiledPlan{T}"/> is both a source-breaking and
    /// binary-breaking change for external implementers. Same rationale as
    /// EnableCheckpointing (#165) — no DIM polyfill on net471.
    /// </para>
    /// </remarks>
    Task SaveAsync(Stream stream, CancellationToken cancellationToken = default);

    /// <summary>
    /// Returns true when this plan's on-disk format, tensor-codec version,
    /// element type, and hardware fingerprint all match the current runtime.
    /// </summary>
    bool IsCompatibleWith(PlanCompatibilityInfo info);
}

/// <summary>
/// A compiled training plan that replays forward + backward passes
/// with pre-allocated gradient buffers. Compile once, train forever.
///
/// Call Step() each training iteration. Gradients are available via
/// the Gradients property after each Step().
/// </summary>
/// <typeparam name="T">The tensor element type.</typeparam>
public interface ICompiledTrainingPlan<T> : IDisposable
{
    /// <summary>
    /// Executes one training step: forward pass, gradient computation, backward pass.
    /// Returns the loss tensor.
    /// </summary>
    Tensor<T> Step();

    /// <summary>
    /// Gradient tensors for each parameter, in the same order as the parameters
    /// passed to compilation. Updated after each Step() call.
    /// </summary>
    Tensor<T>[] Gradients { get; }

    /// <summary>Number of forward execution steps.</summary>
    int ForwardStepCount { get; }

    /// <summary>Number of backward execution steps.</summary>
    int BackwardStepCount { get; }

    /// <summary>
    /// Configures fused optimizer updates that run after each Step().
    /// Once configured, Step() will automatically update parameters using
    /// the specified optimizer — no manual gradient application needed.
    /// </summary>
    /// <param name="optimizerType">The optimizer algorithm (SGD, Adam, etc.).</param>
    /// <param name="learningRate">Learning rate.</param>
    /// <param name="beta1">First moment decay (Adam/AdamW). Default: 0.9.</param>
    /// <param name="beta2">Second moment decay (Adam/AdamW). Default: 0.999.</param>
    /// <param name="eps">Epsilon for numerical stability. Default: 1e-8.</param>
    /// <param name="weightDecay">Weight decay (AdamW/LAMB). Default: 0.</param>
    void ConfigureOptimizer(
        OptimizerType optimizerType,
        float learningRate,
        float beta1 = 0.9f,
        float beta2 = 0.999f,
        float eps = 1e-8f,
        float weightDecay = 0f);

    /// <summary>
    /// Enables gradient checkpointing for this plan, reducing activation memory from
    /// O(N) to O(sqrt(N)) at the cost of ~33% more compute (each segment's forward
    /// runs twice during backward). Call once after compilation, before the training loop.
    /// </summary>
    /// <param name="segmentSize">Steps per checkpoint segment. 0 = auto (sqrt(N)).</param>
    /// <remarks>
    /// <para>
    /// This setting is reconfigurable per plan: if called multiple times, the most
    /// recent call determines the active checkpointing configuration (previous
    /// segment-size choices are overwritten rather than merged). The checkpointing
    /// system wraps the forward actions; gradients remain numerically equivalent
    /// to the non-checkpointed path within floating-point tolerance.
    /// </para>
    /// <para>
    /// <b>BINARY/SOURCE-BREAKING CHANGE WARNING (issue #165):</b> adding this member to
    /// <see cref="ICompiledTrainingPlan{T}"/> is both a source-breaking change for any
    /// downstream consumer that implements this interface directly (not just uses it)
    /// <b>and</b> a binary-breaking change for already-compiled external implementers
    /// at runtime. The built-in <c>CompiledTrainingPlan&lt;T&gt;</c> in this assembly
    /// is updated alongside this interface addition, so internal consumers are
    /// unaffected — but third-party implementers must recompile against this assembly
    /// and add a corresponding method. No default-interface-member polyfill is
    /// provided because .NET Framework 4.7.1 (one of this library's target frameworks)
    /// does not support default interface members. Added per Issue #165 where the
    /// downstream consumer needed interface-level access (not a concrete-class cast)
    /// to route checkpointing through <see cref="CompiledModelCache{T}"/>.
    /// </para>
    /// </remarks>
    void EnableCheckpointing(int segmentSize = 0);

    /// <summary>
    /// Serializes this training plan to <paramref name="stream"/>. Includes
    /// forward steps, backward functions, gradient buffer shapes, parameter
    /// tensor data, and optimizer state (if configured). Load the result with
    /// <see cref="CompiledPlanLoader.LoadTrainingAsync{T}"/>.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>BINARY/SOURCE-BREAKING CHANGE WARNING (issue #166):</b> same
    /// rationale as <see cref="ICompiledPlan{T}.SaveAsync"/> — no DIM
    /// polyfill on net471.
    /// </para>
    /// </remarks>
    Task SaveAsync(Stream stream, CancellationToken cancellationToken = default);

    /// <summary>
    /// Returns true when this plan's on-disk format, tensor-codec version,
    /// element type, and hardware fingerprint all match the current runtime.
    /// </summary>
    bool IsCompatibleWith(PlanCompatibilityInfo info);
}
