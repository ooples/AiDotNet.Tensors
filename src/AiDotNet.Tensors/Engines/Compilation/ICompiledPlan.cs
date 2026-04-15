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
    /// <i>next</i>'s steps as one flat execution. The intermediate tensor
    /// (this plan's final output) is funneled into next's captured input
    /// without allocating any new <see cref="Tensor{T}"/> — production
    /// pipelines like <c>tokenizer → encoder → transformer → classifier</c>
    /// become one compiled kernel with no inter-plan materialization
    /// overhead.
    /// </summary>
    /// <param name="next">The plan to run after this one. Its captured input
    /// shape must equal this plan's final output shape — validated at stitch
    /// time, not at execute time.</param>
    /// <returns>A new <see cref="ICompiledPlan{T}"/> whose
    /// <see cref="Execute"/> is the composed pipeline. Disposing the
    /// stitched plan releases its own resources but does NOT dispose the
    /// underlying <c>this</c> or <paramref name="next"/> — callers retain
    /// ownership of those originals (this lets one plan participate in
    /// multiple stitched pipelines).</returns>
    /// <exception cref="ArgumentNullException"><paramref name="next"/> is null.</exception>
    /// <exception cref="ArgumentException">The output shape of this plan
    /// does not match the input shape of <paramref name="next"/>; or either
    /// plan has no steps to compose.</exception>
    /// <exception cref="NotSupportedException"><paramref name="next"/> is not
    /// the built-in <c>CompiledInferencePlan&lt;T&gt;</c>. Stitching is
    /// implementation-specific because it needs to splice each plan's
    /// internal step array; third-party implementers can opt in by
    /// providing their own <c>Then</c> on a concrete subtype.</exception>
    /// <remarks>
    /// <para>
    /// <b>Semantics:</b> stitching is associative —
    /// <c>(a.Then(b)).Then(c)</c> is structurally equivalent to
    /// <c>a.Then(b.Then(c))</c> (same flat step sequence in the same order)
    /// — and re-entrant: a stitched plan can be the operand of another
    /// <c>Then</c> call. Each <c>Then</c> validates shapes immediately and
    /// throws at composition time rather than failing late at execute time.
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
    ICompiledPlan<T> Then(ICompiledPlan<T> next);
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
}
