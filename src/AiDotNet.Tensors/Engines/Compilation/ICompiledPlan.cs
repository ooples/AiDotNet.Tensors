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

    /// <summary>
    /// Executes the compiled plan with the final output written into
    /// <paramref name="output"/>. Runs every step, then memcpy's the plan's
    /// internal final-output buffer into <paramref name="output"/>'s
    /// storage — one fixed memcpy per call, no per-call allocation of the
    /// output tensor. This is the primitive that makes
    /// <see cref="ICompiledPlan{T}"/> safely composable with CUDA Graph
    /// capture: the captured graph records both the step kernels and the
    /// final memcpy node; each Replay writes into
    /// <paramref name="output"/>'s backing pointer the same way.
    /// </summary>
    /// <param name="output">A caller-owned tensor whose shape equals this plan's
    /// final output shape. Must be contiguous with zero storage offset.</param>
    /// <exception cref="ArgumentNullException"><paramref name="output"/> is null.</exception>
    /// <exception cref="ArgumentException">Shape of <paramref name="output"/>
    /// does not match this plan's final output shape.</exception>
    /// <exception cref="ObjectDisposedException">This plan has been disposed.</exception>
    /// <remarks>
    /// <para>
    /// <b>Usage pattern — CUDA graph capture:</b> the caller owns the output
    /// buffer, so allocate it with a shape you already know (the graph's
    /// output shape is fixed at trace time). If you don't have the shape
    /// in hand, call <see cref="Execute"/> once first — its returned tensor
    /// carries the final shape and you can allocate a matching buffer from
    /// it for subsequent <c>ExecuteInto</c> calls.
    /// <code>
    /// var plan = cache.GetOrCompileInference(shape, trace);
    /// // Derive the output shape from one warm-up Execute, or pass a
    /// // caller-known shape directly.
    /// var first = plan.Execute();
    /// var outputBuf = engine.AllocateTensor&lt;float&gt;(first.Shape.ToArray());
    /// // A second warm-up run outside capture (stream-sync hygiene)
    /// plan.ExecuteInto(outputBuf);
    /// using var scope = new CudaGraphScope(backend, streamHandle);
    /// scope.BeginCapture();
    /// plan.ExecuteInto(outputBuf);
    /// scope.EndCapture();
    /// foreach (var batch in batches)
    /// {
    ///     plan.SetInputs(new[] { inputBuf.CopyFrom(batch) });
    ///     scope.Replay();                      // zero dispatch, zero alloc
    ///     outputBuf.CopyTo(results);
    /// }
    /// </code>
    /// </para>
    /// <para>
    /// <b>BINARY/SOURCE-BREAKING CHANGE (issue #199):</b> same rationale as
    /// <see cref="ThenAsync"/> / <see cref="SaveAsync"/> — adding a member to
    /// a public interface is breaking for external implementers, no DIM
    /// polyfill on net471. Built-in <c>CompiledInferencePlan&lt;T&gt;</c> is
    /// updated in the same PR.
    /// </para>
    /// </remarks>
    void ExecuteInto(Tensor<T> output);

    /// <summary>
    /// Copies the caller's input data into this plan's captured input
    /// buffers. After this call subsequent <see cref="Execute"/> /
    /// <see cref="ExecuteInto"/> read the updated data — one fixed memcpy
    /// per call, no per-call allocation of the captured input tensor.
    /// Implementation uses copy rather than storage rebind so the specialized
    /// kernel paths (which capture array references at compile time) pick
    /// up the new data correctly. Under CUDA Graph capture the copy is
    /// recorded as a device memcpy node and replays deterministically.
    /// </summary>
    /// <param name="inputs">Array of input tensors in graph-input order. For
    /// a single-input plan (the common inference case), pass
    /// <c>new[] { myInput }</c>. Each input must have the same rank, shape,
    /// and contiguity as the corresponding captured input.</param>
    /// <exception cref="ArgumentNullException"><paramref name="inputs"/> is
    /// null.</exception>
    /// <exception cref="ArgumentException"><paramref name="inputs"/> length
    /// doesn't match the plan's captured-input count, or any input's shape
    /// doesn't match the captured input. A zero-input plan accepts
    /// <c>Array.Empty&lt;Tensor&lt;T&gt;&gt;()</c> as a no-op — the length
    /// check still applies, so any other length throws.</exception>
    /// <exception cref="ObjectDisposedException">This plan has been disposed.</exception>
    /// <remarks>
    /// <para>
    /// <b>BINARY/SOURCE-BREAKING CHANGE (issue #199):</b> see
    /// <see cref="ExecuteInto"/>.
    /// </para>
    /// </remarks>
    void SetInputs(Tensor<T>[] inputs);

    /// <summary>Checks whether this plan is valid for the given input shape.</summary>
    bool IsValid(int[] inputShape);

    /// <summary>Number of compiled execution steps.</summary>
    int StepCount { get; }

    /// <summary>
    /// <b>Synchronous</b> two-plan composition: produces a long-lived
    /// stitched plan whose <see cref="Execute"/> runs <i>this</i>'s
    /// steps followed by <i>next</i>'s steps as one flat delegate array.
    /// The intermediate tensor (this plan's final output) shares backing
    /// storage with next's captured input after the call, so data flows
    /// through without any new <see cref="Tensor{T}"/> materialization or
    /// per-execute copy. Production pipelines like
    /// <c>tokenizer → encoder → transformer → classifier</c> become one
    /// compiled plan with no inter-plan materialization overhead.
    /// </summary>
    /// <remarks>
    /// <para>Same semantics as the obsolete <see cref="ThenAsync"/>; the
    /// rename drops the misleading "Async" suffix per issue #296. Use
    /// <see cref="ChainAsync(ICompiledPlan{T}, System.Threading.CancellationToken)"/>
    /// when you want one-shot async pipelining instead of a long-lived
    /// stitched plan.</para>
    /// </remarks>
    /// <inheritdoc cref="ThenAsync(ICompiledPlan{T})"/>
    ICompiledPlan<T> Stitch(ICompiledPlan<T> next);

    /// <summary>
    /// Asynchronously executes this plan on a backend-appropriate
    /// execution stream and returns the final output tensor when the
    /// pipeline finishes.
    /// </summary>
    /// <param name="cancellationToken">Cancellation. Observed before
    /// the await resumes; in-flight kernels run to completion regardless.</param>
    /// <returns>A <see cref="ValueTask{T}"/> resolving to the plan's
    /// final output tensor. Same buffer that
    /// <see cref="Execute"/> would return — caller must read before the
    /// next async call mutates it.</returns>
    /// <remarks>
    /// <para>
    /// CPU backends take a fast path that inlines the step loop on the
    /// calling thread and returns a completed <see cref="ValueTask{TResult}"/>
    /// — there's no native device-stream concept on CPU, so going
    /// through a worker / channel would just add queueing overhead with
    /// no host/device overlap to amortize it against. GPU backends
    /// submit kernels to a native <c>cudaStream_t</c> and poll
    /// <c>cuStreamQuery</c> + <see cref="System.Threading.Tasks.Task.Yield"/>
    /// until the stream drains, so the host thread yields rather than
    /// blocking while the GPU works. Acceptance criterion #9 of issue
    /// #296 forbids <see cref="System.Threading.Tasks.Task.Run"/> on the
    /// per-step hot path: the CPU fast path is fully synchronous (no
    /// Tasks created); the GPU path uses one long-lived worker pool
    /// behind <c>CpuExecutionStream</c> for plans that DO benefit from
    /// host/device overlap. Closes review-comment #298.8R6v (the prior
    /// docs said CPU drains through a worker, which only describes the
    /// pre-fast-path pre-#298 design).
    /// </para>
    /// </remarks>
    ValueTask<Tensor<T>> ExecuteAsync(CancellationToken cancellationToken = default);

    /// <summary>
    /// One-shot async pipeline: queues this plan's steps and
    /// <paramref name="next"/>'s steps onto the same execution stream,
    /// rebinds the boundary tensor (this plan's output → next's
    /// captured input) so no inter-stage copy is paid, and resolves to
    /// <paramref name="next"/>'s final output once the whole pipeline
    /// completes. Replaces the obsolete <see cref="ThenAsync"/> for
    /// callers who want true async chaining rather than a long-lived
    /// stitched plan.
    /// </summary>
    /// <param name="next">The downstream plan. Must be the built-in
    /// <c>CompiledInferencePlan&lt;T&gt;</c> — third-party
    /// implementers cannot be chained because the rebind needs concrete
    /// access to the input-tensor storage.</param>
    /// <param name="cancellationToken">Cancellation token; observed
    /// between the host queue-up and the final stream sync.</param>
    /// <returns>A <see cref="ValueTask{T}"/> resolving to
    /// <paramref name="next"/>'s final output tensor when the full
    /// pipeline drains.</returns>
    /// <remarks>
    /// <para>For chains where <paramref name="next"/> has multiple
    /// captured inputs, use the slot overload
    /// <see cref="ChainAsync(ICompiledPlan{T}, int, System.Threading.CancellationToken)"/>
    /// to identify which input to feed.</para>
    /// </remarks>
    ValueTask<Tensor<T>> ChainAsync(
        ICompiledPlan<T> next,
        CancellationToken cancellationToken = default);

    /// <summary>
    /// Multi-input variant of <see cref="ChainAsync(ICompiledPlan{T}, System.Threading.CancellationToken)"/>:
    /// the boundary rebind targets the captured input of
    /// <paramref name="next"/> at index <paramref name="nextInputSlot"/>.
    /// The other slots keep whatever data <see cref="SetInputs"/>
    /// loaded into them prior to the call (the canonical pattern for
    /// text-encoder → cross-attention noise-predictor: <c>SetInputs</c>
    /// the time-step / noise tensors first, then
    /// <c>ChainAsync(next, slotForCondition)</c> to feed the conditioner
    /// output in).
    /// </summary>
    /// <param name="next">The downstream plan.</param>
    /// <param name="nextInputSlot">Zero-based index into
    /// <paramref name="next"/>'s captured-input array identifying which
    /// input to rebind to this plan's output.</param>
    /// <param name="cancellationToken">Cancellation.</param>
    /// <returns>The same <see cref="ValueTask{T}"/> contract as the
    /// single-input overload.</returns>
    /// <exception cref="ArgumentOutOfRangeException">
    /// <paramref name="nextInputSlot"/> is not a valid index into
    /// <paramref name="next"/>'s captured inputs.</exception>
    ValueTask<Tensor<T>> ChainAsync(
        ICompiledPlan<T> next,
        int nextInputSlot,
        CancellationToken cancellationToken = default);

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
    /// <para>
    /// <b>Obsolete (issue #296):</b> the "Async" suffix is misleading —
    /// this method is synchronous. Prefer <see cref="Stitch"/> for the
    /// long-lived stitched-plan use case, or
    /// <see cref="ChainAsync(ICompiledPlan{T}, System.Threading.CancellationToken)"/>
    /// for one-shot async pipelining over an execution stream. This
    /// member ships <c>[Obsolete]</c> for one minor version, then is
    /// removed.
    /// </para>
    /// </remarks>
    [Obsolete("Use Stitch for sync plan composition or ChainAsync for true async pipelining. Closes #296.", error: false)]
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
    /// CUDA-Graph-capture-safe counterpart to <see cref="Step"/>: runs forward +
    /// backward, writes the final loss into <paramref name="lossOutput"/>, and
    /// leaves gradients in <see cref="Gradients"/>. No per-call allocation of the
    /// loss tensor — the captured graph records a memcpy node that lands in
    /// <paramref name="lossOutput"/>'s backing memory on every replay.
    /// </summary>
    /// <param name="lossOutput">Caller-owned tensor whose shape equals the
    /// plan's loss-tensor shape.</param>
    /// <exception cref="ArgumentNullException"><paramref name="lossOutput"/> is null.</exception>
    /// <exception cref="ArgumentException">Shape mismatch against the plan's loss output.</exception>
    /// <exception cref="ObjectDisposedException">Plan has been disposed.</exception>
    void StepInto(Tensor<T> lossOutput);

    /// <summary>
    /// Copies caller input data into this plan's captured input buffer(s).
    /// Same semantics as <see cref="ICompiledPlan{T}.SetInputs"/> — one
    /// memcpy per call, kernels see the new data next <see cref="Step"/>.
    /// Enables CUDA Graph capture of training loops where the input buffer
    /// is refilled between replays.
    /// </summary>
    /// <param name="inputs">Inputs in captured order.</param>
    /// <exception cref="ArgumentNullException"><paramref name="inputs"/> is null.</exception>
    /// <exception cref="ArgumentException">Length doesn't match the plan's
    /// captured-input count or any input's shape doesn't match.</exception>
    /// <exception cref="ObjectDisposedException">Plan has been disposed.</exception>
    void SetInputs(Tensor<T>[] inputs);

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
