using System.Runtime.CompilerServices;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Autodiff;

/// <summary>
/// Provides the tape recording hook that engine operations call after computing their result.
/// All Record methods are AggressiveInlining so the JIT eliminates the null check entirely
/// when no tape is active (~2ns on the inference hot path).
/// </summary>
/// <remarks>
/// <para><b>Zero-allocation recording:</b> The Record methods construct <see cref="TapeEntry{T}"/>
/// structs with inline input fields (no <c>Tensor&lt;T&gt;[]</c> or <c>int[]</c> allocation).
/// The struct is passed by value to <see cref="GradientTape{T}.Record"/> which stores it
/// in a pre-allocated list. Only the SavedState array (when present) allocates.</para>
/// </remarks>
internal static class DifferentiableOps
{
    /// <summary>
    /// Global flag: true when ANY thread has an active gradient tape.
    /// Checked first in Record methods — when false, the entire method
    /// is skipped without even reading ThreadStatic (saves ~5ns/op).
    /// Set by GradientTape constructor/Dispose.
    /// </summary>
    internal static volatile int _anyTapeActive;

    /// <summary>Fast check: is any tape active on any thread?</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static bool AnyTapeActive() => _anyTapeActive != 0;

    /// <summary>
    /// Per-thread depth counter for active GradientTape lifecycles. Stays > 0
    /// from <see cref="GradientTape{T}"/> constructor through <c>Dispose</c>
    /// — including the backward walk during which <c>Current</c> is suspended
    /// to null. Use this (not <see cref="AnyTapeActive"/>) when you need to
    /// suppress an unrelated subsystem (e.g. AutoTracer) for the current
    /// thread's tape lifetime without affecting inference threads sharing
    /// the process.
    /// </summary>
    [ThreadStatic]
    internal static int _threadTapeDepth;

    /// <summary>Fast check: is any tape active on the calling thread (forward OR backward)?</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static bool ThreadTapeActive() => _threadTapeDepth > 0;

    /// <summary>
    /// Thread-local check: is a tape active on the calling thread for the
    /// given numeric T? Use this when an op needs to switch dispatch paths
    /// (e.g. take a slower tape-aware branch) — using the cross-thread
    /// <see cref="AnyTapeActive"/> would incorrectly trigger the slow path
    /// for a thread that has no tape but happens to share a process with
    /// a thread that does.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static bool IsTapeActiveForThread<T>()
        => _anyTapeActive != 0
           && GradientTape<T>.Current is not null
           && !NoGradScope<T>.IsSuppressed;

    /// <summary>
    /// Test-isolation hook: clears the CALLING THREAD's tape-depth counter only.
    /// A test that constructs a <see cref="GradientTape{T}"/> and never disposes
    /// it (e.g. an assertion throws first) otherwise leaves this stuck on the
    /// thread, which—together with a leaked <c>GradientTape&lt;T&gt;.Current</c>—
    /// corrupts later tests on that thread.
    /// <para>
    /// Deliberately does NOT touch the process-wide <see cref="_anyTapeActive"/>
    /// counter: that is shared across threads, and xUnit runs test classes in
    /// parallel, so zeroing it here would kill a concurrently-running test's
    /// active tape. A leaked global count is benign for correctness (recording
    /// no-ops whenever the thread's <c>Current</c> is null) and is healed on GC
    /// by the GradientTape finalizer.
    /// </para>
    /// </summary>
    internal static void ResetThreadTapeStateForTests()
    {
        _threadTapeDepth = 0;
    }

    // Indexed gradient array: set by ComputeGradients before backward walk,
    // read by AccumulateGrad for O(1) access instead of dictionary hash lookup.
    // ThreadStatic because backward is single-threaded per tape.
    [ThreadStatic]
    internal static object?[]? _indexedGrads;

    /// <summary>
    /// Non-zero while a compiled step runs a backward whose gradient buffers are NOT zeroed beforehand: the first
    /// contribution a buffer receives in this generation is copied in, later ones are added. A compiled plan sets it
    /// only when every backward action accumulates through <see cref="AccumulateGrad{T}"/> (all-generic), where the
    /// write order is fixed, so each buffer's first writer is the same every step. Replaces a memset of every
    /// gradient buffer plus an add for every contribution with one copy per buffer (AiDotNet #1804: 836 memsets and
    /// 393 adds per N-BEATS step on GPU).
    /// </summary>
    [ThreadStatic]
    internal static int GradWriteGeneration;

    private static int s_gradWriteGenerationCounter;

    /// <summary>A process-wide unique, non-zero generation, so a buffer's mark from an earlier step or plan can never
    /// be mistaken for the current one.</summary>
    internal static int NextGradWriteGeneration()
    {
        int generation = System.Threading.Interlocked.Increment(ref s_gradWriteGenerationCounter);
        return generation != 0 ? generation : System.Threading.Interlocked.Increment(ref s_gradWriteGenerationCounter);
    }

    /// <summary>Sets the indexed gradient array for the current backward pass.</summary>
    internal static void SetIndexedGrads(object?[] grads) => _indexedGrads = grads;

    /// <summary>Clears the indexed gradient array after backward completes.</summary>
    internal static void ClearIndexedGrads() => _indexedGrads = null;

    // Parallel sparse-gradient array for embedding-table parameters. Same
    // ThreadStatic + index-by-_gradIndex pattern as _indexedGrads, but each slot
    // holds a List<SparseEmbeddingGradient<T>> (object-typed for type-erasure)
    // instead of a Tensor<T>. Lets an embedding-lookup backward record its sparse
    // contribution (16 rows × 768 dim ≈ 49 KB) instead of materializing the dense
    // [vocab × dim] gradient (≈ 768 MB for paper-default LayoutXLM). Sparse-aware
    // optimizers query GetSparseEmbeddingGradsFor; dense-only optimizers fall back
    // to the existing grads dict (the dense materialization happens lazily, only
    // when ToDense is called).
    [ThreadStatic]
    internal static object?[]? _indexedSparseGrads;

    /// <summary>Sets the indexed sparse-gradient array for the current backward pass.</summary>
    internal static void SetIndexedSparseGrads(object?[] sparseGrads) => _indexedSparseGrads = sparseGrads;

    /// <summary>Clears the indexed sparse-gradient array after backward completes.</summary>
    internal static void ClearIndexedSparseGrads() => _indexedSparseGrads = null;

    /// <summary>
    /// Records a sparse embedding-table gradient contribution for <paramref name="param"/>.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Callers (embedding-lookup backward functions) pre-build the
    /// <see cref="SparseEmbeddingGradient{T}"/> via the engine-free
    /// <see cref="SparseEmbeddingGradient{T}.Build{TIndex}"/> factory so no
    /// <c>[vocabSize, embeddingDim]</c> tensor is allocated at scatter time. Multiple
    /// contributions to the same parameter (e.g. an embedding table read in several
    /// places during the same forward pass) are accumulated as a list so the optimizer
    /// can fold them per-accessed-row at step time.
    /// </para>
    /// <para>
    /// If the sparse-grad array hasn't been wired in by the active backward pass
    /// (legacy callers, tests that bypass <see cref="GradientTape{T}.ComputeGradients"/>),
    /// this method is a no-op. Callers that want a guaranteed gradient contribution
    /// should also call <see cref="AccumulateGrad{T}"/> after a <c>ToDense</c>
    /// materialization — but the canonical path is sparse-only.
    /// </para>
    /// </remarks>
    /// <summary>True when the current backward pass collects sparse embedding gradients for <paramref name="param"/>.</summary>
    internal static bool IsSparseEmbeddingGradWired<T>(Tensor<T> param)
    {
        int idx = param._gradIndex;
        return idx >= 0 && _indexedSparseGrads is not null && idx < _indexedSparseGrads.Length;
    }

    internal static void AccumulateSparseEmbeddingGrad<T>(Tensor<T> param, SparseEmbeddingGradient<T> grad)
    {
        if (param is null) throw new ArgumentNullException(nameof(param));
        int idx = param._gradIndex;
        if (idx < 0 || _indexedSparseGrads is null || idx >= _indexedSparseGrads.Length)
            return; // No sparse-grads array wired in — the dense fallback path is the caller's responsibility.

        var existing = _indexedSparseGrads[idx] as System.Collections.Generic.List<SparseEmbeddingGradient<T>>;
        if (existing is null)
        {
            existing = new System.Collections.Generic.List<SparseEmbeddingGradient<T>>(capacity: 1);
            _indexedSparseGrads[idx] = existing;
        }
        existing.Add(grad);
    }

    /// <summary>
    /// Returns the list of sparse embedding-gradient contributions accumulated for
    /// <paramref name="param"/> during the current backward pass, or <c>null</c> if
    /// none were recorded (the parameter's gradient should be read from the dense
    /// <c>grads</c> dictionary instead). Sparse-aware optimizers call this first;
    /// dense-only optimizers ignore the sparse dict.
    /// </summary>
    public static System.Collections.Generic.IReadOnlyList<SparseEmbeddingGradient<T>>? GetSparseEmbeddingGradsFor<T>(Tensor<T> param)
    {
        if (param is null) throw new ArgumentNullException(nameof(param));
        int idx = param._gradIndex;
        if (idx < 0 || _indexedSparseGrads is null || idx >= _indexedSparseGrads.Length)
            return null;
        return _indexedSparseGrads[idx] as System.Collections.Generic.List<SparseEmbeddingGradient<T>>;
    }

    /// <summary>
    /// True when a backward pass is running with <c>createGraph=true</c> —
    /// backward ops are themselves recorded on the tape for higher-order
    /// differentiation. While this flag is set, <see cref="AccumulateGrad{T}"/>
    /// uses out-of-place <c>TensorAdd</c> instead of <c>TensorAddInPlace</c>
    /// so the gradient tensor identity stays connected to its producing
    /// op through the graph. In-place mutation records an entry keyed on
    /// a <c>savedA.Clone()</c> input, which severs the double-backward
    /// graph — the second <see cref="GradientTape{T}.ComputeGradients"/>
    /// call would observe a disconnected gradient tensor and return an
    /// incomplete result.
    /// </summary>
    /// <remarks>
    /// ThreadStatic — a nested inner tape running backward with
    /// <c>createGraph=false</c> while an outer pass has the flag set
    /// should still see the flag, because the thread is the same. This
    /// is correct: if we're in a backward that records, we're generating
    /// tape entries for the outer higher-order pass and in-place on a
    /// fresh gradient tensor would still sever that graph. The flag is
    /// only cleared in the <c>finally</c> block of the top-level call.
    /// </remarks>
    [ThreadStatic]
    internal static bool _isBackwardCreateGraph;

    /// <summary>
    /// Requested-source reachability for the current backward pass. A null set preserves the
    /// historical unfiltered behavior; a non-null set contains exactly the tensors whose
    /// gradients can reach one of the caller's requested sources.
    /// </summary>
    private static class GradientRelevance<T>
    {
        [ThreadStatic]
        internal static HashSet<Tensor<T>>? Current;
    }

    /// <summary>Installs a requested-source gradient filter for one nested backward scope.</summary>
    internal static GradientRelevanceScope<T> PushGradientRelevance<T>(
        HashSet<Tensor<T>>? relevantTensors)
    {
        var previous = GradientRelevance<T>.Current;
        GradientRelevance<T>.Current = relevantTensors;
        return new GradientRelevanceScope<T>(previous);
    }

    /// <summary>
    /// Returns whether backward work for <paramref name="tensor"/> can contribute to a requested
    /// source. With no explicit source filter every gradient remains required.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static bool IsGradientRequired<T>(Tensor<T> tensor)
    {
        var relevant = GradientRelevance<T>.Current;
        if (relevant is not null) return relevant.Contains(tensor);
        // Without a reachability set, a leaf (no GradFn and not recorded by this tape: the batch, a constant) that
        // is not a requested source can never pass a gradient on, so its gradient is not computed.
        var leaves = GradientLeafSources<T>.Current;
        return leaves is null || tensor.GradFn is not null || leaves.Contains(tensor);
    }

    // The tensors a backward pass may produce gradients for besides those with a GradFn (requested sources,
    // retained and hooked tensors, and the tape's recorded outputs), installed when the full reachability set is
    // skipped as too costly for a small tape.
    private static class GradientLeafSources<T>
    {
        [ThreadStatic]
        internal static HashSet<Tensor<T>>? Current;
    }

    /// <summary>Installs the leaf filter for one backward scope; null leaves every gradient required.</summary>
    internal static GradientLeafSourcesScope<T> PushGradientLeafSources<T>(HashSet<Tensor<T>>? leaves)
    {
        var previous = GradientLeafSources<T>.Current;
        GradientLeafSources<T>.Current = leaves;
        return new GradientLeafSourcesScope<T>(previous);
    }

    internal readonly struct GradientLeafSourcesScope<T> : IDisposable
    {
        private readonly HashSet<Tensor<T>>? _previous;

        internal GradientLeafSourcesScope(HashSet<Tensor<T>>? previous)
        {
            _previous = previous;
        }

        public void Dispose()
        {
            GradientLeafSources<T>.Current = _previous;
        }
    }

    internal readonly struct GradientRelevanceScope<T> : IDisposable
    {
        private readonly HashSet<Tensor<T>>? _previous;

        internal GradientRelevanceScope(HashSet<Tensor<T>>? previous)
        {
            _previous = previous;
        }

        public void Dispose()
        {
            GradientRelevance<T>.Current = _previous;
        }
    }

    /// <summary>
    /// Returns true if a gradient tape is active and not suppressed.
    /// Use this to guard savedState allocation: only create new object[]
    /// when IsRecording is true, avoiding unnecessary GC pressure during inference.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static bool IsRecording<T>()
    {
        return GradientTape<T>.Current is not null && !NoGradScope<T>.IsSuppressed;
    }

    // Engine view methods delegate their storage work to Tensor.Reshape/Transpose and then
    // register an operation-specific backward edge themselves. Suppress the Tensor-level
    // fallback only for that narrow call so a view is recorded exactly once. Direct Tensor
    // view APIs remain fully differentiable and replayable.
    [ThreadStatic]
    private static int s_tensorViewRecordingSuppressionDepth;

    internal static bool IsTensorViewRecordingSuppressed
        => s_tensorViewRecordingSuppressionDepth > 0;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static TensorViewRecordingSuppression SuppressTensorViewRecording()
    {
        s_tensorViewRecordingSuppressionDepth++;
        return new TensorViewRecordingSuppression(new TensorViewRecordingSuppressionState());
    }

    internal readonly struct TensorViewRecordingSuppression : IDisposable
    {
        private readonly TensorViewRecordingSuppressionState? _state;

        internal TensorViewRecordingSuppression(TensorViewRecordingSuppressionState state)
        {
            _state = state;
        }

        public void Dispose()
        {
            // A reference-backed lease makes disposal idempotent even when this
            // readonly struct is copied: every copy observes the same release bit.
            if (_state is null || _state.IsReleased)
            {
                return;
            }

            _state.IsReleased = true;
            s_tensorViewRecordingSuppressionDepth--;
        }
    }

    internal sealed class TensorViewRecordingSuppressionState
    {
        internal bool IsReleased;
    }

    /// <summary>
    /// Records a variadic operation (4+ inputs) to the current gradient tape if one is active.
    /// The caller must provide the pre-allocated inputs array.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static void RecordIfActive<T>(
        string opName,
        Tensor<T> output,
        Tensor<T>[] inputs,
        BackwardFunction<T> backward,
        object[]? savedState = null)
    {
        if (_anyTapeActive == 0) return;
        if (NoGradScope<T>.IsSuppressed) return;
        var tape = GradientTape<T>.Current;
        if (tape is null) return;
        // Note: we always record during forward passes even when a compiled backward exists.
        // This ensures GradFn is set on outputs (needed for createGraph: true) and the tape
        // is populated as a fallback. The compiled backward is used at ComputeGradients time
        // (see GradientTape.ComputeGradients), not here.

        ref var slot = ref tape.RecordSlot(out bool accepted);
        if (!accepted) return;
        // Dispatch the backward to wherever the data actually lives. This is the single funnel every
        // differentiable op passes through, so it needs no per-op upkeep — see NoteDataDevice.
        tape.NoteDataDevice(output);
        slot.OperationName = opName;
        slot.Output = output;
        slot.Backward = backward;
        slot.SavedState = savedState;

        if (inputs.Length <= 3)
        {
            // Use inline fields for 1-3 inputs (avoids overflow array)
            slot.InputCount = (byte)inputs.Length;
            slot.Input0 = inputs.Length > 0 ? inputs[0] : null!;
            slot.Input1 = inputs.Length > 1 ? inputs[1] : null;
            slot.Input2 = inputs.Length > 2 ? inputs[2] : null;
            slot.Version0 = inputs.Length > 0 ? inputs[0].Version : 0;
            slot.Version1 = inputs.Length > 1 ? inputs[1].Version : 0;
            slot.Version2 = inputs.Length > 2 ? inputs[2].Version : 0;
        }
        else
        {
            // Overflow for 4+ inputs
            var versions = new int[inputs.Length];
            for (int i = 0; i < inputs.Length; i++)
                versions[i] = inputs[i].Version;
            slot.InputsOverflow = inputs;
            slot.InputVersionsOverflow = versions;
            slot.InputCount = 0xFF;
            slot.Input0 = inputs[0];
        }

        // Set GradFn for graph-based backward. Rent from the pool —
        // returned during backward cleanup. Issue #319 Phase 3.
        var node = GradNodePool<T>.Rent();
        node.OwningTape = tape;
        node.Backward = backward;
        node.Output = output;
        node.SavedState = savedState;
        switch (inputs.Length)
        {
            case 1:
                node.Input0 = inputs[0];
                node.InputCount = 1;
                break;
            case 2:
                node.Input0 = inputs[0];
                node.Input1 = inputs[1];
                node.InputCount = 2;
                break;
            case 3:
                node.Input0 = inputs[0];
                node.Input1 = inputs[1];
                node.Input2 = inputs[2];
                node.InputCount = 3;
                break;
            default:
                node.Input0 = inputs[0];
                node.InputsOverflow = inputs;
                node.InputCount = 0xFF;
                break;
        }
        output.GradFn = node;

        // Issue #338 tape-pinning: see RecordUnary rationale. Pin every
        // recorded input so a 4+ input op cannot have any of its tape-held
        // tensors recycled before the backward walk consumes them.
        for (int i = 0; i < inputs.Length; i++)
            inputs[i]._pinnedByTape = true;
        PinSavedStateTensors<T>(ref slot);
    }

    /// <summary>
    /// Records a unary operation (single input). Zero heap allocation, zero struct copy.
    /// Writes directly into the arena slot via ref return.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static void RecordUnary<T>(
        string opName,
        Tensor<T> output,
        Tensor<T> input,
        BackwardFunction<T> backward,
        object[]? savedState = null)
    {
        // Fast path: skip ThreadStatic read entirely when no tape exists globally
        if (_anyTapeActive == 0) return;
        var tape = GradientTape<T>.Current;
        if (tape is null || NoGradScope<T>.IsSuppressed) return;
        ref var slot = ref tape.RecordSlot(out bool accepted);
        if (!accepted) return;
        // Dispatch the backward to wherever the data actually lives. This is the single funnel every
        // differentiable op passes through, so it needs no per-op upkeep — see NoteDataDevice.
        tape.NoteDataDevice(output);
        slot.OperationName = opName;
        slot.Output = output;
        slot.Backward = backward;
        slot.SavedState = savedState;
        slot.Input0 = input;
        slot.InputCount = 1;
        slot.Version0 = input.Version;

        // Set GradFn on output for O(1) graph-based backward traversal.
        // Pooled rental — see GradNodePool<T>. Issue #319 Phase 3.
        var node = GradNodePool<T>.Rent();
        node.OwningTape = tape;
        node.Backward = backward;
        node.Output = output;
        node.Input0 = input;
        node.InputCount = 1;
        node.SavedState = savedState;
        output.GradFn = node;

        // Issue #338 tape-pinning: mark input as held by the active tape so
        // TensorPool.Return refuses to reissue it as scratch before the
        // backward walk consumes it. Output stays unpinned — it's freshly
        // produced and may be safely pooled if the consumer drops it before
        // backward runs.
        input._pinnedByTape = true;
        PinSavedStateTensors<T>(ref slot);
    }

    /// <summary>
    /// Records a binary operation (two inputs). Zero heap allocation, zero struct copy.
    /// Writes directly into the arena slot via ref return.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static void RecordBinary<T>(
        string opName,
        Tensor<T> output,
        Tensor<T> a,
        Tensor<T> b,
        BackwardFunction<T> backward,
        object[]? savedState = null)
    {
        if (_anyTapeActive == 0) return;
        var tape = GradientTape<T>.Current;
        if (tape is null || NoGradScope<T>.IsSuppressed) return;
        ref var slot = ref tape.RecordSlot(out bool accepted);
        if (!accepted) return;
        // Dispatch the backward to wherever the data actually lives. This is the single funnel every
        // differentiable op passes through, so it needs no per-op upkeep — see NoteDataDevice.
        tape.NoteDataDevice(output);
        slot.OperationName = opName;
        slot.Output = output;
        slot.Backward = backward;
        slot.SavedState = savedState;
        slot.Input0 = a;
        slot.Input1 = b;
        slot.InputCount = 2;
        slot.Version0 = a.Version;
        slot.Version1 = b.Version;

        // Pooled GradNode rental — issue #319 Phase 3.
        var node = GradNodePool<T>.Rent();
        node.OwningTape = tape;
        node.Backward = backward;
        node.Output = output;
        node.Input0 = a;
        node.Input1 = b;
        node.InputCount = 2;
        node.SavedState = savedState;
        output.GradFn = node;

        // Issue #338 tape-pinning: see RecordUnary rationale. Pin BOTH
        // inputs since the binary backward consumes both.
        a._pinnedByTape = true;
        b._pinnedByTape = true;
        PinSavedStateTensors<T>(ref slot);
    }

    /// <summary>
    /// Issue #338 completion: pins every tensor stored in a recorded op's
    /// saved state against pool/arena reuse, exactly as Record* pins the op's
    /// inputs. Many backward functions read tensors OUT of savedState rather than from the op's
    /// inputs — LayerNorm/BatchNorm/RMSNorm mean/variance/rms, attention weights and softmax
    /// stats, dropout masks, RoPE cos/sin, fused pre-activations. Those buffers are live for the
    /// whole backward pass, but the input-only pin left them poolable: under buffer reuse a later
    /// same-shape allocation could reissue and overwrite one before its backward consumed it,
    /// silently corrupting the gradient (the failure <c>SavedStatePinningReproTests</c> and the
    /// consumer's <c>Gru_ArenaOnEqualsOff</c> surface). Non-tensor entries (epsilon, axes, flags)
    /// are skipped. Near-free no-op when the saved state is null — the common case
    /// for elementwise ops that need no captured state.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static void PinSavedStateTensors<T>(ref TapeEntry<T> entry)
    {
        if (entry.SavedStatePinsHeld) return;
        var savedState = entry.SavedState;
        if (savedState is null) return;
        var visitor = new SavedStatePinVisitor(pin: true);
        SavedStateTensorTraversal.Visit(savedState, ref visitor);
        entry.SavedStatePinsHeld = visitor.VisitedAny;
    }

    /// <summary>
    /// Reverses <see cref="PinSavedStateTensors{T}"/> exactly once for a recorded entry.
    /// The entry-owned lifecycle bit prevents a persistent/cached second cleanup from consuming
    /// a pin owned by another tape that happens to reference the same tensor.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static void UnpinSavedStateTensors<T>(ref TapeEntry<T> entry)
    {
        if (!entry.SavedStatePinsHeld) return;
        var savedState = entry.SavedState;
        if (savedState is null)
        {
            entry.SavedStatePinsHeld = false;
            return;
        }
        var visitor = new SavedStatePinVisitor(pin: false);
        SavedStateTensorTraversal.Visit(savedState, ref visitor);
        entry.SavedStatePinsHeld = false;
    }

    private struct SavedStatePinVisitor : ISavedStateTensorVisitor
    {
        private readonly bool _pin;
        internal bool VisitedAny;

        internal SavedStatePinVisitor(bool pin)
        {
            _pin = pin;
            VisitedAny = false;
        }

        public void Visit(ITensorStorageLeaseSource tensor)
        {
            tensor.SetTapePinned(_pin);
            VisitedAny = true;
        }
    }

    /// <summary>
    /// Issue #338 Phase B.2: same semantics as <see cref="AccumulateGrad{T}"/>
    /// but returns <c>true</c> when the supplied <paramref name="grad"/> was
    /// consumed by an in-place add (and thus safe to pool-return). Returns
    /// <c>false</c> when <paramref name="grad"/> was stored as the slot's
    /// owning gradient or referenced for a future tape pass; in those cases
    /// the caller MUST NOT pool-return <paramref name="grad"/>.
    /// <para>
    /// Lets backward functions pool local intermediate gradient buffers
    /// (like <c>engine.TensorNegate(gradOut)</c> in SubtractBackward) when
    /// they're guaranteed to be consumed by accumulation, without leaking
    /// when they end up as the first-write slot.
    /// </para>
    /// </summary>
    /// <summary>True, and marks the buffer written, when this is its first contribution in the current
    /// <see cref="GradWriteGeneration"/>; false outside a generation or for later contributions.</summary>
    private static bool ClaimFirstWrite<T>(Tensor<T> buffer)
    {
        int generation = GradWriteGeneration;
        if (generation == 0 || buffer._gradWriteGeneration == generation) return false;
        buffer._gradWriteGeneration = generation;
        return true;
    }

    /// <summary>Marks a buffer stored fresh in the current <see cref="GradWriteGeneration"/> as written, so a later
    /// contribution adds to it rather than claiming the first write and overwriting it.</summary>
    private static void MarkWrittenThisStep<T>(Tensor<T> buffer)
    {
        int generation = GradWriteGeneration;
        if (generation != 0) buffer._gradWriteGeneration = generation;
    }

    /// <summary>True when <paramref name="buffer"/> already holds this step's gradient: outside a
    /// <see cref="GradWriteGeneration"/> every stored buffer is live, inside one only a buffer written in it is. A
    /// buffer kept from an earlier compiled step is stale until its first write clears it.</summary>
    internal static bool IsWrittenThisStep<T>(Tensor<T> buffer)
    {
        int generation = GradWriteGeneration;
        return generation == 0 || buffer._gradWriteGeneration == generation;
    }

    /// <summary>The step's first contribution to a gradient buffer that was not zeroed: copied in, not added.</summary>
    private static Tensor<T> CopyFirstWrite<T>(Tensor<T> tensor, Tensor<T> buffer, Tensor<T> contribution, IEngine engine)
    {
        if (!buffer.IsContiguous)
        {
            var previous = buffer;
            buffer = buffer.Contiguous();
            buffer._gradWriteGeneration = GradWriteGeneration;
            ReplaceAccumulatorBufferOwner(tensor, previous, buffer);
        }
        if (!ReferenceEquals(contribution, buffer))
            engine.TensorCopy(contribution, buffer);
        return buffer;
    }

    internal static bool AccumulateGradPoolable<T>(
        Dictionary<Tensor<T>, Tensor<T>> grads,
        Tensor<T> tensor,
        Tensor<T> grad,
        IEngine engine)
    {
        // This overload explicitly transfers disposal responsibility back to its caller.
        // An irrelevant contribution was never donated to an accumulator, so the caller may
        // immediately return its scratch buffer instead of leaking the rental.
        if (!IsGradientRequired(tensor)) return true;

        bool needsOutOfPlace = _isBackwardCreateGraph;
        bool wasAlreadyOwned = !needsOutOfPlace && IsAccumulatorBufferOwned(grad);
        int idx = tensor._gradIndex;
        if (idx >= 0 && _indexedGrads != null && idx < _indexedGrads.Length
            && _indexedGrads[idx] != null)
        {
            // Has existing → accumulation will occur.
            AccumulateGrad(grads, tensor, grad, engine);
            // In-place add consumes grad; out-of-place (createGraph) keeps
            // it linked into the recorded TensorAdd's input chain — DO NOT
            // pool in that case.
            return !needsOutOfPlace && !wasAlreadyOwned;
        }
        if (grads.ContainsKey(tensor))
        {
            AccumulateGrad(grads, tensor, grad, engine);
            return !needsOutOfPlace && !wasAlreadyOwned;
        }
        // First-write donates the buffer when it is unique, or copies it when it
        // is already owned by another gradient slot. Either way, the caller must
        // not return the contribution: the donated buffer is now the accumulator,
        // while an aliased contribution still belongs to its earlier owner.
        AccumulateGrad(grads, tensor, grad, engine);
        return false;
    }

    // Cached once at type init. AccumulateGrad runs per-op on the backward
    // pass, and the env read below executed BEFORE the cheap engine-type check
    // short-circuited, so it fired on every CPU backward op too — part of the
    // ~5.8% GetEnvironmentVariable hot-path cost in the N-BEATS profile
    // (#728/#1804). Debug-only flag; env vars are process-stable.
    private static readonly bool _graphCaptureDebug =
        Environment.GetEnvironmentVariable("AIDOTNET_GRAPH_CAPTURE_DEBUG") == "1";

    /// <summary>
    /// Adds a slice's gradient into ONLY its region of <paramref name="tensor"/>'s existing accumulator, instead of
    /// materializing a full-size zero tensor and adding all of it. A recurrence that slices one tensor per step (an
    /// LSTM's per-timestep input) otherwise does O(T * size) work per sequence in backward; on the CPU that made a
    /// hoisted-projection LSTM 2.4x slower than the per-step form it was meant to beat.
    /// </summary>
    /// <param name="regionStart">Start of the region per axis of the full tensor (rank = tensor rank).</param>
    /// <param name="regionShape">The region's shape per axis of the full tensor; <paramref name="contribution"/> holds
    /// exactly its elements in row-major order (an axis slice's dropped axis has extent 1 here).</param>
    /// <returns>False (nothing done) unless the in-place path applies: CPU engine, no create-graph, an existing
    /// contiguous host accumulator that does not overlap the contribution. The caller then takes its full-size path.</returns>
    internal static bool TryAccumulateRegion<T>(
        Dictionary<Tensor<T>, Tensor<T>> grads, Tensor<T> tensor, Tensor<T> contribution,
        int[] regionStart, int[] regionShape, IEngine engine)
    {
        if (_isBackwardCreateGraph || engine.SupportsGpu || engine is not CpuEngine) return false;
        int idx = tensor._gradIndex;
        bool indexed = idx >= 0 && _indexedGrads != null && idx < _indexedGrads.Length;
        Tensor<T>? existing = indexed
            ? (Tensor<T>?)_indexedGrads![idx]
            : (grads.TryGetValue(tensor, out var found) ? found : null);
        if (existing is null || !existing.IsContiguous || existing.HasPendingGpuData
            || existing.Length != tensor.Length || existing.Rank != regionStart.Length) return false;
        var source = contribution.IsContiguous ? contribution : contribution.Contiguous();
        if (HasOverlappingStorage(existing, source)) return false;

        var numOps = global::AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>();
        var dest = existing.AsWritableSpan();
        if (ClaimFirstWrite(existing)) dest.Clear();   // stale buffer from an earlier step: this is its first write
        var src = source.AsSpan();
        var fullShape = existing._shape;
        int rank = fullShape.Length;
        int rowLength = regionShape[rank - 1];
        int rows = rowLength == 0 ? 0 : src.Length / rowLength;
        for (int row = 0; row < rows; row++)
        {
            int remaining = row;
            int offset = regionStart[rank - 1];
            int stride = fullShape[rank - 1];
            for (int d = rank - 2; d >= 0; d--)
            {
                int coordinate = remaining % regionShape[d];
                remaining /= regionShape[d];
                offset += (regionStart[d] + coordinate) * stride;
                stride *= fullShape[d];
            }
            // One vectorized add per row (it was a virtual numOps.Add per element); element-wise, so bit-identical.
            var destRow = dest.Slice(offset, rowLength);
            numOps.Add(destRow, src.Slice(row * rowLength, rowLength), destRow);
        }

        if (indexed) _indexedGrads![idx] = existing;
        grads[tensor] = existing;
        tensor.Grad = existing;
        return true;
    }

    /// <summary>
    /// For a fused backward that produces an input's whole gradient: the existing host accumulator it may write straight
    /// into, so the gradient is never staged in a temporary and copied. <paramref name="overwrite"/> is true when this is
    /// the buffer's first contribution of the step under a <see cref="GradWriteGeneration"/> (the buffer was not zeroed,
    /// so the caller must store, not add); otherwise the caller adds. Returns null when the direct route does not apply
    /// -- no existing contiguous host accumulator yet (the eager tape's first contribution), a GPU engine, or a
    /// create-graph backward -- and the caller passes a contribution tensor to <see cref="AccumulateGrad{T}"/> instead.
    /// A caller that writes the returned buffer must call <see cref="TensorBase{T}.IncrementVersion"/> afterwards.
    /// </summary>
    internal static Tensor<T>? TryGetDirectGradTarget<T>(
        Dictionary<Tensor<T>, Tensor<T>> grads, Tensor<T> tensor, IEngine engine, out bool overwrite)
    {
        overwrite = false;
        if (_isBackwardCreateGraph || engine.SupportsGpu || engine is not CpuEngine) return null;
        int idx = tensor._gradIndex;
        bool indexed = idx >= 0 && _indexedGrads != null && idx < _indexedGrads.Length;
        Tensor<T>? existing = indexed
            ? (Tensor<T>?)_indexedGrads![idx]
            : (grads.TryGetValue(tensor, out var found) ? found : null);
        if (existing is null || !existing.IsContiguous || existing.HasPendingGpuData
            || existing.Length != tensor.Length) return null;
        overwrite = ClaimFirstWrite(existing);
        if (indexed) _indexedGrads![idx] = existing;
        grads[tensor] = existing;
        tensor.Grad = existing;
        return existing;
    }

    /// <summary>
    /// Accumulates a gradient for a tensor in the gradient dictionary.
    /// If the tensor already has a gradient, the new gradient is added to it.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining
#if !NETFRAMEWORK
        | MethodImplOptions.AggressiveOptimization
#endif
    )]
    internal static void AccumulateGrad<T>(
        Dictionary<Tensor<T>, Tensor<T>> grads,
        Tensor<T> tensor,
        Tensor<T> grad,
        IEngine engine)
    {
        // Requested-source traversal is pruned by the tape/compiled execution loops, and the
        // expensive multi-input backward functions query IsGradientRequired before allocating.
        // Do not drop an arbitrary contribution here: many older backward functions rent a
        // buffer before calling AccumulateGrad, while others pass a borrowed gradOutput reused
        // by a later input. Silently returning from this donation point either leaked the rental
        // or made generic reclamation unsafe. Storing the contribution preserves ownership until
        // the normal filtered-gradient cleanup; AccumulateGradPoolable handles disposable scratch.

        // PR #638 A2: mark this scope as grad accumulation so the DirectGpu engine's resident in-place add
        // fast path engages ONLY here (the dedicated, non-aliased gradient leaf) and never hijacks forward
        // in-place ops on aliased activations (that hijack threw CUDA-700). No-op for non-DirectGpu engines.
        using var _gradAccumScope = (engine as AiDotNet.Tensors.Engines.DirectGpuTensorEngine)?.EnterGradAccumulation();
        // FP32 accumulation is narrower than FP32 backward: only the addition which combines
        // fan-out contributions overrides autocast. The backward kernels that produced each
        // contribution retain their configured FP16/BF16/FP8 precision.
        using var _accumulationPrecisionScope =
            GradientAccumulationPrecisionScope.EnterFloat32AutocastForAddition();

        // Higher-order AD: in-place add records a "TensorAddInPlace"
        // entry whose saved input is a *clone* of the existing gradient,
        // which severs the graph the second backward pass needs to
        // walk. Use out-of-place add so the new gradient tensor is
        // produced by a "TensorAdd" entry whose inputs ARE the original
        // tensor references — the graph stays connected.
        bool needsOutOfPlace = _isBackwardCreateGraph;

        // CONTIGUITY INVARIANT (issue #274): every backward op that
        // permutes / reshapes / slices its incoming grad produces a
        // non-contiguous view tensor (e.g. PermuteBackward returns
        // engine.TensorPermute(...) which is a stride-rewrite, not a
        // contiguous copy). If we store that view as the first
        // gradient and a later AccumulateGrad call hits TensorAddInPlace
        // on it, the engine throws "In-place add requires contiguous
        // target tensor." Materialize for the in-place path only.
        //
        // HIGHER-ORDER AD CAVEAT: when `needsOutOfPlace` is true (the
        // createGraph=true path that records backward ops on the tape
        // for double-backward), we must NOT materialize via .Contiguous()
        // before TensorAdd. Contiguous() produces a fresh tensor whose
        // GradFn is detached from the original op chain — the second
        // backward pass would then fail to walk back through it. The
        // out-of-place TensorAdd itself records a tape entry that
        // preserves graph connectivity through the original grad
        // reference. Keep the original `grad` for the out-of-place
        // path; only materialize for in-place add storage.
        if (_graphCaptureDebug
            && engine is AiDotNet.Tensors.Engines.DirectGpuTensorEngine gde && gde.ResidentStepActive)
        {
            Tensor<T>? exi = grads.TryGetValue(tensor, out var ed) ? ed : null;
            try { System.IO.File.AppendAllText(System.IO.Path.Combine(System.IO.Path.GetTempPath(), "aidotnet_graphcapture_diag.txt"),
                $"[ALIAS] accumgrad op={AiDotNet.Tensors.Engines.DirectGpuTensorEngine.s_currentBackwardOp} len={grad.Length}: gradRes={grad.TryGetGpuBuffer() is not null} gradContig={grad.IsContiguous} existRes={(exi?.TryGetGpuBuffer() is not null)}" + System.Environment.NewLine); } catch { }
        }
        // LAZY, because the out-of-place path never uses it. Computing this eagerly called
        // .Contiguous() on every non-contiguous incoming gradient even when needsOutOfPlace is
        // true -- and that branch deliberately keeps `grad` itself, so the materialized copy was
        // allocated, never read, and immediately garbage. Backward ops that permute, reshape or
        // slice always hand in a non-contiguous view, so this fired on exactly the ops most likely
        // to carry a full parameter-sized gradient.
        Tensor<T>? gradForInPlaceCache = null;
        Tensor<T> GradForInPlace() =>
            gradForInPlaceCache ??= (grad.IsContiguous ? grad : grad.Contiguous());

        // Fast path: use indexed array when grad indices are assigned (avoids hash lookup)
        int idx = tensor._gradIndex;
        if (idx >= 0 && _indexedGrads != null && idx < _indexedGrads.Length)
        {
            var existing = (Tensor<T>?)_indexedGrads[idx];
            if (existing != null)
            {
                Tensor<T> accumulated;
                if (needsOutOfPlace)
                {
                    // Higher-order: keep `grad` (not the materialized
                    // copy) so TensorAdd's recorded tape entry chains
                    // back through the original GradFn lineage.
                    accumulated = engine.TensorAdd(existing, grad);
                }
                else if (ClaimFirstWrite(existing))
                {
                    accumulated = CopyFirstWrite(tensor, existing, GradForInPlace(), engine);
                }
                else
                {
                    // Defensive: if the existing slot is somehow
                    // non-contiguous (e.g. populated outside this
                    // method), materialize before in-place add.
                    if (!existing.IsContiguous)
                    {
                        var previous = existing;
                        existing = existing.Contiguous();
                        ReplaceAccumulatorBufferOwner(tensor, previous, existing);
                    }

                    // A contribution can be the same tensor object as this
                    // destination's first-write accumulator (for example x+x).
                    // Mutating it would change the borrowed contribution before
                    // another input has had a chance to consume it. Detach this
                    // one slot out-of-place; ordinary unique accumulation stays
                    // on the zero-allocation in-place path.
                    if (HasOverlappingStorage(existing, GradForInPlace()))
                    {
                        accumulated = AddAliasedContributionOutOfPlace(existing, GradForInPlace(), engine);
                        ReplaceAccumulatorBufferOwner(tensor, existing, accumulated);
                    }
                    else
                    {
                        using (new NoGradScope<T>()) // accumulation is not a recorded op; skips the in-place op's pre-mutation clone
                            engine.TensorAddInPlace(existing, GradForInPlace());
                        accumulated = existing;
                    }
                }
                _indexedGrads[idx] = accumulated;
                tensor.Grad = accumulated;
            }
            else
            {
                // First-write slot. In createGraph mode keep the original
                // `grad` so future TensorAdd entries can chain through it.
                // Normal backward donates unique scratch without a copy, but
                // isolates a contribution already owned by another slot.
                var stored = needsOutOfPlace
                    ? grad
                    : TakeAccumulatorBuffer(tensor, GradForInPlace(), engine);
                MarkWrittenThisStep(stored);
                _indexedGrads[idx] = stored;
                tensor.Grad = stored;
            }
            grads[tensor] = tensor.Grad!;
            return;
        }

        // Fallback: dictionary path (for ops outside tape or during non-indexed backward)
        if (grads.TryGetValue(tensor, out var existingDict))
        {
            if (needsOutOfPlace)
            {
                var accumulated = engine.TensorAdd(existingDict, grad);
                grads[tensor] = accumulated;
                tensor.Grad = accumulated;
            }
            else if (ClaimFirstWrite(existingDict))
            {
                var written = CopyFirstWrite(tensor, existingDict, GradForInPlace(), engine);
                grads[tensor] = written;
                tensor.Grad = written;
            }
            else
            {
                if (!existingDict.IsContiguous)
                {
                    var previous = existingDict;
                    existingDict = existingDict.Contiguous();
                    grads[tensor] = existingDict;
                    ReplaceAccumulatorBufferOwner(tensor, previous, existingDict);
                }
                if (HasOverlappingStorage(existingDict, GradForInPlace()))
                {
                    var accumulated = AddAliasedContributionOutOfPlace(existingDict, GradForInPlace(), engine);
                    grads[tensor] = accumulated;
                    tensor.Grad = accumulated;
                    ReplaceAccumulatorBufferOwner(tensor, existingDict, accumulated);
                }
                else
                {
                    using (new NoGradScope<T>()) // accumulation is not a recorded op; skips the in-place op's pre-mutation clone
                        engine.TensorAddInPlace(existingDict, GradForInPlace());
                    tensor.Grad = existingDict;
                }
            }
        }
        else
        {
            var stored = needsOutOfPlace
                ? grad
                : TakeAccumulatorBuffer(tensor, GradForInPlace(), engine);
            MarkWrittenThisStep(stored);
            grads[tensor] = stored;
            tensor.Grad = stored;
        }
    }

    /// <summary>
    /// Tracks ownership of donated first-write buffers for one backward step.
    /// </summary>
    /// <remarks>
    /// The reverse map makes the common unique-buffer path O(1) and zero-copy while
    /// identifying the exceptional case where one backward function offers the same
    /// contribution object to several inputs. It is thread-local because a tape's
    /// backward walk is single-threaded. Nested backward operations install their own
    /// map and restore the outer map when they finish, so inner execution cannot erase
    /// the outer operation's donation history.
    /// </remarks>
    private static class GradientAccumulatorOwnership<T>
    {
        [ThreadStatic]
        internal static Dictionary<Tensor<T>, Tensor<T>>? Owners;

        [ThreadStatic]
        internal static Stack<Dictionary<Tensor<T>, Tensor<T>>>? Available;
    }

    private static Dictionary<Tensor<T>, Tensor<T>> GetAccumulatorOwners<T>()
    {
        GradientAccumulatorOwnership<T>.Owners ??=
            new Dictionary<Tensor<T>, Tensor<T>>(ReferenceEqualityComparer<Tensor<T>>.Instance);
        return GradientAccumulatorOwnership<T>.Owners!;
    }

    /// <summary>
    /// Starts one backward operation's donation scope.
    /// </summary>
    /// <remarks>
    /// A node's incoming gradient is fully accumulated before its backward runs,
    /// so that node may donate the buffer to its first input. Ownership must then
    /// remain visible for the rest of the same backward call so a second input
    /// offered the identical object receives an isolated copy.
    /// </remarks>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static Dictionary<Tensor<T>, Tensor<T>>? BeginBackwardStep<T>()
    {
        var previous = GradientAccumulatorOwnership<T>.Owners;
        var available = GradientAccumulatorOwnership<T>.Available;
        var current = available is { Count: > 0 }
            ? available.Pop()
            : new Dictionary<Tensor<T>, Tensor<T>>(ReferenceEqualityComparer<Tensor<T>>.Instance);
        current.Clear();
        GradientAccumulatorOwnership<T>.Owners = current;
        return previous;
    }

    /// <summary>
    /// Releases one donation scope and restores the enclosing scope, if any.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static void EndBackwardStep<T>(Dictionary<Tensor<T>, Tensor<T>>? previous)
    {
        var completed = GradientAccumulatorOwnership<T>.Owners;
        GradientAccumulatorOwnership<T>.Owners = previous;
        if (completed is null || ReferenceEquals(completed, previous))
        {
            return;
        }

        completed.Clear();
        (GradientAccumulatorOwnership<T>.Available ??= new()).Push(completed);
    }

    private static bool IsAccumulatorBufferOwned<T>(Tensor<T> buffer)
        => GetAccumulatorOwners<T>().ContainsKey(buffer);

    private static Tensor<T> TakeAccumulatorBuffer<T>(
        Tensor<T> destination,
        Tensor<T> contribution,
        IEngine engine)
    {
        var owners = GetAccumulatorOwners<T>();
        if (!owners.ContainsKey(contribution))
        {
            owners[contribution] = destination;
            return contribution;
        }

        // This contribution is borrowed by another destination. Use dedicated,
        // non-arena storage for the exceptional copy so its lifetime is not tied
        // to recyclable backward scratch. A contribution still on the device is copied
        // there (a fresh engine result is dedicated storage too); the host copy downloaded
        // it only for the next GPU op to upload it again (17 such round trips per step on
        // an LM's backward).
        Tensor<T> owned;
        if (engine.SupportsGpu && contribution.HasPendingGpuData)
        {
            owned = engine.TensorMultiplyScalar(contribution, AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>().One);
        }
        else
        {
            // The copy overwrites every element, so rent uninitialized: a zeroed fresh array per fan-out
            // (every residual add hands one gradient to two slots) paid page faults on top of the copy.
            owned = AiDotNet.Tensors.Helpers.TensorAllocator.RentUninitialized<T>(contribution._shape);
            contribution.CopyTo(owned.AsWritableSpan());
        }
        owners[owned] = destination;
        return owned;
    }

    private static void ReplaceAccumulatorBufferOwner<T>(
        Tensor<T> destination,
        Tensor<T> previous,
        Tensor<T> replacement)
    {
        var owners = GetAccumulatorOwners<T>();
        owners.Remove(previous);
        owners[replacement] = destination;
    }

    private static bool HasOverlappingStorage<T>(Tensor<T> left, Tensor<T> right)
    {
        if (!ReferenceEquals(left._storage, right._storage))
        {
            return false;
        }

        long leftStart = left._storageOffset;
        long rightStart = right._storageOffset;
        long leftEnd = leftStart + left.Length;
        long rightEnd = rightStart + right.Length;
        return leftStart < rightEnd && rightStart < leftEnd;
    }

    private static Tensor<T> AddAliasedContributionOutOfPlace<T>(
        Tensor<T> existing,
        Tensor<T> contribution,
        IEngine engine)
    {
        // Use the explicit destination API instead of TensorAdd. Besides making
        // ownership unambiguous, this cannot be collapsed by an algebraic/CSE
        // identity when both operands are the same tensor object.
        var accumulated = AiDotNet.Tensors.Helpers.TensorAllocator.RentUninitialized<T>(existing._shape);
        engine.TensorAddInto(accumulated, existing, contribution);
        return accumulated;
    }
}
