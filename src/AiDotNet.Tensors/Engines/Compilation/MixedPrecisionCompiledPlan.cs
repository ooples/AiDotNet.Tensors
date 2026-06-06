using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Compilation;

/// <summary>
/// Phase A of the fused mixed-dtype training plan (Tensors #558, follow-on to #557): a compile-once /
/// replay FORWARD over a mixed-dtype lazy graph. The single-type fused <see cref="CompiledTrainingPlan{T}"/>
/// only compiles <c>LazyNode&lt;float&gt;</c> and drops <see cref="CrossTypeLazyNode{TIn,TOut}"/> /
/// <c>LazyNode&lt;Half&gt;</c>; this plan keeps a heterogeneous, dtype-tagged, topologically-ordered node
/// list and replays each node's <c>Execute</c> into its stable output buffer — no per-step graph retrace,
/// no per-step allocation. Inputs read the buffers written earlier in the same pass (producers-first
/// order), and graph leaves (params / batch input) read their current data, so a training loop just
/// mutates leaf data and calls <see cref="Forward"/> again.
///
/// <para>This is the forward half. Phase B captures the backward (<see cref="MixedPrecisionGraphBackward"/>)
/// into a matching replayable pass; C adds the optimizer + loss scaling; D/E take it to GPU + AiDotNet.
/// Gated behind <c>AIDOTNET_FP16_ACTIVATIONS</c>; the default FP32 fused path is untouched.</para>
/// </summary>
public sealed class MixedPrecisionCompiledPlan
{
    private readonly IEngine _engine;
    private readonly ILazyNode[] _order;   // producers-first topological order
    private readonly Tensor<float> _output;

    /// <summary>The compiled node list in execution (producers-first) order — used by the Phase B backward.</summary>
    internal IReadOnlyList<ILazyNode> Order => _order;
    internal Tensor<float> Output => _output;

    // ── FP16 activation paging (Tensors #558): store the FP32-op activations as Half between their last
    // forward use and their backward use, freeing the float backing in between, so peak resident float
    // memory is the working set rather than the whole activation set. This is the piece that actually
    // realizes the resident-memory win on transformers (matmul outputs are already Half; the big FP32
    // activations — norm/GELU/residual outputs — are what these page). Opt-in (test override or env).
    internal static bool? PagingTestOverride;
    private static bool PagingEnabledDefault =>
        PagingTestOverride ?? (Environment.GetEnvironmentVariable("AIDOTNET_FP16_PAGING") == "1");

    private readonly bool _paging;
    private readonly DirectGpuTensorEngine? _gpuFp16;                      // non-null ⇒ page via on-device FP16 compression
    private Dictionary<Tensor<float>, Half[]>? _halfStore;                 // dropped activations, held as Half (CPU paging)
    private List<Tensor<float>>[]? _pageOutAt;                             // per fwd step: page out after running it
    private Dictionary<ILazyNode, List<Tensor<float>>>? _bwdReadByNode;    // float activations a node's backward reads
    private Dictionary<Tensor<float>, int>? _bwdConsumerCount;            // total backward reads per activation
    private HashSet<Tensor<float>>? _droppedThisStep;                     // storage freed this step; restore next Forward

    private MixedPrecisionCompiledPlan(IEngine engine, ILazyNode[] order, Tensor<float> output, bool paging)
    {
        _engine = engine;
        _order = order;
        _output = output;
        _paging = paging;
        _gpuFp16 = (paging && engine is DirectGpuTensorEngine dg
                    && Environment.GetEnvironmentVariable("AIDOTNET_FP16_GPU_CACHE") == "1") ? dg : null;
        if (_paging) BuildPagingSchedule();
    }

    /// <summary>
    /// Public entry point (Phase E): trace <paramref name="forward"/> under an FP16 autocast scope with
    /// activation storage forced on, then compile the resulting mixed-dtype graph. The delegate builds the
    /// loss by calling ordinary engine ops on the parameter/input tensors (matmuls auto-emit FP16
    /// activations); it must return the scalar loss tensor. The caller then drives training via
    /// <see cref="Step"/>. This is the surface AiDotNet's training path calls — it manages the internal
    /// GraphMode / AutocastScope / activation-storage flag so external callers don't touch internals.
    /// </summary>
    public static MixedPrecisionCompiledPlan Trace(Func<Tensor<float>> forward, IEngine? engine = null)
    {
        if (forward is null) throw new ArgumentNullException(nameof(forward));
        engine ??= AiDotNetEngine.Current;

        var scope = new LazyTensorScope(null);
        var prevForce = MixedPrecisionEmit.TestOverrideEnabled;
        MixedPrecisionEmit.TestOverrideEnabled = true; // force FP16 activation emission for this trace
        Tensor<float> loss;
        using (new Gpu.AutocastScope(Gpu.PrecisionMode.Float16))
        {
            GraphMode.SetCurrent(scope);
            try { loss = forward(); }
            finally { GraphMode.SetCurrent(null); MixedPrecisionEmit.TestOverrideEnabled = prevForce; }
        }
        return Compile(loss, engine);
    }

    /// <summary>
    /// Compile the mixed-dtype graph reachable from <paramref name="finalOutput"/> into a replayable
    /// forward. Detaches each node's output from its lazy source so replay writes buffers directly.
    /// </summary>
    public static MixedPrecisionCompiledPlan Compile(Tensor<float> finalOutput, IEngine? engine = null)
    {
        if (finalOutput is null) throw new ArgumentNullException(nameof(finalOutput));
        engine ??= AiDotNetEngine.Current;
        if (finalOutput.LazySource is not ILazyNode root)
            throw new InvalidOperationException("finalOutput has no lazy source — nothing to compile.");

        var order = TopoOrder(root).ToArray();

        // Detach outputs so AsWritableSpan/AsSpan during replay don't re-trigger Realize.
        foreach (var n in order) n.ClearOutputLazySource();

        return new MixedPrecisionCompiledPlan(engine, order, finalOutput, PagingEnabledDefault);
    }

    /// <summary>Replay the forward: run every node's Execute into its stable buffer; return the output.
    /// With paging on, each activation is downcast to Half and its float backing freed after its last
    /// forward use, so peak resident float = the live working set, not the whole activation set.</summary>
    public Tensor<float> Forward()
    {
        var eng = _engine;
        for (int i = 0; i < _order.Length; i++)
        {
            // Just-in-time: re-give storage to THIS node's output if it was paged/freed last step, right
            // before its Execute writes into it. Restoring all upfront would re-inflate to full float and
            // erase the win; per-node keeps resident float = the live working set.
            if (_paging) RestoreOutputForReplay(_order[i]);
            RunForward(_order[i], eng);
            if (_paging)
                foreach (var t in _pageOutAt![i]) PageOut(t);
        }
        return _output;
    }

    // A node's float output may have had its backing freed on the prior step (paged out / backward-freed);
    // it is re-executed this step, so give it allocated storage just before. Data is overwritten by Execute.
    private void RestoreOutputForReplay(ILazyNode node)
    {
        if (_gpuFp16 is not null) return; // GPU mode: the op re-executes + re-caches a fresh FP32 buffer
        var o = FloatOutputOf(node);
        if (o is null) return;
        if (_droppedThisStep!.Remove(o))
        {
            _halfStore!.Remove(o);
            o.RestoreStorageFromBytes(new byte[(long)o.Length * sizeof(float)]);
        }
    }

    /// <summary>Backward over the captured order, with FP16 activation paging (page-in/free) when on.</summary>
    private MixedPrecisionGraphBackward.Result RunBackward(float seedScale)
    {
        if (!_paging)
            return MixedPrecisionGraphBackward.BackwardOverOrder(_order, _output, _engine, seedScale);

        // Refcount backward reads; page-in before a node's backward reads an activation, free after the last.
        var remaining = new Dictionary<Tensor<float>, int>(_bwdConsumerCount!, ReferenceEqualityComparer<Tensor<float>>.Instance);
        return MixedPrecisionGraphBackward.BackwardOverOrder(_order, _output, _engine, seedScale,
            onBeforeNodeBackward: node =>
            {
                if (_bwdReadByNode!.TryGetValue(node, out var reads))
                    foreach (var t in reads) PageIn(t);
            },
            onAfterNodeBackward: node =>
            {
                if (_bwdReadByNode!.TryGetValue(node, out var reads))
                    foreach (var t in reads)
                        if (--remaining[t] == 0) FreeFloat(t);
            });
    }

    // ── Paging schedule + primitives ──────────────────────────────────────────────────────────────
    internal int PageOutCount { get; private set; }

    private void BuildPagingSchedule()
    {
        _halfStore = new Dictionary<Tensor<float>, Half[]>(ReferenceEqualityComparer<Tensor<float>>.Instance);
        _droppedThisStep = new HashSet<Tensor<float>>(ReferenceEqualityComparer<Tensor<float>>.Instance);
        _bwdConsumerCount = new Dictionary<Tensor<float>, int>(ReferenceEqualityComparer<Tensor<float>>.Instance);
        _bwdReadByNode = new Dictionary<ILazyNode, List<Tensor<float>>>(ReferenceEqualityComparer<ILazyNode>.Instance);
        int n = _order.Length;
        _pageOutAt = new List<Tensor<float>>[n];
        for (int i = 0; i < n; i++) _pageOutAt[i] = new List<Tensor<float>>();

        // Intermediate float activations (node outputs), excluding the loss.
        var floatOutputs = new HashSet<Tensor<float>>(ReferenceEqualityComparer<Tensor<float>>.Instance);
        foreach (var node in _order)
        {
            var o = FloatOutputOf(node);
            if (o is not null && !ReferenceEquals(o, _output)) floatOutputs.Add(o);
        }

        var lastFwdUse = new Dictionary<Tensor<float>, int>(ReferenceEqualityComparer<Tensor<float>>.Instance);
        for (int i = 0; i < n; i++)
        {
            var node = _order[i];
            foreach (var t in FloatInputsOf(node))
                if (floatOutputs.Contains(t)) lastFwdUse[t] = i;

            List<Tensor<float>>? reads = null;
            foreach (var t in BwdReadsFloat(node))
            {
                if (!floatOutputs.Contains(t)) continue;
                (reads ??= new List<Tensor<float>>()).Add(t);
                _bwdConsumerCount[t] = (_bwdConsumerCount.TryGetValue(t, out var c) ? c : 0) + 1;
            }
            if (reads is not null) _bwdReadByNode[node] = reads;
        }

        // Page out an activation after its last forward use — only if its backward actually reads it.
        foreach (var kv in lastFwdUse)
            if (_bwdConsumerCount.TryGetValue(kv.Key, out var c) && c > 0)
                _pageOutAt[kv.Value].Add(kv.Key);
    }

    private static Tensor<float>? FloatOutputOf(ILazyNode node) => node switch
    {
        LazyNode<float> lf => lf.Output,
        CrossTypeLazyNode<Half, float> up => up.Output,
        _ => null
    };

    private static IEnumerable<Tensor<float>> FloatInputsOf(ILazyNode node)
    {
        switch (node)
        {
            case LazyNode<float> lf:
                foreach (var t in lf.GetInputsArray()) yield return t;
                break;
            case CrossTypeLazyNode<float, Half> down:
                yield return down.Input;
                break;
        }
    }

    // Only LazyNode<float> backward reads float activations (its inputs + output). Cross-type backwards
    // just cast gradients; LazyNode<Half> backwards read Half (already small, not paged).
    private static IEnumerable<Tensor<float>> BwdReadsFloat(ILazyNode node)
    {
        if (node is LazyNode<float> lf)
        {
            foreach (var t in lf.GetInputsArray()) yield return t;
            yield return lf.Output;
        }
    }

    private void PageOut(Tensor<float> t)
    {
        if (_gpuFp16 is not null) { _gpuFp16.CompressActivationFp16(t); PageOutCount++; return; }
        if (_halfStore!.ContainsKey(t)) return;
        var span = t.AsSpan();
        var h = new Half[span.Length];
        for (int i = 0; i < span.Length; i++) h[i] = (Half)span[i];
        if (t.TryDropStorageForStreaming()) { _halfStore[t] = h; _droppedThisStep!.Add(t); PageOutCount++; }
        // else: shared/view storage — can't drop safely; leave float resident (discard h).
    }

    private void PageIn(Tensor<float> t)
    {
        if (_gpuFp16 is not null) { _gpuFp16.UpcastActivationFp32(t); return; }
        if (!_halfStore!.TryGetValue(t, out var h)) return; // already float-resident
        var f = new float[h.Length];
        for (int i = 0; i < h.Length; i++) f[i] = (float)h[i];
        t.RestoreStorageFromBytes(MemoryMarshal.AsBytes(f.AsSpan()));
        _halfStore.Remove(t);
    }

    private void FreeFloat(Tensor<float> t)
    {
        // GPU FP16 mode: re-compress after the last backward read (frees the upcast FP32 GPU buffer).
        if (_gpuFp16 is not null) { _gpuFp16.CompressActivationFp16(t); return; }
        // CPU mode: after the last backward read the activation is float-resident and never read again.
        if (!_halfStore!.ContainsKey(t) && t.TryDropStorageForStreaming()) _droppedThisStep!.Add(t);
    }

    /// <summary>
    /// Phase B — compiled mixed-dtype backward. Reverse pass over the captured node order (no re-topo,
    /// and it works after <see cref="Compile"/> detached the lazy sources), seeding dL/dL=ones at the
    /// plan output. Returns the FP32 + FP16 gradient maps. Forward must have been replayed first so the
    /// activations are current. Delegates to the single shared dispatch in
    /// <see cref="MixedPrecisionGraphBackward.BackwardOverOrder"/>.
    /// </summary>
    internal MixedPrecisionGraphBackward.Result Backward()
        => RunBackward(1f);

    /// <summary>Outcome of one <see cref="Step"/>: the (unscaled) loss value and whether FP16 overflowed.</summary>
    public readonly struct StepResult
    {
        public readonly float Loss;
        public readonly bool FoundInfNan;
        public StepResult(float loss, bool infNan) { Loss = loss; FoundInfNan = infNan; }
    }

    /// <summary>
    /// Phase C — one compiled mixed-precision training step: replay forward, scaled mixed-dtype backward,
    /// SGD update on the FP32 master <paramref name="parameters"/>. With a <paramref name="scaler"/> the
    /// backward seed is loss-scaled and grads are unscaled in FP32 (cannot re-underflow); on FP16 overflow
    /// the optimizer step is SKIPPED and the scaler backs off (Micikevicius et al. AMP). Returns the
    /// (unscaled) scalar loss and the overflow flag. The plan output must be the scalar loss.
    /// </summary>
    public StepResult Step(IReadOnlyList<Tensor<float>> parameters, float learningRate, GradScaler? scaler = null)
    {
        if (parameters is null) throw new ArgumentNullException(nameof(parameters));
        var eng = _engine;

        Forward();
        float loss = _output.Length > 0 ? _output.ToArray()[0] : 0f;

        float scale = scaler?.Scale ?? 1f;
        var grads = RunBackward(scale);
        float invScale = 1f / scale;

        // Collect each param's grad (FP32 master space), unscale, check finiteness.
        var pgrads = new Tensor<float>[parameters.Count];
        bool infNan = false;
        for (int i = 0; i < parameters.Count; i++)
        {
            var p = parameters[i];
            Tensor<float>? g = grads.Fp32.TryGetValue(p, out var gf) ? gf : null;
            if (g is null) continue; // param not on the path this step
            var span = g.AsWritableSpan();
            for (int k = 0; k < span.Length; k++)
            {
                span[k] *= invScale;
                if (float.IsNaN(span[k]) || float.IsInfinity(span[k])) infNan = true;
            }
            pgrads[i] = g;
        }

        // Skip the update on overflow so a corrupted (inf/nan) gradient never touches the master weights.
        if (!infNan)
        {
            for (int i = 0; i < parameters.Count; i++)
            {
                var g = pgrads[i];
                if (g is null) continue;
                var w = parameters[i].AsWritableSpan();
                var gs = g.AsSpan();
                for (int k = 0; k < w.Length; k++) w[k] -= learningRate * gs[k];
                parameters[i].IncrementVersion();
            }
        }

        scaler?.Update(infNan);
        return new StepResult(loss, infNan);
    }

    // FP32 Adam master state (moments live in FP32 alongside the FP32 master weights — Micikevicius AMP).
    private Dictionary<Tensor<float>, (float[] M, float[] V)>? _adamState;
    private int _adamStep;

    /// <summary>
    /// Phase (fused-Adam): one compiled mixed-precision training step with the Adam optimizer. Identical
    /// to <see cref="Step"/> but the FP32 master update is Adam (m/v moments kept in FP32, bias-corrected),
    /// matching the optimizer Adam-configured models (e.g. the cortex) use. Loss scaling + skip-on-overflow
    /// via <paramref name="scaler"/> as in <see cref="Step"/>. The plan output must be the scalar loss.
    /// </summary>
    public StepResult StepAdam(
        IReadOnlyList<Tensor<float>> parameters,
        float learningRate,
        float beta1 = 0.9f, float beta2 = 0.999f, float epsilon = 1e-8f, float weightDecay = 0f,
        GradScaler? scaler = null)
    {
        if (parameters is null) throw new ArgumentNullException(nameof(parameters));
        var eng = _engine;

        Forward();
        float loss = _output.Length > 0 ? _output.ToArray()[0] : 0f;

        float scale = scaler?.Scale ?? 1f;
        var grads = RunBackward(scale);
        float invScale = 1f / scale;

        var pgrads = new Tensor<float>[parameters.Count];
        bool infNan = false;
        for (int i = 0; i < parameters.Count; i++)
        {
            if (!grads.Fp32.TryGetValue(parameters[i], out var g)) continue;
            var span = g.AsWritableSpan();
            for (int k = 0; k < span.Length; k++)
            {
                span[k] *= invScale;
                if (float.IsNaN(span[k]) || float.IsInfinity(span[k])) infNan = true;
            }
            pgrads[i] = g;
        }

        // Skip the Adam update (and the step-count/moment advance) on overflow — corrupt grads must never
        // touch the master weights or moments; the scaler backs off and the step is retried at lower scale.
        if (!infNan)
        {
            _adamStep++;
            float bc1 = 1f - (float)Math.Pow(beta1, _adamStep);
            float bc2 = 1f - (float)Math.Pow(beta2, _adamStep);
            _adamState ??= new Dictionary<Tensor<float>, (float[], float[])>(ReferenceEqualityComparer<Tensor<float>>.Instance);

            for (int i = 0; i < parameters.Count; i++)
            {
                var g = pgrads[i];
                if (g is null) continue;
                var p = parameters[i];
                if (!_adamState.TryGetValue(p, out var st))
                {
                    st = (new float[p.Length], new float[p.Length]);
                    _adamState[p] = st;
                }
                var w = p.AsWritableSpan();
                var gs = g.AsSpan();
                var m = st.M; var v = st.V;
                for (int k = 0; k < w.Length; k++)
                {
                    float gk = gs[k];
                    if (weightDecay != 0f) gk += weightDecay * w[k]; // L2 (matches AiDotNet's Adam)
                    m[k] = beta1 * m[k] + (1f - beta1) * gk;
                    v[k] = beta2 * v[k] + (1f - beta2) * gk * gk;
                    float mhat = m[k] / bc1;
                    float vhat = v[k] / bc2;
                    w[k] -= learningRate * mhat / ((float)Math.Sqrt(vhat) + epsilon);
                }
                p.IncrementVersion();
            }
        }

        scaler?.Update(infNan);
        return new StepResult(loss, infNan);
    }

    private static void RunForward(ILazyNode node, IEngine eng)
    {
        switch (node)
        {
            case LazyNode<float> lf: lf.Execute(eng, lf.Output); break;
            case LazyNode<Half> lh: lh.Execute(eng, lh.Output); break;
            case CrossTypeLazyNode<float, Half> down: down.Execute(eng, down.Output); break;
            case CrossTypeLazyNode<Half, float> up: up.Execute(eng, up.Output); break;
            default:
                throw new NotSupportedException(
                    $"MixedPrecisionCompiledPlan: unsupported node type {node.GetType().Name}. " +
                    "Mixed-precision graphs use float/Half LazyNode and float<->Half CrossTypeLazyNode only.");
        }
    }

    /// <summary>Iterative post-order DFS — producers-first (each node after its inputs).</summary>
    private static List<ILazyNode> TopoOrder(ILazyNode root)
    {
        var order = new List<ILazyNode>();
        var state = new Dictionary<ILazyNode, int>();
        var stack = new Stack<ILazyNode>();
        stack.Push(root);
        while (stack.Count > 0)
        {
            var node = stack.Peek();
            if (state.TryGetValue(node, out var s))
            {
                if (s == 0) { state[node] = 1; order.Add(node); }
                stack.Pop();
                continue;
            }
            state[node] = 0;
            foreach (var inp in node.GetInputNodes())
                if (!state.ContainsKey(inp))
                    stack.Push(inp);
        }
        return order;
    }
}
