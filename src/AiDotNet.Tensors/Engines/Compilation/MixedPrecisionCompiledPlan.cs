using System;
using System.Collections.Generic;
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

    private MixedPrecisionCompiledPlan(IEngine engine, ILazyNode[] order, Tensor<float> output)
    {
        _engine = engine;
        _order = order;
        _output = output;
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

        return new MixedPrecisionCompiledPlan(engine, order, finalOutput);
    }

    /// <summary>Replay the forward: run every node's Execute into its stable buffer; return the output.</summary>
    public Tensor<float> Forward()
    {
        var eng = _engine;
        for (int i = 0; i < _order.Length; i++)
            RunForward(_order[i], eng);
        return _output;
    }

    /// <summary>
    /// Phase B — compiled mixed-dtype backward. Reverse pass over the captured node order (no re-topo,
    /// and it works after <see cref="Compile"/> detached the lazy sources), seeding dL/dL=ones at the
    /// plan output. Returns the FP32 + FP16 gradient maps. Forward must have been replayed first so the
    /// activations are current. Delegates to the single shared dispatch in
    /// <see cref="MixedPrecisionGraphBackward.BackwardOverOrder"/>.
    /// </summary>
    internal MixedPrecisionGraphBackward.Result Backward()
        => MixedPrecisionGraphBackward.BackwardOverOrder(_order, _output, _engine);

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
        var grads = MixedPrecisionGraphBackward.BackwardOverOrder(_order, _output, eng, scale);
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
        var grads = MixedPrecisionGraphBackward.BackwardOverOrder(_order, _output, eng, scale);
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
