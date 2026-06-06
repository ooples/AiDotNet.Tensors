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
internal sealed class MixedPrecisionCompiledPlan
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
    public MixedPrecisionGraphBackward.Result Backward()
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
