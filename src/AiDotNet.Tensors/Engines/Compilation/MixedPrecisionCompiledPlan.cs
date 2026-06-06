using System;
using System.Collections.Generic;
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
