using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Compilation;

/// <summary>
/// Mixed-dtype reverse-mode backward over the lazy computation graph (Tensors #555 Phase 2,
/// docs/fp16-activation-storage-design.md). This is the lazy-graph analog of <see cref="MixedPrecisionTape"/>,
/// and the algorithm the compiled training plan's backward will call once FP16 activation buffers exist
/// in the graph: where the eager path had two SEPARATE tapes joined only by a Gauss-Seidel sweep, the
/// lazy graph is a single unified DAG — <see cref="CrossTypeLazyNode{TIn,TOut}"/> links an FP16 node to
/// its FP32 neighbour through <see cref="ILazyNode.GetInputNodes"/> — so ONE reverse-topological pass is
/// exact, no fixpoint required.
///
/// <para>Each node's gradient flows in its own element type: an <see cref="LazyNode{T}"/> runs its
/// <see cref="BackwardFunction{T}"/> against the matching-dtype grad map; a cross-type cast node moves
/// the gradient from the output's grad map into the input's via <see cref="CrossTypeLazyNode{TIn,TOut}.Backward"/>
/// (the verified cast bridge). Params/inputs (graph leaves) accumulate their grads in the map of their
/// own dtype — FP32 for FP32 params, FP16 for FP16 params (the optimizer up-casts + unscales, exactly as
/// <see cref="MixedPrecisionTape"/> does under a <see cref="GradScaler"/>).</para>
///
/// <para>Self-contained and additive: it does not touch the hot single-type <c>CompiledTrainingPlan</c>;
/// it consumes the graph the plan is built from. Gated by MixedPrecisionGraphBackwardTests (CPU).</para>
/// </summary>
internal static class MixedPrecisionGraphBackward
{
    public sealed class Result
    {
        public Dictionary<Tensor<float>, Tensor<float>> Fp32 { get; }
        public Dictionary<Tensor<Half>, Tensor<Half>> Fp16 { get; }
        public Result(Dictionary<Tensor<float>, Tensor<float>> fp32, Dictionary<Tensor<Half>, Tensor<Half>> fp16)
        {
            Fp32 = fp32;
            Fp16 = fp16;
        }
    }

    /// <summary>
    /// Reverse-mode gradients of a scalar FP32 <paramref name="loss"/> over the mixed-dtype lazy graph
    /// reachable from <c>loss.LazySource</c>. The graph must be realized (forward executed) first.
    /// </summary>
    public static Result Backward(Tensor<float> loss, IEngine? engine = null)
    {
        if (loss is null) throw new ArgumentNullException(nameof(loss));
        engine ??= AiDotNetEngine.Current;

        if (loss.LazySource is not ILazyNode root)
            return new Result(
                new Dictionary<Tensor<float>, Tensor<float>>(ReferenceEqualityComparer<Tensor<float>>.Instance),
                new Dictionary<Tensor<Half>, Tensor<Half>>(ReferenceEqualityComparer<Tensor<Half>>.Instance));

        // Producers-first topological order via iterative post-order DFS (graphs can be deep —
        // avoid recursion). Reverse of this list is consumers-first = correct backward order.
        var topo = TopoOrder(root);
        return BackwardOverOrder(topo, loss, engine);
    }

    /// <summary>
    /// Reverse pass over a PRE-COMPUTED producers-first node order, seeding dL/dL = ones at
    /// <paramref name="lossOutput"/>. Shared by the topo-from-loss <see cref="Backward"/> and by
    /// <see cref="MixedPrecisionCompiledPlan"/>, whose captured order survives lazy-source detachment
    /// (so it can't re-topo from <c>loss.LazySource</c>). Single source of the per-node backward dispatch.
    /// </summary>
    internal static Result BackwardOverOrder(IReadOnlyList<ILazyNode> producersFirstOrder, Tensor<float> lossOutput, IEngine engine, float seedScale = 1f,
        Action<ILazyNode>? onBeforeNodeBackward = null, Action<ILazyNode>? onAfterNodeBackward = null)
    {
        var fp32 = new Dictionary<Tensor<float>, Tensor<float>>(ReferenceEqualityComparer<Tensor<float>>.Instance);
        var fp16 = new Dictionary<Tensor<Half>, Tensor<Half>>(ReferenceEqualityComparer<Tensor<Half>>.Instance);

        // Seed dL/dL = seedScale (ones, optionally pre-multiplied by the loss scale so small FP16
        // gradients stay representable — see GradScaler; default 1 = unscaled).
        var seed = new float[lossOutput.Length];
        for (int i = 0; i < seed.Length; i++) seed[i] = seedScale;
        fp32[lossOutput] = new Tensor<float>(seed, lossOutput._shape);

        for (int i = producersFirstOrder.Count - 1; i >= 0; i--)
        {
            // FP16 activation paging hook: restore (upcast) this node's paged activations before its
            // backward reads them (MixedPrecisionCompiledPlan.PageIn); free them again after (PageOut).
            onBeforeNodeBackward?.Invoke(producersFirstOrder[i]);
            switch (producersFirstOrder[i])
            {
                case LazyNode<float> lf:
                    RunSingleType(lf.Output, lf.BackwardFn, lf.GetInputsArray, lf.SavedState, fp32, engine);
                    break;
                case LazyNode<Half> lh:
                    RunSingleType(lh.Output, lh.BackwardFn, lh.GetInputsArray, lh.SavedState, fp16, engine);
                    break;
                case CrossTypeLazyNode<float, Half> down: // FP32 input -> FP16 output (down-cast)
                    if (fp16.TryGetValue(down.Output, out var gDownOut))
                    {
                        var gIn = down.Backward(gDownOut, engine);
                        if (gIn is not null) Accumulate(fp32, down.Input, gIn, engine);
                    }
                    break;
                case CrossTypeLazyNode<Half, float> up: // FP16 input -> FP32 output (up-cast)
                    if (fp32.TryGetValue(up.Output, out var gUpOut))
                    {
                        var gIn = up.Backward(gUpOut, engine);
                        if (gIn is not null) Accumulate(fp16, up.Input, gIn, engine);
                    }
                    break;
                // Other cross-type nodes (e.g. Complex FFT) are forward-only here — no grad edge.
            }
            onAfterNodeBackward?.Invoke(producersFirstOrder[i]);
        }

        return new Result(fp32, fp16);
    }

    private static void RunSingleType<T>(
        Tensor<T> output,
        BackwardFunction<T>? backwardFn,
        Func<Tensor<T>[]> getInputs,
        object[]? savedState,
        Dictionary<Tensor<T>, Tensor<T>> grads,
        IEngine engine)
    {
        if (backwardFn is null) return;
        if (!grads.TryGetValue(output, out var gradOut)) return; // output not downstream of loss
        var inputs = getInputs();
        backwardFn(gradOut, inputs, output, savedState ?? Array.Empty<object>(), engine, grads);
    }

    private static void Accumulate<T>(Dictionary<Tensor<T>, Tensor<T>> grads, Tensor<T> key, Tensor<T> contribution, IEngine engine)
    {
        grads[key] = grads.TryGetValue(key, out var existing)
            ? engine.TensorAdd(existing, contribution)
            : contribution;
    }

    /// <summary>Iterative post-order DFS — returns nodes producers-first (each node after its inputs).</summary>
    private static List<ILazyNode> TopoOrder(ILazyNode root)
    {
        var order = new List<ILazyNode>();
        var state = new Dictionary<ILazyNode, int>(); // 0 = entered, 1 = done
        var stack = new Stack<ILazyNode>();
        stack.Push(root);
        while (stack.Count > 0)
        {
            var node = stack.Peek();
            if (state.TryGetValue(node, out var s))
            {
                if (s == 0)
                {
                    state[node] = 1;
                    order.Add(node);   // all inputs already emitted
                }
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
