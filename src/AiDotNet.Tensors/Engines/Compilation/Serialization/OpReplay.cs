using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Compilation.Serialization;

/// <summary>
/// Replays a compiled step through the engine by calling the engine method
/// directly (not via a closure). Under GraphMode the engine records a
/// LazyNode, which is exactly what the training plan compiler needs to
/// build the backward pass. Used by <see cref="TrainingPlanReader"/>.
/// </summary>
internal static class OpReplay
{
    /// <summary>
    /// Calls the engine method that corresponds to <paramref name="opType"/>,
    /// using the deserialized <paramref name="inputs"/> and
    /// <paramref name="savedState"/>. Under active GraphMode the engine
    /// records a LazyNode. Returns the output tensor (lazy or eager) so
    /// the caller can wire it as input to subsequent ops.
    /// </summary>
    internal static Tensor<T>? ReplayThroughEngine<T>(
        IEngine engine,
        OpType opType,
        string opName,
        Tensor<T>[] inputs,
        object[]? savedState)
    {
        return opType switch
        {
            // ── Parameterless unary ─────────────────────────────────
            OpType.Sigmoid       => engine.Sigmoid(inputs[0]),
            OpType.Tanh          => engine.Tanh(inputs[0]),
            OpType.ReLU          => engine.ReLU(inputs[0]),
            OpType.GELU          => engine.GELU(inputs[0]),
            OpType.Swish         => engine.Swish(inputs[0]),
            OpType.Mish          => engine.Mish(inputs[0]),
            OpType.Softplus      => engine.Softplus(inputs[0]),
            OpType.TensorExp     => engine.TensorExp(inputs[0]),
            OpType.TensorLog     => engine.TensorLog(inputs[0]),
            OpType.TensorSqrt    => engine.TensorSqrt(inputs[0]),
            OpType.TensorAbs     => engine.TensorAbs(inputs[0]),
            OpType.TensorNegate  => engine.TensorNegate(inputs[0]),
            OpType.TensorTranspose => engine.TensorTranspose(inputs[0]),
            OpType.Floor         => engine.TensorFloor(inputs[0]),
            OpType.Ceiling       => engine.TensorCeiling(inputs[0]),
            OpType.Round         => engine.TensorRound(inputs[0]),
            OpType.Sin           => engine.TensorSin(inputs[0]),
            OpType.Cos           => engine.TensorCos(inputs[0]),

            // ── Parameterless binary ────────────────────────────────
            OpType.TensorAdd      => engine.TensorAdd(inputs[0], inputs[1]),
            OpType.TensorSubtract => engine.TensorSubtract(inputs[0], inputs[1]),
            OpType.TensorMultiply => engine.TensorMultiply(inputs[0], inputs[1]),
            OpType.TensorDivide   => engine.TensorDivide(inputs[0], inputs[1]),
            OpType.TensorMax      => engine.TensorMax(inputs[0], inputs[1]),
            OpType.TensorMatMul   => engine.TensorMatMul(inputs[0], inputs[1]),

            // ── Broadcast binary ────────────────────────────────────
            OpType.TensorBroadcastAdd      => engine.TensorBroadcastAdd(inputs[0], inputs[1]),
            OpType.TensorBroadcastSubtract => engine.TensorBroadcastSubtract(inputs[0], inputs[1]),
            OpType.TensorBroadcastMultiply => engine.TensorBroadcastMultiply(inputs[0], inputs[1]),

            // ── Batch MatMul ────────────────────────────────────────
            OpType.BatchMatMul => engine.TensorBatchMatMul(inputs[0], inputs[1]),

            // ── Parameterized activations ───────────────────────────
            OpType.ELU => engine.ELU(inputs[0],
                savedState is { Length: > 0 } && savedState[0] is double d ? d : 1.0),
            OpType.LeakyReLU => ReplayLeakyReLU(engine, inputs[0], savedState),

            // ── Reduce family ───────────────────────────────────────
            OpType.ReduceSum => engine.ReduceSum(inputs[0],
                savedState is { Length: > 0 } && savedState[0] is int axis ? new[] { axis }
                : savedState is { Length: > 0 } && savedState[0] is int[] axArr ? axArr
                : null),
            OpType.Softmax => engine.Softmax(inputs[0],
                savedState is { Length: > 0 } && savedState[0] is int a ? a : -1),
            OpType.Mean => ReplayMean(engine, inputs[0], savedState),

            // ── Conv family ─────────────────────────────────────────
            OpType.Conv2D => engine.Conv2D(inputs[0], inputs[1],
                savedState is { Length: > 0 } && savedState[0] is int[] s && s.Length > 0 ? s[0] : 1,
                savedState is { Length: > 1 } && savedState[1] is int[] p && p.Length > 0 ? p[0] : 0,
                savedState is { Length: > 2 } && savedState[2] is int[] dd && dd.Length > 0 ? dd[0] : 1),
            OpType.MaxPool2D => engine.MaxPool2D(inputs[0],
                savedState is { Length: > 0 } && savedState[0] is int[] ps && ps.Length > 0 ? ps[0] : 2,
                savedState is { Length: > 1 } && savedState[1] is int[] st && st.Length > 0 ? st[0] :
                    (savedState is { Length: > 0 } && savedState[0] is int[] ps2 && ps2.Length > 0 ? ps2[0] : 2)),
            OpType.AvgPool2D => engine.AvgPool2D(inputs[0],
                savedState is { Length: > 0 } && savedState[0] is int[] aps && aps.Length > 0 ? aps[0] : 2,
                savedState is { Length: > 1 } && savedState[1] is int[] ast && ast.Length > 0 ? ast[0] :
                    (savedState is { Length: > 0 } && savedState[0] is int[] aps2 && aps2.Length > 0 ? aps2[0] : 2)),

            // ── Normalization ───────────────────────────────────────
            OpType.BatchNorm => engine.BatchNorm(inputs[0], inputs[1], inputs[2],
                savedState is { Length: > 2 } && savedState[2] is double be ? be : 1e-5, out _, out _),
            OpType.LayerNorm => engine.LayerNorm(inputs[0], inputs[1], inputs[2],
                savedState is { Length: > 2 } && savedState[2] is double le ? le : 1e-5, out _, out _),

            // ── Attention ───────────────────────────────────────────
            // Matches RebuildAttention: scale comes from savedState[0] (nullable
            // double; null means "recompute from d_k"). Mask is not serialized
            // through the current SavedStateSerializer tag set — plans with
            // explicit masks load back with mask=null; see RebuildAttention for
            // the full rationale. Attention weights out-param is discarded —
            // only re-materialized under eager replay if a consumer explicitly
            // records it, which the training replay path does not.
            OpType.ScaledDotProductAttention => engine.ScaledDotProductAttention(
                inputs[0], inputs[1], inputs[2],
                mask: null,
                scale: savedState is { Length: > 0 } && savedState[0] is double attnScale ? attnScale : null,
                attentionWeights: out _),

            // ── FusedLinear: input @ weight (+ bias) ────────────────
            OpType.FusedLinear => ReplayFusedLinear(engine, inputs),

            // ── Loss ────────────────────────────────────────────────
            OpType.CrossEntropyLoss => engine.TensorCrossEntropyLoss(inputs[0], inputs[1]),

            _ => throw new NotSupportedException(
                $"OpType {opType} ('{opName}') is not supported for training plan replay."),
        };
    }

    private static Tensor<T> ReplayLeakyReLU<T>(IEngine engine, Tensor<T> input, object[]? savedState)
    {
        // Alpha is serialized as double (T-erased). Convert back to T via the
        // numeric-ops registry so T=float, double, Half, etc. all round-trip
        // correctly. `default(T)` would collapse LeakyReLU to plain ReLU for
        // numeric types — matches the gap that RebuildLeakyReLU was fixing.
        double alphaDouble = savedState is { Length: > 0 } && savedState[0] is double d ? d : 0.01;
        var alpha = MathHelper.GetNumericOperations<T>().FromDouble(alphaDouble);
        return engine.LeakyReLU(input, alpha);
    }

    private static Tensor<T> ReplayMean<T>(IEngine engine, Tensor<T> input, object[]? savedState)
    {
        // Parallels RebuildMean: null-axes maps to a scalar TensorMean, wrapped
        // in a rank-0 tensor so downstream consumers see a Tensor<T> regardless
        // of the reduction shape.
        int[]? axes = savedState is { Length: > 0 } && savedState[0] is int axis ? new[] { axis }
                    : savedState is { Length: > 0 } && savedState[0] is int[] axArr ? axArr
                    : null;
        bool keepDims = savedState is { Length: > 1 } && savedState[1] is bool kd && kd;
        if (axes is null)
        {
            var scalar = new Tensor<T>(new[] { 1 });
            scalar.AsWritableSpan()[0] = engine.TensorMean(input);
            return scalar;
        }
        return engine.ReduceMean(input, axes, keepDims);
    }

    private static Tensor<T> ReplayFusedLinear<T>(IEngine engine, Tensor<T>[] inputs)
    {
        // Decomposes to MatMul + optional BroadcastAdd. Mirror of RebuildFusedLinear.
        var r = engine.TensorMatMul(inputs[0], inputs[1]);
        return inputs.Length > 2 ? engine.TensorBroadcastAdd(r, inputs[2]) : r;
    }
}
