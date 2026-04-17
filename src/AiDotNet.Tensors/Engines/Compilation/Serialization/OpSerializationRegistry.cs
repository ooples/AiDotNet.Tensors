using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Compilation.Serialization;

/// <summary>
/// Maps each <see cref="OpType"/> to a closure-rebuilder that reconstitutes a
/// compiled step's <c>Action&lt;IEngine, Tensor&lt;T&gt;&gt;</c> from the
/// deserialized inputs, output buffer, and savedState.
///
/// Each rebuilder creates the same lambda the engine's GraphMode recording
/// originally built: call the engine method with the deserialized args, copy
/// result to the output buffer. This guarantees the deserialized plan behaves
/// identically to a freshly compiled one.
/// </summary>
internal static class OpSerializationRegistry<T>
{
    internal static Action<IEngine, Tensor<T>> RebuildForwardClosure(
        OpType opType,
        Tensor<T>[] inputs,
        Tensor<T> output,
        object[]? savedState)
    {
        return opType switch
        {
            // ── Parameterless unary ─────────────────────────────────────
            OpType.Sigmoid       => Unary(inputs, (e, a) => e.Sigmoid(a)),
            OpType.Tanh          => Unary(inputs, (e, a) => e.Tanh(a)),
            OpType.ReLU          => Unary(inputs, (e, a) => e.ReLU(a)),
            OpType.GELU          => Unary(inputs, (e, a) => e.GELU(a)),
            OpType.Swish         => Unary(inputs, (e, a) => e.Swish(a)),
            OpType.Mish          => Unary(inputs, (e, a) => e.Mish(a)),
            OpType.Softplus      => Unary(inputs, (e, a) => e.Softplus(a)),
            OpType.TensorExp     => Unary(inputs, (e, a) => e.TensorExp(a)),
            OpType.TensorLog     => Unary(inputs, (e, a) => e.TensorLog(a)),
            OpType.TensorSqrt    => Unary(inputs, (e, a) => e.TensorSqrt(a)),
            OpType.TensorAbs     => Unary(inputs, (e, a) => e.TensorAbs(a)),
            OpType.TensorNegate  => Unary(inputs, (e, a) => e.TensorNegate(a)),
            OpType.TensorTranspose => Unary(inputs, (e, a) => e.TensorTranspose(a)),
            OpType.Floor   => Unary(inputs, (e, a) => e.TensorFloor(a)),
            OpType.Ceiling => Unary(inputs, (e, a) => e.TensorCeiling(a)),
            OpType.Round   => Unary(inputs, (e, a) => e.TensorRound(a)),
            OpType.Sin           => Unary(inputs, (e, a) => e.TensorSin(a)),
            OpType.Cos           => Unary(inputs, (e, a) => e.TensorCos(a)),

            // ── Parameterless binary ────────────────────────────────────
            OpType.TensorAdd      => Binary(inputs, (e, a, b) => e.TensorAdd(a, b)),
            OpType.TensorSubtract => Binary(inputs, (e, a, b) => e.TensorSubtract(a, b)),
            OpType.TensorMultiply => Binary(inputs, (e, a, b) => e.TensorMultiply(a, b)),
            OpType.TensorDivide   => Binary(inputs, (e, a, b) => e.TensorDivide(a, b)),
            OpType.TensorMax      => Binary(inputs, (e, a, b) => e.TensorMax(a, b)),
            OpType.TensorMatMul   => Binary(inputs, (e, a, b) => e.TensorMatMul(a, b)),

            // ── Broadcast binary ────────────────────────────────────────
            OpType.TensorBroadcastAdd      => Binary(inputs, (e, a, b) => e.TensorBroadcastAdd(a, b)),
            OpType.TensorBroadcastSubtract => Binary(inputs, (e, a, b) => e.TensorBroadcastSubtract(a, b)),
            OpType.TensorBroadcastMultiply => Binary(inputs, (e, a, b) => e.TensorBroadcastMultiply(a, b)),

            // ── Batch MatMul ────────────────────────────────────────────
            OpType.BatchMatMul => Binary(inputs, (e, a, b) => e.TensorBatchMatMul(a, b)),

            // ── Parameterized activations ───────────────────────────────
            OpType.LeakyReLU => RebuildLeakyReLU(inputs, savedState),
            OpType.ELU       => RebuildELU(inputs, savedState),

            // ── Reduce family ───────────────────────────────────────────
            OpType.ReduceSum => RebuildReduceSum(inputs, savedState),
            OpType.Softmax   => RebuildSoftmax(inputs, savedState),
            OpType.Mean      => RebuildMean(inputs, savedState),

            // ── Conv family ─────────────────────────────────────────────
            OpType.Conv2D    => RebuildConv2D(inputs, savedState),
            OpType.MaxPool2D => RebuildMaxPool2D(inputs, savedState),
            OpType.AvgPool2D => RebuildAvgPool2D(inputs, savedState),

            // ── Normalization ───────────────────────────────────────────
            OpType.BatchNorm => RebuildBatchNorm(inputs, savedState),
            OpType.LayerNorm => RebuildLayerNorm(inputs, savedState),

            // ── Attention ───────────────────────────────────────────────
            OpType.ScaledDotProductAttention => RebuildAttention(inputs, savedState),

            // ── FusedLinear ─────────────────────────────────────────────
            OpType.FusedLinear => RebuildFusedLinear(inputs),

            // ── Loss ────────────────────────────────────────────────────
            OpType.CrossEntropyLoss => Binary(inputs, (e, a, b) => e.TensorCrossEntropyLoss(a, b)),

            // ── Ops not yet supported for serialization ─────────────────
            // These are valid OpTypes but their IEngine signatures need
            // individual wiring. They'll throw at load time with a
            // descriptive message so the caller knows to file a gap report.
            _ => throw new NotSupportedException(
                $"OpType {opType} is not yet supported by the plan serialization registry. " +
                "This plan cannot be deserialized. Re-compile from source instead."),
        };
    }

    /// <summary>
    /// Returns true only for OpTypes that have an actual rebuild handler in
    /// the switch above — NOT "anything except Unknown." Callers use this to
    /// decide whether a plan is serializable without having to catch
    /// <see cref="NotSupportedException"/> from a later rebuild, so the two
    /// sides must stay exactly aligned. When adding a new rebuild case,
    /// also add its OpType to the list below.
    /// </summary>
    internal static bool IsSupported(OpType opType) => opType switch
    {
        // Unary
        OpType.Sigmoid or OpType.Tanh or OpType.ReLU or OpType.GELU or
        OpType.Swish or OpType.Mish or OpType.Softplus or OpType.TensorExp or
        OpType.TensorLog or OpType.TensorSqrt or OpType.TensorAbs or
        OpType.TensorNegate or OpType.TensorTranspose or OpType.Floor or
        OpType.Ceiling or OpType.Round or OpType.Sin or OpType.Cos or
        // Binary
        OpType.TensorAdd or OpType.TensorSubtract or OpType.TensorMultiply or
        OpType.TensorDivide or OpType.TensorMax or OpType.TensorMatMul or
        OpType.TensorBroadcastAdd or OpType.TensorBroadcastSubtract or
        OpType.TensorBroadcastMultiply or OpType.BatchMatMul or
        // Parameterized activations + reduce family
        OpType.LeakyReLU or OpType.ELU or OpType.ReduceSum or OpType.Softmax or
        OpType.Mean or
        // Conv / pool / norm / attention / fused
        OpType.Conv2D or OpType.MaxPool2D or OpType.AvgPool2D or
        OpType.BatchNorm or OpType.LayerNorm or
        OpType.ScaledDotProductAttention or OpType.FusedLinear or
        // Loss
        OpType.CrossEntropyLoss => true,
        _ => false,
    };

    // ════════════════════════════════════════════════════════════════════
    // Generic builders
    // ════════════════════════════════════════════════════════════════════

    private static Action<IEngine, Tensor<T>> Unary(
        Tensor<T>[] inputs, Func<IEngine, Tensor<T>, Tensor<T>> op)
    {
        var a = inputs[0];
        return (eng, output) => { var r = op(eng, a); r.AsSpan().CopyTo(output.AsWritableSpan()); };
    }

    private static Action<IEngine, Tensor<T>> Binary(
        Tensor<T>[] inputs, Func<IEngine, Tensor<T>, Tensor<T>, Tensor<T>> op)
    {
        var a = inputs[0]; var b = inputs[1];
        return (eng, output) => { var r = op(eng, a, b); r.AsSpan().CopyTo(output.AsWritableSpan()); };
    }

    // ════════════════════════════════════════════════════════════════════
    // Parameterized activations
    // ════════════════════════════════════════════════════════════════════

    private static Action<IEngine, Tensor<T>> RebuildLeakyReLU(Tensor<T>[] inputs, object[]? state)
    {
        // Extract alpha from savedState. The serializer writes it as a double
        // (T-erased). The earlier implementation passed `default!` — for T=float
        // that's 0.0f, which collapses LeakyReLU to a plain ReLU. The engine
        // does not have a "use last-configured alpha" default; alpha must be
        // supplied on every call. So deserialized plans silently changed
        // behavior. Use MathHelper to convert double→T.
        double alphaDouble = state is { Length: > 0 } && state[0] is double d ? d : 0.01;
        var numOps = MathHelper.GetNumericOperations<T>();
        T alpha = numOps.FromDouble(alphaDouble);
        var a = inputs[0];
        return (eng, output) =>
        {
            var r = eng.LeakyReLU(a, alpha);
            r.AsSpan().CopyTo(output.AsWritableSpan());
        };
    }

    private static Action<IEngine, Tensor<T>> RebuildELU(Tensor<T>[] inputs, object[]? state)
    {
        double alpha = state is { Length: > 0 } && state[0] is double d ? d : 1.0;
        var a = inputs[0];
        return (eng, output) =>
        {
            var r = eng.ELU(a, alpha);
            r.AsSpan().CopyTo(output.AsWritableSpan());
        };
    }

    // ════════════════════════════════════════════════════════════════════
    // Reduce family
    // ════════════════════════════════════════════════════════════════════

    private static Action<IEngine, Tensor<T>> RebuildReduceSum(Tensor<T>[] inputs, object[]? state)
    {
        int[]? axes = state is { Length: > 0 } && state[0] is int axis ? new[] { axis }
                    : state is { Length: > 0 } && state[0] is int[] axArr ? axArr
                    : null;
        var a = inputs[0];
        return (eng, output) =>
        {
            var r = eng.ReduceSum(a, axes);
            r.AsSpan().CopyTo(output.AsWritableSpan());
        };
    }

    private static Action<IEngine, Tensor<T>> RebuildSoftmax(Tensor<T>[] inputs, object[]? state)
    {
        int axis = state is { Length: > 0 } && state[0] is int a ? a : -1;
        var inp = inputs[0];
        return (eng, output) =>
        {
            var r = eng.Softmax(inp, axis);
            r.AsSpan().CopyTo(output.AsWritableSpan());
        };
    }

    private static Action<IEngine, Tensor<T>> RebuildMean(Tensor<T>[] inputs, object[]? state)
    {
        // IEngine.ReduceMean(tensor, axes, keepDims) computes the mean along the
        // given axes. The earlier implementation called ReduceSum and copied it
        // to output — that produced the sum, not the mean. Deserialized plans
        // over- or under-predicted by a factor of N (where N = product of reduced
        // axis lengths), silently producing wildly wrong outputs.
        int[]? axes = state is { Length: > 0 } && state[0] is int axis ? new[] { axis }
                    : state is { Length: > 0 } && state[0] is int[] axArr ? axArr
                    : null;
        bool keepDims = state is { Length: > 1 } && state[1] is bool kd && kd;
        var a = inputs[0];
        return (eng, output) =>
        {
            // When axes is null, reduce over all axes to produce a scalar mean.
            var r = axes is null
                ? ScalarTensor(eng.TensorMean(a))
                : eng.ReduceMean(a, axes, keepDims);
            r.AsSpan().CopyTo(output.AsWritableSpan());
        };
    }

    /// <summary>Wraps a scalar T as a rank-0 (single-element) Tensor&lt;T&gt;.</summary>
    private static Tensor<T> ScalarTensor(T value)
    {
        var t = new Tensor<T>(new[] { 1 });
        t.AsWritableSpan()[0] = value;
        return t;
    }

    // ════════════════════════════════════════════════════════════════════
    // Conv family — SavedState: [int[]{s,s}, int[]{p,p}, int[]{d,d}]
    // ════════════════════════════════════════════════════════════════════

    private static Action<IEngine, Tensor<T>> RebuildConv2D(Tensor<T>[] inputs, object[]? state)
    {
        int stride   = state is { Length: > 0 } && state[0] is int[] s && s.Length > 0 ? s[0] : 1;
        int padding  = state is { Length: > 1 } && state[1] is int[] p && p.Length > 0 ? p[0] : 0;
        int dilation = state is { Length: > 2 } && state[2] is int[] d && d.Length > 0 ? d[0] : 1;
        var input = inputs[0]; var kernel = inputs[1];
        return (eng, output) =>
        {
            var r = eng.Conv2D(input, kernel, stride, padding, dilation);
            r.AsSpan().CopyTo(output.AsWritableSpan());
        };
    }

    private static Action<IEngine, Tensor<T>> RebuildMaxPool2D(Tensor<T>[] inputs, object[]? state)
    {
        int poolSize = state is { Length: > 0 } && state[0] is int[] ps && ps.Length > 0 ? ps[0] : 2;
        int stride   = state is { Length: > 1 } && state[1] is int[] st && st.Length > 0 ? st[0] : poolSize;
        var input = inputs[0];
        return (eng, output) =>
        {
            var r = eng.MaxPool2D(input, poolSize, stride);
            r.AsSpan().CopyTo(output.AsWritableSpan());
        };
    }

    private static Action<IEngine, Tensor<T>> RebuildAvgPool2D(Tensor<T>[] inputs, object[]? state)
    {
        int poolSize = state is { Length: > 0 } && state[0] is int[] ps && ps.Length > 0 ? ps[0] : 2;
        int stride   = state is { Length: > 1 } && state[1] is int[] st && st.Length > 0 ? st[0] : poolSize;
        var input = inputs[0];
        return (eng, output) =>
        {
            var r = eng.AvgPool2D(input, poolSize, stride);
            r.AsSpan().CopyTo(output.AsWritableSpan());
        };
    }

    // ════════════════════════════════════════════════════════════════════
    // Normalization — SavedState: [mean_tensor, var_tensor, epsilon]
    // IEngine BatchNorm: (input, gamma, beta, eps, out mean, out var)
    // ════════════════════════════════════════════════════════════════════

    private static Action<IEngine, Tensor<T>> RebuildBatchNorm(Tensor<T>[] inputs, object[]? state)
    {
        double eps = state is { Length: > 2 } && state[2] is double e ? e : 1e-5;
        var input = inputs[0]; var gamma = inputs[1]; var beta = inputs[2];
        return (eng, output) =>
        {
            var r = eng.BatchNorm(input, gamma, beta, eps, out _, out _);
            r.AsSpan().CopyTo(output.AsWritableSpan());
        };
    }

    private static Action<IEngine, Tensor<T>> RebuildLayerNorm(Tensor<T>[] inputs, object[]? state)
    {
        double eps = state is { Length: > 2 } && state[2] is double e ? e : 1e-5;
        var input = inputs[0]; var gamma = inputs[1]; var beta = inputs[2];
        return (eng, output) =>
        {
            var r = eng.LayerNorm(input, gamma, beta, eps, out _, out _);
            r.AsSpan().CopyTo(output.AsWritableSpan());
        };
    }

    // ════════════════════════════════════════════════════════════════════
    // Attention
    // ════════════════════════════════════════════════════════════════════

    private static Action<IEngine, Tensor<T>> RebuildAttention(Tensor<T>[] inputs, object[]? state)
    {
        // IEngine.ScaledDotProductAttention<T>(q, k, v, mask?, scale?, out attnWeights)
        //
        // scale: recorded as savedState[0] (nullable double). Null means
        // "use default 1/sqrt(d_k)" — the engine recomputes it from query rank.
        // Preserving the explicit value matters for plans that override it
        // (e.g. temperature-scaled attention).
        //
        // mask: NOT round-trippable through the current SavedStateSerializer —
        // the supported tag set covers Tensor<T> but not Tensor<bool>, and
        // mask tensors aren't carried through the Tensor<T>[] inputs array
        // either. Plans compiled with a non-null mask load back with mask=null;
        // callers who need mask persistence must re-compile from source. This
        // is a known limitation tracked against issue #166 — fully fixing it
        // needs a TagBoolTensor addition to SavedStateSerializer.
        double? scale = state is { Length: > 0 } && state[0] is double sv ? sv : null;
        var q = inputs[0]; var k = inputs[1]; var v = inputs[2];
        return (eng, output) =>
        {
            var r = eng.ScaledDotProductAttention(q, k, v, null, scale, out _);
            r.AsSpan().CopyTo(output.AsWritableSpan());
        };
    }

    // ════════════════════════════════════════════════════════════════════
    // FusedLinear — input @ weight + bias
    // ════════════════════════════════════════════════════════════════════

    private static Action<IEngine, Tensor<T>> RebuildFusedLinear(Tensor<T>[] inputs)
    {
        var input = inputs[0]; var weight = inputs[1];
        var bias = inputs.Length > 2 ? inputs[2] : null;
        return (eng, output) =>
        {
            var r = eng.TensorMatMul(input, weight);
            if (bias is not null)
            {
                var added = eng.TensorBroadcastAdd(r, bias);
                added.AsSpan().CopyTo(output.AsWritableSpan());
            }
            else
            {
                r.AsSpan().CopyTo(output.AsWritableSpan());
            }
        };
    }
}
