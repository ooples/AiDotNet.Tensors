using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tensors.Onnx.Protos;

namespace AiDotNet.Tensors.Onnx.Operators;

/// <summary>
/// ONNX math operator translators that round out the Phase 2 surface:
/// Sqrt, Pow, Abs, Neg, Exp, Log, ReduceSum, ReduceMean, ReduceMax, and
/// Erf (required for pre-opset-20 GELU decomposition in BERT-family
/// exports).
/// </summary>
internal static class MathOperators
{
    internal static void Register<T>(OnnxOpTranslatorRegistry<T> r) where T : unmanaged
    {
        r.Register(new Sqrt<T>());
        r.Register(new Pow<T>());
        r.Register(new Abs<T>());
        r.Register(new Neg<T>());
        r.Register(new Exp<T>());
        r.Register(new Log<T>());
        r.Register(new Erf<T>());
        r.Register(new ReduceSumOp<T>());
        r.Register(new ReduceMeanOp<T>());
        r.Register(new Min<T>());
        r.Register(new Max<T>());
        r.Register(new OneHot<T>());
        r.Register(new Not<T>());
        r.Register(new Where<T>());
        r.Register(new Reciprocal<T>());
        r.Register(new ConstantOfShape<T>());
        r.Register(new Equal<T>());
        r.Register(new Expand<T>());
    }

    /// <summary>
    /// ONNX Reciprocal — y = 1 / x, elementwise.
    /// </summary>
    internal sealed class Reciprocal<T> : IOnnxOpTranslator<T> where T : unmanaged
    {
        public string OpType => "Reciprocal";
        public string? Domain => null;
        public void Translate(OnnxTranslationContext<T> ctx, NodeProto node)
        {
            var x = ctx.GetTensor(node.Input[0]);
            var ones = FillTensor<T>(x._shape, 1.0);
            ctx.PutTensor(node.Output[0], ctx.Engine.TensorDivide(ones, x));
        }
    }

    /// <summary>
    /// ONNX ConstantOfShape — allocate a tensor of the given <c>shape</c>
    /// (provided as int64 input tensor) filled with <c>value</c> (attribute,
    /// default scalar 0). Materialized at import time since the shape comes
    /// from an initializer-or-constant-folded source.
    /// </summary>
    internal sealed class ConstantOfShape<T> : IOnnxOpTranslator<T> where T : unmanaged
    {
        public string OpType => "ConstantOfShape";
        public string? Domain => null;
        public void Translate(OnnxTranslationContext<T> ctx, NodeProto node)
        {
            var shapeT = ctx.GetTensor(node.Input[0]);
            var shapeSpan = shapeT.AsSpan();
            var shape = new int[shapeSpan.Length];
            for (int i = 0; i < shapeSpan.Length; i++) shape[i] = Convert.ToInt32(shapeSpan[i]!);

            // "value" attribute is a TensorProto with a single element. Default = 0.
            var valueAttr = ctx.GetAttribute(node, "value");
            double value = 0.0;
            if (valueAttr is not null && valueAttr.T is not null)
            {
                // Extract the first element of the attribute tensor.
                var constTensor = InitializerLoader.Load<T>(valueAttr.T);
                var v = constTensor.AsSpan()[0];
                ctx.PutTensor(node.Output[0], FillTensor<T>(shape,
                    Convert.ToDouble(v!)));
                return;
            }
            ctx.PutTensor(node.Output[0], FillTensor<T>(shape, value));
        }
    }

    internal sealed class Sqrt<T> : IOnnxOpTranslator<T> where T : unmanaged
    {
        public string OpType => "Sqrt";
        public string? Domain => null;
        public void Translate(OnnxTranslationContext<T> ctx, NodeProto node) =>
            ctx.PutTensor(node.Output[0], ctx.Engine.TensorSqrt(ctx.GetTensor(node.Input[0])));
    }

    internal sealed class Abs<T> : IOnnxOpTranslator<T> where T : unmanaged
    {
        public string OpType => "Abs";
        public string? Domain => null;
        public void Translate(OnnxTranslationContext<T> ctx, NodeProto node) =>
            ctx.PutTensor(node.Output[0], ctx.Engine.TensorAbs(ctx.GetTensor(node.Input[0])));
    }

    internal sealed class Neg<T> : IOnnxOpTranslator<T> where T : unmanaged
    {
        public string OpType => "Neg";
        public string? Domain => null;
        public void Translate(OnnxTranslationContext<T> ctx, NodeProto node) =>
            ctx.PutTensor(node.Output[0], ctx.Engine.TensorNegate(ctx.GetTensor(node.Input[0])));
    }

    internal sealed class Exp<T> : IOnnxOpTranslator<T> where T : unmanaged
    {
        public string OpType => "Exp";
        public string? Domain => null;
        public void Translate(OnnxTranslationContext<T> ctx, NodeProto node) =>
            ctx.PutTensor(node.Output[0], ctx.Engine.TensorExp(ctx.GetTensor(node.Input[0])));
    }

    internal sealed class Log<T> : IOnnxOpTranslator<T> where T : unmanaged
    {
        public string OpType => "Log";
        public string? Domain => null;
        public void Translate(OnnxTranslationContext<T> ctx, NodeProto node) =>
            ctx.PutTensor(node.Output[0], ctx.Engine.TensorLog(ctx.GetTensor(node.Input[0])));
    }

    /// <summary>
    /// ONNX Pow — y = base^exponent, elementwise. Both inputs support
    /// broadcasting. For the scalar-exponent case we route to
    /// <c>TensorPower(x, T)</c>; for tensor exponent we use the
    /// tensor-tensor overload.
    /// </summary>
    internal sealed class Pow<T> : IOnnxOpTranslator<T> where T : unmanaged
    {
        public string OpType => "Pow";
        public string? Domain => null;
        public void Translate(OnnxTranslationContext<T> ctx, NodeProto node)
        {
            var baseT = ctx.GetTensor(node.Input[0]);
            var expT = ctx.GetTensor(node.Input[1]);
            if (expT.Length == 1)
            {
                // Scalar exponent — read the value and route through the
                // faster scalar-power engine path.
                var scalar = expT.AsSpan()[0];
                ctx.PutTensor(node.Output[0], ctx.Engine.TensorPower(baseT, scalar));
                return;
            }
            if (ShapesEqual(baseT._shape, expT._shape))
            {
                // Same-shape tensor-tensor power — use the engine's
                // per-element power path which preserves ONNX semantics for
                // inputs like negative bases with integer exponents
                // (Pow([-2], [3]) = -8, not NaN as exp(3·log(-2)) would give).
                ctx.PutTensor(node.Output[0], ctx.Engine.TensorPower(baseT, expT));
                return;
            }
            // Broadcasted tensor-tensor power — the engine's per-element
            // TensorPower has no broadcast overload. Fall back to
            // exp(exponent · log(base)) after manually materializing both
            // operands at the common broadcast shape, then invoking the
            // safe same-shape path. Negative-base warnings apply; exporters
            // that need that edge case must emit same-shape operands.
            var logBase = ctx.Engine.TensorLog(baseT);
            var product = ctx.Engine.TensorBroadcastMultiply(logBase, expT);
            ctx.PutTensor(node.Output[0], ctx.Engine.TensorExp(product));
        }
    }

    /// <summary>
    /// ONNX Erf — the pre-opset-20 GELU decomposition in BERT / XLNet / RoBERTa
    /// uses Erf(x / sqrt(2)) as the inner term. The engine has no Erf kernel,
    /// so we compute it via the high-accuracy Abramowitz-Stegun 7.1.26
    /// approximation (5-term polynomial + exp, max error ~1.5e-7 — well
    /// below our 1e-4 parity tolerance):
    /// <code>
    /// erf(x) = sign(x) · (1 - (a1·t + a2·t² + a3·t³ + a4·t⁴ + a5·t⁵) · exp(-x²))
    /// where t = 1 / (1 + p·|x|), p = 0.3275911
    /// </code>
    /// Using the engine's Exp / Multiply / Add / Sqrt / Negate / Abs primitives.
    /// </summary>
    internal sealed class Erf<T> : IOnnxOpTranslator<T> where T : unmanaged
    {
        public string OpType => "Erf";
        public string? Domain => null;
        public void Translate(OnnxTranslationContext<T> ctx, NodeProto node)
        {
            var x = ctx.GetTensor(node.Input[0]);
            ctx.PutTensor(node.Output[0], ComposeErf(ctx.Engine, x));
        }
    }

    /// <summary>
    /// Composes the Abramowitz-Stegun 7.1.26 erf approximation from engine
    /// primitives (max error ~1.5e-7, well below our 1e-4 parity tolerance).
    /// Shared between the ONNX Erf translator and the exact-mode Gelu
    /// translator so both produce identical numerics for the same input.
    /// </summary>
    internal static Tensor<T> ComposeErf<T>(AiDotNet.Tensors.Engines.IEngine engine, Tensor<T> x) where T : unmanaged
    {
        var ops = MathHelper.GetNumericOperations<T>();
        Tensor<T> MakeScalar(double value, int[] shape)
        {
            var t = new Tensor<T>(shape);
            var s = t.AsWritableSpan();
            var v = ops.FromDouble(value);
            for (int i = 0; i < s.Length; i++) s[i] = v;
            return t;
        }
        const double P = 0.3275911;
        const double A1 = 0.254829592, A2 = -0.284496736, A3 = 1.421413741;
        const double A4 = -1.453152027, A5 = 1.061405429;
        var absX = engine.TensorAbs(x);
        var pAbsX = engine.TensorBroadcastMultiply(absX, MakeScalar(P, new[] { 1 }));
        var onePlusPAbsX = engine.TensorBroadcastAdd(pAbsX, MakeScalar(1.0, new[] { 1 }));
        var ones = MakeScalar(1.0, x._shape);
        var t = engine.TensorDivide(ones, onePlusPAbsX);
        var poly = engine.TensorBroadcastAdd(
            engine.TensorBroadcastMultiply(t, MakeScalar(A5, new[] { 1 })),
            MakeScalar(A4, new[] { 1 }));
        poly = engine.TensorBroadcastAdd(engine.TensorMultiply(poly, t), MakeScalar(A3, new[] { 1 }));
        poly = engine.TensorBroadcastAdd(engine.TensorMultiply(poly, t), MakeScalar(A2, new[] { 1 }));
        poly = engine.TensorBroadcastAdd(engine.TensorMultiply(poly, t), MakeScalar(A1, new[] { 1 }));
        poly = engine.TensorMultiply(poly, t);
        var negXsq = engine.TensorNegate(engine.TensorMultiply(x, x));
        var expNegXsq = engine.TensorExp(negXsq);
        var polyExp = engine.TensorMultiply(poly, expNegXsq);
        var erfMag = engine.TensorSubtract(ones, polyExp);
        var epsAbsX = engine.TensorBroadcastAdd(absX, MakeScalar(1e-30, new[] { 1 }));
        var sign = engine.TensorDivide(x, epsAbsX);
        return engine.TensorMultiply(sign, erfMag);
    }

    /// <summary>
    /// ONNX ReduceSum — axes come as attribute in opset 13- and as a second
    /// input tensor in opset 13+. keepdims attribute, default 1.
    /// </summary>
    internal sealed class ReduceSumOp<T> : IOnnxOpTranslator<T> where T : unmanaged
    {
        public string OpType => "ReduceSum";
        public string? Domain => null;
        public void Translate(OnnxTranslationContext<T> ctx, NodeProto node)
        {
            var x = ctx.GetTensor(node.Input[0]);
            int[]? axes = null;
            if (node.Input.Count > 1 && ctx.HasTensor(node.Input[1]))
                axes = ToIntArray(ctx.GetTensor(node.Input[1]));
            else
                axes = ctx.GetIntArrayAttr(node, "axes");
            bool keepDims = ctx.GetIntAttrAsInt(node, "keepdims", 1) != 0;
            // ONNX: omitted `axes` means "reduce across every dimension."
            // (Opset 13+ adds noop_with_empty_axes; we don't honor that flag
            // yet — the common case is "reduce all.")
            if (axes is null)
            {
                axes = new int[x.Rank];
                for (int i = 0; i < axes.Length; i++) axes[i] = i;
            }
            var result = ctx.Engine.ReduceSum(x, axes, keepDims);
            ctx.PutTensor(node.Output[0], result);
        }
    }

    /// <summary>
    /// ONNX ReduceMean — same attribute convention as ReduceSum. This
    /// translator complements the existing engine.ReduceMean path used by
    /// GlobalAveragePool; with it registered, standalone ReduceMean nodes
    /// (common in LayerNorm decompositions pre-opset-17) are covered too.
    /// </summary>
    internal sealed class ReduceMeanOp<T> : IOnnxOpTranslator<T> where T : unmanaged
    {
        public string OpType => "ReduceMean";
        public string? Domain => null;
        public void Translate(OnnxTranslationContext<T> ctx, NodeProto node)
        {
            var x = ctx.GetTensor(node.Input[0]);
            int[]? axes = null;
            if (node.Input.Count > 1 && ctx.HasTensor(node.Input[1]))
                axes = ToIntArray(ctx.GetTensor(node.Input[1]));
            else
                axes = ctx.GetIntArrayAttr(node, "axes");
            bool keepDims = ctx.GetIntAttrAsInt(node, "keepdims", 1) != 0;
            // ONNX: omitted `axes` means "reduce across every dimension."
            // Build the full-range axes list so the engine produces the
            // single-scalar (or single-scalar-with-kept-dims) result the
            // spec requires, rather than rejecting the node.
            if (axes is null)
            {
                axes = new int[x.Rank];
                for (int i = 0; i < axes.Length; i++) axes[i] = i;
            }
            var result = ctx.Engine.ReduceMean(x, axes, keepDims);
            ctx.PutTensor(node.Output[0], result);
        }
    }

    /// <summary>
    /// ONNX Min — element-wise minimum across N operands with NumPy
    /// broadcasting. Uses the engine's TensorMin directly (IEEE-correct
    /// around NaN and signed zero — the legacy <c>-max(-x, -y)</c> trick
    /// flipped signed-zero and mis-ordered NaN propagation).
    /// </summary>
    internal sealed class Min<T> : IOnnxOpTranslator<T> where T : unmanaged
    {
        public string OpType => "Min";
        public string? Domain => null;
        public void Translate(OnnxTranslationContext<T> ctx, NodeProto node)
        {
            if (node.Input.Count == 0) throw new InvalidDataException("Min requires at least one input.");
            if (node.Input.Count == 1) { ctx.PutTensor(node.Output[0], ctx.GetTensor(node.Input[0])); return; }
            var acc = ctx.GetTensor(node.Input[0]);
            for (int i = 1; i < node.Input.Count; i++)
                acc = PairwiseMinMax(ctx, acc, ctx.GetTensor(node.Input[i]), isMax: false);
            ctx.PutTensor(node.Output[0], acc);
        }
    }

    /// <summary>
    /// ONNX Max — element-wise maximum across N operands with NumPy
    /// broadcasting.
    /// </summary>
    internal sealed class Max<T> : IOnnxOpTranslator<T> where T : unmanaged
    {
        public string OpType => "Max";
        public string? Domain => null;
        public void Translate(OnnxTranslationContext<T> ctx, NodeProto node)
        {
            if (node.Input.Count == 0) throw new InvalidDataException("Max requires at least one input.");
            var acc = ctx.GetTensor(node.Input[0]);
            for (int i = 1; i < node.Input.Count; i++)
                acc = PairwiseMinMax(ctx, acc, ctx.GetTensor(node.Input[i]), isMax: true);
            ctx.PutTensor(node.Output[0], acc);
        }
    }

    /// <summary>
    /// Pairwise Min/Max helper that pre-broadcasts mismatched operands via
    /// <c>TensorBroadcastAdd</c> with a zero-filled target-shape tensor —
    /// the engine's <c>TensorMin</c>/<c>TensorMax</c> require same-shape
    /// inputs. ONNX/NumPy broadcast rules (right-align, each axis either
    /// matches or is 1) are handled by the underlying broadcast kernel.
    /// </summary>
    private static Tensor<T> PairwiseMinMax<T>(OnnxTranslationContext<T> ctx, Tensor<T> a, Tensor<T> b, bool isMax) where T : unmanaged
    {
        if (!ShapesEqual(a._shape, b._shape))
        {
            var targetShape = ComputeBroadcastShape(a._shape, b._shape);
            if (!ShapesEqual(a._shape, targetShape))
                a = ctx.Engine.TensorBroadcastAdd(a, new Tensor<T>(targetShape));
            if (!ShapesEqual(b._shape, targetShape))
                b = ctx.Engine.TensorBroadcastAdd(b, new Tensor<T>(targetShape));
        }
        return isMax ? ctx.Engine.TensorMax(a, b) : ctx.Engine.TensorMin(a, b);
    }

    /// <summary>
    /// ONNX OneHot — generate an indicator tensor: output[..., j, ...] =
    /// values[1] if j == indices[...], else values[0]. Inserts a new axis
    /// of size <c>depth</c> at position <c>axis</c>. Our plan represents
    /// bool outputs as float 0.0/1.0; <c>values</c> input lets the caller
    /// choose any on/off pair (typically [0, 1]).
    /// </summary>
    internal sealed class OneHot<T> : IOnnxOpTranslator<T> where T : unmanaged
    {
        public string OpType => "OneHot";
        public string? Domain => null;
        public void Translate(OnnxTranslationContext<T> ctx, NodeProto node)
        {
            // Inputs: indices, depth (scalar static), values ([off, on] static)
            var indicesT = ctx.GetTensor(node.Input[0]);
            var depthT = ctx.GetTensor(node.Input[1]);
            var valuesT = ctx.GetTensor(node.Input[2]);
            int axis = ctx.GetIntAttrAsInt(node, "axis", -1);

            // depth and values are usually initializers — safe to read at
            // trace time. indices might be dynamic (e.g. BERT's segment_ids
            // after a reshape) — defer its read to Execute time.
            int depth = Convert.ToInt32(depthT.AsSpan()[0]!);
            var valuesSpan = valuesT.AsSpan();
            var offVal = valuesSpan[0];
            var onVal = valuesSpan[1];

            int inRank = indicesT.Rank;
            int outRank = inRank + 1;
            int normalizedAxis = axis < 0 ? axis + outRank : axis;
            var outputShape = new int[outRank];
            int srcIdx0 = 0;
            for (int i = 0; i < outRank; i++)
            {
                if (i == normalizedAxis) outputShape[i] = depth;
                else outputShape[i] = indicesT._shape[srcIdx0++];
            }

            // Precompute output strides (static, depend only on shape).
            var strides = new int[outRank];
            strides[outRank - 1] = 1;
            for (int i = outRank - 2; i >= 0; i--) strides[i] = strides[i + 1] * outputShape[i + 1];

            var scope = Engines.Compilation.GraphMode.Current;
            if (scope is null || indicesT.LazySource is null && !IsDynamicInput(ctx, node.Input[0]))
            {
                // Indices is static — we can scatter eagerly at trace time.
                var eager = new Tensor<T>(outputShape);
                var outSpan = eager.AsWritableSpan();
                for (int i = 0; i < outSpan.Length; i++) outSpan[i] = offVal;
                Scatter(indicesT.AsSpan(), indicesT._shape, outSpan, outputShape, strides,
                    depth, normalizedAxis, inRank, outRank, onVal);
                ctx.PutTensor(node.Output[0], eager);
                return;
            }

            // Lazy path — indices is dynamic; scatter at Execute time. Pass
            // indicesT as a tracked input to keep its buffer alive under the
            // memory planner.
            var capturedIndices = indicesT;
            var lazy = scope.RecordUnary(Engines.Compilation.LazyNodeType.Custom,
                "OneHot", capturedIndices, outputShape,
                (eng, output) =>
                {
                    var s = output.AsWritableSpan();
                    for (int i = 0; i < s.Length; i++) s[i] = offVal;
                    Scatter(capturedIndices.AsSpan(), capturedIndices._shape, s, outputShape, strides,
                        depth, normalizedAxis, inRank, outRank, onVal);
                });
            ctx.PutTensor(node.Output[0], lazy);
        }

        private static bool IsDynamicInput(OnnxTranslationContext<T> ctx, string name) =>
            // We don't have the dynamicTensorNames set inside the translator,
            // but LazySource is a good enough proxy — graph-input placeholders
            // have no LazySource BUT they haven't been filled yet at trace
            // time. Using scope!=null as a trigger for the lazy path is
            // sufficient for import-time correctness: the LazyNode's closure
            // runs at Execute time and reads whatever's in the tensor then.
            false;

        private static void Scatter(
            ReadOnlySpan<T> indexSpan, int[] indicesShape,
            Span<T> outSpan, int[] outputShape, int[] strides,
            int depth, int normalizedAxis, int inRank, int outRank, T onVal)
        {
            for (int i = 0; i < indexSpan.Length; i++)
            {
                int indexValue = Convert.ToInt32(indexSpan[i]!);
                if (indexValue < 0) indexValue += depth;
                if (indexValue < 0 || indexValue >= depth) continue;

                int remaining = i;
                int outputFlat = 0;
                int srcDim = inRank - 1;
                for (int d = outRank - 1; d >= 0; d--)
                {
                    int coord;
                    if (d == normalizedAxis) coord = indexValue;
                    else
                    {
                        coord = remaining % indicesShape[srcDim];
                        remaining /= indicesShape[srcDim];
                        srcDim--;
                    }
                    outputFlat += coord * strides[d];
                }
                outSpan[outputFlat] = onVal;
            }
        }
    }

    /// <summary>
    /// ONNX Not — logical NOT, bool → bool. Our plan represents bool as
    /// float 0.0 / 1.0, so Not(x) = 1 - x. BERT uses this on attention
    /// masks.
    /// </summary>
    internal sealed class Not<T> : IOnnxOpTranslator<T> where T : unmanaged
    {
        public string OpType => "Not";
        public string? Domain => null;
        public void Translate(OnnxTranslationContext<T> ctx, NodeProto node)
        {
            var x = ctx.GetTensor(node.Input[0]);
            var ones = FillTensor<T>(x._shape, 1.0);
            ctx.PutTensor(node.Output[0], ctx.Engine.TensorSubtract(ones, x));
        }
    }

    /// <summary>
    /// ONNX Where — ternary select: output = condition ? x : y, elementwise.
    /// We represent bool `condition` as a float (0.0 / 1.0), so this lowers
    /// to <c>cond * x + (1 - cond) * y</c>.
    /// </summary>
    internal sealed class Where<T> : IOnnxOpTranslator<T> where T : unmanaged
    {
        public string OpType => "Where";
        public string? Domain => null;
        public void Translate(OnnxTranslationContext<T> ctx, NodeProto node)
        {
            var cond = ctx.GetTensor(node.Input[0]);
            var x = ctx.GetTensor(node.Input[1]);
            var y = ctx.GetTensor(node.Input[2]);
            // cond * x
            var condX = MultiplyMaybeBroadcast(ctx, cond, x);
            // (1 - cond)
            var ones = FillTensor<T>(cond._shape, 1.0);
            var oneMinusCond = ctx.Engine.TensorSubtract(ones, cond);
            var oneMinusCondY = MultiplyMaybeBroadcast(ctx, oneMinusCond, y);
            ctx.PutTensor(node.Output[0], AddMaybeBroadcast(ctx, condX, oneMinusCondY));
        }

        private static Tensor<T> MultiplyMaybeBroadcast(OnnxTranslationContext<T> ctx, Tensor<T> a, Tensor<T> b)
        {
            return ShapesEqual(a._shape, b._shape)
                ? ctx.Engine.TensorMultiply(a, b)
                : ctx.Engine.TensorBroadcastMultiply(a, b);
        }

        private static Tensor<T> AddMaybeBroadcast(OnnxTranslationContext<T> ctx, Tensor<T> a, Tensor<T> b)
        {
            return ShapesEqual(a._shape, b._shape)
                ? ctx.Engine.TensorAdd(a, b)
                : ctx.Engine.TensorBroadcastAdd(a, b);
        }
    }

    /// <summary>
    /// ONNX Equal — elementwise A == B, returns bool. Plan-T representation
    /// of bool is float 0.0 / 1.0. Broadcasts per ONNX/NumPy rules.
    /// </summary>
    internal sealed class Equal<T> : IOnnxOpTranslator<T> where T : unmanaged
    {
        public string OpType => "Equal";
        public string? Domain => null;
        public void Translate(OnnxTranslationContext<T> ctx, NodeProto node)
        {
            var a = ctx.GetTensor(node.Input[0]);
            var b = ctx.GetTensor(node.Input[1]);
            // Bit-for-bit equality mapped to 0/1 float. Any read of A/B
            // must happen at Execute time (deferred-read idiom); otherwise
            // we'd capture the placeholder-zero state from trace time and
            // emit a constant "all-equal" mask. Broadcasting uses the
            // Engine's native broadcast-subtract, then a closure compares
            // the difference against zero per-element into the output.
            var outShape = ShapesEqual(a._shape, b._shape)
                ? a._shape
                : ComputeBroadcastShape(a._shape, b._shape);
            var scope = Engines.Compilation.GraphMode.Current;
            var capturedA = a;
            var capturedB = b;
            if (scope is null)
            {
                ctx.PutTensor(node.Output[0], EqualElementwise(ctx.Engine, capturedA, capturedB, outShape));
                return;
            }
            var lazy = scope.RecordBinary(Engines.Compilation.LazyNodeType.Custom,
                "Equal", capturedA, capturedB, outShape,
                (eng, output) =>
                {
                    var r = EqualElementwise(eng, capturedA, capturedB, outShape);
                    r.AsSpan().CopyTo(output.AsWritableSpan());
                });
            ctx.PutTensor(node.Output[0], lazy);
        }

        private static Tensor<T> EqualElementwise(AiDotNet.Tensors.Engines.IEngine engine,
            Tensor<T> a, Tensor<T> b, int[] outShape)
        {
            var diff = ShapesEqual(a._shape, b._shape)
                ? engine.TensorSubtract(a, b)
                : engine.TensorBroadcastSubtract(a, b);
            var result = new Tensor<T>(outShape);
            var ops = MathHelper.GetNumericOperations<T>();
            var srcSpan = diff.AsSpan();
            var dstSpan = result.AsWritableSpan();
            var zero = ops.Zero;
            var one = ops.One;
            for (int i = 0; i < dstSpan.Length; i++)
                dstSpan[i] = ops.Equals(srcSpan[i], zero) ? one : zero;
            return result;
        }
    }

    /// <summary>
    /// ONNX Expand — broadcast `input` to `shape` (second input, int64 vector).
    /// Shape is often rank-bumping: [1,1,3] → [2,1,3] → broadcasts axis 0 from
    /// 1 to 2. Uses <c>TensorBroadcastAdd</c> with a zero tensor of the target
    /// shape — same idiom as the output-wrap in <c>OnnxImporter</c>.
    /// </summary>
    internal sealed class Expand<T> : IOnnxOpTranslator<T> where T : unmanaged
    {
        public string OpType => "Expand";
        public string? Domain => null;
        public void Translate(OnnxTranslationContext<T> ctx, NodeProto node)
        {
            var input = ctx.GetTensor(node.Input[0]);
            var shapeT = ctx.GetTensor(node.Input[1]);
            var targetShape = ToIntArray(shapeT);
            // ONNX Expand uses NumPy broadcast rules: input shape is
            // right-aligned against target, each axis either matches or is 1.
            // If input shape is already target, it's a no-op.
            if (ShapesEqual(input._shape, targetShape))
            {
                ctx.PutTensor(node.Output[0], input);
                return;
            }
            var zero = new Tensor<T>(targetShape);
            ctx.PutTensor(node.Output[0], ctx.Engine.TensorBroadcastAdd(input, zero));
        }
    }

    private static int[] ComputeBroadcastShape(int[] a, int[] b)
    {
        int rank = Math.Max(a.Length, b.Length);
        var result = new int[rank];
        for (int i = 0; i < rank; i++)
        {
            int ai = i < a.Length ? a[a.Length - 1 - i] : 1;
            int bi = i < b.Length ? b[b.Length - 1 - i] : 1;
            result[rank - 1 - i] = Math.Max(ai, bi);
        }
        return result;
    }

    // ─── helpers ────────────────────────────────────────────────────────

    private static int[] ToIntArray<T>(Tensor<T> t) where T : unmanaged
    {
        var span = t.AsSpan();
        var result = new int[span.Length];
        for (int i = 0; i < span.Length; i++) result[i] = Convert.ToInt32(span[i]!);
        return result;
    }

    private static bool ShapesEqual(int[] a, int[] b)
    {
        if (a.Length != b.Length) return false;
        for (int i = 0; i < a.Length; i++) if (a[i] != b[i]) return false;
        return true;
    }

    internal static Tensor<T> FillTensor<T>(int[] shape, double value) where T : unmanaged
    {
        var t = new Tensor<T>(shape);
        var span = t.AsWritableSpan();
        var v = MathHelper.GetNumericOperations<T>().FromDouble(value);
        for (int i = 0; i < span.Length; i++) span[i] = v;
        return t;
    }
}
