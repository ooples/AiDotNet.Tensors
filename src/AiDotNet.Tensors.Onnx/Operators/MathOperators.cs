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
            // Tensor-tensor power. No broadcast overload in the engine; fall
            // back to exp(exponent * log(base)) which handles broadcasting
            // via the engine's binary ops.
            var logBase = ctx.Engine.TensorLog(baseT);
            var product = ShapesEqual(logBase._shape, expT._shape)
                ? ctx.Engine.TensorMultiply(logBase, expT)
                : ctx.Engine.TensorBroadcastMultiply(logBase, expT);
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
            var ops = MathHelper.GetNumericOperations<T>();

            Tensor<T> MakeScalar(double value, int[] shape)
            {
                var t = new Tensor<T>(shape);
                var s = t.AsWritableSpan();
                var v = ops.FromDouble(value);
                for (int i = 0; i < s.Length; i++) s[i] = v;
                return t;
            }

            // Constants from Abramowitz-Stegun 7.1.26.
            const double P = 0.3275911;
            const double A1 = 0.254829592, A2 = -0.284496736, A3 = 1.421413741;
            const double A4 = -1.453152027, A5 = 1.061405429;

            // |x|, sign(x)·erf computed via t = 1 / (1 + p*|x|).
            var absX = ctx.Engine.TensorAbs(x);
            var pAbsX = ctx.Engine.TensorBroadcastMultiply(absX, MakeScalar(P, new[] { 1 }));
            var onePlusPAbsX = ctx.Engine.TensorBroadcastAdd(pAbsX, MakeScalar(1.0, new[] { 1 }));
            var ones = MakeScalar(1.0, x._shape);
            var t = ctx.Engine.TensorDivide(ones, onePlusPAbsX);

            // Polynomial in t: poly = t · (a1 + t·(a2 + t·(a3 + t·(a4 + t·a5))))
            // Horner-style evaluation keeps the op count linear.
            var poly = ctx.Engine.TensorBroadcastAdd(
                ctx.Engine.TensorBroadcastMultiply(t, MakeScalar(A5, new[] { 1 })),
                MakeScalar(A4, new[] { 1 }));
            poly = ctx.Engine.TensorBroadcastAdd(ctx.Engine.TensorMultiply(poly, t), MakeScalar(A3, new[] { 1 }));
            poly = ctx.Engine.TensorBroadcastAdd(ctx.Engine.TensorMultiply(poly, t), MakeScalar(A2, new[] { 1 }));
            poly = ctx.Engine.TensorBroadcastAdd(ctx.Engine.TensorMultiply(poly, t), MakeScalar(A1, new[] { 1 }));
            poly = ctx.Engine.TensorMultiply(poly, t);

            // exp(-x²)
            var negXsq = ctx.Engine.TensorNegate(ctx.Engine.TensorMultiply(x, x));
            var expNegXsq = ctx.Engine.TensorExp(negXsq);

            // erf_mag = 1 - poly · exp(-x²)   (this is |erf(x)|).
            var polyExp = ctx.Engine.TensorMultiply(poly, expNegXsq);
            var erfMag = ctx.Engine.TensorSubtract(ones, polyExp);

            // Apply sign(x): sign = x / |x| (safe for x=0 since |x|>=small
            // epsilon and erf(0) = 0 so sign * 0 is still 0). To avoid
            // divide-by-zero on exact zeros, clamp |x| up by a tiny eps.
            var epsAbsX = ctx.Engine.TensorBroadcastAdd(absX, MakeScalar(1e-30, new[] { 1 }));
            var sign = ctx.Engine.TensorDivide(x, epsAbsX);
            var result = ctx.Engine.TensorMultiply(sign, erfMag);
            ctx.PutTensor(node.Output[0], result);
        }
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
            if (axes is null)
                throw new NotSupportedException("ReduceMean with no axes attribute is not yet supported.");
            var result = ctx.Engine.ReduceMean(x, axes, keepDims);
            ctx.PutTensor(node.Output[0], result);
        }
    }

    /// <summary>
    /// ONNX Min — element-wise minimum across N operands. Implements via
    /// repeated TensorMax on negated inputs (engine has no element-wise
    /// Min). For small N (most BERT graphs use 2-3) this is cheap.
    /// </summary>
    internal sealed class Min<T> : IOnnxOpTranslator<T> where T : unmanaged
    {
        public string OpType => "Min";
        public string? Domain => null;
        public void Translate(OnnxTranslationContext<T> ctx, NodeProto node)
        {
            if (node.Input.Count == 0) throw new InvalidDataException("Min requires at least one input.");
            if (node.Input.Count == 1) { ctx.PutTensor(node.Output[0], ctx.GetTensor(node.Input[0])); return; }
            // min(a, b) = -max(-a, -b)
            var acc = ctx.Engine.TensorNegate(ctx.GetTensor(node.Input[0]));
            for (int i = 1; i < node.Input.Count; i++)
            {
                var next = ctx.Engine.TensorNegate(ctx.GetTensor(node.Input[i]));
                acc = ctx.Engine.TensorMax(acc, next);
            }
            ctx.PutTensor(node.Output[0], ctx.Engine.TensorNegate(acc));
        }
    }

    /// <summary>
    /// ONNX Max — element-wise maximum across N operands.
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
                acc = ctx.Engine.TensorMax(acc, ctx.GetTensor(node.Input[i]));
            ctx.PutTensor(node.Output[0], acc);
        }
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
            // Inputs: indices, depth (scalar), values ([off, on])
            var indicesT = ctx.GetTensor(node.Input[0]);
            var depthT = ctx.GetTensor(node.Input[1]);
            var valuesT = ctx.GetTensor(node.Input[2]);
            int axis = ctx.GetIntAttrAsInt(node, "axis", -1);

            int depth = Convert.ToInt32(depthT.AsSpan()[0]!);
            var valuesSpan = valuesT.AsSpan();
            var offVal = valuesSpan[0];
            var onVal = valuesSpan[1];

            // Build the output shape: insert depth at the normalized axis
            // position. axis = -1 means last.
            int inRank = indicesT.Rank;
            int outRank = inRank + 1;
            int normalizedAxis = axis < 0 ? axis + outRank : axis;
            var outputShape = new int[outRank];
            int srcIdx = 0;
            for (int i = 0; i < outRank; i++)
            {
                if (i == normalizedAxis) outputShape[i] = depth;
                else outputShape[i] = indicesT._shape[srcIdx++];
            }

            // Materialize the one-hot tensor directly. This is a static
            // computation on the indices tensor — no engine op per element.
            var output = new Tensor<T>(outputShape);
            var outSpan = output.AsWritableSpan();
            // Fill with offVal; then scatter onVal for the active indices.
            for (int i = 0; i < outSpan.Length; i++) outSpan[i] = offVal;

            var indexSpan = indicesT.AsSpan();
            // Compute strides for the output shape.
            var strides = new int[outRank];
            strides[outRank - 1] = 1;
            for (int i = outRank - 2; i >= 0; i--) strides[i] = strides[i + 1] * outputShape[i + 1];

            // Walk indices in row-major order; for each, compute output index
            // by inserting the indexed value at `normalizedAxis`.
            for (int i = 0; i < indexSpan.Length; i++)
            {
                int indexValue = Convert.ToInt32(indexSpan[i]!);
                if (indexValue < 0) indexValue += depth;
                if (indexValue < 0 || indexValue >= depth) continue; // out-of-range → all off

                // Decompose i into multi-index over indicesT shape.
                int remaining = i;
                int outputFlat = 0;
                int srcDim = inRank - 1;
                for (int d = outRank - 1; d >= 0; d--)
                {
                    int dimSize = outputShape[d];
                    int coord;
                    if (d == normalizedAxis) coord = indexValue;
                    else
                    {
                        coord = remaining % indicesT._shape[srcDim];
                        remaining /= indicesT._shape[srcDim];
                        srcDim--;
                    }
                    outputFlat += coord * strides[d];
                }
                outSpan[outputFlat] = onVal;
            }

            ctx.PutTensor(node.Output[0], output);
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
