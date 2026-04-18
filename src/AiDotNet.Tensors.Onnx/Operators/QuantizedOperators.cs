using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tensors.Onnx.Protos;

namespace AiDotNet.Tensors.Onnx.Operators;

/// <summary>
/// ONNX quantized operator translators: QuantizeLinear, DequantizeLinear,
/// QLinearMatMul, QLinearConv. Phase 2 scope from Issue #169 — focuses on
/// making quantized models import and execute correctly even though the
/// plan-wide element type T is float. Quantized tensors are represented in
/// the plan as floats that happen to hold integer-valued quantized values
/// (clamped to [-128, 127] for int8 / [0, 255] for uint8); the arithmetic
/// goes through the engine's regular float paths.
///
/// <para>This keeps the plan homogeneous in T while still producing
/// correct outputs. True int8 kernel acceleration would need a separate
/// <c>ICompiledPlan&lt;sbyte&gt;</c> code path with int8-aware GEMM
/// routines — tracked as a follow-up.</para>
/// </summary>
internal static class QuantizedOperators
{
    internal static void Register<T>(OnnxOpTranslatorRegistry<T> r) where T : unmanaged
    {
        r.Register(new QuantizeLinear<T>());
        r.Register(new DequantizeLinear<T>());
        r.Register(new QLinearMatMul<T>());
        r.Register(new QLinearConv<T>());
        r.Register(new DynamicQuantizeLinear<T>());
    }

    /// <summary>
    /// ONNX QuantizeLinear — <c>y = saturate(round(x / y_scale) + y_zero_point)</c>
    /// with saturate clamping to the destination int8/uint8 range.
    /// Per-tensor scale (scalar) is implemented; per-axis (vector) scales
    /// are accepted only when the tensor rank matches.
    /// </summary>
    internal sealed class QuantizeLinear<T> : IOnnxOpTranslator<T> where T : unmanaged
    {
        public string OpType => "QuantizeLinear";
        public string? Domain => null;
        public void Translate(OnnxTranslationContext<T> ctx, NodeProto node)
        {
            var x = ctx.GetTensor(node.Input[0]);
            var yScale = ctx.GetTensor(node.Input[1]);
            var yZeroPoint = node.Input.Count > 2 && ctx.HasTensor(node.Input[2])
                ? ctx.GetTensor(node.Input[2]) : null;

            int qAxis = ResolveQuantAxis(ctx, node, x.Rank);
            var scaled = DivideByScale(ctx, x, yScale, qAxis);
            var rounded = ctx.Engine.TensorRound(scaled);
            var shifted = yZeroPoint is null ? rounded : AddScalarOrPerAxis(ctx, rounded, yZeroPoint, qAxis);
            // Saturate to the destination type's range. ONNX uses the dtype
            // of y_zero_point to select int8 ([-128, 127]) vs uint8 ([0, 255]).
            // We flatten everything to float at import, so we can't read the
            // ONNX TensorProto dtype directly here — instead we heuristically
            // infer uint8 when any zero_point value exceeds the int8 max or
            // every value is non-negative and exceeds int8 min. On a uint8
            // y_zero_point being [0..255] and an int8 one being [-128..127],
            // the first observation that falls outside int8 range flips us to
            // uint8. When no zero_point is provided, ONNX defaults to int8.
            bool isUint8 = yZeroPoint is not null && AnyValueExceedsInt8Max(yZeroPoint);
            int lo = isUint8 ? 0 : -128;
            int hi = isUint8 ? 255 : 127;
            ctx.PutTensor(node.Output[0], Clamp(ctx, shifted, lo, hi));
        }
    }

    /// <summary>
    /// ONNX DequantizeLinear — <c>y = (x - x_zero_point) * x_scale</c>.
    /// </summary>
    internal sealed class DequantizeLinear<T> : IOnnxOpTranslator<T> where T : unmanaged
    {
        public string OpType => "DequantizeLinear";
        public string? Domain => null;
        public void Translate(OnnxTranslationContext<T> ctx, NodeProto node)
        {
            var x = ctx.GetTensor(node.Input[0]);
            var xScale = ctx.GetTensor(node.Input[1]);
            var xZeroPoint = node.Input.Count > 2 && ctx.HasTensor(node.Input[2])
                ? ctx.GetTensor(node.Input[2]) : null;

            int qAxis = ResolveQuantAxis(ctx, node, x.Rank);
            var shifted = xZeroPoint is null ? x : SubScalarOrPerAxis(ctx, x, xZeroPoint, qAxis);
            var scaled = MultiplyByScale(ctx, shifted, xScale, qAxis);
            ctx.PutTensor(node.Output[0], scaled);
        }
    }

    /// <summary>
    /// ONNX QLinearMatMul — decomposed as dequantize-both-inputs → float
    /// matmul → requantize.
    /// </summary>
    internal sealed class QLinearMatMul<T> : IOnnxOpTranslator<T> where T : unmanaged
    {
        public string OpType => "QLinearMatMul";
        public string? Domain => null;
        public void Translate(OnnxTranslationContext<T> ctx, NodeProto node)
        {
            // Inputs: a, a_scale, a_zero_point, b, b_scale, b_zero_point,
            // y_scale, y_zero_point.
            var a = ctx.GetTensor(node.Input[0]);
            var aScale = ctx.GetTensor(node.Input[1]);
            var aZp = ctx.GetTensor(node.Input[2]);
            var b = ctx.GetTensor(node.Input[3]);
            var bScale = ctx.GetTensor(node.Input[4]);
            var bZp = ctx.GetTensor(node.Input[5]);
            var yScale = ctx.GetTensor(node.Input[6]);
            var yZp = ctx.GetTensor(node.Input[7]);

            var aF = MultiplyByScale(ctx, SubScalarOrPerAxis(ctx, a, aZp), aScale);
            var bF = MultiplyByScale(ctx, SubScalarOrPerAxis(ctx, b, bZp), bScale);
            var yF = (a.Rank <= 2 && b.Rank <= 2)
                ? ctx.Engine.TensorMatMul(aF, bF)
                : ctx.Engine.TensorBatchMatMul(aF, bF);
            var yScaled = DivideByScale(ctx, yF, yScale);
            var rounded = ctx.Engine.TensorRound(yScaled);
            var shifted = AddScalarOrPerAxis(ctx, rounded, yZp);
            ctx.PutTensor(node.Output[0], Clamp(ctx, shifted, -128, 127));
        }
    }

    /// <summary>
    /// ONNX QLinearConv — same decomposition as QLinearMatMul with the
    /// float conv handled by the regular Conv translator path.
    /// </summary>
    internal sealed class QLinearConv<T> : IOnnxOpTranslator<T> where T : unmanaged
    {
        public string OpType => "QLinearConv";
        public string? Domain => null;
        public void Translate(OnnxTranslationContext<T> ctx, NodeProto node)
        {
            // Inputs: x, x_scale, x_zero_point, w, w_scale, w_zero_point,
            // y_scale, y_zero_point, B (optional bias, int32 pre-scaled).
            var x = ctx.GetTensor(node.Input[0]);
            var xScale = ctx.GetTensor(node.Input[1]);
            var xZp = ctx.GetTensor(node.Input[2]);
            var w = ctx.GetTensor(node.Input[3]);
            var wScale = ctx.GetTensor(node.Input[4]);
            var wZp = ctx.GetTensor(node.Input[5]);
            var yScale = ctx.GetTensor(node.Input[6]);
            var yZp = ctx.GetTensor(node.Input[7]);
            var bInt = node.Input.Count > 8 && ctx.HasTensor(node.Input[8])
                ? ctx.GetTensor(node.Input[8]) : null;

            // Dequantize inputs to float.
            var xF = MultiplyByScale(ctx, SubScalarOrPerAxis(ctx, x, xZp), xScale);
            var wF = MultiplyByScale(ctx, SubScalarOrPerAxis(ctx, w, wZp), wScale);

            // Extract standard Conv attributes and run a regular float Conv.
            int group = ctx.GetIntAttrAsInt(node, "group", 1);
            var strides = ctx.GetIntArrayAttr(node, "strides") ?? new[] { 1, 1 };
            var dilations = ctx.GetIntArrayAttr(node, "dilations") ?? new[] { 1, 1 };
            var explicitPads = ctx.GetIntArrayAttr(node, "pads");
            var autoPad = ctx.GetStringAttr(node, "auto_pad", null);
            var kernelSpatial = new[] { w._shape[2], w._shape[3] };
            var pad = OnnxAutoPad.Resolve2D("QLinearConv", x._shape, kernelSpatial, strides, autoPad, explicitPads);
            var zeroT = default(T);
            var (paddedInput, symmetricPad) = OnnxAutoPad.ApplyAsymmetric(ctx.Engine, xF, pad, zeroT);

            Tensor<T> convResult;
            if (group == 1)
            {
                convResult = ctx.Engine.Conv2D(paddedInput, wF,
                    stride: strides, padding: symmetricPad, dilation: dilations);
            }
            else
            {
                throw new NotSupportedException("QLinearConv group>1 is not yet supported.");
            }

            // Bias is int32, pre-scaled by x_scale * w_scale. Convert back
            // to float by dividing by the composite scale (= multiplying by
            // 1/(xScale*wScale)). We compute float bias directly for
            // clarity — same result.
            if (bInt is not null)
            {
                var xsTimesWs = ctx.Engine.TensorMultiply(xScale, wScale);
                var bF = MultiplyByScale(ctx, bInt, xsTimesWs);
                var reshapedBias = ctx.Engine.Reshape(bF, new[] { 1, bF._shape[0], 1, 1 });
                convResult = ctx.Engine.TensorBroadcastAdd(convResult, reshapedBias);
            }

            // Requantize output: round(convResult / y_scale) + y_zero_point.
            var yScaled = DivideByScale(ctx, convResult, yScale);
            var rounded = ctx.Engine.TensorRound(yScaled);
            var shifted = AddScalarOrPerAxis(ctx, rounded, yZp);
            ctx.PutTensor(node.Output[0], Clamp(ctx, shifted, -128, 127));
        }
    }

    /// <summary>
    /// ONNX DynamicQuantizeLinear — computes per-tensor scale/zero_point
    /// from min/max then quantizes to uint8. Not common in inference models
    /// but appears in some transformer exports.
    /// </summary>
    internal sealed class DynamicQuantizeLinear<T> : IOnnxOpTranslator<T> where T : unmanaged
    {
        public string OpType => "DynamicQuantizeLinear";
        public string? Domain => null;
        public void Translate(OnnxTranslationContext<T> ctx, NodeProto node)
        {
            throw new NotSupportedException(
                "DynamicQuantizeLinear requires runtime min/max reduction + scale derivation; " +
                "it's on the Phase 3 follow-up list because it introduces data-dependent shape/value flow.");
        }
    }

    // ─── Shared arithmetic helpers ─────────────────────────────────────

    /// <summary>
    /// Reads the ONNX per-axis quantization axis attribute (default 1) and
    /// normalizes negative values against the input rank. ONNX tolerates
    /// the default axis=1 even for rank-0 / rank-1 inputs when the scale
    /// is a scalar (axis is ignored on that path) — so the out-of-range
    /// check is deferred to the per-axis broadcast helpers, which report
    /// a more actionable error citing both the offending axis and the
    /// scale shape that needed it.
    /// </summary>
    private static int ResolveQuantAxis<T>(OnnxTranslationContext<T> ctx, NodeProto node, int inputRank) where T : unmanaged
    {
        int axis = ctx.GetIntAttrAsInt(node, "axis", 1);
        if (axis < 0) axis += inputRank;
        return axis;
    }

    /// <summary>
    /// Multiplies <paramref name="x"/> by <paramref name="scale"/>. When the
    /// scale is a scalar ([1]) we broadcast-multiply; when it's a rank-1
    /// vector we broadcast along <paramref name="axis"/> (ONNX per-axis
    /// convention — defaults to 1 for NCHW weights, 0 for transposed conv
    /// weights, etc).
    /// </summary>
    private static Tensor<T> MultiplyByScale<T>(OnnxTranslationContext<T> ctx, Tensor<T> x, Tensor<T> scale, int axis = 1) where T : unmanaged
    {
        if (scale.Rank == 0 || (scale.Rank == 1 && scale._shape[0] == 1))
        {
            return ctx.Engine.TensorBroadcastMultiply(x, scale);
        }
        if (scale.Rank == 1 && x.Rank >= 1 && axis < x.Rank && scale._shape[0] == x._shape[axis])
        {
            var reshaped = ReshapeForAxisBroadcast(ctx, scale, x.Rank, axis);
            return ctx.Engine.TensorBroadcastMultiply(x, reshaped);
        }
        throw new NotSupportedException(
            $"Quantize scale of shape [{string.Join(",", scale._shape)}] against data of rank {x.Rank} along axis {axis} is not supported.");
    }

    private static Tensor<T> DivideByScale<T>(OnnxTranslationContext<T> ctx, Tensor<T> x, Tensor<T> scale, int axis = 1) where T : unmanaged
    {
        if (scale.Rank == 0 || (scale.Rank == 1 && scale._shape[0] == 1))
        {
            var reciprocal = Reciprocal(ctx, scale);
            return ctx.Engine.TensorBroadcastMultiply(x, reciprocal);
        }
        if (scale.Rank == 1 && x.Rank >= 1 && axis < x.Rank && scale._shape[0] == x._shape[axis])
        {
            var reshaped = ReshapeForAxisBroadcast(ctx, scale, x.Rank, axis);
            return ctx.Engine.TensorBroadcastDivide(x, reshaped);
        }
        throw new NotSupportedException(
            $"Quantize scale of shape [{string.Join(",", scale._shape)}] against data of rank {x.Rank} along axis {axis} is not supported.");
    }

    private static Tensor<T> AddScalarOrPerAxis<T>(OnnxTranslationContext<T> ctx, Tensor<T> x, Tensor<T> zp, int axis = 1) where T : unmanaged
    {
        if (zp.Rank == 0 || (zp.Rank == 1 && zp._shape[0] == 1))
            return ctx.Engine.TensorBroadcastAdd(x, zp);
        if (zp.Rank == 1 && x.Rank >= 1 && axis < x.Rank && zp._shape[0] == x._shape[axis])
        {
            var reshaped = ReshapeForAxisBroadcast(ctx, zp, x.Rank, axis);
            return ctx.Engine.TensorBroadcastAdd(x, reshaped);
        }
        throw new NotSupportedException(
            $"Quantize zero_point of shape [{string.Join(",", zp._shape)}] against data of rank {x.Rank} along axis {axis} is not supported.");
    }

    private static Tensor<T> SubScalarOrPerAxis<T>(OnnxTranslationContext<T> ctx, Tensor<T> x, Tensor<T> zp, int axis = 1) where T : unmanaged
    {
        if (zp.Rank == 0 || (zp.Rank == 1 && zp._shape[0] == 1))
            return ctx.Engine.TensorBroadcastSubtract(x, zp);
        if (zp.Rank == 1 && x.Rank >= 1 && axis < x.Rank && zp._shape[0] == x._shape[axis])
        {
            var reshaped = ReshapeForAxisBroadcast(ctx, zp, x.Rank, axis);
            return ctx.Engine.TensorBroadcastSubtract(x, reshaped);
        }
        throw new NotSupportedException(
            $"Quantize zero_point of shape [{string.Join(",", zp._shape)}] against data of rank {x.Rank} along axis {axis} is not supported.");
    }

    /// <summary>
    /// Reshapes a rank-1 per-axis vector of length N to a rank-<paramref name="outputRank"/>
    /// shape with N at <paramref name="axis"/> and 1 elsewhere — the layout
    /// <c>TensorBroadcast{Add,Sub,Multiply,Divide}</c> expects.
    /// </summary>
    private static Tensor<T> ReshapeForAxisBroadcast<T>(OnnxTranslationContext<T> ctx, Tensor<T> vec, int outputRank, int axis) where T : unmanaged
    {
        var axisShape = new int[outputRank];
        for (int i = 0; i < outputRank; i++) axisShape[i] = 1;
        axisShape[axis] = vec._shape[0];
        return ctx.Engine.Reshape(vec, axisShape);
    }

    private static Tensor<T> Reciprocal<T>(OnnxTranslationContext<T> ctx, Tensor<T> scalar) where T : unmanaged
    {
        // For a scalar scale, compute 1/scale once as an initializer-style
        // tensor. Avoids a TensorDivide which would require full-rank broadcast
        // on every reuse.
        var result = new Tensor<T>(scalar._shape);
        var dst = result.AsWritableSpan();
        var src = scalar.AsSpan();
        var ops = MathHelper.GetNumericOperations<T>();
        var one = ops.FromDouble(1.0);
        for (int i = 0; i < src.Length; i++) dst[i] = ops.Divide(one, src[i]);
        return result;
    }

    /// <summary>
    /// Heuristically detects uint8 (as opposed to int8) quantization by
    /// checking whether any zero-point value exceeds the int8 max (127).
    /// ONNX's actual int8/uint8 selector is the zero_point's TensorProto
    /// dtype, which we've lost by flattening everything to float T at
    /// import; the value-range heuristic is correct in practice because
    /// valid int8 zero_points are always ≤127 and valid uint8 zero_points
    /// are always ≥0 (with ≥128 routinely seen for asymmetric uint8).
    /// </summary>
    private static bool AnyValueExceedsInt8Max<T>(Tensor<T> zeroPoint) where T : unmanaged
    {
        var span = zeroPoint.AsSpan();
        for (int i = 0; i < span.Length; i++)
        {
            double v = Convert.ToDouble(span[i]!);
            if (v > 127.0 || v < -128.0) return true;
        }
        return false;
    }

    private static Tensor<T> Clamp<T>(OnnxTranslationContext<T> ctx, Tensor<T> x, double lo, double hi) where T : unmanaged
    {
        var ops = MathHelper.GetNumericOperations<T>();
        // Build same-shape scalar-broadcast lo/hi tensors. For a scalar input
        // we use shape [1]; for ranked tensors we use a rank-matching shape
        // filled with the bound so TensorMax / TensorMin pick element-wise.
        var loT = FillTensor(x._shape, ops.FromDouble(lo));
        var hiT = FillTensor(x._shape, ops.FromDouble(hi));
        var after_lo = ctx.Engine.TensorMax(x, loT);
        // Element-wise min = -max(-a, -b); engine has no TensorMin op.
        var negA = ctx.Engine.TensorNegate(after_lo);
        var negHi = ctx.Engine.TensorNegate(hiT);
        var negMin = ctx.Engine.TensorMax(negA, negHi);
        return ctx.Engine.TensorNegate(negMin);
    }

    private static Tensor<T> FillTensor<T>(int[] shape, T value) where T : unmanaged
    {
        var t = new Tensor<T>(shape);
        var s = t.AsWritableSpan();
        for (int i = 0; i < s.Length; i++) s[i] = value;
        return t;
    }
}
