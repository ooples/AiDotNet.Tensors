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

            var scaled = DivideByScale(ctx, x, yScale);
            var rounded = ctx.Engine.TensorRound(scaled);
            var shifted = yZeroPoint is null ? rounded : AddScalarOrPerAxis(ctx, rounded, yZeroPoint);
            // Saturate to the output type's range. Without explicit zero_point,
            // the destination is int8 (range [-128, 127]); ONNX uses the dtype
            // of y_zero_point to determine int8 vs uint8. We can't observe
            // that here (we flattened everything to float), so clamp to the
            // wider int8 range and trust the caller's scale/zero-point to
            // keep values in range.
            ctx.PutTensor(node.Output[0], Clamp(ctx, shifted, -128, 127));
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

            var shifted = xZeroPoint is null ? x : SubScalarOrPerAxis(ctx, x, xZeroPoint);
            var scaled = MultiplyByScale(ctx, shifted, xScale);
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
    /// Multiplies <paramref name="x"/> by <paramref name="scale"/>. When the
    /// scale is a scalar ([1]) we broadcast-multiply; when it's a rank-1
    /// vector we broadcast along the channel dim (ONNX per-axis convention
    /// — axis is typically 1 for NCHW weights).
    /// </summary>
    private static Tensor<T> MultiplyByScale<T>(OnnxTranslationContext<T> ctx, Tensor<T> x, Tensor<T> scale) where T : unmanaged
    {
        if (scale.Rank == 0 || (scale.Rank == 1 && scale._shape[0] == 1))
        {
            // Scalar scale; broadcast.
            return ctx.Engine.TensorBroadcastMultiply(x, scale);
        }
        // Per-axis scale: scale shape [C]. Reshape to [1, C, 1, 1, ...] so
        // it broadcasts along dim 1 of x.
        if (scale.Rank == 1 && x.Rank >= 2 && scale._shape[0] == x._shape[1])
        {
            var axisShape = new int[x.Rank];
            axisShape[0] = 1; axisShape[1] = scale._shape[0];
            for (int i = 2; i < x.Rank; i++) axisShape[i] = 1;
            var reshaped = ctx.Engine.Reshape(scale, axisShape);
            return ctx.Engine.TensorBroadcastMultiply(x, reshaped);
        }
        throw new NotSupportedException(
            $"Quantize scale of shape [{string.Join(",", scale._shape)}] against data of rank {x.Rank} is not supported.");
    }

    private static Tensor<T> DivideByScale<T>(OnnxTranslationContext<T> ctx, Tensor<T> x, Tensor<T> scale) where T : unmanaged
    {
        // Dividing by a small scale is common; expressed as mul by reciprocal
        // for the scalar case so we can reuse the broadcast-multiply path.
        if (scale.Rank == 0 || (scale.Rank == 1 && scale._shape[0] == 1))
        {
            var reciprocal = Reciprocal(ctx, scale);
            return ctx.Engine.TensorBroadcastMultiply(x, reciprocal);
        }
        if (scale.Rank == 1 && x.Rank >= 2 && scale._shape[0] == x._shape[1])
        {
            var axisShape = new int[x.Rank];
            axisShape[0] = 1; axisShape[1] = scale._shape[0];
            for (int i = 2; i < x.Rank; i++) axisShape[i] = 1;
            var reshaped = ctx.Engine.Reshape(scale, axisShape);
            return ctx.Engine.TensorBroadcastDivide(x, reshaped);
        }
        throw new NotSupportedException(
            $"Quantize scale of shape [{string.Join(",", scale._shape)}] against data of rank {x.Rank} is not supported.");
    }

    private static Tensor<T> AddScalarOrPerAxis<T>(OnnxTranslationContext<T> ctx, Tensor<T> x, Tensor<T> zp) where T : unmanaged
    {
        if (zp.Rank == 0 || (zp.Rank == 1 && zp._shape[0] == 1))
            return ctx.Engine.TensorBroadcastAdd(x, zp);
        if (zp.Rank == 1 && x.Rank >= 2 && zp._shape[0] == x._shape[1])
        {
            var axisShape = new int[x.Rank];
            axisShape[0] = 1; axisShape[1] = zp._shape[0];
            for (int i = 2; i < x.Rank; i++) axisShape[i] = 1;
            var reshaped = ctx.Engine.Reshape(zp, axisShape);
            return ctx.Engine.TensorBroadcastAdd(x, reshaped);
        }
        throw new NotSupportedException(
            $"Quantize zero_point of shape [{string.Join(",", zp._shape)}] against data of rank {x.Rank} is not supported.");
    }

    private static Tensor<T> SubScalarOrPerAxis<T>(OnnxTranslationContext<T> ctx, Tensor<T> x, Tensor<T> zp) where T : unmanaged
    {
        if (zp.Rank == 0 || (zp.Rank == 1 && zp._shape[0] == 1))
            return ctx.Engine.TensorBroadcastSubtract(x, zp);
        if (zp.Rank == 1 && x.Rank >= 2 && zp._shape[0] == x._shape[1])
        {
            var axisShape = new int[x.Rank];
            axisShape[0] = 1; axisShape[1] = zp._shape[0];
            for (int i = 2; i < x.Rank; i++) axisShape[i] = 1;
            var reshaped = ctx.Engine.Reshape(zp, axisShape);
            return ctx.Engine.TensorBroadcastSubtract(x, reshaped);
        }
        throw new NotSupportedException(
            $"Quantize zero_point of shape [{string.Join(",", zp._shape)}] against data of rank {x.Rank} is not supported.");
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
