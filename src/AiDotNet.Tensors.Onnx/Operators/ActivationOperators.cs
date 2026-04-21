using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.Onnx.Protos;

namespace AiDotNet.Tensors.Onnx.Operators;

/// <summary>
/// ONNX activation operator translators: Relu, Sigmoid, Tanh, Softmax, GELU,
/// plus Erf and LeakyRelu which appear in ViT and BERT preprocessing.
/// </summary>
internal static class ActivationOperators
{
    internal static void Register<T>(OnnxOpTranslatorRegistry<T> r) where T : unmanaged
    {
        r.Register(new Relu<T>());
        r.Register(new Sigmoid<T>());
        r.Register(new Tanh<T>());
        r.Register(new Softmax<T>());
        r.Register(new Gelu<T>());
        r.Register(new LeakyRelu<T>());
        // Real Erf implementation is registered by MathOperators — BERT's
        // pre-opset-20 GELU decomposition uses it, so the stub removed.
    }

    internal sealed class Relu<T> : IOnnxOpTranslator<T> where T : unmanaged
    {
        public string OpType => "Relu";
        public string? Domain => null;
        public void Translate(OnnxTranslationContext<T> ctx, NodeProto node) =>
            ctx.PutTensor(node.Output[0], ctx.Engine.ReLU(ctx.GetTensor(node.Input[0])));
    }

    internal sealed class Sigmoid<T> : IOnnxOpTranslator<T> where T : unmanaged
    {
        public string OpType => "Sigmoid";
        public string? Domain => null;
        public void Translate(OnnxTranslationContext<T> ctx, NodeProto node) =>
            ctx.PutTensor(node.Output[0], ctx.Engine.Sigmoid(ctx.GetTensor(node.Input[0])));
    }

    internal sealed class Tanh<T> : IOnnxOpTranslator<T> where T : unmanaged
    {
        public string OpType => "Tanh";
        public string? Domain => null;
        public void Translate(OnnxTranslationContext<T> ctx, NodeProto node) =>
            ctx.PutTensor(node.Output[0], ctx.Engine.Tanh(ctx.GetTensor(node.Input[0])));
    }

    /// <summary>
    /// ONNX Softmax — default axis is 1 for opsets 1–12 (2D-coercion
    /// semantics) and -1 for opset 13+ (direct-axis semantics). We read
    /// the model's declared default-domain opset from the context so
    /// pre-opset-13 models fed straight into the importer (no ORT
    /// normalization) keep their original axis behaviour.
    /// </summary>
    internal sealed class Softmax<T> : IOnnxOpTranslator<T> where T : unmanaged
    {
        public string OpType => "Softmax";
        public string? Domain => null;
        public void Translate(OnnxTranslationContext<T> ctx, NodeProto node)
        {
            int defaultAxis = ctx.DefaultOpset >= 13 ? -1 : 1;
            int axis = ctx.GetIntAttrAsInt(node, "axis", defaultAxis);
            ctx.PutTensor(node.Output[0], ctx.Engine.Softmax(ctx.GetTensor(node.Input[0]), axis));
        }
    }

    /// <summary>
    /// ONNX Gelu (opset 20+). The <c>approximate</c> attribute selects the
    /// formula:
    /// <list type="bullet">
    /// <item><c>"none"</c> (ONNX default): exact erf form
    /// <c>0.5·x·(1 + erf(x/√2))</c>.</item>
    /// <item><c>"tanh"</c>: Hendrycks &amp; Gimpel tanh approximation
    /// <c>0.5·x·(1 + tanh(√(2/π)·(x + 0.044715·x³)))</c> — what
    /// <c>IEngine.GELU</c> implements.</item>
    /// </list>
    /// We honor the attribute: "tanh" lowers to the engine's tanh-form
    /// GELU; "none" is composed from our Erf translator's A-S approximation
    /// (<c>0.5·x·(1 + Erf(x/√2))</c>) so pre-opset-20 BERT-style exports
    /// that target erf semantics also land on the correct numerics.
    /// </summary>
    internal sealed class Gelu<T> : IOnnxOpTranslator<T> where T : unmanaged
    {
        public string OpType => "Gelu";
        public string? Domain => null;
        public void Translate(OnnxTranslationContext<T> ctx, NodeProto node)
        {
            var x = ctx.GetTensor(node.Input[0]);
            string mode = ctx.GetStringAttr(node, "approximate", "none") ?? "none";
            if (mode == "tanh")
            {
                ctx.PutTensor(node.Output[0], ctx.Engine.GELU(x));
                return;
            }
            if (mode != "none")
                throw new NotSupportedException(
                    $"ONNX Gelu approximate='{mode}' is not recognized. Expected 'none' or 'tanh'.");
            // Exact erf form: 0.5 * x * (1 + erf(x / sqrt(2))).
            double invSqrt2 = 1.0 / Math.Sqrt(2.0);
            var scale = MathOperators.FillTensor<T>(x._shape, invSqrt2);
            var scaled = ctx.Engine.TensorMultiply(x, scale);
            var erf = MathOperators.ComposeErf(ctx.Engine, scaled);
            var ones = MathOperators.FillTensor<T>(x._shape, 1.0);
            var onePlusErf = ctx.Engine.TensorAdd(ones, erf);
            var xTimesOnePlusErf = ctx.Engine.TensorMultiply(x, onePlusErf);
            var half = MathOperators.FillTensor<T>(x._shape, 0.5);
            ctx.PutTensor(node.Output[0], ctx.Engine.TensorMultiply(half, xTimesOnePlusErf));
        }
    }

    internal sealed class LeakyRelu<T> : IOnnxOpTranslator<T> where T : unmanaged
    {
        public string OpType => "LeakyRelu";
        public string? Domain => null;
        public void Translate(OnnxTranslationContext<T> ctx, NodeProto node)
        {
            float alpha = ctx.GetFloatAttr(node, "alpha", 0.01f);
            var a = MathHelper.GetNumericOperations<T>().FromDouble(alpha);
            ctx.PutTensor(node.Output[0], ctx.Engine.LeakyReLU(ctx.GetTensor(node.Input[0]), a));
        }
    }

}
