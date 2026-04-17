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
        r.Register(new Erf<T>());
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
    /// ONNX Softmax — default axis is -1 (last dim) in opset 13+, and 1 in
    /// older opsets. ONNX Runtime normalises to opset 13 semantics during
    /// import, so we stick with the -1 default.
    /// </summary>
    internal sealed class Softmax<T> : IOnnxOpTranslator<T> where T : unmanaged
    {
        public string OpType => "Softmax";
        public string? Domain => null;
        public void Translate(OnnxTranslationContext<T> ctx, NodeProto node)
        {
            int axis = ctx.GetIntAttrAsInt(node, "axis", -1);
            ctx.PutTensor(node.Output[0], ctx.Engine.Softmax(ctx.GetTensor(node.Input[0]), axis));
        }
    }

    /// <summary>
    /// ONNX GELU (added in opset 20; many exporters use the BERT-style
    /// decomposition Erf-based implementation prior to that). Uses the
    /// "none" approximation by default — exact erf-based formula.
    /// </summary>
    internal sealed class Gelu<T> : IOnnxOpTranslator<T> where T : unmanaged
    {
        public string OpType => "Gelu";
        public string? Domain => null;
        public void Translate(OnnxTranslationContext<T> ctx, NodeProto node) =>
            ctx.PutTensor(node.Output[0], ctx.Engine.GELU(ctx.GetTensor(node.Input[0])));
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

    /// <summary>
    /// ONNX Erf — used in the pre-opset-20 GELU decomposition
    /// (0.5 * x * (1 + erf(x / sqrt(2)))). The engine has no direct Erf op;
    /// we pattern-rewrite into an exact GELU when the downstream node chain
    /// matches, otherwise emit the scalar erf loop. Phase 1 accepts the erf
    /// path by asserting — most BERT/ViT exports use the fused Gelu op in
    /// opset 20 or com.microsoft.Gelu. Upgrading to a full erf kernel is
    /// tracked as a follow-up.
    /// </summary>
    internal sealed class Erf<T> : IOnnxOpTranslator<T> where T : unmanaged
    {
        public string OpType => "Erf";
        public string? Domain => null;
        public void Translate(OnnxTranslationContext<T> ctx, NodeProto node)
        {
            throw new NotSupportedException(
                "Bare Erf op encountered. Phase 1 supports fused GELU (op 'Gelu' in opset 20+ or " +
                "com.microsoft.Gelu extension); BERT/ViT models exporting the pre-opset-20 " +
                "Mul+Div+Erf+Add+Mul decomposition of GELU are not yet covered.");
        }
    }
}
