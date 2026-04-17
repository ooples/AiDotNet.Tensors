using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tensors.Onnx.Protos;

namespace AiDotNet.Tensors.Onnx.Operators;

/// <summary>
/// ONNX normalization operator translators: LayerNormalization,
/// BatchNormalization (inference mode).
/// </summary>
internal static class NormOperators
{
    internal static void Register<T>(OnnxOpTranslatorRegistry<T> r) where T : unmanaged
    {
        r.Register(new LayerNormalization<T>());
        r.Register(new BatchNormalization<T>());
    }

    /// <summary>
    /// ONNX LayerNormalization — <c>LayerNorm(X, scale, B, axis, epsilon)</c>.
    /// ONNX normalises across all dims from <c>axis</c> onward. The engine's
    /// <see cref="IEngine.LayerNorm"/> normalises across the last dim only
    /// (transformer-standard); ResNet/BERT/ViT consistently use axis = -1
    /// (the last axis) so the two match. Non-default axis throws.
    /// </summary>
    internal sealed class LayerNormalization<T> : IOnnxOpTranslator<T> where T : unmanaged
    {
        public string OpType => "LayerNormalization";
        public string? Domain => null;
        public void Translate(OnnxTranslationContext<T> ctx, NodeProto node)
        {
            int axis = ctx.GetIntAttrAsInt(node, "axis", -1);
            var x = ctx.GetTensor(node.Input[0]);
            if (axis != -1 && axis != x.Rank - 1)
                throw new NotSupportedException(
                    $"LayerNormalization axis={axis} is not the last dimension (rank={x.Rank}). " +
                    "Phase 1 import requires last-axis normalization (which BERT/ViT/ResNet all use).");
            float epsilon = ctx.GetFloatAttr(node, "epsilon", 1e-5f);
            var scale = ctx.GetTensor(node.Input[1]);
            var bias = ctx.GetTensor(node.Input[2]);
            var y = ctx.Engine.LayerNorm(x, scale, bias, epsilon, out _, out _);
            ctx.PutTensor(node.Output[0], y);
        }
    }

    /// <summary>
    /// ONNX BatchNormalization in inference mode (training_mode=0, the default
    /// for exported models). Inputs: X, scale, B, input_mean, input_var.
    /// The engine's BatchNorm takes (input, gamma, beta, eps) and outputs
    /// mean/var via out params — in inference we ignore those outputs.
    ///
    /// <para>Note: engine's BatchNorm RECOMPUTES batch statistics rather than
    /// using the ONNX-provided running mean/var. For inference-frozen ONNX
    /// models that's a semantic mismatch (ONNX expects (x - running_mean) /
    /// sqrt(running_var + eps)). Fused into a BatchNormInference variant as
    /// follow-up; Phase 1 fails loudly so downstream users know to recompile
    /// with the running-stats-aware path.</para>
    /// </summary>
    internal sealed class BatchNormalization<T> : IOnnxOpTranslator<T> where T : unmanaged
    {
        public string OpType => "BatchNormalization";
        public string? Domain => null;
        public void Translate(OnnxTranslationContext<T> ctx, NodeProto node)
        {
            int trainingMode = ctx.GetIntAttrAsInt(node, "training_mode", 0);
            if (trainingMode != 0)
                throw new NotSupportedException(
                    "BatchNormalization in training_mode=1 is not supported during ONNX import. " +
                    "Re-export the model with training_mode=0 (the default for inference-ready exports).");
            // For inference: BN(x) = scale * (x - running_mean) / sqrt(running_var + eps) + bias
            // The engine's BatchNorm recomputes stats from the current batch, which diverges from
            // ONNX semantics for frozen inference. We decompose manually using engine primitives
            // to preserve ONNX semantics.
            var x = ctx.GetTensor(node.Input[0]);
            var scale = ctx.GetTensor(node.Input[1]);   // [C]
            var bias  = ctx.GetTensor(node.Input[2]);   // [C]
            var mean  = ctx.GetTensor(node.Input[3]);   // [C]
            var varT  = ctx.GetTensor(node.Input[4]);   // [C]
            float epsilon = ctx.GetFloatAttr(node, "epsilon", 1e-5f);

            // normalize = (x - mean) / sqrt(var + eps)
            // Shape: x is [N,C,H,W] for conv BN; mean/var are [C]. We reshape
            // per-channel stats to [1,C,1,1] so broadcast math aligns with x.
            var cShape = MakePerChannelShape(x._shape, scale._shape.Length);
            var meanView = ctx.Engine.Reshape(mean, cShape);
            var varView  = ctx.Engine.Reshape(varT, cShape);
            var scaleView = ctx.Engine.Reshape(scale, cShape);
            var biasView  = ctx.Engine.Reshape(bias, cShape);

            var centered = ctx.Engine.TensorBroadcastSubtract(x, meanView);
            var epsTensor = FillTensor(cShape, epsilon);
            var denom = ctx.Engine.TensorSqrt(ctx.Engine.TensorBroadcastAdd(varView, epsTensor));
            var normalized = ctx.Engine.TensorDivide(centered, ctx.Engine.TensorBroadcastAdd(
                denom, FillTensor(cShape, 0f))); // broadcast denom to match centered shape
            var scaled = ctx.Engine.TensorBroadcastMultiply(normalized, scaleView);
            var result = ctx.Engine.TensorBroadcastAdd(scaled, biasView);
            ctx.PutTensor(node.Output[0], result);
        }

        private static int[] MakePerChannelShape(int[] inputShape, int channelsRank)
        {
            // inputShape is [N, C, ...] (conv BN) or [N, ..., C] (channels-last).
            // ONNX always uses channels-second (NCHW), so the per-channel stat's
            // broadcast shape is [1, C, 1, 1, ...].
            var s = new int[inputShape.Length];
            s[0] = 1;
            s[1] = inputShape[1];
            for (int i = 2; i < s.Length; i++) s[i] = 1;
            return s;
        }

        private static Tensor<T> FillTensor(int[] shape, float value)
        {
            var t = new Tensor<T>(shape);
            var span = t.AsWritableSpan();
            var v = MathHelper.GetNumericOperations<T>().FromDouble(value);
            for (int i = 0; i < span.Length; i++) span[i] = v;
            return t;
        }
    }
}
