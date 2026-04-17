using AiDotNet.Tensors.Onnx.Protos;

namespace AiDotNet.Tensors.Onnx.Operators;

/// <summary>
/// ONNX Attention operator translator for the <c>com.microsoft.Attention</c>
/// extension. This is the op BERT / ViT exports emit for the fused QKV block.
///
/// <para>MS Attention layout varies — this translator handles the 3-input
/// form used by Hugging Face <c>optimum</c> exports: (input, weights, bias,
/// mask_index?, past?, attention_bias?). Weights hold packed Q·K·V in a
/// single [hidden, 3*hidden] matrix; we split into Q, K, V, reshape to
/// per-head, and call <c>IEngine.ScaledDotProductAttention</c>.</para>
/// </summary>
internal static class AttentionOperator
{
    internal static void Register<T>(OnnxOpTranslatorRegistry<T> r) where T : unmanaged
    {
        r.Register(new MicrosoftAttention<T>());
    }

    internal sealed class MicrosoftAttention<T> : IOnnxOpTranslator<T> where T : unmanaged
    {
        public string OpType => "Attention";
        public string? Domain => "com.microsoft";
        public void Translate(OnnxTranslationContext<T> ctx, NodeProto node)
        {
            var input = ctx.GetTensor(node.Input[0]);      // [batch, seq, hidden]
            var weights = ctx.GetTensor(node.Input[1]);    // [hidden, 3*hidden]
            var bias = ctx.GetTensor(node.Input[2]);       // [3*hidden]
            int numHeads = ctx.GetIntAttrAsInt(node, "num_heads", 0);
            if (numHeads <= 0)
                throw new InvalidDataException(
                    "com.microsoft.Attention requires num_heads > 0.");
            if (input.Rank != 3)
                throw new NotSupportedException(
                    $"Phase 1 Attention expects 3D input [batch, seq, hidden]; got rank {input.Rank}.");

            int batch = input._shape[0];
            int seq = input._shape[1];
            int hidden = input._shape[2];
            int headDim = hidden / numHeads;

            // Project to [batch, seq, 3*hidden] then split along last dim.
            var projected = ctx.Engine.TensorMatMul(input, weights);
            var biased = ctx.Engine.TensorBroadcastAdd(projected, bias);

            // Split into Q, K, V along the last dim (width 3*hidden → 3 × hidden).
            var qkv = ctx.Engine.TensorSplit(biased, 3, axis: 2);
            var q = ReshapeForAttention(ctx, qkv[0], batch, seq, numHeads, headDim);
            var k = ReshapeForAttention(ctx, qkv[1], batch, seq, numHeads, headDim);
            var v = ReshapeForAttention(ctx, qkv[2], batch, seq, numHeads, headDim);

            var attn = ctx.Engine.ScaledDotProductAttention(q, k, v, mask: null, scale: null, out _);
            // attn: [batch, heads, seq, headDim] → [batch, seq, hidden].
            var merged = ctx.Engine.TensorPermute(attn, new[] { 0, 2, 1, 3 });
            var output = ctx.Engine.Reshape(merged, new[] { batch, seq, hidden });
            ctx.PutTensor(node.Output[0], output);
        }

        // ONNX Attention's Q/K/V come out as [batch, seq, hidden] after the
        // MatMul + Add. Reshape to [batch, seq, heads, headDim] then permute
        // to [batch, heads, seq, headDim] — the layout SDPA expects.
        private static LinearAlgebra.Tensor<T> ReshapeForAttention(
            OnnxTranslationContext<T> ctx,
            LinearAlgebra.Tensor<T> x,
            int batch, int seq, int numHeads, int headDim)
        {
            var fourD = ctx.Engine.Reshape(x, new[] { batch, seq, numHeads, headDim });
            return ctx.Engine.TensorPermute(fourD, new[] { 0, 2, 1, 3 });
        }
    }
}
