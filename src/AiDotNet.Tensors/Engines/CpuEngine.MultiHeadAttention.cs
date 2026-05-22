using System;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines;

public partial class CpuEngine
{
    /// <summary>
    /// Fused multi-head attention forward — does Q/K/V projection, multi-head
    /// reshape, scaled dot-product attention, and output projection in a
    /// single call. Returns <c>[B, seq, dModel]</c>.
    ///
    /// <para>
    /// Equivalent to chaining the existing primitives:
    /// </para>
    /// <code>
    /// q   = TensorMatMul(input, qWeight)                  // [B, seq, dModel]
    /// k   = TensorMatMul(input, kWeight)                  // [B, seq, dModel]
    /// v   = TensorMatMul(input, vWeight)                  // [B, seq, dModel]
    /// q   = q.Reshape(B, seq, numHeads, dHead).Permute(0,2,1,3)   // [B, H, seq, dH]
    /// k   = k.Reshape(B, seq, numHeads, dHead).Permute(0,2,1,3)
    /// v   = v.Reshape(B, seq, numHeads, dHead).Permute(0,2,1,3)
    /// out = ScaledDotProductAttention(q, k, v)            // [B, H, seq, dH]
    /// out = out.Permute(0,2,1,3).Reshape(B, seq, dModel)
    /// out = TensorMatMul(out, outWeight)                  // [B, seq, dModel]
    /// </code>
    ///
    /// <para>
    /// Why the wrapper exists at all when the parts already exist: the AIsEval
    /// fair-comparison rerun (issue #436 P0) showed Transformer
    /// <c>Predict()</c> at <c>bs=128</c> was <b>16.8× slower than PyTorch</b>
    /// (233 ms vs 13.85 ms). The MHA forward path
    /// (<c>AiDotNet.NeuralNetworks.Layers.MultiHeadAttentionLayer</c>) records
    /// 5+ ops on the autograd tape per call (Q proj, K proj, V proj, SDPA,
    /// output proj — plus the reshapes), each entering and exiting a
    /// <c>GraphMode</c>-aware recording path. Inference doesn't need any of
    /// those tape entries. Bundling the whole MHA chain into one engine call
    /// that explicitly bypasses graph mode reduces 5 dispatches to 1 and
    /// reclaims the autograd-recording overhead. The actual compute stays
    /// identical (same SimdGemm and same SDPA kernel underneath).
    /// </para>
    ///
    /// <para>
    /// <b>Forward-only.</b> Backward is not yet implemented; calling this
    /// inside a <c>GradientTape</c> scope throws. Training paths should
    /// keep using the decomposed primitive chain (which records each op on
    /// the tape) until a fused backward lands in a follow-up PR.
    /// </para>
    /// </summary>
    /// <param name="input">[B, seq, dModel] input sequence.</param>
    /// <param name="qWeight">[dModel, dModel] query projection.</param>
    /// <param name="kWeight">[dModel, dModel] key projection.</param>
    /// <param name="vWeight">[dModel, dModel] value projection.</param>
    /// <param name="outWeight">[dModel, dModel] output projection.</param>
    /// <param name="numHeads">Number of attention heads. Must divide dModel.</param>
    public virtual Tensor<T> MultiHeadAttentionForward<T>(
        Tensor<T> input,
        Tensor<T> qWeight,
        Tensor<T> kWeight,
        Tensor<T> vWeight,
        Tensor<T> outWeight,
        int numHeads)
    {
        if (input is null) throw new ArgumentNullException(nameof(input));
        if (qWeight is null) throw new ArgumentNullException(nameof(qWeight));
        if (kWeight is null) throw new ArgumentNullException(nameof(kWeight));
        if (vWeight is null) throw new ArgumentNullException(nameof(vWeight));
        if (outWeight is null) throw new ArgumentNullException(nameof(outWeight));
        if (input.Rank != 3)
            throw new ArgumentException($"MultiHeadAttentionForward expects rank-3 input [B, seq, dModel]; got rank {input.Rank}.", nameof(input));
        if (numHeads <= 0)
            throw new ArgumentOutOfRangeException(nameof(numHeads), "numHeads must be positive.");

        int batch = input.Shape[0];
        int seqLen = input.Shape[1];
        int dModel = input.Shape[2];

        if (dModel % numHeads != 0)
            throw new ArgumentException($"dModel ({dModel}) must be divisible by numHeads ({numHeads}).", nameof(numHeads));
        int dHead = dModel / numHeads;

        if (qWeight.Rank != 2 || qWeight.Shape[0] != dModel || qWeight.Shape[1] != dModel)
            throw new ArgumentException($"qWeight must be [{dModel}, {dModel}].", nameof(qWeight));
        if (kWeight.Rank != 2 || kWeight.Shape[0] != dModel || kWeight.Shape[1] != dModel)
            throw new ArgumentException($"kWeight must be [{dModel}, {dModel}].", nameof(kWeight));
        if (vWeight.Rank != 2 || vWeight.Shape[0] != dModel || vWeight.Shape[1] != dModel)
            throw new ArgumentException($"vWeight must be [{dModel}, {dModel}].", nameof(vWeight));
        if (outWeight.Rank != 2 || outWeight.Shape[0] != dModel || outWeight.Shape[1] != dModel)
            throw new ArgumentException($"outWeight must be [{dModel}, {dModel}].", nameof(outWeight));

        if (GraphMode.IsActive)
            throw new InvalidOperationException(
                "MultiHeadAttentionForward is inference-only and does not yet support GradientTape. " +
                "For training, call the decomposed Q/K/V + SDPA + output projection primitives directly.");

        // Q/K/V projections. We project the entire [B, seq, dModel] input
        // against each weight in one GEMM. Reshape [B, seq, dModel] to
        // [B*seq, dModel] so the 3D-vs-2D matmul dispatch picks the fast path,
        // then reshape back to [B, seq, dModel].
        var inputFlat = input.Reshape(new[] { batch * seqLen, dModel });
        var qFlat = TensorMatMul(inputFlat, qWeight); // [B*seq, dModel]
        var kFlat = TensorMatMul(inputFlat, kWeight);
        var vFlat = TensorMatMul(inputFlat, vWeight);

        // Reshape to [B, seq, numHeads, dHead] then permute (0,2,1,3) -> [B, numHeads, seq, dHead].
        var q = qFlat.Reshape(new[] { batch, seqLen, numHeads, dHead }).Transpose(new[] { 0, 2, 1, 3 });
        var k = kFlat.Reshape(new[] { batch, seqLen, numHeads, dHead }).Transpose(new[] { 0, 2, 1, 3 });
        var v = vFlat.Reshape(new[] { batch, seqLen, numHeads, dHead }).Transpose(new[] { 0, 2, 1, 3 });

        // SDPA: existing primitive. Out shape [B, numHeads, seq, dHead].
        var attnOut = ScaledDotProductAttention<T>(q, k, v, mask: null, scale: null, out _);

        // Permute back (0,2,1,3) -> [B, seq, numHeads, dHead] then flatten heads -> [B, seq, dModel].
        var concat = attnOut.Transpose(new[] { 0, 2, 1, 3 }).Reshape(new[] { batch, seqLen, dModel });

        // Output projection: [B*seq, dModel] @ [dModel, dModel] -> [B*seq, dModel] -> reshape.
        var concatFlat = concat.Reshape(new[] { batch * seqLen, dModel });
        var outFlat = TensorMatMul(concatFlat, outWeight);
        return outFlat.Reshape(new[] { batch, seqLen, dModel });
    }
}
