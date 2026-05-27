using System;
using System.Buffers;
using System.Runtime.CompilerServices;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Engines.Simd;
using AiDotNet.Tensors.Helpers;
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
    /// reclaims the autograd-recording overhead. The float fast path goes
    /// further: it calls <see cref="SimdGemm.Sgemm"/> directly for all four
    /// GEMMs and writes Q/K/V into <c>[B, heads, seq, dHead]</c> layout
    /// inline (so the four large strided-transpose materializations that
    /// dominated the wrapper's cost are gone).
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
        int numHeads,
        Tensor<bool>? mask = null)
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

        // Float fast path: direct SimdGemm + in-layout Q/K/V scratch + a
        // single output-projection transpose. Avoids the 4 strided-transpose
        // materializations and 4 Tensor<float> intermediates that the
        // generic-T path goes through. This is the path that closes the
        // AIsEval Transformer-inference gap.
        if (typeof(T) == typeof(float))
            return (Tensor<T>)(object)MultiHeadAttentionForwardFloat(
                (Tensor<float>)(object)input,
                (Tensor<float>)(object)qWeight,
                (Tensor<float>)(object)kWeight,
                (Tensor<float>)(object)vWeight,
                (Tensor<float>)(object)outWeight,
                mask,
                batch, seqLen, dModel, numHeads, dHead);

        return MultiHeadAttentionForwardGeneric(
            input, qWeight, kWeight, vWeight, outWeight, mask,
            batch, seqLen, dModel, numHeads, dHead);
    }

    /// <summary>
    /// Float32 fast path. Calls <see cref="SimdGemm.Sgemm"/> directly for the
    /// four projections, writes Q/K/V into <c>[B, heads, seq, dHead]</c>
    /// layout inline via a small post-GEMM scatter (one O(B*seq*dModel)
    /// memory pass instead of three through <see cref="Tensor{T}.Transpose"/>),
    /// reuses pooled scratch buffers across the whole call, and only
    /// constructs <see cref="Tensor{T}"/> wrappers at the boundaries with
    /// <see cref="ScaledDotProductAttention{T}"/>.
    /// </summary>
    private unsafe Tensor<float> MultiHeadAttentionForwardFloat(
        Tensor<float> input,
        Tensor<float> qWeight, Tensor<float> kWeight, Tensor<float> vWeight, Tensor<float> outWeight,
        Tensor<bool>? mask,
        int batch, int seqLen, int dModel, int numHeads, int dHead)
    {
        int totalRows = batch * seqLen;
        int qkvElems = batch * numHeads * seqLen * dHead; // == totalRows * dModel
        var pool = ArrayPool<float>.Shared;

        var qFlatBuf = pool.Rent(totalRows * dModel);  // [B*seq, dModel] GEMM output
        var kFlatBuf = pool.Rent(totalRows * dModel);
        var vFlatBuf = pool.Rent(totalRows * dModel);
        var qHeadBuf = pool.Rent(qkvElems);            // [B, heads, seq, dHead] post-transpose
        var kHeadBuf = pool.Rent(qkvElems);
        var vHeadBuf = pool.Rent(qkvElems);
        var attnHeadBuf = pool.Rent(qkvElems);         // SDPA output
        var concatBuf = pool.Rent(totalRows * dModel); // [B*seq, dModel] post-inverse-transpose

        var output = AutoTensorCache.RentOrAllocate<float>(new[] { batch, seqLen, dModel });

        try
        {
            // ---- Q/K/V projections via direct SimdGemm calls. ----
            // input [B*seq, dModel] @ Weight [dModel, dModel] -> [B*seq, dModel]
            var inputSpan = input.AsSpan();
            int batchSeqRows = totalRows;
            SimdGemm.Sgemm(
                inputSpan, dModel, false,
                qWeight.AsSpan(), dModel, false,
                qFlatBuf.AsSpan(0, totalRows * dModel),
                batchSeqRows, dModel, dModel);
            SimdGemm.Sgemm(
                inputSpan, dModel, false,
                kWeight.AsSpan(), dModel, false,
                kFlatBuf.AsSpan(0, totalRows * dModel),
                batchSeqRows, dModel, dModel);
            SimdGemm.Sgemm(
                inputSpan, dModel, false,
                vWeight.AsSpan(), dModel, false,
                vFlatBuf.AsSpan(0, totalRows * dModel),
                batchSeqRows, dModel, dModel);

            // ---- Transpose Q/K/V from [B*seq, dModel] = [B, seq, H, dHead]
            // (read as 4D with strides 1, seq*H*dHead, etc) into
            // [B, H, seq, dHead]. One tight nested loop per matrix.
            // The mapping for output index (b, h, s, d):
            //   src_off = (b * seq + s) * dModel + h * dHead + d
            //   dst_off = ((b * H + h) * seq + s) * dHead + d
            // dHead is contiguous in both layouts, so the innermost copy is
            // a dHead-element block-copy (Span<T>.CopyTo gives us SIMD).
            TransposeQkv(qFlatBuf, qHeadBuf, batch, seqLen, numHeads, dHead);
            TransposeQkv(kFlatBuf, kHeadBuf, batch, seqLen, numHeads, dHead);
            TransposeQkv(vFlatBuf, vHeadBuf, batch, seqLen, numHeads, dHead);

            // ---- SDPA. Wrap the pooled buffers as Tensor<float> views just
            // long enough to call the existing optimized primitive, which
            // does scale + softmax + matmul internally. The wrapping is
            // O(1) — no data copy. ----
            var qHeadTensor = WrapAsTensor(qHeadBuf, batch, numHeads, seqLen, dHead);
            var kHeadTensor = WrapAsTensor(kHeadBuf, batch, numHeads, seqLen, dHead);
            var vHeadTensor = WrapAsTensor(vHeadBuf, batch, numHeads, seqLen, dHead);

            var attnTensor = ScaledDotProductAttention<float>(
                qHeadTensor, kHeadTensor, vHeadTensor,
                mask: mask, scale: null, out _);

            // Copy SDPA output ([B, H, seq, dHead]) into our pooled buffer
            // so the inverse-transpose is a contiguous read pass.
            attnTensor.AsSpan().Slice(0, qkvElems).CopyTo(attnHeadBuf.AsSpan(0, qkvElems));

            // ---- Inverse transpose: [B, H, seq, dHead] -> [B*seq, dModel]. ----
            InverseTransposeQkv(attnHeadBuf, concatBuf, batch, seqLen, numHeads, dHead);

            // ---- Output projection: [B*seq, dModel] @ outWeight [dModel, dModel] -> output. ----
            SimdGemm.Sgemm(
                concatBuf.AsSpan(0, totalRows * dModel), dModel, false,
                outWeight.AsSpan(), dModel, false,
                output.AsWritableSpan(),
                batchSeqRows, dModel, dModel);

            return output;
        }
        finally
        {
            pool.Return(qFlatBuf);
            pool.Return(kFlatBuf);
            pool.Return(vFlatBuf);
            pool.Return(qHeadBuf);
            pool.Return(kHeadBuf);
            pool.Return(vHeadBuf);
            pool.Return(attnHeadBuf);
            pool.Return(concatBuf);
        }
    }

    /// <summary>
    /// Wraps a rented array slice as a contiguous rank-4 <see cref="Tensor{T}"/>
    /// for passing to <see cref="ScaledDotProductAttention{T}"/>. The
    /// tensor's lifetime is tied to the caller's <c>try/finally</c>;
    /// internal callers must not store the result past the pool return.
    /// </summary>
    private static Tensor<float> WrapAsTensor(float[] buffer, int d0, int d1, int d2, int d3)
    {
        // The buffer is from ArrayPool, which returns a BUCKET-sized array (rounded
        // up to a power of 2) that is ≥ the logical element count and only equals it
        // when the count is itself a power of 2. The Tensor ctor validates
        // data.Length == product(dimensions) exactly, so passing the whole array threw
        // "The number of values does not match the specified shape" whenever
        // d0*d1*d2*d3 wasn't a power of 2 — i.e. every non-power-of-2 sequence length
        // (issue #468). Wrap EXACTLY product(dimensions) elements as a zero-copy
        // Memory view so the length always matches regardless of the rented capacity.
        // Lifetime: this Tensor must not outlive the caller's pool.Return — it is used
        // only as a thunk into SDPA.
        int n = d0 * d1 * d2 * d3;
        var view = Vector<float>.WrapMemory(buffer.AsMemory(0, n));
        return new Tensor<float>(new[] { d0, d1, d2, d3 }, view);
    }

    private static void TransposeQkv(float[] src, float[] dst, int batch, int seq, int numHeads, int dHead)
    {
        // src layout: [B, seq, numHeads, dHead]  (linear stride: dHead, numHeads*dHead, seq*numHeads*dHead)
        // dst layout: [B, numHeads, seq, dHead]
        int srcStrideS = numHeads * dHead;
        int srcStrideB = seq * srcStrideS;
        int dstStrideH = seq * dHead;
        int dstStrideB = numHeads * dstStrideH;

        // Parallelize across batch. Each batch element writes to a disjoint
        // dstStrideB-sized region so there's no contention. For the AIsEval
        // shape (batch=128) this typically hits 4-8x of the cores.
        System.Threading.Tasks.Parallel.For(0, batch, b =>
        {
            var srcSpan = src.AsSpan();
            var dstSpan = dst.AsSpan();
            for (int s = 0; s < seq; s++)
            {
                int srcBase = b * srcStrideB + s * srcStrideS;
                int dstBaseBS = b * dstStrideB + s * dHead;
                for (int h = 0; h < numHeads; h++)
                {
                    int srcOff = srcBase + h * dHead;
                    int dstOff = dstBaseBS + h * dstStrideH;
                    srcSpan.Slice(srcOff, dHead).CopyTo(dstSpan.Slice(dstOff, dHead));
                }
            }
        });
    }

    private static void InverseTransposeQkv(float[] src, float[] dst, int batch, int seq, int numHeads, int dHead)
    {
        // src layout: [B, numHeads, seq, dHead]
        // dst layout: [B, seq, numHeads, dHead]  (== [B*seq, dModel])
        int srcStrideH = seq * dHead;
        int srcStrideB = numHeads * srcStrideH;
        int dstStrideS = numHeads * dHead;
        int dstStrideB = seq * dstStrideS;

        System.Threading.Tasks.Parallel.For(0, batch, b =>
        {
            var srcSpan = src.AsSpan();
            var dstSpan = dst.AsSpan();
            for (int s = 0; s < seq; s++)
            {
                int dstBaseBS = b * dstStrideB + s * dstStrideS;
                int srcBaseBS = b * srcStrideB + s * dHead;
                for (int h = 0; h < numHeads; h++)
                {
                    int srcOff = srcBaseBS + h * srcStrideH;
                    int dstOff = dstBaseBS + h * dHead;
                    srcSpan.Slice(srcOff, dHead).CopyTo(dstSpan.Slice(dstOff, dHead));
                }
            }
        });
    }

    /// <summary>
    /// Generic-T path. Composes existing tensor-level primitives. Used for
    /// double/decimal/BigInteger/custom numerics — those callers tend to
    /// be correctness-driven and don't pay the per-call dispatch overhead
    /// that motivates the float fast path above.
    /// </summary>
    private Tensor<T> MultiHeadAttentionForwardGeneric<T>(
        Tensor<T> input,
        Tensor<T> qWeight, Tensor<T> kWeight, Tensor<T> vWeight, Tensor<T> outWeight,
        Tensor<bool>? mask,
        int batch, int seqLen, int dModel, int numHeads, int dHead)
    {
        var inputFlat = input.Reshape(new[] { batch * seqLen, dModel });
        var qFlat = TensorMatMul(inputFlat, qWeight);
        var kFlat = TensorMatMul(inputFlat, kWeight);
        var vFlat = TensorMatMul(inputFlat, vWeight);

        var q = qFlat.Reshape(new[] { batch, seqLen, numHeads, dHead }).Transpose(new[] { 0, 2, 1, 3 });
        var k = kFlat.Reshape(new[] { batch, seqLen, numHeads, dHead }).Transpose(new[] { 0, 2, 1, 3 });
        var v = vFlat.Reshape(new[] { batch, seqLen, numHeads, dHead }).Transpose(new[] { 0, 2, 1, 3 });

        var attnOut = ScaledDotProductAttention<T>(q, k, v, mask: mask, scale: null, out _);

        var concat = attnOut.Transpose(new[] { 0, 2, 1, 3 }).Reshape(new[] { batch, seqLen, dModel });
        var concatFlat = concat.Reshape(new[] { batch * seqLen, dModel });
        var outFlat = TensorMatMul(concatFlat, outWeight);
        return outFlat.Reshape(new[] { batch, seqLen, dModel });
    }
}
