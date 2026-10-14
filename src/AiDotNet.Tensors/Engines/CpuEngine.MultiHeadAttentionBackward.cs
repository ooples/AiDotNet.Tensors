// Copyright (c) AiDotNet. All rights reserved.

using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Simd;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using static AiDotNet.Tensors.Compatibility.MethodImplHelper;

namespace AiDotNet.Tensors.Engines;

public partial class CpuEngine
{
    // ──────────────────────────────────────────────────────────────────────────
    // Fused, tape-aware multi-head attention forward + backward (float), #436.
    //
    // The decomposed training chain records a dozen tape nodes per call (three projections, reshapes, permutes, SDPA,
    // the output projection) and materialises every intermediate. This variant runs the inference kernel's structure
    // with EXACT softmax, keeps the fused QKV projection, the pre-projection attention output and the per-row softmax
    // log-sum-exp, and records ONE tape node. Its backward recomputes each (batch, head) slice's probabilities from the
    // saved log-sum-exp (P = exp(s - lse): no [B, H, S, S] probability tensor is kept) and accumulates
    //   dV = Pᵀ dO,  dP = dO Vᵀ,  dS = P ⊙ (dP - rowsum(dO ⊙ O)),  dQ = dS K · scale,  dK = dSᵀ Q · scale
    // into one [B·S, 3·D] buffer, from which two GEMMs give the three projection-weight gradients and the input's.
    // Each (batch, head) slice writes its own columns of that buffer, so the result does not depend on the thread count.
    // ──────────────────────────────────────────────────────────────────────────

    /// <summary>Saved state of one fused MHA tape node.</summary>
    private sealed class MhaTrainState
    {
        internal MhaTrainState(float[] qkv, float[] concat, float[] lse, Tensor<bool>? mask,
            int batch, int seq, int dModel, int heads)
        {
            Qkv = qkv; Concat = concat; Lse = lse; Mask = mask;
            Batch = batch; Seq = seq; DModel = dModel; Heads = heads;
        }

        internal float[] Qkv { get; }      // [B*S, 3D]: rows [q | k | v]
        internal float[] Concat { get; }   // [B*S, D]: attention output before the output projection
        internal float[] Lse { get; }      // [B*H*S]: per-row softmax log-sum-exp; -inf for a fully masked row
        internal Tensor<bool>? Mask { get; }
        internal int Batch { get; }
        internal int Seq { get; }
        internal int DModel { get; }
        internal int Heads { get; }
        internal int DHead => DModel / Heads;
    }

    /// <summary>
    /// Training forward: the fused float MHA with exact softmax, recorded as one tape node whose backward is
    /// <see cref="MultiHeadAttentionBackwardFloat"/>.
    /// </summary>
    private Tensor<float> MultiHeadAttentionForwardFloatTrain(
        Tensor<float> input, Tensor<float> qWeight, Tensor<float> kWeight, Tensor<float> vWeight, Tensor<float> outWeight,
        Tensor<bool>? mask, int batch, int seqLen, int dModel, int numHeads, int dHead)
    {
        int rows = batch * seqLen, d3 = 3 * dModel;
        var qkv = new float[rows * d3];
        var concat = new float[rows * dModel];
        var lse = new float[batch * numHeads * seqLen];
        var fusedW = FuseQkvWeights(qWeight, kWeight, vWeight, dModel);
        var inputArr = input.GetFlattenedData();
        SimdGemm.Sgemm(inputArr.AsSpan(0, rows * dModel), dModel, false, fusedW, d3, false, qkv, rows, dModel, d3);

        float scale = (float)(1.0 / Math.Sqrt(dHead));
        int bh = batch * numHeads;
        int chunks = Math.Max(1, Math.Min(bh, CpuParallelSettings.MaxDegreeOfParallelism));
        PersistentParallelExecutor.Instance.Execute(chunks, chunk =>
        {
            var q = new float[seqLen * dHead];
            var kT = new float[dHead * seqLen];
            var v = new float[seqLen * dHead];
            var p = new float[seqLen * seqLen];
            var o = new float[seqLen * dHead];
            for (int s = chunk; s < bh; s += chunks)
            {
                int b = s / numHeads, h = s % numHeads;
                GatherHead(qkv, d3, b, h, seqLen, dHead, 0, q, transposed: false);
                GatherHead(qkv, d3, b, h, seqLen, dHead, dModel, kT, transposed: true);
                GatherHead(qkv, d3, b, h, seqLen, dHead, 2 * dModel, v, transposed: false);
                SimdGemm.SgemmSequential(q, kT, p, seqLen, dHead, seqLen);
                SoftmaxRowsExact(p, seqLen, scale, mask, b, h, lse, s * seqLen);
                SimdGemm.SgemmSequential(p, v, o, seqLen, seqLen, dHead);
                ScatterHead(o, concat, dModel, b, h, seqLen, dHead, 0);
            }
        });

        var output = new Tensor<float>(new[] { batch, seqLen, dModel });
        using (var outputKeep = new KeepAliveScope(output))
        using (var outWeightKeep = new KeepAliveScope(outWeight))
            SimdGemm.Sgemm(concat, dModel, false, outWeight.AsSpanUnmarked(), dModel, false,
                output.AsWritableSpanUnmarked(), rows, dModel, dModel);

        var state = new MhaTrainState(qkv, concat, lse, mask, batch, seqLen, dModel, numHeads);
        DifferentiableOps.RecordIfActive<float>("MultiHeadAttentionForward", output,
            new[] { input, qWeight, kWeight, vWeight, outWeight }, MultiHeadAttentionBackwardFloat, new object[] { state });
        return output;
    }

    /// <summary>
    /// Backward of <see cref="MultiHeadAttentionForwardFloatTrain"/>: gradients for the input and the four projection
    /// weights, each only when the tape needs it.
    /// </summary>
    [MethodImpl(Hot)]
    private static void MultiHeadAttentionBackwardFloat(
        Tensor<float> gradOutput, Tensor<float>[] inp, Tensor<float> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<float>, Tensor<float>> grads)
    {
        var st = (MhaTrainState)savedState[0];
        int batch = st.Batch, seq = st.Seq, dModel = st.DModel, heads = st.Heads, dHead = st.DHead;
        int rows = batch * seq, d3 = 3 * dModel;
        Tensor<float> input = inp[0], qWeight = inp[1], kWeight = inp[2], vWeight = inp[3], outWeight = inp[4];
        bool needInput = DifferentiableOps.IsGradientRequired(input);
        bool needQkvWeights = DifferentiableOps.IsGradientRequired(qWeight) || DifferentiableOps.IsGradientRequired(kWeight)
            || DifferentiableOps.IsGradientRequired(vWeight);
        bool needOutWeight = DifferentiableOps.IsGradientRequired(outWeight);
        var gradY = gradOutput.GetFlattenedData();

        if (needOutWeight)
        {
            // dWo = concatᵀ · dY: [D, B*S] · [B*S, D].
            var gWo = new Tensor<float>(new[] { dModel, dModel });
            using (var keep = new KeepAliveScope(gWo))
                SimdGemm.Sgemm(st.Concat, dModel, true, gradY.AsSpan(0, rows * dModel), dModel, false,
                    gWo.AsWritableSpanUnmarked(), dModel, rows, dModel);
            DifferentiableOps.AccumulateGrad(grads, outWeight, gWo, engine);
        }
        if (!needInput && !needQkvWeights) return;

        // dConcat = dY · Woᵀ.
        var dConcat = new float[rows * dModel];
        using (var keep = new KeepAliveScope(outWeight))
            SimdGemm.Sgemm(gradY.AsSpan(0, rows * dModel), dModel, false, outWeight.AsSpanUnmarked(), dModel, true,
                dConcat, rows, dModel, dModel);

        var dQkv = new float[rows * d3];
        float scale = (float)(1.0 / Math.Sqrt(dHead));
        int bh = batch * heads;
        int chunks = Math.Max(1, Math.Min(bh, CpuParallelSettings.MaxDegreeOfParallelism));
        PersistentParallelExecutor.Instance.Execute(chunks, chunk =>
        {
            int sd = seq * dHead, ss = seq * seq;
            var q = new float[sd]; var k = new float[sd]; var kT = new float[sd];
            var v = new float[sd]; var vT = new float[sd]; var o = new float[sd]; var dO = new float[sd];
            var p = new float[ss]; var pT = new float[ss]; var dP = new float[ss];
            var dQ = new float[sd]; var dK = new float[sd]; var dV = new float[sd];
            for (int s = chunk; s < bh; s += chunks)
            {
                int b = s / heads, h = s % heads;
                GatherHead(st.Qkv, d3, b, h, seq, dHead, 0, q, transposed: false);
                GatherHead(st.Qkv, d3, b, h, seq, dHead, dModel, k, transposed: false);
                GatherHead(st.Qkv, d3, b, h, seq, dHead, dModel, kT, transposed: true);
                GatherHead(st.Qkv, d3, b, h, seq, dHead, 2 * dModel, v, transposed: false);
                GatherHead(st.Qkv, d3, b, h, seq, dHead, 2 * dModel, vT, transposed: true);
                GatherHead(st.Concat, dModel, b, h, seq, dHead, 0, o, transposed: false);
                GatherHead(dConcat, dModel, b, h, seq, dHead, 0, dO, transposed: false);

                // P from the saved log-sum-exp: the forward's exact probabilities, masked entries 0.
                SimdGemm.SgemmSequential(q, kT, p, seq, dHead, seq);
                RecomputeProbabilities(p, seq, scale, st.Mask, b, h, st.Lse, s * seq);

                // dV = Pᵀ · dO.
                Transpose(p, pT, seq, seq);
                SimdGemm.SgemmSequential(pT, dO, dV, seq, seq, dHead);

                // dP = dO · Vᵀ;  dS = P ⊙ (dP - rowsum(dO ⊙ O)), in place in dP.
                SimdGemm.SgemmSequential(dO, vT, dP, seq, dHead, seq);
                for (int i = 0; i < seq; i++)
                {
                    float di = 0f;
                    for (int d = 0; d < dHead; d++) di += dO[i * dHead + d] * o[i * dHead + d];
                    int row = i * seq;
                    for (int j = 0; j < seq; j++) dP[row + j] = p[row + j] * (dP[row + j] - di) * scale;
                }

                // dQ = dS · K;  dK = dSᵀ · Q (the scale is already in dS).
                SimdGemm.SgemmSequential(dP, k, dQ, seq, seq, dHead);
                Transpose(dP, pT, seq, seq);
                SimdGemm.SgemmSequential(pT, q, dK, seq, seq, dHead);

                ScatterHead(dQ, dQkv, d3, b, h, seq, dHead, 0);
                ScatterHead(dK, dQkv, d3, b, h, seq, dHead, dModel);
                ScatterHead(dV, dQkv, d3, b, h, seq, dHead, 2 * dModel);
            }
        });

        if (needQkvWeights)
        {
            // [dWq | dWk | dWv] = Xᵀ · dQKV: [D, B*S] · [B*S, 3D].
            var dW = new float[dModel * d3];
            var inputArr = input.GetFlattenedData();
            SimdGemm.Sgemm(inputArr.AsSpan(0, rows * dModel), dModel, true, dQkv, d3, false, dW, dModel, rows, d3);
            if (DifferentiableOps.IsGradientRequired(qWeight))
                DifferentiableOps.AccumulateGrad(grads, qWeight, SliceColumns(dW, dModel, d3, 0), engine);
            if (DifferentiableOps.IsGradientRequired(kWeight))
                DifferentiableOps.AccumulateGrad(grads, kWeight, SliceColumns(dW, dModel, d3, dModel), engine);
            if (DifferentiableOps.IsGradientRequired(vWeight))
                DifferentiableOps.AccumulateGrad(grads, vWeight, SliceColumns(dW, dModel, d3, 2 * dModel), engine);
        }
        if (needInput)
        {
            // dX = dQKV · [Wq | Wk | Wv]ᵀ: [B*S, 3D] · [3D, D].
            var fusedW = FuseQkvWeights(qWeight, kWeight, vWeight, dModel);
            var gX = new Tensor<float>(new[] { batch, seq, dModel });
            using (var keep = new KeepAliveScope(gX))
                SimdGemm.Sgemm(dQkv, d3, false, fusedW, d3, true, gX.AsWritableSpanUnmarked(), rows, d3, dModel);
            DifferentiableOps.AccumulateGrad(grads, input, gX, engine);
        }
    }

    /// <summary>[Wq | Wk | Wv] side by side: a [D, 3D] matrix.</summary>
    private static float[] FuseQkvWeights(Tensor<float> qWeight, Tensor<float> kWeight, Tensor<float> vWeight, int dModel)
    {
        int d3 = 3 * dModel;
        var fused = new float[dModel * d3];
        var qw = qWeight.GetFlattenedData();
        var kw = kWeight.GetFlattenedData();
        var vw = vWeight.GetFlattenedData();
        for (int r = 0; r < dModel; r++)
        {
            Array.Copy(qw, r * dModel, fused, r * d3, dModel);
            Array.Copy(kw, r * dModel, fused, r * d3 + dModel, dModel);
            Array.Copy(vw, r * dModel, fused, r * d3 + 2 * dModel, dModel);
        }
        return fused;
    }

    /// <summary>
    /// Copies head <paramref name="h"/> of batch <paramref name="b"/> out of a [B*S, stride] buffer (block starting at
    /// column <paramref name="column"/>) into [S, dHead], or [dHead, S] when <paramref name="transposed"/>.
    /// </summary>
    private static void GatherHead(float[] src, int stride, int b, int h, int seq, int dHead, int column,
        float[] dst, bool transposed)
    {
        for (int s = 0; s < seq; s++)
        {
            int from = (b * seq + s) * stride + column + h * dHead;
            if (transposed)
                for (int d = 0; d < dHead; d++) dst[d * seq + s] = src[from + d];
            else
                Array.Copy(src, from, dst, s * dHead, dHead);
        }
    }

    /// <summary>Writes an [S, dHead] head slice back into its columns of a [B*S, stride] buffer.</summary>
    private static void ScatterHead(float[] src, float[] dst, int stride, int b, int h, int seq, int dHead, int column)
    {
        for (int s = 0; s < seq; s++)
            Array.Copy(src, s * dHead, dst, (b * seq + s) * stride + column + h * dHead, dHead);
    }

    private static void Transpose(float[] src, float[] dst, int rows, int cols)
    {
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++) dst[c * rows + r] = src[r * cols + c];
    }

    /// <summary>
    /// Scales, masks and softmaxes each row of <paramref name="scores"/> in place with exact exp, recording the row's
    /// log-sum-exp (-inf for a fully masked row, whose probabilities are all 0, as the inference kernel writes them).
    /// </summary>
    private static void SoftmaxRowsExact(float[] scores, int seq, float scale, Tensor<bool>? mask, int b, int h,
        float[] lse, int lseOffset)
    {
        for (int i = 0; i < seq; i++)
        {
            int row = i * seq;
            float max = float.NegativeInfinity;
            for (int j = 0; j < seq; j++)
            {
                float v = Masked(mask, b, h, i, j) ? float.NegativeInfinity : scores[row + j] * scale;
                scores[row + j] = v;
                if (v > max) max = v;
            }
            if (float.IsNegativeInfinity(max))
            {
                Array.Clear(scores, row, seq);
                lse[lseOffset + i] = float.NegativeInfinity;
                continue;
            }
            float sum = 0f;
            for (int j = 0; j < seq; j++)
            {
                float e = MathF.Exp(scores[row + j] - max);
                scores[row + j] = e;
                sum += e;
            }
            float inv = 1f / sum;
            for (int j = 0; j < seq; j++) scores[row + j] *= inv;
            lse[lseOffset + i] = max + MathF.Log(sum);
        }
    }

    /// <summary>Turns raw Q·Kᵀ rows into the forward's probabilities: exp(s · scale - lse), masked entries 0.</summary>
    private static void RecomputeProbabilities(float[] scores, int seq, float scale, Tensor<bool>? mask, int b, int h,
        float[] lse, int lseOffset)
    {
        for (int i = 0; i < seq; i++)
        {
            int row = i * seq;
            float l = lse[lseOffset + i];
            if (float.IsNegativeInfinity(l))
            {
                Array.Clear(scores, row, seq);
                continue;
            }
            for (int j = 0; j < seq; j++)
                scores[row + j] = Masked(mask, b, h, i, j) ? 0f : MathF.Exp(scores[row + j] * scale - l);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static bool Masked(Tensor<bool>? mask, int b, int h, int i, int j) =>
        mask is not null && !mask[BroadcastMaskIndex(b, mask._shape[0]), BroadcastMaskIndex(h, mask._shape[1]),
            BroadcastMaskIndex(i, mask._shape[2]), BroadcastMaskIndex(j, mask._shape[3])];

    /// <summary>Columns [<paramref name="column"/>, +rows) of a [rows, stride] buffer as a [rows, rows] tensor.</summary>
    private static Tensor<float> SliceColumns(float[] src, int rows, int stride, int column)
    {
        var t = new Tensor<float>(new[] { rows, rows });
        var dst = t.AsWritableSpan();
        for (int r = 0; r < rows; r++) src.AsSpan(r * stride + column, rows).CopyTo(dst.Slice(r * rows, rows));
        return t;
    }
}