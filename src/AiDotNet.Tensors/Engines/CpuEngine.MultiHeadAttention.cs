using System;
using System.Buffers;
using System.Runtime.CompilerServices;
#if NET5_0_OR_GREATER
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
#endif
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

        // Double fast path (#478 / #1675): the generic-T path below materializes the FULL
        // [B*heads, seq, seq] attention scores + weights + four strided-transpose tensors per call,
        // none pooled (scores exceed ArrayPool.Shared's 2^20-element bucket). At paper scale that is
        // ~50x the float path's allocation per MHA layer (measured 54x), and the resulting GC/heap
        // churn — not the math — is what makes a <double> transformer forward ~65x slower than
        // <float> on a memory-bounded host. Route double through the same per-head, single-reused-
        // scratch structure the float fast path uses (no full-scores matrix, no transpose buffers).
        if (typeof(T) == typeof(double))
            return (Tensor<T>)(object)MultiHeadAttentionForwardDouble(
                (Tensor<double>)(object)input,
                (Tensor<double>)(object)qWeight,
                (Tensor<double>)(object)kWeight,
                (Tensor<double>)(object)vWeight,
                (Tensor<double>)(object)outWeight,
                mask, batch, seqLen, dModel, numHeads, dHead);

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
        var pool = ArrayPool<float>.Shared;

        // Fused QKV projection: one [B*seq, dModel] @ [dModel, 3*dModel] GEMM
        // instead of three @ [dModel, dModel]. Reads `input` once (vs 3×), one
        // dispatch (vs 3), and a wider N=3*dModel that the SIMD microkernel
        // utilizes far better than N=dModel — #476 root-cause-2 (managed
        // projection GEMM quality). Output rows are [q(dModel)|k(dModel)|v(dModel)].
        int dModel3 = 3 * dModel;
        var qkvFlatBuf = pool.Rent(totalRows * dModel3);
        var fusedWBuf = pool.Rent(dModel * dModel3);    // [dModel, 3*dModel] = [qW | kW | vW]
        var concatBuf = pool.Rent(totalRows * dModel); // [B*seq, dModel] attention output, pre out-proj

        var output = AutoTensorCache.RentOrAllocate<float>(new[] { batch, seqLen, dModel });

        try
        {
            // ---- Fused Q/K/V projection. ----
            // Concatenate [qW | kW | vW] column-wise into a [dModel, 3*dModel]
            // weight once (cheap: dModel*3*dModel = ~48 KB at the AIsEval shape),
            // then ONE GEMM input[B*seq, dModel] @ fusedW[dModel, 3*dModel] ->
            // qkvFlat[B*seq, 3*dModel]. Each output row is [q|k|v].
            var inputSpan = input.AsSpan();
            int batchSeqRows = totalRows;
            {
                var qWS = qWeight.AsSpan();
                var kWS = kWeight.AsSpan();
                var vWS = vWeight.AsSpan();
                var fW = fusedWBuf.AsSpan();
                for (int kk = 0; kk < dModel; kk++)
                {
                    int dstRow = kk * dModel3;
                    qWS.Slice(kk * dModel, dModel).CopyTo(fW.Slice(dstRow, dModel));
                    kWS.Slice(kk * dModel, dModel).CopyTo(fW.Slice(dstRow + dModel, dModel));
                    vWS.Slice(kk * dModel, dModel).CopyTo(fW.Slice(dstRow + 2 * dModel, dModel));
                }
            }
            using (Profiling.Profiler.OpScope("MHA.QKVproj"))
            SimdGemm.Sgemm(
                inputSpan, dModel, false,
                fusedWBuf.AsSpan(0, dModel * dModel3), dModel3, false,
                qkvFlatBuf.AsSpan(0, totalRows * dModel3),
                batchSeqRows, dModel, dModel3);

            // ---- Transpose-fused SDPA (#476 root-cause-2). ----
            // Instead of three full [B,seq,dModel]→[B,H,seq,dHead] transpose
            // passes + buffers, an SDPA output buffer, and an inverse-transpose
            // pass, fold the per-head gather straight into the SDPA loop: each
            // (b,h) worker GATHERS its Q/K/V slice from the fused qkvFlat buffer
            // (K gathered already-transposed), runs Q·Kᵀ → softmax → P·V, and
            // SCATTERS the result directly into concatBuf at [B,seq,H,dHead] =
            // [B*seq,dModel]. Removes 4 transpose passes + 4 pooled buffers; the
            // gather/scatter is the same byte movement but fused with the GEMMs
            // (per-slice locality, no separate memory round-trips).
            double scaleVal = 1.0 / Math.Sqrt(dHead);
            using (Profiling.Profiler.OpScope("MHA.SDPA"))
            MultiHeadAttentionFusedSdpa(
                qkvFlatBuf, dModel3, /*qCol*/0, /*kCol*/dModel, /*vCol*/2 * dModel,
                mask, scaleVal,
                batch, numHeads, seqLen, dHead,
                concatBuf, dModel);

            // ---- Output projection: [B*seq, dModel] @ outWeight [dModel, dModel] -> output. ----
            using (Profiling.Profiler.OpScope("MHA.OutProj"))
            SimdGemm.Sgemm(
                concatBuf.AsSpan(0, totalRows * dModel), dModel, false,
                outWeight.AsSpan(), dModel, false,
                output.AsWritableSpan(),
                batchSeqRows, dModel, dModel);

            return output;
        }
        finally
        {
            pool.Return(qkvFlatBuf);
            pool.Return(fusedWBuf);
            pool.Return(concatBuf);
        }
    }

    /// <summary>
    /// Transpose-fused scaled-dot-product attention for the MHA float path
    /// (#476 root-cause-2). Reads Q/K/V for each (batch, head) slice DIRECTLY
    /// from the fused QKV projection buffer — no separate [B,H,seq,dHead]
    /// transpose passes or buffers — runs Q·Kᵀ → scaled+masked softmax → P·V
    /// per slice, and scatters the result straight into <paramref name="concat"/>
    /// in [B, seq, H, dHead] = [B*seq, dModel] layout (ready for the output
    /// projection, no inverse transpose).
    /// </summary>
    /// <param name="qkv">Fused projection buffer; row r=(b*seq+s) has stride
    /// <paramref name="qkvRowStride"/>; q/k/v blocks start at qCol/kCol/vCol and
    /// within a block the per-head layout is [..., h*dHead + d].</param>
    private unsafe void MultiHeadAttentionFusedSdpa(
        float[] qkv, int qkvRowStride, int qCol, int kCol, int vCol,
        Tensor<bool>? mask, double scaleValue,
        int batch, int heads, int seq, int dHead,
        float[] concat, int dModel)
    {
        int bhCount = batch * heads;
        float scaleF = (float)scaleValue;
        float negInfF = float.NegativeInfinity;
        var pool = System.Buffers.ArrayPool<float>.Shared;

        // Per-slice scratch: Q [seq×dHead] | V [seq×dHead] | Kᵀ [dHead×seq] | scores [seq×seq].
        int qSoff = 0;
        int vSoff = seq * dHead;
        int ktoff = 2 * seq * dHead;
        int scoff = 3 * seq * dHead;      // dHead*seq == seq*dHead
        int scratchLen = 3 * seq * dHead + seq * seq;

        // Parallelize over batch·heads using the SAME PersistentParallelExecutor as
        // the GEMMs/LayerNorm (NOT Parallel.For/ThreadPool) — a second pool in the
        // forward pass oversubscribed the cores and forced park/wakeup between ops.
        // One pool keeps workers hot across QKV-GEMM → SDPA → out-proj. Each chunk
        // rents scratch once and strides over its (b,h) slices.
        int _sdpaChunks = Math.Max(1, Math.Min(bhCount, CpuParallelSettings.MaxDegreeOfParallelism));
        PersistentParallelExecutor.Instance.Execute(_sdpaChunks, chunk =>
        {
            var scratch = pool.Rent(scratchLen);
            try
            {
                for (int bh = chunk; bh < bhCount; bh += _sdpaChunks)
                {
                int b = bh / heads;
                int h = bh % heads;
                int hQ = qCol + h * dHead;
                int hK = kCol + h * dHead;
                int hV = vCol + h * dHead;

                // Gather: Q contiguous [seq×dHead], V contiguous [seq×dHead],
                // K already TRANSPOSED into [dHead×seq] (so the score GEMM is a
                // plain no-transpose A·B). One strided read pass per slice.
                for (int s = 0; s < seq; s++)
                {
                    int rowBase = (b * seq + s) * qkvRowStride;
                    int qsrc = rowBase + hQ, vsrc = rowBase + hV, ksrc = rowBase + hK;
                    int qdst = qSoff + s * dHead, vdst = vSoff + s * dHead;
                    for (int d = 0; d < dHead; d++)
                    {
                        scratch[qdst + d] = qkv[qsrc + d];
                        scratch[vdst + d] = qkv[vsrc + d];
                        scratch[ktoff + d * seq + s] = qkv[ksrc + d];   // Kᵀ
                    }
                }

                // scores[seq×seq] = Q[seq×dHead] · Kᵀ[dHead×seq]
                Engines.Simd.SimdGemm.SgemmSequential(
                    scratch.AsSpan(qSoff, seq * dHead),
                    scratch.AsSpan(ktoff, dHead * seq),
                    scratch.AsSpan(scoff, seq * seq),
                    seq, dHead, seq);

                // In-place scale + numerically-stable softmax, row by row.
                // SIMD fast path: no mask + seq%8==0 (the transformer-encoder case)
                // — vectorized scale/max, SIMD exp (HerumiExp256.Exp8), vectorized
                // sum/normalize. The scalar MathF.Exp loop (524K exp/forward at
                // bs128) was profiled as ~31% of the whole transformer forward.
#if NET5_0_OR_GREATER
                if (mask == null && (seq & 7) == 0 && Avx2.IsSupported && Fma.IsSupported)
                {
                    fixed (float* scPtr = scratch)
                    fixed (float* tbl = HerumiExp256._table)
                    {
                        var vScale = Vector256.Create(scaleF);
                        for (int i = 0; i < seq; i++)
                        {
                            float* row = scPtr + scoff + i * seq;
                            var vmax = Vector256.Create(negInfF);
                            for (int j = 0; j < seq; j += 8)
                            {
                                var v = Avx.Multiply(Avx.LoadVector256(row + j), vScale);
                                Avx.Store(row + j, v);
                                vmax = Avx.Max(vmax, v);
                            }
                            float maxVal = HMax256(vmax);
                            var vMaxB = Vector256.Create(maxVal);
                            var vsum = Vector256<float>.Zero;
                            for (int j = 0; j < seq; j += 8)
                            {
                                var e = HerumiExp256.Exp8(Avx.Subtract(Avx.LoadVector256(row + j), vMaxB), tbl);
                                Avx.Store(row + j, e);
                                vsum = Avx.Add(vsum, e);
                            }
                            float sumExp = HSum256(vsum);
                            float inv = sumExp != 0f ? 1f / sumExp : 0f;
                            var vinv = Vector256.Create(inv);
                            for (int j = 0; j < seq; j += 8)
                                Avx.Store(row + j, Avx.Multiply(Avx.LoadVector256(row + j), vinv));
                        }
                    }
                }
                else
#endif
                {
                    for (int i = 0; i < seq; i++)
                    {
                        int rowOff = scoff + i * seq;
                        float maxVal = negInfF;
                        for (int j = 0; j < seq; j++)
                        {
                            float v = scratch[rowOff + j] * scaleF;
                            if (mask != null && !mask[b, h, i, j]) v = negInfF;
                            if (v > maxVal) maxVal = v;
                            scratch[rowOff + j] = v;
                        }
                        if (float.IsNegativeInfinity(maxVal))
                        {
                            for (int j = 0; j < seq; j++) scratch[rowOff + j] = 0f;
                            continue;
                        }
                        float sumExp = 0f;
                        for (int j = 0; j < seq; j++)
                        {
                            float e = MathF.Exp(scratch[rowOff + j] - maxVal);
                            scratch[rowOff + j] = e;
                            sumExp += e;
                        }
                        float inv = sumExp != 0f ? 1f / sumExp : 0f;
                        for (int j = 0; j < seq; j++) scratch[rowOff + j] *= inv;
                    }
                }

                // out[seq×dHead] = P[seq×seq] · V[seq×dHead]. Reuse the Q region
                // (no longer needed) as the output buffer.
                Engines.Simd.SimdGemm.SgemmSequential(
                    scratch.AsSpan(scoff, seq * seq),
                    scratch.AsSpan(vSoff, seq * dHead),
                    scratch.AsSpan(qSoff, seq * dHead),
                    seq, seq, dHead);

                // Scatter into concat[B, seq, H, dHead] = [B*seq, dModel].
                for (int s = 0; s < seq; s++)
                {
                    int dst = (b * seq + s) * dModel + h * dHead;
                    int src = qSoff + s * dHead;
                    for (int d = 0; d < dHead; d++)
                        concat[dst + d] = scratch[src + d];
                }
                } // end for bh slice in this chunk
            }
            finally
            {
                pool.Return(scratch);
            }
        });
    }

#if NET5_0_OR_GREATER
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe float HSum256(Vector256<float> v)
    {
        float* t = stackalloc float[8]; Avx.Store(t, v);
        return t[0] + t[1] + t[2] + t[3] + t[4] + t[5] + t[6] + t[7];
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe float HMax256(Vector256<float> v)
    {
        float* t = stackalloc float[8]; Avx.Store(t, v);
        float m = t[0];
        for (int i = 1; i < 8; i++) if (t[i] > m) m = t[i];
        return m;
    }
#endif

    /// <summary>
    /// Double fast path (#478 / #1675). Same allocation discipline as the float fast path — fused
    /// Q/K/V projections into pooled scratch, a per-head fused SDPA that reuses ONE pooled scratch
    /// buffer across all heads (never materializing the full [B*H, seq, seq] scores or the four
    /// strided transposes the generic path allocates), and the output projection — but in managed
    /// (no-intrinsic) code so it runs on both TFMs. The per-head math is identical to
    /// <see cref="ScaledDotProductAttention{T}"/>'s double path; only the buffering changes.
    /// </summary>
    private Tensor<double> MultiHeadAttentionForwardDouble(
        Tensor<double> input,
        Tensor<double> qWeight, Tensor<double> kWeight, Tensor<double> vWeight, Tensor<double> outWeight,
        Tensor<bool>? mask,
        int batch, int seqLen, int dModel, int numHeads, int dHead)
    {
        int totalRows = batch * seqLen;
        int proj = totalRows * dModel;
        var ops = MathHelper.GetNumericOperations<double>();
        var pool = ArrayPool<double>.Shared;

        // Pooled projection scratch (returned at end of call) instead of fresh Tensor intermediates.
        var qBuf = pool.Rent(proj);
        var kBuf = pool.Rent(proj);
        var vBuf = pool.Rent(proj);
        var concatBuf = pool.Rent(proj);
        var output = AutoTensorCache.RentOrAllocate<double>(new[] { batch, seqLen, dModel });
        try
        {
            var inArr = input.GetFlattenedData();
            var qW = qWeight.GetFlattenedData();
            var kW = kWeight.GetFlattenedData();
            var vW = vWeight.GetFlattenedData();
            var oW = outWeight.GetFlattenedData();
            var inMem = new ReadOnlyMemory<double>(inArr, 0, proj);

            // Q/K/V = input[totalRows, dModel] @ W[dModel, dModel]. MultiplyBlocked accumulates into C,
            // so the (un-zeroed) pooled destination must be cleared first.
            void Project(double[] w, double[] dst)
            {
                Array.Clear(dst, 0, proj);
                MatrixMultiplyHelper.MultiplyBlocked(ops, inMem,
                    new ReadOnlyMemory<double>(w, 0, dModel * dModel),
                    new Memory<double>(dst, 0, proj),
                    totalRows, dModel, dModel, dModel, dModel, dModel, allowParallel: true);
            }
            Project(qW, qBuf);
            Project(kW, kBuf);
            Project(vW, vBuf);

            MultiHeadAttentionFusedSdpaDouble(qBuf, kBuf, vBuf, dModel, mask,
                1.0 / Math.Sqrt(dHead), batch, numHeads, seqLen, dHead, concatBuf, dModel);

            // Output projection: concat[totalRows, dModel] @ outWeight[dModel, dModel] -> output.
            var outArr = output.GetDataArray();
            Array.Clear(outArr, 0, proj);
            MatrixMultiplyHelper.MultiplyBlocked(ops,
                new ReadOnlyMemory<double>(concatBuf, 0, proj),
                new ReadOnlyMemory<double>(oW, 0, dModel * dModel),
                new Memory<double>(outArr, 0, proj),
                totalRows, dModel, dModel, dModel, dModel, dModel, allowParallel: true);

            return output;
        }
        finally
        {
            pool.Return(qBuf);
            pool.Return(kBuf);
            pool.Return(vBuf);
            pool.Return(concatBuf);
        }
    }

    /// <summary>
    /// Per-head fused SDPA for the double MHA fast path. Mirrors the float
    /// <see cref="MultiHeadAttentionFusedSdpa"/>: each (batch, head) slice gathers Q/V contiguous
    /// and K already-transposed into ONE pooled scratch (reused across the chunk's slices), runs
    /// Q·Kᵀ → scaled+masked stable softmax → P·V, and scatters into <paramref name="concat"/> in
    /// [B, seq, H, dHead] = [B*seq, dModel] layout. No full-scores matrix, no transpose buffers.
    /// </summary>
    private void MultiHeadAttentionFusedSdpaDouble(
        double[] qd, double[] kd, double[] vd, int rowStride,
        Tensor<bool>? mask, double scaleValue,
        int batch, int heads, int seq, int dHead,
        double[] concat, int dModel)
    {
        int bhCount = batch * heads;
        var ops = MathHelper.GetNumericOperations<double>();
        int qSoff = 0;
        int vSoff = seq * dHead;
        int ktoff = 2 * seq * dHead;
        int scoff = 3 * seq * dHead;
        int scratchLen = 3 * seq * dHead + seq * seq;
        var pool = ArrayPool<double>.Shared;
        int chunks = Math.Max(1, Math.Min(bhCount, CpuParallelSettings.MaxDegreeOfParallelism));

        PersistentParallelExecutor.Instance.Execute(chunks, chunk =>
        {
            var scratch = pool.Rent(scratchLen);
            try
            {
                for (int bh = chunk; bh < bhCount; bh += chunks)
                {
                    int b = bh / heads;
                    int h = bh % heads;
                    int headOff = h * dHead;

                    // Gather: Q [seq×dHead], V [seq×dHead] contiguous; K transposed into [dHead×seq].
                    for (int s = 0; s < seq; s++)
                    {
                        int row = (b * seq + s) * rowStride + headOff;
                        int qdst = qSoff + s * dHead;
                        int vdst = vSoff + s * dHead;
                        for (int d = 0; d < dHead; d++)
                        {
                            scratch[qdst + d] = qd[row + d];
                            scratch[vdst + d] = vd[row + d];
                            scratch[ktoff + d * seq + s] = kd[row + d];   // Kᵀ
                        }
                    }

                    // scores[seq×seq] = Q[seq×dHead] · Kᵀ[dHead×seq]. MultiplyBlocked accumulates → clear.
                    var scoresMem = new Memory<double>(scratch, scoff, seq * seq);
                    scoresMem.Span.Clear();
                    MatrixMultiplyHelper.MultiplyBlocked(ops,
                        new ReadOnlyMemory<double>(scratch, qSoff, seq * dHead),
                        new ReadOnlyMemory<double>(scratch, ktoff, dHead * seq),
                        scoresMem,
                        seq, dHead, seq, dHead, seq, seq, allowParallel: false);

                    // In-place scale + numerically-stable softmax, row by row.
                    for (int i = 0; i < seq; i++)
                    {
                        int rowOff = scoff + i * seq;
                        double maxVal = double.NegativeInfinity;
                        for (int j = 0; j < seq; j++)
                        {
                            double v = scratch[rowOff + j] * scaleValue;
                            if (mask != null && !mask[b, h, i, j]) v = double.NegativeInfinity;
                            if (v > maxVal) maxVal = v;
                            scratch[rowOff + j] = v;
                        }
                        if (double.IsNegativeInfinity(maxVal))
                        {
                            for (int j = 0; j < seq; j++) scratch[rowOff + j] = 0d;
                            continue;
                        }
                        double sumExp = 0d;
                        for (int j = 0; j < seq; j++)
                        {
                            double e = Math.Exp(scratch[rowOff + j] - maxVal);
                            scratch[rowOff + j] = e;
                            sumExp += e;
                        }
                        double inv = sumExp != 0d ? 1d / sumExp : 0d;
                        for (int j = 0; j < seq; j++) scratch[rowOff + j] *= inv;
                    }

                    // out[seq×dHead] = P[seq×seq] · V[seq×dHead], reusing the Q region as output.
                    var outMem = new Memory<double>(scratch, qSoff, seq * dHead);
                    outMem.Span.Clear();
                    MatrixMultiplyHelper.MultiplyBlocked(ops,
                        new ReadOnlyMemory<double>(scratch, scoff, seq * seq),
                        new ReadOnlyMemory<double>(scratch, vSoff, seq * dHead),
                        outMem,
                        seq, seq, dHead, seq, dHead, dHead, allowParallel: false);

                    // Scatter into concat[B, seq, H, dHead] = [B*seq, dModel].
                    for (int s = 0; s < seq; s++)
                    {
                        int dst = (b * seq + s) * dModel + headOff;
                        int src = qSoff + s * dHead;
                        for (int d = 0; d < dHead; d++)
                            concat[dst + d] = scratch[src + d];
                    }
                }
            }
            finally
            {
                pool.Return(scratch);
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
