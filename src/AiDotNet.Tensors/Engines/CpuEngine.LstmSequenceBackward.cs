// Copyright (c) AiDotNet. All rights reserved.

using System;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Simd;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using System.Runtime.CompilerServices;
using static AiDotNet.Tensors.Compatibility.MethodImplHelper;

namespace AiDotNet.Tensors.Engines;

public partial class CpuEngine
{
    // ──────────────────────────────────────────────────────────────────────────
    // Fused, tape-aware LSTM sequence forward + BPTT backward (float).
    //
    // The inference LstmSequenceForwardFloat fuses the cell in-register with
    // APPROXIMATE FastSigmoid/FastTanh and saves no per-timestep state — great for
    // inference, useless for backprop. This training variant runs the SAME math with
    // EXACT sigmoid/tanh (PyTorch-equivalent, finite-difference clean), saves the
    // per-timestep gate activations + cell/hidden states, and records ONE fused tape
    // node whose backward does the whole BPTT. That collapses LSTMLayer's per-timestep
    // training graph (8 MatMuls + ~20 elementwise ops PER timestep → hundreds of tape
    // nodes for a length-32 sequence) into a single node, which is the bulk of the
    // ~4× LSTM training gap vs PyTorch's fused kernel (ooples/AiDotNet#1566).
    //
    // Gate layout matches the inference path and PyTorch nn.LSTM: rows of the [4H, *]
    // weights are ordered i (input), f (forget), g (cell candidate), o (output).
    //   c_t = f·c_{t-1} + i·g     h_t = o·tanh(c_t)
    //
    // ┌─ GPU BACKEND COVERAGE (per .coderabbit.yaml kernel-coverage rule) ──────
    // │ This fused training path is currently CPU-ONLY. The six GPU backends
    // │ (CUDA, HIP, Metal, OpenCL, Vulkan, WebGPU) all inherit CpuEngine's
    // │ LstmSequenceForward dispatch — under an active gradient tape they
    // │ FALL THROUGH to LstmSequenceForwardFloatTrain on the CPU.
    // │
    // │ For inference, four of the six backends ship native LSTM kernels
    // │ (CudaLstmKernels, HipLstmKernels, MpsLstm, OpenCL LstmKernels);
    // │ Vulkan / WebGPU LSTM inference is also CPU-routed today.
    // │
    // │ Tracked follow-up: ooples/AiDotNet.Tensors#587 — native fused-training
    // │ kernels on each GPU backend so a GPU model under tape doesn't pay the
    // │ host↔device round-trip for every step. Until those land, GPU users
    // │ training small LSTMs see correct gradients but on the CPU clock; the
    // │ DirectGpu compiled-plan path bypasses this dispatch entirely.
    // └─────────────────────────────────────────────────────────────────────────
    // ──────────────────────────────────────────────────────────────────────────

    /// <summary>
    /// In-place exact sigmoid over <c>buf[off..off+len)</c>: 1/(1+exp(-x)). Uses
    /// <see cref="SimdKernels.ExpUnsafe"/> (near-exact VML/Herumi vectorized exp on AVX, scalar
    /// elsewhere) for the expensive transcendental; the cheap negate/reciprocal stay scalar. The
    /// exp(-x) form is overflow-safe (large |x| → 0 or 1, never NaN). Result matches the scalar
    /// Math.Exp sigmoid to ~1e-6, so the saved gate values — and the σ(1-σ) gradients computed
    /// from them — are unchanged within the finite-difference tolerance.
    /// </summary>
    [MethodImpl(Hot)]
    private static unsafe void SigmoidExactInPlace(float[] buf, int off, int len)
    {
        ScaleInPlace(buf, off, len, -1f);
        fixed (float* p = &buf[off])
            SimdKernels.ExpUnsafe(p, p, len);
        // 1/(1+e): the vector body is the scalar expression lane-wise (same IEEE ops, same order).
        int i = 0;
        if (System.Numerics.Vector.IsHardwareAccelerated)
        {
            int vw = System.Numerics.Vector<float>.Count;
            var one = System.Numerics.Vector<float>.One;
            for (; i <= len - vw; i += vw)
                (one / (one + new System.Numerics.Vector<float>(buf, off + i))).CopyTo(buf, off + i);
        }
        for (; i < len; i++) buf[off + i] = 1f / (1f + buf[off + i]);
    }

    /// <summary>buf[off..off+len) *= s, vectorized (one IEEE multiply per element, as the scalar loop).</summary>
    private static void ScaleInPlace(float[] buf, int off, int len, float s)
    {
        int i = 0;
        if (System.Numerics.Vector.IsHardwareAccelerated)
        {
            int vw = System.Numerics.Vector<float>.Count;
            var vs = new System.Numerics.Vector<float>(s);
            for (; i <= len - vw; i += vw)
                (new System.Numerics.Vector<float>(buf, off + i) * vs).CopyTo(buf, off + i);
        }
        for (; i < len; i++) buf[off + i] *= s;
    }

    /// <summary>
    /// In-place exact tanh over <c>buf[off..off+len)</c> via the overflow-safe identity
    /// tanh(x) = 2/(1+exp(-2x)) - 1 (no e^{2x}, so large x → ±1, never NaN). Same exp seam and
    /// ~1e-6 accuracy as <see cref="SigmoidExactInPlace"/>.
    /// </summary>
    [MethodImpl(Hot)]
    private static unsafe void TanhExactInPlace(float[] buf, int off, int len)
    {
        ScaleInPlace(buf, off, len, -2f);
        fixed (float* p = &buf[off])
            SimdKernels.ExpUnsafe(p, p, len);
        // 2/(1+e) - 1, lane-wise identical to the scalar expression.
        int i = 0;
        if (System.Numerics.Vector.IsHardwareAccelerated)
        {
            int vw = System.Numerics.Vector<float>.Count;
            var one = System.Numerics.Vector<float>.One;
            var two = new System.Numerics.Vector<float>(2f);
            for (; i <= len - vw; i += vw)
                (two / (one + new System.Numerics.Vector<float>(buf, off + i)) - one).CopyTo(buf, off + i);
        }
        for (; i < len; i++) buf[off + i] = 2f / (1f + buf[off + i]) - 1f;
    }

    /// <summary>
    /// Float training forward: exact activations, saves per-timestep state, and records
    /// a single fused BPTT node on the active tape. Called from LstmSequenceForward when
    /// a gradient tape is active and T == float.
    /// </summary>
    [MethodImpl(Hot)]
    private Tensor<float> LstmSequenceForwardFloatTrain(
        Tensor<float> input, Tensor<float>? h0, Tensor<float>? c0,
        Tensor<float> wIh, Tensor<float> wHh, Tensor<float>? bIh, Tensor<float>? bHh,
        int batch, int seqLen, int inFeatures, int hidden, int gateRows,
        bool returnSequences, bool wantState,
        out Tensor<float> finalHidden, out Tensor<float> finalCell)
    {
        // The fused BPTT node records gradients only for `output`; `finalHidden`
        // / `finalCell` are returned as fresh detached tensors. Letting a tape
        // see them would silently stop gradients at the LSTM boundary, so
        // explicitly reject the (wantState ∧ active-tape) combination. Callers
        // that want differentiable state at chunk boundaries should slice them
        // out of `output` (or call the unfused path that records each timestep).
        if (wantState && DifferentiableOps.ThreadTapeActive())
        {
            throw new ArgumentException(
                "LstmSequenceForwardFloatTrain: wantState=true is not yet supported under an active gradient tape. " +
                "The fused BPTT node records gradients only for the sequence output; the returned finalHidden / " +
                "finalCell tensors carry no backward edge and would silently detach the graph. " +
                "Either set wantState=false, or slice the final hidden/cell out of the returned output tensor.",
                nameof(wantState));
        }

        int G = gateRows;                 // 4 * hidden
        int totalRows = batch * seqLen;

        // A fused op may receive stride-only views from a preceding permute.
        // GetFlattenedData is zero-copy for packed tensors and materializes only
        // a view, while the tape continues to record the original tensor identity.
        var inArr = input.GetFlattenedData(); // [batch, seqLen, inFeatures]
        var wIhArr = wIh.GetFlattenedData();  // [G, inFeatures]
        var wHhArr = wHh.GetFlattenedData();  // [G, hidden]
        float[]? bIhArr = bIh?.GetFlattenedData();
        float[]? bHhArr = bHh?.GetFlattenedData();
        float[]? h0Arr = h0?.GetFlattenedData();
        float[]? c0Arr = c0?.GetFlattenedData();

        // Saved state + scratch (the saved arrays are captured by the backward closure and persist past this call);
        // layouts on LstmTrainWorkspace.
        var ws = new LstmTrainWorkspace(batch, seqLen, inFeatures, hidden);
        var cells = ws.Cells;
        var hiddens = ws.Hiddens;

        var output = returnSequences
            ? new Tensor<float>(new[] { batch, seqLen, hidden })
            : new Tensor<float>(new[] { batch, hidden });
        var outSpan = output.AsWritableSpan();

        // seqLen == 0 corner case: the timestep loop below is skipped, so the
        // last-hidden output stays at its default zero-init. Match the generic
        // (unfused) LSTM implementation, which returns "last hidden = h0" when
        // there are no timesteps: copy the seeded h_0 into the [batch, hidden]
        // output here (returnSequences=true yields an empty [batch, 0, hidden]
        // tensor and needs no fill).
        if (seqLen == 0)
        {
            if (!returnSequences)
            {
                for (int b = 0; b < batch; b++)
                    for (int h = 0; h < hidden; h++)
                        outSpan[b * hidden + h] = h0Arr is null ? 0f : h0Arr[b * hidden + h];
            }

            if (wantState)
            {
                finalHidden = new Tensor<float>(new[] { batch, hidden });
                finalCell = new Tensor<float>(new[] { batch, hidden });
                var fhSpan0 = finalHidden.AsWritableSpan();
                var fcSpan0 = finalCell.AsWritableSpan();
                for (int b = 0; b < batch; b++)
                    for (int h = 0; h < hidden; h++)
                    {
                        fhSpan0[b * hidden + h] = h0Arr is null ? 0f : h0Arr[b * hidden + h];
                        fcSpan0[b * hidden + h] = c0Arr is null ? 0f : c0Arr[b * hidden + h];
                    }
            }
            else
            {
                finalHidden = s_emptyState;
                finalCell = s_emptyState;
            }

            // No timesteps means no gates were computed — there's nothing for
            // BPTT to consume, so don't record a tape node. Callers that pass
            // a gradient of `output` against h0 still get correct semantics
            // because the framework handles "no recorded op" as a no-op edge.
            return output;
        }

        LstmTrainForwardCoreFloat(inArr, wIhArr, wHhArr, bIhArr, bHhArr, h0Arr, c0Arr,
            returnSequences, ws, outSpan);

        // Final-state outs (last timestep). The fused node owns gradients via the
        // returned `output`; final states are passthrough (not differentiated here).
        if (wantState)
        {
            finalHidden = new Tensor<float>(new[] { batch, hidden });
            finalCell = new Tensor<float>(new[] { batch, hidden });
            var fhSpan = finalHidden.AsWritableSpan();
            var fcSpan = finalCell.AsWritableSpan();
            for (int b = 0; b < batch; b++)
                for (int h = 0; h < hidden; h++)
                {
                    fhSpan[b * hidden + h] = hiddens[(b * (seqLen + 1) + seqLen) * hidden + h];
                    fcSpan[b * hidden + h] = cells[(b * (seqLen + 1) + seqLen) * hidden + h];
                }
        }
        else
        {
            finalHidden = s_emptyState;
            finalCell = s_emptyState;
        }

        // Build the differentiable-input array (only the tensors we return grads for).
        // Order is fixed: input, wIh, wHh, [bIh], [bHh], [h0], [c0].
        int nInputs = 3 + (bIh is not null ? 1 : 0) + (bHh is not null ? 1 : 0)
                        + (h0 is not null ? 1 : 0) + (c0 is not null ? 1 : 0);
        var inputsArr = new Tensor<float>[nInputs];
        int idx = 0;
        inputsArr[idx++] = input;
        inputsArr[idx++] = wIh;
        inputsArr[idx++] = wHh;
        int idxBIh = bIh is not null ? idx : -1; if (bIh is not null) inputsArr[idx++] = bIh;
        int idxBHh = bHh is not null ? idx : -1; if (bHh is not null) inputsArr[idx++] = bHh;
        int idxH0 = h0 is not null ? idx : -1; if (h0 is not null) inputsArr[idx++] = h0;
        int idxC0 = c0 is not null ? idx : -1; if (c0 is not null) inputsArr[idx++] = c0;

        // meta carries dims, the returnSequences flag, and the optional-input indices.
        var meta = new int[] { batch, seqLen, inFeatures, hidden, returnSequences ? 1 : 0,
                               idxBIh, idxBHh, idxH0, idxC0 };
        var savedState = new object[] { ws.Gates, cells, hiddens, meta, ws.TanhC, ws };

        DifferentiableOps.RecordIfActive<float>(
            "LstmSequenceForward", output, inputsArr, LstmSequenceBackwardFloat, savedState);

        return output;
    }

    // ── Batch-chunked fused LSTM training kernel ─────────────────────────────────────────────────────────────────────
    // A sample's recurrence never reads another sample's state: h_t of batch row b depends only on h_{t-1} of row b.
    // So the forward and the BPTT backward split the BATCH into fixed chunks and run each chunk's WHOLE sequence on one
    // worker -- one parallel region per pass, where the previous kernel ran every per-step recurrent GEMM and cell on
    // the calling thread and forked/joined the pool for each big GEMM. Inside a chunk every GEMM is a single-threaded
    // direct-kernel call (no packing, no dispatch): the chunk's input projection, its per-step recurrent GEMM, the
    // backward's dh carry, its input gradient and its PARTIAL weight/bias gradients, which are summed in chunk order
    // afterwards. The chunk layout is a function of the batch size alone, never of the core count, so the results are
    // bit-identical for every degree of parallelism (the serial fallback included). Stacked, bidirectional and
    // sequence-to-sequence LSTMs all reach this kernel through LstmSequenceForward.

    /// <summary>Minimum batch rows per chunk. The direct GEMM kernel's register tile is 6 rows (SimdGemm Mr): a narrower
    /// chunk would drop its per-step GEMMs off the direct path.</summary>
    private const int LstmMinChunkRows = 8;

    /// <summary>Upper bound on the chunk count, so a large batch keeps each chunk's per-step GEMM wide enough to pay for
    /// the call.</summary>
    private const int LstmMaxChunks = 16;

    private static int LstmChunkCount(int batch) => Math.Max(1, Math.Min(batch / LstmMinChunkRows, LstmMaxChunks));

    /// <summary>First batch row of chunk <paramref name="c"/>; rows are split as evenly as integer division allows.</summary>
    private static int LstmChunkStart(int c, int batch, int chunks) => (int)((long)c * batch / chunks);

    /// <summary>
    /// C := A·B ([m, k]·[k, n], row-major, no transposes) on the parallel BlasManaged dispatcher. Used for the large
    /// whole-sequence GEMMs when the batch is a single chunk: nothing else runs beside it then, so the GEMM itself must
    /// spread over the pool, as the kernel did before batch chunking (a single-threaded GEMM there made small-batch
    /// LSTMs 18-44% slower).
    /// </summary>
    private static void LstmSoloGemm(ReadOnlySpan<float> a, int lda, ReadOnlySpan<float> b, int ldb, Span<float> c, int m, int n, int k,
        bool transA = false)
        => BlasManaged.BlasManaged.Gemm<float>(a, lda, transA, b, ldb, false, c, n, m, n, k,
            new BlasManaged.BlasOptions<float> { PackingMode = BlasManaged.PackingMode.DisableAutotune });

    /// <summary>
    /// The [G, k] weight gradient from the chunks' partials. A single chunk computes its partial directly in the [G, k]
    /// layout (a transposed-A GEMM, no operand or result transposes), so it is copied as is; several chunks keep the
    /// [k, G] partials, summed in chunk order and transposed once.
    /// </summary>
    private static Tensor<float> LstmWeightGradient(float[][] parts, int chunks, int k, int G)
    {
        if (chunks != 1) return SumChunkPartialsTransposed(parts, chunks, k, G);
        var grad = new Tensor<float>(new[] { G, k });
        parts[0].AsSpan(0, G * k).CopyTo(grad.AsWritableSpan());
        return grad;
    }

    /// <summary>
    /// The fused training forward's arithmetic: forms the transposed weights and the summed bias once, then runs every
    /// batch chunk's input projection and recurrence (<see cref="LstmForwardChunk"/>), writing the saved per-timestep
    /// gates/cells/hiddens/tanh(c) into <paramref name="ws"/> and the hidden outputs into <paramref name="outSpan"/>
    /// (every position of it). Allocation-free apart from the dispatch closure: the eager tape path hands it a fresh
    /// workspace per call, a compiled-plan node one workspace it reuses every step. Requires seqLen &gt; 0.
    /// </summary>
    private static unsafe void LstmTrainForwardCoreFloat(
        float[] inArr, float[] wIhArr, float[] wHhArr,
        float[]? bIhArr, float[]? bHhArr, float[]? h0Arr, float[]? c0Arr,
        bool returnSequences, LstmTrainWorkspace ws, Span<float> outSpan)
    {
        int batch = ws.Batch, seqLen = ws.SeqLen, inFeatures = ws.InFeatures, hidden = ws.Hidden;
        int G = 4 * hidden;

        // Operands every chunk shares (read-only inside the region): WIhᵀ [in, G], WHhᵀ [hidden, G] and bIh + bHh.
        var wIhT = ws.WIhT;
        for (int g = 0; g < G; g++)
            for (int i = 0; i < inFeatures; i++)
                wIhT[i * G + g] = wIhArr[g * inFeatures + i];
        var wHhT = ws.WHhT;
        for (int g = 0; g < G; g++)
            for (int h = 0; h < hidden; h++)
                wHhT[h * G + g] = wHhArr[g * hidden + h];
        var bias = ws.Bias;
        for (int g = 0; g < G; g++)
            bias[g] = (bIhArr is null ? 0f : bIhArr[g]) + (bHhArr is null ? 0f : bHhArr[g]);

        int outLen = outSpan.Length;
        fixed (float* pOut = outSpan)
        {
            // The region is synchronous, so the pinned output outlives every chunk's use of it.
            IntPtr outPtr = (IntPtr)pOut;
            int chunks = ws.Chunks;
            if (chunks == 1)
            {
                LstmForwardChunk(0, inArr, h0Arr, c0Arr, returnSequences, ws, outPtr, outLen);
            }
            else
            {
                long work = (long)batch * seqLen * G * (inFeatures + hidden);
                CpuParallelSettings.ParallelForOrSerial(0, chunks, work,
                    c => LstmForwardChunk(c, inArr, h0Arr, c0Arr, returnSequences, ws, outPtr, outLen),
                    deterministicSafe: true);
            }
        }
    }

    /// <summary>
    /// One chunk's forward: seeds c_0/h_0, forms Wx = (bIh + bHh) + x·WIhᵀ over the chunk's (b, t) rows with one GEMM,
    /// then per timestep accumulates h_{t-1}·WHhᵀ onto those rows IN the time-major saved gates, applies the exact
    /// activations and the cell, and writes c_t, tanh(c_t), h_t and the output rows.
    /// </summary>
    private static unsafe void LstmForwardChunk(int c, float[] inArr, float[]? h0Arr, float[]? c0Arr,
        bool returnSequences, LstmTrainWorkspace ws, IntPtr outPtr, int outLen)
    {
        int batch = ws.Batch, seqLen = ws.SeqLen, inFeatures = ws.InFeatures, hidden = ws.Hidden;
        int G = 4 * hidden;
        int b0 = LstmChunkStart(c, batch, ws.Chunks), b1 = LstmChunkStart(c + 1, batch, ws.Chunks), nb = b1 - b0;
        if (nb == 0) return;
        var outSpan = new Span<float>((void*)outPtr, outLen);
        var gates = ws.Gates;
        var cells = ws.Cells;
        var hiddens = ws.Hiddens;
        var tanhC = ws.TanhC;
        var wx = ws.Wx;
        int hStride = (seqLen + 1) * hidden;

        // c_0 / h_0.
        for (int b = b0; b < b1; b++)
        {
            int baseTc = b * hStride;
            if (c0Arr is null) Array.Clear(cells, baseTc, hidden);
            else Array.Copy(c0Arr, b * hidden, cells, baseTc, hidden);
            if (h0Arr is null) Array.Clear(hiddens, baseTc, hidden);
            else Array.Copy(h0Arr, b * hidden, hiddens, baseTc, hidden);
        }

        // Wx[b, t] = (bIh + bHh) + x[b, t]·WIhᵀ for the chunk's rows: the bias seeds the accumulating GEMM.
        int rows = nb * seqLen;
        int wxOff = b0 * seqLen * G;
        // A batch below two chunks' worth runs as ONE chunk on the calling thread with nothing beside it, so its GEMMs
        // may use the pool themselves (as before chunking); with several chunks every GEMM stays single-threaded inside
        // its worker. SgemmAddInternal's own size gate still decides whether a given GEMM actually parallelizes.
        bool soloChunk = ws.Chunks == 1;
        if (soloChunk)
        {
            LstmSoloGemm(inArr.AsSpan(b0 * seqLen * inFeatures, rows * inFeatures), inFeatures,
                ws.WIhT.AsSpan(0, inFeatures * G), G, wx.AsSpan(wxOff, rows * G), rows, G, inFeatures);
            var bias = ws.Bias;
            for (int r = 0; r < rows; r++)
            {
                int off = wxOff + r * G;
                for (int g = 0; g < G; g++) wx[off + g] += bias[g];
            }
        }
        else
        {
            for (int r = 0; r < rows; r++)
                Array.Copy(ws.Bias, 0, wx, wxOff + r * G, G);
            SimdGemm.SgemmAddInternal(inArr.AsSpan(b0 * seqLen * inFeatures, rows * inFeatures), inFeatures, false,
                ws.WIhT.AsSpan(0, inFeatures * G), G, false,
                wx.AsSpan(wxOff, rows * G), rows, inFeatures, G, allowParallel: false);
        }

        for (int t = 0; t < seqLen; t++)
        {
            // The chunk's rows at step t are contiguous in the time-major gates: seed them with Wx and accumulate
            // h_{t-1}·WHhᵀ in place (h_{t-1} rows are read straight out of the saved hiddens at stride hStride).
            int gBase = (t * batch + b0) * G;
            for (int b = b0; b < b1; b++)
                Array.Copy(wx, (b * seqLen + t) * G, gates, gBase + (b - b0) * G, G);
            SimdGemm.SgemmAddInternal(hiddens.AsSpan(b0 * hStride + t * hidden), hStride, false,
                ws.WHhT.AsSpan(0, hidden * G), G, false,
                gates.AsSpan(gBase, nb * G), nb, hidden, G, allowParallel: false);

            for (int b = b0; b < b1; b++)
            {
                int row = gBase + (b - b0) * G;
                // Exact activations, gate order i, f, g (cell candidate), o.
                SigmoidExactInPlace(gates, row, 2 * hidden);
                TanhExactInPlace(gates, row + 2 * hidden, hidden);
                SigmoidExactInPlace(gates, row + 3 * hidden, hidden);

                // c = f·c_prev + i·g into cells and tanhC; tanhC = tanh(c); h = o·tanh(c).
                int cPrevRow = b * hStride + t * hidden;
                int cCurRow = cPrevRow + hidden;
                int tc = (b * seqLen + t) * hidden;
                CellUpdateRow(gates, row, row + hidden, row + 2 * hidden, cells, cPrevRow, cCurRow, tanhC, tc, hidden);
                TanhExactInPlace(tanhC, tc, hidden);
                HiddenOutRow(gates, row + 3 * hidden, tanhC, tc, hiddens, cCurRow, hidden);

                if (returnSequences)
                    hiddens.AsSpan(cCurRow, hidden).CopyTo(outSpan.Slice((b * seqLen + t) * hidden, hidden));
                else if (t == seqLen - 1)
                    hiddens.AsSpan(cCurRow, hidden).CopyTo(outSpan.Slice(b * hidden, hidden));
            }
        }
    }

    // ── Fused LSTM cell rows, vectorized with System.Numerics.Vector<float> over the hidden axis. Every vector
    //    expression keeps the scalar form's operand order (no FMA contraction), and the scalar tails are that form,
    //    so results do not depend on the vector width. ──────────────────────────────────────────────────────────────

    /// <summary>c = f·c_prev + i·g into cells[cCurRow..] and cbuf[cb..] (gate rows of act at iOff/fOff/gOff).</summary>
    private static void CellUpdateRow(float[] act, int iOff, int fOff, int gOff, float[] cells, int cPrevRow, int cCurRow,
        float[] cbuf, int cb, int len)
    {
        int h = 0;
        if (System.Numerics.Vector.IsHardwareAccelerated)
        {
            int vw = System.Numerics.Vector<float>.Count;
            for (; h <= len - vw; h += vw)
            {
                var c = new System.Numerics.Vector<float>(act, fOff + h) * new System.Numerics.Vector<float>(cells, cPrevRow + h)
                      + new System.Numerics.Vector<float>(act, iOff + h) * new System.Numerics.Vector<float>(act, gOff + h);
                c.CopyTo(cells, cCurRow + h);
                c.CopyTo(cbuf, cb + h);
            }
        }
        for (; h < len; h++)
        {
            float c = act[fOff + h] * cells[cPrevRow + h] + act[iOff + h] * act[gOff + h];
            cells[cCurRow + h] = c;
            cbuf[cb + h] = c;
        }
    }

    /// <summary>h = o·tanh(c) into hiddens[hRow..] (o at act[oOff..], tanh(c) at tanhC[tc..]).</summary>
    private static void HiddenOutRow(float[] act, int oOff, float[] tanhC, int tc, float[] hiddens, int hRow, int len)
    {
        int h = 0;
        if (System.Numerics.Vector.IsHardwareAccelerated)
        {
            int vw = System.Numerics.Vector<float>.Count;
            for (; h <= len - vw; h += vw)
                (new System.Numerics.Vector<float>(act, oOff + h) * new System.Numerics.Vector<float>(tanhC, tc + h))
                    .CopyTo(hiddens, hRow + h);
        }
        for (; h < len; h++) hiddens[hRow + h] = act[oOff + h] * tanhC[tc + h];
    }

    /// <summary>
    /// One BPTT row: from the saved gates (i|f|g|o at gateRow), c_{t-1}, the forward's tanh(c_t) and the carried
    /// dh/dc (plus this step's upstream gradient when <paramref name="gradOut"/> is non-null), writes the four
    /// pre-activation gate gradients to dgatesT[dgtRow..] and dgatesAll[dgaRow..] and carries dc to t-1.
    /// </summary>
    private static void BpttCellRow(float[] gates, int gateRow, float[] cells, int cPrevRow, float[] tanhC, int tcRow,
        float[] dhNext, float[] dcNext, int carryRow, float[]? gradOut, int goRow,
        float[] dgatesT, int dgtRow, float[] dgatesAll, int dgaRow, int hidden)
    {
        int h = 0;
        if (System.Numerics.Vector.IsHardwareAccelerated)
        {
            int vw = System.Numerics.Vector<float>.Count;
            var one = System.Numerics.Vector<float>.One;
            for (; h <= hidden - vw; h += vw)
            {
                var ig = new System.Numerics.Vector<float>(gates, gateRow + h);
                var fg = new System.Numerics.Vector<float>(gates, gateRow + hidden + h);
                var gg = new System.Numerics.Vector<float>(gates, gateRow + 2 * hidden + h);
                var og = new System.Numerics.Vector<float>(gates, gateRow + 3 * hidden + h);
                var cPrev = new System.Numerics.Vector<float>(cells, cPrevRow + h);
                var tc = new System.Numerics.Vector<float>(tanhC, tcRow + h);
                var dhTotal = new System.Numerics.Vector<float>(dhNext, carryRow + h)
                    + (gradOut is null ? System.Numerics.Vector<float>.Zero : new System.Numerics.Vector<float>(gradOut, goRow + h));

                var doPre = dhTotal * tc * og * (one - og);                                   // sigmoid'
                var dcTotal = new System.Numerics.Vector<float>(dcNext, carryRow + h) + dhTotal * og * (one - tc * tc); // tanh'
                var diPre = dcTotal * gg * ig * (one - ig);
                var dgPre = dcTotal * ig * (one - gg * gg);
                var dfPre = dcTotal * cPrev * fg * (one - fg);
                (dcTotal * fg).CopyTo(dcNext, carryRow + h);                                    // dc for t-1

                diPre.CopyTo(dgatesT, dgtRow + h);
                dfPre.CopyTo(dgatesT, dgtRow + hidden + h);
                dgPre.CopyTo(dgatesT, dgtRow + 2 * hidden + h);
                doPre.CopyTo(dgatesT, dgtRow + 3 * hidden + h);
                diPre.CopyTo(dgatesAll, dgaRow + h);
                dfPre.CopyTo(dgatesAll, dgaRow + hidden + h);
                dgPre.CopyTo(dgatesAll, dgaRow + 2 * hidden + h);
                doPre.CopyTo(dgatesAll, dgaRow + 3 * hidden + h);
            }
        }
        for (; h < hidden; h++)
        {
            float ig = gates[gateRow + 0 * hidden + h];
            float fg = gates[gateRow + 1 * hidden + h];
            float gg = gates[gateRow + 2 * hidden + h];
            float og = gates[gateRow + 3 * hidden + h];
            float cPrev = cells[cPrevRow + h];
            float tc = tanhC[tcRow + h];
            float dhTotal = dhNext[carryRow + h] + (gradOut is null ? 0f : gradOut[goRow + h]);

            // h = o · tanh(c)
            float doPre = dhTotal * tc * og * (1f - og);                          // sigmoid'
            float dcTotal = dcNext[carryRow + h] + dhTotal * og * (1f - tc * tc);   // tanh'

            // c = f · c_prev + i · g
            float diPre = dcTotal * gg * ig * (1f - ig);
            float dgPre = dcTotal * ig * (1f - gg * gg);
            float dfPre = dcTotal * cPrev * fg * (1f - fg);
            dcNext[carryRow + h] = dcTotal * fg;                                   // dc for t-1

            dgatesT[dgtRow + 0 * hidden + h] = diPre;
            dgatesT[dgtRow + 1 * hidden + h] = dfPre;
            dgatesT[dgtRow + 2 * hidden + h] = dgPre;
            dgatesT[dgtRow + 3 * hidden + h] = doPre;

            dgatesAll[dgaRow + 0 * hidden + h] = diPre;
            dgatesAll[dgaRow + 1 * hidden + h] = dfPre;
            dgatesAll[dgaRow + 2 * hidden + h] = dgPre;
            dgatesAll[dgaRow + 3 * hidden + h] = doPre;
        }
    }

    /// <summary>sum[0..len) += part[0..len), vectorized; per element the same single add as the scalar loop.</summary>
    private static void LstmAddInto(float[] sum, float[] part, int len)
    {
        int i = 0;
        if (System.Numerics.Vector.IsHardwareAccelerated)
        {
            int vw = System.Numerics.Vector<float>.Count;
            for (; i <= len - vw; i += vw)
                (new System.Numerics.Vector<float>(sum, i) + new System.Numerics.Vector<float>(part, i)).CopyTo(sum, i);
        }
        for (; i < len; i++) sum[i] += part[i];
    }

    /// <summary>
    /// Saved state and scratch of one fused LSTM training forward (G = 4 * hidden). Gates/Cells/Hiddens/TanhC are what
    /// the BPTT backward reads:
    ///   Gates:   [t, b] post-activation i|f|g|o, row (t*batch+b)*G  (TIME-major: a chunk's rows at one step are
    ///            contiguous, so the per-step recurrent GEMM accumulates straight into them)
    ///   Cells:   [b, tc] c_0..c_seqLen, row (b*(seqLen+1)+tc)*hidden  (tc=0 is c0)
    ///   Hiddens: [b, tc] h_0..h_seqLen, row (b*(seqLen+1)+tc)*hidden  (tc=0 is h0)
    ///   TanhC:   [b, t] tanh(c_t) exactly as the forward used it for h_t, row (b*seqLen+t)*hidden
    ///   Wx:      [b, t] forward: bIh + bHh + x·WIhᵀ, row (b*seqLen+t)*G. The backward reuses it as dgatesAll, the
    ///            pre-activation gate gradients in the same row layout (the forward is done with it by then).
    /// The per-chunk backward scratch (<see cref="LstmBackwardScratch"/>) is allocated on the first backward, so a
    /// forward-only use never pays for it.
    /// </summary>
    private sealed class LstmTrainWorkspace
    {
        public LstmTrainWorkspace(int batch, int seqLen, int inFeatures, int hidden)
        {
            Batch = batch;
            SeqLen = seqLen;
            InFeatures = inFeatures;
            Hidden = hidden;
            int g = 4 * hidden;
            Gates = new float[seqLen * batch * g];
            Cells = new float[batch * (seqLen + 1) * hidden];
            Hiddens = new float[batch * (seqLen + 1) * hidden];
            TanhC = new float[batch * seqLen * hidden];
            Wx = new float[batch * seqLen * g];
            WIhT = new float[inFeatures * g];
            WHhT = new float[hidden * g];
            Bias = new float[g];
            Chunks = LstmChunkCount(batch);
            MaxChunkRows = (batch + Chunks - 1) / Chunks;
        }

        public int Batch { get; }
        public int SeqLen { get; }
        public int InFeatures { get; }
        public int Hidden { get; }
        public int Chunks { get; }
        public int MaxChunkRows { get; }
        public float[] Gates { get; }
        public float[] Cells { get; }
        public float[] Hiddens { get; }
        public float[] TanhC { get; }
        public float[] Wx { get; }
        public float[] WIhT { get; }
        public float[] WHhT { get; }
        public float[] Bias { get; }

        /// <summary>The per-chunk backward scratch, allocated on the first backward and reused after it.</summary>
        public LstmBackwardScratch GetBackwardScratch()
            => _backward ??= new LstmBackwardScratch(Chunks, MaxChunkRows, SeqLen, InFeatures, Hidden);

        private LstmBackwardScratch? _backward;
    }

    /// <summary>
    /// Per-chunk backward scratch of a <see cref="LstmTrainWorkspace"/> (G = 4 * hidden, R = rows * seqLen): the
    /// current step's dgates [rows, G], the dh/dc carries [rows, hidden], the transposed chunk input [in, R] and
    /// previous hiddens [hidden, R], and the chunk's partial WIhᵀ [in, G], WHhᵀ [hidden, G] and bias [G] gradients.
    /// </summary>
    private sealed class LstmBackwardScratch
    {
        public LstmBackwardScratch(int chunks, int rows, int seqLen, int inFeatures, int hidden)
        {
            int g = 4 * hidden, seqRows = rows * seqLen;
            DGatesT = new float[chunks][];
            DH = new float[chunks][];
            DC = new float[chunks][];
            XT = new float[chunks][];
            HT = new float[chunks][];
            PartWIh = new float[chunks][];
            PartWHh = new float[chunks][];
            PartB = new float[chunks][];
            for (int c = 0; c < chunks; c++)
            {
                DGatesT[c] = new float[rows * g];
                DH[c] = new float[rows * hidden];
                DC[c] = new float[rows * hidden];
                XT[c] = new float[inFeatures * seqRows];
                HT[c] = new float[hidden * seqRows];
                PartWIh[c] = new float[inFeatures * g];
                PartWHh[c] = new float[hidden * g];
                PartB[c] = new float[g];
            }
        }

        public float[][] DGatesT { get; }
        public float[][] DH { get; }
        public float[][] DC { get; }
        public float[][] XT { get; }
        public float[][] HT { get; }
        public float[][] PartWIh { get; }
        public float[][] PartWHh { get; }
        public float[][] PartB { get; }
    }

    /// <summary>
    /// Compiled-graph form of the fused training LSTM: one lazy node whose forward runs
    /// <see cref="LstmTrainForwardCoreFloat"/> straight into the node's output buffer and whose backward is
    /// <see cref="LstmSequenceBackwardFloat"/>. The node owns one <see cref="LstmTrainWorkspace"/> for its lifetime:
    /// each replayed forward overwrites the saved gates/cells/hiddens that the same step's backward then reads, so a
    /// step allocates no saved state. Inputs and saved-state layout are exactly the eager tape node's.
    /// </summary>
    private static Tensor<float> RecordLstmSequenceTrainFloat(
        Compilation.LazyTensorScope scope,
        Tensor<float> input, Tensor<float>? h0, Tensor<float>? c0,
        Tensor<float> wIh, Tensor<float> wHh, Tensor<float>? bIh, Tensor<float>? bHh,
        int batch, int seqLen, int inFeatures, int hidden, bool returnSequences)
    {
        // Order is fixed (the backward indexes it through meta): input, wIh, wHh, [bIh], [bHh], [h0], [c0].
        int nInputs = 3 + (bIh is not null ? 1 : 0) + (bHh is not null ? 1 : 0)
                        + (h0 is not null ? 1 : 0) + (c0 is not null ? 1 : 0);
        var inputsArr = new Tensor<float>[nInputs];
        int idx = 0;
        inputsArr[idx++] = input;
        inputsArr[idx++] = wIh;
        inputsArr[idx++] = wHh;
        int idxBIh = bIh is not null ? idx : -1; if (bIh is not null) inputsArr[idx++] = bIh;
        int idxBHh = bHh is not null ? idx : -1; if (bHh is not null) inputsArr[idx++] = bHh;
        int idxH0 = h0 is not null ? idx : -1; if (h0 is not null) inputsArr[idx++] = h0;
        int idxC0 = c0 is not null ? idx : -1; if (c0 is not null) inputsArr[idx++] = c0;
        var meta = new int[] { batch, seqLen, inFeatures, hidden, returnSequences ? 1 : 0,
                               idxBIh, idxBHh, idxH0, idxC0 };

        var ws = new LstmTrainWorkspace(batch, seqLen, inFeatures, hidden);
        var savedState = new object[] { ws.Gates, ws.Cells, ws.Hiddens, meta, ws.TanhC, ws };
        var outputShape = returnSequences ? new[] { batch, seqLen, hidden } : new[] { batch, hidden };
        int outputLength = returnSequences ? batch * seqLen * hidden : batch * hidden;

        return scope.RecordVariadic(Compilation.LazyNodeType.Custom, "LstmSequenceTrain", inputsArr, outputShape,
            (eng, output) =>
            {
                var inArr = input.GetFlattenedData();
                var wIhArr = wIh.GetFlattenedData();
                var wHhArr = wHh.GetFlattenedData();
                float[]? bIhArr = bIh?.GetFlattenedData();
                float[]? bHhArr = bHh?.GetFlattenedData();
                float[]? h0Arr = h0?.GetFlattenedData();
                float[]? c0Arr = c0?.GetFlattenedData();
                if (output.IsContiguous && output._gpuBuffer is null && !output.HasPendingGpuData)
                {
                    // The core writes every output position, so the uninitialized plan buffer needs no clear.
                    LstmTrainForwardCoreFloat(inArr, wIhArr, wHhArr, bIhArr, bHhArr, h0Arr, c0Arr,
                        returnSequences, ws, output.AsWritableSpan());
                    output.IncrementVersion();
                }
                else
                {
                    var staged = new Tensor<float>(outputShape);
                    LstmTrainForwardCoreFloat(inArr, wIhArr, wHhArr, bIhArr, bHhArr, h0Arr, c0Arr,
                        returnSequences, ws, staged.AsWritableSpan().Slice(0, outputLength));
                    DirectGpuTensorEngine.CopyResultInto(eng, staged, output);
                }
            },
            LstmSequenceBackwardFloat, savedState);
    }

    /// <summary>
    /// BPTT backward for the fused LSTM. Accumulates gradients for input, wIh, wHh and (when present) bIh, bHh, h0, c0
    /// from the saved per-timestep state, each batch chunk's recurrence on its own worker
    /// (<see cref="LstmBackwardChunk"/>); the chunks' partial weight and bias gradients are then summed in chunk order.
    /// A gradient the active relevance filter says nothing reads (the data input's, typically) is not computed. Runs
    /// with tape recording suppressed (standard backward context), so the internal GEMMs are plain compute.
    /// </summary>
    [MethodImpl(Hot)]
    private static unsafe void LstmSequenceBackwardFloat(
        Tensor<float> gradOutput, Tensor<float>[] inp, Tensor<float> output,
        object[] savedState, IEngine engine, System.Collections.Generic.Dictionary<Tensor<float>, Tensor<float>> grads)
    {
        var meta = (int[])savedState[3];
        var ws = (LstmTrainWorkspace)savedState[5];
        int batch = meta[0], seqLen = meta[1], inFeatures = meta[2], hidden = meta[3];
        bool returnSequences = meta[4] != 0;
        int idxBIh = meta[5], idxBHh = meta[6], idxH0 = meta[7], idxC0 = meta[8];
        int G = 4 * hidden;

        var input = inp[0];
        var wIh = inp[1];
        var wHh = inp[2];
        bool needInput = DifferentiableOps.IsGradientRequired(input);
        bool needWIh = DifferentiableOps.IsGradientRequired(wIh);
        bool needWHh = DifferentiableOps.IsGradientRequired(wHh);
        bool needBIh = idxBIh >= 0 && DifferentiableOps.IsGradientRequired(inp[idxBIh]);
        bool needBHh = idxBHh >= 0 && DifferentiableOps.IsGradientRequired(inp[idxBHh]);
        bool needH0 = idxH0 >= 0 && DifferentiableOps.IsGradientRequired(inp[idxH0]);
        bool needC0 = idxC0 >= 0 && DifferentiableOps.IsGradientRequired(inp[idxC0]);
        bool needBias = needBIh || needBHh;

        var wHhArr = wHh.GetFlattenedData();                         // [G, hidden]
        float[]? wIhArr = needInput ? wIh.GetFlattenedData() : null; // [G, inFeatures]
        float[]? inputArr = needWIh ? input.GetFlattenedData() : null;
        var gradOutArr = gradOutput.GetFlattenedData();

        var scratch = ws.GetBackwardScratch();
        var gradInput = needInput ? new Tensor<float>(new[] { batch, seqLen, inFeatures }) : null;
        int chunks = ws.Chunks;
        fixed (float* pGradInput = gradInput is null ? Span<float>.Empty : gradInput.AsWritableSpan())
        {
            // The region is synchronous, so the pinned input gradient outlives every chunk's use of it.
            IntPtr gradInputPtr = (IntPtr)pGradInput;
            if (chunks == 1)
            {
                LstmBackwardChunk(0, ws, scratch, gradOutArr, returnSequences, wHhArr, wIhArr, inputArr, gradInputPtr,
                    needWHh, needBias, needH0);
            }
            else
            {
                long work = (long)batch * seqLen * G * (inFeatures + hidden);
                CpuParallelSettings.ParallelForOrSerial(0, chunks, work,
                    c => LstmBackwardChunk(c, ws, scratch, gradOutArr, returnSequences, wHhArr, wIhArr, inputArr, gradInputPtr,
                        needWHh, needBias, needH0),
                    deterministicSafe: true);
            }
        }

        if (gradInput is not null)
            DifferentiableOps.AccumulateGrad(grads, input, gradInput, engine);

        // Sum the chunks' partial gradients in chunk order (fixed by the batch size, so thread-count independent), then
        // transpose the [k, G] partial layout back to the [G, k] weight layout.
        if (needWIh)
            DifferentiableOps.AccumulateGrad(grads, wIh, LstmWeightGradient(scratch.PartWIh, chunks, inFeatures, G), engine);
        if (needWHh)
            DifferentiableOps.AccumulateGrad(grads, wHh, LstmWeightGradient(scratch.PartWHh, chunks, hidden, G), engine);

        if (needBias)
        {
            var gradB = scratch.PartB[0];
            for (int c = 1; c < chunks; c++) LstmAddInto(gradB, scratch.PartB[c], G);
            if (needBIh)
            {
                var gb = new Tensor<float>(new[] { G });
                gradB.AsSpan(0, G).CopyTo(gb.AsWritableSpan());
                DifferentiableOps.AccumulateGrad(grads, inp[idxBIh], gb, engine);
            }
            if (needBHh)
            {
                var gb = new Tensor<float>(new[] { G });
                gradB.AsSpan(0, G).CopyTo(gb.AsWritableSpan());
                DifferentiableOps.AccumulateGrad(grads, inp[idxBHh], gb, engine);
            }
        }

        // gradH0 = dh after t = 0 (dh_{-1}); gradC0 = dc after t = 0. Each chunk left its rows in its carries.
        if (needH0 || needC0)
        {
            var gh0 = needH0 ? new Tensor<float>(new[] { batch, hidden }) : null;
            var gc0 = needC0 ? new Tensor<float>(new[] { batch, hidden }) : null;
            var gh0Span = gh0 is null ? Span<float>.Empty : gh0.AsWritableSpan();
            var gc0Span = gc0 is null ? Span<float>.Empty : gc0.AsWritableSpan();
            for (int c = 0; c < chunks; c++)
            {
                int b0 = LstmChunkStart(c, batch, chunks), nb = LstmChunkStart(c + 1, batch, chunks) - b0;
                if (gh0 is not null) scratch.DH[c].AsSpan(0, nb * hidden).CopyTo(gh0Span.Slice(b0 * hidden, nb * hidden));
                if (gc0 is not null) scratch.DC[c].AsSpan(0, nb * hidden).CopyTo(gc0Span.Slice(b0 * hidden, nb * hidden));
            }
            if (gh0 is not null) DifferentiableOps.AccumulateGrad(grads, inp[idxH0], gh0, engine);
            if (gc0 is not null) DifferentiableOps.AccumulateGrad(grads, inp[idxC0], gc0, engine);
        }
    }

    /// <summary>Σ_c parts[c] ([k, G], chunk order) transposed into a fresh [G, k] gradient tensor.</summary>
    private static Tensor<float> SumChunkPartialsTransposed(float[][] parts, int chunks, int k, int G)
    {
        var sum = parts[0];
        for (int c = 1; c < chunks; c++) LstmAddInto(sum, parts[c], k * G);
        var grad = new Tensor<float>(new[] { G, k });
        var dst = grad.AsWritableSpan();
        for (int g = 0; g < G; g++)
            for (int i = 0; i < k; i++)
                dst[g * k + i] = sum[i * G + g];
        return grad;
    }

    /// <summary>
    /// One chunk's BPTT: walks t = seqLen-1..0 over the chunk's rows (cell rows, then dh_{t-1} = dgates_t·WHh), then
    /// forms the chunk's input-gradient rows (dgates·WIh) and its partial weight/bias gradients
    /// (xᵀ·dgates, h_{t-1}ᵀ·dgates, Σ dgates). Leaves dh/dc after t = 0 in the chunk's carries for gradH0/gradC0.
    /// </summary>
    private static unsafe void LstmBackwardChunk(int c, LstmTrainWorkspace ws, LstmBackwardScratch scratch, float[] gradOutArr, bool returnSequences,
        float[] wHhArr, float[]? wIhArr, float[]? inputArr, IntPtr gradInputPtr, bool needWHh, bool needBias, bool needH0)
    {
        int batch = ws.Batch, seqLen = ws.SeqLen, inFeatures = ws.InFeatures, hidden = ws.Hidden;
        int G = 4 * hidden;
        int b0 = LstmChunkStart(c, batch, ws.Chunks), b1 = LstmChunkStart(c + 1, batch, ws.Chunks), nb = b1 - b0;
        int rows = nb * seqLen;
        // One chunk = the whole batch on the calling thread: its GEMMs may use the pool (see LstmForwardChunk).
        bool soloChunk = ws.Chunks == 1;
        var partWIh = scratch.PartWIh[c];
        var partWHh = scratch.PartWHh[c];
        var partB = scratch.PartB[c];
        // Partials are summed over every chunk afterwards, so each starts from zero even when the chunk is empty.
        Array.Clear(partWIh, 0, inFeatures * G);
        Array.Clear(partWHh, 0, hidden * G);
        Array.Clear(partB, 0, G);
        if (nb == 0) return;

        var gates = ws.Gates;
        var cells = ws.Cells;
        var hiddens = ws.Hiddens;
        var tanhC = ws.TanhC;
        var dgatesAll = ws.Wx;
        var dgatesT = scratch.DGatesT[c];
        var dh = scratch.DH[c];
        var dc = scratch.DC[c];
        int hStride = (seqLen + 1) * hidden;
        Array.Clear(dh, 0, nb * hidden);
        Array.Clear(dc, 0, nb * hidden);

        for (int t = seqLen - 1; t >= 0; t--)
        {
            // Upstream dh at this step: returnSequences feeds every step; otherwise only the last step receives the
            // [batch, hidden] gradient.
            bool feed = returnSequences || t == seqLen - 1;
            for (int b = b0; b < b1; b++)
            {
                int r = b - b0;
                int goRow = returnSequences ? (b * seqLen + t) * hidden : b * hidden;
                BpttCellRow(gates, (t * batch + b) * G, cells, b * hStride + t * hidden, tanhC, (b * seqLen + t) * hidden,
                    dh, dc, r * hidden, feed ? gradOutArr : null, goRow,
                    dgatesT, r * G, dgatesAll, (b * seqLen + t) * G, hidden);
            }

            // dh_{t-1} = dgates_t·WHh: the carry into step t-1 (it overwrites the dh this step just consumed). After
            // t = 0 it is the h0 gradient, needed only when h0 is an input.
            if (t > 0 || needH0)
            {
                Array.Clear(dh, 0, nb * hidden);
                SimdGemm.SgemmAddInternal(dgatesT.AsSpan(0, nb * G), G, false, wHhArr.AsSpan(0, G * hidden), hidden, false,
                    dh.AsSpan(0, nb * hidden), nb, G, hidden, allowParallel: false, clearedOutput: true);
            }
        }

        int dgOff = b0 * seqLen * G;
        var dgChunk = dgatesAll.AsSpan(dgOff, rows * G);

        // Input-gradient rows of the chunk: dgates·WIh -> [rows, inFeatures].
        if (gradInputPtr != IntPtr.Zero && wIhArr is not null)
        {
            var gi = new Span<float>((float*)gradInputPtr + (long)b0 * seqLen * inFeatures, rows * inFeatures);
            if (soloChunk)
            {
                LstmSoloGemm(dgChunk, G, wIhArr.AsSpan(0, G * inFeatures), inFeatures, gi, rows, inFeatures, G);
            }
            else
            {
                gi.Clear();
                SimdGemm.SgemmAddInternal(dgChunk, G, false, wIhArr.AsSpan(0, G * inFeatures), inFeatures, false,
                    gi, rows, G, inFeatures, allowParallel: false, clearedOutput: true);
            }
        }

        // Partial dWIhᵀ = xᵀ·dgates over the chunk's rows ([in, G]). A single chunk instead forms dWIh = dgatesᵀ·x
        // directly in the [G, in] gradient layout on the parallel dispatcher (see LstmWeightGradient).
        if (inputArr is not null && soloChunk)
        {
            LstmSoloGemm(dgChunk, G, inputArr.AsSpan(b0 * seqLen * inFeatures, rows * inFeatures), inFeatures,
                partWIh.AsSpan(0, G * inFeatures), G, inFeatures, rows, transA: true);
        }
        else if (inputArr is not null)
        {
            var xt = scratch.XT[c];
            int xOff = b0 * seqLen * inFeatures;
            for (int r = 0; r < rows; r++)
            {
                int src = xOff + r * inFeatures;
                for (int i = 0; i < inFeatures; i++) xt[i * rows + r] = inputArr[src + i];
            }
            SimdGemm.SgemmAddInternal(xt.AsSpan(0, inFeatures * rows), rows, false, dgChunk, G, false,
                partWIh.AsSpan(0, inFeatures * G), inFeatures, rows, G, allowParallel: false, clearedOutput: true);
        }

        // Partial dWHhᵀ = h_{t-1}ᵀ·dgates over the chunk's rows ([hidden, G]); row (b, t) pairs with h_{t-1} of b.
        if (needWHh && soloChunk)
        {
            // h_{t-1} rows of each sample are contiguous in the saved hiddens; gather them row-major and form
            // dWHh = dgatesᵀ·h_{t-1} directly in the [G, hidden] layout.
            var hp = scratch.HT[c];
            for (int b = b0; b < b1; b++)
                Array.Copy(hiddens, b * hStride, hp, (b - b0) * seqLen * hidden, seqLen * hidden);
            LstmSoloGemm(dgChunk, G, hp.AsSpan(0, rows * hidden), hidden, partWHh.AsSpan(0, G * hidden), G, hidden, rows,
                transA: true);
        }
        else if (needWHh)
        {
            var ht = scratch.HT[c];
            for (int b = b0; b < b1; b++)
                for (int t = 0; t < seqLen; t++)
                {
                    int r = (b - b0) * seqLen + t;
                    int src = b * hStride + t * hidden;
                    for (int h = 0; h < hidden; h++) ht[h * rows + r] = hiddens[src + h];
                }
            SimdGemm.SgemmAddInternal(ht.AsSpan(0, hidden * rows), rows, false, dgChunk, G, false,
                partWHh.AsSpan(0, hidden * G), hidden, rows, G, allowParallel: false, clearedOutput: true);
        }

        // Partial bias gradient: Σ over the chunk's rows of dgates (row order; vectorized over the gate axis).
        if (needBias)
        {
            for (int r = 0; r < rows; r++)
            {
                int off = dgOff + r * G;
                int g = 0;
                if (System.Numerics.Vector.IsHardwareAccelerated)
                {
                    int vw = System.Numerics.Vector<float>.Count;
                    for (; g <= G - vw; g += vw)
                        (new System.Numerics.Vector<float>(partB, g) + new System.Numerics.Vector<float>(dgatesAll, off + g))
                            .CopyTo(partB, g);
                }
                for (; g < G; g++) partB[g] += dgatesAll[off + g];
            }
        }
    }

    // ── Double fused BPTT (#478 follow-up): same one-tape-node + pooled-scratch structure as the
    //    float fast path above, so a <double> LSTM under a gradient tape trains FUSED (one backward
    //    node, not ~5·seqLen decomposed tape ops) instead of throwing. Native double arithmetic +
    //    scalar Math.Exp activations; the big GEMMs (Wx / dInput / dWih / dWhh) go through the generic
    //    parallel BlasManaged.Gemm<double>, the tiny per-step recurrent GEMMs stay sequential. ──────

    /// <summary>The large, parallel double GEMM used by the fused LSTM backward pass.</summary>
    /// <remarks>
    /// Named to contrast with the tiny per-step recurrent GEMMs, which stay sequential: those are
    /// too small to pay for a parallel dispatch and would oversubscribe the pool inside the
    /// timestep loop.
    /// </remarks>
    private static void GemmBigD(System.ReadOnlySpan<double> a, int lda, bool transA,
                                 System.ReadOnlySpan<double> b, int ldb, bool transB,
                                 System.Span<double> c, int m, int k, int n)
    {
        BlasManaged.BlasManaged.Gemm<double>(a, lda, transA, b, ldb, transB, c, n, m, n, k,
            new BlasManaged.BlasOptions<double> { PackingMode = BlasManaged.PackingMode.DisableAutotune });
    }

    /// <summary>In-place logistic sigmoid over a double span, at full precision.</summary>
    /// <remarks>"Exact" distinguishes this from the approximated vectorized variants: the backward
    /// pass reuses these activations to form gradients, so an approximation error here is amplified
    /// through the whole BPTT chain.</remarks>
    [MethodImpl(Hot)]
    private static void SigmoidExactInPlaceD(double[] buf, int off, int len)
    {
        for (int i = off; i < off + len; i++) buf[i] = 1.0 / (1.0 + Math.Exp(-buf[i]));
    }

    /// <summary>In-place hyperbolic tangent over a double span, at full precision.</summary>
    /// <remarks>Uses the overflow-safe 2/(1+e^-2x) - 1 form, so a large magnitude saturates to
    /// +/-1 rather than producing NaN.</remarks>
    [MethodImpl(Hot)]
    private static void TanhExactInPlaceD(double[] buf, int off, int len)
    {
        // tanh(x) = 2/(1+exp(-2x)) - 1 (overflow-safe; large |x| → ±1, never NaN).
        for (int i = off; i < off + len; i++) buf[i] = 2.0 / (1.0 + Math.Exp(-2.0 * buf[i])) - 1.0;
    }

    /// <summary>
    /// Fused double LSTM forward that also records the activations its backward pass needs.
    /// </summary>
    /// <remarks>
    /// Separate from the inference forward because training has to retain per-timestep gate
    /// activations for BPTT, which inference discards. <paramref name="wantState"/> is rejected
    /// under an active gradient tape: the fused node records a backward edge only for the sequence
    /// output, so a returned final hidden/cell would silently carry no gradient.
    /// </remarks>
    [MethodImpl(Hot)]
    private Tensor<double> LstmSequenceForwardDoubleTrain(
        Tensor<double> input, Tensor<double>? h0, Tensor<double>? c0,
        Tensor<double> wIh, Tensor<double> wHh, Tensor<double>? bIh, Tensor<double>? bHh,
        int batch, int seqLen, int inFeatures, int hidden, int gateRows,
        bool returnSequences, bool wantState,
        out Tensor<double> finalHidden, out Tensor<double> finalCell)
    {
        if (wantState && DifferentiableOps.ThreadTapeActive())
        {
            throw new ArgumentException(
                "LstmSequenceForwardDoubleTrain: wantState=true is not supported under an active gradient tape. " +
                "The fused BPTT node records gradients only for the sequence output; the returned finalHidden / " +
                "finalCell carry no backward edge. Set wantState=false, or slice the final state out of the output.",
                nameof(wantState));
        }

        int G = gateRows;
        int totalRows = batch * seqLen;

        var inSpan = input.GetFlattenedData().AsSpan();
        var wIhSpan = wIh.GetFlattenedData().AsSpan();
        var wHhSpan = wHh.GetFlattenedData().AsSpan();
        double[]? bIhArr = bIh?.GetFlattenedData();
        double[]? bHhArr = bHh?.GetFlattenedData();
        double[]? h0Arr = h0?.GetFlattenedData();
        double[]? c0Arr = c0?.GetFlattenedData();

        // Saved state (captured by the backward closure). Layout matches the float path.
        var gates = new double[totalRows * G];
        var cells = new double[batch * (seqLen + 1) * hidden];
        var hiddens = new double[batch * (seqLen + 1) * hidden];

        for (int b = 0; b < batch; b++)
        {
            int baseTc = b * (seqLen + 1) * hidden;
            for (int h = 0; h < hidden; h++)
            {
                cells[baseTc + h] = c0Arr is null ? 0.0 : c0Arr[b * hidden + h];
                hiddens[baseTc + h] = h0Arr is null ? 0.0 : h0Arr[b * hidden + h];
            }
        }

        // Wx[b,t,:] = wIh @ x[b,t] (+ bIh): one big GEMM x[totalRows, inF] @ wIh^T[inF, G].
        var wx = new double[totalRows * G];
        GemmBigD(inSpan, inFeatures, false, wIhSpan, inFeatures, true,
                 wx.AsSpan(0, totalRows * G), totalRows, inFeatures, G);
        if (bIhArr is not null)
            for (int r = 0; r < totalRows; r++)
            {
                int off = r * G;
                for (int g = 0; g < G; g++) wx[off + g] += bIhArr[g];
            }

        // Pre-transpose wHh [G, hidden] → wHhT [hidden, G] for the per-step recurrent GEMM.
        var wHhT = new double[hidden * G];
        for (int g = 0; g < G; g++)
            for (int h = 0; h < hidden; h++)
                wHhT[h * G + g] = wHhSpan[g * hidden + h];

        var output = returnSequences
            ? new Tensor<double>(new[] { batch, seqLen, hidden })
            : new Tensor<double>(new[] { batch, hidden });
        var outSpan = output.AsWritableSpan();

        if (seqLen == 0)
        {
            if (!returnSequences)
                for (int b = 0; b < batch; b++)
                    for (int h = 0; h < hidden; h++)
                        outSpan[b * hidden + h] = h0Arr is null ? 0.0 : h0Arr[b * hidden + h];
            if (wantState)
            {
                finalHidden = new Tensor<double>(new[] { batch, hidden });
                finalCell = new Tensor<double>(new[] { batch, hidden });
                var fh0 = finalHidden.AsWritableSpan(); var fc0 = finalCell.AsWritableSpan();
                for (int b = 0; b < batch; b++)
                    for (int h = 0; h < hidden; h++)
                    {
                        fh0[b * hidden + h] = h0Arr is null ? 0.0 : h0Arr[b * hidden + h];
                        fc0[b * hidden + h] = c0Arr is null ? 0.0 : c0Arr[b * hidden + h];
                    }
            }
            else { finalHidden = new Tensor<double>(new[] { 0 }); finalCell = new Tensor<double>(new[] { 0 }); }
            return output;
        }

        var hPrev = new double[batch * hidden];
        var hh = new double[batch * G];
        for (int b = 0; b < batch; b++)
            Array.Copy(hiddens, b * (seqLen + 1) * hidden, hPrev, b * hidden, hidden);

        int bh = batch * hidden;
        var act = new double[4 * bh];
        var cbuf = new double[bh];

        for (int t = 0; t < seqLen; t++)
        {
            // hh = h_prev @ wHhT → [batch, G]. BlasManaged double microkernel — the generic
            // MatrixMultiplyHelper.MultiplyBlocked is ~128x slower at this tiny per-step shape
            // (measured 320ms vs 2.5ms for the 32-step loop), which was the whole double-vs-float gap.
            GemmBigD(hPrev.AsSpan(0, batch * hidden), hidden, false,
                     wHhT.AsSpan(0, hidden * G), G, false,
                     hh.AsSpan(0, batch * G), batch, hidden, G);

            for (int b = 0; b < batch; b++)
            {
                int wxBase = (b * seqLen + t) * G;
                int hhBase = b * G;
                for (int g = 0; g < 4; g++)
                {
                    int wxg = wxBase + g * hidden, hhg = hhBase + g * hidden, dst = g * bh + b * hidden;
                    if (bHhArr is not null)
                    {
                        int bOff = g * hidden;
                        for (int h = 0; h < hidden; h++) act[dst + h] = wx[wxg + h] + hh[hhg + h] + bHhArr[bOff + h];
                    }
                    else
                        for (int h = 0; h < hidden; h++) act[dst + h] = wx[wxg + h] + hh[hhg + h];
                }
            }

            SigmoidExactInPlaceD(act, 0 * bh, bh);
            SigmoidExactInPlaceD(act, 1 * bh, bh);
            TanhExactInPlaceD(act, 2 * bh, bh);
            SigmoidExactInPlaceD(act, 3 * bh, bh);

            for (int b = 0; b < batch; b++)
            {
                int cPrevRow = (b * (seqLen + 1) + t) * hidden;
                int cCurRow = (b * (seqLen + 1) + t + 1) * hidden;
                int gb = b * hidden;
                for (int h = 0; h < hidden; h++)
                {
                    double c = act[1 * bh + gb + h] * cells[cPrevRow + h] + act[0 * bh + gb + h] * act[2 * bh + gb + h];
                    cells[cCurRow + h] = c;
                    cbuf[gb + h] = c;
                }
            }
            TanhExactInPlaceD(cbuf, 0, bh);

            for (int b = 0; b < batch; b++)
            {
                int gateRow = (b * seqLen + t) * G;
                int cCurRow = (b * (seqLen + 1) + t + 1) * hidden;
                int outRow = returnSequences ? (b * seqLen + t) * hidden : b * hidden;
                int gb = b * hidden;
                bool writeOut = returnSequences || t == seqLen - 1;
                for (int h = 0; h < hidden; h++)
                {
                    double hOut = act[3 * bh + gb + h] * cbuf[gb + h];
                    gates[gateRow + 0 * hidden + h] = act[0 * bh + gb + h];
                    gates[gateRow + 1 * hidden + h] = act[1 * bh + gb + h];
                    gates[gateRow + 2 * hidden + h] = act[2 * bh + gb + h];
                    gates[gateRow + 3 * hidden + h] = act[3 * bh + gb + h];
                    hiddens[cCurRow + h] = hOut;
                    if (writeOut) outSpan[outRow + h] = hOut;
                }
            }

            for (int b = 0; b < batch; b++)
                Array.Copy(hiddens, (b * (seqLen + 1) + t + 1) * hidden, hPrev, b * hidden, hidden);
        }

        if (wantState)
        {
            finalHidden = new Tensor<double>(new[] { batch, hidden });
            finalCell = new Tensor<double>(new[] { batch, hidden });
            var fh = finalHidden.AsWritableSpan(); var fc = finalCell.AsWritableSpan();
            for (int b = 0; b < batch; b++)
                for (int h = 0; h < hidden; h++)
                {
                    fh[b * hidden + h] = hiddens[(b * (seqLen + 1) + seqLen) * hidden + h];
                    fc[b * hidden + h] = cells[(b * (seqLen + 1) + seqLen) * hidden + h];
                }
        }
        else { finalHidden = new Tensor<double>(new[] { 0 }); finalCell = new Tensor<double>(new[] { 0 }); }

        int nInputs = 3 + (bIh is not null ? 1 : 0) + (bHh is not null ? 1 : 0)
                        + (h0 is not null ? 1 : 0) + (c0 is not null ? 1 : 0);
        var inputsArr = new Tensor<double>[nInputs];
        int idx = 0;
        inputsArr[idx++] = input;
        inputsArr[idx++] = wIh;
        inputsArr[idx++] = wHh;
        int idxBIh = bIh is not null ? idx : -1; if (bIh is not null) inputsArr[idx++] = bIh;
        int idxBHh = bHh is not null ? idx : -1; if (bHh is not null) inputsArr[idx++] = bHh;
        int idxH0 = h0 is not null ? idx : -1; if (h0 is not null) inputsArr[idx++] = h0;
        int idxC0 = c0 is not null ? idx : -1; if (c0 is not null) inputsArr[idx++] = c0;

        var meta = new int[] { batch, seqLen, inFeatures, hidden, returnSequences ? 1 : 0,
                               idxBIh, idxBHh, idxH0, idxC0 };
        var savedState = new object[] { gates, cells, hiddens, meta };

        DifferentiableOps.RecordIfActive<double>(
            "LstmSequenceForward", output, inputsArr, LstmSequenceBackwardDouble, savedState);

        return output;
    }

    /// <summary>
    /// BPTT backward for the fused double LSTM (mirror of <see cref="LstmSequenceBackwardFloat"/>).
    /// Native double arithmetic; pooled backward scratch; big GEMMs via the generic parallel
    /// BlasManaged.Gemm&lt;double&gt;, the per-step recurrent GEMM sequential.
    /// </summary>
    [MethodImpl(Hot)]
    private static void LstmSequenceBackwardDouble(
        Tensor<double> gradOutput, Tensor<double>[] inp, Tensor<double> output,
        object[] savedState, IEngine engine, System.Collections.Generic.Dictionary<Tensor<double>, Tensor<double>> grads)
    {
        var gates = (double[])savedState[0];
        var cells = (double[])savedState[1];
        var hiddens = (double[])savedState[2];
        var meta = (int[])savedState[3];
        int batch = meta[0], seqLen = meta[1], inFeatures = meta[2], hidden = meta[3];
        bool returnSequences = meta[4] != 0;
        int idxBIh = meta[5], idxBHh = meta[6], idxH0 = meta[7], idxC0 = meta[8];
        int G = 4 * hidden;
        int totalRows = batch * seqLen;

        var input = inp[0];
        var wIh = inp[1];
        var wHh = inp[2];
        var wHhSpan = wHh.GetFlattenedData().AsSpan();
        var wIhSpan = wIh.GetFlattenedData().AsSpan();
        var inputSpan = input.GetFlattenedData().AsSpan();
        var gradOutSpan = gradOutput.GetFlattenedData().AsSpan();

        var pool = System.Buffers.ArrayPool<double>.Shared;
        var dgatesAll = pool.Rent(totalRows * G);
        var dhNext = pool.Rent(batch * hidden);
        var dcNext = pool.Rent(batch * hidden);
        System.Array.Clear(dhNext, 0, batch * hidden);
        System.Array.Clear(dcNext, 0, batch * hidden);
        var dgatesT = pool.Rent(batch * G);
        var dhPrev = pool.Rent(batch * hidden);

        for (int t = seqLen - 1; t >= 0; t--)
        {
            for (int b = 0; b < batch; b++)
            {
                int gateRow = (b * seqLen + t) * G;
                int cPrevRow = (b * (seqLen + 1) + t) * hidden;
                int cCurRow = (b * (seqLen + 1) + t + 1) * hidden;
                int dgtRow = b * G;
                int carryRow = b * hidden;

                int goRow = returnSequences ? (b * seqLen + t) * hidden : b * hidden;
                bool feedThisStep = returnSequences || t == seqLen - 1;

                for (int h = 0; h < hidden; h++)
                {
                    double ig = gates[gateRow + 0 * hidden + h];
                    double fg = gates[gateRow + 1 * hidden + h];
                    double gg = gates[gateRow + 2 * hidden + h];
                    double og = gates[gateRow + 3 * hidden + h];
                    double cCur = cells[cCurRow + h];
                    double cPrev = cells[cPrevRow + h];

                    double dhTotal = dhNext[carryRow + h] + (feedThisStep ? gradOutSpan[goRow + h] : 0.0);
                    double tc = Math.Tanh(cCur);

                    double doPre = dhTotal * tc * og * (1.0 - og);                  // sigmoid'
                    double dcTotal = dcNext[carryRow + h] + dhTotal * og * (1.0 - tc * tc); // tanh'

                    double diPre = dcTotal * gg * ig * (1.0 - ig);
                    double dgPre = dcTotal * ig * (1.0 - gg * gg);
                    double dfPre = dcTotal * cPrev * fg * (1.0 - fg);
                    dcNext[carryRow + h] = dcTotal * fg;

                    dgatesT[dgtRow + 0 * hidden + h] = diPre;
                    dgatesT[dgtRow + 1 * hidden + h] = dfPre;
                    dgatesT[dgtRow + 2 * hidden + h] = dgPre;
                    dgatesT[dgtRow + 3 * hidden + h] = doPre;

                    dgatesAll[gateRow + 0 * hidden + h] = diPre;
                    dgatesAll[gateRow + 1 * hidden + h] = dfPre;
                    dgatesAll[gateRow + 2 * hidden + h] = dgPre;
                    dgatesAll[gateRow + 3 * hidden + h] = doPre;
                }
            }

            // dh_prev = dgates_t @ wHh → [batch, hidden]; carried to t-1. BlasManaged double
            // microkernel (NOT the generic MultiplyBlocked — ~128x slower at this per-step shape).
            GemmBigD(dgatesT.AsSpan(0, batch * G), G, false,
                     wHhSpan.Slice(0, G * hidden), hidden, false,
                     dhPrev.AsSpan(0, batch * hidden), batch, G, hidden);
            Array.Copy(dhPrev, dhNext, batch * hidden);
        }

        // gradInput = dgatesAll @ wIh → [totalRows, inFeatures]
        var gradInput = new Tensor<double>(new[] { batch, seqLen, inFeatures });
        GemmBigD(dgatesAll.AsSpan(0, totalRows * G), G, false,
                 wIhSpan.Slice(0, G * inFeatures), inFeatures, false,
                 gradInput.AsWritableSpan(), totalRows, G, inFeatures);
        DifferentiableOps.AccumulateGrad(grads, input, gradInput, engine);

        // gradWIh = dgatesAll^T @ input2d → [G, inFeatures]
        var gradWIh = new Tensor<double>(new[] { G, inFeatures });
        GemmBigD(dgatesAll.AsSpan(0, totalRows * G), G, true,
                 inputSpan, inFeatures, false,
                 gradWIh.AsWritableSpan(), G, totalRows, inFeatures);
        DifferentiableOps.AccumulateGrad(grads, wIh, gradWIh, engine);

        // gradWHh = dgatesAll^T @ hPrevAll → [G, hidden]; hPrevAll[b,t] = h_{t-1}.
        var hPrevAll = pool.Rent(totalRows * hidden);
        for (int b = 0; b < batch; b++)
            for (int t = 0; t < seqLen; t++)
                Array.Copy(hiddens, (b * (seqLen + 1) + t) * hidden,
                           hPrevAll, (b * seqLen + t) * hidden, hidden);
        var gradWHh = new Tensor<double>(new[] { G, hidden });
        GemmBigD(dgatesAll.AsSpan(0, totalRows * G), G, true,
                 hPrevAll.AsSpan(0, totalRows * hidden), hidden, false,
                 gradWHh.AsWritableSpan(), G, totalRows, hidden);
        DifferentiableOps.AccumulateGrad(grads, wHh, gradWHh, engine);

        // gradBIh = gradBHh = column-sum of dgatesAll over (b,t) → [G].
        if (idxBIh >= 0 || idxBHh >= 0)
        {
            var gradB = new double[G];
            for (int r = 0; r < totalRows; r++)
            {
                int off = r * G;
                for (int g = 0; g < G; g++) gradB[g] += dgatesAll[off + g];
            }
            if (idxBIh >= 0)
            {
                var gb = new Tensor<double>(new[] { G });
                gradB.AsSpan().CopyTo(gb.AsWritableSpan());
                DifferentiableOps.AccumulateGrad(grads, inp[idxBIh], gb, engine);
            }
            if (idxBHh >= 0)
            {
                var gb = new Tensor<double>(new[] { G });
                gradB.AsSpan().CopyTo(gb.AsWritableSpan());
                DifferentiableOps.AccumulateGrad(grads, inp[idxBHh], gb, engine);
            }
        }

        if (idxH0 >= 0)
        {
            var gh0 = new Tensor<double>(new[] { batch, hidden });
            dhNext.AsSpan(0, batch * hidden).CopyTo(gh0.AsWritableSpan());
            DifferentiableOps.AccumulateGrad(grads, inp[idxH0], gh0, engine);
        }
        if (idxC0 >= 0)
        {
            var gc0 = new Tensor<double>(new[] { batch, hidden });
            dcNext.AsSpan(0, batch * hidden).CopyTo(gc0.AsWritableSpan());
            DifferentiableOps.AccumulateGrad(grads, inp[idxC0], gc0, engine);
        }

        pool.Return(dgatesAll);
        pool.Return(dhNext);
        pool.Return(dcNext);
        pool.Return(dgatesT);
        pool.Return(dhPrev);
        pool.Return(hPrevAll);
    }
}
