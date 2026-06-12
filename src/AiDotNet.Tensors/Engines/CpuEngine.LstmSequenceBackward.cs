// Copyright (c) AiDotNet. All rights reserved.

using System;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Simd;
using AiDotNet.Tensors.LinearAlgebra;

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
    /// Routes a LARGE float GEMM to the parallel BlasManaged dispatcher. Same argument
    /// order as <see cref="SimdGemm.Sgemm"/> (result is [m, n], k is the contraction).
    /// The fused LSTM kernel originally ran every GEMM single-threaded (Sgemm /
    /// SgemmSequential), which left the big forward Wx and backward dInput/dWih/dWhh GEMMs
    /// serial and capped the fused-training speedup. These have large m (totalRows or 4·H),
    /// so the M-axis-parallel dispatcher wins; the per-timestep recurrent GEMMs stay on
    /// SgemmSequential (tiny m, and inherently serial across the BPTT recurrence — parallel
    /// dispatch overhead per step would not pay off). DisableAutotune = the static-heuristic
    /// kernel (same determinism contract as the eager-matmul fast path).
    /// </summary>
    private static void GemmBig(System.ReadOnlySpan<float> a, int lda, bool transA,
                                System.ReadOnlySpan<float> b, int ldb, bool transB,
                                System.Span<float> c, int m, int k, int n)
    {
        BlasManaged.BlasManaged.Gemm<float>(a, lda, transA, b, ldb, transB, c, n, m, n, k,
            new BlasManaged.BlasOptions<float> { PackingMode = BlasManaged.PackingMode.DisableAutotune });
    }

    /// <summary>
    /// In-place exact sigmoid over <c>buf[off..off+len)</c>: 1/(1+exp(-x)). Uses
    /// <see cref="SimdKernels.ExpUnsafe"/> (near-exact VML/Herumi vectorized exp on AVX, scalar
    /// elsewhere) for the expensive transcendental; the cheap negate/reciprocal stay scalar. The
    /// exp(-x) form is overflow-safe (large |x| → 0 or 1, never NaN). Result matches the scalar
    /// Math.Exp sigmoid to ~1e-6, so the saved gate values — and the σ(1-σ) gradients computed
    /// from them — are unchanged within the finite-difference tolerance.
    /// </summary>
    private static unsafe void SigmoidExactInPlace(float[] buf, int off, int len)
    {
        fixed (float* p = &buf[off])
        {
            for (int i = 0; i < len; i++) p[i] = -p[i];
            SimdKernels.ExpUnsafe(p, p, len);
            for (int i = 0; i < len; i++) p[i] = 1f / (1f + p[i]);
        }
    }

    /// <summary>
    /// In-place exact tanh over <c>buf[off..off+len)</c> via the overflow-safe identity
    /// tanh(x) = 2/(1+exp(-2x)) - 1 (no e^{2x}, so large x → ±1, never NaN). Same exp seam and
    /// ~1e-6 accuracy as <see cref="SigmoidExactInPlace"/>.
    /// </summary>
    private static unsafe void TanhExactInPlace(float[] buf, int off, int len)
    {
        fixed (float* p = &buf[off])
        {
            for (int i = 0; i < len; i++) p[i] = -2f * p[i];
            SimdKernels.ExpUnsafe(p, p, len);
            for (int i = 0; i < len; i++) p[i] = 2f / (1f + p[i]) - 1f;
        }
    }

    /// <summary>
    /// Float training forward: exact activations, saves per-timestep state, and records
    /// a single fused BPTT node on the active tape. Called from LstmSequenceForward when
    /// a gradient tape is active and T == float.
    /// </summary>
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

        var inSpan = input.AsSpan();      // [batch, seqLen, inFeatures], row = b*seqLen + t
        var wIhSpan = wIh.AsSpan();       // [G, inFeatures]
        var wHhSpan = wHh.AsSpan();       // [G, hidden]
        float[]? bIhArr = bIh?.ToArray();
        float[]? bHhArr = bHh?.ToArray();
        float[]? h0Arr = h0?.ToArray();
        float[]? c0Arr = c0?.ToArray();

        // Saved state (captured by the backward closure → persists past this call).
        //   gates:   [b, t] post-activation i|f|g|o, row (b*seqLen+t)*G
        //   cells:   [b, tc] c_0..c_seqLen, row (b*(seqLen+1)+tc)*hidden  (tc=0 is c0)
        //   hiddens: [b, tc] h_0..h_seqLen, row (b*(seqLen+1)+tc)*hidden  (tc=0 is h0)
        var gates = new float[totalRows * G];
        var cells = new float[batch * (seqLen + 1) * hidden];
        var hiddens = new float[batch * (seqLen + 1) * hidden];

        // c_0 / h_0.
        for (int b = 0; b < batch; b++)
        {
            int baseTc = b * (seqLen + 1) * hidden;
            for (int h = 0; h < hidden; h++)
            {
                cells[baseTc + h] = c0Arr is null ? 0f : c0Arr[b * hidden + h];
                hiddens[baseTc + h] = h0Arr is null ? 0f : h0Arr[b * hidden + h];
            }
        }

        // Wx[b,t,:] = wIh @ x[b,t] (+ bIh), one big GEMM into [totalRows, G].
        var wx = new float[totalRows * G];
        GemmBig(inSpan, inFeatures, false, wIhSpan, inFeatures, true,
                wx.AsSpan(0, totalRows * G), totalRows, inFeatures, G);
        if (bIhArr is not null)
        {
            for (int r = 0; r < totalRows; r++)
            {
                int off = r * G;
                for (int g = 0; g < G; g++) wx[off + g] += bIhArr[g];
            }
        }

        // Pre-transpose wHh [G, hidden] → wHhT [hidden, G] for the per-step recurrent GEMM.
        var wHhT = new float[hidden * G];
        for (int g = 0; g < G; g++)
            for (int h = 0; h < hidden; h++)
                wHhT[h * G + g] = wHhSpan[g * hidden + h];

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

        // Per-timestep scratch: hh = h_prev @ wHhT, [batch, G].
        var hPrev = new float[batch * hidden];
        var hh = new float[batch * G];
        // seed hPrev with h_0
        for (int b = 0; b < batch; b++)
            Array.Copy(hiddens, b * (seqLen + 1) * hidden, hPrev, b * hidden, hidden);

        // Gate-major activation scratch so each gate's batch*hidden pre-activations are
        // contiguous for ONE vectorized exact activation call (vs scalar Math.Exp/Tanh per
        // element, which dominated the cell). act[g*bh + b*hidden + h]; cbuf holds c for tanh(c).
        int bh = batch * hidden;
        var act = new float[4 * bh];
        var cbuf = new float[bh];

        for (int t = 0; t < seqLen; t++)
        {
            // hh = h_prev @ wHhT  → [batch, G]
            SimdGemm.SgemmSequential(hPrev.AsSpan(0, batch * hidden), wHhT.AsSpan(0, hidden * G),
                                     hh.AsSpan(0, batch * G), batch, hidden, G);

            // Gather: act[g] = wx + hh (+ bHh), gate-major. Gate order i,f,g(=cell candidate),o.
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
                    {
                        for (int h = 0; h < hidden; h++) act[dst + h] = wx[wxg + h] + hh[hhg + h];
                    }
                }
            }

            // Exact vectorized activations: i,f,o → sigmoid; g → tanh.
            SigmoidExactInPlace(act, 0 * bh, bh);
            SigmoidExactInPlace(act, 1 * bh, bh);
            TanhExactInPlace(act, 2 * bh, bh);
            SigmoidExactInPlace(act, 3 * bh, bh);

            // Cell: c = f·c_prev + i·g (write raw c to cells; copy to cbuf for tanh(c)).
            for (int b = 0; b < batch; b++)
            {
                int cPrevRow = (b * (seqLen + 1) + t) * hidden;
                int cCurRow = (b * (seqLen + 1) + t + 1) * hidden;
                int gb = b * hidden;
                for (int h = 0; h < hidden; h++)
                {
                    float c = act[1 * bh + gb + h] * cells[cPrevRow + h] + act[0 * bh + gb + h] * act[2 * bh + gb + h];
                    cells[cCurRow + h] = c;
                    cbuf[gb + h] = c;
                }
            }
            TanhExactInPlace(cbuf, 0, bh); // cbuf = tanh(c)

            // h = o·tanh(c); scatter saved gates + hiddens + output.
            for (int b = 0; b < batch; b++)
            {
                int gateRow = (b * seqLen + t) * G;
                int cCurRow = (b * (seqLen + 1) + t + 1) * hidden;
                int outRow = returnSequences ? (b * seqLen + t) * hidden : b * hidden;
                int gb = b * hidden;
                bool writeOut = returnSequences || t == seqLen - 1;
                for (int h = 0; h < hidden; h++)
                {
                    float hOut = act[3 * bh + gb + h] * cbuf[gb + h];
                    gates[gateRow + 0 * hidden + h] = act[0 * bh + gb + h];
                    gates[gateRow + 1 * hidden + h] = act[1 * bh + gb + h];
                    gates[gateRow + 2 * hidden + h] = act[2 * bh + gb + h];
                    gates[gateRow + 3 * hidden + h] = act[3 * bh + gb + h];
                    hiddens[cCurRow + h] = hOut;
                    if (writeOut) outSpan[outRow + h] = hOut;
                }
            }

            // h_prev ← h_t for the next step.
            for (int b = 0; b < batch; b++)
                Array.Copy(hiddens, (b * (seqLen + 1) + t + 1) * hidden, hPrev, b * hidden, hidden);
        }

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
        var savedState = new object[] { gates, cells, hiddens, meta };

        DifferentiableOps.RecordIfActive<float>(
            "LstmSequenceForward", output, inputsArr, LstmSequenceBackwardFloat, savedState);

        return output;
    }

    /// <summary>
    /// BPTT backward for the fused LSTM. Accumulates gradients for input, wIh, wHh and
    /// (when present) bIh, bHh, h0, c0. Uses the saved per-timestep gates/cells/hiddens.
    /// Runs with tape recording suppressed (standard backward context), so the internal
    /// GEMMs are plain compute, not new tape nodes.
    /// </summary>
    private static void LstmSequenceBackwardFloat(
        Tensor<float> gradOutput, Tensor<float>[] inp, Tensor<float> output,
        object[] savedState, IEngine engine, System.Collections.Generic.Dictionary<Tensor<float>, Tensor<float>> grads)
    {
        var gates = (float[])savedState[0];
        var cells = (float[])savedState[1];
        var hiddens = (float[])savedState[2];
        var meta = (int[])savedState[3];
        int batch = meta[0], seqLen = meta[1], inFeatures = meta[2], hidden = meta[3];
        bool returnSequences = meta[4] != 0;
        int idxBIh = meta[5], idxBHh = meta[6], idxH0 = meta[7], idxC0 = meta[8];
        int G = 4 * hidden;
        int totalRows = batch * seqLen;

        var input = inp[0];
        var wIh = inp[1];
        var wHh = inp[2];
        var wHhSpan = wHh.AsSpan();          // [G, hidden]
        var gradOutSpan = gradOutput.AsSpan();

        // Pool the backward scratch (it otherwise allocates several MB/step — dgatesAll and
        // hPrevAll dominate — churning Gen0 GC during training). Returned at method end.
        // ArrayPool.Rent does NOT zero, so the read-accumulate carries (dhNext/dcNext, which
        // the t=seqLen-1 iteration reads before writing) must be cleared; dgatesAll/dgatesT/
        // dhPrev are fully written before any read, so they need no clear.
        var pool = System.Buffers.ArrayPool<float>.Shared;
        // dgatesAll [b,t] pre-activation gate grads, row (b*seqLen+t)*G.
        var dgatesAll = pool.Rent(totalRows * G);
        // Recurrent carries, indexed [b, hidden].
        var dhNext = pool.Rent(batch * hidden);
        var dcNext = pool.Rent(batch * hidden);
        System.Array.Clear(dhNext, 0, batch * hidden);
        System.Array.Clear(dcNext, 0, batch * hidden);
        // Per-timestep scratch for the recurrent dh_prev = dgates_t @ wHh GEMM.
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

                // Upstream dh at this timestep: returnSequences feeds every step; otherwise
                // only the last step receives the [batch, hidden] gradient.
                int goRow = returnSequences ? (b * seqLen + t) * hidden : b * hidden;
                bool feedThisStep = returnSequences || t == seqLen - 1;

                for (int h = 0; h < hidden; h++)
                {
                    float ig = gates[gateRow + 0 * hidden + h];
                    float fg = gates[gateRow + 1 * hidden + h];
                    float gg = gates[gateRow + 2 * hidden + h];
                    float og = gates[gateRow + 3 * hidden + h];
                    float cCur = cells[cCurRow + h];
                    float cPrev = cells[cPrevRow + h];

                    float dhTotal = dhNext[carryRow + h] + (feedThisStep ? gradOutSpan[goRow + h] : 0f);
                    float tc = (float)Math.Tanh(cCur);

                    // h = o · tanh(c)
                    float doPre = dhTotal * tc * og * (1f - og);                 // sigmoid'
                    float dcTotal = dcNext[carryRow + h] + dhTotal * og * (1f - tc * tc); // tanh'

                    // c = f · c_prev + i · g
                    float diPre = dcTotal * gg * ig * (1f - ig);
                    float dgPre = dcTotal * ig * (1f - gg * gg);
                    float dfPre = dcTotal * cPrev * fg * (1f - fg);
                    dcNext[carryRow + h] = dcTotal * fg;                          // → dc for t-1

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

            // dh_prev = dgates_t @ wHh  → [batch, hidden]; carried to t-1 as dhNext.
            SimdGemm.SgemmSequential(dgatesT.AsSpan(0, batch * G), wHhSpan.Slice(0, G * hidden),
                                     dhPrev.AsSpan(0, batch * hidden), batch, G, hidden);
            Array.Copy(dhPrev, dhNext, batch * hidden);
        }

        // gradInput = dgatesAll @ wIh  → [totalRows, inFeatures]
        var gradInput = new Tensor<float>(new[] { batch, seqLen, inFeatures });
        GemmBig(dgatesAll.AsSpan(0, totalRows * G), G, false,
                wIh.AsSpan().Slice(0, G * inFeatures), inFeatures, false,
                gradInput.AsWritableSpan(), totalRows, G, inFeatures);
        DifferentiableOps.AccumulateGrad(grads, input, gradInput, engine);

        // gradWIh = dgatesAll^T @ input2d  → [G, inFeatures]
        var gradWIh = new Tensor<float>(new[] { G, inFeatures });
        GemmBig(dgatesAll.AsSpan(0, totalRows * G), G, true,
                input.AsSpan(), inFeatures, false,
                gradWIh.AsWritableSpan(), G, totalRows, inFeatures);
        DifferentiableOps.AccumulateGrad(grads, wIh, gradWIh, engine);

        // gradWHh = dgatesAll^T @ hPrevAll  → [G, hidden]; hPrevAll[b,t] = h_{t-1}.
        var hPrevAll = pool.Rent(totalRows * hidden);
        for (int b = 0; b < batch; b++)
            for (int t = 0; t < seqLen; t++)
                Array.Copy(hiddens, (b * (seqLen + 1) + t) * hidden,
                           hPrevAll, (b * seqLen + t) * hidden, hidden);
        var gradWHh = new Tensor<float>(new[] { G, hidden });
        GemmBig(dgatesAll.AsSpan(0, totalRows * G), G, true,
                hPrevAll.AsSpan(0, totalRows * hidden), hidden, false,
                gradWHh.AsWritableSpan(), G, totalRows, hidden);
        DifferentiableOps.AccumulateGrad(grads, wHh, gradWHh, engine);

        // gradBIh = gradBHh = column-sum of dgatesAll over (b,t) → [G].
        if (idxBIh >= 0 || idxBHh >= 0)
        {
            var gradB = new float[G];
            for (int r = 0; r < totalRows; r++)
            {
                int off = r * G;
                for (int g = 0; g < G; g++) gradB[g] += dgatesAll[off + g];
            }
            if (idxBIh >= 0)
            {
                var gb = new Tensor<float>(new[] { G });
                gradB.AsSpan().CopyTo(gb.AsWritableSpan());
                DifferentiableOps.AccumulateGrad(grads, inp[idxBIh], gb, engine);
            }
            if (idxBHh >= 0)
            {
                var gb = new Tensor<float>(new[] { G });
                gradB.AsSpan().CopyTo(gb.AsWritableSpan());
                DifferentiableOps.AccumulateGrad(grads, inp[idxBHh], gb, engine);
            }
        }

        // gradH0 = dhNext (dh_prev after t=0); gradC0 = dcNext (dc after t=0).
        if (idxH0 >= 0)
        {
            var gh0 = new Tensor<float>(new[] { batch, hidden });
            dhNext.AsSpan(0, batch * hidden).CopyTo(gh0.AsWritableSpan());
            DifferentiableOps.AccumulateGrad(grads, inp[idxH0], gh0, engine);
        }
        if (idxC0 >= 0)
        {
            var gc0 = new Tensor<float>(new[] { batch, hidden });
            dcNext.AsSpan(0, batch * hidden).CopyTo(gc0.AsWritableSpan());
            DifferentiableOps.AccumulateGrad(grads, inp[idxC0], gc0, engine);
        }

        // Return the pooled backward scratch (all consumers above are done). On an exception
        // mid-backward these leak, which is benign for ArrayPool (the pool just allocates anew).
        pool.Return(dgatesAll);
        pool.Return(dhNext);
        pool.Return(dcNext);
        pool.Return(dgatesT);
        pool.Return(dhPrev);
        pool.Return(hPrevAll);
    }
}
