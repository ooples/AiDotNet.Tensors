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
        SimdGemm.Sgemm(inSpan, inFeatures, false, wIhSpan, inFeatures, true,
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

        for (int t = 0; t < seqLen; t++)
        {
            // hh = h_prev @ wHhT  → [batch, G]
            SimdGemm.SgemmSequential(hPrev.AsSpan(0, batch * hidden), wHhT.AsSpan(0, hidden * G),
                                     hh.AsSpan(0, batch * G), batch, hidden, G);

            for (int b = 0; b < batch; b++)
            {
                int wxRow = (b * seqLen + t) * G;
                int hhRow = b * G;
                int gateRow = (b * seqLen + t) * G;
                int cPrevRow = (b * (seqLen + 1) + t) * hidden;
                int cCurRow = (b * (seqLen + 1) + t + 1) * hidden;
                int outRow = returnSequences ? (b * seqLen + t) * hidden : b * hidden;

                for (int h = 0; h < hidden; h++)
                {
                    float iIn = wx[wxRow + 0 * hidden + h] + hh[hhRow + 0 * hidden + h];
                    float fIn = wx[wxRow + 1 * hidden + h] + hh[hhRow + 1 * hidden + h];
                    float gIn = wx[wxRow + 2 * hidden + h] + hh[hhRow + 2 * hidden + h];
                    float oIn = wx[wxRow + 3 * hidden + h] + hh[hhRow + 3 * hidden + h];
                    if (bHhArr is not null)
                    {
                        iIn += bHhArr[0 * hidden + h]; fIn += bHhArr[1 * hidden + h];
                        gIn += bHhArr[2 * hidden + h]; oIn += bHhArr[3 * hidden + h];
                    }

                    float ig = 1f / (1f + (float)Math.Exp(-iIn));
                    float fg = 1f / (1f + (float)Math.Exp(-fIn));
                    float gg = (float)Math.Tanh(gIn);
                    float og = 1f / (1f + (float)Math.Exp(-oIn));

                    float cPrev = cells[cPrevRow + h];
                    float c = fg * cPrev + ig * gg;
                    float hOut = og * (float)Math.Tanh(c);

                    gates[gateRow + 0 * hidden + h] = ig;
                    gates[gateRow + 1 * hidden + h] = fg;
                    gates[gateRow + 2 * hidden + h] = gg;
                    gates[gateRow + 3 * hidden + h] = og;
                    cells[cCurRow + h] = c;
                    hiddens[cCurRow + h] = hOut;

                    if (returnSequences) outSpan[outRow + h] = hOut;
                    else if (t == seqLen - 1) outSpan[outRow + h] = hOut;
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

        // dgatesAll [b,t] pre-activation gate grads, row (b*seqLen+t)*G.
        var dgatesAll = new float[totalRows * G];
        // Recurrent carries, indexed [b, hidden].
        var dhNext = new float[batch * hidden];
        var dcNext = new float[batch * hidden];
        // Per-timestep scratch for the recurrent dh_prev = dgates_t @ wHh GEMM.
        var dgatesT = new float[batch * G];
        var dhPrev = new float[batch * hidden];

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
        SimdGemm.SgemmSequential(dgatesAll.AsSpan(0, totalRows * G), wIh.AsSpan().Slice(0, G * inFeatures),
                                 gradInput.AsWritableSpan(), totalRows, G, inFeatures);
        DifferentiableOps.AccumulateGrad(grads, input, gradInput, engine);

        // gradWIh = dgatesAll^T @ input2d  → [G, inFeatures]
        var gradWIh = new Tensor<float>(new[] { G, inFeatures });
        SimdGemm.Sgemm(dgatesAll.AsSpan(0, totalRows * G), G, true,
                       input.AsSpan(), inFeatures, false,
                       gradWIh.AsWritableSpan(), G, totalRows, inFeatures);
        DifferentiableOps.AccumulateGrad(grads, wIh, gradWIh, engine);

        // gradWHh = dgatesAll^T @ hPrevAll  → [G, hidden]; hPrevAll[b,t] = h_{t-1}.
        var hPrevAll = new float[totalRows * hidden];
        for (int b = 0; b < batch; b++)
            for (int t = 0; t < seqLen; t++)
                Array.Copy(hiddens, (b * (seqLen + 1) + t) * hidden,
                           hPrevAll, (b * seqLen + t) * hidden, hidden);
        var gradWHh = new Tensor<float>(new[] { G, hidden });
        SimdGemm.Sgemm(dgatesAll.AsSpan(0, totalRows * G), G, true,
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
    }
}
