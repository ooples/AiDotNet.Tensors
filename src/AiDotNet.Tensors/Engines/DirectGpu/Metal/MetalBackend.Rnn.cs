// Copyright (c) AiDotNet. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

namespace AiDotNet.Tensors.Engines.DirectGpu.Metal;

/// <summary>
/// Metal GPU backend implementation - RNN/LSTM/GRU Sequence Operations.
/// Provides efficient sequence modeling for recurrent neural networks with BPTT support.
/// </summary>
public partial class MetalBackend
{
    #region RNN (LSTM/GRU) Sequence Operations

    // ── Fused recurrence / LM-head GPU kernels (#1464) ─────────────────────────────────────
    // Real MSL compute shaders (MetalRecurrenceKernels). One thread per the same work-item the
    // other backends use. Forward only — the differentiable backward runs through the CpuEngine
    // tape (the engine override is forward-only and defers to base when a tape is active).

    // Dispatch a recurrence kernel: bind buffers 0..b-1, then int scalars b..b+s-1, over `total` threads.
    private void DispatchRecurrence(string kernelName, int total, MetalGpuBuffer[] buffers, int[] scalars)
    {
        ThrowIfDisposed();
        var pipeline = GetPipeline("Recurrence", _recurrenceLibrary, kernelName);
        var (threadgroups, threadsPerGroup) = pipeline.Calculate1DDispatch(total);
        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        int idx = 0;
        foreach (var buf in buffers) encoder.SetBuffer(buf, idx++);
        foreach (var sc in scalars) encoder.SetBytes(sc, (ulong)idx++);
        encoder.DispatchThreadgroups(threadgroups, threadsPerGroup);
    }

    private static MetalGpuBuffer M(IGpuBuffer b) => (MetalGpuBuffer)b;

    public void GlaScanForward(
        IGpuBuffer q, IGpuBuffer k, IGpuBuffer v, IGpuBuffer gate, IGpuBuffer output,
        int batch, int seqLen, int modelDim, int numHeads, int headDim)
        => DispatchRecurrence("gla_scan_forward", batch * numHeads * headDim,
            new[] { M(q), M(k), M(v), M(gate), M(output) },
            new[] { batch, seqLen, modelDim, numHeads, headDim });

    // GLA BPTT backward — real MSL kernels (recompute trajectory + reverse sweep with CAS float
    // atomics for the cross-row dQ/dK/dG). dQ/dK/dG must be pre-zeroed (the atomic accumulators).
    public void GlaScanBackward(
        IGpuBuffer dOut, IGpuBuffer q, IGpuBuffer k, IGpuBuffer v, IGpuBuffer gate,
        IGpuBuffer dQ, IGpuBuffer dK, IGpuBuffer dV, IGpuBuffer dG,
        int batch, int seqLen, int modelDim, int numHeads, int headDim)
    {
        if (batch <= 0 || seqLen <= 0 || modelDim <= 0 || numHeads <= 0 || headDim <= 0)
            throw new ArgumentOutOfRangeException(nameof(batch), "GLA dimensions must be positive.");
        int hh = headDim * headDim;
        long totalLong = checked((long)batch * numHeads * headDim);
        long trajLenLong = checked((long)batch * numHeads * seqLen * hh);
        if (totalLong > int.MaxValue || trajLenLong > int.MaxValue)
            throw new ArgumentOutOfRangeException(nameof(batch), "GLA dimensions exceed Metal launch/buffer limits.");
        int total = (int)totalLong;
        // dQ/dK/dG are atomic accumulators in the backward kernel — zero them here so the
        // method is self-contained and correct even if the caller reuses dirty buffers.
        Fill(dQ, 0f, batch * seqLen * modelDim);
        Fill(dK, 0f, batch * seqLen * modelDim);
        Fill(dG, 0f, batch * seqLen * numHeads);
        using var traj = AllocateBuffer((int)trajLenLong);
        DispatchRecurrence("gla_scan_recompute", total,
            new[] { M(k), M(v), M(gate), M(traj) },
            new[] { batch, seqLen, modelDim, numHeads, headDim });
        DispatchRecurrence("gla_scan_backward", total,
            new[] { M(dOut), M(q), M(k), M(v), M(gate), M(traj), M(dQ), M(dK), M(dV), M(dG) },
            new[] { batch, seqLen, modelDim, numHeads, headDim });
    }

    public void XLstmScanForward(
        IGpuBuffer q, IGpuBuffer k, IGpuBuffer v,
        IGpuBuffer iGate, IGpuBuffer fGate, IGpuBuffer oGate, IGpuBuffer output,
        int batch, int seqLen, int modelDim, int numHeads, int headDim)
        => DispatchRecurrence("xlstm_scan_forward", batch * numHeads,
            new[] { M(q), M(k), M(v), M(iGate), M(fGate), M(oGate), M(output) },
            new[] { batch, seqLen, modelDim, numHeads, headDim });

    public void GatedDeltaNetScanForward(
        IGpuBuffer q, IGpuBuffer k, IGpuBuffer v, IGpuBuffer alpha, IGpuBuffer beta, IGpuBuffer output,
        int batch, int seqLen, int modelDim, int numHeads, int headDim)
        => DispatchRecurrence("gated_delta_scan_forward", batch * numHeads * headDim,
            new[] { M(q), M(k), M(v), M(alpha), M(beta), M(output) },
            new[] { batch, seqLen, modelDim, numHeads, headDim });

    public void RgLruScanForward(
        IGpuBuffer value, IGpuBuffer recGate, IGpuBuffer inpGate, IGpuBuffer decay, IGpuBuffer output,
        int batch, int seqLen, int recDim)
        => DispatchRecurrence("rglru_scan_forward", batch * recDim,
            new[] { M(value), M(recGate), M(inpGate), M(decay), M(output) },
            new[] { batch, seqLen, recDim });

    public void Rwkv4WkvForward(
        IGpuBuffer r, IGpuBuffer k, IGpuBuffer v, IGpuBuffer timeDecay, IGpuBuffer timeFirst, IGpuBuffer output,
        int batch, int seqLen, int modelDim)
        => DispatchRecurrence("rwkv4_wkv_forward", batch * modelDim,
            new[] { M(r), M(k), M(v), M(timeDecay), M(timeFirst), M(output) },
            new[] { batch, seqLen, modelDim });

    public void MambaSelectiveScanForward(
        IGpuBuffer x, IGpuBuffer delta, IGpuBuffer aLog, IGpuBuffer bParam, IGpuBuffer cParam, IGpuBuffer dParam,
        IGpuBuffer output, int batch, int seqLen, int innerDim, int stateDim)
        => DispatchRecurrence("mamba_selective_scan_forward", batch * innerDim,
            new[] { M(x), M(delta), M(aLog), M(bParam), M(cParam), M(dParam), M(output) },
            new[] { batch, seqLen, innerDim, stateDim });

    public void Mamba2SsdScanForward(
        IGpuBuffer x, IGpuBuffer delta, IGpuBuffer aLog, IGpuBuffer bParam, IGpuBuffer cParam, IGpuBuffer dParam,
        IGpuBuffer output, int batch, int seqLen, int innerDim, int numHeads, int headDim, int stateDim)
        => DispatchRecurrence("mamba2_ssd_scan_forward", batch * innerDim,
            new[] { M(x), M(delta), M(aLog), M(bParam), M(cParam), M(dParam), M(output) },
            new[] { batch, seqLen, innerDim, numHeads, headDim, stateDim });

    // CE kernels write a per-row loss vector; the host sums the N-element vector (the expensive
    // logit work stays on the GPU). Returns the mean cross-entropy.
    public float FusedLinearCrossEntropyIndex(
        IGpuBuffer hidden, IGpuBuffer weight, IGpuBuffer bias, IGpuBuffer targetIds, int n, int d, int vocab)
        => FusedCeRowLoss("fused_linear_ce_index", hidden, weight, bias, targetIds, n, d, vocab);

    public float FusedLinearCrossEntropyDense(
        IGpuBuffer hidden, IGpuBuffer weight, IGpuBuffer bias, IGpuBuffer target, int n, int d, int vocab)
        => FusedCeRowLoss("fused_linear_ce_dense", hidden, weight, bias, target, n, d, vocab);

    public void FusedLinearCrossEntropyIndex(
        IGpuBuffer hidden, IGpuBuffer weight, IGpuBuffer bias, IGpuBuffer targetIds,
        IGpuBuffer meanLoss, int n, int d, int vocab)
        => FusedCeRowLossResident("fused_linear_ce_index", hidden, weight, bias, targetIds, meanLoss, n, d, vocab);

    public void FusedLinearCrossEntropyDense(
        IGpuBuffer hidden, IGpuBuffer weight, IGpuBuffer bias, IGpuBuffer target,
        IGpuBuffer meanLoss, int n, int d, int vocab)
        => FusedCeRowLossResident("fused_linear_ce_dense", hidden, weight, bias, target, meanLoss, n, d, vocab);

    private float FusedCeRowLoss(
        string kernelName, IGpuBuffer hidden, IGpuBuffer weight, IGpuBuffer bias, IGpuBuffer tgt, int n, int d, int vocab)
    {
        using var meanLoss = AllocateBuffer(1);
        FusedCeRowLossResident(kernelName, hidden, weight, bias, tgt, meanLoss, n, d, vocab);
        return DownloadBuffer(meanLoss)[0];
    }

    private void FusedCeRowLossResident(
        string kernelName, IGpuBuffer hidden, IGpuBuffer weight, IGpuBuffer bias, IGpuBuffer tgt,
        IGpuBuffer meanLoss, int n, int d, int vocab)
    {
        if (n <= 0 || d <= 0 || vocab <= 0)
            throw new ArgumentOutOfRangeException(nameof(n), "Fused CE dimensions (n, d, vocab) must be positive.");
        using var rowLoss = AllocateBuffer(n);
        DispatchRecurrence(kernelName, n,
            new[] { M(hidden), M(weight), M(bias), M(tgt), M(rowLoss) }, new[] { n, d, vocab });
        SumAxis(rowLoss, meanLoss, 1, n);
        Scale(meanLoss, meanLoss, 1f / n, 1);
    }

    /// <summary>
    /// Forward pass for LSTM sequence - processes all timesteps in a single kernel launch.
    /// Efficient for BPTT training with minimal kernel launch overhead.
    /// </summary>
    /// <param name="input">Input sequence [seqLen * batch * inputSize].</param>
    /// <param name="hInit">Initial hidden state [batch * hiddenSize].</param>
    /// <param name="cInit">Initial cell state [batch * hiddenSize].</param>
    /// <param name="weightsIh">Input-to-hidden weights [4 * hiddenSize * inputSize] (gates: i, f, g, o).</param>
    /// <param name="weightsHh">Hidden-to-hidden weights [4 * hiddenSize * hiddenSize].</param>
    /// <param name="biasIh">Input-to-hidden bias [4 * hiddenSize].</param>
    /// <param name="biasHh">Hidden-to-hidden bias [4 * hiddenSize].</param>
    /// <param name="output">Output sequence [seqLen * batch * hiddenSize].</param>
    /// <param name="hFinal">Final hidden state [batch * hiddenSize].</param>
    /// <param name="cFinal">Final cell state [batch * hiddenSize].</param>
    /// <param name="allH">All hidden states cache [(seqLen + 1) * batch * hiddenSize] for backward.</param>
    /// <param name="allC">All cell states cache [(seqLen + 1) * batch * hiddenSize] for backward.</param>
    /// <param name="cacheGates">Gate cache [seqLen * batch * hiddenSize * 4] for backward.</param>
    /// <param name="seqLen">Sequence length.</param>
    /// <param name="batch">Batch size.</param>
    /// <param name="inputSize">Input feature size.</param>
    /// <param name="hiddenSize">Hidden state size.</param>
    public void LstmForwardSequence(
        IGpuBuffer input, IGpuBuffer hInit, IGpuBuffer cInit,
        IGpuBuffer weightsIh, IGpuBuffer weightsHh, IGpuBuffer biasIh, IGpuBuffer biasHh,
        IGpuBuffer output, IGpuBuffer hFinal, IGpuBuffer cFinal,
        IGpuBuffer allH, IGpuBuffer allC, IGpuBuffer cacheGates,
        int seqLen, int batch, int inputSize, int hiddenSize)
    {
        ThrowIfDisposed();
        if (seqLen <= 0 || batch <= 0 || inputSize <= 0 || hiddenSize <= 0) return;
        int stateSize = checked(batch * hiddenSize);
        Copy(hInit, 0, allH, 0, stateSize);
        Copy(cInit, 0, allC, 0, stateSize);
        DispatchResidentMetal("lstm_forward_sequence", batch,
            new[] { input, weightsIh, weightsHh, biasIh, biasHh, output, allH, allC, cacheGates },
            (uint)seqLen, (uint)batch, (uint)inputSize, (uint)hiddenSize);
        Copy(allH, checked(seqLen * stateSize), hFinal, 0, stateSize);
        Copy(allC, checked(seqLen * stateSize), cFinal, 0, stateSize);
    }

    /// <summary>
    /// Backward pass for LSTM sequence - computes gradients via BPTT.
    /// </summary>
    public void LstmBackwardSequence(
        IGpuBuffer gradOutput, IGpuBuffer allH, IGpuBuffer allC, IGpuBuffer cacheGates,
        IGpuBuffer hInit, IGpuBuffer cInit,
        IGpuBuffer weightsIh, IGpuBuffer weightsHh, IGpuBuffer input,
        IGpuBuffer gradInput, IGpuBuffer gradHInit, IGpuBuffer gradCInit,
        IGpuBuffer gradWeightsIh, IGpuBuffer gradWeightsHh, IGpuBuffer gradBiasIh, IGpuBuffer gradBiasHh,
        int seqLen, int batch, int inputSize, int hiddenSize)
    {
        ThrowIfDisposed();
        if (seqLen <= 0 || batch <= 0 || inputSize <= 0 || hiddenSize <= 0) return;
        int stateSize = checked(batch * hiddenSize);
        using var nextHidden = AllocateBuffer(stateSize);
        using var nextCell = AllocateBuffer(stateSize);
        DispatchResidentMetal("lstm_backward_sequence_serial", 1,
            new[] { gradOutput, allH, allC, cacheGates, weightsIh, weightsHh, input,
                gradInput, gradHInit, gradCInit, gradWeightsIh, gradWeightsHh, gradBiasIh,
                nextHidden, nextCell },
            (uint)seqLen, (uint)batch, (uint)inputSize, (uint)hiddenSize);
        Copy(gradBiasIh, gradBiasHh, checked(4 * hiddenSize));
    }

    /// <summary>
    /// Forward pass for GRU sequence - processes all timesteps in a single kernel launch.
    /// Efficient for BPTT training with minimal kernel launch overhead.
    /// </summary>
    public void GruForwardSequence(
        IGpuBuffer input, IGpuBuffer hInit,
        IGpuBuffer weightsIh, IGpuBuffer weightsHh, IGpuBuffer biasIh, IGpuBuffer biasHh,
        IGpuBuffer output, IGpuBuffer hFinal, IGpuBuffer allH, IGpuBuffer cacheGates,
        int seqLen, int batch, int inputSize, int hiddenSize)
    {
        ThrowIfDisposed();
        if (seqLen <= 0 || batch <= 0 || inputSize <= 0 || hiddenSize <= 0) return;
        int stateSize = checked(batch * hiddenSize);
        Copy(hInit, 0, allH, 0, stateSize);
        DispatchResidentMetal("gru_forward_sequence", batch,
            new[] { input, weightsIh, weightsHh, biasIh, biasHh, output, allH, cacheGates },
            (uint)seqLen, (uint)batch, (uint)inputSize, (uint)hiddenSize);
        Copy(allH, checked(seqLen * stateSize), hFinal, 0, stateSize);
    }

    /// <summary>
    /// Backward pass for GRU sequence - computes gradients via BPTT.
    /// </summary>
    public void GruBackwardSequence(
        IGpuBuffer gradOutput, IGpuBuffer allH, IGpuBuffer cacheGates,
        IGpuBuffer weightsIh, IGpuBuffer weightsHh, IGpuBuffer input,
        IGpuBuffer gradInput, IGpuBuffer gradHInit, IGpuBuffer dHBuffer,
        IGpuBuffer gradWeightsIh, IGpuBuffer gradWeightsHh, IGpuBuffer gradBiasIh, IGpuBuffer gradBiasHh,
        int seqLen, int batch, int inputSize, int hiddenSize)
    {
        ThrowIfDisposed();
        if (seqLen <= 0 || batch <= 0 || inputSize <= 0 || hiddenSize <= 0) return;
        DispatchResidentMetal("gru_backward_sequence_serial", 1,
            new[] { gradOutput, allH, cacheGates, weightsIh, weightsHh, input,
                gradInput, gradHInit, dHBuffer, gradWeightsIh, gradWeightsHh, gradBiasIh },
            (uint)seqLen, (uint)batch, (uint)inputSize, (uint)hiddenSize);
        Copy(gradBiasIh, gradBiasHh, checked(3 * hiddenSize));
    }

    /// <summary>
    /// Backward pass for a single GRU cell - computes gate gradients and gradient to previous hidden state.
    /// </summary>
    public void GruCellBackward(
        IGpuBuffer gradH, IGpuBuffer gateR, IGpuBuffer gateZ, IGpuBuffer gateN, IGpuBuffer prevH,
        IGpuBuffer weightsHh,
        IGpuBuffer gradPrevH, IGpuBuffer gradGateR, IGpuBuffer gradGateZ, IGpuBuffer gradGateN,
        int batch, int hiddenSize)
    {
        ThrowIfDisposed();
        int count = checked(batch * hiddenSize);
        if (count <= 0) return;
        DispatchResidentMetal("gru_cell_backward", count,
            new[] { gradH, gateR, gateZ, gateN, prevH, weightsHh,
                gradPrevH, gradGateR, gradGateZ, gradGateN },
            (uint)batch, (uint)hiddenSize);
    }

    #endregion

    #region Helper Methods for RNN

    /// <summary>
    /// Sigmoid activation function for RNN gate computations.
    /// </summary>
    /// <param name="x">Input value.</param>
    /// <returns>Sigmoid of x: 1 / (1 + exp(-x)).</returns>
    private static float SigmoidActivation(float x)
    {
        // Use numerically stable sigmoid implementation
        if (x >= 0)
        {
            float expNegX = (float)Math.Exp(-x);
            return 1.0f / (1.0f + expNegX);
        }
        else
        {
            float expX = (float)Math.Exp(x);
            return expX / (1.0f + expX);
        }
    }

    #endregion
}
