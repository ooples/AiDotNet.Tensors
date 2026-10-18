using System;
using System.Collections.Generic;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA;

// One block per batch row holds every hidden unit, so the sequence kernels need H <= MaxRnnBlockSize.
public sealed partial class CudaBackend : IFusedLstmSequenceTraining, ILstmLastStepTraining
{
    bool ILstmLastStepTraining.SupportsLstmLastStepBackward(int hiddenSize)
        => TryGetLstmRegisterRecurrence("backward_last", hiddenSize, out _) && _kernelCache.ContainsKey("lstm_backward_gemms");

    void ILstmLastStepTraining.LstmBackwardLastStep(IGpuBuffer gradLastHidden, IGpuBuffer allH, IGpuBuffer allC,
        IGpuBuffer cacheGates, IGpuBuffer hInit, IGpuBuffer cInit, IGpuBuffer weightsIh, IGpuBuffer weightsHh,
        IGpuBuffer input, IGpuBuffer gradInput, IGpuBuffer gradHInit, IGpuBuffer gradCInit, IGpuBuffer gradWeightsIh,
        IGpuBuffer gradWeightsHh, IGpuBuffer gradBiasIh, IGpuBuffer gradBiasHh, int seqLen, int batch, int inputSize,
        int hiddenSize)
    {
        if (!TryGetLstmRegisterRecurrence("backward_last", hiddenSize, out _))
            throw new GpuCapabilityUnavailableException($"CUDA last-step LSTM backward needs hidden size <= 64 (got {hiddenSize}).");
        LstmBackwardViaGemm(IntPtr.Zero, gradLastHidden, allH, allC, cacheGates, hInit, cInit, weightsIh, weightsHh, input,
            gradInput, gradHInit, gradCInit, gradWeightsIh, gradWeightsHh, gradBiasIh, gradBiasHh,
            seqLen, batch, inputSize, hiddenSize, lastStepOnly: true);
    }

    int IFusedLstmSequenceTraining.MaxFusedLstmHidden => MaxRnnBlockSize;
    bool IFusedLstmSequenceTraining.LstmBackwardOverwritesGradients => true;

    // Scratch for the GEMM-form LSTM backward, per (batch*seqLen, hidden). Kept for the backend's lifetime so a
    // captured CUDA graph never allocates and its baked pointers stay valid across replays.
    private sealed class LstmBackwardScratch
    {
        public LstmBackwardScratch(CudaBackend backend, int rows, int hidden)
        {
            DGates = backend.AllocateBuffer(rows * 4 * hidden);
            HPrev = backend.AllocateBuffer(rows * hidden);
        }

        public IGpuBuffer DGates { get; }
        public IGpuBuffer HPrev { get; }
        /// <summary>lstm_backward_gemms' split partials, sized on first use (stable afterwards, for captured graphs).</summary>
        public IGpuBuffer? Partials { get; set; }
    }

    private readonly Dictionary<(int Rows, int Hidden), LstmBackwardScratch> _lstmBackwardScratch = new();

    // Forward scratch per (batch*seqLen, hidden): the input projection and the transposed recurrent weights.
    private sealed class LstmForwardScratch
    {
        public LstmForwardScratch(CudaBackend backend, int rows, int hidden)
        {
            XProj = backend.AllocateBuffer(rows * 4 * hidden);
            WhT = backend.AllocateBuffer(4 * hidden * hidden);
        }

        public IGpuBuffer XProj { get; }
        public IGpuBuffer WhT { get; }
    }

    private readonly Dictionary<(int Rows, int Hidden), LstmForwardScratch> _lstmForwardScratch = new();

    /// <summary>
    /// The register-resident recurrence kernel for this hidden size (<c>lstm_{direction}_recurrence_reg_h*_b1</c>: one
    /// batch row per block, 4H threads), or false when H is above what a thread can hold in registers (64). One row per
    /// block measured fastest on the parity LSTM (B 32, H 64, RTX 3080): forward 20.8 / 22.2 / 28.6 us and backward
    /// 15.1 / 21.5 / 37.2 us for 1 / 2 / 4 rows per block.
    /// </summary>
    private bool TryGetLstmRegisterRecurrence(string direction, int hiddenSize, out IntPtr kernel)
    {
        kernel = IntPtr.Zero;
        int padded = hiddenSize <= 32 ? 32 : hiddenSize <= 64 ? 64 : 0;
        return padded != 0 && _kernelCache.TryGetValue($"lstm_{direction}_recurrence_reg_h{padded}_b1", out kernel);
    }

    /// <summary>
    /// The forward as cuDNN structures it: <c>xProj = X·Wiᵀ</c> for every batch*seqLen row in one GEMM, the recurrent
    /// weights transposed once, then <c>lstm_forward_recurrence</c> for the sequential part. Same outputs and caches
    /// as <c>lstm_forward_sequence</c>.
    /// </summary>
    /// <returns>True when the recurrence also wrote <paramref name="hFinal"/> and <paramref name="cFinal"/>.</returns>
    private unsafe bool LstmForwardViaGemm(IntPtr recurrence,
        IGpuBuffer input, IGpuBuffer hInit, IGpuBuffer cInit,
        IGpuBuffer weightsIh, IGpuBuffer weightsHh, IGpuBuffer biasIh, IGpuBuffer biasHh,
        IGpuBuffer output, IGpuBuffer allH, IGpuBuffer allC, IGpuBuffer cacheGates, IGpuBuffer hFinal, IGpuBuffer cFinal,
        int seqLen, int batch, int inputSize, int hiddenSize)
    {
        int rows = checked(batch * seqLen);
        int gates = 4 * hiddenSize;
        LstmForwardScratch scratch;
        lock (_lstmForwardScratch)
        {
            if (_lstmForwardScratch.TryGetValue((rows, hiddenSize), out var cached))
            {
                scratch = cached;
            }
            else
            {
                scratch = new LstmForwardScratch(this, rows, hiddenSize);
                _lstmForwardScratch[(rows, hiddenSize)] = scratch;
            }
        }

        // xProj [R, 4H] = X [R, in] · Wiᵀ (Wi is [4H, in]). The register-resident kernel reads Wh [4H, H] as it is;
        // the other reads WhT [H, 4H] = Whᵀ.
        MatMulTransposed(input, weightsIh, scratch.XProj, rows, gates, inputSize);
        bool register = TryGetLstmRegisterRecurrence("forward", hiddenSize, out var regKernel);
        if (!register) Transpose(weightsHh, scratch.WhT, gates, hiddenSize);

        using (PushContext())
        {
            IntPtr xProjPtr = scratch.XProj.Handle, hInitPtr = hInit.Handle, cInitPtr = cInit.Handle;
            IntPtr whTPtr = register ? weightsHh.Handle : scratch.WhT.Handle, bIhPtr = biasIh.Handle, bHhPtr = biasHh.Handle;
            IntPtr outPtr = output.Handle, allHPtr = allH.Handle, allCPtr = allC.Handle, gatesPtr = cacheGates.Handle;
            IntPtr hFinalPtr = hFinal.Handle, cFinalPtr = cFinal.Handle;
            void** args = stackalloc void*[15];
            args[0] = &xProjPtr;
            args[1] = &hInitPtr;
            args[2] = &cInitPtr;
            args[3] = &whTPtr;
            args[4] = &bIhPtr;
            args[5] = &bHhPtr;
            args[6] = &outPtr;
            args[7] = &allHPtr;
            args[8] = &allCPtr;
            args[9] = &gatesPtr;
            args[10] = &batch;
            args[11] = &seqLen;
            args[12] = &hiddenSize;
            args[13] = &hFinalPtr;   // read by the register kernel only
            args[14] = &cFinalPtr;
            if (register)
                LaunchKernel(regKernel, (uint)batch, (uint)gates, args);
            else
                LaunchKernelWithSharedMem(recurrence, (uint)batch, (uint)hiddenSize, (uint)(hiddenSize * sizeof(float)), args);
        }
        return register;
    }

    private LstmBackwardScratch GetLstmBackwardScratch(int rows, int hidden)
    {
        lock (_lstmBackwardScratch)
        {
            if (!_lstmBackwardScratch.TryGetValue((rows, hidden), out var scratch))
            {
                scratch = new LstmBackwardScratch(this, rows, hidden);
                _lstmBackwardScratch[(rows, hidden)] = scratch;
            }
            return scratch;
        }
    }

    /// <summary>
    /// BPTT as cuDNN structures it: <c>lstm_backward_recurrence</c> runs only the sequential part (gate gradients and
    /// dH_prev per timestep), then the weight, bias and input gradients are dense products over all batch*seqLen rows.
    /// Every gradient buffer is written, not accumulated into (<see cref="IFusedLstmSequenceTraining.LstmBackwardOverwritesGradients"/>).
    /// </summary>
    private unsafe void LstmBackwardViaGemm(IntPtr recurrence,
        IGpuBuffer gradOutput, IGpuBuffer allH, IGpuBuffer allC, IGpuBuffer cacheGates,
        IGpuBuffer hInit, IGpuBuffer cInit, IGpuBuffer weightsIh, IGpuBuffer weightsHh, IGpuBuffer input,
        IGpuBuffer gradInput, IGpuBuffer gradHInit, IGpuBuffer gradCInit,
        IGpuBuffer gradWeightsIh, IGpuBuffer gradWeightsHh, IGpuBuffer gradBiasIh, IGpuBuffer gradBiasHh,
        int seqLen, int batch, int inputSize, int hiddenSize, bool lastStepOnly = false)
    {
        int rows = checked(batch * seqLen);
        int gates = 4 * hiddenSize;
        var scratch = GetLstmBackwardScratch(rows, hiddenSize);

        using (PushContext())
        {
            IntPtr gradOutputPtr = gradOutput.Handle, allHPtr = allH.Handle, allCPtr = allC.Handle;
            IntPtr gatesPtr = cacheGates.Handle, cInitPtr = cInit.Handle, hInitPtr = hInit.Handle, whPtr = weightsHh.Handle;
            IntPtr dGatesPtr = scratch.DGates.Handle, hPrevPtr = scratch.HPrev.Handle;
            IntPtr dHInitPtr = gradHInit.Handle, dCInitPtr = gradCInit.Handle;
            void** args = stackalloc void*[14];
            args[0] = &gradOutputPtr;
            args[1] = &allHPtr;
            args[2] = &allCPtr;
            args[3] = &gatesPtr;
            args[4] = &cInitPtr;
            args[5] = &hInitPtr;
            args[6] = &whPtr;
            args[7] = &dGatesPtr;
            args[8] = &hPrevPtr;
            args[9] = &dHInitPtr;
            args[10] = &dCInitPtr;
            args[11] = &batch;
            args[12] = &seqLen;
            args[13] = &hiddenSize;
            // lastStepOnly: gradOutput is the final hidden state's gradient [batch, hidden] (the caller checked the kernel).
            if (TryGetLstmRegisterRecurrence(lastStepOnly ? "backward_last" : "backward", hiddenSize, out var regKernel))
                LaunchKernel(regKernel, (uint)batch, (uint)gates, args);
            else
                LaunchKernelWithSharedMem(recurrence, (uint)batch, (uint)hiddenSize, (uint)(gates * sizeof(float)), args);
        }

        // [dWi | dWh | dBias] = dGatesᵀ · [X | Hprev | 1] and dX = dGates · Wi in one launch (lstm_backward_gemms); both
        // biases are summed into the gates, so each receives the column sums of dGates. The recurrence used to store a
        // transposed copy of dGates for these products as well: one scattered store per gate per step on its
        // critical path. The kernel is in the recurrence kernels' module, so present whenever the caller found one.
        if (!_kernelCache.TryGetValue("lstm_backward_gemms", out var gemms))
            throw new GpuCapabilityUnavailableException("CUDA kernel not found: lstm_backward_gemms");
        using (PushContext())
        {
            const int Tile = 32, MinRowsPerSplit = 512;
            int wTilesJ = (inputSize + hiddenSize + 1 + Tile - 1) / Tile;
            int nW = (gates + Tile - 1) / Tile * wTilesJ;
            int nX = (rows + Tile - 1) / Tile * ((inputSize + Tile - 1) / Tile);
            // Enough weight-gradient blocks for about two per SM, each reducing at least MinRowsPerSplit rows.
            int splits = Math.Max(1, Math.Min((2 * Math.Max(1, ComputeUnits) + nW - 1) / nW, rows / MinRowsPerSplit));
            int rowsPerSplit = (rows + splits - 1) / splits;
            rowsPerSplit = (rowsPerSplit + 63) / 64 * 64;   // whole 64-row slices
            splits = (rows + rowsPerSplit - 1) / rowsPerSplit;
            IntPtr partialPtr = IntPtr.Zero;
            if (splits > 1)
            {
                int partialFloats = checked(splits * nW * Tile * Tile);
                if (scratch.Partials is null || scratch.Partials.Size < partialFloats)
                {
                    scratch.Partials?.Dispose();
                    scratch.Partials = AllocateBuffer(partialFloats);
                }
                partialPtr = scratch.Partials.Handle;
            }
            IntPtr dGatesPtr = scratch.DGates.Handle, xPtr = input.Handle, hPrevPtr = scratch.HPrev.Handle;
            IntPtr wiPtr = weightsIh.Handle, dWiPtr = gradWeightsIh.Handle, dWhPtr = gradWeightsHh.Handle;
            IntPtr dBiPtr = gradBiasIh.Handle, dBhPtr = gradBiasHh.Handle, dXPtr = gradInput.Handle;
            void** args = stackalloc void*[17];
            args[0] = &dGatesPtr;
            args[1] = &xPtr;
            args[2] = &hPrevPtr;
            args[3] = &wiPtr;
            args[4] = &dWiPtr;
            args[5] = &dWhPtr;
            args[6] = &dBiPtr;
            args[7] = &dBhPtr;
            args[8] = &dXPtr;
            args[9] = &rows;
            args[10] = &inputSize;
            args[11] = &hiddenSize;
            args[12] = &wTilesJ;
            args[13] = &nW;
            args[14] = &splits;
            args[15] = &rowsPerSplit;
            args[16] = &partialPtr;
            LaunchKernel(gemms, (uint)(nW * splits + nX), 512, args);   // LBG_THREADS
            if (splits > 1)
            {
                if (!_kernelCache.TryGetValue("lstm_backward_gemms_reduce", out var reduce))
                    throw new GpuCapabilityUnavailableException("CUDA kernel not found: lstm_backward_gemms_reduce");
                int elements = gates * (inputSize + hiddenSize + 1);
                void** reduceArgs = stackalloc void*[10];
                reduceArgs[0] = &partialPtr; reduceArgs[1] = &dWiPtr; reduceArgs[2] = &dWhPtr;
                reduceArgs[3] = &dBiPtr; reduceArgs[4] = &dBhPtr; reduceArgs[5] = &splits; reduceArgs[6] = &nW;
                reduceArgs[7] = &wTilesJ; reduceArgs[8] = &inputSize; reduceArgs[9] = &hiddenSize;
                LaunchKernel(reduce, (uint)((elements + 255) / 256), 256, reduceArgs);
            }
        }
    }
}
