using System;
using System.Collections.Generic;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA;

// One block per batch row holds every hidden unit, so the sequence kernels need H <= MaxRnnBlockSize.
public sealed partial class CudaBackend : IFusedLstmSequenceTraining
{
    int IFusedLstmSequenceTraining.MaxFusedLstmHidden => MaxRnnBlockSize;

    // Scratch for the GEMM-form LSTM backward, per (batch*seqLen, hidden). Kept for the backend's lifetime so a
    // captured CUDA graph never allocates and its baked pointers stay valid across replays.
    private sealed class LstmBackwardScratch
    {
        public LstmBackwardScratch(CudaBackend backend, int rows, int hidden)
        {
            int gates = 4 * hidden;
            DGates = backend.AllocateBuffer(rows * gates);
            DGatesT = backend.AllocateBuffer(rows * gates);
            HPrev = backend.AllocateBuffer(rows * hidden);
            BiasSum = backend.AllocateBuffer(gates);
        }

        public IGpuBuffer DGates { get; }
        public IGpuBuffer DGatesT { get; }
        public IGpuBuffer HPrev { get; }
        public IGpuBuffer BiasSum { get; }
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
    /// The forward as cuDNN structures it: <c>xProj = X·Wiᵀ</c> for every batch*seqLen row in one GEMM, the recurrent
    /// weights transposed once, then <c>lstm_forward_recurrence</c> for the sequential part. Same outputs and caches
    /// as <c>lstm_forward_sequence</c>.
    /// </summary>
    private unsafe void LstmForwardViaGemm(IntPtr recurrence,
        IGpuBuffer input, IGpuBuffer hInit, IGpuBuffer cInit,
        IGpuBuffer weightsIh, IGpuBuffer weightsHh, IGpuBuffer biasIh, IGpuBuffer biasHh,
        IGpuBuffer output, IGpuBuffer allH, IGpuBuffer allC, IGpuBuffer cacheGates,
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

        // xProj [R, 4H] = X [R, in] · Wiᵀ (Wi is [4H, in]);  WhT [H, 4H] = Whᵀ (Wh is [4H, H]).
        MatMulTransposed(input, weightsIh, scratch.XProj, rows, gates, inputSize);
        Transpose(weightsHh, scratch.WhT, gates, hiddenSize);

        using (PushContext())
        {
            IntPtr xProjPtr = scratch.XProj.Handle, hInitPtr = hInit.Handle, cInitPtr = cInit.Handle;
            IntPtr whTPtr = scratch.WhT.Handle, bIhPtr = biasIh.Handle, bHhPtr = biasHh.Handle;
            IntPtr outPtr = output.Handle, allHPtr = allH.Handle, allCPtr = allC.Handle, gatesPtr = cacheGates.Handle;
            void** args = stackalloc void*[13];
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
            LaunchKernelWithSharedMem(recurrence, (uint)batch, (uint)hiddenSize, (uint)(hiddenSize * sizeof(float)), args);
        }
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
    /// Accumulates into the gradient buffers, as the single-kernel form did, so callers keep zeroing them first.
    /// </summary>
    private unsafe void LstmBackwardViaGemm(IntPtr recurrence,
        IGpuBuffer gradOutput, IGpuBuffer allH, IGpuBuffer allC, IGpuBuffer cacheGates,
        IGpuBuffer hInit, IGpuBuffer cInit, IGpuBuffer weightsIh, IGpuBuffer weightsHh, IGpuBuffer input,
        IGpuBuffer gradInput, IGpuBuffer gradHInit, IGpuBuffer gradCInit,
        IGpuBuffer gradWeightsIh, IGpuBuffer gradWeightsHh, IGpuBuffer gradBiasIh, IGpuBuffer gradBiasHh,
        int seqLen, int batch, int inputSize, int hiddenSize)
    {
        int rows = checked(batch * seqLen);
        int gates = 4 * hiddenSize;
        var scratch = GetLstmBackwardScratch(rows, hiddenSize);

        using (PushContext())
        {
            IntPtr gradOutputPtr = gradOutput.Handle, allHPtr = allH.Handle, allCPtr = allC.Handle;
            IntPtr gatesPtr = cacheGates.Handle, cInitPtr = cInit.Handle, hInitPtr = hInit.Handle, whPtr = weightsHh.Handle;
            IntPtr dGatesPtr = scratch.DGates.Handle, dGatesTPtr = scratch.DGatesT.Handle, hPrevPtr = scratch.HPrev.Handle;
            IntPtr dHInitPtr = gradHInit.Handle, dCInitPtr = gradCInit.Handle;
            void** args = stackalloc void*[15];
            args[0] = &gradOutputPtr;
            args[1] = &allHPtr;
            args[2] = &allCPtr;
            args[3] = &gatesPtr;
            args[4] = &cInitPtr;
            args[5] = &hInitPtr;
            args[6] = &whPtr;
            args[7] = &dGatesPtr;
            args[8] = &dGatesTPtr;
            args[9] = &hPrevPtr;
            args[10] = &dHInitPtr;
            args[11] = &dCInitPtr;
            args[12] = &batch;
            args[13] = &seqLen;
            args[14] = &hiddenSize;
            LaunchKernelWithSharedMem(recurrence, (uint)batch, (uint)hiddenSize, (uint)(gates * sizeof(float)), args);
        }

        // dWi += dGatesT [4H, R] · X [R, in];  dWh += dGatesT · Hprev [R, H];  dX += dGates [R, 4H] · Wi [4H, in].
        Gemm(scratch.DGatesT, input, gradWeightsIh, gates, inputSize, rows, 1f, 1f);
        Gemm(scratch.DGatesT, scratch.HPrev, gradWeightsHh, gates, hiddenSize, rows, 1f, 1f);
        Gemm(scratch.DGates, weightsIh, gradInput, rows, inputSize, gates, 1f, 1f);

        // Both biases are summed into the gates, so each receives the row sums of dGatesT.
        SumAxis(scratch.DGatesT, scratch.BiasSum, gates, rows);
        Add(gradBiasIh, scratch.BiasSum, gradBiasIh, gates);
        Add(gradBiasHh, scratch.BiasSum, gradBiasHh, gates);
    }
}
