// Copyright (c) CrowdTrainer LLC. All rights reserved.
using System;
using System.Collections.Generic;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA;

/// <summary>
/// The fused GRU sequence on CUDA (#1573), structured as the LSTM's (see <c>CudaBackend.FusedLstm.cs</c>): the input
/// projection as one GEMM, the recurrence as one register-resident kernel launch (<c>gru_forward_recurrence_reg_h*</c>),
/// and the backward as one recurrence launch plus the dense products over every batch*seqLen row (three GEMMs and a
/// column-sum kernel). The portable path (<see cref="PortableGruSequence"/>) launched two kernels per time step each way;
/// on the parity GRU (B 32, T 16, In 32, H 64, RTX 3080) the captured training step took 0.78 ms against PyTorch's
/// 0.10 ms. The forward caches keep the portable layout, so the two paths' forwards and backwards interoperate.
/// PyTorch's gate convention (reset after the recurrent sum) only, and hidden sizes up to 64 (a thread's register row).
/// </summary>
public sealed partial class CudaBackend : IFusedGruSequence
{
    private sealed class GruFusedScratch
    {
        public GruFusedScratch(CudaBackend backend, int rows, int hidden)
        {
            XProj = backend.AllocateBuffer(checked(rows * 3 * hidden));
            DGi = backend.AllocateBuffer(checked(rows * 3 * hidden));
            DGh = backend.AllocateBuffer(checked(rows * 3 * hidden));
            HPrev = backend.AllocateBuffer(checked(rows * hidden));
        }

        public IGpuBuffer XProj { get; }
        public IGpuBuffer DGi { get; }
        public IGpuBuffer DGh { get; }
        public IGpuBuffer HPrev { get; }
    }

    private readonly Dictionary<(int Rows, int Hidden), GruFusedScratch> _gruFusedScratch = new();

    private GruFusedScratch GetGruFusedScratch(int rows, int hidden)
    {
        lock (_gruFusedScratch)
        {
            if (!_gruFusedScratch.TryGetValue((rows, hidden), out var scratch))
            {
                scratch = new GruFusedScratch(this, rows, hidden);
                _gruFusedScratch[(rows, hidden)] = scratch;
            }
            return scratch;
        }
    }

    /// <summary>Fused forward / backward runs, process-wide: a test probe that a shape took the fused kernels.</summary>
    internal static long FusedGruForwardRuns, FusedGruBackwardRuns;

    /// <summary>A/B and test switch: false sends every GRU sequence to the portable kernels.</summary>
    internal static bool UseFusedGru = Environment.GetEnvironmentVariable("AIDOTNET_CUDA_FUSED_GRU") != "0";

    private bool TryGetGruKernels(int hiddenSize, out IntPtr forward, out IntPtr backward, out IntPtr bias)
    {
        forward = backward = bias = IntPtr.Zero;
        int padded = hiddenSize <= 32 ? 32 : hiddenSize <= 64 ? 64 : 0;
        return UseFusedGru && padded != 0
            && _kernelCache.TryGetValue($"gru_forward_recurrence_reg_h{padded}", out forward)
            && _kernelCache.TryGetValue($"gru_backward_recurrence_reg_h{padded}", out backward)
            && _kernelCache.TryGetValue("gru_bias_grad_columns", out bias);
    }

    /// <summary>
    /// The GRU forward (reset gate after the recurrent sum) with the portable caches: <paramref name="allH"/>
    /// [(T + 1), B, H], <paramref name="gates"/> [T, B, 3H] (r, z, n) and <paramref name="ghn"/> [T, B, H]. False, having
    /// launched nothing, when the fused kernels do not cover this hidden size.
    /// </summary>
    void IFusedGruSequence.GruForwardSequence(IGpuBuffer input, IGpuBuffer hInit, IGpuBuffer weightsIh, IGpuBuffer weightsHh,
        IGpuBuffer biasIh, IGpuBuffer biasHh, IGpuBuffer output, IGpuBuffer allH, IGpuBuffer gates, IGpuBuffer ghn,
        int seqLen, int batch, int inputSize, int hiddenSize)
    {
        if (!TryGruForwardFused(input, hInit, weightsIh, weightsHh, biasIh, biasHh, output, allH, gates, ghn, seqLen, batch, inputSize, hiddenSize))
            PortableGruSequence.ForwardResetAfter(this, input, hInit, weightsIh, weightsHh, biasIh, biasHh, output, allH, gates, ghn, seqLen, batch, inputSize, hiddenSize);
    }

    void IFusedGruSequence.GruBackwardSequence(IGpuBuffer gradOutput, IGpuBuffer allH, IGpuBuffer gates, IGpuBuffer ghn,
        IGpuBuffer weightsIh, IGpuBuffer weightsHh, IGpuBuffer input, IGpuBuffer gradInput, IGpuBuffer gradHInit,
        IGpuBuffer gradWeightsIh, IGpuBuffer gradWeightsHh, IGpuBuffer gradBiasIh, IGpuBuffer gradBiasHh,
        int seqLen, int batch, int inputSize, int hiddenSize)
    {
        if (!TryGruBackwardFused(gradOutput, allH, gates, ghn, weightsIh, weightsHh, input, gradInput, gradHInit, gradWeightsIh, gradWeightsHh, gradBiasIh, gradBiasHh, seqLen, batch, inputSize, hiddenSize))
            PortableGruSequence.BackwardResetAfter(this, gradOutput, allH, gates, ghn, weightsIh, weightsHh, input, gradInput, gradHInit, gradWeightsIh, gradWeightsHh, gradBiasIh, gradBiasHh, seqLen, batch, inputSize, hiddenSize);
    }

    private unsafe bool TryGruForwardFused(IGpuBuffer input, IGpuBuffer hInit, IGpuBuffer weightsIh, IGpuBuffer weightsHh,
        IGpuBuffer biasIh, IGpuBuffer biasHh, IGpuBuffer output, IGpuBuffer allH, IGpuBuffer gates, IGpuBuffer ghn,
        int seqLen, int batch, int inputSize, int hiddenSize)
    {
        if (!TryGetGruKernels(hiddenSize, out var forward, out _, out _)) return false;
        int rows = checked(batch * seqLen), g3 = 3 * hiddenSize;
        var scratch = GetGruFusedScratch(rows, hiddenSize);
        // xProj [R, 3H] = X [R, In] · Wihᵀ (Wih is [3H, In]); the biases are added in the recurrence.
        MatMulTransposed(input, weightsIh, scratch.XProj, rows, g3, inputSize);
        using (PushContext())
        {
            IntPtr xProjPtr = scratch.XProj.Handle, hInitPtr = hInit.Handle, whPtr = weightsHh.Handle;
            IntPtr bIhPtr = biasIh.Handle, bHhPtr = biasHh.Handle, outPtr = output.Handle;
            IntPtr allHPtr = allH.Handle, gatesPtr = gates.Handle, ghnPtr = ghn.Handle;
            void** args = stackalloc void*[12];
            args[0] = &xProjPtr;
            args[1] = &hInitPtr;
            args[2] = &whPtr;
            args[3] = &bIhPtr;
            args[4] = &bHhPtr;
            args[5] = &outPtr;
            args[6] = &allHPtr;
            args[7] = &gatesPtr;
            args[8] = &ghnPtr;
            args[9] = &batch;
            args[10] = &seqLen;
            args[11] = &hiddenSize;
            LaunchKernel(forward, (uint)batch, (uint)g3, args);
        }
        System.Threading.Interlocked.Increment(ref FusedGruForwardRuns);
        return true;
    }

    /// <summary>
    /// The GRU backward over the forward's caches (portable layout, from either forward). Every gradient output is
    /// written, not accumulated into. False, having launched nothing, when the fused kernels do not cover this hidden size.
    /// </summary>
    private unsafe bool TryGruBackwardFused(IGpuBuffer gradOutput, IGpuBuffer allH, IGpuBuffer gates, IGpuBuffer ghn,
        IGpuBuffer weightsIh, IGpuBuffer weightsHh, IGpuBuffer input, IGpuBuffer gradInput, IGpuBuffer gradHInit,
        IGpuBuffer gradWeightsIh, IGpuBuffer gradWeightsHh, IGpuBuffer gradBiasIh, IGpuBuffer gradBiasHh,
        int seqLen, int batch, int inputSize, int hiddenSize)
    {
        if (!TryGetGruKernels(hiddenSize, out _, out var backward, out var bias)) return false;
        int rows = checked(batch * seqLen), g3 = 3 * hiddenSize;
        var scratch = GetGruFusedScratch(rows, hiddenSize);
        using (PushContext())
        {
            IntPtr gyPtr = gradOutput.Handle, allHPtr = allH.Handle, gatesPtr = gates.Handle, ghnPtr = ghn.Handle;
            IntPtr whPtr = weightsHh.Handle, dgiPtr = scratch.DGi.Handle, dghPtr = scratch.DGh.Handle;
            IntPtr hPrevPtr = scratch.HPrev.Handle, dh0Ptr = gradHInit.Handle;
            void** args = stackalloc void*[12];
            args[0] = &gyPtr;
            args[1] = &allHPtr;
            args[2] = &gatesPtr;
            args[3] = &ghnPtr;
            args[4] = &whPtr;
            args[5] = &dgiPtr;
            args[6] = &dghPtr;
            args[7] = &hPrevPtr;
            args[8] = &dh0Ptr;
            args[9] = &batch;
            args[10] = &seqLen;
            args[11] = &hiddenSize;
            LaunchKernel(backward, (uint)batch, (uint)g3, args);
        }

        // dX [R, In] = dgi [R, 3H] · Wih [3H, In]; dWih [3H, In] = dgiᵀ · X; dWhh [3H, H] = dghᵀ · hPrev.
        Gemm(scratch.DGi, weightsIh, gradInput, rows, inputSize, g3);
        MatMulTransposedA(scratch.DGi, input, gradWeightsIh, g3, inputSize, rows);
        MatMulTransposedA(scratch.DGh, scratch.HPrev, gradWeightsHh, g3, hiddenSize, rows);
        using (PushContext())
        {
            IntPtr dgiPtr = scratch.DGi.Handle, dghPtr = scratch.DGh.Handle;
            IntPtr gbiPtr = gradBiasIh.Handle, gbhPtr = gradBiasHh.Handle;
            void** args = stackalloc void*[6];
            args[0] = &dgiPtr;
            args[1] = &dghPtr;
            args[2] = &gbiPtr;
            args[3] = &gbhPtr;
            args[4] = &rows;
            args[5] = &g3;
            const int ColumnsPerBlock = 8;   // one warp per column, 256 threads
            LaunchKernel(bias, (uint)((g3 + ColumnsPerBlock - 1) / ColumnsPerBlock), 256, args);
        }
        System.Threading.Interlocked.Increment(ref FusedGruBackwardRuns);
        return true;
    }
}
