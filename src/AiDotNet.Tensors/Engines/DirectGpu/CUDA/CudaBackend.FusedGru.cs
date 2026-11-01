// Copyright (c) CrowdTrainer LLC. All rights reserved.
using System;
using System.Collections.Generic;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA;

/// <summary>
/// The fused GRU sequence on CUDA (#1573), structured as the LSTM's (see <c>CudaBackend.FusedLstm.cs</c>): the input
/// projection as one GEMM, the recurrence as one register-resident kernel launch (<c>gru_forward_recurrence_reg_h*</c>),
/// and the backward as one recurrence launch plus one launch for every dense product over the batch*seqLen rows
/// (<c>gru_backward_gemms</c>). The portable path (<see cref="PortableGruSequence"/>) launched two kernels per time step each way;
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

        // gru_backward_gemms' split partials and per-tile arrival counters, grown on demand.
        public IGpuBuffer? Partials { get; set; }
        public IGpuBuffer? Counters { get; set; }

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

    private bool TryGetGruKernels(int hiddenSize, bool lastStepOnly, out IntPtr forward, out IntPtr backward, out IntPtr gemms)
    {
        forward = backward = gemms = IntPtr.Zero;
        int padded = hiddenSize <= 32 ? 32 : hiddenSize <= 64 ? 64 : 0;
        string backwardName = lastStepOnly ? "gru_backward_last_recurrence_reg_h" : "gru_backward_recurrence_reg_h";
        return UseFusedGru && padded != 0
            && _kernelCache.TryGetValue($"gru_forward_recurrence_reg_h{padded}", out forward)
            && _kernelCache.TryGetValue(backwardName + padded, out backward)
            && _kernelCache.TryGetValue("gru_backward_gemms", out gemms);
    }

    /// <summary>
    /// The GRU forward (reset gate after the recurrent sum) with the portable caches: <paramref name="allH"/>
    /// [(T + 1), B, H], <paramref name="gates"/> [T, B, 3H] (r, z, n) and <paramref name="ghn"/> [T, B, H]; the fused
    /// kernels where they cover the hidden size, the portable ones otherwise.
    /// </summary>
    void IFusedGruSequence.GruForwardSequence(IGpuBuffer input, IGpuBuffer hInit, IGpuBuffer weightsIh, IGpuBuffer weightsHh,
        IGpuBuffer biasIh, IGpuBuffer biasHh, IGpuBuffer output, IGpuBuffer allH, IGpuBuffer gates, IGpuBuffer ghn,
        int seqLen, int batch, int inputSize, int hiddenSize, IGpuBuffer? hFinal)
    {
        if (!TryGruForwardFused(input, hInit, weightsIh, weightsHh, biasIh, biasHh, output, allH, gates, ghn, seqLen, batch, inputSize, hiddenSize, hFinal))
            PortableGruSequence.ForwardResetAfter(this, input, hInit, weightsIh, weightsHh, biasIh, biasHh, output, allH, gates, ghn, seqLen, batch, inputSize, hiddenSize, hFinal);
    }

    void IFusedGruSequence.GruBackwardSequence(IGpuBuffer gradOutput, IGpuBuffer allH, IGpuBuffer gates, IGpuBuffer ghn,
        IGpuBuffer weightsIh, IGpuBuffer weightsHh, IGpuBuffer input, IGpuBuffer gradInput, IGpuBuffer gradHInit,
        IGpuBuffer gradWeightsIh, IGpuBuffer gradWeightsHh, IGpuBuffer gradBiasIh, IGpuBuffer gradBiasHh,
        int seqLen, int batch, int inputSize, int hiddenSize, bool lastStepOnly)
    {
        if (!TryGruBackwardFused(gradOutput, allH, gates, ghn, weightsIh, weightsHh, input, gradInput, gradHInit, gradWeightsIh, gradWeightsHh, gradBiasIh, gradBiasHh, seqLen, batch, inputSize, hiddenSize, lastStepOnly))
            PortableGruSequence.BackwardResetAfter(this, gradOutput, allH, gates, ghn, weightsIh, weightsHh, input, gradInput, gradHInit, gradWeightsIh, gradWeightsHh, gradBiasIh, gradBiasHh, seqLen, batch, inputSize, hiddenSize, lastStepOnly);
    }

    private unsafe bool TryGruForwardFused(IGpuBuffer input, IGpuBuffer hInit, IGpuBuffer weightsIh, IGpuBuffer weightsHh,
        IGpuBuffer biasIh, IGpuBuffer biasHh, IGpuBuffer output, IGpuBuffer allH, IGpuBuffer gates, IGpuBuffer ghn,
        int seqLen, int batch, int inputSize, int hiddenSize, IGpuBuffer? hFinal)
    {
        if (!TryGetGruKernels(hiddenSize, false, out var forward, out _, out _)) return false;
        int rows = checked(batch * seqLen), g3 = 3 * hiddenSize;
        var scratch = GetGruFusedScratch(rows, hiddenSize);
        // xProj [R, 3H] = X [R, In] · Wihᵀ (Wih is [3H, In]); the biases are added in the recurrence.
        MatMulTransposed(input, weightsIh, scratch.XProj, rows, g3, inputSize);
        using (PushContext())
        {
            IntPtr xProjPtr = scratch.XProj.Handle, hInitPtr = hInit.Handle, whPtr = weightsHh.Handle;
            IntPtr bIhPtr = biasIh.Handle, bHhPtr = biasHh.Handle, outPtr = output.Handle;
            IntPtr allHPtr = allH.Handle, gatesPtr = gates.Handle, ghnPtr = ghn.Handle;
            IntPtr hFinalPtr = hFinal?.Handle ?? IntPtr.Zero;
            void** args = stackalloc void*[13];
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
            args[12] = &hFinalPtr;
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
        int seqLen, int batch, int inputSize, int hiddenSize, bool lastStepOnly)
    {
        if (!TryGetGruKernels(hiddenSize, lastStepOnly, out _, out var backward, out var gemms)) return false;
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

        // dWih = dgiᵀ X, dWhh = dghᵀ hPrev, both bias gradients and dX = dgi Wih in one launch (gru_backward_gemms,
        // lstm_backward_gemms' scheme): three cuBLAS GEMMs with their split-K combines and a column-sum kernel cost about
        // 22 us of a 137 us captured step at the parity GRU's sizes.
        using (PushContext())
        {
            // One wave of 512-thread blocks: as many row splits of the weight-gradient tiles as fit beside the dX tiles,
            // each split at least two 64-row slices; the last block of a tile sums its splits (see lstm_backward_gemms).
            const int Tile = 32, MinRowsPerSplit = 128;
            int wTilesX = (inputSize + Tile - 1) / Tile;
            int wTilesJ = wTilesX + (hiddenSize + Tile - 1) / Tile;
            int rowTiles = (g3 + Tile - 1) / Tile;
            int nW = rowTiles * wTilesJ;
            int nX = (rows + Tile - 1) / Tile * ((inputSize + Tile - 1) / Tile);
            int oneWave = Math.Max(1, (Math.Max(1, ComputeUnits) - nX) / nW);
            int splits = Math.Max(1, Math.Min(oneWave, rows / MinRowsPerSplit));
            int rowsPerSplit = (rows + splits - 1) / splits;
            rowsPerSplit = (rowsPerSplit + 63) / 64 * 64;
            splits = (rows + rowsPerSplit - 1) / rowsPerSplit;
            IntPtr partialPtr = IntPtr.Zero, countersPtr = IntPtr.Zero;
            if (splits > 1)
            {
                int partialFloats = checked(splits * nW * Tile * Tile + splits * 2 * rowTiles * Tile);
                if (scratch.Partials is null || scratch.Partials.Size < partialFloats)
                {
                    scratch.Partials?.Dispose();
                    scratch.Partials = AllocateBuffer(partialFloats);
                }
                if (scratch.Counters is null || scratch.Counters.Size < nW)
                {
                    scratch.Counters?.Dispose();
                    scratch.Counters = AllocateBuffer(nW);
                    ((IDeviceZeroFill)this).ZeroBuffer(scratch.Counters, nW);
                }
                partialPtr = scratch.Partials.Handle;
                countersPtr = scratch.Counters.Handle;
            }
            IntPtr dgiPtr = scratch.DGi.Handle, dghPtr = scratch.DGh.Handle, xPtr = input.Handle, hPrevPtr = scratch.HPrev.Handle;
            IntPtr wiPtr = weightsIh.Handle, dWiPtr = gradWeightsIh.Handle, dWhPtr = gradWeightsHh.Handle;
            IntPtr dBiPtr = gradBiasIh.Handle, dBhPtr = gradBiasHh.Handle, dXPtr = gradInput.Handle;
            void** args = stackalloc void*[20];
            args[0] = &dgiPtr;
            args[1] = &dghPtr;
            args[2] = &xPtr;
            args[3] = &hPrevPtr;
            args[4] = &wiPtr;
            args[5] = &dWiPtr;
            args[6] = &dWhPtr;
            args[7] = &dBiPtr;
            args[8] = &dBhPtr;
            args[9] = &dXPtr;
            args[10] = &rows;
            args[11] = &inputSize;
            args[12] = &hiddenSize;
            args[13] = &wTilesX;
            args[14] = &wTilesJ;
            args[15] = &nW;
            args[16] = &splits;
            args[17] = &rowsPerSplit;
            args[18] = &partialPtr;
            args[19] = &countersPtr;
            LaunchKernel(gemms, (uint)(nW * splits + nX), 512, args);   // GBG_THREADS
        }
        System.Threading.Interlocked.Increment(ref FusedGruBackwardRuns);
        return true;
    }
}
