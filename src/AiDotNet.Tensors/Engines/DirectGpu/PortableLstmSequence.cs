// Copyright (c) AiDotNet. All rights reserved.
using K = AiDotNet.Tensors.Engines.DirectGpu.BatchExecutionCudaKernels;

namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>
/// <see cref="IDirectGpuBackend.LstmForwardSequence"/> and <see cref="IDirectGpuBackend.LstmBackwardSequence"/> over the
/// shared lstm_p_ kernels (#1107, #1108), meeting the <see cref="IFusedLstmSequenceTraining"/> contract for any hidden
/// size: two launches per forward step, two per backward step, then one each for the input, weight and bias gradients.
/// The caches (allH, allC, gates) are this implementation's own layout, read back only by its backward.
/// </summary>
internal static class PortableLstmSequence
{
    private static void Check(int seqLen, int batch, int inputSize, int hiddenSize)
    {
        if (seqLen <= 0 || batch <= 0 || inputSize <= 0 || hiddenSize <= 0)
            throw new ArgumentOutOfRangeException(nameof(seqLen), "Every LSTM extent must be positive.");
        _ = checked((long)(seqLen + 1) * batch * 4 * hiddenSize + 4L * hiddenSize * (inputSize + hiddenSize));
    }

    public static void Forward(IDirectGpuBackend backend, PortableGenericFft.Launcher launch,
        IGpuBuffer input, IGpuBuffer hInit, IGpuBuffer cInit, IGpuBuffer weightsIh, IGpuBuffer weightsHh,
        IGpuBuffer biasIh, IGpuBuffer biasHh, IGpuBuffer output, IGpuBuffer hFinal, IGpuBuffer cFinal,
        IGpuBuffer allH, IGpuBuffer allC, IGpuBuffer cacheGates, int seqLen, int batch, int inputSize, int hiddenSize)
    {
        Check(seqLen, batch, inputSize, hiddenSize);
        int state = batch * hiddenSize;
        backend.Copy(hInit, 0, allH, 0, state);
        backend.Copy(cInit, 0, allC, 0, state);
        for (int t = 0; t < seqLen; t++)
        {
            launch(K.LstmGates, batch * 4 * hiddenSize, input, weightsIh, weightsHh, biasIh, biasHh, allH, cacheGates,
                t, batch, seqLen, inputSize, hiddenSize);
            launch(K.LstmCell, state, cacheGates, allC, allH, output, t, batch, seqLen, hiddenSize);
        }
        backend.Copy(allH, seqLen * state, hFinal, 0, state);
        backend.Copy(allC, seqLen * state, cFinal, 0, state);
    }

    public static void Backward(IDirectGpuBackend backend, PortableGenericFft.Launcher launch,
        IGpuBuffer gradOutput, IGpuBuffer allH, IGpuBuffer allC, IGpuBuffer cacheGates,
        IGpuBuffer weightsIh, IGpuBuffer weightsHh, IGpuBuffer input,
        IGpuBuffer gradInput, IGpuBuffer gradHInit, IGpuBuffer gradCInit,
        IGpuBuffer gradWeightsIh, IGpuBuffer gradWeightsHh, IGpuBuffer gradBiasIh, IGpuBuffer gradBiasHh,
        int seqLen, int batch, int inputSize, int hiddenSize,
        IGpuBuffer? gradHFinal = null, IGpuBuffer? gradCFinal = null)
    {
        Check(seqLen, batch, inputSize, hiddenSize);
        int state = batch * hiddenSize;
        using var dh = backend.AllocateBuffer(state);
        using var dc = backend.AllocateBuffer(state);
        using var dpre = backend.AllocateBuffer(checked(seqLen * batch * 4 * hiddenSize));
        if (gradHFinal is null) backend.Fill(dh, 0f, state);
        else backend.Copy(gradHFinal, 0, dh, 0, state);
        if (gradCFinal is null) backend.Fill(dc, 0f, state);
        else backend.Copy(gradCFinal, 0, dc, 0, state);
        for (int t = seqLen - 1; t >= 0; t--)
        {
            launch(K.LstmCellBackward, state, gradOutput, cacheGates, allC, dh, dc, dpre, t, batch, seqLen, hiddenSize);
            launch(K.LstmHiddenBackward, state, dpre, weightsHh, dh, t, batch, hiddenSize);
        }
        backend.Copy(dh, 0, gradHInit, 0, state);
        backend.Copy(dc, 0, gradCInit, 0, state);
        launch(K.LstmInputGrad, checked(batch * seqLen * inputSize), dpre, weightsIh, gradInput, batch, seqLen, inputSize, hiddenSize);
        launch(K.LstmWeightIhGrad, checked(4 * hiddenSize * inputSize), dpre, input, gradWeightsIh, batch, seqLen, inputSize, hiddenSize);
        launch(K.LstmWeightHhGrad, checked(4 * hiddenSize * hiddenSize), dpre, allH, gradWeightsHh, batch, seqLen, hiddenSize);
        launch(K.LstmBiasGrad, 4 * hiddenSize, dpre, gradBiasIh, gradBiasHh, batch, seqLen, hiddenSize);
    }
}