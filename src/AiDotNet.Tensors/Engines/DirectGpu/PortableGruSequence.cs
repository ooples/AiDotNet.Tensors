// Copyright (c) AiDotNet. All rights reserved.
using K = AiDotNet.Tensors.Engines.DirectGpu.BatchExecutionCudaKernels;

namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>
/// The fused GRU sequence (#1568) over the shared gru_p_ kernels, on every backend through the launcher of
/// <see cref="IFusedLstmSequenceTraining.StatefulLstmLauncher"/>: two launches per forward step and two per backward step,
/// then one each for the input, weight and bias gradients. The caches (allH, gates, ghn) are this implementation's own
/// layout, read back only by its backward (see <see cref="K.GruSource"/>).
/// </summary>
internal static class PortableGruSequence
{
    /// <summary>Elements of the forward caches: allH [(T + 1), B, H], gates [T, B, 3H] and ghn [T, B, H].</summary>
    public static (int AllH, int Gates, int Ghn) CacheSizes(int seqLen, int batch, int hiddenSize)
    {
        Check(seqLen, batch, 1, hiddenSize);
        return (checked((seqLen + 1) * batch * hiddenSize), checked(seqLen * batch * 3 * hiddenSize),
            checked(seqLen * batch * hiddenSize));
    }

    private static void Check(int seqLen, int batch, int inputSize, int hiddenSize)
    {
        if (seqLen <= 0 || batch <= 0 || inputSize <= 0 || hiddenSize <= 0)
            throw new ArgumentOutOfRangeException(nameof(seqLen), "Every GRU extent must be positive.");
        _ = checked((long)(seqLen + 1) * batch * 6 * hiddenSize + 3L * hiddenSize * (inputSize + hiddenSize));
    }

    public static void Forward(IDirectGpuBackend backend, PortableGenericFft.Launcher launch,
        IGpuBuffer input, IGpuBuffer hInit, IGpuBuffer weightsIh, IGpuBuffer weightsHh, IGpuBuffer biasIh,
        IGpuBuffer biasHh, IGpuBuffer output, IGpuBuffer allH, IGpuBuffer gates, IGpuBuffer ghn,
        int seqLen, int batch, int inputSize, int hiddenSize)
    {
        Check(seqLen, batch, inputSize, hiddenSize);
        int state = batch * hiddenSize;
        backend.Copy(hInit, 0, allH, 0, state);
        using var pre = backend.AllocateBuffer(checked(batch * 6 * hiddenSize));
        for (int t = 0; t < seqLen; t++)
        {
            launch(K.GruProjection, batch * 3 * hiddenSize, input, weightsIh, weightsHh, biasIh, biasHh, allH, pre,
                t, batch, seqLen, inputSize, hiddenSize);
            launch(K.GruCell, state, pre, gates, ghn, allH, output, t, batch, seqLen, hiddenSize);
        }
    }

    /// <summary>
    /// Backpropagation through time. <paramref name="gradOutput"/> is the gradient of every step's output [B, T, H]
    /// (zeros where a step's output was not used); <paramref name="gradHFinal"/>, when given, adds the final state's.
    /// </summary>
    public static void Backward(IDirectGpuBackend backend, PortableGenericFft.Launcher launch,
        IGpuBuffer gradOutput, IGpuBuffer allH, IGpuBuffer gates, IGpuBuffer ghn,
        IGpuBuffer weightsIh, IGpuBuffer weightsHh, IGpuBuffer input,
        IGpuBuffer gradInput, IGpuBuffer gradHInit,
        IGpuBuffer gradWeightsIh, IGpuBuffer gradWeightsHh, IGpuBuffer gradBiasIh, IGpuBuffer gradBiasHh,
        int seqLen, int batch, int inputSize, int hiddenSize, IGpuBuffer? gradHFinal = null)
    {
        Check(seqLen, batch, inputSize, hiddenSize);
        int state = batch * hiddenSize;
        int sums = checked(seqLen * batch * 3 * hiddenSize);
        using var dh = backend.AllocateBuffer(state);
        using var dgi = backend.AllocateBuffer(sums);
        using var dgh = backend.AllocateBuffer(sums);
        if (gradHFinal is null) backend.Fill(dh, 0f, state);
        else backend.Copy(gradHFinal, 0, dh, 0, state);
        for (int t = seqLen - 1; t >= 0; t--)
        {
            launch(K.GruCellBackward, state, gradOutput, gates, ghn, allH, dh, dgi, dgh, t, batch, seqLen, hiddenSize);
            launch(K.GruHiddenBackward, state, gradOutput, gates, dgh, weightsHh, dh, t, batch, seqLen, hiddenSize);
        }
        backend.Copy(dh, 0, gradHInit, 0, state);
        launch(K.GruInputGrad, checked(batch * seqLen * inputSize), dgi, weightsIh, gradInput, batch, seqLen, inputSize, hiddenSize);
        launch(K.GruWeightIhGrad, checked(3 * hiddenSize * inputSize), dgi, input, gradWeightsIh, batch, seqLen, inputSize, hiddenSize);
        launch(K.GruWeightHhGrad, checked(3 * hiddenSize * hiddenSize), dgh, allH, gradWeightsHh, batch, seqLen, hiddenSize);
        launch(K.GruBiasGrad, 3 * hiddenSize, dgi, dgh, gradBiasIh, gradBiasHh, batch, seqLen, hiddenSize);
    }
}
