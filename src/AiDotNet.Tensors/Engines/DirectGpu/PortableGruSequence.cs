// Copyright (c) AiDotNet. All rights reserved.
using K = AiDotNet.Tensors.Engines.DirectGpu.BatchExecutionCudaKernels;

namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>
/// The fused GRU sequence (#1568) over the shared gru_p_ / gru_rb_ kernels, on every backend through the launcher of
/// <see cref="IFusedLstmSequenceTraining.StatefulLstmLauncher"/>. With the reset gate after the recurrent sum
/// (PyTorch): two launches per forward step and two per backward step. With it before the recurrent product (Cho et
/// al., Keras <c>reset_after=False</c>): four and three. Then one launch each for the input, weight and bias
/// gradients. The caches (allH, gates, ghn, and rh for the reset-before form) are this implementation's own layout,
/// read back only by its backward (see <see cref="K.GruSource"/>).
/// </summary>
internal static class PortableGruSequence
{
    /// <summary>Elements of the forward caches: allH [(T + 1), B, H], gates [T, B, 3H], ghn [T, B, H] and rh [T, B, H].</summary>
    public static (int AllH, int Gates, int Ghn, int Rh) CacheSizes(int seqLen, int batch, int hiddenSize)
    {
        Check(seqLen, batch, 1, hiddenSize);
        int state = checked(seqLen * batch * hiddenSize);
        return (checked((seqLen + 1) * batch * hiddenSize), checked(seqLen * batch * 3 * hiddenSize), state, state);
    }

    private static void Check(int seqLen, int batch, int inputSize, int hiddenSize)
    {
        if (seqLen <= 0 || batch <= 0 || inputSize <= 0 || hiddenSize <= 0)
            throw new ArgumentOutOfRangeException(nameof(seqLen), "Every GRU extent must be positive.");
        _ = checked((long)(seqLen + 1) * batch * 6 * hiddenSize + 3L * hiddenSize * (inputSize + hiddenSize));
    }

    /// <summary>
    /// The forward. <paramref name="rh"/> (r * h per step, [T, B, H]) is written only with
    /// <paramref name="resetAfter"/> false, and may be null otherwise.
    /// </summary>
    public static void Forward(IDirectGpuBackend backend, PortableGenericFft.Launcher launch,
        IGpuBuffer input, IGpuBuffer hInit, IGpuBuffer weightsIh, IGpuBuffer weightsHh, IGpuBuffer biasIh,
        IGpuBuffer biasHh, IGpuBuffer output, IGpuBuffer allH, IGpuBuffer gates, IGpuBuffer ghn, IGpuBuffer? rh,
        int seqLen, int batch, int inputSize, int hiddenSize, bool resetAfter = true)
    {
        Check(seqLen, batch, inputSize, hiddenSize);
        if (!resetAfter && rh is null) throw new ArgumentNullException(nameof(rh), "The reset-before form caches r * h.");
        int state = batch * hiddenSize;
        backend.Copy(hInit, 0, allH, 0, state);
        using var pre = backend.AllocateBuffer(checked(batch * 6 * hiddenSize));
        for (int t = 0; t < seqLen; t++)
        {
            launch(K.GruProjection, batch * 3 * hiddenSize, input, weightsIh, weightsHh, biasIh, biasHh, allH, pre,
                t, batch, seqLen, inputSize, hiddenSize);
            if (resetAfter)
            {
                launch(K.GruCell, state, pre, gates, ghn, allH, output, t, batch, seqLen, hiddenSize);
                continue;
            }
            launch(K.GruRbResetHidden, state, pre, allH, gates, rh!, t, batch, hiddenSize);
            launch(K.GruRbCandidateProjection, state, weightsHh, biasHh, rh!, pre, t, batch, hiddenSize);
            launch(K.GruRbCell, state, pre, gates, ghn, allH, output, t, batch, seqLen, hiddenSize);
        }
    }

    /// <summary>
    /// Backpropagation through time. <paramref name="gradOutput"/> is the gradient of every step's output [B, T, H]
    /// (zeros where a step's output was not used); <paramref name="gradHFinal"/>, when given, adds the final state's.
    /// </summary>
    public static void Backward(IDirectGpuBackend backend, PortableGenericFft.Launcher launch,
        IGpuBuffer gradOutput, IGpuBuffer allH, IGpuBuffer gates, IGpuBuffer ghn, IGpuBuffer? rh,
        IGpuBuffer weightsIh, IGpuBuffer weightsHh, IGpuBuffer input,
        IGpuBuffer gradInput, IGpuBuffer gradHInit,
        IGpuBuffer gradWeightsIh, IGpuBuffer gradWeightsHh, IGpuBuffer gradBiasIh, IGpuBuffer gradBiasHh,
        int seqLen, int batch, int inputSize, int hiddenSize, IGpuBuffer? gradHFinal = null, bool resetAfter = true)
    {
        Check(seqLen, batch, inputSize, hiddenSize);
        if (!resetAfter && rh is null) throw new ArgumentNullException(nameof(rh), "The reset-before form needs r * h.");
        int state = batch * hiddenSize;
        int sums = checked(seqLen * batch * 3 * hiddenSize);
        using var dh = backend.AllocateBuffer(state);
        using var dgi = backend.AllocateBuffer(sums);
        using var dgh = backend.AllocateBuffer(sums);
        using var drr = backend.AllocateBuffer(resetAfter ? 1 : state);
        if (gradHFinal is null) backend.Fill(dh, 0f, state);
        else backend.Copy(gradHFinal, 0, dh, 0, state);
        for (int t = seqLen - 1; t >= 0; t--)
        {
            if (resetAfter)
            {
                launch(K.GruCellBackward, state, gradOutput, gates, ghn, allH, dh, dgi, dgh, t, batch, seqLen, hiddenSize);
                launch(K.GruHiddenBackward, state, gradOutput, gates, dgh, weightsHh, dh, t, batch, seqLen, hiddenSize);
                continue;
            }
            launch(K.GruRbCellBackward, state, gradOutput, gates, allH, dh, dgi, dgh, t, batch, seqLen, hiddenSize);
            launch(K.GruRbResetBackward, state, gates, allH, weightsHh, dgi, dgh, drr, t, batch, hiddenSize);
            launch(K.GruRbHiddenBackward, state, gradOutput, gates, dgh, weightsHh, drr, dh, t, batch, seqLen, hiddenSize);
        }
        backend.Copy(dh, 0, gradHInit, 0, state);
        launch(K.GruInputGrad, checked(batch * seqLen * inputSize), dgi, weightsIh, gradInput, batch, seqLen, inputSize, hiddenSize);
        launch(K.GruWeightIhGrad, checked(3 * hiddenSize * inputSize), dgi, input, gradWeightsIh, batch, seqLen, inputSize, hiddenSize);
        if (resetAfter)
            launch(K.GruWeightHhGrad, checked(3 * hiddenSize * hiddenSize), dgh, allH, gradWeightsHh, batch, seqLen, hiddenSize);
        else
            launch(K.GruRbWeightHhGrad, checked(3 * hiddenSize * hiddenSize), dgh, allH, rh!, gradWeightsHh, batch, seqLen, hiddenSize);
        launch(K.GruBiasGrad, 3 * hiddenSize, dgi, dgh, gradBiasIh, gradBiasHh, batch, seqLen, hiddenSize);
    }
}
