// Copyright (c) CrowdTrainer LLC. All rights reserved.
namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>
/// Backends with device kernels for the model-level ops of <c>CpuEngine.TorchModelOps</c> that the host otherwise
/// computes (<see cref="TorchModelOpsKernels"/>): the activation backward, fake quantization, top-k softmax, the
/// sinusoidal time embedding and the linear-chain CRF log-partition with its forward-backward gradients. The
/// activations' forwards are <see cref="TorchParityUnaryOp"/> kernels of <see cref="ITorchParityElementwiseBackend"/>.
/// </summary>
/// <remarks>
/// Implemented by every direct backend (CUDA, HIP, OpenCL, Vulkan, Metal, WebGPU) over the one shared source. All
/// buffers hold float. A launch on a device whose module failed to build throws
/// <see cref="GpuCapabilityUnavailableException"/>, which the engine records as that op's host fallback.
/// </remarks>
internal interface ITorchModelOpsBackend
{
    /// <summary>
    /// <paramref name="gradInput"/>[i] = <paramref name="gradOutput"/>[i] * f'(<paramref name="input"/>[i]) for the
    /// activation <paramref name="op"/> (a <see cref="TorchModelOpsCodes"/> activation code) with parameters p0, p1.
    /// </summary>
    void ModelActivationBackward(int op, IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradInput, int n, float p0, float p1);

    /// <summary>
    /// Fake quantization of n values: <paramref name="output"/> holds the dequantized values and
    /// <paramref name="mask"/> 1 where the quantized value was in [qMin, qMax], else 0. With
    /// <paramref name="derive"/> the step (and, without a given zero point, an asymmetric zero point) comes from the
    /// tensor's min / max, reduced into <paramref name="partial"/> (2 * <see cref="TorchModelOpsCodes.MinMaxSlots"/>
    /// floats) and <paramref name="minMax"/> (2 floats).
    /// </summary>
    void ModelFakeQuantize(IGpuBuffer input, IGpuBuffer partial, IGpuBuffer minMax, IGpuBuffer output, IGpuBuffer mask, int n,
        bool derive, bool symmetric, bool hasZeroPoint, float qMin, float qMax, float fixedStep, float zeroPoint);

    /// <summary>Softmax over each row's k largest entries (ties to the lower index), 0 elsewhere; <paramref name="scratch"/>
    /// holds rows * width floats.</summary>
    void ModelTopKSoftmax(IGpuBuffer input, IGpuBuffer scratch, IGpuBuffer output, int rows, int width, int k);

    /// <summary>[batch, dim] sinusoidal embedding of the timesteps; <paramref name="frequencies"/> holds 3 * halfDim
    /// floats, each frequency split by <see cref="TorchModelOpsCodes.SplitFrequency"/>.</summary>
    void ModelTimeEmbedding(IGpuBuffer timesteps, IGpuBuffer frequencies, IGpuBuffer output, int batch, int halfDim, int dim);

    /// <summary>The timesteps' gradient of <see cref="ModelTimeEmbedding"/>.</summary>
    void ModelTimeEmbeddingBackward(IGpuBuffer timesteps, IGpuBuffer frequencies, IGpuBuffer gradOutput, IGpuBuffer gradTimesteps,
        int batch, int halfDim, int dim);

    /// <summary>
    /// The CRF forward recursion: <paramref name="alpha"/> ([batch, seqLen, tags], log space) and
    /// <paramref name="logZ"/> ([batch]). <paramref name="start"/> / <paramref name="end"/> hold tags scores (zeros when
    /// the caller has none).
    /// </summary>
    void ModelCrfForward(IGpuBuffer emissions, IGpuBuffer transitions, IGpuBuffer start, IGpuBuffer end, IGpuBuffer alpha,
        IGpuBuffer logZ, int batch, int seqLen, int tags);

    /// <summary>
    /// The CRF forward-backward gradients scaled by <paramref name="gradLogZ"/> ([batch]): emissions
    /// ([batch, seqLen, tags]), transitions ([tags, tags]) and the start / end scores ([tags] each).
    /// <paramref name="beta"/> is [batch, seqLen, tags] scratch.
    /// </summary>
    void ModelCrfBackward(IGpuBuffer emissions, IGpuBuffer transitions, IGpuBuffer end, IGpuBuffer alpha, IGpuBuffer logZ,
        IGpuBuffer gradLogZ, IGpuBuffer beta, IGpuBuffer gradEmissions, IGpuBuffer gradTransitions, IGpuBuffer gradStart,
        IGpuBuffer gradEnd, int batch, int seqLen, int tags);
}

/// <summary>Codes and host-side helpers shared by the engine and <see cref="PortableTorchModelOps"/>.</summary>
internal static class TorchModelOpsCodes
{
    /// <summary>Slots of the first min / max pass of fake quantization.</summary>
    public const int MinMaxSlots = 1024;

    /// <summary>The activation code mo_activation_backward reads for a model activation, or -1.</summary>
    public static int ActivationCode(TorchParityUnaryOp op) => op switch
    {
        TorchParityUnaryOp.Lisht => 0,
        TorchParityUnaryOp.BentIdentity => 1,
        TorchParityUnaryOp.Gaussian => 2,
        TorchParityUnaryOp.ScaledTanh => 3,
        TorchParityUnaryOp.ThresholdedRelu => 4,
        TorchParityUnaryOp.Isru => 5,
        TorchParityUnaryOp.Sqrbf => 6,
        TorchParityUnaryOp.SurrogateSpike => 7,
        _ => -1,
    };

    /// <summary>The forward kernel of a model activation, or null for any other op.</summary>
    public static string? ActivationKernel(TorchParityUnaryOp op) => op switch
    {
        TorchParityUnaryOp.Lisht => "mo_lisht",
        TorchParityUnaryOp.BentIdentity => "mo_bent_identity",
        TorchParityUnaryOp.Gaussian => "mo_gaussian",
        TorchParityUnaryOp.ScaledTanh => "mo_scaled_tanh",
        TorchParityUnaryOp.ThresholdedRelu => "mo_thresholded_relu",
        TorchParityUnaryOp.Isru => "mo_isru",
        TorchParityUnaryOp.Sqrbf => "mo_sqrbf",
        TorchParityUnaryOp.SurrogateSpike => "mo_surrogate_spike",
        _ => null,
    };

    /// <summary>
    /// A frequency as three floats f1 + f2 + f3 (12, 12 and 24 significant bits, f1 and f2 cut from the value's bits),
    /// so the kernel forms t * f1 and t * f2 exactly from t's bit-split halves.
    /// </summary>
    public static void SplitFrequency(double f, float[] into, int at)
    {
        float f1 = HighBits((float)f);
        double rest = f - f1;
        float f2 = HighBits((float)rest);
        into[at] = f1;
        into[at + 1] = f2;
        into[at + 2] = (float)(rest - f2);
    }

    // The float with its low 12 significand bits cleared.
    private static float HighBits(float v)
    {
        int bits = System.BitConverter.ToInt32(System.BitConverter.GetBytes(v), 0);
        return System.BitConverter.ToSingle(System.BitConverter.GetBytes(bits & unchecked((int)0xFFFFF000u)), 0);
    }

    /// <summary>
    /// The largest float not above <paramref name="threshold"/>: for a float x, x &gt; that float exactly when x &gt;
    /// threshold, so a device comparison in float agrees with the CPU's comparison against the double.
    /// </summary>
    public static float FloatThresholdBelow(double threshold)
    {
        if (double.IsNaN(threshold)) return float.NaN;
        float f = (float)threshold;
        if ((double)f <= threshold) return f;
        // f rounded up (to the next float, or to +infinity past float's range): step down by one ulp.
        int bits = System.BitConverter.ToInt32(System.BitConverter.GetBytes(f), 0);
        if (f == 0f) return -float.Epsilon;   // threshold in (-eps, 0): the largest float below is -eps (f is +0 or -0)
        bits += f > 0f ? -1 : 1;
        return System.BitConverter.ToSingle(System.BitConverter.GetBytes(bits), 0);
    }
}
