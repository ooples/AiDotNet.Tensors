namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>
/// Capability every GPU backend implements: the per-channel epilogue a convolution's output goes through before the
/// next convolution -- its bias, an inference BatchNorm's scale and shift, and the activation -- as one pass,
/// <c>y = act(((z + bias[c]) * scale[c]) + shift[c])</c>, and its backward to z, <c>dz = act'(y) * dy * scale[c]</c>.
/// Unfused, each of the four steps is its own pass over the activation in each direction (a GAN generator step's
/// frozen discriminator: conv, bias, BatchNorm multiply, BatchNorm add, LeakyReLU).
/// </summary>
/// <remarks>
/// <para>Layout NCHW: element i is in channel <c>(i / spatial) % channels</c>. Each of bias, scale and shift may be null
/// (that step skipped). The steps round as the separate passes do: an add, then a multiply, then an add, each rounded (no
/// fused multiply-add), so where a backend honors that the values equal the unfused passes bit for bit.</para>
/// <para>Activations (<see cref="Identity"/>, <see cref="Relu"/>, <see cref="LeakyRelu"/>): the backward's mask is the
/// output's sign, which is its input's for ReLU and for a LeakyReLU of positive slope; any other slope is refused.</para>
/// </remarks>
internal interface IChannelEpilogueBackend
{
    /// <summary>
    /// <c>output = act(((input + bias[c]) * scale[c]) + shift[c])</c> over [batch, channels, spatial]. False -- nothing
    /// launched -- when the backend cannot run it for the shape or the activation.
    /// </summary>
    bool TryChannelEpilogue(IGpuBuffer input, IGpuBuffer? bias, IGpuBuffer? scale, IGpuBuffer? shift, IGpuBuffer output,
        int batch, int channels, int spatial, int activation, float slope);

    /// <summary>
    /// <c>gradInput = act'(output) * gradOutput * scale[c]</c> (the mask from the forward's output). False -- nothing
    /// launched -- as for the forward.
    /// </summary>
    bool TryChannelEpilogueBackward(IGpuBuffer gradOutput, IGpuBuffer output, IGpuBuffer? scale, IGpuBuffer gradInput,
        int batch, int channels, int spatial, int activation, float slope);
}

/// <summary>The activation codes of <see cref="IChannelEpilogueBackend"/>.</summary>
internal static class ChannelEpilogueActivation
{
    internal const int Identity = 0;
    internal const int Relu = 1;
    internal const int LeakyRelu = 2;

    /// <summary>A LeakyReLU slope the kernels take: positive and finite (the backward's mask is the output's sign).</summary>
    internal static bool IsSlope(float slope) => slope > 0f && !float.IsInfinity(slope) && !float.IsNaN(slope);

    /// <summary>Whether the activation code (and, for LeakyReLU, its slope) is one the kernels take.</summary>
    internal static bool Accepts(int activation, float slope)
        => activation == Identity || activation == Relu || (activation == LeakyRelu && IsSlope(slope));
}
