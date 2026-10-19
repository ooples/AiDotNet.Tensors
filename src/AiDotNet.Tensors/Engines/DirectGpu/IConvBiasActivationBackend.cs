namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>
/// Capability every GPU backend implements: a convolution that adds the per-channel bias and applies the activation as
/// it writes its output (the routes whose kernels cannot run the two passes after it), where the bias add and the
/// activation were two more full passes over the output on every conv layer of a CNN's inference.
/// </summary>
internal interface IConvBiasActivationBackend
{
    /// <summary>
    /// <c>output = act(conv2d(input, kernel) + bias[channel])</c>, computed as Conv2D, Conv2DBiasAdd and the activation
    /// would compute it. <paramref name="bias"/> may be null. False -- nothing launched -- when this backend does not
    /// fuse <paramref name="activation"/> or the shape is empty.
    /// </summary>
    bool TryConv2DBiasActivation(IGpuBuffer input, IGpuBuffer kernel, IGpuBuffer? bias, IGpuBuffer output,
        int batch, int inChannels, int inHeight, int inWidth,
        int outChannels, int outHeight, int outWidth,
        int kernelH, int kernelW,
        int strideH, int strideW, int padH, int padW,
        int dilationH, int dilationW, FusedActivationType activation);
}
