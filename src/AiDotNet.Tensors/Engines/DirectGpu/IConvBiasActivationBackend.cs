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

/// <summary>
/// A backend whose convolution can apply a per-input-channel affine and ReLU as it loads its input (an inference batch
/// norm and ReLU before the convolution), so the normalized activation is never written.
/// </summary>
internal interface IConvPreActivationBackend
{
    /// <summary>
    /// <c>output = act(conv2d(relu(input * scale[ci] + shift[ci]), kernel) + bias[channel])</c>, zero padding applied
    /// after the affine and ReLU. False -- nothing launched -- when this backend cannot run it for the shape.
    /// </summary>
    bool TryConv2DPreActivatedBiasActivation(IGpuBuffer input, IGpuBuffer scale, IGpuBuffer shift, IGpuBuffer kernel,
        IGpuBuffer? bias, IGpuBuffer output,
        int batch, int inChannels, int inHeight, int inWidth,
        int outChannels, int outHeight, int outWidth,
        int kernelH, int kernelW,
        int strideH, int strideW, int padH, int padW,
        int dilationH, int dilationW, FusedActivationType activation);
}

/// <summary>A backend that concatenates two row-major inputs along their rows in one launch.</summary>
internal interface IRowConcatBackend
{
    /// <summary>
    /// Row r of <paramref name="destination"/> = row r of <paramref name="a"/> (<paramref name="colsA"/> floats) then row r
    /// of <paramref name="b"/> (<paramref name="colsB"/>). False -- nothing launched -- when the layout does not allow it.
    /// </summary>
    bool TryConcatRows2(IGpuBuffer a, int colsA, IGpuBuffer b, int colsB, IGpuBuffer destination, int rows);
}
