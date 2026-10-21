// Copyright (c) AiDotNet. All rights reserved.
namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>
/// Optional capability: the input gradient of a 3 x 3 stride-2 padding-1 convolution plus that of the 1 x 1 stride-2
/// padding-0 projection shortcut reading the same input (ResNet's downsampling blocks), in one launch. The shortcut
/// reads the strided convolution's centre-tap input, so its gradient adds to one sub-pixel phase of the same output;
/// the compiled training plan runs the pair as one backward action, so the input's gradient is written once instead of
/// two launches and an accumulation add. False - nothing launched - when the shapes are outside the kernel; the caller
/// then runs the two input gradients separately.
/// </summary>
[BackendSpecificRoute("CUDA one-launch input gradient of a strided 3x3 convolution and its 1x1 projection shortcut for the compiled plan's shortcut pair; every backend composes the two input gradients and an add",
    nameof(IDirectGpuBackend.Conv2DBackwardInput), nameof(IDirectGpuBackend.Add))]
internal interface IConvShortcutPairBackend
{
    /// <summary>True when the kernel takes the pair: <paramref name="outChannels"/> output channels for the 3 x 3
    /// convolution and <paramref name="shortcutChannels"/> for the projection, of one input exactly twice their
    /// output.</summary>
    bool SupportsConv2DBackwardInputShortcutPair(int batch, int inChannels, int inHeight, int inWidth, int outChannels,
        int shortcutChannels, int outHeight, int outWidth);

    /// <summary>
    /// Writes <paramref name="gradInput"/> [N, C, H, W] = the input gradient of the 3 x 3 convolution
    /// (<paramref name="gradOutput"/> [N, K, OH, OW], <paramref name="kernel"/> [K, C, 3, 3]) plus that of the projection
    /// (<paramref name="gradShortcut"/> [N, K2, OH, OW], <paramref name="kernelShortcut"/> [K2, C, 1, 1]).
    /// </summary>
    bool TryConv2DBackwardInputShortcutPair(IGpuBuffer gradOutput, IGpuBuffer kernel, IGpuBuffer gradShortcut,
        IGpuBuffer kernelShortcut, IGpuBuffer gradInput, int batch, int inChannels, int inHeight, int inWidth,
        int outChannels, int shortcutChannels, int outHeight, int outWidth);
}
