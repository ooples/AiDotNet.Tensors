// Copyright (c) AiDotNet. All rights reserved.
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines;

/// <summary>
/// An engine that computes the input gradient of ResNet's downsampling pair in one call: a 3 x 3 stride-2 padding-1
/// convolution plus the 1 x 1 stride-2 padding-0 projection shortcut reading the same input. The compiled training
/// plan (CompiledTrainingPlan.ShortcutPair.cs) runs each such pair as one backward action through this interface;
/// <see cref="DirectGpuTensorEngine"/> implements it over the backend's <c>IConvShortcutPairBackend</c>.
/// </summary>
internal interface IConvShortcutPairEngine
{
    /// <summary>True when <see cref="TryConv2DBackwardInputShortcutPair"/> can take this pair: the
    /// <paramref name="inputShape"/> input, <paramref name="kernel3"/> [K, C, 3, 3], <paramref name="kernel1"/>
    /// [K2, C, 1, 1] and an <paramref name="outH"/> x <paramref name="outW"/> output.</summary>
    bool SupportsConv2DBackwardInputShortcutPair(int[] inputShape, int[] kernel3, int[] kernel1, int outH, int outW);

    /// <summary>The input gradient of both convolutions, summed; null when the pair is not computed (the caller then
    /// runs the two input gradients separately).</summary>
    Tensor<float>? TryConv2DBackwardInputShortcutPair(Tensor<float> gradOutput3, Tensor<float> kernel3,
        Tensor<float> gradOutput1, Tensor<float> kernel1, int[] inputShape);
}
