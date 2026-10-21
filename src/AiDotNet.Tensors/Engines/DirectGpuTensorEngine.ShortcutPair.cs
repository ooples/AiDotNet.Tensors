// Copyright (c) AiDotNet. All rights reserved.
using System;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines;

public partial class DirectGpuTensorEngine
{
    // Pair launches dispatched, process-wide: the tests read it to prove a compiled plan took the pair (the plan's
    // backward step names are published only under the step-profiling switch).
    private static long s_shortcutPairLaunches;
    internal static long ShortcutPairLaunches => System.Threading.Interlocked.Read(ref s_shortcutPairLaunches);

    /// <summary>
    /// True when <see cref="TryConv2DBackwardInputShortcutPair"/> can compute this pair's input gradient on this engine:
    /// a backend with <see cref="IConvShortcutPairBackend"/>, and a 3 x 3 stride-2 padding-1 convolution
    /// (<paramref name="kernel3"/>) with a 1 x 1 stride-2 padding-0 projection (<paramref name="kernel1"/>) of the same
    /// <paramref name="inputShape"/> input.
    /// </summary>
    internal bool SupportsConv2DBackwardInputShortcutPair(int[] inputShape, int[] kernel3, int[] kernel1, int outH, int outW)
        => TryGetBackend(out var backend) && backend is IConvShortcutPairBackend pair
           && inputShape.Length == 4 && kernel3.Length == 4 && kernel1.Length == 4
           && pair.SupportsConv2DBackwardInputShortcutPair(inputShape[0], inputShape[1], inputShape[2], inputShape[3],
               kernel3[0], kernel1[0], outH, outW)
           && kernel3[1] == inputShape[1] && kernel3[2] == 3 && kernel3[3] == 3
           && kernel1[1] == inputShape[1] && kernel1[2] == 1 && kernel1[3] == 1;

    /// <summary>
    /// The input gradient of a 3 x 3 stride-2 convolution plus that of the 1 x 1 stride-2 projection shortcut reading the
    /// same input, in one launch (conv2d_dgrad_s2k3sc): <paramref name="gradOutput3"/> and <paramref name="kernel3"/>
    /// [K, C, 3, 3], <paramref name="gradOutput1"/> and <paramref name="kernel1"/> [K2, C, 1, 1]. Null when the pair is
    /// not computed here (the caller runs the two input gradients separately).
    /// </summary>
    internal Tensor<float>? TryConv2DBackwardInputShortcutPair(Tensor<float> gradOutput3, Tensor<float> kernel3,
        Tensor<float> gradOutput1, Tensor<float> kernel1, int[] inputShape)
    {
        if (!TryGetBackend(out var backend) || backend is not IConvShortcutPairBackend pair
            || gradOutput3.Rank != 4 || gradOutput1.Rank != 4
            || !SupportsConv2DBackwardInputShortcutPair(inputShape, kernel3._shape, kernel1._shape,
                gradOutput3._shape[2], gradOutput3._shape[3])
            || gradOutput1._shape[2] != gradOutput3._shape[2] || gradOutput1._shape[3] != gradOutput3._shape[3]
            || gradOutput3._shape[1] != kernel3._shape[0] || gradOutput1._shape[1] != kernel1._shape[0])
            return null;
        int batch = inputShape[0], inC = inputShape[1], inH = inputShape[2], inW = inputShape[3];
        int k3 = kernel3._shape[0], k1 = kernel1._shape[0], outH = gradOutput3._shape[2], outW = gradOutput3._shape[3];
        try
        {
            using var dy3 = GetOrAllocateBuffer(backend, gradOutput3);
            using var w3 = GetOrAllocateBuffer(backend, kernel3);
            using var dy1 = GetOrAllocateBuffer(backend, gradOutput1);
            using var w1 = GetOrAllocateBuffer(backend, kernel1);
            return DispatchDeferredGpuOp<float>(backend, batch * inC * inH * inW, inputShape, output =>
            {
                if (!pair.TryConv2DBackwardInputShortcutPair(dy3.Buffer, w3.Buffer, dy1.Buffer, w1.Buffer, output, batch, inC,
                        inH, inW, k3, k1, outH, outW))
                    throw new NotSupportedException("conv2d_dgrad_s2k3sc declined the shortcut pair.");
                System.Threading.Interlocked.Increment(ref s_shortcutPairLaunches);
            }, fullyWritten: true);
        }
        catch (NotSupportedException capabilityGap)
        {
            GpuLaunchProbe.OnFallback("Conv2DBackwardInputShortcutPair", capabilityGap);
            return null;
        }
    }
}
