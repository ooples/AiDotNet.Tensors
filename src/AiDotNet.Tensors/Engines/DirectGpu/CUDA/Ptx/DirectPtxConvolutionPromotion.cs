// Copyright (c) AiDotNet. All rights reserved.
// Which generated convolution families may be dispatched to, and why.
//
// The generated kernels were measured against cuDNN in its CUDA-graph lane, at true fp32,
// with locked clocks. The result is mixed, and promotion follows the evidence PER FAMILY
// rather than being granted to the whole set by one environment variable:
//
//   depthwise 3x3 forward / bias+relu / bwd_data   2.08x - 2.99x   PROMOTED
//   maxpool 2x2                                    1.41x           PROMOTED
//   1x1 bias+relu / deep epilogue / bwd_data       1.02x - 1.35x   PROMOTED
//   dense 3x3 forward / bwd_data / bwd_weights     0.33x - 0.65x   EXCLUDED
//   weight gradients, depthwise and 1x1            0.78x - 0.92x   EXCLUDED
//
// The exclusions are the point. Dispatching to conv2d_3x3_bwd_weights would knowingly
// route work to a kernel measured at a third of the alternative's speed, and it stays out
// of reach here even when the feature flag is on, so no environment variable can turn it
// on by accident.
//
// Dense 3x3 is excluded for a reason that will not change with tuning: its warp-stall
// profile is BALANCED -- no unit above 64%, mio_throttle at 2.9%, no dominant stall -- so
// there is no code-generator lever left. Closing that gap needs a different algorithm
// (Winograd or implicit GEMM), which is a different kernel, not a better schedule. The
// weight gradients are excluded for the ordinary reason: split-K bought 16.6x and 35.1x
// over our own prior lowering and still landed behind cuDNN.

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>A generated convolution family, as measured by the bake-off.</summary>
internal enum DirectPtxConvolutionFamily
{
    /// <summary>Depthwise 3x3, forward and gradient-with-respect-to-data.</summary>
    Depthwise3x3,

    /// <summary>2x2 max pooling.</summary>
    MaxPool2x2,

    /// <summary>Dense 1x1, forward, deep epilogue, and gradient-with-respect-to-data.</summary>
    Dense1x1,

    /// <summary>Dense 3x3, any direction.</summary>
    Dense3x3,

    /// <summary>Gradient with respect to weights, any convolution.</summary>
    WeightGradient,

    /// <summary>Transposed 3x3 stride 2.</summary>
    Transposed3x3
}

/// <summary>The measured promotion decision for each generated convolution family.</summary>
internal static class DirectPtxConvolutionPromotion
{
    /// <summary>
    /// Whether a family may be dispatched to, with the reason when it may not.
    /// </summary>
    /// <param name="family">Family the candidate kernel belongs to.</param>
    /// <param name="reason">Why the family is withheld, when this returns false.</param>
    /// <remarks>
    /// Checked IN ADDITION to the feature flag and the architecture predicate, never
    /// instead of them. The flag says "the caller opted into generated convolution"; this
    /// says "and this particular family actually beat the alternative".
    /// </remarks>
    internal static bool IsPromoted(DirectPtxConvolutionFamily family, out string? reason)
    {
        switch (family)
        {
            case DirectPtxConvolutionFamily.Depthwise3x3:
            case DirectPtxConvolutionFamily.MaxPool2x2:
            case DirectPtxConvolutionFamily.Dense1x1:
                reason = null;
                return true;

            case DirectPtxConvolutionFamily.Dense3x3:
                reason = "dense 3x3 is measured at 0.33x-0.65x of cuDNN and its stall " +
                         "profile is balanced, so no code-generator change closes the gap; " +
                         "it needs Winograd or implicit GEMM, which is a different kernel";
                return false;

            case DirectPtxConvolutionFamily.WeightGradient:
                reason = "weight gradients are measured at 0.78x-0.92x of cuDNN even after " +
                         "split-K bought 16.6x-35.1x over our own prior lowering";
                return false;

            case DirectPtxConvolutionFamily.Transposed3x3:
                reason = "transposed 3x3 measures 1.00x, which is parity rather than a win, " +
                         "so dispatching to it trades a known path for an equal one";
                return false;

            default:
                reason = "unknown convolution family " + family + "; promotion is opt-in " +
                         "per family and an unrecognised one is not promoted";
                return false;
        }
    }
}
