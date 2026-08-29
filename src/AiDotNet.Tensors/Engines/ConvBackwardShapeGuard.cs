// Copyright (c) AiDotNet. All rights reserved.
using System;

namespace AiDotNet.Tensors.Engines;

/// <summary>
/// Checks that a convolution backward call was handed a <c>gradOutput</c> whose spatial extents are
/// the ones the forward pass would actually have produced.
/// </summary>
/// <remarks>
/// <para>
/// The conv backward ops validated batch and channel agreement thoroughly and the SPATIAL geometry
/// not at all. <c>gradOutput</c>'s height and width are not free parameters — they are a function of
/// the input extent, the kernel, the stride, the padding and the dilation — so a caller that gets
/// them wrong is describing a convolution that cannot exist. Without this check the op accepted it,
/// walked whichever operand ran out first, and returned values: measured on
/// <c>Conv2DBackwardKernel</c> with an input of width 13 and a 3-wide valid kernel (which produces
/// 11, not 13), the float and double paths returned results of OPPOSITE SIGN and no exception.
/// </para>
/// <para>
/// Silent garbage from an impossible request is the worst available outcome. It is indistinguishable
/// from a real gradient at the call site, so it propagates into whatever consumed it, and the caller
/// learns nothing about the mistake. PyTorch rejects the same mismatch outright.
/// </para>
/// <para>
/// One guard covers the family because the small entry points delegate:
/// <c>Conv1DBackwardKernel</c> reshapes to 4D and calls <c>Conv2DBackwardKernel</c>, and
/// <c>ConvTranspose2DBackwardKernel</c> calls it with the two operands swapped. The swap needs no
/// special case — the transposed forward relation <c>big = (small - 1) * s - 2p + k</c> inverts to
/// exactly the standard form this guard applies, so checking the standard relation on the swapped
/// operands is the correct check for the transposed op.
/// </para>
/// </remarks>
internal static class ConvBackwardShapeGuard
{
    /// <summary>
    /// Throws when <paramref name="gradOutputShape"/>'s trailing spatial dimensions are not the
    /// forward convolution's output extents for the given input, kernel and hyper-parameters.
    /// </summary>
    /// <param name="op">Op name, for the message.</param>
    /// <param name="gradOutputShape">Shape of the supplied gradOutput, laid out [N, C, ...spatial].</param>
    /// <param name="inputShape">Shape of the forward input, laid out [N, C, ...spatial].</param>
    /// <param name="kernelShape">Kernel shape; its TRAILING entries are the spatial extents.</param>
    /// <param name="stride">Per-axis stride.</param>
    /// <param name="padding">Per-axis padding.</param>
    /// <param name="dilation">Per-axis dilation, or <c>null</c> for 1.</param>
    /// <param name="spatialRank">Number of spatial axes (1, 2 or 3).</param>
    public static void ValidateGradOutputSpatial(
        string op,
        int[] gradOutputShape,
        int[] inputShape,
        int[] kernelShape,
        int[] stride,
        int[] padding,
        int[]? dilation,
        int spatialRank)
    {
        if (gradOutputShape is null || inputShape is null || kernelShape is null) return;
        if (stride is null || padding is null) return;

        // Rank disagreements are already reported, and with better messages, by the callers'
        // existing checks. Bail rather than throw a worse one.
        if (gradOutputShape.Length < spatialRank + 2) return;
        if (inputShape.Length < spatialRank + 2) return;
        if (kernelShape.Length < spatialRank) return;
        if (stride.Length < spatialRank || padding.Length < spatialRank) return;
        if (dilation is not null && dilation.Length < spatialRank) return;

        int gradSpatialStart = gradOutputShape.Length - spatialRank;
        int inputSpatialStart = inputShape.Length - spatialRank;
        int kernelSpatialStart = kernelShape.Length - spatialRank;

        for (int axis = 0; axis < spatialRank; axis++)
        {
            int inExtent = inputShape[inputSpatialStart + axis];
            int kExtent = kernelShape[kernelSpatialStart + axis];
            int s = stride[axis];
            int p = padding[axis];
            int d = dilation is null ? 1 : dilation[axis];

            if (s <= 0 || d <= 0) return;   // already rejected upstream

            int effectiveKernel = d * (kExtent - 1) + 1;
            int expected = (inExtent + 2 * p - effectiveKernel) / s + 1;
            int actual = gradOutputShape[gradSpatialStart + axis];

            if (expected != actual)
            {
                throw new ArgumentException(
                    $"{op}: gradOutput spatial axis {axis} is {actual}, but a forward pass over an "
                        + $"input extent of {inExtent} with kernel {kExtent}, stride {s}, padding {p} "
                        + $"and dilation {d} produces {expected}. gradOutput's spatial size is "
                        + "determined by the forward geometry, not chosen independently; a mismatch "
                        + "describes a convolution that cannot exist.",
                    nameof(gradOutputShape));
            }
        }
    }
}
