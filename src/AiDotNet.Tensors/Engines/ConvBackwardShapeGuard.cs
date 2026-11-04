// Copyright (c) CrowdTrainer LLC. All rights reserved.
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
    /// The output padding a transposed convolution's forward pass used, recovered from the shapes its backward is
    /// handed, and checked to be one the forward could have used.
    /// </summary>
    /// <remarks>
    /// <para>
    /// A transposed forward makes <c>out = (in - 1) * s - 2p + k + op</c> with <c>0 &lt;= op &lt; s</c> (PyTorch's
    /// rule for unit dilation): the output padding adds rows at the far edge that the kernel taps write. The backward
    /// ops take no output-padding argument, but they need none: <c>op</c> is the gap between the gradient's extent and
    /// <c>(in - 1) * s - 2p + k</c>. Requiring that gap to lie in <c>[0, s)</c> accepts exactly the gradients some
    /// forward produces and still rejects the rest, which an exact-inversion check (gap zero) did not: it refused every
    /// output-padded transposed convolution's gradient.
    /// </para>
    /// <para>
    /// With the gap in <c>[0, s)</c> the data gradient is the plain forward convolution of the whole gradient (its
    /// floor lands back on <c>in</c>) and the weight gradient sums over every gradient position, so no kernel changes
    /// with it; only the callers that pass an output padding on need the value.
    /// </para>
    /// </remarks>
    /// <param name="op">Op name, for the message.</param>
    /// <param name="gradOutputShape">The gradient's shape, [N, C, ...spatial].</param>
    /// <param name="inputShape">The forward input's shape, [N, C, ...spatial].</param>
    /// <param name="kernelShape">The kernel's shape; its trailing entries are the spatial extents.</param>
    /// <param name="stride">Per-axis stride.</param>
    /// <param name="padding">Per-axis padding.</param>
    /// <param name="spatialRank">Number of spatial axes (2 or 3).</param>
    /// <returns>The per-axis output padding.</returns>
    public static int[] TransposedOutputPadding(
        string op,
        int[] gradOutputShape,
        int[] inputShape,
        int[] kernelShape,
        int[] stride,
        int[] padding,
        int spatialRank)
    {
        if (gradOutputShape is null) throw new ArgumentNullException(nameof(gradOutputShape));
        if (inputShape is null) throw new ArgumentNullException(nameof(inputShape));
        if (kernelShape is null) throw new ArgumentNullException(nameof(kernelShape));
        if (stride is null) throw new ArgumentNullException(nameof(stride));
        if (padding is null) throw new ArgumentNullException(nameof(padding));
        if (gradOutputShape.Length != spatialRank + 2 || inputShape.Length != spatialRank + 2 || kernelShape.Length != spatialRank + 2
            || stride.Length < spatialRank || padding.Length < spatialRank)
            throw new ArgumentException(
                $"{op}: expected rank-{spatialRank + 2} gradient, input and kernel shapes and {spatialRank} strides and paddings.",
                nameof(gradOutputShape));

        var outputPadding = new int[spatialRank];
        for (int axis = 0; axis < spatialRank; axis++)
        {
            int s = stride[axis];
            if (s <= 0)
                throw new ArgumentException($"{op}: stride {axis} must be positive, got {s}.", nameof(stride));
            int inExtent = inputShape[2 + axis], kExtent = kernelShape[2 + axis], actual = gradOutputShape[2 + axis];
            int unpadded = (inExtent - 1) * s - 2 * padding[axis] + kExtent;
            int gap = actual - unpadded;
            if (gap < 0 || gap >= s)
            {
                throw new ArgumentException(
                    $"{op}: gradOutput spatial axis {axis} is {actual}, but a transposed forward pass over an input "
                        + $"extent of {inExtent} with kernel {kExtent}, stride {s} and padding {padding[axis]} produces "
                        + $"{unpadded} plus an output padding in [0, {s}), so {unpadded} to {unpadded + s - 1}. "
                        + "No transposed forward pass produces that size.",
                    nameof(gradOutputShape));
            }
            outputPadding[axis] = gap;
        }
        return outputPadding;
    }

    /// <summary>
    /// Throws when <paramref name="gradOutputShape"/>'s trailing spatial dimensions are not the
    /// forward convolution's output extents for the given input, kernel and hyper-parameters.
    /// </summary>
    /// <param name="op">Op name, for the message.</param>
    /// <param name="gradOutputShape">Shape of the supplied gradOutput, laid out [N, C, ...spatial].</param>
    /// <param name="inputShape">Shape of the forward input, laid out [N, C, ...spatial].</param>
    /// <param name="kernelShape">Kernel shape; its TRAILING entries are the spatial extents.</param>
    /// <param name="stride">Per-axis stride.</param>
    /// <param name="padding">Per-axis padding, or <c>null</c> for none.</param>
    /// <param name="dilation">Per-axis dilation, or <c>null</c> for 1.</param>
    /// <param name="spatialRank">Number of spatial axes (1, 2 or 3).</param>
    public static void ValidateGradOutputSpatial(
        string op,
        int[] gradOutputShape,
        int[] inputShape,
        int[] kernelShape,
        int[] stride,
        int[]? padding,
        int[]? dilation,
        int spatialRank)
    {
        if (gradOutputShape is null || inputShape is null || kernelShape is null) return;
        if (stride is null) return;

        // Rank disagreements are already reported, and with better messages, by the callers'
        // existing checks. Bail rather than throw a worse one.
        if (gradOutputShape.Length < spatialRank + 2) return;
        if (inputShape.Length < spatialRank + 2) return;
        if (kernelShape.Length < spatialRank) return;
        if (stride.Length < spatialRank) return;
        if (padding is not null && padding.Length < spatialRank) return;
        if (dilation is not null && dilation.Length < spatialRank) return;

        int gradSpatialStart = gradOutputShape.Length - spatialRank;
        int inputSpatialStart = inputShape.Length - spatialRank;
        int kernelSpatialStart = kernelShape.Length - spatialRank;

        for (int axis = 0; axis < spatialRank; axis++)
        {
            int inExtent = inputShape[inputSpatialStart + axis];
            int kExtent = kernelShape[kernelSpatialStart + axis];
            int s = stride[axis];
            int p = padding is null ? 0 : padding[axis];
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
