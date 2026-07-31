// Copyright (c) AiDotNet. All rights reserved.
// Promotion follows the measured evidence, per family.
//
// The bake-off result is mixed: six generated kernels beat cuDNN, four lose. Granting
// dispatch to the whole set on one environment variable would knowingly route work to
// conv2d_3x3_bwd_weights, measured at 0.33x -- three times slower than the path it would
// replace. These tests pin the decision so a later "just enable convolution" change cannot
// quietly include the losers.

using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Codegen;

public class DirectPtxConvolutionPromotionTests
{
    /// <summary>The families that beat cuDNN are dispatchable.</summary>
    // The ordinal rather than the enum: the enum is internal and an xUnit theory method
    // must be public, so a public signature cannot name it.
    [Theory]
    [InlineData((int)DirectPtxConvolutionFamily.Depthwise3x3)]   // 2.08x - 2.99x
    [InlineData((int)DirectPtxConvolutionFamily.MaxPool2x2)]     // 1.41x
    [InlineData((int)DirectPtxConvolutionFamily.Dense1x1)]       // 1.02x - 1.35x
    public void WinningFamilies_ArePromoted(int ordinal)
    {
        var family = (DirectPtxConvolutionFamily)ordinal;
        Assert.True(DirectPtxConvolutionPromotion.IsPromoted(family, out string? reason));
        Assert.Null(reason);
    }

    /// <summary>
    /// The families that lose are NOT dispatchable, and each says why. Dense 3x3 is the
    /// one that will not change with tuning -- its stall profile is balanced, so the gap
    /// is algorithmic.
    /// </summary>
    [Theory]
    [InlineData((int)DirectPtxConvolutionFamily.Dense3x3)]         // 0.33x - 0.65x
    [InlineData((int)DirectPtxConvolutionFamily.WeightGradient)]   // 0.78x - 0.92x
    [InlineData((int)DirectPtxConvolutionFamily.Transposed3x3)]    // 1.00x, parity
    public void LosingFamilies_AreWithheldWithAReason(int ordinal)
    {
        var family = (DirectPtxConvolutionFamily)ordinal;
        Assert.False(DirectPtxConvolutionPromotion.IsPromoted(family, out string? reason));
        Assert.False(string.IsNullOrWhiteSpace(reason));
    }

    /// <summary>
    /// An unrecognised family is NOT promoted. Promotion is opt-in per family, so a new
    /// one added without evidence must default to withheld rather than inherit a win.
    /// </summary>
    [Fact]
    public void UnknownFamily_DefaultsToWithheld()
    {
        Assert.False(DirectPtxConvolutionPromotion.IsPromoted(
            (DirectPtxConvolutionFamily)999, out string? reason));
        Assert.Contains("not promoted", reason!, System.StringComparison.Ordinal);
    }

    /// <summary>
    /// The dense-3x3 exclusion must cite the reason it is permanent for this layer, so a
    /// future reader does not retry the tuning that has already failed twice.
    /// </summary>
    [Fact]
    public void Dense3x3Exclusion_CitesTheAlgorithmicCause()
    {
        DirectPtxConvolutionPromotion.IsPromoted(
            DirectPtxConvolutionFamily.Dense3x3, out string? reason);

        Assert.Contains("balanced", reason!, System.StringComparison.Ordinal);
        Assert.Contains("Winograd", reason!, System.StringComparison.Ordinal);
    }
}
