// Copyright (c) CrowdTrainer LLC. All rights reserved.
// The head-to-head's loss check decides which modes may compete for the gated best, so it must fail closed: a mode that
// recorded fewer losses than the reference, or one whose loss went non-finite, is not the same training run.

#if NET5_0_OR_GREATER
using Xunit;

namespace AiDotNet.Tensors.Tests.Parity;

public class HeadToHeadLossMismatchTests
{
    private static readonly double[] Reference = { 1.0, 0.9, 0.8, 0.7 };

    [Fact]
    public void AMatchingCurveAgrees() =>
        Assert.Null(HeadToHeadNetworkHarness.LossMismatch(new[] { 1.0, 0.9, 0.8, 0.7 }, Reference));

    [Fact]
    public void AShorterCurveIsRefused() =>
        Assert.NotNull(HeadToHeadNetworkHarness.LossMismatch(new[] { 1.0, 0.9 }, Reference));

    [Theory]
    [InlineData(double.NaN)]
    [InlineData(double.PositiveInfinity)]
    public void ANonFiniteLossIsRefused(double bad) =>
        Assert.NotNull(HeadToHeadNetworkHarness.LossMismatch(new[] { 1.0, bad, 0.8, 0.7 }, Reference));

    [Fact]
    public void AnEmptyCurveIsRefused() =>
        Assert.NotNull(HeadToHeadNetworkHarness.LossMismatch(System.Array.Empty<double>(), Reference));
}
#endif
