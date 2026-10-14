using AiDotNet.Tensors.Engines.DevicePrimitives;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DevicePrimitives;

/// <summary>
/// The device routing rules of <see cref="CuRand"/>, which touch no CUDA state and so run on every CI machine. The
/// device itself is exercised by <see cref="CuRandDeviceGenerationTests"/>, which needs an NVIDIA GPU.
/// </summary>
public class CuRandDevicePlanTests
{
    [Fact]
    public void Mrg32k3aWaitsUntilDemandEarnsTheSetupThenRoutesTheWholeCall()
    {
        using var rng = new CuRand(7, DeviceRngAlgorithm.Mrg32K3A) { Mrg32k3aSetupBreakEven = 1000 };
        Assert.False(rng.TryPlanExactDevice(600, out _)); // 600 of 1000 earned
        Assert.True(rng.TryPlanExactDevice(600, out var plan)); // 1200 crosses the break-even
        Assert.Equal(0UL, plan.Offset);
        Assert.Equal(600UL, plan.Advance);
        Assert.Equal(1u, plan.Dimensions);
        // The setup was paid, so demand starts again from zero.
        Assert.False(rng.TryPlanExactDevice(600, out _));
    }

    [Fact]
    public void Mrg32k3aStaysOnTheCpuOnceANormalCallMakesTheOffsetInexact()
    {
        using var rng = new CuRand(7, DeviceRngAlgorithm.Mrg32K3A) { Mrg32k3aSetupBreakEven = 1 };
        rng.Normal(new Tensor<float>(new[] { 8 }));
        Assert.False(rng.TryPlanExactDevice(4096, out _));
        rng.Offset = 128; // setting the offset makes the position exact again
        Assert.True(rng.TryPlanExactDevice(4096, out var plan));
        Assert.Equal(128UL, plan.Offset);
    }

    [Fact]
    public void Mrg32k3aOnANonZeroSubsequenceStaysOnTheCpu()
    {
        using var rng = new CuRand(7, DeviceRngAlgorithm.Mrg32K3A, subsequence: 3) { Mrg32k3aSetupBreakEven = 1 };
        Assert.False(rng.TryPlanExactDevice(4096, out _));
    }

    [Fact]
    public void SobolRoutesOnlyWholePointsAboveTheMinimumAndInsideThePeriod()
    {
        using var rng = new CuRand(7, DeviceRngAlgorithm.Sobol, dimensions: 4) { SobolDeviceMinimumLength = 64 };
        Assert.False(rng.TryPlanExactDevice(60, out _)); // below the minimum
        Assert.False(rng.TryPlanExactDevice(66, out _)); // not whole points
        Assert.True(rng.TryPlanExactDevice(128, out var plan));
        Assert.Equal(4u, plan.Dimensions);
        Assert.Equal(32UL, plan.Advance); // points, not values
        rng.Offset = (1UL << 32) - 16; // 32 points would cross the 2^32-point period
        Assert.False(rng.TryPlanExactDevice(128, out _));
    }

    [Fact]
    public void ScrambledSobolAndEmptyCallsNeverPlanADevicePass()
    {
        using var scrambled = new CuRand(7, DeviceRngAlgorithm.ScrambledSobol, dimensions: 2) { SobolDeviceMinimumLength = 2 };
        Assert.False(scrambled.TryPlanExactDevice(1024, out _));
        using var mrg = new CuRand(7, DeviceRngAlgorithm.Mrg32K3A) { Mrg32k3aSetupBreakEven = 1 };
        Assert.False(mrg.TryPlanExactDevice(0, out _));
    }

    [Fact]
    public void CommittingAPlanMovesTheCpuGeneratorToWhereTheDeviceStopped()
    {
        using var mrg = new CuRand(7, DeviceRngAlgorithm.Mrg32K3A) { Mrg32k3aSetupBreakEven = 1, Offset = 10 };
        Assert.True(mrg.TryPlanExactDevice(500, out var plan));
        mrg.CommitPlan(plan);
        Assert.Equal(510UL, mrg.Offset);

        using var sobol = new CuRand(7, DeviceRngAlgorithm.Sobol, dimensions: 4) { SobolDeviceMinimumLength = 4, Offset = 3 };
        Assert.True(sobol.TryPlanExactDevice(400, out var points));
        sobol.CommitPlan(points);
        Assert.Equal(103UL, sobol.Offset); // 400 values of 4 dimensions are 100 points
    }
}
