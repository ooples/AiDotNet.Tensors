using System.Globalization;
using AiDotNet.Tensors.Helpers.Autotune;
using Xunit;

namespace AiDotNet.Tensors.Tests.Helpers.Autotune;

/// <summary>
/// Pure, GPU-free tests for <see cref="GpuDeviceFingerprint"/>. The critical
/// invariant is that <see cref="GpuDeviceFingerprint.ToCacheToken"/> is
/// byte-identical to the legacy CUDA fingerprint string, so wiring the runtime
/// through the structured type does not invalidate any existing on-disk cache.
/// </summary>
public sealed class GpuDeviceFingerprintTests
{
    // Reproduces the exact legacy formatter that DirectPtxRuntime used before
    // the structured fingerprint: gpu-{uuid}-sm{maj}{min}-drv{driver}.
    private static string LegacyToken(string uuid, int maj, int min, int drv) =>
        $"gpu-{uuid}-sm{maj}{min}-drv{drv.ToString(CultureInfo.InvariantCulture)}";

    [Theory]
    [InlineData("deadbeefcafe", 8, 6, 12030)]
    [InlineData("ordinal-0", 9, 0, 12040)]
    [InlineData("aa11bb22", 7, 5, 11080)]
    public void ToCacheToken_IsByteIdenticalToLegacyFingerprint(string uuid, int maj, int min, int drv)
    {
        var fp = GpuDeviceFingerprint.FromCuda("NVIDIA GeForce RTX 3080", uuid, maj, min, drv);
        Assert.Equal(LegacyToken(uuid, maj, min, drv), fp.ToCacheToken());
        Assert.Equal(fp.ToCacheToken(), fp.LocalKey);
    }

    [Theory]
    [InlineData("NVIDIA GeForce RTX 3080", "nvidia")]
    [InlineData("Tesla V100-SXM2-16GB", "nvidia")]
    [InlineData("Quadro RTX 6000", "nvidia")]
    [InlineData("AMD Radeon RX 6800 XT", "amd")]
    [InlineData("AMD Instinct MI250", "amd")]
    [InlineData("Intel(R) Arc(TM) A770 Graphics", "intel")]
    [InlineData("Apple M2 Max", "apple")]
    [InlineData("Some Unknown Accelerator", "other")]
    public void FromCuda_DetectsVendor(string model, string expectedVendor)
    {
        var fp = GpuDeviceFingerprint.FromCuda(model, "uuid1", 8, 6, 12030);
        Assert.Equal(expectedVendor, fp.Vendor);
    }

    [Fact]
    public void ModelKey_ExcludesUuid_SoTwoCardsOfSameModelShareIt_ButLocalKeysDiffer()
    {
        // Two physical RTX 3080s in one box: same model/driver/arch, distinct UUIDs.
        var cardA = GpuDeviceFingerprint.FromCuda("NVIDIA GeForce RTX 3080", "uuidAAA", 8, 6, 12030);
        var cardB = GpuDeviceFingerprint.FromCuda("NVIDIA GeForce RTX 3080", "uuidBBB", 8, 6, 12030);

        // Shareable across identical models...
        Assert.Equal(cardA.ModelKey, cardB.ModelKey);
        Assert.Equal("nvidia|NVIDIA GeForce RTX 3080|sm86|drv12030", cardA.ModelKey);
        // ...but locally tuned per physical card.
        Assert.NotEqual(cardA.LocalKey, cardB.LocalKey);
    }

    [Fact]
    public void ModelKey_DiffersAcrossDriverAndArch()
    {
        var baseline = GpuDeviceFingerprint.FromCuda("NVIDIA GeForce RTX 3080", "u", 8, 6, 12030);
        var newerDriver = GpuDeviceFingerprint.FromCuda("NVIDIA GeForce RTX 3080", "u", 8, 6, 12040);
        var newerArch = GpuDeviceFingerprint.FromCuda("NVIDIA GeForce RTX 4090", "u", 8, 9, 12030);

        Assert.NotEqual(baseline.ModelKey, newerDriver.ModelKey);
        Assert.NotEqual(baseline.ModelKey, newerArch.ModelKey);
    }

    [Fact]
    public void ModelKey_ModelSegmentCannotBreakFieldStructure()
    {
        // A pathological name containing the '|' separator and extra whitespace
        // must not inject extra fields into the 4-field ModelKey.
        var fp = GpuDeviceFingerprint.FromCuda("Weird | GPU   Name", "u", 8, 6, 12030);
        Assert.Equal(4, fp.ModelKey.Split('|').Length);
    }

    [Fact]
    public void TryParseCacheToken_RoundTripsLocalKey()
    {
        var original = GpuDeviceFingerprint.FromCuda("NVIDIA GeForce RTX 3080", "deadbeef", 8, 6, 12030);
        Assert.True(GpuDeviceFingerprint.TryParseCacheToken(original.ToCacheToken(), out var parsed));
        // Vendor/model aren't recoverable from the token, but the local key must round-trip.
        Assert.Equal(original.ToCacheToken(), parsed.ToCacheToken());
        Assert.Equal("deadbeef", parsed.UniqueId);
        Assert.Equal(8, parsed.ArchitectureMajor);
        Assert.Equal(6, parsed.ArchitectureMinor);
        Assert.Equal(12030, parsed.DriverVersion);
    }

    [Theory]
    [InlineData("")]
    [InlineData("not-a-token")]
    [InlineData("gpu--sm86-drv1")]     // empty uuid
    [InlineData("gpu-abc-drv12")]      // no sm segment
    public void TryParseCacheToken_RejectsMalformed(string token)
    {
        Assert.False(GpuDeviceFingerprint.TryParseCacheToken(token, out _));
    }

    [Fact]
    public void GpuKernelId_StructuredOverload_UsesLocalKey()
    {
        var fp = GpuDeviceFingerprint.FromCuda("NVIDIA GeForce RTX 3080", "deadbeef", 8, 6, 12030);
        KernelId viaStruct = GpuFirstRunAutotuner.GpuKernelId("conv2d", "tiled-1x1", fp);
        KernelId viaString = GpuFirstRunAutotuner.GpuKernelId("conv2d", "tiled-1x1", fp.LocalKey);
        Assert.Equal(viaString, viaStruct);
        Assert.Equal("tiled-1x1@" + fp.LocalKey, viaStruct.Name);
    }
}
