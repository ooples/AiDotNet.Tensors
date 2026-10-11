using AiDotNet.Tensors.Engines.DirectGpu.Vulkan;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// The Vulkan device used to be created with no optional features (pEnabledFeatures = null) while shaders already used
/// doubles, 64-bit integers and atomics -- invalid usage that only lenient drivers ran. Creation now enables each
/// feature the device supports.
/// </summary>
[Collection("VulkanGlobalState")]
public class VulkanDeviceFeatureTests
{
    [SkippableFact]
    public void Device_EnablesEverySupportedShaderFeatureItUses()
    {
        bool ready;
        try { ready = VulkanBackend.Instance.Initialize(); }
        catch (System.DllNotFoundException) { ready = false; }
        Skip.IfNot(ready, "Vulkan unavailable.");

        var device = VulkanDevice.Instance;
        Assert.Equal(device.Float64Supported, device.Float64Enabled);
        Assert.Equal(device.Int64Supported, device.Int64Enabled);
    }

    [SkippableFact]
    public void Float64Kernel_RunsWhereTheFeatureIsEnabled()
    {
        bool ready;
        try { ready = VulkanBackend.Instance.Initialize() && VulkanBackend.Instance.IsGlslCompilerAvailable; }
        catch (System.DllNotFoundException) { ready = false; }
        Skip.IfNot(ready, "Vulkan or libshaderc unavailable.");
        Skip.IfNot(VulkanDevice.Instance.Float64Enabled, "The device has no shaderFloat64.");

        // linspace-style points start + i * span / n, evaluated in double on the device and rounded once. The control
        // proves the case discriminates: the same arithmetic in float lands on a different value for some i.
        const int n = 64;
        const double start = 0.1, span = 0.7, denominator = 3.0;
        var backend = VulkanBackend.Instance;
        var kernels = (AiDotNet.Tensors.Engines.DirectGpu.ITorchParityElementwiseBackend)backend;
        using var output = backend.AllocateBuffer(n);
        kernels.TorchParityRange(output, n, start, span, denominator);
        var values = backend.DownloadBuffer(output);
        var expected = new float[n];
        bool floatWouldDiffer = false;
        for (int i = 0; i < n; i++)
        {
            expected[i] = (float)(start + i * span / denominator);
            float inFloat = (float)start + i * (float)span / (float)denominator;
            floatWouldDiffer |= inFloat != expected[i];
        }
        Assert.True(floatWouldDiffer, "control: float evaluation matches the double reference everywhere, so this proves nothing");
        Assert.Equal(expected, values);
    }
}