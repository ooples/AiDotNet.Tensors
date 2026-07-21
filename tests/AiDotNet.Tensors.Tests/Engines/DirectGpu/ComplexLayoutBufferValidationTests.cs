using System.Runtime.Serialization;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.Tensors.Engines.DirectGpu.HIP;
using AiDotNet.Tensors.Engines.DirectGpu.Vulkan;
#if NET5_0_OR_GREATER
using AiDotNet.Tensors.Engines.DirectGpu.WebGpu;
#endif
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// Hardware-independent boundary tests for packed complex buffers. Validation must run before any
/// backend state or kernel launch is touched, because an undersized packed buffer would otherwise
/// read or write past the device allocation.
/// </summary>
public class ComplexLayoutBufferValidationTests
{
    [Theory]
    [InlineData("CUDA")]
    [InlineData("HIP")]
    [InlineData("Vulkan")]
#if NET5_0_OR_GREATER
    [InlineData("WebGPU")]
#endif
    public void InterleaveAndDeinterleave_RejectUndersizedBuffersBeforeDispatch(string backendName)
    {
        const int n = 8;
        IDirectGpuBackend backend = CreateWithoutHardware(backendName);
        var full = new MockGpuBuffer(new float[n]);
        var shortSplit = new MockGpuBuffer(new float[n - 1]);
        var packed = new MockGpuBuffer(new float[2 * n]);
        var shortPacked = new MockGpuBuffer(new float[2 * n - 1]);

        Assert.Throws<ArgumentException>(() =>
            backend.InterleaveComplex(full, full, shortPacked, n));
        Assert.Throws<ArgumentException>(() =>
            backend.DeinterleaveComplex(shortPacked, full, full, n));
        Assert.Throws<ArgumentException>(() =>
            backend.InterleaveComplex(shortSplit, full, packed, n));
        Assert.Throws<ArgumentException>(() =>
            backend.DeinterleaveComplex(packed, full, shortSplit, n));
    }

    private static IDirectGpuBackend CreateWithoutHardware(string backendName)
    {
#pragma warning disable SYSLIB0050 // Validation is deliberately exercised before constructor-owned GPU state.
        object backend = backendName switch
        {
            "CUDA" => FormatterServices.GetUninitializedObject(typeof(CudaBackend)),
            "HIP" => FormatterServices.GetUninitializedObject(typeof(HipBackend)),
            "Vulkan" => FormatterServices.GetUninitializedObject(typeof(VulkanBackend)),
#if NET5_0_OR_GREATER
            "WebGPU" => FormatterServices.GetUninitializedObject(typeof(WebGpuBackend)),
#endif
            _ => throw new ArgumentOutOfRangeException(nameof(backendName), backendName, null)
        };
#pragma warning restore SYSLIB0050
        GC.SuppressFinalize(backend);
        return (IDirectGpuBackend)backend;
    }
}
