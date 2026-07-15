using System;
using System.Reflection;
using AiDotNet.Tensors.Engines;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

public sealed class StatelessDropoutKernelSourceTests
{
    [Theory]
    [InlineData("AiDotNet.Tensors.Engines.DirectGpu.CUDA.Kernels.CudaLossForwardKernels", "GetSource")]
    [InlineData("AiDotNet.Tensors.Engines.DirectGpu.HIP.Kernels.HipLossForwardKernels", "GetSource")]
    [InlineData("AiDotNet.Tensors.Engines.DirectGpu.OpenCL.Kernels.LossForwardKernels", "GetSource")]
    [InlineData("AiDotNet.Tensors.Engines.DirectGpu.Metal.MetalResidentKernels", "Source")]
    [InlineData("AiDotNet.Tensors.Engines.DirectGpu.Vulkan.VulkanResidentKernels", "StatelessDropoutMask")]
    [InlineData("AiDotNet.Tensors.Engines.DirectGpu.WebGpu.WebGpuKernels", "StatelessDropoutMaskSource")]
    public void EveryBackendUsesTheSharedIntegerPcgContract(string typeName, string memberName)
    {
        string source = GetStaticString(typeName, memberName);

        Assert.Contains("747796405u", source);
        Assert.Contains("2891336453u", source);
        Assert.Contains("277803737u", source);
        Assert.Contains("sample", source);
        Assert.Contains("threshold", source);
    }

    private static string GetStaticString(string typeName, string memberName)
    {
        Type type = typeof(DirectGpuTensorEngine).Assembly.GetType(typeName)
            ?? throw new InvalidOperationException($"Kernel source type not found: {typeName}");
        const BindingFlags flags = BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Static;

        MethodInfo? method = type.GetMethod(memberName, flags);
        if (method is not null)
            return (string)(method.Invoke(null, null)
                ?? throw new InvalidOperationException($"{memberName} returned null for {typeName}"));

        PropertyInfo? property = type.GetProperty(memberName, flags);
        if (property is not null)
            return (string)(property.GetValue(null)
                ?? throw new InvalidOperationException($"{memberName} returned null for {typeName}"));

        FieldInfo? field = type.GetField(memberName, flags);
        if (field is not null)
            return (string)(field.GetValue(null)
                ?? throw new InvalidOperationException($"{memberName} returned null for {typeName}"));

        throw new InvalidOperationException($"Static string member not found: {typeName}.{memberName}");
    }
}
