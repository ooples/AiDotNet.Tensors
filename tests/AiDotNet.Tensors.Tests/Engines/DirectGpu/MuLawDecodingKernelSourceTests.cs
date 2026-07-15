using System;
using System.Reflection;
using AiDotNet.Tensors.Engines;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

public sealed class MuLawDecodingKernelSourceTests
{
    [Theory]
    [InlineData("AiDotNet.Tensors.Engines.DirectGpu.CUDA.Kernels.CudaAudioKernels", "GetSource", "const int* __restrict__ input")]
    [InlineData("AiDotNet.Tensors.Engines.DirectGpu.HIP.Kernels.HipAudioKernels", "GetSource", "const int* __restrict__ input")]
    [InlineData("AiDotNet.Tensors.Engines.DirectGpu.OpenCL.OpenClAudioKernels", "GetSource", "__global const int* input")]
    [InlineData("AiDotNet.Tensors.Engines.DirectGpu.Metal.MetalAudioKernels", "Source", "device const int* input")]
    [InlineData("AiDotNet.Tensors.Engines.DirectGpu.Vulkan.VulkanAudioKernels", "MuLawDecoding", "int input_[]")]
    [InlineData("AiDotNet.Tensors.Engines.DirectGpu.WebGpu.WebGpuAudioKernels", "MuLawDecoding", "array<i32>")]
    public void EveryBackendConsumesResidentInt32Codes(
        string typeName, string memberName, string expectedDeclaration)
    {
        string source = GetStaticString(typeName, memberName);

        Assert.Contains(expectedDeclaration, source, StringComparison.Ordinal);
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
