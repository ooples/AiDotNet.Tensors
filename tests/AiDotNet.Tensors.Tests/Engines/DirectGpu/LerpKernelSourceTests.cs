using System;
using System.Reflection;
using AiDotNet.Tensors.Engines;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

public sealed class LerpKernelSourceTests
{
    [Theory]
    [InlineData("AiDotNet.Tensors.Engines.DirectGpu.CUDA.Kernels.CudaFusedKernels", "GetSource", "fmaf(t, b[idx] - a[idx], a[idx])")]
    [InlineData("AiDotNet.Tensors.Engines.DirectGpu.HIP.Kernels.HipFusedKernels", "GetSource", "fmaf(t, b[idx] - a[idx], a[idx])")]
    [InlineData("AiDotNet.Tensors.Engines.DirectGpu.OpenCL.Kernels.FusedKernels", "GetSource", "fma(t, b[idx] - a[idx], a[idx])")]
    [InlineData("AiDotNet.Tensors.Engines.DirectGpu.Metal.MetalKernels", "ElementWiseKernels", "fma(t, B[gid] - A[gid], A[gid])")]
    [InlineData("AiDotNet.Tensors.Engines.DirectGpu.Vulkan.VulkanGlslKernels", "BinaryElementwise", "fma(v0, y - x, x)")]
    [InlineData("AiDotNet.Tensors.Engines.DirectGpu.WebGpu.WebGpuKernels", "LerpFusedSource", "fma(lp_params.t, lp_b[idx] - lp_a[idx], lp_a[idx])")]
    public void EveryBackendUsesExplicitFusedMultiplyAdd(
        string typeName, string memberName, string expectedExpression)
    {
        string source = GetStaticString(typeName, memberName);

        Assert.Contains(expectedExpression, source, StringComparison.Ordinal);
    }

    [Theory]
    [InlineData("AiDotNet.Tensors.Engines.DirectGpu.CUDA.Kernels.CudaFusedKernels", "GetSource", "fmaf(scaleA, a[idx], scaleB * b[idx])")]
    [InlineData("AiDotNet.Tensors.Engines.DirectGpu.HIP.Kernels.HipFusedKernels", "GetSource", "fmaf(scaleA, a[idx], scaleB * b[idx])")]
    [InlineData("AiDotNet.Tensors.Engines.DirectGpu.OpenCL.Kernels.FusedKernels", "GetSource", "fma(scaleA, a[idx], scaleB * b[idx])")]
    [InlineData("AiDotNet.Tensors.Engines.DirectGpu.Metal.MetalKernels", "ElementWiseKernels", "fma(scaleA, A[gid], scaleB * B[gid])")]
    [InlineData("AiDotNet.Tensors.Engines.DirectGpu.Vulkan.VulkanGlslKernels", "BinaryElementwise", "fma(v0, x, v1 * y)")]
    [InlineData("AiDotNet.Tensors.Engines.DirectGpu.WebGpu.WebGpuKernels", "AddScaledSource", "fma(as_params.scaleA, as_a[idx], as_params.scaleB * as_b[idx])")]
    public void EveryBackendUsesExplicitFusedMultiplyAddForAddScaled(
        string typeName, string memberName, string expectedExpression)
    {
        string source = GetStaticString(typeName, memberName);

        Assert.Contains(expectedExpression, source, StringComparison.Ordinal);
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
