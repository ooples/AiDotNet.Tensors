using System;
using System.Reflection;
using AiDotNet.Tensors.Engines;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

public sealed class PowerKernelSourceTests
{
    [Theory]
    [InlineData(
        "AiDotNet.Tensors.Engines.DirectGpu.CUDA.Kernels.CudaActivationKernels", "GetSource", "void power_scalar(",
        "x < 0.0f && exponent == truncf(exponent)", "powf(-x, exponent)", "fmodf(fabsf(exponent), 2.0f) == 1.0f")]
    [InlineData(
        "AiDotNet.Tensors.Engines.DirectGpu.HIP.Kernels.HipActivationKernels", "GetSource", "void power_scalar(",
        "x < 0.0f && exponent == truncf(exponent)", "powf(-x, exponent)", "fmodf(fabsf(exponent), 2.0f) == 1.0f")]
    [InlineData(
        "AiDotNet.Tensors.Engines.DirectGpu.OpenCL.Kernels.ActivationKernels", "GetSource", "__kernel void power_scalar(",
        "x < 0.0f && exponent == trunc(exponent)", "pow(-x, exponent)", "fmod(fabs(exponent), 2.0f) == 1.0f")]
    [InlineData(
        "AiDotNet.Tensors.Engines.DirectGpu.Metal.MetalKernels", "ElementWiseKernels", "kernel void pow_kernel(",
        "power == trunc(power)", "pow(-x, power)", "fmod(fabs(power), 2.0f) == 1.0f")]
    [InlineData(
        "AiDotNet.Tensors.Engines.DirectGpu.Vulkan.VulkanGlslKernels", "UnaryElementwise", "case 0u:",
        "v0 == trunc(v0)", "pow(-x, v0)", "mod(abs(v0), 2.0) == 1.0")]
    [InlineData(
        "AiDotNet.Tensors.Engines.DirectGpu.WebGpu.WebGpuKernels", "ScalarOpsSource", "fn pow_scalar(",
        "exponent == trunc(exponent)", "pow(-x, exponent)", "select(magnitude, -magnitude, is_odd)")]
    public void EveryAcceleratorPowerKernelHandlesNegativeBasesWithIntegralExponents(
        string typeName,
        string memberName,
        string kernelMarker,
        string integralGuard,
        string magnitudeCalculation,
        string parityCalculation)
    {
        string source = GetStaticString(typeName, memberName);
        int kernelStart = source.IndexOf(kernelMarker, StringComparison.Ordinal);

        Assert.True(kernelStart >= 0, $"Kernel marker not found: {typeName}.{memberName} -> {kernelMarker}");
        string kernel = source.Substring(kernelStart, Math.Min(1200, source.Length - kernelStart));
        Assert.Contains(integralGuard, kernel, StringComparison.Ordinal);
        Assert.Contains(magnitudeCalculation, kernel, StringComparison.Ordinal);
        Assert.Contains(parityCalculation, kernel, StringComparison.Ordinal);
    }

    [Theory]
    [InlineData(
        "AiDotNet.Tensors.Engines.DirectGpu.Metal.MetalKernels", "ElementWiseKernels", "kernel void pow_kernel(",
        "B[gid] = NAN")]
    [InlineData(
        "AiDotNet.Tensors.Engines.DirectGpu.Vulkan.VulkanGlslKernels", "UnaryElementwise", "case 0u:",
        "uintBitsToFloat(0x7fc00000u)")]
    [InlineData(
        "AiDotNet.Tensors.Engines.DirectGpu.WebGpu.WebGpuKernels", "ScalarOpsSource", "fn pow_scalar(",
        "bitcast<f32>(0x7fc00000u)")]
    public void ShaderBackendsExplicitlyReturnNaNForNegativeBasesWithFractionalExponents(
        string typeName,
        string memberName,
        string kernelMarker,
        string nanExpression)
    {
        string source = GetStaticString(typeName, memberName);
        int kernelStart = source.IndexOf(kernelMarker, StringComparison.Ordinal);

        Assert.True(kernelStart >= 0, $"Kernel marker not found: {typeName}.{memberName} -> {kernelMarker}");
        string kernel = source.Substring(kernelStart, Math.Min(1200, source.Length - kernelStart));
        Assert.Contains(nanExpression, kernel, StringComparison.Ordinal);
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
