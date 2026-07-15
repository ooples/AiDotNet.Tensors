using System;
using System.Reflection;
using AiDotNet.Tensors.Engines;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// Source-parity guard for the #775 trilinear-interpolation kernels across the backends that carry
/// them. Only OpenCL is runtime-validated on this machine; CUDA/HIP are source-validated, so this
/// asserts their kernel source computes the SAME formula as the OpenCL reference (the 8-corner
/// factorized weights forward + the per-axis gather backward), catching a divergent blind port.
/// </summary>
public sealed class TrilinearKernelSourceTests
{
    private const string CudaConv = "AiDotNet.Tensors.Engines.DirectGpu.CUDA.Kernels.CudaConvolutionKernels";
    private const string HipConv = "AiDotNet.Tensors.Engines.DirectGpu.HIP.Kernels.HipConvolutionKernels";
    private const string OpenClConv = "AiDotNet.Tensors.Engines.DirectGpu.OpenCL.Kernels.ConvolutionKernels";
    private const string MetalExt = "AiDotNet.Tensors.Engines.DirectGpu.Metal.MetalExtendedConvKernels";
    private const string VulkanExt = "AiDotNet.Tensors.Engines.DirectGpu.Vulkan.VulkanExtendedConvKernels";

    // The trilinear forward 8-corner weights factorize identically in every backend's source
    // (C/MSL/GLSL all agree — no float literals, only int 1). OpenCL/CUDA/HIP expose GetSource(), Metal
    // the Source field, Vulkan a per-kernel GLSL property (TrilinearInterpolate).
    [Theory]
    [InlineData(CudaConv, "GetSource")]
    [InlineData(HipConv, "GetSource")]
    [InlineData(OpenClConv, "GetSource")]
    [InlineData(MetalExt, "Source")]
    [InlineData(VulkanExt, "TrilinearInterpolate")]
    public void ForwardEightCornerWeights_MatchAcrossBackends(string typeName, string memberName)
    {
        string source = GetStaticString(typeName, memberName);
        Assert.Contains("w000 = (1 - fz) * (1 - fy) * (1 - fx)", source, StringComparison.Ordinal);
        Assert.Contains("w111 = fz * fy * fx", source, StringComparison.Ordinal);
    }

    // The backward gather weight per axis matches each backend's source; GLSL drops the `f` float suffix,
    // so the Vulkan row carries the GLSL-form marker.
    [Theory]
    [InlineData(CudaConv, "GetSource", "float wz = (gz == z0 ? (1.0f - fz) : 0.0f) + (gz == z1 ? fz : 0.0f);")]
    [InlineData(HipConv, "GetSource", "float wz = (gz == z0 ? (1.0f - fz) : 0.0f) + (gz == z1 ? fz : 0.0f);")]
    [InlineData(OpenClConv, "GetSource", "float wz = (gz == z0 ? (1.0f - fz) : 0.0f) + (gz == z1 ? fz : 0.0f);")]
    [InlineData(MetalExt, "Source", "float wz = (gz == z0 ? (1.0f - fz) : 0.0f) + (gz == z1 ? fz : 0.0f);")]
    [InlineData(VulkanExt, "TrilinearInterpolateBackward", "float wz = (gz == z0 ? (1.0 - fz) : 0.0) + (gz == z1 ? fz : 0.0);")]
    public void BackwardPerAxisGatherWeight_MatchesAcrossBackends(string typeName, string memberName, string marker)
    {
        string source = GetStaticString(typeName, memberName);
        Assert.Contains(marker, source, StringComparison.Ordinal);
    }

    // Every backend that carries the kernels must register both names so the launch can resolve them.
    [Theory]
    [InlineData(CudaConv)]
    [InlineData(HipConv)]
    [InlineData(OpenClConv)]
    public void BothKernelNames_AreRegistered(string typeName)
    {
        string[] names = GetStaticStringArray(typeName, "GetKernelNames");
        Assert.Contains("trilinear_interpolate", names);
        Assert.Contains("trilinear_interpolate_backward", names);
    }

    private static string GetStaticString(string typeName, string memberName) =>
        (string)(InvokeStatic(typeName, memberName)
            ?? throw new InvalidOperationException($"{memberName} returned null for {typeName}"));

    private static string[] GetStaticStringArray(string typeName, string memberName) =>
        (string[])(InvokeStatic(typeName, memberName)
            ?? throw new InvalidOperationException($"{memberName} returned null for {typeName}"));

    private static object? InvokeStatic(string typeName, string memberName)
    {
        Type type = typeof(DirectGpuTensorEngine).Assembly.GetType(typeName)
            ?? throw new InvalidOperationException($"Kernel source type not found: {typeName}");
        const BindingFlags flags = BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Static;
        MethodInfo? method = type.GetMethod(memberName, flags, binder: null, Type.EmptyTypes, modifiers: null);
        if (method is not null) return method.Invoke(null, null);
        FieldInfo? field = type.GetField(memberName, flags);
        if (field is not null) return field.GetValue(null);
        PropertyInfo? property = type.GetProperty(memberName, flags);
        if (property is not null) return property.GetValue(null);
        throw new InvalidOperationException($"Static member not found: {typeName}.{memberName}");
    }
}
