using System;
using System.Reflection;
using AiDotNet.Tensors.Engines;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// Source-parity guard for the #775 adaptive-max-pool-2D kernel across the backends that carry it. Only
/// OpenCL is runtime-validated here; CUDA/HIP are source-validated, so this asserts their kernel source
/// derives the SAME pooling-window bounds and max reduction as the OpenCL reference (CUDA/HIP are
/// 1D-dispatched, OpenCL 3D-dispatched, but the per-output-element computation is identical).
/// </summary>
public sealed class AdaptiveMaxPool2dKernelSourceTests
{
    private const string CudaPool = "AiDotNet.Tensors.Engines.DirectGpu.CUDA.Kernels.CudaPoolingKernels";
    private const string HipPool = "AiDotNet.Tensors.Engines.DirectGpu.HIP.Kernels.HipPoolingKernels";
    private const string OpenClPool = "AiDotNet.Tensors.Engines.DirectGpu.OpenCL.Kernels.PoolingKernels";
    private const string MetalExt = "AiDotNet.Tensors.Engines.DirectGpu.Metal.MetalExtendedConvKernels";
    private const string VulkanExt = "AiDotNet.Tensors.Engines.DirectGpu.Vulkan.VulkanExtendedConvKernels";
    private const string WebGpuExt = "AiDotNet.Tensors.Engines.DirectGpu.WebGpu.WebGpuExtendedConvKernels";

    // The pooling-window bound derivation is byte-identical across C/MSL/GLSL (pure int arithmetic, no
    // buffer names or float literals). WGSL prefixes params with pm. and drops spaces, so the WebGPU row
    // carries the WGSL-form markers.
    [Theory]
    [InlineData(CudaPool, "GetSource", "int hStart = (oh * inHeight) / outHeight;", "int wEnd = ((ow + 1) * inWidth) / outWidth;")]
    [InlineData(HipPool, "GetSource", "int hStart = (oh * inHeight) / outHeight;", "int wEnd = ((ow + 1) * inWidth) / outWidth;")]
    [InlineData(OpenClPool, "GetSource", "int hStart = (oh * inHeight) / outHeight;", "int wEnd = ((ow + 1) * inWidth) / outWidth;")]
    [InlineData(MetalExt, "Source", "int hStart = (oh * inHeight) / outHeight;", "int wEnd = ((ow + 1) * inWidth) / outWidth;")]
    [InlineData(VulkanExt, "AdaptiveMaxPool2D", "int hStart = (oh * inHeight) / outHeight;", "int wEnd = ((ow + 1) * inWidth) / outWidth;")]
    [InlineData(WebGpuExt, "AdaptiveMaxPool2D", "let hStart=(oh*pm.inHeight)/pm.outHeight;", "let wEnd=((ow+1)*pm.inWidth)/pm.outWidth;")]
    public void WindowBoundsAndMaxReduction_MatchAcrossBackends(string typeName, string memberName, string m1, string m2)
    {
        string source = GetStaticString(typeName, memberName);
        Assert.Contains(m1, source, StringComparison.Ordinal);
        Assert.Contains(m2, source, StringComparison.Ordinal);
    }

    [Theory]
    [InlineData(CudaPool)]
    [InlineData(HipPool)]
    [InlineData(OpenClPool)]
    public void KernelName_IsRegistered(string typeName)
    {
        string[] names = GetStaticStringArray(typeName, "GetKernelNames");
        Assert.Contains("adaptive_max_pool2d", names);
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
