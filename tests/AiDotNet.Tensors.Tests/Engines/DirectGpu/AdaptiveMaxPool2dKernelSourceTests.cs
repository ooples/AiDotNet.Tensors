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

    [Theory]
    [InlineData(CudaPool)]
    [InlineData(HipPool)]
    [InlineData(OpenClPool)]
    public void WindowBoundsAndMaxReduction_MatchAcrossBackends(string typeName)
    {
        string source = GetStaticString(typeName, "GetSource");
        Assert.Contains("int hStart = (oh * inHeight) / outHeight;", source, StringComparison.Ordinal);
        Assert.Contains("float maxV = -INFINITY;", source, StringComparison.Ordinal);
        Assert.Contains("input[((b * channels + c) * inHeight + ih) * inWidth + iw]", source, StringComparison.Ordinal);
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
        MethodInfo method = type.GetMethod(memberName, flags)
            ?? throw new InvalidOperationException($"Static method not found: {typeName}.{memberName}");
        return method.Invoke(null, null);
    }
}
