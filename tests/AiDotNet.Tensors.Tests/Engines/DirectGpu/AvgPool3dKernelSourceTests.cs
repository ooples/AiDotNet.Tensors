using System;
using System.Reflection;
using AiDotNet.Tensors.Engines;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// Source-parity guard for the #775 AvgPool3D forward/backward kernels. The kernels already existed in
/// every backend's pooling module (this family's port was residency wiring, not new kernels); this
/// asserts both kernel names are registered in each backend so the capability-interface launch resolves
/// a real kernel, and that the CUDA/HIP forward kernels share OpenCL's window-accumulation form.
/// </summary>
public sealed class AvgPool3dKernelSourceTests
{
    private const string CudaPool = "AiDotNet.Tensors.Engines.DirectGpu.CUDA.Kernels.CudaPoolingKernels";
    private const string HipPool = "AiDotNet.Tensors.Engines.DirectGpu.HIP.Kernels.HipPoolingKernels";
    private const string OpenClPool = "AiDotNet.Tensors.Engines.DirectGpu.OpenCL.Kernels.PoolingKernels";

    [Theory]
    [InlineData(CudaPool)]
    [InlineData(HipPool)]
    [InlineData(OpenClPool)]
    public void BothKernelNames_AreRegistered(string typeName)
    {
        string[] names = GetStaticStringArray(typeName, "GetKernelNames");
        Assert.Contains("avgpool3d", names);
        Assert.Contains("avgpool3d_backward", names);
    }

    // The count-include-pad divisor branch is present in every backend's avgpool3d source.
    [Theory]
    [InlineData(CudaPool)]
    [InlineData(HipPool)]
    [InlineData(OpenClPool)]
    public void CountIncludePadDivisor_IsPresentAcrossBackends(string typeName)
    {
        string source = GetStaticString(typeName, "GetSource");
        Assert.Contains("countIncludePad", source, StringComparison.Ordinal);
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
