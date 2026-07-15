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

    // The trilinear forward 8-corner weights factorize identically in every backend's source.
    [Theory]
    [InlineData(CudaConv)]
    [InlineData(HipConv)]
    [InlineData(OpenClConv)]
    public void ForwardEightCornerWeights_MatchAcrossBackends(string typeName)
    {
        string source = GetStaticString(typeName, "GetSource");
        Assert.Contains("w000 = (1 - fz) * (1 - fy) * (1 - fx)", source, StringComparison.Ordinal);
        Assert.Contains("w111 = fz * fy * fx", source, StringComparison.Ordinal);
    }

    // The backward gather weight per axis is identical in every backend's source.
    [Theory]
    [InlineData(CudaConv)]
    [InlineData(HipConv)]
    [InlineData(OpenClConv)]
    public void BackwardPerAxisGatherWeight_MatchesAcrossBackends(string typeName)
    {
        string source = GetStaticString(typeName, "GetSource");
        Assert.Contains("float wz = (gz == z0 ? (1.0f - fz) : 0.0f) + (gz == z1 ? fz : 0.0f);", source, StringComparison.Ordinal);
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
        MethodInfo method = type.GetMethod(memberName, flags)
            ?? throw new InvalidOperationException($"Static method not found: {typeName}.{memberName}");
        return method.Invoke(null, null);
    }
}
