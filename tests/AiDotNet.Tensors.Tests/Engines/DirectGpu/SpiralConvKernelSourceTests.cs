using System;
using System.Reflection;
using AiDotNet.Tensors.Engines;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// Source-parity guard for the #775 SpiralConv (mesh convolution) kernels (forward + input/weight
/// backward) across the backends that carry them. Only OpenCL is runtime-validated here; CUDA/HIP are
/// source-validated, so this asserts their kernel source performs the SAME neighbour gather + matmul as
/// the OpenCL reference, catching a divergent blind port.
/// </summary>
public sealed class SpiralConvKernelSourceTests
{
    private const string CudaConv = "AiDotNet.Tensors.Engines.DirectGpu.CUDA.Kernels.CudaConvolutionKernels";
    private const string HipConv = "AiDotNet.Tensors.Engines.DirectGpu.HIP.Kernels.HipConvolutionKernels";
    private const string OpenClConv = "AiDotNet.Tensors.Engines.DirectGpu.OpenCL.Kernels.ConvolutionKernels";

    [Theory]
    [InlineData(CudaConv)]
    [InlineData(HipConv)]
    [InlineData(OpenClConv)]
    public void ForwardNeighbourGatherMatmul_MatchesAcrossBackends(string typeName)
    {
        string source = GetStaticString(typeName, "GetSource");
        Assert.Contains("sum += vertexFeatures[neighborIdx * inC + c] * weights[oc * gatheredSize + gatherOffset + c];",
            source, StringComparison.Ordinal);
    }

    [Theory]
    [InlineData(CudaConv)]
    [InlineData(HipConv)]
    [InlineData(OpenClConv)]
    public void BackwardWeightsGather_MatchesAcrossBackends(string typeName)
    {
        string source = GetStaticString(typeName, "GetSource");
        Assert.Contains("sum += gradOutput[v * outC + oc] * vertexFeatures[neighborIdx * inC + ic];",
            source, StringComparison.Ordinal);
    }

    [Theory]
    [InlineData(CudaConv)]
    [InlineData(HipConv)]
    [InlineData(OpenClConv)]
    public void AllThreeKernelNames_AreRegistered(string typeName)
    {
        string[] names = GetStaticStringArray(typeName, "GetKernelNames");
        Assert.Contains("spiral_conv", names);
        Assert.Contains("spiral_conv_backward_input", names);
        Assert.Contains("spiral_conv_backward_weights", names);
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
