using System;
using System.Reflection;
using AiDotNet.Tensors.Engines;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// Source-parity guard for the #775 Depthwise Conv2D backward kernels (input + weight gradient, NCHW,
/// oc = ic*M + m) across the backends that carry them. Only OpenCL is runtime-validated here; CUDA/HIP
/// are source-validated, so this asserts their kernel source performs the SAME gather as the OpenCL
/// reference, catching a divergent blind port.
/// </summary>
public sealed class DepthwiseConv2dBackwardKernelSourceTests
{
    private const string CudaConv = "AiDotNet.Tensors.Engines.DirectGpu.CUDA.Kernels.CudaConvolutionKernels";
    private const string HipConv = "AiDotNet.Tensors.Engines.DirectGpu.HIP.Kernels.HipConvolutionKernels";
    private const string OpenClConv = "AiDotNet.Tensors.Engines.DirectGpu.OpenCL.Kernels.ConvolutionKernels";

    [Theory]
    [InlineData(CudaConv)]
    [InlineData(HipConv)]
    [InlineData(OpenClConv)]
    public void InputGradientWeightGather_MatchesAcrossBackends(string typeName)
    {
        string source = GetStaticString(typeName, "GetSource");
        Assert.Contains("sum += weights[(oc * kH + kh) * kW + kw]", source, StringComparison.Ordinal);
    }

    [Theory]
    [InlineData(CudaConv)]
    [InlineData(HipConv)]
    [InlineData(OpenClConv)]
    public void WeightGradientInputGather_MatchesAcrossBackends(string typeName)
    {
        string source = GetStaticString(typeName, "GetSource");
        Assert.Contains("sum += input[((b * inC + ic) * H + ih) * W + iw]", source, StringComparison.Ordinal);
    }

    [Theory]
    [InlineData(CudaConv)]
    [InlineData(HipConv)]
    [InlineData(OpenClConv)]
    public void BothBackwardKernelNames_AreRegistered(string typeName)
    {
        string[] names = GetStaticStringArray(typeName, "GetKernelNames");
        Assert.Contains("depthwise_conv2d_backward_input", names);
        Assert.Contains("depthwise_conv2d_backward_weights", names);
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
