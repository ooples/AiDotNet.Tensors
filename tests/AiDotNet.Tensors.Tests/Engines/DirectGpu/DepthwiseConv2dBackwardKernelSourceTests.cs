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
    private const string MetalExt = "AiDotNet.Tensors.Engines.DirectGpu.Metal.MetalExtendedConvKernels";
    private const string VulkanExt = "AiDotNet.Tensors.Engines.DirectGpu.Vulkan.VulkanExtendedConvKernels";

    // The weights-gather line is byte-identical everywhere (weights buffer name is the same in GLSL).
    [Theory]
    [InlineData(CudaConv, "GetSource")]
    [InlineData(HipConv, "GetSource")]
    [InlineData(OpenClConv, "GetSource")]
    [InlineData(MetalExt, "Source")]
    [InlineData(VulkanExt, "DepthwiseConv2DBackwardInput")]
    public void InputGradientWeightGather_MatchesAcrossBackends(string typeName, string memberName)
    {
        string source = GetStaticString(typeName, memberName);
        Assert.Contains("sum += weights[(oc * kH + kh) * kW + kw]", source, StringComparison.Ordinal);
    }

    // GLSL renames the reserved 'input' buffer to input_, so the Vulkan row carries the GLSL-form marker.
    [Theory]
    [InlineData(CudaConv, "GetSource", "sum += input[((b * inC + ic) * H + ih) * W + iw]")]
    [InlineData(HipConv, "GetSource", "sum += input[((b * inC + ic) * H + ih) * W + iw]")]
    [InlineData(OpenClConv, "GetSource", "sum += input[((b * inC + ic) * H + ih) * W + iw]")]
    [InlineData(MetalExt, "Source", "sum += input[((b * inC + ic) * H + ih) * W + iw]")]
    [InlineData(VulkanExt, "DepthwiseConv2DBackwardWeights", "sum += input_[((b * inC + ic) * H + ih) * W + iw]")]
    public void WeightGradientInputGather_MatchesAcrossBackends(string typeName, string memberName, string marker)
    {
        string source = GetStaticString(typeName, memberName);
        Assert.Contains(marker, source, StringComparison.Ordinal);
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
        MethodInfo? method = type.GetMethod(memberName, flags, binder: null, Type.EmptyTypes, modifiers: null);
        if (method is not null) return method.Invoke(null, null);
        FieldInfo? field = type.GetField(memberName, flags);
        if (field is not null) return field.GetValue(null);
        PropertyInfo? property = type.GetProperty(memberName, flags);
        if (property is not null) return property.GetValue(null);
        throw new InvalidOperationException($"Static member not found: {typeName}.{memberName}");
    }
}
