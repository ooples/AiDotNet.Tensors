using System;
using System.Reflection;
using AiDotNet.Tensors.Engines;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// Source-parity guard for the #775 Conv3D backward kernels (input + weight gradient, NCDHW no dilation).
/// The kernels already existed in every backend's neural-net module (this family's port was residency
/// wiring, not new kernels); only OpenCL is runtime-validated here, so this asserts the CUDA/HIP kernel
/// source performs the SAME gather as the OpenCL reference, validating the pre-existing blind mirrors.
/// </summary>
public sealed class Conv3dBackwardKernelSourceTests
{
    private const string CudaConv = "AiDotNet.Tensors.Engines.DirectGpu.CUDA.Kernels.CudaNeuralNetKernels";
    private const string HipConv = "AiDotNet.Tensors.Engines.DirectGpu.HIP.Kernels.HipNeuralNetKernels";
    private const string OpenClNeural = "AiDotNet.Tensors.Engines.DirectGpu.OpenCL.Kernels.NeuralNetKernels";
    private const string MetalExt = "AiDotNet.Tensors.Engines.DirectGpu.Metal.MetalExtendedConvKernels";
    private const string VulkanExt = "AiDotNet.Tensors.Engines.DirectGpu.Vulkan.VulkanExtendedConvKernels";
    private const string WebGpuExt = "AiDotNet.Tensors.Engines.DirectGpu.WebGpu.WebGpuExtendedConvKernels";

    private const string KernelIdxC = "int kernelIdx = ((oc * inC + ic) * kD + kd) * kH * kW + kh * kW + kw;";
    private const string KernelIdxWgsl = "let kernelIdx=((oc*pm.inC+ic)*pm.kD+kd)*pm.kH*pm.kW+kh*pm.kW+kw;";

    [Theory]
    [InlineData(CudaConv, "GetSource", KernelIdxC)]
    [InlineData(HipConv, "GetSource", KernelIdxC)]
    [InlineData(OpenClNeural, "GetSource", KernelIdxC)]
    [InlineData(MetalExt, "Source", KernelIdxC)]
    [InlineData(VulkanExt, "Conv3DBackwardInput", KernelIdxC)]
    [InlineData(WebGpuExt, "Conv3DBackwardInput", KernelIdxWgsl)]
    public void InputGradientWeightIndex_MatchesAcrossBackends(string typeName, string memberName, string marker)
    {
        string source = GetStaticString(typeName, memberName);
        Assert.Contains(marker, source, StringComparison.Ordinal);
    }

    private const string InputIdxC = "int inputIdx = ((n * inC + ic) * D + d) * H * W + h * W + w;";
    private const string InputIdxWgsl = "let inputIdx=((n*pm.inC+ic)*pm.D+d)*pm.H*pm.W+h*pm.W+w;";

    [Theory]
    [InlineData(CudaConv, "GetSource", InputIdxC)]
    [InlineData(HipConv, "GetSource", InputIdxC)]
    [InlineData(OpenClNeural, "GetSource", InputIdxC)]
    [InlineData(MetalExt, "Source", InputIdxC)]
    [InlineData(VulkanExt, "Conv3DBackwardWeights", InputIdxC)]
    [InlineData(WebGpuExt, "Conv3DBackwardWeights", InputIdxWgsl)]
    public void WeightGradientInputIndex_MatchesAcrossBackends(string typeName, string memberName, string marker)
    {
        string source = GetStaticString(typeName, memberName);
        Assert.Contains(marker, source, StringComparison.Ordinal);
    }

    // CUDA/HIP register the two backward kernels (OpenCL registers them in the neural-net module).
    [Theory]
    [InlineData(CudaConv)]
    [InlineData(HipConv)]
    public void BothBackwardKernelNames_AreRegistered(string typeName)
    {
        string[] names = GetStaticStringArray(typeName, "GetKernelNames");
        Assert.Contains("conv3d_backward_input", names);
        Assert.Contains("conv3d_backward_weights", names);
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
