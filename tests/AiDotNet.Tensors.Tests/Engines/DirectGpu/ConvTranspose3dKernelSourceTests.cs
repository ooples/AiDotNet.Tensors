using System;
using System.Reflection;
using AiDotNet.Tensors.Engines;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// Source-parity guard for the #775 ConvTranspose3D kernels (forward + input/weight backward) across
/// the backends that carry them. Only OpenCL is runtime-validated here; CUDA/HIP are source-validated,
/// so this asserts their kernel source computes the SAME gather (NCDHW, weights [inC,outC,kD,kH,kW])
/// as the OpenCL reference, catching a divergent blind port.
/// </summary>
public sealed class ConvTranspose3dKernelSourceTests
{
    private const string CudaConv = "AiDotNet.Tensors.Engines.DirectGpu.CUDA.Kernels.CudaConvolutionKernels";
    private const string HipConv = "AiDotNet.Tensors.Engines.DirectGpu.HIP.Kernels.HipConvolutionKernels";
    private const string OpenClConv = "AiDotNet.Tensors.Engines.DirectGpu.OpenCL.Kernels.ConvolutionKernels";
    private const string MetalExt = "AiDotNet.Tensors.Engines.DirectGpu.Metal.MetalExtendedConvKernels";
    private const string VulkanExt = "AiDotNet.Tensors.Engines.DirectGpu.Vulkan.VulkanExtendedConvKernels";
    private const string WebGpuExt = "AiDotNet.Tensors.Engines.DirectGpu.WebGpu.WebGpuExtendedConvKernels";

    // The weight index expression is byte-identical in C/MSL/GLSL (pure index); WGSL prefixes params with
    // pm. and drops spaces, so the WebGPU row carries the WGSL-form marker.
    [Theory]
    [InlineData(CudaConv, "GetSource", "weights[((((ic * outC + oc) * kD + kd) * kH + kh) * kW + kw)]")]
    [InlineData(HipConv, "GetSource", "weights[((((ic * outC + oc) * kD + kd) * kH + kh) * kW + kw)]")]
    [InlineData(OpenClConv, "GetSource", "weights[((((ic * outC + oc) * kD + kd) * kH + kh) * kW + kw)]")]
    [InlineData(MetalExt, "Source", "weights[((((ic * outC + oc) * kD + kd) * kH + kh) * kW + kw)]")]
    [InlineData(VulkanExt, "ConvTranspose3D", "weights[((((ic * outC + oc) * kD + kd) * kH + kh) * kW + kw)]")]
    [InlineData(WebGpuExt, "ConvTranspose3D", "weights[((((ic*pm.outC+oc)*pm.kD+kd)*pm.kH+kh)*pm.kW+kw)]")]
    public void WeightIndexExpression_MatchesAcrossBackends(string typeName, string memberName, string marker)
    {
        string source = GetStaticString(typeName, memberName);
        Assert.Contains(marker, source, StringComparison.Ordinal);
    }

    // The forward transpose-stride test (od + pad - kd, divisible by stride); WGSL row carries WGSL form.
    [Theory]
    [InlineData(CudaConv, "GetSource", "int td = od + padD - kd;", "if (td < 0 || (td % strideD) != 0) continue;")]
    [InlineData(HipConv, "GetSource", "int td = od + padD - kd;", "if (td < 0 || (td % strideD) != 0) continue;")]
    [InlineData(OpenClConv, "GetSource", "int td = od + padD - kd;", "if (td < 0 || (td % strideD) != 0) continue;")]
    [InlineData(MetalExt, "Source", "int td = od + padD - kd;", "if (td < 0 || (td % strideD) != 0) continue;")]
    [InlineData(VulkanExt, "ConvTranspose3D", "int td = od + padD - kd;", "if (td < 0 || (td % strideD) != 0) continue;")]
    [InlineData(WebGpuExt, "ConvTranspose3D", "let td=od+pm.padD-kd;", "if(td<0||(td%pm.strideD)!=0){continue;}")]
    public void ForwardTransposeStrideTest_MatchesAcrossBackends(string typeName, string memberName, string m1, string m2)
    {
        string source = GetStaticString(typeName, memberName);
        Assert.Contains(m1, source, StringComparison.Ordinal);
        Assert.Contains(m2, source, StringComparison.Ordinal);
    }

    [Theory]
    [InlineData(CudaConv)]
    [InlineData(HipConv)]
    [InlineData(OpenClConv)]
    public void AllThreeKernelNames_AreRegistered(string typeName)
    {
        string[] names = GetStaticStringArray(typeName, "GetKernelNames");
        Assert.Contains("conv_transpose3d", names);
        Assert.Contains("conv_transpose3d_backward_input", names);
        Assert.Contains("conv_transpose3d_backward_weights", names);
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
