using System;
using System.Reflection;
using AiDotNet.Tensors.Engines;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>Guards the recurrence and masking semantics in every source-only backend.</summary>
public sealed class MesaRoutedScanKernelSourceTests
{
    [Theory]
    [InlineData("AiDotNet.Tensors.Engines.DirectGpu.CUDA.Kernels.CudaMambaKernels", "GetSource")]
    [InlineData("AiDotNet.Tensors.Engines.DirectGpu.HIP.Kernels.HipMambaKernels", "GetSource")]
    [InlineData("AiDotNet.Tensors.Engines.DirectGpu.OpenCL.Kernels.MambaKernels", "GetSource")]
    [InlineData("AiDotNet.Tensors.Engines.DirectGpu.Metal.MetalRecurrenceKernels", "Source")]
    public void CStyleSources_ContainBothCompleteScans(string typeName,string memberName)
    {
        string source=GetStaticString(typeName,memberName);
        Assert.Contains("mesa_scan_forward",source,StringComparison.Ordinal);
        Assert.Contains("routed_diagonal_ssm_scan_forward",source,StringComparison.Ordinal);
        Assert.Contains("denom",source,StringComparison.Ordinal);
        Assert.Contains("covariance",source,StringComparison.OrdinalIgnoreCase);
        Assert.Contains("active*next",source,StringComparison.Ordinal);
        Assert.Contains("active*y",source,StringComparison.Ordinal);
    }

    [Theory]
    [InlineData("AiDotNet.Tensors.Engines.DirectGpu.Vulkan.VulkanRecurrenceKernels", "MesaScan", "RoutedDiagonalSsmScan")]
    [InlineData("AiDotNet.Tensors.Engines.DirectGpu.WebGpu.WebGpuRecurrenceKernels", "MesaScan", "RoutedDiagonalSsmScan")]
    public void ShaderSources_ContainWoodburyAndMaskedRecurrence(string typeName,string mesaMember,string routedMember)
    {
        string mesa=GetStaticString(typeName,mesaMember),routed=GetStaticString(typeName,routedMember);
        Assert.Contains("denom",mesa,StringComparison.Ordinal);
        Assert.Contains("covariance",mesa,StringComparison.OrdinalIgnoreCase);
        Assert.Contains("mask",routed,StringComparison.Ordinal);
        Assert.Contains("active*next",routed,StringComparison.Ordinal);
        Assert.Contains("active*y",routed,StringComparison.Ordinal);
    }

    private static string GetStaticString(string typeName,string memberName)
    {
        Type type=typeof(DirectGpuTensorEngine).Assembly.GetType(typeName)
            ?? throw new InvalidOperationException($"Kernel source type not found: {typeName}");
        const BindingFlags flags=BindingFlags.Public|BindingFlags.NonPublic|BindingFlags.Static;
        MethodInfo? method=type.GetMethod(memberName,flags,binder:null,Type.EmptyTypes,modifiers:null);
        if(method is not null)return (string)(method.Invoke(null,null)??throw new InvalidOperationException($"{memberName} returned null"));
        PropertyInfo? property=type.GetProperty(memberName,flags);
        if(property is not null)return (string)(property.GetValue(null)??throw new InvalidOperationException($"{memberName} returned null"));
        FieldInfo? field=type.GetField(memberName,flags);
        if(field is not null)return (string)(field.GetValue(null)??throw new InvalidOperationException($"{memberName} returned null"));
        throw new InvalidOperationException($"Static member not found: {typeName}.{memberName}");
    }
}
