using System;
using System.Reflection;
using AiDotNet.Tensors.Engines;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

public sealed class AttentionMaskKernelSourceTests
{
    [Theory]
    [InlineData("AiDotNet.Tensors.Engines.DirectGpu.CUDA.Kernels.CudaAttentionKernels")]
    [InlineData("AiDotNet.Tensors.Engines.DirectGpu.HIP.Kernels.HipAttentionKernels")]
    [InlineData("AiDotNet.Tensors.Engines.DirectGpu.OpenCL.Kernels.AttentionKernels")]
    public void CStyleAttentionKernels_IndexSharedOrFullBooleanMasks(string typeName)
    {
        string source = GetStaticString(typeName, "GetSource");

        Assert.Contains("int maskMode", source);
        Assert.Contains("maskMode == 2 ? bh * seqQ * seqK : 0", source);
        Assert.Contains("mask[maskOffset + ki] == 0.0f", source);
    }

    [Fact]
    public void MetalAttentionKernel_IndexesSharedOrFullBooleanMasks()
    {
        string source = GetStaticString(
            "AiDotNet.Tensors.Engines.DirectGpu.Metal.MetalResidentKernels", "Source");

        Assert.Contains("constant uint& maskMode", source);
        Assert.Contains("maskMode == 2u ? (b * heads + h) * queryLength * keyLength : 0u", source);
        Assert.Contains("mask[maskOffset + i * keyLength + j] == 0.0f", source);
    }

    [Fact]
    public void VulkanAttentionKernel_IndexesSharedOrFullBooleanMasks()
    {
        string source = GetStaticString(
            "AiDotNet.Tensors.Engines.DirectGpu.Vulkan.VulkanGlslKernels", "AttentionForward");

        Assert.Contains("uint maskMode; uint maskBatchStride", source);
        Assert.Contains("maskMode == 2u ? batchIndex * maskBatchStride + queryHead * seqQ * seqK : 0u", source);
        Assert.Contains("maskData[maskOffset + queryIndex * seqK + keyIndex] == 0.0", source);
    }

    [Fact]
    public void WebGpuAttentionKernel_IndexesSharedOrFullBooleanMasks()
    {
        string source = GetStaticString(
            "AiDotNet.Tensors.Engines.DirectGpu.WebGpu.WebGpuKernels", "AttentionSource");

        Assert.Contains("boolean_mask_mode: u32", source);
        Assert.Contains("attention_params.boolean_mask_mode == 2u", source);
        Assert.Contains("attention_bias[mask_base + q_pos * attention_params.seq_k + k_pos] == 0.0", source);
    }

    private static string GetStaticString(string typeName, string memberName)
    {
        Type type = typeof(DirectGpuTensorEngine).Assembly.GetType(typeName)
            ?? throw new InvalidOperationException($"Kernel source type not found: {typeName}");
        MethodInfo? method = type.GetMethod(memberName, BindingFlags.Public | BindingFlags.Static);
        if (method is not null)
            return (string)(method.Invoke(null, null)
                ?? throw new InvalidOperationException($"{memberName} returned null for {typeName}"));

        PropertyInfo? property = type.GetProperty(memberName, BindingFlags.Public | BindingFlags.Static);
        if (property is not null)
            return (string)(property.GetValue(null)
                ?? throw new InvalidOperationException($"{memberName} returned null for {typeName}"));

        FieldInfo? field = type.GetField(memberName, BindingFlags.Public | BindingFlags.Static);
        if (field is not null)
            return (string)(field.GetValue(null)
                ?? throw new InvalidOperationException($"{memberName} returned null for {typeName}"));

        throw new InvalidOperationException($"Static string member not found: {typeName}.{memberName}");
    }
}
