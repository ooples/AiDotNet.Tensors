using System;
using System.Reflection;
using AiDotNet.Tensors.Engines;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

public sealed class EmbeddingKernelSourceTests
{
    [Fact]
    public void HipEmbeddingForward_ReadsUploadedIntIndexBuffer()
    {
        string source = GetKernelSource("AiDotNet.Tensors.Engines.DirectGpu.HIP.Kernels.HipNeuralNetKernels");
        string signature = ExtractFunctionSignature(source, "embedding_forward");

        Assert.Contains("const int* indices", signature);
        Assert.DoesNotContain("const float* indices", signature);
    }

    [Fact]
    public void OpenClEmbeddingForward_ReadsUploadedIntIndexBufferAndUses2DLaunchShape()
    {
        string source = GetKernelSource("AiDotNet.Tensors.Engines.DirectGpu.OpenCL.Kernels.NeuralNetKernels");
        string signature = ExtractFunctionSignature(source, "embedding_lookup");

        Assert.Contains("__global const int* indices", signature);
        Assert.DoesNotContain("__global const float* indices", signature);
        Assert.Contains("const int d = get_global_id(0);", source);
        Assert.Contains("const int idx = get_global_id(1);", source);
    }

    private static string GetKernelSource(string typeName)
    {
        var type = typeof(DirectGpuTensorEngine).Assembly.GetType(typeName)
            ?? throw new InvalidOperationException($"Kernel source type not found: {typeName}");
        var method = type.GetMethod("GetSource", BindingFlags.Public | BindingFlags.Static)
            ?? throw new InvalidOperationException($"GetSource method not found on {typeName}");
        return (string)(method.Invoke(null, null)
            ?? throw new InvalidOperationException($"GetSource returned null for {typeName}"));
    }

    private static string ExtractFunctionSignature(string source, string functionName)
    {
        int nameIndex = source.IndexOf(functionName, StringComparison.Ordinal);
        if (nameIndex < 0)
            throw new InvalidOperationException($"Function not found: {functionName}");

        int openParen = source.IndexOf('(', nameIndex);
        int closeParen = source.IndexOf(')', openParen);
        if (openParen < 0 || closeParen < 0)
            throw new InvalidOperationException($"Function signature not found: {functionName}");

        return source.Substring(nameIndex, closeParen - nameIndex + 1);
    }
}
