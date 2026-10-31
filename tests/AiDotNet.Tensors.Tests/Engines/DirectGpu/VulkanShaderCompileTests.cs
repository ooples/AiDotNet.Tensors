#if NET6_0_OR_GREATER
using System.Collections.Generic;
using System.Reflection;
using AiDotNet.Tensors.Engines.DirectGpu.Vulkan;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// Every GLSL kernel the Vulkan backend builds from a source string compiles. A kernel that does not is found only
/// when a model first reaches it, as "Failed to create GLSL recurrence pipeline" mid-training: the strided-view gather
/// translated from the shared CUDA C named a parameter <c>layout</c>, a GLSL keyword, and GRU training on Vulkan
/// failed at its first non-contiguous slice. Needs only shaderc, no device.
/// </summary>
public sealed class VulkanShaderCompileTests
{
    [SkippableFact]
    public void TranslatedSolverAndRecurrentKernels_AllCompile()
    {
        using var compiler = new VulkanGlslCompiler();
        Skip.IfNot(compiler.IsAvailable, "shaderc is not available.");
        var failures = new List<string>();
        foreach (var solver in VulkanBatchExecutionKernels.Solvers)
            if (compiler.CompileToSpirv(solver.Value) is null) failures.Add($"{solver.Key}: {compiler.LastError}");
        Assert.True(failures.Count == 0, string.Join("\n", failures));
    }

    [SkippableFact]
    public void ResidentKernels_AllCompile()
    {
        using var compiler = new VulkanGlslCompiler();
        Skip.IfNot(compiler.IsAvailable, "shaderc is not available.");
        var failures = new List<string>();
        int compiled = 0;
        foreach (var property in typeof(VulkanResidentKernels).GetProperties(BindingFlags.Public | BindingFlags.Static))
        {
            if (property.PropertyType != typeof(string) || property.GetValue(null) is not string source) continue;
            compiled++;
            if (compiler.CompileToSpirv(source) is null) failures.Add($"{property.Name}: {compiler.LastError}");
        }
        Assert.True(compiled > 0, "VulkanResidentKernels exposes no kernel sources.");
        Assert.True(failures.Count == 0, string.Join("\n", failures));
    }
}
#endif