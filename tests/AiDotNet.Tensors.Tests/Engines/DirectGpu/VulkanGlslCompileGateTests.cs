using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using AiDotNet.Tensors.Engines.DirectGpu.Vulkan;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// Every GLSL compute shader the Vulkan backend ships must compile. A shader that does not compile becomes a null
/// pipeline at its first dispatch, so the op throws "libshaderc required" on a machine that has libshaderc. ClassifyFloat
/// and TakeAlongDim were in that state: they named a buffer <c>input</c>/<c>output</c> (reserved words in GLSL) and
/// assigned a bool to an int. No test compiled them, because every hardware test skips without a runtime compiler.
/// This sweeps every static string in the Vulkan backend that starts with a #version directive.
/// </summary>
public sealed class VulkanGlslCompileGateTests
{
    private const string VersionDirective = "#version";

    [SkippableFact]
    public void EveryShippedGlslComputeShader_Compiles()
    {
        using var compiler = new VulkanGlslCompiler();
        Skip.If(!compiler.IsAvailable, "Vulkan's runtime GLSL compiler (libshaderc) is unavailable.");

        var failures = new List<string>();
        int compiled = 0;
        foreach (var (name, source) in ShippedShaders())
        {
            if (compiler.CompileToSpirv(source) is null)
                failures.Add($"{name}: {compiler.LastError}");
            else
                compiled++;
        }

        Assert.True(compiled > 0, "the sweep found no GLSL shaders; it is not looking in the right place.");
        Assert.True(failures.Count == 0,
            $"{failures.Count} of {failures.Count + compiled} Vulkan GLSL shaders do not compile:{Environment.NewLine}"
            + string.Join(Environment.NewLine, failures));
    }

    private static IEnumerable<(string Name, string Source)> ShippedShaders()
    {
        const BindingFlags AnyStatic = BindingFlags.Static | BindingFlags.Public | BindingFlags.NonPublic;
        var vulkanTypes = typeof(VulkanBackend).Assembly.GetTypes()
            .Where(t => t.Namespace == typeof(VulkanBackend).Namespace && !t.ContainsGenericParameters);
        foreach (var type in vulkanTypes)
        {
            foreach (var property in type.GetProperties(AnyStatic).Where(p => p.PropertyType == typeof(string) && p.GetIndexParameters().Length == 0))
                if (ReadOrNull(() => property.GetValue(null)) is string source && IsShader(source))
                    yield return ($"{type.Name}.{property.Name}", source);
            foreach (var field in type.GetFields(AnyStatic).Where(f => f.FieldType == typeof(string)))
                if (ReadOrNull(() => field.GetValue(null)) is string source && IsShader(source))
                    yield return ($"{type.Name}.{field.Name}", source);
        }
    }

    // A complete compute shader: a #version directive and an entry point. Shared headers (a #version block that other
    // sources are appended to) have no main and are compiled as part of the shaders that include them.
    private static bool IsShader(string text)
        => text.TrimStart().StartsWith(VersionDirective, StringComparison.Ordinal) && text.Contains(EntryPoint);

    private const string EntryPoint = "void main";

    private static object? ReadOrNull(Func<object?> read)
    {
        try { return read(); }
        catch (TargetInvocationException) { return null; }
    }
}
