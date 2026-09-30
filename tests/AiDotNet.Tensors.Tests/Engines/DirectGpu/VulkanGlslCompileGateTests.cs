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
        foreach (var (name, source, readError) in ShippedShaders())
        {
            // A getter that throws is a shader that cannot be built at all; it must fail the gate, not vanish from it.
            if (readError is not null)
            {
                failures.Add($"{name}: reading the shader source threw {readError.GetType().Name}: {readError.Message}");
                continue;
            }
            if (source is null || !IsShader(source)) continue;
            if (compiler.CompileToSpirv(source) is null)
                failures.Add($"{name}: {compiler.LastError}{QuoteFirstErrorLine(source, compiler.LastError)}");
            else
                compiled++;
        }

        Assert.True(compiled > 0, "the sweep found no GLSL shaders; it is not looking in the right place.");
        Assert.True(failures.Count == 0,
            $"{failures.Count} of {failures.Count + compiled} Vulkan GLSL shaders do not compile:{Environment.NewLine}"
            + string.Join(Environment.NewLine, failures));
    }

    [SkippableFact]
    public void NonAsciiSource_CompilesWithItsFullLength()
    {
        // shaderc receives an explicit byte length. Default string marshaling is UTF-8 on Linux, where a multi-byte
        // character made a character-count length too short and cut the end of the shader off - 13 shipped shaders
        // failed there with "unexpected end of file" because of an em dash in a comment. The non-ASCII comment sits
        // right before main's closing lines so any truncation lands on code.
        using var compiler = new VulkanGlslCompiler();
        Skip.If(!compiler.IsAvailable, "Vulkan's runtime GLSL compiler (libshaderc) is unavailable.");
        const string Source =
            "#version 450\n"
            + "layout(local_size_x = 64) in;\n"
            + "layout(std430, binding = 0) buffer B { float data[]; };\n"
            + "void main() {\n"
            + "    // non-ASCII — é ü ∑ → ≥ λ, several bytes each in UTF-8 — before the last statement\n"
            + "    data[gl_GlobalInvocationID.x] = 1.0;\n"
            + "}\n";
        Assert.True(compiler.CompileToSpirv(Source) is not null, $"A shader with non-ASCII comments failed: {compiler.LastError}");
    }

    private static IEnumerable<(string Name, string? Source, Exception? ReadError)> ShippedShaders()
    {
        const BindingFlags AnyStatic = BindingFlags.Static | BindingFlags.Public | BindingFlags.NonPublic;
        var vulkanTypes = typeof(VulkanBackend).Assembly.GetTypes()
            .Where(t => t.Namespace == typeof(VulkanBackend).Namespace && !t.ContainsGenericParameters);
        foreach (var type in vulkanTypes)
        {
            foreach (var property in type.GetProperties(AnyStatic).Where(p => p.PropertyType == typeof(string) && p.GetIndexParameters().Length == 0))
            {
                var (value, error) = Read(() => property.GetValue(null));
                yield return ($"{type.Name}.{property.Name}", value, error);
            }
            foreach (var field in type.GetFields(AnyStatic).Where(f => f.FieldType == typeof(string)))
            {
                var (value, error) = Read(() => field.GetValue(null));
                yield return ($"{type.Name}.{field.Name}", value, error);
            }
        }
    }

    // A complete compute shader: a #version directive and an entry point. Shared headers (a #version block that other
    // sources are appended to) have no main and are compiled as part of the shaders that include them.
    private static bool IsShader(string text)
        => text.TrimStart().StartsWith(VersionDirective, StringComparison.Ordinal) && text.Contains(EntryPoint);

    private const string EntryPoint = "void main";

    private static readonly TimeSpan RegexTimeout = TimeSpan.FromSeconds(1);

    // The source line the compiler's first "kernel.comp:N:" error names, so a failure can be fixed from the report.
    private static string QuoteFirstErrorLine(string source, string? error)
    {
        var match = System.Text.RegularExpressions.Regex.Match(
            error ?? string.Empty, @"kernel\.comp:(\d+):", System.Text.RegularExpressions.RegexOptions.None, RegexTimeout);
        if (!match.Success) return string.Empty;
        var lines = source.Split('\n');
        int line = int.Parse(match.Groups[1].Value, System.Globalization.CultureInfo.InvariantCulture);
        return line >= 1 && line <= lines.Length ? $"    >> {lines[line - 1].Trim()}" : string.Empty;
    }

    // A string member whose getter throws is reported, with its exception, instead of being dropped from the sweep.
    private static (string? Value, Exception? Error) Read(Func<object?> read)
    {
        try { return (read() as string, null); }
        catch (TargetInvocationException ex) { return (null, ex.InnerException ?? ex); }
    }
}
