using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Text.RegularExpressions;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// A kernel source line that opens with a lone "/" where "//" was meant is a syntax error in every kernel dialect
/// (CUDA, HIP, OpenCL C, MSL), but only the backend that compiles it at run time notices. On a host without that
/// device, nothing does. This check reads every embedded kernel source without a device, so a broken comment fails
/// on every host.
/// </summary>
public sealed class KernelSourceCommentSyntaxTests
{
    // A line whose first token is a single '/' followed by prose. "/ word word" is never valid kernel code: a division
    // continued onto a new line has an operand there ("/ sqrtf(x)", "/ (a + b)", "/ n;"), not two bare words.
    private static readonly Regex LoneSlashComment = new(@"^\s*/(?![/*=])\s+[A-Za-z][A-Za-z0-9_-]*\s+[A-Za-z(]", RegexOptions.Compiled);

    private static IEnumerable<(string Name, string Source)> EmbeddedKernelSources()
    {
        var assembly = typeof(AiDotNet.Tensors.Engines.AiDotNetEngine).Assembly;
        Type[] types;
        try { types = assembly.GetTypes(); }
        catch (ReflectionTypeLoadException ex) { types = ex.Types.Where(t => t is not null).ToArray()!; }

        const BindingFlags Static = BindingFlags.Static | BindingFlags.Public | BindingFlags.NonPublic;
        foreach (var type in types)
        {
            if (type.Namespace is null || !type.Namespace.StartsWith("AiDotNet.Tensors.Engines", StringComparison.Ordinal))
                continue;
            if (type.ContainsGenericParameters) continue;

            var getSource = type.GetMethod("GetSource", Static, null, Type.EmptyTypes, null);
            if (getSource is not null && getSource.ReturnType == typeof(string))
            {
                string? text = null;
                try { text = (string?)getSource.Invoke(null, null); }
                catch (TargetInvocationException) { }
                if (!string.IsNullOrEmpty(text)) yield return (type.FullName + ".GetSource()", text!);
            }

            foreach (var field in type.GetFields(Static))
            {
                if (field.IsLiteral && field.FieldType == typeof(string) && field.Name.EndsWith("Source", StringComparison.Ordinal)
                    && field.GetRawConstantValue() is string constant && constant.IndexOf('\n') >= 0)
                    yield return (type.FullName + "." + field.Name, constant);
            }
        }
    }

    [Fact]
    public void Embedded_kernel_sources_have_no_single_slash_comment_lines()
    {
        var sources = EmbeddedKernelSources().ToList();
        Assert.True(sources.Count > 100, $"Only {sources.Count} embedded kernel sources were found; the reflection scan is broken.");

        var bad = new List<string>();
        foreach (var (name, source) in sources)
        {
            string[] lines = source.Split('\n');
            for (int i = 0; i < lines.Length; i++)
            {
                if (LoneSlashComment.IsMatch(lines[i]))
                    bad.Add($"{name} line {i + 1}: {lines[i].Trim()}");
            }
        }

        Assert.True(bad.Count == 0, "Kernel source line(s) open with '/' where a '//' comment was meant:\n  " + string.Join("\n  ", bad));
    }
}
