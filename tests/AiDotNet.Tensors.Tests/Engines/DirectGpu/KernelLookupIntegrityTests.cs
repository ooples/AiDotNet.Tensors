// Copyright (c) CrowdTrainer LLC. All rights reserved.
// Every kernel name a backend resolves must be in its kernel cache once the backend is constructed.
//
// WHY THIS EXISTS. HipBackend looked up max_axis, copy_2d_strided, csr_segmented_{max,min,stddev}, four octonion
// kernels, batched_dot_product, tile_batch and broadcast_multiply_last_axis, and no compiled HIP module provided
// them: some were defined in a source HIP never compiled (HipOctonionKernels, HipDotProductKernels,
// CudaMissingLookupKernels), the rest nowhere. OpenClBackend had the same four octonion kernels in a source it never
// compiled. Each op threw GpuCapabilityUnavailableException and the engine ran it on the CPU, so results stayed
// correct while the device path did no work; TensorAll / TensorAny surfaced it only because their tests count
// fallbacks. A static "is it defined in some source" check (CudaKernelLookupIntegrityTests) cannot see a source that
// is never compiled, so these read the LIVE cache of a constructed backend.
#if NET6_0_OR_GREATER
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.RegularExpressions;
using AiDotNet.Tensors.Engines.DirectGpu.HIP;
using AiDotNet.Tensors.Engines.DirectGpu.OpenCL;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

internal static class KernelLookupIntegrity
{
    private static readonly TimeSpan Timeout = TimeSpan.FromSeconds(10);

    // A method declaration: modifiers, a return type, the name, the parameter list, then a body or an expression body.
    private static readonly Regex MethodPattern = new(
        @"(?m)^[ \t]*(?:(?:public|private|internal|protected|static|unsafe|async|override|virtual|sealed|new|extern)\s+)+[\w<>\[\],\.\?]+(?:\s*<[^>]*>)?\s+(\w+)\s*\(([^)]*)\)\s*(?:where\s+[^{]+)?(\{|=>)",
        RegexOptions.None, Timeout);

    private static readonly Regex CallPattern = new(@"\b(\w+)\s*\(", RegexOptions.None, Timeout);

    private static readonly Regex LiteralPattern = new(@"""([a-z][a-z0-9_]*)""", RegexOptions.None, Timeout);

    // A read of the cache: TryGetValue, ContainsKey or the indexer. An indexer assignment (_kernelCache["x"] = ...) is a
    // registration, not a lookup; counting it made a module compiled only on some devices (OpenCL's mixed-precision
    // kernels, built only where the device reports FP16) fail this test everywhere else.
    private static readonly Regex DirectCachePattern = new(
        @"_kernelCache\s*(?:(?:\.TryGetValue|\.ContainsKey)\s*\(\s*|\[\s*)((?>\w+)|""[a-z][a-z0-9_]*"")(?!\s*\]\s*=(?!=))",
        RegexOptions.None, Timeout);

    private sealed class Method
    {
        public string Name = "";
        public string[] StringParameters = Array.Empty<string>();   // by position; "" where the parameter is not a string
        public string Body = "";
    }

    private static string? RepoRoot()
    {
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir is not null && !Directory.Exists(Path.Combine(dir.FullName, "src", "AiDotNet.Tensors")))
            dir = dir.Parent;
        return dir?.FullName;
    }

    // Splits a parameter or argument list at its top-level commas.
    private static List<string> SplitTopLevel(string list)
    {
        var parts = new List<string>();
        int depth = 0, start = 0;
        bool inString = false;
        for (int i = 0; i < list.Length; i++)
        {
            char ch = list[i];
            if (ch == '"' && (i == 0 || list[i - 1] != '\\')) inString = !inString;
            if (inString) continue;
            if (ch is '(' or '[' or '<' or '{') depth++;
            else if (ch is ')' or ']' or '>' or '}') depth--;
            else if (ch == ',' && depth == 0) { parts.Add(list.Substring(start, i - start).Trim()); start = i + 1; }
        }
        if (list.Trim().Length > 0) parts.Add(list.Substring(start).Trim());
        return parts;
    }

    // The argument list of a call whose opening parenthesis is at openParen, or null when it does not close.
    private static string? Arguments(string text, int openParen)
    {
        int depth = 0;
        bool inString = false;
        for (int i = openParen; i < text.Length; i++)
        {
            char ch = text[i];
            if (ch == '"' && text[i - 1] != '\\') inString = !inString;
            if (inString) continue;
            if (ch == '(') depth++;
            else if (ch == ')' && --depth == 0) return text.Substring(openParen + 1, i - openParen - 1);
        }
        return null;
    }

    private static string Body(string text, int start, bool expressionBodied)
    {
        int depth = 0;
        bool inString = false;
        for (int i = start; i < text.Length; i++)
        {
            char ch = text[i];
            if (ch == '"' && text[i - 1] != '\\') inString = !inString;
            if (inString) continue;
            if (expressionBodied)
            {
                if (ch is '(' or '{') depth++;
                else if (ch is ')' or '}') depth--;
                else if (ch == ';' && depth == 0) return text.Substring(start, i - start);
            }
            else if (ch == '{') depth++;
            else if (ch == '}' && --depth == 0) return text.Substring(start, i - start + 1);
        }
        return text.Substring(start);
    }

    private static List<Method> Methods(string text)
    {
        var methods = new List<Method>();
        foreach (Match m in MethodPattern.Matches(text))
        {
            var parameters = SplitTopLevel(m.Groups[2].Value);
            var names = new string[parameters.Count];
            for (int i = 0; i < parameters.Count; i++)
            {
                var words = parameters[i].Split(new[] { ' ', '\t', '\n' }, StringSplitOptions.RemoveEmptyEntries);
                int eq = Array.IndexOf(words, "=");
                if (eq >= 0) words = words.Take(eq).ToArray();
                names[i] = words.Length >= 2 && (words[words.Length - 2] is "string" or "string?") ? words[words.Length - 1] : "";
            }
            bool expression = m.Groups[3].Value == "=>";
            methods.Add(new Method
            {
                Name = m.Groups[1].Value,
                StringParameters = names,
                Body = Body(text, m.Groups[3].Index + (expression ? 2 : 0), expression),
            });
        }
        return methods;
    }

    // The kernel names an argument expression can evaluate to: a literal, or the literals of a conditional. Anything
    // built at run time (concatenation, interpolation) is not a name this guard can check.
    private static IEnumerable<string> LiteralNames(string argument)
    {
        if (argument.Contains('+') || argument.Contains('$')) yield break;
        foreach (Match lit in LiteralPattern.Matches(argument)) yield return lit.Groups[1].Value;
    }

    /// <summary>
    /// Every kernel name the backend's source files resolve, or null when the sources are not present. A resolver is
    /// a method that indexes the kernel cache with one of its string parameters, or passes one to another resolver
    /// (found to a fixpoint), so a name passed through helpers such as <c>ExecuteUnary("negate_vector", ...)</c> or
    /// <c>RequireResidentKernel(deterministic ? "a" : "b")</c> is found at its call site.
    /// </summary>
    internal static SortedSet<string>? LookedUp(string backendFolder, string filePattern)
    {
        string? root = RepoRoot();
        if (root is null) return null;
        string folder = Path.Combine(root, "src", "AiDotNet.Tensors", "Engines", "DirectGpu", backendFolder);
        string[] files = Directory.GetFiles(folder, filePattern, SearchOption.TopDirectoryOnly);
        Assert.NotEmpty(files);
        var methods = files.SelectMany(f => Methods(File.ReadAllText(f).Replace("\r\n", "\n"))).ToList();

        var names = new SortedSet<string>(StringComparer.Ordinal);
        // resolver name -> argument positions that carry a kernel name
        var resolvers = new Dictionary<string, HashSet<int>>(StringComparer.Ordinal);
        foreach (var method in methods)
        {
            foreach (Match m in DirectCachePattern.Matches(method.Body))
            {
                string key = m.Groups[1].Value;
                if (key.StartsWith("\"", StringComparison.Ordinal)) { names.Add(key.Trim('"')); continue; }
                int index = Array.IndexOf(method.StringParameters, key);
                if (index < 0) continue;
                if (!resolvers.TryGetValue(method.Name, out var set)) resolvers[method.Name] = set = new HashSet<int>();
                set.Add(index);
            }
        }

        bool grew = true;
        while (grew)
        {
            grew = false;
            foreach (var method in methods)
            {
                foreach (Match call in CallPattern.Matches(method.Body))
                {
                    if (!resolvers.TryGetValue(call.Groups[1].Value, out var positions)) continue;
                    string? args = Arguments(method.Body, call.Index + call.Length - 1);
                    if (args is null) continue;
                    var parts = SplitTopLevel(args);
                    foreach (int position in positions.ToArray())
                    {
                        if (position >= parts.Count) continue;
                        string argument = parts[position];
                        foreach (string name in LiteralNames(argument)) names.Add(name);
                        int forwarded = Array.IndexOf(method.StringParameters, argument);
                        if (forwarded < 0) continue;
                        if (!resolvers.TryGetValue(method.Name, out var own)) resolvers[method.Name] = own = new HashSet<int>();
                        if (own.Add(forwarded)) grew = true;
                    }
                }
            }
        }
        return names;
    }
    internal static void AssertAllCompiled(string backend, IEnumerable<string> lookedUp, Func<string, bool> hasCompiledKernel)
    {
        var missing = lookedUp.Where(name => !hasCompiledKernel(name)).ToList();
        Assert.True(missing.Count == 0,
            $"{backend} resolves {missing.Count} kernel name(s) that no compiled module registered: " +
            $"{string.Join(", ", missing)}. Each op throws GpuCapabilityUnavailableException and the engine runs it " +
            "on the CPU. Compile the source that defines it (and load the name), or point the lookup at the real " +
            "kernel with the same arguments and launch geometry.");
    }
}

[Collection("DirectGpuSerial")]
public sealed class HipKernelLookupIntegrityTests : IClassFixture<HipTensorEngineTestFixture>
{
    private readonly HipTensorEngineTestFixture _fixture;

    public HipKernelLookupIntegrityTests(HipTensorEngineTestFixture fixture) => _fixture = fixture;

    [SkippableFact]
    public void EveryHipKernelLookupIsCompiled()
    {
        var lookedUp = KernelLookupIntegrity.LookedUp("HIP", "HipBackend*.cs");
        Skip.If(lookedUp is null, "Repository sources not present.");
        // The positive control: the names this guard was written for must be among the ones it reads.
        // Names that reach the cache only through helpers or a conditional are among them too.
        foreach (string known in new[] { "max_axis", "copy_2d_strided", "csr_segmented_stddev", "octonion_linear_backward_input", "batched_dot_product", "bx_repeat_elements", "dot_product_deterministic", "strided_dot_product" })
            Assert.Contains(known, lookedUp);
        Skip.IfNot(_fixture.IsAvailable, "No HIP device.");
        var backend = _fixture.Engine?.GetBackend() as HipBackend
            ?? throw new InvalidOperationException("The HIP fixture's engine is not on a HipBackend.");
        KernelLookupIntegrity.AssertAllCompiled("HipBackend", lookedUp, backend.HasCompiledKernel);
    }
}

[Collection("OpenClBackend")]
public sealed class OpenClKernelLookupIntegrityTests : IClassFixture<OpenClBackendTestFixture>
{
    private readonly OpenClBackendTestFixture _fixture;

    public OpenClKernelLookupIntegrityTests(OpenClBackendTestFixture fixture) => _fixture = fixture;

    [SkippableFact]
    public void EveryOpenClKernelLookupIsCompiled()
    {
        var lookedUp = KernelLookupIntegrity.LookedUp("OpenCL", "OpenClBackend*.cs");
        Skip.If(lookedUp is null, "Repository sources not present.");
        foreach (string known in new[] { "octonion_linear_backward_input", "negate_vector", "parity210_cumsum_axis" })
            Assert.Contains(known, lookedUp);
        Skip.IfNot(_fixture.IsAvailable, "No OpenCL device.");
        KernelLookupIntegrity.AssertAllCompiled("OpenClBackend", lookedUp, _fixture.Backend.HasCompiledKernel);
    }
}
#endif