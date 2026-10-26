// Copyright (c) AiDotNet. All rights reserved.
// The scalar math the declared elementwise kernels call (#890 phase 4a): every hand-written elementwise kernel --
// relu, gelu, sinf, lgammaf, fmodf, leaky_relu(alpha), ... -- turned into a device FUNCTION from its own source, so a
// declared kernel that calls it computes exactly what the kernel it replaces computed: the same expression, compiled by
// NVRTC with the same fast-math setting as the module it came from.
//
// A kernel is derivable when it is the plain elementwise form every one of these kernels has:
//
//     extern "C" __global__ void NAME(const float* in..., float* out, float scalar..., int size)
//     {   int idx = blockIdx.x * blockDim.x + threadIdx.x;  if (idx >= size) return;  ...  out[idx] = EXPR; }
//
// Its inputs become the function's float arguments, its scalars stay scalars, each store to out[idx] becomes a return.
// Anything else -- a kernel that reads idx otherwise, synchronizes, or exits without storing -- is not derivable, and
// stays on its hand-written kernel.
//
// Each module is its own relocatable library (its helpers have external linkage under -rdc, and two modules may define
// the same helper): a declared kernel links the one module its function comes from.

using System;
using System.Collections.Generic;
using System.Globalization;
using System.Text;
using System.Text.RegularExpressions;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ir;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Kernels;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA;

/// <summary>
/// A hand-written module as CudaBackend compiled it: its name, its raw source and whether it was built with fast math.
/// CudaBackend records one per kernel it loads (the LAST module to define a name wins, as its kernel cache does), so
/// a function is always derived from the copy that actually runs.
/// </summary>
internal sealed record CudaMathModule(string Name, string Source, bool FastMath);

/// <summary>A derived library function: the kernel it came from, its module, and its link symbol.</summary>
/// <param name="Parameters">The hand-written kernel's parameters in order: 'i' an input, 'o' the output, 's' a scalar, 'n' the size.</param>
internal sealed record CudaMathFunction(string Kernel, CudaMathModule Module, int Inputs, string[] Scalars, string Wrapper, string Parameters)
{
    /// <summary>The IR handle a declared body calls it by: inputs first, then scalars.</summary>
    internal CodegenLibraryFunction Handle { get; } =
        new("aidn_" + Kernel, Module.Name, Inputs + Scalars.Length);
}

internal static class CudaMathLibrary
{
    private static readonly Dictionary<(string Module, string Kernel), CudaMathFunction?> s_resolved = new();
    private static readonly Dictionary<(string Module, string Kernel), string> s_refusals = new();
    private static readonly object s_lock = new();

    /// <summary>The function derived from <paramref name="kernel"/>'s definition in <paramref name="module"/>, or null.</summary>
    internal static CudaMathFunction? Resolve(string kernel, CudaMathModule module)
    {
        if (module is null) throw new ArgumentNullException(nameof(module));
        var key = (module.Name, kernel);
        lock (s_lock)
        {
            if (s_resolved.TryGetValue(key, out var known)) return known;
            var function = Derive(kernel, module, module.Source, out string reason);
            if (function is null) s_refusals[key] = reason;
            s_resolved[key] = function;
            return function;
        }
    }

    /// <summary>Why <paramref name="kernel"/> in <paramref name="module"/> is not derivable, or null.</summary>
    internal static string? RefusalOf(string kernel, CudaMathModule module)
    {
        lock (s_lock) return s_refusals.TryGetValue((module.Name, kernel), out string? reason) ? reason : null;
    }

    /// <summary>
    /// The translation unit of a module's library: the module's own source (its helpers), then a non-inlined
    /// extern "C" device function per derived kernel.
    /// </summary>
    internal static string LibrarySource(CudaMathModule module, IEnumerable<CudaMathFunction> functions)
    {
        var sb = new StringBuilder(CudaBackend.PrepareNvrtcSource(Internalize(module.Source)));
        sb.Append("\n// ---- the declared kernels' library: one function per derived kernel\n");
        foreach (var function in functions)
            if (function.Module.Name == module.Name) sb.Append(function.Wrapper).Append('\n');
        return sb.ToString();
    }

    /// <summary>
    /// A module's source as a LIBRARY: its __global__ kernels removed (the library never launches them) and every
    /// remaining device function and variable given internal linkage (static). Under -rdc they were external, and a
    /// kernel linking two modules' libraries -- gelu from one, leaky_relu from another -- met each name both define
    /// twice. Only the derived aidn_ wrappers, unique by kernel, stay external.
    /// </summary>
    internal static string Internalize(string source)
    {
        var sb = new StringBuilder(source.Length);
        int at = 0;
        foreach (Match kernel in Regex.Matches(source, @"extern\s+""C""\s+__global__"))
        {
            if (kernel.Index < at) continue;
            int open = source.IndexOf('{', kernel.Index);
            int semicolon = source.IndexOf(';', kernel.Index);
            if (open < 0 || (semicolon >= 0 && semicolon < open)) continue;   // a declaration, not a definition
            int depth = 0, close = -1;
            for (int i = open; i < source.Length; i++)
            {
                if (source[i] == '{') depth++;
                else if (source[i] == '}' && --depth == 0) { close = i; break; }
            }
            if (close < 0) break;
            sb.Append(source, at, kernel.Index - at);
            at = close + 1;
        }
        sb.Append(source, at, source.Length - at);
        string library = sb.ToString();
        // static before a device function or variable that is not already static and not extern "C".
        return Regex.Replace(library,
            @"(?m)^(?<indent>[ \t]*)(?!static\b)(?!extern\b)(?!template\b)(?<head>(?:inline\s+|__forceinline__\s+|__noinline__\s+)*(?:__device__|__constant__)\b)",
            "${indent}static ${head}");
    }

    private static CudaMathFunction? Derive(string kernel, CudaMathModule module, string source, out string reason)
    {
        var header = Regex.Match(source, @"\bvoid\s+" + Regex.Escape(kernel) + @"\s*\(([^)]*)\)\s*\{");
        if (!header.Success) { reason = kernel + ": no definition with a body"; return null; }
        var inputs = new List<string>();
        var scalars = new List<string>();
        var order = new StringBuilder();
        string? output = null, size = null;
        foreach (string raw in header.Groups[1].Value.Split(','))
        {
            string p = Regex.Replace(raw.Replace("__restrict__", ""), @"\s+", " ").Trim();
            Match m;
            if ((m = Regex.Match(p, @"^const float ?\* ?(\w+)$")).Success) { inputs.Add(m.Groups[1].Value); order.Append('i'); }
            else if ((m = Regex.Match(p, @"^float ?\* ?(\w+)$")).Success)
            {
                if (output is not null) { reason = kernel + ": two outputs"; return null; }
                output = m.Groups[1].Value;
                order.Append('o');
            }
            else if ((m = Regex.Match(p, @"^float (\w+)$")).Success) { scalars.Add(m.Groups[1].Value); order.Append('s'); }
            else if ((m = Regex.Match(p, @"^int (\w+)$")).Success)
            {
                if (size is not null) { reason = kernel + ": more than one int parameter"; return null; }
                size = m.Groups[1].Value;
                order.Append('n');
            }
            else { reason = kernel + ": parameter '" + p + "' is not elementwise"; return null; }
        }
        if (output is null || size is null || inputs.Count == 0) { reason = kernel + ": not an elementwise signature"; return null; }

        // The body: brace-matched from the header's '{'.
        int open = header.Index + header.Length - 1, depth = 0, close = -1;
        for (int i = open; i < source.Length; i++)
        {
            if (source[i] == '{') depth++;
            else if (source[i] == '}' && --depth == 0) { close = i; break; }
        }
        if (close < 0) { reason = kernel + ": unbalanced body"; return null; }
        string body = source.Substring(open + 1, close - open - 1);
        // Comments are not code: where_select's mention of "the condition buffer" made the name test below read it
        // as a second use of the condition, so where_select had no derived function (#1441).
        body = Regex.Replace(body, @"/\*.*?\*/", " ", RegexOptions.Singleline);
        body = Regex.Replace(body, @"//[^\r\n]*", "");

        body = Regex.Replace(body, @"^\s*int\s+idx\s*=\s*blockIdx\.x\s*\*\s*blockDim\.x\s*\+\s*threadIdx\.x\s*;\s*$", "",
            RegexOptions.Multiline);
        body = Regex.Replace(body, @"^\s*if\s*\(\s*idx\s*>=\s*" + Regex.Escape(size) + @"\s*\)\s*return\s*;\s*$", "",
            RegexOptions.Multiline);
        for (int k = 0; k < inputs.Count; k++)
            body = Regex.Replace(body, @"\b" + Regex.Escape(inputs[k]) + @"\s*\[\s*idx\s*\]", "aidn_v" + k.ToString(CultureInfo.InvariantCulture));
        body = Regex.Replace(body, @"\b" + Regex.Escape(output) + @"\s*\[\s*idx\s*\]\s*=(?!=)", "return ");
        // "out[idx] = X; return;" became "return X; return;": the second is unreachable, and invalid in a float function.
        body = Regex.Replace(body, @"(return [^;]*;)\s*return\s*;", "$1");

        // A tensor read anywhere but at idx (mean[0]), or the length itself (a loss averaged over size), is not a
        // function of the element: the kernel is not elementwise.
        foreach (string name in inputs)
            if (Regex.IsMatch(body, @"\b" + Regex.Escape(name) + @"\b")) { reason = kernel + ": reads " + name + " other than at idx"; return null; }
        if (Regex.IsMatch(body, @"\b" + Regex.Escape(output) + @"\b")) { reason = kernel + ": uses " + output + " other than storing at idx"; return null; }
        if (Regex.IsMatch(body, @"\b" + Regex.Escape(size) + @"\b")) { reason = kernel + ": reads the length " + size; return null; }
        foreach (string forbidden in new[] { "idx", "blockIdx", "threadIdx", "blockDim", "gridDim", "__syncthreads", "__shared__", "atomic" })
            if (Regex.IsMatch(body, @"\b" + forbidden + @"\b")) { reason = kernel + ": the body uses " + forbidden; return null; }
        if (Regex.IsMatch(body, @"\breturn\s*;")) { reason = kernel + ": exits without storing"; return null; }
        if (!body.Contains("return ", StringComparison.Ordinal)) { reason = kernel + ": never stores"; return null; }

        var parameters = new List<string>();
        for (int k = 0; k < inputs.Count; k++) parameters.Add("float aidn_v" + k.ToString(CultureInfo.InvariantCulture));
        foreach (string scalar in scalars) parameters.Add("float " + scalar);
        string wrapper = "extern \"C\" __device__ __noinline__ float aidn_" + kernel + "(" + string.Join(", ", parameters) + ")\n{" +
            body + "}\n";
        reason = "";
        return new CudaMathFunction(kernel, module, inputs.Count, scalars.ToArray(), wrapper, order.ToString());
    }
}
