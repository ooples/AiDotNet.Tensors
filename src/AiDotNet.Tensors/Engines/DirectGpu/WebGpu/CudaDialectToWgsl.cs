// Copyright (c) AiDotNet. All rights reserved.
using System.Collections.Generic;
using System.Text;
using System.Text.RegularExpressions;

namespace AiDotNet.Tensors.Engines.DirectGpu.WebGpu;

/// <summary>
/// Translates the portable CUDA dialect the shared kernel sources are written in to WGSL, one module per kernel, bound
/// as WebGpuBackend's batch dispatch binds it: pointer parameters as storage buffers in order, scalars as fields of a
/// uniform struct p after them. The dialect is the C subset both languages express the same way, and the translator
/// relies on it rather than parsing C:
/// <list type="bullet">
/// <item>kernels are <c>extern "C" __global__ __launch_bounds__(256) void name(...)</c> taking <c>const float*</c>,
/// <c>float*</c>, <c>const int*</c>, <c>int*</c> buffers and <c>int</c>, <c>unsigned int</c> or <c>float</c> scalars;
/// a kernel reads its index with <c>bx_gid()</c>;</item>
/// <item>helpers are <c>__device__ __forceinline__ T name(T a, ...)</c> over float, int, unsigned int and bool, and
/// never assign their parameters;</item>
/// <item>one declaration per statement (<c>float x = ...;</c> or <c>float x;</c>), loops <c>for (int k = ...; ...;
/// ...)</c>;</item>
/// <item>no ternary operator, no implicit int / float conversion of variables (casts are written <c>(float)x</c> or
/// <c>(float)(...)</c>), no chained assignment, no local arrays, structs or macros;</item>
/// <item>NaN / infinity tests and constants through the bit helpers (<c>__float_as_uint</c>,
/// <c>__uint_as_float</c>), never isnan or a constant division.</item>
/// </list>
/// </summary>
internal static class CudaDialectToWgsl
{
    private static readonly System.TimeSpan Timeout = System.TimeSpan.FromSeconds(5);

    private static readonly Regex KernelHeader = new(
        @"extern ""C"" __global__ __launch_bounds__\(256\) void (\w+)\(([^)]*)\)\s*\{", RegexOptions.None, Timeout);
    private static readonly Regex HelperHeader = new(
        @"__device__ __forceinline__ (float|int|unsigned int|bool|void) (\w+)\(([^)]*)\)\s*\{", RegexOptions.None, Timeout);

    /// <summary>Kernel name to its WGSL module.</summary>
    public static Dictionary<string, string> Translate(string source)
    {
        var helpers = new StringBuilder();
        var kernels = new List<(string Name, string Params, string Body)>();
        int at = 0;
        while (true)
        {
            var k = KernelHeader.Match(source, at);
            var h = HelperHeader.Match(source, at);
            if (!k.Success && !h.Success) break;
            if (h.Success && (!k.Success || h.Index < k.Index))
            {
                int end = MatchingBrace(source, h.Index + h.Length - 1);
                string body = source.Substring(h.Index + h.Length, end - h.Index - h.Length);
                helpers.Append("fn ").Append(h.Groups[2].Value).Append('(').Append(HelperParams(h.Groups[3].Value)).Append(')');
                if (h.Groups[1].Value != "void") helpers.Append(" -> ").Append(Type(h.Groups[1].Value));
                helpers.Append(" {").Append(Statements(body, null)).Append("}\n");
                at = end + 1;
            }
            else
            {
                int end = MatchingBrace(source, k.Index + k.Length - 1);
                kernels.Add((k.Groups[1].Value, k.Groups[2].Value, source.Substring(k.Index + k.Length, end - k.Index - k.Length)));
                at = end + 1;
            }
        }

        var result = new Dictionary<string, string>(System.StringComparer.Ordinal);
        foreach (var (name, parameters, body) in kernels)
        {
            var bindings = new StringBuilder();
            var fields = new List<string>();
            var scalars = new List<string>();
            int binding = 0;
            foreach (string raw in parameters.Split(','))
            {
                string p = Regex.Replace(raw.Replace("__restrict__", ""), @"\s+", " ", RegexOptions.None, Timeout).Trim();
                var pointer = Regex.Match(p, @"^(const )?(float|int)\* ?(\w+)$", RegexOptions.None, Timeout);
                if (pointer.Success)
                {
                    bindings.Append("@group(0) @binding(").Append(binding++).Append(") var<storage, ")
                        .Append(pointer.Groups[1].Success ? "read" : "read_write").Append("> ").Append(pointer.Groups[3].Value)
                        .Append(" : array<").Append(Type(pointer.Groups[2].Value)).Append(">;\n");
                    continue;
                }
                var scalar = Regex.Match(p, @"^(float|int|unsigned int) (\w+)$", RegexOptions.None, Timeout);
                if (!scalar.Success) throw new System.InvalidOperationException($"Untranslatable parameter '{p}' in {name}.");
                fields.Add(scalar.Groups[2].Value + ": " + Type(scalar.Groups[1].Value));
                scalars.Add(scalar.Groups[2].Value);
            }
            if (fields.Count == 0) fields.Add("unused_: i32");
            var module = new StringBuilder(bindings.ToString());
            module.Append("struct P { ").Append(string.Join(", ", fields)).Append(" };\n");
            module.Append("@group(0) @binding(").Append(binding).Append(") var<uniform> p : P;\n");
            module.Append(helpers);
            module.Append("@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) id: vec3<u32>) {")
                .Append(Statements(body, scalars)).Append("}\n");
            result[name] = module.ToString();
        }
        return result;
    }

    private static string HelperParams(string parameters)
    {
        var parts = new List<string>();
        foreach (string raw in parameters.Split(','))
        {
            string p = raw.Trim();
            if (p.Length == 0) continue;
            var m = Regex.Match(p, @"^(float|int|unsigned int|bool) (\w+)$", RegexOptions.None, Timeout);
            if (!m.Success) throw new System.InvalidOperationException($"Untranslatable helper parameter '{p}'.");
            parts.Add(m.Groups[2].Value + ": " + Type(m.Groups[1].Value));
        }
        return string.Join(", ", parts);
    }

    private static string Type(string c) => c switch
    {
        "float" => "f32",
        "int" => "i32",
        "unsigned int" => "u32",
        "bool" => "bool",
        _ => throw new System.InvalidOperationException($"Untranslatable type '{c}'."),
    };

    private static string Statements(string body, List<string>? scalars)
    {
        string s = body;
        s = Regex.Replace(s, @"(?<![\w.])(\d+\.\d*|\.\d+|\d+)([eE][+-]?\d+)?f\b", "$1$2", RegexOptions.None, Timeout);
        s = Casts(s);
        s = Regex.Replace(s, @"\bfor \((float|int|unsigned int) (\w+) =", m => $"for (var {m.Groups[2].Value}: {Type(m.Groups[1].Value)} =", RegexOptions.None, Timeout);
        s = Regex.Replace(s, @"(?m)^(\s*)(?:const )?(float|int|unsigned int|bool) (\w+)\s*(=|;)",
            m => $"{m.Groups[1].Value}var {m.Groups[3].Value}: {Type(m.Groups[2].Value)} {m.Groups[4].Value}", RegexOptions.None, Timeout);
        s = Regex.Replace(s, @"\b(exp|log|sqrt|floor|ceil|tanh|sin|cos|tan|atan|atan2|trunc|pow)f\(", "$1(", RegexOptions.None, Timeout);
        s = Regex.Replace(s, @"\bfabsf\(", "abs(", RegexOptions.None, Timeout);
        s = Regex.Replace(s, @"\brintf\(", "round(", RegexOptions.None, Timeout);   // both round half to even
        s = s.Replace("__float_as_uint(", "bitcast<u32>(").Replace("__float_as_int(", "bitcast<i32>(")
             .Replace("__int_as_float(", "bitcast<f32>(")
             // A function call, so a NaN / infinity bit pattern is a runtime value (WebGpuShaderPrelude defines it).
             .Replace("__uint_as_float(", "wgsl_f32_bits(")
             .Replace("bx_gid()", "i32(id.x)");
        if (scalars is not null)
            foreach (string name in scalars)
                s = Regex.Replace(s, @"(?<![\w.])" + name + @"\b", "p." + name, RegexOptions.None, Timeout);
        return s;
    }

    // (float)x, (float)a[i], (float)(expr) -> f32(...); likewise int and unsigned int.
    private static string Casts(string s)
    {
        var sb = new StringBuilder();
        int at = 0;
        var cast = new Regex(@"\((float|int|unsigned int)\)\s*", RegexOptions.None, Timeout);
        while (true)
        {
            var m = cast.Match(s, at);
            if (!m.Success) { sb.Append(s, at, s.Length - at); break; }
            sb.Append(s, at, m.Index - at);
            int start = m.Index + m.Length, end;
            if (start < s.Length && s[start] == '(') end = MatchingParen(s, start) + 1;
            else
            {
                end = start;
                while (end < s.Length && (char.IsLetterOrDigit(s[end]) || s[end] == '_' || s[end] == '.')) end++;
                while (end < s.Length && s[end] == '[') end = MatchingBracket(s, end) + 1;
            }
            string operand = Casts(s.Substring(start, end - start));
            if (operand.StartsWith("(", System.StringComparison.Ordinal) && operand.EndsWith(")", System.StringComparison.Ordinal))
                operand = operand.Substring(1, operand.Length - 2);
            sb.Append(Type(m.Groups[1].Value)).Append('(').Append(operand).Append(')');
            at = end;
        }
        return sb.ToString();
    }

    private static int MatchingBrace(string s, int open) => Matching(s, open, '{', '}');
    private static int MatchingParen(string s, int open) => Matching(s, open, '(', ')');
    private static int MatchingBracket(string s, int open) => Matching(s, open, '[', ']');

    private static int Matching(string s, int open, char o, char c)
    {
        int depth = 0;
        for (int k = open; k < s.Length; k++)
        {
            if (s[k] == o) depth++;
            else if (s[k] == c && --depth == 0) return k;
        }
        throw new System.InvalidOperationException($"Unbalanced '{o}' at {open}.");
    }
}
