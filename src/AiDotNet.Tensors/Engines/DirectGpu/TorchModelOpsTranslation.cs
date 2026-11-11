// Copyright (c) CrowdTrainer LLC. All rights reserved.
using System;
using System.Collections.Generic;
using System.Text;
using System.Text.RegularExpressions;

namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>
/// <see cref="TorchModelOpsKernels.Source"/> for the backends that do not read the CUDA dialect: OpenCL C (one program)
/// and GLSL (one compute shader per kernel). Both are mechanical: the dialect is the C subset these languages share,
/// so only the kernel headers, the parameter bindings and the builtin names change.
/// </summary>
internal static class TorchModelOpsTranslation
{
    private static readonly TimeSpan RegexTimeout = TimeSpan.FromSeconds(30);

    private static readonly Regex KernelHeader = new(
        @"extern ""C"" __global__ __launch_bounds__\(256\) void (\w+)\(([^)]*)\)\s*\{", RegexOptions.None, RegexTimeout);

    // Read after GlslExpressions has renamed float2 / unsigned int.
    private static readonly Regex HelperHeader = new(
        @"__device__ __forceinline__ (vec2|float|int|uint|bool) (\w+)\(([^)]*)\)\s*\{", RegexOptions.None, RegexTimeout);

    private static readonly Lazy<string> OpenClSource = new(() => ToOpenCl(TorchModelOpsKernels.Source));

    private static readonly Lazy<Dictionary<string, string>> GlslShaders = new(() => ToGlsl(TorchModelOpsKernels.Source));

    /// <summary>The kernels as one OpenCL C program.</summary>
    public static string OpenCl => OpenClSource.Value;

    /// <summary>Kernel name to its GLSL compute shader: buffers bound 0..n-1 in parameter order, the scalars (int / float,
    /// one 32-bit word each) as a push-constant block in parameter order, 256 invocations per work group.</summary>
    public static IReadOnlyDictionary<string, string> Glsl => GlslShaders.Value;

    private static string ToOpenCl(string source)
    {
        string body = StripComments(source)
            .Replace("extern \"C\" __global__ __launch_bounds__(256) void", "__kernel void")
            .Replace("__device__ __forceinline__ ", "")
            .Replace("bx_gid()", "(int)get_global_id(0)")
            // A group barrier that also orders the group's global-memory writes (the CRF steps read the previous
            // step's values from device memory).
            .Replace("__syncthreads()", "barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE)")
            .Replace("threadIdx.x", "(int)get_local_id(0)")
            .Replace("blockIdx.x", "(int)get_group_id(0)")
            .Replace("blockDim.x", "(int)get_local_size(0)")
            .Replace("__float_as_uint(", "as_uint(")
            .Replace("__uint_as_float(", "as_float(")
            .Replace("make_float2(", "(float2)(");
        body = Regex.Replace(body, @"(?<![\w*])(const )?(float|int)\* ", "__global $1$2* ", RegexOptions.None, RegexTimeout);
        return Regex.Replace(body, @"\b(sqrt|fmax|fmin|fabs|exp|log|rint|floor)f\(", "$1(", RegexOptions.None, RegexTimeout);
    }

    private static Dictionary<string, string> ToGlsl(string source)
    {
        string text = GlslExpressions(StripComments(source));
        var helpers = new StringBuilder();
        var kernels = new List<(string Name, string Params, string Body)>();
        int at = 0;
        while (true)
        {
            var k = KernelHeader.Match(text, at);
            var h = HelperHeader.Match(text, at);
            if (!k.Success && !h.Success) break;
            if (h.Success && (!k.Success || h.Index < k.Index))
            {
                int end = MatchingBrace(text, h.Index + h.Length - 1);
                helpers.Append(GlslType(h.Groups[1].Value)).Append(' ').Append(h.Groups[2].Value).Append('(')
                    .Append(GlslHelperParameters(h.Groups[3].Value)).Append(") {")
                    .Append(text, h.Index + h.Length, end - h.Index - h.Length).Append("}\n");
                at = end + 1;
            }
            else
            {
                int end = MatchingBrace(text, k.Index + k.Length - 1);
                kernels.Add((k.Groups[1].Value, k.Groups[2].Value, text.Substring(k.Index + k.Length, end - k.Index - k.Length)));
                at = end + 1;
            }
        }

        var result = new Dictionary<string, string>(StringComparer.Ordinal);
        foreach (var (name, parameters, body) in kernels)
        {
            var shader = new StringBuilder("#version 450\nlayout(local_size_x = 256) in;\n");
            var push = new StringBuilder();
            int binding = 0;
            foreach (string raw in parameters.Split(','))
            {
                string p = Regex.Replace(raw.Trim(), @"\s+", " ", RegexOptions.None, RegexTimeout);
                var pointer = Regex.Match(p, @"^(const )?(float|int)\* ?(\w+)$", RegexOptions.None, RegexTimeout);
                if (pointer.Success)
                {
                    // A buffer a group-synchronized kernel writes is coherent, so a lane reads its group's writes after
                    // the barrier rather than a cached value.
                    string access = pointer.Groups[1].Success ? "readonly " : body.Contains("barrier()") ? "coherent " : "";
                    shader.Append("layout(set = 0, binding = ").Append(binding).Append(") ")
                        .Append(access).Append("buffer MoBuf").Append(binding).Append(" { ")
                        .Append(pointer.Groups[2].Value).Append(' ').Append(pointer.Groups[3].Value).Append("[]; };\n");
                    binding++;
                    continue;
                }
                var scalar = Regex.Match(p, @"^(float|int) (\w+)$", RegexOptions.None, RegexTimeout);
                if (!scalar.Success) throw new InvalidOperationException($"Untranslatable parameter '{p}' in {name}.");
                push.Append(scalar.Groups[1].Value).Append(' ').Append(scalar.Groups[2].Value).Append("; ");
            }
            if (push.Length > 0) shader.Append("layout(push_constant) uniform MoParams { ").Append(push).Append("};\n");
            shader.Append(helpers).Append("void main() {").Append(body).Append("}\n");
            result[name] = shader.ToString();
        }
        return result;
    }

    private static string GlslExpressions(string s)
    {
        s = s.Replace("bx_gid()", "int(gl_GlobalInvocationID.x)")
            // barrier() alone does not order buffer memory: make the group's buffer writes visible first.
            .Replace("__syncthreads()", "memoryBarrierBuffer(); barrier()")
            .Replace("threadIdx.x", "int(gl_LocalInvocationID.x)")
            .Replace("blockIdx.x", "int(gl_WorkGroupID.x)")
            .Replace("blockDim.x", "int(gl_WorkGroupSize.x)")
            .Replace("__float_as_uint(", "floatBitsToUint(")
            .Replace("__uint_as_float(", "uintBitsToFloat(")
            .Replace("make_float2(", "vec2(")
            .Replace("float2 ", "vec2 ")
            .Replace("unsigned int", "uint");
        s = Regex.Replace(s, @"\b(sqrt|exp|log|floor)f\(", "$1(", RegexOptions.None, RegexTimeout);
        s = Regex.Replace(s, @"\bfabsf\(", "abs(", RegexOptions.None, RegexTimeout);
        s = Regex.Replace(s, @"\bfmaxf\(", "max(", RegexOptions.None, RegexTimeout);
        s = Regex.Replace(s, @"\bfminf\(", "min(", RegexOptions.None, RegexTimeout);
        s = Regex.Replace(s, @"\brintf\(", "roundEven(", RegexOptions.None, RegexTimeout);
        // Every float local is precise: a Vulkan driver may otherwise reassociate float arithmetic, which folds the
        // error terms of the kernels' exact two-sums and bit splits ((hi - (s - bb)) to 0) and loses their precision.
        s = Regex.Replace(s, @"(?m)(^|[;{}])(\s*)(float|vec2) (\w+)(\s*)(=|;)", "$1$2precise $3 $4$5$6", RegexOptions.None, RegexTimeout);
        // A C cast of a parenthesized expression, (uint)(k + 127), is the constructor uint(k + 127) in GLSL; a cast of a
        // name, (int)j, is int(j).
        s = Regex.Replace(s, @"\((float|int|uint)\)\s*\(", "$1(", RegexOptions.None, RegexTimeout);
        return Regex.Replace(s, @"\((float|int|uint)\)\s*(\w+)", "$1($2)", RegexOptions.None, RegexTimeout);
    }

    private static string GlslType(string c) => c switch
    {
        "float2" or "vec2" => "vec2",
        "unsigned int" or "uint" => "uint",
        _ => c,
    };

    private static string GlslHelperParameters(string parameters)
    {
        var parts = new List<string>();
        foreach (string raw in parameters.Split(','))
        {
            string p = raw.Trim();
            if (p.Length > 0) parts.Add(p);
        }
        return string.Join(", ", parts);
    }

    private static string StripComments(string source)
        => Regex.Replace(source, @"//[^\n]*", "", RegexOptions.None, RegexTimeout);

    private static int MatchingBrace(string s, int open)
    {
        int depth = 0;
        for (int k = open; k < s.Length; k++)
        {
            if (s[k] == '{') depth++;
            else if (s[k] == '}' && --depth == 0) return k;
        }
        throw new InvalidOperationException($"Unbalanced '{{' at {open}.");
    }
}
