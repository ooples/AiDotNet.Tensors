// Copyright (c) CrowdTrainer LLC. All rights reserved.

using System;
using System.Collections.Generic;
using System.Globalization;
using System.Security.Cryptography;
using System.Text;
using AiDotNet.Tensors.Engines.Compilation.Codegen;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ir;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA;

/// <summary>
/// Executable CUDA C kernel representation emitted from a <see cref="CodegenGraph"/> for NVRTC compilation and
/// single-launch execution on <see cref="CudaBackend"/>.
/// </summary>
internal sealed class NvrtcCodegenKernel : CodegenKernel
{
    internal string EntryPoint { get; }
    internal string CudaSource { get; }
    internal int OutputElementCount { get; }
    internal bool UsesTrailingBroadcast { get; }
    internal int TrailingCols { get; }

    internal NvrtcCodegenKernel(
        CodegenElementType dtype,
        CodegenGraph graph,
        string entryPoint,
        string cudaSource,
        int outputElementCount,
        bool usesTrailingBroadcast,
        int trailingCols)
        : base(dtype, graph, CodegenTarget.DirectPtx)
    {
        EntryPoint = entryPoint;
        CudaSource = cudaSource;
        OutputElementCount = outputElementCount;
        UsesTrailingBroadcast = usesTrailingBroadcast;
        TrailingCols = trailingCols;
    }

    public override void Execute<T>(T[][] inputs, T[][] outputs)
        => throw new NotSupportedException(
            "NvrtcCodegenKernel executes strictly on resident CUDA device buffers via INativeGpuCodegenExecutor.");
}

/// <summary>
/// Emits NVRTC-ready CUDA C kernels from pointwise and trailing-broadcast <see cref="CodegenGraph"/> DAGs when
/// direct-PTX lowering declines (for example, chains mixing Swish, GELU, Sigmoid, Tanh, Exp, Log, and 1D bias broadcasts).
/// </summary>
internal sealed class NvrtcGraphEmitter : IKernelEmitter
{
    public CodegenTarget Target => CodegenTarget.DirectPtx;

    public CodegenEmitResult Emit(CodegenGraph graph, CodegenElementType dtype)
    {
        if (graph is null) throw new ArgumentNullException(nameof(graph));
        if (dtype != CodegenElementType.Float32)
            return CodegenEmitResult.Decline($"NvrtcGraphEmitter currently targets Float32; got {dtype}.");
        if (graph.InputNodes.Count == 0 || graph.OutputNodes.Count != 1)
            return CodegenEmitResult.Decline("NvrtcGraphEmitter requires >= 1 input and exactly 1 output.");

        var outNode = graph[graph.OutputNodes[0]];
        if (outNode.Dtype != CodegenElementType.Float32)
            return CodegenEmitResult.Decline($"Output node must be Float32; got {outNode.Dtype}.");

        long totalLong = outNode.ElementCount;
        if (totalLong <= 0 || totalLong > int.MaxValue)
            return CodegenEmitResult.Decline($"Invalid output element count {totalLong}.");
        int totalElements = (int)totalLong;
        int trailingCols = outNode.Shape.Length > 0 ? outNode.Shape[outNode.Shape.Length - 1] : 1;

        var inputIsTrailing1D = new bool[graph.InputNodes.Count];
        bool anyTrailingBroadcast = false;
        for (int p = 0; p < graph.InputNodes.Count; p++)
        {
            var inp = graph[graph.InputNodes[p]];
            if (inp.Dtype != CodegenElementType.Float32)
                return CodegenEmitResult.Decline($"Input {p} must be Float32; got {inp.Dtype}.");
            long inpElems = inp.ElementCount;
            if (inpElems == totalElements)
            {
                inputIsTrailing1D[p] = false;
            }
            else if (inp.Shape.Length == 1 && inp.Shape[0] == trailingCols && trailingCols > 0 && totalElements % trailingCols == 0)
            {
                inputIsTrailing1D[p] = true;
                anyTrailingBroadcast = true;
            }
            else
            {
                return CodegenEmitResult.Decline(
                    $"Input {p} shape (elements={inpElems}) is neither identical to output ({totalElements}) nor trailing 1D ({trailingCols}).");
            }
        }

        // Validate every interior op before emitting CUDA C.
        for (int n = 0; n < graph.Count; n++)
        {
            var node = graph[n];
            if (node.Op is CodegenOpKind.LoadInput or CodegenOpKind.StoreOutput)
                continue;
            if (node.Dtype != CodegenElementType.Float32)
                return CodegenEmitResult.Decline($"Interior node {n} ({node.Op}) must be Float32.");
            if (!IsSupportedPointwiseOp(node.Op))
                return CodegenEmitResult.Decline($"Operation {node.Op} is not supported by NvrtcGraphEmitter.");
        }

        string hashSuffix = ComputeGraphSignature(graph);
        string entryPoint = "aidn_nvrtc_fused_" + hashSuffix;

        var sb = new StringBuilder(2048);
        sb.AppendLine("__device__ __forceinline__ float aidn_gelu_f32(float x) {");
        sb.AppendLine("    return 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));");
        sb.AppendLine("}");
        sb.AppendLine("__device__ __forceinline__ float aidn_softplus_f32(float x) {");
        sb.AppendLine("    return x > 20.0f ? x : log1pf(expf(x));");
        sb.AppendLine("}");
        sb.Append("extern \"C\" __global__ void ").Append(entryPoint).Append('(');
        for (int p = 0; p < graph.InputNodes.Count; p++)
        {
            sb.Append("const float* __restrict__ in_").Append(p).Append(", ");
        }
        sb.Append("float* __restrict__ out_0, int n");
        if (anyTrailingBroadcast)
        {
            sb.Append(", int cols");
        }
        sb.AppendLine(") {");
        sb.AppendLine("    int tid = blockIdx.x * blockDim.x + threadIdx.x;");
        sb.AppendLine("    int stride = blockDim.x * gridDim.x;");

        if (!anyTrailingBroadcast)
        {
            // Emit float4 16-byte aligned vectorized path + scalar fallback.
            sb.Append("    bool aligned = (((unsigned long long)out_0) & 15ULL) == 0ULL");
            for (int p = 0; p < graph.InputNodes.Count; p++)
            {
                sb.Append(" && (((unsigned long long)in_").Append(p).Append(") & 15ULL) == 0ULL");
            }
            sb.AppendLine(";");
            sb.AppendLine("    if (aligned && n >= 4) {");
            sb.AppendLine("        int vecCount = n >> 2;");
            sb.AppendLine("        for (int vecIdx = tid; vecIdx < vecCount; vecIdx += stride) {");
            for (int p = 0; p < graph.InputNodes.Count; p++)
            {
                sb.Append("            float4 vin_").Append(p).Append(" = reinterpret_cast<const float4*>(in_").Append(p).AppendLine(")[vecIdx];");
            }
            sb.AppendLine("            float4 vout;");
            EmitLaneBody(sb, graph, inputIsTrailing1D, "x", "            ");
            EmitLaneBody(sb, graph, inputIsTrailing1D, "y", "            ");
            EmitLaneBody(sb, graph, inputIsTrailing1D, "z", "            ");
            EmitLaneBody(sb, graph, inputIsTrailing1D, "w", "            ");
            sb.AppendLine("            reinterpret_cast<float4*>(out_0)[vecIdx] = vout;");
            sb.AppendLine("        }");
            sb.AppendLine("        for (int idx = (vecCount << 2) + tid; idx < n; idx += stride) {");
            EmitScalarBody(sb, graph, inputIsTrailing1D, "            ");
            sb.AppendLine("        }");
            sb.AppendLine("    } else {");
            sb.AppendLine("        for (int idx = tid; idx < n; idx += stride) {");
            EmitScalarBody(sb, graph, inputIsTrailing1D, "            ");
            sb.AppendLine("        }");
            sb.AppendLine("    }");
        }
        else
        {
            sb.AppendLine("    for (int idx = tid; idx < n; idx += stride) {");
            sb.AppendLine("        int col = idx % cols;");
            EmitScalarBody(sb, graph, inputIsTrailing1D, "        ");
            sb.AppendLine("    }");
        }

        sb.AppendLine("}");

        string source = sb.ToString();
        var kernel = new NvrtcCodegenKernel(
            dtype,
            graph,
            entryPoint,
            source,
            totalElements,
            anyTrailingBroadcast,
            trailingCols);
        return CodegenEmitResult.Succeeded(kernel, source);
    }

    private static void EmitLaneBody(StringBuilder sb, CodegenGraph graph, bool[] inputIsTrailing1D, string lane, string indent)
    {
        var inputMap = BuildInputPortMap(graph);
        for (int n = 0; n < graph.Count; n++)
        {
            var node = graph[n];
            string varName = $"t{n}_{lane}";
            if (node.Op == CodegenOpKind.LoadInput)
            {
                int port = inputMap[n];
                _ = inputIsTrailing1D;
                sb.Append(indent).Append("float ").Append(varName).Append(" = vin_").Append(port).Append('.').Append(lane).AppendLine(";");
            }
            else if (node.Op == CodegenOpKind.StoreOutput)
            {
                int src = node.Inputs[0];
                sb.Append(indent).Append("vout.").Append(lane).Append(" = t").Append(src).Append('_').Append(lane).AppendLine(";");
            }
            else
            {
                string expr = EmitOpExpression(node, $"_{lane}");
                sb.Append(indent).Append("float ").Append(varName).Append(" = ").Append(expr).AppendLine(";");
            }
        }
    }

    private static void EmitScalarBody(StringBuilder sb, CodegenGraph graph, bool[] inputIsTrailing1D, string indent)
    {
        var inputMap = BuildInputPortMap(graph);
        for (int n = 0; n < graph.Count; n++)
        {
            var node = graph[n];
            string varName = $"t{n}";
            if (node.Op == CodegenOpKind.LoadInput)
            {
                int port = inputMap[n];
                string idxExpr = inputIsTrailing1D[port] ? "col" : "idx";
                sb.Append(indent).Append("float ").Append(varName).Append(" = in_").Append(port).Append('[').Append(idxExpr).AppendLine("];");
            }
            else if (node.Op == CodegenOpKind.StoreOutput)
            {
                int src = node.Inputs[0];
                sb.Append(indent).Append("out_0[idx] = t").Append(src).AppendLine(";");
            }
            else
            {
                string expr = EmitOpExpression(node, string.Empty);
                sb.Append(indent).Append("float ").Append(varName).Append(" = ").Append(expr).AppendLine(";");
            }
        }
    }

    private static Dictionary<int, int> BuildInputPortMap(CodegenGraph graph)
    {
        var map = new Dictionary<int, int>(graph.InputNodes.Count);
        for (int p = 0; p < graph.InputNodes.Count; p++)
        {
            map[graph.InputNodes[p]] = p;
        }
        return map;
    }

    private static bool IsSupportedPointwiseOp(CodegenOpKind op) => op switch
    {
        CodegenOpKind.Constant or
        CodegenOpKind.Add or CodegenOpKind.Sub or CodegenOpKind.Mul or CodegenOpKind.Div or
        CodegenOpKind.Max or CodegenOpKind.Min or CodegenOpKind.Pow or
        CodegenOpKind.Negate or CodegenOpKind.Reciprocal or CodegenOpKind.Sqrt or CodegenOpKind.Rsqrt or
        CodegenOpKind.Exp or CodegenOpKind.Log or CodegenOpKind.Sin or CodegenOpKind.Cos or CodegenOpKind.Tan or CodegenOpKind.Tanh or
        CodegenOpKind.Abs or CodegenOpKind.Floor or CodegenOpKind.Ceil or CodegenOpKind.Round or
        CodegenOpKind.ReLU or CodegenOpKind.Sigmoid or CodegenOpKind.GELU or CodegenOpKind.Swish or
        CodegenOpKind.LeakyReLU or CodegenOpKind.ELU or CodegenOpKind.SoftPlus or CodegenOpKind.HardSwish => true,
        _ => false
    };

    private static string EmitOpExpression(CodegenNode node, string suffix)
    {
        if (node.Op == CodegenOpKind.Constant)
        {
            float cVal = node.Attribute is not null ? Convert.ToSingle(node.Attribute, CultureInfo.InvariantCulture) : 0.0f;
            return FormatFloat(cVal);
        }

        string a = node.Inputs.Length > 0 ? $"t{node.Inputs[0]}{suffix}" : "0.0f";
        string b = node.Inputs.Length > 1 ? $"t{node.Inputs[1]}{suffix}" : "0.0f";

        return node.Op switch
        {
            CodegenOpKind.Add => $"({a} + {b})",
            CodegenOpKind.Sub => $"({a} - {b})",
            CodegenOpKind.Mul => $"({a} * {b})",
            CodegenOpKind.Div => $"({a} / {b})",
            CodegenOpKind.Max => $"fmaxf({a}, {b})",
            CodegenOpKind.Min => $"fminf({a}, {b})",
            CodegenOpKind.Pow => $"powf({a}, {b})",
            CodegenOpKind.Negate => $"(-{a})",
            CodegenOpKind.Reciprocal => $"(1.0f / {a})",
            CodegenOpKind.Sqrt => $"sqrtf({a})",
            CodegenOpKind.Rsqrt => $"rsqrtf({a})",
            CodegenOpKind.Exp => $"expf({a})",
            CodegenOpKind.Log => $"logf({a})",
            CodegenOpKind.Sin => $"sinf({a})",
            CodegenOpKind.Cos => $"cosf({a})",
            CodegenOpKind.Tan => $"tanf({a})",
            CodegenOpKind.Tanh => $"tanhf({a})",
            CodegenOpKind.Abs => $"fabsf({a})",
            CodegenOpKind.Floor => $"floorf({a})",
            CodegenOpKind.Ceil => $"ceilf({a})",
            CodegenOpKind.Round => $"nearbyintf({a})",
            CodegenOpKind.ReLU => $"fmaxf(0.0f, {a})",
            CodegenOpKind.Sigmoid => $"(1.0f / (1.0f + expf(-{a})))",
            CodegenOpKind.GELU => $"aidn_gelu_f32({a})",
            CodegenOpKind.Swish => $"({a} / (1.0f + expf(-{a})))",
            CodegenOpKind.LeakyReLU => EmitLeakyRelu(node, a),
            CodegenOpKind.ELU => EmitElu(node, a),
            CodegenOpKind.SoftPlus => $"aidn_softplus_f32({a})",
            CodegenOpKind.HardSwish => $"({a} * fminf(fmaxf({a} + 3.0f, 0.0f), 6.0f) * (1.0f / 6.0f))",
            _ => throw new InvalidOperationException($"Unsupported op {node.Op}")
        };
    }

    private static string EmitLeakyRelu(CodegenNode node, string a)
    {
        float slope = node.Attribute is not null ? Convert.ToSingle(node.Attribute, CultureInfo.InvariantCulture) : 0.01f;
        return $"({a} > 0.0f ? {a} : {FormatFloat(slope)} * {a})";
    }

    private static string EmitElu(CodegenNode node, string a)
    {
        float alpha = node.Attribute is not null ? Convert.ToSingle(node.Attribute, CultureInfo.InvariantCulture) : 1.0f;
        return $"({a} > 0.0f ? {a} : {FormatFloat(alpha)} * (expf({a}) - 1.0f))";
    }

    private static string FormatFloat(float value)
    {
        string text = value.ToString("R", CultureInfo.InvariantCulture);
        if (text.IndexOf('.') < 0 && text.IndexOf('e') < 0 && text.IndexOf('E') < 0)
            text += ".0";
        return text + "f";
    }

    private static string ComputeGraphSignature(CodegenGraph graph)
    {
        var desc = new StringBuilder(256);
        for (int i = 0; i < graph.Count; i++)
        {
            var n = graph[i];
            desc.Append((int)n.Op).Append(':');
            for (int k = 0; k < n.Inputs.Length; k++)
            {
                if (k > 0) desc.Append(',');
                desc.Append(n.Inputs[k]);
            }
            if (n.Attribute is not null)
                desc.Append('@').Append(Convert.ToString(n.Attribute, CultureInfo.InvariantCulture));
            desc.Append(';');
        }
        using var sha = SHA256.Create();
        byte[] hash = sha.ComputeHash(Encoding.UTF8.GetBytes(desc.ToString()));
        var hex = new StringBuilder(16);
        for (int i = 0; i < 8; i++) hex.Append(hash[i].ToString("x2", CultureInfo.InvariantCulture));
        return hex.ToString();
    }
}
