// Copyright (c) AiDotNet. All rights reserved.

using System.Collections.Generic;
using System.Text;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ir;

namespace AiDotNet.Tensors.Engines.Compilation.Codegen.OpenCl;

/// <summary>
/// Emits one OpenCL C device kernel for an entire supported pointwise graph. The selected
/// OpenCL driver lowers this source to the native instruction set of the physical device
/// (AMDGPU on AMD, Intel GPU ISA on Intel, or NVIDIA machine code on NVIDIA OpenCL).
/// </summary>
public sealed class OpenClEmitter : IKernelEmitter
{
    private static readonly GpuTargetDialect FloatDialect = new(
        exp: "exp", log: "log", sqrt: "sqrt",
        sin: "sin", cos: "cos", tan: "tan", tanh: "tanh",
        abs: "fabs", floor: "floor", ceil: "ceil", round: "rint",
        max: "fmax", min: "fmin",
        floatZeroLiteral: "0.0f", floatOneLiteral: "1.0f");

    private static readonly GpuTargetDialect DoubleDialect = new(
        exp: "exp", log: "log", sqrt: "sqrt",
        sin: "sin", cos: "cos", tan: "tan", tanh: "tanh",
        abs: "fabs", floor: "floor", ceil: "ceil", round: "rint",
        max: "fmax", min: "fmin",
        floatZeroLiteral: "0.0", floatOneLiteral: "1.0");

    private static readonly HashSet<CodegenElementType> Supported = new()
    {
        CodegenElementType.Float32,
        CodegenElementType.Float64
    };

    /// <inheritdoc />
    public CodegenTarget Target => CodegenTarget.OpenCl;

    /// <inheritdoc />
    public CodegenEmitResult Emit(CodegenGraph graph, CodegenElementType dtype)
    {
        string? decline = GpuEmitterCommon.CheckSupport(graph, dtype, Supported);
        if (decline is not null) return CodegenEmitResult.Decline(decline);

        GpuTargetDialect dialect = dtype == CodegenElementType.Float32
            ? FloatDialect
            : DoubleDialect;
        string scalar = dtype == CodegenElementType.Float32 ? "float" : "double";
        const string entryPoint = "pointwise_kernel";

        var source = new StringBuilder();
        if (dtype == CodegenElementType.Float64)
            source.AppendLine("#pragma OPENCL EXTENSION cl_khr_fp64 : enable");
        source.Append($"__kernel void {entryPoint}(");
        for (int i = 0; i < graph.InputNodes.Count; i++)
            source.Append($"__global const {scalar}* in_{i}, ");
        for (int i = 0; i < graph.OutputNodes.Count; i++)
            source.Append($"__global {scalar}* out_{i}, ");
        source.AppendLine("const int n_elements)");
        source.AppendLine("{");
        source.AppendLine("    const int gid = (int)get_global_id(0);");
        source.AppendLine("    if (gid >= n_elements) return;");

        int inputPort = 0;
        int outputPort = 0;
        for (int i = 0; i < graph.Count; i++)
        {
            CodegenNode node = graph[i];
            switch (node.Op)
            {
                case CodegenOpKind.LoadInput:
                    source.AppendLine($"    {scalar} v{i} = in_{inputPort++}[gid];");
                    break;
                case CodegenOpKind.StoreOutput:
                    source.AppendLine($"    out_{outputPort++}[gid] = v{node.Inputs[0]};");
                    break;
                default:
                    source.AppendLine(
                        $"    {scalar} v{i} = {GpuEmitterCommon.FormatOpExpression(node, dialect)};");
                    break;
            }
        }
        source.AppendLine("}");

        string text = source.ToString();
        return CodegenEmitResult.Succeeded(
            new GpuSourceKernel(graph, dtype, CodegenTarget.OpenCl, text, entryPoint),
            text);
    }
}
