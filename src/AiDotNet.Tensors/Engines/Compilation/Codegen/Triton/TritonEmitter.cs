// Copyright (c) AiDotNet. All rights reserved.
// Triton (CUDA) source-emitting codegen target.

using System.Collections.Generic;
using System.Text;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ir;

namespace AiDotNet.Tensors.Engines.Compilation.Codegen.Triton;

/// <summary>
/// Emits OpenAI Triton kernels for pointwise fusions. Source follows
/// the conventional <c>@triton.jit</c> layout — pointer arguments,
/// <c>BLOCK_SIZE</c> constexpr, <c>tl.program_id(0)</c> element
/// stride with boundary masking.
/// </summary>
/// <remarks>
/// Dispatch wiring (in-process <c>libtriton</c> invocation) is
/// intentionally a follow-up PR. This emitter's responsibility is
/// the string — runtime integration lands once the Triton Python-
/// interop choice is agreed with the team.
/// </remarks>
public sealed class TritonEmitter : IKernelEmitter
{
    /// <inheritdoc/>
    public CodegenTarget Target => CodegenTarget.Triton;

    private static readonly GpuTargetDialect Dialect = new(
        exp: "tl.exp", log: "tl.log", sqrt: "tl.sqrt",
        sin: "tl.sin", cos: "tl.cos", tan: "tl.tan", tanh: "tl.tanh",
        abs: "tl.abs", floor: "tl.floor", ceil: "tl.ceil", round: "tl.round",
        max: "tl.maximum", min: "tl.minimum",
        floatZeroLiteral: "0.0", floatOneLiteral: "1.0");

    private static readonly HashSet<CodegenElementType> Supported = new()
    {
        CodegenElementType.Float32,
        CodegenElementType.Float64,
    };

    /// <inheritdoc/>
    public CodegenEmitResult Emit(CodegenGraph graph, CodegenElementType dtype)
    {
        var decline = GpuEmitterCommon.CheckSupport(graph, dtype, Supported);
        if (decline != null) return CodegenEmitResult.Decline(decline);

        int inputCount = graph.InputNodes.Count;
        int outputCount = graph.OutputNodes.Count;
        const string entryPoint = "pointwise_kernel";

        var sb = new StringBuilder();
        sb.AppendLine("import triton");
        sb.AppendLine("import triton.language as tl");
        sb.AppendLine();
        sb.AppendLine("@triton.jit");
        sb.Append("def ").Append(entryPoint).Append("(");
        for (int i = 0; i < inputCount; i++) sb.Append($"in_{i}_ptr, ");
        for (int i = 0; i < outputCount; i++) sb.Append($"out_{i}_ptr, ");
        sb.AppendLine("n_elements, BLOCK_SIZE: tl.constexpr):");
        sb.AppendLine("    pid = tl.program_id(axis=0)");
        sb.AppendLine("    block_start = pid * BLOCK_SIZE");
        sb.AppendLine("    offsets = block_start + tl.arange(0, BLOCK_SIZE)");
        sb.AppendLine("    mask = offsets < n_elements");

        int inputPort = 0, outputPort = 0;
        for (int i = 0; i < graph.Count; i++)
        {
            var node = graph[i];
            switch (node.Op)
            {
                case CodegenOpKind.LoadInput:
                    sb.Append($"    v{i} = tl.load(in_{inputPort++}_ptr + offsets, mask=mask, other=0.0)").AppendLine();
                    break;
                case CodegenOpKind.StoreOutput:
                    sb.Append($"    tl.store(out_{outputPort++}_ptr + offsets, v{node.Inputs[0]}, mask=mask)").AppendLine();
                    break;
                default:
                    sb.Append($"    v{i} = ").Append(GpuEmitterCommon.FormatOpExpression(node, Dialect)).AppendLine();
                    break;
            }
        }

        var source = sb.ToString();
        return CodegenEmitResult.Succeeded(
            new GpuSourceKernel(graph, dtype, CodegenTarget.Triton, source, entryPoint),
            source);
    }
}
