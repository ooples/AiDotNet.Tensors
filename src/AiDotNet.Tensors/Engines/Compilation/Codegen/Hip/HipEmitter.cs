// Copyright (c) AiDotNet. All rights reserved.
// HIP (AMD ROCm) source-emitting codegen target. Emits C kernels
// that target hipcc / hiprtc.

using System.Collections.Generic;
using System.Text;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ir;

namespace AiDotNet.Tensors.Engines.Compilation.Codegen.Hip;

/// <summary>
/// Emits HIP kernel source for pointwise fusions. Source is
/// C-like — <c>extern "C" __global__ void kernel(...)</c> with
/// <c>blockIdx.x * blockDim.x + threadIdx.x</c> global-thread-index
/// addressing. Matches the kernel launch convention already used by
/// <c>HipBackend</c> for the hand-written kernels the codegen path
/// will eventually compose with.
/// </summary>
public sealed class HipEmitter : IKernelEmitter
{
    /// <inheritdoc/>
    public CodegenTarget Target => CodegenTarget.Hip;

    private static readonly GpuTargetDialect DialectFloat = new(
        exp: "expf", log: "logf", sqrt: "sqrtf",
        sin: "sinf", cos: "cosf", tan: "tanf", tanh: "tanhf",
        abs: "fabsf", floor: "floorf", ceil: "ceilf", round: "roundf",
        max: "fmaxf", min: "fminf",
        floatZeroLiteral: "0.0f", floatOneLiteral: "1.0f");

    private static readonly GpuTargetDialect DialectDouble = new(
        exp: "exp", log: "log", sqrt: "sqrt",
        sin: "sin", cos: "cos", tan: "tan", tanh: "tanh",
        abs: "fabs", floor: "floor", ceil: "ceil", round: "round",
        max: "fmax", min: "fmin",
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

        var dialect = dtype == CodegenElementType.Float32 ? DialectFloat : DialectDouble;
        string scalar = dtype == CodegenElementType.Float32 ? "float" : "double";
        int inputCount = graph.InputNodes.Count;
        int outputCount = graph.OutputNodes.Count;
        const string entryPoint = "pointwise_kernel";

        var sb = new StringBuilder();
        sb.AppendLine("#include <hip/hip_runtime.h>");
        sb.AppendLine("#include <math.h>");
        sb.AppendLine();
        sb.Append($"extern \"C\" __global__ void {entryPoint}(");
        for (int i = 0; i < inputCount; i++) sb.Append($"const {scalar}* __restrict__ in_{i}, ");
        for (int i = 0; i < outputCount; i++) sb.Append($"{scalar}* __restrict__ out_{i}, ");
        sb.AppendLine("int n_elements) {");
        sb.AppendLine("    int gid = blockIdx.x * blockDim.x + threadIdx.x;");
        sb.AppendLine("    if (gid >= n_elements) return;");

        int inputPort = 0, outputPort = 0;
        for (int i = 0; i < graph.Count; i++)
        {
            var node = graph[i];
            switch (node.Op)
            {
                case CodegenOpKind.LoadInput:
                    sb.AppendLine($"    {scalar} v{i} = in_{inputPort++}[gid];");
                    break;
                case CodegenOpKind.StoreOutput:
                    sb.AppendLine($"    out_{outputPort++}[gid] = v{node.Inputs[0]};");
                    break;
                default:
                    sb.AppendLine($"    {scalar} v{i} = {GpuEmitterCommon.FormatOpExpression(node, dialect)};");
                    break;
            }
        }
        sb.AppendLine("}");

        var source = sb.ToString();
        return CodegenEmitResult.Succeeded(
            new GpuSourceKernel(graph, dtype, CodegenTarget.Hip, source, entryPoint),
            source);
    }
}
