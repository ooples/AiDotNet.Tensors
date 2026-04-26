// Copyright (c) AiDotNet. All rights reserved.
// Metal Shading Language source-emitting codegen target.

using System.Collections.Generic;
using System.Text;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ir;

namespace AiDotNet.Tensors.Engines.Compilation.Codegen.Msl;

/// <summary>
/// Emits Metal Shading Language (MSL) compute shader source for
/// pointwise fusions. Matches the <c>kernel void foo(..., uint gid
/// [[thread_position_in_grid]])</c> shape already used by the
/// hand-written Metal kernels in <see cref="Engines.DirectGpu.Metal"/>.
/// </summary>
public sealed class MslEmitter : IKernelEmitter
{
    /// <inheritdoc/>
    public CodegenTarget Target => CodegenTarget.Msl;

    private static readonly GpuTargetDialect Dialect = new(
        // MSL uses namespace metal:: overloads; unqualified names work
        // inside `using namespace metal;`.
        exp: "exp", log: "log", sqrt: "sqrt",
        sin: "sin", cos: "cos", tan: "tan", tanh: "tanh",
        abs: "abs", floor: "floor", ceil: "ceil", round: "round",
        max: "max", min: "min",
        floatZeroLiteral: "0.0f", floatOneLiteral: "1.0f");

    private static readonly HashSet<CodegenElementType> Supported = new()
    {
        CodegenElementType.Float32,
        // Metal supports doubles only on recent Apple silicon; gate
        // behind an explicit dialect variant when a target surfaces.
    };

    /// <inheritdoc/>
    public CodegenEmitResult Emit(CodegenGraph graph, CodegenElementType dtype)
    {
        var decline = GpuEmitterCommon.CheckSupport(graph, dtype, Supported);
        if (decline != null) return CodegenEmitResult.Decline(decline);

        string scalar = "float";
        int inputCount = graph.InputNodes.Count;
        int outputCount = graph.OutputNodes.Count;
        const string entryPoint = "pointwise_kernel";

        var sb = new StringBuilder();
        sb.AppendLine("#include <metal_stdlib>");
        sb.AppendLine("using namespace metal;");
        sb.AppendLine();
        sb.AppendLine($"kernel void {entryPoint}(");
        int bufSlot = 0;
        for (int i = 0; i < inputCount; i++)
            sb.AppendLine($"    device const {scalar}* in_{i} [[buffer({bufSlot++})]],");
        for (int i = 0; i < outputCount; i++)
            sb.AppendLine($"    device {scalar}* out_{i} [[buffer({bufSlot++})]],");
        sb.AppendLine($"    constant uint& n_elements [[buffer({bufSlot})]],");
        sb.AppendLine("    uint gid [[thread_position_in_grid]])");
        sb.AppendLine("{");
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
                    sb.AppendLine($"    {scalar} v{i} = {GpuEmitterCommon.FormatOpExpression(node, Dialect)};");
                    break;
            }
        }
        sb.AppendLine("}");

        var source = sb.ToString();
        return CodegenEmitResult.Succeeded(
            new GpuSourceKernel(graph, dtype, CodegenTarget.Msl, source, entryPoint),
            source);
    }
}
