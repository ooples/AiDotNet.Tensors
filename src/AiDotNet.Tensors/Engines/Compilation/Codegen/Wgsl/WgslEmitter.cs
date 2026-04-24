// Copyright (c) AiDotNet. All rights reserved.
// WebGPU WGSL source-emitting codegen target.

using System.Collections.Generic;
using System.Text;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ir;

namespace AiDotNet.Tensors.Engines.Compilation.Codegen.Wgsl;

/// <summary>
/// Emits WebGPU compute shaders in WGSL. Follows the binding /
/// workgroup convention of the hand-written WGSL shaders already in
/// <see cref="Engines.DirectGpu.WebGpu"/>: <c>@group(0)</c>
/// storage buffers for inputs/outputs, a uniform block for
/// <c>n_elements</c>, <c>@workgroup_size(256)</c> with
/// <c>@builtin(global_invocation_id)</c> addressing.
/// </summary>
public sealed class WgslEmitter : IKernelEmitter
{
    /// <inheritdoc/>
    public CodegenTarget Target => CodegenTarget.Wgsl;

    private static readonly GpuTargetDialect Dialect = new(
        // WGSL built-in names are unsuffixed — they're typed via the
        // operand, so there's no expf/exp distinction.
        exp: "exp", log: "log", sqrt: "sqrt",
        sin: "sin", cos: "cos", tan: "tan", tanh: "tanh",
        abs: "abs", floor: "floor", ceil: "ceil", round: "round",
        max: "max", min: "min",
        floatZeroLiteral: "0.0", floatOneLiteral: "1.0");

    private static readonly HashSet<CodegenElementType> Supported = new()
    {
        // WGSL core spec: f32 mandatory, f64 not in core. Emit float32 only.
        CodegenElementType.Float32,
    };

    /// <inheritdoc/>
    public CodegenEmitResult Emit(CodegenGraph graph, CodegenElementType dtype)
    {
        var decline = GpuEmitterCommon.CheckSupport(graph, dtype, Supported);
        if (decline != null) return CodegenEmitResult.Decline(decline);

        int inputCount = graph.InputNodes.Count;
        int outputCount = graph.OutputNodes.Count;
        const string entryPoint = "main";

        var sb = new StringBuilder();
        int binding = 0;
        for (int i = 0; i < inputCount; i++)
        {
            sb.AppendLine($"@group(0) @binding({binding++}) var<storage, read>       in_{i} : array<f32>;");
        }
        for (int i = 0; i < outputCount; i++)
        {
            sb.AppendLine($"@group(0) @binding({binding++}) var<storage, read_write> out_{i} : array<f32>;");
        }
        sb.AppendLine($"struct P {{ n_elements : u32 }};");
        sb.AppendLine($"@group(0) @binding({binding}) var<uniform> p : P;");
        sb.AppendLine();
        sb.AppendLine($"@compute @workgroup_size(256) fn {entryPoint}(@builtin(global_invocation_id) id : vec3<u32>) {{");
        sb.AppendLine("    let gid : u32 = id.x;");
        sb.AppendLine("    if (gid >= p.n_elements) { return; }");

        int inputPort = 0, outputPort = 0;
        for (int i = 0; i < graph.Count; i++)
        {
            var node = graph[i];
            switch (node.Op)
            {
                case CodegenOpKind.LoadInput:
                    sb.AppendLine($"    let v{i} : f32 = in_{inputPort++}[gid];");
                    break;
                case CodegenOpKind.StoreOutput:
                    sb.AppendLine($"    out_{outputPort++}[gid] = v{node.Inputs[0]};");
                    break;
                default:
                    sb.AppendLine($"    let v{i} : f32 = {GpuEmitterCommon.FormatOpExpression(node, Dialect)};");
                    break;
            }
        }
        sb.AppendLine("}");

        var source = sb.ToString();
        return CodegenEmitResult.Succeeded(
            new GpuSourceKernel(graph, dtype, CodegenTarget.Wgsl, source, entryPoint),
            source);
    }
}
