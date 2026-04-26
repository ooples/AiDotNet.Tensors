// Copyright (c) AiDotNet. All rights reserved.
// GLSL compute (Vulkan) source-emitting codegen target.

using System.Collections.Generic;
using System.Text;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ir;

namespace AiDotNet.Tensors.Engines.Compilation.Codegen.Glsl;

/// <summary>
/// Emits GLSL 450 compute shaders for pointwise fusions. Output is
/// compatible with the Vulkan backend's shaderc-based runtime
/// compilation path — it mirrors the SSBO/push-constant convention
/// used by the hand-written shaders in
/// <see cref="Engines.DirectGpu.Vulkan"/>.
/// </summary>
public sealed class GlslEmitter : IKernelEmitter
{
    /// <inheritdoc/>
    public CodegenTarget Target => CodegenTarget.Glsl;

    private static readonly GpuTargetDialect Dialect = new(
        exp: "exp", log: "log", sqrt: "sqrt",
        sin: "sin", cos: "cos", tan: "tan", tanh: "tanh",
        abs: "abs", floor: "floor", ceil: "ceil", round: "round",
        max: "max", min: "min",
        floatZeroLiteral: "0.0", floatOneLiteral: "1.0");

    private static readonly HashSet<CodegenElementType> Supported = new()
    {
        // GLSL core supports double via GL_ARB_gpu_shader_fp64 — gate
        // behind an explicit Supported entry when needed.
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
        sb.AppendLine("#version 450");
        sb.AppendLine("layout(local_size_x = 256) in;");
        sb.AppendLine();
        int binding = 0;
        for (int i = 0; i < inputCount; i++)
            sb.AppendLine($"layout(set = 0, binding = {binding++}) readonly buffer InBuf{i} {{ float in_{i}[]; }};");
        for (int i = 0; i < outputCount; i++)
            sb.AppendLine($"layout(set = 0, binding = {binding++}) writeonly buffer OutBuf{i} {{ float out_{i}[]; }};");
        sb.AppendLine("layout(push_constant) uniform P { uint n_elements; } p;");
        sb.AppendLine();
        sb.AppendLine($"void {entryPoint}() {{");
        sb.AppendLine("    uint gid = gl_GlobalInvocationID.x;");
        sb.AppendLine("    if (gid >= p.n_elements) return;");

        int inputPort = 0, outputPort = 0;
        for (int i = 0; i < graph.Count; i++)
        {
            var node = graph[i];
            switch (node.Op)
            {
                case CodegenOpKind.LoadInput:
                    sb.AppendLine($"    float v{i} = in_{inputPort++}[gid];");
                    break;
                case CodegenOpKind.StoreOutput:
                    sb.AppendLine($"    out_{outputPort++}[gid] = v{node.Inputs[0]};");
                    break;
                default:
                    sb.AppendLine($"    float v{i} = {GpuEmitterCommon.FormatOpExpression(node, Dialect)};");
                    break;
            }
        }
        sb.AppendLine("}");

        var source = sb.ToString();
        return CodegenEmitResult.Succeeded(
            new GpuSourceKernel(graph, dtype, CodegenTarget.Glsl, source, entryPoint),
            source);
    }
}
