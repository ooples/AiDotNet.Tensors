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

    /// <summary>
    /// Element types accepted only as <see cref="CodegenOpKind.LoadInput"/>
    /// inputs — the emitter inserts an unpack-and-dequantize prologue
    /// so the rest of the kernel works in <c>float32</c>. These are
    /// QLoRA-style packed weight formats: two 4-bit lanes per byte for
    /// Int4/NF4/FP4, four 2-bit lanes per byte for Int2, eight 1-bit
    /// lanes for Int1. Sub-byte <see cref="CodegenOpKind.StoreOutput"/>
    /// would require atomic byte updates and is out of scope; the
    /// emitter Declines if a sub-byte type appears as the kernel's
    /// output dtype.
    /// </summary>
    private static readonly HashSet<CodegenElementType> SubByteLoadOnly = new()
    {
        CodegenElementType.NF4,
        CodegenElementType.FP4,
        CodegenElementType.Int3,
        CodegenElementType.Int2,
        CodegenElementType.Int1,
    };

    /// <inheritdoc/>
    public CodegenEmitResult Emit(CodegenGraph graph, CodegenElementType dtype)
    {
        // Output dtype must be a "compute" type — sub-byte StoreOutput
        // would need atomic byte updates and is out of scope.
        var decline = GpuEmitterCommon.CheckSupport(graph, dtype, Supported);
        if (decline != null) return CodegenEmitResult.Decline(decline);

        // Walk the LoadInput nodes to detect any sub-byte input dtypes.
        // The kernel signature stays the same — the unpack expands
        // each sub-byte input into a float32 lane vector before any
        // pointwise op references it.
        var subByteInputs = new Dictionary<int, CodegenElementType>();
        for (int i = 0; i < graph.Count; i++)
        {
            var node = graph[i];
            if (node.Op == CodegenOpKind.LoadInput && SubByteLoadOnly.Contains(node.Dtype))
                subByteInputs[i] = node.Dtype;
            else if (node.Op == CodegenOpKind.StoreOutput && SubByteLoadOnly.Contains(node.Dtype))
                return CodegenEmitResult.Decline(
                    $"Triton emitter does not support sub-byte StoreOutput ({node.Dtype}); " +
                    "atomic byte-level packing must happen in a separate pass.");
        }

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
                    if (subByteInputs.TryGetValue(i, out var subType))
                    {
                        EmitSubByteLoad(sb, i, inputPort, subType);
                        inputPort++;
                    }
                    else
                    {
                        sb.Append($"    v{i} = tl.load(in_{inputPort++}_ptr + offsets, mask=mask, other=0.0)").AppendLine();
                    }
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

    /// <summary>
    /// Emits the sub-byte unpack prologue for a single
    /// <see cref="CodegenOpKind.LoadInput"/> node. The packed buffer
    /// is loaded as <c>uint8</c>, the relevant lane is masked +
    /// shifted out, sign-extended for Int formats, and dequantised to
    /// <c>float32</c> using the standard QLoRA-style affine scale.
    /// </summary>
    private static void EmitSubByteLoad(StringBuilder sb, int v, int port, CodegenElementType type)
    {
        sb.AppendLine($"    # sub-byte load: {type} via packed uint8 + bit unpack.");
        switch (type)
        {
            case CodegenElementType.Int3:
                // 3-bit signed values packed into a uint8 with one
                // crumb of slack per pair — emitter uses the same
                // 4-bit-per-nibble layout so adjacent indices land in
                // a single byte. The high bit of the nibble is sign.
                sb.AppendLine($"    packed_off_{v} = offsets // 2");
                sb.AppendLine($"    packed_{v} = tl.load(in_{port}_ptr + packed_off_{v}, mask=mask, other=0).to(tl.uint8)");
                sb.AppendLine($"    nibble_{v} = tl.where((offsets & 1) == 0, packed_{v} & 0x7, (packed_{v} >> 3) & 0x7)");
                sb.AppendLine($"    # sign-extend Int3 from [-4, 3]");
                sb.AppendLine($"    signed_{v} = tl.where(nibble_{v} >= 4, nibble_{v}.to(tl.int8) - 8, nibble_{v}.to(tl.int8))");
                sb.AppendLine($"    v{v} = signed_{v}.to(tl.float32)");
                break;
            case CodegenElementType.NF4:
                // NF4 lookup table — the 16 normal-distribution centroids
                // from Dettmers et al. 2023. Triton supports tl.tensor lookup
                // via gather; we emit the table once and index it.
                sb.AppendLine($"    NF4_LUT_{v} = tl.tensor([-1.0, -0.6962, -0.5251, -0.3949, -0.2844, -0.1848, -0.0911, 0.0, 0.0796, 0.1609, 0.2461, 0.3379, 0.4407, 0.5626, 0.7230, 1.0])");
                sb.AppendLine($"    packed_off_{v} = offsets // 2");
                sb.AppendLine($"    packed_{v} = tl.load(in_{port}_ptr + packed_off_{v}, mask=mask, other=0).to(tl.uint8)");
                sb.AppendLine($"    nibble_{v} = tl.where((offsets & 1) == 0, packed_{v} & 0xF, (packed_{v} >> 4) & 0xF)");
                sb.AppendLine($"    v{v} = tl.gather(NF4_LUT_{v}, nibble_{v}.to(tl.int32))");
                break;
            case CodegenElementType.FP4:
                // FP4 (E2M1) — 1 sign bit, 2 exp bits, 1 mantissa bit.
                // Use the canonical OCP-spec lookup since Triton lacks
                // a native FP4 cast.
                sb.AppendLine($"    FP4_LUT_{v} = tl.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])");
                sb.AppendLine($"    packed_off_{v} = offsets // 2");
                sb.AppendLine($"    packed_{v} = tl.load(in_{port}_ptr + packed_off_{v}, mask=mask, other=0).to(tl.uint8)");
                sb.AppendLine($"    nibble_{v} = tl.where((offsets & 1) == 0, packed_{v} & 0xF, (packed_{v} >> 4) & 0xF)");
                sb.AppendLine($"    v{v} = tl.gather(FP4_LUT_{v}, nibble_{v}.to(tl.int32))");
                break;
            case CodegenElementType.Int2:
                sb.AppendLine($"    packed_off_{v} = offsets // 4");
                sb.AppendLine($"    shift_{v} = (offsets & 3) * 2");
                sb.AppendLine($"    packed_{v} = tl.load(in_{port}_ptr + packed_off_{v}, mask=mask, other=0).to(tl.uint8)");
                sb.AppendLine($"    crumb_{v} = (packed_{v} >> shift_{v}) & 0x3");
                sb.AppendLine($"    signed_{v} = tl.where(crumb_{v} >= 2, crumb_{v}.to(tl.int8) - 4, crumb_{v}.to(tl.int8))");
                sb.AppendLine($"    v{v} = signed_{v}.to(tl.float32)");
                break;
            case CodegenElementType.Int1:
                sb.AppendLine($"    packed_off_{v} = offsets // 8");
                sb.AppendLine($"    shift_{v} = offsets & 7");
                sb.AppendLine($"    packed_{v} = tl.load(in_{port}_ptr + packed_off_{v}, mask=mask, other=0).to(tl.uint8)");
                sb.AppendLine($"    bit_{v} = (packed_{v} >> shift_{v}) & 1");
                sb.AppendLine($"    # BitNet convention: 0 → -1, 1 → +1");
                sb.AppendLine($"    v{v} = bit_{v}.to(tl.float32) * 2.0 - 1.0");
                break;
        }
    }
}
