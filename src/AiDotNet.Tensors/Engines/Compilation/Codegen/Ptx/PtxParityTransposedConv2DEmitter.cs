// Copyright (c) AiDotNet. All rights reserved.

using System;
using System.Globalization;
using System.Text;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ir;

namespace AiDotNet.Tensors.Engines.Compilation.Codegen.Ptx;

/// <summary>Emits deterministic 2x2 parity tiles for depthwise stride-2 transpose.</summary>
public sealed class PtxParityTransposedConv2DEmitter
{
    public CodegenParityTransposedConv2DPlan? Plan { get; private set; }
    public uint LaunchBlocks => checked((uint)(Plan?.Blocks ?? 0));
    public int LaunchBlockThreads => Plan is null
        ? 0
        : CodegenParityTransposedConv2DPlan.BlockThreads;

    public string Emit(CodegenKernelSpec spec, int computeMajor, int computeMinor)
    {
        if (!CodegenParityTransposedConv2DPlan.TryCreate(
                spec, out var possible, out string reason))
            throw new NotSupportedException(
                "This spec cannot use parity-transposed 2x2 tiles: " + reason);
        var plan = possible!;
        Plan = plan;

        int inputParam = spec.Inputs[plan.Input].ParameterIndex;
        int weightParam = spec.Inputs[plan.Weights].ParameterIndex;
        int outputParam = spec.Output.ParameterIndex;
        int inputRowBytes = checked(plan.InputWidth * sizeof(float));
        int outputRowBytes = checked(plan.OutputWidth * sizeof(float));
        int outputPlane = checked(plan.OutputHeight * plan.OutputWidth);

        var body = new StringBuilder(6000);
        void L(string line) => body.Append("    ").Append(line).Append('\n');

        L($"ld.param.u64 %rd0, [p{I(inputParam)}];");
        L($"ld.param.u64 %rd1, [p{I(weightParam)}];");
        L($"ld.param.u64 %rd2, [p{I(outputParam)}];");
        L("mov.u32 %r0, %tid.x;");
        L("mov.u32 %r1, %ctaid.x;");
        L($"mad.lo.u32 %r2, %r1, {I(CodegenParityTransposedConv2DPlan.BlockThreads)}, %r0;");
        L($"setp.ge.u32 %p0, %r2, {I(plan.InputElements)};");
        L("@%p0 bra END;");
        L($"rem.u32 %r3, %r2, {I(plan.InputWidth)};       // input column");
        L($"div.u32 %r4, %r2, {I(plan.InputWidth)};");
        L($"rem.u32 %r5, %r4, {I(plan.InputHeight)};      // input row");
        L($"div.u32 %r6, %r4, {I(plan.InputHeight)};      // flattened (n,c)");
        L($"rem.u32 %r7, %r6, {I(plan.Channels)};         // channel");
        L($"setp.lt.u32 %p1, %r3, {I(plan.InputWidth - 1)};");
        L($"setp.lt.u32 %p2, %r5, {I(plan.InputHeight - 1)};");
        L("and.pred %p3, %p1, %p2;");

        L("mul.wide.u32 %rd3, %r2, 4;");
        L("add.u64 %rd4, %rd0, %rd3;");
        L("ld.global.nc.f32 %f0, [%rd4];                  // input[r,c]");
        L("mov.f32 %f1, 0f00000000;");
        L("mov.f32 %f2, 0f00000000;");
        L("mov.f32 %f3, 0f00000000;");
        L("@%p1 ld.global.nc.f32 %f1, [%rd4+4];           // input[r,c+1]");
        L($"@%p2 ld.global.nc.f32 %f2, [%rd4+{I(inputRowBytes)}]; // input[r+1,c]");
        L($"@%p3 ld.global.nc.f32 %f3, [%rd4+{I(inputRowBytes + sizeof(float))}]; // input[r+1,c+1]");

        L("mov.f32 %f4, 0f00000000;                       // output[2r,2c]");
        L("mov.f32 %f5, 0f00000000;                       // output[2r,2c+1]");
        L("mov.f32 %f6, 0f00000000;                       // output[2r+1,2c]");
        L("mov.f32 %f7, 0f00000000;                       // output[2r+1,2c+1]");
        L($"mul.lo.u32 %r8, %r7, {I(CodegenParityTransposedConv2DPlan.KernelSize * CodegenParityTransposedConv2DPlan.KernelSize)};");
        L("mul.wide.u32 %rd5, %r8, 4;");
        L("add.u64 %rd6, %rd1, %rd5;");

        // Hoisting all nine loads measured 24 -> 30 registers with no claimable speedup
        // (23.2 -> 23.0 us), so retain the lower-pressure load/FMA schedule.
        EmitWeightFma(body, tap: 0, input: 3, accumulator: 7);
        EmitWeightFma(body, tap: 1, input: 2, accumulator: 6);
        EmitWeightFma(body, tap: 2, input: 2, accumulator: 7);
        EmitWeightFma(body, tap: 3, input: 1, accumulator: 5);
        EmitWeightFma(body, tap: 4, input: 0, accumulator: 4);
        EmitWeightFma(body, tap: 5, input: 0, accumulator: 5);
        EmitWeightFma(body, tap: 6, input: 1, accumulator: 7);
        EmitWeightFma(body, tap: 7, input: 0, accumulator: 6);
        EmitWeightFma(body, tap: 8, input: 0, accumulator: 7);

        L($"mul.lo.u32 %r9, %r6, {I(outputPlane)};");
        L($"mad.lo.u32 %r9, %r5, {I(2 * plan.OutputWidth)}, %r9;");
        L("mad.lo.u32 %r9, %r3, 2, %r9;");
        L("mul.wide.u32 %rd7, %r9, 4;");
        L("add.u64 %rd7, %rd2, %rd7;");
        L("st.global.f32 [%rd7], %f4;");
        L("@%p1 st.global.f32 [%rd7+4], %f5;");
        L($"@%p2 st.global.f32 [%rd7+{I(outputRowBytes)}], %f6;");
        L($"@%p3 st.global.f32 [%rd7+{I(outputRowBytes + sizeof(float))}], %f7;");
        L("END:");
        L("ret;");

        var text = new StringBuilder(7000);
        text.Append(".version ")
            .Append(PtxAffineEmitter.PtxIsaVersionFor(computeMajor, computeMinor)).Append('\n')
            .Append(".target sm_").Append(I(computeMajor)).Append(I(computeMinor)).Append('\n')
            .Append(".address_size 64\n\n")
            .Append("// generated by PtxParityTransposedConv2DEmitter\n")
            .Append("// one input coordinate owns its deterministic 2x2 output parity tile\n")
            .Append("// ").Append(spec.Describe().Replace("\n", "\n// ")).Append('\n')
            .Append(".visible .entry ").Append(spec.Name).Append("(\n");
        for (int i = 0; i < spec.ParameterCount; i++)
            text.Append("    .param .u64 p").Append(I(i))
                .Append(i == spec.ParameterCount - 1 ? "\n" : ",\n");
        text.Append(")\n{\n")
            .Append("    .reg .pred %p<5>;\n")
            .Append("    .reg .b32 %r<11>;\n")
            .Append("    .reg .b64 %rd<9>;\n")
            .Append("    .reg .f32 %f<9>;\n\n")
            .Append(body)
            .Append("}\n");
        return text.ToString();
    }

    private static void EmitWeightFma(
        StringBuilder body, int tap, int input, int accumulator)
    {
        void L(string line) => body.Append("    ").Append(line).Append('\n');
        L($"ld.global.nc.f32 %f8, [%rd6+{I(tap * sizeof(float))}];");
        L($"fma.rn.f32 %f{I(accumulator)}, %f{I(input)}, %f8, %f{I(accumulator)};");
    }

    private static string I(int value) => value.ToString(CultureInfo.InvariantCulture);
}
