// Copyright (c) AiDotNet. All rights reserved.

using System;
using System.Globalization;
using System.Text;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ir;

namespace AiDotNet.Tensors.Engines.Compilation.Codegen.Ptx;

/// <summary>
/// Emits a cooperative, deterministic FP32 depthwise 3x3 weight-gradient reduction.
/// </summary>
public sealed class PtxDepthwiseConv2DWeightGradientEmitter
{
    public CodegenDepthwiseConv2DWeightGradientPlan? Plan { get; private set; }
    public uint LaunchBlocks => checked((uint)(Plan?.Blocks ?? 0));
    public int LaunchBlockThreads => Plan is null
        ? 0
        : CodegenDepthwiseConv2DWeightGradientPlan.BlockThreads;
    public int SharedMemoryBytes => Plan?.SharedMemoryBytes ?? 0;

    public string Emit(CodegenKernelSpec spec, int computeMajor, int computeMinor)
    {
        if (computeMajor < 7)
            throw new NotSupportedException(
                "The cooperative depthwise weight gradient requires synchronized warp shuffles on sm_70+.");
        if (!CodegenDepthwiseConv2DWeightGradientPlan.TryCreate(
                spec, out var possible, out string reason))
            throw new NotSupportedException(
                "This spec cannot use the cooperative depthwise weight gradient: " + reason);
        var plan = possible!;
        Plan = plan;

        int gradParam = spec.Inputs[plan.GradOutputInput].ParameterIndex;
        int dataParam = spec.Inputs[plan.DataInput].ParameterIndex;
        int outputParam = spec.Output.ParameterIndex;
        int spatial = checked(plan.Height * plan.Width);
        int channelSpatial = checked(plan.Channels * spatial);
        int warpSize = CodegenDepthwiseConv2DWeightGradientPlan.WarpSize;
        int warpShift = PowerOfTwoShift(warpSize);

        var body = new StringBuilder(16384);
        void L(string line) => body.Append("    ").Append(line).Append('\n');

        L($"ld.param.u64 %rd0, [p{I(gradParam)}];");
        L($"ld.param.u64 %rd1, [p{I(dataParam)}];");
        L($"ld.param.u64 %rd2, [p{I(outputParam)}];");
        L("mov.u64 %rd3, warp_sums;");
        L("mov.u32 %r0, %tid.x;");
        L("mov.u32 %r1, %ctaid.x;");
        L($"div.u32 %r2, %r1, {I(CodegenDepthwiseConv2DWeightGradientPlan.KernelSize)}; // channel");
        L($"rem.u32 %r3, %r1, {I(CodegenDepthwiseConv2DWeightGradientPlan.KernelSize)}; // kh");
        L($"and.b32 %r15, %r0, {I(warpSize - 1)};                 // lane");
        L($"shr.u32 %r16, %r0, {I(warpShift)};                  // warp");
        L("mov.f32 %f4, 0f00000000;");
        L("mov.f32 %f5, 0f00000000;");
        L("mov.f32 %f6, 0f00000000;");
        L("mov.u32 %r4, %r0;");
        L("LOOP:");
        L($"setp.ge.u32 %p0, %r4, {I(plan.ReductionElements)};");
        L("@%p0 bra REDUCE;");
        L($"div.u32 %r5, %r4, {I(spatial)};       // n");
        L($"rem.u32 %r6, %r4, {I(spatial)};       // flattened (oh,ow)");
        L($"div.u32 %r7, %r6, {I(plan.Width)};    // oh");
        L($"rem.u32 %r8, %r6, {I(plan.Width)};    // ow");
        L($"mad.lo.u32 %r9, %r5, {I(channelSpatial)}, %r6;");
        L($"mad.lo.u32 %r9, %r2, {I(spatial)}, %r9;");
        L("mul.wide.u32 %rd4, %r9, 4;");
        L("add.u64 %rd5, %rd0, %rd4;");
        L("ld.global.nc.f32 %f0, [%rd5];");
        L("sub.u32 %r10, %r9, %r6;");
        L("mul.wide.u32 %rd6, %r10, 4;");
        L("add.u64 %rd6, %rd1, %rd6;");
        L("add.s32 %r11, %r7, %r3;");
        L("add.s32 %r11, %r11, -1;                // ih = oh + kh - 1");
        L("setp.ge.s32 %p1, %r11, 0;");
        L($"setp.lt.s32 %p2, %r11, {I(plan.Height)};");
        L("and.pred %p3, %p1, %p2;");

        for (int kw = 0; kw < CodegenDepthwiseConv2DWeightGradientPlan.KernelSize; kw++)
        {
            int value = 1 + kw;
            L($"add.s32 %r12, %r8, {I(kw - 1)};   // iw for kw={I(kw)}");
            L("setp.ge.s32 %p4, %r12, 0;");
            L($"setp.lt.s32 %p5, %r12, {I(plan.Width)};");
            L("and.pred %p6, %p4, %p5;");
            L("and.pred %p6, %p3, %p6;");
            L($"mad.lo.u32 %r13, %r11, {I(plan.Width)}, %r12;");
            L("mul.wide.s32 %rd7, %r13, 4;");
            L("add.u64 %rd7, %rd6, %rd7;");
            L($"mov.f32 %f{I(value)}, 0f00000000;");
            L($"@%p6 ld.global.nc.f32 %f{I(value)}, [%rd7];");
        }

        // SOFTWARE PIPELINE TWO INDEPENDENT POSITIONS. The limiter reports 66% long-
        // scoreboard stalls: a warp issues four loads, immediately consumes them, and
        // waits. Issue the next grid-stride position's four loads before either group is
        // consumed, giving the scheduler twice as much independent global latency to hide.
        // The second position is predicated rather than branched, so every thread still
        // reaches the block reduction even when the reduction extent has a tail.
        L($"add.u32 %r21, %r4, {I(CodegenDepthwiseConv2DWeightGradientPlan.BlockThreads)};");
        L($"setp.lt.u32 %p7, %r21, {I(plan.ReductionElements)};");
        L($"div.u32 %r5, %r21, {I(spatial)};");
        L($"rem.u32 %r6, %r21, {I(spatial)};");
        L($"div.u32 %r7, %r6, {I(plan.Width)};");
        L($"rem.u32 %r8, %r6, {I(plan.Width)};");
        L($"mad.lo.u32 %r9, %r5, {I(channelSpatial)}, %r6;");
        L($"mad.lo.u32 %r9, %r2, {I(spatial)}, %r9;");
        L("mul.wide.u32 %rd4, %r9, 4;");
        L("add.u64 %rd5, %rd0, %rd4;");
        L("mov.f32 %f7, 0f00000000;");
        L("@%p7 ld.global.nc.f32 %f7, [%rd5];");
        L("sub.u32 %r10, %r9, %r6;");
        L("mul.wide.u32 %rd6, %r10, 4;");
        L("add.u64 %rd6, %rd1, %rd6;");
        L("add.s32 %r11, %r7, %r3;");
        L("add.s32 %r11, %r11, -1;");
        L("setp.ge.s32 %p1, %r11, 0;");
        L($"setp.lt.s32 %p2, %r11, {I(plan.Height)};");
        L("and.pred %p3, %p1, %p2;");
        L("and.pred %p3, %p3, %p7;");
        for (int kw = 0; kw < CodegenDepthwiseConv2DWeightGradientPlan.KernelSize; kw++)
        {
            int value = 8 + kw;
            L($"add.s32 %r12, %r8, {I(kw - 1)};");
            L("setp.ge.s32 %p4, %r12, 0;");
            L($"setp.lt.s32 %p5, %r12, {I(plan.Width)};");
            L("and.pred %p6, %p4, %p5;");
            L("and.pred %p6, %p3, %p6;");
            L($"mad.lo.u32 %r13, %r11, {I(plan.Width)}, %r12;");
            L("mul.wide.s32 %rd7, %r13, 4;");
            L("add.u64 %rd7, %rd6, %rd7;");
            L($"mov.f32 %f{I(value)}, 0f00000000;");
            L($"@%p6 ld.global.nc.f32 %f{I(value)}, [%rd7];");
        }

        for (int kw = 0; kw < CodegenDepthwiseConv2DWeightGradientPlan.KernelSize; kw++)
        {
            L($"fma.rn.f32 %f{I(4 + kw)}, %f{I(1 + kw)}, %f0, %f{I(4 + kw)};");
            L($"fma.rn.f32 %f{I(4 + kw)}, %f{I(8 + kw)}, %f7, %f{I(4 + kw)};");
        }
        L($"add.u32 %r4, %r4, {I(2 * CodegenDepthwiseConv2DWeightGradientPlan.BlockThreads)};");
        L("bra LOOP;");

        L("REDUCE:");
        for (int kw = 0; kw < CodegenDepthwiseConv2DWeightGradientPlan.KernelSize; kw++)
            EmitWarpReduce(body, 4 + kw, 7 + kw);

        L("setp.ne.u32 %p7, %r15, 0;");
        L("@%p7 bra WARP_SUMS_STORED;");
        L("mul.lo.u32 %r17, %r16, 4;");
        L("cvt.u64.u32 %rd8, %r17;");
        L("add.u64 %rd8, %rd3, %rd8;");
        for (int kw = 0; kw < CodegenDepthwiseConv2DWeightGradientPlan.KernelSize; kw++)
            L($"st.shared.f32 [%rd8+{I(kw * plan.WarpsPerBlock * sizeof(float))}], %f{I(4 + kw)};");
        L("WARP_SUMS_STORED:");
        L("bar.sync 0;");
        L("setp.ne.u32 %p7, %r16, 0;");
        L("@%p7 bra END;");
        L($"setp.lt.u32 %p7, %r15, {I(plan.WarpsPerBlock)};");
        L("mul.wide.u32 %rd8, %r15, 4;");
        L("add.u64 %rd8, %rd3, %rd8;");
        for (int kw = 0; kw < CodegenDepthwiseConv2DWeightGradientPlan.KernelSize; kw++)
        {
            L($"mov.f32 %f{I(4 + kw)}, 0f00000000;");
            L($"@%p7 ld.shared.f32 %f{I(4 + kw)}, [%rd8+{I(kw * plan.WarpsPerBlock * sizeof(float))}];");
        }
        for (int kw = 0; kw < CodegenDepthwiseConv2DWeightGradientPlan.KernelSize; kw++)
            EmitWarpReduce(body, 4 + kw, 7 + kw);
        L("setp.ne.u32 %p7, %r15, 0;");
        L("@%p7 bra END;");
        L($"mad.lo.u32 %r18, %r2, {I(CodegenDepthwiseConv2DWeightGradientPlan.KernelSize)}, %r3;");
        L($"mul.lo.u32 %r18, %r18, {I(CodegenDepthwiseConv2DWeightGradientPlan.KernelSize)};");
        L("mul.wide.u32 %rd9, %r18, 4;");
        L("add.u64 %rd9, %rd2, %rd9;");
        for (int kw = 0; kw < CodegenDepthwiseConv2DWeightGradientPlan.KernelSize; kw++)
            L($"st.global.f32 [%rd9+{I(kw * sizeof(float))}], %f{I(4 + kw)};");
        L("END:");
        L("ret;");

        var text = new StringBuilder(18000);
        text.Append(".version ")
            .Append(PtxAffineEmitter.PtxIsaVersionFor(computeMajor, computeMinor)).Append('\n')
            .Append(".target sm_").Append(I(computeMajor)).Append(I(computeMinor)).Append('\n')
            .Append(".address_size 64\n\n")
            .Append("// generated by PtxDepthwiseConv2DWeightGradientEmitter\n")
            .Append("// one block per (channel,kh), three kw accumulators share dOut\n")
            .Append("// ").Append(spec.Describe().Replace("\n", "\n// ")).Append('\n')
            .Append(".visible .entry ").Append(spec.Name).Append("(\n");
        for (int i = 0; i < spec.ParameterCount; i++)
            text.Append("    .param .u64 p").Append(I(i))
                .Append(i == spec.ParameterCount - 1 ? "\n" : ",\n");
        text.Append(")\n{\n")
            .Append("    .shared .align 16 .b8 warp_sums[")
            .Append(I(plan.SharedMemoryBytes)).Append("];\n")
            .Append("    .reg .pred %p<8>;\n")
            .Append("    .reg .b32 %r<23>;\n")
            .Append("    .reg .b64 %rd<12>;\n")
            .Append("    .reg .f32 %f<12>;\n\n")
            .Append(body)
            .Append("}\n");
        return text.ToString();
    }

    private static void EmitWarpReduce(StringBuilder body, int accumulator, int scratch)
    {
        void L(string line) => body.Append("    ").Append(line).Append('\n');
        int warpSize = CodegenDepthwiseConv2DWeightGradientPlan.WarpSize;
        for (int offset = warpSize / 2; offset >= 1; offset >>= 1)
        {
            L($"mov.b32 %r19, %f{I(accumulator)};");
            L($"shfl.sync.down.b32 %r20, %r19, {I(offset)}, {I(warpSize - 1)}, 0xffffffff;");
            L($"mov.b32 %f{I(scratch)}, %r20;");
            L($"add.rn.f32 %f{I(accumulator)}, %f{I(accumulator)}, %f{I(scratch)};");
        }
    }

    private static int PowerOfTwoShift(int value)
    {
        if (value <= 0 || (value & (value - 1)) != 0)
            throw new InvalidOperationException("WarpSize must be a positive power of two.");
        int shift = 0;
        while ((value >>= 1) != 0) shift++;
        return shift;
    }

    private static string I(int value) => value.ToString(CultureInfo.InvariantCulture);
}
