using System;
using System.Collections.Generic;
using System.Globalization;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

internal enum DirectPtxCrossEntropyTarget
{
    Index,
    Dense
}

/// <summary>
/// Exact-shape fused FP32 projection and cross entropy. The championship cell
/// assigns one warp to each row in a single CTA, retains projected logits in
/// registers, and writes the mean loss once without atomics. Other supported
/// shapes use one block per row and atomic accumulation. Neither path
/// materializes logits in global memory.
/// </summary>
internal sealed class PtxFusedLinearCrossEntropyKernel : IDisposable
{
    internal const string IndexEntryPoint = "aidotnet_fused_linear_ce_index";
    internal const string DenseEntryPoint = "aidotnet_fused_linear_ce_dense";

    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

    internal DirectPtxCrossEntropyTarget TargetKind { get; }
    internal int Rows { get; }
    internal int HiddenDimension { get; }
    internal int Vocabulary { get; }
    internal int BlockThreads { get; }
    internal int StaticSharedBytes => GetStaticSharedBytes(
        TargetKind, Rows, HiddenDimension, Vocabulary, BlockThreads);
    internal string Ptx { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal PtxFusedLinearCrossEntropyKernel(
        DirectPtxRuntime runtime,
        DirectPtxCrossEntropyTarget targetKind,
        int rows,
        int hiddenDimension,
        int vocabulary)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        if (!DirectPtxArchitecture.HasValidatedFusedLinear(
            runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new PlatformNotSupportedException(
                "The fused linear/cross-entropy PTX specializations are measured only on GA10x/SM86.");
        Validate(targetKind, rows, hiddenDimension, vocabulary);

        TargetKind = targetKind;
        Rows = rows;
        HiddenDimension = hiddenDimension;
        Vocabulary = vocabulary;
        BlockThreads = GetBlockThreads(targetKind, rows, hiddenDimension, vocabulary);
        Blueprint = CreateBlueprint(
            runtime.ArchitectureFamily, targetKind, rows, hiddenDimension,
            vocabulary, BlockThreads);
        Ptx = EmitPtx(
            runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor,
            targetKind, rows, hiddenDimension, vocabulary);
        _module = runtime.LoadModule(Ptx);
        string entry = targetKind == DirectPtxCrossEntropyTarget.Index
            ? IndexEntryPoint : DenseEntryPoint;
        _function = _module.GetFunction(entry, out DirectPtxFunctionInfo info);
        int activeBlocks = _module.GetActiveBlocksPerMultiprocessor(_function, BlockThreads);
        Blueprint.ResourceBudget.Validate(entry, info, BlockThreads, activeBlocks);
        Audit = DirectPtxKernelAudit.Create(
            Blueprint, runtime.DeviceFingerprint, Ptx, info,
            BlockThreads, activeBlocks, _module);
    }

    internal unsafe void Launch(
        DirectPtxTensorView hidden,
        DirectPtxTensorView weight,
        DirectPtxTensorView bias,
        DirectPtxTensorView target,
        DirectPtxTensorView meanLoss)
    {
        Require(hidden, Blueprint.Tensors[0], nameof(hidden));
        Require(weight, Blueprint.Tensors[1], nameof(weight));
        Require(bias, Blueprint.Tensors[2], nameof(bias));
        Require(target, Blueprint.Tensors[3], nameof(target));
        Require(meanLoss, Blueprint.Tensors[4], nameof(meanLoss));
        if (Overlaps(meanLoss, hidden) || Overlaps(meanLoss, weight) ||
            Overlaps(meanLoss, bias) || Overlaps(meanLoss, target))
            throw new ArgumentException("Mean-loss output may not alias a fused CE input.");

        IntPtr hiddenPointer = hidden.Pointer;
        IntPtr weightPointer = weight.Pointer;
        IntPtr biasPointer = bias.Pointer;
        IntPtr targetPointer = target.Pointer;
        IntPtr lossPointer = meanLoss.Pointer;
        void** arguments = stackalloc void*[5];
        arguments[0] = &hiddenPointer;
        arguments[1] = &weightPointer;
        arguments[2] = &biasPointer;
        arguments[3] = &targetPointer;
        arguments[4] = &lossPointer;
        uint grid = IsExactIndexCell(TargetKind, Rows, HiddenDimension, Vocabulary)
            ? 1u : checked((uint)Rows);
        _module.Launch(
            _function, grid, 1, 1,
            checked((uint)BlockThreads), 1, 1, 0, arguments);
    }

    public void Dispose() => _module.Dispose();

    internal static string EmitPtx(
        int ccMajor,
        int ccMinor,
        DirectPtxCrossEntropyTarget targetKind,
        int rows,
        int hiddenDimension,
        int vocabulary)
    {
        Validate(targetKind, rows, hiddenDimension, vocabulary);
        if (IsExactIndexCell(targetKind, rows, hiddenDimension, vocabulary))
            return EmitExactIndexPtx(ccMajor, ccMinor);
        bool index = targetKind == DirectPtxCrossEntropyTarget.Index;
        int blockThreads = ThreadsForVocabulary(vocabulary);
        int staticSharedBytes = 2 * blockThreads * sizeof(float);
        var ptx = new StringBuilder(16_384);
        ptx.AppendLine(".version 7.1");
        ptx.AppendLine($".target sm_{ccMajor}{ccMinor}");
        ptx.AppendLine(".address_size 64");
        ptx.AppendLine();
        ptx.AppendLine($"// exact fused projection/CE rows={rows} d={hiddenDimension} vocab={vocabulary}");
        ptx.AppendLine($".visible .entry {(index ? IndexEntryPoint : DenseEntryPoint)}(");
        ptx.AppendLine("    .param .u64 hidden_ptr,");
        ptx.AppendLine("    .param .u64 weight_ptr,");
        ptx.AppendLine("    .param .u64 bias_ptr,");
        ptx.AppendLine("    .param .u64 target_ptr,");
        ptx.AppendLine("    .param .u64 loss_ptr");
        ptx.AppendLine(")");
        ptx.AppendLine($".maxntid {blockThreads}, 1, 1");
        ptx.AppendLine("{");
        ptx.AppendLine("    .reg .pred %p<12>;");
        ptx.AppendLine("    .reg .b32 %r<20>;");
        ptx.AppendLine("    .reg .b64 %rd<24>;");
        ptx.AppendLine("    .reg .f32 %f<16>;");
        ptx.AppendLine($"    .shared .align 16 .b8 scratch[{staticSharedBytes}];");
        ptx.AppendLine("    ld.param.u64 %rd0, [hidden_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd1, [weight_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd2, [bias_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd3, [target_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd4, [loss_ptr];");
        ptx.AppendLine("    mov.u32 %r0, %tid.x;");
        ptx.AppendLine("    mov.u32 %r1, %ctaid.x;");
        ptx.AppendLine($"    mul.wide.u32 %rd5, %r1, {hiddenDimension * sizeof(float)};");
        ptx.AppendLine("    add.u64 %rd6, %rd0, %rd5;");
        ptx.AppendLine("    mov.u64 %rd7, scratch;");
        ptx.AppendLine($"    add.u64 %rd8, %rd7, {blockThreads * sizeof(float)};");
        if (index)
        {
            ptx.AppendLine("    mul.wide.u32 %rd9, %r1, 4;");
            ptx.AppendLine("    add.u64 %rd10, %rd3, %rd9;");
            ptx.AppendLine("    ld.global.nc.f32 %f0, [%rd10];");
            ptx.AppendLine("    add.rn.f32 %f0, %f0, 0f3F000000;");
            ptx.AppendLine("    cvt.rzi.s32.f32 %r10, %f0;");
            ptx.AppendLine("    setp.lt.s32 %p0, %r10, 0;");
            ptx.AppendLine($"    setp.ge.s32 %p1, %r10, {vocabulary};");
            ptx.AppendLine("    or.pred %p2, %p0, %p1;");
            ptx.AppendLine("    @%p2 bra.uni CE_INVALID_INDEX;");
        }

        EmitProjectionPassStart(
            ptx, "CE_MAX_LOOP", hiddenDimension, vocabulary, maxPass: true);
        if (index)
        {
            ptx.AppendLine("    setp.eq.u32 %p4, %r2, %r10;");
            ptx.AppendLine("    @%p4 mov.f32 %f2, %f4;");
        }
        ptx.AppendLine($"    add.u32 %r2, %r2, {blockThreads};");
        ptx.AppendLine("    bra.uni CE_MAX_LOOP;");
        ptx.AppendLine("CE_MAX_REDUCE:");
        EmitSharedPublish(ptx, maxValue: "%f1", auxiliaryValue: index ? "%f2" : "0f00000000");
        EmitReduction(
            ptx, "CE_MAX", blockThreads, maxFirst: true, addSecond: index);

        ptx.AppendLine("    ld.shared.f32 %f5, [%rd7];");
        if (index)
            ptx.AppendLine("    ld.shared.f32 %f2, [%rd8];");
        ptx.AppendLine("    mov.u32 %r2, %r0;");
        ptx.AppendLine("    mov.f32 %f6, 0f00000000;");
        ptx.AppendLine("CE_SUM_LOOP:");
        ptx.AppendLine($"    setp.ge.u32 %p3, %r2, {vocabulary};");
        ptx.AppendLine("    @%p3 bra.uni CE_SUM_REDUCE;");
        EmitProjection(ptx, "SUM", hiddenDimension, vocabulary);
        ptx.AppendLine("    sub.rn.f32 %f7, %f4, %f5;");
        ptx.AppendLine("    mul.rn.f32 %f7, %f7, 0f3FB8AA3B;");
        ptx.AppendLine("    ex2.approx.f32 %f7, %f7;");
        ptx.AppendLine("    add.rn.f32 %f6, %f6, %f7;");
        ptx.AppendLine($"    add.u32 %r2, %r2, {blockThreads};");
        ptx.AppendLine("    bra.uni CE_SUM_LOOP;");
        ptx.AppendLine("CE_SUM_REDUCE:");
        EmitSharedPublish(ptx, maxValue: "%f6", auxiliaryValue: "0f00000000");
        EmitReduction(
            ptx, "CE_SUM", blockThreads, maxFirst: false, addSecond: false);
        ptx.AppendLine("    setp.ne.u32 %p5, %r0, 0;");
        ptx.AppendLine("    @%p5 bra.uni CE_LSE_READY;");
        ptx.AppendLine("    ld.shared.f32 %f8, [%rd7];");
        ptx.AppendLine("    lg2.approx.f32 %f8, %f8;");
        ptx.AppendLine("    fma.rn.f32 %f8, %f8, 0f3F317218, %f5;");
        ptx.AppendLine("    st.shared.f32 [%rd7], %f8;");
        ptx.AppendLine("CE_LSE_READY:");
        ptx.AppendLine("    bar.sync 0;");

        if (index)
        {
            ptx.AppendLine("    @%p5 bra.uni CE_DONE;");
            ptx.AppendLine("    ld.shared.f32 %f8, [%rd7];");
            ptx.AppendLine("    sub.rn.f32 %f10, %f8, %f2;");
            ptx.AppendLine($"    mul.rn.f32 %f10, %f10, {FloatLiteral(1f / rows)};");
            ptx.AppendLine("    atom.global.add.f32 %f11, [%rd4], %f10;");
        }
        else
        {
            ptx.AppendLine("    ld.shared.f32 %f8, [%rd7];");
            ptx.AppendLine("    mov.u32 %r2, %r0;");
            ptx.AppendLine("    mov.f32 %f9, 0f00000000;");
            ptx.AppendLine("CE_DENSE_LOOP:");
            ptx.AppendLine($"    setp.ge.u32 %p3, %r2, {vocabulary};");
            ptx.AppendLine("    @%p3 bra.uni CE_DENSE_REDUCE;");
            EmitProjection(ptx, "DENSE", hiddenDimension, vocabulary);
            ptx.AppendLine($"    mad.lo.u32 %r11, %r1, {vocabulary}, %r2;");
            ptx.AppendLine("    mul.wide.u32 %rd19, %r11, 4;");
            ptx.AppendLine("    add.u64 %rd20, %rd3, %rd19;");
            ptx.AppendLine("    ld.global.nc.f32 %f10, [%rd20];");
            ptx.AppendLine("    sub.rn.f32 %f11, %f4, %f8;");
            ptx.AppendLine("    fma.rn.f32 %f9, %f10, %f11, %f9;");
            ptx.AppendLine($"    add.u32 %r2, %r2, {blockThreads};");
            ptx.AppendLine("    bra.uni CE_DENSE_LOOP;");
            ptx.AppendLine("CE_DENSE_REDUCE:");
            EmitSharedPublish(ptx, maxValue: "%f9", auxiliaryValue: "0f00000000");
            EmitReduction(
                ptx, "CE_DENSE", blockThreads, maxFirst: false, addSecond: false);
            ptx.AppendLine("    setp.ne.u32 %p5, %r0, 0;");
            ptx.AppendLine("    @%p5 bra.uni CE_DONE;");
            ptx.AppendLine("    ld.shared.f32 %f12, [%rd7];");
            ptx.AppendLine("    neg.f32 %f12, %f12;");
            ptx.AppendLine($"    mul.rn.f32 %f12, %f12, {FloatLiteral(1f / rows)};");
            ptx.AppendLine("    atom.global.add.f32 %f13, [%rd4], %f12;");
        }

        ptx.AppendLine("CE_DONE:");
        ptx.AppendLine("    ret;");
        if (index)
        {
            ptx.AppendLine("CE_INVALID_INDEX:");
            ptx.AppendLine("    setp.ne.u32 %p5, %r0, 0;");
            ptx.AppendLine("    @%p5 bra.uni CE_DONE;");
            ptx.AppendLine("    mov.f32 %f10, 0f7F800000;");
            ptx.AppendLine("    atom.global.add.f32 %f11, [%rd4], %f10;");
            ptx.AppendLine("    bra.uni CE_DONE;");
        }
        ptx.AppendLine("}");
        return ptx.ToString();
    }

    private static string EmitExactIndexPtx(int ccMajor, int ccMinor)
    {
        const int blockThreads = 128;
        var ptx = new StringBuilder(16_384);
        ptx.AppendLine(".version 7.1");
        ptx.AppendLine($".target sm_{ccMajor}{ccMinor}");
        ptx.AppendLine(".address_size 64");
        ptx.AppendLine();
        ptx.AppendLine("// exact fused projection/index-CE B4 K16 V32; one warp per row; logits retained in registers");
        ptx.AppendLine($".visible .entry {IndexEntryPoint}(");
        ptx.AppendLine("    .param .u64 hidden_ptr,");
        ptx.AppendLine("    .param .u64 weight_ptr,");
        ptx.AppendLine("    .param .u64 bias_ptr,");
        ptx.AppendLine("    .param .u64 target_ptr,");
        ptx.AppendLine("    .param .u64 loss_ptr");
        ptx.AppendLine(")");
        ptx.AppendLine($".maxntid {blockThreads}, 1, 1");
        ptx.AppendLine("{");
        ptx.AppendLine("    .reg .pred %p<8>;");
        ptx.AppendLine("    .reg .b32 %r<20>;");
        ptx.AppendLine("    .reg .b64 %rd<16>;");
        ptx.AppendLine("    .reg .f32 %f<16>;");
        ptx.AppendLine("    .shared .align 16 .b8 row_loss[16];");
        ptx.AppendLine("    ld.param.u64 %rd0, [hidden_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd1, [weight_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd2, [bias_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd3, [target_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd4, [loss_ptr];");
        ptx.AppendLine("    mov.u32 %r0, %tid.x;");
        ptx.AppendLine("    and.b32 %r1, %r0, 31;");
        ptx.AppendLine("    shr.u32 %r2, %r0, 5;");
        ptx.AppendLine("    mul.wide.u32 %rd5, %r2, 64;");
        ptx.AppendLine("    add.u64 %rd6, %rd0, %rd5;");
        ptx.AppendLine("    mul.wide.u32 %rd7, %r1, 4;");
        ptx.AppendLine("    add.u64 %rd8, %rd6, %rd7;");
        ptx.AppendLine("    setp.lt.u32 %p0, %r1, 16;");
        ptx.AppendLine("    mov.f32 %f1, 0f00000000;");
        ptx.AppendLine("    @%p0 ld.global.f32 %f1, [%rd8];");
        ptx.AppendLine("    mov.b32 %r7, %f1;");
        ptx.AppendLine("    add.u64 %rd9, %rd2, %rd7;");
        ptx.AppendLine("    ld.global.f32 %f0, [%rd9];");
        ptx.AppendLine("    add.u64 %rd10, %rd1, %rd7;");
        for (int k = 0; k < 16; k++)
        {
            ptx.AppendLine($"    shfl.sync.idx.b32 %r8, %r7, {k}, 31, 0xffffffff;");
            ptx.AppendLine("    mov.b32 %f2, %r8;");
            string suffix = k == 0 ? string.Empty : $"+{k * 128}";
            ptx.AppendLine($"    ld.global.f32 %f3, [%rd10{suffix}];");
            ptx.AppendLine("    fma.rn.f32 %f0, %f2, %f3, %f0;");
        }
        ptx.AppendLine("    setp.eq.u32 %p1, %r1, 0;");
        ptx.AppendLine("    mov.f32 %f6, 0f00000000;");
        ptx.AppendLine("    mul.wide.u32 %rd11, %r2, 4;");
        ptx.AppendLine("    add.u64 %rd12, %rd3, %rd11;");
        ptx.AppendLine("    @%p1 ld.global.f32 %f6, [%rd12];");
        ptx.AppendLine("    mov.u32 %r5, 0;");
        ptx.AppendLine("    @%p1 cvt.rni.s32.f32 %r5, %f6;");
        ptx.AppendLine("    shfl.sync.idx.b32 %r6, %r5, 0, 31, 0xffffffff;");

        ptx.AppendLine("    mov.f32 %f4, %f0;");
        EmitWarpFloatReduction(ptx, "%f4", "%f7", "%r8", "%r9", maximum: true);
        ptx.AppendLine("    mov.b32 %r8, %f4;");
        ptx.AppendLine("    shfl.sync.idx.b32 %r9, %r8, 0, 31, 0xffffffff;");
        ptx.AppendLine("    mov.b32 %f4, %r9;");
        ptx.AppendLine("    sub.rn.f32 %f5, %f0, %f4;");
        ptx.AppendLine("    mul.rn.f32 %f5, %f5, 0f3FB8AA3B;");
        ptx.AppendLine("    ex2.approx.f32 %f5, %f5;");
        EmitWarpFloatReduction(ptx, "%f5", "%f7", "%r8", "%r9", maximum: false);
        ptx.AppendLine("    setp.eq.u32 %p2, %r1, %r6;");
        ptx.AppendLine("    mov.f32 %f8, 0f00000000;");
        ptx.AppendLine("    @%p2 mov.f32 %f8, %f0;");
        EmitWarpFloatReduction(ptx, "%f8", "%f7", "%r8", "%r9", maximum: false);

        ptx.AppendLine("    setp.lt.s32 %p3, %r6, 0;");
        ptx.AppendLine("    setp.ge.s32 %p4, %r6, 32;");
        ptx.AppendLine("    or.pred %p5, %p3, %p4;");
        ptx.AppendLine("    @%p1 lg2.approx.f32 %f9, %f5;");
        ptx.AppendLine("    @%p1 fma.rn.f32 %f9, %f9, 0f3F317218, %f4;");
        ptx.AppendLine("    @%p1 sub.rn.f32 %f9, %f9, %f8;");
        ptx.AppendLine("    @%p1 mul.rn.f32 %f9, %f9, 0f3E800000;");
        ptx.AppendLine("    @%p5 mov.f32 %f9, 0f7F800000;");
        ptx.AppendLine("    mov.u64 %rd13, row_loss;");
        ptx.AppendLine("    add.u64 %rd14, %rd13, %rd11;");
        ptx.AppendLine("    @%p1 st.shared.f32 [%rd14], %f9;");
        ptx.AppendLine("    bar.sync 0;");
        ptx.AppendLine("    setp.ne.u32 %p6, %r2, 0;");
        ptx.AppendLine("    @%p6 bra.uni CE_EXACT_DONE;");
        ptx.AppendLine("    setp.lt.u32 %p7, %r1, 4;");
        ptx.AppendLine("    mov.f32 %f10, 0f00000000;");
        ptx.AppendLine("    add.u64 %rd15, %rd13, %rd7;");
        ptx.AppendLine("    @%p7 ld.shared.f32 %f10, [%rd15];");
        EmitWarpFloatReduction(ptx, "%f10", "%f11", "%r8", "%r9", maximum: false);
        ptx.AppendLine("    @%p1 st.global.f32 [%rd4], %f10;");
        ptx.AppendLine("CE_EXACT_DONE:");
        ptx.AppendLine("    ret;");
        ptx.AppendLine("}");
        return ptx.ToString();
    }

    private static void EmitWarpFloatReduction(
        StringBuilder ptx,
        string value,
        string shuffled,
        string valueBits,
        string shuffledBits,
        bool maximum)
    {
        foreach (int offset in new[] { 16, 8, 4, 2, 1 })
        {
            ptx.AppendLine($"    mov.b32 {valueBits}, {value};");
            ptx.AppendLine($"    shfl.sync.down.b32 {shuffledBits}, {valueBits}, {offset}, 31, 0xffffffff;");
            ptx.AppendLine($"    mov.b32 {shuffled}, {shuffledBits};");
            ptx.AppendLine(maximum
                ? $"    max.f32 {value}, {value}, {shuffled};"
                : $"    add.rn.f32 {value}, {value}, {shuffled};");
        }
    }

    private static void EmitProjectionPassStart(
        StringBuilder ptx,
        string loop,
        int hiddenDimension,
        int vocabulary,
        bool maxPass)
    {
        ptx.AppendLine("    mov.u32 %r2, %r0;");
        ptx.AppendLine(maxPass
            ? "    mov.f32 %f1, 0fFE967699;"
            : "    mov.f32 %f1, 0f00000000;");
        ptx.AppendLine("    mov.f32 %f2, 0f00000000;");
        ptx.AppendLine($"{loop}:");
        ptx.AppendLine($"    setp.ge.u32 %p3, %r2, {vocabulary};");
        ptx.AppendLine("    @%p3 bra.uni CE_MAX_REDUCE;");
        EmitProjection(ptx, "MAX", hiddenDimension, vocabulary);
        ptx.AppendLine("    max.f32 %f1, %f1, %f4;");
    }

    private static void EmitProjection(
        StringBuilder ptx,
        string prefix,
        int hiddenDimension,
        int vocabulary)
    {
        ptx.AppendLine("    mul.wide.u32 %rd11, %r2, 4;");
        ptx.AppendLine("    add.u64 %rd12, %rd2, %rd11;");
        ptx.AppendLine("    ld.global.nc.f32 %f4, [%rd12];");
        ptx.AppendLine("    mov.u32 %r3, 0;");
        ptx.AppendLine($"CE_{prefix}_PROJECT_K:");
        ptx.AppendLine($"    setp.ge.u32 %p6, %r3, {hiddenDimension};");
        ptx.AppendLine($"    @%p6 bra.uni CE_{prefix}_PROJECT_DONE;");
        ptx.AppendLine("    mul.wide.u32 %rd13, %r3, 4;");
        ptx.AppendLine("    add.u64 %rd14, %rd6, %rd13;");
        ptx.AppendLine("    ld.global.nc.f32 %f3, [%rd14];");
        ptx.AppendLine($"    mad.lo.u32 %r4, %r3, {vocabulary}, %r2;");
        ptx.AppendLine("    mul.wide.u32 %rd15, %r4, 4;");
        ptx.AppendLine("    add.u64 %rd16, %rd1, %rd15;");
        ptx.AppendLine("    ld.global.nc.f32 %f10, [%rd16];");
        ptx.AppendLine("    fma.rn.f32 %f4, %f3, %f10, %f4;");
        ptx.AppendLine("    add.u32 %r3, %r3, 1;");
        ptx.AppendLine($"    bra.uni CE_{prefix}_PROJECT_K;");
        ptx.AppendLine($"CE_{prefix}_PROJECT_DONE:");
    }

    private static void EmitSharedPublish(
        StringBuilder ptx,
        string maxValue,
        string auxiliaryValue)
    {
        ptx.AppendLine("    mul.wide.u32 %rd17, %r0, 4;");
        ptx.AppendLine("    add.u64 %rd18, %rd7, %rd17;");
        ptx.AppendLine($"    st.shared.f32 [%rd18], {maxValue};");
        ptx.AppendLine("    add.u64 %rd21, %rd8, %rd17;");
        ptx.AppendLine($"    st.shared.f32 [%rd21], {auxiliaryValue};");
        ptx.AppendLine("    bar.sync 0;");
    }

    private static void EmitReduction(
        StringBuilder ptx,
        string prefix,
        int blockThreads,
        bool maxFirst,
        bool addSecond)
    {
        for (int offset = blockThreads / 2; offset >= 1; offset /= 2)
        {
            ptx.AppendLine($"    setp.ge.u32 %p7, %r0, {offset};");
            ptx.AppendLine($"    @%p7 bra.uni {prefix}_SKIP_{offset};");
            ptx.AppendLine($"    add.u32 %r5, %r0, {offset};");
            ptx.AppendLine("    mul.wide.u32 %rd22, %r5, 4;");
            ptx.AppendLine("    add.u64 %rd23, %rd7, %rd22;");
            ptx.AppendLine("    ld.shared.f32 %f12, [%rd18];");
            ptx.AppendLine("    ld.shared.f32 %f13, [%rd23];");
            ptx.AppendLine(maxFirst
                ? "    max.f32 %f12, %f12, %f13;"
                : "    add.rn.f32 %f12, %f12, %f13;");
            ptx.AppendLine("    st.shared.f32 [%rd18], %f12;");
            if (addSecond)
            {
                ptx.AppendLine("    add.u64 %rd23, %rd8, %rd22;");
                ptx.AppendLine("    ld.shared.f32 %f14, [%rd21];");
                ptx.AppendLine("    ld.shared.f32 %f15, [%rd23];");
                ptx.AppendLine("    add.rn.f32 %f14, %f14, %f15;");
                ptx.AppendLine("    st.shared.f32 [%rd21], %f14;");
            }
            ptx.AppendLine($"{prefix}_SKIP_{offset}:");
            ptx.AppendLine("    bar.sync 0;");
        }
    }

    private static DirectPtxKernelBlueprint CreateBlueprint(
        DirectPtxArchitectureFamily architecture,
        DirectPtxCrossEntropyTarget targetKind,
        int rows,
        int hiddenDimension,
        int vocabulary,
        int blockThreads)
    {
        bool index = targetKind == DirectPtxCrossEntropyTarget.Index;
        var hidden = new DirectPtxExtent(rows, hiddenDimension);
        var weight = new DirectPtxExtent(hiddenDimension, vocabulary);
        var bias = new DirectPtxExtent(vocabulary);
        var target = index
            ? new DirectPtxExtent(rows)
            : new DirectPtxExtent(rows, vocabulary);
        var loss = new DirectPtxExtent(1);
        return new DirectPtxKernelBlueprint(
            Operation: index
                ? "fused-linear-cross-entropy-index"
                : "fused-linear-cross-entropy-dense",
            Version: IsExactIndexCell(targetKind, rows, hiddenDimension, vocabulary) ? 2 : 1,
            Architecture: architecture,
            Variant: $"{(IsExactIndexCell(targetKind, rows, hiddenDimension, vocabulary) ? "one-cta-warp-row-register-logits-fp32" : "fp32")}-b{rows}-k{hiddenDimension}-v{vocabulary}",
            Tensors:
            [
                new("hidden", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D,
                    hidden, hidden, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("weight", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.LinearWeightInputMajor,
                    weight, weight, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("bias", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    bias, bias, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("target", DirectPtxPhysicalType.Float32,
                    index ? DirectPtxPhysicalLayout.Vector : DirectPtxPhysicalLayout.RowMajor2D,
                    target, target, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("meanLoss", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    loss, loss, 16,
                    IsExactIndexCell(targetKind, rows, hiddenDimension, vocabulary)
                        ? DirectPtxTensorAccess.Write : DirectPtxTensorAccess.ReadWrite,
                    DirectPtxExtentMode.Exact)
            ],
            ResourceBudget: new DirectPtxResourceBudget(
                MaxRegistersPerThread: IsExactIndexCell(targetKind, rows, hiddenDimension, vocabulary) ? 64 : 48,
                MaxStaticSharedBytes: GetStaticSharedBytes(
                    targetKind, rows, hiddenDimension, vocabulary, blockThreads),
                MaxLocalBytesPerThread: 0,
                MinBlocksPerMultiprocessor:
                    IsExactIndexCell(targetKind, rows, hiddenDimension, vocabulary) ? 8 : 2),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["formula"] = index
                    ? "mean(logsumexp(hidden@weight+bias)-selected_logit)"
                    : "mean(-sum(target*(hidden@weight+bias-logsumexp)))",
                ["projection-layout"] = "canonical-contiguous-input-major-weight",
                ["logits"] = IsExactIndexCell(targetKind, rows, hiddenDimension, vocabulary)
                    ? "register-only-single-projection-pass-never-materialized"
                    : "register-only-recomputed-never-materialized",
                ["row-reductions"] = IsExactIndexCell(targetKind, rows, hiddenDimension, vocabulary)
                    ? "warp-shuffle max/sum/selected; four shared row losses; one barrier"
                    : "shared-max-and-sum",
                ["block-threads"] = blockThreads.ToString(CultureInfo.InvariantCulture),
                ["output"] = IsExactIndexCell(targetKind, rows, hiddenDimension, vocabulary)
                    ? "one directly stored resident scalar; no atomic and no pre-clear"
                    : "one resident scalar; caller clears before launch",
                ["shape-parameters"] = "none",
                ["stride-parameters"] = "none",
                ["temporary-device-allocation"] = "none"
            });
    }

    internal static int ThreadsForVocabulary(int vocabulary) => vocabulary switch
    {
        <= 32 => 32,
        <= 64 => 64,
        <= 128 => 128,
        _ => 256
    };

    internal static bool IsExactIndexCell(
        DirectPtxCrossEntropyTarget targetKind,
        int rows,
        int hiddenDimension,
        int vocabulary) =>
        targetKind == DirectPtxCrossEntropyTarget.Index &&
        rows == 4 && hiddenDimension == 16 && vocabulary == 32;

    private static int GetBlockThreads(
        DirectPtxCrossEntropyTarget targetKind,
        int rows,
        int hiddenDimension,
        int vocabulary) =>
        IsExactIndexCell(targetKind, rows, hiddenDimension, vocabulary)
            ? 128 : ThreadsForVocabulary(vocabulary);

    private static int GetStaticSharedBytes(
        DirectPtxCrossEntropyTarget targetKind,
        int rows,
        int hiddenDimension,
        int vocabulary,
        int blockThreads) =>
        IsExactIndexCell(targetKind, rows, hiddenDimension, vocabulary)
            ? 4 * sizeof(float) : 2 * blockThreads * sizeof(float);

    private static string FloatLiteral(float value) =>
        "0f" + PtxCompat.SingleToInt32Bits(value).ToString("X8", CultureInfo.InvariantCulture);

    private static void Validate(
        DirectPtxCrossEntropyTarget targetKind,
        int rows,
        int hiddenDimension,
        int vocabulary)
    {
        if (!Enum.IsDefined(typeof(DirectPtxCrossEntropyTarget), targetKind))
            throw new ArgumentOutOfRangeException(nameof(targetKind));
        if (rows <= 0 || rows > 65_535)
            throw new ArgumentOutOfRangeException(nameof(rows));
        if (hiddenDimension <= 0 || hiddenDimension > 65_536)
            throw new ArgumentOutOfRangeException(nameof(hiddenDimension));
        if (vocabulary <= 0 || vocabulary > 65_536)
            throw new ArgumentOutOfRangeException(nameof(vocabulary));
        _ = checked(rows * hiddenDimension);
        _ = checked(hiddenDimension * vocabulary);
        _ = checked(rows * vocabulary);
    }

    private static void Require(
        DirectPtxTensorView view,
        DirectPtxTensorContract contract,
        string parameter)
    {
        if (view.Pointer == IntPtr.Zero || view.PhysicalType != contract.PhysicalType ||
            view.Layout != contract.Layout || view.LogicalExtent != contract.LogicalExtent ||
            view.PhysicalExtent != contract.PhysicalExtent || view.ByteLength != contract.RequiredBytes)
            throw new ArgumentException(
                $"{parameter} does not satisfy physical ABI '{contract.Name}'.", parameter);
    }

    private static bool Overlaps(DirectPtxTensorView left, DirectPtxTensorView right)
    {
        nuint leftStart = PtxCompat.ToNuint(left.Pointer);
        nuint rightStart = PtxCompat.ToNuint(right.Pointer);
        nuint leftEnd = checked(leftStart + left.ByteLength);
        nuint rightEnd = checked(rightStart + right.ByteLength);
        return leftStart < rightEnd && rightStart < leftEnd;
    }
}
