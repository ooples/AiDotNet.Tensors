using System;
using System.Collections.Generic;
using System.Globalization;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// Exact-shape fused-linear backward family. One launch assigns disjoint CTA
/// ranges to dInput and dWeight. The championship ReLU cell stages FP32 values
/// as FP16 multiplicands into two split-K shared WMMA partitions, merges their
/// FP32 accumulators into one 32x32 output tile, and uses 80 CTAs to cover the
/// GPU; other supported shapes use the generic scalar topology.
/// The k=0 dWeight tiles also fold dBias in FP32, so neither a masked-gradient
/// tensor nor a separate bias-reduction launch is materialized.
/// </summary>
internal sealed class PtxFusedLinearBackwardKernel : IDisposable
{
    internal const int BlockThreads = 256;
    internal const int ExactBlockThreads = 256;
    internal const int Tile = 16;
    internal const string EntryPoint = "aidotnet_fused_linear_backward";

    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;
    private readonly int _blockThreads;

    internal int M { get; }
    internal int K { get; }
    internal int N { get; }
    internal DirectPtxLinearActivation Activation { get; }
    internal string Ptx { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal IReadOnlyList<DirectPtxKernelAudit> Audits { get; }

    internal PtxFusedLinearBackwardKernel(
        DirectPtxRuntime runtime,
        int m,
        int k,
        int n,
        DirectPtxLinearActivation activation)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        if (!DirectPtxArchitecture.HasValidatedFusedLinear(
            runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new PlatformNotSupportedException(
                "The fused-linear backward PTX family is measured only on GA10x/SM86.");
        Validate(m, k, n, activation);

        M = m;
        K = k;
        N = n;
        Activation = activation;
        Blueprint = CreateBlueprint(runtime.ArchitectureFamily, m, k, n, activation);
        Ptx = EmitPtx(runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor, m, k, n, activation);
        _module = runtime.LoadModule(Ptx);
        _function = _module.GetFunction(EntryPoint, out DirectPtxFunctionInfo info);
        _blockThreads = IsExactChampionshipCell(m, k, n, activation)
            ? ExactBlockThreads : BlockThreads;
        int activeBlocks = _module.GetActiveBlocksPerMultiprocessor(
            _function, _blockThreads);
        Blueprint.ResourceBudget.Validate(
            EntryPoint, info, _blockThreads, activeBlocks);
        DirectPtxKernelAudit audit = DirectPtxKernelAudit.Create(
            Blueprint, runtime.DeviceFingerprint, Ptx, info,
            _blockThreads, activeBlocks, _module);
        Audits = [audit];
    }

    internal unsafe void Launch(
        DirectPtxTensorView gradOutput,
        DirectPtxTensorView input,
        DirectPtxTensorView weights,
        DirectPtxTensorView saved,
        DirectPtxTensorView gradInput,
        DirectPtxTensorView gradWeight,
        DirectPtxTensorView gradBias)
    {
        Require(gradOutput, Blueprint.Tensors[0], nameof(gradOutput));
        Require(input, Blueprint.Tensors[1], nameof(input));
        Require(weights, Blueprint.Tensors[2], nameof(weights));
        Require(saved, Blueprint.Tensors[3], nameof(saved));
        Require(gradInput, Blueprint.Tensors[4], nameof(gradInput));
        Require(gradWeight, Blueprint.Tensors[5], nameof(gradWeight));
        Require(gradBias, Blueprint.Tensors[6], nameof(gradBias));
        if (Overlaps(gradInput, gradOutput) || Overlaps(gradInput, input) ||
            Overlaps(gradInput, weights) || Overlaps(gradInput, saved) ||
            Overlaps(gradWeight, gradOutput) || Overlaps(gradWeight, input) ||
            Overlaps(gradWeight, weights) || Overlaps(gradWeight, saved) ||
            Overlaps(gradBias, gradOutput) || Overlaps(gradBias, input) ||
            Overlaps(gradBias, weights) || Overlaps(gradBias, saved))
            throw new ArgumentException(
                "Fused-linear gradients may not alias saved forward tensors.");
        if (Overlaps(gradInput, gradWeight) || Overlaps(gradInput, gradBias) ||
            Overlaps(gradWeight, gradBias))
            throw new ArgumentException("Fused-linear gradient outputs must be disjoint.");

        IntPtr go = gradOutput.Pointer;
        IntPtr x = input.Pointer;
        IntPtr w = weights.Pointer;
        IntPtr s = saved.Pointer;
        IntPtr gi = gradInput.Pointer;
        IntPtr gw = gradWeight.Pointer;
        IntPtr gb = gradBias.Pointer;

        void** arguments = stackalloc void*[7];
        arguments[0] = &go;
        arguments[1] = &x;
        arguments[2] = &w;
        arguments[3] = &s;
        arguments[4] = &gi;
        arguments[5] = &gw;
        arguments[6] = &gb;
        uint grid = IsExactChampionshipCell(M, K, N, Activation)
            ? 80u
            : checked(Grid(checked(M * K)) + Grid(checked(K * N)));
        _module.Launch(
            _function, grid, 1, 1,
            checked((uint)_blockThreads), 1, 1, 0, arguments);
    }

    private static uint Grid(int elements) =>
        checked((uint)(((long)elements + BlockThreads - 1) / BlockThreads));

    public void Dispose() => _module.Dispose();

    internal static string EmitPtx(
        int ccMajor,
        int ccMinor,
        int m,
        int k,
        int n,
        DirectPtxLinearActivation activation)
    {
        Validate(m, k, n, activation);
        var ptx = new StringBuilder(12_000);
        ptx.AppendLine(".version 7.1");
        ptx.AppendLine($".target sm_{ccMajor}{ccMinor}");
        ptx.AppendLine(".address_size 64");
        ptx.AppendLine($"// fused-linear backward M={m} K={k} N={n} act={activation}; single exact pointer-only ABI");
        if (IsExactChampionshipCell(m, k, n, activation))
            EmitExactReluChampionshipSplitK(ptx);
        else
            EmitCombined(ptx, m, k, n, activation);
        return ptx.ToString();
    }

    private static void EmitExactReluChampionshipSplitK(StringBuilder ptx)
    {
        ptx.AppendLine("// championship M64 K256 N256 ReLU backward; 80 CTAs; two four-warp split-K groups per 32x32 output tile; in-tile dBias fold; no global intermediates");
        ptx.AppendLine($".visible .entry {EntryPoint}(");
        string[] parameters =
        [
            "grad_output_ptr", "input_ptr", "weights_ptr", "saved_ptr",
            "grad_input_ptr", "grad_weight_ptr", "grad_bias_ptr"
        ];
        for (int i = 0; i < parameters.Length; i++)
            ptx.AppendLine($"    .param .u64 {parameters[i]}{(i + 1 == parameters.Length ? string.Empty : ",")}");
        ptx.AppendLine(")");
        ptx.AppendLine($".maxntid {ExactBlockThreads}, 1, 1");
        ptx.AppendLine("{");
        ptx.AppendLine("    .reg .pred %p<8>;");
        ptx.AppendLine("    .reg .b16 %h<8>;");
        ptx.AppendLine("    .reg .b32 %r<32>;");
        ptx.AppendLine("    .reg .b32 %a<8>;");
        ptx.AppendLine("    .reg .b32 %b<8>;");
        ptx.AppendLine("    .reg .b64 %rd<20>;");
        ptx.AppendLine("    .reg .f32 %f<16>;");
        ptx.AppendLine("    .reg .f32 %d<8>;");
        ptx.AppendLine("    .shared .align 16 .b8 wmma_tiles[8192];");
        EmitPointers(
            ptx, "grad_output_ptr", "input_ptr", "weights_ptr", "saved_ptr",
            "grad_input_ptr", "grad_weight_ptr", "grad_bias_ptr");
        ptx.AppendLine("    mov.u32 %r0, %tid.x;");
        ptx.AppendLine("    mov.u32 %r1, %ctaid.x;");
        ptx.AppendLine("    shr.u32 %r13, %r0, 5;");
        ptx.AppendLine("    shr.u32 %r14, %r0, 7;");
        ptx.AppendLine("    and.b32 %r5, %r0, 127;");
        ptx.AppendLine("    and.b32 %r15, %r13, 3;");
        ptx.AppendLine("    shr.u32 %r16, %r15, 1;");
        ptx.AppendLine("    shl.b32 %r16, %r16, 4;");
        ptx.AppendLine("    and.b32 %r17, %r15, 1;");
        ptx.AppendLine("    shl.b32 %r17, %r17, 4;");
        ptx.AppendLine("    add.u32 %r18, %r14, 1;");
        ptx.AppendLine("    setp.lt.u32 %p0, %r1, 16;");
        ptx.AppendLine("    @%p0 bra.uni EXACT_SPLIT_DINPUT_SETUP;");

        ptx.AppendLine("    sub.u32 %r2, %r1, 16;");
        ptx.AppendLine("    shr.u32 %r3, %r2, 3;");
        ptx.AppendLine("    and.b32 %r4, %r2, 7;");
        ptx.AppendLine("    shl.b32 %r3, %r3, 5;");
        ptx.AppendLine("    shl.b32 %r4, %r4, 5;");
        ptx.AppendLine("    setp.eq.u32 %p7, %r3, 0;");
        EmitExactWmmaSharedBaseAndZero(ptx);
        ptx.AppendLine("    mov.f32 %f12, 0f00000000;");
        ptx.AppendLine("    mov.f32 %f13, 0f00000000;");
        ptx.AppendLine("    mov.f32 %f14, 0f00000000;");
        ptx.AppendLine("    mov.f32 %f15, 0f00000000;");
        ptx.AppendLine("    shl.b32 %r8, %r14, 5;");
        ptx.AppendLine("    add.u32 %r19, %r8, 32;");
        ptx.AppendLine("EXACT_SPLIT_DWEIGHT_PANEL:");
        ptx.AppendLine("    setp.ge.u32 %p0, %r8, %r19;");
        ptx.AppendLine("    @%p0 bra.uni EXACT_SPLIT_DWEIGHT_PARTIAL;");
        ptx.AppendLine("    shl.b32 %r9, %r5, 2;");
        ptx.AppendLine("    shr.u32 %r6, %r9, 5;");
        ptx.AppendLine("    and.b32 %r7, %r9, 31;");
        ptx.AppendLine("    add.u32 %r10, %r8, %r6;");
        ptx.AppendLine("    add.u32 %r11, %r3, %r7;");
        ptx.AppendLine("    mad.lo.u32 %r12, %r10, 256, %r11;");
        EmitExactWmmaGlobalAndSharedPointers(ptx, "%rd1", sharedOffset: 0);
        EmitExactWmmaPackStore(ptx, masked: false);
        ptx.AppendLine("    add.u32 %r11, %r4, %r7;");
        ptx.AppendLine("    mad.lo.u32 %r12, %r10, 256, %r11;");
        EmitExactWmmaGlobalAndSharedPointers(ptx, "%rd0", sharedOffset: 1024);
        ptx.AppendLine("    add.u64 %rd11, %rd3, %rd9;");
        EmitExactWmmaPackStore(ptx, masked: true);
        for (int lane = 0; lane < 4; lane++)
            ptx.AppendLine($"    @%p7 add.rn.f32 %f{lane + 12}, %f{lane + 12}, %f{lane + 8};");
        ptx.AppendLine("    bar.sync %r18, 128;");
        ptx.AppendLine("    mul.wide.u32 %rd13, %r16, 2;");
        ptx.AppendLine("    add.u64 %rd13, %rd7, %rd13;");
        ptx.AppendLine("    mul.wide.u32 %rd14, %r17, 2;");
        ptx.AppendLine("    add.u64 %rd14, %rd7, %rd14;");
        ptx.AppendLine("    add.u64 %rd14, %rd14, 1024;");
        ptx.AppendLine("    wmma.load.a.sync.aligned.col.m16n16k16.shared.f16 {%a0,%a1,%a2,%a3,%a4,%a5,%a6,%a7}, [%rd13], 32;");
        ptx.AppendLine("    wmma.load.b.sync.aligned.row.m16n16k16.shared.f16 {%b0,%b1,%b2,%b3,%b4,%b5,%b6,%b7}, [%rd14], 32;");
        ptx.AppendLine("    wmma.mma.sync.aligned.col.row.m16n16k16.f32.f32 {%d0,%d1,%d2,%d3,%d4,%d5,%d6,%d7}, {%a0,%a1,%a2,%a3,%a4,%a5,%a6,%a7}, {%b0,%b1,%b2,%b3,%b4,%b5,%b6,%b7}, {%d0,%d1,%d2,%d3,%d4,%d5,%d6,%d7};");
        ptx.AppendLine("    bar.sync %r18, 128;");
        ptx.AppendLine("    add.u32 %r8, %r8, 16;");
        ptx.AppendLine("    bra.uni EXACT_SPLIT_DWEIGHT_PANEL;");
        ptx.AppendLine("EXACT_SPLIT_DWEIGHT_PARTIAL:");
        EmitExactWmmaPartialStore(ptx);
        ptx.AppendLine("    bar.sync 0;");
        EmitExactWmmaMergedGlobalStore(ptx, "%rd5");
        ptx.AppendLine("    @!%p7 ret;");
        ptx.AppendLine("    mov.u64 %rd12, wmma_tiles;");
        ptx.AppendLine("    mul.wide.u32 %rd13, %r14, 2048;");
        ptx.AppendLine("    add.u64 %rd12, %rd12, %rd13;");
        ptx.AppendLine("    shl.b32 %r9, %r5, 4;");
        ptx.AppendLine("    cvt.u64.u32 %rd13, %r9;");
        ptx.AppendLine("    add.u64 %rd12, %rd12, %rd13;");
        ptx.AppendLine("    st.shared.v4.f32 [%rd12], {%f12,%f13,%f14,%f15};");
        ptx.AppendLine("    bar.sync 0;");
        ptx.AppendLine("    setp.ge.u32 %p6, %r0, 8;");
        ptx.AppendLine("    @%p6 ret;");
        ptx.AppendLine("    shl.b32 %r9, %r0, 4;");
        ptx.AppendLine("    cvt.u64.u32 %rd12, %r9;");
        ptx.AppendLine("    mov.u64 %rd13, wmma_tiles;");
        ptx.AppendLine("    add.u64 %rd12, %rd13, %rd12;");
        ptx.AppendLine("    mov.f32 %f0, 0f00000000;");
        ptx.AppendLine("    mov.f32 %f1, 0f00000000;");
        ptx.AppendLine("    mov.f32 %f2, 0f00000000;");
        ptx.AppendLine("    mov.f32 %f3, 0f00000000;");
        for (int row = 0; row < 16; row++)
        {
            int offset = row * 128;
            string suffix = offset == 0 ? string.Empty : $"+{offset}";
            ptx.AppendLine($"    ld.shared.v4.f32 {{%f4,%f5,%f6,%f7}}, [%rd12{suffix}];");
            ptx.AppendLine($"    ld.shared.v4.f32 {{%f8,%f9,%f10,%f11}}, [%rd12+{offset + 2048}];");
            for (int lane = 0; lane < 4; lane++)
            {
                ptx.AppendLine($"    add.rn.f32 %f{lane}, %f{lane}, %f{lane + 4};");
                ptx.AppendLine($"    add.rn.f32 %f{lane}, %f{lane}, %f{lane + 8};");
            }
        }
        ptx.AppendLine("    shl.b32 %r9, %r0, 2;");
        ptx.AppendLine("    add.u32 %r10, %r4, %r9;");
        ptx.AppendLine("    mul.wide.u32 %rd9, %r10, 4;");
        ptx.AppendLine("    add.u64 %rd10, %rd6, %rd9;");
        ptx.AppendLine("    st.global.v4.f32 [%rd10], {%f0,%f1,%f2,%f3};");
        ptx.AppendLine("    ret;");

        ptx.AppendLine("EXACT_SPLIT_DINPUT_SETUP:");
        ptx.AppendLine("    mov.u32 %r2, %r1;");
        ptx.AppendLine("    shr.u32 %r3, %r2, 3;");
        ptx.AppendLine("    and.b32 %r4, %r2, 7;");
        ptx.AppendLine("    shl.b32 %r3, %r3, 5;");
        ptx.AppendLine("    shl.b32 %r4, %r4, 5;");
        EmitExactWmmaSharedBaseAndZero(ptx);
        ptx.AppendLine("    shl.b32 %r8, %r14, 7;");
        ptx.AppendLine("    add.u32 %r19, %r8, 128;");
        ptx.AppendLine("EXACT_SPLIT_DINPUT_PANEL:");
        ptx.AppendLine("    setp.ge.u32 %p0, %r8, %r19;");
        ptx.AppendLine("    @%p0 bra.uni EXACT_SPLIT_DINPUT_PARTIAL;");
        ptx.AppendLine("    shl.b32 %r9, %r5, 2;");
        ptx.AppendLine("    shr.u32 %r6, %r9, 4;");
        ptx.AppendLine("    and.b32 %r7, %r9, 15;");
        ptx.AppendLine("    add.u32 %r10, %r3, %r6;");
        ptx.AppendLine("    add.u32 %r11, %r8, %r7;");
        ptx.AppendLine("    mad.lo.u32 %r12, %r10, 256, %r11;");
        EmitExactWmmaGlobalAndSharedPointers(ptx, "%rd0", sharedOffset: 0);
        ptx.AppendLine("    add.u64 %rd11, %rd3, %rd9;");
        EmitExactWmmaPackStore(ptx, masked: true);
        ptx.AppendLine("    add.u32 %r10, %r4, %r6;");
        ptx.AppendLine("    mad.lo.u32 %r12, %r10, 256, %r11;");
        EmitExactWmmaGlobalAndSharedPointers(ptx, "%rd2", sharedOffset: 1024);
        EmitExactWmmaPackStore(ptx, masked: false);
        ptx.AppendLine("    bar.sync %r18, 128;");
        ptx.AppendLine("    mul.wide.u32 %rd13, %r16, 32;");
        ptx.AppendLine("    add.u64 %rd13, %rd7, %rd13;");
        ptx.AppendLine("    mul.wide.u32 %rd14, %r17, 32;");
        ptx.AppendLine("    add.u64 %rd14, %rd7, %rd14;");
        ptx.AppendLine("    add.u64 %rd14, %rd14, 1024;");
        ptx.AppendLine("    wmma.load.a.sync.aligned.row.m16n16k16.shared.f16 {%a0,%a1,%a2,%a3,%a4,%a5,%a6,%a7}, [%rd13], 16;");
        ptx.AppendLine("    wmma.load.b.sync.aligned.col.m16n16k16.shared.f16 {%b0,%b1,%b2,%b3,%b4,%b5,%b6,%b7}, [%rd14], 16;");
        ptx.AppendLine("    wmma.mma.sync.aligned.row.col.m16n16k16.f32.f32 {%d0,%d1,%d2,%d3,%d4,%d5,%d6,%d7}, {%a0,%a1,%a2,%a3,%a4,%a5,%a6,%a7}, {%b0,%b1,%b2,%b3,%b4,%b5,%b6,%b7}, {%d0,%d1,%d2,%d3,%d4,%d5,%d6,%d7};");
        ptx.AppendLine("    bar.sync %r18, 128;");
        ptx.AppendLine("    add.u32 %r8, %r8, 16;");
        ptx.AppendLine("    bra.uni EXACT_SPLIT_DINPUT_PANEL;");
        ptx.AppendLine("EXACT_SPLIT_DINPUT_PARTIAL:");
        EmitExactWmmaPartialStore(ptx);
        ptx.AppendLine("    bar.sync 0;");
        EmitExactWmmaMergedGlobalStore(ptx, "%rd4");
        ptx.AppendLine("    ret;");
        ptx.AppendLine("}");
    }

    private static void EmitExactWmmaPartialStore(StringBuilder ptx)
    {
        ptx.AppendLine("    mad.lo.u32 %r9, %r16, 32, %r17;");
        ptx.AppendLine("    mul.wide.u32 %rd12, %r9, 4;");
        ptx.AppendLine("    add.u64 %rd12, %rd7, %rd12;");
        ptx.AppendLine("    wmma.store.d.sync.aligned.row.m16n16k16.shared.f32 [%rd12], {%d0,%d1,%d2,%d3,%d4,%d5,%d6,%d7}, 32;");
    }

    private static void EmitExactWmmaMergedGlobalStore(
        StringBuilder ptx,
        string outputBase)
    {
        ptx.AppendLine("    shl.b32 %r9, %r0, 2;");
        ptx.AppendLine("    shr.u32 %r6, %r9, 5;");
        ptx.AppendLine("    and.b32 %r7, %r9, 31;");
        ptx.AppendLine("    add.u32 %r10, %r3, %r6;");
        ptx.AppendLine("    add.u32 %r11, %r4, %r7;");
        ptx.AppendLine("    mad.lo.u32 %r12, %r10, 256, %r11;");
        ptx.AppendLine("    mul.wide.u32 %rd9, %r12, 4;");
        ptx.AppendLine($"    add.u64 %rd10, {outputBase}, %rd9;");
        ptx.AppendLine("    mul.wide.u32 %rd12, %r9, 4;");
        ptx.AppendLine("    mov.u64 %rd13, wmma_tiles;");
        ptx.AppendLine("    add.u64 %rd12, %rd13, %rd12;");
        ptx.AppendLine("    ld.shared.v4.f32 {%f0,%f1,%f2,%f3}, [%rd12];");
        ptx.AppendLine("    ld.shared.v4.f32 {%f4,%f5,%f6,%f7}, [%rd12+4096];");
        for (int lane = 0; lane < 4; lane++)
            ptx.AppendLine($"    add.rn.f32 %f{lane}, %f{lane}, %f{lane + 4};");
        ptx.AppendLine("    st.global.v4.f32 [%rd10], {%f0,%f1,%f2,%f3};");
    }

    private static void EmitExactWmmaSharedBaseAndZero(StringBuilder ptx)
    {
        ptx.AppendLine("    mov.u64 %rd7, wmma_tiles;");
        ptx.AppendLine("    mul.wide.u32 %rd8, %r14, 4096;");
        ptx.AppendLine("    add.u64 %rd7, %rd7, %rd8;");
        for (int accumulator = 0; accumulator < 8; accumulator++)
            ptx.AppendLine($"    mov.f32 %d{accumulator}, 0f00000000;");
    }

    private static void EmitExactWmmaGlobalAndSharedPointers(
        StringBuilder ptx,
        string globalBase,
        int sharedOffset)
    {
        ptx.AppendLine("    mul.wide.u32 %rd9, %r12, 4;");
        ptx.AppendLine($"    add.u64 %rd10, {globalBase}, %rd9;");
        ptx.AppendLine("    mul.wide.u32 %rd12, %r9, 2;");
        ptx.AppendLine("    add.u64 %rd12, %rd7, %rd12;");
        if (sharedOffset != 0)
            ptx.AppendLine($"    add.u64 %rd12, %rd12, {sharedOffset};");
    }

    private static void EmitExactWmmaPackStore(StringBuilder ptx, bool masked)
    {
        ptx.AppendLine("    ld.global.v4.f32 {%f0,%f1,%f2,%f3}, [%rd10];");
        if (masked)
        {
            ptx.AppendLine("    ld.global.v4.f32 {%f4,%f5,%f6,%f7}, [%rd11];");
            for (int lane = 0; lane < 4; lane++)
            {
                ptx.AppendLine($"    setp.gt.f32 %p{lane + 2}, %f{lane + 4}, 0f00000000;");
                ptx.AppendLine($"    selp.f32 %f{lane + 8}, %f{lane}, 0f00000000, %p{lane + 2};");
                ptx.AppendLine($"    cvt.rn.f16.f32 %h{lane}, %f{lane + 8};");
            }
        }
        else
        {
            for (int lane = 0; lane < 4; lane++)
                ptx.AppendLine($"    cvt.rn.f16.f32 %h{lane}, %f{lane};");
        }
        ptx.AppendLine("    mov.b32 %r20, {%h0,%h1};");
        ptx.AppendLine("    mov.b32 %r21, {%h2,%h3};");
        ptx.AppendLine("    st.shared.u32 [%rd12], %r20;");
        ptx.AppendLine("    st.shared.u32 [%rd12+4], %r21;");
    }

    private static void EmitCombined(
        StringBuilder ptx, int m, int k, int n, DirectPtxLinearActivation activation)
    {
        int mTiles = m / Tile;
        int kTiles = k / Tile;
        int nTiles = n / Tile;
        int gradInputBlocks = checked(mTiles * kTiles);
        EmitHeader(
            ptx, EntryPoint,
            "grad_output_ptr", "input_ptr", "weights_ptr", "saved_ptr",
            "grad_input_ptr", "grad_weight_ptr", "grad_bias_ptr");
        EmitRegisters(ptx);
        ptx.AppendLine("    .shared .align 16 .b8 tile_a[1024];");
        ptx.AppendLine("    .shared .align 16 .b8 tile_b[1024];");
        EmitPointers(
            ptx, "grad_output_ptr", "input_ptr", "weights_ptr", "saved_ptr",
            "grad_input_ptr", "grad_weight_ptr", "grad_bias_ptr");
        ptx.AppendLine("    mov.u64 %rd7, tile_a;");
        ptx.AppendLine("    mov.u64 %rd8, tile_b;");
        ptx.AppendLine("    mov.u32 %r0, %tid.x;");
        ptx.AppendLine("    mov.u32 %r1, %ctaid.x;");
        ptx.AppendLine($"    setp.lt.u32 %p0, %r1, {gradInputBlocks};");
        ptx.AppendLine("    @%p0 bra.uni GRAD_INPUT_PATH;");

        // dWeight = X^T @ maskedGrad. Threads load contiguous X[M,K] and
        // maskedGrad[M,N] panels, then reuse both panels for 16 FMAs. The
        // first K tile also folds each N column into dBias.
        ptx.AppendLine($"    sub.u32 %r1, %r1, {gradInputBlocks};");
        ptx.AppendLine($"    div.u32 %r2, %r1, {nTiles};");
        ptx.AppendLine($"    rem.u32 %r3, %r1, {nTiles};");
        EmitLocalCoordinates(ptx);
        ptx.AppendLine($"    mad.lo.u32 %r6, %r2, {Tile}, %r4;");
        ptx.AppendLine($"    mad.lo.u32 %r7, %r3, {Tile}, %r5;");
        ptx.AppendLine("    setp.eq.u32 %p3, %r2, 0;");
        ptx.AppendLine("    setp.eq.u32 %p4, %r4, 0;");
        ptx.AppendLine("    and.pred %p5, %p3, %p4;");
        ptx.AppendLine("    mov.f32 %f10, 0f00000000;");
        ptx.AppendLine("    mov.f32 %f11, 0f00000000;");
        ptx.AppendLine("    mov.u32 %r8, 0;");
        ptx.AppendLine("GW_TILE_LOOP:");
        ptx.AppendLine($"    setp.ge.u32 %p1, %r8, {m};");
        ptx.AppendLine("    @%p1 bra.uni GW_STORE;");
        ptx.AppendLine("    add.u32 %r9, %r8, %r4;");
        ptx.AppendLine($"    mad.lo.u32 %r10, %r2, {Tile}, %r5;");
        ptx.AppendLine($"    mad.lo.u32 %r12, %r9, {k}, %r10;");
        ptx.AppendLine("    mul.wide.u32 %rd9, %r12, 4;");
        ptx.AppendLine("    add.u64 %rd10, %rd1, %rd9;");
        ptx.AppendLine("    ld.global.nc.f32 %f3, [%rd10];");
        ptx.AppendLine($"    mad.lo.u32 %r10, %r3, {Tile}, %r5;");
        ptx.AppendLine($"    mad.lo.u32 %r12, %r9, {n}, %r10;");
        ptx.AppendLine("    mul.wide.u32 %rd9, %r12, 4;");
        ptx.AppendLine("    add.u64 %rd10, %rd0, %rd9;");
        ptx.AppendLine("    add.u64 %rd11, %rd3, %rd9;");
        ptx.AppendLine("    ld.global.nc.f32 %f0, [%rd10];");
        ptx.AppendLine("    ld.global.nc.f32 %f1, [%rd11];");
        EmitMaskedGradient(ptx, activation);
        EmitStorePanels(ptx, "%f3", "%f2");
        ptx.AppendLine("    bar.sync 0;");
        ptx.AppendLine("    mul.wide.u32 %rd12, %r4, 4;");
        ptx.AppendLine("    add.u64 %rd12, %rd7, %rd12;");
        ptx.AppendLine("    mul.wide.u32 %rd13, %r5, 4;");
        ptx.AppendLine("    add.u64 %rd13, %rd8, %rd13;");
        for (int inner = 0; inner < Tile; inner++)
        {
            string offset = inner == 0 ? string.Empty : $"+{inner * Tile * sizeof(float)}";
            ptx.AppendLine($"    ld.shared.f32 %f0, [%rd12{offset}];");
            ptx.AppendLine($"    ld.shared.f32 %f1, [%rd13{offset}];");
            ptx.AppendLine("    fma.rn.f32 %f10, %f0, %f1, %f10;");
            ptx.AppendLine("    @%p5 add.rn.f32 %f11, %f11, %f1;");
        }
        ptx.AppendLine("    bar.sync 0;");
        ptx.AppendLine($"    add.u32 %r8, %r8, {Tile};");
        ptx.AppendLine("    bra.uni GW_TILE_LOOP;");
        ptx.AppendLine("GW_STORE:");
        ptx.AppendLine($"    mad.lo.u32 %r12, %r6, {n}, %r7;");
        ptx.AppendLine("    mul.wide.u32 %rd9, %r12, 4;");
        ptx.AppendLine("    add.u64 %rd10, %rd5, %rd9;");
        ptx.AppendLine("    st.global.f32 [%rd10], %f10;");
        ptx.AppendLine("    @!%p5 bra KERNEL_DONE;");
        ptx.AppendLine("    mul.wide.u32 %rd9, %r7, 4;");
        ptx.AppendLine("    add.u64 %rd10, %rd6, %rd9;");
        ptx.AppendLine("    st.global.f32 [%rd10], %f11;");
        ptx.AppendLine("    bra.uni KERNEL_DONE;");

        // dInput = maskedGrad @ W. The weight loader transposes a coalesced
        // W[K,N] panel into shared memory so each output thread reads one
        // contiguous shared column during the inner product.
        ptx.AppendLine("GRAD_INPUT_PATH:");
        ptx.AppendLine($"    div.u32 %r2, %r1, {kTiles};");
        ptx.AppendLine($"    rem.u32 %r3, %r1, {kTiles};");
        EmitLocalCoordinates(ptx);
        ptx.AppendLine($"    mad.lo.u32 %r6, %r2, {Tile}, %r4;");
        ptx.AppendLine($"    mad.lo.u32 %r7, %r3, {Tile}, %r5;");
        ptx.AppendLine("    mov.f32 %f10, 0f00000000;");
        ptx.AppendLine("    mov.u32 %r8, 0;");
        ptx.AppendLine("GI_TILE_LOOP:");
        ptx.AppendLine($"    setp.ge.u32 %p1, %r8, {n};");
        ptx.AppendLine("    @%p1 bra.uni GI_STORE;");
        ptx.AppendLine("    add.u32 %r9, %r8, %r5;");
        ptx.AppendLine($"    mad.lo.u32 %r12, %r6, {n}, %r9;");
        ptx.AppendLine("    mul.wide.u32 %rd9, %r12, 4;");
        ptx.AppendLine("    add.u64 %rd10, %rd0, %rd9;");
        ptx.AppendLine("    add.u64 %rd11, %rd3, %rd9;");
        ptx.AppendLine("    ld.global.nc.f32 %f0, [%rd10];");
        ptx.AppendLine("    ld.global.nc.f32 %f1, [%rd11];");
        EmitMaskedGradient(ptx, activation);
        ptx.AppendLine($"    mad.lo.u32 %r10, %r3, {Tile}, %r4;");
        ptx.AppendLine($"    mad.lo.u32 %r12, %r10, {n}, %r9;");
        ptx.AppendLine("    mul.wide.u32 %rd9, %r12, 4;");
        ptx.AppendLine("    add.u64 %rd10, %rd2, %rd9;");
        ptx.AppendLine("    ld.global.nc.f32 %f3, [%rd10];");
        ptx.AppendLine("    mul.wide.u32 %rd9, %r0, 4;");
        ptx.AppendLine("    add.u64 %rd10, %rd7, %rd9;");
        ptx.AppendLine("    st.shared.f32 [%rd10], %f2;");
        ptx.AppendLine($"    mad.lo.u32 %r12, %r5, {Tile}, %r4;");
        ptx.AppendLine("    mul.wide.u32 %rd9, %r12, 4;");
        ptx.AppendLine("    add.u64 %rd10, %rd8, %rd9;");
        ptx.AppendLine("    st.shared.f32 [%rd10], %f3;");
        ptx.AppendLine("    bar.sync 0;");
        ptx.AppendLine($"    mul.wide.u32 %rd12, %r4, {Tile * sizeof(float)};");
        ptx.AppendLine("    add.u64 %rd12, %rd7, %rd12;");
        ptx.AppendLine("    mul.wide.u32 %rd13, %r5, 4;");
        ptx.AppendLine("    add.u64 %rd13, %rd8, %rd13;");
        for (int inner = 0; inner < Tile; inner++)
        {
            string aOffset = inner == 0 ? string.Empty : $"+{inner * sizeof(float)}";
            string bOffset = inner == 0 ? string.Empty : $"+{inner * Tile * sizeof(float)}";
            ptx.AppendLine($"    ld.shared.f32 %f0, [%rd12{aOffset}];");
            ptx.AppendLine($"    ld.shared.f32 %f1, [%rd13{bOffset}];");
            ptx.AppendLine("    fma.rn.f32 %f10, %f0, %f1, %f10;");
        }
        ptx.AppendLine("    bar.sync 0;");
        ptx.AppendLine($"    add.u32 %r8, %r8, {Tile};");
        ptx.AppendLine("    bra.uni GI_TILE_LOOP;");
        ptx.AppendLine("GI_STORE:");
        ptx.AppendLine($"    mad.lo.u32 %r12, %r6, {k}, %r7;");
        ptx.AppendLine("    mul.wide.u32 %rd9, %r12, 4;");
        ptx.AppendLine("    add.u64 %rd10, %rd4, %rd9;");
        ptx.AppendLine("    st.global.f32 [%rd10], %f10;");
        EmitFooter(ptx);
    }

    private static void EmitHeader(StringBuilder ptx, string entry, params string[] parameters)
    {
        ptx.AppendLine();
        ptx.AppendLine($".visible .entry {entry}(");
        for (int i = 0; i < parameters.Length; i++)
            ptx.AppendLine($"    .param .u64 {parameters[i]}{(i + 1 == parameters.Length ? string.Empty : ",")}");
        ptx.AppendLine(")");
        ptx.AppendLine($".maxntid {BlockThreads}, 1, 1");
        ptx.AppendLine("{");
    }

    private static void EmitRegisters(StringBuilder ptx)
    {
        ptx.AppendLine("    .reg .pred %p<8>;");
        ptx.AppendLine("    .reg .b32 %r<16>;");
        ptx.AppendLine("    .reg .b64 %rd<20>;");
        ptx.AppendLine("    .reg .f32 %f<16>;");
    }

    private static void EmitPointers(StringBuilder ptx, params string[] names)
    {
        for (int i = 0; i < names.Length; i++)
            ptx.AppendLine($"    ld.param.u64 %rd{i}, [{names[i]}];");
    }

    private static void EmitLocalCoordinates(StringBuilder ptx)
    {
        ptx.AppendLine($"    div.u32 %r4, %r0, {Tile};");
        ptx.AppendLine($"    rem.u32 %r5, %r0, {Tile};");
    }

    private static void EmitStorePanels(
        StringBuilder ptx, string tileAValue, string tileBValue)
    {
        ptx.AppendLine("    mul.wide.u32 %rd9, %r0, 4;");
        ptx.AppendLine("    add.u64 %rd10, %rd7, %rd9;");
        ptx.AppendLine($"    st.shared.f32 [%rd10], {tileAValue};");
        ptx.AppendLine("    add.u64 %rd11, %rd8, %rd9;");
        ptx.AppendLine($"    st.shared.f32 [%rd11], {tileBValue};");
    }

    private static void EmitMaskedGradient(
        StringBuilder ptx, DirectPtxLinearActivation activation)
    {
        // %f0=gradOutput, %f1=saved forward value/preactivation, %f2=result.
        switch (activation)
        {
            case DirectPtxLinearActivation.Relu:
                ptx.AppendLine("    setp.gt.f32 %p2, %f1, 0f00000000;");
                ptx.AppendLine("    selp.f32 %f2, %f0, 0f00000000, %p2;");
                break;
            case DirectPtxLinearActivation.Sigmoid:
                ptx.AppendLine("    sub.rn.f32 %f4, 0f3F800000, %f1;");
                ptx.AppendLine("    mul.rn.f32 %f4, %f1, %f4;");
                ptx.AppendLine("    mul.rn.f32 %f2, %f0, %f4;");
                break;
            case DirectPtxLinearActivation.Tanh:
                ptx.AppendLine("    mul.rn.f32 %f4, %f1, %f1;");
                ptx.AppendLine("    sub.rn.f32 %f4, 0f3F800000, %f4;");
                ptx.AppendLine("    mul.rn.f32 %f2, %f0, %f4;");
                break;
            case DirectPtxLinearActivation.GeluTanh:
                ptx.AppendLine("    mul.rn.f32 %f4, %f1, %f1;");
                ptx.AppendLine("    mul.rn.f32 %f5, %f4, %f1;");
                ptx.AppendLine($"    fma.rn.f32 %f5, %f5, {Literal(0.044715f)}, %f1;");
                ptx.AppendLine($"    mul.rn.f32 %f5, %f5, {Literal(0.7978845608f)};");
                ptx.AppendLine("    tanh.approx.f32 %f6, %f5;");
                ptx.AppendLine($"    fma.rn.f32 %f8, %f4, {Literal(0.134145f)}, 0f3F800000;");
                ptx.AppendLine($"    mul.rn.f32 %f8, %f8, {Literal(0.7978845608f)};");
                ptx.AppendLine("    mul.rn.f32 %f9, %f6, %f6;");
                ptx.AppendLine("    sub.rn.f32 %f9, 0f3F800000, %f9;");
                ptx.AppendLine("    mul.rn.f32 %f8, %f8, %f9;");
                ptx.AppendLine("    add.rn.f32 %f9, 0f3F800000, %f6;");
                ptx.AppendLine("    mul.rn.f32 %f9, %f9, 0f3F000000;");
                ptx.AppendLine("    mul.rn.f32 %f8, %f1, %f8;");
                ptx.AppendLine("    fma.rn.f32 %f8, %f8, 0f3F000000, %f9;");
                ptx.AppendLine("    mul.rn.f32 %f2, %f0, %f8;");
                break;
            case DirectPtxLinearActivation.Swish:
                ptx.AppendLine("    mul.rn.f32 %f4, %f1, 0f3F000000;");
                ptx.AppendLine("    tanh.approx.f32 %f4, %f4;");
                ptx.AppendLine("    fma.rn.f32 %f4, %f4, 0f3F000000, 0f3F000000;");
                ptx.AppendLine("    sub.rn.f32 %f5, 0f3F800000, %f4;");
                ptx.AppendLine("    mul.rn.f32 %f5, %f4, %f5;");
                ptx.AppendLine("    fma.rn.f32 %f5, %f1, %f5, %f4;");
                ptx.AppendLine("    mul.rn.f32 %f2, %f0, %f5;");
                break;
            default:
                throw new ArgumentOutOfRangeException(nameof(activation));
        }
    }

    private static string Literal(float value) =>
        "0f" + PtxCompat.SingleToInt32Bits(value).ToString("X8", CultureInfo.InvariantCulture);

    private static void EmitFooter(StringBuilder ptx)
    {
        ptx.AppendLine("KERNEL_DONE:");
        ptx.AppendLine("    ret;");
        ptx.AppendLine("}");
    }

    internal static bool IsSupportedShape(int m, int k, int n) =>
        m is 64 or 128 or 256 or 512 &&
        k is 256 or 512 or 1024 or 2048 or 4096 &&
        n is 256 or 512 or 1024 or 2048 or 4096;

    internal static bool IsExactChampionshipCell(
        int m,
        int k,
        int n,
        DirectPtxLinearActivation activation) =>
        m == 64 && k == 256 && n == 256 &&
        activation == DirectPtxLinearActivation.Relu;

    private static void Validate(
        int m, int k, int n, DirectPtxLinearActivation activation)
    {
        if (!IsSupportedShape(m, k, n))
            throw new ArgumentOutOfRangeException(nameof(m));
        if (activation is not (DirectPtxLinearActivation.Relu or
            DirectPtxLinearActivation.Sigmoid or DirectPtxLinearActivation.Tanh or
            DirectPtxLinearActivation.GeluTanh or DirectPtxLinearActivation.Swish))
            throw new ArgumentOutOfRangeException(nameof(activation));
    }

    private static DirectPtxKernelBlueprint CreateBlueprint(
        DirectPtxArchitectureFamily architecture,
        int m,
        int k,
        int n,
        DirectPtxLinearActivation activation)
    {
        var input = new DirectPtxExtent(m, k);
        var output = new DirectPtxExtent(m, n);
        var weights = new DirectPtxExtent(k, n);
        return new DirectPtxKernelBlueprint(
            Operation: "fused-linear-backward",
            Version: IsExactChampionshipCell(m, k, n, activation) ? 8 : 2,
            Architecture: architecture,
            Variant: $"{(IsExactChampionshipCell(m, k, n, activation) ? "wmma32x32-splitk-two-groups" : "fp32")}-m{m}-k{k}-n{n}-{activation}".ToLowerInvariant(),
            Tensors:
            [
                new("gradOutput", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D,
                    output, output, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("input", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D,
                    input, input, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("weights", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.LinearWeightInputMajor,
                    weights, weights, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("saved", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D,
                    output, output, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("gradInput", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D,
                    input, input, 16, DirectPtxTensorAccess.Write, DirectPtxExtentMode.Exact),
                new("gradWeight", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.LinearWeightInputMajor,
                    weights, weights, 16, DirectPtxTensorAccess.Write, DirectPtxExtentMode.Exact),
                new("gradBias", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    new DirectPtxExtent(n), new DirectPtxExtent(n), 16,
                    DirectPtxTensorAccess.Write, DirectPtxExtentMode.Exact)
            ],
            ResourceBudget: new DirectPtxResourceBudget(
                MaxRegistersPerThread:
                    IsExactChampionshipCell(m, k, n, activation) ? 64 : 48,
                MaxStaticSharedBytes: IsExactChampionshipCell(m, k, n, activation)
                    ? 8_192 : 2 * Tile * Tile * sizeof(float),
                MaxLocalBytesPerThread: 0,
                MinBlocksPerMultiprocessor: 4),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["formula"] = "dX,dW,db for activation(X@W+bias)",
                ["activation"] = activation.ToString(),
                ["launch-topology"] = IsExactChampionshipCell(m, k, n, activation)
                    ? "single-grid-16-dinput-plus-64-dweight-cta-ranges-with-two-group-splitk-and-row0-dbias-owners"
                    : "single-grid-disjoint-dinput-dweight-cta-ranges",
                ["tiling"] = IsExactChampionshipCell(m, k, n, activation)
                    ? "one shared 32x32 output tile per CTA; two four-warp split-K groups merged in shared memory"
                    : "16x16-shared-panels-coalesced-global-loads",
                ["masked-gradient"] = "recomputed-in-registers-before-fp16-WMMA-staging; never materialized globally",
                ["bias-gradient"] = "row0-dweight-tiles-fold-fp32-masked-values-in-fixed shared-memory order",
                ["numerical-mode"] = IsExactChampionshipCell(m, k, n, activation)
                    ? "fp32-input-to-fp16-rn-multiplicands; fp32-WMMA-accumulation; fp32-dbias"
                    : "fp32",
                ["global-intermediates"] = "none",
                ["temporary-device-allocation"] = "none",
                ["shape-parameters"] = "none",
                ["stride-parameters"] = "none"
            });
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
