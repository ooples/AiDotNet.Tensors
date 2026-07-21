#if NET5_0_OR_GREATER
using System;
using System.Collections.Generic;
using System.Globalization;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// Deterministic FP32 D=64 FlashAttention backward. Probabilities are
/// recomputed from Q/K and the forward log-sum-exp statistic; no SxS tensor,
/// derivative-score tensor, transpose, atomic, or temporary allocation exists.
/// One warp owns one complete dQ or dK/dV row and writes every result once.
/// </summary>
internal sealed class PtxFlashAttentionBackwardD64Kernel : IDisposable
{
    internal const int HeadDimension = 64;
    internal const int WarpsPerBlock = 4;
    internal const string GradQueryEntryPoint = "aidotnet_flash_attention_backward_dq_d64";
    internal const string GradKeyValueEntryPoint = "aidotnet_flash_attention_backward_dkv_d64";

    private readonly DirectPtxModule _module;
    private readonly IntPtr _gradQueryFunction;
    private readonly IntPtr _gradKeyValueFunction;

    internal int Batch { get; }
    internal int Heads { get; }
    internal int QuerySequence { get; }
    internal int KeyValueSequence { get; }
    internal float Scale { get; }
    internal bool IsCausal { get; }
    internal string Ptx { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit GradQueryAudit { get; }
    internal DirectPtxKernelAudit GradKeyValueAudit { get; }

    internal PtxFlashAttentionBackwardD64Kernel(
        DirectPtxRuntime runtime,
        int batch,
        int heads,
        int querySequence,
        int keyValueSequence,
        float scale,
        bool isCausal)
    {
        ArgumentNullException.ThrowIfNull(runtime);
        if (runtime.ArchitectureFamily != DirectPtxArchitectureFamily.Ampere)
            throw new PlatformNotSupportedException(
                "The checked-in FlashAttention-backward specialization is validated only on Ampere.");
        ValidateShape(batch, heads, querySequence, keyValueSequence, scale);

        Batch = batch;
        Heads = heads;
        QuerySequence = querySequence;
        KeyValueSequence = keyValueSequence;
        Scale = scale;
        IsCausal = isCausal;
        Blueprint = CreateBlueprint(
            runtime.ArchitectureFamily, batch, heads, querySequence, keyValueSequence, isCausal);
        Ptx = EmitPtx(
            runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor,
            batch, heads, querySequence, keyValueSequence, scale, isCausal);
        _module = runtime.LoadModule(Ptx);
        _gradQueryFunction = _module.GetFunction(
            GradQueryEntryPoint, out DirectPtxFunctionInfo gradQueryInfo);
        _gradKeyValueFunction = _module.GetFunction(
            GradKeyValueEntryPoint, out DirectPtxFunctionInfo gradKeyValueInfo);

        int blockThreads = WarpsPerBlock * 32;
        int gradQueryBlocks = _module.GetActiveBlocksPerMultiprocessor(
            _gradQueryFunction, blockThreads);
        int gradKeyValueBlocks = _module.GetActiveBlocksPerMultiprocessor(
            _gradKeyValueFunction, blockThreads);
        Blueprint.ResourceBudget.Validate(
            GradQueryEntryPoint, gradQueryInfo, blockThreads, gradQueryBlocks);
        Blueprint.ResourceBudget.Validate(
            GradKeyValueEntryPoint, gradKeyValueInfo, blockThreads, gradKeyValueBlocks);
        GradQueryAudit = DirectPtxKernelAudit.Create(
            Blueprint, runtime.DeviceFingerprint, Ptx, gradQueryInfo,
            blockThreads, gradQueryBlocks, _module.JitInfoLog);
        GradKeyValueAudit = DirectPtxKernelAudit.Create(
            Blueprint, runtime.DeviceFingerprint, Ptx, gradKeyValueInfo,
            blockThreads, gradKeyValueBlocks, _module.JitInfoLog);
    }

    internal unsafe void Launch(
        DirectPtxTensorView gradOutput,
        DirectPtxTensorView query,
        DirectPtxTensorView key,
        DirectPtxTensorView value,
        DirectPtxTensorView output,
        DirectPtxTensorView softmaxStats,
        DirectPtxTensorView gradQuery,
        DirectPtxTensorView gradKey,
        DirectPtxTensorView gradValue)
    {
        Require(gradOutput, Blueprint.Tensors[0], nameof(gradOutput));
        Require(query, Blueprint.Tensors[1], nameof(query));
        Require(key, Blueprint.Tensors[2], nameof(key));
        Require(value, Blueprint.Tensors[3], nameof(value));
        Require(output, Blueprint.Tensors[4], nameof(output));
        Require(softmaxStats, Blueprint.Tensors[5], nameof(softmaxStats));
        Require(gradQuery, Blueprint.Tensors[6], nameof(gradQuery));
        Require(gradKey, Blueprint.Tensors[7], nameof(gradKey));
        Require(gradValue, Blueprint.Tensors[8], nameof(gradValue));
        RejectOutputAliasing(
            gradOutput, query, key, value, output, softmaxStats,
            gradQuery, gradKey, gradValue);

        IntPtr go = gradOutput.Pointer;
        IntPtr q = query.Pointer;
        IntPtr k = key.Pointer;
        IntPtr v = value.Pointer;
        IntPtr o = output.Pointer;
        IntPtr lse = softmaxStats.Pointer;
        IntPtr gk = gradKey.Pointer;
        IntPtr gv = gradValue.Pointer;
        void** gradKeyValueArguments = stackalloc void*[8];
        gradKeyValueArguments[0] = &go;
        gradKeyValueArguments[1] = &q;
        gradKeyValueArguments[2] = &k;
        gradKeyValueArguments[3] = &v;
        gradKeyValueArguments[4] = &o;
        gradKeyValueArguments[5] = &lse;
        gradKeyValueArguments[6] = &gk;
        gradKeyValueArguments[7] = &gv;
        int keyRows = checked(Batch * Heads * KeyValueSequence);
        _module.Launch(
            _gradKeyValueFunction,
            (uint)((keyRows + WarpsPerBlock - 1) / WarpsPerBlock), 1, 1,
            (uint)(WarpsPerBlock * 32), 1, 1, 0, gradKeyValueArguments);

        IntPtr gq = gradQuery.Pointer;
        void** gradQueryArguments = stackalloc void*[7];
        gradQueryArguments[0] = &go;
        gradQueryArguments[1] = &q;
        gradQueryArguments[2] = &k;
        gradQueryArguments[3] = &v;
        gradQueryArguments[4] = &o;
        gradQueryArguments[5] = &lse;
        gradQueryArguments[6] = &gq;
        int queryRows = checked(Batch * Heads * QuerySequence);
        _module.Launch(
            _gradQueryFunction,
            (uint)((queryRows + WarpsPerBlock - 1) / WarpsPerBlock), 1, 1,
            (uint)(WarpsPerBlock * 32), 1, 1, 0, gradQueryArguments);
    }

    private static void Require(
        DirectPtxTensorView view,
        DirectPtxTensorContract contract,
        string parameter)
    {
        if (view.Pointer == IntPtr.Zero || view.PhysicalType != contract.PhysicalType ||
            view.Layout != contract.Layout || view.LogicalExtent != contract.LogicalExtent ||
            view.PhysicalExtent != contract.PhysicalExtent || view.ByteLength < contract.RequiredBytes)
            throw new ArgumentException(
                $"{parameter} does not satisfy physical ABI '{contract.Name}'.", parameter);
    }

    private static void RejectOutputAliasing(
        DirectPtxTensorView gradOutput,
        DirectPtxTensorView query,
        DirectPtxTensorView key,
        DirectPtxTensorView value,
        DirectPtxTensorView output,
        DirectPtxTensorView softmaxStats,
        DirectPtxTensorView gradQuery,
        DirectPtxTensorView gradKey,
        DirectPtxTensorView gradValue)
    {
        IntPtr go = gradOutput.Pointer, q = query.Pointer, k = key.Pointer;
        IntPtr v = value.Pointer, o = output.Pointer, lse = softmaxStats.Pointer;
        IntPtr gq = gradQuery.Pointer, gk = gradKey.Pointer, gv = gradValue.Pointer;
        if (gq == go || gq == q || gq == k || gq == v || gq == o || gq == lse ||
            gq == gk || gq == gv ||
            gk == go || gk == q || gk == k || gk == v || gk == o || gk == lse ||
            gk == gv ||
            gv == go || gv == q || gv == k || gv == v || gv == o || gv == lse)
            throw new ArgumentException(
                "FlashAttention-backward outputs may not alias inputs or each other.");
    }

    public void Dispose() => _module.Dispose();

    internal static string EmitPtx(
        int ccMajor,
        int ccMinor,
        int batch,
        int heads,
        int querySequence,
        int keyValueSequence,
        float scale,
        bool isCausal)
    {
        ValidateShape(batch, heads, querySequence, keyValueSequence, scale);
        var ptx = new StringBuilder(20_000);
        ptx.AppendLine(".version 7.1");
        ptx.AppendLine($".target sm_{ccMajor}{ccMinor}");
        ptx.AppendLine(".address_size 64");
        ptx.AppendLine();
        EmitGradQuery(ptx, batch, heads, querySequence, keyValueSequence, scale, isCausal);
        ptx.AppendLine();
        EmitGradKeyValue(ptx, batch, heads, querySequence, keyValueSequence, scale, isCausal);
        return ptx.ToString();
    }

    private static void EmitGradQuery(
        StringBuilder ptx,
        int batch,
        int heads,
        int querySequence,
        int keyValueSequence,
        float scale,
        bool isCausal)
    {
        int totalRows = checked(batch * heads * querySequence);
        ptx.AppendLine($".visible .entry {GradQueryEntryPoint}(");
        EmitParameters(ptx, includeGradKeyValue: false);
        ptx.AppendLine(")");
        ptx.AppendLine("{");
        ptx.AppendLine("    .reg .pred %p<4>;");
        ptx.AppendLine("    .reg .b32 %r<32>;");
        ptx.AppendLine("    .reg .b64 %rd<32>;");
        ptx.AppendLine("    .reg .f32 %f<32>;");
        LoadCommonParameters(ptx);
        ptx.AppendLine("    ld.param.u64 %rd6, [grad_query_ptr];");
        EmitWarpRow(ptx, totalRows, querySequence, "DQ_RETURN");
        ptx.AppendLine("    shl.b32 %r7, %r2, 1;");
        ptx.AppendLine("    mul.wide.u32 %rd7, %r4, 256;");
        ptx.AppendLine("    mul.wide.u32 %rd8, %r2, 8;");
        ptx.AppendLine("    add.u64 %rd9, %rd1, %rd7;");
        ptx.AppendLine("    add.u64 %rd9, %rd9, %rd8;");
        ptx.AppendLine("    ld.global.v2.f32 {%f0,%f1}, [%rd9];");
        ptx.AppendLine("    add.u64 %rd10, %rd0, %rd7;");
        ptx.AppendLine("    add.u64 %rd10, %rd10, %rd8;");
        ptx.AppendLine("    ld.global.v2.f32 {%f2,%f3}, [%rd10];");
        ptx.AppendLine("    add.u64 %rd11, %rd4, %rd7;");
        ptx.AppendLine("    add.u64 %rd11, %rd11, %rd8;");
        ptx.AppendLine("    ld.global.v2.f32 {%f4,%f5}, [%rd11];");
        ptx.AppendLine("    mul.rn.f32 %f6, %f2, %f4;");
        ptx.AppendLine("    fma.rn.f32 %f6, %f3, %f5, %f6;");
        EmitWarpSum(ptx, "%f6", "%f7", "%r20", "%r21");
        ptx.AppendLine("    mul.wide.u32 %rd12, %r4, 4;");
        ptx.AppendLine("    add.u64 %rd12, %rd5, %rd12;");
        ptx.AppendLine("    ld.global.f32 %f8, [%rd12];");
        ptx.AppendLine("    mov.f32 %f9, 0f00000000;");
        ptx.AppendLine("    mov.f32 %f10, 0f00000000;");
        ptx.AppendLine("    mov.u32 %r8, 0;");
        ptx.AppendLine("DQ_KEY_LOOP:");
        ptx.AppendLine($"    setp.ge.u32 %p1, %r8, {keyValueSequence};");
        ptx.AppendLine("    @%p1 bra.uni DQ_KEY_DONE;");
        if (isCausal)
        {
            ptx.AppendLine("    setp.gt.u32 %p2, %r8, %r6;");
            ptx.AppendLine("    @%p2 bra.uni DQ_KEY_NEXT;");
        }
        EmitBhKeyPairAddress(ptx, "%rd2", "%rd13", "%r5", "%r8", "%r7", keyValueSequence);
        ptx.AppendLine("    ld.global.v2.f32 {%f11,%f12}, [%rd13];");
        ptx.AppendLine("    mul.rn.f32 %f13, %f0, %f11;");
        ptx.AppendLine("    fma.rn.f32 %f13, %f1, %f12, %f13;");
        EmitWarpSum(ptx, "%f13", "%f14", "%r20", "%r21");
        ptx.AppendLine($"    mul.rn.f32 %f15, %f13, {FloatLiteral(scale)};");
        ptx.AppendLine("    sub.rn.f32 %f15, %f15, %f8;");
        ptx.AppendLine($"    mul.rn.f32 %f15, %f15, {FloatLiteral(1.4426950408889634f)};");
        ptx.AppendLine("    ex2.approx.f32 %f16, %f15;");
        EmitBhKeyPairAddress(ptx, "%rd3", "%rd14", "%r5", "%r8", "%r7", keyValueSequence);
        ptx.AppendLine("    ld.global.v2.f32 {%f17,%f18}, [%rd14];");
        ptx.AppendLine("    mul.rn.f32 %f19, %f2, %f17;");
        ptx.AppendLine("    fma.rn.f32 %f19, %f3, %f18, %f19;");
        EmitWarpSum(ptx, "%f19", "%f20", "%r20", "%r21");
        ptx.AppendLine("    sub.rn.f32 %f21, %f19, %f6;");
        ptx.AppendLine("    mul.rn.f32 %f21, %f21, %f16;");
        ptx.AppendLine($"    mul.rn.f32 %f21, %f21, {FloatLiteral(scale)};");
        ptx.AppendLine("    fma.rn.f32 %f9, %f21, %f11, %f9;");
        ptx.AppendLine("    fma.rn.f32 %f10, %f21, %f12, %f10;");
        ptx.AppendLine("DQ_KEY_NEXT:");
        ptx.AppendLine("    add.u32 %r8, %r8, 1;");
        ptx.AppendLine("    bra.uni DQ_KEY_LOOP;");
        ptx.AppendLine("DQ_KEY_DONE:");
        ptx.AppendLine("    add.u64 %rd15, %rd6, %rd7;");
        ptx.AppendLine("    add.u64 %rd15, %rd15, %rd8;");
        ptx.AppendLine("    st.global.v2.f32 [%rd15], {%f9,%f10};");
        ptx.AppendLine("DQ_RETURN:");
        ptx.AppendLine("    ret;");
        ptx.AppendLine("}");
    }

    private static void EmitGradKeyValue(
        StringBuilder ptx,
        int batch,
        int heads,
        int querySequence,
        int keyValueSequence,
        float scale,
        bool isCausal)
    {
        int totalRows = checked(batch * heads * keyValueSequence);
        ptx.AppendLine($".visible .entry {GradKeyValueEntryPoint}(");
        EmitParameters(ptx, includeGradKeyValue: true);
        ptx.AppendLine(")");
        ptx.AppendLine("{");
        ptx.AppendLine("    .reg .pred %p<4>;");
        ptx.AppendLine("    .reg .b32 %r<32>;");
        ptx.AppendLine("    .reg .b64 %rd<32>;");
        ptx.AppendLine("    .reg .f32 %f<40>;");
        LoadCommonParameters(ptx);
        ptx.AppendLine("    ld.param.u64 %rd6, [grad_key_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd7, [grad_value_ptr];");
        EmitWarpRow(ptx, totalRows, keyValueSequence, "DKV_RETURN");
        ptx.AppendLine("    shl.b32 %r7, %r2, 1;");
        EmitBhKeyPairAddress(ptx, "%rd2", "%rd8", "%r5", "%r6", "%r7", keyValueSequence);
        ptx.AppendLine("    ld.global.v2.f32 {%f0,%f1}, [%rd8];");
        EmitBhKeyPairAddress(ptx, "%rd3", "%rd9", "%r5", "%r6", "%r7", keyValueSequence);
        ptx.AppendLine("    ld.global.v2.f32 {%f2,%f3}, [%rd9];");
        ptx.AppendLine("    mov.f32 %f4, 0f00000000;");
        ptx.AppendLine("    mov.f32 %f5, 0f00000000;");
        ptx.AppendLine("    mov.f32 %f6, 0f00000000;");
        ptx.AppendLine("    mov.f32 %f7, 0f00000000;");
        ptx.AppendLine("    mov.u32 %r8, 0;");
        ptx.AppendLine("DKV_QUERY_LOOP:");
        ptx.AppendLine($"    setp.ge.u32 %p1, %r8, {querySequence};");
        ptx.AppendLine("    @%p1 bra.uni DKV_QUERY_DONE;");
        if (isCausal)
        {
            ptx.AppendLine("    setp.gt.u32 %p2, %r6, %r8;");
            ptx.AppendLine("    @%p2 bra.uni DKV_QUERY_NEXT;");
        }
        ptx.AppendLine($"    mad.lo.u32 %r9, %r5, {querySequence}, %r8;");
        ptx.AppendLine("    mul.wide.u32 %rd10, %r9, 256;");
        ptx.AppendLine("    mul.wide.u32 %rd11, %r2, 8;");
        ptx.AppendLine("    add.u64 %rd12, %rd1, %rd10;");
        ptx.AppendLine("    add.u64 %rd12, %rd12, %rd11;");
        ptx.AppendLine("    ld.global.v2.f32 {%f8,%f9}, [%rd12];");
        ptx.AppendLine("    add.u64 %rd13, %rd0, %rd10;");
        ptx.AppendLine("    add.u64 %rd13, %rd13, %rd11;");
        ptx.AppendLine("    ld.global.v2.f32 {%f10,%f11}, [%rd13];");
        ptx.AppendLine("    add.u64 %rd14, %rd4, %rd10;");
        ptx.AppendLine("    add.u64 %rd14, %rd14, %rd11;");
        ptx.AppendLine("    ld.global.v2.f32 {%f12,%f13}, [%rd14];");
        ptx.AppendLine("    mul.rn.f32 %f14, %f8, %f0;");
        ptx.AppendLine("    fma.rn.f32 %f14, %f9, %f1, %f14;");
        EmitWarpSum(ptx, "%f14", "%f15", "%r20", "%r21");
        ptx.AppendLine($"    mul.rn.f32 %f16, %f14, {FloatLiteral(scale)};");
        ptx.AppendLine("    mul.wide.u32 %rd15, %r9, 4;");
        ptx.AppendLine("    add.u64 %rd15, %rd5, %rd15;");
        ptx.AppendLine("    ld.global.f32 %f17, [%rd15];");
        ptx.AppendLine("    sub.rn.f32 %f16, %f16, %f17;");
        ptx.AppendLine($"    mul.rn.f32 %f16, %f16, {FloatLiteral(1.4426950408889634f)};");
        ptx.AppendLine("    ex2.approx.f32 %f18, %f16;");
        ptx.AppendLine("    mul.rn.f32 %f19, %f10, %f12;");
        ptx.AppendLine("    fma.rn.f32 %f19, %f11, %f13, %f19;");
        EmitWarpSum(ptx, "%f19", "%f20", "%r20", "%r21");
        ptx.AppendLine("    mul.rn.f32 %f21, %f10, %f2;");
        ptx.AppendLine("    fma.rn.f32 %f21, %f11, %f3, %f21;");
        EmitWarpSum(ptx, "%f21", "%f22", "%r20", "%r21");
        ptx.AppendLine("    sub.rn.f32 %f23, %f21, %f19;");
        ptx.AppendLine("    mul.rn.f32 %f23, %f23, %f18;");
        ptx.AppendLine($"    mul.rn.f32 %f23, %f23, {FloatLiteral(scale)};");
        ptx.AppendLine("    fma.rn.f32 %f4, %f23, %f8, %f4;");
        ptx.AppendLine("    fma.rn.f32 %f5, %f23, %f9, %f5;");
        ptx.AppendLine("    fma.rn.f32 %f6, %f18, %f10, %f6;");
        ptx.AppendLine("    fma.rn.f32 %f7, %f18, %f11, %f7;");
        ptx.AppendLine("DKV_QUERY_NEXT:");
        ptx.AppendLine("    add.u32 %r8, %r8, 1;");
        ptx.AppendLine("    bra.uni DKV_QUERY_LOOP;");
        ptx.AppendLine("DKV_QUERY_DONE:");
        EmitBhKeyPairAddress(ptx, "%rd6", "%rd16", "%r5", "%r6", "%r7", keyValueSequence);
        ptx.AppendLine("    st.global.v2.f32 [%rd16], {%f4,%f5};");
        EmitBhKeyPairAddress(ptx, "%rd7", "%rd17", "%r5", "%r6", "%r7", keyValueSequence);
        ptx.AppendLine("    st.global.v2.f32 [%rd17], {%f6,%f7};");
        ptx.AppendLine("DKV_RETURN:");
        ptx.AppendLine("    ret;");
        ptx.AppendLine("}");
    }

    private static void EmitParameters(StringBuilder ptx, bool includeGradKeyValue)
    {
        ptx.AppendLine("    .param .u64 grad_output_ptr,");
        ptx.AppendLine("    .param .u64 query_ptr,");
        ptx.AppendLine("    .param .u64 key_ptr,");
        ptx.AppendLine("    .param .u64 value_ptr,");
        ptx.AppendLine("    .param .u64 output_ptr,");
        ptx.AppendLine("    .param .u64 softmax_stats_ptr,");
        if (includeGradKeyValue)
        {
            ptx.AppendLine("    .param .u64 grad_key_ptr,");
            ptx.AppendLine("    .param .u64 grad_value_ptr");
        }
        else
        {
            ptx.AppendLine("    .param .u64 grad_query_ptr");
        }
    }

    private static void LoadCommonParameters(StringBuilder ptx)
    {
        ptx.AppendLine("    ld.param.u64 %rd0, [grad_output_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd1, [query_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd2, [key_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd3, [value_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd4, [output_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd5, [softmax_stats_ptr];");
    }

    private static void EmitWarpRow(
        StringBuilder ptx,
        int totalRows,
        int sequence,
        string returnLabel)
    {
        ptx.AppendLine("    mov.u32 %r0, %tid.x;");
        ptx.AppendLine("    shr.u32 %r1, %r0, 5;");
        ptx.AppendLine("    and.b32 %r2, %r0, 31;");
        ptx.AppendLine("    mov.u32 %r3, %ctaid.x;");
        ptx.AppendLine($"    mad.lo.u32 %r4, %r3, {WarpsPerBlock}, %r1;");
        ptx.AppendLine($"    setp.ge.u32 %p0, %r4, {totalRows};");
        ptx.AppendLine($"    @%p0 bra.uni {returnLabel};");
        ptx.AppendLine($"    div.u32 %r5, %r4, {sequence};");
        ptx.AppendLine($"    rem.u32 %r6, %r4, {sequence};");
    }

    private static void EmitBhKeyPairAddress(
        StringBuilder ptx,
        string basePointer,
        string destination,
        string batchHead,
        string keyIndex,
        string dimension,
        int keyValueSequence)
    {
        ptx.AppendLine($"    mad.lo.u32 %r13, {batchHead}, {keyValueSequence}, {keyIndex};");
        ptx.AppendLine($"    mad.lo.u32 %r13, %r13, {HeadDimension}, {dimension};");
        ptx.AppendLine("    mul.wide.u32 %rd25, %r13, 4;");
        ptx.AppendLine($"    add.u64 {destination}, {basePointer}, %rd25;");
    }

    private static void EmitWarpSum(
        StringBuilder ptx,
        string value,
        string temporaryFloat,
        string bits,
        string shuffledBits)
    {
        foreach (int delta in new[] { 16, 8, 4, 2, 1 })
        {
            ptx.AppendLine($"    mov.b32 {bits}, {value};");
            ptx.AppendLine($"    shfl.sync.bfly.b32 {shuffledBits}, {bits}, {delta}, 31, 0xffffffff;");
            ptx.AppendLine($"    mov.b32 {temporaryFloat}, {shuffledBits};");
            ptx.AppendLine($"    add.rn.f32 {value}, {value}, {temporaryFloat};");
        }
    }

    private static DirectPtxKernelBlueprint CreateBlueprint(
        DirectPtxArchitectureFamily architecture,
        int batch,
        int heads,
        int querySequence,
        int keyValueSequence,
        bool isCausal)
    {
        var query = new DirectPtxExtent(batch, heads, querySequence, HeadDimension);
        var keyValue = new DirectPtxExtent(batch, heads, keyValueSequence, HeadDimension);
        var stats = new DirectPtxExtent(batch * heads, querySequence);
        return new DirectPtxKernelBlueprint(
            Operation: "flash-attention-recompute-backward-d64",
            Version: 1,
            Architecture: architecture,
            Variant: isCausal ? "deterministic-causal-top-left" : "deterministic-unmasked",
            Tensors:
            [
                new("grad-output", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Bhsd,
                    query, query, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("query", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Bhsd,
                    query, query, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("key", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Bhsd,
                    keyValue, keyValue, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("value", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Bhsd,
                    keyValue, keyValue, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("output", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Bhsd,
                    query, query, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("softmax-stats", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D,
                    stats, stats, 4, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("grad-query", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Bhsd,
                    query, query, 16, DirectPtxTensorAccess.Write, DirectPtxExtentMode.Exact),
                new("grad-key", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Bhsd,
                    keyValue, keyValue, 16, DirectPtxTensorAccess.Write, DirectPtxExtentMode.Exact),
                new("grad-value", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Bhsd,
                    keyValue, keyValue, 16, DirectPtxTensorAccess.Write, DirectPtxExtentMode.Exact)
            ],
            ResourceBudget: new DirectPtxResourceBudget(
                MaxRegistersPerThread: 64,
                MaxStaticSharedBytes: 0,
                MaxLocalBytesPerThread: 0,
                MinBlocksPerMultiprocessor: 4),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["input"] = "fp32-q-k-v-output-lse",
                ["probability"] = "recomputed-exp(score-lse)",
                ["accumulator"] = "fp32",
                ["output"] = "fp32-dq-dk-dv",
                ["layout"] = "canonical-bhsd",
                ["mask"] = isCausal ? "causal-top-left-baked" : "none",
                ["determinism"] = "fixed-order-no-atomics",
                ["global-intermediates"] = "none",
                ["temporary-device-allocation"] = "none",
                ["shape"] = $"b{batch}-h{heads}-sq{querySequence}-sk{keyValueSequence}-d64"
            });
    }

    private static void ValidateShape(
        int batch,
        int heads,
        int querySequence,
        int keyValueSequence,
        float scale)
    {
        if (batch <= 0 || batch > 16) throw new ArgumentOutOfRangeException(nameof(batch));
        if (heads <= 0 || heads > 128) throw new ArgumentOutOfRangeException(nameof(heads));
        if (querySequence is not (16 or 32 or 64 or 128))
            throw new ArgumentOutOfRangeException(nameof(querySequence));
        if (keyValueSequence is not (16 or 32 or 64 or 128))
            throw new ArgumentOutOfRangeException(nameof(keyValueSequence));
        if (!float.IsFinite(scale)) throw new ArgumentOutOfRangeException(nameof(scale));
    }

    private static string FloatLiteral(float value) =>
        "0f" + BitConverter.SingleToInt32Bits(value).ToString("X8", CultureInfo.InvariantCulture);
}
#endif
