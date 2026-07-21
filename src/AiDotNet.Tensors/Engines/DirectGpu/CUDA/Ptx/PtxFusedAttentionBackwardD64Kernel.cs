#if NET5_0_OR_GREATER
using System;
using System.Collections.Generic;
using System.Globalization;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// Deterministic FP32 D=64 backward for attention APIs that already expose the
/// softmax probabilities. A warp first writes each row delta into the final dQ
/// allocation as transient workspace. Independent key warps consume that tiny
/// vector to write dK/dV, then query warps overwrite the workspace with dQ.
/// No additional device allocation, atomics, or SxS gradient is required.
/// </summary>
internal sealed class PtxFusedAttentionBackwardD64Kernel : IDisposable
{
    internal const int HeadDimension = 64;
    internal const int WarpsPerBlock = 4;
    internal const string RowDeltaEntryPoint = "aidotnet_attention_backward_delta_d64";
    internal const string GradQueryEntryPoint = "aidotnet_attention_backward_dq_d64";
    internal const string GradKeyValueEntryPoint = "aidotnet_attention_backward_dkv_d64";

    private readonly DirectPtxModule _module;
    private readonly IntPtr _rowDeltaFunction;
    private readonly IntPtr _gradQueryFunction;
    private readonly IntPtr _gradKeyValueFunction;

    internal int Batch { get; }
    internal int QueryHeads { get; }
    internal int KeyValueHeads { get; }
    internal int QuerySequence { get; }
    internal int KeyValueSequence { get; }
    internal int QueriesPerKeyValueHead { get; }
    internal float Scale { get; }
    internal string Ptx { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxFunctionInfo GradQueryFunctionInfo { get; }
    internal DirectPtxFunctionInfo GradKeyValueFunctionInfo { get; }
    internal DirectPtxFunctionInfo RowDeltaFunctionInfo { get; }
    internal DirectPtxKernelAudit GradQueryAudit { get; }
    internal DirectPtxKernelAudit GradKeyValueAudit { get; }
    internal DirectPtxKernelAudit RowDeltaAudit { get; }

    internal PtxFusedAttentionBackwardD64Kernel(
        DirectPtxRuntime runtime,
        int batch,
        int queryHeads,
        int keyValueHeads,
        int querySequence,
        int keyValueSequence,
        float scale)
    {
        ArgumentNullException.ThrowIfNull(runtime);
        if (runtime.ArchitectureFamily != DirectPtxArchitectureFamily.Ampere)
            throw new PlatformNotSupportedException(
                "The checked-in attention-backward specialization is validated only on Ampere.");
        ValidateShape(batch, queryHeads, keyValueHeads, querySequence, keyValueSequence, scale);

        Batch = batch;
        QueryHeads = queryHeads;
        KeyValueHeads = keyValueHeads;
        QuerySequence = querySequence;
        KeyValueSequence = keyValueSequence;
        QueriesPerKeyValueHead = queryHeads / keyValueHeads;
        Scale = scale;
        Blueprint = CreateBlueprint(
            runtime.ArchitectureFamily, batch, queryHeads, keyValueHeads,
            querySequence, keyValueSequence);
        Ptx = EmitPtx(
            runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor,
            batch, queryHeads, keyValueHeads, querySequence, keyValueSequence, scale);
        _module = runtime.LoadModule(Ptx);
        _rowDeltaFunction = _module.GetFunction(
            RowDeltaEntryPoint, out DirectPtxFunctionInfo rowDeltaInfo);
        _gradQueryFunction = _module.GetFunction(
            GradQueryEntryPoint, out DirectPtxFunctionInfo gradQueryInfo);
        _gradKeyValueFunction = _module.GetFunction(
            GradKeyValueEntryPoint, out DirectPtxFunctionInfo gradKeyValueInfo);
        GradQueryFunctionInfo = gradQueryInfo;
        GradKeyValueFunctionInfo = gradKeyValueInfo;
        RowDeltaFunctionInfo = rowDeltaInfo;

        int blockThreads = WarpsPerBlock * 32;
        int gradQueryBlocks = _module.GetActiveBlocksPerMultiprocessor(
            _gradQueryFunction, blockThreads);
        int rowDeltaBlocks = _module.GetActiveBlocksPerMultiprocessor(
            _rowDeltaFunction, blockThreads);
        int gradKeyValueBlocks = _module.GetActiveBlocksPerMultiprocessor(
            _gradKeyValueFunction, blockThreads);
        Blueprint.ResourceBudget.Validate(
            RowDeltaEntryPoint, rowDeltaInfo, blockThreads, rowDeltaBlocks);
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
        RowDeltaAudit = DirectPtxKernelAudit.Create(
            Blueprint, runtime.DeviceFingerprint, Ptx, rowDeltaInfo,
            blockThreads, rowDeltaBlocks, _module.JitInfoLog);
    }

    internal unsafe void Launch(
        DirectPtxTensorView gradOutput,
        DirectPtxTensorView query,
        DirectPtxTensorView key,
        DirectPtxTensorView value,
        DirectPtxTensorView probabilities,
        DirectPtxTensorView gradQuery,
        DirectPtxTensorView gradKey,
        DirectPtxTensorView gradValue)
    {
        Require(gradOutput, Blueprint.Tensors[0], nameof(gradOutput));
        Require(query, Blueprint.Tensors[1], nameof(query));
        Require(key, Blueprint.Tensors[2], nameof(key));
        Require(value, Blueprint.Tensors[3], nameof(value));
        Require(probabilities, Blueprint.Tensors[4], nameof(probabilities));
        Require(gradQuery, Blueprint.Tensors[5], nameof(gradQuery));
        Require(gradKey, Blueprint.Tensors[6], nameof(gradKey));
        Require(gradValue, Blueprint.Tensors[7], nameof(gradValue));
        RejectOutputAliasing(
            gradOutput, query, key, value, probabilities, gradQuery, gradKey, gradValue);

        IntPtr go = gradOutput.Pointer;
        IntPtr k = key.Pointer;
        IntPtr v = value.Pointer;
        IntPtr p = probabilities.Pointer;
        IntPtr gq = gradQuery.Pointer;
        int queryRows = checked(Batch * QueryHeads * QuerySequence);
        void** rowDeltaArguments = stackalloc void*[4];
        rowDeltaArguments[0] = &go;
        rowDeltaArguments[1] = &v;
        rowDeltaArguments[2] = &p;
        rowDeltaArguments[3] = &gq;
        _module.Launch(
            _rowDeltaFunction, (uint)(queryRows / WarpsPerBlock), 1, 1,
            (uint)(WarpsPerBlock * 32), 1, 1, 0, rowDeltaArguments);

        IntPtr q = query.Pointer;
        IntPtr gk = gradKey.Pointer;
        IntPtr gv = gradValue.Pointer;
        void** gradKeyValueArguments = stackalloc void*[7];
        gradKeyValueArguments[0] = &go;
        gradKeyValueArguments[1] = &q;
        gradKeyValueArguments[2] = &v;
        gradKeyValueArguments[3] = &p;
        gradKeyValueArguments[4] = &gq;
        gradKeyValueArguments[5] = &gk;
        gradKeyValueArguments[6] = &gv;
        int keyRows = checked(Batch * KeyValueHeads * KeyValueSequence);
        _module.Launch(
            _gradKeyValueFunction, (uint)(keyRows / WarpsPerBlock), 1, 1,
            (uint)(WarpsPerBlock * 32), 1, 1, 0, gradKeyValueArguments);

        void** gradQueryArguments = stackalloc void*[5];
        gradQueryArguments[0] = &go;
        gradQueryArguments[1] = &k;
        gradQueryArguments[2] = &v;
        gradQueryArguments[3] = &p;
        gradQueryArguments[4] = &gq;
        _module.Launch(
            _gradQueryFunction, (uint)(queryRows / WarpsPerBlock), 1, 1,
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
        DirectPtxTensorView probabilities,
        DirectPtxTensorView gradQuery,
        DirectPtxTensorView gradKey,
        DirectPtxTensorView gradValue)
    {
        IntPtr gq = gradQuery.Pointer, gk = gradKey.Pointer, gv = gradValue.Pointer;
        if (gq == gradOutput.Pointer || gq == query.Pointer || gq == key.Pointer ||
            gq == value.Pointer || gq == probabilities.Pointer || gq == gk || gq == gv ||
            gk == gradOutput.Pointer || gk == query.Pointer || gk == key.Pointer ||
            gk == value.Pointer || gk == probabilities.Pointer || gk == gv ||
            gv == gradOutput.Pointer || gv == query.Pointer || gv == key.Pointer ||
            gv == value.Pointer || gv == probabilities.Pointer)
            throw new ArgumentException(
                "Attention-backward outputs may not alias inputs or each other.");
    }

    public void Dispose() => _module.Dispose();

    internal static string EmitPtx(
        int ccMajor,
        int ccMinor,
        int batch,
        int queryHeads,
        int keyValueHeads,
        int querySequence,
        int keyValueSequence,
        float scale)
    {
        ValidateShape(batch, queryHeads, keyValueHeads, querySequence, keyValueSequence, scale);
        int queriesPerKeyValueHead = queryHeads / keyValueHeads;
        var ptx = new StringBuilder(24_000);
        ptx.AppendLine(".version 7.1");
        ptx.AppendLine($".target sm_{ccMajor}{ccMinor}");
        ptx.AppendLine(".address_size 64");
        ptx.AppendLine();
        EmitRowDelta(
            ptx, queryHeads, keyValueHeads, querySequence,
            keyValueSequence, queriesPerKeyValueHead);
        ptx.AppendLine();
        EmitGradQuery(
            ptx, queryHeads, keyValueHeads, querySequence,
            keyValueSequence, queriesPerKeyValueHead, scale);
        ptx.AppendLine();
        EmitParallelGradKeyValue(
            ptx, queryHeads, keyValueHeads, querySequence,
            keyValueSequence, queriesPerKeyValueHead, scale);
        return ptx.ToString();
    }

    private static void EmitRowDelta(
        StringBuilder ptx,
        int queryHeads,
        int keyValueHeads,
        int querySequence,
        int keyValueSequence,
        int queriesPerKeyValueHead)
    {
        ptx.AppendLine($".visible .entry {RowDeltaEntryPoint}(");
        ptx.AppendLine("    .param .u64 grad_output_ptr,");
        ptx.AppendLine("    .param .u64 value_ptr,");
        ptx.AppendLine("    .param .u64 probabilities_ptr,");
        ptx.AppendLine("    .param .u64 grad_query_workspace_ptr");
        ptx.AppendLine(")");
        ptx.AppendLine("{");
        ptx.AppendLine("    .reg .pred %p<2>;");
        ptx.AppendLine("    .reg .b32 %r<24>;");
        ptx.AppendLine("    .reg .b64 %rd<32>;");
        ptx.AppendLine("    .reg .f32 %f<16>;");
        ptx.AppendLine("    ld.param.u64 %rd0, [grad_output_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd1, [value_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd2, [probabilities_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd3, [grad_query_workspace_ptr];");
        ptx.AppendLine("    mov.u32 %r0, %tid.x;");
        ptx.AppendLine("    shr.u32 %r1, %r0, 5;");
        ptx.AppendLine("    and.b32 %r2, %r0, 31;");
        ptx.AppendLine("    mov.u32 %r3, %ctaid.x;");
        ptx.AppendLine($"    mad.lo.u32 %r4, %r3, {WarpsPerBlock}, %r1;");
        ptx.AppendLine($"    div.u32 %r5, %r4, {querySequence};");
        ptx.AppendLine($"    rem.u32 %r6, %r5, {queryHeads};");
        ptx.AppendLine($"    div.u32 %r7, %r5, {queryHeads};");
        ptx.AppendLine($"    div.u32 %r8, %r6, {queriesPerKeyValueHead};");
        ptx.AppendLine($"    mad.lo.u32 %r9, %r7, {keyValueHeads}, %r8;");
        ptx.AppendLine("    shl.b32 %r10, %r2, 1;");
        ptx.AppendLine("    mul.wide.u32 %rd4, %r4, 256;");
        ptx.AppendLine("    mul.wide.u32 %rd5, %r2, 8;");
        ptx.AppendLine("    add.u64 %rd6, %rd0, %rd4;");
        ptx.AppendLine("    add.u64 %rd6, %rd6, %rd5;");
        ptx.AppendLine("    ld.global.v2.f32 {%f0,%f1}, [%rd6];");
        ptx.AppendLine($"    mul.wide.u32 %rd7, %r4, {keyValueSequence * sizeof(float)};");
        ptx.AppendLine("    add.u64 %rd7, %rd2, %rd7;");
        ptx.AppendLine("    mov.f32 %f2, 0f00000000;");
        ptx.AppendLine("    mov.u32 %r11, 0;");
        ptx.AppendLine("DELTA_KEY_LOOP:");
        ptx.AppendLine($"    setp.ge.u32 %p0, %r11, {keyValueSequence};");
        ptx.AppendLine("    @%p0 bra.uni DELTA_KEY_DONE;");
        EmitKeyValuePairAddress(ptx, "%rd1", "%rd8", "%r9", "%r11", "%r10", keyValueSequence);
        ptx.AppendLine("    ld.global.v2.f32 {%f3,%f4}, [%rd8];");
        ptx.AppendLine("    mul.rn.f32 %f5, %f0, %f3;");
        ptx.AppendLine("    fma.rn.f32 %f5, %f1, %f4, %f5;");
        EmitWarpSum(ptx, "%f5", "%f6", "%r20", "%r21");
        ptx.AppendLine("    mul.wide.u32 %rd9, %r11, 4;");
        ptx.AppendLine("    add.u64 %rd9, %rd7, %rd9;");
        ptx.AppendLine("    ld.global.f32 %f7, [%rd9];");
        ptx.AppendLine("    fma.rn.f32 %f2, %f7, %f5, %f2;");
        ptx.AppendLine("    add.u32 %r11, %r11, 1;");
        ptx.AppendLine("    bra.uni DELTA_KEY_LOOP;");
        ptx.AppendLine("DELTA_KEY_DONE:");
        ptx.AppendLine("    setp.eq.u32 %p1, %r2, 0;");
        ptx.AppendLine("    add.u64 %rd10, %rd3, %rd4;");
        ptx.AppendLine("    @%p1 st.global.f32 [%rd10], %f2;");
        ptx.AppendLine("    ret;");
        ptx.AppendLine("}");
    }

    private static void EmitGradQuery(
        StringBuilder ptx,
        int queryHeads,
        int keyValueHeads,
        int querySequence,
        int keyValueSequence,
        int queriesPerKeyValueHead,
        float scale)
    {
        ptx.AppendLine($".visible .entry {GradQueryEntryPoint}(");
        ptx.AppendLine("    .param .u64 grad_output_ptr,");
        ptx.AppendLine("    .param .u64 key_ptr,");
        ptx.AppendLine("    .param .u64 value_ptr,");
        ptx.AppendLine("    .param .u64 probabilities_ptr,");
        ptx.AppendLine("    .param .u64 grad_query_ptr");
        ptx.AppendLine(")");
        ptx.AppendLine("{");
        ptx.AppendLine("    .reg .pred %p<4>;");
        ptx.AppendLine("    .reg .b32 %r<32>;");
        ptx.AppendLine("    .reg .b64 %rd<32>;");
        ptx.AppendLine("    .reg .f32 %f<32>;");
        ptx.AppendLine("    ld.param.u64 %rd0, [grad_output_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd1, [key_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd2, [value_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd3, [probabilities_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd4, [grad_query_ptr];");
        ptx.AppendLine("    mov.u32 %r0, %tid.x;");
        ptx.AppendLine("    shr.u32 %r1, %r0, 5;");
        ptx.AppendLine("    and.b32 %r2, %r0, 31;");
        ptx.AppendLine("    mov.u32 %r3, %ctaid.x;");
        ptx.AppendLine($"    mad.lo.u32 %r4, %r3, {WarpsPerBlock}, %r1;");
        ptx.AppendLine($"    div.u32 %r5, %r4, {querySequence};");
        ptx.AppendLine($"    rem.u32 %r6, %r4, {querySequence};");
        ptx.AppendLine($"    rem.u32 %r7, %r5, {queryHeads};");
        ptx.AppendLine($"    div.u32 %r8, %r5, {queryHeads};");
        ptx.AppendLine($"    div.u32 %r9, %r7, {queriesPerKeyValueHead};");
        ptx.AppendLine($"    mad.lo.u32 %r10, %r8, {keyValueHeads}, %r9;");
        ptx.AppendLine("    shl.b32 %r11, %r2, 1;");
        ptx.AppendLine("    mul.wide.u32 %rd5, %r4, 256;");
        ptx.AppendLine("    mul.wide.u32 %rd6, %r2, 8;");
        ptx.AppendLine("    add.u64 %rd7, %rd0, %rd5;");
        ptx.AppendLine("    add.u64 %rd7, %rd7, %rd6;");
        ptx.AppendLine("    ld.global.v2.f32 {%f0,%f1}, [%rd7];");
        ptx.AppendLine($"    mul.wide.u32 %rd8, %r4, {keyValueSequence * sizeof(float)};");
        ptx.AppendLine("    add.u64 %rd8, %rd3, %rd8;");
        ptx.AppendLine("    add.u64 %rd9, %rd4, %rd5;");
        ptx.AppendLine("    ld.global.f32 %f2, [%rd9];");
        ptx.AppendLine("    mov.f32 %f8, 0f00000000;");
        ptx.AppendLine("    mov.f32 %f9, 0f00000000;");
        ptx.AppendLine("    mov.u32 %r12, 0;");
        ptx.AppendLine("GQ_ACCUM_LOOP:");
        ptx.AppendLine($"    setp.ge.u32 %p0, %r12, {keyValueSequence};");
        ptx.AppendLine("    @%p0 bra.uni GQ_ACCUM_DONE;");
        EmitKeyValuePairAddress(ptx, "%rd2", "%rd9", "%r10", "%r12", "%r11", keyValueSequence);
        ptx.AppendLine("    ld.global.v2.f32 {%f3,%f4}, [%rd9];");
        ptx.AppendLine("    mul.rn.f32 %f5, %f0, %f3;");
        ptx.AppendLine("    fma.rn.f32 %f5, %f1, %f4, %f5;");
        EmitWarpSum(ptx, "%f5", "%f6", "%r20", "%r21");
        ptx.AppendLine("    mul.wide.u32 %rd10, %r12, 4;");
        ptx.AppendLine("    add.u64 %rd10, %rd8, %rd10;");
        ptx.AppendLine("    ld.global.f32 %f7, [%rd10];");
        ptx.AppendLine("    sub.rn.f32 %f10, %f5, %f2;");
        ptx.AppendLine("    mul.rn.f32 %f10, %f10, %f7;");
        ptx.AppendLine($"    mul.rn.f32 %f10, %f10, {FloatLiteral(scale)};");
        EmitKeyValuePairAddress(ptx, "%rd1", "%rd11", "%r10", "%r12", "%r11", keyValueSequence);
        ptx.AppendLine("    ld.global.v2.f32 {%f11,%f12}, [%rd11];");
        ptx.AppendLine("    fma.rn.f32 %f8, %f10, %f11, %f8;");
        ptx.AppendLine("    fma.rn.f32 %f9, %f10, %f12, %f9;");
        ptx.AppendLine("    add.u32 %r12, %r12, 1;");
        ptx.AppendLine("    bra.uni GQ_ACCUM_LOOP;");
        ptx.AppendLine("GQ_ACCUM_DONE:");
        ptx.AppendLine("    add.u64 %rd12, %rd4, %rd5;");
        ptx.AppendLine("    add.u64 %rd12, %rd12, %rd6;");
        ptx.AppendLine("    st.global.v2.f32 [%rd12], {%f8,%f9};");
        ptx.AppendLine("    ret;");
        ptx.AppendLine("}");
    }

    private static void EmitParallelGradKeyValue(
        StringBuilder ptx,
        int queryHeads,
        int keyValueHeads,
        int querySequence,
        int keyValueSequence,
        int queriesPerKeyValueHead,
        float scale)
    {
        int rowsPerKeyValueHead = checked(queriesPerKeyValueHead * querySequence);
        ptx.AppendLine($".visible .entry {GradKeyValueEntryPoint}(");
        ptx.AppendLine("    .param .u64 grad_output_ptr,");
        ptx.AppendLine("    .param .u64 query_ptr,");
        ptx.AppendLine("    .param .u64 value_ptr,");
        ptx.AppendLine("    .param .u64 probabilities_ptr,");
        ptx.AppendLine("    .param .u64 grad_query_workspace_ptr,");
        ptx.AppendLine("    .param .u64 grad_key_ptr,");
        ptx.AppendLine("    .param .u64 grad_value_ptr");
        ptx.AppendLine(")");
        ptx.AppendLine("{");
        ptx.AppendLine("    .reg .pred %p<2>;");
        ptx.AppendLine("    .reg .b32 %r<32>;");
        ptx.AppendLine("    .reg .b64 %rd<32>;");
        ptx.AppendLine("    .reg .f32 %f<32>;");
        ptx.AppendLine("    ld.param.u64 %rd0, [grad_output_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd1, [query_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd2, [value_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd3, [probabilities_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd4, [grad_query_workspace_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd5, [grad_key_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd6, [grad_value_ptr];");
        ptx.AppendLine("    mov.u32 %r0, %tid.x;");
        ptx.AppendLine("    shr.u32 %r1, %r0, 5;");
        ptx.AppendLine("    and.b32 %r2, %r0, 31;");
        ptx.AppendLine("    mov.u32 %r3, %ctaid.x;");
        ptx.AppendLine($"    mad.lo.u32 %r4, %r3, {WarpsPerBlock}, %r1;");
        ptx.AppendLine($"    div.u32 %r5, %r4, {keyValueSequence};");
        ptx.AppendLine($"    rem.u32 %r6, %r4, {keyValueSequence};");
        ptx.AppendLine($"    div.u32 %r7, %r5, {keyValueHeads};");
        ptx.AppendLine($"    rem.u32 %r8, %r5, {keyValueHeads};");
        ptx.AppendLine("    shl.b32 %r9, %r2, 1;");
        EmitKeyValuePairAddress(ptx, "%rd2", "%rd7", "%r5", "%r6", "%r9", keyValueSequence);
        ptx.AppendLine("    ld.global.v2.f32 {%f0,%f1}, [%rd7];");
        ptx.AppendLine("    mov.f32 %f2, 0f00000000;");
        ptx.AppendLine("    mov.f32 %f3, 0f00000000;");
        ptx.AppendLine("    mov.f32 %f4, 0f00000000;");
        ptx.AppendLine("    mov.f32 %f5, 0f00000000;");
        ptx.AppendLine("    mov.u32 %r10, 0;");
        ptx.AppendLine("PKV_ROW_LOOP:");
        ptx.AppendLine($"    setp.ge.u32 %p0, %r10, {rowsPerKeyValueHead};");
        ptx.AppendLine("    @%p0 bra.uni PKV_ROWS_DONE;");
        ptx.AppendLine($"    div.u32 %r11, %r10, {querySequence};");
        ptx.AppendLine($"    rem.u32 %r12, %r10, {querySequence};");
        ptx.AppendLine($"    mad.lo.u32 %r13, %r8, {queriesPerKeyValueHead}, %r11;");
        ptx.AppendLine($"    mad.lo.u32 %r14, %r7, {queryHeads}, %r13;");
        ptx.AppendLine($"    mad.lo.u32 %r14, %r14, {querySequence}, %r12;");
        ptx.AppendLine("    mul.wide.u32 %rd8, %r14, 256;");
        ptx.AppendLine("    mul.wide.u32 %rd9, %r2, 8;");
        ptx.AppendLine("    add.u64 %rd10, %rd0, %rd8;");
        ptx.AppendLine("    add.u64 %rd10, %rd10, %rd9;");
        ptx.AppendLine("    ld.global.v2.f32 {%f6,%f7}, [%rd10];");
        ptx.AppendLine("    add.u64 %rd11, %rd1, %rd8;");
        ptx.AppendLine("    add.u64 %rd11, %rd11, %rd9;");
        ptx.AppendLine("    ld.global.v2.f32 {%f8,%f9}, [%rd11];");
        ptx.AppendLine("    mul.rn.f32 %f10, %f6, %f0;");
        ptx.AppendLine("    fma.rn.f32 %f10, %f7, %f1, %f10;");
        EmitWarpSum(ptx, "%f10", "%f11", "%r20", "%r21");
        ptx.AppendLine($"    mul.wide.u32 %rd12, %r14, {keyValueSequence * sizeof(float)};");
        ptx.AppendLine("    mul.wide.u32 %rd13, %r6, 4;");
        ptx.AppendLine("    add.u64 %rd12, %rd3, %rd12;");
        ptx.AppendLine("    add.u64 %rd12, %rd12, %rd13;");
        ptx.AppendLine("    ld.global.f32 %f12, [%rd12];");
        ptx.AppendLine("    add.u64 %rd14, %rd4, %rd8;");
        ptx.AppendLine("    ld.global.f32 %f13, [%rd14];");
        ptx.AppendLine("    sub.rn.f32 %f14, %f10, %f13;");
        ptx.AppendLine("    mul.rn.f32 %f14, %f14, %f12;");
        ptx.AppendLine($"    mul.rn.f32 %f14, %f14, {FloatLiteral(scale)};");
        ptx.AppendLine("    fma.rn.f32 %f2, %f14, %f8, %f2;");
        ptx.AppendLine("    fma.rn.f32 %f3, %f14, %f9, %f3;");
        ptx.AppendLine("    fma.rn.f32 %f4, %f12, %f6, %f4;");
        ptx.AppendLine("    fma.rn.f32 %f5, %f12, %f7, %f5;");
        ptx.AppendLine("    add.u32 %r10, %r10, 1;");
        ptx.AppendLine("    bra.uni PKV_ROW_LOOP;");
        ptx.AppendLine("PKV_ROWS_DONE:");
        EmitKeyValuePairAddress(ptx, "%rd5", "%rd15", "%r5", "%r6", "%r9", keyValueSequence);
        ptx.AppendLine("    st.global.v2.f32 [%rd15], {%f2,%f3};");
        EmitKeyValuePairAddress(ptx, "%rd6", "%rd16", "%r5", "%r6", "%r9", keyValueSequence);
        ptx.AppendLine("    st.global.v2.f32 [%rd16], {%f4,%f5};");
        ptx.AppendLine("    ret;");
        ptx.AppendLine("}");
    }

    private static void EmitKeyValuePairAddress(
        StringBuilder ptx,
        string basePointer,
        string destination,
        string batchKeyValueHead,
        string keyIndex,
        string dimension,
        int keyValueSequence)
    {
        ptx.AppendLine($"    mad.lo.u32 %r13, {batchKeyValueHead}, {keyValueSequence}, {keyIndex};");
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
        int queryHeads,
        int keyValueHeads,
        int querySequence,
        int keyValueSequence)
    {
        var query = new DirectPtxExtent(batch, queryHeads, querySequence, HeadDimension);
        var keyValue = new DirectPtxExtent(batch, keyValueHeads, keyValueSequence, HeadDimension);
        var probabilities = new DirectPtxExtent(batch, queryHeads, querySequence, keyValueSequence);
        return new DirectPtxKernelBlueprint(
            Operation: "attention-probability-backward-d64",
            Version: 1,
            Architecture: architecture,
            Variant: "deterministic-parallel-dkv-output-workspace-delta",
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
                new("probabilities", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Bhsd,
                    probabilities, probabilities, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("grad-query", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Bhsd,
                    query, query, 16, DirectPtxTensorAccess.Write, DirectPtxExtentMode.Exact),
                new("grad-key", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Bhsd,
                    keyValue, keyValue, 16, DirectPtxTensorAccess.Write, DirectPtxExtentMode.Exact),
                new("grad-value", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Bhsd,
                    keyValue, keyValue, 16, DirectPtxTensorAccess.Write, DirectPtxExtentMode.Exact)
            ],
            ResourceBudget: new DirectPtxResourceBudget(
                MaxRegistersPerThread: 56,
                MaxStaticSharedBytes: 0,
                MaxLocalBytesPerThread: 0,
                MinBlocksPerMultiprocessor: 4),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["input"] = "fp32-probabilities",
                ["accumulator"] = "fp32",
                ["output"] = "fp32-dq-dk-dv",
                ["layout"] = "canonical-bhsd",
                ["head-mapping"] = queryHeads == keyValueHeads ? "mha" : keyValueHeads == 1 ? "mqa" : "gqa",
                ["determinism"] = "fixed-order-no-atomics",
                ["row-delta"] = "one-fp32-per-row-in-final-dq-workspace-then-overwritten",
                ["global-intermediates"] = "row-delta-reuses-final-dq-allocation-no-standalone-buffer",
                ["temporary-device-allocation"] = "none",
                ["shape"] = $"b{batch}-hq{queryHeads}-hkv{keyValueHeads}-sq{querySequence}-sk{keyValueSequence}-d64"
            });
    }

    private static void ValidateShape(
        int batch,
        int queryHeads,
        int keyValueHeads,
        int querySequence,
        int keyValueSequence,
        float scale)
    {
        if (batch <= 0 || batch > 16) throw new ArgumentOutOfRangeException(nameof(batch));
        if (queryHeads <= 0 || keyValueHeads <= 0 || queryHeads % keyValueHeads != 0)
            throw new ArgumentOutOfRangeException(nameof(keyValueHeads));
        if (querySequence is not (16 or 32 or 64 or 128))
            throw new ArgumentOutOfRangeException(nameof(querySequence));
        if (keyValueSequence is not (16 or 32 or 64 or 128))
            throw new ArgumentOutOfRangeException(nameof(keyValueSequence));
        if (checked((queryHeads / keyValueHeads) * querySequence) > 2048)
            throw new ArgumentOutOfRangeException(nameof(queryHeads));
        if (!float.IsFinite(scale)) throw new ArgumentOutOfRangeException(nameof(scale));
    }

    private static string FloatLiteral(float value) =>
        "0f" + BitConverter.SingleToInt32Bits(value).ToString("X8", CultureInfo.InvariantCulture);
}
#endif
