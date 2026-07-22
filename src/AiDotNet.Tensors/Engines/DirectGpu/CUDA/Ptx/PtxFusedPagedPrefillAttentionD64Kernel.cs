using System;
using System.Collections.Generic;
using System.Globalization;
using System.Numerics;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// Causal FP32 paged prefill for D=64. A warp owns one (query, head), retains
/// two output components per lane, and applies the baked absolute query
/// position while streaming the block-table-selected K/V pages once.
/// </summary>
internal sealed class PtxFusedPagedPrefillAttentionD64Kernel : IDisposable
{
    internal const int HeadDimension = 64;
    internal const string EntryPoint = "aidotnet_paged_prefill_d64";

    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

    internal int QueryHeads { get; }
    internal int KeyValueHeads { get; }
    internal int QueryCount { get; }
    internal int StartPosition { get; }
    internal int MaximumKeyLength { get; }
    internal int BlockSize { get; }
    internal int PoolBlocks { get; }
    internal int WarpsPerBlock { get; }
    internal float Scale { get; }
    internal nuint QueryBytes { get; }
    internal nuint KeyValueBytes { get; }
    internal nuint BlockTableBytes { get; }
    internal nuint OutputBytes { get; }
    internal string Ptx { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxFunctionInfo FunctionInfo { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal PtxFusedPagedPrefillAttentionD64Kernel(
        DirectPtxRuntime runtime,
        int queryHeads,
        int keyValueHeads,
        int queryCount,
        int startPosition,
        int blockSize,
        int poolBlocks,
        float scale)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        if (runtime.ArchitectureFamily != DirectPtxArchitectureFamily.Ampere)
            throw new PlatformNotSupportedException("The checked-in paged-prefill specialization is validated only on Ampere.");
        ValidateShape(queryHeads, keyValueHeads, queryCount, startPosition, blockSize, poolBlocks, scale);

        QueryHeads = queryHeads;
        KeyValueHeads = keyValueHeads;
        QueryCount = queryCount;
        StartPosition = startPosition;
        MaximumKeyLength = checked(startPosition + queryCount);
        BlockSize = blockSize;
        PoolBlocks = poolBlocks;
        Scale = scale;
        WarpsPerBlock = SelectWarpsPerBlock(queryHeads);
        QueryBytes = checked((nuint)queryCount * (nuint)queryHeads * HeadDimension * sizeof(float));
        KeyValueBytes = checked((nuint)poolBlocks * (nuint)blockSize * (nuint)keyValueHeads * HeadDimension * sizeof(float));
        BlockTableBytes = checked((nuint)((MaximumKeyLength + blockSize - 1) / blockSize) * sizeof(int));
        OutputBytes = QueryBytes;
        Blueprint = CreateBlueprint(
            runtime.ArchitectureFamily, queryHeads, keyValueHeads,
            queryCount, startPosition, blockSize, poolBlocks, WarpsPerBlock);
        Ptx = EmitPtx(
            runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor,
            queryHeads, keyValueHeads, queryCount, startPosition,
            blockSize, poolBlocks, scale, WarpsPerBlock);
        _module = runtime.LoadModule(Ptx);
        _function = _module.GetFunction(EntryPoint, out DirectPtxFunctionInfo functionInfo);
        FunctionInfo = functionInfo;
        int blockThreads = WarpsPerBlock * 32;
        int activeBlocks = _module.GetActiveBlocksPerMultiprocessor(_function, blockThreads);
        Blueprint.ResourceBudget.Validate(EntryPoint, functionInfo, blockThreads, activeBlocks);
        Audit = DirectPtxKernelAudit.Create(
            Blueprint, runtime.DeviceFingerprint, Ptx, functionInfo,
            blockThreads, activeBlocks, _module.JitInfoLog);
    }

    internal unsafe void Launch(
        DirectPtxTensorView query,
        DirectPtxTensorView keyPages,
        DirectPtxTensorView valuePages,
        DirectPtxTensorView blockTable,
        DirectPtxTensorView output)
    {
        Require(query, Blueprint.Tensors[0], nameof(query));
        Require(keyPages, Blueprint.Tensors[1], nameof(keyPages));
        Require(valuePages, Blueprint.Tensors[2], nameof(valuePages));
        Require(blockTable, Blueprint.Tensors[3], nameof(blockTable));
        Require(output, Blueprint.Tensors[4], nameof(output));
        if (output.Pointer == query.Pointer || output.Pointer == keyPages.Pointer ||
            output.Pointer == valuePages.Pointer || output.Pointer == blockTable.Pointer)
            throw new ArgumentException("Paged-prefill output may not alias Q, K, V, or the block table.", nameof(output));

        IntPtr q = query.Pointer, k = keyPages.Pointer, v = valuePages.Pointer;
        IntPtr table = blockTable.Pointer, o = output.Pointer;
        void** arguments = stackalloc void*[5];
        arguments[0] = &q; arguments[1] = &k; arguments[2] = &v;
        arguments[3] = &table; arguments[4] = &o;
        _module.Launch(
            _function, (uint)(QueryHeads / WarpsPerBlock), (uint)QueryCount, 1,
            (uint)(WarpsPerBlock * 32), 1, 1, 0, arguments);
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

    public void Dispose() => _module.Dispose();

    internal static string EmitPtx(
        int ccMajor,
        int ccMinor,
        int queryHeads,
        int keyValueHeads,
        int queryCount,
        int startPosition,
        int blockSize,
        int poolBlocks,
        float scale,
        int warpsPerBlock = 0)
    {
        ValidateShape(queryHeads, keyValueHeads, queryCount, startPosition, blockSize, poolBlocks, scale);
        if (warpsPerBlock == 0) warpsPerBlock = SelectWarpsPerBlock(queryHeads);
        if (warpsPerBlock is not (1 or 2 or 4) || queryHeads % warpsPerBlock != 0)
            throw new ArgumentOutOfRangeException(nameof(warpsPerBlock));

        int maximumKeyLength = checked(startPosition + queryCount);
        int queryHeadsPerKeyValueHead = queryHeads / keyValueHeads;
        string scaleHex = FloatLiteral(scale);
        const string log2E = "0f3FB8AA3B";
        var ptx = new StringBuilder(16_384 + maximumKeyLength * 1_200);
        ptx.AppendLine(".version 7.1");
        ptx.AppendLine($".target sm_{ccMajor}{ccMinor}");
        ptx.AppendLine(".address_size 64");
        ptx.AppendLine();
        ptx.AppendLine($".visible .entry {EntryPoint}(");
        ptx.AppendLine("    .param .u64 query_ptr,");
        ptx.AppendLine("    .param .u64 key_ptr,");
        ptx.AppendLine("    .param .u64 value_ptr,");
        ptx.AppendLine("    .param .u64 block_table_ptr,");
        ptx.AppendLine("    .param .u64 output_ptr");
        ptx.AppendLine(")");
        ptx.AppendLine("{");
        ptx.AppendLine("    .reg .pred %p<2>;");
        ptx.AppendLine("    .reg .b32 %r<24>;");
        ptx.AppendLine("    .reg .b64 %rd<16>;");
        ptx.AppendLine("    .reg .f32 %f<20>;");
        ptx.AppendLine("    ld.param.u64 %rd0, [query_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd1, [key_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd2, [value_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd3, [block_table_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd4, [output_ptr];");
        ptx.AppendLine("    mov.u32 %r0, %tid.x;");
        ptx.AppendLine("    shr.u32 %r1, %r0, 5;");
        ptx.AppendLine("    and.b32 %r2, %r0, 31;");
        ptx.AppendLine("    mov.u32 %r3, %ctaid.x;");
        ptx.AppendLine($"    mad.lo.u32 %r4, %r3, {warpsPerBlock}, %r1;");
        ptx.AppendLine("    mov.u32 %r9, %ctaid.y;");
        ptx.AppendLine($"    mad.lo.u32 %r10, %r9, {queryHeads}, %r4;");
        if (queryHeadsPerKeyValueHead == 1)
            ptx.AppendLine("    mov.u32 %r5, %r4;");
        else if ((queryHeadsPerKeyValueHead & (queryHeadsPerKeyValueHead - 1)) == 0)
            ptx.AppendLine($"    shr.u32 %r5, %r4, {PtxCompat.TrailingZeroCount((uint)queryHeadsPerKeyValueHead)};");
        else
            ptx.AppendLine($"    div.u32 %r5, %r4, {queryHeadsPerKeyValueHead};");
        ptx.AppendLine("    shl.b32 %r8, %r2, 1;");
        ptx.AppendLine("    mul.wide.u32 %rd5, %r10, 256;");
        ptx.AppendLine("    mul.wide.u32 %rd6, %r2, 8;");
        ptx.AppendLine("    add.u64 %rd7, %rd0, %rd5;");
        ptx.AppendLine("    add.u64 %rd7, %rd7, %rd6;");
        ptx.AppendLine("    ld.global.v2.f32 {%f0,%f1}, [%rd7];");
        ptx.AppendLine($"    add.u32 %r11, %r9, {startPosition + 1};");
        ptx.AppendLine("    mov.f32 %f2, 0fFF800000;");
        ptx.AppendLine("    mov.f32 %f3, 0f00000000;");
        ptx.AppendLine("    mov.f32 %f4, 0f00000000;");
        ptx.AppendLine("    mov.f32 %f5, 0f00000000;");

        int lastLogicalBlock = -1;
        for (int token = 0; token < maximumKeyLength; token++)
        {
            ptx.AppendLine($"    // ---- paged prefill token {token} ----");
            ptx.AppendLine($"    setp.ge.u32 %p0, {token}, %r11;");
            ptx.AppendLine($"    @%p0 bra.uni SKIP_TOKEN_{token};");
            int logicalBlock = token / blockSize;
            int position = token % blockSize;
            if (logicalBlock != lastLogicalBlock)
            {
                ptx.AppendLine($"    ld.global.u32 %r6, [%rd3+{logicalBlock * sizeof(int)}];");
                lastLogicalBlock = logicalBlock;
            }
            ptx.AppendLine($"    mad.lo.u32 %r7, %r6, {blockSize}, {position};");
            ptx.AppendLine($"    mad.lo.u32 %r7, %r7, {keyValueHeads}, %r5;");
            ptx.AppendLine("    mad.lo.u32 %r7, %r7, 64, %r8;");
            ptx.AppendLine("    mul.wide.u32 %rd8, %r7, 4;");
            ptx.AppendLine("    add.u64 %rd9, %rd1, %rd8;");
            ptx.AppendLine("    add.u64 %rd10, %rd2, %rd8;");
            ptx.AppendLine("    ld.global.v2.f32 {%f6,%f7}, [%rd9];");
            ptx.AppendLine("    mul.rn.f32 %f8, %f0, %f6;");
            ptx.AppendLine("    fma.rn.f32 %f8, %f1, %f7, %f8;");
            foreach (int delta in new[] { 16, 8, 4, 2, 1 })
            {
                ptx.AppendLine("    mov.b32 %r14, %f8;");
                ptx.AppendLine($"    shfl.sync.bfly.b32 %r15, %r14, {delta}, 31, 0xffffffff;");
                ptx.AppendLine("    mov.b32 %f9, %r15;");
                ptx.AppendLine("    add.rn.f32 %f8, %f8, %f9;");
            }
            ptx.AppendLine($"    mul.rn.f32 %f10, %f8, {scaleHex};");
            ptx.AppendLine("    max.f32 %f11, %f2, %f10;");
            ptx.AppendLine("    sub.rn.f32 %f12, %f2, %f11;");
            ptx.AppendLine($"    mul.rn.f32 %f12, %f12, {log2E};");
            ptx.AppendLine("    ex2.approx.f32 %f12, %f12;");
            ptx.AppendLine("    sub.rn.f32 %f13, %f10, %f11;");
            ptx.AppendLine($"    mul.rn.f32 %f13, %f13, {log2E};");
            ptx.AppendLine("    ex2.approx.f32 %f13, %f13;");
            ptx.AppendLine("    fma.rn.f32 %f3, %f3, %f12, %f13;");
            ptx.AppendLine("    ld.global.v2.f32 {%f14,%f15}, [%rd10];");
            ptx.AppendLine("    mul.rn.f32 %f4, %f4, %f12;");
            ptx.AppendLine("    fma.rn.f32 %f4, %f13, %f14, %f4;");
            ptx.AppendLine("    mul.rn.f32 %f5, %f5, %f12;");
            ptx.AppendLine("    fma.rn.f32 %f5, %f13, %f15, %f5;");
            ptx.AppendLine("    mov.f32 %f2, %f11;");
            ptx.AppendLine($"SKIP_TOKEN_{token}:");
        }

        ptx.AppendLine("    rcp.approx.f32 %f16, %f3;");
        ptx.AppendLine("    mul.rn.f32 %f17, %f4, %f16;");
        ptx.AppendLine("    mul.rn.f32 %f18, %f5, %f16;");
        ptx.AppendLine("    add.u64 %rd11, %rd4, %rd5;");
        ptx.AppendLine("    add.u64 %rd11, %rd11, %rd6;");
        ptx.AppendLine("    st.global.v2.f32 [%rd11], {%f17,%f18};");
        ptx.AppendLine("    ret;");
        ptx.AppendLine("}");
        return ptx.ToString();
    }

    private static DirectPtxKernelBlueprint CreateBlueprint(
        DirectPtxArchitectureFamily architecture,
        int queryHeads,
        int keyValueHeads,
        int queryCount,
        int startPosition,
        int blockSize,
        int poolBlocks,
        int warpsPerBlock)
    {
        int maximumKeyLength = checked(startPosition + queryCount);
        var query = new DirectPtxExtent(queryCount, queryHeads, HeadDimension);
        var pool = new DirectPtxExtent(poolBlocks, blockSize, keyValueHeads, HeadDimension);
        var table = new DirectPtxExtent((maximumKeyLength + blockSize - 1) / blockSize);
        return new DirectPtxKernelBlueprint(
            Operation: "paged-prefill-attention-d64",
            Version: 1,
            Architecture: architecture,
            Variant: $"warp-register-online-w{warpsPerBlock}",
            Tensors:
            [
                new("query", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.SequenceHeadDim,
                    query, query, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("key-pages", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.PagedKv,
                    pool, pool, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("value-pages", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.PagedKv,
                    pool, pool, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("block-table", DirectPtxPhysicalType.Int32, DirectPtxPhysicalLayout.Vector,
                    table, table, 4, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("output", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.SequenceHeadDim,
                    query, query, 16, DirectPtxTensorAccess.Write, DirectPtxExtentMode.Exact)
            ],
            ResourceBudget: new DirectPtxResourceBudget(
                MaxRegistersPerThread: 56,
                MaxStaticSharedBytes: 0,
                MaxLocalBytesPerThread: 0,
                MinBlocksPerMultiprocessor: 8),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["input"] = "fp32",
                ["accumulator"] = "fp32",
                ["output"] = "fp32",
                ["layout"] = "paged-block-token-head-d64",
                ["query-layout"] = "sequence-head-d64",
                ["head-mapping"] = queryHeads == keyValueHeads ? "mha" : keyValueHeads == 1 ? "mqa" : "gqa",
                ["mask"] = "causal-absolute-position-baked",
                ["softmax"] = "single-pass-online-register-resident",
                ["global-intermediates"] = "none",
                ["shape"] = $"q{queryCount}-start{startPosition}-hq{queryHeads}-hkv{keyValueHeads}-bs{blockSize}-pool{poolBlocks}-d64"
            });
    }

    private static int SelectWarpsPerBlock(int queryHeads) =>
        queryHeads % 4 == 0 ? 4 : queryHeads % 2 == 0 ? 2 : 1;

    private static void ValidateShape(
        int queryHeads,
        int keyValueHeads,
        int queryCount,
        int startPosition,
        int blockSize,
        int poolBlocks,
        float scale)
    {
        if (queryHeads <= 0 || keyValueHeads <= 0 || queryHeads % keyValueHeads != 0)
            throw new ArgumentOutOfRangeException(nameof(keyValueHeads));
        if (queryCount is not (2 or 4 or 8 or 16 or 32))
            throw new ArgumentOutOfRangeException(nameof(queryCount));
        if (startPosition < 0 || checked(startPosition + queryCount) > 128)
            throw new ArgumentOutOfRangeException(nameof(startPosition));
        if (blockSize is not (16 or 32)) throw new ArgumentOutOfRangeException(nameof(blockSize));
        int logicalBlocks = (startPosition + queryCount + blockSize - 1) / blockSize;
        if (poolBlocks < logicalBlocks) throw new ArgumentOutOfRangeException(nameof(poolBlocks));
        if (!PtxCompat.IsFinite(scale)) throw new ArgumentOutOfRangeException(nameof(scale));
    }

    private static string FloatLiteral(float value) =>
        "0f" + PtxCompat.SingleToInt32Bits(value).ToString("X8", CultureInfo.InvariantCulture);
}
