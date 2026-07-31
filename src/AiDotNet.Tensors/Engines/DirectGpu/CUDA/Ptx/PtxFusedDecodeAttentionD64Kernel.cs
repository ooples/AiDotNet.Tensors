using System;
using System.Collections.Generic;
using System.Globalization;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// Single-pass FP32 decode attention for D=64. One warp owns one query head:
/// each lane retains two output components in registers while warp shuffles
/// reduce QK. Dense and paged KV are separate physical-ABI domains.
/// </summary>
internal sealed class PtxFusedDecodeAttentionD64Kernel : IDisposable
{
    internal const int HeadDimension = 64;
    internal const string DenseEntryPoint = "aidotnet_flash_decode_d64";
    internal const string PagedEntryPoint = "aidotnet_paged_decode_d64";

    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

    internal bool IsPaged { get; }
    internal int QueryHeads { get; }
    internal int KeyValueHeads { get; }
    internal int SequenceLength { get; }
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

    internal PtxFusedDecodeAttentionD64Kernel(
        DirectPtxRuntime runtime,
        bool isPaged,
        int queryHeads,
        int keyValueHeads,
        int sequenceLength,
        int blockSize,
        int poolBlocks,
        float scale)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        if (runtime.ArchitectureFamily != DirectPtxArchitectureFamily.Ampere)
            throw new PlatformNotSupportedException("The checked-in decode specialization is validated only on Ampere.");
        ValidateShape(isPaged, queryHeads, keyValueHeads, sequenceLength, blockSize, poolBlocks, scale);

        IsPaged = isPaged;
        QueryHeads = queryHeads;
        KeyValueHeads = keyValueHeads;
        SequenceLength = sequenceLength;
        BlockSize = blockSize;
        PoolBlocks = poolBlocks;
        Scale = scale;
        WarpsPerBlock = SelectWarpsPerBlock(queryHeads);
        QueryBytes = checked((nuint)queryHeads * HeadDimension * sizeof(float));
        KeyValueBytes = isPaged
            ? checked((nuint)poolBlocks * (nuint)blockSize * (nuint)keyValueHeads * HeadDimension * sizeof(float))
            : checked((nuint)sequenceLength * (nuint)keyValueHeads * HeadDimension * sizeof(float));
        BlockTableBytes = isPaged
            ? checked((nuint)((sequenceLength + blockSize - 1) / blockSize) * sizeof(int))
            : 0;
        OutputBytes = QueryBytes;
        Blueprint = CreateBlueprint(
            runtime.ArchitectureFamily, isPaged, queryHeads, keyValueHeads,
            sequenceLength, blockSize, poolBlocks, WarpsPerBlock);
        Ptx = EmitPtx(
            runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor,
            isPaged, queryHeads, keyValueHeads, sequenceLength,
            blockSize, poolBlocks, scale, WarpsPerBlock);
        _module = runtime.LoadModule(Ptx);
        _function = _module.GetFunction(
            isPaged ? PagedEntryPoint : DenseEntryPoint,
            out DirectPtxFunctionInfo functionInfo);
        FunctionInfo = functionInfo;
        int blockThreads = WarpsPerBlock * 32;
        int activeBlocks = _module.GetActiveBlocksPerMultiprocessor(_function, blockThreads);
        Blueprint.ResourceBudget.Validate(
            isPaged ? PagedEntryPoint : DenseEntryPoint,
            functionInfo, blockThreads, activeBlocks);
        Audit = DirectPtxKernelAudit.Create(
            Blueprint, runtime.DeviceFingerprint, Ptx, functionInfo,
            blockThreads, activeBlocks, _module.JitInfoLog);
    }

    internal unsafe void LaunchDense(
        DirectPtxTensorView query,
        DirectPtxTensorView key,
        DirectPtxTensorView value,
        DirectPtxTensorView output)
    {
        if (IsPaged) throw new InvalidOperationException("A paged decode module requires a block table.");
        Require(query, Blueprint.Tensors[0], nameof(query));
        Require(key, Blueprint.Tensors[1], nameof(key));
        Require(value, Blueprint.Tensors[2], nameof(value));
        Require(output, Blueprint.Tensors[3], nameof(output));
        RejectAliases(query, key, value, output);

        IntPtr q = query.Pointer, k = key.Pointer, v = value.Pointer, o = output.Pointer;
        void** arguments = stackalloc void*[4];
        arguments[0] = &q; arguments[1] = &k; arguments[2] = &v; arguments[3] = &o;
        Launch(arguments);
    }

    internal unsafe void LaunchPaged(
        DirectPtxTensorView query,
        DirectPtxTensorView key,
        DirectPtxTensorView value,
        DirectPtxTensorView blockTable,
        DirectPtxTensorView output)
    {
        if (!IsPaged) throw new InvalidOperationException("A dense decode module has no block table ABI.");
        Require(query, Blueprint.Tensors[0], nameof(query));
        Require(key, Blueprint.Tensors[1], nameof(key));
        Require(value, Blueprint.Tensors[2], nameof(value));
        Require(blockTable, Blueprint.Tensors[3], nameof(blockTable));
        Require(output, Blueprint.Tensors[4], nameof(output));
        RejectAliases(query, key, value, output);

        IntPtr q = query.Pointer, k = key.Pointer, v = value.Pointer;
        IntPtr table = blockTable.Pointer, o = output.Pointer;
        void** arguments = stackalloc void*[5];
        arguments[0] = &q; arguments[1] = &k; arguments[2] = &v;
        arguments[3] = &table; arguments[4] = &o;
        Launch(arguments);
    }

    private unsafe void Launch(void** arguments) =>
        _module.Launch(
            _function, (uint)(QueryHeads / WarpsPerBlock), 1, 1,
            (uint)(WarpsPerBlock * 32), 1, 1, 0, arguments);

    private static void RejectAliases(
        DirectPtxTensorView query,
        DirectPtxTensorView key,
        DirectPtxTensorView value,
        DirectPtxTensorView output)
    {
        if (output.Pointer == query.Pointer || output.Pointer == key.Pointer || output.Pointer == value.Pointer)
            throw new ArgumentException("Decode output may not alias Q, K, or V.", nameof(output));
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
        bool isPaged,
        int queryHeads,
        int keyValueHeads,
        int sequenceLength,
        int blockSize,
        int poolBlocks,
        float scale,
        int warpsPerBlock = 0)
    {
        ValidateShape(isPaged, queryHeads, keyValueHeads, sequenceLength, blockSize, poolBlocks, scale);
        if (warpsPerBlock == 0) warpsPerBlock = SelectWarpsPerBlock(queryHeads);
        if (warpsPerBlock is not (1 or 2 or 4) || queryHeads % warpsPerBlock != 0)
            throw new ArgumentOutOfRangeException(nameof(warpsPerBlock));

        string scaleHex = FloatLiteral(scale);
        const string log2E = "0f3FB8AA3B";
        var ptx = new StringBuilder(16_384 + sequenceLength * 1_024);
        ptx.AppendLine(".version 7.1");
        ptx.AppendLine($".target sm_{ccMajor}{ccMinor}");
        ptx.AppendLine(".address_size 64");
        ptx.AppendLine();
        ptx.AppendLine($".visible .entry {(isPaged ? PagedEntryPoint : DenseEntryPoint)}(");
        ptx.AppendLine("    .param .u64 query_ptr,");
        ptx.AppendLine("    .param .u64 key_ptr,");
        ptx.AppendLine("    .param .u64 value_ptr,");
        if (isPaged) ptx.AppendLine("    .param .u64 block_table_ptr,");
        ptx.AppendLine("    .param .u64 output_ptr");
        ptx.AppendLine(")");
        ptx.AppendLine("{");
        ptx.AppendLine("    .reg .b32 %r<20>;");
        ptx.AppendLine("    .reg .b64 %rd<16>;");
        ptx.AppendLine("    .reg .f32 %f<20>;");
        ptx.AppendLine("    ld.param.u64 %rd0, [query_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd1, [key_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd2, [value_ptr];");
        if (isPaged)
        {
            ptx.AppendLine("    ld.param.u64 %rd3, [block_table_ptr];");
            ptx.AppendLine("    ld.param.u64 %rd4, [output_ptr];");
        }
        else
        {
            ptx.AppendLine("    ld.param.u64 %rd4, [output_ptr];");
        }
        ptx.AppendLine("    mov.u32 %r0, %tid.x;");
        ptx.AppendLine("    shr.u32 %r1, %r0, 5;");
        ptx.AppendLine("    and.b32 %r2, %r0, 31;");
        ptx.AppendLine("    mov.u32 %r3, %ctaid.x;");
        ptx.AppendLine($"    mad.lo.u32 %r4, %r3, {warpsPerBlock}, %r1;");
        int queryHeadsPerKeyValueHead = queryHeads / keyValueHeads;
        if (queryHeadsPerKeyValueHead == 1)
            ptx.AppendLine("    mov.u32 %r5, %r4;");
        else if ((queryHeadsPerKeyValueHead & (queryHeadsPerKeyValueHead - 1)) == 0)
            ptx.AppendLine($"    shr.u32 %r5, %r4, {PtxCompat.TrailingZeroCount((uint)queryHeadsPerKeyValueHead)};");
        else
            ptx.AppendLine($"    div.u32 %r5, %r4, {queryHeadsPerKeyValueHead};");
        ptx.AppendLine("    shl.b32 %r8, %r2, 1;");
        ptx.AppendLine("    mul.wide.u32 %rd5, %r4, 256;");
        ptx.AppendLine("    mul.wide.u32 %rd6, %r2, 8;");
        ptx.AppendLine("    add.u64 %rd7, %rd0, %rd5;");
        ptx.AppendLine("    add.u64 %rd7, %rd7, %rd6;");
        ptx.AppendLine("    ld.global.v2.f32 {%f0,%f1}, [%rd7];");
        ptx.AppendLine("    mov.f32 %f2, 0fFF800000;");
        ptx.AppendLine("    mov.f32 %f3, 0f00000000;");
        ptx.AppendLine("    mov.f32 %f4, 0f00000000;");
        ptx.AppendLine("    mov.f32 %f5, 0f00000000;");

        int lastLogicalBlock = -1;
        for (int token = 0; token < sequenceLength; token++)
        {
            ptx.AppendLine($"    // ---- decode token {token} ----");
            if (isPaged)
            {
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
            }
            else
            {
                ptx.AppendLine("    mad.lo.u32 %r7, %r5, 64, %r8;");
                ptx.AppendLine("    mul.wide.u32 %rd8, %r7, 4;");
                int tokenBytes = checked(token * keyValueHeads * HeadDimension * sizeof(float));
                ptx.AppendLine("    add.u64 %rd9, %rd1, %rd8;");
                ptx.AppendLine("    add.u64 %rd10, %rd2, %rd8;");
                if (tokenBytes != 0)
                {
                    ptx.AppendLine($"    add.u64 %rd9, %rd9, {tokenBytes};");
                    ptx.AppendLine($"    add.u64 %rd10, %rd10, {tokenBytes};");
                }
            }
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
        bool isPaged,
        int queryHeads,
        int keyValueHeads,
        int sequenceLength,
        int blockSize,
        int poolBlocks,
        int warpsPerBlock)
    {
        var query = new DirectPtxExtent(queryHeads, HeadDimension);
        var output = new DirectPtxExtent(queryHeads, HeadDimension);
        var tensors = new List<DirectPtxTensorContract>(isPaged ? 5 : 4)
        {
            new("query", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D,
                query, query, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact)
        };
        if (isPaged)
        {
            var pool = new DirectPtxExtent(poolBlocks, blockSize, keyValueHeads, HeadDimension);
            int logicalBlocks = (sequenceLength + blockSize - 1) / blockSize;
            var table = new DirectPtxExtent(logicalBlocks);
            tensors.Add(new("key-pages", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.PagedKv,
                pool, pool, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact));
            tensors.Add(new("value-pages", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.PagedKv,
                pool, pool, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact));
            tensors.Add(new("block-table", DirectPtxPhysicalType.Int32, DirectPtxPhysicalLayout.Vector,
                table, table, 4, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact));
        }
        else
        {
            var keyValue = new DirectPtxExtent(sequenceLength, keyValueHeads, HeadDimension);
            tensors.Add(new("key", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.SequenceHeadDim,
                keyValue, keyValue, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact));
            tensors.Add(new("value", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.SequenceHeadDim,
                keyValue, keyValue, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact));
        }
        tensors.Add(new("output", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D,
            output, output, 16, DirectPtxTensorAccess.Write, DirectPtxExtentMode.Exact));

        return new DirectPtxKernelBlueprint(
            Operation: isPaged ? "paged-decode-attention-d64" : "flash-decode-attention-d64",
            Version: 1,
            Architecture: architecture,
            Variant: $"warp-resident-w{warpsPerBlock}",
            Tensors: tensors,
            ResourceBudget: new DirectPtxResourceBudget(
                // Paged addressing keeps the block-table base, current page,
                // and translated K/V addresses live in addition to the dense
                // online-softmax state. Preserve the zero-spill invariant and
                // require high measured occupancy instead of pretending both
                // physical ABIs have the same register budget.
                MaxRegistersPerThread: isPaged ? 48 : 40,
                MaxStaticSharedBytes: 0,
                MaxLocalBytesPerThread: 0,
                MinBlocksPerMultiprocessor: 8),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["input"] = "fp32",
                ["accumulator"] = "fp32",
                ["output"] = "fp32",
                ["layout"] = isPaged ? "paged-block-token-head-d64" : "dense-sequence-head-d64",
                ["head-mapping"] = queryHeads == keyValueHeads ? "mha" : keyValueHeads == 1 ? "mqa" : "gqa",
                ["softmax"] = "single-pass-online-register-resident",
                ["global-intermediates"] = "none",
                ["resource-domain"] = isPaged ? "paged-translation-r48-occ8" : "dense-r40-occ8",
                ["shape"] = isPaged
                    ? $"hq{queryHeads}-hkv{keyValueHeads}-s{sequenceLength}-bs{blockSize}-pool{poolBlocks}-d64"
                    : $"hq{queryHeads}-hkv{keyValueHeads}-s{sequenceLength}-d64"
            });
    }

    private static int SelectWarpsPerBlock(int queryHeads) =>
        queryHeads % 4 == 0 ? 4 : queryHeads % 2 == 0 ? 2 : 1;

    private static void ValidateShape(
        bool isPaged,
        int queryHeads,
        int keyValueHeads,
        int sequenceLength,
        int blockSize,
        int poolBlocks,
        float scale)
    {
        if (queryHeads <= 0 || keyValueHeads <= 0 || queryHeads % keyValueHeads != 0)
            throw new ArgumentOutOfRangeException(nameof(keyValueHeads));
        if (sequenceLength is not (16 or 32 or 64 or 128))
            throw new ArgumentOutOfRangeException(nameof(sequenceLength));
        if (!PtxCompat.IsFinite(scale)) throw new ArgumentOutOfRangeException(nameof(scale));
        if (isPaged)
        {
            if (blockSize is not (16 or 32)) throw new ArgumentOutOfRangeException(nameof(blockSize));
            int logicalBlocks = (sequenceLength + blockSize - 1) / blockSize;
            if (poolBlocks < logicalBlocks) throw new ArgumentOutOfRangeException(nameof(poolBlocks));
        }
        else if (blockSize != 0 || poolBlocks != 0)
        {
            throw new ArgumentException("Dense decode has no page geometry.");
        }
    }

    private static string FloatLiteral(float value) =>
        "0f" + PtxCompat.SingleToInt32Bits(value).ToString("X8", CultureInfo.InvariantCulture);
}
