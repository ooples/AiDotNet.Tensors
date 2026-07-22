#if NET5_0_OR_GREATER
using System;
using System.Globalization;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// Validated-SM86 online FlashAttention family for Sq/Skv in {16,32,64,128}, D=64, including MHA,
/// GQA, and MQA. Each warp owns a 16-row query tile, keeps the FP32 online max/sum and
/// 16x64 output fragment in registers, and the warps in a block share double-buffered
/// 16-row K/V tiles staged with cp.async. No score or probability matrix is materialized
/// in global or shared memory. Batch/head/shape mapping is baked into each emitted module;
/// the PTX contains no dynamic shape, layout, or stride checks. Other architecture
/// families fail closed until their separately tuned specialization passes the release gate.
/// </summary>
internal sealed class PtxOnlineFusedAttention128x64Kernel : IDisposable
{
    internal const string EntryPoint = "aidotnet_online_attention_128x64";
    internal const int DefaultSequenceLength = 128;
    internal const int HeadDimension = 64;
    internal const int QueryTileRows = 16;
    internal const int KeyTileRows = 16;

    internal static bool IsSupportedSequenceLength(int sequenceLength)
        => sequenceLength is 16 or 32 or 64 or 128;

    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

    internal int Batch { get; }
    internal int QueryHeads { get; }
    internal int KeyValueHeads { get; }
    internal int QuerySequence { get; }
    internal int KeyValueSequence { get; }
    internal int CausalQueryOffset { get; }
    internal int BatchHeads => checked(Batch * QueryHeads);
    internal int SequenceLength => QuerySequence;
    internal bool IsCausal { get; }
    internal bool FuseLayerNormGelu { get; }
    internal bool EmitSoftmaxStats { get; }
    internal float Scale { get; }
    internal float Epsilon { get; }
    internal string Ptx { get; }
    internal DirectPtxFunctionInfo FunctionInfo { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }
    internal string JitInfoLog => _module.JitInfoLog;

    private int QueryElementsPerHead => QuerySequence * HeadDimension;
    private int KeyValueElementsPerHead => KeyValueSequence * HeadDimension;
    private int QueryBytesPerHead => QueryElementsPerHead * sizeof(ushort);
    private int KeyValueBytesPerHead => KeyValueElementsPerHead * sizeof(ushort);
    private int OutputBytesPerHead => QueryElementsPerHead * sizeof(float);
    private int StatsBytesPerHead => QuerySequence * sizeof(float);
    internal int WarpsPerBlock { get; }

    internal nuint QBytes => checked((nuint)BatchHeads * (nuint)QueryBytesPerHead);
    internal nuint KBytes => checked((nuint)Batch * (nuint)KeyValueHeads * (nuint)KeyValueBytesPerHead);
    internal nuint VBytes => KBytes;
    internal nuint OutputBytes => checked((nuint)BatchHeads * (nuint)OutputBytesPerHead);
    internal nuint StatsBytes => checked((nuint)BatchHeads * (nuint)StatsBytesPerHead);
    internal const nuint GammaBytes = HeadDimension * sizeof(float);
    internal const nuint BetaBytes = HeadDimension * sizeof(float);

    internal PtxOnlineFusedAttention128x64Kernel(
        DirectPtxRuntime runtime,
        int batchHeads,
        bool isCausal,
        bool fuseLayerNormGelu,
        float scale = 0.125f,
        float epsilon = 1e-5f,
        int sequenceLength = DefaultSequenceLength,
        bool emitSoftmaxStats = true,
        int? warpsPerBlock = null)
        : this(
            runtime, 1, batchHeads, batchHeads, sequenceLength, sequenceLength,
            isCausal, fuseLayerNormGelu, scale, epsilon, emitSoftmaxStats, warpsPerBlock,
            causalQueryOffset: 0)
    {
    }

    internal PtxOnlineFusedAttention128x64Kernel(
        DirectPtxRuntime runtime,
        int batch,
        int queryHeads,
        int keyValueHeads,
        int querySequence,
        int keyValueSequence,
        bool isCausal,
        bool fuseLayerNormGelu,
        float scale = 0.125f,
        float epsilon = 1e-5f,
        bool emitSoftmaxStats = true,
        int? warpsPerBlock = null,
        int causalQueryOffset = 0)
    {
        ArgumentNullException.ThrowIfNull(runtime);
        if (batch <= 0) throw new ArgumentOutOfRangeException(nameof(batch));
        if (queryHeads <= 0) throw new ArgumentOutOfRangeException(nameof(queryHeads));
        if (keyValueHeads <= 0 || queryHeads % keyValueHeads != 0)
            throw new ArgumentOutOfRangeException(
                nameof(keyValueHeads), "Query heads must be evenly divisible by positive KV heads.");
        if (checked((long)batch * queryHeads) > 65535)
            throw new ArgumentOutOfRangeException(nameof(queryHeads), "The flattened query-head grid exceeds CUDA grid-y.");
        if (!float.IsFinite(scale)) throw new ArgumentOutOfRangeException(nameof(scale));
        if (!float.IsFinite(epsilon) || epsilon <= 0) throw new ArgumentOutOfRangeException(nameof(epsilon));
        if (causalQueryOffset < -querySequence)
            throw new ArgumentOutOfRangeException(
                nameof(causalQueryOffset), "The causal offset cannot fully mask every query row.");
        if (!isCausal && causalQueryOffset != 0)
            throw new ArgumentException("A causal query offset is valid only for a causal specialization.", nameof(causalQueryOffset));
        if (!IsSupportedSequenceLength(querySequence))
            throw new ArgumentOutOfRangeException(
                nameof(querySequence), "Online PTX query length must be one of 16, 32, 64, or 128.");
        if (!IsSupportedSequenceLength(keyValueSequence))
            throw new ArgumentOutOfRangeException(
                nameof(keyValueSequence), "Online PTX KV length must be one of 16, 32, 64, or 128.");
        if (!DirectPtxArchitecture.HasValidatedOnlineAttention(
            runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new NotSupportedException(
                $"Online attention has no validated SM {runtime.ComputeCapabilityMajor}.{runtime.ComputeCapabilityMinor} specialization. " +
                "Architecture families fail closed until separately tuned and benchmarked.");

        int queryTiles = querySequence / QueryTileRows;
        int selectedWarps = warpsPerBlock ?? Math.Min(8, queryTiles);
        if (selectedWarps is not (1 or 2 or 4 or 8) || selectedWarps > queryTiles || queryTiles % selectedWarps != 0)
            throw new ArgumentOutOfRangeException(
                nameof(warpsPerBlock), "Warp count must be 1/2/4/8 and evenly divide the query-tile count.");

        Batch = batch;
        QueryHeads = queryHeads;
        KeyValueHeads = keyValueHeads;
        QuerySequence = querySequence;
        KeyValueSequence = keyValueSequence;
        CausalQueryOffset = causalQueryOffset;
        IsCausal = isCausal;
        FuseLayerNormGelu = fuseLayerNormGelu;
        EmitSoftmaxStats = emitSoftmaxStats;
        Scale = scale;
        Epsilon = epsilon;
        WarpsPerBlock = selectedWarps;
        Blueprint = CreateBlueprint(
            runtime.ArchitectureFamily, batch, queryHeads, keyValueHeads,
            querySequence, keyValueSequence,
            isCausal, causalQueryOffset, fuseLayerNormGelu, emitSoftmaxStats, selectedWarps);
        Ptx = EmitFamilyPtx(
            runtime.ComputeCapabilityMajor,
            runtime.ComputeCapabilityMinor,
            queryHeads,
            keyValueHeads,
            isCausal,
            fuseLayerNormGelu,
            scale,
            epsilon,
            querySequence,
            keyValueSequence,
            emitSoftmaxStats,
            selectedWarps,
            causalQueryOffset);
        _module = runtime.LoadModule(Ptx);
        _function = _module.GetFunction(EntryPoint, out DirectPtxFunctionInfo functionInfo);
        FunctionInfo = functionInfo;
        int blockThreads = selectedWarps * 32;
        int activeBlocks = _module.GetActiveBlocksPerMultiprocessor(_function, blockThreads);
        Blueprint.ResourceBudget.Validate(EntryPoint, functionInfo, blockThreads, activeBlocks);
        Audit = DirectPtxKernelAudit.Create(
            Blueprint, runtime.DeviceFingerprint, Ptx, functionInfo,
            blockThreads, activeBlocks, _module.JitInfoLog);
    }

    private static DirectPtxKernelBlueprint CreateBlueprint(
        DirectPtxArchitectureFamily architecture,
        int batch,
        int queryHeads,
        int keyValueHeads,
        int querySequence,
        int keyValueSequence,
        bool isCausal,
        int causalQueryOffset,
        bool fuseLayerNormGelu,
        bool emitSoftmaxStats,
        int warpsPerBlock)
    {
        var queryHalf = new DirectPtxExtent(batch, queryHeads, querySequence, HeadDimension);
        var keyValueHalf = new DirectPtxExtent(batch, keyValueHeads, keyValueSequence, HeadDimension);
        var outputFloat = new DirectPtxExtent(batch, queryHeads, querySequence, HeadDimension);
        var stats = new DirectPtxExtent(batch * queryHeads, querySequence);
        var vector = new DirectPtxExtent(HeadDimension);
        return new DirectPtxKernelBlueprint(
            Operation: "online-attention-d64",
            Version: 3,
            Architecture: architecture,
            Variant: $"q16-k16-w{warpsPerBlock}",
            Tensors:
            [
                new("query", DirectPtxPhysicalType.Float16, DirectPtxPhysicalLayout.Bhsd,
                    queryHalf, queryHalf, 16, DirectPtxTensorAccess.Read),
                new("key", DirectPtxPhysicalType.Float16, DirectPtxPhysicalLayout.Bhsd,
                    keyValueHalf, keyValueHalf, 16, DirectPtxTensorAccess.Read),
                new("value", DirectPtxPhysicalType.Float16, DirectPtxPhysicalLayout.Bhsd,
                    keyValueHalf, keyValueHalf, 16, DirectPtxTensorAccess.Read),
                new("gamma", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    vector, vector, 16, DirectPtxTensorAccess.Read),
                new("beta", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    vector, vector, 16, DirectPtxTensorAccess.Read),
                new("output", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Bhsd,
                    outputFloat, outputFloat, 16, DirectPtxTensorAccess.Write),
                new("softmax-stats", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D,
                    stats, stats, 16, DirectPtxTensorAccess.Write)
            ],
            ResourceBudget: new DirectPtxResourceBudget(
                MaxRegistersPerThread: 96,
                MaxStaticSharedBytes: 32 * 1024,
                MaxLocalBytesPerThread: 0,
                MinBlocksPerMultiprocessor: 1),
            Semantics: new System.Collections.Generic.Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["input"] = "fp16",
                ["accumulator"] = "fp32",
                ["output"] = "fp32",
                ["mask"] = isCausal
                    ? causalQueryOffset == 0
                        ? "causal-top-left"
                        : $"causal-query-offset-{causalQueryOffset}"
                    : "none",
                ["epilogue"] = fuseLayerNormGelu ? "layernorm-affine-tanh-gelu" : "none",
                ["stats"] = emitSoftmaxStats ? "lse" : "none",
                ["layout"] = "dense-bhsd",
                ["head-mapping"] = queryHeads == keyValueHeads
                    ? "mha"
                    : keyValueHeads == 1 ? "mqa" : "gqa",
                ["shape"] = $"b{batch}-hq{queryHeads}-hkv{keyValueHeads}-sq{querySequence}-skv{keyValueSequence}-d64"
            });
    }

    internal unsafe void Launch(
        DirectPtxTensorView query,
        DirectPtxTensorView key,
        DirectPtxTensorView value,
        DirectPtxTensorView gamma,
        DirectPtxTensorView beta,
        DirectPtxTensorView output,
        DirectPtxTensorView softmaxStats)
    {
        Require(query, Blueprint.Tensors[0], nameof(query));
        Require(key, Blueprint.Tensors[1], nameof(key));
        Require(value, Blueprint.Tensors[2], nameof(value));
        Require(output, Blueprint.Tensors[5], nameof(output));
        if (EmitSoftmaxStats)
            Require(softmaxStats, Blueprint.Tensors[6], nameof(softmaxStats));
        if (FuseLayerNormGelu)
        {
            Require(gamma, Blueprint.Tensors[3], nameof(gamma));
            Require(beta, Blueprint.Tensors[4], nameof(beta));
        }
        if (output.Pointer == query.Pointer || output.Pointer == key.Pointer || output.Pointer == value.Pointer)
            throw new ArgumentException("The FP32 output may not alias an FP16 attention input.", nameof(output));

        IntPtr q = query.Pointer;
        IntPtr k = key.Pointer;
        IntPtr v = value.Pointer;
        IntPtr g = gamma.Pointer;
        IntPtr b = beta.Pointer;
        IntPtr o = output.Pointer;
        IntPtr s = EmitSoftmaxStats ? softmaxStats.Pointer : IntPtr.Zero;
        void** args = stackalloc void*[7];
        args[0] = &q;
        args[1] = &k;
        args[2] = &v;
        args[3] = &g;
        args[4] = &b;
        args[5] = &o;
        args[6] = &s;

        _module.Launch(
            _function,
            (uint)((QuerySequence / QueryTileRows) / WarpsPerBlock), (uint)BatchHeads, 1,
            (uint)(WarpsPerBlock * 32), 1, 1,
            0,
            args);
    }

    private static void Require(DirectPtxTensorView view, DirectPtxTensorContract contract, string name)
    {
        if (view.Pointer == IntPtr.Zero || view.Layout != contract.Layout ||
            view.PhysicalType != contract.PhysicalType || view.ByteLength < contract.RequiredBytes ||
            view.LogicalExtent != contract.LogicalExtent || view.PhysicalExtent != contract.PhysicalExtent)
            throw new ArgumentException(
                $"{name} does not satisfy physical ABI '{contract.Name}' " +
                $"({contract.Layout}/{contract.PhysicalType}/{contract.PhysicalExtent}).", name);
    }

    internal double AttentionTflops(float milliseconds)
    {
        double flops = 4.0 * BatchHeads * QuerySequence * KeyValueSequence * HeadDimension;
        return flops / (milliseconds * 1e-3) / 1e12;
    }

    public void Dispose() => _module.Dispose();

    internal static string EmitPtx(
        int ccMajor,
        int ccMinor,
        bool isCausal,
        bool fuseLayerNormGelu,
        float scale,
        float epsilon,
        int sequenceLength = DefaultSequenceLength,
        bool emitSoftmaxStats = true,
        int? warpsPerBlock = null)
    {
        return EmitFamilyPtx(
            ccMajor, ccMinor, queryHeads: 1, keyValueHeads: 1,
            isCausal, fuseLayerNormGelu, scale, epsilon,
            querySequence: sequenceLength, keyValueSequence: sequenceLength,
            emitSoftmaxStats, warpsPerBlock, causalQueryOffset: 0);
    }

    internal static string EmitFamilyPtx(
        int ccMajor,
        int ccMinor,
        int queryHeads,
        int keyValueHeads,
        bool isCausal,
        bool fuseLayerNormGelu,
        float scale,
        float epsilon,
        int querySequence,
        int keyValueSequence,
        bool emitSoftmaxStats = true,
        int? warpsPerBlock = null,
        int causalQueryOffset = 0)
    {
        if (queryHeads <= 0) throw new ArgumentOutOfRangeException(nameof(queryHeads));
        if (keyValueHeads <= 0 || queryHeads % keyValueHeads != 0)
            throw new ArgumentOutOfRangeException(nameof(keyValueHeads));
        if (causalQueryOffset < -querySequence)
            throw new ArgumentOutOfRangeException(nameof(causalQueryOffset));
        if (!isCausal && causalQueryOffset != 0)
            throw new ArgumentException("Causal query offset requires causal masking.", nameof(causalQueryOffset));
        if (!IsSupportedSequenceLength(querySequence))
            throw new ArgumentOutOfRangeException(nameof(querySequence));
        if (!IsSupportedSequenceLength(keyValueSequence))
            throw new ArgumentOutOfRangeException(nameof(keyValueSequence));
        int queryBytesPerHead = querySequence * HeadDimension * sizeof(ushort);
        int keyValueBytesPerHead = keyValueSequence * HeadDimension * sizeof(ushort);
        int outputBytesPerHead = querySequence * HeadDimension * sizeof(float);
        int statsBytesPerHead = querySequence * sizeof(float);
        int queryTiles = querySequence / QueryTileRows;
        int keyValueTiles = keyValueSequence / KeyTileRows;
        int queriesPerKeyValue = queryHeads / keyValueHeads;
        int selectedWarpsPerBlock = warpsPerBlock ?? Math.Min(8, queryTiles);
        if (selectedWarpsPerBlock is not (1 or 2 or 4 or 8) ||
            selectedWarpsPerBlock > queryTiles || queryTiles % selectedWarpsPerBlock != 0)
            throw new ArgumentOutOfRangeException(nameof(warpsPerBlock));
        int warps = selectedWarpsPerBlock;
        string target = $"sm_{ccMajor}{ccMinor}";
        string scaleHex = FloatLiteral(scale);
        string epsilonHex = FloatLiteral(epsilon);
        const string log2E = "0f3FB8AA3B";
        const string ln2 = "0f3F317218";
        const string inv64 = "0f3C800000";
        const string geluA = "0f3F4C422A"; // sqrt(2/pi)
        const string geluB = "0f3D372713"; // 0.044715

        int kShared0 = 2048 * warps;
        int vShared0 = kShared0 + 2048;
        int kShared1 = vShared0 + 2048;
        int vShared1 = kShared1 + 2048;
        int gammaShared = vShared1 + 2048;
        int betaShared = gammaShared + 256;
        int sharedBytes = betaShared + 256;

        var ptx = new StringBuilder(96 * 1024);
        ptx.AppendLine(".version 7.1");
        ptx.AppendLine($".target {target}");
        ptx.AppendLine(".address_size 64");
        ptx.AppendLine();
        ptx.AppendLine($".visible .entry {EntryPoint}(");
        ptx.AppendLine("    .param .u64 q_ptr,");
        ptx.AppendLine("    .param .u64 k_ptr,");
        ptx.AppendLine("    .param .u64 v_ptr,");
        ptx.AppendLine("    .param .u64 gamma_ptr,");
        ptx.AppendLine("    .param .u64 beta_ptr,");
        ptx.AppendLine("    .param .u64 o_ptr,");
        ptx.AppendLine("    .param .u64 stats_ptr");
        ptx.AppendLine(")");
        ptx.AppendLine("{");
        ptx.AppendLine("    .reg .pred %p<8>;");
        ptx.AppendLine("    .reg .b16 %h<8>;");
        ptx.AppendLine("    .reg .b32 %a<4>;");
        ptx.AppendLine("    .reg .b32 %b<4>;");
        ptx.AppendLine("    .reg .b32 %r<32>;");
        ptx.AppendLine("    .reg .b64 %rd<32>;");
        ptx.AppendLine("    .reg .f32 %s<8>;");
        ptx.AppendLine("    .reg .f32 %o<32>;");
        ptx.AppendLine("    .reg .f32 %f<40>;");
        ptx.AppendLine($"    .shared .align 16 .b8 smem[{sharedBytes}];");
        ptx.AppendLine();
        ptx.AppendLine("    ld.param.u64 %rd0, [q_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd1, [k_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd2, [v_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd3, [gamma_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd4, [beta_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd5, [o_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd6, [stats_ptr];");
        ptx.AppendLine("    mov.u32 %r21, %tid.x;"); // whole cooperative block
        ptx.AppendLine("    shr.u32 %r22, %r21, 5;"); // warp owning this query tile
        ptx.AppendLine("    and.b32 %r0, %r21, 31;"); // lane within the owning warp
        ptx.AppendLine("    mov.u32 %r23, %ctaid.x;");
        ptx.AppendLine($"    mad.lo.u32 %r1, %r23, {warps}, %r22;");
        ptx.AppendLine("    mov.u32 %r2, %ctaid.y;");
        ptx.AppendLine("    shr.u32 %r3, %r0, 2;"); // row group 0..7
        ptx.AppendLine("    and.b32 %r4, %r0, 3;"); // lane within row group
        ptx.AppendLine("    add.u32 %r5, %r3, 8;");
        ptx.AppendLine("    shl.b32 %r6, %r4, 1;");
        ptx.AppendLine("    shl.b32 %r7, %r1, 4;");
        ptx.AppendLine("    add.u32 %r8, %r7, %r3;"); // global query row 0
        ptx.AppendLine("    add.u32 %r9, %r8, 8;"); // global query row 1
        if (causalQueryOffset > 0)
        {
            ptx.AppendLine($"    add.u32 %r8, %r8, {causalQueryOffset};");
            ptx.AppendLine($"    add.u32 %r9, %r9, {causalQueryOffset};");
        }
        else if (causalQueryOffset < 0)
        {
            int leadingMaskedRows = -causalQueryOffset;
            ptx.AppendLine($"    setp.ge.u32 %p4, %r8, {leadingMaskedRows};");
            ptx.AppendLine($"    setp.ge.u32 %p5, %r9, {leadingMaskedRows};");
            ptx.AppendLine($"    sub.u32 %r8, %r8, {leadingMaskedRows};");
            ptx.AppendLine($"    sub.u32 %r9, %r9, {leadingMaskedRows};");
        }
        // Shared symbols are already addresses in the shared state space. PTX
        // cvta.to requires a generic-address register source, so use mov for
        // the statically allocated symbol and keep all following arithmetic u64.
        ptx.AppendLine("    mov.u64 %rd7, smem;");
        ptx.AppendLine("    mul.wide.u32 %rd29, %r22, 2048;");
        ptx.AppendLine("    add.u64 %rd28, %rd7, %rd29;"); // warp-private Q tile
        ptx.AppendLine($"    mul.wide.u32 %rd8, %r2, {queryBytesPerHead};");
        ptx.AppendLine("    add.u64 %rd0, %rd0, %rd8;");
        ptx.AppendLine($"    div.u32 %r24, %r2, {queryHeads};");
        ptx.AppendLine($"    mul.lo.u32 %r25, %r24, {queryHeads};");
        ptx.AppendLine("    sub.u32 %r25, %r2, %r25;");
        ptx.AppendLine($"    div.u32 %r25, %r25, {queriesPerKeyValue};");
        ptx.AppendLine($"    mad.lo.u32 %r26, %r24, {keyValueHeads}, %r25;");
        ptx.AppendLine($"    mul.wide.u32 %rd8, %r26, {keyValueBytesPerHead};");
        ptx.AppendLine("    add.u64 %rd1, %rd1, %rd8;");
        ptx.AppendLine("    add.u64 %rd2, %rd2, %rd8;");
        ptx.AppendLine($"    mul.wide.u32 %rd9, %r2, {outputBytesPerHead};");
        ptx.AppendLine("    add.u64 %rd5, %rd5, %rd9;");
        ptx.AppendLine($"    mul.wide.u32 %rd10, %r2, {statsBytesPerHead};");
        ptx.AppendLine("    add.u64 %rd6, %rd6, %rd10;");
        ptx.AppendLine("    mul.wide.u32 %rd11, %r1, 2048;");
        ptx.AppendLine("    add.u64 %rd12, %rd0, %rd11;"); // q tile global
        ptx.AppendLine("    mul.wide.u32 %rd13, %r1, 4096;");
        ptx.AppendLine("    add.u64 %rd5, %rd5, %rd13;"); // output tile global
        ptx.AppendLine("    mul.wide.u32 %rd14, %r1, 64;");
        ptx.AppendLine("    add.u64 %rd6, %rd6, %rd14;"); // stats tile global
        ptx.AppendLine();

        // Per-lane fragment address components reused for every tile.
        ptx.AppendLine("    mul.lo.u32 %r10, %r3, 128;");
        ptx.AppendLine("    shl.b32 %r11, %r4, 2;");
        ptx.AppendLine("    add.u32 %r10, %r10, %r11;"); // row0 Q pair / key-row pair
        ptx.AppendLine("    add.u32 %r11, %r10, 1024;"); // row1 Q / key+8
        ptx.AppendLine("    cvt.u64.u32 %rd25, %r10;");
        ptx.AppendLine("    cvt.u64.u32 %rd26, %r11;");

        // Initial Q and K/V tile 0 are one asynchronous group.
        EmitAsyncTileCopyFromAddress(ptx, "%rd12", "%rd28", 2048, "%r0", 32);
        EmitAsyncTileCopy(ptx, "%rd1", kShared0, 2048, "%r21", warps * 32);
        EmitAsyncTileCopy(ptx, "%rd2", vShared0, 2048, "%r21", warps * 32);
        if (fuseLayerNormGelu)
        {
            ptx.AppendLine("    setp.lt.u32 %p0, %r21, 16;");
            ptx.AppendLine("    mul.wide.u32 %rd15, %r21, 16;");
            ptx.AppendLine($"    add.u64 %rd16, %rd7, {gammaShared};");
            ptx.AppendLine("    add.u64 %rd16, %rd16, %rd15;");
            ptx.AppendLine("    add.u64 %rd17, %rd3, %rd15;");
            ptx.AppendLine("    @%p0 cp.async.ca.shared.global [%rd16], [%rd17], 16;");
            ptx.AppendLine($"    add.u64 %rd16, %rd7, {betaShared};");
            ptx.AppendLine("    add.u64 %rd16, %rd16, %rd15;");
            ptx.AppendLine("    add.u64 %rd17, %rd4, %rd15;");
            ptx.AppendLine("    @%p0 cp.async.ca.shared.global [%rd16], [%rd17], 16;");
        }
        ptx.AppendLine("    cp.async.commit_group;");
        ptx.AppendLine("    cp.async.wait_group 0;");
        ptx.AppendLine("    bar.sync 0;");
        ptx.AppendLine();

        ptx.AppendLine("    mov.f32 %f3, 0fFF800000;"); // running max row0
        ptx.AppendLine("    mov.f32 %f4, 0fFF800000;"); // running max row1
        ptx.AppendLine("    mov.f32 %f5, 0f00000000;"); // running sum row0
        ptx.AppendLine("    mov.f32 %f6, 0f00000000;"); // running sum row1
        for (int i = 0; i < 32; i++) ptx.AppendLine($"    mov.f32 %o{i}, 0f00000000;");
        ptx.AppendLine();

        for (int tile = 0; tile < keyValueTiles; tile++)
        {
            int currentK = (tile & 1) == 0 ? kShared0 : kShared1;
            int currentV = (tile & 1) == 0 ? vShared0 : vShared1;
            int nextK = (tile & 1) == 0 ? kShared1 : kShared0;
            int nextV = (tile & 1) == 0 ? vShared1 : vShared0;

            ptx.AppendLine($"    // ---- online K/V tile {tile} ----");
            if (tile + 1 < keyValueTiles)
            {
                int nextByteOffset = (tile + 1) * 2048;
                ptx.AppendLine($"    add.u64 %rd18, %rd1, {nextByteOffset};");
                ptx.AppendLine($"    add.u64 %rd19, %rd2, {nextByteOffset};");
                EmitAsyncTileCopy(ptx, "%rd18", nextK, 2048, "%r21", warps * 32);
                EmitAsyncTileCopy(ptx, "%rd19", nextV, 2048, "%r21", warps * 32);
                ptx.AppendLine("    cp.async.commit_group;");
            }

            if (isCausal && causalQueryOffset >= 0)
            {
                int skipTileOffset = (causalQueryOffset + KeyTileRows - 1) / KeyTileRows;
                if (skipTileOffset == 0)
                    ptx.AppendLine($"    setp.gt.u32 %p6, {tile}, %r1;");
                else
                {
                    ptx.AppendLine($"    add.u32 %r27, %r1, {skipTileOffset};");
                    ptx.AppendLine($"    setp.gt.u32 %p6, {tile}, %r27;");
                }
                ptx.AppendLine($"    @%p6 bra SKIP_CAUSAL_TILE_{tile};");
            }
            EmitScoreTile(ptx, currentK, tile, isCausal, scaleHex);
            EmitOnlineSoftmaxAndPv(ptx, currentV, log2E, firstTile: tile == 0);
            if (isCausal && causalQueryOffset >= 0)
                ptx.AppendLine($"SKIP_CAUSAL_TILE_{tile}:");

            if (tile + 1 < keyValueTiles)
            {
                ptx.AppendLine("    cp.async.wait_group 0;");
                ptx.AppendLine("    bar.sync 0;");
            }
            ptx.AppendLine();
        }

        // Normalize online accumulators and emit log-sum-exp statistics.
        ptx.AppendLine("    rcp.approx.f32 %f14, %f5;");
        ptx.AppendLine("    rcp.approx.f32 %f15, %f6;");
        for (int n = 0; n < 8; n++)
        {
            int o = n * 4;
            ptx.AppendLine($"    mul.rn.f32 %o{o}, %o{o}, %f14;");
            ptx.AppendLine($"    mul.rn.f32 %o{o + 1}, %o{o + 1}, %f14;");
            ptx.AppendLine($"    mul.rn.f32 %o{o + 2}, %o{o + 2}, %f15;");
            ptx.AppendLine($"    mul.rn.f32 %o{o + 3}, %o{o + 3}, %f15;");
        }
        if (causalQueryOffset < 0)
        {
            for (int n = 0; n < 8; n++)
            {
                int o = n * 4;
                ptx.AppendLine($"    @!%p4 mov.f32 %o{o}, 0f00000000;");
                ptx.AppendLine($"    @!%p4 mov.f32 %o{o + 1}, 0f00000000;");
                ptx.AppendLine($"    @!%p5 mov.f32 %o{o + 2}, 0f00000000;");
                ptx.AppendLine($"    @!%p5 mov.f32 %o{o + 3}, 0f00000000;");
            }
        }
        if (emitSoftmaxStats)
        {
            ptx.AppendLine("    setp.eq.u32 %p0, %r4, 0;");
            ptx.AppendLine("    lg2.approx.f32 %f16, %f5;");
            ptx.AppendLine($"    fma.rn.f32 %f16, %f16, {ln2}, %f3;");
            ptx.AppendLine("    lg2.approx.f32 %f17, %f6;");
            ptx.AppendLine($"    fma.rn.f32 %f17, %f17, {ln2}, %f4;");
            if (causalQueryOffset < 0)
            {
                ptx.AppendLine("    @!%p4 mov.f32 %f16, 0fFF800000;");
                ptx.AppendLine("    @!%p5 mov.f32 %f17, 0fFF800000;");
            }
            ptx.AppendLine("    mul.wide.u32 %rd18, %r3, 4;");
            ptx.AppendLine("    add.u64 %rd18, %rd6, %rd18;");
            ptx.AppendLine("    @%p0 st.global.f32 [%rd18], %f16;");
            ptx.AppendLine("    @%p0 st.global.f32 [%rd18+32], %f17;");
        }

        if (fuseLayerNormGelu)
            EmitLayerNormGelu(ptx, gammaShared, betaShared, inv64, epsilonHex, geluA, geluB);

        EmitOutputStores(ptx);
        ptx.AppendLine("    ret;");
        ptx.AppendLine("}");
        return ptx.ToString();
    }

    private static void EmitAsyncTileCopy(
        StringBuilder ptx,
        string globalBase,
        int sharedBase,
        int bytes,
        string threadRegister,
        int copyThreads)
    {
        int vectors = bytes / 16;
        int participatingThreads = Math.Min(copyThreads, vectors);
        int passes = vectors / participatingThreads;
        bool predicateCopy = copyThreads > participatingThreads;
        if (predicateCopy)
            ptx.AppendLine($"    setp.lt.u32 %p7, {threadRegister}, {participatingThreads};");
        ptx.AppendLine($"    mul.wide.u32 %rd20, {threadRegister}, 16;");
        for (int pass = 0; pass < passes; pass++)
        {
            int offset = pass * participatingThreads * 16;
            ptx.AppendLine($"    add.u64 %rd21, %rd7, {sharedBase + offset};");
            ptx.AppendLine("    add.u64 %rd21, %rd21, %rd20;");
            if (offset == 0)
                ptx.AppendLine($"    add.u64 %rd22, {globalBase}, %rd20;");
            else
            {
                ptx.AppendLine($"    add.u64 %rd22, {globalBase}, {offset};");
                ptx.AppendLine("    add.u64 %rd22, %rd22, %rd20;");
            }
            ptx.AppendLine(predicateCopy
                ? "    @%p7 cp.async.ca.shared.global [%rd21], [%rd22], 16;"
                : "    cp.async.ca.shared.global [%rd21], [%rd22], 16;");
        }
    }

    private static void EmitAsyncTileCopyFromAddress(
        StringBuilder ptx,
        string globalBase,
        string sharedBase,
        int bytes,
        string threadRegister,
        int copyThreads)
    {
        int passes = bytes / (copyThreads * 16);
        ptx.AppendLine($"    mul.wide.u32 %rd20, {threadRegister}, 16;");
        for (int pass = 0; pass < passes; pass++)
        {
            int offset = pass * copyThreads * 16;
            if (offset == 0)
                ptx.AppendLine($"    add.u64 %rd21, {sharedBase}, %rd20;");
            else
            {
                ptx.AppendLine($"    add.u64 %rd21, {sharedBase}, {offset};");
                ptx.AppendLine("    add.u64 %rd21, %rd21, %rd20;");
            }
            if (offset == 0)
                ptx.AppendLine($"    add.u64 %rd22, {globalBase}, %rd20;");
            else
            {
                ptx.AppendLine($"    add.u64 %rd22, {globalBase}, {offset};");
                ptx.AppendLine("    add.u64 %rd22, %rd22, %rd20;");
            }
            ptx.AppendLine("    cp.async.ca.shared.global [%rd21], [%rd22], 16;");
        }
    }

    private static void EmitScoreTile(
        StringBuilder ptx,
        int kShared,
        int tile,
        bool isCausal,
        string scaleHex)
    {
        for (int i = 0; i < 8; i++) ptx.AppendLine($"    mov.f32 %s{i}, 0f00000000;");

        for (int chunk = 0; chunk < HeadDimension; chunk += 16)
        {
            int chunkBytes = chunk * sizeof(ushort);
            ptx.AppendLine($"    add.u64 %rd23, %rd28, {chunkBytes};");
            ptx.AppendLine("    add.u64 %rd23, %rd23, %rd25;");
            ptx.AppendLine("    ld.shared.b32 %a0, [%rd23];");
            ptx.AppendLine("    ld.shared.b32 %a2, [%rd23+16];");
            ptx.AppendLine($"    add.u64 %rd24, %rd28, {chunkBytes};");
            ptx.AppendLine("    add.u64 %rd24, %rd24, %rd26;");
            ptx.AppendLine("    ld.shared.b32 %a1, [%rd24];");
            ptx.AppendLine("    ld.shared.b32 %a3, [%rd24+16];");

            ptx.AppendLine($"    add.u64 %rd23, %rd7, {kShared + chunkBytes};");
            ptx.AppendLine("    add.u64 %rd23, %rd23, %rd25;");
            ptx.AppendLine("    ld.shared.b32 %b0, [%rd23];");
            ptx.AppendLine("    ld.shared.b32 %b1, [%rd23+16];");
            ptx.AppendLine("    mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 " +
                "{%s0,%s1,%s2,%s3}, {%a0,%a1,%a2,%a3}, {%b0,%b1}, {%s0,%s1,%s2,%s3};");

            ptx.AppendLine($"    add.u64 %rd24, %rd7, {kShared + chunkBytes};");
            ptx.AppendLine("    add.u64 %rd24, %rd24, %rd26;");
            ptx.AppendLine("    ld.shared.b32 %b2, [%rd24];");
            ptx.AppendLine("    ld.shared.b32 %b3, [%rd24+16];");
            ptx.AppendLine("    mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 " +
                "{%s4,%s5,%s6,%s7}, {%a0,%a1,%a2,%a3}, {%b2,%b3}, {%s4,%s5,%s6,%s7};");
        }

        for (int i = 0; i < 8; i++)
            ptx.AppendLine($"    mul.rn.f32 %s{i}, %s{i}, {scaleHex};");

        if (isCausal)
        {
            int tileBase = tile * KeyTileRows;
            ptx.AppendLine("    shl.b32 %r12, %r4, 1;");
            if (tileBase != 0) ptx.AppendLine($"    add.u32 %r12, %r12, {tileBase};");
            ptx.AppendLine("    add.u32 %r13, %r12, 1;");
            ptx.AppendLine("    add.u32 %r14, %r12, 8;");
            ptx.AppendLine("    add.u32 %r15, %r12, 9;");
            EmitCausalMask(ptx, "%s0", "%r12", "%r8", "%p0");
            EmitCausalMask(ptx, "%s1", "%r13", "%r8", "%p0");
            EmitCausalMask(ptx, "%s4", "%r14", "%r8", "%p0");
            EmitCausalMask(ptx, "%s5", "%r15", "%r8", "%p0");
            EmitCausalMask(ptx, "%s2", "%r12", "%r9", "%p0");
            EmitCausalMask(ptx, "%s3", "%r13", "%r9", "%p0");
            EmitCausalMask(ptx, "%s6", "%r14", "%r9", "%p0");
            EmitCausalMask(ptx, "%s7", "%r15", "%r9", "%p0");
        }
    }

    private static void EmitCausalMask(
        StringBuilder ptx, string score, string key, string query, string predicate)
    {
        ptx.AppendLine($"    setp.gt.u32 {predicate}, {key}, {query};");
        ptx.AppendLine($"    @{predicate} mov.f32 {score}, 0fFF800000;");
    }

    private static void EmitOnlineSoftmaxAndPv(
        StringBuilder ptx, int vShared, string log2E, bool firstTile)
    {
        // Each four-lane group owns two rows. It collectively holds 16 scores
        // per row across the two m16n8 score fragments.
        ptx.AppendLine("    max.f32 %f0, %s0, %s1;");
        ptx.AppendLine("    max.f32 %f0, %f0, %s4;");
        ptx.AppendLine("    max.f32 %f0, %f0, %s5;");
        ptx.AppendLine("    max.f32 %f1, %s2, %s3;");
        ptx.AppendLine("    max.f32 %f1, %f1, %s6;");
        ptx.AppendLine("    max.f32 %f1, %f1, %s7;");
        EmitGroup4Reduction(ptx, "max.f32", "%f0");
        EmitGroup4Reduction(ptx, "max.f32", "%f1");
        ptx.AppendLine("    max.f32 %f7, %f3, %f0;");
        ptx.AppendLine("    max.f32 %f8, %f4, %f1;");
        ptx.AppendLine("    sub.rn.f32 %f9, %f3, %f7;");
        ptx.AppendLine($"    mul.rn.f32 %f9, %f9, {log2E};");
        ptx.AppendLine("    ex2.approx.f32 %f9, %f9;");
        ptx.AppendLine("    sub.rn.f32 %f10, %f4, %f8;");
        ptx.AppendLine($"    mul.rn.f32 %f10, %f10, {log2E};");
        ptx.AppendLine("    ex2.approx.f32 %f10, %f10;");

        foreach (int index in new[] { 0, 1, 4, 5 })
            EmitExponent(ptx, index, "%f7", log2E);
        foreach (int index in new[] { 2, 3, 6, 7 })
            EmitExponent(ptx, index, "%f8", log2E);

        ptx.AppendLine("    add.rn.f32 %f11, %s0, %s1;");
        ptx.AppendLine("    add.rn.f32 %f11, %f11, %s4;");
        ptx.AppendLine("    add.rn.f32 %f11, %f11, %s5;");
        ptx.AppendLine("    add.rn.f32 %f12, %s2, %s3;");
        ptx.AppendLine("    add.rn.f32 %f12, %f12, %s6;");
        ptx.AppendLine("    add.rn.f32 %f12, %f12, %s7;");
        EmitGroup4Reduction(ptx, "add.rn.f32", "%f11");
        EmitGroup4Reduction(ptx, "add.rn.f32", "%f12");
        ptx.AppendLine("    fma.rn.f32 %f5, %f5, %f9, %f11;");
        ptx.AppendLine("    fma.rn.f32 %f6, %f6, %f10, %f12;");
        ptx.AppendLine("    mov.f32 %f3, %f7;");
        ptx.AppendLine("    mov.f32 %f4, %f8;");

        // The first tile has a zero prior accumulator. On later tiles a row's
        // maximum usually does not change, so predicate away its 16 register
        // rescale operations when exp(oldMax-newMax) is exactly one.
        if (!firstTile)
        {
            ptx.AppendLine("    setp.neu.f32 %p2, %f9, 0f3F800000;");
            ptx.AppendLine("    setp.neu.f32 %p3, %f10, 0f3F800000;");
            for (int n = 0; n < 8; n++)
            {
                int o = n * 4;
                ptx.AppendLine($"    @%p2 mul.rn.f32 %o{o}, %o{o}, %f9;");
                ptx.AppendLine($"    @%p2 mul.rn.f32 %o{o + 1}, %o{o + 1}, %f9;");
                ptx.AppendLine($"    @%p3 mul.rn.f32 %o{o + 2}, %o{o + 2}, %f10;");
                ptx.AppendLine($"    @%p3 mul.rn.f32 %o{o + 3}, %o{o + 3}, %f10;");
            }
        }

        // The score-fragment lane layout is exactly the row-major A operand
        // layout for the P*V MMA, so conversion requires no shared scratch.
        EmitPackHalf2(ptx, "%a0", "%s0", "%s1");
        EmitPackHalf2(ptx, "%a1", "%s2", "%s3");
        EmitPackHalf2(ptx, "%a2", "%s4", "%s5");
        EmitPackHalf2(ptx, "%a3", "%s6", "%s7");

        // B is V[16,64] interpreted as a column-major [K,N] operand. Load
        // the four lane-owned values and pack them into the two f16x2 regs.
        ptx.AppendLine("    shl.b32 %r16, %r4, 1;");
        ptx.AppendLine("    mul.lo.u32 %r16, %r16, 128;");
        ptx.AppendLine("    shl.b32 %r17, %r3, 1;");
        ptx.AppendLine("    add.u32 %r16, %r16, %r17;");
        ptx.AppendLine("    cvt.u64.u32 %rd27, %r16;");
        for (int n = 0; n < 8; n++)
        {
            int colBytes = n * 16;
            int o = n * 4;
            ptx.AppendLine($"    add.u64 %rd23, %rd7, {vShared + colBytes};");
            ptx.AppendLine("    add.u64 %rd23, %rd23, %rd27;");
            ptx.AppendLine("    ld.shared.u16 %h0, [%rd23];");
            ptx.AppendLine("    ld.shared.u16 %h1, [%rd23+128];");
            ptx.AppendLine("    mov.b32 %b0, {%h0,%h1};");
            ptx.AppendLine("    ld.shared.u16 %h2, [%rd23+1024];");
            ptx.AppendLine("    ld.shared.u16 %h3, [%rd23+1152];");
            ptx.AppendLine("    mov.b32 %b1, {%h2,%h3};");
            ptx.AppendLine("    mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 " +
                $"{{%o{o},%o{o + 1},%o{o + 2},%o{o + 3}}}, " +
                "{%a0,%a1,%a2,%a3}, {%b0,%b1}, " +
                $"{{%o{o},%o{o + 1},%o{o + 2},%o{o + 3}}};");
        }
    }

    private static void EmitExponent(StringBuilder ptx, int score, string maximum, string log2E)
    {
        ptx.AppendLine($"    sub.rn.f32 %s{score}, %s{score}, {maximum};");
        ptx.AppendLine($"    mul.rn.f32 %s{score}, %s{score}, {log2E};");
        ptx.AppendLine($"    ex2.approx.f32 %s{score}, %s{score};");
    }

    private static void EmitGroup4Reduction(StringBuilder ptx, string operation, string accumulator)
    {
        foreach (int delta in new[] { 1, 2 })
        {
            ptx.AppendLine($"    mov.b32 %r30, {accumulator};");
            ptx.AppendLine($"    shfl.sync.bfly.b32 %r31, %r30, {delta}, 31, 0xffffffff;");
            ptx.AppendLine("    mov.b32 %f2, %r31;");
            ptx.AppendLine($"    {operation} {accumulator}, {accumulator}, %f2;");
        }
    }

    private static void EmitPackHalf2(StringBuilder ptx, string packed, string low, string high)
    {
        ptx.AppendLine($"    cvt.rn.f16.f32 %h0, {low};");
        ptx.AppendLine($"    cvt.rn.f16.f32 %h1, {high};");
        ptx.AppendLine($"    mov.b32 {packed}, {{%h0,%h1}};");
    }

    private static void EmitLayerNormGelu(
        StringBuilder ptx,
        int gammaShared,
        int betaShared,
        string inv64,
        string epsilon,
        string geluA,
        string geluB)
    {
        ptx.AppendLine("    // Head-local LayerNorm(D=64) + tanh-GELU epilogue.");
        ptx.AppendLine("    mov.f32 %f16, 0f00000000;");
        ptx.AppendLine("    mov.f32 %f17, 0f00000000;");
        ptx.AppendLine("    mov.f32 %f18, 0f00000000;");
        ptx.AppendLine("    mov.f32 %f19, 0f00000000;");
        for (int n = 0; n < 8; n++)
        {
            int o = n * 4;
            ptx.AppendLine($"    add.rn.f32 %f16, %f16, %o{o};");
            ptx.AppendLine($"    add.rn.f32 %f16, %f16, %o{o + 1};");
            ptx.AppendLine($"    add.rn.f32 %f17, %f17, %o{o + 2};");
            ptx.AppendLine($"    add.rn.f32 %f17, %f17, %o{o + 3};");
            ptx.AppendLine($"    fma.rn.f32 %f18, %o{o}, %o{o}, %f18;");
            ptx.AppendLine($"    fma.rn.f32 %f18, %o{o + 1}, %o{o + 1}, %f18;");
            ptx.AppendLine($"    fma.rn.f32 %f19, %o{o + 2}, %o{o + 2}, %f19;");
            ptx.AppendLine($"    fma.rn.f32 %f19, %o{o + 3}, %o{o + 3}, %f19;");
        }
        EmitGroup4Reduction(ptx, "add.rn.f32", "%f16");
        EmitGroup4Reduction(ptx, "add.rn.f32", "%f17");
        EmitGroup4Reduction(ptx, "add.rn.f32", "%f18");
        EmitGroup4Reduction(ptx, "add.rn.f32", "%f19");
        ptx.AppendLine($"    mul.rn.f32 %f20, %f16, {inv64};");
        ptx.AppendLine($"    mul.rn.f32 %f21, %f17, {inv64};");
        ptx.AppendLine($"    mul.rn.f32 %f22, %f18, {inv64};");
        ptx.AppendLine($"    mul.rn.f32 %f23, %f19, {inv64};");
        ptx.AppendLine("    mul.rn.f32 %f28, %f20, %f20;");
        ptx.AppendLine("    sub.rn.f32 %f22, %f22, %f28;");
        ptx.AppendLine("    mul.rn.f32 %f28, %f21, %f21;");
        ptx.AppendLine("    sub.rn.f32 %f23, %f23, %f28;");
        ptx.AppendLine($"    add.rn.f32 %f22, %f22, {epsilon};");
        ptx.AppendLine($"    add.rn.f32 %f23, %f23, {epsilon};");
        ptx.AppendLine("    rsqrt.approx.f32 %f22, %f22;");
        ptx.AppendLine("    rsqrt.approx.f32 %f23, %f23;");

        ptx.AppendLine("    shl.b32 %r18, %r4, 3;");
        ptx.AppendLine("    cvt.u64.u32 %rd27, %r18;");
        for (int n = 0; n < 8; n++)
        {
            int parameterBytes = n * 32;
            int o = n * 4;
            ptx.AppendLine($"    add.u64 %rd23, %rd7, {gammaShared + parameterBytes};");
            ptx.AppendLine("    add.u64 %rd23, %rd23, %rd27;");
            ptx.AppendLine("    ld.shared.f32 %f24, [%rd23];");
            ptx.AppendLine("    ld.shared.f32 %f25, [%rd23+4];");
            ptx.AppendLine($"    add.u64 %rd24, %rd7, {betaShared + parameterBytes};");
            ptx.AppendLine("    add.u64 %rd24, %rd24, %rd27;");
            ptx.AppendLine("    ld.shared.f32 %f26, [%rd24];");
            ptx.AppendLine("    ld.shared.f32 %f27, [%rd24+4];");
            EmitNormalizeAffineGelu(ptx, $"%o{o}", "%f20", "%f22", "%f24", "%f26", geluA, geluB);
            EmitNormalizeAffineGelu(ptx, $"%o{o + 1}", "%f20", "%f22", "%f25", "%f27", geluA, geluB);
            EmitNormalizeAffineGelu(ptx, $"%o{o + 2}", "%f21", "%f23", "%f24", "%f26", geluA, geluB);
            EmitNormalizeAffineGelu(ptx, $"%o{o + 3}", "%f21", "%f23", "%f25", "%f27", geluA, geluB);
        }
    }

    private static void EmitNormalizeAffineGelu(
        StringBuilder ptx,
        string value,
        string mean,
        string inverseStd,
        string gamma,
        string beta,
        string geluA,
        string geluB)
    {
        ptx.AppendLine($"    sub.rn.f32 {value}, {value}, {mean};");
        ptx.AppendLine($"    mul.rn.f32 {value}, {value}, {inverseStd};");
        ptx.AppendLine($"    fma.rn.f32 {value}, {value}, {gamma}, {beta};");
        ptx.AppendLine($"    mul.rn.f32 %f28, {value}, {value};");
        ptx.AppendLine($"    mul.rn.f32 %f28, %f28, {value};");
        ptx.AppendLine($"    fma.rn.f32 %f28, %f28, {geluB}, {value};");
        ptx.AppendLine($"    mul.rn.f32 %f28, %f28, {geluA};");
        ptx.AppendLine("    tanh.approx.f32 %f28, %f28;");
        ptx.AppendLine("    add.rn.f32 %f28, %f28, 0f3F800000;");
        ptx.AppendLine($"    mul.rn.f32 %f28, %f28, {value};");
        ptx.AppendLine($"    mul.rn.f32 {value}, %f28, 0f3F000000;");
    }

    private static void EmitOutputStores(StringBuilder ptx)
    {
        ptx.AppendLine("    mul.lo.u32 %r19, %r3, 256;");
        ptx.AppendLine("    shl.b32 %r20, %r4, 3;");
        ptx.AppendLine("    add.u32 %r19, %r19, %r20;");
        ptx.AppendLine("    cvt.u64.u32 %rd27, %r19;");
        for (int n = 0; n < 8; n++)
        {
            int colBytes = n * 32;
            int o = n * 4;
            ptx.AppendLine($"    add.u64 %rd23, %rd5, {colBytes};");
            ptx.AppendLine("    add.u64 %rd23, %rd23, %rd27;");
            ptx.AppendLine($"    st.global.v2.f32 [%rd23], {{%o{o},%o{o + 1}}};");
            ptx.AppendLine($"    st.global.v2.f32 [%rd23+2048], {{%o{o + 2},%o{o + 3}}};");
        }
    }

    private static string FloatLiteral(float value) =>
        "0f" + BitConverter.SingleToInt32Bits(value).ToString("X8", CultureInfo.InvariantCulture);
}
#endif
