using System;
using System.Collections.Generic;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// Decode-token FP32 QKV projection + bias + interleaved RoPE + dense KV-cache
/// write. Packed Q/K/V and rotated K/V intermediates never exist in global
/// memory. One warp owns one adjacent output-dimension pair for one head.
/// </summary>
internal sealed class PtxFusedQkvRopeCacheD64Kernel : IDisposable
{
    internal const int HeadDimension = 64;
    internal const int WarpsPerBlock = 4;
    internal const string EntryPoint = "aidotnet_qkv_rope_cache_d64";

    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

    internal int Heads { get; }
    internal int ModelDimension { get; }
    internal int CacheCapacity { get; }
    internal int Position { get; }
    internal string Ptx { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal PtxFusedQkvRopeCacheD64Kernel(
        DirectPtxRuntime runtime,
        int heads,
        int cacheCapacity,
        int position)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        if (!DirectPtxArchitecture.HasValidatedQkvRopeCache(
            runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new PlatformNotSupportedException(
                "The checked-in fused QKV/RoPE/cache specialization is validated only on SM86.");
        ValidateShape(heads, cacheCapacity, position);

        Heads = heads;
        ModelDimension = checked(heads * HeadDimension);
        CacheCapacity = cacheCapacity;
        Position = position;
        Blueprint = CreateBlueprint(runtime.ArchitectureFamily, heads, cacheCapacity, position);
        Ptx = EmitPtx(
            runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor,
            heads, cacheCapacity, position);
        _module = runtime.LoadModule(Ptx);
        _function = _module.GetFunction(EntryPoint, out DirectPtxFunctionInfo info);
        int blockThreads = WarpsPerBlock * 32;
        int activeBlocks = _module.GetActiveBlocksPerMultiprocessor(_function, blockThreads);
        Blueprint.ResourceBudget.Validate(EntryPoint, info, blockThreads, activeBlocks);
        Audit = DirectPtxKernelAudit.Create(
            Blueprint, runtime.DeviceFingerprint, Ptx, info,
            blockThreads, activeBlocks, _module.JitInfoLog);
    }

    internal unsafe void Launch(
        DirectPtxTensorView input,
        DirectPtxTensorView packedWeights,
        DirectPtxTensorView bias,
        DirectPtxTensorView cosine,
        DirectPtxTensorView sine,
        DirectPtxTensorView query,
        DirectPtxTensorView keyCache,
        DirectPtxTensorView valueCache)
    {
        Require(input, Blueprint.Tensors[0], nameof(input));
        Require(packedWeights, Blueprint.Tensors[1], nameof(packedWeights));
        Require(bias, Blueprint.Tensors[2], nameof(bias));
        Require(cosine, Blueprint.Tensors[3], nameof(cosine));
        Require(sine, Blueprint.Tensors[4], nameof(sine));
        Require(query, Blueprint.Tensors[5], nameof(query));
        Require(keyCache, Blueprint.Tensors[6], nameof(keyCache));
        Require(valueCache, Blueprint.Tensors[7], nameof(valueCache));
        RejectOutputAliasing(input, packedWeights, bias, cosine, sine, query, keyCache, valueCache);

        IntPtr inputPointer = input.Pointer;
        IntPtr weightPointer = packedWeights.Pointer;
        IntPtr biasPointer = bias.Pointer;
        IntPtr cosinePointer = cosine.Pointer;
        IntPtr sinePointer = sine.Pointer;
        IntPtr queryPointer = query.Pointer;
        IntPtr keyCachePointer = keyCache.Pointer;
        IntPtr valueCachePointer = valueCache.Pointer;
        void** arguments = stackalloc void*[8];
        arguments[0] = &inputPointer;
        arguments[1] = &weightPointer;
        arguments[2] = &biasPointer;
        arguments[3] = &cosinePointer;
        arguments[4] = &sinePointer;
        arguments[5] = &queryPointer;
        arguments[6] = &keyCachePointer;
        arguments[7] = &valueCachePointer;
        int outputPairs = checked(Heads * (HeadDimension / 2));
        _module.Launch(
            _function,
            (uint)((outputPairs + WarpsPerBlock - 1) / WarpsPerBlock), 1, 1,
            (uint)(WarpsPerBlock * 32), 1, 1, 0, arguments);
    }

    public void Dispose() => _module.Dispose();

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

    private static void RejectOutputAliasing(
        DirectPtxTensorView input,
        DirectPtxTensorView packedWeights,
        DirectPtxTensorView bias,
        DirectPtxTensorView cosine,
        DirectPtxTensorView sine,
        DirectPtxTensorView query,
        DirectPtxTensorView keyCache,
        DirectPtxTensorView valueCache)
    {
        if (Overlaps(query, keyCache) || Overlaps(query, valueCache) ||
            Overlaps(keyCache, valueCache) ||
            IsInput(query) || IsInput(keyCache) || IsInput(valueCache))
            throw new ArgumentException("QKV/RoPE/cache outputs may not alias inputs or each other.");

        bool IsInput(DirectPtxTensorView output) =>
            Overlaps(output, input) || Overlaps(output, packedWeights) ||
            Overlaps(output, bias) || Overlaps(output, cosine) || Overlaps(output, sine);

        static bool Overlaps(DirectPtxTensorView left, DirectPtxTensorView right)
        {
            nuint leftStart = PtxCompat.ToNuint(left.Pointer);
            nuint rightStart = PtxCompat.ToNuint(right.Pointer);
            nuint leftEnd = checked(leftStart + left.ByteLength);
            nuint rightEnd = checked(rightStart + right.ByteLength);
            return leftStart < rightEnd && rightStart < leftEnd;
        }
    }

    internal static string EmitPtx(
        int ccMajor,
        int ccMinor,
        int heads,
        int cacheCapacity,
        int position)
    {
        ValidateShape(heads, cacheCapacity, position);
        int modelDimension = checked(heads * HeadDimension);
        int outputPairs = checked(heads * (HeadDimension / 2));
        int weightRowBytes = checked(modelDimension * sizeof(float));
        int keyPlaneOffset = modelDimension;
        int valuePlaneOffset = checked(2 * modelDimension);
        int cacheRowElements = checked(heads * HeadDimension);

        var ptx = new StringBuilder(32_000);
        ptx.AppendLine(".version 7.1");
        ptx.AppendLine($".target sm_{ccMajor}{ccMinor}");
        ptx.AppendLine(".address_size 64");
        ptx.AppendLine();
        ptx.AppendLine($".visible .entry {EntryPoint}(");
        ptx.AppendLine("    .param .u64 input_ptr,");
        ptx.AppendLine("    .param .u64 packed_weights_ptr,");
        ptx.AppendLine("    .param .u64 bias_ptr,");
        ptx.AppendLine("    .param .u64 cosine_ptr,");
        ptx.AppendLine("    .param .u64 sine_ptr,");
        ptx.AppendLine("    .param .u64 query_ptr,");
        ptx.AppendLine("    .param .u64 key_cache_ptr,");
        ptx.AppendLine("    .param .u64 value_cache_ptr");
        ptx.AppendLine(")");
        ptx.AppendLine("{");
        ptx.AppendLine("    .reg .pred %p<3>;");
        ptx.AppendLine("    .reg .b32 %r<24>;");
        ptx.AppendLine("    .reg .b64 %rd<32>;");
        ptx.AppendLine("    .reg .f32 %f<32>;");
        for (int i = 0; i < 8; i++)
            ptx.AppendLine($"    ld.param.u64 %rd{i}, [{ParameterName(i)}];");
        ptx.AppendLine("    mov.u32 %r0, %tid.x;");
        ptx.AppendLine("    and.b32 %r1, %r0, 31;");
        ptx.AppendLine("    shr.u32 %r2, %r0, 5;");
        ptx.AppendLine("    mov.u32 %r3, %ctaid.x;");
        ptx.AppendLine($"    mad.lo.u32 %r4, %r3, {WarpsPerBlock}, %r2;");
        ptx.AppendLine($"    setp.ge.u32 %p0, %r4, {outputPairs};");
        ptx.AppendLine("    @%p0 bra.uni QKV_RETURN;");
        ptx.AppendLine("    shr.u32 %r5, %r4, 5;");
        ptx.AppendLine("    and.b32 %r6, %r4, 31;");
        ptx.AppendLine("    shl.b32 %r7, %r6, 1;");
        ptx.AppendLine($"    mad.lo.u32 %r8, %r5, {HeadDimension}, %r7;");
        ptx.AppendLine($"    add.u32 %r9, %r8, {keyPlaneOffset};");
        ptx.AppendLine($"    add.u32 %r10, %r8, {valuePlaneOffset};");
        ptx.AppendLine("    mul.wide.u32 %rd8, %r1, 4;");
        ptx.AppendLine("    add.u64 %rd9, %rd0, %rd8;");
        EmitWeightRowPointer(ptx, "%rd10", "%rd1", "%r8", weightRowBytes, 0);
        EmitWeightRowPointer(ptx, "%rd11", "%rd1", "%r8", weightRowBytes, 1);
        EmitWeightRowPointer(ptx, "%rd12", "%rd1", "%r9", weightRowBytes, 0);
        EmitWeightRowPointer(ptx, "%rd13", "%rd1", "%r9", weightRowBytes, 1);
        EmitWeightRowPointer(ptx, "%rd14", "%rd1", "%r10", weightRowBytes, 0);
        EmitWeightRowPointer(ptx, "%rd15", "%rd1", "%r10", weightRowBytes, 1);
        for (int f = 0; f < 6; f++)
            ptx.AppendLine($"    mov.f32 %f{f}, 0f00000000;");

        for (int inputOffset = 0; inputOffset < modelDimension; inputOffset += 32)
        {
            int byteOffset = checked(inputOffset * sizeof(float));
            string suffix = byteOffset == 0 ? string.Empty : $"+{byteOffset}";
            ptx.AppendLine($"    ld.global.f32 %f6, [%rd9{suffix}];");
            for (int projection = 0; projection < 6; projection++)
            {
                ptx.AppendLine($"    ld.global.f32 %f7, [%rd{10 + projection}{suffix}];");
                ptx.AppendLine($"    fma.rn.f32 %f{projection}, %f6, %f7, %f{projection};");
            }
        }

        for (int f = 0; f < 6; f++) EmitWarpSum(ptx, $"%f{f}", "%f8", "%r20", "%r21");
        ptx.AppendLine("    setp.ne.u32 %p1, %r1, 0;");
        // Only lane zero owns the bias/RoPE/final-store epilogue. This branch
        // is deliberately divergent and therefore must not carry PTX's .uni
        // promise (the earlier output-pair bounds branch is warp-uniform).
        ptx.AppendLine("    @%p1 bra QKV_RETURN;");
        EmitBiasPair(ptx, "%r8", "%f0", "%f1");
        EmitBiasPair(ptx, "%r9", "%f2", "%f3");
        EmitBiasPair(ptx, "%r10", "%f4", "%f5");
        int ropePairBase = checked(position * (HeadDimension / 2));
        ptx.AppendLine($"    add.u32 %r11, %r6, {ropePairBase};");
        ptx.AppendLine("    mul.wide.u32 %rd16, %r11, 4;");
        ptx.AppendLine("    add.u64 %rd17, %rd3, %rd16;");
        ptx.AppendLine("    add.u64 %rd18, %rd4, %rd16;");
        ptx.AppendLine("    ld.global.f32 %f9, [%rd17];");
        ptx.AppendLine("    ld.global.f32 %f10, [%rd18];");
        ptx.AppendLine("    neg.f32 %f15, %f10;");
        EmitRotation(ptx, "%f0", "%f1", "%f11", "%f12");
        EmitRotation(ptx, "%f2", "%f3", "%f13", "%f14");
        ptx.AppendLine("    mul.wide.u32 %rd19, %r8, 4;");
        ptx.AppendLine("    add.u64 %rd20, %rd5, %rd19;");
        ptx.AppendLine("    st.global.v2.f32 [%rd20], {%f11,%f12};");
        int cachePositionOffset = checked(position * cacheRowElements);
        ptx.AppendLine($"    add.u32 %r12, %r8, {cachePositionOffset};");
        ptx.AppendLine("    mul.wide.u32 %rd21, %r12, 4;");
        ptx.AppendLine("    add.u64 %rd22, %rd6, %rd21;");
        ptx.AppendLine("    add.u64 %rd23, %rd7, %rd21;");
        ptx.AppendLine("    st.global.v2.f32 [%rd22], {%f13,%f14};");
        ptx.AppendLine("    st.global.v2.f32 [%rd23], {%f4,%f5};");
        ptx.AppendLine("QKV_RETURN:");
        ptx.AppendLine("    ret;");
        ptx.AppendLine("}");
        return ptx.ToString();

        static string ParameterName(int index) => index switch
        {
            0 => "input_ptr",
            1 => "packed_weights_ptr",
            2 => "bias_ptr",
            3 => "cosine_ptr",
            4 => "sine_ptr",
            5 => "query_ptr",
            6 => "key_cache_ptr",
            _ => "value_cache_ptr"
        };
    }

    private static void EmitWeightRowPointer(
        StringBuilder ptx,
        string destination,
        string weightBase,
        string outputIndex,
        int weightRowBytes,
        int nextRow)
    {
        ptx.AppendLine($"    mul.wide.u32 %rd24, {outputIndex}, {weightRowBytes};");
        ptx.AppendLine($"    add.u64 {destination}, {weightBase}, %rd24;");
        if (nextRow != 0) ptx.AppendLine($"    add.u64 {destination}, {destination}, {weightRowBytes};");
        ptx.AppendLine($"    add.u64 {destination}, {destination}, %rd8;");
    }

    private static void EmitBiasPair(
        StringBuilder ptx,
        string outputIndex,
        string even,
        string odd)
    {
        ptx.AppendLine($"    mul.wide.u32 %rd24, {outputIndex}, 4;");
        ptx.AppendLine("    add.u64 %rd25, %rd2, %rd24;");
        ptx.AppendLine("    ld.global.v2.f32 {%f7,%f8}, [%rd25];");
        ptx.AppendLine($"    add.rn.f32 {even}, {even}, %f7;");
        ptx.AppendLine($"    add.rn.f32 {odd}, {odd}, %f8;");
    }

    private static void EmitRotation(
        StringBuilder ptx,
        string even,
        string odd,
        string rotatedEven,
        string rotatedOdd)
    {
        ptx.AppendLine($"    mul.rn.f32 {rotatedEven}, {even}, %f9;");
        ptx.AppendLine($"    fma.rn.f32 {rotatedEven}, {odd}, %f15, {rotatedEven};");
        ptx.AppendLine($"    mul.rn.f32 {rotatedOdd}, {even}, %f10;");
        ptx.AppendLine($"    fma.rn.f32 {rotatedOdd}, {odd}, %f9, {rotatedOdd};");
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
        int heads,
        int cacheCapacity,
        int position)
    {
        int modelDimension = checked(heads * HeadDimension);
        var input = new DirectPtxExtent(modelDimension);
        var weights = new DirectPtxExtent(3, heads, HeadDimension, modelDimension);
        var bias = new DirectPtxExtent(3, heads, HeadDimension);
        var rope = new DirectPtxExtent(cacheCapacity, HeadDimension / 2);
        var query = new DirectPtxExtent(1, heads, 1, HeadDimension);
        var cache = new DirectPtxExtent(cacheCapacity, heads, HeadDimension);
        return new DirectPtxKernelBlueprint(
            Operation: "qkv-projection-rope-cache-d64",
            Version: 1,
            Architecture: architecture,
            Variant: $"decode-fp32-h{heads}-capacity{cacheCapacity}-position{position}",
            Tensors:
            [
                new("input", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    input, input, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("packed-qkv-weights", DirectPtxPhysicalType.Float32,
                    DirectPtxPhysicalLayout.PackedQkvWeights,
                    weights, weights, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("qkv-bias", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.PackedQkvBias,
                    bias, bias, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("cosine", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D,
                    rope, rope, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("sine", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D,
                    rope, rope, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("query", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Bhsd,
                    query, query, 16, DirectPtxTensorAccess.Write, DirectPtxExtentMode.Exact),
                new("key-cache", DirectPtxPhysicalType.Float32,
                    DirectPtxPhysicalLayout.SequenceHeadDim,
                    cache, cache, 16, DirectPtxTensorAccess.ReadWrite, DirectPtxExtentMode.Exact),
                new("value-cache", DirectPtxPhysicalType.Float32,
                    DirectPtxPhysicalLayout.SequenceHeadDim,
                    cache, cache, 16, DirectPtxTensorAccess.ReadWrite, DirectPtxExtentMode.Exact)
            ],
            ResourceBudget: new DirectPtxResourceBudget(
                MaxRegistersPerThread: 48,
                MaxStaticSharedBytes: 0,
                MaxLocalBytesPerThread: 0,
                MinBlocksPerMultiprocessor: 8),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["input"] = "one-fp32-decode-token",
                ["weights"] = "output-major-packed-qkv-fp32",
                ["projection"] = "qkv-plus-bias-fp32",
                ["rope"] = "interleaved-q-and-k-baked-position",
                ["cache"] = "dense-sequence-head-d64-baked-position",
                ["output"] = "q-only-plus-in-place-kv-cache-write",
                ["global-intermediates"] = "none",
                ["temporary-device-allocation"] = "none",
                ["stride-parameters"] = "none"
            });
    }

    private static void ValidateShape(int heads, int cacheCapacity, int position)
    {
        if (heads is not (4 or 8 or 16))
            throw new ArgumentOutOfRangeException(nameof(heads), "Supported head buckets are 4, 8, and 16.");
        if (cacheCapacity is not (16 or 32 or 64 or 128))
            throw new ArgumentOutOfRangeException(
                nameof(cacheCapacity), "Supported cache-capacity buckets are 16, 32, 64, and 128.");
        if (position < 0 || position >= cacheCapacity)
            throw new ArgumentOutOfRangeException(nameof(position));
    }
}
