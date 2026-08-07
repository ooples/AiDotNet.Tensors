using System;
using System.Collections.Generic;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// Elementwise masked fill <c>output[i] = mask[i] != 0 ? fillValue : input[i]</c> (issue
/// #840), the pre-softmax masking stage (e.g. causal/padding masks). Purely elementwise over
/// a flat element count, matching the backend's flat-<c>size</c> ABI — one thread owns one
/// aligned pair of striped float4 transactions, with no shared memory, reduction, or global
/// intermediate.
/// The result is exact.
///
/// 256 threads/block, eight elements/thread; supported counts are positive multiples of 256.
/// </summary>
internal sealed class PtxMaskedFillKernel : IDisposable
{
    internal const int BlockThreads = PtxElementwiseShape.BlockThreads;
    internal const int MaxCount = PtxElementwiseShape.MaxCount;
    internal const string EntryPoint = "aidotnet_masked_fill";

    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;
    private readonly PtxElementwiseVariant _variant;

    internal int Count { get; }
    internal string Ptx { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal PtxMaskedFillKernel(DirectPtxRuntime runtime, int count)
        : this(runtime, count, PtxElementwiseVariant.MaskedFillDefault)
    {
    }

    internal PtxMaskedFillKernel(
        DirectPtxRuntime runtime, int count, PtxElementwiseVariant variant)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        if (!DirectPtxArchitecture.HasValidatedSoftmax(
            runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new PlatformNotSupportedException(
                "The checked-in masked-fill specialization is measured only on GA10x/SM86.");
        PtxElementwiseShape.Validate(count, "Masked fill");
        variant.Validate(count);
        Count = count;
        _variant = variant;
        Blueprint = CreateBlueprint(runtime.ArchitectureFamily, count, variant);
        Ptx = EmitPtx(runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor, count, variant);
        var loaded = DirectPtxResourceInitialization.Complete(
            runtime.LoadModule(Ptx),
            module =>
            {
                IntPtr function = module.GetFunction(EntryPoint, out DirectPtxFunctionInfo info);
                int activeBlocks = module.GetActiveBlocksPerMultiprocessor(function, variant.BlockThreads);
                Blueprint.ResourceBudget.Validate(EntryPoint, info, variant.BlockThreads, activeBlocks);
                DirectPtxKernelAudit audit = DirectPtxKernelAudit.Create(
                    Blueprint, runtime.DeviceFingerprint, Ptx, info, variant.BlockThreads, activeBlocks,
                    module);
                return (Function: function, Audit: audit);
            });
        _module = loaded.Resource;
        _function = loaded.Value.Function;
        Audit = loaded.Value.Audit;
    }

    internal unsafe void Launch(
        DirectPtxTensorView input, DirectPtxTensorView mask, DirectPtxTensorView output,
        float fill)
    {
        PtxAbiGuard.Require(input, Blueprint.Tensors[0], nameof(input));
        PtxAbiGuard.Require(mask, Blueprint.Tensors[1], nameof(mask));
        PtxAbiGuard.Require(output, Blueprint.Tensors[2], nameof(output));

        IntPtr inputPointer = input.Pointer;
        IntPtr maskPointer = mask.Pointer;
        IntPtr outputPointer = output.Pointer;
        void** arguments = stackalloc void*[4];
        arguments[0] = &inputPointer;
        arguments[1] = &maskPointer;
        arguments[2] = &outputPointer;
        arguments[3] = &fill;
        _module.Launch(_function, (uint)PtxElementwiseShape.VectorGridBlocks(
                Count, _variant.BlockThreads, _variant.VectorWidth),
            1, 1, (uint)_variant.BlockThreads, 1, 1, 0, arguments);
    }

    public void Dispose() => _module.Dispose();

    internal static string EmitPtx(int ccMajor, int ccMinor, int count)
        => EmitPtx(ccMajor, ccMinor, count, PtxElementwiseVariant.MaskedFillDefault);

    internal static string EmitPtx(
        int ccMajor, int ccMinor, int count, PtxElementwiseVariant variant)
    {
        PtxElementwiseShape.Validate(count, "Masked fill");
        variant.Validate(count);
        int packCount = variant.VectorWidth / 4;
        int packStrideBytes = checked(count * sizeof(float) / packCount);
        var ptx = new StringBuilder(4_000);
        ptx.AppendLine(".version 7.1");
        ptx.AppendLine($".target sm_{ccMajor}{ccMinor}");
        ptx.AppendLine(".address_size 64");
        ptx.AppendLine($"// masked-fill count={count}");
        ptx.AppendLine();
        ptx.AppendLine($".visible .entry {EntryPoint}(");
        ptx.AppendLine("    .param .u64 input_ptr,");
        ptx.AppendLine("    .param .u64 mask_ptr,");
        ptx.AppendLine("    .param .u64 output_ptr,");
        ptx.AppendLine("    .param .f32 fill");
        ptx.AppendLine(")");
        ptx.AppendLine($".maxntid {variant.BlockThreads}, 1, 1");
        ptx.AppendLine("{");
        ptx.AppendLine("    .reg .pred %p<4>;");
        ptx.AppendLine("    .reg .b32 %r<8>;");
        ptx.AppendLine("    .reg .b64 %rd<12>;");
        ptx.AppendLine("    .reg .f32 %f<16>;");
        ptx.AppendLine("    ld.param.u64 %rd0, [input_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd1, [mask_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd2, [output_ptr];");
        ptx.AppendLine("    ld.param.f32 %f8, [fill];");
        ptx.AppendLine("    mov.u32 %r0, %tid.x;");
        ptx.AppendLine("    mov.u32 %r1, %ctaid.x;");
        ptx.AppendLine($"    mad.lo.u32 %r2, %r1, {variant.BlockThreads}, %r0;");
        if (PtxElementwiseShape.RequiresBoundsGuard(
                count, variant.BlockThreads, variant.VectorWidth))
        {
            ptx.AppendLine($"    setp.ge.u32 %p0, %r2, {count / variant.VectorWidth};");
            ptx.AppendLine("    @%p0 bra.uni MASKED_FILL_DONE;");
        }
        ptx.AppendLine("    mul.wide.u32 %rd3, %r2, 16;");
        ptx.AppendLine("    add.u64 %rd4, %rd0, %rd3;");
        ptx.AppendLine("    add.u64 %rd5, %rd1, %rd3;");
        ptx.AppendLine("    add.u64 %rd6, %rd2, %rd3;");
        for (int pack = 0; pack < packCount; pack++)
        {
            EmitFloat4Slice(ptx, "%rd4", "%rd5", "%rd6", variant);
            if (pack + 1 < packCount)
            {
                ptx.AppendLine($"    add.u64 %rd4, %rd4, {packStrideBytes};");
                ptx.AppendLine($"    add.u64 %rd5, %rd5, {packStrideBytes};");
                ptx.AppendLine($"    add.u64 %rd6, %rd6, {packStrideBytes};");
            }
        }
        ptx.AppendLine("MASKED_FILL_DONE:");
        ptx.AppendLine("    ret;");
        ptx.AppendLine("}");
        return ptx.ToString();
    }

    private static void EmitFloat4Slice(
        StringBuilder ptx, string inputAddress, string maskAddress, string outputAddress,
        PtxElementwiseVariant variant)
    {
        ptx.AppendLine($"    ld.global.{variant.LoadModifier}.v4.f32 {{%f0,%f1,%f2,%f3}}, [{inputAddress}];");
        ptx.AppendLine($"    ld.global.{variant.LoadModifier}.v4.f32 {{%f4,%f5,%f6,%f7}}, [{maskAddress}];");
        ptx.AppendLine("    setp.neu.f32 %p0, %f4, 0f00000000;");
        ptx.AppendLine("    setp.neu.f32 %p1, %f5, 0f00000000;");
        ptx.AppendLine("    setp.neu.f32 %p2, %f6, 0f00000000;");
        ptx.AppendLine("    setp.neu.f32 %p3, %f7, 0f00000000;");
        ptx.AppendLine("    selp.f32 %f9, %f8, %f0, %p0;");
        ptx.AppendLine("    selp.f32 %f10, %f8, %f1, %p1;");
        ptx.AppendLine("    selp.f32 %f11, %f8, %f2, %p2;");
        ptx.AppendLine("    selp.f32 %f12, %f8, %f3, %p3;");
        ptx.AppendLine($"    st.global.{variant.StoreModifier}.v4.f32 [{outputAddress}], {{%f9,%f10,%f11,%f12}};");
    }

    private static DirectPtxKernelBlueprint CreateBlueprint(
        DirectPtxArchitectureFamily architecture, int count, PtxElementwiseVariant variant)
    {
        var extent = new DirectPtxExtent(count);
        return new DirectPtxKernelBlueprint(
            Operation: "masked-fill",
            Version: 4,
            Architecture: architecture,
            Variant: $"fp32-count{count}-{variant.Name}",
            Tensors:
            [
                new("input", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    extent, extent, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("mask", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    extent, extent, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("output", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    extent, extent, 16, DirectPtxTensorAccess.Write, DirectPtxExtentMode.Exact)
            ],
            ResourceBudget: new DirectPtxResourceBudget(
                MaxRegistersPerThread: variant.VectorWidth == 16 ? 40 : 28,
                MaxStaticSharedBytes: 0,
                MaxLocalBytesPerThread: 0,
                MinBlocksPerMultiprocessor: 1536 / variant.BlockThreads),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["formula"] = "output[i] = mask[i] != 0 ? fillValue : input[i]",
                ["role"] = "pre-softmax-masking",
                ["vector-width"] = variant.VectorWidth.ToString(System.Globalization.CultureInfo.InvariantCulture),
                ["block-threads"] = variant.BlockThreads.ToString(System.Globalization.CultureInfo.InvariantCulture),
                ["cache-policy"] = $"ld.{variant.LoadModifier}/st.{variant.StoreModifier}",
                ["global-intermediates"] = "none",
                ["temporary-device-allocation"] = "none",
                ["stride-parameters"] = "none"
            });
    }

    internal static bool IsSupportedCount(int count) => PtxElementwiseShape.IsSupported(count);

    internal static bool IsPromotedCount(int count) => PtxElementwiseShape.IsPromoted(count);
}
