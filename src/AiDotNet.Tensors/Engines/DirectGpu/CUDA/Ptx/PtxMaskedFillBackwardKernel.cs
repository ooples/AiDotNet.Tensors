using System;
using System.Collections.Generic;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// Elementwise masked-fill backward <c>gradInput[i] = mask[i] != 0 ? 0 : gradOutput[i]</c>
/// (issue #840): the gradient does not flow through positions overwritten by the fill
/// constant. Purely elementwise over a flat element count, matching the backend's flat-
/// <c>size</c> ABI — one thread owns two aligned float4 transactions striped across the tensor,
/// reduction, or global intermediate. The result is exact.
///
/// 256 threads/block, eight elements/thread; supported counts are positive multiples of 256.
/// </summary>
internal sealed class PtxMaskedFillBackwardKernel : IDisposable
{
    internal const int BlockThreads = PtxElementwiseShape.BlockThreads;
    internal const int MaxCount = PtxElementwiseShape.MaxCount;
    internal const string EntryPoint = "aidotnet_masked_fill_backward";

    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

    internal int Count { get; }
    internal string Ptx { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal PtxMaskedFillBackwardKernel(DirectPtxRuntime runtime, int count)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        if (!DirectPtxArchitecture.HasValidatedSoftmax(
            runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new PlatformNotSupportedException(
                "The checked-in masked-fill-backward specialization is measured only on GA10x/SM86.");
        PtxElementwiseShape.Validate(count, "Masked-fill backward");
        Count = count;
        Blueprint = CreateBlueprint(runtime.ArchitectureFamily, count);
        Ptx = EmitPtx(runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor, count);
        _module = runtime.LoadModule(Ptx);
        _function = _module.GetFunction(EntryPoint, out DirectPtxFunctionInfo info);
        int activeBlocks = _module.GetActiveBlocksPerMultiprocessor(_function, BlockThreads);
        Blueprint.ResourceBudget.Validate(EntryPoint, info, BlockThreads, activeBlocks);
        Audit = DirectPtxKernelAudit.Create(
            Blueprint, runtime.DeviceFingerprint, Ptx, info, BlockThreads, activeBlocks, _module.JitInfoLog);
    }

    internal unsafe void Launch(
        DirectPtxTensorView grad, DirectPtxTensorView mask, DirectPtxTensorView output)
    {
        PtxAbiGuard.Require(grad, Blueprint.Tensors[0], nameof(grad));
        PtxAbiGuard.Require(mask, Blueprint.Tensors[1], nameof(mask));
        PtxAbiGuard.Require(output, Blueprint.Tensors[2], nameof(output));

        IntPtr gradPointer = grad.Pointer;
        IntPtr maskPointer = mask.Pointer;
        IntPtr outputPointer = output.Pointer;
        void** arguments = stackalloc void*[3];
        arguments[0] = &gradPointer;
        arguments[1] = &maskPointer;
        arguments[2] = &outputPointer;
        _module.Launch(_function, (uint)PtxElementwiseShape.VectorGridBlocks(Count),
            1, 1, BlockThreads, 1, 1, 0, arguments);
    }

    public void Dispose() => _module.Dispose();

    internal static string EmitPtx(int ccMajor, int ccMinor, int count)
    {
        PtxElementwiseShape.Validate(count, "Masked-fill backward");
        var ptx = new StringBuilder(4_000);
        ptx.AppendLine(".version 7.1");
        ptx.AppendLine($".target sm_{ccMajor}{ccMinor}");
        ptx.AppendLine(".address_size 64");
        ptx.AppendLine($"// masked-fill-backward count={count}");
        ptx.AppendLine();
        ptx.AppendLine($".visible .entry {EntryPoint}(");
        ptx.AppendLine("    .param .u64 grad_ptr,");
        ptx.AppendLine("    .param .u64 mask_ptr,");
        ptx.AppendLine("    .param .u64 output_ptr");
        ptx.AppendLine(")");
        ptx.AppendLine($".maxntid {BlockThreads}, 1, 1");
        ptx.AppendLine("{");
        ptx.AppendLine("    .reg .pred %p<4>;");
        ptx.AppendLine("    .reg .b32 %r<8>;");
        ptx.AppendLine("    .reg .b64 %rd<12>;");
        ptx.AppendLine("    .reg .f32 %f<16>;");
        ptx.AppendLine("    ld.param.u64 %rd0, [grad_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd1, [mask_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd2, [output_ptr];");
        ptx.AppendLine("    mov.u32 %r0, %tid.x;");
        ptx.AppendLine("    mov.u32 %r1, %ctaid.x;");
        ptx.AppendLine($"    mad.lo.u32 %r2, %r1, {BlockThreads}, %r0;");    // two striped float4 packs
        if (PtxElementwiseShape.RequiresBoundsGuard(count, BlockThreads))
        {
            ptx.AppendLine($"    setp.ge.u32 %p0, %r2, {count / PtxElementwiseShape.VectorWidth};");
            ptx.AppendLine("    @%p0 bra.uni MASKED_FILL_BACKWARD_DONE;");
        }
        ptx.AppendLine("    mul.wide.u32 %rd3, %r2, 16;");
        ptx.AppendLine("    add.u64 %rd4, %rd0, %rd3;");
        ptx.AppendLine("    add.u64 %rd5, %rd1, %rd3;");
        ptx.AppendLine("    add.u64 %rd6, %rd2, %rd3;");
        EmitFloat4Slice(ptx, "%rd4", "%rd5", "%rd6");
        ptx.AppendLine($"    add.u64 %rd4, %rd4, {count * 2};");
        ptx.AppendLine($"    add.u64 %rd5, %rd5, {count * 2};");
        ptx.AppendLine($"    add.u64 %rd6, %rd6, {count * 2};");
        EmitFloat4Slice(ptx, "%rd4", "%rd5", "%rd6");
        ptx.AppendLine("MASKED_FILL_BACKWARD_DONE:");
        ptx.AppendLine("    ret;");
        ptx.AppendLine("}");
        return ptx.ToString();
    }

    private static void EmitFloat4Slice(
        StringBuilder ptx, string gradAddress, string maskAddress, string outputAddress)
    {
        ptx.AppendLine($"    ld.global.nc.v4.f32 {{%f0,%f1,%f2,%f3}}, [{gradAddress}];");
        ptx.AppendLine($"    ld.global.nc.v4.f32 {{%f4,%f5,%f6,%f7}}, [{maskAddress}];");
        ptx.AppendLine("    setp.neu.f32 %p0, %f4, 0f00000000;");
        ptx.AppendLine("    setp.neu.f32 %p1, %f5, 0f00000000;");
        ptx.AppendLine("    setp.neu.f32 %p2, %f6, 0f00000000;");
        ptx.AppendLine("    setp.neu.f32 %p3, %f7, 0f00000000;");
        ptx.AppendLine("    selp.f32 %f9, 0f00000000, %f0, %p0;");
        ptx.AppendLine("    selp.f32 %f10, 0f00000000, %f1, %p1;");
        ptx.AppendLine("    selp.f32 %f11, 0f00000000, %f2, %p2;");
        ptx.AppendLine("    selp.f32 %f12, 0f00000000, %f3, %p3;");
        ptx.AppendLine($"    st.global.wt.v4.f32 [{outputAddress}], {{%f9,%f10,%f11,%f12}};");
    }

    private static DirectPtxKernelBlueprint CreateBlueprint(DirectPtxArchitectureFamily architecture, int count)
    {
        var extent = new DirectPtxExtent(count);
        return new DirectPtxKernelBlueprint(
            Operation: "masked-fill-backward",
            Version: 3,
            Architecture: architecture,
            Variant: $"fp32-count{count}",
            Tensors:
            [
                new("grad", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    extent, extent, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("mask", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    extent, extent, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("output", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    extent, extent, 16, DirectPtxTensorAccess.Write, DirectPtxExtentMode.Exact)
            ],
            ResourceBudget: new DirectPtxResourceBudget(
                MaxRegistersPerThread: 28,
                MaxStaticSharedBytes: 0,
                MaxLocalBytesPerThread: 0,
                MinBlocksPerMultiprocessor: 1),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["formula"] = "gradInput[i] = mask[i] != 0 ? 0 : gradOutput[i]",
                ["role"] = "masked-fill-gradient-gating",
                ["vector-width"] = "8 (two striped, aligned float4 transactions)",
                ["global-intermediates"] = "none",
                ["temporary-device-allocation"] = "none",
                ["stride-parameters"] = "none"
            });
    }

    internal static bool IsSupportedCount(int count) => PtxElementwiseShape.IsSupported(count);

    internal static bool IsPromotedCount(int count) => PtxElementwiseShape.IsPromoted(count);
}
