using System;
using System.Collections.Generic;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// Exact-shape FP32 pairwise XYXY IoU. Each thread reads two boxes, keeps all
/// geometry intermediates in registers, and writes one IoU cell. Shape and
/// layout are baked into PTX; the device ABI contains pointers only.
/// </summary>
internal sealed class PtxFusedPairwiseBoxIouF32Kernel : IDisposable
{
    internal const string EntryPoint = "aidotnet_fused_pairwise_box_iou_f32";
    internal const int BlockThreads = 256;

    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

    internal int BoxCountA { get; }
    internal int BoxCountB { get; }
    internal string Ptx { get; }
    internal DirectPtxFunctionInfo FunctionInfo { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }
    internal string JitInfoLog => _module.JitInfoLog;

    internal static bool IsSupportedShape(int n, int m) =>
        (n, m) is (256, 256) or (1024, 256) or (1024, 1024) or (4096, 256);

    internal static bool IsPromotedShape(int n, int m) => false;

    internal PtxFusedPairwiseBoxIouF32Kernel(DirectPtxRuntime runtime, int n, int m)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        if (!IsSupportedShape(n, m))
            throw new NotSupportedException($"Pairwise BoxIoU shape [{n},{m}] is not emitted.");
        if (!DirectPtxArchitecture.HasValidatedVisionBoxIou(
            runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new NotSupportedException(
                $"Pairwise BoxIoU has no SM {runtime.ComputeCapabilityMajor}.{runtime.ComputeCapabilityMinor} specialization.");

        BoxCountA = n;
        BoxCountB = m;
        Blueprint = CreateBlueprint(runtime.ArchitectureFamily, n, m);
        Ptx = EmitPtx(runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor, n, m);
        _module = runtime.LoadModule(Ptx);
        _function = _module.GetFunction(EntryPoint, out DirectPtxFunctionInfo functionInfo);
        FunctionInfo = functionInfo;
        int activeBlocks = _module.GetActiveBlocksPerMultiprocessor(_function, BlockThreads);
        Blueprint.ResourceBudget.Validate(EntryPoint, functionInfo, BlockThreads, activeBlocks);
        Audit = DirectPtxKernelAudit.Create(
            Blueprint, runtime.DeviceFingerprint, Ptx, functionInfo,
            BlockThreads, activeBlocks, _module.JitInfoLog);
    }

    private static DirectPtxKernelBlueprint CreateBlueprint(
        DirectPtxArchitectureFamily architecture, int n, int m)
    {
        var boxesA = new DirectPtxExtent(n, 4);
        var boxesB = new DirectPtxExtent(m, 4);
        var output = new DirectPtxExtent(n, m);
        return new DirectPtxKernelBlueprint(
            Operation: "pairwise-box-iou-f32",
            Version: 1,
            Architecture: architecture,
            Variant: $"xyxy-n{n}-m{m}-t{BlockThreads}",
            Tensors:
            [
                new("boxes-a", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.BoxXyxy,
                    boxesA, boxesA, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("boxes-b", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.BoxXyxy,
                    boxesB, boxesB, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("iou", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D,
                    output, output, 16, DirectPtxTensorAccess.Write, DirectPtxExtentMode.Exact)
            ],
            ResourceBudget: new DirectPtxResourceBudget(
                MaxRegistersPerThread: 32,
                MaxStaticSharedBytes: 0,
                MaxLocalBytesPerThread: 0,
                MinBlocksPerMultiprocessor: 4),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["coordinates"] = "xyxy",
                ["degenerate-area"] = "max(width,0)*max(height,0)",
                ["zero-union"] = "0",
                ["input"] = "fp32",
                ["accumulator"] = "fp32",
                ["output"] = "fp32",
                ["layout"] = "contiguous boxes [N,4], [M,4]; output [N,M]",
                ["intermediate-global-bytes"] = "0",
                ["determinism"] = "one independent thread per output cell"
            });
    }

    internal unsafe void Launch(
        DirectPtxTensorView boxesA,
        DirectPtxTensorView boxesB,
        DirectPtxTensorView output)
    {
        Require(boxesA, Blueprint.Tensors[0], nameof(boxesA));
        Require(boxesB, Blueprint.Tensors[1], nameof(boxesB));
        Require(output, Blueprint.Tensors[2], nameof(output));
        if (RangesOverlap(output, boxesA) || RangesOverlap(output, boxesB))
            throw new ArgumentException("Pairwise BoxIoU output must not alias either input.", nameof(output));

        IntPtr a = boxesA.Pointer;
        IntPtr b = boxesB.Pointer;
        IntPtr o = output.Pointer;
        void** arguments = stackalloc void*[3];
        arguments[0] = &a;
        arguments[1] = &b;
        arguments[2] = &o;
        uint blocks = checked((uint)(((long)BoxCountA * BoxCountB) / BlockThreads));
        _module.Launch(_function, blocks, 1, 1, BlockThreads, 1, 1, 0, arguments);
    }

    private static void Require(
        DirectPtxTensorView view,
        DirectPtxTensorContract contract,
        string parameter)
    {
        if (view.Pointer == IntPtr.Zero ||
            view.PhysicalType != contract.PhysicalType ||
            view.Layout != contract.Layout ||
            view.LogicalExtent != contract.LogicalExtent ||
            view.PhysicalExtent != contract.PhysicalExtent ||
            view.ByteLength != contract.RequiredBytes ||
            view.AllocationByteLength != contract.RequiredBytes ||
            view.Access != contract.Access)
            throw new ArgumentException(
                $"{parameter} does not satisfy physical ABI '{contract.Name}'.", parameter);
    }

    private static bool RangesOverlap(DirectPtxTensorView left, DirectPtxTensorView right)
    {
        nuint leftStart = PtxCompat.ToNuint(left.Pointer);
        nuint rightStart = PtxCompat.ToNuint(right.Pointer);
        nuint leftEnd = checked(leftStart + left.ByteLength);
        nuint rightEnd = checked(rightStart + right.ByteLength);
        return leftStart < rightEnd && rightStart < leftEnd;
    }

    public void Dispose() => _module.Dispose();

    internal static string EmitPtx(int ccMajor, int ccMinor, int n, int m)
    {
        if (!DirectPtxArchitecture.HasValidatedVisionBoxIou(ccMajor, ccMinor))
            throw new NotSupportedException(
                $"Pairwise BoxIoU has no SM {ccMajor}.{ccMinor} emitter.");
        if (!IsSupportedShape(n, m))
            throw new NotSupportedException($"Pairwise BoxIoU shape [{n},{m}] is not emitted.");
        int mShift = m == 256 ? 8 : 10;
        int mMask = m - 1;
        var ptx = new StringBuilder(4096);
        ptx.AppendLine(".version 7.1");
        ptx.AppendLine($".target sm_{ccMajor}{ccMinor}");
        ptx.AppendLine(".address_size 64");
        ptx.AppendLine();
        ptx.AppendLine($".visible .entry {EntryPoint}(");
        ptx.AppendLine("    .param .u64 boxes_a_ptr,");
        ptx.AppendLine("    .param .u64 boxes_b_ptr,");
        ptx.AppendLine("    .param .u64 output_ptr");
        ptx.AppendLine(")");
        ptx.AppendLine("{");
        ptx.AppendLine("    .reg .pred %p<2>;");
        ptx.AppendLine("    .reg .b32 %r<8>;");
        ptx.AppendLine("    .reg .b64 %rd<10>;");
        ptx.AppendLine("    .reg .f32 %f<24>;");
        ptx.AppendLine("    ld.param.u64 %rd0, [boxes_a_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd1, [boxes_b_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd2, [output_ptr];");
        ptx.AppendLine("    mov.u32 %r0, %ctaid.x;");
        ptx.AppendLine("    mov.u32 %r1, %tid.x;");
        ptx.AppendLine($"    mad.lo.u32 %r2, %r0, {BlockThreads}, %r1;");
        ptx.AppendLine($"    shr.u32 %r3, %r2, {mShift};");
        ptx.AppendLine($"    and.b32 %r4, %r2, {mMask};");
        ptx.AppendLine("    mul.wide.u32 %rd3, %r3, 16;");
        ptx.AppendLine("    mul.wide.u32 %rd4, %r4, 16;");
        ptx.AppendLine("    add.u64 %rd5, %rd0, %rd3;");
        ptx.AppendLine("    add.u64 %rd6, %rd1, %rd4;");
        ptx.AppendLine("    ld.global.v4.f32 {%f0,%f1,%f2,%f3}, [%rd5];");
        ptx.AppendLine("    ld.global.v4.f32 {%f4,%f5,%f6,%f7}, [%rd6];");
        ptx.AppendLine("    mov.f32 %f16, 0f00000000;");
        ptx.AppendLine("    sub.rn.f32 %f8, %f2, %f0;");
        ptx.AppendLine("    max.f32 %f8, %f8, %f16;");
        ptx.AppendLine("    sub.rn.f32 %f9, %f3, %f1;");
        ptx.AppendLine("    max.f32 %f9, %f9, %f16;");
        ptx.AppendLine("    mul.rn.f32 %f10, %f8, %f9;");
        ptx.AppendLine("    sub.rn.f32 %f8, %f6, %f4;");
        ptx.AppendLine("    max.f32 %f8, %f8, %f16;");
        ptx.AppendLine("    sub.rn.f32 %f9, %f7, %f5;");
        ptx.AppendLine("    max.f32 %f9, %f9, %f16;");
        ptx.AppendLine("    mul.rn.f32 %f11, %f8, %f9;");
        ptx.AppendLine("    max.f32 %f12, %f0, %f4;");
        ptx.AppendLine("    max.f32 %f13, %f1, %f5;");
        ptx.AppendLine("    min.f32 %f14, %f2, %f6;");
        ptx.AppendLine("    min.f32 %f15, %f3, %f7;");
        ptx.AppendLine("    sub.rn.f32 %f14, %f14, %f12;");
        ptx.AppendLine("    max.f32 %f14, %f14, %f16;");
        ptx.AppendLine("    sub.rn.f32 %f15, %f15, %f13;");
        ptx.AppendLine("    max.f32 %f15, %f15, %f16;");
        ptx.AppendLine("    mul.rn.f32 %f17, %f14, %f15;");
        ptx.AppendLine("    add.rn.f32 %f18, %f10, %f11;");
        ptx.AppendLine("    sub.rn.f32 %f18, %f18, %f17;");
        ptx.AppendLine("    setp.gt.f32 %p0, %f18, %f16;");
        ptx.AppendLine("    mov.f32 %f19, 0f00000000;");
        ptx.AppendLine("    @%p0 div.rn.f32 %f19, %f17, %f18;");
        ptx.AppendLine("    mul.wide.u32 %rd7, %r2, 4;");
        ptx.AppendLine("    add.u64 %rd8, %rd2, %rd7;");
        ptx.AppendLine("    st.global.f32 [%rd8], %f19;");
        ptx.AppendLine("    ret;");
        ptx.AppendLine("}");
        return ptx.ToString();
    }
}
