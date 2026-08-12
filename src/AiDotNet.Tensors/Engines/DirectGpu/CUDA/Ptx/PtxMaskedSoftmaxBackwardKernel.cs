using System;
using System.Collections.Generic;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// Fuses the softmax Jacobian-vector product with the gradient gate introduced by a preceding
/// masked fill: <c>dX = mask != 0 ? 0 : S * (dY - sum(dY * S))</c>. The mask is consumed only
/// in the final pass, so the composed backward contract needs no global gradient intermediate.
/// This is an oracle candidate, not a promoted dispatch.
/// </summary>
internal sealed class PtxMaskedSoftmaxBackwardKernel : IDisposable
{
    internal const int BlockThreads = PtxRowShape.BlockThreads;
    internal const string EntryPoint = "aidotnet_masked_softmax_backward_row";

    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

    internal int M { get; }
    internal int N { get; }
    internal string Ptx { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal PtxMaskedSoftmaxBackwardKernel(DirectPtxRuntime runtime, int m, int n)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        if (!DirectPtxArchitecture.HasValidatedSoftmax(
            runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new PlatformNotSupportedException(
                "The checked-in masked-softmax-backward candidate is measured only on GA10x/SM86.");
        PtxRowShape.Validate(m, n, "Masked softmax backward");
        M = m;
        N = n;
        Blueprint = CreateBlueprint(runtime.ArchitectureFamily, m, n);
        Ptx = EmitPtx(runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor, m, n);
        var loaded = DirectPtxResourceInitialization.Complete(
            runtime.LoadModule(Ptx),
            module =>
            {
                IntPtr function = module.GetFunction(EntryPoint, out DirectPtxFunctionInfo info);
                int activeBlocks = module.GetActiveBlocksPerMultiprocessor(function, BlockThreads);
                Blueprint.ResourceBudget.Validate(EntryPoint, info, BlockThreads, activeBlocks);
                DirectPtxKernelAudit audit = DirectPtxKernelAudit.Create(
                    Blueprint, runtime.DeviceFingerprint, Ptx, info, BlockThreads, activeBlocks,
                    module);
                return (Function: function, Audit: audit);
            });
        _module = loaded.Resource;
        _function = loaded.Value.Function;
        Audit = loaded.Value.Audit;
    }

    internal unsafe void Launch(
        DirectPtxTensorView softmax,
        DirectPtxTensorView grad,
        DirectPtxTensorView mask,
        DirectPtxTensorView output)
    {
        PtxAbiGuard.Require(softmax, Blueprint.Tensors[0], nameof(softmax));
        PtxAbiGuard.Require(grad, Blueprint.Tensors[1], nameof(grad));
        PtxAbiGuard.Require(mask, Blueprint.Tensors[2], nameof(mask));
        PtxAbiGuard.Require(output, Blueprint.Tensors[3], nameof(output));

        IntPtr softmaxPointer = softmax.Pointer;
        IntPtr gradPointer = grad.Pointer;
        IntPtr maskPointer = mask.Pointer;
        IntPtr outputPointer = output.Pointer;
        void** arguments = stackalloc void*[4];
        arguments[0] = &softmaxPointer;
        arguments[1] = &gradPointer;
        arguments[2] = &maskPointer;
        arguments[3] = &outputPointer;
        _module.Launch(_function, (uint)M, 1, 1, BlockThreads, 1, 1, 0, arguments);
    }

    public void Dispose() => _module.Dispose();

    internal static string EmitPtx(int ccMajor, int ccMinor, int m, int n)
    {
        PtxRowShape.Validate(m, n, "Masked softmax backward");
        int rowBytes = checked(n * sizeof(float));

        var ptx = new StringBuilder(10_000);
        ptx.AppendLine(".version 7.1");
        ptx.AppendLine($".target sm_{ccMajor}{ccMinor}");
        ptx.AppendLine(".address_size 64");
        ptx.AppendLine($"// masked-softmax-backward-row M={m} N={n}");
        ptx.AppendLine();
        ptx.AppendLine($".visible .entry {EntryPoint}(");
        ptx.AppendLine("    .param .u64 softmax_ptr,");
        ptx.AppendLine("    .param .u64 grad_ptr,");
        ptx.AppendLine("    .param .u64 mask_ptr,");
        ptx.AppendLine("    .param .u64 output_ptr");
        ptx.AppendLine(")");
        ptx.AppendLine($".maxntid {BlockThreads}, 1, 1");
        ptx.AppendLine("{");
        ptx.AppendLine("    .reg .pred %p<4>;");
        ptx.AppendLine("    .reg .b32 %r<12>;");
        ptx.AppendLine("    .reg .b64 %rd<24>;");
        ptx.AppendLine("    .reg .f32 %f<20>;");
        ptx.AppendLine($"    .shared .align 16 .b8 row_sh[{n * sizeof(float)}];");
        ptx.AppendLine($"    .shared .align 16 .b8 red[{PtxRowReduce.SharedBytes}];");
        ptx.AppendLine("    ld.param.u64 %rd0, [softmax_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd1, [grad_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd2, [mask_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd3, [output_ptr];");
        ptx.AppendLine("    mov.u64 %rd4, row_sh;");
        ptx.AppendLine("    mov.u64 %rd5, red;");
        ptx.AppendLine("    mov.u32 %r0, %tid.x;");
        ptx.AppendLine("    mov.u32 %r1, %ctaid.x;");
        ptx.AppendLine($"    mul.wide.u32 %rd6, %r1, {rowBytes};");
        ptx.AppendLine("    add.u64 %rd7, %rd0, %rd6;");
        ptx.AppendLine("    add.u64 %rd17, %rd1, %rd6;");
        ptx.AppendLine("    add.u64 %rd18, %rd2, %rd6;");
        ptx.AppendLine("    add.u64 %rd8, %rd3, %rd6;");
        ptx.AppendLine("    mul.wide.u32 %rd9, %r0, 4;");
        ptx.AppendLine("    add.u64 %rd10, %rd5, %rd9;");

        ptx.AppendLine("    mov.f32 %f0, 0f00000000;");
        ptx.AppendLine("    mov.u32 %r3, %r0;");
        ptx.AppendLine("MASKED_SOFTMAX_BWD_LOAD_LOOP:");
        ptx.AppendLine($"    setp.ge.u32 %p0, %r3, {n};");
        ptx.AppendLine("    @%p0 bra.uni MASKED_SOFTMAX_BWD_LOAD_DONE;");
        ptx.AppendLine("    mul.wide.u32 %rd11, %r3, 4;");
        ptx.AppendLine("    add.u64 %rd12, %rd7, %rd11;");
        ptx.AppendLine("    ld.global.nc.f32 %f1, [%rd12];");
        ptx.AppendLine("    add.u64 %rd13, %rd4, %rd11;");
        ptx.AppendLine("    st.shared.f32 [%rd13], %f1;");
        ptx.AppendLine("    add.u64 %rd14, %rd17, %rd11;");
        ptx.AppendLine("    ld.global.nc.f32 %f2, [%rd14];");
        ptx.AppendLine("    fma.rn.f32 %f0, %f2, %f1, %f0;");
        ptx.AppendLine($"    add.u32 %r3, %r3, {BlockThreads};");
        ptx.AppendLine("    bra.uni MASKED_SOFTMAX_BWD_LOAD_LOOP;");
        ptx.AppendLine("MASKED_SOFTMAX_BWD_LOAD_DONE:");
        PtxRowReduce.Emit(ptx, "add.rn.f32", "%f0");
        ptx.AppendLine("    ld.shared.f32 %f3, [%rd5];");
        ptx.AppendLine("    bar.sync 0;");

        ptx.AppendLine("    mov.u32 %r3, %r0;");
        ptx.AppendLine("MASKED_SOFTMAX_BWD_OUT_LOOP:");
        ptx.AppendLine($"    setp.ge.u32 %p0, %r3, {n};");
        ptx.AppendLine("    @%p0 bra.uni MASKED_SOFTMAX_BWD_OUT_DONE;");
        ptx.AppendLine("    mul.wide.u32 %rd11, %r3, 4;");
        ptx.AppendLine("    add.u64 %rd13, %rd4, %rd11;");
        ptx.AppendLine("    ld.shared.f32 %f1, [%rd13];");
        ptx.AppendLine("    add.u64 %rd14, %rd17, %rd11;");
        ptx.AppendLine("    ld.global.nc.f32 %f2, [%rd14];");
        ptx.AppendLine("    sub.rn.f32 %f2, %f2, %f3;");
        ptx.AppendLine("    mul.rn.f32 %f1, %f1, %f2;");
        ptx.AppendLine("    add.u64 %rd16, %rd18, %rd11;");
        ptx.AppendLine("    ld.global.nc.f32 %f4, [%rd16];");
        ptx.AppendLine("    setp.neu.f32 %p1, %f4, 0f00000000;");
        ptx.AppendLine("    selp.f32 %f1, 0f00000000, %f1, %p1;");
        ptx.AppendLine("    add.u64 %rd15, %rd8, %rd11;");
        ptx.AppendLine("    st.global.f32 [%rd15], %f1;");
        ptx.AppendLine($"    add.u32 %r3, %r3, {BlockThreads};");
        ptx.AppendLine("    bra.uni MASKED_SOFTMAX_BWD_OUT_LOOP;");
        ptx.AppendLine("MASKED_SOFTMAX_BWD_OUT_DONE:");
        ptx.AppendLine("    ret;");
        ptx.AppendLine("}");
        return ptx.ToString();
    }

    private static DirectPtxKernelBlueprint CreateBlueprint(
        DirectPtxArchitectureFamily architecture, int m, int n)
    {
        var extent = new DirectPtxExtent(m, n);
        return new DirectPtxKernelBlueprint(
            Operation: "masked-softmax-backward-row",
            Version: 1,
            Architecture: architecture,
            Variant: $"fp32-m{m}-n{n}",
            Tensors:
            [
                new("softmax", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D,
                    extent, extent, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("grad", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D,
                    extent, extent, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("mask", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D,
                    extent, extent, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("output", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D,
                    extent, extent, 16, DirectPtxTensorAccess.Write, DirectPtxExtentMode.Exact)
            ],
            ResourceBudget: new DirectPtxResourceBudget(
                MaxRegistersPerThread: 32,
                MaxStaticSharedBytes: n * sizeof(float) + PtxRowReduce.SharedBytes,
                MaxLocalBytesPerThread: 0,
                MinBlocksPerMultiprocessor: 1),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["formula"] = "dX = mask != 0 ? 0 : S * (dY - sum(dY * S))",
                ["fusion"] = "softmax-backward-plus-masked-fill-backward-no-global-intermediate",
                ["jacobian"] = "exact-softmax-jacobian-vector-product",
                ["reduction"] = PtxRowReduce.Strategy,
                ["global-intermediates"] = "none",
                ["temporary-device-allocation"] = "none",
                ["stride-parameters"] = "none"
            });
    }

    internal static bool IsSupportedShape(int m, int n) => PtxRowShape.IsSupported(m, n);

    internal static bool IsPromotedShape(int m, int n) => false;
}
