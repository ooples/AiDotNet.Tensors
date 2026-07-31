using System;
using System.Collections.Generic;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// Row-wise softmax backward <c>dX[m,n] = S[m,n] * (dY[m,n] - sum_n(dY[m,n] * S[m,n]))</c>
/// over the last axis (issue #840), where <c>S</c> is the forward softmax output and
/// <c>dY</c> the upstream gradient. One block owns one row: a single shared-resident pass
/// caches <c>S</c> and reduces the dot <c>sum(dY * S)</c> with an in-block tree reduction,
/// then an elementwise pass emits the gradient. No global intermediate; the exact Jacobian
/// identity means the result is not approximation-limited.
///
/// One block per row (grid = M), 256 threads. Shared: N floats (S cache) + 256 floats
/// (reduction). Supported N are multiples of 256 so each thread strides the row exactly.
/// </summary>
internal sealed class PtxSoftmaxBackwardKernel : IDisposable
{
    internal const int BlockThreads = PtxRowShape.BlockThreads;
    internal const string EntryPoint = "aidotnet_softmax_backward_row";

    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

    internal int M { get; }
    internal int N { get; }
    internal string Ptx { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal PtxSoftmaxBackwardKernel(DirectPtxRuntime runtime, int m, int n)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        if (!DirectPtxArchitecture.HasValidatedSoftmax(
            runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new PlatformNotSupportedException(
                "The checked-in softmax-backward specialization is measured only on GA10x/SM86.");
        PtxRowShape.Validate(m, n, "Softmax backward");
        M = m;
        N = n;
        Blueprint = CreateBlueprint(runtime.ArchitectureFamily, m, n);
        Ptx = EmitPtx(runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor, m, n);
        _module = runtime.LoadModule(Ptx);
        _function = _module.GetFunction(EntryPoint, out DirectPtxFunctionInfo info);
        int activeBlocks = _module.GetActiveBlocksPerMultiprocessor(_function, BlockThreads);
        Blueprint.ResourceBudget.Validate(EntryPoint, info, BlockThreads, activeBlocks);
        Audit = DirectPtxKernelAudit.Create(
            Blueprint, runtime.DeviceFingerprint, Ptx, info, BlockThreads, activeBlocks, _module.JitInfoLog);
    }

    internal unsafe void Launch(
        DirectPtxTensorView softmax, DirectPtxTensorView grad, DirectPtxTensorView output)
    {
        PtxAbiGuard.Require(softmax, Blueprint.Tensors[0], nameof(softmax));
        PtxAbiGuard.Require(grad, Blueprint.Tensors[1], nameof(grad));
        PtxAbiGuard.Require(output, Blueprint.Tensors[2], nameof(output));

        IntPtr softmaxPointer = softmax.Pointer;
        IntPtr gradPointer = grad.Pointer;
        IntPtr outputPointer = output.Pointer;
        void** arguments = stackalloc void*[3];
        arguments[0] = &softmaxPointer;
        arguments[1] = &gradPointer;
        arguments[2] = &outputPointer;
        _module.Launch(_function, (uint)M, 1, 1, BlockThreads, 1, 1, 0, arguments);
    }

    public void Dispose() => _module.Dispose();

    internal static string EmitPtx(int ccMajor, int ccMinor, int m, int n)
    {
        PtxRowShape.Validate(m, n, "Softmax backward");
        int rowBytes = checked(n * sizeof(float));

        var ptx = new StringBuilder(9_000);
        ptx.AppendLine(".version 7.1");
        ptx.AppendLine($".target sm_{ccMajor}{ccMinor}");
        ptx.AppendLine(".address_size 64");
        ptx.AppendLine($"// softmax-backward-row M={m} N={n}");
        ptx.AppendLine();
        ptx.AppendLine($".visible .entry {EntryPoint}(");
        ptx.AppendLine("    .param .u64 softmax_ptr,");
        ptx.AppendLine("    .param .u64 grad_ptr,");
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
        ptx.AppendLine("    ld.param.u64 %rd2, [output_ptr];");
        ptx.AppendLine("    mov.u64 %rd4, row_sh;");
        ptx.AppendLine("    mov.u64 %rd5, red;");
        ptx.AppendLine("    mov.u32 %r0, %tid.x;");
        ptx.AppendLine("    mov.u32 %r1, %ctaid.x;");
        ptx.AppendLine($"    mul.wide.u32 %rd6, %r1, {rowBytes};");
        ptx.AppendLine("    add.u64 %rd7, %rd0, %rd6;");                 // &S[m,0]
        ptx.AppendLine("    add.u64 %rd17, %rd1, %rd6;");                // &dY[m,0]
        ptx.AppendLine("    add.u64 %rd8, %rd2, %rd6;");                 // &dX[m,0]
        ptx.AppendLine("    mul.wide.u32 %rd9, %r0, 4;");
        ptx.AppendLine("    add.u64 %rd10, %rd5, %rd9;");               // &red[tid]

        // ---- Pass 1: cache S + partial dot(dY, S) ----
        ptx.AppendLine("    mov.f32 %f0, 0f00000000;");
        ptx.AppendLine("    mov.u32 %r3, %r0;");
        ptx.AppendLine("LOAD_LOOP:");
        ptx.AppendLine($"    setp.ge.u32 %p0, %r3, {n};");
        ptx.AppendLine("    @%p0 bra.uni LOAD_DONE;");
        ptx.AppendLine("    mul.wide.u32 %rd11, %r3, 4;");
        ptx.AppendLine("    add.u64 %rd12, %rd7, %rd11;");
        ptx.AppendLine("    ld.global.nc.f32 %f1, [%rd12];");           // S
        ptx.AppendLine("    add.u64 %rd13, %rd4, %rd11;");
        ptx.AppendLine("    st.shared.f32 [%rd13], %f1;");
        ptx.AppendLine("    add.u64 %rd14, %rd17, %rd11;");
        ptx.AppendLine("    ld.global.nc.f32 %f2, [%rd14];");           // dY
        ptx.AppendLine("    fma.rn.f32 %f0, %f2, %f1, %f0;");           // dot += dY*S
        ptx.AppendLine($"    add.u32 %r3, %r3, {BlockThreads};");
        ptx.AppendLine("    bra.uni LOAD_LOOP;");
        ptx.AppendLine("LOAD_DONE:");
        PtxRowReduce.Emit(ptx, "add.rn.f32", "%f0");
        ptx.AppendLine("    ld.shared.f32 %f3, [%rd5];");                // dotTotal
        ptx.AppendLine("    bar.sync 0;");

        // ---- Pass 2: dX = S * (dY - dot) ----
        ptx.AppendLine("    mov.u32 %r3, %r0;");
        ptx.AppendLine("OUT_LOOP:");
        ptx.AppendLine($"    setp.ge.u32 %p0, %r3, {n};");
        ptx.AppendLine("    @%p0 bra.uni OUT_DONE;");
        ptx.AppendLine("    mul.wide.u32 %rd11, %r3, 4;");
        ptx.AppendLine("    add.u64 %rd13, %rd4, %rd11;");
        ptx.AppendLine("    ld.shared.f32 %f1, [%rd13];");             // S
        ptx.AppendLine("    add.u64 %rd14, %rd17, %rd11;");
        ptx.AppendLine("    ld.global.nc.f32 %f2, [%rd14];");           // dY
        ptx.AppendLine("    sub.rn.f32 %f2, %f2, %f3;");               // dY - dot
        ptx.AppendLine("    mul.rn.f32 %f1, %f1, %f2;");               // S * (dY - dot)
        ptx.AppendLine("    add.u64 %rd15, %rd8, %rd11;");
        ptx.AppendLine("    st.global.f32 [%rd15], %f1;");
        ptx.AppendLine($"    add.u32 %r3, %r3, {BlockThreads};");
        ptx.AppendLine("    bra.uni OUT_LOOP;");
        ptx.AppendLine("OUT_DONE:");
        ptx.AppendLine("    ret;");
        ptx.AppendLine("}");
        return ptx.ToString();
    }

    private static DirectPtxKernelBlueprint CreateBlueprint(
        DirectPtxArchitectureFamily architecture, int m, int n)
    {
        var extent = new DirectPtxExtent(m, n);
        return new DirectPtxKernelBlueprint(
            Operation: "softmax-backward-row",
            Version: 2,
            Architecture: architecture,
            Variant: $"fp32-m{m}-n{n}",
            Tensors:
            [
                new("softmax", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D,
                    extent, extent, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("grad", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D,
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
                ["formula"] = "dX[m,n] = S[m,n] * (dY[m,n] - sum_n(dY[m,n] * S[m,n]))",
                ["axis"] = "last",
                ["jacobian"] = "exact-softmax-jacobian-vector-product",
                ["reduction"] = "in-block-tree-reduction-shared",
                ["global-intermediates"] = "none",
                ["temporary-device-allocation"] = "none",
                ["stride-parameters"] = "none"
            });
    }

    internal static bool IsSupportedShape(int m, int n) => PtxRowShape.IsSupported(m, n);

    internal static bool IsPromotedShape(int m, int n) => PtxRowShape.IsPromoted(m, n);
}
