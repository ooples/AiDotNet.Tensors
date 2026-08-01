using System;
using System.Collections.Generic;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// Row-wise log-sum-exp backward <c>dX[m,n] = exp(x[m,n] - lse[m]) * dY[m]</c> over the last
/// axis (issue #840), where <c>lse</c> is the caller-provided [M] forward log-partition and
/// <c>dY</c> is its [M] upstream gradient. One block owns one row and reuses those forward
/// statistics directly, matching the incumbent CUDA contract while avoiding a redundant max
/// and exp-sum reduction. Uses <c>ex2.approx.f32</c>, so a promoted specialization carries
/// ~1e-3 approximation error (disclosed on the release gate).
///
/// One block per row (grid = M), 256 threads, no shared memory. Supported N are multiples of
/// 256 so each thread strides the row exactly.
/// </summary>
internal sealed class PtxLogSumExpBackwardKernel : IDisposable
{
    internal const int BlockThreads = PtxRowShape.BlockThreads;
    internal const string EntryPoint = "aidotnet_logsumexp_backward_row";

    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

    internal int M { get; }
    internal int N { get; }
    internal string Ptx { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal PtxLogSumExpBackwardKernel(DirectPtxRuntime runtime, int m, int n)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        if (!DirectPtxArchitecture.HasValidatedSoftmax(
            runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new PlatformNotSupportedException(
                "The checked-in log-sum-exp-backward specialization is measured only on GA10x/SM86.");
        PtxRowShape.Validate(m, n, "Log-sum-exp backward");
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
                    module.JitInfoLog);
                return (Function: function, Audit: audit);
            });
        _module = loaded.Resource;
        _function = loaded.Value.Function;
        Audit = loaded.Value.Audit;
    }

    internal unsafe void Launch(
        DirectPtxTensorView input,
        DirectPtxTensorView logSumExp,
        DirectPtxTensorView grad,
        DirectPtxTensorView output)
    {
        PtxAbiGuard.Require(input, Blueprint.Tensors[0], nameof(input));
        PtxAbiGuard.Require(logSumExp, Blueprint.Tensors[1], nameof(logSumExp));
        PtxAbiGuard.Require(grad, Blueprint.Tensors[2], nameof(grad));
        PtxAbiGuard.Require(output, Blueprint.Tensors[3], nameof(output));

        IntPtr inputPointer = input.Pointer;
        IntPtr logSumExpPointer = logSumExp.Pointer;
        IntPtr gradPointer = grad.Pointer;
        IntPtr outputPointer = output.Pointer;
        void** arguments = stackalloc void*[4];
        arguments[0] = &inputPointer;
        arguments[1] = &logSumExpPointer;
        arguments[2] = &gradPointer;
        arguments[3] = &outputPointer;
        _module.Launch(_function, (uint)M, 1, 1, BlockThreads, 1, 1, 0, arguments);
    }

    public void Dispose() => _module.Dispose();

    internal static string EmitPtx(int ccMajor, int ccMinor, int m, int n)
    {
        PtxRowShape.Validate(m, n, "Log-sum-exp backward");
        int rowBytes = checked(n * sizeof(float));
        const string Log2e = "0f3FB8AA3B";

        var ptx = new StringBuilder(5_000);
        ptx.AppendLine(".version 7.1");
        ptx.AppendLine($".target sm_{ccMajor}{ccMinor}");
        ptx.AppendLine(".address_size 64");
        ptx.AppendLine($"// logsumexp-backward-row M={m} N={n}");
        ptx.AppendLine();
        ptx.AppendLine($".visible .entry {EntryPoint}(");
        ptx.AppendLine("    .param .u64 input_ptr,");
        ptx.AppendLine("    .param .u64 lse_ptr,");
        ptx.AppendLine("    .param .u64 grad_ptr,");
        ptx.AppendLine("    .param .u64 output_ptr");
        ptx.AppendLine(")");
        ptx.AppendLine($".maxntid {BlockThreads}, 1, 1");
        ptx.AppendLine("{");
        ptx.AppendLine("    .reg .pred %p<2>;");
        ptx.AppendLine("    .reg .b32 %r<8>;");
        ptx.AppendLine("    .reg .b64 %rd<16>;");
        ptx.AppendLine("    .reg .f32 %f<8>;");
        ptx.AppendLine("    ld.param.u64 %rd0, [input_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd1, [lse_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd2, [grad_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd3, [output_ptr];");
        ptx.AppendLine("    mov.u32 %r0, %tid.x;");
        ptx.AppendLine("    mov.u32 %r1, %ctaid.x;");
        ptx.AppendLine($"    mul.wide.u32 %rd4, %r1, {rowBytes};");
        ptx.AppendLine("    add.u64 %rd5, %rd0, %rd4;");               // &input[m,0]
        ptx.AppendLine("    add.u64 %rd6, %rd3, %rd4;");               // &output[m,0]
        ptx.AppendLine("    mul.wide.u32 %rd7, %r1, 4;");
        ptx.AppendLine("    add.u64 %rd8, %rd1, %rd7;");               // &lse[m]
        ptx.AppendLine("    add.u64 %rd9, %rd2, %rd7;");               // &grad[m]
        ptx.AppendLine("    ld.global.nc.f32 %f0, [%rd8];");           // lse[m]
        ptx.AppendLine("    ld.global.nc.f32 %f1, [%rd9];");           // dY[m]
        ptx.AppendLine("    mov.u32 %r2, %r0;");
        ptx.AppendLine("OUT_LOOP:");
        ptx.AppendLine($"    setp.ge.u32 %p0, %r2, {n};");
        ptx.AppendLine("    @%p0 bra.uni OUT_DONE;");
        ptx.AppendLine("    mul.wide.u32 %rd10, %r2, 4;");
        ptx.AppendLine("    add.u64 %rd11, %rd5, %rd10;");
        ptx.AppendLine("    ld.global.nc.f32 %f2, [%rd11];");
        ptx.AppendLine("    sub.rn.f32 %f2, %f2, %f0;");
        ptx.AppendLine($"    mul.rn.f32 %f2, %f2, {Log2e};");
        ptx.AppendLine("    ex2.approx.f32 %f2, %f2;");
        ptx.AppendLine("    mul.rn.f32 %f2, %f2, %f1;");
        ptx.AppendLine("    add.u64 %rd12, %rd6, %rd10;");
        ptx.AppendLine("    st.global.f32 [%rd12], %f2;");
        ptx.AppendLine($"    add.u32 %r2, %r2, {BlockThreads};");
        ptx.AppendLine("    bra.uni OUT_LOOP;");
        ptx.AppendLine("OUT_DONE:");
        ptx.AppendLine("    ret;");
        ptx.AppendLine("}");
        return ptx.ToString();
    }

    private static DirectPtxKernelBlueprint CreateBlueprint(
        DirectPtxArchitectureFamily architecture, int m, int n)
    {
        var matrix = new DirectPtxExtent(m, n);
        var vector = new DirectPtxExtent(m);
        return new DirectPtxKernelBlueprint(
            Operation: "logsumexp-backward-row",
            Version: 2,
            Architecture: architecture,
            Variant: $"fp32-m{m}-n{n}",
            Tensors:
            [
                new("input", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D,
                    matrix, matrix, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("logSumExp", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    vector, vector, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("grad", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    vector, vector, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("output", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D,
                    matrix, matrix, 16, DirectPtxTensorAccess.Write, DirectPtxExtentMode.Exact)
            ],
            ResourceBudget: new DirectPtxResourceBudget(
                MaxRegistersPerThread: 16,
                MaxStaticSharedBytes: 0,
                MaxLocalBytesPerThread: 0,
                MinBlocksPerMultiprocessor: 1),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["formula"] = "dX[m,n] = exp(x[m,n] - lse[m]) * dY[m]",
                ["axis"] = "last",
                ["broadcast"] = "per-row-scalar-upstream-gradient",
                ["forward-statistics"] = "caller-provided-logsumexp-vector",
                ["reduction"] = "none-forward-logsumexp-reused",
                ["global-intermediates"] = "caller-provided-lse-input",
                ["temporary-device-allocation"] = "none",
                ["stride-parameters"] = "none"
            });
    }

    internal static bool IsSupportedShape(int m, int n) => PtxRowShape.IsSupported(m, n);

    internal static bool IsPromotedShape(int m, int n) => PtxRowShape.IsPromoted(m, n);
}
