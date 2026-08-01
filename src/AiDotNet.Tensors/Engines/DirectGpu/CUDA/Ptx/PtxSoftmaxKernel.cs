using System;
using System.Collections.Generic;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// Row-wise numerically-stable softmax <c>output[m,n] = exp(x[m,n] - rowMax) / rowSumExp</c>
/// over the last axis (issue #840). One block owns one row, computes its row max and exp-sum
/// with hierarchical warp reductions, stages exponentials in the final output, then writes the
/// normalized values in place. There is no separate global max/sum/probability allocation;
/// repeated row reads stay cacheable in L1. Uses <c>ex2.approx.f32</c>, so a
/// promoted specialization carries ~1e-3 approximation error (disclosed on the release gate).
///
/// One block per row (grid = M), 256 threads. Shared: 8 warp-leader reduction floats
/// (<c>PtxRowReduce.SharedBytes</c>). Supported N are multiples of 256 so each thread strides
/// the row exactly.
/// </summary>
internal sealed class PtxSoftmaxKernel : IDisposable
{
    internal const int BlockThreads = PtxRowShape.BlockThreads;
    internal const string EntryPoint = "aidotnet_softmax_row";

    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

    internal int M { get; }
    internal int N { get; }
    internal string Ptx { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal PtxSoftmaxKernel(DirectPtxRuntime runtime, int m, int n)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        if (!DirectPtxArchitecture.HasValidatedSoftmax(
            runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new PlatformNotSupportedException(
                "The checked-in softmax specialization is measured only on GA10x/SM86.");
        PtxRowShape.Validate(m, n, "Softmax");
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

    internal unsafe void Launch(DirectPtxTensorView input, DirectPtxTensorView output)
    {
        PtxAbiGuard.Require(input, Blueprint.Tensors[0], nameof(input));
        PtxAbiGuard.Require(output, Blueprint.Tensors[1], nameof(output));

        IntPtr inputPointer = input.Pointer;
        IntPtr outputPointer = output.Pointer;
        void** arguments = stackalloc void*[2];
        arguments[0] = &inputPointer;
        arguments[1] = &outputPointer;
        _module.Launch(_function, (uint)M, 1, 1, BlockThreads, 1, 1, 0, arguments);
    }

    public void Dispose() => _module.Dispose();

    internal static string EmitPtx(int ccMajor, int ccMinor, int m, int n)
    {
        PtxRowShape.Validate(m, n, "Softmax");
        int rowBytes = checked(n * sizeof(float));
        const string Log2e = "0f3FB8AA3B";  // 1.4426950408889634
        const string NegInf = "0fFF800000";

        var ptx = new StringBuilder(10_000);
        ptx.AppendLine(".version 7.1");
        ptx.AppendLine($".target sm_{ccMajor}{ccMinor}");
        ptx.AppendLine(".address_size 64");
        ptx.AppendLine($"// softmax-row M={m} N={n}");
        ptx.AppendLine();
        ptx.AppendLine($".visible .entry {EntryPoint}(");
        ptx.AppendLine("    .param .u64 input_ptr,");
        ptx.AppendLine("    .param .u64 output_ptr");
        ptx.AppendLine(")");
        ptx.AppendLine($".maxntid {BlockThreads}, 1, 1");
        ptx.AppendLine("{");
        ptx.AppendLine("    .reg .pred %p<4>;");
        ptx.AppendLine("    .reg .b32 %r<12>;");
        ptx.AppendLine("    .reg .b64 %rd<20>;");
        ptx.AppendLine("    .reg .f32 %f<20>;");
        ptx.AppendLine($"    .shared .align 16 .b8 red[{PtxRowReduce.SharedBytes}];");
        ptx.AppendLine("    ld.param.u64 %rd0, [input_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd1, [output_ptr];");
        ptx.AppendLine("    mov.u64 %rd5, red;");
        ptx.AppendLine("    mov.u32 %r0, %tid.x;");
        ptx.AppendLine("    mov.u32 %r1, %ctaid.x;");
        ptx.AppendLine($"    mul.wide.u32 %rd6, %r1, {rowBytes};");
        ptx.AppendLine("    add.u64 %rd7, %rd0, %rd6;");                 // &input[m,0]
        ptx.AppendLine("    add.u64 %rd8, %rd1, %rd6;");                 // &output[m,0]
        ptx.AppendLine("    mul.wide.u32 %rd9, %r0, 4;");
        ptx.AppendLine("    add.u64 %rd10, %rd5, %rd9;");               // &red[tid]

        // ---- Pass 1: partial max; the row remains hot in L1 for pass 2 ----
        ptx.AppendLine($"    mov.f32 %f0, {NegInf};");
        for (int column = 0; column < n; column += BlockThreads)
        {
            ptx.AppendLine($"    add.u64 %rd11, %rd9, {column * sizeof(float)};");
            ptx.AppendLine("    add.u64 %rd12, %rd7, %rd11;");
            ptx.AppendLine("    ld.global.ca.f32 %f1, [%rd12];");
            ptx.AppendLine("    max.f32 %f0, %f0, %f1;");
        }
        PtxRowReduce.Emit(ptx, "max.f32", "%f0");
        ptx.AppendLine("    ld.shared.f32 %f2, [%rd5];");                // rowMax
        ptx.AppendLine("    bar.sync 0;");

        // ---- Pass 2: partial sum of exp(x - rowMax) ----
        ptx.AppendLine("    mov.f32 %f0, 0f00000000;");
        for (int column = 0; column < n; column += BlockThreads)
        {
            ptx.AppendLine($"    add.u64 %rd11, %rd9, {column * sizeof(float)};");
            ptx.AppendLine("    add.u64 %rd12, %rd7, %rd11;");
            ptx.AppendLine("    ld.global.ca.f32 %f1, [%rd12];");
            ptx.AppendLine("    sub.rn.f32 %f1, %f1, %f2;");
            ptx.AppendLine($"    mul.rn.f32 %f1, %f1, {Log2e};");
            ptx.AppendLine("    ex2.approx.f32 %f1, %f1;");
            ptx.AppendLine("    add.u64 %rd14, %rd8, %rd11;");
            ptx.AppendLine("    st.global.f32 [%rd14], %f1;");
            ptx.AppendLine("    add.rn.f32 %f0, %f0, %f1;");
        }
        PtxRowReduce.Emit(ptx, "add.rn.f32", "%f0");
        ptx.AppendLine("    ld.shared.f32 %f3, [%rd5];");                // sumExp
        ptx.AppendLine("    bar.sync 0;");
        ptx.AppendLine("    rcp.approx.f32 %f4, %f3;");                  // 1/sumExp

        // ---- Pass 3: normalize the staged output in place ----
        for (int column = 0; column < n; column += BlockThreads)
        {
            ptx.AppendLine($"    add.u64 %rd11, %rd9, {column * sizeof(float)};");
            ptx.AppendLine("    add.u64 %rd14, %rd8, %rd11;");
            ptx.AppendLine("    ld.global.ca.f32 %f1, [%rd14];");
            ptx.AppendLine("    mul.rn.f32 %f1, %f1, %f4;");
            ptx.AppendLine("    st.global.f32 [%rd14], %f1;");
        }
        ptx.AppendLine("    ret;");
        ptx.AppendLine("}");
        return ptx.ToString();
    }

    private static DirectPtxKernelBlueprint CreateBlueprint(
        DirectPtxArchitectureFamily architecture, int m, int n)
    {
        var extent = new DirectPtxExtent(m, n);
        return new DirectPtxKernelBlueprint(
            Operation: "softmax-row",
            Version: 2,
            Architecture: architecture,
            Variant: $"fp32-m{m}-n{n}",
            Tensors:
            [
                new("input", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D,
                    extent, extent, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("output", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D,
                    extent, extent, 16, DirectPtxTensorAccess.Write, DirectPtxExtentMode.Exact)
            ],
            ResourceBudget: new DirectPtxResourceBudget(
                MaxRegistersPerThread: 32,
                MaxStaticSharedBytes: PtxRowReduce.SharedBytes,
                MaxLocalBytesPerThread: 0,
                MinBlocksPerMultiprocessor: 1),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["formula"] = "output[m,n] = exp(x[m,n] - rowMax[m]) / sum_n exp(x[m,n] - rowMax[m])",
                ["axis"] = "last",
                ["stability"] = "row-max-subtracted",
                ["reduction"] = PtxRowReduce.Strategy,
                ["global-intermediates"] = "none",
                ["output-staging"] = "exponentials-normalized-in-place",
                ["temporary-device-allocation"] = "none",
                ["stride-parameters"] = "none"
            });
    }

    internal static bool IsSupportedShape(int m, int n) => PtxRowShape.IsSupported(m, n);

    internal static bool IsPromotedShape(int m, int n) => PtxRowShape.IsPromoted(m, n);
}
