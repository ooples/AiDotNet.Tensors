using System;
using System.Collections.Generic;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// Row-wise sparsemax <c>output = max(z - tau, 0)</c> — the Euclidean projection of each
/// logit row onto the probability simplex (issue #840) — where the threshold <c>tau</c> is
/// the unique root of <c>sum_n max(z[n] - tau, 0) = 1</c>. Rather than the classic sorted
/// closed form, this finds <c>tau</c> by bisection: the sum is continuous and strictly
/// decreasing in <c>tau</c> on <c>[rowMax - 1, rowMax]</c> (endpoints bracket 1 and 0), so
/// <see cref="BisectionSteps"/> halvings resolve it to FP32 precision. One block owns one
/// row and caches it in shared memory; each bisection step reduces its sum with warp shuffles
/// and a warp-leader shared exchange.
/// No sorting, no global intermediate; the projection is exact to the bisection tolerance.
///
/// One block per row (grid = M), 256 threads. Shared: N floats (row cache) + 8 floats
/// (warp-leader exchange). Supported N are multiples of 256 so each thread strides the row exactly.
/// </summary>
internal sealed class PtxSparsemaxKernel : IDisposable
{
    internal const int BlockThreads = PtxRowShape.BlockThreads;
    internal const int BisectionSteps = 30;
    internal const string EntryPoint = "aidotnet_sparsemax_row";

    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

    internal int M { get; }
    internal int N { get; }
    internal string Ptx { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal PtxSparsemaxKernel(DirectPtxRuntime runtime, int m, int n)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        if (!DirectPtxArchitecture.HasValidatedSoftmax(
            runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new PlatformNotSupportedException(
                "The checked-in sparsemax specialization is measured only on GA10x/SM86.");
        PtxRowShape.Validate(m, n, "Sparsemax");
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
        PtxRowShape.Validate(m, n, "Sparsemax");
        int rowBytes = checked(n * sizeof(float));
        const string Half = "0f3F000000"; // 0.5
        const string One = "0f3F800000";  // 1.0
        const string NegInf = "0fFF800000";

        var ptx = new StringBuilder(11_000);
        ptx.AppendLine(".version 7.1");
        ptx.AppendLine($".target sm_{ccMajor}{ccMinor}");
        ptx.AppendLine(".address_size 64");
        ptx.AppendLine($"// sparsemax-row M={m} N={n} bisect={BisectionSteps}");
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
        ptx.AppendLine("    .reg .f32 %f<16>;");
        ptx.AppendLine($"    .shared .align 16 .b8 row_sh[{n * sizeof(float)}];");
        ptx.AppendLine($"    .shared .align 16 .b8 red[{PtxRowReduce.SharedBytes}];");
        ptx.AppendLine("    ld.param.u64 %rd0, [input_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd1, [output_ptr];");
        ptx.AppendLine("    mov.u64 %rd4, row_sh;");
        ptx.AppendLine("    mov.u64 %rd5, red;");
        ptx.AppendLine("    mov.u32 %r0, %tid.x;");
        ptx.AppendLine("    mov.u32 %r1, %ctaid.x;");
        ptx.AppendLine($"    mul.wide.u32 %rd6, %r1, {rowBytes};");
        ptx.AppendLine("    add.u64 %rd7, %rd0, %rd6;");                 // &input[m,0]
        ptx.AppendLine("    add.u64 %rd8, %rd1, %rd6;");                 // &output[m,0]
        ptx.AppendLine("    mul.wide.u32 %rd9, %r0, 4;");
        ptx.AppendLine("    add.u64 %rd10, %rd5, %rd9;");               // &red[tid]

        // ---- Pass 1: cache row + partial max (bracket for tau) ----
        ptx.AppendLine($"    mov.f32 %f0, {NegInf};");
        ptx.AppendLine("    mov.u32 %r3, %r0;");
        ptx.AppendLine("LOAD_LOOP:");
        ptx.AppendLine($"    setp.ge.u32 %p0, %r3, {n};");
        ptx.AppendLine("    @%p0 bra.uni LOAD_DONE;");
        ptx.AppendLine("    mul.wide.u32 %rd11, %r3, 4;");
        ptx.AppendLine("    add.u64 %rd12, %rd7, %rd11;");
        ptx.AppendLine("    ld.global.nc.f32 %f1, [%rd12];");
        ptx.AppendLine("    add.u64 %rd13, %rd4, %rd11;");
        ptx.AppendLine("    st.shared.f32 [%rd13], %f1;");
        ptx.AppendLine("    max.f32 %f0, %f0, %f1;");
        ptx.AppendLine($"    add.u32 %r3, %r3, {BlockThreads};");
        ptx.AppendLine("    bra.uni LOAD_LOOP;");
        ptx.AppendLine("LOAD_DONE:");
        PtxRowReduce.Emit(ptx, "max.f32", "%f0");
        ptx.AppendLine("    ld.shared.f32 %f2, [%rd5];");                // rowMax
        ptx.AppendLine("    bar.sync 0;");
        // lo = rowMax - 1 (S >= 1), hi = rowMax (S = 0).
        ptx.AppendLine($"    add.rn.f32 %f5, %f2, 0fBF800000;");         // lo = rowMax - 1
        ptx.AppendLine("    mov.f32 %f6, %f2;");                          // hi = rowMax

        // ---- Bisection: 30 halvings of tau ----
        ptx.AppendLine("    mov.u32 %r4, 0;");
        ptx.AppendLine("BISECT_LOOP:");
        ptx.AppendLine("    add.rn.f32 %f7, %f5, %f6;");
        ptx.AppendLine($"    mul.rn.f32 %f7, %f7, {Half};");            // mid
        ptx.AppendLine("    mov.f32 %f0, 0f00000000;");                 // partial S
        ptx.AppendLine("    mov.u32 %r3, %r0;");
        ptx.AppendLine("SUM_LOOP:");
        ptx.AppendLine($"    setp.ge.u32 %p0, %r3, {n};");
        ptx.AppendLine("    @%p0 bra.uni SUM_DONE;");
        ptx.AppendLine("    mul.wide.u32 %rd11, %r3, 4;");
        ptx.AppendLine("    add.u64 %rd13, %rd4, %rd11;");
        ptx.AppendLine("    ld.shared.f32 %f1, [%rd13];");
        ptx.AppendLine("    sub.rn.f32 %f1, %f1, %f7;");                // z - mid
        ptx.AppendLine("    max.f32 %f1, %f1, 0f00000000;");            // relu
        ptx.AppendLine("    add.rn.f32 %f0, %f0, %f1;");
        ptx.AppendLine($"    add.u32 %r3, %r3, {BlockThreads};");
        ptx.AppendLine("    bra.uni SUM_LOOP;");
        ptx.AppendLine("SUM_DONE:");
        PtxRowReduce.Emit(ptx, "add.rn.f32", "%f0");
        ptx.AppendLine("    ld.shared.f32 %f3, [%rd5];");                // S(mid)
        ptx.AppendLine("    bar.sync 0;");
        // S(mid) > 1  => tau is larger => raise lo; else lower hi.
        ptx.AppendLine($"    setp.gt.f32 %p1, %f3, {One};");
        ptx.AppendLine("    @%p1 mov.f32 %f5, %f7;");
        ptx.AppendLine("    @!%p1 mov.f32 %f6, %f7;");
        ptx.AppendLine("    add.u32 %r4, %r4, 1;");
        ptx.AppendLine($"    setp.lt.u32 %p2, %r4, {BisectionSteps};");
        ptx.AppendLine("    @%p2 bra.uni BISECT_LOOP;");
        // tau = midpoint of the final bracket.
        ptx.AppendLine("    add.rn.f32 %f8, %f5, %f6;");
        ptx.AppendLine($"    mul.rn.f32 %f8, %f8, {Half};");

        // ---- Output: max(z - tau, 0) ----
        ptx.AppendLine("    mov.u32 %r3, %r0;");
        ptx.AppendLine("OUT_LOOP:");
        ptx.AppendLine($"    setp.ge.u32 %p0, %r3, {n};");
        ptx.AppendLine("    @%p0 bra.uni OUT_DONE;");
        ptx.AppendLine("    mul.wide.u32 %rd11, %r3, 4;");
        ptx.AppendLine("    add.u64 %rd13, %rd4, %rd11;");
        ptx.AppendLine("    ld.shared.f32 %f1, [%rd13];");
        ptx.AppendLine("    sub.rn.f32 %f1, %f1, %f8;");
        ptx.AppendLine("    max.f32 %f1, %f1, 0f00000000;");
        ptx.AppendLine("    add.u64 %rd14, %rd8, %rd11;");
        ptx.AppendLine("    st.global.f32 [%rd14], %f1;");
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
            Operation: "sparsemax-row",
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
                MaxStaticSharedBytes: n * sizeof(float) + PtxRowReduce.SharedBytes,
                MaxLocalBytesPerThread: 0,
                MinBlocksPerMultiprocessor: 1),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["formula"] = "output[m,n] = max(z[m,n] - tau[m], 0), tau s.t. sum_n max(z - tau, 0) = 1",
                ["axis"] = "last",
                ["method"] = $"bisection-{BisectionSteps}-steps-on-[rowMax-1,rowMax]",
                ["reduction"] = PtxRowReduce.Strategy,
                ["global-intermediates"] = "none",
                ["temporary-device-allocation"] = "none",
                ["stride-parameters"] = "none"
            });
    }

    internal static bool IsSupportedShape(int m, int n) => PtxRowShape.IsSupported(m, n);

    internal static bool IsPromotedShape(int m, int n) => PtxRowShape.IsPromoted(m, n);
}
