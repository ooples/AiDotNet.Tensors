using System;
using System.Collections.Generic;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// Row-wise numerically-stable log-softmax <c>output[m,n] = x[m,n] - logsumexp(x[m,:])</c>
/// over the last axis (issue #840), where <c>logsumexp = rowMax + log(sum exp(x - rowMax))</c>.
/// One block owns one row and, in a single shared-resident pass, computes the row max and
/// exp-sum with in-block tree reductions, then subtracts the log-partition from the cached
/// row — no global max/sum intermediate. Uses <c>ex2.approx.f32</c>/<c>lg2.approx.f32</c>, so
/// a promoted specialization carries ~1e-3 approximation error (disclosed on the release gate).
///
/// One block per row (grid = M), 256 threads. Shared: N floats (row cache) + 256 floats
/// (reduction). Supported N are multiples of 256 so each thread strides the row exactly.
/// </summary>
internal sealed class PtxLogSoftmaxKernel : IDisposable
{
    internal const int BlockThreads = 256;
    internal const string EntryPoint = "aidotnet_log_softmax_row";

    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

    internal int M { get; }
    internal int N { get; }
    internal string Ptx { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal PtxLogSoftmaxKernel(DirectPtxRuntime runtime, int m, int n)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        if (!DirectPtxArchitecture.HasValidatedSoftmax(
            runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new PlatformNotSupportedException(
                "The checked-in log-softmax specialization is measured only on GA10x/SM86.");
        ValidateShape(m, n);
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

    internal unsafe void Launch(DirectPtxTensorView input, DirectPtxTensorView output)
    {
        Require(input, Blueprint.Tensors[0], nameof(input));
        Require(output, Blueprint.Tensors[1], nameof(output));

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
        ValidateShape(m, n);
        int rowBytes = checked(n * sizeof(float));
        const string Log2e = "0f3FB8AA3B";  // 1.4426950408889634
        const string Ln2 = "0f3F317218";    // 0.6931471805599453
        const string NegInf = "0fFF800000";

        var ptx = new StringBuilder(10_000);
        ptx.AppendLine(".version 7.1");
        ptx.AppendLine($".target sm_{ccMajor}{ccMinor}");
        ptx.AppendLine(".address_size 64");
        ptx.AppendLine($"// log-softmax-row M={m} N={n}");
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
        ptx.AppendLine($"    .shared .align 16 .b8 row_sh[{n * sizeof(float)}];");
        ptx.AppendLine($"    .shared .align 16 .b8 red[{BlockThreads * sizeof(float)}];");
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

        // ---- Pass 1: cache row + partial max ----
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
        ptx.AppendLine("    st.shared.f32 [%rd10], %f0;");
        ptx.AppendLine("    bar.sync 0;");
        EmitTreeReduce(ptx, "max.f32");
        ptx.AppendLine("    ld.shared.f32 %f2, [%rd5];");                // rowMax
        ptx.AppendLine("    bar.sync 0;");

        // ---- Pass 2: partial sum of exp(x - rowMax) ----
        ptx.AppendLine("    mov.f32 %f0, 0f00000000;");
        ptx.AppendLine("    mov.u32 %r3, %r0;");
        ptx.AppendLine("SUM_LOOP:");
        ptx.AppendLine($"    setp.ge.u32 %p0, %r3, {n};");
        ptx.AppendLine("    @%p0 bra.uni SUM_DONE;");
        ptx.AppendLine("    mul.wide.u32 %rd11, %r3, 4;");
        ptx.AppendLine("    add.u64 %rd13, %rd4, %rd11;");
        ptx.AppendLine("    ld.shared.f32 %f1, [%rd13];");
        ptx.AppendLine("    sub.rn.f32 %f1, %f1, %f2;");
        ptx.AppendLine($"    mul.rn.f32 %f1, %f1, {Log2e};");
        ptx.AppendLine("    ex2.approx.f32 %f1, %f1;");
        ptx.AppendLine("    add.rn.f32 %f0, %f0, %f1;");
        ptx.AppendLine($"    add.u32 %r3, %r3, {BlockThreads};");
        ptx.AppendLine("    bra.uni SUM_LOOP;");
        ptx.AppendLine("SUM_DONE:");
        ptx.AppendLine("    st.shared.f32 [%rd10], %f0;");
        ptx.AppendLine("    bar.sync 0;");
        EmitTreeReduce(ptx, "add.rn.f32");
        ptx.AppendLine("    ld.shared.f32 %f3, [%rd5];");                // sumExp
        ptx.AppendLine("    bar.sync 0;");
        // logZ = rowMax + ln2 * lg2(sumExp)
        ptx.AppendLine("    lg2.approx.f32 %f4, %f3;");
        ptx.AppendLine($"    fma.rn.f32 %f4, %f4, {Ln2}, %f2;");        // logZ

        // ---- Pass 3: output = x - logZ ----
        ptx.AppendLine("    mov.u32 %r3, %r0;");
        ptx.AppendLine("OUT_LOOP:");
        ptx.AppendLine($"    setp.ge.u32 %p0, %r3, {n};");
        ptx.AppendLine("    @%p0 bra.uni OUT_DONE;");
        ptx.AppendLine("    mul.wide.u32 %rd11, %r3, 4;");
        ptx.AppendLine("    add.u64 %rd13, %rd4, %rd11;");
        ptx.AppendLine("    ld.shared.f32 %f1, [%rd13];");
        ptx.AppendLine("    sub.rn.f32 %f1, %f1, %f4;");                // x - logZ
        ptx.AppendLine("    add.u64 %rd14, %rd8, %rd11;");
        ptx.AppendLine("    st.global.f32 [%rd14], %f1;");
        ptx.AppendLine($"    add.u32 %r3, %r3, {BlockThreads};");
        ptx.AppendLine("    bra.uni OUT_LOOP;");
        ptx.AppendLine("OUT_DONE:");
        ptx.AppendLine("    ret;");
        ptx.AppendLine("}");
        return ptx.ToString();
    }

    private static void EmitTreeReduce(StringBuilder ptx, string op)
    {
        foreach (int stride in new[] { 128, 64, 32, 16, 8, 4, 2, 1 })
        {
            ptx.AppendLine($"    setp.lt.u32 %p3, %r0, {stride};");
            ptx.AppendLine("    @%p3 ld.shared.f32 %f10, [%rd10];");
            ptx.AppendLine($"    @%p3 ld.shared.f32 %f11, [%rd10+{stride * sizeof(float)}];");
            ptx.AppendLine($"    @%p3 {op} %f10, %f10, %f11;");
            ptx.AppendLine("    @%p3 st.shared.f32 [%rd10], %f10;");
            ptx.AppendLine("    bar.sync 0;");
        }
    }

    private static DirectPtxKernelBlueprint CreateBlueprint(
        DirectPtxArchitectureFamily architecture, int m, int n)
    {
        var extent = new DirectPtxExtent(m, n);
        return new DirectPtxKernelBlueprint(
            Operation: "log-softmax-row",
            Version: 1,
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
                MaxStaticSharedBytes: (n + BlockThreads) * sizeof(float),
                MaxLocalBytesPerThread: 0,
                MinBlocksPerMultiprocessor: 1),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["formula"] = "output[m,n] = x[m,n] - (rowMax[m] + log(sum_n exp(x[m,n] - rowMax[m])))",
                ["axis"] = "last",
                ["stability"] = "row-max-subtracted-logsumexp",
                ["reduction"] = "in-block-tree-reduction-shared",
                ["global-intermediates"] = "none",
                ["temporary-device-allocation"] = "none",
                ["stride-parameters"] = "none"
            });
    }

    internal static bool IsSupportedShape(int m, int n) =>
        m > 0 && m % 64 == 0 &&
        n > 0 && n % BlockThreads == 0 &&
        m is 64 or 128 or 256 or 512 or 1024 or 2048 &&
        n is 256 or 512 or 1024 or 2048 or 4096;

    internal static bool IsPromotedShape(int m, int n) => false;

    private static void ValidateShape(int m, int n)
    {
        if (!IsSupportedShape(m, n))
            throw new ArgumentOutOfRangeException(
                nameof(m),
                "Log-softmax supports M in {64,128,256,512,1024,2048}, N in {256,512,1024,2048,4096}.");
    }

    private static void Require(DirectPtxTensorView view, DirectPtxTensorContract contract, string parameter)
    {
        if (view.Pointer == IntPtr.Zero || view.PhysicalType != contract.PhysicalType ||
            view.Layout != contract.Layout || view.LogicalExtent != contract.LogicalExtent ||
            view.PhysicalExtent != contract.PhysicalExtent || view.ByteLength != contract.RequiredBytes)
            throw new ArgumentException(
                $"{parameter} does not satisfy physical ABI '{contract.Name}'.", parameter);
    }
}
