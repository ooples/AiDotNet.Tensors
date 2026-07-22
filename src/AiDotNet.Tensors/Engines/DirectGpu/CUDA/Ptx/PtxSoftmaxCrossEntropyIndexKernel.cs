using System;
using System.Collections.Generic;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// Fused row-wise softmax + indexed cross-entropy loss and gradient over precomputed
/// logits, the loss half of <c>FusedLinearCrossEntropyIndex</c>. Given
/// <c>logits[M,N]</c> (= hidden @ W + bias, produced by the GemmBias tile) and integer
/// class targets <c>targets[M]</c>, one block owns one row and, in a single resident
/// pass, computes the numerically-stable row max, the exp-sum, the per-row loss
/// <c>logsumexp - logit[target]</c> and the gradient <c>dLogits = softmax - onehot(target)</c>.
/// The row is cached in shared memory and both reductions use an in-block tree reduction,
/// so there is no global intermediate.
///
/// One block per row (grid = M), 256 threads. Shared: N floats (row cache) + 256 floats
/// (reduction). Supported N are multiples of 256 so each thread strides the row exactly.
/// Uses <c>ex2.approx.f32</c>/<c>lg2.approx.f32</c>, so the loss carries ~1e-3
/// approximation error (disclosed on the release gate).
/// </summary>
internal sealed class PtxSoftmaxCrossEntropyIndexKernel : IDisposable
{
    internal const int BlockThreads = 256;
    internal const string EntryPoint = "aidotnet_softmax_cross_entropy_index";

    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

    internal int M { get; }
    internal int N { get; }
    internal string Ptx { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal PtxSoftmaxCrossEntropyIndexKernel(DirectPtxRuntime runtime, int m, int n)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        if (!DirectPtxArchitecture.HasValidatedFusedLinear(
            runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new PlatformNotSupportedException(
                "The checked-in softmax-cross-entropy specialization is measured only on GA10x/SM86.");
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

    internal unsafe void Launch(
        DirectPtxTensorView logits,
        DirectPtxTensorView targets,
        DirectPtxTensorView loss,
        DirectPtxTensorView dLogits)
    {
        Require(logits, Blueprint.Tensors[0], nameof(logits));
        Require(targets, Blueprint.Tensors[1], nameof(targets));
        Require(loss, Blueprint.Tensors[2], nameof(loss));
        Require(dLogits, Blueprint.Tensors[3], nameof(dLogits));

        IntPtr logitsPointer = logits.Pointer;
        IntPtr targetsPointer = targets.Pointer;
        IntPtr lossPointer = loss.Pointer;
        IntPtr dLogitsPointer = dLogits.Pointer;
        void** arguments = stackalloc void*[4];
        arguments[0] = &logitsPointer;
        arguments[1] = &targetsPointer;
        arguments[2] = &lossPointer;
        arguments[3] = &dLogitsPointer;
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
        const string One = "0f3F800000";

        var ptx = new StringBuilder(12_000);
        ptx.AppendLine(".version 7.1");
        ptx.AppendLine($".target sm_{ccMajor}{ccMinor}");
        ptx.AppendLine(".address_size 64");
        ptx.AppendLine($"// softmax-cross-entropy-index M={m} N={n}");
        ptx.AppendLine();
        ptx.AppendLine($".visible .entry {EntryPoint}(");
        ptx.AppendLine("    .param .u64 logits_ptr,");
        ptx.AppendLine("    .param .u64 targets_ptr,");
        ptx.AppendLine("    .param .u64 loss_ptr,");
        ptx.AppendLine("    .param .u64 dlogits_ptr");
        ptx.AppendLine(")");
        ptx.AppendLine($".maxntid {BlockThreads}, 1, 1");
        ptx.AppendLine("{");
        ptx.AppendLine("    .reg .pred %p<6>;");
        ptx.AppendLine("    .reg .b32 %r<16>;");
        ptx.AppendLine("    .reg .b64 %rd<24>;");
        ptx.AppendLine("    .reg .f32 %f<24>;");
        ptx.AppendLine($"    .shared .align 16 .b8 row_sh[{n * sizeof(float)}];");
        ptx.AppendLine($"    .shared .align 16 .b8 red[{BlockThreads * sizeof(float)}];");
        ptx.AppendLine("    ld.param.u64 %rd0, [logits_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd1, [targets_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd2, [loss_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd3, [dlogits_ptr];");
        ptx.AppendLine("    mov.u64 %rd4, row_sh;");
        ptx.AppendLine("    mov.u64 %rd5, red;");
        ptx.AppendLine("    mov.u32 %r0, %tid.x;");
        ptx.AppendLine("    mov.u32 %r1, %ctaid.x;");                    // row m
        ptx.AppendLine($"    mul.wide.u32 %rd6, %r1, {rowBytes};");       // m * N * 4
        ptx.AppendLine("    add.u64 %rd7, %rd0, %rd6;");                 // &logits[m,0]
        ptx.AppendLine("    add.u64 %rd8, %rd3, %rd6;");                 // &dLogits[m,0]
        ptx.AppendLine("    mul.wide.u32 %rd9, %r0, 4;");
        ptx.AppendLine("    add.u64 %rd10, %rd5, %rd9;");                // &red[tid]

        // ---- Pass 1: cache row + partial max ----
        ptx.AppendLine($"    mov.f32 %f0, {NegInf};");
        ptx.AppendLine("    mov.u32 %r3, %r0;");                         // n = tid
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
        // logZ = rowMax + ln2 * lg2(sumExp); inv = 1/sumExp
        ptx.AppendLine("    lg2.approx.f32 %f4, %f3;");
        ptx.AppendLine($"    fma.rn.f32 %f4, %f4, {Ln2}, %f2;");         // logZ
        ptx.AppendLine("    rcp.approx.f32 %f5, %f3;");                  // inv

        // ---- Pass 3: dLogits = softmax - onehot(target) ----
        ptx.AppendLine("    mul.wide.u32 %rd14, %r1, 4;");
        ptx.AppendLine("    add.u64 %rd15, %rd1, %rd14;");
        ptx.AppendLine("    ld.global.nc.u32 %r4, [%rd15];");           // target index
        ptx.AppendLine("    mov.u32 %r3, %r0;");
        ptx.AppendLine("GRAD_LOOP:");
        ptx.AppendLine($"    setp.ge.u32 %p0, %r3, {n};");
        ptx.AppendLine("    @%p0 bra.uni GRAD_DONE;");
        ptx.AppendLine("    mul.wide.u32 %rd11, %r3, 4;");
        ptx.AppendLine("    add.u64 %rd13, %rd4, %rd11;");
        ptx.AppendLine("    ld.shared.f32 %f1, [%rd13];");
        ptx.AppendLine("    sub.rn.f32 %f1, %f1, %f2;");
        ptx.AppendLine($"    mul.rn.f32 %f1, %f1, {Log2e};");
        ptx.AppendLine("    ex2.approx.f32 %f1, %f1;");
        ptx.AppendLine("    mul.rn.f32 %f1, %f1, %f5;");                 // softmax
        ptx.AppendLine("    setp.eq.u32 %p1, %r3, %r4;");
        ptx.AppendLine($"    selp.f32 %f6, {One}, 0f00000000, %p1;");
        ptx.AppendLine("    sub.rn.f32 %f1, %f1, %f6;");                 // softmax - onehot
        ptx.AppendLine("    add.u64 %rd16, %rd8, %rd11;");
        ptx.AppendLine("    st.global.f32 [%rd16], %f1;");
        ptx.AppendLine($"    add.u32 %r3, %r3, {BlockThreads};");
        ptx.AppendLine("    bra.uni GRAD_LOOP;");
        ptx.AppendLine("GRAD_DONE:");

        // ---- Loss (thread 0): logZ - logit[target] ----
        ptx.AppendLine("    setp.ne.u32 %p2, %r0, 0;");
        ptx.AppendLine("    @%p2 bra.uni CE_END;");
        ptx.AppendLine("    mul.wide.u32 %rd17, %r4, 4;");
        ptx.AppendLine("    add.u64 %rd18, %rd4, %rd17;");
        ptx.AppendLine("    ld.shared.f32 %f7, [%rd18];");
        ptx.AppendLine("    sub.rn.f32 %f7, %f4, %f7;");                 // logZ - logit[target]
        ptx.AppendLine("    add.u64 %rd19, %rd2, %rd14;");
        ptx.AppendLine("    st.global.f32 [%rd19], %f7;");
        ptx.AppendLine("CE_END:");
        ptx.AppendLine("    ret;");
        ptx.AppendLine("}");
        return ptx.ToString();
    }

    // In-block tree reduction over red[0..255] using the given op, leaving the result in
    // red[0]. Predicated (no divergent branches); a barrier separates each halving.
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
        var logits = new DirectPtxExtent(m, n);
        var targets = new DirectPtxExtent(m);
        var loss = new DirectPtxExtent(m);
        var dLogits = new DirectPtxExtent(m, n);
        return new DirectPtxKernelBlueprint(
            Operation: "softmax-cross-entropy-index",
            Version: 1,
            Architecture: architecture,
            Variant: $"fp32-m{m}-n{n}",
            Tensors:
            [
                new("logits", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D,
                    logits, logits, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("targets", DirectPtxPhysicalType.Int32, DirectPtxPhysicalLayout.Vector,
                    targets, targets, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("loss", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    loss, loss, 16, DirectPtxTensorAccess.Write, DirectPtxExtentMode.Exact),
                new("dLogits", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D,
                    dLogits, dLogits, 16, DirectPtxTensorAccess.Write, DirectPtxExtentMode.Exact)
            ],
            ResourceBudget: new DirectPtxResourceBudget(
                MaxRegistersPerThread: 40,
                MaxStaticSharedBytes: (n + BlockThreads) * sizeof(float),
                MaxLocalBytesPerThread: 0,
                MinBlocksPerMultiprocessor: 1),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["formula"] = "loss[m]=logsumexp(logits[m])-logits[m,target[m]]; dLogits=softmax-onehot(target)",
                ["logits-source"] = "hidden@W+bias via the GemmBias tile",
                ["stability"] = "row-max-subtracted-softmax",
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
                "Softmax-cross-entropy supports M in {64,128,256,512,1024,2048}, N in {256,512,1024,2048,4096}.");
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
