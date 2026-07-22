#if NET5_0_OR_GREATER
using System;
using System.Collections.Generic;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

internal enum DirectPtxCapsuleRoutingOperation
{
    WeightedSum,
    Agreement
}

/// <summary>
/// Exact dynamic-routing reductions for a canonical capsule configuration.
/// Each output scalar retains its full reduction in one register and commits
/// exactly once; no routing intermediate is materialized.
/// </summary>
internal sealed class PtxCapsuleRoutingF32Kernel : IDisposable
{
    internal const int Batch = 32;
    internal const int InputCapsules = 32;
    internal const int OutputCapsules = 10;
    internal const int Dimension = 16;
    internal const int WeightedOutputs = Batch * OutputCapsules * Dimension;
    internal const int AgreementOutputs = Batch * InputCapsules * OutputCapsules;
    internal const int BlockThreads = 256;

    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

    internal DirectPtxCapsuleRoutingOperation Operation { get; }
    internal string EntryPoint { get; }
    internal string Ptx { get; }
    internal DirectPtxFunctionInfo FunctionInfo { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal PtxCapsuleRoutingF32Kernel(
        DirectPtxRuntime runtime,
        DirectPtxCapsuleRoutingOperation operation)
    {
        ArgumentNullException.ThrowIfNull(runtime);
        if (!DirectPtxArchitecture.HasValidatedSparseGraph(
            runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new NotSupportedException(
                $"Capsule routing has no SM {runtime.ComputeCapabilityMajor}.{runtime.ComputeCapabilityMinor} experimental specialization.");
        Operation = operation;
        EntryPoint = GetEntryPoint(operation);
        Blueprint = CreateBlueprint(runtime.ArchitectureFamily, operation);
        Ptx = EmitPtx(runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor, operation);
        _module = runtime.LoadModule(Ptx);
        _function = _module.GetFunction(EntryPoint, out DirectPtxFunctionInfo functionInfo);
        FunctionInfo = functionInfo;
        int activeBlocks = _module.GetActiveBlocksPerMultiprocessor(_function, BlockThreads);
        Blueprint.ResourceBudget.Validate(EntryPoint, functionInfo, BlockThreads, activeBlocks);
        Audit = DirectPtxKernelAudit.Create(
            Blueprint, runtime.DeviceFingerprint, Ptx, functionInfo,
            BlockThreads, activeBlocks, _module.JitInfoLog);
    }

    internal static bool SupportsShape(
        int batch,
        int inputCapsules,
        int outputCapsules,
        int dimension) =>
        batch == Batch && inputCapsules == InputCapsules &&
        outputCapsules == OutputCapsules && dimension == Dimension;

    internal unsafe void Launch(
        DirectPtxTensorView first,
        DirectPtxTensorView second,
        DirectPtxTensorView output)
    {
        Require(first, Blueprint.Tensors[0], nameof(first));
        Require(second, Blueprint.Tensors[1], nameof(second));
        Require(output, Blueprint.Tensors[2], nameof(output));
        if (Overlaps(output, first) || Overlaps(output, second))
            throw new ArgumentException("Capsule routing output must be disjoint from both inputs.", nameof(output));
        IntPtr p0 = first.Pointer;
        IntPtr p1 = second.Pointer;
        IntPtr p2 = output.Pointer;
        void** arguments = stackalloc void*[3];
        arguments[0] = &p0;
        arguments[1] = &p1;
        arguments[2] = &p2;
        int outputs = Operation == DirectPtxCapsuleRoutingOperation.WeightedSum
            ? WeightedOutputs : AgreementOutputs;
        _module.Launch(_function, (uint)(outputs / BlockThreads), 1, 1,
            BlockThreads, 1, 1, 0, arguments);
    }

    internal static DirectPtxKernelBlueprint CreateBlueprint(
        DirectPtxArchitectureFamily architecture,
        DirectPtxCapsuleRoutingOperation operation)
    {
        var coupling = new DirectPtxExtent(Batch, InputCapsules, OutputCapsules);
        var predictions = new DirectPtxExtent(Batch, InputCapsules, OutputCapsules, Dimension);
        var routed = new DirectPtxExtent(Batch, OutputCapsules, Dimension);
        IReadOnlyList<DirectPtxTensorContract> tensors = operation == DirectPtxCapsuleRoutingOperation.WeightedSum
            ?
            [
                Exact("coupling", DirectPtxPhysicalLayout.CapsuleCoupling, coupling, DirectPtxTensorAccess.Read),
                Exact("predictions", DirectPtxPhysicalLayout.CapsulePredictions, predictions, DirectPtxTensorAccess.Read),
                Exact("output", DirectPtxPhysicalLayout.CapsuleOutput, routed, DirectPtxTensorAccess.Write)
            ]
            :
            [
                Exact("predictions", DirectPtxPhysicalLayout.CapsulePredictions, predictions, DirectPtxTensorAccess.Read),
                Exact("output", DirectPtxPhysicalLayout.CapsuleOutput, routed, DirectPtxTensorAccess.Read),
                Exact("agreement", DirectPtxPhysicalLayout.CapsuleCoupling, coupling, DirectPtxTensorAccess.Write)
            ];
        return new DirectPtxKernelBlueprint(
            Operation: operation == DirectPtxCapsuleRoutingOperation.WeightedSum
                ? "capsule-weighted-sum-f32" : "capsule-agreement-f32",
            Version: 1,
            Architecture: architecture,
            Variant: "b32-i32-o10-d16",
            Tensors: tensors,
            ResourceBudget: new DirectPtxResourceBudget(32, 0, 0, 6),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["reduction-order"] = operation == DirectPtxCapsuleRoutingOperation.WeightedSum
                    ? "ascending-input-capsule" : "ascending-dimension",
                ["output-write"] = "single-overwrite",
                ["workspace-bytes"] = "0",
                ["intermediate-global-bytes"] = "0"
            });
    }

    internal static string EmitPtx(
        int ccMajor,
        int ccMinor,
        DirectPtxCapsuleRoutingOperation operation)
    {
        if (ccMajor <= 0 || ccMinor < 0) throw new ArgumentOutOfRangeException(nameof(ccMajor));
        bool weighted = operation == DirectPtxCapsuleRoutingOperation.WeightedSum;
        var ptx = new StringBuilder(4096);
        ptx.AppendLine(".version 7.1");
        ptx.AppendLine($".target sm_{ccMajor}{ccMinor}");
        ptx.AppendLine(".address_size 64");
        ptx.AppendLine();
        ptx.AppendLine($".visible .entry {GetEntryPoint(operation)}(");
        ptx.AppendLine($"    .param .u64 {(weighted ? "coupling_ptr" : "predictions_ptr")},");
        ptx.AppendLine($"    .param .u64 {(weighted ? "predictions_ptr" : "output_ptr")},");
        ptx.AppendLine($"    .param .u64 {(weighted ? "output_ptr" : "agreement_ptr")}");
        ptx.AppendLine(")");
        ptx.AppendLine("{");
        ptx.AppendLine("    .reg .pred %p<3>;");
        ptx.AppendLine("    .reg .b32 %r<20>;");
        ptx.AppendLine("    .reg .b64 %rd<20>;");
        ptx.AppendLine("    .reg .f32 %f<5>;");
        if (weighted) EmitWeighted(ptx); else EmitAgreement(ptx);
        ptx.AppendLine("    ret;");
        ptx.AppendLine("}");
        return ptx.ToString();
    }

    private static void EmitWeighted(StringBuilder ptx)
    {
        ptx.AppendLine("    ld.param.u64 %rd0, [coupling_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd1, [predictions_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd2, [output_ptr];");
        EmitIndex(ptx);
        ptx.AppendLine("    and.b32 %r3, %r2, 15;");
        ptx.AppendLine("    shr.u32 %r4, %r2, 4;");
        ptx.AppendLine("    rem.u32 %r5, %r4, 10;");
        ptx.AppendLine("    div.u32 %r6, %r4, 10;");
        ptx.AppendLine("    mov.u32 %r7, 0;");
        ptx.AppendLine("    mov.f32 %f0, 0f00000000;");
        ptx.AppendLine("CAPSULE_WEIGHT_LOOP:");
        ptx.AppendLine("    setp.ge.u32 %p0, %r7, 32;");
        ptx.AppendLine("    @%p0 bra CAPSULE_WEIGHT_DONE;");
        ptx.AppendLine("    mul.lo.u32 %r8, %r6, 320;");
        ptx.AppendLine("    mad.lo.u32 %r8, %r7, 10, %r8;");
        ptx.AppendLine("    add.u32 %r8, %r8, %r5;");
        ptx.AppendLine("    mul.wide.u32 %rd3, %r8, 4;");
        ptx.AppendLine("    add.u64 %rd4, %rd0, %rd3;");
        ptx.AppendLine("    ld.global.f32 %f1, [%rd4];");
        ptx.AppendLine("    shl.b32 %r9, %r8, 4;");
        ptx.AppendLine("    add.u32 %r9, %r9, %r3;");
        ptx.AppendLine("    mul.wide.u32 %rd5, %r9, 4;");
        ptx.AppendLine("    add.u64 %rd6, %rd1, %rd5;");
        ptx.AppendLine("    ld.global.f32 %f2, [%rd6];");
        ptx.AppendLine("    fma.rn.f32 %f0, %f1, %f2, %f0;");
        ptx.AppendLine("    add.u32 %r7, %r7, 1;");
        ptx.AppendLine("    bra CAPSULE_WEIGHT_LOOP;");
        ptx.AppendLine("CAPSULE_WEIGHT_DONE:");
        EmitStore(ptx);
    }

    private static void EmitAgreement(StringBuilder ptx)
    {
        ptx.AppendLine("    ld.param.u64 %rd0, [predictions_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd1, [output_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd2, [agreement_ptr];");
        EmitIndex(ptx);
        ptx.AppendLine("    rem.u32 %r3, %r2, 10;");
        ptx.AppendLine("    div.u32 %r4, %r2, 10;");
        ptx.AppendLine("    rem.u32 %r5, %r4, 32;");
        ptx.AppendLine("    div.u32 %r6, %r4, 32;");
        ptx.AppendLine("    mov.u32 %r7, 0;");
        ptx.AppendLine("    mov.f32 %f0, 0f00000000;");
        ptx.AppendLine("CAPSULE_AGREEMENT_LOOP:");
        ptx.AppendLine("    setp.ge.u32 %p0, %r7, 16;");
        ptx.AppendLine("    @%p0 bra CAPSULE_AGREEMENT_DONE;");
        ptx.AppendLine("    mul.lo.u32 %r8, %r6, 5120;");
        ptx.AppendLine("    mad.lo.u32 %r8, %r5, 160, %r8;");
        ptx.AppendLine("    mad.lo.u32 %r8, %r3, 16, %r8;");
        ptx.AppendLine("    add.u32 %r8, %r8, %r7;");
        ptx.AppendLine("    mul.wide.u32 %rd3, %r8, 4;");
        ptx.AppendLine("    add.u64 %rd4, %rd0, %rd3;");
        ptx.AppendLine("    ld.global.f32 %f1, [%rd4];");
        ptx.AppendLine("    mul.lo.u32 %r9, %r6, 160;");
        ptx.AppendLine("    mad.lo.u32 %r9, %r3, 16, %r9;");
        ptx.AppendLine("    add.u32 %r9, %r9, %r7;");
        ptx.AppendLine("    mul.wide.u32 %rd5, %r9, 4;");
        ptx.AppendLine("    add.u64 %rd6, %rd1, %rd5;");
        ptx.AppendLine("    ld.global.f32 %f2, [%rd6];");
        ptx.AppendLine("    fma.rn.f32 %f0, %f1, %f2, %f0;");
        ptx.AppendLine("    add.u32 %r7, %r7, 1;");
        ptx.AppendLine("    bra CAPSULE_AGREEMENT_LOOP;");
        ptx.AppendLine("CAPSULE_AGREEMENT_DONE:");
        EmitStore(ptx);
    }

    private static void EmitIndex(StringBuilder ptx)
    {
        ptx.AppendLine("    mov.u32 %r0, %tid.x;");
        ptx.AppendLine("    mov.u32 %r1, %ctaid.x;");
        ptx.AppendLine($"    mad.lo.u32 %r2, %r1, {BlockThreads}, %r0;");
    }

    private static void EmitStore(StringBuilder ptx)
    {
        ptx.AppendLine("    mul.wide.u32 %rd7, %r2, 4;");
        ptx.AppendLine("    add.u64 %rd8, %rd2, %rd7;");
        ptx.AppendLine("    st.global.f32 [%rd8], %f0;");
    }

    private static string GetEntryPoint(DirectPtxCapsuleRoutingOperation operation) =>
        operation == DirectPtxCapsuleRoutingOperation.WeightedSum
            ? "aidotnet_capsule_weighted_sum_f32_b32_i32_o10_d16"
            : "aidotnet_capsule_agreement_f32_b32_i32_o10_d16";

    private static DirectPtxTensorContract Exact(
        string name,
        DirectPtxPhysicalLayout layout,
        DirectPtxExtent extent,
        DirectPtxTensorAccess access) =>
        new(name, DirectPtxPhysicalType.Float32, layout, extent, extent, 16, access,
            DirectPtxExtentMode.Exact);

    private static void Require(DirectPtxTensorView view, DirectPtxTensorContract contract, string parameter)
    {
        if (view.Pointer == IntPtr.Zero || view.PhysicalType != contract.PhysicalType ||
            view.Layout != contract.Layout || view.LogicalExtent != contract.LogicalExtent ||
            view.PhysicalExtent != contract.PhysicalExtent || view.ByteLength != contract.RequiredBytes ||
            view.AllocationByteLength != contract.RequiredBytes)
            throw new ArgumentException($"{parameter} does not satisfy exact physical ABI '{contract.Name}'.", parameter);
    }

    private static bool Overlaps(DirectPtxTensorView left, DirectPtxTensorView right)
    {
        nuint leftStart = (nuint)left.Pointer;
        nuint rightStart = (nuint)right.Pointer;
        nuint leftEnd = checked(leftStart + left.ByteLength);
        nuint rightEnd = checked(rightStart + right.ByteLength);
        return leftStart < rightEnd && rightStart < leftEnd;
    }

    public void Dispose() => _module.Dispose();
}
#endif
