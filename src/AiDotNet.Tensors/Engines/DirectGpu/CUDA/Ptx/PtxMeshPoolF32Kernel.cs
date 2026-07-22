#if NET5_0_OR_GREATER
using System;
using System.Collections.Generic;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

internal enum DirectPtxMeshPoolOperation
{
    ComputeScores,
    Gather,
    BackwardAtomic,
    BackwardDeterministic,
    ImportanceBackward,
    ZeroGrad,
    SoftmaxFindMax,
    SoftmaxFinalMax,
    SoftmaxExpSum,
    SoftmaxFinalSum,
    SoftmaxNormalize,
    SoftmaxScores,
    WeightedGather,
    WeightedBackwardAtomic,
    WeightedBackwardDeterministic,
    ScoresBackward
}

/// <summary>
/// Exact direct-PTX MeshPool family. Shape, temperature, layout, and strides
/// are baked into one separately auditable module per operation; the launch
/// ABI contains device pointers only.
/// </summary>
internal sealed class PtxMeshPoolF32Kernel : IDisposable
{
    internal const int NumEdges = 16384;
    internal const int NumKept = 4096;
    internal const int Channels = 64;
    internal const int EdgeElements = NumEdges * Channels;
    internal const int KeptElements = NumKept * Channels;
    internal const int Partials = NumEdges / 256;
    internal const int LegacySoftmaxEdges = 256;
    internal const int BlockThreads = 256;

    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

    internal DirectPtxMeshPoolOperation Operation { get; }
    internal string EntryPoint { get; }
    internal string Ptx { get; }
    internal DirectPtxFunctionInfo FunctionInfo { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal PtxMeshPoolF32Kernel(DirectPtxRuntime runtime, DirectPtxMeshPoolOperation operation)
    {
        ArgumentNullException.ThrowIfNull(runtime);
        if (!DirectPtxArchitecture.HasValidatedSparseGraph(
            runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new NotSupportedException(
                $"MeshPool has no SM {runtime.ComputeCapabilityMajor}.{runtime.ComputeCapabilityMinor} experimental specialization.");
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

    internal static bool SupportsShape(DirectPtxMeshPoolOperation operation,
        int numEdges, int numKept, int channels, float temperature = 1f) =>
        numEdges == (operation == DirectPtxMeshPoolOperation.SoftmaxScores ? LegacySoftmaxEdges : NumEdges) &&
        numKept == NumKept && channels == Channels &&
        BitConverter.SingleToInt32Bits(temperature) == BitConverter.SingleToInt32Bits(1f);

    internal static int GetTensorCount(DirectPtxMeshPoolOperation operation) => operation switch
    {
        DirectPtxMeshPoolOperation.ZeroGrad => 1,
        DirectPtxMeshPoolOperation.SoftmaxFindMax or DirectPtxMeshPoolOperation.SoftmaxFinalMax or
        DirectPtxMeshPoolOperation.SoftmaxFinalSum or DirectPtxMeshPoolOperation.SoftmaxScores => 2,
        DirectPtxMeshPoolOperation.ComputeScores or DirectPtxMeshPoolOperation.Gather or
        DirectPtxMeshPoolOperation.BackwardAtomic or DirectPtxMeshPoolOperation.BackwardDeterministic or
        DirectPtxMeshPoolOperation.SoftmaxNormalize => 3,
        _ => 4
    };

    internal static long GetRequiredBytes(DirectPtxMeshPoolOperation operation, int tensorIndex)
    {
        if ((uint)tensorIndex >= (uint)GetTensorCount(operation))
            throw new ArgumentOutOfRangeException(nameof(tensorIndex));
        int elements = operation switch
        {
            DirectPtxMeshPoolOperation.ComputeScores => tensorIndex switch { 0 => EdgeElements, 1 => Channels, _ => NumEdges },
            DirectPtxMeshPoolOperation.Gather => tensorIndex switch { 0 => EdgeElements, 1 => NumKept, _ => KeptElements },
            DirectPtxMeshPoolOperation.BackwardAtomic or DirectPtxMeshPoolOperation.BackwardDeterministic =>
                tensorIndex switch { 0 => KeptElements, 1 => NumKept, _ => EdgeElements },
            DirectPtxMeshPoolOperation.ImportanceBackward => tensorIndex switch
                { 0 => KeptElements, 1 => EdgeElements, 2 => NumKept, _ => Channels },
            DirectPtxMeshPoolOperation.ZeroGrad => EdgeElements,
            DirectPtxMeshPoolOperation.SoftmaxFindMax => tensorIndex == 0 ? NumEdges : Partials,
            DirectPtxMeshPoolOperation.SoftmaxFinalMax or DirectPtxMeshPoolOperation.SoftmaxFinalSum =>
                tensorIndex == 0 ? Partials : 1,
            DirectPtxMeshPoolOperation.SoftmaxExpSum => tensorIndex switch
                { 0 => NumEdges, 1 => 1, 2 => NumEdges, _ => Partials },
            DirectPtxMeshPoolOperation.SoftmaxNormalize => tensorIndex == 1 ? 1 : NumEdges,
            DirectPtxMeshPoolOperation.SoftmaxScores => LegacySoftmaxEdges,
            DirectPtxMeshPoolOperation.WeightedGather => tensorIndex switch
                { 0 => EdgeElements, 1 => NumEdges, 2 => NumKept, _ => KeptElements },
            DirectPtxMeshPoolOperation.WeightedBackwardAtomic or DirectPtxMeshPoolOperation.WeightedBackwardDeterministic =>
                tensorIndex switch { 0 => KeptElements, 1 => NumEdges, 2 => NumKept, _ => EdgeElements },
            DirectPtxMeshPoolOperation.ScoresBackward => tensorIndex switch
                { 0 => KeptElements, 1 => EdgeElements, 2 => NumKept, _ => NumEdges },
            _ => throw new ArgumentOutOfRangeException(nameof(operation))
        };
        return checked((long)elements * sizeof(float));
    }

    internal unsafe void Launch(ReadOnlySpan<DirectPtxTensorView> tensors)
    {
        if (tensors.Length != Blueprint.Tensors.Count)
            throw new ArgumentException("The MeshPool pointer ABI has the wrong arity.", nameof(tensors));
        for (int i = 0; i < tensors.Length; i++) Require(tensors[i], Blueprint.Tensors[i], $"tensor{i}");
        for (int i = 0; i < tensors.Length; i++)
        {
            if ((Blueprint.Tensors[i].Access & DirectPtxTensorAccess.Write) == 0) continue;
            for (int j = 0; j < tensors.Length; j++)
            {
                if (i != j && Overlaps(tensors[i], tensors[j]))
                    throw new ArgumentException("MeshPool writable tensors must be disjoint from every other ABI tensor.");
            }
        }

        IntPtr* pointers = stackalloc IntPtr[tensors.Length];
        void** arguments = stackalloc void*[tensors.Length];
        for (int i = 0; i < tensors.Length; i++)
        {
            pointers[i] = tensors[i].Pointer;
            arguments[i] = &pointers[i];
        }
        int outputs = Operation switch
        {
            DirectPtxMeshPoolOperation.ComputeScores or DirectPtxMeshPoolOperation.SoftmaxNormalize or
            DirectPtxMeshPoolOperation.ScoresBackward => NumEdges,
            DirectPtxMeshPoolOperation.SoftmaxScores => LegacySoftmaxEdges,
            DirectPtxMeshPoolOperation.Gather or DirectPtxMeshPoolOperation.BackwardAtomic or
            DirectPtxMeshPoolOperation.WeightedGather or DirectPtxMeshPoolOperation.WeightedBackwardAtomic => KeptElements,
            DirectPtxMeshPoolOperation.BackwardDeterministic or DirectPtxMeshPoolOperation.ZeroGrad or
            DirectPtxMeshPoolOperation.WeightedBackwardDeterministic => EdgeElements,
            DirectPtxMeshPoolOperation.ImportanceBackward => Channels,
            DirectPtxMeshPoolOperation.SoftmaxFindMax or DirectPtxMeshPoolOperation.SoftmaxExpSum => Partials,
            _ => 1
        };
        uint grid = checked((uint)((outputs + BlockThreads - 1) / BlockThreads));
        _module.Launch(_function, grid, 1, 1, BlockThreads, 1, 1, 0, arguments);
    }

    internal static DirectPtxKernelBlueprint CreateBlueprint(
        DirectPtxArchitectureFamily architecture,
        DirectPtxMeshPoolOperation operation)
    {
        var edges2d = new DirectPtxExtent(NumEdges, Channels);
        var kept2d = new DirectPtxExtent(NumKept, Channels);
        var edges = new DirectPtxExtent(NumEdges);
        var kept = new DirectPtxExtent(NumKept);
        var channels = new DirectPtxExtent(Channels);
        var partials = new DirectPtxExtent(Partials);
        var scalar = new DirectPtxExtent(1);
        var legacyEdges = new DirectPtxExtent(LegacySoftmaxEdges);
        DirectPtxTensorContract F(string name, DirectPtxPhysicalLayout layout, DirectPtxExtent extent,
            DirectPtxTensorAccess access) => Exact(name, DirectPtxPhysicalType.Float32, layout, extent, access);
        DirectPtxTensorContract I(string name, DirectPtxExtent extent) =>
            Exact(name, DirectPtxPhysicalType.Int32, DirectPtxPhysicalLayout.Vector, extent, DirectPtxTensorAccess.Read);

        IReadOnlyList<DirectPtxTensorContract> tensors = operation switch
        {
            DirectPtxMeshPoolOperation.ComputeScores =>
                [F("input", DirectPtxPhysicalLayout.EdgeMajor2D, edges2d, DirectPtxTensorAccess.Read),
                 F("importance-weights", DirectPtxPhysicalLayout.Vector, channels, DirectPtxTensorAccess.Read),
                 F("scores", DirectPtxPhysicalLayout.Vector, edges, DirectPtxTensorAccess.Write)],
            DirectPtxMeshPoolOperation.Gather =>
                [F("input", DirectPtxPhysicalLayout.EdgeMajor2D, edges2d, DirectPtxTensorAccess.Read), I("kept-indices", kept),
                 F("output", DirectPtxPhysicalLayout.EdgeMajor2D, kept2d, DirectPtxTensorAccess.Write)],
            DirectPtxMeshPoolOperation.BackwardAtomic or DirectPtxMeshPoolOperation.BackwardDeterministic =>
                [F("grad-output", DirectPtxPhysicalLayout.EdgeMajor2D, kept2d, DirectPtxTensorAccess.Read), I("kept-indices", kept),
                 F("grad-input", DirectPtxPhysicalLayout.EdgeMajor2D, edges2d, DirectPtxTensorAccess.ReadWrite)],
            DirectPtxMeshPoolOperation.ImportanceBackward =>
                [F("grad-output", DirectPtxPhysicalLayout.EdgeMajor2D, kept2d, DirectPtxTensorAccess.Read),
                 F("input", DirectPtxPhysicalLayout.EdgeMajor2D, edges2d, DirectPtxTensorAccess.Read), I("kept-indices", kept),
                 F("grad-importance-weights", DirectPtxPhysicalLayout.Vector, channels, DirectPtxTensorAccess.Write)],
            DirectPtxMeshPoolOperation.ZeroGrad =>
                [F("grad-input", DirectPtxPhysicalLayout.EdgeMajor2D, edges2d, DirectPtxTensorAccess.Write)],
            DirectPtxMeshPoolOperation.SoftmaxFindMax =>
                [F("scores", DirectPtxPhysicalLayout.Vector, edges, DirectPtxTensorAccess.Read),
                 F("partial-max", DirectPtxPhysicalLayout.Vector, partials, DirectPtxTensorAccess.Write)],
            DirectPtxMeshPoolOperation.SoftmaxFinalMax =>
                [F("partial-max", DirectPtxPhysicalLayout.Vector, partials, DirectPtxTensorAccess.Read),
                 F("global-max", DirectPtxPhysicalLayout.Vector, scalar, DirectPtxTensorAccess.Write)],
            DirectPtxMeshPoolOperation.SoftmaxExpSum =>
                [F("scores", DirectPtxPhysicalLayout.Vector, edges, DirectPtxTensorAccess.Read),
                 F("global-max", DirectPtxPhysicalLayout.Vector, scalar, DirectPtxTensorAccess.Read),
                 F("exp-values", DirectPtxPhysicalLayout.Vector, edges, DirectPtxTensorAccess.Write),
                 F("partial-sum", DirectPtxPhysicalLayout.Vector, partials, DirectPtxTensorAccess.Write)],
            DirectPtxMeshPoolOperation.SoftmaxFinalSum =>
                [F("partial-sum", DirectPtxPhysicalLayout.Vector, partials, DirectPtxTensorAccess.Read),
                 F("global-sum", DirectPtxPhysicalLayout.Vector, scalar, DirectPtxTensorAccess.Write)],
            DirectPtxMeshPoolOperation.SoftmaxNormalize =>
                [F("exp-values", DirectPtxPhysicalLayout.Vector, edges, DirectPtxTensorAccess.Read),
                 F("global-sum", DirectPtxPhysicalLayout.Vector, scalar, DirectPtxTensorAccess.Read),
                 F("softmax-scores", DirectPtxPhysicalLayout.Vector, edges, DirectPtxTensorAccess.Write)],
            DirectPtxMeshPoolOperation.SoftmaxScores =>
                [F("scores", DirectPtxPhysicalLayout.Vector, legacyEdges, DirectPtxTensorAccess.Read),
                 F("softmax-scores", DirectPtxPhysicalLayout.Vector, legacyEdges, DirectPtxTensorAccess.Write)],
            DirectPtxMeshPoolOperation.WeightedGather =>
                [F("input", DirectPtxPhysicalLayout.EdgeMajor2D, edges2d, DirectPtxTensorAccess.Read),
                 F("scores", DirectPtxPhysicalLayout.Vector, edges, DirectPtxTensorAccess.Read), I("kept-indices", kept),
                 F("output", DirectPtxPhysicalLayout.EdgeMajor2D, kept2d, DirectPtxTensorAccess.Write)],
            DirectPtxMeshPoolOperation.WeightedBackwardAtomic or DirectPtxMeshPoolOperation.WeightedBackwardDeterministic =>
                [F("grad-output", DirectPtxPhysicalLayout.EdgeMajor2D, kept2d, DirectPtxTensorAccess.Read),
                 F("scores", DirectPtxPhysicalLayout.Vector, edges, DirectPtxTensorAccess.Read), I("kept-indices", kept),
                 F("grad-input", DirectPtxPhysicalLayout.EdgeMajor2D, edges2d, DirectPtxTensorAccess.ReadWrite)],
            DirectPtxMeshPoolOperation.ScoresBackward =>
                [F("grad-output", DirectPtxPhysicalLayout.EdgeMajor2D, kept2d, DirectPtxTensorAccess.Read),
                 F("input", DirectPtxPhysicalLayout.EdgeMajor2D, edges2d, DirectPtxTensorAccess.Read), I("kept-indices", kept),
                 F("grad-scores", DirectPtxPhysicalLayout.Vector, edges, DirectPtxTensorAccess.Write)],
            _ => throw new ArgumentOutOfRangeException(nameof(operation))
        };

        bool atomic = operation is DirectPtxMeshPoolOperation.BackwardAtomic or
            DirectPtxMeshPoolOperation.WeightedBackwardAtomic;
        return new DirectPtxKernelBlueprint(
            Operation: GetKernelName(operation), Version: 1, Architecture: architecture,
            Variant: operation == DirectPtxMeshPoolOperation.SoftmaxScores
                ? "edges256-temp1-single-block" : "edges16384-kept4096-c64-temp1", Tensors: tensors,
            ResourceBudget: new DirectPtxResourceBudget(40,
                operation == DirectPtxMeshPoolOperation.SoftmaxScores ? 1024 : 0, 0, 4),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["shape"] = operation == DirectPtxMeshPoolOperation.SoftmaxScores
                    ? "edges=256" : "edges=16384,kept=4096,channels=64",
                ["temperature"] = "1.0f-baked",
                ["reduction-order"] = atomic ? "atomic-nondeterministic" : "ascending-index",
                ["invalid-kept-index"] = "zero-or-ignore-matching-cuda-surface",
                ["parameter-abi"] = "device-pointers-only",
                ["promotion"] = "experimental-hardware-evidence-pending",
                ["workspace-bytes"] = "0",
                ["intermediate-global-bytes"] = operation is DirectPtxMeshPoolOperation.SoftmaxFindMax or
                    DirectPtxMeshPoolOperation.SoftmaxFinalMax or DirectPtxMeshPoolOperation.SoftmaxExpSum or
                    DirectPtxMeshPoolOperation.SoftmaxFinalSum or DirectPtxMeshPoolOperation.SoftmaxNormalize
                    ? "explicit-caller-owned-softmax-pipeline" : "0"
            });
    }

    internal static string EmitPtx(int ccMajor, int ccMinor, DirectPtxMeshPoolOperation operation)
    {
        if (ccMajor <= 0 || ccMinor < 0) throw new ArgumentOutOfRangeException(nameof(ccMajor));
        var ptx = new StringBuilder(32768);
        ptx.AppendLine(".version 7.1");
        ptx.AppendLine($".target sm_{ccMajor}{ccMinor}");
        ptx.AppendLine(".address_size 64");
        ptx.AppendLine();
        switch (operation)
        {
            case DirectPtxMeshPoolOperation.ComputeScores: EmitComputeScores(ptx); break;
            case DirectPtxMeshPoolOperation.Gather: EmitGather(ptx, weighted: false); break;
            case DirectPtxMeshPoolOperation.BackwardAtomic: EmitBackward(ptx, weighted: false, deterministic: false); break;
            case DirectPtxMeshPoolOperation.BackwardDeterministic: EmitBackward(ptx, weighted: false, deterministic: true); break;
            case DirectPtxMeshPoolOperation.ImportanceBackward: EmitImportanceBackward(ptx); break;
            case DirectPtxMeshPoolOperation.ZeroGrad: EmitZero(ptx); break;
            case DirectPtxMeshPoolOperation.SoftmaxFindMax: EmitFindMax(ptx); break;
            case DirectPtxMeshPoolOperation.SoftmaxFinalMax: EmitFinalReduction(ptx, max: true); break;
            case DirectPtxMeshPoolOperation.SoftmaxExpSum: EmitExpSum(ptx); break;
            case DirectPtxMeshPoolOperation.SoftmaxFinalSum: EmitFinalReduction(ptx, max: false); break;
            case DirectPtxMeshPoolOperation.SoftmaxNormalize: EmitNormalize(ptx); break;
            case DirectPtxMeshPoolOperation.SoftmaxScores: EmitSoftmaxScores(ptx); break;
            case DirectPtxMeshPoolOperation.WeightedGather: EmitGather(ptx, weighted: true); break;
            case DirectPtxMeshPoolOperation.WeightedBackwardAtomic: EmitBackward(ptx, weighted: true, deterministic: false); break;
            case DirectPtxMeshPoolOperation.WeightedBackwardDeterministic: EmitBackward(ptx, weighted: true, deterministic: true); break;
            case DirectPtxMeshPoolOperation.ScoresBackward: EmitScoresBackward(ptx); break;
            default: throw new ArgumentOutOfRangeException(nameof(operation));
        }
        return ptx.ToString();
    }

    private static void EmitComputeScores(StringBuilder ptx)
    {
        Header(ptx, DirectPtxMeshPoolOperation.ComputeScores, "input", "weights", "scores"); Registers(ptx);
        LoadPointers(ptx, 3); LinearIndex(ptx); ptx.AppendLine("    mov.f32 %f0, 0f00000000;");
        for (int c = 0; c < Channels; c++)
        {
            ptx.AppendLine($"    shl.b32 %r3, %r2, 6;");
            ptx.AppendLine($"    add.u32 %r3, %r3, {c};");
            Address(ptx, 3, 0, 3); ptx.AppendLine("    ld.global.f32 %f1, [%rd3];");
            ptx.AppendLine($"    mov.u32 %r4, {c};"); Address(ptx, 4, 1, 4);
            ptx.AppendLine("    ld.global.f32 %f2, [%rd4];"); ptx.AppendLine("    fma.rn.f32 %f0, %f1, %f2, %f0;");
        }
        Address(ptx, 2, 2, 5); ptx.AppendLine("    st.global.f32 [%rd5], %f0;"); End(ptx);
    }

    private static void EmitGather(StringBuilder ptx, bool weighted)
    {
        var op = weighted ? DirectPtxMeshPoolOperation.WeightedGather : DirectPtxMeshPoolOperation.Gather;
        Header(ptx, op, weighted ? ["input", "scores", "kept", "output"] : ["input", "kept", "output"]); Registers(ptx);
        LoadPointers(ptx, weighted ? 4 : 3); LinearIndex(ptx);
        ptx.AppendLine("    shr.u32 %r3, %r2, 6;"); ptx.AppendLine("    and.b32 %r4, %r2, 63;");
        Address(ptx, 3, weighted ? 2 : 1, 4); ptx.AppendLine("    ld.global.s32 %r5, [%rd4];");
        ptx.AppendLine("    setp.lt.s32 %p0, %r5, 0;"); ptx.AppendLine($"    setp.ge.s32 %p1, %r5, {NumEdges};");
        ptx.AppendLine("    or.pred %p2, %p0, %p1;"); ptx.AppendLine("    @%p2 bra GATHER_ZERO;");
        ptx.AppendLine("    shl.b32 %r6, %r5, 6;"); ptx.AppendLine("    add.u32 %r6, %r6, %r4;");
        Address(ptx, 6, 0, 5); ptx.AppendLine("    ld.global.f32 %f0, [%rd5];");
        if (weighted)
        {
            Address(ptx, 5, 1, 6); ptx.AppendLine("    ld.global.f32 %f1, [%rd6];");
            ptx.AppendLine("    mul.rn.f32 %f0, %f0, %f1;");
        }
        ptx.AppendLine("    bra GATHER_STORE;"); ptx.AppendLine("GATHER_ZERO:"); ptx.AppendLine("    mov.f32 %f0, 0f00000000;");
        ptx.AppendLine("GATHER_STORE:"); Address(ptx, 2, weighted ? 3 : 2, 7);
        ptx.AppendLine("    st.global.f32 [%rd7], %f0;"); End(ptx);
    }

    private static void EmitBackward(StringBuilder ptx, bool weighted, bool deterministic)
    {
        DirectPtxMeshPoolOperation op = (weighted, deterministic) switch
        {
            (false, false) => DirectPtxMeshPoolOperation.BackwardAtomic,
            (false, true) => DirectPtxMeshPoolOperation.BackwardDeterministic,
            (true, false) => DirectPtxMeshPoolOperation.WeightedBackwardAtomic,
            _ => DirectPtxMeshPoolOperation.WeightedBackwardDeterministic
        };
        Header(ptx, op, weighted ? ["grad_output", "scores", "kept", "grad_input"] :
            ["grad_output", "kept", "grad_input"]); Registers(ptx); LoadPointers(ptx, weighted ? 4 : 3); LinearIndex(ptx);
        if (!deterministic)
        {
            ptx.AppendLine("    shr.u32 %r3, %r2, 6;"); ptx.AppendLine("    and.b32 %r4, %r2, 63;");
            Address(ptx, 3, weighted ? 2 : 1, 4); ptx.AppendLine("    ld.global.s32 %r5, [%rd4];");
            ptx.AppendLine("    setp.lt.s32 %p0, %r5, 0;"); ptx.AppendLine($"    setp.ge.s32 %p1, %r5, {NumEdges};");
            ptx.AppendLine("    or.pred %p2, %p0, %p1;"); ptx.AppendLine("    @%p2 bra BACKWARD_EXIT;");
            Address(ptx, 2, 0, 5); ptx.AppendLine("    ld.global.f32 %f0, [%rd5];");
            if (weighted) { Address(ptx, 5, 1, 6); ptx.AppendLine("    ld.global.f32 %f1, [%rd6];"); ptx.AppendLine("    mul.rn.f32 %f0, %f0, %f1;"); }
            ptx.AppendLine("    shl.b32 %r6, %r5, 6;"); ptx.AppendLine("    add.u32 %r6, %r6, %r4;");
            Address(ptx, 6, weighted ? 3 : 2, 7); ptx.AppendLine("    atom.global.add.f32 %f2, [%rd7], %f0;");
            ptx.AppendLine("BACKWARD_EXIT:"); End(ptx); return;
        }
        ptx.AppendLine("    shr.u32 %r3, %r2, 6;"); ptx.AppendLine("    and.b32 %r4, %r2, 63;");
        ptx.AppendLine("    mov.u32 %r5, 0;"); ptx.AppendLine("    mov.f32 %f0, 0f00000000;");
        ptx.AppendLine("BACKWARD_SCAN:"); ptx.AppendLine($"    setp.ge.u32 %p0, %r5, {NumKept};");
        ptx.AppendLine("    @%p0 bra BACKWARD_DONE;"); Address(ptx, 5, weighted ? 2 : 1, 4);
        ptx.AppendLine("    ld.global.s32 %r6, [%rd4];"); ptx.AppendLine("    setp.ne.s32 %p1, %r6, %r3;");
        ptx.AppendLine("    @%p1 bra BACKWARD_NEXT;"); ptx.AppendLine("    shl.b32 %r7, %r5, 6;");
        ptx.AppendLine("    add.u32 %r7, %r7, %r4;"); Address(ptx, 7, 0, 5); ptx.AppendLine("    ld.global.f32 %f1, [%rd5];");
        if (weighted) { Address(ptx, 3, 1, 6); ptx.AppendLine("    ld.global.f32 %f2, [%rd6];"); ptx.AppendLine("    mul.rn.f32 %f1, %f1, %f2;"); }
        ptx.AppendLine("    add.rn.f32 %f0, %f0, %f1;"); ptx.AppendLine("BACKWARD_NEXT:");
        ptx.AppendLine("    add.u32 %r5, %r5, 1;"); ptx.AppendLine("    bra BACKWARD_SCAN;");
        ptx.AppendLine("BACKWARD_DONE:"); Address(ptx, 2, weighted ? 3 : 2, 7); ptx.AppendLine("    ld.global.f32 %f3, [%rd7];");
        ptx.AppendLine("    add.rn.f32 %f3, %f3, %f0;"); ptx.AppendLine("    st.global.f32 [%rd7], %f3;"); End(ptx);
    }

    private static void EmitImportanceBackward(StringBuilder ptx)
    {
        Header(ptx, DirectPtxMeshPoolOperation.ImportanceBackward, "grad_output", "input", "kept", "grad_weights");
        Registers(ptx); LoadPointers(ptx, 4); LinearIndex(ptx); ptx.AppendLine($"    setp.ge.u32 %p0, %r2, {Channels};");
        ptx.AppendLine("    @%p0 bra IMPORTANCE_EXIT;"); ptx.AppendLine("    mov.u32 %r3, 0;"); ptx.AppendLine("    mov.f32 %f0, 0f00000000;");
        ptx.AppendLine("IMPORTANCE_LOOP:"); ptx.AppendLine($"    setp.ge.u32 %p0, %r3, {NumKept};");
        ptx.AppendLine("    @%p0 bra IMPORTANCE_DONE;"); Address(ptx, 3, 2, 4); ptx.AppendLine("    ld.global.s32 %r4, [%rd4];");
        ptx.AppendLine("    setp.lt.s32 %p1, %r4, 0;"); ptx.AppendLine($"    setp.ge.s32 %p2, %r4, {NumEdges};");
        ptx.AppendLine("    or.pred %p3, %p1, %p2;"); ptx.AppendLine("    @%p3 bra IMPORTANCE_NEXT;");
        ptx.AppendLine("    mov.f32 %f1, 0f00000000;"); ptx.AppendLine("    shl.b32 %r5, %r3, 6;");
        for (int c = 0; c < Channels; c++)
        {
            ptx.AppendLine($"    add.u32 %r6, %r5, {c};"); Address(ptx, 6, 0, 5);
            ptx.AppendLine("    ld.global.f32 %f2, [%rd5];"); ptx.AppendLine("    add.rn.f32 %f1, %f1, %f2;");
        }
        ptx.AppendLine("    shl.b32 %r7, %r4, 6;"); ptx.AppendLine("    add.u32 %r7, %r7, %r2;"); Address(ptx, 7, 1, 6);
        ptx.AppendLine("    ld.global.f32 %f3, [%rd6];"); ptx.AppendLine("    fma.rn.f32 %f0, %f1, %f3, %f0;");
        ptx.AppendLine("IMPORTANCE_NEXT:"); ptx.AppendLine("    add.u32 %r3, %r3, 1;"); ptx.AppendLine("    bra IMPORTANCE_LOOP;");
        ptx.AppendLine("IMPORTANCE_DONE:"); Address(ptx, 2, 3, 7); ptx.AppendLine("    st.global.f32 [%rd7], %f0;");
        ptx.AppendLine("IMPORTANCE_EXIT:"); End(ptx);
    }

    private static void EmitZero(StringBuilder ptx)
    {
        Header(ptx, DirectPtxMeshPoolOperation.ZeroGrad, "grad_input"); Registers(ptx); LoadPointers(ptx, 1); LinearIndex(ptx);
        Address(ptx, 2, 0, 2); ptx.AppendLine("    st.global.u32 [%rd2], 0;"); End(ptx);
    }

    private static void EmitFindMax(StringBuilder ptx)
    {
        Header(ptx, DirectPtxMeshPoolOperation.SoftmaxFindMax, "scores", "partial_max"); Registers(ptx); LoadPointers(ptx, 2); LinearIndex(ptx);
        ptx.AppendLine($"    setp.ge.u32 %p0, %r2, {Partials};"); ptx.AppendLine("    @%p0 bra FIND_MAX_EXIT;");
        ptx.AppendLine("    shl.b32 %r3, %r2, 8;"); ptx.AppendLine("    mov.u32 %r4, 0;"); ptx.AppendLine("    mov.f32 %f0, 0ff800000;");
        ptx.AppendLine("FIND_MAX_LOOP:"); ptx.AppendLine("    setp.ge.u32 %p1, %r4, 256;"); ptx.AppendLine("    @%p1 bra FIND_MAX_DONE;");
        ptx.AppendLine("    add.u32 %r5, %r3, %r4;"); Address(ptx, 5, 0, 3); ptx.AppendLine("    ld.global.f32 %f1, [%rd3];");
        ptx.AppendLine("    max.f32 %f0, %f0, %f1;"); ptx.AppendLine("    add.u32 %r4, %r4, 1;"); ptx.AppendLine("    bra FIND_MAX_LOOP;");
        ptx.AppendLine("FIND_MAX_DONE:"); Address(ptx, 2, 1, 4); ptx.AppendLine("    st.global.f32 [%rd4], %f0;");
        ptx.AppendLine("FIND_MAX_EXIT:"); End(ptx);
    }

    private static void EmitFinalReduction(StringBuilder ptx, bool max)
    {
        var op = max ? DirectPtxMeshPoolOperation.SoftmaxFinalMax : DirectPtxMeshPoolOperation.SoftmaxFinalSum;
        Header(ptx, op, max ? ["partial_max", "global_max"] : ["partial_sum", "global_sum"]); Registers(ptx); LoadPointers(ptx, 2);
        ptx.AppendLine("    mov.u32 %r0, %tid.x;"); ptx.AppendLine("    setp.ne.u32 %p0, %r0, 0;"); ptx.AppendLine("    @%p0 bra FINAL_EXIT;");
        ptx.AppendLine("    mov.u32 %r2, 0;"); ptx.AppendLine(max ? "    mov.f32 %f0, 0ff800000;" : "    mov.f32 %f0, 0f00000000;");
        ptx.AppendLine("FINAL_LOOP:"); ptx.AppendLine($"    setp.ge.u32 %p1, %r2, {Partials};"); ptx.AppendLine("    @%p1 bra FINAL_DONE;");
        Address(ptx, 2, 0, 2); ptx.AppendLine("    ld.global.f32 %f1, [%rd2];");
        ptx.AppendLine(max ? "    max.f32 %f0, %f0, %f1;" : "    add.rn.f32 %f0, %f0, %f1;");
        ptx.AppendLine("    add.u32 %r2, %r2, 1;"); ptx.AppendLine("    bra FINAL_LOOP;");
        ptx.AppendLine("FINAL_DONE:"); ptx.AppendLine("    st.global.f32 [%rd1], %f0;"); ptx.AppendLine("FINAL_EXIT:"); End(ptx);
    }

    private static void EmitExpSum(StringBuilder ptx)
    {
        Header(ptx, DirectPtxMeshPoolOperation.SoftmaxExpSum, "scores", "global_max", "exp_values", "partial_sum");
        Registers(ptx); LoadPointers(ptx, 4); LinearIndex(ptx); ptx.AppendLine($"    setp.ge.u32 %p0, %r2, {Partials};");
        ptx.AppendLine("    @%p0 bra EXP_SUM_EXIT;"); ptx.AppendLine("    ld.global.f32 %f0, [%rd1];");
        ptx.AppendLine("    shl.b32 %r3, %r2, 8;"); ptx.AppendLine("    mov.u32 %r4, 0;"); ptx.AppendLine("    mov.f32 %f1, 0f00000000;");
        ptx.AppendLine("EXP_SUM_LOOP:"); ptx.AppendLine("    setp.ge.u32 %p1, %r4, 256;"); ptx.AppendLine("    @%p1 bra EXP_SUM_DONE;");
        ptx.AppendLine("    add.u32 %r5, %r3, %r4;"); Address(ptx, 5, 0, 4); ptx.AppendLine("    ld.global.f32 %f2, [%rd4];");
        ptx.AppendLine("    sub.rn.f32 %f2, %f2, %f0;"); ptx.AppendLine("    mul.rn.f32 %f2, %f2, 0f3fb8aa3b;");
        ptx.AppendLine("    ex2.approx.f32 %f2, %f2;"); Address(ptx, 5, 2, 5); ptx.AppendLine("    st.global.f32 [%rd5], %f2;");
        ptx.AppendLine("    add.rn.f32 %f1, %f1, %f2;"); ptx.AppendLine("    add.u32 %r4, %r4, 1;"); ptx.AppendLine("    bra EXP_SUM_LOOP;");
        ptx.AppendLine("EXP_SUM_DONE:"); Address(ptx, 2, 3, 6); ptx.AppendLine("    st.global.f32 [%rd6], %f1;");
        ptx.AppendLine("EXP_SUM_EXIT:"); End(ptx);
    }

    private static void EmitNormalize(StringBuilder ptx)
    {
        Header(ptx, DirectPtxMeshPoolOperation.SoftmaxNormalize, "exp_values", "global_sum", "softmax_scores");
        Registers(ptx); LoadPointers(ptx, 3); LinearIndex(ptx); Address(ptx, 2, 0, 3); ptx.AppendLine("    ld.global.f32 %f0, [%rd3];");
        ptx.AppendLine("    ld.global.f32 %f1, [%rd1];"); ptx.AppendLine("    add.rn.f32 %f1, %f1, 0f26901d7d;");
        ptx.AppendLine("    div.rn.f32 %f2, %f0, %f1;"); Address(ptx, 2, 2, 4); ptx.AppendLine("    st.global.f32 [%rd4], %f2;"); End(ptx);
    }

    private static void EmitSoftmaxScores(StringBuilder ptx)
    {
        Header(ptx, DirectPtxMeshPoolOperation.SoftmaxScores, "scores", "softmax_scores");
        Registers(ptx); ptx.AppendLine("    .shared .align 16 .b8 softmax_scratch[1024];");
        LoadPointers(ptx, 2); ptx.AppendLine("    mov.u32 %r0, %tid.x;");
        ptx.AppendLine("    mov.u64 %rd2, softmax_scratch;"); ptx.AppendLine("    mul.wide.u32 %rd3, %r0, 4;");
        ptx.AppendLine("    add.u64 %rd3, %rd2, %rd3;"); Address(ptx, 0, 0, 4);
        ptx.AppendLine("    ld.global.f32 %f0, [%rd4];"); ptx.AppendLine("    st.shared.f32 [%rd3], %f0;");
        ptx.AppendLine("    bar.sync 0;");
        EmitSharedReduction(ptx, max: true);
        ptx.AppendLine("    ld.shared.f32 %f1, [%rd2];"); ptx.AppendLine("    sub.rn.f32 %f2, %f0, %f1;");
        ptx.AppendLine("    mul.rn.f32 %f2, %f2, 0f3fb8aa3b;"); ptx.AppendLine("    ex2.approx.f32 %f2, %f2;");
        ptx.AppendLine("    st.shared.f32 [%rd3], %f2;"); ptx.AppendLine("    bar.sync 0;");
        EmitSharedReduction(ptx, max: false);
        ptx.AppendLine("    ld.shared.f32 %f3, [%rd2];"); ptx.AppendLine("    add.rn.f32 %f3, %f3, 0f26901d7d;");
        ptx.AppendLine("    div.rn.f32 %f2, %f2, %f3;"); Address(ptx, 0, 1, 5);
        ptx.AppendLine("    st.global.f32 [%rd5], %f2;"); End(ptx);
    }

    private static void EmitSharedReduction(StringBuilder ptx, bool max)
    {
        for (int stride = 128; stride >= 1; stride >>= 1)
        {
            ptx.AppendLine($"    setp.ge.u32 %p0, %r0, {stride};");
            ptx.AppendLine($"    @!%p0 add.u64 %rd6, %rd3, {stride * sizeof(float)};");
            ptx.AppendLine("    @!%p0 ld.shared.f32 %f4, [%rd3];");
            ptx.AppendLine("    @!%p0 ld.shared.f32 %f5, [%rd6];");
            ptx.AppendLine(max
                ? "    @!%p0 max.f32 %f4, %f4, %f5;"
                : "    @!%p0 add.rn.f32 %f4, %f4, %f5;");
            ptx.AppendLine("    @!%p0 st.shared.f32 [%rd3], %f4;");
            ptx.AppendLine("    bar.sync 0;");
        }
    }

    private static void EmitScoresBackward(StringBuilder ptx)
    {
        Header(ptx, DirectPtxMeshPoolOperation.ScoresBackward, "grad_output", "input", "kept", "grad_scores");
        Registers(ptx); LoadPointers(ptx, 4); LinearIndex(ptx); ptx.AppendLine("    mov.u32 %r3, 0;"); ptx.AppendLine("    mov.s32 %r4, -1;");
        ptx.AppendLine("SCORES_FIND:"); ptx.AppendLine($"    setp.ge.u32 %p0, %r3, {NumKept};"); ptx.AppendLine("    @%p0 bra SCORES_DOT;");
        Address(ptx, 3, 2, 4); ptx.AppendLine("    ld.global.s32 %r5, [%rd4];"); ptx.AppendLine("    setp.eq.s32 %p1, %r5, %r2;");
        ptx.AppendLine("    @%p1 mov.u32 %r4, %r3;"); ptx.AppendLine("    @%p1 bra SCORES_DOT;"); ptx.AppendLine("    add.u32 %r3, %r3, 1;"); ptx.AppendLine("    bra SCORES_FIND;");
        ptx.AppendLine("SCORES_DOT:"); ptx.AppendLine("    setp.lt.s32 %p2, %r4, 0;"); ptx.AppendLine("    @%p2 bra SCORES_ZERO;");
        ptx.AppendLine("    mov.f32 %f0, 0f00000000;"); ptx.AppendLine("    shl.b32 %r6, %r4, 6;"); ptx.AppendLine("    shl.b32 %r7, %r2, 6;");
        for (int c = 0; c < Channels; c++)
        {
            ptx.AppendLine($"    add.u32 %r8, %r6, {c};"); Address(ptx, 8, 0, 5); ptx.AppendLine("    ld.global.f32 %f1, [%rd5];");
            ptx.AppendLine($"    add.u32 %r9, %r7, {c};"); Address(ptx, 9, 1, 6); ptx.AppendLine("    ld.global.f32 %f2, [%rd6];");
            ptx.AppendLine("    fma.rn.f32 %f0, %f1, %f2, %f0;");
        }
        ptx.AppendLine("    bra SCORES_STORE;"); ptx.AppendLine("SCORES_ZERO:"); ptx.AppendLine("    mov.f32 %f0, 0f00000000;");
        ptx.AppendLine("SCORES_STORE:"); Address(ptx, 2, 3, 7); ptx.AppendLine("    st.global.f32 [%rd7], %f0;"); End(ptx);
    }

    private static void Header(StringBuilder ptx, DirectPtxMeshPoolOperation operation, params string[] pointers)
    {
        ptx.AppendLine($".visible .entry {GetEntryPoint(operation)}(");
        for (int i = 0; i < pointers.Length; i++) ptx.AppendLine($"    .param .u64 p{i}{(i + 1 == pointers.Length ? string.Empty : ",")}");
        ptx.AppendLine(")"); ptx.AppendLine("{");
    }

    private static void Registers(StringBuilder ptx)
    {
        ptx.AppendLine("    .reg .pred %p<8>;"); ptx.AppendLine("    .reg .b32 %r<16>;");
        ptx.AppendLine("    .reg .b64 %rd<20>;"); ptx.AppendLine("    .reg .f32 %f<12>;");
    }

    private static void LoadPointers(StringBuilder ptx, int count)
    {
        for (int i = 0; i < count; i++) ptx.AppendLine($"    ld.param.u64 %rd{i}, [p{i}];");
    }

    private static void LinearIndex(StringBuilder ptx)
    {
        ptx.AppendLine("    mov.u32 %r0, %tid.x;"); ptx.AppendLine("    mov.u32 %r1, %ctaid.x;");
        ptx.AppendLine($"    mad.lo.u32 %r2, %r1, {BlockThreads}, %r0;");
    }

    private static void Address(StringBuilder ptx, int indexRegister, int pointerRegister, int outputAddressRegister)
    {
        ptx.AppendLine($"    mul.wide.u32 %rd{outputAddressRegister}, %r{indexRegister}, 4;");
        ptx.AppendLine($"    add.u64 %rd{outputAddressRegister}, %rd{pointerRegister}, %rd{outputAddressRegister};");
    }

    private static void End(StringBuilder ptx) { ptx.AppendLine("    ret;"); ptx.AppendLine("}"); }

    internal static string GetKernelName(DirectPtxMeshPoolOperation operation) => operation switch
    {
        DirectPtxMeshPoolOperation.ComputeScores => "mesh_pool_compute_scores",
        DirectPtxMeshPoolOperation.Gather => "mesh_pool_gather",
        DirectPtxMeshPoolOperation.BackwardAtomic => "mesh_pool_backward",
        DirectPtxMeshPoolOperation.BackwardDeterministic => "mesh_pool_backward_deterministic",
        DirectPtxMeshPoolOperation.ImportanceBackward => "mesh_pool_importance_backward",
        DirectPtxMeshPoolOperation.ZeroGrad => "mesh_pool_zero_grad",
        DirectPtxMeshPoolOperation.SoftmaxFindMax => "mesh_pool_softmax_find_max",
        DirectPtxMeshPoolOperation.SoftmaxFinalMax => "mesh_pool_softmax_final_max",
        DirectPtxMeshPoolOperation.SoftmaxExpSum => "mesh_pool_softmax_exp_sum",
        DirectPtxMeshPoolOperation.SoftmaxFinalSum => "mesh_pool_softmax_final_sum",
        DirectPtxMeshPoolOperation.SoftmaxNormalize => "mesh_pool_softmax_normalize",
        DirectPtxMeshPoolOperation.SoftmaxScores => "mesh_pool_softmax_scores",
        DirectPtxMeshPoolOperation.WeightedGather => "mesh_pool_weighted_gather",
        DirectPtxMeshPoolOperation.WeightedBackwardAtomic => "mesh_pool_weighted_backward",
        DirectPtxMeshPoolOperation.WeightedBackwardDeterministic => "mesh_pool_weighted_backward_deterministic",
        DirectPtxMeshPoolOperation.ScoresBackward => "mesh_pool_scores_backward",
        _ => throw new ArgumentOutOfRangeException(nameof(operation))
    };

    internal static string GetEntryPoint(DirectPtxMeshPoolOperation operation) =>
        $"aidotnet_{GetKernelName(operation)}_f32_e16384_k4096_c64";

    private static DirectPtxTensorContract Exact(string name, DirectPtxPhysicalType type,
        DirectPtxPhysicalLayout layout, DirectPtxExtent extent, DirectPtxTensorAccess access) =>
        new(name, type, layout, extent, extent, 16, access, DirectPtxExtentMode.Exact);

    private static void Require(DirectPtxTensorView view, DirectPtxTensorContract contract, string parameter)
    {
        if (view.Pointer == IntPtr.Zero || view.PhysicalType != contract.PhysicalType ||
            view.Layout != contract.Layout || view.LogicalExtent != contract.LogicalExtent ||
            view.PhysicalExtent != contract.PhysicalExtent || view.ByteLength != contract.RequiredBytes ||
            view.AllocationByteLength != contract.RequiredBytes || view.Access != contract.Access)
            throw new ArgumentException($"{parameter} does not satisfy exact physical ABI '{contract.Name}'.", parameter);
    }

    private static bool Overlaps(DirectPtxTensorView left, DirectPtxTensorView right)
    {
        nuint leftStart = (nuint)left.Pointer; nuint rightStart = (nuint)right.Pointer;
        nuint leftEnd = checked(leftStart + left.ByteLength); nuint rightEnd = checked(rightStart + right.ByteLength);
        return leftStart < rightEnd && rightStart < leftEnd;
    }

    public void Dispose() => _module.Dispose();
}
#endif
