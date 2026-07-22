#if NET5_0_OR_GREATER
using System;
using System.Collections.Generic;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

internal enum DirectPtxScalarScatterAddMode
{
    Atomic,
    DeterministicOverwrite,
    DeterministicAccumulate
}

/// <summary>
/// Exact scalar indexed scatter-add family. Atomic and fixed-order variants
/// are distinct modules because seeded accumulation and overwrite have
/// different observable contracts.
/// </summary>
internal sealed class PtxScatterAddScalarF32Kernel : IDisposable
{
    internal const int SourceElements = 16384;
    internal const int DestinationElements = 1024;
    internal const int BlockThreads = 256;

    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

    internal DirectPtxScalarScatterAddMode Mode { get; }
    internal string EntryPoint { get; }
    internal string Ptx { get; }
    internal DirectPtxFunctionInfo FunctionInfo { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal PtxScatterAddScalarF32Kernel(
        DirectPtxRuntime runtime,
        DirectPtxScalarScatterAddMode mode)
    {
        ArgumentNullException.ThrowIfNull(runtime);
        if (!DirectPtxArchitecture.HasValidatedSparseGraph(
            runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new NotSupportedException(
                $"Scalar scatter-add has no SM {runtime.ComputeCapabilityMajor}.{runtime.ComputeCapabilityMinor} experimental specialization.");
        Mode = mode;
        EntryPoint = GetEntryPoint(mode);
        Blueprint = CreateBlueprint(runtime.ArchitectureFamily, mode);
        Ptx = EmitPtx(runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor, mode);
        _module = runtime.LoadModule(Ptx);
        _function = _module.GetFunction(EntryPoint, out DirectPtxFunctionInfo functionInfo);
        FunctionInfo = functionInfo;
        int activeBlocks = _module.GetActiveBlocksPerMultiprocessor(_function, BlockThreads);
        Blueprint.ResourceBudget.Validate(EntryPoint, functionInfo, BlockThreads, activeBlocks);
        Audit = DirectPtxKernelAudit.Create(
            Blueprint, runtime.DeviceFingerprint, Ptx, functionInfo,
            BlockThreads, activeBlocks, _module.JitInfoLog);
    }

    internal static bool SupportsShape(int sourceElements, int destinationElements) =>
        sourceElements == SourceElements && destinationElements == DestinationElements;

    internal unsafe void Launch(
        DirectPtxTensorView source,
        DirectPtxTensorView indices,
        DirectPtxTensorView destination)
    {
        Require(source, Blueprint.Tensors[0], nameof(source));
        Require(indices, Blueprint.Tensors[1], nameof(indices));
        Require(destination, Blueprint.Tensors[2], nameof(destination));
        if (Overlaps(destination, source) || Overlaps(destination, indices))
            throw new ArgumentException("Scatter-add destination must be disjoint from source and indices.", nameof(destination));
        IntPtr sourcePointer = source.Pointer;
        IntPtr indicesPointer = indices.Pointer;
        IntPtr destinationPointer = destination.Pointer;
        void** arguments = stackalloc void*[3];
        arguments[0] = &sourcePointer;
        arguments[1] = &indicesPointer;
        arguments[2] = &destinationPointer;
        int workItems = Mode == DirectPtxScalarScatterAddMode.Atomic
            ? SourceElements : DestinationElements;
        _module.Launch(_function, (uint)(workItems / BlockThreads), 1, 1,
            BlockThreads, 1, 1, 0, arguments);
    }

    internal static DirectPtxKernelBlueprint CreateBlueprint(
        DirectPtxArchitectureFamily architecture,
        DirectPtxScalarScatterAddMode mode)
    {
        var source = new DirectPtxExtent(SourceElements);
        var destination = new DirectPtxExtent(DestinationElements);
        return new DirectPtxKernelBlueprint(
            Operation: mode switch
            {
                DirectPtxScalarScatterAddMode.Atomic => "scatter-add-scalar-atomic-f32",
                DirectPtxScalarScatterAddMode.DeterministicOverwrite =>
                    "scatter-add-scalar-deterministic-overwrite-f32",
                DirectPtxScalarScatterAddMode.DeterministicAccumulate =>
                    "scatter-add-scalar-deterministic-accumulate-f32",
                _ => throw new ArgumentOutOfRangeException(nameof(mode))
            },
            Version: 1,
            Architecture: architecture,
            Variant: "src16384-dst1024",
            Tensors:
            [
                new("source", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    source, source, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("indices", DirectPtxPhysicalType.Int32, DirectPtxPhysicalLayout.CooRowIndices,
                    source, source, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("destination", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    destination, destination, 16,
                    mode == DirectPtxScalarScatterAddMode.DeterministicOverwrite
                        ? DirectPtxTensorAccess.Write : DirectPtxTensorAccess.ReadWrite,
                    DirectPtxExtentMode.Exact)
            ],
            ResourceBudget: new DirectPtxResourceBudget(32, 0, 0, 6),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["equation"] = mode == DirectPtxScalarScatterAddMode.DeterministicOverwrite
                    ? "destination[j]=sum_i(indices[i]=j,source[i])"
                    : "destination[indices[i]]+=source[i]",
                ["reduction-order"] = mode == DirectPtxScalarScatterAddMode.Atomic
                    ? "unordered-global-fp32-atomic" : "ascending-source-index",
                ["destination-seed"] = mode == DirectPtxScalarScatterAddMode.DeterministicOverwrite
                    ? "discarded" : "preserved",
                ["workspace-bytes"] = "0",
                ["intermediate-global-bytes"] = "0"
            });
    }

    internal static string EmitPtx(
        int ccMajor,
        int ccMinor,
        DirectPtxScalarScatterAddMode mode)
    {
        if (ccMajor <= 0 || ccMinor < 0) throw new ArgumentOutOfRangeException(nameof(ccMajor));
        var ptx = new StringBuilder(3072);
        ptx.AppendLine(".version 7.1");
        ptx.AppendLine($".target sm_{ccMajor}{ccMinor}");
        ptx.AppendLine(".address_size 64");
        ptx.AppendLine();
        ptx.AppendLine($".visible .entry {GetEntryPoint(mode)}(");
        ptx.AppendLine("    .param .u64 source_ptr,");
        ptx.AppendLine("    .param .u64 indices_ptr,");
        ptx.AppendLine("    .param .u64 destination_ptr");
        ptx.AppendLine(")");
        ptx.AppendLine("{");
        ptx.AppendLine("    .reg .pred %p<3>;");
        ptx.AppendLine("    .reg .b32 %r<12>;");
        ptx.AppendLine("    .reg .b64 %rd<14>;");
        ptx.AppendLine("    .reg .f32 %f<5>;");
        ptx.AppendLine("    ld.param.u64 %rd0, [source_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd1, [indices_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd2, [destination_ptr];");
        ptx.AppendLine("    mov.u32 %r0, %tid.x;");
        ptx.AppendLine("    mov.u32 %r1, %ctaid.x;");
        ptx.AppendLine($"    mad.lo.u32 %r2, %r1, {BlockThreads}, %r0;");
        if (mode == DirectPtxScalarScatterAddMode.Atomic)
        {
            ptx.AppendLine("    mul.wide.u32 %rd3, %r2, 4;");
            ptx.AppendLine("    add.u64 %rd4, %rd0, %rd3;");
            ptx.AppendLine("    add.u64 %rd5, %rd1, %rd3;");
            ptx.AppendLine("    ld.global.f32 %f0, [%rd4];");
            ptx.AppendLine("    ld.global.u32 %r3, [%rd5];");
            ptx.AppendLine("    mul.wide.u32 %rd6, %r3, 4;");
            ptx.AppendLine("    add.u64 %rd7, %rd2, %rd6;");
            ptx.AppendLine("    atom.global.add.f32 %f1, [%rd7], %f0;");
        }
        else
        {
            ptx.AppendLine("    mov.u32 %r3, 0;");
            ptx.AppendLine("    mov.f32 %f0, 0f00000000;");
            ptx.AppendLine("SCATTER_LOOP:");
            ptx.AppendLine($"    setp.ge.u32 %p0, %r3, {SourceElements};");
            ptx.AppendLine("    @%p0 bra SCATTER_DONE;");
            ptx.AppendLine("    mul.wide.u32 %rd3, %r3, 4;");
            ptx.AppendLine("    add.u64 %rd4, %rd1, %rd3;");
            ptx.AppendLine("    ld.global.u32 %r4, [%rd4];");
            ptx.AppendLine("    setp.ne.u32 %p1, %r4, %r2;");
            ptx.AppendLine("    @%p1 bra SCATTER_NEXT;");
            ptx.AppendLine("    add.u64 %rd5, %rd0, %rd3;");
            ptx.AppendLine("    ld.global.f32 %f1, [%rd5];");
            ptx.AppendLine("    add.rn.f32 %f0, %f0, %f1;");
            ptx.AppendLine("SCATTER_NEXT:");
            ptx.AppendLine("    add.u32 %r3, %r3, 1;");
            ptx.AppendLine("    bra SCATTER_LOOP;");
            ptx.AppendLine("SCATTER_DONE:");
            ptx.AppendLine("    mul.wide.u32 %rd6, %r2, 4;");
            ptx.AppendLine("    add.u64 %rd7, %rd2, %rd6;");
            if (mode == DirectPtxScalarScatterAddMode.DeterministicAccumulate)
            {
                ptx.AppendLine("    ld.global.f32 %f2, [%rd7];");
                ptx.AppendLine("    add.rn.f32 %f0, %f2, %f0;");
            }
            ptx.AppendLine("    st.global.f32 [%rd7], %f0;");
        }
        ptx.AppendLine("    ret;");
        ptx.AppendLine("}");
        return ptx.ToString();
    }

    private static string GetEntryPoint(DirectPtxScalarScatterAddMode mode) => mode switch
    {
        DirectPtxScalarScatterAddMode.Atomic => "aidotnet_scatter_add_atomic_f32_src16384_dst1024",
        DirectPtxScalarScatterAddMode.DeterministicOverwrite =>
            "aidotnet_scatter_add_deterministic_f32_src16384_dst1024",
        DirectPtxScalarScatterAddMode.DeterministicAccumulate =>
            "aidotnet_scatter_add_accumulate_deterministic_f32_src16384_dst1024",
        _ => throw new ArgumentOutOfRangeException(nameof(mode))
    };

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
