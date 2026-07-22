using System;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

internal enum DirectPtxVisionOperation
{
    GeneralizedBoxIou,
    DistanceBoxIou,
    CompleteBoxIou,
    BoxArea,
    BoxConvert,
    IoULoss,
    GIoULoss,
    DIoULoss,
    CIoULoss,
    IoULossBackward,
    GIoULossBackward,
    DIoULossBackward,
    CIoULossBackward,
    IouFamilyBackwardA,
    IouFamilyBackwardB,
    Nms,
    MasksToBoxes,
    RoiAlign,
    RoiPool,
    PsRoiAlign,
    PsRoiPool,
    Cross3,
    Meshgrid2D
}

/// <summary>
/// Allocation-free cache key. Every semantic, dimensional, and numerical
/// choice is baked into the generated module rather than passed to PTX.
/// </summary>
internal readonly record struct DirectPtxVisionSpec(
    DirectPtxVisionOperation Operation,
    int D0 = 0,
    int D1 = 0,
    int D2 = 0,
    int D3 = 0,
    int D4 = 0,
    int D5 = 0,
    int D6 = 0,
    int D7 = 0,
    int Flags = 0,
    int ScalarBits = 0);

internal sealed record DirectPtxVisionDefinition(
    DirectPtxKernelBlueprint Blueprint,
    string Ptx,
    uint GridX,
    uint GridY,
    uint GridZ,
    uint BlockX,
    uint BlockY,
    uint BlockZ,
    uint DynamicSharedBytes = 0);

/// <summary>Loaded direct-PTX module for one exact vision specialization.</summary>
internal sealed class PtxVisionKernel : IDisposable
{
    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

    internal DirectPtxVisionSpec Spec { get; }
    internal DirectPtxVisionDefinition Definition { get; }
    internal DirectPtxKernelBlueprint Blueprint => Definition.Blueprint;
    internal DirectPtxFunctionInfo FunctionInfo { get; }
    internal DirectPtxKernelAudit Audit { get; }
    internal string EntryPoint => PtxVisionEmitter.EntryPoint(Spec.Operation);

    internal PtxVisionKernel(DirectPtxRuntime runtime, DirectPtxVisionSpec spec)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        if (!DirectPtxArchitecture.HasValidatedVisionBoxIou(
            runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new NotSupportedException(
                $"Vision operation {spec.Operation} has no SM {runtime.ComputeCapabilityMajor}.{runtime.ComputeCapabilityMinor} specialization.");
        Spec = spec;
        Definition = PtxVisionEmitter.Emit(
            spec, runtime.ArchitectureFamily,
            runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor);
        _module = runtime.LoadModule(Definition.Ptx);
        _function = _module.GetFunction(EntryPoint, out DirectPtxFunctionInfo info);
        FunctionInfo = info;
        int blockThreads = checked((int)(Definition.BlockX * Definition.BlockY * Definition.BlockZ));
        int activeBlocks = _module.GetActiveBlocksPerMultiprocessor(_function, blockThreads);
        Blueprint.ResourceBudget.Validate(EntryPoint, info, blockThreads, activeBlocks);
        Audit = DirectPtxKernelAudit.Create(
            Blueprint, runtime.DeviceFingerprint, Definition.Ptx, info,
            blockThreads, activeBlocks, _module.JitInfoLog);
    }

    internal unsafe void Launch(
        DirectPtxTensorView a,
        DirectPtxTensorView b = default,
        DirectPtxTensorView c = default,
        DirectPtxTensorView d = default,
        DirectPtxTensorView e = default,
        DirectPtxTensorView f = default)
    {
        Validate(a, b, c, d, e, f);
        LaunchUnchecked(a, b, c, d, e, f);
    }

    internal void Validate(
        DirectPtxTensorView a,
        DirectPtxTensorView b = default,
        DirectPtxTensorView c = default,
        DirectPtxTensorView d = default,
        DirectPtxTensorView e = default,
        DirectPtxTensorView f = default)
    {
        int count = Blueprint.Tensors.Count;
        if (count is < 1 or > 6) throw new InvalidOperationException("Vision PTX ABI supports one through six tensors.");
        Span<DirectPtxTensorView> views = stackalloc DirectPtxTensorView[6] { a, b, c, d, e, f };
        ValidateViews(Blueprint, views[..count]);
    }

    internal unsafe void LaunchUnchecked(
        DirectPtxTensorView a,
        DirectPtxTensorView b = default,
        DirectPtxTensorView c = default,
        DirectPtxTensorView d = default,
        DirectPtxTensorView e = default,
        DirectPtxTensorView f = default)
    {
        int count = Blueprint.Tensors.Count;
        Span<DirectPtxTensorView> views = stackalloc DirectPtxTensorView[6] { a, b, c, d, e, f };
        IntPtr* pointers = stackalloc IntPtr[count];
        void** arguments = stackalloc void*[count];
        for (int i = 0; i < count; i++)
        {
            pointers[i] = views[i].Pointer;
            arguments[i] = &pointers[i];
        }
        _module.Launch(
            _function,
            Definition.GridX, Definition.GridY, Definition.GridZ,
            Definition.BlockX, Definition.BlockY, Definition.BlockZ,
            Definition.DynamicSharedBytes, arguments);
    }

    internal static void ValidateViews(
        DirectPtxKernelBlueprint blueprint,
        ReadOnlySpan<DirectPtxTensorView> views)
    {
        PtxCompat.ThrowIfNull(blueprint, nameof(blueprint));
        if (views.Length != blueprint.Tensors.Count)
            throw new ArgumentException(
                $"Vision PTX ABI expects {blueprint.Tensors.Count} tensors; got {views.Length}.",
                nameof(views));
        for (int i = 0; i < views.Length; i++)
            Require(views[i], blueprint.Tensors[i], i);
        RejectWriteAliasing(blueprint, views);
    }

    private static void Require(
        DirectPtxTensorView view, DirectPtxTensorContract contract, int index)
    {
        if (view.Pointer == IntPtr.Zero || view.PhysicalType != contract.PhysicalType ||
            view.Layout != contract.Layout || view.LogicalExtent != contract.LogicalExtent ||
            view.PhysicalExtent != contract.PhysicalExtent ||
            view.ByteLength != contract.RequiredBytes ||
            view.AllocationByteLength != checked(contract.ByteOffset + contract.RequiredBytes) ||
            view.Access != contract.Access)
            throw new ArgumentException(
                $"Tensor argument {index} does not satisfy physical ABI '{contract.Name}'.");
    }

    private static void RejectWriteAliasing(
        DirectPtxKernelBlueprint blueprint,
        ReadOnlySpan<DirectPtxTensorView> views)
    {
        for (int i = 0; i < views.Length; i++)
        for (int j = i + 1; j < views.Length; j++)
        {
            if ((blueprint.Tensors[i].Access & DirectPtxTensorAccess.Write) == 0 &&
                (blueprint.Tensors[j].Access & DirectPtxTensorAccess.Write) == 0)
                continue;
            nuint left = PtxCompat.ToNuint(views[i].Pointer);
            nuint right = PtxCompat.ToNuint(views[j].Pointer);
            if (left < checked(right + views[j].ByteLength) &&
                right < checked(left + views[i].ByteLength))
                throw new ArgumentException(
                    $"Vision PTX tensors '{blueprint.Tensors[i].Name}' and '{blueprint.Tensors[j].Name}' may not alias.");
        }
    }

    public void Dispose() => _module.Dispose();
}
