using System;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines;

public partial class DirectGpuTensorEngine
{
    internal Tensor<T> GroupedConv2DOutput<T>(Tensor<T> input, Tensor<T> kernel, in GroupedConv2DGeometry g)
    {
        using var operation = EnterGpuOp();
        if (!operation.Held || DirectGpuEngine.ShouldFallbackForPrecision<T>() || g.Batch == 0
            || !TryGetBackend(out var backend) || backend is not IGroupedConv2DBackend native)
        {
            var host = TensorAllocator.RentUninitialized<T>(g.OutputShape);
            base.GroupedConv2DInto(host, input, kernel, g);
            return host;
        }
        var output = CreateResidentOutput<T>(backend, g.OutputShape);
        ExecuteGroupedConv2D(output, input, kernel, g, backend, native, callerOwned: false);
        return output;
    }

    internal override void GroupedConv2DInto<T>(Tensor<T> output, Tensor<T> input, Tensor<T> kernel,
        in GroupedConv2DGeometry g)
    {
        using var operation = EnterGpuOp();
        if (!operation.Held || DirectGpuEngine.ShouldFallbackForPrecision<T>() || g.Batch == 0
            || !IsCanonicalDenseAllocation(output) || !TryGetBackend(out var backend) || backend is not IGroupedConv2DBackend native)
        {
            base.GroupedConv2DInto(output, input, kernel, g);
            return;
        }
        output.PrepareForInPlaceWrite();
        ExecuteGroupedConv2D(output, input, kernel, g, backend, native, callerOwned: true);
    }

    private void ExecuteGroupedConv2D<T>(Tensor<T> output, Tensor<T> input, Tensor<T> kernel,
        in GroupedConv2DGeometry g, IDirectGpuBackend backend, IGroupedConv2DBackend native, bool callerOwned)
    {
        using var sourceInput = GetResidentOrPersistentInputBuffer(backend, input);
        using var sourceKernel = GetResidentOrPersistentInputBuffer(backend, kernel);
        var target = GetOrCreateResidentBuffer(backend, output, output.Length, fullyWritten: true);
        using var readInput = SnapshotWeightNormAlias(backend, sourceInput.Buffer, input.Length, target);
        using var readKernel = SnapshotWeightNormAlias(backend, sourceKernel.Buffer, kernel.Length, target);
        if (!native.TryGroupedConv2D(readInput.Buffer, readKernel.Buffer, target, g))
            throw new NotSupportedException("Native grouped convolution declined valid canonical buffers.");
        ResidentSyncCheck("GroupedConv2D");
        output.IncrementVersion();
        BindResidentBuffer(output, target, backend);
        if (callerOwned) MarkCallerOwnedDestination(output);
    }

    internal override void GroupedConv2DBackwardInto<T>(Tensor<T> gradOutput, Tensor<T> input, Tensor<T> kernel,
        Tensor<T>? gradInput, Tensor<T>? gradKernel, in GroupedConv2DGeometry g,
        bool accumulateInput, bool accumulateKernel)
    {
        if (gradInput is null && gradKernel is null) return;
        using var operation = EnterGpuOp();
        if (!operation.Held || DirectGpuEngine.ShouldFallbackForPrecision<T>() || g.Batch == 0
            || (gradInput is not null && !IsCanonicalDenseAllocation(gradInput))
            || (gradKernel is not null && !IsCanonicalDenseAllocation(gradKernel))
            || (gradInput is not null && gradKernel is not null && !ReferenceEquals(gradInput, gradKernel)
                && gradInput.SharesStorageWith(gradKernel))
            || !TryGetBackend(out var backend) || backend is not IGroupedConv2DBackend native)
        {
            base.GroupedConv2DBackwardInto(gradOutput, input, kernel, gradInput, gradKernel, g, accumulateInput, accumulateKernel);
            return;
        }
        bool tied = gradInput is not null && ReferenceEquals(gradInput, gradKernel);
        gradInput?.PrepareForInPlaceWrite();
        if (!tied) gradKernel?.PrepareForInPlaceWrite();
        using var sourceOutput = GetResidentOrPersistentInputBuffer(backend, gradOutput);
        using var sourceInput = gradKernel is not null ? GetResidentOrPersistentInputBuffer(backend, input) : default;
        using var sourceKernel = gradInput is not null ? GetResidentOrPersistentInputBuffer(backend, kernel) : default;
        var dx = gradInput is null ? null : GroupedGradientDestination(backend, gradInput, accumulateInput);
        var dw = tied ? dx : gradKernel is null ? null : GroupedGradientDestination(backend, gradKernel, accumulateKernel);
        // Resolve all primal aliases before either pass writes a destination.
        using var readOutput = SnapshotWeightNormAlias(backend, sourceOutput.Buffer, gradOutput.Length, dx, dw);
        using var readInput = dw is not null ? SnapshotWeightNormAlias(backend, sourceInput.Buffer, input.Length, dx, dw) : default;
        using var readKernel = dx is not null ? SnapshotWeightNormAlias(backend, sourceKernel.Buffer, kernel.Length, dx, dw) : default;
        if (dx is not null && gradInput is not null)
        {
            if (!native.TryGroupedConv2DBackwardInput(readOutput.Buffer, readKernel.Buffer, dx, g, accumulateInput))
                throw new NotSupportedException("Native grouped input gradient declined valid canonical buffers.");
            gradInput.IncrementVersion(); BindResidentBuffer(gradInput, dx, backend); MarkCallerOwnedDestination(gradInput);
        }
        if (dw is not null && gradKernel is not null)
        {
            if (!native.TryGroupedConv2DBackwardKernel(readInput.Buffer, readOutput.Buffer, dw, g, accumulateKernel))
                throw new NotSupportedException("Native grouped kernel gradient declined valid canonical buffers.");
            gradKernel.IncrementVersion(); BindResidentBuffer(gradKernel, dw, backend); MarkCallerOwnedDestination(gradKernel);
        }
        ResidentSyncCheck("GroupedConv2DBackward");
    }

    private IGpuBuffer GroupedGradientDestination<T>(IDirectGpuBackend backend, Tensor<T> gradient, bool accumulate)
    {
        if (!accumulate) return GetOrCreateResidentBuffer(backend, gradient, gradient.Length, fullyWritten: true);
        using var previous = GetResidentOrPersistentInputBuffer(backend, gradient);
        if (!previous.OwnsBuffer) return previous.Buffer;

        // Eager accumulation can start with a host-only gradient. Transfer its upload to the activation cache
        // before either backward kernel runs, so later reads and a failure of the other kernel retain an owner.
        object key = (object?)gradient.GetBackingArrayForCacheLookupUnsafe() ?? gradient.DataVector;
        if (CacheActivation(key, previous.Buffer, gradient._shape, backend, hostVersion: gradient.GpuCacheVersion))
        {
            previous.RelinquishOwnership();
            return previous.Buffer;
        }

        var destination = GetOrCreateResidentBuffer(backend, gradient, gradient.Length, fullyWritten: true);
        backend.Copy(previous.Buffer, 0, destination, 0, gradient.Length);
        previous.DisposeAfterQueuedUse(backend);
        return destination;
    }
}
