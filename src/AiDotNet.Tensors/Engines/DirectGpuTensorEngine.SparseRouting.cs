using System;
using System.Linq;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines;

public partial class DirectGpuTensorEngine
{
    public override Tensor<T> TensorGatherRowValues<T>(Tensor<T> values, Tensor<int> indices)
    {
        using var __gpuOp = EnterGpuOp();
        if (Compilation.GraphMode.IsActive || typeof(T) != typeof(float) || !TryGetBackend(out var backend)
            || SparseRoutingResolver.Resolve(backend) is not { } sr
            || values is null || indices is null || values.Rank < 1 || indices.Rank != values.Rank
            || values._shape[values.Rank - 1] < 1 || indices._shape[indices.Rank - 1] < 1
            || values.Length == 0 || indices.Length == 0)
            return base.TensorGatherRowValues(values!, indices!);

        for (int d = 0; d < values.Rank - 1; d++)
            if (values._shape[d] != indices._shape[d])
                return base.TensorGatherRowValues(values, indices);

        int cols = values._shape[values.Rank - 1];
        int k = indices._shape[indices.Rank - 1];
        int rows = values.Length / cols;
        var contiguousValues = values.IsContiguous ? values : values.Contiguous();
        var contiguousIndices = indices.IsContiguous ? indices : indices.Contiguous();
        try
        {
            using var vB = GetOrAllocateBuffer(backend, contiguousValues);
            using var idxB = GetOrAllocateInt32IndexBuffer(backend, contiguousIndices);
            using var yB = AllocateFullyWrittenOutputBuffer(backend, indices.Length);
            sr.GatherRowValuesForward(vB.Buffer, idxB.Buffer, yB.Buffer, rows, cols, k);
            var result = DeferTensorResult<T>(backend, yB.Buffer, indices.Length, (int[])indices._shape.Clone());
            yB.RelinquishOwnership();
            if (Autodiff.DifferentiableOps.IsRecording<T>())
            {
                var savedIndices = SnapshotIndices(backend, contiguousIndices);
                Autodiff.DifferentiableOps.RecordIfActive(
                    "TensorGatherRowValues", result, new[] { values },
                    CpuEngine.TensorGatherRowValuesAutoBackward<T>,
                    savedState: new object[] { savedIndices, cols });
            }
            return result;
        }
        catch (NotSupportedException capabilityGap)
        {
            GpuLaunchProbe.OnFallback("TensorGatherRowValues", capabilityGap);
            return base.TensorGatherRowValues(values, indices);
        }
    }

    public override Tensor<T> TensorGatherRowValuesBackward<T>(Tensor<T> gradOutput, Tensor<int> indices, int numCols)
    {
        using var __gpuOp = EnterGpuOp();
        if (typeof(T) != typeof(float) || !TryGetBackend(out var backend)
            || SparseRoutingResolver.Resolve(backend) is not { } sr
            || gradOutput is null || indices is null || numCols < 1
            || gradOutput.Rank < 1 || !Enumerable.SequenceEqual(gradOutput._shape, indices._shape)
            || gradOutput.Length == 0)
            return base.TensorGatherRowValuesBackward(gradOutput!, indices!, numCols);

        int k = gradOutput._shape[gradOutput.Rank - 1];
        int rows = gradOutput.Length / k;
        int outLen = checked(rows * numCols);
        var outShape = (int[])gradOutput._shape.Clone();
        outShape[outShape.Length - 1] = numCols;
        var contiguousGrad = gradOutput.IsContiguous ? gradOutput : gradOutput.Contiguous();
        var contiguousIndices = indices.IsContiguous ? indices : indices.Contiguous();
        try
        {
            using var gB = GetOrAllocateBuffer(backend, contiguousGrad);
            using var idxB = GetOrAllocateInt32IndexBuffer(backend, contiguousIndices);
            using var dvB = AllocateFullyWrittenOutputBuffer(backend, outLen);
            sr.GatherRowValuesBackward(gB.Buffer, idxB.Buffer, dvB.Buffer, rows, numCols, k);
            var result = DeferTensorResult<T>(backend, dvB.Buffer, outLen, outShape);
            dvB.RelinquishOwnership();
            return result;
        }
        catch (NotSupportedException capabilityGap)
        {
            GpuLaunchProbe.OnFallback("TensorGatherRowValuesBackward", capabilityGap);
            return base.TensorGatherRowValuesBackward(gradOutput, indices, numCols);
        }
    }

    public override Tensor<T> TopKWeightedRowMix<T>(Tensor<T> table, Tensor<int> indices, Tensor<T> weights)
    {
        using var __gpuOp = EnterGpuOp();
        if (Compilation.GraphMode.IsActive || typeof(T) != typeof(float) || !TryGetBackend(out var backend)
            || SparseRoutingResolver.Resolve(backend) is not { } sr
            || table is null || indices is null || weights is null
            || table.Rank != 2 || indices.Rank < 1 || !Enumerable.SequenceEqual(indices._shape, weights._shape)
            || table._shape[0] < 1 || table._shape[1] < 1 || indices._shape[indices.Rank - 1] < 1
            || indices.Length == 0)
            return base.TopKWeightedRowMix(table!, indices!, weights!);

        int numEntries = table._shape[0], channels = table._shape[1];
        int k = indices._shape[indices.Rank - 1];
        int rows = indices.Length / k;
        int outLen = checked(rows * channels);
        var outShape = (int[])indices._shape.Clone();
        outShape[outShape.Length - 1] = channels;
        var contiguousTable = table.IsContiguous ? table : table.Contiguous();
        var contiguousIndices = indices.IsContiguous ? indices : indices.Contiguous();
        var contiguousWeights = weights.IsContiguous ? weights : weights.Contiguous();
        try
        {
            using var tblB = GetOrAllocateBuffer(backend, contiguousTable);
            using var idxB = GetOrAllocateInt32IndexBuffer(backend, contiguousIndices);
            using var wB = GetOrAllocateBuffer(backend, contiguousWeights);
            using var yB = AllocateFullyWrittenOutputBuffer(backend, outLen);
            sr.TopKWeightedRowMixForward(tblB.Buffer, idxB.Buffer, wB.Buffer, yB.Buffer, rows, numEntries, k, channels);
            var result = DeferTensorResult<T>(backend, yB.Buffer, outLen, outShape);
            yB.RelinquishOwnership();
            if (Autodiff.DifferentiableOps.IsRecording<T>())
            {
                var savedIndices = SnapshotIndices(backend, contiguousIndices);
                Autodiff.DifferentiableOps.RecordIfActive(
                    "TopKWeightedRowMix", result, new[] { table, weights },
                    CpuEngine.TopKWeightedRowMixAutoBackward<T>,
                    savedState: new object[] { savedIndices });
            }
            return result;
        }
        catch (NotSupportedException capabilityGap)
        {
            GpuLaunchProbe.OnFallback("TopKWeightedRowMix", capabilityGap);
            return base.TopKWeightedRowMix(table, indices, weights);
        }
    }

    public override Tensor<T> TopKWeightedRowMixBackwardWeights<T>(
        Tensor<T> gradOutput, Tensor<T> table, Tensor<int> indices)
    {
        using var __gpuOp = EnterGpuOp();
        if (typeof(T) != typeof(float) || !TryGetBackend(out var backend)
            || SparseRoutingResolver.Resolve(backend) is not { } sr
            || gradOutput is null || table is null || indices is null
            || table.Rank != 2 || gradOutput.Rank < 1 || gradOutput.Rank != indices.Rank
            || table._shape[0] < 1 || table._shape[1] < 1 || gradOutput._shape[gradOutput.Rank - 1] != table._shape[1]
            || indices._shape[indices.Rank - 1] < 1 || indices.Length == 0 || gradOutput.Length == 0)
            return base.TopKWeightedRowMixBackwardWeights(gradOutput!, table!, indices!);

        for (int d = 0; d < gradOutput.Rank - 1; d++)
            if (gradOutput._shape[d] != indices._shape[d])
                return base.TopKWeightedRowMixBackwardWeights(gradOutput, table, indices);

        int numEntries = table._shape[0], channels = table._shape[1];
        int k = indices._shape[indices.Rank - 1];
        int rows = gradOutput.Length / channels;
        var contiguousGrad = gradOutput.IsContiguous ? gradOutput : gradOutput.Contiguous();
        var contiguousTable = table.IsContiguous ? table : table.Contiguous();
        var contiguousIndices = indices.IsContiguous ? indices : indices.Contiguous();
        try
        {
            using var goB = GetOrAllocateBuffer(backend, contiguousGrad);
            using var tblB = GetOrAllocateBuffer(backend, contiguousTable);
            using var idxB = GetOrAllocateInt32IndexBuffer(backend, contiguousIndices);
            using var dwB = AllocateFullyWrittenOutputBuffer(backend, indices.Length);
            sr.TopKWeightedRowMixBackwardWeights(goB.Buffer, tblB.Buffer, idxB.Buffer, dwB.Buffer,
                rows, numEntries, k, channels);
            var result = DeferTensorResult<T>(backend, dwB.Buffer, indices.Length, (int[])indices._shape.Clone());
            dwB.RelinquishOwnership();
            return result;
        }
        catch (NotSupportedException capabilityGap)
        {
            GpuLaunchProbe.OnFallback("TopKWeightedRowMixBackwardWeights", capabilityGap);
            return base.TopKWeightedRowMixBackwardWeights(gradOutput, table, indices);
        }
    }

    public override Tensor<T> TopKWeightedRowMixBackwardTable<T>(
        Tensor<T> gradOutput, Tensor<int> indices, Tensor<T> weights, int numEntries)
    {
        using var __gpuOp = EnterGpuOp();
        if (typeof(T) != typeof(float) || !TryGetBackend(out var backend)
            || SparseRoutingResolver.Resolve(backend) is not { } sr
            || gradOutput is null || indices is null || weights is null || numEntries < 1
            || gradOutput.Rank < 1 || gradOutput.Rank != indices.Rank
            || !Enumerable.SequenceEqual(indices._shape, weights._shape)
            || gradOutput._shape[gradOutput.Rank - 1] < 1 || indices._shape[indices.Rank - 1] < 1
            || gradOutput.Length == 0 || indices.Length == 0)
            return base.TopKWeightedRowMixBackwardTable(gradOutput!, indices!, weights!, numEntries);

        for (int d = 0; d < gradOutput.Rank - 1; d++)
            if (gradOutput._shape[d] != indices._shape[d])
                return base.TopKWeightedRowMixBackwardTable(gradOutput, indices, weights, numEntries);

        int channels = gradOutput._shape[gradOutput.Rank - 1];
        int k = indices._shape[indices.Rank - 1];
        int rows = gradOutput.Length / channels;
        int outLen = checked(numEntries * channels);
        var contiguousGrad = gradOutput.IsContiguous ? gradOutput : gradOutput.Contiguous();
        var contiguousIndices = indices.IsContiguous ? indices : indices.Contiguous();
        var contiguousWeights = weights.IsContiguous ? weights : weights.Contiguous();
        try
        {
            using var goB = GetOrAllocateBuffer(backend, contiguousGrad);
            using var idxB = GetOrAllocateInt32IndexBuffer(backend, contiguousIndices);
            using var wB = GetOrAllocateBuffer(backend, contiguousWeights);
            using var dtB = AllocateFullyWrittenOutputBuffer(backend, outLen);
            sr.TopKWeightedRowMixBackwardTable(goB.Buffer, idxB.Buffer, wB.Buffer, dtB.Buffer,
                rows, numEntries, k, channels);
            var result = DeferTensorResult<T>(backend, dtB.Buffer, outLen, new[] { numEntries, channels });
            dtB.RelinquishOwnership();
            return result;
        }
        catch (NotSupportedException capabilityGap)
        {
            GpuLaunchProbe.OnFallback("TopKWeightedRowMixBackwardTable", capabilityGap);
            return base.TopKWeightedRowMixBackwardTable(gradOutput, indices, weights, numEntries);
        }
    }

    /// <summary>
    /// Maximum input length for float-backed GPU histogram counts: IEEE-754 binary32 represents every non-negative
    /// integer up to <c>2^24</c> (<c>16,777,216</c>) without rounding loss.
    /// </summary>
    private const int MaxExactFloatCount = 1 << 24;

    public override Tensor<T> TensorCountIndices<T>(Tensor<int> indices, int numBins)
    {
        using var __gpuOp = EnterGpuOp();
        if (Compilation.GraphMode.IsActive || (typeof(T) != typeof(float) && typeof(T) != typeof(int))
            || !TryGetBackend(out var backend) || SparseRoutingResolver.Resolve(backend) is not { } sr
            || indices is null || numBins <= 0 || indices.Length > MaxExactFloatCount)
            return base.TensorCountIndices<T>(indices!, numBins);

        if (indices.Length == 0)
            return DispatchDeferredGpuOp<T>(backend, numBins, new[] { numBins },
                output => backend.Fill(output, 0f, numBins));

        var contiguousIndices = indices.IsContiguous ? indices : indices.Contiguous();
        try
        {
            using var idxB = GetOrAllocateInt32IndexBuffer(backend, contiguousIndices);
            using var outB = AllocateFullyWrittenOutputBuffer(backend, numBins);
            sr.CountIndicesInt32(idxB.Buffer, outB.Buffer, contiguousIndices.Length, numBins);
            var result = DeferTensorResult<T>(backend, outB.Buffer, numBins, new[] { numBins });
            outB.RelinquishOwnership();
            return result;
        }
        catch (NotSupportedException capabilityGap)
        {
            GpuLaunchProbe.OnFallback("TensorCountIndices", capabilityGap);
            return base.TensorCountIndices<T>(indices, numBins);
        }
    }
}
