using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines;

/// <summary>
/// LogSumExp on the device, forward and backward. The reduced axis is moved last (a device permute when it is not
/// already), one <c>LogSumExpAxis</c> launch reduces each row, and under a tape the node records a backward that runs
/// the <c>LogSumExpBackward</c> kernel. Before this, a recording tape sent the whole op to the CPU composition
/// (max, broadcast subtract, exp, sum, log), so the input and every one of its gradients crossed to the host.
/// </summary>
public partial class DirectGpuTensorEngine
{
    Tensor<T> IEngine.TensorLogSumExp<T>(Tensor<T> tensor, int axis, bool keepDims)
    {
        using var __gpuOp = EnterGpuOp();
        // A graph trace records the CPU op's node; its replay re-enters here outside graph mode.
        if (tensor is null || Compilation.GraphMode.IsActive) return base.TensorLogSumExp(tensor!, axis, keepDims);
        int rank = tensor.Rank;
        int normalizedAxis = axis < 0 ? rank + axis : axis;
        if (rank == 0 || normalizedAxis < 0 || normalizedAxis >= rank)
            throw new ArgumentOutOfRangeException(nameof(axis), $"Axis {axis} out of range for rank {rank}");

        // One kernel per row, tape or not. The node is recorded on the kept-dims result, the shape its backward
        // broadcasts against; dropping the axis is a recorded reshape, as on the CPU engine.
        if (TryLogSumExpKeptOnDevice(tensor, normalizedAxis) is { } kept)
        {
            DifferentiableOps.RecordUnary("LogSumExp", kept, tensor, LogSumExpDeviceBackward<T>,
                new object[] { normalizedAxis });
            if (keepDims) return kept;
            var outShape = new int[rank - 1];
            for (int i = 0, j = 0; i < rank; i++) if (i != normalizedAxis) outShape[j++] = tensor._shape[i];
            return Reshape(kept, outShape);
        }

        // Not silent: a float tensor reaching the host here is a missing device path, which the fallback counters
        // report and a trace warning names. Other element types are host types on this engine.
        if (typeof(T) == typeof(float) && tensor.Length > 0)
        {
            string shape = string.Join(",", tensor._shape);
            if (ThrowOnGpuKernelFallback)
                throw new NotSupportedException($"TensorLogSumExp has no device path on this backend (shape [{shape}]).");
            GpuLaunchProbe.OnFallback("TensorLogSumExp: no device path", null);
            System.Diagnostics.Trace.TraceWarning(
                $"TensorLogSumExp ran on the CPU for a float tensor of shape [{shape}]: the GPU backend has no LogSumExpAxis kernel.");
        }
        return base.TensorLogSumExp(tensor, axis, keepDims);
    }

    // The rows with the reduced axis last, contiguous on the device (a device permute, which records nothing on a tape).
    private Tensor<T> LogSumExpRows<T>(Tensor<T> tensor, int axis)
    {
        var source = tensor.IsContiguous ? tensor : tensor.Contiguous();
        return axis == tensor.Rank - 1 ? source : PermuteContiguousGpu(source, AxisLastPermutation(tensor.Rank, axis));
    }

    private Tensor<T>? TryLogSumExpKeptOnDevice<T>(Tensor<T> tensor, int axis)
    {
        if (typeof(T) != typeof(float) || tensor.Length == 0 || !TryGetBatchBackend(out var batch)
            || batch is not IDirectGpuBackend backend)
            return null;
        int reduce = tensor._shape[axis];
        int outer = tensor.Length / reduce;
        var kept = (int[])tensor._shape.Clone();
        kept[axis] = 1;
        try
        {
            var rows = LogSumExpRows(tensor, axis);
            using var input = GetOrAllocateBuffer(backend, rows);
            return DispatchDeferredGpuOp<T>(backend, outer, kept,
                output => batch.LogSumExpAxis(input.Buffer, output, outer, reduce), fullyWritten: true);
        }
        catch (NotSupportedException ex)
        {
            if (ThrowOnGpuKernelFallback) throw;
            GpuLaunchProbe.OnFallback("TensorLogSumExp", ex);
            return null;
        }
    }

    // dx = dy * exp(x - lse), from the LogSumExpBackward kernel. An interior axis is moved last for the kernel and the
    // gradient moved back.
    private static void LogSumExpDeviceBackward<T>(Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        int axis = (int)savedState[0];
        if (engine.TryGet<DirectGpuTensorEngine>(out var gpu)
            && gpu.TryLogSumExpBackwardOnDevice(gradOutput, inputs[0], output, axis) is { } onDevice)
        {
            DifferentiableOps.AccumulateGrad(grads, inputs[0], onDevice, engine);
            return;
        }
        if (engine.Is<DirectGpuTensorEngine>()) GpuLaunchProbe.OnFallback("TensorLogSumExp backward: composed ops", null);
        BackwardFunctions<T>.LogSumExpBackward(gradOutput, inputs, output, savedState, engine, grads);
    }

    private Tensor<T>? TryLogSumExpBackwardOnDevice<T>(Tensor<T> gradOutput, Tensor<T> input, Tensor<T> lse, int axis)
    {
        using var __gpuOp = EnterGpuOp();
        if (typeof(T) != typeof(float) || input.Length == 0 || !TryGetBackend(out var backend)) return null;
        int rank = input.Rank;
        int reduce = input._shape[axis];
        int outer = input.Length / reduce;
        if (gradOutput.Length != outer || lse.Length != outer) return null;
        try
        {
            var rows = LogSumExpRows(input, axis);
            using var x = GetOrAllocateBuffer(backend, rows);
            using var dy = GetOrAllocateBuffer(backend, gradOutput.IsContiguous ? gradOutput : gradOutput.Contiguous());
            using var l = GetOrAllocateBuffer(backend, lse.IsContiguous ? lse : lse.Contiguous());
            var dx = DispatchDeferredGpuOp<T>(backend, rows.Length, (int[])rows._shape.Clone(),
                output => backend.LogSumExpBackward(dy.Buffer, x.Buffer, l.Buffer, output, outer, reduce), fullyWritten: true);
            if (axis == rank - 1) return dx;
            var inverse = InversePermutation(AxisLastPermutation(rank, axis));
            return PermuteContiguousGpu(dx, inverse);
        }
        catch (NotSupportedException ex)
        {
            if (ThrowOnGpuKernelFallback) throw;
            GpuLaunchProbe.OnFallback("TensorLogSumExp backward", ex);
            return null;
        }
    }
}
