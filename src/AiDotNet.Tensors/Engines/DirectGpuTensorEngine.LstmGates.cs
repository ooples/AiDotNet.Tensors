using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines;

public partial class DirectGpuTensorEngine
{
    /// <summary>Validates and records the cell through the shared wrapper; the packet dispatches to the device kernel.</summary>
    public override Tensor<T> LstmCellFromGates<T>(Tensor<T> gates, Tensor<T> previousCell, out Tensor<T> cell)
        => base.LstmCellFromGates(gates, previousCell, out cell);

    protected override Tensor<T> RunLstmGatePacket<T>(Tensor<T> gates, Tensor<T> previousCell)
    {
        if (typeof(T) != typeof(float) || !TryGetBackend(out var backend)
            || backend is not IFusedLstmSequenceTraining fused)
            return base.RunLstmGatePacket(gates, previousCell);
        int count = previousCell.Length, hidden = previousCell.Shape[previousCell.Rank - 1];
        using var pre = GetOrAllocateBuffer(backend, gates);
        using var old = GetOrAllocateBuffer(backend, previousCell);
        var output = AllocateFullyWrittenOutputBuffer(backend, checked(2 * count));
        try
        {
            fused.StatefulLstmLauncher(BatchExecutionCudaKernels.LstmGateCell, count,
                pre.Buffer, old.Buffer, output.Buffer, count, hidden);
            var result = DeferTensorResult<T>(backend, output.Buffer, 2 * count, new[] { 2 * count });
            output.RelinquishOwnership(); return result;
        }
        finally
        {
            pre.DisposeAfterQueuedUse(backend); old.DisposeAfterQueuedUse(backend);
            output.DisposeAfterQueuedUse(backend);
        }
    }

    internal bool TryLstmGateBackward<T>(Tensor<T> gradient, Tensor<T>[] inputs, Tensor<T> output,
        Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        if (typeof(T) != typeof(float) || !TryGetBackend(out var backend)
            || backend is not IFusedLstmSequenceTraining fused) return false;
        int count = inputs[1].Length, hidden = inputs[1].Shape[inputs[1].Rank - 1];
        using var pre = GetOrAllocateBuffer(backend, inputs[0]);
        using var old = GetOrAllocateBuffer(backend, inputs[1]);
        using var dy = GetOrAllocateBuffer(backend, gradient);
        using var y = GetOrAllocateBuffer(backend, output);
        var dg = AllocateFullyWrittenOutputBuffer(backend, inputs[0].Length);
        var dc = AllocateFullyWrittenOutputBuffer(backend, count);
        try
        {
            fused.StatefulLstmLauncher(BatchExecutionCudaKernels.LstmGateCellBackward, count,
                pre.Buffer, old.Buffer, y.Buffer, dy.Buffer, dg.Buffer, dc.Buffer, count, hidden);
            var gatesGrad = DeferTensorResult<T>(backend, dg.Buffer, inputs[0].Length, inputs[0].Shape.ToArray());
            dg.RelinquishOwnership();
            var cellGrad = DeferTensorResult<T>(backend, dc.Buffer, count, inputs[1].Shape.ToArray());
            dc.RelinquishOwnership();
            DifferentiableOps.AccumulateGrad(grads, inputs[0], gatesGrad, this);
            DifferentiableOps.AccumulateGrad(grads, inputs[1], cellGrad, this);
            return true;
        }
        finally
        {
            pre.DisposeAfterQueuedUse(backend); old.DisposeAfterQueuedUse(backend);
            dy.DisposeAfterQueuedUse(backend); y.DisposeAfterQueuedUse(backend);
            dg.DisposeAfterQueuedUse(backend); dc.DisposeAfterQueuedUse(backend);
        }
    }
}
