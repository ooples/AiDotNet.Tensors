using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines;

public partial class DirectGpuTensorEngine
{
    public override Tensor<T> XLstmScanForward<T>(Tensor<T> qProj, Tensor<T> kProj, Tensor<T> vProj,
        Tensor<T> iGate, Tensor<T> fGate, Tensor<T> oGate, int numHeads)
    {
        if (qProj is null || kProj is null || vProj is null || iGate is null || fGate is null || oGate is null)
            throw new ArgumentNullException(nameof(qProj));
        if (qProj.Rank != 3 || numHeads < 1 || qProj.Shape[2] < 1 || qProj.Shape[2] % numHeads != 0)
            throw new ArgumentException("Expected [batch, time, modelDim] and a positive head count dividing modelDim.");
        var inputs = new[] { qProj, kProj, vProj, iGate, fGate, oGate };
        int b = qProj.Shape[0], t = qProj.Shape[1], m = qProj.Shape[2], d = m / numHeads;
        for (int j = 1; j < inputs.Length; j++)
            if (inputs[j].Rank != 3 || inputs[j].Shape[0] != b || inputs[j].Shape[1] != t ||
                inputs[j].Shape[2] != (j < 3 ? m : numHeads))
                throw new ArgumentException("Incompatible xLSTM projection or gate shape.");
        // The base method records graph metadata only; replay dispatches back to this device implementation.
        if (GraphMode.IsActive)
            return base.XLstmScanForward(qProj, kProj, vProj, iGate, fGate, oGate, numHeads);
        if (!TryGetBackend(out var backend))
            throw new NotSupportedException("DirectGpuTensorEngine xLSTM requires an available GPU backend.");
        if (typeof(T) != typeof(float) || backend is not IFusedLstmSequenceTraining fused)
            throw new NotSupportedException($"{backend.GetType().Name} xLSTM requires float tensors and a fused kernel launcher; requested {typeof(T).Name}.");
        GradientTape<T>.Current?.BindEngineIfUnset(this);
        using var q = GetOrAllocateBuffer(backend, qProj);
        using var k = GetOrAllocateBuffer(backend, kProj);
        using var v = GetOrAllocateBuffer(backend, vProj);
        using var ig = GetOrAllocateBuffer(backend, iGate);
        using var fg = GetOrAllocateBuffer(backend, fGate);
        using var og = GetOrAllocateBuffer(backend, oGate);
        using var state = AllocateFullyWrittenOutputBuffer(backend, checked(b * numHeads * (t + 1) * (d * d + d)));
        using var y = AllocateFullyWrittenOutputBuffer(backend, qProj.Length);
        fused.StatefulLstmLauncher(BatchExecutionCudaKernels.XLstmTrainingForward, b * numHeads,
            q.Buffer, k.Buffer, v.Buffer, ig.Buffer, fg.Buffer, og.Buffer, y.Buffer, state.Buffer, b, t, m, numHeads);
        backend.Synchronize();
        var result = DeferTensorResult<T>(backend, y.Buffer, qProj.Length, qProj.Shape.ToArray());
        y.RelinquishOwnership();
        DifferentiableOps.RecordIfActive("XLstmScan", result, inputs, XLstmScanBackward<T>, savedState: new object[] { numHeads });
        return result;
    }

    internal void XLstmDeviceBackward<T>(Tensor<T> gradient, Tensor<T>[] inputs, int heads,
        Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        if (!TryGetBackend(out var backend))
            throw new NotSupportedException("DirectGpuTensorEngine xLSTM backward requires an available GPU backend.");
        if (typeof(T) != typeof(float) || backend is not IFusedLstmSequenceTraining fused)
            throw new NotSupportedException($"{backend.GetType().Name} xLSTM backward requires float tensors and a fused kernel launcher; requested {typeof(T).Name}.");
        int b = inputs[0].Shape[0], t = inputs[0].Shape[1], m = inputs[0].Shape[2], d = m / heads;
        int n = inputs[0].Length, ng = inputs[3].Length;
        using var q = GetOrAllocateBuffer(backend, inputs[0]);
        using var k = GetOrAllocateBuffer(backend, inputs[1]);
        using var v = GetOrAllocateBuffer(backend, inputs[2]);
        using var ig = GetOrAllocateBuffer(backend, inputs[3]);
        using var fg = GetOrAllocateBuffer(backend, inputs[4]);
        using var og = GetOrAllocateBuffer(backend, inputs[5]);
        using var dy = GetOrAllocateBuffer(backend, gradient);
        using var state = AllocateFullyWrittenOutputBuffer(backend, checked(b * heads * (t + 1) * (d * d + d)));
        using var adjoint = AllocateFullyWrittenOutputBuffer(backend, checked(b * heads * (d * d + d)));
        using var y = AllocateFullyWrittenOutputBuffer(backend, n);
        using var g = AllocateFullyWrittenOutputBuffer(backend, checked(3 * n + 3 * ng));
        // Recompute on device from the current replay inputs rather than retaining stale forward caches.
        fused.StatefulLstmLauncher(BatchExecutionCudaKernels.XLstmTrainingForward, b * heads,
            q.Buffer, k.Buffer, v.Buffer, ig.Buffer, fg.Buffer, og.Buffer, y.Buffer, state.Buffer, b, t, m, heads);
        fused.StatefulLstmLauncher(BatchExecutionCudaKernels.XLstmTrainingBackward, b * heads,
            q.Buffer, k.Buffer, v.Buffer, ig.Buffer, fg.Buffer, og.Buffer, dy.Buffer, state.Buffer,
            adjoint.Buffer, g.Buffer, b, t, m, heads);
        backend.Synchronize();
        var packet = DeferTensorResult<T>(backend, g.Buffer, 3 * n + 3 * ng, new[] { 3 * n + 3 * ng });
        g.RelinquishOwnership();
        for (int j = 0, offset = 0; j < inputs.Length; j++)
        {
            var value = Reshape(TensorNarrow(packet, 0, offset, inputs[j].Length), inputs[j].Shape.ToArray());
            DifferentiableOps.AccumulateGrad(grads, inputs[j], value, this);
            offset += inputs[j].Length;
        }
    }
}
