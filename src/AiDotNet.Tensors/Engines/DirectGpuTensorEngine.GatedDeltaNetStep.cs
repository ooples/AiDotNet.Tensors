using System;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines;

public partial class DirectGpuTensorEngine
{
    /// <inheritdoc />
    public override Tensor<T> GatedDeltaNetStep<T>(Tensor<T> state, Tensor<T> query, Tensor<T> key, Tensor<T> value,
        Tensor<T> alpha, Tensor<T> beta)
    {
        if (GraphMode.IsActive) return base.GatedDeltaNetStep(state, query, key, value, alpha, beta);
        var (rows, dv, dk) = ValidateGatedDeltaNetStep(state, query, key, value, alpha, beta);
        if (typeof(T) != typeof(float) || IsPartialView(state))
            return base.GatedDeltaNetStep(state, query, key, value, alpha, beta);
        // The lease comes before the backend lookup: strict lease checking asserts it on TryGetBackend.
        using var __gpuOp = EnterGpuOp();
        if (!TryGetBackend(out var backend) || backend is not IGatedDeltaNetStepBackend step)
            return base.GatedDeltaNetStep(state, query, key, value, alpha, beta);
        // The state's own persistent buffer, holding its current value: the kernel writes it in place.
        var stateBuffer = PinInputResident(state) ?? throw new InvalidOperationException("The state could not be made device-resident.");
        using var q = GetOrAllocateBuffer(backend, query);
        using var k = GetOrAllocateBuffer(backend, key);
        using var v = GetOrAllocateBuffer(backend, value);
        using var a = GetOrAllocateBuffer(backend, alpha);
        using var b = GetOrAllocateBuffer(backend, beta);
        var output = AllocateFullyWrittenOutputBuffer(backend, rows * dv);   // every element written
        step.GatedDeltaNetStep(stateBuffer, q.Buffer, k.Buffer, v.Buffer, a.Buffer, b.Buffer, output.Buffer, rows, dv, dk);
        state.IncrementVersion();
        BindResidentBuffer(state, stateBuffer, backend);
        return DeferTensorResult<T>(backend, output.Buffer, rows * dv, new[] { rows, dv });
    }
}
