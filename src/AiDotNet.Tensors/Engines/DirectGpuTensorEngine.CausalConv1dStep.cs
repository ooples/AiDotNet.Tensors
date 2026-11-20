using System;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines;

public partial class DirectGpuTensorEngine
{
    /// <inheritdoc />
    public override Tensor<T> CausalConv1dStep<T>(Tensor<T> history, Tensor<T> input, Tensor<T> weight, Tensor<T>? bias, bool silu)
    {
        if (GraphMode.IsActive) return base.CausalConv1dStep(history, input, weight, bias, silu);
        var (taps, batch, channels) = ValidateCausalConv1dStep(history, input, weight, bias);
        if (typeof(T) != typeof(float) || !TryGetBackend(out var backend) || backend is not ICausalConv1dStepBackend step
            || taps == 0 || taps > ICausalConv1dStepBackend.Limits.MaxTaps || IsPartialView(history))
            return base.CausalConv1dStep(history, input, weight, bias, silu);
        using var __gpuOp = EnterGpuOp();
        // The history's own persistent buffer, holding its current value: the kernel shifts it in place.
        var historyBuffer = PinInputResident(history) ?? throw new InvalidOperationException("The history could not be made device-resident.");
        using var x = GetOrAllocateBuffer(backend, input);
        using var w = GetOrAllocateBuffer(backend, weight);
        using OwnedBuffer? b = bias is null ? null : GetOrAllocateBuffer(backend, bias);
        var output = AllocateFullyWrittenOutputBuffer(backend, batch * channels);   // every element written
        step.CausalConv1dStep(historyBuffer, x.Buffer, w.Buffer, b?.Buffer, output.Buffer, taps, batch, channels, silu);
        history.IncrementVersion();
        BindResidentBuffer(history, historyBuffer, backend);
        return DeferTensorResult<T>(backend, output.Buffer, batch * channels, new[] { batch, channels });
    }
}
