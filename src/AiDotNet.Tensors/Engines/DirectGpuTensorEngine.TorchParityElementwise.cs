using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines;

/// <summary>
/// Device paths for PyTorch-parity elementwise ops that computed on the host (#1112). Under an active tape or graph
/// recording they defer to the CPU implementation, which records the backward (the convention of the engine's other
/// parity overrides); so do non-float precisions, mismatched binary shapes and backends without the kernels.
/// </summary>
public partial class DirectGpuTensorEngine
{
    private Tensor<T>? TryParityUnaryOnDevice<T>(Tensor<T> input, TorchParityUnaryOp op, float p0, float p1)
    {
        if (input.Length == 0 || IsTapeActive<T>(op.ToString()) || !TryGetBackend(out var backend)
            || backend is not ITorchParityElementwiseBackend kernels || DirectGpuEngine.ShouldFallbackForPrecision<T>())
            return null;
        try
        {
            int n = input.Length;
            using var x = GetOrAllocateBuffer(backend, input);
            return DispatchDeferredGpuOp<T>(backend, n, (int[])input._shape.Clone(),
                output => kernels.TorchParityUnary(op, x.Buffer, output, n, p0, p1));
        }
        catch (System.Exception ex)
        {
            GpuLaunchProbe.OnFallback("Tensor" + op, ex);
            return null;
        }
    }

    private Tensor<T>? TryParityBinaryOnDevice<T>(Tensor<T> a, Tensor<T> b, TorchParityBinaryOp op)
    {
        if (a.Length == 0 || !a._shape.AsSpan().SequenceEqual(b._shape) || IsTapeActive<T>(op.ToString())
            || !TryGetBackend(out var backend) || backend is not ITorchParityElementwiseBackend kernels
            || DirectGpuEngine.ShouldFallbackForPrecision<T>())
            return null;
        try
        {
            int n = a.Length;
            using var x = GetOrAllocateBuffer(backend, a);
            using var y = GetOrAllocateBuffer(backend, b);
            return DispatchDeferredGpuOp<T>(backend, n, (int[])a._shape.Clone(),
                output => kernels.TorchParityBinary(op, x.Buffer, y.Buffer, output, n));
        }
        catch (System.Exception ex)
        {
            GpuLaunchProbe.OnFallback("Tensor" + op, ex);
            return null;
        }
    }
}
