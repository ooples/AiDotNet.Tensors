using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines;

/// <summary>
/// GPU kernels for the tensor-level unary maths of <see cref="CpuEngine"/>'s ElementwiseExtra file. Every backend
/// already had these kernels (behind the span and vector overloads); the tensor ops dispatch to them and record the same
/// backward as the CPU path, falling back to it when the GPU path is unavailable.
/// </summary>
public partial class DirectGpuTensorEngine
{
    /// <inheritdoc/>
    public override Tensor<T> TensorAcosh<T>(Tensor<T> tensor)
    {
        try
        {
            var output = TryRunUnaryTensor(tensor, static (backend, input, output, size) => backend.Acosh(input, output, size));
            if (output != null)
            {
                Autodiff.DifferentiableOps.RecordUnary("TensorAcosh", output, tensor, Autodiff.BackwardFunctions<T>.AcoshBackward);
                return output;
            }
        }
        catch { }
        return base.TensorAcosh(tensor);
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorAsinh<T>(Tensor<T> tensor)
    {
        try
        {
            var output = TryRunUnaryTensor(tensor, static (backend, input, output, size) => backend.Asinh(input, output, size));
            if (output != null)
            {
                Autodiff.DifferentiableOps.RecordUnary("TensorAsinh", output, tensor, Autodiff.BackwardFunctions<T>.AsinhBackward);
                return output;
            }
        }
        catch { }
        return base.TensorAsinh(tensor);
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorExp2<T>(Tensor<T> tensor)
    {
        try
        {
            var output = TryRunUnaryTensor(tensor, static (backend, input, output, size) => backend.Exp2(input, output, size));
            if (output != null)
            {
                Autodiff.DifferentiableOps.RecordUnary("TensorExp2", output, tensor, Autodiff.BackwardFunctions<T>.Exp2Backward);
                return output;
            }
        }
        catch { }
        return base.TensorExp2(tensor);
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorExpm1<T>(Tensor<T> tensor)
    {
        try
        {
            var output = TryRunUnaryTensor(tensor, static (backend, input, output, size) => backend.ExpM1(input, output, size));
            if (output != null)
            {
                Autodiff.DifferentiableOps.RecordUnary("TensorExpm1", output, tensor, Autodiff.BackwardFunctions<T>.Expm1Backward);
                return output;
            }
        }
        catch { }
        return base.TensorExpm1(tensor);
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorLog10<T>(Tensor<T> tensor)
    {
        try
        {
            var output = TryRunUnaryTensor(tensor, static (backend, input, output, size) => backend.Log10(input, output, size));
            if (output != null)
            {
                Autodiff.DifferentiableOps.RecordUnary("TensorLog10", output, tensor, Autodiff.BackwardFunctions<T>.Log10Backward);
                return output;
            }
        }
        catch { }
        return base.TensorLog10(tensor);
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorLog1p<T>(Tensor<T> tensor)
    {
        try
        {
            var output = TryRunUnaryTensor(tensor, static (backend, input, output, size) => backend.Log1P(input, output, size));
            if (output != null)
            {
                Autodiff.DifferentiableOps.RecordUnary("TensorLog1p", output, tensor, Autodiff.BackwardFunctions<T>.Log1pBackward);
                return output;
            }
        }
        catch { }
        return base.TensorLog1p(tensor);
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorLog2<T>(Tensor<T> tensor)
    {
        try
        {
            var output = TryRunUnaryTensor(tensor, static (backend, input, output, size) => backend.Log2(input, output, size));
            if (output != null)
            {
                Autodiff.DifferentiableOps.RecordUnary("TensorLog2", output, tensor, Autodiff.BackwardFunctions<T>.Log2Backward);
                return output;
            }
        }
        catch { }
        return base.TensorLog2(tensor);
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorRsqrt<T>(Tensor<T> tensor)
    {
        try
        {
            var output = TryRunUnaryTensor(tensor, static (backend, input, output, size) =>
            {
                // The rsqrt kernel is part of the batch-execution backend surface; other backends take the CPU path.
                if (backend is IGpuBatchExecution batch) batch.RsqrtKernel(input, output, size);
                else throw new System.NotSupportedException("This backend has no rsqrt kernel.");
            });
            if (output != null)
            {
                Autodiff.DifferentiableOps.RecordUnary("TensorRsqrt", output, tensor, Autodiff.BackwardFunctions<T>.RsqrtBackward);
                return output;
            }
        }
        catch { }
        return base.TensorRsqrt(tensor);
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorTan<T>(Tensor<T> tensor)
    {
        try
        {
            var output = TryRunUnaryTensor(tensor, static (backend, input, output, size) => backend.Tan(input, output, size));
            if (output != null)
            {
                Autodiff.DifferentiableOps.RecordUnary("TensorTan", output, tensor, Autodiff.BackwardFunctions<T>.TanBackward);
                return output;
            }
        }
        catch { }
        return base.TensorTan(tensor);
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorTrunc<T>(Tensor<T> tensor)
    {
        try
        {
            var output = TryRunUnaryTensor(tensor, static (backend, input, output, size) => backend.Truncate(input, output, size));
            if (output != null)
            {
                Autodiff.DifferentiableOps.RecordUnary("TensorTrunc", output, tensor, Autodiff.BackwardFunctions<T>.TruncBackward);
                return output;
            }
        }
        catch { }
        return base.TensorTrunc(tensor);
    }
}
