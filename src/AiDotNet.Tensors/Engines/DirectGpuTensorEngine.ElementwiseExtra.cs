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
        // Under graph capture the base op records the lazy node (its replay re-enters here, uncaptured, and runs the kernel).
        if (Compilation.GraphMode.IsActive) return base.TensorAcosh(tensor);
        try
        {
            var output = TryRunUnaryTensor(tensor, static (backend, input, output, size) => backend.Acosh(input, output, size));
            if (output != null)
            {
                Autodiff.DifferentiableOps.RecordUnary("TensorAcosh", output, tensor, Autodiff.BackwardFunctions<T>.AcoshBackward);
                return output;
            }
        }
        catch (System.Exception ex) { GpuLaunchProbe.OnFallback("TensorAcosh", ex); }
        return base.TensorAcosh(tensor);
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorAsinh<T>(Tensor<T> tensor)
    {
        // Under graph capture the base op records the lazy node (its replay re-enters here, uncaptured, and runs the kernel).
        if (Compilation.GraphMode.IsActive) return base.TensorAsinh(tensor);
        try
        {
            var output = TryRunUnaryTensor(tensor, static (backend, input, output, size) => backend.Asinh(input, output, size));
            if (output != null)
            {
                Autodiff.DifferentiableOps.RecordUnary("TensorAsinh", output, tensor, Autodiff.BackwardFunctions<T>.AsinhBackward);
                return output;
            }
        }
        catch (System.Exception ex) { GpuLaunchProbe.OnFallback("TensorAsinh", ex); }
        return base.TensorAsinh(tensor);
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorExp2<T>(Tensor<T> tensor)
    {
        // Under graph capture the base op records the lazy node (its replay re-enters here, uncaptured, and runs the kernel).
        if (Compilation.GraphMode.IsActive) return base.TensorExp2(tensor);
        try
        {
            var output = TryRunUnaryTensor(tensor, static (backend, input, output, size) => backend.Exp2(input, output, size));
            if (output != null)
            {
                Autodiff.DifferentiableOps.RecordUnary("TensorExp2", output, tensor, Autodiff.BackwardFunctions<T>.Exp2Backward);
                return output;
            }
        }
        catch (System.Exception ex) { GpuLaunchProbe.OnFallback("TensorExp2", ex); }
        return base.TensorExp2(tensor);
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorExpm1<T>(Tensor<T> tensor)
    {
        // Under graph capture the base op records the lazy node (its replay re-enters here, uncaptured, and runs the kernel).
        if (Compilation.GraphMode.IsActive) return base.TensorExpm1(tensor);
        try
        {
            var output = TryRunUnaryTensor(tensor, static (backend, input, output, size) => backend.ExpM1(input, output, size));
            if (output != null)
            {
                Autodiff.DifferentiableOps.RecordUnary("TensorExpm1", output, tensor, Autodiff.BackwardFunctions<T>.Expm1Backward);
                return output;
            }
        }
        catch (System.Exception ex) { GpuLaunchProbe.OnFallback("TensorExpm1", ex); }
        return base.TensorExpm1(tensor);
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorLog10<T>(Tensor<T> tensor)
    {
        // Under graph capture the base op records the lazy node (its replay re-enters here, uncaptured, and runs the kernel).
        if (Compilation.GraphMode.IsActive) return base.TensorLog10(tensor);
        try
        {
            var output = TryRunUnaryTensor(tensor, static (backend, input, output, size) => backend.Log10(input, output, size));
            if (output != null)
            {
                Autodiff.DifferentiableOps.RecordUnary("TensorLog10", output, tensor, Autodiff.BackwardFunctions<T>.Log10Backward);
                return output;
            }
        }
        catch (System.Exception ex) { GpuLaunchProbe.OnFallback("TensorLog10", ex); }
        return base.TensorLog10(tensor);
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorLog1p<T>(Tensor<T> tensor)
    {
        // Under graph capture the base op records the lazy node (its replay re-enters here, uncaptured, and runs the kernel).
        if (Compilation.GraphMode.IsActive) return base.TensorLog1p(tensor);
        try
        {
            var output = TryRunUnaryTensor(tensor, static (backend, input, output, size) => backend.Log1P(input, output, size));
            if (output != null)
            {
                Autodiff.DifferentiableOps.RecordUnary("TensorLog1p", output, tensor, Autodiff.BackwardFunctions<T>.Log1pBackward);
                return output;
            }
        }
        catch (System.Exception ex) { GpuLaunchProbe.OnFallback("TensorLog1p", ex); }
        return base.TensorLog1p(tensor);
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorLog2<T>(Tensor<T> tensor)
    {
        // Under graph capture the base op records the lazy node (its replay re-enters here, uncaptured, and runs the kernel).
        if (Compilation.GraphMode.IsActive) return base.TensorLog2(tensor);
        try
        {
            var output = TryRunUnaryTensor(tensor, static (backend, input, output, size) => backend.Log2(input, output, size));
            if (output != null)
            {
                Autodiff.DifferentiableOps.RecordUnary("TensorLog2", output, tensor, Autodiff.BackwardFunctions<T>.Log2Backward);
                return output;
            }
        }
        catch (System.Exception ex) { GpuLaunchProbe.OnFallback("TensorLog2", ex); }
        return base.TensorLog2(tensor);
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorRsqrt<T>(Tensor<T> tensor)
    {
        // Under graph capture the base op records the lazy node (its replay re-enters here, uncaptured, and runs the kernel).
        if (Compilation.GraphMode.IsActive) return base.TensorRsqrt(tensor);
        try
        {
            var output = TryRunUnaryTensor(tensor, static (backend, input, output, size) =>
            {
                // The fused rsqrt kernel is on the batch-execution surface (Vulkan); every backend has sqrt and
                // reciprocal, so the others stay on the device with two kernels instead of falling back to the host.
                if (backend is IGpuBatchExecution batch) batch.RsqrtKernel(input, output, size);
                else
                {
                    backend.Sqrt(input, output, size);
                    backend.Reciprocal(output, output, size);
                }
            });
            if (output != null)
            {
                Autodiff.DifferentiableOps.RecordUnary("TensorRsqrt", output, tensor, Autodiff.BackwardFunctions<T>.RsqrtBackward);
                return output;
            }
        }
        catch (System.Exception ex) { GpuLaunchProbe.OnFallback("TensorRsqrt", ex); }
        return base.TensorRsqrt(tensor);
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorTan<T>(Tensor<T> tensor)
    {
        // Under graph capture the base op records the lazy node (its replay re-enters here, uncaptured, and runs the kernel).
        if (Compilation.GraphMode.IsActive) return base.TensorTan(tensor);
        try
        {
            var output = TryRunUnaryTensor(tensor, static (backend, input, output, size) => backend.Tan(input, output, size));
            if (output != null)
            {
                Autodiff.DifferentiableOps.RecordUnary("TensorTan", output, tensor, Autodiff.BackwardFunctions<T>.TanBackward);
                return output;
            }
        }
        catch (System.Exception ex) { GpuLaunchProbe.OnFallback("TensorTan", ex); }
        return base.TensorTan(tensor);
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorTrunc<T>(Tensor<T> tensor)
    {
        // Under graph capture the base op records the lazy node (its replay re-enters here, uncaptured, and runs the kernel).
        if (Compilation.GraphMode.IsActive) return base.TensorTrunc(tensor);
        try
        {
            var output = TryRunUnaryTensor(tensor, static (backend, input, output, size) => backend.Truncate(input, output, size));
            if (output != null)
            {
                Autodiff.DifferentiableOps.RecordUnary("TensorTrunc", output, tensor, Autodiff.BackwardFunctions<T>.TruncBackward);
                return output;
            }
        }
        catch (System.Exception ex) { GpuLaunchProbe.OnFallback("TensorTrunc", ex); }
        return base.TensorTrunc(tensor);
    }
}
