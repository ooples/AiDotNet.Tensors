using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines;

public partial class CpuEngine : IRfftMagnitudeEngine
{
    /// <inheritdoc/>
    public virtual Tensor<T> TensorRfftMagnitude<T>(Tensor<T> input, double epsilon = 1e-9)
    {
        var (_, _, bins) = ValidateRfftMagnitude(input, null, epsilon);
        int[] shape = (int[])input._shape.Clone();
        shape[shape.Length - 1] = bins;
        if (GraphMode.IsActive && GraphMode.Current is { } scope)
        {
            scope.BindEngineIfUnset(this);
            return scope.RecordUnary(LazyNodeType.Custom, "TensorRfftMagnitude", input, shape,
                (engine, output) => engine.TensorRfftMagnitudeInto(output, input, epsilon),
                BackwardFunctions<T>.RfftMagnitudeBackward, new object[] { epsilon });
        }
        var result = TensorAllocator.RentUninitialized<T>(shape);
        TensorRfftMagnitudeInto(result, input, epsilon);
        DifferentiableOps.RecordUnary("TensorRfftMagnitude", result, input,
            BackwardFunctions<T>.RfftMagnitudeBackward, new object[] { epsilon });
        return result;
    }

    /// <inheritdoc/>
    public virtual void TensorRfftMagnitudeInto<T>(Tensor<T> output, Tensor<T> input, double epsilon = 1e-9)
        => RfftMagnitudeIntoCore(output, input, epsilon);

    /// <inheritdoc/>
    public virtual void TensorRfftMagnitudeBackwardInto<T>(Tensor<T> gradOutput, Tensor<T> input, Tensor<T> gradInput,
        double epsilon = 1e-9, bool accumulate = false)
        => RfftMagnitudeBackwardIntoCore(gradOutput, input, gradInput, epsilon, accumulate);
}
