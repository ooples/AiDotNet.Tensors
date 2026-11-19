using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines;

public partial class CpuEngine
{
    /// <inheritdoc/>
    public virtual Tensor<T> TensorWeightNorm<T>(Tensor<T> direction, Tensor<T> gain, double epsilon = 1e-24)
    {
        ValidateWeightNorm(direction, gain, epsilon);
        if (GraphMode.IsActive && GraphMode.Current is { } scope)
        {
            scope.BindEngineIfUnset(this);
            return scope.RecordBinary(LazyNodeType.Custom, "TensorWeightNorm", direction, gain, direction._shape,
                (engine, output) => engine.TensorWeightNormInto(output, direction, gain, epsilon),
                BackwardFunctions<T>.WeightNormBackward, new object[] { epsilon });
        }
        var result = TensorAllocator.RentUninitialized<T>(direction._shape);
        TensorWeightNormInto(result, direction, gain, epsilon);
        DifferentiableOps.RecordBinary("TensorWeightNorm", result, direction, gain,
            BackwardFunctions<T>.WeightNormBackward, new object[] { epsilon });
        return result;
    }
}
