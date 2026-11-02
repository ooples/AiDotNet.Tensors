using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines;

/// <summary>
/// The fused residual join, <c>activation(a + b)</c>, for any <see cref="IEngine"/>.
/// </summary>
public static class TensorAddActivationEngineExtensions
{
    /// <summary>
    /// <c>activation(a + b)</c>: the residual join of a ResNet block (the shortcut added to the branch, then ReLU) as one
    /// op, which the built-in engines compute in one pass instead of an add pass and an activation pass.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="engine">The engine that performs the operation.</param>
    /// <param name="a">The first tensor.</param>
    /// <param name="b">The second tensor (broadcast as <see cref="IEngine.TensorAdd{T}"/> broadcasts).</param>
    /// <param name="activation">The activation applied to the sum; <see cref="FusedActivationType.None"/> is a plain add.</param>
    /// <returns>The activated sum, equal to <c>TensorAdd</c> followed by the activation.</returns>
    public static Tensor<T> TensorAddActivation<T>(this IEngine engine, Tensor<T> a, Tensor<T> b, FusedActivationType activation)
    {
        if (engine is null)
            throw new ArgumentNullException(nameof(engine));
        if (a is null)
            throw new ArgumentNullException(nameof(a));
        if (b is null)
            throw new ArgumentNullException(nameof(b));

        // Every built-in engine derives from CpuEngine; virtual dispatch reaches the GPU kernel where available. Keeping
        // the primitive off IEngine avoids an ABI break for external engines compiled against an earlier package.
        if (engine is CpuEngine builtIn)
            return builtIn.TensorAddActivation(a, b, activation);

        // A third-party engine composes it from the IEngine contract.
        var sum = engine.TensorAdd(a, b);
        return activation switch
        {
            FusedActivationType.None => sum,
            FusedActivationType.ReLU => engine.ReLU(sum),
            FusedActivationType.Sigmoid => engine.Sigmoid(sum),
            FusedActivationType.GELU => engine.GELU(sum),
            FusedActivationType.Tanh => engine.Tanh(sum),
            _ => throw new NotSupportedException(
                $"TensorAddActivation on a third-party engine supports None, ReLU, Sigmoid, GELU and Tanh, not {activation}.")
        };
    }
}
