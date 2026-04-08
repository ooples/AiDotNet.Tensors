using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Compilation;

/// <summary>
/// A compiled inference plan that replays a pre-optimized computation graph.
/// Compile once from a traced forward pass, replay forever with zero allocation.
///
/// Plans are cached by input shape — recompilation occurs automatically
/// when input dimensions change.
/// </summary>
/// <typeparam name="T">The tensor element type.</typeparam>
public interface ICompiledPlan<T> : IDisposable
{
    /// <summary>Executes the compiled plan and returns the output tensor.</summary>
    Tensor<T> Execute();

    /// <summary>Checks whether this plan is valid for the given input shape.</summary>
    bool IsValid(int[] inputShape);

    /// <summary>Number of compiled execution steps.</summary>
    int StepCount { get; }
}

/// <summary>
/// A compiled training plan that replays forward + backward passes
/// with pre-allocated gradient buffers. Compile once, train forever.
///
/// Call Step() each training iteration. Gradients are available via
/// the Gradients property after each Step().
/// </summary>
/// <typeparam name="T">The tensor element type.</typeparam>
public interface ICompiledTrainingPlan<T> : IDisposable
{
    /// <summary>
    /// Executes one training step: forward pass, gradient computation, backward pass.
    /// Returns the loss tensor.
    /// </summary>
    Tensor<T> Step();

    /// <summary>
    /// Gradient tensors for each parameter, in the same order as the parameters
    /// passed to compilation. Updated after each Step() call.
    /// </summary>
    Tensor<T>[] Gradients { get; }

    /// <summary>Number of forward execution steps.</summary>
    int ForwardStepCount { get; }

    /// <summary>Number of backward execution steps.</summary>
    int BackwardStepCount { get; }
}
