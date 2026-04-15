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

    /// <summary>
    /// Configures fused optimizer updates that run after each Step().
    /// Once configured, Step() will automatically update parameters using
    /// the specified optimizer — no manual gradient application needed.
    /// </summary>
    /// <param name="optimizerType">The optimizer algorithm (SGD, Adam, etc.).</param>
    /// <param name="learningRate">Learning rate.</param>
    /// <param name="beta1">First moment decay (Adam/AdamW). Default: 0.9.</param>
    /// <param name="beta2">Second moment decay (Adam/AdamW). Default: 0.999.</param>
    /// <param name="eps">Epsilon for numerical stability. Default: 1e-8.</param>
    /// <param name="weightDecay">Weight decay (AdamW/LAMB). Default: 0.</param>
    void ConfigureOptimizer(
        OptimizerType optimizerType,
        float learningRate,
        float beta1 = 0.9f,
        float beta2 = 0.999f,
        float eps = 1e-8f,
        float weightDecay = 0f);

    /// <summary>
    /// Enables gradient checkpointing for this plan, reducing activation memory from
    /// O(N) to O(sqrt(N)) at the cost of ~33% more compute (each segment's forward
    /// runs twice during backward). Call once after compilation, before the training loop.
    /// </summary>
    /// <param name="segmentSize">Steps per checkpoint segment. 0 = auto (sqrt(N)).</param>
    /// <remarks>
    /// The implementation is idempotent per plan — the most recent call wins. The
    /// checkpointing system wraps the forward actions; gradients remain numerically
    /// equivalent to the non-checkpointed path within floating-point tolerance.
    /// </remarks>
    void EnableCheckpointing(int segmentSize = 0);
}
