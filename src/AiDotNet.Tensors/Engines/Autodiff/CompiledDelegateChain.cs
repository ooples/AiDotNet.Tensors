using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Autodiff;

/// <summary>
/// Pre-compiled delegate chain for backward execution. Captures the exact
/// sequence of backward function calls from the first backward pass and
/// replays them on subsequent passes — no tape walking, no graph traversal,
/// no dictionary lookups. Just straight-line delegate invocations.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
internal sealed class CompiledDelegateChain<T>
{
    private readonly BackwardStep<T>[] _steps;

    internal CompiledDelegateChain(BackwardStep<T>[] steps)
    {
        _steps = steps;
    }

    /// <summary>
    /// Executes the pre-compiled backward chain. Each step is a captured closure
    /// that calls the backward function with the right inputs and accumulates gradients.
    /// </summary>
    internal Dictionary<Tensor<T>, Tensor<T>> Execute(
        Tensor<T> loss,
        IReadOnlyList<Tensor<T>>? sources,
        IEngine engine)
    {
        var grads = new Dictionary<Tensor<T>, Tensor<T>>(
            _steps.Length + 1,
            ReferenceEqualityComparer<Tensor<T>>.Instance);

        // Seed
        var numOps = Helpers.MathHelper.GetNumericOperations<T>();
        Tensor<T> seedGrad;
        if (loss.Length == 1)
            seedGrad = new Tensor<T>(new[] { numOps.One }, new[] { 1 });
        else
        {
            var onesData = new T[loss.Length];
            var one = numOps.One;
            for (int j = 0; j < onesData.Length; j++) onesData[j] = one;
            seedGrad = new Tensor<T>(onesData, loss._shape);
        }
        grads[loss] = seedGrad;

        // Replay chain — straight-line delegate calls, no traversal
        for (int i = 0; i < _steps.Length; i++)
        {
            ref var step = ref _steps[i];
            if (!grads.TryGetValue(step.Output, out var gradOutput))
                continue;

            step.Backward(gradOutput, step.Inputs, step.Output,
                step.SavedState ?? Array.Empty<object>(), engine, grads);
        }

        if (sources is not null)
        {
            var filtered = new Dictionary<Tensor<T>, Tensor<T>>(ReferenceEqualityComparer<Tensor<T>>.Instance);
            foreach (var source in sources)
                if (grads.TryGetValue(source, out var grad))
                    filtered[source] = grad;
            return filtered;
        }

        return grads;
    }
}

/// <summary>A single step in the compiled backward chain.</summary>
internal struct BackwardStep<T>
{
    public Tensor<T> Output;
    public Tensor<T>[] Inputs;
    public BackwardFunction<T> Backward;
    public object[]? SavedState;
}
