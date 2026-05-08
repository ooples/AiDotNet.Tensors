using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Compilation;

/// <summary>
/// Allocation-conscious input helpers for the common single-input compiled-plan case.
/// </summary>
public static class CompiledPlanInputExtensions
{
    /// <summary>
    /// Copies a single input tensor into a single-input inference plan without allocating
    /// the <c>Tensor&lt;T&gt;[1]</c> wrapper required by <see cref="ICompiledPlan{T}.SetInputs"/>.
    /// </summary>
    public static void SetInput<T>(this ICompiledPlan<T> plan, Tensor<T> input)
    {
        if (plan is null) throw new ArgumentNullException(nameof(plan));
        if (plan is CompiledInferencePlan<T> builtIn)
        {
            builtIn.SetInput(input);
            return;
        }

        plan.SetInputs(new[] { input });
    }

    /// <summary>
    /// Copies a single input tensor into a single-input training plan without allocating
    /// the <c>Tensor&lt;T&gt;[1]</c> wrapper required by <see cref="ICompiledTrainingPlan{T}.SetInputs"/>.
    /// </summary>
    public static void SetInput<T>(this ICompiledTrainingPlan<T> plan, Tensor<T> input)
    {
        if (plan is null) throw new ArgumentNullException(nameof(plan));
        if (plan is CompiledTrainingPlan<T> builtIn)
        {
            builtIn.SetInput(input);
            return;
        }

        plan.SetInputs(new[] { input });
    }
}
