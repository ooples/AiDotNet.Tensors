namespace AiDotNet.Tensors.Engines.Compilation;

/// <summary>
/// A training plan whose global-norm clip (<see cref="ICompiledTrainingPlan{T}.SetMaxGradNorm"/>) takes a
/// <see cref="Gpu.NonFiniteGradientPolicy"/>. Every plan <see cref="GraphMode"/> compiles implements it; it is a
/// separate interface so <see cref="ICompiledTrainingPlan{T}"/> keeps its shipped shape.
/// </summary>
public interface INonFiniteGradientPolicyTarget
{
    /// <summary>
    /// What the clip does when a gradient element is NaN or infinite: leave the gradients unscaled (the default),
    /// propagate PyTorch's NaN/0 coefficient, zero only the NaN/Inf elements and clip by the rest, or throw a
    /// <see cref="Gpu.NonFiniteGradientException"/> from the step naming the first offending gradient (PyTorch's
    /// <c>error_if_nonfinite</c>).
    /// </summary>
    void SetNonFiniteGradientPolicy(Gpu.NonFiniteGradientPolicy policy);
}
