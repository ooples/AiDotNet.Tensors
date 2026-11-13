namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>
/// This thread's training-step boundaries: an outermost <see cref="Autodiff.GradientTape{T}"/> begins a step and its
/// dispose ends it. A backend's step arena (<see cref="CUDA.CudaStepArena"/>) records one step's allocations and
/// replays the rest from a packed plan.
/// </summary>
internal static class TrainingStep
{
    [System.ThreadStatic] private static int t_step;
    [System.ThreadStatic] private static bool t_inStep;

    /// <summary>The number of steps this thread has begun.</summary>
    internal static int Current => t_step;

    /// <summary>Whether this thread is inside a step.</summary>
    internal static bool InStep => t_inStep;

    internal static void Begin()
    {
        t_step++;
        t_inStep = true;
    }

    internal static void End() => t_inStep = false;
}
