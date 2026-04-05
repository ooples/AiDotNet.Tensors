using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Compilation;

/// <summary>
/// GPU compiled plan that captures the fused execution as a CUDA graph for
/// zero kernel-launch overhead on subsequent replays.
///
/// Lifecycle:
/// 1. First execution: runs the compiled plan normally (captures CUDA graph)
/// 2. Subsequent executions: single cuGraphLaunch — zero per-kernel overhead
///
/// For non-CUDA backends (OpenCL, WebGPU): falls back to
/// DeferredScope + ExecutionGraph replay.
///
/// This is a placeholder for Phase 5 — actual CUDA graph integration requires
/// the GpuEngine infrastructure. The structural contract is established here
/// so CompiledInferencePlan/CompiledTrainingPlan can target it.
/// </summary>
internal sealed class GpuCompiledPlan<T> : IDisposable
{
    private readonly CompiledStep<T>[] _steps;
    private readonly Tensor<T> _finalOutput;
    private readonly IEngine _engine;
    private bool _captured;
    private bool _disposed;

    internal GpuCompiledPlan(CompiledStep<T>[] steps, Tensor<T> finalOutput, IEngine engine)
    {
        _steps = steps;
        _finalOutput = finalOutput;
        _engine = engine;
    }

    /// <summary>Whether a CUDA graph has been captured for zero-overhead replay.</summary>
    internal bool IsCaptured => _captured;

    /// <summary>
    /// Executes the plan. First call runs eagerly (and captures CUDA graph if available).
    /// Subsequent calls replay the captured graph.
    /// </summary>
    internal Tensor<T> Execute()
    {
        if (_captured)
        {
            // Phase 5+: cuGraphLaunch replay here
            // For now, fall back to sequential execution
        }

        // Sequential execution (used for first call + non-CUDA backends)
        var engine = _engine;
        var steps = _steps;
        for (int i = 0; i < steps.Length; i++)
        {
            steps[i].Execute(engine, steps[i].OutputBuffer);
        }

        _captured = true;
        return _finalOutput;
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        // Phase 5+: release CUDA graph resources here
    }
}
