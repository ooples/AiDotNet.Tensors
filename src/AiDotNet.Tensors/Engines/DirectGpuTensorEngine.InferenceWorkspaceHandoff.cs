using AiDotNet.Tensors.Engines.DirectGpu;

namespace AiDotNet.Tensors.Engines;

public partial class DirectGpuTensorEngine
{
    internal void ReturnIdleCompiledWorkspaceToAllocatorCache()
    {
        if (_compiledWorkspaceBroker is not { RetainedBytes: > 0 }
            || ResidentStepActive || InferenceCaptureActive || TrainingStep.InStep
            || Autodiff.GradientRecording.IsRecording<float>() || Compilation.GraphMode.IsActive
            || Gpu.DeferredScope.Current is not null || Gpu.AutocastScope.IsEnabled) return;
        using var operation = EnterGpuOp();
        if (!operation.Held || !TryGetBackend(out var backend)) return;
        lock (CompiledStepLock)
        {
            if (ResidentStepActive || InferenceCaptureActive || TrainingStep.InStep
                || backend is IGpuGraphCapture { SupportsGraphCapture: true } capture && capture.IsStreamCapturing()) return;
            _compiledWorkspaceBroker.ReturnIdleCellsToBackendCache(backend);
        }
    }
}
