using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Engines.DirectGpu;

namespace AiDotNet.Tensors.Engines;

public partial class DirectGpuTensorEngine
{
    // Called from an owner-aware cache miss, which may already be inside a backend context scope.
    // No plan storage changes or CompiledStepLock: this releases only unborrowed native allocations already in a
    // queue-affine free cache. Acquiring a plan lock here would invert the engine->backend allocation lock order.
    internal bool IsInferenceIdleCacheDemandEligible(IDirectGpuBackend backend)
    {
        if (GetType() != typeof(DirectGpuTensorEngine) || ResidentStepActive || InferenceCaptureActive
            || ScratchPoolingActive || TrainingStep.InStep || GradientRecording.IsRecording<float>()
            || GraphMode.IsActive || Gpu.DeferredScope.Current is not null || Gpu.GpuBufferReleaseDeferral.IsActive
            || Gpu.AutocastScope.IsEnabled) return false;
        using var operation = EnterGpuOp();
        if (!operation.Held || !TryGetBackend(out var current) || !ReferenceEquals(current, backend)) return false;
        return backend is not IGpuGraphCapture { SupportsGraphCapture: true } capture || !capture.IsStreamCapturing();
    }
}
