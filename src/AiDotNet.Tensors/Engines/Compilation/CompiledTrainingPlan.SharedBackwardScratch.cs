using System;
using AiDotNet.Tensors.Engines.DirectGpu;

namespace AiDotNet.Tensors.Engines.Compilation;

internal sealed partial class CompiledTrainingPlan<T>
{
    private readonly bool _shareBackwardScratch =
        Environment.GetEnvironmentVariable("AIDOTNET_COMPILED_SHARED_BACKWARD_SCRATCH") == "1";
    private readonly bool _handoffCompletedScratch =
        Environment.GetEnvironmentVariable("AIDOTNET_COMPILED_BACKEND_SCRATCH_HANDOFF") == "1";

    private void ReturnCompletedBackwardScratch(DirectGpuTensorEngine engine, IDirectGpuBackend backend)
    {
        if (!_shareBackwardScratch || !_workspaceSharingBodyAllowed || !CanShareTransientWorkspace(engine)
            || _scratchPool is not { } pool) return;
        if (pool.ReturnIdleBackwardScratch(backend, engine.CompiledWorkspaceBroker) > 0)
            _lendSchedule = null;
        if (_handoffCompletedScratch)
            engine.CompiledWorkspaceBroker.ReturnIdleCellsToBackendCache(backend);
    }
}
