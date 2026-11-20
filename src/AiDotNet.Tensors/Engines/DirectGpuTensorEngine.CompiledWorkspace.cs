using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines;

public partial class DirectGpuTensorEngine
{
    private CompiledTransientWorkspaceBroker? _compiledWorkspaceBroker;
    internal CompiledTransientWorkspaceBroker CompiledWorkspaceBroker =>
        _compiledWorkspaceBroker ??= new CompiledTransientWorkspaceBroker(this);

    /// <summary>Step scratch this engine's compiled plans co-own (see <see cref="SharedStepScratch"/>).</summary>
    internal SharedStepScratch SharedStepScratch { get; } = new();

    // Only called for private, dead compiled intermediates while their scratch owner still owns the native buffer.
    // Reuse never materializes host data; a stale internal read fails rather than reading another plan's new value.
    internal void DetachCompiledWorkspaceTensor<T>(Tensor<T> tensor)
    {
        var array = tensor.GetBackingArrayForCacheLookupUnsafe();
        HostSync.Release(tensor.DataVector, ReleasedIntermediateMessage);
        if (array is not null) HostSync.Release(array, ReleasedIntermediateMessage);
        if (tensor._gpuMaterializerKey is { } key) HostSync.Release(key, ReleasedIntermediateMessage);
        tensor._gpuMaterializerKey = null;
        // Retain the reusable callback object; a later BindResidentBuffer rebinds it and registers the current key.
        if (array is not null) InvalidateActivationCacheEntry(array);
        InvalidateActivationCacheEntry(tensor.DataVector);
        tensor._gpuBuffer = null;
        tensor._gpuBackend = null;
        tensor._gpuBufferVersion = -1;
    }
}
