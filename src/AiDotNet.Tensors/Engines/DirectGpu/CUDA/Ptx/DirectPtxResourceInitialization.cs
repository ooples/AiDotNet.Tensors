using System;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// Transfers ownership of a newly acquired native resource only after its remaining
/// initialization succeeds. Constructor failures dispose the resource before propagating the
/// original exception, so rejected kernel probes cannot leak loaded driver modules.
/// </summary>
internal static class DirectPtxResourceInitialization
{
    internal static (TResource Resource, TValue Value) Complete<TResource, TValue>(
        TResource resource, Func<TResource, TValue> initialize)
        where TResource : IDisposable
    {
        if (resource is null) throw new ArgumentNullException(nameof(resource));
        if (initialize is null)
        {
            resource.Dispose();
            throw new ArgumentNullException(nameof(initialize));
        }

        try
        {
            return (resource, initialize(resource));
        }
        catch
        {
            // A cleanup failure must not replace the validation/JIT/lookup exception that
            // explains why construction failed. DirectPtxModule.Dispose normally succeeds;
            // this guard preserves the primary failure if the driver is already unhealthy.
            try { resource.Dispose(); }
            catch { }
            throw;
        }
    }
}
