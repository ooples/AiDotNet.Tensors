using System;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// Transfers ownership of a newly acquired native resource only after its remaining
/// initialization succeeds. Constructor failures dispose the resource before propagating the
/// original exception, so rejected kernel probes cannot leak loaded driver modules.
/// </summary>
internal static class DirectPtxResourceInitialization
{
    internal static DirectPtxLoadedKernel LoadKernel(
        DirectPtxRuntime runtime,
        string ptx,
        string entryPoint,
        int blockThreads,
        DirectPtxKernelBlueprint blueprint)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        PtxCompat.ThrowIfNullOrWhiteSpace(ptx, nameof(ptx));
        PtxCompat.ThrowIfNullOrWhiteSpace(entryPoint, nameof(entryPoint));
        PtxCompat.ThrowIfNull(blueprint, nameof(blueprint));
        if (blockThreads <= 0)
            throw new ArgumentOutOfRangeException(nameof(blockThreads));

        var loaded = Complete(
            runtime.LoadModule(ptx),
            module =>
            {
                IntPtr function = module.GetFunction(
                    entryPoint, out DirectPtxFunctionInfo info);
                int activeBlocks = module.GetActiveBlocksPerMultiprocessor(
                    function, blockThreads);
                blueprint.ResourceBudget.Validate(
                    entryPoint, info, blockThreads, activeBlocks);
                DirectPtxKernelAudit audit = DirectPtxKernelAudit.Create(
                    blueprint, runtime.DeviceFingerprint, ptx, info,
                    blockThreads, activeBlocks, module);
                return new DirectPtxLoadedKernelValue(function, audit);
            });

        return new DirectPtxLoadedKernel(
            loaded.Resource, loaded.Value.Function, loaded.Value.Audit);
    }

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
            DisposeWithoutMaskingFailure(resource);
            throw;
        }
    }

    /// <summary>
    /// Releases a partially initialized resource on a construction failure path.
    /// </summary>
    /// <remarks>
    /// A cleanup failure must not replace the validation/JIT/lookup exception that
    /// explains why construction failed. DirectPtxModule.Dispose normally succeeds;
    /// this guard preserves the primary failure if the driver is already unhealthy.
    /// Constructors that own their module directly call this from their catch block,
    /// so the policy has exactly one implementation.
    /// </remarks>
    internal static void DisposeWithoutMaskingFailure(IDisposable resource)
    {
        if (resource is null) return;
        try { resource.Dispose(); }
        catch { }
    }

    private readonly struct DirectPtxLoadedKernelValue
    {
        internal IntPtr Function { get; }
        internal DirectPtxKernelAudit Audit { get; }

        internal DirectPtxLoadedKernelValue(
            IntPtr function, DirectPtxKernelAudit audit)
        {
            Function = function;
            Audit = audit;
        }
    }
}

internal readonly struct DirectPtxLoadedKernel
{
    internal DirectPtxModule Module { get; }
    internal IntPtr Function { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal DirectPtxLoadedKernel(
        DirectPtxModule module,
        IntPtr function,
        DirectPtxKernelAudit audit)
    {
        Module = module;
        Function = function;
        Audit = audit;
    }
}
