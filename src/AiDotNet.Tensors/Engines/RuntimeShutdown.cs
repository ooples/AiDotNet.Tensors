using System;

namespace AiDotNet.Tensors.Engines;

/// <summary>
/// Process/AppDomain teardown detection shared by GPU backends and pools.
/// </summary>
/// <remarks>
/// <para>
/// During CLR shutdown or AppDomain unload, finalizers run after much of the
/// runtime state has already been torn down: <see cref="System.Threading.ThreadLocal{T}"/>
/// instances (and the <see cref="System.Collections.Concurrent.ConcurrentBag{T}"/> built
/// on them) may already be finalized, and native GPU drivers/contexts may already be
/// unloaded. Touching either from a finalizer then throws
/// <see cref="ObjectDisposedException"/> or raises an access violation (0xC0000005) — both
/// fatal on the finalizer thread, killing the whole process.
/// </para>
/// <para>
/// <see cref="Environment.HasShutdownStarted"/> alone proved unreliable (some hosts — e.g.
/// the vstest test host — run finalizers without that flag being set), so this type also
/// latches a flag from the <see cref="AppDomain.ProcessExit"/> / <see cref="AppDomain.DomainUnload"/>
/// events, which fire before the final finalizer sweep.
/// </para>
/// </remarks>
internal static class RuntimeShutdown
{
    private static volatile bool _exiting;

    static RuntimeShutdown()
    {
        try
        {
            AppDomain.CurrentDomain.ProcessExit += static (_, _) => _exiting = true;
            AppDomain.CurrentDomain.DomainUnload += static (_, _) => _exiting = true;
        }
        catch
        {
            // If subscription fails in some constrained host, callers still have the
            // finalizer-path (disposing == false) guard as the primary protection.
        }
    }

    /// <summary>
    /// True once the process/domain has begun tearing down. Finalizer-reachable cleanup
    /// must skip all native driver calls and cross-object managed access when this is true.
    /// </summary>
    internal static bool IsTearingDown =>
        _exiting || Environment.HasShutdownStarted || AppDomain.CurrentDomain.IsFinalizingForUnload();
}
