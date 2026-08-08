using System;
using System.Collections.Generic;

namespace AiDotNet.Tensors.Helpers.Autotune;

/// <summary>
/// Opt-in community tuning exchange (Phase 2). Abstracts the backend (Supabase)
/// behind two operations so the client-side "seed the sweep with community
/// configs, then publish the local winner" flow (<see cref="CommunityAutotune"/>)
/// is testable without a network. Implementations must be safe to call from the
/// dispatch path: they should never throw (return empty / no-op on any error) so
/// telemetry can never break a kernel launch.
/// </summary>
public interface IGpuTuningExchange
{
    /// <summary>True when the exchange is opted-in and usable. When false, callers skip it entirely.</summary>
    bool IsEnabled { get; }

    /// <summary>
    /// Fetches community-reported winners for a hardware class + kernel + shape,
    /// best-first by reported throughput. Returns empty (never throws) on a miss,
    /// a disabled exchange, or any transport error.
    /// </summary>
    IReadOnlyList<GpuTuningProfile> Fetch(string modelKey, string category, string kernelName, string shapeKey);

    /// <summary>Publishes a locally-measured winner. Best-effort; never throws.</summary>
    void Publish(GpuTuningProfile profile);
}

/// <summary>
/// The default exchange: disabled and inert. Used whenever the user has not
/// opted into community tuning, so the local-only path (Phase 1) is the
/// zero-configuration behavior and no network is ever touched.
/// </summary>
public sealed class NullGpuTuningExchange : IGpuTuningExchange
{
    public static readonly NullGpuTuningExchange Instance = new();
    private NullGpuTuningExchange() { }

    public bool IsEnabled => false;

    public IReadOnlyList<GpuTuningProfile> Fetch(
        string modelKey, string category, string kernelName, string shapeKey) =>
        Array.Empty<GpuTuningProfile>();

    public void Publish(GpuTuningProfile profile) { }
}
