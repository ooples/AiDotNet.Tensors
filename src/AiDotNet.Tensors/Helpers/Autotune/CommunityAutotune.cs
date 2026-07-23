using System;
using System.Collections.Generic;
using System.Linq;

namespace AiDotNet.Tensors.Helpers.Autotune;

/// <summary>
/// Phase 2: layers opt-in community tuning over the local first-run sweep
/// (<see cref="GpuFirstRunAutotuner"/>) using the "download-as-candidate,
/// re-verify on-device" trust model.
///
/// <para>On a cache miss it fetches community-reported winners for this exact
/// hardware class + kernel + shape and folds them into the candidate list, then
/// runs the normal on-device sweep. Because the community configs simply compete
/// as extra candidates, a poisoned or hardware-mismatched config loses (or is
/// skipped when it fails to launch) rather than being trusted. A freshly measured
/// local winner is then published back, corroborating the community record.</para>
///
/// <para><b>Safety invariants:</b> (1) community is consulted only when a sweep
/// will actually run (autotune enabled, &gt; 1 candidate), so the disabled/default
/// path is never influenced by the network; (2) local candidates stay first, so
/// the unmeasured safe default (<c>candidates[0]</c>) is always a local config;
/// (3) fetch/publish are best-effort and never throw into dispatch.</para>
/// </summary>
public static class CommunityAutotune
{
    /// <summary>
    /// Resolves the tuned config, seeding the sweep with up to
    /// <paramref name="maxCommunityCandidates"/> distinct community configs.
    /// Falls back to a pure local resolve when the exchange is disabled.
    /// </summary>
    public static AutotuneResolution Resolve(
        IGpuTuningExchange exchange,
        string category,
        string shareableKernelName,
        GpuDeviceFingerprint fingerprint,
        ShapeProfile shape,
        IReadOnlyList<AutotuneCandidate> localCandidates,
        Func<AutotuneCandidate, double> benchmark,
        bool autotuneEnabled,
        int maxCommunityCandidates = 3,
        string? clientHash = null,
        string? aidotnetVersion = null)
    {
        if (exchange is null) throw new ArgumentNullException(nameof(exchange));
        if (localCandidates is null || localCandidates.Count == 0)
            throw new ArgumentException("At least one local candidate is required.", nameof(localCandidates));

        KernelId kernelId = GpuFirstRunAutotuner.GpuKernelId(category, shareableKernelName, fingerprint);
        bool useCommunity = exchange.IsEnabled && autotuneEnabled && localCandidates.Count >= 1;

        IReadOnlyList<AutotuneCandidate> candidates = localCandidates;
        if (useCommunity && AutotuneCache.Lookup(kernelId, shape) is null)
        {
            // Only seed on a genuine cache miss — no point fetching when we already
            // have a local winner for this exact contract on this exact card.
            IReadOnlyList<GpuTuningProfile> community = SafeFetch(
                exchange, fingerprint.ModelKey, category, shareableKernelName, shape.ToFileStem());
            candidates = MergeCommunityCandidates(localCandidates, community, maxCommunityCandidates);
        }

        AutotuneResolution resolution = GpuFirstRunAutotuner.Resolve(
            kernelId, shape, candidates, benchmark, autotuneEnabled);

        // Publish only a freshly MEASURED winner (never a cache hit). This
        // corroborates a community config that just won, or contributes a new one.
        if (exchange.IsEnabled && resolution.Measured)
        {
            GpuTuningProfile profile = GpuTuningProfile.FromWinner(
                fingerprint, category, shareableKernelName, shape, resolution, clientHash, aidotnetVersion);
            try { exchange.Publish(profile); }
            catch { /* telemetry is advisory — never break dispatch */ }
        }

        return resolution;
    }

    /// <summary>
    /// Merges community candidates into the local set: local first (so the safe
    /// default is preserved), then up to <paramref name="maxCommunity"/> distinct
    /// community variants not already offered, best-reported-first. A community
    /// candidate that duplicates a local variant is dropped; one that cannot
    /// launch simply loses the sweep later.
    /// </summary>
    public static IReadOnlyList<AutotuneCandidate> MergeCommunityCandidates(
        IReadOnlyList<AutotuneCandidate> local,
        IReadOnlyList<GpuTuningProfile> community,
        int maxCommunity)
    {
        var result = new List<AutotuneCandidate>(local);
        if (community is null || community.Count == 0 || maxCommunity <= 0)
            return result;

        var seen = new HashSet<string>(StringComparer.Ordinal);
        foreach (AutotuneCandidate c in local) seen.Add(c.Variant);

        int added = 0;
        foreach (GpuTuningProfile profile in community
                     .Where(p => p is not null && !string.IsNullOrEmpty(p.Variant))
                     .OrderByDescending(p => p.MeasuredGflops))
        {
            if (added >= maxCommunity) break;
            if (seen.Add(profile.Variant))
            {
                result.Add(profile.ToCandidate());
                added++;
            }
        }
        return result;
    }

    private static IReadOnlyList<GpuTuningProfile> SafeFetch(
        IGpuTuningExchange exchange, string modelKey, string category, string kernelName, string shapeKey)
    {
        try
        {
            return exchange.Fetch(modelKey, category, kernelName, shapeKey)
                ?? (IReadOnlyList<GpuTuningProfile>)Array.Empty<GpuTuningProfile>();
        }
        catch
        {
            return Array.Empty<GpuTuningProfile>();
        }
    }
}
