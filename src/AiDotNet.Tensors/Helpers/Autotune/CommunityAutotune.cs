using System;
using System.Collections.Generic;
using System.Linq;

namespace AiDotNet.Tensors.Helpers.Autotune;

/// <summary>
/// Phase 2: layers opt-in community tuning over the local first-run sweep
/// (<see cref="GpuFirstRunAutotuner"/>) using a local-allowlist plus
/// on-device remeasurement trust model.
///
/// <para>On a cache miss, or when rehydrating a cached community-only winner, it
/// fetches community-reported winners for this exact hardware class + kernel +
/// shape and folds them into the candidate list, then runs the normal on-device
/// sweep. A community config must first pass the
/// caller's local variant allowlist, then its reported performance is ignored and
/// re-measured on-device. A freshly measured local winner is then published back,
/// corroborating the community record.</para>
///
/// <para><b>Safety invariants:</b> (1) community is consulted only when a sweep
/// will actually run (autotune enabled, &gt; 1 candidate), so the disabled/default
/// path is never influenced by the network; (2) local candidates stay first, so
/// the unmeasured safe default (<c>candidates[0]</c>) is always a local config;
/// (3) remote candidates must pass a caller-owned local allowlist before they can
/// be benchmarked; (4) fetch, conversion, validation, and publish failures never
/// throw into dispatch.</para>
/// </summary>
public static class CommunityAutotune
{
    /// <summary>
    /// Resolves the tuned config, seeding the sweep with up to
    /// <paramref name="maxCommunityCandidates"/> distinct community configs.
    /// Falls back to a pure local resolve when the exchange is disabled.
    /// </summary>
    /// <param name="isCommunityCandidateAllowed">
    /// Local structural allowlist for downloaded configurations. It must accept
    /// only variants implemented and supported by this build for the current shape.
    /// </param>
    public static AutotuneResolution Resolve(
        IGpuTuningExchange exchange,
        string category,
        string shareableKernelName,
        GpuDeviceFingerprint fingerprint,
        ShapeProfile shape,
        IReadOnlyList<AutotuneCandidate> localCandidates,
        Func<AutotuneCandidate, double> benchmark,
        Func<AutotuneCandidate, bool> isCommunityCandidateAllowed,
        bool autotuneEnabled,
        int maxCommunityCandidates = 3,
        string? clientHash = null,
        string? aidotnetVersion = null)
    {
        if (exchange is null) throw new ArgumentNullException(nameof(exchange));
        if (localCandidates is null || localCandidates.Count == 0)
            throw new ArgumentException("At least one local candidate is required.", nameof(localCandidates));
        if (benchmark is null) throw new ArgumentNullException(nameof(benchmark));
        if (isCommunityCandidateAllowed is null)
            throw new ArgumentNullException(nameof(isCommunityCandidateAllowed));

        KernelId kernelId = GpuFirstRunAutotuner.GpuKernelId(category, shareableKernelName, fingerprint);
        // A single local candidate has no tuning decision to make. Keep that
        // default path network-free exactly as the class contract promises.
        bool useCommunity = exchange.IsEnabled && autotuneEnabled && localCandidates.Count > 1;

        IReadOnlyList<AutotuneCandidate> candidates = localCandidates;
        KernelChoice? cached = useCommunity ? AutotuneCache.Lookup(kernelId, shape) : null;
        if (useCommunity &&
            (cached is null || !ContainsVariant(localCandidates, cached.Variant)))
        {
            // Fetch on a genuine cache miss, and also to rehydrate a cached
            // community-only variant that is absent from the local candidate set.
            // A cached local winner remains a zero-network fast path.
            IReadOnlyList<GpuTuningProfile> community = SafeFetch(
                exchange, fingerprint.ModelKey, category, shareableKernelName, shape.ToFileStem());
            candidates = MergeCommunityCandidates(
                localCandidates, community, isCommunityCandidateAllowed, maxCommunityCandidates);
        }

        AutotuneResolution resolution = GpuFirstRunAutotuner.Resolve(
            kernelId, shape, candidates, benchmark, autotuneEnabled);

        // Publish only a freshly MEASURED winner (never a cache hit). This
        // corroborates a community config that just won, or contributes a new one.
        if (exchange.IsEnabled && resolution.Measured)
        {
            try
            {
                GpuTuningProfile profile = GpuTuningProfile.FromWinner(
                    fingerprint, category, shareableKernelName, shape, resolution,
                    clientHash, aidotnetVersion);
                exchange.Publish(profile);
            }
            catch { /* telemetry is advisory — never break dispatch */ }
        }

        return resolution;
    }

    private static bool ContainsVariant(
        IReadOnlyList<AutotuneCandidate> candidates, string variant)
    {
        if (string.IsNullOrEmpty(variant)) return false;
        for (int i = 0; i < candidates.Count; i++)
            if (string.Equals(candidates[i].Variant, variant, StringComparison.Ordinal))
                return true;
        return false;
    }

    /// <summary>
    /// Merges community candidates into the local set: local first (so the safe
    /// default is preserved), then up to <paramref name="maxCommunity"/> distinct
    /// community variants not already offered, best-reported-first. A community
    /// candidate that is malformed, rejected by the local allowlist, or duplicates
    /// a local variant is dropped before the sweep.
    /// </summary>
    public static IReadOnlyList<AutotuneCandidate> MergeCommunityCandidates(
        IReadOnlyList<AutotuneCandidate> local,
        IReadOnlyList<GpuTuningProfile> community,
        Func<AutotuneCandidate, bool> isCommunityCandidateAllowed,
        int maxCommunity)
    {
        if (local is null) throw new ArgumentNullException(nameof(local));
        if (isCommunityCandidateAllowed is null)
            throw new ArgumentNullException(nameof(isCommunityCandidateAllowed));

        var result = new List<AutotuneCandidate>(local);
        if (community is null || community.Count == 0 || maxCommunity <= 0)
            return result;

        var seen = new HashSet<string>(StringComparer.Ordinal);
        foreach (AutotuneCandidate c in local) seen.Add(c.Variant);

        int added = 0;
        foreach (GpuTuningProfile profile in community
                     .Where(p => p is not null && !string.IsNullOrWhiteSpace(p.Variant))
                     .OrderByDescending(p => p.MeasuredGflops))
        {
            if (added >= maxCommunity) break;
            AutotuneCandidate candidate;
            bool allowed;
            try
            {
                candidate = profile.ToCandidate();
                allowed = isCommunityCandidateAllowed(candidate);
            }
            catch
            {
                continue;
            }
            if (!allowed || !seen.Add(candidate.Variant)) continue;

            result.Add(candidate);
            added++;
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
