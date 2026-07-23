using System;
using System.Collections.Generic;
using System.Globalization;

namespace AiDotNet.Tensors.Helpers.Autotune;

/// <summary>
/// One tunable launch configuration: a stable <see cref="Variant"/> id plus the
/// structured <see cref="Parameters"/> that reconstruct the launch config, so a
/// swept winner round-trips losslessly through <see cref="AutotuneCache"/>.
/// </summary>
public readonly struct AutotuneCandidate
{
    /// <summary>Stable identifier for this configuration, e.g. <c>"tile-16"</c>.</summary>
    public string Variant { get; }

    /// <summary>
    /// Structured parameters that reconstruct the launch config (e.g. tile sizes,
    /// warp count). Persisted with the winner so dispatch never has to re-parse
    /// the variant string. Values must be non-null; empty metadata is allowed.
    /// </summary>
    public IReadOnlyDictionary<string, string> Parameters { get; }

    public AutotuneCandidate(string variant, IReadOnlyDictionary<string, string> parameters)
    {
        if (string.IsNullOrWhiteSpace(variant))
            throw new ArgumentException("A candidate variant id is required.", nameof(variant));
        Variant = variant;
        Parameters = parameters ?? EmptyParameters;
    }

    public AutotuneCandidate(string variant) : this(variant, EmptyParameters) { }

    private static readonly IReadOnlyDictionary<string, string> EmptyParameters =
        new Dictionary<string, string>(0, StringComparer.Ordinal);
}

/// <summary>The resolved tuned configuration for a kernel + shape on this device.</summary>
public readonly struct AutotuneResolution
{
    /// <summary>The chosen variant id.</summary>
    public string Variant { get; }
    /// <summary>The chosen variant's structured parameters.</summary>
    public IReadOnlyDictionary<string, string> Parameters { get; }
    /// <summary>Measured throughput of the winner (GFLOPS); 0 when unmeasured (default fallback).</summary>
    public double MeasuredGflops { get; }
    /// <summary>True when returned from the on-disk cache; false when freshly swept or defaulted.</summary>
    public bool FromCache { get; }
    /// <summary>True when an on-device sweep actually ran and measured a winner.</summary>
    public bool Measured { get; }

    internal AutotuneResolution(
        string variant, IReadOnlyDictionary<string, string> parameters,
        double measuredGflops, bool fromCache, bool measured)
    {
        Variant = variant;
        Parameters = parameters;
        MeasuredGflops = measuredGflops;
        FromCache = fromCache;
        Measured = measured;
    }
}

/// <summary>
/// Reusable "tune the first time this GPU is seen, then reuse" orchestration for
/// direct-GPU kernels. It standardizes the pattern that the CUDA attention path
/// already proves inline (<c>DirectPtxAttentionAutotuner</c>): on a cache miss,
/// sweep the candidate launch configs on-device, keep the fastest, and persist
/// it keyed by the GPU device fingerprint so every subsequent run — and every
/// future process — reuses the tuned winner without re-measuring.
///
/// <para><b>Why the fingerprint goes in the kernel name.</b> <see cref="AutotuneCache"/>
/// keys its on-disk directory on the host CPU fingerprint only, so the GPU's
/// identity must live inside <see cref="KernelId.Name"/> or two different cards
/// on one host would collide. <see cref="GpuKernelId"/> enforces that convention.</para>
///
/// <para><b>Device-agnostic by design.</b> The actual measurement is supplied as
/// a <c>benchmark</c> delegate (variant → GFLOPS, higher is better) that runs on
/// the device; this orchestration is pure and unit-testable without a GPU. It is
/// also the single seam where opt-in community tuning (Phase 2) plugs in: a
/// downloaded config is simply prepended to <paramref name="candidates"/> so it
/// competes in — and is re-verified by — the same on-device sweep, which makes a
/// poisoned or hardware-mismatched config lose rather than launch blindly.</para>
/// </summary>
public static class GpuFirstRunAutotuner
{
    /// <summary>
    /// Builds a <see cref="KernelId"/> that folds the GPU device fingerprint into
    /// the kernel name, so tuned configs are keyed per distinct card + driver.
    /// </summary>
    public static KernelId GpuKernelId(string category, string name, string deviceFingerprint)
    {
        if (string.IsNullOrWhiteSpace(category))
            throw new ArgumentException("A kernel category is required.", nameof(category));
        if (string.IsNullOrWhiteSpace(name))
            throw new ArgumentException("A kernel name is required.", nameof(name));
        if (string.IsNullOrWhiteSpace(deviceFingerprint))
            throw new ArgumentException("A GPU device fingerprint is required.", nameof(deviceFingerprint));
        return new KernelId(category, name + "@" + deviceFingerprint);
    }

    /// <summary>
    /// Overload keying on a structured <see cref="GpuDeviceFingerprint"/>, using
    /// its per-card <see cref="GpuDeviceFingerprint.LocalKey"/> so the tuned
    /// winner is cached once per physical device.
    /// </summary>
    public static KernelId GpuKernelId(string category, string name, GpuDeviceFingerprint fingerprint)
        => GpuKernelId(category, name, fingerprint.LocalKey);

    /// <summary>
    /// Resolves the tuned launch config for <paramref name="kernelId"/> +
    /// <paramref name="shape"/> on the current device.
    ///
    /// <list type="number">
    /// <item>Cache hit whose variant is still offered → returned immediately (no measurement).</item>
    /// <item>Autotuning disabled, or a single candidate → the first candidate is used as
    ///   the safe default and nothing is measured or stored.</item>
    /// <item>Otherwise every candidate is benchmarked on-device; the fastest is stored
    ///   (keyed by GPU fingerprint) and returned. A candidate whose benchmark throws or
    ///   returns a non-positive score is skipped, so a launch-failing or slow config can
    ///   never be selected. If all candidates fail, the first candidate is used as the
    ///   default and nothing is stored.</item>
    /// </list>
    /// Persistence is advisory: a read-only home directory never disables a valid
    /// in-memory winner.
    /// </summary>
    /// <param name="benchmark">Runs a candidate on the device and returns its throughput
    /// in GFLOPS (higher is better). May throw to signal an unlaunchable config.</param>
    public static AutotuneResolution Resolve(
        KernelId kernelId,
        ShapeProfile shape,
        IReadOnlyList<AutotuneCandidate> candidates,
        Func<AutotuneCandidate, double> benchmark,
        bool autotuneEnabled)
    {
        if (candidates is null || candidates.Count == 0)
            throw new ArgumentException("At least one candidate is required.", nameof(candidates));
        if (benchmark is null) throw new ArgumentNullException(nameof(benchmark));

        // 1. Cache hit — but only honor a variant that is still on offer, so a
        //    renamed/removed candidate re-tunes instead of launching a stale id.
        KernelChoice? cached = AutotuneCache.Lookup(kernelId, shape);
        if (cached is { } hit && TryFindByVariant(candidates, hit.Variant, out AutotuneCandidate cachedCandidate))
            return new AutotuneResolution(
                cachedCandidate.Variant, cachedCandidate.Parameters,
                hit.MeasuredGflops, fromCache: true, measured: false);

        AutotuneCandidate fallback = candidates[0];

        // 2. Disabled or nothing to choose between → safe default, unmeasured, unstored.
        if (!autotuneEnabled || candidates.Count == 1)
            return new AutotuneResolution(
                fallback.Variant, fallback.Parameters, 0.0, fromCache: false, measured: false);

        // 3. On-device sweep. A candidate that throws or scores <= 0 is skipped.
        AutotuneCandidate best = default;
        double bestGflops = double.NegativeInfinity;
        bool anyMeasured = false;
        foreach (AutotuneCandidate candidate in candidates)
        {
            double gflops;
            try
            {
                gflops = benchmark(candidate);
            }
            catch
            {
                continue; // unlaunchable config (e.g. shared-mem over budget) — cannot win
            }
            if (double.IsNaN(gflops) || gflops <= 0.0) continue;
            if (gflops > bestGflops)
            {
                bestGflops = gflops;
                best = candidate;
                anyMeasured = true;
            }
        }

        if (!anyMeasured)
            return new AutotuneResolution(
                fallback.Variant, fallback.Parameters, 0.0, fromCache: false, measured: false);

        var winner = new KernelChoice
        {
            Variant = best.Variant,
            Parameters = ToStringDictionary(best.Parameters),
            MeasuredGflops = bestGflops,
            RecordedAtUtc = DateTime.UtcNow
        };
        AutotuneCache.TryStore(kernelId, shape, winner); // advisory; ignores read-only home

        return new AutotuneResolution(
            best.Variant, best.Parameters, bestGflops, fromCache: false, measured: true);
    }

    private static bool TryFindByVariant(
        IReadOnlyList<AutotuneCandidate> candidates, string variant, out AutotuneCandidate match)
    {
        if (!string.IsNullOrEmpty(variant))
        {
            for (int i = 0; i < candidates.Count; i++)
            {
                if (string.Equals(candidates[i].Variant, variant, StringComparison.Ordinal))
                {
                    match = candidates[i];
                    return true;
                }
            }
        }
        match = default;
        return false;
    }

    private static Dictionary<string, string> ToStringDictionary(
        IReadOnlyDictionary<string, string> parameters)
    {
        var copy = new Dictionary<string, string>(
            parameters?.Count ?? 0, StringComparer.Ordinal);
        if (parameters is not null)
            foreach (KeyValuePair<string, string> kv in parameters)
                copy[kv.Key] = kv.Value ?? string.Empty;
        return copy;
    }
}
