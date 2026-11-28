// Copyright (c) CrowdTrainer LLC. All rights reserved.

using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text.Json;
using AiDotNet.Tensors.Helpers.Autotune.TunedKernels;

namespace AiDotNet.Tensors.Engines.Compilation.Autotune;

/// <summary>
/// Optional, removable external vendor library candidate provider (e.g., <c>cuBLASLt</c>, <c>cuDNN</c>, <c>oneDNN</c>).
/// Vendor libraries are strictly isolated behind this contract rather than coupled into <c>FusionTileIR</c> or the
/// core graph executor so self-contained <c>Direct-PTX</c> / NVRTC fused tiled kernels serve as the primary standalone
/// backend and can run with zero external vendor dependencies when <see cref="DirectPtxAutotunerOptions.AllowExternalVendorLibraries"/>
/// is <c>false</c>.
/// </summary>
internal interface IExternalLibraryCandidateProvider
{
    /// <summary>Short vendor provider identifier (e.g., <c>"cuBLASLt"</c>).</summary>
    string ProviderName { get; }

    /// <summary>True when the external library is loaded and permitted on the current device.</summary>
    bool IsAvailable { get; }

    /// <summary>Enumerates vendor execution candidates for <paramref name="shape"/>.</summary>
    IReadOnlyList<AutotuneExecutableCandidate> GetVendorCandidates(DirectPtxShapeKey shape);
}

/// <summary>
/// An executable candidate benchmarked by <see cref="DirectPtxAutotuner"/>.
/// </summary>
internal sealed class AutotuneExecutableCandidate
{
    public string CandidateId { get; }
    public TunedKernelOrigin Origin { get; }
    public Action Execute { get; }
    public Action? SynchronizeDevice { get; }

    public AutotuneExecutableCandidate(
        string candidateId,
        TunedKernelOrigin origin,
        Action execute,
        Action? synchronizeDevice = null)
    {
        if (string.IsNullOrWhiteSpace(candidateId))
            throw new ArgumentException("CandidateId cannot be empty.", nameof(candidateId));
        CandidateId = candidateId;
        Origin = origin;
        Execute = execute ?? throw new ArgumentNullException(nameof(execute));
        SynchronizeDevice = synchronizeDevice;
    }
}

/// <summary>
/// Shape and architecture key for persistent Direct-PTX hardware autotuning: <c>(SmVersion, OpFamily, M, N, K, EpilogueTag)</c>.
/// </summary>
internal readonly struct DirectPtxShapeKey : IEquatable<DirectPtxShapeKey>
{
    public int SmVersion { get; }
    public TunedKernelOp OpFamily { get; }
    public int M { get; }
    public int N { get; }
    public int K { get; }
    public string EpilogueTag { get; }

    public DirectPtxShapeKey(int smVersion, TunedKernelOp opFamily, int m, int n, int k, string epilogueTag = "none")
    {
        SmVersion = smVersion;
        OpFamily = opFamily;
        M = m;
        N = n;
        K = k;
        EpilogueTag = string.IsNullOrWhiteSpace(epilogueTag) ? "none" : epilogueTag;
    }

    public string ToPersistenceKey() =>
        DirectPtxKeyFormatter.CreateKey(SmVersion, OpFamily, M, N, K, EpilogueTag);

    public bool Equals(DirectPtxShapeKey other) =>
        SmVersion == other.SmVersion &&
        OpFamily == other.OpFamily &&
        M == other.M &&
        N == other.N &&
        K == other.K &&
        string.Equals(EpilogueTag, other.EpilogueTag, StringComparison.Ordinal);

    public override bool Equals(object? obj) => obj is DirectPtxShapeKey other && Equals(other);

    public override int GetHashCode()
    {
        unchecked
        {
            int h = SmVersion;
            h = (h * 397) ^ (int)OpFamily;
            h = (h * 397) ^ M;
            h = (h * 397) ^ N;
            h = (h * 397) ^ K;
            h = (h * 397) ^ StringComparer.Ordinal.GetHashCode(EpilogueTag);
            return h;
        }
    }

    public override string ToString() =>
        string.Concat(
            "sm", SmVersion.ToString(CultureInfo.InvariantCulture),
            ":", OpFamily.ToString(),
            ":", M.ToString(CultureInfo.InvariantCulture),
            "x", N.ToString(CultureInfo.InvariantCulture),
            "x", K.ToString(CultureInfo.InvariantCulture),
            ":", EpilogueTag);
}

internal static class DirectPtxKeyFormatter
{
    internal static string CreateKey(int sm, TunedKernelOp op, int m, int n, int k, string epilogue)
        => string.Concat(
            "sm", sm.ToString(CultureInfo.InvariantCulture),
            "|", op.ToString(),
            "|", m.ToString(CultureInfo.InvariantCulture),
            "|", n.ToString(CultureInfo.InvariantCulture),
            "|", k.ToString(CultureInfo.InvariantCulture),
            "|", epilogue);
}

/// <summary>
/// Persisted autotuning measurement record comparing self-contained Direct-PTX / NVRTC candidates against optional
/// external vendor library candidates using raw unscaled <see cref="Stopwatch"/> timestamps.
/// </summary>
internal sealed class DirectPtxAutotuneRecord
{
    public string Key { get; set; } = string.Empty;
    public int SmVersion { get; set; }
    public TunedKernelOp OpFamily { get; set; }
    public int M { get; set; }
    public int N { get; set; }
    public int K { get; set; }
    public string EpilogueTag { get; set; } = "none";
    public string WinningCandidateId { get; set; } = string.Empty;
    public TunedKernelOrigin WinnerOrigin { get; set; }
    public double BestDirectPtxMedianMicroseconds { get; set; } = double.NaN;
    public double BestVendorMedianMicroseconds { get; set; } = double.NaN;

    /// <summary>
    /// Unclamped ratio <c>BestDirectPtxMedianMicroseconds / BestVendorMedianMicroseconds</c> (<c>&lt; 1.0</c> means
    /// self-contained Direct-PTX is faster than the vendor library; <c>&gt; 1.0</c> flags a shape where vendor still
    /// leads; <see cref="double.NaN"/> when vendor libraries are disabled or unavailable).
    /// </summary>
    public double DirectPtxVsVendorRatio { get; set; } = double.NaN;

    /// <summary>
    /// Canonical signature of the eligible candidate set (origin + id of every candidate) this record was measured
    /// over. A cached record is reused only when the current eligible set has the same signature.
    /// </summary>
    public string CandidateSetSignature { get; set; } = string.Empty;
}

/// <summary>
/// Options controlling <see cref="DirectPtxAutotuner"/>.
/// </summary>
internal sealed class DirectPtxAutotunerOptions
{
    /// <summary>
    /// When false, external vendor candidates (<see cref="IExternalLibraryCandidateProvider"/>) are completely excluded
    /// so execution relies 100% on self-contained Direct-PTX / NVRTC kernels.
    /// </summary>
    public bool AllowExternalVendorLibraries { get; init; } =
        Environment.GetEnvironmentVariable("AIDOTNET_ALLOW_VENDOR_LIBRARIES") != "0";

    /// <summary>Warmup launches executed per candidate prior to raw <see cref="Stopwatch"/> timing.</summary>
    public int WarmupRuns { get; init; } = 3;

    /// <summary>Measured launches executed per candidate; median latency selects the winner.</summary>
    public int MeasuredRuns { get; init; } = 7;

    /// <summary>Optional explicit disk file path for JSON persistence.</summary>
    public string? PersistenceFilePath { get; init; }
}

/// <summary>
/// Persistent disk-cached hardware autotuner racing self-contained <c>Direct-PTX</c> / NVRTC fused tiled kernels
/// (first-class primary candidates) against optional removable <see cref="IExternalLibraryCandidateProvider"/> candidates.
/// </summary>
internal sealed class DirectPtxAutotuner
{
    private readonly DirectPtxAutotunerOptions _options;
    private readonly IExternalLibraryCandidateProvider? _vendorProvider;
    private readonly ConcurrentDictionary<string, DirectPtxAutotuneRecord> _records = new(StringComparer.Ordinal);
    private readonly object _persistLock = new();

    public DirectPtxAutotuner(
        DirectPtxAutotunerOptions? options = null,
        IExternalLibraryCandidateProvider? vendorProvider = null)
    {
        _options = options ?? new DirectPtxAutotunerOptions();
        if (_options.WarmupRuns < 0)
            throw new ArgumentOutOfRangeException(nameof(options), "WarmupRuns must be non-negative.");
        if (_options.MeasuredRuns < 1)
            throw new ArgumentOutOfRangeException(nameof(options), "MeasuredRuns must be at least 1.");

        _vendorProvider = vendorProvider;
        TryLoadFromDisk();
    }

    public bool AllowExternalVendorLibraries => _options.AllowExternalVendorLibraries;

    public IReadOnlyDictionary<string, DirectPtxAutotuneRecord> Records => _records;

    /// <summary>
    /// Selects the fastest candidate for <paramref name="shape"/>, either from the cache or by live measurement.
    /// Live measurement is sequential: each candidate, in order, runs its warmup launches and then all of its measured
    /// launches (median latency) before the next candidate starts; measurements are not interleaved across candidates.
    /// A cached record is reused only when it was measured over exactly the current eligible candidate set (same
    /// origins and ids); otherwise the eligible set is re-measured. Candidate identity is <c>(Origin, CandidateId)</c>
    /// and must be unique within the eligible set.
    /// Self-contained Direct-PTX candidates are always required; vendor candidates participate only when
    /// <see cref="AllowExternalVendorLibraries"/> is enabled and <see cref="IExternalLibraryCandidateProvider.IsAvailable"/> is true.
    /// </summary>
    public AutotuneExecutableCandidate SelectOrTune(
        DirectPtxShapeKey shape,
        IReadOnlyList<AutotuneExecutableCandidate> directPtxCandidates,
        out DirectPtxAutotuneRecord record)
    {
        if (directPtxCandidates is null || directPtxCandidates.Count == 0)
            throw new ArgumentException("At least one self-contained Direct-PTX candidate is required.", nameof(directPtxCandidates));

        var allCandidates = new List<AutotuneExecutableCandidate>(directPtxCandidates.Count + 4);
        for (int i = 0; i < directPtxCandidates.Count; i++)
        {
            allCandidates.Add(directPtxCandidates[i]);
        }

        if (_options.AllowExternalVendorLibraries && _vendorProvider is { IsAvailable: true })
        {
            var vendorList = _vendorProvider.GetVendorCandidates(shape);
            for (int i = 0; i < vendorList.Count; i++)
            {
                allCandidates.Add(vendorList[i]);
            }
        }

        string candidateSetSignature = ComputeCandidateSetSignature(allCandidates);

        string key = shape.ToString();
        if (_records.TryGetValue(key, out var cached) &&
            string.Equals(cached.CandidateSetSignature, candidateSetSignature, StringComparison.Ordinal))
        {
            // Same eligible set => the cached comparison (winner + vendor ratio) is still valid.
            var matched = allCandidates.FirstOrDefault(c =>
                c.Origin == cached.WinnerOrigin &&
                string.Equals(c.CandidateId, cached.WinningCandidateId, StringComparison.Ordinal));
            if (matched is not null)
            {
                record = cached;
                return matched;
            }
        }

        TunedKernelRegistry.BeginMeasurement();
        try
        {
            double bestOverallUs = double.PositiveInfinity;
            AutotuneExecutableCandidate bestCandidate = allCandidates[0];
            double bestDirectPtxUs = double.PositiveInfinity;
            double bestVendorUs = double.PositiveInfinity;

            for (int c = 0; c < allCandidates.Count; c++)
            {
                var candidate = allCandidates[c];
                double medianUs = MeasureCandidateMedianMicroseconds(candidate, _options.WarmupRuns, _options.MeasuredRuns);

                if (candidate.Origin == TunedKernelOrigin.Vendor)
                {
                    if (medianUs < bestVendorUs) bestVendorUs = medianUs;
                }
                else
                {
                    if (medianUs < bestDirectPtxUs) bestDirectPtxUs = medianUs;
                }

                if (medianUs < bestOverallUs)
                {
                    bestOverallUs = medianUs;
                    bestCandidate = candidate;
                }
            }

            double finalDirectUs = double.IsPositiveInfinity(bestDirectPtxUs) ? double.NaN : bestDirectPtxUs;
            double finalVendorUs = double.IsPositiveInfinity(bestVendorUs) ? double.NaN : bestVendorUs;
            double ratio = (!double.IsNaN(finalDirectUs) && !double.IsNaN(finalVendorUs) && finalVendorUs > 0.0)
                ? finalDirectUs / finalVendorUs
                : double.NaN;

            var measuredRecord = new DirectPtxAutotuneRecord
            {
                Key = key,
                SmVersion = shape.SmVersion,
                OpFamily = shape.OpFamily,
                M = shape.M,
                N = shape.N,
                K = shape.K,
                EpilogueTag = shape.EpilogueTag,
                WinningCandidateId = bestCandidate.CandidateId,
                WinnerOrigin = bestCandidate.Origin,
                BestDirectPtxMedianMicroseconds = finalDirectUs,
                BestVendorMedianMicroseconds = finalVendorUs,
                DirectPtxVsVendorRatio = ratio,
                CandidateSetSignature = candidateSetSignature
            };

            _records[key] = measuredRecord;
            TrySaveToDisk();
            record = measuredRecord;
            return bestCandidate;
        }
        finally
        {
            TunedKernelRegistry.EndMeasurement();
        }
    }

    /// <summary>
    /// Surfaces every tuned shape where an external vendor library currently beats self-contained Direct-PTX
    /// (<c>DirectPtxVsVendorRatio &gt; 1.0</c>), ordered by largest gap first for targeted Direct-PTX replacement.
    /// </summary>
    public IReadOnlyList<DirectPtxAutotuneRecord> GetVendorLeadingShapes()
    {
        return _records.Values
            .Where(r => !double.IsNaN(r.DirectPtxVsVendorRatio) && r.DirectPtxVsVendorRatio > 1.0)
            .OrderByDescending(r => r.DirectPtxVsVendorRatio)
            .ToList();
    }

    /// <summary>
    /// Builds an order-independent signature of the eligible set and rejects duplicate <c>(Origin, CandidateId)</c>
    /// identities, which would make a cached winner ambiguous. Each identity is length-prefixed so the joined
    /// form is unambiguous whatever characters the ids contain.
    /// </summary>
    private static string ComputeCandidateSetSignature(List<AutotuneExecutableCandidate> candidates)
    {
        var identities = new List<string>(candidates.Count);
        var seen = new HashSet<string>(StringComparer.Ordinal);
        for (int i = 0; i < candidates.Count; i++)
        {
            var c = candidates[i] ?? throw new ArgumentException("Candidate lists cannot contain null entries.");
            string identity = string.Concat(((int)c.Origin).ToString(CultureInfo.InvariantCulture), ":", c.CandidateId);
            if (!seen.Add(identity))
                throw new ArgumentException($"Duplicate autotune candidate identity '{c.Origin}:{c.CandidateId}' in the eligible candidate set.");
            identities.Add(identity.Length.ToString(CultureInfo.InvariantCulture) + "#" + identity);
        }
        identities.Sort(StringComparer.Ordinal);
        return string.Join("|", identities);
    }

    private static double MeasureCandidateMedianMicroseconds(
        AutotuneExecutableCandidate candidate,
        int warmupRuns,
        int measuredRuns)
    {
        for (int w = 0; w < warmupRuns; w++)
        {
            candidate.Execute();
        }
        candidate.SynchronizeDevice?.Invoke();

        var samples = new double[measuredRuns];
        double ticksToMicroseconds = 1_000_000.0 / Stopwatch.Frequency;
        for (int m = 0; m < measuredRuns; m++)
        {
            long t0 = Stopwatch.GetTimestamp();
            candidate.Execute();
            candidate.SynchronizeDevice?.Invoke();
            long t1 = Stopwatch.GetTimestamp();
            samples[m] = (t1 - t0) * ticksToMicroseconds;
        }

        Array.Sort(samples);
        int mid = measuredRuns >> 1;
        return (measuredRuns & 1) == 1
            ? samples[mid]
            : 0.5 * (samples[mid - 1] + samples[mid]);
    }

    private static readonly JsonSerializerOptions s_jsonOptions = new()
    {
        NumberHandling = System.Text.Json.Serialization.JsonNumberHandling.AllowNamedFloatingPointLiterals,
        WriteIndented = true
    };

    private void TryLoadFromDisk()
    {
        string? path = _options.PersistenceFilePath;
        if (string.IsNullOrWhiteSpace(path) || !File.Exists(path))
            return;

        try
        {
            string json = File.ReadAllText(path);
            var list = JsonSerializer.Deserialize<List<DirectPtxAutotuneRecord>>(json, s_jsonOptions);
            if (list is null) return;
            for (int i = 0; i < list.Count; i++)
            {
                var item = list[i];
                if (!string.IsNullOrWhiteSpace(item.Key))
                {
                    _records[item.Key] = item;
                }
            }
        }
        catch
        {
            // Corrupted cache file is ignored and overwritten on next tune.
        }
    }

    private void TrySaveToDisk()
    {
        string? path = _options.PersistenceFilePath;
        if (string.IsNullOrWhiteSpace(path))
            return;

        lock (_persistLock)
        {
            string? tempPath = null;
            try
            {
                string? dir = Path.GetDirectoryName(path);
                if (!string.IsNullOrWhiteSpace(dir))
                {
                    Directory.CreateDirectory(dir);
                }

                var snapshot = _records.Values.OrderBy(r => r.Key, StringComparer.Ordinal).ToList();
                string json = JsonSerializer.Serialize(snapshot, s_jsonOptions);
                tempPath = path + "." + Guid.NewGuid().ToString("N") + ".tmp";
                File.WriteAllText(tempPath, json);

                // Atomic publish: the previous tuning file stays intact until the replacement succeeds.
                if (File.Exists(path))
                {
#if NETCOREAPP3_0_OR_GREATER
                    File.Move(tempPath, path, overwrite: true);
#else
                    File.Replace(tempPath, path, destinationBackupFileName: null);
#endif
                }
                else
                {
                    File.Move(tempPath, path);
                }
            }
            catch
            {
                // Best-effort persistence in read-only environments.
            }
            finally
            {
                if (tempPath is not null)
                {
                    try
                    {
                        if (File.Exists(tempPath)) File.Delete(tempPath);
                    }
                    catch
                    {
                        // Ignore staged-file cleanup failures.
                    }
                }
            }
        }
    }
}
