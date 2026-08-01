// Copyright (c) AiDotNet. All rights reserved.
// Measured lowering choices, replacing modelled ones.
//
// The cost model is adequate in regimes it was calibrated on and badly wrong outside
// them. Measured on this catalog, the modelled choice matched the best candidate to
// within the 1.05% noise floor on ten kernels and lost by 5.38x on one:
//
//   conv2d_1x1_bwd_weights   modelled 2128.3 us   measured best 395.6 us
//   conv2d_3x3_bwd_weights   modelled  240.9 us   measured best 211.2 us
//
// Both are weight gradients -- a tiny output with an enormous reduction, a regime the
// model was never calibrated against. Rather than add a term and hope, the autotuner
// measures candidates and this records which one won, so the choice is a fact.

using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;

namespace AiDotNet.Tensors.Engines.Compilation.Codegen.Ir;

/// <summary>Lowering choices that were measured rather than modelled.</summary>
public static class CodegenAutotuneCache
{
    private const double HarnessGainNoiseFloor =
        CodegenMeasurementProtocol.AutotuneGainNoiseFloor;
    private static readonly object Sync = new();
    private static Dictionary<CacheKey, string>? _winners;
    private static string _cachePath = DefaultCachePath();

    private readonly record struct CacheKey(
        string Kernel,
        string Device,
        string Target,
        string Spec,
        string Emitter);

    /// <summary>File the autotuner writes and this reads.</summary>
    public static string CachePath
    {
        get { lock (Sync) return _cachePath; }
        set
        {
            if (string.IsNullOrWhiteSpace(value))
                throw new ArgumentException("Autotune cache path cannot be empty.", nameof(value));
            lock (Sync)
            {
                _cachePath = value;
                _winners = null;
            }
        }
    }

    /// <summary>Forgets the loaded cache, so a fresh autotune run is picked up.</summary>
    public static void Invalidate()
    {
        lock (Sync) _winners = null;
    }

    /// <summary>
    /// The winning candidate name for a kernel and exact build identity, or null when it has
    /// not been tuned under that identity.
    /// </summary>
    /// <remarks>
    /// Rows stamped with a superseded measurement protocol are ignored, exactly as the
    /// release gates ignore them: a lowering chosen under a protocol that let clock
    /// drift into the ratio is not a measured choice, it is a remembered guess.
    /// </remarks>
    public static string? WinnerFor(string kernelName, CodegenAutotuneIdentity identity)
    {
        if (string.IsNullOrEmpty(kernelName)) return null;
        if (identity is null) throw new ArgumentNullException(nameof(identity));

        lock (Sync)
        {
            _winners ??= Load();
            var key = new CacheKey(
                kernelName,
                identity.DeviceFingerprint,
                identity.Target,
                identity.SpecFingerprint,
                identity.EmitterFingerprint);
            return _winners.TryGetValue(key, out string? winner) ? winner : null;
        }
    }

    private static Dictionary<CacheKey, string> Load()
    {
        var map = new Dictionary<CacheKey, string>();
        string path = CachePath;
        string[] lines;
        try
        {
            lines = File.ReadAllLines(path);
        }
        catch (IOException)
        {
            return map;
        }
        catch (UnauthorizedAccessException)
        {
            return map;
        }

        foreach (string line in lines)
        {
            if (line.Length == 0 || line[0] == '#') continue;
            string[] cells = line.Split('\t');
            // A promotable row must carry exact build identity and prove that every
            // candidate in that identity's finite search was considered. Candidate-filtered
            // probes are useful experiments, but accepting one as dispatch would let a
            // hand-picked subset masquerade as autotuning.
            if (cells.Length != 11 ||
                !string.Equals(cells[10], "full", StringComparison.Ordinal))
                continue;
            if (!string.Equals(cells[5], CodegenMeasurementProtocol.Tag, StringComparison.Ordinal))
                continue;

            // Only record a winner that actually beat the modelled choice by more than
            // the harness noise floor. Below that the two are indistinguishable and
            // switching lowerings on noise is how a tuner becomes a random walk.
            if (!double.TryParse(cells[4], NumberStyles.Any, CultureInfo.InvariantCulture, out double gain))
                continue;
            if (gain <= HarnessGainNoiseFloor) continue;

            var key = new CacheKey(cells[0], cells[6], cells[7], cells[8], cells[9]);
            map[key] = cells[1];
        }
        return map;
    }

    private static string DefaultCachePath()
    {
        string? configured =
            Environment.GetEnvironmentVariable("AIDOTNET_CODEGEN_AUTOTUNE_CACHE");
        return string.IsNullOrWhiteSpace(configured)
            ? Path.Combine("artifacts", "autotune.tsv")
            : configured;
    }
}
