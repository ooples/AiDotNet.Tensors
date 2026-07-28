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
    private static readonly object Sync = new();
    private static Dictionary<string, string>? _winners;

    /// <summary>File the autotuner writes and this reads.</summary>
    public static string CachePath { get; set; } =
        Path.Combine("artifacts", "autotune.tsv");

    /// <summary>Forgets the loaded cache, so a fresh autotune run is picked up.</summary>
    public static void Invalidate()
    {
        lock (Sync) _winners = null;
    }

    /// <summary>
    /// The winning candidate name for a kernel, or null when it has not been tuned.
    /// </summary>
    /// <remarks>
    /// Rows stamped with a superseded measurement protocol are ignored, exactly as the
    /// release gates ignore them: a lowering chosen under a protocol that let clock
    /// drift into the ratio is not a measured choice, it is a remembered guess.
    /// </remarks>
    public static string? WinnerFor(string kernelName)
    {
        if (string.IsNullOrEmpty(kernelName)) return null;

        lock (Sync)
        {
            _winners ??= Load();
            return _winners.TryGetValue(kernelName, out string? winner) ? winner : null;
        }
    }

    private static Dictionary<string, string> Load()
    {
        var map = new Dictionary<string, string>(StringComparer.Ordinal);
        string path = CachePath;
        if (!File.Exists(path)) return map;

        foreach (string line in File.ReadAllLines(path))
        {
            if (line.Length == 0 || line[0] == '#') continue;
            string[] cells = line.Split('\t');
            if (cells.Length < 6) continue;
            if (!string.Equals(cells[5], CodegenMeasurementProtocol.Tag, StringComparison.Ordinal))
                continue;

            // Only record a winner that actually beat the modelled choice by more than
            // the harness noise floor. Below that the two are indistinguishable and
            // switching lowerings on noise is how a tuner becomes a random walk.
            if (!double.TryParse(cells[4], NumberStyles.Any, CultureInfo.InvariantCulture, out double gain))
                continue;
            if (gain <= 1.0105) continue;

            map[cells[0]] = cells[1];
        }
        return map;
    }
}
