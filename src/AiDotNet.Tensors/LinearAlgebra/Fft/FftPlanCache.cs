// Copyright (c) AiDotNet. All rights reserved.
// Per-size plan cache: twiddle-step factors for radix-2 and chirp sequences for
// Bluestein. Materialized once per (n, inverse) key, then reused on every
// subsequent Transform1D call. Thread-safe via ConcurrentDictionary.
//
// Why bother? For batched workloads (e.g., STFT with hundreds of frames all
// at the same nFft, or conv-in-spectrum with the same H/W across a batch), the
// forward pre-computation dominates an already-cheap log-N transform. Caching
// the twiddle array cuts that overhead to a single dictionary lookup.

using System;
using System.Collections.Concurrent;

namespace AiDotNet.Tensors.LinearAlgebra.Fft;

/// <summary>
/// Cached per-length FFT plan. Currently stores the chirp factors used by
/// Bluestein — the radix-2 path uses iterative twiddle advancement so it
/// does not need a table. Bluestein computes <c>cos(-sign·π·i²/N)</c> and
/// <c>sin(...)</c> for <c>i = 0..N−1</c>; we cache both arrays keyed on
/// <c>(N, inverse)</c>.
/// </summary>
internal sealed class BluesteinPlan
{
    public int N { get; }
    public bool Inverse { get; }
    public double[] ChirpRe { get; }
    public double[] ChirpIm { get; }
    public int M { get; }
    public double[] BSpectrumRe { get; }
    public double[] BSpectrumIm { get; }

    public BluesteinPlan(int n, bool inverse)
    {
        N = n;
        Inverse = inverse;

        // M = smallest power of 2 ≥ 2N - 1.
        int m = 1;
        while (m < 2 * n - 1) m <<= 1;
        M = m;

        double sign = inverse ? -1.0 : 1.0;
        ChirpRe = new double[n];
        ChirpIm = new double[n];
        for (int i = 0; i < n; i++)
        {
            long sq = ((long)i * i) % (2L * n);
            double phase = -sign * Math.PI * sq / n;
            ChirpRe[i] = Math.Cos(phase);
            ChirpIm[i] = Math.Sin(phase);
        }

        // Pre-FFT the B sequence. b[i] = conj(c[i]) for i in [0, n);
        // b[M - i] = b[i] for i in [1, n); zeros elsewhere.
        var b = new double[2 * m];
        b[0] = ChirpRe[0];
        b[1] = -ChirpIm[0];
        for (int i = 1; i < n; i++)
        {
            double bRe = ChirpRe[i];
            double bIm = -ChirpIm[i];
            b[2 * i] = bRe;
            b[2 * i + 1] = bIm;
            b[2 * (m - i)] = bRe;
            b[2 * (m - i) + 1] = bIm;
        }
        FftKernels.IterativeRadix2NoCache(b, m, inverse: false);
        BSpectrumRe = new double[m];
        BSpectrumIm = new double[m];
        for (int i = 0; i < m; i++)
        {
            BSpectrumRe[i] = b[2 * i];
            BSpectrumIm[i] = b[2 * i + 1];
        }
    }
}

/// <summary>
/// Global, thread-safe FFT plan cache. Lookup is keyed on the
/// <c>(n, inverse)</c> pair; plans are lazily constructed and never evicted
/// (the memory footprint is <c>O(Σ_n n + n · log n)</c> per distinct size,
/// bounded by the number of distinct transform shapes a program uses).
/// </summary>
internal static class FftPlanCache
{
    private static readonly ConcurrentDictionary<(int n, bool inverse), BluesteinPlan> _bluesteinPlans = new();

    public static BluesteinPlan GetOrCreateBluestein(int n, bool inverse)
        => _bluesteinPlans.GetOrAdd((n, inverse), k => new BluesteinPlan(k.n, k.inverse));

    /// <summary>Drops all cached plans. Intended for tests and memory-stress benchmarks.</summary>
    public static void Clear() => _bluesteinPlans.Clear();

    /// <summary>Distinct cached plan count — useful for cache-hit tests.</summary>
    public static int Count => _bluesteinPlans.Count;
}
