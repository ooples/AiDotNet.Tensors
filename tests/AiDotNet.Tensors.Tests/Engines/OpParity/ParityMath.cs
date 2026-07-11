// Copyright (c) AiDotNet. All rights reserved.
// CPU-vs-GPU op-parity scaffold (Tensors #775). ULP-accurate comparison primitives.
#if !NETFRAMEWORK

using System;

namespace AiDotNet.Tensors.Tests.Engines.OpParity;

/// <summary>
/// Per-op acceptance bound for a parity comparison. Expressed primarily in ULPs (units in the
/// last place) — the natural, scale-invariant unit for floating-point agreement — with an
/// absolute floor for values near zero (where ULP distance explodes across the 0 boundary).
/// An exact op (Add, Reshape, Permute) uses <see cref="Ulps"/> == 0; a reduction / softmax /
/// transcendental uses a small ULP budget that reflects its accumulation/rounding error.
/// </summary>
public readonly struct ParityTol
{
    /// <summary>Maximum allowed ULP distance between two float results. The scale-invariant unit;
    /// 0 for exact (bit-identical) ops.</summary>
    public long Ulps { get; }

    /// <summary>Absolute-difference floor: values whose |a-b| is below this pass regardless of
    /// ULP distance. Guards the denormal / around-zero region where 1 ULP is meaningless.</summary>
    public double AbsFloor { get; }

    /// <summary>Relative-difference budget for accumulation-heavy ops (matmul, reductions, softmax)
    /// where a different-but-equally-valid summation order legitimately separates two results by many
    /// ULP. An element passes if it is within EITHER the ULP budget, the abs floor, OR this rel bound.</summary>
    public double Rel { get; }

    public ParityTol(long ulps, double absFloor = 1e-6, double rel = 0.0)
    {
        Ulps = ulps;
        AbsFloor = absFloor;
        Rel = rel;
    }

    /// <summary>Bit-exact: 0 ULP, no slack. For pure data movement and elementwise ops that use
    /// identical scalar math on both engines.</summary>
    public static ParityTol Exact => new ParityTol(0, 0.0, 0.0);

    /// <summary>A few ULP of rounding slack, no relative budget — for elementwise transcendentals
    /// whose per-element result should agree to a handful of ULP (GELU, Sigmoid, Exp, Tanh).</summary>
    public static ParityTol Ulp(long ulps, double absFloor = 1e-6) => new ParityTol(ulps, absFloor, 0.0);

    /// <summary>Relative tolerance (+ small ULP) for accumulation ops (matmul, softmax, LayerNorm,
    /// reductions) where summation order differs between engines.</summary>
    public static ParityTol Accum(double rel, long ulps = 64, double absFloor = 1e-5) => new ParityTol(ulps, absFloor, rel);

    public override string ToString() =>
        Ulps == 0 && AbsFloor == 0.0 && Rel == 0.0 ? "bit-exact" : $"{Ulps} ULP / rel {Rel:E1} / absFloor {AbsFloor:E1}";
}

/// <summary>Outcome of comparing two float arrays: worst observed ULP + abs/rel deltas and where.</summary>
public readonly struct ParityDelta
{
    public long MaxUlp { get; }
    public double MaxAbs { get; }
    public double MaxRel { get; }
    public int WorstIndex { get; }
    public float WorstA { get; }
    public float WorstB { get; }

    public ParityDelta(long maxUlp, double maxAbs, double maxRel, int worstIndex, float worstA, float worstB)
    {
        MaxUlp = maxUlp; MaxAbs = maxAbs; MaxRel = maxRel; WorstIndex = worstIndex; WorstA = worstA; WorstB = worstB;
    }

    public string Describe() =>
        $"maxUlp={MaxUlp} maxAbs={MaxAbs:E3} maxRel={MaxRel:E3} @[{WorstIndex}] a={WorstA:R} b={WorstB:R}";
}

/// <summary>ULP-accurate float comparison used by the parity harness and the double oracle.</summary>
public static class ParityMath
{
    /// <summary>Maps a float's bit pattern to a monotonic signed key so that adjacent representable
    /// floats differ by exactly 1 and the ordering matches numeric order (incl. the ±0 seam →
    /// both map to 0). Standard two's-complement flip done in long to avoid overflow.</summary>
    private static long MonotonicKey(float x)
    {
        int i = BitConverter.SingleToInt32Bits(x);
        return i < 0 ? (long)int.MinValue - i : i;
    }

    /// <summary>ULP distance between two floats. NaN==NaN → 0; a NaN vs non-NaN → max; equal
    /// infinities → 0, opposite → max. Otherwise the difference of monotonic keys.</summary>
    public static long UlpDistance(float a, float b)
    {
        if (a == b) return 0;
        bool an = float.IsNaN(a), bn = float.IsNaN(b);
        if (an && bn) return 0;
        if (an || bn) return long.MaxValue;
        if (float.IsInfinity(a) || float.IsInfinity(b)) return long.MaxValue;
        return Math.Abs(MonotonicKey(a) - MonotonicKey(b));
    }

    /// <summary>Compare two float arrays elementwise and return the worst-case deltas.</summary>
    public static ParityDelta Compare(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        if (a.Length != b.Length)
            throw new ArgumentException($"length mismatch {a.Length} vs {b.Length}");
        long maxUlp = 0; double maxAbs = 0, maxRel = 0; int worst = -1; float wa = 0, wb = 0;
        for (int i = 0; i < a.Length; i++)
        {
            long u = UlpDistance(a[i], b[i]);
            double abs = Math.Abs((double)a[i] - b[i]);
            double denom = Math.Max(Math.Abs((double)a[i]), Math.Abs((double)b[i]));
            double rel = denom > 0 ? abs / denom : 0;
            if (u > maxUlp) { maxUlp = u; worst = i; wa = a[i]; wb = b[i]; }
            if (abs > maxAbs) maxAbs = abs;
            if (rel > maxRel) maxRel = rel;
        }
        return new ParityDelta(maxUlp, maxAbs, maxRel, worst, wa, wb);
    }

    /// <summary>True iff EVERY element passes at least one of the tolerance's three gates:
    /// within the ULP budget, within the absolute floor, or within the relative budget.</summary>
    public static bool Within(ReadOnlySpan<float> a, ReadOnlySpan<float> b, ParityTol tol, out ParityDelta delta)
    {
        delta = Compare(a, b);
        for (int i = 0; i < a.Length; i++)
        {
            if (UlpDistance(a[i], b[i]) <= tol.Ulps) continue;
            double abs = Math.Abs((double)a[i] - b[i]);
            if (abs <= tol.AbsFloor) continue;
            double denom = Math.Max(Math.Abs((double)a[i]), Math.Abs((double)b[i]));
            if (tol.Rel > 0 && denom > 0 && abs / denom <= tol.Rel) continue;
            return false;
        }
        return true;
    }

    /// <summary>Bit-exact equality of two float arrays (for same-engine determinism checks).</summary>
    public static bool BitExact(ReadOnlySpan<float> a, ReadOnlySpan<float> b, out int firstDiff)
    {
        firstDiff = -1;
        if (a.Length != b.Length) return false;
        for (int i = 0; i < a.Length; i++)
        {
            if (BitConverter.SingleToInt32Bits(a[i]) != BitConverter.SingleToInt32Bits(b[i]))
            {
                // NaN payloads may differ harmlessly; treat any-NaN==any-NaN as equal.
                if (float.IsNaN(a[i]) && float.IsNaN(b[i])) continue;
                firstDiff = i; return false;
            }
        }
        return true;
    }

    /// <summary>Round a double-precision oracle result down to float for comparison against a
    /// float engine result. The oracle is the "truth" each engine's float result is bounded against.</summary>
    public static float[] ToFloat(ReadOnlySpan<double> d)
    {
        var f = new float[d.Length];
        for (int i = 0; i < d.Length; i++) f[i] = (float)d[i];
        return f;
    }
}
#endif
