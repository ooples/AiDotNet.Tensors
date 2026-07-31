// Copyright (c) AiDotNet. All rights reserved.
// A version stamp for how a number was measured.
//
// The measurement protocol changed eleven times during this project, and each change
// silently invalidated every number recorded before it:
//
//   v1 -> v2  the estimator. median(A)/median(B) let clock drift during a run leak into
//             the ratio; an A/A comparison whose true value is exactly 1.000x read
//             2.5-3.8% off. Replaced by a paired within-sample ratio, which took the
//             noise floor to 1.05%.
//   v2 -> v3  clock locking. The SM clock was measured swinging 2025 -> 1770 MHz INSIDE
//             a single kernel's three runs, up to 12.6%, which is where the intermittent
//             7.5% run spreads came from.
//   v3 -> v4  true fp32. PyTorch defaults to allow_tf32=True, which routes dense
//             convolution to tensor cores at a 10-bit mantissa -- a different operation
//             from the exact fp32 our kernels verify against an fp64 oracle.
//   v4 -> v5  conformance and stability. Head-to-head measured all A samples before B,
//             autotune used an unpaired host-stopwatch best-of-three, and both still stamped
//             the rows p4. Comparisons now form A/B inside each sample and refuse a result
//             unless A, B and the ratio each converge within the 5% spread gate. Cross-process
//             competitor rows carry and gate both spreads independently.
//   v5 -> v6  exact dispatch identity. The autotune cache was keyed by assembly MVID, so a
//             benchmark-only rebuild invalidated every winner, while competitor evidence
//             was only protocol-keyed and could still be accepted for the now-untuned
//             program. Candidate PTX sets now identify autotune rows, competitor evidence
//             binds the complete selected dispatch, and limiter evidence requires every
//             requested counter before it can name a bottleneck.
//   v6 -> v7  exact competitor geometry and phase-scoped counters. The generated
//             transposed convolution was corrected to the declared 28 -> 55 extent, but
//             the cuDNN lane still requested output_padding=1 and measured 28 -> 56.
//             Split-program profiles also mixed per-metric maxima from their partial and
//             combine launches. The competitor now measures the same operator, and every
//             split phase must have a complete counter set before the longest phase is
//             used for diagnosis.
//   v7 -> v8  reproducible cuDNN plan search. Fresh but individually stable processes
//             selected materially different cuDNN convolution plans (28.7 vs 38.6 us
//             for the same 1x1 shape). The competitor lane now runs three fresh plan
//             searches per shape, records their spread, and compares against the
//             strongest stable plan rather than whichever one a single process chose.
//   v8 -> v9  multi-strategy cuDNN search. Exhaustive benchmark_limit=0 unexpectedly
//             chose a 90 us weight-gradient plan where the default search repeatedly
//             found 41 us, so no single framework selector is treated as an oracle.
//             Four default, two exhaustive, and one deterministic-heuristic fresh
//             processes are searched; the fastest stable plan and its strategy are
//             recorded per shape.
//   v9 -> v10 recoverable stability windows. The stability loop accumulated a max-min
//             spread over every attempt, so one WDDM preemption made convergence
//             mathematically impossible: later clean samples could not remove the old
//             maximum. It now requires three consecutive agreeing samples, preserving
//             the 5% gate while allowing contaminated batches to age out.
//   v10 -> v11 cross-lane stability conformance. StableTimer implemented the consecutive
//              window, but the conveyor and Python competitor lane still took exactly
//              three runs; the conveyor also published the minimum while Python published
//              the median. Both lanes now retry up to fifteen runs, retain the latest
//              three, and publish their median under the same 5% gate. Full-suite generated
//              evidence is collected and contamination-retried per operation, with a capped
//              backoff, so one WDDM burst cannot discard twelve clean rows or phase-lock all
//              immediate retries into the same foreign workload. Post-suite C+G handling now
//              matches the release-evidence script: mixed-process admission is enforced before
//              timing, while the post boundary rejects compute-only work, unsafe temperature,
//              and sustained whole-device utilization. A bounded quiescence window prevents the
//              benchmark's own trailing NVIDIA utilization sample from becoming a false refusal.
//   v11 -> v12 uncertainty-aware winner arbitration. Each autotune candidate was paired with
//              the modelled baseline, but a challenger could replace the current winner after
//              clearing only its own paired spread. Independent ratio windows now combine the
//              incumbent and challenger spreads before changing a recorded schedule, so a 3%
//              median difference between two 2-4% windows remains evidence of a tie.
//
// Nothing marked the old numbers as stale, so they sat in documents and commit messages
// next to fresh ones looking equally authoritative. A number without its protocol is not
// a measurement, so every recorded figure now carries this stamp and a reader can tell
// at a glance whether two numbers are comparable.

using System;

namespace AiDotNet.Tensors.Engines.Compilation.Codegen.Ir;

/// <summary>Identifies how a performance number was obtained.</summary>
public static class CodegenMeasurementProtocol
{
    /// <summary>Smallest measured gain distinguishable from the harness noise floor.</summary>
    public const double AutotuneGainNoiseFloor = 1.0105;

    /// <summary>Largest relative FP32 accumulation deviation accepted by shared gates.</summary>
    public const double AccumulationTolerance = 2e-3;

    /// <summary>
    /// Current protocol version. Increment whenever a change makes new numbers
    /// incomparable with old ones, and add a line to the history in this file.
    /// </summary>
    public const int Version = 12;

    /// <summary>Short tag for manifests and tables, e.g. <c>p5</c>.</summary>
    public static string Tag => "p" + Version.ToString(System.Globalization.CultureInfo.InvariantCulture);

    /// <summary>One-line description of what the current protocol requires.</summary>
    public const string Description =
        "paired within-sample ratios; batched timed regions; clock-drift and <=5% spread gates; " +
        "true-fp32 CUDA-graph replay on both lanes; exact PTX-set autotune and dispatch-bound evidence; " +
        "exact competitor geometry; multi-strategy cuDNN plan search; " +
        "phase-scoped counter profiles; recoverable per-operation stability windows; " +
        "uncertainty-aware autotune winner arbitration";

    /// <summary>
    /// Minimum ratio-of-ratios required for an independently measured challenger to
    /// displace the current autotune winner.
    /// </summary>
    public static double RequiredIndependentCandidateGain(
        double incumbentRelativeSpread, double challengerRelativeSpread)
    {
        if (double.IsNaN(incumbentRelativeSpread) ||
            double.IsInfinity(incumbentRelativeSpread) || incumbentRelativeSpread < 0)
            throw new ArgumentOutOfRangeException(nameof(incumbentRelativeSpread));
        if (double.IsNaN(challengerRelativeSpread) ||
            double.IsInfinity(challengerRelativeSpread) || challengerRelativeSpread < 0)
            throw new ArgumentOutOfRangeException(nameof(challengerRelativeSpread));
        return Math.Max(
            AutotuneGainNoiseFloor,
            1.0 + incumbentRelativeSpread + challengerRelativeSpread);
    }

    /// <summary>
    /// Human-readable stamp to put beside a number.
    /// </summary>
    public static string Stamp(string device) =>
        Tag + " (" + Description + ") on " + (device ?? "unknown device");

    /// <summary>
    /// True when a number recorded under <paramref name="recordedVersion"/> can be
    /// compared with one taken now.
    /// </summary>
    public static bool IsComparable(int recordedVersion) => recordedVersion == Version;

    /// <summary>Explains why a stale number must not be compared.</summary>
    public static string ExplainStaleness(int recordedVersion)
    {
        if (IsComparable(recordedVersion)) return string.Empty;
        if (recordedVersion <= 0) return "recorded with no protocol stamp at all; treat as unusable";
        return "recorded under protocol p" + recordedVersion.ToString(System.Globalization.CultureInfo.InvariantCulture) +
               ", current is " + Tag + "; re-measure before comparing (" + Description + ")";
    }
}
