// Copyright (c) AiDotNet. All rights reserved.
// A version stamp for how a number was measured.
//
// The measurement protocol changed four times during this project, and each change
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
    /// <summary>
    /// Current protocol version. Increment whenever a change makes new numbers
    /// incomparable with old ones, and add a line to the history in this file.
    /// </summary>
    public const int Version = 5;

    /// <summary>Short tag for manifests and tables, e.g. <c>p5</c>.</summary>
    public static string Tag => "p" + Version.ToString(System.Globalization.CultureInfo.InvariantCulture);

    /// <summary>One-line description of what the current protocol requires.</summary>
    public const string Description =
        "paired within-sample ratios; batched timed regions; clock-drift and <=5% spread gates; " +
        "cross-process competitor separately gated at true fp32 under CUDA graphs";

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
