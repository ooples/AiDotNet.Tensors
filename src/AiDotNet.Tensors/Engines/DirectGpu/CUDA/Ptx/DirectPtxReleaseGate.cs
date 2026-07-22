#if NET5_0_OR_GREATER
using System;
using System.Collections.Generic;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// Selects which error metric the release gate enforces. Bounded-output
/// kernels (softmax) use <see cref="Absolute"/>. Reduction/scan kernels emit
/// an output of magnitude O(columns * value), so their absolute error scales
/// with the operand while the meaningful accuracy signal is relative; those use
/// <see cref="Relative"/>.
/// </summary>
internal enum DirectPtxErrorMetric
{
    Absolute,
    Relative
}

internal readonly record struct DirectPtxPerformanceEvidence(
    double MedianMicroseconds,
    double P95Microseconds,
    long ManagedBytesPerCall,
    long TemporaryDeviceBytes,
    double MaxAbsoluteError,
    int LocalBytesPerThread,
    int IndependentRuns)
{
    /// <summary>
    /// Maximum relative error against the FP64 oracle. Only consulted when the
    /// governing policy sets <see cref="DirectPtxErrorMetric.Relative"/>;
    /// defaults to zero so existing absolute-metric evidence is unaffected.
    /// </summary>
    internal double MaxRelativeError { get; init; } = 0.0;
}

internal readonly record struct DirectPtxReleaseGatePolicy(
    double MinimumMedianSpeedup,
    double MaximumP95RegressionFraction,
    long MaximumManagedBytesPerCall,
    long MaximumTemporaryDeviceBytes,
    double MaximumAbsoluteError,
    int RequiredIndependentRuns)
{
    /// <summary>Which error metric to enforce; absolute by default.</summary>
    internal DirectPtxErrorMetric ErrorMetric { get; init; } = DirectPtxErrorMetric.Absolute;

    /// <summary>
    /// Relative-error ceiling, enforced only when <see cref="ErrorMetric"/> is
    /// <see cref="DirectPtxErrorMetric.Relative"/>.
    /// </summary>
    internal double MaximumRelativeError { get; init; } = 5e-5;

    internal static DirectPtxReleaseGatePolicy ProductionDefault => new(
        MinimumMedianSpeedup: 1.10,
        MaximumP95RegressionFraction: 0.0,
        MaximumManagedBytesPerCall: 0,
        MaximumTemporaryDeviceBytes: 0,
        MaximumAbsoluteError: 5e-5,
        RequiredIndependentRuns: 3);

    internal DirectPtxReleaseDecision Evaluate(
        in DirectPtxPerformanceEvidence candidate,
        in DirectPtxPerformanceEvidence bestCompetitor)
    {
        var failures = new List<string>();
        double speedup = bestCompetitor.MedianMicroseconds / candidate.MedianMicroseconds;
        if (!double.IsFinite(speedup) || speedup < MinimumMedianSpeedup)
            failures.Add($"median-speedup={speedup:F3}x<{MinimumMedianSpeedup:F3}x");
        double p95Limit = bestCompetitor.P95Microseconds * (1.0 + MaximumP95RegressionFraction);
        if (candidate.P95Microseconds > p95Limit)
            failures.Add($"p95={candidate.P95Microseconds:F3}us>{p95Limit:F3}us");
        if (candidate.ManagedBytesPerCall > MaximumManagedBytesPerCall)
            failures.Add($"managed-bytes={candidate.ManagedBytesPerCall}>{MaximumManagedBytesPerCall}");
        if (candidate.TemporaryDeviceBytes > MaximumTemporaryDeviceBytes)
            failures.Add($"temporary-device-bytes={candidate.TemporaryDeviceBytes}>{MaximumTemporaryDeviceBytes}");
        if (ErrorMetric == DirectPtxErrorMetric.Relative)
        {
            if (!double.IsFinite(candidate.MaxRelativeError) ||
                candidate.MaxRelativeError > MaximumRelativeError)
                failures.Add(
                    $"max-rel-error={candidate.MaxRelativeError:G6}>{MaximumRelativeError:G6}");
        }
        else if (candidate.MaxAbsoluteError > MaximumAbsoluteError)
        {
            failures.Add($"max-error={candidate.MaxAbsoluteError:G6}>{MaximumAbsoluteError:G6}");
        }
        if (candidate.LocalBytesPerThread != 0)
            failures.Add($"local-bytes/thread={candidate.LocalBytesPerThread}");
        if (candidate.IndependentRuns < RequiredIndependentRuns)
            failures.Add($"independent-runs={candidate.IndependentRuns}<{RequiredIndependentRuns}");
        return new DirectPtxReleaseDecision(failures.Count == 0, speedup, failures);
    }
}

internal sealed record DirectPtxReleaseDecision(
    bool Passed,
    double MedianSpeedup,
    IReadOnlyList<string> Failures);
#endif
