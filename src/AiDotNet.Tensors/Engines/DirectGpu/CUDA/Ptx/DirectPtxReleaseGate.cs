#if NET5_0_OR_GREATER
using System;
using System.Collections.Generic;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

internal readonly record struct DirectPtxPerformanceEvidence(
    double MedianMicroseconds,
    double P95Microseconds,
    long ManagedBytesPerCall,
    long TemporaryDeviceBytes,
    double MaxAbsoluteError,
    int LocalBytesPerThread,
    int IndependentRuns);

internal readonly record struct DirectPtxReleaseGatePolicy(
    double MinimumMedianSpeedup,
    double MaximumP95RegressionFraction,
    long MaximumManagedBytesPerCall,
    long MaximumTemporaryDeviceBytes,
    double MaximumAbsoluteError,
    int RequiredIndependentRuns)
{
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
        if (candidate.MaxAbsoluteError > MaximumAbsoluteError)
            failures.Add($"max-error={candidate.MaxAbsoluteError:G6}>{MaximumAbsoluteError:G6}");
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
