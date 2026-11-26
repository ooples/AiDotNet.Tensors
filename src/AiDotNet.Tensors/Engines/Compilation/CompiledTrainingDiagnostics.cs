// Copyright (c) CrowdTrainer LLC. All rights reserved.
using System;
using System.Threading;

namespace AiDotNet.Tensors.Engines.Compilation;

/// <summary>
/// Process-wide diagnostic traces of compiled training. All are off by default; each writes to
/// <see cref="System.Diagnostics.Trace"/> (or, for the graph-step trace, to files under the temp directory) and costs
/// time, so turn them on only while investigating.
/// </summary>
/// <remarks>
/// The older <c>AIDOTNET_GRAPHSTEP_DIAG=1</c> environment variable is deprecated: it is read only while
/// <see cref="TraceGraphSteps"/> has never been set, with a warning, and a set <see cref="TraceGraphSteps"/> always wins
/// over it.
/// </remarks>
public static class CompiledTrainingDiagnostics
{
    private const string LegacyGraphStepVariable = "AIDOTNET_GRAPHSTEP_DIAG";
    private static int _traceGraphSteps = -1;   // -1: never set, the deprecated variable decides
    private static int _legacyGraphSteps = -1;  // -1: not read yet
    private static int _legacyWarned;
    private static int _traceIdleLoans;
    private static int _recordCaptureAllocationStacks;

    /// <summary>
    /// Traces each captured training step: the graph's segments, the memory a capture pinned (allocated during the
    /// capture vs existing before it), and the checkpoint schedule.
    /// </summary>
    public static bool TraceGraphSteps
    {
        get => GraphStepTraceEnabled;
        set => Volatile.Write(ref _traceGraphSteps, value ? 1 : 0);
    }

    /// <summary>Traces idle step-memory loans: what is lent between steps, and where a loan had to be moved.</summary>
    public static bool TraceIdleLoans
    {
        get => Volatile.Read(ref _traceIdleLoans) != 0;
        set => Volatile.Write(ref _traceIdleLoans, value ? 1 : 0);
    }

    /// <summary>
    /// Records the call stack of every device allocation made while a CUDA graph is being captured, so a capture that
    /// pins memory can be traced to its allocation site. Expensive: one stack trace per allocation.
    /// </summary>
    public static bool RecordCaptureAllocationStacks
    {
        get => Volatile.Read(ref _recordCaptureAllocationStacks) != 0;
        set => Volatile.Write(ref _recordCaptureAllocationStacks, value ? 1 : 0);
    }

    internal static bool GraphStepTraceEnabled
    {
        get
        {
            int set = Volatile.Read(ref _traceGraphSteps);
            if (set >= 0) return set != 0;
            int legacy = Volatile.Read(ref _legacyGraphSteps);
            if (legacy < 0)
            {
                legacy = Environment.GetEnvironmentVariable(LegacyGraphStepVariable) == "1" ? 1 : 0;
                Volatile.Write(ref _legacyGraphSteps, legacy);
            }
            if (legacy == 0) return false;
            if (Interlocked.Exchange(ref _legacyWarned, 1) == 0)
                System.Diagnostics.Trace.TraceWarning(
                    $"{LegacyGraphStepVariable} is deprecated and will be removed: set {nameof(CompiledTrainingDiagnostics)}."
                    + $"{nameof(TraceGraphSteps)} instead, which wins over it once set.");
            return true;
        }
    }

    /// <summary>Tests: back to never set, and forgets the deprecated variable's cached reading and the warning.</summary>
    internal static void ResetForTests()
    {
        Volatile.Write(ref _traceGraphSteps, -1);
        Volatile.Write(ref _legacyGraphSteps, -1);
        Volatile.Write(ref _legacyWarned, 0);
        Volatile.Write(ref _traceIdleLoans, 0);
        Volatile.Write(ref _recordCaptureAllocationStacks, 0);
    }
}
