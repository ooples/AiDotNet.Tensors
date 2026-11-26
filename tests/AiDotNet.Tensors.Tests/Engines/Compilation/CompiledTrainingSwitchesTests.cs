using System;
using System.Threading;
using AiDotNet.Tensors.Engines.Compilation;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

/// <summary>
/// The compiled-training switches: typed diagnostics options (the one legacy environment variable is a deprecated
/// fallback a set option overrides), and the mixed-precision trace switch, which is per thread.
/// </summary>
[Collection(MixedPrecisionTestCollection.Name)]
public sealed class CompiledTrainingSwitchesTests
{
    [Fact]
    public void The_graph_step_trace_reads_the_deprecated_variable_only_until_the_option_is_set()
    {
        const string Legacy = "AIDOTNET_GRAPHSTEP_DIAG";
        string? previous = Environment.GetEnvironmentVariable(Legacy);
        CompiledTrainingDiagnostics.ResetForTests();
        try
        {
            Environment.SetEnvironmentVariable(Legacy, "1");
            Assert.True(CompiledTrainingDiagnostics.TraceGraphSteps);
            CompiledTrainingDiagnostics.TraceGraphSteps = false;
            Assert.False(CompiledTrainingDiagnostics.TraceGraphSteps);
            CompiledTrainingDiagnostics.TraceIdleLoans = true;
            Assert.True(CompiledTrainingDiagnostics.TraceIdleLoans);
        }
        finally
        {
            Environment.SetEnvironmentVariable(Legacy, previous);
            CompiledTrainingDiagnostics.ResetForTests();
        }
    }

    /// <summary>
    /// A mixed-precision trace sets the switch for its own thread only. It was a plain static, so a concurrent trace
    /// on another thread recorded Half activation nodes into a graph whose plan cannot consume them.
    /// </summary>
    [Fact]
    public void The_mixed_precision_trace_switch_is_not_seen_by_other_threads()
    {
        var previous = MixedPrecisionEmit.TestOverrideEnabled;
        MixedPrecisionEmit.TestOverrideEnabled = true;
        try
        {
            Assert.True(MixedPrecisionEmit.Fp16ActivationStorageEnabled);
            bool seenElsewhere = true;
            var other = new Thread(() => seenElsewhere = MixedPrecisionEmit.Fp16ActivationStorageEnabled);
            other.Start();
            other.Join();
            Assert.False(seenElsewhere);
        }
        finally { MixedPrecisionEmit.TestOverrideEnabled = previous; }
    }
}
