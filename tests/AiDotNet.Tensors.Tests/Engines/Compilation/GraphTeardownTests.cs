using System;
using System.Diagnostics;
using System.IO;
using AiDotNet.Tensors.Engines.Compilation;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

/// <summary>Tests that add to the process-wide <see cref="Trace.Listeners"/>, so no other test traces into them.</summary>
[CollectionDefinition(Name, DisableParallelization = true)]
public sealed class TraceListenersSerial
{
    public const string Name = "TraceListenersSerial";
}

/// <summary>
/// CompiledInferencePlan's CUDA graph teardown runs each step best-effort (#1558): a step that throws is traced, with
/// the action and the exception, and the next step still runs. It used to be <c>catch { }</c>, so a failed
/// EndGraphLifetime left eviction suspended with no record.
/// </summary>
[Collection(TraceListenersSerial.Name)]
public class GraphTeardownTests
{
    private sealed class ThrowingListener : TraceListener
    {
        public override void Write(string? message) => throw new IOException("listener broken");
        public override void WriteLine(string? message) => throw new IOException("listener broken");
    }

    [Fact]
    public void AThrowingTraceListener_DoesNotStopTeardown()
    {
        // A listener that throws while the warning is written must not escape: teardown has to carry on to its next
        // step. The lost warning is counted, with the listener's exception.
        var listener = new ThrowingListener();
        Trace.Listeners.Add(listener);
        bool nextRan = false;
        int before = CompiledInferencePlan<float>.UnreportedTeardownFailures;
        try
        {
            CompiledInferencePlan<float>.BestEffortTeardown(
                () => throw new InvalidOperationException("device lost"), "end the graph lifetime on dispose");
            CompiledInferencePlan<float>.BestEffortTeardown(() => nextRan = true, "destroy the captured graph");
        }
        finally
        {
            Trace.Listeners.Remove(listener);
        }

        Assert.True(nextRan);
        Assert.Equal(before + 1, CompiledInferencePlan<float>.UnreportedTeardownFailures);
        Assert.IsType<IOException>(CompiledInferencePlan<float>.LastUnreportedTeardownFailure);
    }

    [Fact]
    public void AFailingTeardownStepIsTracedAndTheNextStepStillRuns()
    {
        var log = new StringWriter();
        var listener = new TextWriterTraceListener(log);
        Trace.Listeners.Add(listener);
        bool nextRan = false;
        try
        {
            CompiledInferencePlan<float>.BestEffortTeardown(
                () => throw new InvalidOperationException("device lost"), "end the graph lifetime on dispose");
            CompiledInferencePlan<float>.BestEffortTeardown(() => nextRan = true, "destroy the captured graph");
            listener.Flush();
        }
        finally
        {
            Trace.Listeners.Remove(listener);
        }

        Assert.True(nextRan);
        string traced = log.ToString();
        Assert.Contains("could not end the graph lifetime on dispose", traced);
        Assert.Contains("device lost", traced);
        Assert.DoesNotContain("destroy the captured graph", traced);
    }
}
