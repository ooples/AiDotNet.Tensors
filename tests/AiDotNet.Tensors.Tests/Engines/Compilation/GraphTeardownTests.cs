using System;
using System.Diagnostics;
using System.IO;
using AiDotNet.Tensors.Engines.Compilation;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

/// <summary>
/// CompiledInferencePlan's CUDA graph teardown runs each step best-effort (#1558): a step that throws is traced, with
/// the action and the exception, and the next step still runs. It used to be <c>catch { }</c>, so a failed
/// EndGraphLifetime left eviction suspended with no record.
/// </summary>
public class GraphTeardownTests
{
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
