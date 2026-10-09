using System;
using System.Threading;
using AiDotNet.Tensors.Helpers;
using Xunit;

namespace AiDotNet.Tensors.Tests.Helpers;

/// <summary>
/// The dispatching thread waits for the woken workers with a bounded spin on the completion count, then the
/// completion event. Execute must return only after EVERY chunk has run, both when the workers finish inside
/// the spin and when a straggler outlasts it, and a completion signal left over from one dispatch (the spin can
/// return before the last worker sets the event) must never end a later dispatch early.
/// </summary>
[Collection("CompilationGlobalState")]
public sealed class PersistentExecutorCompletionWaitTests
{
    [Fact]
    public void Execute_ReturnsOnlyAfterEveryChunk_WhetherWorkersFinishInsideOrAfterTheSpin()
    {
        int chunks = Math.Min(8, Math.Max(2, Environment.ProcessorCount));
        int priorDop = CpuParallelSettings.MaxDegreeOfParallelism;
        try
        {
            CpuParallelSettings.MaxDegreeOfParallelism = Math.Max(2, Math.Min(chunks, Environment.ProcessorCount));
            for (int it = 0; it < 600; it++)
            {
                var done = new int[chunks];
                // Every fourth dispatch has a straggler that outlasts the spin budget (2 ms vs ~100 us), so the
                // dispatcher must fall through to the blocking wait; the dispatch right after it is a fast one,
                // which is where a stale completion signal from the straggler's dispatch would end it early.
                bool straggler = it % 4 == 0;
                PersistentParallelExecutor.Instance.Execute(chunks, c =>
                {
                    if (straggler && c == chunks - 1) Thread.Sleep(2);
                    else if (c != 0) Thread.SpinWait(200);
                    Volatile.Write(ref done[c], 1);
                });
                for (int c = 0; c < chunks; c++)
                    Assert.True(Volatile.Read(ref done[c]) == 1,
                        $"dispatch {it} (straggler={straggler}) returned before chunk {c} of {chunks} ran");
            }
        }
        finally
        {
            CpuParallelSettings.MaxDegreeOfParallelism = priorDop;
        }
    }
}
