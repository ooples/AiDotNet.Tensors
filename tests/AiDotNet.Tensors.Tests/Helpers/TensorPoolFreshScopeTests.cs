using System;
using System.Threading;
using AiDotNet.Tensors.Helpers;
using Xunit;

namespace AiDotNet.Tensors.Tests.Helpers;

public class TensorPoolFreshScopeTests
{
    [Fact]
    public void Scope_AppliesToTheCallingThreadOnly_AndRestoresOnDispose()
    {
        bool processDefault = TensorPool.ForceFreshAllocations;
        bool? seenElsewhere = null;
        using (TensorPool.ForceFreshAllocationsScope(!processDefault))
        {
            Assert.Equal(!processDefault, TensorPool.ForceFreshAllocations);
            var other = new Thread(() => seenElsewhere = TensorPool.ForceFreshAllocations);
            other.Start();
            other.Join();

            using (TensorPool.ForceFreshAllocationsScope(processDefault))
                Assert.Equal(processDefault, TensorPool.ForceFreshAllocations);
            Assert.Equal(!processDefault, TensorPool.ForceFreshAllocations);
        }
        Assert.Equal(processDefault, seenElsewhere);
        Assert.Equal(processDefault, TensorPool.ForceFreshAllocations);
    }

    /// <summary>
    /// An op the scope's thread issues may run its chunks on the library's intra-op workers. Those chunks follow the
    /// issuing thread's policy, as PyTorch's parallel_for carries the caller's thread-local state into its workers.
    /// Before, a chunk on a worker followed the process default, so a compiled step's first trace could take pooled,
    /// uninitialized storage on a worker while the scope forced fresh storage on the calling thread.
    /// </summary>
    [SkippableFact]
    public void Scope_reaches_the_intra_op_workers_running_the_threads_chunks_and_only_them()
    {
        Skip.If(CpuParallelSettings.MaxDegreeOfParallelism < 2, "Needs two or more worker threads.");
        bool processDefault = TensorPool.ForceFreshAllocations;
        const int Chunks = 64;
        var seen = new bool[Chunks];
        var thread = new int[Chunks];
        int caller = Environment.CurrentManagedThreadId;
        void Run() => PersistentParallelExecutor.Instance.Execute(Chunks, c =>
        {
            seen[c] = TensorPool.ForceFreshAllocations;
            thread[c] = Environment.CurrentManagedThreadId;
            Thread.SpinWait(20000);   // keep the chunks long enough for the workers to take some
        });

        using (TensorPool.ForceFreshAllocationsScope(!processDefault)) Run();
        Assert.Contains(thread, id => id != caller);
        for (int c = 0; c < Chunks; c++)
            Assert.True(seen[c] == !processDefault, $"chunk {c} on thread {thread[c]} (caller {caller}) did not see the scope");

        // After the scope the workers are back on the process default.
        Run();
        for (int c = 0; c < Chunks; c++)
            Assert.True(seen[c] == processDefault, $"chunk {c} on thread {thread[c]} kept the scope after it closed");
    }

    [SkippableFact]
    public void Scope_reaches_the_workers_of_a_per_participant_dispatch()
    {
        Skip.If(CpuParallelSettings.MaxDegreeOfParallelism < 2, "Needs two or more worker threads.");
        bool processDefault = TensorPool.ForceFreshAllocations;
        int inits = 0, initsSeeingScope = 0, chunksSeeingScope = 0;
        const int Chunks = 64;
        using (TensorPool.ForceFreshAllocationsScope(!processDefault))
            PersistentParallelExecutor.Instance.Execute<int>(Chunks, 0,
                () =>
                {
                    Interlocked.Increment(ref inits);
                    if (TensorPool.ForceFreshAllocations == !processDefault) Interlocked.Increment(ref initsSeeingScope);
                    return 0;
                },
                (c, _) =>
                {
                    if (TensorPool.ForceFreshAllocations == !processDefault) Interlocked.Increment(ref chunksSeeingScope);
                    Thread.SpinWait(20000);
                },
                _ => { });
        Assert.Equal(inits, initsSeeingScope);
        Assert.Equal(Chunks, chunksSeeingScope);
    }
}
