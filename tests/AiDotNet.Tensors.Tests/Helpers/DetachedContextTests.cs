// Copyright (c) CrowdTrainer LLC. All rights reserved.
using System;
using System.Threading;
using AiDotNet.Tensors.Helpers;
using Xunit;

namespace AiDotNet.Tensors.Tests.Helpers;

/// <summary>
/// The library's long-lived threads must not keep their creator's AsyncLocal values alive. AiDotNet
/// creates the worker pool inside a forward that runs under an AsyncLocal layer observer, which holds
/// every activation of that forward.
/// </summary>
/// <remarks>Serialized with the pool-routing tests: one test sets the process-wide
/// <see cref="CpuParallelSettings.UseCooperativePool"/>.</remarks>
[Collection("BlasManaged-Pool-Serial")]
public class DetachedContextTests
{
    private static readonly AsyncLocal<object?> s_local = new();

    [Fact]
    public void Start_ThreadDoesNotSeeTheCallersAsyncLocal()
    {
        object? seen = "unset";
        s_local.Value = new object();
        try
        {
            var t = new Thread(() => seen = s_local.Value) { IsBackground = true };
            DetachedContext.Start(t);
            t.Join();
        }
        finally
        {
            s_local.Value = null;
        }

        Assert.Null(seen);
    }

    [Fact]
    public void Start_WorksWhenFlowIsAlreadySuppressed()
    {
        using (ExecutionContext.SuppressFlow())
        {
            var t = new Thread(() => { }) { IsBackground = true };
            DetachedContext.Start(t);
            t.Join();
        }
    }

    [SkippableTheory]
    [InlineData(false)]
    [InlineData(true)]
    public void ParallelExecutor_CreatedUnderAnAsyncLocal_WorkersDoNotSeeIt(bool dispatchUnderAnother)
    {
        Skip.If(CpuParallelSettings.MaxDegreeOfParallelism < 2, "a dispatch wakes no worker at MaxDegreeOfParallelism 1");

        var creatorValue = new object();
        PersistentParallelExecutor executor;
        s_local.Value = creatorValue;
        try
        {
            // A fresh pool: the process-wide Instance may already exist.
            executor = (PersistentParallelExecutor)Activator.CreateInstance(typeof(PersistentParallelExecutor), nonPublic: true)!;
        }
        finally
        {
            s_local.Value = null;
        }

        // Observed from inside the workers' own callbacks: a worker whose captured context holds the creator's value
        // reads it there. Dispatch until a worker has run a chunk; the caller can claim every chunk of one dispatch.
        int caller = Environment.CurrentManagedThreadId;
        int workerChunks = 0, sawCreatorValue = 0;
        // Execute dispatches through Parallel.For when the cooperative pool is off (AIDOTNET_COOP_POOL=0), and those
        // callbacks are not the executor's workers: pin it on so the assertions are about the persistent pool.
        bool savedCoop = CpuParallelSettings.UseCooperativePool;
        CpuParallelSettings.UseCooperativePool = true;
        if (dispatchUnderAnother) s_local.Value = new object();
        try
        {
            for (int attempt = 0; attempt < 200 && Volatile.Read(ref workerChunks) == 0; attempt++)
                executor.Execute(64, _ =>
                {
                    if (Environment.CurrentManagedThreadId == caller) return;
                    Interlocked.Increment(ref workerChunks);
                    if (ReferenceEquals(s_local.Value, creatorValue)) Interlocked.Increment(ref sawCreatorValue);
                    Thread.SpinWait(2000);
                });
        }
        finally
        {
            s_local.Value = null;
            CpuParallelSettings.UseCooperativePool = savedCoop;
        }

        Assert.True(workerChunks > 0, "no executor worker ran a chunk, so nothing was verified");
        Assert.True(sawCreatorValue == 0,
            $"{sawCreatorValue} of {workerChunks} worker chunks saw the executor creator's AsyncLocal value");
    }
}