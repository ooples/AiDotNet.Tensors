// Copyright (c) CrowdTrainer LLC. All rights reserved.
using System;
using System.Reflection;
using System.Threading;
using AiDotNet.Tensors.Helpers;
using Xunit;

namespace AiDotNet.Tensors.Tests.Helpers;

/// <summary>
/// The library's long-lived threads must not keep their creator's AsyncLocal values alive. AiDotNet
/// creates the worker pool inside a forward that runs under an AsyncLocal layer observer, which holds
/// every activation of that forward.
/// </summary>
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

    [Theory]
    [InlineData(true, false)]
    [InlineData(false, true)]
    [InlineData(true, true)]
    public void ParallelExecutor_CreatedAndDispatchedUnderAnAsyncLocal_WorkersDoNotCarryIt(bool createUnder, bool dispatchUnder)
    {
        PersistentParallelExecutor executor;
        if (createUnder) s_local.Value = new byte[1024];
        try
        {
            // A fresh pool: the process-wide Instance may already exist.
            executor = (PersistentParallelExecutor)Activator.CreateInstance(typeof(PersistentParallelExecutor), nonPublic: true)!;
            if (dispatchUnder) s_local.Value = new byte[1024];
            for (int i = 0; i < 4; i++)
                executor.Execute(64, _ => Thread.SpinWait(2000));
        }
        finally
        {
            s_local.Value = null;
        }

        var workers = (Thread[])typeof(PersistentParallelExecutor)
            .GetField("_workers", BindingFlags.NonPublic | BindingFlags.Instance)!.GetValue(executor)!;
        // The context a thread keeps for its life. Its name is a runtime detail, so look it up rather than assume it.
        var contextField = typeof(Thread).GetField("_executionContext", BindingFlags.NonPublic | BindingFlags.Instance);
        if (contextField is null) return;   // a runtime without the field (net471): the Start test above covers it

        int carrying = 0;
        foreach (var worker in workers)
        {
            if (contextField.GetValue(worker) is not ExecutionContext context) continue;
            object? seen = null;
            ExecutionContext.Run(context.CreateCopy(), _ => seen = s_local.Value, null);
            if (seen is not null) carrying++;
        }

        Assert.True(carrying == 0, $"{carrying} of {workers.Length} executor workers carry their creator's AsyncLocal value");
    }
}
