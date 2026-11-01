// Copyright (c) CrowdTrainer LLC. All rights reserved.

using System;
using System.Threading;
using AiDotNet.Tensors.Engines;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>GpuScope's per-thread depth bookkeeping; no device needed (an engine-less scope), so it runs on every lane.</summary>
public class GpuScopeDepthTests
{
    [Fact]
    public void Nested_scopes_stay_active_until_the_outermost_closes()
    {
        Assert.False(GpuScope.IsActive);
        var outer = new GpuScope();
        var inner = new GpuScope();
        Assert.True(GpuScope.IsActive);
        inner.Dispose();
        Assert.True(GpuScope.IsActive);
        outer.Dispose();
        Assert.False(GpuScope.IsActive);
    }

    [Fact]
    public void An_extra_dispose_does_not_close_a_scope_still_open()
    {
        var first = new GpuScope();
        first.Dispose();
        first.Dispose();   // depth is already 0: stays 0
        var second = new GpuScope();
        Assert.True(GpuScope.IsActive);
        second.Dispose();
        Assert.False(GpuScope.IsActive);
    }

    [Fact]
    public void A_scope_is_visible_only_on_the_thread_that_opened_it()
    {
        using var scope = new GpuScope();
        bool seenElsewhere = true;
        OnOtherThread(() => seenElsewhere = GpuScope.IsActive);
        Assert.False(seenElsewhere);
        // A dispose on another thread finds no entry there and leaves this thread's scope open.
        OnOtherThread(scope.Dispose);
        Assert.True(GpuScope.IsActive);
    }

    // A dedicated thread: a task could be inlined onto the waiting thread, which would defeat the test.
    private static void OnOtherThread(Action action)
    {
        var thread = new Thread(() => action());
        thread.Start();
        thread.Join();
    }
}
