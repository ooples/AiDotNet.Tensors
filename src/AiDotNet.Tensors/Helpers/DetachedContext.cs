// Copyright (c) CrowdTrainer LLC. All rights reserved.
using System;
using System.Threading;

namespace AiDotNet.Tensors.Helpers;

/// <summary>
/// Creates the library's long-lived threads, timers and tasks without the caller's
/// <see cref="ExecutionContext"/>.
/// </summary>
/// <remarks>
/// <para>
/// <c>Thread.Start</c>, <c>new Timer(...)</c> and <c>Task.Factory.StartNew</c> capture the creating
/// thread's ExecutionContext and keep it for the life of the thread, timer or task. Our worker pools
/// are created lazily on the first parallel call, so whatever <see cref="AsyncLocal{T}"/> values that
/// first caller had stay reachable for the rest of the process.
/// </para>
/// <para>
/// Measured case: AiDotNet runs a model's first forward under an AsyncLocal layer observer that
/// records each layer's input and output tensors. The executor's workers started during that forward
/// and captured the observer, so after the forward DenseNet-121 still held every b16 activation:
/// a 1.39 GB GC heap at CPU inference.
/// </para>
/// </remarks>
internal static class DetachedContext
{
    /// <summary>Runs <paramref name="create"/> with ExecutionContext flow suppressed.</summary>
    public static T Run<T>(Func<T> create)
    {
        if (ExecutionContext.IsFlowSuppressed()) return create();
        using (ExecutionContext.SuppressFlow()) return create();
    }

    /// <summary>Starts <paramref name="thread"/> without the caller's ExecutionContext.</summary>
    public static void Start(Thread thread)
    {
        if (ExecutionContext.IsFlowSuppressed()) { thread.Start(); return; }
        using (ExecutionContext.SuppressFlow()) thread.Start();
    }
}
