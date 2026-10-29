using System.Threading;

namespace AiDotNet.Tensors.Helpers;

/// <summary>
/// Starts a long-lived background thread without the creating thread's execution context.
/// </summary>
/// <remarks>
/// <c>Thread.Start</c> captures the caller's <see cref="ExecutionContext"/>, so a thread that lives for the process keeps
/// every <see cref="AsyncLocal{T}"/> value its creator had set alive, and every op it later runs sees them. The worker
/// pools are created lazily by whatever op first needs them: AiDotNet's first forward runs under a layer-forward
/// observer (an AsyncLocal), and the parallel workers it sized then held that observer's record of a whole training
/// batch's activations for the rest of the process (52 MB on DenseNet-121 at batch 64).
/// </remarks>
internal static class DetachedThread
{
    /// <summary>Starts <paramref name="thread"/> with execution-context flow suppressed.</summary>
    internal static void Start(Thread thread)
    {
        using (ExecutionContext.SuppressFlow())
            thread.Start();
    }
}
