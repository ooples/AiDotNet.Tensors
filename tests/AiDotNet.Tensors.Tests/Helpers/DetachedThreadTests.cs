using System;
using System.Runtime.CompilerServices;
using System.Threading;
using AiDotNet.Tensors.Helpers;
using Xunit;

namespace AiDotNet.Tensors.Tests.Helpers;

/// <summary>
/// The worker pools start their long-lived threads through <see cref="DetachedThread"/>, which must not hand them the
/// creating thread's execution context: a thread started while an AsyncLocal holds a value must neither see that value
/// nor keep it alive (a layer-forward observer active when the pools were first sized held a training batch's
/// activations for the process's life).
/// </summary>
public sealed class DetachedThreadTests
{
    private static readonly AsyncLocal<object?> Ambient = new();

    [Fact]
    public void AStartedThread_NeitherSeesNorRetainsTheCreatorsAsyncLocals()
    {
        using var release = new ManualResetEventSlim(false);
        using var observed = new ManualResetEventSlim(false);
        object? seen = "not read";
        var weak = StartWhileAmbientHoldsAValue(release, observed, value => seen = value);

        Assert.True(observed.Wait(TimeSpan.FromSeconds(30)), "the thread did not run");
        Assert.Null(seen);
        // The thread is still alive (parked on release), as a pool worker would be: the value must already be collectable.
        GC.Collect();
        GC.WaitForPendingFinalizers();
        GC.Collect();
        Assert.False(weak.IsAlive, "the running thread kept its creator's AsyncLocal value alive");
        release.Set();
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    private static WeakReference StartWhileAmbientHoldsAValue(ManualResetEventSlim release, ManualResetEventSlim observed,
        Action<object?> report)
    {
        var payload = new byte[1024];
        var weak = new WeakReference(payload);
        Ambient.Value = payload;
        try
        {
            DetachedThread.Start(new Thread(() =>
            {
                report(Ambient.Value);
                observed.Set();
                release.Wait();
            }) { IsBackground = true });
        }
        finally
        {
            Ambient.Value = null;
        }
        return weak;
    }
}
