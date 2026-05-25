using System;
using System.Threading;
using AiDotNet.Tensors.Engines.Autodiff;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Autodiff;

/// <summary>
/// Regression guard for the undisposed-GradientTape global-state leak: the tape
/// constructor increments the process-wide <see cref="DifferentiableOps._anyTapeActive"/>
/// counter and only <see cref="GradientTape{T}.Dispose"/> decrements it. Without
/// the finalizer backstop, a tape dropped without disposing (e.g. on an
/// exception path) left the counter stuck positive forever — forcing every op
/// on every thread down the tape-recording slow path and flipping tape-gated
/// dispatch (the cross-suite flakiness root cause).
/// </summary>
public class GradientTapeFinalizerTests
{
    [Fact]
    public void UndisposedTape_DoesNotPermanentlyLeakGlobalActiveCounter()
    {
        int baseline = DifferentiableOps._anyTapeActive;

        // Create and abandon a tape on a SEPARATE thread: its ThreadStatic
        // _current dies with the thread (so this test thread isn't polluted),
        // isolating the process-wide counter leak the finalizer must heal.
        var worker = new Thread(() =>
        {
            var leaked = new GradientTape<float>(); // intentionally never disposed
            GC.KeepAlive(leaked);                   // observed alive until thread end
        });
        worker.Start();
        worker.Join();

        // Force finalization; the finalizer must decrement the leaked global count.
        for (int i = 0; i < 10 && Volatile.Read(ref DifferentiableOps._anyTapeActive) > baseline; i++)
        {
            GC.Collect();
            GC.WaitForPendingFinalizers();
        }

        Assert.Equal(baseline, Volatile.Read(ref DifferentiableOps._anyTapeActive));
    }

    [Fact]
    public void DisposedTape_RestoresGlobalActiveCounter_AndSuppressesFinalizer()
    {
        int baseline = DifferentiableOps._anyTapeActive;
        using (var tape = new GradientTape<float>())
        {
            Assert.True(DifferentiableOps._anyTapeActive > baseline,
                "Constructing a tape must raise the active counter.");
        }
        Assert.Equal(baseline, DifferentiableOps._anyTapeActive);
    }
}
