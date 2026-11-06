using System;
using System.Threading;
using System.Threading.Tasks;
using AiDotNet.Tensors.Engines.DirectGpu.HIP;
using Xunit;

namespace AiDotNet.Tensors.Tests.Interop;

public sealed class HipExportLifetimeTests
{
    // These tests exercise managed ownership arbitration without calling a driver or allocating device memory.
    private static HipBackend Owner()
    {
#pragma warning disable SYSLIB0050
        var owner = (HipBackend)System.Runtime.Serialization.FormatterServices.GetUninitializedObject(typeof(HipBackend));
#pragma warning restore SYSLIB0050
        GC.SuppressFinalize(owner);
        return owner;
    }

    [Fact]
    public void ReleasedBuffer_RefusesAnExportHold()
    {
        var buffer = new HipGpuBuffer(IntPtr.Zero, 4, Owner());
        buffer.Release();
        Assert.Null(buffer.HoldForExport());
    }

    [Fact]
    public void ExportedView_HoldsItsParentUntilTheLastDeleter()
    {
        int returns = 0;
        var owner = Owner();
        var parent = new HipGpuBuffer(IntPtr.Zero, 4, owner, _ => Interlocked.Increment(ref returns));
        var view = HipGpuBuffer.CreateView(parent, IntPtr.Zero, 4, owner);
        var first = view.HoldForExport();
        Assert.NotNull(first);
        var second = view.HoldForExport();
        Assert.NotNull(second);
        parent.Dispose();
        view.Dispose();
        Assert.Equal(0, returns);
        first.Dispose();
        Assert.Equal(0, returns);
        second.Dispose();
        second.Dispose();
        Assert.Equal(1, returns);
        parent.Release();
    }

    [Fact]
    public async Task ExportAndDisposeRace_EitherPinsTheLeaseOrRefusesIt()
    {
        var owner = Owner();
        for (int iteration = 0; iteration < 200; iteration++)
        {
            int returns = 0;
            var buffer = new HipGpuBuffer(IntPtr.Zero, 4, owner, _ => Interlocked.Increment(ref returns));
            IDisposable? hold = null;
            using var start = new Barrier(2);
            var export = Task.Run(() =>
            {
                start.SignalAndWait();
                hold = buffer.HoldForExport();
            });
            start.SignalAndWait();
            buffer.Dispose();
            await export;
            Assert.Equal(hold is null ? 1 : 0, returns);
            hold?.Dispose();
            Assert.Equal(1, returns);
            buffer.Release();
        }
    }
}
