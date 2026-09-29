using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Diagnostics;
using AiDotNet.Tensors.Engines.DirectGpu;
using Xunit;

namespace AiDotNet.Tensors.Tests.Parity;

/// <summary>
/// The scope's own contract, independent of any GPU: transfers are reported straight to <see cref="GpuLaunchProbe"/>,
/// exactly as a backend reports them. Shares the serial GPU collection because the probe's scope registry is
/// process-wide.
/// </summary>
[Collection("DirectGpuSerial")]
public sealed class GpuResidencyScopeContractTests
{
    // The scope records whichever backend reports a transfer; its contract does not depend on which one.
    private const GpuBackendType ReportingBackend = GpuBackendType.OpenCl;

    /// <summary>
    /// Scopes nest. Disposing an outer scope while an inner one is open used to unlink nothing and then let the inner
    /// scope's disposal restore the disposed outer scope as current, which collected transfers again.
    /// </summary>
    [Fact]
    public void Dispose_OutOfOrder_IsRefused_AndLeavesBothScopesUsable()
    {
        var outer = GpuResidencyScope.Begin();
        var inner = GpuResidencyScope.Begin();
        try
        {
            var refused = Assert.Throws<InvalidOperationException>(() => outer.Dispose());
            Assert.Contains("nest", refused.Message);

            GpuLaunchProbe.OnUpload(16, ReportingBackend);
            Assert.Equal(1, inner.Uploads);
            Assert.Equal(1, outer.Uploads);
        }
        finally
        {
            inner.Dispose();
            outer.Dispose();
        }

        GpuLaunchProbe.OnUpload(16, ReportingBackend);
        Assert.Equal(1, outer.Uploads);
        Assert.Equal(1, inner.Uploads);
    }

    /// <summary>
    /// A report is final once Dispose returns. A process-wide scope's recorder works from a snapshot of the open
    /// scopes, so a transfer that raced Dispose could land in a report the caller had already read.
    /// </summary>
    [Fact]
    public void ProcessWideScope_AfterDispose_NeverChanges()
    {
        var scope = GpuResidencyScope.Begin(processWide: true);
        GpuLaunchProbe.OnUpload(8, ReportingBackend);
        GpuLaunchProbe.OnReadback(4, ReportingBackend);
        scope.Dispose();
        int events = scope.Events.Count;

        GpuLaunchProbe.OnUpload(8, ReportingBackend);
        GpuLaunchProbe.OnReadback(4, ReportingBackend);
        // What a racing recorder does after Dispose: it adds to the scope it snapshotted.
        var add = typeof(GpuResidencyScope).GetMethod("Add", System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic)
            ?? throw new InvalidOperationException("GpuResidencyScope.Add not found.");
        add.Invoke(scope, new object[] { new GpuTransferEvent(GpuTransferKind.HostToDevice, 8, ReportingBackend, null) });

        Assert.Equal(2, events);
        Assert.Equal(events, scope.Events.Count);
        Assert.Equal(8, scope.BytesUploaded);
        Assert.Equal(4, scope.BytesDownloaded);
    }

    /// <summary>A negative readback cannot subtract from the process totals.</summary>
    [Fact]
    public void Readback_WithANegativeByteCount_IsRejected()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => GpuLaunchProbe.OnReadback(-1, ReportingBackend));
    }
}
