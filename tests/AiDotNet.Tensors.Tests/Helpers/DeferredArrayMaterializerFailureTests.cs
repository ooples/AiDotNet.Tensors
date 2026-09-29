// Copyright (c) AiDotNet. All rights reserved.

using AiDotNet.Tensors.Helpers;
using Xunit;

namespace AiDotNet.Tensors.Tests.Helpers;

/// <summary>
/// A deferred download that fails must stay pending. TryMaterialize used to remove the registration before running the
/// callback, so a download that threw (a stream sync issued during CUDA-graph capture fails with CUDA 900) lost the only
/// valid copy for good: the host array kept its old contents and no longer counted as pending, so every later read
/// silently returned stale data.
/// </summary>
public sealed class HostSyncFailureTests
{
    [Fact]
    public void FailedDownload_StaysPending_AndARetryDelivers()
    {
        var host = new float[] { 1f, 2f, 3f };
        bool fail = true;
        HostSync.Register(host, arr =>
        {
            if (fail) throw new InvalidOperationException("cuStreamSynchronize failed (900)");
            var a = (float[])arr;
            a[0] = 10f; a[1] = 20f; a[2] = 30f;
        });
        try
        {
            Assert.Throws<InvalidOperationException>(() => HostSync.TryMaterialize(host));
            Assert.True(HostSync.IsPending(host), "a failed download must stay pending");
            Assert.Equal(new[] { 1f, 2f, 3f }, host);

            fail = false;
            Assert.True(HostSync.TryMaterialize(host));
            Assert.Equal(new[] { 10f, 20f, 30f }, host);
            Assert.False(HostSync.IsPending(host));
        }
        finally
        {
            HostSync.Remove(host);
        }
    }
}
