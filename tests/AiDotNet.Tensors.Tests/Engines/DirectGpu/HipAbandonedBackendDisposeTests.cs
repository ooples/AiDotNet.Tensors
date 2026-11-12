// Copyright (c) CrowdTrainer LLC. All rights reserved.

using System;
using System.Collections.Generic;
using System.IO;
using System.Text.RegularExpressions;
using AiDotNet.Tensors.Engines.DirectGpu.HIP;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// A HIP backend collected without Dispose is torn down later, on whichever thread allocates next. hipSetDevice is per
/// thread, and that teardown bound the abandoned backend's device and never bound the caller's back, so the caller's
/// next kernels ran on the wrong device; it also waited for the whole other device (hipDeviceSynchronize) on the
/// caller's thread. HIP does not run on CI machines, so the device save/restore is tested through its injectable
/// form, and the teardown's shape is pinned at the source.
/// </summary>
public sealed class HipAbandonedBackendDisposeTests
{
    [Fact]
    public void RunBoundToDevice_BindsTheTargetAndRestoresTheCallersDevice()
    {
        int current = 1;
        var binds = new List<int>();
        int seenInside = -1;
        bool result = HipBackend.RunBoundToDevice(0, () => current, d => { binds.Add(d); current = d; }, () =>
        {
            seenInside = current;
            return true;
        });

        Assert.True(result);
        Assert.Equal(0, seenInside);
        Assert.Equal(new[] { 0, 1 }, binds);
        Assert.Equal(1, current);
    }

    [Fact]
    public void RunBoundToDevice_RestoresTheCallersDevice_WhenTheTeardownThrows()
    {
        int current = 2;
        Assert.Throws<InvalidOperationException>(() =>
            HipBackend.RunBoundToDevice(0, () => current, d => current = d, () => throw new InvalidOperationException("faulted")));
        Assert.Equal(2, current);
    }

    [Fact]
    public void RunBoundToDevice_DoesNotRebind_WhenAlreadyOnTheDevice()
    {
        int current = 3;
        int binds = 0;
        bool result = HipBackend.RunBoundToDevice(3, () => current, d => { binds++; current = d; }, () => false);
        Assert.False(result);
        Assert.Equal(0, binds);
        Assert.Equal(3, current);
    }

    [Fact]
    public void DrainingWithNothingAbandoned_TouchesNoDevice()
    {
        // Runs on every allocation: with no abandoned backend it must return before any HIP call (none is loaded here).
        Assert.True(HipBackend.PendingAbandonedBackends.IsEmpty);
        HipBackend.DrainPendingAbandonedBackends();
    }

    [Fact]
    public void AbandonedTeardown_RestoresTheDevice_AndNeverSynchronizesTheDevice()
    {
        string source = File.ReadAllText(Path.Combine(RepoRoot(), "src", "AiDotNet.Tensors", "Engines", "DirectGpu", "HIP", "HipBackend.cs"));

        string finalizer = Body(source, @"~HipBackend\(\)");
        Assert.Contains("PendingAbandonedBackends.Enqueue(this)", finalizer);
        Assert.DoesNotContain("Dispose", finalizer);

        string teardown = Body(source, @"private bool TryDisposeAbandoned\(\)");
        Assert.Contains("RunBoundToDevice(_hipDeviceIndex", teardown);
        Assert.Contains("hipStreamQuery(_stream)", teardown);
        Assert.DoesNotContain("hipDeviceSynchronize", teardown);
        Assert.DoesNotContain("hipStreamSynchronize", teardown);
        Assert.DoesNotContain("Dispose()", teardown);   // the explicit Dispose synchronizes the device; this must not

        string drain = Body(source, @"internal static void DrainPendingAbandonedBackends\(\)");
        Assert.DoesNotContain("hipDeviceSynchronize", drain);
    }

    // The text of the brace-delimited body following the first match of <paramref name="signature"/>.
    private static string Body(string source, string signature)
    {
        var match = Regex.Match(source, signature);
        Assert.True(match.Success, $"'{signature}' not found in HipBackend.cs");
        int open = source.IndexOf('{', match.Index + match.Length);
        int depth = 0;
        for (int i = open; i < source.Length; i++)
        {
            if (source[i] == '{') depth++;
            else if (source[i] == '}' && --depth == 0) return source.Substring(open, i - open + 1);
        }
        throw new InvalidOperationException($"unbalanced body after '{signature}'");
    }

    private static string RepoRoot()
    {
        var probe = Path.Combine("src", "AiDotNet.Tensors", "Engines", "CpuEngine.cs");
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir is not null && !File.Exists(Path.Combine(dir.FullName, probe)))
            dir = dir.Parent;
        return dir?.FullName ?? throw new InvalidOperationException(
            $"could not locate repo root by walking up from {AppContext.BaseDirectory} looking for {probe}");
    }
}
