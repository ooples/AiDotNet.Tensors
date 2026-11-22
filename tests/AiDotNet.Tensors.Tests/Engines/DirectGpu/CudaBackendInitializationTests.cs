using System;
using System.Diagnostics;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// Where the CUDA driver reports a device and NVRTC loads, the backend must come up. Its constructor catches every
/// initialization failure and reports the backend unavailable, and the CUDA tests skip on an unavailable backend, so a
/// kernel module that no longer compiles turned every CUDA test into a skip and every caller into the OpenCL fallback:
/// NVRTC 13.1 rejected the conv-engine mainloops' generic lambdas as host functions (GTX 1660 Ti), and nothing failed.
/// </summary>
public class CudaBackendInitializationTests
{
    [Fact]
    public void ApiProbe_PublishesSingleInitializedInstanceToEveryConcurrentCaller()
    {
        const int workerCount = 16;
        using var ready = new CountdownEvent(workerCount);
        using var start = new ManualResetEventSlim(false);
        int probeCount = 0;
        var expected = new object();
        var lazy = NvrtcNativeBindings.CreateApiProbe<object?>(() =>
        {
            Interlocked.Increment(ref probeCount);
            Thread.Sleep(50);
            return expected;
        });

        var tasks = new Task<object?>[workerCount];
        for (int i = 0; i < workerCount; i++)
        {
            tasks[i] = Task.Run(() =>
            {
                ready.Signal();
                start.Wait();
                return lazy.Value;
            });
        }

        ready.Wait();
        start.Set();
        Task.WaitAll(tasks);

        Assert.Equal(1, probeCount);
        foreach (var task in tasks)
        {
            Assert.Same(expected, task.Result);
        }
    }

    [SkippableFact]
    public void Backend_IsAvailable_WhereTheDriverHasADeviceAndNvrtcLoads()
    {
        Skip.IfNot(CudaNativeBindings.IsAvailable, "no CUDA driver or device");
        Skip.IfNot(NvrtcNativeBindings.IsAvailable, "NVRTC is not installed");

        // The constructor writes the reason it gave up to Trace; keep it for the failure message.
        var log = new StringWriter();
        var listener = new TextWriterTraceListener(log);
        Trace.Listeners.Add(listener);
        try
        {
            using var backend = new CudaBackend();
            listener.Flush();
            Assert.True(backend.IsAvailable, "CudaBackend failed to initialize on a machine with a CUDA device: " + log);
        }
        finally
        {
            Trace.Listeners.Remove(listener);
            listener.Dispose();
        }
    }

    // A lazy probe caches an exception and rethrows it from every later read, so a failure in the best-effort path
    // discovery must not escape: the probe still runs, once, and every reader gets its answer.
    [Theory]
    [InlineData(typeof(IOException))]
    [InlineData(typeof(UnauthorizedAccessException))]
    [InlineData(typeof(ArgumentException))]
    public void ApiProbe_StillRunsOnce_WhenPathDiscoveryThrows(Type discoveryFailure)
    {
        int discoveries = 0;
        int probes = 0;
        var expected = new object();
        var lazy = NvrtcNativeBindings.CreateApiProbe<object?>(() => NvrtcNativeBindings.ProbeAfterBestEffortDiscovery<object?>(
            () =>
            {
                Interlocked.Increment(ref discoveries);
                throw (Exception)(Activator.CreateInstance(discoveryFailure, "discovery failed")
                    ?? throw new InvalidOperationException($"Could not create {discoveryFailure}."));
            },
            () =>
            {
                Interlocked.Increment(ref probes);
                return expected;
            }));

        Assert.Same(expected, lazy.Value);
        Assert.Same(expected, lazy.Value);
        Assert.Equal(1, discoveries);
        Assert.Equal(1, probes);
    }

    // A failure that is not a path-discovery failure is a defect, not "not installed": it is not swallowed.
    [Fact]
    public void ApiProbe_DoesNotSwallowUnrelatedDiscoveryFailures()
    {
        Assert.Throws<InvalidOperationException>(() => NvrtcNativeBindings.ProbeAfterBestEffortDiscovery<object?>(
            () => throw new InvalidOperationException("defect"),
            () => null));
    }
}
