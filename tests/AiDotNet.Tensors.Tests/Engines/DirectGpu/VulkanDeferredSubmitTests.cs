using System;
using System.Linq;
using System.Reflection;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.Vulkan;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// #1191 and #1192: a Vulkan dispatch is recorded with the ones around it and not waited for; the submit happens
/// when the command buffer is full or a result is needed, and the wait when the host reads. Whatever is
/// read must be what a submit-and-wait backend would have produced: later dispatches see earlier ones' writes, a
/// host upload does not overtake a dispatch that reads the buffer, and a buffer released while a dispatch still
/// uses it is not handed to anyone else until that dispatch has run.
/// </summary>
[Collection("VulkanGlobalState")]
public class VulkanDeferredSubmitTests
{
    // The backend is a process-wide singleton other test classes share, so nothing here disposes it.
    private static VulkanBackend BackendOrSkip()
    {
        VulkanBackend? backend = null;
        try
        {
            backend = VulkanBackend.Instance;
            if (!backend.Initialize() || !backend.IsGlslCompilerAvailable) backend = null;
        }
        catch (Exception ex) when (ex is DllNotFoundException or PlatformNotSupportedException or TypeInitializationException)
        {
            backend = null;
        }
        Skip.If(backend is null, "No Vulkan device with the runtime GLSL compiler.");
        Skip.IfNot(backend!.SubmitsWithoutWaiting, "Submit-and-wait is forced by the environment.");
        return backend;
    }

    // b[i] = a[i] + 1, through the same n-ary dispatch helper the backend's own kernels use.
    private const string AddOne = @"#version 450
layout(local_size_x = 256) in;
layout(set = 0, binding = 0) readonly buffer A { float a[]; };
layout(set = 0, binding = 1) writeonly buffer B { float b[]; };
layout(push_constant) uniform Params { uint size; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= size) return;
    b[idx] = a[idx] + 1.0;
}";

    private static readonly MethodInfo NaryOp =
        typeof(VulkanBackend).GetMethod("GlslNaryOp", BindingFlags.NonPublic | BindingFlags.Instance)
        ?? throw new InvalidOperationException("GlslNaryOp not found");

    private static void RunAddOne(VulkanBackend backend, IGpuBuffer input, IGpuBuffer output, int size)
        => NaryOp.Invoke(backend, new object[] { AddOne, new[] { input, output }, size, new[] { (uint)size }, 256 });

    private static VulkanDevice Device => VulkanDevice.Instance;

    [SkippableFact]
    public void ADispatch_IsPendingUntilTheHostReads()
    {
        var backend = BackendOrSkip();
        const int size = 1000;
        using var input = backend.AllocateBuffer(Enumerable.Range(0, size).Select(i => (float)i).ToArray());
        using var output = backend.AllocateBuffer(size);
        backend.Synchronize();
        Assert.Equal(0, Device.PendingSubmissionCount);
        Assert.Equal(0, Device.OpenDispatchCount);

        long submittedBefore = Device.SubmittedSerial;
        RunAddOne(backend, input, output, size);
        // Nothing has asked for the result: the dispatch is recorded, and not even submitted yet (#1192).
        Assert.Equal(1, Device.OpenDispatchCount);
        Assert.Equal(0, Device.PendingSubmissionCount);
        Assert.Equal(submittedBefore, Device.SubmittedSerial);

        float[] values = backend.DownloadBuffer(output);
        Assert.Equal(0, Device.OpenDispatchCount);
        Assert.Equal(0, Device.PendingSubmissionCount);
        Assert.Equal(submittedBefore + 1, Device.SubmittedSerial);
        Assert.Equal(Device.SubmittedSerial, Device.CompletedSerial);
        for (int i = 0; i < size; i++) Assert.Equal(i + 1f, values[i]);
    }

    [SkippableFact]
    public void AChainOfDispatches_EachSeesThePreviousOnesOutput()
    {
        var backend = BackendOrSkip();
        const int size = 4097, steps = 301; // more steps than the queue holds, so the oldest are completed on the way
        using var ping = backend.AllocateBuffer(Enumerable.Range(0, size).Select(i => i * 0.5f).ToArray());
        using var pong = backend.AllocateBuffer(size);
        for (int step = 0; step < steps; step++)
        {
            if ((step & 1) == 0) RunAddOne(backend, ping, pong, size);
            else RunAddOne(backend, pong, ping, size);
            Assert.True(Device.PendingSubmissionCount <= VulkanDevice.MaxPendingSubmissions);
            Assert.True(Device.OpenDispatchCount < VulkanDevice.MaxCoalescedDispatches);
        }
        // An odd number of steps ends in pong.
        float[] values = backend.DownloadBuffer(pong);
        for (int i = 0; i < size; i++) Assert.Equal(i * 0.5f + steps, values[i]);
    }

    [SkippableFact]
    public void AnUpload_DoesNotOvertakeADispatchThatReadsTheBuffer()
    {
        var backend = BackendOrSkip();
        const int size = 2048;
        float[] first = Enumerable.Repeat(10f, size).ToArray();
        float[] second = Enumerable.Repeat(500f, size).ToArray();
        using var input = backend.AllocateBuffer(first);
        using var output = backend.AllocateBuffer(size);
        RunAddOne(backend, input, output, size);
        // The dispatch above may not have run yet. It must still read the first contents.
        backend.UploadBufferInPlace(second, input);
        Assert.All(backend.DownloadBuffer(output), v => Assert.Equal(11f, v));
        Assert.All(backend.DownloadBuffer(input), v => Assert.Equal(500f, v));
    }

    [SkippableFact]
    public void ABufferReleasedWhileADispatchUsesIt_IsNotReusedUntilTheDispatchHasRun()
    {
        var backend = BackendOrSkip();
        Skip.IfNot(VulkanGpuBuffer.PoolingEnabled, "Vulkan buffer pooling is disabled by the environment.");
        const int size = 77_791; // a size no other test allocates, so the pool's entries for it are this test's
        using var input = backend.AllocateBuffer(Enumerable.Repeat(3f, size).ToArray());
        backend.Synchronize();

        var scratch = backend.AllocateBuffer(size);
        IntPtr scratchHandle = scratch.Handle;
        RunAddOne(backend, input, scratch, size);
        Assert.Equal(1, Device.OpenDispatchCount);
        // The caller is done with it, but the dispatch is not.
        scratch.Dispose();

        // The next allocation of that size must not get the storage the pending dispatch writes.
        using var other = backend.AllocateBuffer(size);
        Assert.NotEqual(scratchHandle, other.Handle);

        // Once the dispatch has run the storage is free again, and the next allocation takes it.
        backend.Synchronize();
        using var reused = backend.AllocateBuffer(size);
        Assert.Equal(scratchHandle, reused.Handle);
    }

    [SkippableFact]
    public void WithSubmitAndWaitForced_NothingIsLeftPending()
    {
        var backend = BackendOrSkip();
        const int size = 512;
        using var input = backend.AllocateBuffer(Enumerable.Repeat(1f, size).ToArray());
        using var output = backend.AllocateBuffer(size);
        backend.Synchronize();
        backend.SubmitsWithoutWaiting = false;
        try
        {
            long submittedBefore = Device.SubmittedSerial;
            RunAddOne(backend, input, output, size);
            Assert.Equal(0, Device.PendingSubmissionCount);
            Assert.Equal(0, Device.OpenDispatchCount);
            // The waited path does not take a serial at all.
            Assert.Equal(submittedBefore, Device.SubmittedSerial);
        }
        finally
        {
            backend.SubmitsWithoutWaiting = true;
        }
        Assert.All(backend.DownloadBuffer(output), v => Assert.Equal(2f, v));
    }

    [SkippableFact]
    public void ConsecutiveDispatches_ShareOneSubmission_UpToTheLimit()
    {
        var backend = BackendOrSkip();
        const int size = 257;
        int limit = VulkanDevice.MaxCoalescedDispatches;
        using var ping = backend.AllocateBuffer(Enumerable.Repeat(0f, size).ToArray());
        using var pong = backend.AllocateBuffer(size);
        backend.Synchronize();
        long submittedBefore = Device.SubmittedSerial;

        // One short of the limit: all recorded, none submitted.
        int steps = 0;
        for (; steps < limit - 1; steps++)
        {
            if ((steps & 1) == 0) RunAddOne(backend, ping, pong, size); else RunAddOne(backend, pong, ping, size);
        }
        Assert.Equal(limit - 1, Device.OpenDispatchCount);
        Assert.Equal(submittedBefore, Device.SubmittedSerial);

        // The one that fills the command buffer submits it: one submission for all of them.
        if ((steps & 1) == 0) RunAddOne(backend, ping, pong, size); else RunAddOne(backend, pong, ping, size);
        steps++;
        Assert.Equal(0, Device.OpenDispatchCount);
        Assert.Equal(submittedBefore + 1, Device.SubmittedSerial);

        // The next opens a new one; the read submits and waits for it.
        if ((steps & 1) == 0) RunAddOne(backend, ping, pong, size); else RunAddOne(backend, pong, ping, size);
        steps++;
        Assert.Equal(1, Device.OpenDispatchCount);
        float[] values = backend.DownloadBuffer((steps & 1) == 1 ? pong : ping);
        Assert.Equal(submittedBefore + 2, Device.SubmittedSerial);
        Assert.Equal(Device.SubmittedSerial, Device.CompletedSerial);
        Assert.All(values, v => Assert.Equal((float)steps, v));
    }

    [SkippableFact]
    public void ARecordedDispatch_SeesAnUploadMadeBeforeIt_AndNotOneMadeAfter()
    {
        // Uploads are waited submissions: they flush what is recorded first. So the order the caller wrote is the
        // order the device runs, with a dispatch on either side of an upload.
        var backend = BackendOrSkip();
        const int size = 600;
        using var input = backend.AllocateBuffer(Enumerable.Repeat(1f, size).ToArray());
        using var first = backend.AllocateBuffer(size);
        using var second = backend.AllocateBuffer(size);
        RunAddOne(backend, input, first, size);                                   // reads 1
        backend.UploadBufferInPlace(Enumerable.Repeat(40f, size).ToArray(), input);
        RunAddOne(backend, input, second, size);                                  // reads 40
        Assert.All(backend.DownloadBuffer(second), v => Assert.Equal(41f, v));
        Assert.All(backend.DownloadBuffer(first), v => Assert.Equal(2f, v));
    }

    [SkippableFact]
    public void ABatch_RunsAfterTheDispatchesSubmittedBeforeIt()
    {
        var backend = BackendOrSkip();
        const int size = 300;
        using var a = backend.AllocateBuffer(Enumerable.Repeat(0f, size).ToArray());
        using var b = backend.AllocateBuffer(size);
        using var c = backend.AllocateBuffer(size);
        RunAddOne(backend, a, b, size);          // not waited for: b = 1
        backend.BeginBatch();
        try { RunAddOne(backend, b, c, size); }  // recorded: c = b + 1
        finally { backend.EndBatch(); }
        Assert.All(backend.DownloadBuffer(c), v => Assert.Equal(2f, v));
    }
}
