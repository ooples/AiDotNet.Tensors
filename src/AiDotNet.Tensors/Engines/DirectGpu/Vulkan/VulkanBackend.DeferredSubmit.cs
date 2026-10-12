// Copyright (c) AiDotNet. All rights reserved.
// Submission without waiting (#1191, phase 2 of #1160).
using System;

namespace AiDotNet.Tensors.Engines.DirectGpu.Vulkan;

public sealed unsafe partial class VulkanBackend
{
    /// <summary>
    /// Environment switch that restores submit-and-wait for every dispatch: <c>AIDOTNET_VULKAN_SYNC_SUBMIT=1</c>.
    /// For debugging and bisecting, where an error should surface at the op that caused it.
    /// </summary>
    internal const string SyncSubmitEnvironmentVariable = "AIDOTNET_VULKAN_SYNC_SUBMIT";

    /// <summary>
    /// Whether a compute dispatch is submitted and left to run (true) or submitted and waited for (false).
    /// </summary>
    /// <remarks>
    /// Waiting cost about 50 of the roughly 80 microseconds every Vulkan op took, whatever its size (#1160), and
    /// nothing needs the result until the host reads it. With this on, a dispatch returns as soon as it is
    /// submitted; <see cref="VulkanDevice.WaitForPendingSubmissions"/> runs before every host read of device
    /// memory, every waited submission, and in <see cref="Synchronize"/>. Consequences a caller can see:
    /// <list type="bullet">
    /// <item>a device error surfaces at the next host read, not at the op that caused it;</item>
    /// <item>timing an op with a stopwatch measures its submission unless the caller calls
    /// <see cref="Synchronize"/> before stopping the clock;</item>
    /// <item>a buffer disposed while a submission still uses it returns to the pool when that submission has
    /// executed, not at once.</item>
    /// </list>
    /// </remarks>
    internal bool SubmitsWithoutWaiting { get; set; } =
        Environment.GetEnvironmentVariable(SyncSubmitEnvironmentVariable) != "1";

    /// <summary>
    /// The command buffer a dispatch records into: one of its own when it will not be waited for, the calling
    /// thread's when it will. Caller holds <c>_computeLock</c> until <see cref="SubmitDispatch"/>.
    /// </summary>
    private IntPtr DispatchCommandBuffer(ThreadCommandResources threadResources, out DeferredCommandSlot slot, out bool deferred)
    {
        deferred = SubmitsWithoutWaiting;
        if (!deferred)
        {
            slot = default;
            return threadResources.CommandBuffer;
        }
        slot = _device.AcquireDeferredSlot();
        return slot.CommandBuffer;
    }

    /// <summary>
    /// Submits the recorded (and ended) command buffer. A submission that is not waited for begins with a
    /// compute-to-compute memory barrier (recorded by the caller), so it sees what the submissions before it wrote
    /// even though they may not have finished when it is submitted.
    /// </summary>
    private void SubmitDispatch(IntPtr commandBuffer, ThreadCommandResources threadResources, DeferredCommandSlot slot,
        bool deferred, VulkanComputePipeline pipeline, IntPtr descriptorSet)
    {
        if (!deferred)
        {
            _device.SubmitAndWait(commandBuffer, threadResources.Fence);
            return;
        }
        try
        {
            _device.SubmitWithoutWaiting(slot, pipeline, descriptorSet, pipeline.BoundBuffers);
        }
        catch
        {
            // Nothing was queued: the set is not bound by anything pending.
            pipeline.ReleaseDescriptorSet(descriptorSet);
            throw;
        }
    }
}
