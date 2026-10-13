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
    /// The command buffer a dispatch records into: the device's open one (already begun, shared with the dispatches
    /// recorded before it, #1192) when it will not be waited for, the calling thread's when it will. Caller holds
    /// <c>_computeLock</c> until <see cref="SubmitDispatch"/>.
    /// </summary>
    private IntPtr DispatchCommandBuffer(ThreadCommandResources threadResources, out bool deferred)
    {
        deferred = SubmitsWithoutWaiting;
        return deferred ? _device.OpenCommandBuffer() : threadResources.CommandBuffer;
    }

    /// <summary>
    /// Finishes one recorded dispatch. A waited dispatch's command buffer (ended by the caller) is submitted and
    /// waited for. An unwaited one stays in the device's open command buffer, which is submitted when it holds
    /// <see cref="VulkanDevice.MaxCoalescedDispatches"/> or when something needs its results. Each unwaited
    /// dispatch is preceded by a compute-to-compute memory barrier (recorded by the caller), so it sees what the
    /// dispatches before it wrote, in this command buffer or an earlier submission.
    /// </summary>
    private void SubmitDispatch(IntPtr commandBuffer, ThreadCommandResources threadResources, bool deferred,
        VulkanComputePipeline pipeline, IntPtr descriptorSet)
    {
        if (!deferred)
        {
            _device.SubmitAndWait(commandBuffer, threadResources.Fence);
            return;
        }
        _device.RecordedDispatch(pipeline, descriptorSet, pipeline.BoundBuffers);
    }
}
