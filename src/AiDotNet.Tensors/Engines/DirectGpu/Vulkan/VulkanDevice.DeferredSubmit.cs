// Copyright (c) AiDotNet. All rights reserved.
// Submission without waiting (#1191): a dispatch is submitted and the caller returns; the wait happens when a
// result is read on the host.
using System;
using System.Collections.Generic;
using System.Threading;

namespace AiDotNet.Tensors.Engines.DirectGpu.Vulkan;

/// <summary>A command buffer and fence for one submission that is not waited for at submit time.</summary>
internal readonly struct DeferredCommandSlot
{
    internal DeferredCommandSlot(IntPtr commandBuffer, IntPtr fence)
    {
        CommandBuffer = commandBuffer;
        Fence = fence;
    }

    internal IntPtr CommandBuffer { get; }
    internal IntPtr Fence { get; }
}

public sealed unsafe partial class VulkanDevice
{
    // One submission that has been handed to the queue and not yet observed complete.
    private readonly struct PendingSubmission
    {
        internal PendingSubmission(long serial, DeferredCommandSlot slot, VulkanComputePipeline pipeline, IntPtr descriptorSet)
        {
            Serial = serial;
            Slot = slot;
            Pipeline = pipeline;
            DescriptorSet = descriptorSet;
        }

        internal long Serial { get; }
        internal DeferredCommandSlot Slot { get; }
        internal VulkanComputePipeline Pipeline { get; }
        internal IntPtr DescriptorSet { get; }
    }

    /// <summary>
    /// Submissions the queue may hold before the next one waits for the oldest. It bounds the command buffers,
    /// fences and descriptor sets in flight, and the device memory of buffers whose release is waiting on them.
    /// </summary>
    internal const int MaxPendingSubmissions = 64;

    // Guards the pending queue, the free slots and the deferred releases. Taken after VulkanBackend's compute lock
    // when both are held, never the other way round; nothing is called back while it is held.
    private readonly object _pendingLock = new();
    private readonly Queue<PendingSubmission> _pending = new();
    private readonly Stack<DeferredCommandSlot> _freeSlots = new();
    private readonly List<(long Serial, Action Release)> _deferredReleases = new();
    private IntPtr _deferredCommandPool;
    private long _submittedSerial;
    private long _completedSerial;

    /// <summary>Serial of the most recent submission made without waiting; 0 before the first.</summary>
    internal long SubmittedSerial => Interlocked.Read(ref _submittedSerial);

    /// <summary>Every submission with a serial up to this one has finished executing.</summary>
    internal long CompletedSerial => Interlocked.Read(ref _completedSerial);

    /// <summary>Submissions handed to the queue and not yet observed complete.</summary>
    internal int PendingSubmissionCount
    {
        get { lock (_pendingLock) return _pending.Count; }
    }

    /// <summary>
    /// A command buffer and fence for a submission that will not be waited for. Recording into the command buffer
    /// needs the same external synchronization as acquiring it: the caller holds the backend's compute lock from
    /// this call until <see cref="SubmitWithoutWaiting"/>.
    /// </summary>
    internal DeferredCommandSlot AcquireDeferredSlot()
    {
        if (!_initialized || _disposed)
            throw new GpuCapabilityUnavailableException("Vulkan device not initialized.");

        List<Action>? releases = null;
        try
        {
            lock (_pendingLock)
            {
                // Recycle what has already finished; if the queue is still full, wait for the oldest.
                ReapCompletedLocked(waitForAll: false, ref releases);
                while (_pending.Count >= MaxPendingSubmissions)
                    CompleteOldestLocked(ref releases);

                if (_freeSlots.Count > 0) return _freeSlots.Pop();
                return CreateDeferredSlotLocked();
            }
        }
        finally
        {
            RunReleases(releases);
        }
    }

    // Caller holds _pendingLock.
    private DeferredCommandSlot CreateDeferredSlotLocked()
    {
        if (_deferredCommandPool == IntPtr.Zero)
        {
            var poolCreateInfo = new VkCommandPoolCreateInfo
            {
                sType = VulkanNativeBindings.VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                pNext = null,
                flags = VkCommandPoolCreateFlags.VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
                queueFamilyIndex = _computeQueueFamily
            };
            int poolResult = VulkanNativeBindings.vkCreateCommandPool(_device, &poolCreateInfo, IntPtr.Zero, out _deferredCommandPool);
            if (poolResult != VulkanNativeBindings.VK_SUCCESS)
            {
                _deferredCommandPool = IntPtr.Zero;
                throw new InvalidOperationException($"Failed to create the deferred-submission command pool: {poolResult}");
            }
        }

        var allocInfo = new VkCommandBufferAllocateInfo
        {
            sType = VulkanNativeBindings.VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            pNext = null,
            commandPool = _deferredCommandPool,
            level = VulkanNativeBindings.VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            commandBufferCount = 1
        };
        IntPtr commandBuffer;
        int allocResult = VulkanNativeBindings.vkAllocateCommandBuffers(_device, &allocInfo, &commandBuffer);
        if (allocResult != VulkanNativeBindings.VK_SUCCESS)
            throw new InvalidOperationException($"Failed to allocate a deferred-submission command buffer: {allocResult}");

        var fenceCreateInfo = new VkFenceCreateInfo
        {
            sType = VulkanNativeBindings.VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
            pNext = null,
            flags = 0
        };
        int fenceResult = VulkanNativeBindings.vkCreateFence(_device, &fenceCreateInfo, IntPtr.Zero, out IntPtr fence);
        if (fenceResult != VulkanNativeBindings.VK_SUCCESS)
            throw new InvalidOperationException($"Failed to create a deferred-submission fence: {fenceResult}");

        return new DeferredCommandSlot(commandBuffer, fence);
    }

    /// <summary>
    /// Submits <paramref name="slot"/>'s recorded command buffer and returns without waiting. The slot and
    /// <paramref name="descriptorSet"/> (retained from <paramref name="pipeline"/>) are handed back when the
    /// submission is observed complete; <paramref name="boundBuffers"/> are marked as used by it. On failure the
    /// slot goes back to the free list and nothing is queued. Returns the submission's serial.
    /// </summary>
    internal long SubmitWithoutWaiting(DeferredCommandSlot slot, VulkanComputePipeline pipeline, IntPtr descriptorSet,
        VulkanBuffer[] boundBuffers)
    {
        EnsureDeviceWarm();
        long serial;
        lock (_submitLock)
        {
            var fencePtr = slot.Fence;
            int resetResult = VulkanNativeBindings.vkResetFences(_device, 1, &fencePtr);
            if (resetResult != VulkanNativeBindings.VK_SUCCESS)
            {
                lock (_pendingLock) _freeSlots.Push(slot);
                throw new InvalidOperationException($"vkResetFences failed with result: {resetResult}");
            }

            var commandBuffer = slot.CommandBuffer;
            var submitInfo = new VkSubmitInfo
            {
                sType = VulkanNativeBindings.VK_STRUCTURE_TYPE_SUBMIT_INFO,
                pNext = null,
                waitSemaphoreCount = 0,
                pWaitSemaphores = null,
                pWaitDstStageMask = null,
                commandBufferCount = 1,
                pCommandBuffers = &commandBuffer,
                signalSemaphoreCount = 0,
                pSignalSemaphores = null
            };
            int submitResult = VulkanNativeBindings.vkQueueSubmit(_computeQueue, 1, &submitInfo, slot.Fence);
            if (submitResult != VulkanNativeBindings.VK_SUCCESS)
            {
                lock (_pendingLock) _freeSlots.Push(slot);
                throw new InvalidOperationException($"vkQueueSubmit failed with error code {submitResult}");
            }

            // The serial is assigned and queued inside the submit lock, so serial order is submission order.
            lock (_pendingLock)
            {
                serial = ++_submittedSerial;
                _pending.Enqueue(new PendingSubmission(serial, slot, pipeline, descriptorSet));
                // Marked under the lock a release takes to compare serials, so a release cannot slip between the
                // submission and the mark.
                foreach (var buffer in boundBuffers) buffer.MarkUsedBySubmission(serial);
            }
        }
        return serial;
    }

    /// <summary>
    /// Waits until every submission made without waiting has executed, then hands back their resources and runs
    /// the buffer releases that were waiting on them. Every host read of device memory goes through here first.
    /// </summary>
    internal void WaitForPendingSubmissions()
    {
        if (_device == IntPtr.Zero) return;
        List<Action>? releases = null;
        try
        {
            lock (_pendingLock)
            {
                if (_pending.Count == 0 && _deferredReleases.Count == 0) return;
                ReapCompletedLocked(waitForAll: true, ref releases);
            }
        }
        finally
        {
            RunReleases(releases);
        }
    }

    /// <summary>
    /// Runs <paramref name="release"/> once the submission with <paramref name="serial"/> has executed: now when it
    /// already has, otherwise when it is observed complete. A buffer a pending dispatch still reads or writes must
    /// not be destroyed, or handed to another owner, before then.
    /// </summary>
    internal void ReleaseAfterSubmission(long serial, Action release)
    {
        if (release is null) throw new ArgumentNullException(nameof(release));
        lock (_pendingLock)
        {
            if (serial > _completedSerial)
            {
                _deferredReleases.Add((serial, release));
                return;
            }
        }
        release();
    }

    // Caller holds _pendingLock. Completes submissions from the oldest: all of them when waitForAll, otherwise only
    // those whose fence has already signalled.
    private void ReapCompletedLocked(bool waitForAll, ref List<Action>? releases)
    {
        while (_pending.Count > 0)
        {
            if (!waitForAll)
            {
                int status = VulkanNativeBindings.vkGetFenceStatus(_device, _pending.Peek().Slot.Fence);
                if (status == VulkanNativeBindings.VK_NOT_READY) break;
                if (status != VulkanNativeBindings.VK_SUCCESS)
                    throw new InvalidOperationException($"vkGetFenceStatus failed with error code {status}");
            }
            CompleteOldestLocked(ref releases);
        }
        if (_pending.Count == 0) CollectReleasesLocked(ref releases);
    }

    // Caller holds _pendingLock. Waits for the oldest pending submission and recycles what it held.
    private void CompleteOldestLocked(ref List<Action>? releases)
    {
        var oldest = _pending.Dequeue();
        var fencePtr = oldest.Slot.Fence;
        int waitResult = VulkanNativeBindings.vkWaitForFences(_device, 1, &fencePtr, 1, ulong.MaxValue);
        // Recycled either way: after a failed wait the device is lost and nothing will run on it again.
        oldest.Pipeline.ReleaseDescriptorSet(oldest.DescriptorSet);
        _freeSlots.Push(oldest.Slot);
        Interlocked.Exchange(ref _completedSerial, oldest.Serial);
        CollectReleasesLocked(ref releases);
        if (waitResult != VulkanNativeBindings.VK_SUCCESS)
            throw new InvalidOperationException(
                $"A Vulkan dispatch submitted earlier failed (vkWaitForFences returned {waitResult}); the error surfaces here because the dispatch was not waited for when it was submitted.");
    }

    // Caller holds _pendingLock. Moves the releases whose submission has executed to the caller's list; they run
    // after the lock is dropped.
    private void CollectReleasesLocked(ref List<Action>? releases)
    {
        if (_deferredReleases.Count == 0) return;
        long completed = _completedSerial;
        for (int i = _deferredReleases.Count - 1; i >= 0; i--)
        {
            if (_deferredReleases[i].Serial > completed) continue;
            (releases ??= new List<Action>()).Add(_deferredReleases[i].Release);
            _deferredReleases.RemoveAt(i);
        }
    }

    private static void RunReleases(List<Action>? releases)
    {
        if (releases is null) return;
        foreach (var release in releases)
        {
            try { release(); }
            catch { /* a failing release must not strand the others or mask the caller's own exception */ }
        }
    }

    // Called from Cleanup, after the device is idle.
    private void DestroyDeferredSubmissionResources()
    {
        lock (_pendingLock)
        {
            if (_device != IntPtr.Zero)
            {
                while (_pending.Count > 0)
                {
                    var fence = _pending.Dequeue().Slot.Fence;
                    if (fence != IntPtr.Zero) VulkanNativeBindings.vkDestroyFence(_device, fence, IntPtr.Zero);
                }
                while (_freeSlots.Count > 0)
                {
                    var fence = _freeSlots.Pop().Fence;
                    if (fence != IntPtr.Zero) VulkanNativeBindings.vkDestroyFence(_device, fence, IntPtr.Zero);
                }
                // Destroying the pool frees every command buffer allocated from it.
                if (_deferredCommandPool != IntPtr.Zero)
                    VulkanNativeBindings.vkDestroyCommandPool(_device, _deferredCommandPool, IntPtr.Zero);
            }
            _pending.Clear();
            _freeSlots.Clear();
            _deferredCommandPool = IntPtr.Zero;
            _deferredReleases.Clear();
        }
    }
}
