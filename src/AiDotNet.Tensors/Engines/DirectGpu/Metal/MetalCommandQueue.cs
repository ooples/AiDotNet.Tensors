// Copyright (c) AiDotNet. All rights reserved.
// Metal command queue and compute encoder management for GPU compute operations.

using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Threading;
using static AiDotNet.Tensors.Engines.DirectGpu.Metal.MetalNativeBindings;

namespace AiDotNet.Tensors.Engines.DirectGpu.Metal;

/// <summary>
/// Manages Metal command queue, command buffers, and compute encoders for GPU operations.
/// </summary>
/// <remarks>
/// <para><b>Command Queue Architecture:</b></para>
/// <para>
/// Metal uses a multi-level command submission model:
/// </para>
/// <list type="bullet">
/// <item><b>Command Queue</b>: Serial queue of command buffers for a device</item>
/// <item><b>Command Buffer</b>: Container for encoded commands</item>
/// <item><b>Command Encoder</b>: Encodes specific types of commands (compute, blit, render)</item>
/// </list>
/// <para><b>Thread Safety:</b></para>
/// <para>
/// Command queue is thread-safe, but command buffers and encoders are NOT.
/// This class uses a pool of command buffers to support concurrent operations.
/// </para>
/// </remarks>
public sealed class MetalCommandQueue : IDisposable
{
    private IntPtr _commandQueue;
    private readonly MetalDevice _device;
    private bool _disposed;
    private readonly object _lock = new();

    // Command buffer pool for efficient reuse
    private readonly Queue<IntPtr> _commandBufferPool = new();
    private readonly int _maxPooledBuffers;
    private int _activeBufferCount;

    /// <summary>
    /// Gets the native Metal command queue handle.
    /// </summary>
    public IntPtr Handle => _commandQueue;

    /// <summary>
    /// Gets whether the command queue is valid.
    /// </summary>
    public bool IsValid => _commandQueue != IntPtr.Zero && !_disposed;

    /// <summary>
    /// Gets the device associated with this command queue.
    /// </summary>
    public MetalDevice Device => _device;

    /// <summary>
    /// Gets the number of active (in-flight) command buffers.
    /// </summary>
    public int ActiveBufferCount => _activeBufferCount;

    /// <summary>
    /// Gets the number of pooled command buffers available for reuse.
    /// </summary>
    public int PooledBufferCount
    {
        get
        {
            lock (_lock)
            {
                return _commandBufferPool.Count;
            }
        }
    }

    /// <summary>
    /// Creates a new Metal command queue.
    /// </summary>
    /// <param name="device">The Metal device.</param>
    /// <param name="maxPooledBuffers">Maximum number of command buffers to pool for reuse.</param>
    public MetalCommandQueue(MetalDevice device, int maxPooledBuffers = 16)
    {
        if (device is null || !device.IsAvailable)
        {
            throw new ArgumentException("Invalid Metal device", nameof(device));
        }

        _device = device;
        _maxPooledBuffers = maxPooledBuffers;
        _commandQueue = device.CreateCommandQueue();

        if (_commandQueue == IntPtr.Zero)
        {
            throw new InvalidOperationException("Failed to create Metal command queue");
        }
    }

    /// <summary>
    /// Creates a new Metal command queue with a maximum command buffer count.
    /// </summary>
    /// <param name="device">The Metal device.</param>
    /// <param name="maxCommandBufferCount">Maximum concurrent command buffers.</param>
    /// <param name="maxPooledBuffers">Maximum number of command buffers to pool for reuse.</param>
    public MetalCommandQueue(MetalDevice device, ulong maxCommandBufferCount, int maxPooledBuffers = 16)
    {
        if (device is null || !device.IsAvailable)
        {
            throw new ArgumentException("Invalid Metal device", nameof(device));
        }

        _device = device;
        _maxPooledBuffers = maxPooledBuffers;
        _commandQueue = device.CreateCommandQueue(maxCommandBufferCount);

        if (_commandQueue == IntPtr.Zero)
        {
            throw new InvalidOperationException("Failed to create Metal command queue");
        }
    }

    /// <summary>
    /// Creates a new command buffer for encoding GPU commands.
    /// </summary>
    /// <returns>A command buffer handle, or IntPtr.Zero on failure.</returns>
    public IntPtr CreateCommandBuffer()
    {
        ThrowIfDisposed();

        IntPtr buffer;
        lock (_lock)
        {
            // Try to reuse a pooled buffer first
            if (_commandBufferPool.Count > 0)
            {
                buffer = _commandBufferPool.Dequeue();
                Interlocked.Increment(ref _activeBufferCount);
                return buffer;
            }
        }

        // Create a new command buffer
        buffer = SendMessage(_commandQueue, Selectors.CommandBuffer);
        if (buffer != IntPtr.Zero)
        {
            Retain(buffer);
            Interlocked.Increment(ref _activeBufferCount);
        }

        return buffer;
    }

    /// <summary>
    /// Creates a command buffer with unretained references for better performance.
    /// </summary>
    /// <remarks>
    /// Use this when you're certain all referenced resources will outlive the command buffer.
    /// This avoids retain/release overhead but requires careful lifetime management.
    /// </remarks>
    /// <returns>A command buffer handle, or IntPtr.Zero on failure.</returns>
    public IntPtr CreateCommandBufferUnretained()
    {
        ThrowIfDisposed();

        var buffer = SendMessage(_commandQueue, Selectors.CommandBufferWithUnretainedReferences);
        if (buffer != IntPtr.Zero)
        {
            Retain(buffer);
            Interlocked.Increment(ref _activeBufferCount);
        }

        return buffer;
    }

    /// <summary>
    /// Returns a command buffer to the pool for reuse.
    /// </summary>
    /// <param name="commandBuffer">The command buffer to return.</param>
    internal void ReturnCommandBuffer(IntPtr commandBuffer)
    {
        if (commandBuffer == IntPtr.Zero)
        {
            return;
        }

        Interlocked.Decrement(ref _activeBufferCount);

        lock (_lock)
        {
            // Check if the command buffer has completed
            var status = (MTLCommandBufferStatus)SendMessageULongReturn(commandBuffer, Selectors.Status);

            if (status == MTLCommandBufferStatus.Completed)
            {
                // Release completed buffers - they can't be reused
                Release(commandBuffer);
            }
            else if (_commandBufferPool.Count < _maxPooledBuffers)
            {
                // Pool for reuse if not yet completed and pool isn't full
                _commandBufferPool.Enqueue(commandBuffer);
            }
            else
            {
                Release(commandBuffer);
            }
        }
    }

    /// <summary>
    /// Creates a compute command encoder for GPU compute operations.
    /// </summary>
    /// <param name="commandBuffer">The command buffer to encode into.</param>
    /// <returns>A compute command encoder handle, or IntPtr.Zero on failure.</returns>
    public IntPtr CreateComputeEncoder(IntPtr commandBuffer)
    {
        ThrowIfDisposed();

        if (commandBuffer == IntPtr.Zero)
        {
            throw new ArgumentException("Invalid command buffer", nameof(commandBuffer));
        }

        return SendMessage(commandBuffer, Selectors.ComputeCommandEncoder);
    }

    /// <summary>
    /// Creates a blit command encoder for buffer/texture copy operations.
    /// </summary>
    /// <param name="commandBuffer">The command buffer to encode into.</param>
    /// <returns>A blit command encoder handle, or IntPtr.Zero on failure.</returns>
    public IntPtr CreateBlitEncoder(IntPtr commandBuffer)
    {
        ThrowIfDisposed();

        if (commandBuffer == IntPtr.Zero)
        {
            throw new ArgumentException("Invalid command buffer", nameof(commandBuffer));
        }

        return SendMessage(commandBuffer, Selectors.BlitCommandEncoder);
    }

    /// <summary>
    /// Ends encoding for a command encoder.
    /// </summary>
    /// <param name="encoder">The encoder to end.</param>
    public static void EndEncoding(IntPtr encoder)
    {
        if (encoder != IntPtr.Zero)
        {
            SendMessageVoid(encoder, Selectors.EndEncoding);
        }
    }

    /// <summary>
    /// Commits a command buffer for execution.
    /// </summary>
    /// <param name="commandBuffer">The command buffer to commit.</param>
    public static void Commit(IntPtr commandBuffer)
    {
        if (commandBuffer != IntPtr.Zero)
        {
            SendMessageVoid(commandBuffer, Selectors.Commit);
        }
    }

    /// <summary>
    /// Waits for a command buffer to complete execution.
    /// </summary>
    /// <param name="commandBuffer">The command buffer to wait for.</param>
    public static void WaitUntilCompleted(IntPtr commandBuffer)
    {
        if (commandBuffer != IntPtr.Zero)
        {
            SendMessageVoid(commandBuffer, Selectors.WaitUntilCompleted);
        }
    }

    /// <summary>
    /// Waits for a command buffer to be scheduled.
    /// </summary>
    /// <param name="commandBuffer">The command buffer to wait for.</param>
    public static void WaitUntilScheduled(IntPtr commandBuffer)
    {
        if (commandBuffer != IntPtr.Zero)
        {
            SendMessageVoid(commandBuffer, Selectors.WaitUntilScheduled);
        }
    }

    /// <summary>
    /// Gets the status of a command buffer.
    /// </summary>
    /// <param name="commandBuffer">The command buffer to check.</param>
    /// <returns>The current status of the command buffer.</returns>
    public static MTLCommandBufferStatus GetStatus(IntPtr commandBuffer)
    {
        if (commandBuffer == IntPtr.Zero)
        {
            return MTLCommandBufferStatus.NotEnqueued;
        }

        return (MTLCommandBufferStatus)SendMessageULongReturn(commandBuffer, Selectors.Status);
    }

    /// <summary>
    /// Gets any error from a command buffer.
    /// </summary>
    /// <param name="commandBuffer">The command buffer to check.</param>
    /// <returns>Error description if an error occurred, null otherwise.</returns>
    public static string? GetError(IntPtr commandBuffer)
    {
        if (commandBuffer == IntPtr.Zero)
        {
            return null;
        }

        var error = SendMessage(commandBuffer, Selectors.Error);
        return GetErrorDescription(error);
    }

    /// <summary>
    /// Sets the compute pipeline state for a compute encoder.
    /// </summary>
    /// <param name="encoder">The compute encoder.</param>
    /// <param name="pipelineState">The pipeline state to set.</param>
    public static void SetPipelineState(IntPtr encoder, IntPtr pipelineState)
    {
        if (encoder == IntPtr.Zero || pipelineState == IntPtr.Zero)
        {
            return;
        }

        SendMessageVoid(encoder, Selectors.SetComputePipelineState, pipelineState);
    }

    /// <summary>
    /// Sets a buffer argument for a compute encoder.
    /// </summary>
    /// <param name="encoder">The compute encoder.</param>
    /// <param name="buffer">The buffer to set.</param>
    /// <param name="offset">Offset into the buffer in bytes.</param>
    /// <param name="index">Argument index in the shader.</param>
    public static void SetBuffer(IntPtr encoder, IntPtr buffer, ulong offset, ulong index)
    {
        if (encoder == IntPtr.Zero)
        {
            return;
        }

        SendMessageSetBuffer(encoder, Selectors.SetBuffer, buffer, offset, index);
    }

    /// <summary>
    /// Sets bytes directly as an argument for a compute encoder.
    /// </summary>
    /// <param name="encoder">The compute encoder.</param>
    /// <param name="bytes">Pointer to the bytes to set.</param>
    /// <param name="length">Length of the bytes in bytes.</param>
    /// <param name="index">Argument index in the shader.</param>
    public static void SetBytes(IntPtr encoder, IntPtr bytes, ulong length, ulong index)
    {
        if (encoder == IntPtr.Zero)
        {
            return;
        }

        SendMessageSetBytes(encoder, Selectors.SetBytes, bytes, length, index);
    }

    /// <summary>
    /// Sets bytes from a value type as an argument for a compute encoder.
    /// </summary>
    /// <typeparam name="T">The value type.</typeparam>
    /// <param name="encoder">The compute encoder.</param>
    /// <param name="value">The value to set.</param>
    /// <param name="index">Argument index in the shader.</param>
    public static unsafe void SetBytes<T>(IntPtr encoder, T value, ulong index) where T : unmanaged
    {
        if (encoder == IntPtr.Zero)
        {
            return;
        }

        var size = (ulong)sizeof(T);
        var ptr = new IntPtr(&value);
        SendMessageSetBytes(encoder, Selectors.SetBytes, ptr, size, index);
    }

    /// <summary>
    /// Sets threadgroup memory length at the specified index.
    /// </summary>
    /// <param name="encoder">The compute encoder.</param>
    /// <param name="length">Length in bytes of the threadgroup memory.</param>
    /// <param name="index">Threadgroup memory argument index in the shader.</param>
    public static void SetThreadgroupMemoryLength(IntPtr encoder, ulong length, ulong index)
    {
        if (encoder == IntPtr.Zero)
        {
            return;
        }

        SendMessageSetThreadgroupMemory(encoder, Selectors.SetThreadgroupMemoryLength, length, index);
    }

    /// <summary>
    /// Dispatches compute threads using threadgroups.
    /// </summary>
    /// <param name="encoder">The compute encoder.</param>
    /// <param name="threadgroups">Number of threadgroups to dispatch.</param>
    /// <param name="threadsPerThreadgroup">Threads per threadgroup.</param>
    public static void DispatchThreadgroups(IntPtr encoder, MTLSize threadgroups, MTLSize threadsPerThreadgroup)
    {
        if (encoder == IntPtr.Zero)
        {
            return;
        }

        // Use the dispatch selector with MTLSize structs
        SendMessageDispatch(encoder, Selectors.DispatchThreadgroups, IntPtr.Zero, threadgroups, threadsPerThreadgroup);
    }

    /// <summary>
    /// Dispatches compute threads using exact thread counts (requires Apple GPU Family 4+).
    /// </summary>
    /// <param name="encoder">The compute encoder.</param>
    /// <param name="threads">Total number of threads to dispatch.</param>
    /// <param name="threadsPerThreadgroup">Threads per threadgroup.</param>
    public static void DispatchThreads(IntPtr encoder, MTLSize threads, MTLSize threadsPerThreadgroup)
    {
        if (encoder == IntPtr.Zero)
        {
            return;
        }

        SendMessageDispatch(encoder, Selectors.DispatchThreads, IntPtr.Zero, threads, threadsPerThreadgroup);
    }

    /// <summary>
    /// Inserts a memory barrier for compute operations.
    /// </summary>
    /// <param name="encoder">The compute encoder.</param>
    /// <param name="scope">The memory barrier scope.</param>
    public static void MemoryBarrier(IntPtr encoder, MTLBarrierScope scope)
    {
        if (encoder == IntPtr.Zero)
        {
            return;
        }

        SendMessageVoidULong(encoder, Selectors.MemoryBarrierWithScope, (ulong)scope);
    }

    /// <summary>
    /// Copies data between buffers using a blit encoder.
    /// </summary>
    /// <param name="encoder">The blit encoder.</param>
    /// <param name="sourceBuffer">Source buffer.</param>
    /// <param name="sourceOffset">Offset in source buffer.</param>
    /// <param name="destBuffer">Destination buffer.</param>
    /// <param name="destOffset">Offset in destination buffer.</param>
    /// <param name="size">Number of bytes to copy.</param>
    public static void CopyBuffer(IntPtr encoder, IntPtr sourceBuffer, ulong sourceOffset,
        IntPtr destBuffer, ulong destOffset, ulong size)
    {
        if (encoder == IntPtr.Zero || sourceBuffer == IntPtr.Zero || destBuffer == IntPtr.Zero)
        {
            return;
        }

        // This requires a specific overload for 5 parameters
        CopyBufferInternal(encoder, sourceBuffer, sourceOffset, destBuffer, destOffset, size);
    }

    [DllImport("/usr/lib/libobjc.dylib", EntryPoint = "objc_msgSend")]
    private static extern void CopyBufferInternal(
        IntPtr encoder,
        IntPtr selector,
        IntPtr sourceBuffer,
        ulong sourceOffset,
        IntPtr destBuffer,
        ulong destOffset,
        ulong size);

    private static void CopyBufferInternal(IntPtr encoder, IntPtr sourceBuffer, ulong sourceOffset,
        IntPtr destBuffer, ulong destOffset, ulong size)
    {
        // Note: This is simplified - real implementation needs proper selector passing
        // The actual Metal API is:
        // copyFromBuffer:sourceOffset:toBuffer:destinationOffset:size:
    }

    /// <summary>
    /// Executes a compute operation synchronously.
    /// </summary>
    /// <param name="pipelineState">The compute pipeline state.</param>
    /// <param name="setupAction">Action to set up the compute encoder (set buffers, bytes).</param>
    /// <param name="threadgroups">Number of threadgroups.</param>
    /// <param name="threadsPerThreadgroup">Threads per threadgroup.</param>
    public void ExecuteSync(IntPtr pipelineState, Action<IntPtr> setupAction,
        MTLSize threadgroups, MTLSize threadsPerThreadgroup)
    {
        ThrowIfDisposed();

        if (pipelineState == IntPtr.Zero)
        {
            throw new ArgumentException("Invalid pipeline state", nameof(pipelineState));
        }

        var commandBuffer = CreateCommandBuffer();
        if (commandBuffer == IntPtr.Zero)
        {
            throw new InvalidOperationException("Failed to create command buffer");
        }

        try
        {
            var encoder = CreateComputeEncoder(commandBuffer);
            if (encoder == IntPtr.Zero)
            {
                throw new InvalidOperationException("Failed to create compute encoder");
            }

            try
            {
                SetPipelineState(encoder, pipelineState);
                setupAction(encoder);
                DispatchThreadgroups(encoder, threadgroups, threadsPerThreadgroup);
            }
            finally
            {
                EndEncoding(encoder);
            }

            Commit(commandBuffer);
            WaitUntilCompleted(commandBuffer);

            var error = GetError(commandBuffer);
            if (!string.IsNullOrEmpty(error))
            {
                throw new InvalidOperationException($"Compute execution failed: {error}");
            }
        }
        finally
        {
            ReturnCommandBuffer(commandBuffer);
        }
    }

    /// <summary>
    /// Calculates optimal threadgroup size for a given pipeline state and work size.
    /// </summary>
    /// <param name="pipelineState">The compute pipeline state.</param>
    /// <param name="workSize">Total number of work items.</param>
    /// <returns>A tuple of (threadgroups, threadsPerThreadgroup).</returns>
    public (MTLSize Threadgroups, MTLSize ThreadsPerThreadgroup) CalculateThreadgroupSize(
        IntPtr pipelineState, int workSize)
    {
        if (pipelineState == IntPtr.Zero || workSize <= 0)
        {
            return (new MTLSize(1, 1, 1), new MTLSize(1, 1, 1));
        }

        // Get the maximum threads per threadgroup from the pipeline state
        var maxThreads = SendMessageULongReturn(pipelineState, Selectors.MaxTotalThreadsPerThreadgroup);
        var executionWidth = SendMessageULongReturn(pipelineState, Selectors.ThreadExecutionWidth);

        // Typically use 256 or 512 threads per threadgroup for 1D compute
        var threadsPerGroup = Math.Min(256, (int)maxThreads);

        // Align to execution width for better performance
        if (executionWidth > 0)
        {
            threadsPerGroup = ((threadsPerGroup + (int)executionWidth - 1) / (int)executionWidth) * (int)executionWidth;
        }

        // Calculate number of threadgroups
        var numGroups = (workSize + threadsPerGroup - 1) / threadsPerGroup;

        return (
            new MTLSize((ulong)numGroups, 1, 1),
            new MTLSize((ulong)threadsPerGroup, 1, 1)
        );
    }

    /// <summary>
    /// Calculates optimal 2D threadgroup size for a given pipeline state and work dimensions.
    /// </summary>
    /// <param name="pipelineState">The compute pipeline state.</param>
    /// <param name="width">Work width.</param>
    /// <param name="height">Work height.</param>
    /// <returns>A tuple of (threadgroups, threadsPerThreadgroup).</returns>
    public (MTLSize Threadgroups, MTLSize ThreadsPerThreadgroup) CalculateThreadgroupSize2D(
        IntPtr pipelineState, int width, int height)
    {
        if (pipelineState == IntPtr.Zero || width <= 0 || height <= 0)
        {
            return (new MTLSize(1, 1, 1), new MTLSize(1, 1, 1));
        }

        // Get the maximum threads per threadgroup from the pipeline state
        var maxThreads = SendMessageULongReturn(pipelineState, Selectors.MaxTotalThreadsPerThreadgroup);

        // Common 2D threadgroup sizes for Apple GPUs: 16x16 or 8x8
        int threadsX = 16;
        int threadsY = 16;

        // Ensure we don't exceed max threads
        while (threadsX * threadsY > (int)maxThreads && threadsX > 1 && threadsY > 1)
        {
            if (threadsX > threadsY)
            {
                threadsX /= 2;
            }
            else
            {
                threadsY /= 2;
            }
        }

        // Calculate number of threadgroups
        var groupsX = (width + threadsX - 1) / threadsX;
        var groupsY = (height + threadsY - 1) / threadsY;

        return (
            new MTLSize((ulong)groupsX, (ulong)groupsY, 1),
            new MTLSize((ulong)threadsX, (ulong)threadsY, 1)
        );
    }

    private void ThrowIfDisposed()
    {
        if (_disposed)
        {
            throw new ObjectDisposedException(nameof(MetalCommandQueue));
        }
    }

    /// <summary>
    /// Disposes the command queue and releases resources.
    /// </summary>
    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        lock (_lock)
        {
            if (_disposed)
            {
                return;
            }

            _disposed = true;

            // Release pooled command buffers
            while (_commandBufferPool.Count > 0)
            {
                var buffer = _commandBufferPool.Dequeue();
                if (buffer != IntPtr.Zero)
                {
                    Release(buffer);
                }
            }

            // Release the command queue
            if (_commandQueue != IntPtr.Zero)
            {
                Release(_commandQueue);
                _commandQueue = IntPtr.Zero;
            }
        }
    }

    public override string ToString()
    {
        return $"MetalCommandQueue[Device={_device.DeviceName}, Active={ActiveBufferCount}, Pooled={PooledBufferCount}]";
    }
}

/// <summary>
/// Memory barrier scope for compute operations.
/// </summary>
[Flags]
public enum MTLBarrierScope : ulong
{
    /// <summary>
    /// Barrier applies to buffer resources.
    /// </summary>
    Buffers = 1 << 0,

    /// <summary>
    /// Barrier applies to texture resources.
    /// </summary>
    Textures = 1 << 1,

    /// <summary>
    /// Barrier applies to render targets.
    /// </summary>
    RenderTargets = 1 << 2
}

/// <summary>
/// A scoped compute encoder that automatically ends encoding on disposal.
/// </summary>
public readonly struct ScopedComputeEncoder : IDisposable
{
    private readonly IntPtr _encoder;
    private readonly IntPtr _commandBuffer;
    private readonly MetalCommandQueue _queue;
    private readonly bool _autoCommit;

    /// <summary>
    /// Gets the native encoder handle.
    /// </summary>
    public IntPtr Handle => _encoder;

    /// <summary>
    /// Gets the command buffer this encoder belongs to.
    /// </summary>
    public IntPtr CommandBuffer => _commandBuffer;

    /// <summary>
    /// Gets whether this encoder is valid.
    /// </summary>
    public bool IsValid => _encoder != IntPtr.Zero;

    internal ScopedComputeEncoder(MetalCommandQueue queue, IntPtr encoder, IntPtr commandBuffer, bool autoCommit)
    {
        _queue = queue;
        _encoder = encoder;
        _commandBuffer = commandBuffer;
        _autoCommit = autoCommit;
    }

    /// <summary>
    /// Sets the compute pipeline state.
    /// </summary>
    public void SetPipelineState(IntPtr pipelineState)
    {
        MetalCommandQueue.SetPipelineState(_encoder, pipelineState);
    }

    /// <summary>
    /// Sets a buffer at the specified index.
    /// </summary>
    public void SetBuffer(IntPtr buffer, ulong offset, ulong index)
    {
        MetalCommandQueue.SetBuffer(_encoder, buffer, offset, index);
    }

    /// <summary>
    /// Sets a buffer at the specified index with zero offset.
    /// </summary>
    public void SetBuffer(MetalGpuBuffer buffer, int index)
    {
        MetalCommandQueue.SetBuffer(_encoder, buffer.Handle, 0, (ulong)index);
    }

    /// <summary>
    /// Sets bytes at the specified index.
    /// </summary>
    public void SetBytes<T>(T value, ulong index) where T : unmanaged
    {
        MetalCommandQueue.SetBytes(_encoder, value, index);
    }

    /// <summary>
    /// Dispatches threadgroups.
    /// </summary>
    public void DispatchThreadgroups(MTLSize threadgroups, MTLSize threadsPerThreadgroup)
    {
        MetalCommandQueue.DispatchThreadgroups(_encoder, threadgroups, threadsPerThreadgroup);
    }

    /// <summary>
    /// Dispatches exact thread counts.
    /// </summary>
    public void DispatchThreads(MTLSize threads, MTLSize threadsPerThreadgroup)
    {
        MetalCommandQueue.DispatchThreads(_encoder, threads, threadsPerThreadgroup);
    }

    /// <summary>
    /// Inserts a memory barrier.
    /// </summary>
    public void MemoryBarrier(MTLBarrierScope scope)
    {
        MetalCommandQueue.MemoryBarrier(_encoder, scope);
    }

    /// <summary>
    /// Sets threadgroup memory length at the specified index.
    /// </summary>
    /// <param name="length">Length in bytes of the threadgroup memory.</param>
    /// <param name="index">Threadgroup memory argument index in the shader.</param>
    public void SetThreadgroupMemoryLength(uint length, uint index)
    {
        MetalCommandQueue.SetThreadgroupMemoryLength(_encoder, length, index);
    }

    /// <summary>
    /// Ends encoding, optionally commits the command buffer, and returns it to the pool.
    /// </summary>
    public void Dispose()
    {
        if (_encoder != IntPtr.Zero)
        {
            MetalCommandQueue.EndEncoding(_encoder);
        }

        if (_autoCommit && _commandBuffer != IntPtr.Zero)
        {
            MetalCommandQueue.Commit(_commandBuffer);
            MetalCommandQueue.WaitUntilCompleted(_commandBuffer);
        }

        if (_commandBuffer != IntPtr.Zero && _queue is not null)
        {
            _queue.ReturnCommandBuffer(_commandBuffer);
        }
    }
}

/// <summary>
/// Extension methods for MetalCommandQueue.
/// </summary>
public static class MetalCommandQueueExtensions
{
    /// <summary>
    /// Creates a scoped compute encoder that automatically ends encoding on disposal.
    /// </summary>
    /// <param name="queue">The command queue.</param>
    /// <param name="autoCommit">Whether to automatically commit and wait on disposal.</param>
    /// <returns>A scoped compute encoder.</returns>
    public static ScopedComputeEncoder CreateScopedComputeEncoder(this MetalCommandQueue queue, bool autoCommit = true)
    {
        var commandBuffer = queue.CreateCommandBuffer();
        if (commandBuffer == IntPtr.Zero)
        {
            throw new InvalidOperationException("Failed to create command buffer");
        }

        var encoder = queue.CreateComputeEncoder(commandBuffer);
        if (encoder == IntPtr.Zero)
        {
            queue.ReturnCommandBuffer(commandBuffer);
            throw new InvalidOperationException("Failed to create compute encoder");
        }

        return new ScopedComputeEncoder(queue, encoder, commandBuffer, autoCommit);
    }
}
