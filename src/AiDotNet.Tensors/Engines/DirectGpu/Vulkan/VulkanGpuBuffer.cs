// Copyright (c) AiDotNet. All rights reserved.
// IGpuBuffer wrapper for Vulkan storage and staging buffer pairs.

using System;

namespace AiDotNet.Tensors.Engines.DirectGpu.Vulkan;

/// <summary>
/// Wraps a Vulkan device-local storage buffer and host-visible staging buffer
/// as an <see cref="IGpuBuffer"/> for use with <see cref="IDirectGpuBackend"/> operations.
/// </summary>
public sealed class VulkanGpuBuffer : IGpuBuffer
{
    internal readonly VulkanBuffer Storage;
    internal readonly VulkanBuffer Staging;
    private bool _disposed;

    /// <inheritdoc/>
    public int Size { get; }

    /// <inheritdoc/>
    public long SizeInBytes => (long)Size * sizeof(float);

    /// <inheritdoc/>
    public IntPtr Handle => Storage.Handle;

    internal VulkanGpuBuffer(VulkanBuffer storage, VulkanBuffer staging, int size)
    {
        Storage = storage ?? throw new ArgumentNullException(nameof(storage));
        Staging = staging ?? throw new ArgumentNullException(nameof(staging));
        if (size <= 0)
            throw new ArgumentOutOfRangeException(nameof(size), "Element count must be positive.");
        Size = size;
    }

    /// <summary>
    /// Creates a new VulkanGpuBuffer with the given element count.
    /// </summary>
    internal static VulkanGpuBuffer Create(int size)
    {
        if (size <= 0)
            throw new ArgumentOutOfRangeException(nameof(size), "Element count must be positive.");

        var storage = VulkanBuffer.CreateStorageBuffer(size);
        var staging = VulkanBuffer.CreateStagingBuffer(size);
        if (storage is null || staging is null)
        {
            storage?.Dispose();
            staging?.Dispose();
            throw new InvalidOperationException("Failed to allocate Vulkan GPU buffer.");
        }

        return new VulkanGpuBuffer(storage, staging, size);
    }

    /// <summary>
    /// Creates a new VulkanGpuBuffer and uploads initial data.
    /// </summary>
    /// <remarks>
    /// The upload is performed synchronously via a staging buffer copy.
    /// For large data sets, consider pre-allocating and uploading in batches.
    /// If upload fails, the buffer is disposed to prevent resource leaks.
    /// </remarks>
    internal static VulkanGpuBuffer Create(float[] data, VulkanBufferTransfer transfer)
    {
        if (data is null) throw new ArgumentNullException(nameof(data));
        if (data.Length == 0) throw new ArgumentException("Data array must not be empty.", nameof(data));
        if (transfer is null) throw new ArgumentNullException(nameof(transfer));

        var buffer = Create(data.Length);
        try
        {
            buffer.Staging.WriteData(data);
            transfer.CopyToDevice(buffer.Staging, buffer.Storage);
        }
        catch
        {
            buffer.Dispose();
            throw;
        }
        return buffer;
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        Storage.Dispose();
        Staging.Dispose();
    }
}
