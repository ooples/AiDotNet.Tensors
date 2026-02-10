// Copyright (c) AiDotNet. All rights reserved.
// Metal device wrapper for Apple Silicon GPU management.

using System;
using System.Runtime.InteropServices;
using static AiDotNet.Tensors.Engines.DirectGpu.Metal.MetalNativeBindings;

namespace AiDotNet.Tensors.Engines.DirectGpu.Metal;

/// <summary>
/// Wrapper for a Metal device (GPU) providing device information and resource creation.
/// </summary>
/// <remarks>
/// <para><b>Apple Silicon Unified Memory:</b></para>
/// <para>
/// On Apple Silicon (M1, M2, M3, M4), the CPU and GPU share unified memory.
/// This eliminates the need for explicit data transfers between CPU and GPU,
/// providing significant performance advantages for memory-bound workloads.
/// </para>
/// <para><b>Supported GPU Families:</b></para>
/// <list type="bullet">
/// <item>Apple GPU Family 7+ (M1 and later) - Full feature support</item>
/// <item>Apple GPU Family 8+ (M2 and later) - Enhanced ray tracing</item>
/// <item>Apple GPU Family 9+ (M3 and later) - Dynamic caching, mesh shaders</item>
/// </list>
/// </remarks>
public sealed class MetalDevice : IDisposable
{
    private IntPtr _device;
    private bool _disposed;
    private readonly object _lock = new();

    /// <summary>
    /// Gets whether Metal is available on this system.
    /// </summary>
    public static bool IsMetalAvailable
    {
        get
        {
            if (!IsPlatformSupported)
            {
                return false;
            }

            try
            {
                var device = CreateDefaultDevice();
                if (device != IntPtr.Zero)
                {
                    Release(device);
                    return true;
                }
                return false;
            }
            catch
            {
                return false;
            }
        }
    }

    /// <summary>
    /// Gets the native Metal device handle.
    /// </summary>
    public IntPtr Handle => _device;

    /// <summary>
    /// Gets whether the device is valid and available.
    /// </summary>
    public bool IsAvailable => _device != IntPtr.Zero;

    /// <summary>
    /// Gets the device name (e.g., "Apple M2 Pro").
    /// </summary>
    public string DeviceName { get; }

    /// <summary>
    /// Gets the maximum number of threads per threadgroup.
    /// </summary>
    public MTLSize MaxThreadsPerThreadgroup { get; }

    /// <summary>
    /// Gets the maximum threadgroup memory length in bytes.
    /// </summary>
    public ulong MaxThreadgroupMemoryLength { get; }

    /// <summary>
    /// Gets the recommended maximum working set size in bytes.
    /// </summary>
    public ulong RecommendedMaxWorkingSetSize { get; }

    /// <summary>
    /// Gets the current allocated memory size in bytes.
    /// </summary>
    public ulong CurrentAllocatedSize
    {
        get
        {
            ThrowIfDisposed();
            return SendMessageULongReturn(_device, Selectors.CurrentAllocatedSize);
        }
    }

    /// <summary>
    /// Gets the GPU registry ID (unique identifier).
    /// </summary>
    public ulong RegistryID { get; }

    /// <summary>
    /// Gets whether the device supports the Apple GPU Family 7 (M1+).
    /// </summary>
    public bool SupportsApple7 { get; }

    /// <summary>
    /// Gets whether the device supports the Apple GPU Family 8 (M2+).
    /// </summary>
    public bool SupportsApple8 { get; }

    /// <summary>
    /// Gets whether the device supports the Apple GPU Family 9 (M3+).
    /// </summary>
    public bool SupportsApple9 { get; }

    /// <summary>
    /// Gets whether the device supports Metal 3 features.
    /// </summary>
    public bool SupportsMetal3 { get; }

    /// <summary>
    /// Gets the estimated number of GPU cores.
    /// </summary>
    public int EstimatedGPUCores { get; }

    /// <summary>
    /// Creates a new Metal device wrapper using the system default device.
    /// </summary>
    public MetalDevice() : this(0)
    {
    }

    /// <summary>
    /// Creates a new Metal device wrapper for the specified device index.
    /// </summary>
    /// <param name="deviceIndex">The device index (0 = default device).</param>
    public MetalDevice(int deviceIndex)
    {
        if (!IsPlatformSupported)
        {
            DeviceName = "Not Available";
            MaxThreadsPerThreadgroup = new MTLSize(1, 1, 1);
            return;
        }

        try
        {
            if (deviceIndex == 0)
            {
                _device = CreateDefaultDevice();
            }
            else
            {
                // Get all devices and select by index
                var allDevices = CopyAllDevices();
                if (allDevices != IntPtr.Zero)
                {
                    var count = (int)SendMessageULongReturn(allDevices, Selectors.Count);
                    if (deviceIndex < count)
                    {
                        _device = SendMessage(allDevices, Selectors.ObjectAtIndex, (IntPtr)deviceIndex);
                        if (_device != IntPtr.Zero)
                        {
                            Retain(_device);
                        }
                    }
                    Release(allDevices);
                }
            }

            if (_device == IntPtr.Zero)
            {
                DeviceName = "Not Available";
                MaxThreadsPerThreadgroup = new MTLSize(1, 1, 1);
                return;
            }

            // Get device name
            var namePtr = SendMessage(_device, Selectors.Name);
            DeviceName = GetStringFromNSString(namePtr) ?? "Unknown Metal Device";

            // Get max threads per threadgroup
            MaxThreadsPerThreadgroup = GetMaxThreadsPerThreadgroup();

            // Get max threadgroup memory
            MaxThreadgroupMemoryLength = SendMessageULongReturn(_device, Selectors.MaxThreadgroupMemoryLength);

            // Get recommended working set size
            RecommendedMaxWorkingSetSize = SendMessageULongReturn(_device, Selectors.RecommendedMaxWorkingSetSize);

            // Get registry ID
            RegistryID = SendMessageULongReturn(_device, Selectors.RegistryID);

            // Check GPU family support
            SupportsApple7 = SupportsGPUFamily(MTLGPUFamily.Apple7);
            SupportsApple8 = SupportsGPUFamily(MTLGPUFamily.Apple8);
            SupportsApple9 = SupportsGPUFamily(MTLGPUFamily.Apple9);
            SupportsMetal3 = SupportsGPUFamily(MTLGPUFamily.Metal3);

            // Estimate GPU cores based on device name and family
            EstimatedGPUCores = EstimateGPUCores();
        }
        catch (Exception ex)
        {
            DeviceName = $"Error: {ex.Message}";
            MaxThreadsPerThreadgroup = new MTLSize(1, 1, 1);
            _device = IntPtr.Zero;
        }
    }

    /// <summary>
    /// Gets the maximum threads per threadgroup as an MTLSize.
    /// </summary>
    private MTLSize GetMaxThreadsPerThreadgroup()
    {
        // On Apple Silicon, typical max is 1024 total threads.
        // The product of all dimensions must not exceed the device's maximum.
        // M1/M2/M3/M4 support up to 1024 total threads per threadgroup.
        // Using (1024, 1, 1) as a safe conservative default that works on all Metal devices.
        // Callers should reshape dimensions based on their specific dispatch needs.

        return new MTLSize(1024, 1, 1);
    }

    /// <summary>
    /// Checks if the device supports a specific GPU family.
    /// </summary>
    public bool SupportsGPUFamily(MTLGPUFamily family)
    {
        if (_device == IntPtr.Zero)
        {
            return false;
        }

        try
        {
            return SendMessageBool(_device, Selectors.SupportsFamily, (IntPtr)(long)family);
        }
        catch
        {
            return false;
        }
    }

    /// <summary>
    /// Estimates the number of GPU cores based on device name.
    /// </summary>
    private int EstimateGPUCores()
    {
        var name = DeviceName.ToUpperInvariant();

        // M4 series (2024)
        if (name.Contains("M4 MAX")) return 40;
        if (name.Contains("M4 PRO")) return 20;
        if (name.Contains("M4")) return 10;

        // M3 series (2023)
        if (name.Contains("M3 MAX")) return 40;
        if (name.Contains("M3 PRO")) return 18;
        if (name.Contains("M3")) return 10;

        // M2 series (2022-2023)
        if (name.Contains("M2 ULTRA")) return 76;
        if (name.Contains("M2 MAX")) return 38;
        if (name.Contains("M2 PRO")) return 19;
        if (name.Contains("M2")) return 10;

        // M1 series (2020-2022)
        if (name.Contains("M1 ULTRA")) return 64;
        if (name.Contains("M1 MAX")) return 32;
        if (name.Contains("M1 PRO")) return 16;
        if (name.Contains("M1")) return 8;

        // Intel Macs with AMD GPUs
        if (name.Contains("RADEON PRO")) return 36;
        if (name.Contains("RADEON")) return 24;

        // Default for unknown devices
        return 8;
    }

    /// <summary>
    /// Creates a new buffer with the specified size.
    /// </summary>
    /// <param name="sizeInBytes">Size of the buffer in bytes.</param>
    /// <param name="options">Resource options for the buffer.</param>
    /// <returns>Handle to the new buffer.</returns>
    public IntPtr CreateBuffer(ulong sizeInBytes, MTLResourceOptions options = MTLResourceOptions.StorageModeShared)
    {
        ThrowIfDisposed();

        if (sizeInBytes == 0)
        {
            throw new ArgumentException("Buffer size must be greater than zero.", nameof(sizeInBytes));
        }

        return SendMessageULong2(_device, Selectors.NewBufferWithLength, sizeInBytes, (ulong)options);
    }

    /// <summary>
    /// Creates a new buffer with initial data.
    /// </summary>
    /// <param name="data">Pointer to the initial data.</param>
    /// <param name="sizeInBytes">Size of the data in bytes.</param>
    /// <param name="options">Resource options for the buffer.</param>
    /// <returns>Handle to the new buffer.</returns>
    public IntPtr CreateBufferWithData(IntPtr data, ulong sizeInBytes, MTLResourceOptions options = MTLResourceOptions.StorageModeShared)
    {
        ThrowIfDisposed();

        if (sizeInBytes == 0)
        {
            throw new ArgumentException("Buffer size must be greater than zero.", nameof(sizeInBytes));
        }

        return SendMessagePtr(_device, Selectors.NewBufferWithBytes, data, sizeInBytes, (ulong)options);
    }

    /// <summary>
    /// Creates a new command queue.
    /// </summary>
    /// <returns>Handle to the new command queue.</returns>
    public IntPtr CreateCommandQueue()
    {
        ThrowIfDisposed();
        return SendMessage(_device, Selectors.NewCommandQueue);
    }

    /// <summary>
    /// Creates a new command queue with a maximum command buffer count.
    /// </summary>
    /// <param name="maxCommandBufferCount">Maximum number of command buffers.</param>
    /// <returns>Handle to the new command queue.</returns>
    public IntPtr CreateCommandQueue(ulong maxCommandBufferCount)
    {
        ThrowIfDisposed();
        return SendMessageULong(_device, Selectors.NewCommandQueueWithMaxCommandBufferCount, maxCommandBufferCount);
    }

    /// <summary>
    /// Creates a new library from Metal Shading Language source code.
    /// </summary>
    /// <param name="source">The MSL source code.</param>
    /// <returns>Handle to the compiled library, or IntPtr.Zero on error.</returns>
    public IntPtr CreateLibrary(string source, out string? error)
    {
        ThrowIfDisposed();
        error = null;

        var sourceNS = CreateNSString(source);
        if (sourceNS == IntPtr.Zero)
        {
            error = "Failed to create NSString from source";
            return IntPtr.Zero;
        }

        try
        {
            IntPtr errorPtr = IntPtr.Zero;
            var library = MetalNativeBindings.SendMessageWithError(_device, Selectors.NewLibraryWithSource, sourceNS, IntPtr.Zero, ref errorPtr);

            if (library == IntPtr.Zero && errorPtr != IntPtr.Zero)
            {
                error = GetErrorDescription(errorPtr);
            }

            return library;
        }
        finally
        {
            Release(sourceNS);
        }
    }

    /// <summary>
    /// Creates a compute pipeline state from a function.
    /// </summary>
    /// <param name="function">The compute function.</param>
    /// <returns>Handle to the pipeline state, or IntPtr.Zero on error.</returns>
    public IntPtr CreateComputePipelineState(IntPtr function, out string? error)
    {
        ThrowIfDisposed();
        error = null;

        if (function == IntPtr.Zero)
        {
            error = "Function handle is null";
            return IntPtr.Zero;
        }

        IntPtr errorPtr = IntPtr.Zero;
        var pipelineState = MetalNativeBindings.SendMessageWithError(_device, Selectors.NewComputePipelineStateWithFunction, function, ref errorPtr);

        if (pipelineState == IntPtr.Zero && errorPtr != IntPtr.Zero)
        {
            error = GetErrorDescription(errorPtr);
        }

        return pipelineState;
    }

    /// <summary>
    /// Gets information about all available Metal devices.
    /// </summary>
    public static MetalDeviceInfo[] GetAvailableDevices()
    {
        if (!IsPlatformSupported)
        {
            return Array.Empty<MetalDeviceInfo>();
        }

        try
        {
            var devices = new System.Collections.Generic.List<MetalDeviceInfo>();
            var allDevices = CopyAllDevices();

            if (allDevices != IntPtr.Zero)
            {
                var count = (int)SendMessageULongReturn(allDevices, Selectors.Count);

                for (int i = 0; i < count; i++)
                {
                    var device = SendMessage(allDevices, Selectors.ObjectAtIndex, (IntPtr)i);
                    if (device != IntPtr.Zero)
                    {
                        var namePtr = SendMessage(device, Selectors.Name);
                        var name = GetStringFromNSString(namePtr) ?? "Unknown";
                        var registryId = SendMessageULongReturn(device, Selectors.RegistryID);
                        var workingSetSize = SendMessageULongReturn(device, Selectors.RecommendedMaxWorkingSetSize);

                        devices.Add(new MetalDeviceInfo
                        {
                            Index = i,
                            Name = name,
                            RegistryID = registryId,
                            RecommendedMaxWorkingSetSize = workingSetSize
                        });
                    }
                }

                Release(allDevices);
            }

            return devices.ToArray();
        }
        catch
        {
            return Array.Empty<MetalDeviceInfo>();
        }
    }

    private void ThrowIfDisposed()
    {
        if (_disposed)
        {
            throw new ObjectDisposedException(nameof(MetalDevice));
        }
    }

    /// <summary>
    /// Disposes the Metal device.
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

            if (_device != IntPtr.Zero)
            {
                Release(_device);
                _device = IntPtr.Zero;
            }
        }
    }
}

/// <summary>
/// Information about a Metal device.
/// </summary>
public readonly struct MetalDeviceInfo
{
    /// <summary>
    /// Device index.
    /// </summary>
    public int Index { get; init; }

    /// <summary>
    /// Device name (e.g., "Apple M2 Pro").
    /// </summary>
    public string Name { get; init; }

    /// <summary>
    /// Unique registry identifier.
    /// </summary>
    public ulong RegistryID { get; init; }

    /// <summary>
    /// Recommended maximum working set size in bytes.
    /// </summary>
    public ulong RecommendedMaxWorkingSetSize { get; init; }

    /// <summary>
    /// Working set size formatted as human-readable string.
    /// </summary>
    public string FormattedWorkingSetSize => RecommendedMaxWorkingSetSize switch
    {
        >= 1024UL * 1024 * 1024 * 1024 => $"{RecommendedMaxWorkingSetSize / (1024.0 * 1024 * 1024 * 1024):F1} TB",
        >= 1024UL * 1024 * 1024 => $"{RecommendedMaxWorkingSetSize / (1024.0 * 1024 * 1024):F1} GB",
        >= 1024UL * 1024 => $"{RecommendedMaxWorkingSetSize / (1024.0 * 1024):F1} MB",
        _ => $"{RecommendedMaxWorkingSetSize} bytes"
    };

    public override string ToString() => $"{Name} ({FormattedWorkingSetSize})";
}
