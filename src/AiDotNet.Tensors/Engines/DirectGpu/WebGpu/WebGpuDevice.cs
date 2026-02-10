// Copyright (c) AiDotNet. All rights reserved.
// WebGPU device management for browser GPU compute.
// Only available in .NET 7+ with Blazor WebAssembly.

#if NET7_0_OR_GREATER
using System;
using System.Text.Json;
using System.Threading.Tasks;

namespace AiDotNet.Tensors.Engines.DirectGpu.WebGpu;

/// <summary>
/// Manages WebGPU device initialization and capabilities.
/// </summary>
/// <remarks>
/// <para><b>WebGPU Device Model:</b></para>
/// <para>
/// WebGPU uses a two-level device model:
/// </para>
/// <list type="bullet">
/// <item><b>GPU Adapter</b>: Represents a physical GPU (navigator.gpu.requestAdapter)</item>
/// <item><b>GPU Device</b>: Logical connection for GPU operations (adapter.requestDevice)</item>
/// </list>
/// <para><b>Browser Compatibility:</b></para>
/// <para>
/// WebGPU is supported in Chrome 113+, Edge 113+, and Firefox (behind flag).
/// Safari support is experimental. Always check IsSupported before use.
/// </para>
/// </remarks>
public sealed class WebGpuDevice : IDisposable
{
    private static WebGpuDevice? _instance;
    private static readonly object _lock = new();

    private bool _initialized;
    private bool _disposed;
    private string _adapterInfo = string.Empty;
    private WebGpuDeviceLimits? _limits;

    /// <summary>
    /// Gets the singleton device instance.
    /// </summary>
    public static WebGpuDevice Instance
    {
        get
        {
            if (_instance is null)
            {
                lock (_lock)
                {
                    _instance ??= new WebGpuDevice();
                }
            }
            return _instance;
        }
    }

    /// <summary>
    /// Gets whether WebGPU is supported in the current browser.
    /// </summary>
    public static bool IsSupported => WebGpuNativeBindings.IsWebGpuSupported();

    /// <summary>
    /// Gets whether the device has been initialized.
    /// </summary>
    public bool IsInitialized => _initialized && !_disposed;

    /// <summary>
    /// Gets the adapter information string.
    /// </summary>
    public string AdapterInfo => _adapterInfo;

    /// <summary>
    /// Gets the device limits.
    /// </summary>
    public WebGpuDeviceLimits? Limits => _limits;

    /// <summary>
    /// Gets the maximum buffer size in bytes.
    /// </summary>
    public long MaxBufferSize => _limits?.MaxBufferSize ?? 256 * 1024 * 1024; // 256MB default

    /// <summary>
    /// Gets the maximum workgroup size.
    /// </summary>
    public int MaxWorkgroupSize => _limits?.MaxComputeInvocationsPerWorkgroup ?? 256;

    /// <summary>
    /// Gets the maximum workgroups per dimension.
    /// </summary>
    public int MaxWorkgroupsPerDimension => _limits?.MaxComputeWorkgroupsPerDimension ?? 65535;

    private WebGpuDevice()
    {
    }

    /// <summary>
    /// Initializes the WebGPU device.
    /// </summary>
    /// <returns>True if initialization succeeded.</returns>
    public async Task<bool> InitializeAsync()
    {
        if (_initialized)
        {
            return true;
        }

        if (!IsSupported)
        {
            return false;
        }

        var success = await WebGpuNativeBindings.InitializeWebGpuAsync();
        if (!success)
        {
            return false;
        }

        _initialized = true;
        _adapterInfo = WebGpuNativeBindings.GetAdapterInfo();

        // Parse device limits
        var limitsJson = WebGpuNativeBindings.GetDeviceLimitsJson();
        if (!string.IsNullOrEmpty(limitsJson))
        {
            try
            {
                _limits = JsonSerializer.Deserialize<WebGpuDeviceLimits>(limitsJson, new JsonSerializerOptions
                {
                    PropertyNameCaseInsensitive = true
                });
            }
            catch
            {
                // Use defaults if parsing fails
                _limits = new WebGpuDeviceLimits
                {
                    MaxBufferSize = 256L * 1024 * 1024,
                    MaxStorageBufferBindingSize = 128L * 1024 * 1024,
                    MaxUniformBufferBindingSize = 64 * 1024,
                    MaxComputeWorkgroupSizeX = 256,
                    MaxComputeWorkgroupSizeY = 256,
                    MaxComputeWorkgroupSizeZ = 64,
                    MaxComputeInvocationsPerWorkgroup = 256,
                    MaxComputeWorkgroupsPerDimension = 65535,
                    MaxBindGroups = 4,
                    MaxBindingsPerBindGroup = 1000,
                    MaxStorageBuffersPerShaderStage = 8
                };
            }
        }

        return true;
    }

    /// <summary>
    /// Creates a GPU buffer.
    /// </summary>
    /// <param name="elementCount">Number of float elements.</param>
    /// <param name="usage">Buffer usage flags.</param>
    /// <returns>A new WebGPU buffer.</returns>
    public WebGpuBuffer CreateBuffer(int elementCount, WebGpuBufferUsage usage = WebGpuBufferUsage.Storage | WebGpuBufferUsage.CopySrc | WebGpuBufferUsage.CopyDst)
    {
        ThrowIfNotInitialized();
        return new WebGpuBuffer(elementCount, usage);
    }

    /// <summary>
    /// Creates a GPU buffer with initial data.
    /// </summary>
    /// <param name="data">Initial data to upload.</param>
    /// <param name="usage">Buffer usage flags.</param>
    /// <returns>A new WebGPU buffer.</returns>
    public WebGpuBuffer CreateBuffer(float[] data, WebGpuBufferUsage usage = WebGpuBufferUsage.Storage | WebGpuBufferUsage.CopySrc | WebGpuBufferUsage.CopyDst)
    {
        ThrowIfNotInitialized();
        return new WebGpuBuffer(data, usage);
    }

    /// <summary>
    /// Creates a uniform buffer for shader parameters.
    /// </summary>
    /// <param name="sizeBytes">Size in bytes (must be multiple of 16).</param>
    /// <returns>A new uniform buffer.</returns>
    public WebGpuBuffer CreateUniformBuffer(int sizeBytes)
    {
        ThrowIfNotInitialized();

        // Uniform buffers must be aligned to 16 bytes
        var alignedSize = (sizeBytes + 15) & ~15;
        var elementCount = (alignedSize + sizeof(float) - 1) / sizeof(float);

        return new WebGpuBuffer(elementCount, WebGpuBufferUsage.Uniform | WebGpuBufferUsage.CopyDst);
    }

    /// <summary>
    /// Calculates optimal workgroup configuration for 1D dispatch.
    /// </summary>
    /// <param name="totalWork">Total number of work items.</param>
    /// <param name="workgroupSize">Preferred workgroup size (default 256).</param>
    /// <returns>Number of workgroups needed.</returns>
    public (int Workgroups, int WorkgroupSize) CalculateWorkgroups1D(int totalWork, int workgroupSize = 256)
    {
        // Clamp to device limits
        workgroupSize = Math.Min(workgroupSize, _limits?.MaxComputeWorkgroupSizeX ?? 256);
        var workgroups = (totalWork + workgroupSize - 1) / workgroupSize;
        workgroups = Math.Min(workgroups, _limits?.MaxComputeWorkgroupsPerDimension ?? 65535);

        return (workgroups, workgroupSize);
    }

    /// <summary>
    /// Calculates optimal workgroup configuration for 2D dispatch.
    /// </summary>
    /// <param name="width">Width dimension.</param>
    /// <param name="height">Height dimension.</param>
    /// <param name="workgroupSizeX">Workgroup size X (default 16).</param>
    /// <param name="workgroupSizeY">Workgroup size Y (default 16).</param>
    /// <returns>Number of workgroups in each dimension.</returns>
    public (int WorkgroupsX, int WorkgroupsY, int SizeX, int SizeY) CalculateWorkgroups2D(
        int width, int height, int workgroupSizeX = 16, int workgroupSizeY = 16)
    {
        // Clamp to device limits
        var maxInvocations = _limits?.MaxComputeInvocationsPerWorkgroup ?? 256;
        while (workgroupSizeX * workgroupSizeY > maxInvocations)
        {
            if (workgroupSizeX > workgroupSizeY)
            {
                workgroupSizeX /= 2;
            }
            else
            {
                workgroupSizeY /= 2;
            }
        }

        var workgroupsX = (width + workgroupSizeX - 1) / workgroupSizeX;
        var workgroupsY = (height + workgroupSizeY - 1) / workgroupSizeY;

        var maxWg = _limits?.MaxComputeWorkgroupsPerDimension ?? 65535;
        workgroupsX = Math.Min(workgroupsX, maxWg);
        workgroupsY = Math.Min(workgroupsY, maxWg);

        return (workgroupsX, workgroupsY, workgroupSizeX, workgroupSizeY);
    }

    private void ThrowIfNotInitialized()
    {
        if (_disposed)
        {
            throw new ObjectDisposedException(nameof(WebGpuDevice));
        }

        if (!_initialized)
        {
            throw new InvalidOperationException("WebGPU device not initialized. Call InitializeAsync() first.");
        }
    }

    /// <summary>
    /// Disposes the device and releases resources.
    /// </summary>
    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        _disposed = true;

        if (_initialized)
        {
            WebGpuNativeBindings.DestroyDevice();
            _initialized = false;
        }
    }

    public override string ToString()
    {
        return $"WebGpuDevice[Initialized={_initialized}, Adapter={_adapterInfo}]";
    }
}
#endif
