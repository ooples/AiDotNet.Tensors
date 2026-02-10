// Copyright (c) AiDotNet. All rights reserved.
// WebGPU shader module and compute pipeline management.
// Only available in .NET 7+ with Blazor WebAssembly.

#if NET7_0_OR_GREATER
using System;
using System.Collections.Concurrent;
using System.Threading.Tasks;

namespace AiDotNet.Tensors.Engines.DirectGpu.WebGpu;

/// <summary>
/// Manages WebGPU shader modules and compute pipelines.
/// </summary>
/// <remarks>
/// <para><b>WGSL Shader Compilation:</b></para>
/// <para>
/// WebGPU uses WGSL (WebGPU Shading Language) for compute shaders.
/// Shaders are compiled at runtime by the browser's WebGPU implementation.
/// </para>
/// <para><b>Pipeline Creation:</b></para>
/// <para>
/// Compute pipelines are created from shader modules asynchronously.
/// Pipelines are cached to avoid redundant compilation.
/// </para>
/// </remarks>
public sealed class WebGpuShaderModule : IDisposable
{
    private readonly ConcurrentDictionary<string, int> _shaderModules = new();
    private readonly ConcurrentDictionary<string, int> _pipelines = new();
    private bool _disposed;

    /// <summary>
    /// Gets the number of cached shader modules.
    /// </summary>
    public int CachedModuleCount => _shaderModules.Count;

    /// <summary>
    /// Gets the number of cached pipelines.
    /// </summary>
    public int CachedPipelineCount => _pipelines.Count;

    /// <summary>
    /// Creates or retrieves a cached shader module.
    /// </summary>
    /// <param name="name">Unique name for the shader.</param>
    /// <param name="wgslSource">WGSL source code.</param>
    /// <returns>Shader module ID.</returns>
    public int GetOrCreateModule(string name, string wgslSource)
    {
        ThrowIfDisposed();

        if (_shaderModules.TryGetValue(name, out var moduleId))
        {
            return moduleId;
        }

        moduleId = WebGpuNativeBindings.CreateShaderModule(wgslSource);
        if (moduleId < 0)
        {
            throw new InvalidOperationException($"Failed to create shader module '{name}'");
        }

        _shaderModules[name] = moduleId;
        return moduleId;
    }

    /// <summary>
    /// Creates or retrieves a cached compute pipeline.
    /// </summary>
    /// <param name="name">Unique name for the pipeline.</param>
    /// <param name="wgslSource">WGSL source code.</param>
    /// <param name="entryPoint">Entry point function name.</param>
    /// <returns>Pipeline ID.</returns>
    public async Task<int> GetOrCreatePipelineAsync(string name, string wgslSource, string entryPoint)
    {
        ThrowIfDisposed();

        var pipelineKey = $"{name}::{entryPoint}";

        if (_pipelines.TryGetValue(pipelineKey, out var pipelineId))
        {
            return pipelineId;
        }

        var moduleId = GetOrCreateModule(name, wgslSource);
        pipelineId = await WebGpuNativeBindings.CreateComputePipelineAsync(moduleId, entryPoint);

        if (pipelineId < 0)
        {
            throw new InvalidOperationException($"Failed to create compute pipeline '{name}::{entryPoint}'");
        }

        _pipelines[pipelineKey] = pipelineId;
        return pipelineId;
    }

    /// <summary>
    /// Clears all cached modules and pipelines.
    /// </summary>
    public void ClearCache()
    {
        foreach (var kvp in _pipelines)
        {
            WebGpuNativeBindings.DestroyPipeline(kvp.Value);
        }
        _pipelines.Clear();

        foreach (var kvp in _shaderModules)
        {
            WebGpuNativeBindings.DestroyShaderModule(kvp.Value);
        }
        _shaderModules.Clear();
    }

    private void ThrowIfDisposed()
    {
        if (_disposed)
        {
            throw new ObjectDisposedException(nameof(WebGpuShaderModule));
        }
    }

    /// <summary>
    /// Disposes all cached resources.
    /// </summary>
    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        _disposed = true;
        ClearCache();
    }
}

/// <summary>
/// Helper for creating bind groups for compute dispatch.
/// </summary>
public sealed class WebGpuBindGroup : IDisposable
{
    private readonly int _bindGroupId;
    private readonly int _pipelineId;
    private bool _disposed;

    /// <summary>
    /// Gets the bind group ID.
    /// </summary>
    public int BindGroupId => _bindGroupId;

    /// <summary>
    /// Gets the pipeline ID this bind group is for.
    /// </summary>
    public int PipelineId => _pipelineId;

    /// <summary>
    /// Gets whether this bind group is valid.
    /// </summary>
    public bool IsValid => _bindGroupId >= 0 && !_disposed;

    /// <summary>
    /// Creates a bind group for a compute pipeline.
    /// </summary>
    /// <param name="pipelineId">Pipeline ID.</param>
    /// <param name="buffers">Buffers to bind.</param>
    public WebGpuBindGroup(int pipelineId, params WebGpuBuffer[] buffers)
    {
        if (buffers is null || buffers.Length == 0)
        {
            throw new ArgumentException("At least one buffer must be provided", nameof(buffers));
        }

        _pipelineId = pipelineId;

        var bufferIds = new int[buffers.Length];
        for (int i = 0; i < buffers.Length; i++)
        {
            if (buffers[i] is null || !buffers[i].IsValid)
            {
                throw new ArgumentException($"Buffer at index {i} is null or invalid", nameof(buffers));
            }
            bufferIds[i] = buffers[i].BufferId;
        }

        _bindGroupId = WebGpuNativeBindings.CreateBindGroup(pipelineId, bufferIds);
        if (_bindGroupId < 0)
        {
            throw new InvalidOperationException("Failed to create bind group");
        }
    }

    /// <summary>
    /// Disposes the bind group.
    /// </summary>
    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        _disposed = true;

        if (_bindGroupId >= 0)
        {
            WebGpuNativeBindings.DestroyBindGroup(_bindGroupId);
        }
    }
}
#endif
