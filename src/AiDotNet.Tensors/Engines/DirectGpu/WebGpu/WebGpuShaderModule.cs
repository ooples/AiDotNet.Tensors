// Copyright (c) CrowdTrainer LLC. All rights reserved.
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

        // A module is its name AND its source. Many call sites share one name for different sources ("SpectralPerf"
        // names eleven of them), and a cache on the name alone handed every later kernel the first one's module,
        // where its entry point does not exist.
        string key = SourceKey(name, wgslSource);
        if (_shaderModules.TryGetValue(key, out var moduleId))
        {
            return moduleId;
        }

        moduleId = WebGpuNativeBindings.CreateShaderModule(WebGpuShaderPrelude.Prepare(wgslSource));
        if (moduleId < 0)
        {
            throw new InvalidOperationException($"Failed to create shader module '{name}'");
        }

        _shaderModules[key] = moduleId;
        return moduleId;
    }

    // Sources are constants or cached statics, so the same source is the same string object on every call: its
    // identity is looked up first and the text is only hashed the first time an object is seen.
    private static readonly System.Runtime.CompilerServices.ConditionalWeakTable<string, string> SourceIds = new();
    private static readonly ConcurrentDictionary<string, int> SourceNumbers = new(StringComparer.Ordinal);
    private static int s_nextSourceNumber;

    private static readonly ConcurrentDictionary<string, int> WorkgroupSizes = new(StringComparer.Ordinal);

    /// <summary>
    /// The x workgroup size <paramref name="entryPoint"/> declares in <paramref name="wgslSource"/>. A one-dimensional
    /// dispatch has to launch ceil(work / that size) workgroups. Most kernels declare 256, but not all (64, 16, 8
    /// and 1 occur), and sizing every dispatch for 256 left the tail of the smaller ones' work unlaunched.
    /// </summary>
    internal static int WorkgroupSizeX(string moduleName, string wgslSource, string entryPoint)
        => WorkgroupSizes.GetOrAdd(SourceKey(moduleName, wgslSource) + "::" + entryPoint, _ =>
        {
            var match = System.Text.RegularExpressions.Regex.Match(wgslSource,
                @"@workgroup_size\(\s*(\d+)[^)]*\)\s*fn\s+" + System.Text.RegularExpressions.Regex.Escape(entryPoint) + @"\b",
                System.Text.RegularExpressions.RegexOptions.None, TimeSpan.FromSeconds(5));
            return match.Success ? int.Parse(match.Groups[1].Value, System.Globalization.CultureInfo.InvariantCulture) : 256;
        });

    /// <summary>A cache key for <paramref name="name"/> compiled from <paramref name="wgslSource"/>.</summary>
    internal static string SourceKey(string name, string wgslSource)
    {
        string id = SourceIds.GetValue(wgslSource, static source =>
            SourceNumbers.GetOrAdd(source, static _ => System.Threading.Interlocked.Increment(ref s_nextSourceNumber))
                .ToString(System.Globalization.CultureInfo.InvariantCulture));
        return name + "#" + id;
    }

    private static readonly System.Collections.Concurrent.ConcurrentDictionary<int, string> PipelineNames =
        new System.Collections.Concurrent.ConcurrentDictionary<int, string>();

    /// <summary>The module and entry point a pipeline id was created from, for the launch journal.</summary>
    internal static string DescribePipeline(int pipelineId)
        => PipelineNames.TryGetValue(pipelineId, out var name) ? name : "webgpu-pipeline#" + pipelineId;

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

        var pipelineKey = $"{SourceKey(name, wgslSource)}::{entryPoint}";

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
        PipelineNames[pipelineId] = $"webgpu:{name}::{entryPoint}";
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
    private System.Collections.Generic.List<WebGpuBuffer>? _snapshots;   // inputs copied because they alias an output
    private WebGpuBuffer[]? _original;

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
        // Every dispatch binds a group first, so this is where the launch journal sees WebGPU work (#996). The
        // work size is set by the dispatch that follows, so it is recorded as unknown (0).
        GpuKernelDiagnostics.RecordLaunch(WebGpuShaderModule.DescribePipeline(pipelineId), 0, 0, buffers.Length);

        var bufferIds = new int[buffers.Length];
        var offsets = new int[buffers.Length];
        var sizes = new int[buffers.Length];
        bool anyView = false;
        for (int i = 0; i < buffers.Length; i++)
        {
            if (buffers[i] is null || !buffers[i].IsValid)
            {
                throw new ArgumentException($"Buffer at index {i} is null or invalid", nameof(buffers));
            }
        }

        // An in-place call binds one buffer twice, as the kernel's input and as its output. WebGPU rejects a pass in
        // which a buffer is writable and also bound anywhere else, so the earlier binding (the input, by this
        // backend's convention) gets a snapshot of the buffer taken before the dispatch.
        for (int i = 0; i < buffers.Length; i++)
        {
            for (int later = i + 1; later < buffers.Length; later++)
            {
                if (buffers[later].BufferId != buffers[i].BufferId || buffers[later].OffsetBytes != buffers[i].OffsetBytes
                    || buffers[later].SizeBytes != buffers[i].SizeBytes) continue;
                if (ReferenceEquals(buffers, _original ??= buffers)) buffers = (WebGpuBuffer[])buffers.Clone();
                var snapshot = new WebGpuBuffer(buffers[i].Size);
                snapshot.CopyFromBuffer(buffers[i], 0, 0, buffers[i].Size);
                (_snapshots ??= new System.Collections.Generic.List<WebGpuBuffer>()).Add(snapshot);
                buffers[i] = snapshot;
                break;
            }
        }

        for (int i = 0; i < buffers.Length; i++)
        {
            bufferIds[i] = buffers[i].BufferId;
            offsets[i] = buffers[i].OffsetBytes;
            sizes[i] = buffers[i].SizeBytes;
            anyView |= buffers[i].IsView;
        }

        // A view binds only its own byte range of the shared buffer (#1142); whole buffers keep the plain call.
        _bindGroupId = anyView
            ? WebGpuNativeBindings.CreateBindGroupRanges(pipelineId, bufferIds, offsets, sizes)
            : WebGpuNativeBindings.CreateBindGroup(pipelineId, bufferIds);
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
            if (_snapshots is not null)
            {
                foreach (var snapshot in _snapshots) snapshot.Dispose();
                _snapshots = null;
            }
        }
    }
}
#endif
