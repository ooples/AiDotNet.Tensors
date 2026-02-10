// Copyright (c) AiDotNet. All rights reserved.
// Metal shader library management for compute kernel compilation and caching.

using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using static AiDotNet.Tensors.Engines.DirectGpu.Metal.MetalNativeBindings;

namespace AiDotNet.Tensors.Engines.DirectGpu.Metal;

/// <summary>
/// Manages Metal shader compilation, caching, and compute pipeline states.
/// </summary>
/// <remarks>
/// <para><b>Shader Compilation Pipeline:</b></para>
/// <list type="number">
/// <item>MSL source code is compiled to MTLLibrary</item>
/// <item>MTLFunction is extracted from the library by name</item>
/// <item>MTLComputePipelineState is created from the function</item>
/// <item>Pipeline state is cached for reuse</item>
/// </list>
/// <para><b>Performance Considerations:</b></para>
/// <para>
/// Shader compilation is expensive. This class caches both libraries and
/// pipeline states to avoid redundant compilation. For production use,
/// consider pre-compiling shaders to metallib format.
/// </para>
/// </remarks>
public sealed class MetalShaderLibrary : IDisposable
{
    private readonly MetalDevice _device;
    private readonly ConcurrentDictionary<string, IntPtr> _libraries = new();
    private readonly ConcurrentDictionary<string, MetalPipelineState> _pipelineStates = new();
    private bool _disposed;
    private readonly object _compileLock = new();

    /// <summary>
    /// Gets the device associated with this shader library.
    /// </summary>
    public MetalDevice Device => _device;

    /// <summary>
    /// Gets the number of cached libraries.
    /// </summary>
    public int CachedLibraryCount => _libraries.Count;

    /// <summary>
    /// Gets the number of cached pipeline states.
    /// </summary>
    public int CachedPipelineStateCount => _pipelineStates.Count;

    /// <summary>
    /// Creates a new Metal shader library manager.
    /// </summary>
    /// <param name="device">The Metal device.</param>
    public MetalShaderLibrary(MetalDevice device)
    {
        _device = device ?? throw new ArgumentNullException(nameof(device));

        if (!device.IsAvailable)
        {
            throw new ArgumentException("Metal device is not available", nameof(device));
        }
    }

    /// <summary>
    /// Compiles a Metal library from MSL source code.
    /// </summary>
    /// <param name="libraryName">Unique name for caching.</param>
    /// <param name="source">Metal Shading Language source code.</param>
    /// <returns>The compiled library handle.</returns>
    public IntPtr CompileLibrary(string libraryName, string source)
    {
        ThrowIfDisposed();

        if (string.IsNullOrEmpty(libraryName))
        {
            throw new ArgumentException("Library name cannot be empty", nameof(libraryName));
        }

        if (string.IsNullOrEmpty(source))
        {
            throw new ArgumentException("Source cannot be empty", nameof(source));
        }

        // Check cache first
        if (_libraries.TryGetValue(libraryName, out var cached))
        {
            return cached;
        }

        lock (_compileLock)
        {
            // Double-check after acquiring lock
            if (_libraries.TryGetValue(libraryName, out cached))
            {
                return cached;
            }

            var library = _device.CreateLibrary(source, out var error);

            if (library == IntPtr.Zero)
            {
                throw new InvalidOperationException(
                    $"Failed to compile Metal library '{libraryName}': {error ?? "Unknown error"}");
            }

            _libraries[libraryName] = library;
            return library;
        }
    }

    /// <summary>
    /// Gets a function from a compiled library.
    /// </summary>
    /// <param name="library">The compiled library.</param>
    /// <param name="functionName">Name of the function.</param>
    /// <returns>The function handle.</returns>
    public IntPtr GetFunction(IntPtr library, string functionName)
    {
        ThrowIfDisposed();

        if (library == IntPtr.Zero)
        {
            throw new ArgumentException("Invalid library handle", nameof(library));
        }

        if (string.IsNullOrEmpty(functionName))
        {
            throw new ArgumentException("Function name cannot be empty", nameof(functionName));
        }

        var nameNS = CreateNSString(functionName);
        if (nameNS == IntPtr.Zero)
        {
            throw new InvalidOperationException("Failed to create NSString for function name");
        }

        try
        {
            var function = SendMessage(library, Selectors.NewFunctionWithName, nameNS);

            if (function == IntPtr.Zero)
            {
                throw new InvalidOperationException($"Function '{functionName}' not found in library");
            }

            return function;
        }
        finally
        {
            Release(nameNS);
        }
    }

    /// <summary>
    /// Creates a compute pipeline state from a function.
    /// </summary>
    /// <param name="function">The compute function.</param>
    /// <returns>Pipeline state information.</returns>
    public MetalPipelineState CreatePipelineState(IntPtr function)
    {
        ThrowIfDisposed();

        if (function == IntPtr.Zero)
        {
            throw new ArgumentException("Invalid function handle", nameof(function));
        }

        var pipelineState = _device.CreateComputePipelineState(function, out var error);

        if (pipelineState == IntPtr.Zero)
        {
            throw new InvalidOperationException(
                $"Failed to create compute pipeline state: {error ?? "Unknown error"}");
        }

        // Get pipeline properties
        var maxThreads = SendMessageULongReturn(pipelineState, Selectors.MaxTotalThreadsPerThreadgroup);
        var executionWidth = SendMessageULongReturn(pipelineState, Selectors.ThreadExecutionWidth);
        var staticMemory = SendMessageULongReturn(pipelineState, Selectors.StaticThreadgroupMemoryLength);

        return new MetalPipelineState(
            pipelineState,
            (int)maxThreads,
            (int)executionWidth,
            (int)staticMemory);
    }

    /// <summary>
    /// Gets or creates a cached pipeline state for a kernel.
    /// </summary>
    /// <param name="libraryName">Library name.</param>
    /// <param name="source">MSL source code.</param>
    /// <param name="functionName">Kernel function name.</param>
    /// <returns>The cached or newly created pipeline state.</returns>
    public MetalPipelineState GetOrCreatePipelineState(string libraryName, string source, string functionName)
    {
        ThrowIfDisposed();

        var cacheKey = $"{libraryName}::{functionName}";

        if (_pipelineStates.TryGetValue(cacheKey, out var cached))
        {
            return cached;
        }

        lock (_compileLock)
        {
            if (_pipelineStates.TryGetValue(cacheKey, out cached))
            {
                return cached;
            }

            var library = CompileLibrary(libraryName, source);
            var function = GetFunction(library, functionName);

            try
            {
                var pipelineState = CreatePipelineState(function);
                _pipelineStates[cacheKey] = pipelineState;
                return pipelineState;
            }
            finally
            {
                // Functions are not needed after pipeline creation
                Release(function);
            }
        }
    }

    /// <summary>
    /// Gets all function names from a library.
    /// </summary>
    /// <param name="library">The library to query.</param>
    /// <returns>List of function names.</returns>
    public IReadOnlyList<string> GetFunctionNames(IntPtr library)
    {
        ThrowIfDisposed();

        if (library == IntPtr.Zero)
        {
            return Array.Empty<string>();
        }

        var names = new List<string>();
        var nsArray = SendMessage(library, Selectors.FunctionNames);

        if (nsArray == IntPtr.Zero)
        {
            return names;
        }

        var count = (int)SendMessageULongReturn(nsArray, Selectors.Count);

        for (int i = 0; i < count; i++)
        {
            var nsString = SendMessage(nsArray, Selectors.ObjectAtIndex, (IntPtr)i);
            var name = GetStringFromNSString(nsString);

            if (name is not null && name.Length > 0)
            {
                names.Add(name);
            }
        }

        return names;
    }

    /// <summary>
    /// Clears all cached libraries and pipeline states.
    /// </summary>
    public void ClearCache()
    {
        lock (_compileLock)
        {
            foreach (var kvp in _pipelineStates)
            {
                kvp.Value.Dispose();
            }
            _pipelineStates.Clear();

            foreach (var kvp in _libraries)
            {
                if (kvp.Value != IntPtr.Zero)
                {
                    Release(kvp.Value);
                }
            }
            _libraries.Clear();
        }
    }

    private void ThrowIfDisposed()
    {
        if (_disposed)
        {
            throw new ObjectDisposedException(nameof(MetalShaderLibrary));
        }
    }

    /// <summary>
    /// Disposes the shader library and releases all cached resources.
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
/// Represents a Metal compute pipeline state with associated metadata.
/// </summary>
public sealed class MetalPipelineState : IDisposable
{
    private IntPtr _pipelineState;
    private bool _disposed;

    /// <summary>
    /// Gets the native pipeline state handle.
    /// </summary>
    public IntPtr Handle => _pipelineState;

    /// <summary>
    /// Gets whether the pipeline state is valid.
    /// </summary>
    public bool IsValid => _pipelineState != IntPtr.Zero && !_disposed;

    /// <summary>
    /// Gets the maximum number of threads per threadgroup.
    /// </summary>
    public int MaxTotalThreadsPerThreadgroup { get; }

    /// <summary>
    /// Gets the thread execution width (SIMD group size).
    /// </summary>
    public int ThreadExecutionWidth { get; }

    /// <summary>
    /// Gets the static threadgroup memory length.
    /// </summary>
    public int StaticThreadgroupMemoryLength { get; }

    internal MetalPipelineState(IntPtr handle, int maxThreads, int executionWidth, int staticMemory)
    {
        _pipelineState = handle;
        MaxTotalThreadsPerThreadgroup = maxThreads;
        ThreadExecutionWidth = executionWidth;
        StaticThreadgroupMemoryLength = staticMemory;
    }

    /// <summary>
    /// Calculates optimal 1D threadgroup configuration.
    /// </summary>
    /// <param name="workSize">Total number of work items.</param>
    /// <returns>Threadgroup and threads-per-threadgroup sizes.</returns>
    public (MTLSize Threadgroups, MTLSize ThreadsPerThreadgroup) Calculate1DDispatch(int workSize)
    {
        if (workSize <= 0)
        {
            return (new MTLSize(1, 1, 1), new MTLSize(1, 1, 1));
        }

        // Use execution width as minimum granularity
        var threadsPerGroup = Math.Min(MaxTotalThreadsPerThreadgroup, 256);

        // Align to execution width
        if (ThreadExecutionWidth > 0)
        {
            threadsPerGroup = ((threadsPerGroup + ThreadExecutionWidth - 1) / ThreadExecutionWidth) * ThreadExecutionWidth;
        }

        var numGroups = (workSize + threadsPerGroup - 1) / threadsPerGroup;

        return (
            new MTLSize((ulong)numGroups, 1, 1),
            new MTLSize((ulong)threadsPerGroup, 1, 1)
        );
    }

    /// <summary>
    /// Calculates optimal 2D threadgroup configuration.
    /// </summary>
    /// <param name="width">Work width.</param>
    /// <param name="height">Work height.</param>
    /// <returns>Threadgroup and threads-per-threadgroup sizes.</returns>
    public (MTLSize Threadgroups, MTLSize ThreadsPerThreadgroup) Calculate2DDispatch(int width, int height)
    {
        if (width <= 0 || height <= 0)
        {
            return (new MTLSize(1, 1, 1), new MTLSize(1, 1, 1));
        }

        // Start with 16x16 for 2D workloads
        int threadsX = 16;
        int threadsY = 16;

        // Ensure we don't exceed max threads
        while (threadsX * threadsY > MaxTotalThreadsPerThreadgroup)
        {
            if (threadsX > threadsY)
            {
                threadsX /= 2;
            }
            else
            {
                threadsY /= 2;
            }

            if (threadsX < 1 || threadsY < 1)
            {
                threadsX = threadsY = 1;
                break;
            }
        }

        var groupsX = (width + threadsX - 1) / threadsX;
        var groupsY = (height + threadsY - 1) / threadsY;

        return (
            new MTLSize((ulong)groupsX, (ulong)groupsY, 1),
            new MTLSize((ulong)threadsX, (ulong)threadsY, 1)
        );
    }

    /// <summary>
    /// Calculates optimal 3D threadgroup configuration.
    /// </summary>
    /// <param name="width">Work width.</param>
    /// <param name="height">Work height.</param>
    /// <param name="depth">Work depth.</param>
    /// <returns>Threadgroup and threads-per-threadgroup sizes.</returns>
    public (MTLSize Threadgroups, MTLSize ThreadsPerThreadgroup) Calculate3DDispatch(int width, int height, int depth)
    {
        if (width <= 0 || height <= 0 || depth <= 0)
        {
            return (new MTLSize(1, 1, 1), new MTLSize(1, 1, 1));
        }

        // Start with 8x8x4 for 3D workloads
        int threadsX = 8;
        int threadsY = 8;
        int threadsZ = 4;

        // Ensure we don't exceed max threads
        while (threadsX * threadsY * threadsZ > MaxTotalThreadsPerThreadgroup)
        {
            if (threadsZ > 1)
            {
                threadsZ /= 2;
            }
            else if (threadsY > threadsX)
            {
                threadsY /= 2;
            }
            else if (threadsX > 1)
            {
                threadsX /= 2;
            }
            else
            {
                break;
            }
        }

        var groupsX = (width + threadsX - 1) / threadsX;
        var groupsY = (height + threadsY - 1) / threadsY;
        var groupsZ = (depth + threadsZ - 1) / threadsZ;

        return (
            new MTLSize((ulong)groupsX, (ulong)groupsY, (ulong)groupsZ),
            new MTLSize((ulong)threadsX, (ulong)threadsY, (ulong)threadsZ)
        );
    }

    /// <summary>
    /// Disposes the pipeline state.
    /// </summary>
    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        _disposed = true;

        if (_pipelineState != IntPtr.Zero)
        {
            Release(_pipelineState);
            _pipelineState = IntPtr.Zero;
        }
    }

    public override string ToString()
    {
        return $"MetalPipelineState[MaxThreads={MaxTotalThreadsPerThreadgroup}, ExecutionWidth={ThreadExecutionWidth}]";
    }
}
