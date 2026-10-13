// Copyright (c) AiDotNet. All rights reserved.
// Complete Vulkan GPU compute backend for tensor operations.

using System;
using System.Buffers;
using System.Collections.Concurrent;
using System.Runtime.CompilerServices;
using System.Threading;
using AiDotNet.Tensors.Engines.Compilation.Codegen;

namespace AiDotNet.Tensors.Engines.DirectGpu.Vulkan;

/// <summary>
/// GPU compute backend using Vulkan for cross-platform acceleration.
/// </summary>
/// <remarks>
/// <para><b>Architecture:</b></para>
/// <para>
/// The Vulkan backend provides hardware-accelerated tensor operations across
/// Windows, Linux, and macOS (via MoltenVK). All operations execute entirely
/// on the GPU with no CPU fallbacks.
/// </para>
/// <para><b>Memory Management:</b></para>
/// <para>
/// Uses a staging buffer pattern for efficient CPU-GPU data transfer:
/// 1. Upload data to host-visible staging buffer
/// 2. Copy from staging to device-local storage buffer
/// 3. Execute compute shader on device-local memory
/// 4. Copy results back through staging buffer
/// </para>
/// <para><b>Pipeline Caching:</b></para>
/// <para>
/// Compute pipelines are cached by kernel type to avoid redundant compilation.
/// Shader modules are compiled once and reused across operations.
/// </para>
/// </remarks>
public sealed unsafe partial class VulkanBackend : IDirectGpuBackend, IGpuBatchExecution, IFusedAdvancedKernels, ICompressedMomentGpuOptimizerBackend, IPixelShuffleBackend, INativeGpuCodegenExecutor
{
    /// <summary>
    /// Vulkan has no cuDNN-equivalent half/bfloat16 conv path — returns
    /// null so IDirectGpuBackend's mixed-precision discovery surface
    /// (Issue #337) correctly reports the capability as absent.
    /// </summary>

    private static readonly Lazy<VulkanBackend> _instance = new(
        () => new VulkanBackend(), LazyThreadSafetyMode.ExecutionAndPublication);

    private readonly VulkanDevice _device;
    private readonly ConcurrentDictionary<(VulkanKernelType kernelType, int bindingCount, uint pushConstantSize), VulkanComputePipeline> _pipelineCache;
    private readonly ConcurrentDictionary<VulkanKernelType, VulkanShaderModule> _shaderCache;
    private readonly ConcurrentDictionary<string, VulkanComputePipeline> _glslPipelineCache;
    private readonly ConcurrentDictionary<(string Source, int Bindings, uint PushSize), VulkanComputePipeline> _glslPipelineBySource = new();
    private readonly ConcurrentDictionary<(string Source, int Bindings, uint PushSize), VulkanComputePipeline> _glslPipelineByInstance =
        new(SourceInstanceComparer.Instance);
    private VulkanGlslCompiler? _glslCompiler;
    private VulkanBufferTransfer? _transfer;
    // The device's dispatch lock: recording a dispatch into the device's open command buffer, and ending and
    // submitting that buffer, exclude each other through it (#1192).
    private readonly object _computeLock = VulkanDevice.Instance.DispatchLock;
    private bool _initialized;
    private volatile bool _disposed;

    // Batch execution state — records multiple dispatches into a single command buffer.
    // Thread-affine: _batchOwnerThreadId tracks which thread owns the batch. Non-owning
    // threads fall through to the non-batch path to avoid recording into another thread's
    // command buffer.
    private bool _batchMode;
    private int _batchOwnerThreadId;
    private ThreadCommandResources _batchThreadRes;
    private int _batchDispatchCount;
    private volatile bool _inSecondaryStream;
    private ThreadCommandResources _secondaryThreadRes;
    private int _secondaryStreamOwnerThreadId;

    /// <summary>
    /// Gets the singleton backend instance.
    /// </summary>
    public static VulkanBackend Instance => _instance.Value;

    /// <summary>
    /// Gets whether the backend is available and initialized.
    /// </summary>
    public bool IsAvailable => _initialized && !_disposed;

    /// <summary>Why the last <see cref="Initialize"/> failed (stage and VkResult), or empty; see <see cref="VulkanDevice.InitializationFailure"/>.</summary>
    public string InitializationFailure =>
        _disposed ? "The Vulkan backend was disposed."
        : _platformFailure.Length > 0 ? _platformFailure
        : _device.InitializationFailure;

    private string _platformFailure = string.Empty;

    /// <summary>
    /// Gets whether runtime GLSL→SPIR-V compilation is available on this
    /// host. Returns <c>false</c> when libshaderc cannot be loaded — which
    /// leaves every GLSL-compiled kernel (octonion linear, Poincaré
    /// distance, Möbius add, …) unable to execute. Tests targeting those
    /// kernels should skip when this is <c>false</c> instead of silently
    /// observing zero-filled output buffers (GlslQuadOp / GlslBinaryOp
    /// take no-op fallbacks when the pipeline can't be created).
    /// </summary>
    public bool IsGlslCompilerAvailable => _initialized && _glslCompiler is not null && _glslCompiler.IsAvailable;

    /// <summary>
    /// Gets the backend name.
    /// </summary>
    public string BackendName => "Vulkan";
    public TensorDevice DeviceType => TensorDevice.Vulkan;

    /// <summary>
    /// Gets the device name.
    /// </summary>
    public string DeviceName => _device.DeviceName;

    /// <summary>
    /// Gets the GPU vendor (AMD, NVIDIA, Intel).
    /// Required by <see cref="IDirectGpuBackend"/>.
    /// </summary>
    public string DeviceVendor => _device.VendorName;

    /// <summary>
    /// Gets the GPU vendor name (convenience alias for <see cref="DeviceVendor"/>).
    /// </summary>
    public string VendorName => _device.VendorName;

    /// <summary>
    /// Gets the number of compute units. Returns the maximum workgroup invocations,
    /// which is the closest Vulkan equivalent to OpenCL's CL_DEVICE_MAX_COMPUTE_UNITS.
    /// </summary>
    public int ComputeUnits => (int)_device.Limits.maxComputeWorkGroupInvocations;

    /// <summary>
    /// Gets the total device-local (global) memory in bytes.
    /// Sums all device-local memory heaps instead of using MaxStorageBufferRange.
    /// </summary>
    public long GlobalMemoryBytes => _device.TotalDeviceLocalMemoryBytes;

    /// <summary>
    /// Gets the local (shared) memory per workgroup in bytes.
    /// </summary>
    public long LocalMemoryBytes => _device.MaxSharedMemorySize;

    /// <summary>
    /// Issue #285: Vulkan's <c>VkPhysicalDeviceLimits.maxStorageBufferRange</c>
    /// is the per-allocation (per-binding) cap. The spec guarantees at least
    /// 128 MiB; modern desktop GPUs typically expose 2 GiB or more.
    /// </summary>
    public long MaxBufferAllocBytes => _device.MaxStorageBufferRange;

    public double TheoreticalGflops { get; }

    /// <summary>
    /// Gets the maximum workgroup size.
    /// </summary>
    public uint MaxWorkgroupSize => _device.MaxWorkgroupSize;

    /// <summary>
    /// Gets the maximum shared memory size.
    /// </summary>
    public uint MaxSharedMemorySize => _device.MaxSharedMemorySize;

    private VulkanBackend()
    {
        _device = VulkanDevice.Instance;
        _pipelineCache = new ConcurrentDictionary<(VulkanKernelType, int, uint), VulkanComputePipeline>();
        _shaderCache = new ConcurrentDictionary<VulkanKernelType, VulkanShaderModule>();
        _glslPipelineCache = new ConcurrentDictionary<string, VulkanComputePipeline>(StringComparer.Ordinal);
        try { _glslCompiler = new VulkanGlslCompiler(); }
        catch { _glslCompiler = null; }
    }

    /// <summary>
    /// Initializes the Vulkan backend.
    /// </summary>
    /// <returns>True if initialization succeeded.</returns>
    public bool Initialize()
    {
        if (_initialized)
        {
            return true;
        }

        if (_disposed)
        {
            return false;
        }

        if (!VulkanNativeBindings.IsPlatformSupported)
        {
            _platformFailure = "The Vulkan loader (vulkan-1) could not be loaded.";
            return false;
        }
        _platformFailure = string.Empty;

        if (!_device.Initialize())
        {
            return false;
        }

        _transfer = new VulkanBufferTransfer();

        _initialized = true;
        return true;
    }

    /// <summary>
    /// Gets or creates a compute pipeline for the specified kernel.
    /// </summary>
    /// <summary>Validates and journals one dispatch (#996) so a device fault can be attributed to the kernel.</summary>
    private static void JournalDispatch(VulkanComputePipeline pipeline, uint workgroups, int workgroupSize)
    {
        long global = (long)workgroups * workgroupSize;
        GpuKernelDiagnostics.ValidateLaunch(pipeline.Name, handleIsValid: pipeline.Handle != IntPtr.Zero, stagedArgCount: 0,
            declaredArgCount: -1, globalSize: global, localSize: workgroupSize);
        GpuKernelDiagnostics.RecordLaunch(pipeline.Name, global, workgroupSize, 0);
    }

    private static readonly Lazy<Dictionary<string, string>> GlslShaderNames = new Lazy<Dictionary<string, string>>(() =>
    {
        var names = new Dictionary<string, string>(StringComparer.Ordinal);
        const System.Reflection.BindingFlags Statics = System.Reflection.BindingFlags.Public
            | System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Static;
        foreach (var type in new[] { typeof(VulkanGlslKernels), typeof(VulkanBatchExecutionKernels) })
        {
            foreach (var property in type.GetProperties(Statics))
                if (property.PropertyType == typeof(string) && property.GetIndexParameters().Length == 0
                    && property.GetValue(null) is string source && !names.ContainsKey(source))
                    names[source] = type.Name + "." + property.Name;
            foreach (var field in type.GetFields(Statics))
                if (field.FieldType == typeof(string) && field.GetValue(null) is string source && !names.ContainsKey(source))
                    names[source] = type.Name + "." + field.Name;
        }
        return names;
    });

    /// <summary>The shader's declaring member when it is one of the library's GLSL sources, else a source-hash tag.</summary>
    private static string GlslShaderName(string source, string sourceHash)
        => GlslShaderNames.Value.TryGetValue(source, out var name) ? name : "glsl:" + sourceHash.Substring(0, 12);

    private VulkanComputePipeline? GetOrCreatePipeline(VulkanKernelType kernelType, int bindingCount, uint pushConstantSize = 0)
    {
        var cacheKey = (kernelType, bindingCount, pushConstantSize);
        if (_pipelineCache.TryGetValue(cacheKey, out var cached))
        {
            return cached;
        }

        var shader = GetOrCreateShader(kernelType);
        if (shader == null)
        {
            return null;
        }

        var pipeline = VulkanComputePipeline.Create(shader, bindingCount, pushConstantSize);
        if (pipeline is not null) pipeline.Name = kernelType.ToString();
        if (pipeline != null)
        {
            _pipelineCache.TryAdd(cacheKey, pipeline);
        }

        return pipeline;
    }

    private void RememberPipeline((string Source, int Bindings, uint PushSize) key, VulkanComputePipeline pipeline)
    {
        _glslPipelineBySource.TryAdd(key, pipeline);
        // Only an interned instance (a literal) is remembered by reference: a source built per call would add an
        // entry, and keep its string alive, on every call.
        if (ReferenceEquals(string.IsInterned(key.Source), key.Source))
            _glslPipelineByInstance.TryAdd(key, pipeline);
    }

    /// <summary>Compares pipeline keys by the source string's identity rather than its text.</summary>
    private sealed class SourceInstanceComparer : IEqualityComparer<(string Source, int Bindings, uint PushSize)>
    {
        internal static readonly SourceInstanceComparer Instance = new();

        public bool Equals((string Source, int Bindings, uint PushSize) x, (string Source, int Bindings, uint PushSize) y) =>
            ReferenceEquals(x.Source, y.Source) && x.Bindings == y.Bindings && x.PushSize == y.PushSize;

        public int GetHashCode((string Source, int Bindings, uint PushSize) key) =>
            unchecked(System.Runtime.CompilerServices.RuntimeHelpers.GetHashCode(key.Source) * 31 + key.Bindings * 17 + (int)key.PushSize);
    }

    /// <summary>
    /// Gets or creates a compute pipeline from a GLSL source string using runtime compilation.
    /// </summary>
    private VulkanComputePipeline? GetOrCreateGlslPipeline(string glslSource, int bindingCount, uint pushConstantSize = 0)
    {
        // Every dispatch comes through here. Hashing the source with SHA-256, formatting the digest and building the
        // key string below cost 2-4 microseconds and 1.5-4 KB of garbage per op. Most kernel sources are string
        // literals, so the same instance comes back every time and is found by reference; a source built at run time
        // is found by its text.
        var sourceKey = (glslSource, bindingCount, pushConstantSize);
        if (_glslPipelineByInstance.TryGetValue(sourceKey, out var known) ||
            _glslPipelineBySource.TryGetValue(sourceKey, out known))
            return known;
        // Use SHA256 instead of GetHashCode to prevent cache key collisions
        string sourceHash;
        using (var sha = System.Security.Cryptography.SHA256.Create())
        {
            var hashBytes = sha.ComputeHash(System.Text.Encoding.UTF8.GetBytes(glslSource));
            sourceHash = BitConverter.ToString(hashBytes).Replace("-", string.Empty);
        }
        string cacheKey = $"{sourceHash}_{bindingCount}_{pushConstantSize}";
        if (_glslPipelineCache.TryGetValue(cacheKey, out var cached))
        {
            RememberPipeline(sourceKey, cached);
            return cached;
        }

        if (_glslCompiler is null || !_glslCompiler.IsAvailable)
            return null;

        var shader = _glslCompiler.CompileToShaderModule(glslSource);
        if (shader is null)
            return null;

        try
        {
            var pipeline = VulkanComputePipeline.Create(shader, bindingCount, pushConstantSize);
        if (pipeline is not null) pipeline.Name = GlslShaderName(glslSource, sourceHash);
            if (pipeline is not null)
            {
                // Another thread may have compiled the same source meanwhile: keep the cached pipeline for every
                // caller and release this one.
                var kept = _glslPipelineCache.GetOrAdd(cacheKey, pipeline);
                if (!ReferenceEquals(kept, pipeline)) { pipeline.Dispose(); pipeline = kept; }
                RememberPipeline(sourceKey, pipeline);
            }
            return pipeline;
        }
        finally
        {
            // Vulkan spec allows destroying shader modules after pipeline creation
            shader.Dispose();
        }
    }

    // The size-only shorthands below push exactly one word. A kernel whose push-constant block is larger must pass
    // every word explicitly: vkCmdPushConstants would otherwise read past the one-element array.
    private static void RequireSizeOnlyPush(uint pushConstantSize)
    {
        if (pushConstantSize != sizeof(uint))
            throw new ArgumentException(
                $"This kernel declares {pushConstantSize} bytes of push constants; pass each value explicitly instead of the size-only shorthand.",
                nameof(pushConstantSize));
    }

    /// <summary>
    /// Executes a GLSL-compiled compute pipeline with 2 buffers + push constants.
    /// </summary>
    private void GlslUnaryOp(string glslSource, IGpuBuffer A, IGpuBuffer B, int size, uint pushConstantSize = sizeof(uint))
    {
        RequireSizeOnlyPush(pushConstantSize);
        // Single-parameter shorthand — pushes {size} as the only push constant
        GlslUnaryOp(glslSource, A, B, size, new uint[] { (uint)size }, pushConstantSize);
    }

    /// <summary>
    /// Executes a GLSL compute pipeline with exactly one storage buffer
    /// (in-place op). Used by kernels that read AND write the same memory —
    /// allocating a two-binding pipeline with the same buffer aliased would
    /// trip the Vulkan "aliased readonly/writable buffer" validation.
    /// </summary>
    private void GlslInPlaceOp(string glslSource, IGpuBuffer buffer, int dispatchSize, uint[] pushConstants, uint pushConstantSize)
    {
        EnsureInitialized();
        if (dispatchSize <= 0) return;
        var pipeline = GetOrCreateGlslPipeline(glslSource, 1, pushConstantSize);
        if (pipeline is null)
        {
            // No CPU fallback for in-place ops: the caller should handle this
            // by catching and routing to a managed path.
            throw new System.InvalidOperationException("Vulkan in-place pipeline creation failed.");
        }
        var vb = AsVulkan(buffer);
        var threadRes = _device.AcquireThreadResources();
        lock (_computeLock)
        {
            pipeline.UpdateDescriptorSet(vb.Storage);
            RecordAndExecuteWithPushData(pipeline, dispatchSize, pushConstants, pushConstantSize, threadRes);
        }
    }

    /// <summary>
    /// Executes a GLSL compute pipeline with 2 buffers and explicit push constant values.
    /// </summary>
    private void GlslUnaryOp(string glslSource, IGpuBuffer A, IGpuBuffer B, int dispatchSize, uint[] pushConstants, uint pushConstantSize)
    {
        EnsureInitialized();
        if (dispatchSize <= 0) return;
        var pipeline = GetOrCreateGlslPipeline(glslSource, 2, pushConstantSize);
        if (pipeline is null)
            throw new InvalidOperationException(
                "Vulkan GLSL pipeline unavailable. Install libshaderc or use another GPU backend.");
        var vbA = AsVulkan(A);
        var vbB = AsVulkan(B);
        var threadRes = _device.AcquireThreadResources();
        lock (_computeLock)
        {
            pipeline.UpdateDescriptorSet(vbA.Storage, vbB.Storage);
            RecordAndExecuteWithPushData(pipeline, dispatchSize, pushConstants, pushConstantSize, threadRes);
        }
    }

    /// <summary>
    /// Executes a GLSL-compiled compute pipeline with 3 buffers + push constants.
    /// </summary>
    private void GlslBinaryOp(string glslSource, IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int size, uint pushConstantSize = sizeof(uint))
    {
        RequireSizeOnlyPush(pushConstantSize);
        GlslBinaryOp(glslSource, A, B, C, size, new uint[] { (uint)size }, pushConstantSize);
    }

    private void GlslBinaryOp(string glslSource, IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int dispatchSize, uint[] pushConstants, uint pushConstantSize)
    {
        EnsureInitialized();
        if (dispatchSize <= 0) return;
        var pipeline = GetOrCreateGlslPipeline(glslSource, 3, pushConstantSize);
        if (pipeline is null)
            throw new InvalidOperationException(
                "Vulkan GLSL pipeline unavailable. Install libshaderc or use another GPU backend.");
        var vbA = AsVulkan(A);
        var vbB = AsVulkan(B);
        var vbC = AsVulkan(C);
        var threadRes = _device.AcquireThreadResources();
        lock (_computeLock)
        {
            pipeline.UpdateDescriptorSet(vbA.Storage, vbB.Storage, vbC.Storage);
            RecordAndExecuteWithPushData(pipeline, dispatchSize, pushConstants, pushConstantSize, threadRes);
        }
    }

    /// <summary>
    /// Executes a GLSL-compiled compute pipeline with N buffers + push constants (#1464 recurrence ops).
    /// Buffers bind to set=0, binding=0..N-1 in order; pushConstants are the kernel's int params as uints.
    /// </summary>
    private void GlslNaryOp(
        string glslSource,
        IGpuBuffer[] buffers,
        int dispatchSize,
        uint[] pushConstants,
        int workgroupSize = VulkanKernels.WorkgroupSize)
    {
        EnsureInitialized();
        if (dispatchSize <= 0) return;
        uint pushConstantSize = (uint)(pushConstants.Length * sizeof(uint));
        var pipeline = GetOrCreateGlslPipeline(glslSource, buffers.Length, pushConstantSize);
        if (pipeline is null)
            throw new InvalidOperationException("Failed to create GLSL recurrence pipeline.");
        var storages = new VulkanBuffer[buffers.Length];
        for (int i = 0; i < buffers.Length; i++) storages[i] = AsVulkan(buffers[i]).Storage;
        var threadRes = _device.AcquireThreadResources();
        lock (_computeLock)
        {
            // Dispose releases the backend-owned placeholder binding under this lock.
            if (_disposed) throw new ObjectDisposedException(nameof(VulkanBackend));
            pipeline.UpdateDescriptorSet(storages);
            RecordAndExecuteWithPushData(
                pipeline, dispatchSize, pushConstants, pushConstantSize, threadRes, workgroupSize);
        }
    }

    /// <summary>
    /// Executes a GLSL-compiled compute pipeline with 1 buffer (output only) + push constants.
    /// Used for generate ops like Eye, Linspace, TriangularMask.
    /// </summary>
    private void GlslGenerateOp(string glslSource, IGpuBuffer O, int size, uint pushConstantSize = sizeof(uint))
    {
        RequireSizeOnlyPush(pushConstantSize);
        GlslGenerateOp(glslSource, O, size, new uint[] { (uint)size }, pushConstantSize);
    }

    private void GlslGenerateOp(string glslSource, IGpuBuffer O, int dispatchSize, uint[] pushConstants, uint pushConstantSize)
    {
        EnsureInitialized();
        if (dispatchSize <= 0) return;
        var pipeline = GetOrCreateGlslPipeline(glslSource, 1, pushConstantSize);
        if (pipeline is null) throw new GpuCapabilityUnavailableException("Vulkan GLSL pipeline unavailable — install libshaderc for runtime compilation.");
        var vbO = AsVulkan(O);
        var threadRes = _device.AcquireThreadResources();
        lock (_computeLock)
        {
            pipeline.UpdateDescriptorSet(vbO.Storage);
            RecordAndExecuteWithPushData(pipeline, dispatchSize, pushConstants, pushConstantSize, threadRes);
        }
    }

    /// <summary>
    /// Executes a GLSL-compiled compute pipeline with 4 buffers + push constants.
    /// Used for backward ops needing grad_output, input, aux, grad_input.
    /// </summary>
    private void GlslQuadOp(string glslSource, IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, IGpuBuffer D, int size, uint pushConstantSize = sizeof(uint))
    {
        RequireSizeOnlyPush(pushConstantSize);
        GlslQuadOp(glslSource, A, B, C, D, size, new uint[] { (uint)size }, pushConstantSize);
    }

    private void GlslQuadOp(string glslSource, IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, IGpuBuffer D, int dispatchSize, uint[] pushConstants, uint pushConstantSize)
    {
        EnsureInitialized();
        if (dispatchSize <= 0) return;
        var pipeline = GetOrCreateGlslPipeline(glslSource, 4, pushConstantSize);
        if (pipeline is null)
            throw new GpuCapabilityUnavailableException("Vulkan GLSL pipeline unavailable - install libshaderc for runtime compilation.");
        var vbA = AsVulkan(A); var vbB = AsVulkan(B); var vbC = AsVulkan(C); var vbD = AsVulkan(D);
        var threadRes = _device.AcquireThreadResources();
        lock (_computeLock)
        {
            pipeline.UpdateDescriptorSet(vbA.Storage, vbB.Storage, vbC.Storage, vbD.Storage);
            RecordAndExecuteWithPushData(pipeline, dispatchSize, pushConstants, pushConstantSize, threadRes);
        }
    }

    /// <summary>
    /// Executes a GLSL-compiled compute pipeline with 5 buffers + push constants.
    /// Used for complex backward ops needing grad_output, input, mean, variance, grad_input.
    /// </summary>
    private void GlslQuintOp(string glslSource, IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, IGpuBuffer D, IGpuBuffer E, int size, uint pushConstantSize = sizeof(uint))
    {
        RequireSizeOnlyPush(pushConstantSize);
        GlslQuintOp(glslSource, A, B, C, D, E, size, new uint[] { (uint)size }, pushConstantSize);
    }

    private void GlslQuintOp(string glslSource, IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, IGpuBuffer D, IGpuBuffer E, int dispatchSize, uint[] pushConstants, uint pushConstantSize)
    {
        EnsureInitialized();
        if (dispatchSize <= 0) return;
        var pipeline = GetOrCreateGlslPipeline(glslSource, 5, pushConstantSize);
        if (pipeline is null) throw new GpuCapabilityUnavailableException("Vulkan GLSL pipeline unavailable — install libshaderc for runtime compilation.");
        var vbA = AsVulkan(A); var vbB = AsVulkan(B); var vbC = AsVulkan(C); var vbD = AsVulkan(D); var vbE = AsVulkan(E);
        var threadRes = _device.AcquireThreadResources();
        lock (_computeLock)
        {
            pipeline.UpdateDescriptorSet(vbA.Storage, vbB.Storage, vbC.Storage, vbD.Storage, vbE.Storage);
            RecordAndExecuteWithPushData(pipeline, dispatchSize, pushConstants, pushConstantSize, threadRes);
        }
    }

    /// <summary>
    /// Executes a GLSL-compiled compute pipeline with 8 buffers + push constants.
    /// Used for HRR bind-accumulate (keyR, keyI, valR, valI, keyIds, valIds,
    /// memR_inout, memI_inout). All modern desktop / mobile Vulkan drivers
    /// report maxPerStageDescriptorStorageBuffers ≥ 16; 8 is well within that.
    /// </summary>
    private void GlslOctOp(string glslSource, IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, IGpuBuffer D, IGpuBuffer E, IGpuBuffer F, IGpuBuffer G, IGpuBuffer H, int dispatchSize, uint[] pushConstants, uint pushConstantSize)
    {
        EnsureInitialized();
        if (dispatchSize <= 0) return;
        var pipeline = GetOrCreateGlslPipeline(glslSource, 8, pushConstantSize);
        if (pipeline is null) throw new GpuCapabilityUnavailableException("Vulkan GLSL pipeline unavailable — install libshaderc for runtime compilation.");
        var vbA = AsVulkan(A); var vbB = AsVulkan(B); var vbC = AsVulkan(C); var vbD = AsVulkan(D);
        var vbE = AsVulkan(E); var vbF = AsVulkan(F); var vbG = AsVulkan(G); var vbH = AsVulkan(H);
        var threadRes = _device.AcquireThreadResources();
        lock (_computeLock)
        {
            pipeline.UpdateDescriptorSet(vbA.Storage, vbB.Storage, vbC.Storage, vbD.Storage, vbE.Storage, vbF.Storage, vbG.Storage, vbH.Storage);
            RecordAndExecuteWithPushData(pipeline, dispatchSize, pushConstants, pushConstantSize, threadRes);
        }
    }

    /// <summary>
    /// Gets or creates a shader module for the specified kernel.
    /// </summary>
    private VulkanShaderModule? GetOrCreateShader(VulkanKernelType kernelType)
    {
        if (_shaderCache.TryGetValue(kernelType, out var cached))
        {
            return cached;
        }

        var spirv = VulkanKernels.GetKernel(kernelType);
        if (spirv == null)
        {
            return null;
        }

        var shader = VulkanShaderModule.Create(spirv);
        if (shader != null)
        {
            _shaderCache.TryAdd(kernelType, shader);
        }

        return shader;
    }

    /// <summary>
    /// Executes a binary operation: C = op(A, B)
    /// </summary>
    /// <param name="a">Input array A.</param>
    /// <param name="b">Input array B.</param>
    /// <param name="result">Output array (must be pre-allocated).</param>
    /// <param name="kernelType">The operation to perform.</param>
    public void ExecuteBinaryOp(ReadOnlySpan<float> a, ReadOnlySpan<float> b, Span<float> result, VulkanKernelType kernelType)
    {
        if (!_initialized || _disposed)
        {
            throw new GpuCapabilityUnavailableException("Vulkan backend not initialized.");
        }

        if (a.Length != b.Length || a.Length != result.Length)
        {
            throw new ArgumentException(
                $"Buffer size mismatch: a.Length={a.Length}, b.Length={b.Length}, result.Length={result.Length}. All must be equal.");
        }

        int size = a.Length;
        if (size == 0)
        {
            return;
        }

        // Create buffers
        using var bufferA = VulkanBuffer.CreateStorageBuffer(size);
        using var bufferB = VulkanBuffer.CreateStorageBuffer(size);
        using var bufferC = VulkanBuffer.CreateStorageBuffer(size);
        using var stagingA = VulkanBuffer.CreateStagingBuffer(size);
        using var stagingB = VulkanBuffer.CreateStagingBuffer(size);
        using var stagingC = VulkanBuffer.CreateStagingBuffer(size);

        if (bufferA == null || bufferB == null || bufferC == null ||
            stagingA == null || stagingB == null || stagingC == null)
        {
            throw new InvalidOperationException("Failed to allocate GPU buffers.");
        }

        // Upload data
        stagingA.WriteData(a);
        stagingB.WriteData(b);
        if (_transfer is null)
            throw new GpuCapabilityUnavailableException("Vulkan buffer transfer not initialized.");
        _transfer.CopyToDevice(stagingA, bufferA);
        _transfer.CopyToDevice(stagingB, bufferB);

        // Get or create pipeline
        var pipeline = GetOrCreatePipeline(kernelType, 3, sizeof(uint));
        if (pipeline == null)
        {
            throw new InvalidOperationException($"Failed to create pipeline for {kernelType}.");
        }

        // Acquire per-thread resources and execute
        var threadRes = _device.AcquireThreadResources();

        // Lock around descriptor set update + dispatch to prevent concurrent mutation
        lock (_computeLock)
        {
            pipeline.UpdateDescriptorSet(bufferA, bufferB, bufferC);
            RecordAndExecuteComputeUnlocked(pipeline, size, threadRes);
        }

        // Download results
        _transfer.CopyFromDevice(bufferC, stagingC);
        stagingC.ReadData(result);
    }

    /// <summary>
    /// Executes a unary operation: B = op(A)
    /// </summary>
    /// <param name="input">Input array.</param>
    /// <param name="result">Output array (must be pre-allocated).</param>
    /// <param name="kernelType">The operation to perform.</param>
    public void ExecuteUnaryOp(ReadOnlySpan<float> input, Span<float> result, VulkanKernelType kernelType)
    {
        if (!_initialized || _disposed)
        {
            throw new GpuCapabilityUnavailableException("Vulkan backend not initialized.");
        }

        if (input.Length != result.Length)
        {
            throw new ArgumentException(
                $"Buffer size mismatch: input.Length={input.Length}, result.Length={result.Length}. Both must be equal.");
        }

        int size = input.Length;
        if (size == 0)
        {
            return;
        }

        // Create buffers
        using var bufferA = VulkanBuffer.CreateStorageBuffer(size);
        using var bufferB = VulkanBuffer.CreateStorageBuffer(size);
        using var stagingA = VulkanBuffer.CreateStagingBuffer(size);
        using var stagingB = VulkanBuffer.CreateStagingBuffer(size);

        if (bufferA == null || bufferB == null || stagingA == null || stagingB == null)
        {
            throw new InvalidOperationException("Failed to allocate GPU buffers.");
        }

        // Upload data
        stagingA.WriteData(input);
        if (_transfer is null)
            throw new GpuCapabilityUnavailableException("Vulkan buffer transfer not initialized.");
        _transfer.CopyToDevice(stagingA, bufferA);

        // Get or create pipeline
        var pipeline = GetOrCreatePipeline(kernelType, 2, sizeof(uint));
        if (pipeline == null)
        {
            throw new InvalidOperationException($"Failed to create pipeline for {kernelType}.");
        }

        // Acquire per-thread resources and execute
        var threadRes = _device.AcquireThreadResources();

        // Lock around descriptor set update + dispatch to prevent concurrent mutation
        lock (_computeLock)
        {
            pipeline.UpdateDescriptorSet(bufferA, bufferB);
            RecordAndExecuteComputeUnlocked(pipeline, size, threadRes);
        }

        // Download results
        _transfer.CopyFromDevice(bufferB, stagingB);
        stagingB.ReadData(result);
    }

    /// <summary>
    /// Executes scalar multiplication: B = A * scalar
    /// </summary>
    /// <param name="input">Input array.</param>
    /// <param name="scalar">Scalar value.</param>
    /// <param name="result">Output array (must be pre-allocated).</param>
    public void ScalarMultiply(ReadOnlySpan<float> input, float scalar, Span<float> result)
    {
        if (!_initialized || _disposed)
        {
            throw new GpuCapabilityUnavailableException("Vulkan backend not initialized.");
        }

        if (input.Length != result.Length)
        {
            throw new ArgumentException(
                $"Buffer size mismatch: input.Length={input.Length}, result.Length={result.Length}. Both must be equal.");
        }

        int size = input.Length;
        if (size == 0)
        {
            return;
        }

        // Create buffers
        using var bufferA = VulkanBuffer.CreateStorageBuffer(size);
        using var bufferB = VulkanBuffer.CreateStorageBuffer(size);
        using var stagingA = VulkanBuffer.CreateStagingBuffer(size);
        using var stagingB = VulkanBuffer.CreateStagingBuffer(size);

        if (bufferA == null || bufferB == null || stagingA == null || stagingB == null)
        {
            throw new InvalidOperationException("Failed to allocate GPU buffers.");
        }

        // Upload data
        stagingA.WriteData(input);
        if (_transfer is null)
            throw new GpuCapabilityUnavailableException("Vulkan buffer transfer not initialized.");
        _transfer.CopyToDevice(stagingA, bufferA);

        // Get or create pipeline (push constants: uint size, float scalar)
        var pipeline = GetOrCreatePipeline(VulkanKernelType.ScalarMultiply, 2, sizeof(uint) + sizeof(float));
        if (pipeline == null)
        {
            throw new InvalidOperationException("Failed to create scalar multiply pipeline.");
        }

        // Acquire per-thread resources and execute
        var threadRes = _device.AcquireThreadResources();

        // Lock around descriptor set update + dispatch to prevent concurrent mutation
        lock (_computeLock)
        {
            pipeline.UpdateDescriptorSet(bufferA, bufferB);
            RecordAndExecuteComputeWithScalarUnlocked(pipeline, size, scalar, threadRes);
        }

        // Download results
        _transfer.CopyFromDevice(bufferB, stagingB);
        stagingB.ReadData(result);
    }

    /// <summary>
    /// Element-wise vector addition: C = A + B
    /// </summary>
    public void Add(ReadOnlySpan<float> a, ReadOnlySpan<float> b, Span<float> result)
    {
        ExecuteBinaryOp(a, b, result, VulkanKernelType.VectorAdd);
    }

    /// <summary>
    /// Element-wise vector subtraction: C = A - B
    /// </summary>
    public void Subtract(ReadOnlySpan<float> a, ReadOnlySpan<float> b, Span<float> result)
    {
        ExecuteBinaryOp(a, b, result, VulkanKernelType.VectorSubtract);
    }

    /// <summary>
    /// Element-wise vector multiplication: C = A * B
    /// </summary>
    public void Multiply(ReadOnlySpan<float> a, ReadOnlySpan<float> b, Span<float> result)
    {
        ExecuteBinaryOp(a, b, result, VulkanKernelType.VectorMultiply);
    }

    /// <summary>
    /// Element-wise vector division: C = A / B
    /// </summary>
    public void Divide(ReadOnlySpan<float> a, ReadOnlySpan<float> b, Span<float> result)
    {
        ExecuteBinaryOp(a, b, result, VulkanKernelType.VectorDivide);
    }

    /// <summary>
    /// ReLU activation: B = max(A, 0)
    /// </summary>
    public void ReLU(ReadOnlySpan<float> input, Span<float> result)
    {
        ExecuteUnaryOp(input, result, VulkanKernelType.ReLU);
    }

    /// <summary>
    /// Sigmoid activation: B = 1 / (1 + exp(-A))
    /// </summary>
    public void Sigmoid(ReadOnlySpan<float> input, Span<float> result)
    {
        ExecuteUnaryOp(input, result, VulkanKernelType.Sigmoid);
    }

    /// <summary>
    /// Tanh activation: B = tanh(A)
    /// </summary>
    public void Tanh(ReadOnlySpan<float> input, Span<float> result)
    {
        // Clamp extreme values to avoid NaN from GPU tanh shader overflow.
        // tanh saturates to +/-1 for |x| > ~10, so clamping preserves correctness.
        var clamped = ArrayPool<float>.Shared.Rent(input.Length);
        try
        {
            for (int i = 0; i < input.Length; i++)
            {
                clamped[i] = Math.Max(-20.0f, Math.Min(20.0f, input[i]));
            }
            ExecuteUnaryOp(clamped.AsSpan(0, input.Length), result, VulkanKernelType.Tanh);
        }
        finally
        {
            ArrayPool<float>.Shared.Return(clamped);
        }
    }

    /// <summary>
    /// Records and executes a compute dispatch with explicit push constant values.
    /// Pushes the exact values needed by the GLSL shader's push_constant block.
    /// Caller MUST hold _computeLock.
    /// </summary>
    private void RecordAndExecuteWithPushData(
        VulkanComputePipeline pipeline,
        int dispatchSize,
        uint[] pushConstants,
        uint pushConstantSize,
        ThreadCommandResources threadRes,
        int workgroupSize = VulkanKernels.WorkgroupSize)
    {
        if (workgroupSize <= 0) throw new ArgumentOutOfRangeException(nameof(workgroupSize));
        if (pushConstants is null) throw new ArgumentNullException(nameof(pushConstants));
        if ((long)pushConstants.Length * sizeof(uint) < pushConstantSize)
            throw new ArgumentException(
                $"The kernel declares {pushConstantSize} bytes of push constants but only {pushConstants.Length * sizeof(uint)} were supplied.",
                nameof(pushConstants));
        // Batch mode: record into batch command buffer without immediate submit
        if (_batchMode && Environment.CurrentManagedThreadId == _batchOwnerThreadId)
        {
            var batchCmd = _batchThreadRes.CommandBuffer;
            if (_batchDispatchCount > 0)
                RecordComputeMemoryBarrier(batchCmd);
            VulkanNativeBindings.vkCmdBindPipeline(batchCmd, VulkanNativeBindings.VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.Handle);
            var batchDs = RetainBatchDescriptorSet(pipeline);
            VulkanNativeBindings.vkCmdBindDescriptorSets(batchCmd, VulkanNativeBindings.VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.Layout, 0, 1, &batchDs, 0, null);
            fixed (uint* ptr = pushConstants)
            {
                VulkanNativeBindings.vkCmdPushConstants(batchCmd, pipeline.Layout, VulkanNativeBindings.VK_SHADER_STAGE_COMPUTE_BIT, 0, pushConstantSize, ptr);
            }
            uint batchWgCount = CalculateWorkgroupCount(dispatchSize, workgroupSize);
            JournalDispatch(pipeline, batchWgCount, workgroupSize);
            VulkanNativeBindings.vkCmdDispatch(batchCmd, batchWgCount, 1, 1);
            _batchDispatchCount++;
            return;
        }

        // Immediate submit path
        var cmdBuffer = DispatchCommandBuffer(threadRes, out bool deferred);
        if (!deferred) VulkanNativeBindings.vkResetCommandBuffer(cmdBuffer, 0);
        var beginInfo = new VkCommandBufferBeginInfo
        {
            sType = VulkanNativeBindings.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            pNext = null,
            flags = VkCommandBufferUsageFlags.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
            pInheritanceInfo = null
        };
        if (!deferred) VulkanNativeBindings.vkBeginCommandBuffer(cmdBuffer, &beginInfo);
        else RecordComputeMemoryBarrier(cmdBuffer);
        VulkanNativeBindings.vkCmdBindPipeline(cmdBuffer, VulkanNativeBindings.VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.Handle);
        var ds = deferred ? pipeline.RetainDescriptorSet() : pipeline.DescriptorSet;
        VulkanNativeBindings.vkCmdBindDescriptorSets(cmdBuffer, VulkanNativeBindings.VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.Layout, 0, 1, &ds, 0, null);

        // Push the caller-provided constant values
        fixed (uint* ptr = pushConstants)
        {
            VulkanNativeBindings.vkCmdPushConstants(cmdBuffer, pipeline.Layout, VulkanNativeBindings.VK_SHADER_STAGE_COMPUTE_BIT, 0, pushConstantSize, ptr);
        }

        uint workgroupCount = CalculateWorkgroupCount(dispatchSize, workgroupSize);
        JournalDispatch(pipeline, workgroupCount, workgroupSize);
        VulkanNativeBindings.vkCmdDispatch(cmdBuffer, workgroupCount, 1, 1);
        if (!deferred) VulkanNativeBindings.vkEndCommandBuffer(cmdBuffer);
        SubmitDispatch(cmdBuffer, threadRes, deferred, pipeline, ds);
    }

    private static uint CalculateWorkgroupCount(int elementCount, int workgroupSize)
    {
        if (elementCount <= 0) throw new ArgumentOutOfRangeException(nameof(elementCount));
        if (workgroupSize <= 0) throw new ArgumentOutOfRangeException(nameof(workgroupSize));
        return checked((uint)(((long)elementCount + workgroupSize - 1L) / workgroupSize));
    }

    /// <summary>
    /// Records and executes a compute dispatch. Caller MUST hold _computeLock.
    /// </summary>
    private void RecordAndExecuteComputeUnlocked(VulkanComputePipeline pipeline, int elementCount, ThreadCommandResources threadRes)
    {
        // Batch mode: record dispatch into the batch command buffer without submit.
        // Thread-affine: only the thread that called BeginBatch can record into the batch.
        // Non-owning threads fall through to the immediate-submit path.
        if (_batchMode && Environment.CurrentManagedThreadId == _batchOwnerThreadId)
        {
            var batchCmd = _batchThreadRes.CommandBuffer;

            // Insert barrier between dispatches to ensure data dependencies
            if (_batchDispatchCount > 0)
                RecordComputeMemoryBarrier(batchCmd);

            // Bind pipeline
            VulkanNativeBindings.vkCmdBindPipeline(
                batchCmd,
                VulkanNativeBindings.VK_PIPELINE_BIND_POINT_COMPUTE,
                pipeline.Handle);

            // Bind descriptor set
            var descriptorSet = RetainBatchDescriptorSet(pipeline);
            VulkanNativeBindings.vkCmdBindDescriptorSets(
                batchCmd,
                VulkanNativeBindings.VK_PIPELINE_BIND_POINT_COMPUTE,
                pipeline.Layout,
                0, 1, &descriptorSet,
                0, null);

            // Push constants
            uint sz = (uint)elementCount;
            VulkanNativeBindings.vkCmdPushConstants(
                batchCmd, pipeline.Layout,
                VulkanNativeBindings.VK_SHADER_STAGE_COMPUTE_BIT,
                0, sizeof(uint), &sz);

            // Dispatch
            uint wgCount = VulkanKernels.CalculateWorkgroupCount(elementCount);
            JournalDispatch(pipeline, wgCount, VulkanKernels.WorkgroupSize);
            VulkanNativeBindings.vkCmdDispatch(batchCmd, wgCount, 1, 1);

            _batchDispatchCount++;
            return;
        }

        // Non-batch mode: record + submit immediately (original path)
        var cmdBuffer = DispatchCommandBuffer(threadRes, out bool deferred);

        if (!deferred) VulkanNativeBindings.vkResetCommandBuffer(cmdBuffer, 0);

        var beginInfo = new VkCommandBufferBeginInfo
        {
            sType = VulkanNativeBindings.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            pNext = null,
            flags = VkCommandBufferUsageFlags.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
            pInheritanceInfo = null
        };

        if (!deferred) VulkanNativeBindings.vkBeginCommandBuffer(cmdBuffer, &beginInfo);
        else RecordComputeMemoryBarrier(cmdBuffer);

        VulkanNativeBindings.vkCmdBindPipeline(
            cmdBuffer,
            VulkanNativeBindings.VK_PIPELINE_BIND_POINT_COMPUTE,
            pipeline.Handle);

        var ds = deferred ? pipeline.RetainDescriptorSet() : pipeline.DescriptorSet;
        VulkanNativeBindings.vkCmdBindDescriptorSets(
            cmdBuffer,
            VulkanNativeBindings.VK_PIPELINE_BIND_POINT_COMPUTE,
            pipeline.Layout,
            0, 1, &ds,
            0, null);

        uint size = (uint)elementCount;
        VulkanNativeBindings.vkCmdPushConstants(
            cmdBuffer, pipeline.Layout,
            VulkanNativeBindings.VK_SHADER_STAGE_COMPUTE_BIT,
            0, sizeof(uint), &size);

        uint workgroupCount = VulkanKernels.CalculateWorkgroupCount(elementCount);
        JournalDispatch(pipeline, workgroupCount, VulkanKernels.WorkgroupSize);
        VulkanNativeBindings.vkCmdDispatch(cmdBuffer, workgroupCount, 1, 1);

        if (!deferred) VulkanNativeBindings.vkEndCommandBuffer(cmdBuffer);
        SubmitDispatch(cmdBuffer, threadRes, deferred, pipeline, ds);
    }

    /// <summary>
    /// Records and executes a compute dispatch with scalar push constant. Caller MUST hold _computeLock.
    /// </summary>
    private void RecordAndExecuteComputeWithScalarUnlocked(VulkanComputePipeline pipeline, int elementCount, float scalar, ThreadCommandResources threadRes)
    {
        var cmdBuffer = DispatchCommandBuffer(threadRes, out bool deferred);

        // Reset command buffer
        if (!deferred) VulkanNativeBindings.vkResetCommandBuffer(cmdBuffer, 0);

        // Begin recording
        var beginInfo = new VkCommandBufferBeginInfo
        {
            sType = VulkanNativeBindings.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            pNext = null,
            flags = VkCommandBufferUsageFlags.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
            pInheritanceInfo = null
        };

        if (!deferred) VulkanNativeBindings.vkBeginCommandBuffer(cmdBuffer, &beginInfo);
        else RecordComputeMemoryBarrier(cmdBuffer);

        // Bind pipeline
        VulkanNativeBindings.vkCmdBindPipeline(
            cmdBuffer,
            VulkanNativeBindings.VK_PIPELINE_BIND_POINT_COMPUTE,
            pipeline.Handle);

        // Bind descriptor set
        var descriptorSet = deferred ? pipeline.RetainDescriptorSet() : pipeline.DescriptorSet;
        VulkanNativeBindings.vkCmdBindDescriptorSets(
            cmdBuffer,
            VulkanNativeBindings.VK_PIPELINE_BIND_POINT_COMPUTE,
            pipeline.Layout,
            0, 1, &descriptorSet,
            0, null);

        // Push constants (size + scalar)
        Span<byte> pushData = stackalloc byte[sizeof(uint) + sizeof(float)];
        uint size = (uint)elementCount;
        Unsafe.WriteUnaligned(ref pushData[0], size);
        Unsafe.WriteUnaligned(ref pushData[sizeof(uint)], scalar);

        fixed (byte* pPushData = pushData)
        {
            VulkanNativeBindings.vkCmdPushConstants(
                cmdBuffer,
                pipeline.Layout,
                VulkanNativeBindings.VK_SHADER_STAGE_COMPUTE_BIT,
                0,
                (uint)pushData.Length,
                pPushData);
        }

        // Dispatch compute
        uint workgroupCount = VulkanKernels.CalculateWorkgroupCount(elementCount);
        JournalDispatch(pipeline, workgroupCount, VulkanKernels.WorkgroupSize);
        VulkanNativeBindings.vkCmdDispatch(cmdBuffer, workgroupCount, 1, 1);

        // End recording
        if (!deferred) VulkanNativeBindings.vkEndCommandBuffer(cmdBuffer);

        // Submit and wait using per-thread fence
        SubmitDispatch(cmdBuffer, threadRes, deferred, pipeline, descriptorSet);
    }

    #region Batch Execution

    /// <summary>
    /// Gets whether this backend supports batch recording.
    /// </summary>
    public bool SupportsBatchExecution => true;

    /// <inheritdoc/>
    public void UploadIntBufferInPlace(int[] data, IGpuBuffer buffer)
    {
        EnsureInitialized();
        if (data is null) throw new ArgumentNullException(nameof(data));
        if (buffer is null) throw new ArgumentNullException(nameof(buffer));
        if (data.Length > buffer.Size)
            throw new ArgumentException($"Host data ({data.Length}) exceeds buffer capacity ({buffer.Size}).", nameof(data));
        if (data.Length == 0) return;
        var packed = new float[data.Length];
        for (int i = 0; i < data.Length; i++)
            packed[i] = Int32BitsToSingleCompat(data[i]);
        UploadToBuffer(packed, buffer);
        // UploadToBuffer stages the whole buffer (a partial update is padded to buffer.Size), so that is
        // what crosses the boundary, not data.Length.
        GpuLaunchProbe.OnUpload((long)buffer.Size * sizeof(float), GpuBackendType.Vulkan);
    }

    /// <inheritdoc/>
    public void UploadBufferInPlace(float[] data, IGpuBuffer buffer)
    {
        EnsureInitialized();
        if (data is null) throw new ArgumentNullException(nameof(data));
        if (buffer is null) throw new ArgumentNullException(nameof(buffer));
        if (data.Length > buffer.Size)
            throw new ArgumentException($"Host data ({data.Length}) exceeds buffer capacity ({buffer.Size}).", nameof(data));
        if (data.Length == 0) return;
        UploadToBuffer(data, buffer);
        // UploadToBuffer stages the whole buffer, so that is what crosses the boundary.
        GpuLaunchProbe.OnUpload((long)buffer.Size * sizeof(float), GpuBackendType.Vulkan);
    }

    /// <inheritdoc/>
    public IGpuBuffer AllocateWorkspaceBuffer(int totalElements)
    {
        return AllocateBuffer(totalElements);
    }

    /// <summary>
    /// Begins recording a batch of GPU operations into a single command buffer.
    /// All subsequent GPU operations (Add, MatMul, etc.) are recorded without submission
    /// until EndBatch is called. This eliminates per-operation kernel launch overhead.
    /// </summary>
    public void BeginBatch()
    {
        EnsureInitialized();
        if (_batchMode)
            throw new InvalidOperationException("Already in batch mode.");

        var threadRes = _device.AcquireThreadResources();
        _batchThreadRes = threadRes;
        _batchDispatchCount = 0;
        ReleaseBatchDescriptorSets();

        var cmdBuffer = threadRes.CommandBuffer;
        VulkanNativeBindings.vkResetCommandBuffer(cmdBuffer, 0);

        var beginInfo = new VkCommandBufferBeginInfo
        {
            sType = VulkanNativeBindings.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            pNext = null,
            flags = VkCommandBufferUsageFlags.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
            pInheritanceInfo = null
        };
        VulkanNativeBindings.vkBeginCommandBuffer(cmdBuffer, &beginInfo);

        _batchMode = true;
        _batchOwnerThreadId = Environment.CurrentManagedThreadId;
    }

    /// <summary>
    /// Ends batch recording and submits all recorded operations as a single GPU submission.
    /// Blocks until all operations complete.
    /// </summary>
    public void EndBatch()
    {
        if (!_batchMode)
            throw new InvalidOperationException("Not in batch mode.");

        var cmdBuffer = _batchThreadRes.CommandBuffer;

        VulkanNativeBindings.vkEndCommandBuffer(cmdBuffer);
        _device.SubmitAndWait(cmdBuffer, _batchThreadRes.Fence);
        ReleaseBatchDescriptorSets();

        _batchMode = false;
        _batchDispatchCount = 0;
        ReleaseBuffersHeldForBatch();
    }

    /// <summary>
    /// Inserts a compute memory barrier between batched dispatches.
    /// Ensures all writes from previous dispatches are visible to subsequent reads.
    /// </summary>
    public void InsertBarrier(IGpuBuffer buffer)
    {
        if (!_batchMode) return;

        RecordComputeMemoryBarrier(_batchThreadRes.CommandBuffer);
    }

    /// <summary>
    /// Compute-to-compute barrier that makes earlier shader writes visible to later shader reads AND writes.
    /// Matching stage masks alone are only an execution dependency; without a memory barrier a following
    /// read-modify-write dispatch (an accumulating reduction) may not see the previous dispatch's writes.
    /// </summary>
    private static unsafe void RecordComputeMemoryBarrier(IntPtr commandBuffer)
    {
        // VkMemoryBarrier: uint32 sType @0, (pad), void* pNext @8, uint32 srcAccessMask @16, uint32 dstAccessMask @20.
        // The previous ulong[4] layout put dstAccessMask in the high half of slot 2, leaving it 0.
        const uint StructureTypeMemoryBarrier = 46, AccessShaderRead = 0x20, AccessShaderWrite = 0x40;
        byte* barrier = stackalloc byte[24];
        new Span<byte>(barrier, 24).Clear();
        *(uint*)barrier = StructureTypeMemoryBarrier;
        *(uint*)(barrier + 16) = AccessShaderWrite;
        *(uint*)(barrier + 20) = AccessShaderRead | AccessShaderWrite;
        VulkanNativeBindings.vkCmdPipelineBarrier(
            commandBuffer,
            (uint)VkPipelineStageFlags.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            (uint)VkPipelineStageFlags.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0, 1, (IntPtr)barrier, 0, null, 0, IntPtr.Zero);
    }

    #endregion

    /// <inheritdoc/>
    public void BeginSecondaryStream()
    {
        int currentThread = Environment.CurrentManagedThreadId;
        if (Interlocked.CompareExchange(ref _secondaryStreamOwnerThreadId, currentThread, 0) != 0)
            throw new InvalidOperationException("Secondary stream is already owned by another thread.");

        // Acquire a separate set of thread resources (command pool + command buffer + fence)
        // for concurrent recording. This allows overlapping compute dispatches.
        _secondaryThreadRes = _device.AcquireThreadResources();
        _inSecondaryStream = true;
    }

    /// <inheritdoc/>
    public void EndSecondaryStream()
    {
        int currentThread = Environment.CurrentManagedThreadId;
        if (!_inSecondaryStream || _secondaryStreamOwnerThreadId != currentThread)
            throw new InvalidOperationException("Not in a secondary stream on this thread.");

        // Every dispatch made inside the stream has been submitted already (each op submits its own command
        // buffer), so ending the stream is waiting for them. This used to submit the thread's command buffer
        // again, which re-ran whatever that buffer last recorded: harmless while that was the stream's own last
        // op, and a stale op over the stream's output once dispatches record into buffers of their own.
        _device.WaitForPendingSubmissions();
        _inSecondaryStream = false;
        Interlocked.Exchange(ref _secondaryStreamOwnerThreadId, 0);
    }

    /// <summary>
    /// Disposes the backend and releases all resources.
    /// </summary>
    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        _disposed = true;

        // Under the compute lock: a dispatch that already holds it finishes (every dispatch submits and waits)
        // before the placeholder is released, and one that takes it afterwards sees _disposed and stops. Without it
        // a call could read UnusedBinding, lose the race to this method, and then bind the released buffer.
        lock (_computeLock)
        {
            // Wait for device to finish
            _device.WaitIdle();

            // Parked buffers must be freed while the device still exists. The placeholder is pooled, so it is
            // released first and the drain frees it with the rest.
            IGpuBuffer? placeholder;
            lock (_unusedBindingLock)
            {
                placeholder = _unusedBindingBuffer;
                _unusedBindingBuffer = null;
            }
            VulkanGpuBuffer.ReleaseAndDrain(placeholder);
        }

        // Per-thread command resources are cleaned up by VulkanDevice.Cleanup()

        // Dispose pipelines
        foreach (var pipeline in _pipelineCache.Values)
        {
            pipeline.Dispose();
        }
        _pipelineCache.Clear();

        // Dispose GLSL-compiled pipelines
        foreach (var pipeline in _glslPipelineCache.Values)
        {
            pipeline.Dispose();
        }
        _glslPipelineCache.Clear();
        DisposeTunedKernelResources();
        _glslPipelineBySource.Clear();
        _glslPipelineByInstance.Clear();

        // Dispose shaders
        foreach (var shader in _shaderCache.Values)
        {
            shader.Dispose();
        }
        _shaderCache.Clear();

        // Dispose GLSL compiler
        _glslCompiler?.Dispose();

        _parityFlagSlots?.Dispose();
        _parityFlagSlots = null;

        // Dispose transfer
        _transfer?.Dispose();

        // Dispose cached complex scratch buffers
        if (_complexScratch is not null)
        {
            foreach (var buf in _complexScratch) buf?.Dispose();
            _complexScratch = null;
        }

        // Note: VulkanDevice is a singleton, don't dispose it here
        _initialized = false;
    }

    /// <summary>
    /// Gets backend information as a string.
    /// </summary>
    public override string ToString()
    {
        if (!_initialized)
        {
            return "VulkanBackend[Not Initialized]";
        }

        return $"VulkanBackend[{_device.VendorName} {_device.DeviceName}]";
    }
}

/// <summary>
/// Extension methods for VulkanBackend operations.
/// </summary>
public static class VulkanBackendExtensions
{
    /// <summary>
    /// Tries to execute a binary operation on the GPU.
    /// </summary>
    /// <param name="backend">The backend instance.</param>
    /// <param name="a">Input array A.</param>
    /// <param name="b">Input array B.</param>
    /// <param name="result">Output array.</param>
    /// <param name="kernelType">The operation.</param>
    /// <returns>True if successful.</returns>
    public static bool TryExecuteBinaryOp(
        this VulkanBackend backend,
        ReadOnlySpan<float> a,
        ReadOnlySpan<float> b,
        Span<float> result,
        VulkanKernelType kernelType)
    {
        try
        {
            if (!backend.IsAvailable)
            {
                return false;
            }

            backend.ExecuteBinaryOp(a, b, result, kernelType);
            return true;
        }
        catch
        {
            return false;
        }
    }

    /// <summary>
    /// Tries to execute a unary operation on the GPU.
    /// </summary>
    /// <param name="backend">The backend instance.</param>
    /// <param name="input">Input array.</param>
    /// <param name="result">Output array.</param>
    /// <param name="kernelType">The operation.</param>
    /// <returns>True if successful.</returns>
    public static bool TryExecuteUnaryOp(
        this VulkanBackend backend,
        ReadOnlySpan<float> input,
        Span<float> result,
        VulkanKernelType kernelType)
    {
        try
        {
            if (!backend.IsAvailable)
            {
                return false;
            }

            backend.ExecuteUnaryOp(input, result, kernelType);
            return true;
        }
        catch
        {
            return false;
        }
    }
}
