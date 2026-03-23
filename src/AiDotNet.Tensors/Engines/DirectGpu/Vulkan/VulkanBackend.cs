// Copyright (c) AiDotNet. All rights reserved.
// Complete Vulkan GPU compute backend for tensor operations.

using System;
using System.Buffers;
using System.Collections.Concurrent;
using System.Runtime.CompilerServices;
using System.Threading;

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
public sealed unsafe partial class VulkanBackend : IDirectGpuBackend, IGpuBatchExecution
{
    private static readonly Lazy<VulkanBackend> _instance = new(
        () => new VulkanBackend(), LazyThreadSafetyMode.ExecutionAndPublication);

    private readonly VulkanDevice _device;
    private readonly ConcurrentDictionary<(VulkanKernelType kernelType, int bindingCount, uint pushConstantSize), VulkanComputePipeline> _pipelineCache;
    private readonly ConcurrentDictionary<VulkanKernelType, VulkanShaderModule> _shaderCache;
    private readonly ConcurrentDictionary<string, VulkanComputePipeline> _glslPipelineCache;
    private VulkanGlslCompiler? _glslCompiler;
    private VulkanBufferTransfer? _transfer;
    private readonly object _computeLock = new object();
    private bool _initialized;
    private bool _disposed;

    // Batch execution state — records multiple dispatches into a single command buffer.
    // Thread-affine: _batchOwnerThreadId tracks which thread owns the batch. Non-owning
    // threads fall through to the non-batch path to avoid recording into another thread's
    // command buffer.
    private bool _batchMode;
    private int _batchOwnerThreadId;
    private ThreadCommandResources _batchThreadRes;
    private int _batchDispatchCount;
    private bool _inSecondaryStream;
    private ThreadCommandResources _secondaryThreadRes;

    /// <summary>
    /// Gets the singleton backend instance.
    /// </summary>
    public static VulkanBackend Instance => _instance.Value;

    /// <summary>
    /// Gets whether the backend is available and initialized.
    /// </summary>
    public bool IsAvailable => _initialized && !_disposed;

    /// <summary>
    /// Gets the backend name.
    /// </summary>
    public string BackendName => "Vulkan";

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
            return false;
        }

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
        if (pipeline != null)
        {
            _pipelineCache.TryAdd(cacheKey, pipeline);
        }

        return pipeline;
    }

    /// <summary>
    /// Gets or creates a compute pipeline from a GLSL source string using runtime compilation.
    /// </summary>
    private VulkanComputePipeline? GetOrCreateGlslPipeline(string glslSource, int bindingCount, uint pushConstantSize = 0)
    {
        // Use SHA256 instead of GetHashCode to prevent cache key collisions
        string sourceHash;
        using (var sha = System.Security.Cryptography.SHA256.Create())
        {
            var hashBytes = sha.ComputeHash(System.Text.Encoding.UTF8.GetBytes(glslSource));
            sourceHash = BitConverter.ToString(hashBytes, 0, 8).Replace("-", "");
        }
        string cacheKey = $"{sourceHash}_{bindingCount}_{pushConstantSize}";
        if (_glslPipelineCache.TryGetValue(cacheKey, out var cached))
            return cached;

        if (_glslCompiler is null || !_glslCompiler.IsAvailable)
            return null;

        var shader = _glslCompiler.CompileToShaderModule(glslSource);
        if (shader is null)
            return null;

        var pipeline = VulkanComputePipeline.Create(shader, bindingCount, pushConstantSize);
        if (pipeline is not null)
            _glslPipelineCache.TryAdd(cacheKey, pipeline);

        return pipeline;
    }

    /// <summary>
    /// Executes a GLSL-compiled compute pipeline with 2 buffers + push constants.
    /// </summary>
    private void GlslUnaryOp(string glslSource, IGpuBuffer A, IGpuBuffer B, int size, uint pushConstantSize = sizeof(uint))
    {
        // Single-parameter shorthand — pushes {size} as the only push constant
        GlslUnaryOp(glslSource, A, B, size, new uint[] { (uint)size }, pushConstantSize);
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
        {
            CpuFallbackUnary(A, B, dispatchSize);
            return;
        }
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
        GlslBinaryOp(glslSource, A, B, C, size, new uint[] { (uint)size }, pushConstantSize);
    }

    private void GlslBinaryOp(string glslSource, IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int dispatchSize, uint[] pushConstants, uint pushConstantSize)
    {
        EnsureInitialized();
        if (dispatchSize <= 0) return;
        var pipeline = GetOrCreateGlslPipeline(glslSource, 3, pushConstantSize);
        if (pipeline is null)
        {
            CpuFallbackBinary(A, B, C, dispatchSize);
            return;
        }
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
    /// Executes a GLSL-compiled compute pipeline with 1 buffer (output only) + push constants.
    /// Used for generate ops like Eye, Linspace, TriangularMask.
    /// </summary>
    private void GlslGenerateOp(string glslSource, IGpuBuffer O, int size, uint pushConstantSize = sizeof(uint))
    {
        EnsureInitialized();
        if (size <= 0) return;
        var pipeline = GetOrCreateGlslPipeline(glslSource, 1, pushConstantSize);
        if (pipeline is null) throw new InvalidOperationException("Vulkan GLSL pipeline unavailable — install libshaderc for runtime compilation.");
        var vbO = AsVulkan(O);
        var threadRes = _device.AcquireThreadResources();
        lock (_computeLock)
        {
            pipeline.UpdateDescriptorSet(vbO.Storage);
            RecordAndExecuteComputeUnlocked(pipeline, size, threadRes);
        }
    }

    /// <summary>
    /// Executes a GLSL-compiled compute pipeline with 4 buffers + push constants.
    /// Used for backward ops needing grad_output, input, aux, grad_input.
    /// </summary>
    private void GlslQuadOp(string glslSource, IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, IGpuBuffer D, int size, uint pushConstantSize = sizeof(uint))
    {
        EnsureInitialized();
        if (size <= 0) return;
        var pipeline = GetOrCreateGlslPipeline(glslSource, 4, pushConstantSize);
        if (pipeline is null) throw new InvalidOperationException("Vulkan GLSL pipeline unavailable — install libshaderc for runtime compilation.");
        var vbA = AsVulkan(A); var vbB = AsVulkan(B); var vbC = AsVulkan(C); var vbD = AsVulkan(D);
        var threadRes = _device.AcquireThreadResources();
        lock (_computeLock)
        {
            pipeline.UpdateDescriptorSet(vbA.Storage, vbB.Storage, vbC.Storage, vbD.Storage);
            RecordAndExecuteComputeUnlocked(pipeline, size, threadRes);
        }
    }

    /// <summary>
    /// Executes a GLSL-compiled compute pipeline with 5 buffers + push constants.
    /// Used for complex backward ops needing grad_output, input, mean, variance, grad_input.
    /// </summary>
    private void GlslQuintOp(string glslSource, IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, IGpuBuffer D, IGpuBuffer E, int size, uint pushConstantSize = sizeof(uint))
    {
        EnsureInitialized();
        if (size <= 0) return;
        var pipeline = GetOrCreateGlslPipeline(glslSource, 5, pushConstantSize);
        if (pipeline is null) throw new InvalidOperationException("Vulkan GLSL pipeline unavailable — install libshaderc for runtime compilation.");
        var vbA = AsVulkan(A); var vbB = AsVulkan(B); var vbC = AsVulkan(C); var vbD = AsVulkan(D); var vbE = AsVulkan(E);
        var threadRes = _device.AcquireThreadResources();
        lock (_computeLock)
        {
            pipeline.UpdateDescriptorSet(vbA.Storage, vbB.Storage, vbC.Storage, vbD.Storage, vbE.Storage);
            RecordAndExecuteComputeUnlocked(pipeline, size, threadRes);
        }
    }

    // CPU fallback when GLSL pipeline creation fails (shaderc unavailable or compilation error).
    // When GLSL pipeline creation fails (shaderc unavailable or compilation error),
    // throw instead of silently returning wrong results. An identity copy would produce
    // incorrect data (e.g., sigmoid returning raw input values) with no indication of failure.
    private void CpuFallbackUnary(IGpuBuffer A, IGpuBuffer B, int size)
    {
        throw new InvalidOperationException(
            "Vulkan GLSL pipeline unavailable — cannot execute compute shader. " +
            "Install libshaderc to enable runtime GLSL compilation, or use a different GPU backend.");
    }

    private void CpuFallbackBinary(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int size)
    {
        throw new InvalidOperationException(
            "Vulkan GLSL pipeline unavailable — cannot execute compute shader. " +
            "Install libshaderc to enable runtime GLSL compilation, or use a different GPU backend.");
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
            throw new InvalidOperationException("Vulkan backend not initialized.");
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
            throw new InvalidOperationException("Vulkan buffer transfer not initialized.");
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
            throw new InvalidOperationException("Vulkan backend not initialized.");
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
            throw new InvalidOperationException("Vulkan buffer transfer not initialized.");
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
            throw new InvalidOperationException("Vulkan backend not initialized.");
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
            throw new InvalidOperationException("Vulkan buffer transfer not initialized.");
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
    private void RecordAndExecuteWithPushData(VulkanComputePipeline pipeline, int dispatchSize, uint[] pushConstants, uint pushConstantSize, ThreadCommandResources threadRes)
    {
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
        VulkanNativeBindings.vkCmdBindPipeline(cmdBuffer, VulkanNativeBindings.VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.Handle);
        var ds = pipeline.DescriptorSet;
        VulkanNativeBindings.vkCmdBindDescriptorSets(cmdBuffer, VulkanNativeBindings.VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.Layout, 0, 1, &ds, 0, null);

        // Push the caller-provided constant values
        fixed (uint* ptr = pushConstants)
        {
            VulkanNativeBindings.vkCmdPushConstants(cmdBuffer, pipeline.Layout, VulkanNativeBindings.VK_SHADER_STAGE_COMPUTE_BIT, 0, pushConstantSize, ptr);
        }

        uint workgroupCount = VulkanKernels.CalculateWorkgroupCount(dispatchSize);
        VulkanNativeBindings.vkCmdDispatch(cmdBuffer, workgroupCount, 1, 1);
        VulkanNativeBindings.vkEndCommandBuffer(cmdBuffer);
        _device.SubmitAndWait(cmdBuffer, threadRes.Fence);
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
            {
                // Full execution + memory dependency barrier between compute dispatches.
                // Uses execution dependency (no VkMemoryBarrier struct needed since
                // Vulkan guarantees that a pipeline barrier with matching stage masks
                // also creates a memory dependency for storage buffer accesses when
                // the stages are COMPUTE_SHADER_BIT on both sides).
                // NOTE: For strict correctness with independent buffer sets, per-buffer
                // VkBufferMemoryBarriers would be more precise but require tracking
                // which buffers each dispatch accesses.
                VulkanNativeBindings.vkCmdPipelineBarrier(
                    batchCmd,
                    (uint)VkPipelineStageFlags.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                    (uint)VkPipelineStageFlags.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                    0, 0, IntPtr.Zero, 0, null, 0, IntPtr.Zero);
            }

            // Bind pipeline
            VulkanNativeBindings.vkCmdBindPipeline(
                batchCmd,
                VulkanNativeBindings.VK_PIPELINE_BIND_POINT_COMPUTE,
                pipeline.Handle);

            // Bind descriptor set
            var descriptorSet = pipeline.DescriptorSet;
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
            VulkanNativeBindings.vkCmdDispatch(batchCmd, wgCount, 1, 1);

            _batchDispatchCount++;
            return;
        }

        // Non-batch mode: record + submit immediately (original path)
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

        VulkanNativeBindings.vkCmdBindPipeline(
            cmdBuffer,
            VulkanNativeBindings.VK_PIPELINE_BIND_POINT_COMPUTE,
            pipeline.Handle);

        var ds = pipeline.DescriptorSet;
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
        VulkanNativeBindings.vkCmdDispatch(cmdBuffer, workgroupCount, 1, 1);

        VulkanNativeBindings.vkEndCommandBuffer(cmdBuffer);
        _device.SubmitAndWait(cmdBuffer, threadRes.Fence);
    }

    /// <summary>
    /// Records and executes a compute dispatch with scalar push constant. Caller MUST hold _computeLock.
    /// </summary>
    private void RecordAndExecuteComputeWithScalarUnlocked(VulkanComputePipeline pipeline, int elementCount, float scalar, ThreadCommandResources threadRes)
    {
        var cmdBuffer = threadRes.CommandBuffer;

        // Reset command buffer
        VulkanNativeBindings.vkResetCommandBuffer(cmdBuffer, 0);

        // Begin recording
        var beginInfo = new VkCommandBufferBeginInfo
        {
            sType = VulkanNativeBindings.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            pNext = null,
            flags = VkCommandBufferUsageFlags.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
            pInheritanceInfo = null
        };

        VulkanNativeBindings.vkBeginCommandBuffer(cmdBuffer, &beginInfo);

        // Bind pipeline
        VulkanNativeBindings.vkCmdBindPipeline(
            cmdBuffer,
            VulkanNativeBindings.VK_PIPELINE_BIND_POINT_COMPUTE,
            pipeline.Handle);

        // Bind descriptor set
        var descriptorSet = pipeline.DescriptorSet;
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
        VulkanNativeBindings.vkCmdDispatch(cmdBuffer, workgroupCount, 1, 1);

        // End recording
        VulkanNativeBindings.vkEndCommandBuffer(cmdBuffer);

        // Submit and wait using per-thread fence
        _device.SubmitAndWait(cmdBuffer, threadRes.Fence);
    }

    #region Batch Execution

    /// <summary>
    /// Gets whether this backend supports batch recording.
    /// </summary>
    public bool SupportsBatchExecution => true;

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

        _batchMode = false;
        _batchDispatchCount = 0;
    }

    /// <summary>
    /// Inserts a compute memory barrier between batched dispatches.
    /// Ensures all writes from previous dispatches are visible to subsequent reads.
    /// </summary>
    public void InsertBarrier(IGpuBuffer buffer)
    {
        if (!_batchMode) return;

        var cmdBuffer = _batchThreadRes.CommandBuffer;

        // Full memory barrier: ensures all shader writes are visible to subsequent reads.
        // VK_STRUCTURE_TYPE_MEMORY_BARRIER = 46, SHADER_WRITE = 0x40, SHADER_READ = 0x20
        unsafe
        {
            var memBarrier = stackalloc ulong[4]; // sType, pNext, srcAccessMask, dstAccessMask
            memBarrier[0] = 46; // VK_STRUCTURE_TYPE_MEMORY_BARRIER
            memBarrier[1] = 0;  // pNext = null
            memBarrier[2] = 0x40; // VK_ACCESS_SHADER_WRITE_BIT
            memBarrier[3] = 0x20; // VK_ACCESS_SHADER_READ_BIT
            VulkanNativeBindings.vkCmdPipelineBarrier(
                cmdBuffer,
                (uint)VkPipelineStageFlags.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                (uint)VkPipelineStageFlags.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                0, 1, (IntPtr)memBarrier, 0, null, 0, IntPtr.Zero);
        }
    }

    #endregion

    /// <inheritdoc/>
    public void BeginSecondaryStream()
    {
        if (_inSecondaryStream)
            throw new InvalidOperationException("Already in a secondary stream.");

        // Acquire a separate set of thread resources (command pool + command buffer + fence)
        // for concurrent recording. This allows overlapping compute dispatches.
        _secondaryThreadRes = _device.AcquireThreadResources();
        _inSecondaryStream = true;
    }

    /// <inheritdoc/>
    public void EndSecondaryStream()
    {
        if (!_inSecondaryStream)
            throw new InvalidOperationException("Not in a secondary stream.");

        // Submit the secondary command buffer and wait for completion
        var cmdBuffer = _secondaryThreadRes.CommandBuffer;
        _device.SubmitAndWait(cmdBuffer, _secondaryThreadRes.Fence);
        _inSecondaryStream = false;
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

        // Wait for device to finish
        _device.WaitIdle();

        // Per-thread command resources are cleaned up by VulkanDevice.Cleanup()

        // Dispose pipelines
        foreach (var pipeline in _pipelineCache.Values)
        {
            pipeline.Dispose();
        }
        _pipelineCache.Clear();

        // Dispose shaders
        foreach (var shader in _shaderCache.Values)
        {
            shader.Dispose();
        }
        _shaderCache.Clear();

        // Dispose transfer
        _transfer?.Dispose();

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
