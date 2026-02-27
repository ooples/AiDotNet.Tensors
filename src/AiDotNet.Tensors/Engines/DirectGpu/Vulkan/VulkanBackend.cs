// Copyright (c) AiDotNet. All rights reserved.
// Complete Vulkan GPU compute backend for tensor operations.

using System;
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
public sealed unsafe partial class VulkanBackend : IDirectGpuBackend
{
    private static readonly Lazy<VulkanBackend> _instance = new(
        () => new VulkanBackend(), LazyThreadSafetyMode.ExecutionAndPublication);

    private readonly VulkanDevice _device;
    private readonly ConcurrentDictionary<(VulkanKernelType kernelType, int bindingCount, uint pushConstantSize), VulkanComputePipeline> _pipelineCache;
    private readonly ConcurrentDictionary<VulkanKernelType, VulkanShaderModule> _shaderCache;
    private VulkanBufferTransfer? _transfer;
    private IntPtr _commandBuffer;
    private readonly object _computeLock = new object();
    private bool _initialized;
    private bool _disposed;

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
    /// Gets the vendor name.
    /// </summary>
    public string VendorName => _device.VendorName;

    /// <summary>
    /// Gets the device vendor.
    /// </summary>
    public string DeviceVendor => _device.VendorName;

    /// <summary>
    /// Gets the number of compute units (workgroup invocations).
    /// </summary>
    public int ComputeUnits => (int)_device.MaxWorkgroupSize;

    /// <summary>
    /// Gets the total global memory in bytes.
    /// </summary>
    public long GlobalMemoryBytes => _device.MaxStorageBufferRange;

    /// <summary>
    /// Gets the local (shared) memory per workgroup in bytes.
    /// </summary>
    public long LocalMemoryBytes => _device.MaxSharedMemorySize;

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
        AllocateCommandBuffer();

        _initialized = true;
        return true;
    }

    private void AllocateCommandBuffer()
    {
        var allocInfo = new VkCommandBufferAllocateInfo
        {
            sType = VulkanNativeBindings.VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            pNext = null,
            commandPool = _device.CommandPool,
            level = VulkanNativeBindings.VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            commandBufferCount = 1
        };

        IntPtr cmdBuffer;
        int result = VulkanNativeBindings.vkAllocateCommandBuffers(_device.Device, &allocInfo, &cmdBuffer);
        if (result != VulkanNativeBindings.VK_SUCCESS || cmdBuffer == IntPtr.Zero)
        {
            throw new InvalidOperationException($"Failed to allocate Vulkan command buffer: {result}");
        }

        _commandBuffer = cmdBuffer;
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
        _transfer!.CopyToDevice(stagingA, bufferA);
        _transfer.CopyToDevice(stagingB, bufferB);

        // Get or create pipeline
        var pipeline = GetOrCreatePipeline(kernelType, 3, sizeof(uint));
        if (pipeline == null)
        {
            throw new InvalidOperationException($"Failed to create pipeline for {kernelType}.");
        }

        // Lock around descriptor set update + dispatch to prevent concurrent mutation
        lock (_computeLock)
        {
            pipeline.UpdateDescriptorSet(bufferA, bufferB, bufferC);
            RecordAndExecuteComputeUnlocked(pipeline, size);
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
        _transfer!.CopyToDevice(stagingA, bufferA);

        // Get or create pipeline
        var pipeline = GetOrCreatePipeline(kernelType, 2, sizeof(uint));
        if (pipeline == null)
        {
            throw new InvalidOperationException($"Failed to create pipeline for {kernelType}.");
        }

        // Lock around descriptor set update + dispatch to prevent concurrent mutation
        lock (_computeLock)
        {
            pipeline.UpdateDescriptorSet(bufferA, bufferB);
            RecordAndExecuteComputeUnlocked(pipeline, size);
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
        _transfer!.CopyToDevice(stagingA, bufferA);

        // Get or create pipeline (push constants: uint size, float scalar)
        var pipeline = GetOrCreatePipeline(VulkanKernelType.ScalarMultiply, 2, sizeof(uint) + sizeof(float));
        if (pipeline == null)
        {
            throw new InvalidOperationException("Failed to create scalar multiply pipeline.");
        }

        // Lock around descriptor set update + dispatch to prevent concurrent mutation
        lock (_computeLock)
        {
            pipeline.UpdateDescriptorSet(bufferA, bufferB);
            RecordAndExecuteComputeWithScalarUnlocked(pipeline, size, scalar);
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
        ExecuteUnaryOp(input, result, VulkanKernelType.Tanh);
    }

    /// <summary>
    /// Records and executes a compute dispatch. Caller MUST hold _computeLock.
    /// </summary>
    private void RecordAndExecuteComputeUnlocked(VulkanComputePipeline pipeline, int elementCount)
    {
        // Reset command buffer
        VulkanNativeBindings.vkResetCommandBuffer(_commandBuffer, 0);

        // Begin recording
        var beginInfo = new VkCommandBufferBeginInfo
        {
            sType = VulkanNativeBindings.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            pNext = null,
            flags = VkCommandBufferUsageFlags.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
            pInheritanceInfo = null
        };

        VulkanNativeBindings.vkBeginCommandBuffer(_commandBuffer, &beginInfo);

        // Bind pipeline
        VulkanNativeBindings.vkCmdBindPipeline(
            _commandBuffer,
            VulkanNativeBindings.VK_PIPELINE_BIND_POINT_COMPUTE,
            pipeline.Handle);

        // Bind descriptor set
        var descriptorSet = pipeline.DescriptorSet;
        VulkanNativeBindings.vkCmdBindDescriptorSets(
            _commandBuffer,
            VulkanNativeBindings.VK_PIPELINE_BIND_POINT_COMPUTE,
            pipeline.Layout,
            0, 1, &descriptorSet,
            0, null);

        // Push constants (size)
        uint size = (uint)elementCount;
        VulkanNativeBindings.vkCmdPushConstants(
            _commandBuffer,
            pipeline.Layout,
            VulkanNativeBindings.VK_SHADER_STAGE_COMPUTE_BIT,
            0,
            sizeof(uint),
            &size);

        // Dispatch compute
        uint workgroupCount = VulkanKernels.CalculateWorkgroupCount(elementCount);
        VulkanNativeBindings.vkCmdDispatch(_commandBuffer, workgroupCount, 1, 1);

        // End recording
        VulkanNativeBindings.vkEndCommandBuffer(_commandBuffer);

        // Submit and wait
        _device.SubmitAndWait(_commandBuffer);
    }

    /// <summary>
    /// Records and executes a compute dispatch with scalar push constant. Caller MUST hold _computeLock.
    /// </summary>
    private void RecordAndExecuteComputeWithScalarUnlocked(VulkanComputePipeline pipeline, int elementCount, float scalar)
    {
        // Reset command buffer
        VulkanNativeBindings.vkResetCommandBuffer(_commandBuffer, 0);

        // Begin recording
        var beginInfo = new VkCommandBufferBeginInfo
        {
            sType = VulkanNativeBindings.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            pNext = null,
            flags = VkCommandBufferUsageFlags.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
            pInheritanceInfo = null
        };

        VulkanNativeBindings.vkBeginCommandBuffer(_commandBuffer, &beginInfo);

        // Bind pipeline
        VulkanNativeBindings.vkCmdBindPipeline(
            _commandBuffer,
            VulkanNativeBindings.VK_PIPELINE_BIND_POINT_COMPUTE,
            pipeline.Handle);

        // Bind descriptor set
        var descriptorSet = pipeline.DescriptorSet;
        VulkanNativeBindings.vkCmdBindDescriptorSets(
            _commandBuffer,
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
                _commandBuffer,
                pipeline.Layout,
                VulkanNativeBindings.VK_SHADER_STAGE_COMPUTE_BIT,
                0,
                (uint)pushData.Length,
                pPushData);
        }

        // Dispatch compute
        uint workgroupCount = VulkanKernels.CalculateWorkgroupCount(elementCount);
        VulkanNativeBindings.vkCmdDispatch(_commandBuffer, workgroupCount, 1, 1);

        // End recording
        VulkanNativeBindings.vkEndCommandBuffer(_commandBuffer);

        // Submit and wait
        _device.SubmitAndWait(_commandBuffer);
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

        // Dispose command buffer
        if (_commandBuffer != IntPtr.Zero)
        {
            var cmdPtr = _commandBuffer;
            VulkanNativeBindings.vkFreeCommandBuffers(
                _device.Device, _device.CommandPool, 1, &cmdPtr);
            _commandBuffer = IntPtr.Zero;
        }

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
