// Copyright (c) AiDotNet. All rights reserved.
// Vulkan shader module management for SPIR-V compute shaders.

using System;
using System.Collections.Generic;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.Vulkan;

/// <summary>
/// Represents a compiled SPIR-V shader module for compute operations.
/// </summary>
/// <remarks>
/// <para><b>SPIR-V Shaders:</b></para>
/// <para>
/// Vulkan uses SPIR-V as its shader intermediate representation. Shaders are
/// pre-compiled from GLSL or HLSL to SPIR-V bytecode and loaded at runtime.
/// </para>
/// <para><b>Compute Shader Layout:</b></para>
/// <para>
/// Compute shaders use storage buffers for input/output data and push constants
/// or uniform buffers for shader parameters. The workgroup size is specified
/// in the shader source.
/// </para>
/// </remarks>
public sealed unsafe class VulkanShaderModule : IDisposable
{
    private readonly VulkanDevice _device;
    private IntPtr _shaderModule;
    private readonly string _entryPoint;
    private bool _disposed;

    /// <summary>
    /// Gets the shader module handle.
    /// </summary>
    public IntPtr Handle => _shaderModule;

    /// <summary>
    /// Gets the shader entry point name.
    /// </summary>
    public string EntryPoint => _entryPoint;

    /// <summary>
    /// Gets whether the module is valid.
    /// </summary>
    public bool IsValid => _shaderModule != IntPtr.Zero && !_disposed;

    /// <summary>
    /// Creates a shader module from SPIR-V bytecode.
    /// </summary>
    /// <param name="spirvCode">The SPIR-V bytecode.</param>
    /// <param name="entryPoint">The shader entry point name.</param>
    private VulkanShaderModule(byte[] spirvCode, string entryPoint)
    {
        _device = VulkanDevice.Instance;
        _entryPoint = entryPoint;
        CreateShaderModule(spirvCode);
    }

    /// <summary>
    /// Creates a shader module from SPIR-V bytecode.
    /// </summary>
    /// <param name="spirvCode">The SPIR-V bytecode as uint array.</param>
    /// <param name="entryPoint">The shader entry point name.</param>
    /// <returns>The shader module, or null if creation failed.</returns>
    public static VulkanShaderModule? Create(uint[] spirvCode, string entryPoint = "main")
    {
        if (spirvCode == null || spirvCode.Length == 0)
        {
            return null;
        }

        var device = VulkanDevice.Instance;
        if (!device.IsInitialized)
        {
            return null;
        }

        // Convert uint[] to byte[]
        var byteCode = new byte[spirvCode.Length * sizeof(uint)];
        Buffer.BlockCopy(spirvCode, 0, byteCode, 0, byteCode.Length);

        var module = new VulkanShaderModule(byteCode, entryPoint);
        return module.IsValid ? module : null;
    }

    /// <summary>
    /// Creates a shader module from SPIR-V bytecode.
    /// </summary>
    /// <param name="spirvCode">The SPIR-V bytecode.</param>
    /// <param name="entryPoint">The shader entry point name.</param>
    /// <returns>The shader module, or null if creation failed.</returns>
    public static VulkanShaderModule? Create(byte[] spirvCode, string entryPoint = "main")
    {
        if (spirvCode == null || spirvCode.Length == 0)
        {
            return null;
        }

        var device = VulkanDevice.Instance;
        if (!device.IsInitialized)
        {
            return null;
        }

        var module = new VulkanShaderModule(spirvCode, entryPoint);
        return module.IsValid ? module : null;
    }

    private void CreateShaderModule(byte[] spirvCode)
    {
        if (!_device.IsInitialized)
        {
            return;
        }

        // SPIR-V spec requires codeSize to be a multiple of 4
        // Copy byte[] into uint[] so pCode points to 32-bit words and codeSize can be rounded up to a 4-byte multiple as required by Vulkan
        int uint32Count = (spirvCode.Length + 3) / 4;
        var alignedCode = new uint[uint32Count];
        Buffer.BlockCopy(spirvCode, 0, alignedCode, 0, spirvCode.Length);

        // codeSize must be a multiple of 4 per Vulkan spec (VUID-VkShaderModuleCreateInfo-codeSize-01085)
        nuint alignedCodeSize = (nuint)(uint32Count * 4);

        fixed (uint* pCode = alignedCode)
        {
            var createInfo = new VkShaderModuleCreateInfo
            {
                sType = VulkanNativeBindings.VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                pNext = null,
                flags = 0,
                codeSize = alignedCodeSize,
                pCode = pCode
            };

            var result = VulkanNativeBindings.vkCreateShaderModule(
                _device.Device, &createInfo, IntPtr.Zero, out _shaderModule);

            if (result != VulkanNativeBindings.VK_SUCCESS)
            {
                _shaderModule = IntPtr.Zero;
            }
        }
    }

    /// <summary>
    /// Creates a pipeline shader stage create info for this module.
    /// </summary>
    /// <param name="entryPointBytes">Pointer to entry point name bytes (must stay alive).</param>
    /// <returns>The shader stage create info.</returns>
    public VkPipelineShaderStageCreateInfo CreateShaderStageInfo(byte* entryPointBytes)
    {
        return new VkPipelineShaderStageCreateInfo
        {
            sType = VulkanNativeBindings.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            pNext = null,
            flags = 0,
            stage = VulkanNativeBindings.VK_SHADER_STAGE_COMPUTE_BIT,
            module = _shaderModule,
            pName = entryPointBytes,
            pSpecializationInfo = null
        };
    }

    /// <summary>
    /// Disposes the shader module.
    /// </summary>
    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        _disposed = true;

        if (_shaderModule != IntPtr.Zero)
        {
            VulkanNativeBindings.vkDestroyShaderModule(_device.Device, _shaderModule, IntPtr.Zero);
            _shaderModule = IntPtr.Zero;
        }
    }
}

/// <summary>
/// Manages a compute pipeline with its associated resources.
/// </summary>
public sealed unsafe class VulkanComputePipeline : IDisposable
{
    private readonly VulkanDevice _device;
    private IntPtr _descriptorSetLayout;

    /// <summary>What the launch journal calls this pipeline (#996).</summary>
    internal string Name { get; set; } = "vulkan-pipeline";
    private IntPtr _pipelineLayout;
    private IntPtr _pipeline;
    // Every descriptor pool this pipeline has created; the last one is the one new sets come from.
    private readonly List<IntPtr> _descriptorPools = new List<IntPtr>();
    private IntPtr _descriptorPool;
    // The set the next dispatch binds. A dispatch that is recorded but not yet executed retains it
    // (RetainDescriptorSet), and the next UpdateDescriptorSet then writes a different set instead of this one.
    private IntPtr _descriptorSet;
    private bool _descriptorSetRetained;
    private readonly Stack<IntPtr> _freeDescriptorSets = new Stack<IntPtr>();
    // Sets a pool beyond the first holds. The first pool holds one: most pipelines never have two dispatches pending.
    private const int DescriptorSetsPerGrowth = 16;
    private readonly int _bindingCount;
    private readonly uint _pushConstantSize;
    private readonly object _dispatchLock = new object();
    private bool _disposed;

    /// <summary>
    /// Gets the pipeline handle.
    /// </summary>
    public IntPtr Handle => _pipeline;

    /// <summary>
    /// Gets the pipeline layout handle.
    /// </summary>
    public IntPtr Layout => _pipelineLayout;

    /// <summary>
    /// Gets the descriptor set handle.
    /// </summary>
    public IntPtr DescriptorSet => _descriptorSet;

    /// <summary>
    /// Gets whether the pipeline is valid.
    /// </summary>
    public bool IsValid => _pipeline != IntPtr.Zero && !_disposed;

    /// <summary>
    /// Gets the number of buffer bindings.
    /// </summary>
    public int BindingCount => _bindingCount;

    private VulkanComputePipeline(int bindingCount, uint pushConstantSize)
    {
        _device = VulkanDevice.Instance;
        _bindingCount = bindingCount;
        _pushConstantSize = pushConstantSize;
    }

    /// <summary>
    /// Creates a compute pipeline from a shader module.
    /// </summary>
    /// <param name="shaderModule">The shader module.</param>
    /// <param name="bindingCount">Number of storage buffer bindings.</param>
    /// <param name="pushConstantSize">Size of push constants in bytes.</param>
    /// <returns>The compute pipeline, or null if creation failed.</returns>
    public static VulkanComputePipeline? Create(
        VulkanShaderModule shaderModule,
        int bindingCount,
        uint pushConstantSize = 0)
    {
        if (!shaderModule.IsValid)
        {
            return null;
        }

        var pipeline = new VulkanComputePipeline(bindingCount, pushConstantSize);

        if (!pipeline.CreateDescriptorSetLayout() ||
            !pipeline.CreatePipelineLayout() ||
            !pipeline.CreatePipeline(shaderModule) ||
            !pipeline.CreateDescriptorPool(1) ||
            (pipeline._descriptorSet = pipeline.AllocateDescriptorSet()) == IntPtr.Zero)
        {
            pipeline.Dispose();
            return null;
        }

        return pipeline;
    }

    private bool CreateDescriptorSetLayout()
    {
        if (_bindingCount == 0)
        {
            var emptyCreateInfo = new VkDescriptorSetLayoutCreateInfo
            {
                sType = VulkanNativeBindings.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                pNext = null,
                flags = 0,
                bindingCount = 0,
                pBindings = null
            };

            var emptyResult = VulkanNativeBindings.vkCreateDescriptorSetLayout(
                _device.Device, &emptyCreateInfo, IntPtr.Zero, out _descriptorSetLayout);

            return emptyResult == VulkanNativeBindings.VK_SUCCESS && _descriptorSetLayout != IntPtr.Zero;
        }

        var bindings = stackalloc VkDescriptorSetLayoutBinding[_bindingCount];

        for (int i = 0; i < _bindingCount; i++)
        {
            bindings[i] = new VkDescriptorSetLayoutBinding
            {
                binding = (uint)i,
                descriptorType = VulkanNativeBindings.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                descriptorCount = 1,
                stageFlags = VulkanNativeBindings.VK_SHADER_STAGE_COMPUTE_BIT,
                pImmutableSamplers = null
            };
        }

        var createInfo = new VkDescriptorSetLayoutCreateInfo
        {
            sType = VulkanNativeBindings.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            pNext = null,
            flags = 0,
            bindingCount = (uint)_bindingCount,
            pBindings = bindings
        };

        var result = VulkanNativeBindings.vkCreateDescriptorSetLayout(
            _device.Device, &createInfo, IntPtr.Zero, out _descriptorSetLayout);

        return result == VulkanNativeBindings.VK_SUCCESS && _descriptorSetLayout != IntPtr.Zero;
    }

    private bool CreatePipelineLayout()
    {
        var setLayout = _descriptorSetLayout;

        VkPushConstantRange pushConstantRange = default;
        uint pushConstantRangeCount = 0;
        VkPushConstantRange* pPushConstantRanges = null;

        if (_pushConstantSize > 0)
        {
            pushConstantRange = new VkPushConstantRange
            {
                stageFlags = VulkanNativeBindings.VK_SHADER_STAGE_COMPUTE_BIT,
                offset = 0,
                size = _pushConstantSize
            };
            pushConstantRangeCount = 1;
            pPushConstantRanges = &pushConstantRange;
        }

        var createInfo = new VkPipelineLayoutCreateInfo
        {
            sType = VulkanNativeBindings.VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            pNext = null,
            flags = 0,
            setLayoutCount = 1,
            pSetLayouts = &setLayout,
            pushConstantRangeCount = pushConstantRangeCount,
            pPushConstantRanges = pPushConstantRanges
        };

        var result = VulkanNativeBindings.vkCreatePipelineLayout(
            _device.Device, &createInfo, IntPtr.Zero, out _pipelineLayout);

        return result == VulkanNativeBindings.VK_SUCCESS && _pipelineLayout != IntPtr.Zero;
    }

    private bool CreatePipeline(VulkanShaderModule shaderModule)
    {
        var entryPointBytes = Encoding.UTF8.GetBytes(shaderModule.EntryPoint + "\0");

        fixed (byte* pEntryPoint = entryPointBytes)
        {
            var stageInfo = shaderModule.CreateShaderStageInfo(pEntryPoint);

            var createInfo = new VkComputePipelineCreateInfo
            {
                sType = VulkanNativeBindings.VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
                pNext = null,
                flags = 0,
                stage = stageInfo,
                layout = _pipelineLayout,
                basePipelineHandle = IntPtr.Zero,
                basePipelineIndex = -1
            };

            IntPtr pipeline;
            var result = VulkanNativeBindings.vkCreateComputePipelines(
                _device.Device, IntPtr.Zero, 1, &createInfo, IntPtr.Zero, &pipeline);

            _pipeline = pipeline;
            return result == VulkanNativeBindings.VK_SUCCESS && _pipeline != IntPtr.Zero;
        }
    }

    private bool CreateDescriptorPool(int sets)
    {
        var poolSize = new VkDescriptorPoolSize
        {
            type = VulkanNativeBindings.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            descriptorCount = checked((uint)(_bindingCount * sets))
        };

        var createInfo = new VkDescriptorPoolCreateInfo
        {
            sType = VulkanNativeBindings.VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            pNext = null,
            flags = 0,
            maxSets = (uint)sets,
            poolSizeCount = 1,
            pPoolSizes = &poolSize
        };

        var result = VulkanNativeBindings.vkCreateDescriptorPool(
            _device.Device, &createInfo, IntPtr.Zero, out IntPtr pool);

        if (result != VulkanNativeBindings.VK_SUCCESS || pool == IntPtr.Zero) return false;
        _descriptorPool = pool;
        _descriptorPools.Add(pool);
        return true;
    }

    // A set from the newest pool, or IntPtr.Zero.
    private IntPtr AllocateDescriptorSet()
    {
        var layout = _descriptorSetLayout;

        var allocInfo = new VkDescriptorSetAllocateInfo
        {
            sType = VulkanNativeBindings.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            pNext = null,
            descriptorPool = _descriptorPool,
            descriptorSetCount = 1,
            pSetLayouts = &layout
        };

        IntPtr descriptorSet;
        var result = VulkanNativeBindings.vkAllocateDescriptorSets(
            _device.Device, &allocInfo, &descriptorSet);

        return result == VulkanNativeBindings.VK_SUCCESS ? descriptorSet : IntPtr.Zero;
    }

    /// <summary>
    /// Keeps the set the last <see cref="UpdateDescriptorSet"/> wrote for a dispatch that is recorded but has not
    /// executed yet, and returns it. Until <see cref="ReleaseDescriptorSet"/> hands it back, the pipeline's next
    /// dispatches write and bind other sets, so they cannot retarget this one.
    /// </summary>
    internal IntPtr RetainDescriptorSet()
    {
        lock (_dispatchLock)
        {
            _descriptorSetRetained = true;
            return _descriptorSet;
        }
    }

    /// <summary>Hands back a set from <see cref="RetainDescriptorSet"/> once the dispatch that bound it has executed.</summary>
    internal void ReleaseDescriptorSet(IntPtr descriptorSet)
    {
        if (descriptorSet == IntPtr.Zero) return;
        lock (_dispatchLock)
        {
            if (_disposed) return;
            if (descriptorSet == _descriptorSet) _descriptorSetRetained = false;
            else _freeDescriptorSets.Push(descriptorSet);
        }
    }

    /// <summary>Descriptor sets this pipeline has allocated, for tests.</summary>
    internal int AllocatedDescriptorSetCount
    {
        get { lock (_dispatchLock) return _descriptorPools.Count == 0 ? 0 : 1 + (_descriptorPools.Count - 1) * DescriptorSetsPerGrowth; }
    }

    // Caller holds _dispatchLock. A set nothing pending has bound: a released one, or one from a new pool.
    private IntPtr TakeFreeDescriptorSet()
    {
        if (_freeDescriptorSets.Count > 0) return _freeDescriptorSets.Pop();
        if (!CreateDescriptorPool(DescriptorSetsPerGrowth))
            throw new InvalidOperationException($"Failed to create a descriptor pool for Vulkan pipeline '{Name}'.");
        for (int i = 0; i < DescriptorSetsPerGrowth; i++)
        {
            IntPtr set = AllocateDescriptorSet();
            if (set == IntPtr.Zero)
                throw new InvalidOperationException($"Failed to allocate a descriptor set for Vulkan pipeline '{Name}'.");
            _freeDescriptorSets.Push(set);
        }
        return _freeDescriptorSets.Pop();
    }

    /// <summary>
    /// Updates the descriptor set with buffer bindings.
    /// </summary>
    /// <param name="buffers">The buffers to bind.</param>
    public void UpdateDescriptorSet(params VulkanBuffer[] buffers)
    {
        if (_disposed || buffers == null || buffers.Length == 0)
        {
            return;
        }

        if (buffers.Length != _bindingCount)
        {
            throw new ArgumentException(
                $"Buffer count mismatch: expected {_bindingCount} buffers for pipeline bindings, got {buffers.Length}. " +
                "All bindings must be provided to avoid undefined behavior.",
                nameof(buffers));
        }

        lock (_dispatchLock)
        {
            // A dispatch recorded but not yet executed reads the set it bound when it is submitted, so that set
            // is left alone and this dispatch gets another.
            if (_descriptorSetRetained)
            {
                _descriptorSet = TakeFreeDescriptorSet();
                _descriptorSetRetained = false;
            }

            int count = buffers.Length;
            var bufferInfos = stackalloc VkDescriptorBufferInfo[count];
            var writes = stackalloc VkWriteDescriptorSet[count];

            for (int i = 0; i < count; i++)
            {
                bufferInfos[i] = buffers[i].GetDescriptorInfo();

                writes[i] = new VkWriteDescriptorSet
                {
                    sType = VulkanNativeBindings.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    pNext = null,
                    dstSet = _descriptorSet,
                    dstBinding = (uint)i,
                    dstArrayElement = 0,
                    descriptorCount = 1,
                    descriptorType = VulkanNativeBindings.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    pImageInfo = null,
                    pBufferInfo = &bufferInfos[i],
                    pTexelBufferView = null
                };
            }

            VulkanNativeBindings.vkUpdateDescriptorSets(
                _device.Device, (uint)count, writes, 0, IntPtr.Zero);
        }
    }

    /// <summary>
    /// Disposes the pipeline and associated resources.
    /// </summary>
    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        _disposed = true;

        if (_pipeline != IntPtr.Zero)
        {
            VulkanNativeBindings.vkDestroyPipeline(_device.Device, _pipeline, IntPtr.Zero);
            _pipeline = IntPtr.Zero;
        }

        if (_pipelineLayout != IntPtr.Zero)
        {
            VulkanNativeBindings.vkDestroyPipelineLayout(_device.Device, _pipelineLayout, IntPtr.Zero);
            _pipelineLayout = IntPtr.Zero;
        }

        // Destroying a pool frees every set allocated from it.
        foreach (IntPtr pool in _descriptorPools)
            VulkanNativeBindings.vkDestroyDescriptorPool(_device.Device, pool, IntPtr.Zero);
        _descriptorPools.Clear();
        _freeDescriptorSets.Clear();
        _descriptorPool = IntPtr.Zero;
        _descriptorSet = IntPtr.Zero;

        if (_descriptorSetLayout != IntPtr.Zero)
        {
            VulkanNativeBindings.vkDestroyDescriptorSetLayout(_device.Device, _descriptorSetLayout, IntPtr.Zero);
            _descriptorSetLayout = IntPtr.Zero;
        }
    }
}
