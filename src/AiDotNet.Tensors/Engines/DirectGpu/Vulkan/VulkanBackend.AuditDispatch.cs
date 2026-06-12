using System;

namespace AiDotNet.Tensors.Engines.DirectGpu.Vulkan;

// Generic N-buffer GLSL dispatch for the missing-kernels audit ops. Push constants are packed as a
// uint[] (ints reinterpreted, floats bit-cast) matching the kernel's push_constant block layout.
public sealed partial class VulkanBackend
{
    private void GlslDispatchN(string glslSource, int dispatchSize, IGpuBuffer[] buffers, uint[] pushConstants)
    {
        EnsureInitialized();
        if (dispatchSize <= 0) return;
        uint pcSize = (uint)(pushConstants.Length * sizeof(uint));
        var pipeline = GetOrCreateGlslPipeline(glslSource, buffers.Length, pcSize);
        if (pipeline is null)
            throw new InvalidOperationException("Vulkan GLSL audit-kernel pipeline creation failed (libshaderc required).");
        var storages = new VulkanBuffer[buffers.Length];
        for (int i = 0; i < buffers.Length; i++) storages[i] = AsVulkan(buffers[i]).Storage;
        var threadRes = _device.AcquireThreadResources();
        lock (_computeLock)
        {
            pipeline.UpdateDescriptorSet(storages);
            RecordAndExecuteWithPushData(pipeline, dispatchSize, pushConstants, pcSize, threadRes);
        }
    }
}
