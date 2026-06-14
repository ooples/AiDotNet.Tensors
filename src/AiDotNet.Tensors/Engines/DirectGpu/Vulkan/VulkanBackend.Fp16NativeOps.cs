// Copyright (c) AiDotNet. All rights reserved.
// FP16-NATIVE elementwise / activation ops for the Vulkan backend (Tensors #558): GELU, ReLU and
// residual add over packed-half activations, computed in FP32 in-register. GPU counterpart of the CPU
// FP16-native emit — keeps the activation chain genuinely Half end-to-end. Runtime GLSL→SPIR-V via
// libshaderc; gated on IsGlslCompilerAvailable.

using System;

namespace AiDotNet.Tensors.Engines.DirectGpu.Vulkan;

public sealed partial class VulkanBackend
{
    /// <summary>
    /// True when the FP16-native op kernels can run: libshaderc must be present for runtime GLSL
    /// compilation (same gate as the FP16 GEMM path).
    /// </summary>
    public bool SupportsFp16NativeOps => IsGlslCompilerAvailable;

    /// <summary>GELU over a packed-half buffer: out[i] = gelu(in[i]); half in/out, FP32 math.</summary>
    public void Fp16Gelu(IGpuBuffer input, IGpuBuffer output, int n)
        => DispatchUnary(VulkanFp16NativeKernels.Gelu, input, output, n);

    /// <summary>ReLU over a packed-half buffer: out[i] = max(in[i], 0); half in/out, FP32 math.</summary>
    public void Fp16Relu(IGpuBuffer input, IGpuBuffer output, int n)
        => DispatchUnary(VulkanFp16NativeKernels.Relu, input, output, n);

    /// <summary>Residual add over packed-half buffers: out[i] = a[i] + b[i]; half in/out, FP32 accumulate.</summary>
    public void Fp16Add(IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, int n)
    {
        if (a is null) throw new ArgumentNullException(nameof(a));
        if (b is null) throw new ArgumentNullException(nameof(b));
        if (output is null) throw new ArgumentNullException(nameof(output));
        if (n <= 0) throw new ArgumentOutOfRangeException(nameof(n), "Element count must be positive.");

        var pipeline = GetOrCreateGlslPipeline(VulkanFp16NativeKernels.Add, 3, sizeof(uint));
        if (pipeline is null)
            throw new NotSupportedException("Vulkan FP16-native ops require libshaderc for runtime GLSL compilation, which is unavailable.");

        var vbA = AsVulkan(a);
        var vbB = AsVulkan(b);
        var vbO = AsVulkan(output);
        uint numWords = ((uint)n + 1u) >> 1;
        var pc = new uint[] { (uint)n };
        var threadRes = _device.AcquireThreadResources();
        lock (_computeLock)
        {
            pipeline.UpdateDescriptorSet(vbA.Storage, vbB.Storage, vbO.Storage);
            RecordAndExecuteWithPushData(pipeline, (int)numWords, pc, sizeof(uint), threadRes);
        }
    }

    private void DispatchUnary(string glsl, IGpuBuffer input, IGpuBuffer output, int n)
    {
        if (input is null) throw new ArgumentNullException(nameof(input));
        if (output is null) throw new ArgumentNullException(nameof(output));
        if (n <= 0) throw new ArgumentOutOfRangeException(nameof(n), "Element count must be positive.");

        var pipeline = GetOrCreateGlslPipeline(glsl, 2, sizeof(uint));
        if (pipeline is null)
            throw new NotSupportedException("Vulkan FP16-native ops require libshaderc for runtime GLSL compilation, which is unavailable.");

        var vbI = AsVulkan(input);
        var vbO = AsVulkan(output);
        uint numWords = ((uint)n + 1u) >> 1;
        var pc = new uint[] { (uint)n };
        var threadRes = _device.AcquireThreadResources();
        lock (_computeLock)
        {
            pipeline.UpdateDescriptorSet(vbI.Storage, vbO.Storage);
            RecordAndExecuteWithPushData(pipeline, (int)numWords, pc, sizeof(uint), threadRes);
        }
    }
}
