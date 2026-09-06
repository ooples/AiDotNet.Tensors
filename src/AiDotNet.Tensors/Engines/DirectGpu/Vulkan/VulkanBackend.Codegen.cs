// Copyright (c) AiDotNet. All rights reserved.

using System.Diagnostics;
using AiDotNet.Tensors.Engines.Compilation.Codegen;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Glsl;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ir;
using AiDotNet.Tensors.Helpers.Autotune;

namespace AiDotNet.Tensors.Engines.DirectGpu.Vulkan;

/// <summary>Evidence returned by a synchronized fused Vulkan compute-pipeline launch.</summary>
public readonly record struct VulkanFusedKernelExecutionEvidence(
    KernelTuningBackend Backend,
    CodegenNativeCompilationKind Compilation,
    CodegenNativeArtifactKind NativeArtifact,
    string DeviceName,
    string DeviceVendor,
    long SpirvSizeBytes,
    TimeSpan SynchronizedLaunchDuration,
    int NativeLaunchCount,
    int LocalWorkgroupSize);

public sealed partial class VulkanBackend
{
    /// <inheritdoc/>
    public KernelTuningBackend NativeCodegenBackend => KernelTuningBackend.Vulkan;

    /// <inheritdoc/>
    public CodegenEmitResult EmitCodegenKernel(CodegenGraph graph, CodegenElementType dtype)
    {
        if (graph is null) return CodegenEmitResult.Decline("Graph is null.");
        int storageBufferCount = checked(graph.InputNodes.Count + graph.OutputNodes.Count);
        uint storageBufferLimit = Math.Min(
            _device.Limits.maxPerStageDescriptorStorageBuffers,
            _device.Limits.maxDescriptorSetStorageBuffers);
        if ((uint)storageBufferCount > storageBufferLimit)
        {
            return CodegenEmitResult.Decline(
                $"Vulkan graph needs {storageBufferCount} storage buffers; device limit is {storageBufferLimit}.");
        }
        if (_device.Limits.maxPushConstantsSize < sizeof(uint))
            return CodegenEmitResult.Decline("Vulkan device cannot bind the generated element-count push constant.");

        uint limit = Math.Min(
            _device.Limits.maxComputeWorkGroupInvocations,
            _device.Limits.maxComputeWorkGroupSizeX);
        if (limit == 0)
            return CodegenEmitResult.Decline("Vulkan device reports no usable compute workgroup width.");
        var emitter = new GlslEmitter
        {
            WorkgroupSize = checked((int)Math.Min((uint)VulkanKernels.WorkgroupSize, limit))
        };
        return emitter.Emit(graph, dtype);
    }

    /// <inheritdoc/>
    public bool CanExecuteCodegenKernel(CodegenKernel kernel) =>
        kernel is GpuSourceKernel sourceKernel &&
        sourceKernel.Target == CodegenTarget.Glsl &&
        sourceKernel.Dtype == CodegenElementType.Float32 &&
        sourceKernel.DeclaredWorkgroupSize is int workgroupSize &&
        workgroupSize <= _device.Limits.maxComputeWorkGroupInvocations &&
        workgroupSize <= _device.Limits.maxComputeWorkGroupSizeX;

    /// <inheritdoc/>
    public ValueTask ExecuteCodegenKernelAsync(
        CodegenKernel kernel,
        IReadOnlyList<IGpuBuffer> inputs,
        IReadOnlyList<IGpuBuffer> outputs,
        int elementCount,
        CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        if (kernel is not GpuSourceKernel sourceKernel ||
            sourceKernel.Target != CodegenTarget.Glsl)
        {
            throw new ArgumentException("The kernel must be emitted for Vulkan GLSL.", nameof(kernel));
        }
        if (sourceKernel.Dtype != CodegenElementType.Float32)
            throw new NotSupportedException("Vulkan fused codegen execution currently requires Float32 buffers.");

        IGpuBuffer[] buffers = ValidateCodegenBuffers(sourceKernel, inputs, outputs, elementCount);
        int workgroupSize = sourceKernel.DeclaredWorkgroupSize ??
            throw new InvalidOperationException("Generated Vulkan GLSL has no declared workgroup size.");
        if (!CanExecuteCodegenKernel(sourceKernel))
            throw new NotSupportedException("Generated GLSL exceeds this Vulkan device's workgroup limits.");
        long workgroupCount = ((long)elementCount + workgroupSize - 1L) / workgroupSize;
        if ((ulong)workgroupCount > _device.Limits.maxComputeWorkGroupCountX)
        {
            throw new NotSupportedException(
                "Generated GLSL requires more X-dimension workgroups than this Vulkan device supports.");
        }
        GlslNaryOp(
            sourceKernel.Source,
            buffers,
            elementCount,
            new[] { checked((uint)elementCount) },
            workgroupSize);
        return default;
    }

    /// <summary>
    /// Compiles or restores a generated SPIR-V pipeline, executes one fused dispatch against
    /// resident buffers, and reports its synchronized host-observed launch duration. The ordinary
    /// execution path remains free of timing calls.
    /// </summary>
    public VulkanFusedKernelExecutionEvidence ExecuteFusedPointwiseProfiled(
        GpuSourceKernel sourceKernel,
        IReadOnlyList<IGpuBuffer> inputs,
        IReadOnlyList<IGpuBuffer> outputs,
        int elementCount)
    {
        EnsureInitialized();
        if (sourceKernel is null) throw new ArgumentNullException(nameof(sourceKernel));
        if (sourceKernel.Target != CodegenTarget.Glsl)
            throw new ArgumentException("The source kernel must target Vulkan GLSL.", nameof(sourceKernel));
        if (sourceKernel.Dtype != CodegenElementType.Float32)
            throw new NotSupportedException("Vulkan fused codegen execution currently requires Float32 buffers.");

        CodegenTargetRuntime runtime = sourceKernel.RequiredRuntime;
        if (runtime.Backend != KernelTuningBackend.Vulkan ||
            runtime.Compilation != CodegenNativeCompilationKind.ShadercToVulkanPipeline ||
            runtime.NativeArtifact != CodegenNativeArtifactKind.VulkanComputePipeline)
        {
            throw new InvalidOperationException(
                "The GLSL source target is not mapped to the Vulkan SPIR-V runtime contract.");
        }

        IGpuBuffer[] buffers = ValidateCodegenBuffers(sourceKernel, inputs, outputs, elementCount);
        int workgroupSize = sourceKernel.DeclaredWorkgroupSize ??
            throw new InvalidOperationException("Generated Vulkan GLSL has no declared workgroup size.");
        if (!CanExecuteCodegenKernel(sourceKernel))
            throw new NotSupportedException("Generated GLSL exceeds this Vulkan device's workgroup limits.");

        VulkanGlslCompiler compiler = _glslCompiler ??
            throw new NotSupportedException("The Vulkan GLSL compiler is unavailable.");
        uint[] spirv = compiler.CompileToSpirv(sourceKernel.Source, sourceKernel.EntryPoint) ??
            throw new InvalidOperationException("Generated GLSL did not compile to SPIR-V.");
        VulkanComputePipeline pipeline = GetOrCreateGlslPipeline(
            sourceKernel.Source, buffers.Length, sizeof(uint)) ??
            throw new InvalidOperationException("Generated SPIR-V did not create a Vulkan compute pipeline.");
        if (!pipeline.IsValid)
            throw new InvalidOperationException("The generated Vulkan compute pipeline is not valid.");

        var stopwatch = Stopwatch.StartNew();
        GlslNaryOp(
            sourceKernel.Source,
            buffers,
            elementCount,
            new[] { checked((uint)elementCount) },
            workgroupSize);
        stopwatch.Stop();
        if (stopwatch.Elapsed <= TimeSpan.Zero)
            throw new InvalidOperationException("Vulkan returned a non-positive synchronized launch duration.");

        return new VulkanFusedKernelExecutionEvidence(
            runtime.Backend,
            runtime.Compilation,
            runtime.NativeArtifact,
            DeviceName,
            DeviceVendor,
            checked((long)spirv.Length * sizeof(uint)),
            stopwatch.Elapsed,
            NativeLaunchCount: 1,
            LocalWorkgroupSize: workgroupSize);
    }

    private static IGpuBuffer[] ValidateCodegenBuffers(
        GpuSourceKernel sourceKernel,
        IReadOnlyList<IGpuBuffer> inputs,
        IReadOnlyList<IGpuBuffer> outputs,
        int elementCount)
    {
        if (inputs is null) throw new ArgumentNullException(nameof(inputs));
        if (outputs is null) throw new ArgumentNullException(nameof(outputs));
        if (inputs.Count != sourceKernel.InputCount)
            throw new ArgumentException("Input buffer count does not match the fused graph.", nameof(inputs));
        if (outputs.Count != sourceKernel.OutputCount)
            throw new ArgumentException("Output buffer count does not match the fused graph.", nameof(outputs));
        GpuEmitterCommon.ValidateLaunchElementCount(sourceKernel, elementCount);

        var buffers = new IGpuBuffer[checked(inputs.Count + outputs.Count)];
        for (int i = 0; i < inputs.Count; i++)
            buffers[i] = ValidateCodegenBuffer(inputs[i], elementCount, nameof(inputs));
        for (int i = 0; i < outputs.Count; i++)
            buffers[inputs.Count + i] = ValidateCodegenBuffer(outputs[i], elementCount, nameof(outputs));
        return buffers;
    }

    private static IGpuBuffer ValidateCodegenBuffer(
        IGpuBuffer buffer,
        int elementCount,
        string parameterName)
    {
        if (buffer is not VulkanGpuBuffer vulkanBuffer)
            throw new ArgumentException("Every buffer must belong to the Vulkan backend.", parameterName);
        if (vulkanBuffer.Size < elementCount)
            throw new ArgumentException("A fused pointwise buffer is smaller than elementCount.", parameterName);
        return vulkanBuffer;
    }
}
