// Copyright (c) AiDotNet. All rights reserved.

#if NET7_0_OR_GREATER
using System.Security.Cryptography;
using System.Text;
using AiDotNet.Tensors.Engines.Compilation.Codegen;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ir;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Wgsl;
using AiDotNet.Tensors.Helpers.Autotune;

namespace AiDotNet.Tensors.Engines.DirectGpu.WebGpu;

public sealed partial class WebGpuBackend
{
    /// <inheritdoc/>
    public KernelTuningBackend NativeCodegenBackend => KernelTuningBackend.WebGpu;

    /// <inheritdoc/>
    public CodegenEmitResult EmitCodegenKernel(CodegenGraph graph, CodegenElementType dtype)
    {
        if (graph is null) return CodegenEmitResult.Decline("Graph is null.");
        int storageBufferCount = checked(graph.InputNodes.Count + graph.OutputNodes.Count);
        WebGpuDeviceLimits? limits = _device.Limits;
        int storageBufferLimit = limits?.MaxStorageBuffersPerShaderStage ?? 8;
        int bindingLimit = limits?.MaxBindingsPerBindGroup ?? 1000;
        if (storageBufferCount > storageBufferLimit)
        {
            return CodegenEmitResult.Decline(
                $"WebGPU graph needs {storageBufferCount} storage buffers; device limit is {storageBufferLimit}.");
        }
        int totalBindingCount = checked(storageBufferCount + 1);
        if (totalBindingCount > bindingLimit)
        {
            return CodegenEmitResult.Decline(
                $"WebGPU graph needs {totalBindingCount} bindings; bind-group limit is {bindingLimit}.");
        }

        int xLimit = _device.Limits?.MaxComputeWorkgroupSizeX ?? _device.MaxWorkgroupSize;
        int limit = Math.Min(_device.MaxWorkgroupSize, xLimit);
        if (limit <= 0)
            return CodegenEmitResult.Decline("WebGPU device reports no usable compute workgroup width.");
        var emitter = new WgslEmitter { WorkgroupSize = Math.Min(256, limit) };
        return emitter.Emit(graph, dtype);
    }

    /// <inheritdoc/>
    public bool CanExecuteCodegenKernel(CodegenKernel kernel) =>
        kernel is GpuSourceKernel sourceKernel &&
        sourceKernel.Target == CodegenTarget.Wgsl &&
        sourceKernel.Dtype == CodegenElementType.Float32 &&
        sourceKernel.DeclaredWorkgroupSize is int workgroupSize &&
        workgroupSize <= _device.MaxWorkgroupSize &&
        workgroupSize <= (_device.Limits?.MaxComputeWorkgroupSizeX ?? _device.MaxWorkgroupSize);

    /// <inheritdoc/>
    public async ValueTask ExecuteCodegenKernelAsync(
        CodegenKernel kernel,
        IReadOnlyList<IGpuBuffer> inputs,
        IReadOnlyList<IGpuBuffer> outputs,
        int elementCount,
        CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        ThrowIfNotInitialized();
        if (kernel is not GpuSourceKernel sourceKernel ||
            sourceKernel.Target != CodegenTarget.Wgsl)
        {
            throw new ArgumentException("The kernel must be emitted for WebGPU WGSL.", nameof(kernel));
        }
        if (sourceKernel.Dtype != CodegenElementType.Float32)
            throw new NotSupportedException("WebGPU fused codegen execution currently requires Float32 buffers.");

        WebGpuBuffer[] buffers = ValidateCodegenBuffers(sourceKernel, inputs, outputs, elementCount);
        string moduleName = CreateCodegenModuleName(sourceKernel);
        int pipelineId = await GetOrCreatePipelineAsync(
            moduleName, sourceKernel.Source, sourceKernel.EntryPoint).ConfigureAwait(false);

        // WebGPU uniform bindings have a 16-byte minimum practical alignment across browsers.
        var parameters = new float[]
        {
            BitConverter.Int32BitsToSingle(elementCount), 0, 0, 0
        };
        using var uniformBuffer = new WebGpuBuffer(
            parameters, WebGpuBufferUsage.Uniform | WebGpuBufferUsage.CopyDst);
        using var bindGroup = new WebGpuBindGroup(pipelineId, buffers);
        int workgroupSize = sourceKernel.DeclaredWorkgroupSize ??
            throw new InvalidOperationException("Generated WGSL has no declared workgroup size.");
        if (!CanExecuteCodegenKernel(sourceKernel))
            throw new NotSupportedException("Generated WGSL exceeds this WebGPU device's workgroup limits.");
        int workgroups = checked((int)(((long)elementCount + workgroupSize - 1L) / workgroupSize));
        if (workgroups > _device.MaxWorkgroupsPerDimension)
        {
            throw new NotSupportedException(
                "Generated WGSL requires more X-dimension workgroups than this WebGPU device supports.");
        }
        await WebGpuNativeBindings.DispatchComputeWithUniformsAsync(
            pipelineId, bindGroup.BindGroupId, uniformBuffer.BufferId,
            workgroups, 1, 1).ConfigureAwait(false);
        await WebGpuNativeBindings.SubmitAndWaitAsync().ConfigureAwait(false);
    }

    private static WebGpuBuffer[] ValidateCodegenBuffers(
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

        var buffers = new WebGpuBuffer[checked(inputs.Count + outputs.Count)];
        for (int i = 0; i < inputs.Count; i++)
            buffers[i] = ValidateCodegenBuffer(inputs[i], elementCount, nameof(inputs));
        for (int i = 0; i < outputs.Count; i++)
            buffers[inputs.Count + i] = ValidateCodegenBuffer(outputs[i], elementCount, nameof(outputs));
        return buffers;
    }

    private static WebGpuBuffer ValidateCodegenBuffer(
        IGpuBuffer buffer,
        int elementCount,
        string parameterName)
    {
        if (buffer is not WebGpuBuffer webGpuBuffer || !webGpuBuffer.IsValid)
            throw new ArgumentException("Every buffer must be a valid WebGPU buffer.", parameterName);
        if (webGpuBuffer.Size < elementCount)
            throw new ArgumentException("A fused pointwise buffer is smaller than elementCount.", parameterName);
        return webGpuBuffer;
    }

    private static string CreateCodegenModuleName(GpuSourceKernel kernel)
    {
        byte[] bytes = Encoding.UTF8.GetBytes(kernel.Source);
        byte[] digest;
        using (SHA256 sha = SHA256.Create()) digest = sha.ComputeHash(bytes);
        return "Codegen-" + BitConverter.ToString(digest).Replace("-", string.Empty);
    }
}
#endif
