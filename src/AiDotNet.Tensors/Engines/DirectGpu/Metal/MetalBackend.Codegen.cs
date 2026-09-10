// Copyright (c) AiDotNet. All rights reserved.

using System.Security.Cryptography;
using System.Text;
using AiDotNet.Tensors.Engines.Compilation.Codegen;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ir;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Msl;
using AiDotNet.Tensors.Helpers.Autotune;

namespace AiDotNet.Tensors.Engines.DirectGpu.Metal;

public sealed partial class MetalBackend
{
    /// <inheritdoc/>
    public KernelTuningBackend NativeCodegenBackend => KernelTuningBackend.Metal;

    /// <inheritdoc/>
    public CodegenEmitResult EmitCodegenKernel(CodegenGraph graph, CodegenElementType dtype)
    {
        if (graph is null) return CodegenEmitResult.Decline("Graph is null.");
        int storageBufferCount = checked(graph.InputNodes.Count + graph.OutputNodes.Count);
        const int MaximumStorageBuffers = 30;
        if (storageBufferCount > MaximumStorageBuffers)
        {
            return CodegenEmitResult.Decline(
                $"Metal graph needs {storageBufferCount} storage buffers; " +
                $"one element-count binding leaves room for {MaximumStorageBuffers}.");
        }
        return new MslEmitter().Emit(graph, dtype);
    }

    /// <inheritdoc/>
    public bool CanExecuteCodegenKernel(CodegenKernel kernel) =>
        kernel is GpuSourceKernel &&
        kernel.Target == CodegenTarget.Msl &&
        kernel.Dtype == CodegenElementType.Float32;

    /// <inheritdoc/>
    public ValueTask ExecuteCodegenKernelAsync(
        CodegenKernel kernel,
        IReadOnlyList<IGpuBuffer> inputs,
        IReadOnlyList<IGpuBuffer> outputs,
        int elementCount,
        CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        ThrowIfDisposed();
        if (kernel is not GpuSourceKernel sourceKernel ||
            sourceKernel.Target != CodegenTarget.Msl)
        {
            throw new ArgumentException("The kernel must be emitted for Metal MSL.", nameof(kernel));
        }
        if (sourceKernel.Dtype != CodegenElementType.Float32)
            throw new NotSupportedException("Metal fused codegen execution currently requires Float32 buffers.");
        if (inputs is null) throw new ArgumentNullException(nameof(inputs));
        if (outputs is null) throw new ArgumentNullException(nameof(outputs));
        if (inputs.Count != sourceKernel.InputCount)
            throw new ArgumentException("Input buffer count does not match the fused graph.", nameof(inputs));
        if (outputs.Count != sourceKernel.OutputCount)
            throw new ArgumentException("Output buffer count does not match the fused graph.", nameof(outputs));
        GpuEmitterCommon.ValidateLaunchElementCount(sourceKernel, elementCount);

        string libraryName = CreateCodegenLibraryName(sourceKernel);
        MetalPipelineState pipeline = _shaderLibrary.GetOrCreatePipelineState(
            libraryName, sourceKernel.Source, sourceKernel.EntryPoint);
        var dispatch = pipeline.Calculate1DDispatch(elementCount);
        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        int binding = 0;
        for (int i = 0; i < inputs.Count; i++)
            encoder.SetBuffer(ValidateCodegenBuffer(inputs[i], elementCount, nameof(inputs)), binding++);
        for (int i = 0; i < outputs.Count; i++)
            encoder.SetBuffer(ValidateCodegenBuffer(outputs[i], elementCount, nameof(outputs)), binding++);
        encoder.SetBytes(checked((uint)elementCount), (ulong)binding);
        encoder.DispatchThreadgroups(dispatch.Threadgroups, dispatch.ThreadsPerThreadgroup);
        return default;
    }

    private MetalGpuBuffer ValidateCodegenBuffer(
        IGpuBuffer buffer,
        int elementCount,
        string parameterName)
    {
        if (buffer is not MetalGpuBuffer metalBuffer ||
            !ReferenceEquals(metalBuffer.OwningDevice, _device))
        {
            throw new ArgumentException("Every buffer must belong to this Metal backend.", parameterName);
        }
        if (metalBuffer.Size < elementCount)
            throw new ArgumentException("A fused pointwise buffer is smaller than elementCount.", parameterName);
        return metalBuffer;
    }

    private static string CreateCodegenLibraryName(GpuSourceKernel kernel)
    {
        byte[] bytes = Encoding.UTF8.GetBytes(kernel.Source);
        byte[] digest;
        using (SHA256 sha = SHA256.Create()) digest = sha.ComputeHash(bytes);
        return "Codegen-" + BitConverter.ToString(digest).Replace("-", string.Empty);
    }
}
