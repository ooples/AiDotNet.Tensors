// Copyright (c) AiDotNet. All rights reserved.

using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ir;
using AiDotNet.Tensors.Helpers.Autotune;

namespace AiDotNet.Tensors.Engines.Compilation.Codegen;

/// <summary>
/// Executes an emitted kernel through a GPU backend's native compiler and command pipeline.
/// Implementations must compile to the artifact declared by <see cref="CodegenKernel.RequiredRuntime"/>,
/// cache that executable artifact or pipeline, and submit one fused device launch.
/// </summary>
public interface INativeGpuCodegenExecutor
{
    /// <summary>The physical runtime backend owned by this executor.</summary>
    KernelTuningBackend NativeCodegenBackend { get; }

    /// <summary>
    /// Emits the representation this physical backend can lower most directly for the active
    /// device. Backend ownership is important for architecture-specialized targets such as PTX.
    /// </summary>
    CodegenEmitResult EmitCodegenKernel(CodegenGraph graph, CodegenElementType dtype);

    /// <summary>Returns whether this executor owns the emitted target's native runtime.</summary>
    bool CanExecuteCodegenKernel(CodegenKernel kernel);

    /// <summary>
    /// Compiles or restores and then executes an emitted kernel against resident device buffers.
    /// </summary>
    ValueTask ExecuteCodegenKernelAsync(
        CodegenKernel kernel,
        IReadOnlyList<IGpuBuffer> inputs,
        IReadOnlyList<IGpuBuffer> outputs,
        int elementCount,
        CancellationToken cancellationToken = default);
}

/// <summary>Safe orchestration helpers for backend-owned native code generation.</summary>
public static class NativeGpuCodegenExecutorExtensions
{
    /// <summary>
    /// Emits for the active physical backend, validates the typed runtime contract, and submits
    /// the resulting fused kernel. A declined graph is returned without executing a fallback.
    /// </summary>
    public static async ValueTask<CodegenEmitResult> EmitAndExecuteCodegenKernelAsync(
        this INativeGpuCodegenExecutor executor,
        CodegenGraph graph,
        CodegenElementType dtype,
        IReadOnlyList<IGpuBuffer> inputs,
        IReadOnlyList<IGpuBuffer> outputs,
        int elementCount,
        CancellationToken cancellationToken = default)
    {
        if (executor is null) throw new ArgumentNullException(nameof(executor));
        if (graph is null) throw new ArgumentNullException(nameof(graph));
        if (inputs is null) throw new ArgumentNullException(nameof(inputs));
        if (outputs is null) throw new ArgumentNullException(nameof(outputs));

        CodegenEmitResult result = executor.EmitCodegenKernel(graph, dtype);
        if (result.Declined) return result;
        CodegenKernel emittedKernel = result.Kernel ??
            throw new InvalidOperationException("A successful GPU emission returned no kernel.");
        if (emittedKernel.RequiredRuntime.Backend != executor.NativeCodegenBackend ||
            !executor.CanExecuteCodegenKernel(emittedKernel))
        {
            throw new InvalidOperationException(
                "The physical backend cannot execute the native target it emitted.");
        }

        await executor.ExecuteCodegenKernelAsync(
            emittedKernel, inputs, outputs, elementCount, cancellationToken).ConfigureAwait(false);
        return result;
    }
}
