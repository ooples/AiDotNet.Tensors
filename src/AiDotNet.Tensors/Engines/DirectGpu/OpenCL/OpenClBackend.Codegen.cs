// Copyright (c) AiDotNet. All rights reserved.

using AiDotNet.Evolution;
using AiDotNet.Tensors.Engines.Compilation.Codegen;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ir;
using AiDotNet.Tensors.Engines.Compilation.Codegen.OpenCl;
using AiDotNet.Tensors.Helpers.Autotune;

namespace AiDotNet.Tensors.Engines.DirectGpu.OpenCL;

/// <summary>Evidence returned by a device-profiled fused OpenCL codegen launch.</summary>
public readonly record struct OpenClFusedKernelExecutionEvidence(
    KernelTuningBackend Backend,
    CodegenNativeCompilationKind Compilation,
    CodegenNativeArtifactKind NativeArtifact,
    KernelTuningDeviceFingerprint Device,
    long NativeArtifactSizeBytes,
    OpenClProgramBuildOrigin ProgramOrigin,
    OpenClProgramBinaryType ProgramBinaryType,
    TimeSpan DeviceDuration,
    int NativeLaunchCount,
    int LocalWorkgroupSize);

public sealed partial class OpenClBackend
{
    private readonly Dictionary<string, CompiledOpenClCodegenKernel> _compiledCodegenKernels = new();
    private readonly object _compiledCodegenKernelLock = new();

    /// <inheritdoc/>
    public KernelTuningBackend NativeCodegenBackend => KernelTuningBackend.OpenCl;

    /// <inheritdoc/>
    public CodegenEmitResult EmitCodegenKernel(CodegenGraph graph, CodegenElementType dtype)
    {
        if (dtype != CodegenElementType.Float32)
            return CodegenEmitResult.Decline("OpenCL backend buffers currently store Float32 elements.");
        return new OpenClEmitter().Emit(graph, dtype);
    }

    /// <inheritdoc/>
    public bool CanExecuteCodegenKernel(CodegenKernel kernel) =>
        kernel is GpuSourceKernel &&
        kernel.Target == CodegenTarget.OpenCl &&
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
        if (kernel is not GpuSourceKernel sourceKernel ||
            sourceKernel.Target != CodegenTarget.OpenCl)
        {
            throw new ArgumentException("The kernel must be emitted for OpenCL.", nameof(kernel));
        }

        ExecuteFusedPointwise(sourceKernel, inputs, outputs, elementCount);
        return default;
    }

    /// <summary>
    /// Compiles (or restores) and executes one fused pointwise graph through the selected OpenCL
    /// driver's native code pipeline. One graph is always one device launch.
    /// </summary>
    public void ExecuteFusedPointwise(
        GpuSourceKernel sourceKernel,
        IReadOnlyList<IGpuBuffer> inputs,
        IReadOnlyList<IGpuBuffer> outputs,
        int elementCount)
    {
        CompiledOpenClCodegenKernel compiled = PrepareFusedPointwiseExecution(
            sourceKernel, inputs, outputs, elementCount);
        StageFusedPointwiseArguments(compiled.Kernel, inputs, outputs, elementCount);
        compiled.Kernel.Execute1D(elementCount, SelectFusedWorkgroupSize(compiled.Kernel, elementCount));
    }

    /// <summary>
    /// Executes the same production fused path on the dedicated profiling queue and reports the
    /// device's own start/end timestamps. This method is for offline tuning and performance proof;
    /// ordinary serving uses <see cref="ExecuteFusedPointwise"/> and pays no profiling cost.
    /// </summary>
    public OpenClFusedKernelExecutionEvidence ExecuteFusedPointwiseProfiled(
        GpuSourceKernel sourceKernel,
        IReadOnlyList<IGpuBuffer> inputs,
        IReadOnlyList<IGpuBuffer> outputs,
        int elementCount)
    {
        if (_context is null || !_context.IsProfilingEnabled)
            throw new NotSupportedException("The selected OpenCL device has no profiling queue.");

        CompiledOpenClCodegenKernel compiled = PrepareFusedPointwiseExecution(
            sourceKernel, inputs, outputs, elementCount);
        StageFusedPointwiseArguments(compiled.Kernel, inputs, outputs, elementCount);
        int localSize = SelectFusedWorkgroupSize(compiled.Kernel, elementCount);
        IntPtr kernelEvent = compiled.Kernel.Execute1DProfiled(elementCount, localSize);
        if (kernelEvent == IntPtr.Zero)
            throw new InvalidOperationException("The OpenCL driver did not return a profiling event.");

        try
        {
            int waitError = OpenClNativeBindings.WaitForEvents(1, new[] { kernelEvent });
            if (waitError != OpenClNativeBindings.CL_SUCCESS)
                throw new InvalidOperationException($"clWaitForEvents failed: {waitError}");
            ulong start = OpenClNativeBindings.GetEventProfilingInfoULong(
                kernelEvent, OpenClNativeBindings.CL_PROFILING_COMMAND_START);
            ulong end = OpenClNativeBindings.GetEventProfilingInfoULong(
                kernelEvent, OpenClNativeBindings.CL_PROFILING_COMMAND_END);
            if (end <= start)
                throw new InvalidOperationException("The OpenCL driver returned a non-positive device duration.");
            ulong nanoseconds = end - start;
            long ticks = checked((long)Math.Max(1UL, nanoseconds / 100UL));
            CodegenTargetRuntime runtime = sourceKernel.RequiredRuntime;
            return new OpenClFusedKernelExecutionEvidence(
                runtime.Backend,
                runtime.Compilation,
                runtime.NativeArtifact,
                CreateOpenClTuningFingerprint(),
                compiled.NativeArtifactSizeBytes,
                compiled.ProgramOrigin,
                compiled.ProgramBinaryType,
                TimeSpan.FromTicks(ticks),
                NativeLaunchCount: 1,
                LocalWorkgroupSize: localSize);
        }
        finally
        {
            OpenClNativeBindings.ReleaseEvent(kernelEvent);
        }
    }

    private CompiledOpenClCodegenKernel PrepareFusedPointwiseExecution(
        GpuSourceKernel sourceKernel,
        IReadOnlyList<IGpuBuffer> inputs,
        IReadOnlyList<IGpuBuffer> outputs,
        int elementCount)
    {
        if (_disposed) throw new ObjectDisposedException(nameof(OpenClBackend));
        if (_context is null || !IsAvailable)
            throw new InvalidOperationException("OpenCL context not available.");
        if (sourceKernel is null) throw new ArgumentNullException(nameof(sourceKernel));
        if (inputs is null) throw new ArgumentNullException(nameof(inputs));
        if (outputs is null) throw new ArgumentNullException(nameof(outputs));
        if (sourceKernel.Target != CodegenTarget.OpenCl)
            throw new ArgumentException("The source kernel must target OpenCL.", nameof(sourceKernel));
        CodegenTargetRuntime runtime = sourceKernel.RequiredRuntime;
        if (runtime.Backend != KernelTuningBackend.OpenCl ||
            runtime.Compilation != CodegenNativeCompilationKind.OpenClDriverCompiler ||
            runtime.NativeArtifact != CodegenNativeArtifactKind.OpenClDeviceBinary)
        {
            throw new InvalidOperationException(
                "The OpenCL source target is not mapped to the OpenCL native runtime contract.");
        }
        if (sourceKernel.Dtype != Compilation.Codegen.Ir.CodegenElementType.Float32)
            throw new NotSupportedException("OpenCL fused backend execution currently requires Float32 buffers.");
        if (inputs.Count != sourceKernel.InputCount)
            throw new ArgumentException("Input buffer count does not match the fused graph.", nameof(inputs));
        if (outputs.Count != sourceKernel.OutputCount)
            throw new ArgumentException("Output buffer count does not match the fused graph.", nameof(outputs));
        GpuEmitterCommon.ValidateLaunchElementCount(sourceKernel, elementCount);

        ValidateFusedBuffers(inputs, elementCount, nameof(inputs));
        ValidateFusedBuffers(outputs, elementCount, nameof(outputs));
        return GetOrCompileCodegenKernel(sourceKernel);
    }

    private void ValidateFusedBuffers(
        IReadOnlyList<IGpuBuffer> buffers,
        int elementCount,
        string parameterName)
    {
        DirectOpenClContext context = _context ??
            throw new InvalidOperationException("OpenCL context not available.");
        for (int i = 0; i < buffers.Count; i++)
        {
            if (buffers[i] is not DirectOpenClGpuBuffer buffer ||
                !ReferenceEquals(buffer.Buffer.OwningContext, context))
                throw new ArgumentException("Every buffer must belong to this OpenCL backend.", parameterName);
            if (buffer.Size < elementCount)
                throw new ArgumentException("A fused pointwise buffer is smaller than elementCount.", parameterName);
        }
    }

    private static void StageFusedPointwiseArguments(
        DirectOpenClKernel kernel,
        IReadOnlyList<IGpuBuffer> inputs,
        IReadOnlyList<IGpuBuffer> outputs,
        int elementCount)
    {
        uint argument = 0;
        for (int i = 0; i < inputs.Count; i++)
            kernel.SetArg(argument++, ((DirectOpenClGpuBuffer)inputs[i]).Buffer.Handle);
        for (int i = 0; i < outputs.Count; i++)
            kernel.SetArg(argument++, ((DirectOpenClGpuBuffer)outputs[i]).Buffer.Handle);
        kernel.SetArg(argument, elementCount);
    }

    private CompiledOpenClCodegenKernel GetOrCompileCodegenKernel(GpuSourceKernel sourceKernel)
    {
        DirectOpenClContext context = _context ??
            throw new InvalidOperationException("OpenCL context not available.");
        string key = EvolutionHash.Combine(new[]
        {
            "opencl-codegen-v1",
            sourceKernel.EntryPoint,
            ((int)sourceKernel.Dtype).ToString(System.Globalization.CultureInfo.InvariantCulture),
            sourceKernel.Source
        });
        lock (_compiledCodegenKernelLock)
        {
            if (_compiledCodegenKernels.TryGetValue(key, out CompiledOpenClCodegenKernel? cached))
                return cached;

            DirectOpenClProgram? program = DirectOpenClProgram.TryCreateFromCache(
                context, sourceKernel.Source, OpenClBuildOptions.OptimizationFlags);
            if (program is null)
            {
                program = new DirectOpenClProgram(context, sourceKernel.Source);
                try
                {
                    program.Build(OpenClBuildOptions.OptimizationFlags);
                }
                catch
                {
                    program.Dispose();
                    throw;
                }
            }

            try
            {
                var compiled = new CompiledOpenClCodegenKernel(
                    program,
                    new DirectOpenClKernel(context, program, sourceKernel.EntryPoint));
                _compiledCodegenKernels.Add(key, compiled);
                return compiled;
            }
            catch
            {
                program.Dispose();
                throw;
            }
        }
    }

    private int SelectFusedWorkgroupSize(DirectOpenClKernel kernel, int elementCount)
    {
        int localSize = ClampLocalSizeForKernel(
            kernel,
            CalculateOptimalWorkGroupSize1D(elementCount),
            localElementBytes: 0);
        DirectOpenClContext context = _context ??
            throw new InvalidOperationException("OpenCL context not available.");
        UIntPtr preferredResult = OpenClNativeBindings.GetKernelWorkGroupInfoSizeT(
            kernel.Handle,
            context.Device,
            OpenClNativeBindings.CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE);
        ulong preferredValue = preferredResult.ToUInt64();
        if (preferredValue > 0 && preferredValue <= int.MaxValue)
        {
            int preferred = checked((int)preferredValue);
            if (localSize >= preferred)
                localSize = Math.Max(preferred, localSize / preferred * preferred);
        }

        return localSize;
    }

    private void DisposeCompiledCodegenKernels()
    {
        lock (_compiledCodegenKernelLock)
        {
            foreach (CompiledOpenClCodegenKernel compiled in _compiledCodegenKernels.Values)
                compiled.Dispose();
            _compiledCodegenKernels.Clear();
        }
    }

    private sealed class CompiledOpenClCodegenKernel : IDisposable
    {
        internal CompiledOpenClCodegenKernel(
            DirectOpenClProgram program,
            DirectOpenClKernel kernel)
        {
            Program = program;
            Kernel = kernel;
        }

        internal DirectOpenClProgram Program { get; }
        internal DirectOpenClKernel Kernel { get; }
        internal OpenClProgramBuildOrigin ProgramOrigin => Program.BuildOrigin;
        internal OpenClProgramBinaryType ProgramBinaryType => Program.BinaryType;
        internal long NativeArtifactSizeBytes => Program.NativeBinarySizeBytes;

        public void Dispose()
        {
            Kernel.Dispose();
            Program.Dispose();
        }
    }
}
