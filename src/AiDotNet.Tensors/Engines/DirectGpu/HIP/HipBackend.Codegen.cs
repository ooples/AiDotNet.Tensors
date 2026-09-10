// Copyright (c) AiDotNet. All rights reserved.

using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text;
using AiDotNet.Tensors.Engines.Compilation.Codegen;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Hip;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ir;
using AiDotNet.Tensors.Helpers.Autotune;

namespace AiDotNet.Tensors.Engines.DirectGpu.HIP;

/// <summary>Evidence returned by a device-profiled fused HIP code-object launch.</summary>
public readonly record struct HipFusedKernelExecutionEvidence(
    KernelTuningBackend Backend,
    CodegenNativeCompilationKind Compilation,
    CodegenNativeArtifactKind NativeArtifact,
    AmdGpuArchitecture Architecture,
    string DeviceName,
    long NativeArtifactSizeBytes,
    TimeSpan DeviceDuration,
    int NativeLaunchCount,
    int BlockSize);

public sealed partial class HipBackend
{
    private readonly Dictionary<string, CompiledHipCodegenKernel> _compiledCodegenKernels =
        new(StringComparer.Ordinal);
    private readonly object _compiledCodegenKernelLock = new();

    /// <inheritdoc/>
    public KernelTuningBackend NativeCodegenBackend => KernelTuningBackend.Hip;

    /// <inheritdoc/>
    public CodegenEmitResult EmitCodegenKernel(CodegenGraph graph, CodegenElementType dtype)
    {
        if (dtype != CodegenElementType.Float32)
            return CodegenEmitResult.Decline("HIP backend buffers currently store Float32 elements.");
        return new HipEmitter().Emit(graph, dtype);
    }

    /// <summary>Size of the most recently selected HIPRTC code object.</summary>
    public long LastCodegenNativeArtifactSizeBytes { get; private set; }

    /// <inheritdoc/>
    public bool CanExecuteCodegenKernel(CodegenKernel kernel) =>
        kernel is GpuSourceKernel &&
        kernel.Target == CodegenTarget.Hip &&
        kernel.Dtype == CodegenElementType.Float32;

    /// <inheritdoc/>
    public unsafe ValueTask ExecuteCodegenKernelAsync(
        CodegenKernel kernel,
        IReadOnlyList<IGpuBuffer> inputs,
        IReadOnlyList<IGpuBuffer> outputs,
        int elementCount,
        CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        if (_disposed) throw new ObjectDisposedException(nameof(HipBackend));
        if (!IsAvailable) throw new InvalidOperationException("HIP is not available.");
        if (kernel is not GpuSourceKernel sourceKernel ||
            sourceKernel.Target != CodegenTarget.Hip)
        {
            throw new ArgumentException("The kernel must be emitted for HIP.", nameof(kernel));
        }
        if (sourceKernel.Dtype != CodegenElementType.Float32)
            throw new NotSupportedException("HIP fused codegen execution currently requires Float32 buffers.");

        HipGpuBuffer[] buffers = ValidateCodegenBuffers(sourceKernel, inputs, outputs, elementCount);
        CompiledHipCodegenKernel compiled = GetOrCompileCodegenKernel(sourceKernel);
        LaunchCodegenKernel(compiled, buffers, elementCount);
        LastCodegenNativeArtifactSizeBytes = compiled.NativeArtifactSizeBytes;
        return default;
    }

    /// <summary>
    /// Executes a fused generated kernel and measures its device time with HIP events. The event
    /// objects are used only for offline proof and tuning; the ordinary execution path remains
    /// free of profiling synchronization.
    /// </summary>
    public HipFusedKernelExecutionEvidence ExecuteFusedPointwiseProfiled(
        GpuSourceKernel sourceKernel,
        IReadOnlyList<IGpuBuffer> inputs,
        IReadOnlyList<IGpuBuffer> outputs,
        int elementCount)
    {
        if (sourceKernel is null) throw new ArgumentNullException(nameof(sourceKernel));
        if (sourceKernel.Target != CodegenTarget.Hip)
            throw new ArgumentException("The source kernel must target HIP.", nameof(sourceKernel));
        if (_disposed) throw new ObjectDisposedException(nameof(HipBackend));
        if (!IsAvailable) throw new InvalidOperationException("HIP is not available.");

        CodegenTargetRuntime runtime = sourceKernel.RequiredRuntime;
        if (runtime.Backend != KernelTuningBackend.Hip ||
            runtime.Compilation != CodegenNativeCompilationKind.HipRtc ||
            runtime.NativeArtifact != CodegenNativeArtifactKind.HipCodeObject)
        {
            throw new InvalidOperationException(
                "The HIP source target is not mapped to the AMD code-object runtime contract.");
        }

        HipGpuBuffer[] buffers = ValidateCodegenBuffers(sourceKernel, inputs, outputs, elementCount);
        CompiledHipCodegenKernel compiled = GetOrCompileCodegenKernel(sourceKernel);
        IntPtr start = IntPtr.Zero;
        IntPtr stop = IntPtr.Zero;
        try
        {
            HipNativeBindings.CheckError(
                HipNativeBindings.hipEventCreate(ref start), "hipEventCreate(codegen start)");
            HipNativeBindings.CheckError(
                HipNativeBindings.hipEventCreate(ref stop), "hipEventCreate(codegen stop)");
            HipNativeBindings.CheckError(
                HipNativeBindings.hipEventRecord(start, _stream), "hipEventRecord(codegen start)");
            int blockSize = checked((int)LaunchCodegenKernel(compiled, buffers, elementCount));
            HipNativeBindings.CheckError(
                HipNativeBindings.hipEventRecord(stop, _stream), "hipEventRecord(codegen stop)");
            HipNativeBindings.CheckError(
                HipNativeBindings.hipEventSynchronize(stop), "hipEventSynchronize(codegen stop)");
            float milliseconds = 0;
            HipNativeBindings.CheckError(
                HipNativeBindings.hipEventElapsedTime(ref milliseconds, start, stop),
                "hipEventElapsedTime(codegen)");
            if (float.IsNaN(milliseconds) || float.IsInfinity(milliseconds) || milliseconds <= 0)
                throw new InvalidOperationException("HIP returned a non-positive generated-kernel duration.");

            LastCodegenNativeArtifactSizeBytes = compiled.NativeArtifactSizeBytes;
            long ticks = Math.Max(1L, checked((long)Math.Round(
                milliseconds * TimeSpan.TicksPerMillisecond,
                MidpointRounding.AwayFromZero)));
            return new HipFusedKernelExecutionEvidence(
                runtime.Backend,
                runtime.Compilation,
                runtime.NativeArtifact,
                _architecture,
                DeviceName,
                compiled.NativeArtifactSizeBytes,
                TimeSpan.FromTicks(ticks),
                NativeLaunchCount: 1,
                BlockSize: blockSize);
        }
        finally
        {
            if (stop != IntPtr.Zero) HipNativeBindings.hipEventDestroy(stop);
            if (start != IntPtr.Zero) HipNativeBindings.hipEventDestroy(start);
        }
    }

    private unsafe uint LaunchCodegenKernel(
        CompiledHipCodegenKernel compiled,
        IReadOnlyList<HipGpuBuffer> buffers,
        int elementCount)
    {
        int argumentCount = checked(buffers.Count + 1);
        IntPtr* bufferHandles = stackalloc IntPtr[buffers.Count];
        void** arguments = stackalloc void*[argumentCount];
        for (int i = 0; i < buffers.Count; i++)
        {
            bufferHandles[i] = buffers[i].Handle;
            arguments[i] = &bufferHandles[i];
        }
        int count = elementCount;
        arguments[buffers.Count] = &count;
        uint blockSize = checked((uint)Math.Min(256, _deviceProps.MaxThreadsPerBlock));
        if (blockSize == 0)
            throw new InvalidOperationException("HIP device reports no usable compute block width.");
        uint gridSize = checked((uint)(((long)elementCount + blockSize - 1) / blockSize));
        LaunchKernel(compiled.Function, gridSize, blockSize, arguments);
        return blockSize;
    }

    private HipGpuBuffer[] ValidateCodegenBuffers(
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

        var buffers = new HipGpuBuffer[checked(inputs.Count + outputs.Count)];
        for (int i = 0; i < inputs.Count; i++)
            buffers[i] = ValidateCodegenBuffer(inputs[i], elementCount, nameof(inputs));
        for (int i = 0; i < outputs.Count; i++)
            buffers[inputs.Count + i] = ValidateCodegenBuffer(outputs[i], elementCount, nameof(outputs));
        return buffers;
    }

    private HipGpuBuffer ValidateCodegenBuffer(
        IGpuBuffer buffer,
        int elementCount,
        string parameterName)
    {
        if (buffer is not HipGpuBuffer hipBuffer ||
            hipBuffer.Handle == IntPtr.Zero ||
            !ReferenceEquals(hipBuffer.OwningBackend, this))
        {
            throw new ArgumentException("Every buffer must belong to this HIP backend.", parameterName);
        }
        if (hipBuffer.Size < elementCount)
            throw new ArgumentException("A fused pointwise buffer is smaller than elementCount.", parameterName);
        return hipBuffer;
    }

    private CompiledHipCodegenKernel GetOrCompileCodegenKernel(GpuSourceKernel sourceKernel)
    {
        string key = CreateCodegenKey(sourceKernel);
        lock (_compiledCodegenKernelLock)
        {
            if (_compiledCodegenKernels.TryGetValue(key, out CompiledHipCodegenKernel? cached))
                return cached;

            CompiledHipCodegenKernel compiled = CompileCodegenKernel(sourceKernel);
            _compiledCodegenKernels.Add(key, compiled);
            return compiled;
        }
    }

    private CompiledHipCodegenKernel CompileCodegenKernel(GpuSourceKernel sourceKernel)
    {
        IntPtr program = IntPtr.Zero;
        HipRtcResult rtcResult = HipNativeBindings.hiprtcCreateProgram(
            ref program, sourceKernel.Source, sourceKernel.EntryPoint,
            0, IntPtr.Zero, IntPtr.Zero);
        if (rtcResult != HipRtcResult.Success)
            throw new InvalidOperationException($"hiprtcCreateProgram failed: {rtcResult}.");

        IntPtr code = IntPtr.Zero;
        IntPtr module = IntPtr.Zero;
        try
        {
            string flags = HipMfmaKernel.GetCompileFlags(_architecture);
            var options = new List<string>(
                flags.Split(new[] { ' ' }, StringSplitOptions.RemoveEmptyEntries))
            {
                "-O3",
                "-ffast-math"
            };
            rtcResult = HipNativeBindings.hiprtcCompileProgram(
                program, options.Count, options.ToArray());
            if (rtcResult != HipRtcResult.Success)
            {
                string log = GetHipRtcLog(program);
                throw new InvalidOperationException(
                    $"HIPRTC failed to compile fused kernel: {rtcResult}. {log}");
            }

            UIntPtr codeSize = UIntPtr.Zero;
            rtcResult = HipNativeBindings.hiprtcGetCodeSize(program, ref codeSize);
            ulong nativeBytes = codeSize.ToUInt64();
            if (rtcResult != HipRtcResult.Success || nativeBytes == 0 || nativeBytes > int.MaxValue)
                throw new InvalidOperationException($"HIPRTC returned an invalid code object size: {nativeBytes}.");

            code = Marshal.AllocHGlobal(checked((int)nativeBytes));
            rtcResult = HipNativeBindings.hiprtcGetCode(program, code);
            if (rtcResult != HipRtcResult.Success)
                throw new InvalidOperationException($"hiprtcGetCode failed: {rtcResult}.");

            HipError loadResult = HipNativeBindings.hipModuleLoadData(ref module, code);
            if (loadResult != HipError.Success || module == IntPtr.Zero)
                throw new InvalidOperationException($"hipModuleLoadData failed: {loadResult}.");

            IntPtr function = IntPtr.Zero;
            HipError functionResult = HipNativeBindings.hipModuleGetFunction(
                ref function, module, sourceKernel.EntryPoint);
            if (functionResult != HipError.Success || function == IntPtr.Zero)
                throw new InvalidOperationException($"hipModuleGetFunction failed: {functionResult}.");

            GpuKernelDiagnostics.RegisterKernelName(function, sourceKernel.EntryPoint);
            var compiled = new CompiledHipCodegenKernel(
                module, function, checked((long)nativeBytes));
            module = IntPtr.Zero;
            return compiled;
        }
        finally
        {
            if (module != IntPtr.Zero) HipNativeBindings.hipModuleUnload(module);
            if (code != IntPtr.Zero) Marshal.FreeHGlobal(code);
            if (program != IntPtr.Zero) HipNativeBindings.hiprtcDestroyProgram(ref program);
        }
    }

    private static string GetHipRtcLog(IntPtr program)
    {
        UIntPtr logSize = UIntPtr.Zero;
        if (HipNativeBindings.hiprtcGetProgramLogSize(program, ref logSize) != HipRtcResult.Success ||
            logSize.ToUInt64() == 0 || logSize.ToUInt64() > int.MaxValue)
        {
            return string.Empty;
        }

        IntPtr log = Marshal.AllocHGlobal(checked((int)logSize.ToUInt64()));
        try
        {
            return HipNativeBindings.hiprtcGetProgramLog(program, log) == HipRtcResult.Success
                ? Marshal.PtrToStringAnsi(log) ?? string.Empty
                : string.Empty;
        }
        finally
        {
            Marshal.FreeHGlobal(log);
        }
    }

    private static string CreateCodegenKey(GpuSourceKernel kernel)
    {
        byte[] bytes = Encoding.UTF8.GetBytes(kernel.Source);
        byte[] digest;
        using (SHA256 sha = SHA256.Create()) digest = sha.ComputeHash(bytes);
        return BitConverter.ToString(digest).Replace("-", string.Empty);
    }

    private void DisposeCompiledCodegenKernels()
    {
        lock (_compiledCodegenKernelLock)
        {
            foreach (CompiledHipCodegenKernel compiled in _compiledCodegenKernels.Values)
                compiled.Dispose();
            _compiledCodegenKernels.Clear();
            LastCodegenNativeArtifactSizeBytes = 0;
        }
    }

    private sealed class CompiledHipCodegenKernel : IDisposable
    {
        private IntPtr _module;
        private IntPtr _function;

        internal CompiledHipCodegenKernel(IntPtr module, IntPtr function, long nativeArtifactSizeBytes)
        {
            _module = module;
            _function = function;
            NativeArtifactSizeBytes = nativeArtifactSizeBytes;
        }

        internal IntPtr Function => _function;
        internal long NativeArtifactSizeBytes { get; }

        public void Dispose()
        {
            if (_function != IntPtr.Zero)
            {
                GpuKernelDiagnostics.UnregisterKernelName(_function);
                _function = IntPtr.Zero;
            }
            if (_module != IntPtr.Zero)
            {
                HipNativeBindings.hipModuleUnload(_module);
                _module = IntPtr.Zero;
            }
        }
    }
}
