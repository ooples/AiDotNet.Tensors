// Copyright (c) CrowdTrainer LLC. All rights reserved.

using System.Security.Cryptography;
using System.Text;
using System.IO;
using AiDotNet.Tensors.Engines.Compilation.Codegen;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ir;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ptx;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;
using AiDotNet.Tensors.Helpers.Autotune;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA;

public sealed partial class CudaBackend
{
    private readonly Dictionary<string, DirectPtxCodegenKernel> _directPtxCodegenKernels =
        new(StringComparer.Ordinal);
    private readonly Dictionary<string, IntPtr> _nvrtcCodegenFunctions =
        new(StringComparer.Ordinal);

    /// <inheritdoc/>
    public KernelTuningBackend NativeCodegenBackend => KernelTuningBackend.Cuda;

    /// <inheritdoc/>
    public CodegenEmitResult EmitCodegenKernel(CodegenGraph graph, CodegenElementType dtype)
    {
        var emitter = new PtxGraphEmitter
        {
            ComputeMajor = _ccMajor,
            ComputeMinor = _ccMinor
        };
        var ptxResult = emitter.Emit(graph, dtype);
        if (!ptxResult.Declined)
        {
            return ptxResult;
        }

        // Fall back to NVRTC CUDA C emission for fused pointwise and trailing-broadcast chains
        // (SwiGLU, GeGLU, Bias+GELU/Swish, multi-op elementwise DAGs) when direct-PTX declines.
        var nvrtcEmitter = new NvrtcGraphEmitter();
        var nvrtcResult = nvrtcEmitter.Emit(graph, dtype);
        return nvrtcResult.Declined ? ptxResult : nvrtcResult;
    }

    /// <summary>
    /// Size of the cubin selected for the most recent generated launch, or zero when the CUDA
    /// driver JIT fallback does not expose its module image.
    /// </summary>
    public long LastCodegenNativeArtifactSizeBytes { get; private set; }

    /// <inheritdoc/>
    public bool CanExecuteCodegenKernel(CodegenKernel kernel) =>
        (kernel is PtxCodegenKernel ptx && ptx.ComputeMajor == _ccMajor && ptx.ComputeMinor == _ccMinor)
        || kernel is NvrtcCodegenKernel;

    /// <inheritdoc/>
    public unsafe ValueTask ExecuteCodegenKernelAsync(
        CodegenKernel kernel,
        IReadOnlyList<IGpuBuffer> inputs,
        IReadOnlyList<IGpuBuffer> outputs,
        int elementCount,
        CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        if (_disposed) throw new ObjectDisposedException(nameof(CudaBackend));
        if (!IsAvailable) throw new GpuCapabilityUnavailableException("CUDA is not available.");
        if (kernel is NvrtcCodegenKernel nvrtcKernel)
        {
            ExecuteNvrtcCodegenKernel(nvrtcKernel, inputs, outputs, elementCount);
            return default;
        }
        if (kernel is not PtxCodegenKernel ptxKernel)
            throw new ArgumentException("The kernel must contain direct PTX or NVRTC CUDA C.", nameof(kernel));
        if (ptxKernel.ComputeMajor != _ccMajor || ptxKernel.ComputeMinor != _ccMinor)
        {
            throw new ArgumentException(
                $"PTX was emitted for sm_{ptxKernel.ComputeMajor}{ptxKernel.ComputeMinor}, " +
                $"but this backend owns sm_{_ccMajor}{_ccMinor}.", nameof(kernel));
        }
        if (elementCount != ptxKernel.OutputElementCount)
        {
            throw new ArgumentException(
                "Direct PTX launch geometry is shape-specialized; elementCount must match the emitted output.",
                nameof(elementCount));
        }

        ValidateCodegenBuffers(ptxKernel, inputs, outputs);
        DirectPtxCodegenKernel compiled;
        lock (_directPtxLock)
        {
            string key = CreateCodegenKey(ptxKernel);
            if (_directPtxCodegenKernels.TryGetValue(key, out DirectPtxCodegenKernel? cached))
            {
                compiled = cached;
            }
            else
            {
                compiled = CompileCodegenKernel(ptxKernel);
                _directPtxCodegenKernels.Add(key, compiled);
            }
        }

        int parameterCount = checked(ptxKernel.InputPortOrder.Count + outputs.Count);
        IntPtr* handles = stackalloc IntPtr[parameterCount];
        void** arguments = stackalloc void*[parameterCount];
        for (int parameter = 0; parameter < ptxKernel.InputPortOrder.Count; parameter++)
        {
            handles[parameter] = inputs[ptxKernel.InputPortOrder[parameter]].Handle;
            arguments[parameter] = &handles[parameter];
        }
        for (int output = 0; output < outputs.Count; output++)
        {
            int parameter = ptxKernel.InputPortOrder.Count + output;
            handles[parameter] = outputs[output].Handle;
            arguments[parameter] = &handles[parameter];
        }

        compiled.Module.Launch(
            compiled.Function,
            ptxKernel.LaunchBlocks, 1, 1,
            ptxKernel.LaunchBlockX, ptxKernel.LaunchBlockY, 1,
            0, arguments);
        LastCodegenNativeArtifactSizeBytes = compiled.NativeArtifactSizeBytes;
        return default;
    }

    private void ValidateCodegenBuffers(
        PtxCodegenKernel kernel,
        IReadOnlyList<IGpuBuffer> inputs,
        IReadOnlyList<IGpuBuffer> outputs)
    {
        if (inputs is null) throw new ArgumentNullException(nameof(inputs));
        if (outputs is null) throw new ArgumentNullException(nameof(outputs));
        if (inputs.Count != kernel.InputCount)
            throw new ArgumentException("Input buffer count does not match the generated graph.", nameof(inputs));
        if (outputs.Count != kernel.OutputCount)
            throw new ArgumentException("Output buffer count does not match the generated graph.", nameof(outputs));

        for (int i = 0; i < inputs.Count; i++)
        {
            long requiredBytes = checked(GetElementCount(kernel.Graph[kernel.Graph.InputNodes[i]].Shape) * sizeof(float));
            ValidateCodegenBuffer(inputs[i], requiredBytes, nameof(inputs));
        }
        for (int i = 0; i < outputs.Count; i++)
        {
            long requiredBytes = checked(GetElementCount(kernel.Graph[kernel.Graph.OutputNodes[i]].Shape) * sizeof(float));
            ValidateCodegenBuffer(outputs[i], requiredBytes, nameof(outputs));
        }
    }

    private void ValidateCodegenBuffer(IGpuBuffer buffer, long requiredBytes, string parameterName)
    {
        // Native memory this backend allocated in its context, or memory imported through DLPack and wrapped for this
        // backend (a producer's device pointer in the same primary context).
        bool ours = buffer switch
        {
            CudaGpuBuffer cuda => cuda.Handle != IntPtr.Zero && cuda.OwningContext == _cudaContext,
            ExternalDeviceBuffer external => external.Handle != IntPtr.Zero && ReferenceEquals(external.Owner, this),
            _ => false,
        };
        if (!ours)
            throw new ArgumentException("Every buffer must belong to this CUDA backend.", parameterName);
        if (buffer.SizeInBytes < requiredBytes)
            throw new ArgumentException("A generated-kernel buffer is smaller than its emitted shape.", parameterName);
    }

    private DirectPtxCodegenKernel CompileCodegenKernel(PtxCodegenKernel kernel)
    {
        _directPtxRuntime ??= new DirectPtxRuntime(_cudaContext, _mainStream);
        DirectPtxModule module = _directPtxRuntime.LoadModule(
            kernel.Ptx, allowExperimentalJitFallback: true);
        try
        {
            IntPtr function = module.GetFunction(kernel.EntryPoint, out _);
            long nativeBytes = module.CubinPath is string path && File.Exists(path)
                ? new FileInfo(path).Length
                : 0;
            return new DirectPtxCodegenKernel(module, function, nativeBytes);
        }
        catch
        {
            module.Dispose();
            throw;
        }
    }

    private static string CreateCodegenKey(PtxCodegenKernel kernel)
    {
        byte[] bytes = Encoding.UTF8.GetBytes(kernel.Ptx);
        byte[] digest;
        using (SHA256 sha = SHA256.Create()) digest = sha.ComputeHash(bytes);
        return BitConverter.ToString(digest).Replace("-", string.Empty);
    }

        private static long GetElementCount(IReadOnlyList<int> shape)
        {
            long count = 1;
            for (int i = 0; i < shape.Count; i++) count = checked(count * shape[i]);
            return count;
        }

    private unsafe void ExecuteNvrtcCodegenKernel(
        NvrtcCodegenKernel kernel,
        IReadOnlyList<IGpuBuffer> inputs,
        IReadOnlyList<IGpuBuffer> outputs,
        int elementCount)
    {
        if (inputs is null) throw new ArgumentNullException(nameof(inputs));
        if (outputs is null) throw new ArgumentNullException(nameof(outputs));
        // Validate against the binding metadata frozen at emission, not the (mutable) source graph: the compiled
        // entry point's parameter list is fixed by the emitted CUDA C.
        if (inputs.Count != kernel.BoundInputCount)
            throw new ArgumentException("Input buffer count does not match the generated graph.", nameof(inputs));
        if (outputs.Count != 1)
            throw new ArgumentException("Output buffer count does not match the generated graph.", nameof(outputs));
        if (elementCount != kernel.OutputElementCount)
        {
            throw new ArgumentException(
                $"NVRTC kernel is shape-specialized; elementCount ({elementCount}) must match the emitted output ({kernel.OutputElementCount}).",
                nameof(elementCount));
        }

        for (int i = 0; i < inputs.Count; i++)
        {
            ValidateCodegenBuffer(inputs[i], kernel.InputRequiredBytes[i], nameof(inputs));
        }
        ValidateCodegenBuffer(outputs[0], checked((long)elementCount * sizeof(float)), nameof(outputs));

        IntPtr function;
        lock (_directPtxLock)
        {
            if (!_nvrtcCodegenFunctions.TryGetValue(kernel.EntryPoint, out function))
            {
                if (IsStreamCapturing())
                {
                    throw new InvalidOperationException(
                        $"NVRTC kernel '{kernel.EntryPoint}' was not warmed up prior to CUDA stream capture.");
                }
                using var _ = PushContext();
                CuBlasNative.CheckCudaResult(CuBlasNative.cuDeviceGet(out int cudaDevice, DeviceOrdinal), "cuDeviceGet(nvrtc codegen)");
                CompileKernelModule(cudaDevice, kernel.CudaSource, kernel.EntryPoint, new[] { kernel.EntryPoint }, useFastMath: false);
                function = _kernelCache[kernel.EntryPoint];
                _nvrtcCodegenFunctions[kernel.EntryPoint] = function;
            }
        }

        int totalArgs = kernel.UsesTrailingBroadcast ? inputs.Count + 3 : inputs.Count + 2;
        IntPtr* handles = stackalloc IntPtr[inputs.Count + 1];
        void** args = stackalloc void*[totalArgs];
        for (int i = 0; i < inputs.Count; i++)
        {
            handles[i] = inputs[i].Handle;
            args[i] = &handles[i];
        }
        handles[inputs.Count] = outputs[0].Handle;
        args[inputs.Count] = &handles[inputs.Count];

        int n = elementCount;
        int cols = kernel.TrailingCols;
        args[inputs.Count + 1] = &n;
        if (kernel.UsesTrailingBroadcast)
        {
            args[inputs.Count + 2] = &cols;
        }

        uint blockX = 256;
        // 64-bit launch sizing so counts near int.MaxValue cannot wrap negative.
        long workItems = !kernel.UsesTrailingBroadcast && elementCount >= 4 ? ((long)elementCount + 3) >> 2 : elementCount;
        uint gridX = (uint)Math.Min(4096L, (workItems + blockX - 1) / blockX);
        if (gridX == 0) gridX = 1;

        using (PushContext())
        {
            lock (GpuDispatchLock)
            {
                LaunchKernel3D(function, gridX, 1, 1, blockX, 1, 1, args, 0);
            }
        }
    }

    private sealed class DirectPtxCodegenKernel : IDisposable
    {
        internal DirectPtxCodegenKernel(
            DirectPtxModule module,
            IntPtr function,
            long nativeArtifactSizeBytes)
        {
            Module = module;
            Function = function;
            NativeArtifactSizeBytes = nativeArtifactSizeBytes;
        }

        internal DirectPtxModule Module { get; }
        internal IntPtr Function { get; }
        internal long NativeArtifactSizeBytes { get; }

        public void Dispose() => Module.Dispose();
    }
}
