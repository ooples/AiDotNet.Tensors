// Copyright (c) AiDotNet. All rights reserved.

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
        return emitter.Emit(graph, dtype);
    }

    /// <summary>
    /// Size of the cubin selected for the most recent generated launch, or zero when the CUDA
    /// driver JIT fallback does not expose its module image.
    /// </summary>
    public long LastCodegenNativeArtifactSizeBytes { get; private set; }

    /// <inheritdoc/>
    public bool CanExecuteCodegenKernel(CodegenKernel kernel) =>
        kernel is PtxCodegenKernel ptx &&
        ptx.ComputeMajor == _ccMajor && ptx.ComputeMinor == _ccMinor;

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
        if (!IsAvailable) throw new InvalidOperationException("CUDA is not available.");
        if (kernel is not PtxCodegenKernel ptxKernel)
            throw new ArgumentException("The kernel must contain direct PTX.", nameof(kernel));
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
        if (buffer is not CudaGpuBuffer cudaBuffer ||
            cudaBuffer.Handle == IntPtr.Zero ||
            cudaBuffer.OwningContext != _cudaContext)
        {
            throw new ArgumentException("Every buffer must belong to this CUDA backend.", parameterName);
        }
        if (cudaBuffer.SizeInBytes < requiredBytes)
            throw new ArgumentException("A generated-kernel buffer is smaller than its emitted shape.", parameterName);
    }

    private DirectPtxCodegenKernel CompileCodegenKernel(PtxCodegenKernel kernel)
    {
        _directPtxRuntime ??= new DirectPtxRuntime(_cudaContext, _stream);
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
