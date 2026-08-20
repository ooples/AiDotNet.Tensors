// Copyright (c) AiDotNet. All rights reserved.
// Generic-storage, arbitrary-length CUDA FFT dispatch.

using System;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Kernels;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA;

public sealed partial class CudaBackend
{
    private const int MaximumGenericFftLength = 1 << 29;
    private const int MaximumGenericFftBatch = 65535;

    private IntPtr _fftGenericFloat32Module;
    private IntPtr _fftGenericFloat16Module;
    private IntPtr _fftGenericBFloat16Module;
    private string? _fftGenericFloat32CompileError;
    private string? _fftGenericFloat16CompileError;
    private string? _fftGenericBFloat16CompileError;

    private void CompileGenericFftModules(int device)
    {
        CompileGenericFftModule(
            device,
            FftElementType.Float32,
            "fft_generic_f32",
            ref _fftGenericFloat32Module,
            ref _fftGenericFloat32CompileError);

        // CUDA half conversion intrinsics are supported from Maxwell (SM 5.3).
        if ((_ccMajor * 10) + _ccMinor >= 53)
        {
            CompileGenericFftModule(
                device,
                FftElementType.Float16,
                "fft_generic_f16",
                ref _fftGenericFloat16Module,
                ref _fftGenericFloat16CompileError);
        }

        // Native bfloat16 storage/conversion intrinsics require Ampere (SM 8.0).
        if (_ccMajor >= 8)
        {
            CompileGenericFftModule(
                device,
                FftElementType.BFloat16,
                "fft_generic_bf16",
                ref _fftGenericBFloat16Module,
                ref _fftGenericBFloat16CompileError);
        }
    }

    private void CompileGenericFftModule(
        int device,
        FftElementType type,
        string moduleName,
        ref IntPtr module,
        ref string? compileError)
    {
        try
        {
            module = CompileKernelModule(
                device,
                CudaFFTGenericKernels.GetSource(type),
                moduleName,
                CudaFFTGenericKernels.GetKernelNames(type),
                useFastMath: false);
            compileError = null;
        }
        catch (OutOfMemoryException)
        {
            // Resource exhaustion is not an optional feature-compatibility failure. Let backend
            // construction surface it rather than continuing in an unknown memory state.
            throw;
        }
        catch (Exception ex)
        {
            // Generic FFT is an optional CUDA capability. Preserve the rest of the backend and
            // expose the failed specialization as unsupported instead of failing engine startup.
            module = IntPtr.Zero;
            compileError = ex.Message;
            System.Diagnostics.Trace.TraceWarning(
                $"CUDA {type} generic FFT kernels are unavailable: {ex.Message}");
        }
    }

    private void DisableGenericFftModules(Exception error)
    {
        try
        {
            UnloadGenericFftModules();
        }
        catch (Exception unloadError)
        {
            // Capability state below remains authoritative even if a partially loaded module
            // cannot be unloaded here; destroying the CUDA context reclaims it at disposal.
            System.Diagnostics.Trace.TraceWarning(
                $"[CudaBackend] Could not unload partially compiled generic FFT modules: " +
                $"{unloadError.GetType().Name}: {unloadError.Message}");
        }
        finally
        {
            _fftGenericFloat32Module = IntPtr.Zero;
            _fftGenericFloat16Module = IntPtr.Zero;
            _fftGenericBFloat16Module = IntPtr.Zero;
            string detail = $"{error.GetType().Name}: {error.Message}";
            _fftGenericFloat32CompileError = detail;
            _fftGenericFloat16CompileError = detail;
            _fftGenericBFloat16CompileError = detail;
        }
    }

    private void UnloadGenericFftModules()
    {
        UnloadGenericFftModule(ref _fftGenericFloat32Module);
        UnloadGenericFftModule(ref _fftGenericFloat16Module);
        UnloadGenericFftModule(ref _fftGenericBFloat16Module);
    }

    private static void UnloadGenericFftModule(ref IntPtr module)
    {
        if (module == IntPtr.Zero)
        {
            return;
        }

        CudaNativeBindings.cuModuleUnload(module);
        module = IntPtr.Zero;
    }

    internal bool SupportsFftElementType(FftElementType type)
    {
        _ = type.ByteSize();
        if (!IsAvailable)
        {
            return false;
        }

        return type switch
        {
            FftElementType.Float32 => _fftGenericFloat32Module != IntPtr.Zero,
            FftElementType.Float16 =>
                ((_ccMajor * 10) + _ccMinor >= 53) && _fftGenericFloat16Module != IntPtr.Zero,
            FftElementType.BFloat16 => _ccMajor >= 8 && _fftGenericBFloat16Module != IntPtr.Zero,
            _ => throw new ArgumentOutOfRangeException(nameof(type), type, "Unknown FFT element type."),
        };
    }

    internal unsafe void ConvertFloatToFftStorage(
        IGpuBuffer source,
        IGpuBuffer destination,
        int count,
        FftElementType type)
    {
        ValidateConversionBuffers(source, destination, count, type, floatToStorage: true);
        IntPtr kernel = RequireGenericFftKernel("fftg_from_f32" + type.KernelSuffix(), type);
        IntPtr sourcePtr = source.Handle;
        IntPtr destinationPtr = destination.Handle;
        void** args = stackalloc void*[3];
        args[0] = &sourcePtr;
        args[1] = &destinationPtr;
        args[2] = &count;

        using var _ = PushContext();
        LaunchKernel(kernel, GridFor(count), DefaultBlockSize, args);
    }

    internal unsafe void ConvertFftStorageToFloat(
        IGpuBuffer source,
        IGpuBuffer destination,
        int count,
        FftElementType type)
    {
        ValidateConversionBuffers(source, destination, count, type, floatToStorage: false);
        IntPtr kernel = RequireGenericFftKernel("fftg_to_f32" + type.KernelSuffix(), type);
        IntPtr sourcePtr = source.Handle;
        IntPtr destinationPtr = destination.Handle;
        void** args = stackalloc void*[3];
        args[0] = &sourcePtr;
        args[1] = &destinationPtr;
        args[2] = &count;

        using var _ = PushContext();
        LaunchKernel(kernel, GridFor(count), DefaultBlockSize, args);
    }

    internal void LaunchFftGeneric(
        IGpuBuffer real,
        IGpuBuffer imaginary,
        int batch,
        int n,
        bool inverse,
        FftElementType type)
    {
        ValidateGenericFftBuffers(real, imaginary, batch, n, type);

        // All allocations, copies and launches use one CUDA context/stream. The outer lock
        // makes the multi-launch transform atomic with respect to other host dispatchers.
        lock (GpuDispatchLock)
        {
            using var _ = PushContext();
            if (CudaFFTGenericKernels.IsPowerOfTwo(n))
            {
                LaunchTypedRadix2(real, imaginary, batch, n, inverse, type);
            }
            else
            {
                LaunchBluestein(real, imaginary, batch, n, inverse, type);
            }
        }
    }

    private void ValidateConversionBuffers(
        IGpuBuffer source,
        IGpuBuffer destination,
        int count,
        FftElementType type,
        bool floatToStorage)
    {
        if (source is null) throw new ArgumentNullException(nameof(source));
        if (destination is null) throw new ArgumentNullException(nameof(destination));
        if (count <= 0) throw new ArgumentOutOfRangeException(nameof(count));
        EnsureGenericFftSupported(type);

        long floatBytes = checked((long)count * sizeof(float));
        long storageBytes = checked((long)count * type.ByteSize());
        long requiredSource = floatToStorage ? floatBytes : storageBytes;
        long requiredDestination = floatToStorage ? storageBytes : floatBytes;
        if (source.SizeInBytes < requiredSource)
        {
            throw new ArgumentException(
                $"Source buffer provides {source.SizeInBytes} bytes; {requiredSource} are required.",
                nameof(source));
        }

        if (destination.SizeInBytes < requiredDestination)
        {
            throw new ArgumentException(
                $"Destination buffer provides {destination.SizeInBytes} bytes; {requiredDestination} are required.",
                nameof(destination));
        }
    }

    private void ValidateGenericFftBuffers(
        IGpuBuffer real,
        IGpuBuffer imaginary,
        int batch,
        int n,
        FftElementType type)
    {
        if (real is null) throw new ArgumentNullException(nameof(real));
        if (imaginary is null) throw new ArgumentNullException(nameof(imaginary));
        if (batch <= 0 || batch > MaximumGenericFftBatch)
        {
            throw new ArgumentOutOfRangeException(
                nameof(batch), batch, $"Batch must be in [1, {MaximumGenericFftBatch}].");
        }

        if (n <= 0 || n > MaximumGenericFftLength)
        {
            throw new ArgumentOutOfRangeException(
                nameof(n), n, $"Transform length must be in [1, {MaximumGenericFftLength}].");
        }

        EnsureGenericFftSupported(type);
        long requiredBytes = checked((long)batch * n * type.ByteSize());
        if (real.SizeInBytes < requiredBytes)
        {
            throw new ArgumentException(
                $"Real buffer provides {real.SizeInBytes} bytes; {requiredBytes} are required.",
                nameof(real));
        }

        if (imaginary.SizeInBytes < requiredBytes)
        {
            throw new ArgumentException(
                $"Imaginary buffer provides {imaginary.SizeInBytes} bytes; {requiredBytes} are required.",
                nameof(imaginary));
        }
    }

    private void EnsureGenericFftSupported(FftElementType type)
    {
        if (SupportsFftElementType(type))
        {
            return;
        }

        string? error = type switch
        {
            FftElementType.Float32 => _fftGenericFloat32CompileError,
            FftElementType.Float16 => _fftGenericFloat16CompileError,
            FftElementType.BFloat16 => _fftGenericBFloat16CompileError,
            _ => throw new ArgumentOutOfRangeException(nameof(type), type, "Unknown FFT element type."),
        };
        string detail = string.IsNullOrWhiteSpace(error) ? string.Empty : $" NVRTC: {error}";
        throw new NotSupportedException(
            $"CUDA generic FFT storage {type} is unavailable on compute capability {_ccMajor}.{_ccMinor}.{detail}");
    }

    private IntPtr RequireGenericFftKernel(string name, FftElementType type)
    {
        EnsureGenericFftSupported(type);
        if (_kernelCache.TryGetValue(name, out IntPtr kernel))
        {
            return kernel;
        }

        throw new InvalidOperationException($"CUDA generic FFT kernel was not registered: {name}.");
    }

    private IntPtr RequireInvariantGenericFftKernel(string name)
    {
        if (_kernelCache.TryGetValue(name, out IntPtr kernel))
        {
            return kernel;
        }

        throw new InvalidOperationException($"CUDA generic FFT kernel was not registered: {name}.");
    }

    private IGpuBuffer AllocateFftScratchBuffer(int size) =>
        new DeferredFftScratchBuffer(this, AllocateBuffer(size));

    private IGpuBuffer AllocateFftScratchByteBuffer(int size) =>
        new DeferredFftScratchBuffer(this, AllocateByteBuffer(size));

    /// <summary>
    /// Gives every temporary FFT allocation stream-safe RAII semantics. The normal async
    /// allocator releases in stream order; the legacy allocator records an event and keeps
    /// the buffer out of its reuse pool until all preceding FFT work has completed.
    /// </summary>
    private sealed class DeferredFftScratchBuffer : IGpuBuffer
    {
        private readonly CudaBackend _owner;
        private IGpuBuffer? _buffer;

        internal DeferredFftScratchBuffer(CudaBackend owner, IGpuBuffer buffer)
        {
            _owner = owner;
            _buffer = buffer;
        }

        private IGpuBuffer Buffer => _buffer ??
            throw new ObjectDisposedException(nameof(DeferredFftScratchBuffer));

        public int Size => Buffer.Size;
        public long SizeInBytes => Buffer.SizeInBytes;
        public IntPtr Handle => Buffer.Handle;

        public void Dispose()
        {
            IGpuBuffer? buffer = System.Threading.Interlocked.Exchange(ref _buffer, null);
            if (buffer is not null)
            {
                _owner.FreeBufferDeferred(buffer);
            }
        }
    }

    private unsafe void LaunchTypedRadix2(
        IGpuBuffer real,
        IGpuBuffer imaginary,
        int batch,
        int n,
        bool inverse,
        FftElementType type)
    {
        int count = checked(batch * n);
        int bytes = checked(count * type.ByteSize());
        int half = Math.Max(1, n / 2);
        using IGpuBuffer twiddleReal = AllocateFftScratchBuffer(half);
        using IGpuBuffer twiddleImaginary = AllocateFftScratchBuffer(half);
        BuildTwiddles(twiddleReal, twiddleImaginary, n, inverse);

        using IGpuBuffer scratchReal = AllocateFftScratchByteBuffer(bytes);
        using IGpuBuffer scratchImaginary = AllocateFftScratchByteBuffer(bytes);
        IntPtr bitReverse = RequireGenericFftKernel("fftg_batched_bit_reverse" + type.KernelSuffix(), type);
        IntPtr realPtr = real.Handle;
        IntPtr imaginaryPtr = imaginary.Handle;
        IntPtr scratchRealPtr = scratchReal.Handle;
        IntPtr scratchImaginaryPtr = scratchImaginary.Handle;
        int log2n = IntegerLog2(n);
        void** reverseArgs = stackalloc void*[7];
        reverseArgs[0] = &realPtr;
        reverseArgs[1] = &imaginaryPtr;
        reverseArgs[2] = &scratchRealPtr;
        reverseArgs[3] = &scratchImaginaryPtr;
        reverseArgs[4] = &batch;
        reverseArgs[5] = &n;
        reverseArgs[6] = &log2n;
        LaunchKernel2D(bitReverse, GridFor(n), (uint)batch, DefaultBlockSize, 1, reverseArgs);
        CopyBufferDtoD(scratchReal, real, bytes);
        CopyBufferDtoD(scratchImaginary, imaginary, bytes);

        IntPtr butterfly = RequireGenericFftKernel("fftg_batched_butterfly" + type.KernelSuffix(), type);
        IntPtr twiddleRealPtr = twiddleReal.Handle;
        IntPtr twiddleImaginaryPtr = twiddleImaginary.Handle;
        void** butterflyArgs = stackalloc void*[7];
        butterflyArgs[0] = &realPtr;
        butterflyArgs[1] = &imaginaryPtr;
        butterflyArgs[2] = &twiddleRealPtr;
        butterflyArgs[3] = &twiddleImaginaryPtr;
        butterflyArgs[4] = &batch;
        butterflyArgs[5] = &n;
        for (int stride = 2; stride <= n; stride *= 2)
        {
            butterflyArgs[6] = &stride;
            LaunchKernel2D(butterfly, GridFor(n / 2), (uint)batch, DefaultBlockSize, 1, butterflyArgs);
            if (stride == n) break;
        }

        if (inverse)
        {
            LaunchTypedScale(real, imaginary, count, 1.0f / n, type);
        }
    }

    private unsafe void LaunchFloat32Radix2(
        IGpuBuffer real,
        IGpuBuffer imaginary,
        int batch,
        int n,
        bool inverse)
    {
        int count = checked(batch * n);
        int half = Math.Max(1, n / 2);
        using IGpuBuffer twiddleReal = AllocateFftScratchBuffer(half);
        using IGpuBuffer twiddleImaginary = AllocateFftScratchBuffer(half);
        BuildTwiddles(twiddleReal, twiddleImaginary, n, inverse);

        using IGpuBuffer scratchReal = AllocateFftScratchBuffer(count);
        using IGpuBuffer scratchImaginary = AllocateFftScratchBuffer(count);
        IntPtr bitReverse = RequireInvariantGenericFftKernel("fftg_f32_bit_reverse");
        IntPtr realPtr = real.Handle;
        IntPtr imaginaryPtr = imaginary.Handle;
        IntPtr scratchRealPtr = scratchReal.Handle;
        IntPtr scratchImaginaryPtr = scratchImaginary.Handle;
        int log2n = IntegerLog2(n);
        void** reverseArgs = stackalloc void*[7];
        reverseArgs[0] = &realPtr;
        reverseArgs[1] = &imaginaryPtr;
        reverseArgs[2] = &scratchRealPtr;
        reverseArgs[3] = &scratchImaginaryPtr;
        reverseArgs[4] = &batch;
        reverseArgs[5] = &n;
        reverseArgs[6] = &log2n;
        LaunchKernel2D(bitReverse, GridFor(n), (uint)batch, DefaultBlockSize, 1, reverseArgs);
        CopyBufferDtoD(scratchReal, real, checked((long)count * sizeof(float)));
        CopyBufferDtoD(scratchImaginary, imaginary, checked((long)count * sizeof(float)));

        IntPtr butterfly = RequireInvariantGenericFftKernel("fftg_f32_butterfly");
        IntPtr twiddleRealPtr = twiddleReal.Handle;
        IntPtr twiddleImaginaryPtr = twiddleImaginary.Handle;
        void** butterflyArgs = stackalloc void*[7];
        butterflyArgs[0] = &realPtr;
        butterflyArgs[1] = &imaginaryPtr;
        butterflyArgs[2] = &twiddleRealPtr;
        butterflyArgs[3] = &twiddleImaginaryPtr;
        butterflyArgs[4] = &batch;
        butterflyArgs[5] = &n;
        for (int stride = 2; stride <= n; stride *= 2)
        {
            butterflyArgs[6] = &stride;
            LaunchKernel2D(butterfly, GridFor(n / 2), (uint)batch, DefaultBlockSize, 1, butterflyArgs);
            if (stride == n) break;
        }

        if (inverse)
        {
            LaunchFloat32Scale(real, imaginary, count, 1.0f / n);
        }
    }

    private unsafe void BuildTwiddles(IGpuBuffer real, IGpuBuffer imaginary, int n, bool inverse)
    {
        IntPtr kernel = RequireInvariantGenericFftKernel("fftg_build_twiddles");
        IntPtr realPtr = real.Handle;
        IntPtr imaginaryPtr = imaginary.Handle;
        int half = Math.Max(1, n / 2);
        int inverseValue = inverse ? 1 : 0;
        void** args = stackalloc void*[5];
        args[0] = &realPtr;
        args[1] = &imaginaryPtr;
        args[2] = &half;
        args[3] = &n;
        args[4] = &inverseValue;
        LaunchKernel(kernel, GridFor(half), DefaultBlockSize, args);
    }

    private unsafe void LaunchTypedScale(
        IGpuBuffer real,
        IGpuBuffer imaginary,
        int count,
        float scale,
        FftElementType type)
    {
        IntPtr kernel = RequireGenericFftKernel("fftg_scale" + type.KernelSuffix(), type);
        IntPtr realPtr = real.Handle;
        IntPtr imaginaryPtr = imaginary.Handle;
        void** args = stackalloc void*[4];
        args[0] = &realPtr;
        args[1] = &imaginaryPtr;
        args[2] = &count;
        args[3] = &scale;
        LaunchKernel(kernel, GridFor(count), DefaultBlockSize, args);
    }

    private unsafe void LaunchFloat32Scale(
        IGpuBuffer real,
        IGpuBuffer imaginary,
        int count,
        float scale)
    {
        IntPtr kernel = RequireInvariantGenericFftKernel("fftg_f32_scale");
        IntPtr realPtr = real.Handle;
        IntPtr imaginaryPtr = imaginary.Handle;
        void** args = stackalloc void*[4];
        args[0] = &realPtr;
        args[1] = &imaginaryPtr;
        args[2] = &count;
        args[3] = &scale;
        LaunchKernel(kernel, GridFor(count), DefaultBlockSize, args);
    }

    private unsafe void LaunchBluestein(
        IGpuBuffer real,
        IGpuBuffer imaginary,
        int batch,
        int n,
        bool inverse,
        FftElementType type)
    {
        int m = CudaFFTGenericKernels.BluesteinLength(n);
        int workspaceCount = checked(batch * m);
        using IGpuBuffer chirpReal = AllocateFftScratchBuffer(n);
        using IGpuBuffer chirpImaginary = AllocateFftScratchBuffer(n);
        using IGpuBuffer kernelReal = AllocateFftScratchBuffer(m);
        using IGpuBuffer kernelImaginary = AllocateFftScratchBuffer(m);
        using IGpuBuffer workspaceReal = AllocateFftScratchBuffer(workspaceCount);
        using IGpuBuffer workspaceImaginary = AllocateFftScratchBuffer(workspaceCount);

        BuildChirp(chirpReal, chirpImaginary, n, inverse);
        BuildPaddedChirp(kernelReal, kernelImaginary, n, m, inverse);
        LaunchFloat32Radix2(kernelReal, kernelImaginary, 1, m, inverse: false);
        LaunchBluesteinPremultiply(
            real, imaginary, chirpReal, chirpImaginary,
            workspaceReal, workspaceImaginary, batch, n, m, type);
        LaunchFloat32Radix2(workspaceReal, workspaceImaginary, batch, m, inverse: false);
        LaunchBluesteinPointwise(
            workspaceReal, workspaceImaginary, kernelReal, kernelImaginary, batch, m);
        LaunchFloat32Radix2(workspaceReal, workspaceImaginary, batch, m, inverse: true);
        LaunchBluesteinPostmultiply(
            workspaceReal, workspaceImaginary, chirpReal, chirpImaginary,
            real, imaginary, batch, n, m, inverse ? 1.0f / n : 1.0f, type);
    }

    private unsafe void BuildChirp(IGpuBuffer real, IGpuBuffer imaginary, int n, bool inverse)
    {
        IntPtr kernel = RequireInvariantGenericFftKernel("fftg_build_chirp");
        IntPtr realPtr = real.Handle;
        IntPtr imaginaryPtr = imaginary.Handle;
        int inverseValue = inverse ? 1 : 0;
        void** args = stackalloc void*[4];
        args[0] = &realPtr;
        args[1] = &imaginaryPtr;
        args[2] = &n;
        args[3] = &inverseValue;
        LaunchKernel(kernel, GridFor(n), DefaultBlockSize, args);
    }

    private unsafe void BuildPaddedChirp(
        IGpuBuffer real,
        IGpuBuffer imaginary,
        int n,
        int m,
        bool inverse)
    {
        IntPtr kernel = RequireInvariantGenericFftKernel("fftg_build_chirp_padded");
        IntPtr realPtr = real.Handle;
        IntPtr imaginaryPtr = imaginary.Handle;
        int inverseValue = inverse ? 1 : 0;
        void** args = stackalloc void*[5];
        args[0] = &realPtr;
        args[1] = &imaginaryPtr;
        args[2] = &n;
        args[3] = &m;
        args[4] = &inverseValue;
        LaunchKernel(kernel, GridFor(m), DefaultBlockSize, args);
    }

    private unsafe void LaunchBluesteinPremultiply(
        IGpuBuffer inputReal,
        IGpuBuffer inputImaginary,
        IGpuBuffer chirpReal,
        IGpuBuffer chirpImaginary,
        IGpuBuffer workspaceReal,
        IGpuBuffer workspaceImaginary,
        int batch,
        int n,
        int m,
        FftElementType type)
    {
        IntPtr kernel = RequireGenericFftKernel("fftg_bluestein_premul" + type.KernelSuffix(), type);
        IntPtr inputRealPtr = inputReal.Handle;
        IntPtr inputImaginaryPtr = inputImaginary.Handle;
        IntPtr chirpRealPtr = chirpReal.Handle;
        IntPtr chirpImaginaryPtr = chirpImaginary.Handle;
        IntPtr workspaceRealPtr = workspaceReal.Handle;
        IntPtr workspaceImaginaryPtr = workspaceImaginary.Handle;
        void** args = stackalloc void*[9];
        args[0] = &inputRealPtr;
        args[1] = &inputImaginaryPtr;
        args[2] = &chirpRealPtr;
        args[3] = &chirpImaginaryPtr;
        args[4] = &workspaceRealPtr;
        args[5] = &workspaceImaginaryPtr;
        args[6] = &batch;
        args[7] = &n;
        args[8] = &m;
        LaunchKernel2D(kernel, GridFor(m), (uint)batch, DefaultBlockSize, 1, args);
    }

    private unsafe void LaunchBluesteinPointwise(
        IGpuBuffer real,
        IGpuBuffer imaginary,
        IGpuBuffer kernelReal,
        IGpuBuffer kernelImaginary,
        int batch,
        int m)
    {
        IntPtr kernel = RequireInvariantGenericFftKernel("fftg_bluestein_pointwise");
        IntPtr realPtr = real.Handle;
        IntPtr imaginaryPtr = imaginary.Handle;
        IntPtr kernelRealPtr = kernelReal.Handle;
        IntPtr kernelImaginaryPtr = kernelImaginary.Handle;
        void** args = stackalloc void*[6];
        args[0] = &realPtr;
        args[1] = &imaginaryPtr;
        args[2] = &kernelRealPtr;
        args[3] = &kernelImaginaryPtr;
        args[4] = &batch;
        args[5] = &m;
        LaunchKernel2D(kernel, GridFor(m), (uint)batch, DefaultBlockSize, 1, args);
    }

    private unsafe void LaunchBluesteinPostmultiply(
        IGpuBuffer workspaceReal,
        IGpuBuffer workspaceImaginary,
        IGpuBuffer chirpReal,
        IGpuBuffer chirpImaginary,
        IGpuBuffer outputReal,
        IGpuBuffer outputImaginary,
        int batch,
        int n,
        int m,
        float scale,
        FftElementType type)
    {
        IntPtr kernel = RequireGenericFftKernel("fftg_bluestein_postmul" + type.KernelSuffix(), type);
        IntPtr workspaceRealPtr = workspaceReal.Handle;
        IntPtr workspaceImaginaryPtr = workspaceImaginary.Handle;
        IntPtr chirpRealPtr = chirpReal.Handle;
        IntPtr chirpImaginaryPtr = chirpImaginary.Handle;
        IntPtr outputRealPtr = outputReal.Handle;
        IntPtr outputImaginaryPtr = outputImaginary.Handle;
        void** args = stackalloc void*[10];
        args[0] = &workspaceRealPtr;
        args[1] = &workspaceImaginaryPtr;
        args[2] = &chirpRealPtr;
        args[3] = &chirpImaginaryPtr;
        args[4] = &outputRealPtr;
        args[5] = &outputImaginaryPtr;
        args[6] = &batch;
        args[7] = &n;
        args[8] = &m;
        args[9] = &scale;
        LaunchKernel2D(kernel, GridFor(n), (uint)batch, DefaultBlockSize, 1, args);
    }

    private static uint GridFor(int count) =>
        checked((uint)((count + (long)DefaultBlockSize - 1) / DefaultBlockSize));

    private static int IntegerLog2(int n)
    {
        int result = 0;
        while ((1 << result) < n)
        {
            result++;
        }

        return result;
    }
}
