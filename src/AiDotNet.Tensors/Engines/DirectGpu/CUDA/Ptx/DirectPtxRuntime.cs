using System;
using System.Runtime.InteropServices;
using System.Globalization;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// Minimal CUDA Driver-API runtime for loading and launching hand-emitted PTX.
/// It deliberately has no dependency on cudart, NVRTC, cuBLAS, or cuDNN.
/// </summary>
internal sealed class DirectPtxRuntime : IDisposable
{
    [ThreadStatic] private static IntPtr s_scopedContext;
    [ThreadStatic] private static int s_scopeDepth;
    private IntPtr _context;
    private readonly IntPtr _stream;
    private readonly bool _ownsContext;
    private bool _disposed;

    internal int DeviceOrdinal { get; }
    internal string DeviceName { get; }
    internal int ComputeCapabilityMajor { get; }
    internal int ComputeCapabilityMinor { get; }
    internal int MaxThreadsPerMultiprocessor { get; }
    internal DirectPtxArchitectureFamily ArchitectureFamily { get; }
    internal string DeviceUuid { get; }
    internal int DriverVersion { get; }
    internal string DeviceFingerprint { get; }
    internal Helpers.Autotune.GpuDeviceFingerprint Fingerprint { get; }
    internal IntPtr Stream => _stream;

    internal static bool IsAvailable => CudaNativeBindings.IsAvailable;

    internal DirectPtxRuntime(int deviceOrdinal = 0)
    {
        Check(CuBlasNative.cuInit(0), "cuInit");
        Check(CuBlasNative.cuDeviceGet(out int device, deviceOrdinal), "cuDeviceGet");

        var name = new StringBuilder(256);
        Check(CuBlasNative.cuDeviceGetName(name, name.Capacity, device), "cuDeviceGetName");
        Check(CuBlasNative.cuDeviceGetAttribute(
            out int major, (int)CudaDeviceAttribute.ComputeCapabilityMajor, device),
            "cuDeviceGetAttribute(ComputeCapabilityMajor)");
        Check(CuBlasNative.cuDeviceGetAttribute(
            out int minor, (int)CudaDeviceAttribute.ComputeCapabilityMinor, device),
            "cuDeviceGetAttribute(ComputeCapabilityMinor)");
        Check(CuBlasNative.cuDeviceGetAttribute(
            out int maxThreadsPerMultiprocessor,
            (int)CudaDeviceAttribute.MaxThreadsPerMultiprocessor, device),
            "cuDeviceGetAttribute(MaxThreadsPerMultiprocessor)");

        Check(CuBlasNative.cuCtxCreate(out _context, 0, device), "cuCtxCreate");
        // cuCtxCreate makes the context current. Detach it so every operation
        // below has an explicit, balanced push/pop boundary.
        Check(CuBlasNative.cuCtxPopCurrent(out IntPtr popped), "cuCtxPopCurrent");
        if (popped != _context)
        {
            CuBlasNative.cuCtxDestroy(_context);
            _context = IntPtr.Zero;
            throw new InvalidOperationException("CUDA returned a different context from cuCtxPopCurrent.");
        }

        _stream = IntPtr.Zero;
        _ownsContext = true;
        DeviceOrdinal = deviceOrdinal;
        DeviceName = name.ToString();
        ComputeCapabilityMajor = major;
        ComputeCapabilityMinor = minor;
        MaxThreadsPerMultiprocessor = maxThreadsPerMultiprocessor;
        ArchitectureFamily = DirectPtxArchitecture.Classify(major, minor);
        DeviceUuid = QueryDeviceUuid(device);
        DriverVersion = CudaNativeBindings.DriverVersion;
        Fingerprint = Helpers.Autotune.GpuDeviceFingerprint.FromCuda(
            DeviceName, DeviceUuid, major, minor, DriverVersion);
        // Byte-identical to the legacy fingerprint string, so existing on-disk
        // autotune caches keyed by DeviceFingerprint remain valid.
        DeviceFingerprint = Fingerprint.ToCacheToken();
    }

    /// <summary>
    /// Creates a non-owning PTX runtime over an existing CUDA context and
    /// stream. Production kernels use this path so resident AiDotNet buffers,
    /// generated modules, events, and launches all share one ordering domain.
    /// </summary>
    internal DirectPtxRuntime(IntPtr borrowedContext, IntPtr borrowedStream)
    {
        if (borrowedContext == IntPtr.Zero)
            throw new ArgumentException("A borrowed CUDA context cannot be null.", nameof(borrowedContext));

        _context = borrowedContext;
        _stream = borrowedStream;
        _ownsContext = false;

        using var _ = Enter();
        Check(CudaNativeBindings.cuCtxGetDevice(out int device), "cuCtxGetDevice");
        var name = new StringBuilder(256);
        Check(CuBlasNative.cuDeviceGetName(name, name.Capacity, device), "cuDeviceGetName");
        Check(CuBlasNative.cuDeviceGetAttribute(
            out int major, (int)CudaDeviceAttribute.ComputeCapabilityMajor, device),
            "cuDeviceGetAttribute(ComputeCapabilityMajor)");
        Check(CuBlasNative.cuDeviceGetAttribute(
            out int minor, (int)CudaDeviceAttribute.ComputeCapabilityMinor, device),
            "cuDeviceGetAttribute(ComputeCapabilityMinor)");
        Check(CuBlasNative.cuDeviceGetAttribute(
            out int maxThreadsPerMultiprocessor,
            (int)CudaDeviceAttribute.MaxThreadsPerMultiprocessor, device),
            "cuDeviceGetAttribute(MaxThreadsPerMultiprocessor)");

        DeviceOrdinal = device;
        DeviceName = name.ToString();
        ComputeCapabilityMajor = major;
        ComputeCapabilityMinor = minor;
        MaxThreadsPerMultiprocessor = maxThreadsPerMultiprocessor;
        ArchitectureFamily = DirectPtxArchitecture.Classify(major, minor);
        DeviceUuid = QueryDeviceUuid(device);
        DriverVersion = CudaNativeBindings.DriverVersion;
        Fingerprint = Helpers.Autotune.GpuDeviceFingerprint.FromCuda(
            DeviceName, DeviceUuid, major, minor, DriverVersion);
        // Byte-identical to the legacy fingerprint string, so existing on-disk
        // autotune caches keyed by DeviceFingerprint remain valid.
        DeviceFingerprint = Fingerprint.ToCacheToken();
    }

    private static unsafe string QueryDeviceUuid(int device)
    {
        try
        {
            if (CudaNativeBindings.cuDeviceGetUuidV2(out CudaDeviceUuid uuid, device) != CudaResult.Success)
                return $"ordinal-{device}";
            byte[] bytes = new byte[16];
            for (int i = 0; i < bytes.Length; i++) bytes[i] = uuid.Bytes[i];
            return PtxCompat.ToHexString(bytes).ToLowerInvariant();
        }
        catch (EntryPointNotFoundException)
        {
            return $"ordinal-{device}";
        }
    }


    internal ContextScope Enter()
    {
        PtxCompat.ThrowIfDisposed(_disposed, this);
        return new ContextScope(_context);
    }

    internal DirectPtxBuffer AllocateBytes(nuint bytes)
    {
        if (bytes == 0) throw new ArgumentOutOfRangeException(nameof(bytes));
        using var _ = Enter();
        Check(CudaNativeBindings.cuMemAlloc(out IntPtr pointer, checked((ulong)bytes)), "cuMemAlloc");
        return new DirectPtxBuffer(this, pointer, bytes);
    }

    internal DirectPtxModule LoadModule(string ptx)
    {
        if (string.IsNullOrWhiteSpace(ptx)) throw new ArgumentException("PTX cannot be empty.", nameof(ptx));
        using var _ = Enter();
        IntPtr text = Marshal.StringToHGlobalAnsi(ptx);
        const int logBytes = 16 * 1024;
        IntPtr errorLog = Marshal.AllocHGlobal(logBytes);
        IntPtr infoLog = Marshal.AllocHGlobal(logBytes);
        try
        {
            unsafe
            {
                new Span<byte>((void*)errorLog, logBytes).Clear();
                new Span<byte>((void*)infoLog, logBytes).Clear();
            }
            // CUjit_option values: INFO_LOG_BUFFER=3, INFO_LOG_SIZE=4,
            // ERROR_LOG_BUFFER=5, ERROR_LOG_SIZE=6. Size values are passed
            // by value through the void* option slot per the Driver API ABI.
            int[] options = [3, 4, 5, 6];
            IntPtr[] values = [infoLog, (IntPtr)logBytes, errorLog, (IntPtr)logBytes];
            CudaResult result = CudaNativeBindings.cuModuleLoadDataEx(
                out IntPtr module, text, (uint)options.Length, options, values);
            if (result != CudaResult.Success)
            {
                string error = Marshal.PtrToStringAnsi(errorLog) ?? string.Empty;
                string info = Marshal.PtrToStringAnsi(infoLog) ?? string.Empty;
                throw new InvalidOperationException(
                    $"cuModuleLoadDataEx(PTX) failed with CUDA driver status {(int)result} ({result}).\n" +
                    $"JIT error log:\n{error}\nJIT info log:\n{info}");
            }
            string jitInfo = Marshal.PtrToStringAnsi(infoLog) ?? string.Empty;
            return new DirectPtxModule(this, module, jitInfo);
        }
        finally
        {
            Marshal.FreeHGlobal(errorLog);
            Marshal.FreeHGlobal(infoLog);
            Marshal.FreeHGlobal(text);
        }
    }

    internal void Synchronize()
    {
        using var _ = Enter();
        if (_stream != IntPtr.Zero)
            Check(CudaNativeBindings.cuStreamSynchronize(_stream), "cuStreamSynchronize");
        else
            Check(CuBlasNative.cuCtxSynchronize(), "cuCtxSynchronize");
    }

    internal float MeasureKernelMilliseconds(Action launch, int warmup, int iterations)
    {
        PtxCompat.ThrowIfNull(launch, nameof(launch));
        if (warmup < 0) throw new ArgumentOutOfRangeException(nameof(warmup));
        if (iterations <= 0) throw new ArgumentOutOfRangeException(nameof(iterations));

        for (int i = 0; i < warmup; i++) launch();
        Synchronize();

        using var _ = Enter();
        Check(CudaNativeBindings.cuEventCreate(out IntPtr start, CudaNativeBindings.CU_EVENT_DEFAULT), "cuEventCreate(start)");
        Check(CudaNativeBindings.cuEventCreate(out IntPtr stop, CudaNativeBindings.CU_EVENT_DEFAULT), "cuEventCreate(stop)");
        try
        {
            Check(CudaNativeBindings.cuEventRecord(start, _stream), "cuEventRecord(start)");
            for (int i = 0; i < iterations; i++) launch();
            Check(CudaNativeBindings.cuEventRecord(stop, _stream), "cuEventRecord(stop)");
            Check(CudaNativeBindings.cuEventSynchronize(stop), "cuEventSynchronize(stop)");
            Check(CudaNativeBindings.cuEventElapsedTime(out float elapsed, start, stop), "cuEventElapsedTime");
            return elapsed / iterations;
        }
        finally
        {
            CudaNativeBindings.cuEventDestroy(start);
            CudaNativeBindings.cuEventDestroy(stop);
        }
    }

    /// <summary>
    /// Returns a distribution of CUDA-event device times. Each sample is an
    /// average of a small back-to-back launch group, which keeps event-record
    /// overhead from dominating kernels in the 10-microsecond range while
    /// still exposing run-to-run p95/p99 variation.
    /// </summary>
    internal float[] MeasureKernelSamples(
        Action launch, int warmup, int samples, int launchesPerSample)
    {
        PtxCompat.ThrowIfNull(launch, nameof(launch));
        if (warmup < 0) throw new ArgumentOutOfRangeException(nameof(warmup));
        if (samples <= 0) throw new ArgumentOutOfRangeException(nameof(samples));
        if (launchesPerSample <= 0) throw new ArgumentOutOfRangeException(nameof(launchesPerSample));

        for (int i = 0; i < warmup; i++) launch();
        Synchronize();

        var starts = new IntPtr[samples];
        var stops = new IntPtr[samples];
        var result = new float[samples];
        using var _ = Enter();
        try
        {
            for (int i = 0; i < samples; i++)
            {
                Check(CudaNativeBindings.cuEventCreate(out starts[i], CudaNativeBindings.CU_EVENT_DEFAULT),
                    "cuEventCreate(sample start)");
                Check(CudaNativeBindings.cuEventCreate(out stops[i], CudaNativeBindings.CU_EVENT_DEFAULT),
                    "cuEventCreate(sample stop)");
            }

            for (int sample = 0; sample < samples; sample++)
            {
                Check(CudaNativeBindings.cuEventRecord(starts[sample], _stream), "cuEventRecord(sample start)");
                for (int launchIndex = 0; launchIndex < launchesPerSample; launchIndex++) launch();
                Check(CudaNativeBindings.cuEventRecord(stops[sample], _stream), "cuEventRecord(sample stop)");
            }

            Check(CudaNativeBindings.cuEventSynchronize(stops[^1]), "cuEventSynchronize(samples)");
            for (int sample = 0; sample < samples; sample++)
            {
                Check(CudaNativeBindings.cuEventElapsedTime(
                    out float elapsed, starts[sample], stops[sample]), "cuEventElapsedTime(sample)");
                result[sample] = elapsed / launchesPerSample;
            }
            return result;
        }
        finally
        {
            for (int i = 0; i < samples; i++)
            {
                if (starts[i] != IntPtr.Zero) CudaNativeBindings.cuEventDestroy(starts[i]);
                if (stops[i] != IntPtr.Zero) CudaNativeBindings.cuEventDestroy(stops[i]);
            }
        }
    }

    internal static void Check(CudaResult result, string operation)
    {
        if (result != CudaResult.Success)
            throw new InvalidOperationException($"{operation} failed with CUDA driver status {(int)result} ({result}).");
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        if (_ownsContext && _context != IntPtr.Zero)
        {
            CuBlasNative.cuCtxDestroy(_context);
        }
        _context = IntPtr.Zero;
    }

    internal readonly struct ContextScope : IDisposable
    {
        private readonly bool _pushed;
        private readonly bool _enteredNewContext;
        private readonly IntPtr _previousTrackedContext;
        private readonly int _previousTrackedDepth;

        internal ContextScope(IntPtr context)
        {
            _previousTrackedContext = s_scopedContext;
            _previousTrackedDepth = s_scopeDepth;
            if (s_scopedContext == context && s_scopeDepth > 0)
            {
                s_scopeDepth++;
                _pushed = false;
                _enteredNewContext = false;
                return;
            }

            Check(CudaNativeBindings.cuCtxGetCurrent(out IntPtr current), "cuCtxGetCurrent");
            if (current == context)
            {
                _pushed = false;
            }
            else
            {
                Check(CuBlasNative.cuCtxPushCurrent(context), "cuCtxPushCurrent");
                _pushed = true;
            }
            s_scopedContext = context;
            s_scopeDepth = 1;
            _enteredNewContext = true;
        }

        public void Dispose()
        {
            if (!_enteredNewContext)
            {
                s_scopeDepth--;
                return;
            }

            s_scopedContext = _previousTrackedContext;
            s_scopeDepth = _previousTrackedDepth;
            if (_pushed)
                Check(CuBlasNative.cuCtxPopCurrent(out _), "cuCtxPopCurrent");
        }
    }
}

internal sealed class DirectPtxBuffer : IDisposable
{
    private readonly DirectPtxRuntime _runtime;
    private IntPtr _pointer;

    internal IntPtr Pointer => _pointer;
    internal nuint ByteLength { get; }

    internal DirectPtxBuffer(DirectPtxRuntime runtime, IntPtr pointer, nuint byteLength)
    {
        _runtime = runtime;
        _pointer = pointer;
        ByteLength = byteLength;
    }

    internal unsafe void Upload<T>(ReadOnlySpan<T> source) where T : unmanaged
    {
        nuint bytes = checked((nuint)source.Length * (nuint)sizeof(T));
        if (bytes > ByteLength) throw new ArgumentException("Source is larger than the device buffer.", nameof(source));
        using var _ = _runtime.Enter();
        fixed (T* pSource = source)
            DirectPtxRuntime.Check(
                CudaNativeBindings.cuMemcpyHtoD(_pointer, (IntPtr)pSource, checked((ulong)bytes)),
                "cuMemcpyHtoD");
    }

    internal unsafe void Download<T>(Span<T> destination) where T : unmanaged
    {
        nuint bytes = checked((nuint)destination.Length * (nuint)sizeof(T));
        if (bytes > ByteLength) throw new ArgumentException("Destination is larger than the device buffer.", nameof(destination));
        using var _ = _runtime.Enter();
        fixed (T* pDestination = destination)
            DirectPtxRuntime.Check(
                CudaNativeBindings.cuMemcpyDtoH((IntPtr)pDestination, _pointer, checked((ulong)bytes)),
                "cuMemcpyDtoH");
    }

    public void Dispose()
    {
        if (_pointer == IntPtr.Zero) return;
        using var _ = _runtime.Enter();
        DirectPtxRuntime.Check(CudaNativeBindings.cuMemFree(_pointer), "cuMemFree");
        _pointer = IntPtr.Zero;
    }
}

internal sealed class DirectPtxModule : IDisposable
{
    private readonly DirectPtxRuntime _runtime;
    private IntPtr _module;
    internal string JitInfoLog { get; }

    internal DirectPtxModule(DirectPtxRuntime runtime, IntPtr module, string jitInfoLog)
    {
        _runtime = runtime;
        _module = module;
        JitInfoLog = jitInfoLog;
    }

    internal IntPtr GetFunction(string name)
        => GetFunction(name, out _);

    internal IntPtr GetFunction(string name, out DirectPtxFunctionInfo info)
    {
        using var _ = _runtime.Enter();
        DirectPtxRuntime.Check(
            CudaNativeBindings.cuModuleGetFunction(out IntPtr function, _module, name),
            $"cuModuleGetFunction({name})");
        info = DirectPtxFunctionInfo.Query(function);
        if (info.LocalBytesPerThread != 0)
            throw new InvalidOperationException(
                $"Direct PTX kernel '{name}' was rejected: CUDA JIT allocated " +
                $"{info.LocalBytesPerThread} local bytes/thread (register spill or local stack). " +
                "The direct-kernel contract requires zero local memory.");
        return function;
    }

    internal unsafe void Launch(
        IntPtr function,
        uint gridX, uint gridY, uint gridZ,
        uint blockX, uint blockY, uint blockZ,
        uint sharedMemoryBytes,
        void** arguments)
    {
        using var _ = _runtime.Enter();
        DirectPtxRuntime.Check(
            CudaNativeBindings.cuLaunchKernel(
                function,
                gridX, gridY, gridZ,
                blockX, blockY, blockZ,
                sharedMemoryBytes,
                _runtime.Stream,
                (IntPtr)arguments,
                IntPtr.Zero),
            "cuLaunchKernel(PTX)");
    }

    internal int GetActiveBlocksPerMultiprocessor(
        IntPtr function,
        int blockThreads,
        nuint dynamicSharedBytes = 0)
    {
        using var _ = _runtime.Enter();
        DirectPtxRuntime.Check(
            CudaNativeBindings.cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
                out int blocks, function, blockThreads, checked((nint)dynamicSharedBytes), 0),
            "cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags");
        return blocks;
    }

    public void Dispose()
    {
        if (_module == IntPtr.Zero) return;
        using var _ = _runtime.Enter();
        DirectPtxRuntime.Check(CudaNativeBindings.cuModuleUnload(_module), "cuModuleUnload");
        _module = IntPtr.Zero;
    }
}

internal readonly record struct DirectPtxFunctionInfo(
    int MaxThreadsPerBlock,
    int StaticSharedBytes,
    int ConstBytes,
    int LocalBytesPerThread,
    int RegistersPerThread,
    int PtxVersion,
    int BinaryVersion)
{
    internal static DirectPtxFunctionInfo Query(IntPtr function)
    {
        static int Get(IntPtr f, CudaFunctionAttribute attribute)
        {
            DirectPtxRuntime.Check(
                CudaNativeBindings.cuFuncGetAttribute(out int value, attribute, f),
                $"cuFuncGetAttribute({attribute})");
            return value;
        }

        return new DirectPtxFunctionInfo(
            Get(function, CudaFunctionAttribute.MaxThreadsPerBlock),
            Get(function, CudaFunctionAttribute.SharedSizeBytes),
            Get(function, CudaFunctionAttribute.ConstSizeBytes),
            Get(function, CudaFunctionAttribute.LocalSizeBytes),
            Get(function, CudaFunctionAttribute.NumRegisters),
            Get(function, CudaFunctionAttribute.PtxVersion),
            Get(function, CudaFunctionAttribute.BinaryVersion));
    }
}
