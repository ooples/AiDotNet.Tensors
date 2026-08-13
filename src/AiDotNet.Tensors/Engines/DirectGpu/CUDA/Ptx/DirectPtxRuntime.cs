using System;
using System.Collections.Generic;
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
    private readonly bool _ownsStream;
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
    internal uint StreamFlags
    {
        get
        {
            using var _ = Enter();
            Check(CudaNativeBindings.cuStreamGetFlags(_stream, out uint flags),
                "cuStreamGetFlags");
            return flags;
        }
    }

    internal static bool IsAvailable => CudaNativeBindings.IsAvailable;

    internal DirectPtxRuntime(int deviceOrdinal = 0)
    {
        uint contextScheduling = CudaContextScheduling.ResolveFromEnvironment();
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
        // Standalone buffers use the synchronous Driver-API copy calls, which execute
        // in the legacy default-stream ordering domain. A blocking stream preserves
        // copy-before-launch ordering; CU_STREAM_NON_BLOCKING would be independent
        // and can let a freshly launched block observe pre-upload allocation contents.
        Check(CudaNativeBindings.cuStreamCreate(out _stream, CudaNativeBindings.CU_STREAM_DEFAULT),
            "cuStreamCreate(blocking)");
        _ownsStream = true;
        // cuCtxCreate makes the context current. Detach it so every operation
        // below has an explicit, balanced push/pop boundary.
        Check(CuBlasNative.cuCtxPopCurrent(out IntPtr popped), "cuCtxPopCurrent");
        if (popped != _context)
        {
            CuBlasNative.cuCtxDestroy(_context);
            _context = IntPtr.Zero;
            throw new InvalidOperationException("CUDA returned a different context from cuCtxPopCurrent.");
        }

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
        _ownsStream = false;

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

    internal DirectPtxModule LoadModule(string ptx, bool allowExperimentalJitFallback = false)
    {
        if (string.IsNullOrWhiteSpace(ptx)) throw new ArgumentException("PTX cannot be empty.", nameof(ptx));
        using var _ = Enter();
        DirectPtxCubinArtifact artifact;
        try
        {
            artifact = DirectPtxCubinArtifactCache.Resolve(this, ptx);
        }
        catch (EntryPointNotFoundException) when (allowExperimentalJitFallback)
        {
            return LoadJitModule(ptx);
        }
        IntPtr image = Marshal.AllocHGlobal(artifact.Image.Length);
        try
        {
            Marshal.Copy(artifact.Image, 0, image, artifact.Image.Length);
            Check(CudaNativeBindings.cuModuleLoadData(out IntPtr module, image),
                "cuModuleLoadData(compiled direct-PTX cubin)");
            return new DirectPtxModule(this, module, artifact);
        }
        finally
        {
            Marshal.FreeHGlobal(image);
        }
    }

    private unsafe DirectPtxModule LoadJitModule(string ptx)
    {
        IntPtr text = Marshal.StringToHGlobalAnsi(ptx);
        const int logBytes = 16 * 1024;
        IntPtr errorLog = Marshal.AllocHGlobal(logBytes);
        IntPtr infoLog = Marshal.AllocHGlobal(logBytes);
        try
        {
            new Span<byte>((void*)errorLog, logBytes).Clear();
            new Span<byte>((void*)infoLog, logBytes).Clear();
            int[] options = [3, 4, 5, 6];
            IntPtr[] values = [infoLog, (IntPtr)logBytes, errorLog, (IntPtr)logBytes];
            CudaResult result = CudaNativeBindings.cuModuleLoadDataEx(
                out IntPtr module, text, (uint)options.Length, options, values);
            if (result != CudaResult.Success)
            {
                string error = Marshal.PtrToStringAnsi(errorLog) ?? string.Empty;
                string info = Marshal.PtrToStringAnsi(infoLog) ?? string.Empty;
                throw new InvalidOperationException(
                    $"Experimental cuModuleLoadDataEx(PTX) fallback failed with CUDA driver status " +
                    $"{(int)result} ({result}).\nJIT error log:\n{error}\nJIT info log:\n{info}");
            }
            return new DirectPtxModule(
                this, module, Marshal.PtrToStringAnsi(infoLog) ?? string.Empty);
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
        Check(CudaNativeBindings.cuStreamSynchronize(_stream), "cuStreamSynchronize");
    }

    internal DirectPtxGraph CaptureGraph(Action launch)
    {
        PtxCompat.ThrowIfNull(launch, nameof(launch));
        if (_stream == IntPtr.Zero)
            throw new NotSupportedException(
                "CUDA graph capture requires an explicit non-default stream.");

        using var _ = Enter();
        Check(CudaNativeBindings.cuStreamBeginCapture(
            _stream, CudaNativeBindings.CU_STREAM_CAPTURE_MODE_THREAD_LOCAL),
            "cuStreamBeginCapture");
        IntPtr graph = IntPtr.Zero;
        bool endCaptureCalled = false;
        try
        {
            launch();
            // EndCapture terminates the capture even when it reports an error. Mark
            // the attempt first so no failure below can issue a second EndCapture.
            endCaptureCalled = true;
            Check(CudaNativeBindings.cuStreamEndCapture(_stream, out graph),
                "cuStreamEndCapture");
            Check(CudaNativeBindings.cuGraphInstantiate(
                out IntPtr graphExec, graph, 0), "cuGraphInstantiate");
            return new DirectPtxGraph(this, graphExec);
        }
        catch
        {
            if (!endCaptureCalled &&
                CudaNativeBindings.cuStreamEndCapture(_stream, out IntPtr aborted) ==
                    CudaResult.Success &&
                aborted != IntPtr.Zero)
                CudaNativeBindings.cuGraphDestroy(aborted);
            throw;
        }
        finally
        {
            if (graph != IntPtr.Zero) CudaNativeBindings.cuGraphDestroy(graph);
        }
    }

    internal void LaunchGraph(IntPtr graphExec)
    {
        using var _ = Enter();
        Check(CudaNativeBindings.cuGraphLaunch(graphExec, _stream), "cuGraphLaunch");
    }

    internal void DestroyGraph(IntPtr graphExec)
    {
        if (graphExec == IntPtr.Zero || _context == IntPtr.Zero) return;
        using var _ = Enter();
        CudaNativeBindings.cuGraphExecDestroy(graphExec);
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

    /// <summary>
    /// Measures a microkernel geometry from a captured multi-launch graph so
    /// host submission gaps cannot dominate the tuner. The returned values are
    /// per-kernel milliseconds, matching <see cref="MeasureKernelSamples"/>.
    /// </summary>
    internal float[] MeasureCapturedKernelSamples(
        Action launch, int warmup, int samples, int launchesPerSample)
    {
        PtxCompat.ThrowIfNull(launch, nameof(launch));
        if (warmup < 0) throw new ArgumentOutOfRangeException(nameof(warmup));
        if (samples <= 0) throw new ArgumentOutOfRangeException(nameof(samples));
        if (launchesPerSample <= 0) throw new ArgumentOutOfRangeException(nameof(launchesPerSample));

        using var _ = Enter();
        IntPtr graph = IntPtr.Zero;
        IntPtr graphExec = IntPtr.Zero;
        IntPtr start = IntPtr.Zero;
        IntPtr stop = IntPtr.Zero;
        bool captureActive = false;
        try
        {
            Check(CudaNativeBindings.cuEventCreate(out start, CudaNativeBindings.CU_EVENT_DEFAULT),
                "cuEventCreate(tuner start)");
            Check(CudaNativeBindings.cuEventCreate(out stop, CudaNativeBindings.CU_EVENT_DEFAULT),
                "cuEventCreate(tuner stop)");
            Check(CudaNativeBindings.cuStreamBeginCapture(
                _stream, CudaNativeBindings.CU_STREAM_CAPTURE_MODE_THREAD_LOCAL),
                "cuStreamBeginCapture(tuner)");
            captureActive = true;
            Check(CudaNativeBindings.cuEventRecordWithFlags(
                start, _stream, CudaNativeBindings.CU_EVENT_RECORD_EXTERNAL),
                "cuEventRecordWithFlags(tuner start)");
            for (int i = 0; i < launchesPerSample; i++) launch();
            Check(CudaNativeBindings.cuEventRecordWithFlags(
                stop, _stream, CudaNativeBindings.CU_EVENT_RECORD_EXTERNAL),
                "cuEventRecordWithFlags(tuner stop)");
            CudaResult endResult = CudaNativeBindings.cuStreamEndCapture(_stream, out graph);
            captureActive = false;
            Check(endResult, "cuStreamEndCapture(tuner)");
            if (graph == IntPtr.Zero)
                throw new InvalidOperationException("CUDA tuner capture returned a null graph.");
            Check(CudaNativeBindings.cuGraphInstantiate(out graphExec, graph, 0),
                "cuGraphInstantiate(tuner)");
            CudaNativeBindings.cuGraphDestroy(graph);
            graph = IntPtr.Zero;

            for (int i = 0; i < warmup; i++)
                Check(CudaNativeBindings.cuGraphLaunch(graphExec, _stream), "cuGraphLaunch(tuner warmup)");
            Synchronize();

            var result = new float[samples];
            for (int sample = 0; sample < samples; sample++)
            {
                Check(CudaNativeBindings.cuGraphLaunch(graphExec, _stream), "cuGraphLaunch(tuner sample)");
                Check(CudaNativeBindings.cuEventSynchronize(stop), "cuEventSynchronize(tuner stop)");
                Check(CudaNativeBindings.cuEventElapsedTime(out float elapsed, start, stop),
                    "cuEventElapsedTime(tuner)");
                result[sample] = elapsed / launchesPerSample;
            }
            return result;
        }
        finally
        {
            if (captureActive)
            {
                CudaNativeBindings.cuStreamEndCapture(_stream, out IntPtr aborted);
                if (aborted != IntPtr.Zero) CudaNativeBindings.cuGraphDestroy(aborted);
            }
            if (start != IntPtr.Zero) CudaNativeBindings.cuEventDestroy(start);
            if (stop != IntPtr.Zero) CudaNativeBindings.cuEventDestroy(stop);
            if (graphExec != IntPtr.Zero) CudaNativeBindings.cuGraphExecDestroy(graphExec);
            if (graph != IntPtr.Zero) CudaNativeBindings.cuGraphDestroy(graph);
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
        if (_ownsStream && _stream != IntPtr.Zero && _context != IntPtr.Zero)
        {
            using var _ = Enter();
            CudaNativeBindings.cuStreamDestroy(_stream);
        }
        if (_ownsContext && _context != IntPtr.Zero)
        {
            CuBlasNative.cuCtxDestroy(_context);
        }
        _context = IntPtr.Zero;
        _disposed = true;
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

internal sealed class DirectPtxGraph : IDisposable
{
    private readonly DirectPtxRuntime _runtime;
    private IntPtr _graphExec;

    internal DirectPtxGraph(DirectPtxRuntime runtime, IntPtr graphExec)
    {
        _runtime = runtime;
        _graphExec = graphExec;
    }

    internal void Launch()
    {
        if (_graphExec == IntPtr.Zero)
            throw new ObjectDisposedException(nameof(DirectPtxGraph));
        _runtime.LaunchGraph(_graphExec);
    }

    public void Dispose()
    {
        if (_graphExec == IntPtr.Zero) return;
        _runtime.DestroyGraph(_graphExec);
        _graphExec = IntPtr.Zero;
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
        // Do not let a host write race earlier work on this runtime's stream.
        // A stream-local barrier preserves ordering without stalling unrelated
        // streams or other contexts on the device.
        _runtime.Synchronize();
        fixed (T* pSource = source)
        {
            DirectPtxRuntime.Check(
                CudaNativeBindings.cuMemcpyHtoD(_pointer, (IntPtr)pSource, checked((ulong)bytes)),
                "cuMemcpyHtoD");
            // Standalone runtimes launch on a CU_STREAM_NON_BLOCKING stream.
            // A synchronous pageable-host copy is issued in the null-stream
            // ordering domain and therefore does not establish an edge to that
            // stream. Complete the transfer before the caller can enqueue a
            // kernel, or concurrent contexts can observe an incompletely staged
            // input and leave apparently random output blocks at zero.
            DirectPtxRuntime.Check(
                CudaNativeBindings.cuCtxSynchronize(), "cuCtxSynchronize(upload)");
            // The synchronous pageable-host copy stages through the default
            // stream. Complete that stream before a caller can enqueue new work
            // on the runtime's CU_STREAM_NON_BLOCKING stream.
            DirectPtxRuntime.Check(
                CudaNativeBindings.cuStreamSynchronize(IntPtr.Zero),
                "cuStreamSynchronize(upload staging)");
        }
    }

    internal unsafe void Download<T>(Span<T> destination) where T : unmanaged
    {
        nuint bytes = checked((nuint)destination.Length * (nuint)sizeof(T));
        if (bytes > ByteLength) throw new ArgumentException("Destination is larger than the device buffer.", nameof(destination));
        using var _ = _runtime.Enter();
        // The null-stream DtoH copy does not wait for work in the runtime's
        // non-blocking stream. Make Download independently correct even when a
        // caller omits an explicit Synchronize before reading the result.
        DirectPtxRuntime.Check(
            CudaNativeBindings.cuCtxSynchronize(), "cuCtxSynchronize(download)");
        // Make Download independently correct when the caller omits an explicit
        // barrier, while waiting only for the stream that produces this buffer.
        _runtime.Synchronize();
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
    private const uint DefaultDynamicSharedMemoryLimitBytes = 48 * 1024;
    private readonly DirectPtxRuntime _runtime;
    private readonly object _dynamicSharedMemoryLock = new();
    private readonly Dictionary<IntPtr, int> _dynamicSharedMemoryLimits = new();
    private IntPtr _module;
    internal string JitInfoLog { get; }
    internal DirectPtxModuleImageKind ImageKind { get; }
    internal string CubinSha256 { get; }
    internal string CubinSourceKey { get; }
    internal string? CubinPath { get; }

    internal DirectPtxModule(
        DirectPtxRuntime runtime, IntPtr module, DirectPtxCubinArtifact artifact)
    {
        _runtime = runtime;
        _module = module;
        JitInfoLog = artifact.CompilerLog;
        ImageKind = artifact.ImageKind;
        CubinSha256 = artifact.CubinSha256;
        CubinSourceKey = artifact.SourceKey;
        CubinPath = artifact.Path;
    }

    internal DirectPtxModule(
        DirectPtxRuntime runtime, IntPtr module, string experimentalJitInfoLog)
    {
        _runtime = runtime;
        _module = module;
        JitInfoLog = experimentalJitInfoLog;
        ImageKind = DirectPtxModuleImageKind.DriverJitPtx;
        CubinSha256 = string.Empty;
        CubinSourceKey = string.Empty;
        CubinPath = null;
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
        // CUDA requires a per-function opt-in above the portable 48 KiB default.
        // Keep this at the common launch boundary so an emitter cannot expose a
        // correct byte count while a caller accidentally omits the matching attribute.
        if (sharedMemoryBytes > DefaultDynamicSharedMemoryLimitBytes)
            SetMaxDynamicSharedMemory(function, checked((int)sharedMemoryBytes));

        using var _ = _runtime.Enter();
        LaunchCurrentContext(
            function, gridX, gridY, gridZ, blockX, blockY, blockZ,
            sharedMemoryBytes, arguments);
    }

    /// <summary>
    /// Launches after the owning backend has established the CUDA context on
    /// the calling thread. Driver-only callers use <see cref="Launch"/>; this
    /// entry point lets validated resident dispatch avoid a second
    /// cuCtxGetCurrent call for every kernel submission.
    /// </summary>
    internal unsafe void LaunchCurrentContext(
        IntPtr function,
        uint gridX, uint gridY, uint gridZ,
        uint blockX, uint blockY, uint blockZ,
        uint sharedMemoryBytes,
        void** arguments)
    {
        PtxCompat.ThrowIfDisposed(_module == IntPtr.Zero, this);
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

    internal void SetMaxDynamicSharedMemory(IntPtr function, int bytes)
    {
        if (bytes < 0) throw new ArgumentOutOfRangeException(nameof(bytes));
        lock (_dynamicSharedMemoryLock)
        {
            if (_dynamicSharedMemoryLimits.TryGetValue(function, out int configured) &&
                configured >= bytes)
                return;

            using var _ = _runtime.Enter();
            DirectPtxRuntime.Check(
                CudaNativeBindings.cuFuncSetAttribute(
                    function, CudaFunctionAttribute.MaxDynamicSharedSizeBytes, bytes),
                "cuFuncSetAttribute(MaxDynamicSharedSizeBytes)");
            _dynamicSharedMemoryLimits[function] = bytes;
        }
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
