// Copyright (c) AiDotNet. All rights reserved.
// Direct CUDA backend for NVIDIA GPUs (Driver API + NVRTC + cuBLAS fallback).
using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.IO;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Kernels;
using AiDotNet.Tensors.Engines.Gpu;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA;

public sealed partial class CudaBackend : IAsyncGpuBackend, IFusedAdvancedKernels, AiDotNet.Tensors.Engines.Gpu.IGpuHalfPrecisionBackend
{
    // During teardown the CUDA driver/context may already be destroyed, so calling driver
    // APIs (cuCtxPushCurrent / cuMemFree / cuCtxPopCurrent / cuCtxDestroy / cuModuleUnload)
    // from a finalizer raises an access violation (0xC0000005) — a Corrupted-State
    // Exception that catch cannot trap and that is fatal on the finalizer thread, killing
    // the whole process. The OS reclaims device memory + the context at exit anyway.
    private static bool IsRuntimeTearingDown => RuntimeShutdown.IsTearingDown;

    // Live CUDA contexts. A device-buffer finalizer must NOT free against a context that has already
    // been destroyed — under per-test engine teardown (each test builds its own CudaBackend/context),
    // a buffer that escapes explicit Dispose is GC-finalized LATER and would call cuCtxPushCurrent /
    // cuMemFree on a dead context, raising an uncatchable, finalizer-fatal 0xC0000005 that kills the
    // whole process (the documented "no suite crash" hazard — IsRuntimeTearingDown only covers PROCESS
    // exit, not per-test context destruction). cuCtxDestroy already reclaimed that buffer's memory, so
    // the finalizer skips the native free when its context is no longer registered here. Added at
    // cuCtxCreate, removed immediately before cuCtxDestroy.
    internal static readonly ConcurrentDictionary<IntPtr, byte> LiveContexts = new();

    // Serializes a buffer's native free against its context's destruction. The LiveContexts check
    // alone is a TOCTOU race: a GC-thread buffer finalizer can observe its context as live, then a
    // concurrent engine Dispose removes + cuCtxDestroys it, and the finalizer's cuCtxPushCurrent /
    // cuMemFree then hit the just-freed context → uncatchable 0xC0000005. Holding this lock across
    // {check-live + free} on the finalizer side and {remove + cuCtxDestroy} on the dispose side makes
    // the two mutually exclusive: a buffer either frees before the destroy or is skipped after it.
    internal static readonly object ContextLifecycleLock = new();

    private const int DefaultBlockSize = 256;
    private const int MaxRnnBlockSize = 1024;
    private readonly ConcurrentDictionary<string, IntPtr> _kernelCache;
    private IntPtr _cudaContext;
    private IntPtr _stream;
    private IntPtr _cublasHandle;
    // Stream-ordered allocator (#558 layer 6): when enabled, device buffers are allocated/freed via
    // cuMemAllocAsync/cuMemFreeAsync on _stream, so freed memory is reused in stream order without a host
    // sync (race-free + bounds mid-step peak). DEFAULT ON; opt out with AIDOTNET_CUDA_ASYNC_ALLOC=0
    // (the legacy synchronous allocator, which can hit a sticky CUDA-700 under heavy buffer churn).
    private bool _asyncAlloc;
    private IntPtr _asyncMemPool; // the stream-ordered default mem pool (when _asyncAlloc); trimmed on sync-path OOM
    /// <summary>Diagnostic (#558): the nvrtc error if the FP16 kernel module failed to compile (which
    /// silently disables the whole FP16 GPU path). Null when FP16 compiled successfully.</summary>
    internal static string? Fp16ModuleCompileError;
    // cuDNN helpers — lazily initialized on first Conv2D call that routes
    // through the cuDNN dispatch path. Kept private so disposal is linked
    // to this backend's lifetime; the helpers' cuDNN handles survive
    // repeated kernel invocations (workspace reuse + descriptor cache).
    private CuDnnContext? _cudnnContext;
    private CuDnnConvolution? _cudnnConv;
    // Guards EnsureCudnnConv lazy-init against concurrent first callers.
    // Without this, two threads entering EnsureCudnnConv simultaneously
    // can both observe _cudnnConv == null and construct separate
    // CuDnnContext/CuDnnConvolution instances — leaking the losing one
    // and ending up with two threads working against different contexts.
    private readonly object _cudnnInitLock = new();
    private CudaStream? _defaultStream;
    // True iff the fused-advanced kernel module compiled and registered all
    // its kernels successfully. Public surface methods gate on this so a
    // partial / failed compile surfaces a clear NotSupportedException at the
    // call site instead of an opaque "kernel not found" deep in dispatch.
    private bool _fusedAdvancedKernelsAvailable;
    private IntPtr _activationModule;
    private IntPtr _convolutionModule;
    private IntPtr _fusedConvolutionModule;
    private IntPtr _poolingModule;
    private IntPtr _normalizationModule;
    private IntPtr _neuralNetModule;
    private IntPtr _fusedModule;
    private IntPtr _attentionModule;
    private IntPtr _fftModule;
    private IntPtr _spectralPerfModule;
    private IntPtr _spatialTransformerModule;
    private IntPtr _sparseModule;
    private IntPtr _locallyConnectedModule;
    private IntPtr _deformableConvModule;
    private IntPtr _capsuleModule;
    private IntPtr _specializedModule;
    private IntPtr _lstmModule;
    private IntPtr _glaModule; // fused GLA scan kernels (#1464)
    private IntPtr _xlstmModule; // fused xLSTM scan kernels (#1464)
    private IntPtr _gatedDeltaNetModule; // fused Gated DeltaNet scan kernels (#1464)
    private IntPtr _rgLruModule; // fused RG-LRU scan kernels (#1464)
    private IntPtr _rwkv4Module; // fused RWKV-4 WKV scan kernels (#1464)
    private IntPtr _mambaModule; // fused Mamba selective scan kernels (#1464)
    private IntPtr _mamba2Module; // fused Mamba-2 SSD scan kernels (#1464)
    private IntPtr _fusedCeModule; // fused linear + cross-entropy kernels (#1464)
    private IntPtr _gruModule;
    private IntPtr _snnModule;
    private IntPtr _fp16Module;
    private IntPtr _parity210Module;
    private IntPtr _detectionModule;
    private IntPtr _geometryModule;
    private IntPtr _roiModule;
    private IntPtr _audioModule;
    private IntPtr _linalgModule;
    private bool _disposed;

    // Process-exit guard: at process/AppDomain teardown the CUDA driver may already be unloaded, so calling ANY
    // native CUDA API from a finalizer (CudaGpuBuffer.Release, CudaBackend.Dispose) FATAL-crashes the process — it
    // is NOT a catchable .NET exception. Skip native cleanup when exiting; the OS reclaims GPU memory on process death.
    internal static volatile bool ProcessExiting;
    static CudaBackend()
    {
        try
        {
            AppDomain.CurrentDomain.ProcessExit += (_, __) => ProcessExiting = true;
            AppDomain.CurrentDomain.DomainUnload += (_, __) => ProcessExiting = true;
        }
        catch { }
    }

    private const int MaxPooledBufferElements = 16_777_216;
    private const int MaxPooledBuffersPerSize = 4;
    private readonly GpuBufferPool<CudaGpuBuffer> _bufferPool =
        new GpuBufferPool<CudaGpuBuffer>(MaxPooledBuffersPerSize, MaxPooledBufferElements);
    // Event-based deferred buffer free (#555 / #226). A buffer evicted mid-step may still
    // be referenced by an in-flight async kernel; returning it to the pool (for reuse OR a
    // real cuMemFree when the bucket is full) before that kernel finishes is the CUDA-700
    // illegal-access race. Instead we record a lightweight (timing-disabled) event on the
    // compute stream and hold the buffer in this queue until the event completes — polled
    // NON-blocking (cuEventQuery) so there is no host stall (unlike a full cuStreamSynchronize).
    private readonly object _deferredFreeLock = new();
    private readonly Queue<(IGpuBuffer Buffer, CudaEvent Event)> _deferredFrees = new();
    private bool _supportsCooperativeLaunch;
    private int _multiProcessorCount;
    private readonly CudaPinnedBufferPool _pinnedPool = new();
    private IntPtr _wmmaModule;
    private int _ccMajor;
    private int _ccMinor;

    /// <summary>
    /// True when this device supports the bfloat16 GEMM / MHA fanout
    /// paths (BatchedGemmExFanout(useBFloat16:true) and
    /// MultiHeadAttentionScoresFanoutMixed(useBFloat16:true)). Requires
    /// compute capability ≥ 8.0 (Ampere+), where the BF16 tensor cores
    /// live. Distinct from IGpuMixedPrecisionConvBackend's BF16 conv
    /// surface: a host can have BF16 GEMM/MHA support without cuDNN
    /// (or with cuDNN-conv disabled), which is what makes a single
    /// capability flag wrong for cross-cutting test gates.
    /// </summary>
    public bool SupportsBFloat16GemmFanout => IsAvailable && _ccMajor >= 8;
    private int _clockRateKHz;
    private bool _hasWmmaSupport;

    [ThreadStatic]
    private static IntPtr _threadCurrentContext;

    public bool IsAvailable { get; }
    public string BackendName => "CUDA";
    public TensorDevice DeviceType => TensorDevice.CUDA;
    public string DeviceName { get; }
    public string DeviceVendor => "NVIDIA";
    public int ComputeUnits { get; }
    public long GlobalMemoryBytes { get; }
    public long LocalMemoryBytes { get; }
    /// <summary>
    /// CUDA does not expose a fixed per-allocation cap as a device attribute —
    /// in practice individual <c>cudaMalloc</c> calls are limited by the
    /// largest contiguous free VRAM block. We use the device's TOTAL VRAM
    /// (from <c>cuDeviceTotalMem</c>) as a conservative upper bound — a
    /// hard ceiling rather than a "current free memory" tracker. Issue #285.
    /// </summary>
    public long MaxBufferAllocBytes { get; }
    public double TheoreticalGflops { get; private set; }

    // IAsyncGpuBackend properties
    public bool SupportsMultiStream => true;
    public bool SupportsEvents => true;
    public bool SupportsAsyncTransfer => true;
    public bool SupportsGraphCapture => false; // CUDA graphs not yet implemented
    public int MaxConcurrentStreams => 16;
    public IGpuStream DefaultStream => _defaultStream ?? throw new InvalidOperationException("Backend not initialized");

    public static bool IsCudaAvailable => CudaNativeBindings.IsAvailable && NvrtcNativeBindings.IsAvailable;

    /// <summary>
    /// Ensures CUDA is available, throwing a detailed beginner-friendly exception if not.
    /// Call this method before attempting to use CUDA features to get helpful error messages.
    /// </summary>
    /// <param name="feature">Optional description of the feature requiring CUDA.</param>
    /// <exception cref="Exceptions.CudaNotFoundException">
    /// Thrown with detailed troubleshooting steps when CUDA is not available.
    /// </exception>
    /// <example>
    /// <code>
    /// // Check CUDA availability before use
    /// CudaBackend.EnsureAvailable("matrix multiplication");
    ///
    /// // Now safe to create the backend
    /// using var cuda = new CudaBackend();
    /// </code>
    /// </example>
    public static void EnsureAvailable(string feature = "NVIDIA GPU acceleration")
    {
        if (!IsCudaAvailable)
        {
            throw new Exceptions.CudaNotFoundException(feature);
        }
    }

    /// <summary>
    /// Creates a CudaBackend, throwing a detailed beginner-friendly exception if CUDA is not available.
    /// This is the recommended way to create a CudaBackend for beginners.
    /// </summary>
    /// <param name="deviceIndex">The CUDA device index (default: 0 for the first GPU).</param>
    /// <returns>A new CudaBackend instance.</returns>
    /// <exception cref="Exceptions.CudaNotFoundException">
    /// Thrown with detailed troubleshooting steps when CUDA is not available.
    /// </exception>
    public static CudaBackend CreateOrThrow(int deviceIndex = 0)
    {
        EnsureAvailable();
        var backend = new CudaBackend(deviceIndex);
        if (!backend.IsAvailable)
        {
            throw new Exceptions.CudaNotFoundException(Exceptions.CudaUnavailableReason.NoDevices,
                "CUDA backend initialization");
        }
        return backend;
    }

    public CudaBackend() : this(0)
    {
    }

    public CudaBackend(int deviceIndex)
    {
        _kernelCache = new ConcurrentDictionary<string, IntPtr>(StringComparer.Ordinal);

        if (!CudaNativeBindings.IsAvailable || !NvrtcNativeBindings.IsAvailable)
        {
            IsAvailable = false;
            DeviceName = "None";
            return;
        }

        try
        {
            CuBlasNative.CheckCudaResult(CuBlasNative.cuInit(0), "cuInit");
            CuBlasNative.CheckCudaResult(CuBlasNative.cuDeviceGet(out int device, deviceIndex), "cuDeviceGet");

            var nameBuilder = new StringBuilder(256);
            CuBlasNative.CheckCudaResult(
                CuBlasNative.cuDeviceGetName(nameBuilder, nameBuilder.Capacity, device),
                "cuDeviceGetName");
            DeviceName = nameBuilder.ToString();

            CuBlasNative.CheckCudaResult(
                CuBlasNative.cuDeviceGetAttribute(out int multiprocessors, (int)CudaDeviceAttribute.MultiprocessorCount, device),
                "cuDeviceGetAttribute(MultiprocessorCount)");
            ComputeUnits = multiprocessors;
            _multiProcessorCount = multiprocessors;

            // Check cooperative launch support (requires compute capability 6.0+)
            var coopResult = CuBlasNative.cuDeviceGetAttribute(
                out int coopLaunchSupport, (int)CudaDeviceAttribute.CooperativeLaunch, device);
            _supportsCooperativeLaunch = coopResult == CudaResult.Success && coopLaunchSupport != 0;

            CuBlasNative.CheckCudaResult(
                CuBlasNative.cuDeviceGetAttribute(out int sharedMem, (int)CudaDeviceAttribute.MaxSharedMemoryPerBlock, device),
                "cuDeviceGetAttribute(MaxSharedMemoryPerBlock)");
            LocalMemoryBytes = sharedMem;

            CuBlasNative.CheckCudaResult(CudaNativeBindings.cuDeviceTotalMem(out ulong totalMem, device), "cuDeviceTotalMem");
            GlobalMemoryBytes = (long)totalMem;
            // Issue #285: per-allocation cap. CUDA has no formal cap attribute;
            // practical limit is the largest contiguous free block. We use
            // total VRAM as the upper bound. NOTE: this means the cap guard
            // only catches obviously-impossible requests (> total VRAM). If
            // a smaller-than-VRAM allocation hits cudaErrorOutOfMemory due
            // to fragmentation, the existing CheckCudaResult path surfaces
            // it as a generic CUDA error — the chunker won't catch that.
            // Translating cuMemAlloc OOM → GpuBufferTooLargeException is
            // tracked as follow-up work.
            MaxBufferAllocBytes = (long)totalMem;

            CuBlasNative.CheckCudaResult(CuBlasNative.cuCtxCreate(out _cudaContext, 0, device), "cuCtxCreate");
            LiveContexts[_cudaContext] = 0; // register: a buffer finalizer may only free against a live context
            CuBlasNative.CheckCudaResult(CudaNativeBindings.cuStreamCreate(out _stream, 0), "cuStreamCreate");
            _defaultStream = new CudaStream(this, _stream, GpuStreamType.Default, ownsHandle: false);

            // Stream-ordered allocator setup (#558 layer 6). Retain freed memory in the pool for reuse
            // (caching-allocator behaviour) by maxing the release threshold; disable if the pool query
            // fails (older driver / no pool support) so AllocateBuffer falls back to the sync path.
            //
            // DEFAULT ON (opt out with AIDOTNET_CUDA_ASYNC_ALLOC=0). Stream-ordered alloc/free
            // (cuMemAllocAsync / cuMemFreeAsync on _stream) is REQUIRED for correctness under the async
            // training pipeline: the legacy synchronous custom-pool path frees device memory with cuMemFree
            // (on pool-bucket overflow and via the CudaGpuBuffer finalizer) the instant the host requests
            // it — but a kernel launched earlier may not have EXECUTED yet (work is queued async on
            // _stream). That kernel then accesses freed memory → a use-after-free that faults the context
            // with a STICKY CUDA 700 (illegal address). It reproduces only on large models under
            // VRAM-pressure buffer churn (the d768/L6 cortex) and vanishes entirely under
            // CUDA_LAUNCH_BLOCKING=1 — the signature of an async free-before-execute race. cuMemFreeAsync
            // is stream-ordered (the free is enqueued AFTER in-flight kernels on _stream), eliminating the
            // race by construction. Blocking stream FLAGS do NOT fix it (the free is host-side, not
            // stream-ordered); only stream-ordered freeing does.
            // Tri-state the env var so we never SILENTLY drop onto the unsafe sync path: null = default (on),
            // "0" = explicit opt-out (sync), anything else = explicit opt-in. The sync path is not merely
            // "slower" — it is the use-after-free described above, so a no-mempool box that lands on it is
            // still exposed to the sticky CUDA-700. We therefore surface the loss of the stream-ordered pool
            // rather than hide it: an explicit opt-in throws (the caller asked for a guarantee we can't honour),
            // and the default-on case emits a one-time warning before falling back (so older drivers still run,
            // but the operator is told why churn-heavy training may fault).
            string? asyncAllocSetting = Environment.GetEnvironmentVariable("AIDOTNET_CUDA_ASYNC_ALLOC");
            bool asyncExplicitlyRequested = asyncAllocSetting is not null && asyncAllocSetting != "0";
            _asyncAlloc = asyncAllocSetting != "0";
            if (_asyncAlloc)
            {
                if (CuBlasNative.cuDeviceGetDefaultMemPool(out var memPool, device) == CudaResult.Success)
                {
                    _asyncMemPool = memPool;
                    ulong threshold = ulong.MaxValue; // CU_MEMPOOL_ATTR_RELEASE_THRESHOLD = 4
                    var poolAttrStatus = CuBlasNative.cuMemPoolSetAttribute(memPool, 4, ref threshold);
                    if (poolAttrStatus != CudaResult.Success)
                    {
                        // Non-fatal: the pool still works, it just won't RETAIN freed blocks for reuse
                        // (it releases to the OS at each sync point) — a churn-performance loss, not a
                        // correctness one. Surface it instead of discarding the status.
                        System.Diagnostics.Trace.TraceWarning(
                            $"[CudaBackend] cuMemPoolSetAttribute(RELEASE_THRESHOLD) failed: {poolAttrStatus}. " +
                            "Stream-ordered pool caching is disabled; expect more allocator churn under load.");
                    }
                }
                else if (asyncExplicitlyRequested)
                {
                    // The caller explicitly demanded the race-free allocator but the driver has no
                    // stream-ordered mem-pool support — do NOT silently substitute the unsafe sync path.
                    throw new NotSupportedException(
                        "AIDOTNET_CUDA_ASYNC_ALLOC was set but this CUDA driver exposes no stream-ordered " +
                        "default memory pool (cuDeviceGetDefaultMemPool failed). The legacy synchronous " +
                        "allocator can free device memory before queued kernels execute (sticky CUDA-700 under " +
                        "buffer churn). Upgrade the driver, or set AIDOTNET_CUDA_ASYNC_ALLOC=0 to deliberately " +
                        "accept the synchronous allocator.");
                }
                else
                {
                    // Default-on, but no pool support. Fall back so older drivers still run, but make the
                    // residual use-after-free risk visible instead of pretending the default is safe here.
                    _asyncAlloc = false;
                    System.Diagnostics.Trace.TraceWarning(
                        "[CudaBackend] Stream-ordered CUDA memory pools are unavailable on this driver; " +
                        "falling back to the legacy synchronous allocator. Under heavy GPU buffer churn this " +
                        "can hit a sticky CUDA-700 use-after-free. Upgrade the driver to enable cuMemAllocAsync, " +
                        "or set AIDOTNET_CUDA_ASYNC_ALLOC=0 to silence this warning.");
                }
            }

            CuBlasNative.CheckCublasStatus(CuBlasNative.cublasCreate(out _cublasHandle), "cublasCreate");
            CuBlasNative.CheckCublasStatus(CuBlasNative.cublasSetStream(_cublasHandle, _stream), "cublasSetStream");

            // Math mode dispatch:
            //   - Ampere+ (compute capability ≥ 8.0) + AllowTF32 = true:
            //     CUBLAS_TF32_TENSOR_OP_MATH. Fp32 GEMMs run on the Tensor
            //     Cores with TF32 multiply + fp32 accumulate, ~5× the FLOPs
            //     of strict fp32. Loss curves match strict fp32 to within
            //     measurement noise on every NVIDIA-published benchmark
            //     (BERT, ResNet-50, Mask-R-CNN, T5).
            //   - Volta / Turing (compute capability < 8.0) OR AllowTF32 = false:
            //     legacy CUBLAS_TENSOR_OP_MATH, which gives the V100-class
            //     fp16 Tensor-Core path for HGEMM but leaves fp32 GEMMs at
            //     strict fp32.
            // Compute-capability check uses the value we just read; AllowTF32
            // honors AIDOTNET_DISABLE_TF32 env var + runtime override.
            var cc = GetComputeCapability(device);
            _ccMajor = cc.Major;
            _ccMinor = cc.Minor;

            bool useTF32 = _ccMajor >= 8 && CudaDispatchPolicy.AllowTF32;
            int mathMode = useTF32
                ? CuBlasNative.CUBLAS_TF32_TENSOR_OP_MATH
                : CuBlasNative.CUBLAS_TENSOR_OP_MATH;
            // Wrap in CheckCublasStatus so a math-mode set failure surfaces
            // immediately rather than leaving the handle in an unknown state.
            // Same pattern as cublasCreate / cublasSetStream above; a silent
            // failure here would make every subsequent fp32 GEMM run with
            // unspecified Tensor-Core behaviour (could be CUBLAS_DEFAULT_MATH,
            // strict fp32 with no acceleration; could be undefined).
            CuBlasNative.CheckCublasStatus(
                CuBlasNative.cublasSetMathMode(_cublasHandle, mathMode),
                "cublasSetMathMode");

            // Query clock rate for theoretical GFLOPS calculation
            if (CuBlasNative.cuDeviceGetAttribute(out int clockKHz, (int)CudaDeviceAttribute.ClockRate, device) == CudaResult.Success)
            {
                _clockRateKHz = clockKHz;
            }

            // Compute theoretical peak FP32 GFLOPS from hardware specs
            TheoreticalGflops = ComputeTheoreticalGflops(_ccMajor, _ccMinor, _multiProcessorCount, _clockRateKHz);

            CompileAllKernels(device);

            IsAvailable = true;

            // CUDA-graph feasibility/benefit probe (AIDOTNET_CUDA_GRAPH_PROBE=1): capture a pure-GPU op
            // sequence into a cudaGraph and compare graph-replay vs individual-launch latency. Quantifies
            // whether capture works on this stream and how much per-launch overhead replay removes — the
            // evidence basis for the CUDA-graph training-step work. Trace-only; no effect on normal runs.
            if (Environment.GetEnvironmentVariable("AIDOTNET_CUDA_GRAPH_PROBE") == "1")
                RunCudaGraphProbe();
        }
        catch (Exception ex)
        {
            System.Diagnostics.Debug.WriteLine($"CudaBackend initialization failed: {ex.GetType().Name}: {ex.Message}");
            System.Diagnostics.Trace.WriteLine($"CudaBackend initialization failed: {ex.GetType().Name}: {ex.Message}");
            DeviceName = "None";
            IsAvailable = false;
            Dispose();
        }
    }

    private readonly struct CudaContextScope : IDisposable
    {
        private readonly bool _pushed;

        public CudaContextScope(IntPtr context)
        {
            _pushed = false;
            // Push only if this thread doesn't already have the right context.
            if (context != IntPtr.Zero && _threadCurrentContext != context)
            {
                CuBlasNative.CheckCudaResult(CuBlasNative.cuCtxPushCurrent(context), "cuCtxPushCurrent");
                _threadCurrentContext = context;
                _pushed = true;
            }
        }

        public void Dispose()
        {
            if (_pushed)
            {
                CuBlasNative.CheckCudaResult(CuBlasNative.cuCtxPopCurrent(out _), "cuCtxPopCurrent");
                _threadCurrentContext = IntPtr.Zero;
            }
        }
    }

    private CudaContextScope PushContext()
    {
        return new CudaContextScope(_cudaContext);
    }

    private static string? GetCudaIncludePath()
    {
        // Check CUDA_PATH environment variable first
        var cudaPath = Environment.GetEnvironmentVariable("CUDA_PATH");
        if (string.IsNullOrEmpty(cudaPath) && RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            cudaPath = Environment.GetEnvironmentVariable("CUDA_PATH", EnvironmentVariableTarget.Machine);

        // Fall back to scanning standard install location
        if (string.IsNullOrEmpty(cudaPath) && RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
        {
            var basePath = Path.Combine(
                Environment.GetFolderPath(Environment.SpecialFolder.ProgramFiles),
                "NVIDIA GPU Computing Toolkit", "CUDA");
            if (Directory.Exists(basePath))
            {
                var versions = Directory.GetDirectories(basePath, "v*");
                Array.Sort(versions);
                if (versions.Length > 0)
                    cudaPath = versions[^1];
            }
        }

        if (string.IsNullOrEmpty(cudaPath))
            return null;

        var includePath = Path.Combine(cudaPath, "include");
        return Directory.Exists(includePath) ? includePath : null;
    }

    private static (int Major, int Minor) GetComputeCapability(int device)
    {
        CuBlasNative.CheckCudaResult(
            CuBlasNative.cuDeviceGetAttribute(out int major, (int)CudaDeviceAttribute.ComputeCapabilityMajor, device),
            "cuDeviceGetAttribute(ComputeCapabilityMajor)");
        CuBlasNative.CheckCudaResult(
            CuBlasNative.cuDeviceGetAttribute(out int minor, (int)CudaDeviceAttribute.ComputeCapabilityMinor, device),
            "cuDeviceGetAttribute(ComputeCapabilityMinor)");

        if (major <= 0)
            return (5, 2);

        return (major, minor);
    }

    /// <summary>
    /// Computes theoretical peak FP32 GFLOPS from GPU hardware specs.
    /// Formula: SMs * FP32_cores_per_SM * 2 (FMA) * clock_GHz
    /// </summary>
    private static double ComputeTheoreticalGflops(int ccMajor, int ccMinor, int smCount, int clockRateKHz)
    {
        if (smCount <= 0 || clockRateKHz <= 0)
        {
            // Fallback: conservative estimate
            return 8000;
        }

        // FP32 CUDA cores per SM by compute capability
        int coresPerSm = ccMajor switch
        {
            3 => 192,  // Kepler
            5 => 128,  // Maxwell
            6 => ccMinor == 0 ? 64 : 128, // Pascal (GP100 vs GP10x)
            7 => ccMinor == 0 ? 64 : 64,  // Volta/Turing
            8 => ccMinor == 0 ? 64 : 128,  // Ampere (GA100 vs GA10x)
            9 => 128,  // Hopper/Ada Lovelace
            10 => 128, // Blackwell
            _ => 128   // Future architectures
        };

        // GFLOPS = cores * 2 (FMA = multiply + add) * clock_GHz
        double clockGHz = clockRateKHz / 1_000_000.0;
        return smCount * coresPerSm * 2.0 * clockGHz;
    }

    private void CompileActivationKernels(int device)
    {
        using var _ = PushContext();

        string source = CudaActivationKernels.GetSource();
        string[] kernelNames = CudaActivationKernels.GetKernelNames();

        var (major, minor) = GetComputeCapability(device);
        string arch = $"--gpu-architecture=compute_{major}{minor}";
        string[] options = new[] { arch, "--use_fast_math" };

        IntPtr program = IntPtr.Zero;
        var result = NvrtcNativeBindings.nvrtcCreateProgram(
            ref program,
            source,
            "activation_kernels.cu",
            0,
            IntPtr.Zero,
            IntPtr.Zero);
        if (result != NvrtcResult.Success)
            throw new InvalidOperationException($"NVRTC program creation failed: {NvrtcNativeBindings.GetErrorString(result)}");

        result = NvrtcNativeBindings.nvrtcCompileProgram(program, options.Length, options);
        if (result != NvrtcResult.Success)
        {
            string log = GetNvrtcLog(program);
            NvrtcNativeBindings.nvrtcDestroyProgram(ref program);
            throw new InvalidOperationException($"NVRTC compile failed: {NvrtcNativeBindings.GetErrorString(result)}\n{log}");
        }

        result = NvrtcNativeBindings.nvrtcGetPTXSize(program, out UIntPtr ptxSize);
        if (result != NvrtcResult.Success || ptxSize == UIntPtr.Zero)
        {
            NvrtcNativeBindings.nvrtcDestroyProgram(ref program);
            throw new InvalidOperationException("NVRTC failed to return PTX size.");
        }

        IntPtr ptx = Marshal.AllocHGlobal((int)ptxSize);
        result = NvrtcNativeBindings.nvrtcGetPTX(program, ptx);
        NvrtcNativeBindings.nvrtcDestroyProgram(ref program);

        if (result != NvrtcResult.Success)
        {
            Marshal.FreeHGlobal(ptx);
            throw new InvalidOperationException($"NVRTC get PTX failed: {NvrtcNativeBindings.GetErrorString(result)}");
        }

        CuBlasNative.CheckCudaResult(CudaNativeBindings.cuModuleLoadData(out _activationModule, ptx), "cuModuleLoadData");
        Marshal.FreeHGlobal(ptx);

        foreach (var kernelName in kernelNames)
        {
            CuBlasNative.CheckCudaResult(
                CudaNativeBindings.cuModuleGetFunction(out IntPtr kernel, _activationModule, kernelName),
                $"cuModuleGetFunction({kernelName})");
            _kernelCache[kernelName] = kernel;
        }
    }

    /// <summary>
    /// Gets the disk cache directory for compiled CUBIN files.
    /// Returns null if caching is disabled or directory cannot be created.
    /// </summary>
    private static string? GetKernelCacheDirectory()
    {
        try
        {
            string cacheDir = Path.Combine(
                Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
                "AiDotNet", "Tensors", "kernel_cache");
            Directory.CreateDirectory(cacheDir);
            return cacheDir;
        }
        catch
        {
            return null;
        }
    }

    /// <summary>
    /// Computes a hash key for the kernel cache based on source code and compilation options.
    /// </summary>
    private static string ComputeCacheKey(string source, string arch)
    {
        using var sha = System.Security.Cryptography.SHA256.Create();
        byte[] hash = sha.ComputeHash(System.Text.Encoding.UTF8.GetBytes(source + "|" + arch));
        return BitConverter.ToString(hash).Replace("-", "").Substring(0, 32);
    }

    private IntPtr CompileKernelModule(int device, string source, string moduleName, string[] kernelNames)
    {
        // NVRTC doesn't support standard C headers — strip #include <math.h> etc.
        source = source.Replace("#include <math.h>", "// math.h stripped for NVRTC (built-in)")
                       .Replace("#include <float.h>", "// float.h stripped for NVRTC (built-in)")
                       .Replace("#include <stdio.h>", "// stdio.h stripped for NVRTC");

        const string nvrtcPreamble = @"
#ifndef INFINITY
#define INFINITY __int_as_float(0x7f800000)
#endif
#ifndef NAN
#define NAN __int_as_float(0x7fffffff)
#endif
#ifndef FLT_MAX
#define FLT_MAX 3.402823466e+38f
#endif
#ifndef FLT_MIN
#define FLT_MIN 1.175494351e-38f
#endif
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#ifndef M_PI_F
#define M_PI_F 3.14159265f
#endif
";
        source = nvrtcPreamble + source;

        var (major, minor) = GetComputeCapability(device);
        string arch = $"--gpu-architecture=sm_{major}{minor}";

        // Try loading from disk cache first
        string? cacheDir = GetKernelCacheDirectory();
        string cacheKey = ComputeCacheKey(source, arch);
        string? cacheFile = cacheDir != null ? Path.Combine(cacheDir, $"{moduleName}_{cacheKey}.cubin") : null;

        if (cacheFile != null && File.Exists(cacheFile))
        {
            try
            {
                byte[] cachedBinary = File.ReadAllBytes(cacheFile);
                IntPtr cachedPtr = Marshal.AllocHGlobal(cachedBinary.Length);
                try
                {
                    Marshal.Copy(cachedBinary, 0, cachedPtr, cachedBinary.Length);
                    CuBlasNative.CheckCudaResult(
                        CudaNativeBindings.cuModuleLoadData(out IntPtr cachedModule, cachedPtr),
                        $"cuModuleLoadData({moduleName} cached)");

                    foreach (var kernelName in kernelNames)
                    {
                        CuBlasNative.CheckCudaResult(
                            CudaNativeBindings.cuModuleGetFunction(out IntPtr kernel, cachedModule, kernelName),
                            $"cuModuleGetFunction({kernelName})");
                        _kernelCache[kernelName] = kernel;
                    }

                    return cachedModule;
                }
                finally
                {
                    Marshal.FreeHGlobal(cachedPtr);
                }
            }
            catch
            {
                // Cache corrupted or incompatible — fall through to recompile
                try { File.Delete(cacheFile); } catch { }
            }
        }

        // Compile with NVRTC
        var optionsList = new List<string> { arch, "--use_fast_math" };
        var cudaInclude = GetCudaIncludePath();
        if (cudaInclude != null)
            optionsList.Add($"--include-path={cudaInclude}");

        string[] options = optionsList.ToArray();

        IntPtr program = IntPtr.Zero;
        var result = NvrtcNativeBindings.nvrtcCreateProgram(
            ref program, source, moduleName + ".cu", 0, IntPtr.Zero, IntPtr.Zero);
        if (result != NvrtcResult.Success)
            throw new InvalidOperationException($"NVRTC program creation failed for {moduleName}: {NvrtcNativeBindings.GetErrorString(result)}");

        result = NvrtcNativeBindings.nvrtcCompileProgram(program, options.Length, options);
        if (result != NvrtcResult.Success)
        {
            string log = GetNvrtcLog(program);
            NvrtcNativeBindings.nvrtcDestroyProgram(ref program);
            throw new InvalidOperationException($"NVRTC compile failed for {moduleName}: {NvrtcNativeBindings.GetErrorString(result)}\n{log}");
        }

        bool useCubin = arch.Contains("sm_");
        IntPtr binary;
        UIntPtr binarySize;

        if (useCubin)
        {
            result = NvrtcNativeBindings.nvrtcGetCUBINSize(program, out binarySize);
            if (result != NvrtcResult.Success || binarySize == UIntPtr.Zero)
            {
                NvrtcNativeBindings.nvrtcDestroyProgram(ref program);
                throw new InvalidOperationException($"NVRTC failed to return CUBIN size for {moduleName}.");
            }
            binary = Marshal.AllocHGlobal((int)binarySize);
            result = NvrtcNativeBindings.nvrtcGetCUBIN(program, binary);
        }
        else
        {
            result = NvrtcNativeBindings.nvrtcGetPTXSize(program, out binarySize);
            if (result != NvrtcResult.Success || binarySize == UIntPtr.Zero)
            {
                NvrtcNativeBindings.nvrtcDestroyProgram(ref program);
                throw new InvalidOperationException($"NVRTC failed to return PTX size for {moduleName}.");
            }
            binary = Marshal.AllocHGlobal((int)binarySize);
            result = NvrtcNativeBindings.nvrtcGetPTX(program, binary);
        }

        NvrtcNativeBindings.nvrtcDestroyProgram(ref program);

        if (result != NvrtcResult.Success)
        {
            Marshal.FreeHGlobal(binary);
            throw new InvalidOperationException($"NVRTC get {(useCubin ? "CUBIN" : "PTX")} failed for {moduleName}: {NvrtcNativeBindings.GetErrorString(result)}");
        }

        // Save to disk cache before loading
        if (cacheFile != null && useCubin)
        {
            try
            {
                byte[] binaryBytes = new byte[(int)binarySize];
                Marshal.Copy(binary, binaryBytes, 0, binaryBytes.Length);
                File.WriteAllBytes(cacheFile, binaryBytes);
            }
            catch
            {
                // Non-fatal: caching failure doesn't block kernel loading
            }
        }

        IntPtr module;
        try
        {
            CuBlasNative.CheckCudaResult(CudaNativeBindings.cuModuleLoadData(out module, binary), $"cuModuleLoadData({moduleName})");
        }
        finally
        {
            Marshal.FreeHGlobal(binary);
        }

        foreach (var kernelName in kernelNames)
        {
            CuBlasNative.CheckCudaResult(
                CudaNativeBindings.cuModuleGetFunction(out IntPtr kernel, module, kernelName),
                $"cuModuleGetFunction({kernelName})");
            _kernelCache[kernelName] = kernel;
        }

        return module;
    }

    private void CompileAllKernels(int device)
    {
        using var _ = PushContext();

        _activationModule = CompileKernelModule(device, CudaActivationKernels.GetSource(), "activation_kernels", CudaActivationKernels.GetKernelNames());
        _convolutionModule = CompileKernelModule(device, CudaConvolutionKernels.GetSource(), "convolution_kernels", CudaConvolutionKernels.GetKernelNames());
        _fusedConvolutionModule = CompileKernelModule(device, CudaFusedConvolutionKernels.GetSource(), "fused_convolution_kernels", CudaFusedConvolutionKernels.GetKernelNames());
        _poolingModule = CompileKernelModule(device, CudaPoolingKernels.GetSource(), "pooling_kernels", CudaPoolingKernels.GetKernelNames());
        _normalizationModule = CompileKernelModule(device, CudaNormalizationKernels.GetSource(), "normalization_kernels", CudaNormalizationKernels.GetKernelNames());
        _neuralNetModule = CompileKernelModule(device, CudaNeuralNetKernels.GetSource(), "neuralnet_kernels", CudaNeuralNetKernels.GetKernelNames());
        _fusedModule = CompileKernelModule(device, CudaFusedKernels.GetSource(), "fused_kernels", CudaFusedKernels.GetKernelNames());
        _attentionModule = CompileKernelModule(device, CudaAttentionKernels.GetSource(), "attention_kernels", CudaAttentionKernels.GetKernelNames());
        _fftModule = CompileKernelModule(device, Kernels.CudaFFTKernels.GetSource(), "fft_kernels", Kernels.CudaFFTKernels.GetKernelNames());
        _spectralPerfModule = CompileKernelModule(device, Kernels.CudaSpectralPerfKernels.GetSource(), "spectral_perf_kernels", Kernels.CudaSpectralPerfKernels.GetKernelNames());
        _sparseModule = CompileKernelModule(device, CudaSparseKernels.GetSource(), "sparse_kernels", CudaSparseKernels.GetKernelNames());
        _spatialTransformerModule = CompileKernelModule(device, CudaSpatialTransformerKernels.GetSource(), "spatial_transformer_kernels", CudaSpatialTransformerKernels.GetKernelNames());

        // Compile Locally Connected kernels (unique weights per spatial position)
        _locallyConnectedModule = CompileKernelModule(device, CudaLocallyConnectedKernels.GetSource(), "locally_connected_kernels", CudaLocallyConnectedKernels.GetKernelNames());

        // Compile Deformable Convolution kernels (DCNv2 with learnable offsets and masks)
        _deformableConvModule = CompileKernelModule(device, CudaDeformableConvolutionKernels.GetSource(), "deformable_conv_kernels", CudaDeformableConvolutionKernels.GetKernelNames());

        // Compile Capsule Network kernels (prediction transform, routing)
        _capsuleModule = CompileKernelModule(device, CudaCapsuleKernels.GetSource(), "capsule_kernels", CudaCapsuleKernels.GetKernelNames());

        // Compile Specialized kernels (hyperbolic geometry, octonion algebra, quantum computing)
        _specializedModule = CompileKernelModule(device, CudaSpecializedKernels.GetSource(), "specialized_kernels", CudaSpecializedKernels.GetKernelNames());

        // Compile SNN kernels (STDP, spike traces, RBF, PRNG, 2:4 structured sparsity)
        _snnModule = CompileKernelModule(device, CudaSnnKernels.GetSource(), "snn_kernels", CudaSnnKernels.GetKernelNames());

        // Compile GPU-resident optimizer kernels (adam/adamw/sgd/momentum/nag/
        // adagrad/rmsprop/adadelta/adamax/nadam/lamb/lars/lion/amsgrad/ftrl
        // *_update). Without this the in-place GPU optimizer step (GpuOptimizer.
        // TryXStep → IDirectGpuBackend.XUpdate) throws "CUDA kernel not found:
        // <name>_update" for every kernel that is NOT also duplicated into the
        // neuralnet module (adam/adagrad/rmsprop/lamb/adamw/sgd/... were dead),
        // so every optimizer silently fell back to the CPU weight update.
        CompileKernelModule(device, CudaOptimizerKernels.GetSource(), "optimizer_kernels", CudaOptimizerKernels.GetKernelNames());

        // Compile reduction kernels (mean, variance, std, norm, logsumexp, product, cumsum)
        CompileKernelModule(device, CudaReductionKernels.GetSource(), "reduction_kernels", CudaReductionKernels.GetKernelNames());

        // Compile broadcast/scalar/element-wise utility kernels
        CompileKernelModule(device, CudaBroadcastKernels.GetSource(), "broadcast_kernels", CudaBroadcastKernels.GetKernelNames());

        // Compile gated activation kernels (GLU, GeGLU, ReGLU, SwiGLU, derivatives)
        CompileKernelModule(device, CudaGatedActivationKernels.GetSource(), "gated_activation_kernels", CudaGatedActivationKernels.GetKernelNames());

        // Compile shape/layout kernels (concat, slice, pad, tile, pixel shuffle, utility)
        CompileKernelModule(device, CudaShapeKernels.GetSource(), "shape_kernels", CudaShapeKernels.GetKernelNames());

        // Compile loss forward kernels (cross-entropy, MSE, BCE, dropout mask, gaussian noise)
        CompileKernelModule(device, CudaLossForwardKernels.GetSource(), "loss_forward_kernels", CudaLossForwardKernels.GetKernelNames());

        // Compile fused linear + activation kernels (MatMul + Bias + ReLU/Sigmoid/Tanh/GELU/Swish)
        // Optional — core ops still work without these.
        try
        {
            CompileKernelModule(device, Kernels.CudaFusedLinearKernels.GetSource(), "fused_linear_kernels", Kernels.CudaFusedLinearKernels.GetKernelNames());
        }
        catch { }

        // Fused-advanced kernels (LoRA, DDIM, sparse-linear) are an OPTIONAL
        // capability — older CUDA toolkits / drivers may reject the dynamic-
        // shared-mem patterns the kernels use. If compilation fails we still
        // want the rest of the backend to come up; we just refuse the fused
        // dispatch methods at runtime via the availability flag.
        try
        {
            CompileKernelModule(device, Kernels.CudaFusedAdvancedKernels.GetSource(), "fused_advanced_kernels", Kernels.CudaFusedAdvancedKernels.GetKernelNames());
            // Verify all expected kernel names actually landed in the cache.
            // CompileKernelModule may partially populate on a per-kernel
            // failure; we want all-or-nothing for the capability flag.
            bool allLoaded = true;
            foreach (var name in Kernels.CudaFusedAdvancedKernels.GetKernelNames())
            {
                if (!_kernelCache.ContainsKey(name)) { allLoaded = false; break; }
            }
            _fusedAdvancedKernelsAvailable = allLoaded;
        }
        catch (OutOfMemoryException)
        {
            // Process-level resource problem; do NOT silently downgrade.
            throw;
        }
        catch (Exception ex)
        {
            System.Diagnostics.Debug.WriteLine(
                $"[CudaBackend] Fused-advanced kernel compile failed: {ex.GetType().Name}: {ex.Message}. " +
                "Backend will reject FusedLoRAForward/FusedDDIMStep/FusedSparseLinear.");
            _fusedAdvancedKernelsAvailable = false;
        }

        // Compile IoU loss kernels (IoU, GIoU, DIoU, CIoU forward + backward)
        // Optional — CPU composition fallback provides identical results.
        try
        {
            CompileKernelModule(device, Kernels.CudaIoUKernels.GetSource(), "iou_kernels", Kernels.CudaIoUKernels.GetKernelNames());
        }
        catch { }

        // Compile softmax variant + GEMM extension kernels
        CompileKernelModule(device, CudaSoftmaxVariantKernels.GetSource(), "softmax_variant_kernels", CudaSoftmaxVariantKernels.GetKernelNames());

        // Self-contained FP16 kernels (#558 layer 7): manual IEEE-754 half<->float, NO cuda_fp16.h, so
        // they compile with the driver alone (the header-based CudaFp16Kernels below silently fails on
        // driver-only machines). This is the FP16 path that actually works without a CUDA Toolkit.
        CompileKernelModule(device, Kernels.CudaFp16NativeKernels.GetSource(), "fp16_native_kernels", Kernels.CudaFp16NativeKernels.GetKernelNames());

        // Compile FP16 conversion kernels (half-precision float conversion)
        // May fail if NVRTC doesn't have cuda_fp16.h (minimal CUDA Toolkit install).
        try
        {
            _fp16Module = CompileKernelModule(device, CudaFp16Kernels.GetSource(), "fp16_kernels", CudaFp16Kernels.GetKernelNames());
        }
        catch (Exception fp16Ex)
        {
            // FP16 kernels are optional — fall back to FP32 paths. Capture the error: a silent failure
            // here disables the ENTIRE FP16 GPU path (convert + half activations + Tensor-Core throughput),
            // which is invisible otherwise (#558). Surfaced via Fp16ModuleCompileError for diagnostics.
            Fp16ModuleCompileError = fp16Ex.Message;
        }

        // Compile LSTM sequence kernels (forward/backward for BPTT training)
        try
        {
            _lstmModule = CompileKernelModule(device, CudaLstmKernels.GetSource(), "lstm_kernels", CudaLstmKernels.GetKernelNames());
        }
        catch
        {
            // LSTM kernels may need special headers — optional.
        }

        // Compile fused recurrence / LM-head kernels (#1464) in their OWN try/catch so an
        // optional-LSTM compile failure above doesn't also disable the recurrence family.
        try
        {
            _glaModule = CompileKernelModule(device, Kernels.CudaGlaKernels.GetSource(), "gla_kernels", Kernels.CudaGlaKernels.GetKernelNames());
            _xlstmModule = CompileKernelModule(device, Kernels.CudaXLstmKernels.GetSource(), "xlstm_kernels", Kernels.CudaXLstmKernels.GetKernelNames());
            _gatedDeltaNetModule = CompileKernelModule(device, Kernels.CudaGatedDeltaNetKernels.GetSource(), "gated_delta_net_kernels", Kernels.CudaGatedDeltaNetKernels.GetKernelNames());
            _rgLruModule = CompileKernelModule(device, Kernels.CudaRgLruKernels.GetSource(), "rglru_kernels", Kernels.CudaRgLruKernels.GetKernelNames());
            _rwkv4Module = CompileKernelModule(device, Kernels.CudaRwkv4Kernels.GetSource(), "rwkv4_kernels", Kernels.CudaRwkv4Kernels.GetKernelNames());
            _mambaModule = CompileKernelModule(device, Kernels.CudaMambaKernels.GetSource(), "mamba_kernels", Kernels.CudaMambaKernels.GetKernelNames());
            _mamba2Module = CompileKernelModule(device, Kernels.CudaMamba2Kernels.GetSource(), "mamba2_kernels", Kernels.CudaMamba2Kernels.GetKernelNames());
            _fusedCeModule = CompileKernelModule(device, Kernels.CudaFusedLinearCeKernels.GetSource(), "fused_ce_kernels", Kernels.CudaFusedLinearCeKernels.GetKernelNames());
        }
        catch
        {
            // Fused recurrence / LM-head kernels are optional.
        }

        // Compile GRU sequence kernels (forward/backward for BPTT training)
        // Needs cooperative_groups.h which may not be in minimal CUDA Toolkit installs.
        try
        {
            _gruModule = CompileKernelModule(device, CudaGruKernels.GetSource(), "gru_kernels", CudaGruKernels.GetKernelNames());
        }
        catch
        {
            // GRU kernels need cooperative_groups.h — optional.
        }

        // Compile WMMA Tensor Core kernels for Volta+ (sm_70+)
        if (_ccMajor >= 7)
        {
            try
            {
                _wmmaModule = CompileKernelModule(device, CudaWmmaKernels.GetSource(), "wmma_kernels", CudaWmmaKernels.GetKernelNames());
                _hasWmmaSupport = true;
            }
            catch
            {
                // WMMA compilation may fail if NVRTC doesn't support mma.h headers.
                // Fall back to Phase 3 tiled GEMM kernels silently.
                _hasWmmaSupport = false;
            }
        }

        // Compile split-buffer complex kernels for native Tensor<Complex<T>> operations
        CompileKernelModule(device, Kernels.CudaComplexKernels.GetSource(), "complex_kernels", Kernels.CudaComplexKernels.GetKernelNames());

        // Parity-210: 40 hot-path kernels (movement / cumulative / indexing /
        // element-wise special / pairwise) covering the #210 op surface.
        // Optional — if NVRTC rejects something on an older toolkit we fall
        // back to the CPU reference via CpuEngine inheritance.
        try
        {
            _parity210Module = CompileKernelModule(device,
                Kernels.CudaParity210Kernels.GetSource(),
                "parity210_kernels",
                Kernels.CudaParity210Kernels.GetKernelNames());
        }
        catch
        {
            _parity210Module = IntPtr.Zero;
        }

        // Vision detection kernels (Issue #217). Same best-effort policy as
        // parity-210: NVRTC failures fall through to the CpuEngine path
        // (DirectGpuTensorEngine.Detection.cs catches and base.* delegates).
        try
        {
            _detectionModule = CompileKernelModule(device,
                Kernels.CudaDetectionKernels.GetSource(),
                "detection_kernels",
                Kernels.CudaDetectionKernels.GetKernelNames());
        }
        catch (Exception ex)
        {
            System.Diagnostics.Debug.WriteLine($"CUDA detection kernel compilation failed: {ex.Message}");
            _detectionModule = IntPtr.Zero;
        }

        // Geometry / sampling kernels (Issue #217 second half). Same
        // best-effort policy — NVRTC failure falls through to CpuEngine.
        try
        {
            _geometryModule = CompileKernelModule(device,
                Kernels.CudaGeometryKernels.GetSource(),
                "geometry_kernels",
                Kernels.CudaGeometryKernels.GetKernelNames());
        }
        catch (Exception ex)
        {
            System.Diagnostics.Debug.WriteLine($"CUDA geometry kernel compilation failed: {ex.Message}");
            _geometryModule = IntPtr.Zero;
        }

        // RoI kernels (Issue #217 tail).
        try
        {
            _roiModule = CompileKernelModule(device,
                Kernels.CudaRoiKernels.GetSource(),
                "roi_kernels",
                Kernels.CudaRoiKernels.GetKernelNames());
        }
        catch (Exception ex)
        {
            System.Diagnostics.Debug.WriteLine($"CUDA RoI kernel compilation failed: {ex.Message}");
            _roiModule = IntPtr.Zero;
        }

        // Audio kernels (Issue #217 tail).
        try
        {
            _audioModule = CompileKernelModule(device,
                Kernels.CudaAudioKernels.GetSource(),
                "audio_kernels",
                Kernels.CudaAudioKernels.GetKernelNames());
        }
        catch (Exception ex)
        {
            // Preserve the compile error for post-mortem — without this
            // log, a NVRTC failure silently degrades the audio ops to
            // CpuEngine with no indication why.
            System.Diagnostics.Debug.WriteLine($"CUDA audio kernel compilation failed: {ex.Message}");
            _audioModule = IntPtr.Zero;
        }

        // Linalg decomposition kernels (#211 moat #2). Same best-effort policy:
        // NVRTC failures fall through to the CPU reference via ILinalgBackend
        // not being advertised by this backend.
        try
        {
            _linalgModule = CompileKernelModule(device,
                Kernels.CudaLinalgKernels.GetSource(),
                "linalg_kernels",
                Kernels.CudaLinalgKernels.GetKernelNames());
        }
        catch
        {
            _linalgModule = IntPtr.Zero;
        }
    }

    private static string GetNvrtcLog(IntPtr program)
    {
        var result = NvrtcNativeBindings.nvrtcGetProgramLogSize(program, out UIntPtr logSize);
        if (result != NvrtcResult.Success || logSize == UIntPtr.Zero)
            return string.Empty;

        IntPtr logPtr = Marshal.AllocHGlobal((int)logSize);
        NvrtcNativeBindings.nvrtcGetProgramLog(program, logPtr);
        string log = Marshal.PtrToStringAnsi(logPtr) ?? string.Empty;
        Marshal.FreeHGlobal(logPtr);
        return log;
    }

    public IGpuBuffer AllocateBuffer(float[] data)
    {
        if (!IsAvailable)
            throw new InvalidOperationException("CUDA backend is not available.");

        int size = data.Length;
        if (size <= 0)
            throw new ArgumentOutOfRangeException(nameof(data), "Buffer size must be positive.");
        ulong byteSize = (ulong)size * sizeof(float);
        // Issue #285: per-allocation cap check before cuMemAlloc.
        GpuBufferSizeGuard.EnsureFits("CUDA", (long)byteSize, MaxBufferAllocBytes, DeviceName);

        // CUDA driver API calls are required for device memory operations.
        using var _ = PushContext();
        // Stream-ordered allocation (same default mem pool as the activation buffers) when async-alloc is
        // on. Without this, weight/persistent uploads take the legacy SYNC cuMemAlloc path below, which sees
        // device free=0 once the async pool has reserved VRAM (release threshold maxed for caching) — the
        // forward-only eval path then OOMs on its first weight upload while the GPU is "full" of pool-reserved
        // (but reusable) bytes. Drawing weights from the same pool removes that sync/async split.
        if (_asyncAlloc)
        {
            var pAsync = AllocDeviceMemoryAsync(byteSize);
            try
            {
                unsafe
                {
                    fixed (float* src = data)
                    {
                        CuBlasNative.CheckCudaResult(
                            CuBlasNative.cuMemcpyHtoD(pAsync, (IntPtr)src, byteSize),
                            "cuMemcpyHtoD");
                    }
                }
            }
            catch { CuBlasNative.cuMemFreeAsync(pAsync, _stream); throw; }
            return new CudaGpuBuffer(_cudaContext, pAsync, size, returnToPool: null, asyncFreeStream: _stream);
        }
        if (_bufferPool.TryRent(size, out var pooled) && pooled != null)
        {
            unsafe
            {
                fixed (float* src = data)
                {
                    CuBlasNative.CheckCudaResult(
                        CuBlasNative.cuMemcpyHtoD(pooled.Handle, (IntPtr)src, byteSize), // lgtm[cs/call-to-unmanaged-code] CUDA interop requires native driver calls.
                        "cuMemcpyHtoD");
                }
            }
            return pooled;
        }

        IntPtr devicePtr = AllocDeviceMemoryWithRetry(byteSize);

        try
        {
            unsafe
            {
                fixed (float* src = data)
                {
                    CuBlasNative.CheckCudaResult(
                        CuBlasNative.cuMemcpyHtoD(devicePtr, (IntPtr)src, byteSize),
                        "cuMemcpyHtoD");
                }
            }
        }
        catch
        {
            CuBlasNative.cuMemFree(devicePtr);
            throw;
        }

        return new CudaGpuBuffer(_cudaContext, devicePtr, size, _bufferPool.Return);
    }

    /// <summary>
    /// Race-free deferred free of a GPU buffer (#555). Records a stream event marking the
    /// buffer's last enqueued use and holds the buffer until that event completes, so it is
    /// never returned to the pool (and therefore never reused or cuMemFree'd) while an
    /// in-flight kernel still references it. Non-blocking: drains already-completed deferrals
    /// each call. Use this instead of <c>buffer.Dispose()</c> for activations freed during a
    /// training step (the within-step VRAM-pressure offload path).
    /// </summary>
    public void FreeBufferDeferred(IGpuBuffer? buffer)
    {
        if (buffer == null) return;
        // Stream-ordered buffers (#558 layer 6): cuMemFreeAsync is already ordered after prior stream
        // work, so free immediately — deferring via an event would delay pool reclamation and erase the
        // mid-step reuse that bounds peak. Disposing routes to cuMemFreeAsync (see CudaGpuBuffer.Release).
        if ((buffer is CudaGpuBuffer cb && cb.IsAsyncFreed) || (buffer is CudaGpuByteBuffer bb && bb.IsAsyncFreed))
        {
            buffer.Dispose();
            return;
        }
        CudaEvent evt;
        using (PushContext())
            evt = new CudaEvent(this, _defaultStream, enableTiming: false); // creates + records on the compute stream
        lock (_deferredFreeLock)
            _deferredFrees.Enqueue((buffer, evt));
        DrainDeferredFrees(blockOldest: false);
    }

    /// <summary>
    /// Returns buffers whose deferred-free event has completed to the pool. With
    /// <paramref name="blockOldest"/> = true, waits on the front event (used to reclaim
    /// memory under genuine allocation pressure). FIFO: events recorded later complete
    /// later, so once the front is incomplete the rest are too (fast early-out).
    /// </summary>
    public void DrainDeferredFrees(bool blockOldest)
    {
        lock (_deferredFreeLock)
        {
            while (_deferredFrees.Count > 0)
            {
                var (buf, evt) = _deferredFrees.Peek();
                if (!evt.IsComplete)
                {
                    if (!blockOldest) break;
                    evt.Synchronize();
                }
                _deferredFrees.Dequeue();
                buf.Dispose();   // returns to the pool — now provably safe
                evt.Dispose();
            }
        }
    }

    /// <summary>
    /// Bounds cross-step VRAM growth from per-step transient activation buffers that are
    /// dereferenced but not yet GC-finalized (their device memory is only released by the
    /// <c>OwnedBuffer</c>/<c>CudaGpuBuffer</c> finalizer, which under fast training stepping
    /// cannot keep pace — manifesting as a steady ~1 GB/min climb to OOM with "drained 0
    /// pooled buffers", because the leaked bytes are held by un-collected managed objects,
    /// not the pool). Cheap when not pressured: one <c>cuMemGetInfo</c> and a fraction check.
    /// Under pressure (free fraction below <paramref name="freeFractionThreshold"/>) it forces
    /// the pending finalizers to run — reclaiming the transient device memory deterministically
    /// — then drains completed deferred frees back to the pool. Called once per training step
    /// from <c>CompiledTrainingPlan.StepEager</c>.
    ///
    /// Hysteresis: a workload whose healthy steady state holds most of the device (e.g. a large
    /// compiled-graph training run with a sized activation cache) sits below the threshold on
    /// EVERY step, and an unconditional sweep there costs two full GCs plus a finalizer wait per
    /// step (measured ~19x wall-clock on a 12 GB card at ~71% resident). Each ineffective sweep
    /// (reclaimed less than 1% of the device) doubles a step cooldown up to 256; any effective
    /// sweep or a return to an unpressured state resets it, so genuine leaks are still reclaimed
    /// at full cadence.
    /// </summary>
    public void ReclaimUnderPressure(double freeFractionThreshold = 0.20)
    {
        // Bounds: a free-fraction threshold only makes sense in (0, 1]. > 1 would force the full
        // GC/finalizer sweep on EVERY call (free/total can never exceed 1), and <= 0 (or NaN) would
        // disable reclamation entirely — both defeat the purpose, so reject rather than silently misbehave.
        if (!(freeFractionThreshold > 0.0 && freeFractionThreshold <= 1.0))
            throw new ArgumentOutOfRangeException(nameof(freeFractionThreshold), freeFractionThreshold,
                "freeFractionThreshold must be in the range (0, 1].");
        if (!IsAvailable) return;
        if (_reclaimCooldownRemaining > 0)
        {
            _reclaimCooldownRemaining--;
            return;
        }
        ulong free, total;
        using (PushContext())
        {
            if (CudaNativeBindings.cuMemGetInfo(out free, out total) != CudaResult.Success || total == 0)
                return;
        }
        if ((double)free / total > freeFractionThreshold)
        {
            _reclaimBackoffSteps = 1; // unpressured — re-arm full cadence
            return;
        }

        // Pressured: run pending finalizers so dereferenced transient GPU buffers are released,
        // then reclaim any completed deferred frees. Two collects bracket the finalizer wait so
        // objects promoted to the finalizer queue on the first pass are reclaimed on the second.
        GC.Collect();
        GC.WaitForPendingFinalizers();
        GC.Collect();
        DrainDeferredFrees(blockOldest: false);

        // Effectiveness check: a workload legitimately resident below the threshold reclaims ~nothing;
        // back off exponentially so the per-step GC tax amortizes away. A sweep that frees >=1% of the
        // device means transient garbage IS accumulating — keep full cadence.
        ulong freeAfter = free;
        using (PushContext())
        {
            if (CudaNativeBindings.cuMemGetInfo(out var f2, out _) == CudaResult.Success)
                freeAfter = f2;
        }
        bool effective = freeAfter > free && (freeAfter - free) >= total / 100;
        _reclaimBackoffSteps = effective ? 1 : Math.Min(_reclaimBackoffSteps * 2, 256);
        _reclaimCooldownRemaining = effective ? 0 : _reclaimBackoffSteps;
    }

    private int _reclaimCooldownRemaining;
    private int _reclaimBackoffSteps = 1;

    /// <summary>
    /// Allocates device memory with an OOM-recovery retry. If the driver reports
    /// out-of-memory, drain the buffer pool — a real <c>cuMemFree</c> of every
    /// retained-for-reuse buffer — and retry the allocation once. The pool holds
    /// bounded-but-real device memory (up to <c>_maxPerSize</c> buffers per
    /// power-of-two bucket); under genuine pressure reclaiming it lets the
    /// allocation through. This is the fix for the cross-step working-set growth
    /// that OOM'd a 12 GB card on a tiny d256/L2 cortex past ~200K tokens
    /// (200K trained, 250K OOM'd at the same per-step cost → accumulation, not a
    /// single-step peak). If it still OOMs after draining, report the actual
    /// free/total device memory (cuMemGetInfo) in the exception so the residual
    /// consumer is visible rather than a bare "Out of memory".
    /// </summary>
    private IntPtr AllocDeviceMemoryWithRetry(ulong byteSize)
    {
        var result = CuBlasNative.cuMemAlloc(out IntPtr devicePtr, byteSize);
        if (result == CudaResult.OutOfMemory)
        {
            // Reclaim event-deferred frees first (#555) — under genuine pressure block on
            // their events so their device memory returns to the pool — then drain the pool.
            DrainDeferredFrees(blockOldest: true);
            int drained = _bufferPool.DrainAll();
            // When _asyncAlloc is on, freed stream-ordered allocations are RETAINED in the default mem
            // pool (release threshold maxed for caching), so they show as device free=0 and the legacy
            // cuMemAlloc here can't reach them. Sync the context so pending cuMemFreeAsync reclamations
            // land in the pool, then trim the pool back to the OS so this sync allocation can succeed.
            if (_asyncMemPool != IntPtr.Zero)
            {
                // A failed sync here is almost always a STICKY context error (the very CUDA-700 this
                // allocator guards against), NOT out-of-memory — throw with its real name so the OOM
                // diagnostic below isn't misleading. Trim is best-effort: warn but still attempt the retry.
                CuBlasNative.CheckCudaResult(CuBlasNative.cuCtxSynchronize(), "cuCtxSynchronize (async-pool OOM reclamation)");
                var trimStatus = CuBlasNative.cuMemPoolTrimTo(_asyncMemPool, 0);
                if (trimStatus != CudaResult.Success)
                    System.Diagnostics.Trace.TraceWarning(
                        $"[CudaBackend] cuMemPoolTrimTo failed during sync-path OOM reclamation: {trimStatus}; the cuMemAlloc retry may still OOM.");
            }
            result = CuBlasNative.cuMemAlloc(out devicePtr, byteSize);
            if (result != CudaResult.Success)
            {
                ulong free = 0, total = 0;
                try { CudaNativeBindings.cuMemGetInfo(out free, out total); }
                catch { /* diagnostic only — never mask the real failure */ }
                GpuMemoryTracker.Dump($"cuMemAlloc OOM requesting {byteSize} bytes (free={free})");
                throw new InvalidOperationException(
                    $"cuMemAlloc failed: {CuBlasNative.GetCudaErrorString(result)} " +
                    $"(requested {byteSize} bytes; drained {drained} pooled buffers; " +
                    $"device free={free} total={total} bytes)");
            }
        }
        else
        {
            CuBlasNative.CheckCudaResult(result, "cuMemAlloc");
        }
        GpuMemoryTracker.OnAlloc(devicePtr, (long)byteSize);
        return devicePtr;
    }

    public IGpuBuffer AllocateBuffer(int size)
    {
        if (!IsAvailable)
            throw new InvalidOperationException("CUDA backend is not available.");

        if (size <= 0)
            throw new ArgumentOutOfRangeException(nameof(size), "Buffer size must be positive.");
        // Issue #285: per-allocation cap check before cuMemAlloc.
        GpuBufferSizeGuard.EnsureFits("CUDA", (long)size * sizeof(float), MaxBufferAllocBytes, DeviceName);

        // CUDA driver API calls are required for device memory operations.
        using var _ = PushContext();
        if (_asyncAlloc)
        {
            var p = AllocDeviceMemoryAsync((ulong)size * sizeof(float));
            CuBlasNative.CheckCudaResult(CuBlasNative.cuMemsetD32(p, 0, (ulong)size), "cuMemsetD32");
            return new CudaGpuBuffer(_cudaContext, p, size, returnToPool: null, asyncFreeStream: _stream);
        }
        if (_bufferPool.TryRent(size, out var pooled) && pooled != null)
        {
            CuBlasNative.CheckCudaResult(
                CuBlasNative.cuMemsetD32(pooled.Handle, 0, (ulong)size), // lgtm[cs/call-to-unmanaged-code] CUDA interop requires native driver calls.
                "cuMemsetD32");
            return pooled;
        }

        ulong byteSize = (ulong)size * sizeof(float);
        IntPtr devicePtr = AllocDeviceMemoryWithRetry(byteSize);
        CuBlasNative.CheckCudaResult(CuBlasNative.cuMemsetD32(devicePtr, 0, (ulong)size), "cuMemsetD32");
        return new CudaGpuBuffer(_cudaContext, devicePtr, size, _bufferPool.Return);
    }

    public IGpuBuffer AllocateByteBuffer(int size)
    {
        if (!IsAvailable)
            throw new InvalidOperationException("CUDA backend is not available.");

        if (size <= 0)
            throw new ArgumentOutOfRangeException(nameof(size), "Buffer size must be positive.");
        GpuBufferSizeGuard.EnsureFits("CUDA", size, MaxBufferAllocBytes, DeviceName);

        using var _ = PushContext();
        if (_asyncAlloc)
        {
            var pA = AllocDeviceMemoryAsync((ulong)size);
            return new CudaGpuByteBuffer(_cudaContext, pA, size, asyncFreeStream: _stream);
        }
        CuBlasNative.CheckCudaResult(
            CuBlasNative.cuMemAlloc(out IntPtr devicePtr, (ulong)size),
            "cuMemAlloc(byte)");

        return new CudaGpuByteBuffer(_cudaContext, devicePtr, size);
    }

    // Stream-ordered device allocation (#558 layer 6). On OOM, sync the context (lets pending
    // cuMemFreeAsync reclamations complete) and retry once before surfacing the error.
    private IntPtr AllocDeviceMemoryAsync(ulong byteSize)
    {
        var r = CuBlasNative.cuMemAllocAsync(out IntPtr p, byteSize, _stream);
        if (r != CudaResult.Success)
        {
            // Sync so pending cuMemFreeAsync reclamations land back in the pool, then TRIM the pool's
            // retained-but-unused reservation back to the OS. After a fused/resident training run the pool
            // has reserved up to the training peak (release threshold maxed for caching); a later phase with
            // a different allocation shape (e.g. forward-only eval re-uploading weights) can then OOM here
            // while almost all of VRAM is pool-reserved-but-free. Trimming returns it so the retry succeeds.
            // As in the sync-path reclamation: a failed sync is a real (often sticky) error, so throw
            // with its real name rather than let it masquerade as the cuMemAllocAsync OOM below.
            // Trim is best-effort.
            CuBlasNative.CheckCudaResult(CuBlasNative.cuCtxSynchronize(), "cuCtxSynchronize (async-pool OOM reclamation)");
            if (_asyncMemPool != IntPtr.Zero)
            {
                var trimStatus = CuBlasNative.cuMemPoolTrimTo(_asyncMemPool, 0);
                if (trimStatus != CudaResult.Success)
                    System.Diagnostics.Trace.TraceWarning(
                        $"[CudaBackend] cuMemPoolTrimTo failed during async-path OOM reclamation: {trimStatus}; the cuMemAllocAsync retry may still OOM.");
            }
            r = CuBlasNative.cuMemAllocAsync(out p, byteSize, _stream);
        }
        if (r != CudaResult.Success) GpuMemoryTracker.Dump($"cuMemAllocAsync OOM requesting {byteSize} bytes");
        CuBlasNative.CheckCudaResult(r, "cuMemAllocAsync");
        GpuMemoryTracker.OnAlloc(p, (long)byteSize);
        return p;
    }

    public float[] DownloadBuffer(IGpuBuffer buffer)
    {
        var result = new float[buffer.Size];
        DownloadBuffer(buffer, result);
        return result;
    }

    public void DownloadBuffer(IGpuBuffer buffer, float[] destination)
    {
        // #226: never issue a device→host copy from a released buffer. After CudaGpuBuffer.Release
        // the device pointer is zeroed; cuMemcpyDtoH from a null/freed pointer is an ILLEGAL memory
        // access that corrupts the CUDA context — every subsequent driver call then fails with error
        // 700, cascading across the whole process. Surface a clean managed exception BEFORE touching
        // the driver so the deferred-materializer caller (and the fused-training fault handler) can
        // recover on the eager path instead, leaving the context intact.
        if (buffer.Handle == IntPtr.Zero)
            throw new ObjectDisposedException(nameof(buffer),
                "GPU buffer was released before its deferred download (issue #226).");

        if (destination.Length < buffer.Size)
            throw new ArgumentException("Destination array is too small.", nameof(destination));

        using var _ = PushContext();

        // Synchronize the compute stream before the device→host copy. Kernels
        // launch on `_stream` (a non-default stream), but cuMemcpyDtoH below
        // issues on the null stream. With a non-blocking compute stream the
        // null stream does NOT implicitly wait for it, so the copy can race
        // ahead of a just-launched kernel and read stale device memory — most
        // visibly the in-place GPU-resident optimizer update, whose write would
        // then be silently lost on the immediately-following host read (the
        // value oscillated run-to-run between updated and pre-step). One stream
        // sync on the host-read path (already a synchronization point) closes
        // the race with no effect on the GPU-resident training hot path.
        CuBlasNative.CheckCudaResult(CudaNativeBindings.cuStreamSynchronize(_stream), "cuStreamSynchronize(download)");

        ulong byteSize = (ulong)(buffer.Size * sizeof(float));

        unsafe
        {
            fixed (float* dst = destination)
            {
                CuBlasNative.CheckCudaResult(
                    CuBlasNative.cuMemcpyDtoH((IntPtr)dst, buffer.Handle, byteSize),
                    "cuMemcpyDtoH");
            }
        }
    }

    public void Gemm(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int M, int N, int K, float alpha = 1.0f, float beta = 0.0f)
    {
        if (!IsAvailable)
            throw new InvalidOperationException("CUDA backend is not available.");

        ValidateGemmArgs(A, B, C, M, N, K);

        using var _ = PushContext();
        float alphaVal = alpha;
        float betaVal = beta;

        // Row-major C = A * B. Use cuBLAS column-major trick: C^T = B^T * A^T.
        CuBlasNative.CheckCublasStatus(
            CuBlasNative.cublasSgemm(
                _cublasHandle,
                CublasOperation.None,
                CublasOperation.None,
                N, M, K,
                ref alphaVal,
                B.Handle, N,
                A.Handle, K,
                ref betaVal,
                C.Handle, N),
            "cublasSgemm");
    }

    public void MatMulTransposed(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int M, int N, int K, float alpha = 1.0f, float beta = 0.0f)
    {
        if (!IsAvailable)
            throw new InvalidOperationException("CUDA backend is not available.");
        ValidateGemmArgs(A, B, C, M, N, K);

        using var _ = PushContext();
        float alphaVal = alpha;
        float betaVal = beta;

        // Row-major C[M,N] = A[M,K] · Bᵀ. Memory equivalences (column-
        // major view of the same bytes):
        //   B_row[N,K] === B_col[K,N]   (lda = K)
        //   A_row[M,K] === A_col[K,M]   (lda = K)
        //   C_row[M,N] === C_col[N,M]   (lda = N)
        // We need cuBLAS to compute C_col[N,M] = B_col[K,N]ᵀ · A_col[K,M]
        // because (A · Bᵀ)_row === (B · Aᵀ)_col. So:
        //   op(A_cu = my B) = Transpose → N×K
        //   op(B_cu = my A) = None      → K×M
        //   m_cu=N, n_cu=M, k_cu=K, ldA=K, ldB=K, ldC=N.
        // Earlier version had the transpose flags swapped — produced
        // wrong results for any non-square input.
        CuBlasNative.CheckCublasStatus(
            CuBlasNative.cublasSgemm(
                _cublasHandle,
                CublasOperation.Transpose,   // op(my B) — transposed, becomes N×K
                CublasOperation.None,        // op(my A) — no transpose, stays K×M
                N, M, K,
                ref alphaVal,
                B.Handle, K,                 // lda for B_col[K,N] = K
                A.Handle, K,                 // ldb for A_col[K,M] = K
                ref betaVal,
                C.Handle, N),                // ldc for C_col[N,M] = N
            "cublasSgemm(MatMulTransposed)");
    }

    public IGpuBuffer MatMul(IGpuBuffer A, IGpuBuffer B, int M, int N, int K)
    {
        ValidateGemmArgs(A, B, null, M, N, K);
        // Telemetry scope must reflect the path that actually executed, not
        // the policy flag — Gemm below is unconditionally cuBLAS, so label
        // the scope "MatMul.cuBLAS" regardless of UseCublasForMatMul. When a
        // handwritten GEMM fallback lands later and MatMul branches on the
        // flag, move this scope into each branch so profiler stats match
        // the kernel that ran.
        using var _profile = CudaDispatchPolicy.Scope(
            "MatMul",
            useVendor: true);
        var output = AllocateBuffer(M * N);
        Gemm(A, B, output, M, N, K, 1.0f, 0.0f);
        return output;
    }

    public void BatchedGemm(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int M, int N, int K, int batchCount, float alpha = 1.0f, float beta = 0.0f)
    {
        if (!IsAvailable)
            throw new InvalidOperationException("CUDA backend is not available.");

        ValidateBatchedGemmArgs(A, B, C, M, N, K, batchCount);

        using var _ = PushContext();
        float alphaVal = alpha;
        float betaVal = beta;

        long strideA = (long)M * K;
        long strideB = (long)K * N;
        long strideC = (long)M * N;

        CuBlasNative.CheckCublasStatus(
            CuBlasNative.cublasSgemmStridedBatched(
                _cublasHandle,
                CublasOperation.None,
                CublasOperation.None,
                N, M, K,
                ref alphaVal,
                B.Handle, N, strideB,
                A.Handle, K, strideA,
                ref betaVal,
                C.Handle, N, strideC,
                batchCount),
            "cublasSgemmStridedBatched");
    }

    /// <summary>
    /// Issue #335 items 3+4 production wiring: fans <paramref name="batchCount"/>
    /// independent SGEMMs of identical shape across the supplied
    /// scheduler's stream pool. Each slice runs on its own stream, so
    /// the cuBLAS launches overlap rather than serialising on the
    /// default stream.
    /// <para>
    /// Use this in preference to <see cref="BatchedGemm"/> when:
    /// <list type="bullet">
    ///   <item>Per-slice work is small (M·N·K is at the attention-shape
    ///   scale e.g. 256·256·64 — small enough that cuBLAS strided-batched
    ///   doesn't saturate the GPU)</item>
    ///   <item>The caller already has a <see cref="GpuStreamScheduler"/>
    ///   live (e.g., MHA per-head dispatch, parallel ResNet branches)</item>
    /// </list>
    /// For large per-slice work, <see cref="BatchedGemm"/>'s strided-
    /// batched path remains faster because the in-slice parallelism
    /// already saturates the SMs.
    /// </para>
    /// </summary>
    /// <param name="scheduler">Stream scheduler whose pool provides the
    /// per-slice streams. Caller owns lifetime — see
    /// <see cref="DirectGpuTensorEngine.GetStreamScheduler"/>.</param>
    public void BatchedGemmFanout(
        IGpuBuffer A, IGpuBuffer B, IGpuBuffer C,
        int M, int N, int K, int batchCount,
        AiDotNet.Tensors.Engines.Gpu.GpuStreamScheduler scheduler,
        float alpha = 1.0f, float beta = 0.0f)
    {
        if (!IsAvailable) throw new InvalidOperationException("CUDA backend is not available.");
        if (scheduler is null) throw new ArgumentNullException(nameof(scheduler));
        ValidateBatchedGemmArgs(A, B, C, M, N, K, batchCount);

        long strideA = (long)M * K;
        long strideB = (long)K * N;
        long strideC = (long)M * N;
        long elemBytesA = sizeof(float);

        // Build one launch action per slice. Each captures its slice
        // offsets; the scheduler binds cuBLAS to the lease stream before
        // calling each lambda, so each cublasSgemm queues onto the
        // assigned stream.
        var launches = new System.Collections.Generic.List<System.Action<AiDotNet.Tensors.Engines.Gpu.IGpuStream>>(batchCount);
        for (int b = 0; b < batchCount; b++)
        {
            int slice = b;
            launches.Add(_ =>
            {
                // The scheduler binds cuBLAS to this stream prior to
                // invoking the lambda (see GpuStreamScheduler.Dispatch's
                // per-lease cublasSetStream call). We just issue the
                // launch — it queues onto the bound stream.
                float aVal = alpha, bVal = beta;
                IntPtr aPtr = (IntPtr)((long)A.Handle + slice * strideA * elemBytesA);
                IntPtr bPtr = (IntPtr)((long)B.Handle + slice * strideB * elemBytesA);
                IntPtr cPtr = (IntPtr)((long)C.Handle + slice * strideC * elemBytesA);
                CuBlasNative.CheckCublasStatus(
                    CuBlasNative.cublasSgemm(
                        _cublasHandle,
                        CublasOperation.None,
                        CublasOperation.None,
                        N, M, K,
                        ref aVal,
                        bPtr, N,
                        aPtr, K,
                        ref bVal,
                        cPtr, N),
                    $"cublasSgemm slice {slice}");
            });
        }

        using var batch = scheduler.Dispatch(launches);
        scheduler.SynchronizeEvents(batch);
    }

    /// <summary>
    /// Issue #335 item 4: Multi-head attention scoring fanout. Computes
    /// scores[h] = Q[h] · K[h]^T for each of <paramref name="numHeads"/>
    /// attention heads, with each head's GEMM running on its own stream
    /// from the scheduler's pool. Use the result tensor as input to a
    /// scaled-softmax + value projection pipeline.
    /// <para>
    /// Shapes (NCHW-style packed):
    /// <list type="bullet">
    ///   <item><paramref name="Q"/> packed as [batch, numHeads, seqLen, headDim]</item>
    ///   <item><paramref name="K"/> packed as [batch, numHeads, seqLen, headDim]</item>
    ///   <item><paramref name="scores"/> output packed as [batch, numHeads, seqLen, seqLen]</item>
    /// </list>
    /// One head per stream = numHeads concurrent SGEMMs of shape
    /// [seqLen, headDim] · [headDim, seqLen]. The cuBLAS launches overlap
    /// rather than serialising on the default stream.
    /// </para>
    /// </summary>
    public void MultiHeadAttentionScoresFanout(
        IGpuBuffer Q, IGpuBuffer K, IGpuBuffer scores,
        int batch, int numHeads, int seqLen, int headDim,
        AiDotNet.Tensors.Engines.Gpu.GpuStreamScheduler scheduler,
        float alpha = 1.0f, float beta = 0.0f)
    {
        if (!IsAvailable) throw new InvalidOperationException("CUDA backend is not available.");
        if (Q is null) throw new ArgumentNullException(nameof(Q));
        if (K is null) throw new ArgumentNullException(nameof(K));
        if (scores is null) throw new ArgumentNullException(nameof(scores));
        if (scheduler is null) throw new ArgumentNullException(nameof(scheduler));
        if (batch <= 0) throw new ArgumentOutOfRangeException(nameof(batch));
        if (numHeads <= 0) throw new ArgumentOutOfRangeException(nameof(numHeads));
        if (seqLen <= 0) throw new ArgumentOutOfRangeException(nameof(seqLen));
        if (headDim <= 0) throw new ArgumentOutOfRangeException(nameof(headDim));

        // Compute capacity math inside a `checked` block so an overflow
        // (e.g. seqLen=65536, headDim=65536 wrapping past long.MaxValue)
        // throws OverflowException instead of silently producing a
        // negative `required*` that would bypass the buffer guards below
        // and feed overflowed offsets into the raw cuBLAS pointer math.
        long perHeadQ, perHeadK, perHeadS, perBatchQ, perBatchK, perBatchS;
        long requiredQ, requiredK, requiredS;
        try
        {
            checked
            {
                perHeadQ = (long)seqLen * headDim;
                perHeadK = (long)seqLen * headDim;
                perHeadS = (long)seqLen * seqLen;
                perBatchQ = perHeadQ * numHeads;
                perBatchK = perHeadK * numHeads;
                perBatchS = perHeadS * numHeads;
                requiredQ = (long)batch * perBatchQ;
                requiredK = (long)batch * perBatchK;
                requiredS = (long)batch * perBatchS;
            }
        }
        catch (OverflowException)
        {
            throw new ArgumentException(
                $"MHA shape too large for fanout dispatch: B={batch}, H={numHeads}, S={seqLen}, D={headDim}.");
        }
        long elemBytes = sizeof(float);

        // Validate buffer capacity BEFORE the per-head offset math. Without
        // this, an undersized Q/K/scores buffer silently produces device
        // pointers past the allocation, which cuBLAS will read/write into —
        // typically a use-after-free or another tensor's data. The required
        // element count is (batch * perBatchX); IGpuBuffer.Size returns the
        // element count, so we compare directly.
        if (Q.Size < requiredQ)
            throw new ArgumentException(
                $"Q buffer too small: capacity={Q.Size} elements, required={requiredQ} for shape [B={batch}, H={numHeads}, S={seqLen}, D={headDim}].",
                nameof(Q));
        if (K.Size < requiredK)
            throw new ArgumentException(
                $"K buffer too small: capacity={K.Size} elements, required={requiredK} for shape [B={batch}, H={numHeads}, S={seqLen}, D={headDim}].",
                nameof(K));
        if (scores.Size < requiredS)
            throw new ArgumentException(
                $"scores buffer too small: capacity={scores.Size} elements, required={requiredS} for shape [B={batch}, H={numHeads}, S={seqLen}].",
                nameof(scores));

        var launches = new System.Collections.Generic.List<System.Action<AiDotNet.Tensors.Engines.Gpu.IGpuStream>>(batch * numHeads);
        for (int bi = 0; bi < batch; bi++)
        {
            for (int h = 0; h < numHeads; h++)
            {
                int batchIdx = bi, headIdx = h;
                launches.Add(_ =>
                {
                    float aVal = alpha, bVal = beta;
                    IntPtr qPtr = (IntPtr)((long)Q.Handle
                        + (batchIdx * perBatchQ + headIdx * perHeadQ) * elemBytes);
                    IntPtr kPtr = (IntPtr)((long)K.Handle
                        + (batchIdx * perBatchK + headIdx * perHeadK) * elemBytes);
                    IntPtr sPtr = (IntPtr)((long)scores.Handle
                        + (batchIdx * perBatchS + headIdx * perHeadS) * elemBytes);

                    // scores[h] = Q[h] · K[h]^T
                    // cuBLAS is column-major; for row-major Q · K^T we
                    // ask cuBLAS for K^T_col · Q_col which is K_row · Q_row^T.
                    // Result is stored in column-major scores; reading
                    // it as row-major gives the row-major Q · K^T we want.
                    CuBlasNative.CheckCublasStatus(
                        CuBlasNative.cublasSgemm(
                            _cublasHandle,
                            CublasOperation.Transpose,  // K^T
                            CublasOperation.None,        // Q
                            seqLen, seqLen, headDim,
                            ref aVal,
                            kPtr, headDim,
                            qPtr, headDim,
                            ref bVal,
                            sPtr, seqLen),
                        $"cublasSgemm MHA score batch={batchIdx} head={headIdx}");
                });
            }
        }

        using var b1 = scheduler.Dispatch(launches);
        scheduler.SynchronizeEvents(b1);
    }

    /// <summary>
    /// Mixed-precision MHA scoring fanout via <c>cublasGemmEx</c>: input
    /// Q / K are fp16 (or bf16), accumulator + output are fp32 — the
    /// standard AMP pattern. Combines #335 multi-stream dispatch with
    /// #337 mixed precision in one call.
    /// </summary>
    /// <param name="useBFloat16">When true, treats Q / K as bfloat16
    /// instead of fp16. Requires compute capability >= 8.0 (Ampere+).</param>
    public unsafe void MultiHeadAttentionScoresFanoutMixed(
        IGpuBuffer Q, IGpuBuffer K, IGpuBuffer scores,
        int batch, int numHeads, int seqLen, int headDim,
        AiDotNet.Tensors.Engines.Gpu.GpuStreamScheduler scheduler,
        bool useBFloat16 = false,
        float alpha = 1.0f, float beta = 0.0f)
    {
        if (!IsAvailable) throw new InvalidOperationException("CUDA backend is not available.");
        if (Q is null) throw new ArgumentNullException(nameof(Q));
        if (K is null) throw new ArgumentNullException(nameof(K));
        if (scores is null) throw new ArgumentNullException(nameof(scores));
        if (scheduler is null) throw new ArgumentNullException(nameof(scheduler));
        // Restore the positive-dimension guards the fp32 sister method has —
        // without these, a zero or negative dim flows straight into the
        // launches-list `Count` sizing and the perHead/perBatch offset math.
        if (batch <= 0) throw new ArgumentOutOfRangeException(nameof(batch));
        if (numHeads <= 0) throw new ArgumentOutOfRangeException(nameof(numHeads));
        if (seqLen <= 0) throw new ArgumentOutOfRangeException(nameof(seqLen));
        if (headDim <= 0) throw new ArgumentOutOfRangeException(nameof(headDim));
        if (useBFloat16 && _ccMajor < 8)
            throw new NotSupportedException(
                $"BFloat16 MHA requires compute capability >= 8.0 (Ampere+); device reports {_ccMajor}.{_ccMinor}.");

        int inDtype = useBFloat16 ? CuBlasNative.CUDA_R_16BF : CuBlasNative.CUDA_R_16F;
        // Checked arithmetic — see MultiHeadAttentionScoresFanout for the
        // overflow-mode rationale. Mixed-precision adds a *elemBytesIn
        // factor that can also overflow on huge shapes.
        long perHeadElems, perHeadScoreElems, perBatchQK, perBatchS;
        long requiredQkBytes, requiredS;
        long elemBytesIn = 2;        // fp16 / bf16
        long elemBytesOut = sizeof(float);
        try
        {
            checked
            {
                perHeadElems = (long)seqLen * headDim;
                perHeadScoreElems = (long)seqLen * seqLen;
                perBatchQK = perHeadElems * numHeads;
                perBatchS = perHeadScoreElems * numHeads;
                requiredQkBytes = (long)batch * perBatchQK * elemBytesIn;
                requiredS = (long)batch * perBatchS;
            }
        }
        catch (OverflowException)
        {
            throw new ArgumentException(
                $"Mixed-precision MHA shape too large for fanout dispatch: B={batch}, H={numHeads}, S={seqLen}, D={headDim}.");
        }

        // Validate buffer capacity BEFORE per-head offset math. Q/K are
        // 2-byte (fp16/bf16) so SizeInBytes is the right surface here;
        // scores is fp32 so element count suffices.
        if (Q.SizeInBytes < requiredQkBytes)
            throw new ArgumentException(
                $"Q buffer too small: capacity={Q.SizeInBytes} bytes, required={requiredQkBytes} for mixed-precision shape [B={batch}, H={numHeads}, S={seqLen}, D={headDim}].",
                nameof(Q));
        if (K.SizeInBytes < requiredQkBytes)
            throw new ArgumentException(
                $"K buffer too small: capacity={K.SizeInBytes} bytes, required={requiredQkBytes} for mixed-precision shape [B={batch}, H={numHeads}, S={seqLen}, D={headDim}].",
                nameof(K));
        if (scores.Size < requiredS)
            throw new ArgumentException(
                $"scores buffer too small: capacity={scores.Size} elements, required={requiredS} for shape [B={batch}, H={numHeads}, S={seqLen}].",
                nameof(scores));

        var launches = new System.Collections.Generic.List<System.Action<AiDotNet.Tensors.Engines.Gpu.IGpuStream>>(batch * numHeads);
        for (int bi = 0; bi < batch; bi++)
        {
            for (int h = 0; h < numHeads; h++)
            {
                int batchIdx = bi, headIdx = h;
                launches.Add(_ =>
                {
                    float aVal = alpha, bVal = beta;
                    IntPtr alphaPtr = (IntPtr)(&aVal);
                    IntPtr betaPtr = (IntPtr)(&bVal);
                    IntPtr qPtr = (IntPtr)((long)Q.Handle
                        + (batchIdx * perBatchQK + headIdx * perHeadElems) * elemBytesIn);
                    IntPtr kPtr = (IntPtr)((long)K.Handle
                        + (batchIdx * perBatchQK + headIdx * perHeadElems) * elemBytesIn);
                    IntPtr sPtr = (IntPtr)((long)scores.Handle
                        + (batchIdx * perBatchS + headIdx * perHeadScoreElems) * elemBytesOut);

                    CuBlasNative.CheckCublasStatus(
                        CuBlasNative.cublasGemmEx(
                            _cublasHandle,
                            CublasOperation.Transpose, CublasOperation.None,
                            seqLen, seqLen, headDim,
                            alphaPtr,
                            kPtr, inDtype, headDim,
                            qPtr, inDtype, headDim,
                            betaPtr,
                            sPtr, CuBlasNative.CUDA_R_32F, seqLen,
                            CuBlasNative.CUBLAS_COMPUTE_32F,
                            0 /* CUBLAS_GEMM_DEFAULT */),
                        $"cublasGemmEx MHA score batch={batchIdx} head={headIdx}");
                });
            }
        }

        using var b1 = scheduler.Dispatch(launches);
        scheduler.SynchronizeEvents(b1);
    }

    /// <summary>
    /// Multi-head attention output projection fanout: for each head,
    /// computes <c>output[h] = attention[h] · V[h]</c>. Companion to
    /// <see cref="MultiHeadAttentionScoresFanout"/>; the two together
    /// cover the bracketing GEMMs of the MHA pipeline (the middle
    /// softmax+scaling step is element-wise and stream-parallel via the
    /// existing Softmax helper).
    /// <para>
    /// Shapes (NCHW-style packed):
    /// <list type="bullet">
    ///   <item><paramref name="attention"/> packed as [batch, numHeads, seqLen, seqLen]</item>
    ///   <item><paramref name="V"/> packed as [batch, numHeads, seqLen, headDim]</item>
    ///   <item><paramref name="output"/> packed as [batch, numHeads, seqLen, headDim]</item>
    /// </list>
    /// </para>
    /// </summary>
    public void MultiHeadAttentionOutputFanout(
        IGpuBuffer attention, IGpuBuffer V, IGpuBuffer output,
        int batch, int numHeads, int seqLen, int headDim,
        AiDotNet.Tensors.Engines.Gpu.GpuStreamScheduler scheduler,
        float alpha = 1.0f, float beta = 0.0f)
    {
        if (!IsAvailable) throw new InvalidOperationException("CUDA backend is not available.");
        if (attention is null) throw new ArgumentNullException(nameof(attention));
        if (V is null) throw new ArgumentNullException(nameof(V));
        if (output is null) throw new ArgumentNullException(nameof(output));
        if (scheduler is null) throw new ArgumentNullException(nameof(scheduler));
        if (batch <= 0) throw new ArgumentOutOfRangeException(nameof(batch));
        if (numHeads <= 0) throw new ArgumentOutOfRangeException(nameof(numHeads));
        if (seqLen <= 0) throw new ArgumentOutOfRangeException(nameof(seqLen));
        if (headDim <= 0) throw new ArgumentOutOfRangeException(nameof(headDim));

        // Checked arithmetic — see MultiHeadAttentionScoresFanout for the
        // overflow-mode rationale.
        long perHeadA, perHeadV, perHeadO, perBatchA, perBatchV, perBatchO;
        long requiredA, requiredV, requiredO;
        try
        {
            checked
            {
                perHeadA = (long)seqLen * seqLen;
                perHeadV = (long)seqLen * headDim;
                perHeadO = (long)seqLen * headDim;
                perBatchA = perHeadA * numHeads;
                perBatchV = perHeadV * numHeads;
                perBatchO = perHeadO * numHeads;
                requiredA = (long)batch * perBatchA;
                requiredV = (long)batch * perBatchV;
                requiredO = (long)batch * perBatchO;
            }
        }
        catch (OverflowException)
        {
            throw new ArgumentException(
                $"MHA output shape too large for fanout dispatch: B={batch}, H={numHeads}, S={seqLen}, D={headDim}.");
        }
        long elemBytes = sizeof(float);

        // Validate buffer capacity BEFORE the per-head offset math (see
        // MultiHeadAttentionScoresFanout for the rationale).
        if (attention.Size < requiredA)
            throw new ArgumentException(
                $"attention buffer too small: capacity={attention.Size} elements, required={requiredA} for shape [B={batch}, H={numHeads}, S={seqLen}].",
                nameof(attention));
        if (V.Size < requiredV)
            throw new ArgumentException(
                $"V buffer too small: capacity={V.Size} elements, required={requiredV} for shape [B={batch}, H={numHeads}, S={seqLen}, D={headDim}].",
                nameof(V));
        if (output.Size < requiredO)
            throw new ArgumentException(
                $"output buffer too small: capacity={output.Size} elements, required={requiredO} for shape [B={batch}, H={numHeads}, S={seqLen}, D={headDim}].",
                nameof(output));

        var launches = new System.Collections.Generic.List<System.Action<AiDotNet.Tensors.Engines.Gpu.IGpuStream>>(batch * numHeads);
        for (int bi = 0; bi < batch; bi++)
        {
            for (int h = 0; h < numHeads; h++)
            {
                int batchIdx = bi, headIdx = h;
                launches.Add(_ =>
                {
                    float aVal = alpha, bVal = beta;
                    IntPtr aPtr = (IntPtr)((long)attention.Handle
                        + (batchIdx * perBatchA + headIdx * perHeadA) * elemBytes);
                    IntPtr vPtr = (IntPtr)((long)V.Handle
                        + (batchIdx * perBatchV + headIdx * perHeadV) * elemBytes);
                    IntPtr oPtr = (IntPtr)((long)output.Handle
                        + (batchIdx * perBatchO + headIdx * perHeadO) * elemBytes);

                    // output[h] = attention[h] · V[h]
                    // Row-major attention [seqLen, seqLen] · V [seqLen, headDim] = output [seqLen, headDim].
                    // For cuBLAS column-major: V_col^T · attention_col^T -> output_col^T (which reads as row-major output).
                    CuBlasNative.CheckCublasStatus(
                        CuBlasNative.cublasSgemm(
                            _cublasHandle,
                            CublasOperation.None,
                            CublasOperation.None,
                            headDim, seqLen, seqLen,
                            ref aVal,
                            vPtr, headDim,
                            aPtr, seqLen,
                            ref bVal,
                            oPtr, headDim),
                        $"cublasSgemm MHA output batch={batchIdx} head={headIdx}");
                });
            }
        }

        using var b1 = scheduler.Dispatch(launches);
        scheduler.SynchronizeEvents(b1);
    }

    /// <summary>
    /// Mixed-precision counterpart to <see cref="BatchedGemmFanout"/>:
    /// fp16 / bf16 input A, B + fp32 accumulator output C. Routes each
    /// slice through <c>cublasGemmEx</c> with <c>COMPUTE_32F</c> on its
    /// own pool stream.
    /// <para>
    /// Combines #335 multi-stream dispatch with #337 mixed precision —
    /// the standard AMP attention pattern in transformer backbones.
    /// </para>
    /// </summary>
    /// <param name="useBFloat16">When true, A/B are bfloat16 (CUDA_R_16BF).
    /// Otherwise fp16 (CUDA_R_16F). BF16 requires Ampere+ (cc >= 8.0).</param>
    public unsafe void BatchedGemmExFanout(
        IGpuBuffer A, IGpuBuffer B, IGpuBuffer C,
        int M, int N, int K, int batchCount,
        AiDotNet.Tensors.Engines.Gpu.GpuStreamScheduler scheduler,
        bool useBFloat16 = false,
        float alpha = 1.0f, float beta = 0.0f)
    {
        if (!IsAvailable) throw new InvalidOperationException("CUDA backend is not available.");
        if (scheduler is null) throw new ArgumentNullException(nameof(scheduler));
        // A/B are 2-byte (fp16/bf16); C is fp32. Use the byte-aware
        // validator — the element-count overload would interpret A.Size
        // as float elements and accept undersized half buffers.
        ValidateBatchedGemmArgsBytes(A, B, C, M, N, K, batchCount,
            elemBytesA: 2, elemBytesB: 2, elemBytesC: sizeof(float));

        if (useBFloat16 && _ccMajor < 8)
            throw new NotSupportedException(
                $"BFloat16 GEMM requires compute capability >= 8.0 (Ampere+); device reports {_ccMajor}.{_ccMinor}.");

        int inDtype = useBFloat16 ? CuBlasNative.CUDA_R_16BF : CuBlasNative.CUDA_R_16F;
        long strideA = (long)M * K;
        long strideB = (long)K * N;
        long strideC = (long)M * N;
        long elemBytesIn = 2;
        long elemBytesOut = sizeof(float);

        var launches = new System.Collections.Generic.List<System.Action<AiDotNet.Tensors.Engines.Gpu.IGpuStream>>(batchCount);
        for (int b = 0; b < batchCount; b++)
        {
            int slice = b;
            launches.Add(_ =>
            {
                float aVal = alpha, bVal = beta;
                IntPtr alphaPtr = (IntPtr)(&aVal);
                IntPtr betaPtr = (IntPtr)(&bVal);
                IntPtr aPtr = (IntPtr)((long)A.Handle + slice * strideA * elemBytesIn);
                IntPtr bPtr = (IntPtr)((long)B.Handle + slice * strideB * elemBytesIn);
                IntPtr cPtr = (IntPtr)((long)C.Handle + slice * strideC * elemBytesOut);

                CuBlasNative.CheckCublasStatus(
                    CuBlasNative.cublasGemmEx(
                        _cublasHandle,
                        CublasOperation.None, CublasOperation.None,
                        N, M, K,
                        alphaPtr,
                        bPtr, inDtype, N,
                        aPtr, inDtype, K,
                        betaPtr,
                        cPtr, CuBlasNative.CUDA_R_32F, N,
                        CuBlasNative.CUBLAS_COMPUTE_32F,
                        0 /* CUBLAS_GEMM_DEFAULT */),
                    $"cublasGemmEx slice {slice}");
            });
        }

        using var batch = scheduler.Dispatch(launches);
        scheduler.SynchronizeEvents(batch);
    }

    /// <summary>
    /// Double-precision counterpart to <see cref="BatchedGemmFanout"/>.
    /// Fans <paramref name="batchCount"/> independent <c>cublasDgemm</c>
    /// launches across the scheduler's stream pool — same pattern as the
    /// fp32 variant but for <c>double</c> input/output buffers.
    /// </summary>
    public void BatchedDgemmFanout(
        IGpuBuffer A, IGpuBuffer B, IGpuBuffer C,
        int M, int N, int K, int batchCount,
        AiDotNet.Tensors.Engines.Gpu.GpuStreamScheduler scheduler,
        double alpha = 1.0, double beta = 0.0)
    {
        if (!IsAvailable) throw new InvalidOperationException("CUDA backend is not available.");
        if (scheduler is null) throw new ArgumentNullException(nameof(scheduler));
        // All three buffers are fp64 (8-byte). The fp32-shaped validator
        // would underestimate the byte requirement by 2× and accept a
        // half-sized double buffer that then overruns inside cublasDgemm.
        ValidateBatchedGemmArgsBytes(A, B, C, M, N, K, batchCount,
            elemBytesA: sizeof(double), elemBytesB: sizeof(double), elemBytesC: sizeof(double));

        long strideA = (long)M * K;
        long strideB = (long)K * N;
        long strideC = (long)M * N;
        long elemBytes = sizeof(double);

        var launches = new System.Collections.Generic.List<System.Action<AiDotNet.Tensors.Engines.Gpu.IGpuStream>>(batchCount);
        for (int b = 0; b < batchCount; b++)
        {
            int slice = b;
            launches.Add(_ =>
            {
                double aVal = alpha, bVal = beta;
                IntPtr aPtr = (IntPtr)((long)A.Handle + slice * strideA * elemBytes);
                IntPtr bPtr = (IntPtr)((long)B.Handle + slice * strideB * elemBytes);
                IntPtr cPtr = (IntPtr)((long)C.Handle + slice * strideC * elemBytes);
                CuBlasNative.CheckCublasStatus(
                    CuBlasNative.cublasDgemm(
                        _cublasHandle,
                        CublasOperation.None,
                        CublasOperation.None,
                        N, M, K,
                        ref aVal,
                        bPtr, N,
                        aPtr, K,
                        ref bVal,
                        cPtr, N),
                    $"cublasDgemm slice {slice}");
            });
        }

        using var batch = scheduler.Dispatch(launches);
        scheduler.SynchronizeEvents(batch);
    }

    public IGpuBuffer GemmBiasRelu(IGpuBuffer A, IGpuBuffer B, IGpuBuffer bias, int M, int N, int K)
    {
        ValidateBiasBuffer(bias, N);
        IGpuBuffer? temp = null;
        IGpuBuffer? output = null;
        try
        {
            temp = GemmBias(A, B, bias, M, N, K);
            output = AllocateBuffer(M * N);
            Relu(temp, output, M * N);
            // Synchronize removed: kernel serializes on same stream, temp pool return doesn't free memory
            temp.Dispose();
            return output;
        }
        catch
        {
            output?.Dispose();
            temp?.Dispose();
            throw;
        }
    }

    public IGpuBuffer GemmBiasGelu(IGpuBuffer A, IGpuBuffer B, IGpuBuffer bias, int M, int N, int K)
    {
        ValidateBiasBuffer(bias, N);
        IGpuBuffer? temp = null;
        IGpuBuffer? output = null;
        try
        {
            temp = GemmBias(A, B, bias, M, N, K);
            output = AllocateBuffer(M * N);
            Gelu(temp, output, M * N);
            temp.Dispose();
            return output;
        }
        catch
        {
            output?.Dispose();
            temp?.Dispose();
            throw;
        }
    }

    public IGpuBuffer GemmBiasSigmoid(IGpuBuffer A, IGpuBuffer B, IGpuBuffer bias, int M, int N, int K)
    {
        ValidateBiasBuffer(bias, N);
        IGpuBuffer? temp = null;
        IGpuBuffer? output = null;
        try
        {
            temp = GemmBias(A, B, bias, M, N, K);
            output = AllocateBuffer(M * N);
            Sigmoid(temp, output, M * N);
            temp.Dispose();
            return output;
        }
        catch
        {
            output?.Dispose();
            temp?.Dispose();
            throw;
        }
    }

    public IGpuBuffer GemmBiasTanh(IGpuBuffer A, IGpuBuffer B, IGpuBuffer bias, int M, int N, int K)
    {
        ValidateBiasBuffer(bias, N);
        IGpuBuffer? temp = null;
        IGpuBuffer? output = null;
        try
        {
            temp = GemmBias(A, B, bias, M, N, K);
            output = AllocateBuffer(M * N);
            Tanh(temp, output, M * N);
            temp.Dispose();
            return output;
        }
        catch
        {
            output?.Dispose();
            temp?.Dispose();
            throw;
        }
    }

    public IGpuBuffer GemmBias(IGpuBuffer A, IGpuBuffer B, IGpuBuffer bias, int M, int N, int K)
    {
        ValidateBiasBuffer(bias, N);
        var output = MatMul(A, B, M, N, K);
        ApplyBiasInPlace(output, bias, M, N);
        return output;
    }

    public IGpuBuffer GemmBiasSwish(IGpuBuffer A, IGpuBuffer B, IGpuBuffer bias, int M, int N, int K)
    {
        ValidateBiasBuffer(bias, N);
        IGpuBuffer? temp = null;
        IGpuBuffer? output = null;
        try
        {
            temp = GemmBias(A, B, bias, M, N, K);
            output = AllocateBuffer(M * N);
            Silu(temp, output, M * N);
            temp.Dispose();
            return output;
        }
        catch
        {
            output?.Dispose();
            temp?.Dispose();
            throw;
        }
    }

    public unsafe IGpuBuffer GemmBiasLeakyRelu(IGpuBuffer A, IGpuBuffer B, IGpuBuffer bias, int M, int N, int K, float alpha = 0.01f)
    {
        ValidateBiasBuffer(bias, N);
        var output = AllocateBuffer(M * N);

        // LeakyReLU kernel has an extra alpha parameter
        string wmmaName = "gemm_bias_leaky_relu_wmma";
        string scalarName = "gemm_bias_leaky_relu";
        bool useWmma = _hasWmmaSupport && _kernelCache.TryGetValue(wmmaName, out var wmmaKernel);
        string kernelName = useWmma ? wmmaName : scalarName;

        if (!_kernelCache.TryGetValue(kernelName, out var kernel))
            throw new InvalidOperationException($"CUDA fused kernel not found: {kernelName}");

        using var _ = PushContext();
        IntPtr aPtr = A.Handle;
        IntPtr bPtr = B.Handle;
        IntPtr biasPtr = bias.Handle;
        IntPtr outPtr = output.Handle;
        int m = M, n = N, k = K;

        void** args = stackalloc void*[8];
        args[0] = &aPtr;
        args[1] = &bPtr;
        args[2] = &biasPtr;
        args[3] = &outPtr;
        args[4] = &m;
        args[5] = &n;
        args[6] = &k;
        args[7] = &alpha;

        if (useWmma)
        {
            const int WMMA_TILE = 32;
            uint gridX = (uint)((N + WMMA_TILE - 1) / WMMA_TILE);
            uint gridY = (uint)((M + WMMA_TILE - 1) / WMMA_TILE);
            CuBlasNative.CheckCudaResult(
                CudaNativeBindings.cuLaunchKernel(kernel, gridX, gridY, 1, 128, 1, 1,
                    0, _stream, (IntPtr)args, IntPtr.Zero),
                "cuLaunchKernel(WMMA LeakyReLU)");
        }
        else
        {
            const int BM = 128;
            const int BN = 128;
            uint gridX = (uint)((N + BN - 1) / BN);
            uint gridY = (uint)((M + BM - 1) / BM);
            LaunchKernel2D(kernel, gridX, gridY, 1, 16, 16, args);
        }

        return output;
    }

    public unsafe void BiasAdd(IGpuBuffer A, IGpuBuffer bias, IGpuBuffer C, int M, int N)
    {
        if (!_kernelCache.TryGetValue("bias_add_out", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: bias_add_out");

        using var _ = PushContext();
        uint gridX = (uint)((N + DefaultBlockSize - 1) / DefaultBlockSize);
        uint gridY = (uint)M;
        IntPtr aPtr = A.Handle;
        IntPtr biasPtr = bias.Handle;
        IntPtr cPtr = C.Handle;
        int rows = M;
        int cols = N;
        void** args = stackalloc void*[5];
        args[0] = &aPtr;
        args[1] = &biasPtr;
        args[2] = &cPtr;
        args[3] = &rows;
        args[4] = &cols;
        LaunchKernel2D(kernel, gridX, gridY, DefaultBlockSize, 1, args);
    }

    public unsafe void Conv2DBiasAdd(IGpuBuffer output, IGpuBuffer bias, int batch, int channels, int spatialSize)
    {
        if (!_kernelCache.TryGetValue("conv2d_bias_add", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: conv2d_bias_add");

        using var _ = PushContext();
        // 2D grid: x=spatial, y=batch*channels — eliminates integer division in kernel
        uint gridX = (uint)((spatialSize + DefaultBlockSize - 1) / DefaultBlockSize);
        uint gridY = (uint)(batch * channels);
        IntPtr outPtr = output.Handle;
        IntPtr biasPtr = bias.Handle;
        void** args = stackalloc void*[5];
        args[0] = &outPtr;
        args[1] = &biasPtr;
        args[2] = &batch;
        args[3] = &channels;
        args[4] = &spatialSize;
        LaunchKernel2D(kernel, gridX, gridY, DefaultBlockSize, 1, args);
    }

    public void Add(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int size)
    {
        LaunchElementwiseKernel("add_vectors", A, B, C, size);
    }

    public void AddRelu(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int size) { Add(A, B, C, size); Relu(C, C, size); }
    public void AddSigmoid(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int size) { Add(A, B, C, size); Sigmoid(C, C, size); }
    public void AddGelu(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int size) { Add(A, B, C, size); Gelu(C, C, size); }

    public void Subtract(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int size)
    {
        LaunchElementwiseKernel("subtract_vectors", A, B, C, size);
    }

    public void Multiply(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int size)
    {
        LaunchElementwiseKernel("multiply_vectors", A, B, C, size);
    }

    public void Divide(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int size)
    {
        LaunchElementwiseKernel("divide_vectors", A, B, C, size);
    }

    public void Min(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int size)
    {
        LaunchElementwiseKernel("min_vectors", A, B, C, size);
    }

    public void Max(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int size)
    {
        LaunchElementwiseKernel("max_vectors", A, B, C, size);
    }

    public void Scale(IGpuBuffer A, IGpuBuffer B, float scalar, int size)
    {
        LaunchScaleKernel(A, B, scalar, size);
    }

    public unsafe void StridedGather(IGpuBuffer src, IGpuBuffer dst, int offset, int stride, int count)
    {
        if (count <= 0) return;
        if (offset < 0) throw new ArgumentOutOfRangeException(nameof(offset));
        if (stride <= 0) throw new ArgumentOutOfRangeException(nameof(stride));
        int lastIdx = offset + (count - 1) * stride;
        if (lastIdx >= src.Size) throw new ArgumentOutOfRangeException(nameof(count), $"Strided gather would access index {lastIdx} but source size is {src.Size}.");
        if (count > dst.Size) throw new ArgumentOutOfRangeException(nameof(count), $"Count ({count}) exceeds destination size ({dst.Size}).");

        if (!_kernelCache.TryGetValue("strided_gather", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: strided_gather. Register the kernel or use CPU fallback.");

        using var _ = PushContext();
        uint grid = (uint)((count + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr srcPtr = src.Handle;
        IntPtr dstPtr = dst.Handle;
        int off = offset;
        int str = stride;
        int cnt = count;
        void** args = stackalloc void*[5];
        args[0] = &srcPtr;
        args[1] = &dstPtr;
        args[2] = &off;
        args[3] = &str;
        args[4] = &cnt;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    public unsafe void StridedScatter(IGpuBuffer src, IGpuBuffer dst, int offset, int stride, int count)
    {
        if (count <= 0) return;
        if (offset < 0) throw new ArgumentOutOfRangeException(nameof(offset));
        if (stride <= 0) throw new ArgumentOutOfRangeException(nameof(stride));
        int lastIdx = offset + (count - 1) * stride;
        if (lastIdx >= dst.Size) throw new ArgumentOutOfRangeException(nameof(count), $"Strided scatter would access index {lastIdx} but destination size is {dst.Size}.");
        if (count > src.Size) throw new ArgumentOutOfRangeException(nameof(count), $"Count ({count}) exceeds source size ({src.Size}).");

        if (!_kernelCache.TryGetValue("strided_scatter", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: strided_scatter. Register the kernel or use CPU fallback.");

        using var _ = PushContext();
        uint grid = (uint)((count + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr srcPtr = src.Handle;
        IntPtr dstPtr = dst.Handle;
        int off = offset;
        int str = stride;
        int cnt = count;
        void** args = stackalloc void*[5];
        args[0] = &srcPtr;
        args[1] = &dstPtr;
        args[2] = &off;
        args[3] = &str;
        args[4] = &cnt;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    public void Power(IGpuBuffer A, IGpuBuffer B, float exponent, int size)
    {
        LaunchUnaryWithScalarKernel("power_scalar", A, B, exponent, size);
    }

    public void Abs(IGpuBuffer A, IGpuBuffer B, int size)
    {
        LaunchUnaryKernel("abs_vector", A, B, size);
    }

    public void Exp(IGpuBuffer A, IGpuBuffer B, int size)
    {
        LaunchUnaryKernel("exp_vector", A, B, size);
    }

    public void Exp2(IGpuBuffer A, IGpuBuffer B, int size)
    {
        LaunchUnaryKernel("exp2_vector", A, B, size);
    }

    public void Exp10(IGpuBuffer A, IGpuBuffer B, int size)
    {
        LaunchUnaryKernel("exp10_vector", A, B, size);
    }

    public void ExpM1(IGpuBuffer A, IGpuBuffer B, int size)
    {
        LaunchUnaryKernel("expm1_vector", A, B, size);
    }

    public void Log(IGpuBuffer A, IGpuBuffer B, int size)
    {
        LaunchUnaryKernel("log_vector", A, B, size);
    }

    public void Log2(IGpuBuffer A, IGpuBuffer B, int size)
    {
        LaunchUnaryKernel("log2_vector", A, B, size);
    }

    public void Log1P(IGpuBuffer A, IGpuBuffer B, int size)
    {
        LaunchUnaryKernel("log1p_vector", A, B, size);
    }

    public void Sqrt(IGpuBuffer A, IGpuBuffer B, int size)
    {
        LaunchUnaryKernel("sqrt_vector", A, B, size);
    }

    public void Sign(IGpuBuffer A, IGpuBuffer B, int size)
    {
        LaunchUnaryKernel("sign_vector", A, B, size);
    }

    public void Relu(IGpuBuffer A, IGpuBuffer B, int size)
    {
        LaunchUnaryKernel("relu", A, B, size);
    }

    public void Sigmoid(IGpuBuffer A, IGpuBuffer B, int size)
    {
        LaunchUnaryKernel("sigmoid", A, B, size);
    }

    public void Tanh(IGpuBuffer A, IGpuBuffer B, int size)
    {
        LaunchUnaryKernel("tanh_activation", A, B, size);
    }

    public void Gelu(IGpuBuffer A, IGpuBuffer B, int size)
    {
        LaunchUnaryKernel("gelu", A, B, size);
    }

    public void Softmax(IGpuBuffer A, IGpuBuffer B, int batchSize, int features)
    {
        LaunchSoftmaxKernel(A, B, batchSize, features);
    }

    public unsafe void Squash(IGpuBuffer input, IGpuBuffer output, int numCapsules, int capsuleDim, float epsilon)
    {
        if (!_kernelCache.TryGetValue("squash", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: squash");

        using var _ = PushContext();
        uint grid = (uint)((numCapsules + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr inputPtr = input.Handle;
        IntPtr outputPtr = output.Handle;
        int numCaps = numCapsules;
        int capDim = capsuleDim;
        float eps = epsilon;
        void** args = stackalloc void*[5];
        args[0] = &inputPtr;
        args[1] = &outputPtr;
        args[2] = &numCaps;
        args[3] = &capDim;
        args[4] = &eps;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    public unsafe void SquashBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradInput, int numCapsules, int capsuleDim, float epsilon)
    {
        if (!_kernelCache.TryGetValue("squash_backward", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: squash_backward");

        using var _ = PushContext();
        uint grid = (uint)((numCapsules + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr gradOutPtr = gradOutput.Handle;
        IntPtr inputPtr = input.Handle;
        IntPtr gradInPtr = gradInput.Handle;
        int numCaps = numCapsules;
        int capDim = capsuleDim;
        float eps = epsilon;
        void** args = stackalloc void*[6];
        args[0] = &gradOutPtr;
        args[1] = &inputPtr;
        args[2] = &gradInPtr;
        args[3] = &numCaps;
        args[4] = &capDim;
        args[5] = &eps;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    public unsafe void CapsulePredictions(IGpuBuffer input, IGpuBuffer weights, IGpuBuffer output,
        int batchSize, int inputCapsules, int inputDim, int outputCapsules, int outputDim)
    {
        if (!_kernelCache.TryGetValue("capsule_predictions", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: capsule_predictions");

        using var _ = PushContext();
        int totalOutputs = batchSize * inputCapsules * outputCapsules * outputDim;
        uint grid = (uint)((totalOutputs + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr inputPtr = input.Handle;
        IntPtr weightsPtr = weights.Handle;
        IntPtr outputPtr = output.Handle;
        int bs = batchSize;
        int inCaps = inputCapsules;
        int inDim = inputDim;
        int outCaps = outputCapsules;
        int outDim = outputDim;
        void** args = stackalloc void*[8];
        args[0] = &inputPtr;
        args[1] = &weightsPtr;
        args[2] = &outputPtr;
        args[3] = &bs;
        args[4] = &inCaps;
        args[5] = &inDim;
        args[6] = &outCaps;
        args[7] = &outDim;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    public unsafe void CapsuleTransform(IGpuBuffer input, IGpuBuffer weights, IGpuBuffer output,
        int batchSize, int inputCapsules, int inputDim, int numCapsules, int capsuleDim)
    {
        if (!_kernelCache.TryGetValue("capsule_transform", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: capsule_transform");

        using var _ = PushContext();
        int totalOutputs = batchSize * inputCapsules * numCapsules * capsuleDim;
        uint grid = (uint)((totalOutputs + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr inputPtr = input.Handle;
        IntPtr weightsPtr = weights.Handle;
        IntPtr outputPtr = output.Handle;
        int bs = batchSize;
        int inCaps = inputCapsules;
        int inDim = inputDim;
        int nCaps = numCapsules;
        int capDim = capsuleDim;
        void** args = stackalloc void*[8];
        args[0] = &inputPtr;
        args[1] = &weightsPtr;
        args[2] = &outputPtr;
        args[3] = &bs;
        args[4] = &inCaps;
        args[5] = &inDim;
        args[6] = &nCaps;
        args[7] = &capDim;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    public unsafe void CapsuleWeightedSum(IGpuBuffer coupling, IGpuBuffer predictions, IGpuBuffer output,
        int batchSize, int inputCapsules, int outputCapsules, int capsuleDim)
    {
        if (!_kernelCache.TryGetValue("capsule_weighted_sum", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: capsule_weighted_sum");

        using var _ = PushContext();
        int totalOutputs = batchSize * outputCapsules * capsuleDim;
        uint grid = (uint)((totalOutputs + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr couplingPtr = coupling.Handle;
        IntPtr predsPtr = predictions.Handle;
        IntPtr outputPtr = output.Handle;
        int bs = batchSize;
        int inCaps = inputCapsules;
        int outCaps = outputCapsules;
        int capDim = capsuleDim;
        void** args = stackalloc void*[7];
        args[0] = &couplingPtr;
        args[1] = &predsPtr;
        args[2] = &outputPtr;
        args[3] = &bs;
        args[4] = &inCaps;
        args[5] = &outCaps;
        args[6] = &capDim;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    public unsafe void CapsuleAgreement(IGpuBuffer predictions, IGpuBuffer output, IGpuBuffer agreement,
        int batchSize, int inputCapsules, int outputCapsules, int capsuleDim)
    {
        if (!_kernelCache.TryGetValue("capsule_agreement", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: capsule_agreement");

        using var _ = PushContext();
        int totalOutputs = batchSize * inputCapsules * outputCapsules;
        uint grid = (uint)((totalOutputs + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr predsPtr = predictions.Handle;
        IntPtr outputPtr = output.Handle;
        IntPtr agreementPtr = agreement.Handle;
        int bs = batchSize;
        int inCaps = inputCapsules;
        int outCaps = outputCapsules;
        int capDim = capsuleDim;
        void** args = stackalloc void*[7];
        args[0] = &predsPtr;
        args[1] = &outputPtr;
        args[2] = &agreementPtr;
        args[3] = &bs;
        args[4] = &inCaps;
        args[5] = &outCaps;
        args[6] = &capDim;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    public unsafe void TileBatch(IGpuBuffer input, IGpuBuffer output, int repeats, int innerSize)
    {
        if (!_kernelCache.TryGetValue("tile_batch", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: tile_batch");

        using var _ = PushContext();
        int totalSize = repeats * innerSize;
        uint grid = (uint)((totalSize + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr inputPtr = input.Handle;
        IntPtr outputPtr = output.Handle;
        int reps = repeats;
        int inner = innerSize;
        void** args = stackalloc void*[4];
        args[0] = &inputPtr;
        args[1] = &outputPtr;
        args[2] = &reps;
        args[3] = &inner;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    public unsafe void TileAxis(IGpuBuffer input, IGpuBuffer output, int outerSize, int axisSize, int innerSize, int repeats)
    {
        if (!_kernelCache.TryGetValue("tile_axis", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: tile_axis");

        using var _ = PushContext();
        int totalSize = outerSize * axisSize * repeats * innerSize;
        uint grid = (uint)((totalSize + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr inputPtr = input.Handle;
        IntPtr outputPtr = output.Handle;
        int outer = outerSize;
        int axis = axisSize;
        int inner = innerSize;
        int reps = repeats;
        void** args = stackalloc void*[6];
        args[0] = &inputPtr;
        args[1] = &outputPtr;
        args[2] = &outer;
        args[3] = &axis;
        args[4] = &inner;
        args[5] = &reps;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    #region Trigonometric Operations

    public void Sin(IGpuBuffer A, IGpuBuffer B, int size)
    {
        LaunchUnaryKernel("sin_vector", A, B, size);
    }

    public void Cos(IGpuBuffer A, IGpuBuffer B, int size)
    {
        LaunchUnaryKernel("cos_vector", A, B, size);
    }

    public void Tan(IGpuBuffer A, IGpuBuffer B, int size)
    {
        LaunchUnaryKernel("tan_vector", A, B, size);
    }

    public void Asin(IGpuBuffer A, IGpuBuffer B, int size)
    {
        LaunchUnaryKernel("asin_vector", A, B, size);
    }

    public void Acos(IGpuBuffer A, IGpuBuffer B, int size)
    {
        LaunchUnaryKernel("acos_vector", A, B, size);
    }

    public void Atan(IGpuBuffer A, IGpuBuffer B, int size)
    {
        LaunchUnaryKernel("atan_vector", A, B, size);
    }

    #endregion

    #region Hyperbolic Operations

    public void Sinh(IGpuBuffer A, IGpuBuffer B, int size)
    {
        LaunchUnaryKernel("sinh_vector", A, B, size);
    }

    public void Cosh(IGpuBuffer A, IGpuBuffer B, int size)
    {
        LaunchUnaryKernel("cosh_vector", A, B, size);
    }

    public void Asinh(IGpuBuffer A, IGpuBuffer B, int size)
    {
        LaunchUnaryKernel("asinh_vector", A, B, size);
    }

    public void Acosh(IGpuBuffer A, IGpuBuffer B, int size)
    {
        LaunchUnaryKernel("acosh_vector", A, B, size);
    }

    public void Atanh(IGpuBuffer A, IGpuBuffer B, int size)
    {
        LaunchUnaryKernel("atanh_vector", A, B, size);
    }

    #endregion

    #region Additional Unary Operations

    public void Reciprocal(IGpuBuffer A, IGpuBuffer B, int size)
    {
        LaunchUnaryKernel("reciprocal_vector", A, B, size);
    }

    public void Cbrt(IGpuBuffer A, IGpuBuffer B, int size)
    {
        LaunchUnaryKernel("cbrt_vector", A, B, size);
    }

    public void Log10(IGpuBuffer A, IGpuBuffer B, int size)
    {
        LaunchUnaryKernel("log10_vector", A, B, size);
    }

    public void Negate(IGpuBuffer A, IGpuBuffer B, int size)
    {
        LaunchUnaryKernel("negate_vector", A, B, size);
    }

    public void Floor(IGpuBuffer A, IGpuBuffer B, int size)
    {
        LaunchUnaryKernel("floor_vector", A, B, size);
    }

    public void Ceiling(IGpuBuffer A, IGpuBuffer B, int size)
    {
        LaunchUnaryKernel("ceil_vector", A, B, size);
    }

    public void Round(IGpuBuffer A, IGpuBuffer B, int size)
    {
        LaunchUnaryKernel("round_vector", A, B, size);
    }

    public void Truncate(IGpuBuffer A, IGpuBuffer B, int size)
    {
        LaunchUnaryKernel("trunc_vector", A, B, size);
    }

    #endregion

    public unsafe void Enforce2x4Sparsity(IGpuBuffer denseInput, IGpuBuffer sparseValues, IGpuBuffer sparseIndices, int M, int K)
    {
        if (K % 4 != 0)
            throw new ArgumentException("K must be a multiple of 4 for 2:4 structured sparsity.");
        if (M <= 0 || K <= 0)
            throw new ArgumentOutOfRangeException(nameof(M), "M and K must be positive.");
        // Validate buffer sizes: dense is M*K, sparse values is M*(K/2), indices is M*(K/4)
        if ((long)M * K > denseInput.Size)
            throw new ArgumentOutOfRangeException(nameof(M), $"M*K ({(long)M * K}) exceeds denseInput length ({denseInput.Size}).");
        if ((long)M * (K / 2) > sparseValues.Size)
            throw new ArgumentOutOfRangeException(nameof(M), $"M*(K/2) ({(long)M * (K / 2)}) exceeds sparseValues length ({sparseValues.Size}).");

        if (!_kernelCache.TryGetValue("enforce_2x4_sparsity", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: enforce_2x4_sparsity");

        using var _ = PushContext();
        IntPtr densePtr = denseInput.Handle;
        IntPtr valsPtr = sparseValues.Handle;
        IntPtr idxPtr = sparseIndices.Handle;
        int mVal = M, kVal = K;
        void** args = stackalloc void*[5];
        args[0] = &densePtr;
        args[1] = &valsPtr;
        args[2] = &idxPtr;
        args[3] = &mVal;
        args[4] = &kVal;
        long totalGroupsLong = (long)M * (K / 4);
        uint totalGroups = totalGroupsLong > uint.MaxValue ? uint.MaxValue : (uint)totalGroupsLong;
        uint grid = (totalGroups + DefaultBlockSize - 1) / DefaultBlockSize;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    public unsafe void Decompress2x4Sparse(IGpuBuffer sparseValues, IGpuBuffer sparseIndices, IGpuBuffer denseOutput, int M, int K)
    {
        if (K % 4 != 0)
            throw new ArgumentException("K must be a multiple of 4 for 2:4 structured sparsity.");

        if (!_kernelCache.TryGetValue("decompress_2x4_sparse", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: decompress_2x4_sparse");

        using var _ = PushContext();
        IntPtr valsPtr = sparseValues.Handle;
        IntPtr idxPtr = sparseIndices.Handle;
        IntPtr densePtr = denseOutput.Handle;
        int mVal = M, kVal = K;
        void** args = stackalloc void*[5];
        args[0] = &valsPtr;
        args[1] = &idxPtr;
        args[2] = &densePtr;
        args[3] = &mVal;
        args[4] = &kVal;
        uint total = (uint)(M * K);
        uint grid = (total + DefaultBlockSize - 1) / DefaultBlockSize;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    public unsafe void SparseGemm(IGpuBuffer sparseAValues, IGpuBuffer sparseAIndices, IGpuBuffer B, IGpuBuffer C, int M, int N, int K, float alpha = 1.0f, float beta = 0.0f)
    {
        if (K % 4 != 0)
            throw new ArgumentException("K must be a multiple of 4 for 2:4 sparse GEMM.");

        if (!_kernelCache.TryGetValue("sparse_gemm_2x4", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: sparse_gemm_2x4");

        using var _ = PushContext();
        IntPtr valsPtr = sparseAValues.Handle;
        IntPtr idxPtr = sparseAIndices.Handle;
        IntPtr bPtr = B.Handle;
        IntPtr cPtr = C.Handle;
        int mVal = M, nVal = N, kVal = K;
        float alphaVal = alpha, betaVal = beta;
        void** args = stackalloc void*[9];
        args[0] = &valsPtr;
        args[1] = &idxPtr;
        args[2] = &bPtr;
        args[3] = &cPtr;
        args[4] = &mVal;
        args[5] = &nVal;
        args[6] = &kVal;
        args[7] = &alphaVal;
        args[8] = &betaVal;
        const int blockDim = 16;
        uint gridX = (uint)((N + blockDim - 1) / blockDim);
        uint gridY = (uint)((M + blockDim - 1) / blockDim);
        LaunchKernel2D(kernel, gridX, gridY, blockDim, blockDim, args);
    }

    public unsafe IGpuBuffer SparseGemmBiasRelu(IGpuBuffer sparseAValues, IGpuBuffer sparseAIndices, IGpuBuffer B, IGpuBuffer bias, int M, int N, int K)
    {
        if (K % 4 != 0)
            throw new ArgumentException("K must be a multiple of 4 for 2:4 sparse GEMM.");

        if (!_kernelCache.TryGetValue("sparse_gemm_bias_relu", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: sparse_gemm_bias_relu");

        var output = AllocateBuffer(M * N);
        using var _ = PushContext();
        IntPtr valsPtr = sparseAValues.Handle;
        IntPtr idxPtr = sparseAIndices.Handle;
        IntPtr bPtr = B.Handle;
        IntPtr biasPtr = bias.Handle;
        IntPtr outPtr = output.Handle;
        int mVal = M, nVal = N, kVal = K;
        void** args = stackalloc void*[8];
        args[0] = &valsPtr;
        args[1] = &idxPtr;
        args[2] = &bPtr;
        args[3] = &biasPtr;
        args[4] = &outPtr;
        args[5] = &mVal;
        args[6] = &nVal;
        args[7] = &kVal;
        const int blockDim = 16;
        uint gridX = (uint)((N + blockDim - 1) / blockDim);
        uint gridY = (uint)((M + blockDim - 1) / blockDim);
        LaunchKernel2D(kernel, gridX, gridY, blockDim, blockDim, args);
        return output;
    }

    #region CSR Sparse Operations (General Sparsity)

    /// <inheritdoc/>
    public unsafe void CsrSpMM(
        IGpuBuffer csrValues,
        IGpuBuffer csrColIndices,
        IGpuBuffer csrRowPointers,
        IGpuBuffer denseB,
        IGpuBuffer output,
        int M, int K, int N, int nnz)
    {
        if (!IsAvailable)
            throw new InvalidOperationException("CUDA backend is not available.");

        if (!_kernelCache.TryGetValue("csr_spmm", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: csr_spmm");

        using var _ = PushContext();

        // Launch configuration: rows x ceil(N/blockSize) grid
        uint gridX = (uint)M;
        uint gridY = (uint)((N + DefaultBlockSize - 1) / DefaultBlockSize);

        IntPtr valuesPtr = csrValues.Handle;
        IntPtr colIndicesPtr = csrColIndices.Handle;
        IntPtr rowPointersPtr = csrRowPointers.Handle;
        IntPtr denseBPtr = denseB.Handle;
        IntPtr outputPtr = output.Handle;

        void** args = stackalloc void*[9];
        args[0] = &valuesPtr;
        args[1] = &colIndicesPtr;
        args[2] = &rowPointersPtr;
        args[3] = &denseBPtr;
        args[4] = &outputPtr;
        args[5] = &M;
        args[6] = &K;
        args[7] = &N;
        args[8] = &nnz;

        LaunchKernel2D(kernel, gridX, gridY, (uint)DefaultBlockSize, 1, args);
    }

    /// <summary>
    /// Warp-per-row CSR SpMM (issue #515): launches <c>csr_spmm_warp</c>, where each
    /// warp (32 lanes) owns one output row and the lanes stride over columns. Best
    /// for small/moderate N (the lanes cover the columns with coalesced B reads);
    /// the 2-D <see cref="CsrSpMM"/> is better for wide N. Same numerics as
    /// <see cref="CsrSpMM"/> (per-output sequential reduction → deterministic).
    /// </summary>
    public unsafe void CsrSpMMWarp(
        IGpuBuffer csrValues,
        IGpuBuffer csrColIndices,
        IGpuBuffer csrRowPointers,
        IGpuBuffer denseB,
        IGpuBuffer output,
        int M, int K, int N, int nnz)
    {
        if (!IsAvailable)
            throw new InvalidOperationException("CUDA backend is not available.");

        if (!_kernelCache.TryGetValue("csr_spmm_warp", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: csr_spmm_warp");

        using var _ = PushContext();

        // One warp (32 lanes) per row; DefaultBlockSize threads/block => 8 warps/block.
        uint gridX = (uint)((M + 7) / 8);

        IntPtr valuesPtr = csrValues.Handle;
        IntPtr colIndicesPtr = csrColIndices.Handle;
        IntPtr rowPointersPtr = csrRowPointers.Handle;
        IntPtr denseBPtr = denseB.Handle;
        IntPtr outputPtr = output.Handle;

        void** args = stackalloc void*[9];
        args[0] = &valuesPtr;
        args[1] = &colIndicesPtr;
        args[2] = &rowPointersPtr;
        args[3] = &denseBPtr;
        args[4] = &outputPtr;
        args[5] = &M;
        args[6] = &K;
        args[7] = &N;
        args[8] = &nnz;

        LaunchKernel(kernel, gridX, (uint)DefaultBlockSize, args);
    }

    /// <summary>
    /// FP64 CSR SpMM (issue #515): launches <c>csr_spmm_double</c>. The CSR-value /
    /// dense-B / output buffers carry <c>double</c> payloads (byte buffers); the int
    /// CSR index buffers are unchanged. Same 2-D launch + numerics as the float
    /// <see cref="CsrSpMM"/>.
    /// </summary>
    public unsafe void CsrSpMMDouble(
        IGpuBuffer csrValues,
        IGpuBuffer csrColIndices,
        IGpuBuffer csrRowPointers,
        IGpuBuffer denseB,
        IGpuBuffer output,
        int M, int K, int N, int nnz)
    {
        if (!IsAvailable)
            throw new InvalidOperationException("CUDA backend is not available.");

        if (!_kernelCache.TryGetValue("csr_spmm_double", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: csr_spmm_double");

        using var _ = PushContext();

        uint gridX = (uint)M;
        uint gridY = (uint)((N + DefaultBlockSize - 1) / DefaultBlockSize);

        IntPtr valuesPtr = csrValues.Handle;
        IntPtr colIndicesPtr = csrColIndices.Handle;
        IntPtr rowPointersPtr = csrRowPointers.Handle;
        IntPtr denseBPtr = denseB.Handle;
        IntPtr outputPtr = output.Handle;

        void** args = stackalloc void*[9];
        args[0] = &valuesPtr;
        args[1] = &colIndicesPtr;
        args[2] = &rowPointersPtr;
        args[3] = &denseBPtr;
        args[4] = &outputPtr;
        args[5] = &M;
        args[6] = &K;
        args[7] = &N;
        args[8] = &nnz;

        LaunchKernel2D(kernel, gridX, gridY, (uint)DefaultBlockSize, 1, args);
    }

    /// <summary>
    /// 128-bit-vectorized CSR SpMM (issue #515): launches <c>csr_spmm_vec4</c>, where
    /// each thread computes 4 adjacent output columns via aligned <c>float4</c> loads.
    /// Requires <paramref name="N"/> % 4 == 0 (the caller's gate). Same numerics as
    /// <see cref="CsrSpMM"/> (per-output sequential reduction → deterministic).
    /// </summary>
    public unsafe void CsrSpMMVec4(
        IGpuBuffer csrValues,
        IGpuBuffer csrColIndices,
        IGpuBuffer csrRowPointers,
        IGpuBuffer denseB,
        IGpuBuffer output,
        int M, int K, int N, int nnz)
    {
        if (!IsAvailable)
            throw new InvalidOperationException("CUDA backend is not available.");

        // Fail fast on misaligned N: each thread computes 4 adjacent output
        // columns via float4 loads, so a non-multiple-of-4 N would silently
        // TRUNCATE the grid to N/4 thread-columns and leave the last
        // N % 4 output columns unwritten. Upstream routing gates on this,
        // but a direct call must not rely on it.
        if ((N & 3) != 0)
            throw new ArgumentException(
                $"CsrSpMMVec4 requires N divisible by 4 (got {N}); the float4-vectorized kernel " +
                "would silently skip the last N % 4 output columns. Use CsrSpMM for unaligned N.",
                nameof(N));

        if (!_kernelCache.TryGetValue("csr_spmm_vec4", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: csr_spmm_vec4");

        using var _ = PushContext();

        // Each thread owns 4 columns, so the y-grid spans N/4 thread-columns.
        uint gridX = (uint)M;
        uint gridY = (uint)((N / 4 + DefaultBlockSize - 1) / DefaultBlockSize);

        IntPtr valuesPtr = csrValues.Handle;
        IntPtr colIndicesPtr = csrColIndices.Handle;
        IntPtr rowPointersPtr = csrRowPointers.Handle;
        IntPtr denseBPtr = denseB.Handle;
        IntPtr outputPtr = output.Handle;

        void** args = stackalloc void*[9];
        args[0] = &valuesPtr;
        args[1] = &colIndicesPtr;
        args[2] = &rowPointersPtr;
        args[3] = &denseBPtr;
        args[4] = &outputPtr;
        args[5] = &M;
        args[6] = &K;
        args[7] = &N;
        args[8] = &nnz;

        LaunchKernel2D(kernel, gridX, gridY, (uint)DefaultBlockSize, 1, args);
    }

    /// <inheritdoc/>
    public unsafe void CsrSpMMBias(
        IGpuBuffer csrValues,
        IGpuBuffer csrColIndices,
        IGpuBuffer csrRowPointers,
        IGpuBuffer denseB,
        IGpuBuffer bias,
        IGpuBuffer output,
        int M, int K, int N, int nnz)
    {
        if (!IsAvailable)
            throw new InvalidOperationException("CUDA backend is not available.");

        if (!_kernelCache.TryGetValue("csr_spmm_bias", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: csr_spmm_bias");

        using var _ = PushContext();

        uint gridX = (uint)M;
        uint gridY = (uint)((N + DefaultBlockSize - 1) / DefaultBlockSize);

        IntPtr valuesPtr = csrValues.Handle;
        IntPtr colIndicesPtr = csrColIndices.Handle;
        IntPtr rowPointersPtr = csrRowPointers.Handle;
        IntPtr denseBPtr = denseB.Handle;
        IntPtr biasPtr = bias.Handle;
        IntPtr outputPtr = output.Handle;

        void** args = stackalloc void*[10];
        args[0] = &valuesPtr;
        args[1] = &colIndicesPtr;
        args[2] = &rowPointersPtr;
        args[3] = &denseBPtr;
        args[4] = &biasPtr;
        args[5] = &outputPtr;
        args[6] = &M;
        args[7] = &K;
        args[8] = &N;
        args[9] = &nnz;

        LaunchKernel2D(kernel, gridX, gridY, (uint)DefaultBlockSize, 1, args);
    }

    /// <inheritdoc/>
    public unsafe void ScatterAddEdges(
        IGpuBuffer input,
        IGpuBuffer sourceIndices,
        IGpuBuffer targetIndices,
        IGpuBuffer? edgeValues,
        IGpuBuffer output,
        int numNodes, int numEdges, int features)
    {
        if (!IsAvailable)
            throw new InvalidOperationException("CUDA backend is not available.");

        using var _ = PushContext();

        // First zero the output buffer
        ZeroBuffer(output, numNodes * features);

        IntPtr inputPtr = input.Handle;
        IntPtr sourcePtr = sourceIndices.Handle;
        IntPtr targetPtr = targetIndices.Handle;
        IntPtr edgeValuesPtr = edgeValues?.Handle ?? IntPtr.Zero;
        IntPtr outputPtr = output.Handle;
        int hasEdgeValues = edgeValues is not null ? 1 : 0;

        void** args2 = stackalloc void*[9];
        args2[0] = &inputPtr;
        args2[1] = &sourcePtr;
        args2[2] = &targetPtr;
        args2[3] = &edgeValuesPtr;
        args2[4] = &outputPtr;
        args2[5] = &numNodes;
        args2[6] = &numEdges;
        args2[7] = &features;
        args2[8] = &hasEdgeValues;

        if (GpuDeterminism.IsActive)
        {
            // Issue #382: per-target-node thread scans edges in fixed order.
            if (!_kernelCache.TryGetValue("scatter_add_edges_deterministic", out var krnlD))
                throw new InvalidOperationException("CUDA kernel not found: scatter_add_edges_deterministic");
            uint gridDX = (uint)numNodes;
            uint gridDY = (uint)((features + DefaultBlockSize - 1) / DefaultBlockSize);
            LaunchKernel2D(krnlD, gridDX, gridDY, (uint)DefaultBlockSize, 1, args2);
            return;
        }

        if (!_kernelCache.TryGetValue("scatter_add_edges", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: scatter_add_edges");

        uint gridX = (uint)numEdges;
        uint gridY = (uint)((features + DefaultBlockSize - 1) / DefaultBlockSize);
        LaunchKernel2D(kernel, gridX, gridY, (uint)DefaultBlockSize, 1, args2);
    }

    private unsafe void ZeroBuffer(IGpuBuffer buffer, int size)
    {
        if (!_kernelCache.TryGetValue("zero_buffer", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: zero_buffer");

        uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr bufferPtr = buffer.Handle;

        void** args = stackalloc void*[2];
        args[0] = &bufferPtr;
        args[1] = &size;

        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    private unsafe void LaunchKernel2D(IntPtr kernel, uint gridX, uint gridY, uint blockX, uint blockY, void** args)
    {
        CuBlasNative.CheckCudaResult(
            CudaNativeBindings.cuLaunchKernel(
                kernel,
                gridX, gridY, 1,
                blockX, blockY, 1,
                0,
                _stream,
                (IntPtr)args,
                IntPtr.Zero),
            "cuLaunchKernel");
    }

    /// <inheritdoc/>
    public unsafe void CsrSegmentedMax(
        IGpuBuffer csrColIndices,
        IGpuBuffer csrRowPointers,
        IGpuBuffer input,
        IGpuBuffer output,
        int M, int K, int N)
    {
        if (!IsAvailable)
            throw new InvalidOperationException("CUDA backend is not available.");

        if (!_kernelCache.TryGetValue("csr_segmented_max", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: csr_segmented_max");

        using var _ = PushContext();

        // Launch configuration: rows x ceil(N/blockSize) grid
        uint gridX = (uint)M;
        uint gridY = (uint)((N + DefaultBlockSize - 1) / DefaultBlockSize);

        IntPtr colIndicesPtr = csrColIndices.Handle;
        IntPtr rowPointersPtr = csrRowPointers.Handle;
        IntPtr inputPtr = input.Handle;
        IntPtr outputPtr = output.Handle;

        void** args = stackalloc void*[7];
        args[0] = &colIndicesPtr;
        args[1] = &rowPointersPtr;
        args[2] = &inputPtr;
        args[3] = &outputPtr;
        args[4] = &M;
        args[5] = &K;
        args[6] = &N;

        LaunchKernel2D(kernel, gridX, gridY, (uint)DefaultBlockSize, 1, args);
    }

    /// <inheritdoc/>
    public unsafe void CsrSegmentedMin(
        IGpuBuffer csrColIndices,
        IGpuBuffer csrRowPointers,
        IGpuBuffer input,
        IGpuBuffer output,
        int M, int K, int N)
    {
        if (!IsAvailable)
            throw new InvalidOperationException("CUDA backend is not available.");

        if (!_kernelCache.TryGetValue("csr_segmented_min", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: csr_segmented_min");

        using var _ = PushContext();

        // Launch configuration: rows x ceil(N/blockSize) grid
        uint gridX = (uint)M;
        uint gridY = (uint)((N + DefaultBlockSize - 1) / DefaultBlockSize);

        IntPtr colIndicesPtr = csrColIndices.Handle;
        IntPtr rowPointersPtr = csrRowPointers.Handle;
        IntPtr inputPtr = input.Handle;
        IntPtr outputPtr = output.Handle;

        void** args = stackalloc void*[7];
        args[0] = &colIndicesPtr;
        args[1] = &rowPointersPtr;
        args[2] = &inputPtr;
        args[3] = &outputPtr;
        args[4] = &M;
        args[5] = &K;
        args[6] = &N;

        LaunchKernel2D(kernel, gridX, gridY, (uint)DefaultBlockSize, 1, args);
    }

    /// <inheritdoc/>
    public unsafe void CsrSegmentedStdDev(
        IGpuBuffer csrColIndices,
        IGpuBuffer csrRowPointers,
        IGpuBuffer input,
        IGpuBuffer output,
        int M, int K, int N,
        float epsilon = 1e-8f)
    {
        if (!IsAvailable)
            throw new InvalidOperationException("CUDA backend is not available.");

        if (!_kernelCache.TryGetValue("csr_segmented_stddev", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: csr_segmented_stddev");

        using var _ = PushContext();

        // Launch configuration: rows x ceil(N/blockSize) grid
        uint gridX = (uint)M;
        uint gridY = (uint)((N + DefaultBlockSize - 1) / DefaultBlockSize);

        IntPtr colIndicesPtr = csrColIndices.Handle;
        IntPtr rowPointersPtr = csrRowPointers.Handle;
        IntPtr inputPtr = input.Handle;
        IntPtr outputPtr = output.Handle;

        void** args = stackalloc void*[8];
        args[0] = &colIndicesPtr;
        args[1] = &rowPointersPtr;
        args[2] = &inputPtr;
        args[3] = &outputPtr;
        args[4] = &M;
        args[5] = &K;
        args[6] = &N;
        args[7] = &epsilon;

        LaunchKernel2D(kernel, gridX, gridY, (uint)DefaultBlockSize, 1, args);
    }

    #endregion

    public float Sum(IGpuBuffer A, int size)
    {
        if (!IsAvailable)
            throw new InvalidOperationException("CUDA backend is not available.");

        if (size <= 0)
            return 0.0f;

        using var _ = PushContext();
        int blockSize = DefaultBlockSize;
        int gridSize = (size + blockSize - 1) / blockSize;

        using var partialBuffer = AllocateBuffer(gridSize);
        LaunchReductionKernel("reduce_sum", A, partialBuffer, size, blockSize);
        // Synchronize() removed: DownloadBuffer uses cuMemcpyDtoH which is synchronous
        var partials = DownloadBuffer(partialBuffer);
        float sum = 0.0f;
        for (int i = 0; i < partials.Length; i++)
            sum += partials[i];
        return sum;
    }

    /// <summary>
    /// Performs a full GPU-side sum reduction, iteratively reducing until only one element remains,
    /// then downloads just that single scalar value to avoid multiple D2H transfers for partial sums.
    /// </summary>
    private float SumGpuReduction(IGpuBuffer A, int size)
    {
        if (!IsAvailable)
            throw new InvalidOperationException("CUDA backend is not available.");

        if (size <= 0)
            return 0.0f;

        using var _ = PushContext();
        int blockSize = DefaultBlockSize;
        int currentSize = size;

        // We need at least one iteration
        IGpuBuffer currentBuffer = A;
        IGpuBuffer? tempBuffer1 = null;
        IGpuBuffer? tempBuffer2 = null;

        try
        {
            while (currentSize > 1)
            {
                int gridSize = (currentSize + blockSize - 1) / blockSize;

                // Allocate output buffer for this reduction pass
                var outputBuffer = (tempBuffer1 is null || tempBuffer1.Size < gridSize)
                    ? AllocateBuffer(gridSize)
                    : tempBuffer1;

                LaunchReductionKernel("reduce_sum", currentBuffer, outputBuffer, currentSize, blockSize);

                // Swap buffers for next iteration
                if (currentBuffer != A)
                {
                    // Return previous temp buffer to pool (swap)
                    tempBuffer2?.Dispose();
                    tempBuffer2 = tempBuffer1;
                }
                tempBuffer1 = outputBuffer;
                currentBuffer = outputBuffer;
                currentSize = gridSize;
            }

            // Synchronize() removed: DownloadBuffer uses cuMemcpyDtoH which is synchronous
            var result = new float[1];
            DownloadBuffer(currentBuffer, result);
            return result[0];
        }
        finally
        {
            tempBuffer1?.Dispose();
            tempBuffer2?.Dispose();
        }
    }

    public float Max(IGpuBuffer A, int size)
    {
        if (!IsAvailable)
            throw new InvalidOperationException("CUDA backend is not available.");

        if (size <= 0)
            return float.MinValue;

        using var _ = PushContext();
        int blockSize = DefaultBlockSize;
        int gridSize = (size + blockSize - 1) / blockSize;

        using var partialBuffer = AllocateBuffer(gridSize);
        LaunchReductionKernel("reduce_max", A, partialBuffer, size, blockSize);
        // Synchronize() removed: DownloadBuffer uses cuMemcpyDtoH which is synchronous
        var partials = DownloadBuffer(partialBuffer);
        float max = float.MinValue;
        for (int i = 0; i < partials.Length; i++)
            if (partials[i] > max) max = partials[i];
        return max;
    }

    public float Min(IGpuBuffer A, int size)
    {
        if (!IsAvailable)
            throw new InvalidOperationException("CUDA backend is not available.");

        if (size <= 0)
            return float.MaxValue;

        using var _ = PushContext();
        int blockSize = DefaultBlockSize;
        int gridSize = (size + blockSize - 1) / blockSize;

        using var partialBuffer = AllocateBuffer(gridSize);
        LaunchReductionKernel("reduce_min", A, partialBuffer, size, blockSize);
        var partials = DownloadBuffer(partialBuffer);
        float min = float.MaxValue;
        for (int i = 0; i < partials.Length; i++)
            if (partials[i] < min) min = partials[i];
        return min;
    }

    public void SumAxis(IGpuBuffer A, IGpuBuffer B, int outerSize, int reduceSize)
    {
        if (!IsAvailable)
            throw new InvalidOperationException("CUDA backend is not available.");

        if (outerSize <= 0 || reduceSize <= 0)
            throw new ArgumentException("outerSize and reduceSize must be positive.");

        long requiredInput = (long)outerSize * reduceSize;
        if (requiredInput > int.MaxValue || A.Size < requiredInput)
            throw new ArgumentException("Input buffer size is too small for the specified dimensions.");

        if (B.Size < outerSize)
            throw new ArgumentException("Output buffer size is too small for the specified dimensions.");

        if (!_kernelCache.TryGetValue("sum_axis", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: sum_axis");

        using var _ = PushContext();
        uint grid = (uint)((outerSize + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr inputPtr = A.Handle;
        IntPtr outputPtr = B.Handle;
        int outer = outerSize;
        int reduce = reduceSize;
        unsafe
        {
            void** args = stackalloc void*[4];
            args[0] = &inputPtr;
            args[1] = &outputPtr;
            args[2] = &outer;
            args[3] = &reduce;
            LaunchKernel(kernel, grid, DefaultBlockSize, args);
        }
    }

    /// <summary>Probe (AIDOTNET_CUDA_GRAPH_PROBE=1): confirms CUDA graph capture/instantiate/launch work on
    /// this backend's stream, and measures how much per-launch overhead graph replay removes vs issuing each
    /// op individually. Pure-GPU op sequence (DtoD copies) — the capturable subset; downloads can't be captured.</summary>
    private void RunCudaGraphProbe()
    {
        void Log(string s) => System.Diagnostics.Trace.WriteLine("[CUDA-GRAPH-PROBE] " + s);
        IntPtr src = IntPtr.Zero, dst = IntPtr.Zero, graph = IntPtr.Zero, exec = IntPtr.Zero;
        try
        {
            const ulong N = 4096;
            Log($"alloc src={CudaNativeBindings.cuMemAlloc(out src, N)} dst={CudaNativeBindings.cuMemAlloc(out dst, N)}");
            const int OPS = 16;
            var rb = CudaNativeBindings.cuStreamBeginCapture(_stream, CudaNativeBindings.CU_STREAM_CAPTURE_MODE_GLOBAL);
            for (int i = 0; i < OPS; i++) CudaNativeBindings.cuMemcpyDtoDAsync(dst, src, N, _stream);
            var re = CudaNativeBindings.cuStreamEndCapture(_stream, out graph);
            var ri = CudaNativeBindings.cuGraphInstantiate(out exec, graph, 0);
            Log($"beginCapture={rb} endCapture={re} instantiate={ri} graph={graph != IntPtr.Zero} exec={exec != IntPtr.Zero}");
            if ((int)ri != 0 || exec == IntPtr.Zero) { Log("INSTANTIATE FAILED -> capture not usable as-is on this stream"); return; }
            CudaNativeBindings.cuGraphLaunch(exec, _stream); CudaNativeBindings.cuStreamSynchronize(_stream); // warm
            const int REP = 2000;
            var swG = System.Diagnostics.Stopwatch.StartNew();
            for (int i = 0; i < REP; i++) CudaNativeBindings.cuGraphLaunch(exec, _stream);
            CudaNativeBindings.cuStreamSynchronize(_stream); swG.Stop();
            var swD = System.Diagnostics.Stopwatch.StartNew();
            for (int i = 0; i < REP * OPS; i++) CudaNativeBindings.cuMemcpyDtoDAsync(dst, src, N, _stream);
            CudaNativeBindings.cuStreamSynchronize(_stream); swD.Stop();
            double g = swG.Elapsed.TotalMilliseconds, d = swD.Elapsed.TotalMilliseconds;
            Log($"{REP} graph-replays ({REP * OPS} ops): {g:F1}ms  |  {REP * OPS} direct launches: {d:F1}ms");
            Log($"VERDICT: capture WORKS; replay removes per-launch overhead -> {d / System.Math.Max(0.01, g):F2}x fewer launch-cost for the same op count");
        }
        catch (Exception ex) { Log("EXCEPTION " + ex.GetType().Name + ": " + ex.Message); }
        finally
        {
            if (exec != IntPtr.Zero) CudaNativeBindings.cuGraphExecDestroy(exec);
            if (graph != IntPtr.Zero) CudaNativeBindings.cuGraphDestroy(graph);
            if (src != IntPtr.Zero) CudaNativeBindings.cuMemFree(src);
            if (dst != IntPtr.Zero) CudaNativeBindings.cuMemFree(dst);
        }
    }

    public void Synchronize()
    {
        if (!IsAvailable)
            return;

        using var _ = PushContext();
        CuBlasNative.CheckCudaResult(CudaNativeBindings.cuStreamSynchronize(_stream), "cuStreamSynchronize");
    }

    #region IAsyncGpuBackend Implementation

    /// <inheritdoc/>
    public IGpuStream CreateStream(GpuStreamType streamType)
    {
        return new CudaStream(this, streamType, 0);
    }

    /// <inheritdoc/>
    public IGpuStream CreateStream(GpuStreamType streamType, int priority)
    {
        return new CudaStream(this, streamType, priority);
    }

    /// <inheritdoc/>
    public IGpuEvent CreateEvent()
    {
        return new CudaEvent(this, null, enableTiming: false);
    }

    /// <inheritdoc/>
    public IGpuEvent CreateEvent(bool enableTiming)
    {
        return new CudaEvent(this, null, enableTiming);
    }

    /// <inheritdoc/>
    public void RecordEvent(IGpuEvent gpuEvent, IGpuStream stream)
    {
        if (gpuEvent is not CudaEvent cudaEvent)
            throw new ArgumentException("Event must be a CudaEvent", nameof(gpuEvent));

        cudaEvent.Record(stream);
    }

    /// <inheritdoc/>
    public void StreamWaitEvent(IGpuStream stream, IGpuEvent gpuEvent)
    {
        if (stream is not CudaStream cudaStream)
            throw new ArgumentException("Stream must be a CudaStream", nameof(stream));

        cudaStream.WaitEvent(gpuEvent);
    }

    /// <inheritdoc/>
    public GpuSyncPoint CreateSyncPoint(IGpuStream stream)
    {
        if (stream is not CudaStream cudaStream)
            throw new ArgumentException("Stream must be a CudaStream", nameof(stream));

        return new CudaSyncPoint(this, cudaStream);
    }

    /// <inheritdoc/>
    public GpuSyncPoint CreateSyncPoint()
    {
        if (_defaultStream == null)
            throw new InvalidOperationException("Backend not initialized");

        return new CudaSyncPoint(this, _defaultStream);
    }

    /// <inheritdoc/>
    public unsafe void UploadBufferAsync(float[] data, IGpuBuffer buffer, IGpuStream stream)
    {
        if (data == null) throw new ArgumentNullException(nameof(data));
        if (buffer == null) throw new ArgumentNullException(nameof(buffer));
        if (stream == null) throw new ArgumentNullException(nameof(stream));
        if (data.Length > buffer.Size)
            throw new ArgumentException($"Host data ({data.Length} elements) exceeds buffer size ({buffer.Size} elements).");

        using var _ = PushContext();
        int byteSize = data.Length * sizeof(float);

        // Use pinned host memory for true async DMA transfer.
        // Unlike a fixed block, pinned memory stays resident without synchronization,
        // enabling real compute-transfer overlap.
        var pinnedPtr = _pinnedPool.Rent(byteSize);
        try
        {
            fixed (float* src = data)
            {
                Buffer.MemoryCopy(src, (void*)pinnedPtr, byteSize, byteSize);
            }

            var result = CudaNativeBindings.cuMemcpyHtoDAsync(
                buffer.Handle,
                pinnedPtr,
                (ulong)byteSize,
                stream.Handle);
            CuBlasNative.CheckCudaResult(result, "cuMemcpyHtoDAsync");

            // Record an event so we know when to return the pinned buffer.
            // For now, we synchronize the stream to safely return the buffer.
            // A future optimization could use event callbacks to defer the return.
            var syncResult = CudaNativeBindings.cuStreamSynchronize(stream.Handle);
            CuBlasNative.CheckCudaResult(syncResult, "cuStreamSynchronize(pinned upload)");
        }
        finally
        {
            _pinnedPool.Return(pinnedPtr, byteSize);
        }
    }

    /// <summary>Synchronously overwrite an EXISTING buffer's contents in place (stable pointer, no
    /// realloc). Used to refresh the persistent CUDA-graph input each step OUTSIDE capture so the
    /// captured forward reads fresh data without a re-upload INSIDE capture (which is CUDA 906
    /// STREAM_CAPTURE_UNSUPPORTED). Keeping the pointer stable preserves the captured graph's validity.</summary>
    public unsafe void UploadBufferInPlace(float[] data, IGpuBuffer buffer)
    {
        if (data == null) throw new ArgumentNullException(nameof(data));
        if (buffer == null) throw new ArgumentNullException(nameof(buffer));
        // Fail fast with a managed exception BEFORE touching the driver — matches the rest of the
        // backend. Without this, a failed backend init or a freed graph-input buffer (zero handle)
        // would fall through to a raw cuMemcpyHtoD and fault the CUDA context (error 700) instead.
        if (!IsAvailable)
            throw new InvalidOperationException("CUDA backend is not available.");
        if (buffer.Handle == IntPtr.Zero)
            throw new ObjectDisposedException(nameof(buffer),
                "GPU buffer was released before its in-place upload.");
        if (data.Length > buffer.Size)
            throw new ArgumentException($"Host data ({data.Length}) exceeds buffer ({buffer.Size}).");
        using var _ = PushContext();
        ulong byteSize = (ulong)data.Length * sizeof(float);
        fixed (float* src = data)
            CuBlasNative.CheckCudaResult(
                CuBlasNative.cuMemcpyHtoD(buffer.Handle, (IntPtr)src, byteSize),
                "cuMemcpyHtoD(graph input refresh in-place)");
    }

    /// <summary>In-place int upload (no allocation) — used to refresh a STABLE embedding-index buffer each
    /// step so the per-step path has no <c>cuMemAlloc</c> (which is synchronous and serializes the step).</summary>
    public unsafe void UploadIntBufferInPlace(int[] data, IGpuBuffer buffer)
    {
        if (data == null) throw new ArgumentNullException(nameof(data));
        if (buffer == null) throw new ArgumentNullException(nameof(buffer));
        if (!IsAvailable)
            throw new InvalidOperationException("CUDA backend is not available.");
        if (buffer.Handle == IntPtr.Zero)
            throw new ObjectDisposedException(nameof(buffer),
                "GPU buffer was released before its in-place upload.");
        if (data.Length > buffer.Size)
            throw new ArgumentException($"Host data ({data.Length}) exceeds buffer ({buffer.Size}).");
        using var _ = PushContext();
        ulong byteSize = (ulong)data.Length * sizeof(int);
        fixed (int* src = data)
            CuBlasNative.CheckCudaResult(
                CuBlasNative.cuMemcpyHtoD(buffer.Handle, (IntPtr)src, byteSize),
                "cuMemcpyHtoD(embedding index refresh in-place)");
    }


    /// <inheritdoc/>
    public unsafe void UploadBufferAsync(ReadOnlySpan<float> data, IGpuBuffer buffer, IGpuStream stream)
    {
        if (buffer == null) throw new ArgumentNullException(nameof(buffer));
        if (stream == null) throw new ArgumentNullException(nameof(stream));
        if (data.Length > buffer.Size)
            throw new ArgumentException($"Host data ({data.Length} elements) exceeds buffer size ({buffer.Size} elements).");

        using var _ = PushContext();
        int byteSize = data.Length * sizeof(float);

        var pinnedPtr = _pinnedPool.Rent(byteSize);
        try
        {
            fixed (float* src = data)
            {
                Buffer.MemoryCopy(src, (void*)pinnedPtr, byteSize, byteSize);
            }

            var result = CudaNativeBindings.cuMemcpyHtoDAsync(
                buffer.Handle,
                pinnedPtr,
                (ulong)byteSize,
                stream.Handle);
            CuBlasNative.CheckCudaResult(result, "cuMemcpyHtoDAsync");

            var syncResult = CudaNativeBindings.cuStreamSynchronize(stream.Handle);
            CuBlasNative.CheckCudaResult(syncResult, "cuStreamSynchronize(pinned upload)");
        }
        finally
        {
            _pinnedPool.Return(pinnedPtr, byteSize);
        }
    }

    /// <inheritdoc/>
    public unsafe void DownloadBufferAsync(IGpuBuffer buffer, float[] destination, IGpuStream stream)
    {
        if (buffer == null) throw new ArgumentNullException(nameof(buffer));
        if (destination == null) throw new ArgumentNullException(nameof(destination));
        if (stream == null) throw new ArgumentNullException(nameof(stream));
        if (destination.Length > buffer.Size)
            throw new ArgumentException($"Destination ({destination.Length} elements) exceeds buffer size ({buffer.Size} elements).");

        using var _ = PushContext();
        int byteSize = destination.Length * sizeof(float);

        var pinnedPtr = _pinnedPool.Rent(byteSize);
        try
        {
            var result = CudaNativeBindings.cuMemcpyDtoHAsync(
                pinnedPtr,
                buffer.Handle,
                (ulong)byteSize,
                stream.Handle);
            CuBlasNative.CheckCudaResult(result, "cuMemcpyDtoHAsync");

            // Must synchronize before copying out of pinned buffer
            var syncResult = CudaNativeBindings.cuStreamSynchronize(stream.Handle);
            CuBlasNative.CheckCudaResult(syncResult, "cuStreamSynchronize(pinned download)");

            fixed (float* dst = destination)
            {
                Buffer.MemoryCopy((void*)pinnedPtr, dst, byteSize, byteSize);
            }
        }
        finally
        {
            _pinnedPool.Return(pinnedPtr, byteSize);
        }
    }

    /// <inheritdoc/>
    public IGpuBuffer AllocateBufferAsync(float[] data, IGpuStream stream)
    {
        var buffer = AllocateBuffer(data.Length);
        UploadBufferAsync(data, buffer, stream);
        return buffer;
    }

    /// <inheritdoc/>
    public void CopyBufferAsync(IGpuBuffer source, IGpuBuffer destination, int size, IGpuStream stream)
    {
        if (source == null) throw new ArgumentNullException(nameof(source));
        if (destination == null) throw new ArgumentNullException(nameof(destination));
        if (stream == null) throw new ArgumentNullException(nameof(stream));

        using var _ = PushContext();
        ulong byteSize = (ulong)size * sizeof(float);
        var result = CudaNativeBindings.cuMemcpyDtoDAsync(
            destination.Handle,
            source.Handle,
            byteSize,
            stream.Handle);
        CuBlasNative.CheckCudaResult(result, "cuMemcpyDtoDAsync");
    }

    /// <inheritdoc/>
    public void GemmAsync(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int M, int N, int K,
        float alpha, float beta, IGpuStream stream)
    {
        if (!IsAvailable)
            throw new InvalidOperationException("CUDA backend is not available.");

        ValidateGemmArgs(A, B, C, M, N, K);

        using var _ = PushContext();
        float alphaVal = alpha;
        float betaVal = beta;

        // Set cuBLAS to use the specified stream
        CuBlasNative.CheckCublasStatus(CuBlasNative.cublasSetStream(_cublasHandle, stream.Handle), "cublasSetStream");

        try
        {
            // Row-major C = A * B. Use cuBLAS column-major trick: C^T = B^T * A^T.
            CuBlasNative.CheckCublasStatus(
                CuBlasNative.cublasSgemm(
                    _cublasHandle,
                    CublasOperation.None,
                    CublasOperation.None,
                    N, M, K,
                    ref alphaVal,
                    B.Handle, N,
                    A.Handle, K,
                    ref betaVal,
                    C.Handle, N),
                "cublasSgemm");
        }
        finally
        {
            // Restore the default stream
            CuBlasNative.cublasSetStream(_cublasHandle, _stream);
        }
    }

    /// <inheritdoc/>
    public void FusedGemmBiasActivationAsync(IGpuBuffer A, IGpuBuffer B, IGpuBuffer bias, IGpuBuffer output,
        int M, int N, int K, FusedActivationType activation, IGpuStream stream)
    {
        if (!IsAvailable)
            throw new InvalidOperationException("CUDA backend is not available.");

        // Map activation to fused kernel name
        string kernelName = activation switch
        {
            FusedActivationType.ReLU => "fused_gemm_bias_relu",
            FusedActivationType.Sigmoid => "fused_gemm_bias_sigmoid",
            FusedActivationType.Tanh => "fused_gemm_bias_tanh",
            FusedActivationType.None => "fused_gemm_bias",
            _ => throw new NotSupportedException($"Activation type {activation} not supported for fused GEMM")
        };

        ExecuteFusedGemmOnStream(kernelName, A, B, bias, output, M, N, K, stream);
    }

    /// <inheritdoc/>
    public void SynchronizeStream(IGpuStream stream)
    {
        if (stream == null) throw new ArgumentNullException(nameof(stream));

        using var _ = PushContext();
        var result = CudaNativeBindings.cuStreamSynchronize(stream.Handle);
        CuBlasNative.CheckCudaResult(result, "cuStreamSynchronize");
    }

    /// <inheritdoc/>
    public bool QueryStreamComplete(IGpuStream stream)
    {
        if (stream is not CudaStream cudaStream)
            throw new ArgumentException("Stream must be a CudaStream", nameof(stream));

        return cudaStream.Query();
    }

    /// <inheritdoc/>
    public bool QueryEventComplete(IGpuEvent gpuEvent)
    {
        if (gpuEvent is not CudaEvent cudaEvent)
            throw new ArgumentException("Event must be a CudaEvent", nameof(gpuEvent));

        return cudaEvent.Query();
    }

    /// <inheritdoc/>
    public float GetEventElapsedTime(IGpuEvent start, IGpuEvent end)
    {
        if (start is not CudaEvent cudaStart)
            throw new ArgumentException("Start event must be a CudaEvent", nameof(start));
        if (end is not CudaEvent cudaEnd)
            throw new ArgumentException("End event must be a CudaEvent", nameof(end));

        return cudaEnd.GetElapsedTime(cudaStart);
    }

    /// <summary>
    /// Launches a kernel on a specific stream.
    /// </summary>
    private unsafe void LaunchKernelOnStream(IntPtr kernel, uint gridX, uint blockX, void** args, IntPtr stream, uint sharedMem = 0)
    {
        CuBlasNative.CheckCudaResult(
            CudaNativeBindings.cuLaunchKernel(
                kernel,
                gridX, 1, 1,
                blockX, 1, 1,
                sharedMem,
                stream,
                (IntPtr)args,
                IntPtr.Zero),
            "cuLaunchKernel");
    }

    /// <summary>
    /// Launches a 2D kernel on a specific stream.
    /// </summary>
    private unsafe void LaunchKernel2DOnStream(IntPtr kernel, uint gridX, uint gridY, uint gridZ, uint blockX, uint blockY, void** args, IntPtr stream, uint sharedMem = 0)
    {
        CuBlasNative.CheckCudaResult(
            CudaNativeBindings.cuLaunchKernel(
                kernel,
                gridX, gridY, gridZ,
                blockX, blockY, 1,
                sharedMem,
                stream,
                (IntPtr)args,
                IntPtr.Zero),
            "cuLaunchKernel2D");
    }

    /// <summary>
    /// Executes a fused GEMM kernel on a specific stream.
    /// </summary>
    private unsafe void ExecuteFusedGemmOnStream(string kernelName, IGpuBuffer A, IGpuBuffer B, IGpuBuffer bias, IGpuBuffer output, int M, int N, int K, IGpuStream stream)
    {
        if (!_kernelCache.TryGetValue(kernelName, out var kernel))
            throw new InvalidOperationException($"CUDA fused kernel not found: {kernelName}");

        using var _ = PushContext();

        // 128x128 CTA tile with 16x16 thread block (8x8 register blocking per thread)
        const int BM = 128;
        const int BN = 128;
        const int BLOCK_DIM = 16;
        uint gridX = (uint)((N + BN - 1) / BN);
        uint gridY = (uint)((M + BM - 1) / BM);

        IntPtr aPtr = A.Handle;
        IntPtr bPtr = B.Handle;
        IntPtr biasPtr = bias.Handle;
        IntPtr outPtr = output.Handle;
        int m = M, n = N, k = K;

        void** args = stackalloc void*[7];
        args[0] = &aPtr;
        args[1] = &bPtr;
        args[2] = &biasPtr;
        args[3] = &outPtr;
        args[4] = &m;
        args[5] = &n;
        args[6] = &k;

        LaunchKernel2DOnStream(kernel, gridX, gridY, 1, BLOCK_DIM, BLOCK_DIM, args, stream.Handle);
    }

    #endregion

    private unsafe void LaunchUnaryKernel(string kernelName, IGpuBuffer input, IGpuBuffer output, int size)
    {
        // Try vec4 variant for 4x throughput on bandwidth-limited ops
        if (size % 4 == 0 && _kernelCache.TryGetValue(kernelName + "_vec4", out var vec4Kernel))
        {
            using var _v = PushContext();
            int size4 = size / 4;
            uint gridV = (uint)((size4 + DefaultBlockSize - 1) / DefaultBlockSize);
            IntPtr inPtrV = input.Handle;
            IntPtr outPtrV = output.Handle;
            void** argsV = stackalloc void*[3];
            argsV[0] = &inPtrV;
            argsV[1] = &outPtrV;
            argsV[2] = &size4;
            LaunchKernel(vec4Kernel, gridV, DefaultBlockSize, argsV);
            return;
        }

        if (!_kernelCache.TryGetValue(kernelName, out var kernel))
            throw new InvalidOperationException($"CUDA kernel not found: {kernelName}");

        using var _ = PushContext();
        uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr inputPtr = input.Handle;
        IntPtr outputPtr = output.Handle;
        int n = size;
        void** args = stackalloc void*[3];
        args[0] = &inputPtr;
        args[1] = &outputPtr;
        args[2] = &n;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    private unsafe void LaunchElementwiseKernel(string kernelName, IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int size)
    {
        // Try vec4 variant for 4x throughput on bandwidth-limited ops
        if (size % 4 == 0 && _kernelCache.TryGetValue(kernelName + "_vec4", out var vec4Kernel))
        {
            using var _v = PushContext();
            int size4 = size / 4;
            uint gridV = (uint)((size4 + DefaultBlockSize - 1) / DefaultBlockSize);
            IntPtr aPtrV = A.Handle;
            IntPtr bPtrV = B.Handle;
            IntPtr cPtrV = C.Handle;
            void** argsV = stackalloc void*[4];
            argsV[0] = &aPtrV;
            argsV[1] = &bPtrV;
            argsV[2] = &cPtrV;
            argsV[3] = &size4;
            LaunchKernel(vec4Kernel, gridV, DefaultBlockSize, argsV);
            return;
        }

        if (!_kernelCache.TryGetValue(kernelName, out var kernel))
            throw new InvalidOperationException($"CUDA kernel not found: {kernelName}");

        using var _ = PushContext();
        uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr aPtr = A.Handle;
        IntPtr bPtr = B.Handle;
        IntPtr cPtr = C.Handle;
        int n = size;
        void** args = stackalloc void*[4];
        args[0] = &aPtr;
        args[1] = &bPtr;
        args[2] = &cPtr;
        args[3] = &n;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    private unsafe void LaunchScaleKernel(IGpuBuffer A, IGpuBuffer B, float scalar, int size)
    {
        // Try vec4 variant
        if (size % 4 == 0 && _kernelCache.TryGetValue("scale_vector_vec4", out var vec4Kernel))
        {
            using var _v = PushContext();
            int size4 = size / 4;
            uint gridV = (uint)((size4 + DefaultBlockSize - 1) / DefaultBlockSize);
            IntPtr aPtrV = A.Handle;
            IntPtr bPtrV = B.Handle;
            float scalarV = scalar;
            void** argsV = stackalloc void*[4];
            argsV[0] = &aPtrV;
            argsV[1] = &bPtrV;
            argsV[2] = &scalarV;
            argsV[3] = &size4;
            LaunchKernel(vec4Kernel, gridV, DefaultBlockSize, argsV);
            return;
        }

        if (!_kernelCache.TryGetValue("scale_vector", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: scale_vector");

        using var _ = PushContext();
        uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr aPtr = A.Handle;
        IntPtr bPtr = B.Handle;
        float scalarVal = scalar;
        int n = size;
        void** args = stackalloc void*[4];
        args[0] = &aPtr;
        args[1] = &bPtr;
        args[2] = &scalarVal;
        args[3] = &n;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    private unsafe void LaunchUnaryWithScalarKernel(string kernelName, IGpuBuffer A, IGpuBuffer B, float scalar, int size)
    {
        if (!_kernelCache.TryGetValue(kernelName, out var kernel))
            throw new InvalidOperationException($"CUDA kernel not found: {kernelName}");

        using var _ = PushContext();
        uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr aPtr = A.Handle;
        IntPtr bPtr = B.Handle;
        float scalarVal = scalar;
        int n = size;
        void** args = stackalloc void*[4];
        args[0] = &aPtr;
        args[1] = &bPtr;
        args[2] = &scalarVal;
        args[3] = &n;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    private unsafe void LaunchSoftmaxKernel(IGpuBuffer input, IGpuBuffer output, int batchSize, int features)
    {
        if (!_kernelCache.TryGetValue("softmax", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: softmax");

        using var _ = PushContext();

        // 1 block per batch element, 256 threads cooperate on parallel reduction
        // Shared memory: ceil(256/32) = 8 warps * sizeof(float) = 32 bytes
        uint grid = (uint)batchSize;
        const uint blockSize = 256;
        uint sharedMem = (blockSize / 32) * sizeof(float);

        IntPtr inputPtr = input.Handle;
        IntPtr outputPtr = output.Handle;
        int batches = batchSize;
        int feats = features;
        void** args = stackalloc void*[4];
        args[0] = &inputPtr;
        args[1] = &outputPtr;
        args[2] = &batches;
        args[3] = &feats;

        CuBlasNative.CheckCudaResult(
            CudaNativeBindings.cuLaunchKernel(kernel, grid, 1, 1, blockSize, 1, 1,
                sharedMem, _stream, (IntPtr)args, IntPtr.Zero),
            "cuLaunchKernel(softmax)");
    }

    private unsafe void LaunchReductionKernel(string kernelName, IGpuBuffer input, IGpuBuffer output, int size, int blockSize)
    {
        if (!_kernelCache.TryGetValue(kernelName, out var kernel))
            throw new InvalidOperationException($"CUDA kernel not found: {kernelName}");

        uint grid = (uint)((size + blockSize - 1) / blockSize);
        IntPtr inputPtr = input.Handle;
        IntPtr outputPtr = output.Handle;
        int n = size;
        void** args = stackalloc void*[3];
        args[0] = &inputPtr;
        args[1] = &outputPtr;
        args[2] = &n;

        uint sharedBytes = (uint)(blockSize * sizeof(float));
        LaunchKernelWithSharedMem(kernel, grid, (uint)blockSize, sharedBytes, args);
    }

    private unsafe void ApplyBiasInPlace(IGpuBuffer data, IGpuBuffer bias, int rows, int cols)
    {
        if (!_kernelCache.TryGetValue("bias_add", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: bias_add");

        using var _ = PushContext();
        uint gridX = (uint)((cols + DefaultBlockSize - 1) / DefaultBlockSize);
        uint gridY = (uint)rows;
        IntPtr dataPtr = data.Handle;
        IntPtr biasPtr = bias.Handle;
        int r = rows;
        int c = cols;
        void** args = stackalloc void*[4];
        args[0] = &dataPtr;
        args[1] = &biasPtr;
        args[2] = &r;
        args[3] = &c;
        LaunchKernel2D(kernel, gridX, gridY, DefaultBlockSize, 1, args);
    }

    // CUDA's "current context" is PER-THREAD: cuCtxCreate makes the context current only on the
    // CREATING thread. This backend otherwise never re-establishes it, so every native call assumes
    // it runs on that thread. Under xUnit's parallel / thread-pooled test execution (and any caller
    // that drives a GPU op from a different thread than the one that built the engine), the executing
    // thread may have NO context — or a DIFFERENT engine's context — current, so cuLaunchKernel /
    // cuMemcpy / cuMemAlloc operate on the wrong context and fault with an uncatchable 0xC0000005.
    // Re-assert this engine's context on the calling thread before native work. cuCtxSetCurrent is a
    // cheap per-thread driver call (no cross-thread effect); skipped once the context is torn down.
    internal void EnsureContextCurrent()
    {
        IntPtr ctx = _cudaContext;
        if (ctx != IntPtr.Zero && !IsRuntimeTearingDown && LiveContexts.ContainsKey(ctx))
            CudaNativeBindings.cuCtxSetCurrent(ctx);
        DrainPendingFinalizerFrees();
    }

    // Device frees DEFERRED from buffer FINALIZERS. Calling the CUDA driver (cuCtxPushCurrent /
    // cuMemFree) from the GC finalizer thread races with op threads' driver entry and faults with an
    // uncatchable 0xC0000005 under the heavy GC the full test suite produces (DirectGpu-only, with low
    // GC, never crashes). Instead the finalizer just enqueues the pointer, and the free runs here on a
    // real op thread (with this engine's context current) at the next GPU op. A pending entry whose
    // context was meanwhile destroyed is dropped — cuCtxDestroy already reclaimed that memory.
    internal static readonly ConcurrentQueue<(IntPtr Ptr, IntPtr Ctx, IntPtr Stream)> PendingFinalizerFrees = new();

    internal static void DrainPendingFinalizerFrees()
    {
        while (PendingFinalizerFrees.TryDequeue(out var f))
        {
            if (f.Ptr == IntPtr.Zero) continue;
            lock (ContextLifecycleLock)
            {
                if (f.Ctx == IntPtr.Zero || IsRuntimeTearingDown || ProcessExiting
                    || !LiveContexts.ContainsKey(f.Ctx))
                    continue; // context gone — its device memory was reclaimed by cuCtxDestroy
                try
                {
                    CuBlasNative.cuCtxPushCurrent(f.Ctx);
                    if (f.Stream != IntPtr.Zero) CuBlasNative.cuMemFreeAsync(f.Ptr, f.Stream);
                    else CuBlasNative.cuMemFree(f.Ptr);
                    CuBlasNative.cuCtxPopCurrent(out _);
                }
                catch { }
            }
        }
    }

    private unsafe void LaunchKernel(IntPtr kernel, uint gridX, uint blockX, void** args)
    {
        LaunchKernelWithSharedMem(kernel, gridX, blockX, 0, args);
    }

    private unsafe void LaunchKernelWithSharedMem(IntPtr kernel, uint gridX, uint blockX, uint sharedMemBytes, void** args)
    {
        CuBlasNative.CheckCudaResult(
            CudaNativeBindings.cuLaunchKernel(
                kernel,
                gridX, 1, 1,
                blockX, 1, 1,
                sharedMemBytes,
                _stream,
                (IntPtr)args,
                IntPtr.Zero),
            "cuLaunchKernel");
    }

    /// <summary>
    /// Launch a cooperative kernel that supports grid-level synchronization via cooperative_groups.
    /// Cooperative kernels can use grid.sync() for cross-block synchronization.
    /// Throws if device doesn't support cooperative launch or grid exceeds limits.
    /// </summary>
    private unsafe void LaunchCooperativeKernel(IntPtr kernel, uint gridX, uint blockX, uint sharedMemBytes, void** args)
    {
        // Check cooperative launch support
        if (!_supportsCooperativeLaunch)
        {
            throw new InvalidOperationException(
                "Cooperative kernel launch is not supported on this device. " +
                "Cooperative launch requires CUDA compute capability 6.0 or higher. " +
                "Use cell-level operations instead of sequence-level operations.");
        }

        // Check grid size limits for cooperative launch
        var occResult = CudaNativeBindings.cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
            out int maxBlocksPerSm, kernel, (int)blockX, (nint)sharedMemBytes, 0);
        if (occResult != CudaResult.Success)
        {
            throw new InvalidOperationException(
                $"Failed to query occupancy for cooperative kernel: {occResult}. " +
                "Use cell-level operations instead of sequence-level operations.");
        }

        int maxCooperativeBlocks = maxBlocksPerSm * _multiProcessorCount;
        if (gridX > maxCooperativeBlocks)
        {
            throw new InvalidOperationException(
                $"Grid size ({gridX}) exceeds cooperative launch limit ({maxCooperativeBlocks}) " +
                $"for this device ({maxBlocksPerSm} blocks/SM × {_multiProcessorCount} SMs). " +
                "Reduce batch size or use cell-level operations.");
        }

        CuBlasNative.CheckCudaResult(
            CudaNativeBindings.cuLaunchCooperativeKernel(
                kernel,
                gridX, 1, 1,
                blockX, 1, 1,
                sharedMemBytes,
                _stream,
                (IntPtr)args),
            "cuLaunchCooperativeKernel");
    }

    private unsafe void LaunchKernel2D(IntPtr kernel, uint gridX, uint gridY, uint gridZ, uint blockX, uint blockY, void** args)
    {
        CuBlasNative.CheckCudaResult(
            CudaNativeBindings.cuLaunchKernel(
                kernel,
                gridX, gridY, gridZ,
                blockX, blockY, 1,
                0,
                _stream,
                (IntPtr)args,
                IntPtr.Zero),
            "cuLaunchKernel2D");
    }

    private unsafe void LaunchKernel2DWithSharedMem(IntPtr kernel, uint gridX, uint gridY, uint blockX, uint blockY, uint sharedMemBytes, void** args)
    {
        CuBlasNative.CheckCudaResult(
            CudaNativeBindings.cuLaunchKernel(
                kernel,
                gridX, gridY, 1,
                blockX, blockY, 1,
                sharedMemBytes,
                _stream,
                (IntPtr)args,
                IntPtr.Zero),
            "cuLaunchKernel2DSharedMem");
    }

    private unsafe void LaunchKernel3D(IntPtr kernel, uint gridX, uint gridY, uint gridZ, uint blockX, uint blockY, uint blockZ, void** args, uint sharedMemBytes = 0)
    {
        CuBlasNative.CheckCudaResult(
            CudaNativeBindings.cuLaunchKernel(
                kernel,
                gridX, gridY, gridZ,
                blockX, blockY, blockZ,
                sharedMemBytes,
                _stream,
                (IntPtr)args,
                IntPtr.Zero),
            "cuLaunchKernel3D");
    }

    private static void ValidateGemmArgs(IGpuBuffer A, IGpuBuffer B, IGpuBuffer? C, int M, int N, int K)
    {
        if (M <= 0 || N <= 0 || K <= 0)
            throw new ArgumentException("Matrix dimensions M, N, K must be positive.");

        long requiredA = (long)M * K;
        long requiredB = (long)K * N;
        long requiredC = (long)M * N;

        if (requiredA > int.MaxValue || requiredB > int.MaxValue || requiredC > int.MaxValue)
            throw new ArgumentException("Matrix dimensions are too large.");

        if (A.Size < requiredA)
            throw new ArgumentException("Buffer A is too small for the specified dimensions.");
        if (B.Size < requiredB)
            throw new ArgumentException("Buffer B is too small for the specified dimensions.");
        if (C != null && C.Size < requiredC)
            throw new ArgumentException("Buffer C is too small for the specified dimensions.");
    }

    private static void ValidateBiasBuffer(IGpuBuffer bias, int n)
    {
        if (bias == null)
            throw new ArgumentNullException(nameof(bias));
        if (n <= 0)
            throw new ArgumentOutOfRangeException(nameof(n), "N must be positive.");
        if (bias.Size < n)
            throw new ArgumentException("Bias buffer size must be at least N.", nameof(bias));
    }

    /// <summary>
    /// Checks if a value is a power of two. Required for CUDA shared-memory reductions
    /// that use the standard tree-reduction pattern.
    /// </summary>
    private static bool IsPowerOfTwo(int value) => value > 0 && (value & (value - 1)) == 0;

    /// <summary>
    /// Validates that the state size is a power of two, as required by quantum kernel
    /// shared-memory reductions.
    /// </summary>
    private static void ValidateQuantumStateSize(int stateSize, string paramName)
    {
        if (stateSize <= 0)
            throw new ArgumentOutOfRangeException(paramName, stateSize, "State size must be positive.");
        if (!IsPowerOfTwo(stateSize))
            throw new ArgumentException(
                $"Quantum kernels require state size to be a power of two for shared-memory reductions. Got: {stateSize}",
                paramName);
    }

    private unsafe void ExecuteFusedGemm(string kernelName, IGpuBuffer A, IGpuBuffer B, IGpuBuffer bias, IGpuBuffer output, int M, int N, int K)
    {
        // Try WMMA tensor core kernel first (sm_70+, 2-4x faster for large matrices)
        string wmmaName = kernelName + "_wmma";
        if (_hasWmmaSupport && _kernelCache.TryGetValue(wmmaName, out var wmmaKernel))
        {
            ExecuteWmmaGemm(wmmaKernel, A, B, bias, output, M, N, K);
            return;
        }

        if (!_kernelCache.TryGetValue(kernelName, out var kernel))
            throw new InvalidOperationException($"CUDA fused kernel not found: {kernelName}");

        using var _ = PushContext();

        // 128x128 CTA tile with 16x16 thread block (8x8 register blocking per thread)
        const int BM = 128;
        const int BN = 128;
        const int BLOCK_DIM = 16;
        uint gridX = (uint)((N + BN - 1) / BN);
        uint gridY = (uint)((M + BM - 1) / BM);

        IntPtr aPtr = A.Handle;
        IntPtr bPtr = B.Handle;
        IntPtr biasPtr = bias.Handle;
        IntPtr outPtr = output.Handle;
        int m = M, n = N, k = K;

        void** args = stackalloc void*[7];
        args[0] = &aPtr;
        args[1] = &bPtr;
        args[2] = &biasPtr;
        args[3] = &outPtr;
        args[4] = &m;
        args[5] = &n;
        args[6] = &k;

        LaunchKernel2D(kernel, gridX, gridY, 1, BLOCK_DIM, BLOCK_DIM, args);
    }

    private unsafe void ExecuteWmmaGemm(IntPtr kernel, IGpuBuffer A, IGpuBuffer B, IGpuBuffer bias, IGpuBuffer output, int M, int N, int K)
    {
        using var _ = PushContext();

        // 32x32 CTA tile with 128 threads (4 warps, each computing 16x16 via tensor cores)
        const int WMMA_TILE = 32;
        const int WMMA_THREADS = 128;
        uint gridX = (uint)((N + WMMA_TILE - 1) / WMMA_TILE);
        uint gridY = (uint)((M + WMMA_TILE - 1) / WMMA_TILE);

        IntPtr aPtr = A.Handle;
        IntPtr bPtr = B.Handle;
        IntPtr biasPtr = bias.Handle;
        IntPtr outPtr = output.Handle;
        int m = M, n = N, k = K;

        void** args = stackalloc void*[7];
        args[0] = &aPtr;
        args[1] = &bPtr;
        args[2] = &biasPtr;
        args[3] = &outPtr;
        args[4] = &m;
        args[5] = &n;
        args[6] = &k;

        CuBlasNative.CheckCudaResult(
            CudaNativeBindings.cuLaunchKernel(kernel, gridX, gridY, 1, WMMA_THREADS, 1, 1,
                0, _stream, (IntPtr)args, IntPtr.Zero),
            "cuLaunchKernel(WMMA)");
    }

    private static void ValidateBatchedGemmArgs(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int M, int N, int K, int batchCount)
    {
        // Element-count overload — assumes 4-byte float layout via
        // IGpuBuffer.Size's semantics. Mixed-precision (fp16/bf16) and
        // fp64 callers MUST use the byte-aware overload below instead;
        // their per-element strides differ from the float assumption
        // baked into ValidateGemmArgs / A.Size comparisons.
        if (batchCount <= 0)
            throw new ArgumentException("batchCount must be positive.");

        ValidateGemmArgs(A, B, C, M, N, K);

        long requiredA = (long)batchCount * M * K;
        long requiredB = (long)batchCount * K * N;
        long requiredC = (long)batchCount * M * N;

        if (requiredA > int.MaxValue || requiredB > int.MaxValue || requiredC > int.MaxValue)
            throw new ArgumentException("Batched GEMM dimensions are too large.");

        if (A.Size < requiredA)
            throw new ArgumentException("Buffer A is too small for the specified batch dimensions.");
        if (B.Size < requiredB)
            throw new ArgumentException("Buffer B is too small for the specified batch dimensions.");
        if (C.Size < requiredC)
            throw new ArgumentException("Buffer C is too small for the specified batch dimensions.");
    }

    /// <summary>
    /// Byte-aware batched-GEMM validator for mixed-precision and fp64
    /// fanout paths where the per-buffer element stride differs from the
    /// 4-byte float assumption that <see cref="IGpuBuffer.Size"/>'s
    /// element-count semantics bakes in. Compares against
    /// <see cref="IGpuBuffer.SizeInBytes"/> so an undersized half/bfloat16
    /// or double buffer can't pass validation and then walk past its
    /// allocation inside cublasGemmEx / cublasDgemm.
    /// </summary>
    private static void ValidateBatchedGemmArgsBytes(
        IGpuBuffer A, IGpuBuffer B, IGpuBuffer C,
        int M, int N, int K, int batchCount,
        int elemBytesA, int elemBytesB, int elemBytesC)
    {
        if (batchCount <= 0)
            throw new ArgumentException("batchCount must be positive.");
        if (M <= 0 || N <= 0 || K <= 0)
            throw new ArgumentException("GEMM dimensions M, N, K must be positive.");
        if (A is null) throw new ArgumentNullException(nameof(A));
        if (B is null) throw new ArgumentNullException(nameof(B));
        if (C is null) throw new ArgumentNullException(nameof(C));

        // Overflow-safe checked arithmetic — the element-count overload
        // had an `> int.MaxValue` upper-bound guard but this byte-aware
        // overload doesn't, and an unchecked long * long multiplication
        // can wrap negative (e.g. batchCount=1B × M=1B × K=4 = ~4×10^18,
        // past long.MaxValue / 2). A wrapped requiredXBytes would let an
        // impossibly small buffer pass validation and walk into wrapped
        // pointer math inside cuBLAS.
        long requiredAElems, requiredBElems, requiredCElems;
        long requiredABytes, requiredBBytes, requiredCBytes;
        try
        {
            checked
            {
                requiredAElems = (long)batchCount * M * K;
                requiredBElems = (long)batchCount * K * N;
                requiredCElems = (long)batchCount * M * N;
                requiredABytes = requiredAElems * elemBytesA;
                requiredBBytes = requiredBElems * elemBytesB;
                requiredCBytes = requiredCElems * elemBytesC;
            }
        }
        catch (OverflowException)
        {
            throw new ArgumentException(
                $"Batched GEMM dimensions too large for fanout dispatch: M={M}, N={N}, K={K}, batchCount={batchCount}.");
        }

        if (A.SizeInBytes < requiredABytes)
            throw new ArgumentException(
                $"Buffer A too small: capacity={A.SizeInBytes} bytes, required={requiredABytes} bytes ({requiredAElems} elements × {elemBytesA}).",
                nameof(A));
        if (B.SizeInBytes < requiredBBytes)
            throw new ArgumentException(
                $"Buffer B too small: capacity={B.SizeInBytes} bytes, required={requiredBBytes} bytes ({requiredBElems} elements × {elemBytesB}).",
                nameof(B));
        if (C.SizeInBytes < requiredCBytes)
            throw new ArgumentException(
                $"Buffer C too small: capacity={C.SizeInBytes} bytes, required={requiredCBytes} bytes ({requiredCElems} elements × {elemBytesC}).",
                nameof(C));
    }

    #region Convolution Operations

    public unsafe void Conv2D(IGpuBuffer input, IGpuBuffer kernel, IGpuBuffer output,
        int batch, int inChannels, int inHeight, int inWidth,
        int outChannels, int outHeight, int outWidth,
        int kernelH, int kernelW,
        int strideH, int strideW, int padH, int padW,
        int dilationH, int dilationW)
    {
        // cuDNN fast path (issue #201). Gated on the UseCudnnForConv
        // policy + CuDnnConvolution.IsAvailable at init. Dispatches the
        // GPU-pointer variant directly — caller-owned input / kernel /
        // output buffers, no host round-trip. Scope is opened here so
        // PerformanceProfiler stats distinguish "Conv2D.cuDNN" vs
        // "Conv2D.generic" for consumers verifying which path ran.
        if (CudaDispatchPolicy.UseCudnnForConv)
        {
            using var _profileCudnn = CudaDispatchPolicy.Scope("Conv2D", useVendor: true);
            using var _ctxCudnn = PushContext();
            EnsureCudnnConv();
            // Stream affinity: cuDNN kernels launch onto this backend's
            // default stream so they interleave correctly with the
            // surrounding Cuda ops. Check the status — if stream binding
            // silently fails, subsequent cuDNN work runs with wrong
            // stream ordering and produces nondeterministic results.
            CuDnnContext.CheckStatus(
                CuDnnNative.cudnnSetStream(_cudnnContext!.Handle, _stream),
                "cudnnSetStream");
            _cudnnConv!.Conv2DForwardGpu(
                inputDevPtr: input.Handle,
                filterDevPtr: kernel.Handle,
                outputDevPtr: output.Handle,
                n: batch, c: inChannels, h: inHeight, w: inWidth,
                k: outChannels, filterH: kernelH, filterW: kernelW,
                outputHeight: outHeight, outputWidth: outWidth,
                padH: padH, padW: padW,
                strideH: strideH, strideW: strideW,
                dilationH: dilationH, dilationW: dilationW);
            return;
        }

        // Generic-kernel path — Winograd / tiled / im2col — when the policy
        // opts out (debug / forced-generic) or cuDNN isn't available.
        using var _profile = CudaDispatchPolicy.Scope(
            "Conv2D",
            useVendor: false);
        using var _ = PushContext();
        IntPtr inputPtr = input.Handle;
        IntPtr kernelPtr = kernel.Handle;
        IntPtr outputPtr = output.Handle;

        // Route 3x3 stride-1 dilation-1 convolutions through Winograd F(2x2,3x3)
        if (kernelH == 3 && kernelW == 3 && strideH == 1 && strideW == 1 &&
            dilationH == 1 && dilationW == 1 &&
            _kernelCache.TryGetValue("conv2d_winograd_f2x2_3x3", out var winogradKernel))
        {
            int tilesH = (outHeight + 1) / 2;
            int tilesW = (outWidth + 1) / 2;
            int totalTiles = batch * outChannels * tilesH * tilesW;
            uint gridX = (uint)((totalTiles + DefaultBlockSize - 1) / DefaultBlockSize);

            void** wArgs = stackalloc void*[12];
            wArgs[0] = &inputPtr;
            wArgs[1] = &kernelPtr;
            wArgs[2] = &outputPtr;
            wArgs[3] = &batch;
            wArgs[4] = &inChannels;
            wArgs[5] = &inHeight;
            wArgs[6] = &inWidth;
            wArgs[7] = &outChannels;
            wArgs[8] = &outHeight;
            wArgs[9] = &outWidth;
            wArgs[10] = &padH;
            wArgs[11] = &padW;
            LaunchKernel(winogradKernel, gridX, DefaultBlockSize, wArgs);
            return;
        }

        // Use shared-memory tiled kernel when available
        if (_kernelCache.TryGetValue("conv2d_tiled", out var tiledKernel))
        {
            const int TILE_OUT = 16;
            uint gx = (uint)((outWidth + TILE_OUT - 1) / TILE_OUT);
            uint gy = (uint)((outHeight + TILE_OUT - 1) / TILE_OUT);
            uint gz = (uint)(batch * outChannels);

            // Shared memory: one input channel tile at a time
            int effKH = (kernelH - 1) * dilationH + 1;
            int effKW = (kernelW - 1) * dilationW + 1;
            int tileInH = TILE_OUT * strideH + effKH - strideH;
            int tileInW = TILE_OUT * strideW + effKW - strideW;
            uint sharedMem = (uint)(tileInH * tileInW * sizeof(float));

            void** tArgs = stackalloc void*[17];
            tArgs[0] = &inputPtr;
            tArgs[1] = &kernelPtr;
            tArgs[2] = &outputPtr;
            tArgs[3] = &batch;
            tArgs[4] = &inChannels;
            tArgs[5] = &inHeight;
            tArgs[6] = &inWidth;
            tArgs[7] = &outChannels;
            tArgs[8] = &outHeight;
            tArgs[9] = &outWidth;
            tArgs[10] = &kernelH;
            tArgs[11] = &kernelW;
            tArgs[12] = &strideH;
            tArgs[13] = &strideW;
            tArgs[14] = &padH;
            tArgs[15] = &padW;
            tArgs[16] = &dilationH;
            // Note: dilationW is tArgs[17] but the kernel reads it from the parameter list
            // The tiled kernel uses the same 17 params as conv2d_direct (minus dilationW which
            // is passed as the 18th)
            void** tArgs2 = stackalloc void*[18];
            for (int i = 0; i < 17; i++) tArgs2[i] = tArgs[i];
            tArgs2[17] = &dilationW;

            LaunchKernel3D(tiledKernel, gx, gy, gz, TILE_OUT, TILE_OUT, 1, tArgs2, sharedMem);
            return;
        }

        // Fallback: direct convolution with correct 3D grid
        if (!_kernelCache.TryGetValue("conv2d_direct", out var cudaKernel))
            throw new InvalidOperationException("CUDA kernel not found: conv2d_direct");

        {
            const int BLOCK = 16;
            uint dgx = (uint)((outWidth + BLOCK - 1) / BLOCK);
            uint dgy = (uint)((outHeight + BLOCK - 1) / BLOCK);
            uint dgz = (uint)(batch * outChannels);

            void** dArgs = stackalloc void*[18];
            dArgs[0] = &inputPtr;
            dArgs[1] = &kernelPtr;
            dArgs[2] = &outputPtr;
            dArgs[3] = &batch;
            dArgs[4] = &inChannels;
            dArgs[5] = &inHeight;
            dArgs[6] = &inWidth;
            dArgs[7] = &outChannels;
            dArgs[8] = &outHeight;
            dArgs[9] = &outWidth;
            dArgs[10] = &kernelH;
            dArgs[11] = &kernelW;
            dArgs[12] = &strideH;
            dArgs[13] = &strideW;
            dArgs[14] = &padH;
            dArgs[15] = &padW;
            dArgs[16] = &dilationH;
            dArgs[17] = &dilationW;
            LaunchKernel3D(cudaKernel, dgx, dgy, dgz, (uint)BLOCK, (uint)BLOCK, 1, dArgs, 0);
        }
    }

    /// <summary>Lazily spin up the cuDNN context + convolution helper on
    /// the first call that routes Conv2D through the vendor path. Both
    /// live for the lifetime of this backend and are released in
    /// <see cref="Dispose"/>. Thread-safe via double-checked locking so
    /// concurrent Conv2D calls on the first dispatch can't each
    /// double-initialize the cuDNN helper.</summary>
    private void EnsureCudnnConv()
    {
        if (_cudnnConv is not null) return;
        lock (_cudnnInitLock)
        {
            // Re-check inside the lock — another thread may have won the
            // race to construct and we'd otherwise leak its instance.
            if (_cudnnConv is not null) return;
            _cudnnContext ??= new CuDnnContext();
            _cudnnConv = new CuDnnConvolution(_cudnnContext);
        }
    }

    public unsafe void Conv2DBackwardInput(IGpuBuffer gradOutput, IGpuBuffer kernel, IGpuBuffer gradInput,
        int batch, int inChannels, int inHeight, int inWidth,
        int outChannels, int outHeight, int outWidth,
        int kernelH, int kernelW,
        int strideH, int strideW, int padH, int padW,
        int dilationH, int dilationW)
    {
        if (!_kernelCache.TryGetValue("conv2d_backward_input", out var cudaKernel))
            throw new InvalidOperationException("CUDA kernel not found: conv2d_backward_input");

        using var _ = PushContext();
        const int BLOCK = 16;
        uint gx = (uint)((inWidth + BLOCK - 1) / BLOCK);
        uint gy = (uint)((inHeight + BLOCK - 1) / BLOCK);
        uint gz = (uint)(batch * inChannels);

        IntPtr gradOutputPtr = gradOutput.Handle;
        IntPtr kernelPtr = kernel.Handle;
        IntPtr gradInputPtr = gradInput.Handle;
        void** args = stackalloc void*[18];
        args[0] = &gradOutputPtr;
        args[1] = &kernelPtr;
        args[2] = &gradInputPtr;
        args[3] = &batch;
        args[4] = &inChannels;
        args[5] = &inHeight;
        args[6] = &inWidth;
        args[7] = &outChannels;
        args[8] = &outHeight;
        args[9] = &outWidth;
        args[10] = &kernelH;
        args[11] = &kernelW;
        args[12] = &strideH;
        args[13] = &strideW;
        args[14] = &padH;
        args[15] = &padW;
        args[16] = &dilationH;
        args[17] = &dilationW;
        LaunchKernel3D(cudaKernel, gx, gy, gz, (uint)BLOCK, (uint)BLOCK, 1, args, 0);
    }

    public unsafe void Conv2DBackwardKernel(IGpuBuffer input, IGpuBuffer gradOutput, IGpuBuffer gradKernel,
        int batch, int inChannels, int inHeight, int inWidth,
        int outChannels, int outHeight, int outWidth,
        int kernelH, int kernelW,
        int strideH, int strideW, int padH, int padW,
        int dilationH, int dilationW)
    {
        if (!_kernelCache.TryGetValue("conv2d_backward_kernel", out var cudaKernel))
            throw new InvalidOperationException("CUDA kernel not found: conv2d_backward_kernel");

        using var _ = PushContext();
        const int BLOCK = 16;
        uint gx = (uint)((kernelW + BLOCK - 1) / BLOCK);
        uint gy = (uint)((kernelH + BLOCK - 1) / BLOCK);
        uint gz = (uint)(outChannels * inChannels);

        IntPtr inputPtr = input.Handle;
        IntPtr gradOutputPtr = gradOutput.Handle;
        IntPtr gradKernelPtr = gradKernel.Handle;
        void** args = stackalloc void*[18];
        args[0] = &inputPtr;
        args[1] = &gradOutputPtr;
        args[2] = &gradKernelPtr;
        args[3] = &batch;
        args[4] = &inChannels;
        args[5] = &inHeight;
        args[6] = &inWidth;
        args[7] = &outChannels;
        args[8] = &outHeight;
        args[9] = &outWidth;
        args[10] = &kernelH;
        args[11] = &kernelW;
        args[12] = &strideH;
        args[13] = &strideW;
        args[14] = &padH;
        args[15] = &padW;
        args[16] = &dilationH;
        args[17] = &dilationW;
        LaunchKernel3D(cudaKernel, gx, gy, gz, (uint)BLOCK, (uint)BLOCK, 1, args, 0);
    }

    public unsafe void Conv1D(IGpuBuffer input, IGpuBuffer kernel, IGpuBuffer output,
        int batch, int inChannels, int inLength,
        int outChannels, int outLength, int kernelLength,
        int stride, int padding, int dilation)
    {
        // Conv1D via Conv2D with height=1: [B,C,L] -> [B,C,1,L]
        Conv2D(input, kernel, output,
            batch, inChannels, 1, inLength,
            outChannels, 1, outLength,
            1, kernelLength,
            1, stride, 0, padding, 1, dilation);
    }

    public unsafe void Conv1DBackwardInput(IGpuBuffer gradOutput, IGpuBuffer kernel, IGpuBuffer gradInput,
        int batch, int inChannels, int inLength,
        int outChannels, int outLength, int kernelLength,
        int stride, int padding, int dilation)
    {
        Conv2DBackwardInput(gradOutput, kernel, gradInput,
            batch, inChannels, 1, inLength,
            outChannels, 1, outLength,
            1, kernelLength, 1, stride, 0, padding, 1, dilation);
    }

    public unsafe void Conv1DBackwardKernel(IGpuBuffer input, IGpuBuffer gradOutput, IGpuBuffer gradKernel,
        int batch, int inChannels, int inLength,
        int outChannels, int outLength, int kernelLength,
        int stride, int padding, int dilation)
    {
        Conv2DBackwardKernel(input, gradOutput, gradKernel,
            batch, inChannels, 1, inLength,
            outChannels, 1, outLength,
            1, kernelLength, 1, stride, 0, padding, 1, dilation);
    }

    public unsafe void Unfold(IGpuBuffer input, IGpuBuffer output,
        int batch, int channels, int height, int width,
        int kernelH, int kernelW, int strideH, int strideW, int padH, int padW)
    {
        if (strideH <= 0 || strideW <= 0) throw new ArgumentException("Stride must be positive.");
        using var _ = PushContext();
        if (!_kernelCache.TryGetValue("im2col", out var im2colKernel))
            throw new InvalidOperationException("CUDA kernel not found: im2col");

        IntPtr inputPtr = input.Handle;
        IntPtr outputPtr = output.Handle;
        int outH = (height + 2 * padH - kernelH) / strideH + 1;
        int outW = (width + 2 * padW - kernelW) / strideW + 1;
        if (outH <= 0 || outW <= 0) throw new ArgumentException($"Invalid Unfold output dimensions {outH}x{outW}.");
        int totalPatches = batch * outH * outW;
        int dilationH = 1, dilationW = 1;

        void** args = stackalloc void*[15];
        args[0] = &inputPtr; args[1] = &outputPtr;
        args[2] = &batch; args[3] = &channels; args[4] = &height; args[5] = &width;
        args[6] = &kernelH; args[7] = &kernelW; args[8] = &strideH; args[9] = &strideW;
        args[10] = &padH; args[11] = &padW; args[12] = &dilationH; args[13] = &dilationW;
        args[14] = &outH;

        uint gridX = (uint)((totalPatches + DefaultBlockSize - 1) / DefaultBlockSize);
        LaunchKernel(im2colKernel, gridX, DefaultBlockSize, args);
    }

    public unsafe void Fold(IGpuBuffer input, IGpuBuffer output,
        int batch, int channels, int outputH, int outputW,
        int kernelH, int kernelW, int strideH, int strideW, int padH, int padW)
    {
        if (strideH <= 0 || strideW <= 0) throw new ArgumentException("Stride must be positive.");
        using var _ = PushContext();
        if (!_kernelCache.TryGetValue("col2im", out var col2imKernel))
            throw new InvalidOperationException("CUDA kernel not found: col2im");

        IntPtr inputPtr = input.Handle;
        IntPtr outputPtr = output.Handle;
        int outH = (outputH + 2 * padH - kernelH) / strideH + 1;
        int outW = (outputW + 2 * padW - kernelW) / strideW + 1;
        int totalSize = batch * channels * outputH * outputW;
        int dilationH = 1, dilationW = 1;

        // Zero output first
        ZeroBuffer(output, totalSize);

        void** args = stackalloc void*[15];
        args[0] = &inputPtr; args[1] = &outputPtr;
        args[2] = &batch; args[3] = &channels; args[4] = &outputH; args[5] = &outputW;
        args[6] = &kernelH; args[7] = &kernelW; args[8] = &strideH; args[9] = &strideW;
        args[10] = &padH; args[11] = &padW; args[12] = &dilationH; args[13] = &dilationW;
        args[14] = &outH;

        uint gridX = (uint)((totalSize + DefaultBlockSize - 1) / DefaultBlockSize);
        LaunchKernel(col2imKernel, gridX, DefaultBlockSize, args);
    }

    public unsafe void Conv3D(IGpuBuffer input, IGpuBuffer kernel, IGpuBuffer output,
        int batch, int inChannels, int inDepth, int inHeight, int inWidth,
        int outChannels, int outDepth, int outHeight, int outWidth,
        int kernelD, int kernelH, int kernelW,
        int strideD, int strideH, int strideW,
        int padD, int padH, int padW,
        int dilationD, int dilationH, int dilationW)
    {
        if (!_kernelCache.TryGetValue("conv3d_direct", out var cudaKernel))
            throw new InvalidOperationException("CUDA kernel not found: conv3d_direct");

        using var _ = PushContext();
        int totalOutput = batch * outChannels * outDepth * outHeight * outWidth;
        uint gridX = (uint)((totalOutput + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr inputPtr = input.Handle;
        IntPtr kernelPtr = kernel.Handle;
        IntPtr outputPtr = output.Handle;
        void** args = stackalloc void*[23];
        args[0] = &inputPtr;
        args[1] = &kernelPtr;
        args[2] = &outputPtr;
        args[3] = &batch;
        args[4] = &inChannels;
        args[5] = &inDepth;
        args[6] = &inHeight;
        args[7] = &inWidth;
        args[8] = &outChannels;
        args[9] = &outDepth;
        args[10] = &outHeight;
        args[11] = &outWidth;
        args[12] = &kernelD;
        args[13] = &kernelH;
        args[14] = &kernelW;
        args[15] = &strideD;
        args[16] = &strideH;
        args[17] = &strideW;
        args[18] = &padD;
        args[19] = &padH;
        args[20] = &padW;
        args[21] = &totalOutput;
        LaunchKernel(cudaKernel, gridX, DefaultBlockSize, args);
    }

    public unsafe void DepthwiseConv2D(IGpuBuffer input, IGpuBuffer kernel, IGpuBuffer output,
        int batch, int channels, int inHeight, int inWidth,
        int outHeight, int outWidth,
        int kernelH, int kernelW,
        int strideH, int strideW, int padH, int padW)
    {
        if (!_kernelCache.TryGetValue("depthwise_conv2d", out var cudaKernel))
            throw new InvalidOperationException("CUDA kernel not found: depthwise_conv2d");

        using var _ = PushContext();
        IntPtr inputPtr = input.Handle;
        IntPtr kernelPtr = kernel.Handle;
        IntPtr outputPtr = output.Handle;
        void** args = stackalloc void*[15];
        args[0] = &inputPtr;
        args[1] = &kernelPtr;
        args[2] = &outputPtr;
        args[3] = &batch;
        args[4] = &channels;
        args[5] = &inHeight;
        args[6] = &inWidth;
        args[7] = &outHeight;
        args[8] = &outWidth;
        args[9] = &kernelH;
        args[10] = &kernelW;
        args[11] = &strideH;
        args[12] = &strideW;
        args[13] = &padH;
        args[14] = &padW;
        // Kernel uses 3D grid: blockIdx.x→outWidth, blockIdx.y→outHeight, blockIdx.z→batch*channels
        const uint blockDimXY = 16;
        uint gx = (uint)((outWidth + blockDimXY - 1) / blockDimXY);
        uint gy = (uint)((outHeight + blockDimXY - 1) / blockDimXY);
        uint gz = (uint)(batch * channels);
        LaunchKernel3D(cudaKernel, gx, gy, gz, blockDimXY, blockDimXY, 1, args);
    }

    public unsafe void ConvTranspose2D(IGpuBuffer input, IGpuBuffer kernel, IGpuBuffer output,
        int batch, int inChannels, int inHeight, int inWidth,
        int outChannels, int outHeight, int outWidth,
        int kernelH, int kernelW,
        int strideH, int strideW, int padH, int padW,
        int outputPadH, int outputPadW)
    {
        if (!_kernelCache.TryGetValue("conv_transpose2d", out var cudaKernel))
            throw new InvalidOperationException("CUDA kernel not found: conv_transpose2d");

        using var _ = PushContext();
        IntPtr inputPtr = input.Handle;
        IntPtr kernelPtr = kernel.Handle;
        IntPtr outputPtr = output.Handle;
        void** args = stackalloc void*[18];
        args[0] = &inputPtr;
        args[1] = &kernelPtr;
        args[2] = &outputPtr;
        args[3] = &batch;
        args[4] = &inChannels;
        args[5] = &inHeight;
        args[6] = &inWidth;
        args[7] = &outChannels;
        args[8] = &outHeight;
        args[9] = &outWidth;
        args[10] = &kernelH;
        args[11] = &kernelW;
        args[12] = &strideH;
        args[13] = &strideW;
        args[14] = &padH;
        args[15] = &padW;
        args[16] = &outputPadH;
        args[17] = &outputPadW;
        // Kernel uses 3D grid: blockIdx.x→outWidth, blockIdx.y→outHeight, blockIdx.z→batch*outChannels
        const uint blockDimXY = 16;
        uint gx = (uint)((outWidth + blockDimXY - 1) / blockDimXY);
        uint gy = (uint)((outHeight + blockDimXY - 1) / blockDimXY);
        uint gz = (uint)(batch * outChannels);
        LaunchKernel3D(cudaKernel, gx, gy, gz, blockDimXY, blockDimXY, 1, args);
    }

    public unsafe void ConvTranspose2DBackwardInput(IGpuBuffer gradOutput, IGpuBuffer kernel, IGpuBuffer gradInput,
        int batch, int inChannels, int inHeight, int inWidth,
        int outChannels, int outHeight, int outWidth,
        int kernelH, int kernelW,
        int strideH, int strideW, int padH, int padW,
        int outputPadH, int outputPadW)
    {
        // Buffer size validation
        long requiredGradOutput = (long)batch * outChannels * outHeight * outWidth;
        long requiredKernel = (long)inChannels * outChannels * kernelH * kernelW;
        long requiredGradInput = (long)batch * inChannels * inHeight * inWidth;

        if (gradOutput.Size < requiredGradOutput)
            throw new ArgumentException($"gradOutput buffer too small: expected at least {requiredGradOutput} elements, got {gradOutput.Size}.", nameof(gradOutput));
        if (kernel.Size < requiredKernel)
            throw new ArgumentException($"kernel buffer too small: expected at least {requiredKernel} elements, got {kernel.Size}.", nameof(kernel));
        if (gradInput.Size < requiredGradInput)
            throw new ArgumentException($"gradInput buffer too small: expected at least {requiredGradInput} elements, got {gradInput.Size}.", nameof(gradInput));

        if (!_kernelCache.TryGetValue("conv_transpose2d_backward_input", out var cudaKernel))
            throw new InvalidOperationException("CUDA kernel not found: conv_transpose2d_backward_input");

        using var _ = PushContext();
        int totalInput = batch * inChannels * inHeight * inWidth;
        uint gridX = (uint)((totalInput + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr gradOutputPtr = gradOutput.Handle;
        IntPtr kernelPtr = kernel.Handle;
        IntPtr gradInputPtr = gradInput.Handle;
        void** args = stackalloc void*[19];
        args[0] = &gradOutputPtr;
        args[1] = &kernelPtr;
        args[2] = &gradInputPtr;
        args[3] = &batch;
        args[4] = &inChannels;
        args[5] = &inHeight;
        args[6] = &inWidth;
        args[7] = &outChannels;
        args[8] = &outHeight;
        args[9] = &outWidth;
        args[10] = &kernelH;
        args[11] = &kernelW;
        args[12] = &strideH;
        args[13] = &strideW;
        args[14] = &padH;
        args[15] = &padW;
        args[16] = &outputPadH;
        args[17] = &outputPadW;
        args[18] = &totalInput;
        LaunchKernel(cudaKernel, gridX, DefaultBlockSize, args);
    }

    public unsafe void ConvTranspose2DBackwardKernel(IGpuBuffer input, IGpuBuffer gradOutput, IGpuBuffer gradKernel,
        int batch, int inChannels, int inHeight, int inWidth,
        int outChannels, int outHeight, int outWidth,
        int kernelH, int kernelW,
        int strideH, int strideW, int padH, int padW,
        int outputPadH, int outputPadW)
    {
        // Buffer size validation
        long requiredInput = (long)batch * inChannels * inHeight * inWidth;
        long requiredGradOutput = (long)batch * outChannels * outHeight * outWidth;
        long requiredGradKernel = (long)inChannels * outChannels * kernelH * kernelW;

        if (input.Size < requiredInput)
            throw new ArgumentException($"input buffer too small: expected at least {requiredInput} elements, got {input.Size}.", nameof(input));
        if (gradOutput.Size < requiredGradOutput)
            throw new ArgumentException($"gradOutput buffer too small: expected at least {requiredGradOutput} elements, got {gradOutput.Size}.", nameof(gradOutput));
        if (gradKernel.Size < requiredGradKernel)
            throw new ArgumentException($"gradKernel buffer too small: expected at least {requiredGradKernel} elements, got {gradKernel.Size}.", nameof(gradKernel));

        if (!_kernelCache.TryGetValue("conv_transpose2d_backward_kernel", out var cudaKernel))
            throw new InvalidOperationException("CUDA kernel not found: conv_transpose2d_backward_kernel");

        using var _ = PushContext();
        int totalKernel = inChannels * outChannels * kernelH * kernelW;
        uint gridX = (uint)((totalKernel + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr inputPtr = input.Handle;
        IntPtr gradOutputPtr = gradOutput.Handle;
        IntPtr gradKernelPtr = gradKernel.Handle;
        void** args = stackalloc void*[19];
        args[0] = &inputPtr;
        args[1] = &gradOutputPtr;
        args[2] = &gradKernelPtr;
        args[3] = &batch;
        args[4] = &inChannels;
        args[5] = &inHeight;
        args[6] = &inWidth;
        args[7] = &outChannels;
        args[8] = &outHeight;
        args[9] = &outWidth;
        args[10] = &kernelH;
        args[11] = &kernelW;
        args[12] = &strideH;
        args[13] = &strideW;
        args[14] = &padH;
        args[15] = &padW;
        args[16] = &outputPadH;
        args[17] = &outputPadW;
        args[18] = &totalKernel;
        LaunchKernel(cudaKernel, gridX, DefaultBlockSize, args);
    }

    #region Locally Connected Convolution Operations

    public unsafe void LocallyConnectedConv2D(IGpuBuffer input, IGpuBuffer weights, IGpuBuffer? bias, IGpuBuffer output,
        int batch, int inChannels, int inHeight, int inWidth,
        int outChannels, int outHeight, int outWidth,
        int kernelH, int kernelW,
        int strideH, int strideW)
    {
        if (!_kernelCache.TryGetValue("locally_connected_conv2d", out var cudaKernel))
            throw new InvalidOperationException("CUDA kernel not found: locally_connected_conv2d");

        using var _ = PushContext();
        const int blockSize = 16;
        uint gridX = (uint)((outWidth + blockSize - 1) / blockSize);
        uint gridY = (uint)((outHeight + blockSize - 1) / blockSize);
        uint gridZ = (uint)(batch * outChannels);

        IntPtr inputPtr = input.Handle;
        IntPtr weightsPtr = weights.Handle;
        IntPtr biasPtr = bias?.Handle ?? IntPtr.Zero;
        IntPtr outputPtr = output.Handle;
        int hasBias = bias is not null ? 1 : 0;

        void** args = stackalloc void*[16];
        args[0] = &inputPtr;
        args[1] = &weightsPtr;
        args[2] = &biasPtr;
        args[3] = &outputPtr;
        args[4] = &batch;
        args[5] = &inChannels;
        args[6] = &inHeight;
        args[7] = &inWidth;
        args[8] = &outChannels;
        args[9] = &outHeight;
        args[10] = &outWidth;
        args[11] = &kernelH;
        args[12] = &kernelW;
        args[13] = &strideH;
        args[14] = &strideW;
        args[15] = &hasBias;
        LaunchKernel2D(cudaKernel, gridX, gridY, gridZ, (uint)blockSize, (uint)blockSize, args);
    }

    public unsafe void LocallyConnectedConv2DBackwardInput(IGpuBuffer gradOutput, IGpuBuffer weights, IGpuBuffer gradInput,
        int batch, int inChannels, int inHeight, int inWidth,
        int outChannels, int outHeight, int outWidth,
        int kernelH, int kernelW,
        int strideH, int strideW)
    {
        if (!_kernelCache.TryGetValue("locally_connected_conv2d_backward_input", out var cudaKernel))
            throw new InvalidOperationException("CUDA kernel not found: locally_connected_conv2d_backward_input");

        using var _ = PushContext();
        int totalInputSize = batch * inChannels * inHeight * inWidth;
        uint gridX = (uint)((totalInputSize + DefaultBlockSize - 1) / DefaultBlockSize);

        IntPtr gradOutputPtr = gradOutput.Handle;
        IntPtr weightsPtr = weights.Handle;
        IntPtr gradInputPtr = gradInput.Handle;

        void** args = stackalloc void*[14];
        args[0] = &gradOutputPtr;
        args[1] = &weightsPtr;
        args[2] = &gradInputPtr;
        args[3] = &batch;
        args[4] = &inChannels;
        args[5] = &inHeight;
        args[6] = &inWidth;
        args[7] = &outChannels;
        args[8] = &outHeight;
        args[9] = &outWidth;
        args[10] = &kernelH;
        args[11] = &kernelW;
        args[12] = &strideH;
        args[13] = &strideW;
        LaunchKernel(cudaKernel, gridX, DefaultBlockSize, args);
    }

    public unsafe void LocallyConnectedConv2DBackwardWeights(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradWeights,
        int batch, int inChannels, int inHeight, int inWidth,
        int outChannels, int outHeight, int outWidth,
        int kernelH, int kernelW,
        int strideH, int strideW)
    {
        if (!_kernelCache.TryGetValue("locally_connected_conv2d_backward_weights", out var cudaKernel))
            throw new InvalidOperationException("CUDA kernel not found: locally_connected_conv2d_backward_weights");

        using var _ = PushContext();
        int totalWeights = outHeight * outWidth * outChannels * inChannels * kernelH * kernelW;
        uint gridX = (uint)((totalWeights + DefaultBlockSize - 1) / DefaultBlockSize);

        IntPtr gradOutputPtr = gradOutput.Handle;
        IntPtr inputPtr = input.Handle;
        IntPtr gradWeightsPtr = gradWeights.Handle;

        void** args = stackalloc void*[14];
        args[0] = &gradOutputPtr;
        args[1] = &inputPtr;
        args[2] = &gradWeightsPtr;
        args[3] = &batch;
        args[4] = &inChannels;
        args[5] = &inHeight;
        args[6] = &inWidth;
        args[7] = &outChannels;
        args[8] = &outHeight;
        args[9] = &outWidth;
        args[10] = &kernelH;
        args[11] = &kernelW;
        args[12] = &strideH;
        args[13] = &strideW;
        LaunchKernel(cudaKernel, gridX, DefaultBlockSize, args);
    }

    public unsafe void LocallyConnectedConv2DBackwardBias(IGpuBuffer gradOutput, IGpuBuffer gradBias,
        int batch, int outChannels, int outHeight, int outWidth)
    {
        if (!_kernelCache.TryGetValue("locally_connected_conv2d_backward_bias", out var cudaKernel))
            throw new InvalidOperationException("CUDA kernel not found: locally_connected_conv2d_backward_bias");

        using var _ = PushContext();
        uint gridX = (uint)((outChannels + DefaultBlockSize - 1) / DefaultBlockSize);

        IntPtr gradOutputPtr = gradOutput.Handle;
        IntPtr gradBiasPtr = gradBias.Handle;

        void** args = stackalloc void*[6];
        args[0] = &gradOutputPtr;
        args[1] = &gradBiasPtr;
        args[2] = &batch;
        args[3] = &outChannels;
        args[4] = &outHeight;
        args[5] = &outWidth;
        LaunchKernel(cudaKernel, gridX, DefaultBlockSize, args);
    }

    #endregion

    #region Deformable Convolution Operations

    public unsafe void DeformableConv2D(IGpuBuffer input, IGpuBuffer weights, IGpuBuffer offsets, IGpuBuffer? mask, IGpuBuffer output,
        int batch, int inChannels, int inHeight, int inWidth,
        int outChannels, int outHeight, int outWidth,
        int kernelH, int kernelW,
        int strideH, int strideW, int padH, int padW,
        int dilationH, int dilationW,
        int groups, int deformGroups)
    {
        if (!_kernelCache.TryGetValue("deformable_conv2d", out var cudaKernel))
            throw new InvalidOperationException("CUDA kernel not found: deformable_conv2d");

        using var _ = PushContext();
        const int blockSize = 16;
        uint gridX = (uint)((outWidth + blockSize - 1) / blockSize);
        uint gridY = (uint)((outHeight + blockSize - 1) / blockSize);
        uint gridZ = (uint)(batch * outChannels);

        IntPtr inputPtr = input.Handle;
        IntPtr weightsPtr = weights.Handle;
        IntPtr offsetsPtr = offsets.Handle;
        IntPtr maskPtr = mask?.Handle ?? IntPtr.Zero;
        IntPtr outputPtr = output.Handle;
        int hasMask = mask is not null ? 1 : 0;

        void** args = stackalloc void*[23];
        args[0] = &inputPtr;
        args[1] = &weightsPtr;
        args[2] = &offsetsPtr;
        args[3] = &maskPtr;
        args[4] = &outputPtr;
        args[5] = &batch;
        args[6] = &inChannels;
        args[7] = &inHeight;
        args[8] = &inWidth;
        args[9] = &outChannels;
        args[10] = &outHeight;
        args[11] = &outWidth;
        args[12] = &kernelH;
        args[13] = &kernelW;
        args[14] = &strideH;
        args[15] = &strideW;
        args[16] = &padH;
        args[17] = &padW;
        args[18] = &dilationH;
        args[19] = &dilationW;
        args[20] = &groups;
        args[21] = &deformGroups;
        args[22] = &hasMask;
        LaunchKernel2D(cudaKernel, gridX, gridY, gridZ, (uint)blockSize, (uint)blockSize, args);
    }

    public unsafe void DeformableConv2DBackwardInput(IGpuBuffer gradOutput, IGpuBuffer weights, IGpuBuffer offsets, IGpuBuffer? mask, IGpuBuffer gradInput,
        int batch, int inChannels, int inHeight, int inWidth,
        int outChannels, int outHeight, int outWidth,
        int kernelH, int kernelW,
        int strideH, int strideW, int padH, int padW,
        int dilationH, int dilationW,
        int groups, int deformGroups)
    {
        if (!_kernelCache.TryGetValue("deformable_conv2d_backward_input", out var cudaKernel))
            throw new InvalidOperationException("CUDA kernel not found: deformable_conv2d_backward_input");

        using var _ = PushContext();
        int totalInputSize = batch * inChannels * inHeight * inWidth;
        uint gridX = (uint)((totalInputSize + DefaultBlockSize - 1) / DefaultBlockSize);

        IntPtr gradOutputPtr = gradOutput.Handle;
        IntPtr weightsPtr = weights.Handle;
        IntPtr offsetsPtr = offsets.Handle;
        IntPtr maskPtr = mask?.Handle ?? IntPtr.Zero;
        IntPtr gradInputPtr = gradInput.Handle;
        int hasMask = mask is not null ? 1 : 0;

        void** args = stackalloc void*[23];
        args[0] = &gradOutputPtr;
        args[1] = &weightsPtr;
        args[2] = &offsetsPtr;
        args[3] = &maskPtr;
        args[4] = &gradInputPtr;
        args[5] = &batch;
        args[6] = &inChannels;
        args[7] = &inHeight;
        args[8] = &inWidth;
        args[9] = &outChannels;
        args[10] = &outHeight;
        args[11] = &outWidth;
        args[12] = &kernelH;
        args[13] = &kernelW;
        args[14] = &strideH;
        args[15] = &strideW;
        args[16] = &padH;
        args[17] = &padW;
        args[18] = &dilationH;
        args[19] = &dilationW;
        args[20] = &groups;
        args[21] = &deformGroups;
        args[22] = &hasMask;
        LaunchKernel(cudaKernel, gridX, DefaultBlockSize, args);
    }

    public unsafe void DeformableConv2DBackwardWeights(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer offsets, IGpuBuffer? mask, IGpuBuffer gradWeights,
        int batch, int inChannels, int inHeight, int inWidth,
        int outChannels, int outHeight, int outWidth,
        int kernelH, int kernelW,
        int strideH, int strideW, int padH, int padW,
        int dilationH, int dilationW,
        int groups, int deformGroups)
    {
        if (!_kernelCache.TryGetValue("deformable_conv2d_backward_weights", out var cudaKernel))
            throw new InvalidOperationException("CUDA kernel not found: deformable_conv2d_backward_weights");

        using var _ = PushContext();
        int inChannelsPerGroup = inChannels / groups;
        int totalWeights = outChannels * inChannelsPerGroup * kernelH * kernelW;
        uint gridX = (uint)((totalWeights + DefaultBlockSize - 1) / DefaultBlockSize);

        IntPtr gradOutputPtr = gradOutput.Handle;
        IntPtr inputPtr = input.Handle;
        IntPtr offsetsPtr = offsets.Handle;
        IntPtr maskPtr = mask?.Handle ?? IntPtr.Zero;
        IntPtr gradWeightsPtr = gradWeights.Handle;
        int hasMask = mask is not null ? 1 : 0;

        void** args = stackalloc void*[23];
        args[0] = &gradOutputPtr;
        args[1] = &inputPtr;
        args[2] = &offsetsPtr;
        args[3] = &maskPtr;
        args[4] = &gradWeightsPtr;
        args[5] = &batch;
        args[6] = &inChannels;
        args[7] = &inHeight;
        args[8] = &inWidth;
        args[9] = &outChannels;
        args[10] = &outHeight;
        args[11] = &outWidth;
        args[12] = &kernelH;
        args[13] = &kernelW;
        args[14] = &strideH;
        args[15] = &strideW;
        args[16] = &padH;
        args[17] = &padW;
        args[18] = &dilationH;
        args[19] = &dilationW;
        args[20] = &groups;
        args[21] = &deformGroups;
        args[22] = &hasMask;
        LaunchKernel(cudaKernel, gridX, DefaultBlockSize, args);
    }

    public unsafe void DeformableConv2DBackwardOffset(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer weights, IGpuBuffer offsets, IGpuBuffer? mask, IGpuBuffer gradOffsets,
        int batch, int inChannels, int inHeight, int inWidth,
        int outChannels, int outHeight, int outWidth,
        int kernelH, int kernelW,
        int strideH, int strideW, int padH, int padW,
        int dilationH, int dilationW,
        int groups, int deformGroups)
    {
        if (!_kernelCache.TryGetValue("deformable_conv2d_backward_offset", out var cudaKernel))
            throw new InvalidOperationException("CUDA kernel not found: deformable_conv2d_backward_offset");

        using var _ = PushContext();
        const int blockSize = 16;
        int offsetChannels = 2 * kernelH * kernelW * deformGroups;
        uint gridX = (uint)((outWidth + blockSize - 1) / blockSize);
        uint gridY = (uint)((outHeight + blockSize - 1) / blockSize);
        uint gridZ = (uint)(batch * offsetChannels);

        IntPtr gradOutputPtr = gradOutput.Handle;
        IntPtr inputPtr = input.Handle;
        IntPtr weightsPtr = weights.Handle;
        IntPtr offsetsPtr = offsets.Handle;
        IntPtr maskPtr = mask?.Handle ?? IntPtr.Zero;
        IntPtr gradOffsetsPtr = gradOffsets.Handle;
        int hasMask = mask is not null ? 1 : 0;

        void** args = stackalloc void*[24];
        args[0] = &gradOutputPtr;
        args[1] = &inputPtr;
        args[2] = &weightsPtr;
        args[3] = &offsetsPtr;
        args[4] = &maskPtr;
        args[5] = &gradOffsetsPtr;
        args[6] = &batch;
        args[7] = &inChannels;
        args[8] = &inHeight;
        args[9] = &inWidth;
        args[10] = &outChannels;
        args[11] = &outHeight;
        args[12] = &outWidth;
        args[13] = &kernelH;
        args[14] = &kernelW;
        args[15] = &strideH;
        args[16] = &strideW;
        args[17] = &padH;
        args[18] = &padW;
        args[19] = &dilationH;
        args[20] = &dilationW;
        args[21] = &groups;
        args[22] = &deformGroups;
        args[23] = &hasMask;
        LaunchKernel2D(cudaKernel, gridX, gridY, gridZ, (uint)blockSize, (uint)blockSize, args);
    }

    public unsafe void DeformableConv2DBackwardMask(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer weights, IGpuBuffer offsets, IGpuBuffer gradMask,
        int batch, int inChannels, int inHeight, int inWidth,
        int outChannels, int outHeight, int outWidth,
        int kernelH, int kernelW,
        int strideH, int strideW, int padH, int padW,
        int dilationH, int dilationW,
        int groups, int deformGroups)
    {
        if (!_kernelCache.TryGetValue("deformable_conv2d_backward_mask", out var cudaKernel))
            throw new InvalidOperationException("CUDA kernel not found: deformable_conv2d_backward_mask");

        using var _ = PushContext();
        const int blockSize = 16;
        int maskChannels = kernelH * kernelW * deformGroups;
        uint gridX = (uint)((outWidth + blockSize - 1) / blockSize);
        uint gridY = (uint)((outHeight + blockSize - 1) / blockSize);
        uint gridZ = (uint)(batch * maskChannels);

        IntPtr gradOutputPtr = gradOutput.Handle;
        IntPtr inputPtr = input.Handle;
        IntPtr weightsPtr = weights.Handle;
        IntPtr offsetsPtr = offsets.Handle;
        IntPtr gradMaskPtr = gradMask.Handle;

        void** args = stackalloc void*[22];
        args[0] = &gradOutputPtr;
        args[1] = &inputPtr;
        args[2] = &weightsPtr;
        args[3] = &offsetsPtr;
        args[4] = &gradMaskPtr;
        args[5] = &batch;
        args[6] = &inChannels;
        args[7] = &inHeight;
        args[8] = &inWidth;
        args[9] = &outChannels;
        args[10] = &outHeight;
        args[11] = &outWidth;
        args[12] = &kernelH;
        args[13] = &kernelW;
        args[14] = &strideH;
        args[15] = &strideW;
        args[16] = &padH;
        args[17] = &padW;
        args[18] = &dilationH;
        args[19] = &dilationW;
        args[20] = &groups;
        args[21] = &deformGroups;
        LaunchKernel2D(cudaKernel, gridX, gridY, gridZ, (uint)blockSize, (uint)blockSize, args);
    }

    #endregion

    #endregion


    #region Pooling Operations

    public unsafe void MaxPool2D(IGpuBuffer input, IGpuBuffer output, IGpuBuffer? indices,
        int batch, int channels, int inHeight, int inWidth,
        int outHeight, int outWidth,
        int kernelH, int kernelW,
        int strideH, int strideW, int padH, int padW)
    {
        if (!_kernelCache.TryGetValue("maxpool2d", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: maxpool2d");

        using var _ = PushContext();
        const int blockSize = 16;
        uint gridX = (uint)((outWidth + blockSize - 1) / blockSize);
        uint gridY = (uint)((outHeight + blockSize - 1) / blockSize);
        uint gridZ = (uint)(batch * channels);

        IntPtr inputPtr = input.Handle;
        IntPtr outputPtr = output.Handle;
        IntPtr indicesPtr = indices?.Handle ?? IntPtr.Zero;
        int saveIndices = indices is not null ? 1 : 0;
        void** args = stackalloc void*[16];
        args[0] = &inputPtr;
        args[1] = &outputPtr;
        args[2] = &indicesPtr;
        args[3] = &batch;
        args[4] = &channels;
        args[5] = &inHeight;
        args[6] = &inWidth;
        args[7] = &outHeight;
        args[8] = &outWidth;
        args[9] = &kernelH;
        args[10] = &kernelW;
        args[11] = &strideH;
        args[12] = &strideW;
        args[13] = &padH;
        args[14] = &padW;
        args[15] = &saveIndices;
        LaunchKernel2D(kernel, gridX, gridY, gridZ, (uint)blockSize, (uint)blockSize, args);
    }

    public unsafe void MaxPool2DBackward(IGpuBuffer gradOutput, IGpuBuffer indices, IGpuBuffer gradInput,
        int batch, int channels, int inHeight, int inWidth,
        int outHeight, int outWidth,
        int kernelH, int kernelW,
        int strideH, int strideW, int padH, int padW)
    {
        using var _ = PushContext();
        const int blockSize = 16;
        IntPtr gradOutPtr = gradOutput.Handle;
        IntPtr indicesPtr = indices.Handle;
        IntPtr gradInPtr = gradInput.Handle;
        // Both maxpool2d_backward kernels take NINE parameters — the last two are
        // (outHeight, outWidth). Passing only eight args (a single outWidth) left the
        // kernel's outWidth reading an unfilled arg slot — garbage — which blew up
        // outIdx into an out-of-bounds gradOutput/indices read and a hard CUDA access
        // violation (0xC0000005) one step into CNN backprop (AiDotNet#1468). Pass both.
        int outH = outHeight;
        int outW = outWidth;

        if (GpuDeterminism.IsActive)
        {
            // Issue #382: route to atomic-free variant.
            if (!_kernelCache.TryGetValue("maxpool2d_backward_deterministic", out var kernelD))
                throw new InvalidOperationException("CUDA kernel not found: maxpool2d_backward_deterministic");
            void** argsD = stackalloc void*[9];
            argsD[0] = &gradOutPtr;
            argsD[1] = &indicesPtr;
            argsD[2] = &gradInPtr;
            argsD[3] = &batch;
            argsD[4] = &channels;
            argsD[5] = &inHeight;
            argsD[6] = &inWidth;
            argsD[7] = &outH;
            argsD[8] = &outW;
            // Deterministic variant grid: per (b, c, ih, iw) input cell.
            uint gridDX = (uint)((inWidth + blockSize - 1) / blockSize);
            uint gridDY = (uint)((inHeight + blockSize - 1) / blockSize);
            uint gridDZ = (uint)(batch * channels);
            LaunchKernel2D(kernelD, gridDX, gridDY, gridDZ, (uint)blockSize, (uint)blockSize, argsD);
            return;
        }

        if (!_kernelCache.TryGetValue("maxpool2d_backward", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: maxpool2d_backward");

        // The non-deterministic kernel SCATTERS via atomicAdd to max cells only — non-max cells get no
        // write, so gradInput must read as zero. AllocateBuffer zeroes it via a null-stream cuMemsetD32,
        // which is NOT ordered against this kernel's non-blocking compute stream and can lag under load
        // (the deterministic kernel sidesteps this by assigning every cell; this scatter cannot). Zero
        // it again STREAM-ORDERED on _stream so the memset strictly precedes the kernel, no host sync.
        CuBlasNative.CheckCudaResult(
            CudaNativeBindings.cuMemsetD8Async(gradInPtr, 0, (ulong)((long)gradInput.Size * sizeof(float)), _stream),
            "cuMemsetD8Async(maxpool2d_backward gradInput)");

        uint gridX = (uint)((outWidth + blockSize - 1) / blockSize);
        uint gridY = (uint)((outHeight + blockSize - 1) / blockSize);
        uint gridZ = (uint)(batch * channels);

        void** args = stackalloc void*[9];
        args[0] = &gradOutPtr;
        args[1] = &indicesPtr;
        args[2] = &gradInPtr;
        args[3] = &batch;
        args[4] = &channels;
        args[5] = &inHeight;
        args[6] = &inWidth;
        args[7] = &outH;
        args[8] = &outW;
        LaunchKernel2D(kernel, gridX, gridY, gridZ, (uint)blockSize, (uint)blockSize, args);
    }

    public unsafe void AvgPool2D(IGpuBuffer input, IGpuBuffer output,
        int batch, int channels, int inHeight, int inWidth,
        int outHeight, int outWidth,
        int kernelH, int kernelW,
        int strideH, int strideW, int padH, int padW,
        bool countIncludePad)
    {
        if (!_kernelCache.TryGetValue("avgpool2d", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: avgpool2d");

        using var _ = PushContext();
        const int blockSize = 16;
        uint gridX = (uint)((outWidth + blockSize - 1) / blockSize);
        uint gridY = (uint)((outHeight + blockSize - 1) / blockSize);
        uint gridZ = (uint)(batch * channels);

        IntPtr inputPtr = input.Handle;
        IntPtr outputPtr = output.Handle;
        int countPad = countIncludePad ? 1 : 0;
        void** args = stackalloc void*[14];
        args[0] = &inputPtr;
        args[1] = &outputPtr;
        args[2] = &batch;
        args[3] = &channels;
        args[4] = &inHeight;
        args[5] = &inWidth;
        args[6] = &outHeight;
        args[7] = &outWidth;
        args[8] = &kernelH;
        args[9] = &kernelW;
        args[10] = &strideH;
        args[11] = &strideW;
        args[12] = &padH;
        args[13] = &padW;
        LaunchKernel2D(kernel, gridX, gridY, gridZ, (uint)blockSize, (uint)blockSize, args);
    }

    public unsafe void AvgPool2DBackward(IGpuBuffer gradOutput, IGpuBuffer gradInput,
        int batch, int channels, int inHeight, int inWidth,
        int outHeight, int outWidth,
        int kernelH, int kernelW,
        int strideH, int strideW, int padH, int padW,
        bool countIncludePad)
    {
        if (!_kernelCache.TryGetValue("avgpool2d_backward", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: avgpool2d_backward");

        using var _ = PushContext();
        const int blockSize = 16;
        uint gridX = (uint)((inWidth + blockSize - 1) / blockSize);
        uint gridY = (uint)((inHeight + blockSize - 1) / blockSize);
        uint gridZ = (uint)(batch * channels);

        IntPtr gradOutPtr = gradOutput.Handle;
        IntPtr gradInPtr = gradInput.Handle;
        int countPad = countIncludePad ? 1 : 0;
        // Kernel signature has 15 parameters — gradOutput, gradInput, the 12
        // shape/stride/pad ints, and countIncludePad. The original 14-slot
        // stackalloc dropped the countIncludePad slot, so cuLaunchKernel read
        // garbage stack memory for parameter 15 and crashed with 0xC0000005
        // on small inputs (issue surfaced by
        // AvgPool2D_IntArrayOverload_ProducesNonZeroInputGradient). Match
        // the kernel signature exactly.
        void** args = stackalloc void*[15];
        args[0] = &gradOutPtr;
        args[1] = &gradInPtr;
        args[2] = &batch;
        args[3] = &channels;
        args[4] = &inHeight;
        args[5] = &inWidth;
        args[6] = &outHeight;
        args[7] = &outWidth;
        args[8] = &kernelH;
        args[9] = &kernelW;
        args[10] = &strideH;
        args[11] = &strideW;
        args[12] = &padH;
        args[13] = &padW;
        args[14] = &countPad;
        LaunchKernel2D(kernel, gridX, gridY, gridZ, (uint)blockSize, (uint)blockSize, args);
    }

    public unsafe void GlobalAvgPool2D(IGpuBuffer input, IGpuBuffer output, int batch, int channels, int height, int width)
    {
        if (!_kernelCache.TryGetValue("global_avgpool2d", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: global_avgpool2d");

        using var _ = PushContext();
        uint grid = (uint)((batch * channels + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr inputPtr = input.Handle;
        IntPtr outputPtr = output.Handle;
        void** args = stackalloc void*[6];
        args[0] = &inputPtr;
        args[1] = &outputPtr;
        args[2] = &batch;
        args[3] = &channels;
        args[4] = &height;
        args[5] = &width;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    public unsafe void GlobalMaxPool2D(IGpuBuffer input, IGpuBuffer output, int batch, int channels, int height, int width)
    {
        // Overload without indices - backward pass will not be available
        GlobalMaxPool2D(input, output, (IGpuBuffer?)null, batch, channels, height, width);
    }

    // Interface implementation for non-nullable indices (required for backward pass)
    void IDirectGpuBackend.GlobalMaxPool2D(IGpuBuffer input, IGpuBuffer output, IGpuBuffer indices, int batch, int channels, int height, int width)
        => GlobalMaxPool2D(input, output, (IGpuBuffer?)indices, batch, channels, height, width);

    public unsafe void GlobalMaxPool2D(IGpuBuffer input, IGpuBuffer output, IGpuBuffer? indices, int batch, int channels, int height, int width)
    {
        if (!_kernelCache.TryGetValue("global_maxpool2d", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: global_maxpool2d");

        using var _ = PushContext();
        uint grid = (uint)((batch * channels + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr inputPtr = input.Handle;
        IntPtr outputPtr = output.Handle;
        // Pass IntPtr.Zero if indices is null; kernel checks saveIndices flag
        IntPtr indicesPtr = indices?.Handle ?? IntPtr.Zero;
        int saveIndices = indices is not null ? 1 : 0;
        void** args = stackalloc void*[8];
        args[0] = &inputPtr;
        args[1] = &outputPtr;
        args[2] = &indicesPtr;
        args[3] = &batch;
        args[4] = &channels;
        args[5] = &height;
        args[6] = &width;
        args[7] = &saveIndices;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    public unsafe void GlobalAvgPool2DBackward(IGpuBuffer gradOutput, IGpuBuffer gradInput, int batch, int channels, int height, int width)
    {
        if (!_kernelCache.TryGetValue("global_avgpool2d_backward", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: global_avgpool2d_backward");

        using var _ = PushContext();
        int totalElements = batch * channels * height * width;
        uint grid = (uint)((totalElements + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr gradOutPtr = gradOutput.Handle;
        IntPtr gradInPtr = gradInput.Handle;
        void** args = stackalloc void*[6];
        args[0] = &gradOutPtr;
        args[1] = &gradInPtr;
        args[2] = &batch;
        args[3] = &channels;
        args[4] = &height;
        args[5] = &width;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    public unsafe void GlobalMaxPool2DBackward(IGpuBuffer gradOutput, IGpuBuffer indices, IGpuBuffer gradInput, int batch, int channels, int height, int width)
    {
        using var _ = PushContext();
        Fill(gradInput, 0f, batch * channels * height * width);

        int totalOutputs = batch * channels;
        uint grid = (uint)((totalOutputs + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr gradOutPtr = gradOutput.Handle;
        IntPtr indicesPtr = indices.Handle;
        IntPtr gradInPtr = gradInput.Handle;
        void** args = stackalloc void*[7];
        args[0] = &gradOutPtr;
        args[1] = &indicesPtr;
        args[2] = &gradInPtr;
        args[3] = &batch;
        args[4] = &channels;
        args[5] = &height;
        args[6] = &width;

        string kernelName = GpuDeterminism.IsActive
            ? "global_maxpool2d_backward_deterministic"
            : "global_maxpool2d_backward";
        if (!_kernelCache.TryGetValue(kernelName, out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: " + kernelName);

        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    public unsafe void AdaptiveAvgPool2D(IGpuBuffer input, IGpuBuffer output, int batch, int channels, int inHeight, int inWidth, int outHeight, int outWidth)
    {
        if (!_kernelCache.TryGetValue("adaptive_avgpool2d", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: adaptive_avgpool2d");

        using var _ = PushContext();
        const int blockSize = 16;
        uint gridX = (uint)((outWidth + blockSize - 1) / blockSize);
        uint gridY = (uint)((outHeight + blockSize - 1) / blockSize);
        uint gridZ = (uint)(batch * channels);

        IntPtr inputPtr = input.Handle;
        IntPtr outputPtr = output.Handle;
        void** args = stackalloc void*[8];
        args[0] = &inputPtr;
        args[1] = &outputPtr;
        args[2] = &batch;
        args[3] = &channels;
        args[4] = &inHeight;
        args[5] = &inWidth;
        args[6] = &outHeight;
        args[7] = &outWidth;
        LaunchKernel2D(kernel, gridX, gridY, gridZ, (uint)blockSize, (uint)blockSize, args);
    }

    public unsafe void MaxPool3D(IGpuBuffer input, IGpuBuffer output, IGpuBuffer? indices,
        int batch, int channels,
        int inDepth, int inHeight, int inWidth,
        int outDepth, int outHeight, int outWidth,
        int kernelD, int kernelH, int kernelW,
        int strideD, int strideH, int strideW)
    {
        if (!_kernelCache.TryGetValue("maxpool3d", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: maxpool3d");

        using var _ = PushContext();
        const int blockSize = 8;
        uint gridX = (uint)((outWidth + blockSize - 1) / blockSize);
        uint gridY = (uint)((outHeight + blockSize - 1) / blockSize);
        uint gridZ = (uint)(batch * channels * outDepth);

        IntPtr inputPtr = input.Handle;
        IntPtr outputPtr = output.Handle;
        IntPtr indicesPtr = indices?.Handle ?? IntPtr.Zero;
        int saveIndices = indices is not null ? 1 : 0;

        void** args = stackalloc void*[18];
        args[0] = &inputPtr;
        args[1] = &outputPtr;
        args[2] = &indicesPtr;
        args[3] = &batch;
        args[4] = &channels;
        args[5] = &inDepth;
        args[6] = &inHeight;
        args[7] = &inWidth;
        args[8] = &outDepth;
        args[9] = &outHeight;
        args[10] = &outWidth;
        args[11] = &kernelD;
        args[12] = &kernelH;
        args[13] = &kernelW;
        args[14] = &strideD;
        args[15] = &strideH;
        args[16] = &strideW;
        args[17] = &saveIndices;

        LaunchKernel2D(kernel, gridX, gridY, gridZ, (uint)blockSize, (uint)blockSize, args);
    }

    public unsafe void MaxPool3DBackward(IGpuBuffer gradOutput, IGpuBuffer indices, IGpuBuffer gradInput,
        int batch, int channels,
        int inDepth, int inHeight, int inWidth,
        int outDepth, int outHeight, int outWidth)
    {
        using var _ = PushContext();
        const int blockSize = 8;
        IntPtr gradOutPtr = gradOutput.Handle;
        IntPtr indicesPtr = indices.Handle;
        IntPtr gradInPtr = gradInput.Handle;

        void** args = stackalloc void*[11];
        args[0] = &gradOutPtr;
        args[1] = &indicesPtr;
        args[2] = &gradInPtr;
        args[3] = &batch;
        args[4] = &channels;
        args[5] = &inDepth;
        args[6] = &inHeight;
        args[7] = &inWidth;
        args[8] = &outDepth;
        args[9] = &outHeight;
        args[10] = &outWidth;

        if (GpuDeterminism.IsActive)
        {
            // Issue #382: deterministic kernel parallelizes per input cell (iw, ih, b*c*id).
            if (!_kernelCache.TryGetValue("maxpool3d_backward_deterministic", out var kernelD))
                throw new InvalidOperationException("CUDA kernel not found: maxpool3d_backward_deterministic");
            uint gridDX = (uint)((inWidth + blockSize - 1) / blockSize);
            uint gridDY = (uint)((inHeight + blockSize - 1) / blockSize);
            uint gridDZ = (uint)(batch * channels * inDepth);
            LaunchKernel2D(kernelD, gridDX, gridDY, gridDZ, (uint)blockSize, (uint)blockSize, args);
            return;
        }

        if (!_kernelCache.TryGetValue("maxpool3d_backward", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: maxpool3d_backward");

        uint gridX = (uint)((outWidth + blockSize - 1) / blockSize);
        uint gridY = (uint)((outHeight + blockSize - 1) / blockSize);
        uint gridZ = (uint)(batch * channels * outDepth);
        LaunchKernel2D(kernel, gridX, gridY, gridZ, (uint)blockSize, (uint)blockSize, args);
    }

    public unsafe void NearestNeighborUpsample3D(IGpuBuffer input, IGpuBuffer output,
        int batch, int channels,
        int inDepth, int inHeight, int inWidth,
        int scaleD, int scaleH, int scaleW)
    {
        if (!_kernelCache.TryGetValue("nearest_upsample3d", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: nearest_upsample3d");

        using var _ = PushContext();
        int outDepth = inDepth * scaleD;
        int outHeight = inHeight * scaleH;
        int outWidth = inWidth * scaleW;

        const int blockSize = 8;
        uint gridX = (uint)((outWidth + blockSize - 1) / blockSize);
        uint gridY = (uint)((outHeight + blockSize - 1) / blockSize);
        uint gridZ = (uint)(batch * channels * outDepth);

        IntPtr inputPtr = input.Handle;
        IntPtr outputPtr = output.Handle;

        void** args = stackalloc void*[10];
        args[0] = &inputPtr;
        args[1] = &outputPtr;
        args[2] = &batch;
        args[3] = &channels;
        args[4] = &inDepth;
        args[5] = &inHeight;
        args[6] = &inWidth;
        args[7] = &scaleD;
        args[8] = &scaleH;
        args[9] = &scaleW;

        LaunchKernel2D(kernel, gridX, gridY, gridZ, (uint)blockSize, (uint)blockSize, args);
    }

    public unsafe void NearestNeighborUpsample3DBackward(IGpuBuffer gradOutput, IGpuBuffer gradInput,
        int batch, int channels,
        int inDepth, int inHeight, int inWidth,
        int scaleD, int scaleH, int scaleW)
    {
        using var _ = PushContext();
        const int blockSize = 8;
        IntPtr gradOutPtr = gradOutput.Handle;
        IntPtr gradInPtr = gradInput.Handle;

        if (GpuDeterminism.IsActive)
        {
            // Issue #382: per-input-cell scan of scale-cube output cells.
            if (!_kernelCache.TryGetValue("nearest_upsample3d_backward_deterministic", out var kernelD))
                throw new InvalidOperationException("CUDA kernel not found: nearest_upsample3d_backward_deterministic");
            void** argsD = stackalloc void*[10];
            argsD[0] = &gradOutPtr;
            argsD[1] = &gradInPtr;
            argsD[2] = &batch;
            argsD[3] = &channels;
            argsD[4] = &inDepth;
            argsD[5] = &inHeight;
            argsD[6] = &inWidth;
            argsD[7] = &scaleD;
            argsD[8] = &scaleH;
            argsD[9] = &scaleW;
            uint gridDX = (uint)((inWidth + blockSize - 1) / blockSize);
            uint gridDY = (uint)((inHeight + blockSize - 1) / blockSize);
            uint gridDZ = (uint)(batch * channels * inDepth);
            LaunchKernel2D(kernelD, gridDX, gridDY, gridDZ, (uint)blockSize, (uint)blockSize, argsD);
            return;
        }

        if (!_kernelCache.TryGetValue("nearest_upsample3d_backward", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: nearest_upsample3d_backward");

        int outDepth = inDepth * scaleD;
        int outHeight = inHeight * scaleH;
        int outWidth = inWidth * scaleW;

        uint gridX = (uint)((outWidth + blockSize - 1) / blockSize);
        uint gridY = (uint)((outHeight + blockSize - 1) / blockSize);
        uint gridZ = (uint)(batch * channels * outDepth);

        void** args = stackalloc void*[10];
        args[0] = &gradOutPtr;
        args[1] = &gradInPtr;
        args[2] = &batch;
        args[3] = &channels;
        args[4] = &inDepth;
        args[5] = &inHeight;
        args[6] = &inWidth;
        args[7] = &scaleD;
        args[8] = &scaleH;
        args[9] = &scaleW;

        LaunchKernel2D(kernel, gridX, gridY, gridZ, (uint)blockSize, (uint)blockSize, args);
    }

    #endregion

    #region Spatial Transformer Operations

    public unsafe void AffineGrid(IGpuBuffer theta, IGpuBuffer grid, int batch, int outputHeight, int outputWidth)
    {
        if (!_kernelCache.TryGetValue("affine_grid", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: affine_grid");

        using var _ = PushContext();
        const int blockSize = 16;
        uint gridX = (uint)((outputWidth + blockSize - 1) / blockSize);
        uint gridY = (uint)((outputHeight + blockSize - 1) / blockSize);
        uint gridZ = (uint)batch;

        IntPtr thetaPtr = theta.Handle;
        IntPtr gridPtr = grid.Handle;
        void** args = stackalloc void*[5];
        args[0] = &thetaPtr;
        args[1] = &gridPtr;
        args[2] = &batch;
        args[3] = &outputHeight;
        args[4] = &outputWidth;
        LaunchKernel2D(kernel, gridX, gridY, gridZ, (uint)blockSize, (uint)blockSize, args);
    }

    public unsafe void GridSample(IGpuBuffer input, IGpuBuffer grid, IGpuBuffer output,
        int batch, int channels, int inHeight, int inWidth, int outHeight, int outWidth,
        int paddingMode = 0, bool alignCorners = false)
    {
        if (!_kernelCache.TryGetValue("grid_sample", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: grid_sample");

        using var _ = PushContext();
        const int blockSize = 16;
        uint gridX = (uint)((outWidth + blockSize - 1) / blockSize);
        uint gridY = (uint)((outHeight + blockSize - 1) / blockSize);
        uint gridZ = (uint)(batch * channels);

        IntPtr inputPtr = input.Handle;
        IntPtr gridPtr = grid.Handle;
        IntPtr outputPtr = output.Handle;
        int alignCornersInt = alignCorners ? 1 : 0;

        void** args = stackalloc void*[12];
        args[0] = &inputPtr;
        args[1] = &gridPtr;
        args[2] = &outputPtr;
        args[3] = &batch;
        args[4] = &channels;
        args[5] = &inHeight;
        args[6] = &inWidth;
        args[7] = &outHeight;
        args[8] = &outWidth;
        args[9] = &paddingMode;
        args[10] = &alignCornersInt;
        LaunchKernel2D(kernel, gridX, gridY, gridZ, (uint)blockSize, (uint)blockSize, args);
    }

    public unsafe void GridSampleBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer grid,
        IGpuBuffer gradInput, IGpuBuffer gradGrid,
        int batch, int channels, int inHeight, int inWidth, int outHeight, int outWidth,
        int paddingMode = 0, bool alignCorners = false)
    {
        using var _ = PushContext();
        const int blockSize = 16;
        IntPtr gradOutputPtr = gradOutput.Handle;
        IntPtr inputPtr = input.Handle;
        IntPtr gridPtr = grid.Handle;
        IntPtr gradInputPtr = gradInput.Handle;
        IntPtr gradGridPtr = gradGrid.Handle;
        int alignCornersInt = alignCorners ? 1 : 0;

        if (GpuDeterminism.IsActive)
        {
            // Issue #382: split into gradGrid (per output position) + gradInput
            // (per input cell) atomic-free kernels.
            if (!_kernelCache.TryGetValue("grid_sample_backward_grad_grid_deterministic", out var krnlGrid))
                throw new InvalidOperationException("CUDA kernel not found: grid_sample_backward_grad_grid_deterministic");

            void** argsG = stackalloc void*[13];
            argsG[0] = &gradOutputPtr;
            argsG[1] = &inputPtr;
            argsG[2] = &gridPtr;
            argsG[3] = &gradGridPtr;
            argsG[4] = &batch;
            argsG[5] = &channels;
            argsG[6] = &inHeight;
            argsG[7] = &inWidth;
            argsG[8] = &outHeight;
            argsG[9] = &outWidth;
            argsG[10] = &paddingMode;
            argsG[11] = &alignCornersInt;
            uint gridGX = (uint)((outWidth + blockSize - 1) / blockSize);
            uint gridGY = (uint)((outHeight + blockSize - 1) / blockSize);
            uint gridGZ = (uint)batch;
            LaunchKernel2D(krnlGrid, gridGX, gridGY, gridGZ, (uint)blockSize, (uint)blockSize, argsG);

            if (!_kernelCache.TryGetValue("grid_sample_backward_grad_input_deterministic", out var krnlIn))
                throw new InvalidOperationException("CUDA kernel not found: grid_sample_backward_grad_input_deterministic");

            void** argsI = stackalloc void*[11];
            argsI[0] = &gradOutputPtr;
            argsI[1] = &gridPtr;
            argsI[2] = &gradInputPtr;
            argsI[3] = &batch;
            argsI[4] = &channels;
            argsI[5] = &inHeight;
            argsI[6] = &inWidth;
            argsI[7] = &outHeight;
            argsI[8] = &outWidth;
            argsI[9] = &paddingMode;
            argsI[10] = &alignCornersInt;
            uint gridIX = (uint)((inWidth + blockSize - 1) / blockSize);
            uint gridIY = (uint)((inHeight + blockSize - 1) / blockSize);
            uint gridIZ = (uint)(batch * channels);
            LaunchKernel2D(krnlIn, gridIX, gridIY, gridIZ, (uint)blockSize, (uint)blockSize, argsI);
            return;
        }

        if (!_kernelCache.TryGetValue("grid_sample_backward", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: grid_sample_backward");

        uint gridX = (uint)((outWidth + blockSize - 1) / blockSize);
        uint gridY = (uint)((outHeight + blockSize - 1) / blockSize);
        uint gridZ = (uint)batch;

        void** args = stackalloc void*[14];
        args[0] = &gradOutputPtr;
        args[1] = &inputPtr;
        args[2] = &gridPtr;
        args[3] = &gradInputPtr;
        args[4] = &gradGridPtr;
        args[5] = &batch;
        args[6] = &channels;
        args[7] = &inHeight;
        args[8] = &inWidth;
        args[9] = &outHeight;
        args[10] = &outWidth;
        args[11] = &paddingMode;
        args[12] = &alignCornersInt;
        LaunchKernel2D(kernel, gridX, gridY, gridZ, (uint)blockSize, (uint)blockSize, args);
    }

    #endregion

    #region Normalization Operations

    public unsafe void BatchNorm(IGpuBuffer input, IGpuBuffer output, IGpuBuffer gamma, IGpuBuffer beta,
        IGpuBuffer runningMean, IGpuBuffer runningVar, IGpuBuffer saveMean, IGpuBuffer saveInvVar,
        int batch, int channels, int spatialSize, float epsilon, float momentum, bool training)
    {
        if (!_kernelCache.TryGetValue("batchnorm_forward", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: batchnorm_forward");

        // Dispatch path telemetry — see Conv2D for the rationale.
        using var _profile = CudaDispatchPolicy.Scope(
            "BatchNorm",
            useVendor: false);
        using var _ = PushContext();
        uint gridX = (uint)channels;
        IntPtr inputPtr = input.Handle;
        IntPtr outputPtr = output.Handle;
        IntPtr gammaPtr = gamma.Handle;
        IntPtr betaPtr = beta.Handle;
        IntPtr runMeanPtr = runningMean.Handle;
        IntPtr runVarPtr = runningVar.Handle;
        IntPtr saveMeanPtr = saveMean.Handle;
        IntPtr saveInvVarPtr = saveInvVar.Handle;
        int trainingInt = training ? 1 : 0;
        void** args = stackalloc void*[14];
        args[0] = &inputPtr;
        args[1] = &outputPtr;
        args[2] = &gammaPtr;
        args[3] = &betaPtr;
        args[4] = &runMeanPtr;
        args[5] = &runVarPtr;
        args[6] = &saveMeanPtr;
        args[7] = &saveInvVarPtr;
        args[8] = &batch;
        args[9] = &channels;
        args[10] = &spatialSize;
        args[11] = &epsilon;
        args[12] = &momentum;
        args[13] = &trainingInt;
        // 1 block per channel, 256 threads, 1 shared array for parallel reduction
        LaunchKernelWithSharedMem(kernel, gridX, DefaultBlockSize, (uint)(DefaultBlockSize * sizeof(float)), args);
    }

    public unsafe bool TryFusedBatchNormActivation(IGpuBuffer input, IGpuBuffer output, IGpuBuffer gamma, IGpuBuffer beta,
        IGpuBuffer runningMean, IGpuBuffer runningVar, IGpuBuffer saveMean, IGpuBuffer saveInvVar,
        int batch, int channels, int spatialSize, float epsilon, float momentum, bool training,
        FusedActivationType activation)
    {
        // Fused kernels only support inference mode (use running stats, no save mean/var)
        if (training || activation == FusedActivationType.None)
            return false;

        string kernelName = activation switch
        {
            FusedActivationType.ReLU => "batchnorm_relu",
            FusedActivationType.GELU => "batchnorm_gelu",
            FusedActivationType.Sigmoid => "batchnorm_sigmoid",
            FusedActivationType.Tanh => "batchnorm_tanh",
            _ => ""
        };

        if (string.IsNullOrEmpty(kernelName) || !_kernelCache.TryGetValue(kernelName, out var kernel))
            return false;

        using var _ = PushContext();
        int totalSize = batch * channels * spatialSize;
        uint grid = (uint)((totalSize + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr inputPtr = input.Handle;
        IntPtr outputPtr = output.Handle;
        IntPtr gammaPtr = gamma.Handle;
        IntPtr betaPtr = beta.Handle;
        IntPtr runMeanPtr = runningMean.Handle;
        IntPtr runVarPtr = runningVar.Handle;
        void** args = stackalloc void*[10];
        args[0] = &inputPtr;
        args[1] = &outputPtr;
        args[2] = &gammaPtr;
        args[3] = &betaPtr;
        args[4] = &runMeanPtr;
        args[5] = &runVarPtr;
        args[6] = &batch;
        args[7] = &channels;
        args[8] = &spatialSize;
        args[9] = &epsilon;
        LaunchKernel(kernel, grid, (uint)DefaultBlockSize, args);
        return true;
    }

    public unsafe void BatchNormBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gamma,
        IGpuBuffer saveMean, IGpuBuffer saveInvVar, IGpuBuffer gradInput, IGpuBuffer gradGamma, IGpuBuffer gradBeta,
        int batch, int channels, int spatialSize, float epsilon)
    {
        if (!_kernelCache.TryGetValue("batchnorm_backward", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: batchnorm_backward");

        using var _ = PushContext();
        uint gridX = (uint)channels;
        IntPtr gradOutputPtr = gradOutput.Handle;
        IntPtr inputPtr = input.Handle;
        IntPtr gammaPtr = gamma.Handle;
        IntPtr saveMeanPtr = saveMean.Handle;
        IntPtr saveInvVarPtr = saveInvVar.Handle;
        IntPtr gradInputPtr = gradInput.Handle;
        IntPtr gradGammaPtr = gradGamma.Handle;
        IntPtr gradBetaPtr = gradBeta.Handle;
        void** args = stackalloc void*[12];
        args[0] = &gradOutputPtr;
        args[1] = &inputPtr;
        args[2] = &gammaPtr;
        args[3] = &saveMeanPtr;
        args[4] = &saveInvVarPtr;
        args[5] = &gradInputPtr;
        args[6] = &gradGammaPtr;
        args[7] = &gradBetaPtr;
        args[8] = &batch;
        args[9] = &channels;
        args[10] = &spatialSize;
        args[11] = &epsilon;
        // 3 shared arrays for dGamma, dBeta, sumDyXmu parallel reductions
        LaunchKernelWithSharedMem(kernel, gridX, DefaultBlockSize, (uint)(3 * DefaultBlockSize * sizeof(float)), args);
    }

    public unsafe void LayerNorm(IGpuBuffer input, IGpuBuffer output, IGpuBuffer gamma, IGpuBuffer beta,
        IGpuBuffer saveMean, IGpuBuffer saveInvVar, int batchSize, int normalizedSize, float epsilon)
    {
        if (!_kernelCache.TryGetValue("layernorm_forward", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: layernorm_forward");

        using var _ = PushContext();
        uint gridX = (uint)batchSize;
        IntPtr inputPtr = input.Handle;
        IntPtr outputPtr = output.Handle;
        IntPtr gammaPtr = gamma.Handle;
        IntPtr betaPtr = beta.Handle;
        IntPtr saveMeanPtr = saveMean.Handle;
        IntPtr saveInvVarPtr = saveInvVar.Handle;
        void** args = stackalloc void*[9];
        args[0] = &inputPtr;
        args[1] = &outputPtr;
        args[2] = &gammaPtr;
        args[3] = &betaPtr;
        args[4] = &saveMeanPtr;
        args[5] = &saveInvVarPtr;
        args[6] = &batchSize;
        args[7] = &normalizedSize;
        args[8] = &epsilon;
        // 1 block per batch element, 1 shared array
        LaunchKernelWithSharedMem(kernel, gridX, DefaultBlockSize, (uint)(DefaultBlockSize * sizeof(float)), args);
    }

    public unsafe void LayerNormBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gamma,
        IGpuBuffer saveMean, IGpuBuffer saveInvVar, IGpuBuffer gradInput, IGpuBuffer gradGamma, IGpuBuffer gradBeta,
        int batchSize, int normalizedSize, float epsilon)
    {
        if (!_kernelCache.TryGetValue("layernorm_backward", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: layernorm_backward");

        using var _ = PushContext();
        uint gridX = (uint)batchSize;
        IntPtr gradOutputPtr = gradOutput.Handle;
        IntPtr inputPtr = input.Handle;
        IntPtr gammaPtr = gamma.Handle;
        IntPtr saveMeanPtr = saveMean.Handle;
        IntPtr saveInvVarPtr = saveInvVar.Handle;
        IntPtr gradInputPtr = gradInput.Handle;
        IntPtr gradGammaPtr = gradGamma.Handle;
        IntPtr gradBetaPtr = gradBeta.Handle;
        void** args = stackalloc void*[11];
        args[0] = &gradOutputPtr;
        args[1] = &inputPtr;
        args[2] = &gammaPtr;
        args[3] = &saveMeanPtr;
        args[4] = &saveInvVarPtr;
        args[5] = &gradInputPtr;
        args[6] = &gradGammaPtr;
        args[7] = &gradBetaPtr;
        args[8] = &batchSize;
        args[9] = &normalizedSize;
        args[10] = &epsilon;
        // 2 shared arrays for sumDy and sumDyXmu
        LaunchKernelWithSharedMem(kernel, gridX, DefaultBlockSize, (uint)(2 * DefaultBlockSize * sizeof(float)), args);

        // The layernorm_backward kernel above computes ONLY gradInput — it leaves
        // gradGamma/gradBeta untouched (they require a reduction ACROSS the batch, not a
        // per-row block result). Launch the dedicated param-gradient kernel; without it,
        // LayerNorm's affine gamma/beta receive ZERO gradient on GPU and never train
        // (every transformer layer has a LayerNorm — caught by LayerNormGradientCheckTests).
        if (!_kernelCache.TryGetValue("layernorm_grad_params", out var paramKernel))
            throw new InvalidOperationException("CUDA kernel not found: layernorm_grad_params");
        uint paramGrid = (uint)((normalizedSize + DefaultBlockSize - 1) / DefaultBlockSize);
        void** pargs = stackalloc void*[8];
        pargs[0] = &gradOutputPtr;
        pargs[1] = &inputPtr;
        pargs[2] = &saveMeanPtr;
        pargs[3] = &saveInvVarPtr;
        pargs[4] = &gradGammaPtr;
        pargs[5] = &gradBetaPtr;
        pargs[6] = &batchSize;
        pargs[7] = &normalizedSize;
        LaunchKernel(paramKernel, paramGrid, DefaultBlockSize, pargs);
    }

    public unsafe void GroupNorm(IGpuBuffer input, IGpuBuffer output, IGpuBuffer gamma, IGpuBuffer beta,
        IGpuBuffer saveMean, IGpuBuffer saveInvVar, int batch, int numGroups, int channels, int spatialSize, float epsilon)
    {
        if (!_kernelCache.TryGetValue("groupnorm_forward", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: groupnorm_forward");

        using var _ = PushContext();
        uint gridX = (uint)(batch * numGroups);
        IntPtr inputPtr = input.Handle;
        IntPtr outputPtr = output.Handle;
        IntPtr gammaPtr = gamma.Handle;
        IntPtr betaPtr = beta.Handle;
        IntPtr saveMeanPtr = saveMean.Handle;
        IntPtr saveInvVarPtr = saveInvVar.Handle;
        void** args = stackalloc void*[11];
        args[0] = &inputPtr;
        args[1] = &outputPtr;
        args[2] = &gammaPtr;
        args[3] = &betaPtr;
        args[4] = &saveMeanPtr;
        args[5] = &saveInvVarPtr;
        args[6] = &batch;
        args[7] = &numGroups;
        args[8] = &channels;
        args[9] = &spatialSize;
        args[10] = &epsilon;
        // 1 block per (batch, group), 1 shared array
        LaunchKernelWithSharedMem(kernel, gridX, DefaultBlockSize, (uint)(DefaultBlockSize * sizeof(float)), args);
    }

    public unsafe void InstanceNorm(IGpuBuffer input, IGpuBuffer output, IGpuBuffer gamma, IGpuBuffer beta,
        IGpuBuffer saveMean, IGpuBuffer saveInvVar, int batch, int channels, int spatialSize, float epsilon)
    {
        if (!_kernelCache.TryGetValue("instancenorm_forward", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: instancenorm_forward");

        using var _ = PushContext();
        uint gridX = (uint)(batch * channels);
        IntPtr inputPtr = input.Handle;
        IntPtr outputPtr = output.Handle;
        IntPtr gammaPtr = gamma.Handle;
        IntPtr betaPtr = beta.Handle;
        IntPtr saveMeanPtr = saveMean.Handle;
        IntPtr saveInvVarPtr = saveInvVar.Handle;
        void** args = stackalloc void*[10];
        args[0] = &inputPtr;
        args[1] = &outputPtr;
        args[2] = &gammaPtr;
        args[3] = &betaPtr;
        args[4] = &saveMeanPtr;
        args[5] = &saveInvVarPtr;
        args[6] = &batch;
        args[7] = &channels;
        args[8] = &spatialSize;
        args[9] = &epsilon;
        // 1 block per (batch, channel), 1 shared array
        LaunchKernelWithSharedMem(kernel, gridX, DefaultBlockSize, (uint)(DefaultBlockSize * sizeof(float)), args);
    }

    public unsafe void InstanceNormBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gamma,
        IGpuBuffer saveMean, IGpuBuffer saveInvVar,
        IGpuBuffer gradInput, IGpuBuffer gradGamma, IGpuBuffer gradBeta,
        int batch, int channels, int spatialSize, float epsilon)
    {
        // Parameter validation
        if (batch <= 0)
            throw new ArgumentOutOfRangeException(nameof(batch), batch, "batch must be positive.");
        if (channels <= 0)
            throw new ArgumentOutOfRangeException(nameof(channels), channels, "channels must be positive.");
        if (spatialSize <= 0)
            throw new ArgumentOutOfRangeException(nameof(spatialSize), spatialSize, "spatialSize must be positive (would cause divide-by-zero).");

        // Buffer size validation
        long requiredSpatial = (long)batch * channels * spatialSize;
        long requiredStats = (long)batch * channels;

        if (gradOutput.Size < requiredSpatial)
            throw new ArgumentException($"gradOutput buffer too small: expected at least {requiredSpatial} elements, got {gradOutput.Size}.", nameof(gradOutput));
        if (input.Size < requiredSpatial)
            throw new ArgumentException($"input buffer too small: expected at least {requiredSpatial} elements, got {input.Size}.", nameof(input));
        if (gamma.Size < channels)
            throw new ArgumentException($"gamma buffer too small: expected at least {channels} elements, got {gamma.Size}.", nameof(gamma));
        if (saveMean.Size < requiredStats)
            throw new ArgumentException($"saveMean buffer too small: expected at least {requiredStats} elements, got {saveMean.Size}.", nameof(saveMean));
        if (saveInvVar.Size < requiredStats)
            throw new ArgumentException($"saveInvVar buffer too small: expected at least {requiredStats} elements, got {saveInvVar.Size}.", nameof(saveInvVar));
        if (gradInput.Size < requiredSpatial)
            throw new ArgumentException($"gradInput buffer too small: expected at least {requiredSpatial} elements, got {gradInput.Size}.", nameof(gradInput));
        if (gradGamma.Size < channels)
            throw new ArgumentException($"gradGamma buffer too small: expected at least {channels} elements, got {gradGamma.Size}.", nameof(gradGamma));
        if (gradBeta.Size < channels)
            throw new ArgumentException($"gradBeta buffer too small: expected at least {channels} elements, got {gradBeta.Size}.", nameof(gradBeta));

        // Use instancenorm_backward kernel if available, otherwise fall back to layernorm pattern
        if (!_kernelCache.TryGetValue("instancenorm_backward", out var kernel))
        {
            // Fallback: implement using basic CUDA operations
            // This computes: dx = invStd * (1/N) * (N * delta - sum(delta) - xNorm * sum(delta * xNorm))
            // where delta = gradOutput * gamma

            // For now, use CPU fallback via buffer download/upload
            var gradOutData = DownloadBuffer(gradOutput);
            var inputData = DownloadBuffer(input);
            var gammaData = DownloadBuffer(gamma);
            var meanData = DownloadBuffer(saveMean);
            var invVarData = DownloadBuffer(saveInvVar);
            var gradInputData = new float[gradOutData.Length];
            var gradGammaData = new float[channels];
            var gradBetaData = new float[channels];

            for (int b = 0; b < batch; b++)
            {
                for (int c = 0; c < channels; c++)
                {
                    int offset = (b * channels + c) * spatialSize;
                    float meanVal = meanData[b * channels + c];
                    float invStd = invVarData[b * channels + c];
                    float g = gammaData[c];

                    // First pass: compute sums for gradient correction
                    float sumDelta = 0.0f;
                    float sumDeltaXNorm = 0.0f;
                    for (int s = 0; s < spatialSize; s++)
                    {
                        float go = gradOutData[offset + s];
                        float x = inputData[offset + s];
                        float xNorm = (x - meanVal) * invStd;
                        float delta = go * g;

                        gradGammaData[c] += go * xNorm;
                        gradBetaData[c] += go;

                        sumDelta += delta;
                        sumDeltaXNorm += delta * xNorm;
                    }

                    // Second pass: compute gradInput with proper correction terms
                    float invN = 1.0f / spatialSize;
                    for (int s = 0; s < spatialSize; s++)
                    {
                        float go = gradOutData[offset + s];
                        float x = inputData[offset + s];
                        float xNorm = (x - meanVal) * invStd;
                        float delta = go * g;

                        // dx = invStd * invN * (N * delta - sum(delta) - xNorm * sum(delta * xNorm))
                        gradInputData[offset + s] = invStd * invN * (spatialSize * delta - sumDelta - xNorm * sumDeltaXNorm);
                    }
                }
            }

            // Upload results to GPU buffers
            using var ctx = PushContext();
            fixed (float* gradInputDataPtr = gradInputData)
            {
                CuBlasNative.CheckCudaResult(
                    CuBlasNative.cuMemcpyHtoD(gradInput.Handle, (IntPtr)gradInputDataPtr, (ulong)(gradInputData.Length * sizeof(float))),
                    "cuMemcpyHtoD (InstanceNormBackward gradInput)");
            }
            fixed (float* gradGammaDataPtr = gradGammaData)
            {
                CuBlasNative.CheckCudaResult(
                    CuBlasNative.cuMemcpyHtoD(gradGamma.Handle, (IntPtr)gradGammaDataPtr, (ulong)(gradGammaData.Length * sizeof(float))),
                    "cuMemcpyHtoD (InstanceNormBackward gradGamma)");
            }
            fixed (float* gradBetaDataPtr = gradBetaData)
            {
                CuBlasNative.CheckCudaResult(
                    CuBlasNative.cuMemcpyHtoD(gradBeta.Handle, (IntPtr)gradBetaDataPtr, (ulong)(gradBetaData.Length * sizeof(float))),
                    "cuMemcpyHtoD (InstanceNormBackward gradBeta)");
            }
            return;
        }

        using var _ = PushContext();
        uint gridX = (uint)(batch * channels);
        IntPtr gradOutputPtr = gradOutput.Handle;
        IntPtr inputPtr = input.Handle;
        IntPtr gammaPtr = gamma.Handle;
        IntPtr saveMeanPtr = saveMean.Handle;
        IntPtr saveInvVarPtr = saveInvVar.Handle;
        IntPtr gradInputPtr = gradInput.Handle;
        IntPtr gradGammaPtr = gradGamma.Handle;
        IntPtr gradBetaPtr = gradBeta.Handle;
        void** args = stackalloc void*[12];
        args[0] = &gradOutputPtr;
        args[1] = &inputPtr;
        args[2] = &gammaPtr;
        args[3] = &saveMeanPtr;
        args[4] = &saveInvVarPtr;
        args[5] = &gradInputPtr;
        args[6] = &gradGammaPtr;
        args[7] = &gradBetaPtr;
        args[8] = &batch;
        args[9] = &channels;
        args[10] = &spatialSize;
        args[11] = &epsilon;
        // InstanceNormBackward: 1 block per (batch, channel)
        LaunchKernelWithSharedMem(kernel, gridX, DefaultBlockSize, (uint)(DefaultBlockSize * sizeof(float)), args);
    }

    public unsafe void RmsNorm(IGpuBuffer input, IGpuBuffer output, IGpuBuffer gamma, IGpuBuffer saveRms,
        int batchSize, int normalizedSize, float epsilon)
    {
        if (!_kernelCache.TryGetValue("rmsnorm_forward", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: rmsnorm_forward");

        using var _ = PushContext();
        uint gridX = (uint)batchSize;
        IntPtr inputPtr = input.Handle;
        IntPtr outputPtr = output.Handle;
        IntPtr gammaPtr = gamma.Handle;
        IntPtr saveRmsPtr = saveRms.Handle;
        void** args = stackalloc void*[7];
        args[0] = &inputPtr;
        args[1] = &outputPtr;
        args[2] = &gammaPtr;
        args[3] = &saveRmsPtr;
        args[4] = &batchSize;
        args[5] = &normalizedSize;
        args[6] = &epsilon;
        // 1 block per batch element, 1 shared array
        LaunchKernelWithSharedMem(kernel, gridX, DefaultBlockSize, (uint)(DefaultBlockSize * sizeof(float)), args);
    }

    public unsafe void RmsNormBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gamma, IGpuBuffer saveRms,
        IGpuBuffer gradInput, IGpuBuffer gradGamma, int batchSize, int normalizedSize, float epsilon)
    {
        // Compute gradInput using rmsnorm_backward kernel
        if (!_kernelCache.TryGetValue("rmsnorm_backward", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: rmsnorm_backward");

        using var _ = PushContext();
        uint gridX = (uint)batchSize;
        IntPtr gradOutputPtr = gradOutput.Handle;
        IntPtr inputPtr = input.Handle;
        IntPtr gammaPtr = gamma.Handle;
        IntPtr saveRmsPtr = saveRms.Handle;
        IntPtr gradInputPtr = gradInput.Handle;
        IntPtr gradGammaPtr = gradGamma.Handle;
        void** args = stackalloc void*[9];
        args[0] = &gradOutputPtr;
        args[1] = &inputPtr;
        args[2] = &gammaPtr;
        args[3] = &saveRmsPtr;
        args[4] = &gradInputPtr;
        args[5] = &gradGammaPtr;
        args[6] = &batchSize;
        args[7] = &normalizedSize;
        args[8] = &epsilon;
        // 1 block per batch element, 1 shared array
        LaunchKernelWithSharedMem(kernel, gridX, DefaultBlockSize, (uint)(DefaultBlockSize * sizeof(float)), args);

        // Compute gradGamma using rmsnorm_grad_gamma kernel
        if (!_kernelCache.TryGetValue("rmsnorm_grad_gamma", out var kernel2))
            throw new InvalidOperationException("CUDA kernel not found: rmsnorm_grad_gamma");

        uint gridGamma = (uint)((normalizedSize + DefaultBlockSize - 1) / DefaultBlockSize);
        void** args2 = stackalloc void*[6];
        args2[0] = &gradOutputPtr;
        args2[1] = &inputPtr;
        args2[2] = &saveRmsPtr;
        args2[3] = &gradGammaPtr;
        args2[4] = &batchSize;
        args2[5] = &normalizedSize;
        LaunchKernel(kernel2, gridGamma, DefaultBlockSize, args2);
    }

    #endregion


    #region Dropout Operations

    public unsafe void Dropout(IGpuBuffer input, IGpuBuffer output, IGpuBuffer mask, int size, float dropoutRate, ulong seed, bool training)
    {
        if (!_kernelCache.TryGetValue("dropout_forward", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: dropout_forward");

        using var _ = PushContext();
        uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr inputPtr = input.Handle;
        IntPtr outputPtr = output.Handle;
        IntPtr maskPtr = mask.Handle;
        int trainingInt = training ? 1 : 0;
        void** args = stackalloc void*[6];
        args[0] = &inputPtr;
        args[1] = &outputPtr;
        args[2] = &maskPtr;
        args[3] = &size;
        args[4] = &dropoutRate;
        args[5] = &seed;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    public unsafe void DropoutBackward(IGpuBuffer gradOutput, IGpuBuffer mask, IGpuBuffer gradInput, int size, float dropoutRate)
    {
        if (!_kernelCache.TryGetValue("dropout_backward", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: dropout_backward");

        using var _ = PushContext();
        uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr gradOutputPtr = gradOutput.Handle;
        IntPtr maskPtr = mask.Handle;
        IntPtr gradInputPtr = gradInput.Handle;
        void** args = stackalloc void*[5];
        args[0] = &gradOutputPtr;
        args[1] = &maskPtr;
        args[2] = &gradInputPtr;
        args[3] = &size;
        args[4] = &dropoutRate;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    public unsafe bool TryFusedBiasDropout(IGpuBuffer input, IGpuBuffer output, IGpuBuffer bias, IGpuBuffer mask,
        int rows, int cols, float dropoutRate, float scale)
    {
        if (!_kernelCache.TryGetValue("bias_dropout", out var kernel))
            return false;

        using var _ = PushContext();
        // 2D grid: x = columns (threads per row), y = rows
        uint gridX = (uint)((cols + DefaultBlockSize - 1) / DefaultBlockSize);
        uint gridY = (uint)rows;
        IntPtr inputPtr = input.Handle;
        IntPtr outputPtr = output.Handle;
        IntPtr biasPtr = bias.Handle;
        IntPtr maskPtr = mask.Handle;
        void** args = stackalloc void*[8];
        args[0] = &inputPtr;
        args[1] = &outputPtr;
        args[2] = &biasPtr;
        args[3] = &maskPtr;
        args[4] = &rows;
        args[5] = &cols;
        args[6] = &dropoutRate;
        args[7] = &scale;

        LaunchKernel2D(kernel, gridX, gridY, (uint)DefaultBlockSize, 1, args);
        return true;
    }

    #endregion


    #region Embedding Operations

    public unsafe void Embedding(IGpuBuffer indices, IGpuBuffer embeddingTable, IGpuBuffer output, int numIndices, int embeddingDim)
    {
        if (!_kernelCache.TryGetValue("embedding_forward", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: embedding_forward");

        using var _ = PushContext();
        uint grid = (uint)((numIndices + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr idxPtr = indices.Handle;
        IntPtr tablePtr = embeddingTable.Handle;
        IntPtr outputPtr = output.Handle;
        void** args = stackalloc void*[5];
        args[0] = &idxPtr;
        args[1] = &tablePtr;
        args[2] = &outputPtr;
        args[3] = &numIndices;
        args[4] = &embeddingDim;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    public unsafe void EmbeddingBackward(IGpuBuffer gradOutput, IGpuBuffer indices, IGpuBuffer gradEmbedding, int numIndices, int embeddingDim, int vocabSize)
    {
        using var _ = PushContext();
        IntPtr gradOutPtr = gradOutput.Handle;
        IntPtr idxPtr = indices.Handle;
        IntPtr gradEmbPtr = gradEmbedding.Handle;

        if (GpuDeterminism.IsActive)
        {
            // Issue #382: atomicAdd in embedding_backward is FP-non-deterministic
            // across runs. Route to the atomic-free deterministic variant — one
            // thread per (v, d) output cell scans numIndices in fixed order.
            if (!_kernelCache.TryGetValue("embedding_backward_deterministic", out var kernelD))
                throw new InvalidOperationException("CUDA kernel not found: embedding_backward_deterministic");

            void** argsD = stackalloc void*[6];
            argsD[0] = &gradOutPtr;
            argsD[1] = &idxPtr;
            argsD[2] = &gradEmbPtr;
            argsD[3] = &numIndices;
            argsD[4] = &embeddingDim;
            argsD[5] = &vocabSize;

            // Grid: ceil(embeddingDim / 16) x ceil(vocabSize / 16), Block: 16x16
            uint gridX = (uint)((embeddingDim + 15) / 16);
            uint gridY = (uint)((vocabSize + 15) / 16);
            LaunchKernel2D(kernelD, gridX, gridY, 16, 16, argsD);
            return;
        }

        if (!_kernelCache.TryGetValue("embedding_backward", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: embedding_backward");

        uint grid = (uint)((numIndices + DefaultBlockSize - 1) / DefaultBlockSize);
        void** args = stackalloc void*[5];
        args[0] = &gradOutPtr;
        args[1] = &idxPtr;
        args[2] = &gradEmbPtr;
        args[3] = &numIndices;
        args[4] = &embeddingDim;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    public IGpuBuffer AllocateIntBuffer(int size)
    {
        if (!IsAvailable)
            throw new InvalidOperationException("CUDA backend is not available.");

        if (size <= 0)
            throw new ArgumentOutOfRangeException(nameof(size), "Buffer size must be positive.");
        // Issue #285: per-allocation cap check before cuMemAlloc.
        GpuBufferSizeGuard.EnsureFits("CUDA", (long)size * sizeof(int), MaxBufferAllocBytes, DeviceName);

        using var _ = PushContext();
        ulong byteSize = (ulong)size * sizeof(int);
        CuBlasNative.CheckCudaResult(CuBlasNative.cuMemAlloc(out IntPtr devicePtr, byteSize), "cuMemAlloc(int)");
        CuBlasNative.CheckCudaResult(CuBlasNative.cuMemsetD32(devicePtr, 0, (ulong)size), "cuMemsetD32(int)");
        return new CudaGpuBuffer(_cudaContext, devicePtr, size);
    }

    public IGpuBuffer AllocateIntBuffer(int[] data)
    {
        if (!IsAvailable)
            throw new InvalidOperationException("CUDA backend is not available.");
        if (data is null)
            throw new ArgumentNullException(nameof(data));

        using var _ = PushContext();
        int size = data.Length;
        if (size <= 0)
            throw new ArgumentOutOfRangeException(nameof(data), "Buffer size must be positive.");
        ulong byteSize = (ulong)size * sizeof(int);
        // Issue #285: per-allocation cap check before cuMemAlloc.
        GpuBufferSizeGuard.EnsureFits("CUDA", (long)byteSize, MaxBufferAllocBytes, DeviceName);

        CuBlasNative.CheckCudaResult(CuBlasNative.cuMemAlloc(out IntPtr devicePtr, byteSize), "cuMemAlloc(int)");

        try
        {
            unsafe
            {
                fixed (int* src = data)
                {
                    CuBlasNative.CheckCudaResult(
                        CuBlasNative.cuMemcpyHtoD(devicePtr, (IntPtr)src, byteSize),
                        "cuMemcpyHtoD(int)");
                }
            }
        }
        catch
        {
            CuBlasNative.cuMemFree(devicePtr);
            throw;
        }

        return new CudaGpuBuffer(_cudaContext, devicePtr, size);
    }

    #endregion


    #region Attention Operations

    public void ScaledDotProductAttention(IGpuBuffer query, IGpuBuffer key, IGpuBuffer value,
        IGpuBuffer output, IGpuBuffer? attentionWeights, IGpuBuffer? mask,
        int batch, int numHeads, int seqLen, int headDim, float scale, bool isCausal)
    {
        // Attention: softmax(Q * K^T / sqrt(d_k)) * V
        using var _ = PushContext();
        int batchHeads = batch * numHeads;
        int qkSize = seqLen * seqLen;

        // Allocate temporary buffers
        using var scores = AllocateBuffer(batchHeads * qkSize);
        using var keyTransposed = AllocateBuffer(batchHeads * seqLen * headDim);

        // Transpose K: [batch*heads, seqLen, headDim] -> [batch*heads, headDim, seqLen]
        BatchedTranspose(key, keyTransposed, batchHeads, seqLen, headDim);

        // Q * K^T: [batch*heads, seqLen, headDim] x [batch*heads, headDim, seqLen] -> [batch*heads, seqLen, seqLen]
        BatchedGemm(query, keyTransposed, scores, seqLen, seqLen, headDim, batchHeads);

        // Scale by 1/sqrt(d_k)
        Scale(scores, scores, scale, batchHeads * qkSize);

        // Apply causal mask if needed
        if (isCausal && mask is not null)
        {
            Add(scores, mask, scores, batchHeads * qkSize);
        }

        // Softmax along the last dimension
        Softmax(scores, scores, batchHeads * seqLen, seqLen);

        // Copy attention weights if requested
        if (attentionWeights is not null)
        {
            Copy(scores, attentionWeights, batchHeads * qkSize);
        }

        // Multiply by V: [batch*heads, seqLen, seqLen] x [batch*heads, seqLen, headDim] -> [batch*heads, seqLen, headDim]
        BatchedGemm(scores, value, output, seqLen, headDim, seqLen, batchHeads);
    }

    public void ScaledDotProductAttentionBackward(IGpuBuffer gradOutput, IGpuBuffer query, IGpuBuffer key, IGpuBuffer value,
        IGpuBuffer attentionWeights, IGpuBuffer gradQuery, IGpuBuffer gradKey, IGpuBuffer gradValue,
        int batch, int numHeads, int seqLen, int headDim, float scale, bool isCausal)
    {
        using var _ = PushContext();
        int batchHeads = batch * numHeads;
        int qkSize = seqLen * seqLen;

        // Allocate temporary buffers
        using var gradScores = AllocateBuffer(batchHeads * qkSize);
        using var tempScores = AllocateBuffer(batchHeads * qkSize);
        using var attnTransposed = AllocateBuffer(batchHeads * qkSize);
        using var valueTransposed = AllocateBuffer(batchHeads * seqLen * headDim);
        using var gradScoresTransposed = AllocateBuffer(batchHeads * qkSize);

        // Transpose attention weights: [batch*heads, seqLen, seqLen]
        BatchedTranspose(attentionWeights, attnTransposed, batchHeads, seqLen, seqLen);

        // gradValue = attention_weights^T * gradOutput
        BatchedGemm(attnTransposed, gradOutput, gradValue, seqLen, headDim, seqLen, batchHeads);

        // Transpose V: [batch*heads, seqLen, headDim] -> [batch*heads, headDim, seqLen]
        BatchedTranspose(value, valueTransposed, batchHeads, seqLen, headDim);

        // gradScores = gradOutput * V^T
        BatchedGemm(gradOutput, valueTransposed, gradScores, seqLen, seqLen, headDim, batchHeads);

        // Softmax backward
        SoftmaxBackward(gradScores, attentionWeights, tempScores, batchHeads * seqLen, seqLen);

        // Scale
        Scale(tempScores, gradScores, scale, batchHeads * qkSize);

        // gradQuery = gradScores * K
        BatchedGemm(gradScores, key, gradQuery, seqLen, headDim, seqLen, batchHeads);

        // Transpose gradScores for gradKey computation
        BatchedTranspose(gradScores, gradScoresTransposed, batchHeads, seqLen, seqLen);

        // gradKey = gradScores^T * Q
        BatchedGemm(gradScoresTransposed, query, gradKey, seqLen, headDim, seqLen, batchHeads);
    }

    public void FlashAttention(IGpuBuffer query, IGpuBuffer key, IGpuBuffer value,
        IGpuBuffer output, IGpuBuffer? mask, int batch, int numHeads, int seqLen, int headDim, float scale, bool isCausal)
    {
        // Allocate temporary buffer for softmax stats (not returned but required by FlashAttentionV2)
        using var softmaxStats = AllocateBuffer(batch * numHeads * seqLen);

        // Use FlashAttentionV2 which is the proper GPU-accelerated implementation
        FlashAttentionV2(query, key, value, output, softmaxStats, batch, numHeads, seqLen, seqLen, headDim, scale, isCausal);
    }

    public unsafe void FlashAttentionV2(IGpuBuffer query, IGpuBuffer key, IGpuBuffer value,
        IGpuBuffer output, IGpuBuffer softmaxStats,
        int batch, int numHeads, int seqQ, int seqK, int headDim, float scale, bool isCausal,
        IGpuBuffer? attentionBias = null, int biasBatchStride = 0)
    {
        using var _ = PushContext();
        var kernel = _kernelCache["flash_attention_v2"];
        uint gridX = (uint)((seqQ + 31) / 32);
        uint gridY = (uint)(batch * numHeads);
        int causalFlag = isCausal ? 1 : 0;
        int hasBias = attentionBias is not null ? 1 : 0;
        IntPtr biasPtr = attentionBias is not null ? attentionBias.Handle : IntPtr.Zero;

        void** args = stackalloc void*[15];
        IntPtr qPtr = query.Handle;
        IntPtr kPtr = key.Handle;
        IntPtr vPtr = value.Handle;
        IntPtr oPtr = output.Handle;
        IntPtr sPtr = softmaxStats.Handle;
        args[0] = &qPtr;
        args[1] = &kPtr;
        args[2] = &vPtr;
        args[3] = &oPtr;
        args[4] = &sPtr;
        args[5] = &batch;
        args[6] = &numHeads;
        args[7] = &seqQ;
        args[8] = &seqK;
        args[9] = &headDim;
        args[10] = &scale;
        args[11] = &causalFlag;
        args[12] = &biasPtr;
        args[13] = &hasBias;
        args[14] = &biasBatchStride;

        // Shared memory: 2 * ATTN_BC(32) * headDim * sizeof(float)
        uint sharedBytes = (uint)(2 * 32 * headDim * sizeof(float));
        LaunchKernel2DWithSharedMem(kernel, gridX, gridY, 32, 1, sharedBytes, args);
    }

    public unsafe void FlashAttentionBackward(IGpuBuffer gradOutput, IGpuBuffer query, IGpuBuffer key, IGpuBuffer value,
        IGpuBuffer output, IGpuBuffer softmaxStats,
        IGpuBuffer gradQuery, IGpuBuffer gradKey, IGpuBuffer gradValue,
        int batch, int numHeads, int seqQ, int seqK, int headDim, float scale, bool isCausal,
        IGpuBuffer? attentionBias = null, int biasBatchStride = 0)
    {
        using var _ = PushContext();
        int causalFlag = isCausal ? 1 : 0;
        int hasBias = attentionBias is not null ? 1 : 0;
        IntPtr biasPtr = attentionBias is not null ? attentionBias.Handle : IntPtr.Zero;

        IntPtr goPtr = gradOutput.Handle;
        IntPtr qPtr = query.Handle;
        IntPtr kPtr = key.Handle;
        IntPtr vPtr = value.Handle;
        IntPtr oPtr = output.Handle;
        IntPtr sPtr = softmaxStats.Handle;
        IntPtr gqPtr = gradQuery.Handle;
        IntPtr gkPtr = gradKey.Handle;
        IntPtr gvPtr = gradValue.Handle;

        if (GpuDeterminism.IsActive)
        {
            // Issue #382: tiled flash_attention_backward uses atomicAdd for
            // gradKey/gradValue — non-deterministic across runs. Split into two
            // atomic-free kernels:
            //   gradq: per (bh, qi) writes gradQuery only
            //   gradkv: per (bh, ki, d) accumulates gradKey/gradValue with fixed qi order
            var kernelQ = _kernelCache["flash_attention_backward_gradq_deterministic"];
            void** argsQ = stackalloc void*[17];
            argsQ[0] = &goPtr;
            argsQ[1] = &qPtr;
            argsQ[2] = &kPtr;
            argsQ[3] = &vPtr;
            argsQ[4] = &oPtr;
            argsQ[5] = &sPtr;
            argsQ[6] = &gqPtr;
            argsQ[7] = &batch;
            argsQ[8] = &numHeads;
            argsQ[9] = &seqQ;
            argsQ[10] = &seqK;
            argsQ[11] = &headDim;
            argsQ[12] = &scale;
            argsQ[13] = &causalFlag;
            argsQ[14] = &biasPtr;
            argsQ[15] = &hasBias;
            argsQ[16] = &biasBatchStride;

            uint gridQX = (uint)((seqQ + 63) / 64);
            uint gridQY = (uint)(batch * numHeads);
            LaunchKernel2D(kernelQ, gridQX, gridQY, 64, 1, argsQ);

            var kernelKV = _kernelCache["flash_attention_backward_gradkv_deterministic"];
            void** argsKV = stackalloc void*[18];
            argsKV[0] = &goPtr;
            argsKV[1] = &qPtr;
            argsKV[2] = &kPtr;
            argsKV[3] = &vPtr;
            argsKV[4] = &oPtr;
            argsKV[5] = &sPtr;
            argsKV[6] = &gkPtr;
            argsKV[7] = &gvPtr;
            argsKV[8] = &batch;
            argsKV[9] = &numHeads;
            argsKV[10] = &seqQ;
            argsKV[11] = &seqK;
            argsKV[12] = &headDim;
            argsKV[13] = &scale;
            argsKV[14] = &causalFlag;
            argsKV[15] = &biasPtr;
            argsKV[16] = &hasBias;
            argsKV[17] = &biasBatchStride;

            uint gridKVX = (uint)((headDim + 15) / 16);
            uint gridKVY = (uint)((seqK + 3) / 4);
            uint gridKVZ = (uint)(batch * numHeads);
            LaunchKernel3D(kernelKV, gridKVX, gridKVY, gridKVZ, 16, 4, 1, argsKV);
            return;
        }

        var kernel = _kernelCache["flash_attention_backward"];
        uint gridX = (uint)((seqQ + 31) / 32);
        uint gridY = (uint)(batch * numHeads);

        void** args = stackalloc void*[19];
        args[0] = &goPtr;
        args[1] = &qPtr;
        args[2] = &kPtr;
        args[3] = &vPtr;
        args[4] = &oPtr;
        args[5] = &sPtr;
        args[6] = &gqPtr;
        args[7] = &gkPtr;
        args[8] = &gvPtr;
        args[9] = &batch;
        args[10] = &numHeads;
        args[11] = &seqQ;
        args[12] = &seqK;
        args[13] = &headDim;
        args[14] = &scale;
        args[15] = &causalFlag;
        args[16] = &biasPtr;
        args[17] = &hasBias;
        args[18] = &biasBatchStride;

        uint sharedBytes = (uint)(2 * 32 * headDim * sizeof(float));
        LaunchKernel2DWithSharedMem(kernel, gridX, gridY, 32, 1, sharedBytes, args);
    }

    public unsafe void GroupedQueryAttention(IGpuBuffer query, IGpuBuffer key, IGpuBuffer value,
        IGpuBuffer output, IGpuBuffer? attentionWeights,
        int batch, int numQHeads, int numKVHeads, int seqQ, int seqK, int headDim, float scale, bool isCausal)
    {
        using var _ = PushContext();
        var kernel = _kernelCache["grouped_query_attention"];
        uint gridX = (uint)((seqQ + 31) / 32);
        uint gridY = (uint)(batch * numQHeads);
        int queriesPerKV = numQHeads / numKVHeads;
        int causalFlag = isCausal ? 1 : 0;
        int storeWeights = attentionWeights != null ? 1 : 0;
        IntPtr wPtr = attentionWeights?.Handle ?? IntPtr.Zero;

        void** args = stackalloc void*[14];
        IntPtr qPtr = query.Handle;
        IntPtr kPtr = key.Handle;
        IntPtr vPtr = value.Handle;
        IntPtr oPtr = output.Handle;
        args[0] = &qPtr;
        args[1] = &kPtr;
        args[2] = &vPtr;
        args[3] = &oPtr;
        args[4] = &wPtr;
        args[5] = &batch;
        args[6] = &numQHeads;
        args[7] = &numKVHeads;
        args[8] = &queriesPerKV;
        args[9] = &seqQ;
        args[10] = &seqK;
        args[11] = &headDim;
        args[12] = &scale;
        args[13] = &causalFlag;

        void** args2 = stackalloc void*[15];
        for (int i = 0; i < 14; i++) args2[i] = args[i];
        args2[14] = &storeWeights;

        uint sharedBytes = (uint)(2 * 32 * headDim * sizeof(float));
        LaunchKernel2DWithSharedMem(kernel, gridX, gridY, 32, 1, sharedBytes, args2);
    }

    public unsafe void GroupedQueryAttentionBackward(IGpuBuffer gradOutput, IGpuBuffer query, IGpuBuffer key, IGpuBuffer value,
        IGpuBuffer attentionWeights,
        IGpuBuffer gradQuery, IGpuBuffer gradKey, IGpuBuffer gradValue,
        int batch, int numQHeads, int numKVHeads, int seqQ, int seqK, int headDim, float scale)
    {
        using var _ = PushContext();
        int queriesPerKV = numQHeads / numKVHeads;

        IntPtr goPtr = gradOutput.Handle;
        IntPtr qPtr = query.Handle;
        IntPtr kPtr = key.Handle;
        IntPtr vPtr = value.Handle;
        IntPtr wPtr = attentionWeights.Handle;
        IntPtr gqPtr = gradQuery.Handle;
        IntPtr gkPtr = gradKey.Handle;
        IntPtr gvPtr = gradValue.Handle;

        if (GpuDeterminism.IsActive)
        {
            // Issue #382: split tiled GQA backward into gradq (per bqh, qi) and
            // gradkv (per b, kvh, ki, d). No atomicAdd in either kernel.
            var kqGqa = _kernelCache["grouped_query_attention_backward_gradq_deterministic"];
            void** argsQg = stackalloc void*[14];
            argsQg[0] = &goPtr;
            argsQg[1] = &qPtr;
            argsQg[2] = &kPtr;
            argsQg[3] = &vPtr;
            argsQg[4] = &wPtr;
            argsQg[5] = &gqPtr;
            argsQg[6] = &batch;
            argsQg[7] = &numQHeads;
            argsQg[8] = &numKVHeads;
            argsQg[9] = &queriesPerKV;
            argsQg[10] = &seqQ;
            argsQg[11] = &seqK;
            argsQg[12] = &headDim;
            argsQg[13] = &scale;
            uint gqX = (uint)((seqQ + 63) / 64);
            uint gqY = (uint)(batch * numQHeads);
            LaunchKernel2D(kqGqa, gqX, gqY, 64, 1, argsQg);

            var kkvGqa = _kernelCache["grouped_query_attention_backward_gradkv_deterministic"];
            void** argsKVg = stackalloc void*[15];
            argsKVg[0] = &goPtr;
            argsKVg[1] = &qPtr;
            argsKVg[2] = &kPtr;
            argsKVg[3] = &vPtr;
            argsKVg[4] = &wPtr;
            argsKVg[5] = &gkPtr;
            argsKVg[6] = &gvPtr;
            argsKVg[7] = &batch;
            argsKVg[8] = &numQHeads;
            argsKVg[9] = &numKVHeads;
            argsKVg[10] = &queriesPerKV;
            argsKVg[11] = &seqQ;
            argsKVg[12] = &seqK;
            argsKVg[13] = &headDim;
            argsKVg[14] = &scale;
            uint gkvX = (uint)((headDim + 15) / 16);
            uint gkvY = (uint)((seqK + 3) / 4);
            uint gkvZ = (uint)(batch * numKVHeads);
            LaunchKernel3D(kkvGqa, gkvX, gkvY, gkvZ, 16, 4, 1, argsKVg);
            return;
        }

        var kernel = _kernelCache["grouped_query_attention_backward"];
        uint gridX = (uint)((seqQ + 31) / 32);
        uint gridY = (uint)(batch * numQHeads);

        void** args = stackalloc void*[16];
        args[0] = &goPtr;
        args[1] = &qPtr;
        args[2] = &kPtr;
        args[3] = &vPtr;
        args[4] = &wPtr;
        args[5] = &gqPtr;
        args[6] = &gkPtr;
        args[7] = &gvPtr;
        args[8] = &batch;
        args[9] = &numQHeads;
        args[10] = &numKVHeads;
        args[11] = &queriesPerKV;
        args[12] = &seqQ;
        args[13] = &seqK;
        args[14] = &headDim;
        args[15] = &scale;

        uint sharedBytes = (uint)(2 * 32 * headDim * sizeof(float));
        LaunchKernel2DWithSharedMem(kernel, gridX, gridY, 32, 1, sharedBytes, args);
    }

    #endregion


    #region Transpose and Reshape Operations

    public unsafe void Transpose(IGpuBuffer A, IGpuBuffer B, int rows, int cols)
    {
        if (!_kernelCache.TryGetValue("transpose_2d", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: transpose_2d");

        using var _ = PushContext();
        // Kernel reads `row = blockIdx.y * blockDim.y + threadIdx.y` and
        // `col = blockIdx.x * blockDim.x + threadIdx.x` — it needs a real
        // 2D launch. A 1D launch leaves blockIdx.y/threadIdx.y == 0, so
        // only row 0 ever gets written and rows ≥ 1 of B stay uninitialized
        // (this was the root cause of the prior "GPU Transpose kernel
        // diverges from CpuEngine reference" observation).
        const uint blockX = 16, blockY = 16;
        uint gridX = ((uint)cols + blockX - 1) / blockX;
        uint gridY = ((uint)rows + blockY - 1) / blockY;
        IntPtr aPtr = A.Handle;
        IntPtr bPtr = B.Handle;
        void** args = stackalloc void*[4];
        args[0] = &aPtr;
        args[1] = &bPtr;
        args[2] = &rows;
        args[3] = &cols;
        LaunchKernel2D(kernel, gridX, gridY, 1, blockX, blockY, args);
    }

    public unsafe void BatchedTranspose(IGpuBuffer A, IGpuBuffer B, int batch, int rows, int cols)
    {
        if (!_kernelCache.TryGetValue("batched_transpose", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: batched_transpose");

        using var _ = PushContext();
        // batched_transpose reads blockIdx.{x,y,z} and threadIdx.{x,y}.
        // Must be launched 3D so each batch slice's rows are reached;
        // the prior 1D dispatch only covered the slice-0 first-row corner.
        const uint blockX = 16, blockY = 16;
        uint gridX = ((uint)cols + blockX - 1) / blockX;
        uint gridY = ((uint)rows + blockY - 1) / blockY;
        uint gridZ = (uint)batch;
        IntPtr aPtr = A.Handle;
        IntPtr bPtr = B.Handle;
        void** args = stackalloc void*[5];
        args[0] = &aPtr;
        args[1] = &bPtr;
        args[2] = &batch;
        args[3] = &rows;
        args[4] = &cols;
        LaunchKernel3D(kernel, gridX, gridY, gridZ, blockX, blockY, 1, args);
    }

    public unsafe void Permute(IGpuBuffer input, IGpuBuffer output, int[] shape, int[] permutation)
    {
        // Handle common fast paths first
        if (shape.Length == 2 && permutation.Length == 2 && permutation[0] == 1 && permutation[1] == 0)
        {
            Transpose(input, output, shape[0], shape[1]);
            return;
        }

        if (shape.Length == 3 && permutation.Length == 3)
        {
            int batch = shape[0];
            int rows = shape[1];
            int cols = shape[2];

            // Handle (0, 2, 1) - transpose last two dimensions
            if (permutation[0] == 0 && permutation[1] == 2 && permutation[2] == 1)
            {
                BatchedTranspose(input, output, batch, rows, cols);
                return;
            }
        }

        // General permute using dedicated kernel
        if (!_kernelCache.TryGetValue("permute_general", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: permute_general");

        using var _ = PushContext();

        int ndims = shape.Length;
        int totalSize = 1;
        for (int i = 0; i < ndims; i++)
            totalSize *= shape[i];

        // Compute input strides (row-major)
        int[] inputStrides = new int[ndims];
        inputStrides[ndims - 1] = 1;
        for (int i = ndims - 2; i >= 0; i--)
            inputStrides[i] = inputStrides[i + 1] * shape[i + 1];

        // Compute output shape and strides after permutation
        int[] outputShape = new int[ndims];
        for (int i = 0; i < ndims; i++)
            outputShape[i] = shape[permutation[i]];

        int[] outputStrides = new int[ndims];
        outputStrides[ndims - 1] = 1;
        for (int i = ndims - 2; i >= 0; i--)
            outputStrides[i] = outputStrides[i + 1] * outputShape[i + 1];

        // Allocate device memory for strides and permutation
        using var inputStridesBuffer = AllocateIntBuffer(inputStrides);
        using var outputStridesBuffer = AllocateIntBuffer(outputStrides);
        using var permutationBuffer = AllocateIntBuffer(permutation);

        uint grid = (uint)((totalSize + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr inputPtr = input.Handle;
        IntPtr outputPtr = output.Handle;
        IntPtr inputStridesPtr = inputStridesBuffer.Handle;
        IntPtr outputStridesPtr = outputStridesBuffer.Handle;
        IntPtr permutationPtr = permutationBuffer.Handle;

        void** args = stackalloc void*[7];
        args[0] = &inputPtr;
        args[1] = &outputPtr;
        args[2] = &inputStridesPtr;
        args[3] = &outputStridesPtr;
        args[4] = &permutationPtr;
        args[5] = &ndims;
        args[6] = &totalSize;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }
    public void Copy(IGpuBuffer source, IGpuBuffer destination, int size)
    {
        if (!IsAvailable)
            throw new InvalidOperationException("CUDA backend is not available.");

        using var _ = PushContext();
        ulong byteSize = (ulong)size * sizeof(float);
        CuBlasNative.CheckCudaResult(
            CuBlasNative.cuMemcpyDtoD(destination.Handle, source.Handle, byteSize),
            "cuMemcpyDtoD");
    }

    public void Fill(IGpuBuffer buffer, float value, int size)
    {
        if (!IsAvailable)
            throw new InvalidOperationException("CUDA backend is not available.");

        using var _ = PushContext();
        // cuMemsetD32 sets 32-bit values (net471 compatible conversion)
        byte[] bytes = BitConverter.GetBytes(value);
        uint bits = BitConverter.ToUInt32(bytes, 0);
        CuBlasNative.CheckCudaResult(
            CuBlasNative.cuMemsetD32(buffer.Handle, bits, (ulong)size),
            "cuMemsetD32");
    }

    /// <inheritdoc/>
    public unsafe void Copy2DStrided(IGpuBuffer source, IGpuBuffer destination, int numRows, int srcCols, int destTotalCols, int destColOffset)
    {
        if (!IsAvailable)
            throw new InvalidOperationException("CUDA backend is not available.");

        if (!_kernelCache.TryGetValue("copy_2d_strided", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: copy_2d_strided");

        using var _ = PushContext();

        // Launch configuration: srcCols x numRows grid
        uint gridX = (uint)((srcCols + DefaultBlockSize - 1) / DefaultBlockSize);
        uint gridY = (uint)numRows;

        IntPtr srcPtr = source.Handle;
        IntPtr dstPtr = destination.Handle;

        void** args = stackalloc void*[6];
        args[0] = &srcPtr;
        args[1] = &dstPtr;
        args[2] = &numRows;
        args[3] = &srcCols;
        args[4] = &destTotalCols;
        args[5] = &destColOffset;

        CuBlasNative.CheckCudaResult(
            CudaNativeBindings.cuLaunchKernel(
                kernel,
                gridX, gridY, 1,
                (uint)DefaultBlockSize, 1, 1,
                0,
                _stream,
                (IntPtr)args,
                IntPtr.Zero),
            "cuLaunchKernel (copy_2d_strided)");
    }

    /// <inheritdoc/>
    public unsafe void NearestNeighborUpsample(IGpuBuffer input, IGpuBuffer output, int batchChannels, int height, int width, int scaleFactor)
    {
        if (!IsAvailable)
            throw new InvalidOperationException("CUDA backend is not available.");

        // Check for the kernel, if not available fall back to CPU implementation via memcpy pattern
        if (!_kernelCache.TryGetValue("nearest_neighbor_upsample", out var kernel))
        {
            // Fallback: Download, upsample on CPU, upload
            NearestNeighborUpsampleFallback(input, output, batchChannels, height, width, scaleFactor);
            return;
        }

        using var _ = PushContext();

        int outHeight = height * scaleFactor;
        int outWidth = width * scaleFactor;
        int outputSize = batchChannels * outHeight * outWidth;

        // Launch configuration: output elements / block size
        uint grid = (uint)((outputSize + DefaultBlockSize - 1) / DefaultBlockSize);

        IntPtr srcPtr = input.Handle;
        IntPtr dstPtr = output.Handle;

        void** args = stackalloc void*[7];
        args[0] = &srcPtr;
        args[1] = &dstPtr;
        args[2] = &batchChannels;
        args[3] = &height;
        args[4] = &width;
        args[5] = &scaleFactor;
        args[6] = &outputSize;

        CuBlasNative.CheckCudaResult(
            CudaNativeBindings.cuLaunchKernel(
                kernel,
                grid, 1, 1,
                (uint)DefaultBlockSize, 1, 1,
                0,
                _stream,
                (IntPtr)args,
                IntPtr.Zero),
            "cuLaunchKernel (nearest_neighbor_upsample)");
    }

    /// <summary>
    /// CPU fallback for nearest-neighbor upsampling when kernel is not available.
    /// </summary>
    private unsafe void NearestNeighborUpsampleFallback(IGpuBuffer input, IGpuBuffer output, int batchChannels, int height, int width, int scaleFactor)
    {
        int inputSize = batchChannels * height * width;
        int outHeight = height * scaleFactor;
        int outWidth = width * scaleFactor;
        int outputSize = batchChannels * outHeight * outWidth;

        // Download input using existing method
        var inputData = new float[inputSize];
        DownloadBuffer(input, inputData);

        // Perform CPU upsampling
        var outputData = new float[outputSize];
        for (int bc = 0; bc < batchChannels; bc++)
        {
            for (int oh = 0; oh < outHeight; oh++)
            {
                for (int ow = 0; ow < outWidth; ow++)
                {
                    int ih = oh / scaleFactor;
                    int iw = ow / scaleFactor;
                    int inputIdx = bc * height * width + ih * width + iw;
                    int outputIdx = bc * outHeight * outWidth + oh * outWidth + ow;
                    outputData[outputIdx] = inputData[inputIdx];
                }
            }
        }

        // Upload output using CUDA memory copy
        using var _ = PushContext();
        ulong byteSize = (ulong)outputSize * sizeof(float);
        fixed (float* src = outputData)
        {
            CuBlasNative.CheckCudaResult(
                CuBlasNative.cuMemcpyHtoD(output.Handle, (IntPtr)src, byteSize),
                "cuMemcpyHtoD (upsample fallback)");
        }
    }

    /// <inheritdoc/>
    public unsafe void NearestNeighborUpsampleBackward(IGpuBuffer gradOutput, IGpuBuffer gradInput, int batchChannels, int height, int width, int scaleFactor)
    {
        if (!IsAvailable)
            throw new InvalidOperationException("CUDA backend is not available.");

        // Use checked arithmetic to detect overflow in dimension calculations
        long inputSizeLong;
        try
        {
            checked
            {
                inputSizeLong = (long)batchChannels * height * width;
            }
        }
        catch (OverflowException)
        {
            throw new OverflowException($"NearestNeighborUpsampleBackward: Input dimensions overflow (batchChannels={batchChannels}, height={height}, width={width}).");
        }

        // Validate that computed count fits into int (required for kernel parameters)
        if (inputSizeLong > int.MaxValue)
        {
            throw new InvalidOperationException($"NearestNeighborUpsampleBackward: Input size {inputSizeLong} exceeds int.MaxValue.");
        }
        int inputSize = (int)inputSizeLong;

        if (!_kernelCache.TryGetValue("nearest_neighbor_upsample_backward", out var kernel))
        {
            // Fallback: CPU implementation
            NearestNeighborUpsampleBackwardFallback(gradOutput, gradInput, batchChannels, height, width, scaleFactor);
            return;
        }

        using var _ = PushContext();

        // Kernel iterates over input elements, accumulating from scaleFactor x scaleFactor output regions
        // No zeroing needed - kernel writes directly (not +=)
        // Use long arithmetic for grid calculation, then validate for uint range
        long gridLong = (inputSizeLong + DefaultBlockSize - 1) / DefaultBlockSize;
        if (gridLong > uint.MaxValue)
        {
            throw new InvalidOperationException($"NearestNeighborUpsampleBackward: Grid size {gridLong} exceeds uint.MaxValue.");
        }
        uint grid = (uint)gridLong;

        IntPtr gradOutPtr = gradOutput.Handle;
        IntPtr gradInPtr = gradInput.Handle;

        void** args = stackalloc void*[7];
        args[0] = &gradOutPtr;
        args[1] = &gradInPtr;
        args[2] = &batchChannels;
        args[3] = &height;
        args[4] = &width;
        args[5] = &scaleFactor;
        args[6] = &inputSize;

        CuBlasNative.CheckCudaResult(
            CudaNativeBindings.cuLaunchKernel(
                kernel,
                grid, 1, 1,
                (uint)DefaultBlockSize, 1, 1,
                0,
                _stream,
                (IntPtr)args,
                IntPtr.Zero),
            "cuLaunchKernel (nearest_neighbor_upsample_backward)");
    }

    /// <summary>
    /// CPU fallback for nearest-neighbor upsampling backward when kernel is not available.
    /// </summary>
    private unsafe void NearestNeighborUpsampleBackwardFallback(IGpuBuffer gradOutput, IGpuBuffer gradInput, int batchChannels, int height, int width, int scaleFactor)
    {
        // Use checked arithmetic for all dimension calculations
        int outHeight, outWidth;
        long outputSizeLong, inputSizeLong;
        try
        {
            checked
            {
                outHeight = height * scaleFactor;
                outWidth = width * scaleFactor;
                outputSizeLong = (long)batchChannels * outHeight * outWidth;
                inputSizeLong = (long)batchChannels * height * width;
            }
        }
        catch (OverflowException)
        {
            throw new OverflowException($"NearestNeighborUpsampleBackwardFallback: Dimension overflow (batchChannels={batchChannels}, height={height}, width={width}, scaleFactor={scaleFactor}).");
        }

        // Validate sizes fit in int (required for array indexing)
        if (outputSizeLong > int.MaxValue)
        {
            throw new InvalidOperationException($"NearestNeighborUpsampleBackwardFallback: Output size {outputSizeLong} exceeds int.MaxValue.");
        }
        if (inputSizeLong > int.MaxValue)
        {
            throw new InvalidOperationException($"NearestNeighborUpsampleBackwardFallback: Input size {inputSizeLong} exceeds int.MaxValue.");
        }
        int outputSize = (int)outputSizeLong;
        int inputSize = (int)inputSizeLong;

        // Download gradOutput
        var gradOutData = new float[outputSize];
        DownloadBuffer(gradOutput, gradOutData);

        // Accumulate gradients on CPU
        var gradInData = new float[inputSize];
        for (int bc = 0; bc < batchChannels; bc++)
        {
            for (int oh = 0; oh < outHeight; oh++)
            {
                for (int ow = 0; ow < outWidth; ow++)
                {
                    int ih = oh / scaleFactor;
                    int iw = ow / scaleFactor;
                    int inputIdx = bc * height * width + ih * width + iw;
                    int outputIdx = bc * outHeight * outWidth + oh * outWidth + ow;
                    gradInData[inputIdx] += gradOutData[outputIdx];
                }
            }
        }

        // Upload gradInput - compute byte size from checked long element count
        using var _ = PushContext();
        ulong byteSize;
        try
        {
            checked
            {
                byteSize = (ulong)(inputSizeLong * sizeof(float));
            }
        }
        catch (OverflowException)
        {
            throw new OverflowException($"NearestNeighborUpsampleBackwardFallback: Byte size overflow for {inputSizeLong} elements.");
        }
        fixed (float* src = gradInData)
        {
            CuBlasNative.CheckCudaResult(
                CuBlasNative.cuMemcpyHtoD(gradInput.Handle, (IntPtr)src, byteSize),
                "cuMemcpyHtoD (upsample backward fallback)");
        }
    }

    #endregion


    #region Activation Gradient Operations

    public unsafe void ReluBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradInput, int size)
    {
        if (!_kernelCache.TryGetValue("relu_backward", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: relu_backward");

        using var _ = PushContext();
        uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr gradOutPtr = gradOutput.Handle;
        IntPtr inputPtr = input.Handle;
        IntPtr gradInPtr = gradInput.Handle;
        int n = size;
        void** args = stackalloc void*[4];
        args[0] = &gradOutPtr;
        args[1] = &inputPtr;
        args[2] = &gradInPtr;
        args[3] = &n;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    public unsafe void SigmoidBackward(IGpuBuffer gradOutput, IGpuBuffer output, IGpuBuffer gradInput, int size)
    {
        if (!_kernelCache.TryGetValue("sigmoid_backward", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: sigmoid_backward");

        using var _ = PushContext();
        uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr gradOutPtr = gradOutput.Handle;
        IntPtr outPtr = output.Handle;
        IntPtr gradInPtr = gradInput.Handle;
        int n = size;
        void** args = stackalloc void*[4];
        args[0] = &gradOutPtr;
        args[1] = &outPtr;
        args[2] = &gradInPtr;
        args[3] = &n;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    public unsafe void TanhBackward(IGpuBuffer gradOutput, IGpuBuffer output, IGpuBuffer gradInput, int size)
    {
        if (!_kernelCache.TryGetValue("tanh_backward", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: tanh_backward");

        using var _ = PushContext();
        uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr gradOutPtr = gradOutput.Handle;
        IntPtr outPtr = output.Handle;
        IntPtr gradInPtr = gradInput.Handle;
        int n = size;
        void** args = stackalloc void*[4];
        args[0] = &gradOutPtr;
        args[1] = &outPtr;
        args[2] = &gradInPtr;
        args[3] = &n;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    public unsafe void GeluBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradInput, int size)
    {
        if (!_kernelCache.TryGetValue("gelu_backward", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: gelu_backward");

        using var _ = PushContext();
        uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr gradOutPtr = gradOutput.Handle;
        IntPtr inputPtr = input.Handle;
        IntPtr gradInPtr = gradInput.Handle;
        int n = size;
        void** args = stackalloc void*[4];
        args[0] = &gradOutPtr;
        args[1] = &inputPtr;
        args[2] = &gradInPtr;
        args[3] = &n;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    public unsafe void SoftmaxBackward(IGpuBuffer gradOutput, IGpuBuffer output, IGpuBuffer gradInput, int batchSize, int features)
    {
        if (!_kernelCache.TryGetValue("softmax_backward", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: softmax_backward");

        using var _ = PushContext();
        uint grid = (uint)batchSize;
        IntPtr gradOutPtr = gradOutput.Handle;
        IntPtr outPtr = output.Handle;
        IntPtr gradInPtr = gradInput.Handle;
        int batch = batchSize;
        int feat = features;
        void** args = stackalloc void*[5];
        args[0] = &gradOutPtr;
        args[1] = &outPtr;
        args[2] = &gradInPtr;
        args[3] = &batch;
        args[4] = &feat;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    public unsafe void LeakyRelu(IGpuBuffer A, IGpuBuffer B, float alpha, int size)
    {
        if (!_kernelCache.TryGetValue("leaky_relu", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: leaky_relu");

        using var _ = PushContext();
        uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr inputPtr = A.Handle;
        IntPtr outputPtr = B.Handle;
        float alphaVal = alpha;
        int n = size;
        void** args = stackalloc void*[4];
        args[0] = &inputPtr;
        args[1] = &outputPtr;
        args[2] = &alphaVal;
        args[3] = &n;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    public unsafe void LeakyReluBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradInput, float alpha, int size)
    {
        if (!_kernelCache.TryGetValue("leaky_relu_backward", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: leaky_relu_backward");

        using var _ = PushContext();
        uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr gradOutPtr = gradOutput.Handle;
        IntPtr inputPtr = input.Handle;
        IntPtr gradInPtr = gradInput.Handle;
        float alphaVal = alpha;
        int n = size;
        void** args = stackalloc void*[5];
        args[0] = &gradOutPtr;
        args[1] = &inputPtr;
        args[2] = &gradInPtr;
        args[3] = &alphaVal;
        args[4] = &n;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    public unsafe void Elu(IGpuBuffer A, IGpuBuffer B, float alpha, int size)
    {
        if (!_kernelCache.TryGetValue("elu", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: elu");

        using var _ = PushContext();
        uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr inputPtr = A.Handle;
        IntPtr outputPtr = B.Handle;
        float alphaVal = alpha;
        int n = size;
        void** args = stackalloc void*[4];
        args[0] = &inputPtr;
        args[1] = &outputPtr;
        args[2] = &alphaVal;
        args[3] = &n;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    public unsafe void EluBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer output, IGpuBuffer gradInput, float alpha, int size)
    {
        if (!_kernelCache.TryGetValue("elu_backward", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: elu_backward");

        using var _ = PushContext();
        uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr gradOutPtr = gradOutput.Handle;
        IntPtr inputPtr = input.Handle;
        IntPtr outPtr = output.Handle;
        IntPtr gradInPtr = gradInput.Handle;
        float alphaVal = alpha;
        int n = size;
        void** args = stackalloc void*[6];
        args[0] = &gradOutPtr;
        args[1] = &inputPtr;
        args[2] = &outPtr;
        args[3] = &gradInPtr;
        args[4] = &alphaVal;
        args[5] = &n;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    public void Swish(IGpuBuffer A, IGpuBuffer B, int size)
    {
        LaunchUnaryKernel("swish", A, B, size);
    }

    public unsafe void SwishBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradInput, int size)
    {
        if (!_kernelCache.TryGetValue("swish_backward", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: swish_backward");

        using var _ = PushContext();
        uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr gradOutPtr = gradOutput.Handle;
        IntPtr inputPtr = input.Handle;
        IntPtr gradInPtr = gradInput.Handle;
        int n = size;
        void** args = stackalloc void*[4];
        args[0] = &gradOutPtr;
        args[1] = &inputPtr;
        args[2] = &gradInPtr;
        args[3] = &n;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    public void Silu(IGpuBuffer A, IGpuBuffer B, int size)
    {
        LaunchUnaryKernel("silu", A, B, size);
    }

    public void Mish(IGpuBuffer A, IGpuBuffer B, int size)
    {
        LaunchUnaryKernel("mish", A, B, size);
    }

    public void Softplus(IGpuBuffer A, IGpuBuffer B, int size)
    {
        LaunchUnaryKernel("softplus", A, B, size);
    }

    public void Hardswish(IGpuBuffer A, IGpuBuffer B, int size)
    {
        LaunchUnaryKernel("hardswish", A, B, size);
    }

    public unsafe void Selu(IGpuBuffer A, IGpuBuffer B, float alpha, float scale, int size)
    {
        if (!_kernelCache.TryGetValue("selu", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: selu");

        using var _ = PushContext();
        uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr aPtr = A.Handle;
        IntPtr bPtr = B.Handle;
        void** args = stackalloc void*[5];
        args[0] = &aPtr;
        args[1] = &bPtr;
        args[2] = &alpha;
        args[3] = &scale;
        args[4] = &size;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    public void Hardsigmoid(IGpuBuffer A, IGpuBuffer B, int size)
    {
        LaunchUnaryKernel("hardsigmoid", A, B, size);
    }

    public unsafe void Hardtanh(IGpuBuffer A, IGpuBuffer B, float minVal, float maxVal, int size)
    {
        if (!_kernelCache.TryGetValue("hardtanh", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: hardtanh");

        using var _ = PushContext();
        uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr aPtr = A.Handle;
        IntPtr bPtr = B.Handle;
        void** args = stackalloc void*[5];
        args[0] = &aPtr;
        args[1] = &bPtr;
        args[2] = &minVal;
        args[3] = &maxVal;
        args[4] = &size;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    // SiLU backward uses SwishBackward since they're mathematically equivalent
    public void SiluBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradInput, int size)
    {
        SwishBackward(gradOutput, input, gradInput, size);
    }

    public unsafe void MishBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradInput, int size)
    {
        if (!_kernelCache.TryGetValue("mish_backward", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: mish_backward");

        using var _ = PushContext();
        uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr gradOutPtr = gradOutput.Handle;
        IntPtr inputPtr = input.Handle;
        IntPtr gradInPtr = gradInput.Handle;
        int n = size;
        void** args = stackalloc void*[4];
        args[0] = &gradOutPtr;
        args[1] = &inputPtr;
        args[2] = &gradInPtr;
        args[3] = &n;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    public unsafe void SoftplusBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradInput, int size)
    {
        if (!_kernelCache.TryGetValue("softplus_backward", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: softplus_backward");

        using var _ = PushContext();
        uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr gradOutPtr = gradOutput.Handle;
        IntPtr inputPtr = input.Handle;
        IntPtr gradInPtr = gradInput.Handle;
        int n = size;
        void** args = stackalloc void*[4];
        args[0] = &gradOutPtr;
        args[1] = &inputPtr;
        args[2] = &gradInPtr;
        args[3] = &n;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    public unsafe void HardswishBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradInput, int size)
    {
        if (!_kernelCache.TryGetValue("hardswish_backward", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: hardswish_backward");

        using var _ = PushContext();
        uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr gradOutPtr = gradOutput.Handle;
        IntPtr inputPtr = input.Handle;
        IntPtr gradInPtr = gradInput.Handle;
        int n = size;
        void** args = stackalloc void*[4];
        args[0] = &gradOutPtr;
        args[1] = &inputPtr;
        args[2] = &gradInPtr;
        args[3] = &n;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    public unsafe void SeluBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradInput, float alpha, float scale, int size)
    {
        if (!_kernelCache.TryGetValue("selu_backward", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: selu_backward");

        using var _ = PushContext();
        uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr gradOutPtr = gradOutput.Handle;
        IntPtr inputPtr = input.Handle;
        IntPtr gradInPtr = gradInput.Handle;
        int n = size;
        void** args = stackalloc void*[6];
        args[0] = &gradOutPtr;
        args[1] = &inputPtr;
        args[2] = &gradInPtr;
        args[3] = &alpha;
        args[4] = &scale;
        args[5] = &n;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    public unsafe void HardsigmoidBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradInput, int size)
    {
        if (!_kernelCache.TryGetValue("hardsigmoid_backward", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: hardsigmoid_backward");

        using var _ = PushContext();
        uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr gradOutPtr = gradOutput.Handle;
        IntPtr inputPtr = input.Handle;
        IntPtr gradInPtr = gradInput.Handle;
        int n = size;
        void** args = stackalloc void*[4];
        args[0] = &gradOutPtr;
        args[1] = &inputPtr;
        args[2] = &gradInPtr;
        args[3] = &n;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    public unsafe void HardtanhBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradInput, float minVal, float maxVal, int size)
    {
        if (!_kernelCache.TryGetValue("hardtanh_backward", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: hardtanh_backward");

        using var _ = PushContext();
        uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr gradOutPtr = gradOutput.Handle;
        IntPtr inputPtr = input.Handle;
        IntPtr gradInPtr = gradInput.Handle;
        int n = size;
        void** args = stackalloc void*[6];
        args[0] = &gradOutPtr;
        args[1] = &inputPtr;
        args[2] = &gradInPtr;
        args[3] = &minVal;
        args[4] = &maxVal;
        args[5] = &n;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    public unsafe void Relu6(IGpuBuffer A, IGpuBuffer B, int size)
    {
        if (!_kernelCache.TryGetValue("relu6", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: relu6");
        using var _ = PushContext();
        uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr aPtr = A.Handle; IntPtr bPtr = B.Handle; int n = size;
        void** args = stackalloc void*[3]; args[0] = &aPtr; args[1] = &bPtr; args[2] = &n;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    public unsafe void Relu6Backward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradInput, int size)
    {
        if (!_kernelCache.TryGetValue("relu6_backward", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: relu6_backward");
        using var _ = PushContext();
        uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr goPtr = gradOutput.Handle; IntPtr iPtr = input.Handle; IntPtr giPtr = gradInput.Handle; int n = size;
        void** args = stackalloc void*[4]; args[0] = &goPtr; args[1] = &iPtr; args[2] = &giPtr; args[3] = &n;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    public unsafe void PRelu(IGpuBuffer input, IGpuBuffer alpha, IGpuBuffer output, int size, int alphaSize)
    {
        if (!_kernelCache.TryGetValue("prelu", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: prelu");
        using var _ = PushContext();
        uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr iPtr = input.Handle; IntPtr aPtr = alpha.Handle; IntPtr oPtr = output.Handle; int n = size; int aSize = alphaSize;
        void** args = stackalloc void*[5]; args[0] = &iPtr; args[1] = &aPtr; args[2] = &oPtr; args[3] = &n; args[4] = &aSize;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    public unsafe void PReluBackwardInput(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer alpha, IGpuBuffer gradInput, int size, int alphaSize)
    {
        if (!_kernelCache.TryGetValue("prelu_backward_input", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: prelu_backward_input");
        using var _ = PushContext();
        uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr goPtr = gradOutput.Handle; IntPtr iPtr = input.Handle; IntPtr aPtr = alpha.Handle; IntPtr giPtr = gradInput.Handle; int n = size; int aSize = alphaSize;
        void** args = stackalloc void*[6]; args[0] = &goPtr; args[1] = &iPtr; args[2] = &aPtr; args[3] = &giPtr; args[4] = &n; args[5] = &aSize;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    public unsafe void PReluBackwardAlpha(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradAlpha, int size, int alphaSize)
    {
        using var _ = PushContext();
        IntPtr goPtr = gradOutput.Handle; IntPtr iPtr = input.Handle; IntPtr gaPtr = gradAlpha.Handle; int n = size; int aSize = alphaSize;
        void** args = stackalloc void*[5]; args[0] = &goPtr; args[1] = &iPtr; args[2] = &gaPtr; args[3] = &n; args[4] = &aSize;

        if (GpuDeterminism.IsActive)
        {
            // Issue #382: per-alpha-channel thread scans size in strided order.
            if (!_kernelCache.TryGetValue("prelu_backward_alpha_deterministic", out var kernelD))
                throw new InvalidOperationException("CUDA kernel not found: prelu_backward_alpha_deterministic");
            uint gridD = (uint)((alphaSize + DefaultBlockSize - 1) / DefaultBlockSize);
            LaunchKernel(kernelD, gridD, DefaultBlockSize, args);
            return;
        }

        if (!_kernelCache.TryGetValue("prelu_backward_alpha", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: prelu_backward_alpha");
        uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    public unsafe void RRelu(IGpuBuffer input, IGpuBuffer noise, IGpuBuffer output, int size)
    {
        if (!_kernelCache.TryGetValue("rrelu", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: rrelu");
        using var _ = PushContext();
        uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr iPtr = input.Handle; IntPtr nPtr = noise.Handle; IntPtr oPtr = output.Handle; int n = size;
        void** args = stackalloc void*[4]; args[0] = &iPtr; args[1] = &nPtr; args[2] = &oPtr; args[3] = &n;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    public unsafe void RReluBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer noise, IGpuBuffer gradInput, int size)
    {
        if (!_kernelCache.TryGetValue("rrelu_backward", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: rrelu_backward");
        using var _ = PushContext();
        uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr goPtr = gradOutput.Handle; IntPtr iPtr = input.Handle; IntPtr nPtr = noise.Handle; IntPtr giPtr = gradInput.Handle; int n = size;
        void** args = stackalloc void*[5]; args[0] = &goPtr; args[1] = &iPtr; args[2] = &nPtr; args[3] = &giPtr; args[4] = &n;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    public unsafe void Threshold(IGpuBuffer input, IGpuBuffer output, float threshold, float value, int size)
    {
        if (!_kernelCache.TryGetValue("threshold_forward", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: threshold_forward");
        using var _ = PushContext();
        uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr iPtr = input.Handle; IntPtr oPtr = output.Handle; int n = size;
        void** args = stackalloc void*[5]; args[0] = &iPtr; args[1] = &oPtr; args[2] = &threshold; args[3] = &value; args[4] = &n;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    public unsafe void ThresholdBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradInput, float threshold, int size)
    {
        if (!_kernelCache.TryGetValue("threshold_backward", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: threshold_backward");
        using var _ = PushContext();
        uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr goPtr = gradOutput.Handle; IntPtr iPtr = input.Handle; IntPtr giPtr = gradInput.Handle; int n = size;
        void** args = stackalloc void*[5]; args[0] = &goPtr; args[1] = &iPtr; args[2] = &giPtr; args[3] = &threshold; args[4] = &n;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    public unsafe void ReciprocalBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradInput, int size)
    {
        if (!_kernelCache.TryGetValue("reciprocal_backward", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: reciprocal_backward");
        using var _ = PushContext();
        uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr goPtr = gradOutput.Handle; IntPtr iPtr = input.Handle; IntPtr giPtr = gradInput.Handle; int n = size;
        void** args = stackalloc void*[4]; args[0] = &goPtr; args[1] = &iPtr; args[2] = &giPtr; args[3] = &n;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    public unsafe void VarBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer mean, IGpuBuffer gradInput, int outerSize, int reduceSize)
    {
        if (!_kernelCache.TryGetValue("var_backward", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: var_backward");
        using var _ = PushContext();
        int total = outerSize * reduceSize;
        uint grid = (uint)((total + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr goPtr = gradOutput.Handle, iPtr = input.Handle, mPtr = mean.Handle, giPtr = gradInput.Handle;
        int os = outerSize, rs = reduceSize;
        void** args = stackalloc void*[6];
        args[0] = &goPtr; args[1] = &iPtr; args[2] = &mPtr; args[3] = &giPtr; args[4] = &os; args[5] = &rs;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    public unsafe void StdBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer mean, IGpuBuffer std, IGpuBuffer gradInput, int outerSize, int reduceSize)
    {
        if (!_kernelCache.TryGetValue("std_backward", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: std_backward");
        using var _ = PushContext();
        int total = outerSize * reduceSize;
        uint grid = (uint)((total + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr goPtr = gradOutput.Handle, iPtr = input.Handle, mPtr = mean.Handle, sPtr = std.Handle, giPtr = gradInput.Handle;
        int os = outerSize, rs = reduceSize;
        void** args = stackalloc void*[7];
        args[0] = &goPtr; args[1] = &iPtr; args[2] = &mPtr; args[3] = &sPtr; args[4] = &giPtr; args[5] = &os; args[6] = &rs;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    public unsafe void MaskedFillBackward(IGpuBuffer gradOutput, IGpuBuffer mask, IGpuBuffer gradInput, int size)
    {
        if (!_kernelCache.TryGetValue("masked_fill_backward", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: masked_fill_backward");
        using var _ = PushContext();
        uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr goPtr = gradOutput.Handle, mPtr = mask.Handle, giPtr = gradInput.Handle;
        int n = size;
        void** args = stackalloc void*[4];
        args[0] = &goPtr; args[1] = &mPtr; args[2] = &giPtr; args[3] = &n;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    public unsafe void WhereBackward(IGpuBuffer gradOutput, IGpuBuffer condition, IGpuBuffer gradX, IGpuBuffer gradY, int size)
    {
        if (!_kernelCache.TryGetValue("where_backward", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: where_backward");
        using var _ = PushContext();
        uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr goPtr = gradOutput.Handle, cPtr = condition.Handle, gxPtr = gradX.Handle, gyPtr = gradY.Handle;
        int n = size;
        void** args = stackalloc void*[5];
        args[0] = &goPtr; args[1] = &cPtr; args[2] = &gxPtr; args[3] = &gyPtr; args[4] = &n;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    public unsafe void NormBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer norm, IGpuBuffer gradInput, int outerSize, int reduceSize)
    {
        if (!_kernelCache.TryGetValue("norm_backward", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: norm_backward");
        using var _ = PushContext();
        int total = outerSize * reduceSize;
        uint grid = (uint)((total + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr goPtr = gradOutput.Handle, iPtr = input.Handle, nPtr = norm.Handle, giPtr = gradInput.Handle;
        int os = outerSize, rs = reduceSize;
        void** args = stackalloc void*[6];
        args[0] = &goPtr; args[1] = &iPtr; args[2] = &nPtr; args[3] = &giPtr; args[4] = &os; args[5] = &rs;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    public unsafe void LogSumExpBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer lse, IGpuBuffer gradInput, int outerSize, int reduceSize)
    {
        if (!_kernelCache.TryGetValue("logsumexp_backward", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: logsumexp_backward");
        using var _ = PushContext();
        int total = outerSize * reduceSize;
        uint grid = (uint)((total + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr goPtr = gradOutput.Handle, iPtr = input.Handle, lPtr = lse.Handle, giPtr = gradInput.Handle;
        int os = outerSize, rs = reduceSize;
        void** args = stackalloc void*[6];
        args[0] = &goPtr; args[1] = &iPtr; args[2] = &lPtr; args[3] = &giPtr; args[4] = &os; args[5] = &rs;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    public unsafe void AvgPool1D(IGpuBuffer input, IGpuBuffer output, int batch, int channels, int inLength, int outLength, int kernelSize, int stride)
    {
        if (!_kernelCache.TryGetValue("avg_pool1d", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: avg_pool1d");
        using var _ = PushContext();
        long totalL = (long)batch * channels * outLength; int total = checked((int)totalL);
        uint grid = (uint)((total + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr iPtr = input.Handle, oPtr = output.Handle;
        void** args = stackalloc void*[8];
        args[0] = &iPtr; args[1] = &oPtr; args[2] = &batch; args[3] = &channels;
        args[4] = &inLength; args[5] = &outLength; args[6] = &kernelSize; args[7] = &stride;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    public unsafe void MaxPool1D(IGpuBuffer input, IGpuBuffer output, int batch, int channels, int inLength, int outLength, int kernelSize, int stride)
    {
        if (!_kernelCache.TryGetValue("max_pool1d", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: max_pool1d");
        using var _ = PushContext();
        long totalL = (long)batch * channels * outLength; int total = checked((int)totalL);
        uint grid = (uint)((total + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr iPtr = input.Handle, oPtr = output.Handle;
        void** args = stackalloc void*[8];
        args[0] = &iPtr; args[1] = &oPtr; args[2] = &batch; args[3] = &channels;
        args[4] = &inLength; args[5] = &outLength; args[6] = &kernelSize; args[7] = &stride;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    public unsafe void BilinearUpsample2D(IGpuBuffer input, IGpuBuffer output, int batch, int channels, int inH, int inW, int outH, int outW)
    {
        if (!_kernelCache.TryGetValue("bilinear_upsample2d", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: bilinear_upsample2d");
        using var _ = PushContext();
        long totalL2 = (long)batch * channels * outH * outW; int total = checked((int)totalL2);
        uint grid = (uint)((total + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr iPtr = input.Handle, oPtr = output.Handle;
        void** args = stackalloc void*[8];
        args[0] = &iPtr; args[1] = &oPtr; args[2] = &batch; args[3] = &channels;
        args[4] = &inH; args[5] = &inW; args[6] = &outH; args[7] = &outW;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    public unsafe void ScatterMean(IGpuBuffer source, IGpuBuffer indices, IGpuBuffer output, IGpuBuffer counts, int sourceSize, int outputSize, int featureSize)
    {
        using var _ = PushContext();
        // Initialize output and counts to zero before accumulation
        Fill(output, 0f, outputSize * featureSize);
        Fill(counts, 0f, outputSize);
        IntPtr sPtr = source.Handle, idxPtr = indices.Handle, oPtr = output.Handle, cPtr = counts.Handle;

        if (GpuDeterminism.IsActive)
        {
            // Issue #382: route to atomic-free variant — per (dstRow, col) output cell.
            if (!_kernelCache.TryGetValue("scatter_mean_activation_deterministic", out var kernelD))
                throw new InvalidOperationException("CUDA kernel not found: scatter_mean_activation_deterministic");
            void** argsD = stackalloc void*[7];
            argsD[0] = &sPtr; argsD[1] = &idxPtr; argsD[2] = &oPtr; argsD[3] = &cPtr;
            argsD[4] = &sourceSize; argsD[5] = &outputSize; argsD[6] = &featureSize;
            const int blockSize = 16;
            uint gridDX = (uint)((featureSize + blockSize - 1) / blockSize);
            uint gridDY = (uint)((outputSize + blockSize - 1) / blockSize);
            LaunchKernel2D(kernelD, gridDX, gridDY, (uint)blockSize, (uint)blockSize, argsD);
        }
        else
        {
            if (!_kernelCache.TryGetValue("scatter_mean", out var scatterKernel))
                throw new InvalidOperationException("CUDA kernel not found: scatter_mean");
            uint grid1 = (uint)((sourceSize + DefaultBlockSize - 1) / DefaultBlockSize);
            void** args1 = stackalloc void*[6];
            args1[0] = &sPtr; args1[1] = &idxPtr; args1[2] = &oPtr; args1[3] = &cPtr;
            args1[4] = &sourceSize; args1[5] = &featureSize;
            LaunchKernel(scatterKernel, grid1, DefaultBlockSize, args1);
        }

        // Step 2: divide by counts (deterministic by construction — one thread per cell).
        // scatter_mean_divide expects flat output element count (kernel guards
        // `idx < outputSize` and indexes `output[idx]`), so pass outputSize *
        // featureSize as the flat element count. Pre-existing bug surfaced by
        // PR #390 review: prior code passed row count and dispatched only
        // `outputSize` work-items, normalizing just the first row.
        if (!_kernelCache.TryGetValue("scatter_mean_divide", out var divideKernel))
            throw new InvalidOperationException("CUDA kernel not found: scatter_mean_divide");
        int outputFlat = outputSize * featureSize;
        uint grid2 = (uint)((outputFlat + DefaultBlockSize - 1) / DefaultBlockSize);
        void** args2 = stackalloc void*[4];
        args2[0] = &oPtr; args2[1] = &cPtr; args2[2] = &outputFlat; args2[3] = &featureSize;
        LaunchKernel(divideKernel, grid2, DefaultBlockSize, args2);
    }

    #endregion

    #region Loss Function GPU Kernel Operations

    public unsafe void L1Loss(IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer loss, int batchSize, int numFeatures)
    {
        if (!_kernelCache.TryGetValue("l1_loss", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: l1_loss");
        using var _ = PushContext();
        uint grid = (uint)((batchSize + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr pPtr = predictions.Handle; IntPtr tPtr = targets.Handle; IntPtr lPtr = loss.Handle; int bs = batchSize; int nf = numFeatures;
        void** args = stackalloc void*[5]; args[0] = &pPtr; args[1] = &tPtr; args[2] = &lPtr; args[3] = &bs; args[4] = &nf;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    public unsafe void HuberLoss(IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer loss, int batchSize, int numFeatures, float delta)
    {
        if (!_kernelCache.TryGetValue("huber_loss", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: huber_loss");
        using var _ = PushContext();
        uint grid = (uint)((batchSize + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr pPtr = predictions.Handle; IntPtr tPtr = targets.Handle; IntPtr lPtr = loss.Handle; int bs = batchSize; int nf = numFeatures;
        void** args = stackalloc void*[6]; args[0] = &pPtr; args[1] = &tPtr; args[2] = &lPtr; args[3] = &bs; args[4] = &nf; args[5] = &delta;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    public unsafe void BceWithLogitsLoss(IGpuBuffer logits, IGpuBuffer targets, IGpuBuffer loss, int size)
    {
        if (!_kernelCache.TryGetValue("bce_with_logits_loss", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: bce_with_logits_loss");
        using var _ = PushContext();
        uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr lPtr = logits.Handle; IntPtr tPtr = targets.Handle; IntPtr oPtr = loss.Handle; int n = size;
        void** args = stackalloc void*[4]; args[0] = &lPtr; args[1] = &tPtr; args[2] = &oPtr; args[3] = &n;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    public unsafe void NllLoss(IGpuBuffer logProbs, IGpuBuffer targets, IGpuBuffer loss, int batchSize, int numClasses)
    {
        if (!_kernelCache.TryGetValue("nll_loss", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: nll_loss");
        using var _ = PushContext();
        uint grid = (uint)((batchSize + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr lpPtr = logProbs.Handle; IntPtr tPtr = targets.Handle; IntPtr lPtr = loss.Handle; int bs = batchSize; int nc = numClasses;
        void** args = stackalloc void*[5]; args[0] = &lpPtr; args[1] = &tPtr; args[2] = &lPtr; args[3] = &bs; args[4] = &nc;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    public unsafe void KlDivLoss(IGpuBuffer input, IGpuBuffer target, IGpuBuffer loss, int size)
    {
        if (!_kernelCache.TryGetValue("kl_div_loss", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: kl_div_loss");
        using var _ = PushContext();
        uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr iPtr = input.Handle; IntPtr tPtr = target.Handle; IntPtr lPtr = loss.Handle; int n = size;
        void** args = stackalloc void*[4]; args[0] = &iPtr; args[1] = &tPtr; args[2] = &lPtr; args[3] = &n;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    public unsafe void MseLossBackward(IGpuBuffer gradOutput, IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer gradInput, int size, float invN)
    {
        if (!_kernelCache.TryGetValue("mse_loss_backward", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: mse_loss_backward");
        using var _ = PushContext();
        uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr goPtr = gradOutput.Handle; IntPtr pPtr = predictions.Handle; IntPtr tPtr = targets.Handle; IntPtr giPtr = gradInput.Handle; int n = size;
        void** args = stackalloc void*[6]; args[0] = &goPtr; args[1] = &pPtr; args[2] = &tPtr; args[3] = &giPtr; args[4] = &n; args[5] = &invN;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    public unsafe void L1LossBackward(IGpuBuffer gradOutput, IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer gradInput, int size, float invN)
    {
        if (!_kernelCache.TryGetValue("l1_loss_backward", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: l1_loss_backward");
        using var _ = PushContext();
        uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr goPtr = gradOutput.Handle; IntPtr pPtr = predictions.Handle; IntPtr tPtr = targets.Handle; IntPtr giPtr = gradInput.Handle; int n = size;
        void** args = stackalloc void*[6]; args[0] = &goPtr; args[1] = &pPtr; args[2] = &tPtr; args[3] = &giPtr; args[4] = &n; args[5] = &invN;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    public unsafe void HuberLossBackward(IGpuBuffer gradOutput, IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer gradInput, int size, float invN, float delta)
    {
        if (!_kernelCache.TryGetValue("huber_loss_backward", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: huber_loss_backward");
        using var _ = PushContext();
        uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr goPtr = gradOutput.Handle; IntPtr pPtr = predictions.Handle; IntPtr tPtr = targets.Handle; IntPtr giPtr = gradInput.Handle; int n = size;
        void** args = stackalloc void*[7]; args[0] = &goPtr; args[1] = &pPtr; args[2] = &tPtr; args[3] = &giPtr; args[4] = &n; args[5] = &invN; args[6] = &delta;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    public unsafe void BceWithLogitsBackward(IGpuBuffer gradOutput, IGpuBuffer logits, IGpuBuffer targets, IGpuBuffer gradInput, int size, float invN)
    {
        if (!_kernelCache.TryGetValue("bce_with_logits_backward", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: bce_with_logits_backward");
        using var _ = PushContext();
        uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr goPtr = gradOutput.Handle; IntPtr lPtr = logits.Handle; IntPtr tPtr = targets.Handle; IntPtr giPtr = gradInput.Handle; int n = size;
        void** args = stackalloc void*[6]; args[0] = &goPtr; args[1] = &lPtr; args[2] = &tPtr; args[3] = &giPtr; args[4] = &n; args[5] = &invN;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    #endregion

    #region StopGradient and Fused Linear Operations

    public void CopyBuffer(IGpuBuffer source, IGpuBuffer destination, int size)
    {
        using var _ = PushContext();
        CuBlasNative.CheckCudaResult(
            CuBlasNative.cuMemcpyDtoD(destination.Handle, source.Handle, (nuint)(size * sizeof(float))),
            "cuMemcpyDtoD (CopyBuffer)");
    }

    public unsafe void FusedLinearReLU(IGpuBuffer input, IGpuBuffer weight, IGpuBuffer bias, IGpuBuffer output,
        int batchSize, int inFeatures, int outFeatures)
    {
        if (batchSize <= 0 || inFeatures <= 0 || outFeatures <= 0) return;
        if (!_kernelCache.TryGetValue("fused_linear_relu", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: fused_linear_relu");
        using var _ = PushContext();
        int total = batchSize * outFeatures;
        uint grid = (uint)((total + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr iPtr = input.Handle, wPtr = weight.Handle, bPtr = bias.Handle, oPtr = output.Handle;
        void** args = stackalloc void*[7];
        args[0] = &iPtr; args[1] = &wPtr; args[2] = &bPtr; args[3] = &oPtr;
        args[4] = &batchSize; args[5] = &inFeatures; args[6] = &outFeatures;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    public unsafe void FusedLinearSigmoid(IGpuBuffer input, IGpuBuffer weight, IGpuBuffer bias, IGpuBuffer output,
        int batchSize, int inFeatures, int outFeatures)
    {
        if (batchSize <= 0 || inFeatures <= 0 || outFeatures <= 0) return;
        if (!_kernelCache.TryGetValue("fused_linear_sigmoid", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: fused_linear_sigmoid");
        using var _ = PushContext();
        int total = batchSize * outFeatures;
        uint grid = (uint)((total + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr iPtr = input.Handle, wPtr = weight.Handle, bPtr = bias.Handle, oPtr = output.Handle;
        void** args = stackalloc void*[7];
        args[0] = &iPtr; args[1] = &wPtr; args[2] = &bPtr; args[3] = &oPtr;
        args[4] = &batchSize; args[5] = &inFeatures; args[6] = &outFeatures;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    public unsafe void FusedLinearTanh(IGpuBuffer input, IGpuBuffer weight, IGpuBuffer bias, IGpuBuffer output,
        int batchSize, int inFeatures, int outFeatures)
    {
        if (batchSize <= 0 || inFeatures <= 0 || outFeatures <= 0) return;
        if (!_kernelCache.TryGetValue("fused_linear_tanh", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: fused_linear_tanh");
        using var _ = PushContext();
        int total = batchSize * outFeatures;
        uint grid = (uint)((total + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr iPtr = input.Handle, wPtr = weight.Handle, bPtr = bias.Handle, oPtr = output.Handle;
        void** args = stackalloc void*[7];
        args[0] = &iPtr; args[1] = &wPtr; args[2] = &bPtr; args[3] = &oPtr;
        args[4] = &batchSize; args[5] = &inFeatures; args[6] = &outFeatures;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    public unsafe void FusedLinearGELU(IGpuBuffer input, IGpuBuffer weight, IGpuBuffer bias, IGpuBuffer output,
        int batchSize, int inFeatures, int outFeatures)
    {
        if (batchSize <= 0 || inFeatures <= 0 || outFeatures <= 0) return;
        if (!_kernelCache.TryGetValue("fused_linear_gelu", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: fused_linear_gelu");
        using var _ = PushContext();
        int total = batchSize * outFeatures;
        uint grid = (uint)((total + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr iPtr = input.Handle, wPtr = weight.Handle, bPtr = bias.Handle, oPtr = output.Handle;
        void** args = stackalloc void*[7];
        args[0] = &iPtr; args[1] = &wPtr; args[2] = &bPtr; args[3] = &oPtr;
        args[4] = &batchSize; args[5] = &inFeatures; args[6] = &outFeatures;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    public unsafe void FusedLinearSwish(IGpuBuffer input, IGpuBuffer weight, IGpuBuffer bias, IGpuBuffer output,
        int batchSize, int inFeatures, int outFeatures)
    {
        if (batchSize <= 0 || inFeatures <= 0 || outFeatures <= 0) return;
        if (!_kernelCache.TryGetValue("fused_linear_swish", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: fused_linear_swish");
        using var _ = PushContext();
        int total = batchSize * outFeatures;
        uint grid = (uint)((total + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr iPtr = input.Handle, wPtr = weight.Handle, bPtr = bias.Handle, oPtr = output.Handle;
        void** args = stackalloc void*[7];
        args[0] = &iPtr; args[1] = &wPtr; args[2] = &bPtr; args[3] = &oPtr;
        args[4] = &batchSize; args[5] = &inFeatures; args[6] = &outFeatures;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    public unsafe void FusedLinearReLUBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer weight,
        IGpuBuffer preActivation, IGpuBuffer gradInput, IGpuBuffer gradWeight, IGpuBuffer gradBias,
        int batchSize, int inFeatures, int outFeatures)
    {
        // Grad input kernel
        if (!_kernelCache.TryGetValue("fused_linear_relu_backward_grad_input", out var giKernel))
            throw new InvalidOperationException("CUDA kernel not found: fused_linear_relu_backward_grad_input");
        using var _ = PushContext();
        int totalIn = batchSize * inFeatures;
        uint gridIn = (uint)((totalIn + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr goPtr = gradOutput.Handle, wPtr = weight.Handle, paPtr = preActivation.Handle, giPtr = gradInput.Handle;
        void** argsIn = stackalloc void*[7];
        argsIn[0] = &goPtr; argsIn[1] = &wPtr; argsIn[2] = &paPtr; argsIn[3] = &giPtr;
        argsIn[4] = &batchSize; argsIn[5] = &inFeatures; argsIn[6] = &outFeatures;
        LaunchKernel(giKernel, gridIn, DefaultBlockSize, argsIn);

        // Bias gradient kernel
        if (!_kernelCache.TryGetValue("fused_linear_bias_grad", out var bgKernel))
            throw new InvalidOperationException("CUDA kernel not found: fused_linear_bias_grad");
        uint gridBias = (uint)((outFeatures + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr gbPtr = gradBias.Handle;
        int activationType = 0; // ReLU
        void** argsBias = stackalloc void*[6];
        argsBias[0] = &goPtr; argsBias[1] = &paPtr; argsBias[2] = &gbPtr;
        argsBias[3] = &batchSize; argsBias[4] = &outFeatures; argsBias[5] = &activationType;
        LaunchKernel(bgKernel, gridBias, DefaultBlockSize, argsBias);

        // Weight gradient kernel: gradWeight[i,j] = sum_b(input[b,i] * masked_grad[b,j])
        if (!_kernelCache.TryGetValue("fused_linear_weight_grad", out var wgKernel))
            throw new InvalidOperationException("CUDA kernel not found: fused_linear_weight_grad");
        int totalW = inFeatures * outFeatures;
        uint gridW = (uint)((totalW + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr iPtr = input.Handle, gwPtr = gradWeight.Handle;
        void** argsW = stackalloc void*[8];
        argsW[0] = &goPtr; argsW[1] = &iPtr; argsW[2] = &paPtr; argsW[3] = &gwPtr;
        argsW[4] = &batchSize; argsW[5] = &inFeatures; argsW[6] = &outFeatures; argsW[7] = &activationType;
        LaunchKernel(wgKernel, gridW, DefaultBlockSize, argsW);
    }

    public unsafe void FusedLinearSigmoidBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer weight,
        IGpuBuffer output, IGpuBuffer gradInput, IGpuBuffer gradWeight, IGpuBuffer gradBias,
        int batchSize, int inFeatures, int outFeatures)
    {
        if (!_kernelCache.TryGetValue("fused_linear_sigmoid_backward_grad_input", out var giKernel))
            throw new InvalidOperationException("CUDA kernel not found: fused_linear_sigmoid_backward_grad_input");
        using var _ = PushContext();
        int totalIn = batchSize * inFeatures;
        uint gridIn = (uint)((totalIn + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr goPtr = gradOutput.Handle, wPtr = weight.Handle, oPtr = output.Handle, giPtr = gradInput.Handle;
        void** argsIn = stackalloc void*[7];
        argsIn[0] = &goPtr; argsIn[1] = &wPtr; argsIn[2] = &oPtr; argsIn[3] = &giPtr;
        argsIn[4] = &batchSize; argsIn[5] = &inFeatures; argsIn[6] = &outFeatures;
        LaunchKernel(giKernel, gridIn, DefaultBlockSize, argsIn);

        if (!_kernelCache.TryGetValue("fused_linear_bias_grad", out var bgKernel))
            throw new InvalidOperationException("CUDA kernel not found: fused_linear_bias_grad");
        uint gridBias = (uint)((outFeatures + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr gbPtr = gradBias.Handle;
        int activationType = 1; // Sigmoid
        void** argsBias = stackalloc void*[6];
        argsBias[0] = &goPtr; argsBias[1] = &oPtr; argsBias[2] = &gbPtr;
        argsBias[3] = &batchSize; argsBias[4] = &outFeatures; argsBias[5] = &activationType;
        LaunchKernel(bgKernel, gridBias, DefaultBlockSize, argsBias);

        if (!_kernelCache.TryGetValue("fused_linear_weight_grad", out var wgKernel))
            throw new InvalidOperationException("CUDA kernel not found: fused_linear_weight_grad");
        int totalW = inFeatures * outFeatures;
        uint gridW = (uint)((totalW + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr iPtr = input.Handle, gwPtr = gradWeight.Handle;
        void** argsW = stackalloc void*[8];
        argsW[0] = &goPtr; argsW[1] = &iPtr; argsW[2] = &oPtr; argsW[3] = &gwPtr;
        argsW[4] = &batchSize; argsW[5] = &inFeatures; argsW[6] = &outFeatures; argsW[7] = &activationType;
        LaunchKernel(wgKernel, gridW, DefaultBlockSize, argsW);
    }

    public unsafe void FusedLinearTanhBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer weight,
        IGpuBuffer output, IGpuBuffer gradInput, IGpuBuffer gradWeight, IGpuBuffer gradBias,
        int batchSize, int inFeatures, int outFeatures)
    {
        if (!_kernelCache.TryGetValue("fused_linear_tanh_backward_grad_input", out var giKernel))
            throw new InvalidOperationException("CUDA kernel not found: fused_linear_tanh_backward_grad_input");
        using var _ = PushContext();
        int totalIn = batchSize * inFeatures;
        uint gridIn = (uint)((totalIn + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr goPtr = gradOutput.Handle, wPtr = weight.Handle, oPtr = output.Handle, giPtr = gradInput.Handle;
        void** argsIn = stackalloc void*[7];
        argsIn[0] = &goPtr; argsIn[1] = &wPtr; argsIn[2] = &oPtr; argsIn[3] = &giPtr;
        argsIn[4] = &batchSize; argsIn[5] = &inFeatures; argsIn[6] = &outFeatures;
        LaunchKernel(giKernel, gridIn, DefaultBlockSize, argsIn);

        if (!_kernelCache.TryGetValue("fused_linear_bias_grad", out var bgKernel))
            throw new InvalidOperationException("CUDA kernel not found: fused_linear_bias_grad");
        uint gridBias = (uint)((outFeatures + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr gbPtr = gradBias.Handle;
        int activationType = 2; // Tanh
        void** argsBias = stackalloc void*[6];
        argsBias[0] = &goPtr; argsBias[1] = &oPtr; argsBias[2] = &gbPtr;
        argsBias[3] = &batchSize; argsBias[4] = &outFeatures; argsBias[5] = &activationType;
        LaunchKernel(bgKernel, gridBias, DefaultBlockSize, argsBias);

        if (!_kernelCache.TryGetValue("fused_linear_weight_grad", out var wgKernel))
            throw new InvalidOperationException("CUDA kernel not found: fused_linear_weight_grad");
        int totalW = inFeatures * outFeatures;
        uint gridW = (uint)((totalW + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr iPtr = input.Handle, gwPtr = gradWeight.Handle;
        void** argsW = stackalloc void*[8];
        argsW[0] = &goPtr; argsW[1] = &iPtr; argsW[2] = &oPtr; argsW[3] = &gwPtr;
        argsW[4] = &batchSize; argsW[5] = &inFeatures; argsW[6] = &outFeatures; argsW[7] = &activationType;
        LaunchKernel(wgKernel, gridW, DefaultBlockSize, argsW);
    }

    public unsafe void FusedLinearGELUBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer weight,
        IGpuBuffer preActivation, IGpuBuffer gradInput, IGpuBuffer gradWeight, IGpuBuffer gradBias,
        int batchSize, int inFeatures, int outFeatures)
    {
        if (!_kernelCache.TryGetValue("fused_linear_gelu_backward_grad_input", out var giKernel))
            throw new InvalidOperationException("CUDA kernel not found: fused_linear_gelu_backward_grad_input");
        using var _ = PushContext();
        int totalIn = batchSize * inFeatures;
        uint gridIn = (uint)((totalIn + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr goPtr = gradOutput.Handle, wPtr = weight.Handle, paPtr = preActivation.Handle, giPtr = gradInput.Handle;
        void** argsIn = stackalloc void*[7];
        argsIn[0] = &goPtr; argsIn[1] = &wPtr; argsIn[2] = &paPtr; argsIn[3] = &giPtr;
        argsIn[4] = &batchSize; argsIn[5] = &inFeatures; argsIn[6] = &outFeatures;
        LaunchKernel(giKernel, gridIn, DefaultBlockSize, argsIn);

        if (!_kernelCache.TryGetValue("fused_linear_bias_grad", out var bgKernel))
            throw new InvalidOperationException("CUDA kernel not found: fused_linear_bias_grad");
        uint gridBias = (uint)((outFeatures + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr gbPtr = gradBias.Handle;
        int activationType = 3; // GELU
        void** argsBias = stackalloc void*[6];
        argsBias[0] = &goPtr; argsBias[1] = &paPtr; argsBias[2] = &gbPtr;
        argsBias[3] = &batchSize; argsBias[4] = &outFeatures; argsBias[5] = &activationType;
        LaunchKernel(bgKernel, gridBias, DefaultBlockSize, argsBias);

        if (!_kernelCache.TryGetValue("fused_linear_weight_grad", out var wgKernel))
            throw new InvalidOperationException("CUDA kernel not found: fused_linear_weight_grad");
        int totalW = inFeatures * outFeatures;
        uint gridW = (uint)((totalW + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr iPtr = input.Handle, gwPtr = gradWeight.Handle;
        void** argsW = stackalloc void*[8];
        argsW[0] = &goPtr; argsW[1] = &iPtr; argsW[2] = &paPtr; argsW[3] = &gwPtr;
        argsW[4] = &batchSize; argsW[5] = &inFeatures; argsW[6] = &outFeatures; argsW[7] = &activationType;
        LaunchKernel(wgKernel, gridW, DefaultBlockSize, argsW);
    }

    public unsafe void FusedLinearSwishBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer weight,
        IGpuBuffer preActivation, IGpuBuffer gradInput, IGpuBuffer gradWeight, IGpuBuffer gradBias,
        int batchSize, int inFeatures, int outFeatures)
    {
        if (!_kernelCache.TryGetValue("fused_linear_swish_backward_grad_input", out var giKernel))
            throw new InvalidOperationException("CUDA kernel not found: fused_linear_swish_backward_grad_input");
        using var _ = PushContext();
        int totalIn = batchSize * inFeatures;
        uint gridIn = (uint)((totalIn + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr goPtr = gradOutput.Handle, wPtr = weight.Handle, paPtr = preActivation.Handle, giPtr = gradInput.Handle;
        void** argsIn = stackalloc void*[7];
        argsIn[0] = &goPtr; argsIn[1] = &wPtr; argsIn[2] = &paPtr; argsIn[3] = &giPtr;
        argsIn[4] = &batchSize; argsIn[5] = &inFeatures; argsIn[6] = &outFeatures;
        LaunchKernel(giKernel, gridIn, DefaultBlockSize, argsIn);

        if (!_kernelCache.TryGetValue("fused_linear_bias_grad", out var bgKernel))
            throw new InvalidOperationException("CUDA kernel not found: fused_linear_bias_grad");
        uint gridBias = (uint)((outFeatures + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr gbPtr = gradBias.Handle;
        int activationType = 4; // Swish
        void** argsBias = stackalloc void*[6];
        argsBias[0] = &goPtr; argsBias[1] = &paPtr; argsBias[2] = &gbPtr;
        argsBias[3] = &batchSize; argsBias[4] = &outFeatures; argsBias[5] = &activationType;
        LaunchKernel(bgKernel, gridBias, DefaultBlockSize, argsBias);

        if (!_kernelCache.TryGetValue("fused_linear_weight_grad", out var wgKernel))
            throw new InvalidOperationException("CUDA kernel not found: fused_linear_weight_grad");
        int totalW = inFeatures * outFeatures;
        uint gridW = (uint)((totalW + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr iPtr = input.Handle, gwPtr = gradWeight.Handle;
        void** argsW = stackalloc void*[8];
        argsW[0] = &goPtr; argsW[1] = &iPtr; argsW[2] = &paPtr; argsW[3] = &gwPtr;
        argsW[4] = &batchSize; argsW[5] = &inFeatures; argsW[6] = &outFeatures; argsW[7] = &activationType;
        LaunchKernel(wgKernel, gridW, DefaultBlockSize, argsW);
    }

    #endregion

    #region IoU Loss GPU Operations

    public unsafe void IoULoss(IGpuBuffer predicted, IGpuBuffer target, IGpuBuffer loss, int numBoxes)
    {
        if (numBoxes <= 0) return;
        if (!_kernelCache.TryGetValue("iou_loss", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: iou_loss");
        using var _ = PushContext();
        uint grid = (uint)((numBoxes + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr pPtr = predicted.Handle, tPtr = target.Handle, lPtr = loss.Handle;
        void** args = stackalloc void*[4];
        args[0] = &pPtr; args[1] = &tPtr; args[2] = &lPtr; args[3] = &numBoxes;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    public unsafe void GIoULoss(IGpuBuffer predicted, IGpuBuffer target, IGpuBuffer loss, int numBoxes)
    {
        if (numBoxes <= 0) return;
        if (!_kernelCache.TryGetValue("giou_loss", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: giou_loss");
        using var _ = PushContext();
        uint grid = (uint)((numBoxes + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr pPtr = predicted.Handle, tPtr = target.Handle, lPtr = loss.Handle;
        void** args = stackalloc void*[4];
        args[0] = &pPtr; args[1] = &tPtr; args[2] = &lPtr; args[3] = &numBoxes;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    public unsafe void DIoULoss(IGpuBuffer predicted, IGpuBuffer target, IGpuBuffer loss, int numBoxes)
    {
        if (numBoxes <= 0) return;
        if (!_kernelCache.TryGetValue("diou_loss", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: diou_loss");
        using var _ = PushContext();
        uint grid = (uint)((numBoxes + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr pPtr = predicted.Handle, tPtr = target.Handle, lPtr = loss.Handle;
        void** args = stackalloc void*[4];
        args[0] = &pPtr; args[1] = &tPtr; args[2] = &lPtr; args[3] = &numBoxes;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    public unsafe void CIoULoss(IGpuBuffer predicted, IGpuBuffer target, IGpuBuffer loss, int numBoxes)
    {
        if (numBoxes <= 0) return;
        if (!_kernelCache.TryGetValue("ciou_loss", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: ciou_loss");
        using var _ = PushContext();
        uint grid = (uint)((numBoxes + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr pPtr = predicted.Handle, tPtr = target.Handle, lPtr = loss.Handle;
        void** args = stackalloc void*[4];
        args[0] = &pPtr; args[1] = &tPtr; args[2] = &lPtr; args[3] = &numBoxes;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    public unsafe void IoULossBackward(IGpuBuffer gradOutput, IGpuBuffer predicted, IGpuBuffer target,
        IGpuBuffer gradPredicted, int numBoxes)
    {
        if (numBoxes <= 0) return;
        if (!_kernelCache.TryGetValue("iou_loss_backward", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: iou_loss_backward");
        using var _ = PushContext();
        uint grid = (uint)((numBoxes + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr goPtr = gradOutput.Handle, pPtr = predicted.Handle, tPtr = target.Handle, gpPtr = gradPredicted.Handle;
        void** args = stackalloc void*[5];
        args[0] = &goPtr; args[1] = &pPtr; args[2] = &tPtr; args[3] = &gpPtr; args[4] = &numBoxes;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    public unsafe void GIoULossBackward(IGpuBuffer gradOutput, IGpuBuffer predicted, IGpuBuffer target,
        IGpuBuffer gradPredicted, int numBoxes)
    {
        if (numBoxes <= 0) return;
        if (!_kernelCache.TryGetValue("giou_loss_backward", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: giou_loss_backward");
        using var _ = PushContext();
        uint grid = (uint)((numBoxes + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr goPtr = gradOutput.Handle, pPtr = predicted.Handle, tPtr = target.Handle, gpPtr = gradPredicted.Handle;
        void** args = stackalloc void*[5];
        args[0] = &goPtr; args[1] = &pPtr; args[2] = &tPtr; args[3] = &gpPtr; args[4] = &numBoxes;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    public unsafe void DIoULossBackward(IGpuBuffer gradOutput, IGpuBuffer predicted, IGpuBuffer target,
        IGpuBuffer gradPredicted, int numBoxes)
    {
        if (numBoxes <= 0) return;
        if (!_kernelCache.TryGetValue("diou_loss_backward", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: diou_loss_backward");
        using var _ = PushContext();
        uint grid = (uint)((numBoxes + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr goPtr = gradOutput.Handle, pPtr = predicted.Handle, tPtr = target.Handle, gpPtr = gradPredicted.Handle;
        void** args = stackalloc void*[5];
        args[0] = &goPtr; args[1] = &pPtr; args[2] = &tPtr; args[3] = &gpPtr; args[4] = &numBoxes;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    public unsafe void CIoULossBackward(IGpuBuffer gradOutput, IGpuBuffer predicted, IGpuBuffer target,
        IGpuBuffer gradPredicted, int numBoxes)
    {
        if (numBoxes <= 0) return;
        if (!_kernelCache.TryGetValue("ciou_loss_backward", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: ciou_loss_backward");
        using var _ = PushContext();
        uint grid = (uint)((numBoxes + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr goPtr = gradOutput.Handle, pPtr = predicted.Handle, tPtr = target.Handle, gpPtr = gradPredicted.Handle;
        void** args = stackalloc void*[5];
        args[0] = &goPtr; args[1] = &pPtr; args[2] = &tPtr; args[3] = &gpPtr; args[4] = &numBoxes;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    #endregion


    #region Loss Function Operations

    public unsafe float CrossEntropyLoss(IGpuBuffer predictions, IGpuBuffer targets, int batchSize, int numClasses)
    {
        // Validate parameters to prevent division-by-zero and OOB access
        if (batchSize <= 0)
            throw new ArgumentOutOfRangeException(nameof(batchSize), batchSize, "batchSize must be positive to avoid division-by-zero.");
        if (numClasses <= 0)
            throw new ArgumentOutOfRangeException(nameof(numClasses), numClasses, "numClasses must be positive.");

        if (!_kernelCache.TryGetValue("cross_entropy_loss", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: cross_entropy_loss");

        using var _ = PushContext();
        using var temp = AllocateBuffer(batchSize);
        uint grid = (uint)((batchSize + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr predsPtr = predictions.Handle;
        IntPtr targetsPtr = targets.Handle;
        IntPtr outputPtr = temp.Handle;
        void** args = stackalloc void*[5];
        args[0] = &predsPtr;
        args[1] = &targetsPtr;
        args[2] = &outputPtr;
        args[3] = &batchSize;
        args[4] = &numClasses;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);

        // Use GPU-only reduction to avoid downloading all partials to host
        // This iteratively reduces on GPU until a single scalar remains
        return SumGpuReduction(temp, batchSize) / batchSize;
    }

    public unsafe void CrossEntropyBackward(IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer gradInput, int batchSize, int numClasses)
    {
        // Validate parameters to prevent OOB access
        if (batchSize <= 0)
            throw new ArgumentOutOfRangeException(nameof(batchSize), batchSize, "batchSize must be positive.");
        if (numClasses <= 0)
            throw new ArgumentOutOfRangeException(nameof(numClasses), numClasses, "numClasses must be positive.");

        if (!_kernelCache.TryGetValue("cross_entropy_backward", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: cross_entropy_backward");

        using var _ = PushContext();
        uint grid = (uint)((batchSize * numClasses + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr predsPtr = predictions.Handle;
        IntPtr targetsPtr = targets.Handle;
        IntPtr gradPtr = gradInput.Handle;
        int totalSize = batchSize * numClasses;
        void** args = stackalloc void*[5];
        args[0] = &predsPtr;
        args[1] = &targetsPtr;
        args[2] = &gradPtr;
        args[3] = &batchSize;
        args[4] = &numClasses;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    public unsafe float BinaryCrossEntropyLoss(IGpuBuffer predictions, IGpuBuffer targets, int size)
    {
        if (!_kernelCache.TryGetValue("bce_loss", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: bce_loss");

        using var _ = PushContext();
        using var temp = AllocateBuffer(size);
        uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr predsPtr = predictions.Handle;
        IntPtr targetsPtr = targets.Handle;
        IntPtr outputPtr = temp.Handle;
        void** args = stackalloc void*[4];
        args[0] = &predsPtr;
        args[1] = &targetsPtr;
        args[2] = &outputPtr;
        args[3] = &size;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
        return Sum(temp, size) / size;
    }

    public unsafe void BinaryCrossEntropyBackward(IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer gradInput, int size)
    {
        if (!_kernelCache.TryGetValue("bce_backward", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: bce_backward");

        using var _ = PushContext();
        uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr predsPtr = predictions.Handle;
        IntPtr targetsPtr = targets.Handle;
        IntPtr gradPtr = gradInput.Handle;
        void** args = stackalloc void*[4];
        args[0] = &predsPtr;
        args[1] = &targetsPtr;
        args[2] = &gradPtr;
        args[3] = &size;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    public unsafe float MseLoss(IGpuBuffer predictions, IGpuBuffer targets, int size)
    {
        if (!_kernelCache.TryGetValue("mse_loss", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: mse_loss");

        using var _ = PushContext();
        using var temp = AllocateBuffer(size);
        uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr predsPtr = predictions.Handle;
        IntPtr targetsPtr = targets.Handle;
        IntPtr outputPtr = temp.Handle;
        void** args = stackalloc void*[4];
        args[0] = &predsPtr;
        args[1] = &targetsPtr;
        args[2] = &outputPtr;
        args[3] = &size;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
        return Sum(temp, size) / size;
    }

    public unsafe void MseBackward(IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer gradInput, int size)
    {
        if (!_kernelCache.TryGetValue("mse_backward", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: mse_backward");

        using var _ = PushContext();
        uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr predsPtr = predictions.Handle;
        IntPtr targetsPtr = targets.Handle;
        IntPtr gradPtr = gradInput.Handle;
        void** args = stackalloc void*[4];
        args[0] = &predsPtr;
        args[1] = &targetsPtr;
        args[2] = &gradPtr;
        args[3] = &size;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    public unsafe float SmoothL1Loss(IGpuBuffer predictions, IGpuBuffer targets, int size, float beta)
    {
        if (!_kernelCache.TryGetValue("smooth_l1_loss", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: smooth_l1_loss");

        using var _ = PushContext();
        using var temp = AllocateBuffer(size);
        uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr predsPtr = predictions.Handle;
        IntPtr targetsPtr = targets.Handle;
        IntPtr outputPtr = temp.Handle;
        void** args = stackalloc void*[5];
        args[0] = &predsPtr;
        args[1] = &targetsPtr;
        args[2] = &outputPtr;
        args[3] = &size;
        args[4] = &beta;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
        return Sum(temp, size) / size;
    }

    public unsafe void SmoothL1Backward(IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer gradInput, int size, float beta)
    {
        if (!_kernelCache.TryGetValue("smooth_l1_backward", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: smooth_l1_backward");

        using var _ = PushContext();
        uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr predsPtr = predictions.Handle;
        IntPtr targetsPtr = targets.Handle;
        IntPtr gradPtr = gradInput.Handle;
        void** args = stackalloc void*[5];
        args[0] = &predsPtr;
        args[1] = &targetsPtr;
        args[2] = &gradPtr;
        args[3] = &size;
        args[4] = &beta;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    public unsafe float TripletLoss(IGpuBuffer anchor, IGpuBuffer positive, IGpuBuffer negative, int batchSize, int embeddingDim, float margin)
    {
        if (anchor is null) throw new ArgumentNullException(nameof(anchor));
        if (positive is null) throw new ArgumentNullException(nameof(positive));
        if (negative is null) throw new ArgumentNullException(nameof(negative));
        if (batchSize <= 0) throw new ArgumentOutOfRangeException(nameof(batchSize), "Batch size must be positive.");
        if (embeddingDim <= 0) throw new ArgumentOutOfRangeException(nameof(embeddingDim), "Embedding dimension must be positive.");

        if (!_kernelCache.TryGetValue("triplet_loss", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: triplet_loss");

        using var _ = PushContext();
        using var lossBuffer = AllocateBuffer(batchSize);
        uint grid = (uint)((batchSize + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr anchorPtr = anchor.Handle;
        IntPtr positivePtr = positive.Handle;
        IntPtr negativePtr = negative.Handle;
        IntPtr lossPtr = lossBuffer.Handle;
        void** args = stackalloc void*[7];
        args[0] = &anchorPtr;
        args[1] = &positivePtr;
        args[2] = &negativePtr;
        args[3] = &lossPtr;
        args[4] = &batchSize;
        args[5] = &embeddingDim;
        args[6] = &margin;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
        return Sum(lossBuffer, batchSize) / batchSize;
    }

    public unsafe void TripletLossBackward(IGpuBuffer anchor, IGpuBuffer positive, IGpuBuffer negative,
        IGpuBuffer gradAnchor, IGpuBuffer gradPositive, IGpuBuffer gradNegative,
        int batchSize, int embeddingDim, float margin)
    {
        if (anchor is null) throw new ArgumentNullException(nameof(anchor));
        if (positive is null) throw new ArgumentNullException(nameof(positive));
        if (negative is null) throw new ArgumentNullException(nameof(negative));
        if (gradAnchor is null) throw new ArgumentNullException(nameof(gradAnchor));
        if (gradPositive is null) throw new ArgumentNullException(nameof(gradPositive));
        if (gradNegative is null) throw new ArgumentNullException(nameof(gradNegative));
        if (batchSize <= 0) throw new ArgumentOutOfRangeException(nameof(batchSize), "Batch size must be positive.");
        if (embeddingDim <= 0) throw new ArgumentOutOfRangeException(nameof(embeddingDim), "Embedding dimension must be positive.");

        if (!_kernelCache.TryGetValue("triplet_loss_backward", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: triplet_loss_backward");

        using var _ = PushContext();
        // Launch with batchSize blocks, embeddingDim threads per block
        uint blockSize = (uint)Math.Min(embeddingDim, 1024);
        IntPtr anchorPtr = anchor.Handle;
        IntPtr positivePtr = positive.Handle;
        IntPtr negativePtr = negative.Handle;
        IntPtr gradAnchorPtr = gradAnchor.Handle;
        IntPtr gradPositivePtr = gradPositive.Handle;
        IntPtr gradNegativePtr = gradNegative.Handle;
        void** args = stackalloc void*[9];
        args[0] = &anchorPtr;
        args[1] = &positivePtr;
        args[2] = &negativePtr;
        args[3] = &gradAnchorPtr;
        args[4] = &gradPositivePtr;
        args[5] = &gradNegativePtr;
        args[6] = &batchSize;
        args[7] = &embeddingDim;
        args[8] = &margin;
        LaunchKernel(kernel, (uint)batchSize, blockSize, args);
    }

    public unsafe float HuberLoss(IGpuBuffer predictions, IGpuBuffer targets, int size, float delta)
    {
        if (predictions is null) throw new ArgumentNullException(nameof(predictions));
        if (targets is null) throw new ArgumentNullException(nameof(targets));
        if (!_kernelCache.TryGetValue("huber_loss", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: huber_loss");

        using var _ = PushContext();
        using var outputBuffer = AllocateBuffer(size);
        uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr predPtr = predictions.Handle;
        IntPtr targetPtr = targets.Handle;
        IntPtr outPtr = outputBuffer.Handle;
        void** args = stackalloc void*[5];
        args[0] = &predPtr;
        args[1] = &targetPtr;
        args[2] = &outPtr;
        args[3] = &delta;
        args[4] = &size;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
        return Sum(outputBuffer, size) / size;
    }

    public unsafe void HuberBackward(IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer gradInput, int size, float delta)
    {
        if (predictions is null) throw new ArgumentNullException(nameof(predictions));
        if (targets is null) throw new ArgumentNullException(nameof(targets));
        if (gradInput is null) throw new ArgumentNullException(nameof(gradInput));
        if (!_kernelCache.TryGetValue("huber_gradient", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: huber_gradient");

        using var _ = PushContext();
        uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr predPtr = predictions.Handle;
        IntPtr targetPtr = targets.Handle;
        IntPtr gradPtr = gradInput.Handle;
        void** args = stackalloc void*[5];
        args[0] = &predPtr;
        args[1] = &targetPtr;
        args[2] = &gradPtr;
        args[3] = &delta;
        args[4] = &size;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    public unsafe float FocalLoss(IGpuBuffer predictions, IGpuBuffer targets, int size, float alpha, float gamma)
    {
        if (predictions is null) throw new ArgumentNullException(nameof(predictions));
        if (targets is null) throw new ArgumentNullException(nameof(targets));
        if (!_kernelCache.TryGetValue("focal_loss", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: focal_loss");

        using var _ = PushContext();
        using var outputBuffer = AllocateBuffer(size);
        uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr predPtr = predictions.Handle;
        IntPtr targetPtr = targets.Handle;
        IntPtr outPtr = outputBuffer.Handle;
        float epsilon = 1e-7f;
        void** args = stackalloc void*[7];
        args[0] = &predPtr;
        args[1] = &targetPtr;
        args[2] = &outPtr;
        args[3] = &alpha;
        args[4] = &gamma;
        args[5] = &epsilon;
        args[6] = &size;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
        return Sum(outputBuffer, size) / size;
    }

    public unsafe void FocalBackward(IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer gradInput, int size, float alpha, float gamma)
    {
        if (predictions is null) throw new ArgumentNullException(nameof(predictions));
        if (targets is null) throw new ArgumentNullException(nameof(targets));
        if (gradInput is null) throw new ArgumentNullException(nameof(gradInput));
        if (!_kernelCache.TryGetValue("focal_gradient", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: focal_gradient");

        using var _ = PushContext();
        uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr predPtr = predictions.Handle;
        IntPtr targetPtr = targets.Handle;
        IntPtr gradPtr = gradInput.Handle;
        float epsilon = 1e-7f;
        void** args = stackalloc void*[7];
        args[0] = &predPtr;
        args[1] = &targetPtr;
        args[2] = &gradPtr;
        args[3] = &alpha;
        args[4] = &gamma;
        args[5] = &epsilon;
        args[6] = &size;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    public unsafe float MaeLoss(IGpuBuffer predictions, IGpuBuffer targets, int size)
    {
        if (predictions is null) throw new ArgumentNullException(nameof(predictions));
        if (targets is null) throw new ArgumentNullException(nameof(targets));
        if (!_kernelCache.TryGetValue("mae_loss", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: mae_loss");

        using var _ = PushContext();
        using var outputBuffer = AllocateBuffer(size);
        uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr predPtr = predictions.Handle;
        IntPtr targetPtr = targets.Handle;
        IntPtr outPtr = outputBuffer.Handle;
        void** args = stackalloc void*[4];
        args[0] = &predPtr;
        args[1] = &targetPtr;
        args[2] = &outPtr;
        args[3] = &size;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
        return Sum(outputBuffer, size) / size;
    }

    public unsafe void MaeBackward(IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer gradInput, int size)
    {
        if (predictions is null) throw new ArgumentNullException(nameof(predictions));
        if (targets is null) throw new ArgumentNullException(nameof(targets));
        if (gradInput is null) throw new ArgumentNullException(nameof(gradInput));
        if (!_kernelCache.TryGetValue("mae_gradient", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: mae_gradient");

        using var _ = PushContext();
        uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr predPtr = predictions.Handle;
        IntPtr targetPtr = targets.Handle;
        IntPtr gradPtr = gradInput.Handle;
        void** args = stackalloc void*[4];
        args[0] = &predPtr;
        args[1] = &targetPtr;
        args[2] = &gradPtr;
        args[3] = &size;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    public unsafe float LogCoshLoss(IGpuBuffer predictions, IGpuBuffer targets, int size)
    {
        if (predictions is null) throw new ArgumentNullException(nameof(predictions));
        if (targets is null) throw new ArgumentNullException(nameof(targets));
        if (!_kernelCache.TryGetValue("log_cosh_loss", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: log_cosh_loss");

        using var _ = PushContext();
        using var outputBuffer = AllocateBuffer(size);
        uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr predPtr = predictions.Handle;
        IntPtr targetPtr = targets.Handle;
        IntPtr outPtr = outputBuffer.Handle;
        void** args = stackalloc void*[4];
        args[0] = &predPtr;
        args[1] = &targetPtr;
        args[2] = &outPtr;
        args[3] = &size;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
        return Sum(outputBuffer, size) / size;
    }

    public unsafe void LogCoshBackward(IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer gradInput, int size)
    {
        if (predictions is null) throw new ArgumentNullException(nameof(predictions));
        if (targets is null) throw new ArgumentNullException(nameof(targets));
        if (gradInput is null) throw new ArgumentNullException(nameof(gradInput));
        if (!_kernelCache.TryGetValue("log_cosh_gradient", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: log_cosh_gradient");

        using var _ = PushContext();
        uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr predPtr = predictions.Handle;
        IntPtr targetPtr = targets.Handle;
        IntPtr gradPtr = gradInput.Handle;
        void** args = stackalloc void*[4];
        args[0] = &predPtr;
        args[1] = &targetPtr;
        args[2] = &gradPtr;
        args[3] = &size;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    public unsafe float QuantileLoss(IGpuBuffer predictions, IGpuBuffer targets, int size, float quantile)
    {
        if (predictions is null) throw new ArgumentNullException(nameof(predictions));
        if (targets is null) throw new ArgumentNullException(nameof(targets));
        if (!_kernelCache.TryGetValue("quantile_loss", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: quantile_loss");

        using var _ = PushContext();
        using var outputBuffer = AllocateBuffer(size);
        uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr predPtr = predictions.Handle;
        IntPtr targetPtr = targets.Handle;
        IntPtr outPtr = outputBuffer.Handle;
        void** args = stackalloc void*[5];
        args[0] = &predPtr;
        args[1] = &targetPtr;
        args[2] = &outPtr;
        args[3] = &quantile;
        args[4] = &size;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
        return Sum(outputBuffer, size) / size;
    }

    public unsafe void QuantileBackward(IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer gradInput, int size, float quantile)
    {
        if (predictions is null) throw new ArgumentNullException(nameof(predictions));
        if (targets is null) throw new ArgumentNullException(nameof(targets));
        if (gradInput is null) throw new ArgumentNullException(nameof(gradInput));
        if (!_kernelCache.TryGetValue("quantile_gradient", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: quantile_gradient");

        using var _ = PushContext();
        uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr predPtr = predictions.Handle;
        IntPtr targetPtr = targets.Handle;
        IntPtr gradPtr = gradInput.Handle;
        void** args = stackalloc void*[5];
        args[0] = &predPtr;
        args[1] = &targetPtr;
        args[2] = &gradPtr;
        args[3] = &quantile;
        args[4] = &size;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    public unsafe float HingeLoss(IGpuBuffer predictions, IGpuBuffer targets, int size)
    {
        if (predictions is null) throw new ArgumentNullException(nameof(predictions));
        if (targets is null) throw new ArgumentNullException(nameof(targets));
        if (!_kernelCache.TryGetValue("hinge_loss", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: hinge_loss");

        using var _ = PushContext();
        using var outputBuffer = AllocateBuffer(size);
        uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr predPtr = predictions.Handle;
        IntPtr targetPtr = targets.Handle;
        IntPtr outPtr = outputBuffer.Handle;
        void** args = stackalloc void*[4];
        args[0] = &predPtr;
        args[1] = &targetPtr;
        args[2] = &outPtr;
        args[3] = &size;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
        return Sum(outputBuffer, size) / size;
    }

    public unsafe void HingeBackward(IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer gradInput, int size)
    {
        if (predictions is null) throw new ArgumentNullException(nameof(predictions));
        if (targets is null) throw new ArgumentNullException(nameof(targets));
        if (gradInput is null) throw new ArgumentNullException(nameof(gradInput));
        if (!_kernelCache.TryGetValue("hinge_gradient", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: hinge_gradient");

        using var _ = PushContext();
        uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr predPtr = predictions.Handle;
        IntPtr targetPtr = targets.Handle;
        IntPtr gradPtr = gradInput.Handle;
        void** args = stackalloc void*[4];
        args[0] = &predPtr;
        args[1] = &targetPtr;
        args[2] = &gradPtr;
        args[3] = &size;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    public unsafe float SquaredHingeLoss(IGpuBuffer predictions, IGpuBuffer targets, int size)
    {
        if (predictions is null) throw new ArgumentNullException(nameof(predictions));
        if (targets is null) throw new ArgumentNullException(nameof(targets));
        if (!_kernelCache.TryGetValue("squared_hinge_loss", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: squared_hinge_loss");

        using var _ = PushContext();
        using var outputBuffer = AllocateBuffer(size);
        uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr predPtr = predictions.Handle;
        IntPtr targetPtr = targets.Handle;
        IntPtr outPtr = outputBuffer.Handle;
        void** args = stackalloc void*[4];
        args[0] = &predPtr;
        args[1] = &targetPtr;
        args[2] = &outPtr;
        args[3] = &size;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
        return Sum(outputBuffer, size) / size;
    }

    public unsafe void SquaredHingeBackward(IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer gradInput, int size)
    {
        if (predictions is null) throw new ArgumentNullException(nameof(predictions));
        if (targets is null) throw new ArgumentNullException(nameof(targets));
        if (gradInput is null) throw new ArgumentNullException(nameof(gradInput));
        if (!_kernelCache.TryGetValue("squared_hinge_gradient", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: squared_hinge_gradient");

        using var _ = PushContext();
        uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr predPtr = predictions.Handle;
        IntPtr targetPtr = targets.Handle;
        IntPtr gradPtr = gradInput.Handle;
        void** args = stackalloc void*[4];
        args[0] = &predPtr;
        args[1] = &targetPtr;
        args[2] = &gradPtr;
        args[3] = &size;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    public unsafe float PoissonLoss(IGpuBuffer predictions, IGpuBuffer targets, int size)
    {
        if (predictions is null) throw new ArgumentNullException(nameof(predictions));
        if (targets is null) throw new ArgumentNullException(nameof(targets));
        if (!_kernelCache.TryGetValue("poisson_loss", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: poisson_loss");

        using var _ = PushContext();
        using var outputBuffer = AllocateBuffer(size);
        uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr predPtr = predictions.Handle;
        IntPtr targetPtr = targets.Handle;
        IntPtr outPtr = outputBuffer.Handle;
        float epsilon = 1e-7f;
        void** args = stackalloc void*[5];
        args[0] = &predPtr;
        args[1] = &targetPtr;
        args[2] = &outPtr;
        args[3] = &epsilon;
        args[4] = &size;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
        return Sum(outputBuffer, size) / size;
    }

    public unsafe void PoissonBackward(IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer gradInput, int size)
    {
        if (predictions is null) throw new ArgumentNullException(nameof(predictions));
        if (targets is null) throw new ArgumentNullException(nameof(targets));
        if (gradInput is null) throw new ArgumentNullException(nameof(gradInput));
        if (!_kernelCache.TryGetValue("poisson_gradient", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: poisson_gradient");

        using var _ = PushContext();
        uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr predPtr = predictions.Handle;
        IntPtr targetPtr = targets.Handle;
        IntPtr gradPtr = gradInput.Handle;
        float epsilon = 1e-7f;
        void** args = stackalloc void*[5];
        args[0] = &predPtr;
        args[1] = &targetPtr;
        args[2] = &gradPtr;
        args[3] = &epsilon;
        args[4] = &size;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    public unsafe float ExponentialLoss(IGpuBuffer predictions, IGpuBuffer targets, int size)
    {
        if (predictions is null) throw new ArgumentNullException(nameof(predictions));
        if (targets is null) throw new ArgumentNullException(nameof(targets));
        if (!_kernelCache.TryGetValue("exponential_loss", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: exponential_loss");

        using var _ = PushContext();
        using var outputBuffer = AllocateBuffer(size);
        uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr predPtr = predictions.Handle;
        IntPtr targetPtr = targets.Handle;
        IntPtr outPtr = outputBuffer.Handle;
        void** args = stackalloc void*[4];
        args[0] = &predPtr;
        args[1] = &targetPtr;
        args[2] = &outPtr;
        args[3] = &size;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
        return Sum(outputBuffer, size) / size;
    }

    public unsafe void ExponentialBackward(IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer gradInput, int size)
    {
        if (predictions is null) throw new ArgumentNullException(nameof(predictions));
        if (targets is null) throw new ArgumentNullException(nameof(targets));
        if (gradInput is null) throw new ArgumentNullException(nameof(gradInput));
        if (!_kernelCache.TryGetValue("exponential_gradient", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: exponential_gradient");

        using var _ = PushContext();
        uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr predPtr = predictions.Handle;
        IntPtr targetPtr = targets.Handle;
        IntPtr gradPtr = gradInput.Handle;
        void** args = stackalloc void*[4];
        args[0] = &predPtr;
        args[1] = &targetPtr;
        args[2] = &gradPtr;
        args[3] = &size;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    public unsafe float ModifiedHuberLoss(IGpuBuffer predictions, IGpuBuffer targets, int size)
    {
        if (predictions is null) throw new ArgumentNullException(nameof(predictions));
        if (targets is null) throw new ArgumentNullException(nameof(targets));
        if (!_kernelCache.TryGetValue("modified_huber_loss", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: modified_huber_loss");

        using var _ = PushContext();
        using var outputBuffer = AllocateBuffer(size);
        uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr predPtr = predictions.Handle;
        IntPtr targetPtr = targets.Handle;
        IntPtr outPtr = outputBuffer.Handle;
        void** args = stackalloc void*[4];
        args[0] = &predPtr;
        args[1] = &targetPtr;
        args[2] = &outPtr;
        args[3] = &size;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
        return Sum(outputBuffer, size) / size;
    }

    public unsafe void ModifiedHuberBackward(IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer gradInput, int size)
    {
        if (predictions is null) throw new ArgumentNullException(nameof(predictions));
        if (targets is null) throw new ArgumentNullException(nameof(targets));
        if (gradInput is null) throw new ArgumentNullException(nameof(gradInput));
        if (!_kernelCache.TryGetValue("modified_huber_gradient", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: modified_huber_gradient");

        using var _ = PushContext();
        uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr predPtr = predictions.Handle;
        IntPtr targetPtr = targets.Handle;
        IntPtr gradPtr = gradInput.Handle;
        void** args = stackalloc void*[4];
        args[0] = &predPtr;
        args[1] = &targetPtr;
        args[2] = &gradPtr;
        args[3] = &size;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    public unsafe float CategoricalCrossEntropyLoss(IGpuBuffer predictions, IGpuBuffer targets, int size)
    {
        if (predictions is null) throw new ArgumentNullException(nameof(predictions));
        if (targets is null) throw new ArgumentNullException(nameof(targets));
        if (!_kernelCache.TryGetValue("categorical_cross_entropy_loss", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: categorical_cross_entropy_loss");

        using var _ = PushContext();
        using var outputBuffer = AllocateBuffer(size);
        uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr predPtr = predictions.Handle;
        IntPtr targetPtr = targets.Handle;
        IntPtr outPtr = outputBuffer.Handle;
        float epsilon = 1e-7f;
        void** args = stackalloc void*[5];
        args[0] = &predPtr;
        args[1] = &targetPtr;
        args[2] = &outPtr;
        args[3] = &epsilon;
        args[4] = &size;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
        return Sum(outputBuffer, size) / size;
    }

    public unsafe void CategoricalCrossEntropyBackward(IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer gradInput, int size)
    {
        if (predictions is null) throw new ArgumentNullException(nameof(predictions));
        if (targets is null) throw new ArgumentNullException(nameof(targets));
        if (gradInput is null) throw new ArgumentNullException(nameof(gradInput));
        if (!_kernelCache.TryGetValue("categorical_cross_entropy_gradient", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: categorical_cross_entropy_gradient");

        using var _ = PushContext();
        uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr predPtr = predictions.Handle;
        IntPtr targetPtr = targets.Handle;
        IntPtr gradPtr = gradInput.Handle;
        float epsilon = 1e-7f;
        void** args = stackalloc void*[5];
        args[0] = &predPtr;
        args[1] = &targetPtr;
        args[2] = &gradPtr;
        args[3] = &epsilon;
        args[4] = &size;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    public unsafe float CharbonnierLoss(IGpuBuffer predictions, IGpuBuffer targets, int size, float epsilon)
    {
        if (predictions is null) throw new ArgumentNullException(nameof(predictions));
        if (targets is null) throw new ArgumentNullException(nameof(targets));
        if (!_kernelCache.TryGetValue("charbonnier_loss", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: charbonnier_loss");

        using var _ = PushContext();
        using var outputBuffer = AllocateBuffer(size);
        uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr predPtr = predictions.Handle;
        IntPtr targetPtr = targets.Handle;
        IntPtr outPtr = outputBuffer.Handle;
        void** args = stackalloc void*[5];
        args[0] = &predPtr;
        args[1] = &targetPtr;
        args[2] = &outPtr;
        args[3] = &epsilon;
        args[4] = &size;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
        return Sum(outputBuffer, size) / size;
    }

    public unsafe void CharbonnierBackward(IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer gradInput, int size, float epsilon)
    {
        if (predictions is null) throw new ArgumentNullException(nameof(predictions));
        if (targets is null) throw new ArgumentNullException(nameof(targets));
        if (gradInput is null) throw new ArgumentNullException(nameof(gradInput));
        if (!_kernelCache.TryGetValue("charbonnier_gradient", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: charbonnier_gradient");

        using var _ = PushContext();
        uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr predPtr = predictions.Handle;
        IntPtr targetPtr = targets.Handle;
        IntPtr gradPtr = gradInput.Handle;
        void** args = stackalloc void*[5];
        args[0] = &predPtr;
        args[1] = &targetPtr;
        args[2] = &gradPtr;
        args[3] = &epsilon;
        args[4] = &size;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    public unsafe float ElasticNetLoss(IGpuBuffer predictions, IGpuBuffer targets, int size, float l1Weight, float l2Weight)
    {
        if (predictions is null) throw new ArgumentNullException(nameof(predictions));
        if (targets is null) throw new ArgumentNullException(nameof(targets));
        if (!_kernelCache.TryGetValue("elastic_net_loss", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: elastic_net_loss");

        using var _ = PushContext();
        using var outputBuffer = AllocateBuffer(size);
        uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr predPtr = predictions.Handle;
        IntPtr targetPtr = targets.Handle;
        IntPtr outPtr = outputBuffer.Handle;
        void** args = stackalloc void*[6];
        args[0] = &predPtr;
        args[1] = &targetPtr;
        args[2] = &outPtr;
        args[3] = &l1Weight;
        args[4] = &l2Weight;
        args[5] = &size;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
        return Sum(outputBuffer, size) / size;
    }

    public unsafe void ElasticNetBackward(IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer gradInput, int size, float l1Weight, float l2Weight)
    {
        if (predictions is null) throw new ArgumentNullException(nameof(predictions));
        if (targets is null) throw new ArgumentNullException(nameof(targets));
        if (gradInput is null) throw new ArgumentNullException(nameof(gradInput));
        if (!_kernelCache.TryGetValue("elastic_net_gradient", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: elastic_net_gradient");

        using var _ = PushContext();
        uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr predPtr = predictions.Handle;
        IntPtr targetPtr = targets.Handle;
        IntPtr gradPtr = gradInput.Handle;
        void** args = stackalloc void*[6];
        args[0] = &predPtr;
        args[1] = &targetPtr;
        args[2] = &gradPtr;
        args[3] = &l1Weight;
        args[4] = &l2Weight;
        args[5] = &size;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    // Contrastive Loss
    public unsafe float ContrastiveLoss(IGpuBuffer output1, IGpuBuffer output2, IGpuBuffer labels, int batchSize, int embeddingDim, float margin)
    {
        if (output1 is null) throw new ArgumentNullException(nameof(output1));
        if (output2 is null) throw new ArgumentNullException(nameof(output2));
        if (labels is null) throw new ArgumentNullException(nameof(labels));
        if (!_kernelCache.TryGetValue("contrastive_loss", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: contrastive_loss");

        using var _ = PushContext();
        using var outputBuffer = AllocateBuffer(batchSize);
        uint grid = (uint)((batchSize + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr output1Ptr = output1.Handle;
        IntPtr output2Ptr = output2.Handle;
        IntPtr labelsPtr = labels.Handle;
        IntPtr outputPtr = outputBuffer.Handle;
        void** args = stackalloc void*[7];
        args[0] = &output1Ptr;
        args[1] = &output2Ptr;
        args[2] = &labelsPtr;
        args[3] = &outputPtr;
        args[4] = &batchSize;
        args[5] = &embeddingDim;
        args[6] = &margin;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
        return Sum(outputBuffer, batchSize) / batchSize;
    }

    public unsafe void ContrastiveBackward(IGpuBuffer output1, IGpuBuffer output2, IGpuBuffer labels,
        IGpuBuffer gradOutput1, IGpuBuffer gradOutput2,
        int batchSize, int embeddingDim, float margin)
    {
        if (output1 is null) throw new ArgumentNullException(nameof(output1));
        if (output2 is null) throw new ArgumentNullException(nameof(output2));
        if (labels is null) throw new ArgumentNullException(nameof(labels));
        if (gradOutput1 is null) throw new ArgumentNullException(nameof(gradOutput1));
        if (gradOutput2 is null) throw new ArgumentNullException(nameof(gradOutput2));
        if (!_kernelCache.TryGetValue("contrastive_loss_backward", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: contrastive_loss_backward");

        using var _ = PushContext();
        int totalSize = batchSize * embeddingDim;
        uint grid = (uint)((totalSize + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr output1Ptr = output1.Handle;
        IntPtr output2Ptr = output2.Handle;
        IntPtr labelsPtr = labels.Handle;
        IntPtr gradOutput1Ptr = gradOutput1.Handle;
        IntPtr gradOutput2Ptr = gradOutput2.Handle;
        void** args = stackalloc void*[8];
        args[0] = &output1Ptr;
        args[1] = &output2Ptr;
        args[2] = &labelsPtr;
        args[3] = &gradOutput1Ptr;
        args[4] = &gradOutput2Ptr;
        args[5] = &batchSize;
        args[6] = &embeddingDim;
        args[7] = &margin;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    #endregion


    #region Utility Operations

    public unsafe void Clamp(IGpuBuffer A, IGpuBuffer B, float min, float max, int size)
    {
        if (!_kernelCache.TryGetValue("clamp", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: clamp");

        using var _ = PushContext();
        uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr inputPtr = A.Handle;
        IntPtr outputPtr = B.Handle;
        void** args = stackalloc void*[5];
        args[0] = &inputPtr;
        args[1] = &outputPtr;
        args[2] = &min;
        args[3] = &max;
        args[4] = &size;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    public unsafe float L2Norm(IGpuBuffer A, int size)
    {
        if (!_kernelCache.TryGetValue("l2_norm_squared", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: l2_norm_squared");

        using var _ = PushContext();
        using var temp = AllocateBuffer(size);
        uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr inputPtr = A.Handle;
        IntPtr outputPtr = temp.Handle;
        void** args = stackalloc void*[3];
        args[0] = &inputPtr;
        args[1] = &outputPtr;
        args[2] = &size;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
        float sumSquared = Sum(temp, size);
        return (float)Math.Sqrt(sumSquared);
    }

    public void ClipByValue(IGpuBuffer A, IGpuBuffer B, float clipValue, int size)
    {
        Clamp(A, B, -clipValue, clipValue, size);
    }

    public void ClipByNorm(IGpuBuffer A, IGpuBuffer B, float maxNorm, int size)
    {
        float norm = L2Norm(A, size);
        if (norm > maxNorm)
        {
            float scale = maxNorm / norm;
            Scale(A, B, scale, size);
        }
        else
        {
            Copy(A, B, size);
        }
    }

    public unsafe void Fma(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, IGpuBuffer D, int size)
    {
        // D = A * B + C
        Multiply(A, B, D, size);
        Add(D, C, D, size);
    }

    public unsafe void Lerp(IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, float t, int size)
    {
        if (size <= 0)
            throw new ArgumentOutOfRangeException(nameof(size), size, "Size must be positive.");
        if (a.Size < size)
            throw new ArgumentException($"Buffer 'a' capacity ({a.Size}) is less than size ({size}).", nameof(a));
        if (b.Size < size)
            throw new ArgumentException($"Buffer 'b' capacity ({b.Size}) is less than size ({size}).", nameof(b));
        if (output.Size < size)
            throw new ArgumentException($"Buffer 'output' capacity ({output.Size}) is less than size ({size}).", nameof(output));

        if (!_kernelCache.TryGetValue("lerp_fused", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: lerp_fused");

        using var _ = PushContext();
        uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr aPtr = a.Handle;
        IntPtr bPtr = b.Handle;
        IntPtr outPtr = output.Handle;
        float tVal = t;
        int n = size;
        void** args = stackalloc void*[5];
        args[0] = &aPtr;
        args[1] = &bPtr;
        args[2] = &outPtr;
        args[3] = &tVal;
        args[4] = &n;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    public unsafe void AddScaled(IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, float scaleA, float scaleB, int size)
    {
        if (size <= 0)
            throw new ArgumentOutOfRangeException(nameof(size), size, "Size must be positive.");
        if (a.Size < size)
            throw new ArgumentException($"Buffer 'a' capacity ({a.Size}) is less than size ({size}).", nameof(a));
        if (b.Size < size)
            throw new ArgumentException($"Buffer 'b' capacity ({b.Size}) is less than size ({size}).", nameof(b));
        if (output.Size < size)
            throw new ArgumentException($"Buffer 'output' capacity ({output.Size}) is less than size ({size}).", nameof(output));

        if (!_kernelCache.TryGetValue("add_scaled", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: add_scaled");

        using var _ = PushContext();
        uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr aPtr = a.Handle;
        IntPtr bPtr = b.Handle;
        IntPtr outPtr = output.Handle;
        float sA = scaleA;
        float sB = scaleB;
        int n = size;
        void** args = stackalloc void*[6];
        args[0] = &aPtr;
        args[1] = &bPtr;
        args[2] = &outPtr;
        args[3] = &sA;
        args[4] = &sB;
        args[5] = &n;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    public unsafe float StdDev(IGpuBuffer input, int size)
    {
        if (size <= 0)
            throw new ArgumentOutOfRangeException(nameof(size), size, "Size must be positive.");
        if (input.Size < size)
            throw new ArgumentException($"Buffer 'input' capacity ({input.Size}) is less than size ({size}).", nameof(input));
        if (size <= 1) return 0.0f;

        using var _ = PushContext();
        int blockSize = DefaultBlockSize;
        int gridSize = (size + blockSize - 1) / blockSize;

        // Step 1: Compute mean via GPU reduction
        // Issue #382: under DeterministicMode, the deterministic variant computes
        // mean directly (writes sum/size) and uses a single-block grid; in non-
        // deterministic mode the multi-block atomic kernel writes sum (caller divides).
        float mean;
        string meanKernelName = GpuDeterminism.IsActive ? "reduce_mean_kernel_deterministic" : "reduce_mean_kernel";
        if (_kernelCache.TryGetValue(meanKernelName, out var meanKernel))
        {
            var zeroData = new float[1];
            using var meanBuffer = AllocateBuffer(zeroData);

            IntPtr inputPtr = input.Handle;
            IntPtr meanPtr = meanBuffer.Handle;
            int n = size;
            void** args = stackalloc void*[3];
            args[0] = &inputPtr;
            args[1] = &meanPtr;
            args[2] = &n;
            uint sharedBytes = (uint)(blockSize * sizeof(float));
            uint launchGrid = GpuDeterminism.IsActive ? 1u : (uint)gridSize;
            LaunchKernelWithSharedMem(meanKernel, launchGrid, (uint)blockSize, sharedBytes, args);
            // Synchronize() removed: DownloadBuffer uses cuMemcpyDtoH which is synchronous
            float[] meanResult = DownloadBuffer(meanBuffer);
            // Both kernels now write the raw sum; normalize once on the host so
            // deterministic and non-deterministic paths agree (CodeRabbit PR #390 —
            // previously the deterministic kernel divided internally and the host
            // skipped the division, but that was fragile and easy to mis-match).
            mean = meanResult[0] / size;
        }
        else
        {
            mean = Sum(input, size) / size;
        }

        // Step 2: Compute variance via GPU reduction
        float variance;
        string varKernelName = GpuDeterminism.IsActive ? "reduce_variance_kernel_deterministic" : "reduce_variance_kernel";
        if (_kernelCache.TryGetValue(varKernelName, out var varKernel))
        {
            var zeroData = new float[1];
            using var varianceBuffer = AllocateBuffer(zeroData);

            IntPtr inputPtr = input.Handle;
            IntPtr varPtr = varianceBuffer.Handle;
            float meanVal = mean;
            int n = size;
            void** args = stackalloc void*[4];
            args[0] = &inputPtr;
            args[1] = &varPtr;
            args[2] = &meanVal;
            args[3] = &n;
            uint sharedBytes = (uint)(blockSize * sizeof(float));
            uint launchGrid = GpuDeterminism.IsActive ? 1u : (uint)gridSize;
            LaunchKernelWithSharedMem(varKernel, launchGrid, (uint)blockSize, sharedBytes, args);
            // Synchronize() removed: DownloadBuffer uses cuMemcpyDtoH which is synchronous
            float[] varResult = DownloadBuffer(varianceBuffer);
            variance = varResult[0] / size;
        }
        else
        {
            // Fallback: download and compute on CPU
            float[] data = DownloadBuffer(input);
            float varSum = 0.0f;
            for (int i = 0; i < size; i++)
            {
                float diff = data[i] - mean;
                varSum += diff * diff;
            }
            variance = varSum / size;
        }

        // Clamp variance to avoid NaN from floating-point round-off
        variance = Math.Max(0, variance);
        return MathF.Sqrt(variance);
    }

    public unsafe void ScatterAdd(IGpuBuffer source, IGpuBuffer indices, IGpuBuffer destination, int sourceSize, int destSize)
    {
        // ScatterAdd shares the embedding_backward atomic kernel — same op with
        // embDim = 1 (each input is a single value), accumulating onto destination.
        //
        // We do NOT route to embedding_backward_deterministic here even when
        // GpuDeterminism.IsActive: that kernel was designed for the embedding
        // backward path and uses PLAIN ASSIGNMENT (`gradEmbedding[v,d] = sum`)
        // — overwrite, not accumulate — because its caller passes a freshly-
        // allocated zero-init buffer. ScatterAdd's wrapper instead SEEDS the
        // destination with the base tensor via backend.Copy (the `+=` target),
        // so an overwriting kernel would wipe out the seed and return just the
        // sum of the indexed contributions — diverging from CPU by exactly the
        // destination magnitude (~1.74 absolute error in the random-tensor
        // test, GpuCpuAutoDifferentialTests post-PR-#582).
        // The atomic-add kernel is still correct under determinism for THIS
        // use case (scatter-add's result is order-independent when reduced to
        // a single sum per cell — duplicate indices commute up to fp32
        // associativity, which is the same caveat the deterministic variant
        // was meant to dodge for embedding-backward, not scatter-add).
        using var _ = PushContext();
        IntPtr srcPtr = source.Handle;
        IntPtr idxPtr = indices.Handle;
        IntPtr dstPtr = destination.Handle;
        int embDim = 1;

        if (!_kernelCache.TryGetValue("embedding_backward", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: embedding_backward");

        uint grid = (uint)((sourceSize + DefaultBlockSize - 1) / DefaultBlockSize);
        void** args = stackalloc void*[5];
        args[0] = &srcPtr;
        args[1] = &idxPtr;
        args[2] = &dstPtr;
        args[3] = &sourceSize;
        args[4] = &embDim;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    public unsafe void ScatterAddBackward(IGpuBuffer gradDestination, IGpuBuffer indices, IGpuBuffer gradSource,
        int numIndices, int featureSize)
    {
        // ScatterAddBackward is essentially a Gather operation
        if (!_kernelCache.TryGetValue("embedding_forward", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: embedding_forward");

        using var _ = PushContext();
        uint grid = (uint)((numIndices + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr idxPtr = indices.Handle;
        IntPtr gradDstPtr = gradDestination.Handle;
        IntPtr gradSrcPtr = gradSource.Handle;
        void** args = stackalloc void*[5];
        args[0] = &idxPtr;
        args[1] = &gradDstPtr;
        args[2] = &gradSrcPtr;
        args[3] = &numIndices;
        args[4] = &featureSize;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    public unsafe void Gather(IGpuBuffer source, IGpuBuffer indices, IGpuBuffer output, int numIndices, int featureSize)
    {
        if (!_kernelCache.TryGetValue("embedding_forward", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: embedding_forward");

        using var _ = PushContext();
        uint grid = (uint)((numIndices + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr idxPtr = indices.Handle;
        IntPtr srcPtr = source.Handle;
        IntPtr outPtr = output.Handle;
        void** args = stackalloc void*[5];
        args[0] = &idxPtr;
        args[1] = &srcPtr;
        args[2] = &outPtr;
        args[3] = &numIndices;
        args[4] = &featureSize;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    #endregion


    #region Comparison Operations

    public unsafe void GreaterThan(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int size)
    {
        if (!_kernelCache.TryGetValue("greater_than", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: greater_than");

        using var _ = PushContext();
        uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr aPtr = A.Handle;
        IntPtr bPtr = B.Handle;
        IntPtr cPtr = C.Handle;
        void** args = stackalloc void*[4];
        args[0] = &aPtr;
        args[1] = &bPtr;
        args[2] = &cPtr;
        args[3] = &size;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    public unsafe void LessThan(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int size)
    {
        if (!_kernelCache.TryGetValue("less_than", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: less_than");

        using var _ = PushContext();
        uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr aPtr = A.Handle;
        IntPtr bPtr = B.Handle;
        IntPtr cPtr = C.Handle;
        void** args = stackalloc void*[4];
        args[0] = &aPtr;
        args[1] = &bPtr;
        args[2] = &cPtr;
        args[3] = &size;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    // IEEE classify (isnan/isinf/isfinite): a real CUDA kernel (bit-pattern, like the OpenCL
    // classify_float) is a HW-validation follow-up; the engine catches this and falls back to the
    // correct CPU path until then.
    public unsafe void ClassifyFloat(IGpuBuffer A, IGpuBuffer C, int mode, int size)
    {
        var kernel = ResolveParity210Kernel("parity210_classify_float");
        using var _ = PushContext();
        int __total = size; if (__total <= 0) return;
        uint gsz = (uint)((__total + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr la0 = A.Handle; IntPtr la1 = C.Handle; int la2 = mode; int la3 = size;
        void** args = stackalloc void*[4];
        args[0] = &la0; args[1] = &la1; args[2] = &la2; args[3] = &la3;
        LaunchKernel(kernel, gsz, DefaultBlockSize, args);
    }
    public unsafe void TakeAlongDim(IGpuBuffer input, IGpuBuffer indices, IGpuBuffer output, int outerSize, int axisOut, int innerSize, int axisIn)
    {
        var kernel = ResolveParity210Kernel("parity210_take_along_dim_f");
        using var _ = PushContext();
        int __total = outerSize*axisOut*innerSize; if (__total <= 0) return;
        uint gsz = (uint)((__total + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr la0 = input.Handle; IntPtr la1 = indices.Handle; IntPtr la2 = output.Handle; int la3 = outerSize; int la4 = axisOut; int la5 = innerSize; int la6 = axisIn;
        void** args = stackalloc void*[7];
        args[0] = &la0; args[1] = &la1; args[2] = &la2; args[3] = &la3; args[4] = &la4; args[5] = &la5; args[6] = &la6;
        LaunchKernel(kernel, gsz, DefaultBlockSize, args);
    }
    public unsafe void Cross3(IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, int outerSize, int innerSize)
    {
        var kernel = ResolveParity210Kernel("parity210_cross3");
        using var _ = PushContext();
        int __total = outerSize*innerSize; if (__total <= 0) return;
        uint gsz = (uint)((__total + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr la0 = a.Handle; IntPtr la1 = b.Handle; IntPtr la2 = output.Handle; int la3 = outerSize; int la4 = innerSize;
        void** args = stackalloc void*[5];
        args[0] = &la0; args[1] = &la1; args[2] = &la2; args[3] = &la3; args[4] = &la4;
        LaunchKernel(kernel, gsz, DefaultBlockSize, args);
    }
    public unsafe void Ldexp(IGpuBuffer input, IGpuBuffer exponents, IGpuBuffer output, int size)
    {
        var kernel = ResolveParity210Kernel("parity210_ldexp");
        using var _ = PushContext();
        int __total = size; if (__total <= 0) return;
        uint gsz = (uint)((__total + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr la0 = input.Handle; IntPtr la1 = exponents.Handle; IntPtr la2 = output.Handle; int la3 = size;
        void** args = stackalloc void*[4];
        args[0] = &la0; args[1] = &la1; args[2] = &la2; args[3] = &la3;
        LaunchKernel(kernel, gsz, DefaultBlockSize, args);
    }
    public unsafe void Kron2D(IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, int am, int an, int bp, int bq)
    {
        var kernel = ResolveParity210Kernel("parity210_kron2d");
        using var _ = PushContext();
        int __total = (am*bp)*(an*bq); if (__total <= 0) return;
        uint gsz = (uint)((__total + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr la0 = a.Handle; IntPtr la1 = b.Handle; IntPtr la2 = output.Handle; int la3 = am; int la4 = an; int la5 = bp; int la6 = bq;
        void** args = stackalloc void*[7];
        args[0] = &la0; args[1] = &la1; args[2] = &la2; args[3] = &la3; args[4] = &la4; args[5] = &la5; args[6] = &la6;
        LaunchKernel(kernel, gsz, DefaultBlockSize, args);
    }
    public unsafe void SearchSorted(IGpuBuffer sortedSeq, IGpuBuffer values, IGpuBuffer output, int seqLen, int numValues, int right)
    {
        var kernel = ResolveParity210Kernel("parity210_search_sorted");
        using var _ = PushContext();
        int __total = numValues; if (__total <= 0) return;
        uint gsz = (uint)((__total + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr la0 = sortedSeq.Handle; IntPtr la1 = values.Handle; IntPtr la2 = output.Handle; int la3 = seqLen; int la4 = numValues; int la5 = right;
        void** args = stackalloc void*[6];
        args[0] = &la0; args[1] = &la1; args[2] = &la2; args[3] = &la3; args[4] = &la4; args[5] = &la5;
        LaunchKernel(kernel, gsz, DefaultBlockSize, args);
    }
    public unsafe void NextAfter(IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, int size)
    {
        var kernel = ResolveParity210Kernel("parity210_next_after");
        using var _ = PushContext();
        int __total = size; if (__total <= 0) return;
        uint gsz = (uint)((__total + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr la0 = a.Handle; IntPtr la1 = b.Handle; IntPtr la2 = output.Handle; int la3 = size;
        void** args = stackalloc void*[4];
        args[0] = &la0; args[1] = &la1; args[2] = &la2; args[3] = &la3;
        LaunchKernel(kernel, gsz, DefaultBlockSize, args);
    }
    public unsafe void IndexWrite(IGpuBuffer output, IGpuBuffer indices, IGpuBuffer source, float fillValue, int mode, int outerSize, int idxAxis, int innerSize, int dstAxis)
    {
        var kernel = ResolveParity210Kernel("parity210_index_write");
        using var _ = PushContext();
        int __total = outerSize*idxAxis*innerSize; if (__total <= 0) return;
        uint gsz = (uint)((__total + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr la0 = output.Handle; IntPtr la1 = indices.Handle; IntPtr la2 = source.Handle; float la3 = fillValue; int la4 = mode; int la5 = outerSize; int la6 = idxAxis; int la7 = innerSize; int la8 = dstAxis;
        void** args = stackalloc void*[9];
        args[0] = &la0; args[1] = &la1; args[2] = &la2; args[3] = &la3; args[4] = &la4; args[5] = &la5; args[6] = &la6; args[7] = &la7; args[8] = &la8;
        LaunchKernel(kernel, gsz, DefaultBlockSize, args);
    }
    public unsafe void CDist(IGpuBuffer x1, IGpuBuffer x2, IGpuBuffer output, int m, int n, int d, float p)
    {
        var kernel = ResolveParity210Kernel("parity210_cdist");
        using var _ = PushContext();
        int __total = m*n; if (__total <= 0) return;
        uint gsz = (uint)((__total + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr la0 = x1.Handle; IntPtr la1 = x2.Handle; IntPtr la2 = output.Handle; int la3 = m; int la4 = n; int la5 = d; float la6 = p;
        void** args = stackalloc void*[7];
        args[0] = &la0; args[1] = &la1; args[2] = &la2; args[3] = &la3; args[4] = &la4; args[5] = &la5; args[6] = &la6;
        LaunchKernel(kernel, gsz, DefaultBlockSize, args);
    }
    public unsafe void PDist(IGpuBuffer input, IGpuBuffer output, int n, int d, float p)
    {
        var kernel = ResolveParity210Kernel("parity210_pdist");
        using var _ = PushContext();
        int __total = n*n; if (__total <= 0) return;
        uint gsz = (uint)((__total + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr la0 = input.Handle; IntPtr la1 = output.Handle; int la2 = n; int la3 = d; float la4 = p;
        void** args = stackalloc void*[5];
        args[0] = &la0; args[1] = &la1; args[2] = &la2; args[3] = &la3; args[4] = &la4;
        LaunchKernel(kernel, gsz, DefaultBlockSize, args);
    }
    public unsafe void Histc(IGpuBuffer input, IGpuBuffer hist, int n, int bins, float mn, float mx)
    {
        var kernel = ResolveParity210Kernel("parity210_histc");
        using var _ = PushContext();
        int __total = n; if (__total <= 0) return;
        uint gsz = (uint)((__total + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr la0 = input.Handle; IntPtr la1 = hist.Handle; int la2 = n; int la3 = bins; float la4 = mn; float la5 = mx;
        void** args = stackalloc void*[6];
        args[0] = &la0; args[1] = &la1; args[2] = &la2; args[3] = &la3; args[4] = &la4; args[5] = &la5;
        LaunchKernel(kernel, gsz, DefaultBlockSize, args);
    }
    public unsafe void BitonicStep(IGpuBuffer values, IGpuBuffer indices, int rowLen, int k, int j, int numRows, int descending)
    {
        var kernel = ResolveParity210Kernel("parity210_bitonic_step");
        using var _ = PushContext();
        int __total = numRows*rowLen; if (__total <= 0) return;
        uint gsz = (uint)((__total + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr la0 = values.Handle; IntPtr la1 = indices.Handle; int la2 = rowLen; int la3 = k; int la4 = j; int la5 = numRows; int la6 = descending;
        void** args = stackalloc void*[7];
        args[0] = &la0; args[1] = &la1; args[2] = &la2; args[3] = &la3; args[4] = &la4; args[5] = &la5; args[6] = &la6;
        LaunchKernel(kernel, gsz, DefaultBlockSize, args);
    }
    public unsafe void CopyRows(IGpuBuffer src, IGpuBuffer dst, int srcRowLen, int dstRowLen, int numRows, int copyLen)
    {
        var kernel = ResolveParity210Kernel("parity210_copy_rows");
        using var _ = PushContext();
        int __total = numRows*copyLen; if (__total <= 0) return;
        uint gsz = (uint)((__total + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr la0 = src.Handle; IntPtr la1 = dst.Handle; int la2 = srcRowLen; int la3 = dstRowLen; int la4 = numRows; int la5 = copyLen;
        void** args = stackalloc void*[6];
        args[0] = &la0; args[1] = &la1; args[2] = &la2; args[3] = &la3; args[4] = &la4; args[5] = &la5;
        LaunchKernel(kernel, gsz, DefaultBlockSize, args);
    }
    public unsafe void IotaPad(IGpuBuffer idx, int l, int p, int numRows)
    {
        var kernel = ResolveParity210Kernel("parity210_iota_pad");
        using var _ = PushContext();
        int __total = numRows*p; if (__total <= 0) return;
        uint gsz = (uint)((__total + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr la0 = idx.Handle; int la1 = l; int la2 = p; int la3 = numRows;
        void** args = stackalloc void*[4];
        args[0] = &la0; args[1] = &la1; args[2] = &la2; args[3] = &la3;
        LaunchKernel(kernel, gsz, DefaultBlockSize, args);
    }
    public unsafe void Rwkv7Forward(IGpuBuffer r, IGpuBuffer k, IGpuBuffer v, IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, IGpuBuffer sbuf, int batch, int seqLen, int modelDim, int numHeads, int headDim)
    {
        var kernel = ResolveParity210Kernel("parity210_rwkv7_forward");
        using var _ = PushContext();
        int __total = batch*numHeads; if (__total <= 0) return;
        uint gsz = (uint)((__total + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr la0 = r.Handle; IntPtr la1 = k.Handle; IntPtr la2 = v.Handle; IntPtr la3 = a.Handle; IntPtr la4 = b.Handle; IntPtr la5 = output.Handle; IntPtr la6 = sbuf.Handle; int la7 = batch; int la8 = seqLen; int la9 = modelDim; int la10 = numHeads; int la11 = headDim;
        void** args = stackalloc void*[12];
        args[0] = &la0; args[1] = &la1; args[2] = &la2; args[3] = &la3; args[4] = &la4; args[5] = &la5; args[6] = &la6; args[7] = &la7; args[8] = &la8; args[9] = &la9; args[10] = &la10; args[11] = &la11;
        LaunchKernel(kernel, gsz, DefaultBlockSize, args);
    }
    public unsafe void HierarchicalSoftmaxPaths(IGpuBuffer acts, IGpuBuffer output, int rows, int treeDepth, int numClasses)
    {
        var kernel = ResolveParity210Kernel("parity210_hsoftmax_paths");
        using var _ = PushContext();
        int __total = rows*numClasses; if (__total <= 0) return;
        uint gsz = (uint)((__total + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr la0 = acts.Handle; IntPtr la1 = output.Handle; int la2 = rows; int la3 = treeDepth; int la4 = numClasses;
        void** args = stackalloc void*[5];
        args[0] = &la0; args[1] = &la1; args[2] = &la2; args[3] = &la3; args[4] = &la4;
        LaunchKernel(kernel, gsz, DefaultBlockSize, args);
    }
    public unsafe void IsIn(IGpuBuffer elements, IGpuBuffer sortedTest, IGpuBuffer mask, int numElements, int testLen)
    {
        var kernel = ResolveParity210Kernel("parity210_isin");
        using var _ = PushContext();
        int __total = numElements; if (__total <= 0) return;
        uint gsz = (uint)((__total + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr la0 = elements.Handle; IntPtr la1 = sortedTest.Handle; IntPtr la2 = mask.Handle; int la3 = numElements; int la4 = testLen;
        void** args = stackalloc void*[5];
        args[0] = &la0; args[1] = &la1; args[2] = &la2; args[3] = &la3; args[4] = &la4;
        LaunchKernel(kernel, gsz, DefaultBlockSize, args);
    }
    public unsafe void CopyBlock2D(IGpuBuffer block, IGpuBuffer output, int blockRows, int blockCols, int totalCols, int rowOff, int colOff)
    {
        var kernel = ResolveParity210Kernel("parity210_copy_block_2d");
        using var _ = PushContext();
        int __total = blockRows*blockCols; if (__total <= 0) return;
        uint gsz = (uint)((__total + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr la0 = block.Handle; IntPtr la1 = output.Handle; int la2 = blockRows; int la3 = blockCols; int la4 = totalCols; int la5 = rowOff; int la6 = colOff;
        void** args = stackalloc void*[7];
        args[0] = &la0; args[1] = &la1; args[2] = &la2; args[3] = &la3; args[4] = &la4; args[5] = &la5; args[6] = &la6;
        LaunchKernel(kernel, gsz, DefaultBlockSize, args);
    }
    public unsafe void Zeta(IGpuBuffer x, IGpuBuffer q, IGpuBuffer output, int size)
    {
        var kernel = ResolveParity210Kernel("parity210_zeta");
        using var _ = PushContext();
        int __total = size; if (__total <= 0) return;
        uint gsz = (uint)((__total + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr la0 = x.Handle; IntPtr la1 = q.Handle; IntPtr la2 = output.Handle; int la3 = size;
        void** args = stackalloc void*[4];
        args[0] = &la0; args[1] = &la1; args[2] = &la2; args[3] = &la3;
        LaunchKernel(kernel, gsz, DefaultBlockSize, args);
    }
    public unsafe void Polygamma(IGpuBuffer x, IGpuBuffer output, int n, int size)
    {
        var kernel = ResolveParity210Kernel("parity210_polygamma");
        using var _ = PushContext();
        int __total = size; if (__total <= 0) return;
        uint gsz = (uint)((__total + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr la0 = x.Handle; IntPtr la1 = output.Handle; int la2 = n; int la3 = size;
        void** args = stackalloc void*[4];
        args[0] = &la0; args[1] = &la1; args[2] = &la2; args[3] = &la3;
        LaunchKernel(kernel, gsz, DefaultBlockSize, args);
    }
    public unsafe void ShiftedDiff(IGpuBuffer x, IGpuBuffer mask, int n)
    {
        var kernel = ResolveParity210Kernel("parity210_shifted_diff");
        using var _ = PushContext();
        int __total = n; if (__total <= 0) return;
        uint grid = (uint)((__total + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr la0 = x.Handle; IntPtr la1 = mask.Handle; int la2 = n;
        void** args = stackalloc void*[3];
        args[0] = &la0; args[1] = &la1; args[2] = &la2;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }
    public unsafe void ReflectPad1d(IGpuBuffer input, IGpuBuffer output, int batch, int l, int lp, int pad)
    {
        var kernel = ResolveParity210Kernel("parity210_reflect_pad_1d");
        using var _ = PushContext();
        int __total = batch*lp; if (__total <= 0) return;
        uint grid = (uint)((__total + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr la0 = input.Handle; IntPtr la1 = output.Handle; int la2 = batch; int la3 = l; int la4 = lp; int la5 = pad;
        void** args = stackalloc void*[6];
        args[0] = &la0; args[1] = &la1; args[2] = &la2; args[3] = &la3; args[4] = &la4; args[5] = &la5;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }
    public unsafe void StftMagPhase(IGpuBuffer padded, IGpuBuffer window, IGpuBuffer mag, IGpuBuffer phase, int batch, int lp, int nFft, int hop, int numFrames, int numFreqs)
    {
        var kernel = ResolveParity210Kernel("parity210_stft_mag_phase");
        using var _ = PushContext();
        int __total = batch*numFreqs*numFrames; if (__total <= 0) return;
        uint grid = (uint)((__total + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr la0 = padded.Handle; IntPtr la1 = window.Handle; IntPtr la2 = mag.Handle; IntPtr la3 = phase.Handle; int la4 = batch; int la5 = lp; int la6 = nFft; int la7 = hop; int la8 = numFrames; int la9 = numFreqs;
        void** args = stackalloc void*[10];
        args[0] = &la0; args[1] = &la1; args[2] = &la2; args[3] = &la3; args[4] = &la4; args[5] = &la5; args[6] = &la6; args[7] = &la7; args[8] = &la8; args[9] = &la9;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }
    public unsafe void PhaseVocoder(IGpuBuffer mag, IGpuBuffer phase, IGpuBuffer newMag, IGpuBuffer newPhase, int leading, int nFramesV, int nFreqV, int outFrames, float rate)
    {
        var kernel = ResolveParity210Kernel("parity210_phase_vocoder");
        using var _ = PushContext();
        int __total = leading*nFreqV; if (__total <= 0) return;
        uint grid = (uint)((__total + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr la0 = mag.Handle; IntPtr la1 = phase.Handle; IntPtr la2 = newMag.Handle; IntPtr la3 = newPhase.Handle; int la4 = leading; int la5 = nFramesV; int la6 = nFreqV; int la7 = outFrames; float la8 = rate;
        void** args = stackalloc void*[9];
        args[0] = &la0; args[1] = &la1; args[2] = &la2; args[3] = &la3; args[4] = &la4; args[5] = &la5; args[6] = &la6; args[7] = &la7; args[8] = &la8;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }
    public unsafe void BuildSpectrum(IGpuBuffer mag, IGpuBuffer phase, IGpuBuffer specRe, IGpuBuffer specIm, int batch, int numFreqs, int numFrames, int nFft)
    {
        var kernel = ResolveParity210Kernel("parity210_build_spectrum");
        using var _ = PushContext();
        int __total = batch*numFrames; if (__total <= 0) return;
        uint grid = (uint)((__total + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr la0 = mag.Handle; IntPtr la1 = phase.Handle; IntPtr la2 = specRe.Handle; IntPtr la3 = specIm.Handle; int la4 = batch; int la5 = numFreqs; int la6 = numFrames; int la7 = nFft;
        void** args = stackalloc void*[8];
        args[0] = &la0; args[1] = &la1; args[2] = &la2; args[3] = &la3; args[4] = &la4; args[5] = &la5; args[6] = &la6; args[7] = &la7;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }
    public unsafe void IstftFromSpectrum(IGpuBuffer specRe, IGpuBuffer specIm, IGpuBuffer window, IGpuBuffer result, IGpuBuffer windowSum, int batch, int numFrames, int nFft, int hop, int outputLength, int center)
    {
        var kernel = ResolveParity210Kernel("parity210_istft_from_spectrum");
        using var _ = PushContext();
        int __total = batch*numFrames*nFft; if (__total <= 0) return;
        uint grid = (uint)((__total + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr la0 = specRe.Handle; IntPtr la1 = specIm.Handle; IntPtr la2 = window.Handle; IntPtr la3 = result.Handle; IntPtr la4 = windowSum.Handle; int la5 = batch; int la6 = numFrames; int la7 = nFft; int la8 = hop; int la9 = outputLength; int la10 = center;
        void** args = stackalloc void*[11];
        args[0] = &la0; args[1] = &la1; args[2] = &la2; args[3] = &la3; args[4] = &la4; args[5] = &la5; args[6] = &la6; args[7] = &la7; args[8] = &la8; args[9] = &la9; args[10] = &la10;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }
    public unsafe void IstftNormalize(IGpuBuffer result, IGpuBuffer windowSum, int total)
    {
        var kernel = ResolveParity210Kernel("parity210_istft_normalize");
        using var _ = PushContext();
        int __total = total; if (__total <= 0) return;
        uint grid = (uint)((__total + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr la0 = result.Handle; IntPtr la1 = windowSum.Handle; int la2 = total;
        void** args = stackalloc void*[3];
        args[0] = &la0; args[1] = &la1; args[2] = &la2;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }
    public unsafe void HistogramDD(IGpuBuffer samples, IGpuBuffer hist, IGpuBuffer bins, IGpuBuffer mins, IGpuBuffer maxs, int n, int d)
    {
        var kernel = ResolveParity210Kernel("parity210_histogramdd");
        using var _ = PushContext();
        int __total = n; if (__total <= 0) return;
        uint grid = (uint)((__total + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr la0 = samples.Handle; IntPtr la1 = hist.Handle; IntPtr la2 = bins.Handle; IntPtr la3 = mins.Handle; IntPtr la4 = maxs.Handle; int la5 = n; int la6 = d;
        void** args = stackalloc void*[7];
        args[0] = &la0; args[1] = &la1; args[2] = &la2; args[3] = &la3; args[4] = &la4; args[5] = &la5; args[6] = &la6;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }
    public unsafe void MasksToBoxes(IGpuBuffer masks, IGpuBuffer output, int n, int h, int w)
    {
        var kernel = ResolveParity210Kernel("parity210_masks_to_boxes");
        using var _ = PushContext();
        int __total = n; if (__total <= 0) return;
        uint grid = (uint)((__total + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr la0 = masks.Handle; IntPtr la1 = output.Handle; int la2 = n; int la3 = h; int la4 = w;
        void** args = stackalloc void*[5];
        args[0] = &la0; args[1] = &la1; args[2] = &la2; args[3] = &la3; args[4] = &la4;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }
    public unsafe void PairwiseIou(IGpuBuffer boxes, IGpuBuffer iou, int n)
    {
        var kernel = ResolveParity210Kernel("parity210_pairwise_iou");
        using var _ = PushContext();
        int __total = n*n; if (__total <= 0) return;
        uint grid = (uint)((__total + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr la0 = boxes.Handle; IntPtr la1 = iou.Handle; int la2 = n;
        void** args = stackalloc void*[3];
        args[0] = &la0; args[1] = &la1; args[2] = &la2;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }
    public unsafe void LogicalOp(IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, int mode, int n)
    {
        var kernel = ResolveParity210Kernel("parity210_logical_op");
        using var _ = PushContext();
        int __total = n; if (__total <= 0) return;
        uint grid = (uint)((__total + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr la0 = a.Handle; IntPtr la1 = b.Handle; IntPtr la2 = output.Handle; int la3 = mode; int la4 = n;
        void** args = stackalloc void*[5];
        args[0] = &la0; args[1] = &la1; args[2] = &la2; args[3] = &la3; args[4] = &la4;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }
    public unsafe void LogicalNot(IGpuBuffer a, IGpuBuffer output, int n)
    {
        var kernel = ResolveParity210Kernel("parity210_logical_not");
        using var _ = PushContext();
        int __total = n; if (__total <= 0) return;
        uint grid = (uint)((__total + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr la0 = a.Handle; IntPtr la1 = output.Handle; int la2 = n;
        void** args = stackalloc void*[3];
        args[0] = &la0; args[1] = &la1; args[2] = &la2;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }
    public unsafe void GridSampleBackwardInputNhwc(IGpuBuffer gradOut, IGpuBuffer grid, IGpuBuffer gradIn, int batch, int h, int w, int c, int outH, int outW)
    {
        var kernel = ResolveParity210Kernel("parity210_gridsample_backward_input");
        using var _ = PushContext();
        int __total = batch*outH*outW*c; if (__total <= 0) return;
        uint gsz = (uint)((__total + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr la0 = gradOut.Handle; IntPtr la1 = grid.Handle; IntPtr la2 = gradIn.Handle; int la3 = batch; int la4 = h; int la5 = w; int la6 = c; int la7 = outH; int la8 = outW;
        void** args = stackalloc void*[9];
        args[0] = &la0; args[1] = &la1; args[2] = &la2; args[3] = &la3; args[4] = &la4; args[5] = &la5; args[6] = &la6; args[7] = &la7; args[8] = &la8;
        LaunchKernel(kernel, gsz, DefaultBlockSize, args);
    }
    public unsafe void GridSampleBackwardGridNhwc(IGpuBuffer gradOut, IGpuBuffer input, IGpuBuffer grid, IGpuBuffer gradGrid, int batch, int h, int w, int c, int outH, int outW)
    {
        var kernel = ResolveParity210Kernel("parity210_gridsample_backward_grid");
        using var _ = PushContext();
        int __total = batch*outH*outW; if (__total <= 0) return;
        uint gsz = (uint)((__total + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr la0 = gradOut.Handle; IntPtr la1 = input.Handle; IntPtr la2 = grid.Handle; IntPtr la3 = gradGrid.Handle; int la4 = batch; int la5 = h; int la6 = w; int la7 = c; int la8 = outH; int la9 = outW;
        void** args = stackalloc void*[10];
        args[0] = &la0; args[1] = &la1; args[2] = &la2; args[3] = &la3; args[4] = &la4; args[5] = &la5; args[6] = &la6; args[7] = &la7; args[8] = &la8; args[9] = &la9;
        LaunchKernel(kernel, gsz, DefaultBlockSize, args);
    }
    public unsafe void ScatterReduce(IGpuBuffer output, IGpuBuffer source, IGpuBuffer index, int outerSize, int srcDim, int dstDim, int innerSize, int mode)
    {
        var kernel = ResolveParity210Kernel("parity210_scatter_reduce");
        using var _ = PushContext();
        int __total = outerSize*srcDim*innerSize; if (__total <= 0) return;
        uint gsz = (uint)((__total + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr la0 = output.Handle; IntPtr la1 = source.Handle; IntPtr la2 = index.Handle; int la3 = outerSize; int la4 = srcDim; int la5 = dstDim; int la6 = innerSize; int la7 = mode;
        void** args = stackalloc void*[8];
        args[0] = &la0; args[1] = &la1; args[2] = &la2; args[3] = &la3; args[4] = &la4; args[5] = &la5; args[6] = &la6; args[7] = &la7;
        LaunchKernel(kernel, gsz, DefaultBlockSize, args);
    }
    public unsafe void Unfold(IGpuBuffer src, IGpuBuffer dst, int outerSize, int dimSize, int innerSize, int nWindows, int size, int step)
    {
        var kernel = ResolveParity210Kernel("parity210_unfold");
        using var _ = PushContext();
        int __total = outerSize*nWindows*innerSize*size; if (__total <= 0) return;
        uint gsz = (uint)((__total + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr la0 = src.Handle; IntPtr la1 = dst.Handle; int la2 = outerSize; int la3 = dimSize; int la4 = innerSize; int la5 = nWindows; int la6 = size; int la7 = step;
        void** args = stackalloc void*[8];
        args[0] = &la0; args[1] = &la1; args[2] = &la2; args[3] = &la3; args[4] = &la4; args[5] = &la5; args[6] = &la6; args[7] = &la7;
        LaunchKernel(kernel, gsz, DefaultBlockSize, args);
    }

    public unsafe void Equal(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int size)
    {
        if (!_kernelCache.TryGetValue("equal", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: equal");

        using var _ = PushContext();
        uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr aPtr = A.Handle;
        IntPtr bPtr = B.Handle;
        IntPtr cPtr = C.Handle;
        void** args = stackalloc void*[4];
        args[0] = &aPtr;
        args[1] = &bPtr;
        args[2] = &cPtr;
        args[3] = &size;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    public unsafe void Where(IGpuBuffer condition, IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int size)
    {
        if (!_kernelCache.TryGetValue("where_select", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: where_select");

        using var _ = PushContext();
        uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr condPtr = condition.Handle;
        IntPtr aPtr = A.Handle;
        IntPtr bPtr = B.Handle;
        IntPtr cPtr = C.Handle;
        void** args = stackalloc void*[5];
        args[0] = &condPtr;
        args[1] = &aPtr;
        args[2] = &bPtr;
        args[3] = &cPtr;
        args[4] = &size;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    public unsafe void NotEqualScalar(IGpuBuffer A, IGpuBuffer C, float scalar, int size)
    {
        if (!_kernelCache.TryGetValue("not_equal_scalar", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: not_equal_scalar");

        using var _ = PushContext();
        uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr aPtr = A.Handle;
        IntPtr cPtr = C.Handle;
        void** args = stackalloc void*[4];
        args[0] = &aPtr;
        args[1] = &cPtr;
        args[2] = &scalar;
        args[3] = &size;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    #endregion


    #region Statistics Operations

    public unsafe void MeanAxis(IGpuBuffer A, IGpuBuffer B, int outerSize, int reduceSize)
    {
        if (!_kernelCache.TryGetValue("mean_axis", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: mean_axis");

        using var _ = PushContext();
        uint grid = (uint)((outerSize + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr aPtr = A.Handle;
        IntPtr bPtr = B.Handle;
        void** args = stackalloc void*[4];
        args[0] = &aPtr;
        args[1] = &bPtr;
        args[2] = &outerSize;
        args[3] = &reduceSize;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    public unsafe void VarAxis(IGpuBuffer A, IGpuBuffer mean, IGpuBuffer variance, int outerSize, int reduceSize)
    {
        if (!_kernelCache.TryGetValue("var_axis", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: var_axis");

        using var _ = PushContext();
        uint grid = (uint)((outerSize + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr aPtr = A.Handle;
        IntPtr meanPtr = mean.Handle;
        IntPtr varPtr = variance.Handle;
        void** args = stackalloc void*[5];
        args[0] = &aPtr;
        args[1] = &meanPtr;
        args[2] = &varPtr;
        args[3] = &outerSize;
        args[4] = &reduceSize;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    public unsafe void ArgMax(IGpuBuffer A, IGpuBuffer indices, int outerSize, int reduceSize)
    {
        if (!_kernelCache.TryGetValue("argmax", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: argmax");

        using var _ = PushContext();
        uint grid = (uint)((outerSize + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr aPtr = A.Handle;
        IntPtr idxPtr = indices.Handle;
        void** args = stackalloc void*[4];
        args[0] = &aPtr;
        args[1] = &idxPtr;
        args[2] = &outerSize;
        args[3] = &reduceSize;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    public unsafe void ArgMin(IGpuBuffer A, IGpuBuffer indices, int outerSize, int reduceSize)
    {
        if (!_kernelCache.TryGetValue("argmin", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: argmin");

        using var _ = PushContext();
        uint grid = (uint)((outerSize + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr aPtr = A.Handle;
        IntPtr idxPtr = indices.Handle;
        void** args = stackalloc void*[4];
        args[0] = &aPtr;
        args[1] = &idxPtr;
        args[2] = &outerSize;
        args[3] = &reduceSize;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    public unsafe void MaxAxis(IGpuBuffer A, IGpuBuffer B, int outerSize, int reduceSize)
    {
        if (!_kernelCache.TryGetValue("max_axis", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: max_axis");

        using var _ = PushContext();
        uint grid = (uint)((outerSize + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr aPtr = A.Handle;
        IntPtr bPtr = B.Handle;
        void** args = stackalloc void*[4];
        args[0] = &aPtr;
        args[1] = &bPtr;
        args[2] = &outerSize;
        args[3] = &reduceSize;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    public unsafe void TopK(IGpuBuffer A, IGpuBuffer values, IGpuBuffer indices, int outerSize, int reduceSize, int k, bool sorted = true)
    {
        // Use optimized small-k kernel for k <= 8
        string kernelName = k <= 8 ? "topk_small" : "topk";
        if (!_kernelCache.TryGetValue(kernelName, out var kernel))
            throw new InvalidOperationException($"CUDA kernel not found: {kernelName}");

        using var _ = PushContext();
        // One block per row
        uint gridX = (uint)outerSize;
        IntPtr aPtr = A.Handle;
        IntPtr valPtr = values.Handle;
        IntPtr idxPtr = indices.Handle;

        if (k <= 8)
        {
            // topk_small kernel: input, values, indices, outerSize, reduceSize, k
            int paramK = k;
            void** args = stackalloc void*[6];
            args[0] = &aPtr;
            args[1] = &valPtr;
            args[2] = &idxPtr;
            args[3] = &outerSize;
            args[4] = &reduceSize;
            args[5] = &paramK;
            LaunchKernel(kernel, gridX, (uint)Math.Min(256, reduceSize), args);
        }
        else
        {
            // topk kernel needs shared memory for top-k values and indices
            int sortedInt = sorted ? 1 : 0;
            void** args = stackalloc void*[7];
            args[0] = &aPtr;
            args[1] = &valPtr;
            args[2] = &idxPtr;
            args[3] = &outerSize;
            args[4] = &reduceSize;
            args[5] = &k;
            args[6] = &sortedInt;
            uint sharedMem = (uint)(k * (sizeof(float) + sizeof(int)));
            LaunchKernelWithSharedMem(kernel, gridX, (uint)Math.Min(256, reduceSize), sharedMem, args);
        }
    }

    public unsafe void BroadcastMultiplyLastAxis(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int outerSize, int innerSize)
    {
        if (!_kernelCache.TryGetValue("broadcast_multiply_last_axis", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: broadcast_multiply_last_axis");

        using var _ = PushContext();
        int totalSize = outerSize * innerSize;
        uint grid = (uint)((totalSize + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr aPtr = A.Handle;
        IntPtr bPtr = B.Handle;
        IntPtr cPtr = C.Handle;
        void** args = stackalloc void*[5];
        args[0] = &aPtr;
        args[1] = &bPtr;
        args[2] = &cPtr;
        args[3] = &outerSize;
        args[4] = &innerSize;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    public unsafe void BroadcastMultiplyFirstAxis(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int outerSize, int innerSize)
    {
        if (!_kernelCache.TryGetValue("broadcast_multiply_first_axis", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: broadcast_multiply_first_axis");

        using var _ = PushContext();
        int totalSize = outerSize * innerSize;
        uint grid = (uint)((totalSize + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr aPtr = A.Handle;
        IntPtr bPtr = B.Handle;
        IntPtr cPtr = C.Handle;
        void** args = stackalloc void*[5];
        args[0] = &aPtr;
        args[1] = &bPtr;
        args[2] = &cPtr;
        args[3] = &outerSize;
        args[4] = &innerSize;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    #endregion


    #region Optimizer Operations

    public unsafe void SgdMomentumUpdate(IGpuBuffer param, IGpuBuffer gradient, IGpuBuffer velocity,
        float learningRate, float momentum, float weightDecay, int size)
    {
        // Validate buffer parameters
        if (param is null) throw new ArgumentNullException(nameof(param));
        if (gradient is null) throw new ArgumentNullException(nameof(gradient));
        if (velocity is null) throw new ArgumentNullException(nameof(velocity));
        if (size <= 0)
            throw new ArgumentOutOfRangeException(nameof(size), "Size must be positive.");

        if (!_kernelCache.TryGetValue("sgd_momentum_update", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: sgd_momentum_update");

        using var _ = PushContext();
        uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr paramPtr = param.Handle;
        IntPtr gradPtr = gradient.Handle;
        IntPtr velPtr = velocity.Handle;
        void** args = stackalloc void*[7];
        args[0] = &paramPtr;
        args[1] = &gradPtr;
        args[2] = &velPtr;
        args[3] = &learningRate;
        args[4] = &momentum;
        args[5] = &weightDecay;
        args[6] = &size;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    public unsafe void AdamUpdate(IGpuBuffer param, IGpuBuffer gradient, IGpuBuffer m, IGpuBuffer v,
        float learningRate, float beta1, float beta2, float epsilon, float weightDecay, int step, int size)
    {
        // Validate buffer parameters
        if (param is null) throw new ArgumentNullException(nameof(param));
        if (gradient is null) throw new ArgumentNullException(nameof(gradient));
        if (m is null) throw new ArgumentNullException(nameof(m));
        if (v is null) throw new ArgumentNullException(nameof(v));
        if (size <= 0)
            throw new ArgumentOutOfRangeException(nameof(size), "Size must be positive.");
        if (step < 1)
            throw new ArgumentOutOfRangeException(nameof(step), "Step must be at least 1.");
        if (epsilon <= 0)
            throw new ArgumentOutOfRangeException(nameof(epsilon), "Epsilon must be positive.");

        if (!_kernelCache.TryGetValue("adam_update", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: adam_update");

        using var _ = PushContext();
        uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr paramPtr = param.Handle;
        IntPtr gradPtr = gradient.Handle;
        IntPtr mPtr = m.Handle;
        IntPtr vPtr = v.Handle;
        void** args = stackalloc void*[11];
        args[0] = &paramPtr;
        args[1] = &gradPtr;
        args[2] = &mPtr;
        args[3] = &vPtr;
        args[4] = &learningRate;
        args[5] = &beta1;
        args[6] = &beta2;
        args[7] = &epsilon;
        args[8] = &weightDecay;
        args[9] = &step;
        args[10] = &size;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    public unsafe void AdamWUpdate(IGpuBuffer param, IGpuBuffer gradient, IGpuBuffer m, IGpuBuffer v,
        float learningRate, float beta1, float beta2, float epsilon, float weightDecay, int step, int size)
    {
        // Validate buffer parameters
        if (param is null) throw new ArgumentNullException(nameof(param));
        if (gradient is null) throw new ArgumentNullException(nameof(gradient));
        if (m is null) throw new ArgumentNullException(nameof(m));
        if (v is null) throw new ArgumentNullException(nameof(v));
        if (size <= 0)
            throw new ArgumentOutOfRangeException(nameof(size), "Size must be positive.");
        if (step < 1)
            throw new ArgumentOutOfRangeException(nameof(step), "Step must be at least 1.");
        if (epsilon <= 0)
            throw new ArgumentOutOfRangeException(nameof(epsilon), "Epsilon must be positive.");

        if (!_kernelCache.TryGetValue("adamw_update", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: adamw_update");

        using var _ = PushContext();
        uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr paramPtr = param.Handle;
        IntPtr gradPtr = gradient.Handle;
        IntPtr mPtr = m.Handle;
        IntPtr vPtr = v.Handle;
        void** args = stackalloc void*[11];
        args[0] = &paramPtr;
        args[1] = &gradPtr;
        args[2] = &mPtr;
        args[3] = &vPtr;
        args[4] = &learningRate;
        args[5] = &beta1;
        args[6] = &beta2;
        args[7] = &epsilon;
        args[8] = &weightDecay;
        args[9] = &step;
        args[10] = &size;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    /// <inheritdoc/>
    public unsafe void RmspropUpdate(IGpuBuffer param, IGpuBuffer gradient, IGpuBuffer squaredAvg,
        float learningRate, float rho, float epsilon, float weightDecay, int size)
    {
        // Validate buffer parameters
        if (param is null) throw new ArgumentNullException(nameof(param));
        if (gradient is null) throw new ArgumentNullException(nameof(gradient));
        if (squaredAvg is null) throw new ArgumentNullException(nameof(squaredAvg));
        if (size <= 0)
            throw new ArgumentOutOfRangeException(nameof(size), "Size must be positive.");
        if (epsilon <= 0)
            throw new ArgumentOutOfRangeException(nameof(epsilon), "Epsilon must be positive.");

        if (!_kernelCache.TryGetValue("rmsprop_update", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: rmsprop_update");

        using var _ = PushContext();
        uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr paramPtr = param.Handle;
        IntPtr gradPtr = gradient.Handle;
        IntPtr sqAvgPtr = squaredAvg.Handle;
        void** args = stackalloc void*[8];
        args[0] = &paramPtr;
        args[1] = &gradPtr;
        args[2] = &sqAvgPtr;
        args[3] = &learningRate;
        args[4] = &rho;
        args[5] = &epsilon;
        args[6] = &weightDecay;
        args[7] = &size;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    /// <inheritdoc/>
    public unsafe void AdagradUpdate(IGpuBuffer param, IGpuBuffer gradient, IGpuBuffer accumulatedGrad,
        float learningRate, float epsilon, float weightDecay, int size)
    {
        // Validate buffer parameters
        if (param is null) throw new ArgumentNullException(nameof(param));
        if (gradient is null) throw new ArgumentNullException(nameof(gradient));
        if (accumulatedGrad is null) throw new ArgumentNullException(nameof(accumulatedGrad));
        if (size <= 0)
            throw new ArgumentOutOfRangeException(nameof(size), "Size must be positive.");
        if (epsilon <= 0)
            throw new ArgumentOutOfRangeException(nameof(epsilon), "Epsilon must be positive.");

        if (!_kernelCache.TryGetValue("adagrad_update", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: adagrad_update");

        using var _ = PushContext();
        uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr paramPtr = param.Handle;
        IntPtr gradPtr = gradient.Handle;
        IntPtr accumPtr = accumulatedGrad.Handle;
        void** args = stackalloc void*[7];
        args[0] = &paramPtr;
        args[1] = &gradPtr;
        args[2] = &accumPtr;
        args[3] = &learningRate;
        args[4] = &epsilon;
        args[5] = &weightDecay;
        args[6] = &size;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    /// <inheritdoc/>
    public unsafe void NagUpdate(IGpuBuffer param, IGpuBuffer gradient, IGpuBuffer velocity,
        float learningRate, float momentum, float weightDecay, int size)
    {
        // Validate buffer parameters
        if (param is null) throw new ArgumentNullException(nameof(param));
        if (gradient is null) throw new ArgumentNullException(nameof(gradient));
        if (velocity is null) throw new ArgumentNullException(nameof(velocity));
        if (size <= 0)
            throw new ArgumentOutOfRangeException(nameof(size), "Size must be positive.");

        if (!_kernelCache.TryGetValue("nag_update", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: nag_update");

        using var _ = PushContext();
        uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr paramPtr = param.Handle;
        IntPtr gradPtr = gradient.Handle;
        IntPtr velPtr = velocity.Handle;
        void** args = stackalloc void*[7];
        args[0] = &paramPtr;
        args[1] = &gradPtr;
        args[2] = &velPtr;
        args[3] = &learningRate;
        args[4] = &momentum;
        args[5] = &weightDecay;
        args[6] = &size;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    /// <inheritdoc/>
    public unsafe void LarsUpdate(IGpuBuffer param, IGpuBuffer gradient, IGpuBuffer velocity,
        float learningRate, float momentum, float weightDecay, float trustCoeff, int size)
    {
        // Validate buffer parameters
        if (param is null) throw new ArgumentNullException(nameof(param));
        if (gradient is null) throw new ArgumentNullException(nameof(gradient));
        if (velocity is null) throw new ArgumentNullException(nameof(velocity));
        if (size <= 0)
            throw new ArgumentOutOfRangeException(nameof(size), "Size must be positive.");
        if (trustCoeff <= 0)
            throw new ArgumentOutOfRangeException(nameof(trustCoeff), "Trust coefficient must be positive.");

        if (!_kernelCache.TryGetValue("lars_update", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: lars_update");

        using var _ = PushContext();
        uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr paramPtr = param.Handle;
        IntPtr gradPtr = gradient.Handle;
        IntPtr velPtr = velocity.Handle;
        void** args = stackalloc void*[8];
        args[0] = &paramPtr;
        args[1] = &gradPtr;
        args[2] = &velPtr;
        args[3] = &learningRate;
        args[4] = &momentum;
        args[5] = &weightDecay;
        args[6] = &trustCoeff;
        args[7] = &size;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    /// <inheritdoc/>
    public unsafe void LambUpdate(IGpuBuffer param, IGpuBuffer gradient, IGpuBuffer m, IGpuBuffer v,
        float learningRate, float beta1, float beta2, float epsilon, float weightDecay, int step, int size)
    {
        // Validate buffer parameters
        if (param is null) throw new ArgumentNullException(nameof(param));
        if (gradient is null) throw new ArgumentNullException(nameof(gradient));
        if (m is null) throw new ArgumentNullException(nameof(m));
        if (v is null) throw new ArgumentNullException(nameof(v));
        if (size <= 0)
            throw new ArgumentOutOfRangeException(nameof(size), "Size must be positive.");
        if (step < 1)
            throw new ArgumentOutOfRangeException(nameof(step), "Step must be at least 1.");
        if (epsilon <= 0)
            throw new ArgumentOutOfRangeException(nameof(epsilon), "Epsilon must be positive.");

        if (!_kernelCache.TryGetValue("lamb_update", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: lamb_update");

        using var _ = PushContext();
        uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr paramPtr = param.Handle;
        IntPtr gradPtr = gradient.Handle;
        IntPtr mPtr = m.Handle;
        IntPtr vPtr = v.Handle;
        float trustRatio = 1.0f; // Default: no layer-wise scaling (degenerates to AdamW)
        void** args = stackalloc void*[12];
        args[0] = &paramPtr;
        args[1] = &gradPtr;
        args[2] = &mPtr;
        args[3] = &vPtr;
        args[4] = &learningRate;
        args[5] = &beta1;
        args[6] = &beta2;
        args[7] = &epsilon;
        args[8] = &weightDecay;
        args[9] = &trustRatio;
        args[10] = &step;
        args[11] = &size;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    /// <inheritdoc/>
    public unsafe void SgdUpdate(IGpuBuffer param, IGpuBuffer gradient,
        float learningRate, float weightDecay, int size)
    {
        // Validate buffer parameters
        if (param is null) throw new ArgumentNullException(nameof(param));
        if (gradient is null) throw new ArgumentNullException(nameof(gradient));
        if (size <= 0)
            throw new ArgumentOutOfRangeException(nameof(size), "Size must be positive.");

        if (!_kernelCache.TryGetValue("sgd_update", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: sgd_update");

        using var _ = PushContext();
        uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr paramPtr = param.Handle;
        IntPtr gradPtr = gradient.Handle;
        void** args = stackalloc void*[5];
        args[0] = &paramPtr;
        args[1] = &gradPtr;
        args[2] = &learningRate;
        args[3] = &weightDecay;
        args[4] = &size;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    /// <summary>
    /// Proximal gradient (ISTA) step with L1 soft-threshold prox, in place on the GPU:
    /// <c>tmp = param - lr*grad; param = sign(tmp)*max(|tmp| - l1Strength, 0)</c>.
    /// </summary>
    public unsafe void ProximalL1Update(IGpuBuffer param, IGpuBuffer gradient,
        float learningRate, float l1Strength, int size)
    {
        if (param is null) throw new ArgumentNullException(nameof(param));
        if (gradient is null) throw new ArgumentNullException(nameof(gradient));
        if (size <= 0)
            throw new ArgumentOutOfRangeException(nameof(size), "Size must be positive.");
        // Validate device-buffer capacity before launch: GpuOptimizer passes p.Length
        // for `size` without cross-checking the gradient buffer, so a shape mismatch
        // (gradient.Size < size) would otherwise read past gradient.Handle — an illegal
        // device access / silent corruption instead of a clear managed exception.
        if (param.Size < size)
            throw new ArgumentException($"param buffer too small: capacity={param.Size}, required={size}.", nameof(param));
        if (gradient.Size < size)
            throw new ArgumentException($"gradient buffer too small: capacity={gradient.Size}, required={size}.", nameof(gradient));

        if (!_kernelCache.TryGetValue("proximal_l1_update", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: proximal_l1_update");

        using var _ = PushContext();
        uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr paramPtr = param.Handle;
        IntPtr gradPtr = gradient.Handle;
        void** args = stackalloc void*[5];
        args[0] = &paramPtr;
        args[1] = &gradPtr;
        args[2] = &learningRate;
        args[3] = &l1Strength;
        args[4] = &size;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    /// <summary>
    /// 8-bit Adam blockwise update in place on the GPU. mQ/vQ are int8 quantized
    /// moments (one byte per element); mScales/vScales are one double per block.
    /// One CUDA block per quant block, 256 threads, shared-memory maxAbs reduction.
    /// Matches the CPU Adam8BitOptimizer's CompressBothMoments / percentile>=100 /
    /// deterministic-rounding path.
    /// </summary>
    public unsafe void Adam8BitUpdate(IGpuBuffer param, IGpuBuffer gradient,
        IGpuBuffer mQuant, IGpuBuffer vQuant, IGpuBuffer mScales, IGpuBuffer vScales,
        float learningRate, float beta1, float beta2, float epsilon,
        float oneMinusBeta1, float oneMinusBeta2, float biasCorrection1, float biasCorrection2,
        int blockSize, int paramLength, int numBlocks)
    {
        if (param is null) throw new ArgumentNullException(nameof(param));
        if (gradient is null) throw new ArgumentNullException(nameof(gradient));
        if (mQuant is null || vQuant is null || mScales is null || vScales is null)
            throw new ArgumentNullException(nameof(mQuant));
        if (paramLength <= 0 || blockSize <= 0 || numBlocks <= 0)
            throw new ArgumentOutOfRangeException(nameof(paramLength));

        if (!_kernelCache.TryGetValue("adam8bit_update", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: adam8bit_update");

        using var _ = PushContext();
        IntPtr pPtr = param.Handle, gPtr = gradient.Handle;
        IntPtr mqPtr = mQuant.Handle, vqPtr = vQuant.Handle, msPtr = mScales.Handle, vsPtr = vScales.Handle;
        void** args = stackalloc void*[17];
        args[0] = &pPtr; args[1] = &gPtr; args[2] = &mqPtr; args[3] = &vqPtr; args[4] = &msPtr; args[5] = &vsPtr;
        args[6] = &learningRate; args[7] = &beta1; args[8] = &beta2; args[9] = &epsilon;
        args[10] = &oneMinusBeta1; args[11] = &oneMinusBeta2; args[12] = &biasCorrection1; args[13] = &biasCorrection2;
        args[14] = &blockSize; args[15] = &paramLength; args[16] = &numBlocks;
        // One CUDA block per quant block (256 threads); the kernel grid-strides within each.
        LaunchKernel(kernel, (uint)numBlocks, 256u, args);
    }

    /// <inheritdoc/>
    public unsafe void AdadeltaUpdate(IGpuBuffer param, IGpuBuffer gradient, IGpuBuffer accumGrad, IGpuBuffer accumUpdate,
        float rho, float epsilon, float weightDecay, int size)
    {
        // Validate buffer parameters
        if (param is null) throw new ArgumentNullException(nameof(param));
        if (gradient is null) throw new ArgumentNullException(nameof(gradient));
        if (accumGrad is null) throw new ArgumentNullException(nameof(accumGrad));
        if (accumUpdate is null) throw new ArgumentNullException(nameof(accumUpdate));
        if (size <= 0)
            throw new ArgumentOutOfRangeException(nameof(size), "Size must be positive.");
        if (epsilon <= 0)
            throw new ArgumentOutOfRangeException(nameof(epsilon), "Epsilon must be positive.");

        if (!_kernelCache.TryGetValue("adadelta_update", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: adadelta_update");

        using var _ = PushContext();
        uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr paramPtr = param.Handle;
        IntPtr gradPtr = gradient.Handle;
        IntPtr accumGradPtr = accumGrad.Handle;
        IntPtr accumUpdatePtr = accumUpdate.Handle;
        void** args = stackalloc void*[8];
        args[0] = &paramPtr;
        args[1] = &gradPtr;
        args[2] = &accumGradPtr;
        args[3] = &accumUpdatePtr;
        args[4] = &rho;
        args[5] = &epsilon;
        args[6] = &weightDecay;
        args[7] = &size;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    /// <inheritdoc/>
    public unsafe void AmsgradUpdate(IGpuBuffer param, IGpuBuffer gradient, IGpuBuffer m, IGpuBuffer v, IGpuBuffer vMax,
        float learningRate, float beta1, float beta2, float epsilon, float weightDecay, int step, int size)
    {
        // Validate buffer parameters
        if (param is null) throw new ArgumentNullException(nameof(param));
        if (gradient is null) throw new ArgumentNullException(nameof(gradient));
        if (m is null) throw new ArgumentNullException(nameof(m));
        if (v is null) throw new ArgumentNullException(nameof(v));
        if (vMax is null) throw new ArgumentNullException(nameof(vMax));
        if (size <= 0)
            throw new ArgumentOutOfRangeException(nameof(size), "Size must be positive.");
        if (step < 1)
            throw new ArgumentOutOfRangeException(nameof(step), "Step must be at least 1.");
        if (epsilon <= 0)
            throw new ArgumentOutOfRangeException(nameof(epsilon), "Epsilon must be positive.");

        if (!_kernelCache.TryGetValue("amsgrad_update", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: amsgrad_update");

        using var _ = PushContext();
        uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr paramPtr = param.Handle;
        IntPtr gradPtr = gradient.Handle;
        IntPtr mPtr = m.Handle;
        IntPtr vPtr = v.Handle;
        IntPtr vMaxPtr = vMax.Handle;
        void** args = stackalloc void*[12];
        args[0] = &paramPtr;
        args[1] = &gradPtr;
        args[2] = &mPtr;
        args[3] = &vPtr;
        args[4] = &vMaxPtr;
        args[5] = &learningRate;
        args[6] = &beta1;
        args[7] = &beta2;
        args[8] = &epsilon;
        args[9] = &weightDecay;
        args[10] = &step;
        args[11] = &size;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    /// <inheritdoc/>
    public unsafe void AdamaxUpdate(IGpuBuffer param, IGpuBuffer gradient, IGpuBuffer m, IGpuBuffer u,
        float learningRate, float beta1, float beta2, float epsilon, float weightDecay, int step, int size)
    {
        // Validate buffer parameters
        if (param is null) throw new ArgumentNullException(nameof(param));
        if (gradient is null) throw new ArgumentNullException(nameof(gradient));
        if (m is null) throw new ArgumentNullException(nameof(m));
        if (u is null) throw new ArgumentNullException(nameof(u));
        if (size <= 0)
            throw new ArgumentOutOfRangeException(nameof(size), "Size must be positive.");
        if (step < 1)
            throw new ArgumentOutOfRangeException(nameof(step), "Step must be at least 1.");
        if (epsilon <= 0)
            throw new ArgumentOutOfRangeException(nameof(epsilon), "Epsilon must be positive.");

        if (!_kernelCache.TryGetValue("adamax_update", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: adamax_update");

        using var _ = PushContext();
        uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr paramPtr = param.Handle;
        IntPtr gradPtr = gradient.Handle;
        IntPtr mPtr = m.Handle;
        IntPtr uPtr = u.Handle;
        void** args = stackalloc void*[11];
        args[0] = &paramPtr;
        args[1] = &gradPtr;
        args[2] = &mPtr;
        args[3] = &uPtr;
        args[4] = &learningRate;
        args[5] = &beta1;
        args[6] = &beta2;
        args[7] = &epsilon;
        args[8] = &weightDecay;
        args[9] = &step;
        args[10] = &size;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    /// <inheritdoc/>
    public unsafe void LionUpdate(IGpuBuffer param, IGpuBuffer gradient, IGpuBuffer m,
        float learningRate, float beta1, float beta2, float weightDecay, int size)
    {
        // Validate buffer parameters
        if (param is null) throw new ArgumentNullException(nameof(param));
        if (gradient is null) throw new ArgumentNullException(nameof(gradient));
        if (m is null) throw new ArgumentNullException(nameof(m));
        if (size <= 0)
            throw new ArgumentOutOfRangeException(nameof(size), "Size must be positive.");

        if (!_kernelCache.TryGetValue("lion_update", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: lion_update");

        using var _ = PushContext();
        uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr paramPtr = param.Handle;
        IntPtr gradPtr = gradient.Handle;
        IntPtr mPtr = m.Handle;
        void** args = stackalloc void*[8];
        args[0] = &paramPtr;
        args[1] = &gradPtr;
        args[2] = &mPtr;
        args[3] = &learningRate;
        args[4] = &beta1;
        args[5] = &beta2;
        args[6] = &weightDecay;
        args[7] = &size;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    /// <inheritdoc/>
    public unsafe void NadamUpdate(IGpuBuffer param, IGpuBuffer gradient, IGpuBuffer m, IGpuBuffer v,
        float learningRate, float beta1, float beta2, float epsilon, float weightDecay, int step, int size)
    {
        // Validate buffer parameters
        if (param is null) throw new ArgumentNullException(nameof(param));
        if (gradient is null) throw new ArgumentNullException(nameof(gradient));
        if (m is null) throw new ArgumentNullException(nameof(m));
        if (v is null) throw new ArgumentNullException(nameof(v));
        if (size <= 0)
            throw new ArgumentOutOfRangeException(nameof(size), "Size must be positive.");
        if (step < 1)
            throw new ArgumentOutOfRangeException(nameof(step), "Step must be at least 1.");
        if (epsilon <= 0)
            throw new ArgumentOutOfRangeException(nameof(epsilon), "Epsilon must be positive.");

        if (!_kernelCache.TryGetValue("nadam_update", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: nadam_update");

        using var _ = PushContext();
        uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr paramPtr = param.Handle;
        IntPtr gradPtr = gradient.Handle;
        IntPtr mPtr = m.Handle;
        IntPtr vPtr = v.Handle;
        void** args = stackalloc void*[11];
        args[0] = &paramPtr;
        args[1] = &gradPtr;
        args[2] = &mPtr;
        args[3] = &vPtr;
        args[4] = &learningRate;
        args[5] = &beta1;
        args[6] = &beta2;
        args[7] = &epsilon;
        args[8] = &weightDecay;
        args[9] = &step;
        args[10] = &size;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    /// <inheritdoc/>
    public unsafe void FtrlUpdate(IGpuBuffer param, IGpuBuffer gradient, IGpuBuffer z, IGpuBuffer n,
        float learningRate, float l1Reg, float l2Reg, float beta, int size)
    {
        // Validate buffer parameters
        if (param is null) throw new ArgumentNullException(nameof(param));
        if (gradient is null) throw new ArgumentNullException(nameof(gradient));
        if (z is null) throw new ArgumentNullException(nameof(z));
        if (n is null) throw new ArgumentNullException(nameof(n));
        if (size <= 0)
            throw new ArgumentOutOfRangeException(nameof(size), "Size must be positive.");

        if (!_kernelCache.TryGetValue("ftrl_update", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: ftrl_update");

        using var _ = PushContext();
        uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr paramPtr = param.Handle;
        IntPtr gradPtr = gradient.Handle;
        IntPtr zPtr = z.Handle;
        IntPtr nPtr = n.Handle;
        void** args = stackalloc void*[9];
        args[0] = &paramPtr;
        args[1] = &gradPtr;
        args[2] = &zPtr;
        args[3] = &nPtr;
        args[4] = &learningRate;
        args[5] = &l1Reg;
        args[6] = &l2Reg;
        args[7] = &beta;
        args[8] = &size;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    // ==================================================================
    // PR #567 — Native sparse scatter-update kernels
    // ==================================================================
    // Each method below launches the matching sparse_*_update CUDA kernel
    // (defined in CudaOptimizerKernels.cs) with one thread per non-zero
    // gradient entry. Reads / writes only (param[idx], state[idx]) for the
    // nnz indices the autodiff producer published; the other (size - nnz)
    // entries are never touched. Memory traffic is O(nnz) vs. the dense
    // path's O(size).
    // ==================================================================

    private static void EnsureSparseArgs(IGpuBuffer? param, IGpuBuffer? sparseIndices, IGpuBuffer? sparseValues, int nnz)
    {
        if (param is null) throw new ArgumentNullException(nameof(param));
        if (sparseIndices is null) throw new ArgumentNullException(nameof(sparseIndices));
        if (sparseValues is null) throw new ArgumentNullException(nameof(sparseValues));
        if (nnz < 0) throw new ArgumentOutOfRangeException(nameof(nnz));
    }

    /// <inheritdoc/>
    public unsafe void SparseSgdUpdate(IGpuBuffer param,
        IGpuBuffer sparseIndices, IGpuBuffer sparseValues, int nnz,
        float learningRate, float weightDecay)
    {
        EnsureSparseArgs(param, sparseIndices, sparseValues, nnz);
        if (nnz == 0) return;
        if (!_kernelCache.TryGetValue("sparse_sgd_update", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: sparse_sgd_update");
        using var _ = PushContext();
        uint grid = (uint)((nnz + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr pP = param.Handle, pI = sparseIndices.Handle, pV = sparseValues.Handle;
        void** args = stackalloc void*[6];
        args[0] = &pP; args[1] = &pI; args[2] = &pV;
        args[3] = &learningRate; args[4] = &weightDecay; args[5] = &nnz;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    /// <inheritdoc/>
    public unsafe void SparseSgdMomentumUpdate(IGpuBuffer param, IGpuBuffer velocity,
        IGpuBuffer sparseIndices, IGpuBuffer sparseValues, int nnz,
        float learningRate, float momentum, float weightDecay)
    {
        EnsureSparseArgs(param, sparseIndices, sparseValues, nnz);
        if (velocity is null) throw new ArgumentNullException(nameof(velocity));
        if (nnz == 0) return;
        if (!_kernelCache.TryGetValue("sparse_sgd_momentum_update", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: sparse_sgd_momentum_update");
        using var _ = PushContext();
        uint grid = (uint)((nnz + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr pP = param.Handle, pI = sparseIndices.Handle, pV = sparseValues.Handle, pVel = velocity.Handle;
        void** args = stackalloc void*[8];
        args[0] = &pP; args[1] = &pI; args[2] = &pV; args[3] = &pVel;
        args[4] = &learningRate; args[5] = &momentum; args[6] = &weightDecay; args[7] = &nnz;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    /// <inheritdoc/>
    public unsafe void SparseAdamUpdate(IGpuBuffer param, IGpuBuffer m, IGpuBuffer v,
        IGpuBuffer sparseIndices, IGpuBuffer sparseValues, int nnz,
        float learningRate, float beta1, float beta2, float epsilon, float weightDecay, int step)
    {
        EnsureSparseArgs(param, sparseIndices, sparseValues, nnz);
        if (m is null) throw new ArgumentNullException(nameof(m));
        if (v is null) throw new ArgumentNullException(nameof(v));
        if (step < 1) throw new ArgumentOutOfRangeException(nameof(step));
        if (epsilon <= 0) throw new ArgumentOutOfRangeException(nameof(epsilon));
        if (nnz == 0) return;
        if (!_kernelCache.TryGetValue("sparse_adam_update", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: sparse_adam_update");
        using var _ = PushContext();
        uint grid = (uint)((nnz + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr pP = param.Handle, pI = sparseIndices.Handle, pV = sparseValues.Handle;
        IntPtr pM = m.Handle, pVv = v.Handle;
        void** args = stackalloc void*[12];
        args[0] = &pP; args[1] = &pI; args[2] = &pV;
        args[3] = &pM; args[4] = &pVv;
        args[5] = &learningRate; args[6] = &beta1; args[7] = &beta2;
        args[8] = &epsilon; args[9] = &weightDecay; args[10] = &step; args[11] = &nnz;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    /// <inheritdoc/>
    public unsafe void SparseAdamWUpdate(IGpuBuffer param, IGpuBuffer m, IGpuBuffer v,
        IGpuBuffer sparseIndices, IGpuBuffer sparseValues, int nnz,
        float learningRate, float beta1, float beta2, float epsilon, float weightDecay, int step)
    {
        EnsureSparseArgs(param, sparseIndices, sparseValues, nnz);
        if (m is null) throw new ArgumentNullException(nameof(m));
        if (v is null) throw new ArgumentNullException(nameof(v));
        if (step < 1) throw new ArgumentOutOfRangeException(nameof(step));
        if (nnz == 0) return;
        if (!_kernelCache.TryGetValue("sparse_adamw_update", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: sparse_adamw_update");
        using var _ = PushContext();
        uint grid = (uint)((nnz + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr pP = param.Handle, pI = sparseIndices.Handle, pV = sparseValues.Handle;
        IntPtr pM = m.Handle, pVv = v.Handle;
        void** args = stackalloc void*[12];
        args[0] = &pP; args[1] = &pI; args[2] = &pV;
        args[3] = &pM; args[4] = &pVv;
        args[5] = &learningRate; args[6] = &beta1; args[7] = &beta2;
        args[8] = &epsilon; args[9] = &weightDecay; args[10] = &step; args[11] = &nnz;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    /// <inheritdoc/>
    public unsafe void SparseRmspropUpdate(IGpuBuffer param, IGpuBuffer squaredAvg,
        IGpuBuffer sparseIndices, IGpuBuffer sparseValues, int nnz,
        float learningRate, float rho, float epsilon, float weightDecay)
    {
        EnsureSparseArgs(param, sparseIndices, sparseValues, nnz);
        if (squaredAvg is null) throw new ArgumentNullException(nameof(squaredAvg));
        if (nnz == 0) return;
        if (!_kernelCache.TryGetValue("sparse_rmsprop_update", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: sparse_rmsprop_update");
        using var _ = PushContext();
        uint grid = (uint)((nnz + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr pP = param.Handle, pI = sparseIndices.Handle, pV = sparseValues.Handle, pSq = squaredAvg.Handle;
        void** args = stackalloc void*[9];
        args[0] = &pP; args[1] = &pI; args[2] = &pV; args[3] = &pSq;
        args[4] = &learningRate; args[5] = &rho; args[6] = &epsilon; args[7] = &weightDecay; args[8] = &nnz;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    /// <inheritdoc/>
    public unsafe void SparseAdagradUpdate(IGpuBuffer param, IGpuBuffer accumulatedGrad,
        IGpuBuffer sparseIndices, IGpuBuffer sparseValues, int nnz,
        float learningRate, float epsilon, float weightDecay)
    {
        EnsureSparseArgs(param, sparseIndices, sparseValues, nnz);
        if (accumulatedGrad is null) throw new ArgumentNullException(nameof(accumulatedGrad));
        if (nnz == 0) return;
        if (!_kernelCache.TryGetValue("sparse_adagrad_update", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: sparse_adagrad_update");
        using var _ = PushContext();
        uint grid = (uint)((nnz + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr pP = param.Handle, pI = sparseIndices.Handle, pV = sparseValues.Handle, pA = accumulatedGrad.Handle;
        void** args = stackalloc void*[8];
        args[0] = &pP; args[1] = &pI; args[2] = &pV; args[3] = &pA;
        args[4] = &learningRate; args[5] = &epsilon; args[6] = &weightDecay; args[7] = &nnz;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    /// <inheritdoc/>
    public unsafe void SparseNagUpdate(IGpuBuffer param, IGpuBuffer velocity,
        IGpuBuffer sparseIndices, IGpuBuffer sparseValues, int nnz,
        float learningRate, float momentum, float weightDecay)
    {
        EnsureSparseArgs(param, sparseIndices, sparseValues, nnz);
        if (velocity is null) throw new ArgumentNullException(nameof(velocity));
        if (nnz == 0) return;
        if (!_kernelCache.TryGetValue("sparse_nag_update", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: sparse_nag_update");
        using var _ = PushContext();
        uint grid = (uint)((nnz + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr pP = param.Handle, pI = sparseIndices.Handle, pV = sparseValues.Handle, pVel = velocity.Handle;
        void** args = stackalloc void*[8];
        args[0] = &pP; args[1] = &pI; args[2] = &pV; args[3] = &pVel;
        args[4] = &learningRate; args[5] = &momentum; args[6] = &weightDecay; args[7] = &nnz;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    /// <inheritdoc/>
    public unsafe void SparseAdadeltaUpdate(IGpuBuffer param, IGpuBuffer accumGrad, IGpuBuffer accumUpdate,
        IGpuBuffer sparseIndices, IGpuBuffer sparseValues, int nnz,
        float rho, float epsilon, float weightDecay)
    {
        EnsureSparseArgs(param, sparseIndices, sparseValues, nnz);
        if (accumGrad is null) throw new ArgumentNullException(nameof(accumGrad));
        if (accumUpdate is null) throw new ArgumentNullException(nameof(accumUpdate));
        if (nnz == 0) return;
        if (!_kernelCache.TryGetValue("sparse_adadelta_update", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: sparse_adadelta_update");
        using var _ = PushContext();
        uint grid = (uint)((nnz + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr pP = param.Handle, pI = sparseIndices.Handle, pV = sparseValues.Handle;
        IntPtr pAg = accumGrad.Handle, pAu = accumUpdate.Handle;
        void** args = stackalloc void*[9];
        args[0] = &pP; args[1] = &pI; args[2] = &pV; args[3] = &pAg; args[4] = &pAu;
        args[5] = &rho; args[6] = &epsilon; args[7] = &weightDecay; args[8] = &nnz;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    /// <inheritdoc/>
    public unsafe void SparseAmsgradUpdate(IGpuBuffer param, IGpuBuffer m, IGpuBuffer v, IGpuBuffer vMax,
        IGpuBuffer sparseIndices, IGpuBuffer sparseValues, int nnz,
        float learningRate, float beta1, float beta2, float epsilon, float weightDecay, int step)
    {
        EnsureSparseArgs(param, sparseIndices, sparseValues, nnz);
        if (m is null) throw new ArgumentNullException(nameof(m));
        if (v is null) throw new ArgumentNullException(nameof(v));
        if (vMax is null) throw new ArgumentNullException(nameof(vMax));
        if (step < 1) throw new ArgumentOutOfRangeException(nameof(step));
        if (nnz == 0) return;
        if (!_kernelCache.TryGetValue("sparse_amsgrad_update", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: sparse_amsgrad_update");
        using var _ = PushContext();
        uint grid = (uint)((nnz + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr pP = param.Handle, pI = sparseIndices.Handle, pV = sparseValues.Handle;
        IntPtr pM = m.Handle, pVv = v.Handle, pVm = vMax.Handle;
        void** args = stackalloc void*[13];
        args[0] = &pP; args[1] = &pI; args[2] = &pV;
        args[3] = &pM; args[4] = &pVv; args[5] = &pVm;
        args[6] = &learningRate; args[7] = &beta1; args[8] = &beta2;
        args[9] = &epsilon; args[10] = &weightDecay; args[11] = &step; args[12] = &nnz;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    /// <inheritdoc/>
    public unsafe void SparseAdamaxUpdate(IGpuBuffer param, IGpuBuffer m, IGpuBuffer u,
        IGpuBuffer sparseIndices, IGpuBuffer sparseValues, int nnz,
        float learningRate, float beta1, float beta2, float epsilon, float weightDecay, int step)
    {
        EnsureSparseArgs(param, sparseIndices, sparseValues, nnz);
        if (m is null) throw new ArgumentNullException(nameof(m));
        if (u is null) throw new ArgumentNullException(nameof(u));
        if (step < 1) throw new ArgumentOutOfRangeException(nameof(step));
        if (nnz == 0) return;
        if (!_kernelCache.TryGetValue("sparse_adamax_update", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: sparse_adamax_update");
        using var _ = PushContext();
        uint grid = (uint)((nnz + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr pP = param.Handle, pI = sparseIndices.Handle, pV = sparseValues.Handle;
        IntPtr pM = m.Handle, pU = u.Handle;
        void** args = stackalloc void*[12];
        args[0] = &pP; args[1] = &pI; args[2] = &pV;
        args[3] = &pM; args[4] = &pU;
        args[5] = &learningRate; args[6] = &beta1; args[7] = &beta2;
        args[8] = &epsilon; args[9] = &weightDecay; args[10] = &step; args[11] = &nnz;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    /// <inheritdoc/>
    public unsafe void SparseLionUpdate(IGpuBuffer param, IGpuBuffer m,
        IGpuBuffer sparseIndices, IGpuBuffer sparseValues, int nnz,
        float learningRate, float beta1, float beta2, float weightDecay)
    {
        EnsureSparseArgs(param, sparseIndices, sparseValues, nnz);
        if (m is null) throw new ArgumentNullException(nameof(m));
        if (nnz == 0) return;
        if (!_kernelCache.TryGetValue("sparse_lion_update", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: sparse_lion_update");
        using var _ = PushContext();
        uint grid = (uint)((nnz + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr pP = param.Handle, pI = sparseIndices.Handle, pV = sparseValues.Handle, pM = m.Handle;
        void** args = stackalloc void*[9];
        args[0] = &pP; args[1] = &pI; args[2] = &pV; args[3] = &pM;
        args[4] = &learningRate; args[5] = &beta1; args[6] = &beta2; args[7] = &weightDecay; args[8] = &nnz;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    /// <inheritdoc/>
    public unsafe void SparseNadamUpdate(IGpuBuffer param, IGpuBuffer m, IGpuBuffer v,
        IGpuBuffer sparseIndices, IGpuBuffer sparseValues, int nnz,
        float learningRate, float beta1, float beta2, float epsilon, float weightDecay, int step)
    {
        EnsureSparseArgs(param, sparseIndices, sparseValues, nnz);
        if (m is null) throw new ArgumentNullException(nameof(m));
        if (v is null) throw new ArgumentNullException(nameof(v));
        if (step < 1) throw new ArgumentOutOfRangeException(nameof(step));
        if (nnz == 0) return;
        if (!_kernelCache.TryGetValue("sparse_nadam_update", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: sparse_nadam_update");
        using var _ = PushContext();
        uint grid = (uint)((nnz + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr pP = param.Handle, pI = sparseIndices.Handle, pV = sparseValues.Handle;
        IntPtr pM = m.Handle, pVv = v.Handle;
        void** args = stackalloc void*[12];
        args[0] = &pP; args[1] = &pI; args[2] = &pV;
        args[3] = &pM; args[4] = &pVv;
        args[5] = &learningRate; args[6] = &beta1; args[7] = &beta2;
        args[8] = &epsilon; args[9] = &weightDecay; args[10] = &step; args[11] = &nnz;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    /// <inheritdoc/>
    public unsafe void SparseFtrlUpdate(IGpuBuffer param, IGpuBuffer z, IGpuBuffer n,
        IGpuBuffer sparseIndices, IGpuBuffer sparseValues, int nnz,
        float learningRate, float l1Reg, float l2Reg, float beta)
    {
        EnsureSparseArgs(param, sparseIndices, sparseValues, nnz);
        if (z is null) throw new ArgumentNullException(nameof(z));
        if (n is null) throw new ArgumentNullException(nameof(n));
        if (nnz == 0) return;
        if (!_kernelCache.TryGetValue("sparse_ftrl_update", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: sparse_ftrl_update");
        using var _ = PushContext();
        uint grid = (uint)((nnz + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr pP = param.Handle, pI = sparseIndices.Handle, pV = sparseValues.Handle;
        IntPtr pZ = z.Handle, pN = n.Handle;
        void** args = stackalloc void*[10];
        args[0] = &pP; args[1] = &pI; args[2] = &pV;
        args[3] = &pZ; args[4] = &pN;
        args[5] = &learningRate; args[6] = &l1Reg; args[7] = &l2Reg; args[8] = &beta; args[9] = &nnz;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    /// <inheritdoc/>
    public unsafe void SparseProximalL1Update(IGpuBuffer param,
        IGpuBuffer sparseIndices, IGpuBuffer sparseValues, int nnz,
        float learningRate, float l1Strength)
    {
        EnsureSparseArgs(param, sparseIndices, sparseValues, nnz);
        if (nnz == 0) return;
        if (!_kernelCache.TryGetValue("sparse_proximal_l1_update", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: sparse_proximal_l1_update");
        using var _ = PushContext();
        uint grid = (uint)((nnz + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr pP = param.Handle, pI = sparseIndices.Handle, pV = sparseValues.Handle;
        void** args = stackalloc void*[6];
        args[0] = &pP; args[1] = &pI; args[2] = &pV;
        args[3] = &learningRate; args[4] = &l1Strength; args[5] = &nnz;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    /// <inheritdoc/>
    public unsafe void ConvertToFp16(IGpuBuffer input, IGpuBuffer output, int size)
    {
        // The toolkit (cuda_fp16.h-based) kernel is the faster __half2 packed
        // path, but it only compiles on machines with the CUDA Toolkit
        // header. On driver-only hosts CompileKernelModule(... CudaFp16Kernels)
        // silently fails (see CudaBackend ctor) and the toolkit kernel never
        // registers — which used to make MatrixMultiply<Half> throw
        // "CUDA kernel not found: convert_fp32_to_fp16" and fall to the
        // 56-second CPU scalar path (#560). Fall through to the self-contained
        // native kernel here so the public API works on every CUDA system.
        if (_kernelCache.TryGetValue("convert_fp32_to_fp16", out var kernel))
        {
            using var _ = PushContext();
            uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
            IntPtr inPtr = input.Handle;
            IntPtr outPtr = output.Handle;
            void** args = stackalloc void*[3];
            args[0] = &inPtr;
            args[1] = &outPtr;
            args[2] = &size;
            LaunchKernel(kernel, grid, DefaultBlockSize, args);
            return;
        }

        // Driver-only fallback — exists by construction (compiled in the
        // ctor as part of fp16_native_kernels, not the optional fp16_kernels
        // module). Same FP32→FP16 semantics.
        ConvertToFp16Native(input, output, size);
    }

    /// <summary>
    /// FP16-NATIVE GELU (#558 layer 7 pilot): reads the FP16 activation buffer DIRECTLY, computes GELU in
    /// FP32 in-register (__half2 packed, 2 elems/thread), writes FP16 — no separate FP32 buffer is ever
    /// materialized, so there is no convert transient and the activation stays half-size end-to-end. This
    /// is the kernel shape the GPU memory win + Tensor-Core throughput require. <paramref name="size"/> is
    /// the element count; <paramref name="input"/>/<paramref name="output"/> are FP16 (byte) buffers.
    /// </summary>
    public unsafe void Fp16Gelu(IGpuBuffer input, IGpuBuffer output, int size)
        => LaunchUnaryFp16("fp16_gelu_native", input, output, size);

    /// <summary>
    /// True when the self-contained FP16-native op kernels (GELU/ReLU/add — no cuda_fp16.h) are compiled.
    /// They are built in the ctor as part of <c>fp16_native_kernels</c>, so this is available driver-only.
    /// </summary>
    public bool SupportsFp16NativeOps => IsAvailable && _kernelCache.ContainsKey("fp16_gelu_native");

    /// <summary>FP16-native ReLU: reads FP16 directly, max(x,0) in FP32, writes FP16 (no FP32 buffer).</summary>
    public unsafe void Fp16Relu(IGpuBuffer input, IGpuBuffer output, int size)
        => LaunchUnaryFp16("fp16_relu_native", input, output, size);

    /// <summary>FP16-native residual add: out = a + b; reads FP16 directly, FP32 accumulate, writes FP16.</summary>
    public unsafe void Fp16Add(IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, int size)
    {
        if (a is null) throw new ArgumentNullException(nameof(a));
        if (b is null) throw new ArgumentNullException(nameof(b));
        if (output is null) throw new ArgumentNullException(nameof(output));
        if (size <= 0) throw new ArgumentOutOfRangeException(nameof(size), "Element count must be positive.");
        if (!_kernelCache.TryGetValue("fp16_add_native", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: fp16_add_native");
        using var _ = PushContext();
        uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr aPtr = a.Handle, bPtr = b.Handle, outPtr = output.Handle;
        void** args = stackalloc void*[4];
        args[0] = &aPtr; args[1] = &bPtr; args[2] = &outPtr; args[3] = &size;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    /// <summary>FP32→FP16 via the self-contained native kernel (no cuda_fp16.h). Output is a half buffer.</summary>
    public unsafe void ConvertToFp16Native(IGpuBuffer input, IGpuBuffer output, int size)
        => LaunchConvertFp16("convert_fp32_to_fp16_native", input, output, size);

    /// <summary>FP16→FP32 via the self-contained native kernel (no cuda_fp16.h). Input is a half buffer.</summary>
    public unsafe void ConvertToFp32Native(IGpuBuffer input, IGpuBuffer output, int size)
        => LaunchConvertFp16("convert_fp16_to_fp32_native", input, output, size);

    /// <summary>
    /// FP16 Tensor-Core GEMM for the hot single-matmul path (#558 layer 7): row-major C[M,N] = A[M,K]·B[K,N]
    /// with FP16 inputs and an FP32 output. Multiplies in FP16 on the Tensor Cores, accumulates in FP32
    /// (<c>CUBLAS_COMPUTE_32F</c>) — full-rate Tensor-Core throughput (≈2× the TF32 <c>cublasSgemm</c> the
    /// default <see cref="Gemm"/> uses) at half the input bandwidth. This is the de-batched mirror of the
    /// proven <see cref="BatchedGemmExFanout"/> convention; <paramref name="Ahalf"/>/<paramref name="Bhalf"/>
    /// are FP16 (byte) buffers, <paramref name="C"/> is FP32. cuBLAS is column-major, so we compute
    /// Cᵀ = Bᵀ·Aᵀ by passing (N,M,K) with B then A — identical to the FP32 path. <c>cublasGemmEx</c> needs
    /// no <c>cuda_fp16.h</c> (the half pointers are runtime, not compile-time), so this works driver-only —
    /// pair with <see cref="ConvertToFp16Native"/> to feed it from FP32, or with FP16-resident activations
    /// to skip the convert entirely.
    /// </summary>
    public unsafe void GemmFp16(IGpuBuffer Ahalf, IGpuBuffer Bhalf, IGpuBuffer C, int M, int N, int K, float alpha = 1.0f, float beta = 0.0f)
    {
        if (!IsAvailable) throw new InvalidOperationException("CUDA backend is not available.");
        if (Ahalf is null) throw new ArgumentNullException(nameof(Ahalf));
        if (Bhalf is null) throw new ArgumentNullException(nameof(Bhalf));
        if (C is null) throw new ArgumentNullException(nameof(C));
        if (M <= 0 || N <= 0 || K <= 0) throw new ArgumentException($"GEMM dims must be positive (M={M}, N={N}, K={K}).");
        // A,B are 2-byte half; C is 4-byte float. Reject undersized buffers (byte-aware, since the
        // element-count view would misread half buffers as float-sized).
        if (Ahalf.SizeInBytes < (long)M * K * 2) throw new ArgumentException($"A half buffer too small: {Ahalf.SizeInBytes} < {(long)M * K * 2}.");
        if (Bhalf.SizeInBytes < (long)K * N * 2) throw new ArgumentException($"B half buffer too small: {Bhalf.SizeInBytes} < {(long)K * N * 2}.");
        if (C.SizeInBytes < (long)M * N * sizeof(float)) throw new ArgumentException($"C buffer too small: {C.SizeInBytes} < {(long)M * N * sizeof(float)}.");

        using var _ = PushContext();
        float aVal = alpha, bVal = beta;
        IntPtr alphaPtr = (IntPtr)(&aVal);
        IntPtr betaPtr = (IntPtr)(&bVal);
        var status = CuBlasNative.cublasGemmEx(
            _cublasHandle,
            CublasOperation.None, CublasOperation.None,
            N, M, K,
            alphaPtr,
            Bhalf.Handle, CuBlasNative.CUDA_R_16F, N,
            Ahalf.Handle, CuBlasNative.CUDA_R_16F, K,
            betaPtr,
            C.Handle, CuBlasNative.CUDA_R_32F, N,
            CuBlasNative.CUBLAS_COMPUTE_32F,
            0 /* CUBLAS_GEMM_DEFAULT — selects Tensor Cores under the handle's math mode */);

        // CUBLAS_STATUS_NOT_SUPPORTED is a CAPABILITY signal, not an operational failure:
        // the FP16-in / FP32-compute cublasGemmEx config isn't available on this device
        // (some GPUs/driver combos lack it even at compute capability >= 5). Surface it as
        // NotSupportedException — the semantically correct type — so callers can cleanly
        // fall back to the TF32 cublasSgemm path instead of treating it as a hard error.
        // The compute-capability floor (SupportsHgemm, cc >= 5) is necessary but not
        // sufficient, so we key off the authoritative runtime status here.
        if (status == AiDotNet.Tensors.Engines.CublasStatus.NotSupported)
        {
            throw new NotSupportedException(
                $"FP16 Tensor-Core GEMM (cublasGemmEx CUDA_R_16F in / CUBLAS_COMPUTE_32F) is not " +
                $"supported on this device (compute capability {_ccMajor}.{_ccMinor}); " +
                $"use the TF32 Gemm path instead.");
        }
        CuBlasNative.CheckCublasStatus(status, "cublasGemmEx(GemmFp16)");
    }

    private unsafe void LaunchUnaryFp16(string kernelName, IGpuBuffer input, IGpuBuffer output, int size)
    {
        if (!_kernelCache.TryGetValue(kernelName, out var kernel))
            throw new InvalidOperationException($"CUDA kernel not found: {kernelName}");
        using var _ = PushContext();
        uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);  // 1 elem/thread
        IntPtr inPtr = input.Handle, outPtr = output.Handle;
        void** args = stackalloc void*[3];
        args[0] = &inPtr; args[1] = &outPtr; args[2] = &size;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    private unsafe void LaunchConvertFp16(string kernelName, IGpuBuffer input, IGpuBuffer output, int size)
        => LaunchUnaryFp16(kernelName, input, output, size);

    /// <inheritdoc/>
    public unsafe void ConvertToFp32(IGpuBuffer input, IGpuBuffer output, int size)
    {
        if (!_kernelCache.TryGetValue("convert_fp16_to_fp32", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: convert_fp16_to_fp32");

        using var _ = PushContext();
        uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr inPtr = input.Handle;
        IntPtr outPtr = output.Handle;
        void** args = stackalloc void*[3];
        args[0] = &inPtr;
        args[1] = &outPtr;
        args[2] = &size;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    #endregion

    #region FFT and Signal Processing

    /// <inheritdoc/>
    public unsafe void FFT(IGpuBuffer inputReal, IGpuBuffer inputImag, IGpuBuffer outputReal, IGpuBuffer outputImag, int n, bool inverse)
    {
        if (!IsAvailable) throw new InvalidOperationException("CUDA backend not available");

        using var _ = PushContext();

        // Copy input to output for in-place FFT
        CudaCopyBuffer(inputReal, outputReal, n);
        CudaCopyBuffer(inputImag, outputImag, n);

        int log2n = (int)MathHelper.Log2(n);
        IntPtr outRealPtr = outputReal.Handle;
        IntPtr outImagPtr = outputImag.Handle;
        int inv = inverse ? 1 : 0;

        // Bit-reversal permutation
        if (_kernelCache.TryGetValue("bit_reverse_permutation", out var bitRevKernel))
        {
            uint grid = (uint)((n + DefaultBlockSize - 1) / DefaultBlockSize);
            void** args = stackalloc void*[4];
            args[0] = &outRealPtr;
            args[1] = &outImagPtr;
            args[2] = &n;
            args[3] = &log2n;
            LaunchKernel(bitRevKernel, grid, (uint)DefaultBlockSize, args);
        }

        // FFT butterfly stages
        if (_kernelCache.TryGetValue("fft_butterfly", out var butterflyKernel))
        {
            // Pre-allocate args outside the loop to avoid CA2014
            void** butterflyArgs = stackalloc void*[5];
            butterflyArgs[0] = &outRealPtr;
            butterflyArgs[1] = &outImagPtr;
            butterflyArgs[2] = &n;
            butterflyArgs[4] = &inv;

            for (int stride = 2; stride <= n; stride *= 2)
            {
                uint grid = (uint)((n / 2 + DefaultBlockSize - 1) / DefaultBlockSize);
                butterflyArgs[3] = &stride;
                LaunchKernel(butterflyKernel, grid, (uint)DefaultBlockSize, butterflyArgs);
            }
        }

        // Scale for inverse FFT
        if (inverse && _kernelCache.TryGetValue("scale_inverse", out var scaleKernel))
        {
            uint grid = (uint)((n + DefaultBlockSize - 1) / DefaultBlockSize);
            void** args = stackalloc void*[3];
            args[0] = &outRealPtr;
            args[1] = &outImagPtr;
            args[2] = &n;
            LaunchKernel(scaleKernel, grid, (uint)DefaultBlockSize, args);
        }
    }

    /// <inheritdoc/>
    public unsafe void RFFT(IGpuBuffer input, IGpuBuffer outputReal, IGpuBuffer outputImag, int n)
    {
        if (!IsAvailable) throw new InvalidOperationException("CUDA backend not available");

        using var _ = PushContext();

        // Allocate temporary buffers
        var tempReal = AllocateBuffer(n);
        var tempImag = AllocateBuffer(n);

        try
        {
            // Copy input to tempReal, zero tempImag
            CudaCopyBuffer(input, tempReal, n);
            CuBlasNative.CheckCudaResult(CuBlasNative.cuMemsetD32(tempImag.Handle, 0, (ulong)n), "cuMemsetD32");

            // Perform full complex FFT
            FFT(tempReal, tempImag, tempReal, tempImag, n, false);

            // Extract positive frequencies
            if (_kernelCache.TryGetValue("rfft_postprocess", out var rfftKernel))
            {
                int outLen = n / 2 + 1;
                uint grid = (uint)((outLen + DefaultBlockSize - 1) / DefaultBlockSize);
                IntPtr tempRealPtr = tempReal.Handle;
                IntPtr tempImagPtr = tempImag.Handle;
                IntPtr outRealPtr = outputReal.Handle;
                IntPtr outImagPtr = outputImag.Handle;
                void** args = stackalloc void*[5];
                args[0] = &tempRealPtr;
                args[1] = &tempImagPtr;
                args[2] = &outRealPtr;
                args[3] = &outImagPtr;
                args[4] = &n;
                LaunchKernel(rfftKernel, grid, (uint)DefaultBlockSize, args);
            }
        }
        finally
        {
            tempReal.Dispose();
            tempImag.Dispose();
        }
    }

    /// <inheritdoc/>
    public unsafe void IRFFT(IGpuBuffer inputReal, IGpuBuffer inputImag, IGpuBuffer output, int n)
    {
        if (!IsAvailable) throw new InvalidOperationException("CUDA backend not available");

        using var _ = PushContext();

        var tempReal = AllocateBuffer(n);
        var tempImag = AllocateBuffer(n);

        try
        {
            // Reconstruct negative frequencies
            if (_kernelCache.TryGetValue("irfft_preprocess", out var irfftKernel))
            {
                uint grid = (uint)((n + DefaultBlockSize - 1) / DefaultBlockSize);
                IntPtr inRealPtr = inputReal.Handle;
                IntPtr inImagPtr = inputImag.Handle;
                IntPtr tempRealPtr = tempReal.Handle;
                IntPtr tempImagPtr = tempImag.Handle;
                void** args = stackalloc void*[5];
                args[0] = &inRealPtr;
                args[1] = &inImagPtr;
                args[2] = &tempRealPtr;
                args[3] = &tempImagPtr;
                args[4] = &n;
                LaunchKernel(irfftKernel, grid, (uint)DefaultBlockSize, args);
            }

            // Perform inverse FFT
            FFT(tempReal, tempImag, tempReal, tempImag, n, true);

            // Copy real part to output
            CudaCopyBuffer(tempReal, output, n);
        }
        finally
        {
            tempReal.Dispose();
            tempImag.Dispose();
        }
    }

    /// <inheritdoc/>
    public unsafe void BatchedFFT(IGpuBuffer inputReal, IGpuBuffer inputImag, IGpuBuffer outputReal, IGpuBuffer outputImag, int batch, int n, bool inverse)
    {
        if (!IsAvailable) throw new InvalidOperationException("CUDA backend not available");

        using var _ = PushContext();

        // Copy input to output for in-place FFT
        CudaCopyBuffer(inputReal, outputReal, batch * n);
        CudaCopyBuffer(inputImag, outputImag, batch * n);

        int log2n = (int)MathHelper.Log2(n);
        IntPtr outRealPtr = outputReal.Handle;
        IntPtr outImagPtr = outputImag.Handle;
        int inv = inverse ? 1 : 0;

        // Batched bit-reversal
        if (_kernelCache.TryGetValue("batched_bit_reverse", out var bitRevKernel))
        {
            void** args = stackalloc void*[5];
            args[0] = &outRealPtr;
            args[1] = &outImagPtr;
            args[2] = &batch;
            args[3] = &n;
            args[4] = &log2n;
            LaunchKernel2D(bitRevKernel, (uint)((n + 15) / 16), (uint)batch, 1, 16, 1, args);
        }

        // Batched FFT butterfly stages
        if (_kernelCache.TryGetValue("batched_fft_butterfly", out var butterflyKernel))
        {
            // Pre-allocate args outside the loop to avoid CA2014
            void** butterflyArgs = stackalloc void*[6];
            butterflyArgs[0] = &outRealPtr;
            butterflyArgs[1] = &outImagPtr;
            butterflyArgs[2] = &batch;
            butterflyArgs[3] = &n;
            butterflyArgs[5] = &inv;

            for (int stride = 2; stride <= n; stride *= 2)
            {
                butterflyArgs[4] = &stride;
                LaunchKernel2D(butterflyKernel, (uint)((n / 2 + 15) / 16), (uint)batch, 1, 16, 1, butterflyArgs);
            }
        }

        // Scale for inverse FFT
        if (inverse && _kernelCache.TryGetValue("scale_inverse", out var scaleKernel))
        {
            int total = batch * n;
            uint grid = (uint)((total + DefaultBlockSize - 1) / DefaultBlockSize);
            void** args = stackalloc void*[3];
            args[0] = &outRealPtr;
            args[1] = &outImagPtr;
            args[2] = &total;
            LaunchKernel(scaleKernel, grid, (uint)DefaultBlockSize, args);
        }
    }

    /// <inheritdoc/>
    public unsafe void FFT2D(IGpuBuffer inputReal, IGpuBuffer inputImag, IGpuBuffer outputReal, IGpuBuffer outputImag, int height, int width, bool inverse)
    {
        if (!IsAvailable) throw new InvalidOperationException("CUDA backend not available");

        using var _ = PushContext();

        // Copy input to output for in-place FFT
        CudaCopyBuffer(inputReal, outputReal, height * width);
        CudaCopyBuffer(inputImag, outputImag, height * width);

        int log2Width = (int)MathHelper.Log2(width);
        int log2Height = (int)MathHelper.Log2(height);
        IntPtr outRealPtr = outputReal.Handle;
        IntPtr outImagPtr = outputImag.Handle;
        int inv = inverse ? 1 : 0;

        // Row-wise FFT (process each row as a separate FFT)
        if (_kernelCache.TryGetValue("fft_rows_butterfly", out var rowButterfly))
        {
            // Pre-allocate args outside the loop to avoid CA2014
            void** rowArgs = stackalloc void*[6];
            rowArgs[0] = &outRealPtr;
            rowArgs[1] = &outImagPtr;
            rowArgs[2] = &height;
            rowArgs[3] = &width;
            rowArgs[5] = &inv;

            for (int stride = 2; stride <= width; stride *= 2)
            {
                rowArgs[4] = &stride;
                LaunchKernel2D(rowButterfly, (uint)((width / 2 + 15) / 16), (uint)((height + 15) / 16), 1, 16, 16, rowArgs);
            }
        }

        // Column-wise FFT
        if (_kernelCache.TryGetValue("fft_cols_butterfly", out var colButterfly))
        {
            // Pre-allocate args outside the loop to avoid CA2014
            void** colArgs = stackalloc void*[6];
            colArgs[0] = &outRealPtr;
            colArgs[1] = &outImagPtr;
            colArgs[2] = &height;
            colArgs[3] = &width;
            colArgs[5] = &inv;

            for (int stride = 2; stride <= height; stride *= 2)
            {
                colArgs[4] = &stride;
                LaunchKernel2D(colButterfly, (uint)((height / 2 + 15) / 16), (uint)((width + 15) / 16), 1, 16, 16, colArgs);
            }
        }

        // Scale for inverse FFT
        if (inverse && _kernelCache.TryGetValue("scale_inverse", out var scaleKernel))
        {
            int total = height * width;
            uint grid = (uint)((total + DefaultBlockSize - 1) / DefaultBlockSize);
            void** args = stackalloc void*[3];
            args[0] = &outRealPtr;
            args[1] = &outImagPtr;
            args[2] = &total;
            LaunchKernel(scaleKernel, grid, (uint)DefaultBlockSize, args);
        }
    }

    /// <inheritdoc/>
    /// <inheritdoc/>
    public unsafe void BatchedFFT2D(IGpuBuffer inputReal, IGpuBuffer inputImag,
        IGpuBuffer outputReal, IGpuBuffer outputImag,
        int batch, int height, int width, bool inverse)
    {
        if (!IsAvailable) throw new InvalidOperationException("CUDA backend not available");
        if (batch <= 0 || height <= 0 || width <= 0) return;

        using var _ = PushContext();

        int sliceSize = height * width;

        if (batch == 1)
        {
            FFT2D(inputReal, inputImag, outputReal, outputImag, height, width, inverse);
            return;
        }

        // Temp slice buffers reused across all slices — fully GPU-resident, zero downloads
        var tempInR = AllocateBuffer(sliceSize);
        var tempInI = AllocateBuffer(sliceSize);
        var tempOutR = AllocateBuffer(sliceSize);
        var tempOutI = AllocateBuffer(sliceSize);
        try
        {
            for (int b = 0; b < batch; b++)
            {
                int off = b * sliceSize;
                Copy(inputReal, off, tempInR, 0, sliceSize);
                Copy(inputImag, off, tempInI, 0, sliceSize);
                FFT2D(tempInR, tempInI, tempOutR, tempOutI, height, width, inverse);
                Copy(tempOutR, 0, outputReal, off, sliceSize);
                Copy(tempOutI, 0, outputImag, off, sliceSize);
            }
        }
        finally
        {
            tempInR.Dispose(); tempInI.Dispose();
            tempOutR.Dispose(); tempOutI.Dispose();
        }
    }

    /// <inheritdoc/>
    public unsafe void ApplyWindow(IGpuBuffer input, IGpuBuffer window, IGpuBuffer output, int n)
    {
        if (!IsAvailable) throw new InvalidOperationException("CUDA backend not available");

        if (!_kernelCache.TryGetValue("apply_window", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: apply_window");

        using var _ = PushContext();
        uint grid = (uint)((n + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr inPtr = input.Handle;
        IntPtr winPtr = window.Handle;
        IntPtr outPtr = output.Handle;
        void** args = stackalloc void*[4];
        args[0] = &inPtr;
        args[1] = &winPtr;
        args[2] = &outPtr;
        args[3] = &n;
        LaunchKernel(kernel, grid, (uint)DefaultBlockSize, args);
    }

    /// <inheritdoc/>
    public unsafe void ComplexMagnitude(IGpuBuffer real, IGpuBuffer imag, IGpuBuffer magnitude, int n)
    {
        if (!IsAvailable) throw new InvalidOperationException("CUDA backend not available");

        if (!_kernelCache.TryGetValue("complex_magnitude", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: complex_magnitude");

        using var _ = PushContext();
        uint grid = (uint)((n + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr realPtr = real.Handle;
        IntPtr imagPtr = imag.Handle;
        IntPtr magPtr = magnitude.Handle;
        void** args = stackalloc void*[4];
        args[0] = &realPtr;
        args[1] = &imagPtr;
        args[2] = &magPtr;
        args[3] = &n;
        LaunchKernel(kernel, grid, (uint)DefaultBlockSize, args);
    }

    /// <inheritdoc/>
    public unsafe void ComplexPhase(IGpuBuffer real, IGpuBuffer imag, IGpuBuffer phase, int n)
    {
        if (!IsAvailable) throw new InvalidOperationException("CUDA backend not available");

        if (!_kernelCache.TryGetValue("complex_phase", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: complex_phase");

        using var _ = PushContext();
        uint grid = (uint)((n + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr realPtr = real.Handle;
        IntPtr imagPtr = imag.Handle;
        IntPtr phasePtr = phase.Handle;
        void** args = stackalloc void*[4];
        args[0] = &realPtr;
        args[1] = &imagPtr;
        args[2] = &phasePtr;
        args[3] = &n;
        LaunchKernel(kernel, grid, (uint)DefaultBlockSize, args);
    }

    /// <inheritdoc/>
    public unsafe void PolarToComplex(IGpuBuffer magnitude, IGpuBuffer phase, IGpuBuffer real, IGpuBuffer imag, int n)
    {
        if (!IsAvailable) throw new InvalidOperationException("CUDA backend not available");

        if (!_kernelCache.TryGetValue("polar_to_complex", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: polar_to_complex");

        using var _ = PushContext();
        uint grid = (uint)((n + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr magPtr = magnitude.Handle;
        IntPtr phasePtr = phase.Handle;
        IntPtr realPtr = real.Handle;
        IntPtr imagPtr = imag.Handle;
        void** args = stackalloc void*[5];
        args[0] = &magPtr;
        args[1] = &phasePtr;
        args[2] = &realPtr;
        args[3] = &imagPtr;
        args[4] = &n;
        LaunchKernel(kernel, grid, (uint)DefaultBlockSize, args);
    }

    /// <inheritdoc/>
    public unsafe void ApplyMelFilterbank(IGpuBuffer powerSpec, IGpuBuffer filterbank, IGpuBuffer melSpec, int numFrames, int numFreqs, int nMels)
    {
        if (!IsAvailable) throw new InvalidOperationException("CUDA backend not available");

        if (!_kernelCache.TryGetValue("apply_mel_filterbank", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: apply_mel_filterbank");

        using var _ = PushContext();
        IntPtr powerPtr = powerSpec.Handle;
        IntPtr fbPtr = filterbank.Handle;
        IntPtr melPtr = melSpec.Handle;
        void** args = stackalloc void*[6];
        args[0] = &powerPtr;
        args[1] = &fbPtr;
        args[2] = &melPtr;
        args[3] = &numFrames;
        args[4] = &numFreqs;
        args[5] = &nMels;
        LaunchKernel2D(kernel, (uint)((nMels + 31) / 32), (uint)numFrames, 1, 32, 1, args);
    }

    /// <inheritdoc/>
    public unsafe void PowerToDb(IGpuBuffer power, IGpuBuffer db, int n, float refValue, float minDb)
    {
        if (!IsAvailable) throw new InvalidOperationException("CUDA backend not available");

        if (!_kernelCache.TryGetValue("power_to_db", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: power_to_db");

        using var _ = PushContext();
        uint grid = (uint)((n + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr powerPtr = power.Handle;
        IntPtr dbPtr = db.Handle;
        void** args = stackalloc void*[5];
        args[0] = &powerPtr;
        args[1] = &dbPtr;
        args[2] = &n;
        args[3] = &refValue;
        args[4] = &minDb;
        LaunchKernel(kernel, grid, (uint)DefaultBlockSize, args);
    }

    /// <inheritdoc/>
    public unsafe void DbToPower(IGpuBuffer db, IGpuBuffer power, int n, float refValue)
    {
        if (!IsAvailable) throw new InvalidOperationException("CUDA backend not available");

        if (!_kernelCache.TryGetValue("db_to_power", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: db_to_power");

        using var _ = PushContext();
        uint grid = (uint)((n + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr dbPtr = db.Handle;
        IntPtr powerPtr = power.Handle;
        void** args = stackalloc void*[4];
        args[0] = &dbPtr;
        args[1] = &powerPtr;
        args[2] = &n;
        args[3] = &refValue;
        LaunchKernel(kernel, grid, (uint)DefaultBlockSize, args);
    }

    private void CudaCopyBuffer(IGpuBuffer src, IGpuBuffer dst, int size)
    {
        ulong byteSize = (ulong)size * sizeof(float);
        CuBlasNative.CheckCudaResult(
            CuBlasNative.cuMemcpyDtoD(dst.Handle, src.Handle, byteSize),
            "cuMemcpyDtoD");
    }

    public void Copy(IGpuBuffer source, int sourceOffset, IGpuBuffer destination, int destinationOffset, int length)
    {
        if (!IsAvailable)
            throw new InvalidOperationException("CUDA backend is not available.");

        using var _ = PushContext();
        IntPtr srcPtr = source.Handle + sourceOffset * sizeof(float);
        IntPtr dstPtr = destination.Handle + destinationOffset * sizeof(float);
        ulong byteSize = (ulong)length * sizeof(float);
        CuBlasNative.CheckCudaResult(
            CuBlasNative.cuMemcpyDtoD(dstPtr, srcPtr, byteSize),
            "cuMemcpyDtoD(strided)");
    }

    /// <inheritdoc/>
    public bool ArgMaxIndicesAreBitReinterpreted => false; // CUDA kernel uses (float)cast

    public unsafe void ArgMaxAxis(IGpuBuffer A, IGpuBuffer indices, int outerSize, int reduceSize)
    {
        if (outerSize <= 0 || reduceSize <= 0)
            throw new ArgumentOutOfRangeException(nameof(outerSize), "outerSize and reduceSize must be positive.");
        if ((long)outerSize * reduceSize > A.Size)
            throw new ArgumentOutOfRangeException(nameof(outerSize), $"outerSize*reduceSize ({(long)outerSize * reduceSize}) exceeds buffer A length ({A.Size}).");

        if (!_kernelCache.TryGetValue("argmax_axis", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: argmax_axis");

        using var _ = PushContext();
        IntPtr inputPtr = A.Handle;
        IntPtr indicesPtr = indices.Handle;
        int outerSizeVal = outerSize;
        int reduceSizeVal = reduceSize;
        void** args = stackalloc void*[4];
        args[0] = &inputPtr;
        args[1] = &indicesPtr;
        args[2] = &outerSizeVal;
        args[3] = &reduceSizeVal;
        uint grid = (uint)((outerSize + DefaultBlockSize - 1) / DefaultBlockSize);
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    public unsafe void GenerateRandomUniform(IGpuBuffer output, int size, float min, float max, ulong seed)
    {
        if (!_kernelCache.TryGetValue("generate_random_uniform", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: generate_random_uniform");

        using var _ = PushContext();
        IntPtr outputPtr = output.Handle;
        int sizeVal = size;
        float minVal = min;
        float maxVal = max;
        ulong seedVal = seed;
        void** args = stackalloc void*[5];
        args[0] = &outputPtr;
        args[1] = &sizeVal;
        args[2] = &minVal;
        args[3] = &maxVal;
        args[4] = &seedVal;
        uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    public unsafe void GenerateRandomNormal(IGpuBuffer output, int size, float mean, float stdDev, ulong seed)
    {
        if (!_kernelCache.TryGetValue("generate_random_normal", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: generate_random_normal");

        using var _ = PushContext();
        IntPtr outputPtr = output.Handle;
        int sizeVal = size;
        float meanVal = mean;
        float stdDevVal = stdDev;
        ulong seedVal = seed;
        void** args = stackalloc void*[5];
        args[0] = &outputPtr;
        args[1] = &sizeVal;
        args[2] = &meanVal;
        args[3] = &stdDevVal;
        args[4] = &seedVal;
        uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    public unsafe void GenerateSecureRandomUniform(IGpuBuffer output, int size, float min, float max)
    {
        if (size <= 0) return;
        using var _ = PushContext();
        var data = new float[size];
        try
        {
            Helpers.SimdRandom.SecureFillFloats(data.AsSpan());
            float range = max - min;
            for (int i = 0; i < size; i++) data[i] = data[i] * range + min;
            fixed (float* ptr = data)
            {
                CuBlasNative.CheckCudaResult(
                    CuBlasNative.cuMemcpyHtoD(output.Handle, (IntPtr)ptr, (ulong)(size * sizeof(float))),
                    "cuMemcpyHtoD (GenerateSecureRandomUniform)");
            }
        }
        finally { Array.Clear(data, 0, size); }
    }

    public unsafe void RbfForward(IGpuBuffer input, IGpuBuffer centers, IGpuBuffer epsilons, IGpuBuffer output, int batchSize, int numCenters, int inputDim)
    {
        if (!_kernelCache.TryGetValue("rbf_forward", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: rbf_forward");

        using var _ = PushContext();
        IntPtr inputPtr = input.Handle;
        IntPtr centersPtr = centers.Handle;
        IntPtr epsilonsPtr = epsilons.Handle;
        IntPtr outputPtr = output.Handle;
        int batchSizeVal = batchSize;
        int numCentersVal = numCenters;
        int inputDimVal = inputDim;
        void** args = stackalloc void*[7];
        args[0] = &inputPtr;
        args[1] = &centersPtr;
        args[2] = &epsilonsPtr;
        args[3] = &outputPtr;
        args[4] = &batchSizeVal;
        args[5] = &numCentersVal;
        args[6] = &inputDimVal;
        uint totalWork = (uint)(batchSize * numCenters);
        uint grid = (totalWork + DefaultBlockSize - 1) / DefaultBlockSize;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    public unsafe void StdpUpdate(IGpuBuffer weights, IGpuBuffer preTrace, IGpuBuffer postTrace, IGpuBuffer preSpike, IGpuBuffer postSpike,
        float ltpRate, float ltdRate, float homeostasisRate, float minWeight, float maxWeight, int numPre, int numPost)
    {
        if (!_kernelCache.TryGetValue("stdp_update", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: stdp_update");

        using var _ = PushContext();
        IntPtr wPtr = weights.Handle;
        IntPtr preTPtr = preTrace.Handle;
        IntPtr postTPtr = postTrace.Handle;
        IntPtr preSPtr = preSpike.Handle;
        IntPtr postSPtr = postSpike.Handle;
        int numPreVal = numPre;
        int numPostVal = numPost;
        void** args = stackalloc void*[12];
        args[0] = &wPtr;
        args[1] = &preTPtr;
        args[2] = &postTPtr;
        args[3] = &preSPtr;
        args[4] = &postSPtr;
        args[5] = &ltpRate;
        args[6] = &ltdRate;
        args[7] = &homeostasisRate;
        args[8] = &minWeight;
        args[9] = &maxWeight;
        args[10] = &numPreVal;
        args[11] = &numPostVal;
        uint totalWork = (uint)(numPre * numPost);
        uint grid = (totalWork + DefaultBlockSize - 1) / DefaultBlockSize;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    public unsafe void UpdateTraces(IGpuBuffer traces, IGpuBuffer spikes, IGpuBuffer input, float decay, float threshold, int size)
    {
        if (!_kernelCache.TryGetValue("update_traces", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: update_traces");

        using var _ = PushContext();
        IntPtr tracesPtr = traces.Handle;
        IntPtr spikesPtr = spikes.Handle;
        IntPtr inputPtr = input.Handle;
        int sizeVal = size;
        void** args = stackalloc void*[6];
        args[0] = &tracesPtr;
        args[1] = &spikesPtr;
        args[2] = &inputPtr;
        args[3] = &decay;
        args[4] = &threshold;
        args[5] = &sizeVal;
        uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    #endregion

    #region Hyperbolic Geometry Operations

    public unsafe void PoincareProject(IGpuBuffer input, IGpuBuffer output, int batchSize, int dim, float curvature, float epsilon = 1e-5f)
    {
        if (!_kernelCache.TryGetValue("poincare_project", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: poincare_project");

        using var _ = PushContext();
        uint grid = (uint)((batchSize + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr inputPtr = input.Handle;
        IntPtr outputPtr = output.Handle;

        void** args = stackalloc void*[6];
        args[0] = &inputPtr;
        args[1] = &outputPtr;
        args[2] = &batchSize;
        args[3] = &dim;
        args[4] = &curvature;
        args[5] = &epsilon;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    public unsafe void MobiusAdd(IGpuBuffer x, IGpuBuffer y, IGpuBuffer output, int batchSize, int dim, float curvature)
    {
        if (!_kernelCache.TryGetValue("mobius_add", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: mobius_add");

        using var _ = PushContext();
        uint grid = (uint)((batchSize + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr xPtr = x.Handle;
        IntPtr yPtr = y.Handle;
        IntPtr outputPtr = output.Handle;

        void** args = stackalloc void*[6];
        args[0] = &xPtr;
        args[1] = &yPtr;
        args[2] = &outputPtr;
        args[3] = &batchSize;
        args[4] = &dim;
        args[5] = &curvature;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    public unsafe void PoincareExpMap(IGpuBuffer basePoint, IGpuBuffer tangentVec, IGpuBuffer output, int batchSize, int dim, float curvature)
    {
        if (!_kernelCache.TryGetValue("poincare_exp_map", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: poincare_exp_map");

        using var _ = PushContext();
        uint grid = (uint)((batchSize + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr basePtr = basePoint.Handle;
        IntPtr tangentPtr = tangentVec.Handle;
        IntPtr outputPtr = output.Handle;

        void** args = stackalloc void*[6];
        args[0] = &basePtr;
        args[1] = &tangentPtr;
        args[2] = &outputPtr;
        args[3] = &batchSize;
        args[4] = &dim;
        args[5] = &curvature;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    public unsafe void PoincareDistance(IGpuBuffer x, IGpuBuffer y, IGpuBuffer output, int batchSize, int dim, float curvature)
    {
        if (!_kernelCache.TryGetValue("poincare_distance", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: poincare_distance");

        using var _ = PushContext();
        uint grid = (uint)((batchSize + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr xPtr = x.Handle;
        IntPtr yPtr = y.Handle;
        IntPtr outputPtr = output.Handle;

        void** args = stackalloc void*[6];
        args[0] = &xPtr;
        args[1] = &yPtr;
        args[2] = &outputPtr;
        args[3] = &batchSize;
        args[4] = &dim;
        args[5] = &curvature;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    public unsafe void HyperbolicLinearForward(IGpuBuffer input, IGpuBuffer weights, IGpuBuffer biases, IGpuBuffer output,
        int batchSize, int inputFeatures, int outputFeatures, float curvature, float epsilon)
    {
        if (!_kernelCache.TryGetValue("hyperbolic_linear_forward", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: hyperbolic_linear_forward");

        using var _ = PushContext();
        int totalThreads = batchSize * outputFeatures;
        uint grid = (uint)((totalThreads + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr inputPtr = input.Handle;
        IntPtr weightsPtr = weights.Handle;
        IntPtr biasesPtr = biases.Handle;
        IntPtr outputPtr = output.Handle;

        void** args = stackalloc void*[9];
        args[0] = &inputPtr;
        args[1] = &weightsPtr;
        args[2] = &biasesPtr;
        args[3] = &outputPtr;
        args[4] = &batchSize;
        args[5] = &inputFeatures;
        args[6] = &outputFeatures;
        args[7] = &curvature;
        args[8] = &epsilon;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    /// <inheritdoc/>
    public unsafe void HyperbolicLinearBackwardInput(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer weights, IGpuBuffer gradInput,
        int batchSize, int inputFeatures, int outputFeatures, float curvature)
    {
        if (!_kernelCache.TryGetValue("hyperbolic_linear_backward_input", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: hyperbolic_linear_backward_input");

        using var _ = PushContext();
        int totalThreads = batchSize * inputFeatures;
        uint grid = (uint)((totalThreads + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr gradOutputPtr = gradOutput.Handle;
        IntPtr inputPtr = input.Handle;
        IntPtr weightsPtr = weights.Handle;
        IntPtr gradInputPtr = gradInput.Handle;

        void** args = stackalloc void*[8];
        args[0] = &gradOutputPtr;
        args[1] = &inputPtr;
        args[2] = &weightsPtr;
        args[3] = &gradInputPtr;
        args[4] = &batchSize;
        args[5] = &inputFeatures;
        args[6] = &outputFeatures;
        args[7] = &curvature;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    /// <inheritdoc/>
    public unsafe void HyperbolicLinearBackwardWeights(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradWeights,
        int batchSize, int inputFeatures, int outputFeatures, float curvature)
    {
        if (!_kernelCache.TryGetValue("hyperbolic_linear_backward_weights", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: hyperbolic_linear_backward_weights");

        using var _ = PushContext();
        int totalThreads = outputFeatures * inputFeatures;
        uint grid = (uint)((totalThreads + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr gradOutputPtr = gradOutput.Handle;
        IntPtr inputPtr = input.Handle;
        IntPtr gradWeightsPtr = gradWeights.Handle;

        void** args = stackalloc void*[7];
        args[0] = &gradOutputPtr;
        args[1] = &inputPtr;
        args[2] = &gradWeightsPtr;
        args[3] = &batchSize;
        args[4] = &inputFeatures;
        args[5] = &outputFeatures;
        args[6] = &curvature;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    /// <inheritdoc/>
    public unsafe void HyperbolicLinearBackwardBiases(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradBiases,
        int batchSize, int inputFeatures, int outputFeatures, float curvature)
    {
        if (!_kernelCache.TryGetValue("hyperbolic_linear_backward_biases", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: hyperbolic_linear_backward_biases");

        using var _ = PushContext();
        int totalThreads = outputFeatures;  // One thread per bias element
        uint grid = (uint)((totalThreads + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr gradOutputPtr = gradOutput.Handle;
        IntPtr inputPtr = input.Handle;
        IntPtr gradBiasesPtr = gradBiases.Handle;

        void** args = stackalloc void*[7];
        args[0] = &gradOutputPtr;
        args[1] = &inputPtr;
        args[2] = &gradBiasesPtr;
        args[3] = &batchSize;
        args[4] = &inputFeatures;
        args[5] = &outputFeatures;
        args[6] = &curvature;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    #endregion

    #region Octonion Algebra Operations

    public unsafe void OctonionMultiply(IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, int count)
    {
        if (!_kernelCache.TryGetValue("octonion_multiply", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: octonion_multiply");

        using var _ = PushContext();
        uint grid = (uint)((count + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr aPtr = a.Handle;
        IntPtr bPtr = b.Handle;
        IntPtr outputPtr = output.Handle;

        void** args = stackalloc void*[4];
        args[0] = &aPtr;
        args[1] = &bPtr;
        args[2] = &outputPtr;
        args[3] = &count;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    public unsafe void OctonionAdd(IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, int count)
    {
        if (!_kernelCache.TryGetValue("octonion_add", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: octonion_add");

        using var _ = PushContext();
        int totalElements = count * 8;
        uint grid = (uint)((totalElements + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr aPtr = a.Handle;
        IntPtr bPtr = b.Handle;
        IntPtr outputPtr = output.Handle;

        void** args = stackalloc void*[4];
        args[0] = &aPtr;
        args[1] = &bPtr;
        args[2] = &outputPtr;
        args[3] = &count;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    public unsafe void OctonionLinearForward(IGpuBuffer input, IGpuBuffer weights, IGpuBuffer biases, IGpuBuffer output,
        int batchSize, int inputFeatures, int outputFeatures)
    {
        if (!_kernelCache.TryGetValue("octonion_linear_forward", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: octonion_linear_forward");

        using var _ = PushContext();
        int totalThreads = batchSize * outputFeatures;
        uint grid = (uint)((totalThreads + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr inputPtr = input.Handle;
        IntPtr weightsPtr = weights.Handle;
        IntPtr biasesPtr = biases.Handle;
        IntPtr outputPtr = output.Handle;

        void** args = stackalloc void*[7];
        args[0] = &inputPtr;
        args[1] = &weightsPtr;
        args[2] = &biasesPtr;
        args[3] = &outputPtr;
        args[4] = &batchSize;
        args[5] = &inputFeatures;
        args[6] = &outputFeatures;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    public unsafe void OctonionLinearBackwardInput(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer weights, IGpuBuffer gradInput,
        int batchSize, int inputFeatures, int outputFeatures)
    {
        if (!_kernelCache.TryGetValue("octonion_linear_backward_input", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: octonion_linear_backward_input");

        using var _ = PushContext();
        int totalThreads = batchSize * inputFeatures;
        uint grid = (uint)((totalThreads + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr gradOutputPtr = gradOutput.Handle;
        IntPtr inputPtr = input.Handle;
        IntPtr weightsPtr = weights.Handle;
        IntPtr gradInputPtr = gradInput.Handle;

        void** args = stackalloc void*[7];
        args[0] = &gradOutputPtr;
        args[1] = &inputPtr;
        args[2] = &weightsPtr;
        args[3] = &gradInputPtr;
        args[4] = &batchSize;
        args[5] = &inputFeatures;
        args[6] = &outputFeatures;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    public unsafe void OctonionLinearBackwardWeights(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradWeights,
        int batchSize, int inputFeatures, int outputFeatures)
    {
        if (!_kernelCache.TryGetValue("octonion_linear_backward_weights", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: octonion_linear_backward_weights");

        using var _ = PushContext();
        int totalThreads = outputFeatures * inputFeatures;
        uint grid = (uint)((totalThreads + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr gradOutputPtr = gradOutput.Handle;
        IntPtr inputPtr = input.Handle;
        IntPtr gradWeightsPtr = gradWeights.Handle;

        void** args = stackalloc void*[6];
        args[0] = &gradOutputPtr;
        args[1] = &inputPtr;
        args[2] = &gradWeightsPtr;
        args[3] = &batchSize;
        args[4] = &inputFeatures;
        args[5] = &outputFeatures;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    public unsafe void OctonionLinearBackwardBiases(IGpuBuffer gradOutput, IGpuBuffer gradBiases,
        int batchSize, int outputFeatures)
    {
        if (!_kernelCache.TryGetValue("octonion_linear_backward_biases", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: octonion_linear_backward_biases");

        using var _ = PushContext();
        uint grid = (uint)((outputFeatures + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr gradOutputPtr = gradOutput.Handle;
        IntPtr gradBiasesPtr = gradBiases.Handle;

        void** args = stackalloc void*[4];
        args[0] = &gradOutputPtr;
        args[1] = &gradBiasesPtr;
        args[2] = &batchSize;
        args[3] = &outputFeatures;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    #endregion

    #region Fused Kernel Operations

    public void HyperbolicLinearForwardFused(IGpuBuffer input, IGpuBuffer weights, IGpuBuffer biases, IGpuBuffer output,
        int batchSize, int inputFeatures, int outputFeatures, float curvature, float epsilon)
        => HyperbolicLinearForward(input, weights, biases, output, batchSize, inputFeatures, outputFeatures, curvature, epsilon);

    public unsafe void OctonionLinearForwardFusedReLU(IGpuBuffer input, IGpuBuffer weights, IGpuBuffer biases, IGpuBuffer output,
        int batchSize, int inputFeatures, int outputFeatures)
    {
        if (!_kernelCache.TryGetValue("octonion_linear_forward_fused_relu", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: octonion_linear_forward_fused_relu");

        using var _ = PushContext();
        int totalOutputs = batchSize * outputFeatures;
        uint grid = (uint)((totalOutputs + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr inputPtr = input.Handle;
        IntPtr weightsPtr = weights.Handle;
        IntPtr biasesPtr = biases.Handle;
        IntPtr outputPtr = output.Handle;

        void** args = stackalloc void*[7];
        args[0] = &inputPtr;
        args[1] = &weightsPtr;
        args[2] = &biasesPtr;
        args[3] = &outputPtr;
        args[4] = &batchSize;
        args[5] = &inputFeatures;
        args[6] = &outputFeatures;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    #endregion

    #region Complex Tensor Operations

    public unsafe void ComplexMultiply(IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, int numPairs)
    {
        if (numPairs <= 0) return;
        if (numPairs * 2 > a.Size || numPairs * 2 > b.Size || numPairs * 2 > output.Size)
            throw new ArgumentException($"numPairs ({numPairs}) requires {numPairs * 2} elements but buffer sizes are a={a.Size}, b={b.Size}, out={output.Size}.");
        if (!_kernelCache.TryGetValue("complex_multiply", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: complex_multiply");
        using var _ = PushContext();
        uint grid = (uint)((numPairs + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr pA = a.Handle, pB = b.Handle, pO = output.Handle;
        void** args = stackalloc void*[4];
        args[0] = &pA; args[1] = &pB; args[2] = &pO; args[3] = &numPairs;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    public unsafe void ComplexConjugate(IGpuBuffer input, IGpuBuffer output, int numPairs)
    {
        if (numPairs <= 0) return;
        if (numPairs * 2 > input.Size || numPairs * 2 > output.Size)
            throw new ArgumentException($"numPairs ({numPairs}) requires {numPairs * 2} elements but buffer sizes are in={input.Size}, out={output.Size}.");
        if (!_kernelCache.TryGetValue("complex_conjugate", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: complex_conjugate");
        using var _ = PushContext();
        uint grid = (uint)((numPairs + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr pI = input.Handle, pO = output.Handle;
        void** args = stackalloc void*[3];
        args[0] = &pI; args[1] = &pO; args[2] = &numPairs;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    public unsafe void ComplexMagnitude(IGpuBuffer input, IGpuBuffer output, int numPairs)
    {
        if (numPairs <= 0) return;
        if (numPairs * 2 > input.Size)
            throw new ArgumentException($"numPairs ({numPairs}) requires {numPairs * 2} elements but input buffer has {input.Size}.");
        if (numPairs > output.Size)
            throw new ArgumentException($"numPairs ({numPairs}) exceeds output buffer size ({output.Size}).");
        if (!_kernelCache.TryGetValue("complex_magnitude", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: complex_magnitude");
        using var _ = PushContext();
        uint grid = (uint)((numPairs + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr pI = input.Handle, pO = output.Handle;
        void** args = stackalloc void*[3];
        args[0] = &pI; args[1] = &pO; args[2] = &numPairs;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    // --- Split-buffer native Complex<T> operations ---

    private static void ValidateSplitBuffers(int n, string opName, params IGpuBuffer[] buffers)
    {
        foreach (var buf in buffers)
        {
            if (buf is null) throw new ArgumentNullException(opName, "GPU buffer cannot be null.");
            if (n > buf.Size) throw new ArgumentException($"{opName}: n ({n}) exceeds buffer size ({buf.Size}).");
        }
    }

    public unsafe void SplitComplexMultiply(IGpuBuffer aReal, IGpuBuffer aImag, IGpuBuffer bReal, IGpuBuffer bImag,
        IGpuBuffer outReal, IGpuBuffer outImag, int n)
    {
        if (n <= 0) return;
        ValidateSplitBuffers(n, nameof(SplitComplexMultiply), aReal, aImag, bReal, bImag, outReal, outImag);
        if (!_kernelCache.TryGetValue("split_complex_multiply", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: split_complex_multiply. Register CudaComplexKernels.");
        using var _ = PushContext();
        uint grid = (uint)((n + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr pAR = aReal.Handle, pAI = aImag.Handle, pBR = bReal.Handle, pBI = bImag.Handle;
        IntPtr pOR = outReal.Handle, pOI = outImag.Handle;
        void** args = stackalloc void*[7];
        args[0] = &pAR; args[1] = &pAI; args[2] = &pBR; args[3] = &pBI;
        args[4] = &pOR; args[5] = &pOI; args[6] = &n;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    public unsafe void SplitComplexConjugate(IGpuBuffer inReal, IGpuBuffer inImag,
        IGpuBuffer outReal, IGpuBuffer outImag, int n)
    {
        if (n <= 0) return;
        ValidateSplitBuffers(n, nameof(SplitComplexConjugate), inReal, inImag, outReal, outImag);
        if (!_kernelCache.TryGetValue("split_complex_conjugate", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: split_complex_conjugate. Register CudaComplexKernels.");
        using var _ = PushContext();
        uint grid = (uint)((n + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr pIR = inReal.Handle, pII = inImag.Handle, pOR = outReal.Handle, pOI = outImag.Handle;
        void** args = stackalloc void*[5];
        args[0] = &pIR; args[1] = &pII; args[2] = &pOR; args[3] = &pOI; args[4] = &n;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    public unsafe void SplitComplexMagnitude(IGpuBuffer inReal, IGpuBuffer inImag, IGpuBuffer outMag, int n)
    {
        if (n <= 0) return;
        ValidateSplitBuffers(n, nameof(SplitComplexMagnitude), inReal, inImag, outMag);
        if (!_kernelCache.TryGetValue("split_complex_magnitude", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: split_complex_magnitude. Register CudaComplexKernels.");
        using var _ = PushContext();
        uint grid = (uint)((n + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr pIR = inReal.Handle, pII = inImag.Handle, pO = outMag.Handle;
        void** args = stackalloc void*[4];
        args[0] = &pIR; args[1] = &pII; args[2] = &pO; args[3] = &n;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    public unsafe void SplitComplexMagnitudeSquared(IGpuBuffer inReal, IGpuBuffer inImag, IGpuBuffer outMagSq, int n)
    {
        if (n <= 0) return;
        ValidateSplitBuffers(n, nameof(SplitComplexMagnitudeSquared), inReal, inImag, outMagSq);
        if (!_kernelCache.TryGetValue("split_complex_magnitude_squared", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: split_complex_magnitude_squared. Register CudaComplexKernels.");
        using var _ = PushContext();
        uint grid = (uint)((n + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr pIR = inReal.Handle, pII = inImag.Handle, pO = outMagSq.Handle;
        void** args = stackalloc void*[4];
        args[0] = &pIR; args[1] = &pII; args[2] = &pO; args[3] = &n;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    public unsafe void SplitComplexPhase(IGpuBuffer inReal, IGpuBuffer inImag, IGpuBuffer outPhase, int n)
    {
        if (n <= 0) return;
        ValidateSplitBuffers(n, nameof(SplitComplexPhase), inReal, inImag, outPhase);
        if (!_kernelCache.TryGetValue("split_complex_phase", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: split_complex_phase. Register CudaComplexKernels.");
        using var _ = PushContext();
        uint grid = (uint)((n + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr pIR = inReal.Handle, pII = inImag.Handle, pO = outPhase.Handle;
        void** args = stackalloc void*[4];
        args[0] = &pIR; args[1] = &pII; args[2] = &pO; args[3] = &n;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    public unsafe void SplitComplexFromPolar(IGpuBuffer mag, IGpuBuffer phase,
        IGpuBuffer outReal, IGpuBuffer outImag, int n)
    {
        if (n <= 0) return;
        ValidateSplitBuffers(n, nameof(SplitComplexFromPolar), mag, phase, outReal, outImag);
        if (!_kernelCache.TryGetValue("split_complex_from_polar", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: split_complex_from_polar. Register CudaComplexKernels.");
        using var _ = PushContext();
        uint grid = (uint)((n + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr pM = mag.Handle, pP = phase.Handle, pOR = outReal.Handle, pOI = outImag.Handle;
        void** args = stackalloc void*[5];
        args[0] = &pM; args[1] = &pP; args[2] = &pOR; args[3] = &pOI; args[4] = &n;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    public unsafe void SplitComplexScale(IGpuBuffer inReal, IGpuBuffer inImag,
        IGpuBuffer outReal, IGpuBuffer outImag, float scalar, int n)
    {
        if (n <= 0) return;
        ValidateSplitBuffers(n, nameof(SplitComplexScale), inReal, inImag, outReal, outImag);
        if (!_kernelCache.TryGetValue("split_complex_scale", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: split_complex_scale. Register CudaComplexKernels.");
        using var _ = PushContext();
        uint grid = (uint)((n + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr pIR = inReal.Handle, pII = inImag.Handle, pOR = outReal.Handle, pOI = outImag.Handle;
        void** args = stackalloc void*[6];
        args[0] = &pIR; args[1] = &pII; args[2] = &pOR; args[3] = &pOI; args[4] = &scalar; args[5] = &n;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    public unsafe void SplitComplexAdd(IGpuBuffer aReal, IGpuBuffer aImag, IGpuBuffer bReal, IGpuBuffer bImag,
        IGpuBuffer outReal, IGpuBuffer outImag, int n)
    {
        if (n <= 0) return;
        ValidateSplitBuffers(n, nameof(SplitComplexAdd), aReal, aImag, bReal, bImag, outReal, outImag);
        if (!_kernelCache.TryGetValue("split_complex_add", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: split_complex_add. Register CudaComplexKernels.");
        using var _ = PushContext();
        uint grid = (uint)((n + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr pAR = aReal.Handle, pAI = aImag.Handle, pBR = bReal.Handle, pBI = bImag.Handle;
        IntPtr pOR = outReal.Handle, pOI = outImag.Handle;
        void** args = stackalloc void*[7];
        args[0] = &pAR; args[1] = &pAI; args[2] = &pBR; args[3] = &pBI;
        args[4] = &pOR; args[5] = &pOI; args[6] = &n;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    public unsafe void SplitComplexCrossSpectral(IGpuBuffer xReal, IGpuBuffer xImag, IGpuBuffer yReal, IGpuBuffer yImag,
        IGpuBuffer outReal, IGpuBuffer outImag, int n)
    {
        if (n <= 0) return;
        ValidateSplitBuffers(n, nameof(SplitComplexCrossSpectral), xReal, xImag, yReal, yImag, outReal, outImag);
        if (!_kernelCache.TryGetValue("split_complex_cross_spectral", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: split_complex_cross_spectral. Register CudaComplexKernels.");
        using var _ = PushContext();
        uint grid = (uint)((n + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr pXR = xReal.Handle, pXI = xImag.Handle, pYR = yReal.Handle, pYI = yImag.Handle;
        IntPtr pOR = outReal.Handle, pOI = outImag.Handle;
        void** args = stackalloc void*[7];
        args[0] = &pXR; args[1] = &pXI; args[2] = &pYR; args[3] = &pYI;
        args[4] = &pOR; args[5] = &pOI; args[6] = &n;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    public unsafe void SplitComplexTopK(IGpuBuffer inReal, IGpuBuffer inImag, IGpuBuffer outReal, IGpuBuffer outImag, int n, int k)
    {
        if (n <= 0 || k <= 0) return;
        ValidateSplitBuffers(n, nameof(SplitComplexTopK), inReal, inImag, outReal, outImag);
        if (!_kernelCache.TryGetValue("split_complex_topk", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: split_complex_topk");

        // Compute threshold on CPU: download magnitudes, find K-th largest
        var magBuf = AllocateBuffer(n);
        try
        {
        SplitComplexMagnitudeSquared(inReal, inImag, magBuf, n);
        var magData = DownloadBuffer(magBuf);
        Array.Sort(magData);
        Array.Reverse(magData);
        float threshold = k <= n ? magData[Math.Min(k, n) - 1] : 0f;

        using var _ = PushContext();
        uint grid = (uint)((n + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr pIR = inReal.Handle, pII = inImag.Handle, pOR = outReal.Handle, pOI = outImag.Handle;
        void** args = stackalloc void*[6];
        args[0] = &pIR; args[1] = &pII; args[2] = &pOR; args[3] = &pOI; args[4] = &threshold; args[5] = &n;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
        }
        finally { magBuf.Dispose(); }
    }

    public unsafe void SoftmaxRows(IGpuBuffer input, IGpuBuffer output, int rows, int cols)
    {
        if (rows <= 0 || cols <= 0) return;
        if (!_kernelCache.TryGetValue("softmax_rows", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: softmax_rows");
        using var _ = PushContext();
        uint grid = (uint)rows;
        int blockSize = Math.Min(256, cols);
        int sharedMem = blockSize * sizeof(float);
        IntPtr pI = input.Handle, pO = output.Handle;
        void** args = stackalloc void*[4];
        args[0] = &pI; args[1] = &pO; args[2] = &rows; args[3] = &cols;
        // Was an UNCHECKED launch on the DEFAULT stream (IntPtr.Zero) while every other op uses _stream — a fault
        // here went unsurfaced and could read buffers out of order vs the rest of the pipeline. Check the result and
        // use _stream for ordering.
        CuBlasNative.CheckCudaResult(
            CudaNativeBindings.cuLaunchKernel(kernel, grid, 1, 1, (uint)blockSize, 1, 1,
                (uint)sharedMem, _stream, (IntPtr)args, IntPtr.Zero),
            "cuLaunchKernel(softmax_rows)");
    }

    // ─── HRR binding primitives (issue #248) ────────────────────────

    /// <inheritdoc/>
    public unsafe void SplitComplexUnitPhaseCodebook(
        IGpuBuffer outReal, IGpuBuffer outImag,
        int seed, int V, int D, bool kPsk, int k)
    {
        long total = (long)V * D;
        if (total <= 0) return;
        if (total > int.MaxValue)
            throw new ArgumentException($"V*D = {total} exceeds int.MaxValue.");
        if (kPsk && k <= 0)
            throw new ArgumentOutOfRangeException(nameof(k),
                "k must be positive when kPsk is true.");
        int n = (int)total;
        ValidateSplitBuffers(n, nameof(SplitComplexUnitPhaseCodebook), outReal, outImag);
        if (!_kernelCache.TryGetValue("hrr_unit_phase_codebook", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: hrr_unit_phase_codebook. Register CudaComplexKernels.");
        using var _ = PushContext();
        uint grid = (uint)((n + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr pOR = outReal.Handle, pOI = outImag.Handle;
        int kPskI = kPsk ? 1 : 0;
        void** args = stackalloc void*[7];
        args[0] = &pOR; args[1] = &pOI; args[2] = &seed;
        args[3] = &V; args[4] = &D; args[5] = &kPskI; args[6] = &k;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    /// <inheritdoc/>
    public unsafe void SplitComplexPhaseCoherenceDecode(
        IGpuBuffer codesReal, IGpuBuffer codesImag,
        IGpuBuffer queryReal, IGpuBuffer queryImag,
        IGpuBuffer outScores,
        int V, int D)
    {
        if (V <= 0 || D <= 0) return;
        long codeCount = (long)V * D;
        if (codesReal is null || codesImag is null || queryReal is null || queryImag is null || outScores is null)
            throw new ArgumentNullException(nameof(codesReal),
                "All five GPU buffers (codesReal/codesImag/queryReal/queryImag/outScores) must be non-null.");
        if (codesReal.Size < codeCount || codesImag.Size < codeCount)
            throw new ArgumentException(
                $"codes buffers must each hold at least V*D = {codeCount} elements " +
                $"(got {codesReal.Size}, {codesImag.Size}).");
        if (queryReal.Size < D || queryImag.Size < D)
            throw new ArgumentException(
                $"query buffers must each hold at least D = {D} elements " +
                $"(got {queryReal.Size}, {queryImag.Size}).");
        if (outScores.Size < V)
            throw new ArgumentException(
                $"outScores must hold at least V = {V} elements (got {outScores.Size}).");
        if (!_kernelCache.TryGetValue("hrr_phase_coherence_decode", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: hrr_phase_coherence_decode. Register CudaComplexKernels.");
        using var _ = PushContext();
        uint grid = (uint)V;
        int blockSize = Math.Min(DefaultBlockSize, NextPowerOfTwoForReduction(D));
        uint sharedMem = (uint)blockSize * sizeof(float);
        IntPtr pCR = codesReal.Handle, pCI = codesImag.Handle;
        IntPtr pQR = queryReal.Handle, pQI = queryImag.Handle;
        IntPtr pOS = outScores.Handle;
        void** args = stackalloc void*[7];
        args[0] = &pCR; args[1] = &pCI;
        args[2] = &pQR; args[3] = &pQI;
        args[4] = &pOS; args[5] = &V; args[6] = &D;
        // Route through the shared launch helper so this runs on _stream
        // and cuLaunchKernel failures are surfaced by CheckCudaResult,
        // matching the rest of the backend.
        LaunchKernelWithSharedMem(kernel, grid, (uint)blockSize, sharedMem, args);
    }

    /// <inheritdoc/>
    public unsafe void SplitComplexHrrBindAccumulate(
        IGpuBuffer keyCodeReal, IGpuBuffer keyCodeImag,
        IGpuBuffer valPermCodeReal, IGpuBuffer valPermCodeImag,
        IGpuBuffer keyIds, IGpuBuffer valIds,
        IGpuBuffer memoryReal, IGpuBuffer memoryImag,
        int N, int D)
    {
        if (N <= 0 || D <= 0) return;
        if (keyCodeReal is null || keyCodeImag is null || valPermCodeReal is null || valPermCodeImag is null
            || keyIds is null || valIds is null || memoryReal is null || memoryImag is null)
            throw new ArgumentNullException(nameof(keyCodeReal),
                "All eight GPU buffers must be non-null for SplitComplexHrrBindAccumulate.");
        if (keyIds.Size < N || valIds.Size < N)
            throw new ArgumentException(
                $"ID buffers must each hold at least N = {N} elements " +
                $"(got keyIds={keyIds.Size}, valIds={valIds.Size}).");
        if (memoryReal.Size < D || memoryImag.Size < D)
            throw new ArgumentException(
                $"Memory buffers must each hold at least D = {D} elements " +
                $"(got memoryReal={memoryReal.Size}, memoryImag={memoryImag.Size}).");
        if (keyCodeReal.Size < D || keyCodeImag.Size < D
            || valPermCodeReal.Size < D || valPermCodeImag.Size < D)
            throw new ArgumentException(
                $"Each codebook buffer must hold at least one full row of D = {D} elements.");
        // Derive vocabulary sizes from codebook capacities so the
        // kernel can reject out-of-range ids without reading past
        // allocated memory — mirrors the CPU path's range check in
        // NativeHRRBindAccumulate and the guards in Vulkan/WebGPU.
        long nKeysL = keyCodeReal.Size / D;
        long nValsL = valPermCodeReal.Size / D;
        if (nKeysL <= 0 || nValsL <= 0)
            throw new ArgumentException(
                $"Codebook buffers must hold at least one full row of D = {D} elements.");
        if (nKeysL > int.MaxValue || nValsL > int.MaxValue)
            throw new ArgumentException("Codebook vocabulary size exceeds int.MaxValue.");
        int nKeys = (int)nKeysL;
        int nVals = (int)nValsL;
        if (!_kernelCache.TryGetValue("hrr_bind_accumulate", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: hrr_bind_accumulate. Register CudaComplexKernels.");
        using var _ = PushContext();
        uint grid = (uint)((D + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr pKR = keyCodeReal.Handle, pKI = keyCodeImag.Handle;
        IntPtr pVR = valPermCodeReal.Handle, pVI = valPermCodeImag.Handle;
        IntPtr pKId = keyIds.Handle, pVId = valIds.Handle;
        IntPtr pMR = memoryReal.Handle, pMI = memoryImag.Handle;
        void** args = stackalloc void*[12];
        args[0] = &pKR; args[1] = &pKI;
        args[2] = &pVR; args[3] = &pVI;
        args[4] = &pKId; args[5] = &pVId;
        args[6] = &pMR; args[7] = &pMI;
        args[8] = &N; args[9] = &D;
        args[10] = &nKeys; args[11] = &nVals;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    // Round D up to the next power of two, capped at 1024 (max CUDA
    // block size). Used by the phase-coherence reduce to size the
    // block-level tree sum.
    private static int NextPowerOfTwoForReduction(int D)
    {
        int bs = 1;
        while (bs < D && bs < 1024) bs <<= 1;
        return Math.Max(32, bs);
    }

    /// <inheritdoc/>
    public void SpectralFilter(IGpuBuffer inputReal, IGpuBuffer filterReal, IGpuBuffer filterImag,
        IGpuBuffer outputReal, int batch, int height, int width, int filterSliceCount)
    {
        if (!IsAvailable) throw new InvalidOperationException("CUDA backend not available");
        if (batch <= 0 || height <= 0 || width <= 0) return;
        if (filterSliceCount <= 0 || (filterSliceCount != 1 && filterSliceCount != batch))
            throw new ArgumentException($"filterSliceCount must be 1 (shared) or batch ({batch}). Got {filterSliceCount}.");

        using var _ = PushContext();
        int sliceSize = height * width;
        int totalSize = batch * sliceSize;

        // All temp buffers on GPU — zero CPU round-trips for intermediates
        IGpuBuffer? fftR = null, fftI = null, mulR = null, mulI = null, ifftI = null, zeroI = null;
        try
        {
            fftR = AllocateBuffer(totalSize);
            fftI = AllocateBuffer(totalSize);
            mulR = AllocateBuffer(totalSize);
            mulI = AllocateBuffer(totalSize);
            ifftI = AllocateBuffer(totalSize);
            zeroI = AllocateBuffer(totalSize);
        Fill(zeroI, 0f, totalSize);

            BatchedFFT2D(inputReal, zeroI, fftR, fftI, batch, height, width, inverse: false);

            if (filterSliceCount == batch)
            {
                SplitComplexMultiply(fftR, fftI, filterReal, filterImag, mulR, mulI, totalSize);
            }
            else
            {
                var bcastFR = AllocateBuffer(totalSize);
                var bcastFI = AllocateBuffer(totalSize);
                try
                {
                    for (int b = 0; b < batch; b++)
                    {
                        Copy(filterReal, 0, bcastFR, b * sliceSize, sliceSize);
                        Copy(filterImag, 0, bcastFI, b * sliceSize, sliceSize);
                    }
                    SplitComplexMultiply(fftR, fftI, bcastFR, bcastFI, mulR, mulI, totalSize);
                }
                finally { bcastFR.Dispose(); bcastFI.Dispose(); }
            }

            BatchedFFT2D(mulR, mulI, outputReal, ifftI, batch, height, width, inverse: true);
        }
        finally
        {
            fftR?.Dispose(); fftI?.Dispose();
            mulR?.Dispose(); mulI?.Dispose();
            ifftI?.Dispose(); zeroI?.Dispose();
        }
    }

    /// <inheritdoc/>
    public unsafe void Atan2Elementwise(IGpuBuffer real, IGpuBuffer imag, IGpuBuffer output, int n)
    {
        if (n <= 0) return;
        if (!_kernelCache.TryGetValue("atan2_elementwise", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: atan2_elementwise");
        using var _ = PushContext();
        IntPtr ip = imag.Handle, rp = real.Handle, op = output.Handle;
        void** args = stackalloc void*[4];
        args[0] = &ip; args[1] = &rp; args[2] = &op; args[3] = &n;
        uint grid = (uint)((n + DefaultBlockSize - 1) / DefaultBlockSize);
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    /// <inheritdoc/>
    public unsafe void NormalizeRowsFused(IGpuBuffer input, IGpuBuffer output, int rows, int cols)
    {
        if (rows <= 0 || cols <= 0) return;
        if (!_kernelCache.TryGetValue("normalize_rows_fused", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: normalize_rows_fused");
        using var _ = PushContext();
        IntPtr ip = input.Handle, op = output.Handle;
        void** args = stackalloc void*[4];
        args[0] = &ip; args[1] = &op; args[2] = &rows; args[3] = &cols;
        uint grid = (uint)rows;
        // Tree reduction requires a power-of-two block size.
        uint block = 32;
        uint cap = (uint)Math.Min(256, cols);
        while (block * 2 <= cap) block *= 2;
        uint sharedMem = block * sizeof(float);
        CudaNativeBindings.cuLaunchKernel(kernel, grid, 1, 1, block, 1, 1,
            sharedMem, IntPtr.Zero, (IntPtr)args, IntPtr.Zero);
    }

    /// <inheritdoc/>
    public unsafe void AnalyticSignalMask(IGpuBuffer specReal, IGpuBuffer specImag,
        IGpuBuffer outReal, IGpuBuffer outImag, int batch, int fftSize, int binLow, int binHigh)
    {
        if (batch <= 0 || fftSize <= 0) return;
        long totalL = (long)batch * fftSize;
        if (totalL <= 0 || totalL > int.MaxValue) return;
        int total = (int)totalL;
        if (!_kernelCache.TryGetValue("analytic_signal_mask", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: analytic_signal_mask");
        using var _ = PushContext();
        IntPtr srP = specReal.Handle, siP = specImag.Handle, orP = outReal.Handle, oiP = outImag.Handle;
        void** args = stackalloc void*[8];
        args[0] = &srP; args[1] = &siP; args[2] = &orP; args[3] = &oiP;
        args[4] = &batch; args[5] = &fftSize; args[6] = &binLow; args[7] = &binHigh;
        uint grid = (uint)((total + DefaultBlockSize - 1) / DefaultBlockSize);
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    /// <inheritdoc/>
    public unsafe void BispectrumGather(IGpuBuffer specReal, IGpuBuffer specImag,
        IGpuBuffer outReal, IGpuBuffer outImag, int maxF1, int maxF2)
    {
        if (maxF1 <= 0 || maxF2 <= 0) return;
        long totalL = (long)maxF1 * maxF2;
        if (totalL <= 0 || totalL > int.MaxValue) return;
        int total = (int)totalL;
        if (!_kernelCache.TryGetValue("bispectrum_gather", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: bispectrum_gather");
        using var _ = PushContext();
        IntPtr srP = specReal.Handle, siP = specImag.Handle, orP = outReal.Handle, oiP = outImag.Handle;
        void** args = stackalloc void*[6];
        args[0] = &srP; args[1] = &siP; args[2] = &orP; args[3] = &oiP;
        args[4] = &maxF1; args[5] = &maxF2;
        uint grid = (uint)((total + DefaultBlockSize - 1) / DefaultBlockSize);
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    /// <inheritdoc/>
    public unsafe void TrispectrumGather(IGpuBuffer specReal, IGpuBuffer specImag,
        IGpuBuffer outReal, IGpuBuffer outImag, int maxF1, int maxF2, int maxF3)
    {
        if (maxF1 <= 0 || maxF2 <= 0 || maxF3 <= 0) return;
        long totalL = (long)maxF1 * maxF2 * maxF3;
        if (totalL <= 0 || totalL > int.MaxValue) return;
        int total = (int)totalL;
        if (!_kernelCache.TryGetValue("trispectrum_gather", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: trispectrum_gather");
        using var _ = PushContext();
        IntPtr srP = specReal.Handle, siP = specImag.Handle, orP = outReal.Handle, oiP = outImag.Handle;
        void** args = stackalloc void*[7];
        args[0] = &srP; args[1] = &siP; args[2] = &orP; args[3] = &oiP;
        args[4] = &maxF1; args[5] = &maxF2; args[6] = &maxF3;
        uint grid = (uint)((total + DefaultBlockSize - 1) / DefaultBlockSize);
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    /// <inheritdoc/>
    public unsafe void CavityBounceInplace(IGpuBuffer workReal, IGpuBuffer workImag, int total, float invN)
    {
        if (total <= 0) return;
        if (!_kernelCache.TryGetValue("cavity_bounce_inplace", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: cavity_bounce_inplace");
        using var _ = PushContext();
        IntPtr wr = workReal.Handle, wi = workImag.Handle;
        void** args = stackalloc void*[4];
        args[0] = &wr; args[1] = &wi; args[2] = &total; args[3] = &invN;
        uint grid = (uint)((total + DefaultBlockSize - 1) / DefaultBlockSize);
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    /// <inheritdoc/>
    public unsafe void WidebandLogBinPool(IGpuBuffer magBuf, IGpuBuffer output,
        int totalSegBatch, int fftSize, int numBins, int usable)
    {
        if (totalSegBatch <= 0 || fftSize <= 0 || numBins <= 0 || usable <= 0) return;
        long totalL = (long)totalSegBatch * numBins;
        if (totalL <= 0 || totalL > int.MaxValue) return;
        int total = (int)totalL;
        if (!_kernelCache.TryGetValue("wideband_log_bin_pool", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: wideband_log_bin_pool");
        using var _ = PushContext();
        IntPtr mp = magBuf.Handle, op = output.Handle;
        void** args = stackalloc void*[6];
        args[0] = &mp; args[1] = &op;
        args[2] = &totalSegBatch; args[3] = &fftSize; args[4] = &numBins; args[5] = &usable;
        uint grid = (uint)((total + DefaultBlockSize - 1) / DefaultBlockSize);
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    /// <inheritdoc/>
    public unsafe void MelFilterbankApply(IGpuBuffer powerSpec, IGpuBuffer melFilters, IGpuBuffer melEnergy,
        int totalSegBatch, int specBins, int melBins)
    {
        if (totalSegBatch <= 0 || specBins <= 0 || melBins <= 0) return;
        long totalL = (long)totalSegBatch * melBins;
        if (totalL <= 0 || totalL > int.MaxValue) return;
        int total = (int)totalL;
        if (!_kernelCache.TryGetValue("mel_filterbank_apply", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: mel_filterbank_apply");
        using var _ = PushContext();
        IntPtr ps = powerSpec.Handle, mf = melFilters.Handle, me = melEnergy.Handle;
        void** args = stackalloc void*[6];
        args[0] = &ps; args[1] = &mf; args[2] = &me;
        args[3] = &totalSegBatch; args[4] = &specBins; args[5] = &melBins;
        uint grid = (uint)((total + DefaultBlockSize - 1) / DefaultBlockSize);
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    /// <inheritdoc/>
    public unsafe void MfccLog1p(IGpuBuffer input, IGpuBuffer output, int n)
    {
        if (n <= 0) return;
        if (!_kernelCache.TryGetValue("mfcc_log1p", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: mfcc_log1p");
        using var _ = PushContext();
        IntPtr ip = input.Handle, op = output.Handle;
        void** args = stackalloc void*[3];
        args[0] = &ip; args[1] = &op; args[2] = &n;
        uint grid = (uint)((n + DefaultBlockSize - 1) / DefaultBlockSize);
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    /// <inheritdoc/>
    public unsafe void PacPhaseBinMi(IGpuBuffer thetaPhase, IGpuBuffer gammaAmp, IGpuBuffer output,
        int batch, int numSamples, int numGammaBands, int gammaIdx)
    {
        if (batch <= 0) return;
        if (numSamples <= 0)
            throw new ArgumentOutOfRangeException(nameof(numSamples), "numSamples must be positive.");
        if (numGammaBands <= 0)
            throw new ArgumentOutOfRangeException(nameof(numGammaBands), "numGammaBands must be positive.");
        if (gammaIdx < 0 || gammaIdx >= numGammaBands)
            throw new ArgumentOutOfRangeException(nameof(gammaIdx), $"gammaIdx must be in [0, {numGammaBands}).");
        using var _ = PushContext();
        IntPtr tp = thetaPhase.Handle, ga = gammaAmp.Handle, op = output.Handle;
        void** args = stackalloc void*[7];
        args[0] = &tp; args[1] = &ga; args[2] = &op;
        args[3] = &batch; args[4] = &numSamples; args[5] = &numGammaBands; args[6] = &gammaIdx;
        uint grid = (uint)batch;

        if (GpuDeterminism.IsActive)
        {
            // Issue #382: block size = NUM_PHASE_BINS = 18 (one thread per bin),
            // no shared memory needed since the deterministic variant uses statically
            // sized __shared__ inside the kernel.
            if (!_kernelCache.TryGetValue("pac_phase_bin_mi_deterministic", out var kernelD))
                throw new InvalidOperationException("CUDA kernel not found: pac_phase_bin_mi_deterministic");
            CuBlasNative.CheckCudaResult(
                CudaNativeBindings.cuLaunchKernel(kernelD, grid, 1, 1, 18, 1, 1,
                    0, _stream, (IntPtr)args, IntPtr.Zero),
                "cuLaunchKernel(pac_phase_bin_mi_deterministic)");
            return;
        }

        if (!_kernelCache.TryGetValue("pac_phase_bin_mi", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: pac_phase_bin_mi");
        uint block = 256;
        uint sharedMem = (uint)(2 * 18 * sizeof(float));
        CuBlasNative.CheckCudaResult(
            CudaNativeBindings.cuLaunchKernel(kernel, grid, 1, 1, block, 1, 1,
                sharedMem, _stream, (IntPtr)args, IntPtr.Zero),
            "cuLaunchKernel(pac_phase_bin_mi)");
    }

    #endregion

    #region Quantum Computing Operations

    public unsafe void QuantumMeasurement(IGpuBuffer realPart, IGpuBuffer imagPart, IGpuBuffer probabilities, int batchSize, int stateSize)
    {
        if (!_kernelCache.TryGetValue("quantum_measurement", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: quantum_measurement");

        using var _ = PushContext();
        int totalThreads = batchSize * stateSize;
        uint grid = (uint)((totalThreads + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr realPtr = realPart.Handle;
        IntPtr imagPtr = imagPart.Handle;
        IntPtr probPtr = probabilities.Handle;

        void** args = stackalloc void*[5];
        args[0] = &realPtr;
        args[1] = &imagPtr;
        args[2] = &probPtr;
        args[3] = &batchSize;
        args[4] = &stateSize;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    public unsafe void NormalizeProbabilities(IGpuBuffer probabilities, int batchSize, int stateSize)
    {
        if (!_kernelCache.TryGetValue("normalize_probabilities", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: normalize_probabilities");

        // Validate state size is power of two for shared-memory reduction
        ValidateQuantumStateSize(stateSize, nameof(stateSize));

        using var _ = PushContext();
        // One block per batch element
        uint grid = (uint)batchSize;
        uint blockDim = (uint)Math.Min(DefaultBlockSize, stateSize);
        uint sharedMemBytes = blockDim * sizeof(float);
        IntPtr probPtr = probabilities.Handle;

        void** args = stackalloc void*[3];
        args[0] = &probPtr;
        args[1] = &batchSize;
        args[2] = &stateSize;
        LaunchKernelWithSharedMem(kernel, grid, blockDim, sharedMemBytes, args);
    }

    public unsafe void ComplexMatVec(IGpuBuffer matReal, IGpuBuffer matImag, IGpuBuffer vecReal, IGpuBuffer vecImag,
        IGpuBuffer outReal, IGpuBuffer outImag, int batchSize, int dim)
    {
        if (!_kernelCache.TryGetValue("complex_matvec", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: complex_matvec");

        using var _ = PushContext();
        int totalThreads = batchSize * dim;
        uint grid = (uint)((totalThreads + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr matRealPtr = matReal.Handle;
        IntPtr matImagPtr = matImag.Handle;
        IntPtr vecRealPtr = vecReal.Handle;
        IntPtr vecImagPtr = vecImag.Handle;
        IntPtr outRealPtr = outReal.Handle;
        IntPtr outImagPtr = outImag.Handle;

        void** args = stackalloc void*[8];
        args[0] = &matRealPtr;
        args[1] = &matImagPtr;
        args[2] = &vecRealPtr;
        args[3] = &vecImagPtr;
        args[4] = &outRealPtr;
        args[5] = &outImagPtr;
        args[6] = &batchSize;
        args[7] = &dim;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    public unsafe void QuantumRotation(IGpuBuffer stateReal, IGpuBuffer stateImag, IGpuBuffer outReal, IGpuBuffer outImag,
        IGpuBuffer angles, int numQubits, int batchSize)
    {
        if (!_kernelCache.TryGetValue("quantum_rotation", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: quantum_rotation");

        // Validate numQubits is in a reasonable range (1-30 to avoid overflow)
        if (numQubits <= 0 || numQubits > 30)
            throw new ArgumentOutOfRangeException(nameof(numQubits), numQubits,
                "Number of qubits must be between 1 and 30 to avoid integer overflow.");

        using var _ = PushContext();
        int dim = 1 << numQubits;
        // dim is guaranteed to be power of two since it's 2^numQubits
        uint blockDim = (uint)Math.Min(DefaultBlockSize, dim);
        IntPtr stateRealPtr = stateReal.Handle;
        IntPtr stateImagPtr = stateImag.Handle;
        IntPtr outRealPtr = outReal.Handle;
        IntPtr outImagPtr = outImag.Handle;
        IntPtr anglesPtr = angles.Handle;

        void** args = stackalloc void*[7];
        args[0] = &stateRealPtr;
        args[1] = &stateImagPtr;
        args[2] = &outRealPtr;
        args[3] = &outImagPtr;
        args[4] = &anglesPtr;
        args[5] = &numQubits;
        args[6] = &batchSize;
        LaunchKernel(kernel, (uint)batchSize, blockDim, args);
    }

    public unsafe void MeasurementForward(IGpuBuffer input, IGpuBuffer output, int batchSize, int stateSize)
    {
        if (!_kernelCache.TryGetValue("measurement_forward", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: measurement_forward");

        // Validate state size is power of two for shared-memory reduction
        ValidateQuantumStateSize(stateSize, nameof(stateSize));

        using var _ = PushContext();
        // One block per batch element, uses shared memory for reduction
        uint grid = (uint)batchSize;
        uint blockDim = (uint)Math.Min(DefaultBlockSize, stateSize);
        uint sharedMemBytes = blockDim * sizeof(float);
        IntPtr inputPtr = input.Handle;
        IntPtr outputPtr = output.Handle;

        void** args = stackalloc void*[4];
        args[0] = &inputPtr;
        args[1] = &outputPtr;
        args[2] = &batchSize;
        args[3] = &stateSize;
        LaunchKernelWithSharedMem(kernel, grid, blockDim, sharedMemBytes, args);
    }

    #endregion

    #region RNN (LSTM/GRU) Sequence Operations

    // ── Fused GLA (Gated Linear Attention) scan (#1464) ────────────────────────────────────
    public unsafe void GlaScanForward(
        IGpuBuffer q, IGpuBuffer k, IGpuBuffer v, IGpuBuffer gate, IGpuBuffer output,
        int batch, int seqLen, int modelDim, int numHeads, int headDim)
    {
        if (batch <= 0 || seqLen <= 0 || modelDim <= 0 || numHeads <= 0 || headDim <= 0)
            throw new ArgumentOutOfRangeException(nameof(batch), "GLA dimensions must be positive.");
        if (modelDim != numHeads * headDim)
            throw new ArgumentException($"modelDim ({modelDim}) must equal numHeads * headDim ({numHeads * headDim}).");
        if (headDim > Kernels.CudaGlaKernels.MaxHeadDim)
            throw new InvalidOperationException(
                $"GLA headDim ({headDim}) exceeds max ({Kernels.CudaGlaKernels.MaxHeadDim}).");

        long tokenCount = checked((long)batch * seqLen);
        long qkvElems = checked(tokenCount * modelDim);
        long gateElems = checked(tokenCount * numHeads);
        if (q.Size < qkvElems || k.Size < qkvElems || v.Size < qkvElems || output.Size < qkvElems)
            throw new ArgumentException("Q/K/V/output buffer is too small for the requested GLA shape.");
        if (gate.Size < gateElems)
            throw new ArgumentException("gate buffer is too small for the requested GLA shape.");

        if (!_kernelCache.TryGetValue("gla_scan_forward", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: gla_scan_forward");

        using var _ = PushContext();
        IntPtr pq = q.Handle, pk = k.Handle, pv = v.Handle, pg = gate.Handle, po = output.Handle;
        void** args = stackalloc void*[10];
        args[0] = &pq; args[1] = &pk; args[2] = &pv; args[3] = &pg; args[4] = &po;
        args[5] = &batch; args[6] = &seqLen; args[7] = &modelDim; args[8] = &numHeads; args[9] = &headDim;
        uint total = (uint)(batch * numHeads * headDim);
        LaunchKernel(kernel, (total + (uint)DefaultBlockSize - 1) / (uint)DefaultBlockSize, (uint)DefaultBlockSize, args);
    }

    // dQ/dK/dV ([batch,seqLen,modelDim]) and dG ([batch,seqLen,numHeads]); dQ/dK/dG are the
    // atomic accumulators and are zeroed internally below (no caller pre-zeroing required).
    public unsafe void GlaScanBackward(
        IGpuBuffer dOut, IGpuBuffer q, IGpuBuffer k, IGpuBuffer v, IGpuBuffer gate,
        IGpuBuffer dQ, IGpuBuffer dK, IGpuBuffer dV, IGpuBuffer dG,
        int batch, int seqLen, int modelDim, int numHeads, int headDim)
    {
        if (batch <= 0 || seqLen <= 0 || modelDim <= 0 || numHeads <= 0 || headDim <= 0)
            throw new ArgumentOutOfRangeException(nameof(batch), "GLA dimensions must be positive.");
        if (modelDim != numHeads * headDim)
            throw new ArgumentException($"modelDim ({modelDim}) must equal numHeads * headDim ({numHeads * headDim}).");
        if (headDim > Kernels.CudaGlaKernels.MaxHeadDim)
            throw new InvalidOperationException(
                $"GLA headDim ({headDim}) exceeds max ({Kernels.CudaGlaKernels.MaxHeadDim}).");
        if (!_kernelCache.TryGetValue("gla_scan_recompute", out var recompute) ||
            !_kernelCache.TryGetValue("gla_scan_backward", out var backward))
            throw new InvalidOperationException("CUDA kernel not found: gla_scan_recompute / gla_scan_backward");

        using var _ = PushContext();
        // dQ/dK/dG are atomic accumulators in the backward kernel — zero them here so the
        // method is self-contained and correct even if the caller reuses dirty buffers.
        Fill(dQ, 0f, batch * seqLen * modelDim);
        Fill(dK, 0f, batch * seqLen * modelDim);
        Fill(dG, 0f, batch * seqLen * numHeads);
        long trajLenL = checked((long)batch * numHeads * seqLen * headDim * headDim);
        if (trajLenL > int.MaxValue)
            throw new ArgumentException("GLA trajectory buffer is too large for a single allocation.");
        int trajLen = (int)trajLenL;
        using var trajBuf = AllocateBuffer(trajLen);
        uint total = (uint)(batch * numHeads * headDim);
        uint grid = (total + (uint)DefaultBlockSize - 1) / (uint)DefaultBlockSize;

        IntPtr pk = k.Handle, pv = v.Handle, pg = gate.Handle, ptr = trajBuf.Handle;
        void** ra = stackalloc void*[9];
        ra[0] = &pk; ra[1] = &pv; ra[2] = &pg; ra[3] = &ptr;
        ra[4] = &batch; ra[5] = &seqLen; ra[6] = &modelDim; ra[7] = &numHeads; ra[8] = &headDim;
        LaunchKernel(recompute, grid, (uint)DefaultBlockSize, ra);

        IntPtr pdo = dOut.Handle, pq = q.Handle;
        IntPtr pdq = dQ.Handle, pdk = dK.Handle, pdv = dV.Handle, pdg = dG.Handle;
        void** ba = stackalloc void*[15];
        ba[0] = &pdo; ba[1] = &pq; ba[2] = &pk; ba[3] = &pv; ba[4] = &pg; ba[5] = &ptr;
        ba[6] = &pdq; ba[7] = &pdk; ba[8] = &pdv; ba[9] = &pdg;
        ba[10] = &batch; ba[11] = &seqLen; ba[12] = &modelDim; ba[13] = &numHeads; ba[14] = &headDim;
        LaunchKernel(backward, grid, (uint)DefaultBlockSize, ba);
    }

    // ── Fused xLSTM (mLSTM) scan forward (#1464) ───────────────────────────────────────────
    public unsafe void XLstmScanForward(
        IGpuBuffer q, IGpuBuffer k, IGpuBuffer v,
        IGpuBuffer iGate, IGpuBuffer fGate, IGpuBuffer oGate, IGpuBuffer output,
        int batch, int seqLen, int modelDim, int numHeads, int headDim)
    {
        if (batch <= 0 || seqLen <= 0 || modelDim <= 0 || numHeads <= 0 || headDim <= 0)
            throw new ArgumentOutOfRangeException(nameof(batch), "xLSTM dimensions must be positive.");
        if (modelDim != numHeads * headDim)
            throw new ArgumentException($"modelDim ({modelDim}) must equal numHeads * headDim ({numHeads * headDim}).");
        if (headDim > Kernels.CudaXLstmKernels.MaxHeadDim)
            throw new InvalidOperationException(
                $"xLSTM headDim ({headDim}) exceeds max ({Kernels.CudaXLstmKernels.MaxHeadDim}).");
        if (!_kernelCache.TryGetValue("xlstm_scan_forward", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: xlstm_scan_forward");

        using var _ = PushContext();
        IntPtr pq = q.Handle, pk = k.Handle, pv = v.Handle;
        IntPtr pi = iGate.Handle, pf = fGate.Handle, po = oGate.Handle, pout = output.Handle;
        void** args = stackalloc void*[12];
        args[0] = &pq; args[1] = &pk; args[2] = &pv; args[3] = &pi; args[4] = &pf; args[5] = &po; args[6] = &pout;
        args[7] = &batch; args[8] = &seqLen; args[9] = &modelDim; args[10] = &numHeads; args[11] = &headDim;
        uint total = (uint)(batch * numHeads);
        LaunchKernel(kernel, (total + (uint)DefaultBlockSize - 1) / (uint)DefaultBlockSize, (uint)DefaultBlockSize, args);
    }

    // ── Fused Gated DeltaNet scan forward (#1464) ──────────────────────────────────────────
    public unsafe void GatedDeltaNetScanForward(
        IGpuBuffer q, IGpuBuffer k, IGpuBuffer v, IGpuBuffer alpha, IGpuBuffer beta, IGpuBuffer output,
        int batch, int seqLen, int modelDim, int numHeads, int headDim)
    {
        if (batch <= 0 || seqLen <= 0 || modelDim <= 0 || numHeads <= 0 || headDim <= 0)
            throw new ArgumentOutOfRangeException(nameof(batch), "GatedDeltaNet dimensions must be positive.");
        if (modelDim != numHeads * headDim)
            throw new ArgumentException($"modelDim ({modelDim}) must equal numHeads * headDim ({numHeads * headDim}).");
        if (headDim > Kernels.CudaGatedDeltaNetKernels.MaxHeadDim)
            throw new InvalidOperationException(
                $"GatedDeltaNet headDim ({headDim}) exceeds max ({Kernels.CudaGatedDeltaNetKernels.MaxHeadDim}).");
        if (!_kernelCache.TryGetValue("gated_delta_scan_forward", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: gated_delta_scan_forward");

        using var _ = PushContext();
        IntPtr pq = q.Handle, pk = k.Handle, pv = v.Handle, pa = alpha.Handle, pb = beta.Handle, pout = output.Handle;
        void** args = stackalloc void*[11];
        args[0] = &pq; args[1] = &pk; args[2] = &pv; args[3] = &pa; args[4] = &pb; args[5] = &pout;
        args[6] = &batch; args[7] = &seqLen; args[8] = &modelDim; args[9] = &numHeads; args[10] = &headDim;
        uint total = (uint)(batch * numHeads * headDim);
        LaunchKernel(kernel, (total + (uint)DefaultBlockSize - 1) / (uint)DefaultBlockSize, (uint)DefaultBlockSize, args);
    }

    // ── Fused RG-LRU scan forward (#1464) ──────────────────────────────────────────────────
    public unsafe void RgLruScanForward(
        IGpuBuffer value, IGpuBuffer recGate, IGpuBuffer inpGate, IGpuBuffer decay, IGpuBuffer output,
        int batch, int seqLen, int recDim)
    {
        if (batch <= 0 || seqLen <= 0 || recDim <= 0)
            throw new ArgumentOutOfRangeException(nameof(batch), "RG-LRU dimensions must be positive.");
        if (!_kernelCache.TryGetValue("rglru_scan_forward", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: rglru_scan_forward");

        using var _ = PushContext();
        IntPtr pv = value.Handle, pr = recGate.Handle, pi = inpGate.Handle, pd = decay.Handle, pout = output.Handle;
        void** args = stackalloc void*[8];
        args[0] = &pv; args[1] = &pr; args[2] = &pi; args[3] = &pd; args[4] = &pout;
        args[5] = &batch; args[6] = &seqLen; args[7] = &recDim;
        uint total = (uint)(batch * recDim);
        LaunchKernel(kernel, (total + (uint)DefaultBlockSize - 1) / (uint)DefaultBlockSize, (uint)DefaultBlockSize, args);
    }

    // ── Fused RWKV-4 WKV scan forward (#1464) ──────────────────────────────────────────────
    public unsafe void Rwkv4WkvForward(
        IGpuBuffer r, IGpuBuffer k, IGpuBuffer v, IGpuBuffer timeDecay, IGpuBuffer timeFirst, IGpuBuffer output,
        int batch, int seqLen, int modelDim)
    {
        if (batch <= 0 || seqLen <= 0 || modelDim <= 0)
            throw new ArgumentOutOfRangeException(nameof(batch), "RWKV-4 dimensions must be positive.");
        if (!_kernelCache.TryGetValue("rwkv4_wkv_forward", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: rwkv4_wkv_forward");

        using var _ = PushContext();
        IntPtr pr = r.Handle, pk = k.Handle, pv = v.Handle, pd = timeDecay.Handle, pf = timeFirst.Handle, pout = output.Handle;
        void** args = stackalloc void*[9];
        args[0] = &pr; args[1] = &pk; args[2] = &pv; args[3] = &pd; args[4] = &pf; args[5] = &pout;
        args[6] = &batch; args[7] = &seqLen; args[8] = &modelDim;
        uint total = (uint)(batch * modelDim);
        LaunchKernel(kernel, (total + (uint)DefaultBlockSize - 1) / (uint)DefaultBlockSize, (uint)DefaultBlockSize, args);
    }

    // ── Fused Mamba selective scan forward (#1464) ─────────────────────────────────────────
    public unsafe void MambaSelectiveScanForward(
        IGpuBuffer x, IGpuBuffer delta, IGpuBuffer aLog, IGpuBuffer bParam, IGpuBuffer cParam, IGpuBuffer dParam,
        IGpuBuffer output, int batch, int seqLen, int innerDim, int stateDim)
    {
        if (batch <= 0 || seqLen <= 0 || innerDim <= 0 || stateDim <= 0)
            throw new ArgumentOutOfRangeException(nameof(batch), "Mamba dimensions must be positive.");
        if (stateDim > Kernels.CudaMambaKernels.MaxStateDim)
            throw new InvalidOperationException(
                $"Mamba stateDim ({stateDim}) exceeds max ({Kernels.CudaMambaKernels.MaxStateDim}).");
        if (!_kernelCache.TryGetValue("mamba_selective_scan_forward", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: mamba_selective_scan_forward");

        using var _ = PushContext();
        IntPtr px = x.Handle, pdt = delta.Handle, pa = aLog.Handle, pb = bParam.Handle, pc = cParam.Handle, pd = dParam.Handle, pout = output.Handle;
        void** args = stackalloc void*[11];
        args[0] = &px; args[1] = &pdt; args[2] = &pa; args[3] = &pb; args[4] = &pc; args[5] = &pd; args[6] = &pout;
        args[7] = &batch; args[8] = &seqLen; args[9] = &innerDim; args[10] = &stateDim;
        uint total = (uint)(batch * innerDim);
        LaunchKernel(kernel, (total + (uint)DefaultBlockSize - 1) / (uint)DefaultBlockSize, (uint)DefaultBlockSize, args);
    }

    // ── Fused Mamba-2 SSD scan forward (#1464) ─────────────────────────────────────────────
    public unsafe void Mamba2SsdScanForward(
        IGpuBuffer x, IGpuBuffer delta, IGpuBuffer aLog, IGpuBuffer bParam, IGpuBuffer cParam, IGpuBuffer dParam,
        IGpuBuffer output, int batch, int seqLen, int innerDim, int numHeads, int headDim, int stateDim)
    {
        if (batch <= 0 || seqLen <= 0 || innerDim <= 0 || numHeads <= 0 || headDim <= 0 || stateDim <= 0)
            throw new ArgumentOutOfRangeException(nameof(batch), "Mamba-2 dimensions must be positive.");
        if (innerDim != numHeads * headDim)
            throw new ArgumentException($"innerDim ({innerDim}) must equal numHeads * headDim ({numHeads * headDim}).");
        if (stateDim > Kernels.CudaMamba2Kernels.MaxStateDim)
            throw new InvalidOperationException(
                $"Mamba-2 stateDim ({stateDim}) exceeds max ({Kernels.CudaMamba2Kernels.MaxStateDim}).");
        if (!_kernelCache.TryGetValue("mamba2_ssd_scan_forward", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: mamba2_ssd_scan_forward");

        using var _ = PushContext();
        IntPtr px = x.Handle, pdt = delta.Handle, pa = aLog.Handle, pb = bParam.Handle, pc = cParam.Handle, pd = dParam.Handle, pout = output.Handle;
        void** args = stackalloc void*[13];
        args[0] = &px; args[1] = &pdt; args[2] = &pa; args[3] = &pb; args[4] = &pc; args[5] = &pd; args[6] = &pout;
        args[7] = &batch; args[8] = &seqLen; args[9] = &innerDim; args[10] = &numHeads; args[11] = &headDim; args[12] = &stateDim;
        uint total = (uint)(batch * innerDim);
        LaunchKernel(kernel, (total + (uint)DefaultBlockSize - 1) / (uint)DefaultBlockSize, (uint)DefaultBlockSize, args);
    }

    // ── Fused linear (LM head) + cross-entropy (#1464) ─────────────────────────────────────
    public unsafe float FusedLinearCrossEntropyIndex(
        IGpuBuffer hidden, IGpuBuffer weight, IGpuBuffer bias, IGpuBuffer targetIds, int n, int d, int vocab)
        => FusedCeLaunch("fused_linear_ce_index", hidden, weight, bias, targetIds, n, d, vocab);

    public unsafe float FusedLinearCrossEntropyDense(
        IGpuBuffer hidden, IGpuBuffer weight, IGpuBuffer bias, IGpuBuffer target, int n, int d, int vocab)
        => FusedCeLaunch("fused_linear_ce_dense", hidden, weight, bias, target, n, d, vocab);

    private unsafe float FusedCeLaunch(
        string kernelName, IGpuBuffer hidden, IGpuBuffer weight, IGpuBuffer bias, IGpuBuffer tgt, int n, int d, int vocab)
    {
        if (n <= 0 || d <= 0 || vocab <= 0)
            throw new ArgumentOutOfRangeException(nameof(n), "Fused CE dimensions (n, d, vocab) must be positive.");
        if (!_kernelCache.TryGetValue(kernelName, out var kernel))
            throw new InvalidOperationException($"CUDA kernel not found: {kernelName}");
        using var _ = PushContext();
        using var lossBuf = AllocateBuffer(1); // zeroed
        IntPtr ph = hidden.Handle, pw = weight.Handle, pb = bias.Handle, pt = tgt.Handle, pl = lossBuf.Handle;
        void** args = stackalloc void*[8];
        args[0] = &ph; args[1] = &pw; args[2] = &pb; args[3] = &pt; args[4] = &pl;
        args[5] = &n; args[6] = &d; args[7] = &vocab;
        uint total = (uint)n;
        LaunchKernel(kernel, (total + (uint)DefaultBlockSize - 1) / (uint)DefaultBlockSize, (uint)DefaultBlockSize, args);
        var loss = DownloadBuffer(lossBuf);
        return loss[0] / n;
    }

    public unsafe void LstmForwardSequence(
        IGpuBuffer input, IGpuBuffer hInit, IGpuBuffer cInit,
        IGpuBuffer weightsIh, IGpuBuffer weightsHh, IGpuBuffer biasIh, IGpuBuffer biasHh,
        IGpuBuffer output, IGpuBuffer hFinal, IGpuBuffer cFinal,
        IGpuBuffer allH, IGpuBuffer allC, IGpuBuffer cacheGates,
        int seqLen, int batch, int inputSize, int hiddenSize)
    {
        if (!_kernelCache.TryGetValue("lstm_forward_sequence", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: lstm_forward_sequence");

        // Validate hiddenSize fits in a block for correct synchronization
        if (hiddenSize > MaxRnnBlockSize)
        {
            throw new InvalidOperationException(
                $"LSTM forward sequence hiddenSize ({hiddenSize}) exceeds max block size ({MaxRnnBlockSize}). " +
                "The kernel requires all hidden units of a batch to fit within a single block for " +
                "correct synchronization. Use smaller hiddenSize or use cell-level LSTM operations.");
        }

        using var _ = PushContext();

        // Grid = one block per batch sample, block = hiddenSize threads per batch
        // This ensures __syncthreads() correctly synchronizes all threads for the same batch
        uint grid = (uint)batch;
        uint blockSize = (uint)hiddenSize;

        IntPtr inputPtr = input.Handle;
        IntPtr hInitPtr = hInit.Handle;
        IntPtr cInitPtr = cInit.Handle;
        IntPtr weightsIhPtr = weightsIh.Handle;
        IntPtr weightsHhPtr = weightsHh.Handle;
        IntPtr biasIhPtr = biasIh.Handle;
        IntPtr biasHhPtr = biasHh.Handle;
        IntPtr outputPtr = output.Handle;
        IntPtr allHPtr = allH.Handle;
        IntPtr allCPtr = allC.Handle;
        IntPtr cacheGatesPtr = cacheGates.Handle;

        // Kernel signature: input, h_init, c_init, Wi, Wh, biasIh, biasHh, output, h_states, c_states, gates, batch, timeSteps, inputSize, hiddenSize
        void** args = stackalloc void*[15];
        args[0] = &inputPtr;
        args[1] = &hInitPtr;
        args[2] = &cInitPtr;
        args[3] = &weightsIhPtr;
        args[4] = &weightsHhPtr;
        args[5] = &biasIhPtr;
        args[6] = &biasHhPtr;
        args[7] = &outputPtr;
        args[8] = &allHPtr;       // h_states cache
        args[9] = &allCPtr;       // c_states cache
        args[10] = &cacheGatesPtr; // gates cache
        args[11] = &batch;
        args[12] = &seqLen;       // timeSteps
        args[13] = &inputSize;
        args[14] = &hiddenSize;

        LaunchKernel(kernel, grid, blockSize, args);

        // Copy the last timestep from allH and allC into hFinal and cFinal
        // allH layout: [(seqLen + 1) * batch * hiddenSize] where index 0 is hInit
        // So final hidden state is at index seqLen (last timestep output)
        int finalStateOffset = seqLen * batch * hiddenSize;
        int stateSize = batch * hiddenSize;
        ulong byteSize = (ulong)(stateSize * sizeof(float));

        // Device-to-device copy from allH[seqLen] to hFinal
        IntPtr srcH = IntPtr.Add(allH.Handle, finalStateOffset * sizeof(float));
        var resultH = CudaNativeBindings.cuMemcpyDtoDAsync(
            hFinal.Handle,
            srcH,
            byteSize,
            _stream);
        CuBlasNative.CheckCudaResult(resultH, "cuMemcpyDtoDAsync (hFinal from allH)");

        // Device-to-device copy from allC[seqLen] to cFinal
        IntPtr srcC = IntPtr.Add(allC.Handle, finalStateOffset * sizeof(float));
        var resultC = CudaNativeBindings.cuMemcpyDtoDAsync(
            cFinal.Handle,
            srcC,
            byteSize,
            _stream);
        CuBlasNative.CheckCudaResult(resultC, "cuMemcpyDtoDAsync (cFinal from allC)");
    }

    public unsafe void LstmBackwardSequence(
        IGpuBuffer gradOutput, IGpuBuffer allH, IGpuBuffer allC, IGpuBuffer cacheGates,
        IGpuBuffer hInit, IGpuBuffer cInit,
        IGpuBuffer weightsIh, IGpuBuffer weightsHh, IGpuBuffer input,
        IGpuBuffer gradInput, IGpuBuffer gradHInit, IGpuBuffer gradCInit,
        IGpuBuffer gradWeightsIh, IGpuBuffer gradWeightsHh, IGpuBuffer gradBiasIh, IGpuBuffer gradBiasHh,
        int seqLen, int batch, int inputSize, int hiddenSize)
    {
        if (!_kernelCache.TryGetValue("lstm_backward_sequence", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: lstm_backward_sequence");

        // Validate hiddenSize fits in a block for correct synchronization
        if (hiddenSize > MaxRnnBlockSize)
        {
            throw new InvalidOperationException(
                $"LSTM backward sequence hiddenSize ({hiddenSize}) exceeds max block size ({MaxRnnBlockSize}). " +
                "The kernel requires all hidden units of a batch to fit within a single block for " +
                "correct synchronization. Use smaller hiddenSize or use cell-level LSTM operations.");
        }

        using var _ = PushContext();

        // Grid = one block per batch sample, block = hiddenSize threads per batch
        // This ensures __syncthreads() correctly synchronizes all threads for the same batch
        uint grid = (uint)batch;

        IntPtr gradOutputPtr = gradOutput.Handle;
        IntPtr allHPtr = allH.Handle;
        IntPtr allCPtr = allC.Handle;
        IntPtr cacheGatesPtr = cacheGates.Handle;
        IntPtr cInitPtr = cInit.Handle;
        IntPtr hInitPtr = hInit.Handle;
        IntPtr inputPtr = input.Handle;
        IntPtr weightsIhPtr = weightsIh.Handle;
        IntPtr weightsHhPtr = weightsHh.Handle;
        IntPtr gradInputPtr = gradInput.Handle;
        IntPtr gradWeightsIhPtr = gradWeightsIh.Handle;
        IntPtr gradWeightsHhPtr = gradWeightsHh.Handle;
        IntPtr gradBiasIhPtr = gradBiasIh.Handle;
        IntPtr gradBiasHhPtr = gradBiasHh.Handle;
        IntPtr gradHInitPtr = gradHInit.Handle;
        IntPtr gradCInitPtr = gradCInit.Handle;

        // Kernel signature: gradOutput, h_states, c_states, gates, c_init, h_init, input, Wi, Wh,
        //                   gradInput, dWi, dWh, dBiasIh, dBiasHh, dH_init, dC_init, batch, timeSteps, inputSize, hiddenSize
        void** args = stackalloc void*[20];
        args[0] = &gradOutputPtr;
        args[1] = &allHPtr;        // h_states
        args[2] = &allCPtr;        // c_states
        args[3] = &cacheGatesPtr;  // gates
        args[4] = &cInitPtr;       // c_init
        args[5] = &hInitPtr;       // h_init
        args[6] = &inputPtr;
        args[7] = &weightsIhPtr;   // Wi
        args[8] = &weightsHhPtr;   // Wh
        args[9] = &gradInputPtr;
        args[10] = &gradWeightsIhPtr;  // dWi
        args[11] = &gradWeightsHhPtr;  // dWh
        args[12] = &gradBiasIhPtr;
        args[13] = &gradBiasHhPtr;
        args[14] = &gradHInitPtr;  // dH_init
        args[15] = &gradCInitPtr;  // dC_init
        args[16] = &batch;
        args[17] = &seqLen;        // timeSteps
        args[18] = &inputSize;
        args[19] = &hiddenSize;

        LaunchKernel(kernel, grid, (uint)hiddenSize, args);
    }

    public unsafe void GruForwardSequence(
        IGpuBuffer input, IGpuBuffer hInit,
        IGpuBuffer weightsIh, IGpuBuffer weightsHh, IGpuBuffer biasIh, IGpuBuffer biasHh,
        IGpuBuffer output, IGpuBuffer hFinal, IGpuBuffer allH, IGpuBuffer cacheGates,
        int seqLen, int batch, int inputSize, int hiddenSize)
    {
        if (!_kernelCache.TryGetValue("gru_forward_sequence", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: gru_forward_sequence");

        // The forward kernel uses __syncthreads() which only synchronizes within a block.
        // If hiddenSize > MaxRnnBlockSize, threads within a batch would span multiple blocks,
        // causing a race condition when reading reset gates computed by other threads.
        // Enforce one-block-per-batch constraint for correctness.
        if (hiddenSize > MaxRnnBlockSize)
        {
            throw new InvalidOperationException(
                $"GRU forward sequence hiddenSize ({hiddenSize}) exceeds max block size ({MaxRnnBlockSize}). " +
                "The kernel requires all hidden units of a batch to fit within a single block for " +
                "correct synchronization. Use smaller hiddenSize or use cell-level GRU operations.");
        }

        using var _ = PushContext();

        // Grid = one block per batch sample, block = hiddenSize threads per batch
        // This ensures __syncthreads() correctly synchronizes all threads for the same batch
        uint grid = (uint)batch;

        IntPtr inputPtr = input.Handle;
        IntPtr hInitPtr = hInit.Handle;
        IntPtr weightsIhPtr = weightsIh.Handle;
        IntPtr weightsHhPtr = weightsHh.Handle;
        IntPtr biasIhPtr = biasIh.Handle;
        IntPtr biasHhPtr = biasHh.Handle;
        IntPtr outputPtr = output.Handle;
        IntPtr hFinalPtr = hFinal.Handle;
        IntPtr allHPtr = allH.Handle;
        IntPtr cacheGatesPtr = cacheGates.Handle;

        void** args = stackalloc void*[14];
        args[0] = &inputPtr;
        args[1] = &hInitPtr;
        args[2] = &weightsIhPtr;
        args[3] = &weightsHhPtr;
        args[4] = &biasIhPtr;
        args[5] = &biasHhPtr;
        args[6] = &outputPtr;
        args[7] = &hFinalPtr;
        args[8] = &allHPtr;
        args[9] = &cacheGatesPtr;
        args[10] = &seqLen;
        args[11] = &batch;
        args[12] = &inputSize;
        args[13] = &hiddenSize;

        LaunchKernel(kernel, grid, (uint)hiddenSize, args);
    }

    public unsafe void GruBackwardSequence(
        IGpuBuffer gradOutput, IGpuBuffer allH, IGpuBuffer cacheGates,
        IGpuBuffer weightsIh, IGpuBuffer weightsHh, IGpuBuffer input,
        IGpuBuffer gradInput, IGpuBuffer gradHInit, IGpuBuffer dHBuffer,
        IGpuBuffer gradWeightsIh, IGpuBuffer gradWeightsHh, IGpuBuffer gradBiasIh, IGpuBuffer gradBiasHh,
        int seqLen, int batch, int inputSize, int hiddenSize)
    {
        if (!_kernelCache.TryGetValue("gru_backward_sequence", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: gru_backward_sequence");

        // Validate hiddenSize fits in a block for correct synchronization
        if (hiddenSize > MaxRnnBlockSize)
        {
            throw new InvalidOperationException(
                $"GRU backward sequence hiddenSize ({hiddenSize}) exceeds max block size ({MaxRnnBlockSize}). " +
                "The kernel requires all hidden units of a batch to fit within a single block for " +
                "correct synchronization. Use smaller hiddenSize or use cell-level GRU operations.");
        }

        using var _ = PushContext();

        // Grid = one block per batch sample, block = hiddenSize threads per batch
        // This ensures __syncthreads() correctly synchronizes all threads for the same batch
        uint grid = (uint)batch;

        // Shared memory size for accumulated hidden gradients (one float per thread)
        uint sharedMemSize = (uint)(DefaultBlockSize * sizeof(float));

        IntPtr gradOutputPtr = gradOutput.Handle;
        IntPtr allHPtr = allH.Handle;
        IntPtr cacheGatesPtr = cacheGates.Handle;
        IntPtr weightsIhPtr = weightsIh.Handle;
        IntPtr weightsHhPtr = weightsHh.Handle;
        IntPtr inputPtr = input.Handle;
        IntPtr gradInputPtr = gradInput.Handle;
        IntPtr gradHInitPtr = gradHInit.Handle;
        IntPtr dHBufferPtr = dHBuffer.Handle;
        IntPtr gradWeightsIhPtr = gradWeightsIh.Handle;
        IntPtr gradWeightsHhPtr = gradWeightsHh.Handle;
        IntPtr gradBiasIhPtr = gradBiasIh.Handle;
        IntPtr gradBiasHhPtr = gradBiasHh.Handle;

        void** args = stackalloc void*[17];
        args[0] = &gradOutputPtr;
        args[1] = &allHPtr;
        args[2] = &cacheGatesPtr;
        args[3] = &weightsIhPtr;
        args[4] = &weightsHhPtr;
        args[5] = &inputPtr;
        args[6] = &gradInputPtr;
        args[7] = &gradHInitPtr;
        args[8] = &dHBufferPtr;
        args[9] = &gradWeightsIhPtr;
        args[10] = &gradWeightsHhPtr;
        args[11] = &gradBiasIhPtr;
        args[12] = &gradBiasHhPtr;
        args[13] = &seqLen;
        args[14] = &batch;
        args[15] = &inputSize;
        args[16] = &hiddenSize;

        // Use cooperative kernel launch for grid-wide synchronization (grid.sync())
        LaunchCooperativeKernel(kernel, grid, (uint)hiddenSize, sharedMemSize, args);
    }

    public unsafe void GruCellBackward(
        IGpuBuffer gradH, IGpuBuffer gateR, IGpuBuffer gateZ, IGpuBuffer gateN, IGpuBuffer prevH,
        IGpuBuffer weightsHh,
        IGpuBuffer gradPrevH, IGpuBuffer gradGateR, IGpuBuffer gradGateZ, IGpuBuffer gradGateN,
        int batch, int hiddenSize)
    {
        using var _ = PushContext();

        int totalThreads = batch * hiddenSize;
        uint grid = (uint)((totalThreads + DefaultBlockSize - 1) / DefaultBlockSize);

        // Step 1: Call gru_cell_backward_unified to compute gate gradients and partial gradPrevH
        if (!_kernelCache.TryGetValue("gru_cell_backward_unified", out var cellBackwardKernel))
            throw new InvalidOperationException("CUDA kernel not found: gru_cell_backward_unified");

        IntPtr gradHPtr = gradH.Handle;
        IntPtr gateRPtr = gateR.Handle;
        IntPtr gateZPtr = gateZ.Handle;
        IntPtr gateNPtr = gateN.Handle;
        IntPtr prevHPtr = prevH.Handle;
        IntPtr weightsHhPtr = weightsHh.Handle;
        IntPtr gradPrevHPtr = gradPrevH.Handle;
        IntPtr gradGateRPtr = gradGateR.Handle;
        IntPtr gradGateZPtr = gradGateZ.Handle;
        IntPtr gradGateNPtr = gradGateN.Handle;

        void** args1 = stackalloc void*[12];
        args1[0] = &gradHPtr;
        args1[1] = &gateRPtr;
        args1[2] = &gateZPtr;
        args1[3] = &gateNPtr;
        args1[4] = &prevHPtr;
        args1[5] = &weightsHhPtr;
        args1[6] = &gradPrevHPtr;
        args1[7] = &gradGateRPtr;
        args1[8] = &gradGateZPtr;
        args1[9] = &gradGateNPtr;
        args1[10] = &batch;
        args1[11] = &hiddenSize;

        LaunchKernel(cellBackwardKernel, grid, DefaultBlockSize, args1);

        // Step 2: Call gru_backward_prevh_unified to compute full gradPrevH using all gate gradients
        if (!_kernelCache.TryGetValue("gru_backward_prevh_unified", out var prevhKernel))
            throw new InvalidOperationException("CUDA kernel not found: gru_backward_prevh_unified");

        void** args2 = stackalloc void*[10];
        args2[0] = &gradGateRPtr;
        args2[1] = &gradGateZPtr;
        args2[2] = &gradGateNPtr;
        args2[3] = &gradHPtr;
        args2[4] = &gateRPtr;
        args2[5] = &gateZPtr;
        args2[6] = &weightsHhPtr;
        args2[7] = &gradPrevHPtr;
        args2[8] = &batch;
        args2[9] = &hiddenSize;

        LaunchKernel(prevhKernel, grid, DefaultBlockSize, args2);
    }

    #endregion


    public void Dispose()
    {
        Dispose(disposing: true);
        GC.SuppressFinalize(this);
    }

    private void Dispose(bool disposing)
    {
        if (_disposed)
            return;

        _disposed = true;
        // Finalizer path (disposing == false) or any process/domain teardown:
        //   - The managed members (GpuBufferPool's ConcurrentBag/ThreadLocal, the cuDNN
        //     helpers) may already have been finalized by the runtime — touching them
        //     throws ObjectDisposedException from the finalizer thread (fatal).
        //   - The CUDA context may already be destroyed — every driver call below
        //     (cublasDestroy / cuStreamDestroy / cuModuleUnload / cuCtxDestroy) would
        //     raise an uncatchable access violation (0xC0000005), also finalizer-fatal.
        // The OS reclaims the context and all device memory at process exit, so the only
        // safe action here is to do nothing. Explicit Dispose during normal operation
        // (disposing == true, not tearing down) still performs the full native cleanup.
        // Guard on the union of both teardown signals: IsRuntimeTearingDown (this PR) and
        // ProcessExiting (the merged #536 CUDA-finalizer-safety fix) — either means the
        // driver/runtime may be gone, so skip the native + pool cleanup entirely.
        if (!disposing || IsRuntimeTearingDown || ProcessExiting)
            return;

        // Flush every event-deferred free BEFORE tearing down the pool/context, blocking
        // on incomplete events (the stream/context are still alive right here, so the
        // waits are safe). Anything left queued would survive this Dispose holding its
        // now-stale _context handle; when those CudaGpuBuffer instances later finalize
        // they would cuCtxPushCurrent/cuMemFree against a DESTROYED context — the exact
        // finalizer-fatal 0xC0000005 path this method's teardown guards exist to prevent.
        DrainDeferredFrees(blockOldest: true);

        _pinnedPool.Dispose();
        _bufferPool.Dispose();

        // cuDNN helper disposes its workspace + context. Only created if
        // Conv2D routed through the vendor path at least once.
        if (_cudnnConv is not null)
        {
            _cudnnConv.Dispose();
            _cudnnConv = null;
        }
        if (_cudnnContext is not null)
        {
            _cudnnContext.Dispose();
            _cudnnContext = null;
        }

        if (_cublasHandle != IntPtr.Zero)
        {
            CuBlasNative.cublasDestroy(_cublasHandle);
            _cublasHandle = IntPtr.Zero;
        }

        if (_stream != IntPtr.Zero)
        {
            CudaNativeBindings.cuStreamDestroy(_stream);
            _stream = IntPtr.Zero;
        }

        if (_convolutionModule != IntPtr.Zero)
        {
            CudaNativeBindings.cuModuleUnload(_convolutionModule);
            _convolutionModule = IntPtr.Zero;
        }

        if (_poolingModule != IntPtr.Zero)
        {
            CudaNativeBindings.cuModuleUnload(_poolingModule);
            _poolingModule = IntPtr.Zero;
        }

        if (_normalizationModule != IntPtr.Zero)
        {
            CudaNativeBindings.cuModuleUnload(_normalizationModule);
            _normalizationModule = IntPtr.Zero;
        }

        if (_neuralNetModule != IntPtr.Zero)
        {
            CudaNativeBindings.cuModuleUnload(_neuralNetModule);
            _neuralNetModule = IntPtr.Zero;
        }

        if (_activationModule != IntPtr.Zero)
        {
            CudaNativeBindings.cuModuleUnload(_activationModule);
            _activationModule = IntPtr.Zero;
        }

        if (_parity210Module != IntPtr.Zero)
        {
            CudaNativeBindings.cuModuleUnload(_parity210Module);
            _parity210Module = IntPtr.Zero;
        }

        if (_fftModule != IntPtr.Zero)
        {
            CudaNativeBindings.cuModuleUnload(_fftModule);
            _fftModule = IntPtr.Zero;
        }

        if (_spectralPerfModule != IntPtr.Zero)
        {
            CudaNativeBindings.cuModuleUnload(_spectralPerfModule);
            _spectralPerfModule = IntPtr.Zero;
        }

        if (_spatialTransformerModule != IntPtr.Zero)
        {
            CudaNativeBindings.cuModuleUnload(_spatialTransformerModule);
            _spatialTransformerModule = IntPtr.Zero;
        }

        if (_locallyConnectedModule != IntPtr.Zero)
        {
            CudaNativeBindings.cuModuleUnload(_locallyConnectedModule);
            _locallyConnectedModule = IntPtr.Zero;
        }

        if (_deformableConvModule != IntPtr.Zero)
        {
            CudaNativeBindings.cuModuleUnload(_deformableConvModule);
            _deformableConvModule = IntPtr.Zero;
        }

        if (_capsuleModule != IntPtr.Zero)
        {
            CudaNativeBindings.cuModuleUnload(_capsuleModule);
            _capsuleModule = IntPtr.Zero;
        }

        if (_specializedModule != IntPtr.Zero)
        {
            CudaNativeBindings.cuModuleUnload(_specializedModule);
            _specializedModule = IntPtr.Zero;
        }

        if (_fp16Module != IntPtr.Zero)
        {
            CudaNativeBindings.cuModuleUnload(_fp16Module);
            _fp16Module = IntPtr.Zero;
        }

        if (_lstmModule != IntPtr.Zero)
        {
            CudaNativeBindings.cuModuleUnload(_lstmModule);
            _lstmModule = IntPtr.Zero;
        }

        if (_glaModule != IntPtr.Zero)
        {
            CudaNativeBindings.cuModuleUnload(_glaModule);
            _glaModule = IntPtr.Zero;
        }

        if (_xlstmModule != IntPtr.Zero)
        {
            CudaNativeBindings.cuModuleUnload(_xlstmModule);
            _xlstmModule = IntPtr.Zero;
        }

        if (_gatedDeltaNetModule != IntPtr.Zero)
        {
            CudaNativeBindings.cuModuleUnload(_gatedDeltaNetModule);
            _gatedDeltaNetModule = IntPtr.Zero;
        }

        if (_rgLruModule != IntPtr.Zero)
        {
            CudaNativeBindings.cuModuleUnload(_rgLruModule);
            _rgLruModule = IntPtr.Zero;
        }

        if (_rwkv4Module != IntPtr.Zero)
        {
            CudaNativeBindings.cuModuleUnload(_rwkv4Module);
            _rwkv4Module = IntPtr.Zero;
        }

        if (_mambaModule != IntPtr.Zero)
        {
            CudaNativeBindings.cuModuleUnload(_mambaModule);
            _mambaModule = IntPtr.Zero;
        }

        if (_mamba2Module != IntPtr.Zero)
        {
            CudaNativeBindings.cuModuleUnload(_mamba2Module);
            _mamba2Module = IntPtr.Zero;
        }

        if (_fusedCeModule != IntPtr.Zero)
        {
            CudaNativeBindings.cuModuleUnload(_fusedCeModule);
            _fusedCeModule = IntPtr.Zero;
        }

        if (_gruModule != IntPtr.Zero)
        {
            CudaNativeBindings.cuModuleUnload(_gruModule);
            _gruModule = IntPtr.Zero;
        }

        if (_snnModule != IntPtr.Zero)
        {
            CudaNativeBindings.cuModuleUnload(_snnModule);
            _snnModule = IntPtr.Zero;
        }

        if (_wmmaModule != IntPtr.Zero)
        {
            CudaNativeBindings.cuModuleUnload(_wmmaModule);
            _wmmaModule = IntPtr.Zero;
        }

        if (_fusedConvolutionModule != IntPtr.Zero)
        {
            CudaNativeBindings.cuModuleUnload(_fusedConvolutionModule);
            _fusedConvolutionModule = IntPtr.Zero;
        }

        if (_fusedModule != IntPtr.Zero)
        {
            CudaNativeBindings.cuModuleUnload(_fusedModule);
            _fusedModule = IntPtr.Zero;
        }

        if (_attentionModule != IntPtr.Zero)
        {
            CudaNativeBindings.cuModuleUnload(_attentionModule);
            _attentionModule = IntPtr.Zero;
        }

        if (_sparseModule != IntPtr.Zero)
        {
            CudaNativeBindings.cuModuleUnload(_sparseModule);
            _sparseModule = IntPtr.Zero;
        }

        if (_linalgModule != IntPtr.Zero)
        {
            CudaNativeBindings.cuModuleUnload(_linalgModule);
            _linalgModule = IntPtr.Zero;
        }

        if (_detectionModule != IntPtr.Zero)
        {
            CudaNativeBindings.cuModuleUnload(_detectionModule);
            _detectionModule = IntPtr.Zero;
        }

        if (_geometryModule != IntPtr.Zero)
        {
            CudaNativeBindings.cuModuleUnload(_geometryModule);
            _geometryModule = IntPtr.Zero;
        }

        if (_roiModule != IntPtr.Zero)
        {
            CudaNativeBindings.cuModuleUnload(_roiModule);
            _roiModule = IntPtr.Zero;
        }

        if (_audioModule != IntPtr.Zero)
        {
            CudaNativeBindings.cuModuleUnload(_audioModule);
            _audioModule = IntPtr.Zero;
        }

        if (_cudaContext != IntPtr.Zero)
        {
            // Unregister + destroy under the lifecycle lock so a concurrent buffer finalizer can't
            // observe the context as live and then fault on it after we destroy it (TOCTOU). A buffer
            // finalized after this point sees it gone and skips its native free (cuCtxDestroy reclaims
            // that memory anyway).
            lock (ContextLifecycleLock)
            {
                LiveContexts.TryRemove(_cudaContext, out _);
                CuBlasNative.cuCtxDestroy(_cudaContext);
                _cudaContext = IntPtr.Zero;
            }
        }

        GC.SuppressFinalize(this);
    }

    #region Fused Kernel Dispatch (84 methods)

    // Helper: launch a kernel by name with stackalloc args
    private unsafe void LaunchFusedUnary(string kernelName, IGpuBuffer input, IGpuBuffer output, int size)
    {
        if (!_kernelCache.TryGetValue(kernelName, out var kernel))
            throw new InvalidOperationException($"CUDA kernel not found: {kernelName}");
        using var _ = PushContext();
        IntPtr inPtr = input.Handle, outPtr = output.Handle;
        void** args = stackalloc void*[3];
        args[0] = &inPtr; args[1] = &outPtr; args[2] = &size;
        LaunchKernel(kernel, (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize), DefaultBlockSize, args);
    }

    private unsafe void LaunchFusedBinary(string kernelName, IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, int size)
    {
        if (!_kernelCache.TryGetValue(kernelName, out var kernel))
            throw new InvalidOperationException($"CUDA kernel not found: {kernelName}");
        using var _ = PushContext();
        IntPtr aPtr = a.Handle, bPtr = b.Handle, outPtr = output.Handle;
        void** args = stackalloc void*[4];
        args[0] = &aPtr; args[1] = &bPtr; args[2] = &outPtr; args[3] = &size;
        LaunchKernel(kernel, (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize), DefaultBlockSize, args);
    }

    private unsafe void LaunchFusedAxis(string kernelName, IGpuBuffer input, IGpuBuffer output, int outerSize, int reduceSize)
    {
        if (!_kernelCache.TryGetValue(kernelName, out var kernel))
            throw new InvalidOperationException($"CUDA kernel not found: {kernelName}");
        using var _ = PushContext();
        IntPtr inPtr = input.Handle, outPtr = output.Handle;
        void** args = stackalloc void*[4];
        args[0] = &inPtr; args[1] = &outPtr; args[2] = &outerSize; args[3] = &reduceSize;
        LaunchKernel(kernel, (uint)((outerSize + DefaultBlockSize - 1) / DefaultBlockSize), DefaultBlockSize, args);
    }

    private unsafe void LaunchFusedScalar(string kernelName, IGpuBuffer input, IGpuBuffer output, float scalar, int size)
    {
        if (!_kernelCache.TryGetValue(kernelName, out var kernel))
            throw new InvalidOperationException($"CUDA kernel not found: {kernelName}");
        using var _ = PushContext();
        IntPtr inPtr = input.Handle, outPtr = output.Handle;
        void** args = stackalloc void*[4];
        args[0] = &inPtr; args[1] = &outPtr; args[2] = &scalar; args[3] = &size;
        LaunchKernel(kernel, (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize), DefaultBlockSize, args);
    }

    // --- Reductions ---
    // Issue #382: when GpuDeterminism.IsActive, route to single-block strided
    // deterministic variants (no inter-block atomic combine).
    public unsafe void ReduceMean(IGpuBuffer input, IGpuBuffer output, int size)
        => LaunchFusedUnaryReduce(
            GpuDeterminism.IsActive ? "reduce_mean_deterministic" : "reduce_mean",
            input, output, size, GpuDeterminism.IsActive);
    public void ReduceProduct(IGpuBuffer input, IGpuBuffer output, int size) => LaunchFusedUnary("reduce_product", input, output, size);
    public unsafe void ReduceNormL2(IGpuBuffer input, IGpuBuffer output, int size)
        => LaunchFusedUnaryReduce(
            GpuDeterminism.IsActive ? "reduce_norm_l2_deterministic" : "reduce_norm_l2",
            input, output, size, GpuDeterminism.IsActive);
    public unsafe void ReduceSumOfSquares(IGpuBuffer input, IGpuBuffer output, int size)
        => LaunchFusedUnaryReduce(
            GpuDeterminism.IsActive ? "reduce_sum_of_squares_deterministic" : "reduce_sum_of_squares",
            input, output, size, GpuDeterminism.IsActive);

    /// <summary>
    /// Issue #382: variant of LaunchFusedUnary for scalar reductions that
    /// honors deterministic mode. When deterministic=true, launches a single block
    /// (the deterministic kernels do a strided reduce in one block) and pre-zeros
    /// the output (the kernel does a direct write, not an atomic accumulate).
    /// </summary>
    private unsafe void LaunchFusedUnaryReduce(string kernelName, IGpuBuffer input, IGpuBuffer output, int size, bool deterministic)
    {
        if (!_kernelCache.TryGetValue(kernelName, out var kernel))
            throw new InvalidOperationException($"CUDA kernel not found: {kernelName}");
        using var _ = PushContext();
        if (deterministic) Fill(output, 0f, 1);
        IntPtr inPtr = input.Handle, outPtr = output.Handle;
        void** args = stackalloc void*[3];
        args[0] = &inPtr; args[1] = &outPtr; args[2] = &size;
        uint grid = deterministic ? 1u : (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }
    public void ReduceMaxMagnitude(IGpuBuffer input, IGpuBuffer output, int size) => LaunchFusedUnary("reduce_max_magnitude", input, output, size);
    public void ReduceMinMagnitude(IGpuBuffer input, IGpuBuffer output, int size) => LaunchFusedUnary("reduce_min_magnitude", input, output, size);

    public unsafe void ReduceLogSumExp(IGpuBuffer input, IGpuBuffer output, float maxVal, int size)
    {
        using var _ = PushContext();
        IntPtr inPtr = input.Handle, outPtr = output.Handle;
        void** args = stackalloc void*[4];
        args[0] = &inPtr; args[1] = &outPtr; args[2] = &maxVal; args[3] = &size;

        if (GpuDeterminism.IsActive)
        {
            if (!_kernelCache.TryGetValue("reduce_logsumexp_deterministic", out var kernelD))
                throw new InvalidOperationException("CUDA kernel not found: reduce_logsumexp_deterministic");
            Fill(output, 0f, 1);
            LaunchKernel(kernelD, 1, DefaultBlockSize, args);
            return;
        }

        if (!_kernelCache.TryGetValue("reduce_logsumexp", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: reduce_logsumexp");
        LaunchKernel(kernel, (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize), DefaultBlockSize, args);
    }

    public void VarianceAxis(IGpuBuffer input, IGpuBuffer output, int outerSize, int reduceSize) => LaunchFusedAxis("variance_axis", input, output, outerSize, reduceSize);
    public void StdAxis(IGpuBuffer input, IGpuBuffer output, int outerSize, int reduceSize) => LaunchFusedAxis("std_axis", input, output, outerSize, reduceSize);
    public void ProductAxis(IGpuBuffer input, IGpuBuffer output, int outerSize, int reduceSize) => LaunchFusedAxis("product_axis", input, output, outerSize, reduceSize);
    public void NormAxis(IGpuBuffer input, IGpuBuffer output, int outerSize, int reduceSize) => LaunchFusedAxis("norm_axis", input, output, outerSize, reduceSize);
    public void LogSumExpAxis(IGpuBuffer input, IGpuBuffer output, int outerSize, int reduceSize) => LaunchFusedAxis("logsumexp_axis", input, output, outerSize, reduceSize);
    public void CumSumAxis(IGpuBuffer input, IGpuBuffer output, int outerSize, int innerSize) => LaunchFusedAxis("cumsum_axis", input, output, outerSize, innerSize);
    public void ScalarMinusTensor(IGpuBuffer input, IGpuBuffer output, float scalar, int size) => LaunchFusedScalar("scalar_minus_tensor", input, output, scalar, size);
    public void NormalizeL2(IGpuBuffer input, IGpuBuffer output, int outerSize, int innerSize) => LaunchFusedAxis("normalize_l2", input, output, outerSize, innerSize);
    public void ReduceSumBackward(IGpuBuffer gradOutput, IGpuBuffer gradInput, int outerSize, int reduceSize) => LaunchFusedAxis("reduce_sum_backward", gradOutput, gradInput, outerSize, reduceSize);
    public void ReduceMeanBackward(IGpuBuffer gradOutput, IGpuBuffer gradInput, int outerSize, int reduceSize) => LaunchFusedAxis("reduce_mean_backward", gradOutput, gradInput, outerSize, reduceSize);

    public unsafe void ReduceMaxBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer maxValues, IGpuBuffer gradInput, int outerSize, int reduceSize)
    {
        if (!_kernelCache.TryGetValue("reduce_max_backward", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: reduce_max_backward");
        using var _ = PushContext();
        IntPtr goPtr = gradOutput.Handle, inPtr = input.Handle, maxPtr = maxValues.Handle, giPtr = gradInput.Handle;
        void** args = stackalloc void*[6];
        args[0] = &goPtr; args[1] = &inPtr; args[2] = &maxPtr; args[3] = &giPtr; args[4] = &outerSize; args[5] = &reduceSize;
        uint total = (uint)(outerSize * reduceSize);
        LaunchKernel(kernel, (total + DefaultBlockSize - 1) / DefaultBlockSize, DefaultBlockSize, args);
    }

    public unsafe void ReduceVarianceBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer means, IGpuBuffer gradInput, int outerSize, int reduceSize)
    {
        if (!_kernelCache.TryGetValue("reduce_variance_backward", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: reduce_variance_backward");
        using var _ = PushContext();
        IntPtr goPtr = gradOutput.Handle, inPtr = input.Handle, mPtr = means.Handle, giPtr = gradInput.Handle;
        void** args = stackalloc void*[6];
        args[0] = &goPtr; args[1] = &inPtr; args[2] = &mPtr; args[3] = &giPtr; args[4] = &outerSize; args[5] = &reduceSize;
        uint total = (uint)(outerSize * reduceSize);
        LaunchKernel(kernel, (total + DefaultBlockSize - 1) / DefaultBlockSize, DefaultBlockSize, args);
    }

    public void ReduceLogVariance(IGpuBuffer input, IGpuBuffer output, int outerSize, int reduceSize) => LaunchFusedAxis("reduce_log_variance", input, output, outerSize, reduceSize);

    public unsafe void ReduceLogVarianceBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer means, IGpuBuffer variances, IGpuBuffer gradInput, int outerSize, int reduceSize)
    {
        if (!_kernelCache.TryGetValue("reduce_log_variance_backward", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: reduce_log_variance_backward");
        using var _ = PushContext();
        IntPtr goPtr = gradOutput.Handle, inPtr = input.Handle, mPtr = means.Handle, vPtr = variances.Handle, giPtr = gradInput.Handle;
        void** args = stackalloc void*[7];
        args[0] = &goPtr; args[1] = &inPtr; args[2] = &mPtr; args[3] = &vPtr; args[4] = &giPtr; args[5] = &outerSize; args[6] = &reduceSize;
        uint total = (uint)(outerSize * reduceSize);
        LaunchKernel(kernel, (total + DefaultBlockSize - 1) / DefaultBlockSize, DefaultBlockSize, args);
    }

    // --- Broadcast / Scalar ---
    public unsafe void BroadcastAddLast(IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, int outerSize, int innerSize)
    {
        if (!_kernelCache.TryGetValue("broadcast_add_last", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: broadcast_add_last");
        using var _ = PushContext();
        IntPtr aPtr = a.Handle, bPtr = b.Handle, outPtr = output.Handle;
        void** args = stackalloc void*[5];
        args[0] = &aPtr; args[1] = &bPtr; args[2] = &outPtr; args[3] = &outerSize; args[4] = &innerSize;
        uint total = (uint)(outerSize * innerSize);
        LaunchKernel(kernel, (total + DefaultBlockSize - 1) / DefaultBlockSize, DefaultBlockSize, args);
    }
    public void BroadcastSubLast(IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, int outerSize, int innerSize) { BroadcastOpLast("broadcast_sub_last", a, b, output, outerSize, innerSize); }
    public void BroadcastMulLast(IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, int outerSize, int innerSize) { BroadcastOpLast("broadcast_mul_last", a, b, output, outerSize, innerSize); }
    public void BroadcastDivLast(IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, int outerSize, int innerSize) { BroadcastOpLast("broadcast_div_last", a, b, output, outerSize, innerSize); }
    public void BroadcastAddFirst(IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, int outerSize, int innerSize) { BroadcastOpLast("broadcast_add_first", a, b, output, outerSize, innerSize); }
    public void BroadcastMulFirst(IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, int outerSize, int innerSize) { BroadcastOpLast("broadcast_mul_first", a, b, output, outerSize, innerSize); }

    private unsafe void BroadcastOpLast(string kernelName, IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, int outerSize, int innerSize)
    {
        if (!_kernelCache.TryGetValue(kernelName, out var kernel))
            throw new InvalidOperationException($"CUDA kernel not found: {kernelName}");
        using var _ = PushContext();
        IntPtr aPtr = a.Handle, bPtr = b.Handle, outPtr = output.Handle;
        void** args = stackalloc void*[5];
        args[0] = &aPtr; args[1] = &bPtr; args[2] = &outPtr; args[3] = &outerSize; args[4] = &innerSize;
        uint total = (uint)(outerSize * innerSize);
        LaunchKernel(kernel, (total + DefaultBlockSize - 1) / DefaultBlockSize, DefaultBlockSize, args);
    }

    public void AddScalar(IGpuBuffer input, IGpuBuffer output, float scalar, int size) => LaunchFusedScalar("add_scalar", input, output, scalar, size);
    public void SubScalar(IGpuBuffer input, IGpuBuffer output, float scalar, int size) => LaunchFusedScalar("sub_scalar", input, output, scalar, size);
    public void DivScalar(IGpuBuffer input, IGpuBuffer output, float scalar, int size) => LaunchFusedScalar("div_scalar", input, output, scalar, size);
    public void PowScalar(IGpuBuffer input, IGpuBuffer output, float exponent, int size) => LaunchFusedScalar("pow_scalar", input, output, exponent, size);
    public void FracKernel(IGpuBuffer input, IGpuBuffer output, int size) => LaunchFusedUnary("frac_kernel", input, output, size);

    public unsafe void ClipKernel(IGpuBuffer input, IGpuBuffer output, float min, float max, int size)
    {
        if (!_kernelCache.TryGetValue("clip_kernel", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: clip_kernel");
        using var _ = PushContext();
        IntPtr inPtr = input.Handle, outPtr = output.Handle;
        void** args = stackalloc void*[5];
        args[0] = &inPtr; args[1] = &outPtr; args[2] = &min; args[3] = &max; args[4] = &size;
        LaunchKernel(kernel, (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize), DefaultBlockSize, args);
    }

    public void RsqrtKernel(IGpuBuffer input, IGpuBuffer output, int size) => LaunchFusedUnary("rsqrt_kernel", input, output, size);

    public unsafe void SinCosKernel(IGpuBuffer input, IGpuBuffer sinOutput, IGpuBuffer cosOutput, int size)
    {
        if (!_kernelCache.TryGetValue("sincos_kernel", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: sincos_kernel");
        using var _ = PushContext();
        IntPtr inPtr = input.Handle, sinPtr = sinOutput.Handle, cosPtr = cosOutput.Handle;
        void** args = stackalloc void*[4];
        args[0] = &inPtr; args[1] = &sinPtr; args[2] = &cosPtr; args[3] = &size;
        LaunchKernel(kernel, (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize), DefaultBlockSize, args);
    }

    public void EqualsKernel(IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, int size) => LaunchFusedBinary("equals_kernel", a, b, output, size);
    public void NotEqualsKernel(IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, int size) => LaunchFusedBinary("not_equals_kernel", a, b, output, size);

    // --- Gated Activations ---
    public void GluForward(IGpuBuffer input, IGpuBuffer output, int outerSize, int halfDim) => LaunchFusedAxis("glu_forward", input, output, outerSize, halfDim);
    public unsafe void GluBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradInput, int outerSize, int halfDim)
    {
        if (!_kernelCache.TryGetValue("glu_backward", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: glu_backward");
        using var _ = PushContext();
        IntPtr goPtr = gradOutput.Handle, inPtr = input.Handle, giPtr = gradInput.Handle;
        void** args = stackalloc void*[5];
        args[0] = &goPtr; args[1] = &inPtr; args[2] = &giPtr; args[3] = &outerSize; args[4] = &halfDim;
        uint total = (uint)(outerSize * halfDim);
        LaunchKernel(kernel, (total + DefaultBlockSize - 1) / DefaultBlockSize, DefaultBlockSize, args);
    }
    public void GeGluForward(IGpuBuffer input, IGpuBuffer output, int outerSize, int halfDim) => LaunchFusedAxis("geglu_forward", input, output, outerSize, halfDim);
    public unsafe void GeGluBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradInput, int outerSize, int halfDim) { LaunchGatedBackward("geglu_backward", gradOutput, input, gradInput, outerSize, halfDim); }
    public void ReGluForward(IGpuBuffer input, IGpuBuffer output, int outerSize, int halfDim) => LaunchFusedAxis("reglu_forward", input, output, outerSize, halfDim);
    public unsafe void ReGluBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradInput, int outerSize, int halfDim) { LaunchGatedBackward("reglu_backward", gradOutput, input, gradInput, outerSize, halfDim); }
    public void SwiGluForward(IGpuBuffer input, IGpuBuffer output, int outerSize, int halfDim) => LaunchFusedAxis("swiglu_forward", input, output, outerSize, halfDim);
    public unsafe void SwiGluBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradInput, int outerSize, int halfDim) { LaunchGatedBackward("swiglu_backward", gradOutput, input, gradInput, outerSize, halfDim); }

    private unsafe void LaunchGatedBackward(string kernelName, IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradInput, int outerSize, int halfDim)
    {
        if (!_kernelCache.TryGetValue(kernelName, out var kernel))
            throw new InvalidOperationException($"CUDA kernel not found: {kernelName}");
        using var _ = PushContext();
        IntPtr goPtr = gradOutput.Handle, inPtr = input.Handle, giPtr = gradInput.Handle;
        void** args = stackalloc void*[5];
        args[0] = &goPtr; args[1] = &inPtr; args[2] = &giPtr; args[3] = &outerSize; args[4] = &halfDim;
        uint total = (uint)(outerSize * halfDim);
        LaunchKernel(kernel, (total + DefaultBlockSize - 1) / DefaultBlockSize, DefaultBlockSize, args);
    }

    public void ReluDerivative(IGpuBuffer input, IGpuBuffer output, int size) => LaunchFusedUnary("relu_derivative", input, output, size);
    public void SigmoidDerivative(IGpuBuffer sigmoidOutput, IGpuBuffer output, int size) => LaunchFusedUnary("sigmoid_derivative", sigmoidOutput, output, size);
    public void TanhDerivative(IGpuBuffer tanhOutput, IGpuBuffer output, int size) => LaunchFusedUnary("tanh_derivative", tanhOutput, output, size);

    // --- Shape / Layout ---
    public unsafe void ConcatAxis(IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, int outerSize, int aInnerSize, int bInnerSize)
    {
        if (!_kernelCache.TryGetValue("concat_axis", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: concat_axis");
        using var _ = PushContext();
        IntPtr aPtr = a.Handle, bPtr = b.Handle, outPtr = output.Handle;
        void** args = stackalloc void*[6];
        args[0] = &aPtr; args[1] = &bPtr; args[2] = &outPtr; args[3] = &outerSize; args[4] = &aInnerSize; args[5] = &bInnerSize;
        uint total = (uint)(outerSize * (aInnerSize + bInnerSize));
        LaunchKernel(kernel, (total + DefaultBlockSize - 1) / DefaultBlockSize, DefaultBlockSize, args);
    }

    public unsafe void SliceLastAxis(IGpuBuffer input, IGpuBuffer output, int outerSize, int inputInnerSize, int start, int sliceSize)
    {
        if (!_kernelCache.TryGetValue("slice_last_axis", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: slice_last_axis");
        using var _ = PushContext();
        IntPtr inPtr = input.Handle, outPtr = output.Handle;
        void** args = stackalloc void*[6];
        args[0] = &inPtr; args[1] = &outPtr; args[2] = &outerSize; args[3] = &inputInnerSize; args[4] = &start; args[5] = &sliceSize;
        uint total = (uint)(outerSize * sliceSize);
        LaunchKernel(kernel, (total + DefaultBlockSize - 1) / DefaultBlockSize, DefaultBlockSize, args);
    }

    public unsafe void SetSliceLastAxis(IGpuBuffer output, IGpuBuffer values, int outerSize, int outputInnerSize, int start, int sliceSize)
    {
        if (!_kernelCache.TryGetValue("set_slice_last_axis", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: set_slice_last_axis");
        using var _ = PushContext();
        IntPtr outPtr = output.Handle, valPtr = values.Handle;
        void** args = stackalloc void*[6];
        args[0] = &outPtr; args[1] = &valPtr; args[2] = &outerSize; args[3] = &outputInnerSize; args[4] = &start; args[5] = &sliceSize;
        uint total = (uint)(outerSize * sliceSize);
        LaunchKernel(kernel, (total + DefaultBlockSize - 1) / DefaultBlockSize, DefaultBlockSize, args);
    }

    public unsafe void SliceAxis(IGpuBuffer input, IGpuBuffer output, int outerSize, int axisSize, int stride, int index)
    {
        // For last axis (stride=1), delegate to optimized SliceLastAxis kernel
        if (stride == 1)
        {
            SliceLastAxis(input, output, outerSize, axisSize, index, 1);
            return;
        }
        // General case: each thread copies one element from the correct strided position
        // GPU kernel: out[outer * stride + s] = in[outer * axisSize * stride + index * stride + s]
        if (!_kernelCache.TryGetValue("slice_axis", out var kernel))
            throw new InvalidOperationException("CUDA slice_axis kernel not compiled. Ensure CudaShapeKernels are loaded.");
        using var _ = PushContext();
        IntPtr inPtr = input.Handle, outPtr = output.Handle;
        void** args = stackalloc void*[6];
        args[0] = &inPtr; args[1] = &outPtr; args[2] = &outerSize; args[3] = &axisSize; args[4] = &stride; args[5] = &index;
        uint total = (uint)(outerSize * stride);
        LaunchKernel(kernel, (total + DefaultBlockSize - 1) / DefaultBlockSize, DefaultBlockSize, args);
    }

    public unsafe void SetSliceAxis(IGpuBuffer output, IGpuBuffer values, int outerSize, int axisSize, int stride, int index)
    {
        if (stride == 1)
        {
            SetSliceLastAxis(output, values, outerSize, axisSize, index, 1);
            return;
        }
        if (!_kernelCache.TryGetValue("set_slice_axis", out var kernel))
            throw new InvalidOperationException("CUDA set_slice_axis kernel not compiled. Ensure CudaShapeKernels are loaded.");
        using var _ = PushContext();
        IntPtr outPtr = output.Handle, valPtr = values.Handle;
        void** args = stackalloc void*[6];
        args[0] = &outPtr; args[1] = &valPtr; args[2] = &outerSize; args[3] = &axisSize; args[4] = &stride; args[5] = &index;
        uint total = (uint)(outerSize * stride);
        LaunchKernel(kernel, (total + DefaultBlockSize - 1) / DefaultBlockSize, DefaultBlockSize, args);
    }

    public void Stack2(IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, int size) => LaunchFusedBinary("stack_2", a, b, output, size);

    public unsafe void Pad2D(IGpuBuffer input, IGpuBuffer output, int batch, int channels, int inH, int inW, int outH, int outW, int padTop, int padLeft, float padValue)
    {
        if (!_kernelCache.TryGetValue("pad_2d", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: pad_2d");
        using var _ = PushContext();
        IntPtr inPtr = input.Handle, outPtr = output.Handle;
        void** args = stackalloc void*[10];
        args[0] = &inPtr; args[1] = &outPtr; args[2] = &batch; args[3] = &channels;
        args[4] = &inH; args[5] = &inW; args[6] = &outH; args[7] = &outW;
        args[8] = &padTop; args[9] = &padLeft;
        // padValue needs special handling — append it
        void** argsExt = stackalloc void*[11];
        for (int i = 0; i < 10; i++) argsExt[i] = args[i];
        argsExt[10] = &padValue;
        uint total = (uint)(batch * channels * outH * outW);
        LaunchKernel(kernel, (total + DefaultBlockSize - 1) / DefaultBlockSize, DefaultBlockSize, argsExt);
    }

    public unsafe void Pad2DBackward(IGpuBuffer gradOutput, IGpuBuffer gradInput, int batch, int channels, int inH, int inW, int outH, int outW, int padTop, int padLeft)
    {
        if (!_kernelCache.TryGetValue("pad_2d_backward", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: pad_2d_backward");
        using var _ = PushContext();
        IntPtr goPtr = gradOutput.Handle, giPtr = gradInput.Handle;
        void** args = stackalloc void*[10];
        args[0] = &goPtr; args[1] = &giPtr; args[2] = &batch; args[3] = &channels;
        args[4] = &inH; args[5] = &inW; args[6] = &outH; args[7] = &outW; args[8] = &padTop; args[9] = &padLeft;
        uint total = (uint)(batch * channels * inH * inW);
        LaunchKernel(kernel, (total + DefaultBlockSize - 1) / DefaultBlockSize, DefaultBlockSize, args);
    }

    public unsafe void TileLastAxis(IGpuBuffer input, IGpuBuffer output, int outerSize, int innerSize, int repeats)
    {
        if (!_kernelCache.TryGetValue("tile_last_axis", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: tile_last_axis");
        using var _ = PushContext();
        IntPtr inPtr = input.Handle, outPtr = output.Handle;
        void** args = stackalloc void*[5];
        args[0] = &inPtr; args[1] = &outPtr; args[2] = &outerSize; args[3] = &innerSize; args[4] = &repeats;
        uint total = (uint)(outerSize * innerSize * repeats);
        LaunchKernel(kernel, (total + DefaultBlockSize - 1) / DefaultBlockSize, DefaultBlockSize, args);
    }

    public unsafe void RepeatElements(IGpuBuffer input, IGpuBuffer output, int outerSize, int innerSize, int repeats)
    {
        if (!_kernelCache.TryGetValue("repeat_elements", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: repeat_elements");
        using var _ = PushContext();
        IntPtr inPtr = input.Handle, outPtr = output.Handle;
        void** args = stackalloc void*[5];
        args[0] = &inPtr; args[1] = &outPtr; args[2] = &outerSize; args[3] = &innerSize; args[4] = &repeats;
        uint total = (uint)(outerSize * innerSize * repeats);
        LaunchKernel(kernel, (total + DefaultBlockSize - 1) / DefaultBlockSize, DefaultBlockSize, args);
    }

    public unsafe void PixelShuffle(IGpuBuffer input, IGpuBuffer output, int batch, int channels, int inH, int inW, int scale)
    {
        if (!_kernelCache.TryGetValue("pixel_shuffle", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: pixel_shuffle");
        using var _ = PushContext();
        IntPtr inPtr = input.Handle, outPtr = output.Handle;
        void** args = stackalloc void*[7];
        args[0] = &inPtr; args[1] = &outPtr; args[2] = &batch; args[3] = &channels; args[4] = &inH; args[5] = &inW; args[6] = &scale;
        uint total = (uint)(batch * channels * inH * scale * inW * scale);
        LaunchKernel(kernel, (total + DefaultBlockSize - 1) / DefaultBlockSize, DefaultBlockSize, args);
    }

    public unsafe void PixelShuffleBackward(IGpuBuffer gradOutput, IGpuBuffer gradInput, int batch, int channels, int inH, int inW, int scale)
    {
        if (!_kernelCache.TryGetValue("pixel_shuffle_backward", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: pixel_shuffle_backward");
        using var _ = PushContext();
        IntPtr goPtr = gradOutput.Handle, giPtr = gradInput.Handle;
        void** args = stackalloc void*[7];
        args[0] = &goPtr; args[1] = &giPtr; args[2] = &batch; args[3] = &channels; args[4] = &inH; args[5] = &inW; args[6] = &scale;
        uint total = (uint)(batch * channels * scale * scale * inH * inW);
        LaunchKernel(kernel, (total + DefaultBlockSize - 1) / DefaultBlockSize, DefaultBlockSize, args);
    }

    public unsafe void Crop2D(IGpuBuffer input, IGpuBuffer output, int batch, int channels, int inH, int inW, int outH, int outW, int offsetH, int offsetW)
    {
        if (!_kernelCache.TryGetValue("crop_2d", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: crop_2d");
        using var _ = PushContext();
        IntPtr inPtr = input.Handle, outPtr = output.Handle;
        void** args = stackalloc void*[10];
        args[0] = &inPtr; args[1] = &outPtr; args[2] = &batch; args[3] = &channels;
        args[4] = &inH; args[5] = &inW; args[6] = &outH; args[7] = &outW; args[8] = &offsetH; args[9] = &offsetW;
        uint total = (uint)(batch * channels * outH * outW);
        LaunchKernel(kernel, (total + DefaultBlockSize - 1) / DefaultBlockSize, DefaultBlockSize, args);
    }

    public unsafe void Crop2DBackward(IGpuBuffer gradOutput, IGpuBuffer gradInput, int batch, int channels, int inH, int inW, int outH, int outW, int offsetH, int offsetW)
    {
        using var _ = PushContext();
        IntPtr goPtr = gradOutput.Handle, giPtr = gradInput.Handle;
        void** args = stackalloc void*[10];
        args[0] = &goPtr; args[1] = &giPtr; args[2] = &batch; args[3] = &channels;
        args[4] = &inH; args[5] = &inW; args[6] = &outH; args[7] = &outW; args[8] = &offsetH; args[9] = &offsetW;
        uint total = (uint)(batch * channels * outH * outW);

        // Issue #382: original uses atomicAdd defensively (mapping is one-to-one
        // per launch). Deterministic variant uses direct +=.
        string kernelName = GpuDeterminism.IsActive
            ? "crop_2d_backward_deterministic"
            : "crop_2d_backward";
        if (!_kernelCache.TryGetValue(kernelName, out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: " + kernelName);
        LaunchKernel(kernel, (total + DefaultBlockSize - 1) / DefaultBlockSize, DefaultBlockSize, args);
    }

    public unsafe void EyeKernel(IGpuBuffer output, int n)
    {
        if (!_kernelCache.TryGetValue("eye_kernel", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: eye_kernel");
        using var _ = PushContext();
        IntPtr outPtr = output.Handle;
        void** args = stackalloc void*[2];
        args[0] = &outPtr; args[1] = &n;
        uint total = (uint)(n * n);
        LaunchKernel(kernel, (total + DefaultBlockSize - 1) / DefaultBlockSize, DefaultBlockSize, args);
    }

    public unsafe void LinspaceKernel(IGpuBuffer output, float start, float step, int size)
    {
        if (!_kernelCache.TryGetValue("linspace_kernel", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: linspace_kernel");
        using var _ = PushContext();
        IntPtr outPtr = output.Handle;
        void** args = stackalloc void*[4];
        args[0] = &outPtr; args[1] = &start; args[2] = &step; args[3] = &size;
        LaunchKernel(kernel, (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize), DefaultBlockSize, args);
    }

    public unsafe void OneHotKernel(IGpuBuffer indices, IGpuBuffer output, int batchSize, int numClasses)
    {
        if (!_kernelCache.TryGetValue("one_hot_kernel", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: one_hot_kernel");
        using var _ = PushContext();
        IntPtr idxPtr = indices.Handle, outPtr = output.Handle;
        void** args = stackalloc void*[4];
        args[0] = &idxPtr; args[1] = &outPtr; args[2] = &batchSize; args[3] = &numClasses;
        uint total = (uint)(batchSize * numClasses);
        LaunchKernel(kernel, (total + DefaultBlockSize - 1) / DefaultBlockSize, DefaultBlockSize, args);
    }

    public unsafe void DiagKernel(IGpuBuffer input, IGpuBuffer output, int n)
    {
        if (!_kernelCache.TryGetValue("diag_kernel", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: diag_kernel");
        using var _ = PushContext();
        IntPtr inPtr = input.Handle, outPtr = output.Handle;
        void** args = stackalloc void*[3];
        args[0] = &inPtr; args[1] = &outPtr; args[2] = &n;
        uint total = (uint)(n * n);
        LaunchKernel(kernel, (total + DefaultBlockSize - 1) / DefaultBlockSize, DefaultBlockSize, args);
    }

    public unsafe void ExtractDiagKernel(IGpuBuffer input, IGpuBuffer output, int n, int cols)
    {
        if (!_kernelCache.TryGetValue("extract_diag_kernel", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: extract_diag_kernel");
        using var _ = PushContext();
        IntPtr inPtr = input.Handle, outPtr = output.Handle;
        void** args = stackalloc void*[4];
        args[0] = &inPtr; args[1] = &outPtr; args[2] = &n; args[3] = &cols;
        LaunchKernel(kernel, (uint)((n + DefaultBlockSize - 1) / DefaultBlockSize), DefaultBlockSize, args);
    }

    public unsafe void TriangularMask(IGpuBuffer output, int rows, int cols, int diagonal, float maskValue)
    {
        if (!_kernelCache.TryGetValue("triangular_mask", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: triangular_mask");
        using var _ = PushContext();
        IntPtr outPtr = output.Handle;
        void** args = stackalloc void*[5];
        args[0] = &outPtr; args[1] = &rows; args[2] = &cols; args[3] = &diagonal; args[4] = &maskValue;
        uint total = (uint)(rows * cols);
        LaunchKernel(kernel, (total + DefaultBlockSize - 1) / DefaultBlockSize, DefaultBlockSize, args);
    }

    public unsafe void MaskedFillKernel(IGpuBuffer input, IGpuBuffer mask, IGpuBuffer output, float fillValue, int size)
    {
        if (!_kernelCache.TryGetValue("masked_fill_kernel", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: masked_fill_kernel");
        using var _ = PushContext();
        IntPtr inPtr = input.Handle, maskPtr = mask.Handle, outPtr = output.Handle;
        void** args = stackalloc void*[5];
        args[0] = &inPtr; args[1] = &maskPtr; args[2] = &outPtr; args[3] = &fillValue; args[4] = &size;
        LaunchKernel(kernel, (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize), DefaultBlockSize, args);
    }

    public unsafe void IndexSelect(IGpuBuffer input, IGpuBuffer indices, IGpuBuffer output, int numIndices, int innerSize)
    {
        if (!_kernelCache.TryGetValue("index_select", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: index_select");
        using var _ = PushContext();
        IntPtr inPtr = input.Handle, idxPtr = indices.Handle, outPtr = output.Handle;
        void** args = stackalloc void*[5];
        args[0] = &inPtr; args[1] = &idxPtr; args[2] = &outPtr; args[3] = &numIndices; args[4] = &innerSize;
        uint total = (uint)(numIndices * innerSize);
        LaunchKernel(kernel, (total + DefaultBlockSize - 1) / DefaultBlockSize, DefaultBlockSize, args);
    }

    // --- Loss Forward ---
    public void CrossEntropyLoss(IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer loss, int batchSize, int numClasses) { LaunchLoss("cross_entropy_loss", predictions, targets, loss, batchSize, numClasses); }
    public void MseLoss(IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer loss, int batchSize, int numFeatures) { LaunchLoss("mse_loss", predictions, targets, loss, batchSize, numFeatures); }

    private unsafe void LaunchLoss(string kernelName, IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer loss, int batchSize, int dim)
    {
        if (!_kernelCache.TryGetValue(kernelName, out var kernel))
            throw new InvalidOperationException($"CUDA kernel not found: {kernelName}");
        using var _ = PushContext();
        IntPtr pPtr = predictions.Handle, tPtr = targets.Handle, lPtr = loss.Handle;
        void** args = stackalloc void*[5];
        args[0] = &pPtr; args[1] = &tPtr; args[2] = &lPtr; args[3] = &batchSize; args[4] = &dim;
        LaunchKernel(kernel, (uint)((batchSize + DefaultBlockSize - 1) / DefaultBlockSize), DefaultBlockSize, args);
    }

    public void BceLoss(IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer loss, int size) => LaunchFusedBinary("bce_loss", predictions, targets, loss, size);

    public unsafe void DropoutMask(IGpuBuffer mask, int size, float keepProb, ulong seed)
    {
        if (!_kernelCache.TryGetValue("dropout_mask", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: dropout_mask");
        using var _ = PushContext();
        IntPtr maskPtr = mask.Handle;
        void** args = stackalloc void*[4];
        args[0] = &maskPtr; args[1] = &size; args[2] = &keepProb; args[3] = &seed;
        LaunchKernel(kernel, (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize), DefaultBlockSize, args);
    }

    public unsafe void GaussianNoise(IGpuBuffer output, int size, float mean, float stdDev, ulong seed)
    {
        if (!_kernelCache.TryGetValue("gaussian_noise", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: gaussian_noise");
        using var _ = PushContext();
        IntPtr outPtr = output.Handle;
        void** args = stackalloc void*[5];
        args[0] = &outPtr; args[1] = &size; args[2] = &mean; args[3] = &stdDev; args[4] = &seed;
        LaunchKernel(kernel, (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize), DefaultBlockSize, args);
    }

    // --- Softmax Variants + Distance ---
    public void LogSoftmax(IGpuBuffer input, IGpuBuffer output, int outerSize, int innerSize) => LaunchFusedAxis("log_softmax", input, output, outerSize, innerSize);

    public unsafe void GumbelSoftmax(IGpuBuffer logits, IGpuBuffer output, int outerSize, int innerSize, float temperature, ulong seed)
    {
        if (!_kernelCache.TryGetValue("gumbel_softmax", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: gumbel_softmax");
        using var _ = PushContext();
        IntPtr inPtr = logits.Handle, outPtr = output.Handle;
        void** args = stackalloc void*[6];
        args[0] = &inPtr; args[1] = &outPtr; args[2] = &outerSize; args[3] = &innerSize; args[4] = &temperature; args[5] = &seed;
        LaunchKernel(kernel, (uint)((outerSize + DefaultBlockSize - 1) / DefaultBlockSize), DefaultBlockSize, args);
    }

    public void Sparsemax(IGpuBuffer input, IGpuBuffer output, int outerSize, int innerSize) => LaunchFusedAxis("sparsemax", input, output, outerSize, innerSize);
    public void TaylorSoftmax(IGpuBuffer input, IGpuBuffer output, int outerSize, int innerSize) => LaunchFusedAxis("taylor_softmax", input, output, outerSize, innerSize);
    public void SphericalSoftmax(IGpuBuffer input, IGpuBuffer output, int outerSize, int innerSize) => LaunchFusedAxis("spherical_softmax", input, output, outerSize, innerSize);

    public unsafe void BatchDotProduct(IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, int batchSize, int dim)
    {
        if (!_kernelCache.TryGetValue("batch_dot_product", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: batch_dot_product");
        using var _ = PushContext();
        IntPtr aPtr = a.Handle, bPtr = b.Handle, outPtr = output.Handle;
        void** args = stackalloc void*[5];
        args[0] = &aPtr; args[1] = &bPtr; args[2] = &outPtr; args[3] = &batchSize; args[4] = &dim;
        LaunchKernel(kernel, (uint)((batchSize + DefaultBlockSize - 1) / DefaultBlockSize), DefaultBlockSize, args);
    }

    public unsafe void OuterProduct(IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, int M, int N)
    {
        if (!_kernelCache.TryGetValue("outer_product", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: outer_product");
        using var _ = PushContext();
        IntPtr aPtr = a.Handle, bPtr = b.Handle, outPtr = output.Handle;
        void** args = stackalloc void*[5];
        args[0] = &aPtr; args[1] = &bPtr; args[2] = &outPtr; args[3] = &M; args[4] = &N;
        uint total = (uint)(M * N);
        LaunchKernel(kernel, (total + DefaultBlockSize - 1) / DefaultBlockSize, DefaultBlockSize, args);
    }

    public unsafe void BatchOuterProduct(IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, int batchSize, int M, int N)
    {
        if (!_kernelCache.TryGetValue("batch_outer_product", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: batch_outer_product");
        using var _ = PushContext();
        IntPtr aPtr = a.Handle, bPtr = b.Handle, outPtr = output.Handle;
        void** args = stackalloc void*[6];
        args[0] = &aPtr; args[1] = &bPtr; args[2] = &outPtr; args[3] = &batchSize; args[4] = &M; args[5] = &N;
        uint total = (uint)(batchSize * M * N);
        LaunchKernel(kernel, (total + DefaultBlockSize - 1) / DefaultBlockSize, DefaultBlockSize, args);
    }

    public unsafe void CosineSimilarity(IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, int batchSize, int dim)
    {
        if (!_kernelCache.TryGetValue("cosine_similarity", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: cosine_similarity");
        using var _ = PushContext();
        IntPtr aPtr = a.Handle, bPtr = b.Handle, outPtr = output.Handle;
        void** args = stackalloc void*[5];
        args[0] = &aPtr; args[1] = &bPtr; args[2] = &outPtr; args[3] = &batchSize; args[4] = &dim;
        LaunchKernel(kernel, (uint)((batchSize + DefaultBlockSize - 1) / DefaultBlockSize), DefaultBlockSize, args);
    }

    public unsafe void PairwiseDistance(IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, int M, int N, int dim)
    {
        if (!_kernelCache.TryGetValue("pairwise_distance", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: pairwise_distance");
        using var _ = PushContext();
        IntPtr aPtr = a.Handle, bPtr = b.Handle, outPtr = output.Handle;
        void** args = stackalloc void*[6];
        args[0] = &aPtr; args[1] = &bPtr; args[2] = &outPtr; args[3] = &M; args[4] = &N; args[5] = &dim;
        uint total = (uint)(M * N);
        LaunchKernel(kernel, (total + DefaultBlockSize - 1) / DefaultBlockSize, DefaultBlockSize, args);
    }

    public unsafe void PairwiseDistanceSquared(IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, int M, int N, int dim)
    {
        if (!_kernelCache.TryGetValue("pairwise_distance_squared", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: pairwise_distance_squared");
        using var _ = PushContext();
        IntPtr aPtr = a.Handle, bPtr = b.Handle, outPtr = output.Handle;
        void** args = stackalloc void*[6];
        args[0] = &aPtr; args[1] = &bPtr; args[2] = &outPtr; args[3] = &M; args[4] = &N; args[5] = &dim;
        uint total = (uint)(M * N);
        LaunchKernel(kernel, (total + DefaultBlockSize - 1) / DefaultBlockSize, DefaultBlockSize, args);
    }

    #endregion

    
    public unsafe void DotProduct(IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, int size)
    {
        using var _ = PushContext();
        IntPtr ap=a.Handle, bp=b.Handle, op=output.Handle;
        void** args = stackalloc void*[4];
        args[0]=&ap; args[1]=&bp; args[2]=&op; args[3]=&size;

        if (GpuDeterminism.IsActive)
        {
            // Issue #382: single-block strided reduce, no inter-block atomic combine.
            if (!_kernelCache.TryGetValue("dot_product_deterministic", out var kernelD))
                throw new InvalidOperationException("CUDA kernel not found: dot_product_deterministic");
            // Pre-zero output: single-block kernel does direct write.
            Fill(output, 0f, 1);
            LaunchKernel(kernelD, 1, DefaultBlockSize, args);
            return;
        }

        if (!_kernelCache.TryGetValue("dot_product", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: dot_product");
        LaunchKernel(kernel, (uint)((size+DefaultBlockSize-1)/DefaultBlockSize), DefaultBlockSize, args);
    }

    public unsafe void StridedDotProduct(IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, int size, int strideA, int strideB, int count)
    {
        if (!_kernelCache.TryGetValue("strided_dot_product", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: strided_dot_product");
        using var _ = PushContext();
        IntPtr ap=a.Handle, bp=b.Handle, op=output.Handle;
        void** args = stackalloc void*[7];
        args[0]=&ap; args[1]=&bp; args[2]=&op; args[3]=&size; args[4]=&strideA; args[5]=&strideB; args[6]=&count;
        LaunchKernel(kernel, (uint)((count+DefaultBlockSize-1)/DefaultBlockSize), DefaultBlockSize, args);
    }

    public unsafe void BatchedDotProduct(IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, int batchSize, int dim)
    {
        if (!_kernelCache.TryGetValue("batched_dot_product", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: batched_dot_product");
        using var _ = PushContext();
        IntPtr ap=a.Handle, bp=b.Handle, op=output.Handle;
        void** args = stackalloc void*[5];
        args[0]=&ap; args[1]=&bp; args[2]=&op; args[3]=&batchSize; args[4]=&dim;
        LaunchKernel(kernel, (uint)((batchSize+DefaultBlockSize-1)/DefaultBlockSize), DefaultBlockSize, args);
    }

    ~CudaBackend()
    {
        // Finalizer: never touch managed members or the CUDA driver — Dispose(bool)'s
        // !disposing/IsRuntimeTearingDown/ProcessExiting guard makes this a no-op (the OS
        // reclaims the context + device memory at process exit). The try/catch is
        // belt-and-suspenders: a finalizer must NEVER throw, even if that contract changes
        // or a future undisposed-backend path reaches managed pools that are already
        // finalized during GC (ObjectDisposedException).
        try { Dispose(disposing: false); } catch { }
    }

    internal sealed class CudaGpuBuffer : IGpuBuffer, IPoolableGpuBuffer
    {
        private IntPtr _context;
        private IntPtr _devicePtr;
        private readonly Action<CudaGpuBuffer>? _returnToPool;
        private readonly IntPtr _asyncFreeStream; // non-zero ⇒ free via cuMemFreeAsync on this stream (#558 layer 6)
        private int _poolState;

        public int Size { get; }
        public long SizeInBytes { get; }
        public IntPtr Handle => _devicePtr;
        internal bool IsAsyncFreed => _asyncFreeStream != IntPtr.Zero;

        public CudaGpuBuffer(IntPtr context, IntPtr devicePtr, int size, Action<CudaGpuBuffer>? returnToPool = null,
            IntPtr asyncFreeStream = default)
        {
            _context = context;
            _devicePtr = devicePtr;
            Size = size;
            SizeInBytes = (long)size * sizeof(float);
            _returnToPool = returnToPool;
            _asyncFreeStream = asyncFreeStream;
        }

        public void MarkRented()
        {
            Interlocked.Exchange(ref _poolState, 0);
        }

        public void Release()
        {
            if (Interlocked.Exchange(ref _poolState, 2) == 2)
                return;

            if (_devicePtr == IntPtr.Zero)
                return;

            // During CLR/AppDomain teardown the CUDA context may already be gone;
            // calling the driver here raises an uncatchable, finalizer-fatal access
            // violation. Skip the native free (the OS reclaims device memory at exit)
            // and just clear the handles.
            if (!IsRuntimeTearingDown)
            {
                // Skip the native free at process exit (the driver is gone — any native CUDA
                // call FATAL-crashes, uncatchable); otherwise free, suppressing any disposal
                // error so a finalizer can't crash.
                // Hold the lifecycle lock across the live-check + free so a concurrent engine Dispose
                // can't cuCtxDestroy this context between the check and the cuCtxPushCurrent/free.
                lock (ContextLifecycleLock)
                {
                    if (_context != IntPtr.Zero && !ProcessExiting && LiveContexts.ContainsKey(_context))
                    {
                        try
                        {
                            CuBlasNative.cuCtxPushCurrent(_context);
                            GpuMemoryTracker.OnFree(_devicePtr);
                            // Stream-ordered free (#558 layer 6): race-safe + pool-reused in stream order.
                            if (_asyncFreeStream != IntPtr.Zero)
                                CuBlasNative.cuMemFreeAsync(_devicePtr, _asyncFreeStream);
                            else
                                CuBlasNative.cuMemFree(_devicePtr);
                            CuBlasNative.cuCtxPopCurrent(out _);
                        }
                        catch
                        {
                            // Suppress disposal errors to avoid crashing finalizers.
                        }
                    }
                }
            }

            _devicePtr = IntPtr.Zero;
            _context = IntPtr.Zero;
            GC.SuppressFinalize(this);
        }

        public void Dispose()
        {
            if (_returnToPool == null)
            {
                Release();
                return;
            }

            if (Interlocked.CompareExchange(ref _poolState, 1, 0) != 0)
                return;

            try
            {
                _returnToPool(this);
            }
            catch (Exception)
            {
                // If returning to pool fails for any reason, release directly
                Release();
            }
        }

        ~CudaGpuBuffer()
        {
            // NEVER call the CUDA driver on the GC finalizer thread (races with op threads →
            // 0xC0000005). Defer the device free to a real op thread via the pending-free queue.
            var ptr = _devicePtr;
            if (ptr != IntPtr.Zero && !IsRuntimeTearingDown && !ProcessExiting)
                PendingFinalizerFrees.Enqueue((ptr, _context, _asyncFreeStream));
            _devicePtr = IntPtr.Zero;
            _context = IntPtr.Zero;
        }
    }

    internal sealed class CudaGpuByteBuffer : IGpuBuffer
    {
        private IntPtr _context;
        private IntPtr _devicePtr;
        private readonly IntPtr _asyncFreeStream; // #558 layer 6: free via cuMemFreeAsync when non-zero

        public int Size { get; }
        public long SizeInBytes { get; }
        public IntPtr Handle => _devicePtr;
        internal bool IsAsyncFreed => _asyncFreeStream != IntPtr.Zero;

        public CudaGpuByteBuffer(IntPtr context, IntPtr devicePtr, int size, IntPtr asyncFreeStream = default)
        {
            _context = context;
            _devicePtr = devicePtr;
            Size = size;
            SizeInBytes = size;
            _asyncFreeStream = asyncFreeStream;
        }

        public void Dispose()
        {
            if (_devicePtr == IntPtr.Zero)
                return;

            // See CudaGpuBuffer.Release: skip native CUDA calls during teardown to
            // avoid a finalizer-fatal access violation on an already-destroyed context.
            if (!IsRuntimeTearingDown)
            {
                // Skip the native free at process exit (the driver is gone — any native CUDA
                // call FATAL-crashes, uncatchable); otherwise free, suppressing any disposal
                // error so a finalizer can't crash.
                // Hold the lifecycle lock across the live-check + free so a concurrent engine Dispose
                // can't cuCtxDestroy this context between the check and the cuCtxPushCurrent/free.
                lock (ContextLifecycleLock)
                {
                    if (_context != IntPtr.Zero && !ProcessExiting && LiveContexts.ContainsKey(_context))
                    {
                        try
                        {
                            CuBlasNative.cuCtxPushCurrent(_context);
                            GpuMemoryTracker.OnFree(_devicePtr);
                            // Stream-ordered free (#558 layer 6): race-safe + pool-reused in stream order.
                            if (_asyncFreeStream != IntPtr.Zero)
                                CuBlasNative.cuMemFreeAsync(_devicePtr, _asyncFreeStream);
                            else
                                CuBlasNative.cuMemFree(_devicePtr);
                            CuBlasNative.cuCtxPopCurrent(out _);
                        }
                        catch
                        {
                            // Suppress disposal errors to avoid crashing finalizers.
                        }
                    }
                }
            }

            _devicePtr = IntPtr.Zero;
            _context = IntPtr.Zero;
            GC.SuppressFinalize(this);
        }

        ~CudaGpuByteBuffer()
        {
            // NEVER call the CUDA driver on the GC finalizer thread — defer the device free to a real
            // op thread via the pending-free queue (see CudaBackend.PendingFinalizerFrees).
            var ptr = _devicePtr;
            if (ptr != IntPtr.Zero && !IsRuntimeTearingDown && !ProcessExiting)
                PendingFinalizerFrees.Enqueue((ptr, _context, _asyncFreeStream));
            _devicePtr = IntPtr.Zero;
            _context = IntPtr.Zero;
        }
    }
}
