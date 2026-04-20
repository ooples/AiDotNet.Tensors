// Copyright (c) AiDotNet. All rights reserved.
// HIP backend for AMD GPU with real MFMA (Matrix Fused Multiply-Add) support.
// Target: 25,000+ GFLOPS on MI200, 15,000+ GFLOPS on RX 7900.

using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Threading;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.HIP.Kernels;
using AiDotNet.Tensors.Engines.DirectGpu.Sparsity;
using AiDotNet.Tensors.Engines.Gpu;
using FusedActivationType = AiDotNet.Tensors.Engines.FusedActivationType;

namespace AiDotNet.Tensors.Engines.DirectGpu.HIP;

/// <summary>
/// HIP backend for AMD GPUs with real MFMA (Matrix Core) acceleration.
/// Automatically selects the optimal kernel based on GPU architecture:
/// - CDNA (MI100/200/300): Full MFMA with wave64
/// - RDNA3 (RX 7000): WMMA with wave32
/// - RDNA2 (RX 6000): Optimized scalar with wave32
/// </summary>
/// <remarks>
/// <para><b>Performance Targets:</b></para>
/// <list type="bullet">
/// <item>MI200: 25,000+ GFLOPS (full MFMA)</item>
/// <item>MI100: 20,000+ GFLOPS (MFMA)</item>
/// <item>RX 7900 XTX: 15,000+ GFLOPS (WMMA)</item>
/// <item>RX 6800 XT: 8,000+ GFLOPS (optimized scalar)</item>
/// </list>
/// </remarks>
public sealed partial class HipBackend : IAsyncGpuBackend
{
    private IntPtr _stream;
    private HipStream? _defaultStream;
    private IntPtr _mfmaModule;
    private IntPtr _mfmaGemmF32;
    private IntPtr _mfmaGemmF16;
    private IntPtr _scalarGemmF32;
    private IntPtr _rdnaGemmWave32;
    private readonly Dictionary<string, IntPtr> _kernelCache;
    private AmdGpuArchitecture _architecture;
    private bool _disposed;
    private const int MaxPooledBufferElements = 16_777_216;
    private const int MaxPooledBuffersPerSize = 4;
    private readonly GpuBufferPool<HipGpuBuffer> _bufferPool =
        new GpuBufferPool<HipGpuBuffer>(MaxPooledBuffersPerSize, MaxPooledBufferElements);
    private readonly HipPinnedBufferPool _pinnedPool = new();
    private HipDeviceProperties _deviceProps;
    private bool _supportsCooperativeLaunch;

    // Additional kernel modules
    private IntPtr _activationModule;
    private IntPtr _neuralNetModule;
    private IntPtr _convolutionModule;
    private IntPtr _fusedConvolutionModule;
    private IntPtr _poolingModule;
    private IntPtr _normalizationModule;
    private IntPtr _fusedModule;
    private IntPtr _attentionModule;
    private IntPtr _fftModule;
    private IntPtr _spectralPerfModule;
    private IntPtr _sparseModule;
    private IntPtr _locallyConnectedModule;
    private IntPtr _deformableConvModule;
    private IntPtr _optimizerModule;
    private IntPtr _specializedModule;
    private IntPtr _fp16Module;
    private IntPtr _spatialTransformerModule;
    private IntPtr _lstmModule;
    private IntPtr _gruModule;
    private IntPtr _capsuleModule;
    private IntPtr _snnModule;
    private IntPtr _dotProductModule;
    private IntPtr _reductionModule2;
    private IntPtr _broadcastModule;
    private IntPtr _gatedModule;
    private IntPtr _shapeModule;
    private IntPtr _lossModule;
    private IntPtr _softmaxVarModule;
    private IntPtr _fusedLinearModule;
    private IntPtr _iouModule;
    private IntPtr _complexModule;
    private IntPtr _parity210Module;
    private IntPtr _linalgModule;
    private IntPtr _hipblasHandle;
    private bool _hipblasAvailable;

    private const int DefaultBlockSize = 256;
    private const int MaxRnnBlockSize = 1024;
    private const string GemmVendorImplEnvVar = "AIDOTNET_GPU_GEMM_IMPL";
    private const string GemmVendorThresholdEnvVar = "AIDOTNET_GPU_GEMM_VENDOR_THRESHOLD";
    private const long DefaultVendorGemmThreshold = 128L * 128L * 128L;

    public bool IsAvailable { get; }
    public string BackendName => $"HIP ({GetKernelTypeName()})";
    public TensorDevice DeviceType => TensorDevice.HIP;
    public string DeviceName { get; }

    private string GetKernelTypeName()
    {
        return _architecture switch
        {
            AmdGpuArchitecture.MI100 or AmdGpuArchitecture.MI200 or AmdGpuArchitecture.MI300 => "MFMA",
            AmdGpuArchitecture.RDNA3 => "RDNA3",
            AmdGpuArchitecture.RDNA2 => "RDNA2",
            AmdGpuArchitecture.RDNA => "RDNA",
            _ => "Scalar"
        };
    }
    public string DeviceVendor => "AMD";
    public int ComputeUnits { get; }
    public long GlobalMemoryBytes { get; }
    public long LocalMemoryBytes { get; }
    public double TheoreticalGflops { get; }

    private void ReturnBufferToPool(HipGpuBuffer buffer)
    {
        if (_disposed)
        {
            buffer.Release();
            return;
        }

        _bufferPool.Return(buffer);
    }

    /// <summary>
    /// Gets the detected AMD GPU architecture.
    /// </summary>
    public AmdGpuArchitecture Architecture => _architecture;

    /// <summary>
    /// Gets whether HIP is available on this system.
    /// </summary>
    public static bool IsHipAvailable => HipNativeBindings.IsAvailable;

    /// <summary>
    /// Enables or disables diagnostic logging for HIP operations.
    /// </summary>
    public static bool EnableDiagnostics
    {
        get => HipNativeBindings.EnableDiagnostics;
        set => HipNativeBindings.EnableDiagnostics = value;
    }

    /// <summary>
    /// Gets whether elementwise operations are GPU-accelerated.
    /// Returns true for HipBackend as all elementwise operations use GPU kernels.
    /// </summary>
    public bool SupportsElementwiseGpu => true;

    #region IAsyncGpuBackend Properties

    /// <inheritdoc/>
    public bool SupportsMultiStream => true;

    /// <inheritdoc/>
    public bool SupportsEvents => true;

    /// <inheritdoc/>
    public bool SupportsAsyncTransfer => true;

    /// <inheritdoc/>
    public bool SupportsGraphCapture => false; // HIP graph capture is more limited

    /// <inheritdoc/>
    public int MaxConcurrentStreams => 16;

    /// <inheritdoc/>
    public IGpuStream DefaultStream
    {
        get
        {
            if (_defaultStream == null && IsAvailable)
            {
                var newStream = new HipStream(this, _stream, GpuStreamType.Default, true);
                System.Threading.Interlocked.CompareExchange(ref _defaultStream, newStream, null);
                // If another thread won the race, dispose our unused stream
                if (_defaultStream != newStream)
                {
                    newStream.Dispose();
                }
            }
            return _defaultStream ?? throw new InvalidOperationException("Backend not available");
        }
    }

    #endregion

    public HipBackend() : this(0)
    {
    }

    public HipBackend(int deviceIndex)
    {
        _kernelCache = new Dictionary<string, IntPtr>();

        if (!HipNativeBindings.IsAvailable)
        {
            IsAvailable = false;
            DeviceName = "None";
            return;
        }

        try
        {
            // Initialize HIP and get device info
            var result = HipNativeBindings.hipSetDevice(deviceIndex);
            if (result != HipError.Success)
            {
                IsAvailable = false;
                DeviceName = "None";
                return;
            }

            // Get device properties
            _deviceProps = new HipDeviceProperties();
            result = HipNativeBindings.hipGetDeviceProperties(ref _deviceProps, deviceIndex);
            if (result != HipError.Success)
            {
                IsAvailable = false;
                DeviceName = "None";
                return;
            }

            DeviceName = _deviceProps.Name?.Trim() ?? "Unknown AMD GPU";
            ComputeUnits = _deviceProps.MultiProcessorCount;
            GlobalMemoryBytes = (long)(ulong)_deviceProps.TotalGlobalMem;
            LocalMemoryBytes = (long)(ulong)_deviceProps.SharedMemPerBlock;

            // Detect architecture from GCN arch name
            _architecture = DetectArchitecture(_deviceProps.GcnArchName, 0);

            // Check cooperative launch support
            int coopLaunchSupport = 0;
            result = HipNativeBindings.hipDeviceGetAttribute(
                ref coopLaunchSupport,
                HipDeviceAttribute.CooperativeLaunch,
                deviceIndex);
            _supportsCooperativeLaunch = result == HipError.Success && coopLaunchSupport != 0;

            // Create compute stream
            result = HipNativeBindings.hipStreamCreate(ref _stream);
            if (result != HipError.Success)
            {
                IsAvailable = false;
                return;
            }

            TryInitializeHipBlas();

            // Compile MFMA kernels
            CompileKernels();

            // Verify at least one GEMM kernel compiled successfully
            bool hasGemmKernel = _kernelCache.ContainsKey("scalar_gemm_f32") ||
                                 _kernelCache.ContainsKey("mfma_gemm_f32") ||
                                 _kernelCache.ContainsKey("rdna_gemm_wave32");

            if (!hasGemmKernel)
            {
                Console.WriteLine("[HipBackend] No GEMM kernels compiled - backend not available");
                IsAvailable = false;
                return;
            }

            IsAvailable = true;
        }
        catch (Exception ex)
        {
            System.Diagnostics.Debug.WriteLine($"HipBackend initialization failed: {ex.Message}");
            IsAvailable = false;
            DeviceName = "None";
        }
    }

    private AmdGpuArchitecture DetectArchitecture(string gcnArchName, int gcnArch)
    {
        // Parse GCN architecture name (e.g., "gfx90a", "gfx1100", "gfx1030")
        if (string.IsNullOrEmpty(gcnArchName))
        {
            return gcnArch switch
            {
                >= 1100 => AmdGpuArchitecture.RDNA3,
                >= 1030 => AmdGpuArchitecture.RDNA2,
                >= 1010 => AmdGpuArchitecture.RDNA,
                >= 940 => AmdGpuArchitecture.MI300,
                >= 908 => AmdGpuArchitecture.MI100,
                _ => AmdGpuArchitecture.GCN
            };
        }

        string archLower = gcnArchName.ToLowerInvariant();
        if (archLower.Contains("gfx942") || archLower.Contains("gfx941") || archLower.Contains("gfx940"))
            return AmdGpuArchitecture.MI300;
        if (archLower.Contains("gfx90a"))
            return AmdGpuArchitecture.MI200;
        if (archLower.Contains("gfx908"))
            return AmdGpuArchitecture.MI100;
        if (archLower.Contains("gfx1100") || archLower.Contains("gfx1101") || archLower.Contains("gfx1102"))
            return AmdGpuArchitecture.RDNA3;
        if (archLower.Contains("gfx1030") || archLower.Contains("gfx1031") || archLower.Contains("gfx1032"))
            return AmdGpuArchitecture.RDNA2;
        if (archLower.Contains("gfx1010") || archLower.Contains("gfx1011") || archLower.Contains("gfx1012"))
            return AmdGpuArchitecture.RDNA;

        return AmdGpuArchitecture.GCN;
    }

    private void TryInitializeHipBlas()
    {
        if (!HipBlasNative.IsAvailable)
        {
            return;
        }

        try
        {
            var status = HipBlasNative.hipblasCreate(ref _hipblasHandle); // lgtm[cs/call-to-unmanaged-code] HIP BLAS uses native bindings.
            if (status != HipBlasNative.HipBlasStatus.Success)
            {
                _hipblasHandle = IntPtr.Zero;
                return;
            }

            status = HipBlasNative.hipblasSetStream(_hipblasHandle, _stream); // lgtm[cs/call-to-unmanaged-code] HIP BLAS uses native bindings.
            if (status != HipBlasNative.HipBlasStatus.Success)
            {
                HipBlasNative.hipblasDestroy(_hipblasHandle); // lgtm[cs/call-to-unmanaged-code] HIP BLAS uses native bindings.
                _hipblasHandle = IntPtr.Zero;
                return;
            }

            _hipblasAvailable = true;
        }
        catch (DllNotFoundException)
        {
            _hipblasHandle = IntPtr.Zero;
            _hipblasAvailable = false;
        }
        catch (EntryPointNotFoundException)
        {
            _hipblasHandle = IntPtr.Zero;
            _hipblasAvailable = false;
        }
        catch (BadImageFormatException)
        {
            _hipblasHandle = IntPtr.Zero;
            _hipblasAvailable = false;
        }
        catch (SEHException)
        {
            _hipblasHandle = IntPtr.Zero;
            _hipblasAvailable = false;
        }
    }

    private static long GetEnvLong(string name, long defaultValue)
    {
        var value = Environment.GetEnvironmentVariable(name);
        return long.TryParse(value, out var parsed) && parsed > 0 ? parsed : defaultValue;
    }

    private static bool ShouldUseVendorGemm(int m, int n, int k)
    {
        string? mode = Environment.GetEnvironmentVariable(GemmVendorImplEnvVar);
        if (IsVendorGemmDisabled(mode))
        {
            return false;
        }

        if (IsVendorGemmForced(mode))
        {
            return true;
        }

        if (m <= 0 || n <= 0 || k <= 0)
        {
            return false;
        }

        ulong work = (ulong)m * (ulong)n * (ulong)k;
        ulong threshold = (ulong)GetEnvLong(GemmVendorThresholdEnvVar, DefaultVendorGemmThreshold);
        return work >= threshold;
    }

    private static bool IsVendorGemmForced(string? mode)
    {
        return string.Equals(mode, "vendor", StringComparison.OrdinalIgnoreCase) ||
               string.Equals(mode, "hipblas", StringComparison.OrdinalIgnoreCase);
    }

    private static bool IsVendorGemmDisabled(string? mode)
    {
        return string.Equals(mode, "builtin", StringComparison.OrdinalIgnoreCase) ||
               string.Equals(mode, "internal", StringComparison.OrdinalIgnoreCase) ||
               string.Equals(mode, "kernel", StringComparison.OrdinalIgnoreCase) ||
               string.Equals(mode, "custom", StringComparison.OrdinalIgnoreCase);
    }

    private void CompileKernels()
    {
        // Get kernel source
        string source = HipMfmaKernel.GetSource();
        string compileFlags = HipMfmaKernel.GetCompileFlags(_architecture);

        Console.WriteLine($"[HipBackend] Compiling kernels for {_architecture} with flags: {compileFlags}");

        try
        {
            // Compile MFMA/GEMM module
            CompileKernelModule(source, "mfma_gemm", ref _mfmaModule, new[]
            {
                "mfma_gemm_f32", "mfma_gemm_f16", "scalar_gemm_f32", "rdna_gemm_wave32"
            });

            // Store GEMM kernel handles for quick lookup
            if (_kernelCache.TryGetValue("mfma_gemm_f32", out var mfmaF32))
                _mfmaGemmF32 = mfmaF32;
            if (_kernelCache.TryGetValue("mfma_gemm_f16", out var mfmaF16))
                _mfmaGemmF16 = mfmaF16;
            if (_kernelCache.TryGetValue("scalar_gemm_f32", out var scalarF32))
                _scalarGemmF32 = scalarF32;
            if (_kernelCache.TryGetValue("rdna_gemm_wave32", out var rdnaWave32))
                _rdnaGemmWave32 = rdnaWave32;

            // Compile Activation kernels
            CompileKernelModule(HipActivationKernels.GetSource(), "activation", ref _activationModule,
                HipActivationKernels.GetKernelNames());

            // Compile Neural Net kernels
            CompileKernelModule(HipNeuralNetKernels.GetSource(), "neural_net", ref _neuralNetModule,
                HipNeuralNetKernels.GetKernelNames());

            // Compile Convolution kernels
            CompileKernelModule(HipConvolutionKernels.GetSource(), "convolution", ref _convolutionModule,
                HipConvolutionKernels.GetKernelNames());

            // Compile Fused Convolution kernels (Conv2D + Bias/BatchNorm + Activation)
            CompileKernelModule(HipFusedConvolutionKernels.GetSource(), "fused_convolution", ref _fusedConvolutionModule,
                HipFusedConvolutionKernels.GetKernelNames());

            // Compile Pooling kernels
            CompileKernelModule(HipPoolingKernels.GetSource(), "pooling", ref _poolingModule,
                HipPoolingKernels.GetKernelNames());

            // Compile Normalization kernels
            CompileKernelModule(HipNormalizationKernels.GetSource(), "normalization", ref _normalizationModule,
                HipNormalizationKernels.GetKernelNames());

            // Compile Fused kernels (GEMM + Bias + Activation in single pass)
            CompileKernelModule(HipFusedKernels.GetSource(), "fused", ref _fusedModule,
                HipFusedKernels.GetKernelNames());

            // Compile Attention kernels (FlashAttention, GQA, ScaledDotProduct)
            CompileKernelModule(HipAttentionKernels.GetSource(), "attention", ref _attentionModule,
                HipAttentionKernels.GetKernelNames());

            // Compile FFT kernels (Cooley-Tukey radix-2 FFT, STFT, Mel spectrogram)
            CompileKernelModule(HipFFTKernels.GetSource(), "fft", ref _fftModule,
                HipFFTKernels.GetKernelNames());
            CompileKernelModule(Kernels.HipSpectralPerfKernels.GetSource(), "spectral_perf", ref _spectralPerfModule,
                Kernels.HipSpectralPerfKernels.GetKernelNames());

            // Compile Sparse kernels (CSR SpMM, GNN message passing)
            CompileKernelModule(HipSparseKernels.GetSource(), "sparse", ref _sparseModule,
                HipSparseKernels.GetKernelNames());

            // Compile Locally Connected kernels (unique weights per spatial position)
            CompileKernelModule(HipLocallyConnectedKernels.GetSource(), "locally_connected", ref _locallyConnectedModule,
                HipLocallyConnectedKernels.GetKernelNames());

            // Compile Deformable Convolution kernels (DCNv2 with learnable offsets and masks)
            CompileKernelModule(HipDeformableConvolutionKernels.GetSource(), "deformable_conv", ref _deformableConvModule,
                HipDeformableConvolutionKernels.GetKernelNames());

            // Compile Optimizer kernels (SGD, Adam, AdamW, RMSprop, etc.)
            // Optimizer kernels need precise math for bias correction (1 - beta^t divisions)
            CompileKernelModule(HipOptimizerKernels.GetSource(), "optimizer", ref _optimizerModule,
                HipOptimizerKernels.GetKernelNames(), useFastMath: false);

            // Compile Specialized kernels (hyperbolic geometry, octonion algebra, quantum computing)
            CompileKernelModule(Kernels.HipSpecializedKernels.GetSource(), "specialized", ref _specializedModule,
                Kernels.HipSpecializedKernels.GetKernelNames());

            // Compile FP16 conversion kernels (half-precision float conversion)
            CompileKernelModule(HipFp16Kernels.GetSource(), "fp16", ref _fp16Module,
                HipFp16Kernels.GetKernelNames());

            // Compile Spatial Transformer kernels (TopK, AffineGrid, GridSample)
            CompileKernelModule(Kernels.HipSpatialTransformerKernels.GetSource(), "spatial_transformer", ref _spatialTransformerModule,
                Kernels.HipSpatialTransformerKernels.GetKernelNames());

            // Compile LSTM sequence kernels (forward/backward for BPTT training)
            CompileKernelModule(HipLstmKernels.GetSource(), "lstm", ref _lstmModule,
                HipLstmKernels.GetKernelNames());

            // Compile Fused Linear + Activation kernels
            CompileKernelModule(Kernels.HipFusedLinearKernels.GetSource(), "fused_linear", ref _fusedLinearModule,
                Kernels.HipFusedLinearKernels.GetKernelNames());

            // Compile IoU loss kernels
            CompileKernelModule(Kernels.HipIoUKernels.GetSource(), "iou", ref _iouModule,
                Kernels.HipIoUKernels.GetKernelNames());

            // Compile GRU sequence kernels (forward/backward for BPTT training)
            CompileKernelModule(HipGruKernels.GetSource(), "gru", ref _gruModule,
                HipGruKernels.GetKernelNames());

            // Compile Capsule Network kernels
            CompileKernelModule(Kernels.HipCapsuleKernels.GetSource(), "capsule", ref _capsuleModule,
                Kernels.HipCapsuleKernels.GetKernelNames());

            // Compile SNN kernels (STDP, spike traces, RBF, PRNG, 2:4 structured sparsity)
            CompileKernelModule(Kernels.HipSnnKernels.GetSource(), "snn", ref _snnModule,
                Kernels.HipSnnKernels.GetKernelNames());

            // Compile reduction, broadcast, gated activation, shape, loss, softmax variant kernels
            // Module handles stored in fields for proper Dispose cleanup
            CompileKernelModule(Kernels.HipReductionKernels.GetSource(), "reduction", ref _reductionModule2, Kernels.HipReductionKernels.GetKernelNames());
            CompileKernelModule(Kernels.HipBroadcastKernels.GetSource(), "broadcast", ref _broadcastModule, Kernels.HipBroadcastKernels.GetKernelNames());
            CompileKernelModule(Kernels.HipGatedActivationKernels.GetSource(), "gated_activation", ref _gatedModule, Kernels.HipGatedActivationKernels.GetKernelNames());
            CompileKernelModule(Kernels.HipShapeKernels.GetSource(), "shape", ref _shapeModule, Kernels.HipShapeKernels.GetKernelNames());
            CompileKernelModule(Kernels.HipLossForwardKernels.GetSource(), "loss_forward", ref _lossModule, Kernels.HipLossForwardKernels.GetKernelNames());
            CompileKernelModule(Kernels.HipSoftmaxVariantKernels.GetSource(), "softmax_variant", ref _softmaxVarModule, Kernels.HipSoftmaxVariantKernels.GetKernelNames());

            // Compile split-buffer complex kernels for native Tensor<Complex<T>> operations
            CompileKernelModule(Kernels.HipComplexKernels.GetSource(), "complex", ref _complexModule, Kernels.HipComplexKernels.GetKernelNames());

            // Parity-210 hot-path kernels. Same surface as CUDA's parity210_* set.
            try
            {
                CompileKernelModule(Kernels.HipParity210Kernels.GetSource(), "parity210",
                    ref _parity210Module, Kernels.HipParity210Kernels.GetKernelNames());
            }
            catch
            {
                _parity210Module = IntPtr.Zero;
            }

            // Linalg decomposition kernels (#211 moat #2).
            try
            {
                CompileKernelModule(Kernels.HipLinalgKernels.GetSource(), "linalg",
                    ref _linalgModule, Kernels.HipLinalgKernels.GetKernelNames());
            }
            catch
            {
                _linalgModule = IntPtr.Zero;
            }

            Console.WriteLine($"[HipBackend] Kernel compilation complete. Available kernels: {_kernelCache.Count}");
            System.Diagnostics.Debug.WriteLine($"HIP kernels compiled successfully for {_architecture}. Total: {_kernelCache.Count}");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[HipBackend] Kernel compilation EXCEPTION: {ex.GetType().Name}: {ex.Message}");
            System.Diagnostics.Debug.WriteLine($"HIP kernel compilation failed: {ex.Message}");
        }
    }

    private void CompileKernelModule(string source, string moduleName, ref IntPtr module, string[] kernelNames)
    {
        CompileKernelModule(source, moduleName, ref module, kernelNames, useFastMath: true);
    }

    private void CompileKernelModule(string source, string moduleName, ref IntPtr module, string[] kernelNames, bool useFastMath)
    {
        string compileFlags = HipMfmaKernel.GetCompileFlags(_architecture);

        IntPtr prog = IntPtr.Zero;
        var rtcResult = HipNativeBindings.hiprtcCreateProgram(
            ref prog, source, moduleName, 0, IntPtr.Zero, IntPtr.Zero);

        if (rtcResult != HipRtcResult.Success)
        {
            Console.WriteLine($"[HipBackend] Failed to create program for {moduleName}: {rtcResult}");
            return;
        }

        var options = new List<string>(compileFlags.Split(new[] { ' ' }, StringSplitOptions.RemoveEmptyEntries))
        {
            "-O3"
        };
        if (useFastMath)
        {
            options.Add("-ffast-math");
        }
        rtcResult = HipNativeBindings.hiprtcCompileProgram(prog, options.Count, options.ToArray());

        if (rtcResult != HipRtcResult.Success)
        {
            UIntPtr logSize = UIntPtr.Zero;
            HipNativeBindings.hiprtcGetProgramLogSize(prog, ref logSize);
            if ((ulong)logSize > 0)
            {
                IntPtr logPtr = Marshal.AllocHGlobal((int)(ulong)logSize);
                HipNativeBindings.hiprtcGetProgramLog(prog, logPtr);
                string log = Marshal.PtrToStringAnsi(logPtr) ?? "";
                Marshal.FreeHGlobal(logPtr);
                Console.WriteLine($"[HipBackend] Compile failed for {moduleName}: {log}");
            }
            HipNativeBindings.hiprtcDestroyProgram(ref prog);
            return;
        }

        UIntPtr codeSize = UIntPtr.Zero;
        rtcResult = HipNativeBindings.hiprtcGetCodeSize(prog, ref codeSize);
        if (rtcResult != HipRtcResult.Success || (ulong)codeSize == 0)
        {
            HipNativeBindings.hiprtcDestroyProgram(ref prog);
            return;
        }

        IntPtr code = Marshal.AllocHGlobal((int)(ulong)codeSize);
        rtcResult = HipNativeBindings.hiprtcGetCode(prog, code);
        HipNativeBindings.hiprtcDestroyProgram(ref prog);

        if (rtcResult != HipRtcResult.Success)
        {
            Marshal.FreeHGlobal(code);
            return;
        }

        var hipResult = HipNativeBindings.hipModuleLoadData(ref module, code);
        Marshal.FreeHGlobal(code);

        if (hipResult != HipError.Success)
        {
            Console.WriteLine($"[HipBackend] Failed to load module {moduleName}: {hipResult}");
            return;
        }

        // Load all kernel functions
        foreach (var kernelName in kernelNames)
        {
            IntPtr func = IntPtr.Zero;
            hipResult = HipNativeBindings.hipModuleGetFunction(ref func, module, kernelName);
            if (hipResult == HipError.Success)
                _kernelCache[kernelName] = func;
        }
    }

    private unsafe void LaunchKernel(IntPtr kernel, uint gridX, uint blockSize, IntPtr[] args, uint sharedMem = 0)
    {
        LaunchKernelOnStream(kernel, gridX, blockSize, args, _stream, sharedMem);
    }

    /// <summary>
    /// Launch kernel with stackalloc void** args (zero GCHandle allocation overhead).
    /// </summary>
    private unsafe void LaunchKernel(IntPtr kernel, uint gridX, uint blockSize, void** args, uint sharedMem = 0)
    {
        var result = HipNativeBindings.hipModuleLaunchKernel(
            kernel, gridX, 1, 1, blockSize, 1, 1,
            sharedMem, _stream, (IntPtr)args, IntPtr.Zero);
        HipNativeBindings.CheckError(result, "hipModuleLaunchKernel");
    }

    /// <summary>
    /// Launch 2D kernel with stackalloc void** args (zero GCHandle allocation overhead).
    /// </summary>
    private unsafe void LaunchKernel2D(IntPtr kernel, uint gridX, uint gridY, uint blockX, uint blockY, void** args, uint sharedMem = 0)
    {
        var result = HipNativeBindings.hipModuleLaunchKernel(
            kernel, gridX, gridY, 1, blockX, blockY, 1,
            sharedMem, _stream, (IntPtr)args, IntPtr.Zero);
        HipNativeBindings.CheckError(result, "hipModuleLaunchKernel");
    }

    private unsafe void LaunchKernelWithSharedMem(IntPtr kernel, uint gridX, uint blockSize, uint sharedMemBytes, IntPtr[] args)
    {
        LaunchKernelOnStream(kernel, gridX, blockSize, args, _stream, sharedMemBytes);
    }

    private unsafe void LaunchKernelWithSharedMem(IntPtr kernel, uint gridX, uint blockSize, uint sharedMemBytes, void** args)
    {
        var result = HipNativeBindings.hipModuleLaunchKernel(
            kernel, gridX, 1, 1, blockSize, 1, 1,
            sharedMemBytes, _stream, (IntPtr)args, IntPtr.Zero);
        HipNativeBindings.CheckError(result, "hipModuleLaunchKernel");
    }

    private unsafe void LaunchKernelOnStream(IntPtr kernel, uint gridX, uint blockSize, IntPtr[] args, IntPtr stream, uint sharedMem = 0)
    {
        fixed (IntPtr* argsPtr = args)
        {
            var result = HipNativeBindings.hipModuleLaunchKernel(
                kernel, gridX, 1, 1, blockSize, 1, 1,
                sharedMem, stream, (IntPtr)argsPtr, IntPtr.Zero);
            HipNativeBindings.CheckError(result, "hipModuleLaunchKernel");
        }
    }

    /// <summary>
    /// Launch a cooperative kernel that supports grid-level synchronization via cooperative_groups.
    /// Cooperative kernels can use grid.sync() for cross-block synchronization.
    /// Throws if device doesn't support cooperative launch or grid exceeds limits.
    /// </summary>
    private unsafe void LaunchCooperativeKernel(IntPtr kernel, uint gridX, uint blockSize, uint sharedMemBytes, IntPtr[] args)
    {
        // Check cooperative launch support
        if (!_supportsCooperativeLaunch)
        {
            throw new InvalidOperationException(
                "Cooperative kernel launch is not supported on this HIP device. " +
                "Cooperative launch is required for kernels using grid.sync() for cross-block synchronization. " +
                "Use cell-level operations instead of sequence-level operations, or use a device with " +
                "cooperative launch capability.");
        }

        // Check grid size limits for cooperative launch
        var occResult = HipNativeBindings.hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
            out int maxBlocksPerSm, kernel, (int)blockSize, (nuint)sharedMemBytes, 0);
        if (occResult != HipError.Success)
        {
            throw new InvalidOperationException(
                $"Failed to query occupancy for cooperative kernel: {occResult}. " +
                "Use cell-level operations instead of sequence-level operations.");
        }

        int maxCooperativeBlocks = maxBlocksPerSm * _deviceProps.MultiProcessorCount;
        if (gridX > maxCooperativeBlocks)
        {
            throw new InvalidOperationException(
                $"Grid size ({gridX}) exceeds cooperative launch limit ({maxCooperativeBlocks}) " +
                $"for this device ({maxBlocksPerSm} blocks/SM × {_deviceProps.MultiProcessorCount} SMs). " +
                "Reduce batch size or use cell-level operations.");
        }

        fixed (IntPtr* argsPtr = args)
        {
            // Use hipModuleLaunchCooperativeKernel for module-obtained kernels (via hipModuleGetFunction)
            var result = HipNativeBindings.hipModuleLaunchCooperativeKernel(
                kernel, gridX, 1, 1, blockSize, 1, 1,
                sharedMemBytes, _stream, (IntPtr)argsPtr);
            HipNativeBindings.CheckError(result, "hipModuleLaunchCooperativeKernel");
        }
    }

    private unsafe void LaunchCooperativeKernel(IntPtr kernel, uint gridX, uint blockSize, uint sharedMemBytes, void** args)
    {
        if (!_supportsCooperativeLaunch)
        {
            throw new InvalidOperationException(
                "Cooperative kernel launch is not supported on this HIP device.");
        }

        var occResult = HipNativeBindings.hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
            out int maxBlocksPerSm, kernel, (int)blockSize, (nuint)sharedMemBytes, 0);
        if (occResult != HipError.Success)
        {
            throw new InvalidOperationException(
                $"Failed to query occupancy for cooperative kernel: {occResult}.");
        }

        int maxCooperativeBlocks = maxBlocksPerSm * _deviceProps.MultiProcessorCount;
        if (gridX > maxCooperativeBlocks)
        {
            throw new InvalidOperationException(
                $"Grid size ({gridX}) exceeds cooperative launch limit ({maxCooperativeBlocks}).");
        }

        var result = HipNativeBindings.hipModuleLaunchCooperativeKernel(
            kernel, gridX, 1, 1, blockSize, 1, 1,
            sharedMemBytes, _stream, (IntPtr)args);
        HipNativeBindings.CheckError(result, "hipModuleLaunchCooperativeKernel");
    }

    private unsafe void LaunchKernel2D(IntPtr kernel, uint gridX, uint gridY, uint blockX, uint blockY, IntPtr[] args, uint sharedMem = 0)
    {
        LaunchKernel2DOnStream(kernel, gridX, gridY, blockX, blockY, args, _stream, sharedMem);
    }

    private unsafe void LaunchKernel2DOnStream(IntPtr kernel, uint gridX, uint gridY, uint blockX, uint blockY, IntPtr[] args, IntPtr stream, uint sharedMem = 0)
    {
        fixed (IntPtr* argsPtr = args)
        {
            // HIP driver API calls are required for kernel dispatch.
            var result = HipNativeBindings.hipModuleLaunchKernel(
                kernel, gridX, gridY, 1, blockX, blockY, 1,
                sharedMem, stream, (IntPtr)argsPtr, IntPtr.Zero);
            HipNativeBindings.CheckError(result, "hipModuleLaunchKernel");
        }
    }

    private unsafe void LaunchKernel3D(IntPtr kernel, uint gridX, uint gridY, uint gridZ, uint blockX, uint blockY, uint blockZ, IntPtr[] args, uint sharedMem = 0)
    {
        LaunchKernel3DOnStream(kernel, gridX, gridY, gridZ, blockX, blockY, blockZ, args, _stream, sharedMem);
    }

    private unsafe void LaunchKernel3D(IntPtr kernel, uint gridX, uint gridY, uint gridZ, uint blockX, uint blockY, uint blockZ, void** args, uint sharedMem = 0)
    {
        var result = HipNativeBindings.hipModuleLaunchKernel(
            kernel, gridX, gridY, gridZ, blockX, blockY, blockZ,
            sharedMem, _stream, (IntPtr)args, IntPtr.Zero);
        HipNativeBindings.CheckError(result, "hipModuleLaunchKernel");
    }

    private unsafe void LaunchKernel3DOnStream(IntPtr kernel, uint gridX, uint gridY, uint gridZ, uint blockX, uint blockY, uint blockZ, IntPtr[] args, IntPtr stream, uint sharedMem = 0)
    {
        fixed (IntPtr* argsPtr = args)
        {
            // HIP driver API calls are required for kernel dispatch.
            var result = HipNativeBindings.hipModuleLaunchKernel(
                kernel, gridX, gridY, gridZ, blockX, blockY, blockZ,
                sharedMem, stream, (IntPtr)argsPtr, IntPtr.Zero);
            HipNativeBindings.CheckError(result, "hipModuleLaunchKernel");
        }
    }

    private unsafe void ExecuteFusedGemm(string kernelName, IGpuBuffer A, IGpuBuffer B, IGpuBuffer bias, IGpuBuffer output, int M, int N, int K)
    {
        ExecuteFusedGemmOnStream(kernelName, A, B, bias, output, M, N, K, _stream, synchronize: true);
    }

    private unsafe void ExecuteFusedGemmOnStream(string kernelName, IGpuBuffer A, IGpuBuffer B, IGpuBuffer bias, IGpuBuffer output, int M, int N, int K, IntPtr stream, bool synchronize)
    {
        if (!_kernelCache.TryGetValue(kernelName, out var kernel))
            throw new InvalidOperationException($"HIP fused kernel not found: {kernelName}");

        // 128x128 CTA tile with 16x16 thread block (8x8 register blocking per thread)
        const int BM = 128;
        const int BN = 128;
        const int BLOCK_DIM = 16;
        uint gridX = (uint)((N + BN - 1) / BN);
        uint gridY = (uint)((M + BM - 1) / BM);

        IntPtr aHandle = A.Handle, bHandle = B.Handle, biasHandle = bias.Handle, outHandle = output.Handle;
        var args = new IntPtr[]
        {
            (IntPtr)(&aHandle),
            (IntPtr)(&bHandle),
            (IntPtr)(&biasHandle),
            (IntPtr)(&outHandle),
            (IntPtr)(&M),
            (IntPtr)(&N),
            (IntPtr)(&K)
        };

        // HIP kernel launch uses unmanaged interop with the driver API.
        LaunchKernel2DOnStream(kernel, gridX, gridY, BLOCK_DIM, BLOCK_DIM, args, stream);
        if (synchronize)
        {
            // HIP stream synchronization requires unmanaged interop.
            var syncResult = HipNativeBindings.hipStreamSynchronize(stream);
            HipNativeBindings.CheckError(syncResult, "hipStreamSynchronize");
        }
    }

    #region Memory Management

    public unsafe IGpuBuffer AllocateBuffer(float[] data)
    {
        IntPtr devicePtr = IntPtr.Zero;
        var size = (UIntPtr)(data.Length * sizeof(float));

        if (_bufferPool.TryRent(data.Length, out var pooled) && pooled != null)
        {
            fixed (float* dataPtr = data)
            {
                var result = HipNativeBindings.hipMemcpy(
                    pooled.Handle,
                    (IntPtr)dataPtr,
                    size,
                    HipMemcpyKind.HostToDevice); // lgtm[cs/call-to-unmanaged-code] HIP interop requires native driver calls.
                HipNativeBindings.CheckError(result, "hipMemcpy H2D");
            }

            return pooled;
        }

        var allocResult = HipNativeBindings.hipMalloc(ref devicePtr, size); // lgtm[cs/call-to-unmanaged-code] HIP interop requires native driver calls.
        HipNativeBindings.CheckError(allocResult, "hipMalloc");

        // Copy data to device
        fixed (float* dataPtr = data)
        {
            var copyResult = HipNativeBindings.hipMemcpy(
                devicePtr,
                (IntPtr)dataPtr,
                size,
                HipMemcpyKind.HostToDevice); // lgtm[cs/call-to-unmanaged-code] HIP interop requires native driver calls.
            HipNativeBindings.CheckError(copyResult, "hipMemcpy H2D");
        }

        return new HipGpuBuffer(devicePtr, data.Length, ReturnBufferToPool);
    }

    public IGpuBuffer AllocateBuffer(int size)
    {
        IntPtr devicePtr = IntPtr.Zero;
        var sizeBytes = (UIntPtr)(size * sizeof(float));

        if (_bufferPool.TryRent(size, out var pooled) && pooled != null)
        {
            var zeroResult = HipNativeBindings.hipMemset(pooled.Handle, 0, sizeBytes); // lgtm[cs/call-to-unmanaged-code] HIP interop requires native driver calls.
            HipNativeBindings.CheckError(zeroResult, "hipMemset");
            return pooled;
        }

        var allocResult = HipNativeBindings.hipMalloc(ref devicePtr, sizeBytes); // lgtm[cs/call-to-unmanaged-code] HIP interop requires native driver calls.
        HipNativeBindings.CheckError(allocResult, "hipMalloc");

        // Zero-initialize
        var memsetResult = HipNativeBindings.hipMemset(devicePtr, 0, sizeBytes); // lgtm[cs/call-to-unmanaged-code] HIP interop requires native driver calls.
        HipNativeBindings.CheckError(memsetResult, "hipMemset");

        return new HipGpuBuffer(devicePtr, size, ReturnBufferToPool);
    }

    public float[] DownloadBuffer(IGpuBuffer buffer)
    {
        var hipBuffer = (HipGpuBuffer)buffer;
        float[] result = new float[hipBuffer.Size];
        DownloadBuffer(buffer, result);
        return result;
    }

    public unsafe void DownloadBuffer(IGpuBuffer buffer, float[] destination)
    {
        var hipBuffer = (HipGpuBuffer)buffer;
        var size = (UIntPtr)(hipBuffer.Size * sizeof(float));

        fixed (float* destPtr = destination)
        {
            var result = HipNativeBindings.hipMemcpy(
                (IntPtr)destPtr,
                hipBuffer.Handle,
                size,
                HipMemcpyKind.DeviceToHost);
            HipNativeBindings.CheckError(result, "hipMemcpy D2H");
        }
    }

    #endregion

    #region GEMM Operations

    public unsafe void Gemm(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int M, int N, int K, float alpha = 1.0f, float beta = 0.0f)
    {
        if (TryExecuteHipBlasGemm(A, B, C, M, N, K, alpha, beta))
        {
            return;
        }

        var bufferA = (HipGpuBuffer)A;
        var bufferB = (HipGpuBuffer)B;
        var bufferC = (HipGpuBuffer)C;

        // Select kernel based on architecture and matrix size
        IntPtr kernel = SelectGemmKernel(M, N, K);

        if (kernel == IntPtr.Zero)
        {
            // Fallback to CPU (should not happen if initialized properly)
            throw new InvalidOperationException("No suitable GEMM kernel available");
        }

        // Calculate grid/block dimensions based on kernel type
        GemmKernelType kernelType = GetKernelType(kernel);

        int tileM, tileN, blockSize;
        switch (kernelType)
        {
            case GemmKernelType.Mfma:
                // MFMA: 128x128 workgroup tiles, 256 threads (4 warps of 64)
                tileM = 128;
                tileN = 128;
                blockSize = 256;
                break;
            case GemmKernelType.RdnaWave32:
                // RDNA wave32: 32x32 tiles, 256 threads
                tileM = 32;
                tileN = 32;
                blockSize = 256;
                break;
            case GemmKernelType.Scalar:
            default:
                // Scalar: 16x16 tiles, 256 threads
                tileM = 16;
                tileN = 16;
                blockSize = 256;
                break;
        }

        uint gridDimX = (uint)((M + tileM - 1) / tileM);
        uint gridDimY = (uint)((N + tileN - 1) / tileN);

        // Prepare kernel arguments using stackalloc (zero GCHandle overhead)
        var argA = bufferA.Handle;
        var argB = bufferB.Handle;
        var argC = bufferC.Handle;

        void** args = stackalloc void*[8];
        args[0] = &argA;
        args[1] = &argB;
        args[2] = &argC;
        args[3] = &M;
        args[4] = &N;
        args[5] = &K;
        args[6] = &alpha;
        args[7] = &beta;

        var result = HipNativeBindings.hipModuleLaunchKernel(
            kernel,
            gridDimX, gridDimY, 1,
            (uint)blockSize, 1, 1,
            0, _stream,
            (IntPtr)args,
            IntPtr.Zero);
        HipNativeBindings.CheckError(result, "hipModuleLaunchKernel");

        // Synchronize
        var syncResult = HipNativeBindings.hipStreamSynchronize(_stream);
        HipNativeBindings.CheckError(syncResult, "hipStreamSynchronize");
    }

    private bool TryExecuteHipBlasGemm(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int M, int N, int K, float alpha, float beta)
    {
        if (!_hipblasAvailable || _hipblasHandle == IntPtr.Zero)
        {
            return false;
        }

        if (!ShouldUseVendorGemm(M, N, K))
        {
            return false;
        }

        var bufferA = (HipGpuBuffer)A;
        var bufferB = (HipGpuBuffer)B;
        var bufferC = (HipGpuBuffer)C;

        float alphaVal = alpha;
        float betaVal = beta;

        // hipBLAS uses column-major; swap A/B and M/N so C^T = B^T * A^T for row-major inputs.
        var status = HipBlasNative.hipblasSgemm(
            _hipblasHandle,
            HipBlasNative.HipBlasOperation.None,
            HipBlasNative.HipBlasOperation.None,
            N, M, K,
            ref alphaVal,
            bufferB.Handle, N,
            bufferA.Handle, K,
            ref betaVal,
            bufferC.Handle, N); // lgtm[cs/call-to-unmanaged-code] HIP BLAS uses native bindings.

        if (status != HipBlasNative.HipBlasStatus.Success)
        {
            if (EnableDiagnostics)
            {
                Console.WriteLine($"hipBLAS SGEMM failed with status: {status}");
            }
            return false;
        }

        var syncResult = HipNativeBindings.hipStreamSynchronize(_stream); // lgtm[cs/call-to-unmanaged-code] HIP interop requires native driver calls.
        HipNativeBindings.CheckError(syncResult, "hipStreamSynchronize (hipBLAS)");
        return true;
    }

    private IntPtr SelectGemmKernel(int M, int N, int K)
    {
        // Select kernel based on architecture - use the recommended kernel from HipMfmaKernel
        string recommendedKernel = HipMfmaKernel.GetRecommendedKernel(_architecture);

        if (_kernelCache.TryGetValue(recommendedKernel, out var kernel))
            return kernel;

        // Fallback order: try scalar first (works everywhere), then RDNA wave32, then MFMA
        if (_kernelCache.TryGetValue("scalar_gemm_f32", out var scalarKernel))
            return scalarKernel;

        if (_kernelCache.TryGetValue("rdna_gemm_wave32", out var rdnaKernel))
            return rdnaKernel;

        if (_kernelCache.TryGetValue("mfma_gemm_f32", out var mfmaKernel))
            return mfmaKernel;

        return IntPtr.Zero;
    }

    /// <summary>
    /// Gets the kernel type for the currently selected kernel.
    /// </summary>
    private GemmKernelType GetKernelType(IntPtr kernel)
    {
        if (_kernelCache.TryGetValue("mfma_gemm_f32", out var mfmaKernel) && kernel == mfmaKernel)
            return GemmKernelType.Mfma;
        if (_kernelCache.TryGetValue("mfma_gemm_f16", out var mfmaF16Kernel) && kernel == mfmaF16Kernel)
            return GemmKernelType.Mfma;
        if (_kernelCache.TryGetValue("rdna_gemm_wave32", out var rdnaKernel) && kernel == rdnaKernel)
            return GemmKernelType.RdnaWave32;
        if (_kernelCache.TryGetValue("scalar_gemm_f32", out var scalarKernel) && kernel == scalarKernel)
            return GemmKernelType.Scalar;

        return GemmKernelType.Scalar;  // Default to scalar dimensions
    }

    private enum GemmKernelType
    {
        Mfma,       // 128x128 workgroup tiles, 256 threads
        RdnaWave32, // 32x32 tiles, 256 threads
        Scalar      // 16x16 tiles, 256 threads
    }

    public IGpuBuffer MatMul(IGpuBuffer A, IGpuBuffer B, int M, int N, int K)
    {
        var C = AllocateBuffer(M * N);
        Gemm(A, B, C, M, N, K, 1.0f, 0.0f);
        return C;
    }

    public unsafe void BatchedGemm(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int M, int N, int K, int batchCount, float alpha = 1.0f, float beta = 0.0f)
    {
        if (batchCount <= 0)
            throw new ArgumentException("Batch count must be positive", nameof(batchCount));

        if (!_kernelCache.TryGetValue("batched_gemm", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: batched_gemm");

            {
            IntPtr _p0 = A.Handle;
            IntPtr _p1 = B.Handle;
            IntPtr _p2 = C.Handle;
            void** args = stackalloc void*[9];
            args[0] = &_p0;
            args[1] = &_p1;
            args[2] = &_p2;
            args[3] = &M;
            args[4] = &N;
            args[5] = &K;
            args[6] = &batchCount;
            args[7] = &alpha;
            args[8] = &beta;


            // 3D grid: (N tiles, M tiles, batches) — must match BATCHED_TILE_SIZE in kernel
            const int tileSize = 32;
            uint gridX = (uint)((N + tileSize - 1) / tileSize);
            uint gridY = (uint)((M + tileSize - 1) / tileSize);
            uint gridZ = (uint)batchCount;
            LaunchKernel3D(krnl, gridX, gridY, gridZ, (uint)tileSize, (uint)tileSize, 1, args);
            Synchronize();
            }
    }

    #endregion

    #region Fused Operations

    public IGpuBuffer GemmBiasRelu(IGpuBuffer A, IGpuBuffer B, IGpuBuffer bias, int M, int N, int K)
    {
        IGpuBuffer? temp = null;
        IGpuBuffer? output = null;
        try
        {
            temp = GemmBias(A, B, bias, M, N, K);
            output = AllocateBuffer(M * N);
            Relu(temp, output, M * N);
            Synchronize(); // Ensure stream completes before returning buffer
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
        IGpuBuffer? temp = null;
        IGpuBuffer? output = null;
        try
        {
            temp = GemmBias(A, B, bias, M, N, K);
            output = AllocateBuffer(M * N);
            Gelu(temp, output, M * N);
            Synchronize();
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
        IGpuBuffer? temp = null;
        IGpuBuffer? output = null;
        try
        {
            temp = GemmBias(A, B, bias, M, N, K);
            output = AllocateBuffer(M * N);
            Sigmoid(temp, output, M * N);
            Synchronize();
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
        IGpuBuffer? temp = null;
        IGpuBuffer? output = null;
        try
        {
            temp = GemmBias(A, B, bias, M, N, K);
            output = AllocateBuffer(M * N);
            Tanh(temp, output, M * N);
            Synchronize();
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
        // Use optimized GEMM (hipBLAS/tiled) + in-place bias add (no self-copy since A==C is handled)
        var output = MatMul(A, B, M, N, K);
        BiasAddInPlace(output, bias, M, N);
        return output;
    }

    public IGpuBuffer GemmBiasSwish(IGpuBuffer A, IGpuBuffer B, IGpuBuffer bias, int M, int N, int K)
    {
        IGpuBuffer? temp = null;
        IGpuBuffer? output = null;
        try
        {
            temp = GemmBias(A, B, bias, M, N, K);
            output = AllocateBuffer(M * N);
            Silu(temp, output, M * N);
            Synchronize();
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
        var output = AllocateBuffer(M * N);

        if (!_kernelCache.TryGetValue("gemm_bias_leaky_relu", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: gemm_bias_leaky_relu");

        const int BM = 128;
        const int BN = 128;
        uint gridX = (uint)((N + BN - 1) / BN);
        uint gridY = (uint)((M + BM - 1) / BM);

            {
            IntPtr _p0 = A.Handle;
            IntPtr _p1 = B.Handle;
            IntPtr _p2 = bias.Handle;
            IntPtr _p3 = output.Handle;
            void** args = stackalloc void*[8];
            args[0] = &_p0;
            args[1] = &_p1;
            args[2] = &_p2;
            args[3] = &_p3;
            args[4] = &M;
            args[5] = &N;
            args[6] = &K;
            args[7] = &alpha;


            LaunchKernel2D(krnl, gridX, gridY, 16, 16, args);
            Synchronize();
            }

        return output;
    }

    /// <summary>
    /// In-place bias add — skips the D2D copy when the output buffer already contains the data.
    /// </summary>
    private unsafe void BiasAddInPlace(IGpuBuffer output, IGpuBuffer bias, int M, int N)
    {
        if (!_kernelCache.TryGetValue("bias_add", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: bias_add");

            {
            IntPtr _p0 = output.Handle;
            IntPtr _p1 = bias.Handle;
            void** args = stackalloc void*[4];
            args[0] = &_p0;
            args[1] = &_p1;
            args[2] = &M;
            args[3] = &N;


            // 2D grid: gridX covers columns, gridY covers rows
            // Eliminates ~20-cycle integer division from idx % cols in the kernel
            uint gridX = (uint)(N + DefaultBlockSize - 1) / DefaultBlockSize;
            uint gridY = (uint)M;
            LaunchKernel2D(krnl, gridX, gridY, DefaultBlockSize, 1, args);
            Synchronize();
            }
    }

    public unsafe void BiasAdd(IGpuBuffer A, IGpuBuffer bias, IGpuBuffer C, int M, int N)
    {
        if (!_kernelCache.TryGetValue("bias_add", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: bias_add");

        // Copy A to C only when they are different buffers (bias_add is in-place)
        if (A.Handle != C.Handle)
        {
            var size = (UIntPtr)(M * N * sizeof(float));
            var result = HipNativeBindings.hipMemcpy(C.Handle, A.Handle, size, HipMemcpyKind.DeviceToDevice);
            HipNativeBindings.CheckError(result, "hipMemcpy D2D");
        }

            {
            IntPtr _p0 = C.Handle;
            IntPtr _p1 = bias.Handle;
            void** args = stackalloc void*[4];
            args[0] = &_p0;
            args[1] = &_p1;
            args[2] = &M;
            args[3] = &N;


            // 2D grid: gridX covers columns, gridY covers rows
            uint gridX = (uint)(N + DefaultBlockSize - 1) / DefaultBlockSize;
            uint gridY = (uint)M;
            LaunchKernel2D(krnl, gridX, gridY, DefaultBlockSize, 1, args);
            Synchronize();
            }
    }

    /// <summary>
    /// Adds bias to Conv2D output in NCHW format.
    /// Operation: output[b, c, h, w] += bias[c]
    /// </summary>
    public unsafe void Conv2DBiasAdd(IGpuBuffer output, IGpuBuffer bias, int batch, int channels, int spatialSize)
    {
        if (!_kernelCache.TryGetValue("conv2d_bias_add", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: conv2d_bias_add");

            {
            IntPtr _p0 = output.Handle;
            IntPtr _p1 = bias.Handle;
            void** args = stackalloc void*[5];
            args[0] = &_p0;
            args[1] = &_p1;
            args[2] = &batch;
            args[3] = &channels;
            args[4] = &spatialSize;


            // 2D grid: x=spatial, y=batch*channels — eliminates integer division in kernel
            uint gridX = (uint)((spatialSize + DefaultBlockSize - 1) / DefaultBlockSize);
            uint gridY = (uint)(batch * channels);
            LaunchKernel2D(krnl, gridX, gridY, DefaultBlockSize, 1, args);
            Synchronize();
            }
    }

    #endregion

    #region Element-wise Operations

    public void Add(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int size)
    {
        LaunchBinaryOpAutoVec4("add_vectors", A, B, C, size);
    }
    public void AddRelu(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int size) { Add(A, B, C, size); Relu(C, C, size); }
    public void AddSigmoid(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int size) { Add(A, B, C, size); Sigmoid(C, C, size); }
    public void AddGelu(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int size) { Add(A, B, C, size); Gelu(C, C, size); }

    public void Subtract(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int size)
    {
        LaunchBinaryOpAutoVec4("subtract_vectors", A, B, C, size);
    }

    public void Multiply(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int size)
    {
        LaunchBinaryOpAutoVec4("multiply_vectors", A, B, C, size);
    }

    public void Divide(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int size)
    {
        LaunchBinaryOpAutoVec4("divide_vectors", A, B, C, size);
    }

    public void Min(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int size)
    {
        LaunchBinaryOpAutoVec4("min_vectors", A, B, C, size);
    }

    public void Max(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int size)
    {
        LaunchBinaryOpAutoVec4("max_vectors", A, B, C, size);
    }

    public unsafe void Scale(IGpuBuffer A, IGpuBuffer B, float scalar, int size)
    {
        if (size % 4 == 0 && _kernelCache.TryGetValue("scale_vector_vec4", out var vec4Krnl))
        {
            int size4 = size / 4;
                {
                IntPtr _p0 = A.Handle;
                IntPtr _p1 = B.Handle;
                void** args = stackalloc void*[4];
                args[0] = &_p0;
                args[1] = &_p1;
                args[2] = &scalar;
                args[3] = &size4;


                uint grid = (uint)((size4 + DefaultBlockSize - 1) / DefaultBlockSize);
                LaunchKernel(vec4Krnl, grid, DefaultBlockSize, args);
                Synchronize();
                }
            return;
        }

        if (!_kernelCache.TryGetValue("scale_vector", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: scale_vector");

            {
            IntPtr _p0 = A.Handle;
            IntPtr _p1 = B.Handle;
            void** args = stackalloc void*[4];
            args[0] = &_p0;
            args[1] = &_p1;
            args[2] = &scalar;
            args[3] = &size;


            uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
            LaunchKernel(krnl, grid, DefaultBlockSize, args);
            Synchronize();
            }
    }

    public unsafe void StridedGather(IGpuBuffer src, IGpuBuffer dst, int offset, int stride, int count)
    {
        if (count <= 0) return;
        if (offset < 0) throw new ArgumentOutOfRangeException(nameof(offset));
        if (stride <= 0) throw new ArgumentOutOfRangeException(nameof(stride));
        if (offset + (count - 1) * stride >= src.Size) throw new ArgumentOutOfRangeException(nameof(count));
        if (count > dst.Size) throw new ArgumentOutOfRangeException(nameof(count));

        if (!_kernelCache.TryGetValue("strided_gather", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: strided_gather");

        IntPtr srcPtr = src.Handle;
        IntPtr dstPtr = dst.Handle;
        int off = offset, str = stride, cnt = count;
        void** args = stackalloc void*[5];
        args[0] = &srcPtr; args[1] = &dstPtr; args[2] = &off; args[3] = &str; args[4] = &cnt;
        uint grid = (uint)((count + DefaultBlockSize - 1) / DefaultBlockSize);
        LaunchKernel(krnl, grid, DefaultBlockSize, args);
        Synchronize();
    }

    public unsafe void StridedScatter(IGpuBuffer src, IGpuBuffer dst, int offset, int stride, int count)
    {
        if (count <= 0) return;
        if (offset < 0) throw new ArgumentOutOfRangeException(nameof(offset));
        if (stride <= 0) throw new ArgumentOutOfRangeException(nameof(stride));
        if (offset + (count - 1) * stride >= dst.Size) throw new ArgumentOutOfRangeException(nameof(count));
        if (count > src.Size) throw new ArgumentOutOfRangeException(nameof(count));

        if (!_kernelCache.TryGetValue("strided_scatter", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: strided_scatter");

        IntPtr srcPtr = src.Handle;
        IntPtr dstPtr = dst.Handle;
        int off = offset, str = stride, cnt = count;
        void** args = stackalloc void*[5];
        args[0] = &srcPtr; args[1] = &dstPtr; args[2] = &off; args[3] = &str; args[4] = &cnt;
        uint grid = (uint)((count + DefaultBlockSize - 1) / DefaultBlockSize);
        LaunchKernel(krnl, grid, DefaultBlockSize, args);
        Synchronize();
    }

    public void Power(IGpuBuffer A, IGpuBuffer B, float exponent, int size)
    {
        // Power uses same pattern as Scale
        if (!_kernelCache.TryGetValue("power_scalar", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: power_scalar");

        LaunchScalarOp(krnl, A, B, exponent, size);
    }

    private unsafe void LaunchScalarOp(IntPtr krnl, IGpuBuffer A, IGpuBuffer B, float scalar, int size)
    {
            {
            IntPtr _p0 = A.Handle;
            IntPtr _p1 = B.Handle;
            void** args = stackalloc void*[4];
            args[0] = &_p0;
            args[1] = &_p1;
            args[2] = &scalar;
            args[3] = &size;


            uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
            LaunchKernel(krnl, grid, DefaultBlockSize, args);
            Synchronize();
            }
    }

    public void Abs(IGpuBuffer A, IGpuBuffer B, int size)
    {
        LaunchUnaryOpAutoVec4("abs_vector", A, B, size);
    }

    public void Exp(IGpuBuffer A, IGpuBuffer B, int size)
    {
        LaunchUnaryOpAutoVec4("exp_vector", A, B, size);
    }

    public void Exp2(IGpuBuffer A, IGpuBuffer B, int size)
    {
        if (!_kernelCache.TryGetValue("exp2_vector", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: exp2_vector");

        LaunchUnaryOp(krnl, A, B, size);
    }

    public void Exp10(IGpuBuffer A, IGpuBuffer B, int size)
    {
        if (!_kernelCache.TryGetValue("exp10_vector", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: exp10_vector");

        LaunchUnaryOp(krnl, A, B, size);
    }

    public void ExpM1(IGpuBuffer A, IGpuBuffer B, int size)
    {
        if (!_kernelCache.TryGetValue("expm1_vector", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: expm1_vector");

        LaunchUnaryOp(krnl, A, B, size);
    }

    public void Log(IGpuBuffer A, IGpuBuffer B, int size)
    {
        LaunchUnaryOpAutoVec4("log_vector", A, B, size);
    }

    public void Log2(IGpuBuffer A, IGpuBuffer B, int size)
    {
        if (!_kernelCache.TryGetValue("log2_vector", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: log2_vector");

        LaunchUnaryOp(krnl, A, B, size);
    }

    public void Log1P(IGpuBuffer A, IGpuBuffer B, int size)
    {
        if (!_kernelCache.TryGetValue("log1p_vector", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: log1p_vector");

        LaunchUnaryOp(krnl, A, B, size);
    }

    public void Sqrt(IGpuBuffer A, IGpuBuffer B, int size)
    {
        LaunchUnaryOpAutoVec4("sqrt_vector", A, B, size);
    }

    public void Sign(IGpuBuffer A, IGpuBuffer B, int size)
    {
        if (!_kernelCache.TryGetValue("sign_vector", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: sign_vector");

        LaunchUnaryOp(krnl, A, B, size);
    }

    public void Relu(IGpuBuffer A, IGpuBuffer B, int size)
    {
        LaunchUnaryOpAutoVec4("relu", A, B, size);
    }

    public void Sigmoid(IGpuBuffer A, IGpuBuffer B, int size)
    {
        LaunchUnaryOpAutoVec4("sigmoid", A, B, size);
    }

    private unsafe void LaunchUnaryOp(IntPtr krnl, IGpuBuffer A, IGpuBuffer B, int size)
    {
            {
            IntPtr _p0 = A.Handle;
            IntPtr _p1 = B.Handle;
            void** args = stackalloc void*[3];
            args[0] = &_p0;
            args[1] = &_p1;
            args[2] = &size;


            uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
            LaunchKernel(krnl, grid, DefaultBlockSize, args);
            Synchronize();
            }
    }

    /// <summary>
    /// Launches a unary op with automatic vec4 selection when size is divisible by 4.
    /// </summary>
    private unsafe void LaunchUnaryOpAutoVec4(string kernelName, IGpuBuffer A, IGpuBuffer B, int size)
    {
        if (size % 4 == 0 && _kernelCache.TryGetValue(kernelName + "_vec4", out var vec4Krnl))
        {
            int size4 = size / 4;
                {
                IntPtr _p0 = A.Handle;
                IntPtr _p1 = B.Handle;
                void** args = stackalloc void*[3];
                args[0] = &_p0;
                args[1] = &_p1;
                args[2] = &size4;
                uint grid = (uint)((size4 + DefaultBlockSize - 1) / DefaultBlockSize);
                LaunchKernel(vec4Krnl, grid, DefaultBlockSize, args);
                Synchronize();
                }
            return;
        }

        if (!_kernelCache.TryGetValue(kernelName, out var krnl))
            throw new InvalidOperationException($"HIP kernel not found: {kernelName}");
        LaunchUnaryOp(krnl, A, B, size);
    }

    public void Tanh(IGpuBuffer A, IGpuBuffer B, int size)
    {
        LaunchUnaryOpAutoVec4("tanh_activation", A, B, size);
    }

    public void Gelu(IGpuBuffer A, IGpuBuffer B, int size)
    {
        LaunchUnaryOpAutoVec4("gelu", A, B, size);
    }

    public unsafe void LeakyRelu(IGpuBuffer A, IGpuBuffer B, float alpha, int size)
    {
        if (!_kernelCache.TryGetValue("leaky_relu", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: leaky_relu");

            {
            IntPtr _p0 = A.Handle;
            IntPtr _p1 = B.Handle;
            void** args = stackalloc void*[4];
            args[0] = &_p0;
            args[1] = &_p1;
            args[2] = &alpha;
            args[3] = &size;


            uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
            LaunchKernel(krnl, grid, DefaultBlockSize, args);
            Synchronize();
            }
    }

    public unsafe void LeakyReluBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradInput, float alpha, int size)
    {
        if (!_kernelCache.TryGetValue("leaky_relu_backward", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: leaky_relu_backward");

            {
            IntPtr _p0 = gradOutput.Handle;
            IntPtr _p1 = input.Handle;
            IntPtr _p2 = gradInput.Handle;
            void** args = stackalloc void*[5];
            args[0] = &_p0;
            args[1] = &_p1;
            args[2] = &_p2;
            args[3] = &alpha;
            args[4] = &size;


            uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
            LaunchKernel(krnl, grid, DefaultBlockSize, args);
            Synchronize();
            }
    }

    public unsafe void Elu(IGpuBuffer A, IGpuBuffer B, float alpha, int size)
    {
        if (!_kernelCache.TryGetValue("elu", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: elu");

            {
            IntPtr _p0 = A.Handle;
            IntPtr _p1 = B.Handle;
            void** args = stackalloc void*[4];
            args[0] = &_p0;
            args[1] = &_p1;
            args[2] = &alpha;
            args[3] = &size;


            uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
            LaunchKernel(krnl, grid, DefaultBlockSize, args);
            Synchronize();
            }
    }

    public unsafe void EluBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer output, IGpuBuffer gradInput, float alpha, int size)
    {
        if (!_kernelCache.TryGetValue("elu_backward", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: elu_backward");

            {
            IntPtr _p0 = gradOutput.Handle;
            IntPtr _p1 = input.Handle;
            IntPtr _p2 = output.Handle;
            IntPtr _p3 = gradInput.Handle;
            void** args = stackalloc void*[6];
            args[0] = &_p0;
            args[1] = &_p1;
            args[2] = &_p2;
            args[3] = &_p3;
            args[4] = &alpha;
            args[5] = &size;


            uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
            LaunchKernel(krnl, grid, DefaultBlockSize, args);
            Synchronize();
            }
    }

    public void Swish(IGpuBuffer A, IGpuBuffer B, int size)
    {
        LaunchUnaryOpAutoVec4("swish", A, B, size);
    }

    public unsafe void SwishBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradInput, int size)
    {
        if (!_kernelCache.TryGetValue("swish_backward", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: swish_backward");

            {
            IntPtr _p0 = gradOutput.Handle;
            IntPtr _p1 = input.Handle;
            IntPtr _p2 = gradInput.Handle;
            void** args = stackalloc void*[4];
            args[0] = &_p0;
            args[1] = &_p1;
            args[2] = &_p2;
            args[3] = &size;


            uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
            LaunchKernel(krnl, grid, DefaultBlockSize, args);
            Synchronize();
            }
    }

    public void Silu(IGpuBuffer A, IGpuBuffer B, int size)
    {
        // SiLU is the same as Swish
        Swish(A, B, size);
    }

    public void Mish(IGpuBuffer A, IGpuBuffer B, int size)
    {
        if (!_kernelCache.TryGetValue("mish_vector", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: mish_vector");

        LaunchUnaryOp(krnl, A, B, size);
    }

    public void Softplus(IGpuBuffer A, IGpuBuffer B, int size)
    {
        if (!_kernelCache.TryGetValue("softplus_vector", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: softplus_vector");

        LaunchUnaryOp(krnl, A, B, size);
    }

    public void Hardswish(IGpuBuffer A, IGpuBuffer B, int size)
    {
        if (!_kernelCache.TryGetValue("hardswish_vector", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: hardswish_vector");

        LaunchUnaryOp(krnl, A, B, size);
    }

    public unsafe void Selu(IGpuBuffer A, IGpuBuffer B, float alpha, float scale, int size)
    {
        if (!_kernelCache.TryGetValue("selu_vector", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: selu_vector");

            {
            IntPtr _p0 = A.Handle;
            IntPtr _p1 = B.Handle;
            void** args = stackalloc void*[5];
            args[0] = &_p0;
            args[1] = &_p1;
            args[2] = &alpha;
            args[3] = &scale;
            args[4] = &size;


            uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
            LaunchKernel(krnl, grid, DefaultBlockSize, args);
            Synchronize();
            }
    }

    public void Hardsigmoid(IGpuBuffer A, IGpuBuffer B, int size)
    {
        if (!_kernelCache.TryGetValue("hardsigmoid_vector", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: hardsigmoid_vector");

        LaunchUnaryOp(krnl, A, B, size);
    }

    public unsafe void Hardtanh(IGpuBuffer A, IGpuBuffer B, float minVal, float maxVal, int size)
    {
        if (!_kernelCache.TryGetValue("hardtanh_vector", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: hardtanh_vector");

            {
            IntPtr _p0 = A.Handle;
            IntPtr _p1 = B.Handle;
            void** args = stackalloc void*[5];
            args[0] = &_p0;
            args[1] = &_p1;
            args[2] = &minVal;
            args[3] = &maxVal;
            args[4] = &size;


            uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
            LaunchKernel(krnl, grid, DefaultBlockSize, args);
            Synchronize();
            }
    }

    // SiLU backward uses SwishBackward since they're mathematically equivalent
    public void SiluBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradInput, int size)
    {
        SwishBackward(gradOutput, input, gradInput, size);
    }

    public unsafe void MishBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradInput, int size)
    {
        if (!_kernelCache.TryGetValue("mish_backward", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: mish_backward");

            {
            IntPtr _p0 = gradOutput.Handle;
            IntPtr _p1 = input.Handle;
            IntPtr _p2 = gradInput.Handle;
            void** args = stackalloc void*[4];
            args[0] = &_p0;
            args[1] = &_p1;
            args[2] = &_p2;
            args[3] = &size;


            uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
            LaunchKernel(krnl, grid, DefaultBlockSize, args);
            Synchronize();
            }
    }

    public unsafe void SoftplusBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradInput, int size)
    {
        if (!_kernelCache.TryGetValue("softplus_backward", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: softplus_backward");

            {
            IntPtr _p0 = gradOutput.Handle;
            IntPtr _p1 = input.Handle;
            IntPtr _p2 = gradInput.Handle;
            void** args = stackalloc void*[4];
            args[0] = &_p0;
            args[1] = &_p1;
            args[2] = &_p2;
            args[3] = &size;


            uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
            LaunchKernel(krnl, grid, DefaultBlockSize, args);
            Synchronize();
            }
    }

    public unsafe void HardswishBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradInput, int size)
    {
        if (!_kernelCache.TryGetValue("hardswish_backward", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: hardswish_backward");

            {
            IntPtr _p0 = gradOutput.Handle;
            IntPtr _p1 = input.Handle;
            IntPtr _p2 = gradInput.Handle;
            void** args = stackalloc void*[4];
            args[0] = &_p0;
            args[1] = &_p1;
            args[2] = &_p2;
            args[3] = &size;


            uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
            LaunchKernel(krnl, grid, DefaultBlockSize, args);
            Synchronize();
            }
    }

    public unsafe void SeluBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradInput, float alpha, float scale, int size)
    {
        if (!_kernelCache.TryGetValue("selu_backward", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: selu_backward");

            {
            IntPtr _p0 = gradOutput.Handle;
            IntPtr _p1 = input.Handle;
            IntPtr _p2 = gradInput.Handle;
            void** args = stackalloc void*[6];
            args[0] = &_p0;
            args[1] = &_p1;
            args[2] = &_p2;
            args[3] = &alpha;
            args[4] = &scale;
            args[5] = &size;


            uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
            LaunchKernel(krnl, grid, DefaultBlockSize, args);
            Synchronize();
            }
    }

    public unsafe void HardsigmoidBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradInput, int size)
    {
        if (!_kernelCache.TryGetValue("hardsigmoid_backward", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: hardsigmoid_backward");

            {
            IntPtr _p0 = gradOutput.Handle;
            IntPtr _p1 = input.Handle;
            IntPtr _p2 = gradInput.Handle;
            void** args = stackalloc void*[4];
            args[0] = &_p0;
            args[1] = &_p1;
            args[2] = &_p2;
            args[3] = &size;


            uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
            LaunchKernel(krnl, grid, DefaultBlockSize, args);
            Synchronize();
            }
    }

    public unsafe void HardtanhBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradInput, float minVal, float maxVal, int size)
    {
        if (!_kernelCache.TryGetValue("hardtanh_backward", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: hardtanh_backward");

            {
            IntPtr _p0 = gradOutput.Handle;
            IntPtr _p1 = input.Handle;
            IntPtr _p2 = gradInput.Handle;
            void** args = stackalloc void*[6];
            args[0] = &_p0;
            args[1] = &_p1;
            args[2] = &_p2;
            args[3] = &minVal;
            args[4] = &maxVal;
            args[5] = &size;


            uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
            LaunchKernel(krnl, grid, DefaultBlockSize, args);
            Synchronize();
            }
    }

    public unsafe void Relu6(IGpuBuffer A, IGpuBuffer B, int size)
    {
        if (!_kernelCache.TryGetValue("relu6", out var krnl)) throw new InvalidOperationException("HIP kernel not found: relu6");
        IntPtr _p0 = A.Handle; IntPtr _p1 = B.Handle;
        void** args = stackalloc void*[3]; args[0] = &_p0; args[1] = &_p1; args[2] = &size;
        uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
        LaunchKernel(krnl, grid, DefaultBlockSize, args); Synchronize();
    }

    public unsafe void Relu6Backward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradInput, int size)
    {
        if (!_kernelCache.TryGetValue("relu6_backward", out var krnl)) throw new InvalidOperationException("HIP kernel not found: relu6_backward");
        IntPtr _p0 = gradOutput.Handle; IntPtr _p1 = input.Handle; IntPtr _p2 = gradInput.Handle;
        void** args = stackalloc void*[4]; args[0] = &_p0; args[1] = &_p1; args[2] = &_p2; args[3] = &size;
        uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
        LaunchKernel(krnl, grid, DefaultBlockSize, args); Synchronize();
    }

    public unsafe void PRelu(IGpuBuffer input, IGpuBuffer alpha, IGpuBuffer output, int size, int alphaSize)
    {
        if (!_kernelCache.TryGetValue("prelu", out var krnl)) throw new InvalidOperationException("HIP kernel not found: prelu");
        IntPtr _p0 = input.Handle; IntPtr _p1 = alpha.Handle; IntPtr _p2 = output.Handle;
        void** args = stackalloc void*[5]; args[0] = &_p0; args[1] = &_p1; args[2] = &_p2; args[3] = &size; args[4] = &alphaSize;
        uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
        LaunchKernel(krnl, grid, DefaultBlockSize, args); Synchronize();
    }

    public unsafe void PReluBackwardInput(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer alpha, IGpuBuffer gradInput, int size, int alphaSize)
    {
        if (!_kernelCache.TryGetValue("prelu_backward_input", out var krnl)) throw new InvalidOperationException("HIP kernel not found: prelu_backward_input");
        IntPtr _p0 = gradOutput.Handle; IntPtr _p1 = input.Handle; IntPtr _p2 = alpha.Handle; IntPtr _p3 = gradInput.Handle;
        void** args = stackalloc void*[6]; args[0] = &_p0; args[1] = &_p1; args[2] = &_p2; args[3] = &_p3; args[4] = &size; args[5] = &alphaSize;
        uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
        LaunchKernel(krnl, grid, DefaultBlockSize, args); Synchronize();
    }

    public unsafe void PReluBackwardAlpha(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradAlpha, int size, int alphaSize)
    {
        if (!_kernelCache.TryGetValue("prelu_backward_alpha", out var krnl)) throw new InvalidOperationException("HIP kernel not found: prelu_backward_alpha");
        IntPtr _p0 = gradOutput.Handle; IntPtr _p1 = input.Handle; IntPtr _p2 = gradAlpha.Handle;
        void** args = stackalloc void*[5]; args[0] = &_p0; args[1] = &_p1; args[2] = &_p2; args[3] = &size; args[4] = &alphaSize;
        uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
        LaunchKernel(krnl, grid, DefaultBlockSize, args); Synchronize();
    }

    public unsafe void RRelu(IGpuBuffer input, IGpuBuffer noise, IGpuBuffer output, int size)
    {
        if (!_kernelCache.TryGetValue("rrelu", out var krnl)) throw new InvalidOperationException("HIP kernel not found: rrelu");
        IntPtr _p0 = input.Handle; IntPtr _p1 = noise.Handle; IntPtr _p2 = output.Handle;
        void** args = stackalloc void*[4]; args[0] = &_p0; args[1] = &_p1; args[2] = &_p2; args[3] = &size;
        uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
        LaunchKernel(krnl, grid, DefaultBlockSize, args); Synchronize();
    }

    public unsafe void RReluBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer noise, IGpuBuffer gradInput, int size)
    {
        if (!_kernelCache.TryGetValue("rrelu_backward", out var krnl)) throw new InvalidOperationException("HIP kernel not found: rrelu_backward");
        IntPtr _p0 = gradOutput.Handle; IntPtr _p1 = input.Handle; IntPtr _p2 = noise.Handle; IntPtr _p3 = gradInput.Handle;
        void** args = stackalloc void*[5]; args[0] = &_p0; args[1] = &_p1; args[2] = &_p2; args[3] = &_p3; args[4] = &size;
        uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
        LaunchKernel(krnl, grid, DefaultBlockSize, args); Synchronize();
    }

    public unsafe void Threshold(IGpuBuffer input, IGpuBuffer output, float threshold, float value, int size)
    {
        if (!_kernelCache.TryGetValue("threshold_forward", out var krnl)) throw new InvalidOperationException("HIP kernel not found: threshold_forward");
        IntPtr _p0 = input.Handle; IntPtr _p1 = output.Handle;
        void** args = stackalloc void*[5]; args[0] = &_p0; args[1] = &_p1; args[2] = &threshold; args[3] = &value; args[4] = &size;
        uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
        LaunchKernel(krnl, grid, DefaultBlockSize, args); Synchronize();
    }

    public unsafe void ThresholdBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradInput, float threshold, int size)
    {
        if (!_kernelCache.TryGetValue("threshold_backward", out var krnl)) throw new InvalidOperationException("HIP kernel not found: threshold_backward");
        IntPtr _p0 = gradOutput.Handle; IntPtr _p1 = input.Handle; IntPtr _p2 = gradInput.Handle;
        void** args = stackalloc void*[5]; args[0] = &_p0; args[1] = &_p1; args[2] = &_p2; args[3] = &threshold; args[4] = &size;
        uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
        LaunchKernel(krnl, grid, DefaultBlockSize, args); Synchronize();
    }

    public unsafe void ReciprocalBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradInput, int size)
    {
        if (!_kernelCache.TryGetValue("reciprocal_backward", out var krnl)) throw new InvalidOperationException("HIP kernel not found: reciprocal_backward");
        IntPtr _p0 = gradOutput.Handle; IntPtr _p1 = input.Handle; IntPtr _p2 = gradInput.Handle;
        void** args = stackalloc void*[4]; args[0] = &_p0; args[1] = &_p1; args[2] = &_p2; args[3] = &size;
        uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
        LaunchKernel(krnl, grid, DefaultBlockSize, args); Synchronize();
    }

    public unsafe void VarBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer mean, IGpuBuffer gradInput, int outerSize, int reduceSize)
    {
        if (!_kernelCache.TryGetValue("var_backward", out var krnl)) throw new InvalidOperationException("HIP kernel not found: var_backward");
        int total = outerSize * reduceSize;
        IntPtr _p0 = gradOutput.Handle, _p1 = input.Handle, _p2 = mean.Handle, _p3 = gradInput.Handle;
        void** args = stackalloc void*[6]; args[0] = &_p0; args[1] = &_p1; args[2] = &_p2; args[3] = &_p3; args[4] = &outerSize; args[5] = &reduceSize;
        uint grid = (uint)((total + DefaultBlockSize - 1) / DefaultBlockSize);
        LaunchKernel(krnl, grid, DefaultBlockSize, args); Synchronize();
    }

    public unsafe void StdBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer mean, IGpuBuffer std, IGpuBuffer gradInput, int outerSize, int reduceSize)
    {
        if (!_kernelCache.TryGetValue("std_backward", out var krnl)) throw new InvalidOperationException("HIP kernel not found: std_backward");
        int total = outerSize * reduceSize;
        IntPtr _p0 = gradOutput.Handle, _p1 = input.Handle, _p2 = mean.Handle, _p3 = std.Handle, _p4 = gradInput.Handle;
        void** args = stackalloc void*[7]; args[0] = &_p0; args[1] = &_p1; args[2] = &_p2; args[3] = &_p3; args[4] = &_p4; args[5] = &outerSize; args[6] = &reduceSize;
        uint grid = (uint)((total + DefaultBlockSize - 1) / DefaultBlockSize);
        LaunchKernel(krnl, grid, DefaultBlockSize, args); Synchronize();
    }

    public unsafe void MaskedFillBackward(IGpuBuffer gradOutput, IGpuBuffer mask, IGpuBuffer gradInput, int size)
    {
        if (!_kernelCache.TryGetValue("masked_fill_backward", out var krnl)) throw new InvalidOperationException("HIP kernel not found: masked_fill_backward");
        IntPtr _p0 = gradOutput.Handle, _p1 = mask.Handle, _p2 = gradInput.Handle;
        void** args = stackalloc void*[4]; args[0] = &_p0; args[1] = &_p1; args[2] = &_p2; args[3] = &size;
        uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
        LaunchKernel(krnl, grid, DefaultBlockSize, args); Synchronize();
    }

    public unsafe void WhereBackward(IGpuBuffer gradOutput, IGpuBuffer condition, IGpuBuffer gradX, IGpuBuffer gradY, int size)
    {
        if (!_kernelCache.TryGetValue("where_backward", out var krnl)) throw new InvalidOperationException("HIP kernel not found: where_backward");
        IntPtr _p0 = gradOutput.Handle, _p1 = condition.Handle, _p2 = gradX.Handle, _p3 = gradY.Handle;
        void** args = stackalloc void*[5]; args[0] = &_p0; args[1] = &_p1; args[2] = &_p2; args[3] = &_p3; args[4] = &size;
        uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
        LaunchKernel(krnl, grid, DefaultBlockSize, args); Synchronize();
    }

    public unsafe void NormBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer norm, IGpuBuffer gradInput, int outerSize, int reduceSize)
    {
        if (!_kernelCache.TryGetValue("norm_backward", out var krnl)) throw new InvalidOperationException("HIP kernel not found: norm_backward");
        int total = outerSize * reduceSize;
        IntPtr _p0 = gradOutput.Handle, _p1 = input.Handle, _p2 = norm.Handle, _p3 = gradInput.Handle;
        void** args = stackalloc void*[6]; args[0] = &_p0; args[1] = &_p1; args[2] = &_p2; args[3] = &_p3; args[4] = &outerSize; args[5] = &reduceSize;
        uint grid = (uint)((total + DefaultBlockSize - 1) / DefaultBlockSize);
        LaunchKernel(krnl, grid, DefaultBlockSize, args); Synchronize();
    }

    public unsafe void LogSumExpBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer lse, IGpuBuffer gradInput, int outerSize, int reduceSize)
    {
        if (!_kernelCache.TryGetValue("logsumexp_backward", out var krnl)) throw new InvalidOperationException("HIP kernel not found: logsumexp_backward");
        int total = outerSize * reduceSize;
        IntPtr _p0 = gradOutput.Handle, _p1 = input.Handle, _p2 = lse.Handle, _p3 = gradInput.Handle;
        void** args = stackalloc void*[6]; args[0] = &_p0; args[1] = &_p1; args[2] = &_p2; args[3] = &_p3; args[4] = &outerSize; args[5] = &reduceSize;
        uint grid = (uint)((total + DefaultBlockSize - 1) / DefaultBlockSize);
        LaunchKernel(krnl, grid, DefaultBlockSize, args); Synchronize();
    }

    public unsafe void AvgPool1D(IGpuBuffer input, IGpuBuffer output, int batch, int channels, int inLength, int outLength, int kernelSize, int stride)
    {
        if (!_kernelCache.TryGetValue("avg_pool1d", out var krnl)) throw new InvalidOperationException("HIP kernel not found: avg_pool1d");
        int total = checked((int)((long)batch * channels * outLength));
        IntPtr _p0 = input.Handle, _p1 = output.Handle;
        void** args = stackalloc void*[8]; args[0] = &_p0; args[1] = &_p1; args[2] = &batch; args[3] = &channels; args[4] = &inLength; args[5] = &outLength; args[6] = &kernelSize; args[7] = &stride;
        uint grid = (uint)((total + DefaultBlockSize - 1) / DefaultBlockSize);
        LaunchKernel(krnl, grid, DefaultBlockSize, args); Synchronize();
    }

    public unsafe void MaxPool1D(IGpuBuffer input, IGpuBuffer output, int batch, int channels, int inLength, int outLength, int kernelSize, int stride)
    {
        if (!_kernelCache.TryGetValue("max_pool1d", out var krnl)) throw new InvalidOperationException("HIP kernel not found: max_pool1d");
        int total = checked((int)((long)batch * channels * outLength));
        IntPtr _p0 = input.Handle, _p1 = output.Handle;
        void** args = stackalloc void*[8]; args[0] = &_p0; args[1] = &_p1; args[2] = &batch; args[3] = &channels; args[4] = &inLength; args[5] = &outLength; args[6] = &kernelSize; args[7] = &stride;
        uint grid = (uint)((total + DefaultBlockSize - 1) / DefaultBlockSize);
        LaunchKernel(krnl, grid, DefaultBlockSize, args); Synchronize();
    }

    public unsafe void BilinearUpsample2D(IGpuBuffer input, IGpuBuffer output, int batch, int channels, int inH, int inW, int outH, int outW)
    {
        if (!_kernelCache.TryGetValue("bilinear_upsample2d", out var krnl)) throw new InvalidOperationException("HIP kernel not found: bilinear_upsample2d");
        int total = checked((int)((long)batch * channels * outH * outW));
        IntPtr _p0 = input.Handle, _p1 = output.Handle;
        void** args = stackalloc void*[8]; args[0] = &_p0; args[1] = &_p1; args[2] = &batch; args[3] = &channels; args[4] = &inH; args[5] = &inW; args[6] = &outH; args[7] = &outW;
        uint grid = (uint)((total + DefaultBlockSize - 1) / DefaultBlockSize);
        LaunchKernel(krnl, grid, DefaultBlockSize, args); Synchronize();
    }

    public unsafe void ScatterMean(IGpuBuffer source, IGpuBuffer indices, IGpuBuffer output, IGpuBuffer counts, int sourceSize, int outputSize, int featureSize)
    {
        // Initialize output and counts to zero before accumulation
        Fill(output, 0f, outputSize * featureSize);
        Fill(counts, 0f, outputSize);
        if (!_kernelCache.TryGetValue("scatter_mean", out var k1)) throw new InvalidOperationException("HIP kernel not found: scatter_mean");
        if (!_kernelCache.TryGetValue("scatter_mean_divide", out var k2)) throw new InvalidOperationException("HIP kernel not found: scatter_mean_divide");
        IntPtr sPtr = source.Handle, idxPtr = indices.Handle, oPtr = output.Handle, cPtr = counts.Handle;
        void** a1 = stackalloc void*[6]; a1[0] = &sPtr; a1[1] = &idxPtr; a1[2] = &oPtr; a1[3] = &cPtr; a1[4] = &sourceSize; a1[5] = &featureSize;
        LaunchKernel(k1, (uint)((sourceSize + DefaultBlockSize - 1) / DefaultBlockSize), DefaultBlockSize, a1);
        void** a2 = stackalloc void*[4]; a2[0] = &oPtr; a2[1] = &cPtr; a2[2] = &outputSize; a2[3] = &featureSize;
        LaunchKernel(k2, (uint)((outputSize + DefaultBlockSize - 1) / DefaultBlockSize), DefaultBlockSize, a2); Synchronize();
    }

    public unsafe void L1Loss(IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer loss, int batchSize, int numFeatures)
    {
        if (!_kernelCache.TryGetValue("l1_loss", out var krnl)) throw new InvalidOperationException("HIP kernel not found: l1_loss");
        IntPtr _p0 = predictions.Handle; IntPtr _p1 = targets.Handle; IntPtr _p2 = loss.Handle;
        void** args = stackalloc void*[5]; args[0] = &_p0; args[1] = &_p1; args[2] = &_p2; args[3] = &batchSize; args[4] = &numFeatures;
        uint grid = (uint)((batchSize + DefaultBlockSize - 1) / DefaultBlockSize);
        LaunchKernel(krnl, grid, DefaultBlockSize, args); Synchronize();
    }

    public unsafe void HuberLoss(IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer loss, int batchSize, int numFeatures, float delta)
    {
        if (!_kernelCache.TryGetValue("huber_loss", out var krnl)) throw new InvalidOperationException("HIP kernel not found: huber_loss");
        IntPtr _p0 = predictions.Handle; IntPtr _p1 = targets.Handle; IntPtr _p2 = loss.Handle;
        void** args = stackalloc void*[6]; args[0] = &_p0; args[1] = &_p1; args[2] = &_p2; args[3] = &batchSize; args[4] = &numFeatures; args[5] = &delta;
        uint grid = (uint)((batchSize + DefaultBlockSize - 1) / DefaultBlockSize);
        LaunchKernel(krnl, grid, DefaultBlockSize, args); Synchronize();
    }

    public unsafe void BceWithLogitsLoss(IGpuBuffer logits, IGpuBuffer targets, IGpuBuffer loss, int size)
    {
        if (!_kernelCache.TryGetValue("bce_with_logits_loss", out var krnl)) throw new InvalidOperationException("HIP kernel not found: bce_with_logits_loss");
        IntPtr _p0 = logits.Handle; IntPtr _p1 = targets.Handle; IntPtr _p2 = loss.Handle;
        void** args = stackalloc void*[4]; args[0] = &_p0; args[1] = &_p1; args[2] = &_p2; args[3] = &size;
        uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
        LaunchKernel(krnl, grid, DefaultBlockSize, args); Synchronize();
    }

    public unsafe void NllLoss(IGpuBuffer logProbs, IGpuBuffer targets, IGpuBuffer loss, int batchSize, int numClasses)
    {
        if (!_kernelCache.TryGetValue("nll_loss", out var krnl)) throw new InvalidOperationException("HIP kernel not found: nll_loss");
        IntPtr _p0 = logProbs.Handle; IntPtr _p1 = targets.Handle; IntPtr _p2 = loss.Handle;
        void** args = stackalloc void*[5]; args[0] = &_p0; args[1] = &_p1; args[2] = &_p2; args[3] = &batchSize; args[4] = &numClasses;
        uint grid = (uint)((batchSize + DefaultBlockSize - 1) / DefaultBlockSize);
        LaunchKernel(krnl, grid, DefaultBlockSize, args); Synchronize();
    }

    public unsafe void KlDivLoss(IGpuBuffer input, IGpuBuffer target, IGpuBuffer loss, int size)
    {
        if (!_kernelCache.TryGetValue("kl_div_loss", out var krnl)) throw new InvalidOperationException("HIP kernel not found: kl_div_loss");
        IntPtr _p0 = input.Handle; IntPtr _p1 = target.Handle; IntPtr _p2 = loss.Handle;
        void** args = stackalloc void*[4]; args[0] = &_p0; args[1] = &_p1; args[2] = &_p2; args[3] = &size;
        uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
        LaunchKernel(krnl, grid, DefaultBlockSize, args); Synchronize();
    }

    public unsafe void MseLossBackward(IGpuBuffer gradOutput, IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer gradInput, int size, float invN)
    {
        if (!_kernelCache.TryGetValue("mse_loss_backward", out var krnl)) throw new InvalidOperationException("HIP kernel not found: mse_loss_backward");
        IntPtr _p0 = gradOutput.Handle; IntPtr _p1 = predictions.Handle; IntPtr _p2 = targets.Handle; IntPtr _p3 = gradInput.Handle;
        void** args = stackalloc void*[6]; args[0] = &_p0; args[1] = &_p1; args[2] = &_p2; args[3] = &_p3; args[4] = &size; args[5] = &invN;
        uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
        LaunchKernel(krnl, grid, DefaultBlockSize, args); Synchronize();
    }

    public unsafe void L1LossBackward(IGpuBuffer gradOutput, IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer gradInput, int size, float invN)
    {
        if (!_kernelCache.TryGetValue("l1_loss_backward", out var krnl)) throw new InvalidOperationException("HIP kernel not found: l1_loss_backward");
        IntPtr _p0 = gradOutput.Handle; IntPtr _p1 = predictions.Handle; IntPtr _p2 = targets.Handle; IntPtr _p3 = gradInput.Handle;
        void** args = stackalloc void*[6]; args[0] = &_p0; args[1] = &_p1; args[2] = &_p2; args[3] = &_p3; args[4] = &size; args[5] = &invN;
        uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
        LaunchKernel(krnl, grid, DefaultBlockSize, args); Synchronize();
    }

    public unsafe void HuberLossBackward(IGpuBuffer gradOutput, IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer gradInput, int size, float invN, float delta)
    {
        if (!_kernelCache.TryGetValue("huber_loss_backward", out var krnl)) throw new InvalidOperationException("HIP kernel not found: huber_loss_backward");
        IntPtr _p0 = gradOutput.Handle; IntPtr _p1 = predictions.Handle; IntPtr _p2 = targets.Handle; IntPtr _p3 = gradInput.Handle;
        void** args = stackalloc void*[7]; args[0] = &_p0; args[1] = &_p1; args[2] = &_p2; args[3] = &_p3; args[4] = &size; args[5] = &invN; args[6] = &delta;
        uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
        LaunchKernel(krnl, grid, DefaultBlockSize, args); Synchronize();
    }

    public unsafe void BceWithLogitsBackward(IGpuBuffer gradOutput, IGpuBuffer logits, IGpuBuffer targets, IGpuBuffer gradInput, int size, float invN)
    {
        if (!_kernelCache.TryGetValue("bce_with_logits_backward", out var krnl)) throw new InvalidOperationException("HIP kernel not found: bce_with_logits_backward");
        IntPtr _p0 = gradOutput.Handle; IntPtr _p1 = logits.Handle; IntPtr _p2 = targets.Handle; IntPtr _p3 = gradInput.Handle;
        void** args = stackalloc void*[6]; args[0] = &_p0; args[1] = &_p1; args[2] = &_p2; args[3] = &_p3; args[4] = &size; args[5] = &invN;
        uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
        LaunchKernel(krnl, grid, DefaultBlockSize, args); Synchronize();
    }

    #region StopGradient, Fused Linear, and IoU Operations

    public void CopyBuffer(IGpuBuffer source, IGpuBuffer destination, int size)
    {
        if (size <= 0) return;
        var result = HipNativeBindings.hipMemcpy(destination.Handle, source.Handle, (nuint)(size * sizeof(float)), HipMemcpyKind.DeviceToDevice);
        HipNativeBindings.CheckError(result, "hipMemcpy CopyBuffer");
    }

    public unsafe void FusedLinearReLU(IGpuBuffer input, IGpuBuffer weight, IGpuBuffer bias, IGpuBuffer output, int batchSize, int inFeatures, int outFeatures) { LaunchFusedLinear("fused_linear_relu", input, weight, bias, output, batchSize, inFeatures, outFeatures); }
    public unsafe void FusedLinearSigmoid(IGpuBuffer input, IGpuBuffer weight, IGpuBuffer bias, IGpuBuffer output, int batchSize, int inFeatures, int outFeatures) { LaunchFusedLinear("fused_linear_sigmoid", input, weight, bias, output, batchSize, inFeatures, outFeatures); }
    public unsafe void FusedLinearTanh(IGpuBuffer input, IGpuBuffer weight, IGpuBuffer bias, IGpuBuffer output, int batchSize, int inFeatures, int outFeatures) { LaunchFusedLinear("fused_linear_tanh", input, weight, bias, output, batchSize, inFeatures, outFeatures); }
    public unsafe void FusedLinearGELU(IGpuBuffer input, IGpuBuffer weight, IGpuBuffer bias, IGpuBuffer output, int batchSize, int inFeatures, int outFeatures) { LaunchFusedLinear("fused_linear_gelu", input, weight, bias, output, batchSize, inFeatures, outFeatures); }
    public unsafe void FusedLinearSwish(IGpuBuffer input, IGpuBuffer weight, IGpuBuffer bias, IGpuBuffer output, int batchSize, int inFeatures, int outFeatures) { LaunchFusedLinear("fused_linear_swish", input, weight, bias, output, batchSize, inFeatures, outFeatures); }
    public unsafe void FusedLinearReLUBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer weight, IGpuBuffer preActivation, IGpuBuffer gradInput, IGpuBuffer gradWeight, IGpuBuffer gradBias, int batchSize, int inFeatures, int outFeatures) { LaunchFusedLinearBackwardHip("fused_linear_relu_backward_grad_input", gradOutput, input, weight, preActivation, gradInput, gradWeight, gradBias, batchSize, inFeatures, outFeatures, 0); }
    public unsafe void FusedLinearSigmoidBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer weight, IGpuBuffer output, IGpuBuffer gradInput, IGpuBuffer gradWeight, IGpuBuffer gradBias, int batchSize, int inFeatures, int outFeatures) { LaunchFusedLinearBackwardHip("fused_linear_sigmoid_backward_grad_input", gradOutput, input, weight, output, gradInput, gradWeight, gradBias, batchSize, inFeatures, outFeatures, 1); }
    public unsafe void FusedLinearTanhBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer weight, IGpuBuffer output, IGpuBuffer gradInput, IGpuBuffer gradWeight, IGpuBuffer gradBias, int batchSize, int inFeatures, int outFeatures) { LaunchFusedLinearBackwardHip("fused_linear_tanh_backward_grad_input", gradOutput, input, weight, output, gradInput, gradWeight, gradBias, batchSize, inFeatures, outFeatures, 2); }
    public unsafe void FusedLinearGELUBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer weight, IGpuBuffer preActivation, IGpuBuffer gradInput, IGpuBuffer gradWeight, IGpuBuffer gradBias, int batchSize, int inFeatures, int outFeatures) { LaunchFusedLinearBackwardHip("fused_linear_gelu_backward_grad_input", gradOutput, input, weight, preActivation, gradInput, gradWeight, gradBias, batchSize, inFeatures, outFeatures, 3); }
    public unsafe void FusedLinearSwishBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer weight, IGpuBuffer preActivation, IGpuBuffer gradInput, IGpuBuffer gradWeight, IGpuBuffer gradBias, int batchSize, int inFeatures, int outFeatures) { LaunchFusedLinearBackwardHip("fused_linear_swish_backward_grad_input", gradOutput, input, weight, preActivation, gradInput, gradWeight, gradBias, batchSize, inFeatures, outFeatures, 4); }
    public unsafe void IoULoss(IGpuBuffer predicted, IGpuBuffer target, IGpuBuffer loss, int numBoxes) { LaunchIoU("iou_loss", predicted, target, loss, numBoxes); }
    public unsafe void GIoULoss(IGpuBuffer predicted, IGpuBuffer target, IGpuBuffer loss, int numBoxes) { LaunchIoU("giou_loss", predicted, target, loss, numBoxes); }
    public unsafe void DIoULoss(IGpuBuffer predicted, IGpuBuffer target, IGpuBuffer loss, int numBoxes) { LaunchIoU("diou_loss", predicted, target, loss, numBoxes); }
    public unsafe void CIoULoss(IGpuBuffer predicted, IGpuBuffer target, IGpuBuffer loss, int numBoxes) { LaunchIoU("ciou_loss", predicted, target, loss, numBoxes); }
    public unsafe void IoULossBackward(IGpuBuffer gradOutput, IGpuBuffer predicted, IGpuBuffer target, IGpuBuffer gradPredicted, int numBoxes) { LaunchIoUBackward("iou_loss_backward", gradOutput, predicted, target, gradPredicted, numBoxes); }
    public unsafe void GIoULossBackward(IGpuBuffer gradOutput, IGpuBuffer predicted, IGpuBuffer target, IGpuBuffer gradPredicted, int numBoxes) { LaunchIoUBackward("giou_loss_backward", gradOutput, predicted, target, gradPredicted, numBoxes); }
    public unsafe void DIoULossBackward(IGpuBuffer gradOutput, IGpuBuffer predicted, IGpuBuffer target, IGpuBuffer gradPredicted, int numBoxes) { LaunchIoUBackward("diou_loss_backward", gradOutput, predicted, target, gradPredicted, numBoxes); }
    public unsafe void CIoULossBackward(IGpuBuffer gradOutput, IGpuBuffer predicted, IGpuBuffer target, IGpuBuffer gradPredicted, int numBoxes) { LaunchIoUBackward("ciou_loss_backward", gradOutput, predicted, target, gradPredicted, numBoxes); }

    private unsafe void LaunchFusedLinear(string kernelName, IGpuBuffer input, IGpuBuffer weight, IGpuBuffer bias, IGpuBuffer output, int batchSize, int inFeatures, int outFeatures)
    {
        if (!_kernelCache.TryGetValue(kernelName, out var krnl)) throw new InvalidOperationException($"HIP kernel not found: {kernelName}");
        int total = batchSize * outFeatures;
        IntPtr iPtr = input.Handle, wPtr = weight.Handle, bPtr = bias.Handle, oPtr = output.Handle;
        void** args = stackalloc void*[7]; args[0] = &iPtr; args[1] = &wPtr; args[2] = &bPtr; args[3] = &oPtr;
        args[4] = &batchSize; args[5] = &inFeatures; args[6] = &outFeatures;
        LaunchKernel(krnl, (uint)((total + DefaultBlockSize - 1) / DefaultBlockSize), DefaultBlockSize, args); Synchronize();
    }

    private unsafe void LaunchIoU(string kernelName, IGpuBuffer predicted, IGpuBuffer target, IGpuBuffer loss, int numBoxes)
    {
        if (!_kernelCache.TryGetValue(kernelName, out var krnl)) throw new InvalidOperationException($"HIP kernel not found: {kernelName}");
        IntPtr pPtr = predicted.Handle, tPtr = target.Handle, lPtr = loss.Handle;
        void** args = stackalloc void*[4]; args[0] = &pPtr; args[1] = &tPtr; args[2] = &lPtr; args[3] = &numBoxes;
        LaunchKernel(krnl, (uint)((numBoxes + DefaultBlockSize - 1) / DefaultBlockSize), DefaultBlockSize, args); Synchronize();
    }

    private unsafe void LaunchIoUBackward(string kernelName, IGpuBuffer gradOutput, IGpuBuffer predicted, IGpuBuffer target, IGpuBuffer gradPredicted, int numBoxes)
    {
        if (!_kernelCache.TryGetValue(kernelName, out var krnl)) throw new InvalidOperationException($"HIP kernel not found: {kernelName}");
        IntPtr goPtr = gradOutput.Handle, pPtr = predicted.Handle, tPtr = target.Handle, gpPtr = gradPredicted.Handle;
        void** args = stackalloc void*[5]; args[0] = &goPtr; args[1] = &pPtr; args[2] = &tPtr; args[3] = &gpPtr; args[4] = &numBoxes;
        LaunchKernel(krnl, (uint)((numBoxes + DefaultBlockSize - 1) / DefaultBlockSize), DefaultBlockSize, args); Synchronize();
    }

    private unsafe void LaunchFusedLinearBackwardHip(string gradInputKernel, IGpuBuffer gradOutput, IGpuBuffer input,
        IGpuBuffer weight, IGpuBuffer saved, IGpuBuffer gradInput, IGpuBuffer gradWeight, IGpuBuffer gradBias,
        int batchSize, int inFeatures, int outFeatures, int activationType)
    {
        // Kernel 1: grad_input
        if (!_kernelCache.TryGetValue(gradInputKernel, out var giKernel))
            throw new InvalidOperationException($"HIP kernel not found: {gradInputKernel}");
        int totalIn = batchSize * inFeatures;
        IntPtr goPtr = gradOutput.Handle, wPtr = weight.Handle, sPtr = saved.Handle, giPtr = gradInput.Handle;
        void** argsGI = stackalloc void*[7];
        argsGI[0] = &goPtr; argsGI[1] = &wPtr; argsGI[2] = &sPtr; argsGI[3] = &giPtr;
        argsGI[4] = &batchSize; argsGI[5] = &inFeatures; argsGI[6] = &outFeatures;
        LaunchKernel(giKernel, (uint)((totalIn + DefaultBlockSize - 1) / DefaultBlockSize), DefaultBlockSize, argsGI);

        // Kernel 2: weight gradient
        if (!_kernelCache.TryGetValue("fused_linear_weight_grad", out var wgKernel))
            throw new InvalidOperationException("HIP kernel not found: fused_linear_weight_grad");
        int totalW = inFeatures * outFeatures;
        IntPtr iPtr = input.Handle, gwPtr = gradWeight.Handle;
        void** argsWG = stackalloc void*[8];
        argsWG[0] = &goPtr; argsWG[1] = &iPtr; argsWG[2] = &sPtr; argsWG[3] = &gwPtr;
        argsWG[4] = &batchSize; argsWG[5] = &inFeatures; argsWG[6] = &outFeatures; argsWG[7] = &activationType;
        LaunchKernel(wgKernel, (uint)((totalW + DefaultBlockSize - 1) / DefaultBlockSize), DefaultBlockSize, argsWG);

        // Kernel 3: bias gradient
        if (!_kernelCache.TryGetValue("fused_linear_bias_grad", out var bgKernel))
            throw new InvalidOperationException("HIP kernel not found: fused_linear_bias_grad");
        IntPtr gbPtr = gradBias.Handle;
        void** argsBG = stackalloc void*[6];
        argsBG[0] = &goPtr; argsBG[1] = &sPtr; argsBG[2] = &gbPtr;
        argsBG[3] = &batchSize; argsBG[4] = &outFeatures; argsBG[5] = &activationType;
        LaunchKernel(bgKernel, (uint)((outFeatures + DefaultBlockSize - 1) / DefaultBlockSize), DefaultBlockSize, argsBG);
        Synchronize();
    }

    #endregion

    #region Trigonometric Operations

    public void Sin(IGpuBuffer A, IGpuBuffer B, int size)
    {
        if (!_kernelCache.TryGetValue("sin_vector", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: sin_vector");

        LaunchUnaryOp(krnl, A, B, size);
    }

    public void Cos(IGpuBuffer A, IGpuBuffer B, int size)
    {
        if (!_kernelCache.TryGetValue("cos_vector", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: cos_vector");

        LaunchUnaryOp(krnl, A, B, size);
    }

    public void Tan(IGpuBuffer A, IGpuBuffer B, int size)
    {
        if (!_kernelCache.TryGetValue("tan_vector", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: tan_vector");

        LaunchUnaryOp(krnl, A, B, size);
    }

    public void Asin(IGpuBuffer A, IGpuBuffer B, int size)
    {
        if (!_kernelCache.TryGetValue("asin_vector", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: asin_vector");

        LaunchUnaryOp(krnl, A, B, size);
    }

    public void Acos(IGpuBuffer A, IGpuBuffer B, int size)
    {
        if (!_kernelCache.TryGetValue("acos_vector", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: acos_vector");

        LaunchUnaryOp(krnl, A, B, size);
    }

    public void Atan(IGpuBuffer A, IGpuBuffer B, int size)
    {
        if (!_kernelCache.TryGetValue("atan_vector", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: atan_vector");

        LaunchUnaryOp(krnl, A, B, size);
    }

    #endregion

    #region Hyperbolic Operations

    public void Sinh(IGpuBuffer A, IGpuBuffer B, int size)
    {
        if (!_kernelCache.TryGetValue("sinh_vector", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: sinh_vector");

        LaunchUnaryOp(krnl, A, B, size);
    }

    public void Cosh(IGpuBuffer A, IGpuBuffer B, int size)
    {
        if (!_kernelCache.TryGetValue("cosh_vector", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: cosh_vector");

        LaunchUnaryOp(krnl, A, B, size);
    }

    public void Asinh(IGpuBuffer A, IGpuBuffer B, int size)
    {
        if (!_kernelCache.TryGetValue("asinh_vector", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: asinh_vector");

        LaunchUnaryOp(krnl, A, B, size);
    }

    public void Acosh(IGpuBuffer A, IGpuBuffer B, int size)
    {
        if (!_kernelCache.TryGetValue("acosh_vector", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: acosh_vector");

        LaunchUnaryOp(krnl, A, B, size);
    }

    public void Atanh(IGpuBuffer A, IGpuBuffer B, int size)
    {
        if (!_kernelCache.TryGetValue("atanh_vector", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: atanh_vector");

        LaunchUnaryOp(krnl, A, B, size);
    }

    #endregion

    #region Additional Unary Operations

    public void Reciprocal(IGpuBuffer A, IGpuBuffer B, int size)
    {
        if (!_kernelCache.TryGetValue("reciprocal_vector", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: reciprocal_vector");

        LaunchUnaryOp(krnl, A, B, size);
    }

    public void Cbrt(IGpuBuffer A, IGpuBuffer B, int size)
    {
        if (!_kernelCache.TryGetValue("cbrt_vector", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: cbrt_vector");

        LaunchUnaryOp(krnl, A, B, size);
    }

    public void Log10(IGpuBuffer A, IGpuBuffer B, int size)
    {
        if (!_kernelCache.TryGetValue("log10_vector", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: log10_vector");

        LaunchUnaryOp(krnl, A, B, size);
    }

    public void Negate(IGpuBuffer A, IGpuBuffer B, int size)
    {
        LaunchUnaryOpAutoVec4("negate_vector", A, B, size);
    }

    public void Floor(IGpuBuffer A, IGpuBuffer B, int size)
    {
        if (!_kernelCache.TryGetValue("floor_vector", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: floor_vector");

        LaunchUnaryOp(krnl, A, B, size);
    }

    public void Ceiling(IGpuBuffer A, IGpuBuffer B, int size)
    {
        if (!_kernelCache.TryGetValue("ceil_vector", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: ceil_vector");

        LaunchUnaryOp(krnl, A, B, size);
    }

    public void Round(IGpuBuffer A, IGpuBuffer B, int size)
    {
        if (!_kernelCache.TryGetValue("round_vector", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: round_vector");

        LaunchUnaryOp(krnl, A, B, size);
    }

    public void Truncate(IGpuBuffer A, IGpuBuffer B, int size)
    {
        if (!_kernelCache.TryGetValue("trunc_vector", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: trunc_vector");

        LaunchUnaryOp(krnl, A, B, size);
    }

    #endregion

    public unsafe void Softmax(IGpuBuffer A, IGpuBuffer B, int batchSize, int features)
    {
        if (!_kernelCache.TryGetValue("softmax", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: softmax");

        // 1 block per batch element, 256 threads cooperate on parallel reduction
        // Shared memory: ceil(256/64) = 4 wavefronts * sizeof(float) = 16 bytes
        uint sharedBytes = (DefaultBlockSize / 64 + 1) * sizeof(float);

            {
            IntPtr _p0 = A.Handle;
            IntPtr _p1 = B.Handle;
            void** args = stackalloc void*[4];
            args[0] = &_p0;
            args[1] = &_p1;
            args[2] = &batchSize;
            args[3] = &features;


            uint grid = (uint)batchSize;
            LaunchKernel(krnl, grid, DefaultBlockSize, args, sharedBytes);
            Synchronize();
            }
    }

    public unsafe void Squash(IGpuBuffer input, IGpuBuffer output, int numCapsules, int capsuleDim, float epsilon)
    {
        if (!_kernelCache.TryGetValue("squash", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: squash");

            {
            IntPtr _p0 = input.Handle;
            IntPtr _p1 = output.Handle;
            void** args = stackalloc void*[5];
            args[0] = &_p0;
            args[1] = &_p1;
            args[2] = &numCapsules;
            args[3] = &capsuleDim;
            args[4] = &epsilon;


            uint grid = (uint)((numCapsules + DefaultBlockSize - 1) / DefaultBlockSize);
            LaunchKernel(krnl, grid, DefaultBlockSize, args);
            Synchronize();
            }
    }

    public unsafe void SquashBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradInput, int numCapsules, int capsuleDim, float epsilon)
    {
        if (!_kernelCache.TryGetValue("squash_backward", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: squash_backward");

            {
            IntPtr _p0 = gradOutput.Handle;
            IntPtr _p1 = input.Handle;
            IntPtr _p2 = gradInput.Handle;
            void** args = stackalloc void*[6];
            args[0] = &_p0;
            args[1] = &_p1;
            args[2] = &_p2;
            args[3] = &numCapsules;
            args[4] = &capsuleDim;
            args[5] = &epsilon;


            uint grid = (uint)((numCapsules + DefaultBlockSize - 1) / DefaultBlockSize);
            LaunchKernel(krnl, grid, DefaultBlockSize, args);
            Synchronize();
            }
    }

    public unsafe void CapsulePredictions(IGpuBuffer input, IGpuBuffer weights, IGpuBuffer output,
        int batchSize, int inputCapsules, int inputDim, int outputCapsules, int outputDim)
    {
        if (!_kernelCache.TryGetValue("capsule_predictions", out var kernel))
            throw new InvalidOperationException("HIP kernel not found: capsule_predictions");

        IntPtr inPtr = input.Handle, wPtr = weights.Handle, outPtr = output.Handle;
        void** args = stackalloc void*[8];
        args[0] = &inPtr; args[1] = &wPtr; args[2] = &outPtr;
        args[3] = &batchSize; args[4] = &inputCapsules; args[5] = &inputDim;
        args[6] = &outputCapsules; args[7] = &outputDim;
        uint total = (uint)(batchSize * inputCapsules * outputCapsules * outputDim);
        LaunchKernel(kernel, (total + DefaultBlockSize - 1) / DefaultBlockSize, DefaultBlockSize, args);
    }

    public unsafe void CapsuleTransform(IGpuBuffer input, IGpuBuffer weights, IGpuBuffer output,
        int batchSize, int inputCapsules, int inputDim, int numCapsules, int capsuleDim)
    {
        if (!_kernelCache.TryGetValue("capsule_transform", out var kernel))
            throw new InvalidOperationException("HIP kernel not found: capsule_transform");

        IntPtr inPtr = input.Handle, wPtr = weights.Handle, outPtr = output.Handle;
        void** args = stackalloc void*[8];
        args[0] = &inPtr; args[1] = &wPtr; args[2] = &outPtr;
        args[3] = &batchSize; args[4] = &inputCapsules; args[5] = &inputDim;
        args[6] = &numCapsules; args[7] = &capsuleDim;
        uint total = (uint)(batchSize * inputCapsules * numCapsules * capsuleDim);
        LaunchKernel(kernel, (total + DefaultBlockSize - 1) / DefaultBlockSize, DefaultBlockSize, args);
    }

    public unsafe void CapsuleWeightedSum(IGpuBuffer coupling, IGpuBuffer predictions, IGpuBuffer output,
        int batchSize, int inputCapsules, int outputCapsules, int capsuleDim)
    {
        if (!_kernelCache.TryGetValue("capsule_weighted_sum", out var kernel))
            throw new InvalidOperationException("HIP kernel not found: capsule_weighted_sum");

        IntPtr cPtr = coupling.Handle, pPtr = predictions.Handle, outPtr = output.Handle;
        void** args = stackalloc void*[7];
        args[0] = &cPtr; args[1] = &pPtr; args[2] = &outPtr;
        args[3] = &batchSize; args[4] = &inputCapsules; args[5] = &outputCapsules; args[6] = &capsuleDim;
        uint total = (uint)(batchSize * outputCapsules * capsuleDim);
        LaunchKernel(kernel, (total + DefaultBlockSize - 1) / DefaultBlockSize, DefaultBlockSize, args);
    }

    public unsafe void CapsuleAgreement(IGpuBuffer predictions, IGpuBuffer output, IGpuBuffer agreement,
        int batchSize, int inputCapsules, int outputCapsules, int capsuleDim)
    {
        if (!_kernelCache.TryGetValue("capsule_agreement", out var kernel))
            throw new InvalidOperationException("HIP kernel not found: capsule_agreement");

        IntPtr pPtr = predictions.Handle, outPtr = output.Handle, agPtr = agreement.Handle;
        void** args = stackalloc void*[7];
        args[0] = &pPtr; args[1] = &outPtr; args[2] = &agPtr;
        args[3] = &batchSize; args[4] = &inputCapsules; args[5] = &outputCapsules; args[6] = &capsuleDim;
        uint total = (uint)(batchSize * inputCapsules * outputCapsules);
        LaunchKernel(kernel, (total + DefaultBlockSize - 1) / DefaultBlockSize, DefaultBlockSize, args);
    }

    public unsafe void TileBatch(IGpuBuffer input, IGpuBuffer output, int repeats, int innerSize)
    {
        if (!_kernelCache.TryGetValue("tile_batch", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: tile_batch");

        int totalSize = repeats * innerSize;
            {
            IntPtr _p0 = input.Handle;
            IntPtr _p1 = output.Handle;
            void** args = stackalloc void*[4];
            args[0] = &_p0;
            args[1] = &_p1;
            args[2] = &repeats;
            args[3] = &innerSize;


            uint grid = (uint)((totalSize + DefaultBlockSize - 1) / DefaultBlockSize);
            LaunchKernel(krnl, grid, DefaultBlockSize, args);
            Synchronize();
            }
    }

    public unsafe void TileAxis(IGpuBuffer input, IGpuBuffer output, int outerSize, int axisSize, int innerSize, int repeats)
    {
        if (!_kernelCache.TryGetValue("tile_axis", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: tile_axis");

        int totalSize = outerSize * axisSize * repeats * innerSize;
            {
            IntPtr _p0 = input.Handle;
            IntPtr _p1 = output.Handle;
            void** args = stackalloc void*[6];
            args[0] = &_p0;
            args[1] = &_p1;
            args[2] = &outerSize;
            args[3] = &axisSize;
            args[4] = &innerSize;
            args[5] = &repeats;


            uint grid = (uint)((totalSize + DefaultBlockSize - 1) / DefaultBlockSize);
            LaunchKernel(krnl, grid, DefaultBlockSize, args);
            Synchronize();
            }
    }

    private void UploadToBuffer(IGpuBuffer buffer, float[] data)
    {
        var hipBuffer = (HipGpuBuffer)buffer;
        var size = (UIntPtr)(data.Length * sizeof(float));

        GCHandle handle = GCHandle.Alloc(data, GCHandleType.Pinned);
        try
        {
            var result = HipNativeBindings.hipMemcpy(
                hipBuffer.Handle,
                handle.AddrOfPinnedObject(),
                size,
                HipMemcpyKind.HostToDevice);
            HipNativeBindings.CheckError(result, "hipMemcpy H2D");
        }
        finally
        {
            handle.Free();
        }
    }

    #endregion

    #region Sparse Operations (2:4 Structured Sparsity)

    public IGpuBuffer AllocateByteBuffer(int size)
    {
        IntPtr devicePtr = IntPtr.Zero;
        var sizeBytes = (UIntPtr)size;

        var result = HipNativeBindings.hipMalloc(ref devicePtr, sizeBytes);
        HipNativeBindings.CheckError(result, "hipMalloc (byte buffer)");

        // Zero-initialize
        result = HipNativeBindings.hipMemset(devicePtr, 0, sizeBytes);
        HipNativeBindings.CheckError(result, "hipMemset (byte buffer)");

        return new HipGpuByteBuffer(devicePtr, size);
    }

    public void Enforce2x4Sparsity(IGpuBuffer denseInput, IGpuBuffer sparseValues, IGpuBuffer sparseIndices, int M, int K)
    {
        if (K % 4 != 0)
            throw new ArgumentException("K must be divisible by 4 for 2:4 sparsity", nameof(K));

        // Download dense data
        var denseData = DownloadBuffer(denseInput);

        // Use SparsityUtils for CPU-side sparsity enforcement
        var compressed = SparsityUtils.CompressTo2x4(denseData, M, K);

        // Upload compressed values
        UploadToBuffer(sparseValues, compressed.Values);

        // Upload indices
        var byteBuffer = (HipGpuByteBuffer)sparseIndices;
        UploadBytesToBuffer(byteBuffer, compressed.Indices);
    }

    public void Decompress2x4Sparse(IGpuBuffer sparseValues, IGpuBuffer sparseIndices, IGpuBuffer denseOutput, int M, int K)
    {
        if (K % 4 != 0)
            throw new ArgumentException("K must be divisible by 4 for 2:4 sparsity", nameof(K));

        // Download compressed data
        var values = DownloadBuffer(sparseValues);
        var indices = DownloadBytesFromBuffer((HipGpuByteBuffer)sparseIndices);

        // Create compressed representation
        var compressed = new Compressed2x4Sparse(values, indices, M, K);

        // Decompress
        var denseData = SparsityUtils.DecompressFrom2x4(compressed);

        // Upload result
        UploadToBuffer(denseOutput, denseData);
    }

    public void SparseGemm(
        IGpuBuffer sparseAValues, IGpuBuffer sparseAIndices,
        IGpuBuffer B, IGpuBuffer C,
        int M, int N, int K,
        float alpha = 1.0f, float beta = 0.0f)
    {
        if (K % 4 != 0)
            throw new ArgumentException("K must be divisible by 4 for 2:4 sparsity", nameof(K));

        // Download sparse A data
        var aValues = DownloadBuffer(sparseAValues);
        var aIndices = DownloadBytesFromBuffer((HipGpuByteBuffer)sparseAIndices);

        // Create compressed representation
        var sparseA = new Compressed2x4Sparse(aValues, aIndices, M, K);

        // Download B and existing C
        var bData = DownloadBuffer(B);
        var cData = DownloadBuffer(C);

        // Execute sparse GEMM on CPU (later can be HIP-accelerated)
        SparsityUtils.SparseGemmCpu(sparseA, bData, cData, N, alpha, beta);

        // Upload result
        UploadToBuffer(C, cData);
    }

    public IGpuBuffer SparseGemmBiasRelu(
        IGpuBuffer sparseAValues, IGpuBuffer sparseAIndices,
        IGpuBuffer B, IGpuBuffer bias,
        int M, int N, int K)
    {
        if (K % 4 != 0)
            throw new ArgumentException("K must be divisible by 4 for 2:4 sparsity", nameof(K));

        // Create output buffer
        var C = AllocateBuffer(M * N);

        // Execute sparse GEMM
        SparseGemm(sparseAValues, sparseAIndices, B, C, M, N, K, 1.0f, 0.0f);

        // Download, apply bias + ReLU
        var cData = DownloadBuffer(C);
        var biasData = DownloadBuffer(bias);

        for (int row = 0; row < M; row++)
        {
            for (int col = 0; col < N; col++)
            {
                int idx = row * N + col;
                float val = cData[idx] + biasData[col];
                cData[idx] = Math.Max(0, val);
            }
        }

        // Upload back
        UploadToBuffer(C, cData);
        return C;
    }

    #region CSR Sparse Operations (General Sparsity)

    /// <inheritdoc/>
    public void CsrSpMM(
        IGpuBuffer csrValues,
        IGpuBuffer csrColIndices,
        IGpuBuffer csrRowPointers,
        IGpuBuffer denseB,
        IGpuBuffer output,
        int M, int K, int N, int nnz)
    {
        if (!IsAvailable)
            throw new InvalidOperationException("HIP backend is not available.");

        if (!_kernelCache.TryGetValue("csr_spmm", out var kernel))
            throw new InvalidOperationException("HIP kernel not found: csr_spmm");

        // Launch configuration: rows x ceil(N/blockSize) grid
        int gridX = M;
        int gridY = (N + DefaultBlockSize - 1) / DefaultBlockSize;

        var valuesHandle = ((HipGpuBuffer)csrValues).Handle;
        var colIndicesHandle = ((HipGpuBuffer)csrColIndices).Handle;
        var rowPointersHandle = ((HipGpuBuffer)csrRowPointers).Handle;
        var denseBHandle = ((HipGpuBuffer)denseB).Handle;
        var outputHandle = ((HipGpuBuffer)output).Handle;

        LaunchKernel2D(kernel, gridX, gridY, DefaultBlockSize, 1,
            valuesHandle, colIndicesHandle, rowPointersHandle, denseBHandle, outputHandle,
            M, K, N, nnz);
    }

    /// <inheritdoc/>
    public void CsrSpMMBias(
        IGpuBuffer csrValues,
        IGpuBuffer csrColIndices,
        IGpuBuffer csrRowPointers,
        IGpuBuffer denseB,
        IGpuBuffer bias,
        IGpuBuffer output,
        int M, int K, int N, int nnz)
    {
        if (!IsAvailable)
            throw new InvalidOperationException("HIP backend is not available.");

        if (!_kernelCache.TryGetValue("csr_spmm_bias", out var kernel))
            throw new InvalidOperationException("HIP kernel not found: csr_spmm_bias");

        int gridX = M;
        int gridY = (N + DefaultBlockSize - 1) / DefaultBlockSize;

        var valuesHandle = ((HipGpuBuffer)csrValues).Handle;
        var colIndicesHandle = ((HipGpuBuffer)csrColIndices).Handle;
        var rowPointersHandle = ((HipGpuBuffer)csrRowPointers).Handle;
        var denseBHandle = ((HipGpuBuffer)denseB).Handle;
        var biasHandle = ((HipGpuBuffer)bias).Handle;
        var outputHandle = ((HipGpuBuffer)output).Handle;

        LaunchKernel2DBias(kernel, gridX, gridY, DefaultBlockSize, 1,
            valuesHandle, colIndicesHandle, rowPointersHandle, denseBHandle, biasHandle, outputHandle,
            M, K, N, nnz);
    }

    /// <inheritdoc/>
    public void ScatterAddEdges(
        IGpuBuffer input,
        IGpuBuffer sourceIndices,
        IGpuBuffer targetIndices,
        IGpuBuffer? edgeValues,
        IGpuBuffer output,
        int numNodes, int numEdges, int features)
    {
        if (!IsAvailable)
            throw new InvalidOperationException("HIP backend is not available.");

        if (!_kernelCache.TryGetValue("scatter_add_edges", out var kernel))
            throw new InvalidOperationException("HIP kernel not found: scatter_add_edges");

        if (!_kernelCache.TryGetValue("zero_buffer", out var zeroKernel))
            throw new InvalidOperationException("HIP kernel not found: zero_buffer");

        // First zero the output buffer
        int outputSize = numNodes * features;
        int zeroGrid = (outputSize + DefaultBlockSize - 1) / DefaultBlockSize;
        var outputHandle = ((HipGpuBuffer)output).Handle;
        LaunchKernel1D(zeroKernel, zeroGrid, DefaultBlockSize, outputHandle, outputSize);
        Synchronize();

        // Launch scatter-add kernel
        int gridX = numEdges;
        int gridY = (features + DefaultBlockSize - 1) / DefaultBlockSize;

        var inputHandle = ((HipGpuBuffer)input).Handle;
        var sourceHandle = ((HipGpuBuffer)sourceIndices).Handle;
        var targetHandle = ((HipGpuBuffer)targetIndices).Handle;
        var edgeValuesHandle = edgeValues is not null ? ((HipGpuBuffer)edgeValues).Handle : IntPtr.Zero;
        int hasEdgeValues = edgeValues is not null ? 1 : 0;

        LaunchScatterAddKernel(kernel, gridX, gridY, DefaultBlockSize, 1,
            inputHandle, sourceHandle, targetHandle, edgeValuesHandle, outputHandle,
            numNodes, numEdges, features, hasEdgeValues);
    }

    /// <inheritdoc/>
    public void CsrSegmentedMax(
        IGpuBuffer csrColIndices,
        IGpuBuffer csrRowPointers,
        IGpuBuffer input,
        IGpuBuffer output,
        int M, int K, int N)
    {
        if (!IsAvailable)
            throw new InvalidOperationException("HIP backend is not available.");

        if (!_kernelCache.TryGetValue("csr_segmented_max", out var kernel))
            throw new InvalidOperationException("HIP kernel not found: csr_segmented_max");

        int gridX = M;
        int gridY = (N + DefaultBlockSize - 1) / DefaultBlockSize;

        var colIndicesHandle = ((HipGpuBuffer)csrColIndices).Handle;
        var rowPointersHandle = ((HipGpuBuffer)csrRowPointers).Handle;
        var inputHandle = ((HipGpuBuffer)input).Handle;
        var outputHandle = ((HipGpuBuffer)output).Handle;

        LaunchCsrSegmentedKernel(kernel, gridX, gridY, DefaultBlockSize, 1,
            colIndicesHandle, rowPointersHandle, inputHandle, outputHandle, M, K, N);
    }

    /// <inheritdoc/>
    public void CsrSegmentedMin(
        IGpuBuffer csrColIndices,
        IGpuBuffer csrRowPointers,
        IGpuBuffer input,
        IGpuBuffer output,
        int M, int K, int N)
    {
        if (!IsAvailable)
            throw new InvalidOperationException("HIP backend is not available.");

        if (!_kernelCache.TryGetValue("csr_segmented_min", out var kernel))
            throw new InvalidOperationException("HIP kernel not found: csr_segmented_min");

        int gridX = M;
        int gridY = (N + DefaultBlockSize - 1) / DefaultBlockSize;

        var colIndicesHandle = ((HipGpuBuffer)csrColIndices).Handle;
        var rowPointersHandle = ((HipGpuBuffer)csrRowPointers).Handle;
        var inputHandle = ((HipGpuBuffer)input).Handle;
        var outputHandle = ((HipGpuBuffer)output).Handle;

        LaunchCsrSegmentedKernel(kernel, gridX, gridY, DefaultBlockSize, 1,
            colIndicesHandle, rowPointersHandle, inputHandle, outputHandle, M, K, N);
    }

    /// <inheritdoc/>
    public void CsrSegmentedStdDev(
        IGpuBuffer csrColIndices,
        IGpuBuffer csrRowPointers,
        IGpuBuffer input,
        IGpuBuffer output,
        int M, int K, int N,
        float epsilon = 1e-8f)
    {
        if (!IsAvailable)
            throw new InvalidOperationException("HIP backend is not available.");

        if (!_kernelCache.TryGetValue("csr_segmented_stddev", out var kernel))
            throw new InvalidOperationException("HIP kernel not found: csr_segmented_stddev");

        int gridX = M;
        int gridY = (N + DefaultBlockSize - 1) / DefaultBlockSize;

        var colIndicesHandle = ((HipGpuBuffer)csrColIndices).Handle;
        var rowPointersHandle = ((HipGpuBuffer)csrRowPointers).Handle;
        var inputHandle = ((HipGpuBuffer)input).Handle;
        var outputHandle = ((HipGpuBuffer)output).Handle;

        LaunchCsrSegmentedStdDevKernel(kernel, gridX, gridY, DefaultBlockSize, 1,
            colIndicesHandle, rowPointersHandle, inputHandle, outputHandle, M, K, N, epsilon);
    }

    private unsafe void LaunchCsrSegmentedKernel(IntPtr kernel, int gridX, int gridY, int blockX, int blockY,
        IntPtr colIndices, IntPtr rowPointers, IntPtr input, IntPtr output, int M, int K, int N)
    {
        void*[] args = new void*[7];
        fixed (void** argsPtr = args)
        {
            IntPtr[] handles = [colIndices, rowPointers, input, output];
            int[] ints = [M, K, N];

            fixed (IntPtr* h = handles)
            fixed (int* i = ints)
            {
                argsPtr[0] = &h[0];
                argsPtr[1] = &h[1];
                argsPtr[2] = &h[2];
                argsPtr[3] = &h[3];
                argsPtr[4] = &i[0];
                argsPtr[5] = &i[1];
                argsPtr[6] = &i[2];

                var result = HipNativeBindings.hipModuleLaunchKernel(
                    kernel,
                    (uint)gridX, (uint)gridY, 1,
                    (uint)blockX, (uint)blockY, 1,
                    0, _stream, (IntPtr)argsPtr, IntPtr.Zero);
                HipNativeBindings.CheckError(result, "hipModuleLaunchKernel (csr_segmented)");
            }
        }
    }

    private unsafe void LaunchCsrSegmentedStdDevKernel(IntPtr kernel, int gridX, int gridY, int blockX, int blockY,
        IntPtr colIndices, IntPtr rowPointers, IntPtr input, IntPtr output, int M, int K, int N, float epsilon)
    {
        void*[] args = new void*[8];
        fixed (void** argsPtr = args)
        {
            IntPtr[] handles = [colIndices, rowPointers, input, output];
            int[] ints = [M, K, N];
            float epsCopy = epsilon;

            fixed (IntPtr* h = handles)
            fixed (int* i = ints)
            {
                argsPtr[0] = &h[0];
                argsPtr[1] = &h[1];
                argsPtr[2] = &h[2];
                argsPtr[3] = &h[3];
                argsPtr[4] = &i[0];
                argsPtr[5] = &i[1];
                argsPtr[6] = &i[2];
                argsPtr[7] = &epsCopy;

                var result = HipNativeBindings.hipModuleLaunchKernel(
                    kernel,
                    (uint)gridX, (uint)gridY, 1,
                    (uint)blockX, (uint)blockY, 1,
                    0, _stream, (IntPtr)argsPtr, IntPtr.Zero);
                HipNativeBindings.CheckError(result, "hipModuleLaunchKernel (csr_segmented_stddev)");
            }
        }
    }

    private unsafe void LaunchKernel2D(IntPtr kernel, int gridX, int gridY, int blockX, int blockY,
        IntPtr values, IntPtr colIndices, IntPtr rowPointers, IntPtr denseB, IntPtr output,
        int M, int K, int N, int nnz)
    {
        void*[] args = new void*[9];
        fixed (void** argsPtr = args)
        {
            IntPtr[] handles = [values, colIndices, rowPointers, denseB, output];
            int[] ints = [M, K, N, nnz];

            fixed (IntPtr* h = handles)
            fixed (int* i = ints)
            {
                argsPtr[0] = &h[0];
                argsPtr[1] = &h[1];
                argsPtr[2] = &h[2];
                argsPtr[3] = &h[3];
                argsPtr[4] = &h[4];
                argsPtr[5] = &i[0];
                argsPtr[6] = &i[1];
                argsPtr[7] = &i[2];
                argsPtr[8] = &i[3];

                var result = HipNativeBindings.hipModuleLaunchKernel(
                    kernel,
                    (uint)gridX, (uint)gridY, 1,
                    (uint)blockX, (uint)blockY, 1,
                    0, _stream, (IntPtr)argsPtr, IntPtr.Zero);
                HipNativeBindings.CheckError(result, "hipModuleLaunchKernel (csr_spmm)");
            }
        }
    }

    private unsafe void LaunchKernel2DBias(IntPtr kernel, int gridX, int gridY, int blockX, int blockY,
        IntPtr values, IntPtr colIndices, IntPtr rowPointers, IntPtr denseB, IntPtr bias, IntPtr output,
        int M, int K, int N, int nnz)
    {
        void*[] args = new void*[10];
        fixed (void** argsPtr = args)
        {
            IntPtr[] handles = [values, colIndices, rowPointers, denseB, bias, output];
            int[] ints = [M, K, N, nnz];

            fixed (IntPtr* h = handles)
            fixed (int* i = ints)
            {
                argsPtr[0] = &h[0];
                argsPtr[1] = &h[1];
                argsPtr[2] = &h[2];
                argsPtr[3] = &h[3];
                argsPtr[4] = &h[4];
                argsPtr[5] = &h[5];
                argsPtr[6] = &i[0];
                argsPtr[7] = &i[1];
                argsPtr[8] = &i[2];
                argsPtr[9] = &i[3];

                var result = HipNativeBindings.hipModuleLaunchKernel(
                    kernel,
                    (uint)gridX, (uint)gridY, 1,
                    (uint)blockX, (uint)blockY, 1,
                    0, _stream, (IntPtr)argsPtr, IntPtr.Zero);
                HipNativeBindings.CheckError(result, "hipModuleLaunchKernel (csr_spmm_bias)");
            }
        }
    }

    private unsafe void LaunchKernel1D(IntPtr kernel, int grid, int block, IntPtr buffer, int size)
    {
        void*[] args = new void*[2];
        fixed (void** argsPtr = args)
        {
            IntPtr bufferCopy = buffer;
            int sizeCopy = size;
            argsPtr[0] = &bufferCopy;
            argsPtr[1] = &sizeCopy;

            var result = HipNativeBindings.hipModuleLaunchKernel(
                kernel,
                (uint)grid, 1, 1,
                (uint)block, 1, 1,
                0, _stream, (IntPtr)argsPtr, IntPtr.Zero);
            HipNativeBindings.CheckError(result, "hipModuleLaunchKernel (zero_buffer)");
        }
    }

    private unsafe void LaunchScatterAddKernel(IntPtr kernel, int gridX, int gridY, int blockX, int blockY,
        IntPtr input, IntPtr source, IntPtr target, IntPtr edgeValues, IntPtr output,
        int numNodes, int numEdges, int features, int hasEdgeValues)
    {
        void*[] args = new void*[9];
        fixed (void** argsPtr = args)
        {
            IntPtr[] handles = [input, source, target, edgeValues, output];
            int[] ints = [numNodes, numEdges, features, hasEdgeValues];

            fixed (IntPtr* h = handles)
            fixed (int* i = ints)
            {
                argsPtr[0] = &h[0];
                argsPtr[1] = &h[1];
                argsPtr[2] = &h[2];
                argsPtr[3] = &h[3];
                argsPtr[4] = &h[4];
                argsPtr[5] = &i[0];
                argsPtr[6] = &i[1];
                argsPtr[7] = &i[2];
                argsPtr[8] = &i[3];

                var result = HipNativeBindings.hipModuleLaunchKernel(
                    kernel,
                    (uint)gridX, (uint)gridY, 1,
                    (uint)blockX, (uint)blockY, 1,
                    0, _stream, (IntPtr)argsPtr, IntPtr.Zero);
                HipNativeBindings.CheckError(result, "hipModuleLaunchKernel (scatter_add_edges)");
            }
        }
    }

    #endregion

    private void UploadBytesToBuffer(HipGpuByteBuffer buffer, byte[] data)
    {
        var size = (UIntPtr)data.Length;

        GCHandle handle = GCHandle.Alloc(data, GCHandleType.Pinned);
        try
        {
            var result = HipNativeBindings.hipMemcpy(
                buffer.Handle,
                handle.AddrOfPinnedObject(),
                size,
                HipMemcpyKind.HostToDevice);
            HipNativeBindings.CheckError(result, "hipMemcpy H2D (bytes)");
        }
        finally
        {
            handle.Free();
        }
    }

    private byte[] DownloadBytesFromBuffer(HipGpuByteBuffer buffer)
    {
        byte[] result = new byte[buffer.Size];
        var size = (UIntPtr)buffer.Size;

        GCHandle handle = GCHandle.Alloc(result, GCHandleType.Pinned);
        try
        {
            var hipResult = HipNativeBindings.hipMemcpy(
                handle.AddrOfPinnedObject(),
                buffer.Handle,
                size,
                HipMemcpyKind.DeviceToHost);
            HipNativeBindings.CheckError(hipResult, "hipMemcpy D2H (bytes)");
        }
        finally
        {
            handle.Free();
        }

        return result;
    }

    #endregion

    #region Reduction Operations

    public float Sum(IGpuBuffer A, int size)
    {
        if (!_kernelCache.TryGetValue("reduce_sum", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: reduce_sum");

        return ReduceToScalar(krnl, A, size);
    }

    public float Max(IGpuBuffer A, int size)
    {
        if (!_kernelCache.TryGetValue("reduce_max", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: reduce_max");

        return ReduceToScalar(krnl, A, size);
    }

    public float Min(IGpuBuffer A, int size)
    {
        if (!_kernelCache.TryGetValue("reduce_min", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: reduce_min");

        return ReduceToScalar(krnl, A, size);
    }

    private unsafe float ReduceToScalar(IntPtr kernel, IGpuBuffer input, int size)
    {
        const int blockSize = 256;
        int currentSize = size;
        IGpuBuffer currentInput = input;
        IGpuBuffer? tempBuffer1 = null;
        IGpuBuffer? tempBuffer2 = null;

        try
        {
            while (currentSize > 1)
            {
                int gridSize = (currentSize + blockSize - 1) / blockSize;
                int sharedMemSize = blockSize * sizeof(float);

                // Allocate output buffer for partial results
                IGpuBuffer output;
                if (tempBuffer1 is null)
                {
                    tempBuffer1 = AllocateBuffer(gridSize);
                    output = tempBuffer1;
                }
                else if (currentInput == tempBuffer1)
                {
                    if (tempBuffer2 is null || ((HipGpuBuffer)tempBuffer2).Size < gridSize)
                    {
                        tempBuffer2?.Dispose();
                        tempBuffer2 = AllocateBuffer(gridSize);
                    }
                    output = tempBuffer2;
                }
                else
                {
                    output = tempBuffer1;
                }

                    IntPtr reduceInputPtr = currentInput.Handle;
                    IntPtr reduceOutputPtr = output.Handle;
#pragma warning disable CA2014
                    void** reduceArgs = stackalloc void*[3];
#pragma warning restore CA2014
                    reduceArgs[0] = &reduceInputPtr;
                    reduceArgs[1] = &reduceOutputPtr;
                    reduceArgs[2] = &currentSize;

                    LaunchKernel(kernel, (uint)gridSize, (uint)blockSize, reduceArgs, (uint)sharedMemSize);
                    Synchronize();

                currentInput = output;
                currentSize = gridSize;
            }

            // Download the single result
            var result = new float[1];
            var hipResult = (HipGpuBuffer)currentInput;
            GCHandle handle = GCHandle.Alloc(result, GCHandleType.Pinned);
            try
            {
                var res = HipNativeBindings.hipMemcpy(
                    handle.AddrOfPinnedObject(),
                    hipResult.Handle,
                    (UIntPtr)sizeof(float),
                    HipMemcpyKind.DeviceToHost);
                HipNativeBindings.CheckError(res, "hipMemcpy D2H");
            }
            finally
            {
                handle.Free();
            }

            return result[0];
        }
        finally
        {
            tempBuffer1?.Dispose();
            tempBuffer2?.Dispose();
        }
    }

    public unsafe void SumAxis(IGpuBuffer A, IGpuBuffer B, int outerSize, int reduceSize)
    {
        if (!_kernelCache.TryGetValue("sum_axis", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: sum_axis");

            {
            IntPtr _p0 = A.Handle;
            IntPtr _p1 = B.Handle;
            void** args = stackalloc void*[4];
            args[0] = &_p0;
            args[1] = &_p1;
            args[2] = &outerSize;
            args[3] = &reduceSize;


            uint grid = (uint)((outerSize + DefaultBlockSize - 1) / DefaultBlockSize);
            LaunchKernel(krnl, grid, DefaultBlockSize, args);
            Synchronize();
            }
    }

    #endregion

    #region Convolution Operations

    public unsafe void Conv2D(IGpuBuffer input, IGpuBuffer kernel, IGpuBuffer output,
        int batch, int inChannels, int inHeight, int inWidth,
        int outChannels, int outHeight, int outWidth,
        int kernelH, int kernelW,
        int strideH, int strideW, int padH, int padW,
        int dilationH, int dilationW)
    {
        IntPtr pInput = input.Handle, pKernel = kernel.Handle, pOutput = output.Handle;

        // Route 3x3 stride-1 dilation-1 through Winograd F(2x2,3x3)
        if (kernelH == 3 && kernelW == 3 && strideH == 1 && strideW == 1 &&
            dilationH == 1 && dilationW == 1 &&
            _kernelCache.TryGetValue("conv2d_winograd_f2x2_3x3", out var winogradKrnl))
        {
            int tilesH = (outHeight + 1) / 2;
            int tilesW = (outWidth + 1) / 2;
            int totalTiles = batch * outChannels * tilesH * tilesW;
            uint grid = (uint)((totalTiles + 255) / 256);

            var args = new IntPtr[]
            {
                (IntPtr)(&pInput), (IntPtr)(&pKernel), (IntPtr)(&pOutput),
                (IntPtr)(&batch), (IntPtr)(&inChannels), (IntPtr)(&inHeight), (IntPtr)(&inWidth),
                (IntPtr)(&outChannels), (IntPtr)(&outHeight), (IntPtr)(&outWidth),
                (IntPtr)(&padH), (IntPtr)(&padW)
            };
            LaunchKernel(winogradKrnl, grid, 256, args);
            Synchronize();
            return;
        }

        // Use shared-memory tiled kernel when available
        if (_kernelCache.TryGetValue("conv2d_tiled", out var tiledKrnl))
        {
            const int TILE_OUT = 16;
            uint gx = (uint)((outWidth + TILE_OUT - 1) / TILE_OUT);
            uint gy = (uint)((outHeight + TILE_OUT - 1) / TILE_OUT);
            uint gz = (uint)(batch * outChannels);

            int effKH = (kernelH - 1) * dilationH + 1;
            int effKW = (kernelW - 1) * dilationW + 1;
            int tileInH = TILE_OUT * strideH + effKH - strideH;
            int tileInW = TILE_OUT * strideW + effKW - strideW;
            uint sharedMem = (uint)(tileInH * tileInW * sizeof(float));

            var args = new IntPtr[]
            {
                (IntPtr)(&pInput), (IntPtr)(&pKernel), (IntPtr)(&pOutput),
                (IntPtr)(&batch), (IntPtr)(&inChannels), (IntPtr)(&inHeight), (IntPtr)(&inWidth),
                (IntPtr)(&outChannels), (IntPtr)(&outHeight), (IntPtr)(&outWidth),
                (IntPtr)(&kernelH), (IntPtr)(&kernelW), (IntPtr)(&strideH), (IntPtr)(&strideW),
                (IntPtr)(&padH), (IntPtr)(&padW), (IntPtr)(&dilationH), (IntPtr)(&dilationW)
            };
            LaunchKernel3D(tiledKrnl, gx, gy, gz, TILE_OUT, TILE_OUT, 1, args, sharedMem);
            Synchronize();
            return;
        }

        // Fallback: direct convolution
        if (!_kernelCache.TryGetValue("conv2d_direct", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: conv2d_direct");

        {
            var args = new IntPtr[]
            {
                (IntPtr)(&pInput), (IntPtr)(&pKernel), (IntPtr)(&pOutput),
                (IntPtr)(&batch), (IntPtr)(&inChannels), (IntPtr)(&inHeight), (IntPtr)(&inWidth),
                (IntPtr)(&outChannels), (IntPtr)(&outHeight), (IntPtr)(&outWidth),
                (IntPtr)(&kernelH), (IntPtr)(&kernelW), (IntPtr)(&strideH), (IntPtr)(&strideW),
                (IntPtr)(&padH), (IntPtr)(&padW), (IntPtr)(&dilationH), (IntPtr)(&dilationW)
            };
            LaunchKernel3D(krnl, (uint)((outWidth + 15) / 16), (uint)((outHeight + 15) / 16),
                (uint)(batch * outChannels), 16, 16, 1, args);
            Synchronize();
        }
    }

    public unsafe void Conv2DBackwardInput(IGpuBuffer gradOutput, IGpuBuffer kernel, IGpuBuffer gradInput,
        int batch, int inChannels, int inHeight, int inWidth,
        int outChannels, int outHeight, int outWidth,
        int kernelH, int kernelW,
        int strideH, int strideW, int padH, int padW,
        int dilationH, int dilationW)
    {
        if (!_kernelCache.TryGetValue("conv2d_backward_input", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: conv2d_backward_input");

            {
            var pGradOut = gradOutput.Handle;
            var pKernel = kernel.Handle;
            var pGradIn = gradInput.Handle;
            void** args = stackalloc void*[18];
            args[0] = &pGradOut;
            args[1] = &pKernel;
            args[2] = &pGradIn;
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


            uint gridX = (uint)((inWidth + 15) / 16);
            uint gridY = (uint)((inHeight + 15) / 16);
            uint gridZ = (uint)(batch * inChannels);
            LaunchKernel3D(krnl, gridX, gridY, gridZ, 16, 16, 1, args);
            Synchronize();
            }
    }

    public unsafe void Conv2DBackwardKernel(IGpuBuffer input, IGpuBuffer gradOutput, IGpuBuffer gradKernel,
        int batch, int inChannels, int inHeight, int inWidth,
        int outChannels, int outHeight, int outWidth,
        int kernelH, int kernelW,
        int strideH, int strideW, int padH, int padW,
        int dilationH, int dilationW)
    {
        if (!_kernelCache.TryGetValue("conv2d_backward_kernel", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: conv2d_backward_kernel");

            {
            var pInput = input.Handle;
            var pGradOut = gradOutput.Handle;
            var pGradKernel = gradKernel.Handle;
            void** args = stackalloc void*[18];
            args[0] = &pInput;
            args[1] = &pGradOut;
            args[2] = &pGradKernel;
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


            uint gridX = (uint)((kernelW + 15) / 16);
            uint gridY = (uint)((kernelH + 15) / 16);
            uint gridZ = (uint)(outChannels * inChannels);
            LaunchKernel3D(krnl, gridX, gridY, gridZ, 16, 16, 1, args);
            Synchronize();
            }
    }

    public unsafe void Conv1D(IGpuBuffer input, IGpuBuffer kernel, IGpuBuffer output,
        int batch, int inChannels, int inLength,
        int outChannels, int outLength, int kernelLength,
        int stride, int padding, int dilation)
    {
        Conv2D(input, kernel, output, batch, inChannels, 1, inLength,
            outChannels, 1, outLength, 1, kernelLength, 1, stride, 0, padding, 1, dilation);
    }

    public unsafe void Conv1DBackwardInput(IGpuBuffer gradOutput, IGpuBuffer kernel, IGpuBuffer gradInput,
        int batch, int inChannels, int inLength,
        int outChannels, int outLength, int kernelLength,
        int stride, int padding, int dilation)
    {
        Conv2DBackwardInput(gradOutput, kernel, gradInput, batch, inChannels, 1, inLength,
            outChannels, 1, outLength, 1, kernelLength, 1, stride, 0, padding, 1, dilation);
    }

    public unsafe void Conv1DBackwardKernel(IGpuBuffer input, IGpuBuffer gradOutput, IGpuBuffer gradKernel,
        int batch, int inChannels, int inLength,
        int outChannels, int outLength, int kernelLength,
        int stride, int padding, int dilation)
    {
        Conv2DBackwardKernel(input, gradOutput, gradKernel, batch, inChannels, 1, inLength,
            outChannels, 1, outLength, 1, kernelLength, 1, stride, 0, padding, 1, dilation);
    }

    public unsafe void Unfold(IGpuBuffer input, IGpuBuffer output,
        int batch, int channels, int height, int width,
        int kernelH, int kernelW, int strideH, int strideW, int padH, int padW)
    {
        if (strideH <= 0 || strideW <= 0) throw new ArgumentException("Stride must be positive.");
        int unfoldOutH = (height + 2 * padH - kernelH) / strideH + 1;
        int unfoldOutW = (width + 2 * padW - kernelW) / strideW + 1;
        if (unfoldOutH <= 0 || unfoldOutW <= 0) throw new ArgumentException($"Invalid Unfold output dimensions {unfoldOutH}x{unfoldOutW}.");

        if (!_kernelCache.TryGetValue("im2col", out var im2colKernel))
            throw new InvalidOperationException("HIP kernel not found: im2col");

        IntPtr inputPtr = input.Handle;
        IntPtr outputPtr = output.Handle;
        int outH = (height + 2 * padH - kernelH) / strideH + 1;
        int outW = (width + 2 * padW - kernelW) / strideW + 1;
        int totalPatches = batch * outH * outW;
        int dilationH = 1, dilationW = 1;

        void** args = stackalloc void*[15];
        args[0] = &inputPtr; args[1] = &outputPtr;
        args[2] = &batch; args[3] = &channels; args[4] = &height; args[5] = &width;
        args[6] = &kernelH; args[7] = &kernelW; args[8] = &strideH; args[9] = &strideW;
        args[10] = &padH; args[11] = &padW; args[12] = &dilationH; args[13] = &dilationW;
        args[14] = &outH;

        uint gridX = (uint)((totalPatches + 255) / 256);
        LaunchKernel(im2colKernel, gridX, 256, args);
        Synchronize();
    }

    public unsafe void Fold(IGpuBuffer input, IGpuBuffer output,
        int batch, int channels, int outputH, int outputW,
        int kernelH, int kernelW, int strideH, int strideW, int padH, int padW)
    {
        if (strideH <= 0 || strideW <= 0) throw new ArgumentException("Stride must be positive.");

        if (!_kernelCache.TryGetValue("col2im", out var col2imKernel))
            throw new InvalidOperationException("HIP kernel not found: col2im");

        IntPtr inputPtr = input.Handle;
        IntPtr outputPtr = output.Handle;
        int outH = (outputH + 2 * padH - kernelH) / strideH + 1;
        int totalSize = batch * channels * outputH * outputW;
        int dilationH = 1, dilationW = 1;

        // Zero output buffer for accumulation
        var memsetResult = HipNativeBindings.hipMemset(output.Handle, 0, (nuint)(totalSize * sizeof(float)));
        HipNativeBindings.CheckError(memsetResult, "hipMemset");

        void** args = stackalloc void*[15];
        args[0] = &inputPtr; args[1] = &outputPtr;
        args[2] = &batch; args[3] = &channels; args[4] = &outputH; args[5] = &outputW;
        args[6] = &kernelH; args[7] = &kernelW; args[8] = &strideH; args[9] = &strideW;
        args[10] = &padH; args[11] = &padW; args[12] = &dilationH; args[13] = &dilationW;
        args[14] = &outH;

        uint gridX = (uint)((totalSize + 255) / 256);
        LaunchKernel(col2imKernel, gridX, 256, args);
        Synchronize();
    }

    public unsafe void Conv3D(IGpuBuffer input, IGpuBuffer kernel, IGpuBuffer output,
        int batch, int inChannels, int inDepth, int inHeight, int inWidth,
        int outChannels, int outDepth, int outHeight, int outWidth,
        int kernelD, int kernelH, int kernelW,
        int strideD, int strideH, int strideW,
        int padD, int padH, int padW,
        int dilationD, int dilationH, int dilationW)
    {
        if (!_kernelCache.TryGetValue("conv3d_direct", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: conv3d_direct");

            {
            var pInput = input.Handle;
            var pKernel = kernel.Handle;
            var pOutput = output.Handle;
            void** args = stackalloc void*[25];
            args[0] = &pInput;
            args[1] = &pKernel;
            args[2] = &pOutput;
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
            args[21] = &dilationD;
            args[22] = &dilationH;
            args[23] = &dilationW;



            uint gridX = (uint)((outWidth + 7) / 8);
            uint gridY = (uint)((outHeight + 7) / 8);
            uint gridZ = (uint)(batch * outChannels * outDepth);
            LaunchKernel3D(krnl, gridX, gridY, gridZ, 8, 8, 1, args);
            Synchronize();

            }
    }

    public unsafe void DepthwiseConv2D(IGpuBuffer input, IGpuBuffer kernel, IGpuBuffer output,
        int batch, int channels, int inHeight, int inWidth,
        int outHeight, int outWidth,
        int kernelH, int kernelW,
        int strideH, int strideW, int padH, int padW)
    {
        if (!_kernelCache.TryGetValue("depthwise_conv2d", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: depthwise_conv2d");

            {
            var pInput = input.Handle;
            var pKernel = kernel.Handle;
            var pOutput = output.Handle;
            void** args = stackalloc void*[15];
            args[0] = &pInput;
            args[1] = &pKernel;
            args[2] = &pOutput;
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


            uint gridX = (uint)((outWidth + 15) / 16);
            uint gridY = (uint)((outHeight + 15) / 16);
            uint gridZ = (uint)(batch * channels);
            LaunchKernel3D(krnl, gridX, gridY, gridZ, 16, 16, 1, args);
            Synchronize();

            }
    }

    public unsafe void ConvTranspose2D(IGpuBuffer input, IGpuBuffer kernel, IGpuBuffer output,
        int batch, int inChannels, int inHeight, int inWidth,
        int outChannels, int outHeight, int outWidth,
        int kernelH, int kernelW,
        int strideH, int strideW, int padH, int padW,
        int outputPadH, int outputPadW)
    {
        if (!_kernelCache.TryGetValue("conv_transpose2d", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: conv_transpose2d");

            {
            var pInput = input.Handle;
            var pKernel = kernel.Handle;
            var pOutput = output.Handle;
            void** args = stackalloc void*[18];
            args[0] = &pInput;
            args[1] = &pKernel;
            args[2] = &pOutput;
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


            uint gridX = (uint)((outWidth + 15) / 16);
            uint gridY = (uint)((outHeight + 15) / 16);
            uint gridZ = (uint)(batch * outChannels);
            LaunchKernel3D(krnl, gridX, gridY, gridZ, 16, 16, 1, args);
            Synchronize();
            }
    }

    public unsafe void ConvTranspose2DBackwardInput(IGpuBuffer gradOutput, IGpuBuffer kernel, IGpuBuffer gradInput,
        int batch, int inChannels, int inHeight, int inWidth,
        int outChannels, int outHeight, int outWidth,
        int kernelH, int kernelW,
        int strideH, int strideW, int padH, int padW,
        int outputPadH, int outputPadW)
    {
        if (!_kernelCache.TryGetValue("conv_transpose2d_backward_input", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: conv_transpose2d_backward_input");

        int totalInput = batch * inChannels * inHeight * inWidth;
            {
            var pGradOutput = gradOutput.Handle;
            var pKernel = kernel.Handle;
            var pGradInput = gradInput.Handle;
            void** args = stackalloc void*[19];
            args[0] = &pGradOutput;
            args[1] = &pKernel;
            args[2] = &pGradInput;
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


            uint grid = (uint)((totalInput + DefaultBlockSize - 1) / DefaultBlockSize);
            LaunchKernel(krnl, grid, DefaultBlockSize, args);
            Synchronize();
            }
    }

    public unsafe void ConvTranspose2DBackwardKernel(IGpuBuffer input, IGpuBuffer gradOutput, IGpuBuffer gradKernel,
        int batch, int inChannels, int inHeight, int inWidth,
        int outChannels, int outHeight, int outWidth,
        int kernelH, int kernelW,
        int strideH, int strideW, int padH, int padW,
        int outputPadH, int outputPadW)
    {
        if (!_kernelCache.TryGetValue("conv_transpose2d_backward_kernel", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: conv_transpose2d_backward_kernel");

        int totalKernel = inChannels * outChannels * kernelH * kernelW;
            {
            var pInput = input.Handle;
            var pGradOutput = gradOutput.Handle;
            var pGradKernel = gradKernel.Handle;
            void** args = stackalloc void*[19];
            args[0] = &pInput;
            args[1] = &pGradOutput;
            args[2] = &pGradKernel;
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


            uint grid = (uint)((totalKernel + DefaultBlockSize - 1) / DefaultBlockSize);
            LaunchKernel(krnl, grid, DefaultBlockSize, args);
            Synchronize();
            }
    }

    #endregion

    #region Pooling Operations

    public unsafe void MaxPool2D(IGpuBuffer input, IGpuBuffer output, IGpuBuffer? indices,
        int batch, int channels, int inHeight, int inWidth,
        int outHeight, int outWidth,
        int kernelH, int kernelW,
        int strideH, int strideW, int padH, int padW)
    {
        if (!_kernelCache.TryGetValue("maxpool2d", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: maxpool2d");

            {
            var pInput = input.Handle;
            var pOutput = output.Handle;
            var pIndices = indices?.Handle ?? IntPtr.Zero;
            int saveIndices = indices is not null ? 1 : 0;
            void** args = stackalloc void*[16];
            args[0] = &pInput;
            args[1] = &pOutput;
            args[2] = &pIndices;
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


            uint gridX = (uint)((outWidth + 15) / 16);
            uint gridY = (uint)((outHeight + 15) / 16);
            uint gridZ = (uint)(batch * channels);
            LaunchKernel3D(krnl, gridX, gridY, gridZ, 16, 16, 1, args);
            Synchronize();

            }
    }

    public unsafe void MaxPool2DBackward(IGpuBuffer gradOutput, IGpuBuffer indices, IGpuBuffer gradInput,
        int batch, int channels, int inHeight, int inWidth,
        int outHeight, int outWidth,
        int kernelH, int kernelW,
        int strideH, int strideW, int padH, int padW)
    {
        if (!_kernelCache.TryGetValue("maxpool2d_backward", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: maxpool2d_backward");

            {
            IntPtr _p0 = gradOutput.Handle;
            IntPtr _p1 = indices.Handle;
            IntPtr _p2 = gradInput.Handle;
            void** args = stackalloc void*[9];
            args[0] = &_p0;
            args[1] = &_p1;
            args[2] = &_p2;
            args[3] = &batch;
            args[4] = &channels;
            args[5] = &inHeight;
            args[6] = &inWidth;
            args[7] = &outHeight;
            args[8] = &outWidth;


            uint gridX = (uint)((outWidth + 15) / 16);
            uint gridY = (uint)((outHeight + 15) / 16);
            uint gridZ = (uint)(batch * channels);
            LaunchKernel3D(krnl, gridX, gridY, gridZ, 16, 16, 1, args);
            Synchronize();
            }
    }

    public unsafe void AvgPool2D(IGpuBuffer input, IGpuBuffer output,
        int batch, int channels, int inHeight, int inWidth,
        int outHeight, int outWidth,
        int kernelH, int kernelW,
        int strideH, int strideW, int padH, int padW,
        bool countIncludePad)
    {
        if (!_kernelCache.TryGetValue("avgpool2d", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: avgpool2d");

            {
            int countPadInt = countIncludePad ? 1 : 0;
            IntPtr _p0 = input.Handle;
            IntPtr _p1 = output.Handle;
            void** args = stackalloc void*[15];
            args[0] = &_p0;
            args[1] = &_p1;
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
            args[14] = &countPadInt;


            uint gridX = (uint)((outWidth + 15) / 16);
            uint gridY = (uint)((outHeight + 15) / 16);
            uint gridZ = (uint)(batch * channels);
            LaunchKernel3D(krnl, gridX, gridY, gridZ, 16, 16, 1, args);
            Synchronize();

            }
    }

    public unsafe void AvgPool2DBackward(IGpuBuffer gradOutput, IGpuBuffer gradInput,
        int batch, int channels, int inHeight, int inWidth,
        int outHeight, int outWidth,
        int kernelH, int kernelW,
        int strideH, int strideW, int padH, int padW,
        bool countIncludePad)
    {
        if (!_kernelCache.TryGetValue("avgpool2d_backward", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: avgpool2d_backward");

            {
            int countPadInt = countIncludePad ? 1 : 0;
            IntPtr _p0 = gradOutput.Handle;
            IntPtr _p1 = gradInput.Handle;
            void** args = stackalloc void*[15];
            args[0] = &_p0;
            args[1] = &_p1;
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
            args[14] = &countPadInt;


            uint gridX = (uint)((inWidth + 15) / 16);
            uint gridY = (uint)((inHeight + 15) / 16);
            uint gridZ = (uint)(batch * channels);
            LaunchKernel3D(krnl, gridX, gridY, gridZ, 16, 16, 1, args);
            Synchronize();

            }
    }

    public unsafe void GlobalAvgPool2D(IGpuBuffer input, IGpuBuffer output, int batch, int channels, int height, int width)
    {
        if (!_kernelCache.TryGetValue("global_avgpool2d", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: global_avgpool2d");

            {
            IntPtr _p0 = input.Handle;
            IntPtr _p1 = output.Handle;
            void** args = stackalloc void*[6];
            args[0] = &_p0;
            args[1] = &_p1;
            args[2] = &batch;
            args[3] = &channels;
            args[4] = &height;
            args[5] = &width;


            uint grid = (uint)((batch * channels + DefaultBlockSize - 1) / DefaultBlockSize);
            LaunchKernel(krnl, grid, DefaultBlockSize, args);
            Synchronize();
            }
    }

    public unsafe void GlobalMaxPool2D(IGpuBuffer input, IGpuBuffer output, int batch, int channels, int height, int width)
    {
        // Overload without indices - call main implementation with null indices
        GlobalMaxPool2D(input, output, (IGpuBuffer?)null, batch, channels, height, width);
    }

    // Interface implementation for non-nullable indices (required for backward pass)
    void IDirectGpuBackend.GlobalMaxPool2D(IGpuBuffer input, IGpuBuffer output, IGpuBuffer indices, int batch, int channels, int height, int width)
        => GlobalMaxPool2D(input, output, (IGpuBuffer?)indices, batch, channels, height, width);

    public unsafe void GlobalMaxPool2D(IGpuBuffer input, IGpuBuffer output, IGpuBuffer? indices, int batch, int channels, int height, int width)
    {
        if (!_kernelCache.TryGetValue("global_maxpool2d", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: global_maxpool2d");

        int saveIndices = indices is not null ? 1 : 0;
        IntPtr indicesPtr = indices?.Handle ?? IntPtr.Zero;

            {
            IntPtr _p0 = input.Handle;
            IntPtr _p1 = output.Handle;
            void** args = stackalloc void*[8];
            args[0] = &_p0;
            args[1] = &_p1;
            args[2] = &indicesPtr;
            args[3] = &batch;
            args[4] = &channels;
            args[5] = &height;
            args[6] = &width;
            args[7] = &saveIndices;


            uint grid = (uint)((batch * channels + DefaultBlockSize - 1) / DefaultBlockSize);
            LaunchKernel(krnl, grid, DefaultBlockSize, args);
            Synchronize();
            }
    }

    public unsafe void GlobalAvgPool2DBackward(IGpuBuffer gradOutput, IGpuBuffer gradInput, int batch, int channels, int height, int width)
    {
        if (!_kernelCache.TryGetValue("global_avgpool2d_backward", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: global_avgpool2d_backward");

            {
            IntPtr _p0 = gradOutput.Handle;
            IntPtr _p1 = gradInput.Handle;
            void** args = stackalloc void*[6];
            args[0] = &_p0;
            args[1] = &_p1;
            args[2] = &batch;
            args[3] = &channels;
            args[4] = &height;
            args[5] = &width;


            int totalElements = batch * channels * height * width;
            uint grid = (uint)((totalElements + DefaultBlockSize - 1) / DefaultBlockSize);
            LaunchKernel(krnl, grid, DefaultBlockSize, args);
            Synchronize();
            }
    }

    public unsafe void GlobalMaxPool2DBackward(IGpuBuffer gradOutput, IGpuBuffer indices, IGpuBuffer gradInput, int batch, int channels, int height, int width)
    {
        if (!_kernelCache.TryGetValue("global_maxpool2d_backward", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: global_maxpool2d_backward");

        // First zero out the gradient input
        Fill(gradInput, 0f, batch * channels * height * width);

        int totalOutputs = batch * channels;
        uint grid = (uint)((totalOutputs + DefaultBlockSize - 1) / DefaultBlockSize);

            {
            IntPtr _p0 = gradOutput.Handle;
            IntPtr _p1 = indices.Handle;
            IntPtr _p2 = gradInput.Handle;
            void** args = stackalloc void*[7];
            args[0] = &_p0;
            args[1] = &_p1;
            args[2] = &_p2;
            args[3] = &batch;
            args[4] = &channels;
            args[5] = &height;
            args[6] = &width;


            LaunchKernel(krnl, grid, DefaultBlockSize, args);
            Synchronize();
            }
    }

    public unsafe void AdaptiveAvgPool2D(IGpuBuffer input, IGpuBuffer output, int batch, int channels, int inHeight, int inWidth, int outHeight, int outWidth)
    {
        if (!_kernelCache.TryGetValue("adaptive_avgpool2d", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: adaptive_avgpool2d");

            {
            IntPtr _p0 = input.Handle;
            IntPtr _p1 = output.Handle;
            void** args = stackalloc void*[8];
            args[0] = &_p0;
            args[1] = &_p1;
            args[2] = &batch;
            args[3] = &channels;
            args[4] = &inHeight;
            args[5] = &inWidth;
            args[6] = &outHeight;
            args[7] = &outWidth;


            uint gridX = (uint)((outWidth + 15) / 16);
            uint gridY = (uint)((outHeight + 15) / 16);
            uint gridZ = (uint)(batch * channels);
            LaunchKernel3D(krnl, gridX, gridY, gridZ, 16, 16, 1, args);
            Synchronize();
            }
    }

    public unsafe void MaxPool3D(IGpuBuffer input, IGpuBuffer output, IGpuBuffer? indices,
        int batch, int channels,
        int inDepth, int inHeight, int inWidth,
        int outDepth, int outHeight, int outWidth,
        int kernelD, int kernelH, int kernelW,
        int strideD, int strideH, int strideW)
    {
        if (!_kernelCache.TryGetValue("maxpool3d", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: maxpool3d");

        int saveIndices = indices is not null ? 1 : 0;
        IntPtr indicesPtr = indices?.Handle ?? IntPtr.Zero;

            {
            IntPtr _p0 = input.Handle;
            IntPtr _p1 = output.Handle;
            void** args = stackalloc void*[18];
            args[0] = &_p0;
            args[1] = &_p1;
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


            uint gridX = (uint)((outWidth + 7) / 8);
            uint gridY = (uint)((outHeight + 7) / 8);
            uint gridZ = (uint)(batch * channels * outDepth);
            LaunchKernel3D(krnl, gridX, gridY, gridZ, 8, 8, 1, args);
            Synchronize();
            }
    }

    public unsafe void MaxPool3DBackward(IGpuBuffer gradOutput, IGpuBuffer indices, IGpuBuffer gradInput,
        int batch, int channels,
        int inDepth, int inHeight, int inWidth,
        int outDepth, int outHeight, int outWidth)
    {
        if (!_kernelCache.TryGetValue("maxpool3d_backward", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: maxpool3d_backward");

            {
            IntPtr _p0 = gradOutput.Handle;
            IntPtr _p1 = indices.Handle;
            IntPtr _p2 = gradInput.Handle;
            void** args = stackalloc void*[11];
            args[0] = &_p0;
            args[1] = &_p1;
            args[2] = &_p2;
            args[3] = &batch;
            args[4] = &channels;
            args[5] = &inDepth;
            args[6] = &inHeight;
            args[7] = &inWidth;
            args[8] = &outDepth;
            args[9] = &outHeight;
            args[10] = &outWidth;


            uint gridX = (uint)((outWidth + 7) / 8);
            uint gridY = (uint)((outHeight + 7) / 8);
            uint gridZ = (uint)(batch * channels * outDepth);
            LaunchKernel3D(krnl, gridX, gridY, gridZ, 8, 8, 1, args);
            Synchronize();
            }
    }

    public unsafe void NearestNeighborUpsample3D(IGpuBuffer input, IGpuBuffer output,
        int batch, int channels,
        int inDepth, int inHeight, int inWidth,
        int scaleD, int scaleH, int scaleW)
    {
        if (!_kernelCache.TryGetValue("nearest_upsample3d", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: nearest_upsample3d");

        int outDepth = inDepth * scaleD;
        int outHeight = inHeight * scaleH;
        int outWidth = inWidth * scaleW;

            {
            IntPtr _p0 = input.Handle;
            IntPtr _p1 = output.Handle;
            void** args = stackalloc void*[10];
            args[0] = &_p0;
            args[1] = &_p1;
            args[2] = &batch;
            args[3] = &channels;
            args[4] = &inDepth;
            args[5] = &inHeight;
            args[6] = &inWidth;
            args[7] = &scaleD;
            args[8] = &scaleH;
            args[9] = &scaleW;


            uint gridX = (uint)((outWidth + 7) / 8);
            uint gridY = (uint)((outHeight + 7) / 8);
            uint gridZ = (uint)(batch * channels * outDepth);
            LaunchKernel3D(krnl, gridX, gridY, gridZ, 8, 8, 1, args);
            Synchronize();
            }
    }

    public unsafe void NearestNeighborUpsample3DBackward(IGpuBuffer gradOutput, IGpuBuffer gradInput,
        int batch, int channels,
        int inDepth, int inHeight, int inWidth,
        int scaleD, int scaleH, int scaleW)
    {
        if (!_kernelCache.TryGetValue("nearest_upsample3d_backward", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: nearest_upsample3d_backward");

        int outDepth = inDepth * scaleD;
        int outHeight = inHeight * scaleH;
        int outWidth = inWidth * scaleW;

            {
            IntPtr _p0 = gradOutput.Handle;
            IntPtr _p1 = gradInput.Handle;
            void** args = stackalloc void*[10];
            args[0] = &_p0;
            args[1] = &_p1;
            args[2] = &batch;
            args[3] = &channels;
            args[4] = &inDepth;
            args[5] = &inHeight;
            args[6] = &inWidth;
            args[7] = &scaleD;
            args[8] = &scaleH;
            args[9] = &scaleW;


            uint gridX = (uint)((outWidth + 7) / 8);
            uint gridY = (uint)((outHeight + 7) / 8);
            uint gridZ = (uint)(batch * channels * outDepth);
            LaunchKernel3D(krnl, gridX, gridY, gridZ, 8, 8, 1, args);
            Synchronize();
            }
    }

    #endregion

    #region Normalization Operations

    public unsafe void BatchNorm(IGpuBuffer input, IGpuBuffer output, IGpuBuffer gamma, IGpuBuffer beta,
        IGpuBuffer runningMean, IGpuBuffer runningVar, IGpuBuffer saveMean, IGpuBuffer saveInvVar,
        int batch, int channels, int spatialSize, float epsilon, float momentum, bool training)
    {
        if (!_kernelCache.TryGetValue("batchnorm_forward", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: batchnorm_forward");

            {
            int trainingInt = training ? 1 : 0;
            IntPtr _p0 = input.Handle;
            IntPtr _p1 = output.Handle;
            IntPtr _p2 = gamma.Handle;
            IntPtr _p3 = beta.Handle;
            IntPtr _p4 = runningMean.Handle;
            IntPtr _p5 = runningVar.Handle;
            IntPtr _p6 = saveMean.Handle;
            IntPtr _p7 = saveInvVar.Handle;
            void** args = stackalloc void*[14];
            args[0] = &_p0;
            args[1] = &_p1;
            args[2] = &_p2;
            args[3] = &_p3;
            args[4] = &_p4;
            args[5] = &_p5;
            args[6] = &_p6;
            args[7] = &_p7;
            args[8] = &batch;
            args[9] = &channels;
            args[10] = &spatialSize;
            args[11] = &epsilon;
            args[12] = &momentum;
            args[13] = &trainingInt;


            uint grid = (uint)((channels + DefaultBlockSize - 1) / DefaultBlockSize);
            LaunchKernel(krnl, grid, DefaultBlockSize, args);
            Synchronize();
            }
    }

    public unsafe bool TryFusedBatchNormActivation(IGpuBuffer input, IGpuBuffer output, IGpuBuffer gamma, IGpuBuffer beta,
        IGpuBuffer runningMean, IGpuBuffer runningVar, IGpuBuffer saveMean, IGpuBuffer saveInvVar,
        int batch, int channels, int spatialSize, float epsilon, float momentum, bool training,
        FusedActivationType activation)
    {
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

        if (string.IsNullOrEmpty(kernelName) || !_kernelCache.TryGetValue(kernelName, out var krnl))
            return false;

        int totalSize = batch * channels * spatialSize;
        uint grid = (uint)((totalSize + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr _p0 = input.Handle;
        IntPtr _p1 = output.Handle;
        IntPtr _p2 = gamma.Handle;
        IntPtr _p3 = beta.Handle;
        IntPtr _p4 = runningMean.Handle;
        IntPtr _p5 = runningVar.Handle;
        void** args = stackalloc void*[10];
        args[0] = &_p0;
        args[1] = &_p1;
        args[2] = &_p2;
        args[3] = &_p3;
        args[4] = &_p4;
        args[5] = &_p5;
        args[6] = &batch;
        args[7] = &channels;
        args[8] = &spatialSize;
        args[9] = &epsilon;
        LaunchKernel(krnl, grid, DefaultBlockSize, args);
        Synchronize();
        return true;
    }

    public unsafe void BatchNormBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gamma,
        IGpuBuffer saveMean, IGpuBuffer saveInvVar, IGpuBuffer gradInput, IGpuBuffer gradGamma, IGpuBuffer gradBeta,
        int batch, int channels, int spatialSize, float epsilon)
    {
        if (!_kernelCache.TryGetValue("batchnorm_backward", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: batchnorm_backward");

            {
            IntPtr _p0 = gradOutput.Handle;
            IntPtr _p1 = input.Handle;
            IntPtr _p2 = gamma.Handle;
            IntPtr _p3 = saveMean.Handle;
            IntPtr _p4 = saveInvVar.Handle;
            IntPtr _p5 = gradInput.Handle;
            IntPtr _p6 = gradGamma.Handle;
            IntPtr _p7 = gradBeta.Handle;
            void** args = stackalloc void*[12];
            args[0] = &_p0;
            args[1] = &_p1;
            args[2] = &_p2;
            args[3] = &_p3;
            args[4] = &_p4;
            args[5] = &_p5;
            args[6] = &_p6;
            args[7] = &_p7;
            args[8] = &batch;
            args[9] = &channels;
            args[10] = &spatialSize;
            args[11] = &epsilon;


            uint grid = (uint)((channels + DefaultBlockSize - 1) / DefaultBlockSize);
            LaunchKernel(krnl, grid, DefaultBlockSize, args);
            Synchronize();
            }
    }

    public unsafe void LayerNorm(IGpuBuffer input, IGpuBuffer output, IGpuBuffer gamma, IGpuBuffer beta,
        IGpuBuffer saveMean, IGpuBuffer saveInvVar, int batchSize, int normalizedSize, float epsilon)
    {
        if (!_kernelCache.TryGetValue("layernorm_forward", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: layernorm_forward");

            {
            IntPtr _p0 = input.Handle;
            IntPtr _p1 = output.Handle;
            IntPtr _p2 = gamma.Handle;
            IntPtr _p3 = beta.Handle;
            IntPtr _p4 = saveMean.Handle;
            IntPtr _p5 = saveInvVar.Handle;
            void** args = stackalloc void*[9];
            args[0] = &_p0;
            args[1] = &_p1;
            args[2] = &_p2;
            args[3] = &_p3;
            args[4] = &_p4;
            args[5] = &_p5;
            args[6] = &batchSize;
            args[7] = &normalizedSize;
            args[8] = &epsilon;


            uint grid = (uint)((batchSize + DefaultBlockSize - 1) / DefaultBlockSize);
            LaunchKernel(krnl, grid, DefaultBlockSize, args);
            Synchronize();
            }
    }

    public unsafe void LayerNormBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gamma,
        IGpuBuffer saveMean, IGpuBuffer saveInvVar, IGpuBuffer gradInput, IGpuBuffer gradGamma, IGpuBuffer gradBeta,
        int batchSize, int normalizedSize, float epsilon)
    {
        // First pass: compute gradInput
        if (!_kernelCache.TryGetValue("layernorm_backward", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: layernorm_backward");

            {
            IntPtr _p0 = gradOutput.Handle;
            IntPtr _p1 = input.Handle;
            IntPtr _p2 = gamma.Handle;
            IntPtr _p3 = saveMean.Handle;
            IntPtr _p4 = saveInvVar.Handle;
            IntPtr _p5 = gradInput.Handle;
            IntPtr _p6 = gradGamma.Handle;
            IntPtr _p7 = gradBeta.Handle;
            void** args = stackalloc void*[11];
            args[0] = &_p0;
            args[1] = &_p1;
            args[2] = &_p2;
            args[3] = &_p3;
            args[4] = &_p4;
            args[5] = &_p5;
            args[6] = &_p6;
            args[7] = &_p7;
            args[8] = &batchSize;
            args[9] = &normalizedSize;
            args[10] = &epsilon;


            uint grid = (uint)((batchSize + DefaultBlockSize - 1) / DefaultBlockSize);
            LaunchKernel(krnl, grid, DefaultBlockSize, args);
            Synchronize();
            }

        // Second pass: compute gradGamma and gradBeta
        if (_kernelCache.TryGetValue("layernorm_grad_params", out var gradKrnl))
        {
                {
                IntPtr _p0 = gradOutput.Handle;
                IntPtr _p1 = input.Handle;
                IntPtr _p2 = saveMean.Handle;
                IntPtr _p3 = saveInvVar.Handle;
                IntPtr _p4 = gradGamma.Handle;
                IntPtr _p5 = gradBeta.Handle;
                void** args2 = stackalloc void*[8];
                args2[0] = &_p0;
                args2[1] = &_p1;
                args2[2] = &_p2;
                args2[3] = &_p3;
                args2[4] = &_p4;
                args2[5] = &_p5;
                args2[6] = &batchSize;
                args2[7] = &normalizedSize;


                uint grid2 = (uint)((normalizedSize + DefaultBlockSize - 1) / DefaultBlockSize);
                LaunchKernel(gradKrnl, grid2, DefaultBlockSize, args2);
                Synchronize();
                }
        }
    }

    public unsafe void GroupNorm(IGpuBuffer input, IGpuBuffer output, IGpuBuffer gamma, IGpuBuffer beta,
        IGpuBuffer saveMean, IGpuBuffer saveInvVar, int batch, int numGroups, int channels, int spatialSize, float epsilon)
    {
        if (!_kernelCache.TryGetValue("groupnorm_forward", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: groupnorm_forward");

            {
            IntPtr _p0 = input.Handle;
            IntPtr _p1 = output.Handle;
            IntPtr _p2 = gamma.Handle;
            IntPtr _p3 = beta.Handle;
            IntPtr _p4 = saveMean.Handle;
            IntPtr _p5 = saveInvVar.Handle;
            void** args = stackalloc void*[11];
            args[0] = &_p0;
            args[1] = &_p1;
            args[2] = &_p2;
            args[3] = &_p3;
            args[4] = &_p4;
            args[5] = &_p5;
            args[6] = &batch;
            args[7] = &numGroups;
            args[8] = &channels;
            args[9] = &spatialSize;
            args[10] = &epsilon;


            uint grid = (uint)((batch * numGroups + DefaultBlockSize - 1) / DefaultBlockSize);
            LaunchKernel(krnl, grid, DefaultBlockSize, args);
            Synchronize();
            }
    }

    public unsafe void InstanceNorm(IGpuBuffer input, IGpuBuffer output, IGpuBuffer gamma, IGpuBuffer beta,
        IGpuBuffer saveMean, IGpuBuffer saveInvVar, int batch, int channels, int spatialSize, float epsilon)
    {
        if (!_kernelCache.TryGetValue("instancenorm_forward", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: instancenorm_forward");

            {
            IntPtr _p0 = input.Handle;
            IntPtr _p1 = output.Handle;
            IntPtr _p2 = gamma.Handle;
            IntPtr _p3 = beta.Handle;
            IntPtr _p4 = saveMean.Handle;
            IntPtr _p5 = saveInvVar.Handle;
            void** args = stackalloc void*[10];
            args[0] = &_p0;
            args[1] = &_p1;
            args[2] = &_p2;
            args[3] = &_p3;
            args[4] = &_p4;
            args[5] = &_p5;
            args[6] = &batch;
            args[7] = &channels;
            args[8] = &spatialSize;
            args[9] = &epsilon;


            uint grid = (uint)((batch * channels + DefaultBlockSize - 1) / DefaultBlockSize);
            LaunchKernel(krnl, grid, DefaultBlockSize, args);
            Synchronize();
            }
    }

    public unsafe void InstanceNormBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gamma,
        IGpuBuffer saveMean, IGpuBuffer saveInvVar,
        IGpuBuffer gradInput, IGpuBuffer gradGamma, IGpuBuffer gradBeta,
        int batch, int channels, int spatialSize, float epsilon)
    {
        if (!IsAvailable)
            throw new InvalidOperationException("HIP backend is not available.");

        // Validate input dimensions
        long expectedInputSize = (long)batch * channels * spatialSize;
        if (expectedInputSize > int.MaxValue)
            throw new ArgumentException($"InstanceNormBackward: Total size {expectedInputSize} exceeds int.MaxValue.");

        int totalSize = (int)expectedInputSize;
        int statsSize = batch * channels;

        if (gradOutput.Size < totalSize)
            throw new ArgumentException($"InstanceNormBackward: gradOutput size {gradOutput.Size} is less than expected {totalSize}.");
        if (input.Size < totalSize)
            throw new ArgumentException($"InstanceNormBackward: input size {input.Size} is less than expected {totalSize}.");
        if (gamma.Size < channels)
            throw new ArgumentException($"InstanceNormBackward: gamma size {gamma.Size} is less than channels {channels}.");
        if (saveMean.Size < statsSize)
            throw new ArgumentException($"InstanceNormBackward: saveMean size {saveMean.Size} is less than expected {statsSize}.");
        if (saveInvVar.Size < statsSize)
            throw new ArgumentException($"InstanceNormBackward: saveInvVar size {saveInvVar.Size} is less than expected {statsSize}.");
        if (gradInput.Size < totalSize)
            throw new ArgumentException($"InstanceNormBackward: gradInput size {gradInput.Size} is less than expected {totalSize}.");
        if (gradGamma.Size < channels)
            throw new ArgumentException($"InstanceNormBackward: gradGamma size {gradGamma.Size} is less than channels {channels}.");
        if (gradBeta.Size < channels)
            throw new ArgumentException($"InstanceNormBackward: gradBeta size {gradBeta.Size} is less than channels {channels}.");

        // Try to use GPU kernels
        if (_kernelCache.TryGetValue("instancenorm_backward_sums", out var sumsKernel) &&
            _kernelCache.TryGetValue("instancenorm_backward", out var backwardKernel))
        {
            // Allocate temporary buffers for intermediate sums
            using var sumDy = AllocateBuffer(statsSize);
            using var sumDyXhat = AllocateBuffer(statsSize);

            // Zero out accumulated buffers (gradGamma, gradBeta use atomicAdd)
            Fill(gradGamma, 0f, channels);
            Fill(gradBeta, 0f, channels);
            Fill(sumDy, 0f, statsSize);
            Fill(sumDyXhat, 0f, statsSize);

            // Kernel expects N, C, H, W; we have batch, channels, spatialSize
            // Set H=spatialSize, W=1 to match the expected layout
            int N = batch;
            int C = channels;
            int H = spatialSize;
            int W = 1;

            uint grid = (uint)((totalSize + DefaultBlockSize - 1) / DefaultBlockSize);

            // Pass 1: Compute sums for gradient correction
                {
                IntPtr _p0 = gradOutput.Handle;
                IntPtr _p1 = input.Handle;
                IntPtr _p2 = saveMean.Handle;
                IntPtr _p3 = saveInvVar.Handle;
                IntPtr _p4 = gamma.Handle;
                IntPtr _p5 = sumDy.Handle;
                IntPtr _p6 = sumDyXhat.Handle;
                IntPtr _p7 = gradGamma.Handle;
                IntPtr _p8 = gradBeta.Handle;
                void** args1 = stackalloc void*[13];
                args1[0] = &_p0;
                args1[1] = &_p1;
                args1[2] = &_p2;
                args1[3] = &_p3;
                args1[4] = &_p4;
                args1[5] = &_p5;
                args1[6] = &_p6;
                args1[7] = &_p7;
                args1[8] = &_p8;
                args1[9] = &N;
                args1[10] = &C;
                args1[11] = &H;
                args1[12] = &W;


                LaunchKernel(sumsKernel, grid, DefaultBlockSize, args1);
                Synchronize();
                }

            // Pass 2: Compute final input gradients
                {
                IntPtr _p0 = gradOutput.Handle;
                IntPtr _p1 = input.Handle;
                IntPtr _p2 = saveMean.Handle;
                IntPtr _p3 = saveInvVar.Handle;
                IntPtr _p4 = gamma.Handle;
                IntPtr _p5 = sumDy.Handle;
                IntPtr _p6 = sumDyXhat.Handle;
                IntPtr _p7 = gradInput.Handle;
                void** args2 = stackalloc void*[12];
                args2[0] = &_p0;
                args2[1] = &_p1;
                args2[2] = &_p2;
                args2[3] = &_p3;
                args2[4] = &_p4;
                args2[5] = &_p5;
                args2[6] = &_p6;
                args2[7] = &_p7;
                args2[8] = &N;
                args2[9] = &C;
                args2[10] = &H;
                args2[11] = &W;


                LaunchKernel(backwardKernel, grid, DefaultBlockSize, args2);
                Synchronize();
                }

            return;
        }

        // CPU fallback only if kernels are not available
        System.Diagnostics.Debug.WriteLine("HipBackend InstanceNormBackward is executing on CPU fallback; GPU kernels not available.");
        InstanceNormBackwardCpuFallback(gradOutput, input, gamma, saveMean, saveInvVar,
            gradInput, gradGamma, gradBeta, batch, channels, spatialSize);
    }

    private void InstanceNormBackwardCpuFallback(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gamma,
        IGpuBuffer saveMean, IGpuBuffer saveInvVar,
        IGpuBuffer gradInput, IGpuBuffer gradGamma, IGpuBuffer gradBeta,
        int batch, int channels, int spatialSize)
    {
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

        // Upload results to GPU buffers using hipMemcpy
        var handle1 = GCHandle.Alloc(gradInputData, GCHandleType.Pinned);
        try
        {
            var result = HipNativeBindings.hipMemcpy(
                gradInput.Handle,
                handle1.AddrOfPinnedObject(),
                (UIntPtr)(gradInputData.Length * sizeof(float)),
                HipMemcpyKind.HostToDevice);
            HipNativeBindings.CheckError(result, "hipMemcpy H2D (InstanceNormBackward gradInput)");
        }
        finally
        {
            handle1.Free();
        }

        var handle2 = GCHandle.Alloc(gradGammaData, GCHandleType.Pinned);
        try
        {
            var result = HipNativeBindings.hipMemcpy(
                gradGamma.Handle,
                handle2.AddrOfPinnedObject(),
                (UIntPtr)(gradGammaData.Length * sizeof(float)),
                HipMemcpyKind.HostToDevice);
            HipNativeBindings.CheckError(result, "hipMemcpy H2D (InstanceNormBackward gradGamma)");
        }
        finally
        {
            handle2.Free();
        }

        var handle3 = GCHandle.Alloc(gradBetaData, GCHandleType.Pinned);
        try
        {
            var result = HipNativeBindings.hipMemcpy(
                gradBeta.Handle,
                handle3.AddrOfPinnedObject(),
                (UIntPtr)(gradBetaData.Length * sizeof(float)),
                HipMemcpyKind.HostToDevice);
            HipNativeBindings.CheckError(result, "hipMemcpy H2D (InstanceNormBackward gradBeta)");
        }
        finally
        {
            handle3.Free();
        }
    }

    public unsafe void RmsNorm(IGpuBuffer input, IGpuBuffer output, IGpuBuffer gamma, IGpuBuffer saveRms,
        int batchSize, int normalizedSize, float epsilon)
    {
        if (!_kernelCache.TryGetValue("rmsnorm_forward", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: rmsnorm_forward");

            {
            IntPtr _p0 = input.Handle;
            IntPtr _p1 = output.Handle;
            IntPtr _p2 = gamma.Handle;
            IntPtr _p3 = saveRms.Handle;
            void** args = stackalloc void*[7];
            args[0] = &_p0;
            args[1] = &_p1;
            args[2] = &_p2;
            args[3] = &_p3;
            args[4] = &batchSize;
            args[5] = &normalizedSize;
            args[6] = &epsilon;


            uint grid = (uint)((batchSize + DefaultBlockSize - 1) / DefaultBlockSize);
            LaunchKernel(krnl, grid, DefaultBlockSize, args);
            Synchronize();
            }
    }

    public unsafe void RmsNormBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gamma, IGpuBuffer saveRms,
        IGpuBuffer gradInput, IGpuBuffer gradGamma, int batchSize, int normalizedSize, float epsilon)
    {
        if (!_kernelCache.TryGetValue("rmsnorm_backward", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: rmsnorm_backward");

            {
            IntPtr _p0 = gradOutput.Handle;
            IntPtr _p1 = input.Handle;
            IntPtr _p2 = gamma.Handle;
            IntPtr _p3 = saveRms.Handle;
            IntPtr _p4 = gradInput.Handle;
            IntPtr _p5 = gradGamma.Handle;
            void** args = stackalloc void*[9];
            args[0] = &_p0;
            args[1] = &_p1;
            args[2] = &_p2;
            args[3] = &_p3;
            args[4] = &_p4;
            args[5] = &_p5;
            args[6] = &batchSize;
            args[7] = &normalizedSize;
            args[8] = &epsilon;


            uint grid = (uint)batchSize;
            LaunchKernel(krnl, grid, DefaultBlockSize, args);
            Synchronize();
            }

        // Compute gradGamma
        if (!_kernelCache.TryGetValue("rmsnorm_grad_gamma", out var krnl2))
            throw new InvalidOperationException("HIP kernel not found: rmsnorm_grad_gamma");

            {
            IntPtr _p0 = gradOutput.Handle;
            IntPtr _p1 = input.Handle;
            IntPtr _p2 = saveRms.Handle;
            IntPtr _p3 = gradGamma.Handle;
            void** args2 = stackalloc void*[6];
            args2[0] = &_p0;
            args2[1] = &_p1;
            args2[2] = &_p2;
            args2[3] = &_p3;
            args2[4] = &batchSize;
            args2[5] = &normalizedSize;


            uint grid2 = (uint)((normalizedSize + DefaultBlockSize - 1) / DefaultBlockSize);
            LaunchKernel(krnl2, grid2, DefaultBlockSize, args2);
            Synchronize();
            }
    }

    #endregion

    #region Dropout Operations

    public unsafe void Dropout(IGpuBuffer input, IGpuBuffer output, IGpuBuffer mask, int size, float dropoutRate, ulong seed, bool training)
    {
        if (!training)
        {
            // During inference, just copy input to output
            Copy(input, output, size);
            return;
        }

        // Generate mask using CPU and upload (GPU RNG would need additional kernel)
        float scale = 1.0f / (1.0f - dropoutRate);
        var rand = new Random((int)(seed % int.MaxValue));
        var maskData = new float[size];
        for (int i = 0; i < size; i++)
            maskData[i] = rand.NextDouble() > dropoutRate ? 1.0f : 0.0f;

        // Upload mask to GPU
        fixed (float* ptr = maskData)
        {
            var result = HipNativeBindings.hipMemcpy(
                mask.Handle, (IntPtr)ptr,
                (UIntPtr)(size * sizeof(float)),
                HipMemcpyKind.HostToDevice);
            HipNativeBindings.CheckError(result, "hipMemcpy H2D (mask)");
        }

        if (!_kernelCache.TryGetValue("dropout_forward", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: dropout_forward");

            {
            IntPtr _p0 = input.Handle;
            IntPtr _p1 = output.Handle;
            IntPtr _p2 = mask.Handle;
            void** args = stackalloc void*[5];
            args[0] = &_p0;
            args[1] = &_p1;
            args[2] = &_p2;
            args[3] = &scale;
            args[4] = &size;


            uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
            LaunchKernel(krnl, grid, DefaultBlockSize, args);
            Synchronize();
            }
    }

    public unsafe void DropoutBackward(IGpuBuffer gradOutput, IGpuBuffer mask, IGpuBuffer gradInput, int size, float dropoutRate)
    {
        if (!_kernelCache.TryGetValue("dropout_backward", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: dropout_backward");

        float scale = 1.0f / (1.0f - dropoutRate);
            {
            IntPtr _p0 = gradOutput.Handle;
            IntPtr _p1 = mask.Handle;
            IntPtr _p2 = gradInput.Handle;
            void** args = stackalloc void*[5];
            args[0] = &_p0;
            args[1] = &_p1;
            args[2] = &_p2;
            args[3] = &scale;
            args[4] = &size;


            uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
            LaunchKernel(krnl, grid, DefaultBlockSize, args);
            Synchronize();
            }
    }

    public unsafe bool TryFusedBiasDropout(IGpuBuffer input, IGpuBuffer output, IGpuBuffer bias, IGpuBuffer mask,
        int rows, int cols, float dropoutRate, float scale)
    {
        if (!_kernelCache.TryGetValue("bias_dropout", out var kernel))
            return false;

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
        Synchronize();
        return true;
    }

    #endregion

    #region Embedding Operations

    public unsafe void Embedding(IGpuBuffer indices, IGpuBuffer embeddingTable, IGpuBuffer output, int numIndices, int embeddingDim)
    {
        if (!_kernelCache.TryGetValue("embedding_forward", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: embedding_forward");

            {
            IntPtr _p0 = indices.Handle;
            IntPtr _p1 = embeddingTable.Handle;
            IntPtr _p2 = output.Handle;
            void** args = stackalloc void*[5];
            args[0] = &_p0;
            args[1] = &_p1;
            args[2] = &_p2;
            args[3] = &numIndices;
            args[4] = &embeddingDim;


            uint grid = (uint)((numIndices + DefaultBlockSize - 1) / DefaultBlockSize);
            LaunchKernel(krnl, grid, DefaultBlockSize, args);
            Synchronize();
            }
    }

    public unsafe void EmbeddingBackward(IGpuBuffer gradOutput, IGpuBuffer indices, IGpuBuffer gradEmbedding, int numIndices, int embeddingDim, int vocabSize)
    {
        if (!_kernelCache.TryGetValue("embedding_backward", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: embedding_backward");

            {
            IntPtr _p0 = gradOutput.Handle;
            IntPtr _p1 = indices.Handle;
            IntPtr _p2 = gradEmbedding.Handle;
            void** args = stackalloc void*[5];
            args[0] = &_p0;
            args[1] = &_p1;
            args[2] = &_p2;
            args[3] = &numIndices;
            args[4] = &embeddingDim;


            uint grid = (uint)((numIndices + DefaultBlockSize - 1) / DefaultBlockSize);
            LaunchKernel(krnl, grid, DefaultBlockSize, args);
            Synchronize();
            }
    }

    public IGpuBuffer AllocateIntBuffer(int size)
    {
        IntPtr devicePtr = IntPtr.Zero;
        var sizeBytes = (UIntPtr)(size * sizeof(int));

        var result = HipNativeBindings.hipMalloc(ref devicePtr, sizeBytes);
        HipNativeBindings.CheckError(result, "hipMalloc(int)");

        result = HipNativeBindings.hipMemset(devicePtr, 0, sizeBytes);
        HipNativeBindings.CheckError(result, "hipMemset(int)");

        return new HipGpuBuffer(devicePtr, size);
    }

    public IGpuBuffer AllocateIntBuffer(int[] data)
    {
        IntPtr devicePtr = IntPtr.Zero;
        var size = data.Length;
        var sizeBytes = (UIntPtr)(size * sizeof(int));

        var result = HipNativeBindings.hipMalloc(ref devicePtr, sizeBytes);
        HipNativeBindings.CheckError(result, "hipMalloc(int)");

        GCHandle handle = GCHandle.Alloc(data, GCHandleType.Pinned);
        try
        {
            result = HipNativeBindings.hipMemcpy(
                devicePtr,
                handle.AddrOfPinnedObject(),
                sizeBytes,
                HipMemcpyKind.HostToDevice);
            HipNativeBindings.CheckError(result, "hipMemcpy H2D (int)");
        }
        finally
        {
            handle.Free();
        }

        return new HipGpuBuffer(devicePtr, size);
    }

    #region Locally Connected Convolution Operations

    public unsafe void LocallyConnectedConv2D(IGpuBuffer input, IGpuBuffer weights, IGpuBuffer? bias, IGpuBuffer output,
        int batch, int inChannels, int inHeight, int inWidth,
        int outChannels, int outHeight, int outWidth,
        int kernelH, int kernelW,
        int strideH, int strideW)
    {
        if (!_kernelCache.TryGetValue("locally_connected_conv2d", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: locally_connected_conv2d");

            {
            var pInput = input.Handle;
            var pWeights = weights.Handle;
            var pBias = bias?.Handle ?? IntPtr.Zero;
            var pOutput = output.Handle;
            int hasBias = bias != null ? 1 : 0;
            void** args = stackalloc void*[16];
            args[0] = &pInput;
            args[1] = &pWeights;
            args[2] = &pBias;
            args[3] = &pOutput;
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



            // Grid dimensions: outWidth x outHeight x (batch * outChannels)
            uint gridX = (uint)((outWidth + 15) / 16);
            uint gridY = (uint)((outHeight + 15) / 16);
            uint gridZ = (uint)(batch * outChannels);
            LaunchKernel3D(krnl, gridX, gridY, gridZ, 16, 16, 1, args);
            Synchronize();

            }
    }

    public unsafe void LocallyConnectedConv2DBackwardInput(IGpuBuffer gradOutput, IGpuBuffer weights, IGpuBuffer gradInput,
        int batch, int inChannels, int inHeight, int inWidth,
        int outChannels, int outHeight, int outWidth,
        int kernelH, int kernelW,
        int strideH, int strideW)
    {
        if (!_kernelCache.TryGetValue("locally_connected_conv2d_backward_input", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: locally_connected_conv2d_backward_input");

            {
            IntPtr _p0 = gradOutput.Handle;
            IntPtr _p1 = weights.Handle;
            IntPtr _p2 = gradInput.Handle;
            void** args = stackalloc void*[14];
            args[0] = &_p0;
            args[1] = &_p1;
            args[2] = &_p2;
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


            int totalSize = batch * inChannels * inHeight * inWidth;
            uint grid = (uint)((totalSize + DefaultBlockSize - 1) / DefaultBlockSize);
            LaunchKernel(krnl, grid, DefaultBlockSize, args);
            Synchronize();
            }
    }

    public unsafe void LocallyConnectedConv2DBackwardWeights(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradWeights,
        int batch, int inChannels, int inHeight, int inWidth,
        int outChannels, int outHeight, int outWidth,
        int kernelH, int kernelW,
        int strideH, int strideW)
    {
        if (!_kernelCache.TryGetValue("locally_connected_conv2d_backward_weights", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: locally_connected_conv2d_backward_weights");

            {
            IntPtr _p0 = gradOutput.Handle;
            IntPtr _p1 = input.Handle;
            IntPtr _p2 = gradWeights.Handle;
            void** args = stackalloc void*[14];
            args[0] = &_p0;
            args[1] = &_p1;
            args[2] = &_p2;
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


            int totalWeights = outHeight * outWidth * outChannels * inChannels * kernelH * kernelW;
            uint grid = (uint)((totalWeights + DefaultBlockSize - 1) / DefaultBlockSize);
            LaunchKernel(krnl, grid, DefaultBlockSize, args);
            Synchronize();
            }
    }

    public unsafe void LocallyConnectedConv2DBackwardBias(IGpuBuffer gradOutput, IGpuBuffer gradBias,
        int batch, int outChannels, int outHeight, int outWidth)
    {
        if (!_kernelCache.TryGetValue("locally_connected_conv2d_backward_bias", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: locally_connected_conv2d_backward_bias");

            {
            IntPtr _p0 = gradOutput.Handle;
            IntPtr _p1 = gradBias.Handle;
            void** args = stackalloc void*[6];
            args[0] = &_p0;
            args[1] = &_p1;
            args[2] = &batch;
            args[3] = &outChannels;
            args[4] = &outHeight;
            args[5] = &outWidth;


            uint grid = (uint)((outChannels + DefaultBlockSize - 1) / DefaultBlockSize);
            LaunchKernel(krnl, grid, DefaultBlockSize, args);
            Synchronize();
            }
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
        if (!_kernelCache.TryGetValue("deformable_conv2d", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: deformable_conv2d");

            {
            var pInput = input.Handle;
            var pWeights = weights.Handle;
            var pOffsets = offsets.Handle;
            var pMask = mask?.Handle ?? IntPtr.Zero;
            var pOutput = output.Handle;
            int hasMask = mask != null ? 1 : 0;
            void** args = stackalloc void*[23];
            args[0] = &pInput;
            args[1] = &pWeights;
            args[2] = &pOffsets;
            args[3] = &pMask;
            args[4] = &pOutput;
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

            // Grid dimensions: outWidth x outHeight x (batch * outChannels)
            uint gridX = (uint)((outWidth + 15) / 16);
            uint gridY = (uint)((outHeight + 15) / 16);
            uint gridZ = (uint)(batch * outChannels);
            LaunchKernel3D(krnl, gridX, gridY, gridZ, 16, 16, 1, args);
            Synchronize();

            }
    }

    public unsafe void DeformableConv2DBackwardInput(IGpuBuffer gradOutput, IGpuBuffer weights, IGpuBuffer offsets, IGpuBuffer? mask, IGpuBuffer gradInput,
        int batch, int inChannels, int inHeight, int inWidth,
        int outChannels, int outHeight, int outWidth,
        int kernelH, int kernelW,
        int strideH, int strideW, int padH, int padW,
        int dilationH, int dilationW,
        int groups, int deformGroups)
    {
        if (!_kernelCache.TryGetValue("deformable_conv2d_backward_input", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: deformable_conv2d_backward_input");

            {
            var pGradOutput = gradOutput.Handle;
            var pWeights = weights.Handle;
            var pOffsets = offsets.Handle;
            var pMask = mask?.Handle ?? IntPtr.Zero;
            var pGradInput = gradInput.Handle;
            int hasMask = mask != null ? 1 : 0;
            void** args = stackalloc void*[23];
            args[0] = &pGradOutput;
            args[1] = &pWeights;
            args[2] = &pOffsets;
            args[3] = &pMask;
            args[4] = &pGradInput;
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

            int totalInputSize = batch * inChannels * inHeight * inWidth;
            uint grid = (uint)((totalInputSize + DefaultBlockSize - 1) / DefaultBlockSize);
            LaunchKernel(krnl, grid, DefaultBlockSize, args);
            Synchronize();

            }
    }

    public unsafe void DeformableConv2DBackwardWeights(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer offsets, IGpuBuffer? mask, IGpuBuffer gradWeights,
        int batch, int inChannels, int inHeight, int inWidth,
        int outChannels, int outHeight, int outWidth,
        int kernelH, int kernelW,
        int strideH, int strideW, int padH, int padW,
        int dilationH, int dilationW,
        int groups, int deformGroups)
    {
        if (!_kernelCache.TryGetValue("deformable_conv2d_backward_weights", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: deformable_conv2d_backward_weights");

            {
            var pGradOutput = gradOutput.Handle;
            var pInput = input.Handle;
            var pOffsets = offsets.Handle;
            var pMask = mask?.Handle ?? IntPtr.Zero;
            var pGradWeights = gradWeights.Handle;
            int hasMask = mask != null ? 1 : 0;
            void** args = stackalloc void*[23];
            args[0] = &pGradOutput;
            args[1] = &pInput;
            args[2] = &pOffsets;
            args[3] = &pMask;
            args[4] = &pGradWeights;
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

            int inChannelsPerGroup = inChannels / groups;
            int totalWeights = outChannels * inChannelsPerGroup * kernelH * kernelW;
            uint grid = (uint)((totalWeights + DefaultBlockSize - 1) / DefaultBlockSize);
            LaunchKernel(krnl, grid, DefaultBlockSize, args);
            Synchronize();

            }
    }

    public unsafe void DeformableConv2DBackwardOffset(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer weights, IGpuBuffer offsets, IGpuBuffer? mask, IGpuBuffer gradOffsets,
        int batch, int inChannels, int inHeight, int inWidth,
        int outChannels, int outHeight, int outWidth,
        int kernelH, int kernelW,
        int strideH, int strideW, int padH, int padW,
        int dilationH, int dilationW,
        int groups, int deformGroups)
    {
        if (!_kernelCache.TryGetValue("deformable_conv2d_backward_offset", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: deformable_conv2d_backward_offset");

            {
            var pGradOutput = gradOutput.Handle;
            var pInput = input.Handle;
            var pWeights = weights.Handle;
            var pOffsets = offsets.Handle;
            var pMask = mask?.Handle ?? IntPtr.Zero;
            var pGradOffsets = gradOffsets.Handle;
            int hasMask = mask != null ? 1 : 0;
            void** args = stackalloc void*[24];
            args[0] = &pGradOutput;
            args[1] = &pInput;
            args[2] = &pWeights;
            args[3] = &pOffsets;
            args[4] = &pMask;
            args[5] = &pGradOffsets;
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


            // Grid for offset gradients: outWidth x outHeight x (batch * 2*kH*kW*deformGroups)
            int offsetChannels = 2 * kernelH * kernelW * deformGroups;
            uint gridX = (uint)((outWidth + 15) / 16);
            uint gridY = (uint)((outHeight + 15) / 16);
            uint gridZ = (uint)(batch * offsetChannels);
            LaunchKernel3D(krnl, gridX, gridY, gridZ, 16, 16, 1, args);
            Synchronize();
            }
    }

    public unsafe void DeformableConv2DBackwardMask(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer weights, IGpuBuffer offsets, IGpuBuffer gradMask,
        int batch, int inChannels, int inHeight, int inWidth,
        int outChannels, int outHeight, int outWidth,
        int kernelH, int kernelW,
        int strideH, int strideW, int padH, int padW,
        int dilationH, int dilationW,
        int groups, int deformGroups)
    {
        if (!_kernelCache.TryGetValue("deformable_conv2d_backward_mask", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: deformable_conv2d_backward_mask");

            {
            var pGradOutput = gradOutput.Handle;
            var pInput = input.Handle;
            var pWeights = weights.Handle;
            var pOffsets = offsets.Handle;
            var pGradMask = gradMask.Handle;
            void** args = stackalloc void*[22];
            args[0] = &pGradOutput;
            args[1] = &pInput;
            args[2] = &pWeights;
            args[3] = &pOffsets;
            args[4] = &pGradMask;
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


            // Grid for mask gradients: outWidth x outHeight x (batch * kH*kW*deformGroups)
            int maskChannels = kernelH * kernelW * deformGroups;
            uint gridX = (uint)((outWidth + 15) / 16);
            uint gridY = (uint)((outHeight + 15) / 16);
            uint gridZ = (uint)(batch * maskChannels);
            LaunchKernel3D(krnl, gridX, gridY, gridZ, 16, 16, 1, args);
            Synchronize();
            }
    }

    #endregion

    #endregion

    #region Attention Operations

    public unsafe void ScaledDotProductAttention(IGpuBuffer query, IGpuBuffer key, IGpuBuffer value,
        IGpuBuffer output, IGpuBuffer? attentionWeights, IGpuBuffer? mask,
        int batch, int numHeads, int seqLen, int headDim, float scale, bool isCausal)
    {
        // Compute attention: softmax((Q @ K^T) * scale) @ V
        int totalBatches = batch * numHeads;
        int matrixSize = seqLen * headDim;

        // Allocate temporary buffers for Q @ K^T result (scores)
        using var scores = AllocateBuffer(new float[totalBatches * seqLen * seqLen]);
        using var keyT = AllocateBuffer(new float[totalBatches * headDim * seqLen]);

        // Transpose K for each batch
        BatchedTranspose(key, keyT, totalBatches, seqLen, headDim);

        // Q @ K^T for each batch
        for (int b = 0; b < totalBatches; b++)
        {
            int qOffset = b * matrixSize;
            int kOffset = b * matrixSize;
            int sOffset = b * seqLen * seqLen;

            // Use offset-based gemm or slice buffers
            using var qSlice = AllocateBuffer(new float[seqLen * headDim]);
            using var kSlice = AllocateBuffer(new float[headDim * seqLen]);
            using var sSlice = AllocateBuffer(new float[seqLen * seqLen]);

            Copy(query, qSlice, seqLen * headDim);
            Copy(keyT, kSlice, headDim * seqLen);
            Gemm(qSlice, kSlice, sSlice, seqLen, seqLen, headDim);
            Scale(sSlice, sSlice, scale, seqLen * seqLen);
            Softmax(sSlice, sSlice, seqLen, seqLen);

            // Store attention weights if requested
            if (attentionWeights is not null)
                Copy(sSlice, attentionWeights, seqLen * seqLen);

            // scores @ V
            using var outSlice = AllocateBuffer(new float[seqLen * headDim]);
            using var vSlice = AllocateBuffer(new float[seqLen * headDim]);
            Copy(value, vSlice, seqLen * headDim);
            Gemm(sSlice, vSlice, outSlice, seqLen, headDim, seqLen);
            Copy(outSlice, output, seqLen * headDim);
        }
    }

    public void ScaledDotProductAttentionBackward(IGpuBuffer gradOutput, IGpuBuffer query, IGpuBuffer key, IGpuBuffer value,
        IGpuBuffer attentionWeights, IGpuBuffer gradQuery, IGpuBuffer gradKey, IGpuBuffer gradValue,
        int batch, int numHeads, int seqLen, int headDim, float scale, bool isCausal)
    {
        // Simplified backward pass using stored attention weights
        int totalBatches = batch * numHeads;

        for (int b = 0; b < totalBatches; b++)
        {
            // gradValue = attentionWeights^T @ gradOutput
            using var attnT = AllocateBuffer(new float[seqLen * seqLen]);
            using var attnSlice = AllocateBuffer(new float[seqLen * seqLen]);
            using var gradOutSlice = AllocateBuffer(new float[seqLen * headDim]);
            using var gradVSlice = AllocateBuffer(new float[seqLen * headDim]);

            Copy(attentionWeights, attnSlice, seqLen * seqLen);
            Transpose(attnSlice, attnT, seqLen, seqLen);
            Copy(gradOutput, gradOutSlice, seqLen * headDim);
            Gemm(attnT, gradOutSlice, gradVSlice, seqLen, headDim, seqLen);
            Copy(gradVSlice, gradValue, seqLen * headDim);

            // gradQuery = (gradScores @ K) where gradScores = gradAttn * scale
            using var gradScores = AllocateBuffer(new float[seqLen * seqLen]);
            using var vSlice = AllocateBuffer(new float[seqLen * headDim]);
            Copy(value, vSlice, seqLen * headDim);
            using var vT = AllocateBuffer(new float[headDim * seqLen]);
            Transpose(vSlice, vT, seqLen, headDim);
            Gemm(gradOutSlice, vT, gradScores, seqLen, seqLen, headDim);

            // Apply softmax backward
            using var softmaxGrad = AllocateBuffer(new float[seqLen * seqLen]);
            SoftmaxBackward(gradScores, attnSlice, softmaxGrad, seqLen, seqLen);
            Scale(softmaxGrad, softmaxGrad, scale, seqLen * seqLen);

            // gradQuery = gradScores @ K
            using var kSlice = AllocateBuffer(new float[seqLen * headDim]);
            using var gradQSlice = AllocateBuffer(new float[seqLen * headDim]);
            Copy(key, kSlice, seqLen * headDim);
            Gemm(softmaxGrad, kSlice, gradQSlice, seqLen, headDim, seqLen);
            Copy(gradQSlice, gradQuery, seqLen * headDim);

            // gradKey = gradScores^T @ Q
            using var gradScoresT = AllocateBuffer(new float[seqLen * seqLen]);
            Transpose(softmaxGrad, gradScoresT, seqLen, seqLen);
            using var qSlice = AllocateBuffer(new float[seqLen * headDim]);
            using var gradKSlice = AllocateBuffer(new float[seqLen * headDim]);
            Copy(query, qSlice, seqLen * headDim);
            Gemm(gradScoresT, qSlice, gradKSlice, seqLen, headDim, seqLen);
            Copy(gradKSlice, gradKey, seqLen * headDim);
        }
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
        if (!_kernelCache.TryGetValue("flash_attention_v2", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: flash_attention_v2");

        int causalFlag = isCausal ? 1 : 0;
        int hasBias = attentionBias is not null ? 1 : 0;
        IntPtr biasPtr = attentionBias is not null ? attentionBias.Handle : IntPtr.Zero;
            {
            IntPtr _p0 = query.Handle;
            IntPtr _p1 = key.Handle;
            IntPtr _p2 = value.Handle;
            IntPtr _p3 = output.Handle;
            IntPtr _p4 = softmaxStats.Handle;
            void** args = stackalloc void*[15];
            args[0] = &_p0;
            args[1] = &_p1;
            args[2] = &_p2;
            args[3] = &_p3;
            args[4] = &_p4;
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


            uint gridX = (uint)((seqQ + 31) / 32);
            uint gridY = (uint)(batch * numHeads);
            uint sharedBytes = (uint)(2 * 32 * headDim * sizeof(float));
            LaunchKernel2D(krnl, gridX, gridY, 32, 1, args, sharedBytes);
            Synchronize();
            }
    }

    public unsafe void FlashAttentionBackward(IGpuBuffer gradOutput, IGpuBuffer query, IGpuBuffer key, IGpuBuffer value,
        IGpuBuffer output, IGpuBuffer softmaxStats,
        IGpuBuffer gradQuery, IGpuBuffer gradKey, IGpuBuffer gradValue,
        int batch, int numHeads, int seqQ, int seqK, int headDim, float scale, bool isCausal,
        IGpuBuffer? attentionBias = null, int biasBatchStride = 0)
    {
        if (!_kernelCache.TryGetValue("flash_attention_backward", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: flash_attention_backward");

        int causalFlag = isCausal ? 1 : 0;
        int hasBias = attentionBias is not null ? 1 : 0;
        IntPtr biasPtr = attentionBias is not null ? attentionBias.Handle : IntPtr.Zero;
            {
            IntPtr _p0 = gradOutput.Handle;
            IntPtr _p1 = query.Handle;
            IntPtr _p2 = key.Handle;
            IntPtr _p3 = value.Handle;
            IntPtr _p4 = output.Handle;
            IntPtr _p5 = softmaxStats.Handle;
            IntPtr _p6 = gradQuery.Handle;
            IntPtr _p7 = gradKey.Handle;
            IntPtr _p8 = gradValue.Handle;
            void** args = stackalloc void*[19];
            args[0] = &_p0;
            args[1] = &_p1;
            args[2] = &_p2;
            args[3] = &_p3;
            args[4] = &_p4;
            args[5] = &_p5;
            args[6] = &_p6;
            args[7] = &_p7;
            args[8] = &_p8;
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


            uint gridX = (uint)((seqQ + 31) / 32);
            uint gridY = (uint)(batch * numHeads);
            uint sharedBytes = (uint)(2 * 32 * headDim * sizeof(float));
            LaunchKernel2D(krnl, gridX, gridY, 32, 1, args, sharedBytes);
            Synchronize();
            }
    }

    public unsafe void GroupedQueryAttention(IGpuBuffer query, IGpuBuffer key, IGpuBuffer value,
        IGpuBuffer output, IGpuBuffer? attentionWeights,
        int batch, int numQHeads, int numKVHeads, int seqQ, int seqK, int headDim, float scale, bool isCausal)
    {
        if (!_kernelCache.TryGetValue("grouped_query_attention", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: grouped_query_attention");

        int queriesPerKV = numQHeads / numKVHeads;
        int causalFlag = isCausal ? 1 : 0;
        int storeWeights = attentionWeights != null ? 1 : 0;
        IntPtr wPtr = attentionWeights?.Handle ?? IntPtr.Zero;

            {
            IntPtr _p0 = query.Handle;
            IntPtr _p1 = key.Handle;
            IntPtr _p2 = value.Handle;
            IntPtr _p3 = output.Handle;
            void** args = stackalloc void*[15];
            args[0] = &_p0;
            args[1] = &_p1;
            args[2] = &_p2;
            args[3] = &_p3;
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
            args[14] = &storeWeights;


            uint gridX = (uint)((seqQ + 31) / 32);
            uint gridY = (uint)(batch * numQHeads);
            uint sharedBytes = (uint)(2 * 32 * headDim * sizeof(float));
            LaunchKernel2D(krnl, gridX, gridY, 32, 1, args, sharedBytes);
            Synchronize();
            }
    }

    public unsafe void GroupedQueryAttentionBackward(IGpuBuffer gradOutput, IGpuBuffer query, IGpuBuffer key, IGpuBuffer value,
        IGpuBuffer attentionWeights,
        IGpuBuffer gradQuery, IGpuBuffer gradKey, IGpuBuffer gradValue,
        int batch, int numQHeads, int numKVHeads, int seqQ, int seqK, int headDim, float scale)
    {
        if (!_kernelCache.TryGetValue("grouped_query_attention_backward", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: grouped_query_attention_backward");

        int queriesPerKV = numQHeads / numKVHeads;

            {
            IntPtr _p0 = gradOutput.Handle;
            IntPtr _p1 = query.Handle;
            IntPtr _p2 = key.Handle;
            IntPtr _p3 = value.Handle;
            IntPtr _p4 = attentionWeights.Handle;
            IntPtr _p5 = gradQuery.Handle;
            IntPtr _p6 = gradKey.Handle;
            IntPtr _p7 = gradValue.Handle;
            void** args = stackalloc void*[16];
            args[0] = &_p0;
            args[1] = &_p1;
            args[2] = &_p2;
            args[3] = &_p3;
            args[4] = &_p4;
            args[5] = &_p5;
            args[6] = &_p6;
            args[7] = &_p7;
            args[8] = &batch;
            args[9] = &numQHeads;
            args[10] = &numKVHeads;
            args[11] = &queriesPerKV;
            args[12] = &seqQ;
            args[13] = &seqK;
            args[14] = &headDim;
            args[15] = &scale;


            uint gridX = (uint)((seqQ + 31) / 32);
            uint gridY = (uint)(batch * numQHeads);
            uint sharedBytes = (uint)(2 * 32 * headDim * sizeof(float));
            LaunchKernel2D(krnl, gridX, gridY, 32, 1, args, sharedBytes);
            Synchronize();
            }
    }

    #endregion

    #region Transpose and Reshape Operations

    public unsafe void Transpose(IGpuBuffer A, IGpuBuffer B, int rows, int cols)
    {
        if (!_kernelCache.TryGetValue("transpose_2d", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: transpose_2d");

            {
            IntPtr _p0 = A.Handle;
            IntPtr _p1 = B.Handle;
            void** args = stackalloc void*[4];
            args[0] = &_p0;
            args[1] = &_p1;
            args[2] = &rows;
            args[3] = &cols;


            uint gridX = (uint)((cols + 15) / 16);
            uint gridY = (uint)((rows + 15) / 16);
            LaunchKernel2D(krnl, gridX, gridY, 16, 16, args);
            Synchronize();
            }
    }

    public unsafe void BatchedTranspose(IGpuBuffer A, IGpuBuffer B, int batch, int rows, int cols)
    {
        if (!_kernelCache.TryGetValue("batched_transpose", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: batched_transpose");

            {
            IntPtr _p0 = A.Handle;
            IntPtr _p1 = B.Handle;
            void** args = stackalloc void*[5];
            args[0] = &_p0;
            args[1] = &_p1;
            args[2] = &batch;
            args[3] = &rows;
            args[4] = &cols;


            uint gridX = (uint)((cols + 15) / 16);
            uint gridY = (uint)((rows + 15) / 16);
            uint gridZ = (uint)batch;
            LaunchKernel3D(krnl, gridX, gridY, gridZ, 16, 16, 1, args);
            Synchronize();
            }
    }

    public unsafe void Permute(IGpuBuffer input, IGpuBuffer output, int[] shape, int[] permutation)
    {
        int ndims = shape.Length;
        if (ndims == 2 && permutation[0] == 1 && permutation[1] == 0)
        {
            Transpose(input, output, shape[0], shape[1]);
            return;
        }

        if (!_kernelCache.TryGetValue("permute_general", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: permute_general");

        int totalSize = 1;
        for (int i = 0; i < ndims; i++) totalSize *= shape[i];

        int[] inputStrides = new int[ndims];
        inputStrides[ndims - 1] = 1;
        for (int i = ndims - 2; i >= 0; i--)
            inputStrides[i] = inputStrides[i + 1] * shape[i + 1];

        int[] outputShape = new int[ndims];
        for (int i = 0; i < ndims; i++)
            outputShape[i] = shape[permutation[i]];

        int[] outputStrides = new int[ndims];
        outputStrides[ndims - 1] = 1;
        for (int i = ndims - 2; i >= 0; i--)
            outputStrides[i] = outputStrides[i + 1] * outputShape[i + 1];

        using var inputStridesBuffer = AllocateIntBuffer(inputStrides);
        using var outputStridesBuffer = AllocateIntBuffer(outputStrides);
        using var permutationBuffer = AllocateIntBuffer(permutation);

            {
            IntPtr _p0 = input.Handle;
            IntPtr _p1 = output.Handle;
            IntPtr _p2 = inputStridesBuffer.Handle;
            IntPtr _p3 = outputStridesBuffer.Handle;
            IntPtr _p4 = permutationBuffer.Handle;
            void** args = stackalloc void*[7];
            args[0] = &_p0;
            args[1] = &_p1;
            args[2] = &_p2;
            args[3] = &_p3;
            args[4] = &_p4;
            args[5] = &ndims;
            args[6] = &totalSize;


            uint grid = (uint)((totalSize + DefaultBlockSize - 1) / DefaultBlockSize);
            LaunchKernel(krnl, grid, DefaultBlockSize, args);
            Synchronize();
            }
    }

    public void Copy(IGpuBuffer source, IGpuBuffer destination, int size)
    {
        var sizeBytes = (UIntPtr)(size * sizeof(float));
        var result = HipNativeBindings.hipMemcpy(
            destination.Handle,
            source.Handle,
            sizeBytes,
            HipMemcpyKind.DeviceToDevice);
        HipNativeBindings.CheckError(result, "hipMemcpy D2D");
    }

    public unsafe void Fill(IGpuBuffer buffer, float value, int size)
    {
        if (!IsAvailable)
            throw new InvalidOperationException("HIP backend is not available.");

        // For zero, use hipMemset (efficient for 0 bytes = 0.0f in IEEE 754)
        if (value == 0.0f)
        {
            var sizeBytes = (UIntPtr)(size * sizeof(float));
            var result = HipNativeBindings.hipMemset(buffer.Handle, 0, sizeBytes);
            HipNativeBindings.CheckError(result, "hipMemset (Fill zero)");
            Synchronize();
            return;
        }

        // For non-zero values, use the fill_buffer kernel
        if (!_kernelCache.TryGetValue("fill_buffer", out var kernel))
            throw new InvalidOperationException("HIP kernel not found: fill_buffer");

        int grid = (size + DefaultBlockSize - 1) / DefaultBlockSize;
        var dstHandle = ((HipGpuBuffer)buffer).Handle;

        void*[] args = new void*[3];
        fixed (void** argsPtr = args)
        {
            IntPtr[] handles = [dstHandle];
            float[] floats = [value];
            int[] ints = [size];

            fixed (IntPtr* h = handles)
            fixed (float* f = floats)
            fixed (int* i = ints)
            {
                argsPtr[0] = &h[0];
                argsPtr[1] = &f[0];
                argsPtr[2] = &i[0];

                var result = HipNativeBindings.hipModuleLaunchKernel(
                    kernel,
                    (uint)grid, 1, 1,
                    (uint)DefaultBlockSize, 1, 1,
                    0, _stream, (IntPtr)argsPtr, IntPtr.Zero);
                HipNativeBindings.CheckError(result, "hipModuleLaunchKernel (fill_buffer)");
            }
        }
        Synchronize();
    }

    /// <inheritdoc/>
    public unsafe void Copy2DStrided(IGpuBuffer source, IGpuBuffer destination, int numRows, int srcCols, int destTotalCols, int destColOffset)
    {
        if (!IsAvailable)
            throw new InvalidOperationException("HIP backend is not available.");

        if (!_kernelCache.TryGetValue("copy_2d_strided", out var kernel))
            throw new InvalidOperationException("HIP kernel not found: copy_2d_strided");

        int gridX = (srcCols + DefaultBlockSize - 1) / DefaultBlockSize;
        int gridY = numRows;

        var srcHandle = ((HipGpuBuffer)source).Handle;
        var dstHandle = ((HipGpuBuffer)destination).Handle;

        void*[] args = new void*[6];
        fixed (void** argsPtr = args)
        {
            IntPtr[] handles = [srcHandle, dstHandle];
            int[] ints = [numRows, srcCols, destTotalCols, destColOffset];

            fixed (IntPtr* h = handles)
            fixed (int* i = ints)
            {
                argsPtr[0] = &h[0];
                argsPtr[1] = &h[1];
                argsPtr[2] = &i[0];
                argsPtr[3] = &i[1];
                argsPtr[4] = &i[2];
                argsPtr[5] = &i[3];

                var result = HipNativeBindings.hipModuleLaunchKernel(
                    kernel,
                    (uint)gridX, (uint)gridY, 1,
                    (uint)DefaultBlockSize, 1, 1,
                    0, _stream, (IntPtr)argsPtr, IntPtr.Zero);
                HipNativeBindings.CheckError(result, "hipModuleLaunchKernel (copy_2d_strided)");
            }
        }
    }

    /// <inheritdoc/>
    public unsafe void NearestNeighborUpsample(IGpuBuffer input, IGpuBuffer output, int batchChannels, int height, int width, int scaleFactor)
    {
        if (!IsAvailable)
            throw new InvalidOperationException("HIP backend is not available.");

        // Check for the kernel, if not available fall back to CPU implementation
        if (!_kernelCache.TryGetValue("nearest_neighbor_upsample", out var kernel))
        {
            NearestNeighborUpsampleFallback(input, output, batchChannels, height, width, scaleFactor);
            return;
        }

        int outHeight = height * scaleFactor;
        int outWidth = width * scaleFactor;
        int outputSize = batchChannels * outHeight * outWidth;

        int grid = (outputSize + DefaultBlockSize - 1) / DefaultBlockSize;

        var srcHandle = ((HipGpuBuffer)input).Handle;
        var dstHandle = ((HipGpuBuffer)output).Handle;

        void*[] args = new void*[7];
        fixed (void** argsPtr = args)
        {
            IntPtr[] handles = [srcHandle, dstHandle];
            int[] ints = [batchChannels, height, width, scaleFactor, outputSize];

            fixed (IntPtr* h = handles)
            fixed (int* i = ints)
            {
                argsPtr[0] = &h[0];
                argsPtr[1] = &h[1];
                argsPtr[2] = &i[0];
                argsPtr[3] = &i[1];
                argsPtr[4] = &i[2];
                argsPtr[5] = &i[3];
                argsPtr[6] = &i[4];

                var result = HipNativeBindings.hipModuleLaunchKernel(
                    kernel,
                    (uint)grid, 1, 1,
                    (uint)DefaultBlockSize, 1, 1,
                    0, _stream, (IntPtr)argsPtr, IntPtr.Zero);
                HipNativeBindings.CheckError(result, "hipModuleLaunchKernel (nearest_neighbor_upsample)");
            }
        }
    }

    /// <summary>
    /// CPU fallback for nearest-neighbor upsampling when kernel is not available.
    /// </summary>
    private void NearestNeighborUpsampleFallback(IGpuBuffer input, IGpuBuffer output, int batchChannels, int height, int width, int scaleFactor)
    {
        int outHeight = height * scaleFactor;
        int outWidth = width * scaleFactor;
        int outputSize = batchChannels * outHeight * outWidth;

        // Download input using existing method
        var inputData = DownloadBuffer(input);

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

        // Upload output using existing method
        UploadToBuffer(output, outputData);
    }

    /// <inheritdoc/>
    public unsafe void NearestNeighborUpsampleBackward(IGpuBuffer gradOutput, IGpuBuffer gradInput, int batchChannels, int height, int width, int scaleFactor)
    {
        if (!IsAvailable)
            throw new InvalidOperationException("HIP backend is not available.");

        // Calculate input size - kernel iterates over input elements, not output elements
        // This avoids race conditions by having each thread write to a unique input location
        int inputSize = batchChannels * height * width;

        if (!_kernelCache.TryGetValue("nearest_neighbor_upsample_backward", out var kernel))
        {
            NearestNeighborUpsampleBackwardFallback(gradOutput, gradInput, batchChannels, height, width, scaleFactor);
            return;
        }

        // Grid size based on input elements (each thread handles one input element)
        int grid = (inputSize + DefaultBlockSize - 1) / DefaultBlockSize;

        var gradOutHandle = ((HipGpuBuffer)gradOutput).Handle;
        var gradInHandle = ((HipGpuBuffer)gradInput).Handle;

        void*[] args = new void*[7];
        fixed (void** argsPtr = args)
        {
            IntPtr[] handles = [gradOutHandle, gradInHandle];
            int[] ints = [batchChannels, height, width, scaleFactor, inputSize];

            fixed (IntPtr* h = handles)
            fixed (int* i = ints)
            {
                argsPtr[0] = &h[0];
                argsPtr[1] = &h[1];
                argsPtr[2] = &i[0];
                argsPtr[3] = &i[1];
                argsPtr[4] = &i[2];
                argsPtr[5] = &i[3];
                argsPtr[6] = &i[4];

                var result = HipNativeBindings.hipModuleLaunchKernel(
                    kernel,
                    (uint)grid, 1, 1,
                    (uint)DefaultBlockSize, 1, 1,
                    0, _stream, (IntPtr)argsPtr, IntPtr.Zero);
                HipNativeBindings.CheckError(result, "hipModuleLaunchKernel (nearest_neighbor_upsample_backward)");
            }
        }
    }

    /// <summary>
    /// CPU fallback for nearest-neighbor upsampling backward when kernel is not available.
    /// </summary>
    private void NearestNeighborUpsampleBackwardFallback(IGpuBuffer gradOutput, IGpuBuffer gradInput, int batchChannels, int height, int width, int scaleFactor)
    {
        int outHeight = height * scaleFactor;
        int outWidth = width * scaleFactor;
        int outputSize = batchChannels * outHeight * outWidth;
        int inputSize = batchChannels * height * width;

        // Download gradOutput
        var gradOutData = DownloadBuffer(gradOutput);

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

        // Upload gradInput
        UploadToBuffer(gradInput, gradInData);
    }

    #endregion

    #region Activation Gradient Operations

    public unsafe void ReluBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradInput, int size)
    {
        if (!_kernelCache.TryGetValue("relu_backward", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: relu_backward");

        LaunchUnaryOp(krnl, gradOutput.Handle, input.Handle, gradInput.Handle, size);
    }

    public unsafe void SigmoidBackward(IGpuBuffer gradOutput, IGpuBuffer output, IGpuBuffer gradInput, int size)
    {
        if (!_kernelCache.TryGetValue("sigmoid_backward", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: sigmoid_backward");

        LaunchUnaryOp(krnl, gradOutput.Handle, output.Handle, gradInput.Handle, size);
    }

    public unsafe void TanhBackward(IGpuBuffer gradOutput, IGpuBuffer output, IGpuBuffer gradInput, int size)
    {
        if (!_kernelCache.TryGetValue("tanh_backward", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: tanh_backward");

        LaunchUnaryOp(krnl, gradOutput.Handle, output.Handle, gradInput.Handle, size);
    }

    public unsafe void GeluBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradInput, int size)
    {
        if (!_kernelCache.TryGetValue("gelu_backward", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: gelu_backward");

        LaunchUnaryOp(krnl, gradOutput.Handle, input.Handle, gradInput.Handle, size);
    }

    public unsafe void SoftmaxBackward(IGpuBuffer gradOutput, IGpuBuffer output, IGpuBuffer gradInput, int batchSize, int features)
    {
        if (!_kernelCache.TryGetValue("softmax_backward", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: softmax_backward");

            {
            IntPtr _p0 = gradOutput.Handle;
            IntPtr _p1 = output.Handle;
            IntPtr _p2 = gradInput.Handle;
            void** args = stackalloc void*[5];
            args[0] = &_p0;
            args[1] = &_p1;
            args[2] = &_p2;
            args[3] = &batchSize;
            args[4] = &features;


            uint grid = (uint)((batchSize + DefaultBlockSize - 1) / DefaultBlockSize);
            LaunchKernel(krnl, grid, DefaultBlockSize, args);
            Synchronize();
            }
    }

    #endregion

    #region Helper Methods

    private unsafe void LaunchElementwiseOp(IntPtr krnl, IntPtr input, IntPtr output, int size)
    {
            {
            void** args = stackalloc void*[3];
            args[0] = &input;
            args[1] = &output;
            args[2] = &size;


            uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
            LaunchKernel(krnl, grid, DefaultBlockSize, args);
            Synchronize();
            }
    }

    private unsafe void LaunchUnaryOp(IntPtr krnl, IntPtr a, IntPtr b, IntPtr c, int size)
    {
            {
            void** args = stackalloc void*[4];
            args[0] = &a;
            args[1] = &b;
            args[2] = &c;
            args[3] = &size;


            uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
            LaunchKernel(krnl, grid, DefaultBlockSize, args);
            Synchronize();
            }
    }

    #endregion

    #region Loss Function Operations

    public unsafe float CrossEntropyLoss(IGpuBuffer predictions, IGpuBuffer targets, int batchSize, int numClasses)
    {
        if (!_kernelCache.TryGetValue("cross_entropy_loss", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: cross_entropy_loss");

        using var lossBuffer = AllocateBuffer(new float[batchSize]);

            {
            IntPtr _p0 = predictions.Handle;
            IntPtr _p1 = targets.Handle;
            IntPtr _p2 = lossBuffer.Handle;
            void** args = stackalloc void*[5];
            args[0] = &_p0;
            args[1] = &_p1;
            args[2] = &_p2;
            args[3] = &batchSize;
            args[4] = &numClasses;


            uint grid = (uint)((batchSize + DefaultBlockSize - 1) / DefaultBlockSize);
            LaunchKernel(krnl, grid, DefaultBlockSize, args);
            Synchronize();
            }

        var lossData = new float[batchSize];
        DownloadBuffer(lossBuffer, lossData);
        float sum = 0;
        for (int i = 0; i < batchSize; i++) sum += lossData[i];
        return sum / batchSize;
    }

    public unsafe void CrossEntropyBackward(IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer gradInput, int batchSize, int numClasses)
    {
        if (!_kernelCache.TryGetValue("cross_entropy_backward", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: cross_entropy_backward");

        int total = batchSize * numClasses;
            {
            IntPtr _p0 = predictions.Handle;
            IntPtr _p1 = targets.Handle;
            IntPtr _p2 = gradInput.Handle;
            void** args = stackalloc void*[5];
            args[0] = &_p0;
            args[1] = &_p1;
            args[2] = &_p2;
            args[3] = &batchSize;
            args[4] = &numClasses;


            uint grid = (uint)((total + DefaultBlockSize - 1) / DefaultBlockSize);
            LaunchKernel(krnl, grid, DefaultBlockSize, args);
            Synchronize();
            }
    }

    public unsafe float BinaryCrossEntropyLoss(IGpuBuffer predictions, IGpuBuffer targets, int size)
    {
        if (!_kernelCache.TryGetValue("bce_loss", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: bce_loss");

        using var lossBuffer = AllocateBuffer(new float[size]);

            {
            IntPtr _p0 = predictions.Handle;
            IntPtr _p1 = targets.Handle;
            IntPtr _p2 = lossBuffer.Handle;
            void** args = stackalloc void*[4];
            args[0] = &_p0;
            args[1] = &_p1;
            args[2] = &_p2;
            args[3] = &size;


            uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
            LaunchKernel(krnl, grid, DefaultBlockSize, args);
            Synchronize();
            }

        var lossData = new float[size];
        DownloadBuffer(lossBuffer, lossData);
        float sum = 0;
        for (int i = 0; i < size; i++) sum += lossData[i];
        return sum / size;
    }

    public unsafe void BinaryCrossEntropyBackward(IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer gradInput, int size)
    {
        if (!_kernelCache.TryGetValue("bce_backward", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: bce_backward");

        LaunchUnaryOp(krnl, predictions.Handle, targets.Handle, gradInput.Handle, size);
    }

    public unsafe float MseLoss(IGpuBuffer predictions, IGpuBuffer targets, int size)
    {
        if (!_kernelCache.TryGetValue("mse_loss", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: mse_loss");

        using var lossBuffer = AllocateBuffer(new float[size]);

            {
            IntPtr _p0 = predictions.Handle;
            IntPtr _p1 = targets.Handle;
            IntPtr _p2 = lossBuffer.Handle;
            void** args = stackalloc void*[4];
            args[0] = &_p0;
            args[1] = &_p1;
            args[2] = &_p2;
            args[3] = &size;


            uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
            LaunchKernel(krnl, grid, DefaultBlockSize, args);
            Synchronize();
            }

        var lossData = new float[size];
        DownloadBuffer(lossBuffer, lossData);
        float sum = 0;
        for (int i = 0; i < size; i++) sum += lossData[i];
        return sum / size;
    }

    public unsafe void MseBackward(IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer gradInput, int size)
    {
        if (!_kernelCache.TryGetValue("mse_backward", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: mse_backward");

        LaunchUnaryOp(krnl, predictions.Handle, targets.Handle, gradInput.Handle, size);
    }

    public unsafe float SmoothL1Loss(IGpuBuffer predictions, IGpuBuffer targets, int size, float beta)
    {
        if (!_kernelCache.TryGetValue("smooth_l1_loss", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: smooth_l1_loss");

        using var lossBuffer = AllocateBuffer(new float[size]);

            {
            IntPtr _p0 = predictions.Handle;
            IntPtr _p1 = targets.Handle;
            IntPtr _p2 = lossBuffer.Handle;
            void** args = stackalloc void*[5];
            args[0] = &_p0;
            args[1] = &_p1;
            args[2] = &_p2;
            args[3] = &size;
            args[4] = &beta;


            uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
            LaunchKernel(krnl, grid, DefaultBlockSize, args);
            Synchronize();
            }

        var lossData = new float[size];
        DownloadBuffer(lossBuffer, lossData);
        float sum = 0;
        for (int i = 0; i < size; i++) sum += lossData[i];
        return sum / size;
    }

    public unsafe void SmoothL1Backward(IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer gradInput, int size, float beta)
    {
        if (!_kernelCache.TryGetValue("smooth_l1_backward", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: smooth_l1_backward");

            {
            IntPtr _p0 = predictions.Handle;
            IntPtr _p1 = targets.Handle;
            IntPtr _p2 = gradInput.Handle;
            void** args = stackalloc void*[5];
            args[0] = &_p0;
            args[1] = &_p1;
            args[2] = &_p2;
            args[3] = &size;
            args[4] = &beta;


            uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
            LaunchKernel(krnl, grid, DefaultBlockSize, args);
            Synchronize();
            }
    }

    public unsafe float TripletLoss(IGpuBuffer anchor, IGpuBuffer positive, IGpuBuffer negative, int batchSize, int embeddingDim, float margin)
    {
        if (anchor is null) throw new ArgumentNullException(nameof(anchor));
        if (positive is null) throw new ArgumentNullException(nameof(positive));
        if (negative is null) throw new ArgumentNullException(nameof(negative));
        if (batchSize <= 0) throw new ArgumentOutOfRangeException(nameof(batchSize), "Batch size must be positive.");
        if (embeddingDim <= 0) throw new ArgumentOutOfRangeException(nameof(embeddingDim), "Embedding dimension must be positive.");

        if (!_kernelCache.TryGetValue("triplet_loss", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: triplet_loss");

        using var lossBuffer = AllocateBuffer(batchSize);
            {
            IntPtr _p0 = anchor.Handle;
            IntPtr _p1 = positive.Handle;
            IntPtr _p2 = negative.Handle;
            IntPtr _p3 = lossBuffer.Handle;
            void** args = stackalloc void*[7];
            args[0] = &_p0;
            args[1] = &_p1;
            args[2] = &_p2;
            args[3] = &_p3;
            args[4] = &batchSize;
            args[5] = &embeddingDim;
            args[6] = &margin;


            uint grid = (uint)((batchSize + DefaultBlockSize - 1) / DefaultBlockSize);
            LaunchKernel(krnl, grid, DefaultBlockSize, args);
            Synchronize();
            }

        float[] lossData = DownloadBuffer(lossBuffer);
        float sum = 0;
        for (int i = 0; i < batchSize; i++) sum += lossData[i];
        return sum / batchSize;
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

        if (!_kernelCache.TryGetValue("triplet_loss_backward", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: triplet_loss_backward");

            {
            IntPtr _p0 = anchor.Handle;
            IntPtr _p1 = positive.Handle;
            IntPtr _p2 = negative.Handle;
            IntPtr _p3 = gradAnchor.Handle;
            IntPtr _p4 = gradPositive.Handle;
            IntPtr _p5 = gradNegative.Handle;
            void** args = stackalloc void*[9];
            args[0] = &_p0;
            args[1] = &_p1;
            args[2] = &_p2;
            args[3] = &_p3;
            args[4] = &_p4;
            args[5] = &_p5;
            args[6] = &batchSize;
            args[7] = &embeddingDim;
            args[8] = &margin;


            // Launch with batchSize blocks, embeddingDim threads per block
            uint blockSize = (uint)Math.Min(embeddingDim, 1024);
            LaunchKernel(krnl, (uint)batchSize, blockSize, args);
            Synchronize();
            }
    }

    // Huber Loss
    public unsafe float HuberLoss(IGpuBuffer predictions, IGpuBuffer targets, int size, float delta)
    {
        if (predictions is null) throw new ArgumentNullException(nameof(predictions));
        if (targets is null) throw new ArgumentNullException(nameof(targets));
        if (!_kernelCache.TryGetValue("huber_loss", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: huber_loss");

        using var outputBuffer = AllocateBuffer(size);
            {
            IntPtr _p0 = predictions.Handle;
            IntPtr _p1 = targets.Handle;
            IntPtr _p2 = outputBuffer.Handle;
            void** args = stackalloc void*[5];
            args[0] = &_p0;
            args[1] = &_p1;
            args[2] = &_p2;
            args[3] = &delta;
            args[4] = &size;


            uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
            LaunchKernel(krnl, grid, DefaultBlockSize, args);
            Synchronize();
            }

        return Sum(outputBuffer, size) / size;
    }

    public unsafe void HuberBackward(IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer gradInput, int size, float delta)
    {
        if (predictions is null) throw new ArgumentNullException(nameof(predictions));
        if (targets is null) throw new ArgumentNullException(nameof(targets));
        if (gradInput is null) throw new ArgumentNullException(nameof(gradInput));
        if (!_kernelCache.TryGetValue("huber_gradient", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: huber_gradient");

            {
            IntPtr _p0 = predictions.Handle;
            IntPtr _p1 = targets.Handle;
            IntPtr _p2 = gradInput.Handle;
            void** args = stackalloc void*[5];
            args[0] = &_p0;
            args[1] = &_p1;
            args[2] = &_p2;
            args[3] = &delta;
            args[4] = &size;


            uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
            LaunchKernel(krnl, grid, DefaultBlockSize, args);
            Synchronize();
            }
    }

    // Focal Loss
    public unsafe float FocalLoss(IGpuBuffer predictions, IGpuBuffer targets, int size, float alpha, float gamma)
    {
        if (predictions is null) throw new ArgumentNullException(nameof(predictions));
        if (targets is null) throw new ArgumentNullException(nameof(targets));
        if (!_kernelCache.TryGetValue("focal_loss", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: focal_loss");

        using var outputBuffer = AllocateBuffer(size);
            {
            IntPtr _p0 = predictions.Handle;
            IntPtr _p1 = targets.Handle;
            IntPtr _p2 = outputBuffer.Handle;
            void** args = stackalloc void*[6];
            args[0] = &_p0;
            args[1] = &_p1;
            args[2] = &_p2;
            args[3] = &alpha;
            args[4] = &gamma;
            args[5] = &size;


            uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
            LaunchKernel(krnl, grid, DefaultBlockSize, args);
            Synchronize();
            }

        return Sum(outputBuffer, size) / size;
    }

    public unsafe void FocalBackward(IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer gradInput, int size, float alpha, float gamma)
    {
        if (predictions is null) throw new ArgumentNullException(nameof(predictions));
        if (targets is null) throw new ArgumentNullException(nameof(targets));
        if (gradInput is null) throw new ArgumentNullException(nameof(gradInput));
        if (!_kernelCache.TryGetValue("focal_gradient", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: focal_gradient");

            {
            IntPtr _p0 = predictions.Handle;
            IntPtr _p1 = targets.Handle;
            IntPtr _p2 = gradInput.Handle;
            void** args = stackalloc void*[6];
            args[0] = &_p0;
            args[1] = &_p1;
            args[2] = &_p2;
            args[3] = &alpha;
            args[4] = &gamma;
            args[5] = &size;


            uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
            LaunchKernel(krnl, grid, DefaultBlockSize, args);
            Synchronize();
            }
    }

    // MAE Loss
    public unsafe float MaeLoss(IGpuBuffer predictions, IGpuBuffer targets, int size)
    {
        if (predictions is null) throw new ArgumentNullException(nameof(predictions));
        if (targets is null) throw new ArgumentNullException(nameof(targets));
        if (!_kernelCache.TryGetValue("mae_loss", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: mae_loss");

        using var outputBuffer = AllocateBuffer(size);
            {
            IntPtr _p0 = predictions.Handle;
            IntPtr _p1 = targets.Handle;
            IntPtr _p2 = outputBuffer.Handle;
            void** args = stackalloc void*[4];
            args[0] = &_p0;
            args[1] = &_p1;
            args[2] = &_p2;
            args[3] = &size;


            uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
            LaunchKernel(krnl, grid, DefaultBlockSize, args);
            Synchronize();
            }

        return Sum(outputBuffer, size) / size;
    }

    public unsafe void MaeBackward(IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer gradInput, int size)
    {
        if (predictions is null) throw new ArgumentNullException(nameof(predictions));
        if (targets is null) throw new ArgumentNullException(nameof(targets));
        if (gradInput is null) throw new ArgumentNullException(nameof(gradInput));
        if (!_kernelCache.TryGetValue("mae_gradient", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: mae_gradient");

            {
            IntPtr _p0 = predictions.Handle;
            IntPtr _p1 = targets.Handle;
            IntPtr _p2 = gradInput.Handle;
            void** args = stackalloc void*[4];
            args[0] = &_p0;
            args[1] = &_p1;
            args[2] = &_p2;
            args[3] = &size;


            uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
            LaunchKernel(krnl, grid, DefaultBlockSize, args);
            Synchronize();
            }
    }

    // Log-Cosh Loss
    public unsafe float LogCoshLoss(IGpuBuffer predictions, IGpuBuffer targets, int size)
    {
        if (predictions is null) throw new ArgumentNullException(nameof(predictions));
        if (targets is null) throw new ArgumentNullException(nameof(targets));
        if (!_kernelCache.TryGetValue("log_cosh_loss", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: log_cosh_loss");

        using var outputBuffer = AllocateBuffer(size);
            {
            IntPtr _p0 = predictions.Handle;
            IntPtr _p1 = targets.Handle;
            IntPtr _p2 = outputBuffer.Handle;
            void** args = stackalloc void*[4];
            args[0] = &_p0;
            args[1] = &_p1;
            args[2] = &_p2;
            args[3] = &size;


            uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
            LaunchKernel(krnl, grid, DefaultBlockSize, args);
            Synchronize();
            }

        return Sum(outputBuffer, size) / size;
    }

    public unsafe void LogCoshBackward(IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer gradInput, int size)
    {
        if (predictions is null) throw new ArgumentNullException(nameof(predictions));
        if (targets is null) throw new ArgumentNullException(nameof(targets));
        if (gradInput is null) throw new ArgumentNullException(nameof(gradInput));
        if (!_kernelCache.TryGetValue("log_cosh_gradient", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: log_cosh_gradient");

            {
            IntPtr _p0 = predictions.Handle;
            IntPtr _p1 = targets.Handle;
            IntPtr _p2 = gradInput.Handle;
            void** args = stackalloc void*[4];
            args[0] = &_p0;
            args[1] = &_p1;
            args[2] = &_p2;
            args[3] = &size;


            uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
            LaunchKernel(krnl, grid, DefaultBlockSize, args);
            Synchronize();
            }
    }

    // Quantile Loss
    public unsafe float QuantileLoss(IGpuBuffer predictions, IGpuBuffer targets, int size, float quantile)
    {
        if (predictions is null) throw new ArgumentNullException(nameof(predictions));
        if (targets is null) throw new ArgumentNullException(nameof(targets));
        if (!_kernelCache.TryGetValue("quantile_loss", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: quantile_loss");

        using var outputBuffer = AllocateBuffer(size);
            {
            IntPtr _p0 = predictions.Handle;
            IntPtr _p1 = targets.Handle;
            IntPtr _p2 = outputBuffer.Handle;
            void** args = stackalloc void*[5];
            args[0] = &_p0;
            args[1] = &_p1;
            args[2] = &_p2;
            args[3] = &quantile;
            args[4] = &size;


            uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
            LaunchKernel(krnl, grid, DefaultBlockSize, args);
            Synchronize();
            }

        return Sum(outputBuffer, size) / size;
    }

    public unsafe void QuantileBackward(IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer gradInput, int size, float quantile)
    {
        if (predictions is null) throw new ArgumentNullException(nameof(predictions));
        if (targets is null) throw new ArgumentNullException(nameof(targets));
        if (gradInput is null) throw new ArgumentNullException(nameof(gradInput));
        if (!_kernelCache.TryGetValue("quantile_gradient", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: quantile_gradient");

            {
            IntPtr _p0 = predictions.Handle;
            IntPtr _p1 = targets.Handle;
            IntPtr _p2 = gradInput.Handle;
            void** args = stackalloc void*[5];
            args[0] = &_p0;
            args[1] = &_p1;
            args[2] = &_p2;
            args[3] = &quantile;
            args[4] = &size;


            uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
            LaunchKernel(krnl, grid, DefaultBlockSize, args);
            Synchronize();
            }
    }

    // Hinge Loss
    public unsafe float HingeLoss(IGpuBuffer predictions, IGpuBuffer targets, int size)
    {
        if (predictions is null) throw new ArgumentNullException(nameof(predictions));
        if (targets is null) throw new ArgumentNullException(nameof(targets));
        if (!_kernelCache.TryGetValue("hinge_loss", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: hinge_loss");

        using var outputBuffer = AllocateBuffer(size);
            {
            IntPtr _p0 = predictions.Handle;
            IntPtr _p1 = targets.Handle;
            IntPtr _p2 = outputBuffer.Handle;
            void** args = stackalloc void*[4];
            args[0] = &_p0;
            args[1] = &_p1;
            args[2] = &_p2;
            args[3] = &size;


            uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
            LaunchKernel(krnl, grid, DefaultBlockSize, args);
            Synchronize();
            }

        return Sum(outputBuffer, size) / size;
    }

    public unsafe void HingeBackward(IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer gradInput, int size)
    {
        if (predictions is null) throw new ArgumentNullException(nameof(predictions));
        if (targets is null) throw new ArgumentNullException(nameof(targets));
        if (gradInput is null) throw new ArgumentNullException(nameof(gradInput));
        if (!_kernelCache.TryGetValue("hinge_gradient", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: hinge_gradient");

            {
            IntPtr _p0 = predictions.Handle;
            IntPtr _p1 = targets.Handle;
            IntPtr _p2 = gradInput.Handle;
            void** args = stackalloc void*[4];
            args[0] = &_p0;
            args[1] = &_p1;
            args[2] = &_p2;
            args[3] = &size;


            uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
            LaunchKernel(krnl, grid, DefaultBlockSize, args);
            Synchronize();
            }
    }

    // Squared Hinge Loss
    public unsafe float SquaredHingeLoss(IGpuBuffer predictions, IGpuBuffer targets, int size)
    {
        if (predictions is null) throw new ArgumentNullException(nameof(predictions));
        if (targets is null) throw new ArgumentNullException(nameof(targets));
        if (!_kernelCache.TryGetValue("squared_hinge_loss", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: squared_hinge_loss");

        using var outputBuffer = AllocateBuffer(size);
            {
            IntPtr _p0 = predictions.Handle;
            IntPtr _p1 = targets.Handle;
            IntPtr _p2 = outputBuffer.Handle;
            void** args = stackalloc void*[4];
            args[0] = &_p0;
            args[1] = &_p1;
            args[2] = &_p2;
            args[3] = &size;


            uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
            LaunchKernel(krnl, grid, DefaultBlockSize, args);
            Synchronize();
            }

        return Sum(outputBuffer, size) / size;
    }

    public unsafe void SquaredHingeBackward(IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer gradInput, int size)
    {
        if (predictions is null) throw new ArgumentNullException(nameof(predictions));
        if (targets is null) throw new ArgumentNullException(nameof(targets));
        if (gradInput is null) throw new ArgumentNullException(nameof(gradInput));
        if (!_kernelCache.TryGetValue("squared_hinge_gradient", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: squared_hinge_gradient");

            {
            IntPtr _p0 = predictions.Handle;
            IntPtr _p1 = targets.Handle;
            IntPtr _p2 = gradInput.Handle;
            void** args = stackalloc void*[4];
            args[0] = &_p0;
            args[1] = &_p1;
            args[2] = &_p2;
            args[3] = &size;


            uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
            LaunchKernel(krnl, grid, DefaultBlockSize, args);
            Synchronize();
            }
    }

    // Poisson Loss
    public unsafe float PoissonLoss(IGpuBuffer predictions, IGpuBuffer targets, int size)
    {
        if (predictions is null) throw new ArgumentNullException(nameof(predictions));
        if (targets is null) throw new ArgumentNullException(nameof(targets));
        if (!_kernelCache.TryGetValue("poisson_loss", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: poisson_loss");

        using var outputBuffer = AllocateBuffer(size);
            {
            IntPtr _p0 = predictions.Handle;
            IntPtr _p1 = targets.Handle;
            IntPtr _p2 = outputBuffer.Handle;
            void** args = stackalloc void*[4];
            args[0] = &_p0;
            args[1] = &_p1;
            args[2] = &_p2;
            args[3] = &size;


            uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
            LaunchKernel(krnl, grid, DefaultBlockSize, args);
            Synchronize();
            }

        return Sum(outputBuffer, size) / size;
    }

    public unsafe void PoissonBackward(IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer gradInput, int size)
    {
        if (predictions is null) throw new ArgumentNullException(nameof(predictions));
        if (targets is null) throw new ArgumentNullException(nameof(targets));
        if (gradInput is null) throw new ArgumentNullException(nameof(gradInput));
        if (!_kernelCache.TryGetValue("poisson_gradient", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: poisson_gradient");

            {
            IntPtr _p0 = predictions.Handle;
            IntPtr _p1 = targets.Handle;
            IntPtr _p2 = gradInput.Handle;
            void** args = stackalloc void*[4];
            args[0] = &_p0;
            args[1] = &_p1;
            args[2] = &_p2;
            args[3] = &size;


            uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
            LaunchKernel(krnl, grid, DefaultBlockSize, args);
            Synchronize();
            }
    }

    // Exponential Loss
    public unsafe float ExponentialLoss(IGpuBuffer predictions, IGpuBuffer targets, int size)
    {
        if (predictions is null) throw new ArgumentNullException(nameof(predictions));
        if (targets is null) throw new ArgumentNullException(nameof(targets));
        if (!_kernelCache.TryGetValue("exponential_loss", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: exponential_loss");

        using var outputBuffer = AllocateBuffer(size);
            {
            IntPtr _p0 = predictions.Handle;
            IntPtr _p1 = targets.Handle;
            IntPtr _p2 = outputBuffer.Handle;
            void** args = stackalloc void*[4];
            args[0] = &_p0;
            args[1] = &_p1;
            args[2] = &_p2;
            args[3] = &size;


            uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
            LaunchKernel(krnl, grid, DefaultBlockSize, args);
            Synchronize();
            }

        return Sum(outputBuffer, size) / size;
    }

    public unsafe void ExponentialBackward(IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer gradInput, int size)
    {
        if (predictions is null) throw new ArgumentNullException(nameof(predictions));
        if (targets is null) throw new ArgumentNullException(nameof(targets));
        if (gradInput is null) throw new ArgumentNullException(nameof(gradInput));
        if (!_kernelCache.TryGetValue("exponential_gradient", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: exponential_gradient");

            {
            IntPtr _p0 = predictions.Handle;
            IntPtr _p1 = targets.Handle;
            IntPtr _p2 = gradInput.Handle;
            void** args = stackalloc void*[4];
            args[0] = &_p0;
            args[1] = &_p1;
            args[2] = &_p2;
            args[3] = &size;


            uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
            LaunchKernel(krnl, grid, DefaultBlockSize, args);
            Synchronize();
            }
    }

    // Modified Huber Loss
    public unsafe float ModifiedHuberLoss(IGpuBuffer predictions, IGpuBuffer targets, int size)
    {
        if (predictions is null) throw new ArgumentNullException(nameof(predictions));
        if (targets is null) throw new ArgumentNullException(nameof(targets));
        if (!_kernelCache.TryGetValue("modified_huber_loss", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: modified_huber_loss");

        using var outputBuffer = AllocateBuffer(size);
            {
            IntPtr _p0 = predictions.Handle;
            IntPtr _p1 = targets.Handle;
            IntPtr _p2 = outputBuffer.Handle;
            void** args = stackalloc void*[4];
            args[0] = &_p0;
            args[1] = &_p1;
            args[2] = &_p2;
            args[3] = &size;


            uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
            LaunchKernel(krnl, grid, DefaultBlockSize, args);
            Synchronize();
            }

        return Sum(outputBuffer, size) / size;
    }

    public unsafe void ModifiedHuberBackward(IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer gradInput, int size)
    {
        if (predictions is null) throw new ArgumentNullException(nameof(predictions));
        if (targets is null) throw new ArgumentNullException(nameof(targets));
        if (gradInput is null) throw new ArgumentNullException(nameof(gradInput));
        if (!_kernelCache.TryGetValue("modified_huber_gradient", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: modified_huber_gradient");

            {
            IntPtr _p0 = predictions.Handle;
            IntPtr _p1 = targets.Handle;
            IntPtr _p2 = gradInput.Handle;
            void** args = stackalloc void*[4];
            args[0] = &_p0;
            args[1] = &_p1;
            args[2] = &_p2;
            args[3] = &size;


            uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
            LaunchKernel(krnl, grid, DefaultBlockSize, args);
            Synchronize();
            }
    }

    // Categorical Cross-Entropy Loss
    public unsafe float CategoricalCrossEntropyLoss(IGpuBuffer predictions, IGpuBuffer targets, int size)
    {
        if (predictions is null) throw new ArgumentNullException(nameof(predictions));
        if (targets is null) throw new ArgumentNullException(nameof(targets));
        if (!_kernelCache.TryGetValue("categorical_cross_entropy_loss", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: categorical_cross_entropy_loss");

        using var outputBuffer = AllocateBuffer(size);
            {
            IntPtr _p0 = predictions.Handle;
            IntPtr _p1 = targets.Handle;
            IntPtr _p2 = outputBuffer.Handle;
            void** args = stackalloc void*[4];
            args[0] = &_p0;
            args[1] = &_p1;
            args[2] = &_p2;
            args[3] = &size;


            uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
            LaunchKernel(krnl, grid, DefaultBlockSize, args);
            Synchronize();
            }

        return Sum(outputBuffer, size) / size;
    }

    public unsafe void CategoricalCrossEntropyBackward(IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer gradInput, int size)
    {
        if (predictions is null) throw new ArgumentNullException(nameof(predictions));
        if (targets is null) throw new ArgumentNullException(nameof(targets));
        if (gradInput is null) throw new ArgumentNullException(nameof(gradInput));
        if (!_kernelCache.TryGetValue("categorical_cross_entropy_gradient", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: categorical_cross_entropy_gradient");

            {
            IntPtr _p0 = predictions.Handle;
            IntPtr _p1 = targets.Handle;
            IntPtr _p2 = gradInput.Handle;
            void** args = stackalloc void*[4];
            args[0] = &_p0;
            args[1] = &_p1;
            args[2] = &_p2;
            args[3] = &size;


            uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
            LaunchKernel(krnl, grid, DefaultBlockSize, args);
            Synchronize();
            }
    }

    // Charbonnier Loss
    public unsafe float CharbonnierLoss(IGpuBuffer predictions, IGpuBuffer targets, int size, float epsilon)
    {
        if (predictions is null) throw new ArgumentNullException(nameof(predictions));
        if (targets is null) throw new ArgumentNullException(nameof(targets));
        if (!_kernelCache.TryGetValue("charbonnier_loss", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: charbonnier_loss");

        using var outputBuffer = AllocateBuffer(size);
            {
            IntPtr _p0 = predictions.Handle;
            IntPtr _p1 = targets.Handle;
            IntPtr _p2 = outputBuffer.Handle;
            void** args = stackalloc void*[5];
            args[0] = &_p0;
            args[1] = &_p1;
            args[2] = &_p2;
            args[3] = &epsilon;
            args[4] = &size;


            uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
            LaunchKernel(krnl, grid, DefaultBlockSize, args);
            Synchronize();
            }

        return Sum(outputBuffer, size) / size;
    }

    public unsafe void CharbonnierBackward(IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer gradInput, int size, float epsilon)
    {
        if (predictions is null) throw new ArgumentNullException(nameof(predictions));
        if (targets is null) throw new ArgumentNullException(nameof(targets));
        if (gradInput is null) throw new ArgumentNullException(nameof(gradInput));
        if (!_kernelCache.TryGetValue("charbonnier_gradient", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: charbonnier_gradient");

            {
            IntPtr _p0 = predictions.Handle;
            IntPtr _p1 = targets.Handle;
            IntPtr _p2 = gradInput.Handle;
            void** args = stackalloc void*[5];
            args[0] = &_p0;
            args[1] = &_p1;
            args[2] = &_p2;
            args[3] = &epsilon;
            args[4] = &size;


            uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
            LaunchKernel(krnl, grid, DefaultBlockSize, args);
            Synchronize();
            }
    }

    // Elastic Net Loss
    public unsafe float ElasticNetLoss(IGpuBuffer predictions, IGpuBuffer targets, int size, float l1Weight, float l2Weight)
    {
        if (predictions is null) throw new ArgumentNullException(nameof(predictions));
        if (targets is null) throw new ArgumentNullException(nameof(targets));
        if (!_kernelCache.TryGetValue("elastic_net_loss", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: elastic_net_loss");

        using var outputBuffer = AllocateBuffer(size);
            {
            IntPtr _p0 = predictions.Handle;
            IntPtr _p1 = targets.Handle;
            IntPtr _p2 = outputBuffer.Handle;
            void** args = stackalloc void*[6];
            args[0] = &_p0;
            args[1] = &_p1;
            args[2] = &_p2;
            args[3] = &l1Weight;
            args[4] = &l2Weight;
            args[5] = &size;


            uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
            LaunchKernel(krnl, grid, DefaultBlockSize, args);
            Synchronize();
            }

        return Sum(outputBuffer, size) / size;
    }

    public unsafe void ElasticNetBackward(IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer gradInput, int size, float l1Weight, float l2Weight)
    {
        if (predictions is null) throw new ArgumentNullException(nameof(predictions));
        if (targets is null) throw new ArgumentNullException(nameof(targets));
        if (gradInput is null) throw new ArgumentNullException(nameof(gradInput));
        if (!_kernelCache.TryGetValue("elastic_net_gradient", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: elastic_net_gradient");

            {
            IntPtr _p0 = predictions.Handle;
            IntPtr _p1 = targets.Handle;
            IntPtr _p2 = gradInput.Handle;
            void** args = stackalloc void*[6];
            args[0] = &_p0;
            args[1] = &_p1;
            args[2] = &_p2;
            args[3] = &l1Weight;
            args[4] = &l2Weight;
            args[5] = &size;


            uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
            LaunchKernel(krnl, grid, DefaultBlockSize, args);
            Synchronize();
            }
    }

    // Contrastive Loss
    public unsafe float ContrastiveLoss(IGpuBuffer anchor, IGpuBuffer other, IGpuBuffer labels, int batchSize, int embeddingDim, float margin)
    {
        if (anchor is null) throw new ArgumentNullException(nameof(anchor));
        if (other is null) throw new ArgumentNullException(nameof(other));
        if (labels is null) throw new ArgumentNullException(nameof(labels));
        if (!_kernelCache.TryGetValue("contrastive_loss", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: contrastive_loss");

        using var outputBuffer = AllocateBuffer(batchSize);
            {
            IntPtr _p0 = anchor.Handle;
            IntPtr _p1 = other.Handle;
            IntPtr _p2 = labels.Handle;
            IntPtr _p3 = outputBuffer.Handle;
            void** args = stackalloc void*[7];
            args[0] = &_p0;
            args[1] = &_p1;
            args[2] = &_p2;
            args[3] = &_p3;
            args[4] = &batchSize;
            args[5] = &embeddingDim;
            args[6] = &margin;


            uint grid = (uint)((batchSize + DefaultBlockSize - 1) / DefaultBlockSize);
            LaunchKernel(krnl, grid, DefaultBlockSize, args);
            Synchronize();
            }

        return Sum(outputBuffer, batchSize) / batchSize;
    }

    public unsafe void ContrastiveBackward(IGpuBuffer anchor, IGpuBuffer other, IGpuBuffer labels,
        IGpuBuffer gradAnchor, IGpuBuffer gradOther,
        int batchSize, int embeddingDim, float margin)
    {
        if (anchor is null) throw new ArgumentNullException(nameof(anchor));
        if (other is null) throw new ArgumentNullException(nameof(other));
        if (labels is null) throw new ArgumentNullException(nameof(labels));
        if (gradAnchor is null) throw new ArgumentNullException(nameof(gradAnchor));
        if (gradOther is null) throw new ArgumentNullException(nameof(gradOther));
        if (!_kernelCache.TryGetValue("contrastive_loss_backward", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: contrastive_loss_backward");

        int totalSize = batchSize * embeddingDim;
            {
            IntPtr _p0 = anchor.Handle;
            IntPtr _p1 = other.Handle;
            IntPtr _p2 = labels.Handle;
            IntPtr _p3 = gradAnchor.Handle;
            IntPtr _p4 = gradOther.Handle;
            void** args = stackalloc void*[8];
            args[0] = &_p0;
            args[1] = &_p1;
            args[2] = &_p2;
            args[3] = &_p3;
            args[4] = &_p4;
            args[5] = &batchSize;
            args[6] = &embeddingDim;
            args[7] = &margin;


            uint grid = (uint)((totalSize + DefaultBlockSize - 1) / DefaultBlockSize);
            LaunchKernel(krnl, grid, DefaultBlockSize, args);
            Synchronize();
            }
    }

    #endregion

    #region Utility Operations

    public unsafe void Clamp(IGpuBuffer A, IGpuBuffer B, float min, float max, int size)
    {
        if (!_kernelCache.TryGetValue("clamp", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: clamp");

            {
            IntPtr _p0 = A.Handle;
            IntPtr _p1 = B.Handle;
            void** args = stackalloc void*[5];
            args[0] = &_p0;
            args[1] = &_p1;
            args[2] = &min;
            args[3] = &max;
            args[4] = &size;


            uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
            LaunchKernel(krnl, grid, DefaultBlockSize, args);
            Synchronize();
            }
    }

    public unsafe float L2Norm(IGpuBuffer A, int size)
    {
        if (!_kernelCache.TryGetValue("l2_norm_squared", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: l2_norm_squared");

        using var squaredBuffer = AllocateBuffer(new float[size]);
        LaunchElementwiseOp(krnl, A.Handle, squaredBuffer.Handle, size);

        var data = new float[size];
        DownloadBuffer(squaredBuffer, data);
        float sum = 0;
        for (int i = 0; i < size; i++) sum += data[i];
        return (float)Math.Sqrt(sum);
    }

    public void ClipByValue(IGpuBuffer A, IGpuBuffer B, float clipValue, int size)
    {
        Clamp(A, B, -clipValue, clipValue, size);
    }

    public unsafe void ClipByNorm(IGpuBuffer A, IGpuBuffer B, float maxNorm, int size)
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

        if (!_kernelCache.TryGetValue("lerp_fused", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: lerp_fused");

            {
            IntPtr _p0 = a.Handle;
            IntPtr _p1 = b.Handle;
            IntPtr _p2 = output.Handle;
            void** args = stackalloc void*[5];
            args[0] = &_p0;
            args[1] = &_p1;
            args[2] = &_p2;
            args[3] = &t;
            args[4] = &size;


            uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
            LaunchKernel(krnl, grid, DefaultBlockSize, args);
            Synchronize();
            }
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

        if (!_kernelCache.TryGetValue("add_scaled", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: add_scaled");

            {
            IntPtr _p0 = a.Handle;
            IntPtr _p1 = b.Handle;
            IntPtr _p2 = output.Handle;
            void** args = stackalloc void*[6];
            args[0] = &_p0;
            args[1] = &_p1;
            args[2] = &_p2;
            args[3] = &scaleA;
            args[4] = &scaleB;
            args[5] = &size;


            uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
            LaunchKernel(krnl, grid, DefaultBlockSize, args);
            Synchronize();
            }
    }

    public unsafe float StdDev(IGpuBuffer input, int size)
    {
        if (size <= 0)
            throw new ArgumentOutOfRangeException(nameof(size), size, "Size must be positive.");
        if (input.Size < size)
            throw new ArgumentException($"Buffer 'input' capacity ({input.Size}) is less than size ({size}).", nameof(input));
        if (size <= 1) return 0.0f;

        const int blockSize = 256;
        int gridSize = (size + blockSize - 1) / blockSize;

        // Step 1: Compute mean via GPU reduction
        float mean;
        if (_kernelCache.TryGetValue("reduce_mean_kernel", out var meanKernel))
        {
            var zeroData = new float[1];
            using var meanBuffer = AllocateBuffer(zeroData);

                {
                IntPtr _p0 = input.Handle;
                IntPtr _p1 = meanBuffer.Handle;
                void** meanArgs = stackalloc void*[3];
                meanArgs[0] = &_p0;
                meanArgs[1] = &_p1;
                meanArgs[2] = &size;


                uint sharedBytes = (uint)(blockSize * sizeof(float));
                LaunchKernelWithSharedMem(meanKernel, (uint)gridSize, (uint)blockSize, sharedBytes, meanArgs);
                Synchronize();
                }

            float[] meanResult = DownloadBuffer(meanBuffer);
            mean = meanResult[0] / size;
        }
        else
        {
            mean = Sum(input, size) / size;
        }

        // Step 2: Compute variance via GPU reduction
        float variance;
        if (_kernelCache.TryGetValue("reduce_variance_kernel", out var varKernel))
        {
            var zeroData = new float[1];
            using var varianceBuffer = AllocateBuffer(zeroData);

                {
                IntPtr _p0 = input.Handle;
                IntPtr _p1 = varianceBuffer.Handle;
                void** varArgs = stackalloc void*[4];
                varArgs[0] = &_p0;
                varArgs[1] = &_p1;
                varArgs[2] = &mean;
                varArgs[3] = &size;


                uint sharedBytes = (uint)(blockSize * sizeof(float));
                LaunchKernelWithSharedMem(varKernel, (uint)gridSize, (uint)blockSize, sharedBytes, varArgs);
                Synchronize();
                }

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
        // CPU fallback for scatter - no dedicated kernel
        var srcData = new float[sourceSize];
        var idxData = new float[sourceSize];
        var dstData = new float[destSize];

        DownloadBuffer(source, srcData);
        DownloadBuffer(indices, idxData);
        DownloadBuffer(destination, dstData);

        for (int i = 0; i < sourceSize; i++)
        {
            int idx = (int)idxData[i];
            if (idx >= 0 && idx < destSize)
                dstData[idx] += srcData[i];
        }

        UploadToBuffer(destination, dstData);
    }

    public unsafe void ScatterAddBackward(IGpuBuffer gradDestination, IGpuBuffer indices, IGpuBuffer gradSource,
        int numIndices, int featureSize)
    {
        // ScatterAddBackward is essentially a Gather operation
        // Uses the embedding forward kernel since it's equivalent
        Embedding(indices, gradDestination, gradSource, numIndices, featureSize);
    }

    public unsafe void Gather(IGpuBuffer source, IGpuBuffer indices, IGpuBuffer output, int numIndices, int featureSize)
    {
        // Uses the embedding forward kernel since it's equivalent
        Embedding(indices, source, output, numIndices, featureSize);
    }

    #endregion

    #region Comparison Operations

    public unsafe void GreaterThan(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int size)
    {
        if (!_kernelCache.TryGetValue("greater_than", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: greater_than");

        LaunchBinaryOp(krnl, A, B, C, size);
    }

    public unsafe void LessThan(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int size)
    {
        if (!_kernelCache.TryGetValue("less_than", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: less_than");

        LaunchBinaryOp(krnl, A, B, C, size);
    }

    public unsafe void Equal(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int size)
    {
        if (!_kernelCache.TryGetValue("equals", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: equals");

        LaunchBinaryOp(krnl, A, B, C, size);
    }

    public unsafe void Where(IGpuBuffer condition, IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int size)
    {
        if (!_kernelCache.TryGetValue("where_cond", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: where_cond");

            {
            IntPtr _p0 = condition.Handle;
            IntPtr _p1 = A.Handle;
            IntPtr _p2 = B.Handle;
            IntPtr _p3 = C.Handle;
            void** args = stackalloc void*[5];
            args[0] = &_p0;
            args[1] = &_p1;
            args[2] = &_p2;
            args[3] = &_p3;
            args[4] = &size;


            uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
            LaunchKernel(krnl, grid, DefaultBlockSize, args);
            Synchronize();
            }
    }

    public unsafe void NotEqualScalar(IGpuBuffer A, IGpuBuffer C, float scalar, int size)
    {
        if (!_kernelCache.TryGetValue("not_equal_scalar", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: not_equal_scalar");

            {
            IntPtr _p0 = A.Handle;
            IntPtr _p1 = C.Handle;
            void** args = stackalloc void*[4];
            args[0] = &_p0;
            args[1] = &_p1;
            args[2] = &scalar;
            args[3] = &size;


            uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
            LaunchKernel(krnl, grid, DefaultBlockSize, args);
            Synchronize();
            }
    }

    private unsafe void LaunchBinaryOp(IntPtr krnl, IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int size)
    {
            {
            IntPtr _p0 = A.Handle;
            IntPtr _p1 = B.Handle;
            IntPtr _p2 = C.Handle;
            void** args = stackalloc void*[4];
            args[0] = &_p0;
            args[1] = &_p1;
            args[2] = &_p2;
            args[3] = &size;


            uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
            LaunchKernel(krnl, grid, DefaultBlockSize, args);
            Synchronize();
            }
    }

    /// <summary>
    /// Launches a binary op with automatic vec4 selection when size is divisible by 4.
    /// </summary>
    private unsafe void LaunchBinaryOpAutoVec4(string kernelName, IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int size)
    {
        if (size % 4 == 0 && _kernelCache.TryGetValue(kernelName + "_vec4", out var vec4Krnl))
        {
            int size4 = size / 4;
                {
                IntPtr _p0 = A.Handle;
                IntPtr _p1 = B.Handle;
                IntPtr _p2 = C.Handle;
                void** args = stackalloc void*[4];
                args[0] = &_p0;
                args[1] = &_p1;
                args[2] = &_p2;
                args[3] = &size4;
                uint grid = (uint)((size4 + DefaultBlockSize - 1) / DefaultBlockSize);
                LaunchKernel(vec4Krnl, grid, DefaultBlockSize, args);
                Synchronize();
                }
            return;
        }

        if (!_kernelCache.TryGetValue(kernelName, out var krnl))
            throw new InvalidOperationException($"HIP kernel not found: {kernelName}");
        LaunchBinaryOp(krnl, A, B, C, size);
    }

    #endregion

    #region Statistics Operations

    public unsafe void MeanAxis(IGpuBuffer A, IGpuBuffer B, int outerSize, int reduceSize)
    {
        if (!_kernelCache.TryGetValue("sum_axis", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: sum_axis");

        // First sum, then divide
            {
            IntPtr _p0 = A.Handle;
            IntPtr _p1 = B.Handle;
            void** args = stackalloc void*[4];
            args[0] = &_p0;
            args[1] = &_p1;
            args[2] = &outerSize;
            args[3] = &reduceSize;


            uint grid = (uint)((outerSize + DefaultBlockSize - 1) / DefaultBlockSize);
            LaunchKernel(krnl, grid, DefaultBlockSize, args);
            Synchronize();
            }

        // Divide by reduceSize to get mean
        Scale(B, B, 1.0f / reduceSize, outerSize);
    }

    public unsafe void VarAxis(IGpuBuffer A, IGpuBuffer mean, IGpuBuffer variance, int outerSize, int reduceSize)
    {
        if (!_kernelCache.TryGetValue("compute_mean_var", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: compute_mean_var");

            {
            IntPtr _p0 = A.Handle;
            IntPtr _p1 = mean.Handle;
            IntPtr _p2 = variance.Handle;
            void** args = stackalloc void*[5];
            args[0] = &_p0;
            args[1] = &_p1;
            args[2] = &_p2;
            args[3] = &outerSize;
            args[4] = &reduceSize;


            uint grid = (uint)((reduceSize + DefaultBlockSize - 1) / DefaultBlockSize);
            LaunchKernel(krnl, grid, DefaultBlockSize, args);
            Synchronize();
            }
    }

    public unsafe void ArgMax(IGpuBuffer A, IGpuBuffer indices, int outerSize, int reduceSize)
    {
        if (!_kernelCache.TryGetValue("argmax_axis", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: argmax_axis");

            {
            IntPtr _p0 = A.Handle;
            IntPtr _p1 = indices.Handle;
            void** args = stackalloc void*[4];
            args[0] = &_p0;
            args[1] = &_p1;
            args[2] = &outerSize;
            args[3] = &reduceSize;


            uint grid = (uint)((outerSize + DefaultBlockSize - 1) / DefaultBlockSize);
            LaunchKernel(krnl, grid, DefaultBlockSize, args);
            Synchronize();
            }
    }

    public unsafe void ArgMin(IGpuBuffer A, IGpuBuffer indices, int outerSize, int reduceSize)
    {
        if (!_kernelCache.TryGetValue("argmin_axis", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: argmin_axis");

            {
            IntPtr _p0 = A.Handle;
            IntPtr _p1 = indices.Handle;
            void** args = stackalloc void*[4];
            args[0] = &_p0;
            args[1] = &_p1;
            args[2] = &outerSize;
            args[3] = &reduceSize;


            uint grid = (uint)((outerSize + DefaultBlockSize - 1) / DefaultBlockSize);
            LaunchKernel(krnl, grid, DefaultBlockSize, args);
            Synchronize();
            }
    }

    public unsafe void MaxAxis(IGpuBuffer A, IGpuBuffer B, int outerSize, int reduceSize)
    {
        if (!_kernelCache.TryGetValue("max_axis", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: max_axis");

            {
            IntPtr _p0 = A.Handle;
            IntPtr _p1 = B.Handle;
            void** args = stackalloc void*[4];
            args[0] = &_p0;
            args[1] = &_p1;
            args[2] = &outerSize;
            args[3] = &reduceSize;


            uint grid = (uint)((outerSize + DefaultBlockSize - 1) / DefaultBlockSize);
            LaunchKernel(krnl, grid, DefaultBlockSize, args);
            Synchronize();
            }
    }

    public unsafe void TopK(IGpuBuffer A, IGpuBuffer values, IGpuBuffer indices, int outerSize, int reduceSize, int k, bool sorted = true)
    {
        if (!_kernelCache.TryGetValue("topk", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: topk");

            {
            IntPtr _p0 = A.Handle;
            IntPtr _p1 = values.Handle;
            IntPtr _p2 = indices.Handle;
            var _p3 = sorted ? 1 : 0;
            void** args = stackalloc void*[7];
            args[0] = &_p0;
            args[1] = &_p1;
            args[2] = &_p2;
            args[3] = &outerSize;
            args[4] = &reduceSize;
            args[5] = &k;
            args[6] = &_p3;


            // One block per row, 256 threads per block
            uint grid = (uint)outerSize;
            uint sharedMem = (uint)(k * (sizeof(float) + sizeof(int)));
            LaunchKernel(krnl, grid, DefaultBlockSize, args, sharedMem);
            Synchronize();
            }
    }

    public unsafe void AffineGrid(IGpuBuffer theta, IGpuBuffer grid, int batch, int outputHeight, int outputWidth)
    {
        if (!_kernelCache.TryGetValue("affine_grid", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: affine_grid");

            {
            IntPtr _p0 = theta.Handle;
            IntPtr _p1 = grid.Handle;
            void** args = stackalloc void*[5];
            args[0] = &_p0;
            args[1] = &_p1;
            args[2] = &batch;
            args[3] = &outputHeight;
            args[4] = &outputWidth;


            // 3D grid: (width, height, batch)
            uint gridX = (uint)((outputWidth + 15) / 16);
            uint gridY = (uint)((outputHeight + 15) / 16);
            uint gridZ = (uint)batch;
            LaunchKernel3D(krnl, gridX, gridY, gridZ, 16, 16, 1, args);
            Synchronize();
            }
    }

    public unsafe void GridSample(IGpuBuffer input, IGpuBuffer grid, IGpuBuffer output,
        int batch, int channels, int inHeight, int inWidth, int outHeight, int outWidth,
        int paddingMode = 0, bool alignCorners = false)
    {
        if (!_kernelCache.TryGetValue("grid_sample", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: grid_sample");

            {
            IntPtr _p0 = input.Handle;
            IntPtr _p1 = grid.Handle;
            IntPtr _p2 = output.Handle;
            void** args = stackalloc void*[11];
            args[0] = &_p0;
            args[1] = &_p1;
            args[2] = &_p2;
            args[3] = &batch;
            args[4] = &channels;
            args[5] = &inHeight;
            args[6] = &inWidth;
            args[7] = &outHeight;
            args[8] = &outWidth;
            args[9] = &paddingMode;
            int alignCornersInt = alignCorners ? 1 : 0;
            args[10] = &alignCornersInt;

                uint gridX = (uint)((outWidth + 15) / 16);
                uint gridY = (uint)((outHeight + 15) / 16);
                uint gridZ = (uint)(batch * channels);
                LaunchKernel3D(krnl, gridX, gridY, gridZ, 16, 16, 1, args);
                Synchronize();
            }
    }

    public unsafe void GridSampleBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer grid,
        IGpuBuffer gradInput, IGpuBuffer gradGrid,
        int batch, int channels, int inHeight, int inWidth, int outHeight, int outWidth,
        int paddingMode = 0, bool alignCorners = false)
    {
        if (!_kernelCache.TryGetValue("grid_sample_backward", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: grid_sample_backward");

            {
            IntPtr _p0 = gradOutput.Handle;
            IntPtr _p1 = input.Handle;
            IntPtr _p2 = grid.Handle;
            IntPtr _p3 = gradInput.Handle;
            IntPtr _p4 = gradGrid.Handle;
            void** args = stackalloc void*[13];
            args[0] = &_p0;
            args[1] = &_p1;
            args[2] = &_p2;
            args[3] = &_p3;
            args[4] = &_p4;
            args[5] = &batch;
            args[6] = &channels;
            args[7] = &inHeight;
            args[8] = &inWidth;
            args[9] = &outHeight;
            args[10] = &outWidth;
            args[11] = &paddingMode;
            int alignCornersInt = alignCorners ? 1 : 0;
            args[12] = &alignCornersInt;

                uint gridX = (uint)((outWidth + 15) / 16);
                uint gridY = (uint)((outHeight + 15) / 16);
                uint gridZ = (uint)batch;
                LaunchKernel3D(krnl, gridX, gridY, gridZ, 16, 16, 1, args);
                Synchronize();
            }
    }

    public unsafe void BroadcastMultiplyLastAxis(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int outerSize, int innerSize)
    {
        if (!_kernelCache.TryGetValue("broadcast_multiply_last_axis", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: broadcast_multiply_last_axis");

            {
            IntPtr _p0 = A.Handle;
            IntPtr _p1 = B.Handle;
            IntPtr _p2 = C.Handle;
            void** args = stackalloc void*[5];
            args[0] = &_p0;
            args[1] = &_p1;
            args[2] = &_p2;
            args[3] = &outerSize;
            args[4] = &innerSize;


            int totalSize = outerSize * innerSize;
            uint grid = (uint)((totalSize + DefaultBlockSize - 1) / DefaultBlockSize);
            LaunchKernel(krnl, grid, DefaultBlockSize, args);
            Synchronize();
            }
    }

    public unsafe void BroadcastMultiplyFirstAxis(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int outerSize, int innerSize)
    {
        if (!_kernelCache.TryGetValue("broadcast_multiply_first_axis", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: broadcast_multiply_first_axis");

            {
            IntPtr _p0 = A.Handle;
            IntPtr _p1 = B.Handle;
            IntPtr _p2 = C.Handle;
            void** args = stackalloc void*[5];
            args[0] = &_p0;
            args[1] = &_p1;
            args[2] = &_p2;
            args[3] = &outerSize;
            args[4] = &innerSize;


            int totalSize = outerSize * innerSize;
            uint grid = (uint)((totalSize + DefaultBlockSize - 1) / DefaultBlockSize);
            LaunchKernel(krnl, grid, DefaultBlockSize, args);
            Synchronize();
            }
    }

    #endregion

    #region FFT and Signal Processing

    /// <summary>
    /// Performs in-place FFT or IFFT using the Cooley-Tukey radix-2 algorithm.
    /// </summary>
    public unsafe void FFT(IGpuBuffer inputReal, IGpuBuffer inputImag, IGpuBuffer outputReal, IGpuBuffer outputImag, int n, bool inverse)
    {
        if (!IsAvailable)
            throw new InvalidOperationException("HIP backend not available");

        // Copy input to output buffers for in-place operation
        HipCopyBuffer(inputReal, outputReal, n);
        HipCopyBuffer(inputImag, outputImag, n);

        int log2n = (int)Math.Log(n, 2);

        // Bit-reversal permutation
        if (_kernelCache.TryGetValue("bit_reverse_permutation", out var bitRevKernel))
        {
            uint gridSize = (uint)((n + DefaultBlockSize - 1) / DefaultBlockSize);
                {
                IntPtr _p0 = outputReal.Handle;
                IntPtr _p1 = outputImag.Handle;
                void** args = stackalloc void*[4];
                args[0] = &_p0;
                args[1] = &_p1;
                args[2] = &n;
                args[3] = &log2n;


                LaunchKernel(bitRevKernel, gridSize, (uint)DefaultBlockSize, args);
                }
        }

        // Butterfly stages
        if (_kernelCache.TryGetValue("fft_butterfly", out var butterflyKernel))
        {
            int inverseFlag = inverse ? 1 : 0;
            for (int stride = 2; stride <= n; stride *= 2)
            {
                int numButterflies = n / 2;
                uint gridSize = (uint)((numButterflies + DefaultBlockSize - 1) / DefaultBlockSize);

                    {
                    IntPtr _p0 = outputReal.Handle;
                    IntPtr _p1 = outputImag.Handle;
                    #pragma warning disable CA2014
                    void** args = stackalloc void*[5];
                    #pragma warning restore CA2014
                    args[0] = &_p0;
                    args[1] = &_p1;
                    args[2] = &n;
                    args[3] = &stride;
                    args[4] = &inverseFlag;


                    LaunchKernel(butterflyKernel, gridSize, (uint)DefaultBlockSize, args);
                    }
            }
        }

        // Scale by 1/N for inverse FFT
        if (inverse && _kernelCache.TryGetValue("scale_inverse", out var scaleKernel))
        {
            uint gridSize = (uint)((n + DefaultBlockSize - 1) / DefaultBlockSize);
                {
                IntPtr _p0 = outputReal.Handle;
                IntPtr _p1 = outputImag.Handle;
                void** args = stackalloc void*[3];
                args[0] = &_p0;
                args[1] = &_p1;
                args[2] = &n;


                LaunchKernel(scaleKernel, gridSize, (uint)DefaultBlockSize, args);
                }
        }

        Synchronize();
    }

    /// <summary>
    /// Real-to-complex FFT exploiting conjugate symmetry.
    /// </summary>
    public unsafe void RFFT(IGpuBuffer input, IGpuBuffer outputReal, IGpuBuffer outputImag, int n)
    {
        if (!IsAvailable)
            throw new InvalidOperationException("HIP backend not available");

        // Allocate temporary buffers for full complex FFT
        using var tempReal = AllocateBuffer(n);
        using var tempImag = AllocateBuffer(n);

        // Copy real input to tempReal, zero tempImag
        HipCopyBuffer(input, tempReal, n);
        HipZeroBuffer(tempImag, n);

        // Perform full FFT
        FFT(tempReal, tempImag, tempReal, tempImag, n, inverse: false);

        // Post-process to extract positive frequencies (n/2 + 1 elements)
        if (_kernelCache.TryGetValue("rfft_postprocess", out var postprocessKernel))
        {
            int outLen = n / 2 + 1;
            uint gridSize = (uint)((outLen + DefaultBlockSize - 1) / DefaultBlockSize);

                {
                IntPtr _p0 = tempReal.Handle;
                IntPtr _p1 = tempImag.Handle;
                IntPtr _p2 = outputReal.Handle;
                IntPtr _p3 = outputImag.Handle;
                void** args = stackalloc void*[5];
                args[0] = &_p0;
                args[1] = &_p1;
                args[2] = &_p2;
                args[3] = &_p3;
                args[4] = &n;


                LaunchKernel(postprocessKernel, gridSize, (uint)DefaultBlockSize, args);
                }
        }

        Synchronize();
    }

    /// <summary>
    /// Complex-to-real IFFT exploiting conjugate symmetry.
    /// </summary>
    public unsafe void IRFFT(IGpuBuffer inputReal, IGpuBuffer inputImag, IGpuBuffer output, int n)
    {
        if (!IsAvailable)
            throw new InvalidOperationException("HIP backend not available");

        // Allocate temporary buffers for full complex FFT
        using var tempReal = AllocateBuffer(n);
        using var tempImag = AllocateBuffer(n);

        // Pre-process to reconstruct negative frequencies
        if (_kernelCache.TryGetValue("irfft_preprocess", out var preprocessKernel))
        {
            uint gridSize = (uint)((n + DefaultBlockSize - 1) / DefaultBlockSize);

                {
                IntPtr _p0 = inputReal.Handle;
                IntPtr _p1 = inputImag.Handle;
                IntPtr _p2 = tempReal.Handle;
                IntPtr _p3 = tempImag.Handle;
                void** args = stackalloc void*[5];
                args[0] = &_p0;
                args[1] = &_p1;
                args[2] = &_p2;
                args[3] = &_p3;
                args[4] = &n;


                LaunchKernel(preprocessKernel, gridSize, (uint)DefaultBlockSize, args);
                }
        }

        // Perform full inverse FFT
        using var outputImag = AllocateBuffer(n);
        FFT(tempReal, tempImag, output, outputImag, n, inverse: true);
    }

    /// <summary>
    /// Batched FFT for processing multiple signals in parallel.
    /// </summary>
    public unsafe void BatchedFFT(IGpuBuffer inputReal, IGpuBuffer inputImag, IGpuBuffer outputReal, IGpuBuffer outputImag, int batch, int n, bool inverse)
    {
        if (!IsAvailable)
            throw new InvalidOperationException("HIP backend not available");

        // Copy input to output buffers for in-place operation
        HipCopyBuffer(inputReal, outputReal, batch * n);
        HipCopyBuffer(inputImag, outputImag, batch * n);

        int log2n = (int)Math.Log(n, 2);

        // Batched bit-reversal permutation
        if (_kernelCache.TryGetValue("batched_bit_reverse", out var bitRevKernel))
        {
            uint gridX = (uint)((n + DefaultBlockSize - 1) / DefaultBlockSize);
            uint gridY = (uint)batch;

                {
                IntPtr _p0 = outputReal.Handle;
                IntPtr _p1 = outputImag.Handle;
                void** args = stackalloc void*[5];
                args[0] = &_p0;
                args[1] = &_p1;
                args[2] = &batch;
                args[3] = &n;
                args[4] = &log2n;


                LaunchKernel2D(bitRevKernel, gridX, gridY, (uint)DefaultBlockSize, 1, args);
                }
        }

        // Batched butterfly stages
        if (_kernelCache.TryGetValue("batched_fft_butterfly", out var butterflyKernel))
        {
            int inverseFlag = inverse ? 1 : 0;
            for (int stride = 2; stride <= n; stride *= 2)
            {
                int numButterflies = n / 2;
                uint gridX = (uint)((numButterflies + DefaultBlockSize - 1) / DefaultBlockSize);
                uint gridZ = (uint)batch;

                    {
                    IntPtr _p0 = outputReal.Handle;
                    IntPtr _p1 = outputImag.Handle;
#pragma warning disable CA2014
                    void** args = stackalloc void*[6];
#pragma warning restore CA2014
                    args[0] = &_p0;
                    args[1] = &_p1;
                    args[2] = &batch;
                    args[3] = &n;
                    args[4] = &stride;
                    args[5] = &inverseFlag;

                    LaunchKernel3D(butterflyKernel, gridX, 1, gridZ, (uint)DefaultBlockSize, 1, 1, args);
                    }
            }
        }

        // Scale by 1/N for inverse FFT (batched)
        if (inverse && _kernelCache.TryGetValue("scale_inverse", out var scaleKernel))
        {
            int total = batch * n;
            uint gridSize = (uint)((total + DefaultBlockSize - 1) / DefaultBlockSize);
                {
                IntPtr _p0 = outputReal.Handle;
                IntPtr _p1 = outputImag.Handle;
                void** args = stackalloc void*[3];
                args[0] = &_p0;
                args[1] = &_p1;
                args[2] = &total;


                LaunchKernel(scaleKernel, gridSize, (uint)DefaultBlockSize, args);
                }
        }

        Synchronize();
    }

    /// <summary>
    /// 2D FFT using row-column decomposition.
    /// </summary>
    public unsafe void FFT2D(IGpuBuffer inputReal, IGpuBuffer inputImag, IGpuBuffer outputReal, IGpuBuffer outputImag, int height, int width, bool inverse)
    {
        if (!IsAvailable)
            throw new InvalidOperationException("HIP backend not available");

        int total = height * width;
        int log2Width = (int)Math.Log(width, 2);
        int log2Height = (int)Math.Log(height, 2);

        // Copy input to output buffers for in-place operation
        HipCopyBuffer(inputReal, outputReal, total);
        HipCopyBuffer(inputImag, outputImag, total);

        // Row-wise bit reversal and FFT
        if (_kernelCache.TryGetValue("bit_reverse_permutation", out var bitRevKernel) &&
            _kernelCache.TryGetValue("fft_rows_butterfly", out var rowsButterflyKernel))
        {
            // Bit reversal for each row (using batched approach)
            for (int row = 0; row < height; row++)
            {
                int offset = row * width;
                // Note: For production, we should use offset buffers or batched kernel
                // This is a simplified version that operates row by row
            }

            int inverseFlag = inverse ? 1 : 0;
            for (int stride = 2; stride <= width; stride *= 2)
            {
                int numButterflies = width / 2;
                uint gridX = (uint)((numButterflies + DefaultBlockSize - 1) / DefaultBlockSize);
                uint gridY = (uint)height;
                    {
                    IntPtr _p0 = outputReal.Handle;
                    IntPtr _p1 = outputImag.Handle;
#pragma warning disable CA2014
                    void** args = stackalloc void*[6];
#pragma warning restore CA2014
                    args[0] = &_p0;
                    args[1] = &_p1;
                    args[2] = &height;
                    args[3] = &width;
                    args[4] = &stride;
                    args[5] = &inverseFlag;

                    LaunchKernel2D(rowsButterflyKernel, gridX, gridY, (uint)DefaultBlockSize, 1, args);
                    }
            }
        }

        // Column-wise FFT
        if (_kernelCache.TryGetValue("fft_cols_butterfly", out var colsButterflyKernel))
        {
            int inverseFlag = inverse ? 1 : 0;
            for (int stride = 2; stride <= height; stride *= 2)
            {
                int numButterflies = height / 2;
                uint gridX = (uint)((numButterflies + DefaultBlockSize - 1) / DefaultBlockSize);
                uint gridY = (uint)width;

                    {
                    IntPtr _p0 = outputReal.Handle;
                    IntPtr _p1 = outputImag.Handle;
#pragma warning disable CA2014
                    void** args = stackalloc void*[6];
#pragma warning restore CA2014
                    args[0] = &_p0;
                    args[1] = &_p1;
                    args[2] = &height;
                    args[3] = &width;
                    args[4] = &stride;
                    args[5] = &inverseFlag;


                    LaunchKernel2D(colsButterflyKernel, gridX, gridY, (uint)DefaultBlockSize, 1, args);
                    }
            }
        }

        // Scale by 1/(height*width) for inverse FFT
        if (inverse && _kernelCache.TryGetValue("scale_inverse", out var scaleKernel))
        {
            uint gridSize = (uint)((total + DefaultBlockSize - 1) / DefaultBlockSize);
                {
                IntPtr _p0 = outputReal.Handle;
                IntPtr _p1 = outputImag.Handle;
                void** args = stackalloc void*[3];
                args[0] = &_p0;
                args[1] = &_p1;
                args[2] = &total;


                LaunchKernel(scaleKernel, gridSize, (uint)DefaultBlockSize, args);
                }
        }

        Synchronize();
    }

    /// <summary>
    /// Applies a window function element-wise.
    /// </summary>
    /// <inheritdoc/>
    /// <inheritdoc/>
    public void BatchedFFT2D(IGpuBuffer inputReal, IGpuBuffer inputImag,
        IGpuBuffer outputReal, IGpuBuffer outputImag,
        int batch, int height, int width, bool inverse)
    {
        if (batch <= 0 || height <= 0 || width <= 0) return;
        int sliceSize = height * width;

        if (batch == 1)
        {
            FFT2D(inputReal, inputImag, outputReal, outputImag, height, width, inverse);
            return;
        }

        // Temp slice buffers reused across all slices — fully GPU-resident
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

    public unsafe void ApplyWindow(IGpuBuffer input, IGpuBuffer window, IGpuBuffer output, int n)
    {
        if (!IsAvailable)
            throw new InvalidOperationException("HIP backend not available");

        if (!_kernelCache.TryGetValue("apply_window", out var kernel))
            throw new InvalidOperationException("apply_window kernel not found");

        uint gridSize = (uint)((n + DefaultBlockSize - 1) / DefaultBlockSize);

            {
            IntPtr _p0 = input.Handle;
            IntPtr _p1 = window.Handle;
            IntPtr _p2 = output.Handle;
            void** args = stackalloc void*[4];
            args[0] = &_p0;
            args[1] = &_p1;
            args[2] = &_p2;
            args[3] = &n;


            LaunchKernel(kernel, gridSize, (uint)DefaultBlockSize, args);
            Synchronize();
            }
    }

    /// <summary>
    /// Computes magnitude from complex numbers: magnitude = sqrt(real^2 + imag^2).
    /// </summary>
    public unsafe void ComplexMagnitude(IGpuBuffer real, IGpuBuffer imag, IGpuBuffer magnitude, int n)
    {
        if (!IsAvailable)
            throw new InvalidOperationException("HIP backend not available");

        if (!_kernelCache.TryGetValue("complex_magnitude", out var kernel))
            throw new InvalidOperationException("complex_magnitude kernel not found");

        uint gridSize = (uint)((n + DefaultBlockSize - 1) / DefaultBlockSize);

            {
            IntPtr _p0 = real.Handle;
            IntPtr _p1 = imag.Handle;
            IntPtr _p2 = magnitude.Handle;
            void** args = stackalloc void*[4];
            args[0] = &_p0;
            args[1] = &_p1;
            args[2] = &_p2;
            args[3] = &n;


            LaunchKernel(kernel, gridSize, (uint)DefaultBlockSize, args);
            Synchronize();
            }
    }

    /// <summary>
    /// Computes phase from complex numbers: phase = atan2(imag, real).
    /// </summary>
    public unsafe void ComplexPhase(IGpuBuffer real, IGpuBuffer imag, IGpuBuffer phase, int n)
    {
        if (!IsAvailable)
            throw new InvalidOperationException("HIP backend not available");

        if (!_kernelCache.TryGetValue("complex_phase", out var kernel))
            throw new InvalidOperationException("complex_phase kernel not found");

        uint gridSize = (uint)((n + DefaultBlockSize - 1) / DefaultBlockSize);

            {
            IntPtr _p0 = real.Handle;
            IntPtr _p1 = imag.Handle;
            IntPtr _p2 = phase.Handle;
            void** args = stackalloc void*[4];
            args[0] = &_p0;
            args[1] = &_p1;
            args[2] = &_p2;
            args[3] = &n;


            LaunchKernel(kernel, gridSize, (uint)DefaultBlockSize, args);
            Synchronize();
            }
    }

    /// <summary>
    /// Converts polar coordinates to complex: real = mag*cos(phase), imag = mag*sin(phase).
    /// </summary>
    public unsafe void PolarToComplex(IGpuBuffer magnitude, IGpuBuffer phase, IGpuBuffer real, IGpuBuffer imag, int n)
    {
        if (!IsAvailable)
            throw new InvalidOperationException("HIP backend not available");

        if (!_kernelCache.TryGetValue("polar_to_complex", out var kernel))
            throw new InvalidOperationException("polar_to_complex kernel not found");

        uint gridSize = (uint)((n + DefaultBlockSize - 1) / DefaultBlockSize);

            {
            IntPtr _p0 = magnitude.Handle;
            IntPtr _p1 = phase.Handle;
            IntPtr _p2 = real.Handle;
            IntPtr _p3 = imag.Handle;
            void** args = stackalloc void*[5];
            args[0] = &_p0;
            args[1] = &_p1;
            args[2] = &_p2;
            args[3] = &_p3;
            args[4] = &n;


            LaunchKernel(kernel, gridSize, (uint)DefaultBlockSize, args);
            Synchronize();
            }
    }

    /// <summary>
    /// Applies Mel filterbank to power spectrum.
    /// </summary>
    public unsafe void ApplyMelFilterbank(IGpuBuffer powerSpec, IGpuBuffer filterbank, IGpuBuffer melSpec, int numFrames, int numFreqs, int nMels)
    {
        if (!IsAvailable)
            throw new InvalidOperationException("HIP backend not available");

        if (!_kernelCache.TryGetValue("apply_mel_filterbank", out var kernel))
            throw new InvalidOperationException("apply_mel_filterbank kernel not found");

        uint gridX = (uint)numFrames;
        uint gridY = (uint)((nMels + DefaultBlockSize - 1) / DefaultBlockSize);

            {
            IntPtr _p0 = powerSpec.Handle;
            IntPtr _p1 = filterbank.Handle;
            IntPtr _p2 = melSpec.Handle;
            void** args = stackalloc void*[6];
            args[0] = &_p0;
            args[1] = &_p1;
            args[2] = &_p2;
            args[3] = &numFrames;
            args[4] = &numFreqs;
            args[5] = &nMels;


            LaunchKernel2D(kernel, gridX, gridY, 1, (uint)DefaultBlockSize, args);
            Synchronize();
            }
    }

    /// <summary>
    /// Converts power spectrum to decibels.
    /// </summary>
    public unsafe void PowerToDb(IGpuBuffer power, IGpuBuffer db, int n, float refValue, float minDb)
    {
        if (!IsAvailable)
            throw new InvalidOperationException("HIP backend not available");

        if (!_kernelCache.TryGetValue("power_to_db", out var kernel))
            throw new InvalidOperationException("power_to_db kernel not found");

        uint gridSize = (uint)((n + DefaultBlockSize - 1) / DefaultBlockSize);

            {
            IntPtr _p0 = power.Handle;
            IntPtr _p1 = db.Handle;
            void** args = stackalloc void*[5];
            args[0] = &_p0;
            args[1] = &_p1;
            args[2] = &n;
            args[3] = &refValue;
            args[4] = &minDb;


            LaunchKernel(kernel, gridSize, (uint)DefaultBlockSize, args);
            Synchronize();
            }
    }

    /// <summary>
    /// Converts decibels to power spectrum.
    /// </summary>
    public unsafe void DbToPower(IGpuBuffer db, IGpuBuffer power, int n, float refValue)
    {
        if (!IsAvailable)
            throw new InvalidOperationException("HIP backend not available");

        if (!_kernelCache.TryGetValue("db_to_power", out var kernel))
            throw new InvalidOperationException("db_to_power kernel not found");

        uint gridSize = (uint)((n + DefaultBlockSize - 1) / DefaultBlockSize);

            {
            IntPtr _p0 = db.Handle;
            IntPtr _p1 = power.Handle;
            void** args = stackalloc void*[4];
            args[0] = &_p0;
            args[1] = &_p1;
            args[2] = &n;
            args[3] = &refValue;


            LaunchKernel(kernel, gridSize, (uint)DefaultBlockSize, args);
            Synchronize();
            }
    }

    /// <summary>
    /// Helper method to copy GPU buffer contents.
    /// </summary>
    private void HipCopyBuffer(IGpuBuffer source, IGpuBuffer destination, int count)
    {
        var size = (UIntPtr)(count * sizeof(float));
        var result = HipNativeBindings.hipMemcpy(
            destination.Handle,
            source.Handle,
            size,
            HipMemcpyKind.DeviceToDevice);
        HipNativeBindings.CheckError(result, "hipMemcpy D2D");
    }

    /// <summary>
    /// Helper method to zero GPU buffer contents.
    /// </summary>
    private void HipZeroBuffer(IGpuBuffer buffer, int count)
    {
        var size = (UIntPtr)(count * sizeof(float));
        var result = HipNativeBindings.hipMemset(buffer.Handle, 0, size);
        HipNativeBindings.CheckError(result, "hipMemset");
    }

    #endregion

    #region IAsyncGpuBackend Methods

    /// <inheritdoc/>
    public IGpuStream CreateStream(GpuStreamType streamType)
    {
        return new HipStream(this, streamType);
    }

    /// <inheritdoc/>
    public IGpuStream CreateStream(GpuStreamType streamType, int priority)
    {
        return new HipStream(this, streamType, priority);
    }

    /// <inheritdoc/>
    public IGpuEvent CreateEvent()
    {
        return new HipEvent(this, null, true);
    }

    /// <inheritdoc/>
    public IGpuEvent CreateEvent(bool enableTiming)
    {
        return new HipEvent(this, null, enableTiming);
    }

    /// <inheritdoc/>
    public void RecordEvent(IGpuEvent gpuEvent, IGpuStream stream)
    {
        if (gpuEvent is not HipEvent hipEvent)
        {
            throw new ArgumentException("Event must be a HipEvent", nameof(gpuEvent));
        }

        if (stream is not HipStream)
        {
            throw new ArgumentException("Stream must be a HipStream", nameof(stream));
        }

        hipEvent.Record(stream);
    }

    /// <inheritdoc/>
    public void StreamWaitEvent(IGpuStream stream, IGpuEvent gpuEvent)
    {
        if (stream is not HipStream hipStream)
        {
            throw new ArgumentException("Stream must be a HipStream", nameof(stream));
        }

        hipStream.WaitEvent(gpuEvent);
    }

    /// <inheritdoc/>
    public GpuSyncPoint CreateSyncPoint(IGpuStream stream)
    {
        if (stream is not HipStream hipStream)
        {
            throw new ArgumentException("Stream must be a HipStream", nameof(stream));
        }

        return new HipSyncPoint(this, hipStream);
    }

    /// <inheritdoc/>
    public GpuSyncPoint CreateSyncPoint()
    {
        return CreateSyncPoint(DefaultStream);
    }

    /// <inheritdoc/>
    public unsafe void UploadBufferAsync(float[] data, IGpuBuffer buffer, IGpuStream stream)
    {
        int byteSize = data.Length * sizeof(float);
        var pinnedPtr = _pinnedPool.Rent(byteSize);
        try
        {
            // Copy managed data to pinned host memory
            fixed (float* src = data)
            {
                Buffer.MemoryCopy(src, (void*)pinnedPtr, byteSize, byteSize);
            }

            var result = HipNativeBindings.hipMemcpyAsync(
                buffer.Handle,
                pinnedPtr,
                (UIntPtr)byteSize,
                HipMemcpyKind.HostToDevice,
                stream.Handle);
            HipNativeBindings.CheckError(result, "hipMemcpyAsync H2D");

            // Synchronize to safely return the pinned buffer
            var syncResult = HipNativeBindings.hipStreamSynchronize(stream.Handle);
            HipNativeBindings.CheckError(syncResult, "hipStreamSynchronize(pinned upload)");
        }
        finally
        {
            _pinnedPool.Return(pinnedPtr, byteSize);
        }
    }

    /// <inheritdoc/>
    public unsafe void UploadBufferAsync(ReadOnlySpan<float> data, IGpuBuffer buffer, IGpuStream stream)
    {
        int byteSize = data.Length * sizeof(float);
        var pinnedPtr = _pinnedPool.Rent(byteSize);
        try
        {
            fixed (float* src = data)
            {
                Buffer.MemoryCopy(src, (void*)pinnedPtr, byteSize, byteSize);
            }

            var result = HipNativeBindings.hipMemcpyAsync(
                buffer.Handle,
                pinnedPtr,
                (UIntPtr)byteSize,
                HipMemcpyKind.HostToDevice,
                stream.Handle);
            HipNativeBindings.CheckError(result, "hipMemcpyAsync H2D");

            var syncResult = HipNativeBindings.hipStreamSynchronize(stream.Handle);
            HipNativeBindings.CheckError(syncResult, "hipStreamSynchronize(pinned upload)");
        }
        finally
        {
            _pinnedPool.Return(pinnedPtr, byteSize);
        }
    }

    /// <inheritdoc/>
    public unsafe void DownloadBufferAsync(IGpuBuffer buffer, float[] destination, IGpuStream stream)
    {
        int byteSize = destination.Length * sizeof(float);
        var pinnedPtr = _pinnedPool.Rent(byteSize);
        try
        {
            var result = HipNativeBindings.hipMemcpyAsync(
                pinnedPtr,
                buffer.Handle,
                (UIntPtr)byteSize,
                HipMemcpyKind.DeviceToHost,
                stream.Handle);
            HipNativeBindings.CheckError(result, "hipMemcpyAsync D2H");

            // Must synchronize before copying out of pinned buffer
            var syncResult = HipNativeBindings.hipStreamSynchronize(stream.Handle);
            HipNativeBindings.CheckError(syncResult, "hipStreamSynchronize(pinned download)");

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
        var byteSize = (UIntPtr)(size * sizeof(float));
        var result = HipNativeBindings.hipMemcpyAsync(
            destination.Handle,
            source.Handle,
            byteSize,
            HipMemcpyKind.DeviceToDevice,
            stream.Handle);
        HipNativeBindings.CheckError(result, "hipMemcpyAsync D2D");
    }

    /// <inheritdoc/>
    public void GemmAsync(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int M, int N, int K,
        float alpha, float beta, IGpuStream stream)
    {
        GemmOnStream(A, B, C, M, N, K, alpha, beta, stream.Handle, synchronize: false);
    }

    /// <inheritdoc/>
    public void FusedGemmBiasActivationAsync(IGpuBuffer A, IGpuBuffer B, IGpuBuffer bias, IGpuBuffer output,
        int M, int N, int K, FusedActivationType activation, IGpuStream stream)
    {
        // Map activation type to appropriate kernel
        string kernelName = activation switch
        {
            FusedActivationType.ReLU => "gemm_bias_relu",
            FusedActivationType.Sigmoid => "gemm_bias_sigmoid",
            FusedActivationType.Tanh => "gemm_bias_tanh",
            FusedActivationType.None => "gemm_bias",
            _ => "gemm_bias_relu" // Default to ReLU
        };

        ExecuteFusedGemmOnStream(kernelName, A, B, bias, output, M, N, K, stream.Handle, synchronize: false);
    }

    /// <summary>
    /// Executes GEMM on a specific stream with optional synchronization.
    /// </summary>
    private unsafe void GemmOnStream(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int M, int N, int K,
        float alpha, float beta, IntPtr stream, bool synchronize)
    {
        var bufferA = (HipGpuBuffer)A;
        var bufferB = (HipGpuBuffer)B;
        var bufferC = (HipGpuBuffer)C;

        IntPtr kernel = SelectGemmKernel(M, N, K);
        if (kernel == IntPtr.Zero)
        {
            throw new InvalidOperationException("No suitable GEMM kernel available");
        }

        GemmKernelType kernelType = GetKernelType(kernel);

        int tileM, tileN, blockSize;
        switch (kernelType)
        {
            case GemmKernelType.Mfma:
                tileM = 128;
                tileN = 128;
                blockSize = 256;
                break;
            case GemmKernelType.RdnaWave32:
                tileM = 32;
                tileN = 32;
                blockSize = 256;
                break;
            case GemmKernelType.Scalar:
            default:
                tileM = 16;
                tileN = 16;
                blockSize = 256;
                break;
        }

        uint gridDimX = (uint)((M + tileM - 1) / tileM);
        uint gridDimY = (uint)((N + tileN - 1) / tileN);

        IntPtr[] kernelArgs = new IntPtr[8];
            {
            var argA = bufferA.Handle;
            var argB = bufferB.Handle;
            var argC = bufferC.Handle;
            void** args = stackalloc void*[8];
            args[0] = &argA;
            args[1] = &argB;
            args[2] = &argC;
            args[3] = &M;
            args[4] = &N;
            args[5] = &K;
            args[6] = &alpha;
            args[7] = &beta;

            for (int i = 0; i < 8; i++)
            {
            }

            GCHandle kernelParamsHandle = GCHandle.Alloc(kernelArgs, GCHandleType.Pinned);
            try
            {
                var result = HipNativeBindings.hipModuleLaunchKernel(
                    kernel,
                    gridDimX, gridDimY, 1,
                    (uint)blockSize, 1, 1,
                    0,
                    stream,
                    kernelParamsHandle.AddrOfPinnedObject(),
                    IntPtr.Zero);

                HipNativeBindings.CheckError(result, "hipModuleLaunchKernel");
            }
            finally
            {
                kernelParamsHandle.Free();
            }
            }

        if (synchronize)
        {
            var syncResult = HipNativeBindings.hipStreamSynchronize(stream);
            HipNativeBindings.CheckError(syncResult, "hipStreamSynchronize");
        }
    }

    /// <inheritdoc/>
    public void SynchronizeStream(IGpuStream stream)
    {
        if (stream is not HipStream)
        {
            throw new ArgumentException("Stream must be a HipStream", nameof(stream));
        }

        stream.Synchronize();
    }

    /// <inheritdoc/>
    public bool QueryStreamComplete(IGpuStream stream)
    {
        if (stream is not HipStream)
        {
            throw new ArgumentException("Stream must be a HipStream", nameof(stream));
        }

        return stream.Query();
    }

    /// <inheritdoc/>
    public bool QueryEventComplete(IGpuEvent gpuEvent)
    {
        if (gpuEvent is not HipEvent)
        {
            throw new ArgumentException("Event must be a HipEvent", nameof(gpuEvent));
        }

        return gpuEvent.Query();
    }

    /// <inheritdoc/>
    public float GetEventElapsedTime(IGpuEvent start, IGpuEvent end)
    {
        if (end is not HipEvent hipEnd)
        {
            throw new ArgumentException("Event must be a HipEvent", nameof(end));
        }

        return hipEnd.GetElapsedTime(start);
    }

    #endregion

    public void Synchronize()
    {
        if (_stream != IntPtr.Zero)
        {
            var result = HipNativeBindings.hipStreamSynchronize(_stream);
            // Don't throw on sync errors, just log
            if (result != HipError.Success)
            {
                System.Diagnostics.Debug.WriteLine($"hipStreamSynchronize warning: {result}");
            }
        }
    }

    public void Copy(IGpuBuffer source, int sourceOffset, IGpuBuffer destination, int destinationOffset, int length)
    {
        IntPtr srcPtr = source.Handle + sourceOffset * sizeof(float);
        IntPtr dstPtr = destination.Handle + destinationOffset * sizeof(float);
        var sizeBytes = (UIntPtr)(length * sizeof(float));
        var result = HipNativeBindings.hipMemcpy(dstPtr, srcPtr, sizeBytes, HipMemcpyKind.DeviceToDevice);
        HipNativeBindings.CheckError(result, "hipMemcpy D2D (strided)");
    }

    public unsafe void ArgMaxAxis(IGpuBuffer A, IGpuBuffer indices, int outerSize, int reduceSize)
    {
        if (!_kernelCache.TryGetValue("argmax_axis", out var kernel))
            throw new InvalidOperationException("HIP kernel not found: argmax_axis");

        IntPtr inputPtr = A.Handle;
        IntPtr indicesPtr = indices.Handle;
        void** args = stackalloc void*[4];
        args[0] = &inputPtr; args[1] = &indicesPtr; args[2] = &outerSize; args[3] = &reduceSize;
        uint grid = (uint)((outerSize + DefaultBlockSize - 1) / DefaultBlockSize);
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    public unsafe void GenerateRandomUniform(IGpuBuffer output, int size, float min, float max, ulong seed)
    {
        if (!_kernelCache.TryGetValue("generate_random_uniform", out var kernel))
            throw new InvalidOperationException("HIP kernel not found: generate_random_uniform");

        IntPtr outputPtr = output.Handle;
        void** args = stackalloc void*[5];
        args[0] = &outputPtr; args[1] = &size; args[2] = &min; args[3] = &max; args[4] = &seed;
        uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    public unsafe void GenerateRandomNormal(IGpuBuffer output, int size, float mean, float stdDev, ulong seed)
    {
        if (!_kernelCache.TryGetValue("generate_random_normal", out var kernel))
            throw new InvalidOperationException("HIP kernel not found: generate_random_normal");

        IntPtr outputPtr = output.Handle;
        void** args = stackalloc void*[5];
        args[0] = &outputPtr; args[1] = &size; args[2] = &mean; args[3] = &stdDev; args[4] = &seed;
        uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    public void GenerateSecureRandomUniform(IGpuBuffer output, int size, float min, float max)
    {
        if (size <= 0) return;
        var data = new float[size];
        try
        {
            Helpers.SimdRandom.SecureFillFloats(data.AsSpan());
            float range = max - min;
            for (int i = 0; i < size; i++) data[i] = data[i] * range + min;
            UploadToBuffer(output, data);
        }
        finally { Array.Clear(data, 0, size); }
    }

    public unsafe void RbfForward(IGpuBuffer input, IGpuBuffer centers, IGpuBuffer epsilons, IGpuBuffer output, int batchSize, int numCenters, int inputDim)
    {
        if (!_kernelCache.TryGetValue("rbf_forward", out var kernel))
            throw new InvalidOperationException("HIP kernel not found: rbf_forward");

        IntPtr inputPtr = input.Handle, centersPtr = centers.Handle;
        IntPtr epsilonsPtr = epsilons.Handle, outputPtr = output.Handle;
        void** args = stackalloc void*[7];
        args[0] = &inputPtr; args[1] = &centersPtr; args[2] = &epsilonsPtr; args[3] = &outputPtr;
        args[4] = &batchSize; args[5] = &numCenters; args[6] = &inputDim;
        uint total = (uint)(batchSize * numCenters);
        LaunchKernel(kernel, (total + DefaultBlockSize - 1) / DefaultBlockSize, DefaultBlockSize, args);
    }

    public unsafe void StdpUpdate(IGpuBuffer weights, IGpuBuffer preTrace, IGpuBuffer postTrace, IGpuBuffer preSpike, IGpuBuffer postSpike,
        float ltpRate, float ltdRate, float homeostasisRate, float minWeight, float maxWeight, int numPre, int numPost)
    {
        if (!_kernelCache.TryGetValue("stdp_update", out var kernel))
            throw new InvalidOperationException("HIP kernel not found: stdp_update");

        IntPtr wPtr = weights.Handle, preTPtr = preTrace.Handle, postTPtr = postTrace.Handle;
        IntPtr preSPtr = preSpike.Handle, postSPtr = postSpike.Handle;
        void** args = stackalloc void*[12];
        args[0] = &wPtr; args[1] = &preTPtr; args[2] = &postTPtr;
        args[3] = &preSPtr; args[4] = &postSPtr;
        args[5] = &ltpRate; args[6] = &ltdRate; args[7] = &homeostasisRate;
        args[8] = &minWeight; args[9] = &maxWeight;
        args[10] = &numPre; args[11] = &numPost;
        uint total = (uint)(numPre * numPost);
        LaunchKernel(kernel, (total + DefaultBlockSize - 1) / DefaultBlockSize, DefaultBlockSize, args);
    }

    public unsafe void UpdateTraces(IGpuBuffer traces, IGpuBuffer spikes, IGpuBuffer input, float decay, float threshold, int size)
    {
        if (!_kernelCache.TryGetValue("update_traces", out var kernel))
            throw new InvalidOperationException("HIP kernel not found: update_traces");

        IntPtr tracesPtr = traces.Handle, spikesPtr = spikes.Handle, inputPtr = input.Handle;
        void** args = stackalloc void*[6];
        args[0] = &tracesPtr; args[1] = &spikesPtr; args[2] = &inputPtr;
        args[3] = &decay; args[4] = &threshold; args[5] = &size;
        uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    #region Hyperbolic Geometry Operations

    public unsafe void PoincareProject(IGpuBuffer input, IGpuBuffer output, int batchSize, int dim, float curvature, float epsilon = 1e-5f)
    {
        if (!_kernelCache.TryGetValue("poincare_project", out var kernel))
            throw new InvalidOperationException("HIP kernel not found: poincare_project");

        uint grid = (uint)((batchSize + DefaultBlockSize - 1) / DefaultBlockSize);
            {
            IntPtr _p0 = input.Handle;
            IntPtr _p1 = output.Handle;
            void** args = stackalloc void*[6];
            args[0] = &_p0;
            args[1] = &_p1;
            args[2] = &batchSize;
            args[3] = &dim;
            args[4] = &curvature;
            args[5] = &epsilon;


            LaunchKernel(kernel, grid, DefaultBlockSize, args);
            Synchronize();
            }
    }

    public unsafe void MobiusAdd(IGpuBuffer x, IGpuBuffer y, IGpuBuffer output, int batchSize, int dim, float curvature)
    {
        if (!_kernelCache.TryGetValue("mobius_add", out var kernel))
            throw new InvalidOperationException("HIP kernel not found: mobius_add");

        uint grid = (uint)((batchSize + DefaultBlockSize - 1) / DefaultBlockSize);
            {
            IntPtr _p0 = x.Handle;
            IntPtr _p1 = y.Handle;
            IntPtr _p2 = output.Handle;
            void** args = stackalloc void*[6];
            args[0] = &_p0;
            args[1] = &_p1;
            args[2] = &_p2;
            args[3] = &batchSize;
            args[4] = &dim;
            args[5] = &curvature;


            LaunchKernel(kernel, grid, DefaultBlockSize, args);
            Synchronize();
            }
    }

    public unsafe void PoincareExpMap(IGpuBuffer basePoint, IGpuBuffer tangentVec, IGpuBuffer output, int batchSize, int dim, float curvature)
    {
        if (!_kernelCache.TryGetValue("poincare_exp_map", out var kernel))
            throw new InvalidOperationException("HIP kernel not found: poincare_exp_map");

        uint grid = (uint)((batchSize + DefaultBlockSize - 1) / DefaultBlockSize);
            {
            IntPtr _p0 = basePoint.Handle;
            IntPtr _p1 = tangentVec.Handle;
            IntPtr _p2 = output.Handle;
            void** args = stackalloc void*[6];
            args[0] = &_p0;
            args[1] = &_p1;
            args[2] = &_p2;
            args[3] = &batchSize;
            args[4] = &dim;
            args[5] = &curvature;


            LaunchKernel(kernel, grid, DefaultBlockSize, args);
            Synchronize();
            }
    }

    public unsafe void PoincareDistance(IGpuBuffer x, IGpuBuffer y, IGpuBuffer output, int batchSize, int dim, float curvature)
    {
        if (!_kernelCache.TryGetValue("poincare_distance", out var kernel))
            throw new InvalidOperationException("HIP kernel not found: poincare_distance");

        uint grid = (uint)((batchSize + DefaultBlockSize - 1) / DefaultBlockSize);
            {
            IntPtr _p0 = x.Handle;
            IntPtr _p1 = y.Handle;
            IntPtr _p2 = output.Handle;
            void** args = stackalloc void*[6];
            args[0] = &_p0;
            args[1] = &_p1;
            args[2] = &_p2;
            args[3] = &batchSize;
            args[4] = &dim;
            args[5] = &curvature;


            LaunchKernel(kernel, grid, DefaultBlockSize, args);
            Synchronize();
            }
    }

    // Maximum dimension supported by hyperbolic kernels (due to shared memory limits)
    private const int HyperbolicMaxDim = 128;

    public unsafe void HyperbolicLinearForward(IGpuBuffer input, IGpuBuffer weights, IGpuBuffer biases, IGpuBuffer output,
        int batchSize, int inputFeatures, int outputFeatures, float curvature, float epsilon)
    {
        // Validate dimension limits - kernel will silently clamp dimensions > 128
        if (inputFeatures > HyperbolicMaxDim)
            throw new ArgumentOutOfRangeException(nameof(inputFeatures),
                $"HyperbolicLinearForward requires inputFeatures <= {HyperbolicMaxDim}. Got {inputFeatures}.");
        if (outputFeatures > HyperbolicMaxDim)
            throw new ArgumentOutOfRangeException(nameof(outputFeatures),
                $"HyperbolicLinearForward requires outputFeatures <= {HyperbolicMaxDim}. Got {outputFeatures}.");

        if (!_kernelCache.TryGetValue("hyperbolic_linear_forward", out var kernel))
            throw new InvalidOperationException("HIP kernel not found: hyperbolic_linear_forward");

        int totalThreads = batchSize * outputFeatures;
        uint grid = (uint)((totalThreads + DefaultBlockSize - 1) / DefaultBlockSize);
            {
            IntPtr _p0 = input.Handle;
            IntPtr _p1 = weights.Handle;
            IntPtr _p2 = biases.Handle;
            IntPtr _p3 = output.Handle;
            void** args = stackalloc void*[9];
            args[0] = &_p0;
            args[1] = &_p1;
            args[2] = &_p2;
            args[3] = &_p3;
            args[4] = &batchSize;
            args[5] = &inputFeatures;
            args[6] = &outputFeatures;
            args[7] = &curvature;
            args[8] = &epsilon;


            LaunchKernel(kernel, grid, DefaultBlockSize, args);
            Synchronize();
            }
    }

    /// <inheritdoc/>
    public unsafe void HyperbolicLinearBackwardInput(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer weights, IGpuBuffer gradInput,
        int batchSize, int inputFeatures, int outputFeatures, float curvature)
    {
        // Validate dimension limits - kernel will silently clamp dimensions > 128
        if (inputFeatures > HyperbolicMaxDim)
            throw new ArgumentOutOfRangeException(nameof(inputFeatures),
                $"HyperbolicLinearBackwardInput requires inputFeatures <= {HyperbolicMaxDim}. Got {inputFeatures}.");
        if (outputFeatures > HyperbolicMaxDim)
            throw new ArgumentOutOfRangeException(nameof(outputFeatures),
                $"HyperbolicLinearBackwardInput requires outputFeatures <= {HyperbolicMaxDim}. Got {outputFeatures}.");

        if (!_kernelCache.TryGetValue("hyperbolic_linear_backward_input", out var kernel))
            throw new InvalidOperationException("HIP kernel not found: hyperbolic_linear_backward_input");

        int totalThreads = batchSize * inputFeatures;
        uint grid = (uint)((totalThreads + DefaultBlockSize - 1) / DefaultBlockSize);
            {
            IntPtr _p0 = gradOutput.Handle;
            IntPtr _p1 = input.Handle;
            IntPtr _p2 = weights.Handle;
            IntPtr _p3 = gradInput.Handle;
            void** args = stackalloc void*[8];
            args[0] = &_p0;
            args[1] = &_p1;
            args[2] = &_p2;
            args[3] = &_p3;
            args[4] = &batchSize;
            args[5] = &inputFeatures;
            args[6] = &outputFeatures;
            args[7] = &curvature;


            LaunchKernel(kernel, grid, DefaultBlockSize, args);
            Synchronize();
            }
    }

    /// <inheritdoc/>
    public unsafe void HyperbolicLinearBackwardWeights(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradWeights,
        int batchSize, int inputFeatures, int outputFeatures, float curvature)
    {
        // Validate dimension limits - kernel will silently clamp dimensions > 128
        if (inputFeatures > HyperbolicMaxDim)
            throw new ArgumentOutOfRangeException(nameof(inputFeatures),
                $"HyperbolicLinearBackwardWeights requires inputFeatures <= {HyperbolicMaxDim}. Got {inputFeatures}.");
        if (outputFeatures > HyperbolicMaxDim)
            throw new ArgumentOutOfRangeException(nameof(outputFeatures),
                $"HyperbolicLinearBackwardWeights requires outputFeatures <= {HyperbolicMaxDim}. Got {outputFeatures}.");

        if (!_kernelCache.TryGetValue("hyperbolic_linear_backward_weights", out var kernel))
            throw new InvalidOperationException("HIP kernel not found: hyperbolic_linear_backward_weights");

        int totalThreads = outputFeatures * inputFeatures;
        uint grid = (uint)((totalThreads + DefaultBlockSize - 1) / DefaultBlockSize);
            {
            IntPtr _p0 = gradOutput.Handle;
            IntPtr _p1 = input.Handle;
            IntPtr _p2 = gradWeights.Handle;
            void** args = stackalloc void*[7];
            args[0] = &_p0;
            args[1] = &_p1;
            args[2] = &_p2;
            args[3] = &batchSize;
            args[4] = &inputFeatures;
            args[5] = &outputFeatures;
            args[6] = &curvature;


            LaunchKernel(kernel, grid, DefaultBlockSize, args);
            Synchronize();
            }
    }

    /// <inheritdoc/>
    public unsafe void HyperbolicLinearBackwardBiases(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradBiases,
        int batchSize, int inputFeatures, int outputFeatures, float curvature)
    {
        if (!_kernelCache.TryGetValue("hyperbolic_linear_backward_biases", out var kernel))
            throw new InvalidOperationException("HIP kernel not found: hyperbolic_linear_backward_biases");

        // Bias gradients: one per output feature (not outputFeatures * inputFeatures)
        int totalThreads = outputFeatures;
        uint grid = (uint)((totalThreads + DefaultBlockSize - 1) / DefaultBlockSize);
            {
            IntPtr _p0 = gradOutput.Handle;
            IntPtr _p1 = input.Handle;
            IntPtr _p2 = gradBiases.Handle;
            void** args = stackalloc void*[7];
            args[0] = &_p0;
            args[1] = &_p1;
            args[2] = &_p2;
            args[3] = &batchSize;
            args[4] = &inputFeatures;
            args[5] = &outputFeatures;
            args[6] = &curvature;


            LaunchKernel(kernel, grid, DefaultBlockSize, args);
            Synchronize();
            }
    }

    #endregion

    #region Octonion Algebra Operations

    public unsafe void OctonionMultiply(IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, int count)
    {
        if (!_kernelCache.TryGetValue("octonion_multiply", out var kernel))
            throw new InvalidOperationException("HIP kernel not found: octonion_multiply");

        uint grid = (uint)((count + DefaultBlockSize - 1) / DefaultBlockSize);
            {
            IntPtr _p0 = a.Handle;
            IntPtr _p1 = b.Handle;
            IntPtr _p2 = output.Handle;
            void** args = stackalloc void*[4];
            args[0] = &_p0;
            args[1] = &_p1;
            args[2] = &_p2;
            args[3] = &count;


            LaunchKernel(kernel, grid, DefaultBlockSize, args);
            Synchronize();
            }
    }

    public unsafe void OctonionAdd(IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, int count)
    {
        if (!_kernelCache.TryGetValue("octonion_add", out var kernel))
            throw new InvalidOperationException("HIP kernel not found: octonion_add");

        int totalElements = count * 8;
        uint grid = (uint)((totalElements + DefaultBlockSize - 1) / DefaultBlockSize);
            {
            IntPtr _p0 = a.Handle;
            IntPtr _p1 = b.Handle;
            IntPtr _p2 = output.Handle;
            void** args = stackalloc void*[4];
            args[0] = &_p0;
            args[1] = &_p1;
            args[2] = &_p2;
            args[3] = &count;


            LaunchKernel(kernel, grid, DefaultBlockSize, args);
            Synchronize();
            }
    }

    public unsafe void OctonionLinearForward(IGpuBuffer input, IGpuBuffer weights, IGpuBuffer biases, IGpuBuffer output,
        int batchSize, int inputFeatures, int outputFeatures)
    {
        if (!_kernelCache.TryGetValue("octonion_linear_forward", out var kernel))
            throw new InvalidOperationException("HIP kernel not found: octonion_linear_forward");

        int totalThreads = batchSize * outputFeatures;
        uint grid = (uint)((totalThreads + DefaultBlockSize - 1) / DefaultBlockSize);
            {
            IntPtr _p0 = input.Handle;
            IntPtr _p1 = weights.Handle;
            IntPtr _p2 = biases.Handle;
            IntPtr _p3 = output.Handle;
            void** args = stackalloc void*[7];
            args[0] = &_p0;
            args[1] = &_p1;
            args[2] = &_p2;
            args[3] = &_p3;
            args[4] = &batchSize;
            args[5] = &inputFeatures;
            args[6] = &outputFeatures;


            LaunchKernel(kernel, grid, DefaultBlockSize, args);
            Synchronize();
            }
    }

    public unsafe void OctonionLinearBackwardInput(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer weights, IGpuBuffer gradInput,
        int batchSize, int inputFeatures, int outputFeatures)
    {
        if (!_kernelCache.TryGetValue("octonion_linear_backward_input", out var kernel))
            throw new InvalidOperationException("HIP kernel not found: octonion_linear_backward_input");

        int totalThreads = batchSize * inputFeatures;
        uint grid = (uint)((totalThreads + DefaultBlockSize - 1) / DefaultBlockSize);
            {
            IntPtr _p0 = gradOutput.Handle;
            IntPtr _p1 = input.Handle;
            IntPtr _p2 = weights.Handle;
            IntPtr _p3 = gradInput.Handle;
            void** args = stackalloc void*[7];
            args[0] = &_p0;
            args[1] = &_p1;
            args[2] = &_p2;
            args[3] = &_p3;
            args[4] = &batchSize;
            args[5] = &inputFeatures;
            args[6] = &outputFeatures;


            LaunchKernel(kernel, grid, DefaultBlockSize, args);
            Synchronize();
            }
    }

    public unsafe void OctonionLinearBackwardWeights(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradWeights,
        int batchSize, int inputFeatures, int outputFeatures)
    {
        if (!_kernelCache.TryGetValue("octonion_linear_backward_weights", out var kernel))
            throw new InvalidOperationException("HIP kernel not found: octonion_linear_backward_weights");

        int totalThreads = outputFeatures * inputFeatures;
        uint grid = (uint)((totalThreads + DefaultBlockSize - 1) / DefaultBlockSize);
            {
            IntPtr _p0 = gradOutput.Handle;
            IntPtr _p1 = input.Handle;
            IntPtr _p2 = gradWeights.Handle;
            void** args = stackalloc void*[6];
            args[0] = &_p0;
            args[1] = &_p1;
            args[2] = &_p2;
            args[3] = &batchSize;
            args[4] = &inputFeatures;
            args[5] = &outputFeatures;


            LaunchKernel(kernel, grid, DefaultBlockSize, args);
            Synchronize();
            }
    }

    public unsafe void OctonionLinearBackwardBiases(IGpuBuffer gradOutput, IGpuBuffer gradBiases,
        int batchSize, int outputFeatures)
    {
        if (!_kernelCache.TryGetValue("octonion_linear_backward_biases", out var kernel))
            throw new InvalidOperationException("HIP kernel not found: octonion_linear_backward_biases");

        uint grid = (uint)((outputFeatures + DefaultBlockSize - 1) / DefaultBlockSize);
            {
            IntPtr _p0 = gradOutput.Handle;
            IntPtr _p1 = gradBiases.Handle;
            void** args = stackalloc void*[4];
            args[0] = &_p0;
            args[1] = &_p1;
            args[2] = &batchSize;
            args[3] = &outputFeatures;


            LaunchKernel(kernel, grid, DefaultBlockSize, args);
            Synchronize();
            }
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
            throw new InvalidOperationException("HIP kernel not found: octonion_linear_forward_fused_relu");

        int totalOutputs = batchSize * outputFeatures;
        uint grid = (uint)((totalOutputs + DefaultBlockSize - 1) / DefaultBlockSize);
        {
            IntPtr _p0 = input.Handle;
            IntPtr _p1 = weights.Handle;
            IntPtr _p2 = biases.Handle;
            IntPtr _p3 = output.Handle;
            void** args = stackalloc void*[7];
            args[0] = &_p0;
            args[1] = &_p1;
            args[2] = &_p2;
            args[3] = &_p3;
            args[4] = &batchSize;
            args[5] = &inputFeatures;
            args[6] = &outputFeatures;
            LaunchKernel(kernel, grid, (uint)DefaultBlockSize, args);
        }
    }

    #endregion

    #region Complex Tensor Operations

    public unsafe void ComplexMultiply(IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, int numPairs)
    {
        if (numPairs <= 0) return;
        if (numPairs * 2 > a.Size || numPairs * 2 > b.Size || numPairs * 2 > output.Size)
            throw new ArgumentException($"numPairs ({numPairs}) requires {numPairs * 2} elements but buffer sizes are a={a.Size}, b={b.Size}, out={output.Size}.");
        if (!_kernelCache.TryGetValue("complex_multiply", out var kernel))
            throw new InvalidOperationException("HIP kernel not found: complex_multiply");
        uint grid = (uint)((numPairs + DefaultBlockSize - 1) / DefaultBlockSize);
        {
            IntPtr _p0 = a.Handle, _p1 = b.Handle, _p2 = output.Handle;
            void** args = stackalloc void*[4];
            args[0] = &_p0; args[1] = &_p1; args[2] = &_p2; args[3] = &numPairs;
            LaunchKernel(kernel, grid, (uint)DefaultBlockSize, args);
        }
    }

    public unsafe void ComplexConjugate(IGpuBuffer input, IGpuBuffer output, int numPairs)
    {
        if (numPairs <= 0) return;
        if (numPairs * 2 > input.Size || numPairs * 2 > output.Size)
            throw new ArgumentException($"numPairs ({numPairs}) requires {numPairs * 2} elements but buffer sizes are in={input.Size}, out={output.Size}.");
        if (!_kernelCache.TryGetValue("complex_conjugate", out var kernel))
            throw new InvalidOperationException("HIP kernel not found: complex_conjugate");
        uint grid = (uint)((numPairs + DefaultBlockSize - 1) / DefaultBlockSize);
        {
            IntPtr _p0 = input.Handle, _p1 = output.Handle;
            void** args = stackalloc void*[3];
            args[0] = &_p0; args[1] = &_p1; args[2] = &numPairs;
            LaunchKernel(kernel, grid, (uint)DefaultBlockSize, args);
        }
    }

    public unsafe void ComplexMagnitude(IGpuBuffer input, IGpuBuffer output, int numPairs)
    {
        if (numPairs <= 0) return;
        if (numPairs * 2 > input.Size)
            throw new ArgumentException($"numPairs ({numPairs}) requires {numPairs * 2} elements but input buffer has {input.Size}.");
        if (numPairs > output.Size)
            throw new ArgumentException($"numPairs ({numPairs}) exceeds output buffer size ({output.Size}).");
        if (!_kernelCache.TryGetValue("complex_magnitude", out var kernel))
            throw new InvalidOperationException("HIP kernel not found: complex_magnitude");
        uint grid = (uint)((numPairs + DefaultBlockSize - 1) / DefaultBlockSize);
        {
            IntPtr _p0 = input.Handle, _p1 = output.Handle;
            void** args = stackalloc void*[3];
            args[0] = &_p0; args[1] = &_p1; args[2] = &numPairs;
            LaunchKernel(kernel, grid, (uint)DefaultBlockSize, args);
        }
    }

    #endregion

    #region Quantum Computing Operations

    public unsafe void QuantumMeasurement(IGpuBuffer realPart, IGpuBuffer imagPart, IGpuBuffer probabilities, int batchSize, int stateSize)
    {
        if (!_kernelCache.TryGetValue("quantum_measurement", out var kernel))
            throw new InvalidOperationException("HIP kernel not found: quantum_measurement");

        int totalElements = batchSize * stateSize;
        uint grid = (uint)((totalElements + DefaultBlockSize - 1) / DefaultBlockSize);
            {
            IntPtr _p0 = realPart.Handle;
            IntPtr _p1 = imagPart.Handle;
            IntPtr _p2 = probabilities.Handle;
            void** args = stackalloc void*[5];
            args[0] = &_p0;
            args[1] = &_p1;
            args[2] = &_p2;
            args[3] = &batchSize;
            args[4] = &stateSize;


            LaunchKernel(kernel, grid, DefaultBlockSize, args);
            Synchronize();
            }
    }

    public unsafe void NormalizeProbabilities(IGpuBuffer probabilities, int batchSize, int stateSize)
    {
        if (!_kernelCache.TryGetValue("normalize_probabilities", out var kernel))
            throw new InvalidOperationException("HIP kernel not found: normalize_probabilities");

        uint sharedMemSize = (uint)(DefaultBlockSize * sizeof(float));
            {
            IntPtr _p0 = probabilities.Handle;
            void** args = stackalloc void*[3];
            args[0] = &_p0;
            args[1] = &batchSize;
            args[2] = &stateSize;


            LaunchKernel(kernel, (uint)batchSize, DefaultBlockSize, args, sharedMemSize);
            Synchronize();
            }
    }

    public unsafe void ComplexMatVec(IGpuBuffer matReal, IGpuBuffer matImag, IGpuBuffer vecReal, IGpuBuffer vecImag,
        IGpuBuffer outReal, IGpuBuffer outImag, int batchSize, int dim)
    {
        if (!_kernelCache.TryGetValue("complex_matvec", out var kernel))
            throw new InvalidOperationException("HIP kernel not found: complex_matvec");

        int totalElements = batchSize * dim;
        uint grid = (uint)((totalElements + DefaultBlockSize - 1) / DefaultBlockSize);
            {
            IntPtr _p0 = matReal.Handle;
            IntPtr _p1 = matImag.Handle;
            IntPtr _p2 = vecReal.Handle;
            IntPtr _p3 = vecImag.Handle;
            IntPtr _p4 = outReal.Handle;
            IntPtr _p5 = outImag.Handle;
            void** args = stackalloc void*[8];
            args[0] = &_p0;
            args[1] = &_p1;
            args[2] = &_p2;
            args[3] = &_p3;
            args[4] = &_p4;
            args[5] = &_p5;
            args[6] = &batchSize;
            args[7] = &dim;


            LaunchKernel(kernel, grid, DefaultBlockSize, args);
            Synchronize();
            }
    }

    public unsafe void QuantumRotation(IGpuBuffer stateReal, IGpuBuffer stateImag, IGpuBuffer outReal, IGpuBuffer outImag,
        IGpuBuffer angles, int numQubits, int batchSize)
    {
        if (!_kernelCache.TryGetValue("quantum_rotation", out var kernel))
            throw new InvalidOperationException("HIP kernel not found: quantum_rotation");

            {
            IntPtr _p0 = stateReal.Handle;
            IntPtr _p1 = stateImag.Handle;
            IntPtr _p2 = outReal.Handle;
            IntPtr _p3 = outImag.Handle;
            IntPtr _p4 = angles.Handle;
            void** args = stackalloc void*[7];
            args[0] = &_p0;
            args[1] = &_p1;
            args[2] = &_p2;
            args[3] = &_p3;
            args[4] = &_p4;
            args[5] = &numQubits;
            args[6] = &batchSize;


            LaunchKernel(kernel, (uint)batchSize, DefaultBlockSize, args);
            Synchronize();
            }
    }

    public unsafe void MeasurementForward(IGpuBuffer input, IGpuBuffer output, int batchSize, int stateSize)
    {
        if (!_kernelCache.TryGetValue("measurement_forward", out var kernel))
            throw new InvalidOperationException("HIP kernel not found: measurement_forward");

        uint sharedMemSize = (uint)(DefaultBlockSize * sizeof(float));
            {
            IntPtr _p0 = input.Handle;
            IntPtr _p1 = output.Handle;
            void** args = stackalloc void*[4];
            args[0] = &_p0;
            args[1] = &_p1;
            args[2] = &batchSize;
            args[3] = &stateSize;


            LaunchKernel(kernel, (uint)batchSize, DefaultBlockSize, args, sharedMemSize);
            Synchronize();
            }
    }

    #endregion

    #region RNN (LSTM/GRU) Sequence Operations

    public unsafe void LstmForwardSequence(
        IGpuBuffer input, IGpuBuffer hInit, IGpuBuffer cInit,
        IGpuBuffer weightsIh, IGpuBuffer weightsHh, IGpuBuffer biasIh, IGpuBuffer biasHh,
        IGpuBuffer output, IGpuBuffer hFinal, IGpuBuffer cFinal,
        IGpuBuffer allH, IGpuBuffer allC, IGpuBuffer cacheGates,
        int seqLen, int batch, int inputSize, int hiddenSize)
    {
        if (!_kernelCache.TryGetValue("lstm_forward_sequence", out var kernel))
            throw new InvalidOperationException("HIP kernel not found: lstm_forward_sequence");

        // Validate hiddenSize fits in a block for correct synchronization
        if (hiddenSize > MaxRnnBlockSize)
        {
            throw new InvalidOperationException(
                $"LSTM forward sequence hiddenSize ({hiddenSize}) exceeds max block size ({MaxRnnBlockSize}). " +
                "The kernel requires all hidden units of a batch to fit within a single block for " +
                "correct synchronization. Use smaller hiddenSize or use cell-level LSTM operations.");
        }

        // Grid = one block per batch sample, block = hiddenSize threads per batch
        // This ensures __syncthreads() correctly synchronizes all threads for the same batch
        uint grid = (uint)batch;

        // Kernel signature: input, h_init, c_init, Wi, Wh, biasIh, biasHh, output, h_states, c_states, gates, batch, timeSteps, inputSize, hiddenSize
            {
            IntPtr _p0 = input.Handle;
            IntPtr _p1 = hInit.Handle;
            IntPtr _p2 = cInit.Handle;
            IntPtr _p3 = weightsIh.Handle;
            IntPtr _p4 = weightsHh.Handle;
            IntPtr _p5 = biasIh.Handle;
            IntPtr _p6 = biasHh.Handle;
            IntPtr _p7 = output.Handle;
            IntPtr _p8 = allH.Handle;
            IntPtr _p9 = allC.Handle;
            IntPtr _p10 = cacheGates.Handle;
            void** args = stackalloc void*[15];
            args[0] = &_p0;
            args[1] = &_p1;
            args[2] = &_p2;
            args[3] = &_p3;
            args[4] = &_p4;
            args[5] = &_p5;
            args[6] = &_p6;
            args[7] = &_p7;
            args[8] = &_p8;
            args[9] = &_p9;
            args[10] = &_p10;
            args[11] = &batch;
            args[12] = &seqLen;
            args[13] = &inputSize;
            args[14] = &hiddenSize;


            LaunchKernel(kernel, grid, (uint)hiddenSize, args);
            Synchronize();

            // Copy the last timestep from allH and allC into hFinal and cFinal
            // allH layout: [(seqLen + 1) * batch * hiddenSize] where index 0 is hInit
            // So final hidden state is at index seqLen (last timestep output)
            int finalStateOffset = seqLen * batch * hiddenSize;
            int stateSize = batch * hiddenSize;
            var byteSize = (UIntPtr)(stateSize * sizeof(float));

            // Device-to-device copy from allH[seqLen] to hFinal
            IntPtr srcH = IntPtr.Add(allH.Handle, finalStateOffset * sizeof(float));
            var resultH = HipNativeBindings.hipMemcpy(
                hFinal.Handle,
                srcH,
                byteSize,
                HipMemcpyKind.DeviceToDevice);
            HipNativeBindings.CheckError(resultH, "hipMemcpy D2D (hFinal from allH)");

            // Device-to-device copy from allC[seqLen] to cFinal
            IntPtr srcC = IntPtr.Add(allC.Handle, finalStateOffset * sizeof(float));
            var resultC = HipNativeBindings.hipMemcpy(
                cFinal.Handle,
                srcC,
                byteSize,
                HipMemcpyKind.DeviceToDevice);
            HipNativeBindings.CheckError(resultC, "hipMemcpy D2D (cFinal from allC)");
            }
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
            throw new InvalidOperationException("HIP kernel not found: lstm_backward_sequence");

        // Validate hiddenSize fits in a block for correct synchronization
        if (hiddenSize > MaxRnnBlockSize)
        {
            throw new InvalidOperationException(
                $"LSTM backward sequence hiddenSize ({hiddenSize}) exceeds max block size ({MaxRnnBlockSize}). " +
                "The kernel requires all hidden units of a batch to fit within a single block for " +
                "correct synchronization. Use smaller hiddenSize or use cell-level LSTM operations.");
        }

        // Grid = one block per batch sample, block = hiddenSize threads per batch
        // This ensures __syncthreads() correctly synchronizes all threads for the same batch
        uint grid = (uint)batch;

        // Kernel signature: gradOutput, h_states, c_states, gates, c_init, h_init, input, Wi, Wh,
        //                   gradInput, dWi, dWh, dBiasIh, dBiasHh, dH_init, dC_init, batch, timeSteps, inputSize, hiddenSize
            {
            IntPtr _p0 = gradOutput.Handle;
            IntPtr _p1 = allH.Handle;
            IntPtr _p2 = allC.Handle;
            IntPtr _p3 = cacheGates.Handle;
            IntPtr _p4 = cInit.Handle;
            IntPtr _p5 = hInit.Handle;
            IntPtr _p6 = input.Handle;
            IntPtr _p7 = weightsIh.Handle;
            IntPtr _p8 = weightsHh.Handle;
            IntPtr _p9 = gradInput.Handle;
            IntPtr _p10 = gradWeightsIh.Handle;
            IntPtr _p11 = gradWeightsHh.Handle;
            IntPtr _p12 = gradBiasIh.Handle;
            IntPtr _p13 = gradBiasHh.Handle;
            IntPtr _p14 = gradHInit.Handle;
            IntPtr _p15 = gradCInit.Handle;
            void** args = stackalloc void*[20];
            args[0] = &_p0;
            args[1] = &_p1;
            args[2] = &_p2;
            args[3] = &_p3;
            args[4] = &_p4;
            args[5] = &_p5;
            args[6] = &_p6;
            args[7] = &_p7;
            args[8] = &_p8;
            args[9] = &_p9;
            args[10] = &_p10;
            args[11] = &_p11;
            args[12] = &_p12;
            args[13] = &_p13;
            args[14] = &_p14;
            args[15] = &_p15;
            args[16] = &batch;
            args[17] = &seqLen;
            args[18] = &inputSize;
            args[19] = &hiddenSize;


            LaunchKernel(kernel, grid, (uint)hiddenSize, args);
            Synchronize();
            }
    }

    public unsafe void GruForwardSequence(
        IGpuBuffer input, IGpuBuffer hInit,
        IGpuBuffer weightsIh, IGpuBuffer weightsHh, IGpuBuffer biasIh, IGpuBuffer biasHh,
        IGpuBuffer output, IGpuBuffer hFinal, IGpuBuffer allH, IGpuBuffer cacheGates,
        int seqLen, int batch, int inputSize, int hiddenSize)
    {
        if (!_kernelCache.TryGetValue("gru_forward_sequence", out var kernel))
            throw new InvalidOperationException("HIP kernel not found: gru_forward_sequence");

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

        // Grid = one block per batch sample, block = hiddenSize threads per batch
        // This ensures __syncthreads() correctly synchronizes all threads for the same batch
        uint grid = (uint)batch;

            {
            IntPtr _p0 = input.Handle;
            IntPtr _p1 = hInit.Handle;
            IntPtr _p2 = weightsIh.Handle;
            IntPtr _p3 = weightsHh.Handle;
            IntPtr _p4 = biasIh.Handle;
            IntPtr _p5 = biasHh.Handle;
            IntPtr _p6 = output.Handle;
            IntPtr _p7 = hFinal.Handle;
            IntPtr _p8 = allH.Handle;
            IntPtr _p9 = cacheGates.Handle;
            void** args = stackalloc void*[14];
            args[0] = &_p0;
            args[1] = &_p1;
            args[2] = &_p2;
            args[3] = &_p3;
            args[4] = &_p4;
            args[5] = &_p5;
            args[6] = &_p6;
            args[7] = &_p7;
            args[8] = &_p8;
            args[9] = &_p9;
            args[10] = &seqLen;
            args[11] = &batch;
            args[12] = &inputSize;
            args[13] = &hiddenSize;


            LaunchKernel(kernel, grid, (uint)hiddenSize, args);
            Synchronize();
            }
    }

    public unsafe void GruBackwardSequence(
        IGpuBuffer gradOutput, IGpuBuffer allH, IGpuBuffer cacheGates,
        IGpuBuffer weightsIh, IGpuBuffer weightsHh, IGpuBuffer input,
        IGpuBuffer gradInput, IGpuBuffer gradHInit, IGpuBuffer dHBuffer,
        IGpuBuffer gradWeightsIh, IGpuBuffer gradWeightsHh, IGpuBuffer gradBiasIh, IGpuBuffer gradBiasHh,
        int seqLen, int batch, int inputSize, int hiddenSize)
    {
        if (!_kernelCache.TryGetValue("gru_backward_sequence", out var kernel))
            throw new InvalidOperationException("HIP kernel not found: gru_backward_sequence");

        // Validate hiddenSize fits in a block for correct synchronization
        if (hiddenSize > MaxRnnBlockSize)
        {
            throw new InvalidOperationException(
                $"GRU backward sequence hiddenSize ({hiddenSize}) exceeds max block size ({MaxRnnBlockSize}). " +
                "The kernel requires all hidden units of a batch to fit within a single block for " +
                "correct synchronization. Use smaller hiddenSize or use cell-level GRU operations.");
        }

        // Grid = one block per batch sample, block = hiddenSize threads per batch
        // This ensures __syncthreads() correctly synchronizes all threads for the same batch
        uint grid = (uint)batch;

        // Shared memory size for accumulated hidden gradients (one float per thread)
        uint sharedMemSize = (uint)(DefaultBlockSize * sizeof(float));

            {
            IntPtr _p0 = gradOutput.Handle;
            IntPtr _p1 = allH.Handle;
            IntPtr _p2 = cacheGates.Handle;
            IntPtr _p3 = weightsIh.Handle;
            IntPtr _p4 = weightsHh.Handle;
            IntPtr _p5 = input.Handle;
            IntPtr _p6 = gradInput.Handle;
            IntPtr _p7 = gradHInit.Handle;
            IntPtr _p8 = dHBuffer.Handle;
            IntPtr _p9 = gradWeightsIh.Handle;
            IntPtr _p10 = gradWeightsHh.Handle;
            IntPtr _p11 = gradBiasIh.Handle;
            IntPtr _p12 = gradBiasHh.Handle;
            void** args = stackalloc void*[17];
            args[0] = &_p0;
            args[1] = &_p1;
            args[2] = &_p2;
            args[3] = &_p3;
            args[4] = &_p4;
            args[5] = &_p5;
            args[6] = &_p6;
            args[7] = &_p7;
            args[8] = &_p8;
            args[9] = &_p9;
            args[10] = &_p10;
            args[11] = &_p11;
            args[12] = &_p12;
            args[13] = &seqLen;
            args[14] = &batch;
            args[15] = &inputSize;
            args[16] = &hiddenSize;


            // Use cooperative kernel launch for grid-wide synchronization (grid.sync())
            LaunchCooperativeKernel(kernel, grid, DefaultBlockSize, sharedMemSize, args);
            Synchronize();
            }
    }

    public unsafe void GruCellBackward(
        IGpuBuffer gradH, IGpuBuffer gateR, IGpuBuffer gateZ, IGpuBuffer gateN, IGpuBuffer prevH,
        IGpuBuffer weightsHh,
        IGpuBuffer gradPrevH, IGpuBuffer gradGateR, IGpuBuffer gradGateZ, IGpuBuffer gradGateN,
        int batch, int hiddenSize)
    {
        int totalThreads = batch * hiddenSize;
        uint grid = (uint)((totalThreads + DefaultBlockSize - 1) / DefaultBlockSize);

        // Step 1: Call gru_cell_backward_unified to compute gate gradients and partial gradPrevH
        if (!_kernelCache.TryGetValue("gru_cell_backward_unified", out var cellBackwardKernel))
            throw new InvalidOperationException("HIP kernel not found: gru_cell_backward_unified");

            {
            IntPtr _p0 = gradH.Handle;
            IntPtr _p1 = gateR.Handle;
            IntPtr _p2 = gateZ.Handle;
            IntPtr _p3 = gateN.Handle;
            IntPtr _p4 = prevH.Handle;
            IntPtr _p5 = weightsHh.Handle;
            IntPtr _p6 = gradPrevH.Handle;
            IntPtr _p7 = gradGateR.Handle;
            IntPtr _p8 = gradGateZ.Handle;
            IntPtr _p9 = gradGateN.Handle;
            void** args1 = stackalloc void*[12];
            args1[0] = &_p0;
            args1[1] = &_p1;
            args1[2] = &_p2;
            args1[3] = &_p3;
            args1[4] = &_p4;
            args1[5] = &_p5;
            args1[6] = &_p6;
            args1[7] = &_p7;
            args1[8] = &_p8;
            args1[9] = &_p9;
            args1[10] = &batch;
            args1[11] = &hiddenSize;


            LaunchKernel(cellBackwardKernel, grid, DefaultBlockSize, args1);
            Synchronize();
            }

        // Step 2: Call gru_backward_prevh_unified to compute full gradPrevH using all gate gradients
        if (!_kernelCache.TryGetValue("gru_backward_prevh_unified", out var prevhKernel))
            throw new InvalidOperationException("HIP kernel not found: gru_backward_prevh_unified");

            {
            IntPtr _p0 = gradGateR.Handle;
            IntPtr _p1 = gradGateZ.Handle;
            IntPtr _p2 = gradGateN.Handle;
            IntPtr _p3 = gradH.Handle;
            IntPtr _p4 = gateR.Handle;
            IntPtr _p5 = gateZ.Handle;
            IntPtr _p6 = weightsHh.Handle;
            IntPtr _p7 = gradPrevH.Handle;
            void** args2 = stackalloc void*[10];
            args2[0] = &_p0;
            args2[1] = &_p1;
            args2[2] = &_p2;
            args2[3] = &_p3;
            args2[4] = &_p4;
            args2[5] = &_p5;
            args2[6] = &_p6;
            args2[7] = &_p7;
            args2[8] = &batch;
            args2[9] = &hiddenSize;


            LaunchKernel(prevhKernel, grid, DefaultBlockSize, args2);
            Synchronize();
            }
    }

    #endregion

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;

        // Dispose the default stream wrapper (does not destroy underlying stream)
        _defaultStream?.Dispose();
        _defaultStream = null;
        _bufferPool.Dispose();
        _pinnedPool.Dispose();

        // Unload all kernel modules
        if (_mfmaModule != IntPtr.Zero)
        {
            HipNativeBindings.hipModuleUnload(_mfmaModule);
            _mfmaModule = IntPtr.Zero;
        }
        if (_activationModule != IntPtr.Zero)
        {
            HipNativeBindings.hipModuleUnload(_activationModule);
            _activationModule = IntPtr.Zero;
        }
        if (_neuralNetModule != IntPtr.Zero)
        {
            HipNativeBindings.hipModuleUnload(_neuralNetModule);
            _neuralNetModule = IntPtr.Zero;
        }
        if (_convolutionModule != IntPtr.Zero)
        {
            HipNativeBindings.hipModuleUnload(_convolutionModule);
            _convolutionModule = IntPtr.Zero;
        }
        if (_poolingModule != IntPtr.Zero)
        {
            HipNativeBindings.hipModuleUnload(_poolingModule);
            _poolingModule = IntPtr.Zero;
        }
        if (_normalizationModule != IntPtr.Zero)
        {
            HipNativeBindings.hipModuleUnload(_normalizationModule);
            _normalizationModule = IntPtr.Zero;
        }
        if (_fusedModule != IntPtr.Zero)
        {
            HipNativeBindings.hipModuleUnload(_fusedModule);
            _fusedModule = IntPtr.Zero;
        }
        if (_attentionModule != IntPtr.Zero)
        {
            HipNativeBindings.hipModuleUnload(_attentionModule);
            _attentionModule = IntPtr.Zero;
        }
        if (_fftModule != IntPtr.Zero)
        {
            HipNativeBindings.hipModuleUnload(_fftModule);
            _fftModule = IntPtr.Zero;
        }
        if (_spectralPerfModule != IntPtr.Zero)
        {
            HipNativeBindings.hipModuleUnload(_spectralPerfModule);
            _spectralPerfModule = IntPtr.Zero;
        }
        if (_locallyConnectedModule != IntPtr.Zero)
        {
            HipNativeBindings.hipModuleUnload(_locallyConnectedModule);
            _locallyConnectedModule = IntPtr.Zero;
        }
        if (_deformableConvModule != IntPtr.Zero)
        {
            HipNativeBindings.hipModuleUnload(_deformableConvModule);
            _deformableConvModule = IntPtr.Zero;
        }
        if (_optimizerModule != IntPtr.Zero)
        {
            HipNativeBindings.hipModuleUnload(_optimizerModule);
            _optimizerModule = IntPtr.Zero;
        }
        if (_fusedConvolutionModule != IntPtr.Zero)
        {
            HipNativeBindings.hipModuleUnload(_fusedConvolutionModule);
            _fusedConvolutionModule = IntPtr.Zero;
        }
        if (_sparseModule != IntPtr.Zero)
        {
            // lgtm[cs/call-to-unmanaged-code] HIP interop requires native driver calls.
            HipNativeBindings.hipModuleUnload(_sparseModule);
            _sparseModule = IntPtr.Zero;
        }
        if (_specializedModule != IntPtr.Zero)
        {
            HipNativeBindings.hipModuleUnload(_specializedModule);
            _specializedModule = IntPtr.Zero;
        }
        if (_fp16Module != IntPtr.Zero)
        {
            HipNativeBindings.hipModuleUnload(_fp16Module);
            _fp16Module = IntPtr.Zero;
        }

        if (_lstmModule != IntPtr.Zero)
        {
            HipNativeBindings.hipModuleUnload(_lstmModule);
            _lstmModule = IntPtr.Zero;
        }

        if (_gruModule != IntPtr.Zero)
        {
            HipNativeBindings.hipModuleUnload(_gruModule);
            _gruModule = IntPtr.Zero;
        }

        if (_capsuleModule != IntPtr.Zero)
        {
            HipNativeBindings.hipModuleUnload(_capsuleModule);
            _capsuleModule = IntPtr.Zero;
        }

        if (_snnModule != IntPtr.Zero)
        {
            HipNativeBindings.hipModuleUnload(_snnModule);
            _snnModule = IntPtr.Zero;
        }

        // Unload all additional kernel modules
        foreach (var modField in new[] { _dotProductModule, _reductionModule2, _broadcastModule, _gatedModule, _shapeModule, _lossModule, _softmaxVarModule, _fusedLinearModule, _iouModule, _complexModule })
        {
            if (modField != IntPtr.Zero)
                HipNativeBindings.hipModuleUnload(modField);
        }
        _dotProductModule = _reductionModule2 = _broadcastModule = _gatedModule = _shapeModule = _lossModule = _softmaxVarModule = _fusedLinearModule = _iouModule = _complexModule = IntPtr.Zero;

        if (_hipblasHandle != IntPtr.Zero)
        {
            HipBlasNative.hipblasDestroy(_hipblasHandle); // lgtm[cs/call-to-unmanaged-code] HIP BLAS uses native bindings.
            _hipblasHandle = IntPtr.Zero;
            _hipblasAvailable = false;
        }

        if (_stream != IntPtr.Zero)
        {
            HipNativeBindings.hipStreamDestroy(_stream);
            _stream = IntPtr.Zero;
        }

        _kernelCache.Clear();
    }

    #region Optimizer Operations

    /// <inheritdoc/>
    public unsafe void SgdMomentumUpdate(IGpuBuffer param, IGpuBuffer gradient, IGpuBuffer velocity,
        float learningRate, float momentum, float weightDecay, int size)
    {
        if (!_kernelCache.TryGetValue("sgd_momentum_update", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: sgd_momentum_update");

            {
            IntPtr _p0 = ((HipGpuBuffer)param).Handle;
            IntPtr _p1 = ((HipGpuBuffer)gradient).Handle;
            IntPtr _p2 = ((HipGpuBuffer)velocity).Handle;
            void** args = stackalloc void*[7];
            args[0] = &_p0;
            args[1] = &_p1;
            args[2] = &_p2;
            args[3] = &learningRate;
            args[4] = &momentum;
            args[5] = &weightDecay;
            args[6] = &size;


            uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
            LaunchKernel(krnl, grid, DefaultBlockSize, args);
            Synchronize();
            }
    }

    /// <inheritdoc/>
    public unsafe void SgdUpdate(IGpuBuffer param, IGpuBuffer gradient,
        float learningRate, float weightDecay, int size)
    {
        if (!_kernelCache.TryGetValue("sgd_update", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: sgd_update");

            {
            IntPtr _p0 = ((HipGpuBuffer)param).Handle;
            IntPtr _p1 = ((HipGpuBuffer)gradient).Handle;
            void** args = stackalloc void*[5];
            args[0] = &_p0;
            args[1] = &_p1;
            args[2] = &learningRate;
            args[3] = &weightDecay;
            args[4] = &size;


            uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
            LaunchKernel(krnl, grid, DefaultBlockSize, args);
            Synchronize();
            }
    }

    /// <inheritdoc/>
    public unsafe void AdamUpdate(IGpuBuffer param, IGpuBuffer gradient, IGpuBuffer m, IGpuBuffer v,
        float learningRate, float beta1, float beta2, float epsilon, float weightDecay, int step, int size)
    {
        if (!_kernelCache.TryGetValue("adam_update", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: adam_update");

            {
            IntPtr _p0 = ((HipGpuBuffer)param).Handle;
            IntPtr _p1 = ((HipGpuBuffer)gradient).Handle;
            IntPtr _p2 = ((HipGpuBuffer)m).Handle;
            IntPtr _p3 = ((HipGpuBuffer)v).Handle;
            void** args = stackalloc void*[11];
            args[0] = &_p0;
            args[1] = &_p1;
            args[2] = &_p2;
            args[3] = &_p3;
            args[4] = &learningRate;
            args[5] = &beta1;
            args[6] = &beta2;
            args[7] = &epsilon;
            args[8] = &weightDecay;
            args[9] = &step;
            args[10] = &size;


            uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
            LaunchKernel(krnl, grid, DefaultBlockSize, args);
            Synchronize();
            }
    }

    /// <inheritdoc/>
    public unsafe void AdamWUpdate(IGpuBuffer param, IGpuBuffer gradient, IGpuBuffer m, IGpuBuffer v,
        float learningRate, float beta1, float beta2, float epsilon, float weightDecay, int step, int size)
    {
        if (!_kernelCache.TryGetValue("adamw_update", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: adamw_update");

            {
            IntPtr _p0 = ((HipGpuBuffer)param).Handle;
            IntPtr _p1 = ((HipGpuBuffer)gradient).Handle;
            IntPtr _p2 = ((HipGpuBuffer)m).Handle;
            IntPtr _p3 = ((HipGpuBuffer)v).Handle;
            void** args = stackalloc void*[11];
            args[0] = &_p0;
            args[1] = &_p1;
            args[2] = &_p2;
            args[3] = &_p3;
            args[4] = &learningRate;
            args[5] = &beta1;
            args[6] = &beta2;
            args[7] = &epsilon;
            args[8] = &weightDecay;
            args[9] = &step;
            args[10] = &size;


            uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
            LaunchKernel(krnl, grid, DefaultBlockSize, args);
            Synchronize();
            }
    }

    /// <inheritdoc/>
    public unsafe void RmspropUpdate(IGpuBuffer param, IGpuBuffer gradient, IGpuBuffer squaredAvg,
        float learningRate, float rho, float epsilon, float weightDecay, int size)
    {
        if (!_kernelCache.TryGetValue("rmsprop_update", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: rmsprop_update");

            {
            IntPtr _p0 = ((HipGpuBuffer)param).Handle;
            IntPtr _p1 = ((HipGpuBuffer)gradient).Handle;
            IntPtr _p2 = ((HipGpuBuffer)squaredAvg).Handle;
            void** args = stackalloc void*[8];
            args[0] = &_p0;
            args[1] = &_p1;
            args[2] = &_p2;
            args[3] = &learningRate;
            args[4] = &rho;
            args[5] = &epsilon;
            args[6] = &weightDecay;
            args[7] = &size;


            uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
            LaunchKernel(krnl, grid, DefaultBlockSize, args);
            Synchronize();
            }
    }

    /// <inheritdoc/>
    public unsafe void AdagradUpdate(IGpuBuffer param, IGpuBuffer gradient, IGpuBuffer accumulatedGrad,
        float learningRate, float epsilon, float weightDecay, int size)
    {
        if (!_kernelCache.TryGetValue("adagrad_update", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: adagrad_update");

            {
            IntPtr _p0 = ((HipGpuBuffer)param).Handle;
            IntPtr _p1 = ((HipGpuBuffer)gradient).Handle;
            IntPtr _p2 = ((HipGpuBuffer)accumulatedGrad).Handle;
            void** args = stackalloc void*[7];
            args[0] = &_p0;
            args[1] = &_p1;
            args[2] = &_p2;
            args[3] = &learningRate;
            args[4] = &epsilon;
            args[5] = &weightDecay;
            args[6] = &size;


            uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
            LaunchKernel(krnl, grid, DefaultBlockSize, args);
            Synchronize();
            }
    }

    /// <inheritdoc/>
    public unsafe void NagUpdate(IGpuBuffer param, IGpuBuffer gradient, IGpuBuffer velocity,
        float learningRate, float momentum, float weightDecay, int size)
    {
        if (!_kernelCache.TryGetValue("nag_update", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: nag_update");

            {
            IntPtr _p0 = ((HipGpuBuffer)param).Handle;
            IntPtr _p1 = ((HipGpuBuffer)gradient).Handle;
            IntPtr _p2 = ((HipGpuBuffer)velocity).Handle;
            void** args = stackalloc void*[7];
            args[0] = &_p0;
            args[1] = &_p1;
            args[2] = &_p2;
            args[3] = &learningRate;
            args[4] = &momentum;
            args[5] = &weightDecay;
            args[6] = &size;


            uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
            LaunchKernel(krnl, grid, DefaultBlockSize, args);
            Synchronize();
            }
    }

    /// <inheritdoc/>
    public unsafe void LarsUpdate(IGpuBuffer param, IGpuBuffer gradient, IGpuBuffer velocity,
        float learningRate, float momentum, float weightDecay, float trustCoeff, int size)
    {
        if (!_kernelCache.TryGetValue("lars_update", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: lars_update");

            {
            IntPtr _p0 = ((HipGpuBuffer)param).Handle;
            IntPtr _p1 = ((HipGpuBuffer)gradient).Handle;
            IntPtr _p2 = ((HipGpuBuffer)velocity).Handle;
            void** args = stackalloc void*[8];
            args[0] = &_p0;
            args[1] = &_p1;
            args[2] = &_p2;
            args[3] = &learningRate;
            args[4] = &momentum;
            args[5] = &weightDecay;
            args[6] = &trustCoeff;
            args[7] = &size;


            uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
            LaunchKernel(krnl, grid, DefaultBlockSize, args);
            Synchronize();
            }
    }

    /// <inheritdoc/>
    public unsafe void LambUpdate(IGpuBuffer param, IGpuBuffer gradient, IGpuBuffer m, IGpuBuffer v,
        float learningRate, float beta1, float beta2, float epsilon, float weightDecay, int step, int size)
    {
        if (!_kernelCache.TryGetValue("lamb_update", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: lamb_update");

        float trustRatio = 1.0f; // Default: no layer-wise scaling (degenerates to AdamW)
            {
            IntPtr _p0 = ((HipGpuBuffer)param).Handle;
            IntPtr _p1 = ((HipGpuBuffer)gradient).Handle;
            IntPtr _p2 = ((HipGpuBuffer)m).Handle;
            IntPtr _p3 = ((HipGpuBuffer)v).Handle;
            void** args = stackalloc void*[12];
            args[0] = &_p0;
            args[1] = &_p1;
            args[2] = &_p2;
            args[3] = &_p3;
            args[4] = &learningRate;
            args[5] = &beta1;
            args[6] = &beta2;
            args[7] = &epsilon;
            args[8] = &weightDecay;
            args[9] = &trustRatio;
            args[10] = &step;
            args[11] = &size;


            uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
            LaunchKernel(krnl, grid, DefaultBlockSize, args);
            Synchronize();
            }
    }

    /// <inheritdoc/>
    public unsafe void AdadeltaUpdate(IGpuBuffer param, IGpuBuffer gradient, IGpuBuffer accumGrad, IGpuBuffer accumUpdate,
        float rho, float epsilon, float weightDecay, int size)
    {
        if (!_kernelCache.TryGetValue("adadelta_update", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: adadelta_update");

            {
            IntPtr _p0 = ((HipGpuBuffer)param).Handle;
            IntPtr _p1 = ((HipGpuBuffer)gradient).Handle;
            IntPtr _p2 = ((HipGpuBuffer)accumGrad).Handle;
            IntPtr _p3 = ((HipGpuBuffer)accumUpdate).Handle;
            void** args = stackalloc void*[8];
            args[0] = &_p0;
            args[1] = &_p1;
            args[2] = &_p2;
            args[3] = &_p3;
            args[4] = &rho;
            args[5] = &epsilon;
            args[6] = &weightDecay;
            args[7] = &size;


            uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
            LaunchKernel(krnl, grid, DefaultBlockSize, args);
            Synchronize();
            }
    }

    /// <inheritdoc/>
    public unsafe void AmsgradUpdate(IGpuBuffer param, IGpuBuffer gradient, IGpuBuffer m, IGpuBuffer v, IGpuBuffer vMax,
        float learningRate, float beta1, float beta2, float epsilon, float weightDecay, int step, int size)
    {
        if (!_kernelCache.TryGetValue("amsgrad_update", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: amsgrad_update");

            {
            IntPtr _p0 = ((HipGpuBuffer)param).Handle;
            IntPtr _p1 = ((HipGpuBuffer)gradient).Handle;
            IntPtr _p2 = ((HipGpuBuffer)m).Handle;
            IntPtr _p3 = ((HipGpuBuffer)v).Handle;
            IntPtr _p4 = ((HipGpuBuffer)vMax).Handle;
            void** args = stackalloc void*[12];
            args[0] = &_p0;
            args[1] = &_p1;
            args[2] = &_p2;
            args[3] = &_p3;
            args[4] = &_p4;
            args[5] = &learningRate;
            args[6] = &beta1;
            args[7] = &beta2;
            args[8] = &epsilon;
            args[9] = &weightDecay;
            args[10] = &step;
            args[11] = &size;


            uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
            LaunchKernel(krnl, grid, DefaultBlockSize, args);
            Synchronize();
            }
    }

    /// <inheritdoc/>
    public unsafe void AdamaxUpdate(IGpuBuffer param, IGpuBuffer gradient, IGpuBuffer m, IGpuBuffer u,
        float learningRate, float beta1, float beta2, float epsilon, float weightDecay, int step, int size)
    {
        if (!_kernelCache.TryGetValue("adamax_update", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: adamax_update");

            {
            IntPtr _p0 = ((HipGpuBuffer)param).Handle;
            IntPtr _p1 = ((HipGpuBuffer)gradient).Handle;
            IntPtr _p2 = ((HipGpuBuffer)m).Handle;
            IntPtr _p3 = ((HipGpuBuffer)u).Handle;
            void** args = stackalloc void*[11];
            args[0] = &_p0;
            args[1] = &_p1;
            args[2] = &_p2;
            args[3] = &_p3;
            args[4] = &learningRate;
            args[5] = &beta1;
            args[6] = &beta2;
            args[7] = &epsilon;
            args[8] = &weightDecay;
            args[9] = &step;
            args[10] = &size;


            uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
            LaunchKernel(krnl, grid, DefaultBlockSize, args);
            Synchronize();
            }
    }

    /// <inheritdoc/>
    public unsafe void LionUpdate(IGpuBuffer param, IGpuBuffer gradient, IGpuBuffer m,
        float learningRate, float beta1, float beta2, float weightDecay, int size)
    {
        if (!_kernelCache.TryGetValue("lion_update", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: lion_update");

            {
            IntPtr _p0 = ((HipGpuBuffer)param).Handle;
            IntPtr _p1 = ((HipGpuBuffer)gradient).Handle;
            IntPtr _p2 = ((HipGpuBuffer)m).Handle;
            void** args = stackalloc void*[8];
            args[0] = &_p0;
            args[1] = &_p1;
            args[2] = &_p2;
            args[3] = &learningRate;
            args[4] = &beta1;
            args[5] = &beta2;
            args[6] = &weightDecay;
            args[7] = &size;


            uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
            LaunchKernel(krnl, grid, DefaultBlockSize, args);
            Synchronize();
            }
    }

    /// <inheritdoc/>
    public unsafe void NadamUpdate(IGpuBuffer param, IGpuBuffer gradient, IGpuBuffer m, IGpuBuffer v,
        float learningRate, float beta1, float beta2, float epsilon, float weightDecay, int step, int size)
    {
        if (!_kernelCache.TryGetValue("nadam_update", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: nadam_update");

            {
            IntPtr _p0 = ((HipGpuBuffer)param).Handle;
            IntPtr _p1 = ((HipGpuBuffer)gradient).Handle;
            IntPtr _p2 = ((HipGpuBuffer)m).Handle;
            IntPtr _p3 = ((HipGpuBuffer)v).Handle;
            void** args = stackalloc void*[11];
            args[0] = &_p0;
            args[1] = &_p1;
            args[2] = &_p2;
            args[3] = &_p3;
            args[4] = &learningRate;
            args[5] = &beta1;
            args[6] = &beta2;
            args[7] = &epsilon;
            args[8] = &weightDecay;
            args[9] = &step;
            args[10] = &size;


            uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
            LaunchKernel(krnl, grid, DefaultBlockSize, args);
            Synchronize();
            }
    }

    /// <inheritdoc/>
    public unsafe void FtrlUpdate(IGpuBuffer param, IGpuBuffer gradient, IGpuBuffer z, IGpuBuffer n,
        float learningRate, float l1Reg, float l2Reg, float beta, int size)
    {
        if (!_kernelCache.TryGetValue("ftrl_update", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: ftrl_update");

            {
            IntPtr _p0 = ((HipGpuBuffer)param).Handle;
            IntPtr _p1 = ((HipGpuBuffer)gradient).Handle;
            IntPtr _p2 = ((HipGpuBuffer)z).Handle;
            IntPtr _p3 = ((HipGpuBuffer)n).Handle;
            void** args = stackalloc void*[9];
            args[0] = &_p0;
            args[1] = &_p1;
            args[2] = &_p2;
            args[3] = &_p3;
            args[4] = &learningRate;
            args[5] = &l1Reg;
            args[6] = &l2Reg;
            args[7] = &beta;
            args[8] = &size;


            uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
            LaunchKernel(krnl, grid, DefaultBlockSize, args);
            Synchronize();
            }
    }

    /// <inheritdoc/>
    public unsafe void ConvertToFp16(IGpuBuffer input, IGpuBuffer output, int size)
    {
        if (!_kernelCache.TryGetValue("convert_fp32_to_fp16", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: convert_fp32_to_fp16. FP16 conversion requires a proper GPU kernel.");

            {
            IntPtr _p0 = ((HipGpuBuffer)input).Handle;
            IntPtr _p1 = ((HipGpuBuffer)output).Handle;
            void** args = stackalloc void*[3];
            args[0] = &_p0;
            args[1] = &_p1;
            args[2] = &size;


            uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
            LaunchKernel(krnl, grid, DefaultBlockSize, args);
            Synchronize();
            }
    }

    /// <inheritdoc/>
    public unsafe void ConvertToFp32(IGpuBuffer input, IGpuBuffer output, int size)
    {
        if (!_kernelCache.TryGetValue("convert_fp16_to_fp32", out var krnl))
            throw new InvalidOperationException("HIP kernel not found: convert_fp16_to_fp32. FP32 conversion requires a proper GPU kernel.");

            {
            IntPtr _p0 = ((HipGpuBuffer)input).Handle;
            IntPtr _p1 = ((HipGpuBuffer)output).Handle;
            void** args = stackalloc void*[3];
            args[0] = &_p0;
            args[1] = &_p1;
            args[2] = &size;


            uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
            LaunchKernel(krnl, grid, DefaultBlockSize, args);
            Synchronize();
            }
    }

    #endregion
// Fused kernel dispatch methods    public void ReduceMean(IGpuBuffer i, IGpuBuffer o, int sz) { LaunchFusedAxis("reduce_mean", i, o, 1, sz); }    public unsafe void ClipKernel(IGpuBuffer i, IGpuBuffer o, float mn, float mx, int sz) { if(!_kernelCache.TryGetValue("clip_kernel",out var k))throw new InvalidOperationException("HIP kernel not found: clip_kernel"); using var _=PushContext(); IntPtr ip=i.Handle,op=o.Handle; void** a=stackalloc void*[5]; a[0]=&ip;a[1]=&op;a[2]=&mn;a[3]=&mx;a[4]=&sz; LaunchKernel(k,(uint)((sz+DefaultBlockSize-1)/DefaultBlockSize),DefaultBlockSize,a); }    public void PowScalar(IGpuBuffer i, IGpuBuffer o, float ex, int sz) { LaunchFusedScalar("pow_scalar", i, o, ex, sz); }    public void FracKernel(IGpuBuffer i, IGpuBuffer o, int sz) { LaunchFusedUnary("frac_kernel", i, o, sz); }    public unsafe void EyeKernel(IGpuBuffer o, int n) { if(!_kernelCache.TryGetValue("eye_kernel",out var k))throw new InvalidOperationException("HIP kernel not found: eye_kernel"); using var _=PushContext(); IntPtr op=o.Handle; void** a=stackalloc void*[2]; a[0]=&op;a[1]=&n; LaunchKernel(k,(uint)((n*n+DefaultBlockSize-1)/DefaultBlockSize),DefaultBlockSize,a); }    public unsafe void OneHotKernel(IGpuBuffer idx, IGpuBuffer o, int bs, int nc) { if(!_kernelCache.TryGetValue("one_hot_kernel",out var k))throw new InvalidOperationException("HIP kernel not found: one_hot_kernel"); using var _=PushContext(); IntPtr ip=idx.Handle,op=o.Handle; void** a=stackalloc void*[4]; a[0]=&ip;a[1]=&op;a[2]=&bs;a[3]=&nc; LaunchKernel(k,(uint)((bs*nc+DefaultBlockSize-1)/DefaultBlockSize),DefaultBlockSize,a); }    public unsafe void MaskedFillKernel(IGpuBuffer i, IGpuBuffer m, IGpuBuffer o, float fv, int sz) { if(!_kernelCache.TryGetValue("masked_fill_kernel",out var k))throw new InvalidOperationException("HIP kernel not found: masked_fill_kernel"); using var _=PushContext(); IntPtr ip=i.Handle,mp=m.Handle,op=o.Handle; void** a=stackalloc void*[5]; a[0]=&ip;a[1]=&mp;a[2]=&op;a[3]=&fv;a[4]=&sz; LaunchKernel(k,(uint)((sz+DefaultBlockSize-1)/DefaultBlockSize),DefaultBlockSize,a); }    public void EqualsKernel(IGpuBuffer a1, IGpuBuffer b1, IGpuBuffer o, int sz) { LaunchFusedBinary("equals_kernel", a1, b1, o, sz); }    public void NotEqualsKernel(IGpuBuffer a1, IGpuBuffer b1, IGpuBuffer o, int sz) { LaunchFusedBinary("not_equals_kernel", a1, b1, o, sz); }    public unsafe void OuterProduct(IGpuBuffer a1, IGpuBuffer b1, IGpuBuffer o, int M, int N) { if(!_kernelCache.TryGetValue("outer_product",out var k))throw new InvalidOperationException("HIP kernel not found: outer_product"); using var _=PushContext(); IntPtr ap=a1.Handle,bp=b1.Handle,op=o.Handle; void** a=stackalloc void*[5]; a[0]=&ap;a[1]=&bp;a[2]=&op;a[3]=&M;a[4]=&N; LaunchKernel(k,(uint)((M*N+DefaultBlockSize-1)/DefaultBlockSize),DefaultBlockSize,a); }    public unsafe void BatchDotProduct(IGpuBuffer a1, IGpuBuffer b1, IGpuBuffer o, int bs, int dim) { if(!_kernelCache.TryGetValue("batch_dot_product",out var k))throw new InvalidOperationException("HIP kernel not found: batch_dot_product"); using var _=PushContext(); IntPtr ap=a1.Handle,bp=b1.Handle,op=o.Handle; void** a=stackalloc void*[5]; a[0]=&ap;a[1]=&bp;a[2]=&op;a[3]=&bs;a[4]=&dim; LaunchKernel(k,(uint)((bs+DefaultBlockSize-1)/DefaultBlockSize),DefaultBlockSize,a); }    public void GluForward(IGpuBuffer i, IGpuBuffer o, int os, int hd) { LaunchFusedAxis("glu_forward", i, o, os, hd); }    public void GeGluForward(IGpuBuffer i, IGpuBuffer o, int os, int hd) { LaunchFusedAxis("geglu_forward", i, o, os, hd); }    public void ReGluForward(IGpuBuffer i, IGpuBuffer o, int os, int hd) { LaunchFusedAxis("reglu_forward", i, o, os, hd); }    public void SwiGluForward(IGpuBuffer i, IGpuBuffer o, int os, int hd) { LaunchFusedAxis("swiglu_forward", i, o, os, hd); }    public void BceLoss(IGpuBuffer p, IGpuBuffer t, IGpuBuffer l, int sz) { LaunchFusedBinary("bce_loss", p, t, l, sz); }

    /// <inheritdoc/>
    public void SpectralFilter(IGpuBuffer inputReal, IGpuBuffer filterReal, IGpuBuffer filterImag,
        IGpuBuffer outputReal, int batch, int height, int width, int filterSliceCount)
    {
        if (batch <= 0 || height <= 0 || width <= 0) return;
        if (filterSliceCount <= 0 || (filterSliceCount != 1 && filterSliceCount != batch))
            throw new ArgumentException($"filterSliceCount must be 1 (shared) or batch ({batch}). Got {filterSliceCount}.");

        int sliceSize = height * width;
        int totalSize = batch * sliceSize;

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
            throw new InvalidOperationException("HIP kernel not found: atan2_elementwise");
        IntPtr ip = imag.Handle, rp = real.Handle, op = output.Handle;
        void** args = stackalloc void*[4];
        args[0] = &ip; args[1] = &rp; args[2] = &op; args[3] = &n;
        LaunchKernel(kernel, (uint)((n + DefaultBlockSize - 1) / DefaultBlockSize), DefaultBlockSize, args);
    }

    /// <inheritdoc/>
    public unsafe void NormalizeRowsFused(IGpuBuffer input, IGpuBuffer output, int rows, int cols)
    {
        if (rows <= 0 || cols <= 0) return;
        if (!_kernelCache.TryGetValue("normalize_rows_fused", out var kernel))
            throw new InvalidOperationException("HIP kernel not found: normalize_rows_fused");
        IntPtr ip = input.Handle, op = output.Handle;
        void** args = stackalloc void*[4];
        args[0] = &ip; args[1] = &op; args[2] = &rows; args[3] = &cols;
        // The kernel uses a tree reduction that requires a power-of-two threadgroup size.
        uint block = 32;
        uint cap = (uint)Math.Min(256, cols);
        while (block * 2 <= cap) block *= 2;
        LaunchKernelWithSharedMem(kernel, (uint)rows, block, block * sizeof(float),
            new IntPtr[] { (IntPtr)args[0], (IntPtr)args[1], (IntPtr)args[2], (IntPtr)args[3] });
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
            throw new InvalidOperationException("HIP kernel not found: analytic_signal_mask");
        IntPtr srP = specReal.Handle, siP = specImag.Handle, orP = outReal.Handle, oiP = outImag.Handle;
        void** args = stackalloc void*[8];
        args[0] = &srP; args[1] = &siP; args[2] = &orP; args[3] = &oiP;
        args[4] = &batch; args[5] = &fftSize; args[6] = &binLow; args[7] = &binHigh;
        LaunchKernel(kernel, (uint)((total + DefaultBlockSize - 1) / DefaultBlockSize), DefaultBlockSize, args);
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
            throw new InvalidOperationException("HIP kernel not found: bispectrum_gather");
        IntPtr srP = specReal.Handle, siP = specImag.Handle, orP = outReal.Handle, oiP = outImag.Handle;
        void** args = stackalloc void*[6];
        args[0] = &srP; args[1] = &siP; args[2] = &orP; args[3] = &oiP;
        args[4] = &maxF1; args[5] = &maxF2;
        LaunchKernel(kernel, (uint)((total + DefaultBlockSize - 1) / DefaultBlockSize), DefaultBlockSize, args);
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
            throw new InvalidOperationException("HIP kernel not found: trispectrum_gather");
        IntPtr srP = specReal.Handle, siP = specImag.Handle, orP = outReal.Handle, oiP = outImag.Handle;
        void** args = stackalloc void*[7];
        args[0] = &srP; args[1] = &siP; args[2] = &orP; args[3] = &oiP;
        args[4] = &maxF1; args[5] = &maxF2; args[6] = &maxF3;
        LaunchKernel(kernel, (uint)((total + DefaultBlockSize - 1) / DefaultBlockSize), DefaultBlockSize, args);
    }

    /// <inheritdoc/>
    public unsafe void CavityBounceInplace(IGpuBuffer workReal, IGpuBuffer workImag, int total, float invN)
    {
        if (total <= 0) return;
        if (!_kernelCache.TryGetValue("cavity_bounce_inplace", out var kernel))
            throw new InvalidOperationException("HIP kernel not found: cavity_bounce_inplace");
        IntPtr wr = workReal.Handle, wi = workImag.Handle;
        void** args = stackalloc void*[4];
        args[0] = &wr; args[1] = &wi; args[2] = &total; args[3] = &invN;
        LaunchKernel(kernel, (uint)((total + DefaultBlockSize - 1) / DefaultBlockSize), DefaultBlockSize, args);
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
            throw new InvalidOperationException("HIP kernel not found: wideband_log_bin_pool");
        IntPtr mp = magBuf.Handle, op = output.Handle;
        void** args = stackalloc void*[6];
        args[0] = &mp; args[1] = &op;
        args[2] = &totalSegBatch; args[3] = &fftSize; args[4] = &numBins; args[5] = &usable;
        LaunchKernel(kernel, (uint)((total + DefaultBlockSize - 1) / DefaultBlockSize), DefaultBlockSize, args);
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
            throw new InvalidOperationException("HIP kernel not found: mel_filterbank_apply");
        IntPtr ps = powerSpec.Handle, mf = melFilters.Handle, me = melEnergy.Handle;
        void** args = stackalloc void*[6];
        args[0] = &ps; args[1] = &mf; args[2] = &me;
        args[3] = &totalSegBatch; args[4] = &specBins; args[5] = &melBins;
        LaunchKernel(kernel, (uint)((total + DefaultBlockSize - 1) / DefaultBlockSize), DefaultBlockSize, args);
    }

    /// <inheritdoc/>
    public unsafe void MfccLog1p(IGpuBuffer input, IGpuBuffer output, int n)
    {
        if (n <= 0) return;
        if (!_kernelCache.TryGetValue("mfcc_log1p", out var kernel))
            throw new InvalidOperationException("HIP kernel not found: mfcc_log1p");
        IntPtr ip = input.Handle, op = output.Handle;
        void** args = stackalloc void*[3];
        args[0] = &ip; args[1] = &op; args[2] = &n;
        LaunchKernel(kernel, (uint)((n + DefaultBlockSize - 1) / DefaultBlockSize), DefaultBlockSize, args);
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
        if (!_kernelCache.TryGetValue("pac_phase_bin_mi", out var kernel))
            throw new InvalidOperationException("HIP kernel not found: pac_phase_bin_mi");
        IntPtr tp = thetaPhase.Handle, ga = gammaAmp.Handle, op = output.Handle;
        void** args = stackalloc void*[7];
        args[0] = &tp; args[1] = &ga; args[2] = &op;
        args[3] = &batch; args[4] = &numSamples; args[5] = &numGammaBands; args[6] = &gammaIdx;
        LaunchKernelWithSharedMem(kernel, (uint)batch, 256, (uint)(2 * 18 * sizeof(float)),
            new IntPtr[] { (IntPtr)args[0], (IntPtr)args[1], (IntPtr)args[2], (IntPtr)args[3], (IntPtr)args[4], (IntPtr)args[5], (IntPtr)args[6] });
    }
}

/// <summary>
/// HIP GPU buffer wrapper implementing IGpuBuffer.
/// </summary>
internal sealed class HipGpuBuffer : IGpuBuffer, IPoolableGpuBuffer
{
    public IntPtr Handle { get; }
    public int Size { get; }
    public long SizeInBytes => Size * sizeof(float);
    private readonly Action<HipGpuBuffer>? _returnToPool;
    private int _poolState;

    public HipGpuBuffer(IntPtr handle, int size, Action<HipGpuBuffer>? returnToPool = null)
    {
        Handle = handle;
        Size = size;
        _returnToPool = returnToPool;
    }

    public void MarkRented()
    {
        Interlocked.Exchange(ref _poolState, 0);
    }

    public void Release()
    {
        if (Interlocked.Exchange(ref _poolState, 2) == 2)
            return;

        if (Handle != IntPtr.Zero)
        {
            var result = HipNativeBindings.hipFree(Handle);
            if (result != HipError.Success)
            {
                System.Diagnostics.Debug.WriteLine($"hipFree warning: {result}");
            }
        }
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
}

/// <summary>
/// HIP GPU byte buffer wrapper implementing IGpuBuffer.
/// Used for sparse matrix indices (1 byte per group of 4 elements).
/// </summary>
internal sealed class HipGpuByteBuffer : IGpuBuffer
{
    public IntPtr Handle { get; }
    public int Size { get; }
    public long SizeInBytes => Size;
    private bool _disposed;

    public HipGpuByteBuffer(IntPtr handle, int size)
    {
        Handle = handle;
        Size = size;
    }

    public void Dispose()
    {
        if (_disposed) return;

        if (Handle != IntPtr.Zero)
        {
            var result = HipNativeBindings.hipFree(Handle);
            if (result != HipError.Success)
            {
                System.Diagnostics.Debug.WriteLine($"hipFree (byte buffer) warning: {result}");
            }
        }

        _disposed = true;
    }
}
