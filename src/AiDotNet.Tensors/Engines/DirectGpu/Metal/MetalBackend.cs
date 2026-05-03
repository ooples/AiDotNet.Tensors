// Copyright (c) AiDotNet. All rights reserved.
// Metal GPU backend for Apple Silicon - IDirectGpuBackend implementation.

using System;
using System.Collections.Concurrent;
using System.Runtime.InteropServices;
using static AiDotNet.Tensors.Engines.DirectGpu.Metal.MetalNativeBindings;

namespace AiDotNet.Tensors.Engines.DirectGpu.Metal;

/// <summary>
/// Metal GPU backend implementation for Apple Silicon (M1, M2, M3, M4 chips).
/// </summary>
/// <remarks>
/// <para><b>Key Features:</b></para>
/// <list type="bullet">
/// <item>Unified memory architecture - zero-copy CPU/GPU data sharing</item>
/// <item>Metal Performance Shaders (MPS) for optimized BLAS/neural network ops</item>
/// <item>Apple GPU Family 7-9 feature support</item>
/// <item>Efficient compute shader dispatch with optimal threadgroup sizing</item>
/// </list>
/// <para><b>Performance Notes:</b></para>
/// <para>
/// On Apple Silicon, the unified memory eliminates PCIe transfer overhead,
/// making Metal competitive with discrete GPUs for memory-bound workloads.
/// The MPS library provides hand-tuned implementations of common operations.
/// </para>
/// </remarks>
public sealed partial class MetalBackend : IDirectGpuBackend
{
    private readonly MetalDevice _device;
    private readonly MetalCommandQueue _commandQueue;
    private readonly MetalShaderLibrary _shaderLibrary;
    private bool _disposed;
    private readonly object _lock = new();

    // Cached pipeline states for common operations
    private readonly ConcurrentDictionary<string, MetalPipelineState> _pipelineCache = new();

    // Pre-compiled kernel libraries
    private IntPtr _elementWiseLibrary;
    private IntPtr _activationLibrary;
    private IntPtr _trigLibrary;
    private IntPtr _reductionLibrary;
    private IntPtr _matrixLibrary;
    private IntPtr _normalizationLibrary;
    private IntPtr _convolutionLibrary;
    private IntPtr _attentionLibrary;
    private IntPtr _optimizerLibrary;
    private IntPtr _lossLibrary;
    private IntPtr _comparisonLibrary;
    private IntPtr _randomLibrary;
    private IntPtr _dotProductLibrary;
    private IntPtr _fusedLinearLibrary;
    private IntPtr _iouLibrary;
    private IntPtr _hyperbolicLibrary;
    private IntPtr _octonionLibrary;
    private IntPtr _spectralPerfLibrary;
    private IntPtr _parity210Library;
    private IntPtr _linalgLibrary;
    private IntPtr _fftLibrary;
    private IntPtr _detectionLibrary;
    private IntPtr _geometryLibrary;
    private IntPtr _roiLibrary;
    private IntPtr _audioLibrary;

    #region Properties

    /// <summary>
    /// Gets whether this backend is available and initialized.
    /// </summary>
    public bool IsAvailable => _device?.IsAvailable == true && !_disposed;

    /// <summary>
    /// Gets the backend name.
    /// </summary>
    public string BackendName => "Metal";
    public TensorDevice DeviceType => TensorDevice.Metal;

    /// <summary>
    /// Gets the GPU device name.
    /// </summary>
    public string DeviceName => _device?.DeviceName ?? "Not Available";

    /// <summary>
    /// Gets the GPU vendor.
    /// </summary>
    public string DeviceVendor => "Apple";

    /// <summary>
    /// Gets the number of compute units (GPU cores).
    /// </summary>
    public int ComputeUnits => _device?.EstimatedGPUCores ?? 0;

    /// <summary>
    /// Gets the total global memory in bytes.
    /// </summary>
    public long GlobalMemoryBytes => (long)(_device?.RecommendedMaxWorkingSetSize ?? 0);

    /// <summary>
    /// Gets the local (shared) memory per workgroup in bytes.
    /// </summary>
    public long LocalMemoryBytes => (long)(_device?.MaxThreadgroupMemoryLength ?? 32768);

    /// <summary>
    /// Issue #285: per-allocation cap from <c>MTLDevice.maxBufferLength</c>.
    /// This is the hard limit Metal enforces on a single
    /// <c>newBufferWithLength:</c> call; it can be substantially smaller
    /// than <see cref="GlobalMemoryBytes"/> (recommendedMaxWorkingSetSize)
    /// on Apple Silicon and discrete-GPU Macs alike. NOTE: a request
    /// smaller than this cap that still fails (e.g. due to fragmentation
    /// or transient memory pressure) returns nil from
    /// <c>newBufferWithLength:</c>, which our wrapper surfaces as a
    /// generic <c>NullReferenceException</c> — translating that into
    /// <see cref="GpuBufferTooLargeException"/> is follow-up work.
    /// </summary>
    public long MaxBufferAllocBytes => (long)(_device?.MaxBufferLength ?? 0);

    public double TheoreticalGflops { get; }

    #endregion

    #region Constructor and Initialization

    /// <summary>
    /// Creates a new Metal backend using the default device.
    /// </summary>
    public MetalBackend() : this(0)
    {
    }

    /// <summary>
    /// Creates a new Metal backend for the specified device index.
    /// </summary>
    /// <param name="deviceIndex">Device index (0 = default device).</param>
    public MetalBackend(int deviceIndex)
    {
        if (!IsPlatformSupported)
        {
            throw new PlatformNotSupportedException("Metal is only available on macOS/iOS");
        }

        _device = new MetalDevice(deviceIndex);
        if (!_device.IsAvailable)
        {
            throw new InvalidOperationException("Failed to initialize Metal device");
        }

        _commandQueue = new MetalCommandQueue(_device);
        _shaderLibrary = new MetalShaderLibrary(_device);

        InitializeKernelLibraries();
    }

    /// <summary>
    /// Initializes pre-compiled kernel libraries for common operations.
    /// </summary>
    private void InitializeKernelLibraries()
    {
        try
        {
            // Compile element-wise operations
            _elementWiseLibrary = _shaderLibrary.CompileLibrary("ElementWise", MetalKernels.ElementWiseKernels);

            // Compile activation functions
            _activationLibrary = _shaderLibrary.CompileLibrary("Activation", MetalKernels.ActivationKernels);

            // Compile trigonometric functions
            _trigLibrary = _shaderLibrary.CompileLibrary("Trig", MetalKernels.TrigKernels);

            // Compile reduction operations
            _reductionLibrary = _shaderLibrary.CompileLibrary("Reduction", MetalKernels.ReductionKernels);

            // Compile matrix operations
            _matrixLibrary = _shaderLibrary.CompileLibrary("Matrix", MetalKernels.MatrixKernels);

            // Compile normalization operations
            _normalizationLibrary = _shaderLibrary.CompileLibrary("Normalization", MetalKernels.NormalizationKernels);

            // Compile convolution operations
            _convolutionLibrary = _shaderLibrary.CompileLibrary("Convolution", MetalKernels.ConvolutionKernels);

            // Compile attention operations
            _attentionLibrary = _shaderLibrary.CompileLibrary("Attention", MetalKernels.AttentionKernels);

            // Compile optimizer operations
            _optimizerLibrary = _shaderLibrary.CompileLibrary("Optimizer", MetalKernels.OptimizerKernels);

            // Compile loss function operations
            _lossLibrary = _shaderLibrary.CompileLibrary("Loss", MetalKernels.LossKernels);

            // Compile comparison operations
            _comparisonLibrary = _shaderLibrary.CompileLibrary("Comparison", MetalKernels.ComparisonKernels);

            // Compile random operations
            _randomLibrary = _shaderLibrary.CompileLibrary("Random", MetalKernels.RandomKernels);

            // Compile dot product operations
            _dotProductLibrary = _shaderLibrary.CompileLibrary("DotProduct", MetalKernels.DotProductKernels);

            // Compile hyperbolic geometry operations
            _hyperbolicLibrary = _shaderLibrary.CompileLibrary("Hyperbolic", MetalKernels.HyperbolicKernels);

            // Compile octonion algebra operations
            _octonionLibrary = _shaderLibrary.CompileLibrary("Octonion", MetalKernels.OctonionKernels);
        }
        catch (Exception ex)
        {
            // Log but don't fail - kernels will be compiled on-demand
            System.Diagnostics.Debug.WriteLine($"Metal kernel pre-compilation warning: {ex.Message}");
        }

        // Compile Issue #160 spectral perf kernels independently so a failure here does not
        // prevent the following (unrelated) libraries from being compiled.
        try
        {
            _spectralPerfLibrary = _shaderLibrary.CompileLibrary("SpectralPerf", MetalKernels.SpectralPerfKernels);
        }
        catch (Exception ex)
        {
            System.Diagnostics.Debug.WriteLine($"Metal SpectralPerf pre-compilation warning: {ex.Message}");
        }

        try
        {
            _fusedLinearLibrary = _shaderLibrary.CompileLibrary("FusedLinear", MetalKernels.FusedLinearKernels);
        }
        catch (Exception ex)
        {
            System.Diagnostics.Debug.WriteLine($"Metal FusedLinear pre-compilation warning: {ex.Message}");
        }

        try
        {
            _iouLibrary = _shaderLibrary.CompileLibrary("IoULoss", MetalKernels.IoULossKernels);
        }
        catch (Exception ex)
        {
            System.Diagnostics.Debug.WriteLine($"Metal IoULoss pre-compilation warning: {ex.Message}");
        }

        // Parity-210 hot-path kernels — same function surface as CUDA/HIP.
        try
        {
            _parity210Library = _shaderLibrary.CompileLibrary("Parity210",
                MetalParity210Kernels.Source);
        }
        catch (Exception ex)
        {
            // Parity-210 library is optional; CPU fallback via CpuEngine inheritance stays intact.
            System.Diagnostics.Debug.WriteLine($"Metal Parity-210 pre-compilation warning: {ex.Message}");
            _parity210Library = IntPtr.Zero;
        }

        // Linalg decomposition kernels (#211 moat #2).
        try
        {
            _linalgLibrary = _shaderLibrary.CompileLibrary("Linalg",
                MetalLinalgKernels.Source);
        }
        catch (Exception ex)
        {
            System.Diagnostics.Debug.WriteLine($"Metal Linalg pre-compilation warning: {ex.Message}");
            _linalgLibrary = IntPtr.Zero;
        }

        // Vision detection kernels (#217). IDetectionBackend dispatch.
        try
        {
            _detectionLibrary = _shaderLibrary.CompileLibrary("Detection",
                MetalDetectionKernels.Source);
        }
        catch (Exception ex)
        {
            System.Diagnostics.Debug.WriteLine($"Metal Detection pre-compilation warning: {ex.Message}");
            _detectionLibrary = IntPtr.Zero;
        }

        // Geometry / sampling kernels (#217). IGeometryBackend dispatch.
        try
        {
            _geometryLibrary = _shaderLibrary.CompileLibrary("Geometry",
                MetalGeometryKernels.Source);
        }
        catch (Exception ex)
        {
            System.Diagnostics.Debug.WriteLine($"Metal Geometry pre-compilation warning: {ex.Message}");
            _geometryLibrary = IntPtr.Zero;
        }

        // RoI family (#217 tail). IRoiBackend dispatch.
        try
        {
            _roiLibrary = _shaderLibrary.CompileLibrary("Roi", MetalRoiKernels.Source);
        }
        catch (Exception ex)
        {
            System.Diagnostics.Debug.WriteLine($"Metal RoI pre-compilation warning: {ex.Message}");
            _roiLibrary = IntPtr.Zero;
        }

        // Audio primitives (#217 tail). IAudioBackend dispatch.
        try
        {
            _audioLibrary = _shaderLibrary.CompileLibrary("Audio", MetalAudioKernels.Source);
        }
        catch (Exception ex)
        {
            System.Diagnostics.Debug.WriteLine($"Metal Audio pre-compilation warning: {ex.Message}");
            _audioLibrary = IntPtr.Zero;
        }

        // Parity-212 FFT kernels — custom radix-2 Cooley-Tukey (no external
        // FFT library). Optional in the same sense as Parity-210: if the
        // library fails to compile, IFftBackend.LaunchFft throws and the
        // Fft module falls back to CPU.
        try
        {
            _fftLibrary = _shaderLibrary.CompileLibrary("Fft212", MetalFftKernels.Source);
        }
        catch (Exception ex)
        {
            System.Diagnostics.Debug.WriteLine($"Metal FFT pre-compilation warning: {ex.Message}");
            _fftLibrary = IntPtr.Zero;
        }
    }

    /// <summary>
    /// Gets or creates a pipeline state for a kernel.
    /// </summary>
    private MetalPipelineState GetPipeline(string libraryName, IntPtr library, string kernelName)
    {
        var cacheKey = $"{libraryName}::{kernelName}";

        return _pipelineCache.GetOrAdd(cacheKey, _ =>
        {
            var function = _shaderLibrary.GetFunction(library, kernelName);
            try
            {
                return _shaderLibrary.CreatePipelineState(function);
            }
            finally
            {
                Release(function);
            }
        });
    }

    #endregion

    #region Memory Management

    /// <summary>
    /// Allocates a GPU buffer and uploads data.
    /// </summary>
    public IGpuBuffer AllocateBuffer(float[] data)
    {
        ThrowIfDisposed();

        if (data is null || data.Length == 0)
        {
            throw new ArgumentException("Data cannot be null or empty", nameof(data));
        }
        // Issue #285: per-allocation cap check.
        GpuBufferSizeGuard.EnsureFits("Metal", (long)data.Length * sizeof(float), MaxBufferAllocBytes, DeviceName);

        return new MetalGpuBuffer(_device, data);
    }

    /// <summary>
    /// Allocates an empty GPU buffer.
    /// </summary>
    public IGpuBuffer AllocateBuffer(int size)
    {
        ThrowIfDisposed();

        if (size <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(size), "Size must be positive");
        }
        // Issue #285: per-allocation cap check.
        GpuBufferSizeGuard.EnsureFits("Metal", (long)size * sizeof(float), MaxBufferAllocBytes, DeviceName);

        return new MetalGpuBuffer(_device, size);
    }

    /// <summary>
    /// Downloads GPU buffer contents to CPU.
    /// </summary>
    public float[] DownloadBuffer(IGpuBuffer buffer)
    {
        ThrowIfDisposed();

        if (buffer is not MetalGpuBuffer metalBuffer)
        {
            throw new ArgumentException("Buffer must be a MetalGpuBuffer", nameof(buffer));
        }

        var result = new float[buffer.Size];
        metalBuffer.CopyTo(result);
        return result;
    }

    /// <summary>
    /// Downloads GPU buffer contents to existing CPU array.
    /// </summary>
    public void DownloadBuffer(IGpuBuffer buffer, float[] destination)
    {
        ThrowIfDisposed();

        if (buffer is not MetalGpuBuffer metalBuffer)
        {
            throw new ArgumentException("Buffer must be a MetalGpuBuffer", nameof(buffer));
        }

        metalBuffer.CopyTo(destination);
    }

    /// <summary>
    /// Uploads CPU array data to a GPU buffer.
    /// </summary>
    public void UploadToBuffer(IGpuBuffer buffer, float[] source)
    {
        ThrowIfDisposed();

        if (buffer is not MetalGpuBuffer metalBuffer)
        {
            throw new ArgumentException("Buffer must be a MetalGpuBuffer", nameof(buffer));
        }

        metalBuffer.CopyFrom(source);
    }

    /// <summary>
    /// Copies data between GPU buffers.
    /// </summary>
    public void Copy(IGpuBuffer source, int srcOffset, IGpuBuffer destination, int destOffset, int size)
    {
        ThrowIfDisposed();

        if (source is not MetalGpuBuffer srcBuffer || destination is not MetalGpuBuffer destBuffer)
        {
            throw new ArgumentException("Buffers must be MetalGpuBuffer");
        }

        // Use element-wise copy kernel
        var pipeline = GetPipeline("ElementWise", _elementWiseLibrary, "copy_buffer");
        var (threadgroups, threadsPerGroup) = pipeline.Calculate1DDispatch(size);

        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        encoder.SetBuffer(srcBuffer, 0);
        encoder.SetBuffer(destBuffer, 1);
        encoder.SetBytes((uint)size, 2);
        encoder.DispatchThreadgroups(threadgroups, threadsPerGroup);
    }

    /// <summary>
    /// Allocates a byte buffer for sparse indices.
    /// </summary>
    public IGpuBuffer AllocateByteBuffer(int size)
    {
        ThrowIfDisposed();

        if (size <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(size), "Size must be positive");
        }

        // For byte buffers, we allocate as floats and use size/4. Round up
        // in long-space first (size could be near int.MaxValue and `size + 3`
        // would overflow if done in int), then check the actual allocation
        // bytes against the cap.
        long floatSizeLong = ((long)size + 3) / 4;
        long actualBytes = floatSizeLong * sizeof(float);
        GpuBufferSizeGuard.EnsureFits("Metal", actualBytes, MaxBufferAllocBytes, DeviceName);
        // After the cap check, the float-count is guaranteed to fit in int
        // (because actualBytes <= MaxBufferAllocBytes <= long.MaxValue and
        // floatSizeLong * 4 <= int.MaxValue when actualBytes is reasonable).
        return new MetalGpuBuffer(_device, checked((int)floatSizeLong));
    }

    /// <summary>
    /// Allocates an integer buffer.
    /// </summary>
    public IGpuBuffer AllocateIntBuffer(int size)
    {
        ThrowIfDisposed();

        if (size <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(size), "Size must be positive");
        }
        GpuBufferSizeGuard.EnsureFits("Metal", (long)size * sizeof(int), MaxBufferAllocBytes, DeviceName);

        // Metal can store int32 in the same buffer format as float32
        return new MetalGpuBuffer(_device, size);
    }

    /// <summary>
    /// Allocates an integer buffer with initial data.
    /// </summary>
    public IGpuBuffer AllocateIntBuffer(int[] data)
    {
        ThrowIfDisposed();

        if (data is null || data.Length == 0)
        {
            throw new ArgumentException("Data cannot be null or empty", nameof(data));
        }
        GpuBufferSizeGuard.EnsureFits("Metal", (long)data.Length * sizeof(int), MaxBufferAllocBytes, DeviceName);

        // Convert int array to float array for buffer allocation
        var floatData = new float[data.Length];
        for (int i = 0; i < data.Length; i++)
        {
            floatData[i] = Int32BitsToSingleCompat(data[i]);
        }

        return new MetalGpuBuffer(_device, floatData);
    }

    /// <summary>
    /// .NET Framework compatible implementation of BitConverter.Int32BitsToSingle.
    /// </summary>
    private static unsafe float Int32BitsToSingleCompat(int value)
    {
        return *(float*)&value;
    }

    /// <summary>
    /// .NET Framework compatible implementation of BitConverter.SingleToInt32Bits.
    /// </summary>
    private static unsafe int SingleToInt32BitsCompat(float value)
    {
        return *(int*)&value;
    }

    #endregion

    #region GEMM Operations

    /// <summary>
    /// General matrix multiplication: C = alpha * A * B + beta * C
    /// </summary>
    /// <inheritdoc/>
    public void MatMulTransposed(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int M, int N, int K, float alpha = 1.0f, float beta = 0.0f)
    {
        ThrowIfDisposed();
        // The matmul_tiled_transposed kernel writes C unconditionally
        // (no β·C blend). Fail fast on β != 0 rather than silently
        // discarding the existing C contents.
        if (Math.Abs(beta) > 1e-7f)
            throw new NotSupportedException(
                "Metal MatMulTransposed currently supports only beta == 0. " +
                "For beta != 0, route through TensorMatMul(A, Transpose(B)).");
        if (A is not MetalGpuBuffer aBuffer || B is not MetalGpuBuffer bBuffer || C is not MetalGpuBuffer cBuffer)
            throw new ArgumentException("Buffers must be MetalGpuBuffer");

        // Custom tiled kernel that reads B with the [N, K] → Bᵀ index pattern.
        // Skips the materialized transpose copy that the generic Gemm + a
        // separate transpose dispatch would do.
        var pipeline = GetPipeline("Matrix", _matrixLibrary, "matmul_tiled_transposed");
        var (threadgroups, threadsPerGroup) = pipeline.Calculate2DDispatch(N, M);

        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        encoder.SetBuffer(aBuffer, 0);
        encoder.SetBuffer(bBuffer, 1);
        encoder.SetBuffer(cBuffer, 2);
        encoder.SetBytes((uint)M, 3);
        encoder.SetBytes((uint)N, 4);
        encoder.SetBytes((uint)K, 5);
        encoder.DispatchThreadgroups(threadgroups, threadsPerGroup);

        if (Math.Abs(alpha - 1.0f) > 1e-7f)
            Scale(cBuffer, cBuffer, alpha, M * N);
        // beta-blend with prior C contents not implemented for the transposed
        // path yet; callers that need beta!=0 fall back through the generic
        // Gemm + materialize-transpose path. (Not wired here; CPU dispatcher
        // handles that case via TensorMatMul(a, transpose(b))).
    }

    public void Gemm(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int M, int N, int K, float alpha = 1.0f, float beta = 0.0f)
    {
        ThrowIfDisposed();

        if (A is not MetalGpuBuffer aBuffer || B is not MetalGpuBuffer bBuffer || C is not MetalGpuBuffer cBuffer)
        {
            throw new ArgumentException("Buffers must be MetalGpuBuffer");
        }

        // For optimized GEMM, we would use MPS
        // For now, use tiled matrix multiplication kernel
        var pipeline = GetPipeline("Matrix", _matrixLibrary, "matmul_tiled");
        var (threadgroups, threadsPerGroup) = pipeline.Calculate2DDispatch(N, M);

        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        encoder.SetBuffer(aBuffer, 0);
        encoder.SetBuffer(bBuffer, 1);
        encoder.SetBuffer(cBuffer, 2);
        encoder.SetBytes((uint)M, 3);
        encoder.SetBytes((uint)N, 4);
        encoder.SetBytes((uint)K, 5);
        encoder.DispatchThreadgroups(threadgroups, threadsPerGroup);

        // Handle alpha/beta if not default
        if (Math.Abs(alpha - 1.0f) > 1e-7f || Math.Abs(beta) > 1e-7f)
        {
            // Scale result by alpha and add beta*C_old
            // This is simplified - full implementation would handle beta * C_old
            if (Math.Abs(alpha - 1.0f) > 1e-7f)
            {
                Scale(cBuffer, cBuffer, alpha, M * N);
            }
        }
    }

    /// <summary>
    /// Matrix multiplication with new output buffer: C = A * B
    /// </summary>
    public IGpuBuffer MatMul(IGpuBuffer A, IGpuBuffer B, int M, int N, int K)
    {
        ThrowIfDisposed();

        var C = AllocateBuffer(M * N);
        Gemm(A, B, C, M, N, K);
        return C;
    }

    /// <summary>
    /// Batched matrix multiplication for many small matrices.
    /// </summary>
    public void BatchedGemm(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int M, int N, int K, int batchCount, float alpha = 1.0f, float beta = 0.0f)
    {
        ThrowIfDisposed();

        if (A is not MetalGpuBuffer aBuffer || B is not MetalGpuBuffer bBuffer || C is not MetalGpuBuffer cBuffer)
        {
            throw new ArgumentException("Buffers must be MetalGpuBuffer");
        }

        var pipeline = GetPipeline("Matrix", _matrixLibrary, "batch_matmul");
        var (threadgroups, threadsPerGroup) = pipeline.Calculate3DDispatch(N, M, batchCount);

        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        encoder.SetBuffer(aBuffer, 0);
        encoder.SetBuffer(bBuffer, 1);
        encoder.SetBuffer(cBuffer, 2);
        encoder.SetBytes((uint)batchCount, 3);
        encoder.SetBytes((uint)M, 4);
        encoder.SetBytes((uint)N, 5);
        encoder.SetBytes((uint)K, 6);
        encoder.DispatchThreadgroups(threadgroups, threadsPerGroup);
    }

    #endregion

    #region Fused Operations

    /// <summary>
    /// Fused GEMM + Bias + ReLU: output = ReLU(A * B + bias)
    /// </summary>
    public IGpuBuffer GemmBiasRelu(IGpuBuffer A, IGpuBuffer B, IGpuBuffer bias, int M, int N, int K)
    {
        ThrowIfDisposed();

        var C = AllocateBuffer(M * N);
        Gemm(A, B, C, M, N, K);
        BiasAdd(C, bias, C, M, N);
        Relu(C, C, M * N);
        return C;
    }

    /// <summary>
    /// Fused GEMM + Bias + GELU: output = GELU(A * B + bias)
    /// </summary>
    public IGpuBuffer GemmBiasGelu(IGpuBuffer A, IGpuBuffer B, IGpuBuffer bias, int M, int N, int K)
    {
        ThrowIfDisposed();

        var C = AllocateBuffer(M * N);
        Gemm(A, B, C, M, N, K);
        BiasAdd(C, bias, C, M, N);
        Gelu(C, C, M * N);
        return C;
    }

    /// <summary>
    /// Fused GEMM + Bias + Sigmoid: output = Sigmoid(A * B + bias)
    /// </summary>
    public IGpuBuffer GemmBiasSigmoid(IGpuBuffer A, IGpuBuffer B, IGpuBuffer bias, int M, int N, int K)
    {
        ThrowIfDisposed();

        var C = AllocateBuffer(M * N);
        Gemm(A, B, C, M, N, K);
        BiasAdd(C, bias, C, M, N);
        Sigmoid(C, C, M * N);
        return C;
    }

    /// <summary>
    /// Fused GEMM + Bias + Tanh: output = Tanh(A * B + bias)
    /// </summary>
    public IGpuBuffer GemmBiasTanh(IGpuBuffer A, IGpuBuffer B, IGpuBuffer bias, int M, int N, int K)
    {
        ThrowIfDisposed();

        var C = AllocateBuffer(M * N);
        Gemm(A, B, C, M, N, K);
        BiasAdd(C, bias, C, M, N);
        Tanh(C, C, M * N);
        return C;
    }

    /// <summary>
    /// Fused GEMM + Bias (no activation): output = A * B + bias
    /// </summary>
    public IGpuBuffer GemmBias(IGpuBuffer A, IGpuBuffer B, IGpuBuffer bias, int M, int N, int K)
    {
        ThrowIfDisposed();

        var C = AllocateBuffer(M * N);
        Gemm(A, B, C, M, N, K);
        BiasAdd(C, bias, C, M, N);
        return C;
    }

    public IGpuBuffer GemmBiasSwish(IGpuBuffer A, IGpuBuffer B, IGpuBuffer bias, int M, int N, int K)
    {
        ThrowIfDisposed();
        var temp = GemmBias(A, B, bias, M, N, K);
        var output = AllocateBuffer(M * N);
        Silu(temp, output, M * N);
        temp.Dispose();
        return output;
    }

    public IGpuBuffer GemmBiasLeakyRelu(IGpuBuffer A, IGpuBuffer B, IGpuBuffer bias, int M, int N, int K, float alpha = 0.01f)
    {
        ThrowIfDisposed();
        var temp = GemmBias(A, B, bias, M, N, K);
        var output = AllocateBuffer(M * N);
        LeakyRelu(temp, output, alpha, M * N);
        temp.Dispose();
        return output;
    }

    #endregion

    #region Broadcast Operations

    /// <summary>
    /// Adds a bias vector to each row of a matrix.
    /// </summary>
    public void BiasAdd(IGpuBuffer A, IGpuBuffer bias, IGpuBuffer C, int M, int N)
    {
        ThrowIfDisposed();

        // For each row i: C[i,j] = A[i,j] + bias[j]
        BroadcastAddLastAxis(A, bias, C, M, N);
    }

    /// <summary>
    /// Adds bias to Conv2D output in NCHW format.
    /// </summary>
    public void Conv2DBiasAdd(IGpuBuffer output, IGpuBuffer bias, int batch, int channels, int spatialSize)
    {
        ThrowIfDisposed();

        if (output is not MetalGpuBuffer outBuffer || bias is not MetalGpuBuffer biasBuffer)
        {
            throw new ArgumentException("Buffers must be MetalGpuBuffer");
        }

        // For NCHW: broadcast bias[c] across all [b, c, h*w] positions
        // Treat as 2D: outer = batch * channels, inner = spatialSize
        // Each thread handles one (b*channels + c, s) position
        var pipeline = GetPipeline("ElementWise", _elementWiseLibrary, "broadcast_add_last");
        var (threadgroups, threadsPerGroup) = pipeline.Calculate2DDispatch(spatialSize, batch * channels);

        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        encoder.SetBuffer(outBuffer, 0);   // A (in-place)
        encoder.SetBuffer(biasBuffer, 1);  // B (bias repeated per channel)
        encoder.SetBuffer(outBuffer, 2);   // C (output, same as A for in-place)
        encoder.SetBytes((uint)(batch * channels), 3);
        encoder.SetBytes((uint)spatialSize, 4);
        encoder.DispatchThreadgroups(threadgroups, threadsPerGroup);
    }

    /// <summary>
    /// Broadcast multiply along last axis: C[i,j] = A[i,j] * B[j]
    /// </summary>
    public void BroadcastMultiplyLastAxis(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int outerSize, int innerSize)
    {
        ThrowIfDisposed();

        if (A is not MetalGpuBuffer aBuffer || B is not MetalGpuBuffer bBuffer || C is not MetalGpuBuffer cBuffer)
        {
            throw new ArgumentException("Buffers must be MetalGpuBuffer");
        }

        var pipeline = GetPipeline("ElementWise", _elementWiseLibrary, "broadcast_mul_last");
        var (threadgroups, threadsPerGroup) = pipeline.Calculate2DDispatch(innerSize, outerSize);

        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        encoder.SetBuffer(aBuffer, 0);
        encoder.SetBuffer(bBuffer, 1);
        encoder.SetBuffer(cBuffer, 2);
        encoder.SetBytes((uint)outerSize, 3);
        encoder.SetBytes((uint)innerSize, 4);
        encoder.DispatchThreadgroups(threadgroups, threadsPerGroup);
    }

    /// <summary>
    /// Broadcast multiply along first axis: C[i,j] = A[i,j] * B[i]
    /// </summary>
    public void BroadcastMultiplyFirstAxis(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int outerSize, int innerSize)
    {
        ThrowIfDisposed();

        if (A is not MetalGpuBuffer aBuffer || B is not MetalGpuBuffer bBuffer || C is not MetalGpuBuffer cBuffer)
        {
            throw new ArgumentException("Buffers must be MetalGpuBuffer");
        }

        // For first axis broadcast: C[i,j] = A[i,j] * B[i]
        // We can use the same kernel but swap dimensions
        var pipeline = GetPipeline("ElementWise", _elementWiseLibrary, "broadcast_mul_first");
        var (threadgroups, threadsPerGroup) = pipeline.Calculate2DDispatch(innerSize, outerSize);

        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        encoder.SetBuffer(aBuffer, 0);
        encoder.SetBuffer(bBuffer, 1);
        encoder.SetBuffer(cBuffer, 2);
        encoder.SetBytes((uint)outerSize, 3);
        encoder.SetBytes((uint)innerSize, 4);
        encoder.DispatchThreadgroups(threadgroups, threadsPerGroup);
    }

    /// <summary>
    /// Broadcast add along last axis: C[i,j] = A[i,j] + B[j]
    /// </summary>
    private void BroadcastAddLastAxis(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int outerSize, int innerSize)
    {
        if (A is not MetalGpuBuffer aBuffer || B is not MetalGpuBuffer bBuffer || C is not MetalGpuBuffer cBuffer)
        {
            throw new ArgumentException("Buffers must be MetalGpuBuffer");
        }

        var pipeline = GetPipeline("ElementWise", _elementWiseLibrary, "broadcast_add_last");
        var (threadgroups, threadsPerGroup) = pipeline.Calculate2DDispatch(innerSize, outerSize);

        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        encoder.SetBuffer(aBuffer, 0);
        encoder.SetBuffer(bBuffer, 1);
        encoder.SetBuffer(cBuffer, 2);
        encoder.SetBytes((uint)outerSize, 3);
        encoder.SetBytes((uint)innerSize, 4);
        encoder.DispatchThreadgroups(threadgroups, threadsPerGroup);
    }

    #endregion

    #region Dot Product Operations

    public void DotProduct(IGpuBuffer a, IGpuBuffer b, IGpuBuffer result, int size)
    {
        ThrowIfDisposed();
        if (a is not MetalGpuBuffer aBuffer || b is not MetalGpuBuffer bBuffer || result is not MetalGpuBuffer resultBuffer)
            throw new ArgumentException("Buffers must be MetalGpuBuffer");
        if (size <= 0)
        {
            // Zero result for empty input
            Scale(result, result, 0f, Math.Max(1, result.Size));
            return;
        }
        if (size > a.Size) throw new ArgumentOutOfRangeException(nameof(size), $"Size ({size}) exceeds buffer A length ({a.Size}).");
        if (size > b.Size) throw new ArgumentOutOfRangeException(nameof(size), $"Size ({size}) exceeds buffer B length ({b.Size}).");

        uint threadgroupSize = 256;
        uint numGroups = (uint)Math.Min((size + threadgroupSize - 1) / threadgroupSize, 256);

        // Pass 1: compute partial dot products per threadgroup
        using var partialBuf = AllocateBuffer((int)numGroups);
        var partialBuffer = (MetalGpuBuffer)partialBuf;

        var pipeline = GetPipeline("DotProduct", _dotProductLibrary, "dot_product");
        using (var encoder = _commandQueue.CreateScopedComputeEncoder())
        {
            encoder.SetPipelineState(pipeline.Handle);
            encoder.SetBuffer(aBuffer, 0);
            encoder.SetBuffer(bBuffer, 1);
            encoder.SetBuffer(partialBuffer, 2);
            encoder.SetBytes((uint)size, 3);
            encoder.SetThreadgroupMemoryLength(threadgroupSize * sizeof(float), 0);
            encoder.DispatchThreadgroups(
                new MTLSize(numGroups, 1, 1),
                new MTLSize(threadgroupSize, 1, 1));
        }

        // Pass 2: reduce partial sums to final result
        var reducePipeline = GetPipeline("DotProduct", _dotProductLibrary, "reduce_partial_sums");
        using (var encoder = _commandQueue.CreateScopedComputeEncoder())
        {
            encoder.SetPipelineState(reducePipeline.Handle);
            encoder.SetBuffer(partialBuffer, 0);
            encoder.SetBuffer(resultBuffer, 1);
            encoder.SetBytes(numGroups, 2);
            uint reduceThreads = Math.Min(threadgroupSize, numGroups);
            encoder.SetThreadgroupMemoryLength(reduceThreads * sizeof(float), 0);
            encoder.DispatchThreadgroups(
                new MTLSize(1, 1, 1),
                new MTLSize(reduceThreads, 1, 1));
        }
    }

    public void StridedDotProduct(IGpuBuffer a, IGpuBuffer b, IGpuBuffer result,
        int aSize, int bSize, int bOffset, int bStride)
    {
        ThrowIfDisposed();
        if (a is not MetalGpuBuffer aBuffer || b is not MetalGpuBuffer bBuffer || result is not MetalGpuBuffer resultBuffer)
            throw new ArgumentException("Buffers must be MetalGpuBuffer");

        // Strided dot product uses single-thread kernel (windowed access doesn't parallelize)
        var pipeline = GetPipeline("DotProduct", _dotProductLibrary, "strided_dot_product");
        using (var encoder = _commandQueue.CreateScopedComputeEncoder())
        {
            encoder.SetPipelineState(pipeline.Handle);
            encoder.SetBuffer(aBuffer, 0);
            encoder.SetBuffer(bBuffer, 1);
            encoder.SetBuffer(resultBuffer, 2);
            encoder.SetBytes((uint)aSize, 3);
            encoder.SetBytes((uint)bSize, 4);
            encoder.SetBytes((uint)bOffset, 5);
            encoder.SetBytes((uint)bStride, 6);
            encoder.DispatchThreadgroups(
                new MTLSize(1, 1, 1),
                new MTLSize(1, 1, 1));
        }
    }

    public void BatchedDotProduct(IGpuBuffer a, IGpuBuffer b, IGpuBuffer result,
        int batchSize, int vecSize)
    {
        ThrowIfDisposed();
        if (a is not MetalGpuBuffer aBuffer || b is not MetalGpuBuffer bBuffer || result is not MetalGpuBuffer resultBuffer)
            throw new ArgumentException("Buffers must be MetalGpuBuffer");

        uint threadgroupSize = (uint)Math.Min(256, vecSize);

        var pipeline = GetPipeline("DotProduct", _dotProductLibrary, "batched_dot_product");
        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        encoder.SetBuffer(aBuffer, 0);
        encoder.SetBuffer(bBuffer, 1);
        encoder.SetBuffer(resultBuffer, 2);
        encoder.SetBytes((uint)batchSize, 3);
        encoder.SetBytes((uint)vecSize, 4);
        encoder.SetThreadgroupMemoryLength(threadgroupSize * sizeof(float), 0);
        encoder.DispatchThreadgroups(
            new MTLSize(1, (ulong)batchSize, 1),
            new MTLSize(threadgroupSize, 1, 1));
    }

    #endregion

    #region Element-wise Operations

    /// <summary>
    /// Element-wise addition: C = A + B
    /// </summary>
    public void Add(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int size)
    {
        ThrowIfDisposed();
        ExecuteElementWiseOp("add", A, B, C, size);
    }
    public void AddRelu(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int size) { Add(A, B, C, size); Relu(C, C, size); }
    public void AddSigmoid(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int size) { Add(A, B, C, size); Sigmoid(C, C, size); }
    public void AddGelu(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int size) { Add(A, B, C, size); Gelu(C, C, size); }

    /// <summary>
    /// Element-wise subtraction: C = A - B
    /// </summary>
    public void Subtract(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int size)
    {
        ThrowIfDisposed();
        ExecuteElementWiseOp("subtract", A, B, C, size);
    }

    /// <summary>
    /// Element-wise multiplication: C = A * B
    /// </summary>
    public void Multiply(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int size)
    {
        ThrowIfDisposed();
        ExecuteElementWiseOp("multiply", A, B, C, size);
    }

    /// <summary>
    /// Element-wise division: C = A / B
    /// </summary>
    public void Divide(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int size)
    {
        ThrowIfDisposed();
        ExecuteElementWiseOp("divide", A, B, C, size);
    }

    /// <summary>
    /// Element-wise minimum: C = min(A, B)
    /// </summary>
    public void Min(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int size)
    {
        ThrowIfDisposed();
        ExecuteElementWiseOp("minimum", A, B, C, size);
    }

    /// <summary>
    /// Element-wise maximum: C = max(A, B)
    /// </summary>
    public void Max(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int size)
    {
        ThrowIfDisposed();
        ExecuteElementWiseOp("maximum", A, B, C, size);
    }

    /// <summary>
    /// Scalar multiplication: B = A * scalar
    /// </summary>
    public void Scale(IGpuBuffer A, IGpuBuffer B, float scalar, int size)
    {
        ThrowIfDisposed();

        if (A is not MetalGpuBuffer aBuffer || B is not MetalGpuBuffer bBuffer)
        {
            throw new ArgumentException("Buffers must be MetalGpuBuffer");
        }

        var pipeline = GetPipeline("ElementWise", _elementWiseLibrary, "multiply_scalar");
        var (threadgroups, threadsPerGroup) = pipeline.Calculate1DDispatch(size);

        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        encoder.SetBuffer(aBuffer, 0);  // A: input
        encoder.SetBuffer(bBuffer, 1);  // B: output
        encoder.SetBytes(scalar, 2);    // scalar
        encoder.SetBytes((uint)size, 3);
        encoder.DispatchThreadgroups(threadgroups, threadsPerGroup);
    }

    public void StridedGather(IGpuBuffer src, IGpuBuffer dst, int offset, int stride, int count)
    {
        ThrowIfDisposed();
        if (count <= 0) return;
        if (offset < 0) throw new ArgumentOutOfRangeException(nameof(offset));
        if (stride <= 0) throw new ArgumentOutOfRangeException(nameof(stride));
        if (offset + (count - 1) * stride >= src.Size) throw new ArgumentOutOfRangeException(nameof(count));

        if (src is not MetalGpuBuffer srcBuffer || dst is not MetalGpuBuffer dstBuffer)
            throw new ArgumentException("Buffers must be MetalGpuBuffer");

        var pipeline = GetPipeline("ElementWise", _elementWiseLibrary, "strided_gather");
        var (threadgroups, threadsPerGroup) = pipeline.Calculate1DDispatch(count);

        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        encoder.SetBuffer(srcBuffer, 0);
        encoder.SetBuffer(dstBuffer, 1);
        encoder.SetBytes((uint)offset, 2);
        encoder.SetBytes((uint)stride, 3);
        encoder.SetBytes((uint)count, 4);
        encoder.DispatchThreadgroups(threadgroups, threadsPerGroup);
    }

    public void StridedScatter(IGpuBuffer src, IGpuBuffer dst, int offset, int stride, int count)
    {
        ThrowIfDisposed();
        if (count <= 0) return;
        if (offset < 0) throw new ArgumentOutOfRangeException(nameof(offset));
        if (stride <= 0) throw new ArgumentOutOfRangeException(nameof(stride));
        if (offset + (count - 1) * stride >= dst.Size) throw new ArgumentOutOfRangeException(nameof(count));

        if (src is not MetalGpuBuffer srcBuffer || dst is not MetalGpuBuffer dstBuffer)
            throw new ArgumentException("Buffers must be MetalGpuBuffer");

        var pipeline = GetPipeline("ElementWise", _elementWiseLibrary, "strided_scatter");
        var (threadgroups, threadsPerGroup) = pipeline.Calculate1DDispatch(count);

        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        encoder.SetBuffer(srcBuffer, 0);
        encoder.SetBuffer(dstBuffer, 1);
        encoder.SetBytes((uint)offset, 2);
        encoder.SetBytes((uint)stride, 3);
        encoder.SetBytes((uint)count, 4);
        encoder.DispatchThreadgroups(threadgroups, threadsPerGroup);
    }

    /// <summary>
    /// Power with scalar exponent: B = A ^ exponent
    /// </summary>
    public void Power(IGpuBuffer A, IGpuBuffer B, float exponent, int size)
    {
        ThrowIfDisposed();

        if (A is not MetalGpuBuffer aBuffer || B is not MetalGpuBuffer bBuffer)
        {
            throw new ArgumentException("Buffers must be MetalGpuBuffer");
        }

        var pipeline = GetPipeline("ElementWise", _elementWiseLibrary, "pow_kernel");
        var (threadgroups, threadsPerGroup) = pipeline.Calculate1DDispatch(size);

        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        encoder.SetBuffer(aBuffer, 0);  // A: input
        encoder.SetBuffer(bBuffer, 1);  // B: output
        encoder.SetBytes(exponent, 2);  // power exponent
        encoder.SetBytes((uint)size, 3);
        encoder.DispatchThreadgroups(threadgroups, threadsPerGroup);
    }

    /// <summary>
    /// Absolute value: B = abs(A)
    /// </summary>
    public void Abs(IGpuBuffer A, IGpuBuffer B, int size)
    {
        ThrowIfDisposed();
        ExecuteUnaryOp("abs_kernel", A, B, size, _elementWiseLibrary);
    }

    /// <summary>
    /// Exponential: B = exp(A)
    /// </summary>
    public void Exp(IGpuBuffer A, IGpuBuffer B, int size)
    {
        ThrowIfDisposed();
        ExecuteUnaryOp("exp_kernel", A, B, size, _elementWiseLibrary);
    }

    /// <summary>
    /// Base-2 exponential: B = 2^A
    /// </summary>
    public void Exp2(IGpuBuffer A, IGpuBuffer B, int size)
    {
        ThrowIfDisposed();
        // Use exp(A * ln(2))
        var ln2 = 0.693147180559945f;
        Scale(A, B, ln2, size);
        Exp(B, B, size);
    }

    /// <summary>
    /// Base-10 exponential: B = 10^A
    /// </summary>
    public void Exp10(IGpuBuffer A, IGpuBuffer B, int size)
    {
        ThrowIfDisposed();
        // Use exp(A * ln(10))
        var ln10 = 2.302585092994046f;
        Scale(A, B, ln10, size);
        Exp(B, B, size);
    }

    /// <summary>
    /// Expm1: B = exp(A) - 1
    /// </summary>
    public void ExpM1(IGpuBuffer A, IGpuBuffer B, int size)
    {
        ThrowIfDisposed();
        Exp(A, B, size);
        AddScalar(B, B, -1.0f, size);
    }

    /// <summary>
    /// Natural log: B = log(A)
    /// </summary>
    public void Log(IGpuBuffer A, IGpuBuffer B, int size)
    {
        ThrowIfDisposed();
        ExecuteUnaryOp("log_kernel", A, B, size, _elementWiseLibrary);
    }

    /// <summary>
    /// Base-2 log: B = log2(A)
    /// </summary>
    public void Log2(IGpuBuffer A, IGpuBuffer B, int size)
    {
        ThrowIfDisposed();
        Log(A, B, size);
        Scale(B, B, 1.0f / 0.693147180559945f, size);
    }

    /// <summary>
    /// Log1p: B = log(1 + A)
    /// </summary>
    public void Log1P(IGpuBuffer A, IGpuBuffer B, int size)
    {
        ThrowIfDisposed();
        AddScalar(A, B, 1.0f, size);
        Log(B, B, size);
    }

    /// <summary>
    /// Square root: B = sqrt(A)
    /// </summary>
    public void Sqrt(IGpuBuffer A, IGpuBuffer B, int size)
    {
        ThrowIfDisposed();
        ExecuteUnaryOp("sqrt_kernel", A, B, size, _elementWiseLibrary);
    }

    /// <summary>
    /// Sign: B = sign(A)
    /// </summary>
    public void Sign(IGpuBuffer A, IGpuBuffer B, int size)
    {
        ThrowIfDisposed();
        ExecuteUnaryOp("sign_kernel", A, B, size, _trigLibrary);
    }

    /// <summary>
    /// Helper to add a scalar: B = A + scalar
    /// </summary>
    public void AddScalar(IGpuBuffer A, IGpuBuffer B, float scalar, int size)
    {
        if (A is not MetalGpuBuffer aBuffer || B is not MetalGpuBuffer bBuffer)
        {
            throw new ArgumentException("Buffers must be MetalGpuBuffer");
        }

        var pipeline = GetPipeline("ElementWise", _elementWiseLibrary, "add_scalar");
        var (threadgroups, threadsPerGroup) = pipeline.Calculate1DDispatch(size);

        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        encoder.SetBuffer(aBuffer, 0);  // A: input
        encoder.SetBuffer(bBuffer, 1);  // B: output
        encoder.SetBytes(scalar, 2);    // scalar value
        encoder.SetBytes((uint)size, 3);
        encoder.DispatchThreadgroups(threadgroups, threadsPerGroup);
    }

    /// <summary>
    /// Helper to execute binary element-wise operations.
    /// </summary>
    private void ExecuteElementWiseOp(string kernelName, IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int size)
    {
        if (A is not MetalGpuBuffer aBuffer || B is not MetalGpuBuffer bBuffer || C is not MetalGpuBuffer cBuffer)
        {
            throw new ArgumentException("Buffers must be MetalGpuBuffer");
        }

        var pipeline = GetPipeline("ElementWise", _elementWiseLibrary, kernelName);
        var (threadgroups, threadsPerGroup) = pipeline.Calculate1DDispatch(size);

        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        encoder.SetBuffer(aBuffer, 0);
        encoder.SetBuffer(bBuffer, 1);
        encoder.SetBuffer(cBuffer, 2);
        encoder.SetBytes((uint)size, 3);
        encoder.DispatchThreadgroups(threadgroups, threadsPerGroup);
    }

    /// <summary>
    /// Helper to execute unary operations.
    /// </summary>
    private void ExecuteUnaryOp(string kernelName, IGpuBuffer A, IGpuBuffer B, int size, IntPtr library)
    {
        if (A is not MetalGpuBuffer aBuffer || B is not MetalGpuBuffer bBuffer)
        {
            throw new ArgumentException("Buffers must be MetalGpuBuffer");
        }

        var libraryName = library == _elementWiseLibrary ? "ElementWise" :
                          library == _activationLibrary ? "Activation" :
                          library == _trigLibrary ? "Trig" : "Unknown";

        var pipeline = GetPipeline(libraryName, library, kernelName);
        var (threadgroups, threadsPerGroup) = pipeline.Calculate1DDispatch(size);

        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        encoder.SetBuffer(aBuffer, 0);
        encoder.SetBuffer(bBuffer, 1);
        encoder.SetBytes((uint)size, 2);
        encoder.DispatchThreadgroups(threadgroups, threadsPerGroup);
    }

    #endregion

    #region Activation Functions

    /// <summary>
    /// ReLU activation: B = max(0, A)
    /// </summary>
    public void Relu(IGpuBuffer A, IGpuBuffer B, int size)
    {
        ThrowIfDisposed();
        ExecuteUnaryOp("relu", A, B, size, _activationLibrary);
    }

    /// <summary>
    /// Sigmoid activation: B = 1 / (1 + exp(-A))
    /// </summary>
    public void Sigmoid(IGpuBuffer A, IGpuBuffer B, int size)
    {
        ThrowIfDisposed();
        ExecuteUnaryOp("sigmoid", A, B, size, _activationLibrary);
    }

    /// <summary>
    /// Tanh activation: B = tanh(A)
    /// </summary>
    public void Tanh(IGpuBuffer A, IGpuBuffer B, int size)
    {
        ThrowIfDisposed();
        ExecuteUnaryOp("tanh_activation", A, B, size, _activationLibrary);
    }

    /// <summary>
    /// GELU activation.
    /// </summary>
    public void Gelu(IGpuBuffer A, IGpuBuffer B, int size)
    {
        ThrowIfDisposed();
        ExecuteUnaryOp("gelu", A, B, size, _activationLibrary);
    }

    /// <summary>
    /// Softmax activation along last dimension.
    /// </summary>
    public void Softmax(IGpuBuffer A, IGpuBuffer B, int batchSize, int features)
    {
        ThrowIfDisposed();

        if (A is not MetalGpuBuffer aBuffer || B is not MetalGpuBuffer bBuffer)
        {
            throw new ArgumentException("Buffers must be MetalGpuBuffer");
        }

        var pipeline = GetPipeline("Normalization", _normalizationLibrary, "softmax_row");
        var (threadgroups, threadsPerGroup) = pipeline.Calculate1DDispatch(batchSize);

        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        encoder.SetBuffer(aBuffer, 0);
        encoder.SetBuffer(bBuffer, 1);
        encoder.SetBytes((uint)batchSize, 2);
        encoder.SetBytes((uint)features, 3);
        encoder.DispatchThreadgroups(threadgroups, threadsPerGroup);
    }

    #endregion

    #region Reduction Operations

    /// <summary>
    /// Sum all elements in buffer using GPU parallel reduction.
    /// </summary>
    public float Sum(IGpuBuffer A, int size)
    {
        ThrowIfDisposed();

        if (A is not MetalGpuBuffer aBuffer)
        {
            throw new ArgumentException("Buffer must be MetalGpuBuffer", nameof(A));
        }

        // Use GPU parallel reduction with multiple passes
        const int threadgroupSize = 256;
        int numThreadgroups = (size + threadgroupSize - 1) / threadgroupSize;

        // Allocate buffer for partial sums
        using var partialSums = new MetalGpuBuffer(_device, numThreadgroups);

        // First pass: reduce to partial sums
        var pipeline = GetPipeline("Reduction", _reductionLibrary, "sum_reduce");

        using (var encoder = _commandQueue.CreateScopedComputeEncoder())
        {
            encoder.SetPipelineState(pipeline.Handle);
            encoder.SetBuffer(aBuffer, 0);
            encoder.SetBuffer(partialSums, 1);
            encoder.SetBytes((uint)size, 2);
            encoder.SetThreadgroupMemoryLength((uint)(threadgroupSize * sizeof(float)), 0);
            encoder.DispatchThreadgroups(
                new MTLSize((ulong)numThreadgroups, 1, 1),
                new MTLSize((ulong)threadgroupSize, 1, 1));
        }

        // If we have multiple threadgroups, reduce again
        while (numThreadgroups > 1)
        {
            int prevSize = numThreadgroups;
            numThreadgroups = (numThreadgroups + threadgroupSize - 1) / threadgroupSize;
            using var newPartialSums = new MetalGpuBuffer(_device, numThreadgroups);

            using (var encoder = _commandQueue.CreateScopedComputeEncoder())
            {
                encoder.SetPipelineState(pipeline.Handle);
                encoder.SetBuffer(partialSums, 0);
                encoder.SetBuffer(newPartialSums, 1);
                encoder.SetBytes((uint)prevSize, 2);
                encoder.SetThreadgroupMemoryLength((uint)(threadgroupSize * sizeof(float)), 0);
                encoder.DispatchThreadgroups(
                    new MTLSize((ulong)numThreadgroups, 1, 1),
                    new MTLSize((ulong)threadgroupSize, 1, 1));
            }

            if (numThreadgroups == 1)
            {
                var result = new float[1];
                newPartialSums.CopyTo(result);
                return result[0];
            }
        }

        var finalResult = new float[1];
        partialSums.CopyTo(finalResult);
        return finalResult[0];
    }

    /// <summary>
    /// Find maximum element in buffer using GPU parallel reduction.
    /// </summary>
    public float Max(IGpuBuffer A, int size)
    {
        ThrowIfDisposed();

        if (A is not MetalGpuBuffer aBuffer)
        {
            throw new ArgumentException("Buffer must be MetalGpuBuffer", nameof(A));
        }

        const int threadgroupSize = 256;
        int numThreadgroups = (size + threadgroupSize - 1) / threadgroupSize;

        using var partialMax = new MetalGpuBuffer(_device, numThreadgroups);

        var pipeline = GetPipeline("Reduction", _reductionLibrary, "max_reduce");

        using (var encoder = _commandQueue.CreateScopedComputeEncoder())
        {
            encoder.SetPipelineState(pipeline.Handle);
            encoder.SetBuffer(aBuffer, 0);
            encoder.SetBuffer(partialMax, 1);
            encoder.SetBytes((uint)size, 2);
            encoder.SetThreadgroupMemoryLength((uint)(threadgroupSize * sizeof(float)), 0);
            encoder.DispatchThreadgroups(
                new MTLSize((ulong)numThreadgroups, 1, 1),
                new MTLSize((ulong)threadgroupSize, 1, 1));
        }

        while (numThreadgroups > 1)
        {
            int prevSize = numThreadgroups;
            numThreadgroups = (numThreadgroups + threadgroupSize - 1) / threadgroupSize;
            using var newPartialMax = new MetalGpuBuffer(_device, numThreadgroups);

            using (var encoder = _commandQueue.CreateScopedComputeEncoder())
            {
                encoder.SetPipelineState(pipeline.Handle);
                encoder.SetBuffer(partialMax, 0);
                encoder.SetBuffer(newPartialMax, 1);
                encoder.SetBytes((uint)prevSize, 2);
                encoder.SetThreadgroupMemoryLength((uint)(threadgroupSize * sizeof(float)), 0);
                encoder.DispatchThreadgroups(
                    new MTLSize((ulong)numThreadgroups, 1, 1),
                    new MTLSize((ulong)threadgroupSize, 1, 1));
            }

            if (numThreadgroups == 1)
            {
                var result = new float[1];
                newPartialMax.CopyTo(result);
                return result[0];
            }
        }

        var finalResult = new float[1];
        partialMax.CopyTo(finalResult);
        return finalResult[0];
    }

    /// <summary>
    /// Find minimum element in buffer using GPU parallel reduction.
    /// </summary>
    public float Min(IGpuBuffer A, int size)
    {
        ThrowIfDisposed();

        if (A is not MetalGpuBuffer aBuffer)
        {
            throw new ArgumentException("Buffer must be MetalGpuBuffer", nameof(A));
        }

        const int threadgroupSize = 256;
        int numThreadgroups = (size + threadgroupSize - 1) / threadgroupSize;

        using var partialMin = new MetalGpuBuffer(_device, numThreadgroups);

        var pipeline = GetPipeline("Reduction", _reductionLibrary, "min_reduce");

        using (var encoder = _commandQueue.CreateScopedComputeEncoder())
        {
            encoder.SetPipelineState(pipeline.Handle);
            encoder.SetBuffer(aBuffer, 0);
            encoder.SetBuffer(partialMin, 1);
            encoder.SetBytes((uint)size, 2);
            encoder.SetThreadgroupMemoryLength((uint)(threadgroupSize * sizeof(float)), 0);
            encoder.DispatchThreadgroups(
                new MTLSize((ulong)numThreadgroups, 1, 1),
                new MTLSize((ulong)threadgroupSize, 1, 1));
        }

        while (numThreadgroups > 1)
        {
            int prevSize = numThreadgroups;
            numThreadgroups = (numThreadgroups + threadgroupSize - 1) / threadgroupSize;
            using var newPartialMin = new MetalGpuBuffer(_device, numThreadgroups);

            using (var encoder = _commandQueue.CreateScopedComputeEncoder())
            {
                encoder.SetPipelineState(pipeline.Handle);
                encoder.SetBuffer(partialMin, 0);
                encoder.SetBuffer(newPartialMin, 1);
                encoder.SetBytes((uint)prevSize, 2);
                encoder.SetThreadgroupMemoryLength((uint)(threadgroupSize * sizeof(float)), 0);
                encoder.DispatchThreadgroups(
                    new MTLSize((ulong)numThreadgroups, 1, 1),
                    new MTLSize((ulong)threadgroupSize, 1, 1));
            }

            if (numThreadgroups == 1)
            {
                var result = new float[1];
                newPartialMin.CopyTo(result);
                return result[0];
            }
        }

        var finalResult = new float[1];
        partialMin.CopyTo(finalResult);
        return finalResult[0];
    }

    /// <summary>
    /// Sum along axis for batched data using GPU kernel.
    /// </summary>
    public void SumAxis(IGpuBuffer A, IGpuBuffer B, int outerSize, int reduceSize)
    {
        ThrowIfDisposed();

        if (A is not MetalGpuBuffer aBuffer || B is not MetalGpuBuffer bBuffer)
        {
            throw new ArgumentException("Buffers must be MetalGpuBuffer");
        }

        var pipeline = GetPipeline("Reduction", _reductionLibrary, "sum_axis");
        var (threadgroups, threadsPerGroup) = pipeline.Calculate1DDispatch(outerSize);

        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        encoder.SetBuffer(aBuffer, 0);
        encoder.SetBuffer(bBuffer, 1);
        encoder.SetBytes((uint)outerSize, 2);
        encoder.SetBytes((uint)reduceSize, 3);
        encoder.DispatchThreadgroups(threadgroups, threadsPerGroup);
    }

    /// <summary>
    /// Mean along axis for batched data using GPU kernel.
    /// </summary>
    public void MeanAxis(IGpuBuffer A, IGpuBuffer B, int outerSize, int reduceSize)
    {
        ThrowIfDisposed();

        if (A is not MetalGpuBuffer aBuffer || B is not MetalGpuBuffer bBuffer)
        {
            throw new ArgumentException("Buffers must be MetalGpuBuffer");
        }

        var pipeline = GetPipeline("Reduction", _reductionLibrary, "mean_axis");
        var (threadgroups, threadsPerGroup) = pipeline.Calculate1DDispatch(outerSize);

        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        encoder.SetBuffer(aBuffer, 0);
        encoder.SetBuffer(bBuffer, 1);
        encoder.SetBytes((uint)outerSize, 2);
        encoder.SetBytes((uint)reduceSize, 3);
        encoder.DispatchThreadgroups(threadgroups, threadsPerGroup);
    }

    /// <summary>
    /// Variance along axis for batched data using GPU kernel.
    /// </summary>
    public void VarAxis(IGpuBuffer A, IGpuBuffer B, int outerSize, int reduceSize)
    {
        ThrowIfDisposed();

        if (A is not MetalGpuBuffer aBuffer || B is not MetalGpuBuffer bBuffer)
        {
            throw new ArgumentException("Buffers must be MetalGpuBuffer");
        }

        var pipeline = GetPipeline("Reduction", _reductionLibrary, "var_axis");
        var (threadgroups, threadsPerGroup) = pipeline.Calculate1DDispatch(outerSize);

        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        encoder.SetBuffer(aBuffer, 0);
        encoder.SetBuffer(bBuffer, 1);
        encoder.SetBytes((uint)outerSize, 2);
        encoder.SetBytes((uint)reduceSize, 3);
        encoder.DispatchThreadgroups(threadgroups, threadsPerGroup);
    }

    /// <summary>
    /// Max along axis for batched data using GPU kernel.
    /// </summary>
    public void MaxAxis(IGpuBuffer A, IGpuBuffer B, int outerSize, int reduceSize)
    {
        ThrowIfDisposed();

        if (A is not MetalGpuBuffer aBuffer || B is not MetalGpuBuffer bBuffer)
        {
            throw new ArgumentException("Buffers must be MetalGpuBuffer");
        }

        var pipeline = GetPipeline("Reduction", _reductionLibrary, "max_axis");
        var (threadgroups, threadsPerGroup) = pipeline.Calculate1DDispatch(outerSize);

        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        encoder.SetBuffer(aBuffer, 0);
        encoder.SetBuffer(bBuffer, 1);
        encoder.SetBytes((uint)outerSize, 2);
        encoder.SetBytes((uint)reduceSize, 3);
        encoder.DispatchThreadgroups(threadgroups, threadsPerGroup);
    }

    /// <summary>
    /// ArgMax along axis for batched data using GPU kernel.
    /// </summary>
    /// <inheritdoc/>
    public bool ArgMaxIndicesAreBitReinterpreted => false; // Metal kernel uses (float)cast

    public void ArgMaxAxis(IGpuBuffer A, IGpuBuffer indices, int outerSize, int reduceSize)
    {
        ThrowIfDisposed();

        if (A is not MetalGpuBuffer aBuffer || indices is not MetalGpuBuffer idxBuffer)
        {
            throw new ArgumentException("Buffers must be MetalGpuBuffer");
        }

        var pipeline = GetPipeline("Reduction", _reductionLibrary, "argmax_axis");
        var (threadgroups, threadsPerGroup) = pipeline.Calculate1DDispatch(outerSize);

        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        encoder.SetBuffer(aBuffer, 0);
        encoder.SetBuffer(idxBuffer, 1);
        encoder.SetBytes((uint)outerSize, 2);
        encoder.SetBytes((uint)reduceSize, 3);
        encoder.DispatchThreadgroups(threadgroups, threadsPerGroup);
    }

    /// <summary>
    /// ArgMin along axis for batched data using GPU kernel.
    /// </summary>
    public void ArgMinAxis(IGpuBuffer A, IGpuBuffer indices, int outerSize, int reduceSize)
    {
        ThrowIfDisposed();

        if (A is not MetalGpuBuffer aBuffer || indices is not MetalGpuBuffer idxBuffer)
        {
            throw new ArgumentException("Buffers must be MetalGpuBuffer");
        }

        var pipeline = GetPipeline("Reduction", _reductionLibrary, "argmin_axis");
        var (threadgroups, threadsPerGroup) = pipeline.Calculate1DDispatch(outerSize);

        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        encoder.SetBuffer(aBuffer, 0);
        encoder.SetBuffer(idxBuffer, 1);
        encoder.SetBytes((uint)outerSize, 2);
        encoder.SetBytes((uint)reduceSize, 3);
        encoder.DispatchThreadgroups(threadgroups, threadsPerGroup);
    }

    #endregion

    #region Synchronization

    /// <summary>
    /// Waits for all GPU operations to complete.
    /// </summary>
    public void Synchronize()
    {
        ThrowIfDisposed();
        // Metal operations are automatically synchronized via command buffer wait
        // Additional explicit sync if needed
    }

    #endregion

    #region Utility Methods

    private void ThrowIfDisposed()
    {
        if (_disposed)
        {
            throw new ObjectDisposedException(nameof(MetalBackend));
        }
    }

    /// <summary>
    /// Disposes the Metal backend and releases all resources.
    /// </summary>
    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        lock (_lock)
        {
            if (_disposed)
            {
                return;
            }

            _disposed = true;

            // Clear pipeline cache
            foreach (var kvp in _pipelineCache)
            {
                kvp.Value.Dispose();
            }
            _pipelineCache.Clear();

            // Dispose shader library
            _shaderLibrary?.Dispose();

            // Dispose command queue
            _commandQueue?.Dispose();

            // Dispose device
            _device?.Dispose();
        }
    }

    public override string ToString()
    {
        return $"MetalBackend[{DeviceName}, {ComputeUnits} cores, {GlobalMemoryBytes / (1024.0 * 1024 * 1024):F1} GB]";
    }

    #endregion

    // --- Split-buffer native Complex<T> operations (Metal dispatch) ---

    private IntPtr _complexLibrary;
    private readonly object _complexLibraryLock = new();
    private void EnsureComplexLibrary()
    {
        if (_complexLibrary != IntPtr.Zero) return;
        lock (_complexLibraryLock)
        {
            if (_complexLibrary == IntPtr.Zero)
                _complexLibrary = _shaderLibrary.CompileLibrary("Complex", MetalComplexKernels.GetSource());
        }
    }
    private static MetalGpuBuffer AsMetal(IGpuBuffer buf) => (MetalGpuBuffer)buf;

    private void DispatchComplexMetal(string kernel, MetalGpuBuffer[] bufs, int n, float? scalar = null)
    {
        ThrowIfDisposed(); EnsureComplexLibrary();
        var pipeline = GetPipeline("Complex", _complexLibrary, kernel);
        var (tg, tpg) = pipeline.Calculate1DDispatch(n);
        using var enc = _commandQueue.CreateScopedComputeEncoder();
        enc.SetPipelineState(pipeline.Handle);
        for (int i = 0; i < bufs.Length; i++) enc.SetBuffer(bufs[i], i);
        int next = bufs.Length;
        if (scalar.HasValue) enc.SetBytes(scalar.Value, (ulong)next++);
        enc.SetBytes((uint)n, (ulong)next);
        enc.DispatchThreadgroups(tg, tpg);
    }

    public void SplitComplexMultiply(IGpuBuffer aR, IGpuBuffer aI, IGpuBuffer bR, IGpuBuffer bI, IGpuBuffer oR, IGpuBuffer oI, int n) { if (n > 0) DispatchComplexMetal("split_complex_multiply", [AsMetal(aR), AsMetal(aI), AsMetal(bR), AsMetal(bI), AsMetal(oR), AsMetal(oI)], n); }
    public void SplitComplexConjugate(IGpuBuffer iR, IGpuBuffer iI, IGpuBuffer oR, IGpuBuffer oI, int n) { if (n > 0) DispatchComplexMetal("split_complex_conjugate", [AsMetal(iR), AsMetal(iI), AsMetal(oR), AsMetal(oI)], n); }
    public void SplitComplexMagnitude(IGpuBuffer iR, IGpuBuffer iI, IGpuBuffer o, int n) { if (n > 0) DispatchComplexMetal("split_complex_magnitude", [AsMetal(iR), AsMetal(iI), AsMetal(o)], n); }
    public void SplitComplexMagnitudeSquared(IGpuBuffer iR, IGpuBuffer iI, IGpuBuffer o, int n) { if (n > 0) DispatchComplexMetal("split_complex_magnitude_squared", [AsMetal(iR), AsMetal(iI), AsMetal(o)], n); }
    public void SplitComplexPhase(IGpuBuffer iR, IGpuBuffer iI, IGpuBuffer o, int n) { if (n > 0) DispatchComplexMetal("split_complex_phase", [AsMetal(iR), AsMetal(iI), AsMetal(o)], n); }
    public void SplitComplexFromPolar(IGpuBuffer m, IGpuBuffer p, IGpuBuffer oR, IGpuBuffer oI, int n) { if (n > 0) DispatchComplexMetal("split_complex_from_polar", [AsMetal(m), AsMetal(p), AsMetal(oR), AsMetal(oI)], n); }
    public void SplitComplexScale(IGpuBuffer iR, IGpuBuffer iI, IGpuBuffer oR, IGpuBuffer oI, float s, int n) { if (n > 0) DispatchComplexMetal("split_complex_scale", [AsMetal(iR), AsMetal(iI), AsMetal(oR), AsMetal(oI)], n, s); }
    public void SplitComplexAdd(IGpuBuffer aR, IGpuBuffer aI, IGpuBuffer bR, IGpuBuffer bI, IGpuBuffer oR, IGpuBuffer oI, int n) { if (n > 0) DispatchComplexMetal("split_complex_add", [AsMetal(aR), AsMetal(aI), AsMetal(bR), AsMetal(bI), AsMetal(oR), AsMetal(oI)], n); }
    public void SplitComplexCrossSpectral(IGpuBuffer xR, IGpuBuffer xI, IGpuBuffer yR, IGpuBuffer yI, IGpuBuffer oR, IGpuBuffer oI, int n) { if (n > 0) DispatchComplexMetal("split_complex_cross_spectral", [AsMetal(xR), AsMetal(xI), AsMetal(yR), AsMetal(yI), AsMetal(oR), AsMetal(oI)], n); }

    public void SplitComplexTopK(IGpuBuffer inReal, IGpuBuffer inImag, IGpuBuffer outReal, IGpuBuffer outImag, int n, int k)
    {
        if (n <= 0 || k <= 0) return;
        var magBuf = AllocateBuffer(n);
        try
        {
            SplitComplexMagnitudeSquared(inReal, inImag, magBuf, n);
            var magData = DownloadBuffer(magBuf);
            Array.Sort(magData); Array.Reverse(magData);
            float threshold = k <= n ? magData[Math.Min(k, n) - 1] : 0f;
            DispatchComplexMetal("split_complex_topk", [AsMetal(inReal), AsMetal(inImag), AsMetal(outReal), AsMetal(outImag)], n, threshold);
        }
        finally { magBuf.Dispose(); }
    }

    public void SoftmaxRows(IGpuBuffer input, IGpuBuffer output, int rows, int cols)
    {
        if (rows <= 0 || cols <= 0) return;
        EnsureComplexLibrary();
        var pipeline = GetPipeline("Complex", _complexLibrary, "softmax_rows");
        var (tg, tpg) = pipeline.Calculate1DDispatch(rows);
        using var enc = _commandQueue.CreateScopedComputeEncoder();
        enc.SetPipelineState(pipeline.Handle);
        enc.SetBuffer(AsMetal(input), 0);
        enc.SetBuffer(AsMetal(output), 1);
        enc.SetBytes((uint)rows, (ulong)2);
        enc.SetBytes((uint)cols, (ulong)3);
        enc.DispatchThreadgroups(tg, tpg);
    }

    // ─── HRR binding primitives (issue #248) ────────────────────────

    public void SplitComplexUnitPhaseCodebook(
        IGpuBuffer outReal, IGpuBuffer outImag, int seed, int V, int D, bool kPsk, int k)
    {
        ThrowIfDisposed();
        if (V < 0) throw new ArgumentOutOfRangeException(nameof(V), "V must be >= 0.");
        if (D < 0) throw new ArgumentOutOfRangeException(nameof(D), "D must be >= 0.");
        if (V == 0 || D == 0) return;
        long total = (long)V * D;
        if (total > int.MaxValue) throw new ArgumentException($"V*D = {total} exceeds int.MaxValue.");
        if (kPsk && k <= 0) throw new ArgumentOutOfRangeException(nameof(k));
        int n = (int)total;
        if (outReal is null || outImag is null)
            throw new ArgumentNullException(nameof(outReal), "Output buffers must be non-null.");
        var outR = AsMetal(outReal);
        var outI = AsMetal(outImag);
        if (outR.Size < n || outI.Size < n)
            throw new ArgumentException(
                $"Output buffers must each hold at least V*D = {n} elements " +
                $"(got outReal={outR.Size}, outImag={outI.Size}).");
        EnsureComplexLibrary();
        var pipeline = GetPipeline("Complex", _complexLibrary, "hrr_unit_phase_codebook");
        var (tg, tpg) = pipeline.Calculate1DDispatch(n);
        using var enc = _commandQueue.CreateScopedComputeEncoder();
        enc.SetPipelineState(pipeline.Handle);
        enc.SetBuffer(outR, 0);
        enc.SetBuffer(outI, 1);
        enc.SetBytes(seed, 2);
        enc.SetBytes(V, 3);
        enc.SetBytes(D, 4);
        enc.SetBytes(kPsk ? 1 : 0, 5);
        enc.SetBytes(k, 6);
        enc.DispatchThreadgroups(tg, tpg);
    }

    public void SplitComplexPhaseCoherenceDecode(
        IGpuBuffer codesReal, IGpuBuffer codesImag,
        IGpuBuffer queryReal, IGpuBuffer queryImag,
        IGpuBuffer outScores, int V, int D)
    {
        ThrowIfDisposed();
        if (V < 0) throw new ArgumentOutOfRangeException(nameof(V), "V must be >= 0.");
        if (D < 0) throw new ArgumentOutOfRangeException(nameof(D), "D must be >= 0.");
        if (V == 0 || D == 0) return;
        if (codesReal is null || codesImag is null || queryReal is null || queryImag is null || outScores is null)
            throw new ArgumentNullException(nameof(codesReal),
                "All five GPU buffers must be non-null for SplitComplexPhaseCoherenceDecode.");
        long codeCount = (long)V * D;
        if (codeCount > int.MaxValue) throw new ArgumentException($"V*D = {codeCount} exceeds int.MaxValue.");
        var cR = AsMetal(codesReal);
        var cI = AsMetal(codesImag);
        var qR = AsMetal(queryReal);
        var qI = AsMetal(queryImag);
        var os = AsMetal(outScores);
        if (cR.Size < codeCount || cI.Size < codeCount)
            throw new ArgumentException(
                $"codes buffers must each hold at least V*D = {codeCount} elements " +
                $"(got {cR.Size}, {cI.Size}).");
        if (qR.Size < D || qI.Size < D)
            throw new ArgumentException(
                $"query buffers must each hold at least D = {D} elements " +
                $"(got {qR.Size}, {qI.Size}).");
        if (os.Size < V)
            throw new ArgumentException(
                $"outScores must hold at least V = {V} elements (got {os.Size}).");
        EnsureComplexLibrary();
        var pipeline = GetPipeline("Complex", _complexLibrary, "hrr_phase_coherence_decode");
        var (tg, tpg) = pipeline.Calculate1DDispatch(V);
        using var enc = _commandQueue.CreateScopedComputeEncoder();
        enc.SetPipelineState(pipeline.Handle);
        enc.SetBuffer(cR, 0);
        enc.SetBuffer(cI, 1);
        enc.SetBuffer(qR, 2);
        enc.SetBuffer(qI, 3);
        enc.SetBuffer(os, 4);
        enc.SetBytes(V, 5);
        enc.SetBytes(D, 6);
        enc.DispatchThreadgroups(tg, tpg);
    }

    public void SplitComplexHrrBindAccumulate(
        IGpuBuffer keyCodeReal, IGpuBuffer keyCodeImag,
        IGpuBuffer valPermCodeReal, IGpuBuffer valPermCodeImag,
        IGpuBuffer keyIds, IGpuBuffer valIds,
        IGpuBuffer memoryReal, IGpuBuffer memoryImag,
        int N, int D)
    {
        ThrowIfDisposed();
        if (N < 0) throw new ArgumentOutOfRangeException(nameof(N), "N must be >= 0.");
        if (D < 0) throw new ArgumentOutOfRangeException(nameof(D), "D must be >= 0.");
        if (N == 0 || D == 0) return;
        if (keyCodeReal is null || keyCodeImag is null || valPermCodeReal is null || valPermCodeImag is null
            || keyIds is null || valIds is null || memoryReal is null || memoryImag is null)
            throw new ArgumentNullException(nameof(keyCodeReal),
                "All eight GPU buffers must be non-null for SplitComplexHrrBindAccumulate.");
        var kR = AsMetal(keyCodeReal);
        var kI = AsMetal(keyCodeImag);
        var vR = AsMetal(valPermCodeReal);
        var vI = AsMetal(valPermCodeImag);
        var kIds = AsMetal(keyIds);
        var vIds = AsMetal(valIds);
        var mR = AsMetal(memoryReal);
        var mI = AsMetal(memoryImag);
        if (kIds.Size < N || vIds.Size < N)
            throw new ArgumentException(
                $"ID buffers must each hold at least N = {N} elements " +
                $"(got keyIds={kIds.Size}, valIds={vIds.Size}).");
        if (mR.Size < D || mI.Size < D)
            throw new ArgumentException(
                $"Memory buffers must each hold at least D = {D} elements " +
                $"(got memoryReal={mR.Size}, memoryImag={mI.Size}).");
        if (kR.Size < D || kI.Size < D || vR.Size < D || vI.Size < D)
            throw new ArgumentException(
                $"Each codebook buffer must hold at least one full row of D = {D} elements.");
        // Derive vocabulary sizes for in-shader id-bounds checks.
        long nKeysL = kR.Size / D;
        long nValsL = vR.Size / D;
        if (nKeysL > int.MaxValue || nValsL > int.MaxValue)
            throw new ArgumentException("Codebook vocabulary size exceeds int.MaxValue.");
        int nKeys = (int)nKeysL;
        int nVals = (int)nValsL;
        EnsureComplexLibrary();
        var pipeline = GetPipeline("Complex", _complexLibrary, "hrr_bind_accumulate");
        var (tg, tpg) = pipeline.Calculate1DDispatch(D);
        using var enc = _commandQueue.CreateScopedComputeEncoder();
        enc.SetPipelineState(pipeline.Handle);
        enc.SetBuffer(kR, 0);
        enc.SetBuffer(kI, 1);
        enc.SetBuffer(vR, 2);
        enc.SetBuffer(vI, 3);
        enc.SetBuffer(kIds, 4);
        enc.SetBuffer(vIds, 5);
        enc.SetBuffer(mR, 6);
        enc.SetBuffer(mI, 7);
        enc.SetBytes(N, 8);
        enc.SetBytes(D, 9);
        enc.SetBytes(nKeys, 10);
        enc.SetBytes(nVals, 11);
        enc.DispatchThreadgroups(tg, tpg);
    }
}
