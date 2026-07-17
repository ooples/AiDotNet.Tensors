// Copyright (c) AiDotNet. All rights reserved.
// IDirectGpuBackend implementation for VulkanBackend using GPU-accelerated and CPU-fallback operations.

using System;
using System.Runtime.CompilerServices;

namespace AiDotNet.Tensors.Engines.DirectGpu.Vulkan;

public sealed unsafe partial class VulkanBackend
{
    #region Internal Helpers

    private enum ResidentUnaryOp : uint
    {
        Power,
        Abs,
        Exp,
        Exp2,
        Exp10,
        ExpM1,
        Log,
        Log2,
        Log1P,
        Sqrt,
        Sign,
        Gelu,
        Sin,
        Cos,
        Tan,
        Asin,
        Acos,
        Atan,
        Sinh,
        Cosh,
        Asinh,
        Acosh,
        Atanh,
        Reciprocal,
        Cbrt,
        Log10,
        Negate,
        Floor,
        Ceiling,
        Round,
        Truncate,
        LeakyRelu,
        Elu,
        Swish,
        Mish,
        Softplus,
        HardSwish,
        Selu,
        HardSigmoid,
        Clamp,
        NotEqualScalar,
        PowerToDb,
        DbToPower
    }

    private enum ResidentBinaryOp : uint
    {
        Min,
        Max,
        LeakyReluBackward,
        ReluBackward,
        SigmoidBackward,
        TanhBackward,
        SoftplusBackward,
        HardSwishBackward,
        SeluBackward,
        HardSigmoidBackward,
        HardTanhBackward,
        Greater,
        Less,
        Equal,
        Lerp,
        AddScaled,
        ComplexMagnitude,
        ComplexPhase,
        Multiply,
        SwishBackward,
        GeluBackward,
        MishBackward
    }

    private void ResidentUnary(ResidentUnaryOp op, IGpuBuffer input, IGpuBuffer output, int size,
        float p0 = 0f, float p1 = 0f, float p2 = 0f)
    {
        var pushConstants = new uint[]
        {
            (uint)size, (uint)op, FloatBits(p0), FloatBits(p1), FloatBits(p2)
        };
        GlslUnaryOp(VulkanGlslKernels.UnaryElementwise, input, output, size, pushConstants, 5 * sizeof(uint));
    }

    private void ResidentBinary(ResidentBinaryOp op, IGpuBuffer left, IGpuBuffer right, IGpuBuffer output,
        int size, float p0 = 0f, float p1 = 0f, float p2 = 0f)
    {
        var pushConstants = new uint[]
        {
            (uint)size, (uint)op, FloatBits(p0), FloatBits(p1), FloatBits(p2)
        };
        GlslBinaryOp(VulkanGlslKernels.BinaryElementwise, left, right, output, size, pushConstants, 5 * sizeof(uint));
    }

    // Bit conversion helpers compatible with net471 and later
    private static int SingleToInt32BitsCompat(float value)
    {
        return Unsafe.As<float, int>(ref value);
    }

    private static float Int32BitsToSingleCompat(int value)
    {
        return Unsafe.As<int, float>(ref value);
    }

    private static float AsinhCompat(float x)
    {
        return (float)Math.Log(x + Math.Sqrt(x * (double)x + 1.0));
    }

    private static float AcoshCompat(float x)
    {
        return (float)Math.Log(x + Math.Sqrt(x * (double)x - 1.0));
    }

    private static float AtanhCompat(float x)
    {
        return 0.5f * (float)Math.Log((1.0 + x) / (1.0 - x));
    }

    private static float CbrtCompat(float x)
    {
        return (float)Math.Pow(Math.Abs(x), 1.0 / 3.0) * Math.Sign(x);
    }

    private static float Log2Compat(float x)
    {
        return (float)(Math.Log(x) / Math.Log(2.0));
    }

    private static double AcoshDoubleCompat(double x)
    {
        return Math.Log(x + Math.Sqrt(x * x - 1.0));
    }

    private static void ArrayFillCompat(float[] array, float value)
    {
        for (int i = 0; i < array.Length; i++) array[i] = value;
    }

    private static VulkanGpuBuffer AsVulkan(IGpuBuffer buffer)
    {
        if (buffer is not VulkanGpuBuffer vb)
            throw new ArgumentException("Buffer is not a VulkanGpuBuffer.", nameof(buffer));
        return vb;
    }

    private void EnsureInitialized()
    {
        if (!_initialized || _disposed)
            throw new InvalidOperationException("Vulkan backend not initialized.");
    }

    private void UploadToBuffer(float[] data, IGpuBuffer buffer)
    {
        var vb = AsVulkan(buffer);
        if (_transfer is null)
            throw new InvalidOperationException("Vulkan buffer transfer not initialized.");
        // Pad data to buffer.Size to avoid writing stale staging contents for partial uploads
        if (data.Length < vb.Size)
        {
            var padded = new float[vb.Size];
            Array.Copy(data, 0, padded, 0, data.Length);
            vb.Staging.WriteData(padded);
        }
        else
        {
            vb.Staging.WriteData(data);
        }
        _transfer.CopyToDevice(vb.Staging, vb.Storage);
    }

    private void GpuBinaryOp(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int size, VulkanKernelType kernelType)
    {
        EnsureInitialized();
        if (size < 0)
            throw new ArgumentOutOfRangeException(nameof(size), "Size must be non-negative.");
        if (size == 0) return;
        if (size > A.Size)
            throw new ArgumentOutOfRangeException(nameof(size), $"Size ({size}) exceeds first input buffer length ({A.Size}).");
        if (size > B.Size)
            throw new ArgumentOutOfRangeException(nameof(size), $"Size ({size}) exceeds second input buffer length ({B.Size}).");
        if (size > C.Size)
            throw new ArgumentOutOfRangeException(nameof(size), $"Size ({size}) exceeds output buffer length ({C.Size}).");

        var vbA = AsVulkan(A);
        var vbB = AsVulkan(B);
        var vbC = AsVulkan(C);

        var pipeline = GetOrCreatePipeline(kernelType, 3, sizeof(uint));
        if (pipeline is null)
            throw new InvalidOperationException($"Failed to create pipeline for {kernelType}.");

        var threadRes = _device.AcquireThreadResources();
        lock (_computeLock)
        {
            pipeline.UpdateDescriptorSet(vbA.Storage, vbB.Storage, vbC.Storage);
            RecordAndExecuteComputeUnlocked(pipeline, size, threadRes);
        }
    }

    private void GpuUnaryOp(IGpuBuffer A, IGpuBuffer B, int size, VulkanKernelType kernelType)
    {
        EnsureInitialized();
        if (size < 0)
            throw new ArgumentOutOfRangeException(nameof(size), "Size must be non-negative.");
        if (size == 0) return;
        if (size > A.Size)
            throw new ArgumentOutOfRangeException(nameof(size), $"Size ({size}) exceeds input buffer length ({A.Size}).");
        if (size > B.Size)
            throw new ArgumentOutOfRangeException(nameof(size), $"Size ({size}) exceeds output buffer length ({B.Size}).");

        var vbA = AsVulkan(A);
        var vbB = AsVulkan(B);

        var pipeline = GetOrCreatePipeline(kernelType, 2, sizeof(uint));
        if (pipeline is null)
            throw new InvalidOperationException($"Failed to create pipeline for {kernelType}.");

        var threadRes = _device.AcquireThreadResources();
        lock (_computeLock)
        {
            pipeline.UpdateDescriptorSet(vbA.Storage, vbB.Storage);
            RecordAndExecuteComputeUnlocked(pipeline, size, threadRes);
        }
    }

    private float ResidentScalarReduce(IGpuBuffer input, int size, uint operation)
    {
        EnsureInitialized();
        if (size <= 0) throw new ArgumentOutOfRangeException(nameof(size), "Size must be positive.");
        if (size > input.Size)
            throw new ArgumentOutOfRangeException(nameof(size), $"Size ({size}) exceeds buffer length ({input.Size}).");
        using var result = AllocateBuffer(1);
        GlslUnaryOp(VulkanGlslKernels.ScalarReduce, input, result, 1,
            new uint[] { (uint)size, operation }, 2 * sizeof(uint));
        return DownloadBuffer(result)[0];
    }

    #endregion

    #region Memory Management

    public IGpuBuffer AllocateBuffer(float[] data)
    {
        EnsureInitialized();
        if (data is null)
            throw new ArgumentNullException(nameof(data));
        if (_transfer is null)
            throw new InvalidOperationException("Vulkan buffer transfer not initialized.");
        // Issue #285: per-allocation cap check before VkBuffer creation.
        GpuBufferSizeGuard.EnsureFits("Vulkan", (long)data.Length * sizeof(float), MaxBufferAllocBytes, DeviceName);
        return VulkanGpuBuffer.Create(data, _transfer);
    }

    public IGpuBuffer AllocateBuffer(int size)
    {
        EnsureInitialized();
        if (size <= 0)
            throw new ArgumentOutOfRangeException(nameof(size), "Buffer size must be positive.");
        GpuBufferSizeGuard.EnsureFits("Vulkan", (long)size * sizeof(float), MaxBufferAllocBytes, DeviceName);
        return VulkanGpuBuffer.Create(size);
    }

    public float[] DownloadBuffer(IGpuBuffer buffer)
    {
        GpuLaunchProbe.OnReadback((long)buffer.Size * sizeof(float));
        EnsureInitialized();
        var vb = AsVulkan(buffer);
        if (_transfer is null)
            throw new InvalidOperationException("Vulkan buffer transfer not initialized.");
        _transfer.CopyFromDevice(vb.Storage, vb.Staging);
        var result = new float[vb.Size];
        vb.Staging.ReadData(result);
        return result;
    }

    public void DownloadBuffer(IGpuBuffer buffer, float[] destination)
    {
        GpuLaunchProbe.OnReadback((long)buffer.Size * sizeof(float));
        EnsureInitialized();
        if (destination.Length < buffer.Size)
            throw new ArgumentException($"Destination array length ({destination.Length}) is less than buffer size ({buffer.Size}).", nameof(destination));
        var vb = AsVulkan(buffer);
        if (_transfer is null)
            throw new InvalidOperationException("Vulkan buffer transfer not initialized.");
        _transfer.CopyFromDevice(vb.Storage, vb.Staging);
        vb.Staging.ReadData(destination);
    }

    public byte[] DownloadByteBuffer(IGpuBuffer buffer, int byteCount)
    {
        EnsureInitialized();
        if (byteCount < 0)
            throw new ArgumentOutOfRangeException(nameof(byteCount), "Byte count must be non-negative.");
        var vb = AsVulkan(buffer);
        if (byteCount > vb.SizeInBytes)
            throw new ArgumentException($"Requested byte count ({byteCount}) exceeds buffer capacity ({vb.SizeInBytes}).", nameof(byteCount));
        if (_transfer is null)
            throw new InvalidOperationException("Vulkan buffer transfer not initialized.");

        var result = new byte[byteCount];
        if (byteCount == 0)
            return result;

        _transfer.CopyFromDevice(vb.Storage, vb.Staging);
        vb.Staging.ReadRawBytes(result);
        return result;
    }

    public void UploadByteBuffer(IGpuBuffer buffer, byte[] data)
    {
        EnsureInitialized();
        if (data is null)
            throw new ArgumentNullException(nameof(data));
        var vb = AsVulkan(buffer);
        if (data.LongLength > vb.SizeInBytes)
            throw new ArgumentException($"Host data ({data.Length} bytes) exceeds buffer capacity ({vb.SizeInBytes} bytes).", nameof(data));
        if (_transfer is null)
            throw new InvalidOperationException("Vulkan buffer transfer not initialized.");
        if (data.Length == 0)
            return;

        vb.Staging.WriteRawData<byte>(data);
        _transfer.CopyToDevice(vb.Staging, vb.Storage);
    }

    public void Copy(IGpuBuffer source, int srcOffset, IGpuBuffer destination, int destOffset, int size)
    {
        EnsureInitialized();
        if (size < 0)
            throw new ArgumentOutOfRangeException(nameof(size), "Size must be non-negative.");
        if (size == 0) return;
        if (srcOffset < 0)
            throw new ArgumentOutOfRangeException(nameof(srcOffset), "Source offset must be non-negative.");
        if (destOffset < 0)
            throw new ArgumentOutOfRangeException(nameof(destOffset), "Destination offset must be non-negative.");
        if (srcOffset + size > source.Size)
            throw new ArgumentOutOfRangeException(nameof(size), $"Source offset ({srcOffset}) + size ({size}) exceeds source buffer length ({source.Size}).");
        if (destOffset + size > destination.Size)
            throw new ArgumentOutOfRangeException(nameof(size), $"Destination offset ({destOffset}) + size ({size}) exceeds destination buffer length ({destination.Size}).");

        if (_transfer is null)
            throw new InvalidOperationException("Vulkan transfer manager not initialized.");
        if (source is not VulkanGpuBuffer src || destination is not VulkanGpuBuffer dst)
            throw new ArgumentException("Buffers must be VulkanGpuBuffer instances.");

        GpuLaunchProbe.OnLaunch();
        _transfer.CopyDeviceToDevice(
            src.Storage, checked((ulong)srcOffset * sizeof(float)),
            dst.Storage, checked((ulong)destOffset * sizeof(float)),
            checked((ulong)size * sizeof(float)));
    }

    public IGpuBuffer AllocateIntBuffer(int size)
    {
        EnsureInitialized();
        if (size <= 0)
            throw new ArgumentOutOfRangeException(nameof(size), "Int buffer size must be positive.");
        GpuBufferSizeGuard.EnsureFits("Vulkan", (long)size * sizeof(int), MaxBufferAllocBytes, DeviceName);
        return VulkanGpuBuffer.Create(size);
    }

    public IGpuBuffer AllocateIntBuffer(int[] data)
    {
        EnsureInitialized();
        if (data is null)
            throw new ArgumentNullException(nameof(data));
        GpuBufferSizeGuard.EnsureFits("Vulkan", (long)data.Length * sizeof(int), MaxBufferAllocBytes, DeviceName);
        var floatData = new float[data.Length];
        for (int i = 0; i < data.Length; i++)
            floatData[i] = Int32BitsToSingleCompat(data[i]);
        if (_transfer is null)
            throw new InvalidOperationException("Vulkan buffer transfer not initialized.");
        return VulkanGpuBuffer.Create(floatData, _transfer);
    }

    /// <summary>
    /// Allocates a GPU buffer for raw byte data.
    /// </summary>
    /// <param name="size">The number of bytes to allocate.</param>
    /// <returns>
    /// A GPU buffer whose <see cref="IGpuBuffer.Size"/> is the number of float elements
    /// needed to hold <paramref name="size"/> bytes (i.e., <c>ceil(size / 4)</c>), and whose
    /// <see cref="IGpuBuffer.SizeInBytes"/> is <c>Size * 4</c>.
    /// Callers must track the original byte count separately for correct data interpretation.
    /// </returns>
    /// <remarks>
    /// Vulkan storage buffers are float-typed, so byte data is packed into float elements.
    /// For example, <c>AllocateByteBuffer(10)</c> returns a buffer with <c>Size == 3</c>
    /// (3 floats = 12 bytes, enough for the requested 10 bytes).
    /// </remarks>
    public IGpuBuffer AllocateByteBuffer(int size)
    {
        EnsureInitialized();
        if (size <= 0)
            throw new ArgumentOutOfRangeException(nameof(size), "Byte buffer size must be positive.");
        // Vulkan storage buffers are float32-typed, so the actual allocation
        // is `floatCount * sizeof(float)` bytes. Round up in long-space
        // first to avoid overflow when `size` is near int.MaxValue, then
        // check the actual byte count against the cap.
        long floatCountLong = ((long)size + sizeof(float) - 1) / sizeof(float);
        long actualBytes = floatCountLong * sizeof(float);
        GpuBufferSizeGuard.EnsureFits("Vulkan", actualBytes, MaxBufferAllocBytes, DeviceName);
        return VulkanGpuBuffer.Create(checked((int)floatCountLong));
    }

    #endregion

    #region GEMM Operations

    public void Gemm(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int M, int N, int K, float alpha = 1.0f, float beta = 0.0f)
    {
        EnsureInitialized();
        ValidateGemmBuffers(A, B, C, M, N, K, false);
        if (!TryGlslGemmFp32(A, B, C, M, N, K, alpha, beta))
            throw new InvalidOperationException("Vulkan GEMM pipeline creation failed.");
    }

    /// <summary>
    /// Runs C = A·B (row-major, M×K · K×N) on the GPU via the GLSL GEMM kernel.
    /// Returns false (so the caller can fall back to the managed loop) when the
    /// dimensions are non-positive or libshaderc isn't available to compile the
    /// shader. One invocation per output element; push constants carry {M, N, K}.
    /// </summary>
    private bool TryGlslGemmFp32(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int M, int N, int K,
        float alpha, float beta)
    {
        if (M <= 0 || N <= 0 || K <= 0)
            return false;

        var pipeline = GetOrCreateGlslPipeline(VulkanGemmKernels.GemmFp32, 3, 5 * sizeof(uint));
        if (pipeline is null)
            return false;

        var vbA = AsVulkan(A);
        var vbB = AsVulkan(B);
        var vbC = AsVulkan(C);
        var pushConstants = new uint[] { (uint)M, (uint)N, (uint)K, FloatBits(alpha), FloatBits(beta) };
        var threadRes = _device.AcquireThreadResources();
        lock (_computeLock)
        {
            pipeline.UpdateDescriptorSet(vbA.Storage, vbB.Storage, vbC.Storage);
            RecordAndExecuteWithPushData(pipeline, M * N, pushConstants, 5 * sizeof(uint), threadRes);
        }

        return true;
    }

    /// <summary>Runs C = alpha * A * B-transpose + beta * C entirely on the Vulkan device.</summary>
    private bool TryGlslGemmTransposedFp32(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C,
        int M, int N, int K, float alpha, float beta)
    {
        var pipeline = GetOrCreateGlslPipeline(
            VulkanGemmKernels.GemmFp32Transposed, 3, 5 * sizeof(uint));
        if (pipeline is null)
            return false;

        var pushConstants = new uint[]
        {
            (uint)M,
            (uint)N,
            (uint)K,
            unchecked((uint)SingleToInt32BitsCompat(alpha)),
            unchecked((uint)SingleToInt32BitsCompat(beta))
        };
        var threadRes = _device.AcquireThreadResources();
        lock (_computeLock)
        {
            pipeline.UpdateDescriptorSet(AsVulkan(A).Storage, AsVulkan(B).Storage, AsVulkan(C).Storage);
            RecordAndExecuteWithPushData(pipeline, M * N, pushConstants, 5 * sizeof(uint), threadRes);
        }

        return true;
    }

    public IGpuBuffer MatMul(IGpuBuffer A, IGpuBuffer B, int M, int N, int K)
    {
        var result = AllocateBuffer(M * N);
        Gemm(A, B, result, M, N, K);
        return result;
    }

    /// <inheritdoc/>
    public void MatMulTransposed(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int M, int N, int K, float alpha = 1.0f, float beta = 0.0f)
    {
        EnsureInitialized();
        // Shape validation up-front — without this, mismatched shapes
        // would IndexOutOfRange inside the inner loop instead of failing
        // with a clear ArgumentException. CUDA backend validates the same
        // contract via ValidateGemmArgs.
        if (M <= 0 || N <= 0 || K <= 0)
            throw new ArgumentOutOfRangeException(nameof(M), "Matrix dimensions M, N, K must all be positive.");
        if ((long)A.Size < (long)M * K)
            throw new ArgumentException($"A.Size {A.Size} < M*K = {(long)M * K}.", nameof(A));
        if ((long)B.Size < (long)N * K)
            throw new ArgumentException($"B.Size {B.Size} < N*K = {(long)N * K}.", nameof(B));
        if ((long)C.Size < (long)M * N)
            throw new ArgumentException($"C.Size {C.Size} < M*N = {(long)M * N}.", nameof(C));

        if (!TryGlslGemmTransposedFp32(A, B, C, M, N, K, alpha, beta))
            throw new InvalidOperationException("Vulkan transposed GEMM pipeline creation failed.");
    }

    public void BatchedGemm(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int M, int N, int K, int batchCount, float alpha = 1.0f, float beta = 0.0f)
    {
        EnsureInitialized();
        if (batchCount <= 0) throw new ArgumentOutOfRangeException(nameof(batchCount));
        ValidateGemmBuffers(A, B, C, M, N, K, false, batchCount);
        var pushConstants = new uint[]
        {
            (uint)M, (uint)N, (uint)K, (uint)batchCount, FloatBits(alpha), FloatBits(beta)
        };
        GlslBinaryOp(VulkanGemmKernels.BatchedGemmFp32, A, B, C, batchCount * M * N,
            pushConstants, 6 * sizeof(uint));
    }

    private static void ValidateGemmBuffers(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C,
        int M, int N, int K, bool transposeB, int batchCount = 1)
    {
        if (M <= 0 || N <= 0 || K <= 0)
            throw new ArgumentOutOfRangeException(nameof(M), "Matrix dimensions M, N, K must all be positive.");
        long aRequired = (long)batchCount * M * K;
        long bRequired = (long)batchCount * (transposeB ? N * K : K * N);
        long cRequired = (long)batchCount * M * N;
        if (A.Size < aRequired) throw new ArgumentException($"A.Size {A.Size} < required {aRequired}.", nameof(A));
        if (B.Size < bRequired) throw new ArgumentException($"B.Size {B.Size} < required {bRequired}.", nameof(B));
        if (C.Size < cRequired) throw new ArgumentException($"C.Size {C.Size} < required {cRequired}.", nameof(C));
    }

    #endregion

    #region Fused GEMM Operations

    private IGpuBuffer GemmBiasActivation(IGpuBuffer A, IGpuBuffer B, IGpuBuffer bias, int M, int N, int K,
        string shader, float? alpha = null)
    {
        EnsureInitialized();
        if (M <= 0 || N <= 0 || K <= 0) throw new ArgumentOutOfRangeException(nameof(M));
        var result = AllocateBuffer(M * N);
        var pushConstants = alpha.HasValue
            ? new uint[] { (uint)M, (uint)K, (uint)N, FloatBits(alpha.Value) }
            : new uint[] { (uint)M, (uint)K, (uint)N };
        GlslQuadOp(shader, A, B, bias, result, M * N, pushConstants,
            (uint)(pushConstants.Length * sizeof(uint)));
        return result;
    }

    public IGpuBuffer GemmBiasRelu(IGpuBuffer A, IGpuBuffer B, IGpuBuffer bias, int M, int N, int K)
        => GemmBiasActivation(A, B, bias, M, N, K, VulkanGlslKernels.FusedLinearReLU);

    public IGpuBuffer GemmBiasGelu(IGpuBuffer A, IGpuBuffer B, IGpuBuffer bias, int M, int N, int K)
        => GemmBiasActivation(A, B, bias, M, N, K, VulkanGlslKernels.FusedLinearGELU);

    public IGpuBuffer GemmBiasSigmoid(IGpuBuffer A, IGpuBuffer B, IGpuBuffer bias, int M, int N, int K)
        => GemmBiasActivation(A, B, bias, M, N, K, VulkanGlslKernels.FusedLinearSigmoid);

    public IGpuBuffer GemmBiasTanh(IGpuBuffer A, IGpuBuffer B, IGpuBuffer bias, int M, int N, int K)
        => GemmBiasActivation(A, B, bias, M, N, K, VulkanGlslKernels.FusedLinearTanh);

    public IGpuBuffer GemmBias(IGpuBuffer A, IGpuBuffer B, IGpuBuffer bias, int M, int N, int K)
        => GemmBiasActivation(A, B, bias, M, N, K, VulkanGlslKernels.FusedLinearIdentity);

    public IGpuBuffer GemmBiasSwish(IGpuBuffer A, IGpuBuffer B, IGpuBuffer bias, int M, int N, int K)
        => GemmBiasActivation(A, B, bias, M, N, K, VulkanGlslKernels.FusedLinearSwish);

    public IGpuBuffer GemmBiasLeakyRelu(IGpuBuffer A, IGpuBuffer B, IGpuBuffer bias, int M, int N, int K, float alpha = 0.01f)
        => GemmBiasActivation(A, B, bias, M, N, K, VulkanGlslKernels.FusedLinearLeakyRelu, alpha);

    #endregion

    #region Broadcast Operations

    public void BiasAdd(IGpuBuffer A, IGpuBuffer bias, IGpuBuffer C, int M, int N)
    {
        GlslBinaryOp(VulkanGlslKernels.BiasAdd, A, bias, C, M * N,
            new uint[] { (uint)(M * N), (uint)N }, 2 * sizeof(uint));
    }

    public void Conv2DBiasAdd(IGpuBuffer output, IGpuBuffer bias, int batch, int channels, int spatialSize)
    {
        GlslUnaryOp(VulkanGlslKernels.Conv2DBiasAdd, output, bias, batch * channels * spatialSize,
            new uint[] { (uint)(batch * channels * spatialSize), (uint)channels, (uint)spatialSize },
            3 * sizeof(uint));
    }

    #endregion

    #region Element-wise Operations (GPU-accelerated)

    public void Add(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int size)
        => GpuBinaryOp(A, B, C, size, VulkanKernelType.VectorAdd);
    public void AddRelu(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int size) { Add(A, B, C, size); Relu(C, C, size); }
    public void AddSigmoid(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int size) { Add(A, B, C, size); Sigmoid(C, C, size); }
    public void AddGelu(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int size) { Add(A, B, C, size); Gelu(C, C, size); }

    public void Subtract(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int size)
        => GpuBinaryOp(A, B, C, size, VulkanKernelType.VectorSubtract);

    public void Multiply(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int size)
        => GpuBinaryOp(A, B, C, size, VulkanKernelType.VectorMultiply);

    public void Divide(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int size)
        => GpuBinaryOp(A, B, C, size, VulkanKernelType.VectorDivide);

    public void Relu(IGpuBuffer A, IGpuBuffer B, int size)
        => GpuUnaryOp(A, B, size, VulkanKernelType.ReLU);

    public void Sigmoid(IGpuBuffer A, IGpuBuffer B, int size)
        => GpuUnaryOp(A, B, size, VulkanKernelType.Sigmoid);

    public void Tanh(IGpuBuffer A, IGpuBuffer B, int size)
        => GpuUnaryOp(A, B, size, VulkanKernelType.Tanh);

    public void DotProduct(IGpuBuffer a, IGpuBuffer b, IGpuBuffer result, int size)
    {
        EnsureInitialized();
        if (size <= 0)
        {
            Scale(result, result, 0f, Math.Max(1, result.Size));
            return;
        }

        if (size > a.Size) throw new ArgumentOutOfRangeException(nameof(size), $"Size ({size}) exceeds buffer A length ({a.Size}).");
        if (size > b.Size) throw new ArgumentOutOfRangeException(nameof(size), $"Size ({size}) exceeds buffer B length ({b.Size}).");

        // Zero result first to ensure deterministic output
        Scale(result, result, 0f, Math.Max(1, result.Size));

        // Direct GPU dispatch using pre-compiled DotProduct SPIR-V kernel
        var vbA = AsVulkan(a);
        var vbB = AsVulkan(b);
        var vbR = AsVulkan(result);

        var pipeline = GetOrCreatePipeline(VulkanKernelType.DotProduct, 3, sizeof(uint));
        if (pipeline is null)
            throw new InvalidOperationException("Failed to create DotProduct pipeline.");

        var threadRes = _device.AcquireThreadResources();
        lock (_computeLock)
        {
            pipeline.UpdateDescriptorSet(vbA.Storage, vbB.Storage, vbR.Storage);
            RecordAndExecuteComputeUnlocked(pipeline, size, threadRes);
        }
    }

    public void StridedDotProduct(IGpuBuffer a, IGpuBuffer b, IGpuBuffer result,
        int aSize, int bSize, int bOffset, int bStride)
    {
        EnsureInitialized();
        if (aSize <= 0) return;

        GlslBinaryOp(VulkanGlslKernels.StridedDot, a, b, result, 1,
            new uint[] { (uint)aSize, (uint)bSize, unchecked((uint)bOffset), unchecked((uint)bStride) },
            4 * sizeof(uint));
    }

    public void BatchedDotProduct(IGpuBuffer a, IGpuBuffer b, IGpuBuffer result,
        int batchSize, int vecSize)
    {
        EnsureInitialized();
        if (batchSize <= 0 || vecSize <= 0) return;
        if ((long)batchSize * vecSize > a.Size) throw new ArgumentOutOfRangeException(nameof(batchSize), $"batchSize*vecSize ({(long)batchSize * vecSize}) exceeds buffer A length ({a.Size}).");
        if ((long)batchSize * vecSize > b.Size) throw new ArgumentOutOfRangeException(nameof(batchSize), $"batchSize*vecSize ({(long)batchSize * vecSize}) exceeds buffer B length ({b.Size}).");

        // Element-wise multiply all batches, then reduce each batch
        using var temp = AllocateBuffer(batchSize * vecSize);
        Multiply(a, b, temp, batchSize * vecSize);
        // Synchronize to ensure Multiply completes before SumAxis reads temp
        Synchronize();
        SumAxis(temp, result, batchSize, vecSize);
    }

    public void Scale(IGpuBuffer A, IGpuBuffer B, float scalar, int size)
    {
        EnsureInitialized();
        var vbA = AsVulkan(A);
        var vbB = AsVulkan(B);
        var pipeline = GetOrCreatePipeline(VulkanKernelType.ScalarMultiply, 2, sizeof(uint) + sizeof(float));
        if (pipeline is null)
            throw new InvalidOperationException("Failed to create scalar multiply pipeline.");

        var threadRes = _device.AcquireThreadResources();
        lock (_computeLock)
        {
            pipeline.UpdateDescriptorSet(vbA.Storage, vbB.Storage);
            RecordAndExecuteComputeWithScalarUnlocked(pipeline, size, scalar, threadRes);
        }
    }

    #endregion

    #region Element-wise Operations

    public void Min(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int size)
        => ResidentBinary(ResidentBinaryOp.Min, A, B, C, size);

    public void Max(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int size)
        => ResidentBinary(ResidentBinaryOp.Max, A, B, C, size);

    private const string StridedGatherGlsl = @"
#version 450
layout(local_size_x = 256) in;
layout(set = 0, binding = 0) readonly buffer Src { float src[]; };
layout(set = 0, binding = 1) writeonly buffer Dst { float dst[]; };
layout(push_constant) uniform Params { uint offset; uint stride; uint count; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx < count) { dst[idx] = src[offset + idx * stride]; }
}";

    private const string StridedScatterGlsl = @"
#version 450
layout(local_size_x = 256) in;
layout(set = 0, binding = 0) readonly buffer Src { float src[]; };
layout(set = 0, binding = 1) buffer Dst { float dst[]; };
layout(push_constant) uniform Params { uint offset; uint stride; uint count; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx < count) { dst[offset + idx * stride] = src[idx]; }
}";

    public void StridedGather(IGpuBuffer src, IGpuBuffer dst, int offset, int stride, int count)
    {
        EnsureInitialized();
        var pipeline = GetOrCreateGlslPipeline(StridedGatherGlsl, 2, 3 * sizeof(uint));
        if (pipeline is not null)
        {
            var vbSrc = AsVulkan(src);
            var vbDst = AsVulkan(dst);
            var threadRes = _device.AcquireThreadResources();
            lock (_computeLock)
            {
                pipeline.UpdateDescriptorSet(vbSrc.Storage, vbDst.Storage);
                var pushConstants = new uint[] { (uint)offset, (uint)stride, (uint)count };
                RecordAndExecuteWithPushData(pipeline, count, pushConstants, 3 * sizeof(uint), threadRes);
            }
            return;
        }

        throw new InvalidOperationException("Vulkan strided-gather pipeline creation failed.");
    }

    public void StridedScatter(IGpuBuffer src, IGpuBuffer dst, int offset, int stride, int count)
    {
        EnsureInitialized();
        var pipeline = GetOrCreateGlslPipeline(StridedScatterGlsl, 2, 3 * sizeof(uint));
        if (pipeline is not null)
        {
            var vbSrc = AsVulkan(src);
            var vbDst = AsVulkan(dst);
            var threadRes = _device.AcquireThreadResources();
            lock (_computeLock)
            {
                pipeline.UpdateDescriptorSet(vbSrc.Storage, vbDst.Storage);
                var pushConstants = new uint[] { (uint)offset, (uint)stride, (uint)count };
                RecordAndExecuteWithPushData(pipeline, count, pushConstants, 3 * sizeof(uint), threadRes);
            }
            return;
        }

        throw new InvalidOperationException("Vulkan strided-scatter pipeline creation failed.");
    }

    public void Power(IGpuBuffer A, IGpuBuffer B, float exponent, int size)
        => ResidentUnary(ResidentUnaryOp.Power, A, B, size, exponent);

    public void Abs(IGpuBuffer A, IGpuBuffer B, int size)
        => ResidentUnary(ResidentUnaryOp.Abs, A, B, size);

    public void Exp(IGpuBuffer A, IGpuBuffer B, int size)
        => ResidentUnary(ResidentUnaryOp.Exp, A, B, size);

    public void Exp2(IGpuBuffer A, IGpuBuffer B, int size)
        => ResidentUnary(ResidentUnaryOp.Exp2, A, B, size);

    public void Exp10(IGpuBuffer A, IGpuBuffer B, int size)
        => ResidentUnary(ResidentUnaryOp.Exp10, A, B, size);

    public void ExpM1(IGpuBuffer A, IGpuBuffer B, int size)
        => ResidentUnary(ResidentUnaryOp.ExpM1, A, B, size);

    public void Log(IGpuBuffer A, IGpuBuffer B, int size)
        => ResidentUnary(ResidentUnaryOp.Log, A, B, size);

    public void Log2(IGpuBuffer A, IGpuBuffer B, int size)
        => ResidentUnary(ResidentUnaryOp.Log2, A, B, size);

    public void Log1P(IGpuBuffer A, IGpuBuffer B, int size)
        => ResidentUnary(ResidentUnaryOp.Log1P, A, B, size);

    public void Sqrt(IGpuBuffer A, IGpuBuffer B, int size)
        => ResidentUnary(ResidentUnaryOp.Sqrt, A, B, size);

    public void Sign(IGpuBuffer A, IGpuBuffer B, int size)
        => ResidentUnary(ResidentUnaryOp.Sign, A, B, size);

    public void Gelu(IGpuBuffer A, IGpuBuffer B, int size)
        => ResidentUnary(ResidentUnaryOp.Gelu, A, B, size);

    public void Softmax(IGpuBuffer A, IGpuBuffer B, int batchSize, int features)
    {
        if (batchSize <= 0 || features <= 0) return;
        GlslUnaryOp(VulkanGlslKernels.SoftmaxRows, A, B, batchSize,
            new uint[] { (uint)batchSize, (uint)features }, 2 * sizeof(uint));
    }

    #endregion

    #region Capsule Operations

    public void Squash(IGpuBuffer input, IGpuBuffer output, int numCapsules, int capsuleDim, float epsilon)
    {
        GlslUnaryOp(VulkanGlslKernels.CapsuleSquash, input, output, numCapsules,
            new uint[] { (uint)numCapsules, (uint)capsuleDim, FloatBits(epsilon) }, 3 * sizeof(uint));
    }

    public void SquashBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradInput, int numCapsules, int capsuleDim, float epsilon)
    {
        GlslBinaryOp(VulkanGlslKernels.CapsuleSquashBackward, gradOutput, input, gradInput, numCapsules,
            new uint[] { (uint)numCapsules, (uint)capsuleDim, FloatBits(epsilon) }, 3 * sizeof(uint));
    }

    public void CapsulePredictions(IGpuBuffer input, IGpuBuffer weights, IGpuBuffer output,
        int batchSize, int inputCapsules, int inputDim, int outputCapsules, int outputDim)
    {
        int total = checked(batchSize * inputCapsules * outputCapsules * outputDim);
        GlslBinaryOp(VulkanGlslKernels.CapsulePredictions, input, weights, output, total,
            new uint[] { (uint)batchSize, (uint)inputCapsules, (uint)inputDim, (uint)outputCapsules, (uint)outputDim },
            5 * sizeof(uint));
    }

    public void CapsuleTransform(IGpuBuffer input, IGpuBuffer weights, IGpuBuffer output,
        int batchSize, int inputCapsules, int inputDim, int numCapsules, int capsuleDim)
        => CapsulePredictions(input, weights, output, batchSize, inputCapsules, inputDim, numCapsules, capsuleDim);

    public void CapsuleWeightedSum(IGpuBuffer coupling, IGpuBuffer predictions, IGpuBuffer output,
        int batchSize, int inputCapsules, int outputCapsules, int capsuleDim)
    {
        int total = checked(batchSize * outputCapsules * capsuleDim);
        GlslBinaryOp(VulkanGlslKernels.CapsuleWeightedSum, coupling, predictions, output, total,
            new uint[] { (uint)batchSize, (uint)inputCapsules, (uint)outputCapsules, (uint)capsuleDim },
            4 * sizeof(uint));
    }

    public void CapsuleAgreement(IGpuBuffer predictions, IGpuBuffer output, IGpuBuffer agreement,
        int batchSize, int inputCapsules, int outputCapsules, int capsuleDim)
    {
        int total = checked(batchSize * inputCapsules * outputCapsules);
        GlslBinaryOp(VulkanGlslKernels.CapsuleAgreement, predictions, output, agreement, total,
            new uint[] { (uint)batchSize, (uint)inputCapsules, (uint)outputCapsules, (uint)capsuleDim },
            4 * sizeof(uint));
    }

    public void TileBatch(IGpuBuffer input, IGpuBuffer output, int repeats, int innerSize)
    {
        GlslUnaryOp(VulkanGlslKernels.TileBatch, input, output, repeats * innerSize,
            new uint[] { (uint)repeats, (uint)innerSize }, 2 * sizeof(uint));
    }

    public void TileAxis(IGpuBuffer input, IGpuBuffer output, int outerSize, int axisSize, int innerSize, int repeats)
    {
        int total = checked(outerSize * axisSize * repeats * innerSize);
        GlslUnaryOp(VulkanGlslKernels.TileAxisGlsl, input, output, total,
            new uint[] { (uint)outerSize, (uint)axisSize, (uint)innerSize, (uint)repeats },
            4 * sizeof(uint));
    }

    #endregion

    #region Trigonometric Operations

    public void Sin(IGpuBuffer A, IGpuBuffer B, int size) => ResidentUnary(ResidentUnaryOp.Sin, A, B, size);
    public void Cos(IGpuBuffer A, IGpuBuffer B, int size) => ResidentUnary(ResidentUnaryOp.Cos, A, B, size);
    public void Tan(IGpuBuffer A, IGpuBuffer B, int size) => ResidentUnary(ResidentUnaryOp.Tan, A, B, size);
    public void Asin(IGpuBuffer A, IGpuBuffer B, int size) => ResidentUnary(ResidentUnaryOp.Asin, A, B, size);
    public void Acos(IGpuBuffer A, IGpuBuffer B, int size) => ResidentUnary(ResidentUnaryOp.Acos, A, B, size);
    public void Atan(IGpuBuffer A, IGpuBuffer B, int size) => ResidentUnary(ResidentUnaryOp.Atan, A, B, size);

    #endregion

    #region Hyperbolic Operations

    public void Sinh(IGpuBuffer A, IGpuBuffer B, int size) => ResidentUnary(ResidentUnaryOp.Sinh, A, B, size);
    public void Cosh(IGpuBuffer A, IGpuBuffer B, int size) => ResidentUnary(ResidentUnaryOp.Cosh, A, B, size);
    public void Asinh(IGpuBuffer A, IGpuBuffer B, int size) => ResidentUnary(ResidentUnaryOp.Asinh, A, B, size);
    public void Acosh(IGpuBuffer A, IGpuBuffer B, int size) => ResidentUnary(ResidentUnaryOp.Acosh, A, B, size);
    public void Atanh(IGpuBuffer A, IGpuBuffer B, int size) => ResidentUnary(ResidentUnaryOp.Atanh, A, B, size);

    #endregion

    #region Additional Unary Operations

    public void Reciprocal(IGpuBuffer A, IGpuBuffer B, int size) => ResidentUnary(ResidentUnaryOp.Reciprocal, A, B, size);
    public void Cbrt(IGpuBuffer A, IGpuBuffer B, int size) => ResidentUnary(ResidentUnaryOp.Cbrt, A, B, size);
    public void Log10(IGpuBuffer A, IGpuBuffer B, int size) => ResidentUnary(ResidentUnaryOp.Log10, A, B, size);
    public void Negate(IGpuBuffer A, IGpuBuffer B, int size) => ResidentUnary(ResidentUnaryOp.Negate, A, B, size);
    public void Floor(IGpuBuffer A, IGpuBuffer B, int size) => ResidentUnary(ResidentUnaryOp.Floor, A, B, size);
    public void Ceiling(IGpuBuffer A, IGpuBuffer B, int size) => ResidentUnary(ResidentUnaryOp.Ceiling, A, B, size);
    public void Round(IGpuBuffer A, IGpuBuffer B, int size) => ResidentUnary(ResidentUnaryOp.Round, A, B, size);
    public void Truncate(IGpuBuffer A, IGpuBuffer B, int size) => ResidentUnary(ResidentUnaryOp.Truncate, A, B, size);

    #endregion

    #region Reduction Operations

    public float Sum(IGpuBuffer A, int size) => ResidentScalarReduce(A, size, 0);

    public float Max(IGpuBuffer A, int size) => ResidentScalarReduce(A, size, 1);
    public float Min(IGpuBuffer A, int size) => ResidentScalarReduce(A, size, 2);

    public void SumAxis(IGpuBuffer A, IGpuBuffer B, int outerSize, int reduceSize)
    {
        GlslUnaryOp(VulkanGlslKernels.SumAxis, A, B, outerSize,
            new uint[] { (uint)outerSize, (uint)reduceSize }, 2 * sizeof(uint));
    }

    #endregion

    #region Fused Operations

    public void Lerp(IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, float t, int size)
        => ResidentBinary(ResidentBinaryOp.Lerp, a, b, output, size, t);

    public void AddScaled(IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, float scaleA, float scaleB, int size)
        => ResidentBinary(ResidentBinaryOp.AddScaled, a, b, output, size, scaleA, scaleB);

    public float StdDev(IGpuBuffer input, int size)
    {
        return ResidentScalarReduce(input, size, 3);
    }

    #endregion

    #region Synchronization

    public void Synchronize()
    {
        if (_initialized && !_disposed)
            _device.WaitIdle();
    }

    #endregion

    #region Transpose and Reshape

    public void Transpose(IGpuBuffer A, IGpuBuffer B, int rows, int cols)
    {
        GlslUnaryOp(VulkanGlslKernels.Transpose, A, B, rows * cols,
            new uint[] { 1, (uint)rows, (uint)cols }, 3 * sizeof(uint));
    }

    public void BatchedTranspose(IGpuBuffer A, IGpuBuffer B, int batch, int rows, int cols)
    {
        GlslUnaryOp(VulkanGlslKernels.Transpose, A, B, batch * rows * cols,
            new uint[] { (uint)batch, (uint)rows, (uint)cols }, 3 * sizeof(uint));
    }

    public void Permute(IGpuBuffer input, IGpuBuffer output, int[] shape, int[] permutation)
    {
        EnsureInitialized();

        if (shape.Length != permutation.Length)
            throw new ArgumentException(
                $"Shape rank ({shape.Length}) must match permutation length ({permutation.Length}).");

        int ndim = shape.Length;

        // Validate permutation: must be a valid permutation of [0..ndim-1]
        var seen = new bool[ndim];
        for (int i = 0; i < ndim; i++)
        {
            if (permutation[i] < 0 || permutation[i] >= ndim)
                throw new ArgumentOutOfRangeException(nameof(permutation),
                    $"Permutation index {permutation[i]} at position {i} is out of range [0, {ndim}).");
            if (seen[permutation[i]])
                throw new ArgumentException(
                    $"Duplicate index {permutation[i]} in permutation.", nameof(permutation));
            seen[permutation[i]] = true;
        }

        int totalSize = 1;
        for (int i = 0; i < ndim; i++) totalSize *= shape[i];

        var outShape = new int[ndim];
        for (int i = 0; i < ndim; i++) outShape[i] = shape[permutation[i]];

        var outStrides = new int[ndim];
        var inStrides = new int[ndim];
        outStrides[ndim - 1] = 1;
        inStrides[ndim - 1] = 1;
        for (int i = ndim - 2; i >= 0; i--)
        {
            outStrides[i] = outStrides[i + 1] * outShape[i + 1];
            inStrides[i] = inStrides[i + 1] * shape[i + 1];
        }

        var sourceStrides = new int[ndim];
        for (int d = 0; d < ndim; d++) sourceStrides[d] = inStrides[permutation[d]];
        string outStrideValues = string.Join(", ", Array.ConvertAll(outStrides, value => value + "u"));
        string sourceStrideValues = string.Join(", ", Array.ConvertAll(sourceStrides, value => value + "u"));
        string shader = $@"#version 450
layout(local_size_x = 256) in;
layout(set = 0, binding = 0) readonly buffer Input {{ float inputData[]; }};
layout(set = 0, binding = 1) writeonly buffer Output {{ float outputData[]; }};
layout(push_constant) uniform Params {{ uint totalSize; }};
const uint outStrides[{ndim}] = uint[{ndim}]({outStrideValues});
const uint sourceStrides[{ndim}] = uint[{ndim}]({sourceStrideValues});
void main() {{
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= totalSize) return;
    uint remaining = idx, sourceIndex = 0u;
    for (uint d = 0u; d < {ndim}u; ++d) {{
        uint coordinate = remaining / outStrides[d];
        remaining %= outStrides[d];
        sourceIndex += coordinate * sourceStrides[d];
    }}
    outputData[idx] = inputData[sourceIndex];
}}";
        GlslUnaryOp(shader, input, output, totalSize);
    }

    public void Copy(IGpuBuffer source, IGpuBuffer destination, int size)
    {
        Copy(source, 0, destination, 0, size);
    }

    public void Copy2DStrided(IGpuBuffer source, IGpuBuffer destination, int numRows,
        int srcCols, int destTotalCols, int destColOffset)
    {
        EnsureInitialized();

        if (destTotalCols <= 0)
            throw new ArgumentOutOfRangeException(nameof(destTotalCols), destTotalCols, "destTotalCols must be positive.");
        if (destColOffset < 0 || destColOffset >= destTotalCols)
            throw new ArgumentOutOfRangeException(nameof(destColOffset), destColOffset,
                $"destColOffset must be in [0, {destTotalCols}).");
        if (numRows < 0)
            throw new ArgumentOutOfRangeException(nameof(numRows), numRows, "numRows must be non-negative.");
        if (srcCols <= 0)
            throw new ArgumentOutOfRangeException(nameof(srcCols), srcCols, "srcCols must be positive.");

        int colsToCopy = Math.Min(srcCols, destTotalCols - destColOffset);
        if (numRows == 0 || colsToCopy == 0) return;
        GlslUnaryOp(VulkanGlslKernels.Copy2DStrided, source, destination, numRows * colsToCopy,
            new uint[] { (uint)numRows, (uint)srcCols, (uint)destTotalCols, (uint)destColOffset, (uint)colsToCopy },
            5 * sizeof(uint));
    }

    public void NearestNeighborUpsample(IGpuBuffer input, IGpuBuffer output, int batchChannels, int height, int width, int scaleFactor)
    {
        int outH = height * scaleFactor, outW = width * scaleFactor;
        GlslUnaryOp(VulkanGlslKernels.NearestNeighborUpsample, input, output, batchChannels * outH * outW,
            new uint[] { (uint)batchChannels, (uint)height, (uint)width, (uint)scaleFactor }, 4 * sizeof(uint));
    }

    public void NearestNeighborUpsampleBackward(IGpuBuffer gradOutput, IGpuBuffer gradInput, int batchChannels, int height, int width, int scaleFactor)
    {
        GlslUnaryOp(VulkanGlslKernels.NearestNeighborUpsampleBackward, gradOutput, gradInput,
            batchChannels * height * width,
            new uint[] { (uint)batchChannels, (uint)height, (uint)width, (uint)scaleFactor }, 4 * sizeof(uint));
    }

    public void Fill(IGpuBuffer buffer, float value, int size)
    {
        GlslGenerateOp(VulkanGlslKernels.FillGlsl, buffer, size,
            new uint[] { FloatBits(value), (uint)size }, 2 * sizeof(uint));
    }

    #endregion

    #region Random Number Generation

    public void GenerateRandomUniform(IGpuBuffer output, int size, float min, float max, ulong seed)
    {
        EnsureInitialized();
        if (size <= 0) return;
        GlslGenerateOp(VulkanResidentKernels.RandomGenerate, output, size,
            new uint[] { (uint)size, (uint)seed, (uint)(seed >> 32), 0u, FloatBits(min), FloatBits(max), 0u, 0u },
            8 * sizeof(uint));
    }

    public void GenerateStatelessDropoutMask(
        IGpuBuffer output, int size, uint threshold, float scale, uint seed)
    {
        EnsureInitialized();
        if (size <= 0) return;
        GlslGenerateOp(VulkanResidentKernels.StatelessDropoutMask, output, size,
            new uint[] { (uint)size, threshold, FloatBits(scale), seed }, 4 * sizeof(uint));
    }

    public void GenerateRandomNormal(IGpuBuffer output, int size, float mean, float stdDev, ulong seed)
    {
        EnsureInitialized();
        if (size <= 0) return;
        GlslGenerateOp(VulkanResidentKernels.RandomGenerate, output, size,
            new uint[] { (uint)size, (uint)seed, (uint)(seed >> 32), 1u, 0u, 0u, FloatBits(mean), FloatBits(stdDev) },
            8 * sizeof(uint));
    }

    public void GenerateSecureRandomUniform(IGpuBuffer output, int size, float min, float max)
    {
        EnsureInitialized();
        if (size <= 0) return;
        GenerateRandomUniform(output, size, min, max, GpuRandomSeed.Create());
    }

    #endregion
}
