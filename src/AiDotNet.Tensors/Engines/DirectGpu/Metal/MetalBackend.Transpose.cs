// Copyright (c) AiDotNet. All rights reserved.
// Metal GPU backend - Transpose, Reshape, and Utility operations.

namespace AiDotNet.Tensors.Engines.DirectGpu.Metal;

public sealed partial class MetalBackend
{
    #region Transpose and Reshape

    /// <summary>
    /// 2D matrix transpose using Metal compute kernel.
    /// </summary>
    public void Transpose(IGpuBuffer A, IGpuBuffer B, int rows, int cols)
    {
        ThrowIfDisposed();

        if (A is not MetalGpuBuffer aBuffer || B is not MetalGpuBuffer bBuffer)
        {
            throw new ArgumentException("Buffers must be MetalGpuBuffer");
        }

        var pipeline = GetPipeline("Matrix", _matrixLibrary, "transpose");
        var (threadgroups, threadsPerGroup) = pipeline.Calculate2DDispatch(cols, rows);

        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        encoder.SetBuffer(aBuffer, 0);
        encoder.SetBuffer(bBuffer, 1);
        encoder.SetBytes((uint)rows, 2);
        encoder.SetBytes((uint)cols, 3);
        encoder.DispatchThreadgroups(threadgroups, threadsPerGroup);
    }

    /// <summary>
    /// Batched matrix transpose using Metal compute kernel.
    /// </summary>
    public void BatchedTranspose(IGpuBuffer A, IGpuBuffer B, int batch, int rows, int cols)
    {
        ThrowIfDisposed();

        if (A is not MetalGpuBuffer aBuffer || B is not MetalGpuBuffer bBuffer)
        {
            throw new ArgumentException("Buffers must be MetalGpuBuffer");
        }

        var pipeline = GetPipeline("Matrix", _matrixLibrary, "batched_transpose");
        var (threadgroups, threadsPerGroup) = pipeline.Calculate3DDispatch(cols, rows, batch);

        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        encoder.SetBuffer(aBuffer, 0);
        encoder.SetBuffer(bBuffer, 1);
        encoder.SetBytes((uint)batch, 2);
        encoder.SetBytes((uint)rows, 3);
        encoder.SetBytes((uint)cols, 4);
        encoder.DispatchThreadgroups(threadgroups, threadsPerGroup);
    }

    /// <summary>
    /// General tensor permutation.
    /// </summary>
    public void Permute(IGpuBuffer input, IGpuBuffer output, int[] shape, int[] permutation)
    {
        ThrowIfDisposed();
        int dimensions = shape.Length;
        if (dimensions == 0)
            throw new ArgumentException("Shape must contain at least one dimension.", nameof(shape));
        if (permutation.Length != dimensions)
            throw new ArgumentException($"Permutation length ({permutation.Length}) must match shape dimensions ({dimensions}).", nameof(permutation));

        var seen = new bool[dimensions];
        var outputShape = new int[dimensions];
        var inputStrides = new int[dimensions];
        var outputStrides = new int[dimensions];
        for (int i = 0; i < dimensions; ++i)
        {
            int axis = permutation[i];
            if ((uint)axis >= (uint)dimensions)
                throw new ArgumentOutOfRangeException(nameof(permutation), $"Permutation index {axis} at position {i} is out of range [0, {dimensions}).");
            if (seen[axis])
                throw new ArgumentException($"Duplicate index {axis} in permutation array.", nameof(permutation));
            seen[axis] = true;
            outputShape[i] = shape[axis];
        }

        inputStrides[dimensions - 1] = 1;
        outputStrides[dimensions - 1] = 1;
        for (int i = dimensions - 2; i >= 0; --i)
        {
            inputStrides[i] = checked(inputStrides[i + 1] * shape[i + 1]);
            outputStrides[i] = checked(outputStrides[i + 1] * outputShape[i + 1]);
        }
        int count = 1;
        for (int i = 0; i < dimensions; ++i)
            count = checked(count * shape[i]);
        if (count <= 0) return;

        using var inputStrideBuffer = AllocateIntBuffer(inputStrides);
        using var outputStrideBuffer = AllocateIntBuffer(outputStrides);
        using var permutationBuffer = AllocateIntBuffer(permutation);
        bool aliasesInput = ReferenceEquals(input, output);
        using var temporary = aliasesInput ? AllocateBuffer(count) : null;
        IGpuBuffer target = temporary ?? output;
        DispatchResidentMetal("permute_tensor", count,
            new[] { input, target, inputStrideBuffer, outputStrideBuffer, permutationBuffer },
            (uint)dimensions, (uint)count);
        if (temporary is not null)
            Copy(temporary, output, count);
    }

    /// <summary>
    /// Copy data between buffers.
    /// </summary>
    public void Copy(IGpuBuffer source, IGpuBuffer destination, int size)
    {
        Copy(source, 0, destination, 0, size);
    }

    /// <summary>
    /// Fill buffer with a constant value.
    /// </summary>
    public void Fill(IGpuBuffer buffer, float value, int size)
    {
        ThrowIfDisposed();
        if (size <= 0) return;
        if (buffer is not MetalGpuBuffer metalBuffer)
            throw new ArgumentException("Buffer must be MetalGpuBuffer", nameof(buffer));

        var pipeline = GetPipeline("ElementWise", _elementWiseLibrary, "fill_buffer");
        var (threadgroups, threadsPerGroup) = pipeline.Calculate1DDispatch(size);
        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        encoder.SetBuffer(metalBuffer, 0);
        encoder.SetBytes(value, 1);
        encoder.SetBytes((uint)size, 2);
        encoder.DispatchThreadgroups(threadgroups, threadsPerGroup);
    }

    /// <summary>
    /// Copy 2D region with different strides.
    /// </summary>
    public void Copy2DStrided(IGpuBuffer source, IGpuBuffer destination, int numRows, int srcCols, int destTotalCols, int destColOffset)
    {
        ThrowIfDisposed();
        int count = checked(numRows * srcCols);
        if (count <= 0) return;
        bool aliasesSource = ReferenceEquals(source, destination);
        using var sourceCopy = aliasesSource ? AllocateBuffer(count) : null;
        IGpuBuffer effectiveSource = source;
        if (sourceCopy is not null)
        {
            Copy(source, sourceCopy, count);
            effectiveSource = sourceCopy;
        }
        DispatchResidentMetal("copy_2d_strided", count, new[] { effectiveSource, destination },
            (uint)numRows, (uint)srcCols, (uint)destTotalCols, (uint)destColOffset);
    }

    /// <summary>
    /// Nearest-neighbor upsampling for 2D spatial data.
    /// </summary>
    public void NearestNeighborUpsample(IGpuBuffer input, IGpuBuffer output, int batchChannels, int height, int width, int scaleFactor)
    {
        ThrowIfDisposed();
        if (scaleFactor <= 0) throw new ArgumentOutOfRangeException(nameof(scaleFactor));
        int count = checked(batchChannels * height * scaleFactor * width * scaleFactor);
        if (count <= 0) return;
        bool aliasesInput = ReferenceEquals(input, output);
        using var temporary = aliasesInput ? AllocateBuffer(count) : null;
        IGpuBuffer target = temporary ?? output;
        DispatchResidentMetal("nearest_upsample_2d", count, new[] { input, target },
            (uint)batchChannels, (uint)height, (uint)width, (uint)scaleFactor);
        if (temporary is not null) Copy(temporary, output, count);
    }

    /// <summary>
    /// Nearest-neighbor upsampling backward pass for 2D.
    /// </summary>
    public void NearestNeighborUpsampleBackward(IGpuBuffer gradOutput, IGpuBuffer gradInput, int batchChannels, int height, int width, int scaleFactor)
    {
        ThrowIfDisposed();
        if (scaleFactor <= 0) throw new ArgumentOutOfRangeException(nameof(scaleFactor));
        int count = checked(batchChannels * height * width);
        if (count <= 0) return;
        DispatchResidentMetal("nearest_upsample_2d_backward", count, new[] { gradOutput, gradInput },
            (uint)batchChannels, (uint)height, (uint)width, (uint)scaleFactor);
    }

    /// <summary>
    /// 3D nearest-neighbor upsampling.
    /// </summary>
    public void NearestNeighborUpsample3D(IGpuBuffer input, IGpuBuffer output,
        int batch, int channels, int inDepth, int inHeight, int inWidth,
        int scaleD, int scaleH, int scaleW)
    {
        ThrowIfDisposed();
        if (scaleD <= 0 || scaleH <= 0 || scaleW <= 0)
            throw new ArgumentOutOfRangeException(nameof(scaleD), "Scale factors must be positive.");
        int count = checked(batch * channels * inDepth * scaleD * inHeight * scaleH * inWidth * scaleW);
        if (count <= 0) return;
        bool aliasesInput = ReferenceEquals(input, output);
        using var temporary = aliasesInput ? AllocateBuffer(count) : null;
        IGpuBuffer target = temporary ?? output;
        DispatchResidentMetal("nearest_upsample_3d", count, new[] { input, target },
            (uint)batch, (uint)channels, (uint)inDepth, (uint)inHeight, (uint)inWidth,
            (uint)scaleD, (uint)scaleH, (uint)scaleW);
        if (temporary is not null) Copy(temporary, output, count);
    }

    /// <summary>
    /// 3D nearest-neighbor upsampling backward pass.
    /// </summary>
    public void NearestNeighborUpsample3DBackward(IGpuBuffer gradOutput, IGpuBuffer gradInput,
        int batch, int channels, int inDepth, int inHeight, int inWidth,
        int scaleD, int scaleH, int scaleW)
    {
        ThrowIfDisposed();
        if (scaleD <= 0 || scaleH <= 0 || scaleW <= 0)
            throw new ArgumentOutOfRangeException(nameof(scaleD), "Scale factors must be positive.");
        int count = checked(batch * channels * inDepth * inHeight * inWidth);
        if (count <= 0) return;
        DispatchResidentMetal("nearest_upsample_3d_backward", count, new[] { gradOutput, gradInput },
            (uint)batch, (uint)channels, (uint)inDepth, (uint)inHeight, (uint)inWidth,
            (uint)scaleD, (uint)scaleH, (uint)scaleW);
    }

    #endregion

    #region Gradient Clipping and Utility

    /// <summary>
    /// Clamp values to a range.
    /// </summary>
    public void Clamp(IGpuBuffer A, IGpuBuffer B, float min, float max, int size)
    {
        ThrowIfDisposed();
        if (size <= 0) return;
        if (A is not MetalGpuBuffer input || B is not MetalGpuBuffer output)
            throw new ArgumentException("Buffers must be MetalGpuBuffer.");
        var pipeline = GetPipeline("ElementWise", _elementWiseLibrary, "clamp_kernel");
        var (threadgroups, threadsPerGroup) = pipeline.Calculate1DDispatch(size);
        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        encoder.SetBuffer(input, 0);
        encoder.SetBuffer(output, 1);
        encoder.SetBytes(min, 2);
        encoder.SetBytes(max, 3);
        encoder.SetBytes((uint)size, 4);
        encoder.DispatchThreadgroups(threadgroups, threadsPerGroup);
    }

    /// <summary>
    /// Compute L2 norm.
    /// </summary>
    public float L2Norm(IGpuBuffer A, int size)
    {
        ThrowIfDisposed();
        if (size <= 0) return 0f;
        using var result = AllocateBuffer(1);
        DispatchResidentMetal("l2_norm_serial", 1, new[] { A, result }, (uint)size);
        return DownloadBuffer(result)[0];
    }

    /// <summary>
    /// Clip gradients by value.
    /// </summary>
    public void ClipByValue(IGpuBuffer A, IGpuBuffer B, float clipValue, int size)
    {
        ThrowIfDisposed();
        Clamp(A, B, -clipValue, clipValue, size);
    }

    /// <summary>
    /// Clip gradients by norm.
    /// </summary>
    public void ClipByNorm(IGpuBuffer A, IGpuBuffer B, float maxNorm, int size)
    {
        ThrowIfDisposed();
        if (size <= 0) return;
        if (A is not MetalGpuBuffer input || B is not MetalGpuBuffer output)
            throw new ArgumentException("Buffers must be MetalGpuBuffer.");
        if (_residentLibrary == IntPtr.Zero)
            throw new InvalidOperationException("Metal resident kernels are unavailable.");
        var pipeline = GetPipeline("Resident", _residentLibrary, "clip_by_norm_serial");
        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        encoder.SetBuffer(input, 0);
        encoder.SetBuffer(output, 1);
        encoder.SetBytes((uint)size, 2);
        encoder.SetBytes(maxNorm, 3);
        var (threadgroups, threadsPerGroup) = pipeline.Calculate1DDispatch(1);
        encoder.DispatchThreadgroups(threadgroups, threadsPerGroup);
    }

    /// <summary>
    /// Fused multiply-add: D = A * B + C
    /// </summary>
    public void Fma(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, IGpuBuffer D, int size)
    {
        ThrowIfDisposed();
        if (size <= 0) return;
        if (A is not MetalGpuBuffer a || B is not MetalGpuBuffer b ||
            C is not MetalGpuBuffer c || D is not MetalGpuBuffer output)
            throw new ArgumentException("Buffers must be MetalGpuBuffer.");
        var pipeline = GetPipeline("ElementWise", _elementWiseLibrary, "fma_kernel");
        var (threadgroups, threadsPerGroup) = pipeline.Calculate1DDispatch(size);
        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        encoder.SetBuffer(a, 0);
        encoder.SetBuffer(b, 1);
        encoder.SetBuffer(c, 2);
        encoder.SetBuffer(output, 3);
        encoder.SetBytes((uint)size, 4);
        encoder.DispatchThreadgroups(threadgroups, threadsPerGroup);
    }

    /// <summary>
    /// Fused linear interpolation: output = a + t * (b - a)
    /// </summary>
    public void Lerp(IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, float t, int size)
    {
        ThrowIfDisposed();

        if (a is not MetalGpuBuffer aBuffer || b is not MetalGpuBuffer bBuffer || output is not MetalGpuBuffer outBuffer)
        {
            throw new ArgumentException("Buffers must be MetalGpuBuffer");
        }

        var pipeline = GetPipeline("ElementWise", _elementWiseLibrary, "lerp_fused");
        var (threadgroups, threadsPerGroup) = pipeline.Calculate1DDispatch(size);

        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        encoder.SetBuffer(aBuffer, 0);
        encoder.SetBuffer(bBuffer, 1);
        encoder.SetBuffer(outBuffer, 2);
        encoder.SetBytes(t, 3);
        encoder.SetBytes((uint)size, 4);
        encoder.DispatchThreadgroups(threadgroups, threadsPerGroup);
    }

    /// <summary>
    /// Fused scaled addition: output = scaleA * a + scaleB * b
    /// </summary>
    public void AddScaled(IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, float scaleA, float scaleB, int size)
    {
        ThrowIfDisposed();

        if (a is not MetalGpuBuffer aBuffer || b is not MetalGpuBuffer bBuffer || output is not MetalGpuBuffer outBuffer)
        {
            throw new ArgumentException("Buffers must be MetalGpuBuffer");
        }

        var pipeline = GetPipeline("ElementWise", _elementWiseLibrary, "add_scaled");
        var (threadgroups, threadsPerGroup) = pipeline.Calculate1DDispatch(size);

        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        encoder.SetBuffer(aBuffer, 0);
        encoder.SetBuffer(bBuffer, 1);
        encoder.SetBuffer(outBuffer, 2);
        encoder.SetBytes(scaleA, 3);
        encoder.SetBytes(scaleB, 4);
        encoder.SetBytes((uint)size, 5);
        encoder.DispatchThreadgroups(threadgroups, threadsPerGroup);
    }

    /// <summary>
    /// Computes standard deviation across elements: sqrt(variance).
    /// Uses GPU reductions for both mean and variance to avoid downloading the input buffer.
    /// </summary>
    public float StdDev(IGpuBuffer input, int size)
    {
        ThrowIfDisposed();

        if (size <= 0)
            throw new ArgumentOutOfRangeException(nameof(size), size, "Size must be positive.");
        if (input.Size < size)
            throw new ArgumentException($"Buffer 'input' capacity ({input.Size}) is less than size ({size}).", nameof(input));
        if (size <= 1) return 0.0f;

        // Step 1: Compute mean via GPU Sum reduction
        float mean = Sum(input, size) / size;

        // Step 2: Compute variance entirely on GPU using existing element-wise ops:
        //   temp = input - mean  (per-element subtract via AddScalar with -mean)
        //   temp = temp * temp   (per-element square via Multiply)
        //   variance = Sum(temp) / size  (GPU reduction)
        using var temp = new MetalGpuBuffer(_device, size);
        AddScalar(input, temp, -mean, size);
        Multiply(temp, temp, temp, size);
        float varianceSum = Sum(temp, size);

        // Clamp variance to avoid NaN from floating-point round-off
        float variance = Math.Max(0, varianceSum / size);
        return MathF.Sqrt(variance);
    }

    /// <summary>
    /// Scatter-add operation.
    /// </summary>
    public void ScatterAdd(IGpuBuffer source, IGpuBuffer indices, IGpuBuffer destination, int sourceSize, int destSize)
    {
        ThrowIfDisposed();
        if (sourceSize <= 0 || destSize <= 0) return;
        bool aliasesSource = ReferenceEquals(source, destination);
        using var sourceCopy = aliasesSource ? AllocateBuffer(sourceSize) : null;
        IGpuBuffer effectiveSource = source;
        if (sourceCopy is not null)
        {
            Copy(source, sourceCopy, sourceSize);
            effectiveSource = sourceCopy;
        }
        DispatchResidentMetal("scatter_add_deterministic", destSize,
            new[] { effectiveSource, indices, destination }, (uint)sourceSize, (uint)destSize);
    }

    /// <summary>
    /// Scatter-add backward (gather operation).
    /// </summary>
    public void ScatterAddBackward(IGpuBuffer gradDestination, IGpuBuffer indices, IGpuBuffer gradSource,
        int numIndices, int featureSize)
    {
        ThrowIfDisposed();
        int count = checked(numIndices * featureSize);
        if (count <= 0) return;
        DispatchResidentMetal("indexed_gather", count,
            new[] { gradDestination, indices, gradSource }, (uint)numIndices, (uint)featureSize);
    }

    /// <summary>
    /// Gather operation.
    /// </summary>
    public void Gather(IGpuBuffer source, IGpuBuffer indices, IGpuBuffer output, int numIndices, int featureSize)
    {
        ThrowIfDisposed();
        int count = checked(numIndices * featureSize);
        if (count <= 0) return;
        bool aliasesSource = ReferenceEquals(source, output);
        using var temporary = aliasesSource ? AllocateBuffer(count) : null;
        IGpuBuffer target = temporary ?? output;
        DispatchResidentMetal("indexed_gather", count,
            new[] { source, indices, target }, (uint)numIndices, (uint)featureSize);
        if (temporary is not null) Copy(temporary, output, count);
    }

    #endregion

    #region Mixed Precision

    /// <summary>
    /// Convert FP32 to FP16.
    /// </summary>
    public void ConvertToFp16(IGpuBuffer input, IGpuBuffer output, int size)
    {
        ThrowIfDisposed();
        if (size <= 0) return;
        int packedCount = checked((size + 1) / 2);
        DispatchResidentMetal("convert_to_fp16_packed", packedCount,
            new[] { input, output }, (uint)size);
    }

    /// <summary>
    /// Convert FP16 to FP32.
    /// </summary>
    public void ConvertToFp32(IGpuBuffer input, IGpuBuffer output, int size)
    {
        ThrowIfDisposed();
        if (size <= 0) return;
        int requiredFloats = checked((size + 1) / 2);
        if (input.Size < requiredFloats)
            throw new ArgumentException($"Input buffer too small: has {input.Size} floats but needs at least {requiredFloats} to hold {size} FP16 values.", nameof(input));
        DispatchResidentMetal("convert_from_fp16_packed", size,
            new[] { input, output }, (uint)size);
    }

    private static ushort FloatToHalf(float value)
    {
        int bits = SingleToInt32BitsCompat(value);
        int sign = (bits >> 16) & 0x8000;
        int exp = ((bits >> 23) & 0xFF) - 112;
        int mantissa = bits & 0x7FFFFF;

        if (exp <= 0)
        {
            if (exp < -10)
            {
                return (ushort)sign;
            }
            mantissa |= 0x800000;
            int shift = 14 - exp;
            mantissa >>= shift;
            return (ushort)(sign | mantissa);
        }
        else if (exp >= 31)
        {
            if ((bits & 0x7FFFFFFF) > 0x7F800000)
            {
                return 0x7FFF; // NaN
            }
            return (ushort)(sign | 0x7C00); // Infinity
        }

        return (ushort)(sign | (exp << 10) | (mantissa >> 13));
    }

    private static float HalfToFloat(ushort value)
    {
        int sign = (value & 0x8000) << 16;
        int exp = (value >> 10) & 0x1F;
        int mantissa = value & 0x3FF;

        if (exp == 0)
        {
            if (mantissa == 0)
            {
                return Int32BitsToSingleCompat(sign);
            }
            // Subnormal
            while ((mantissa & 0x400) == 0)
            {
                mantissa <<= 1;
                exp--;
            }
            exp++;
            mantissa &= 0x3FF;
        }
        else if (exp == 31)
        {
            if (mantissa == 0)
            {
                return Int32BitsToSingleCompat(sign | 0x7F800000); // Infinity
            }
            return Int32BitsToSingleCompat(sign | 0x7FC00000); // NaN
        }

        exp += 112;
        int bits = sign | (exp << 23) | (mantissa << 13);
        return Int32BitsToSingleCompat(bits);
    }

    #endregion
}
