// Copyright (c) AiDotNet. All rights reserved.
// Metal GPU backend - Transpose, Reshape, and Utility operations.

namespace AiDotNet.Tensors.Engines.DirectGpu.Metal;

public sealed partial class MetalBackend
{
    #region Transpose and Reshape

    /// <summary>
    /// 2D matrix transpose.
    /// </summary>
    /// <remarks>
    /// TODO: Replace CPU fallback with a Metal compute kernel to avoid GPU-CPU-GPU round-trip.
    /// A proper implementation would use a Metal shader with threadgroup memory for coalesced reads/writes.
    /// </remarks>
    public void Transpose(IGpuBuffer A, IGpuBuffer B, int rows, int cols)
    {
        ThrowIfDisposed();

        // CPU fallback: download, transpose, upload
        // This incurs a GPU->CPU->GPU round-trip penalty.
        var aData = DownloadBuffer(A);
        var bData = new float[rows * cols];

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                bData[j * rows + i] = aData[i * cols + j];
            }
        }

        UploadToBuffer(B, bData);
    }

    /// <summary>
    /// Batched matrix transpose.
    /// </summary>
    public void BatchedTranspose(IGpuBuffer A, IGpuBuffer B, int batch, int rows, int cols)
    {
        ThrowIfDisposed();

        var aData = DownloadBuffer(A);
        var bData = new float[batch * rows * cols];

        for (int b = 0; b < batch; b++)
        {
            int offset = b * rows * cols;
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    bData[offset + j * rows + i] = aData[offset + i * cols + j];
                }
            }
        }

        UploadToBuffer(B, bData);
    }

    /// <summary>
    /// General tensor permutation.
    /// </summary>
    public void Permute(IGpuBuffer input, IGpuBuffer output, int[] shape, int[] permutation)
    {
        ThrowIfDisposed();

        var inputData = DownloadBuffer(input);
        int ndim = shape.Length;

        // Compute output shape and strides
        var outputShape = new int[ndim];
        for (int i = 0; i < ndim; i++)
        {
            outputShape[i] = shape[permutation[i]];
        }

        // Compute input strides
        var inputStrides = new int[ndim];
        inputStrides[ndim - 1] = 1;
        for (int i = ndim - 2; i >= 0; i--)
        {
            inputStrides[i] = inputStrides[i + 1] * shape[i + 1];
        }

        // Compute output strides
        var outputStrides = new int[ndim];
        outputStrides[ndim - 1] = 1;
        for (int i = ndim - 2; i >= 0; i--)
        {
            outputStrides[i] = outputStrides[i + 1] * outputShape[i + 1];
        }

        int totalSize = 1;
        for (int i = 0; i < ndim; i++)
        {
            totalSize *= shape[i];
        }

        var outputData = new float[totalSize];

        // Preallocate coordinate arrays outside loop to avoid per-element GC pressure
        var outCoords = new int[ndim];
        var inCoords = new int[ndim];

        for (int outIdx = 0; outIdx < totalSize; outIdx++)
        {
            // Convert output index to coordinates
            int remaining = outIdx;
            for (int d = 0; d < ndim; d++)
            {
                outCoords[d] = remaining / outputStrides[d];
                remaining %= outputStrides[d];
            }

            // Convert to input coordinates using inverse permutation
            for (int d = 0; d < ndim; d++)
            {
                inCoords[permutation[d]] = outCoords[d];
            }

            // Convert input coordinates to index
            int inIdx = 0;
            for (int d = 0; d < ndim; d++)
            {
                inIdx += inCoords[d] * inputStrides[d];
            }

            outputData[outIdx] = inputData[inIdx];
        }

        UploadToBuffer(output, outputData);
    }

    /// <summary>
    /// Copy data between buffers.
    /// </summary>
    public void Copy(IGpuBuffer source, IGpuBuffer destination, int size)
    {
        ThrowIfDisposed();
        var data = DownloadBuffer(source);
        UploadToBuffer(destination, data);
    }

    /// <summary>
    /// Fill buffer with a constant value.
    /// </summary>
    public void Fill(IGpuBuffer buffer, float value, int size)
    {
        ThrowIfDisposed();
        var data = new float[size];
        for (int i = 0; i < data.Length; i++) data[i] = value;
        UploadToBuffer(buffer, data);
    }

    /// <summary>
    /// Copy 2D region with different strides.
    /// </summary>
    public void Copy2DStrided(IGpuBuffer source, IGpuBuffer destination, int numRows, int srcCols, int destTotalCols, int destColOffset)
    {
        ThrowIfDisposed();

        var srcData = DownloadBuffer(source);
        var destData = DownloadBuffer(destination);

        for (int row = 0; row < numRows; row++)
        {
            for (int col = 0; col < srcCols; col++)
            {
                destData[row * destTotalCols + destColOffset + col] = srcData[row * srcCols + col];
            }
        }

        UploadToBuffer(destination, destData);
    }

    /// <summary>
    /// Nearest-neighbor upsampling for 2D spatial data.
    /// </summary>
    public void NearestNeighborUpsample(IGpuBuffer input, IGpuBuffer output, int batchChannels, int height, int width, int scaleFactor)
    {
        ThrowIfDisposed();

        var inputData = DownloadBuffer(input);
        int outHeight = height * scaleFactor;
        int outWidth = width * scaleFactor;
        var outputData = new float[batchChannels * outHeight * outWidth];

        for (int bc = 0; bc < batchChannels; bc++)
        {
            for (int oh = 0; oh < outHeight; oh++)
            {
                int ih = oh / scaleFactor;
                for (int ow = 0; ow < outWidth; ow++)
                {
                    int iw = ow / scaleFactor;
                    int outIdx = bc * outHeight * outWidth + oh * outWidth + ow;
                    int inIdx = bc * height * width + ih * width + iw;
                    outputData[outIdx] = inputData[inIdx];
                }
            }
        }

        UploadToBuffer(output, outputData);
    }

    /// <summary>
    /// Nearest-neighbor upsampling backward pass for 2D.
    /// </summary>
    public void NearestNeighborUpsampleBackward(IGpuBuffer gradOutput, IGpuBuffer gradInput, int batchChannels, int height, int width, int scaleFactor)
    {
        ThrowIfDisposed();

        var gradOutputData = DownloadBuffer(gradOutput);
        int outHeight = height * scaleFactor;
        int outWidth = width * scaleFactor;
        var gradInputData = new float[batchChannels * height * width];

        for (int bc = 0; bc < batchChannels; bc++)
        {
            for (int oh = 0; oh < outHeight; oh++)
            {
                int ih = oh / scaleFactor;
                for (int ow = 0; ow < outWidth; ow++)
                {
                    int iw = ow / scaleFactor;
                    int outIdx = bc * outHeight * outWidth + oh * outWidth + ow;
                    int inIdx = bc * height * width + ih * width + iw;
                    gradInputData[inIdx] += gradOutputData[outIdx];
                }
            }
        }

        UploadToBuffer(gradInput, gradInputData);
    }

    /// <summary>
    /// 3D nearest-neighbor upsampling.
    /// </summary>
    public void NearestNeighborUpsample3D(IGpuBuffer input, IGpuBuffer output,
        int batch, int channels, int inDepth, int inHeight, int inWidth,
        int scaleD, int scaleH, int scaleW)
    {
        ThrowIfDisposed();

        var inputData = DownloadBuffer(input);
        int outDepth = inDepth * scaleD;
        int outHeight = inHeight * scaleH;
        int outWidth = inWidth * scaleW;
        var outputData = new float[batch * channels * outDepth * outHeight * outWidth];

        for (int b = 0; b < batch; b++)
        {
            for (int c = 0; c < channels; c++)
            {
                for (int od = 0; od < outDepth; od++)
                {
                    int id = od / scaleD;
                    for (int oh = 0; oh < outHeight; oh++)
                    {
                        int ih = oh / scaleH;
                        for (int ow = 0; ow < outWidth; ow++)
                        {
                            int iw = ow / scaleW;
                            int outIdx = ((b * channels + c) * outDepth + od) * outHeight * outWidth + oh * outWidth + ow;
                            int inIdx = ((b * channels + c) * inDepth + id) * inHeight * inWidth + ih * inWidth + iw;
                            outputData[outIdx] = inputData[inIdx];
                        }
                    }
                }
            }
        }

        UploadToBuffer(output, outputData);
    }

    /// <summary>
    /// 3D nearest-neighbor upsampling backward pass.
    /// </summary>
    public void NearestNeighborUpsample3DBackward(IGpuBuffer gradOutput, IGpuBuffer gradInput,
        int batch, int channels, int inDepth, int inHeight, int inWidth,
        int scaleD, int scaleH, int scaleW)
    {
        ThrowIfDisposed();

        var gradOutputData = DownloadBuffer(gradOutput);
        int outDepth = inDepth * scaleD;
        int outHeight = inHeight * scaleH;
        int outWidth = inWidth * scaleW;
        var gradInputData = new float[batch * channels * inDepth * inHeight * inWidth];

        for (int b = 0; b < batch; b++)
        {
            for (int c = 0; c < channels; c++)
            {
                for (int od = 0; od < outDepth; od++)
                {
                    int id = od / scaleD;
                    for (int oh = 0; oh < outHeight; oh++)
                    {
                        int ih = oh / scaleH;
                        for (int ow = 0; ow < outWidth; ow++)
                        {
                            int iw = ow / scaleW;
                            int outIdx = ((b * channels + c) * outDepth + od) * outHeight * outWidth + oh * outWidth + ow;
                            int inIdx = ((b * channels + c) * inDepth + id) * inHeight * inWidth + ih * inWidth + iw;
                            gradInputData[inIdx] += gradOutputData[outIdx];
                        }
                    }
                }
            }
        }

        UploadToBuffer(gradInput, gradInputData);
    }

    #endregion

    #region Gradient Clipping and Utility

    /// <summary>
    /// Clamp values to a range.
    /// </summary>
    public void Clamp(IGpuBuffer A, IGpuBuffer B, float min, float max, int size)
    {
        ThrowIfDisposed();

        var aData = DownloadBuffer(A);
        var bData = new float[size];

        for (int i = 0; i < size; i++)
        {
            bData[i] = MathF.Max(min, MathF.Min(max, aData[i]));
        }

        UploadToBuffer(B, bData);
    }

    /// <summary>
    /// Compute L2 norm.
    /// </summary>
    public float L2Norm(IGpuBuffer A, int size)
    {
        ThrowIfDisposed();

        var aData = DownloadBuffer(A);
        float sumSq = 0;

        for (int i = 0; i < size; i++)
        {
            sumSq += aData[i] * aData[i];
        }

        return MathF.Sqrt(sumSq);
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

        var aData = DownloadBuffer(A);
        var bData = new float[size];

        float norm = L2Norm(A, size);

        if (norm > maxNorm)
        {
            float scale = maxNorm / norm;
            for (int i = 0; i < size; i++)
            {
                bData[i] = aData[i] * scale;
            }
        }
        else
        {
            Array.Copy(aData, bData, size);
        }

        UploadToBuffer(B, bData);
    }

    /// <summary>
    /// Fused multiply-add: D = A * B + C
    /// </summary>
    public void Fma(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, IGpuBuffer D, int size)
    {
        ThrowIfDisposed();

        var aData = DownloadBuffer(A);
        var bData = DownloadBuffer(B);
        var cData = DownloadBuffer(C);
        var dData = new float[size];

        for (int i = 0; i < size; i++)
        {
            dData[i] = aData[i] * bData[i] + cData[i];
        }

        UploadToBuffer(D, dData);
    }

    /// <summary>
    /// Scatter-add operation.
    /// </summary>
    public void ScatterAdd(IGpuBuffer source, IGpuBuffer indices, IGpuBuffer destination, int sourceSize, int destSize)
    {
        ThrowIfDisposed();

        var srcData = DownloadBuffer(source);
        var idxData = DownloadIntBuffer(indices, sourceSize);
        var destData = DownloadBuffer(destination);

        for (int i = 0; i < sourceSize; i++)
        {
            int idx = idxData[i];
            if (idx < 0 || idx >= destSize)
            {
                throw new ArgumentOutOfRangeException(nameof(indices), $"Index {idx} at position {i} is out of range [0, {destSize})");
            }

            destData[idx] += srcData[i];
        }

        UploadToBuffer(destination, destData);
    }

    /// <summary>
    /// Scatter-add backward (gather operation).
    /// </summary>
    public void ScatterAddBackward(IGpuBuffer gradDestination, IGpuBuffer indices, IGpuBuffer gradSource,
        int numIndices, int featureSize)
    {
        ThrowIfDisposed();

        var gradDestData = DownloadBuffer(gradDestination);
        var idxData = DownloadIntBuffer(indices, numIndices);
        var gradSrcData = new float[numIndices * featureSize];

        int maxGradIdx = gradDestData.Length / featureSize;
        for (int i = 0; i < numIndices; i++)
        {
            int idx = idxData[i];
            if (idx < 0 || idx >= maxGradIdx)
            {
                throw new ArgumentOutOfRangeException(nameof(indices), $"Index {idx} at position {i} is out of range [0, {maxGradIdx})");
            }

            for (int f = 0; f < featureSize; f++)
            {
                gradSrcData[i * featureSize + f] = gradDestData[idx * featureSize + f];
            }
        }

        UploadToBuffer(gradSource, gradSrcData);
    }

    /// <summary>
    /// Gather operation.
    /// </summary>
    public void Gather(IGpuBuffer source, IGpuBuffer indices, IGpuBuffer output, int numIndices, int featureSize)
    {
        ThrowIfDisposed();

        var srcData = DownloadBuffer(source);
        var idxData = DownloadIntBuffer(indices, numIndices);
        var outData = new float[numIndices * featureSize];

        int maxSrcIdx = srcData.Length / featureSize;
        for (int i = 0; i < numIndices; i++)
        {
            int idx = idxData[i];
            if (idx < 0 || idx >= maxSrcIdx)
            {
                throw new ArgumentOutOfRangeException(nameof(indices), $"Index {idx} at position {i} is out of range [0, {maxSrcIdx})");
            }

            for (int f = 0; f < featureSize; f++)
            {
                outData[i * featureSize + f] = srcData[idx * featureSize + f];
            }
        }

        UploadToBuffer(output, outData);
    }

    #endregion

    #region Mixed Precision

    /// <summary>
    /// Convert FP32 to FP16.
    /// </summary>
    public void ConvertToFp16(IGpuBuffer input, IGpuBuffer output, int size)
    {
        ThrowIfDisposed();

        var inputData = DownloadBuffer(input);
        var outputBytes = new byte[size * 2];

        for (int i = 0; i < size; i++)
        {
            ushort fp16 = FloatToHalf(inputData[i]);
            outputBytes[i * 2] = (byte)(fp16 & 0xFF);
            outputBytes[i * 2 + 1] = (byte)((fp16 >> 8) & 0xFF);
        }

        // Upload as float buffer (reinterpreted bytes)
        var floatData = new float[(size + 1) / 2];
        Buffer.BlockCopy(outputBytes, 0, floatData, 0, size * 2);
        UploadToBuffer(output, floatData);
    }

    /// <summary>
    /// Convert FP16 to FP32.
    /// </summary>
    public void ConvertToFp32(IGpuBuffer input, IGpuBuffer output, int size)
    {
        ThrowIfDisposed();

        var inputFloatData = DownloadBuffer(input);
        var inputBytes = new byte[size * 2];
        Buffer.BlockCopy(inputFloatData, 0, inputBytes, 0, size * 2);

        var outputData = new float[size];

        for (int i = 0; i < size; i++)
        {
            ushort fp16 = (ushort)(inputBytes[i * 2] | (inputBytes[i * 2 + 1] << 8));
            outputData[i] = HalfToFloat(fp16);
        }

        UploadToBuffer(output, outputData);
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
