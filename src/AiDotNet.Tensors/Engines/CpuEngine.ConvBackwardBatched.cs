using System;
using System.Buffers;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Tensors.Engines;

/// <summary>
/// The float32 im2col conv backward over the whole batch at once. The per-image path ran one GEMM per image whose
/// reduction axis was that image's output positions - 16 for a 256->512 stride-2 conv on 8x8 inputs - so each GEMM
/// rewrote the full kernel-sized output for a handful of multiply-adds, and the batch-parallel variant then allocated
/// a kernel-sized gradient per image and summed them serially. Laying the batch's columns side by side gives one GEMM
/// whose reduction axis is batch x output positions.
/// </summary>
public partial class CpuEngine
{
    // The batch-wide column buffer is colH x (batch * colW) floats. 16M (64 MB) is also the largest operand SimdGemm
    // materializes a transpose of; larger problems keep the per-image path. Without native BLAS the GEMMs run on SimdGemm:
    // BlasProvider.TryGemmEx would serve them from BlasManaged, measured ~5x slower on transposed operands.
    private const long ConvBackwardBatchedMaxColumnElements = 16L * 1024 * 1024;

    // Stride-1 input gradients run as a transposed conv, which wins on large planes (1.4 ms against 8 ms batched at 64
    // channels on 32x32) and loses on small ones; the batched path takes output planes up to this many positions.
    private const int ConvBackwardInputBatchedMaxPlane = 64;

    private static bool UseBatchedConvBackward(int batch, int colH, int colW, int outChannels)
        => batch > 1
           && (long)colH * batch * colW <= ConvBackwardBatchedMaxColumnElements
           && (long)outChannels * batch * colW <= ConvBackwardBatchedMaxColumnElements;

    /// <summary>Repacks a [batch, outChannels, colW] gradient as [outChannels, batch * colW].</summary>
    private static void PackConvGradOutputByChannel(float[] gradOutput, float[] packed, int batch, int outChannels, int colW)
    {
        int colWAll = batch * colW;
        CpuParallelSettings.ParallelForOrSerial(0, outChannels, (long)outChannels * colWAll, oc =>
        {
            for (int b = 0; b < batch; b++)
                Array.Copy(gradOutput, (b * outChannels + oc) * colW, packed, oc * colWAll + b * colW, colW);
        }, deterministicSafe: true);
    }

    private static void Conv2DBackwardKernelBatchedFloat(
        float[] destF, int destOff, bool accumulate, float[] gradOutputF, float[] inputF,
        int batch, int inChannels, int height, int width,
        int outChannels, int kernelHeight, int kernelWidth,
        int strideH, int strideW, int padH, int padW, int dilationH, int dilationW,
        int outputHeight, int outputWidth)
    {
        int colH = inChannels * kernelHeight * kernelWidth;
        int colW = outputHeight * outputWidth;
        int colWAll = batch * colW;
        int totalLen = outChannels * colH;
        int inputSliceSize = inChannels * height * width;
        var pool = ArrayPool<float>.Shared;
        var cols = pool.Rent(colH * colWAll);
        var packedGrad = pool.Rent(outChannels * colWAll);
        var result = accumulate ? pool.Rent(totalLen) : destF;
        int resultOff = accumulate ? 0 : destOff;
        try
        {
            CpuParallelSettings.ParallelForOrSerial(0, batch * inChannels, (long)colH * colWAll, bc =>
            {
                int b = bc / inChannels, c = bc % inChannels;
                Im2ColHelper.Im2ColStridedSingleChannelRange(
                    new ReadOnlySpan<float>(inputF, b * inputSliceSize, inputSliceSize),
                    new Span<float>(cols, 0, colH * colWAll), colWAll, b * colW,
                    c, c + 1, height, width, kernelHeight, kernelWidth,
                    strideH, strideW, padH, padW, dilationH, dilationW, outputHeight, outputWidth);
            }, deterministicSafe: true);
            PackConvGradOutputByChannel(gradOutputF, packedGrad, batch, outChannels, colW);

            // dW[outC, colH] = G[outC, batch*colW] . cols[colH, batch*colW]^T
            Array.Clear(result, resultOff, totalLen);
            if (!BlasProvider.IsAvailable || !BlasProvider.TryGemmEx(
                    outChannels, colH, colWAll,
                    packedGrad, 0, colWAll, false,
                    cols, 0, colWAll, true,
                    result, resultOff, colH))
            {
                Simd.SimdGemm.Sgemm(
                    new ReadOnlySpan<float>(packedGrad, 0, outChannels * colWAll), colWAll, false,
                    new ReadOnlySpan<float>(cols, 0, colH * colWAll), colWAll, true,
                    new Span<float>(result, resultOff, totalLen),
                    outChannels, colWAll, colH);
            }
            if (accumulate)
                for (int i = 0; i < totalLen; i++) destF[destOff + i] += result[i];
        }
        finally
        {
            pool.Return(cols);
            pool.Return(packedGrad);
            if (accumulate) pool.Return(result);
        }
    }

    /// <summary>Accumulates into <paramref name="destF"/>; the caller clears it first when not accumulating.</summary>
    private static void Conv2DBackwardInputBatchedFloat(
        float[] destF, int destOff, float[] gradOutputF, float[] kernelF,
        int batch, int inChannels, int height, int width,
        int outChannels, int kernelHeight, int kernelWidth,
        int strideH, int strideW, int padH, int padW, int dilationH, int dilationW,
        int outputHeight, int outputWidth)
    {
        int kHW = kernelHeight * kernelWidth;
        int colH = inChannels * kHW;
        int colW = outputHeight * outputWidth;
        int colWAll = batch * colW;
        int planeSize = height * width;
        var pool = ArrayPool<float>.Shared;
        var cols = pool.Rent(colH * colWAll);
        var packedGrad = pool.Rent(outChannels * colWAll);
        try
        {
            PackConvGradOutputByChannel(gradOutputF, packedGrad, batch, outChannels, colW);

            // cols[colH, batch*colW] = W[outC, colH]^T . G[outC, batch*colW]
            Array.Clear(cols, 0, colH * colWAll);
            if (!BlasProvider.IsAvailable || !BlasProvider.TryGemmEx(
                    colH, colWAll, outChannels,
                    kernelF, 0, colH, true,
                    packedGrad, 0, colWAll, false,
                    cols, 0, colWAll))
            {
                Simd.SimdGemm.Sgemm(
                    new ReadOnlySpan<float>(kernelF, 0, outChannels * colH), colH, true,
                    new ReadOnlySpan<float>(packedGrad, 0, outChannels * colWAll), colWAll, false,
                    new Span<float>(cols, 0, colH * colWAll),
                    colH, outChannels, colWAll);
            }

            // Each (image, channel) plane is written by exactly one worker.
            CpuParallelSettings.ParallelForOrSerial(0, batch * inChannels, (long)colH * colWAll, bc =>
            {
                int b = bc / inChannels, c = bc % inChannels;
                Im2ColHelper.Col2ImAccumulateStrided(
                    new ReadOnlySpan<float>(cols, c * kHW * colWAll, kHW * colWAll), colWAll, b * colW,
                    new Span<float>(destF, destOff + (b * inChannels + c) * planeSize, planeSize),
                    1, height, width, kernelHeight, kernelWidth,
                    strideH, strideW, padH, padW, dilationH, dilationW, outputHeight, outputWidth);
            }, deterministicSafe: true);
        }
        finally
        {
            pool.Return(cols);
            pool.Return(packedGrad);
        }
    }
}
