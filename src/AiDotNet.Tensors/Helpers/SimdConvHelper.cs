using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using System.Threading.Tasks;
using static AiDotNet.Tensors.Compatibility.MethodImplHelper;

namespace AiDotNet.Tensors.Helpers;

/// <summary>
/// SIMD-optimized convolution kernels using AVX2/AVX-512 intrinsics.
/// Provides highly optimized direct convolution for common kernel sizes.
/// </summary>
internal static class SimdConvHelper
{
    private static readonly bool UseAvx512 = Avx512F.IsSupported;
    private static readonly bool UseAvx2 = Avx2.IsSupported;
    private static readonly bool UseFma = Fma.IsSupported;
    private static readonly bool UseSse41 = Sse41.IsSupported;

    // Minimum output size for parallel processing (lowered for more parallelism)
    private const int ParallelThreshold = 1024; // 32x32 output

    // Output channel block size for better cache utilization
    private const int ChannelBlockSize = 4;

    // #209 close-parity A/B testing: choose the Conv3x3Stride1 variant at runtime.
    //   "per-channel"  → existing Conv3x3Stride1SingleChannel kernel (1 oc per task)
    //   "block4"       → Conv3x3Stride1Pad1_OcBlock4 (4 oc per task, register-resident)
    //   "block2"       → Conv3x3Stride1Pad1_OcBlock2 (2 oc per task, balances parallelism)
    //   "auto"         → choose based on outChannels and core count
    // Set via env var AIDOTNET_CONV3X3_VARIANT or by direct field assignment.
    internal enum Conv3x3Variant { Auto, PerChannel, Block2, Block4 }
    internal static Conv3x3Variant ActiveConv3x3Variant = ParseConv3x3VariantFromEnv();

    private static Conv3x3Variant ParseConv3x3VariantFromEnv()
    {
        // Normalise to lowercase so "Block4" / "BLOCK4" / "PerChannel" all
        // resolve. Other env-var parsers in this repo (e.g. MatrixMultiplyHelper.ReadEnvBool,
        // OpenClBackend.InitDiagnosticOutput) are case-insensitive; matching
        // that convention avoids surprising failures when users copy-paste
        // a variant name with a different casing into their CI / shell config.
        var v = Environment.GetEnvironmentVariable("AIDOTNET_CONV3X3_VARIANT")?.ToLowerInvariant();
        return v switch
        {
            "per-channel" or "perchannel" or "per_channel" => Conv3x3Variant.PerChannel,
            "block2"                                       => Conv3x3Variant.Block2,
            "block4"                                       => Conv3x3Variant.Block4,
            _                                              => Conv3x3Variant.Auto,
        };
    }

    /// <summary>
    /// Check if SIMD-optimized double-precision convolution is available.
    /// Currently only 3×3 stride=1 (the most common ResNet/transformer pattern).
    /// </summary>
    public static bool CanUseSimdConvDouble(int kernelH, int kernelW, int strideH, int strideW)
    {
        if (!UseAvx2 || !UseFma) return false;
        return kernelH == 3 && kernelW == 3 && strideH == 1 && strideW == 1;
    }

    /// <summary>
    /// Performs 3x3 convolution with stride=1 dilation=1 for double precision.
    /// Register-resident Vector256&lt;double&gt; (4 doubles/vec) inner kernel.
    /// Mirrors the float Conv3x3Stride1 design — eliminates the im2col+GEMM
    /// detour that was the cause of the 3.8× gap to libtorch on Conv2D_Double.
    /// </summary>
    public static unsafe void Conv3x3Stride1Double(
        double* input, double* kernel, double* output,
        int batch, int inChannels, int height, int width,
        int outChannels, int padH, int padW, int dilationH, int dilationW)
    {
        int outHeight = height + 2 * padH - (dilationH * 2 + 1) + 1;
        int outWidth = width + 2 * padW - (dilationW * 2 + 1) + 1;
        int outputSize = outHeight * outWidth;

        bool useParallel = outputSize >= ParallelThreshold && Environment.ProcessorCount > 1;

        for (int b = 0; b < batch; b++)
        {
            double* inputBatch = input + b * inChannels * height * width;
            double* outputBatch = output + b * outChannels * outHeight * outWidth;

            if (useParallel)
            {
                CpuParallelSettings.LightweightParallel(outChannels, oc =>
                {
                    Conv3x3Stride1SingleChannelDouble(
                        inputBatch, kernel + oc * inChannels * 9,
                        outputBatch + oc * outputSize,
                        inChannels, height, width, outHeight, outWidth, padH, padW, dilationH, dilationW);
                });
            }
            else
            {
                for (int oc = 0; oc < outChannels; oc++)
                {
                    Conv3x3Stride1SingleChannelDouble(
                        inputBatch, kernel + oc * inChannels * 9,
                        outputBatch + oc * outputSize,
                        inChannels, height, width, outHeight, outWidth, padH, padW, dilationH, dilationW);
                }
            }
        }
    }

    [MethodImpl(HotInline)]
    private static unsafe void Conv3x3Stride1SingleChannelDouble(
        double* input, double* kernelOc, double* outputChannel,
        int inChannels, int height, int width, int outHeight, int outWidth,
        int padH, int padW, int dilationH, int dilationW)
    {
        int outputSize = outHeight * outWidth;
        new Span<double>(outputChannel, outputSize).Clear();

        for (int ic = 0; ic < inChannels; ic++)
        {
            double* inputChannel = input + ic * height * width;
            double* kernelChannel = kernelOc + ic * 9;
            Conv3x3SingleChannelFmaDouble(inputChannel, kernelChannel, outputChannel,
                height, width, outHeight, outWidth, padH, padW, dilationH, dilationW);
        }
    }

    /// <summary>
    /// Per-(ic, oc) Vector256&lt;double&gt; 3×3 conv. Loads 9 broadcasted kernel
    /// values, then sweeps the output rows with 4-double FMA accumulators.
    /// Boundary rows/cols use a scalar fallback for correctness.
    /// </summary>
    [MethodImpl(HotInline)]
    private static unsafe void Conv3x3SingleChannelFmaDouble(
        double* input, double* kernel, double* output,
        int height, int width, int outHeight, int outWidth,
        int padH, int padW, int dilationH, int dilationW)
    {
        // 9 broadcasted kernel values held across the entire (oh, ow) sweep.
        Vector256<double> k00 = Vector256.Create(kernel[0]);
        Vector256<double> k01 = Vector256.Create(kernel[1]);
        Vector256<double> k02 = Vector256.Create(kernel[2]);
        Vector256<double> k10 = Vector256.Create(kernel[3]);
        Vector256<double> k11 = Vector256.Create(kernel[4]);
        Vector256<double> k12 = Vector256.Create(kernel[5]);
        Vector256<double> k20 = Vector256.Create(kernel[6]);
        Vector256<double> k21 = Vector256.Create(kernel[7]);
        Vector256<double> k22 = Vector256.Create(kernel[8]);

        if (dilationH == 1 && dilationW == 1)
        {
            int ohStart = padH > 0 ? padH : 0;
            int ohEnd = outHeight - (padH > 0 ? padH : 0);
            if (ohEnd > outHeight) ohEnd = outHeight;

            // Top boundary rows
            for (int topOh = 0; topOh < ohStart && topOh < outHeight; topOh++)
                ProcessBoundaryRowDouble(input, kernel, output + topOh * outWidth,
                    height, width, outWidth, topOh, padH, padW);

            // Interior rows: SIMD sweep on 4-double output chunks.
            // Edge cols (left ow < padW, right ow >= outWidth - padW - 3)
            // handled by scalar fallback.
            for (int oh = ohStart; oh < ohEnd; oh++)
            {
                int ih0 = oh - padH;
                double* inputRow0 = input + ih0       * width;
                double* inputRow1 = input + (ih0 + 1) * width;
                double* inputRow2 = input + (ih0 + 2) * width;
                double* outputRow = output + oh * outWidth;

                int owStart = padW > 0 ? padW : 0;
                int owEnd = outWidth - (padW > 0 ? padW : 0) - 3;

                // Left boundary cols
                for (int leftOw = 0; leftOw < owStart && leftOw < outWidth; leftOw++)
                    outputRow[leftOw] += ScalarConv3x3Double(input, kernel, height, width, oh, leftOw, padH, padW);

                int ow = owStart;
                for (; ow < owEnd; ow += 4)
                {
                    int iw = ow - padW;
                    Vector256<double> r0_0 = Avx.LoadVector256(inputRow0 + iw);
                    Vector256<double> r0_1 = Avx.LoadVector256(inputRow0 + iw + 1);
                    Vector256<double> r0_2 = Avx.LoadVector256(inputRow0 + iw + 2);
                    Vector256<double> r1_0 = Avx.LoadVector256(inputRow1 + iw);
                    Vector256<double> r1_1 = Avx.LoadVector256(inputRow1 + iw + 1);
                    Vector256<double> r1_2 = Avx.LoadVector256(inputRow1 + iw + 2);
                    Vector256<double> r2_0 = Avx.LoadVector256(inputRow2 + iw);
                    Vector256<double> r2_1 = Avx.LoadVector256(inputRow2 + iw + 1);
                    Vector256<double> r2_2 = Avx.LoadVector256(inputRow2 + iw + 2);

                    Vector256<double> acc = Fma.MultiplyAdd(r0_0, k00, Vector256<double>.Zero);
                    acc = Fma.MultiplyAdd(r0_1, k01, acc);
                    acc = Fma.MultiplyAdd(r0_2, k02, acc);
                    acc = Fma.MultiplyAdd(r1_0, k10, acc);
                    acc = Fma.MultiplyAdd(r1_1, k11, acc);
                    acc = Fma.MultiplyAdd(r1_2, k12, acc);
                    acc = Fma.MultiplyAdd(r2_0, k20, acc);
                    acc = Fma.MultiplyAdd(r2_1, k21, acc);
                    acc = Fma.MultiplyAdd(r2_2, k22, acc);

                    Vector256<double> current = Avx.LoadVector256(outputRow + ow);
                    Avx.Store(outputRow + ow, Avx.Add(current, acc));
                }

                // Right boundary cols (scalar)
                for (; ow < outWidth; ow++)
                    outputRow[ow] += ScalarConv3x3Double(input, kernel, height, width, oh, ow, padH, padW);
            }

            // Bottom boundary rows
            for (int bottomOh = ohEnd; bottomOh < outHeight; bottomOh++)
                ProcessBoundaryRowDouble(input, kernel, output + bottomOh * outWidth,
                    height, width, outWidth, bottomOh, padH, padW);
        }
        else
        {
            // Dilation: scalar fallback (rare path)
            for (int oh = 0; oh < outHeight; oh++)
                for (int ow = 0; ow < outWidth; ow++)
                    output[oh * outWidth + ow] += ScalarConv3x3DoubleDilated(
                        input, kernel, height, width, oh, ow, padH, padW, dilationH, dilationW);
        }
    }

    [MethodImpl(HotInline)]
    private static unsafe double ScalarConv3x3Double(
        double* input, double* kernel,
        int height, int width, int oh, int ow, int padH, int padW)
    {
        double sum = 0;
        int ihBase = oh - padH;
        int iwBase = ow - padW;
        for (int kh = 0; kh < 3; kh++)
        {
            int ih = ihBase + kh;
            if (ih < 0 || ih >= height) continue;
            for (int kw = 0; kw < 3; kw++)
            {
                int iw = iwBase + kw;
                if (iw < 0 || iw >= width) continue;
                sum += input[ih * width + iw] * kernel[kh * 3 + kw];
            }
        }
        return sum;
    }

    [MethodImpl(HotInline)]
    private static unsafe double ScalarConv3x3DoubleDilated(
        double* input, double* kernel,
        int height, int width, int oh, int ow, int padH, int padW, int dilationH, int dilationW)
    {
        double sum = 0;
        int ihBase = oh - padH;
        int iwBase = ow - padW;
        for (int kh = 0; kh < 3; kh++)
        {
            int ih = ihBase + kh * dilationH;
            if (ih < 0 || ih >= height) continue;
            for (int kw = 0; kw < 3; kw++)
            {
                int iw = iwBase + kw * dilationW;
                if (iw < 0 || iw >= width) continue;
                sum += input[ih * width + iw] * kernel[kh * 3 + kw];
            }
        }
        return sum;
    }

    [MethodImpl(HotInline)]
    private static unsafe void ProcessBoundaryRowDouble(
        double* input, double* kernel, double* outputRow,
        int height, int width, int outWidth, int oh, int padH, int padW)
    {
        for (int ow = 0; ow < outWidth; ow++)
            outputRow[ow] += ScalarConv3x3Double(input, kernel, height, width, oh, ow, padH, padW);
    }

    /// <summary>
    /// Check if SIMD-optimized convolution is available for this configuration.
    /// </summary>
    public static bool CanUseSimdConv(int kernelH, int kernelW, int strideH, int strideW)
    {
        // SIMD convolution requires AVX2 and works best with stride=1
        if (!UseAvx2)
        {
            return false;
        }

        // Optimized for common kernel sizes with stride=1
        return (kernelH == 3 && kernelW == 3 && strideH == 1 && strideW == 1) ||
               (kernelH == 1 && kernelW == 1 && strideH == 1 && strideW == 1);
    }

    /// <summary>
    /// Performs 3x3 convolution with stride=1 using AVX2 intrinsics.
    /// </summary>
    public static unsafe void Conv3x3Stride1(
        float* input, float* kernel, float* output,
        int batch, int inChannels, int height, int width,
        int outChannels, int padH, int padW, int dilationH, int dilationW)
    {
        int outHeight = height + 2 * padH - (dilationH * 2 + 1) + 1;
        int outWidth = width + 2 * padW - (dilationW * 2 + 1) + 1;
        int outputSize = outHeight * outWidth;

        bool useParallel = outputSize >= ParallelThreshold && Environment.ProcessorCount > 1;

        // #209 close-parity A/B: select Conv3x3 variant.
        // Pad=1 dilation=1 are required by the OcBlock kernels.
        bool padOneNoDilation = padH == 1 && padW == 1 && dilationH == 1 && dilationW == 1;
        bool canUseBlock4 = UseFma && padOneNoDilation && outChannels >= 4 && (outChannels & 3) == 0;
        bool canUseBlock2 = UseFma && padOneNoDilation && outChannels >= 2 && (outChannels & 1) == 0;

        // Auto policy: prefer the variant that gives ~one task per core.
        //   - 16-core Ryzen, outChannels=32: per-channel = 32 tasks (2× cores),
        //     block2 = 16 tasks (1× cores), block4 = 8 tasks (0.5× cores).
        //     block2 maximizes parallelism without over-subscribing.
        //   - With outChannels >> cores (e.g. 256 oc on 16 cores), per-channel
        //     is fine; block2/block4 lose nothing on parallelism and gain
        //     register-resident accumulator amortization.
        Conv3x3Variant variant = ActiveConv3x3Variant;
        if (variant == Conv3x3Variant.Auto)
        {
            int cores = Math.Max(1, Environment.ProcessorCount);
            // Minimum tasks per core for healthy parallel utilization
            int minTasks = cores;
            if (canUseBlock4 && (outChannels / 4) >= minTasks)
                variant = Conv3x3Variant.Block4;
            else if (canUseBlock2 && (outChannels / 2) >= minTasks)
                variant = Conv3x3Variant.Block2;
            else
                variant = Conv3x3Variant.PerChannel;
        }
        // Fallback if requested variant not applicable (wrong shape).
        if (variant == Conv3x3Variant.Block4 && !canUseBlock4) variant = Conv3x3Variant.PerChannel;
        if (variant == Conv3x3Variant.Block2 && !canUseBlock2) variant = Conv3x3Variant.PerChannel;

        for (int b = 0; b < batch; b++)
        {
            float* inputBatch = input + b * inChannels * height * width;
            float* outputBatch = output + b * outChannels * outHeight * outWidth;

            if (variant == Conv3x3Variant.Block4)
            {
                int numBlocks = outChannels / 4;
                if (useParallel)
                {
                    CpuParallelSettings.LightweightParallel(numBlocks, ocb =>
                    {
                        Conv3x3Stride1Pad1_OcBlock4(
                            inputBatch, kernel + ocb * 4 * inChannels * 9,
                            outputBatch + ocb * 4 * outputSize,
                            inChannels, height, width, outHeight, outWidth);
                    });
                }
                else
                {
                    for (int ocb = 0; ocb < numBlocks; ocb++)
                        Conv3x3Stride1Pad1_OcBlock4(
                            inputBatch, kernel + ocb * 4 * inChannels * 9,
                            outputBatch + ocb * 4 * outputSize,
                            inChannels, height, width, outHeight, outWidth);
                }
            }
            else if (variant == Conv3x3Variant.Block2)
            {
                int numBlocks = outChannels / 2;
                if (useParallel)
                {
                    CpuParallelSettings.LightweightParallel(numBlocks, ocb =>
                    {
                        Conv3x3Stride1Pad1_OcBlock2(
                            inputBatch, kernel + ocb * 2 * inChannels * 9,
                            outputBatch + ocb * 2 * outputSize,
                            inChannels, height, width, outHeight, outWidth);
                    });
                }
                else
                {
                    for (int ocb = 0; ocb < numBlocks; ocb++)
                        Conv3x3Stride1Pad1_OcBlock2(
                            inputBatch, kernel + ocb * 2 * inChannels * 9,
                            outputBatch + ocb * 2 * outputSize,
                            inChannels, height, width, outHeight, outWidth);
                }
            }
            else if (useParallel)
            {
                CpuParallelSettings.LightweightParallel(outChannels, oc =>
                {
                    Conv3x3Stride1SingleChannel(
                        inputBatch, kernel + oc * inChannels * 9,
                        outputBatch + oc * outputSize,
                        inChannels, height, width, outHeight, outWidth,
                        padH, padW, dilationH, dilationW);
                });
            }
            else
            {
                for (int oc = 0; oc < outChannels; oc++)
                    Conv3x3Stride1SingleChannel(
                        inputBatch, kernel + oc * inChannels * 9,
                        outputBatch + oc * outputSize,
                        inChannels, height, width, outHeight, outWidth,
                        padH, padW, dilationH, dilationW);
            }
        }
    }

    /// <summary>
    /// 2-oc-blocked variant of <see cref="Conv3x3Stride1Pad1_OcBlock4"/>.
    /// Holds 2 accumulators per chunk for better parallelism granularity
    /// when outChannels / 2 ≈ core count (e.g. outChannels=32 on 16 cores
    /// gives 16 tasks vs 8 with the 4-oc variant).
    /// </summary>
    [MethodImpl(HotInline)]
    private static unsafe void Conv3x3Stride1Pad1_OcBlock2(
        float* input,
        float* kBlock,   // [2, inChannels, 9]
        float* outBlock, // [2, outHeight, outWidth]
        int inChannels, int height, int width,
        int outHeight, int outWidth)
    {
        int ocStride = outHeight * outWidth;
        int icSpatialStride = height * width;

        for (int oc = 0; oc < 2; oc++)
            new Span<float>(outBlock + oc * ocStride, ocStride).Clear();

        for (int oc = 0; oc < 2; oc++)
        {
            float* outChan = outBlock + oc * ocStride;
            float* kChan = kBlock + oc * inChannels * 9;
            ProcessConv3x3BoundaryRowAllIc(input, kChan, outChan,
                inChannels, icSpatialStride, height, width, outWidth, oh: 0);
            if (outHeight > 1)
                ProcessConv3x3BoundaryRowAllIc(input, kChan, outChan,
                    inChannels, icSpatialStride, height, width, outWidth, oh: outHeight - 1);
        }

        for (int oh = 1; oh < outHeight - 1; oh++)
        {
            int ihTop = oh - 1;
            // Left boundary col
            for (int oc = 0; oc < 2; oc++)
            {
                float* outChan = outBlock + oc * ocStride;
                float* kChan = kBlock + oc * inChannels * 9;
                outChan[oh * outWidth + 0] = ScalarConv3x3At(input, kChan,
                    inChannels, icSpatialStride, height, width, oh, 0, 1, 1);
            }

            int owStart = 1, owEnd = outWidth - 1;
            int owSimdEnd = owStart + ((owEnd - owStart) & ~7);

            for (int ow = owStart; ow < owSimdEnd; ow += 8)
            {
                int iw = ow - 1;
                var acc0 = Vector256<float>.Zero;
                var acc1 = Vector256<float>.Zero;

                for (int ic = 0; ic < inChannels; ic++)
                {
                    float* inputChan = input + ic * icSpatialStride;
                    var r0_0 = Avx.LoadVector256(inputChan + ihTop       * width + iw);
                    var r0_1 = Avx.LoadVector256(inputChan + ihTop       * width + iw + 1);
                    var r0_2 = Avx.LoadVector256(inputChan + ihTop       * width + iw + 2);
                    var r1_0 = Avx.LoadVector256(inputChan + (ihTop + 1) * width + iw);
                    var r1_1 = Avx.LoadVector256(inputChan + (ihTop + 1) * width + iw + 1);
                    var r1_2 = Avx.LoadVector256(inputChan + (ihTop + 1) * width + iw + 2);
                    var r2_0 = Avx.LoadVector256(inputChan + (ihTop + 2) * width + iw);
                    var r2_1 = Avx.LoadVector256(inputChan + (ihTop + 2) * width + iw + 1);
                    var r2_2 = Avx.LoadVector256(inputChan + (ihTop + 2) * width + iw + 2);

                    float* k0 = kBlock + 0 * inChannels * 9 + ic * 9;
                    float* k1 = kBlock + 1 * inChannels * 9 + ic * 9;

                    acc0 = Fma.MultiplyAdd(r0_0, Vector256.Create(k0[0]), acc0);
                    acc0 = Fma.MultiplyAdd(r0_1, Vector256.Create(k0[1]), acc0);
                    acc0 = Fma.MultiplyAdd(r0_2, Vector256.Create(k0[2]), acc0);
                    acc0 = Fma.MultiplyAdd(r1_0, Vector256.Create(k0[3]), acc0);
                    acc0 = Fma.MultiplyAdd(r1_1, Vector256.Create(k0[4]), acc0);
                    acc0 = Fma.MultiplyAdd(r1_2, Vector256.Create(k0[5]), acc0);
                    acc0 = Fma.MultiplyAdd(r2_0, Vector256.Create(k0[6]), acc0);
                    acc0 = Fma.MultiplyAdd(r2_1, Vector256.Create(k0[7]), acc0);
                    acc0 = Fma.MultiplyAdd(r2_2, Vector256.Create(k0[8]), acc0);

                    acc1 = Fma.MultiplyAdd(r0_0, Vector256.Create(k1[0]), acc1);
                    acc1 = Fma.MultiplyAdd(r0_1, Vector256.Create(k1[1]), acc1);
                    acc1 = Fma.MultiplyAdd(r0_2, Vector256.Create(k1[2]), acc1);
                    acc1 = Fma.MultiplyAdd(r1_0, Vector256.Create(k1[3]), acc1);
                    acc1 = Fma.MultiplyAdd(r1_1, Vector256.Create(k1[4]), acc1);
                    acc1 = Fma.MultiplyAdd(r1_2, Vector256.Create(k1[5]), acc1);
                    acc1 = Fma.MultiplyAdd(r2_0, Vector256.Create(k1[6]), acc1);
                    acc1 = Fma.MultiplyAdd(r2_1, Vector256.Create(k1[7]), acc1);
                    acc1 = Fma.MultiplyAdd(r2_2, Vector256.Create(k1[8]), acc1);
                }

                Avx.Store(outBlock + 0 * ocStride + oh * outWidth + ow, acc0);
                Avx.Store(outBlock + 1 * ocStride + oh * outWidth + ow, acc1);
            }

            for (int ow = owSimdEnd; ow < owEnd; ow++)
            {
                for (int oc = 0; oc < 2; oc++)
                {
                    float* outChan = outBlock + oc * ocStride;
                    float* kChan = kBlock + oc * inChannels * 9;
                    outChan[oh * outWidth + ow] = ScalarConv3x3At(input, kChan,
                        inChannels, icSpatialStride, height, width, oh, ow, 1, 1);
                }
            }

            for (int oc = 0; oc < 2; oc++)
            {
                float* outChan = outBlock + oc * ocStride;
                float* kChan = kBlock + oc * inChannels * 9;
                outChan[oh * outWidth + outWidth - 1] = ScalarConv3x3At(input, kChan,
                    inChannels, icSpatialStride, height, width, oh, outWidth - 1, 1, 1);
            }
        }
    }

    /// <summary>
    /// 3×3 stride=1 padding=1 dilation=1 conv that processes 4 output channels
    /// at a time. For each (oh, ow_chunk) position it holds 4 AVX2 accumulators
    /// in registers across the entire input-channel reduction loop, then writes
    /// 4 outputs once at the end. This eliminates the output round-trip
    /// (load+add+store per ic per chunk) that the per-channel kernel does.
    ///
    /// <para>Mirrors oneDNN's <c>jit_avx2_conv_kernel_f32</c> structure:
    /// <c>nb_oc_blocking = 4</c>, accumulators stay register-resident through
    /// the ic reduction. AVX2 has 16 ymm registers; we use 4 for accumulators
    /// + 9 for 3×3 input vectors per ic = 13 ymm budget — fits without spills.</para>
    ///
    /// <para>Boundary rows (oh = 0, oh = outHeight-1) and the 8-aligned tail
    /// of each row are handled by a scalar fallback for correctness; the
    /// interior dominates the FLOPs at ResNet/transformer shapes
    /// (e.g. 64×64 = 4096 outputs, 62×56 = 3472 are interior, 85% of work).</para>
    /// </summary>
    [MethodImpl(HotInline)]
    private static unsafe void Conv3x3Stride1Pad1_OcBlock4(
        float* input,    // [inChannels, height, width]
        float* kBlock,   // [4, inChannels, 9] — 4 output channels' kernels (3×3 each)
        float* outBlock, // [4, outHeight, outWidth] — 4 output channels' outputs
        int inChannels, int height, int width,
        int outHeight, int outWidth)
    {
        int ocStride = outHeight * outWidth;
        int icSpatialStride = height * width;

        // Clear all 4 output channels.
        for (int oc = 0; oc < 4; oc++)
            new Span<float>(outBlock + oc * ocStride, ocStride).Clear();

        // Boundary rows: top (oh=0) and bottom (oh=outHeight-1) need
        // padding-aware kernel position skipping. Delegate to the existing
        // scalar fallback per-channel for correctness.
        for (int oc = 0; oc < 4; oc++)
        {
            float* outChan = outBlock + oc * ocStride;
            float* kChan = kBlock + oc * inChannels * 9;

            // Top boundary row
            ProcessConv3x3BoundaryRowAllIc(input, kChan, outChan,
                inChannels, icSpatialStride, height, width, outWidth, oh: 0);
            // Bottom boundary row (only if distinct from top)
            if (outHeight > 1)
            {
                ProcessConv3x3BoundaryRowAllIc(input, kChan, outChan,
                    inChannels, icSpatialStride, height, width, outWidth, oh: outHeight - 1);
            }
        }

        // Interior rows: oh in [1, outHeight - 2]. For these rows ih = oh - 1,
        // ih+1, ih+2 are all in-bounds (no top/bottom boundary), but the
        // left and right ow boundaries still need scalar handling.
        // Interior cols where ow_chunk + 8 <= outWidth-1 and ow_chunk >= 1
        // are the SIMD fast path.
        for (int oh = 1; oh < outHeight - 1; oh++)
        {
            int ihTop = oh - 1; // padH=1
            // Process the leftmost output column ow=0 with scalar (boundary).
            for (int oc = 0; oc < 4; oc++)
            {
                float* outChan = outBlock + oc * ocStride;
                float* kChan = kBlock + oc * inChannels * 9;
                outChan[oh * outWidth + 0] = ScalarConv3x3At(input, kChan,
                    inChannels, icSpatialStride, height, width, oh, 0, 1, 1);
            }

            // Interior columns: ow from 1 to outWidth - 2, processed in chunks
            // of 8. The chunk requires ih, ih+1, ih+2 input rows × ow-1, ow,
            // ow+1 column offsets — all in-bounds at padding=1 + interior oh.
            int owStart = 1;
            int owEnd = outWidth - 1; // exclusive — last col is right boundary
            int owSimdEnd = owStart + ((owEnd - owStart) & ~7);

            for (int ow = owStart; ow < owSimdEnd; ow += 8)
            {
                int iw = ow - 1; // padW=1, kernel column 0 starts at iw

                // Hold 4 oc accumulators across the ic reduction.
                var acc0 = Vector256<float>.Zero;
                var acc1 = Vector256<float>.Zero;
                var acc2 = Vector256<float>.Zero;
                var acc3 = Vector256<float>.Zero;

                for (int ic = 0; ic < inChannels; ic++)
                {
                    float* inputChan = input + ic * icSpatialStride;
                    // Load 9 input vectors (3 input rows × 3 column offsets) ONCE
                    // per ic — these are shared across all 4 output channels.
                    var r0_0 = Avx.LoadVector256(inputChan + ihTop       * width + iw);
                    var r0_1 = Avx.LoadVector256(inputChan + ihTop       * width + iw + 1);
                    var r0_2 = Avx.LoadVector256(inputChan + ihTop       * width + iw + 2);
                    var r1_0 = Avx.LoadVector256(inputChan + (ihTop + 1) * width + iw);
                    var r1_1 = Avx.LoadVector256(inputChan + (ihTop + 1) * width + iw + 1);
                    var r1_2 = Avx.LoadVector256(inputChan + (ihTop + 1) * width + iw + 2);
                    var r2_0 = Avx.LoadVector256(inputChan + (ihTop + 2) * width + iw);
                    var r2_1 = Avx.LoadVector256(inputChan + (ihTop + 2) * width + iw + 1);
                    var r2_2 = Avx.LoadVector256(inputChan + (ihTop + 2) * width + iw + 2);

                    // For each of the 4 output channels: broadcast the 9 kernel
                    // values for THIS (oc, ic) pair and FMA into the corresponding
                    // accumulator. This is the loop-tiling pattern PyTorch's
                    // oneDNN AVX2 kernel uses (jit_avx2_conv_kernel_f32).
                    float* k0 = kBlock + 0 * inChannels * 9 + ic * 9;
                    float* k1 = kBlock + 1 * inChannels * 9 + ic * 9;
                    float* k2 = kBlock + 2 * inChannels * 9 + ic * 9;
                    float* k3 = kBlock + 3 * inChannels * 9 + ic * 9;

                    acc0 = Fma.MultiplyAdd(r0_0, Vector256.Create(k0[0]), acc0);
                    acc0 = Fma.MultiplyAdd(r0_1, Vector256.Create(k0[1]), acc0);
                    acc0 = Fma.MultiplyAdd(r0_2, Vector256.Create(k0[2]), acc0);
                    acc0 = Fma.MultiplyAdd(r1_0, Vector256.Create(k0[3]), acc0);
                    acc0 = Fma.MultiplyAdd(r1_1, Vector256.Create(k0[4]), acc0);
                    acc0 = Fma.MultiplyAdd(r1_2, Vector256.Create(k0[5]), acc0);
                    acc0 = Fma.MultiplyAdd(r2_0, Vector256.Create(k0[6]), acc0);
                    acc0 = Fma.MultiplyAdd(r2_1, Vector256.Create(k0[7]), acc0);
                    acc0 = Fma.MultiplyAdd(r2_2, Vector256.Create(k0[8]), acc0);

                    acc1 = Fma.MultiplyAdd(r0_0, Vector256.Create(k1[0]), acc1);
                    acc1 = Fma.MultiplyAdd(r0_1, Vector256.Create(k1[1]), acc1);
                    acc1 = Fma.MultiplyAdd(r0_2, Vector256.Create(k1[2]), acc1);
                    acc1 = Fma.MultiplyAdd(r1_0, Vector256.Create(k1[3]), acc1);
                    acc1 = Fma.MultiplyAdd(r1_1, Vector256.Create(k1[4]), acc1);
                    acc1 = Fma.MultiplyAdd(r1_2, Vector256.Create(k1[5]), acc1);
                    acc1 = Fma.MultiplyAdd(r2_0, Vector256.Create(k1[6]), acc1);
                    acc1 = Fma.MultiplyAdd(r2_1, Vector256.Create(k1[7]), acc1);
                    acc1 = Fma.MultiplyAdd(r2_2, Vector256.Create(k1[8]), acc1);

                    acc2 = Fma.MultiplyAdd(r0_0, Vector256.Create(k2[0]), acc2);
                    acc2 = Fma.MultiplyAdd(r0_1, Vector256.Create(k2[1]), acc2);
                    acc2 = Fma.MultiplyAdd(r0_2, Vector256.Create(k2[2]), acc2);
                    acc2 = Fma.MultiplyAdd(r1_0, Vector256.Create(k2[3]), acc2);
                    acc2 = Fma.MultiplyAdd(r1_1, Vector256.Create(k2[4]), acc2);
                    acc2 = Fma.MultiplyAdd(r1_2, Vector256.Create(k2[5]), acc2);
                    acc2 = Fma.MultiplyAdd(r2_0, Vector256.Create(k2[6]), acc2);
                    acc2 = Fma.MultiplyAdd(r2_1, Vector256.Create(k2[7]), acc2);
                    acc2 = Fma.MultiplyAdd(r2_2, Vector256.Create(k2[8]), acc2);

                    acc3 = Fma.MultiplyAdd(r0_0, Vector256.Create(k3[0]), acc3);
                    acc3 = Fma.MultiplyAdd(r0_1, Vector256.Create(k3[1]), acc3);
                    acc3 = Fma.MultiplyAdd(r0_2, Vector256.Create(k3[2]), acc3);
                    acc3 = Fma.MultiplyAdd(r1_0, Vector256.Create(k3[3]), acc3);
                    acc3 = Fma.MultiplyAdd(r1_1, Vector256.Create(k3[4]), acc3);
                    acc3 = Fma.MultiplyAdd(r1_2, Vector256.Create(k3[5]), acc3);
                    acc3 = Fma.MultiplyAdd(r2_0, Vector256.Create(k3[6]), acc3);
                    acc3 = Fma.MultiplyAdd(r2_1, Vector256.Create(k3[7]), acc3);
                    acc3 = Fma.MultiplyAdd(r2_2, Vector256.Create(k3[8]), acc3);
                }

                // Store all 4 accumulators ONCE at the end of the ic reduction.
                Avx.Store(outBlock + 0 * ocStride + oh * outWidth + ow, acc0);
                Avx.Store(outBlock + 1 * ocStride + oh * outWidth + ow, acc1);
                Avx.Store(outBlock + 2 * ocStride + oh * outWidth + ow, acc2);
                Avx.Store(outBlock + 3 * ocStride + oh * outWidth + ow, acc3);
            }

            // Tail: scalar interior columns from owSimdEnd to outWidth-2.
            for (int ow = owSimdEnd; ow < owEnd; ow++)
            {
                for (int oc = 0; oc < 4; oc++)
                {
                    float* outChan = outBlock + oc * ocStride;
                    float* kChan = kBlock + oc * inChannels * 9;
                    outChan[oh * outWidth + ow] = ScalarConv3x3At(input, kChan,
                        inChannels, icSpatialStride, height, width, oh, ow, 1, 1);
                }
            }

            // Right boundary col (ow = outWidth - 1, scalar with boundary check).
            for (int oc = 0; oc < 4; oc++)
            {
                float* outChan = outBlock + oc * ocStride;
                float* kChan = kBlock + oc * inChannels * 9;
                outChan[oh * outWidth + outWidth - 1] = ScalarConv3x3At(input, kChan,
                    inChannels, icSpatialStride, height, width, oh, outWidth - 1, 1, 1);
            }
        }
    }

    /// <summary>Scalar 3×3 conv at one output position with full boundary handling.</summary>
    [MethodImpl(HotInline)]
    private static unsafe float ScalarConv3x3At(
        float* input, float* kernelOc,
        int inChannels, int icSpatialStride, int height, int width,
        int oh, int ow, int padH, int padW)
    {
        float sum = 0f;
        int ihBase = oh - padH;
        int iwBase = ow - padW;
        for (int ic = 0; ic < inChannels; ic++)
        {
            float* inputChan = input + ic * icSpatialStride;
            float* kChan = kernelOc + ic * 9;
            for (int kh = 0; kh < 3; kh++)
            {
                int ih = ihBase + kh;
                if (ih < 0 || ih >= height) continue;
                for (int kw = 0; kw < 3; kw++)
                {
                    int iw = iwBase + kw;
                    if (iw < 0 || iw >= width) continue;
                    sum += inputChan[ih * width + iw] * kChan[kh * 3 + kw];
                }
            }
        }
        return sum;
    }

    /// <summary>Process a full boundary row (oh = 0 or oh = outHeight-1) for one output channel.</summary>
    [MethodImpl(HotInline)]
    private static unsafe void ProcessConv3x3BoundaryRowAllIc(
        float* input, float* kernelOc, float* outChan,
        int inChannels, int icSpatialStride,
        int height, int width, int outWidth, int oh)
    {
        for (int ow = 0; ow < outWidth; ow++)
        {
            outChan[oh * outWidth + ow] += ScalarConv3x3At(input, kernelOc,
                inChannels, icSpatialStride, height, width, oh, ow, 1, 1);
        }
    }

    /// <summary>
    /// Performs 1x1 convolution (pointwise) using AVX2 intrinsics.
    /// </summary>
    public static unsafe void Conv1x1(
        float* input, float* kernel, float* output,
        int batch, int inChannels, int height, int width, int outChannels)
    {
        int spatialSize = height * width;

        for (int b = 0; b < batch; b++)
        {
            float* inputBatch = input + b * inChannels * spatialSize;
            float* outputBatch = output + b * outChannels * spatialSize;

            // 1x1 conv is essentially matrix multiply: [outChannels, inChannels] @ [inChannels, spatialSize]
            // Use GEMM-like approach with AVX2
            Conv1x1Gemm(inputBatch, kernel, outputBatch, outChannels, inChannels, spatialSize);
        }
    }

    [MethodImpl(HotInline)]
    private static unsafe void Conv3x3Stride1SingleChannel(
        float* input, float* kernelOc, float* outputChannel,
        int inChannels, int height, int width, int outHeight, int outWidth,
        int padH, int padW, int dilationH, int dilationW)
    {
        // Clear output
        int outputSize = outHeight * outWidth;
        new Span<float>(outputChannel, outputSize).Clear();

        // Process each input channel
        for (int ic = 0; ic < inChannels; ic++)
        {
            float* inputChannel = input + ic * height * width;
            float* kernelChannel = kernelOc + ic * 9;

            // Load 3x3 kernel into registers
            if (UseFma)
            {
                Conv3x3SingleChannelFma(inputChannel, kernelChannel, outputChannel,
                    height, width, outHeight, outWidth, padH, padW, dilationH, dilationW);
            }
            else
            {
                Conv3x3SingleChannelAvx2(inputChannel, kernelChannel, outputChannel,
                    height, width, outHeight, outWidth, padH, padW, dilationH, dilationW);
            }
        }
    }

    [MethodImpl(HotInline)]
    private static unsafe void Conv3x3SingleChannelFma(
        float* input, float* kernel, float* output,
        int height, int width, int outHeight, int outWidth,
        int padH, int padW, int dilationH, int dilationW)
    {
        // Load kernel values into SIMD registers (broadcast)
        Vector256<float> k00 = Vector256.Create(kernel[0]);
        Vector256<float> k01 = Vector256.Create(kernel[1]);
        Vector256<float> k02 = Vector256.Create(kernel[2]);
        Vector256<float> k10 = Vector256.Create(kernel[3]);
        Vector256<float> k11 = Vector256.Create(kernel[4]);
        Vector256<float> k12 = Vector256.Create(kernel[5]);
        Vector256<float> k20 = Vector256.Create(kernel[6]);
        Vector256<float> k21 = Vector256.Create(kernel[7]);
        Vector256<float> k22 = Vector256.Create(kernel[8]);

        // Pre-allocate boundary handling buffer outside loop to avoid CA2014
        float* boundaryBuffer = stackalloc float[8];

        // Fast path for no-dilation case (most common)
        if (dilationH == 1 && dilationW == 1)
        {
            // Process interior rows (no top/bottom boundary)
            int ohStart = padH > 0 ? padH : 0;
            int ohEnd = outHeight - (padH > 0 ? padH : 0);
            if (ohEnd > outHeight) ohEnd = outHeight;

            // Handle boundary rows at top
            for (int topOh = 0; topOh < ohStart && topOh < outHeight; topOh++)
            {
                ProcessBoundaryRow(input, kernel, output + topOh * outWidth,
                    height, width, outWidth, topOh, padH, padW, boundaryBuffer);
            }

            // Fast interior processing with fully unrolled kernel
            // Process 2 output rows at a time for better instruction-level parallelism
            int oh = ohStart;
            for (; oh + 1 < ohEnd; oh += 2)
            {
                int ih0 = oh - padH;
                float* inputRow0 = input + ih0 * width;
                float* inputRow1 = input + (ih0 + 1) * width;
                float* inputRow2 = input + (ih0 + 2) * width;
                float* inputRow3 = input + (ih0 + 3) * width;
                float* outputRow0 = output + oh * outWidth;
                float* outputRow1 = output + (oh + 1) * outWidth;

                // Interior columns (no boundary handling needed)
                int owStart = padW > 0 ? padW : 0;
                int owEnd = outWidth - (padW > 0 ? padW : 0) - 7;

                // Handle left boundary columns
                for (int leftOw = 0; leftOw < owStart && leftOw < outWidth; leftOw++)
                {
                    outputRow0[leftOw] += ComputeScalarConv3x3(input, kernel, height, width, oh, leftOw, padH, padW);
                    outputRow1[leftOw] += ComputeScalarConv3x3(input, kernel, height, width, oh + 1, leftOw, padH, padW);
                }

                // Process 8 elements at a time in the interior - 2 rows simultaneously
                int ow = owStart;
                for (; ow < owEnd; ow += 8)
                {
                    int iw = ow - padW;

                    // Prefetch next cache line (64 bytes ahead = 16 floats)
                    if (UseSse41 && ow + 16 < owEnd)
                    {
                        Sse.Prefetch1(inputRow0 + iw + 16);
                        Sse.Prefetch1(inputRow1 + iw + 16);
                        Sse.Prefetch1(inputRow2 + iw + 16);
                        Sse.Prefetch1(inputRow3 + iw + 16);
                    }

                    // Load vectors for row 0 output
                    Vector256<float> r0_0 = Avx.LoadVector256(inputRow0 + iw);
                    Vector256<float> r0_1 = Avx.LoadVector256(inputRow0 + iw + 1);
                    Vector256<float> r0_2 = Avx.LoadVector256(inputRow0 + iw + 2);
                    Vector256<float> r1_0 = Avx.LoadVector256(inputRow1 + iw);
                    Vector256<float> r1_1 = Avx.LoadVector256(inputRow1 + iw + 1);
                    Vector256<float> r1_2 = Avx.LoadVector256(inputRow1 + iw + 2);
                    Vector256<float> r2_0 = Avx.LoadVector256(inputRow2 + iw);
                    Vector256<float> r2_1 = Avx.LoadVector256(inputRow2 + iw + 1);
                    Vector256<float> r2_2 = Avx.LoadVector256(inputRow2 + iw + 2);

                    // Convolution for first output row
                    Vector256<float> acc0 = Fma.MultiplyAdd(r0_0, k00, Vector256<float>.Zero);
                    acc0 = Fma.MultiplyAdd(r0_1, k01, acc0);
                    acc0 = Fma.MultiplyAdd(r0_2, k02, acc0);
                    acc0 = Fma.MultiplyAdd(r1_0, k10, acc0);
                    acc0 = Fma.MultiplyAdd(r1_1, k11, acc0);
                    acc0 = Fma.MultiplyAdd(r1_2, k12, acc0);
                    acc0 = Fma.MultiplyAdd(r2_0, k20, acc0);
                    acc0 = Fma.MultiplyAdd(r2_1, k21, acc0);
                    acc0 = Fma.MultiplyAdd(r2_2, k22, acc0);

                    // Load additional row for second output
                    Vector256<float> r3_0 = Avx.LoadVector256(inputRow3 + iw);
                    Vector256<float> r3_1 = Avx.LoadVector256(inputRow3 + iw + 1);
                    Vector256<float> r3_2 = Avx.LoadVector256(inputRow3 + iw + 2);

                    // Convolution for second output row (reuse r1, r2 rows)
                    Vector256<float> acc1 = Fma.MultiplyAdd(r1_0, k00, Vector256<float>.Zero);
                    acc1 = Fma.MultiplyAdd(r1_1, k01, acc1);
                    acc1 = Fma.MultiplyAdd(r1_2, k02, acc1);
                    acc1 = Fma.MultiplyAdd(r2_0, k10, acc1);
                    acc1 = Fma.MultiplyAdd(r2_1, k11, acc1);
                    acc1 = Fma.MultiplyAdd(r2_2, k12, acc1);
                    acc1 = Fma.MultiplyAdd(r3_0, k20, acc1);
                    acc1 = Fma.MultiplyAdd(r3_1, k21, acc1);
                    acc1 = Fma.MultiplyAdd(r3_2, k22, acc1);

                    // Store both output rows
                    Vector256<float> current0 = Avx.LoadVector256(outputRow0 + ow);
                    Vector256<float> current1 = Avx.LoadVector256(outputRow1 + ow);
                    Avx.Store(outputRow0 + ow, Avx.Add(current0, acc0));
                    Avx.Store(outputRow1 + ow, Avx.Add(current1, acc1));
                }

                // Handle right boundary columns
                for (; ow < outWidth; ow++)
                {
                    outputRow0[ow] += ComputeScalarConv3x3(input, kernel, height, width, oh, ow, padH, padW);
                    outputRow1[ow] += ComputeScalarConv3x3(input, kernel, height, width, oh + 1, ow, padH, padW);
                }
            }

            // Handle remaining single row if ohEnd - ohStart is odd
            for (; oh < ohEnd; oh++)
            {
                int ih0 = oh - padH;
                float* inputRow0 = input + ih0 * width;
                float* inputRow1 = input + (ih0 + 1) * width;
                float* inputRow2 = input + (ih0 + 2) * width;
                float* outputRow = output + oh * outWidth;

                int owStart = padW > 0 ? padW : 0;
                int owEnd = outWidth - (padW > 0 ? padW : 0) - 7;

                for (int leftOw = 0; leftOw < owStart && leftOw < outWidth; leftOw++)
                {
                    outputRow[leftOw] += ComputeScalarConv3x3(input, kernel, height, width, oh, leftOw, padH, padW);
                }

                int ow2 = owStart;
                for (; ow2 < owEnd; ow2 += 8)
                {
                    int iw = ow2 - padW;
                    Vector256<float> r0_0 = Avx.LoadVector256(inputRow0 + iw);
                    Vector256<float> r0_1 = Avx.LoadVector256(inputRow0 + iw + 1);
                    Vector256<float> r0_2 = Avx.LoadVector256(inputRow0 + iw + 2);
                    Vector256<float> r1_0 = Avx.LoadVector256(inputRow1 + iw);
                    Vector256<float> r1_1 = Avx.LoadVector256(inputRow1 + iw + 1);
                    Vector256<float> r1_2 = Avx.LoadVector256(inputRow1 + iw + 2);
                    Vector256<float> r2_0 = Avx.LoadVector256(inputRow2 + iw);
                    Vector256<float> r2_1 = Avx.LoadVector256(inputRow2 + iw + 1);
                    Vector256<float> r2_2 = Avx.LoadVector256(inputRow2 + iw + 2);

                    Vector256<float> acc = Fma.MultiplyAdd(r0_0, k00, Vector256<float>.Zero);
                    acc = Fma.MultiplyAdd(r0_1, k01, acc);
                    acc = Fma.MultiplyAdd(r0_2, k02, acc);
                    acc = Fma.MultiplyAdd(r1_0, k10, acc);
                    acc = Fma.MultiplyAdd(r1_1, k11, acc);
                    acc = Fma.MultiplyAdd(r1_2, k12, acc);
                    acc = Fma.MultiplyAdd(r2_0, k20, acc);
                    acc = Fma.MultiplyAdd(r2_1, k21, acc);
                    acc = Fma.MultiplyAdd(r2_2, k22, acc);

                    Vector256<float> current = Avx.LoadVector256(outputRow + ow2);
                    Avx.Store(outputRow + ow2, Avx.Add(current, acc));
                }

                for (; ow2 < outWidth; ow2++)
                {
                    outputRow[ow2] += ComputeScalarConv3x3(input, kernel, height, width, oh, ow2, padH, padW);
                }
            }

            // Handle boundary rows at bottom
            for (int bottomOh = ohEnd; bottomOh < outHeight; bottomOh++)
            {
                ProcessBoundaryRow(input, kernel, output + bottomOh * outWidth,
                    height, width, outWidth, bottomOh, padH, padW, boundaryBuffer);
            }
        }
        else
        {
            // Dilation case - use original general loop
            Conv3x3SingleChannelFmaWithDilation(input, kernel, output,
                height, width, outHeight, outWidth, padH, padW, dilationH, dilationW, boundaryBuffer);
        }
    }

    [MethodImpl(HotInline)]
    private static unsafe float ComputeScalarConv3x3(float* input, float* kernel,
        int height, int width, int oh, int ow, int padH, int padW)
    {
        float sum = 0f;
        int ihBase = oh - padH;
        int iwBase = ow - padW;

        for (int kh = 0; kh < 3; kh++)
        {
            int ih = ihBase + kh;
            if (ih < 0 || ih >= height) continue;

            for (int kw = 0; kw < 3; kw++)
            {
                int iw = iwBase + kw;
                if (iw >= 0 && iw < width)
                {
#if NET5_0_OR_GREATER
                    sum = UseFma
                        ? MathF.FusedMultiplyAdd(input[ih * width + iw], kernel[kh * 3 + kw], sum)
                        : sum + (input[ih * width + iw] * kernel[kh * 3 + kw]);
#else
                    sum += input[ih * width + iw] * kernel[kh * 3 + kw];
#endif
                }
            }
        }
        return sum;
    }

    [MethodImpl(HotInline)]
    private static unsafe void ProcessBoundaryRow(float* input, float* kernel, float* outputRow,
        int height, int width, int outWidth, int oh, int padH, int padW, float* boundaryBuffer)
    {
        for (int ow = 0; ow < outWidth; ow++)
        {
            outputRow[ow] += ComputeScalarConv3x3(input, kernel, height, width, oh, ow, padH, padW);
        }
    }

    [MethodImpl(HotInline)]
    private static unsafe void Conv3x3SingleChannelFmaWithDilation(
        float* input, float* kernel, float* output,
        int height, int width, int outHeight, int outWidth,
        int padH, int padW, int dilationH, int dilationW, float* boundaryBuffer)
    {
        // General case with dilation support
        for (int oh = 0; oh < outHeight; oh++)
        {
            int ihBase = oh - padH;
            float* outputRow = output + oh * outWidth;

            int ow = 0;
            int vectorEnd = outWidth - 7;

            for (; ow < vectorEnd; ow += 8)
            {
                Vector256<float> acc = Vector256.Create(0f);
                int iwBase = ow - padW;

                for (int kh = 0; kh < 3; kh++)
                {
                    int ih = ihBase + kh * dilationH;
                    if (ih < 0 || ih >= height) continue;

                    float* inputRow = input + ih * width;

                    for (int kw = 0; kw < 3; kw++)
                    {
                        int iwStart = iwBase + kw * dilationW;
                        float kVal = kernel[kh * 3 + kw];

                        Vector256<float> inputVec;
                        if (iwStart >= 0 && iwStart + 7 < width)
                        {
                            inputVec = Avx.LoadVector256(inputRow + iwStart);
                        }
                        else
                        {
                            for (int i = 0; i < 8; i++)
                            {
                                int iw = iwStart + i;
                                boundaryBuffer[i] = (iw >= 0 && iw < width) ? inputRow[iw] : 0f;
                            }
                            inputVec = Avx.LoadVector256(boundaryBuffer);
                        }

                        Vector256<float> kVec = Vector256.Create(kVal);
                        acc = Fma.MultiplyAdd(inputVec, kVec, acc);
                    }
                }

                Vector256<float> current = Avx.LoadVector256(outputRow + ow);
                Avx.Store(outputRow + ow, Avx.Add(current, acc));
            }

            for (; ow < outWidth; ow++)
            {
                outputRow[ow] += ComputeScalarConv3x3WithDilation(input, kernel, height, width, oh, ow, padH, padW, dilationH, dilationW);
            }
        }
    }

    [MethodImpl(HotInline)]
    private static unsafe float ComputeScalarConv3x3WithDilation(float* input, float* kernel,
        int height, int width, int oh, int ow, int padH, int padW, int dilationH, int dilationW)
    {
        float sum = 0f;
        int ihBase = oh - padH;
        int iwBase = ow - padW;

        for (int kh = 0; kh < 3; kh++)
        {
            int ih = ihBase + kh * dilationH;
            if (ih < 0 || ih >= height) continue;

            for (int kw = 0; kw < 3; kw++)
            {
                int iw = iwBase + kw * dilationW;
                if (iw >= 0 && iw < width)
                {
#if NET5_0_OR_GREATER
                    sum = UseFma
                        ? MathF.FusedMultiplyAdd(input[ih * width + iw], kernel[kh * 3 + kw], sum)
                        : sum + (input[ih * width + iw] * kernel[kh * 3 + kw]);
#else
                    sum += input[ih * width + iw] * kernel[kh * 3 + kw];
#endif
                }
            }
        }
        return sum;
    }

    [MethodImpl(HotInline)]
    private static unsafe void Conv3x3SingleChannelAvx2(
        float* input, float* kernel, float* output,
        int height, int width, int outHeight, int outWidth,
        int padH, int padW, int dilationH, int dilationW)
    {
        // Load kernel values into SIMD registers (broadcast)
        Vector256<float> k00 = Vector256.Create(kernel[0]);
        Vector256<float> k01 = Vector256.Create(kernel[1]);
        Vector256<float> k02 = Vector256.Create(kernel[2]);
        Vector256<float> k10 = Vector256.Create(kernel[3]);
        Vector256<float> k11 = Vector256.Create(kernel[4]);
        Vector256<float> k12 = Vector256.Create(kernel[5]);
        Vector256<float> k20 = Vector256.Create(kernel[6]);
        Vector256<float> k21 = Vector256.Create(kernel[7]);
        Vector256<float> k22 = Vector256.Create(kernel[8]);

        // Pre-allocate boundary handling buffer outside loop to avoid CA2014
        float* boundaryBuffer = stackalloc float[8];

        // Fast path for no-dilation case (most common)
        if (dilationH == 1 && dilationW == 1)
        {
            // Process interior rows (no top/bottom boundary)
            int ohStart = padH > 0 ? padH : 0;
            int ohEnd = outHeight - (padH > 0 ? padH : 0);
            if (ohEnd > outHeight) ohEnd = outHeight;

            // Handle boundary rows at top
            for (int oh = 0; oh < ohStart && oh < outHeight; oh++)
            {
                ProcessBoundaryRow(input, kernel, output + oh * outWidth,
                    height, width, outWidth, oh, padH, padW, boundaryBuffer);
            }

            // Fast interior processing with fully unrolled kernel
            for (int oh = ohStart; oh < ohEnd; oh++)
            {
                int ih0 = oh - padH;
                float* inputRow0 = input + ih0 * width;
                float* inputRow1 = input + (ih0 + 1) * width;
                float* inputRow2 = input + (ih0 + 2) * width;
                float* outputRow = output + oh * outWidth;

                // Interior columns (no boundary handling needed)
                int owStart = padW > 0 ? padW : 0;
                int owEnd = outWidth - (padW > 0 ? padW : 0) - 7;

                // Handle left boundary columns
                for (int leftOw = 0; leftOw < owStart && leftOw < outWidth; leftOw++)
                {
                    outputRow[leftOw] += ComputeScalarConv3x3(input, kernel, height, width, oh, leftOw, padH, padW);
                }

                // Process 8 elements at a time in the interior
                int ow = owStart;
                for (; ow < owEnd; ow += 8)
                {
                    int iw = ow - padW;

                    // Load vectors from each input row
                    Vector256<float> r0_0 = Avx.LoadVector256(inputRow0 + iw);
                    Vector256<float> r0_1 = Avx.LoadVector256(inputRow0 + iw + 1);
                    Vector256<float> r0_2 = Avx.LoadVector256(inputRow0 + iw + 2);
                    Vector256<float> r1_0 = Avx.LoadVector256(inputRow1 + iw);
                    Vector256<float> r1_1 = Avx.LoadVector256(inputRow1 + iw + 1);
                    Vector256<float> r1_2 = Avx.LoadVector256(inputRow1 + iw + 2);
                    Vector256<float> r2_0 = Avx.LoadVector256(inputRow2 + iw);
                    Vector256<float> r2_1 = Avx.LoadVector256(inputRow2 + iw + 1);
                    Vector256<float> r2_2 = Avx.LoadVector256(inputRow2 + iw + 2);

                    // Fully unrolled 3x3 convolution without FMA
                    Vector256<float> acc = Avx.Multiply(r0_0, k00);
                    acc = Avx.Add(acc, Avx.Multiply(r0_1, k01));
                    acc = Avx.Add(acc, Avx.Multiply(r0_2, k02));
                    acc = Avx.Add(acc, Avx.Multiply(r1_0, k10));
                    acc = Avx.Add(acc, Avx.Multiply(r1_1, k11));
                    acc = Avx.Add(acc, Avx.Multiply(r1_2, k12));
                    acc = Avx.Add(acc, Avx.Multiply(r2_0, k20));
                    acc = Avx.Add(acc, Avx.Multiply(r2_1, k21));
                    acc = Avx.Add(acc, Avx.Multiply(r2_2, k22));

                    // Add to output
                    Vector256<float> current = Avx.LoadVector256(outputRow + ow);
                    Avx.Store(outputRow + ow, Avx.Add(current, acc));
                }

                // Handle right boundary columns
                for (; ow < outWidth; ow++)
                {
                    outputRow[ow] += ComputeScalarConv3x3(input, kernel, height, width, oh, ow, padH, padW);
                }
            }

            // Handle boundary rows at bottom
            for (int oh = ohEnd; oh < outHeight; oh++)
            {
                ProcessBoundaryRow(input, kernel, output + oh * outWidth,
                    height, width, outWidth, oh, padH, padW, boundaryBuffer);
            }
        }
        else
        {
            // Dilation case - use general loop
            Conv3x3SingleChannelAvx2WithDilation(input, kernel, output,
                height, width, outHeight, outWidth, padH, padW, dilationH, dilationW, boundaryBuffer);
        }
    }

    [MethodImpl(HotInline)]
    private static unsafe void Conv3x3SingleChannelAvx2WithDilation(
        float* input, float* kernel, float* output,
        int height, int width, int outHeight, int outWidth,
        int padH, int padW, int dilationH, int dilationW, float* boundaryBuffer)
    {
        // General case with dilation support
        for (int oh = 0; oh < outHeight; oh++)
        {
            int ihBase = oh - padH;
            float* outputRow = output + oh * outWidth;

            int ow = 0;
            int vectorEnd = outWidth - 7;

            for (; ow < vectorEnd; ow += 8)
            {
                Vector256<float> acc = Vector256.Create(0f);
                int iwBase = ow - padW;

                for (int kh = 0; kh < 3; kh++)
                {
                    int ih = ihBase + kh * dilationH;
                    if (ih < 0 || ih >= height) continue;

                    float* inputRow = input + ih * width;

                    for (int kw = 0; kw < 3; kw++)
                    {
                        int iwStart = iwBase + kw * dilationW;
                        float kVal = kernel[kh * 3 + kw];

                        Vector256<float> inputVec;
                        if (iwStart >= 0 && iwStart + 7 < width)
                        {
                            inputVec = Avx.LoadVector256(inputRow + iwStart);
                        }
                        else
                        {
                            for (int i = 0; i < 8; i++)
                            {
                                int iw = iwStart + i;
                                boundaryBuffer[i] = (iw >= 0 && iw < width) ? inputRow[iw] : 0f;
                            }
                            inputVec = Avx.LoadVector256(boundaryBuffer);
                        }

                        Vector256<float> kVec = Vector256.Create(kVal);
                        acc = Avx.Add(acc, Avx.Multiply(inputVec, kVec));
                    }
                }

                Vector256<float> current = Avx.LoadVector256(outputRow + ow);
                Avx.Store(outputRow + ow, Avx.Add(current, acc));
            }

            for (; ow < outWidth; ow++)
            {
                outputRow[ow] += ComputeScalarConv3x3WithDilation(input, kernel, height, width, oh, ow, padH, padW, dilationH, dilationW);
            }
        }
    }

    [MethodImpl(HotInline)]
    private static unsafe void Conv1x1Gemm(
        float* input, float* kernel, float* output,
        int outChannels, int inChannels, int spatialSize)
    {
        // 1x1 conv is GEMM: output[oc, spatial] = sum_ic(kernel[oc, ic] * input[ic, spatial])
        //
        // #209 close-parity: register-resident accumulator across the ic loop.
        // Previously this loop did `output[oc, s] += kernel[oc, ic] * input[ic, s]`,
        // re-loading and re-storing the output row on every ic iteration —
        // a load+fma+store round-trip that bottlenecks on L1 bandwidth instead
        // of FMA throughput. The new form holds the accumulator vector in a
        // ymm register, FMAs across all ic values, then stores ONCE at the
        // end. Same idea as the Conv3x3Stride1Pad1_OcBlock4 fast path.
        //
        // 4-oc unrolling: when outChannels >= 4, hold 4 accumulator vectors
        // per spatial chunk and reuse the input load 4× (saves 75% of input
        // reads from memory).

        // Clear output
        new Span<float>(output, outChannels * spatialSize).Clear();

        if (UseFma && outChannels >= 4 && (outChannels & 3) == 0)
        {
            // 4-oc-blocked path: process 4 output channels per inner iteration.
            int spatialEnd8 = spatialSize - 7;
            for (int ocb = 0; ocb < outChannels; ocb += 4)
            {
                float* out0 = output + (ocb + 0) * spatialSize;
                float* out1 = output + (ocb + 1) * spatialSize;
                float* out2 = output + (ocb + 2) * spatialSize;
                float* out3 = output + (ocb + 3) * spatialSize;
                float* k0 = kernel + (ocb + 0) * inChannels;
                float* k1 = kernel + (ocb + 1) * inChannels;
                float* k2 = kernel + (ocb + 2) * inChannels;
                float* k3 = kernel + (ocb + 3) * inChannels;

                int s = 0;
                for (; s < spatialEnd8; s += 8)
                {
                    Vector256<float> acc0 = Vector256<float>.Zero;
                    Vector256<float> acc1 = Vector256<float>.Zero;
                    Vector256<float> acc2 = Vector256<float>.Zero;
                    Vector256<float> acc3 = Vector256<float>.Zero;
                    for (int ic = 0; ic < inChannels; ic++)
                    {
                        Vector256<float> inVec = Avx.LoadVector256(input + ic * spatialSize + s);
                        acc0 = Fma.MultiplyAdd(inVec, Vector256.Create(k0[ic]), acc0);
                        acc1 = Fma.MultiplyAdd(inVec, Vector256.Create(k1[ic]), acc1);
                        acc2 = Fma.MultiplyAdd(inVec, Vector256.Create(k2[ic]), acc2);
                        acc3 = Fma.MultiplyAdd(inVec, Vector256.Create(k3[ic]), acc3);
                    }
                    Avx.Store(out0 + s, acc0);
                    Avx.Store(out1 + s, acc1);
                    Avx.Store(out2 + s, acc2);
                    Avx.Store(out3 + s, acc3);
                }
                // Tail: scalar over remaining spatial elements
                for (; s < spatialSize; s++)
                {
                    float a0 = 0f, a1 = 0f, a2 = 0f, a3 = 0f;
                    for (int ic = 0; ic < inChannels; ic++)
                    {
                        float v = input[ic * spatialSize + s];
                        a0 += v * k0[ic]; a1 += v * k1[ic];
                        a2 += v * k2[ic]; a3 += v * k3[ic];
                    }
                    out0[s] = a0; out1[s] = a1; out2[s] = a2; out3[s] = a3;
                }
            }
            return;
        }

        // Fallback: per-oc register-resident accumulator (still better than
        // the previous load+fma+store round-trip).
        int spatialEnd8b = spatialSize - 7;
        for (int oc = 0; oc < outChannels; oc++)
        {
            float* outChan = output + oc * spatialSize;
            float* kRow = kernel + oc * inChannels;
            int s = 0;
            for (; s < spatialEnd8b; s += 8)
            {
                Vector256<float> acc = Vector256<float>.Zero;
                for (int ic = 0; ic < inChannels; ic++)
                {
                    Vector256<float> inVec = Avx.LoadVector256(input + ic * spatialSize + s);
                    Vector256<float> kVec = Vector256.Create(kRow[ic]);
                    acc = UseFma ? Fma.MultiplyAdd(inVec, kVec, acc) : Avx.Add(acc, Avx.Multiply(inVec, kVec));
                }
                Avx.Store(outChan + s, acc);
            }
            for (; s < spatialSize; s++)
            {
                float a = 0f;
                for (int ic = 0; ic < inChannels; ic++)
                    a += kRow[ic] * input[ic * spatialSize + s];
                outChan[s] = a;
            }
        }
    }
}
