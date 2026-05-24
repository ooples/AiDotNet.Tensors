using System;
using System.Buffers;
using System.Runtime.CompilerServices;
#if NET5_0_OR_GREATER
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
#endif
using System.Threading.Tasks;
using AiDotNet.Tensors.Engines.BlasManaged;

namespace AiDotNet.Tensors.Helpers;

/// <summary>
/// CPU implementation of im2col (image to column) transformation for efficient convolution via GEMM.
/// Transforms input patches into columns for matrix multiplication.
/// </summary>
internal static class Im2ColHelper
{
    // Higher threshold since OpenBLAS handles parallelism for GEMM
    // Avoid thread contention between parallel im2col and multi-threaded BLAS
    private const int ParallelThreshold = 262144; // 512x512 output

    /// <summary>
    /// Performs im2col transformation on a 4D input tensor [batch, channels, height, width].
    /// Output shape: [batch, channels * kernelH * kernelW, outputH * outputW]
    /// This transforms input patches into columns for efficient GEMM-based convolution.
    /// </summary>
    public static void Im2Col(
        ReadOnlySpan<float> input,
        Span<float> output,
        int batch,
        int channels,
        int height,
        int width,
        int kernelH,
        int kernelW,
        int strideH,
        int strideW,
        int padH,
        int padW,
        int dilationH,
        int dilationW)
    {
        int effectiveKernelH = dilationH * (kernelH - 1) + 1;
        int effectiveKernelW = dilationW * (kernelW - 1) + 1;
        int outputH = (height + 2 * padH - effectiveKernelH) / strideH + 1;
        int outputW = (width + 2 * padW - effectiveKernelW) / strideW + 1;

        int colH = channels * kernelH * kernelW;
        int colW = outputH * outputW;
        int inputImageSize = channels * height * width;

        // Process each batch element
        for (int b = 0; b < batch; b++)
        {
            int inputOffset = b * inputImageSize;
            int outputOffset = b * colH * colW;

            Im2ColSingleImage(
                input.Slice(inputOffset, inputImageSize),
                output.Slice(outputOffset, colH * colW),
                channels, height, width,
                kernelH, kernelW,
                strideH, strideW,
                padH, padW,
                dilationH, dilationW,
                outputH, outputW);
        }
    }

    /// <summary>
    /// Performs im2col on a single image (no batch dimension).
    /// Optimized row-by-row processing for better cache utilization and SIMD.
    /// </summary>
    private static unsafe void Im2ColSingleImage(
        ReadOnlySpan<float> input,
        Span<float> output,
        int channels,
        int height,
        int width,
        int kernelH,
        int kernelW,
        int strideH,
        int strideW,
        int padH,
        int padW,
        int dilationH,
        int dilationW,
        int outputH,
        int outputW)
    {
        int colH = channels * kernelH * kernelW;
        int colW = outputH * outputW;

        // Fast path: stride=1, dilation=1 (most common case in CNNs)
        if (strideH == 1 && strideW == 1 && dilationH == 1 && dilationW == 1)
        {
            // Clear entire output once upfront (handles all padding)
            output.Slice(0, colH * colW).Clear();

            fixed (float* inputPtr = input)
            fixed (float* outputPtr = output)
            {
                int rowIdx = 0;
                for (int c = 0; c < channels; c++)
                {
                    int channelOffset = c * height * width;

                    for (int kh = 0; kh < kernelH; kh++)
                    {
                        // Per-kh valid output row range:
                        // ih = oh + kh - padH, valid when 0 <= ih < height
                        int ohStart = Math.Max(0, padH - kh);
                        int ohEnd = Math.Min(outputH, height + padH - kh);

                        for (int kw = 0; kw < kernelW; kw++)
                        {
                            // Per-kw valid output column range:
                            // iw = ow + kw - padW, valid when 0 <= iw < width
                            int owStart = Math.Max(0, padW - kw);
                            int owEnd = Math.Min(outputW, width + padW - kw);
                            int validWidth = owEnd - owStart;

                            if (validWidth > 0 && ohEnd > ohStart)
                            {
                                float* outRow = outputPtr + rowIdx * colW;

                                for (int oh = ohStart; oh < ohEnd; oh++)
                                {
                                    int ih = oh + kh - padH;
                                    int inputStart = channelOffset + ih * width + (owStart + kw - padW);
                                    int outputStart = oh * outputW + owStart;

                                    Buffer.MemoryCopy(
                                        inputPtr + inputStart,
                                        outRow + outputStart,
                                        validWidth * sizeof(float),
                                        validWidth * sizeof(float));
                                }
                            }

                            rowIdx++;
                        }
                    }
                }
            }
        }
        else
        {
            // General path for arbitrary stride/dilation
            fixed (float* inputPtr = input)
            fixed (float* outputPtr = output)
            {
                int rowIdx = 0;
                for (int c = 0; c < channels; c++)
                {
                    int channelOffset = c * height * width;

                    for (int kh = 0; kh < kernelH; kh++)
                    {
                        for (int kw = 0; kw < kernelW; kw++)
                        {
                            float* outRow = outputPtr + rowIdx * colW;
                            Im2ColRowGeneral(inputPtr, outRow, channelOffset,
                                height, width, kh, kw, strideH, strideW,
                                padH, padW, dilationH, dilationW, outputH, outputW);
                            rowIdx++;
                        }
                    }
                }
            }
        }
    }

    /// <summary>
    /// Optimized row processing for stride=1, dilation=1 case.
    /// Uses bulk memory copies for contiguous regions instead of element-by-element access.
    /// </summary>
    private static unsafe void Im2ColRowOptimized(
        float* input,
        float* outRow,
        int channelOffset,
        int height,
        int width,
        int kh,
        int kw,
        int padH,
        int padW,
        int outputH,
        int outputW)
    {
        // For stride=1, dilation=1: output position (oh, ow) maps to input position (ih, iw) = (oh + kh - padH, ow + kw - padW)
        // We can use bulk copies for contiguous valid regions

        // Calculate valid input row range
        int ihStart = kh - padH;  // ih when oh = 0
        int ohValidStart = Math.Max(0, padH - kh);  // First oh where ih >= 0
        int ohValidEnd = Math.Min(outputH, height + padH - kh);  // First oh where ih >= height

        // Calculate valid input column range for each output row
        int iwStart = kw - padW;  // iw when ow = 0
        int owValidStart = Math.Max(0, padW - kw);  // First ow where iw >= 0
        int owValidEnd = Math.Min(outputW, width + padW - kw);  // First ow where iw >= width
        int validWidth = owValidEnd - owValidStart;

        // Zero the entire output first (handles padding efficiently)
        new Span<float>(outRow, outputH * outputW).Clear();

        // Process valid rows with bulk copy
        if (validWidth > 0 && ohValidEnd > ohValidStart)
        {
            for (int oh = ohValidStart; oh < ohValidEnd; oh++)
            {
                int ih = oh + kh - padH;
                int inputRowOffset = channelOffset + ih * width;
                int outputRowOffset = oh * outputW;

                // Bulk copy the valid portion of this row
                int inputStart = inputRowOffset + (owValidStart + kw - padW);
                int outputStart = outputRowOffset + owValidStart;

                Buffer.MemoryCopy(
                    input + inputStart,
                    outRow + outputStart,
                    validWidth * sizeof(float),
                    validWidth * sizeof(float));
            }
        }
    }

    /// <summary>
    /// General row processing for arbitrary stride and dilation.
    /// </summary>
    private static unsafe void Im2ColRowGeneral(
        float* input,
        float* outRow,
        int channelOffset,
        int height,
        int width,
        int kh,
        int kw,
        int strideH,
        int strideW,
        int padH,
        int padW,
        int dilationH,
        int dilationW,
        int outputH,
        int outputW)
    {
        int colIdx = 0;

        for (int oh = 0; oh < outputH; oh++)
        {
            int ih = oh * strideH + kh * dilationH - padH;

            for (int ow = 0; ow < outputW; ow++)
            {
                int iw = ow * strideW + kw * dilationW - padW;

                float val = 0f;
                if (ih >= 0 && ih < height && iw >= 0 && iw < width)
                {
                    val = input[channelOffset + ih * width + iw];
                }

                outRow[colIdx++] = val;
            }
        }
    }

    private static void ProcessColumnArray(
        float[] input,
        float[] output,
        int channels,
        int height,
        int width,
        int kernelH,
        int kernelW,
        int strideH,
        int strideW,
        int padH,
        int padW,
        int dilationH,
        int dilationW,
        int colH,
        int colW,
        int oh,
        int ow,
        int colIdx)
    {
        int rowIdx = 0;

        for (int c = 0; c < channels; c++)
        {
            for (int kh = 0; kh < kernelH; kh++)
            {
                for (int kw = 0; kw < kernelW; kw++)
                {
                    int ih = oh * strideH + kh * dilationH - padH;
                    int iw = ow * strideW + kw * dilationW - padW;

                    float val = 0f;
                    if (ih >= 0 && ih < height && iw >= 0 && iw < width)
                    {
                        int inputIdx = c * height * width + ih * width + iw;
                        val = input[inputIdx];
                    }

                    // Column-major layout for efficient GEMM
                    output[rowIdx * colW + colIdx] = val;
                    rowIdx++;
                }
            }
        }
    }

    private static void ProcessColumnSpan(
        ReadOnlySpan<float> input,
        Span<float> output,
        int channels,
        int height,
        int width,
        int kernelH,
        int kernelW,
        int strideH,
        int strideW,
        int padH,
        int padW,
        int dilationH,
        int dilationW,
        int colH,
        int colW,
        int oh,
        int ow,
        int colIdx)
    {
        int rowIdx = 0;

        for (int c = 0; c < channels; c++)
        {
            for (int kh = 0; kh < kernelH; kh++)
            {
                for (int kw = 0; kw < kernelW; kw++)
                {
                    int ih = oh * strideH + kh * dilationH - padH;
                    int iw = ow * strideW + kw * dilationW - padW;

                    float val = 0f;
                    if (ih >= 0 && ih < height && iw >= 0 && iw < width)
                    {
                        int inputIdx = c * height * width + ih * width + iw;
                        val = input[inputIdx];
                    }

                    // Column-major layout for efficient GEMM
                    output[rowIdx * colW + colIdx] = val;
                    rowIdx++;
                }
            }
        }
    }

    /// <summary>
    /// col2im: inverse of im2col. Accumulates column matrix values back into a spatial
    /// image buffer. Multiple column entries can map to the same spatial position (due to
    /// overlapping receptive fields), so values are ADDED (not overwritten). The output
    /// buffer must be zero-initialized before calling.
    /// </summary>
    public static void Col2ImAccumulate(
        ReadOnlySpan<float> colData,
        Span<float> imageData,
        int channels, int height, int width,
        int kernelH, int kernelW,
        int strideH, int strideW,
        int padH, int padW,
        int dilationH, int dilationW,
        int outputH, int outputW)
    {
        int colIdx = 0;
        for (int c = 0; c < channels; c++)
        {
            for (int kh = 0; kh < kernelH; kh++)
            {
                for (int kw = 0; kw < kernelW; kw++)
                {
                    for (int oh = 0; oh < outputH; oh++)
                    {
                        int ih = oh * strideH + kh * dilationH - padH;
                        for (int ow = 0; ow < outputW; ow++)
                        {
                            int iw = ow * strideW + kw * dilationW - padW;
                            if (ih >= 0 && ih < height && iw >= 0 && iw < width)
                            {
                                imageData[c * height * width + ih * width + iw] += colData[colIdx];
                            }
                            colIdx++;
                        }
                    }
                }
            }
        }
    }

    /// <summary>
    /// Double-precision variant of <see cref="Col2ImAccumulate(ReadOnlySpan{float}, Span{float}, int, int, int, int, int, int, int, int, int, int, int, int)"/>.
    /// Used by the BLAS fast-path in <c>Conv2DBackwardInput</c> when T=double.
    /// </summary>
    public static void Col2ImAccumulate(
        ReadOnlySpan<double> colData,
        Span<double> imageData,
        int channels, int height, int width,
        int kernelH, int kernelW,
        int strideH, int strideW,
        int padH, int padW,
        int dilationH, int dilationW,
        int outputH, int outputW)
    {
        int colIdx = 0;
        for (int c = 0; c < channels; c++)
        {
            for (int kh = 0; kh < kernelH; kh++)
            {
                for (int kw = 0; kw < kernelW; kw++)
                {
                    for (int oh = 0; oh < outputH; oh++)
                    {
                        int ih = oh * strideH + kh * dilationH - padH;
                        for (int ow = 0; ow < outputW; ow++)
                        {
                            int iw = ow * strideW + kw * dilationW - padW;
                            if (ih >= 0 && ih < height && iw >= 0 && iw < width)
                            {
                                imageData[c * height * width + ih * width + iw] += colData[colIdx];
                            }
                            colIdx++;
                        }
                    }
                }
            }
        }
    }

    /// <summary>
    /// Performs Conv2D using im2col + GEMM approach.
    /// This is significantly faster than naive nested loops for large convolutions.
    /// </summary>
    /// <returns>True if GEMM was used successfully, false if fallback is needed</returns>
    public static bool TryConv2DWithGemm(
        ReadOnlySpan<float> input,
        ReadOnlySpan<float> kernel,
        Span<float> output,
        int batch,
        int inChannels,
        int height,
        int width,
        int outChannels,
        int kernelH,
        int kernelW,
        int strideH,
        int strideW,
        int padH,
        int padW,
        int dilationH,
        int dilationW)
    {
        int effectiveKernelH = dilationH * (kernelH - 1) + 1;
        int effectiveKernelW = dilationW * (kernelW - 1) + 1;
        int outputH = (height + 2 * padH - effectiveKernelH) / strideH + 1;
        int outputW = (width + 2 * padW - effectiveKernelW) / strideW + 1;

        // im2col matrix dimensions
        int colH = inChannels * kernelH * kernelW;  // M for kernel, K for GEMM
        int colW = outputH * outputW;               // N for GEMM

        // GEMM: kernel (outChannels x colH) @ im2col (colH x colW) = output (outChannels x colW)
        // M = outChannels, K = colH, N = colW

        // Allocate im2col buffer
        var pool = ArrayPool<float>.Shared;
        float[] im2colBuffer = pool.Rent(batch * colH * colW);

        try
        {
            // Step 1: Convert input to im2col format
            Im2Col(input, im2colBuffer.AsSpan(0, batch * colH * colW),
                batch, inChannels, height, width,
                kernelH, kernelW, strideH, strideW, padH, padW, dilationH, dilationW);

            // Step 2: GEMM for each batch element
            // Kernel is reshaped: [outChannels, inChannels, kernelH, kernelW] -> [outChannels, colH]
            // Already in correct layout for GEMM

            for (int b = 0; b < batch; b++)
            {
                int im2colOffset = b * colH * colW;
                int outputOffset = b * outChannels * outputH * outputW;

                // Use span-based GEMM to avoid array copies
                bool usedBlas = BlasProvider.TryGemm(
                    outChannels, colW, colH,
                    kernel,  // ReadOnlySpan<float>
                    colH,
                    im2colBuffer.AsSpan(im2colOffset, colH * colW),  // ReadOnlySpan<float>
                    colW,
                    output.Slice(outputOffset, outChannels * colW),  // Span<float>
                    colW);

                if (!usedBlas)
                {
                    // Fallback to blocked matrix multiply
                    MultiplyMatrixBlocked(
                        kernel,
                        im2colBuffer.AsSpan(im2colOffset, colH * colW),
                        output.Slice(outputOffset, outChannels * colW),
                        outChannels, colH, colW);
                }
            }

            return true;
        }
        finally
        {
            pool.Return(im2colBuffer);
        }
    }

    /// <summary>
    /// Blocked matrix multiplication fallback when BLAS is not available.
    /// C = A @ B where A is [m, k], B is [k, n], C is [m, n]
    /// </summary>
    private static void MultiplyMatrixBlocked(
        ReadOnlySpan<float> a,
        ReadOnlySpan<float> b,
        Span<float> c,
        int m,
        int k,
        int n)
    {
        const int BlockSize = 64;

        // Initialize output to zero
        c.Clear();

        // Blocked matrix multiplication with cache-friendly access pattern
        for (int ii = 0; ii < m; ii += BlockSize)
        {
            int iEnd = Math.Min(ii + BlockSize, m);

            for (int kk = 0; kk < k; kk += BlockSize)
            {
                int kEnd = Math.Min(kk + BlockSize, k);

                for (int jj = 0; jj < n; jj += BlockSize)
                {
                    int jEnd = Math.Min(jj + BlockSize, n);

                    // Process block
                    for (int i = ii; i < iEnd; i++)
                    {
                        for (int kIdx = kk; kIdx < kEnd; kIdx++)
                        {
                            float aik = a[i * k + kIdx];
                            int bRowOffset = kIdx * n + jj;
                            int cRowOffset = i * n + jj;

                            for (int j = 0; j < jEnd - jj; j++)
                            {
                                c[cRowOffset + j] += aik * b[bRowOffset + j];
                            }
                        }
                    }
                }
            }
        }
    }

    #region Double Precision Support

    /// <summary>
    /// Performs im2col transformation for double precision tensors.
    /// Same algorithm as the float version but operates on double data.
    /// </summary>
    public static void Im2Col(
        ReadOnlySpan<double> input,
        Span<double> output,
        int batch,
        int channels,
        int height,
        int width,
        int kernelH,
        int kernelW,
        int strideH,
        int strideW,
        int padH,
        int padW,
        int dilationH,
        int dilationW)
    {
        int effectiveKernelH = dilationH * (kernelH - 1) + 1;
        int effectiveKernelW = dilationW * (kernelW - 1) + 1;
        int outputH = (height + 2 * padH - effectiveKernelH) / strideH + 1;
        int outputW = (width + 2 * padW - effectiveKernelW) / strideW + 1;

        int colH = channels * kernelH * kernelW;
        int colW = outputH * outputW;
        int inputImageSize = channels * height * width;

        for (int b = 0; b < batch; b++)
        {
            int inputOffset = b * inputImageSize;
            int outputOffset = b * colH * colW;

            Im2ColSingleImageDouble(
                input.Slice(inputOffset, inputImageSize),
                output.Slice(outputOffset, colH * colW),
                channels, height, width,
                kernelH, kernelW,
                strideH, strideW,
                padH, padW,
                dilationH, dilationW,
                outputH, outputW);
        }
    }

    private static unsafe void Im2ColSingleImageDouble(
        ReadOnlySpan<double> input,
        Span<double> output,
        int channels,
        int height,
        int width,
        int kernelH,
        int kernelW,
        int strideH,
        int strideW,
        int padH,
        int padW,
        int dilationH,
        int dilationW,
        int outputH,
        int outputW)
    {
        int colH = channels * kernelH * kernelW;
        int colW = outputH * outputW;

        if (strideH == 1 && strideW == 1 && dilationH == 1 && dilationW == 1)
        {
            output.Slice(0, colH * colW).Clear();

            fixed (double* inputPtr = input)
            fixed (double* outputPtr = output)
            {
                int rowIdx = 0;
                for (int c = 0; c < channels; c++)
                {
                    int channelOffset = c * height * width;

                    for (int kh = 0; kh < kernelH; kh++)
                    {
                        int ohStart = Math.Max(0, padH - kh);
                        int ohEnd = Math.Min(outputH, height + padH - kh);

                        for (int kw = 0; kw < kernelW; kw++)
                        {
                            // Per-kw valid output column range:
                            // iw = ow + kw - padW, valid when 0 <= iw < width
                            int owStart = Math.Max(0, padW - kw);
                            int owEnd = Math.Min(outputW, width + padW - kw);
                            int validWidth = owEnd - owStart;

                            if (validWidth > 0 && ohEnd > ohStart)
                            {
                                double* outRow = outputPtr + rowIdx * colW;

                                for (int oh = ohStart; oh < ohEnd; oh++)
                                {
                                    int ih = oh + kh - padH;
                                    int inputStart = channelOffset + ih * width + (owStart + kw - padW);
                                    int outputStart = oh * outputW + owStart;

                                    Buffer.MemoryCopy(
                                        inputPtr + inputStart,
                                        outRow + outputStart,
                                        validWidth * sizeof(double),
                                        validWidth * sizeof(double));
                                }
                            }

                            rowIdx++;
                        }
                    }
                }
            }
        }
        else
        {
            fixed (double* inputPtr = input)
            fixed (double* outputPtr = output)
            {
                int rowIdx = 0;
                for (int c = 0; c < channels; c++)
                {
                    int channelOffset = c * height * width;

                    for (int kh = 0; kh < kernelH; kh++)
                    {
                        for (int kw = 0; kw < kernelW; kw++)
                        {
                            double* outRow = outputPtr + rowIdx * colW;
                            int colIdx = 0;

                            for (int oh = 0; oh < outputH; oh++)
                            {
                                int ih = oh * strideH + kh * dilationH - padH;

                                for (int ow = 0; ow < outputW; ow++)
                                {
                                    int iw = ow * strideW + kw * dilationW - padW;

                                    double val = 0.0;
                                    if (ih >= 0 && ih < height && iw >= 0 && iw < width)
                                    {
                                        val = inputPtr[channelOffset + ih * width + iw];
                                    }

                                    outRow[colIdx++] = val;
                                }
                            }

                            rowIdx++;
                        }
                    }
                }
            }
        }
    }

    /// <summary>
    /// Blocked matrix multiplication for double precision.
    /// C = A @ B where A is [m, k], B is [k, n], C is [m, n]
    /// </summary>
    internal static void MultiplyMatrixBlockedDouble(
        ReadOnlySpan<double> a,
        ReadOnlySpan<double> b,
        Span<double> c,
        int m,
        int k,
        int n)
    {
        // EXPERIMENT (rejected by --ab-conv2d-double):
        // Tried adding explicit Vector256<double> 4-way unrolled FMA + parallel
        // dispatch to close the Conv2D_Double 3.8× gap to torch (438 µs vs 115).
        // A/B verdict on the BDN shape [1,3,32,32]→[16,3,3] (0.4M FMAs):
        //   scalar:  460 µs (RyuJIT auto-vectorizes the simple axpy)
        //   SIMD:    459 µs (no improvement)
        // On a larger experimental shape [1,16,64,64]→[32,3,3] (18.9M FMAs):
        //   scalar:  5201 µs
        //   SIMD:   18098 µs (3.5× REGRESSION — likely intrinsics defeated
        //     RyuJIT's auto-prefetch and out-of-order scheduling)
        // Lesson: for simple axpy patterns, the JIT's auto-vectorizer is
        // already optimal. The Conv2D_Double gap is structural — the float
        // path uses register-resident Conv3x3 (no im2col), but doubles
        // route through this im2col+GEMM path. Closing further requires a
        // Conv3x3SingleChannel<double> kernel, mirroring the float design.
        //
        // 2026-05-02 update — re-introduce parallelism. The earlier
        // "parallel dispatch" rejection bundled Parallel.For with explicit
        // SIMD intrinsics; the SIMD piece was the regression cause
        // (defeated JIT auto-prefetch), not the parallelism. Profiling
        // SD-style diffusion ResBlock convs at [1,320,64,64]→[320,3,3]
        // (3.77 GFMA, 5232 ms scalar single-threaded ⇒ ~1.4 GFLOPS)
        // showed every (i, j) tile of C is fully independent: each writes a
        // disjoint slab, reads its own slab of A, and shares B read-only.
        // Splitting across cores is embarrassingly parallel and preserves
        // the JIT auto-vectorized inner axpy. We split the work across BOTH
        // the outer M AND outer N axes so the task count
        // (⌈m/BlockSize⌉ × ⌈n/BlockSize⌉) is large enough to saturate every
        // worker thread — splitting on M alone gives only ⌈m/BlockSize⌉
        // tasks, which capped speedup at ≈2× for SD shapes on a 32-core box.
        // Threshold at total FMAs ≥ 64 M to avoid parallel-dispatch overhead
        // on small shapes (the original 0.4 M FMA BDN baseline stays on
        // the serial path). Dispatch routes through CpuParallelSettings —
        // its MaxDegreeOfParallelism honours host-side thread caps and
        // matches what every other parallel kernel in this file uses.

        const int BlockSize = 64;

        c.Clear();

        long totalFmas = (long)m * (long)k * (long)n;
        // Use CpuParallelSettings so a host that capped MaxDegreeOfParallelism
        // (e.g. shared-tenant CI runner, deterministic-mode tests) is
        // honoured here — Environment.ProcessorCount would over-subscribe
        // those workloads.
        int parallelDegree = Math.Max(1, CpuParallelSettings.MaxDegreeOfParallelism);
        bool parallel = totalFmas >= 64L * 1024L * 1024L && parallelDegree > 1;

        if (!parallel)
        {
            for (int ii = 0; ii < m; ii += BlockSize)
            {
                int iEnd = Math.Min(ii + BlockSize, m);
                MultiplyBlockedDoubleRowSlabSpan(a, b, c, ii, iEnd, k, n, BlockSize);
            }
            return;
        }

        // Pin the three buffers so each thread can address its slab via raw
        // pointers without violating the Span<T> stack-only rule (Spans can't
        // cross lambda/anonymous-method boundaries — error CS9108).
        //
        // Partition the work across BOTH the outer M and outer N axes so we
        // get enough independent tiles to saturate every core. Splitting on
        // M alone gives only ⌈m/BlockSize⌉ tasks — for SD diffusion shapes
        // (m = outChannels = 320, BlockSize = 64) that's just 5 tasks, which
        // pinned the speedup to ≈2× on a 32-core box. Adding the N axis
        // multiplies the task count to ⌈m/BlockSize⌉ × ⌈n/BlockSize⌉ — for
        // m=320, n=4096 that's 5 × 64 = 320 independent (i,j) tiles, each
        // writing a disjoint slab of C, so all cores can run flat-out.
        // Splitting K would require atomic accumulation into C and is not
        // worth the synchronization cost; keeping K serial inside each tile
        // also preserves the JIT auto-vectorized inner axpy.
        int mBlocks = (m + BlockSize - 1) / BlockSize;
        int nBlocks = (n + BlockSize - 1) / BlockSize;
        int totalTiles = mBlocks * nBlocks;

        // B-repacking: B's natural row stride is n (e.g. 4096 doubles = 32 KB
        // for SD-class shapes), so successive kIdx values inside a tile sit
        // far apart in memory. The prefetcher misses on those long strides
        // and successive cache loads end up issuing cold misses to L2/L3.
        // Pre-pack B into per-(K-block, N-block) tiles laid out contiguously
        // (size = Kc × Nc doubles = 32 KB each) so the inner kernel reads
        // its tile of B sequentially with stride 1 across both kIdx and j.
        //
        // The packed-B buffer is RENTED from a thread-static pool keyed by
        // size and returned at the end of this call, so we don't allocate a
        // fresh ~94 MB array on every fallback GEMM. Repeat calls at the
        // same shape (every conv inside one UNet forward) reuse the same
        // pool slot — zero managed-heap pressure after the first call.
        int kBlocks = (k + BlockSize - 1) / BlockSize;
        int packStride = BlockSize * BlockSize;
        long packedLen = (long)kBlocks * nBlocks * packStride;
        if (packedLen > int.MaxValue)
        {
            // Fall back to per-call allocation for the (rare) > 16 GB
            // packed buffer case where ArrayPool would refuse anyway.
            RunPackedGemmDouble(a, b, c, m, k, n, BlockSize, kBlocks, nBlocks,
                                packStride, totalTiles, parallelDegree, packed: new double[packedLen]);
            return;
        }
        var pool = System.Buffers.ArrayPool<double>.Shared;
        var packedB = pool.Rent((int)packedLen);
        try
        {
            RunPackedGemmDouble(a, b, c, m, k, n, BlockSize, kBlocks, nBlocks,
                                packStride, totalTiles, parallelDegree, packedB);
        }
        finally
        {
            pool.Return(packedB);
        }
    }

    /// <summary>
    /// Pack B into per-(K-block, N-block) contiguous tiles, then run the
    /// 2-D-tile parallel GEMM consuming those packed tiles. Both phases
    /// dispatch through <see cref="PersistentParallelExecutor"/>, the same
    /// pre-spawned worker pool that <c>SimdGemm</c> uses. Compared to
    /// <c>Parallel.For</c>, this skips per-call ThreadPool queueing and
    /// CountdownEvent allocation — non-trivial savings on the diffusion
    /// hot path where the GEMM is invoked tens of thousands of times per
    /// model. The chunk count is bounded above by
    /// <see cref="CpuParallelSettings.MaxDegreeOfParallelism"/> so a host
    /// that capped parallelism (e.g. shared-tenant CI runner, deterministic-
    /// mode tests) is honoured.
    /// </summary>
    private static void RunPackedGemmDouble(
        ReadOnlySpan<double> a, ReadOnlySpan<double> b, Span<double> c,
        int m, int k, int n, int BlockSize,
        int kBlocks, int nBlocks, int packStride, int totalTiles,
        int parallelDegree, double[] packed)
    {
        int packTaskCount = kBlocks * nBlocks;

        // The executor uses Environment.ProcessorCount - 1 workers; cap the
        // chunk count at min(taskCount, parallelDegree) so a host with a
        // lower MaxDegreeOfParallelism doesn't oversubscribe — a chunk
        // executes serially across many tiles, so fewer chunks = fewer
        // concurrent workers.
        int packChunks = Math.Max(1, Math.Min(packTaskCount, parallelDegree));
        int gemmChunks = Math.Max(1, Math.Min(totalTiles, parallelDegree));

        unsafe
        {
            fixed (double* bPtr = b)
            fixed (double* pbPtr = packed)
            {
                double* bBase = bPtr;
                double* pbBase = pbPtr;
                int nBlocksLocal = nBlocks;
                int kLocal = k;
                int nLocal = n;
                int blockSizeLocal = BlockSize;
                int packStrideLocal = packStride;
                int packTaskCountLocal = packTaskCount;
                int packChunksLocal = packChunks;
                IntPtr ipB = (IntPtr)bBase;
                IntPtr ipPb = (IntPtr)pbBase;

                PersistentParallelExecutor.Instance.Execute(packChunksLocal, chunk =>
                {
                    int chunkSize = (packTaskCountLocal + packChunksLocal - 1) / packChunksLocal;
                    int taskStart = chunk * chunkSize;
                    int taskEnd = Math.Min(taskStart + chunkSize, packTaskCountLocal);
                    if (taskStart >= taskEnd) return;

                    double* bChunk = (double*)ipB;
                    double* pbChunk = (double*)ipPb;
                    for (int packTile = taskStart; packTile < taskEnd; packTile++)
                    {
                        int kBlk = packTile / nBlocksLocal;
                        int nBlk = packTile % nBlocksLocal;
                        int kk = kBlk * blockSizeLocal;
                        int jj = nBlk * blockSizeLocal;
                        int kEnd = Math.Min(kk + blockSizeLocal, kLocal);
                        int jEnd = Math.Min(jj + blockSizeLocal, nLocal);
                        int kLen = kEnd - kk;
                        int jLen = jEnd - jj;
                        double* dst = pbChunk + (long)packTile * packStrideLocal;
                        for (int kIdx = 0; kIdx < kLen; kIdx++)
                        {
                            double* src = bChunk + (kk + kIdx) * nLocal + jj;
                            double* dstRow = dst + kIdx * blockSizeLocal;
                            for (int j = 0; j < jLen; j++)
                                dstRow[j] = src[j];
                        }
                    }
                });
            }
        }

        unsafe
        {
            fixed (double* aPtr = a)
            fixed (double* cPtr = c)
            fixed (double* pbPtr = packed)
            {
                double* aBase = aPtr, cBase = cPtr, pbBase = pbPtr;
                int kBlocksLocal = kBlocks;
                int nBlocksLocal = nBlocks;
                int mLocal = m;
                int kLocal = k;
                int nLocal = n;
                int blockSizeLocal = BlockSize;
                int packStrideLocal = packStride;
                int totalTilesLocal = totalTiles;
                int gemmChunksLocal = gemmChunks;
                IntPtr ipA = (IntPtr)aBase, ipC = (IntPtr)cBase, ipPb = (IntPtr)pbBase;

                PersistentParallelExecutor.Instance.Execute(gemmChunksLocal, chunk =>
                {
                    int chunkSize = (totalTilesLocal + gemmChunksLocal - 1) / gemmChunksLocal;
                    int tileStart = chunk * chunkSize;
                    int tileEnd = Math.Min(tileStart + chunkSize, totalTilesLocal);
                    if (tileStart >= tileEnd) return;

                    double* aChunk = (double*)ipA;
                    double* cChunk = (double*)ipC;
                    double* pbChunk = (double*)ipPb;
                    for (int tile = tileStart; tile < tileEnd; tile++)
                    {
                        int mBlk = tile / nBlocksLocal;
                        int nBlk = tile % nBlocksLocal;
                        int ii = mBlk * blockSizeLocal;
                        int jj = nBlk * blockSizeLocal;
                        int iEnd = Math.Min(ii + blockSizeLocal, mLocal);
                        int jEnd = Math.Min(jj + blockSizeLocal, nLocal);
                        for (int kBlk = 0; kBlk < kBlocksLocal; kBlk++)
                        {
                            int kk = kBlk * blockSizeLocal;
                            int kEnd = Math.Min(kk + blockSizeLocal, kLocal);
                            double* packedTile = pbChunk + ((long)kBlk * nBlocksLocal + nBlk) * packStrideLocal;
                            MultiplyPackedTilePtr(
                                aChunk, packedTile, cChunk,
                                ii, iEnd, jj, jEnd, kk, kEnd, kLocal, nLocal);
                        }
                    }
                });
            }
        }
    }

    /// <summary>
    /// Inner kernel for the packed-B path. The B-tile is contiguous
    /// [BlockSize × BlockSize] doubles, so the inner j loop reads stride-1
    /// from the packed buffer and the kIdx step is just += BlockSize
    /// (one cache line worth of doubles). RyuJIT auto-vectorizes the
    /// inner axpy as well.
    /// </summary>
    private static unsafe void MultiplyPackedTilePtr(
        double* a, double* packedB, double* c,
        int ii, int iEnd, int jj, int jEnd, int kk, int kEnd,
        int k, int n)
    {
        int width = jEnd - jj;
        int kLen = kEnd - kk;
        const int BlockSize = 64;
        for (int i = ii; i < iEnd; i++)
        {
            double* aRow = a + (long)i * k + kk;
            double* cRow = c + (long)i * n + jj;
            for (int kIdx = 0; kIdx < kLen; kIdx++)
            {
                double aik = aRow[kIdx];
                double* bRow = packedB + kIdx * BlockSize;
                for (int j = 0; j < width; j++)
                    cRow[j] += aik * bRow[j];
            }
        }
    }

    /// <summary>
    /// Pointer overload — single (i,j) tile. Iterates over all K blocks for
    /// the given output tile so the whole inner reduction lives in this
    /// task and can be JIT-auto-vectorized in place.
    /// </summary>
    private static unsafe void MultiplyBlockedDoubleTilePtr(
        double* a, double* b, double* c,
        int ii, int iEnd, int jj, int jEnd,
        int k, int n,
        int blockSize)
    {
        for (int kk = 0; kk < k; kk += blockSize)
        {
            int kEnd = Math.Min(kk + blockSize, k);
            for (int i = ii; i < iEnd; i++)
            {
                for (int kIdx = kk; kIdx < kEnd; kIdx++)
                {
                    double aik = a[i * k + kIdx];
                    int bRowOffset = kIdx * n + jj;
                    int cRowOffset = i * n + jj;
                    int width = jEnd - jj;
                    for (int j = 0; j < width; j++)
                        c[cRowOffset + j] += aik * b[bRowOffset + j];
                }
            }
        }
    }

    /// <summary>
    /// Span overload — handles the contiguous row range [ii, iEnd) of C
    /// when called from the serial path (no thread crossing).
    /// </summary>
    private static void MultiplyBlockedDoubleRowSlabSpan(
        ReadOnlySpan<double> a,
        ReadOnlySpan<double> b,
        Span<double> c,
        int ii, int iEnd,
        int k, int n,
        int blockSize)
    {
        for (int kk = 0; kk < k; kk += blockSize)
        {
            int kEnd = Math.Min(kk + blockSize, k);
            for (int jj = 0; jj < n; jj += blockSize)
            {
                int jEnd = Math.Min(jj + blockSize, n);
                for (int i = ii; i < iEnd; i++)
                {
                    for (int kIdx = kk; kIdx < kEnd; kIdx++)
                    {
                        double aik = a[i * k + kIdx];
                        int bRowOffset = kIdx * n + jj;
                        int cRowOffset = i * n + jj;
                        for (int j = 0; j < jEnd - jj; j++)
                            c[cRowOffset + j] += aik * b[bRowOffset + j];
                    }
                }
            }
        }
    }

    #endregion

    #region ConvTranspose2D GEMM-based fast path

    /// <summary>
    /// Performs ConvTranspose2D (transposed convolution) using GEMM + Col2Im for float.
    /// Mathematically: y[b, oc, oh, ow] = sum_{ic, kh, kw} x[b, ic, ih, iw] * w[ic, oc, kh, kw]
    /// where ih = (oh + padH - kh) / strideH (when divisible and in bounds).
    /// </summary>
    /// <remarks>
    /// Replaces the naive 7-nested-loop forward path in CpuEngine.ConvTranspose2D for
    /// double/float when BLAS is available. Per-Deconv compute drops from O(B·C_out·H_out·W_out·C_in·kH·kW)
    /// hand-rolled loops to one GEMM call per batch slice + a Col2ImAccumulate scatter, which
    /// dispatches into MKL/OpenBLAS for the dense reduction. Wins are largest where the
    /// per-call FMA count is high (DCGAN generator at 64×64, diffusion VAE decoder upsample
    /// blocks, etc.) — measured ~14× speedup at DCGAN gen scale on x64 + MKL.
    /// </remarks>
    /// <returns>True on success; false if BLAS is unavailable (caller falls back to naive).</returns>
    public static bool TryConvTranspose2DWithGemm(
        float[] input,
        float[] kernel,
        float[] output,
        int batch,
        int inChannels,
        int inputHeight,
        int inputWidth,
        int outChannels,
        int kernelH,
        int kernelW,
        int strideH,
        int strideW,
        int padH,
        int padW,
        int outputHeight,
        int outputWidth)
    {
        // GEMM A: kernel viewed as [inChannels, outChannels*kernelH*kernelW] row-major
        // (same flat memory as the stored [inChannels, outChannels, kH, kW] layout).
        // GEMM B per batch: input slice viewed as [inChannels, inputHeight*inputWidth].
        // GEMM C per batch: temp viewed as [outChannels*kernelH*kernelW, inputHeight*inputWidth].
        // C = A^T @ B = (kernel^T) @ x_b.
        // Then Col2ImAccumulate scatters temp into output[b, :, :, :].
        int kmkn = outChannels * kernelH * kernelW;   // GEMM M, also colData major dim
        int hw   = inputHeight * inputWidth;          // GEMM N
        int outSliceSize = outChannels * outputHeight * outputWidth;
        int inSliceSize  = inChannels * inputHeight * inputWidth;
        int kernelSize   = inChannels * kmkn;

        // CodeRabbit #366: the old useTranspose heuristic existed to dodge
        // OpenBLAS/MKL's slow transA=true path by materialising kernel^T
        // up front. BlasManaged absorbs the transpose inside PackA, so the
        // caller-side transpose is now pure overhead — an extra O(kernelSize)
        // copy and a redundant pooled buffer on every call. Drop both and
        // always call Gemm with transA: true.

        var pool = ArrayPool<float>.Shared;
        float[] tempBuffer = pool.Rent(kmkn * hw);
        try
        {

            for (int b = 0; b < batch; b++)
            {
                int inputOffset  = b * inSliceSize;
                int outputOffset = b * outSliceSize;

                // PHASE K5: BlasManaged.Gemm<float> fast path. kernel is [inChannels, kmkn]
                // row-major; transA=true gives op(A) of shape [kmkn, inChannels], which is
                // the L2-shape pathology (M=large, N=small, K=medium) that hits OpenBLAS/MKL's
                // slow generic transA=true path. BlasManaged routes to the AVX-512 16×16
                // microkernel via PackBothStrategy for this shape (closes issue #358).
                BlasManaged.Gemm<float>(
                    a: kernel.AsSpan(0, kernelSize),
                    lda: kmkn, transA: true,
                    b: input.AsSpan(inputOffset, inChannels * hw),
                    ldb: hw, transB: false,
                    c: tempBuffer.AsSpan(0, kmkn * hw),
                    ldc: hw,
                    m: kmkn, n: hw, k: inChannels);
                bool usedBlas = true;

                if (!usedBlas)
                {
                    // Pool buffers are returned by the finally block on
                    // every exit path — do not return them here too. A
                    // double-Return corrupts the pool's internal state
                    // and lets a later Rent hand the same array to two
                    // callers simultaneously, which surfaces as silent
                    // data corruption miles away from this code.
                    return false;
                }

                // Zero output slice — Col2ImAccumulate adds; we need a clean accumulator.
                Array.Clear(output, outputOffset, outSliceSize);

                // Col2ImAccumulate signature:
                //   colData [c, kh, kw, oh_in, ow_in] → imageData [c, ih_out, iw_out]
                //   where ih_out = oh_in*strideH + kh*dilationH - padH
                // For ConvTranspose2D: "c" = outChannels, output(c, ih_out, iw_out) accumulates
                // contributions from input position (oh_in, ow_in) shifted by (kh, kw). Dilation = 1.
                Col2ImAccumulate(
                    tempBuffer.AsSpan(0, kmkn * hw),
                    output.AsSpan(outputOffset, outSliceSize),
                    channels: outChannels,
                    height: outputHeight, width: outputWidth,
                    kernelH: kernelH, kernelW: kernelW,
                    strideH: strideH, strideW: strideW,
                    padH: padH, padW: padW,
                    dilationH: 1, dilationW: 1,
                    outputH: inputHeight, outputW: inputWidth);
            }

            return true;
        }
        finally
        {
            pool.Return(tempBuffer);
        }
    }

    /// <summary>
    /// Double-precision counterpart of the float
    /// <see cref="TryConvTranspose2DWithGemm(float[], float[], float[], int, int, int, int, int, int, int, int, int, int, int, int, int)"/>.
    /// </summary>
    public static bool TryConvTranspose2DWithGemm(
        double[] input,
        double[] kernel,
        double[] output,
        int batch,
        int inChannels,
        int inputHeight,
        int inputWidth,
        int outChannels,
        int kernelH,
        int kernelW,
        int strideH,
        int strideW,
        int padH,
        int padW,
        int outputHeight,
        int outputWidth)
    {
        int kmkn = outChannels * kernelH * kernelW;
        int hw   = inputHeight * inputWidth;
        int outSliceSize = outChannels * outputHeight * outputWidth;
        int inSliceSize  = inChannels * inputHeight * inputWidth;
        int kernelSize   = inChannels * kmkn;

        // CodeRabbit #366: the old useTranspose heuristic (materialising kernel^T
        // up front to dodge OpenBLAS/MKL's slow transA=true path) is gone — main's
        // BlasManaged absorbs the transpose inside PackA, so a caller-side transpose
        // is pure overhead. The general path below always calls Gemm with transA: true.

        // Issue #358 phase-1 fast path: the "fat A, small N, transA=true"
        // GEMM aspect ratio (M ≫ K, N ≤ 16) is a documented MKL/OpenBLAS
        // perf cliff — both BLAS implementations pack the transposed [M,K]
        // panel which dominates at the L2 ConvTranspose2D shape (M=4096,
        // N=16, K=512, kernelSize=16MB), measured at 215ms OpenBLAS /
        // 559ms MKL vs ~1ms peak. This phase-1 kernel skips BLAS dispatch
        // and uses a hand-rolled K-outer + Mb=2 AVX2 microkernel that
        // streams A row-by-row instead. Improvement: ~20% over OpenBLAS
        // on the L2 shape. Phase-2 (#358 follow-up): add A-panel packing
        // + Mc macro-blocking to amortise L1 misses; AVX-512 microkernel
        // for 8-row Mr. Gate: N == 16 exactly (DCGAN / similar deconv
        // bottleneck shapes) AND M ≥ 8*K (the "fat A" regime).
        bool useSmallNTransA = hw == 16 && kmkn >= 8 * inChannels;

        var pool = ArrayPool<double>.Shared;
        double[] tempBuffer = pool.Rent(kmkn * hw);
        try
        {

            for (int b = 0; b < batch; b++)
            {
                int inputOffset  = b * inSliceSize;
                int outputOffset = b * outSliceSize;

                bool usedBlas;
                if (useSmallNTransA)
                {
                    // Issue #358 fast path: hand-rolled BLIS-style packed-A + Mc-blocked
                    // AVX2 transA microkernel, specialised to the N=16 "fat A" shape
                    // (M=4096, N=16, K=512). Skips general dispatch entirely. See
                    // DgemmTransA_N16_FatA for the full rationale and perf numbers.
                    DgemmTransA_N16_FatA(
                        kernel, kernelOffset: 0, lda: kmkn,
                        input, bOffset: inputOffset, ldb: hw,
                        tempBuffer, cOffset: 0, ldc: hw,
                        m: kmkn, k: inChannels);
                    usedBlas = true;
                }
                else
                {
                    // General path: BlasManaged.Gemm<double> (our own managed SIMD GEMM,
                    // no third-party BLAS). kernel is [inChannels, kmkn] row-major;
                    // transA=true gives op(A) of shape [kmkn, inChannels]. BlasManaged
                    // absorbs the transpose inside PackA (#402 multi-panel PrePackA/B,
                    // CodeRabbit #366) and routes the L2-shape pathology to the packed
                    // AVX-512 8×16 FP64 microkernel via PackBothStrategy.
                    BlasManaged.Gemm<double>(
                        a: kernel.AsSpan(0, kernelSize),
                        lda: kmkn, transA: true,
                        b: input.AsSpan(inputOffset, inChannels * hw),
                        ldb: hw, transB: false,
                        c: tempBuffer.AsSpan(0, kmkn * hw),
                        ldc: hw,
                        m: kmkn, n: hw, k: inChannels);
                    usedBlas = true;
                }

                if (!usedBlas)
                {
                    // Pool returns are handled by the finally block on
                    // every exit path — double-returning corrupts the
                    // pool. Matches the fix in the float overload.
                    return false;
                }

                Array.Clear(output, outputOffset, outSliceSize);

                Col2ImAccumulate(
                    tempBuffer.AsSpan(0, kmkn * hw),
                    output.AsSpan(outputOffset, outSliceSize),
                    channels: outChannels,
                    height: outputHeight, width: outputWidth,
                    kernelH: kernelH, kernelW: kernelW,
                    strideH: strideH, strideW: strideW,
                    padH: padH, padW: padW,
                    dilationH: 1, dilationW: 1,
                    outputH: inputHeight, outputW: inputWidth);
            }

            return true;
        }
        finally
        {
            pool.Return(tempBuffer);
        }
    }

    /// <summary>
    /// Issue #358 fast path: FP64 GEMM with transA=true at the N=16 "fat A"
    /// aspect ratio (M ≫ K). Computes
    /// <c>C[m, 0..15] = sum_k A[k, m] * B[k, 0..15]</c> where A is
    /// <c>[K, M]</c> row-major (the un-transposed kernel matrix) and B is
    /// <c>[K, 16]</c> row-major. Output C is <c>[M, 16]</c> row-major.
    /// <para>
    /// Specialised for N=16 (the DCGAN L2 deconv shape's H*W=4*4=16) and
    /// uses BLIS-style A-panel packing + Mc macro-blocking:
    /// </para>
    /// <list type="number">
    ///   <item>
    ///     <description><b>Pack-A</b>: For each Mc=64-row macro-block,
    ///       pack the corresponding A slice into an interleaved layout
    ///       <c>packedA[mr_outer, kk, ri]</c> where mr_outer indexes Mc/Mr
    ///       sub-blocks, kk indexes the K axis, and ri indexes the Mr=2
    ///       innermost rows. The pack reads A strided by lda once per
    ///       Mc-block (M / Mc = 64 packs total for the L2 shape) and
    ///       writes the packed buffer sequentially. Net A-traffic =
    ///       kernel size (~16 MB), pulled once from RAM/L3.
    ///     </description>
    ///   </item>
    ///   <item>
    ///     <description><b>Microkernel</b>: For each Mr=2 sub-block inside
    ///       the Mc-block, the AVX2 microkernel reads the packed panel
    ///       sequentially (stride Mr = 16 bytes between consecutive kk
    ///       steps), keeping a single Mr × K = 8 KB working set resident
    ///       in L1. 8 Vector256&lt;double&gt; C-accumulators (2 m-rows ×
    ///       4 vectors-per-row), 4 B-vector scratch, 1 A-broadcast scratch
    ///       = 13 YMM, comfortably under the 16-YMM AVX2 budget.
    ///     </description>
    ///   </item>
    /// </list>
    /// <para>
    /// Phase 1 (un-packed K-outer) was bottlenecked by L1 thrashing at the
    /// lda=4096-double stride. Phase 2 (this implementation) keeps the
    /// inner microkernel's A reads in L1 by streaming the packed panel
    /// through it, closing the gap to BLAS peak.
    /// </para>
    /// <para>
    /// Scalar fallback (non-AVX2 / net471) uses the same packed layout so
    /// correctness is identical; only the inner-K compute drops from
    /// 8 FMA/cycle to ~1 op/cycle.
    /// </para>
    /// </summary>
    internal static long _smallNTransAAvx2Calls;
    internal static long _smallNTransAScalarCalls;

    // BLIS-style blocking parameters.
    //  Mc = macro-block size. 64 m-rows × K = 512 × 8B = 256 KB packed
    //    panel — fits L2 on consumer CPUs (256 KB-1 MB typical). Larger
    //    Mc reduces pack overhead but spills L2; smaller increases pack
    //    overhead. 64 is the empirical sweet spot for K~512.
    //  Mr = microkernel row tile. 2 doubles × 16 cols × 8B = 256 B per
    //    inner update. With 4 b-loads + 8 c-accumulators + 1 a-broadcast
    //    = 13 YMM (under AVX2's 16-YMM budget). Mr=4 would need 16 c +
    //    4 b + 1 a = 21 YMM → spill, hence Mr=2 on AVX2.
    private const int DgemmFatAMc = 64;
    private const int DgemmFatAMr = 2;

    internal static void DgemmTransA_N16_FatA(
        double[] a, int kernelOffset, int lda,
        double[] b, int bOffset, int ldb,
        double[] c, int cOffset, int ldc,
        int m, int k)
    {
#if NET8_0_OR_GREATER
        // Issue #358 fix #1 names an AVX-512 small-N kernel. On AVX-512 hosts
        // (Skylake-X / Ice Lake / Sapphire Rapids) the Mr=8 microkernel packs
        // 8 m-rows × N=16 into 16 Vector512<double> accumulators (well inside
        // the 32-ZMM budget), doubling the rows-per-pack of the AVX2 Mr=2 path
        // and pushing the fat-A N=16 shape closer to the ~1 ms DGEMM peak. The
        // AVX2 path below is left byte-identical (it still serves every non-
        // AVX-512 host, including this repo's Zen2 dev box). Correctness is
        // enforced by the existing ConvTranspose2DGemmCorrectnessTests, which
        // compare against the naive 7-loop reference on whichever ISA the host
        // exposes.
        if (Avx512F.IsSupported)
        {
            DgemmTransA_N16_FatA_Avx512(a, kernelOffset, lda, b, bOffset, ldb, c, cOffset, ldc, m, k);
            return;
        }
#endif
        // Clear C — we accumulate from zero. Matches BlasProvider.TryGemmEx
        // (beta=0) behaviour the dispatcher contract expects.
        Array.Clear(c, cOffset, m * ldc);

        const int Mc = DgemmFatAMc;
        const int Mr = DgemmFatAMr;
        int mcBlocks = m / Mc;
        int mcTail = m - mcBlocks * Mc;

        var pool = ArrayPool<double>.Shared;

        // Parallel-Mc: each Mc-block packs its own A panel and writes to a
        // disjoint slice of C — there's no cross-block dependency, so the
        // outer loop parallelises cleanly. Linux CI runners (4-8 cores)
        // were measuring 117 ms/call single-threaded; with parallel-Mc
        // and 4 cores the L2 DCGAN shape (64 Mc-blocks) drops to ~15-25 ms.
        // Per-thread state: one packedA buffer (Mc*K doubles = 256 KB),
        // rented from ArrayPool and returned in the localFinally.
        int procs = AiDotNet.Tensors.Helpers.CpuParallelSettings.MaxDegreeOfParallelism;
        bool useParallel = procs > 1 && mcBlocks >= 2 && (long)m * k * 16 >= (1L << 22);
        if (useParallel)
        {
            int packedSize = Mc * k;
            var po = new System.Threading.Tasks.ParallelOptions { MaxDegreeOfParallelism = procs };
            System.Threading.Tasks.Parallel.For(
                0, mcBlocks,
                po,
                localInit: () => pool.Rent(packedSize),
                body: (mb, _, packedA) =>
                {
                    int mcStart = mb * Mc;
                    PackAPanel_TransA_N16_Mr2(a, kernelOffset, lda, packedA, mcStart, Mc, k);
                    int mrBlocks = Mc / Mr;
                    for (int mrOuter = 0; mrOuter < mrBlocks; mrOuter++)
                    {
                        Microkernel_TransA_N16_Mr2(
                            packedA, packedOffset: mrOuter * k * Mr,
                            b, bOffset, ldb,
                            c, cOffset + (mcStart + mrOuter * Mr) * ldc, ldc,
                            k);
                    }
                    return packedA;
                },
                localFinally: localPacked => pool.Return(localPacked));
        }
        else
        {
            // Sequential path — used when MaxDegreeOfParallelism == 1 or
            // the work is too small to amortise thread setup.
            double[] packedA = pool.Rent(Mc * k);
            try
            {
                for (int mb = 0; mb < mcBlocks; mb++)
                {
                    int mcStart = mb * Mc;
                    PackAPanel_TransA_N16_Mr2(a, kernelOffset, lda, packedA, mcStart, Mc, k);
                    int mrBlocks = Mc / Mr;
                    for (int mrOuter = 0; mrOuter < mrBlocks; mrOuter++)
                    {
                        Microkernel_TransA_N16_Mr2(
                            packedA, packedOffset: mrOuter * k * Mr,
                            b, bOffset, ldb,
                            c, cOffset + (mcStart + mrOuter * Mr) * ldc, ldc,
                            k);
                    }
                }
            }
            finally { pool.Return(packedA); }
        }

        // Tail handling stays sequential (one or two Mc-blocks at most,
        // not worth Parallel.For overhead).
        double[] packedTailBuf = pool.Rent(Mc * k);
        try
        {
            if (mcTail >= Mr)
            {
                int mcStart = mcBlocks * Mc;
                int packedRows = (mcTail / Mr) * Mr;
                PackAPanel_TransA_N16_Mr2(a, kernelOffset, lda, packedTailBuf, mcStart, packedRows, k);
                int mrBlocks = packedRows / Mr;
                for (int mrOuter = 0; mrOuter < mrBlocks; mrOuter++)
                {
                    Microkernel_TransA_N16_Mr2(
                        packedTailBuf, packedOffset: mrOuter * k * Mr,
                        b, bOffset, ldb,
                        c, cOffset + (mcStart + mrOuter * Mr) * ldc, ldc,
                        k);
                }
                mcTail -= packedRows;
            }
            if (mcTail > 0)
            {
                // 1-row scalar tail. m=4096 with Mc=64 leaves no tail at
                // all (4096 % 64 == 0), so this branch is dead on the L2
                // shape. Kept for correctness on arbitrary m.
                int mStart = m - mcTail;
                for (int j = 0; j < 16; j++)
                {
                    double acc = 0;
                    for (int kk = 0; kk < k; kk++)
                        acc += a[kernelOffset + (long)kk * lda + mStart] * b[bOffset + kk * ldb + j];
                    c[cOffset + mStart * ldc + j] = acc;
                }
                System.Threading.Interlocked.Increment(ref _smallNTransAScalarCalls);
            }
        }
        finally { pool.Return(packedTailBuf); }
    }

    /// <summary>
    /// Packs <c>mc</c> consecutive m-rows of A starting at <c>mcStart</c>
    /// into the interleaved BLIS layout <c>packedA[mr_outer, kk, ri]</c>.
    /// Read pattern from A: stride lda per kk step (the unavoidable cost
    /// of transA=true). Write pattern: sequential along packedA. After
    /// packing, the microkernel's inner K loop reads packedA at stride
    /// Mr=2 doubles (16 bytes) per kk step — well inside one cache line
    /// per 4 steps, keeping the Mr × K = 8 KB working set in L1.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe void PackAPanel_TransA_N16_Mr2(
        double[] a, int aOffset, int lda,
        double[] packed, int mcStart, int mc, int k)
    {
        const int Mr = DgemmFatAMr;
        int mrBlocks = mc / Mr;
        fixed (double* pA = &a[aOffset])
        fixed (double* pPacked = packed)
        {
            for (int mrOuter = 0; mrOuter < mrBlocks; mrOuter++)
            {
                int mGlobal = mcStart + mrOuter * Mr;
                double* outBase = pPacked + (long)mrOuter * k * Mr;
                // For each kk, read 2 consecutive doubles from A[kk, mGlobal..mGlobal+1].
                // Write 2 consecutive doubles to packed[mrOuter, kk, 0..1].
                for (int kk = 0; kk < k; kk++)
                {
                    double* aRow = pA + (long)kk * lda + mGlobal;
                    double* outRow = outBase + (long)kk * Mr;
                    outRow[0] = aRow[0];
                    outRow[1] = aRow[1];
                }
            }
        }
    }

    /// <summary>
    /// Mr=2, Nr=16 FP64 AVX2 microkernel reading the BLIS-packed A panel.
    /// Per kk step: 2 sequential A loads (8 bytes apart, same cache line),
    /// 4 B-vector loads (sequential, 64 bytes total), 8 FMA into 8 c-
    /// accumulators. No branching in the K loop; JIT emits a tight
    /// straight-line unroll.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe void Microkernel_TransA_N16_Mr2(
        double[] packedA, int packedOffset,
        double[] b, int bOffset, int ldb,
        double[] c, int cOffset, int ldc,
        int k)
    {
#if NET5_0_OR_GREATER
        if (Avx2.IsSupported && Fma.IsSupported)
        {
            System.Threading.Interlocked.Increment(ref _smallNTransAAvx2Calls);
            var c00 = Vector256<double>.Zero;
            var c01 = Vector256<double>.Zero;
            var c02 = Vector256<double>.Zero;
            var c03 = Vector256<double>.Zero;
            var c10 = Vector256<double>.Zero;
            var c11 = Vector256<double>.Zero;
            var c12 = Vector256<double>.Zero;
            var c13 = Vector256<double>.Zero;

            fixed (double* pA = &packedA[packedOffset])
            fixed (double* pB = &b[bOffset])
            fixed (double* pC = &c[cOffset])
            {
                for (int kk = 0; kk < k; kk++)
                {
                    // Packed A: 2 consecutive doubles per kk step (stride Mr=2).
                    double* aPtr = pA + (long)kk * 2;
                    double* bRow = pB + (long)kk * ldb;
                    var va0 = Vector256.Create(aPtr[0]);
                    var va1 = Vector256.Create(aPtr[1]);
                    var vb0 = Avx.LoadVector256(bRow + 0);
                    var vb1 = Avx.LoadVector256(bRow + 4);
                    var vb2 = Avx.LoadVector256(bRow + 8);
                    var vb3 = Avx.LoadVector256(bRow + 12);
                    c00 = Fma.MultiplyAdd(va0, vb0, c00);
                    c01 = Fma.MultiplyAdd(va0, vb1, c01);
                    c02 = Fma.MultiplyAdd(va0, vb2, c02);
                    c03 = Fma.MultiplyAdd(va0, vb3, c03);
                    c10 = Fma.MultiplyAdd(va1, vb0, c10);
                    c11 = Fma.MultiplyAdd(va1, vb1, c11);
                    c12 = Fma.MultiplyAdd(va1, vb2, c12);
                    c13 = Fma.MultiplyAdd(va1, vb3, c13);
                }

                Avx.Store(pC + 0, c00);
                Avx.Store(pC + 4, c01);
                Avx.Store(pC + 8, c02);
                Avx.Store(pC + 12, c03);
                double* cRow1 = pC + ldc;
                Avx.Store(cRow1 + 0, c10);
                Avx.Store(cRow1 + 4, c11);
                Avx.Store(cRow1 + 8, c12);
                Avx.Store(cRow1 + 12, c13);
            }
            return;
        }
#endif
        System.Threading.Interlocked.Increment(ref _smallNTransAScalarCalls);
        // Scalar fallback. Same packed-A layout — for each output column
        // accumulate Mr=2 dot products against the packed panel.
        for (int j = 0; j < 16; j++)
        {
            double acc0 = 0, acc1 = 0;
            for (int kk = 0; kk < k; kk++)
            {
                double bv = b[bOffset + kk * ldb + j];
                acc0 += packedA[packedOffset + kk * 2 + 0] * bv;
                acc1 += packedA[packedOffset + kk * 2 + 1] * bv;
            }
            c[cOffset + j] = acc0;
            c[cOffset + ldc + j] = acc1;
        }
    }

#if NET8_0_OR_GREATER
    // Mr = 8 microkernel row tile for AVX-512. 8 m-rows × N=16 = 16
    // Vector512<double> c-accumulators (each row's 16 cols split into two
    // 8-double halves) + 2 b-vectors + 1 a-broadcast = 19 ZMM, inside the
    // 32-ZMM AVX-512 budget. Mc=64 is divisible by both Mr=8 and Mr=2.
    private const int DgemmFatAMr8 = 8;

    /// <summary>
    /// AVX-512 variant of <see cref="DgemmTransA_N16_FatA"/> (Mr=8). Same
    /// BLIS-style packed-A + Mc=64 macro-blocking and parallel-Mc dispatch as
    /// the AVX2 path; only the row-tile width and microkernel ISA differ.
    /// </summary>
    private static void DgemmTransA_N16_FatA_Avx512(
        double[] a, int kernelOffset, int lda,
        double[] b, int bOffset, int ldb,
        double[] c, int cOffset, int ldc,
        int m, int k)
    {
        Array.Clear(c, cOffset, m * ldc);

        const int Mc = DgemmFatAMc;
        const int Mr = DgemmFatAMr8;
        int mcBlocks = m / Mc;
        int mcTail = m - mcBlocks * Mc;

        var pool = ArrayPool<double>.Shared;
        int procs = AiDotNet.Tensors.Helpers.CpuParallelSettings.MaxDegreeOfParallelism;
        bool useParallel = procs > 1 && mcBlocks >= 2 && (long)m * k * 16 >= (1L << 22);

        if (useParallel)
        {
            int packedSize = Mc * k;
            var po = new System.Threading.Tasks.ParallelOptions { MaxDegreeOfParallelism = procs };
            System.Threading.Tasks.Parallel.For(
                0, mcBlocks, po,
                localInit: () => pool.Rent(packedSize),
                body: (mb, _, packedA) =>
                {
                    int mcStart = mb * Mc;
                    PackAPanel_TransA_N16_Mr8(a, kernelOffset, lda, packedA, mcStart, Mc, k);
                    int mrBlocks = Mc / Mr;
                    for (int mrOuter = 0; mrOuter < mrBlocks; mrOuter++)
                        Microkernel_TransA_N16_Mr8(
                            packedA, mrOuter * k * Mr, b, bOffset, ldb,
                            c, cOffset + (mcStart + mrOuter * Mr) * ldc, ldc, k);
                    return packedA;
                },
                localFinally: localPacked => pool.Return(localPacked));
        }
        else
        {
            double[] packedA = pool.Rent(Mc * k);
            try
            {
                for (int mb = 0; mb < mcBlocks; mb++)
                {
                    int mcStart = mb * Mc;
                    PackAPanel_TransA_N16_Mr8(a, kernelOffset, lda, packedA, mcStart, Mc, k);
                    int mrBlocks = Mc / Mr;
                    for (int mrOuter = 0; mrOuter < mrBlocks; mrOuter++)
                        Microkernel_TransA_N16_Mr8(
                            packedA, mrOuter * k * Mr, b, bOffset, ldb,
                            c, cOffset + (mcStart + mrOuter * Mr) * ldc, ldc, k);
                }
            }
            finally { pool.Return(packedA); }
        }

        // Tail: pack the largest Mr=8 multiple, then a 1-row scalar remainder.
        double[] packedTailBuf = pool.Rent(Mc * k);
        try
        {
            if (mcTail >= Mr)
            {
                int mcStart = mcBlocks * Mc;
                int packedRows = (mcTail / Mr) * Mr;
                PackAPanel_TransA_N16_Mr8(a, kernelOffset, lda, packedTailBuf, mcStart, packedRows, k);
                int mrBlocks = packedRows / Mr;
                for (int mrOuter = 0; mrOuter < mrBlocks; mrOuter++)
                    Microkernel_TransA_N16_Mr8(
                        packedTailBuf, mrOuter * k * Mr, b, bOffset, ldb,
                        c, cOffset + (mcStart + mrOuter * Mr) * ldc, ldc, k);
                mcTail -= packedRows;
            }
            for (int r = 0; r < mcTail; r++)
            {
                int mGlobal = m - mcTail + r;
                for (int j = 0; j < 16; j++)
                {
                    double acc = 0;
                    for (int kk = 0; kk < k; kk++)
                        acc += a[kernelOffset + (long)kk * lda + mGlobal] * b[bOffset + kk * ldb + j];
                    c[cOffset + mGlobal * ldc + j] = acc;
                }
            }
        }
        finally { pool.Return(packedTailBuf); }
    }

    /// <summary>Mr=8 BLIS pack: packed[mrOuter, kk, ri] for ri in 0..7.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe void PackAPanel_TransA_N16_Mr8(
        double[] a, int aOffset, int lda,
        double[] packed, int mcStart, int mc, int k)
    {
        const int Mr = DgemmFatAMr8;
        int mrBlocks = mc / Mr;
        fixed (double* pA = &a[aOffset])
        fixed (double* pPacked = packed)
        {
            for (int mrOuter = 0; mrOuter < mrBlocks; mrOuter++)
            {
                int mGlobal = mcStart + mrOuter * Mr;
                double* outBase = pPacked + (long)mrOuter * k * Mr;
                for (int kk = 0; kk < k; kk++)
                {
                    double* aRow = pA + (long)kk * lda + mGlobal;
                    double* outRow = outBase + (long)kk * Mr;
                    for (int ri = 0; ri < Mr; ri++)
                        outRow[ri] = aRow[ri];
                }
            }
        }
    }

    /// <summary>
    /// Mr=8, Nr=16 FP64 AVX-512 microkernel over the BLIS-packed panel. Per kk:
    /// 2 B-loads (Vector512, the full N=16 row), 8 a-broadcasts, 16 fused
    /// multiply-adds into 16 ZMM accumulators (8 rows × 2 halves). Mirrors the
    /// proven Avx512Fp64_8x16 FMA pattern in the BlasManaged subsystem.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe void Microkernel_TransA_N16_Mr8(
        double[] packedA, int packedOffset,
        double[] b, int bOffset, int ldb,
        double[] c, int cOffset, int ldc,
        int k)
    {
        const int Mr = DgemmFatAMr8;
        var cLo0 = Vector512<double>.Zero; var cHi0 = Vector512<double>.Zero;
        var cLo1 = Vector512<double>.Zero; var cHi1 = Vector512<double>.Zero;
        var cLo2 = Vector512<double>.Zero; var cHi2 = Vector512<double>.Zero;
        var cLo3 = Vector512<double>.Zero; var cHi3 = Vector512<double>.Zero;
        var cLo4 = Vector512<double>.Zero; var cHi4 = Vector512<double>.Zero;
        var cLo5 = Vector512<double>.Zero; var cHi5 = Vector512<double>.Zero;
        var cLo6 = Vector512<double>.Zero; var cHi6 = Vector512<double>.Zero;
        var cLo7 = Vector512<double>.Zero; var cHi7 = Vector512<double>.Zero;

        fixed (double* pA = &packedA[packedOffset])
        fixed (double* pB = &b[bOffset])
        fixed (double* pC = &c[cOffset])
        {
            for (int kk = 0; kk < k; kk++)
            {
                double* aPtr = pA + (long)kk * Mr;
                double* bRow = pB + (long)kk * ldb;
                var vbLo = Avx512F.LoadVector512(bRow + 0);
                var vbHi = Avx512F.LoadVector512(bRow + 8);
                var a0 = Vector512.Create(aPtr[0]); cLo0 = Avx512F.FusedMultiplyAdd(a0, vbLo, cLo0); cHi0 = Avx512F.FusedMultiplyAdd(a0, vbHi, cHi0);
                var a1 = Vector512.Create(aPtr[1]); cLo1 = Avx512F.FusedMultiplyAdd(a1, vbLo, cLo1); cHi1 = Avx512F.FusedMultiplyAdd(a1, vbHi, cHi1);
                var a2 = Vector512.Create(aPtr[2]); cLo2 = Avx512F.FusedMultiplyAdd(a2, vbLo, cLo2); cHi2 = Avx512F.FusedMultiplyAdd(a2, vbHi, cHi2);
                var a3 = Vector512.Create(aPtr[3]); cLo3 = Avx512F.FusedMultiplyAdd(a3, vbLo, cLo3); cHi3 = Avx512F.FusedMultiplyAdd(a3, vbHi, cHi3);
                var a4 = Vector512.Create(aPtr[4]); cLo4 = Avx512F.FusedMultiplyAdd(a4, vbLo, cLo4); cHi4 = Avx512F.FusedMultiplyAdd(a4, vbHi, cHi4);
                var a5 = Vector512.Create(aPtr[5]); cLo5 = Avx512F.FusedMultiplyAdd(a5, vbLo, cLo5); cHi5 = Avx512F.FusedMultiplyAdd(a5, vbHi, cHi5);
                var a6 = Vector512.Create(aPtr[6]); cLo6 = Avx512F.FusedMultiplyAdd(a6, vbLo, cLo6); cHi6 = Avx512F.FusedMultiplyAdd(a6, vbHi, cHi6);
                var a7 = Vector512.Create(aPtr[7]); cLo7 = Avx512F.FusedMultiplyAdd(a7, vbLo, cLo7); cHi7 = Avx512F.FusedMultiplyAdd(a7, vbHi, cHi7);
            }

            Avx512F.Store(pC + 0 * ldc + 0, cLo0); Avx512F.Store(pC + 0 * ldc + 8, cHi0);
            Avx512F.Store(pC + 1 * ldc + 0, cLo1); Avx512F.Store(pC + 1 * ldc + 8, cHi1);
            Avx512F.Store(pC + 2 * ldc + 0, cLo2); Avx512F.Store(pC + 2 * ldc + 8, cHi2);
            Avx512F.Store(pC + 3 * ldc + 0, cLo3); Avx512F.Store(pC + 3 * ldc + 8, cHi3);
            Avx512F.Store(pC + 4 * ldc + 0, cLo4); Avx512F.Store(pC + 4 * ldc + 8, cHi4);
            Avx512F.Store(pC + 5 * ldc + 0, cLo5); Avx512F.Store(pC + 5 * ldc + 8, cHi5);
            Avx512F.Store(pC + 6 * ldc + 0, cLo6); Avx512F.Store(pC + 6 * ldc + 8, cHi6);
            Avx512F.Store(pC + 7 * ldc + 0, cLo7); Avx512F.Store(pC + 7 * ldc + 8, cHi7);
        }
    }
#endif

    #endregion
}
