// Copyright (c) AiDotNet. All rights reserved.
namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>
/// <see cref="IFp16ActivationBackend"/> for backends without FP16-native kernels: each operation widens its FP16
/// operands into FP32 device scratch, runs the backend's own FP32 kernel, and narrows the FP16 results back (IEEE
/// half, round to nearest even), so the outputs match the native kernels' FP32-compute contract. The capability flags
/// stay false on these backends: the engine keeps activations in FP32 there, and these are the correct, slower forms.
/// </summary>
internal static class PortableFp16Activation
{
    private static IGpuBuffer Widen(IDirectGpuBackend backend, IGpuBuffer half, int count)
    {
        var wide = backend.AllocateBuffer(Math.Max(1, count));
        try { if (count > 0) backend.ConvertToFp32(half, wide, count); }
        catch { wide.Dispose(); throw; }
        return wide;
    }

    private static void Narrow(IDirectGpuBackend backend, IGpuBuffer wide, IGpuBuffer half, int count)
    {
        if (count > 0) backend.ConvertToFp16(wide, half, count);
    }

    public static void Add(IDirectGpuBackend backend, IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, int size)
    {
        using var wa = Widen(backend, a, size);
        using var wb = Widen(backend, b, size);
        using var sum = backend.AllocateBuffer(Math.Max(1, size));
        backend.Add(wa, wb, sum, size);
        Narrow(backend, sum, output, size);
    }

    // io[b, c, s] += bias[b, c]: the rows of io are the (b, c) pairs, so a one-batch per-channel bias over them is it.
    public static void BroadcastAddChannel(IDirectGpuBackend backend, IGpuBuffer ioHalf, IGpuBuffer biasHalf, int batch,
        int channels, int spatial)
    {
        int rows = checked(batch * channels), count = checked(rows * spatial);
        using var io = Widen(backend, ioHalf, count);
        using var bias = Widen(backend, biasHalf, rows);
        backend.Conv2DBiasAdd(io, bias, 1, rows, spatial);
        Narrow(backend, io, ioHalf, count);
    }

    public static void Unary(IDirectGpuBackend backend, IGpuBuffer input, IGpuBuffer output, int size, bool gelu)
    {
        using var x = Widen(backend, input, size);
        using var y = backend.AllocateBuffer(Math.Max(1, size));
        if (gelu) backend.Gelu(x, y, size);
        else backend.Relu(x, y, size);
        Narrow(backend, y, output, size);
    }

    public static void UnaryBackward(IDirectGpuBackend backend, IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradInput,
        int size, bool gelu)
    {
        using var g = Widen(backend, gradOutput, size);
        using var x = Widen(backend, input, size);
        using var dx = backend.AllocateBuffer(Math.Max(1, size));
        if (gelu) backend.GeluBackward(g, x, dx, size);
        else backend.ReluBackward(g, x, dx, size);
        Narrow(backend, dx, gradInput, size);
    }

    public static void SoftmaxBackward(IDirectGpuBackend backend, IGpuBuffer gradOutput, IGpuBuffer output, IGpuBuffer gradInput,
        int batchSize, int features)
    {
        int count = checked(batchSize * features);
        using var g = Widen(backend, gradOutput, count);
        using var y = Widen(backend, output, count);
        using var dx = backend.AllocateBuffer(Math.Max(1, count));
        backend.SoftmaxBackward(g, y, dx, batchSize, features);
        Narrow(backend, dx, gradInput, count);
    }

    // saveInvVar already holds 1 / sqrt(var + eps), which is what the FP32 kernel reads, so its epsilon is unused.
    public static void LayerNormBackward(IDirectGpuBackend backend, IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer? gamma,
        IGpuBuffer saveMean, IGpuBuffer saveInvVar, IGpuBuffer? gradInput, IGpuBuffer? gradGamma, IGpuBuffer? gradBeta,
        int rows, int cols)
    {
        int count = checked(rows * cols);
        using var g = Widen(backend, gradOutput, count);
        using var x = Widen(backend, input, count);
        using var ga = gamma is null ? backend.AllocateBuffer(Math.Max(1, cols)) : Widen(backend, gamma, cols);
        if (gamma is null) backend.Fill(ga, 1f, cols);
        using var dx = backend.AllocateBuffer(Math.Max(1, count));
        using var dg = backend.AllocateBuffer(Math.Max(1, cols));
        using var db = backend.AllocateBuffer(Math.Max(1, cols));
        backend.LayerNormBackward(g, x, ga, saveMean, saveInvVar, dx, dg, db, rows, cols, 0f);
        if (gradInput is not null) Narrow(backend, dx, gradInput, count);
        if (gradGamma is not null) Narrow(backend, dg, gradGamma, cols);
        if (gradBeta is not null) Narrow(backend, db, gradBeta, cols);
    }

    public static void GroupNormSwish(IDirectGpuBackend backend, IGpuBuffer inHalf, IGpuBuffer gammaFp32, IGpuBuffer betaFp32,
        IGpuBuffer outHalf, int batch, int numGroups, int channels, int spatial, float eps)
    {
        int count = checked(batch * channels * spatial), stats = checked(batch * numGroups);
        using var x = Widen(backend, inHalf, count);
        using var y = backend.AllocateBuffer(Math.Max(1, count));
        using var mean = backend.AllocateBuffer(Math.Max(1, stats));
        using var invVar = backend.AllocateBuffer(Math.Max(1, stats));
        backend.GroupNorm(x, y, gammaFp32, betaFp32, mean, invVar, batch, numGroups, channels, spatial, eps);
        backend.Swish(y, y, count);
        Narrow(backend, y, outHalf, count);
    }

    public static void Conv2dHalfWeights(IDirectGpuBackend backend, IGpuBuffer input, IGpuBuffer weightsHalf, IGpuBuffer output,
        int batch, int inChannels, int inHeight, int inWidth, int outChannels, int outHeight, int outWidth,
        int kernelH, int kernelW, int strideH, int strideW, int padH, int padW, int dilationH, int dilationW)
    {
        using var w = Widen(backend, weightsHalf, checked(outChannels * inChannels * kernelH * kernelW));
        backend.Conv2D(input, w, output, batch, inChannels, inHeight, inWidth, outChannels, outHeight, outWidth,
            kernelH, kernelW, strideH, strideW, padH, padW, dilationH, dilationW);
    }

    public static void GemmHalfOut(IDirectGpuBackend backend, IGpuBuffer aHalf, IGpuBuffer bHalf, IGpuBuffer cHalf,
        int m, int n, int k, float alpha, float beta)
    {
        int cCount = checked(m * n);
        using var a = Widen(backend, aHalf, checked(m * k));
        using var b = Widen(backend, bHalf, checked(k * n));
        using var c = beta != 0f ? Widen(backend, cHalf, cCount) : backend.AllocateBuffer(Math.Max(1, cCount));
        backend.Gemm(a, b, c, m, n, k, alpha, beta);
        Narrow(backend, c, cHalf, cCount);
    }

    public static void CopyBytes(IDirectGpuBackend backend, IGpuBuffer source, IGpuBuffer destination, long byteCount)
    {
        if (source is null) throw new ArgumentNullException(nameof(source));
        if (destination is null) throw new ArgumentNullException(nameof(destination));
        if (byteCount <= 0) return;
        if (byteCount % sizeof(float) == 0)
        {
            backend.Copy(source, destination, checked((int)(byteCount / sizeof(float))));
            return;
        }
        // A range ending mid-word (an odd count of halves) goes through the host so no byte past it is written.
        backend.UploadByteBuffer(destination, backend.DownloadByteBuffer(source, checked((int)byteCount)));
    }

    // im2col of an FP16 NCHW input into FP16 [K, N]: K = (c, kh, kw) rows, N = (b, oh, ow) columns, 0 outside the image,
    // as CUDA's im2col_kn_fp16_from_fp16. The values are copied, not computed, so a host staging reproduces it exactly.
    public static void UnfoldKN(IDirectGpuBackend backend, IGpuBuffer inputHalf, IGpuBuffer outputHalf, int batch, int channels,
        int height, int width, int kernelH, int kernelW, int strideH, int strideW, int padH, int padW, int dilationH, int dilationW)
    {
        int outH = (height + 2 * padH - dilationH * (kernelH - 1) - 1) / strideH + 1;
        int outW = (width + 2 * padW - dilationW * (kernelW - 1) - 1) / strideW + 1;
        int n = checked(batch * outH * outW), kRows = checked(channels * kernelH * kernelW);
        byte[] input = backend.DownloadByteBuffer(inputHalf, checked(batch * channels * height * width * 2));
        var output = new byte[checked((long)n * kRows * 2)];
        for (int k = 0; k < kRows; k++)
        {
            int kw = k % kernelW, kh = (k / kernelW) % kernelH, c = k / (kernelW * kernelH);
            for (int col = 0; col < n; col++)
            {
                int b = col / (outH * outW), rem = col % (outH * outW), oh = rem / outW, ow = rem % outW;
                int ih = oh * strideH - padH + kh * dilationH, iw = ow * strideW - padW + kw * dilationW;
                if (ih < 0 || ih >= height || iw < 0 || iw >= width) continue;
                long src = ((((long)b * channels + c) * height + ih) * width + iw) * 2, dst = ((long)k * n + col) * 2;
                output[dst] = input[src];
                output[dst + 1] = input[src + 1];
            }
        }
        backend.UploadByteBuffer(outputHalf, output);
    }
}