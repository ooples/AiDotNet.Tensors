using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Gpu;

/// <summary>
/// #1650/#638 (#671) — validates the DIRECT FP16 convolution (<see cref="CudaBackend.Conv2dDirectFp16Hw"/>) on real
/// hardware. This kernel handles the tiny-spatial deep convs the im2col + Tensor-Core GEMM path deliberately skips
/// (small N, where im2col + launch overhead exceeds the GEMM win). It rounds the FP32 input to FP16 per access,
/// multiplies by FP16 weights, and accumulates in FP32 — exactly the GemmFp16In32fOut numerics. This proves the
/// result matches an FP16-rounded CPU reference within FP16 tolerance, so the path is correct (CUDA-first; parity
/// for the other backends is a tracked follow-up). Skips on a box without a CUDA toolkit (the kernel needs
/// cuda_fp16.h, gated by <see cref="CudaBackend.Fp16DirectConvAvailable"/>).
/// </summary>
public class Conv2dDirectFp16HwTests
{
    [SkippableFact]
    public void Conv2dDirectFp16_matches_fp32_reference_within_fp16_tolerance()
    {
        Skip.IfNot(CudaNativeBindings.IsAvailable, "CUDA driver not available");
        var eng = AiDotNetEngine.Current;
        Skip.IfNot(eng is DirectGpuTensorEngine, "active engine is not the CUDA backend");
        // Warm up so the engine compiles its kernel modules (incl. the FP16 module that owns conv2d_direct_fp16hw).
        var warm = new Tensor<float>(new float[] { 1, 2, 3, 4 }, new[] { 2, 2 });
        _ = eng.TensorMatMul(warm, warm);
        var b = ((DirectGpuTensorEngine)eng).TestBackend as CudaBackend;
        Skip.IfNot(b is not null && b.IsAvailable, "CudaBackend not available");
        Skip.IfNot(b!.Fp16DirectConvAvailable, "conv2d_direct_fp16hw not compiled (needs the CUDA toolkit / cuda_fp16.h)");

        // A tiny-spatial deep conv: N = b*outH*outW = 64 (the documented tiny boundary), K = inC*kh*kw = 576,
        // M = outC = 32 — exactly the shape the im2col+GEMM path skips and the direct kernel is wired for.
        const int batch = 1, inC = 64, H = 8, W = 8, outC = 32, kh = 3, kw = 3;
        const int strideH = 1, strideW = 1, padH = 1, padW = 1, dilH = 1, dilW = 1;
        int outH = (H + 2 * padH - ((kh - 1) * dilH + 1)) / strideH + 1; // 8
        int outW = (W + 2 * padW - ((kw - 1) * dilW + 1)) / strideW + 1; // 8

        var rng = new Random(1650);
        var input = new float[batch * inC * H * W];
        var weights = new float[outC * inC * kh * kw];
        for (int i = 0; i < input.Length; i++) input[i] = (float)(rng.NextDouble() * 2 - 1);   // [-1,1] — safe FP16 range
        for (int i = 0; i < weights.Length; i++) weights[i] = (float)(rng.NextDouble() * 2 - 1);

        // FP16-rounded CPU reference (input + weights rounded to half, FP32 multiply-accumulate) — mirrors the
        // kernel exactly, so the only residual difference is FP32 accumulation order.
        static float ToHalfF(float x) => (float)(Half)x;
        var cpu = new float[batch * outC * outH * outW];
        for (int bb = 0; bb < batch; bb++)
            for (int oc = 0; oc < outC; oc++)
                for (int oh = 0; oh < outH; oh++)
                    for (int ow = 0; ow < outW; ow++)
                    {
                        double acc = 0;
                        for (int ic = 0; ic < inC; ic++)
                            for (int r = 0; r < kh; r++)
                            {
                                int ih = oh * strideH - padH + r * dilH;
                                if (ih < 0 || ih >= H) continue;
                                for (int s = 0; s < kw; s++)
                                {
                                    int iw = ow * strideW - padW + s * dilW;
                                    if (iw < 0 || iw >= W) continue;
                                    float inV = ToHalfF(input[((bb * inC + ic) * H + ih) * W + iw]);
                                    float wV = ToHalfF(weights[((oc * inC + ic) * kh + r) * kw + s]);
                                    acc += (double)inV * wV;
                                }
                            }
                        cpu[((bb * outC + oc) * outH + oh) * outW + ow] = (float)acc;
                    }

        using var dInput = b.AllocateBuffer(input);
        using var dWeightsF32 = b.AllocateBuffer(weights);
        using var hWeights = b.AllocateByteBuffer(weights.Length * 2); // FP16 weights
        b.ConvertToFp16Native(dWeightsF32, hWeights, weights.Length);
        Assert.Equal((long)weights.Length * 2, hWeights.SizeInBytes);  // genuinely half

        using var dOut = b.AllocateBuffer(cpu.Length); // FP32 accumulate output
        try
        {
            b.Conv2dDirectFp16Hw(dInput, hWeights, dOut,
                batch, inC, H, W, outC, outH, outW, kh, kw, strideH, strideW, padH, padW, dilH, dilW);
        }
        catch (NotSupportedException ex)
        {
            Skip.If(true, $"direct FP16 conv not supported on this device: {ex.Message}");
            return;
        }
        var got = b.DownloadBuffer(dOut);

        double num = 0, den = 0;
        for (int i = 0; i < cpu.Length; i++)
        {
            Assert.False(float.IsNaN(got[i]) || float.IsInfinity(got[i]), $"non-finite output at {i}: {got[i]}");
            double d = got[i] - cpu[i];
            num += d * d;
            den += (double)cpu[i] * cpu[i];
        }
        double relL2 = den > 0 ? Math.Sqrt(num / den) : Math.Sqrt(num);
        Assert.True(relL2 < 0.02, $"direct FP16 conv relL2 {relL2:F5} exceeds FP16 tolerance vs the FP16-rounded CPU reference");
    }
}
