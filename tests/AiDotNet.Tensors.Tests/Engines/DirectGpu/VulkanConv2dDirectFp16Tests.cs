#if NET6_0_OR_GREATER
using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.Vulkan;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// #682 on Vulkan: the direct FP16 convolution (the port of CUDA's <c>conv2d_direct_fp16hw</c>, #671). The FP32
/// input is rounded to FP16 per access, the weights are packed halves, and the products accumulate in FP32. The
/// reference does the same on the CPU, so what is left is FP32 accumulation rounding.
/// Mirrors <c>Conv2dDirectFp16HwTests</c> (CUDA).
/// </summary>
[Collection("VulkanGlobalState")]
public class VulkanConv2dDirectFp16Tests
{
    // The backend is a process-wide singleton other test classes share, so nothing here disposes it.
    private static VulkanBackend BackendOrSkip()
    {
        VulkanBackend? backend = null;
        try
        {
            backend = VulkanBackend.Instance;
            if (!backend.Initialize() || !backend.IsGlslCompilerAvailable) backend = null;
        }
        catch (Exception ex) when (ex is DllNotFoundException or PlatformNotSupportedException or TypeInitializationException)
        {
            backend = null;
        }
        Skip.If(backend is null, "No Vulkan device with the runtime GLSL compiler.");
        Skip.IfNot(backend!.Fp16DirectConvAvailable, "The direct FP16 convolution shader did not build on this device.");
        return backend;
    }

    private static float AsHalf(float x) => (float)(Half)x;

    private static int OutExtent(int size, int kernel, int stride, int pad, int dilation)
        => (size + 2 * pad - ((kernel - 1) * dilation + 1)) / stride + 1;

    // batch, inC, H, W, outC, kH, kW, strideH, strideW, padH, padW, dilationH, dilationW
    public static IEnumerable<object[]> Geometries()
    {
        yield return new object[] { 1, 64, 8, 8, 32, 3, 3, 1, 1, 1, 1, 1, 1 };   // the tiny deep conv the engine routes here
        yield return new object[] { 2, 3, 9, 7, 5, 3, 3, 2, 2, 1, 1, 1, 1 };     // stride 2, odd weight count (135): a half-filled last word
        yield return new object[] { 1, 4, 10, 10, 3, 3, 3, 1, 1, 2, 2, 2, 2 };   // dilation 2
        yield return new object[] { 3, 8, 5, 6, 7, 1, 1, 1, 1, 0, 0, 1, 1 };     // 1x1, no padding
        yield return new object[] { 1, 2, 6, 11, 4, 3, 5, 1, 2, 0, 2, 1, 1 };    // rectangular kernel, stride and padding differ per axis
        yield return new object[] { 1, 1, 4, 4, 1, 4, 4, 1, 1, 0, 0, 1, 1 };     // a single output element
    }

    [SkippableTheory]
    [MemberData(nameof(Geometries))]
    public void MatchesTheFp16RoundedCpuReference(int batch, int inC, int h, int w, int outC, int kH, int kW,
        int strideH, int strideW, int padH, int padW, int dilH, int dilW)
    {
        var backend = BackendOrSkip();
        int outH = OutExtent(h, kH, strideH, padH, dilH), outW = OutExtent(w, kW, strideW, padW, dilW);
        var rng = new Random(batch * 131 + inC * 17 + kH * 5 + strideH);
        var input = new float[batch * inC * h * w];
        var weights = new float[outC * inC * kH * kW];
        for (int i = 0; i < input.Length; i++) input[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int i = 0; i < weights.Length; i++) weights[i] = (float)(rng.NextDouble() * 2 - 1);

        var expected = new float[batch * outC * outH * outW];
        for (int b = 0; b < batch; b++)
            for (int oc = 0; oc < outC; oc++)
                for (int oh = 0; oh < outH; oh++)
                    for (int ow = 0; ow < outW; ow++)
                    {
                        double acc = 0;
                        for (int ic = 0; ic < inC; ic++)
                            for (int r = 0; r < kH; r++)
                            {
                                int ih = oh * strideH - padH + r * dilH;
                                if (ih < 0 || ih >= h) continue;
                                for (int s = 0; s < kW; s++)
                                {
                                    int iw = ow * strideW - padW + s * dilW;
                                    if (iw < 0 || iw >= w) continue;
                                    acc += (double)AsHalf(input[((b * inC + ic) * h + ih) * w + iw])
                                        * AsHalf(weights[((oc * inC + ic) * kH + r) * kW + s]);
                                }
                            }
                        expected[((b * outC + oc) * outH + oh) * outW + ow] = (float)acc;
                    }

        using var dInput = backend.AllocateBuffer(input);
        using var dWeights = backend.AllocateBuffer(weights);
        using var dWeightsHalf = backend.AllocateBuffer((weights.Length + 1) / 2); // packed: two halves per word
        backend.ConvertToFp16(dWeights, dWeightsHalf, weights.Length);
        using var dOut = backend.AllocateBuffer(expected.Length);
        backend.Fill(dOut, float.NaN, expected.Length); // an element the kernel does not write stays NaN and fails

        backend.Conv2dDirectFp16Hw(dInput, dWeightsHalf, dOut, batch, inC, h, w, outC, outH, outW,
            kH, kW, strideH, strideW, padH, padW, dilH, dilW);
        float[] got = backend.DownloadBuffer(dOut);

        double num = 0, den = 0;
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.False(float.IsNaN(got[i]) || float.IsInfinity(got[i]), $"non-finite output at {i}: {got[i]}");
            double d = got[i] - expected[i];
            num += d * d;
            den += (double)expected[i] * expected[i];
        }
        double relL2 = den > 0 ? Math.Sqrt(num / den) : Math.Sqrt(num);
        // The reference rounds the same operands the kernel does, so only FP32 accumulation is left. CUDA's test
        // allows 0.02 against the same reference; this is held far tighter.
        Assert.True(relL2 < 1e-5, $"direct FP16 conv relL2 {relL2:E3} against the FP16-rounded CPU reference");
    }

    [SkippableFact]
    public void TheInputIsRoundedToFp16_ToNearestEven_AcrossTheWholeRange()
    {
        // A 1x1 convolution with one unit weight returns each input after the kernel's FP16 rounding, so the output
        // must be bit-identical to (float)(Half)x: ties, the subnormal range, the largest finite half, overflow.
        var backend = BackendOrSkip();
        float[] input =
        {
            0f, -0f, 1f, -1f, 1.0001f, 1.0004883f /* tie: rounds to even, 1.0 */, 1.0014648f /* tie: rounds up to even */,
            1.0006f, -1.0006f, 0.1f, 0.3333333f, 123.456f, -98765.4f /* overflows */, 65504f /* largest finite half */,
            65519.99f /* rounds down to 65504 */, 65520f /* rounds to infinity */, 1e-5f, 6.1035156e-5f /* 2^-14, smallest normal */,
            6.0e-5f /* subnormal half */, 5.9604645e-8f /* 2^-24, smallest subnormal */, 2.9802322e-8f /* tie to zero */,
            4.5e-8f /* rounds up to 2^-24 */, 1e-10f /* underflows to zero */, -3.0e-6f, 3.1415927f, 2049f, 2047.5f,
            float.PositiveInfinity, float.NegativeInfinity,
        };
        using var dInput = backend.AllocateBuffer(input);
        using var dWeight = backend.AllocateBuffer(new[] { 1f });
        using var dWeightHalf = backend.AllocateBuffer(1);
        backend.ConvertToFp16(dWeight, dWeightHalf, 1);
        using var dOut = backend.AllocateBuffer(input.Length);
        backend.Fill(dOut, float.NaN, input.Length);

        backend.Conv2dDirectFp16Hw(dInput, dWeightHalf, dOut, 1, 1, 1, input.Length, 1, 1, input.Length, 1, 1, 1, 1, 0, 0, 1, 1);
        float[] got = backend.DownloadBuffer(dOut);
        for (int i = 0; i < input.Length; i++)
        {
            float expected = AsHalf(input[i]);
            Assert.True(BitConverter.SingleToInt32Bits(expected) == BitConverter.SingleToInt32Bits(got[i])
                    // 0 * 1 keeps the sign of zero on the CPU; a shader may return +0 for -0, which is the same value.
                    || (expected == 0f && got[i] == 0f),
                $"input {input[i]:R}: expected {expected:R}, got {got[i]:R}");
        }
    }

    [SkippableFact]
    public void TheInterfaceRoute_RunsTheDirectKernel_NotTheWidenedFp32Convolution()
    {
        // The two differ in one observable way: the direct kernel rounds the INPUT to FP16; widening the weights and
        // running the FP32 convolution does not. 1.0001 is not a half (the spacing at 1 is about 0.00098) and
        // rounds to exactly 1, so with unit weights every output is exactly the tap count through the direct
        // kernel, and about 1.0001 times it through the FP32 convolution.
        var backend = BackendOrSkip();
        const int inC = 16, size = 6, outC = 2, k = 3;
        int outSize = OutExtent(size, k, 1, 0, 1);
        var input = new float[inC * size * size];
        var weights = new float[outC * inC * k * k];
        for (int i = 0; i < input.Length; i++) input[i] = 1.0001f;
        for (int i = 0; i < weights.Length; i++) weights[i] = 1f;
        Assert.Equal(1f, AsHalf(1.0001f));

        using var dInput = backend.AllocateBuffer(input);
        using var dWeights = backend.AllocateBuffer(weights);
        using var dWeightsHalf = backend.AllocateBuffer((weights.Length + 1) / 2);
        backend.ConvertToFp16(dWeights, dWeightsHalf, weights.Length);
        using var dOut = backend.AllocateBuffer(outC * outSize * outSize);

        var route = (IFp16ActivationBackend)backend;
        Assert.True(route.Fp16DirectConvAvailable, "the engine only takes this route when the capability reports true");
        route.Conv2dDirectFp16Hw(dInput, dWeightsHalf, dOut, 1, inC, size, size, outC, outSize, outSize, k, k, 1, 1, 0, 0, 1, 1);
        float taps = inC * k * k;
        Assert.All(backend.DownloadBuffer(dOut), v => Assert.Equal(taps, v));

        // The control: the FP32 convolution on the same operands does NOT give the tap count, so the assertion
        // above cannot pass by accident.
        backend.Conv2D(dInput, dWeights, dOut, 1, inC, size, size, outC, outSize, outSize, k, k, 1, 1, 0, 0, 1, 1);
        Assert.All(backend.DownloadBuffer(dOut), v => Assert.NotEqual(taps, v));
    }

    [SkippableFact]
    public void RejectsAnOutputExtentThatDoesNotFollowFromTheGeometry_AndShortBuffers()
    {
        var backend = BackendOrSkip();
        using var input = backend.AllocateBuffer(1 * 2 * 5 * 5);
        using var weightsHalf = backend.AllocateBuffer((3 * 2 * 3 * 3 + 1) / 2);
        using var output = backend.AllocateBuffer(1 * 3 * 3 * 3);

        // 5x5, 3x3 kernel, no padding gives 3x3.
        Assert.Throws<ArgumentException>(() =>
            backend.Conv2dDirectFp16Hw(input, weightsHalf, output, 1, 2, 5, 5, 3, 4, 4, 3, 3, 1, 1, 0, 0, 1, 1));
        using var shortWeights = backend.AllocateBuffer(3);
        Assert.Throws<ArgumentException>(() =>
            backend.Conv2dDirectFp16Hw(input, shortWeights, output, 1, 2, 5, 5, 3, 3, 3, 3, 3, 1, 1, 0, 0, 1, 1));
        using var shortOutput = backend.AllocateBuffer(5);
        Assert.Throws<ArgumentException>(() =>
            backend.Conv2dDirectFp16Hw(input, weightsHalf, shortOutput, 1, 2, 5, 5, 3, 3, 3, 3, 3, 1, 1, 0, 0, 1, 1));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            backend.Conv2dDirectFp16Hw(input, weightsHalf, output, 1, 2, 5, 5, 3, 3, 3, 3, 3, 0, 1, 0, 0, 1, 1));
    }
}
#endif
