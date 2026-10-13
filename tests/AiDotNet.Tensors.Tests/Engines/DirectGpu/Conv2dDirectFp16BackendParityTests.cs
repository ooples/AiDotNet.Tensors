#if NET6_0_OR_GREATER
using System;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.HIP;
using AiDotNet.Tensors.Engines.DirectGpu.Metal;
using AiDotNet.Tensors.Engines.DirectGpu.OpenCL;
using AiDotNet.Tensors.Engines.DirectGpu.Vulkan;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// #682: the direct FP16 convolution through <see cref="IFp16ActivationBackend"/> on the backends that run the
/// shared <c>dc_p_conv2d_direct_fp16hw</c> kernel (OpenCL, HIP, Metal, WebGPU), against a CPU reference that does
/// what the route documents: the FP32 input rounded to FP16 per access, FP16 weights, products accumulated in FP32.
/// Vulkan, which has its own GLSL kernel, runs the same checks so the two implementations are held to one contract.
/// </summary>
[Collection("DirectGpuSerial")]
public sealed class Conv2dDirectFp16BackendParityTests
{
    private static float AsHalf(float x) => (float)(Half)x;

    private static int OutExtent(int size, int kernel, int stride, int pad, int dilation)
        => (size + 2 * pad - ((kernel - 1) * dilation + 1)) / stride + 1;

    // batch, inC, H, W, outC, kH, kW, strideH, strideW, padH, padW, dilationH, dilationW
    private static readonly int[][] Geometries =
    {
        new[] { 1, 64, 8, 8, 32, 3, 3, 1, 1, 1, 1, 1, 1 },   // the tiny deep conv the engine routes here
        new[] { 2, 3, 9, 7, 5, 3, 3, 2, 2, 1, 1, 1, 1 },     // stride 2, odd weight count (135): a half-filled last word
        new[] { 1, 4, 10, 10, 3, 3, 3, 1, 1, 2, 2, 2, 2 },   // dilation 2
        new[] { 3, 8, 5, 6, 7, 1, 1, 1, 1, 0, 0, 1, 1 },     // 1x1, no padding
        new[] { 1, 2, 6, 11, 4, 3, 5, 1, 2, 0, 2, 1, 1 },    // rectangular kernel, stride and padding differ per axis
        new[] { 1, 1, 4, 4, 1, 4, 4, 1, 1, 0, 0, 1, 1 },     // a single output element
        new[] { 5, 2, 3, 40, 3, 3, 3, 1, 1, 1, 1, 1, 1 },    // 1,800 outputs: several work-groups
    };

    private static IGpuBuffer HalfWords(IDirectGpuBackend backend, float[] weights)
    {
        using var fp32 = backend.AllocateBuffer(weights);
        var half = backend.AllocateBuffer((weights.Length + 1) / 2); // packed: two halves per word
        backend.ConvertToFp16(fp32, half, weights.Length);
        return half;
    }

    private delegate void DirectConv(IGpuBuffer input, IGpuBuffer weightsHalf, IGpuBuffer output, int batch, int inChannels,
        int inHeight, int inWidth, int outChannels, int outHeight, int outWidth, int kernelH, int kernelW, int strideH, int strideW,
        int padH, int padW, int dilationH, int dilationW);

    private static void Check(IDirectGpuBackend backend)
    {
        var route = Assert.IsAssignableFrom<IFp16ActivationBackend>(backend);
        Assert.True(route.Fp16DirectConvAvailable, "the engine only takes this route when the capability reports true");
        Check(backend, route.Conv2dDirectFp16Hw);
    }

    private static void Check(IDirectGpuBackend backend, DirectConv conv)
    {
        foreach (int[] g in Geometries) CheckGeometry(backend, conv, g);
        CheckRounding(backend, conv);
        CheckItIsNotTheWidenedFp32Convolution(backend, conv);
        CheckAnOddCountOfHalvesInAByteBuffer(backend, conv);
    }

    private static void CheckGeometry(IDirectGpuBackend backend, DirectConv conv, int[] g)
    {
        int batch = g[0], inC = g[1], h = g[2], w = g[3], outC = g[4], kH = g[5], kW = g[6];
        int strideH = g[7], strideW = g[8], padH = g[9], padW = g[10], dilH = g[11], dilW = g[12];
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
        using var dWeightsHalf = HalfWords(backend, weights);
        using var dOut = backend.AllocateBuffer(expected.Length);
        backend.Fill(dOut, float.NaN, expected.Length); // an element the kernel does not write stays NaN and fails

        conv(dInput, dWeightsHalf, dOut, batch, inC, h, w, outC, outH, outW,
            kH, kW, strideH, strideW, padH, padW, dilH, dilW);
        float[] got = backend.DownloadBuffer(dOut);

        string what = string.Join(",", g);
        double num = 0, den = 0;
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.False(float.IsNaN(got[i]) || float.IsInfinity(got[i]), $"[{what}] non-finite output at {i}: {got[i]}");
            double d = got[i] - expected[i];
            num += d * d;
            den += (double)expected[i] * expected[i];
        }
        double relL2 = den > 0 ? Math.Sqrt(num / den) : Math.Sqrt(num);
        // The reference rounds the same operands the kernel does, so only FP32 accumulation is left.
        Assert.True(relL2 < 1e-5, $"[{what}] direct FP16 conv relL2 {relL2:E3} against the FP16-rounded CPU reference");
    }

    // A 1x1 convolution with one unit weight returns each input after the kernel's FP16 rounding, so the output must
    // be bit-identical to (float)(Half)x: ties, the subnormal range, the largest finite half, overflow.
    private static void CheckRounding(IDirectGpuBackend backend, DirectConv conv)
    {
        float[] input =
        {
            0f, -0f, 1f, -1f, 1.0001f, 1.0004883f /* tie: rounds to even, 1.0 */, 1.0014648f /* tie: rounds up to even */,
            1.0006f, -1.0006f, 0.1f, 0.3333333f, 123.456f, -98765.4f /* overflows */, 65504f /* largest finite half */,
            65519.99f /* rounds down to 65504 */, 65520f /* rounds to infinity */, 1e-5f, 6.1035156e-5f /* 2^-14, smallest normal */,
            6.0e-5f /* subnormal half */, 6.1e-5f /* rounds up to the smallest normal */, 5.9604645e-8f /* 2^-24, smallest subnormal */,
            2.9802322e-8f /* tie to zero */, 8.940697e-8f /* 1.5 steps: tie, rounds to 2 */, 1.4901161e-7f /* 2.5 steps: tie, rounds to 2 */,
            4.5e-8f /* rounds up to 2^-24 */, 2.0e-8f /* below half a step: zero */, 1e-10f, 1e-40f /* an FP32 subnormal */,
            -3.0e-6f, 3.1415927f, 2049f, 2047.5f, float.PositiveInfinity, float.NegativeInfinity,
        };
        using var dInput = backend.AllocateBuffer(input);
        using var dWeightHalf = HalfWords(backend, new[] { 1f });
        using var dOut = backend.AllocateBuffer(input.Length);
        backend.Fill(dOut, float.NaN, input.Length);

        conv(dInput, dWeightHalf, dOut, 1, 1, 1, input.Length, 1, 1, input.Length, 1, 1, 1, 1, 0, 0, 1, 1);
        float[] got = backend.DownloadBuffer(dOut);
        for (int i = 0; i < input.Length; i++)
        {
            float expected = AsHalf(input[i]);
            Assert.True(BitConverter.SingleToInt32Bits(expected) == BitConverter.SingleToInt32Bits(got[i])
                    // 0 * 1 keeps the sign of zero on the CPU; a kernel may return +0 for -0, which is the same value.
                    || (expected == 0f && got[i] == 0f),
                $"input {input[i]:R}: expected {expected:R}, got {got[i]:R}");
        }
    }

    // The direct kernel rounds the INPUT to FP16; widening the weights and running the FP32 convolution (what these
    // backends did before #682) does not. 1.0001 is not a half and rounds to exactly 1, so with unit weights every
    // output is exactly the tap count through the direct kernel, and about 1.0001 times it through the FP32 one.
    private static void CheckItIsNotTheWidenedFp32Convolution(IDirectGpuBackend backend, DirectConv conv)
    {
        const int inC = 16, size = 6, outC = 2, k = 3;
        int outSize = OutExtent(size, k, 1, 0, 1);
        var input = new float[inC * size * size];
        var weights = new float[outC * inC * k * k];
        for (int i = 0; i < input.Length; i++) input[i] = 1.0001f;
        for (int i = 0; i < weights.Length; i++) weights[i] = 1f;
        Assert.Equal(1f, AsHalf(1.0001f));

        using var dInput = backend.AllocateBuffer(input);
        using var dWeights = backend.AllocateBuffer(weights);
        using var dWeightsHalf = HalfWords(backend, weights);
        using var dOut = backend.AllocateBuffer(outC * outSize * outSize);
        int outputs = outC * outSize * outSize;

        conv(dInput, dWeightsHalf, dOut, 1, inC, size, size, outC, outSize, outSize, k, k, 1, 1, 0, 0, 1, 1);
        float taps = inC * k * k;
        Assert.All(backend.DownloadBuffer(dOut).AsSpan(0, outputs).ToArray(), v => Assert.Equal(taps, v));

        // The control: the old route on the same operands does NOT give the tap count, so the assertion above cannot
        // pass by accident.
        PortableFp16Activation.Conv2dHalfWeights(backend, dInput, dWeightsHalf, dOut, 1, inC, size, size, outC, outSize, outSize,
            k, k, 1, 1, 0, 0, 1, 1);
        Assert.All(backend.DownloadBuffer(dOut).AsSpan(0, outputs).ToArray(), v => Assert.NotEqual(taps, v));
    }

    // 3 weights in a 6-byte buffer: the kernel reads 32-bit words, so the last half's word does not exist in full
    // and the route has to pad it.
    private static void CheckAnOddCountOfHalvesInAByteBuffer(IDirectGpuBackend backend, DirectConv conv)
    {
        float[] weights = { 0.5f, -2f, 3f };
        var bytes = new byte[weights.Length * 2];
        for (int i = 0; i < weights.Length; i++)
        {
            ushort bits = BitConverter.HalfToUInt16Bits((Half)weights[i]);
            bytes[2 * i] = (byte)bits;
            bytes[2 * i + 1] = (byte)(bits >> 8);
        }
        float[] input = { 1f, 2f, 4f, 8f, 16f };
        using var dInput = backend.AllocateBuffer(input);
        using var dWeightsHalf = backend.AllocateByteBuffer(bytes.Length);
        backend.UploadByteBuffer(dWeightsHalf, bytes);
        using var dOut = backend.AllocateBuffer(3);
        backend.Fill(dOut, float.NaN, 3);

        // [1, 1, 1, 5] with a 1x3 kernel: three outputs.
        conv(dInput, dWeightsHalf, dOut, 1, 1, 1, 5, 1, 1, 3, 1, 3, 1, 1, 0, 0, 1, 1);
        float[] got = backend.DownloadBuffer(dOut);
        for (int o = 0; o < 3; o++)
            Assert.Equal(input[o] * 0.5f + input[o + 1] * -2f + input[o + 2] * 3f, got[o]);
    }

    [SkippableFact]
    public void OpenCl_DirectFp16Conv_MatchesTheFp16RoundedCpuReference()
    {
        using var backend = new OpenClBackend();
        Skip.IfNot(backend.IsAvailable, "OpenCL is not available.");
        Check(backend);
    }

    [SkippableFact]
    public void Vulkan_DirectFp16Conv_MatchesTheFp16RoundedCpuReference()
    {
        var backend = VulkanBackend.Instance;
        Skip.IfNot(backend.Initialize() && backend.IsGlslCompilerAvailable, "Vulkan with a GLSL compiler is not available.");
        Skip.IfNot(backend.Fp16DirectConvAvailable, "The direct FP16 convolution shader did not build on this device.");
        Check(backend);
    }

    [SkippableFact]
    public void Hip_DirectFp16Conv_MatchesTheFp16RoundedCpuReference()
    {
        using var backend = new HipBackend();
        Skip.IfNot(backend.IsAvailable, "HIP is not available.");
        Check(backend);
    }

    [SkippableFact]
    public void Metal_DirectFp16Conv_MatchesTheFp16RoundedCpuReference()
    {
        Skip.IfNot(OperatingSystem.IsMacOS() || OperatingSystem.IsIOS(), "Metal is only available on macOS/iOS.");
        using var backend = new MetalBackend();
        Skip.IfNot(backend.IsAvailable, "Metal is not available.");
        Check(backend);
    }

    [SkippableFact]
    public void OpenCl_RejectsAnOutputExtentThatDoesNotFollowFromTheGeometry_AndShortBuffers()
    {
        using var backend = new OpenClBackend();
        Skip.IfNot(backend.IsAvailable, "OpenCL is not available.");
        var route = (IFp16ActivationBackend)backend;
        using var input = backend.AllocateBuffer(1 * 2 * 5 * 5);
        using var weightsHalf = backend.AllocateBuffer((3 * 2 * 3 * 3 + 1) / 2);
        using var output = backend.AllocateBuffer(1 * 3 * 3 * 3);

        // 5x5, 3x3 kernel, no padding gives 3x3.
        Assert.Throws<ArgumentException>(() =>
            route.Conv2dDirectFp16Hw(input, weightsHalf, output, 1, 2, 5, 5, 3, 4, 4, 3, 3, 1, 1, 0, 0, 1, 1));
        using var shortWeights = backend.AllocateBuffer(3);
        Assert.Throws<ArgumentException>(() =>
            route.Conv2dDirectFp16Hw(input, shortWeights, output, 1, 2, 5, 5, 3, 3, 3, 3, 3, 1, 1, 0, 0, 1, 1));
        using var shortOutput = backend.AllocateBuffer(5);
        Assert.Throws<ArgumentException>(() =>
            route.Conv2dDirectFp16Hw(input, weightsHalf, shortOutput, 1, 2, 5, 5, 3, 3, 3, 3, 3, 1, 1, 0, 0, 1, 1));
        using var shortInput = backend.AllocateBuffer(7);
        Assert.Throws<ArgumentException>(() =>
            route.Conv2dDirectFp16Hw(shortInput, weightsHalf, output, 1, 2, 5, 5, 3, 3, 3, 3, 3, 1, 1, 0, 0, 1, 1));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            route.Conv2dDirectFp16Hw(input, weightsHalf, output, 1, 2, 5, 5, 3, 3, 3, 3, 3, 0, 1, 0, 0, 1, 1));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            route.Conv2dDirectFp16Hw(input, weightsHalf, output, 1, 2, 5, 5, 3, 3, 3, 3, 3, 1, 1, -1, 0, 1, 1));
    }

#if NET7_0_OR_GREATER
    [SkippableFact]
    public void WebGpu_DirectFp16Conv_MatchesTheFp16RoundedCpuReference()
    {
        using var backend = new AiDotNet.Tensors.Engines.DirectGpu.WebGpu.WebGpuBackend();
        Skip.IfNot(backend.IsAvailable, "WebGPU is not available.");
        Check(backend);
    }

    // The shared kernel through a launcher (the WebGPU WGSL port on Dawn), with buffers on the host backend.
    [SkippableFact]
    public void WebGpuWgsl_DirectFp16Conv_MatchesTheFp16RoundedCpuReference_OnDawn()
    {
        using var dawn = DawnWgslLauncher.TryCreate();
        Skip.If(dawn is null, "Set AIDOTNET_WEBGPU_NODE_DIR to a directory with node_modules/webgpu (npm install webgpu).");
        using var host = new OpenClBackend();
        Skip.IfNot(host.IsAvailable, "OpenCL hosts the buffers and is not available.");
        var launch = dawn!.For(host);
        Check(host, (input, weightsHalf, output, batch, inC, inH, inW, outC, outH, outW, kH, kW, sH, sW, pH, pW, dH, dW)
            => PortableFp16Activation.Conv2dDirectHalfWeights(host, launch, input, weightsHalf, output, batch, inC, inH, inW,
                outC, outH, outW, kH, kW, sH, sW, pH, pW, dH, dW));
    }
#endif
}
#endif
