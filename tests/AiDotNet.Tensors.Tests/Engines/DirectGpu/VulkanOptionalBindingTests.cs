using System;
using System.Linq;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.Vulkan;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// The Vulkan kernels with an optional operand (pooling's argmax indices, a bias, a modulation mask) run a variant
/// without that binding when the operand is left out, instead of allocating and filling a placeholder buffer per call.
/// The variant must give what the full kernel gives with the neutral operand supplied: bit for bit where the operand
/// is only written or added, to rounding where it was a factor inside a sum.
/// </summary>
[Collection("VulkanGlobalState")]
public class VulkanOptionalBindingTests
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
        return backend!;
    }

    private static float[] Values(int count, int seed, float scale = 1f)
    {
        var rng = new Random(seed);
        var a = new float[count];
        for (int i = 0; i < count; i++) a[i] = (float)((rng.NextDouble() * 2 - 1) * scale);
        return a;
    }

    private static void AssertSameBits(string what, float[] expected, float[] actual, int count)
    {
        bool anyNonZero = false;
        for (int i = 0; i < count; i++)
        {
            Assert.True(BitConverter.ToInt32(BitConverter.GetBytes(expected[i]), 0) == BitConverter.ToInt32(BitConverter.GetBytes(actual[i]), 0),
                $"{what}[{i}]: with the operand {expected[i]:R}, without it {actual[i]:R}");
            anyNonZero |= expected[i] != 0f;
        }
        // A comparison of two all-zero buffers would pass with nothing computed.
        Assert.True(anyNonZero, $"{what}: the reference result is all zeros");
    }

    // Multiplying by a mask of ones read from memory and multiplying by nothing are the same value, but with the
    // multiply gone the compiler may fuse the remaining multiply into the running sum, so sums can differ in the
    // last bits.
    private static void AssertSameToRounding(string what, float[] expected, float[] actual, int count)
    {
        double scale = 0;
        for (int i = 0; i < count; i++) scale = Math.Max(scale, Math.Abs(expected[i]));
        Assert.True(scale > 0, $"{what}: the reference result is all zeros");
        for (int i = 0; i < count; i++)
            Assert.True(Math.Abs((double)expected[i] - actual[i]) <= 4e-6 * scale,
                $"{what}[{i}]: with the operand {expected[i]:R}, without it {actual[i]:R}");
    }

    [SkippableTheory]
    [InlineData(2, 3, 9, 11, 2, 2, 2, 2, 0, 0)]
    [InlineData(1, 4, 13, 7, 3, 3, 2, 2, 1, 1)]
    [InlineData(3, 2, 8, 8, 3, 2, 1, 2, 1, 0)]
    public void MaxPool2D_WithoutIndices_MatchesWithIndices_AndCpu(int batch, int channels, int h, int w,
        int kh, int kw, int sh, int sw, int ph, int pw)
    {
        var backend = BackendOrSkip();
        int oh = (h + 2 * ph - kh) / sh + 1, ow = (w + 2 * pw - kw) / sw + 1, total = batch * channels * oh * ow;
        float[] x = Values(batch * channels * h * w, 3 + kh, 5f);
        using var xb = backend.AllocateBuffer(x);
        using var withIdx = backend.AllocateBuffer(total);
        using var without = backend.AllocateBuffer(total);
        using var idx = backend.AllocateIntBuffer(total);
        backend.Fill(without, float.NaN, total);
        backend.MaxPool2D(xb, withIdx, idx, batch, channels, h, w, oh, ow, kh, kw, sh, sw, ph, pw);
        backend.MaxPool2D(xb, without, null, batch, channels, h, w, oh, ow, kh, kw, sh, sw, ph, pw);
        float[] expected = backend.DownloadBuffer(withIdx), actual = backend.DownloadBuffer(without);
        AssertSameBits("max pool 2d", expected, actual, total);

        for (int b = 0; b < batch; b++)
            for (int c = 0; c < channels; c++)
                for (int oy = 0; oy < oh; oy++)
                    for (int ox = 0; ox < ow; ox++)
                    {
                        float m = float.NegativeInfinity;
                        for (int ky = 0; ky < kh; ky++)
                            for (int kx = 0; kx < kw; kx++)
                            {
                                int iy = oy * sh - ph + ky, ix = ox * sw - pw + kx;
                                if (iy >= 0 && iy < h && ix >= 0 && ix < w) m = Math.Max(m, x[((b * channels + c) * h + iy) * w + ix]);
                            }
                        Assert.Equal(m, actual[((b * channels + c) * oh + oy) * ow + ox]);
                    }
    }

    [SkippableFact]
    public void MaxPool3D_WithoutIndices_MatchesWithIndices()
    {
        var backend = BackendOrSkip();
        const int batch = 2, channels = 3, d = 6, h = 7, w = 8, k = 2, s = 2;
        int od = (d - k) / s + 1, oh = (h - k) / s + 1, ow = (w - k) / s + 1, total = batch * channels * od * oh * ow;
        using var xb = backend.AllocateBuffer(Values(batch * channels * d * h * w, 5, 4f));
        using var withIdx = backend.AllocateBuffer(total);
        using var without = backend.AllocateBuffer(total);
        using var idx = backend.AllocateIntBuffer(total);
        backend.Fill(without, float.NaN, total);
        backend.MaxPool3D(xb, withIdx, idx, batch, channels, d, h, w, od, oh, ow, k, k, k, s, s, s);
        backend.MaxPool3D(xb, without, null, batch, channels, d, h, w, od, oh, ow, k, k, k, s, s, s);
        AssertSameBits("max pool 3d", backend.DownloadBuffer(withIdx), backend.DownloadBuffer(without), total);
    }

    [SkippableFact]
    public void LocallyConnectedConv2D_WithoutBias_MatchesAZeroBias()
    {
        var backend = BackendOrSkip();
        const int batch = 2, inC = 3, h = 7, w = 6, outC = 4, kh = 3, kw = 2, sh = 2, sw = 1;
        int oh = (h - kh) / sh + 1, ow = (w - kw) / sw + 1, total = batch * outC * oh * ow;
        using var xb = backend.AllocateBuffer(Values(batch * inC * h * w, 7));
        using var wb = backend.AllocateBuffer(Values(oh * ow * outC * inC * kh * kw, 9));
        using var zero = backend.AllocateBuffer(new float[outC]);
        using var withBias = backend.AllocateBuffer(total);
        using var without = backend.AllocateBuffer(total);
        backend.Fill(without, float.NaN, total);
        backend.LocallyConnectedConv2D(xb, wb, zero, withBias, batch, inC, h, w, outC, oh, ow, kh, kw, sh, sw);
        backend.LocallyConnectedConv2D(xb, wb, null, without, batch, inC, h, w, outC, oh, ow, kh, kw, sh, sw);
        AssertSameBits("locally connected", backend.DownloadBuffer(withBias), backend.DownloadBuffer(without), total);
    }

    [SkippableFact]
    public void DeformableConv2D_ForwardAndBackward_WithoutMask_MatchAnAllOnesMask()
    {
        var backend = BackendOrSkip();
        const int batch = 2, inC = 4, h = 7, w = 6, outC = 4, k = 3, groups = 2, dg = 2;
        const int oh = 7, ow = 6; // stride 1, pad 1, dilation 1
        int outTotal = batch * outC * oh * ow, inTotal = batch * inC * h * w;
        int weightTotal = outC * (inC / groups) * k * k, offsetTotal = batch * dg * 2 * k * k * oh * ow;
        int maskTotal = batch * dg * k * k * oh * ow;
        using var xb = backend.AllocateBuffer(Values(inTotal, 11));
        using var wb = backend.AllocateBuffer(Values(weightTotal, 13));
        using var ob = backend.AllocateBuffer(Values(offsetTotal, 17, 0.8f));
        using var gob = backend.AllocateBuffer(Values(outTotal, 19));
        using var ones = backend.AllocateBuffer(Enumerable.Repeat(1f, maskTotal).ToArray());

        void Check(string what, int count, Action<IGpuBuffer?, IGpuBuffer> run)
        {
            using var withMask = backend.AllocateBuffer(count);
            using var without = backend.AllocateBuffer(count);
            backend.Fill(withMask, 0f, count);
            backend.Fill(without, float.NaN, count);
            run(ones, withMask);
            run(null, without);
            AssertSameToRounding(what, backend.DownloadBuffer(withMask), backend.DownloadBuffer(without), count);
        }

        Check("deformable forward", outTotal, (mask, output) =>
            backend.DeformableConv2D(xb, wb, ob, mask, output, batch, inC, h, w, outC, oh, ow, k, k, 1, 1, 1, 1, 1, 1, groups, dg));
        Check("deformable backward input", inTotal, (mask, output) =>
            backend.DeformableConv2DBackwardInput(gob, wb, ob, mask, output, batch, inC, h, w, outC, oh, ow, k, k, 1, 1, 1, 1, 1, 1, groups, dg));
        Check("deformable backward weights", weightTotal, (mask, output) =>
            backend.DeformableConv2DBackwardWeights(gob, xb, ob, mask, output, batch, inC, h, w, outC, oh, ow, k, k, 1, 1, 1, 1, 1, 1, groups, dg));
        Check("deformable backward offset", offsetTotal, (mask, output) =>
            backend.DeformableConv2DBackwardOffset(gob, xb, wb, ob, mask, output, batch, inC, h, w, outC, oh, ow, k, k, 1, 1, 1, 1, 1, 1, groups, dg));
    }

    public static TheoryData<string, string, string, int> Variants() => new()
    {
        { "max pool 2d", VulkanConvPoolKernels.MaxPool2DNoIndices, "indices", 2 },
        { "max pool 3d", VulkanConvPoolKernels.MaxPool3DNoIndices, "indices", 2 },
        { "locally connected", VulkanConvPoolKernels.LocallyConnectedConv2DNoBias, "bias", 3 },
        { "deformable forward", VulkanConvPoolKernels.DeformableConv2DNoMask, "mask", 4 },
        { "deformable backward input", VulkanConvPoolKernels.DeformableConv2DBackwardInputNoMask, "mask", 4 },
        { "deformable backward weights", VulkanConvPoolKernels.DeformableConv2DBackwardWeightsNoMask, "mask", 4 },
        { "deformable backward offset", VulkanConvPoolKernels.DeformableConv2DBackwardOffsetNoMask, "mask", 5 },
    };

    [Theory]
    [MemberData(nameof(Variants))]
    public void Variant_DropsTheOperand_AndKeepsItsBindingsContiguous(string what, string source, string operand, int bindings)
    {
        Assert.DoesNotContain(operand + "[", source);
        for (int b = 0; b < bindings; b++)
            Assert.True(source.Contains($"layout(set=0, binding={b})"), $"{what}: binding {b} is missing");
        Assert.False(source.Contains($"layout(set=0, binding={bindings})"), $"{what}: an extra binding {bindings} remains");
    }

    // No device needed: libshaderc compiles on the CPU, so this runs wherever the native library loads (CI included).
    [SkippableTheory]
    [MemberData(nameof(Variants))]
    public void Variant_CompilesToSpirv(string what, string source, string operand, int bindings)
    {
        _ = operand; _ = bindings;
        VulkanGlslCompiler compiler;
        try { compiler = new VulkanGlslCompiler(); }
        catch (Exception ex) when (ex is DllNotFoundException or TypeInitializationException or BadImageFormatException) { compiler = null!; }
        Skip.If(compiler is null || !compiler.IsAvailable, "libshaderc is not available.");
        using (compiler)
        {
            uint[]? spirv = compiler!.CompileToSpirv(source);
            Assert.True(spirv is { Length: > 5 }, $"{what} did not compile: {compiler.LastError}");
        }
    }

    [Fact]
    public void WithoutBinding_RejectsASourceItCannotRewrite()
    {
        const string source = "layout(set=0, binding=0) buffer B0 { float a[]; };\nlayout(set=0, binding=1) buffer B1 { float b[]; };\nvoid main() { b[0] = a[0]; }";
        Assert.Throws<ArgumentException>(() => VulkanConvPoolKernels.WithoutBinding(source, 2, "a[0]", "0.0"));
        Assert.Throws<ArgumentException>(() => VulkanConvPoolKernels.WithoutBinding(source, 0, "missing[0]", "0.0"));
        string rewritten = VulkanConvPoolKernels.WithoutBinding(source, 0, "a[0]", "0.0");
        Assert.Equal("\nlayout(set=0, binding=0) buffer B1 { float b[]; };\nvoid main() { b[0] = 0.0; }", rewritten);
    }
}
