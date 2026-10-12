#if NET6_0_OR_GREATER
using System;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.HIP;
using AiDotNet.Tensors.Engines.DirectGpu.OpenCL;
using AiDotNet.Tensors.Engines.DirectGpu.Vulkan;
using AiDotNet.Tensors.Engines.Gpu;
using AiDotNet.Tensors.Helpers;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// <see cref="IGpuMixedPrecisionConvBackend"/> on every non-CUDA backend (#1142) against cuDNN's contract: contiguous
/// half / bfloat16 NCHW operands, FP32 accumulation, round-to-nearest-even into the 16-bit output, output overwritten.
/// The reference convolves the stored 16-bit values in double and rounds once, so a backend may differ from it only
/// where FP32 accumulation order moves a value across a 16-bit rounding boundary (one unit in the last place).
/// </summary>
[Collection("DirectGpuSerial")]
public sealed class MixedPrecisionConvBackendParityTests
{
    private readonly record struct Conv(int Batch, int InC, int InH, int InW, int OutC, int KH, int KW,
        int SH, int SW, int PH, int PW, int DH, int DW)
    {
        public int OutH => (InH + 2 * PH - DH * (KH - 1) - 1) / SH + 1;
        public int OutW => (InW + 2 * PW - DW * (KW - 1) - 1) / SW + 1;
        public int InputCount => Batch * InC * InH * InW;
        public int KernelCount => OutC * InC * KH * KW;
        public int OutputCount => Batch * OutC * OutH * OutW;
    }

    private static readonly Conv[] Cases =
    {
        new(2, 3, 7, 6, 4, 3, 2, 2, 1, 1, 1, 1, 2),
        new(1, 1, 3, 3, 1, 2, 2, 1, 1, 0, 0, 1, 1), // 9 input elements: the last storage word is half empty
    };

    private static ushort ToBits(float value, GpuMixedPrecisionDataType type)
    {
        if (type == GpuMixedPrecisionDataType.Half) return BitConverter.HalfToUInt16Bits((Half)value);
        uint f = BitConverter.SingleToUInt32Bits(value);
        if ((f & 0x7fffffffu) > 0x7f800000u) return (ushort)((f >> 16) | 0x40u);
        return (ushort)((f + 0x7fffu + ((f >> 16) & 1u)) >> 16);
    }

    private static float FromBits(ushort bits, GpuMixedPrecisionDataType type)
        => type == GpuMixedPrecisionDataType.Half
            ? (float)BitConverter.UInt16BitsToHalf(bits)
            : BitConverter.UInt32BitsToSingle((uint)bits << 16);

    private static float Quantize(double value, GpuMixedPrecisionDataType type)
        => type == GpuMixedPrecisionDataType.Float32 ? (float)value : FromBits(ToBits((float)value, type), type);

    private static float[] RandomValues(int count, int seed, GpuMixedPrecisionDataType type)
    {
        var rng = RandomHelper.CreateSeededRandom(seed);
        var values = new float[count];
        for (int i = 0; i < count; i++) values[i] = Quantize(rng.NextDouble() * 2 - 1, type);
        return values;
    }

    private static IGpuBuffer Upload(IDirectGpuBackend backend, float[] values, GpuMixedPrecisionDataType type)
    {
        if (type == GpuMixedPrecisionDataType.Float32) return backend.AllocateBuffer(values);
        var words = new float[(values.Length + 1) / 2];
        for (int w = 0; w < words.Length; w++)
        {
            uint lo = ToBits(values[2 * w], type);
            uint hi = 2 * w + 1 < values.Length ? ToBits(values[2 * w + 1], type) : 0u;
            words[w] = BitConverter.UInt32BitsToSingle(lo | (hi << 16));
        }
        return backend.AllocateBuffer(words);
    }

    /// <summary>An output buffer pre-filled with a pattern, so a kernel that accumulates instead of writing shows.</summary>
    private static IGpuBuffer Garbage(IDirectGpuBackend backend, int count, GpuMixedPrecisionDataType type)
    {
        int slots = type == GpuMixedPrecisionDataType.Float32 ? count : (count + 1) / 2;
        var fill = new float[slots];
        for (int i = 0; i < slots; i++) fill[i] = type == GpuMixedPrecisionDataType.Float32 ? 7.5f : BitConverter.UInt32BitsToSingle(0x47004700u);
        return backend.AllocateBuffer(fill);
    }

    private static float[] Download(IDirectGpuBackend backend, IGpuBuffer buffer, int count, GpuMixedPrecisionDataType type)
    {
        float[] raw = backend.DownloadBuffer(buffer);
        if (type == GpuMixedPrecisionDataType.Float32) return raw.AsSpan(0, count).ToArray();
        var values = new float[count];
        for (int i = 0; i < count; i++)
        {
            uint word = BitConverter.SingleToUInt32Bits(raw[i / 2]);
            values[i] = FromBits((ushort)((i & 1) != 0 ? word >> 16 : word & 0xffffu), type);
        }
        return values;
    }

    private static void ForEachTap(Conv c, Action<int, int, int> visit)
    {
        for (int n = 0; n < c.Batch; n++)
            for (int o = 0; o < c.OutC; o++)
                for (int oh = 0; oh < c.OutH; oh++)
                    for (int ow = 0; ow < c.OutW; ow++)
                        for (int ci = 0; ci < c.InC; ci++)
                            for (int kh = 0; kh < c.KH; kh++)
                                for (int kw = 0; kw < c.KW; kw++)
                                {
                                    int ih = oh * c.SH - c.PH + kh * c.DH, iw = ow * c.SW - c.PW + kw * c.DW;
                                    if (ih < 0 || ih >= c.InH || iw < 0 || iw >= c.InW) continue;
                                    visit(((n * c.InC + ci) * c.InH + ih) * c.InW + iw,
                                        ((o * c.InC + ci) * c.KH + kh) * c.KW + kw,
                                        ((n * c.OutC + o) * c.OutH + oh) * c.OutW + ow);
                                }
    }

    private static void AssertWithinOneUlp(double[] exact, float[] actual, GpuMixedPrecisionDataType type, string what)
    {
        // One unit in the last place of the stored format, relative to the value (absolute near zero).
        double ulp = type switch
        {
            GpuMixedPrecisionDataType.Half => 1.0 / 1024,
            GpuMixedPrecisionDataType.BFloat16 => 1.0 / 128,
            _ => 1e-5,
        };
        for (int i = 0; i < exact.Length; i++)
        {
            double allowed = ulp * Math.Max(Math.Abs(exact[i]), 1.0 / 64) + 1e-6;
            Assert.True(Math.Abs(Quantize(exact[i], type) - actual[i]) <= allowed,
                $"{what} {type} [{i}]: expected {Quantize(exact[i], type)}, got {actual[i]}");
        }
    }

    private static void Check(IDirectGpuBackend backend)
    {
        var mixed = backend.MixedPrecisionConv;
        Assert.NotNull(mixed);
        var conv = Assert.IsAssignableFrom<IGpuMixedPrecisionConvBackend>(mixed);
        Assert.True(conv.SupportsHalfConv && conv.SupportsBFloat16Conv);
        foreach (var type in new[] { GpuMixedPrecisionDataType.Half, GpuMixedPrecisionDataType.BFloat16, GpuMixedPrecisionDataType.Float32 })
            foreach (var c in Cases)
            {
                float[] x = RandomValues(c.InputCount, 1 + c.InputCount, type);
                float[] w = RandomValues(c.KernelCount, 2 + c.KernelCount, type);
                float[] dy = RandomValues(c.OutputCount, 3 + c.OutputCount, type);
                var y = new double[c.OutputCount];
                var dx = new double[c.InputCount];
                var dw = new double[c.KernelCount];
                ForEachTap(c, (xi, wi, yi) =>
                {
                    y[yi] += (double)x[xi] * w[wi];
                    dx[xi] += (double)dy[yi] * w[wi];
                    dw[wi] += (double)x[xi] * dy[yi];
                });

                using var xBuf = Upload(backend, x, type);
                using var wBuf = Upload(backend, w, type);
                using var dyBuf = Upload(backend, dy, type);
                using var yBuf = Garbage(backend, c.OutputCount, type);
                using var dxBuf = Garbage(backend, c.InputCount, type);
                using var dwBuf = Garbage(backend, c.KernelCount, type);
                conv.Conv2DMixed(xBuf, wBuf, yBuf, c.Batch, c.InC, c.InH, c.InW, c.OutC, c.OutH, c.OutW,
                    c.KH, c.KW, c.SH, c.SW, c.PH, c.PW, c.DH, c.DW, type);
                conv.Conv2DBackwardDataMixed(wBuf, dyBuf, dxBuf, c.Batch, c.InC, c.InH, c.InW, c.OutC, c.OutH, c.OutW,
                    c.KH, c.KW, c.SH, c.SW, c.PH, c.PW, c.DH, c.DW, type);
                conv.Conv2DBackwardFilterMixed(xBuf, dyBuf, dwBuf, c.Batch, c.InC, c.InH, c.InW, c.OutC, c.OutH, c.OutW,
                    c.KH, c.KW, c.SH, c.SW, c.PH, c.PW, c.DH, c.DW, type);
                AssertWithinOneUlp(y, Download(backend, yBuf, c.OutputCount, type), type, $"forward {c}");
                AssertWithinOneUlp(dx, Download(backend, dxBuf, c.InputCount, type), type, $"backward-data {c}");
                AssertWithinOneUlp(dw, Download(backend, dwBuf, c.KernelCount, type), type, $"backward-filter {c}");
            }
    }

    [SkippableFact]
    public void OpenCl_MixedPrecisionConv_MatchesCudnnContract()
    {
        using var backend = new OpenClBackend();
        Skip.IfNot(backend.IsAvailable, "OpenCL is not available.");
        Check(backend);
    }

    [SkippableFact]
    public void Vulkan_MixedPrecisionConv_MatchesCudnnContract()
    {
        var backend = VulkanBackend.Instance;
        Skip.IfNot(backend.Initialize() && backend.IsGlslCompilerAvailable, "Vulkan with a GLSL compiler is not available.");
        Check(backend);
    }

    [SkippableFact]
    public void Hip_MixedPrecisionConv_MatchesCudnnContract()
    {
        using var backend = new HipBackend();
        Skip.IfNot(backend.IsAvailable, "HIP is not available.");
        Check(backend);
    }
}
#endif
