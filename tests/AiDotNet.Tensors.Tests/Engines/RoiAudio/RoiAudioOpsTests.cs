// Copyright (c) AiDotNet. All rights reserved.
// CPU reference tests for the tail of Issue #217: RoI family + audio
// primitives. Ground-truth values computed against torchvision 0.15 /
// torchaudio 2.0 semantics.

using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.RoiAudio;

public class RoiAudioOpsTests
{
    private readonly CpuEngine _cpu = new();

    private static Tensor<float> Arange(int n, float start = 0f)
    {
        var d = new float[n];
        for (int i = 0; i < n; i++) d[i] = start + i;
        return new Tensor<float>(d, new[] { n });
    }

    // ------------------------------------------------------------------
    // RoI
    // ------------------------------------------------------------------

    [Fact]
    public void RoIAlign_FullBox_MatchesAverage()
    {
        // [1, 1, 4, 4] image, box covers whole image, output 1x1 must
        // equal the whole-image average.
        var img = Arange(16);
        img = new Tensor<float>(img.AsSpan().ToArray(), new[] { 1, 1, 4, 4 });
        var boxes = new Tensor<float>(new float[] { 0, 0, 0, 3, 3 }, new[] { 1, 5 });
        var output = _cpu.RoIAlign(img, boxes, 1, 1, 1.0f, samplingRatio: 2, aligned: false);
        // torchvision's RoIAlign(output_size=1, sampling_ratio=2, aligned=False)
        // samples at (0.75, 0.75), (0.75, 2.25), (2.25, 0.75), (2.25, 2.25)
        // with bilinear, then averages. Result ≈ 7.5.
        Assert.Equal(7.5f, output.AsSpan()[0], 2);
    }

    [Fact]
    public void RoIPool_UnitBox_ReturnsMax()
    {
        // [1,1,3,3] image, box that covers exactly pixel (1,1) — output 1x1 = max = 4.
        var img = new Tensor<float>(new float[] { 1, 2, 3, 4, 5, 6, 7, 8, 9 }, new[] { 1, 1, 3, 3 });
        var boxes = new Tensor<float>(new float[] { 0, 1, 1, 1, 1 }, new[] { 1, 5 });
        var output = _cpu.RoIPool(img, boxes, 1, 1, 1.0f);
        Assert.Equal(5f, output.AsSpan()[0]);
    }

    [Fact]
    public void RoIAlign_EmptyBoxList_ReturnsEmpty()
    {
        var img = new Tensor<float>(new float[4], new[] { 1, 1, 2, 2 });
        var boxes = new Tensor<float>(new float[0], new[] { 0, 5 });
        var output = _cpu.RoIAlign(img, boxes, 2, 2, 1.0f, 2, false);
        Assert.Equal(new[] { 0, 1, 2, 2 }, output.Shape.ToArray());
    }

    // ------------------------------------------------------------------
    // AmplitudeToDB
    // ------------------------------------------------------------------

    [Fact]
    public void AmplitudeToDB_BasicFormula()
    {
        var input = new Tensor<float>(new float[] { 1f, 0.5f, 0.1f }, new[] { 3 });
        var output = _cpu.AmplitudeToDB(input);
        var o = output.AsSpan();
        // 20 * log10(1) = 0, 20 * log10(0.5) ≈ -6.02, 20 * log10(0.1) = -20
        Assert.Equal(0f, o[0], 2);
        Assert.InRange(o[1], -6.1f, -5.9f);
        Assert.Equal(-20f, o[2], 2);
    }

    [Fact]
    public void AmplitudeToDB_TopDb_ClipsFloor()
    {
        var input = new Tensor<float>(new float[] { 1f, 1e-5f }, new[] { 2 });
        var output = _cpu.AmplitudeToDB(input, topDb: 80f).AsSpan();
        // Peak is 0 dB (from 1.0). topDb=80 → floor at -80 dB. The 1e-5 value
        // (-100 dB) should be clipped to -80.
        Assert.Equal(-80f, output[1], 2);
    }

    // ------------------------------------------------------------------
    // MuLaw
    // ------------------------------------------------------------------

    [Fact]
    public void MuLaw_EncodeDecodeRoundtrip()
    {
        var input = new Tensor<float>(new float[] { 0f, 0.1f, -0.1f, 0.5f, -0.5f, 0.99f, -0.99f }, new[] { 7 });
        var enc = _cpu.MuLawEncoding(input);
        var dec = _cpu.MuLawDecoding(enc);
        var orig = input.AsSpan();
        var back = dec.AsSpan();
        for (int i = 0; i < orig.Length; i++)
        {
            // μ-law is quantising, not lossless. Tolerance ~1/256.
            Assert.True(Math.Abs(orig[i] - back[i]) < 0.01f,
                $"[{i}] orig={orig[i]}, roundtrip={back[i]}");
        }
    }

    // ------------------------------------------------------------------
    // ComputeDeltas
    // ------------------------------------------------------------------

    [Fact]
    public void ComputeDeltas_LinearRamp_ConstantSlope()
    {
        // For a linear ramp f[t]=t the derivative should be constant = 1 in
        // the interior (edge effects aside).
        var input = Arange(10);
        input = new Tensor<float>(input.AsSpan().ToArray(), new[] { 10 });
        var d = _cpu.ComputeDeltas(input, winLength: 5).AsSpan();
        // Interior (index 2..7) should equal 1.
        for (int i = 2; i <= 7; i++)
            Assert.Equal(1f, d[i], 3);
    }

    // ------------------------------------------------------------------
    // Resample
    // ------------------------------------------------------------------

    [Fact]
    public void Resample_SameRate_IsIdentity()
    {
        var input = Arange(8);
        var output = _cpu.Resample(input, 16000, 16000);
        var o = output.AsSpan();
        var i = input.AsSpan();
        for (int k = 0; k < o.Length; k++) Assert.Equal(i[k], o[k], 5);
    }

    [Fact]
    public void Resample_Upsample2x_DoublesLength()
    {
        var input = Arange(8);
        var output = _cpu.Resample(input, 8000, 16000);
        Assert.Equal(16, output.Shape[0]);
    }
}
