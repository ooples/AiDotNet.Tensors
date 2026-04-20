using System;
using AiDotNet.Tensors.Engines.Simd;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Simd;

/// <summary>
/// Correctness tests for Int8Quantizer (Path D foundation).
/// Verifies symmetric per-tensor quantize/dequantize round-trip stays within
/// the expected ~0.4% relative error band (1/254 LSB resolution).
/// </summary>
public class Int8QuantizerTests
{
    [Fact]
    public void ComputeSymmetricScale_AllZeros_ReturnsOne()
    {
        var data = new float[64];
        Assert.Equal(1f, Int8Quantizer.ComputeSymmetricScale(data));
    }

    [Fact]
    public void ComputeSymmetricScale_KnownMax_ReturnsMaxOver127()
    {
        var data = new float[100];
        for (int i = 0; i < data.Length; i++) data[i] = (i - 50) * 0.1f;
        // max|x| = 5.0 (data[100] would be, but length is 100 so data[99]=4.9)
        // Actually max |val| at i=0: |−5.0| = 5.0
        float expected = 5.0f / 127f;
        float actual = Int8Quantizer.ComputeSymmetricScale(data);
        Assert.Equal(expected, actual, precision: 6);
    }

    [Fact]
    public void QuantizeDequantize_SmallValues_RoundTripsExactly()
    {
        // Values that quantize to exact int8: [-127, 0, 127] * scale
        var data = new[] { -1.27f, 0f, 1.27f };
        float scale = 0.01f;  // chosen so 1.27 / 0.01 = 127
        var q = new sbyte[3];
        var dq = new float[3];
        Int8Quantizer.QuantizeFloat32ToInt8(data, q, scale);
        Int8Quantizer.DequantizeInt8ToFloat32(q, dq, scale);
        Assert.Equal((sbyte)-127, q[0]);
        Assert.Equal((sbyte)0, q[1]);
        Assert.Equal((sbyte)127, q[2]);
        Assert.Equal(-1.27f, dq[0], precision: 5);
        Assert.Equal(0f, dq[1]);
        Assert.Equal(1.27f, dq[2], precision: 5);
    }

    [Fact]
    public void QuantizeDequantize_OverRange_ClampsTo127()
    {
        var data = new[] { 1000f, -1000f };
        float scale = 1f;
        var q = new sbyte[2];
        Int8Quantizer.QuantizeFloat32ToInt8(data, q, scale);
        Assert.Equal((sbyte)127, q[0]);
        Assert.Equal((sbyte)-127, q[1]);
    }

    [Theory]
    [InlineData(64)]
    [InlineData(256)]
    [InlineData(768)]   // BERT hidden
    [InlineData(3072)]  // BERT FFN
    [InlineData(1234)]  // non-power-of-2, exercises tail
    public void RoundTripError_BertScaleData_StaysUnder1Percent(int n)
    {
        var rng = new Random(0xD01 + n);
        var data = new float[n];
        // BERT-like activation distribution: roughly N(0, 1) clipped to [-3, 3]
        for (int i = 0; i < n; i++)
        {
            double u1 = 1.0 - rng.NextDouble();
            double u2 = 1.0 - rng.NextDouble();
            double z = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
            if (z > 3.0) z = 3.0;
            if (z < -3.0) z = -3.0;
            data[i] = (float)z;
        }
        var (maxRelLarge, rms, snrDb) = Int8Quantizer.RoundTripError(data);
        // For values >10× the scale (well above noise floor), expect
        // ≤ 1/254 ≈ 0.4% relative error. Allow 1% to cover non-uniform
        // rounding edge cases.
        Assert.True(maxRelLarge < 0.01,
            $"max relative error (large values) {maxRelLarge:F4} exceeds 1% (n={n})");
        // SNR for INT8 symmetric quantization on Gaussian data should hit
        // ~30-40 dB. Anything above 25 dB is acceptable for inference.
        Assert.True(snrDb > 25.0,
            $"SNR {snrDb:F1} dB below 25 dB (n={n}, rms={rms:F6})");
    }

    [Fact]
    public void Quantize_SimdAndScalarMatch_BertFfnScale()
    {
        // Verify SIMD path matches scalar bit-exactly (same MidpointRounding)
        var rng = new Random(0xD42);
        int n = 1024;
        var data = new float[n];
        for (int i = 0; i < n; i++) data[i] = (float)(rng.NextDouble() * 6.0 - 3.0);
        float scale = Int8Quantizer.ComputeSymmetricScale(data);

        var qSimd = new sbyte[n];
        Int8Quantizer.QuantizeFloat32ToInt8(data, qSimd, scale);

        var qScalar = new sbyte[n];
        float invScale = 1f / scale;
        for (int i = 0; i < n; i++)
        {
            float v = data[i] * invScale;
            v = (float)Math.Round(v, MidpointRounding.ToEven);
            if (v < -127f) v = -127f;
            if (v > 127f) v = 127f;
            qScalar[i] = (sbyte)v;
        }

        // Allow at most 1 LSB difference per lane due to potential difference
        // between Avx.RoundToNearestInteger (MXCSR-controlled) and
        // MathF.Round (banker's rounding, hardcoded).
        int diffs = 0;
        for (int i = 0; i < n; i++)
            if (Math.Abs(qSimd[i] - qScalar[i]) > 1) diffs++;
        Assert.Equal(0, diffs);
    }
}
