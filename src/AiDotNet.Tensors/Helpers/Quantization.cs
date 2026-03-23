using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Helpers;

/// <summary>
/// Quantization mode for tensor compression.
/// </summary>
public enum QuantizationMode
{
    /// <summary>Symmetric quantization: zero_point = 0, range = [-max, max].</summary>
    Symmetric,

    /// <summary>Asymmetric quantization: zero_point != 0, range = [min, max].</summary>
    Asymmetric
}

/// <summary>
/// Parameters for quantizing/dequantizing tensors.
/// </summary>
public sealed class QuantizationParams
{
    /// <summary>Scale factor: float_value = (quantized_value - ZeroPoint) * Scale.</summary>
    public float Scale { get; }

    /// <summary>Zero point offset for asymmetric quantization.</summary>
    public int ZeroPoint { get; }

    /// <summary>Quantization mode.</summary>
    public QuantizationMode Mode { get; }

    /// <summary>Number of bits (8 for INT8, 16 for FP16).</summary>
    public int Bits { get; }

    public QuantizationParams(float scale, int zeroPoint, QuantizationMode mode, int bits = 8)
    {
        Scale = scale;
        ZeroPoint = zeroPoint;
        Mode = mode;
        Bits = bits;
    }

    /// <summary>
    /// Computes quantization parameters from tensor statistics.
    /// </summary>
    public static QuantizationParams FromTensor(Tensor<float> tensor, QuantizationMode mode = QuantizationMode.Symmetric, int bits = 8)
    {
        var span = tensor.AsSpan();
        float min = float.MaxValue, max = float.MinValue;
        for (int i = 0; i < span.Length; i++)
        {
            if (span[i] < min) min = span[i];
            if (span[i] > max) max = span[i];
        }

        int qmin = -(1 << (bits - 1));
        int qmax = (1 << (bits - 1)) - 1;

        if (mode == QuantizationMode.Symmetric)
        {
            float absMax = Math.Max(Math.Abs(min), Math.Abs(max));
            float scale = absMax / qmax;
            if (scale == 0) scale = 1f;
            return new QuantizationParams(scale, 0, mode, bits);
        }
        else
        {
            float scale = (max - min) / (qmax - qmin);
            if (scale == 0) scale = 1f;
            int zeroPoint = (int)Math.Round(qmin - min / scale);
            zeroPoint = Math.Max(qmin, Math.Min(qmax, zeroPoint));
            return new QuantizationParams(scale, zeroPoint, mode, bits);
        }
    }
}

/// <summary>
/// Quantization and dequantization operations for tensor compression.
/// Supports INT8 symmetric/asymmetric quantization for inference acceleration.
/// </summary>
public static class Quantization
{
    /// <summary>
    /// Quantizes a float tensor to INT8.
    /// </summary>
    public static Tensor<sbyte> QuantizeInt8(Tensor<float> input, QuantizationParams qparams)
    {
        if (qparams.Scale <= 0)
            throw new ArgumentException("Scale must be positive.", nameof(qparams));

        var src = input.AsSpan();
        var result = new Tensor<sbyte>(input._shape);
        var dst = result.AsWritableSpan();
        float invScale = 1f / qparams.Scale;
        int zp = qparams.ZeroPoint;

        int length = src.Length;

        for (int i = 0; i < length; i++)
        {
            float val = src[i] * invScale + zp;
            val = MathF.Round(val);
            dst[i] = (sbyte)Math.Max(-128, Math.Min(127, (int)val));
        }

        return result;
    }

    /// <summary>
    /// Dequantizes an INT8 tensor back to float.
    /// </summary>
    public static Tensor<float> DequantizeInt8(Tensor<sbyte> input, QuantizationParams qparams)
    {
        var src = input.AsSpan();
        var result = new Tensor<float>(input._shape);
        var dst = result.AsWritableSpan();
        float scale = qparams.Scale;
        int zp = qparams.ZeroPoint;

        for (int i = 0; i < src.Length; i++)
        {
            dst[i] = (src[i] - zp) * scale;
        }

        return result;
    }

    /// <summary>
    /// Quantizes a float tensor to FP16 (Half precision).
    /// </summary>
    public static Tensor<Half> QuantizeFP16(Tensor<float> input)
    {
        var src = input.AsSpan();
        var result = new Tensor<Half>(input._shape);
        var dst = result.AsWritableSpan();

        for (int i = 0; i < src.Length; i++)
        {
            dst[i] = (Half)src[i];
        }

        return result;
    }

    /// <summary>
    /// Dequantizes an FP16 tensor back to float.
    /// </summary>
    public static Tensor<float> DequantizeFP16(Tensor<Half> input)
    {
        var src = input.AsSpan();
        var result = new Tensor<float>(input._shape);
        var dst = result.AsWritableSpan();

        for (int i = 0; i < src.Length; i++)
        {
            dst[i] = (float)src[i];
        }

        return result;
    }

    /// <summary>
    /// Computes quantization error (MSE) between original and quantized+dequantized tensor.
    /// </summary>
    public static double ComputeQuantizationError(Tensor<float> original, Tensor<float> dequantized)
    {
        var a = original.AsSpan();
        var b = dequantized.AsSpan();
        double sumSqErr = 0;
        for (int i = 0; i < a.Length; i++)
        {
            double diff = a[i] - b[i];
            sumSqErr += diff * diff;
        }
        return sumSqErr / a.Length;
    }
}
