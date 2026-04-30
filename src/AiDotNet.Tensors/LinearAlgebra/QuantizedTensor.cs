// Copyright (c) AiDotNet. All rights reserved.

using System;
using AiDotNet.Tensors.NumericOperations;

namespace AiDotNet.Tensors.LinearAlgebra;

/// <summary>Quantization scheme — symmetric pivots around zero (no zero-point);
/// asymmetric supports an offset for distributions that aren't zero-centered.</summary>
public enum QuantizationScheme
{
    SymmetricPerGroup = 0,
    AsymmetricPerGroup = 1,
}

/// <summary>Bit-width of the quantized tensor's lanes.</summary>
public enum QuantizationBits
{
    Int8 = 8,
    Int4 = 4,
}

/// <summary>
/// Cross-backend int8 / int4 quantized tensor wrapper (issue #276 sub-feature 3).
/// Carries the raw byte payload + per-group scale metadata; consumed by
/// fused dequant-matmul kernels in every direct-GPU backend (CUDA / HIP /
/// Metal / OpenCL / Vulkan / WebGPU) and the CPU SIMD reference path.
///
/// <para>int8 path stores one sbyte per element; int4 packs two values
/// per byte via <see cref="PackedInt4"/>. Both share the same
/// <see cref="QuantizationScale"/> shape (per-group symmetric / asymmetric).
/// llama.cpp Q8_0 / Q4_0 layouts are bit-compatible at GroupSize=32.</para>
/// </summary>
public sealed class QuantizedTensor<T> where T : struct
{
    /// <summary>Logical tensor shape (post-dequantization).</summary>
    public int[] Shape { get; }

    /// <summary>Total element count.</summary>
    public int Length { get; }

    /// <summary>Bit-width.</summary>
    public QuantizationBits Bits { get; }

    /// <summary>Quantization scheme.</summary>
    public QuantizationScheme Scheme { get; }

    /// <summary>Raw payload — sbyte[] for int8, byte[] (PackedInt4 storage) for int4.</summary>
    public byte[] Payload { get; }

    /// <summary>Per-group scale metadata.</summary>
    public QuantizationScale Scale { get; }

    private QuantizedTensor(int[] shape, int length, QuantizationBits bits, QuantizationScheme scheme,
                           byte[] payload, QuantizationScale scale)
    {
        Shape = shape;
        Length = length;
        Bits = bits;
        Scheme = scheme;
        Payload = payload;
        Scale = scale;
    }

    /// <summary>Calibrate-from-float (absmax per-group). Default group=32 matches GGUF.</summary>
    public static QuantizedTensor<sbyte> FromFloatInt8(Tensor<float> source, int groupSize = 32,
                                                      QuantizationScheme scheme = QuantizationScheme.SymmetricPerGroup)
    {
        if (source is null) throw new ArgumentNullException(nameof(source));
        if (scheme != QuantizationScheme.SymmetricPerGroup)
            throw new NotSupportedException("Asymmetric int8 quantization not yet wired.");
        var src = source.AsSpan();
        var raw = new sbyte[src.Length];
        var scale = QuantizationHelpersInt8.QuantizeInt8(src, raw, groupSize);
        // Reinterpret sbyte[] as byte[] for cross-backend pointer plumbing.
        var bytes = new byte[raw.Length];
        Buffer.BlockCopy(raw, 0, bytes, 0, raw.Length);
        return new QuantizedTensor<sbyte>(
            (int[])source._shape.Clone(), src.Length, QuantizationBits.Int8, scheme, bytes, scale);
    }

    /// <summary>Calibrate-from-float for int4 (per-group symmetric, GGUF Q4_0 compatible).</summary>
    public static QuantizedTensor<PackedInt4> FromFloatInt4(Tensor<float> source, int groupSize = 32,
                                                           QuantizationScheme scheme = QuantizationScheme.SymmetricPerGroup)
    {
        if (source is null) throw new ArgumentNullException(nameof(source));
        if (scheme != QuantizationScheme.SymmetricPerGroup)
            throw new NotSupportedException("Asymmetric int4 quantization not yet wired.");
        var src = source.AsSpan();
        int packedLen = (src.Length + 1) / 2;
        var packed = new PackedInt4[packedLen];
        var scale = QuantizationHelpers.QuantizeInt4(src, packed, groupSize);
        var bytes = new byte[packedLen];
        for (int i = 0; i < packedLen; i++) bytes[i] = packed[i].RawValue;
        return new QuantizedTensor<PackedInt4>(
            (int[])source._shape.Clone(), src.Length, QuantizationBits.Int4, scheme, bytes, scale);
    }

    /// <summary>Round-trip: dequantize back to float for kernels that
    /// don't have a fused dequant-matmul path.</summary>
    public Tensor<float> Dequantize()
    {
        var t = new Tensor<float>(Shape);
        var dst = t.AsWritableSpan();
        if (Bits == QuantizationBits.Int8)
        {
            // Reinterpret byte[] as sbyte[] — same memory, signed view.
            var asSbyte = new sbyte[Payload.Length];
            Buffer.BlockCopy(Payload, 0, asSbyte, 0, Payload.Length);
            QuantizationHelpersInt8.DequantizeInt8(asSbyte, Scale, dst);
        }
        else // Int4
        {
            var packed = new PackedInt4[Payload.Length];
            for (int i = 0; i < Payload.Length; i++) packed[i] = new PackedInt4(Payload[i]);
            QuantizationHelpers.DequantizeInt4(packed, Scale, dst);
        }
        return t;
    }
}
