// Copyright (c) AiDotNet. All rights reserved.

using System;
using System.Collections.Generic;
using AiDotNet.Tensors.NumericOperations;

namespace AiDotNet.Tensors.LinearAlgebra;

/// <summary>
/// Issue #276 sub-feature 3 (continued): quantization-aware training (QAT).
/// During training: gradients accumulate in float, but the FORWARD pass
/// reads the weight as if it were quantized — i.e. quantize → dequantize
/// round-trip on every read so the loss landscape includes the
/// quantization noise. Backward uses the straight-through estimator
/// (STE) so the gradient flows through the round-trip unchanged.
///
/// <para>At the optimizer step the float master weights are updated;
/// the inference-time quantized payload is re-quantized from the
/// updated master. Same shape as PyTorch's <c>torch.ao.quantization.QAT</c>.</para>
///
/// <para>Reference: Jacob et al. "Quantization and Training of Neural
/// Networks for Efficient Integer-Arithmetic-Only Inference" (CVPR 2018).</para>
/// </summary>
public sealed class QatTrainingHook
{
    private readonly QuantizationBits _bits;
    private readonly int _groupSize;
    private readonly QuantizationScheme _scheme;
    private readonly Dictionary<object, FloatMaster> _masters = new();

    public QatTrainingHook(
        QuantizationBits bits,
        int groupSize = 32,
        QuantizationScheme scheme = QuantizationScheme.SymmetricPerGroup)
    {
        if (bits != QuantizationBits.Int8 && bits != QuantizationBits.Int4)
            throw new ArgumentOutOfRangeException(nameof(bits),
                $"QAT only supports Int8 and Int4; got {bits}.");
        if (groupSize <= 0 || (groupSize & 1) != 0)
            throw new ArgumentOutOfRangeException(nameof(groupSize),
                $"groupSize must be a positive even integer; got {groupSize}.");
        if (scheme != QuantizationScheme.SymmetricPerGroup)
            throw new NotSupportedException(
                $"QAT currently supports SymmetricPerGroup only; got {scheme}.");
        _bits = bits;
        _groupSize = groupSize;
        _scheme = scheme;
    }

    /// <summary>Registers a float weight for QAT. The hook keeps the
    /// fp32 master copy and returns a "fake-quantized" view that reads
    /// the weight through quantize→dequantize on every forward.</summary>
    public Tensor<float> RegisterFloatMaster(object key, Tensor<float> floatWeight)
    {
        if (floatWeight is null) throw new ArgumentNullException(nameof(floatWeight));
        // Materialize non-contiguous views — AsSpan throws on non-contiguous,
        // and registering a sliced/transposed parameter is a legitimate use.
        var contig = floatWeight.IsContiguous ? floatWeight : floatWeight.Contiguous();
        var master = new FloatMaster
        {
            Buffer = contig.AsSpan().ToArray(),
            Shape = (int[])contig._shape.Clone(),
        };
        _masters[key] = master;
        return FakeQuantize(key);
    }

    /// <summary>Returns the fake-quantized view of the master. Forward
    /// reads this; backward STE means the gradient propagates back to
    /// the float master without modification (the round-trip is treated
    /// as identity by the autograd graph).</summary>
    public Tensor<float> FakeQuantize(object key)
    {
        var m = _masters[key];
        var fakeQ = new Tensor<float>(m.Shape);
        var src = m.Buffer.AsSpan();
        var dst = fakeQ.AsWritableSpan();
        if (_bits == QuantizationBits.Int8)
        {
            var raw = new sbyte[src.Length];
            var scale = QuantizationHelpersInt8.QuantizeInt8(src, raw, _groupSize);
            QuantizationHelpersInt8.DequantizeInt8(raw, scale, dst);
        }
        else // Int4
        {
            var packed = new PackedInt4[(src.Length + 1) / 2];
            var scale = QuantizationHelpers.QuantizeInt4(src, packed, _groupSize);
            QuantizationHelpers.DequantizeInt4(packed, scale, dst);
        }
        return fakeQ;
    }

    /// <summary>Optimizer-step hook: applies the float gradient to the
    /// master, then re-quantizes the inference payload. Caller calls
    /// once per parameter per step.</summary>
    public void OptimizerStep(object key, ReadOnlySpan<float> floatGradient, float learningRate)
    {
        var m = _masters[key];
        if (floatGradient.Length != m.Buffer.Length)
            throw new ArgumentException("Gradient length must match master length.");
        for (int i = 0; i < m.Buffer.Length; i++)
            m.Buffer[i] -= learningRate * floatGradient[i];
        // Inference payload is freshly fake-quantized on next FakeQuantize call.
    }

    /// <summary>Snapshots the current quantized payload + scale metadata
    /// for export — typically saved to disk as a model checkpoint.</summary>
    public QuantizedTensor<sbyte> ExportInt8(object key)
    {
        if (_bits != QuantizationBits.Int8)
            throw new InvalidOperationException("ExportInt8 requires the hook to be configured for Int8.");
        var m = _masters[key];
        var t = new Tensor<float>(m.Shape);
        m.Buffer.AsSpan().CopyTo(t.AsWritableSpan());
        return QuantizedTensor<sbyte>.FromFloatInt8(t, _groupSize, _scheme);
    }

    /// <summary>Same for int4.</summary>
    public QuantizedTensor<PackedInt4> ExportInt4(object key)
    {
        if (_bits != QuantizationBits.Int4)
            throw new InvalidOperationException("ExportInt4 requires the hook to be configured for Int4.");
        var m = _masters[key];
        var t = new Tensor<float>(m.Shape);
        m.Buffer.AsSpan().CopyTo(t.AsWritableSpan());
        return QuantizedTensor<PackedInt4>.FromFloatInt4(t, _groupSize, _scheme);
    }

    private sealed class FloatMaster
    {
        public float[] Buffer = Array.Empty<float>();
        public int[] Shape = Array.Empty<int>();
    }
}
