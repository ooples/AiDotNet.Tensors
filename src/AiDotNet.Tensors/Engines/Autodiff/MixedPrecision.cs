// Copyright (c) AiDotNet. All rights reserved.

using System;
using System.Collections.Generic;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Autodiff;

/// <summary>
/// Issue #276 sub-feature 1 (continued): mixed-precision training plumbing.
/// PyTorch parity with <c>torch.cuda.amp</c> / <c>torch.bfloat16</c>:
///
/// <list type="bullet">
///   <item>Storage at low precision (bf16 / fp16) — saves memory.</item>
///   <item>Master weights at fp32 — preserves precision across many
///   small optimizer updates.</item>
///   <item>Gradient accumulator at fp32 — bf16 has only 7 mantissa bits;
///   accumulating in bf16 truncates ~99% of small gradients.</item>
///   <item>Loss scaling — multiplies loss by a scalar before backward
///   so small gradients don't underflow bf16/fp16. Optimizer step
///   divides by the scale before applying.</item>
/// </list>
///
/// <para>Reference: Micikevicius et al. "Mixed Precision Training"
/// (ICLR 2018). The dynamic-loss-scaling schedule (double on success,
/// halve on overflow detected via NaN/Inf in gradients) is what every
/// modern transformer implementation uses.</para>
/// </summary>
public sealed class MixedPrecisionConfig
{
    /// <summary>Loss-scale factor. Default 65536 = 2^16, matching
    /// PyTorch GradScaler's initial value.</summary>
    public float LossScale { get; set; } = 65536f;

    /// <summary>Dynamic-loss-scale schedule: double the scale every
    /// <see cref="GrowthInterval"/> successful steps; halve on any
    /// step that produced inf/nan gradients.</summary>
    public bool DynamicLossScale { get; set; } = true;

    /// <summary>Steps without overflow before doubling the scale.</summary>
    public int GrowthInterval { get; set; } = 2000;

    /// <summary>Multiplicative factor on growth (default 2.0).</summary>
    public float GrowthFactor { get; set; } = 2.0f;

    /// <summary>Multiplicative factor on backoff (default 0.5).</summary>
    public float BackoffFactor { get; set; } = 0.5f;

    /// <summary>Minimum loss scale. Below this, training abandons mixed
    /// precision and falls back to fp32 for the offending step.</summary>
    public float MinLossScale { get; set; } = 1.0f;
}

/// <summary>Runtime state for the mixed-precision schedule. Tracks the
/// current loss scale + step counter; <see cref="Update"/> is called
/// after every optimizer step with whether the step had inf/nan grads.</summary>
public sealed class GradScaler
{
    private readonly MixedPrecisionConfig _config;
    private float _scale;
    private int _stepsSinceLastBackoff;

    public GradScaler(MixedPrecisionConfig? config = null)
    {
        _config = config ?? new MixedPrecisionConfig();
        _scale = _config.LossScale;
    }

    /// <summary>Current loss-scale factor. Multiply the loss by this
    /// before <see cref="GradientTape{T}.ComputeGradients"/>.</summary>
    public float Scale => _scale;

    /// <summary>Pre-backward: scale the loss in-place. Caller passes the
    /// scaled loss to ComputeGradients. After backward, divide each
    /// gradient by <see cref="Scale"/> via <see cref="UnscaleGradients{T}"/>.</summary>
    public Tensor<float> ScaleLoss(Tensor<float> loss)
    {
        var result = new Tensor<float>(loss._shape);
        var src = loss.AsSpan();
        var dst = result.AsWritableSpan();
        for (int i = 0; i < src.Length; i++) dst[i] = src[i] * _scale;
        return result;
    }

    /// <summary>Post-backward: divide every entry in <paramref name="grads"/>
    /// by the current scale. Returns whether any gradient contains NaN/Inf
    /// — caller passes this back to <see cref="Update"/> to drive the
    /// dynamic-scale schedule.</summary>
    public bool UnscaleGradients<T>(IDictionary<Tensor<T>, Tensor<T>> grads)
    {
        var ops = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>();
        T invScale = ops.FromDouble(1.0 / _scale);
        bool foundInfNan = false;
        foreach (var kv in grads)
        {
            var span = kv.Value.AsWritableSpan();
            for (int i = 0; i < span.Length; i++)
            {
                T scaled = ops.Multiply(span[i], invScale);
                if (ops.IsNaN(scaled) || ops.IsInfinity(scaled)) foundInfNan = true;
                span[i] = scaled;
            }
        }
        return foundInfNan;
    }

    /// <summary>Update the scale schedule. Call once per optimizer step.</summary>
    public void Update(bool foundInfNan)
    {
        if (!_config.DynamicLossScale) return;
        if (foundInfNan)
        {
            _scale = MathF.Max(_config.MinLossScale, _scale * _config.BackoffFactor);
            _stepsSinceLastBackoff = 0;
        }
        else
        {
            _stepsSinceLastBackoff++;
            if (_stepsSinceLastBackoff >= _config.GrowthInterval)
            {
                _scale *= _config.GrowthFactor;
                _stepsSinceLastBackoff = 0;
            }
        }
    }
}

/// <summary>Master-weight mirror that lives in fp32 alongside a low-
/// precision (bf16/fp16) compute weight. The optimizer step updates the
/// fp32 master, then casts back to the low-precision storage. This is
/// the standard pattern from Micikevicius et al. for bf16/fp16 training
/// without precision loss across many small updates.</summary>
public sealed class MasterWeights
{
    private readonly Dictionary<object, float[]> _master = new();

    /// <summary>Registers a low-precision compute weight + its initial
    /// fp32 master copy. Pass the same <paramref name="key"/> to
    /// <see cref="GetMaster"/> / <see cref="UpdateMaster"/>.</summary>
    public void Register(object key, ReadOnlySpan<float> initialFp32)
    {
        var copy = new float[initialFp32.Length];
        initialFp32.CopyTo(copy);
        _master[key] = copy;
    }

    /// <summary>Retrieves the fp32 master copy for <paramref name="key"/>.</summary>
    public float[] GetMaster(object key) =>
        _master.TryGetValue(key, out var v) ? v : throw new KeyNotFoundException();

    /// <summary>Applies a gradient (already unscaled) to the fp32 master
    /// via the user's optimizer formula, then writes back into the
    /// low-precision compute weight. Standard SGD/Adam round-trip.</summary>
    public void UpdateMaster(object key, Action<float[]> optimizerUpdate, Action<float[]> writeBack)
    {
        if (!_master.TryGetValue(key, out var master))
            throw new KeyNotFoundException();
        optimizerUpdate(master);
        writeBack(master);
    }

    /// <summary>Removes a master weight. Call when the model parameter
    /// is freed.</summary>
    public void Unregister(object key) => _master.Remove(key);
}
