using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Gpu;

/// <summary>
/// Generic gradient scaler — PyTorch-parity <c>torch.cuda.amp.GradScaler</c>.
/// Prevents fp16 gradient underflow by scaling the loss before backward,
/// unscaling gradients before the optimizer step, and dynamically adjusting
/// the scale factor.
///
/// <para>Generic counterpart to the non-generic <see cref="GradScaler"/>.
/// Use <see cref="GradScaler{T}"/> when T is known at compile time — it
/// avoids <c>INumericOperations</c> virtual dispatch per scale/unscale
/// call and plugs cleanly into generic pipelines that flow any numeric
/// type (float / double / Half / Float8E4M3).</para>
///
/// <para><b>Dynamic scaling loop:</b> start at <see cref="InitialScale"/>,
/// grow by <see cref="GrowthFactor"/> after <see cref="GrowthInterval"/>
/// consecutive overflow-free steps, back off by <see cref="BackoffFactor"/>
/// on any inf/nan gradient. Clamped to [<see cref="MinScale"/>,
/// <see cref="MaxScale"/>]. Static scaling available by setting
/// <see cref="DynamicScaling"/> to false.</para>
/// </summary>
public sealed class GradScaler<T>
{
    private readonly INumericOperations<T> _numOps;
    private double _scale;
    private int _consecutiveGoodSteps;
    private bool _foundInfOrNan;

    /// <summary>Initial scale factor applied to the loss on the first call.</summary>
    public double InitialScale { get; }

    /// <summary>Current scale factor. Changes after each <see cref="Update"/>
    /// when <see cref="DynamicScaling"/> is true.</summary>
    public double Scale => _scale;

    /// <summary>True when the most recent <see cref="Unscale(Tensor{T}[])"/>
    /// / <see cref="UnscaleGradientsAndCheck"/> observed inf or nan.</summary>
    public bool FoundInfOrNan => _foundInfOrNan;

    /// <summary>When false, <see cref="Update"/> leaves the scale unchanged
    /// (static scaling mode). Default true.</summary>
    public bool DynamicScaling { get; set; } = true;

    /// <summary>Steps without overflow before <see cref="Scale"/> is multiplied
    /// by <see cref="GrowthFactor"/>. Default 2000 (PyTorch default).</summary>
    public int GrowthInterval { get; set; } = 2000;

    /// <summary>Multiplicative growth applied on successful streak. Default 2.0.</summary>
    public double GrowthFactor { get; set; } = 2.0;

    /// <summary>Multiplicative backoff applied on overflow. Default 0.5.</summary>
    public double BackoffFactor { get; set; } = 0.5;

    /// <summary>Scale is never dropped below this floor. Default 1.0.</summary>
    public double MinScale { get; set; } = 1.0;

    /// <summary>Scale is never raised above this ceiling. Default 2^24.</summary>
    public double MaxScale { get; set; } = 16_777_216.0;

    /// <summary>
    /// Creates a new generic gradient scaler.
    /// </summary>
    /// <param name="initialScale">Initial scale. Default 65536 (2^16).</param>
    public GradScaler(double initialScale = 65536.0)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        InitialScale = initialScale;
        _scale = initialScale;
    }

    /// <summary>
    /// Resets the scaler to its initial configuration — same scale that was
    /// passed to the constructor (or <paramref name="newInitialScale"/> if
    /// supplied). Clears overflow + streak state. Use when restarting
    /// training or when recovering from a non-recoverable nan.
    /// </summary>
    public void Reset(double? newInitialScale = null)
    {
        _scale = newInitialScale ?? InitialScale;
        _consecutiveGoodSteps = 0;
        _foundInfOrNan = false;
    }

    /// <summary>Scales the supplied loss by the current scale factor. Call
    /// before <c>Backward</c> so the scaled loss flows through gradient
    /// computation and keeps small-magnitude gradients representable.</summary>
    public Tensor<T> ScaleLoss(Tensor<T> loss, IEngine engine)
    {
        if (loss is null) throw new ArgumentNullException(nameof(loss));
        if (engine is null) throw new ArgumentNullException(nameof(engine));
        return engine.TensorMultiplyScalar(loss, _numOps.FromDouble(_scale));
    }

    /// <summary>Scalar overload — scales a single loss value. Useful for
    /// custom loss functions that return a scalar rather than a
    /// <see cref="Tensor{T}"/>.</summary>
    public T ScaleLoss(T loss) => _numOps.Multiply(loss, _numOps.FromDouble(_scale));

    /// <summary>Unscale a single gradient scalar.</summary>
    public T UnscaleGradient(T gradient) => _numOps.Multiply(gradient, _numOps.FromDouble(1.0 / _scale));

    /// <summary>Unscales gradient tensors in place (each tensor replaced with
    /// its unscaled version). Sets <see cref="FoundInfOrNan"/> if any
    /// element overflows. Call before <c>Optimizer.Step</c>.</summary>
    public void Unscale(Tensor<T>[] gradients, IEngine engine)
    {
        if (gradients is null) throw new ArgumentNullException(nameof(gradients));
        if (engine is null) throw new ArgumentNullException(nameof(engine));
        var invScale = _numOps.FromDouble(1.0 / _scale);
        _foundInfOrNan = false;
        for (int i = 0; i < gradients.Length; i++)
        {
            if (gradients[i] is null) continue;
            gradients[i] = engine.TensorMultiplyScalar(gradients[i], invScale);
            if (DetectOverflow(gradients[i]))
            {
                _foundInfOrNan = true;
                return;
            }
        }
    }

    /// <summary>Vector-of-Vector overload: unscale flat gradient arrays that
    /// don't wear tensor shape. Common in custom optimizers that store
    /// gradients as <see cref="Vector{T}"/> per parameter. Unlike the
    /// <see cref="Tensor{T}"/> overload, this path does its scaling via
    /// <see cref="INumericOperations{T}"/> directly — no engine needed,
    /// so we don't thread one through.</summary>
    public void Unscale(Vector<T>[] gradients)
    {
        if (gradients is null) throw new ArgumentNullException(nameof(gradients));
        var invScale = _numOps.FromDouble(1.0 / _scale);
        _foundInfOrNan = false;
        for (int i = 0; i < gradients.Length; i++)
        {
            var g = gradients[i];
            for (int j = 0; j < g.Length; j++)
            {
                T v = _numOps.Multiply(g[j], invScale);
                g[j] = v;
                if (HasOverflow(v))
                {
                    _foundInfOrNan = true;
                    return;
                }
            }
        }
    }

    /// <summary>Single-tensor unscale + overflow check in one call. Returns
    /// true when the step should proceed (no overflow observed), false
    /// when the caller should skip the optimizer step.
    /// <para>Two-pass implementation: the first pass scans for overflow
    /// without mutating <paramref name="grads"/>. If any unscaled value
    /// would be Inf/NaN, the tensor is left untouched and <c>false</c> is
    /// returned — callers that discard the tensor after skipping don't
    /// care, but callers that reuse it get a consistent state rather than
    /// a half-unscaled mess from the previous aborted attempt.</para></summary>
    public bool UnscaleGradientsAndCheck(Tensor<T> grads)
    {
        if (grads is null) throw new ArgumentNullException(nameof(grads));
        var invScale = _numOps.FromDouble(1.0 / _scale);
        var data = grads.GetDataArray();
        // Pass 1: scan — no writes. If we find an overflow, bail before
        // mutating so the tensor stays fully scaled (consistent state).
        for (int i = 0; i < data.Length; i++)
        {
            if (HasOverflow(_numOps.Multiply(data[i], invScale)))
            {
                _foundInfOrNan = true;
                return false;
            }
        }
        // Pass 2: clean scan complete — write the unscaled values back.
        for (int i = 0; i < data.Length; i++)
            data[i] = _numOps.Multiply(data[i], invScale);
        _foundInfOrNan = false;
        return true;
    }

    /// <summary>True iff <paramref name="value"/> is inf or nan.</summary>
    public bool HasOverflow(T value)
    {
        double v = _numOps.ToDouble(value);
        return double.IsInfinity(v) || double.IsNaN(v);
    }

    /// <summary>Scans a tensor for inf or nan. O(n) pass; returns on first hit.</summary>
    public bool DetectOverflow(Tensor<T> grad)
    {
        if (grad is null) return false;
        var data = grad.GetDataArray();
        for (int i = 0; i < data.Length; i++)
            if (HasOverflow(data[i])) return true;
        return false;
    }

    /// <summary>Scans a vector for inf or nan.</summary>
    public bool DetectOverflow(Vector<T> grad)
    {
        if (grad is null) return false;
        for (int i = 0; i < grad.Length; i++)
            if (HasOverflow(grad[i])) return true;
        return false;
    }

    /// <summary>True when the optimizer should run — i.e. no overflow was
    /// observed since the last <see cref="Update"/>.</summary>
    public bool ShouldStep() => !_foundInfOrNan;

    /// <summary>
    /// Adjusts <see cref="Scale"/> based on whether overflow was seen since
    /// the last call. Growth by <see cref="GrowthFactor"/> after
    /// <see cref="GrowthInterval"/> consecutive clean steps; backoff by
    /// <see cref="BackoffFactor"/> on overflow; clamped to
    /// [<see cref="MinScale"/>, <see cref="MaxScale"/>].
    /// </summary>
    public void Update()
    {
        if (!DynamicScaling) { _foundInfOrNan = false; return; }
        if (_foundInfOrNan)
        {
            _scale = Math.Max(_scale * BackoffFactor, MinScale);
            _consecutiveGoodSteps = 0;
        }
        else
        {
            _consecutiveGoodSteps++;
            if (_consecutiveGoodSteps >= GrowthInterval)
            {
                _scale = Math.Min(_scale * GrowthFactor, MaxScale);
                _consecutiveGoodSteps = 0;
            }
        }
        _foundInfOrNan = false;
    }
}
