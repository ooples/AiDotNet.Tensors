namespace AiDotNet.Tensors.Engines.Gpu;

/// <summary>
/// Gradient scaler for mixed precision training, equivalent to PyTorch's
/// <c>torch.cuda.amp.GradScaler</c>. Prevents fp16 gradient underflow by
/// scaling the loss before backward, unscaling gradients before optimizer step,
/// and dynamically adjusting the scale factor.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> When training with fp16 (half precision), small gradient
/// values can become zero (underflow) because fp16 can't represent very small numbers.
/// GradScaler multiplies the loss by a large number before computing gradients, which
/// keeps them in fp16's representable range. Before the optimizer updates weights, the
/// gradients are divided back by the same scale factor.</para>
///
/// <para><b>Usage:</b></para>
/// <code>
/// var scaler = new GradScaler();
/// using var autocast = new AutocastScope(PrecisionMode.Float16);
///
/// var loss = model.Forward(input);
/// var scaledLoss = scaler.ScaleLoss(loss, engine); // Multiply loss by scale factor
/// tape.Backward(scaledLoss);
///
/// scaler.Unscale(gradients, engine);               // Divide gradients by scale factor
/// if (scaler.ShouldStep())                         // Check for inf/nan
///     optimizer.Step(parameters, gradients);     // Only step if gradients are valid
/// scaler.Update();                              // Adjust scale factor for next iteration
/// </code>
/// </remarks>
public sealed class GradScaler
{
    private double _scale;
    private double _growthFactor;
    private double _backoffFactor;
    private int _growthInterval;
    private int _consecutiveGoodSteps;
    private bool _foundInfOrNan;

    /// <summary>
    /// Gets the current scale factor.
    /// </summary>
    public double Scale => _scale;

    /// <summary>
    /// Gets whether inf or nan gradients were detected in the last unscale call.
    /// </summary>
    public bool FoundInfOrNan => _foundInfOrNan;

    /// <summary>
    /// Creates a new GradScaler with the specified parameters.
    /// </summary>
    /// <param name="initialScale">Initial scale factor (default 65536 = 2^16).</param>
    /// <param name="growthFactor">Factor to increase scale by after successful steps (default 2.0).</param>
    /// <param name="backoffFactor">Factor to decrease scale by after inf/nan (default 0.5).</param>
    /// <param name="growthInterval">Number of consecutive good steps before increasing scale (default 2000).</param>
    public GradScaler(
        double initialScale = 65536.0,
        double growthFactor = 2.0,
        double backoffFactor = 0.5,
        int growthInterval = 2000)
    {
        _scale = initialScale;
        _growthFactor = growthFactor;
        _backoffFactor = backoffFactor;
        _growthInterval = growthInterval;
    }

    /// <summary>
    /// Scales a loss tensor by the current scale factor.
    /// Call this before backward() to prevent fp16 gradient underflow.
    /// </summary>
    public LinearAlgebra.Tensor<T> ScaleLoss<T>(LinearAlgebra.Tensor<T> loss, IEngine engine)
    {
        var numOps = Helpers.MathHelper.GetNumericOperations<T>();
        return engine.TensorMultiplyScalar(loss, numOps.FromDouble(_scale));
    }

    /// <summary>
    /// Unscales gradients by dividing by the current scale factor.
    /// Also checks for inf/nan values in the gradients.
    /// Call this before optimizer.Step().
    /// </summary>
    public void Unscale<T>(LinearAlgebra.Tensor<T>[] gradients, IEngine engine)
    {
        var numOps = Helpers.MathHelper.GetNumericOperations<T>();
        var invScale = numOps.FromDouble(1.0 / _scale);
        _foundInfOrNan = false;

        for (int i = 0; i < gradients.Length; i++)
        {
            if (gradients[i] is null) continue;
            gradients[i] = engine.TensorMultiplyScalar(gradients[i], invScale);

            // Check for inf/nan in unscaled gradients
            var data = gradients[i].GetDataArray();
            int len = gradients[i].Length; // Use logical length, not array length (ArrayPool may over-allocate)
            for (int j = 0; j < len; j++)
            {
                double val = numOps.ToDouble(data[j]);
                if (double.IsInfinity(val) || double.IsNaN(val))
                {
                    _foundInfOrNan = true;
                    return;
                }
            }
        }
    }

    /// <summary>
    /// Checks if the optimizer should step (no inf/nan detected).
    /// Returns true if gradients are valid and the optimizer should proceed.
    /// </summary>
    public bool ShouldStep()
    {
        return !_foundInfOrNan;
    }

    /// <summary>
    /// Updates the scale factor based on whether inf/nan was found.
    /// Call this after every training step.
    /// </summary>
    public void Update()
    {
        if (_foundInfOrNan)
        {
            _scale = Math.Max(_scale * _backoffFactor, 1.0);
            _consecutiveGoodSteps = 0;
        }
        else
        {
            _consecutiveGoodSteps++;
            if (_consecutiveGoodSteps >= _growthInterval)
            {
                _scale *= _growthFactor;
                _consecutiveGoodSteps = 0;
            }
        }

        _foundInfOrNan = false;
    }
}
