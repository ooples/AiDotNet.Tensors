// Copyright (c) AiDotNet. All rights reserved.
// Fused activation type enum for IEngine abstraction layer.

namespace AiDotNet.Tensors.Engines;

/// <summary>
/// Specifies the activation function to use in fused operations.
/// </summary>
/// <remarks>
/// <para><b>Purpose:</b></para>
/// <para>
/// This enum provides type-safe selection of activation functions for fused operations
/// in IEngine. The engine implementation decides whether to use GPU fused kernels or
/// CPU sequential operations based on hardware availability and operation size.
/// </para>
/// <para><b>Performance Benefits:</b></para>
/// <para>
/// Fused operations combine multiple steps (e.g., MatMul + Bias + Activation) into a
/// single operation, eliminating intermediate memory allocations and transfers.
/// On GPU, this can provide 20-50% speedup over separate operations.
/// </para>
/// </remarks>
public enum FusedActivationType
{
    /// <summary>
    /// No activation function applied (identity).
    /// Output = Linear transformation only.
    /// </summary>
    None = 0,

    /// <summary>
    /// Rectified Linear Unit: f(x) = max(0, x)
    /// Fast and effective, most common choice for hidden layers.
    /// </summary>
    ReLU = 1,

    /// <summary>
    /// Gaussian Error Linear Unit: f(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    /// Popular in transformer architectures (BERT, GPT).
    /// </summary>
    GELU = 2,

    /// <summary>
    /// Sigmoid: f(x) = 1 / (1 + exp(-x))
    /// Maps output to range (0, 1), useful for binary classification and gates.
    /// </summary>
    Sigmoid = 3,

    /// <summary>
    /// Hyperbolic Tangent: f(x) = tanh(x)
    /// Maps output to range (-1, 1), useful for hidden layers in RNNs.
    /// </summary>
    Tanh = 4,

    /// <summary>
    /// Leaky ReLU: f(x) = x if x > 0, else alpha * x
    /// Prevents "dying ReLU" problem by allowing small gradients for negative inputs.
    /// </summary>
    LeakyReLU = 5,

    /// <summary>
    /// Swish/SiLU: f(x) = x * sigmoid(x)
    /// Self-gated activation, often outperforms ReLU in deep networks.
    /// </summary>
    Swish = 6,

    /// <summary>
    /// Softmax: exp(x_i) / sum(exp(x_j))
    /// Normalizes outputs to probability distribution, used for classification.
    /// </summary>
    Softmax = 7,

    /// <summary>
    /// Exponential Linear Unit: f(x) = x if x > 0, else alpha * (exp(x) - 1)
    /// Smoother version of ReLU that avoids dead neurons.
    /// </summary>
    ELU = 8,

    /// <summary>
    /// Scaled Exponential Linear Unit: f(x) = scale * (x if x > 0, else alpha * (exp(x) - 1))
    /// Self-normalizing activation for deep networks.
    /// </summary>
    SELU = 9,

    /// <summary>
    /// Softplus: f(x) = log(1 + exp(x))
    /// Smooth approximation of ReLU.
    /// </summary>
    Softplus = 10,

    /// <summary>
    /// Mish: f(x) = x * tanh(softplus(x)) = x * tanh(log(1 + exp(x)))
    /// Self-regularized activation that often outperforms Swish.
    /// </summary>
    Mish = 11,

    /// <summary>
    /// HardSwish: f(x) = x * clip((x + 3) / 6, 0, 1)
    /// Efficient approximation of Swish for mobile/edge deployment.
    /// </summary>
    HardSwish = 12,

    /// <summary>
    /// HardSigmoid: f(x) = clip((x + 3) / 6, 0, 1)
    /// Efficient approximation of Sigmoid for mobile/edge deployment.
    /// </summary>
    HardSigmoid = 13,

    /// <summary>
    /// HardTanh: f(x) = clip(x, -1, 1)
    /// Efficient approximation of Tanh with bounded outputs.
    /// </summary>
    HardTanh = 14,

    /// <summary>
    /// ReLU6: f(x) = min(max(0, x), 6)
    /// ReLU clamped at 6; common in mobile/quantized networks (MobileNet).
    /// </summary>
    ReLU6 = 15,

    /// <summary>
    /// SoftSign: f(x) = x / (1 + |x|)
    /// Smooth, bounded (-1, 1) alternative to Tanh with polynomial tails.
    /// </summary>
    SoftSign = 16,

    /// <summary>
    /// CELU: f(x) = max(0, x) + min(0, alpha * (exp(x / alpha) - 1))
    /// Continuously-differentiable ELU variant (Barron, 2017). Parametric (alpha,
    /// default 1) — supply via <see cref="FusedActivationParams.Alpha"/>.
    /// </summary>
    CELU = 17,

    /// <summary>
    /// ThresholdedReLU: f(x) = x if x &gt; theta, else 0
    /// Parametric (theta, default 1) — supply via <see cref="FusedActivationParams.Theta"/>.
    /// </summary>
    ThresholdedReLU = 18,

    /// <summary>
    /// ScaledTanh: f(x) = alpha * tanh(beta * x)
    /// Parametric (alpha/beta, defaults 1/1) — supply via
    /// <see cref="FusedActivationParams.Alpha"/> / <see cref="FusedActivationParams.Beta"/>.
    /// </summary>
    ScaledTanh = 19
}

/// <summary>
/// Optional scalar parameters for parametric fused activations. Omit (null) to use
/// each activation's canonical default. Only the field(s) relevant to the chosen
/// <see cref="FusedActivationType"/> are read.
/// </summary>
/// <remarks>
/// A class with nullable init properties (not a record struct) so a missing field
/// can be distinguished from an explicit value and resolved to the per-activation
/// default. Lets <c>FusedLinear</c> / <c>MlpForward</c> / the activation epilogue
/// fuse LeakyReLU (any slope), ELU / CELU (any alpha), ThresholdedReLU (theta),
/// and ScaledTanh (alpha, beta) — not only the hardcoded default value.
/// </remarks>
public sealed class FusedActivationParams
{
    /// <summary>LeakyReLU slope, ELU/CELU/ScaledTanh alpha. Default: LeakyReLU 0.01, others 1.</summary>
    public float? Alpha { get; init; }
    /// <summary>ScaledTanh inner scale beta. Default 1.</summary>
    public float? Beta { get; init; }
    /// <summary>ThresholdedReLU threshold. Default 1.</summary>
    public float? Theta { get; init; }
}
