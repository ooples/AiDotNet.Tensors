using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines;

/// <summary>
/// Differentiable model-level ops: structured prediction (linear-chain CRF), hyperbolic layers, quantization-aware
/// training, spiking surrogates, mixture-of-experts routing, diffusion time embeddings, soft (differentiable) trees and
/// instance-based learners, and standalone activations whose kernels the fused-activation path already carries.
/// </summary>
public partial interface IEngine
{
    /// <summary>
    /// Log-partition function log Z of a linear-chain conditional random field, by the forward algorithm in log space
    /// (Lafferty et al. 2001; the normalizer of <c>torchcrf.CRF</c> / <c>tf_addons.text.crf_log_norm</c>).
    /// </summary>
    /// <remarks>
    /// <para>
    /// log Z = log Σ_y exp(start[y₀] + Σ_t emissions[t, y_t] + Σ_{t&gt;0} transitions[y_{t-1}, y_t] + end[y_{S-1}]).
    /// The backward is exact: ∂ log Z / ∂ emissions[t, j] is the posterior marginal p(y_t = j), ∂ / ∂ transitions[i, j]
    /// the expected count of the i → j transition, and ∂ / ∂ start / end the first / last tag marginals
    /// (forward-backward).
    /// </para>
    /// </remarks>
    /// <param name="emissions">Per-step tag scores, <c>[seqLen, numTags]</c> or batched <c>[batch, seqLen, numTags]</c>.</param>
    /// <param name="transitions">Transition scores <c>[numTags, numTags]</c>, indexed [from, to].</param>
    /// <param name="startScores">Optional scores for the first tag, <c>[numTags]</c>; zero when null.</param>
    /// <param name="endScores">Optional scores for the last tag, <c>[numTags]</c>; zero when null.</param>
    /// <returns><c>[1]</c> for unbatched emissions, <c>[batch]</c> for batched ones.</returns>
    Tensor<T> TensorCrfForward<T>(Tensor<T> emissions, Tensor<T> transitions, Tensor<T>? startScores = null, Tensor<T>? endScores = null);

    /// <summary>
    /// Hyperbolic (Poincaré-ball) linear layer of Ganea et al. 2018, "Hyperbolic Neural Networks" (the
    /// <c>geoopt</c> / <c>hyptorch</c> <c>HypLinear</c>): y = proj(exp₀(W · log₀(x)) ⊕_c b).
    /// </summary>
    /// <remarks>
    /// <para>
    /// log₀(x) = artanh(√c‖x‖) x / (√c‖x‖) and exp₀(v) = tanh(√c‖v‖) v / (√c‖v‖) are the logarithmic and exponential
    /// maps at the origin, ⊕_c is Möbius addition and proj rescales a row whose norm reaches (1 - 1e-5)/√c back
    /// inside the ball — the same formulas and boundary as <see cref="PoincareExpMap{T}"/>, <see cref="PoincareLogMap{T}"/>,
    /// <see cref="MobiusAdd{T}"/> and <see cref="PoincareProject{T}"/>. It is composed from recorded tensor ops, so it
    /// is differentiable to any order and runs wherever those ops run.
    /// </para>
    /// </remarks>
    /// <param name="input">Points in the ball, <c>[..., inFeatures]</c>.</param>
    /// <param name="weights">Weight matrix <c>[outFeatures, inFeatures]</c> (the <c>nn.Linear</c> layout).</param>
    /// <param name="bias">Optional bias point in the ball, <c>[outFeatures]</c>, added by Möbius addition.</param>
    /// <param name="curvature">Ball curvature; its magnitude c is used (−1 and 1 both mean the unit ball). Must be non-zero.</param>
    /// <returns>Points in the ball, <c>[..., outFeatures]</c>.</returns>
    Tensor<T> TensorHyperbolicLinear<T>(Tensor<T> input, Tensor<T> weights, Tensor<T>? bias = null, double curvature = 1.0);

    /// <summary>
    /// Per-tensor affine fake quantization: quantize to <paramref name="numBits"/>-bit integers and dequantize
    /// (<c>torch.fake_quantize_per_tensor_affine</c>).
    /// </summary>
    /// <remarks>
    /// <para>
    /// q = clamp(round(x / scale) + zeroPoint, qMin, qMax) with round-half-to-even, and y = (q - zeroPoint)·scale. The
    /// range is [-2^(b-1), 2^(b-1) - 1] when <paramref name="symmetric"/> and [0, 2^b - 1] otherwise. The gradient is
    /// the straight-through estimator restricted to the representable range, as in PyTorch: dy/dx = 1 where
    /// qMin ≤ round(x / scale) + zeroPoint ≤ qMax and 0 where the value was clamped.
    /// </para>
    /// <para>
    /// A null or zero <paramref name="scale"/> is derived from the data: max|x| / qMax when symmetric, (max - min) /
    /// (qMax - qMin) otherwise (1 when that is zero). A null <paramref name="zeroPoint"/> is 0, except for an asymmetric
    /// derived scale, where it is round(qMin - min / scale) clamped to the range so that min maps to qMin.
    /// </para>
    /// </remarks>
    /// <param name="tensor">The values to fake-quantize.</param>
    /// <param name="numBits">Integer width, 1 to 30.</param>
    /// <param name="scale">Quantization step; null or 0 derives it from the data.</param>
    /// <param name="zeroPoint">Integer offset; null chooses it as described above.</param>
    /// <param name="symmetric">Signed symmetric range when true, unsigned asymmetric range when false.</param>
    Tensor<T> TensorFakeQuantize<T>(Tensor<T> tensor, int numBits = 8, double? scale = null, int? zeroPoint = null, bool symmetric = true);

    /// <summary>
    /// Spiking-neuron firing with a surrogate gradient (Zenke &amp; Ganguli 2018, "SuperSpike"; the <c>snntorch</c>
    /// <c>surrogate.sigmoid</c> spike): the forward is the Heaviside step H(v - threshold), 1 where
    /// v &gt; threshold, and the backward replaces its zero-almost-everywhere derivative by
    /// β·σ(β(v - threshold))·(1 - σ(β(v - threshold))).
    /// </summary>
    /// <param name="membranePotential">Membrane potentials v.</param>
    /// <param name="threshold">Firing threshold.</param>
    /// <param name="surrogateBeta">Sharpness β of the sigmoid whose derivative stands in for the step's.</param>
    Tensor<T> TensorSurrogateSpike<T>(Tensor<T> membranePotential, double threshold = 1.0, double surrogateBeta = 1.0);

    /// <summary>
    /// Softmax over the <paramref name="k"/> largest scores of each row of the last axis, zero elsewhere — the
    /// mixture-of-experts top-k router (Shazeer et al. 2017; <c>softmax(topk(x).values)</c> scattered back).
    /// </summary>
    /// <remarks>
    /// Ties are broken toward the lower index. The gradient flows through the selected entries only, as the softmax
    /// Jacobian of the selection; the selection itself is piecewise constant.
    /// </remarks>
    /// <param name="scores">Router logits <c>[..., numExperts]</c>.</param>
    /// <param name="k">Number of entries kept per row, 1 to the last axis' length.</param>
    Tensor<T> TensorTopKSoftmax<T>(Tensor<T> scores, int k);

    /// <summary>
    /// Sinusoidal timestep embedding for diffusion models (Ho et al. 2020, after Vaswani et al. 2017; the
    /// <c>diffusers</c> <c>get_timestep_embedding</c> layout with sine first):
    /// out[b, i] = sin(t_b·f_i) and out[b, i + half] = cos(t_b·f_i) for i &lt; half = dim / 2, with
    /// f_i = exp(-ln(10000)·i / max(half - 1, 1)). An odd <paramref name="embeddingDim"/> leaves the last column zero.
    /// </summary>
    /// <remarks>Differentiable in the timesteps: d out[b, i] / d t_b = f_i·cos(t_b·f_i) and -f_i·sin(t_b·f_i).</remarks>
    /// <param name="timesteps">Timesteps <c>[batch]</c> or <c>[batch, 1]</c>.</param>
    /// <param name="embeddingDim">Embedding width (at least 1).</param>
    /// <returns><c>[batch, embeddingDim]</c>.</returns>
    Tensor<T> TensorSinusoidalTimeEmbedding<T>(Tensor<T> timesteps, int embeddingDim);

    /// <summary>
    /// A soft (sigmoid-gated) decision-tree split, the differentiable node of soft decision trees (Irsoy et al. 2012;
    /// Frosst &amp; Hinton 2017): p = σ((threshold - x[featureIndex]) / temperature) and
    /// out = p·leftValue + (1 - p)·rightValue.
    /// </summary>
    /// <param name="input">Feature values; <paramref name="featureIndex"/> indexes their row-major flattening.</param>
    /// <param name="leftValue">Value taken when the feature is below the threshold.</param>
    /// <param name="rightValue">Value taken when the feature is above the threshold; same shape as <paramref name="leftValue"/>.</param>
    /// <param name="featureIndex">Index of the split feature.</param>
    /// <param name="threshold">Split threshold.</param>
    /// <param name="temperature">Split softness (positive); smaller is sharper.</param>
    /// <returns>A tensor shaped like <paramref name="leftValue"/>.</returns>
    Tensor<T> TensorSoftSplit<T>(Tensor<T> input, Tensor<T> leftValue, Tensor<T> rightValue, int featureIndex, double threshold, double temperature = 1.0);

    /// <summary>
    /// Soft k-nearest-neighbours (Goldberger et al. 2004, neighbourhood components; Plötz &amp; Roth 2018): attention
    /// over every support vector, w = softmax(-‖x - s_i‖² / temperature), out = Σ_i w_i·labels[i].
    /// </summary>
    /// <param name="input">Query <c>[features]</c> or a batch of queries <c>[batch, features]</c>.</param>
    /// <param name="supportVectors">Support points <c>[n, features]</c> (or <c>[n]</c> for one feature).</param>
    /// <param name="labels">Labels <c>[n]</c> or <c>[n, outputs]</c>.</param>
    /// <param name="temperature">Softmax temperature (positive).</param>
    /// <returns><c>[outputs]</c> for one query (<c>[1]</c> for <c>[n]</c> labels), <c>[batch, outputs]</c> for a batch.</returns>
    Tensor<T> TensorSoftKnn<T>(Tensor<T> input, Tensor<T> supportVectors, Tensor<T> labels, double temperature = 1.0);

    /// <summary>
    /// Soft locally-weighted (Nadaraya–Watson kernel) regression with a Gaussian kernel of the given bandwidth:
    /// w = softmax(-‖x - xTrain_i‖² / bandwidth), out = Σ_i w_i·yTrain[i]. Same computation and shapes as
    /// <see cref="TensorSoftKnn{T}"/> with the bandwidth in the temperature's place.
    /// </summary>
    /// <param name="input">Query <c>[features]</c> or <c>[batch, features]</c>.</param>
    /// <param name="xTrain">Training inputs <c>[n, features]</c>.</param>
    /// <param name="yTrain">Training targets <c>[n]</c> or <c>[n, outputs]</c>.</param>
    /// <param name="bandwidth">Kernel bandwidth (positive).</param>
    Tensor<T> TensorSoftLocallyWeighted<T>(Tensor<T> input, Tensor<T> xTrain, Tensor<T> yTrain, double bandwidth = 1.0);

    /// <summary>LiSHT, x·tanh(x) (Roy et al. 2019, "LiSHT: Non-Parametric Linearly Scaled Hyperbolic Tangent").</summary>
    Tensor<T> TensorLisht<T>(Tensor<T> tensor);

    /// <summary>Bent identity, (√(x² + 1) - 1) / 2 + x.</summary>
    Tensor<T> TensorBentIdentity<T>(Tensor<T> tensor);

    /// <summary>Gaussian activation, exp(-x²).</summary>
    Tensor<T> TensorGaussian<T>(Tensor<T> tensor);

    /// <summary>
    /// Scaled hyperbolic tangent, α·tanh(β·x) (LeCun et al. 1998, "Efficient BackProp", which recommends
    /// α = 1.7159, β = 2/3). (1 - e^(-βx)) / (1 + e^(-βx)) is the special case α = 1 with β halved.
    /// </summary>
    Tensor<T> TensorScaledTanh<T>(Tensor<T> tensor, double alpha = 1.0, double beta = 1.0);

    /// <summary>Thresholded ReLU, x where x &gt; threshold and 0 elsewhere (Konda et al. 2015; Keras <c>ThresholdedReLU</c>).</summary>
    Tensor<T> TensorThresholdedRelu<T>(Tensor<T> tensor, double threshold = 1.0);

    /// <summary>Inverse square root unit, x / √(1 + α·x²) (Carlile et al. 2017, "Improving Deep Learning by Inverse Square Root Linear Units").</summary>
    Tensor<T> TensorIsru<T>(Tensor<T> tensor, double alpha = 1.0);

    /// <summary>Square radial basis function (Gaussian RBF) activation, exp(-β·x²).</summary>
    Tensor<T> TensorSqrbf<T>(Tensor<T> tensor, double beta = 1.0);

    /// <summary>log(softmin(x)) = log_softmax(-x) along <paramref name="axis"/> (<c>torch.log_softmax(-x, dim)</c>).</summary>
    Tensor<T> TensorLogSoftmin<T>(Tensor<T> tensor, int axis = -1);
}
