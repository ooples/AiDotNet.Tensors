using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines;

/// <summary>
/// Model-level ops (see the matching <see cref="IEngine"/> members): the CRF log-partition, fake quantization, surrogate
/// spikes, top-k routing and time embeddings compute in double precision and record their own backward; the hyperbolic
/// linear layer and the soft tree / instance-based ops are compositions of recorded engine ops; the activations run the
/// fused-activation kernels of <see cref="CpuFusedOperations"/>.
/// </summary>
public partial class CpuEngine
{
    #region Activations (fused-activation kernels)

    // y = f(x) with the fused-activation path's kernel f and its exact derivative f'.
    private static Tensor<T> FusedKernelUnary<T>(string opName, Tensor<T> tensor, FusedActivationType type,
        FusedActivationParams? parameters, Func<IEngine, Tensor<T>> replay)
    {
        var f = CpuFusedOperations.GetDoubleActivation(type, parameters);
        var derivative = CpuFusedOperations.GetDoubleActivationDerivative(type, parameters);
        return SpecialUnary(opName, tensor, f, (x, _) => derivative(x), replay, FusedKernelRecordedDerivative<T>(type, parameters));
    }

    /// <inheritdoc/>
    public virtual Tensor<T> TensorLisht<T>(Tensor<T> tensor)
        => FusedKernelUnary("TensorLisht", tensor, FusedActivationType.LiSHT, null, e => e.TensorLisht(tensor));

    /// <inheritdoc/>
    public virtual Tensor<T> TensorBentIdentity<T>(Tensor<T> tensor)
        => FusedKernelUnary("TensorBentIdentity", tensor, FusedActivationType.BentIdentity, null, e => e.TensorBentIdentity(tensor));

    /// <inheritdoc/>
    public virtual Tensor<T> TensorGaussian<T>(Tensor<T> tensor)
        => FusedKernelUnary("TensorGaussian", tensor, FusedActivationType.Gaussian, null, e => e.TensorGaussian(tensor));

    /// <inheritdoc/>
    public virtual Tensor<T> TensorScaledTanh<T>(Tensor<T> tensor, double alpha = 1.0, double beta = 1.0)
        => FusedKernelUnary("TensorScaledTanh", tensor, FusedActivationType.ScaledTanh,
            new FusedActivationParams { Alpha = (float)alpha, Beta = (float)beta }, e => e.TensorScaledTanh(tensor, alpha, beta));

    /// <inheritdoc/>
    public virtual Tensor<T> TensorThresholdedRelu<T>(Tensor<T> tensor, double threshold = 1.0)
        => FusedKernelUnary("TensorThresholdedRelu", tensor, FusedActivationType.ThresholdedReLU,
            new FusedActivationParams { Theta = (float)threshold }, e => e.TensorThresholdedRelu(tensor, threshold));

    /// <inheritdoc/>
    public virtual Tensor<T> TensorIsru<T>(Tensor<T> tensor, double alpha = 1.0)
    {
        if (alpha < 0 || double.IsNaN(alpha)) throw new ArgumentOutOfRangeException(nameof(alpha), "ISRU alpha must be non-negative.");
        return FusedKernelUnary("TensorIsru", tensor, FusedActivationType.ISRU,
            new FusedActivationParams { Alpha = (float)alpha }, e => e.TensorIsru(tensor, alpha));
    }

    /// <inheritdoc/>
    public virtual Tensor<T> TensorSqrbf<T>(Tensor<T> tensor, double beta = 1.0)
        => FusedKernelUnary("TensorSqrbf", tensor, FusedActivationType.SQRBF,
            new FusedActivationParams { Beta = (float)beta }, e => e.TensorSqrbf(tensor, beta));

    /// <inheritdoc/>
    public virtual Tensor<T> TensorLogSoftmin<T>(Tensor<T> tensor, int axis = -1)
    {
        if (tensor == null) throw new ArgumentNullException(nameof(tensor));
        return TensorLogSoftmax(TensorNegate(tensor), NormalizeModelOpAxis(axis, tensor.Rank));
    }

    private static int NormalizeModelOpAxis(int axis, int rank)
    {
        int normalized = axis < 0 ? axis + rank : axis;
        if (normalized < 0 || normalized >= rank)
            throw new ArgumentOutOfRangeException(nameof(axis), $"Axis {axis} is out of range for a rank-{rank} tensor.");
        return normalized;
    }

    #endregion

    // Under an inference trace: one opaque node over the op's tensor operands, whose replay re-runs the op on the
    // plan's engine (the device kernels on a GPU engine). Training traces record through TryRecordCompiledOp.
    private protected bool TryCaptureModelOpInference<T>(string opName, Tensor<T>?[] inputs, Func<IEngine, Tensor<T>> forward,
        [System.Diagnostics.CodeAnalysis.NotNullWhen(true)] out Tensor<T>? captured)
    {
        captured = null;
        if (!GraphMode.IsInferenceTrace) return false;
        var operands = new List<Tensor<T>>(inputs.Length);
        foreach (var input in inputs)
            if (input is not null && !operands.Contains(input)) operands.Add(input);
        captured = CaptureInferenceKernel(operands.ToArray(), forward, opName);
        return true;
    }

    #region Spiking, quantization, routing, embeddings

    /// <inheritdoc/>
    public virtual Tensor<T> TensorSurrogateSpike<T>(Tensor<T> membranePotential, double threshold = 1.0, double surrogateBeta = 1.0)
    {
        if (double.IsNaN(surrogateBeta)) throw new ArgumentOutOfRangeException(nameof(surrogateBeta), "surrogateBeta must be a number.");
        return SpecialUnary("TensorSurrogateSpike", membranePotential, v => v > threshold ? 1.0 : 0.0,
            (v, _) =>
            {
                double sigmoid = 1.0 / (1.0 + Math.Exp(-surrogateBeta * (v - threshold)));
                return surrogateBeta * sigmoid * (1.0 - sigmoid);
            },
            e => e.TensorSurrogateSpike(membranePotential, threshold, surrogateBeta),
            (e, v, _) => DSurrogateSpike(e, v, threshold, surrogateBeta));
    }

    /// <inheritdoc/>
    public virtual Tensor<T> TensorFakeQuantize<T>(Tensor<T> tensor, int numBits = 8, double? scale = null, int? zeroPoint = null, bool symmetric = true)
    {
        if (tensor == null) throw new ArgumentNullException(nameof(tensor));
        if (numBits < 1 || numBits > 30) throw new ArgumentOutOfRangeException(nameof(numBits), "numBits must be between 1 and 30.");
        if (scale is { } given && (double.IsNaN(given) || double.IsInfinity(given) || given < 0))
            throw new ArgumentOutOfRangeException(nameof(scale), "scale must be a finite non-negative number (0 or null derives it).");
        if (TryRecordCompiledOp("TensorFakeQuantize", new Tensor<T>?[] { tensor },
                e => e.TensorFakeQuantize(tensor, numBits, scale, zeroPoint, symmetric), out var compiledNode))
            return compiledNode;
        if (TryCaptureModelOpInference("TensorFakeQuantize", new Tensor<T>?[] { tensor }, e => e.TensorFakeQuantize(tensor, numBits, scale, zeroPoint, symmetric), out var captured)) return captured;
        GraphMode.ThrowIfActiveUnsupported(GraphCaptureLimitation.HostBoundary);

        double qMin = symmetric ? -(1L << (numBits - 1)) : 0;
        double qMax = symmetric ? (1L << (numBits - 1)) - 1 : (1L << numBits) - 1;
        var x = RoIAlignPlan.ToDoubles(tensor);
        double step;
        double zp;
        if (scale is { } fixedScale && fixedScale != 0)
        {
            step = fixedScale;
            zp = zeroPoint ?? 0;
        }
        else
        {
            double min = 0, max = 0;
            if (x.Length > 0)
            {
                min = double.PositiveInfinity;
                max = double.NegativeInfinity;
                for (int i = 0; i < x.Length; i++)
                {
                    if (x[i] < min) min = x[i];
                    if (x[i] > max) max = x[i];
                }
            }
            step = symmetric ? Math.Max(Math.Abs(min), Math.Abs(max)) / qMax : (max - min) / (qMax - qMin);
            if (step == 0 || double.IsNaN(step) || double.IsInfinity(step)) step = 1;
            zp = zeroPoint ?? (symmetric ? 0 : Math.Min(Math.Max(Math.Round(qMin - min / step), qMin), qMax));
        }

        var y = new double[x.Length];
        var pass = new double[x.Length];
        for (int i = 0; i < x.Length; i++)
        {
            double q = Math.Round(x[i] / step) + zp;
            pass[i] = q >= qMin && q <= qMax ? 1.0 : 0.0;
            double clamped = q < qMin ? qMin : q > qMax ? qMax : q;
            y[i] = (clamped - zp) * step;
        }

        var shape = (int[])tensor._shape.Clone();
        var result = RoIAlignPlan.FromDoubles<T>(y, shape);
        DifferentiableOps.RecordUnary("TensorFakeQuantize", result, tensor, BackwardFunctions<T>.MaskedPassThroughBackward,
            new object[] { RoIAlignPlan.FromDoubles<T>(pass, (int[])shape.Clone()) });
        return result;
    }

    /// <inheritdoc/>
    public virtual Tensor<T> TensorTopKSoftmax<T>(Tensor<T> scores, int k)
    {
        if (scores == null) throw new ArgumentNullException(nameof(scores));
        if (scores.Rank == 0) throw new ArgumentException("TensorTopKSoftmax needs at least one axis.", nameof(scores));
        int width = scores._shape[scores.Rank - 1];
        if (k < 1 || k > width)
            throw new ArgumentOutOfRangeException(nameof(k), $"k must be between 1 and the last axis' length {width}; got {k}.");
        if (TryRecordCompiledOp("TensorTopKSoftmax", new Tensor<T>?[] { scores }, e => e.TensorTopKSoftmax(scores, k), out var compiledNode))
            return compiledNode;
        if (TryCaptureModelOpInference("TensorTopKSoftmax", new Tensor<T>?[] { scores }, e => e.TensorTopKSoftmax(scores, k), out var captured)) return captured;
        GraphMode.ThrowIfActiveUnsupported(GraphCaptureLimitation.HostBoundary);

        var x = RoIAlignPlan.ToDoubles(scores);
        var y = new double[x.Length];
        int rows = width == 0 ? 0 : x.Length / width;
        var order = new int[width];
        var keys = new double[width];
        for (int r = 0; r < rows; r++)
        {
            int offset = r * width;
            for (int i = 0; i < width; i++)
            {
                order[i] = i;
                keys[i] = x[offset + i];
            }
            // Largest first; equal scores keep the lower index first.
            Array.Sort(order, (a, b) =>
            {
                int byValue = keys[b].CompareTo(keys[a]);
                return byValue != 0 ? byValue : a.CompareTo(b);
            });
            double max = keys[order[0]];
            double sum = 0;
            for (int i = 0; i < k; i++) sum += Math.Exp(keys[order[i]] - max);
            for (int i = 0; i < k; i++) y[offset + order[i]] = Math.Exp(keys[order[i]] - max) / sum;
        }

        var result = RoIAlignPlan.FromDoubles<T>(y, (int[])scores._shape.Clone());
        DifferentiableOps.RecordUnary("TensorTopKSoftmax", result, scores, BackwardFunctions<T>.TopKSoftmaxBackward);
        return result;
    }

    /// <inheritdoc/>
    public virtual Tensor<T> TensorSinusoidalTimeEmbedding<T>(Tensor<T> timesteps, int embeddingDim)
    {
        if (timesteps == null) throw new ArgumentNullException(nameof(timesteps));
        if (embeddingDim < 1) throw new ArgumentOutOfRangeException(nameof(embeddingDim), "embeddingDim must be at least 1.");
        if (timesteps.Rank != 1 && !(timesteps.Rank == 2 && timesteps._shape[1] == 1))
            throw new ArgumentException(
                $"timesteps must be [batch] or [batch, 1]; got [{string.Join(", ", timesteps._shape)}].", nameof(timesteps));
        if (TryRecordCompiledOp("TensorSinusoidalTimeEmbedding", new Tensor<T>?[] { timesteps },
                e => e.TensorSinusoidalTimeEmbedding(timesteps, embeddingDim), out var compiledNode))
            return compiledNode;
        if (TryCaptureModelOpInference("TensorSinusoidalTimeEmbedding", new Tensor<T>?[] { timesteps }, e => e.TensorSinusoidalTimeEmbedding(timesteps, embeddingDim), out var captured)) return captured;
        GraphMode.ThrowIfActiveUnsupported(GraphCaptureLimitation.HostBoundary);

        int batch = timesteps._shape[0];
        int half = embeddingDim / 2;
        double logTimescale = Math.Log(10000.0) / Math.Max(half - 1, 1);
        var freq = new double[half];
        for (int i = 0; i < half; i++) freq[i] = Math.Exp(-logTimescale * i);
        var t = RoIAlignPlan.ToDoubles(timesteps);
        var y = new double[batch * embeddingDim];
        for (int b = 0; b < batch; b++)
        {
            for (int i = 0; i < half; i++)
            {
                double angle = t[b] * freq[i];
                y[b * embeddingDim + i] = Math.Sin(angle);
                y[b * embeddingDim + i + half] = Math.Cos(angle);
            }
        }

        var result = RoIAlignPlan.FromDoubles<T>(y, new[] { batch, embeddingDim });
        DifferentiableOps.RecordUnary("TensorSinusoidalTimeEmbedding", result, timesteps,
            BackwardFunctions<T>.SinusoidalTimeEmbeddingBackwardWithSecondOrder, new object[] { freq, embeddingDim });
        return result;
    }

    #endregion

    #region Linear-chain CRF

    /// <inheritdoc/>
    public virtual Tensor<T> TensorCrfForward<T>(Tensor<T> emissions, Tensor<T> transitions, Tensor<T>? startScores = null, Tensor<T>? endScores = null)
    {
        if (emissions == null) throw new ArgumentNullException(nameof(emissions));
        if (transitions == null) throw new ArgumentNullException(nameof(transitions));
        if (emissions.Rank != 2 && emissions.Rank != 3)
            throw new ArgumentException(
                $"emissions must be [seqLen, numTags] or [batch, seqLen, numTags]; got [{string.Join(", ", emissions._shape)}].", nameof(emissions));
        bool batched = emissions.Rank == 3;
        int batch = batched ? emissions._shape[0] : 1;
        int seqLen = emissions._shape[emissions.Rank - 2];
        int numTags = emissions._shape[emissions.Rank - 1];
        if (seqLen < 1 || numTags < 1) throw new ArgumentException("emissions needs at least one step and one tag.", nameof(emissions));
        if (transitions.Rank != 2 || transitions._shape[0] != numTags || transitions._shape[1] != numTags)
            throw new ArgumentException($"transitions must be [{numTags}, {numTags}]; got [{string.Join(", ", transitions._shape)}].", nameof(transitions));
        if (startScores is not null && startScores.Length != numTags)
            throw new ArgumentException($"startScores must hold {numTags} scores; got {startScores.Length}.", nameof(startScores));
        if (endScores is not null && endScores.Length != numTags)
            throw new ArgumentException($"endScores must hold {numTags} scores; got {endScores.Length}.", nameof(endScores));
        if (TryRecordCompiledOp("TensorCrfForward", new Tensor<T>?[] { emissions, transitions, startScores, endScores },
                e => e.TensorCrfForward(emissions, transitions, startScores, endScores), out var compiledNode))
            return compiledNode;
        if (TryCaptureModelOpInference("TensorCrfForward", new Tensor<T>?[] { emissions, transitions, startScores, endScores }, e => e.TensorCrfForward(emissions, transitions, startScores, endScores), out var captured)) return captured;
        GraphMode.ThrowIfActiveUnsupported(GraphCaptureLimitation.HostBoundary);

        var state = new BackwardFunctions<T>.CrfForwardState
        {
            Batch = batch,
            SeqLen = seqLen,
            NumTags = numTags,
            HasStart = startScores is not null,
            HasEnd = endScores is not null,
            Emissions = RoIAlignPlan.ToDoubles(emissions),
            Transitions = RoIAlignPlan.ToDoubles(transitions),
            Start = startScores is null ? new double[numTags] : RoIAlignPlan.ToDoubles(startScores),
            End = endScores is null ? new double[numTags] : RoIAlignPlan.ToDoubles(endScores),
            Alpha = new double[batch * seqLen * numTags],
            LogZ = new double[batch],
        };
        var emit = state.Emissions;
        var trans = state.Transitions;
        var alpha = state.Alpha;
        var terms = new double[numTags];
        for (int b = 0; b < batch; b++)
        {
            int baseOffset = b * seqLen * numTags;
            for (int j = 0; j < numTags; j++) alpha[baseOffset + j] = state.Start[j] + emit[baseOffset + j];
            for (int t = 1; t < seqLen; t++)
            {
                int previous = baseOffset + (t - 1) * numTags;
                for (int j = 0; j < numTags; j++)
                {
                    for (int i = 0; i < numTags; i++) terms[i] = alpha[previous + i] + trans[i * numTags + j];
                    alpha[baseOffset + t * numTags + j] = emit[baseOffset + t * numTags + j] + BackwardFunctions<T>.LogSumExp(terms);
                }
            }
            int lastStep = baseOffset + (seqLen - 1) * numTags;
            for (int j = 0; j < numTags; j++) terms[j] = alpha[lastStep + j] + state.End[j];
            state.LogZ[b] = BackwardFunctions<T>.LogSumExp(terms);
        }

        var result = RoIAlignPlan.FromDoubles<T>((double[])state.LogZ.Clone(), new[] { batched ? batch : 1 });
        var inputs = new List<Tensor<T>>(4) { emissions, transitions };
        if (startScores is not null) inputs.Add(startScores);
        if (endScores is not null) inputs.Add(endScores);
        DifferentiableOps.RecordIfActive("TensorCrfForward", result, inputs.ToArray(), BackwardFunctions<T>.CrfForwardBackwardWithSecondOrder,
            new object[] { state });
        return result;
    }

    #endregion

    #region Hyperbolic linear (composed)

    // Same boundary as CpuHyperbolicManifoldEngine: a point is kept within (1 - 1e-5) / sqrt(c) of the origin.
    private const double PoincareBoundaryFactor = 1.0 - 1e-5;

    /// <inheritdoc/>
    public virtual Tensor<T> TensorHyperbolicLinear<T>(Tensor<T> input, Tensor<T> weights, Tensor<T>? bias = null, double curvature = 1.0)
    {
        if (input == null) throw new ArgumentNullException(nameof(input));
        if (weights == null) throw new ArgumentNullException(nameof(weights));
        double c = Math.Abs(curvature);
        if (!(c > 0) || double.IsInfinity(c))
            throw new ArgumentOutOfRangeException(nameof(curvature), "curvature must be a finite non-zero number.");
        if (weights.Rank != 2)
            throw new ArgumentException($"weights must be [outFeatures, inFeatures]; got [{string.Join(", ", weights._shape)}].", nameof(weights));
        int outFeatures = weights._shape[0], inFeatures = weights._shape[1];
        if (input.Rank < 1 || input._shape[input.Rank - 1] != inFeatures)
            throw new ArgumentException(
                $"input's last axis must be {inFeatures} (weights' inFeatures); got [{string.Join(", ", input._shape)}].", nameof(input));
        if (bias is not null && bias.Length != outFeatures)
            throw new ArgumentException($"bias must hold {outFeatures} values; got {bias.Length}.", nameof(bias));

        var ops = MathHelper.GetNumericOperations<T>();
        int rows = inFeatures == 0 ? 0 : input.Length / inFeatures;
        var x = input.Rank == 2 ? input : Reshape(input, new[] { rows, inFeatures });

        var tangent = PoincareMapAtOrigin(x, c, logarithmic: true, ops);          // log0(x), [rows, in]
        var image = TensorMatMulTransposed(tangent, weights);                       // W log0(x), [rows, out]
        var y = PoincareMapAtOrigin(image, c, logarithmic: false, ops);            // exp0(.), [rows, out]
        if (bias is not null)
        {
            var b = ProjectToPoincareBall(Reshape(bias, new[] { 1, outFeatures }), c, ops);
            y = MobiusAddRows(y, TensorBroadcastTo(b, new[] { rows, outFeatures }), c, ops);
        }
        y = ProjectToPoincareBall(y, c, ops);

        if (input.Rank == 2) return y;
        var outShape = (int[])input._shape.Clone();
        outShape[outShape.Length - 1] = outFeatures;
        return Reshape(y, outShape);
    }

    // Row norms ‖x_r‖ of a [rows, d] tensor as [rows, 1], floored away from 0 so the maps' v / ‖v‖ stay finite.
    private Tensor<T> RowNorms<T>(Tensor<T> x, Interfaces.INumericOperations<T> ops)
    {
        var squared = ReduceSum(TensorMultiply(x, x), new[] { 1 }, keepDims: true);
        return TensorSqrt(TensorClamp(squared, ops.FromDouble(1e-30), ops.MaxValue));
    }

    // Scales each row of x by a [rows, 1] factor.
    private Tensor<T> ScaleRows<T>(Tensor<T> x, Tensor<T> factor)
        => TensorMultiply(x, TensorBroadcastTo(factor, (int[])x._shape.Clone()));

    // log0(x) = artanh(√c‖x‖) x / (√c‖x‖) or exp0(v) = tanh(√c‖v‖) v / (√c‖v‖), row by row.
    private Tensor<T> PoincareMapAtOrigin<T>(Tensor<T> x, double c, bool logarithmic, Interfaces.INumericOperations<T> ops)
    {
        var scaledNorm = TensorMultiplyScalar(RowNorms(x, ops), ops.FromDouble(Math.Sqrt(c)));
        var factor = logarithmic
            ? TensorAtanh(TensorClamp(scaledNorm, ops.Zero, ops.FromDouble(PoincareBoundaryFactor)))
            : NonNegativeTanh(scaledNorm, ops);
        return ScaleRows(x, TensorDivide(factor, scaledNorm));
    }

    // tanh(z) for z ≥ 0 as expm1(2z) / (expm1(2z) + 2): exact near 0, and built from ops whose backward is itself
    // differentiable, so the composed ops support second derivatives. z is capped at 20, where tanh is 1 in double.
    private Tensor<T> NonNegativeTanh<T>(Tensor<T> z, Interfaces.INumericOperations<T> ops)
    {
        var e = TensorExpm1(TensorMultiplyScalar(TensorClamp(z, ops.Zero, ops.FromDouble(20)), ops.FromDouble(2)));
        return TensorDivide(e, TensorAddScalar(e, ops.FromDouble(2)));
    }

    // σ(z) = 1 / (1 + e^(-z)) from ops whose backward is itself differentiable; z is clamped to ±40, beyond which σ is
    // 0 or 1 to double precision.
    private Tensor<T> SmoothSigmoid<T>(Tensor<T> z, Interfaces.INumericOperations<T> ops)
    {
        var denominator = TensorAddScalar(TensorExp(TensorNegate(TensorClamp(z, ops.FromDouble(-40), ops.FromDouble(40)))), ops.One);
        return TensorDivide(TensorBroadcastTo(Tensor<T>.CreateDefault(new[] { 1 }, ops.One), (int[])denominator._shape.Clone()), denominator);
    }

    // Rescales each row whose norm exceeds (1 - 1e-5) / √c onto that radius.
    private Tensor<T> ProjectToPoincareBall<T>(Tensor<T> x, double c, Interfaces.INumericOperations<T> ops)
    {
        double maxNorm = PoincareBoundaryFactor / Math.Sqrt(c);
        var norms = RowNorms(x, ops);
        var shrink = TensorClamp(TensorDivide(TensorBroadcastTo(Tensor<T>.CreateDefault(new[] { 1, 1 }, ops.FromDouble(maxNorm)),
            (int[])norms._shape.Clone()), norms), ops.Zero, ops.One);
        return ScaleRows(x, shrink);
    }

    // Möbius addition x ⊕_c y = ((1 + 2c⟨x,y⟩ + c‖y‖²) x + (1 - c‖x‖²) y) / (1 + 2c⟨x,y⟩ + c²‖x‖²‖y‖²), row by row.
    private Tensor<T> MobiusAddRows<T>(Tensor<T> x, Tensor<T> y, double c, Interfaces.INumericOperations<T> ops)
    {
        var dot = ReduceSum(TensorMultiply(x, y), new[] { 1 }, keepDims: true);
        var xx = ReduceSum(TensorMultiply(x, x), new[] { 1 }, keepDims: true);
        var yy = ReduceSum(TensorMultiply(y, y), new[] { 1 }, keepDims: true);
        var twoCDot = TensorMultiplyScalar(dot, ops.FromDouble(2 * c));
        var xCoefficient = TensorAddScalar(TensorAdd(twoCDot, TensorMultiplyScalar(yy, ops.FromDouble(c))), ops.One);
        var yCoefficient = TensorAddScalar(TensorMultiplyScalar(xx, ops.FromDouble(-c)), ops.One);
        var denominator = TensorAddScalar(TensorAdd(twoCDot, TensorMultiplyScalar(TensorMultiply(xx, yy), ops.FromDouble(c * c))), ops.One);
        var numerator = TensorAdd(ScaleRows(x, xCoefficient), ScaleRows(y, yCoefficient));
        return ScaleRows(numerator, TensorDivide(TensorBroadcastTo(Tensor<T>.CreateDefault(new[] { 1, 1 }, ops.One),
            (int[])denominator._shape.Clone()), denominator));
    }

    #endregion

    #region Soft trees and instance-based learners (composed)

    /// <inheritdoc/>
    public virtual Tensor<T> TensorSoftSplit<T>(Tensor<T> input, Tensor<T> leftValue, Tensor<T> rightValue, int featureIndex,
        double threshold, double temperature = 1.0)
    {
        if (input == null) throw new ArgumentNullException(nameof(input));
        if (leftValue == null) throw new ArgumentNullException(nameof(leftValue));
        if (rightValue == null) throw new ArgumentNullException(nameof(rightValue));
        if (featureIndex < 0 || featureIndex >= input.Length)
            throw new ArgumentOutOfRangeException(nameof(featureIndex), $"featureIndex must index the {input.Length} input values; got {featureIndex}.");
        if (!leftValue._shape.AsSpan().SequenceEqual(rightValue._shape))
            throw new ArgumentException(
                $"leftValue [{string.Join(", ", leftValue._shape)}] and rightValue [{string.Join(", ", rightValue._shape)}] must have the same shape.",
                nameof(rightValue));
        if (!(temperature > 0) || double.IsInfinity(temperature))
            throw new ArgumentOutOfRangeException(nameof(temperature), "temperature must be a finite positive number.");

        var ops = MathHelper.GetNumericOperations<T>();
        // x_f as a [1, 1] tensor, read as the product with a one-hot column: a matmul, whose backward (unlike a slice's
        // or a full reduction's) carries second derivatives.
        var oneHot = new Tensor<T>(new[] { input.Length, 1 });
        oneHot[featureIndex] = ops.One;
        var feature = TensorMatMul(Reshape(input, new[] { 1, input.Length }), oneHot);
        // p_left = σ((threshold - x_f) / temperature)
        var logit = TensorMultiplyScalar(TensorAddScalar(feature, ops.FromDouble(-threshold)), ops.FromDouble(-1.0 / temperature));
        var pLeft = SmoothSigmoid(logit, ops);
        // p spread over the values' shape as the product of a ones column with p: a matmul, whose backward (unlike the
        // scalar-to-tensor broadcast's) carries second derivatives.
        var onesColumn = Tensor<T>.CreateDefault(new[] { leftValue.Length, 1 }, ops.One);
        var gate = Reshape(TensorMatMul(onesColumn, Reshape(pLeft, new[] { 1, 1 })), (int[])leftValue._shape.Clone());
        // p·left + (1 - p)·right = right + p·(left - right)
        return TensorAdd(rightValue, TensorMultiply(gate, TensorSubtract(leftValue, rightValue)));
    }

    /// <inheritdoc/>
    public virtual Tensor<T> TensorSoftKnn<T>(Tensor<T> input, Tensor<T> supportVectors, Tensor<T> labels, double temperature = 1.0)
        => SoftKernelRegression(input, supportVectors, labels, temperature, nameof(temperature));

    /// <inheritdoc/>
    public virtual Tensor<T> TensorSoftLocallyWeighted<T>(Tensor<T> input, Tensor<T> xTrain, Tensor<T> yTrain, double bandwidth = 1.0)
        => SoftKernelRegression(input, xTrain, yTrain, bandwidth, nameof(bandwidth));

    // softmax(-‖x - s_i‖² / temperature) · labels, for one query or a batch of queries.
    private Tensor<T> SoftKernelRegression<T>(Tensor<T> input, Tensor<T> supportVectors, Tensor<T> labels, double temperature,
        string temperatureName)
    {
        if (input == null) throw new ArgumentNullException(nameof(input));
        if (supportVectors == null) throw new ArgumentNullException(nameof(supportVectors));
        if (labels == null) throw new ArgumentNullException(nameof(labels));
        if (!(temperature > 0) || double.IsInfinity(temperature))
            throw new ArgumentOutOfRangeException(temperatureName, $"{temperatureName} must be a finite positive number.");
        if (supportVectors.Rank != 1 && supportVectors.Rank != 2)
            throw new ArgumentException($"supportVectors must be [n, features] or [n]; got [{string.Join(", ", supportVectors._shape)}].", nameof(supportVectors));
        int n = supportVectors._shape[0];
        int features = supportVectors.Rank == 2 ? supportVectors._shape[1] : 1;
        if (input.Rank != 1 && input.Rank != 2)
            throw new ArgumentException($"input must be [features] or [batch, features]; got [{string.Join(", ", input._shape)}].", nameof(input));
        if (input._shape[input.Rank - 1] != features)
            throw new ArgumentException($"input's last axis must be {features} (the support vectors' features); got [{string.Join(", ", input._shape)}].", nameof(input));
        if (labels.Rank < 1 || labels._shape[0] != n)
            throw new ArgumentException($"labels must be [{n}] or [{n}, outputs]; got [{string.Join(", ", labels._shape)}].", nameof(labels));

        var ops = MathHelper.GetNumericOperations<T>();
        int batch = input.Rank == 2 ? input._shape[0] : 1;
        int outputs = 1;
        for (int axis = 1; axis < labels.Rank; axis++) outputs *= labels._shape[axis];
        // No queries, or no support points to attend over (an empty softmax): an empty or zero result, as the weighted
        // sum over nothing.
        if (batch == 0 || n == 0)
            return input.Rank == 1 ? new Tensor<T>(new[] { outputs }) : new Tensor<T>(new[] { batch, outputs });
        var queries = Reshape(input, new[] { batch, 1, features });
        var points = Reshape(supportVectors, new[] { 1, n, features });
        var span = new[] { batch, n, features };
        var difference = TensorSubtract(TensorBroadcastTo(queries, span), TensorBroadcastTo(points, (int[])span.Clone()));
        var squaredDistance = ReduceSum(TensorMultiply(difference, difference), new[] { 2 }, keepDims: false); // [batch, n]
        // softmax as exp(log_softmax): the same weights, through ops whose backward is itself differentiable.
        var weights = TensorExp(TensorLogSoftmax(TensorMultiplyScalar(squaredDistance, ops.FromDouble(-1.0 / temperature)), 1));
        var result = TensorMatMul(weights, Reshape(labels, new[] { n, outputs }));             // [batch, outputs]
        return input.Rank == 1 ? Reshape(result, new[] { outputs }) : result;
    }

    #endregion
}
