using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Autodiff;

/// <summary>
/// Contains backward function implementations for all differentiable engine operations.
/// Each method is a <see cref="BackwardFunction{T}"/> that computes input gradients
/// and accumulates them into the gradient dictionary.
/// </summary>
internal static class BackwardFunctions<T>
{
    // ──────────────────────────────────────────────────────────────
    // Trivial: gradient is grad_output or scaled grad_output
    // ──────────────────────────────────────────────────────────────

    /// <summary>d(a+b)/da = 1, d(a+b)/db = 1</summary>
    internal static void AddBackward(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        DifferentiableOps.AccumulateGrad(grads, inputs[0], gradOutput, engine);
        DifferentiableOps.AccumulateGrad(grads, inputs[1], gradOutput, engine);
    }

    /// <summary>d(a-b)/da = 1, d(a-b)/db = -1</summary>
    internal static void SubtractBackward(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        DifferentiableOps.AccumulateGrad(grads, inputs[0], gradOutput, engine);
        var negGrad = engine.TensorNegate(gradOutput);
        DifferentiableOps.AccumulateGrad(grads, inputs[1], negGrad, engine);
    }

    /// <summary>d(-x)/dx = -1</summary>
    internal static void NegateBackward(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        var negGrad = engine.TensorNegate(gradOutput);
        DifferentiableOps.AccumulateGrad(grads, inputs[0], negGrad, engine);
    }

    /// <summary>d(x+s)/dx = 1</summary>
    internal static void AddScalarBackward(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        DifferentiableOps.AccumulateGrad(grads, inputs[0], gradOutput, engine);
    }

    /// <summary>d(x-s)/dx = 1</summary>
    internal static void SubtractScalarBackward(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        DifferentiableOps.AccumulateGrad(grads, inputs[0], gradOutput, engine);
    }

    /// <summary>d(x*s)/dx = s (scalar stored in savedState[0])</summary>
    internal static void MultiplyScalarBackward(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        var scalar = (T)savedState[0];
        var grad = engine.TensorMultiplyScalar(gradOutput, scalar);
        DifferentiableOps.AccumulateGrad(grads, inputs[0], grad, engine);
    }

    /// <summary>d(x/s)/dx = 1/s (scalar stored in savedState[0])</summary>
    internal static void DivideScalarBackward(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        var scalar = (T)savedState[0];
        var numOps = MathHelper.GetNumericOperations<T>();
        var invScalar = numOps.Divide(numOps.One, scalar);
        var grad = engine.TensorMultiplyScalar(gradOutput, invScalar);
        DifferentiableOps.AccumulateGrad(grads, inputs[0], grad, engine);
    }

    // ──────────────────────────────────────────────────────────────
    // Element-wise: gradient depends on inputs/outputs
    // ──────────────────────────────────────────────────────────────

    /// <summary>d(a*b)/da = grad*b, d(a*b)/db = grad*a</summary>
    internal static void MultiplyBackward(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        var gradA = engine.TensorMultiply(gradOutput, inputs[1]);
        var gradB = engine.TensorMultiply(gradOutput, inputs[0]);
        DifferentiableOps.AccumulateGrad(grads, inputs[0], gradA, engine);
        DifferentiableOps.AccumulateGrad(grads, inputs[1], gradB, engine);
    }

    /// <summary>d(a/b)/da = grad/b, d(a/b)/db = -grad*a/(b*b)</summary>
    internal static void DivideBackward(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        var gradA = engine.TensorDivide(gradOutput, inputs[1]);
        var bSquared = engine.TensorMultiply(inputs[1], inputs[1]);
        var negGradA = engine.TensorNegate(engine.TensorMultiply(gradOutput, inputs[0]));
        var gradB = engine.TensorDivide(negGradA, bSquared);
        DifferentiableOps.AccumulateGrad(grads, inputs[0], gradA, engine);
        DifferentiableOps.AccumulateGrad(grads, inputs[1], gradB, engine);
    }

    /// <summary>d(exp(x))/dx = grad * exp(x) = grad * output</summary>
    internal static void ExpBackward(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        var grad = engine.TensorMultiply(gradOutput, output);
        DifferentiableOps.AccumulateGrad(grads, inputs[0], grad, engine);
    }

    /// <summary>d(log(x))/dx = grad / x</summary>
    internal static void LogBackward(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        var grad = engine.TensorDivide(gradOutput, inputs[0]);
        DifferentiableOps.AccumulateGrad(grads, inputs[0], grad, engine);
    }

    /// <summary>d(sqrt(x))/dx = grad / (2 * sqrt(x)) = grad / (2 * output)</summary>
    internal static void SqrtBackward(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var two = numOps.FromDouble(2.0);
        var twoOutput = engine.TensorMultiplyScalar(output, two);
        var grad = engine.TensorDivide(gradOutput, twoOutput);
        DifferentiableOps.AccumulateGrad(grads, inputs[0], grad, engine);
    }

    /// <summary>d(|x|)/dx = grad * sign(x)</summary>
    internal static void AbsBackward(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        var signTensor = engine.TensorSign(inputs[0]);
        var grad = engine.TensorMultiply(gradOutput, signTensor);
        DifferentiableOps.AccumulateGrad(grads, inputs[0], grad, engine);
    }

    /// <summary>d(x^p)/dx = grad * p * x^(p-1) (exponent stored in savedState[0])</summary>
    internal static void PowerBackward(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var exponent = (T)savedState[0];
        var expMinus1 = numOps.Subtract(exponent, numOps.One);
        var xPowExpMinus1 = engine.TensorPower(inputs[0], expMinus1);
        var scaled = engine.TensorMultiplyScalar(xPowExpMinus1, exponent);
        var grad = engine.TensorMultiply(gradOutput, scaled);
        DifferentiableOps.AccumulateGrad(grads, inputs[0], grad, engine);
    }

    /// <summary>d(base^exp)/d(base) = grad * exp * base^(exp-1), for element-wise tensor power</summary>
    internal static void PowerTensorBackward(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var bases = inputs[0];
        var exponents = inputs[1];

        // d/d(base) = grad * exp * base^(exp - 1)
        var expMinus1 = engine.TensorAddScalar(exponents, numOps.FromDouble(-1.0));
        var basePow = engine.TensorPower(bases, expMinus1);
        var scaled = engine.TensorMultiply(exponents, basePow);
        var gradBase = engine.TensorMultiply(gradOutput, scaled);
        DifferentiableOps.AccumulateGrad(grads, bases, gradBase, engine);

        // d/d(exp) = grad * base^exp * ln(base)
        var logBase = engine.TensorLog(bases);
        var dExp = engine.TensorMultiply(output, logBase);
        var gradExp = engine.TensorMultiply(gradOutput, dExp);
        DifferentiableOps.AccumulateGrad(grads, exponents, gradExp, engine);
    }

    /// <summary>d(sin(x))/dx = grad * cos(x)</summary>
    internal static void SinBackward(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        var cosX = engine.TensorCos(inputs[0]);
        var grad = engine.TensorMultiply(gradOutput, cosX);
        DifferentiableOps.AccumulateGrad(grads, inputs[0], grad, engine);
    }

    /// <summary>d(cos(x))/dx = -grad * sin(x)</summary>
    internal static void CosBackward(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        var sinX = engine.TensorSin(inputs[0]);
        var negSinX = engine.TensorNegate(sinX);
        var grad = engine.TensorMultiply(gradOutput, negSinX);
        DifferentiableOps.AccumulateGrad(grads, inputs[0], grad, engine);
    }

    /// <summary>d(clamp(x, min, max))/dx = grad where min &lt; x &lt; max, else 0</summary>
    internal static void ClampBackward(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        // d/dx clamp(x, min, max) = 1 if min <= x <= max, else 0
        // Compute via: mask = (x >= min) * (x <= max), then grad * mask
        var numOps = MathHelper.GetNumericOperations<T>();
        var min = numOps.FromDouble((double)savedState[0]);
        var max = numOps.FromDouble((double)savedState[1]);

        // Clamp the input to [min, max], compare with original — equal means in range
        var clamped = engine.TensorClamp(inputs[0], min, max);
        // Where clamped == input, gradient passes through; otherwise zero
        var diff = engine.TensorSubtract(clamped, inputs[0]);
        var absDiff = engine.TensorAbs(diff);
        // absDiff is 0 where in range, nonzero where clamped — use as mask
        // Create ones tensor and subtract the sign of absDiff
        var ones = engine.TensorAddScalar(engine.TensorMultiplyScalar(absDiff, numOps.Zero), numOps.One);
        var sign = engine.TensorSign(absDiff);
        var mask = engine.TensorSubtract(ones, sign);
        var grad = engine.TensorMultiply(gradOutput, mask);
        DifferentiableOps.AccumulateGrad(grads, inputs[0], grad, engine);
    }

    // ──────────────────────────────────────────────────────────────
    // Activations: delegate to engine backward methods
    // ──────────────────────────────────────────────────────────────

    /// <summary>Uses engine.ReluBackward(gradOutput, input)</summary>
    internal static void ReLUBackward(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        var grad = engine.ReluBackward(gradOutput, inputs[0]);
        DifferentiableOps.AccumulateGrad(grads, inputs[0], grad, engine);
    }

    /// <summary>Uses engine.SigmoidBackward(gradOutput, output)</summary>
    internal static void SigmoidBackward(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        var grad = engine.SigmoidBackward(gradOutput, output);
        DifferentiableOps.AccumulateGrad(grads, inputs[0], grad, engine);
    }

    /// <summary>Uses engine.TanhBackward(gradOutput, output)</summary>
    internal static void TanhBackward(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        var grad = engine.TanhBackward(gradOutput, output);
        DifferentiableOps.AccumulateGrad(grads, inputs[0], grad, engine);
    }

    /// <summary>Uses engine.GeluBackward(gradOutput, input)</summary>
    internal static void GELUBackward(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        var grad = engine.GeluBackward(gradOutput, inputs[0]);
        DifferentiableOps.AccumulateGrad(grads, inputs[0], grad, engine);
    }

    /// <summary>Uses engine.LeakyReluBackward(gradOutput, input, negativeSlope)</summary>
    internal static void LeakyReLUBackward(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        var negativeSlope = (double)savedState[0];
        var grad = engine.LeakyReluBackward(gradOutput, inputs[0], negativeSlope);
        DifferentiableOps.AccumulateGrad(grads, inputs[0], grad, engine);
    }

    /// <summary>Swish backward: d(x*sigmoid(x))/dx = sigmoid(x) + x*sigmoid(x)*(1-sigmoid(x))</summary>
    internal static void SwishBackward(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var sig = engine.Sigmoid(inputs[0]);
        var oneMinusSig = engine.TensorSubtract(
            CreateOnes(inputs[0].Shape.ToArray(), numOps), sig);
        var xTimesSig = engine.TensorMultiply(inputs[0], sig);
        var xSigOneMinusSig = engine.TensorMultiply(xTimesSig, oneMinusSig);
        var derivative = engine.TensorAdd(sig, xSigOneMinusSig);
        var grad = engine.TensorMultiply(gradOutput, derivative);
        DifferentiableOps.AccumulateGrad(grads, inputs[0], grad, engine);
    }

    /// <summary>Mish backward: d(x*tanh(softplus(x)))/dx</summary>
    internal static void MishBackward(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        // softplus = log(1 + exp(x))
        var expX = engine.TensorExp(inputs[0]);
        var ones = CreateOnes(inputs[0].Shape.ToArray(), numOps);
        var onePlusExp = engine.TensorAdd(ones, expX);
        var softplus = engine.TensorLog(onePlusExp);
        var tanhSp = engine.Tanh(softplus);
        // sigmoid = exp(x) / (1 + exp(x))
        var sigmoid = engine.TensorDivide(expX, onePlusExp);
        // d(tanh(sp))/dx = (1 - tanh(sp)^2) * sigmoid
        var tanhSq = engine.TensorMultiply(tanhSp, tanhSp);
        var oneMinusTanhSq = engine.TensorSubtract(ones, tanhSq);
        var dtanhDx = engine.TensorMultiply(oneMinusTanhSq, sigmoid);
        // d(mish)/dx = tanh(sp) + x * dtanh/dx
        var xDtanh = engine.TensorMultiply(inputs[0], dtanhDx);
        var derivative = engine.TensorAdd(tanhSp, xDtanh);
        var grad = engine.TensorMultiply(gradOutput, derivative);
        DifferentiableOps.AccumulateGrad(grads, inputs[0], grad, engine);
    }

    /// <summary>Uses engine.SoftmaxBackward(gradOutput, output, axis)</summary>
    internal static void SoftmaxBackward(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        var axis = (int)savedState[0];
        var grad = engine.SoftmaxBackward(gradOutput, output, axis);
        DifferentiableOps.AccumulateGrad(grads, inputs[0], grad, engine);
    }

    /// <summary>ELU backward: grad * (output > 0 ? 1 : output + alpha)</summary>
    internal static void ELUBackward(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var alpha = numOps.FromDouble((double)savedState[0]);
        var derivative = TensorPool<T>.RentZeroed(output.Shape.ToArray());
        for (int i = 0; i < output.Length; i++)
        {
            var val = output.GetFlat(i);
            if (numOps.GreaterThan(val, numOps.Zero))
                derivative.SetFlat(i, numOps.One);
            else
                derivative.SetFlat(i, numOps.Add(val, alpha));
        }
        var grad = engine.TensorMultiply(gradOutput, derivative);
        DifferentiableOps.AccumulateGrad(grads, inputs[0], grad, engine);
    }

    // ──────────────────────────────────────────────────────────────
    // Matrix operations
    // ──────────────────────────────────────────────────────────────

    /// <summary>d(A@B)/dA = grad @ B^T, d(A@B)/dB = A^T @ grad</summary>
    internal static void MatMulBackward(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        var bT = engine.TensorTranspose(inputs[1]);
        var gradA = engine.TensorMatMul(gradOutput, bT);

        var aT = engine.TensorTranspose(inputs[0]);
        var gradB = engine.TensorMatMul(aT, gradOutput);

        DifferentiableOps.AccumulateGrad(grads, inputs[0], gradA, engine);
        DifferentiableOps.AccumulateGrad(grads, inputs[1], gradB, engine);
    }

    /// <summary>d(A^T)/dA = transpose(grad)</summary>
    internal static void TransposeBackward(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        var grad = engine.TensorTranspose(gradOutput);
        DifferentiableOps.AccumulateGrad(grads, inputs[0], grad, engine);
    }

    /// <summary>d(permute(x, axes))/dx = permute(grad, inverse_axes)</summary>
    internal static void PermuteBackward(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        var axes = (int[])savedState[0];
        // Compute inverse permutation
        var inverseAxes = new int[axes.Length];
        for (int i = 0; i < axes.Length; i++)
            inverseAxes[axes[i]] = i;
        var grad = engine.TensorPermute(gradOutput, inverseAxes);
        DifferentiableOps.AccumulateGrad(grads, inputs[0], grad, engine);
    }

    /// <summary>d(reshape(x))/dx = reshape(grad, original_shape)</summary>
    internal static void ReshapeBackward(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        var originalShape = (int[])savedState[0];
        var grad = engine.Reshape(gradOutput, originalShape);
        DifferentiableOps.AccumulateGrad(grads, inputs[0], grad, engine);
    }

    // ──────────────────────────────────────────────────────────────
    // Broadcast operations: reduce gradient along broadcast dims
    // ──────────────────────────────────────────────────────────────

    /// <summary>d(broadcast_add(a,b))/da = reduce_grad(grad, a.shape), d/db = reduce_grad(grad, b.shape)</summary>
    internal static void BroadcastAddBackward(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        var gradA = ReduceGradToShape(gradOutput, inputs[0].Shape.ToArray(), engine);
        var gradB = ReduceGradToShape(gradOutput, inputs[1].Shape.ToArray(), engine);
        DifferentiableOps.AccumulateGrad(grads, inputs[0], gradA, engine);
        DifferentiableOps.AccumulateGrad(grads, inputs[1], gradB, engine);
    }

    /// <summary>d(broadcast_sub(a,b))/da = reduce(grad), d/db = -reduce(grad)</summary>
    internal static void BroadcastSubtractBackward(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        var gradA = ReduceGradToShape(gradOutput, inputs[0].Shape.ToArray(), engine);
        var negGrad = engine.TensorNegate(gradOutput);
        var gradB = ReduceGradToShape(negGrad, inputs[1].Shape.ToArray(), engine);
        DifferentiableOps.AccumulateGrad(grads, inputs[0], gradA, engine);
        DifferentiableOps.AccumulateGrad(grads, inputs[1], gradB, engine);
    }

    /// <summary>d(broadcast_mul(a,b))/da = reduce(grad*b), d/db = reduce(grad*a)</summary>
    internal static void BroadcastMultiplyBackward(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        var fullGradA = engine.TensorBroadcastMultiply(gradOutput, inputs[1]);
        var fullGradB = engine.TensorBroadcastMultiply(gradOutput, inputs[0]);
        var gradA = ReduceGradToShape(fullGradA, inputs[0].Shape.ToArray(), engine);
        var gradB = ReduceGradToShape(fullGradB, inputs[1].Shape.ToArray(), engine);
        DifferentiableOps.AccumulateGrad(grads, inputs[0], gradA, engine);
        DifferentiableOps.AccumulateGrad(grads, inputs[1], gradB, engine);
    }

    /// <summary>d(broadcast_div(a,b))/da = reduce(grad/b), d/db = reduce(-grad*a/b^2)</summary>
    internal static void BroadcastDivideBackward(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        var fullGradA = engine.TensorBroadcastDivide(gradOutput, inputs[1]);
        var gradA = ReduceGradToShape(fullGradA, inputs[0].Shape.ToArray(), engine);

        var bSquared = engine.TensorBroadcastMultiply(inputs[1], inputs[1]);
        var negGradA = engine.TensorNegate(engine.TensorBroadcastMultiply(gradOutput, inputs[0]));
        var fullGradB = engine.TensorBroadcastDivide(negGradA, bSquared);
        var gradB = ReduceGradToShape(fullGradB, inputs[1].Shape.ToArray(), engine);

        DifferentiableOps.AccumulateGrad(grads, inputs[0], gradA, engine);
        DifferentiableOps.AccumulateGrad(grads, inputs[1], gradB, engine);
    }

    // ──────────────────────────────────────────────────────────────
    // Neural network layers: delegate to engine backward methods
    // ──────────────────────────────────────────────────────────────

    /// <summary>Conv2D backward: uses engine.Conv2DBackwardInput and Conv2DBackwardKernel</summary>
    internal static void Conv2DBackward(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        var stride = (int[])savedState[0];
        var padding = (int[])savedState[1];
        var dilation = (int[])savedState[2];

        var gradInput = engine.Conv2DBackwardInput(
            gradOutput, inputs[1], inputs[0].Shape.ToArray(), stride, padding, dilation);
        var gradKernel = engine.Conv2DBackwardKernel(
            gradOutput, inputs[0], inputs[1].Shape.ToArray(), stride, padding, dilation);

        DifferentiableOps.AccumulateGrad(grads, inputs[0], gradInput, engine);
        DifferentiableOps.AccumulateGrad(grads, inputs[1], gradKernel, engine);
    }

    /// <summary>Conv1D backward: reshapes 3D inputs to 4D, delegates to Conv2DBackward logic, reshapes back</summary>
    internal static void Conv1DBackward(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        var stride = (int)savedState[0];
        var padding = (int)savedState[1];
        var dilation = (int)savedState[2];

        var gradInput = engine.Conv1DBackwardInput(
            gradOutput, inputs[1], inputs[0].Shape.ToArray(), stride, padding, dilation);
        var gradKernel = engine.Conv1DBackwardKernel(
            gradOutput, inputs[0], inputs[1].Shape.ToArray(), stride, padding, dilation);

        DifferentiableOps.AccumulateGrad(grads, inputs[0], gradInput, engine);
        DifferentiableOps.AccumulateGrad(grads, inputs[1], gradKernel, engine);
    }

    /// <summary>Conv3D backward: uses engine.Conv3DBackwardInput and Conv3DBackwardKernel</summary>
    internal static void Conv3DBackward(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        var stride = (int[])savedState[0];
        var padding = (int[])savedState[1];
        var dilation = (int[])savedState[2];

        var gradInput = engine.Conv3DBackwardInput(
            gradOutput, inputs[1], inputs[0].Shape.ToArray(), stride, padding, dilation);
        var gradKernel = engine.Conv3DBackwardKernel(
            gradOutput, inputs[0], inputs[1].Shape.ToArray(), stride, padding, dilation);

        DifferentiableOps.AccumulateGrad(grads, inputs[0], gradInput, engine);
        DifferentiableOps.AccumulateGrad(grads, inputs[1], gradKernel, engine);
    }

    /// <summary>GridSample backward: uses engine.GridSampleBackwardInput and GridSampleBackwardGrid</summary>
    internal static void GridSampleBackward(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        // inputs[0] = input, inputs[1] = grid
        var gradInput = engine.GridSampleBackwardInput(gradOutput, inputs[1], inputs[0].Shape.ToArray());
        var gradGrid = engine.GridSampleBackwardGrid(gradOutput, inputs[0], inputs[1]);

        DifferentiableOps.AccumulateGrad(grads, inputs[0], gradInput, engine);
        DifferentiableOps.AccumulateGrad(grads, inputs[1], gradGrid, engine);
    }

    /// <summary>Unfold backward: fold the gradient back to input shape</summary>
    internal static void UnfoldBackward(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        var kernelSize = (int[])savedState[0];
        var stride = (int[])savedState[1];
        var padding = (int[])savedState[2];
        var outputSize = new[] { inputs[0].Shape.ToArray()[2], inputs[0].Shape.ToArray()[3] };

        var grad = engine.Fold(gradOutput, outputSize, kernelSize, stride, padding);
        DifferentiableOps.AccumulateGrad(grads, inputs[0], grad, engine);
    }

    /// <summary>Fold backward: unfold the gradient back to column shape</summary>
    internal static void FoldBackward(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        var kernelSize = (int[])savedState[0];
        var stride = (int[])savedState[1];
        var padding = (int[])savedState[2];

        var grad = engine.Unfold(gradOutput, kernelSize, stride, padding);
        DifferentiableOps.AccumulateGrad(grads, inputs[0], grad, engine);
    }

    /// <summary>MaxPool2D backward: uses engine.MaxPool2DBackward with saved max indices</summary>
    internal static void MaxPool2DBackward(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        var maxIndices = (int[,,,,])savedState[0];
        var poolSize = (int[])savedState[1];
        var stride = (int[])savedState[2];

        var grad = engine.MaxPool2DBackward(gradOutput, maxIndices, inputs[0].Shape.ToArray(), poolSize, stride);
        DifferentiableOps.AccumulateGrad(grads, inputs[0], grad, engine);
    }

    /// <summary>AvgPool2D backward: uses engine.AvgPool2DBackward</summary>
    internal static void AvgPool2DBackward(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        var poolSize = (int[])savedState[0];
        var stride = (int[])savedState[1];

        var grad = engine.AvgPool2DBackward(gradOutput, inputs[0].Shape.ToArray(), poolSize, stride);
        DifferentiableOps.AccumulateGrad(grads, inputs[0], grad, engine);
    }

    /// <summary>BatchNorm backward: uses engine.BatchNormBackward</summary>
    internal static void BatchNormBackward(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        // inputs[0] = input, inputs[1] = gamma, inputs[2] = beta
        var mean = (Tensor<T>)savedState[0];
        var variance = (Tensor<T>)savedState[1];
        var epsilon = (double)savedState[2];

        var gradInput = engine.BatchNormBackward(
            gradOutput, inputs[0], inputs[1], mean, variance, epsilon,
            out var gradGamma, out var gradBeta);

        DifferentiableOps.AccumulateGrad(grads, inputs[0], gradInput, engine);
        DifferentiableOps.AccumulateGrad(grads, inputs[1], gradGamma, engine);
        DifferentiableOps.AccumulateGrad(grads, inputs[2], gradBeta, engine);
    }

    /// <summary>LayerNorm backward: uses engine.LayerNormBackward</summary>
    internal static void LayerNormBackward(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        var mean = (Tensor<T>)savedState[0];
        var variance = (Tensor<T>)savedState[1];
        var epsilon = (double)savedState[2];

        var gradInput = engine.LayerNormBackward(
            gradOutput, inputs[0], inputs[1], mean, variance, epsilon,
            out var gradGamma, out var gradBeta);

        DifferentiableOps.AccumulateGrad(grads, inputs[0], gradInput, engine);
        DifferentiableOps.AccumulateGrad(grads, inputs[1], gradGamma, engine);
        DifferentiableOps.AccumulateGrad(grads, inputs[2], gradBeta, engine);
    }

    /// <summary>GroupNorm backward: uses engine.GroupNormBackward</summary>
    internal static void GroupNormBackward(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        var numGroups = (int)savedState[0];
        var mean = (Tensor<T>)savedState[1];
        var variance = (Tensor<T>)savedState[2];
        var epsilon = (double)savedState[3];

        var gradInput = engine.GroupNormBackward(
            gradOutput, inputs[0], numGroups, inputs[1], mean, variance, epsilon,
            out var gradGamma, out var gradBeta);

        DifferentiableOps.AccumulateGrad(grads, inputs[0], gradInput, engine);
        DifferentiableOps.AccumulateGrad(grads, inputs[1], gradGamma, engine);
        DifferentiableOps.AccumulateGrad(grads, inputs[2], gradBeta, engine);
    }

    /// <summary>RMSNorm backward: uses engine.RMSNormBackward</summary>
    internal static void RMSNormBackward(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        var rms = (Tensor<T>)savedState[0];
        var epsilon = (double)savedState[1];

        var gradInput = engine.RMSNormBackward(
            gradOutput, inputs[0], inputs[1], rms, epsilon, out var gradGamma);

        DifferentiableOps.AccumulateGrad(grads, inputs[0], gradInput, engine);
        DifferentiableOps.AccumulateGrad(grads, inputs[1], gradGamma, engine);
    }

    /// <summary>InstanceNorm backward: uses engine.InstanceNormBackward</summary>
    internal static void InstanceNormBackward(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        var mean = (Tensor<T>)savedState[0];
        var variance = (Tensor<T>)savedState[1];
        var epsilon = (double)savedState[2];

        var gradInput = engine.InstanceNormBackward(
            gradOutput, inputs[0], inputs[1], mean, variance, epsilon,
            out var gradGamma, out var gradBeta);

        DifferentiableOps.AccumulateGrad(grads, inputs[0], gradInput, engine);
        DifferentiableOps.AccumulateGrad(grads, inputs[1], gradGamma, engine);
        DifferentiableOps.AccumulateGrad(grads, inputs[2], gradBeta, engine);
    }

    /// <summary>Dropout backward: uses engine.DropoutBackward with saved mask</summary>
    internal static void DropoutBackward(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        var mask = (Tensor<T>)savedState[0];
        var dropoutRate = (double)savedState[1];

        var grad = engine.DropoutBackward(gradOutput, mask, dropoutRate);
        DifferentiableOps.AccumulateGrad(grads, inputs[0], grad, engine);
    }

    /// <summary>Embedding backward: uses engine.EmbeddingBackward</summary>
    internal static void EmbeddingBackward(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        var indices = (Tensor<int>)savedState[0];
        var vocabSize = (int)savedState[1];
        var embeddingDim = (int)savedState[2];

        var grad = engine.EmbeddingBackward(gradOutput, indices, vocabSize, embeddingDim);
        // Gradient flows to the embedding table (inputs[0])
        DifferentiableOps.AccumulateGrad(grads, inputs[0], grad, engine);
    }

    // ──────────────────────────────────────────────────────────────
    // Reduction operations
    // ──────────────────────────────────────────────────────────────

    /// <summary>d(sum(x, axes))/dx = broadcast grad back to input shape along reduced axes</summary>
    internal static void ReduceSumBackward(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        var axes = (int[])savedState[0];
        var keepDims = (bool)savedState[1];
        var inputShape = inputs[0].Shape.ToArray();

        // If keepDims was false, we need to reinsert singleton dimensions at the reduced axes
        // so we can broadcast back to the original shape
        var expandedGrad = gradOutput;
        if (!keepDims)
        {
            // Sort axes so we insert in order
            var sortedAxes = axes.OrderBy(a => a).ToArray();
            foreach (var axis in sortedAxes)
            {
                expandedGrad = engine.TensorExpandDims(expandedGrad, axis);
            }
        }

        // Now expandedGrad has the same rank as input with size-1 at reduced dims — tile to match
        var grad = BroadcastGradToShape(expandedGrad, inputShape, engine);
        DifferentiableOps.AccumulateGrad(grads, inputs[0], grad, engine);
    }

    /// <summary>d(mean(x))/dx = grad / N (broadcast and scale)</summary>
    internal static void ReduceMeanBackward(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        var axes = (int[])savedState[0];
        var grad = engine.ReduceMeanBackward(gradOutput, inputs[0].Shape.ToArray(), axes);
        DifferentiableOps.AccumulateGrad(grads, inputs[0], grad, engine);
    }

    // ──────────────────────────────────────────────────────────────
    // Shape operations
    // ──────────────────────────────────────────────────────────────

    /// <summary>Slice backward: scatter grad into zeros at the sliced positions</summary>
    internal static void SliceBackward(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        var start = (int[])savedState[0];
        var inputShape = inputs[0].Shape.ToArray();
        var numOps = MathHelper.GetNumericOperations<T>();

        // Create zeros with input shape, then set the sliced region to gradOutput
        var grad = TensorPool<T>.RentZeroed(inputShape);
        engine.TensorFill(grad, numOps.Zero);
        engine.TensorSetSlice(grad, gradOutput, start);
        DifferentiableOps.AccumulateGrad(grads, inputs[0], grad, engine);
    }

    /// <summary>Gather backward: scatter grad to source positions</summary>
    internal static void GatherBackward(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        var indices = (Tensor<int>)savedState[0];
        var axis = (int)savedState[1];

        var grad = engine.ScatterAddBackward(gradOutput, indices, inputs[0].Shape.ToArray(), axis);
        DifferentiableOps.AccumulateGrad(grads, inputs[0], grad, engine);
    }

    /// <summary>ScatterAdd backward: gather grad from destination positions</summary>
    internal static void ScatterAddBackward(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        var indices = (Tensor<int>)savedState[0];
        var axis = (int)savedState[1];

        // Gradient flows to both destination and updates
        DifferentiableOps.AccumulateGrad(grads, inputs[0], gradOutput, engine);
        var gradUpdates = engine.TensorGather(gradOutput, indices, axis);
        DifferentiableOps.AccumulateGrad(grads, inputs[1], gradUpdates, engine);
    }

    /// <summary>ExpandDims backward: squeeze the added dimension</summary>
    internal static void ExpandDimsBackward(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        var axis = (int)savedState[0];
        var grad = engine.TensorSqueeze(gradOutput, axis);
        DifferentiableOps.AccumulateGrad(grads, inputs[0], grad, engine);
    }

    /// <summary>Squeeze backward: expand the squeezed dimension</summary>
    internal static void SqueezeBackward(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        var axis = (int)savedState[0];
        var grad = engine.TensorExpandDims(gradOutput, axis);
        DifferentiableOps.AccumulateGrad(grads, inputs[0], grad, engine);
    }

    // ──────────────────────────────────────────────────────────────
    // Helper methods
    // ──────────────────────────────────────────────────────────────

    /// <summary>
    /// Reduces a gradient tensor to a target shape by summing along broadcast dimensions.
    /// </summary>
    private static Tensor<T> ReduceGradToShape(Tensor<T> grad, int[] targetShape, IEngine engine)
    {
        var gradShape = grad.Shape.ToArray();

        // If shapes already match, no reduction needed
        if (ShapesEqual(gradShape, targetShape))
            return grad;

        // Pad target shape with leading 1s to match rank
        var paddedTarget = PadShapeToRank(targetShape, gradShape.Length);

        // Find axes where target has size 1 but grad has size > 1 (broadcast dims)
        var reduceAxes = new List<int>();
        for (int i = 0; i < gradShape.Length; i++)
        {
            if (paddedTarget[i] == 1 && gradShape[i] > 1)
                reduceAxes.Add(i);
        }

        // Also reduce leading dimensions that were added by broadcasting
        for (int i = 0; i < gradShape.Length - targetShape.Length; i++)
        {
            if (!reduceAxes.Contains(i))
                reduceAxes.Add(i);
        }

        if (reduceAxes.Count == 0)
            return grad;

        var reduced = engine.ReduceSum(grad, reduceAxes.ToArray(), keepDims: true);
        return engine.Reshape(reduced, targetShape);
    }

    /// <summary>
    /// Broadcasts a gradient tensor to a target shape.
    /// </summary>
    private static Tensor<T> BroadcastGradToShape(Tensor<T> grad, int[] targetShape, IEngine engine)
    {
        var gradShape = grad.Shape.ToArray();
        if (ShapesEqual(gradShape, targetShape))
            return grad;

        // If grad is scalar-like (length 1), tile it to target shape
        if (grad.Length == 1)
        {
            var result = TensorPool<T>.RentZeroed(targetShape);
            engine.TensorFill(result, grad.GetFlat(0));
            return result;
        }

        // Reshape and tile as needed
        return engine.TensorTile(engine.Reshape(grad, PadShapeToRank(gradShape, targetShape.Length)),
            ComputeTileMultiples(PadShapeToRank(gradShape, targetShape.Length), targetShape));
    }

    private static bool ShapesEqual(int[] a, int[] b)
    {
        if (a.Length != b.Length) return false;
        for (int i = 0; i < a.Length; i++)
            if (a[i] != b[i]) return false;
        return true;
    }

    private static int[] PadShapeToRank(int[] shape, int rank)
    {
        if (shape.Length >= rank) return shape;
        var padded = new int[rank];
        var offset = rank - shape.Length;
        for (int i = 0; i < offset; i++) padded[i] = 1;
        Array.Copy(shape, 0, padded, offset, shape.Length);
        return padded;
    }

    private static int[] ComputeTileMultiples(int[] currentShape, int[] targetShape)
    {
        var multiples = new int[targetShape.Length];
        for (int i = 0; i < targetShape.Length; i++)
            multiples[i] = currentShape[i] == 1 ? targetShape[i] : 1;
        return multiples;
    }

    private static Tensor<T> CreateOnes(int[] shape, INumericOperations<T> numOps)
    {
        int length = 1;
        for (int i = 0; i < shape.Length; i++)
            length *= shape[i];
        var data = new T[length];
        for (int i = 0; i < data.Length; i++)
            data[i] = numOps.One;
        return new Tensor<T>(data, shape);
    }

    // ──────────────────────────────────────────────────────────────
    // Issue #76: Missing backward functions
    // ──────────────────────────────────────────────────────────────

    /// <summary>d(softplus(x))/dx = sigmoid(beta*x)</summary>
    internal static void SoftplusBackward(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        double beta = savedState.Length > 0 ? (double)savedState[0] : 1.0;
        // d(softplus(x))/dx = sigmoid(beta*x)
        var scaled = engine.TensorMultiplyScalar(inputs[0], numOps.FromDouble(beta));
        var sig = engine.TensorSigmoid(scaled);
        var grad = engine.TensorMultiply(gradOutput, sig);
        DifferentiableOps.AccumulateGrad(grads, inputs[0], grad, engine);
    }

    /// <summary>d(selu(x))/dx = lambda if x >= 0, lambda*alpha*exp(x) if x < 0</summary>
    internal static void SELUBackward(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        var grad = engine.SeluBackward(gradOutput, inputs[0]);
        DifferentiableOps.AccumulateGrad(grads, inputs[0], grad, engine);
    }

    /// <summary>d(hardsigmoid(x))/dx = 1/6 if -3 < x < 3, else 0</summary>
    internal static void HardSigmoidBackward(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        var grad = engine.HardsigmoidBackward(gradOutput, inputs[0]);
        DifferentiableOps.AccumulateGrad(grads, inputs[0], grad, engine);
    }

    /// <summary>d(hardswish(x))/dx</summary>
    internal static void HardSwishBackward(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        // derivative = clamp((2x+3)/6, 0, 1) — use engine ops for graph recording
        var twoX = engine.TensorMultiplyScalar(inputs[0], numOps.FromDouble(2.0));
        var twoXPlus3 = engine.TensorAddScalar(twoX, numOps.FromDouble(3.0));
        var scaled = engine.TensorDivideScalar(twoXPlus3, numOps.FromDouble(6.0));
        var clamped = engine.TensorClamp(scaled, numOps.Zero, numOps.One);
        var grad = engine.TensorMultiply(gradOutput, clamped);
        DifferentiableOps.AccumulateGrad(grads, inputs[0], grad, engine);
    }

    /// <summary>d(relu6(x))/dx = 1 if 0 < x < 6, else 0</summary>
    internal static void ReLU6Backward(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        var grad = engine.Relu6Backward(gradOutput, inputs[0]);
        DifferentiableOps.AccumulateGrad(grads, inputs[0], grad, engine);
    }

    /// <summary>Concatenate backward: split gradient along concat axis</summary>
    internal static void ConcatenateBackward(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        int axis = (int)savedState[0];
        int offset = 0;
        var gradShape = gradOutput.Shape.ToArray();
        for (int i = 0; i < inputs.Length; i++)
        {
            int size = inputs[i].Shape[axis];
            // Build start/length arrays for TensorSlice
            var start = new int[gradShape.Length];
            var length = (int[])gradShape.Clone();
            start[axis] = offset;
            length[axis] = size;
            var sliced = engine.TensorSlice(gradOutput, start, length);
            DifferentiableOps.AccumulateGrad(grads, inputs[i], sliced, engine);
            offset += size;
        }
    }

    /// <summary>ConvTranspose2D backward</summary>
    internal static void ConvTranspose2DBackward(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        int[] stride = (int[])savedState[0];
        int[] padding = (int[])savedState[1];

        var inputGrad = engine.ConvTranspose2DBackwardInput(gradOutput, inputs[1],
            inputs[0].Shape.ToArray(), stride, padding);
        var kernelGrad = engine.ConvTranspose2DBackwardKernel(gradOutput, inputs[0],
            inputs[1].Shape.ToArray(), stride, padding);

        DifferentiableOps.AccumulateGrad(grads, inputs[0], inputGrad, engine);
        DifferentiableOps.AccumulateGrad(grads, inputs[1], kernelGrad, engine);
    }

    /// <summary>Mean backward (global): gradient is 1/n for all elements</summary>
    internal static void MeanBackward(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        T scale = numOps.Divide(gradOutput[0], numOps.FromDouble(inputs[0].Length));
        var grad = TensorPool<T>.Rent(inputs[0].Shape.ToArray());
        engine.TensorFill(grad, scale);
        DifferentiableOps.AccumulateGrad(grads, inputs[0], grad, engine);
    }

    // ──────────────────────────────────────────────────────────────
    // Batch 2: Bmm, Upsample, AdaptivePool, Tile, PReLU, Threshold
    // ──────────────────────────────────────────────────────────────

    /// <summary>Batched matmul backward: dA = grad @ B^T, dB = A^T @ grad</summary>
    internal static void BatchMatMulBackward(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        // For ND tensors, transpose the last 2 dimensions using TensorPermute
        int rank = inputs[1].Rank;
        if (rank >= 3)
        {
            // Build permutation that swaps last two dims
            var perm = new int[rank];
            for (int i = 0; i < rank - 2; i++) perm[i] = i;
            perm[rank - 2] = rank - 1;
            perm[rank - 1] = rank - 2;

            var bT = engine.TensorPermute(inputs[1], perm);
            var gradA = engine.BatchMatMul(gradOutput, bT);
            var aT = engine.TensorPermute(inputs[0], perm);
            var gradB = engine.BatchMatMul(aT, gradOutput);
            DifferentiableOps.AccumulateGrad(grads, inputs[0], gradA, engine);
            DifferentiableOps.AccumulateGrad(grads, inputs[1], gradB, engine);
        }
        else
        {
            // 2D case
            var bT = engine.TensorTranspose(inputs[1]);
            var gradA = engine.TensorMatMul(gradOutput, bT);
            var aT = engine.TensorTranspose(inputs[0]);
            var gradB = engine.TensorMatMul(aT, gradOutput);
            DifferentiableOps.AccumulateGrad(grads, inputs[0], gradA, engine);
            DifferentiableOps.AccumulateGrad(grads, inputs[1], gradB, engine);
        }
    }

    /// <summary>Upsample (nearest) backward: sum gradients from upsampled positions</summary>
    internal static void UpsampleBackward(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        int scaleH = (int)savedState[0];
        int scaleW = (int)savedState[1];
        var inputGrad = engine.UpsampleBackward(gradOutput, inputs[0].Shape.ToArray(), scaleH, scaleW);
        DifferentiableOps.AccumulateGrad(grads, inputs[0], inputGrad, engine);
    }

    /// <summary>AdaptiveAvgPool2D backward: distribute gradient evenly over adaptive windows</summary>
    internal static void AdaptiveAvgPool2DBackward(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var inShape = inputs[0].Shape.ToArray();
        int batch = inShape[0], channels = inShape[1], inH = inShape[2], inW = inShape[3];
        int outH = (int)savedState[0], outW = (int)savedState[1];

        var inputGrad = TensorPool<T>.RentZeroed(inShape);
        for (int b = 0; b < batch; b++)
        for (int c = 0; c < channels; c++)
        for (int oh = 0; oh < outH; oh++)
        {
            int hStart = (int)Math.Floor((double)oh * inH / outH);
            int hEnd = (int)Math.Ceiling((double)(oh + 1) * inH / outH);
            for (int ow = 0; ow < outW; ow++)
            {
                int wStart = (int)Math.Floor((double)ow * inW / outW);
                int wEnd = (int)Math.Ceiling((double)(ow + 1) * inW / outW);
                int count = (hEnd - hStart) * (wEnd - wStart);
                T g = numOps.Divide(gradOutput[((b * channels + c) * outH + oh) * outW + ow],
                    numOps.FromDouble(count));
                for (int ih = hStart; ih < hEnd; ih++)
                for (int iw = wStart; iw < wEnd; iw++)
                {
                    int idx = ((b * channels + c) * inH + ih) * inW + iw;
                    inputGrad[idx] = numOps.Add(inputGrad[idx], g);
                }
            }
        }
        DifferentiableOps.AccumulateGrad(grads, inputs[0], inputGrad, engine);
    }

    /// <summary>Tile backward: sum gradient from all repeated positions back to source</summary>
    internal static void TileBackward(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var inShape = inputs[0].Shape.ToArray();
        var outShape = gradOutput.Shape.ToArray();
        int totalOut = gradOutput.Length;

        var inputGrad = TensorPool<T>.RentZeroed(inShape);
        for (int flat = 0; flat < totalOut; flat++)
        {
            int remaining = flat;
            int srcIdx = 0;
            int srcStride = 1;
            for (int d = inShape.Length - 1; d >= 0; d--)
            {
                int coord = remaining % outShape[d];
                remaining /= outShape[d];
                srcIdx += (coord % inShape[d]) * srcStride;
                srcStride *= inShape[d];
            }
            if (srcIdx < 0 || srcIdx >= inputs[0].Length)
                throw new InvalidOperationException($"TileBackward: computed source index {srcIdx} is out of range [0, {inputs[0].Length}). This indicates a shape or stride computation bug.");
            inputGrad[srcIdx] = numOps.Add(inputGrad[srcIdx], gradOutput[flat]);
        }
        DifferentiableOps.AccumulateGrad(grads, inputs[0], inputGrad, engine);
    }

    /// <summary>PReLU backward: dx = grad if x >= 0, alpha*grad if x < 0; dalpha = x*grad where x < 0</summary>
    internal static void PReLUBackward(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        var (inputGrad, alphaGrad) = engine.PReLUBackward(gradOutput, inputs[0], inputs[1]);
        DifferentiableOps.AccumulateGrad(grads, inputs[0], inputGrad, engine);
        DifferentiableOps.AccumulateGrad(grads, inputs[1], alphaGrad, engine);
    }

    /// <summary>Threshold backward: gradient passes through where x > threshold</summary>
    internal static void ThresholdBackward(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        double threshold = (double)savedState[0];
        var grad = engine.ThresholdBackward(gradOutput, inputs[0], threshold);
        DifferentiableOps.AccumulateGrad(grads, inputs[0], grad, engine);
    }

    /// <summary>Flatten backward: reshape gradient back to input shape</summary>
    internal static void FlattenBackward(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        var grad = engine.Reshape(gradOutput, inputs[0].Shape.ToArray());
        DifferentiableOps.AccumulateGrad(grads, inputs[0], grad, engine);
    }

    /// <summary>Narrow/slice backward: put gradient back at sliced position, zero elsewhere</summary>
    internal static void NarrowBackward(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        int dim = (int)savedState[0];
        int start = (int)savedState[1];
        int length = (int)savedState[2];
        var inShape = inputs[0].Shape.ToArray();
        var inputGrad = TensorPool<T>.RentZeroed(inShape);

        int outerSize = 1, innerSize = 1;
        int dimSize = inShape[dim];
        for (int i = 0; i < dim; i++) outerSize *= inShape[i];
        for (int i = dim + 1; i < inShape.Length; i++) innerSize *= inShape[i];

        for (int outer = 0; outer < outerSize; outer++)
        for (int d = 0; d < length; d++)
        for (int inner = 0; inner < innerSize; inner++)
        {
            int srcFlat = (outer * dimSize + (start + d)) * innerSize + inner;
            int gradFlat = (outer * length + d) * innerSize + inner;
            if (srcFlat < 0 || srcFlat >= inputGrad.Length)
                throw new InvalidOperationException($"NarrowBackward: source index {srcFlat} out of range [0, {inputGrad.Length}).");
            if (gradFlat < 0 || gradFlat >= gradOutput.Length)
                throw new InvalidOperationException($"NarrowBackward: gradient index {gradFlat} out of range [0, {gradOutput.Length}).");
            inputGrad[srcFlat] = gradOutput[gradFlat];
        }
        DifferentiableOps.AccumulateGrad(grads, inputs[0], inputGrad, engine);
    }

    /// <summary>MSE loss backward: d(MSE)/d(pred) = 2*(pred-target)/n</summary>
    internal static void MSELossBackward(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var predictions = inputs[0];
        var targets = inputs[1];
        T scale = numOps.FromDouble(2.0 * numOps.ToDouble(gradOutput[0]) / predictions.Length);
        var diff = engine.TensorSubtract(predictions, targets);
        var grad = engine.TensorMultiplyScalar(diff, scale);
        DifferentiableOps.AccumulateGrad(grads, predictions, grad, engine);
    }

    /// <summary>BCE with logits loss backward</summary>
    internal static void BCEWithLogitsLossBackward(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var logits = inputs[0];
        var targets = inputs[1];
        // sigmoid(logits) - targets, scaled by gradOutput[0] / n
        var sig = engine.TensorSigmoid(logits);
        var diff = engine.TensorSubtract(sig, targets);
        T scale = numOps.FromDouble(numOps.ToDouble(gradOutput[0]) / logits.Length);
        var grad = engine.TensorMultiplyScalar(diff, scale);
        DifferentiableOps.AccumulateGrad(grads, logits, grad, engine);
    }

    /// <summary>Cross-entropy loss backward</summary>
    internal static void CrossEntropyLossBackward(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var logits = inputs[0];
        var targets = inputs[1];
        int n = logits.Shape[0];
        // Cross-entropy gradient = (softmax(logits) - targets) / n * gradOutput[0]
        int axis = logits.Rank - 1;
        var softmax = engine.TensorSoftmax(logits, axis);
        var diff = engine.TensorSubtract(softmax, targets);
        T scaleT = numOps.FromDouble(numOps.ToDouble(gradOutput[0]) / n);
        var grad = engine.TensorMultiplyScalar(diff, scaleT);
        DifferentiableOps.AccumulateGrad(grads, logits, grad, engine);
    }

    // ──────────────────────────────────────────────────────────────
    // Batch 3: Remaining ops for full PyTorch parity
    // ──────────────────────────────────────────────────────────────

    /// <summary>IndexSelect backward: scatter gradient back along selected dimension</summary>
    internal static void IndexSelectBackward(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        int[] indices = (int[])savedState[0];
        int dim = (int)savedState[1];
        var inShape = inputs[0].Shape.ToArray();
        var inputGrad = TensorPool<T>.RentZeroed(inShape);

        int outerSize = 1, innerSize = 1;
        int dimSize = inShape[dim];
        for (int i = 0; i < dim; i++) outerSize *= inShape[i];
        for (int i = dim + 1; i < inShape.Length; i++) innerSize *= inShape[i];

        for (int outer = 0; outer < outerSize; outer++)
        for (int idx = 0; idx < indices.Length; idx++)
        {
            int srcDimIdx = indices[idx];
            for (int inner = 0; inner < innerSize; inner++)
            {
                int srcFlat = (outer * dimSize + srcDimIdx) * innerSize + inner;
                int gradFlat = (outer * indices.Length + idx) * innerSize + inner;
                if (srcFlat < 0 || srcFlat >= inputGrad.Length)
                    throw new InvalidOperationException($"IndexSelectBackward: source index {srcFlat} out of range [0, {inputGrad.Length}).");
                if (gradFlat < 0 || gradFlat >= gradOutput.Length)
                    throw new InvalidOperationException($"IndexSelectBackward: gradient index {gradFlat} out of range [0, {gradOutput.Length}).");
                inputGrad[srcFlat] = numOps.Add(inputGrad[srcFlat], gradOutput[gradFlat]);
            }
        }
        DifferentiableOps.AccumulateGrad(grads, inputs[0], inputGrad, engine);
    }

    /// <summary>L1 loss backward</summary>
    internal static void L1LossBackward(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var target = inputs[1];
        double scale = numOps.ToDouble(gradOutput[0]) / inputs[0].Length;
        var diff = engine.TensorSubtract(inputs[0], target);
        var signDiff = engine.TensorSign(diff);
        T scaleT = numOps.FromDouble(scale);
        var grad = engine.TensorMultiplyScalar(signDiff, scaleT);
        DifferentiableOps.AccumulateGrad(grads, inputs[0], grad, engine);
    }

    /// <summary>Huber loss backward: quadratic for small errors, linear for large</summary>
    internal static void HuberLossBackward(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var target = inputs[1];
        double delta = (double)savedState[0];
        T scaleT = numOps.FromDouble(numOps.ToDouble(gradOutput[0]) / inputs[0].Length);
        // Huber gradient = clamp(diff, -delta, delta) * scale
        // clamp gives diff when |d|<=delta, delta*sign(d) when |d|>delta — exactly the Huber derivative
        var diff = engine.TensorSubtract(inputs[0], target);
        var clamped = engine.TensorClamp(diff, numOps.FromDouble(-delta), numOps.FromDouble(delta));
        var grad = engine.TensorMultiplyScalar(clamped, scaleT);
        DifferentiableOps.AccumulateGrad(grads, inputs[0], grad, engine);
    }

    /// <summary>KL divergence loss backward</summary>
    internal static void KLDivLossBackward(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var target = inputs[1];
        // d(KL)/d(input) = -target / N * gradOutput[0]
        T scaleT = numOps.FromDouble(-numOps.ToDouble(gradOutput[0]) / inputs[0].Length);
        var grad = engine.TensorMultiplyScalar(target, scaleT);
        DifferentiableOps.AccumulateGrad(grads, inputs[0], grad, engine);
    }

    /// <summary>NLL loss backward</summary>
    internal static void NLLLossBackward(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        // NLL backward is inherently index-based (scatter -scale at target class positions).
        // No element-wise engine op can express this — same as PyTorch's implementation.
        var numOps = MathHelper.GetNumericOperations<T>();
        var targetTensor = inputs[1];
        int n = inputs[0].Shape[0];
        int c = inputs[0].Length / n;
        T negScale = numOps.FromDouble(-numOps.ToDouble(gradOutput[0]) / n);
        var dx = TensorPool<T>.RentZeroed(inputs[0].Shape.ToArray());
        engine.TensorFill(dx, numOps.Zero);
        for (int b = 0; b < n; b++)
        {
            int target = (int)numOps.ToDouble(targetTensor[b]);
            if (target >= 0 && target < c)
                dx[b * c + target] = negScale;
        }
        DifferentiableOps.AccumulateGrad(grads, inputs[0], dx, engine);
    }

    /// <summary>UpsampleBilinear backward</summary>
    internal static void UpsampleBilinearBackward(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var inShape = inputs[0].Shape.ToArray();
        int h = inShape[^2], w = inShape[^1];
        var goShape = gradOutput.Shape.ToArray();
        int outH = goShape[^2], outW = goShape[^1];
        int batchChannels = 1;
        for (int i = 0; i < inShape.Length - 2; i++) batchChannels *= inShape[i];

        var inputGrad = TensorPool<T>.RentZeroed(inShape);
        for (int bc = 0; bc < batchChannels; bc++)
        for (int oy = 0; oy < outH; oy++)
        {
            // Match forward's coordinate mapping: (oy + 0.5) * h / outH - 0.5
            double srcY = (oy + 0.5) * h / outH - 0.5;
            int y0 = Math.Max(0, (int)Math.Floor(srcY));
            int y1 = Math.Min(y0 + 1, h - 1);
            double fy = srcY - y0;
            for (int ox = 0; ox < outW; ox++)
            {
                double srcX = (ox + 0.5) * w / outW - 0.5;
                int x0 = Math.Max(0, (int)Math.Floor(srcX));
                int x1 = Math.Min(x0 + 1, w - 1);
                double fx = srcX - x0;
                double g = numOps.ToDouble(gradOutput[bc * outH * outW + oy * outW + ox]);
                int baseIdx = bc * h * w;
                inputGrad[baseIdx + y0 * w + x0] = numOps.Add(inputGrad[baseIdx + y0 * w + x0], numOps.FromDouble(g * (1 - fy) * (1 - fx)));
                inputGrad[baseIdx + y0 * w + x1] = numOps.Add(inputGrad[baseIdx + y0 * w + x1], numOps.FromDouble(g * (1 - fy) * fx));
                inputGrad[baseIdx + y1 * w + x0] = numOps.Add(inputGrad[baseIdx + y1 * w + x0], numOps.FromDouble(g * fy * (1 - fx)));
                inputGrad[baseIdx + y1 * w + x1] = numOps.Add(inputGrad[baseIdx + y1 * w + x1], numOps.FromDouble(g * fy * fx));
            }
        }
        DifferentiableOps.AccumulateGrad(grads, inputs[0], inputGrad, engine);
    }

    /// <summary>ConstantPad backward: extract unpadded region from gradient</summary>
    internal static void ConstantPadBackward(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        int[] padding = (int[])savedState[0];
        var inShape = inputs[0].Shape.ToArray();
        var inputGrad = TensorPool<T>.RentZeroed(inShape);
        var outShape = gradOutput.Shape.ToArray();

        int totalSrc = inputs[0].Length;
        for (int flat = 0; flat < totalSrc; flat++)
        {
            int remaining = flat;
            var coords = new int[inShape.Length];
            for (int d = inShape.Length - 1; d >= 0; d--)
            {
                coords[d] = remaining % inShape[d];
                remaining /= inShape[d];
            }
            int dstIdx = 0;
            int stride = 1;
            for (int d = inShape.Length - 1; d >= 0; d--)
            {
                int padBefore = d * 2 < padding.Length ? padding[d * 2] : 0;
                dstIdx += (coords[d] + padBefore) * stride;
                stride *= outShape[d];
            }
            if (dstIdx < gradOutput.Length)
                inputGrad[flat] = gradOutput[dstIdx];
        }
        DifferentiableOps.AccumulateGrad(grads, inputs[0], inputGrad, engine);
    }

    /// <summary>RReLU backward</summary>
    internal static void RReLUBackward(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var noise = (Tensor<T>)savedState[0];
        // RReLU: x >= 0 ? grad : grad * noise
        // Build positive mask: ceil(clamp((sign(x)+1)/2, 0, 1)) gives 1 for x>=0, 0 for x<0
        var rawSign = engine.TensorSign(inputs[0]);
        var shifted = engine.TensorAddScalar(rawSign, numOps.One);
        var halved = engine.TensorMultiplyScalar(shifted, numOps.FromDouble(0.5));
        var sign = engine.TensorCeiling(engine.TensorClamp(halved, numOps.Zero, numOps.One));
        // posGrad = grad * sign(relu(x))  (gradient where x >= 0)
        var posGrad = engine.TensorMultiply(gradOutput, sign);
        // negMask = 1 - sign  (1 where x < 0, 0 where x >= 0)
        var ones = engine.TensorAddScalar(engine.TensorMultiplyScalar(sign, numOps.Zero), numOps.One);
        var negMask = engine.TensorSubtract(ones, sign);
        // negGrad = grad * noise * negMask
        var negGrad = engine.TensorMultiply(engine.TensorMultiply(gradOutput, noise), negMask);
        var grad = engine.TensorAdd(posGrad, negGrad);
        DifferentiableOps.AccumulateGrad(grads, inputs[0], grad, engine);
    }

    /// <summary>LogSoftmax backward</summary>
    internal static void LogSoftmaxBackward(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        // d(log_softmax)/dx = gradOutput - softmax * sum(gradOutput)
        // softmax = exp(log_softmax) = exp(output)
        var softmax = engine.TensorExp(output);
        // For each row, compute sum(gradOutput) and subtract softmax * sum
        // This is a per-row operation. Use engine ops for the computation.
        var numOps = MathHelper.GetNumericOperations<T>();
        int lastDim = inputs[0].Shape[^1];
        int outerSize = inputs[0].Length / lastDim;
        var dx = TensorPool<T>.RentZeroed(inputs[0].Shape.ToArray());

        for (int outer = 0; outer < outerSize; outer++)
        {
            int offset = outer * lastDim;
            T sumGrad = numOps.Zero;
            for (int d = 0; d < lastDim; d++)
                sumGrad = numOps.Add(sumGrad, gradOutput[offset + d]);
            for (int d = 0; d < lastDim; d++)
                dx[offset + d] = numOps.Subtract(gradOutput[offset + d],
                    numOps.Multiply(softmax[offset + d], sumGrad));
        }
        DifferentiableOps.AccumulateGrad(grads, inputs[0], dx, engine);
    }

    /// <summary>Split backward: scatter chunk gradient back to correct position in input gradient</summary>
    internal static void SplitBackward(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        int axis = (int)savedState[0];
        int start = (int)savedState[1];
        int[] originalShape = (int[])savedState[2];
        var numOps = MathHelper.GetNumericOperations<T>();

        // Create a zero gradient with original input shape, then copy chunk into correct position
        var grad = TensorPool<T>.RentZeroed(originalShape);
        engine.TensorFill(grad, numOps.Zero);
        var startArr = new int[originalShape.Length];
        startArr[axis] = start;
        engine.TensorSetSlice(grad, gradOutput, startArr);
        DifferentiableOps.AccumulateGrad(grads, inputs[0], grad, engine);
    }

    /// <summary>AvgPool1D backward</summary>
    internal static void AvgPool1DBackward(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        int kernelSize = (int)savedState[0];
        int stride = (int)savedState[1];
        var inShape = inputs[0].Shape.ToArray();
        int batch = inShape[0], channels = inShape[1], length = inShape[2];
        int outLength = (length - kernelSize) / stride + 1;
        T scale = numOps.FromDouble(1.0 / kernelSize);

        var inputGrad = TensorPool<T>.RentZeroed(inShape);
        for (int b = 0; b < batch; b++)
        for (int c = 0; c < channels; c++)
        for (int o = 0; o < outLength; o++)
        {
            T g = numOps.Multiply(gradOutput[(b * channels + c) * outLength + o], scale);
            int start = o * stride;
            for (int k = 0; k < kernelSize; k++)
                inputGrad[(b * channels + c) * length + start + k] =
                    numOps.Add(inputGrad[(b * channels + c) * length + start + k], g);
        }
        DifferentiableOps.AccumulateGrad(grads, inputs[0], inputGrad, engine);
    }

    /// <summary>MaxPool1D backward: route gradient to max element via argmax</summary>
    internal static void MaxPool1DBackward(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        // Argmax indices are flat offsets — scatter gradient to those positions
        // Use engine ops: create zeros, then add gradient at argmax positions
        var numOps = MathHelper.GetNumericOperations<T>();
        int[] argmax = (int[])savedState[0];
        var inputGrad = TensorPool<T>.RentZeroed(inputs[0].Shape.ToArray());
        var gradData = gradOutput.GetDataArray();
        var resultData = inputGrad.GetDataArray();
        for (int i = 0; i < argmax.Length; i++)
        {
            int idx = argmax[i];
            if (idx >= 0 && idx < resultData.Length)
                resultData[idx] = numOps.Add(resultData[idx], gradData[i]);
        }
        DifferentiableOps.AccumulateGrad(grads, inputs[0], inputGrad, engine);
    }

    /// <summary>Cosine similarity backward</summary>
    internal static void CosineSimilarityBackward(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var a = inputs[0];
        var b = inputs[1];
        double g = numOps.ToDouble(gradOutput[0]);

        // Compute norms and dot product using engine ops
        var aa = engine.TensorMultiply(a, a);
        var bb = engine.TensorMultiply(b, b);
        var ab = engine.TensorMultiply(a, b);
        double aNormSq = numOps.ToDouble(engine.TensorSum(aa));
        double bNormSq = numOps.ToDouble(engine.TensorSum(bb));
        double dot = numOps.ToDouble(engine.TensorSum(ab));
        double eps = 1e-8;
        double normProd = Math.Sqrt(aNormSq + eps) * Math.Sqrt(bNormSq + eps);
        double cosim = dot / normProd;

        // gradA = g * (b / normProd - a * cosim / (||a||^2 + eps))
        var bScaled = engine.TensorMultiplyScalar(b, numOps.FromDouble(g / normProd));
        var aScaled = engine.TensorMultiplyScalar(a, numOps.FromDouble(g * cosim / (aNormSq + eps)));
        var gradA = engine.TensorSubtract(bScaled, aScaled);
        // gradB = g * (a / normProd - b * cosim / (||b||^2 + eps))
        var aScaled2 = engine.TensorMultiplyScalar(a, numOps.FromDouble(g / normProd));
        var bScaled2 = engine.TensorMultiplyScalar(b, numOps.FromDouble(g * cosim / (bNormSq + eps)));
        var gradB = engine.TensorSubtract(aScaled2, bScaled2);
        DifferentiableOps.AccumulateGrad(grads, inputs[0], gradA, engine);
        DifferentiableOps.AccumulateGrad(grads, inputs[1], gradB, engine);
    }

    /// <summary>Reciprocal backward: d(1/x)/dx = -1/x^2</summary>
    internal static void ReciprocalBackward(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        var grad = engine.ReciprocalBackward(gradOutput, output);
        DifferentiableOps.AccumulateGrad(grads, inputs[0], grad, engine);
    }

    /// <summary>Sign backward: zero gradient (piecewise constant)</summary>
    internal static void SignBackward(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        // Sign has zero gradient everywhere
        var zero = TensorPool<T>.RentZeroed(inputs[0].Shape.ToArray());
        DifferentiableOps.AccumulateGrad(grads, inputs[0], zero, engine);
    }

    /// <summary>Floor/Ceil/Round backward: straight-through estimator (pass gradient through)</summary>
    internal static void StraightThroughBackward(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        DifferentiableOps.AccumulateGrad(grads, inputs[0], gradOutput, engine);
    }

    /// <summary>Mish backward: d/dx[x*tanh(softplus(x))]</summary>
    internal static void MishBackwardFull(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        var grad = engine.MishBackward(gradOutput, inputs[0]);
        DifferentiableOps.AccumulateGrad(grads, inputs[0], grad, engine);
    }

    // ──────────────────────────────────────────────────────────────
    // Missing Phase 1 ops
    // ──────────────────────────────────────────────────────────────

    /// <summary>Stack backward: split gradient along the stacked axis</summary>
    internal static void StackBackward(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        int axis = (int)savedState[0];
        int numTensors = inputs.Length;
        // Split gradOutput along the stacked axis
        int sliceSize = gradOutput.Shape[axis] / numTensors;
        for (int i = 0; i < numTensors; i++)
        {
            // Slice along axis, then squeeze the axis dimension
            var sliced = gradOutput.Slice(axis, i * sliceSize, (i + 1) * sliceSize);
            var squeezed = engine.TensorSqueeze(sliced, axis);
            DifferentiableOps.AccumulateGrad(grads, inputs[i], squeezed, engine);
        }
    }

    /// <summary>Var backward: d(var(x))/dx_i = 2*(x_i - mean) / n</summary>
    internal static void VarBackward(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        // Compute mean via engine (GPU-resident)
        var mean = engine.TensorMeanDiff(inputs[0]);
        var grad = engine.VarBackward(gradOutput, inputs[0], mean, Array.Empty<int>());
        DifferentiableOps.AccumulateGrad(grads, inputs[0], grad, engine);
    }

    /// <summary>Std backward: d(std(x))/dx_i = (x_i - mean) / (n * std)</summary>
    internal static void StdBackward(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        var mean = engine.TensorMeanDiff(inputs[0]);
        var grad = engine.StdBackward(gradOutput, inputs[0], mean, output, Array.Empty<int>());
        DifferentiableOps.AccumulateGrad(grads, inputs[0], grad, engine);
    }

    /// <summary>Square backward: d(x^2)/dx = 2*x</summary>
    internal static void SquareBackward(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var two = numOps.FromDouble(2.0);
        var scaled = engine.TensorMultiplyScalar(inputs[0], two);
        var grad = engine.TensorMultiply(gradOutput, scaled);
        DifferentiableOps.AccumulateGrad(grads, inputs[0], grad, engine);
    }

    /// <summary>LogSumExp backward: d(log(sum(exp(x))))/dx_i = exp(x_i) / sum(exp(x)) = softmax(x)_i</summary>
    internal static void LogSumExpBackward(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        // dx = gradOutput * exp(x - lse) = gradOutput * softmax(x)
        var shifted = engine.TensorBroadcastSubtract(inputs[0], output);
        var softmax = engine.TensorExp(shifted);
        var grad = engine.TensorBroadcastMultiply(gradOutput, softmax);
        DifferentiableOps.AccumulateGrad(grads, inputs[0], grad, engine);
    }

    /// <summary>Norm backward: d(||x||)/dx_i = x_i / ||x||</summary>
    internal static void NormBackward(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        // dx = gradOutput * x / norm — use broadcast ops for scalar norm
        var numOps = MathHelper.GetNumericOperations<T>();
        var scaledInput = engine.TensorBroadcastMultiply(gradOutput, inputs[0]);
        var normClamped = engine.TensorClamp(output, numOps.FromDouble(1e-12), numOps.FromDouble(1e30));
        var grad = engine.TensorBroadcastDivide(scaledInput, normClamped);
        DifferentiableOps.AccumulateGrad(grads, inputs[0], grad, engine);
    }

    /// <summary>AdaptiveMaxPool2D backward: route gradient to argmax positions</summary>
    internal static void AdaptiveMaxPool2DBackward(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        // Argmax indices are flat offsets — scatter gradient to those positions
        var numOps = MathHelper.GetNumericOperations<T>();
        int[] argmax = (int[])savedState[0];
        var inputGrad = TensorPool<T>.RentZeroed(inputs[0].Shape.ToArray());
        var gradData = gradOutput.GetDataArray();
        var resultData = inputGrad.GetDataArray();
        for (int i = 0; i < argmax.Length; i++)
        {
            int idx = argmax[i];
            if (idx >= 0 && idx < resultData.Length)
                resultData[idx] = numOps.Add(resultData[idx], gradData[i]);
        }
        DifferentiableOps.AccumulateGrad(grads, inputs[0], inputGrad, engine);
    }

    /// <summary>Where backward: gradient flows through the selected branch</summary>
    internal static void WhereBackward(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        // Convert bool[] condition to float tensor for GPU dispatch
        if (savedState[0] is Tensor<T> condTensor)
        {
            // Already a tensor — use engine multiply for routing
            var gradX = engine.TensorMultiply(gradOutput, condTensor);
            var ones = engine.TensorAddScalar(engine.TensorMultiplyScalar(condTensor, numOps.Zero), numOps.One);
            var invCond = engine.TensorSubtract(ones, condTensor);
            var gradY = engine.TensorMultiply(gradOutput, invCond);
            DifferentiableOps.AccumulateGrad(grads, inputs[0], gradX, engine);
            DifferentiableOps.AccumulateGrad(grads, inputs[1], gradY, engine);
        }
        else
        {
            var condition = (bool[])savedState[0];
            var condData = new T[condition.Length];
            for (int i = 0; i < condition.Length; i++)
                condData[i] = condition[i] ? numOps.One : numOps.Zero;
            var condT = new Tensor<T>(condData, inputs[0].Shape.ToArray());
            var gradX = engine.TensorMultiply(gradOutput, condT);
            var invCond = engine.TensorSubtract(
                engine.TensorAddScalar(engine.TensorMultiplyScalar(condT, numOps.Zero), numOps.One), condT);
            var gradY = engine.TensorMultiply(gradOutput, invCond);
            DifferentiableOps.AccumulateGrad(grads, inputs[0], gradX, engine);
            DifferentiableOps.AccumulateGrad(grads, inputs[1], gradY, engine);
        }
    }

    /// <summary>MaskedFill backward: zero gradient where mask is true</summary>
    internal static void MaskedFillBackward(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        if (savedState[0] is Tensor<T> maskTensor)
        {
            // Tensor mask — use engine multiply: grad * (1 - mask)
            var ones = engine.TensorAddScalar(engine.TensorMultiplyScalar(maskTensor, numOps.Zero), numOps.One);
            var invMask = engine.TensorSubtract(ones, engine.TensorSign(engine.TensorAbs(maskTensor)));
            var grad = engine.TensorMultiply(gradOutput, invMask);
            DifferentiableOps.AccumulateGrad(grads, inputs[0], grad, engine);
        }
        else
        {
            var mask = (bool[])savedState[0];
            var maskData = new T[mask.Length];
            for (int i = 0; i < mask.Length; i++)
                maskData[i] = mask[i] ? numOps.One : numOps.Zero;
            var maskT = new Tensor<T>(maskData, inputs[0].Shape.ToArray());
            var ones = engine.TensorAddScalar(engine.TensorMultiplyScalar(maskT, numOps.Zero), numOps.One);
            var invMask = engine.TensorSubtract(ones, maskT);
            var grad = engine.TensorMultiply(gradOutput, invMask);
            DifferentiableOps.AccumulateGrad(grads, inputs[0], grad, engine);
        }
    }

    /// <summary>Element-wise max backward: gradient flows to whichever input was larger.
    /// Ties (a == b) route gradient to inputs[0] (asymmetric subgradient, consistent with PyTorch).</summary>
    internal static void MaxBackward(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        // Use saved input clones if available (safe against in-place mutation)
        var a = savedState is { Length: >= 1 } && savedState[0] is Tensor<T> sa ? sa : inputs[0];
        var numOps = MathHelper.GetNumericOperations<T>();
        var zeros = engine.TensorMultiplyScalar(gradOutput, numOps.Zero);
        var diff = engine.TensorSubtract(output, a);
        var absDiff = engine.TensorAbs(diff);
        var maskSign = engine.TensorSign(absDiff);
        var aOnes = engine.TensorAddScalar(zeros, numOps.One);
        var aMask = engine.TensorSubtract(aOnes, maskSign);
        var gradA = engine.TensorMultiply(gradOutput, aMask);
        var bMask = engine.TensorSubtract(aOnes, aMask);
        var gradB = engine.TensorMultiply(gradOutput, bMask);
        DifferentiableOps.AccumulateGrad(grads, inputs[0], gradA, engine);
        DifferentiableOps.AccumulateGrad(grads, inputs[1], gradB, engine);
    }

    /// <summary>Element-wise min backward: gradient flows to whichever input was smaller.
    /// Ties (a == b) route gradient to inputs[0] (asymmetric subgradient, consistent with PyTorch).</summary>
    internal static void MinBackward(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        var a = savedState is { Length: >= 1 } && savedState[0] is Tensor<T> sa ? sa : inputs[0];
        var numOps = MathHelper.GetNumericOperations<T>();
        var zeros = engine.TensorMultiplyScalar(gradOutput, numOps.Zero);
        var diff = engine.TensorSubtract(output, a);
        var absDiff = engine.TensorAbs(diff);
        var maskSign = engine.TensorSign(absDiff);
        var aOnes = engine.TensorAddScalar(zeros, numOps.One);
        var aMask = engine.TensorSubtract(aOnes, maskSign);
        var gradA = engine.TensorMultiply(gradOutput, aMask);
        var bMask = engine.TensorSubtract(aOnes, aMask);
        var gradB = engine.TensorMultiply(gradOutput, bMask);
        DifferentiableOps.AccumulateGrad(grads, inputs[0], gradA, engine);
        DifferentiableOps.AccumulateGrad(grads, inputs[1], gradB, engine);
    }

    // ══════════════════════════════════════════════════════════════════
    // Fused backward kernels — combine sequential backward ops into
    // single passes to reduce intermediate allocations and data traversals.
    // ══════════════════════════════════════════════════════════════════

    /// <summary>
    /// Fused backward for MatMul + Bias Add + ReLU.
    /// Combines three backward passes into one:
    /// 1. Apply ReLU mask to gradOutput
    /// 2. Compute MatMul backward (gradA = maskedGrad @ B^T, gradB = A^T @ maskedGrad)
    /// 3. Bias gradient = sum of maskedGrad along batch dimension
    /// Saves 2 intermediate tensor allocations vs separate backward calls.
    /// </summary>
    /// <remarks>
    /// savedState[0] = pre-activation tensor (MatMul + Bias output, before ReLU)
    /// inputs[0] = input to MatMul (A)
    /// inputs[1] = weight matrix (B)
    /// inputs[2] = bias vector
    /// </remarks>
    internal static void FusedMatMulAddReLUBackward(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        // Step 1: Apply ReLU mask — gradient is zero where pre-activation was <= 0
        var preActivation = (Tensor<T>)savedState[0];
        var maskedGrad = engine.ReluBackward(gradOutput, preActivation);

        // Step 2: MatMul backward with the already-masked gradient
        var bT = engine.TensorTranspose(inputs[1]);
        var gradA = engine.TensorMatMul(maskedGrad, bT);

        var aT = engine.TensorTranspose(inputs[0]);
        var gradB = engine.TensorMatMul(aT, maskedGrad);

        // Step 3: Bias gradient = sum of maskedGrad along batch dimension (axis 0)
        var gradBias = engine.ReduceSum(maskedGrad, new[] { 0 }, keepDims: false);

        DifferentiableOps.AccumulateGrad(grads, inputs[0], gradA, engine);
        DifferentiableOps.AccumulateGrad(grads, inputs[1], gradB, engine);
        DifferentiableOps.AccumulateGrad(grads, inputs[2], gradBias, engine);
    }

    /// <summary>
    /// Fused backward for MatMul + Bias Add + Sigmoid.
    /// </summary>
    internal static void FusedMatMulAddSigmoidBackward(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        // Sigmoid backward: grad * sigmoid(x) * (1 - sigmoid(x)) = grad * output * (1 - output)
        var maskedGrad = engine.SigmoidBackward(gradOutput, output);

        var bT = engine.TensorTranspose(inputs[1]);
        var gradA = engine.TensorMatMul(maskedGrad, bT);

        var aT = engine.TensorTranspose(inputs[0]);
        var gradB = engine.TensorMatMul(aT, maskedGrad);

        var gradBias = engine.ReduceSum(maskedGrad, new[] { 0 }, keepDims: false);

        DifferentiableOps.AccumulateGrad(grads, inputs[0], gradA, engine);
        DifferentiableOps.AccumulateGrad(grads, inputs[1], gradB, engine);
        DifferentiableOps.AccumulateGrad(grads, inputs[2], gradBias, engine);
    }

    /// <summary>
    /// Fused backward for MatMul + Bias Add + Tanh.
    /// </summary>
    internal static void FusedMatMulAddTanhBackward(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        // Tanh backward: grad * (1 - tanh(x)^2) = grad * (1 - output^2)
        var maskedGrad = engine.TanhBackward(gradOutput, output);

        var bT = engine.TensorTranspose(inputs[1]);
        var gradA = engine.TensorMatMul(maskedGrad, bT);

        var aT = engine.TensorTranspose(inputs[0]);
        var gradB = engine.TensorMatMul(aT, maskedGrad);

        var gradBias = engine.ReduceSum(maskedGrad, new[] { 0 }, keepDims: false);

        DifferentiableOps.AccumulateGrad(grads, inputs[0], gradA, engine);
        DifferentiableOps.AccumulateGrad(grads, inputs[1], gradB, engine);
        DifferentiableOps.AccumulateGrad(grads, inputs[2], gradBias, engine);
    }

    /// <summary>
    /// Fused backward for MatMul + Bias Add + GELU.
    /// </summary>
    internal static void FusedMatMulAddGELUBackward(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        var preActivation = (Tensor<T>)savedState[0];
        var maskedGrad = engine.GeluBackward(gradOutput, preActivation);

        var bT = engine.TensorTranspose(inputs[1]);
        var gradA = engine.TensorMatMul(maskedGrad, bT);

        var aT = engine.TensorTranspose(inputs[0]);
        var gradB = engine.TensorMatMul(aT, maskedGrad);

        var gradBias = engine.ReduceSum(maskedGrad, new[] { 0 }, keepDims: false);

        DifferentiableOps.AccumulateGrad(grads, inputs[0], gradA, engine);
        DifferentiableOps.AccumulateGrad(grads, inputs[1], gradB, engine);
        DifferentiableOps.AccumulateGrad(grads, inputs[2], gradBias, engine);
    }

    /// <summary>
    /// Fused backward for MatMul + Bias Add + Swish/SiLU.
    /// </summary>
    internal static void FusedMatMulAddSwishBackward(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        var preActivation = (Tensor<T>)savedState[0];
        // Swish backward: grad * (sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x)))
        var sigmoid = engine.TensorSigmoid(preActivation);
        var numOps = MathHelper.GetNumericOperations<T>();
        var oneMinusSigmoid = engine.ScalarMinusTensor(numOps.One, sigmoid);
        var xSigmoidDerivative = engine.TensorMultiply(
            preActivation, engine.TensorMultiply(sigmoid, oneMinusSigmoid));
        var swishDerivative = engine.TensorAdd(sigmoid, xSigmoidDerivative);
        var maskedGrad = engine.TensorMultiply(gradOutput, swishDerivative);

        var bT = engine.TensorTranspose(inputs[1]);
        var gradA = engine.TensorMatMul(maskedGrad, bT);

        var aT = engine.TensorTranspose(inputs[0]);
        var gradB = engine.TensorMatMul(aT, maskedGrad);

        var gradBias = engine.ReduceSum(maskedGrad, new[] { 0 }, keepDims: false);

        DifferentiableOps.AccumulateGrad(grads, inputs[0], gradA, engine);
        DifferentiableOps.AccumulateGrad(grads, inputs[1], gradB, engine);
        DifferentiableOps.AccumulateGrad(grads, inputs[2], gradBias, engine);
    }
}
