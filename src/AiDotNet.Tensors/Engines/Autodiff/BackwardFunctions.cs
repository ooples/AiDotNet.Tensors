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
        var numOps = MathHelper.GetNumericOperations<T>();
        var signTensor = new Tensor<T>(inputs[0].Shape.ToArray());
        for (int i = 0; i < inputs[0].Length; i++)
        {
            var val = inputs[0].GetFlat(i);
            if (numOps.GreaterThan(val, numOps.Zero))
                signTensor.SetFlat(i, numOps.One);
            else if (numOps.LessThan(val, numOps.Zero))
                signTensor.SetFlat(i, numOps.Negate(numOps.One));
            else
                signTensor.SetFlat(i, numOps.Zero);
        }
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
        var numOps = MathHelper.GetNumericOperations<T>();
        var min = (T)savedState[0];
        var max = (T)savedState[1];
        var mask = new Tensor<T>(inputs[0].Shape.ToArray());
        for (int i = 0; i < inputs[0].Length; i++)
        {
            var val = inputs[0].GetFlat(i);
            bool inRange = numOps.GreaterThan(val, min) && numOps.LessThan(val, max);
            bool atBound = numOps.Equals(val, min) || numOps.Equals(val, max);
            mask.SetFlat(i, (inRange || atBound) ? numOps.One : numOps.Zero);
        }
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
        var derivative = new Tensor<T>(output.Shape.ToArray());
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
        var grad = new Tensor<T>(inputShape);
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
            var result = new Tensor<T>(targetShape);
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
}
