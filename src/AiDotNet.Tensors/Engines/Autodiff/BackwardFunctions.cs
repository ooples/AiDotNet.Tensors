using AiDotNet.Tensors.Engines.Simd;
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
            CreateOnes(inputs[0]._shape, numOps), sig);
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
        var ones = CreateOnes(inputs[0]._shape, numOps);
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
        var derivative = TensorPool<T>.RentZeroed(output._shape);
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
        // BLAS fast path: compute transposed GEMMs into fresh buffers, then accumulate.
        // Allocates two float[] arrays but avoids materializing full transpose tensors.
        if (typeof(T) == typeof(float) && inputs[0].Rank == 2 && inputs[1].Rank == 2 && BlasProvider.IsAvailable)
        {
            var dCArr = (gradOutput as Tensor<float>)?.GetDataArray();
            var aArr = (inputs[0] as Tensor<float>)?.GetDataArray();
            var bArr = (inputs[1] as Tensor<float>)?.GetDataArray();

            if (dCArr is not null && aArr is not null && bArr is not null)
            {
                int M = gradOutput._shape[0], K = gradOutput._shape[1];
                int N_a = inputs[1]._shape[0]; // rows of B = cols of grad_A
                int N_b = gradOutput._shape[1]; // cols of grad_B

                // Pool gradient buffers via AutoTensorCache — zero allocation on steps 2+.
                var gradATensor = Helpers.AutoTensorCache.RentOrAllocate<T>(inputs[0]._shape);
                var gradBTensor = Helpers.AutoTensorCache.RentOrAllocate<T>(inputs[1]._shape);
                var gradAData = (float[])(object)gradATensor.GetDataArray();
                var gradBData = (float[])(object)gradBTensor.GetDataArray();
                // No Array.Clear needed — TryGemmEx uses beta=0 which overwrites C entirely

                bool okA = BlasProvider.TryGemmEx(M, N_a, K, dCArr, 0, K, false, bArr, 0, K, true, gradAData, 0, N_a);
                bool okB = BlasProvider.TryGemmEx(inputs[0]._shape[1], N_b, inputs[0]._shape[0],
                    aArr, 0, inputs[0]._shape[1], true, dCArr, 0, N_b, false, gradBData, 0, N_b);

                if (okA && okB)
                {
                    DifferentiableOps.AccumulateGrad(grads, inputs[0], gradATensor, engine);
                    DifferentiableOps.AccumulateGrad(grads, inputs[1], gradBTensor, engine);
                    return;
                }
                // Either GEMM refused — fall through to generic path
            }
        }

        // Fallback: transpose last two dims (works for all ranks)
        var bT = TransposeLastTwoDims(inputs[1], engine);
        var gradAFallback = engine.TensorMatMul(gradOutput, bT);

        var aT = TransposeLastTwoDims(inputs[0], engine);
        var gradBFallback = engine.TensorMatMul(aT, gradOutput);

        DifferentiableOps.AccumulateGrad(grads, inputs[0], gradAFallback, engine);
        DifferentiableOps.AccumulateGrad(grads, inputs[1], gradBFallback, engine);
    }

    /// <summary>d(A^T)/dA = transpose(grad)</summary>
    internal static void TransposeBackward(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        var grad = TransposeLastTwoDims(gradOutput, engine);
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
        var gradA = ReduceGradToShape(gradOutput, inputs[0]._shape, engine);
        var gradB = ReduceGradToShape(gradOutput, inputs[1]._shape, engine);
        DifferentiableOps.AccumulateGrad(grads, inputs[0], gradA, engine);
        DifferentiableOps.AccumulateGrad(grads, inputs[1], gradB, engine);
    }

    /// <summary>d(broadcast_sub(a,b))/da = reduce(grad), d/db = -reduce(grad)</summary>
    internal static void BroadcastSubtractBackward(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        var gradA = ReduceGradToShape(gradOutput, inputs[0]._shape, engine);
        var negGrad = engine.TensorNegate(gradOutput);
        var gradB = ReduceGradToShape(negGrad, inputs[1]._shape, engine);
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
        var gradA = ReduceGradToShape(fullGradA, inputs[0]._shape, engine);
        var gradB = ReduceGradToShape(fullGradB, inputs[1]._shape, engine);
        DifferentiableOps.AccumulateGrad(grads, inputs[0], gradA, engine);
        DifferentiableOps.AccumulateGrad(grads, inputs[1], gradB, engine);
    }

    /// <summary>d(broadcast_div(a,b))/da = reduce(grad/b), d/db = reduce(-grad*a/b^2)</summary>
    internal static void BroadcastDivideBackward(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        var fullGradA = engine.TensorBroadcastDivide(gradOutput, inputs[1]);
        var gradA = ReduceGradToShape(fullGradA, inputs[0]._shape, engine);

        var bSquared = engine.TensorBroadcastMultiply(inputs[1], inputs[1]);
        var negGradA = engine.TensorNegate(engine.TensorBroadcastMultiply(gradOutput, inputs[0]));
        var fullGradB = engine.TensorBroadcastDivide(negGradA, bSquared);
        var gradB = ReduceGradToShape(fullGradB, inputs[1]._shape, engine);

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
            gradOutput, inputs[1], inputs[0]._shape, stride, padding, dilation);
        var gradKernel = engine.Conv2DBackwardKernel(
            gradOutput, inputs[0], inputs[1]._shape, stride, padding, dilation);

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
            gradOutput, inputs[1], inputs[0]._shape, stride, padding, dilation);
        var gradKernel = engine.Conv1DBackwardKernel(
            gradOutput, inputs[0], inputs[1]._shape, stride, padding, dilation);

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
            gradOutput, inputs[1], inputs[0]._shape, stride, padding, dilation);
        var gradKernel = engine.Conv3DBackwardKernel(
            gradOutput, inputs[0], inputs[1]._shape, stride, padding, dilation);

        DifferentiableOps.AccumulateGrad(grads, inputs[0], gradInput, engine);
        DifferentiableOps.AccumulateGrad(grads, inputs[1], gradKernel, engine);
    }

    /// <summary>GridSample backward: uses engine.GridSampleBackwardInput and GridSampleBackwardGrid</summary>
    internal static void GridSampleBackward(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        // inputs[0] = input, inputs[1] = grid
        var gradInput = engine.GridSampleBackwardInput(gradOutput, inputs[1], inputs[0]._shape);
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
        var outputSize = new[] { inputs[0]._shape[2], inputs[0]._shape[3] };

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

        var grad = engine.MaxPool2DBackward(gradOutput, maxIndices, inputs[0]._shape, poolSize, stride);
        DifferentiableOps.AccumulateGrad(grads, inputs[0], grad, engine);
    }

    /// <summary>AvgPool2D backward: uses engine.AvgPool2DBackward</summary>
    internal static void AvgPool2DBackward(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        var poolSize = (int[])savedState[0];
        var stride = (int[])savedState[1];

        var grad = engine.AvgPool2DBackward(gradOutput, inputs[0]._shape, poolSize, stride);
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
        var inputShape = inputs[0]._shape;

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
        var grad = engine.ReduceMeanBackward(gradOutput, inputs[0]._shape, axes);
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
        var inputShape = inputs[0]._shape;
        var numOps = MathHelper.GetNumericOperations<T>();

        // Create zeros with input shape, set the sliced region to gradOutput.
        // TensorSetSlice returns a new tensor (does not modify in-place).
        var data = new T[inputShape.Aggregate(1, (a, b) => a * b)];
        var zeros = new Tensor<T>(data, inputShape);
        var grad = engine.TensorSetSlice(zeros, gradOutput, start);
        DifferentiableOps.AccumulateGrad(grads, inputs[0], grad, engine);
    }

    /// <summary>Gather backward: scatter grad to source positions</summary>
    internal static void GatherBackward(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        var indices = (Tensor<int>)savedState[0];
        var axis = (int)savedState[1];

        var grad = engine.ScatterAddBackward(gradOutput, indices, inputs[0]._shape, axis);
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
        var gradShape = grad._shape;

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
        var gradShape = grad._shape;
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
        var gradShape = gradOutput._shape;
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
            inputs[0]._shape, stride, padding);
        var kernelGrad = engine.ConvTranspose2DBackwardKernel(gradOutput, inputs[0],
            inputs[1]._shape, stride, padding);

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
        // Use RentUninitialized to go through TensorArena (cache-hot reuse)
        // instead of TensorPool (separate pool that misses arena + clears).
        // Use _shape directly to avoid Shape.ToArray() allocation.
        var grad = TensorAllocator.RentUninitialized<T>(inputs[0]._shape);
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
            var bT = TransposeLastTwoDims(inputs[1], engine);
            var gradA = engine.TensorMatMul(gradOutput, bT);
            var aT = TransposeLastTwoDims(inputs[0], engine);
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
        var inputGrad = engine.UpsampleBackward(gradOutput, inputs[0]._shape, scaleH, scaleW);
        DifferentiableOps.AccumulateGrad(grads, inputs[0], inputGrad, engine);
    }

    /// <summary>AdaptiveAvgPool2D backward: distribute gradient evenly over adaptive windows</summary>
    internal static void AdaptiveAvgPool2DBackward(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var inShape = inputs[0]._shape;
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
        var inShape = inputs[0]._shape;
        var outShape = gradOutput._shape;
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
        var grad = engine.Reshape(gradOutput, inputs[0]._shape);
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
        var inShape = inputs[0]._shape;
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
        int[] indices = savedState[0] is Tensor<int> indicesTensor
            ? indicesTensor.GetFlattenedData()
            : (int[])savedState[0];
        int dim = (int)savedState[1];
        var inShape = inputs[0]._shape;
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
        var dx = TensorPool<T>.RentZeroed(inputs[0]._shape);
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
        var inShape = inputs[0]._shape;
        int h = inShape[^2], w = inShape[^1];
        var goShape = gradOutput._shape;
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
        var inShape = inputs[0]._shape;
        var inputGrad = TensorPool<T>.RentZeroed(inShape);
        var outShape = gradOutput._shape;

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
        var dx = TensorPool<T>.RentZeroed(inputs[0]._shape);

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
        var inShape = inputs[0]._shape;
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
        var inputGrad = TensorPool<T>.RentZeroed(inputs[0]._shape);
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
        var zero = TensorPool<T>.RentZeroed(inputs[0]._shape);
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
        var inputGrad = TensorPool<T>.RentZeroed(inputs[0]._shape);
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
            var condT = new Tensor<T>(condData, inputs[0]._shape);
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
        else if (savedState[0] is Tensor<bool> boolMask)
        {
            var boolSpan = boolMask.AsSpan();
            var maskData = new T[boolSpan.Length];
            for (int i = 0; i < boolSpan.Length; i++)
                maskData[i] = boolSpan[i] ? numOps.Zero : numOps.One;
            var invMask = new Tensor<T>(maskData, inputs[0]._shape);
            var grad = engine.TensorMultiply(gradOutput, invMask);
            DifferentiableOps.AccumulateGrad(grads, inputs[0], grad, engine);
        }
        else
        {
            var mask = (bool[])savedState[0];
            var maskData = new T[mask.Length];
            for (int i = 0; i < mask.Length; i++)
                maskData[i] = mask[i] ? numOps.One : numOps.Zero;
            var maskT = new Tensor<T>(maskData, inputs[0]._shape);
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
    internal static unsafe void FusedMatMulAddReLUBackward(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        var preActivation = (Tensor<T>)savedState[0];

        // Fused BLAS path: ReLU mask + transposed GEMM in minimal allocations
        if (typeof(T) == typeof(float) && inputs[0].Rank == 2 && inputs[1].Rank == 2
            && BlasProvider.IsAvailable && gradOutput.IsContiguous && preActivation.IsContiguous)
        {
            int M = inputs[0]._shape[0]; // batch
            int K = inputs[0]._shape[1]; // in_features
            int N = inputs[1]._shape[1]; // out_features

            var gArr = (float[])(object)gradOutput.GetDataArray();
            var paArr = (float[])(object)preActivation.GetDataArray();
            var inArr = (float[])(object)inputs[0].GetDataArray();
            var wArr = (float[])(object)inputs[1].GetDataArray();

            // Step 1: Fused ReLU mask — pool the masked gradient buffer
            var maskedTensor = Helpers.AutoTensorCache.RentOrAllocate<float>(new[] { M, N });
            var maskedArr = maskedTensor.GetDataArray();
            fixed (float* pG = gArr, pPA = paArr, pM = maskedArr)
            {
                SimdKernels.ReluBackwardUnsafe(pG, pPA, pM, M * N);
            }

            // Step 2: gradInput = maskedGrad @ W^T  (transposed BLAS, pooled buffer)
            var gradInput = Helpers.AutoTensorCache.RentOrAllocate<T>(inputs[0]._shape);
            var gradInputArr = (float[])(object)gradInput.GetDataArray();
            if (!BlasProvider.TryGemmEx(M, K, N, maskedArr, 0, N, false, wArr, 0, N, true, gradInputArr, 0, K))
                goto fusedReluFallback;

            // Step 3: gradWeight = input^T @ maskedGrad  (transposed BLAS, pooled buffer)
            var gradWeight = Helpers.AutoTensorCache.RentOrAllocate<T>(inputs[1]._shape);
            var gradWeightArr = (float[])(object)gradWeight.GetDataArray();
            if (!BlasProvider.TryGemmEx(K, N, M, inArr, 0, K, true, maskedArr, 0, N, false, gradWeightArr, 0, N))
                goto fusedReluFallback;

            // Step 4: gradBias = sum(maskedGrad, axis=0) — single SIMD pass, pooled buffer
            var gradBias = Helpers.AutoTensorCache.RentOrAllocate<T>(inputs[2]._shape);
            var biasArr = (float[])(object)gradBias.GetDataArray();
            Array.Clear(biasArr, 0, N);
            fixed (float* pM = maskedArr, pB = biasArr)
            {
                for (int row = 0; row < M; row++)
                {
                    float* rowPtr = pM + row * N;
                    for (int j = 0; j < N; j++)
                        pB[j] += rowPtr[j];
                }
            }

            DifferentiableOps.AccumulateGrad(grads, inputs[0], gradInput, engine);
            DifferentiableOps.AccumulateGrad(grads, inputs[1], gradWeight, engine);
            DifferentiableOps.AccumulateGrad(grads, inputs[2], gradBias, engine);
            return;
        }

        // Fallback: separate ops (non-float, non-2D, or BLAS refused)
        fusedReluFallback:
        var maskedGrad = engine.ReluBackward(gradOutput, preActivation);
        var bT = TransposeLastTwoDims(inputs[1], engine);
        var gradA = engine.TensorMatMul(maskedGrad, bT);
        var aT = TransposeLastTwoDims(inputs[0], engine);
        var gradB = engine.TensorMatMul(aT, maskedGrad);
        var reduceAxes = Enumerable.Range(0, maskedGrad.Shape.Length - 1).ToArray();
        var gradBiasFallback = engine.ReduceSum(maskedGrad, reduceAxes, keepDims: false);

        DifferentiableOps.AccumulateGrad(grads, inputs[0], gradA, engine);
        DifferentiableOps.AccumulateGrad(grads, inputs[1], gradB, engine);
        DifferentiableOps.AccumulateGrad(grads, inputs[2], gradBiasFallback, engine);
    }

    /// <summary>
    /// Shared fused backward: activation_backward → transposed BLAS GEMM for both gradients → bias sum.
    /// Eliminates transpose allocations by using BLAS transA/transB flags directly.
    /// </summary>
    private static unsafe void FusedLinearActivationBackwardCore(
        float[] maskedArr, int M, int K, int N,
        Tensor<T>[] inputs, Dictionary<Tensor<T>, Tensor<T>> grads, IEngine engine)
    {
        var inArr = (float[])(object)inputs[0].GetDataArray();
        var wArr = (float[])(object)inputs[1].GetDataArray();

        // gradInput = maskedGrad @ W^T — transposed BLAS, pooled buffer
        // No Array.Clear — TryGemmEx uses beta=0 which overwrites C entirely
        var gradInput = Helpers.AutoTensorCache.RentOrAllocate<T>(inputs[0]._shape);
        var gradInputArr = (float[])(object)gradInput.GetDataArray();
        bool okInput = BlasProvider.TryGemmEx(M, K, N, maskedArr, 0, N, false, wArr, 0, N, true, gradInputArr, 0, K);

        // gradWeight = input^T @ maskedGrad — transposed BLAS, pooled buffer
        var gradWeight = Helpers.AutoTensorCache.RentOrAllocate<T>(inputs[1]._shape);
        var gradWeightArr = (float[])(object)gradWeight.GetDataArray();
        bool okWeight = BlasProvider.TryGemmEx(K, N, M, inArr, 0, K, true, maskedArr, 0, N, false, gradWeightArr, 0, N);

        if (!okInput || !okWeight)
        {
            // BLAS unavailable — fall back to engine-based backward.
            // Wrap maskedArr into a tensor for the fallback path.
            var maskedTensor = new Tensor<T>((T[])(object)maskedArr, new[] { M, N });
            FusedLinearActivationBackwardFallback(maskedTensor, inputs, grads, engine);
            return;
        }

        // gradBias = sum(maskedGrad, axis=0) — single pass, pooled buffer
        var gradBias = Helpers.AutoTensorCache.RentOrAllocate<T>(inputs[2]._shape);
        var biasArr = (float[])(object)gradBias.GetDataArray();
        Array.Clear(biasArr, 0, N);
        fixed (float* pM = maskedArr, pB = biasArr)
        {
            for (int row = 0; row < M; row++)
            {
                float* rowPtr = pM + row * N;
                for (int j = 0; j < N; j++)
                    pB[j] += rowPtr[j];
            }
        }

        DifferentiableOps.AccumulateGrad(grads, inputs[0], gradInput, engine);
        DifferentiableOps.AccumulateGrad(grads, inputs[1], gradWeight, engine);
        DifferentiableOps.AccumulateGrad(grads, inputs[2], gradBias, engine);
    }

    /// <summary>Fallback fused backward using engine calls (non-float or non-2D).</summary>
    private static void FusedLinearActivationBackwardFallback(
        Tensor<T> maskedGrad, Tensor<T>[] inputs,
        Dictionary<Tensor<T>, Tensor<T>> grads, IEngine engine)
    {
        var bT = TransposeLastTwoDims(inputs[1], engine);
        var gradA = engine.TensorMatMul(maskedGrad, bT);
        var aT = TransposeLastTwoDims(inputs[0], engine);
        var gradB = engine.TensorMatMul(aT, maskedGrad);
        var reduceAxes = Enumerable.Range(0, maskedGrad.Shape.Length - 1).ToArray();
        var gradBias = engine.ReduceSum(maskedGrad, reduceAxes, keepDims: false);

        DifferentiableOps.AccumulateGrad(grads, inputs[0], gradA, engine);
        DifferentiableOps.AccumulateGrad(grads, inputs[1], gradB, engine);
        DifferentiableOps.AccumulateGrad(grads, inputs[2], gradBias, engine);
    }

    internal static void FusedMatMulAddSigmoidBackward(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        var maskedGrad = engine.SigmoidBackward(gradOutput, output);
        if (typeof(T) == typeof(float) && inputs[0].Rank == 2 && BlasProvider.IsAvailable)
        {
            FusedLinearActivationBackwardCore(
                (float[])(object)maskedGrad.GetDataArray(),
                inputs[0]._shape[0], inputs[0]._shape[1], inputs[1]._shape[1],
                inputs, grads, engine);
            return;
        }
        FusedLinearActivationBackwardFallback(maskedGrad, inputs, grads, engine);
    }

    internal static void FusedMatMulAddTanhBackward(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        var maskedGrad = engine.TanhBackward(gradOutput, output);
        if (typeof(T) == typeof(float) && inputs[0].Rank == 2 && BlasProvider.IsAvailable)
        {
            FusedLinearActivationBackwardCore(
                (float[])(object)maskedGrad.GetDataArray(),
                inputs[0]._shape[0], inputs[0]._shape[1], inputs[1]._shape[1],
                inputs, grads, engine);
            return;
        }
        FusedLinearActivationBackwardFallback(maskedGrad, inputs, grads, engine);
    }

    internal static void FusedMatMulAddGELUBackward(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        var preActivation = (Tensor<T>)savedState[0];
        var maskedGrad = engine.GeluBackward(gradOutput, preActivation);
        if (typeof(T) == typeof(float) && inputs[0].Rank == 2 && BlasProvider.IsAvailable)
        {
            FusedLinearActivationBackwardCore(
                (float[])(object)maskedGrad.GetDataArray(),
                inputs[0]._shape[0], inputs[0]._shape[1], inputs[1]._shape[1],
                inputs, grads, engine);
            return;
        }
        FusedLinearActivationBackwardFallback(maskedGrad, inputs, grads, engine);
    }

    internal static void FusedMatMulAddSwishBackward(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        var preActivation = (Tensor<T>)savedState[0];
        var maskedGrad = engine.SwishBackward(gradOutput, preActivation);
        if (typeof(T) == typeof(float) && inputs[0].Rank == 2 && BlasProvider.IsAvailable)
        {
            FusedLinearActivationBackwardCore(
                (float[])(object)maskedGrad.GetDataArray(),
                inputs[0]._shape[0], inputs[0]._shape[1], inputs[1]._shape[1],
                inputs, grads, engine);
            return;
        }
        FusedLinearActivationBackwardFallback(maskedGrad, inputs, grads, engine);
    }

    // ──────────────────────────────────────────────────────────────
    // IoU Loss Backward: analytical gradients for bounding box regression
    // ──────────────────────────────────────────────────────────────

    /// <summary>
    /// IoU loss backward: d(1 - IoU)/d(predicted) with analytical per-coordinate gradients.
    /// inputs[0] = predicted [N,4], inputs[1] = target [N,4], savedState[0] = IoU per-box [N]
    /// </summary>
    internal static void IoULossBackward(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var predicted = inputs[0];
        var target = inputs[1];
        int n = predicted.Shape[0];
        var gradPred = new T[n * 4];
        var eps = numOps.FromDouble(1e-7);

        for (int i = 0; i < n; i++)
        {
            int o = i * 4;
            var px1 = predicted[o]; var py1 = predicted[o + 1]; var px2 = predicted[o + 2]; var py2 = predicted[o + 3];
            var tx1 = target[o]; var ty1 = target[o + 1]; var tx2 = target[o + 2]; var ty2 = target[o + 3];

            // Intersection
            var ix1 = MaxVal(px1, tx1, numOps); var iy1 = MaxVal(py1, ty1, numOps);
            var ix2 = MinVal(px2, tx2, numOps); var iy2 = MinVal(py2, ty2, numOps);
            var iw = MaxVal(numOps.Subtract(ix2, ix1), numOps.Zero, numOps);
            var ih = MaxVal(numOps.Subtract(iy2, iy1), numOps.Zero, numOps);
            var iA = numOps.Multiply(iw, ih);

            // Areas
            var pw = MaxVal(numOps.Subtract(px2, px1), numOps.Zero, numOps);
            var ph = MaxVal(numOps.Subtract(py2, py1), numOps.Zero, numOps);
            var predArea = numOps.Multiply(pw, ph);
            var targArea = numOps.Multiply(MaxVal(numOps.Subtract(tx2, tx1), numOps.Zero, numOps),
                                           MaxVal(numOps.Subtract(ty2, ty1), numOps.Zero, numOps));
            var uA = numOps.Add(numOps.Subtract(numOps.Add(predArea, targArea), iA), eps);
            var uSq = numOps.Multiply(uA, uA);

            // Indicator: intersection has positive area
            bool hasIntersection = numOps.ToDouble(iw) > 0 && numOps.ToDouble(ih) > 0;
            var hi = hasIntersection ? numOps.One : numOps.Zero;

            // dIntersection/d(px1,py1,px2,py2)
            var dI0 = numOps.Multiply(hi, numOps.Multiply(numOps.ToDouble(px1) > numOps.ToDouble(tx1) ? numOps.FromDouble(-1.0) : numOps.Zero, ih));
            var dI1 = numOps.Multiply(hi, numOps.Multiply(numOps.ToDouble(py1) > numOps.ToDouble(ty1) ? numOps.FromDouble(-1.0) : numOps.Zero, iw));
            var dI2 = numOps.Multiply(hi, numOps.Multiply(numOps.ToDouble(px2) < numOps.ToDouble(tx2) ? numOps.One : numOps.Zero, ih));
            var dI3 = numOps.Multiply(hi, numOps.Multiply(numOps.ToDouble(py2) < numOps.ToDouble(ty2) ? numOps.One : numOps.Zero, iw));

            // dUnion/d(px1,py1,px2,py2)
            var dU0 = numOps.Subtract(numOps.Negate(ph), dI0);
            var dU1 = numOps.Subtract(numOps.Negate(pw), dI1);
            var dU2 = numOps.Subtract(ph, dI2);
            var dU3 = numOps.Subtract(pw, dI3);

            // dIoU/d(coord) = (dI*U - I*dU) / U^2, loss = 1 - IoU, so d(loss) = -dIoU
            var go = gradOutput[i];
            gradPred[o] = numOps.Multiply(go, numOps.Negate(numOps.Divide(numOps.Subtract(numOps.Multiply(dI0, uA), numOps.Multiply(iA, dU0)), uSq)));
            gradPred[o + 1] = numOps.Multiply(go, numOps.Negate(numOps.Divide(numOps.Subtract(numOps.Multiply(dI1, uA), numOps.Multiply(iA, dU1)), uSq)));
            gradPred[o + 2] = numOps.Multiply(go, numOps.Negate(numOps.Divide(numOps.Subtract(numOps.Multiply(dI2, uA), numOps.Multiply(iA, dU2)), uSq)));
            gradPred[o + 3] = numOps.Multiply(go, numOps.Negate(numOps.Divide(numOps.Subtract(numOps.Multiply(dI3, uA), numOps.Multiply(iA, dU3)), uSq)));
        }

        var gradTensor = new Tensor<T>(gradPred, new[] { n, 4 });
        DifferentiableOps.AccumulateGrad(grads, inputs[0], gradTensor, engine);
    }

    /// <summary>GIoU loss backward: IoU gradient + enclosing area penalty gradient.</summary>
    internal static void GIoULossBackward(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var predicted = inputs[0]; var target = inputs[1];
        int n = predicted.Shape[0];
        var gradPred = new T[n * 4];
        var eps = numOps.FromDouble(1e-7);

        for (int i = 0; i < n; i++)
        {
            int o = i * 4;
            var px1 = predicted[o]; var py1 = predicted[o + 1]; var px2 = predicted[o + 2]; var py2 = predicted[o + 3];
            var tx1 = target[o]; var ty1 = target[o + 1]; var tx2 = target[o + 2]; var ty2 = target[o + 3];

            var ix1 = MaxVal(px1, tx1, numOps); var iy1 = MaxVal(py1, ty1, numOps);
            var ix2 = MinVal(px2, tx2, numOps); var iy2 = MinVal(py2, ty2, numOps);
            var iw = MaxVal(numOps.Subtract(ix2, ix1), numOps.Zero, numOps);
            var ih = MaxVal(numOps.Subtract(iy2, iy1), numOps.Zero, numOps);
            var iA = numOps.Multiply(iw, ih);

            var pw = MaxVal(numOps.Subtract(px2, px1), numOps.Zero, numOps);
            var ph = MaxVal(numOps.Subtract(py2, py1), numOps.Zero, numOps);
            var predArea = numOps.Multiply(pw, ph);
            var targArea = numOps.Multiply(MaxVal(numOps.Subtract(tx2, tx1), numOps.Zero, numOps),
                                           MaxVal(numOps.Subtract(ty2, ty1), numOps.Zero, numOps));
            var uA = numOps.Add(numOps.Subtract(numOps.Add(predArea, targArea), iA), eps);
            var uSq = numOps.Multiply(uA, uA);

            bool hasIntersection = numOps.ToDouble(iw) > 0 && numOps.ToDouble(ih) > 0;
            var hi = hasIntersection ? numOps.One : numOps.Zero;

            var dI = new T[4];
            dI[0] = numOps.Multiply(hi, numOps.Multiply(numOps.ToDouble(px1) > numOps.ToDouble(tx1) ? numOps.FromDouble(-1.0) : numOps.Zero, ih));
            dI[1] = numOps.Multiply(hi, numOps.Multiply(numOps.ToDouble(py1) > numOps.ToDouble(ty1) ? numOps.FromDouble(-1.0) : numOps.Zero, iw));
            dI[2] = numOps.Multiply(hi, numOps.Multiply(numOps.ToDouble(px2) < numOps.ToDouble(tx2) ? numOps.One : numOps.Zero, ih));
            dI[3] = numOps.Multiply(hi, numOps.Multiply(numOps.ToDouble(py2) < numOps.ToDouble(ty2) ? numOps.One : numOps.Zero, iw));
            var dU = new T[] {
                numOps.Subtract(numOps.Negate(ph), dI[0]), numOps.Subtract(numOps.Negate(pw), dI[1]),
                numOps.Subtract(ph, dI[2]), numOps.Subtract(pw, dI[3])
            };

            // IoU gradient (for -IoU component)
            var iouGrad = new T[4];
            for (int c = 0; c < 4; c++)
                iouGrad[c] = numOps.Divide(numOps.Subtract(numOps.Multiply(dI[c], uA), numOps.Multiply(iA, dU[c])), uSq);

            // Enclosing box
            var encW = numOps.Subtract(MaxVal(px2, tx2, numOps), MinVal(px1, tx1, numOps));
            var encH = numOps.Subtract(MaxVal(py2, ty2, numOps), MinVal(py1, ty1, numOps));
            var encA = numOps.Add(numOps.Multiply(encW, encH), eps);
            var encASq = numOps.Multiply(encA, encA);

            var dEncA = new T[4];
            dEncA[0] = numOps.Multiply(numOps.ToDouble(px1) < numOps.ToDouble(tx1) ? numOps.FromDouble(-1.0) : numOps.Zero, encH);
            dEncA[1] = numOps.Multiply(numOps.ToDouble(py1) < numOps.ToDouble(ty1) ? numOps.FromDouble(-1.0) : numOps.Zero, encW);
            dEncA[2] = numOps.Multiply(numOps.ToDouble(px2) > numOps.ToDouble(tx2) ? numOps.One : numOps.Zero, encH);
            dEncA[3] = numOps.Multiply(numOps.ToDouble(py2) > numOps.ToDouble(ty2) ? numOps.One : numOps.Zero, encW);

            // GIoU penalty = (U - encA) / encA → d/d(coord) = -(dU*encA - U*dEncA)/encA^2
            var go = gradOutput[i];
            for (int c = 0; c < 4; c++)
            {
                var dPenalty = numOps.Negate(numOps.Divide(
                    numOps.Subtract(numOps.Multiply(dU[c], encA), numOps.Multiply(uA, dEncA[c])), encASq));
                gradPred[o + c] = numOps.Multiply(go, numOps.Subtract(dPenalty, iouGrad[c]));
            }
        }

        var gradTensor = new Tensor<T>(gradPred, new[] { n, 4 });
        DifferentiableOps.AccumulateGrad(grads, inputs[0], gradTensor, engine);
    }

    /// <summary>DIoU loss backward: IoU gradient + distance/diagonal penalty gradient.</summary>
    internal static void DIoULossBackward(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var predicted = inputs[0]; var target = inputs[1];
        int n = predicted.Shape[0];
        var gradPred = new T[n * 4];
        var eps = numOps.FromDouble(1e-7);

        for (int i = 0; i < n; i++)
        {
            int o = i * 4;
            var px1 = predicted[o]; var py1 = predicted[o + 1]; var px2 = predicted[o + 2]; var py2 = predicted[o + 3];
            var tx1 = target[o]; var ty1 = target[o + 1]; var tx2 = target[o + 2]; var ty2 = target[o + 3];

            // IoU gradient (same as IoULossBackward)
            var ix1 = MaxVal(px1, tx1, numOps); var iy1 = MaxVal(py1, ty1, numOps);
            var ix2 = MinVal(px2, tx2, numOps); var iy2 = MinVal(py2, ty2, numOps);
            var iw = MaxVal(numOps.Subtract(ix2, ix1), numOps.Zero, numOps);
            var ih = MaxVal(numOps.Subtract(iy2, iy1), numOps.Zero, numOps);
            var iA = numOps.Multiply(iw, ih);
            var pw = MaxVal(numOps.Subtract(px2, px1), numOps.Zero, numOps);
            var ph = MaxVal(numOps.Subtract(py2, py1), numOps.Zero, numOps);
            var predArea = numOps.Multiply(pw, ph);
            var targArea = numOps.Multiply(MaxVal(numOps.Subtract(tx2, tx1), numOps.Zero, numOps),
                                           MaxVal(numOps.Subtract(ty2, ty1), numOps.Zero, numOps));
            var uA = numOps.Add(numOps.Subtract(numOps.Add(predArea, targArea), iA), eps);
            var uSq = numOps.Multiply(uA, uA);
            bool hasIntersection = numOps.ToDouble(iw) > 0 && numOps.ToDouble(ih) > 0;
            var hi = hasIntersection ? numOps.One : numOps.Zero;

            var dI = new T[4];
            dI[0] = numOps.Multiply(hi, numOps.Multiply(numOps.ToDouble(px1) > numOps.ToDouble(tx1) ? numOps.FromDouble(-1.0) : numOps.Zero, ih));
            dI[1] = numOps.Multiply(hi, numOps.Multiply(numOps.ToDouble(py1) > numOps.ToDouble(ty1) ? numOps.FromDouble(-1.0) : numOps.Zero, iw));
            dI[2] = numOps.Multiply(hi, numOps.Multiply(numOps.ToDouble(px2) < numOps.ToDouble(tx2) ? numOps.One : numOps.Zero, ih));
            dI[3] = numOps.Multiply(hi, numOps.Multiply(numOps.ToDouble(py2) < numOps.ToDouble(ty2) ? numOps.One : numOps.Zero, iw));
            var dU = new T[] {
                numOps.Subtract(numOps.Negate(ph), dI[0]), numOps.Subtract(numOps.Negate(pw), dI[1]),
                numOps.Subtract(ph, dI[2]), numOps.Subtract(pw, dI[3])
            };
            var iouGrad = new T[4];
            for (int c = 0; c < 4; c++)
                iouGrad[c] = numOps.Divide(numOps.Subtract(numOps.Multiply(dI[c], uA), numOps.Multiply(iA, dU[c])), uSq);

            // Center distance: rho^2 = dx^2 + dy^2 where dx = 0.5*(px1+px2) - 0.5*(tx1+tx2)
            var half = numOps.FromDouble(0.5);
            var dx = numOps.Subtract(numOps.Multiply(half, numOps.Add(px1, px2)), numOps.Multiply(half, numOps.Add(tx1, tx2)));
            var dy = numOps.Subtract(numOps.Multiply(half, numOps.Add(py1, py2)), numOps.Multiply(half, numOps.Add(ty1, ty2)));
            var rhoSq = numOps.Add(numOps.Multiply(dx, dx), numOps.Multiply(dy, dy));
            var dRho = new T[] { dx, dy, dx, dy }; // d(rhoSq)/d(px1)=dx, etc.

            // Diagonal of enclosing box: c^2 = encDx^2 + encDy^2
            var encDx = numOps.Subtract(MaxVal(px2, tx2, numOps), MinVal(px1, tx1, numOps));
            var encDy = numOps.Subtract(MaxVal(py2, ty2, numOps), MinVal(py1, ty1, numOps));
            var cSq = numOps.Add(numOps.Add(numOps.Multiply(encDx, encDx), numOps.Multiply(encDy, encDy)), eps);
            var cSqSq = numOps.Multiply(cSq, cSq);

            var two = numOps.FromDouble(2.0);
            var dCSq = new T[4];
            dCSq[0] = numOps.Multiply(numOps.Multiply(two, encDx), numOps.ToDouble(px1) < numOps.ToDouble(tx1) ? numOps.FromDouble(-1.0) : numOps.Zero);
            dCSq[1] = numOps.Multiply(numOps.Multiply(two, encDy), numOps.ToDouble(py1) < numOps.ToDouble(ty1) ? numOps.FromDouble(-1.0) : numOps.Zero);
            dCSq[2] = numOps.Multiply(numOps.Multiply(two, encDx), numOps.ToDouble(px2) > numOps.ToDouble(tx2) ? numOps.One : numOps.Zero);
            dCSq[3] = numOps.Multiply(numOps.Multiply(two, encDy), numOps.ToDouble(py2) > numOps.ToDouble(ty2) ? numOps.One : numOps.Zero);

            var go = gradOutput[i];
            for (int c = 0; c < 4; c++)
            {
                var dDist = numOps.Divide(numOps.Subtract(numOps.Multiply(dRho[c], cSq), numOps.Multiply(rhoSq, dCSq[c])), cSqSq);
                gradPred[o + c] = numOps.Multiply(go, numOps.Add(numOps.Negate(iouGrad[c]), dDist));
            }
        }

        var gradTensor = new Tensor<T>(gradPred, new[] { n, 4 });
        DifferentiableOps.AccumulateGrad(grads, inputs[0], gradTensor, engine);
    }

    /// <summary>CIoU loss backward: DIoU gradient + aspect ratio penalty gradient.</summary>
    internal static void CIoULossBackward(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var predicted = inputs[0]; var target = inputs[1];
        int n = predicted.Shape[0];
        var gradPred = new T[n * 4];
        var eps = numOps.FromDouble(1e-7);

        for (int i = 0; i < n; i++)
        {
            int o = i * 4;
            var px1 = predicted[o]; var py1 = predicted[o + 1]; var px2 = predicted[o + 2]; var py2 = predicted[o + 3];
            var tx1 = target[o]; var ty1 = target[o + 1]; var tx2 = target[o + 2]; var ty2 = target[o + 3];

            // IoU computation
            var ix1 = MaxVal(px1, tx1, numOps); var iy1 = MaxVal(py1, ty1, numOps);
            var ix2 = MinVal(px2, tx2, numOps); var iy2 = MinVal(py2, ty2, numOps);
            var iw = MaxVal(numOps.Subtract(ix2, ix1), numOps.Zero, numOps);
            var ih = MaxVal(numOps.Subtract(iy2, iy1), numOps.Zero, numOps);
            var iA = numOps.Multiply(iw, ih);
            var pw = numOps.Add(MaxVal(numOps.Subtract(px2, px1), numOps.Zero, numOps), eps);
            var ph = numOps.Add(MaxVal(numOps.Subtract(py2, py1), numOps.Zero, numOps), eps);
            var predArea = numOps.Multiply(pw, ph);
            var tw = numOps.Add(MaxVal(numOps.Subtract(tx2, tx1), numOps.Zero, numOps), eps);
            var th = numOps.Add(MaxVal(numOps.Subtract(ty2, ty1), numOps.Zero, numOps), eps);
            var targArea = numOps.Multiply(tw, th);
            var uA = numOps.Add(numOps.Subtract(numOps.Add(predArea, targArea), iA), eps);
            var uSq = numOps.Multiply(uA, uA);
            var iou = numOps.Divide(iA, uA);
            bool hasIntersection = numOps.ToDouble(iw) > 0 && numOps.ToDouble(ih) > 0;
            var hi = hasIntersection ? numOps.One : numOps.Zero;

            var dI = new T[4];
            dI[0] = numOps.Multiply(hi, numOps.Multiply(numOps.ToDouble(px1) > numOps.ToDouble(tx1) ? numOps.FromDouble(-1.0) : numOps.Zero, ih));
            dI[1] = numOps.Multiply(hi, numOps.Multiply(numOps.ToDouble(py1) > numOps.ToDouble(ty1) ? numOps.FromDouble(-1.0) : numOps.Zero, iw));
            dI[2] = numOps.Multiply(hi, numOps.Multiply(numOps.ToDouble(px2) < numOps.ToDouble(tx2) ? numOps.One : numOps.Zero, ih));
            dI[3] = numOps.Multiply(hi, numOps.Multiply(numOps.ToDouble(py2) < numOps.ToDouble(ty2) ? numOps.One : numOps.Zero, iw));
            var dU = new T[] {
                numOps.Subtract(numOps.Negate(ph), dI[0]), numOps.Subtract(numOps.Negate(pw), dI[1]),
                numOps.Subtract(ph, dI[2]), numOps.Subtract(pw, dI[3])
            };
            var iouGrad = new T[4];
            for (int c = 0; c < 4; c++)
                iouGrad[c] = numOps.Divide(numOps.Subtract(numOps.Multiply(dI[c], uA), numOps.Multiply(iA, dU[c])), uSq);

            // Distance penalty (same as DIoU)
            var half = numOps.FromDouble(0.5);
            var dx = numOps.Subtract(numOps.Multiply(half, numOps.Add(px1, px2)), numOps.Multiply(half, numOps.Add(tx1, tx2)));
            var dy = numOps.Subtract(numOps.Multiply(half, numOps.Add(py1, py2)), numOps.Multiply(half, numOps.Add(ty1, ty2)));
            var rhoSq = numOps.Add(numOps.Multiply(dx, dx), numOps.Multiply(dy, dy));
            var dRho = new T[] { dx, dy, dx, dy };
            var encDx = numOps.Subtract(MaxVal(px2, tx2, numOps), MinVal(px1, tx1, numOps));
            var encDy = numOps.Subtract(MaxVal(py2, ty2, numOps), MinVal(py1, ty1, numOps));
            var cSq = numOps.Add(numOps.Add(numOps.Multiply(encDx, encDx), numOps.Multiply(encDy, encDy)), eps);
            var cSqSq = numOps.Multiply(cSq, cSq);
            var two = numOps.FromDouble(2.0);
            var dCSq = new T[4];
            dCSq[0] = numOps.Multiply(numOps.Multiply(two, encDx), numOps.ToDouble(px1) < numOps.ToDouble(tx1) ? numOps.FromDouble(-1.0) : numOps.Zero);
            dCSq[1] = numOps.Multiply(numOps.Multiply(two, encDy), numOps.ToDouble(py1) < numOps.ToDouble(ty1) ? numOps.FromDouble(-1.0) : numOps.Zero);
            dCSq[2] = numOps.Multiply(numOps.Multiply(two, encDx), numOps.ToDouble(px2) > numOps.ToDouble(tx2) ? numOps.One : numOps.Zero);
            dCSq[3] = numOps.Multiply(numOps.Multiply(two, encDy), numOps.ToDouble(py2) > numOps.ToDouble(ty2) ? numOps.One : numOps.Zero);

            // Aspect ratio penalty: v = (4/pi^2) * (atan(tw/th) - atan(pw/ph))^2
            double piSq = Math.PI * Math.PI;
            var fourOverPiSq = numOps.FromDouble(4.0 / piSq);
            double predAtan = Math.Atan(numOps.ToDouble(pw) / numOps.ToDouble(ph));
            double targAtan = Math.Atan(numOps.ToDouble(tw) / numOps.ToDouble(th));
            double atanDiff = targAtan - predAtan;
            var v = numOps.FromDouble((4.0 / piSq) * atanDiff * atanDiff);
            var alpha = numOps.Divide(v, numOps.Add(numOps.Add(numOps.Subtract(numOps.One, iou), v), eps));

            // dv/d(pw) and dv/d(ph) via atan derivative: d(atan(w/h))/dw = h/(w^2+h^2)
            double rss = numOps.ToDouble(pw) * numOps.ToDouble(pw) + numOps.ToDouble(ph) * numOps.ToDouble(ph);
            double dAtanPw = numOps.ToDouble(ph) / rss;
            double dAtanPh = -numOps.ToDouble(pw) / rss;
            // dv/d(px1) = dv/d(pw) * d(pw)/d(px1) = 2*(4/pi^2)*atanDiff*dAtanPw * (-1)
            var dV = new T[4];
            dV[0] = numOps.FromDouble(2.0 * (4.0 / piSq) * atanDiff * dAtanPw); // d(pw)/d(px1) = -1, but pw = px2-px1 so dAtanPw for pw
            dV[1] = numOps.FromDouble(2.0 * (4.0 / piSq) * atanDiff * dAtanPh);
            dV[2] = numOps.FromDouble(2.0 * (4.0 / piSq) * atanDiff * (-dAtanPw));
            dV[3] = numOps.FromDouble(2.0 * (4.0 / piSq) * atanDiff * (-dAtanPh));

            var go = gradOutput[i];
            for (int c = 0; c < 4; c++)
            {
                var dDist = numOps.Divide(numOps.Subtract(numOps.Multiply(dRho[c], cSq), numOps.Multiply(rhoSq, dCSq[c])), cSqSq);
                var totalGrad = numOps.Add(numOps.Add(numOps.Negate(iouGrad[c]), dDist), numOps.Multiply(alpha, dV[c]));
                gradPred[o + c] = numOps.Multiply(go, totalGrad);
            }
        }

        var gradTensor = new Tensor<T>(gradPred, new[] { n, 4 });
        DifferentiableOps.AccumulateGrad(grads, inputs[0], gradTensor, engine);
    }

    // Helper: max of two scalars
    /// <summary>
    /// Transposes the last two dimensions of an ND tensor.
    /// For 2D: standard transpose. For 3D+: swaps dims[-2] and dims[-1].
    /// </summary>
    private static Tensor<T> TransposeLastTwoDims(Tensor<T> tensor, IEngine engine)
    {
        if (tensor.Rank <= 2)
            return engine.TensorTranspose(tensor);

        // Build permutation: [0, 1, ..., rank-3, rank-1, rank-2]
        int rank = tensor.Rank;
        var perm = new int[rank];
        for (int i = 0; i < rank - 2; i++) perm[i] = i;
        perm[rank - 2] = rank - 1;
        perm[rank - 1] = rank - 2;
        return engine.TensorPermute(tensor, perm);
    }

    private static T MaxVal(T a, T b, INumericOperations<T> ops) =>
        ops.ToDouble(a) >= ops.ToDouble(b) ? a : b;

    // Helper: min of two scalars
    private static T MinVal(T a, T b, INumericOperations<T> ops) =>
        ops.ToDouble(a) <= ops.ToDouble(b) ? a : b;

    // =====================================================================
    // Complex Tensor Backward Functions
    // =====================================================================

    internal static void ComplexMultiplyBackward(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        var a = inputs[0];
        var b = inputs[1];
        var ops = MathHelper.GetNumericOperations<T>();
        int pairs = gradOutput.Length / 2;

        // d(a*b)/da = conj(b), d(a*b)/db = conj(a) — applied via chain rule
        {
            var gradA = new Tensor<T>(a._shape);
            for (int i = 0; i < pairs; i++)
            {
                int idx = i * 2;
                T gRe = gradOutput.GetFlat(idx), gIm = gradOutput.GetFlat(idx + 1);
                T bRe = b.GetFlat(idx), bIm = b.GetFlat(idx + 1);
                // grad_a = grad_out * conj(b)
                gradA.SetFlat(idx, ops.Add(ops.Multiply(gRe, bRe), ops.Multiply(gIm, bIm)));
                gradA.SetFlat(idx + 1, ops.Subtract(ops.Multiply(gIm, bRe), ops.Multiply(gRe, bIm)));
            }
            DifferentiableOps.AccumulateGrad(grads, a, gradA, engine);
        }

        {
            var gradB = new Tensor<T>(b._shape);
            for (int i = 0; i < pairs; i++)
            {
                int idx = i * 2;
                T gRe = gradOutput.GetFlat(idx), gIm = gradOutput.GetFlat(idx + 1);
                T aRe = a.GetFlat(idx), aIm = a.GetFlat(idx + 1);
                // grad_b = conj(a) * grad_out
                gradB.SetFlat(idx, ops.Add(ops.Multiply(aRe, gRe), ops.Multiply(aIm, gIm)));
                gradB.SetFlat(idx + 1, ops.Subtract(ops.Multiply(aRe, gIm), ops.Multiply(aIm, gRe)));
            }
            DifferentiableOps.AccumulateGrad(grads, b, gradB, engine);
        }
    }

    internal static void ComplexConjugateBackward(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        // Conjugate of conjugate gradient: negate odd indices again
        var grad = engine.TensorComplexConjugate(gradOutput);
        DifferentiableOps.AccumulateGrad(grads, inputs[0], grad, engine);
    }

    internal static void ComplexMagnitudeBackward(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        // d|z|/dz = z / |z|, so d|z|/d(re) = re/|z|, d|z|/d(im) = im/|z|
        var a = (Tensor<T>)savedState[0]; // original input
        var ops = MathHelper.GetNumericOperations<T>();
        int pairs = a.Length / 2;

        var gradA = new Tensor<T>(a._shape);
        for (int i = 0; i < pairs; i++)
        {
            int idx = i * 2;
            T re = a.GetFlat(idx), im = a.GetFlat(idx + 1);
            T mag = output.GetFlat(i);
            T gOut = gradOutput.GetFlat(i);
            T eps = ops.FromDouble(1e-8);
            T safeMag = ops.Add(mag, eps);
            gradA.SetFlat(idx, ops.Multiply(gOut, ops.Divide(re, safeMag)));
            gradA.SetFlat(idx + 1, ops.Multiply(gOut, ops.Divide(im, safeMag)));
        }
        DifferentiableOps.AccumulateGrad(grads, inputs[0], gradA, engine);
    }

    // =====================================================================
    // CTC Loss Backward
    // =====================================================================

    internal static void CTCLossBackward(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        var logProbs = (Tensor<T>)savedState[0];
        var targets = (Tensor<int>)savedState[1];
        var inputLengths = (int[])savedState[2];
        var targetLengths = (int[])savedState[3];
        int blank = (int)savedState[4];

        var ops = MathHelper.GetNumericOperations<T>();
        int maxT = logProbs._shape[0];
        int batchSize = logProbs._shape[1];
        int numClasses = logProbs._shape[2];

        var grad = new Tensor<T>(logProbs._shape);

        int targetOffset = 0;
        for (int n = 0; n < batchSize; n++)
        {
            int T_n = inputLengths[n];
            int U_n = targetLengths[n];
            int S = 2 * U_n + 1;

            var expandedLabels = new int[S];
            for (int s = 0; s < S; s++)
                expandedLabels[s] = (s % 2 == 0) ? blank : targets.GetFlat(targetOffset + s / 2);

            double negInf = double.NegativeInfinity;

            // Alpha (forward) pass
            var alpha = new double[T_n, S];
            for (int t = 0; t < T_n; t++)
                for (int s = 0; s < S; s++)
                    alpha[t, s] = negInf;
            alpha[0, 0] = ops.ToDouble(logProbs[0, n, expandedLabels[0]]);
            if (S > 1) alpha[0, 1] = ops.ToDouble(logProbs[0, n, expandedLabels[1]]);
            for (int t = 1; t < T_n; t++)
            {
                for (int s = 0; s < S; s++)
                {
                    double lp = ops.ToDouble(logProbs[t, n, expandedLabels[s]]);
                    double prev = alpha[t - 1, s];
                    if (s >= 1) prev = LogSumExpHelper(prev, alpha[t - 1, s - 1]);
                    if (s >= 2 && expandedLabels[s] != blank && expandedLabels[s] != expandedLabels[s - 2])
                        prev = LogSumExpHelper(prev, alpha[t - 1, s - 2]);
                    alpha[t, s] = prev + lp;
                }
            }

            // Beta (backward) pass
            var beta = new double[T_n, S];
            for (int t = 0; t < T_n; t++)
                for (int s = 0; s < S; s++)
                    beta[t, s] = negInf;
            beta[T_n - 1, S - 1] = 0;
            if (S >= 2) beta[T_n - 1, S - 2] = 0;
            for (int t = T_n - 2; t >= 0; t--)
            {
                for (int s = S - 1; s >= 0; s--)
                {
                    double next = beta[t + 1, s] + ops.ToDouble(logProbs[t + 1, n, expandedLabels[s]]);
                    if (s + 1 < S)
                        next = LogSumExpHelper(next, beta[t + 1, s + 1] + ops.ToDouble(logProbs[t + 1, n, expandedLabels[s + 1]]));
                    if (s + 2 < S && expandedLabels[s + 2] != blank && expandedLabels[s + 2] != expandedLabels[s])
                        next = LogSumExpHelper(next, beta[t + 1, s + 2] + ops.ToDouble(logProbs[t + 1, n, expandedLabels[s + 2]]));
                    beta[t, s] = next;
                }
            }

            // Total log prob
            double logProbTotal = alpha[T_n - 1, S - 1];
            if (S >= 2) logProbTotal = LogSumExpHelper(logProbTotal, alpha[T_n - 1, S - 2]);

            // Gradient: d(-logP)/d(logProbs[t,n,k]) = prob[t,k] - (1/P) * sum_s(alpha*beta for label s==k)
            double gOut = ops.ToDouble(gradOutput.GetFlat(n));
            for (int t = 0; t < T_n; t++)
            {
                // Accumulate alpha*beta per class
                var abSum = new double[numClasses];
                for (int i = 0; i < numClasses; i++) abSum[i] = negInf;
                for (int s = 0; s < S; s++)
                {
                    int k = expandedLabels[s];
                    abSum[k] = LogSumExpHelper(abSum[k], alpha[t, s] + beta[t, s]);
                }

                for (int k = 0; k < numClasses; k++)
                {
                    double logProbTK = ops.ToDouble(logProbs[t, n, k]);
                    double probTK = Math.Exp(logProbTK);
                    double gradVal = probTK - Math.Exp(abSum[k] - logProbTotal);
                    grad[t, n, k] = ops.FromDouble(gOut * gradVal);
                }
            }

            targetOffset += U_n;
        }

        DifferentiableOps.AccumulateGrad(grads, inputs[0], grad, engine);
    }

    private static double LogSumExpHelper(double a, double b)
    {
        if (double.IsNegativeInfinity(a)) return b;
        if (double.IsNegativeInfinity(b)) return a;
        double max = Math.Max(a, b);
        return max + Math.Log(Math.Exp(a - max) + Math.Exp(b - max));
    }

    // ──────────────────────────────────────────────────────────────
    // Gated Linear Units: delegate to engine IEngine backward methods
    // (GPU-compatible — engine dispatches to GPU when active)
    // ──────────────────────────────────────────────────────────────

    /// <summary>GLU backward via engine.GLUBackward</summary>
    internal static void GLUBackward(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        var dim = (int)savedState[0];
        var grad = engine.GLUBackward(gradOutput, inputs[0], dim);
        DifferentiableOps.AccumulateGrad(grads, inputs[0], grad, engine);
    }

    /// <summary>Sparsemax backward via engine.SparsemaxBackward</summary>
    internal static void SparsemaxBackward(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        var axis = (int)savedState[0];
        var grad = engine.SparsemaxBackward(gradOutput, output, axis);
        DifferentiableOps.AccumulateGrad(grads, inputs[0], grad, engine);
    }

    /// <summary>TaylorSoftmax backward via engine.TaylorSoftmaxBackward</summary>
    internal static void TaylorSoftmaxBackward(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        var order = (int)savedState[0];
        var axis = (int)savedState[1];
        var grad = engine.TaylorSoftmaxBackward(gradOutput, inputs[0], output, order, axis);
        DifferentiableOps.AccumulateGrad(grads, inputs[0], grad, engine);
    }

    /// <summary>SphericalSoftmax backward via engine.SphericalSoftmaxBackward</summary>
    internal static void SphericalSoftmaxBackward(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        var axis = (int)savedState[0];
        var grad = engine.SphericalSoftmaxBackward(gradOutput, inputs[0], output, axis);
        DifferentiableOps.AccumulateGrad(grads, inputs[0], grad, engine);
    }

    // ──────────────────────────────────────────────────────────────
    // Pooling & Convolution backward: delegate to engine methods
    // ──────────────────────────────────────────────────────────────

    /// <summary>MaxPool2D with indices backward via engine</summary>
    internal static void MaxPool2DWithIndicesBackward(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        var maxIndices = (int[,,,,])savedState[0];
        var inputShape = inputs[0]._shape;
        var poolSize = (int[])savedState[1];
        var stride = (int[])savedState[2];
        var grad = engine.MaxPool2DBackward(gradOutput, maxIndices, inputShape, poolSize, stride);
        DifferentiableOps.AccumulateGrad(grads, inputs[0], grad, engine);
    }

    /// <summary>AvgPool3D backward via engine</summary>
    internal static void AvgPool3DBackward(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        var inputShape = inputs[0]._shape;
        var poolSize = (int[])savedState[0];
        var stride = (int[])savedState[1];
        var padding = (int[])savedState[2];
        var grad = engine.AvgPool3DBackward(gradOutput, inputShape, poolSize, stride, padding);
        DifferentiableOps.AccumulateGrad(grads, inputs[0], grad, engine);
    }

    /// <summary>MaxPool3D backward via engine</summary>
    internal static void MaxPool3DBackward(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        var maxIndices = (int[,,,,,])savedState[0];
        var inputShape = inputs[0]._shape;
        var poolSize = (int[])savedState[1];
        var stride = (int[])savedState[2];
        var grad = engine.MaxPool3DBackward(gradOutput, maxIndices, inputShape, poolSize, stride);
        DifferentiableOps.AccumulateGrad(grads, inputs[0], grad, engine);
    }

    /// <summary>DepthwiseConv2D backward via engine backward helpers</summary>
    internal static void DepthwiseConv2DBackward(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        var stride = (int[])savedState[0];
        var padding = (int[])savedState[1];
        var inputGrad = engine.DepthwiseConv2DBackwardInput(gradOutput, inputs[1], inputs[0]._shape, stride, padding);
        DifferentiableOps.AccumulateGrad(grads, inputs[0], inputGrad, engine);
        var kernelGrad = engine.DepthwiseConv2DBackwardKernel(gradOutput, inputs[0], inputs[1]._shape, stride, padding);
        DifferentiableOps.AccumulateGrad(grads, inputs[1], kernelGrad, engine);
    }

    /// <summary>ConvTranspose3D backward via engine backward helpers</summary>
    internal static void ConvTranspose3DBackward(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        var stride = (int[])savedState[0];
        var padding = (int[])savedState[1];
        var inputGrad = engine.ConvTranspose3DBackwardInput(gradOutput, inputs[1], inputs[0]._shape, stride, padding);
        DifferentiableOps.AccumulateGrad(grads, inputs[0], inputGrad, engine);
        var kernelGrad = engine.ConvTranspose3DBackwardKernel(gradOutput, inputs[0], inputs[1]._shape, stride, padding);
        DifferentiableOps.AccumulateGrad(grads, inputs[1], kernelGrad, engine);
    }

    /// <summary>Upsample3D backward via engine</summary>
    internal static void Upsample3DBackward(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        var inputShape = inputs[0]._shape;
        var scaleD = (int)savedState[0];
        var scaleH = (int)savedState[1];
        var scaleW = (int)savedState[2];
        var grad = engine.Upsample3DBackward(gradOutput, inputShape, scaleD, scaleH, scaleW);
        DifferentiableOps.AccumulateGrad(grads, inputs[0], grad, engine);
    }

    /// <summary>PixelShuffle backward via engine</summary>
    internal static void PixelShuffleBackward(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        var inputShape = inputs[0]._shape;
        var upscaleFactor = (int)savedState[0];
        var grad = engine.PixelShuffleBackward(gradOutput, inputShape, upscaleFactor);
        DifferentiableOps.AccumulateGrad(grads, inputs[0], grad, engine);
    }

    /// <summary>Crop backward via engine</summary>
    internal static void CropBackward(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        var inputShape = inputs[0]._shape;
        var top = (int)savedState[0];
        var left = (int)savedState[1];
        var grad = engine.CropBackward(gradOutput, inputShape, top, left);
        DifferentiableOps.AccumulateGrad(grads, inputs[0], grad, engine);
    }

    /// <summary>Pad backward via engine</summary>
    internal static void PadBackward(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        var inputShape = inputs[0]._shape;
        var padTop = (int)savedState[0];
        var padLeft = (int)savedState[1];
        var grad = engine.PadBackward(gradOutput, padTop, padLeft, inputShape);
        DifferentiableOps.AccumulateGrad(grads, inputs[0], grad, engine);
    }

    /// <summary>LocallyConnectedConv2D backward via engine backward helpers</summary>
    internal static void LocallyConnectedConv2DBackward(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        var stride = (int[])savedState[0];
        var inputGrad = engine.LocallyConnectedConv2DBackwardInput(gradOutput, inputs[1], inputs[0]._shape, stride);
        DifferentiableOps.AccumulateGrad(grads, inputs[0], inputGrad, engine);
        var weightsGrad = engine.LocallyConnectedConv2DBackwardWeights(gradOutput, inputs[0], inputs[1]._shape, stride);
        DifferentiableOps.AccumulateGrad(grads, inputs[1], weightsGrad, engine);
    }

    /// <summary>RBFKernel backward via engine</summary>
    internal static void RBFKernelBackward(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        // inputs: [0]=input, [1]=centers, [2]=epsilons
        // RBFKernelBackward signature: (gradOutput, input, centers, epsilons, output)
        var (gradX, gradCenters, gradEpsilons) = engine.RBFKernelBackward(gradOutput, inputs[0], inputs[1], inputs[2], output);
        DifferentiableOps.AccumulateGrad(grads, inputs[0], gradX, engine);
        DifferentiableOps.AccumulateGrad(grads, inputs[1], gradCenters, engine);
        DifferentiableOps.AccumulateGrad(grads, inputs[2], gradEpsilons, engine);
    }

    /// <summary>
    /// OctonionMatMulTensor backward. Forward: output[b,o,:] = sum_i(weight[o,i,:] * input[b,i,:]).
    /// Gradients use the 8x8 Jacobians of octonion multiplication (non-commutative).
    /// For r = w * x: dR/dX = Jacobian_B(w), dR/dW = Jacobian_A(x).
    /// gradInput[b,i,:] = sum_o (J_B(w[o,i])^T @ gradOut[b,o,:])
    /// gradWeight[o,i,:] = sum_b (J_A(x[b,i])^T @ gradOut[b,o,:])
    /// </summary>
    internal static void OctonionMatMulBackward(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        var input = inputs[0];  // [batch, inputFeatures, 8]
        var weight = inputs[1]; // [outputFeatures, inputFeatures, 8]
        var numOps = MathHelper.GetNumericOperations<T>();

        int batch = input._shape[0];
        int inputFeatures = input._shape[1];
        int outputFeatures = weight._shape[0];

        var gradInput = new Tensor<T>(input._shape);
        var gradWeight = new Tensor<T>(weight._shape);

        var w = new double[8];
        var x = new double[8];
        var g = new double[8];
        var jac = new double[64]; // 8x8 Jacobian

        for (int b = 0; b < batch; b++)
        {
            for (int i = 0; i < inputFeatures; i++)
            {
                for (int o = 0; o < outputFeatures; o++)
                {
                    // Load weight[o, i, :] and input[b, i, :] and gradOutput[b, o, :]
                    for (int c = 0; c < 8; c++)
                    {
                        w[c] = numOps.ToDouble(weight[o, i, c]);
                        x[c] = numOps.ToDouble(input[b, i, c]);
                        g[c] = numOps.ToDouble(gradOutput[b, o, c]);
                    }

                    // gradInput += J_B(w)^T @ gradOut  (Jacobian of r=w*x w.r.t. x, transposed)
                    OctonionJacobianB(w, jac);
                    for (int ci = 0; ci < 8; ci++)
                    {
                        double sum = 0;
                        for (int co = 0; co < 8; co++)
                            sum += jac[co * 8 + ci] * g[co]; // J^T: column ci of J = row ci of J^T
                        gradInput[b, i, ci] = numOps.Add(gradInput[b, i, ci], numOps.FromDouble(sum));
                    }

                    // gradWeight += J_A(x)^T @ gradOut  (Jacobian of r=w*x w.r.t. w, transposed)
                    OctonionJacobianA(x, jac);
                    for (int ci = 0; ci < 8; ci++)
                    {
                        double sum = 0;
                        for (int co = 0; co < 8; co++)
                            sum += jac[co * 8 + ci] * g[co];
                        gradWeight[o, i, ci] = numOps.Add(gradWeight[o, i, ci], numOps.FromDouble(sum));
                    }
                }
            }
        }

        DifferentiableOps.AccumulateGrad(grads, input, gradInput, engine);
        DifferentiableOps.AccumulateGrad(grads, weight, gradWeight, engine);
    }

    /// <summary>Jacobian of r = a * b w.r.t. a (left factor). 8x8 row-major.</summary>
    private static void OctonionJacobianA(double[] b, double[] jac)
    {
        // Row 0: r0 = a0*b0 - a1*b1 - a2*b2 - a3*b3 - a4*b4 - a5*b5 - a6*b6 - a7*b7
        jac[0] = b[0]; jac[1] = -b[1]; jac[2] = -b[2]; jac[3] = -b[3]; jac[4] = -b[4]; jac[5] = -b[5]; jac[6] = -b[6]; jac[7] = -b[7];
        // Row 1: r1 = a0*b1 + a1*b0 + a2*b3 - a3*b2 + a4*b5 - a5*b4 - a6*b7 + a7*b6
        jac[8] = b[1]; jac[9] = b[0]; jac[10] = b[3]; jac[11] = -b[2]; jac[12] = b[5]; jac[13] = -b[4]; jac[14] = -b[7]; jac[15] = b[6];
        // Row 2: r2 = a0*b2 - a1*b3 + a2*b0 + a3*b1 + a4*b6 + a5*b7 - a6*b4 - a7*b5
        jac[16] = b[2]; jac[17] = -b[3]; jac[18] = b[0]; jac[19] = b[1]; jac[20] = b[6]; jac[21] = b[7]; jac[22] = -b[4]; jac[23] = -b[5];
        // Row 3: r3 = a0*b3 + a1*b2 - a2*b1 + a3*b0 + a4*b7 - a5*b6 + a6*b5 - a7*b4
        jac[24] = b[3]; jac[25] = b[2]; jac[26] = -b[1]; jac[27] = b[0]; jac[28] = b[7]; jac[29] = -b[6]; jac[30] = b[5]; jac[31] = -b[4];
        // Row 4: r4 = a0*b4 - a1*b5 - a2*b6 - a3*b7 + a4*b0 + a5*b1 + a6*b2 + a7*b3
        jac[32] = b[4]; jac[33] = -b[5]; jac[34] = -b[6]; jac[35] = -b[7]; jac[36] = b[0]; jac[37] = b[1]; jac[38] = b[2]; jac[39] = b[3];
        // Row 5: r5 = a0*b5 + a1*b4 - a2*b7 + a3*b6 - a4*b1 + a5*b0 - a6*b3 + a7*b2
        jac[40] = b[5]; jac[41] = b[4]; jac[42] = -b[7]; jac[43] = b[6]; jac[44] = -b[1]; jac[45] = b[0]; jac[46] = -b[3]; jac[47] = b[2];
        // Row 6: r6 = a0*b6 + a1*b7 + a2*b4 - a3*b5 - a4*b2 + a5*b3 + a6*b0 - a7*b1
        jac[48] = b[6]; jac[49] = b[7]; jac[50] = b[4]; jac[51] = -b[5]; jac[52] = -b[2]; jac[53] = b[3]; jac[54] = b[0]; jac[55] = -b[1];
        // Row 7: r7 = a0*b7 - a1*b6 + a2*b5 + a3*b4 - a4*b3 - a5*b2 + a6*b1 + a7*b0
        jac[56] = b[7]; jac[57] = -b[6]; jac[58] = b[5]; jac[59] = b[4]; jac[60] = -b[3]; jac[61] = -b[2]; jac[62] = b[1]; jac[63] = b[0];
    }

    /// <summary>Jacobian of r = a * b w.r.t. b (right factor). 8x8 row-major.</summary>
    private static void OctonionJacobianB(double[] a, double[] jac)
    {
        // Row 0: r0 = a0*b0 - a1*b1 - a2*b2 - a3*b3 - a4*b4 - a5*b5 - a6*b6 - a7*b7
        jac[0] = a[0]; jac[1] = -a[1]; jac[2] = -a[2]; jac[3] = -a[3]; jac[4] = -a[4]; jac[5] = -a[5]; jac[6] = -a[6]; jac[7] = -a[7];
        // Row 1: r1 = a0*b1 + a1*b0 + a2*b3 - a3*b2 + a4*b5 - a5*b4 - a6*b7 + a7*b6
        jac[8] = a[1]; jac[9] = a[0]; jac[10] = -a[3]; jac[11] = a[2]; jac[12] = -a[5]; jac[13] = a[4]; jac[14] = a[7]; jac[15] = -a[6];
        // Row 2
        jac[16] = a[2]; jac[17] = a[3]; jac[18] = a[0]; jac[19] = -a[1]; jac[20] = -a[6]; jac[21] = -a[7]; jac[22] = a[4]; jac[23] = a[5];
        // Row 3
        jac[24] = a[3]; jac[25] = -a[2]; jac[26] = a[1]; jac[27] = a[0]; jac[28] = -a[7]; jac[29] = a[6]; jac[30] = -a[5]; jac[31] = a[4];
        // Row 4
        jac[32] = a[4]; jac[33] = a[5]; jac[34] = a[6]; jac[35] = a[7]; jac[36] = a[0]; jac[37] = -a[1]; jac[38] = -a[2]; jac[39] = -a[3];
        // Row 5
        jac[40] = a[5]; jac[41] = -a[4]; jac[42] = a[7]; jac[43] = -a[6]; jac[44] = a[1]; jac[45] = a[0]; jac[46] = a[3]; jac[47] = -a[2];
        // Row 6
        jac[48] = a[6]; jac[49] = -a[7]; jac[50] = -a[4]; jac[51] = a[5]; jac[52] = a[2]; jac[53] = -a[3]; jac[54] = a[0]; jac[55] = a[1];
        // Row 7
        jac[56] = a[7]; jac[57] = a[6]; jac[58] = -a[5]; jac[59] = -a[4]; jac[60] = a[3]; jac[61] = a[2]; jac[62] = -a[1]; jac[63] = a[0];
    }

    /// <summary>TensorBinaryCrossEntropy backward via engine</summary>
    internal static void BinaryCrossEntropyBackward(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var epsilon = numOps.FromDouble((double)savedState[0]);
        var grad = engine.TensorBinaryCrossEntropyBackward(inputs[0], inputs[1], epsilon);
        var scaledGrad = engine.TensorMultiply(gradOutput, grad);
        DifferentiableOps.AccumulateGrad(grads, inputs[0], scaledGrad, engine);
    }

    /// <summary>TensorTrilinearInterpolate backward via engine</summary>
    internal static void TrilinearInterpolateBackward(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        var positions = (Tensor<T>)savedState[0];
        var grad = engine.TensorTrilinearInterpolateBackward(gradOutput, inputs[0], positions);
        DifferentiableOps.AccumulateGrad(grads, inputs[0], grad, engine);
    }

    /// <summary>ReduceMax backward: gradient flows only to the max element</summary>
    internal static void ReduceMaxBackward(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        var maxIndices = (int[])savedState[0];
        var inputShape = inputs[0]._shape;
        var grad = engine.ReduceMaxBackward(gradOutput, maxIndices, inputShape);
        DifferentiableOps.AccumulateGrad(grads, inputs[0], grad, engine);
    }

    /// <summary>ReduceVariance backward via engine</summary>
    internal static void ReduceVarianceBackward(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        var axes = (int[])savedState[0];
        var mean = (Tensor<T>)savedState[1];
        var grad = engine.ReduceVarianceBackward(gradOutput, inputs[0], mean, axes);
        DifferentiableOps.AccumulateGrad(grads, inputs[0], grad, engine);
    }

    /// <summary>ReduceLogVariance backward via engine</summary>
    internal static void ReduceLogVarianceBackward(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        var axes = (int[])savedState[0];
        var mean = (Tensor<T>)savedState[1];
        var variance = (Tensor<T>)savedState[2];
        var grad = engine.ReduceLogVarianceBackward(gradOutput, inputs[0], mean, variance, axes);
        DifferentiableOps.AccumulateGrad(grads, inputs[0], grad, engine);
    }

    /// <summary>ScatterMean backward via engine</summary>
    internal static void ScatterMeanBackward(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        var indices = (Tensor<int>)savedState[0];
        var counts = (Tensor<int>)savedState[1];
        var sourceShape = inputs[0]._shape;
        var grad = engine.ScatterMeanBackward(gradOutput, indices, counts, sourceShape);
        DifferentiableOps.AccumulateGrad(grads, inputs[0], grad, engine);
    }

    /// <summary>ScatterMax backward via engine</summary>
    internal static void ScatterMaxBackward(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        var indices = (Tensor<int>)savedState[0];
        var argmax = (Tensor<int>)savedState[1];
        var sourceShape = inputs[0]._shape;
        var grad = engine.ScatterMaxBackward(gradOutput, argmax, sourceShape);
        DifferentiableOps.AccumulateGrad(grads, inputs[0], grad, engine);
    }

    /// <summary>ScatterSoftmax backward via engine</summary>
    internal static void ScatterSoftmaxBackward(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        var indices = (Tensor<int>)savedState[0];
        var grad = engine.ScatterSoftmaxBackward(gradOutput, output, indices);
        DifferentiableOps.AccumulateGrad(grads, inputs[0], grad, engine);
    }

    // ──────────────────────────────────────────────────────────────
    // Element-wise math: composed from engine ops (GPU-transparent)
    // ──────────────────────────────────────────────────────────────

    /// <summary>Scatter backward: input gets grad with scattered positions zeroed, values gets gathered grad</summary>
    internal static void ScatterBackward(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        var indices = savedState[0] is Tensor<int> idxTensor
            ? idxTensor.GetFlattenedData()
            : (int[])savedState[0];
        var axis = (int)savedState[1];

        // dL/dinput = gradOutput with scattered positions zeroed
        var gradInput = gradOutput.Clone();
        var gradInputData = gradInput.GetDataArray();
        var inputShape = inputs[0]._shape;
        int axisSize = inputShape[axis];
        int innerSize = 1;
        for (int i = axis + 1; i < inputShape.Length; i++) innerSize *= inputShape[i];
        int outerSize = 1;
        for (int i = 0; i < axis; i++) outerSize *= inputShape[i];

        var numOps = MathHelper.GetNumericOperations<T>();
        for (int outer = 0; outer < outerSize; outer++)
        {
            for (int idx = 0; idx < indices.Length; idx++)
            {
                int dstIdx = indices[idx];
                int dstBase = outer * axisSize * innerSize + dstIdx * innerSize;
                for (int inner = 0; inner < innerSize; inner++)
                    gradInputData[dstBase + inner] = numOps.Zero;
            }
        }
        DifferentiableOps.AccumulateGrad(grads, inputs[0], gradInput, engine);

        // dL/dvalues = gather from gradOutput at indices
        var gradValues = engine.Gather(gradOutput, new Tensor<int>(new[] { indices.Length }, new Vector<int>(indices)), axis);
        DifferentiableOps.AccumulateGrad(grads, inputs[1], gradValues, engine);
    }

    /// <summary>d(cosh(x))/dx = sinh(x)</summary>
    internal static void CoshBackward(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        var sinhX = engine.TensorSinh(inputs[0]);
        var grad = engine.TensorMultiply(gradOutput, sinhX);
        DifferentiableOps.AccumulateGrad(grads, inputs[0], grad, engine);
    }

    /// <summary>d(sinh(x))/dx = cosh(x)</summary>
    internal static void SinhBackward(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        var coshX = engine.TensorCosh(inputs[0]);
        var grad = engine.TensorMultiply(gradOutput, coshX);
        DifferentiableOps.AccumulateGrad(grads, inputs[0], grad, engine);
    }

    /// <summary>d(frac(x))/dx = 1 (straight-through estimator past floor)</summary>
    internal static void FracBackward(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        DifferentiableOps.AccumulateGrad(grads, inputs[0], gradOutput, engine);
    }

    /// <summary>d(x^scalar)/dx = scalar * x^(scalar-1)</summary>
    internal static void TensorPowBackward(
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

    /// <summary>d(re^2+im^2)/dre = 2*re*grad, d/dim = 2*im*grad</summary>
    internal static void ComplexMagnitudeSquaredBackward(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var two = numOps.FromDouble(2.0);
        var gradReal = engine.TensorMultiplyScalar(engine.TensorMultiply(gradOutput, inputs[0]), two);
        var gradImag = engine.TensorMultiplyScalar(engine.TensorMultiply(gradOutput, inputs[1]), two);
        DifferentiableOps.AccumulateGrad(grads, inputs[0], gradReal, engine);
        DifferentiableOps.AccumulateGrad(grads, inputs[1], gradImag, engine);
    }

    /// <summary>d(scaleA*a + scaleB*b)/da = scaleA*grad, d/db = scaleB*grad</summary>
    internal static void AddScaledBackward(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        var scaleA = (T)savedState[0];
        var scaleB = (T)savedState[1];
        var gradA = engine.TensorMultiplyScalar(gradOutput, scaleA);
        var gradB = engine.TensorMultiplyScalar(gradOutput, scaleB);
        DifferentiableOps.AccumulateGrad(grads, inputs[0], gradA, engine);
        DifferentiableOps.AccumulateGrad(grads, inputs[1], gradB, engine);
    }

    /// <summary>d(outer(a,b))/da = grad @ b, d/db = a^T @ grad</summary>
    internal static void OuterProductBackward(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        // grad is [M, N], a is [M], b is [N]
        // da = sum(grad * b_broadcast, axis=1) = grad @ b
        var gradA = engine.TensorMatMul(gradOutput, inputs[1].Reshape(new[] { inputs[1].Length, 1 }));
        gradA = gradA.Reshape(inputs[0]._shape);
        DifferentiableOps.AccumulateGrad(grads, inputs[0], gradA, engine);
        // db = sum(grad * a_broadcast, axis=0) = a^T @ grad
        var gradB = engine.TensorMatMul(inputs[0].Reshape(new[] { 1, inputs[0].Length }), gradOutput);
        gradB = gradB.Reshape(inputs[1]._shape);
        DifferentiableOps.AccumulateGrad(grads, inputs[1], gradB, engine);
    }

    /// <summary>d(lerp(a, b, t))/da = (1-t)*grad, d/db = t*grad</summary>
    internal static void LerpBackward(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var t = (T)savedState[0];
        var oneMinusT = numOps.Subtract(numOps.One, t);
        var gradA = engine.TensorMultiplyScalar(gradOutput, oneMinusT);
        var gradB = engine.TensorMultiplyScalar(gradOutput, t);
        DifferentiableOps.AccumulateGrad(grads, inputs[0], gradA, engine);
        DifferentiableOps.AccumulateGrad(grads, inputs[1], gradB, engine);
    }

    /// <summary>TensorAddMany: gradient is grad for each input (cloned to avoid shared mutation)</summary>
    internal static void AddManyBackward(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        // Each input gets its own copy of gradOutput to prevent AccumulateGrad
        // from mutating a shared tensor via TensorAddInPlace
        for (int i = 0; i < inputs.Length; i++)
        {
            var grad = i == 0 ? gradOutput : gradOutput.Clone();
            DifferentiableOps.AccumulateGrad(grads, inputs[i], grad, engine);
        }
    }

    /// <summary>TensorMultiplyMany: product rule for N inputs, handles zeros correctly</summary>
    internal static void MultiplyManyBackward(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        // For each input i, gradient = grad * product_of_all_others
        // Compute via: grad_i = grad * (product / inputs[i])
        // But division by zero fails, so compute product_of_others directly
        for (int i = 0; i < inputs.Length; i++)
        {
            Tensor<T> productOfOthers;
            if (inputs.Length == 2)
            {
                productOfOthers = inputs[1 - i];
            }
            else
            {
                // Compute product of all inputs except i
                productOfOthers = i == 0 ? inputs[1].Clone() : inputs[0].Clone();
                for (int j = (i == 0 ? 2 : 1); j < inputs.Length; j++)
                {
                    if (j == i) continue;
                    productOfOthers = engine.TensorMultiply(productOfOthers, inputs[j]);
                }
            }
            var grad = engine.TensorMultiply(gradOutput, productOfOthers);
            DifferentiableOps.AccumulateGrad(grads, inputs[i], grad, engine);
        }
    }

    /// <summary>CumSum backward: reverse cumulative sum via suffix sums</summary>
    internal static void CumSumBackward(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        // Backward of cumsum along axis is reverse cumsum (suffix sum) of gradient
        // suffix_sum[i] = total - cumsum[i] + grad[i]
        var axis = (int)savedState[0];
        var totalSum = engine.ReduceSum(gradOutput, new[] { axis }, keepDims: true);
        var cumGrad = engine.TensorCumSum(gradOutput, axis);
        // Use BroadcastAdd since totalSum has keepDims=true (size-1 on axis)
        var totalPlusGrad = engine.TensorBroadcastAdd(totalSum, gradOutput);
        var grad = engine.TensorSubtract(totalPlusGrad, cumGrad);
        DifferentiableOps.AccumulateGrad(grads, inputs[0], grad, engine);
    }

    /// <summary>RepeatElements backward: sum over repeated dimension</summary>
    internal static void RepeatElementsBackward(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var repeats = (int)savedState[0];
        var axis = (int)savedState[1];
        var inputShape = inputs[0]._shape;
        // Sum every `repeats` consecutive elements along axis
        var grad = new Tensor<T>(inputShape);
        var gradData = gradOutput.GetFlattenedData();
        var resultData = grad.GetDataArray();

        int innerSize = 1;
        for (int d = axis + 1; d < inputShape.Length; d++) innerSize *= inputShape[d];
        int outerSize = 1;
        for (int d = 0; d < axis; d++) outerSize *= inputShape[d];
        int axisSize = inputShape[axis];

        for (int outer = 0; outer < outerSize; outer++)
        {
            for (int a = 0; a < axisSize; a++)
            {
                for (int inner = 0; inner < innerSize; inner++)
                {
                    T sum = numOps.Zero;
                    for (int r = 0; r < repeats; r++)
                    {
                        int srcIdx = outer * (axisSize * repeats * innerSize) + (a * repeats + r) * innerSize + inner;
                        sum = numOps.Add(sum, gradData[srcIdx]);
                    }
                    int dstIdx = outer * (axisSize * innerSize) + a * innerSize + inner;
                    resultData[dstIdx] = sum;
                }
            }
        }
        DifferentiableOps.AccumulateGrad(grads, inputs[0], grad, engine);
    }

    /// <summary>SliceAxis backward: scatter gradient back into zero-filled input shape</summary>
    internal static void SliceAxisBackward(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        var axis = (int)savedState[0];
        var index = (int)savedState[1];
        var inputShape = inputs[0]._shape;
        var grad = new Tensor<T>(inputShape); // zero-initialized
        // Place gradOutput into grad at the correct slice
        engine.TensorSetSliceAxis(grad, gradOutput, axis, index);
        DifferentiableOps.AccumulateGrad(grads, inputs[0], grad, engine);
    }

    /// <summary>Diagonal extraction backward: scatter back to zero matrix</summary>
    internal static void DiagonalBackward(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        var inputShape = inputs[0]._shape;
        var grad = new Tensor<T>(inputShape); // zero
        int diagLen = gradOutput.Length;
        for (int i = 0; i < diagLen; i++)
            grad[i, i] = gradOutput[i];
        DifferentiableOps.AccumulateGrad(grads, inputs[0], grad, engine);
    }

    /// <summary>PairwiseDistanceSquared backward: d(||xi-xj||^2)/dX = 2*(xi-xj) for each pair</summary>
    internal static void PairwiseDistanceSquaredBackward(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        // For pairwise distance: D[i,j] = ||x_i - x_j||^2
        // dD[i,j]/dx_i = 2*(x_i - x_j), dD[i,j]/dx_j = -2*(x_i - x_j)
        // Sum gradient contributions over all pairs
        var numOps = MathHelper.GetNumericOperations<T>();
        var input = inputs[0];
        int n = input._shape[0];
        int d = input._shape[1];
        var grad = new Tensor<T>(input._shape);

        // D[i,j] = ||x_i - x_j||^2
        // dL/dx_i = sum_j 2*(g[i,j] + g[j,i])*(x_i - x_j)
        // (row contribution from D[i,j] + column contribution from D[j,i])
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                // Both g[i,j] and g[j,i] contribute to gradient of x_i
                T gSum = numOps.Add(gradOutput[i, j], gradOutput[j, i]);
                var twoG = numOps.Multiply(numOps.FromDouble(2.0), gSum);
                for (int k = 0; k < d; k++)
                {
                    T diff = numOps.Subtract(input[i, k], input[j, k]);
                    grad[i, k] = numOps.Add(grad[i, k], numOps.Multiply(twoG, diff));
                }
            }
        }
        DifferentiableOps.AccumulateGrad(grads, inputs[0], grad, engine);
    }

    /// <summary>FusedLinear backward: decomposed as MatMul + BroadcastAdd backward</summary>
    internal static void FusedLinearBackward(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        FusedLinearBackwardCore(gradOutput, inputs, output, savedState, engine, grads);
    }

    internal static void FusedLinearWithActivationBackward(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        // Apply activation derivative to gradOutput before linear backward.
        // We need the pre-activation (linear output) for correct GELU/Swish derivatives.
        if (savedState is { Length: >= 1 })
        {
            var activation = (FusedActivationType)savedState[0];
            // Re-compute pre-activation: linear = input @ weights + bias
            var preActivation = engine.TensorMatMul(inputs[0], inputs[1]);
            if (inputs.Length > 2)
                preActivation = engine.TensorBroadcastAdd(preActivation, inputs[2]);
            gradOutput = ApplyActivationDerivative(gradOutput, preActivation, activation, engine);
        }

        FusedLinearBackwardCore(gradOutput, inputs, output, savedState, engine, grads);
    }

    private static Tensor<T> ApplyActivationDerivative(
        Tensor<T> gradOutput, Tensor<T> preActivation, FusedActivationType activation, IEngine engine)
    {
        if (activation == FusedActivationType.None) return gradOutput;

        // Use CpuEngine.ApplyFusedActivationBackward which dispatches via ActivationRegistry
        // This is OCP-compliant: new activations register themselves, no switch needed here
        if (engine is CpuEngine cpuEngine)
            return cpuEngine.ApplyFusedActivationBackward(gradOutput, preActivation, activation);

        // Fallback for non-CPU engines
        return gradOutput;
    }

    private static unsafe void FusedLinearBackwardCore(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        // Fused BLAS path: transposed GEMM + inline bias sum — no transpose allocation
        if (typeof(T) == typeof(float) && inputs[0].Rank == 2 && inputs[1].Rank == 2
            && BlasProvider.IsAvailable && gradOutput.IsContiguous)
        {
            int M = inputs[0]._shape[0]; // batch
            int K = inputs[0]._shape[1]; // in_features
            int N = inputs[1]._shape[1]; // out_features

            var gArr = (float[])(object)gradOutput.GetDataArray();
            var inArr = (float[])(object)inputs[0].GetDataArray();
            var wArr = (float[])(object)inputs[1].GetDataArray();

            // Pool gradient buffers via AutoTensorCache — zero alloc on steps 2+
            var inputGrad = Helpers.AutoTensorCache.RentOrAllocate<T>(inputs[0]._shape);
            var weightGrad = Helpers.AutoTensorCache.RentOrAllocate<T>(inputs[1]._shape);
            var gradInputArr = (float[])(object)inputGrad.GetDataArray();
            var gradWeightArr = (float[])(object)weightGrad.GetDataArray();
            // No Array.Clear — TryGemmEx with beta=0 overwrites C entirely

            bool okInput = BlasProvider.TryGemmEx(M, K, N, gArr, 0, N, false, wArr, 0, N, true, gradInputArr, 0, K);
            bool okWeight = BlasProvider.TryGemmEx(K, N, M, inArr, 0, K, true, gArr, 0, N, false, gradWeightArr, 0, N);

            if (!okInput || !okWeight)
                goto fusedFallback; // BLAS refused — use engine fallback for all

            DifferentiableOps.AccumulateGrad(grads, inputs[0], inputGrad, engine);
            DifferentiableOps.AccumulateGrad(grads, inputs[1], weightGrad, engine);

            // dL/dbias = sum(gradOutput, axis=0) — inline loop, pooled buffer
            if (inputs.Length > 2)
            {
                var biasGrad = Helpers.AutoTensorCache.RentOrAllocate<T>(inputs[2]._shape);
                var biasArr = (float[])(object)biasGrad.GetDataArray();
                Array.Clear(biasArr, 0, N);
                fixed (float* pG = gArr, pB = biasArr)
                {
                    for (int row = 0; row < M; row++)
                    {
                        float* rowPtr = pG + row * N;
                        for (int j = 0; j < N; j++)
                            pB[j] += rowPtr[j];
                    }
                }
                DifferentiableOps.AccumulateGrad(grads, inputs[2], biasGrad, engine);
            }
            return;
        }

        // Fallback: engine calls (non-float or non-2D, or BLAS refused)
        fusedFallback:
        var weightT = TransposeLastTwoDims(inputs[1], engine);
        var inputGradFb = engine.TensorMatMul(gradOutput, weightT);
        DifferentiableOps.AccumulateGrad(grads, inputs[0], inputGradFb, engine);

        var inputT = TransposeLastTwoDims(inputs[0], engine);
        var weightGradFb = engine.TensorMatMul(inputT, gradOutput);
        DifferentiableOps.AccumulateGrad(grads, inputs[1], weightGradFb, engine);

        if (inputs.Length > 2)
        {
            var biasGrad = engine.ReduceSum(gradOutput, new[] { 0 }, keepDims: false);
            DifferentiableOps.AccumulateGrad(grads, inputs[2], biasGrad, engine);
        }
    }

    /// <summary>
    /// Roll backward: shifts the incoming gradient by the negated shifts.
    /// Roll is a permutation, so its transpose is itself-with-negated-shifts.
    /// </summary>
    internal static void RollBackward(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        var shifts = (int[])savedState[0];
        var axes = (int[])savedState[1];
        var negShifts = new int[shifts.Length];
        for (int i = 0; i < shifts.Length; i++) negShifts[i] = -shifts[i];
        var grad = engine.TensorRoll(gradOutput, negShifts, axes);
        DifferentiableOps.AccumulateGrad(grads, inputs[0], grad, engine);
    }

    /// <summary>
    /// Flip backward: flip is an involution, so the transpose is itself.
    /// </summary>
    internal static void FlipBackward(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        var axes = (int[])savedState[0];
        var grad = engine.TensorFlip(gradOutput, axes);
        DifferentiableOps.AccumulateGrad(grads, inputs[0], grad, engine);
    }

    /// <summary>
    /// RepeatInterleave backward: sum-reduce along the repeated axis in
    /// stride-<c>repeats</c> chunks so every source position gets the sum
    /// of its <c>repeats</c> downstream copies.
    /// </summary>
    internal static void RepeatInterleaveBackward(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        int repeats = (int)savedState[0];
        int dim = (int)savedState[1];
        var input = inputs[0];
        var grad = new Tensor<T>(input._shape);

        int rank = input.Rank;
        int axisLen = input._shape[dim];
        int outerSize = 1; for (int k = 0; k < dim; k++) outerSize *= input._shape[k];
        int innerSize = 1; for (int k = dim + 1; k < rank; k++) innerSize *= input._shape[k];

        var ops = MathHelper.GetNumericOperations<T>();
        var src = gradOutput.AsSpan();
        var dst = grad.AsWritableSpan();

        int outerStrideSrc = axisLen * repeats * innerSize;
        int outerStrideDst = axisLen * innerSize;
        for (int outer = 0; outer < outerSize; outer++)
            for (int i = 0; i < axisLen; i++)
                for (int inner = 0; inner < innerSize; inner++)
                {
                    T sum = ops.Zero;
                    int dstPos = outer * outerStrideDst + i * innerSize + inner;
                    int srcBase = outer * outerStrideSrc + i * repeats * innerSize + inner;
                    for (int r = 0; r < repeats; r++)
                        sum = ops.Add(sum, src[srcBase + r * innerSize]);
                    dst[dstPos] = sum;
                }

        DifferentiableOps.AccumulateGrad(grads, input, grad, engine);
    }

    /// <summary>
    /// CumProd backward. For y_i = ∏_{j≤i} x_j, we have
    ///   dy_i/dx_k = y_i / x_k   (for k ≤ i)  — valid only when x_k ≠ 0.
    /// So dL/dx_k = Σ_{i≥k} dL/dy_i · (y_i / x_k).
    /// This v1 implementation routes through a per-axis loop on CPU; it's
    /// correct but not fast. Zero inputs produce NaN — a caveat matching
    /// PyTorch's cumprod-backward semantics.
    /// </summary>
    internal static void CumProdBackward(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        int axis = (int)savedState[0];
        var ops = MathHelper.GetNumericOperations<T>();
        var input = inputs[0];
        var grad = new Tensor<T>(input._shape);
        var src = input.AsSpan();
        var y = output.AsSpan();
        var dY = gradOutput.AsSpan();
        var dX = grad.AsWritableSpan();

        int rank = input.Rank;
        if (axis < 0) axis += rank;
        int axisLen = input._shape[axis];
        int outerSize = 1; for (int k = 0; k < axis; k++) outerSize *= input._shape[k];
        int innerSize = 1; for (int k = axis + 1; k < rank; k++) innerSize *= input._shape[k];

        for (int outer = 0; outer < outerSize; outer++)
            for (int inner = 0; inner < innerSize; inner++)
                for (int k = 0; k < axisLen; k++)
                {
                    int xPos = outer * axisLen * innerSize + k * innerSize + inner;
                    T acc = ops.Zero;
                    for (int i = k; i < axisLen; i++)
                    {
                        int yPos = outer * axisLen * innerSize + i * innerSize + inner;
                        acc = ops.Add(acc, ops.Multiply(dY[yPos], ops.Divide(y[yPos], src[xPos])));
                    }
                    dX[xPos] = acc;
                }

        DifferentiableOps.AccumulateGrad(grads, input, grad, engine);
    }

    /// <summary>
    /// CumMax backward: gradient flows to the position that *set* the running
    /// max at each output index. When the same argmax wins multiple steps,
    /// the contributions accumulate on that input position.
    /// </summary>
    internal static void CumMaxBackward(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
        => CumExtremaBackward(gradOutput, inputs, output, savedState, engine, grads, isMax: true);

    /// <summary>CumMin backward: symmetric to CumMax (argmin).</summary>
    internal static void CumMinBackward(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
        => CumExtremaBackward(gradOutput, inputs, output, savedState, engine, grads, isMax: false);

    private static void CumExtremaBackward(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads,
        bool isMax)
    {
        int axis = (int)savedState[0];
        var ops = MathHelper.GetNumericOperations<T>();
        var input = inputs[0];
        var grad = new Tensor<T>(input._shape);
        var src = input.AsSpan();
        var dY = gradOutput.AsSpan();
        var dX = grad.AsWritableSpan();
        var zero = ops.Zero;
        for (int i = 0; i < dX.Length; i++) dX[i] = zero;

        int rank = input.Rank;
        if (axis < 0) axis += rank;
        int axisLen = input._shape[axis];
        int outerSize = 1; for (int k = 0; k < axis; k++) outerSize *= input._shape[k];
        int innerSize = 1; for (int k = axis + 1; k < rank; k++) innerSize *= input._shape[k];

        for (int outer = 0; outer < outerSize; outer++)
            for (int inner = 0; inner < innerSize; inner++)
            {
                int argExt = 0;
                T currExt = src[outer * axisLen * innerSize + inner];
                for (int i = 0; i < axisLen; i++)
                {
                    int pos = outer * axisLen * innerSize + i * innerSize + inner;
                    var v = src[pos];
                    bool updates = isMax ? ops.GreaterThan(v, currExt) : ops.LessThan(v, currExt);
                    if (i == 0 || updates)
                    {
                        currExt = v;
                        argExt = i;
                    }
                    int argPos = outer * axisLen * innerSize + argExt * innerSize + inner;
                    dX[argPos] = ops.Add(dX[argPos], dY[pos]);
                }
            }

        DifferentiableOps.AccumulateGrad(grads, input, grad, engine);
    }

    /// <summary>
    /// LogCumSumExp backward. Using softmax relationship:
    ///   y_i = log Σ_{j≤i} exp(x_j)
    ///   dy_i/dx_k = exp(x_k - y_i)   for k ≤ i.
    /// So dL/dx_k = Σ_{i≥k} dL/dy_i · exp(x_k - y_i).
    /// </summary>
    internal static void LogCumSumExpBackward(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        int axis = (int)savedState[0];
        var ops = MathHelper.GetNumericOperations<T>();
        var input = inputs[0];
        var grad = new Tensor<T>(input._shape);
        var src = input.AsSpan();
        var y = output.AsSpan();
        var dY = gradOutput.AsSpan();
        var dX = grad.AsWritableSpan();

        int rank = input.Rank;
        if (axis < 0) axis += rank;
        int axisLen = input._shape[axis];
        int outerSize = 1; for (int k = 0; k < axis; k++) outerSize *= input._shape[k];
        int innerSize = 1; for (int k = axis + 1; k < rank; k++) innerSize *= input._shape[k];

        for (int outer = 0; outer < outerSize; outer++)
            for (int inner = 0; inner < innerSize; inner++)
                for (int k = 0; k < axisLen; k++)
                {
                    int xPos = outer * axisLen * innerSize + k * innerSize + inner;
                    T acc = ops.Zero;
                    for (int i = k; i < axisLen; i++)
                    {
                        int yPos = outer * axisLen * innerSize + i * innerSize + inner;
                        acc = ops.Add(acc, ops.Multiply(dY[yPos], ops.Exp(ops.Subtract(src[xPos], y[yPos]))));
                    }
                    dX[xPos] = acc;
                }

        DifferentiableOps.AccumulateGrad(grads, input, grad, engine);
    }

    /// <summary>ClampMin backward: gradient passes only where x &gt;= min.</summary>
    internal static void ClampMinBackward(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        var min = (T)savedState[0];
        var ops = MathHelper.GetNumericOperations<T>();
        var input = inputs[0];
        var grad = new Tensor<T>(input._shape);
        var src = input.AsSpan();
        var gsrc = gradOutput.AsSpan();
        var dst = grad.AsWritableSpan();
        var zero = ops.Zero;
        for (int i = 0; i < src.Length; i++)
            dst[i] = ops.GreaterThanOrEquals(src[i], min) ? gsrc[i] : zero;
        DifferentiableOps.AccumulateGrad(grads, input, grad, engine);
    }

    /// <summary>ClampMax backward: gradient passes only where x &lt;= max.</summary>
    internal static void ClampMaxBackward(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        var max = (T)savedState[0];
        var ops = MathHelper.GetNumericOperations<T>();
        var input = inputs[0];
        var grad = new Tensor<T>(input._shape);
        var src = input.AsSpan();
        var gsrc = gradOutput.AsSpan();
        var dst = grad.AsWritableSpan();
        var zero = ops.Zero;
        for (int i = 0; i < src.Length; i++)
            dst[i] = ops.LessThanOrEquals(src[i], max) ? gsrc[i] : zero;
        DifferentiableOps.AccumulateGrad(grads, input, grad, engine);
    }

    /// <summary>
    /// IndexAdd backward: dL/dinput = gradOutput (clone); dL/dsource = gather
    /// gradOutput at the scatter positions.
    /// v1 stores only the input's grad contribution here; source grad is a
    /// follow-up when the op is re-wired through RecordBinary.
    /// </summary>
    internal static void IndexAddBackward(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        // dL/d(input tensor) = dL/d(output) because result = input + scattered(source).
        DifferentiableOps.AccumulateGrad(grads, inputs[0], gradOutput, engine);
    }

    /// <summary>
    /// IndexCopy backward: gradient at copied positions is zeroed on the
    /// input (the original values are overwritten); rest passes through.
    /// </summary>
    internal static void IndexCopyBackward(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        int axis = (int)savedState[0];
        var indices = (Tensor<int>)savedState[1];
        var input = inputs[0];
        // Clone gradOutput; zero-out positions that got overwritten.
        var ops = MathHelper.GetNumericOperations<T>();
        var grad = (Tensor<T>)gradOutput.Clone();
        var dst = grad.AsWritableSpan();
        int rank = input.Rank;
        if (axis < 0) axis += rank;
        int axisSize = input._shape[axis];
        int outerSize = 1; for (int k = 0; k < axis; k++) outerSize *= input._shape[k];
        int innerSize = 1; for (int k = axis + 1; k < rank; k++) innerSize *= input._shape[k];
        var zero = ops.Zero;
        var idx = indices.AsSpan();
        for (int outer = 0; outer < outerSize; outer++)
            for (int i = 0; i < idx.Length; i++)
            {
                int target = idx[i];
                for (int inner = 0; inner < innerSize; inner++)
                {
                    dst[outer * axisSize * innerSize + target * innerSize + inner] = zero;
                }
            }
        DifferentiableOps.AccumulateGrad(grads, input, grad, engine);
    }

    /// <summary>
    /// IndexFill backward: zero the gradient at filled positions; rest passes
    /// through (gradient w.r.t. the scalar fill-value is not tracked in v1).
    /// </summary>
    internal static void IndexFillBackward(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        // Same shape/iteration as IndexCopy — just zero the filled positions.
        int axis = (int)savedState[0];
        var indices = (Tensor<int>)savedState[1];
        var input = inputs[0];
        var ops = MathHelper.GetNumericOperations<T>();
        var grad = (Tensor<T>)gradOutput.Clone();
        var dst = grad.AsWritableSpan();
        int rank = input.Rank;
        if (axis < 0) axis += rank;
        int axisSize = input._shape[axis];
        int outerSize = 1; for (int k = 0; k < axis; k++) outerSize *= input._shape[k];
        int innerSize = 1; for (int k = axis + 1; k < rank; k++) innerSize *= input._shape[k];
        var zero = ops.Zero;
        var idx = indices.AsSpan();
        for (int outer = 0; outer < outerSize; outer++)
            for (int i = 0; i < idx.Length; i++)
            {
                int target = idx[i];
                for (int inner = 0; inner < innerSize; inner++)
                    dst[outer * axisSize * innerSize + target * innerSize + inner] = zero;
            }
        DifferentiableOps.AccumulateGrad(grads, input, grad, engine);
    }

    /// <summary>
    /// TakeAlongDim backward: scatter incoming gradient back along `dim` at
    /// the gathered positions. Duplicate indices accumulate.
    /// </summary>
    internal static void TakeAlongDimBackward(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        var indices = (Tensor<int>)savedState[0];
        int dim = (int)savedState[1];
        var input = inputs[0];
        var ops = MathHelper.GetNumericOperations<T>();
        var grad = new Tensor<T>(input._shape);
        var dst = grad.AsWritableSpan();
        var src = gradOutput.AsSpan();
        var idx = indices.AsSpan();
        var zero = ops.Zero;
        for (int i = 0; i < dst.Length; i++) dst[i] = zero;

        int rank = input.Rank;
        if (dim < 0) dim += rank;
        int srcAxis = input._shape[dim];
        int idxAxis = indices._shape[dim];
        int outerSize = 1; for (int k = 0; k < dim; k++) outerSize *= input._shape[k];
        int innerSize = 1; for (int k = dim + 1; k < rank; k++) innerSize *= input._shape[k];

        for (int outer = 0; outer < outerSize; outer++)
            for (int i = 0; i < idxAxis; i++)
                for (int inner = 0; inner < innerSize; inner++)
                {
                    int idxPos = outer * idxAxis * innerSize + i * innerSize + inner;
                    int target = idx[idxPos];
                    int dstPos = outer * srcAxis * innerSize + target * innerSize + inner;
                    dst[dstPos] = ops.Add(dst[dstPos], src[idxPos]);
                }

        DifferentiableOps.AccumulateGrad(grads, input, grad, engine);
    }

    /// <summary>
    /// MaskedScatter backward: gradient passes through at non-masked positions
    /// (original input survives); masked positions get zero on the input side
    /// (they were overwritten by source).
    /// </summary>
    internal static void MaskedScatterBackward(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        var mask = (Tensor<Bit>)savedState[0];
        var ops = MathHelper.GetNumericOperations<T>();
        var grad = (Tensor<T>)gradOutput.Clone();
        var dst = grad.AsWritableSpan();
        var maskSpan = mask.AsSpan();
        var zero = ops.Zero;
        for (int i = 0; i < dst.Length; i++)
            if ((bool)maskSpan[i]) dst[i] = zero;
        DifferentiableOps.AccumulateGrad(grads, inputs[0], grad, engine);
    }

    /// <summary>
    /// Take backward: scatter the incoming gradient (same shape as indices)
    /// back to the flattened input shape at each indexed position. Duplicate
    /// indices accumulate.
    /// </summary>
    internal static void TakeBackward(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        var indices = (Tensor<int>)savedState[0];
        var inputShape = (int[])savedState[1];

        var grad = new Tensor<T>(inputShape);
        var numOps = MathHelper.GetNumericOperations<T>();
        var dst = grad.AsWritableSpan();
        var src = gradOutput.AsSpan();
        var idx = indices.AsSpan();
        // dst starts at zero (tensor default); accumulate into indexed slots.
        var zero = numOps.Zero;
        for (int i = 0; i < dst.Length; i++) dst[i] = zero;
        for (int i = 0; i < idx.Length; i++)
        {
            int pos = idx[i];
            dst[pos] = numOps.Add(dst[pos], src[i]);
        }
        DifferentiableOps.AccumulateGrad(grads, inputs[0], grad, engine);
    }

    /// <summary>
    /// MaskedSelect backward: scatter the incoming 1-D gradient back to the
    /// original tensor shape at mask-true positions; zero elsewhere.
    /// </summary>
    internal static void MaskedSelectBackward(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        var mask = (Tensor<Bit>)savedState[0];
        var inputShape = (int[])savedState[1];

        var grad = new Tensor<T>(inputShape);
        var numOps = MathHelper.GetNumericOperations<T>();
        var dest = grad.AsWritableSpan();
        var src = gradOutput.AsSpan();
        var maskSpan = mask.AsSpan();
        var zero = numOps.Zero;

        int r = 0;
        for (int i = 0; i < maskSpan.Length; i++)
        {
            dest[i] = (bool)maskSpan[i] ? src[r++] : zero;
        }

        DifferentiableOps.AccumulateGrad(grads, inputs[0], grad, engine);
    }

    /// <summary>
    /// Backward for the general einsum path. For an n-operand forward
    ///     out = einsum("X_1,X_2,...,X_n -> Y", A_1, A_2, ..., A_n)
    /// the gradient w.r.t. the i-th input is
    ///     dL/dA_i = einsum("X_1,...,Y,...,X_n -> X_i",
    ///                      A_1, ..., A_{i-1}, gradOutput, A_{i+1}, ..., A_n)
    /// — the i-th operand slot swaps in `gradOutput` carrying Y's labels, and
    /// the output labels become X_i's labels.
    /// </summary>
    /// <remarks>
    /// v1 limitations (caller guarantees these at record-time):
    ///   - 2+ operands.
    ///   - No repeated labels within a single operand (no diagonals).
    /// Caller skips the tape-record when these are violated so backward is
    /// never invoked for unsupported cases.
    /// </remarks>
    internal static void EinsumBackward(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        var equationStr = (string)savedState[0];
        var eq = Engines.Einsum.EinsumEquation.Parse(equationStr);

        string outLabels = eq.Output.ToString();
        var operandStrs = new string[eq.Operands.Count];
        for (int i = 0; i < operandStrs.Length; i++) operandStrs[i] = eq.Operands[i].ToString();

        var bwdInputs = new Tensor<T>[inputs.Length];
        for (int i = 0; i < inputs.Length; i++)
        {
            // Build the derived equation with operand i replaced by output labels.
            var parts = new string[inputs.Length];
            for (int j = 0; j < inputs.Length; j++)
                parts[j] = (j == i) ? outLabels : operandStrs[j];
            string bwdEq = string.Join(",", parts) + "->" + operandStrs[i];

            for (int j = 0; j < inputs.Length; j++)
                bwdInputs[j] = (j == i) ? gradOutput : inputs[j];

            var gradI = engine.TensorEinsum(bwdEq, bwdInputs);
            DifferentiableOps.AccumulateGrad(grads, inputs[i], gradI, engine);
        }
    }
}
