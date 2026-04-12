using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Autodiff;

/// <summary>
/// Integration tests proving Issue #144 (OctonionMatMulTensor backward) and
/// Issue #145 (RBFKernel backward) are fixed. Each test runs a forward pass
/// through a GradientTape, computes backward, and verifies:
///   1. No crash (the original bugs caused ArgumentException / shape mismatch)
///   2. Gradient shapes match input shapes
///   3. Gradients are non-zero (proving the backward function actually computed something)
///   4. Numerical gradient check (finite differences) agrees with analytical gradients
/// </summary>
public class BackwardBugFixTests
{
    private readonly CpuEngine _engine = new();

    // ================================================================
    // Issue #144: OctonionMatMulTensor backward
    // ================================================================

    [Fact]
    public void OctonionMatMulTensor_Backward_ProducesCorrectShapeGradients()
    {
        // Forward: output[b,o,:] = sum_i(weight[o,i,:] * input[b,i,:])
        // input: [batch=2, inputFeatures=3, 8]
        // weight: [outputFeatures=4, inputFeatures=3, 8]
        var rng = new Random(42);
        var input = new Tensor<double>([2, 3, 8]);
        var weight = new Tensor<double>([4, 3, 8]);
        for (int i = 0; i < input.Length; i++) input[i] = rng.NextDouble() * 2 - 1;
        for (int i = 0; i < weight.Length; i++) weight[i] = rng.NextDouble() * 2 - 1;

        using var tape = new GradientTape<double>();
        var output = _engine.OctonionMatMulTensor(input, weight);

        // Sum output to get scalar loss for backward
        var loss = _engine.ReduceSum(output, null);

        var grads = tape.ComputeGradients(loss,
            new[] { input, weight });

        Assert.True(grads.ContainsKey(input), "No gradient for input");
        Assert.True(grads.ContainsKey(weight), "No gradient for weight");

        // Shapes must match inputs
        Assert.Equal(input.Shape.ToArray(), grads[input].Shape.ToArray());
        Assert.Equal(weight.Shape.ToArray(), grads[weight].Shape.ToArray());

        // Gradients must be non-zero
        bool inputHasNonZero = false;
        bool weightHasNonZero = false;
        for (int i = 0; i < grads[input].Length; i++)
            if (Math.Abs(grads[input][i]) > 1e-10) { inputHasNonZero = true; break; }
        for (int i = 0; i < grads[weight].Length; i++)
            if (Math.Abs(grads[weight][i]) > 1e-10) { weightHasNonZero = true; break; }

        Assert.True(inputHasNonZero, "Input gradient is all zeros");
        Assert.True(weightHasNonZero, "Weight gradient is all zeros");
    }

    [Fact]
    public void OctonionMatMulTensor_Backward_NumericalGradientCheck()
    {
        // Numerical gradient check: perturb each input element by eps, measure
        // change in loss, compare to analytical gradient.
        var rng = new Random(123);
        var input = new Tensor<double>([1, 2, 8]);
        var weight = new Tensor<double>([2, 2, 8]);
        for (int i = 0; i < input.Length; i++) input[i] = rng.NextDouble() * 0.5;
        for (int i = 0; i < weight.Length; i++) weight[i] = rng.NextDouble() * 0.5;

        // Analytical gradient
        using var tape = new GradientTape<double>();
        var output = _engine.OctonionMatMulTensor(input, weight);
        var loss = _engine.ReduceSum(output, null);
        var grads = tape.ComputeGradients(loss,
            new[] { input });

        var analyticalGrad = grads[input];

        // Numerical gradient for input (central differences)
        double eps = 1e-5;
        for (int idx = 0; idx < Math.Min(8, input.Length); idx++)
        {
            double orig = input[idx];

            input[idx] = orig + eps;
            var outPlus = _engine.OctonionMatMulTensor(input, weight);
            double lossPlus = _engine.TensorSum(outPlus);

            input[idx] = orig - eps;
            var outMinus = _engine.OctonionMatMulTensor(input, weight);
            double lossMinus = _engine.TensorSum(outMinus);

            input[idx] = orig;

            double numericalGrad = (lossPlus - lossMinus) / (2 * eps);
            double analyticalVal = analyticalGrad[idx];

            Assert.True(Math.Abs(numericalGrad - analyticalVal) < 1e-3,
                $"Input grad mismatch at [{idx}]: numerical={numericalGrad:F6}, analytical={analyticalVal:F6}, " +
                $"diff={Math.Abs(numericalGrad - analyticalVal):E2}");
        }
    }

    // ================================================================
    // Issue #145: RBFKernel backward
    // ================================================================

    [Fact]
    public void RBFKernel_Backward_ProducesGradientsForAllInputsIncludingEpsilons()
    {
        // RBFKernel: output[b,c] = exp(-epsilon[c] * ||input[b] - centers[c]||^2)
        // input: [batch=3, features=4]
        // centers: [numCenters=2, features=4]
        // epsilons: [numCenters=2]
        var rng = new Random(42);
        var input = new Tensor<double>([3, 4]);
        var centers = new Tensor<double>([2, 4]);
        var epsilons = new Tensor<double>([2]);
        for (int i = 0; i < input.Length; i++) input[i] = rng.NextDouble();
        for (int i = 0; i < centers.Length; i++) centers[i] = rng.NextDouble();
        epsilons[0] = 1.0;
        epsilons[1] = 2.0;

        using var tape = new GradientTape<double>();
        var output = _engine.RBFKernel(input, centers, epsilons);
        var loss = _engine.ReduceSum(output, null);

        var grads = tape.ComputeGradients(loss,
            new[] { input, centers, epsilons });

        // All 3 inputs must have gradients
        Assert.True(grads.ContainsKey(input), "No gradient for input");
        Assert.True(grads.ContainsKey(centers), "No gradient for centers");
        Assert.True(grads.ContainsKey(epsilons), "No gradient for epsilons");

        // Shapes must match
        Assert.Equal(input.Shape.ToArray(), grads[input].Shape.ToArray());
        Assert.Equal(centers.Shape.ToArray(), grads[centers].Shape.ToArray());
        Assert.Equal(epsilons.Shape.ToArray(), grads[epsilons].Shape.ToArray());

        // Gradients must be non-zero
        bool inputNonZero = false, centersNonZero = false, epsilonsNonZero = false;
        for (int i = 0; i < grads[input].Length; i++)
            if (Math.Abs(grads[input][i]) > 1e-10) { inputNonZero = true; break; }
        for (int i = 0; i < grads[centers].Length; i++)
            if (Math.Abs(grads[centers][i]) > 1e-10) { centersNonZero = true; break; }
        for (int i = 0; i < grads[epsilons].Length; i++)
            if (Math.Abs(grads[epsilons][i]) > 1e-10) { epsilonsNonZero = true; break; }

        Assert.True(inputNonZero, "Input gradient is all zeros");
        Assert.True(centersNonZero, "Centers gradient is all zeros");
        Assert.True(epsilonsNonZero, "Epsilons gradient is all zeros — per-center widths not flowing");
    }

    [Fact]
    public void RBFKernel_Backward_NumericalGradientCheck_Epsilons()
    {
        // Verify epsilon gradients match numerical (central differences).
        // This specifically tests the fix for issue #145 where only
        // epsilonsData[0] was saved.
        var input = new Tensor<double>([2, 3]);
        var centers = new Tensor<double>([2, 3]);
        var epsilons = new Tensor<double>([2]);

        input[0] = 1; input[1] = 2; input[2] = 3;
        input[3] = 4; input[4] = 5; input[5] = 6;
        centers[0] = 0.5; centers[1] = 1.5; centers[2] = 2.5;
        centers[3] = 3.5; centers[4] = 4.5; centers[5] = 5.5;
        epsilons[0] = 0.5;
        epsilons[1] = 1.5;

        // Analytical gradient
        using var tape = new GradientTape<double>();
        var output = _engine.RBFKernel(input, centers, epsilons);
        var loss = _engine.ReduceSum(output, null);
        var grads = tape.ComputeGradients(loss,
            new[] { epsilons });

        var analyticalGrad = grads[epsilons];

        // Numerical gradient for epsilons
        double eps = 1e-5;
        for (int idx = 0; idx < epsilons.Length; idx++)
        {
            double orig = epsilons[idx];

            epsilons[idx] = orig + eps;
            var outPlus = _engine.RBFKernel(input, centers, epsilons);
            double lossPlus = _engine.TensorSum(outPlus);

            epsilons[idx] = orig - eps;
            var outMinus = _engine.RBFKernel(input, centers, epsilons);
            double lossMinus = _engine.TensorSum(outMinus);

            epsilons[idx] = orig;

            double numericalGrad = (lossPlus - lossMinus) / (2 * eps);
            double analyticalVal = analyticalGrad[idx];

            Assert.True(Math.Abs(numericalGrad - analyticalVal) < 1e-3,
                $"Epsilon grad mismatch at [{idx}]: numerical={numericalGrad:F6}, analytical={analyticalVal:F6}, " +
                $"diff={Math.Abs(numericalGrad - analyticalVal):E2}. " +
                $"If epsilon[0] and epsilon[1] produce identical gradients, the old scalar-only bug is still present.");
        }

        // Verify the two epsilon gradients are DIFFERENT (proves per-center widths work)
        Assert.NotEqual(analyticalGrad[0], analyticalGrad[1]);
    }
}
