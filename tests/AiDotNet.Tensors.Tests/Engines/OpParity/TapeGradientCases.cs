using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Tests.Engines.OpParity;

/// <summary>
/// Declared tape-gradient cases for <see cref="TapeGradientParityHarness"/>.
///
/// Each case builds its own inputs, runs the forward through IEngine, and returns the output plus the
/// tensors to differentiate against IN A FIXED ORDER. The order matters: the harness concatenates gradients
/// per source so a defect in ONE operand is visible even when the others are exact — which is precisely how
/// Scatter presented (d(values) exact, d(input) wrong by 3.14e-01).
///
/// CALLS GO THROUGH IEngine DELIBERATELY. DirectGpuTensorEngine implements most ops as EXPLICIT interface
/// implementations, so calling them on the concrete type silently resolves to the inherited CpuEngine
/// method — a GPU suite here once passed 6/6 while running entirely on CPU.
///
/// COVERAGE IS PARTIAL AND DELIBERATELY VISIBLE. Ops without a case report NO-CASE in the parity log rather
/// than passing silently. Automatic coverage of all ~176 registry ops is not possible from OpCase alone:
/// RunFloat closes over its inputs and returns only the output, and ComputeGradients keys results by tensor
/// REFERENCE, so the CPU and GPU runs produce dictionaries with no stable cross-engine alignment. Closing
/// that gap properly means exposing inputs from the registry, which is an OpCase change, not a harness one.
/// </summary>
public static class TapeGradientCases
{
    private static Tensor<float> Rand(int[] shape, int seed, double lo = -1.0, double hi = 1.0)
    {
        var rng = new Random(seed);
        var t = new Tensor<float>(shape);
        for (int i = 0; i < t.Length; i++) t[i] = (float)(lo + rng.NextDouble() * (hi - lo));
        return t;
    }

    /// <summary>Registers every declared case. Idempotent so repeated fixture construction is safe.</summary>
    public static void Register()
    {
        var cases = TapeGradientParityHarness.TapeCases;
        if (cases.Count > 0) return;

        // --- arithmetic backwards: gradient consumes forward-computed values, divergence probe applies ---

        cases["TensorCosh"] = engine =>
        {
            var x = Rand([4, 16], seed: 21, lo: -2.0, hi: 2.0);
            return (engine.TensorCosh(x), new[] { x });
        };

        cases["TensorSinh"] = engine =>
        {
            var x = Rand([4, 16], seed: 22, lo: -2.0, hi: 2.0);
            return (engine.TensorSinh(x), new[] { x });
        };

        cases["RFFT"] = engine =>
        {
            var x = Rand([8, 128], seed: 23);
            return (engine.RFFT(x), new[] { x });
        };

        // Three differentiable inputs. EPSILONS ARE UNIFORM ON PURPOSE: CpuEngine saves only epsilons[0] as
        // backward state, so with varying epsilons the CPU reference is itself wrong for centers 1..n and
        // agreeing with it would prove nothing.
        cases["RBFKernel"] = engine =>
        {
            var x = Rand([4, 3], seed: 51);
            var c = Rand([5, 3], seed: 52);
            var eps = new Tensor<float>([5]);
            for (int i = 0; i < eps.Length; i++) eps[i] = 0.75f;
            return (engine.RBFKernel(x, c, eps), new[] { x, c, eps });
        };

        // --- index/geometry backwards: gradient never reads forward-produced values, so CPU and GPU agree
        //     BIT-FOR-BIT and the harness must use the residency probe (see GradientIsExactOnBothEngines) ---

        cases["Upsample3D"] = engine =>
        {
            var x = Rand([2, 2, 3, 4, 4], seed: 24);
            return (engine.Upsample3D(x, 2, 2, 2), new[] { x });
        };

        cases["AffineGrid"] = engine =>
        {
            var theta = Rand([2, 2, 3], seed: 41);
            return (engine.AffineGrid(theta, 5, 7), new[] { theta });
        };

        // TIE-FREE INPUT REQUIRED: with equal pooled values CPU and GPU may select different argmax, which
        // leaves the forward identical while gradients land on different elements. Continuous random values
        // make exact ties measure-zero.
        cases["MaxPool3DWithIndices"] = engine =>
        {
            var x = Rand([2, 2, 4, 4, 4], seed: 61, lo: -5.0, hi: 5.0);
            return (engine.MaxPool3DWithIndices(x, new[] { 2, 2, 2 }, new[] { 2, 2, 2 }, out _), new[] { x });
        };
    }
}
