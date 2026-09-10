using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Autodiff;

/// <summary>
/// Forward and gradient guards for the tensor-level inverse trigonometric ops.
/// </summary>
/// <remarks>
/// <para>
/// Issue #905 reports the failure these pin down. Before <c>TensorAsin</c>/<c>TensorAcos</c>/
/// <c>TensorAtan</c>/<c>TensorAtan2</c> existed, the only inverse trigonometry on the tensor
/// surface was <c>NativeAtan2</c>, which is registered non-differentiable. A loss defined over an
/// angle therefore compiled, ran, and trained - while the gradient through the angle was dropped
/// on the floor. Nothing threw. The model simply never learned that path.
/// </para>
/// <para>
/// That makes <see cref="Atan2_GradientReachesTheInput_NotSilentlyZero"/> the load-bearing test
/// here: every forward test below would have passed against the old severed behaviour too.
/// </para>
/// </remarks>
[Collection("EngineCurrentGlobalState")]
public class InverseTrigGradientTests
{
    private readonly IEngine _engine = AiDotNetEngine.Current;

    /// <summary>
    /// Tolerance for assertions that route only through the op under test.
    /// </summary>
    private const double ExactTolerance = 1e-9;

    /// <summary>
    /// Tolerance for assertions that compose the op under test with other engine ops.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <c>AiDotNetEngine.Current</c> is <c>DirectGpuTensorEngine</c> wherever a GPU is present, and
    /// on a device without native fp64 its elementwise kernels evaluate a <c>Tensor&lt;double&gt;</c>
    /// in single precision. That is a property of the engine, not of these ops - a bare
    /// <c>TensorAdd</c> of two doubles reproduces it with no inverse trigonometry involved - and the
    /// repo's other gradient tests use the same 1e-6 for the same reason. Every derivative below is
    /// an O(1) quantity, so a wrong formula misses by far more than this.
    /// </para>
    /// </remarks>
    private const double ComposedTolerance = 1e-6;

    // Forward correctness

    [Fact]
    public void Asin_Acos_Atan_MatchSystemMath()
    {
        var data = new[] { -0.9, -0.5, -0.25, 0.0, 0.25, 0.5, 0.9 };
        var x = new Tensor<double>(data, new[] { data.Length });

        var asin = _engine.TensorAsin(x);
        var acos = _engine.TensorAcos(x);
        var atan = _engine.TensorAtan(x);

        for (int i = 0; i < data.Length; i++)
        {
            Assert.Equal(Math.Asin(data[i]), asin[i], ExactTolerance);
            Assert.Equal(Math.Acos(data[i]), acos[i], ExactTolerance);
            Assert.Equal(Math.Atan(data[i]), atan[i], ExactTolerance);
        }
    }

    [Fact]
    public void Atan_HandlesTheFullRealLine()
    {
        // atan is the only one of the three defined everywhere, and its asymptotic behaviour is
        // what makes it usable as a soft clamp - so check the tails, not only the middle.
        var data = new[] { -1e6, -1000.0, -1.0, 0.0, 1.0, 1000.0, 1e6 };
        var x = new Tensor<double>(data, new[] { data.Length });

        var atan = _engine.TensorAtan(x);

        for (int i = 0; i < data.Length; i++)
        {
            Assert.Equal(Math.Atan(data[i]), atan[i], ExactTolerance);
            Assert.True(Math.Abs(atan[i]) < Math.PI / 2);
        }
    }

    [Fact]
    public void AsinPlusAcos_IsHalfPi()
    {
        var data = new[] { -1.0, -0.6, 0.0, 0.3, 1.0 };
        var x = new Tensor<double>(data, new[] { data.Length });

        var sum = _engine.TensorAdd(_engine.TensorAsin(x), _engine.TensorAcos(x));

        for (int i = 0; i < data.Length; i++)
        {
            Assert.Equal(Math.PI / 2, sum[i], ComposedTolerance);
        }
    }

    [Fact]
    public void Asin_RoundTripsThroughSin()
    {
        var data = new[] { -0.95, -0.4, 0.0, 0.4, 0.95 };
        var x = new Tensor<double>(data, new[] { data.Length });

        var roundTrip = _engine.TensorSin(_engine.TensorAsin(x));

        for (int i = 0; i < data.Length; i++)
        {
            Assert.Equal(data[i], roundTrip[i], ComposedTolerance);
        }
    }

    [Fact]
    public void Asin_Acos_Atan_WorkInSinglePrecision()
    {
        var data = new[] { -0.75f, -0.1f, 0.3f, 0.8f };
        var x = new Tensor<float>(data, new[] { data.Length });

        var asin = _engine.TensorAsin(x);
        var acos = _engine.TensorAcos(x);
        var atan = _engine.TensorAtan(x);

        for (int i = 0; i < data.Length; i++)
        {
            Assert.Equal(MathF.Asin(data[i]), asin[i], 1e-6f);
            Assert.Equal(MathF.Acos(data[i]), acos[i], 1e-6f);
            Assert.Equal(MathF.Atan(data[i]), atan[i], 1e-6f);
        }
    }

    // Four-quadrant behaviour

    [Fact]
    public void Atan2_CoversAllFourQuadrants()
    {
        var ys = new[] { 1.0, 1.0, -1.0, -1.0, 0.0, 3.0 };
        var xs = new[] { 1.0, -1.0, -1.0, 1.0, -2.0, 0.0 };
        var y = new Tensor<double>(ys, new[] { ys.Length });
        var x = new Tensor<double>(xs, new[] { xs.Length });

        var angle = _engine.TensorAtan2(y, x);

        for (int i = 0; i < ys.Length; i++)
        {
            Assert.Equal(Math.Atan2(ys[i], xs[i]), angle[i], ExactTolerance);
        }
    }

    [Fact]
    public void Atan2_DiffersFromAtanOfTheRatio_WhenXIsNegative()
    {
        // The whole point of the two-argument form: atan(y/x) collapses the second and third
        // quadrants onto the fourth and first. If this ever fails, someone has quietly
        // reimplemented atan2 as a division.
        var y = new Tensor<double>(new[] { 1.0, -1.0 }, new[] { 2 });
        var x = new Tensor<double>(new[] { -1.0, -1.0 }, new[] { 2 });

        var angle = _engine.TensorAtan2(y, x);

        Assert.Equal(3 * Math.PI / 4, angle[0], ExactTolerance);
        Assert.Equal(-3 * Math.PI / 4, angle[1], ExactTolerance);
        Assert.True(Math.Abs(angle[0] - Math.Atan(1.0 / -1.0)) > 1.0);
    }

    [Fact]
    public void Atan2_RecoversTheAngleFromItsSineAndCosine()
    {
        var angles = new[] { -3.0, -1.2, 0.0, 0.7, 2.9 };
        var sin = new double[angles.Length];
        var cos = new double[angles.Length];
        for (int i = 0; i < angles.Length; i++)
        {
            sin[i] = Math.Sin(angles[i]);
            cos[i] = Math.Cos(angles[i]);
        }

        var recovered = _engine.TensorAtan2(
            new Tensor<double>(sin, new[] { angles.Length }),
            new Tensor<double>(cos, new[] { angles.Length }));

        for (int i = 0; i < angles.Length; i++)
        {
            Assert.Equal(angles[i], recovered[i], 1e-12);
        }
    }

    [Fact]
    public void Atan2_RejectsMismatchedShapes()
    {
        var y = new Tensor<double>(new[] { 1.0, 2.0, 3.0, 4.0 }, new[] { 2, 2 });
        var x = new Tensor<double>(new[] { 1.0, 2.0, 3.0, 4.0 }, new[] { 4 });

        Assert.Throws<ArgumentException>(() => _engine.TensorAtan2(y, x));
    }

    // Gradients

    [Fact]
    public void Asin_Gradient_IsInverseSqrtOneMinusSquare()
    {
        var data = new[] { -0.8, -0.3, 0.0, 0.3, 0.8 };
        var x = new Tensor<double>(data, new[] { data.Length });

        using var tape = new GradientTape<double>();
        var loss = _engine.ReduceSum(_engine.TensorAsin(x), null);
        var gx = tape.ComputeGradients(loss, new[] { x })[x];

        for (int i = 0; i < data.Length; i++)
        {
            var expected = 1.0 / Math.Sqrt(1.0 - (data[i] * data[i]));
            Assert.Equal(expected, gx[i], ComposedTolerance);
        }
    }

    [Fact]
    public void Acos_Gradient_IsTheNegatedAsinGradient()
    {
        var data = new[] { -0.8, -0.3, 0.0, 0.3, 0.8 };
        var x = new Tensor<double>(data, new[] { data.Length });

        using var tape = new GradientTape<double>();
        var loss = _engine.ReduceSum(_engine.TensorAcos(x), null);
        var gx = tape.ComputeGradients(loss, new[] { x })[x];

        for (int i = 0; i < data.Length; i++)
        {
            var expected = -1.0 / Math.Sqrt(1.0 - (data[i] * data[i]));
            Assert.Equal(expected, gx[i], ComposedTolerance);
        }
    }

    [Fact]
    public void Atan_Gradient_IsInverseOnePlusSquare()
    {
        var data = new[] { -5.0, -1.0, 0.0, 1.0, 5.0 };
        var x = new Tensor<double>(data, new[] { data.Length });

        using var tape = new GradientTape<double>();
        var loss = _engine.ReduceSum(_engine.TensorAtan(x), null);
        var gx = tape.ComputeGradients(loss, new[] { x })[x];

        for (int i = 0; i < data.Length; i++)
        {
            var expected = 1.0 / (1.0 + (data[i] * data[i]));
            Assert.Equal(expected, gx[i], ComposedTolerance);
        }
    }

    [Fact]
    public void Atan2_Gradient_MatchesTheClosedFormForBothInputs()
    {
        var ys = new[] { 1.0, 2.0, -1.5, 0.5 };
        var xs = new[] { 2.0, -1.0, -0.5, 3.0 };
        var y = new Tensor<double>(ys, new[] { ys.Length });
        var x = new Tensor<double>(xs, new[] { xs.Length });

        using var tape = new GradientTape<double>();
        var loss = _engine.ReduceSum(_engine.TensorAtan2(y, x), null);
        var grads = tape.ComputeGradients(loss, new[] { y, x });

        for (int i = 0; i < ys.Length; i++)
        {
            var denominator = (xs[i] * xs[i]) + (ys[i] * ys[i]);
            Assert.Equal(xs[i] / denominator, grads[y][i], ComposedTolerance);
            Assert.Equal(-ys[i] / denominator, grads[x][i], ComposedTolerance);
        }
    }

    [Fact]
    public void Atan_Gradient_MatchesFiniteDifferences()
    {
        // An independent check on the analytic derivative: central differences in double, on a
        // smooth op with a bounded second derivative, agree well past the tolerance below.
        var data = new[] { -3.0, -0.7, 0.2, 1.4 };
        const double h = 1e-6;

        var x = new Tensor<double>((double[])data.Clone(), new[] { data.Length });
        using var tape = new GradientTape<double>();
        var loss = _engine.ReduceSum(_engine.TensorAtan(x), null);
        var gx = tape.ComputeGradients(loss, new[] { x })[x];

        for (int i = 0; i < data.Length; i++)
        {
            var numeric = (Math.Atan(data[i] + h) - Math.Atan(data[i] - h)) / (2 * h);
            Assert.Equal(numeric, gx[i], ComposedTolerance);
        }
    }

    [Fact]
    public void Atan2_GradientReachesTheInput_NotSilentlyZero()
    {
        // The regression this whole file exists for. Every forward assertion above would also pass
        // against an implementation that computes the right angle and records nothing on the tape;
        // this one would not. A severed op yields either a missing entry or an all-zero gradient,
        // and both are caught here.
        var y = new Tensor<double>(new[] { 0.6, -1.3, 2.0 }, new[] { 3 });
        var x = new Tensor<double>(new[] { 0.8, 0.4, -1.1 }, new[] { 3 });

        using var tape = new GradientTape<double>();
        var loss = _engine.ReduceSum(_engine.TensorAtan2(y, x), null);
        var grads = tape.ComputeGradients(loss, new[] { y, x });

        Assert.True(grads.ContainsKey(y), "no gradient recorded for the numerator of atan2");
        Assert.True(grads.ContainsKey(x), "no gradient recorded for the denominator of atan2");

        var sawNonZero = false;
        for (int i = 0; i < 3; i++)
        {
            if (grads[y][i] != 0.0 || grads[x][i] != 0.0)
            {
                sawNonZero = true;
            }
        }

        Assert.True(sawNonZero, "atan2 gradients are all zero - the tape connection is severed");
    }

    [Fact]
    public void Asin_GradientSurvivesAChainThroughTheAngle()
    {
        // The realistic shape from issue #905: an angle is recovered mid-graph and the loss is
        // defined downstream of it. The gradient must survive the whole round trip.
        var data = new[] { -0.5, 0.25, 0.75 };
        var x = new Tensor<double>(data, new[] { data.Length });

        using var tape = new GradientTape<double>();
        var angle = _engine.TensorAsin(x);
        var loss = _engine.ReduceSum(_engine.TensorMultiply(angle, angle), null);
        var gx = tape.ComputeGradients(loss, new[] { x })[x];

        for (int i = 0; i < data.Length; i++)
        {
            // d/dx asin(x)^2 = 2*asin(x)/sqrt(1 - x^2)
            var expected = 2 * Math.Asin(data[i]) / Math.Sqrt(1.0 - (data[i] * data[i]));
            Assert.Equal(expected, gx[i], ComposedTolerance);
        }
    }
}
