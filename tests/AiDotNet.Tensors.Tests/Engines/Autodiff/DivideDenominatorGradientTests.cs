using System;
using System.Threading.Tasks;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Autodiff;

[Collection("EngineCurrentGlobalState")]
public sealed class DivideDenominatorGradientTests : IDisposable
{
    private readonly IEngine _priorEngine = AiDotNetEngine.Current;
    private readonly CpuEngine _engine = new();

    public DivideDenominatorGradientTests() => AiDotNetEngine.Current = _engine;

    public void Dispose() => AiDotNetEngine.Current = _priorEngine;

    [Fact]
    public async Task NumeratorAndDenominatorGradients_MatchFiniteDifferences()
    {
        await Task.Yield();

        var numerator = new Tensor<double>(new[] { 4 });
        var denominator = new Tensor<double>(new[] { 4 });
        var projection = new Tensor<double>(new[] { 4 });
        for (int i = 0; i < 4; i++)
        {
            numerator[i] = -0.4 + (0.3 * i);
            denominator[i] = 0.55 + (0.2 * i);
            projection[i] = 0.2 - (0.07 * i);
        }

        Tensor<double> numeratorGradient;
        Tensor<double> denominatorGradient;
        using (var tape = new GradientTape<double>())
        {
            var ratio = _engine.TensorDivide(numerator, denominator);
            var objective = _engine.ReduceSum(_engine.TensorMultiply(ratio, projection));
            var gradients = tape.ComputeGradients(objective, new[] { numerator, denominator });
            numeratorGradient = gradients[numerator];
            denominatorGradient = gradients[denominator];
        }

        AssertGradient(numerator, numeratorGradient, numerator, denominator, projection);
        AssertGradient(denominator, denominatorGradient, numerator, denominator, projection);
    }

    [Fact]
    public async Task RangeReducedAtanComposite_GradientMatchesFiniteDifferences()
    {
        await Task.Yield();

        var input = new Tensor<double>(new[] { 4 });
        var projection = new Tensor<double>(new[] { 4 });
        input[0] = -0.73;
        input[1] = -0.42;
        input[2] = 0.38;
        input[3] = 1.27;
        for (int i = 0; i < projection.Length; i++) projection[i] = 0.2 - (0.07 * i);

        Tensor<double> analytical;
        using (var tape = new GradientTape<double>())
        {
            var objective = _engine.ReduceSum(_engine.TensorMultiply(RangeReducedAtan(input), projection));
            analytical = tape.ComputeGradients(objective, new[] { input })[input];
        }

        const double step = 1e-6;
        for (int i = 0; i < input.Length; i++)
        {
            double original = input[i];
            input[i] = original + step;
            double plus = Project(RangeReducedAtan(input), projection);
            input[i] = original - step;
            double minus = Project(RangeReducedAtan(input), projection);
            input[i] = original;

            double numerical = (plus - minus) / (2.0 * step);
            Assert.True(
                Math.Abs(analytical[i] - numerical) < 1e-8,
                $"index {i}: analytical={analytical[i]:G17}, numerical={numerical:G17}");
        }
    }

    [Theory]
    [InlineData(false, false)]
    [InlineData(true, false)]
    [InlineData(true, true)]
    public async Task AngularDifferenceFanOut_RealAndImaginaryGradientsMatchFiniteDifferences(
        bool includeFrequencyDifference,
        bool includeTimeDifference)
    {
        await Task.Yield();

        var real = CreateGrid(0.55, 0.025);
        var imaginary = CreateGrid(-0.35, 0.02);
        var targetReal = CreateGrid(0.44, 0.021);
        var targetImaginary = CreateGrid(-0.46, 0.017);

        Tensor<double> realGradient;
        Tensor<double> imaginaryGradient;
        using (var tape = new GradientTape<double>())
        {
            var objective = AngularDifferenceLoss(
                real, imaginary, targetReal, targetImaginary,
                includeFrequencyDifference, includeTimeDifference);
            var gradients = tape.ComputeGradients(objective, new[] { real, imaginary });
            realGradient = gradients[real];
            imaginaryGradient = gradients[imaginary];
        }

        AssertAngularGradient(
            real, realGradient, real, imaginary, targetReal, targetImaginary,
            includeFrequencyDifference, includeTimeDifference);
        AssertAngularGradient(
            imaginary, imaginaryGradient, real, imaginary, targetReal, targetImaginary,
            includeFrequencyDifference, includeTimeDifference);
    }

    private void AssertAngularGradient(
        Tensor<double> source,
        Tensor<double> analytical,
        Tensor<double> real,
        Tensor<double> imaginary,
        Tensor<double> targetReal,
        Tensor<double> targetImaginary,
        bool includeFrequencyDifference,
        bool includeTimeDifference)
    {
        const double step = 1e-6;
        for (int i = 0; i < source.Length; i++)
        {
            double original = source[i];
            source[i] = original + step;
            double plus = AngularDifferenceLoss(
                real, imaginary, targetReal, targetImaginary,
                includeFrequencyDifference, includeTimeDifference)[0];
            source[i] = original - step;
            double minus = AngularDifferenceLoss(
                real, imaginary, targetReal, targetImaginary,
                includeFrequencyDifference, includeTimeDifference)[0];
            source[i] = original;

            double numerical = (plus - minus) / (2.0 * step);
            Assert.True(
                Math.Abs(analytical[i] - numerical) < 1e-7,
                $"index {i}: analytical={analytical[i]:G17}, numerical={numerical:G17}");
        }
    }

    private Tensor<double> AngularDifferenceLoss(
        Tensor<double> real,
        Tensor<double> imaginary,
        Tensor<double> targetReal,
        Tensor<double> targetImaginary,
        bool includeFrequencyDifference,
        bool includeTimeDifference)
    {
        var phase = Phase(real, imaginary);
        var target = Phase(targetReal, targetImaginary);
        var loss = MeanAbs(AntiWrap(_engine.TensorSubtract(phase, target)));
        if (includeFrequencyDifference)
        {
            loss = _engine.TensorAdd(loss, MeanAbs(AntiWrap(_engine.TensorSubtract(
                Difference(phase, axis: 1), Difference(target, axis: 1)))));
        }
        if (includeTimeDifference)
        {
            loss = _engine.TensorAdd(loss, MeanAbs(AntiWrap(_engine.TensorSubtract(
                Difference(phase, axis: 0), Difference(target, axis: 0)))));
        }
        return loss;
    }

    private Tensor<double> Phase(Tensor<double> real, Tensor<double> imaginary)
    {
        var signReal = SignStar(real);
        var signImaginary = SignStar(imaginary);
        var safeReal = _engine.TensorAdd(real, _engine.TensorMultiply(signReal, ConstantLike(real, 1e-7)));
        var principal = RangeReducedAtan(_engine.TensorDivide(imaginary, safeReal));
        var correction = _engine.TensorMultiply(
            _engine.TensorMultiply(
                signImaginary,
                _engine.TensorSubtract(signReal, ConstantLike(real, 1.0))),
            ConstantLike(real, Math.PI / 2.0));
        return _engine.TensorSubtract(principal, correction);
    }

    private Tensor<double> SignStar(Tensor<double> input)
    {
        var sign = _engine.TensorSign(input);
        return _engine.TensorAdd(
            sign,
            _engine.TensorSubtract(ConstantLike(input, 1.0), _engine.TensorAbs(sign)));
    }

    private Tensor<double> AntiWrap(Tensor<double> input)
    {
        var turns = _engine.TensorRound(_engine.TensorMultiply(
            input, ConstantLike(input, 1.0 / (2.0 * Math.PI))));
        return _engine.TensorSubtract(
            input,
            _engine.TensorMultiply(turns, ConstantLike(input, 2.0 * Math.PI)));
    }

    private Tensor<double> Difference(Tensor<double> input, int axis)
    {
        int length = input.Shape[axis];
        return _engine.TensorSubtract(
            _engine.TensorNarrow(input, axis, 1, length - 1),
            _engine.TensorNarrow(input, axis, 0, length - 1));
    }

    private Tensor<double> MeanAbs(Tensor<double> input)
        => _engine.ReduceMean(
            _engine.TensorAbs(input),
            System.Linq.Enumerable.Range(0, input.Shape.Length).ToArray(),
            keepDims: false);

    private static Tensor<double> CreateGrid(double offset, double step)
    {
        var result = new Tensor<double>(new[] { 2, 5 });
        for (int i = 0; i < result.Length; i++) result[i] = offset + (step * i);
        return result;
    }

    private Tensor<double> RangeReducedAtan(Tensor<double> input)
    {
        var sign = _engine.TensorSign(input);
        var magnitude = _engine.TensorAbs(input);
        var one = ConstantLike(input, 1.0);
        var isLarge = _engine.TensorGreaterThan(magnitude, one);
        var reciprocal = _engine.TensorReciprocal(_engine.TensorClampMin(magnitude, 1.0));
        var reduced = _engine.TensorWhere(isLarge, reciprocal, magnitude);
        var squared = _engine.TensorMultiply(reduced, reduced);

        var polynomial = ConstantLike(input, 0.0208351);
        polynomial = _engine.TensorAdd(_engine.TensorMultiply(polynomial, squared), ConstantLike(input, -0.0851330));
        polynomial = _engine.TensorAdd(_engine.TensorMultiply(polynomial, squared), ConstantLike(input, 0.1801410));
        polynomial = _engine.TensorAdd(_engine.TensorMultiply(polynomial, squared), ConstantLike(input, -0.3302995));
        polynomial = _engine.TensorAdd(_engine.TensorMultiply(polynomial, squared), ConstantLike(input, 0.9998660));
        polynomial = _engine.TensorMultiply(reduced, polynomial);

        var large = _engine.TensorSubtract(ConstantLike(input, Math.PI / 2.0), polynomial);
        var magnitudeAtan = _engine.TensorWhere(isLarge, large, polynomial);
        return _engine.TensorMultiply(sign, magnitudeAtan);
    }

    private static Tensor<double> ConstantLike(Tensor<double> like, double value)
    {
        var result = new Tensor<double>(like.Shape.ToArray());
        for (int i = 0; i < result.Length; i++) result[i] = value;
        return result;
    }

    private void AssertGradient(
        Tensor<double> source,
        Tensor<double> analytical,
        Tensor<double> numerator,
        Tensor<double> denominator,
        Tensor<double> projection)
    {
        const double step = 1e-6;
        for (int i = 0; i < source.Length; i++)
        {
            double original = source[i];
            source[i] = original + step;
            double plus = Project(numerator, denominator, projection);
            source[i] = original - step;
            double minus = Project(numerator, denominator, projection);
            source[i] = original;

            double numerical = (plus - minus) / (2.0 * step);
            Assert.True(
                Math.Abs(analytical[i] - numerical) < 1e-9,
                $"index {i}: analytical={analytical[i]:G17}, numerical={numerical:G17}");
        }
    }

    private double Project(
        Tensor<double> numerator,
        Tensor<double> denominator,
        Tensor<double> projection)
    {
        var ratio = _engine.TensorDivide(numerator, denominator);
        double sum = 0.0;
        for (int i = 0; i < ratio.Length; i++) sum += ratio[i] * projection[i];
        return sum;
    }

    private static double Project(Tensor<double> value, Tensor<double> projection)
    {
        double sum = 0.0;
        for (int i = 0; i < value.Length; i++) sum += value[i] * projection[i];
        return sum;
    }
}
