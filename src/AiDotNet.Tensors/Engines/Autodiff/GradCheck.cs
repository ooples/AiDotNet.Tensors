using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Text;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Autodiff;

/// <summary>Settings of a <see cref="GradCheck"/> run. Unset tolerances and step follow the element type.</summary>
public sealed class GradCheckOptions
{
    /// <summary>
    /// Central-difference step. Null: 1e-6 for <see cref="double"/> (as <c>torch.autograd.gradcheck</c>), 1e-3 for
    /// other element types, whose rounding would swamp a smaller step.
    /// </summary>
    public double? Epsilon { get; init; }

    /// <summary>Absolute tolerance. Null: 1e-5 for <see cref="double"/>, 1e-3 otherwise.</summary>
    public double? AbsoluteTolerance { get; init; }

    /// <summary>Relative tolerance. Null: 1e-3 for <see cref="double"/>, 1e-2 otherwise.</summary>
    public double? RelativeTolerance { get; init; }

    /// <summary>
    /// Also checks second derivatives (<c>torch.autograd.gradgradcheck</c>): the gradient of
    /// Σ_i ⟨∂(Σ f·u)/∂x_i, v_i⟩, taken by a second backward through a <c>createGraph</c> first backward, against
    /// its central differences.
    /// </summary>
    public bool CheckSecondOrder { get; init; }

    /// <summary>
    /// Contracts the output with a seeded random weight u in [-1, 1] (the default, which cannot hide an error behind a
    /// symmetric cancellation); false contracts with ones, i.e. checks the gradient of the plain sum.
    /// </summary>
    public bool RandomOutputGradient { get; init; } = true;

    /// <summary>Seed of the random output (and second-order) weights.</summary>
    public int Seed { get; init; }

    /// <summary>Checks only the first this-many elements of each input; null checks every element.</summary>
    public int? MaxElementsPerInput { get; init; }

    /// <summary>Indices of the inputs to check; null checks all of them.</summary>
    public IReadOnlyList<int>? InputsToCheck { get; init; }

    /// <summary>How many of the worst mismatches the result lists.</summary>
    public int MaxReportedMismatches { get; init; } = 10;

    /// <summary>The engine that contracts the output for the analytical backward; null uses <see cref="AiDotNetEngine.Current"/>.</summary>
    public IEngine? Engine { get; init; }

    internal GradCheckOptions WithSecondOrder() => new()
    {
        Epsilon = Epsilon,
        AbsoluteTolerance = AbsoluteTolerance,
        RelativeTolerance = RelativeTolerance,
        CheckSecondOrder = true,
        RandomOutputGradient = RandomOutputGradient,
        Seed = Seed,
        MaxElementsPerInput = MaxElementsPerInput,
        InputsToCheck = InputsToCheck,
        MaxReportedMismatches = MaxReportedMismatches,
        Engine = Engine,
    };
}

/// <summary>One element whose analytical and numerical derivatives disagree beyond the tolerance.</summary>
public sealed class GradCheckMismatch
{
    internal GradCheckMismatch(int order, int inputIndex, int elementIndex, double analytical, double numerical, double allowed)
    {
        Order = order;
        InputIndex = inputIndex;
        ElementIndex = elementIndex;
        Analytical = analytical;
        Numerical = numerical;
        AbsoluteError = Math.Abs(analytical - numerical);
        double scale = Math.Max(Math.Abs(analytical), Math.Abs(numerical));
        RelativeError = scale > 0 ? AbsoluteError / scale : 0;
        Allowed = allowed;
    }

    /// <summary>1 for a first derivative, 2 for a second derivative.</summary>
    public int Order { get; }

    /// <summary>Index of the input among the checked function's inputs.</summary>
    public int InputIndex { get; }

    /// <summary>Flat (row-major) index of the element within that input.</summary>
    public int ElementIndex { get; }

    /// <summary>The derivative the backward computed.</summary>
    public double Analytical { get; }

    /// <summary>The central-difference derivative.</summary>
    public double Numerical { get; }

    /// <summary>|analytical - numerical|.</summary>
    public double AbsoluteError { get; }

    /// <summary>|analytical - numerical| / max(|analytical|, |numerical|).</summary>
    public double RelativeError { get; }

    /// <summary>The error the tolerances allowed here: atol + rtol·|numerical|.</summary>
    public double Allowed { get; }

    /// <inheritdoc/>
    public override string ToString() => string.Format(CultureInfo.InvariantCulture,
        "order {0}, input {1}[{2}]: analytical {3:G8} vs numerical {4:G8} (|diff| {5:E3} > allowed {6:E3})",
        Order, InputIndex, ElementIndex, Analytical, Numerical, AbsoluteError, Allowed);
}

/// <summary>The outcome of a <see cref="GradCheck"/> run.</summary>
public sealed class GradCheckResult
{
    internal GradCheckResult(int elementsChecked, int failedElements, double maxAbsoluteError, double maxRelativeError,
        IReadOnlyList<GradCheckMismatch> worstMismatches, bool secondOrderChecked, IReadOnlyList<int> inputsWithoutGradient)
    {
        ElementsChecked = elementsChecked;
        FailedElements = failedElements;
        MaxAbsoluteError = maxAbsoluteError;
        MaxRelativeError = maxRelativeError;
        WorstMismatches = worstMismatches;
        SecondOrderChecked = secondOrderChecked;
        InputsWithoutGradient = inputsWithoutGradient;
    }

    /// <summary>True when every checked derivative is within tolerance.</summary>
    public bool Passed => FailedElements == 0;

    /// <summary>Number of derivatives compared (first and second order together).</summary>
    public int ElementsChecked { get; }

    /// <summary>Number of derivatives outside tolerance.</summary>
    public int FailedElements { get; }

    /// <summary>Largest |analytical - numerical| over every compared derivative.</summary>
    public double MaxAbsoluteError { get; }

    /// <summary>Largest |analytical - numerical| / max(|analytical|, |numerical|) over every compared derivative.</summary>
    public double MaxRelativeError { get; }

    /// <summary>The worst disagreements, largest excess over the allowed error first.</summary>
    public IReadOnlyList<GradCheckMismatch> WorstMismatches { get; }

    /// <summary>True when second derivatives were checked too.</summary>
    public bool SecondOrderChecked { get; }

    /// <summary>
    /// Checked inputs the backward gave no gradient (compared as zero, as <c>torch.autograd.gradcheck</c> treats an
    /// undefined gradient: a pass means the output really does not depend on them).
    /// </summary>
    public IReadOnlyList<int> InputsWithoutGradient { get; }

    /// <summary>Throws an <see cref="InvalidOperationException"/> describing the mismatches when the check failed.</summary>
    public void ThrowIfFailed()
    {
        if (!Passed) throw new InvalidOperationException(ToString());
    }

    /// <inheritdoc/>
    public override string ToString()
    {
        var text = new StringBuilder();
        text.Append(string.Format(CultureInfo.InvariantCulture,
            "GradCheck {0}: {1}/{2} derivatives out of tolerance{3}, max |diff| {4:E3}, max rel {5:E3}",
            Passed ? "PASSED" : "FAILED", FailedElements, ElementsChecked,
            SecondOrderChecked ? " (first and second order)" : "", MaxAbsoluteError, MaxRelativeError));
        if (InputsWithoutGradient.Count > 0)
            text.Append(", no gradient for input(s) ").Append(string.Join(", ", InputsWithoutGradient));
        foreach (var mismatch in WorstMismatches) text.Append('\n').Append("  ").Append(mismatch);
        return text.ToString();
    }
}

/// <summary>
/// Gradient checking against central finite differences (<c>torch.autograd.gradcheck</c> /
/// <c>torch.autograd.gradgradcheck</c>): runs a function under a <see cref="GradientTape{T}"/>, compares each input's
/// recorded gradient with (f(x + ε) - f(x - ε)) / 2ε element by element, and reports the worst mismatches.
/// </summary>
/// <remarks>
/// <para>
/// A tensor-valued function is reduced to a scalar by contracting its output with a fixed weight u (seeded random by
/// default), so the check covers the whole vector-Jacobian product, not only the gradient of the sum. An element fails
/// when |analytical - numerical| &gt; atol + rtol·|numerical|, PyTorch's criterion.
/// </para>
/// <para>
/// The numerical side perturbs the inputs in place (and restores them) and evaluates the function with recording
/// suppressed, so the function must read its inputs afresh on each call. Run checks in <see cref="double"/>: in
/// <see cref="float"/> the finite differences carry rounding error of order ulp/ε, which the looser default
/// tolerances for other element types only partly absorb.
/// </para>
/// </remarks>
public static class GradCheck
{
    /// <summary>Checks the gradient of a one-input function.</summary>
    public static GradCheckResult Check<T>(Func<Tensor<T>, Tensor<T>> function, Tensor<T> input, GradCheckOptions? options = null)
    {
        if (function == null) throw new ArgumentNullException(nameof(function));
        if (input == null) throw new ArgumentNullException(nameof(input));
        return CheckMany(inputs => function(inputs[0]), new[] { input }, options);
    }

    /// <summary>Checks the gradients of a function of several inputs.</summary>
    public static GradCheckResult CheckMany<T>(Func<IReadOnlyList<Tensor<T>>, Tensor<T>> function, IReadOnlyList<Tensor<T>> inputs,
        GradCheckOptions? options = null)
    {
        if (function == null) throw new ArgumentNullException(nameof(function));
        if (inputs == null) throw new ArgumentNullException(nameof(inputs));
        for (int i = 0; i < inputs.Count; i++)
            if (inputs[i] is null) throw new ArgumentException($"inputs[{i}] is null.", nameof(inputs));
        return new Run<T>(function, inputs, options ?? new GradCheckOptions()).Execute();
    }

    /// <summary>Checks first and second derivatives of a one-input function (<c>torch.autograd.gradgradcheck</c>).</summary>
    public static GradCheckResult GradGradCheck<T>(Func<Tensor<T>, Tensor<T>> function, Tensor<T> input, GradCheckOptions? options = null)
        => Check(function, input, (options ?? new GradCheckOptions()).WithSecondOrder());

    /// <summary>Checks first and second derivatives of a function of several inputs.</summary>
    public static GradCheckResult GradGradCheckMany<T>(Func<IReadOnlyList<Tensor<T>>, Tensor<T>> function, IReadOnlyList<Tensor<T>> inputs,
        GradCheckOptions? options = null)
        => CheckMany(function, inputs, (options ?? new GradCheckOptions()).WithSecondOrder());

    /// <summary>Central-difference gradient of a scalar function: g[i] = (f(x + ε·e_i) - f(x - ε·e_i)) / 2ε.</summary>
    public static Tensor<T> NumericalGradient<T>(Func<Tensor<T>, T> scalarFunction, Tensor<T> input, double? epsilon = null)
    {
        if (scalarFunction == null) throw new ArgumentNullException(nameof(scalarFunction));
        if (input == null) throw new ArgumentNullException(nameof(input));
        var ops = MathHelper.GetNumericOperations<T>();
        var gradient = CentralDifferences(input, epsilon ?? DefaultEpsilon<T>(), () => ops.ToDouble(scalarFunction(input)), input.Length);
        return RoIAlignPlan.FromDoubles<T>(gradient, (int[])input._shape.Clone());
    }

    /// <summary>
    /// Central-difference vector-Jacobian product of a tensor function: the gradient of ⟨f(x), outputGradient⟩.
    /// </summary>
    public static Tensor<T> NumericalVectorJacobianProduct<T>(Func<Tensor<T>, Tensor<T>> function, Tensor<T> input, Tensor<T> outputGradient,
        double? epsilon = null)
    {
        if (function == null) throw new ArgumentNullException(nameof(function));
        if (input == null) throw new ArgumentNullException(nameof(input));
        return NumericalGradients(inputs => function(inputs[0]), new[] { input }, outputGradient, epsilon)[0];
    }

    /// <summary>Central-difference vector-Jacobian products of a function of several inputs, one per input.</summary>
    public static Tensor<T>[] NumericalGradients<T>(Func<IReadOnlyList<Tensor<T>>, Tensor<T>> function, IReadOnlyList<Tensor<T>> inputs,
        Tensor<T> outputGradient, double? epsilon = null)
    {
        if (function == null) throw new ArgumentNullException(nameof(function));
        if (inputs == null) throw new ArgumentNullException(nameof(inputs));
        if (outputGradient == null) throw new ArgumentNullException(nameof(outputGradient));
        var weight = RoIAlignPlan.ToDoubles(outputGradient);
        double step = epsilon ?? DefaultEpsilon<T>();
        var result = new Tensor<T>[inputs.Count];
        for (int i = 0; i < inputs.Count; i++)
        {
            var gradient = CentralDifferences(inputs[i], step, () => Contract(function(inputs), weight), inputs[i].Length);
            result[i] = RoIAlignPlan.FromDoubles<T>(gradient, (int[])inputs[i]._shape.Clone());
        }
        return result;
    }

    /// <summary>
    /// Vector-Jacobian products from the backward: the gradients of ⟨f(inputs), outputGradient⟩ recorded by a
    /// <see cref="GradientTape{T}"/>, zero for an input the output does not reach.
    /// </summary>
    public static Tensor<T>[] AnalyticalGradients<T>(Func<IReadOnlyList<Tensor<T>>, Tensor<T>> function, IReadOnlyList<Tensor<T>> inputs,
        Tensor<T> outputGradient, IEngine? engine = null)
    {
        if (function == null) throw new ArgumentNullException(nameof(function));
        if (inputs == null) throw new ArgumentNullException(nameof(inputs));
        if (outputGradient == null) throw new ArgumentNullException(nameof(outputGradient));
        var eng = engine ?? AiDotNetEngine.Current;
        using var tape = new GradientTape<T>();
        var output = function(inputs);
        var loss = eng.ReduceSum(eng.TensorMultiply(output, outputGradient), null);
        var grads = tape.ComputeGradients(loss, inputs.ToArray());
        return inputs.Select(input => grads.TryGetValue(input, out var g) && g is not null
            ? RoIAlignPlan.FromDoubles<T>(RoIAlignPlan.ToDoubles(g), (int[])input._shape.Clone())
            : new Tensor<T>((int[])input._shape.Clone())).ToArray();
    }

    /// <summary>
    /// Compares an analytical gradient with a numerical one element by element: an element fails when
    /// |analytical - numerical| &gt; absoluteTolerance + relativeTolerance·|numerical|.
    /// </summary>
    public static GradCheckResult Compare<T>(Tensor<T> analytical, Tensor<T> numerical, double absoluteTolerance, double relativeTolerance,
        int maxReportedMismatches = 10)
    {
        if (analytical == null) throw new ArgumentNullException(nameof(analytical));
        if (numerical == null) throw new ArgumentNullException(nameof(numerical));
        if (!analytical._shape.AsSpan().SequenceEqual(numerical._shape))
            throw new ArgumentException(
                $"Shapes differ: analytical [{string.Join(", ", analytical._shape)}], numerical [{string.Join(", ", numerical._shape)}].");
        var tally = new Tally(absoluteTolerance, relativeTolerance);
        var a = RoIAlignPlan.ToDoubles(analytical);
        var n = RoIAlignPlan.ToDoubles(numerical);
        for (int i = 0; i < a.Length; i++) tally.Add(1, 0, i, a[i], n[i]);
        return tally.ToResult(maxReportedMismatches, secondOrder: false, Array.Empty<int>());
    }

    internal static double DefaultEpsilon<T>() => typeof(T) == typeof(double) ? 1e-6 : 1e-3;

    private static double DefaultAbsoluteTolerance<T>() => typeof(T) == typeof(double) ? 1e-5 : 1e-3;

    private static double DefaultRelativeTolerance<T>() => typeof(T) == typeof(double) ? 1e-3 : 1e-2;

    // ⟨output, weight⟩ in double precision.
    private static double Contract<T>(Tensor<T> output, double[] weight)
    {
        var values = RoIAlignPlan.ToDoubles(output);
        if (values.Length != weight.Length)
            throw new ArgumentException($"The function's output has {values.Length} elements; the output gradient has {weight.Length}.");
        double sum = 0;
        for (int i = 0; i < values.Length; i++) sum += values[i] * weight[i];
        return sum;
    }

    // d evaluate / d input[i] for the first `count` elements, by central differences with the step the element type
    // actually represents (x ± ε rounds in float), with recording suppressed and each element restored afterwards.
    [ScalarKernel("Finite differences perturb one element at a time and re-run the whole function between writes.")]
    private static double[] CentralDifferences<T>(Tensor<T> input, double epsilon, Func<double> evaluate, int count)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        var gradient = new double[input.Length];
        using var noGrad = new NoGradScope<T>();
        for (int i = 0; i < count; i++)
        {
            T original = input[i];
            double center = ops.ToDouble(original);
            T plus = ops.FromDouble(center + epsilon);
            T minus = ops.FromDouble(center - epsilon);
            double step = ops.ToDouble(plus) - ops.ToDouble(minus);
            try
            {
                input[i] = plus;
                double high = evaluate();
                input[i] = minus;
                double low = evaluate();
                gradient[i] = step != 0 ? (high - low) / step : 0;
            }
            finally
            {
                input[i] = original;
            }
        }
        return gradient;
    }

    private sealed class Tally
    {
        private readonly double _atol, _rtol;
        private readonly List<GradCheckMismatch> _failures = new();
        private int _checked;
        private double _maxAbs, _maxRel;

        internal Tally(double atol, double rtol)
        {
            _atol = atol;
            _rtol = rtol;
        }

        internal void Add(int order, int inputIndex, int element, double analytical, double numerical)
        {
            _checked++;
            double diff = Math.Abs(analytical - numerical);
            double scale = Math.Max(Math.Abs(analytical), Math.Abs(numerical));
            if (!double.IsNaN(diff))
            {
                _maxAbs = Math.Max(_maxAbs, diff);
                if (scale > 0) _maxRel = Math.Max(_maxRel, diff / scale);
            }
            double allowed = _atol + _rtol * Math.Abs(numerical);
            // A NaN on either side is a failure, never a silent pass.
            if (!(diff <= allowed))
                _failures.Add(new GradCheckMismatch(order, inputIndex, element, analytical, numerical, allowed));
        }

        internal GradCheckResult ToResult(int maxReported, bool secondOrder, IReadOnlyList<int> inputsWithoutGradient)
        {
            var worst = _failures
                .OrderByDescending(f => double.IsNaN(f.AbsoluteError) ? double.PositiveInfinity : f.AbsoluteError - f.Allowed)
                .Take(Math.Max(0, maxReported))
                .ToList();
            return new GradCheckResult(_checked, _failures.Count, _maxAbs, _maxRel, worst, secondOrder, inputsWithoutGradient);
        }
    }

    // One check: the analytical first (and second) derivatives against their central differences.
    private sealed class Run<T>
    {
        private readonly Func<IReadOnlyList<Tensor<T>>, Tensor<T>> _function;
        private readonly IReadOnlyList<Tensor<T>> _inputs;
        private readonly GradCheckOptions _options;
        private readonly IEngine _engine;
        private readonly int[] _checkedInputs;
        private readonly double _epsilon;

        internal Run(Func<IReadOnlyList<Tensor<T>>, Tensor<T>> function, IReadOnlyList<Tensor<T>> inputs, GradCheckOptions options)
        {
            _function = function;
            _inputs = inputs;
            _options = options;
            _engine = options.Engine ?? AiDotNetEngine.Current;
            _checkedInputs = (options.InputsToCheck ?? Enumerable.Range(0, inputs.Count).ToArray()).ToArray();
            foreach (int index in _checkedInputs)
                if (index < 0 || index >= inputs.Count)
                    throw new ArgumentOutOfRangeException(nameof(options), $"InputsToCheck names input {index}; there are {inputs.Count}.");
            _epsilon = options.Epsilon ?? DefaultEpsilon<T>();
        }

        internal GradCheckResult Execute()
        {
            var tally = new Tally(_options.AbsoluteTolerance ?? DefaultAbsoluteTolerance<T>(),
                _options.RelativeTolerance ?? DefaultRelativeTolerance<T>());

            Tensor<T> probe;
            using (new NoGradScope<T>()) probe = _function(_inputs);
            var outputShape = (int[])probe._shape.Clone();
            var weight = Weights(probe.Length, _options.Seed);
            var weightTensor = RoIAlignPlan.FromDoubles<T>(weight, outputShape);

            var analytical = FirstOrder(weightTensor, createGraph: false, out var missing);
            foreach (int index in _checkedInputs)
            {
                var input = _inputs[index];
                var numerical = CentralDifferences(input, _epsilon, () => Contract(_function(_inputs), weight), Count(input));
                var a = analytical[index];
                for (int e = 0; e < Count(input); e++) tally.Add(1, index, e, a[e], numerical[e]);
            }

            if (_options.CheckSecondOrder) SecondOrder(tally, weightTensor);
            return tally.ToResult(_options.MaxReportedMismatches, _options.CheckSecondOrder, missing);
        }

        private int Count(Tensor<T> input) => Math.Min(input.Length, _options.MaxElementsPerInput ?? int.MaxValue);

        private double[] Weights(int length, int seed)
        {
            var weight = new double[length];
            if (!_options.RandomOutputGradient)
            {
                for (int i = 0; i < length; i++) weight[i] = 1;
                return weight;
            }
            var rng = RandomHelper.CreateSeededRandom(seed);
            for (int i = 0; i < length; i++) weight[i] = rng.NextDouble() * 2 - 1;
            return weight;
        }

        // Gradients of ⟨f(inputs), weight⟩ per input index (zeros where the backward gives none), as doubles.
        private Dictionary<int, double[]> FirstOrder(Tensor<T> weightTensor, bool createGraph, out List<int> missing)
        {
            using var tape = new GradientTape<T>();
            var output = _function(_inputs);
            var loss = _engine.ReduceSum(_engine.TensorMultiply(output, weightTensor), null);
            var grads = tape.ComputeGradients(loss, _checkedInputs.Select(i => _inputs[i]).ToArray(), createGraph);
            missing = new List<int>();
            var result = new Dictionary<int, double[]>();
            foreach (int index in _checkedInputs)
            {
                if (grads.TryGetValue(_inputs[index], out var g) && g is not null)
                {
                    result[index] = RoIAlignPlan.ToDoubles(g);
                }
                else
                {
                    missing.Add(index);
                    result[index] = new double[_inputs[index].Length];
                }
            }
            return result;
        }

        // h(x) = Σ_i ⟨∂⟨f(x), u⟩/∂x_i, v_i⟩: its gradient by double backward against its central differences.
        private void SecondOrder(Tally tally, Tensor<T> weightTensor)
        {
            var directions = new Dictionary<int, double[]>();
            foreach (int index in _checkedInputs)
                directions[index] = Weights(_inputs[index].Length, unchecked(_options.Seed * 31 + 17 + index));

            Dictionary<Tensor<T>, Tensor<T>> second;
            using (var tape = new GradientTape<T>())
            {
                var output = _function(_inputs);
                var loss = _engine.ReduceSum(_engine.TensorMultiply(output, weightTensor), null);
                var sources = _checkedInputs.Select(i => _inputs[i]).ToArray();
                var first = tape.ComputeGradients(loss, sources, createGraph: true);
                Tensor<T>? h = null;
                foreach (int index in _checkedInputs)
                {
                    if (!first.TryGetValue(_inputs[index], out var g) || g is null) continue;
                    var v = RoIAlignPlan.FromDoubles<T>(directions[index], (int[])g._shape.Clone());
                    var term = _engine.ReduceSum(_engine.TensorMultiply(g, v), null);
                    h = h is null ? term : _engine.TensorAdd(h, term);
                }
                second = h is null ? new Dictionary<Tensor<T>, Tensor<T>>() : tape.ComputeGradients(h, sources);
            }

            double H()
            {
                double sum = 0;
                var grads = FirstOrder(weightTensor, createGraph: false, out _);
                foreach (int index in _checkedInputs)
                {
                    var g = grads[index];
                    var v = directions[index];
                    for (int e = 0; e < g.Length; e++) sum += g[e] * v[e];
                }
                return sum;
            }

            foreach (int index in _checkedInputs)
            {
                var input = _inputs[index];
                var analytical = second.TryGetValue(input, out var g2) && g2 is not null
                    ? RoIAlignPlan.ToDoubles(g2)
                    : new double[input.Length];
                var numerical = CentralDifferencesRecording(input, _epsilon, H, Count(input));
                for (int e = 0; e < Count(input); e++) tally.Add(2, index, e, analytical[e], numerical[e]);
            }
        }

        // Central differences of a function that itself records (each evaluation runs a backward), so recording is
        // left on: CentralDifferences' NoGradScope would zero the inner first-order gradients.
        [ScalarKernel("Finite differences perturb one element at a time and re-run the whole function between writes.")]
        private static double[] CentralDifferencesRecording(Tensor<T> input, double epsilon, Func<double> evaluate, int count)
        {
            var ops = MathHelper.GetNumericOperations<T>();
            var gradient = new double[input.Length];
            for (int i = 0; i < count; i++)
            {
                T original = input[i];
                double center = ops.ToDouble(original);
                T plus = ops.FromDouble(center + epsilon);
                T minus = ops.FromDouble(center - epsilon);
                double step = ops.ToDouble(plus) - ops.ToDouble(minus);
                try
                {
                    input[i] = plus;
                    double high = evaluate();
                    input[i] = minus;
                    double low = evaluate();
                    gradient[i] = step != 0 ? (high - low) / step : 0;
                }
                finally
                {
                    input[i] = original;
                }
            }
            return gradient;
        }
    }
}
