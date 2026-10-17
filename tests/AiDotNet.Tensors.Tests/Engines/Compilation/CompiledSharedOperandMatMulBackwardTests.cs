using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

/// <summary>
/// The compiled training plan's MatMul backward when an operand has several consumers. Each gradient is one transposed
/// GEMM into the plan's buffer: overwriting (beta = 0) for a single-consumer operand, accumulating (beta = 1) for a
/// shared one, whose buffer the plan zeroes at the start of every step. These steps used to take the generic backward
/// (rented gradients, then a copy or a separate add into the plan buffer).
/// </summary>
/// <remarks>
/// The graph is N-BEATS' doubly-residual stack in AiDotNet's column-major layout: the residual feeds both a block's
/// first layer and its subtraction, the last hidden activation feeds both theta projections, the backcast / forecast
/// bases are constants, and the biases are [out, 1] columns broadcast over the batch. Gradients are checked against
/// the eager tape on both BLAS routes (native beta = 1, and the managed accumulate kernels), over two steps so a
/// buffer that is not re-zeroed between steps would show up doubled.
/// </remarks>
[Collection("CompilationGlobalState")]
public sealed class CompiledSharedOperandMatMulBackwardTests : IDisposable
{
    private const int Blocks = 3, Layers = 2, Hidden = 48, Lookback = 8, Horizon = 4, Theta = 3, Batch = 40;

    private readonly IEngine _priorEngine = AiDotNetEngine.Current;

    public CompiledSharedOperandMatMulBackwardTests()
    {
        AiDotNetEngine.Current = new CpuEngine();
    }

    public void Dispose()
    {
        AiDotNetEngine.Current = _priorEngine;
    }

    private sealed class Model<T>
    {
        public readonly Tensor<T>[][] Weights = new Tensor<T>[Blocks][];
        public readonly Tensor<T>[][] Biases = new Tensor<T>[Blocks][];
        public readonly Tensor<T>[] BasisBackcast = new Tensor<T>[Blocks];
        public readonly Tensor<T>[] BasisForecast = new Tensor<T>[Blocks];
        public readonly List<Tensor<T>> Parameters = new();
    }

    private static Tensor<T> Make<T>(int[] shape, Random rng, double scale, Func<double, T> convert)
    {
        int n = 1;
        foreach (var d in shape) n *= d;
        var data = new T[n];
        for (int i = 0; i < n; i++) data[i] = convert((rng.NextDouble() * 2 - 1) * scale);
        return new Tensor<T>(data, shape);
    }

    private static Model<T> BuildModel<T>(Func<double, T> convert)
    {
        var rng = new Random(2295);
        var model = new Model<T>();
        for (int b = 0; b < Blocks; b++)
        {
            model.Weights[b] = new Tensor<T>[Layers + 2];
            model.Biases[b] = new Tensor<T>[Layers + 2];
            for (int l = 0; l < Layers + 2; l++)
            {
                int inSize = l == 0 ? Lookback : Hidden;
                int outSize = l < Layers ? Hidden : Theta;
                model.Weights[b][l] = Make([outSize, inSize], rng, Math.Sqrt(2.0 / inSize), convert);
                model.Biases[b][l] = Make([outSize, 1], rng, 0.1, convert);
                model.Parameters.Add(model.Weights[b][l]);
                model.Parameters.Add(model.Biases[b][l]);
            }
            model.BasisBackcast[b] = Make([Lookback, Theta], rng, 1.0, convert);
            model.BasisForecast[b] = Make([Horizon, Theta], rng, 1.0, convert);
        }
        return model;
    }

    // AiDotNet NBEATSModel.RunForwardStack + NBEATSBlock.ForwardTapeColumns + an MSE loss.
    private static Tensor<T> Loss<T>(IEngine e, Model<T> m, Tensor<T> input, Tensor<T> target)
    {
        var residual = e.TensorPermute(input, new[] { 1, 0 });
        Tensor<T>? forecastSum = null;
        for (int b = 0; b < Blocks; b++)
        {
            var h = residual;
            for (int l = 0; l < Layers; l++)
                h = e.ReLU(e.TensorAdd(e.TensorMatMul(m.Weights[b][l], h), m.Biases[b][l]));
            var thetaB = e.TensorAdd(e.TensorMatMul(m.Weights[b][Layers], h), m.Biases[b][Layers]);
            var thetaF = e.TensorAdd(e.TensorMatMul(m.Weights[b][Layers + 1], h), m.Biases[b][Layers + 1]);
            residual = e.TensorSubtract(residual, e.TensorMatMul(m.BasisBackcast[b], thetaB));
            var forecast = e.TensorMatMul(m.BasisForecast[b], thetaF);
            forecastSum = forecastSum is null ? forecast : e.TensorAdd(forecastSum, forecast);
        }
        var prediction = e.TensorPermute(forecastSum ?? throw new InvalidOperationException("No blocks."), new[] { 1, 0 });
        var diff = e.TensorSubtract(prediction, target);
        return e.ReduceMean(e.TensorMultiply(diff, diff), new[] { 0, 1 }, keepDims: false);
    }

    private static void CheckAgainstEagerTape<T>(Func<double, T> convert, Func<T, double> toDouble, double tolerance)
    {
        var model = BuildModel(convert);
        var rng = new Random(77);
        var input = Make([Batch, Lookback], rng, 1.0, convert);
        var target = Make([Batch, Horizon], rng, 1.0, convert);
        var parameters = model.Parameters.ToArray();
        var engine = new CpuEngine();

        Dictionary<Tensor<T>, Tensor<T>> expected;
        using (var tape = new GradientTape<T>())
        {
            var eagerLoss = Loss(engine, model, input, target);
            expected = tape.ComputeGradients(eagerLoss, parameters);
        }

        ICompiledTrainingPlan<T> plan;
        using (var scope = GraphMode.EnableTraining(parameters))
        {
            var loss = Loss(engine, model, input, target);
            plan = scope.CompileTraining(parameters, loss);
        }

        using (plan)
        {
            var kinds = ((CompiledTrainingPlan<T>)plan).BackwardStepKinds;
            Assert.DoesNotContain("generic:TensorMatMul", kinds);
            Assert.Equal(Blocks * (Layers + 4), kinds.Count(k => k == "specialized:TensorMatMul"));

            for (int step = 0; step < 2; step++)
            {
                plan.Step();
                var gradients = plan.Gradients;
                for (int p = 0; p < parameters.Length; p++)
                {
                    // The last block's backcast branch feeds a residual nothing reads, so the tape has no gradient for
                    // it; the plan must leave that gradient at zero.
                    var want = expected.TryGetValue(parameters[p], out var tapeGradient)
                        ? tapeGradient.ToArray()
                        : new T[parameters[p].Length];
                    var got = gradients[p].ToArray();
                    double scale = 1e-12;
                    foreach (var v in want) scale = Math.Max(scale, Math.Abs(toDouble(v)));
                    for (int i = 0; i < want.Length; i++)
                    {
                        double diff = Math.Abs(toDouble(got[i]) - toDouble(want[i]));
                        Assert.True(diff <= tolerance * scale,
                            $"step {step}, parameter {p} [{string.Join(",", parameters[p]._shape)}], element {i}: "
                            + $"compiled {toDouble(got[i]):G9}, eager {toDouble(want[i]):G9}");
                    }
                }
            }
        }
    }

    [Fact]
    public void NBeatsStack_Float_GradientsMatchEagerTape()
    {
        CheckAgainstEagerTape<float>(v => (float)v, v => v, 1e-4);
    }

    [Fact]
    public void NBeatsStack_Double_GradientsMatchEagerTape()
    {
        CheckAgainstEagerTape<double>(v => v, v => v, 1e-10);
    }

    /// <summary>
    /// One [rows, 1] bias added to two [rows, cols] tensors: its gradient is the sum of both contributions' row sums,
    /// so the per-row reduction must ADD into the bias gradient for the second consumer, and the sum must restart
    /// from zero every step.
    /// </summary>
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void SharedColumnBias_GradientIsTheSumOfBothRowSums(bool asDouble)
    {
        if (asDouble) CheckSharedColumnBias<double>(v => v, v => v, 1e-12);
        else CheckSharedColumnBias<float>(v => (float)v, v => v, 1e-5);
    }

    private static void CheckSharedColumnBias<T>(Func<double, T> convert, Func<T, double> toDouble, double tolerance)
    {
        const int rows = 24, cols = 33;
        var rng = new Random(5);
        var x1 = Make<T>([rows, cols], rng, 1.0, convert);
        var x2 = Make<T>([rows, cols], rng, 1.0, convert);
        var bias = Make<T>([rows, 1], rng, 1.0, convert);
        var parameters = new[] { x1, x2, bias };
        var engine = new CpuEngine();

        ICompiledTrainingPlan<T> plan;
        using (var scope = GraphMode.EnableTraining(parameters))
        {
            var y1 = engine.TensorAdd(x1, bias);
            var y2 = engine.TensorAdd(x2, bias);
            var loss = engine.TensorAdd(
                engine.ReduceSum(engine.TensorMultiply(y1, y1), null),
                engine.ReduceSum(engine.TensorMultiply(y2, y2), null));
            plan = scope.CompileTraining(parameters, loss);
        }

        // d/db sum((x1 + b)^2) + sum((x2 + b)^2) = row sums of 2(x1 + b) + 2(x2 + b).
        var a1 = x1.ToArray(); var a2 = x2.ToArray(); var bv = bias.ToArray();
        var expected = new double[rows];
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++)
                expected[r] += 2 * (toDouble(a1[r * cols + c]) + toDouble(bv[r])) + 2 * (toDouble(a2[r * cols + c]) + toDouble(bv[r]));

        using (plan)
        {
            Assert.Equal(2, ((CompiledTrainingPlan<T>)plan).BackwardStepKinds.Count(k => k == "specialized:TensorBroadcastAdd"));
            for (int step = 0; step < 2; step++)
            {
                plan.Step();
                var got = plan.Gradients[2].ToArray();
                double scale = expected.Max(v => Math.Abs(v)) + 1e-12;
                for (int r = 0; r < rows; r++)
                    Assert.True(Math.Abs(toDouble(got[r]) - expected[r]) <= tolerance * scale,
                        $"step {step}, db[{r}] = {toDouble(got[r]):G9}, expected {expected[r]:G9}");
            }
        }
    }
}