using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

/// <summary>
/// The compiled training plan's specialized Divide backward when an operand has several consumers. A shared operand's
/// gradient is the sum of every consumer's contribution, so the delegate must add into its buffer unless its write is
/// the step's first (AccumulatesInto), as the Add, Multiply and MatMul specializations do.
/// </summary>
/// <remarks>
/// Divide used to copy its two gradients into the plan's buffers whatever the operand's consumer count, so a tensor
/// that was a divisor and also a factor elsewhere kept only the divide's share: <c>h / s + h * s</c> trained on a
/// gradient that was wrong by its whole multiply term (AiDotNet FEDformer's Fourier cross-attention divides and
/// multiplies by one sech, and every parameter upstream of it updated 20-80% off the eager tape). Gradients are checked against the eager tape over two steps, so a buffer that is
/// not re-zeroed between steps would also show up.
/// </remarks>
[Collection("CompilationGlobalState")]
public sealed class CompiledSharedOperandElementwiseBackwardTests : IDisposable
{
    private const int Rows = 6, Inner = 4, Columns = 5;

    private readonly IEngine _priorEngine = AiDotNetEngine.Current;

    public CompiledSharedOperandElementwiseBackwardTests()
    {
        AiDotNetEngine.Current = new CpuEngine();
    }

    public void Dispose()
    {
        AiDotNetEngine.Current = _priorEngine;
    }

    private static Tensor<double> Make(int[] shape, Random rng)
    {
        int n = 1;
        foreach (var d in shape) n *= d;
        var data = new double[n];
        for (int i = 0; i < n; i++) data[i] = rng.NextDouble() * 2 - 1;
        return new Tensor<double>(data, shape);
    }

    private static Tensor<double> MeanSquare(IEngine e, Tensor<double> x)
        => e.ReduceMean(e.TensorMultiply(x, x), new[] { 0, 1 }, keepDims: false);

    /// <summary>h / s + h * s with s = h * h + 1: s is the divisor and a factor, and h feeds all three.</summary>
    private static Tensor<double> DivisorAlsoAFactor(IEngine e, Tensor<double>[] p, Tensor<double> x)
    {
        var h = e.TensorMatMul(p[0], x);
        var s = e.TensorAddScalar(e.TensorMultiply(h, h), 1.0);
        return MeanSquare(e, e.TensorAdd(e.TensorDivide(h, s), e.TensorMultiply(h, s)));
    }

    /// <summary>h / (h * h + 1): the numerator is shared with the divisor's own subgraph.</summary>
    private static Tensor<double> NumeratorInsideTheDivisor(IEngine e, Tensor<double>[] p, Tensor<double> x)
    {
        var h = e.TensorMatMul(p[0], x);
        return MeanSquare(e, e.TensorDivide(h, e.TensorAddScalar(e.TensorMultiply(h, h), 1.0)));
    }

    public static IEnumerable<object[]> Graphs()
    {
        yield return new object[] { nameof(DivisorAlsoAFactor), "specialized:TensorDivide" };
        yield return new object[] { nameof(NumeratorInsideTheDivisor), "specialized:TensorDivide" };
    }

    private static Func<IEngine, Tensor<double>[], Tensor<double>, Tensor<double>> Graph(string name) => name switch
    {
        nameof(DivisorAlsoAFactor) => DivisorAlsoAFactor,
        nameof(NumeratorInsideTheDivisor) => NumeratorInsideTheDivisor,
        _ => throw new ArgumentOutOfRangeException(nameof(name)),
    };

    [Theory]
    [MemberData(nameof(Graphs))]
    public void SharedOperandGradientsMatchTheEagerTape(string graphName, string specializedKind)
    {
        var graph = Graph(graphName);
        var rng = new Random(2345);
        var parameters = new[] { Make([Rows, Inner], rng) };
        var input = Make([Inner, Columns], rng);
        var engine = new CpuEngine();

        Dictionary<Tensor<double>, Tensor<double>> expected;
        using (var tape = new GradientTape<double>())
        {
            var eagerLoss = graph(engine, parameters, input);
            expected = tape.ComputeGradients(eagerLoss, parameters);
        }

        ICompiledTrainingPlan<double> plan;
        using (var scope = GraphMode.EnableTraining(parameters))
        {
            var loss = graph(engine, parameters, input);
            plan = scope.CompileTraining(parameters, loss);
        }

        using (plan)
        {
            // The specialized delegate is the code under test; a generic fallback would pass without exercising it.
            Assert.Contains(specializedKind, ((CompiledTrainingPlan<double>)plan).BackwardStepKinds);

            for (int step = 0; step < 2; step++)
            {
                plan.Step();
                var gradients = plan.Gradients;
                for (int p = 0; p < parameters.Length; p++)
                {
                    var want = expected.TryGetValue(parameters[p], out var tapeGradient)
                        ? tapeGradient.ToArray()
                        : new double[parameters[p].Length];
                    var got = gradients[p].ToArray();
                    double scale = 1e-12;
                    foreach (var v in want) scale = Math.Max(scale, Math.Abs(v));
                    for (int i = 0; i < want.Length; i++)
                    {
                        Assert.True(Math.Abs(got[i] - want[i]) <= 1e-10 * scale,
                            $"{graphName}, step {step}, parameter {p}, element {i}: compiled {got[i]:G12}, eager {want[i]:G12}");
                    }
                }
            }
        }
    }
}