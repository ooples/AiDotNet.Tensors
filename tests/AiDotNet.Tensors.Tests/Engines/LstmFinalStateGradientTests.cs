using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Engines.Optimization;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.OpenCL;

namespace AiDotNet.Tensors.Tests.Engines;

[Collection("CompilationGlobalState")]
public class LstmFinalStateGradientTests
{
    private static T Value<T>(double value) => (T)Convert.ChangeType(value, typeof(T));
    private static Tensor<T> Filled<T>(int[] shape, int seed)
    {
        var random = new Random(seed);
        var result = new Tensor<T>(shape);
        for (int i = 0; i < result.Length; i++) result[i] = Value<T>((random.NextDouble() - 0.5) * 0.6);
        return result;
    }
    // x, WIh, WHh, BIh, BHh, H0, C0 -- all optional inputs exercised independently by the loss.
    private static Tensor<T>[] Inputs<T>(int batch = 2, int steps = 3) => new[]
    {
        Filled<T>(new[] { batch, steps, 2 }, 1), Filled<T>(new[] { 8, 2 }, 2),
        Filled<T>(new[] { 8, 2 }, 3), Filled<T>(new[] { 8 }, 4), Filled<T>(new[] { 8 }, 5),
        Filled<T>(new[] { batch, 2 }, 6), Filled<T>(new[] { batch, 2 }, 7)
    };
    private static Tensor<T> Loss<T>(CpuEngine e, Tensor<T>[] p, bool sequence, int mask, bool reference = false)
    {
        Tensor<T> y, h, c;
        if (!reference)
            y = e.LstmSequenceForward(p[0], p[5], p[6], p[1], p[2], p[3], p[4], out h, out c, sequence);
        else
        {
            int batch = p[0].Shape[0], steps = p[0].Shape[1];
            h = p[5]; c = p[6];
            var outputs = new List<Tensor<T>>();
            for (int t = 0; t < steps; t++)
            {
                var x = e.Reshape(e.TensorSlice(p[0], new[] { 0, t, 0 }, new[] { batch, 1, 2 }), new[] { batch, 2 });
                var gates = e.TensorAdd(e.TensorAdd(e.TensorMatMulTransposed(x, p[1]), e.TensorMatMulTransposed(h, p[2])), e.TensorAdd(p[3], p[4]));
                Tensor<T> Gate(int g) => e.TensorSlice(gates, new[] { 0, 2 * g }, new[] { batch, 2 });
                var i = e.Sigmoid(Gate(0)); var f = e.Sigmoid(Gate(1));
                var g = e.Tanh(Gate(2)); var o = e.Sigmoid(Gate(3));
                c = e.TensorAdd(e.TensorMultiply(f, c), e.TensorMultiply(i, g));
                h = e.TensorMultiply(o, e.Tanh(c));
                outputs.Add(e.Reshape(h, new[] { batch, 1, 2 }));
            }
            y = sequence ? e.TensorConcatenate(outputs.ToArray(), 1) : h;
        }
        Tensor<T>? loss = null;
        void Add(Tensor<T> value)
        {
            var term = e.ReduceSum(e.TensorMultiply(value, value), null);
            loss = loss is null ? term : e.TensorAdd(loss, term);
        }
        if ((mask & 1) != 0) Add(y);
        if ((mask & 2) != 0) Add(h);
        if ((mask & 4) != 0) Add(c);
        return loss!;
    }
    private static void Near<T>(Tensor<T> expected, Tensor<T> actual, double tolerance)
    {
        Assert.Equal(expected.Shape, actual.Shape);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(Convert.ToDouble(expected[i]) - Convert.ToDouble(actual[i])) <= tolerance,
                $"element {i}: {expected[i]} vs {actual[i]}");
    }

    [Theory]
    [InlineData(true, true, 1)] [InlineData(true, true, 2)] [InlineData(true, true, 4)] [InlineData(true, true, 7)]
    [InlineData(false, true, 1)] [InlineData(false, true, 2)] [InlineData(false, true, 4)] [InlineData(false, true, 7)]
    [InlineData(true, false, 7)] [InlineData(false, false, 7)]
    public void IndependentStateAdjoints_MatchRecurrence(bool single, bool sequence, int mask)
    {
        if (single) Compare<float>(sequence, mask, 17, 3e-6); else Compare<double>(sequence, mask, 2, 1e-11);
    }
    private static void Compare<T>(bool sequence, int mask, int batch, double tolerance, CpuEngine? engine = null)
    {
        var e = engine ?? new CpuEngine(); var p = Inputs<T>(batch);
        Tensor<T> actual;
        Dictionary<Tensor<T>, Tensor<T>> gradients;
        using (var tape = new GradientTape<T>())
        {
            actual = Loss(e, p, sequence, mask);
            Assert.True(tape.EntryCount < 20, $"Expected one fused recurrence plus output views, got {tape.EntryCount} tape entries");
            gradients = tape.ComputeGradients(actual, p);
        }
        using var referenceTape = new GradientTape<T>();
        var expected = Loss(new CpuEngine(), p, sequence, mask, reference: true);
        var reference = referenceTape.ComputeGradients(expected, p);
        Near(expected, actual, tolerance);
        foreach (var parameter in p) Near(reference[parameter], gradients[parameter], tolerance);
    }

    [Theory]
    [InlineData(true)] [InlineData(false)]
    public void AllStateAdjoints_MatchFiniteDifferences(bool sequence)
    {
        var e = new CpuEngine(); var p = Inputs<double>(1, 2);
        Dictionary<Tensor<double>, Tensor<double>> gradients;
        using (var tape = new GradientTape<double>()) gradients = tape.ComputeGradients(Loss(e, p, sequence, 7), p);
        const double epsilon = 1e-5;
        double Evaluate() { using var tape = new GradientTape<double>(); return Loss(e, p, sequence, 7)[0]; }
        foreach (var parameter in p)
            for (int i = 0; i < parameter.Length; i++)
            {
                double original = parameter[i];
                parameter[i] = original + epsilon; double plus = Evaluate();
                parameter[i] = original - epsilon; double minus = Evaluate();
                parameter[i] = original;
                Assert.True(Math.Abs((plus - minus) / (2 * epsilon) - gradients[parameter][i]) < 1e-7);
            }
    }

    [Theory]
    [InlineData(true, true)] [InlineData(true, false)] [InlineData(false, true)] [InlineData(false, false)]
    public void CompiledStatefulTraining_ReplaysOneRecurrence(bool single, bool sequence)
    {
        if (single) Replay<float>(sequence, 2e-6); else Replay<double>(sequence, 1e-11);
    }
    private static void Replay<T>(bool sequence, double tolerance, CpuEngine? engine = null)
    {
        var e = engine ?? new CpuEngine(); var p = Inputs<T>();
        ICompiledTrainingPlan<T> plan;
        using (var scope = GraphMode.Enable())
        {
            Loss(e, p, sequence, 7);
            plan = scope.CompileTraining(p);
        }
        using (plan)
        {
            Assert.True(Assert.IsType<CompiledTrainingPlan<T>>(plan).ForwardStepCount < 25);
            plan.ConfigureOptimizer(OptimizerType.SGD, learningRate: 0.0f);
            for (int step = 0; step < 3; step++)
            {
                p[0][0] = Value<T>(0.13 * step); p[6][0] = Value<T>(-0.2 * step);
                Tensor<T> expected;
                Dictionary<Tensor<T>, Tensor<T>> gradients;
                using (var tape = new GradientTape<T>())
                { expected = Loss(new CpuEngine(), p, sequence, 7); gradients = tape.ComputeGradients(expected, p); }
                var actual = plan.Step();
                Near(expected, actual, tolerance);
                foreach (var parameter in p) Near(gradients[parameter], parameter.Grad!, tolerance);
            }
        }
    }

    [Theory]
    [InlineData(true)] [InlineData(false)]
    public void ChunkBoundaries_PropagateStateGradients(bool single)
    {
        if (single) Chunks<float>(3e-6); else Chunks<double>(1e-11);
    }
    private static void Chunks<T>(double tolerance, CpuEngine? engine = null)
    {
        var e = engine ?? new CpuEngine(); var p = Inputs<T>();
        Dictionary<Tensor<T>, Tensor<T>> expected;
        Tensor<T> expectedLoss;
        using (var tape = new GradientTape<T>())
        { expectedLoss = Loss(e, p, true, 7); expected = tape.ComputeGradients(expectedLoss, p); }
        using var chunkTape = new GradientTape<T>();
        var firstInput = e.TensorSlice(p[0], new[] { 0, 0, 0 }, new[] { 2, 1, 2 });
        var secondInput = e.TensorSlice(p[0], new[] { 0, 1, 0 }, new[] { 2, 2, 2 });
        var first = e.LstmSequenceForward(firstInput, p[5], p[6], p[1], p[2], p[3], p[4], out var h1, out var c1, true);
        var second = e.LstmSequenceForward(secondInput, h1, c1, p[1], p[2], p[3], p[4], out var h, out var c, true);
        var sequence = e.TensorConcatenate(new[] { first, second }, 1);
        Tensor<T> SquareSum(Tensor<T> value) => e.ReduceSum(e.TensorMultiply(value, value), null);
        var loss = e.TensorAdd(SquareSum(sequence), e.TensorAdd(SquareSum(h), SquareSum(c)));
        var actual = chunkTape.ComputeGradients(loss, p);
        Near(expectedLoss, loss, tolerance);
        foreach (var parameter in p) Near(expected[parameter], actual[parameter], tolerance);
    }

    [SkippableFact]
    public void OpenCl_StateAdjointsAndCompiledReplay_MatchCpu()
    {
        using var backend = new OpenClBackend();
        Skip.IfNot(backend.IsAvailable, "OpenCL is not available.");
        using var direct = new DirectGpuEngine(backend);
        using var engine = new DirectGpuTensorEngine(direct);
        foreach (bool sequence in new[] { true, false })
        {
            foreach (int mask in new[] { 1, 2, 4, 7 }) Compare<float>(sequence, mask, 3, 3e-5, engine);
            Replay<float>(sequence, 3e-5, engine);
        }
        Chunks<float>(3e-5, engine);
        foreach (bool states in new[] { false, true })
            foreach (bool biases in new[] { false, true }) OptionalInputs<float>(states, biases, engine, 3e-5);
    }

    [Theory]
    [InlineData(false, false)] [InlineData(false, true)] [InlineData(true, false)] [InlineData(true, true)]
    public void AbsentStateAndBiasInputs_PreserveGradients(bool states, bool biases)
        => OptionalInputs<double>(states, biases, new CpuEngine(), 1e-11);

    private static void OptionalInputs<T>(bool states, bool biases, CpuEngine e, double tolerance)
    {
        var p = Inputs<T>();
        if (!states) { p[5] = new Tensor<T>(new[] { 2, 2 }); p[6] = new Tensor<T>(new[] { 2, 2 }); }
        if (!biases) { p[3] = new Tensor<T>(new[] { 8 }); p[4] = new Tensor<T>(new[] { 8 }); }
        var sources = p.Take(3).Concat(biases ? p.Skip(3).Take(2) : Array.Empty<Tensor<T>>())
            .Concat(states ? p.Skip(5) : Array.Empty<Tensor<T>>()).ToArray();
        Dictionary<Tensor<T>, Tensor<T>> actual;
        using (var tape = new GradientTape<T>())
        {
            var y = e.LstmSequenceForward(p[0], states ? p[5] : null, states ? p[6] : null,
                p[1], p[2], biases ? p[3] : null, biases ? p[4] : null, out var h, out var c, true);
            Tensor<T> Square(Tensor<T> v) => e.ReduceSum(e.TensorMultiply(v, v), null);
            actual = tape.ComputeGradients(e.TensorAdd(Square(y), e.TensorAdd(Square(h), Square(c))), sources);
        }
        using var reference = new GradientTape<T>();
        var expected = reference.ComputeGradients(Loss(new CpuEngine(), p, true, 7, reference: true), sources);
        foreach (var parameter in sources) Near(expected[parameter], actual[parameter], tolerance);
    }

    [Fact]
    public void EmptySequence_PreservesInitialStateGradients()
    {
        var e = new CpuEngine(); var p = Inputs<double>(2, 0);
        using var tape = new GradientTape<double>();
        var y = e.LstmSequenceForward(p[0], p[5], p[6], p[1], p[2], null, null, out var h, out var c, false);
        var loss = e.ReduceSum(e.TensorAdd(y, e.TensorAdd(h, c)), null);
        var gradients = tape.ComputeGradients(loss, new[] { p[5], p[6] });
        Assert.All(gradients[p[5]].ToArray(), value => Assert.Equal(2, value));
        Assert.All(gradients[p[6]].ToArray(), value => Assert.Equal(1, value));
    }
}
