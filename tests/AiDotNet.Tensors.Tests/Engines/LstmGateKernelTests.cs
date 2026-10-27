using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.OpenCL;
using AiDotNet.Tensors.Engines.Optimization;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

[Collection("CompilationGlobalState")]
public class LstmGateKernelTests
{
    [Fact]
    public void WideFloatGates_MatchDecomposedGradientsWithSaturatedInputs()
    {
        var engine = new CpuEngine(); var gates = Values<float>(new[] { 2, 256 }); var old = Values<float>(new[] { 2, 64 });
        for (int i = 0; i < gates.Length; i++) gates[i] *= 20;
        var sources = new[] { gates, old };
        Tensor<float> actual; Dictionary<Tensor<float>, Tensor<float>> gradients;
        using (var tape = new GradientTape<float>())
        {
            actual = Loss(engine, gates, old, 3, false);
            gradients = tape.ComputeGradients(actual, sources);
        }
        using var reference = new GradientTape<float>();
        var expected = Loss(engine, gates, old, 3, true);
        var expectedGradients = reference.ComputeGradients(expected, sources);
        Near(expected, actual, 3e-4);
        foreach (var source in sources) Near(expectedGradients[source], gradients[source], 3e-5);
    }

    [Fact, Trait("Category", "RequiresOpenCL")]
    public void TensorCopy_PermutedSourceRemainsResident()
    {
        using var backend = new OpenClBackend(); Assert.True(backend.IsAvailable);
        using var direct = new DirectGpuEngine(backend); using var engine = new DirectGpuTensorEngine(direct);
        var source = engine.TensorAddScalar(new Tensor<float>(new float[] { 1, 2, 3, 4, 5, 6 }, new[] { 2, 3 }), 1f);
        var transposed = source.Transpose(new[] { 1, 0 });
        var destination = new Tensor<float>(new[] { 3, 2 });
        using (var scope = AiDotNet.Tensors.Engines.Diagnostics.GpuResidencyScope.Begin(captureOperations: true))
        {
            ((IEngine)engine).TensorCopy(transposed, destination);
            Assert.True(scope.Downloads == 0, string.Join("\n", scope.Events));
        }
        Assert.Equal(new float[] { 2, 5, 3, 6, 4, 7 }, destination.ToArray());
    }

    [Fact, Trait("Category", "RequiresOpenCL")]
    public void TensorCopy_CpuCowPeerRemainsIndependentOnGpuWrite()
    {
        using var backend = new OpenClBackend(); Assert.True(backend.IsAvailable);
        using var direct = new DirectGpuEngine(backend); using var engine = new DirectGpuTensorEngine(direct);
        var original = new Tensor<float>(new float[] { 1, 2, 3 }, new[] { 3 });
        var clone = (Tensor<float>)original.CloneShared();
        var replacement = engine.TensorAddScalar(new Tensor<float>(new[] { 3 }), 9f);
        using (var scope = AiDotNet.Tensors.Engines.Diagnostics.GpuResidencyScope.Begin(captureOperations: true))
        {
            ((IEngine)engine).TensorCopy(replacement, original);
            Assert.True(scope.Downloads == 0, string.Join("\n", scope.Events));
        }
        Assert.Equal(new float[] { 9, 9, 9 }, original.ToArray());
        Assert.Equal(new float[] { 1, 2, 3 }, clone.ToArray());
    }

    private static Tensor<T> Values<T>(int[] shape)
    {
        var result = new Tensor<T>(shape);
        for (int i = 0; i < result.Length; i++) result[i] = (T)Convert.ChangeType(Math.Sin(i + 0.4) * 1.7, typeof(T));
        return result;
    }
    private static Tensor<T> Loss<T>(CpuEngine e, Tensor<T> gates, Tensor<T> old, int mask, bool reference)
    {
        Tensor<T> h, c;
        if (reference)
        {
            int width = old.Shape[old.Rank - 1];
            Tensor<T> Gate(int i) => e.TensorNarrow(gates, gates.Rank - 1, i * width, width);
            c = e.TensorAdd(e.TensorMultiply(e.Sigmoid(Gate(1)), old), e.TensorMultiply(e.Sigmoid(Gate(0)), e.Tanh(Gate(2))));
            h = e.TensorMultiply(e.Sigmoid(Gate(3)), e.Tanh(c));
        }
        else h = e.LstmCellFromGates(gates, old, out c);
        Tensor<T> Square(Tensor<T> x) => e.ReduceSum(e.TensorMultiply(x, x), null);
        return mask == 1 ? Square(h) : mask == 2 ? Square(c) : e.TensorAdd(Square(h), Square(c));
    }
    private static void Near<T>(Tensor<T> a, Tensor<T> b, double tol)
    {
        Assert.Equal(a.Shape, b.Shape);
        for (int i = 0; i < a.Length; i++) Assert.InRange(Math.Abs(Convert.ToDouble(a[i]) - Convert.ToDouble(b[i])), 0, tol);
    }
    private static void Check<T>(CpuEngine engine, bool spatial, int mask, double tol)
    {
        var shape = spatial ? new[] { 2, 3, 4, 3 } : new[] { 2, 3 };
        var gatesShape = (int[])shape.Clone(); gatesShape[gatesShape.Length - 1] *= 4;
        var gates = Values<T>(gatesShape); var old = Values<T>(shape); var sources = new[] { gates, old };
        Tensor<T> actual;
        Dictionary<Tensor<T>, Tensor<T>> gradients;
        using (var tape = new GradientTape<T>())
        {
            actual = Loss(engine, gates, old, mask, false);
            Assert.True(tape.EntryCount < 13);
            gradients = tape.ComputeGradients(actual, sources);
        }
        using var referenceTape = new GradientTape<T>();
        var expected = Loss(new CpuEngine(), gates, old, mask, true);
        var expectedGrads = referenceTape.ComputeGradients(expected, sources);
        Near(expected, actual, tol);
        foreach (var source in sources) Near(expectedGrads[source], gradients[source], tol);
    }
    [Theory]
    [InlineData(true, true, 1)] [InlineData(true, true, 2)] [InlineData(true, true, 3)]
    [InlineData(false, true, 1)] [InlineData(false, true, 2)] [InlineData(false, true, 3)]
    [InlineData(true, false, 3)] [InlineData(false, false, 3)]
    public void IndependentAdjoints_MatchDecomposedCell(bool single, bool spatial, int mask)
    {
        if (single) Check<float>(new CpuEngine(), spatial, mask, 3e-5);
        else Check<double>(new CpuEngine(), spatial, mask, 1e-11);
    }
    private static void Replay(CpuEngine engine)
    {
        var gates = Values<float>(new[] { 2, 12 }); var old = Values<float>(new[] { 2, 3 });
        var sources = new[] { gates, old };
        ICompiledTrainingPlan<float> plan;
        using (var scope = GraphMode.Enable()) { Loss(engine, gates, old, 3, false); plan = scope.CompileTraining(sources); }
        using (plan)
        {
            Assert.True(Assert.IsType<CompiledTrainingPlan<float>>(plan).ForwardStepCount < 15);
            plan.ConfigureOptimizer(OptimizerType.SGD, learningRate: 0);
            for (int i = 0; i < 3; i++)
            {
                gates[0] = i * 0.3f; old[0] = -i * 0.1f;
                Tensor<float> expected; Dictionary<Tensor<float>, Tensor<float>> grads;
                using (var tape = new GradientTape<float>())
                { expected = Loss(new CpuEngine(), gates, old, 3, true); grads = tape.ComputeGradients(expected, sources); }
                Near(expected, plan.Step(), 2e-5);
                foreach (var source in sources) Near(grads[source], source.Grad!, 2e-5);
            }
        }
    }
    [Fact] public void CompiledGateKernel_Replays() => Replay(new CpuEngine());
    [Fact, Trait("Category", "RequiresOpenCL")]
    public void OpenCl_GateKernelAndReplay_MatchCpu()
    {
        using var backend = new OpenClBackend(); Assert.True(backend.IsAvailable, "OpenCL unavailable.");
        using var direct = new DirectGpuEngine(backend); using var engine = new DirectGpuTensorEngine(direct);
        foreach (int mask in new[] { 1, 2, 3 }) Check<float>(engine, true, mask, 3e-5);
        Replay(engine);
    }
    [Fact]
    public void GateKernel_RejectsIncompatibleShapes()
    {
        var e = new CpuEngine();
        Assert.Throws<ArgumentException>(() => e.LstmCellFromGates(new Tensor<double>(new[] { 2, 7 }), new Tensor<double>(new[] { 2, 2 }), out _));
        Assert.Throws<ArgumentException>(() => e.LstmCellFromGates(new Tensor<double>(new[] { 3, 8 }), new Tensor<double>(new[] { 2, 2 }), out _));
    }
}
