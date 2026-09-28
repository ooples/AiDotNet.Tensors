using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// The compiled training plan's specialized TensorAdd / TensorSubtract backward must give each input a gradient of
/// that input's own shape. It copied (or accumulated) the output gradient straight into each input's buffer, which
/// is right only when both inputs have the output's shape. An operand the engine broadcast, or one with the same
/// element count at a different rank, got the output gradient unreduced: "All tensor shapes must match" when the
/// operand was multi-consumer, a span-length failure otherwise. In AiDotNet the plan swallows that and trains the
/// model on the eager tape for the rest of the run (DeepFactor, found by the model-performance census).
/// </summary>
public class CompiledAddSubtractBackwardShapeTests
{
    public static TheoryData<int[], int[], bool, bool> Shapes() => new()
    {
        // a shape, b shape, subtract, b consumed twice (accumulating gradient)
        { new[] { 4, 5 }, new[] { 4, 5 }, false, false },
        { new[] { 4, 5 }, new[] { 5 }, false, false },
        { new[] { 4, 5 }, new[] { 5 }, false, true },
        { new[] { 4, 5 }, new[] { 1, 5 }, false, true },
        { new[] { 3, 4, 5 }, new[] { 4, 5 }, false, false },
        { new[] { 4, 5 }, new[] { 5 }, true, false },
        { new[] { 4, 5 }, new[] { 5 }, true, true },
        { new[] { 5 }, new[] { 1, 5 }, false, false },
    };

    [Theory]
    [MemberData(nameof(Shapes))]
    public void CompiledGradients_MatchTheTape_AndHaveEachInputsShape(int[] aShape, int[] bShape, bool subtract, bool bTwice)
    {
        var prior = AiDotNetEngine.Current;
        try
        {
            var engine = new CpuEngine();
            AiDotNetEngine.Current = engine;
            var a = Filled(aShape, 1);
            var b = Filled(bShape, 2);

            Tensor<float> Forward()
            {
                var y = subtract ? engine.TensorSubtract(a, b) : engine.TensorAdd(a, b);
                // Square so the gradient depends on the value, not just on the shape.
                var loss = engine.ReduceSum(engine.TensorMultiply(y, y), null);
                if (!bTwice) return loss;
                return engine.TensorAdd(loss, engine.ReduceSum(engine.TensorMultiply(b, b), null));
            }

            float[] eagerA, eagerB;
            float eagerLoss;
            using (var tape = new GradientTape<float>())
            {
                var loss = Forward();
                eagerLoss = loss.GetFlattenedData()[0];
                var grads = tape.ComputeGradients(loss, new[] { a, b });
                eagerA = (float[])grads[a].GetFlattenedData().Clone();
                eagerB = (float[])grads[b].GetFlattenedData().Clone();
            }

            ICompiledTrainingPlan<float> plan;
            using (var scope = GraphMode.Enable())
            {
                Forward();
                plan = scope.CompileTraining(new[] { a, b });
            }

            try
            {
                var loss = plan.Step();
                Assert.Equal(eagerLoss, loss[0], 3);
                Assert.Equal(aShape, plan.Gradients[0].Shape.ToArray());
                Assert.Equal(bShape, plan.Gradients[1].Shape.ToArray());
                AssertClose(eagerA, plan.Gradients[0].GetFlattenedData(), "a");
                AssertClose(eagerB, plan.Gradients[1].GetFlattenedData(), "b");

                // A second step must not stack on the first step's gradient.
                plan.Step();
                AssertClose(eagerB, plan.Gradients[1].GetFlattenedData(), "b (second step)");
            }
            finally
            {
                plan.Dispose();
            }
        }
        finally
        {
            AiDotNetEngine.Current = prior;
        }
    }

    /// <summary>
    /// The broadcast-add specialization with a multi-consumer left operand whose sum is then reshaped: the
    /// output's gradient buffer can come from the reshape, same elements under a different shape.
    /// </summary>
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void BroadcastAdd_ReshapedOutput_MultiConsumerOperand_MatchesTheTape(bool viaBroadcastAdd)
    {
        var prior = AiDotNetEngine.Current;
        try
        {
            var engine = new CpuEngine();
            AiDotNetEngine.Current = engine;
            var a = Filled(new[] { 2, 3, 4 }, 3);
            var b = Filled(new[] { 4 }, 4);

            Tensor<float> Forward()
            {
                var y = viaBroadcastAdd ? engine.TensorBroadcastAdd(a, b) : engine.TensorAdd(a, engine.TensorBroadcastAdd(engine.TensorMultiplyScalar(a, 0f), b));
                var flat = engine.Reshape(y, new[] { 6, 4 });
                var main = engine.ReduceSum(engine.TensorMultiply(flat, flat), null);
                return engine.TensorAdd(main, engine.ReduceSum(engine.TensorMultiply(a, a), null));
            }

            float[] eagerA, eagerB;
            using (var tape = new GradientTape<float>())
            {
                var loss = Forward();
                var grads = tape.ComputeGradients(loss, new[] { a, b });
                eagerA = (float[])grads[a].GetFlattenedData().Clone();
                eagerB = (float[])grads[b].GetFlattenedData().Clone();
            }

            ICompiledTrainingPlan<float> plan;
            using (var scope = GraphMode.Enable())
            {
                Forward();
                plan = scope.CompileTraining(new[] { a, b });
            }

            try
            {
                plan.Step();
                Assert.Equal(a.Shape.ToArray(), plan.Gradients[0].Shape.ToArray());
                Assert.Equal(b.Shape.ToArray(), plan.Gradients[1].Shape.ToArray());
                AssertClose(eagerA, plan.Gradients[0].GetFlattenedData(), "a");
                AssertClose(eagerB, plan.Gradients[1].GetFlattenedData(), "b");
            }
            finally
            {
                plan.Dispose();
            }
        }
        finally
        {
            AiDotNetEngine.Current = prior;
        }
    }
    /// <summary>
    /// DeepFactor's failing shape pair: a [64] operand whose sum is lifted to [1, 64]. The census run reported
    /// "a [64], b [1, 64], destination [64]" from the broadcast-add backward's accumulation.
    /// </summary>
    [Theory]
    [InlineData(new[] { 64 }, new[] { 1, 64 }, false)]
    [InlineData(new[] { 64 }, new[] { 1, 64 }, true)]
    [InlineData(new[] { 1, 64 }, new[] { 64 }, false)]
    [InlineData(new[] { 1, 64 }, new[] { 64 }, true)]
    public void BroadcastAdd_RankLiftedSum_MultiConsumerOperand_MatchesTheTape(int[] vShape, int[] liftTo, bool vFirst)
    {
        var prior = AiDotNetEngine.Current;
        try
        {
            var engine = new CpuEngine();
            AiDotNetEngine.Current = engine;
            var v = Filled(vShape, 5);
            var c = Filled(new[] { 1 }, 6);

            Tensor<float> Forward()
            {
                var y = vFirst ? engine.TensorBroadcastAdd(v, c) : engine.TensorBroadcastAdd(engine.TensorMultiplyScalar(v, 1f), c);
                var lifted = engine.Reshape(y, liftTo);
                var main = engine.ReduceSum(engine.TensorMultiply(lifted, lifted), null);
                return engine.TensorAdd(main, engine.ReduceSum(engine.TensorMultiply(v, v), null));
            }

            float[] eagerV, eagerC;
            using (var tape = new GradientTape<float>())
            {
                var loss = Forward();
                var grads = tape.ComputeGradients(loss, new[] { v, c });
                eagerV = (float[])grads[v].GetFlattenedData().Clone();
                eagerC = (float[])grads[c].GetFlattenedData().Clone();
            }

            ICompiledTrainingPlan<float> plan;
            using (var scope = GraphMode.Enable())
            {
                Forward();
                plan = scope.CompileTraining(new[] { v, c });
            }

            try
            {
                plan.Step();
                Assert.Equal(vShape, plan.Gradients[0].Shape.ToArray());
                Assert.Equal(new[] { 1 }, plan.Gradients[1].Shape.ToArray());
                AssertClose(eagerV, plan.Gradients[0].GetFlattenedData(), "v");
                AssertClose(eagerC, plan.Gradients[1].GetFlattenedData(), "c");
            }
            finally
            {
                plan.Dispose();
            }
        }
        finally
        {
            AiDotNetEngine.Current = prior;
        }
    }
    /// <summary>
    /// The case DeepFactor hit: a lower-rank right operand that needs no reduction ([64] against [1, 64]) and is
    /// consumed twice, so its gradient accumulates. Padding it to the output's rank gives no reduce axes, and the
    /// accumulation then added the [1, 64] output gradient into the [64] buffer, which the shape check refuses.
    /// </summary>
    [Theory]
    [InlineData(new[] { 1, 64 }, new[] { 64 })]
    [InlineData(new[] { 1, 1, 8 }, new[] { 8 })]
    [InlineData(new[] { 1, 3, 4 }, new[] { 3, 4 })]
    public void BroadcastAdd_LowerRankOperandWithoutReduction_Accumulating_MatchesTheTape(int[] aShape, int[] bShape)
    {
        var prior = AiDotNetEngine.Current;
        try
        {
            var engine = new CpuEngine();
            AiDotNetEngine.Current = engine;
            var a = Filled(aShape, 7);
            var b = Filled(bShape, 8);

            Tensor<float> Forward()
            {
                var y = engine.TensorBroadcastAdd(a, b);
                var main = engine.ReduceSum(engine.TensorMultiply(y, y), null);
                return engine.TensorAdd(main, engine.ReduceSum(engine.TensorMultiply(b, b), null));
            }

            float[] eagerA, eagerB;
            using (var tape = new GradientTape<float>())
            {
                var loss = Forward();
                var grads = tape.ComputeGradients(loss, new[] { a, b });
                eagerA = (float[])grads[a].GetFlattenedData().Clone();
                eagerB = (float[])grads[b].GetFlattenedData().Clone();
            }

            ICompiledTrainingPlan<float> plan;
            using (var scope = GraphMode.Enable())
            {
                Forward();
                plan = scope.CompileTraining(new[] { a, b });
            }

            try
            {
                plan.Step();
                Assert.Equal(aShape, plan.Gradients[0].Shape.ToArray());
                Assert.Equal(bShape, plan.Gradients[1].Shape.ToArray());
                AssertClose(eagerA, plan.Gradients[0].GetFlattenedData(), "a");
                AssertClose(eagerB, plan.Gradients[1].GetFlattenedData(), "b");
                plan.Step();
                AssertClose(eagerB, plan.Gradients[1].GetFlattenedData(), "b (second step)");
            }
            finally
            {
                plan.Dispose();
            }
        }
        finally
        {
            AiDotNetEngine.Current = prior;
        }
    }
    /// <summary>
    /// The shape-mismatch refusal names all three shapes: "All tensor shapes must match." alone is what hid the
    /// [64] against [1, 64] case above for a whole census cycle.
    /// </summary>
    [Fact]
    public void ElementwiseInto_ShapeMismatch_NamesEveryShape()
    {
        var engine = new CpuEngine();
        var destination = new Tensor<float>(new float[64], new[] { 64 });
        var a = new Tensor<float>(new float[64], new[] { 64 });
        var b = new Tensor<float>(new float[64], new[] { 1, 64 });

        var add = Assert.Throws<ArgumentException>(() => engine.TensorAddInto(destination, a, b));
        Assert.Contains("a [64], b [1, 64], destination [64]", add.Message);

        var multiply = Assert.Throws<ArgumentException>(() => engine.TensorMultiplyInto(destination, a, b));
        Assert.Contains("a [64], b [1, 64], destination [64]", multiply.Message);
    }
    private static Tensor<float> Filled(int[] shape, int seed)
    {
        int n = 1;
        foreach (int d in shape) n *= d;
        var data = new float[n];
        var rng = new Random(seed);
        for (int i = 0; i < n; i++) data[i] = (float)(rng.NextDouble() * 2 - 1);
        return new Tensor<float>(data, shape);
    }

    private static void AssertClose(float[] expected, float[] actual, string name)
    {
        Assert.True(expected.Length == actual.Length, $"{name}: gradient has {actual.Length} elements, expected {expected.Length}.");
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - actual[i]) <= 1e-4f * Math.Max(1f, Math.Abs(expected[i])),
                $"{name}[{i}]: compiled {actual[i]}, tape {expected[i]}.");
    }
}
