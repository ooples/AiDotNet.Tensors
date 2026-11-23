using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

/// <summary>
/// A genuine broadcast (a size-1 axis expanded) is a zero-add. Its zero operand was a tensor of the whole target shape,
/// which a compiled training plan holds as a constant input and refreshes from the host on every graph replay.
/// </summary>
public class BroadcastToCompiledInputTests
{
    [Fact]
    public void Expanding_a_size_one_axis_records_no_target_sized_constant()
    {
        var engine = new CpuEngine();
        var p = new Tensor<float>(new float[] { 1f, -2f, 3f, 0.5f, 4f, -1f }, new[] { 2, 3, 1 });
        var target = new[] { 2, 3, 64 };
        CompiledTrainingPlan<float> plan;
        using (var scope = GraphMode.EnableTraining(new[] { p }))
        {
            var expanded = engine.TensorBroadcastTo(p, target);
            var loss = engine.ReduceSum(engine.TensorMultiply(expanded, expanded), null, false);
            plan = Assert.IsType<CompiledTrainingPlan<float>>(scope.CompileTraining(new[] { p }, loss));
        }
        using (plan)
        {
            // The zero operand needs the target extent only on the expanded axis: 64 elements, not 2 x 3 x 64.
            Assert.InRange(plan.GraphExternalLeafElementsForTest, 0, 64);
            plan.ConfigureOptimizer(OptimizerType.SGD, learningRate: 0.0);
            float lossValue = plan.Step().ToArray()[0];
            float expected = 0f;
            foreach (float v in new[] { 1f, -2f, 3f, 0.5f, 4f, -1f }) expected += 64f * v * v;
            Assert.Equal(expected, lossValue, 3);
            var grad = plan.Gradients[0].ToArray();
            var values = new[] { 1f, -2f, 3f, 0.5f, 4f, -1f };
            for (int i = 0; i < values.Length; i++) Assert.Equal(128f * values[i], grad[i], 3);   // d/dp sum_64 p^2
        }
    }

    [Fact]
    public void Broadcast_to_a_higher_rank_expands_leading_and_size_one_axes()
    {
        var engine = new CpuEngine();
        var x = new Tensor<float>(new float[] { 1f, 2f, 3f }, new[] { 3, 1 });
        var y = engine.TensorBroadcastTo(x, new[] { 2, 3, 4 });
        Assert.Equal(new[] { 2, 3, 4 }, y.Shape.ToArray());
        var data = y.ToArray();
        for (int b = 0; b < 2; b++)
            for (int r = 0; r < 3; r++)
                for (int c = 0; c < 4; c++)
                    Assert.Equal(r + 1f, data[(b * 3 + r) * 4 + c]);
    }
}
